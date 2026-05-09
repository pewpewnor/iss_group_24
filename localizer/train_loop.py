"""Inner training loop for the localizer (one full sweep)."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from localizer.loss import total_loss
from localizer.model import MultiShotLocalizer


_RUNNING_KEYS = ("loss", "patch_ce", "l1", "giou", "grad_norm")


def train_one_pass(
    *,
    model: MultiShotLocalizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    scaler,
    use_amp: bool = False,
    use_box_loss: bool = True,
    progress: bool = True,
    progress_every: int = 10,
) -> dict[str, float]:
    """Run one training pass.

    use_box_loss : suppress L1+GIoU. At L1 the box_head is frozen so those
                   terms produce zero gradient and are pure overhead; at
                   L2/L3 they're informative.
    """
    model.train()
    if not any(p.requires_grad for p in model.owlv2.parameters()):
        # Vision model frozen: keep eval-mode for stability.
        try:
            model.owlv2.eval()
        except AttributeError:
            pass

    running = {k: 0.0 for k in _RUNNING_KEYS}
    n_batches = 0
    accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    amp_enabled = use_amp and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    n_batches_total = len(loader) if hasattr(loader, "__len__") else None
    t_start = time.time()
    if progress:
        suffix = "(box_loss off)" if not use_box_loss else "(box_loss on)"
        print(f"  training : {n_batches_total or '?'} batches  {suffix}", flush=True)

    for batch_idx, batch in enumerate(loader):
        sup = batch["support_imgs"].to(device, non_blocking=True)
        sup_mask = batch["support_mask"].to(device, non_blocking=True)
        qry = batch["query_img"].to(device, non_blocking=True)
        gt_bbox = batch["query_bbox"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            out = model(sup, sup_mask, qry)
            losses = total_loss(
                out, gt_bbox,
                lambda_patch_ce=float(cfg.get("lambda_patch_ce", 1.0)),
                lambda_l1=float(cfg.get("lambda_l1", 5.0)),
                lambda_giou=float(cfg.get("lambda_giou", 2.0)),
                use_box_loss=use_box_loss,
                label_smoothing=float(cfg.get("patch_ce_label_smoothing", 0.0)),
            )
            loss = losses["loss"] / accum_steps

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_count += 1
        if accum_count >= accum_steps:
            if amp_enabled and scaler is not None:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    grad_clip,
                )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_count = 0
            running["grad_norm"] += float(gn)

        running["loss"]     += float(losses["loss"].detach().item())
        running["patch_ce"] += float(losses["patch_ce"].detach().item())
        running["l1"]       += float(losses["l1"].detach().item())
        running["giou"]     += float(losses["giou"].detach().item())
        n_batches += 1

        if progress and (n_batches % progress_every == 0
                         or n_batches == n_batches_total):
            elapsed = time.time() - t_start
            rate = n_batches / max(elapsed, 1e-6)
            avg_loss = running["loss"] / max(n_batches, 1)
            avg_pce = running["patch_ce"] / max(n_batches, 1)
            if n_batches_total:
                pct = 100.0 * n_batches / n_batches_total
                eta = (n_batches_total - n_batches) / max(rate, 1e-6)
                print(f"  [{n_batches}/{n_batches_total}={pct:5.1f}%]  "
                      f"elapsed={elapsed:5.1f}s  eta={eta:5.1f}s  "
                      f"rate={rate:.2f}b/s  loss={avg_loss:.4f}  "
                      f"patch_ce={avg_pce:.4f}", flush=True)
            else:
                print(f"  [{n_batches}]  elapsed={elapsed:5.1f}s  "
                      f"rate={rate:.2f}b/s  loss={avg_loss:.4f}  "
                      f"patch_ce={avg_pce:.4f}", flush=True)

    if n_batches > 0:
        for k in running:
            running[k] /= max(n_batches, 1)
    running["n_steps"] = n_batches
    # Snapshot the residual gate scalar at end-of-pass. alpha=0 ⇒ prototype
    # is the zero-shot baseline; alpha>0 ⇒ fusion is contributing.
    try:
        running["alpha"] = float(model.alpha.detach().item())
    except AttributeError:
        running["alpha"] = 0.0
    return running
