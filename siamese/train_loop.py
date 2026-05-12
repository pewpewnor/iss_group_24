"""Inner training loop for the siamese."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from siamese.loss import total_loss
from siamese.model import MultiShotSiamese


_RUNNING_KEYS = ("loss", "focal", "variance", "decorrelation", "grad_norm")


def train_one_pass(
    *,
    model: MultiShotSiamese,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    scaler,
    use_amp: bool = False,
    progress: bool = True,
    progress_every: int = 10,
    hard_neg_recorder: dict | None = None,
) -> dict[str, float]:
    """Run one training pass.

    Optional hard_neg_recorder dict (instance_id -> list[{"path": str}]) is
    populated with negatives whose existence_prob > 0.5 (the model
    incorrectly predicted "present"). The siamese trainer feeds this back
    into the dataset's hard_neg_cache for the next epoch.
    """
    model.train()
    if not any(p.requires_grad for p in model.dinov2.parameters()):
        try:
            model.dinov2.eval()
        except AttributeError:
            pass

    running = {k: 0.0 for k in _RUNNING_KEYS}
    # ``grad_norm`` is averaged over OPTIMIZER STEPS (= n_batches / accum_steps),
    # not raw batches. AMP can return nan from ``clip_grad_norm_`` when a step
    # overflows; the scaler then skips ``optimizer.step()`` automatically and
    # the displayed grad_norm should ignore those nan reads rather than poison
    # the running sum. We track step count and nan-step count separately.
    grad_steps = 0
    grad_nan_steps = 0
    n_batches = 0
    accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    amp_enabled = use_amp and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    n_batches_total = len(loader) if hasattr(loader, "__len__") else None
    t_start = time.time()
    if progress:
        print(f"  training : {n_batches_total or '?'} batches", flush=True)

    for batch_idx, batch in enumerate(loader):
        sup = batch["support_imgs"].to(device, non_blocking=True)
        sup_mask = batch["support_mask"].to(device, non_blocking=True)
        qry = batch["query_img"].to(device, non_blocking=True)
        is_present = batch["is_present"].to(device, non_blocking=True)
        instance_ids = batch["instance_id"]

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            out = model(sup, sup_mask, qry)
            losses = total_loss(
                out, is_present,
                focal_alpha=float(cfg["focal_alpha"]),
                focal_gamma=float(cfg["focal_gamma"]),
                variance_target=float(cfg["variance_target"]),
                variance_weight=float(cfg["variance_weight"]),
                decorr_weight=float(cfg["decorr_weight"]),
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
            grad_steps += 1
            gn_f = float(gn)
            if gn_f != gn_f or gn_f == float("inf") or gn_f == float("-inf"):
                # AMP overflow on this step; scaler skipped optimizer.step().
                grad_nan_steps += 1
            else:
                running["grad_norm"] += gn_f

        running["loss"] += float(losses["loss"].detach().item())
        running["focal"] += float(losses["focal"].detach().item())
        running["variance"] += float(losses["variance"].detach().item())
        running["decorrelation"] += float(losses["decorrelation"].detach().item())

        # Hard-negative recording: for each negative episode where pred>0.5,
        # log the (anchor_instance_id, negative_query_path) pair. The dataset's
        # negative sampler will draw from this cache at rate ``hard_neg_frac``.
        # The literal negative path is surfaced by EpisodeDataset as
        # ``batch["query_path"]`` (empty string for positives).
        if hard_neg_recorder is not None:
            query_paths = batch.get("query_path")
            if query_paths is not None:
                with torch.no_grad():
                    preds = out["existence_prob"].detach().cpu()
                    for i in range(preds.size(0)):
                        if (not bool(is_present[i].item())) and float(preds[i]) > 0.5:
                            iid = instance_ids[i]
                            qpath = query_paths[i] if i < len(query_paths) else ""
                            if qpath:
                                hard_neg_recorder.setdefault(iid, []).append(
                                    {"path": qpath}
                                )

        n_batches += 1
        if progress and (n_batches % progress_every == 0
                         or n_batches == n_batches_total):
            elapsed = time.time() - t_start
            rate = n_batches / max(elapsed, 1e-6)
            avg_loss = running["loss"] / max(n_batches, 1)
            if n_batches_total:
                pct = 100.0 * n_batches / n_batches_total
                eta = (n_batches_total - n_batches) / max(rate, 1e-6)
                print(f"  [{n_batches}/{n_batches_total}={pct:5.1f}%]  "
                      f"elapsed={elapsed:5.1f}s  eta={eta:5.1f}s  "
                      f"rate={rate:.2f}b/s  loss={avg_loss:.4f}", flush=True)
            else:
                print(f"  [{n_batches}]  elapsed={elapsed:5.1f}s  "
                      f"rate={rate:.2f}b/s  loss={avg_loss:.4f}", flush=True)

    if n_batches > 0:
        # All keys except ``grad_norm`` are per-batch averages.
        for k in running:
            if k == "grad_norm":
                continue
            running[k] /= max(n_batches, 1)
        # ``grad_norm`` is a per-optimizer-step average over finite reads only.
        finite_steps = grad_steps - grad_nan_steps
        running["grad_norm"] = (
            running["grad_norm"] / finite_steps if finite_steps > 0 else 0.0
        )
    running["n_steps"] = n_batches
    running["grad_steps"] = grad_steps
    running["grad_nan_steps"] = grad_nan_steps
    return running
