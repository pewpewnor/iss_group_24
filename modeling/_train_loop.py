"""Inner training loop: one full sweep over a dataloader.

Returns the running mean of every loss term + grad norm + step count.
Supports gradient accumulation: the optimiser steps every
``cfg["grad_accum_steps"]`` mini-batches.
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader

from modeling.loss import total_loss
from modeling.model import OWLv2FewShotLocalizer


_RUNNING_KEYS = (
    "loss", "focal", "l1", "giou", "box_loss",
    "box_area_penalty", "existence_kl",
    "margin", "nt_xent",
    "grad_norm",
)


def train_one_pass(
    *,
    model: OWLv2FewShotLocalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: SequentialLR,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    use_box_loss: bool,
    scaler: "torch.amp.GradScaler | None" = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """One sweep through ``loader``.  Returns averaged train metrics."""
    model.train()
    # Always set OWLv2 vision_model to eval for batchnorm-like behaviour
    # (it doesn't have BN but this signals it's not training even if a
    # subset of params is frozen).
    if not any(p.requires_grad for p in model.owlv2.owlv2.vision_model.parameters()):
        model.owlv2.owlv2.vision_model.eval()

    running = {k: 0.0 for k in _RUNNING_KEYS}
    n_batches = 0
    accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    amp_enabled = use_amp and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    for batch_idx, batch in enumerate(loader):
        support_imgs = batch["support_imgs"].to(device, non_blocking=True)
        query_img = batch["query_img"].to(device, non_blocking=True)
        gt_bbox = batch["query_bbox"].to(device, non_blocking=True)
        is_present = batch["is_present"].to(device, non_blocking=True)
        instance_id = batch.get("instance_id")

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            out = model(support_imgs, query_img)
            losses = total_loss(
                out, gt_bbox, is_present,
                use_box_loss=use_box_loss,
                focal_alpha=float(cfg["focal_alpha"]),
                focal_gamma=float(cfg["focal_gamma"]),
                lambda_l1=float(cfg["lambda_l1"]),
                lambda_giou=float(cfg["lambda_giou"]),
                anti_collapse_weight=float(cfg["anti_collapse_weight"]),
                box_size_threshold=float(cfg["box_size_threshold"]),
                existence_kl_threshold=float(cfg["existence_kl_threshold"]),
                margin_weight=float(cfg.get("margin_weight", 0.5)),
                margin_value=float(cfg.get("margin_value", 1.0)),
                contrastive_weight=float(cfg.get("contrastive_weight", 0.1)),
                contrastive_temp=float(cfg.get("contrastive_temp", 0.1)),
                instance_id=instance_id,
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

        running["loss"] += float(losses["loss"].detach().item())
        for k in ("focal", "l1", "giou", "box_loss",
                  "box_area_penalty", "existence_kl",
                  "margin", "nt_xent"):
            v = losses.get(k)
            if v is not None:
                running[k] += float(v.detach().item()) if torch.is_tensor(v) else float(v)
        n_batches += 1

    if n_batches > 0:
        for k in running:
            running[k] /= max(n_batches, 1)
    running["n_steps"] = n_batches
    # Snapshot the residual aggregator gate at end-of-pass.  This is a
    # global scalar — not a per-batch quantity — so we don't average it.
    # It tells us whether the aggregator's correction is being trusted
    # by the model: alpha=0 means "use only the OWLv2 image-guided
    # baseline prototype", alpha>0 means "blend in the learned correction".
    try:
        running["aggregator_alpha"] = float(model.aggregator_alpha.detach().item())
    except AttributeError:
        running["aggregator_alpha"] = 0.0
    return running
