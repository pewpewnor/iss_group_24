"""Inner training loop: one full sweep over a dataloader.

Returns the running average of every loss term + grad norm + step count.
The combination + weighting of regularisation losses (NT-Xent / VICReg /
Barlow / triplet) lives here because it's training-loop-only.

Phase 0 last-resort additions:
- Mixed precision (autocast + GradScaler) for ~2× T4 throughput.
- EMA model update on every optimiser step.
- Source-loss-weight annealing schedule (interpolates between an early
  ``source_loss_weights_warmup`` and a final ``source_loss_weights`` over
  ``source_weight_anneal_epochs`` epochs of the current stage).
"""

from __future__ import annotations

import random as _random

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader

from modeling.loss import (
    barlow_twins_loss,
    nt_xent_loss,
    total_loss,
    triplet_loss,
    vicreg_loss,
)
from modeling.model import FewShotLocalizer, ModelEMA


_RUNNING_KEYS = (
    "loss", "qfl", "neg_qfl", "centerness", "dfl", "giou", "pred_iou",
    "presence", "attn", "aux", "entropy_reg", "reg_l2_prior", "proto_norm",
    "contrastive_presence", "feature_spread",
    "nt_xent", "vicreg", "barlow", "triplet", "grad_norm",
)


def _interp_weights(
    warmup: dict[str, float] | None,
    final: dict[str, float],
    progress: float,
) -> dict[str, float]:
    """Linearly interpolate per-source weights from ``warmup`` to ``final`` by
    ``progress`` ∈ [0, 1]. If ``warmup`` is None, return ``final`` directly."""
    if not warmup:
        return final
    out: dict[str, float] = {}
    keys = set(warmup) | set(final)
    for k in keys:
        a = float(warmup.get(k, final.get(k, 1.0)))
        b = float(final.get(k, warmup.get(k, 1.0)))
        out[k] = a + (b - a) * max(0.0, min(1.0, progress))
    return out


def train_one_pass(
    *,
    model: FewShotLocalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: SequentialLR,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    multi_scale: bool,
    multi_scale_sizes: tuple[int, ...] = (192, 224, 256),
    ema: ModelEMA | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    epoch_progress: float = 1.0,
) -> dict[str, float]:
    """One full sweep through ``loader``. Returns averaged train metrics.

    Args:
        epoch_progress : 0..1 fraction of the source-weight annealing schedule
            elapsed at this epoch (computed by the caller). Used to interpolate
            ``source_loss_weights_warmup`` → ``source_loss_weights``.
    """
    model.train()
    model.backbone.eval()                                                     # freeze BN running stats

    running = {k: 0.0 for k in _RUNNING_KEYS}
    n_batches = 0
    rng = _random.Random(cfg.get("seed", 42))

    # Resolve effective per-source loss weights for this epoch.
    final_slw = cfg.get("source_loss_weights") or {}
    warmup_slw = cfg.get("source_loss_weights_warmup")
    effective_slw = _interp_weights(warmup_slw, final_slw, epoch_progress)

    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        if multi_scale:
            sz = rng.choice(multi_scale_sizes)
            loader.dataset.set_img_size(sz)                                  # type: ignore[attr-defined]
            target = (sz, sz)
            current_h = batch["query_img"].shape[-1]
            if current_h != sz:
                batch["query_img"] = F.interpolate(
                    batch["query_img"], size=target, mode="bilinear", align_corners=False
                )
                support = batch["support_imgs"]
                B, K = support.shape[:2]
                batch["support_imgs"] = F.interpolate(
                    support.view(-1, 3, *support.shape[-2:]),
                    size=target, mode="bilinear", align_corners=False,
                ).view(B, K, 3, sz, sz)
                scale = float(sz) / float(current_h)
                batch["query_bbox"] = batch["query_bbox"] * scale
                batch["support_bboxes"] = batch["support_bboxes"] * scale

        support_imgs = batch["support_imgs"].to(device)
        support_bboxes = batch["support_bboxes"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)

        if effective_slw:
            sample_w = torch.tensor(
                [float(effective_slw.get(s, 1.0)) for s in batch.get("source", [])],
                dtype=torch.float32,
                device=device,
            )
        else:
            sample_w = None

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(support_imgs, query_img, support_bboxes=support_bboxes)
            losses = total_loss(
                out, gt_bbox, is_present,
                support_bboxes=support_bboxes,
                img_size=query_img.shape[-1],
                presence_weight=cfg["presence_weight"],
                attn_loss_weight=cfg["attn_loss_weight"],
                aux_weight=cfg["aux_weight"],
                entropy_reg_weight=cfg["entropy_reg_weight"],
                reg_l2_prior_weight=cfg["reg_l2_prior_weight"],
                proto_norm_weight=cfg["proto_norm_weight"],
                contrastive_presence_weight=cfg.get("contrastive_presence_weight", 0.5),
                feature_spread_weight=cfg.get("feature_spread_weight", 0.1),
                use_hardness_weighted_presence=cfg.get(
                    "use_hardness_weighted_presence", True
                ),
                sample_loss_weight=sample_w,
            )
            loss = losses["loss"]

            nt_v = vr_v = bt_v = trip_v = 0.0
            per_shot = out.get("per_shot_prototype")
            con_target = per_shot if per_shot is not None else out["prototype"]
            if cfg["contrastive"] and con_target is not None:
                nt = nt_xent_loss(con_target, temperature=cfg["contrastive_temp"])
                loss = loss + cfg["contrastive_weight"] * nt
                nt_v = float(nt.detach().item())
            if cfg["vicreg"] and con_target is not None:
                vr = vicreg_loss(con_target)
                loss = loss + cfg["vicreg_weight"] * vr
                vr_v = float(vr.detach().item())
            if cfg["barlow"] and per_shot is not None:
                bt = barlow_twins_loss(per_shot)
                loss = loss + cfg["barlow_weight"] * bt
                bt_v = float(bt.detach().item())
            if cfg["triplet"]:
                trip = triplet_loss(out["prototype"], list(batch["instance_id"]))
                loss = loss + cfg["triplet_weight"] * trip
                trip_v = float(trip.detach().item())

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg["grad_clip"],
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg["grad_clip"],
            )
            optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        running["loss"] += float(loss.item())
        for k in (
            "qfl", "neg_qfl", "centerness", "dfl", "giou", "pred_iou",
            "presence", "attn", "aux", "entropy_reg",
            "reg_l2_prior", "proto_norm",
            "contrastive_presence", "feature_spread",
        ):
            v = losses.get(k)
            running[k] += float(v) if v is not None else 0.0
        running["nt_xent"] += nt_v
        running["vicreg"] += vr_v
        running["barlow"] += bt_v
        running["triplet"] += trip_v
        running["grad_norm"] += float(gn)
        n_batches += 1

    if n_batches > 0:
        for k in running:
            running[k] /= n_batches
    running["n_steps"] = n_batches
    return running
