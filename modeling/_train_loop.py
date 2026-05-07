"""Inner training loop: one full sweep over a dataloader.

Returns the running average of every loss term + grad norm + step count.
The combination + weighting of regularisation losses (NT-Xent / VICReg /
Barlow / triplet) lives here because it's training-loop-only.
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
from modeling.model import FewShotLocalizer


_RUNNING_KEYS = (
    "loss", "qfl", "neg_qfl", "centerness", "dfl", "giou",
    "presence", "attn", "aux", "entropy_reg", "reg_l2_prior", "proto_norm",
    "nt_xent", "vicreg", "barlow", "triplet", "grad_norm",
)


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
) -> dict[str, float]:
    """One full sweep through ``loader``. Returns averaged train metrics."""
    model.train()
    model.backbone.eval()                                                     # freeze BN running stats

    running = {k: 0.0 for k in _RUNNING_KEYS}
    n_batches = 0
    rng = _random.Random(cfg.get("seed", 42))

    for batch in loader:
        if multi_scale:
            sz = rng.choice(multi_scale_sizes)
            loader.dataset.set_img_size(sz)                                  # type: ignore[attr-defined]
            target = (sz, sz)
            if batch["query_img"].shape[-2:] != target:
                batch["query_img"] = F.interpolate(
                    batch["query_img"], size=target, mode="bilinear", align_corners=False
                )
                support = batch["support_imgs"]
                B, K = support.shape[:2]
                batch["support_imgs"] = F.interpolate(
                    support.view(-1, 3, *support.shape[-2:]),
                    size=target, mode="bilinear", align_corners=False,
                ).view(B, K, 3, sz, sz)

        support_imgs = batch["support_imgs"].to(device)
        support_bboxes = batch["support_bboxes"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)

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
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            cfg["grad_clip"],
        )
        optimizer.step()
        scheduler.step()

        running["loss"] += float(loss.item())
        for k in ("qfl", "neg_qfl", "centerness", "dfl", "giou",
                  "presence", "attn", "aux", "entropy_reg",
                  "reg_l2_prior", "proto_norm"):
            running[k] += float(losses[k])
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
