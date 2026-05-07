"""Per-stage optimiser + LR scheduler factories.

Stage 1: heads only, backbone frozen.
Stage 2: heads + backbone.features[7:].
Stage 3: full unfreeze.

The scheduler is one continuous ``LinearLR(warmup) → CosineAnnealingLR(rest)``
across the entire stage's total step count.
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from modeling.model import FewShotLocalizer


# ---------------------------------------------------------------------------
# Param group helpers
# ---------------------------------------------------------------------------


def heads_params(model: FewShotLocalizer) -> list:
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    return [p for p in model.parameters() if id(p) not in backbone_ids]


def backbone_upper_params(
    model: FewShotLocalizer, upper_idx: int = 6
) -> list:
    """``features[upper_idx:]`` parameters. Stage 2 default: idx=6 so two
    extra MobileNet blocks join gradient (was idx=7)."""
    return [
        p
        for i, blk in enumerate(model.backbone.features)
        if i >= upper_idx
        for p in blk.parameters()
    ]


def backbone_lower_params(
    model: FewShotLocalizer, upper_idx: int = 6
) -> list:
    return [
        p
        for i, blk in enumerate(model.backbone.features)
        if i < upper_idx
        for p in blk.parameters()
    ]


# ---------------------------------------------------------------------------
# Optimiser factories
# ---------------------------------------------------------------------------


def build_optimizer_for_stage(
    stage: int, model: FewShotLocalizer, cfg: dict
) -> torch.optim.Optimizer:
    wd = cfg["weight_decay"]
    # Stage 2 unfreeze depth (defaults to 6 so two MobileNet blocks join).
    s2_idx = int(cfg.get("stage2_unfreeze_idx", 6))
    s3_idx = int(cfg.get("stage3_split_idx", 6))
    if stage == 1:
        model.backbone.freeze_all()
        groups = [
            {"params": heads_params(model), "lr": cfg["lr_heads_s1"], "weight_decay": wd}
        ]
    elif stage == 2:
        model.backbone.freeze_lower(freeze_idx_exclusive=s2_idx)
        groups = [
            {"params": backbone_upper_params(model, upper_idx=s2_idx),
             "lr": cfg["lr_backbone_upper_s2"], "weight_decay": wd},
            {"params": heads_params(model), "lr": cfg["lr_heads_s2"], "weight_decay": wd},
        ]
    elif stage == 3:
        model.backbone.unfreeze_all()
        groups = [
            {"params": backbone_lower_params(model, upper_idx=s3_idx),
             "lr": cfg["lr_backbone_lower_s3"], "weight_decay": wd},
            {"params": backbone_upper_params(model, upper_idx=s3_idx),
             "lr": cfg["lr_backbone_upper_s3"], "weight_decay": wd},
            {"params": heads_params(model), "lr": cfg["lr_heads_s3"], "weight_decay": wd},
        ]
    else:
        raise ValueError(f"unknown stage {stage}")
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float
) -> SequentialLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    cosine_steps = max(1, total_steps - warmup_steps)
    warm = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_steps])
