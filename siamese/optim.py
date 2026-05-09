"""Per-stage optimizer + scheduler factory for the siamese."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from siamese.model import MultiShotSiamese


def build_optimizer_for_stage(
    stage: str, model: MultiShotSiamese, cfg: dict,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter] | None]:
    wd = float(cfg["weight_decay"])
    lora_params: list[torch.nn.Parameter] | None = None

    if stage == "S2":
        lora_params = model.attach_lora(
            r=int(cfg["lora_r"]),
            alpha=int(cfg["lora_alpha"]),
            dropout=float(cfg["lora_dropout"]),
            last_n_layers=int(cfg["lora_last_n_layers"]),
        )

    model.freeze_backbone()
    for p in model.head_params():
        p.requires_grad = True

    groups: list[dict] = [{
        "params": model.head_params(),
        "lr": float(cfg["lr_head"]),
        "weight_decay": wd, "name": "head",
    }]

    if stage == "S2" and lora_params:
        for p in lora_params:
            p.requires_grad = True
        groups.append({
            "params": lora_params,
            "lr": float(cfg["lr_lora"]),
            "weight_decay": wd, "name": "lora",
        })

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    return optimizer, lora_params


def build_scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_frac: float,
) -> SequentialLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    cosine_steps = max(1, total_steps - warmup_steps)
    warm = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_steps])
