"""Per-stage optimiser + scheduler factories.

Stage 1.1: aggregator + existence_head.
Stage 1.2: + box_head + class_head + layer_norm.
Stage 2.3: + LoRA adapters on last 4 vision blocks.
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from modeling.model import OWLv2FewShotLocalizer


def build_optimizer_for_stage(
    stage: str, model: OWLv2FewShotLocalizer, cfg: dict
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter] | None]:
    """Returns (optimizer, lora_params).  ``lora_params`` is None unless
    stage == "2_3", in which case it's the list of LoRA parameter tensors
    (so the trainer can clip them separately if needed).
    """
    wd = float(cfg["weight_decay"])
    lora_params: list[torch.nn.Parameter] | None = None

    # Stage 2.3: attach LoRA first (this re-freezes everything except adapters).
    if stage == "2_3":
        lora_params = model.attach_lora(
            r=int(cfg["lora_r"]),
            alpha=int(cfg["lora_alpha"]),
            dropout=float(cfg["lora_dropout"]),
            last_n_layers=int(cfg["lora_layers"]),
        )

    # Always start from a clean freeze, then opt-in.
    model.freeze_owlv2_all()
    for p in model.aggregator.parameters():
        p.requires_grad = True
    for p in model.existence_head.parameters():
        p.requires_grad = True
    model.aggregator_alpha.requires_grad = True

    groups: list[dict] = [
        # Aggregator network + the residual gate scalar.  The gate joins this
        # group so it sees the same LR / weight decay schedule.
        {"params": list(model.aggregator.parameters()) + [model.aggregator_alpha],
         "lr": float(cfg["lr_aggregator"]),
         "weight_decay": wd, "name": "aggregator"},
        # Existence head: tiny (~5K params) + the *only* learner driving the
        # focal loss in Stage 1.1.  Standard weight decay was pulling its
        # bias toward zero → constant-output collapse.  Disable WD entirely
        # for this group; the head is small enough that overfit is not the
        # failure mode we're worried about.
        {"params": list(model.existence_head.parameters()),
         "lr": float(cfg["lr_existence"]),
         "weight_decay": 0.0, "name": "existence_head"},
    ]

    if stage in ("1_2", "2_3"):
        model.unfreeze_owlv2_heads()
        groups.append({
            "params": model.class_head_params(),
            "lr": float(cfg["lr_class"]),
            "weight_decay": wd, "name": "class_head",
        })
        groups.append({
            "params": model.box_head_params(),
            "lr": float(cfg["lr_box"]),
            "weight_decay": wd, "name": "box_head",
        })

    if stage == "2_3":
        # Re-enable LoRA params (freeze_owlv2_all turned them off).
        for p in lora_params:                                            # type: ignore[possibly-undefined]
            p.requires_grad = True
        if lora_params:
            groups.append({
                "params": lora_params,
                "lr": float(cfg["lr_lora"]),
                "weight_decay": wd, "name": "lora",
            })

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    return optimizer, lora_params


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float
) -> SequentialLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    cosine_steps = max(1, total_steps - warmup_steps)
    warm = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_steps])
