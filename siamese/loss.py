"""Siamese existence loss = focal_BCE (asymmetric) + variance reg + decorrelation reg.

The user's stated priorities:
  - Reduce false positives above all else.
  - Heavily penalize representation collapse.
  - Prevent center-guessing / cheating.

  → focal_alpha=0.75 makes false positives more expensive than false negatives
    (relative to standard alpha=0.25).
  → variance_reg penalises low pooled-embedding variance across the batch
    (collapse mode #1: every output is the same vector).
  → decorrelation_reg penalises high inter-dimensional correlation
    (collapse mode #2: all dims encode the same thing).

The three terms are summed with configurable weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid-focal BCE.

    alpha weights the POSITIVE class's loss. With alpha=0.75 we apply
    weight 0.75 on positives and 0.25 on negatives — this matches the
    standard focal definition.

    Asymmetry note: the user wants to penalise false positives more.
    A FALSE POSITIVE is a *negative* example predicted positive, which
    contributes to the alpha=0.75 NEGATIVE-class weight branch. We
    therefore want a HIGHER alpha-on-NEGATIVES — equivalently, a LOWER
    alpha here. Setting alpha=0.25 puts more weight on negatives' loss
    and so penalises false positives more.

    Default in the cfg: alpha=0.25 (penalises FP more, matching user
    priority). Caller can flip to 0.75 if they want to emphasise recall.
    """
    p = torch.sigmoid(logits)
    eps = 1e-6
    ce_pos = -targets * torch.log(p.clamp(min=eps))
    ce_neg = -(1 - targets) * torch.log((1 - p).clamp(min=eps))
    fl_pos = alpha * (1 - p).pow(gamma) * ce_pos
    fl_neg = (1 - alpha) * p.pow(gamma) * ce_neg
    return (fl_pos + fl_neg).mean()


def variance_reg(pooled: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    """Penalise low std across the batch dimension.

    pooled : (B, D)
    For each dimension d, compute std over B.  Penalise relu(target - std_d).
    Mean over D. Returns a scalar.

    With B=1 the std is 0 and this term is a constant penalty — that's
    intentional, it says "you should be running with batch size > 1".
    """
    if pooled.size(0) < 2:
        return pooled.new_zeros(())
    std = pooled.std(dim=0, unbiased=False)                  # (D,)
    return F.relu(target - std).mean()


def decorrelation_reg(pooled: torch.Tensor) -> torch.Tensor:
    """Frobenius distance from off-diagonal correlation to zero.

    pooled : (B, D)
    Returns ||corr - I||_F^2 / D^2 (averaged so it's roughly O(1)).
    """
    B, D = pooled.shape
    if B < 2:
        return pooled.new_zeros(())
    p = pooled - pooled.mean(dim=0, keepdim=True)
    p = p / (p.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    corr = (p.t() @ p) / (B - 1)                             # (D, D)
    eye = torch.eye(D, device=corr.device, dtype=corr.dtype)
    off = corr - eye
    return (off ** 2).mean()


def total_loss(
    out: dict[str, torch.Tensor],
    is_present: torch.Tensor,
    *,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    variance_target: float = 0.5,
    variance_weight: float = 0.1,
    decorr_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    targets = is_present.float()
    logit = out["existence_logit"]
    pooled = out["pooled"]
    focal = focal_bce_loss(logit, targets, focal_alpha, focal_gamma)
    var = variance_reg(pooled, target=variance_target)
    decor = decorrelation_reg(pooled)
    total = focal + variance_weight * var + decorr_weight * decor
    return {
        "loss": total, "focal": focal, "variance": var, "decorrelation": decor,
    }
