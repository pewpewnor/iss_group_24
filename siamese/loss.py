"""Siamese existence loss = focal_BCE + variance reg + decorrelation reg.

Focal-α semantics (sigmoid focal, this implementation):
  alpha = weight on POSITIVE-class CE.   (1 - alpha) = weight on NEGATIVE-class CE.

  alpha = 0.5 ⇒ balanced.
  alpha < 0.5 ⇒ heavier penalty on FALSE POSITIVES (negatives dominate).
  alpha > 0.5 ⇒ heavier penalty on FALSE NEGATIVES (positives dominate).

Default ``alpha=0.5`` (balanced) — this fixes the previous training failure
mode where alpha=0.25 + neg_prob=0.75 collapsed predictions toward 0 and
no positive ever crossed the 0.5 threshold. The variance / decorrelation
regularisers continue to prevent representation collapse.

The three terms are summed with configurable weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid-focal BCE."""
    p = torch.sigmoid(logits)
    eps = 1e-6
    ce_pos = -targets * torch.log(p.clamp(min=eps))
    ce_neg = -(1 - targets) * torch.log((1 - p).clamp(min=eps))
    fl_pos = alpha * (1 - p).pow(gamma) * ce_pos
    fl_neg = (1 - alpha) * p.pow(gamma) * ce_neg
    return (fl_pos + fl_neg).mean()


def variance_reg(pooled: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    """Penalise low std across the batch dimension."""
    if pooled.size(0) < 2:
        return pooled.new_zeros(())
    std = pooled.std(dim=0, unbiased=False)
    return F.relu(target - std).mean()


def decorrelation_reg(pooled: torch.Tensor) -> torch.Tensor:
    """||corr - I||_F^2 / D^2 — penalises collapse to a single direction."""
    B, D = pooled.shape
    if B < 2:
        return pooled.new_zeros(())
    p = pooled - pooled.mean(dim=0, keepdim=True)
    p = p / (p.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    corr = (p.t() @ p) / (B - 1)
    eye = torch.eye(D, device=corr.device, dtype=corr.dtype)
    off = corr - eye
    return (off ** 2).mean()


def total_loss(
    out: dict[str, torch.Tensor],
    is_present: torch.Tensor,
    *,
    focal_alpha: float = 0.5,
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
