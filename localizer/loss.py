"""Localizer loss = L1 + GIoU on the top-1 predicted box.

Computed only over POSITIVE episodes (the localizer trainer guarantees this).
No focal loss, no anti-collapse, no contrastive — those are siamese concerns.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _box_area(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def giou_loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """1 - GIoU. Vectorized over a batch."""
    inter_x1 = torch.maximum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    inter_y1 = torch.maximum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    inter_x2 = torch.minimum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    inter_y2 = torch.minimum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_p = _box_area(pred_xyxy)
    area_g = _box_area(gt_xyxy)
    union = area_p + area_g - inter + 1e-6
    iou = inter / union
    enc_x1 = torch.minimum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    enc_y1 = torch.minimum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    enc_x2 = torch.maximum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    enc_y2 = torch.maximum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-6
    giou = iou - (enc_area - union) / enc_area
    return 1.0 - giou


def total_loss(
    out: dict[str, torch.Tensor],
    gt_bbox_cxcywh: torch.Tensor,
    *,
    lambda_l1: float = 5.0,
    lambda_giou: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Localizer L1 + GIoU loss.

    Uses ``soft_box`` (softmax-weighted average of all patch boxes by
    pred_logits) so the gradient flows through the prototype/fusion path
    even when the box_head itself is frozen. ``best_box`` is reported for
    diagnostics (it's the argmax pick used in metrics) but not used by the
    loss directly.
    """
    pred = out.get("soft_box", out["best_box"])             # (B, 4)
    l1 = F.l1_loss(pred, gt_bbox_cxcywh, reduction="mean")
    pred_xyxy = _cxcywh_to_xyxy(pred)
    gt_xyxy = _cxcywh_to_xyxy(gt_bbox_cxcywh)
    giou = giou_loss(pred_xyxy, gt_xyxy).mean()
    total = lambda_l1 * l1 + lambda_giou * giou
    return {"loss": total, "l1": l1, "giou": giou}
