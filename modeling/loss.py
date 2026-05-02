"""FCOS-style target assignment + GIoU box loss + sigmoid focal classification loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_targets(
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    grid: int = 14,
    stride: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense targets.

    Returns:
      conf_target: (B, 1, G, G) in {0, 1}
      reg_target:  (B, 4, G, G) — only meaningful where pos_mask is True
      pos_mask:    (B, G, G) bool
    """
    b = gt_bbox.shape[0]
    device = gt_bbox.device

    coords = (torch.arange(grid, device=device, dtype=gt_bbox.dtype) + 0.5) * stride
    cy_grid, cx_grid = torch.meshgrid(coords, coords, indexing="ij")  # (G, G)

    x1 = gt_bbox[:, 0].view(b, 1, 1)
    y1 = gt_bbox[:, 1].view(b, 1, 1)
    x2 = gt_bbox[:, 2].view(b, 1, 1)
    y2 = gt_bbox[:, 3].view(b, 1, 1)

    inside = (
        (cx_grid.unsqueeze(0) >= x1)
        & (cx_grid.unsqueeze(0) <= x2)
        & (cy_grid.unsqueeze(0) >= y1)
        & (cy_grid.unsqueeze(0) <= y2)
    )  # (B, G, G)
    inside = inside & is_present.view(b, 1, 1)

    # Guarantee the cell containing the GT centre is positive even when the GT
    # box is smaller than one cell: otherwise tiny objects produce no positives.
    cx = (x1 + x2).squeeze(-1).squeeze(-1) * 0.5
    cy = (y1 + y2).squeeze(-1).squeeze(-1) * 0.5
    j_center = (cx / stride).long().clamp(0, grid - 1)
    i_center = (cy / stride).long().clamp(0, grid - 1)
    b_idx = torch.arange(b, device=device)
    inside[b_idx, i_center, j_center] = inside[b_idx, i_center, j_center] | is_present

    conf_target = inside.float().unsqueeze(1)  # (B, 1, G, G)

    j_idx = (
        torch.arange(grid, device=device, dtype=gt_bbox.dtype)
        .view(1, 1, grid)
        .expand(b, grid, grid)
    )
    i_idx = (
        torch.arange(grid, device=device, dtype=gt_bbox.dtype)
        .view(1, grid, 1)
        .expand(b, grid, grid)
    )

    cx_b = cx.view(b, 1, 1)
    cy_b = cy.view(b, 1, 1)
    w_b = (x2 - x1).clamp(min=1.0)
    h_b = (y2 - y1).clamp(min=1.0)

    dx = cx_b / stride - j_idx
    dy = cy_b / stride - i_idx
    dw = torch.log(w_b / stride).expand(b, grid, grid)
    dh = torch.log(h_b / stride).expand(b, grid, grid)

    reg_target = torch.stack([dx, dy, dw, dh], dim=1)  # (B, 4, G, G)
    return conf_target, reg_target, inside


def decode_pred_box(reg: torch.Tensor, stride: int = 16) -> torch.Tensor:
    """Decode dense regression map to dense (B, 4, G, G) xyxy boxes."""
    b, _, gh, gw = reg.shape
    device = reg.device
    j_idx = (
        torch.arange(gw, device=device, dtype=reg.dtype).view(1, 1, gw).expand(b, gh, gw)
    )
    i_idx = (
        torch.arange(gh, device=device, dtype=reg.dtype).view(1, gh, 1).expand(b, gh, gw)
    )
    cx_abs = (j_idx + reg[:, 0]) * stride
    cy_abs = (i_idx + reg[:, 1]) * stride
    w_abs = torch.exp(reg[:, 2].clamp(max=6.0)) * stride
    h_abs = torch.exp(reg[:, 3].clamp(max=6.0)) * stride
    boxes = torch.stack(
        [cx_abs - w_abs / 2, cy_abs - h_abs / 2, cx_abs + w_abs / 2, cy_abs + h_abs / 2],
        dim=1,
    )
    return boxes


def giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Element-wise GIoU between matched (..., 4) boxes in xyxy."""
    ax1, ay1, ax2, ay2 = boxes_a.unbind(-1)
    bx1, by1, bx2, by2 = boxes_b.unbind(-1)
    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    iou = inter / union
    enc_x1 = torch.minimum(ax1, bx1)
    enc_y1 = torch.minimum(ay1, by1)
    enc_x2 = torch.maximum(ax2, bx2)
    enc_y2 = torch.maximum(ay2, by2)
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-6
    return iou - (enc_area - union) / enc_area


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t).pow(gamma) * bce


def total_loss(
    pred: dict[str, torch.Tensor],
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    grid: int = 14,
    stride: int = 16,
) -> dict[str, torch.Tensor]:
    reg_pred = pred["reg"]
    conf_logits = pred["conf"]

    conf_target, _, pos_mask = make_targets(gt_bbox, is_present, grid=grid, stride=stride)

    focal = sigmoid_focal_loss(conf_logits, conf_target).mean()

    if pos_mask.any():
        decoded = decode_pred_box(reg_pred, stride=stride)  # (B, 4, G, G)
        b = gt_bbox.shape[0]
        gt_exp = gt_bbox.view(b, 4, 1, 1).expand_as(decoded)
        pred_pos = decoded.permute(0, 2, 3, 1)[pos_mask]      # (P, 4)
        gt_pos = gt_exp.permute(0, 2, 3, 1)[pos_mask]         # (P, 4)
        box_loss = (1.0 - giou(pred_pos, gt_pos)).mean()
    else:
        box_loss = torch.zeros((), device=reg_pred.device)

    loss = focal + box_loss
    return {"loss": loss, "focal": focal.detach(), "box": box_loss.detach()}
