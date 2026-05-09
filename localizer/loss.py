"""Localizer loss.

Components (each weight is overridable from cfg):

  L_patch_ce
      Cross-entropy on the patch index. The target is the patch on OWLv2's
      grid whose anchor centre is closest to the GT bbox centre. This loss
      directly demands prototype DISCRIMINATION across patches — there is
      no fixed-point "pred everywhere" solution like the soft-box trick had.

  L_l1 + L_giou
      Standard DETR-style box regression on the argmax-selected box.
      Computed via gather(pred_boxes, best_idx) where best_idx is the
      argmax of pred_logits. The gradient on `pred_boxes[best_idx]` only
      reaches the box_head (and the vision backbone via LoRA) — at L1 the
      box_head is frozen, so this term contributes ZERO gradient at L1
      and is silently a no-op there. The patch-CE drives all L1 learning.

L1 stage (box_head frozen):       L = L_patch_ce
L2/L3 stages (box_head trainable): L = L_patch_ce + λ_l1 * L_l1 + λ_giou * L_giou

Computed only over POSITIVE episodes (the trainer guarantees this).
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


# ---------------------------------------------------------------------------
# Patch-classification target
# ---------------------------------------------------------------------------


def gt_patch_index(
    gt_bbox_cxcywh: torch.Tensor, gh: int, gw: int,
) -> torch.Tensor:
    """For each batch element, return the index of the patch whose centre is
    closest to the GT bbox centre.

    gt_bbox_cxcywh : (B, 4) in [0, 1] — letterboxed image coordinates.
    gh, gw         : OWLv2's patch grid (rows, cols), e.g. 48×48 at S=768.

    Returns: (B,) long tensor in [0, gh*gw).

    OWLv2 lays out patches row-major. Each patch i = row*gw + col owns a
    cell centred at ((col + 0.5)/gw, (row + 0.5)/gh) in [0, 1].
    We snap the GT centre to the nearest cell.
    """
    if gt_bbox_cxcywh.dim() != 2 or gt_bbox_cxcywh.size(-1) != 4:
        raise ValueError(
            f"gt_bbox_cxcywh must be (B, 4), got {tuple(gt_bbox_cxcywh.shape)}"
        )
    cx = gt_bbox_cxcywh[:, 0].clamp(min=0.0, max=1.0 - 1e-6)
    cy = gt_bbox_cxcywh[:, 1].clamp(min=0.0, max=1.0 - 1e-6)
    col = (cx * gw).floor().long().clamp(min=0, max=gw - 1)
    row = (cy * gh).floor().long().clamp(min=0, max=gh - 1)
    return row * gw + col


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------


def total_loss(
    out: dict[str, torch.Tensor],
    gt_bbox_cxcywh: torch.Tensor,
    *,
    lambda_patch_ce: float = 1.0,
    lambda_l1: float = 5.0,
    lambda_giou: float = 2.0,
    use_box_loss: bool = True,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Localizer total loss.

    Args:
      out               : output dict from MultiShotLocalizer.forward
      gt_bbox_cxcywh    : (B, 4) GT box, normalised cxcywh in [0, 1]
      lambda_patch_ce   : weight on the patch-classification (CE) term
      lambda_l1         : weight on the L1 box loss (only when use_box_loss=True)
      lambda_giou       : weight on the GIoU box loss (only when use_box_loss=True)
      use_box_loss      : disable to suppress the L1+GIoU terms.
                          Set to False at L1 (where box_head is frozen);
                          True at L2 / L3.
      label_smoothing   : pass-through to F.cross_entropy.

    Returns:
      {"loss", "patch_ce", "l1", "giou"}
    """
    pred_logits: torch.Tensor = out["pred_logits"]               # (B, P)
    pred_boxes:  torch.Tensor = out["pred_boxes"]                # (B, P, 4)
    gh, gw = out["patch_grid"]
    P = pred_logits.size(-1)
    if gh * gw != P:
        # The class predictor's output P should match the box predictor's grid.
        raise RuntimeError(
            f"patch_grid {gh}×{gw}={gh*gw} does not match pred_logits P={P}"
        )

    # Patch-CE target: index of the patch whose anchor centre is nearest GT centre.
    target_idx = gt_patch_index(gt_bbox_cxcywh, gh, gw).to(pred_logits.device)
    patch_ce = F.cross_entropy(pred_logits, target_idx, label_smoothing=label_smoothing)

    losses: dict[str, torch.Tensor] = {"patch_ce": patch_ce}
    total = lambda_patch_ce * patch_ce

    # Box regression on the argmax-selected box. Only meaningful (i.e.
    # produces gradient that can change anything) when box_head is
    # trainable. At L1 we skip these terms via use_box_loss=False.
    if use_box_loss:
        ar = torch.arange(pred_logits.size(0), device=pred_logits.device)
        best_idx = pred_logits.argmax(dim=-1)
        pred_box = pred_boxes[ar, best_idx]                      # (B, 4)
        l1 = F.l1_loss(pred_box, gt_bbox_cxcywh, reduction="mean")
        pred_xyxy = _cxcywh_to_xyxy(pred_box)
        gt_xyxy = _cxcywh_to_xyxy(gt_bbox_cxcywh)
        giou = giou_loss(pred_xyxy, gt_xyxy).mean()
        losses["l1"] = l1
        losses["giou"] = giou
        total = total + lambda_l1 * l1 + lambda_giou * giou
    else:
        losses["l1"] = patch_ce.new_zeros(())
        losses["giou"] = patch_ce.new_zeros(())

    losses["loss"] = total
    return losses
