"""Localizer loss.

The trained forward now emits, per query:
  pred_logits_fg : (B, P)   raw foreground logits over P patches.
  bg_logit       : (B,)     learned background scalar logit.
  pred_boxes     : (B, P, 4) cxcywh in [0,1] per patch.

We define a JOINT softmax over (P+1) classes (P foreground patches + 1
background) and supervise it with cross-entropy:

  * Positive episode → target = nearest patch index of GT centre.
    Optionally soft-smear the target over a 3×3 spatial neighbourhood.
  * Negative episode → target = P (the bg column).

Box regression (L1 + GIoU) is applied ONLY to positives, on the argmax-fg
patch. We also add a log-area regulariser that directly penalises the box
head outputting too-small / too-large areas relative to GT.

L1 stage (box_head frozen):        L = L_patch_ce
L2 / L3 stages (box_head trainable):
                                   L = L_patch_ce + λ_l1 * L_l1
                                     + λ_giou * L_giou
                                     + λ_log_area * L_log_area
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def _cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _box_area(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def giou_loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """1 - GIoU. Vectorised over a batch."""
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
    return 1.0 - (iou - (enc_area - union) / enc_area)


# ---------------------------------------------------------------------------
# Patch-classification target
# ---------------------------------------------------------------------------


def gt_patch_index(
    gt_bbox_cxcywh: torch.Tensor, gh: int, gw: int,
) -> torch.Tensor:
    """For each batch element, return the index of the patch whose centre is
    closest to the GT bbox centre (row * gw + col)."""
    if gt_bbox_cxcywh.dim() != 2 or gt_bbox_cxcywh.size(-1) != 4:
        raise ValueError(
            f"gt_bbox_cxcywh must be (B, 4), got {tuple(gt_bbox_cxcywh.shape)}"
        )
    cx = gt_bbox_cxcywh[:, 0].clamp(min=0.0, max=1.0 - 1e-6)
    cy = gt_bbox_cxcywh[:, 1].clamp(min=0.0, max=1.0 - 1e-6)
    col = (cx * gw).floor().long().clamp(min=0, max=gw - 1)
    row = (cy * gh).floor().long().clamp(min=0, max=gh - 1)
    return row * gw + col


def _soft_patch_target(
    target_idx: torch.Tensor, gh: int, gw: int,
    neighbour_radius: int, neighbour_weight: float,
) -> torch.Tensor:
    """Build a (B, P) soft target distribution.

    The centre patch gets weight (1 - neighbour_weight).  The remaining
    neighbour_weight is uniformly spread over the 8-neighbourhood
    (radius=1) or larger window. If neighbour_weight = 0 we return a
    one-hot.
    """
    B = target_idx.shape[0]
    P = gh * gw
    device = target_idx.device
    targets = torch.zeros(B, P, device=device, dtype=torch.float32)
    if neighbour_weight <= 0 or neighbour_radius <= 0:
        targets[torch.arange(B, device=device), target_idx] = 1.0
        return targets

    centre_w = 1.0 - neighbour_weight
    for b in range(B):
        idx = int(target_idx[b].item())
        row, col = idx // gw, idx % gw
        nbr: list[int] = []
        for dr in range(-neighbour_radius, neighbour_radius + 1):
            for dc in range(-neighbour_radius, neighbour_radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < gh and 0 <= cc < gw:
                    nbr.append(rr * gw + cc)
        targets[b, idx] = centre_w
        if nbr:
            w_each = neighbour_weight / len(nbr)
            for n in nbr:
                targets[b, n] = w_each
        else:
            targets[b, idx] = 1.0
    return targets


def _ce_joint_soft(
    joint_logits: torch.Tensor, soft_fg: torch.Tensor, is_present: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    """Cross-entropy over (P + 1) joint logits.

    Positives use ``soft_fg`` (B, P) as the target over the first P columns
    (and zero on the bg column).
    Negatives are one-hot on the bg column (index P).

    label_smoothing : standard softmax-CE label smoothing applied to BOTH
                      positive and negative targets.
    """
    B, Pp1 = joint_logits.shape
    P = Pp1 - 1
    targets = torch.zeros(B, Pp1, device=joint_logits.device, dtype=joint_logits.dtype)
    if is_present.any():
        pos_mask = is_present
        targets[pos_mask, :P] = soft_fg[pos_mask].to(targets.dtype)
    if (~is_present).any():
        neg_mask = ~is_present
        targets[neg_mask, P] = 1.0

    if label_smoothing > 0.0:
        eps = float(label_smoothing)
        targets = targets * (1.0 - eps) + eps / Pp1

    log_prob = F.log_softmax(joint_logits, dim=-1)
    loss = -(targets * log_prob).sum(dim=-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------


def total_loss(
    out: dict[str, torch.Tensor],
    gt_bbox_cxcywh: torch.Tensor,
    is_present: torch.Tensor,
    *,
    lambda_patch_ce: float = 1.0,
    lambda_l1: float = 2.0,
    lambda_giou: float = 4.0,
    lambda_log_area: float = 0.2,
    use_box_loss: bool = True,
    label_smoothing: float = 0.05,
    neighbour_radius: int = 1,
    neighbour_weight: float = 0.30,
) -> dict[str, torch.Tensor]:
    """Localizer total loss with abstain channel + log-area regularisation.

    Args:
      out            : output dict from ``MultiShotLocalizer.forward``.
      gt_bbox_cxcywh : (B, 4) GT box in [0, 1] cxcywh (zeros for negatives).
      is_present     : (B,) bool — True for positive episodes, False otherwise.
                       Negatives target the bg (abstain) class.
      lambda_l1, lambda_giou : weights on the box regression terms.
      lambda_log_area: weight on |log(pred_area / gt_area)|^2  (size regulariser).
      use_box_loss   : when False, suppress all box-side terms (L1 stage).
      label_smoothing: applied uniformly across (P+1) classes.
      neighbour_radius, neighbour_weight :
          Soft-target spatial smoothing for the positive patch-CE.
          radius=1 + weight=0.30 means the GT patch absorbs 70 % of the mass and
          its 8 neighbours share the remaining 30 %.

    Returns:
      {"loss", "patch_ce", "l1", "giou", "log_area"}.
    """
    pred_logits_fg: torch.Tensor = out["pred_logits_fg"]        # (B, P)
    bg_logit:      torch.Tensor = out["bg_logit"]               # (B,)
    pred_boxes:    torch.Tensor = out["pred_boxes"]             # (B, P, 4)
    gh, gw = out["patch_grid"]
    P = pred_logits_fg.size(-1)
    if gh * gw != P:
        raise RuntimeError(
            f"patch_grid {gh}×{gw}={gh*gw} does not match pred_logits P={P}"
        )

    device = pred_logits_fg.device
    if is_present.dtype != torch.bool:
        is_present = is_present.to(torch.bool)
    is_present = is_present.to(device)

    # Joint logits.
    joint = torch.cat([pred_logits_fg, bg_logit.unsqueeze(-1)], dim=-1)  # (B, P+1)

    # Soft positive targets in fg-only space.
    target_idx = gt_patch_index(gt_bbox_cxcywh, gh, gw).to(device)
    soft_fg = _soft_patch_target(
        target_idx, gh, gw,
        neighbour_radius=int(neighbour_radius),
        neighbour_weight=float(neighbour_weight),
    )

    patch_ce = _ce_joint_soft(joint, soft_fg, is_present, label_smoothing)

    losses: dict[str, torch.Tensor] = {"patch_ce": patch_ce}
    total = lambda_patch_ce * patch_ce

    # ── Box regression — positives only. ─────────────────────────────────
    losses["l1"] = patch_ce.new_zeros(())
    losses["giou"] = patch_ce.new_zeros(())
    losses["log_area"] = patch_ce.new_zeros(())
    if use_box_loss and is_present.any():
        pos_mask = is_present
        ar = torch.arange(pred_logits_fg.size(0), device=device)[pos_mask]
        best_idx = pred_logits_fg.argmax(dim=-1)[pos_mask]
        pred_box = pred_boxes[ar, best_idx]                       # (Bp, 4)
        gt_box   = gt_bbox_cxcywh[pos_mask]                       # (Bp, 4)

        l1 = F.l1_loss(pred_box, gt_box, reduction="mean")
        pred_xyxy = _cxcywh_to_xyxy(pred_box)
        gt_xyxy = _cxcywh_to_xyxy(gt_box)
        giou = giou_loss(pred_xyxy, gt_xyxy).mean()

        # Log-area regulariser: directly fights the "too small" pathology
        # observed at L3. Use (log(a+ε) - log(g+ε))² so it's symmetric and
        # bounded for tiny boxes. Clamped at 25 (i.e. ~5 nats of log-ratio
        # difference) to prevent early-training explosions on degenerate boxes
        # from dominating the total loss.
        pred_area = (pred_box[:, 2].clamp(min=0) * pred_box[:, 3].clamp(min=0)).clamp(min=1e-6)
        gt_area   = (gt_box[:, 2].clamp(min=0) * gt_box[:, 3].clamp(min=0)).clamp(min=1e-6)
        log_area = (pred_area.log() - gt_area.log()).pow(2).clamp(max=25.0).mean()

        losses["l1"] = l1
        losses["giou"] = giou
        losses["log_area"] = log_area
        total = total + lambda_l1 * l1 + lambda_giou * giou + lambda_log_area * log_area

    losses["loss"] = total
    return losses
