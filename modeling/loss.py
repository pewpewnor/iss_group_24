"""FCOS-style target assignment + loss functions for few-shot localization.

Losses:
  varifocal_loss          — soft focal loss with IoU-based positive targets
  area_weighted_giou_loss — GIoU loss up-weighted for small objects
  presence BCE            — binary cross-entropy with IoU label smoothing
  attention_bbox_loss     — KL between learned saliency attn and bbox target
  nt_xent_loss            — prototype uniformity (contrastive)
  vicreg_loss             — prototype variance + covariance regularisation
  triplet_loss            — hard-negative prototype margin loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


# ---------------------------------------------------------------------------
# Box IoU utility  (element-wise matched pairs — more efficient than box_iou)
# ---------------------------------------------------------------------------


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Plain IoU for matched (..., 4) pairs in xyxy format."""
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-6)


# ---------------------------------------------------------------------------
# Target assignment
# ---------------------------------------------------------------------------


def make_targets(
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    grid: int = 14,
    stride: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FCOS-style dense targets.

    Returns:
      conf_target: (B, 1, G, G) in {0, 1}
      reg_target:  (B, 4, G, G) — only meaningful where pos_mask is True
      pos_mask:    (B, G, G) bool
    """
    b = gt_bbox.shape[0]
    device = gt_bbox.device

    coords = (torch.arange(grid, device=device, dtype=gt_bbox.dtype) + 0.5) * stride
    cy_grid, cx_grid = torch.meshgrid(coords, coords, indexing="ij")

    x1 = gt_bbox[:, 0].view(b, 1, 1)
    y1 = gt_bbox[:, 1].view(b, 1, 1)
    x2 = gt_bbox[:, 2].view(b, 1, 1)
    y2 = gt_bbox[:, 3].view(b, 1, 1)

    inside = (
        (cx_grid.unsqueeze(0) >= x1)
        & (cx_grid.unsqueeze(0) <= x2)
        & (cy_grid.unsqueeze(0) >= y1)
        & (cy_grid.unsqueeze(0) <= y2)
    ) & is_present.view(b, 1, 1)

    # Guarantee at least the GT-centre cell is positive for tiny objects
    cx = (x1 + x2).squeeze(-1).squeeze(-1) * 0.5
    cy = (y1 + y2).squeeze(-1).squeeze(-1) * 0.5
    j_center = (cx / stride).long().clamp(0, grid - 1)
    i_center = (cy / stride).long().clamp(0, grid - 1)
    b_idx = torch.arange(b, device=device)
    inside[b_idx, i_center, j_center] = inside[b_idx, i_center, j_center] | is_present

    conf_target = inside.float().unsqueeze(1)  # (B, 1, G, G)

    j_idx = torch.arange(grid, device=device, dtype=gt_bbox.dtype).view(1, 1, grid).expand(b, grid, grid)
    i_idx = torch.arange(grid, device=device, dtype=gt_bbox.dtype).view(1, grid, 1).expand(b, grid, grid)
    cx_b, cy_b = cx.view(b, 1, 1), cy.view(b, 1, 1)
    w_b = (x2 - x1).clamp(min=1.0)
    h_b = (y2 - y1).clamp(min=1.0)

    reg_target = torch.stack(
        [cx_b / stride - j_idx, cy_b / stride - i_idx,
         torch.log(w_b / stride).expand(b, grid, grid),
         torch.log(h_b / stride).expand(b, grid, grid)],
        dim=1,
    )
    return conf_target, reg_target, inside


def decode_pred_box(reg: torch.Tensor, stride: int = 16) -> torch.Tensor:
    """Decode dense (B, 4, G, G) regression map to xyxy boxes."""
    b, _, gh, gw = reg.shape
    device = reg.device
    j_idx = torch.arange(gw, device=device, dtype=reg.dtype).view(1, 1, gw).expand(b, gh, gw)
    i_idx = torch.arange(gh, device=device, dtype=reg.dtype).view(1, gh, 1).expand(b, gh, gw)
    cx_abs = (j_idx + reg[:, 0]) * stride
    cy_abs = (i_idx + reg[:, 1]) * stride
    w_abs = torch.exp(reg[:, 2].clamp(max=6.0)) * stride
    h_abs = torch.exp(reg[:, 3].clamp(max=6.0)) * stride
    return torch.stack(
        [cx_abs - w_abs / 2, cy_abs - h_abs / 2, cx_abs + w_abs / 2, cy_abs + h_abs / 2],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Classification losses
# ---------------------------------------------------------------------------


def varifocal_loss(
    pred_logits: torch.Tensor,
    gt_score: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Varifocal loss: soft focal loss with IoU-quality as positive target.

    Positive cells (gt_score > 0) are weighted by how far the prediction is from
    the target IoU. Negative cells (gt_score = 0) are down-weighted by the
    focal term when the model is already confident they are negative.
    Returns sum (caller normalises by num_pos).
    """
    pred_sigmoid = torch.sigmoid(pred_logits)
    weight = (
        alpha * (gt_score - pred_sigmoid).abs().pow(gamma) * gt_score
        + (1 - alpha) * pred_sigmoid.pow(gamma) * (1 - gt_score)
    )
    loss = F.binary_cross_entropy_with_logits(pred_logits, gt_score, reduction="none")
    return (loss * weight).sum()


# ---------------------------------------------------------------------------
# Box regression loss
# ---------------------------------------------------------------------------


def area_weighted_giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """GIoU loss with inverse-area weighting to up-weight small objects.

    Small objects (< 10% image area) receive up to 3x the gradient of large ones,
    counteracting the natural bias of GIoU toward large boxes.
    """
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    norm_area = gt_area / (224.0 * 224.0)
    weights = 1.0 + 2.0 * torch.exp(-5.0 * norm_area)
    loss = generalized_box_iou_loss(pred, gt, reduction="none")
    return (loss * weights).mean()


# ---------------------------------------------------------------------------
# Presence classification helpers
# ---------------------------------------------------------------------------


def _smooth_presence_targets(
    is_present: torch.Tensor,
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    smooth: float = 0.1,
) -> torch.Tensor:
    """IoU-based label smoothing for presence BCE.

    For positive episodes, the target is softened from 1.0 to
    (1 - smooth) * IoU + smooth * 0.5. A model that localizes well gets a target
    closer to 1; poor localisation gets a softer target, reducing overconfident
    penalisation on ambiguous GT boxes.
    """
    targets = is_present.float()
    present = is_present.bool()
    if present.any():
        iou = _iou_xyxy(pred_bbox[present], gt_bbox[present])
        targets = targets.clone()
        targets[present] = (1 - smooth) * iou + smooth * 0.5
    return targets


# ---------------------------------------------------------------------------
# Prototype regularisation losses
# ---------------------------------------------------------------------------


def nt_xent_loss(prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent uniformity loss — drives all prototypes to be maximally distinct."""
    B = prototypes.shape[0]
    if B < 2:
        return torch.zeros((), device=prototypes.device)
    p = F.normalize(prototypes, dim=-1)
    sim = torch.mm(p, p.T) / temperature
    sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=prototypes.device), float("-inf"))
    return torch.logsumexp(sim, dim=1).mean()


def vicreg_loss(
    prototypes: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance + covariance terms (no invariance — single-view setup)."""
    B, D = prototypes.shape
    if B < 2:
        return torch.zeros((), device=prototypes.device)
    z = prototypes - prototypes.mean(dim=0)
    var_loss = F.relu(gamma - torch.sqrt(z.var(dim=0) + eps)).mean()
    cov = (z.T @ z) / (B - 1)
    cov_loss = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / D
    return var_loss + 0.04 * cov_loss


def triplet_loss(
    protos: torch.Tensor,
    instance_ids: list[str],
    margin: float = 0.3,
) -> torch.Tensor:
    """Hard-negative prototype triplet loss.

    For each anchor, finds the hardest negative (most similar prototype from a
    different instance) and the hardest positive (if a same-instance prototype
    exists in the batch). Pushes (anchor, positive) closer and (anchor, negative)
    further apart with a margin.

    Note: in episodic training each batch element is a unique instance, so positive
    pairs are rare. The loss primarily contributes as a hard-negative push alongside
    NT-Xent.
    """
    B = protos.shape[0]
    if B < 2:
        return torch.zeros((), device=protos.device)
    normed = F.normalize(protos, dim=1)
    sims = normed @ normed.T  # (B, B)
    loss = torch.zeros((), device=protos.device)
    count = 0
    for i, iid in enumerate(instance_ids):
        pos_mask = torch.tensor(
            [j != i and instance_ids[j] == iid for j in range(B)],
            device=protos.device,
        )
        neg_mask = torch.tensor(
            [instance_ids[j] != iid for j in range(B)],
            device=protos.device,
        )
        if not pos_mask.any() or not neg_mask.any():
            continue
        pos_sim = sims[i][pos_mask].max()
        neg_sim = sims[i][neg_mask].max()  # hardest negative
        loss = loss + F.relu(neg_sim - pos_sim + margin)
        count += 1
    return loss / max(count, 1)


# ---------------------------------------------------------------------------
# Saliency attention auxiliary loss
# ---------------------------------------------------------------------------


def attention_bbox_loss(
    attn_map: torch.Tensor,
    support_bboxes: torch.Tensor,
    img_size: int = 224,
    stride: int = 32,
) -> torch.Tensor:
    """KL divergence between learned saliency attention and a bbox target.

    During training we have support bboxes (from cleaner.py); use them to
    pull the saliency attention toward the foreground region. At inference
    the model uses the learned attention freely without needing any bbox.

    Args:
        attn_map: (B, K, H, W) — softmax-normalised over (H, W) per support
        support_bboxes: (B, K, 4) in image coords (xyxy, 224 image space)

    Returns:
        Scalar mean KL across (B, K) supports.
    """
    b, k, h, w = attn_map.shape
    device = attn_map.device
    dtype = attn_map.dtype

    # Cell centres in image coords
    cy = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    cx = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride

    bb = support_bboxes.reshape(b * k, 4).to(dtype)
    x1 = bb[:, 0:1].unsqueeze(-1)  # (B*K, 1, 1)
    y1 = bb[:, 1:2].unsqueeze(-1)
    x2 = bb[:, 2:3].unsqueeze(-1)
    y2 = bb[:, 3:4].unsqueeze(-1)

    inside = (
        (cx.view(1, 1, w) >= x1) & (cx.view(1, 1, w) <= x2)
        & (cy.view(1, h, 1) >= y1) & (cy.view(1, h, 1) <= y2)
    )  # (B*K, H, W)
    target = inside.to(dtype)
    target_sum = target.sum(dim=(1, 2), keepdim=True)
    # Empty bbox -> uniform target (no signal, doesn't push attention anywhere)
    safe = target_sum > 0
    uniform = torch.full_like(target, 1.0 / (h * w))
    target = torch.where(safe, target / target_sum.clamp(min=1.0), uniform)
    target = target.view(b, k, h, w)

    eps = 1e-8
    kl = (target * (torch.log(target + eps) - torch.log(attn_map + eps))).sum(dim=(2, 3))
    return kl.mean()


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------


def total_loss(
    pred: dict[str, torch.Tensor],
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    grid: int = 14,
    stride: int = 16,
    support_bboxes: torch.Tensor | None = None,
    attn_loss_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    reg_pred = pred["reg"]
    conf_logits = pred["conf"]

    conf_target, _, pos_mask = make_targets(gt_bbox, is_present, grid=grid, stride=stride)
    num_pos = pos_mask.sum().clamp(min=1)

    # Decode all cells once — reused for varifocal targets, box loss, and label smoothing
    decoded = decode_pred_box(reg_pred, stride=stride)  # (B, 4, G, G)
    b = gt_bbox.shape[0]
    gt_exp = gt_bbox.view(b, 4, 1, 1).expand_as(decoded)

    # Varifocal loss: build soft targets from predicted IoU at each cell
    with torch.no_grad():
        iou_flat = _iou_xyxy(
            decoded.permute(0, 2, 3, 1).reshape(-1, 4),
            gt_exp.permute(0, 2, 3, 1).reshape(-1, 4),
        )
        gt_score_map = iou_flat.view(b, 1, grid, grid) * conf_target
    focal = varifocal_loss(conf_logits, gt_score_map) / num_pos

    # Area-weighted GIoU box loss (positive cells only)
    if pos_mask.any():
        pred_pos = decoded.permute(0, 2, 3, 1)[pos_mask]   # (P, 4)
        gt_pos = gt_exp.permute(0, 2, 3, 1)[pos_mask]      # (P, 4)
        box_loss = area_weighted_giou_loss(pred_pos, gt_pos)
    else:
        box_loss = torch.zeros((), device=reg_pred.device)

    # Presence BCE with IoU-based label smoothing
    if "presence_logit" in pred:
        flat_conf = conf_logits.view(b, -1)
        best = flat_conf.argmax(dim=1)
        b_idx = torch.arange(b, device=reg_pred.device)
        best_boxes = decoded.permute(0, 2, 3, 1)[b_idx, (best // grid).long(), (best % grid).long()]
        with torch.no_grad():
            smooth_targets = _smooth_presence_targets(is_present, best_boxes, gt_bbox)
        presence_loss = F.binary_cross_entropy_with_logits(
            pred["presence_logit"], smooth_targets
        )
    else:
        presence_loss = torch.zeros((), device=reg_pred.device)

    # Saliency attention auxiliary loss (only when both attn map and bboxes available)
    attn_loss = torch.zeros((), device=reg_pred.device)
    support_attn = pred.get("support_attn")
    if support_attn is not None and support_bboxes is not None:
        attn_loss = attention_bbox_loss(support_attn, support_bboxes)

    loss = focal + box_loss + presence_loss + attn_loss_weight * attn_loss
    return {
        "loss": loss,
        "focal": focal.detach(),
        "box": box_loss.detach(),
        "presence": presence_loss.detach(),
        "attn": attn_loss.detach(),
    }
