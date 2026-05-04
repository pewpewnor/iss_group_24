"""FCOS-style target assignment + loss functions for few-shot localization.

Losses:
  focal_loss              — standard binary focal loss (hard 0/1 targets)
  area_weighted_giou_loss — GIoU loss up-weighted for small objects
  presence BCE            — binary cross-entropy
  attention_bbox_loss     — KL between aggregated support attention and bbox region
  nt_xent_loss            — prototype uniformity (contrastive)
  vicreg_loss             — prototype variance + covariance regularisation
  triplet_loss            — hard-negative prototype margin loss

Notes on the loss redesign
--------------------------
The previous varifocal_loss had a saddle at init: with sigmoid≈0.5 and the
positive-cell IoU floor clamped at 0.5, the weight `(gt - sigmoid)^γ * gt`
evaluates to 0 — positive cells contributed *nothing* to the gradient, so
the only force pushed predictions toward 0 (driven by negative cells), and
the conf head collapsed to "predict 0 everywhere". Standard binary focal loss
with hard 1/0 targets replaces it; positive cells now always receive a
non-zero pulling-up gradient.

The attention bbox auxiliary loss weight has been bumped 0.1 → 1.0 because
it sat dead-flat at ~1.0 KL through 10 stage-1 epochs with the previous
weight, meaning the support tokenizer never learned where the foreground was.

Distillation between saliency-pooled and ROI-pooled descriptors is removed —
the new model has no single "prototype vector" to distill into; the
SupportTokenizer extracts M region tokens directly.
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


def _containment_ratio(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Fraction of the GT box covered by the prediction: area(pred ∩ gt) / area(gt).

    Distinct from IoU: containment is asymmetric and penalises under-coverage
    only. A loose prediction that fully contains the GT scores 1.0; a tight
    prediction that only partially overlaps the GT scores < 1.0. Useful when
    "did we cover the object" matters more than "did we over-shoot".
    """
    inter_x1 = torch.maximum(pred[..., 0], gt[..., 0])
    inter_y1 = torch.maximum(pred[..., 1], gt[..., 1])
    inter_x2 = torch.minimum(pred[..., 2], gt[..., 2])
    inter_y2 = torch.minimum(pred[..., 3], gt[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    gt_area = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
    return inter / (gt_area + 1e-6)


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


def focal_loss(
    pred_logits: torch.Tensor,
    target_binary: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Standard binary focal loss (Lin et al. 2017) with hard 0/1 targets.

    alpha=0.75 up-weights positives (rare class — typically 1–10 cells out of
    196). Returns the *sum* over all elements; caller normalises by num_pos.
    """
    p = torch.sigmoid(pred_logits)
    pt = p * target_binary + (1 - p) * (1 - target_binary)
    focal_w = (1 - pt).clamp(min=1e-6).pow(gamma)
    alpha_w = alpha * target_binary + (1 - alpha) * (1 - target_binary)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target_binary, reduction="none"
    )
    return (alpha_w * focal_w * bce).sum()


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
    """Hard-negative prototype triplet loss."""
    B = protos.shape[0]
    if B < 2:
        return torch.zeros((), device=protos.device)
    normed = F.normalize(protos, dim=1)
    sims = normed @ normed.T
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
        neg_sim = sims[i][neg_mask].max()
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
    """KL divergence between aggregated support attention and a bbox target.

    Accepts:
      attn_map shape (B, K, M, H, W)  — new model: M region tokens per support.
                                        Aggregated by sum across M then renormalised.
      attn_map shape (B, K, H, W)     — legacy single-attention shape (kept for
                                        backwards compat).
    The resulting (B, K, H, W) probability map is matched against a uniform-
    inside-bbox target by KL.
    """
    if attn_map.dim() == 5:
        # Aggregate across M tokens — each row of attn already sums to 1, so the
        # sum has total mass M; divide it back to a probability distribution.
        agg = attn_map.sum(dim=2)
        agg = agg / agg.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    elif attn_map.dim() == 4:
        agg = attn_map
    else:
        raise ValueError(f"attn_map has unexpected shape {attn_map.shape}")

    b, k, h, w = agg.shape
    device = agg.device
    dtype = agg.dtype

    cy = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    cx = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride

    bb = support_bboxes.reshape(b * k, 4).to(dtype)
    x1 = bb[:, 0:1].unsqueeze(-1)
    y1 = bb[:, 1:2].unsqueeze(-1)
    x2 = bb[:, 2:3].unsqueeze(-1)
    y2 = bb[:, 3:4].unsqueeze(-1)

    inside = (
        (cx.view(1, 1, w) >= x1) & (cx.view(1, 1, w) <= x2)
        & (cy.view(1, h, 1) >= y1) & (cy.view(1, h, 1) <= y2)
    )
    target = inside.to(dtype)
    target_sum = target.sum(dim=(1, 2), keepdim=True)
    safe = target_sum > 0
    uniform = torch.full_like(target, 1.0 / (h * w))
    target = torch.where(safe, target / target_sum.clamp(min=1.0), uniform)
    target = target.view(b, k, h, w)

    eps = 1e-8
    kl = (target * (torch.log(target + eps) - torch.log(agg + eps))).sum(dim=(2, 3))
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
    attn_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Weighted sum of focal + box + presence + attention losses.

    Loss weights:
      focal:    1.0 (hard-target focal, sum normalised by num_pos)
      box:      1.0 (area-weighted GIoU on positive cells only)
      presence: 1.0 (BCE on (B,) presence_logit)
      attn:     1.0 (KL between aggregated support attention and bbox region) —
                bumped from 0.1 because the auxiliary signal is what teaches the
                SupportTokenizer where the foreground is. Without it the M
                region queries drift toward whatever minimises the dense detection
                loss, which is uninformative gradient through cross-attention.
    """
    reg_pred = pred["reg"]
    conf_logits = pred["conf"]

    conf_target, _, pos_mask = make_targets(gt_bbox, is_present, grid=grid, stride=stride)
    num_pos = pos_mask.sum().clamp(min=1)

    # Focal loss with hard binary targets — positive cells always receive a
    # gradient pulling sigmoid up, regardless of init.
    focal = focal_loss(conf_logits, conf_target) / num_pos

    # Area-weighted GIoU box loss (positive cells only)
    decoded = decode_pred_box(reg_pred, stride=stride)  # (B, 4, G, G)
    b = gt_bbox.shape[0]
    gt_exp = gt_bbox.view(b, 4, 1, 1).expand_as(decoded)
    if pos_mask.any():
        pred_pos = decoded.permute(0, 2, 3, 1)[pos_mask]   # (P, 4)
        gt_pos = gt_exp.permute(0, 2, 3, 1)[pos_mask]      # (P, 4)
        box_loss = area_weighted_giou_loss(pred_pos, gt_pos)
    else:
        box_loss = torch.zeros((), device=reg_pred.device)

    # Presence BCE
    if "presence_logit" in pred:
        presence_loss = F.binary_cross_entropy_with_logits(
            pred["presence_logit"], is_present.float()
        )
    else:
        presence_loss = torch.zeros((), device=reg_pred.device)

    # Aggregated support-attention bbox auxiliary loss
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
