"""FCOS-style target assignment + loss functions for few-shot localization.

Losses:
  quality_focal_loss      — IoU-as-soft-target focal (GFL family); replaces
                            hard-target focal so the conf head learns to score
                            *how good* a localisation is, not just whether it's
                            positive — direct mAP boost on dense detectors.
  area_weighted_ciou_loss — CIoU (center-distance + aspect-ratio terms) with
                            inverse-area weighting for small objects.
  focal_bce               — binary focal BCE used for presence (handles the
                            present/absent class imbalance smoothly).
  attention_bbox_loss     — KL between aggregated support attention and bbox region
  nt_xent_loss            — SupCon (per-shot) or uniformity (bag-level)
  vicreg_loss             — prototype variance + covariance regularisation
  barlow_twins_loss       — cross-correlation decorrelation between shot views;
                            orthogonal anti-collapse signal complementing VICReg.
  triplet_loss            — hard-negative prototype margin loss
  focal_loss              — kept for backwards compat (now unused in total_loss)
  area_weighted_giou_loss — kept for backwards compat (CIoU is the default)

Notes on the loss redesign
--------------------------
The previous varifocal_loss had a saddle at init: with sigmoid≈0.5 and the
positive-cell IoU floor clamped at 0.5, the weight `(gt - sigmoid)^γ * gt`
evaluates to 0 — positive cells contributed *nothing* to the gradient. Hard-
target binary focal fixed the saddle. The current Quality-Focal formulation
goes one step further: positive-cell targets are the actual `IoU(pred, gt)`
of the decoded box (detached), so confidence is calibrated to localisation
quality. NMS-ranked mAP improves directly. The saddle is gone because at
init both the target IoU is small AND sigmoid is 0.5, so the |target-p|^β
weight is non-zero and pulls positive cells up.

CIoU adds two terms over GIoU: a normalised centre-distance penalty and an
aspect-ratio penalty. Drop-in replacement; reliably +0.5–1.5 mAP on dense
detectors with no instability risk.

Barlow-Twins decorrelation on per-shot prototypes complements VICReg: VICReg
fights within-batch variance/covariance collapse on a single set of vectors;
Barlow operates on TWO views (shot 0 vs mean of remaining shots) and pushes
their cross-correlation matrix toward identity, attacking feature-dimension
redundancy directly. Both signals can run simultaneously; they're orthogonal.

Distillation between saliency-pooled and ROI-pooled descriptors is removed —
the new model has no single "prototype vector" to distill into; the
SupportTokenizer extracts M region tokens directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss, generalized_box_iou_loss


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

    Kept for backwards compat. Quality Focal Loss is the active conf-head loss.
    """
    p = torch.sigmoid(pred_logits)
    pt = p * target_binary + (1 - p) * (1 - target_binary)
    focal_w = (1 - pt).clamp(min=1e-6).pow(gamma)
    alpha_w = alpha * target_binary + (1 - alpha) * (1 - target_binary)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target_binary, reduction="none"
    )
    return (alpha_w * focal_w * bce).sum()


def quality_focal_loss(
    pred_logits: torch.Tensor,
    target_quality: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    """Quality-Focal Loss (GFL family) with continuous IoU-quality targets.

    Args:
        pred_logits:    raw logits, any shape
        target_quality: same shape, in [0, 1]. For dense detectors:
                          positive cells → IoU(decoded_pred_box, gt) (DETACHED)
                          negative cells → 0
        beta: focusing exponent on the |target - sigmoid(logit)| weight.

    Loss form (per element):
        |target - σ(logit)|^β · BCE(logit, target)

    Returns the *sum* over all elements; caller normalises by num_pos.
    Why this beats hard-target focal for mAP:
      The conf head is now trained to predict the IoU of its own decoded box.
      Cells with poorly-localised boxes get smaller scores, so NMS-ranked
      precision-recall sweeps place the well-localised boxes higher → mAP
      especially at the high-IoU end (0.75–0.95) goes up.
    """
    p = torch.sigmoid(pred_logits)
    weight = (target_quality - p).abs().clamp(min=1e-6).pow(beta)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target_quality, reduction="none"
    )
    return (weight * bce).sum()


def focal_bce(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sample-level binary focal cross-entropy (mean reduction).

    Used for the presence head: with NEG_PROB=0.3 the present/absent split
    is mildly imbalanced and many examples sit close to the boundary. The
    γ=2 focusing weight directs gradient onto hard examples (the ones with
    sigmoid ≈ 0.5), sharpening the present/absent margin. α=0.5 keeps the
    classes balanced — the dataset already controls the prior via NEG_PROB.
    """
    p = torch.sigmoid(pred_logits)
    pt = p * target + (1 - p) * (1 - target)
    focal_w = (1 - pt).clamp(min=1e-6).pow(gamma)
    alpha_w = alpha * target + (1 - alpha) * (1 - target)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    return (alpha_w * focal_w * bce).mean()


# ---------------------------------------------------------------------------
# Box regression loss
# ---------------------------------------------------------------------------


def area_weighted_giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """GIoU loss with inverse-area weighting (legacy; CIoU is the default)."""
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    norm_area = gt_area / (224.0 * 224.0)
    weights = 1.0 + 2.0 * torch.exp(-5.0 * norm_area)
    loss = generalized_box_iou_loss(pred, gt, reduction="none")
    return (loss * weights).mean()


def area_weighted_ciou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """CIoU loss with inverse-area weighting to up-weight small objects.

    CIoU = IoU − ρ²(centres)/c² − α·v(aspect ratio).  Drop-in replacement for
    GIoU — provides centre-distance and aspect-ratio gradients that GIoU lacks,
    which directly tightens box localisation and lifts mAP especially at the
    higher IoU thresholds (0.75–0.95).
    """
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    norm_area = gt_area / (224.0 * 224.0)
    weights = 1.0 + 2.0 * torch.exp(-5.0 * norm_area)
    loss = complete_box_iou_loss(pred, gt, reduction="none")
    return (loss * weights).mean()


# ---------------------------------------------------------------------------
# Prototype regularisation losses
# ---------------------------------------------------------------------------


def nt_xent_loss(prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent — uniformity (B,D) or supervised (B,K,D) variant.

    (B, D)   : pure uniformity — pushes all B prototypes apart. Same behaviour
               as the original implementation; one prototype per episode means
               B≈16 vectors per step, which is too few to learn from.
    (B, K, D): supervised contrastive — flattens to (B*K, D) and treats the K
               shots from the same episode as positives, all other shots as
               negatives. This is the right signal for the problem: "different
               views of the same instance should embed close, different
               instances should embed far". With B=16, K=5 we get B*K=80 anchors
               per step instead of 16 — a 5x signal density bump that directly
               addresses the K=5 plateau.
    """
    if prototypes.dim() == 3:
        b, k, d = prototypes.shape
        if b * k < 2 or k < 2:
            return torch.zeros((), device=prototypes.device)
        z = F.normalize(prototypes.reshape(b * k, d), dim=-1)
        sim = z @ z.T / temperature                                     # (BK, BK)
        eye = torch.eye(b * k, dtype=torch.bool, device=z.device)
        # Use a large negative (not -inf) to avoid 0*-inf=NaN downstream when
        # pos_mask zeroes out the diagonal entries of log_prob.
        sim = sim.masked_fill(eye, -1e9)
        # positive mask: same episode, not self
        ep_idx = torch.arange(b, device=z.device).repeat_interleave(k)
        pos_mask = (ep_idx.unsqueeze(0) == ep_idx.unsqueeze(1)) & ~eye  # (BK, BK)
        # SupCon: for each anchor, log-sum-exp over all, minus mean log-prob of positives.
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
        return loss.mean()
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
    """VICReg variance + covariance terms.

    Accepts (B, D) or (B, K, D). For the 3D case the per-shot prototypes are
    flattened across the B*K axis so VICReg fights collapse over a much larger
    sample (80 vs 16 with B=16, K=5), and crucially the *within-episode*
    variance contributes to the variance term — directly penalising the
    failure mode where the K shots produce near-identical summaries.
    """
    if prototypes.dim() == 3:
        b, k, d = prototypes.shape
        prototypes = prototypes.reshape(b * k, d)
    B, D = prototypes.shape
    if B < 2:
        return torch.zeros((), device=prototypes.device)
    z = prototypes - prototypes.mean(dim=0)
    var_loss = F.relu(gamma - torch.sqrt(z.var(dim=0) + eps)).mean()
    cov = (z.T @ z) / (B - 1)
    cov_loss = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / D
    return var_loss + 0.04 * cov_loss


def barlow_twins_loss(
    per_shot_prototypes: torch.Tensor,
    lambda_off: float = 0.005,
) -> torch.Tensor:
    """Cross-correlation decorrelation between two views of the same episode.

    Args:
        per_shot_prototypes: (B, K, D) — K shot summaries per episode.
        lambda_off: weight on the off-diagonal redundancy term.

    Method:
        View 1 = shot 0 of each episode; View 2 = mean of shots 1..K-1.
        Standardise each view per dimension across the batch, then compute
        the (D, D) cross-correlation matrix C between views. Push diagonal →
        1 (invariance: the two views of the same episode should agree on each
        feature) and off-diagonal → 0 (redundancy reduction: features should
        encode independent information).

    Why this is orthogonal to VICReg:
        VICReg's covariance term decorrelates *within a single batch of
        prototypes*. Barlow Twins decorrelates *between two paired views*,
        which is a strictly stronger anti-collapse signal because it requires
        the model to encode invariant-but-decorrelated features per pair.
        Both run cheaply; combining them is the standard recipe in modern
        SSL pipelines.

    Returns 0 if K < 2 or B < 2 (cannot form pairs / standardise).
    """
    if per_shot_prototypes.dim() != 3:
        raise ValueError(
            f"barlow_twins_loss expects (B, K, D); got {per_shot_prototypes.shape}"
        )
    b, k, d = per_shot_prototypes.shape
    if k < 2 or b < 2:
        return torch.zeros((), device=per_shot_prototypes.device)

    z1 = per_shot_prototypes[:, 0]                       # (B, D)
    z2 = per_shot_prototypes[:, 1:].mean(dim=1)          # (B, D)

    # Standardise per dimension across the batch (mean 0, std 1)
    z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
    z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)

    c = (z1.T @ z2) / b                                  # (D, D)
    on_diag = (c.diagonal() - 1.0).pow(2).sum()
    off_diag = c.pow(2).sum() - c.diagonal().pow(2).sum()
    return (on_diag + lambda_off * off_diag) / d


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
    attn_loss_weight: float = 0.5,
    presence_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Weighted sum of QFL (cls) + CIoU (box) + focal-BCE (presence) + attn.

    Loss weights:
      qfl:      1.0  (Quality Focal Loss; sum normalised by num_pos)
      box:      1.0  (area-weighted CIoU on positive cells only)
      presence: presence_weight  (focal BCE on (B,) presence_logit; default 1.0)
      attn:     attn_loss_weight (caller; default 0.5 for Phase 1, 1.0 for Phase 2)

    The presence_weight knob exists because the presence head's raw focal-BCE
    typically lands around 0.03–0.10 when training is going well — an order
    of magnitude smaller than qfl + box. Without up-weighting, the presence
    head receives almost no gradient pressure relative to the localizer,
    which on hard target-domain transfers (HOTS scenes, InsDet products)
    manifests as the head defaulting to "always present" (mean_score_neg
    drifting up to 0.7+). Set presence_weight=2-3x in Phase 2 stages to
    rebalance.

    Returned dict still uses the key "focal" for the cls term so existing
    plotting/logging continues to work — but the value is now the QFL.
    """
    reg_pred = pred["reg"]
    conf_logits = pred["conf"]

    _, _, pos_mask = make_targets(gt_bbox, is_present, grid=grid, stride=stride)
    num_pos = pos_mask.sum().clamp(min=1)

    # Decode pred boxes once — used by both the box loss and the QFL targets.
    decoded = decode_pred_box(reg_pred, stride=stride)              # (B, 4, G, G)
    b = gt_bbox.shape[0]
    gt_exp = gt_bbox.view(b, 4, 1, 1).expand_as(decoded)

    # QFL: per-cell IoU(pred, gt) on positives, 0 elsewhere. Detached so the
    # target doesn't backprop through the regression branch (paper recipe).
    pred_xyxy = decoded.permute(0, 2, 3, 1)                         # (B, G, G, 4)
    gt_xyxy = gt_exp.permute(0, 2, 3, 1)                            # (B, G, G, 4)
    cell_iou = _iou_xyxy(pred_xyxy, gt_xyxy).clamp(0.0, 1.0)        # (B, G, G)
    quality_target = (cell_iou * pos_mask.float()).detach().unsqueeze(1)  # (B,1,G,G)
    qfl = quality_focal_loss(conf_logits, quality_target) / num_pos

    # Area-weighted CIoU box loss on positive cells
    if pos_mask.any():
        pred_pos = decoded.permute(0, 2, 3, 1)[pos_mask]            # (P, 4)
        gt_pos = gt_exp.permute(0, 2, 3, 1)[pos_mask]               # (P, 4)
        box_loss = area_weighted_ciou_loss(pred_pos, gt_pos)
    else:
        box_loss = torch.zeros((), device=reg_pred.device)

    # Presence: focal BCE for sharper present/absent margin (combined with
    # the [ABSENT] support token and IoU-aware conf, this gives a much cleaner
    # "object in scene" decision than plain BCE).
    if "presence_logit" in pred:
        presence_loss = focal_bce(
            pred["presence_logit"], is_present.float()
        )
    else:
        presence_loss = torch.zeros((), device=reg_pred.device)

    # Aggregated support-attention bbox auxiliary loss
    attn_loss = torch.zeros((), device=reg_pred.device)
    support_attn = pred.get("support_attn")
    if support_attn is not None and support_bboxes is not None:
        attn_loss = attention_bbox_loss(support_attn, support_bboxes)

    loss = qfl + box_loss + presence_weight * presence_loss + attn_loss_weight * attn_loss
    return {
        "loss": loss,
        "focal": qfl.detach(),     # legacy key — value is now QFL
        "box": box_loss.detach(),
        "presence": presence_loss.detach(),
        "attn": attn_loss.detach(),
    }
