"""Loss stack for the few-shot Siamese localiser.

Per-cell on positives (TaskAlignedAssigner-selected cells):
    - Quality Focal Loss with target = IoU(pred, gt) (detached)
    - Distribution Focal Loss + GIoU on the (l, t, r, b) DFL distributions
    - Centerness BCE with FCOS centerness target

Per-cell on negatives:
    - neg_qfl: focal BCE with target=0 over ~pos_mask cells, weight 0.5

Per-image:
    - Class-balanced presence focal-BCE: 0.5*pos_term + 0.5*neg_term (γ=2.5)

Aux head: total_loss applied to layer-0 decoder output, weight 0.5.

Anti-collapse regularisers:
    - Conf-map entropy (sign-flipped: penalise high entropy on absent episodes,
      penalise low entropy on multi-positive present episodes).
    - Off-positive `reg` L2 prior on the integrated reg expectation.
    - Prototype L2-norm prior (||prototype|| → 1).

Auxiliary supervision:
    - Attention-bbox KL loss.
    - SupCon NT-Xent on per-shot prototypes.
    - VICReg variance + covariance.
    - Barlow Twins on shot-paired views.
    - Optional triplet on (B, dim) prototypes.

This module exports:
    total_loss(...)        — combines all dense + presence + aux head terms.
    nt_xent_loss / vicreg_loss / barlow_twins_loss / triplet_loss
    attention_bbox_loss
    proto_norm_loss
    _iou_xyxy / _containment_ratio    (used by evaluate.py)
    task_aligned_assign(...)          (TOOD positive assignment)
    dfl_target / dfl_loss             (helpers for DFL)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from modeling.model import (
    DFL_BINS,
    GRID_P3,
    GRID_P4,
    STRIDE_P3,
    STRIDE_P4,
    decode_ltrb_to_xyxy,
    dfl_expectation,
)


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-6)


def _containment_ratio(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(pred[..., 0], gt[..., 0])
    inter_y1 = torch.maximum(pred[..., 1], gt[..., 1])
    inter_x2 = torch.minimum(pred[..., 2], gt[..., 2])
    inter_y2 = torch.minimum(pred[..., 3], gt[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    gt_area = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
    return inter / (gt_area + 1e-6)


# ---------------------------------------------------------------------------
# DFL helpers
# ---------------------------------------------------------------------------


def dfl_target(
    ltrb: torch.Tensor, dfl_bins: int = DFL_BINS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce two-bin DFL targets (left, right indices and weights).

    For a continuous target value t in [0, dfl_bins-1], the DFL paper places
    weight (right-t) on bin floor(t) and weight (t-left) on bin floor(t)+1.

    Returns:
        idx_l : (..., 4) long indices of the lower bin
        idx_r : (..., 4) long indices of the upper bin
        wl    : (..., 4) lower-bin weight (1 - frac)
        wr    : (..., 4) upper-bin weight (= frac)
    """
    t = ltrb.clamp(min=0.0, max=float(dfl_bins - 1) - 1e-3)
    idx_l = t.floor().long()
    idx_r = (idx_l + 1).clamp(max=dfl_bins - 1)
    wr = (t - idx_l.to(t.dtype))
    wl = 1.0 - wr
    return idx_l, idx_r, wl, wr


def dfl_loss(
    reg_logits_pos: torch.Tensor,
    ltrb_target: torch.Tensor,
    dfl_bins: int = DFL_BINS,
) -> torch.Tensor:
    """Distribution Focal Loss on positive cells.

    reg_logits_pos: (P, 4*dfl_bins) — DFL bin logits at positive cells (P=#pos).
    ltrb_target:    (P, 4) — continuous (l, t, r, b) targets in stride units.
    """
    if reg_logits_pos.numel() == 0:
        return reg_logits_pos.new_zeros(())
    p = reg_logits_pos.shape[0]
    logits = reg_logits_pos.view(p, 4, dfl_bins)
    log_p = F.log_softmax(logits, dim=-1)
    idx_l, idx_r, wl, wr = dfl_target(ltrb_target, dfl_bins=dfl_bins)
    # Gather per-coord log-probs at the two bins.
    nll_l = -log_p.gather(-1, idx_l.unsqueeze(-1)).squeeze(-1)               # (P, 4)
    nll_r = -log_p.gather(-1, idx_r.unsqueeze(-1)).squeeze(-1)               # (P, 4)
    return (wl * nll_l + wr * nll_r).mean()


# ---------------------------------------------------------------------------
# TaskAlignedAssigner (TOOD) — conf-aware positive assignment
# ---------------------------------------------------------------------------


def _grid_centres(grid_h: int, grid_w: int, stride: int, device, dtype):
    j = torch.arange(grid_w, device=device, dtype=dtype)
    i = torch.arange(grid_h, device=device, dtype=dtype)
    cy = (i + 0.5) * stride
    cx = (j + 0.5) * stride
    return cx, cy


def task_aligned_assign(
    decoded_boxes: torch.Tensor,
    conf_logits: torch.Tensor,
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    stride: int,
    grid_h: int,
    grid_w: int,
    alpha: float = 1.0,
    beta: float = 6.0,
    center_radius: float = 1.5,
    topk_iou: int = 10,
    iou_floor: float = 0.05,
    cold_start_iou_thr: float = 0.10,
    cold_start_topq: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TOOD positive assignment with a center-sampling fallback for cold-start.

    For each present-episode GT box:
      1) Restrict candidate cells to those whose centre lies within `center_radius * stride`
         of the GT centre AND inside the GT bbox.
      2) If the *best* IoU across candidates is below ``cold_start_iou_thr``,
         the regressor hasn't started learning. TOOD's alignment metric
         ``σ(conf)^α · IoU^β`` is uninformative in that regime — IoU^6 with
         IoU≈0 collapses to numerical zero and the assigner picks effectively
         random cells. Fall back to plain center-sampling: take the
         ``cold_start_topq`` candidates closest to the GT centre.
      3) Otherwise: compute alignment metric ``t = σ(conf)^α · (IoU+ε)^β``
         over candidates, pick top-q cells by t, where q = round(sum of
         top-``topk_iou`` IoU values), clamped to [1, len(candidates)].

    The ε floor (``iou_floor``) keeps the alignment score well-conditioned
    even when IoU is small but non-zero — without it β=6 makes the score
    range vary by 12+ orders of magnitude across cells.

    Args:
        decoded_boxes : (B, 4, H, W) decoded xyxy boxes in image coords.
        conf_logits   : (B, 1, H, W).
        gt_bbox       : (B, 4) xyxy in image coords.
        is_present    : (B,) bool.

    Returns:
        pos_mask        : (B, H, W) bool — assigned positives.
        cell_iou        : (B, H, W) IoU(decoded, gt). Zero on absent episodes.
        ltrb_target     : (B, 4, H, W) continuous (l, t, r, b) targets in stride units
                          (only meaningful where pos_mask is True).
    """
    B = decoded_boxes.shape[0]
    device = decoded_boxes.device
    dtype = decoded_boxes.dtype

    cx, cy = _grid_centres(grid_h, grid_w, stride, device, dtype)            # (W,), (H,)
    cy_grid = cy.view(1, grid_h, 1).expand(B, grid_h, grid_w)
    cx_grid = cx.view(1, 1, grid_w).expand(B, grid_h, grid_w)

    x1 = gt_bbox[:, 0].view(B, 1, 1)
    y1 = gt_bbox[:, 1].view(B, 1, 1)
    x2 = gt_bbox[:, 2].view(B, 1, 1)
    y2 = gt_bbox[:, 3].view(B, 1, 1)
    gx = (x1 + x2) * 0.5
    gy = (y1 + y2) * 0.5

    inside_box = (cx_grid >= x1) & (cx_grid <= x2) & (cy_grid >= y1) & (cy_grid <= y2)
    near_centre = (
        (cx_grid - gx).abs() <= center_radius * stride
    ) & (
        (cy_grid - gy).abs() <= center_radius * stride
    )
    candidate = inside_box & near_centre & is_present.view(B, 1, 1)

    # Continuous (l, t, r, b) targets in stride units (for DFL on positives).
    l = (cx_grid - x1) / stride
    t = (cy_grid - y1) / stride
    r = (x2 - cx_grid) / stride
    b_ = (y2 - cy_grid) / stride
    ltrb_target = torch.stack([l, t, r, b_], dim=1).clamp(min=0.0)

    # Per-cell IoU(pred, gt).
    pred_xyxy = decoded_boxes.permute(0, 2, 3, 1)                            # (B, H, W, 4)
    gt_xyxy = gt_bbox.view(B, 1, 1, 4).expand_as(pred_xyxy)
    cell_iou = _iou_xyxy(pred_xyxy, gt_xyxy).clamp(min=0.0, max=1.0)
    # Mask iou to candidates (so non-candidates don't get picked by topk).
    cell_iou_masked = torch.where(candidate, cell_iou, cell_iou.new_zeros(()))

    conf_prob = torch.sigmoid(conf_logits.squeeze(1))                         # (B, H, W)
    # ε-floored alignment metric. IoU+ε keeps the score well-conditioned in
    # the cold-start regime where every IoU is ~0.
    iou_term = (cell_iou_masked + iou_floor) ** beta
    align = (conf_prob.detach() ** alpha) * iou_term
    align = torch.where(candidate, align, align.new_zeros(()))

    pos_mask = torch.zeros_like(candidate, dtype=torch.bool)

    # Centre-distance score (cells closer to GT centre score higher) — used
    # for the cold-start fallback. Negative L1 distance so torch.topk picks
    # the closest candidates.
    dx = (cx_grid - gx).abs()
    dy = (cy_grid - gy).abs()
    centre_score = -(dx + dy)
    centre_score = torch.where(candidate, centre_score, centre_score.new_full((), float("-inf")))

    HW = grid_h * grid_w
    for bi in range(B):
        if not bool(is_present[bi].item()):
            continue
        cand = candidate[bi].view(-1)
        if cand.sum() == 0:
            # Fallback: force the cell at the GT centre to be positive.
            j_c = (gx[bi].item() / stride)
            i_c = (gy[bi].item() / stride)
            j_c = int(min(max(j_c, 0), grid_w - 1))
            i_c = int(min(max(i_c, 0), grid_h - 1))
            pos_mask[bi, i_c, j_c] = True
            continue

        flat_iou = cell_iou_masked[bi].view(-1)
        n_cand = int(cand.sum().item())

        best_iou = float(flat_iou.max().item())
        if best_iou < cold_start_iou_thr:
            # Cold-start path: pure center-sampling. Take up to
            # ``cold_start_topq`` cells closest to the GT centre.
            q = max(1, min(cold_start_topq, n_cand, HW))
            flat_centre = centre_score[bi].view(-1)
            topq = torch.topk(flat_centre, k=q).indices
        else:
            flat_align = align[bi].view(-1)
            topk = min(topk_iou, n_cand)
            top_iou = torch.topk(flat_iou, k=topk).values
            q = int(top_iou.sum().clamp(min=1.0).round().item())
            q = max(1, min(q, n_cand, HW))
            topq = torch.topk(flat_align, k=q).indices

        pos_flat = pos_mask[bi].view(-1).clone()
        pos_flat[topq] = True
        pos_mask[bi] = pos_flat.view(grid_h, grid_w)

    cell_iou = torch.where(is_present.view(B, 1, 1), cell_iou, cell_iou.new_zeros(()))
    return pos_mask, cell_iou, ltrb_target


# ---------------------------------------------------------------------------
# Quality Focal Loss / negative-cell QFL
# ---------------------------------------------------------------------------


def quality_focal_loss(
    pred_logits: torch.Tensor,
    target_quality: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    p = torch.sigmoid(pred_logits)
    weight = (target_quality - p).abs().clamp(min=1e-6).pow(beta)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_quality, reduction="none")
    return (weight * bce).sum()


def negative_qfl(
    pred_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE against target=0 on negative (non-positive) cells.

    Gives the conf head explicit gradient pressure on absent episodes and
    out-of-box cells of present episodes — without it, conf saturates uniformly
    high and AP ranking collapses.
    """
    neg = ~pos_mask                                                           # (B, H, W)
    if neg.sum() == 0:
        return pred_logits.new_zeros(())
    logits = pred_logits.squeeze(1)[neg]
    target = torch.zeros_like(logits)
    p = torch.sigmoid(logits)
    focal_w = p.clamp(min=1e-6).pow(gamma)                                    # (1 - target)^... = p^γ
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (focal_w * bce).mean()


# ---------------------------------------------------------------------------
# Centerness target + BCE
# ---------------------------------------------------------------------------


def centerness_target(ltrb: torch.Tensor) -> torch.Tensor:
    """FCOS centerness target = sqrt(min(l,r)/max(l,r) · min(t,b)/max(t,b)).

    ltrb: (..., 4)  in stride units (any unit; ratio is scale-free).
    """
    l, t, r, b = ltrb.unbind(-1)
    lr = torch.minimum(l, r) / (torch.maximum(l, r).clamp(min=1e-6))
    tb = torch.minimum(t, b) / (torch.maximum(t, b).clamp(min=1e-6))
    return torch.sqrt((lr * tb).clamp(min=0.0))


# ---------------------------------------------------------------------------
# Class-balanced presence focal-BCE
# ---------------------------------------------------------------------------


def class_balanced_focal_bce(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.5,
) -> torch.Tensor:
    """Sample-level focal BCE with explicit pos/neg class balancing.

    pred_logits: (B,)  raw logits.
    target:      (B,)  in {0, 1}.

    loss = 0.5 * (sum of focal-BCE on present) / num_present
         + 0.5 * (sum of focal-BCE on absent ) / num_absent

    If a class is empty in the batch, the contribution is dropped (loss term
    is half-empty and takes the value of the other class only). This makes
    the loss magnitude invariant to NEG_PROB and immune to the "always
    present" bias.
    """
    p = torch.sigmoid(pred_logits)
    pt = p * target + (1 - p) * (1 - target)
    focal_w = (1 - pt).clamp(min=1e-6).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    raw = focal_w * bce
    pos_mask = target > 0.5
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum().clamp(min=1).to(raw.dtype)
    n_neg = neg_mask.sum().clamp(min=1).to(raw.dtype)
    pos_term = (raw * pos_mask.to(raw.dtype)).sum() / n_pos
    neg_term = (raw * neg_mask.to(raw.dtype)).sum() / n_neg
    return 0.5 * pos_term + 0.5 * neg_term


# ---------------------------------------------------------------------------
# Anti-collapse regularisers
# ---------------------------------------------------------------------------


def conf_entropy_reg(
    conf_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    is_present: torch.Tensor,
) -> torch.Tensor:
    """Sign-flipped entropy regulariser on the conf map.

    For absent episodes: penalise high entropy (we want a sharp 'nothing' signal,
    i.e. low entropy is fine; high entropy = uniformly hot = collapse).
    For present episodes with multiple positives: penalise low entropy (we want
    a spread of hot cells covering the object, not a single spike).
    """
    B, _, H, W = conf_logits.shape
    p = torch.sigmoid(conf_logits.view(B, -1))
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-6)                        # normalise
    ent = -(p * (p.clamp(min=1e-8)).log()).sum(dim=1)                          # (B,)
    # Absent episodes → +entropy (penalise high entropy).
    absent = ~is_present                                                       # (B,)
    abs_term = (ent * absent.to(ent.dtype)).sum() / absent.sum().clamp(min=1).to(ent.dtype)
    # Multi-pos present episodes → -entropy (penalise low entropy).
    n_pos = pos_mask.view(B, -1).sum(dim=1)
    multi_pos = is_present & (n_pos >= 2)
    if multi_pos.any():
        mp_term = -(ent * multi_pos.to(ent.dtype)).sum() / multi_pos.sum().clamp(min=1).to(
            ent.dtype
        )
    else:
        mp_term = ent.new_zeros(())
    return abs_term + mp_term


def reg_l2_prior_off_pos(
    reg_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    dfl_bins: int = DFL_BINS,
) -> torch.Tensor:
    """L2 prior on the integrated reg expectation at non-positive cells.

    Without this, the regression head can learn arbitrary outputs at negative
    cells (the loss never sees them) and the global decode statistics drift —
    the well-known 'corner collapse' failure where every absent prediction
    lands at the same image corner.

    Loss = mean_{cells outside pos_mask} ||ltrb_pred||^2 (on integrated
    expectations in stride units). Tiny coefficient is enough.
    """
    ltrb = dfl_expectation(reg_logits, dfl_bins=dfl_bins)                     # (B, 4, H, W)
    neg = ~pos_mask                                                            # (B, H, W)
    if neg.sum() == 0:
        return reg_logits.new_zeros(())
    sq = (ltrb ** 2).sum(dim=1)                                                # (B, H, W)
    return (sq * neg.to(sq.dtype)).sum() / neg.sum().clamp(min=1).to(sq.dtype)


def proto_norm_loss(prototype: torch.Tensor) -> torch.Tensor:
    """(||prototype||_2 - 1)^2 — keeps prototype on the unit sphere.

    Prevents the silent collapse where ||prototype|| shrinks to 0 to reduce
    pairwise distances in NT-Xent / hard-neg miner.
    """
    if prototype.numel() == 0:
        return prototype.new_zeros(())
    norms = prototype.norm(dim=-1)
    return ((norms - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Prototype regularisation losses (NT-Xent / VICReg / Barlow / triplet)
# ---------------------------------------------------------------------------


def nt_xent_loss(prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if prototypes.dim() == 3:
        b, k, d = prototypes.shape
        if b * k < 2 or k < 2:
            return torch.zeros((), device=prototypes.device)
        z = F.normalize(prototypes.reshape(b * k, d), dim=-1)
        sim = z @ z.T / temperature
        eye = torch.eye(b * k, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, -1e9)
        ep_idx = torch.arange(b, device=z.device).repeat_interleave(k)
        pos_mask = (ep_idx.unsqueeze(0) == ep_idx.unsqueeze(1)) & ~eye
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
    prototypes: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
) -> torch.Tensor:
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
    per_shot_prototypes: torch.Tensor, lambda_off: float = 0.005
) -> torch.Tensor:
    if per_shot_prototypes.dim() != 3:
        raise ValueError(f"barlow_twins_loss expects (B, K, D); got {per_shot_prototypes.shape}")
    b, k, d = per_shot_prototypes.shape
    if k < 2 or b < 2:
        return torch.zeros((), device=per_shot_prototypes.device)
    z1 = per_shot_prototypes[:, 0]
    z2 = per_shot_prototypes[:, 1:].mean(dim=1)
    z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
    z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)
    c = (z1.T @ z2) / b
    on_diag = (c.diagonal() - 1.0).pow(2).sum()
    off_diag = c.pow(2).sum() - c.diagonal().pow(2).sum()
    return (on_diag + lambda_off * off_diag) / d


def triplet_loss(
    protos: torch.Tensor, instance_ids: list[str], margin: float = 0.3
) -> torch.Tensor:
    B = protos.shape[0]
    if B < 2:
        return torch.zeros((), device=protos.device)
    normed = F.normalize(protos, dim=1)
    sims = normed @ normed.T
    loss = torch.zeros((), device=protos.device)
    count = 0
    for i, iid in enumerate(instance_ids):
        pos_mask = torch.tensor(
            [j != i and instance_ids[j] == iid for j in range(B)], device=protos.device
        )
        neg_mask = torch.tensor(
            [instance_ids[j] != iid for j in range(B)], device=protos.device
        )
        if not pos_mask.any() or not neg_mask.any():
            continue
        pos_sim = sims[i][pos_mask].max()
        neg_sim = sims[i][neg_mask].max()
        loss = loss + F.relu(neg_sim - pos_sim + margin)
        count += 1
    return loss / max(count, 1)


# ---------------------------------------------------------------------------
# Attention bbox loss
# ---------------------------------------------------------------------------


def attention_bbox_loss(
    attn_map: torch.Tensor,
    support_bboxes: torch.Tensor,
    img_size: int = 224,
    stride: int | None = None,
) -> torch.Tensor:
    if attn_map.dim() == 5:
        agg = attn_map.sum(dim=2)
        agg = agg / agg.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    elif attn_map.dim() == 4:
        agg = attn_map
    else:
        raise ValueError(f"attn_map has unexpected shape {attn_map.shape}")

    b, k, h, w = agg.shape
    device = agg.device
    dtype = agg.dtype
    # Derive stride from img_size / h so the function works at any input size
    # (multi-scale training: 192 → P5 6×6, 224 → 7×7, 256 → 8×8).
    if stride is None:
        stride = img_size / float(h)
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
# Discrimination + anti-collapse losses (Phase 0 last-resort)
# ---------------------------------------------------------------------------


def presence_aware_contrastive(
    enriched_query: torch.Tensor,
    prototype: torch.Tensor,
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    img_size: int = 224,
    margin: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE between query features and support prototype, presence-aware.

    For PRESENT episodes:
      - positive query feature = mean over enriched_query cells inside the GT bbox.
      - target: positive_query_feat ↔ prototype should be close (cosine sim ~ 1).
      - negatives: every OTHER episode's prototype in the batch.

    For ABSENT episodes:
      - "anti-positive": GAP(enriched_query) should be FAR from prototype
        (cosine sim ≤ -margin or at least small).
      - This directly attacks ``presence_acc_neg ≈ 0.5``: the only way the
        presence head can distinguish absent from present is if the
        feature space already does.

    Args:
        enriched_query: (B, D, H, W) cross-attention output (the query feature
            the detection head sees).
        prototype:      (B, D) bag-level support summary.
        gt_bbox:        (B, 4) xyxy in image coords.
        is_present:     (B,) bool.

    Returns scalar loss.
    """
    B, D, H, W = enriched_query.shape
    device = enriched_query.device
    dtype = enriched_query.dtype

    # Pool query feature inside GT bbox for present episodes (else GAP).
    stride = img_size / float(H)
    cy = (torch.arange(H, device=device, dtype=dtype) + 0.5) * stride
    cx = (torch.arange(W, device=device, dtype=dtype) + 0.5) * stride
    cy_grid = cy.view(1, H, 1).expand(B, H, W)
    cx_grid = cx.view(1, 1, W).expand(B, H, W)

    x1 = gt_bbox[:, 0].view(B, 1, 1)
    y1 = gt_bbox[:, 1].view(B, 1, 1)
    x2 = gt_bbox[:, 2].view(B, 1, 1)
    y2 = gt_bbox[:, 3].view(B, 1, 1)
    inside = (cx_grid >= x1) & (cx_grid <= x2) & (cy_grid >= y1) & (cy_grid <= y2)
    inside_f = inside.to(dtype)

    # GAP fallback for absent / empty inside-mask.
    gap = enriched_query.mean(dim=(2, 3))                                     # (B, D)
    inside_sum = inside_f.sum(dim=(1, 2)).clamp(min=1.0)
    inside_mean = (
        enriched_query * inside_f.unsqueeze(1)
    ).sum(dim=(2, 3)) / inside_sum.unsqueeze(1)                               # (B, D)
    use_inside = is_present & (inside.flatten(1).sum(dim=1) > 0)
    q_feat = torch.where(use_inside.view(B, 1), inside_mean, gap)             # (B, D)

    # Cosine sims.
    q_n = F.normalize(q_feat, dim=-1)
    p_n = F.normalize(prototype, dim=-1)
    sim = (q_n * p_n).sum(dim=-1)                                             # (B,)

    # InfoNCE for present: positive = same-episode prototype, negatives = all others.
    pos_mask = is_present
    if pos_mask.any():
        # Diag = positive sim. Off-diag = negative sims.
        sim_matrix = q_n @ p_n.t() / temperature                              # (B, B)
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        pos_diag = log_prob.diagonal()                                        # (B,)
        nce_loss = -(pos_diag * pos_mask.to(pos_diag.dtype)).sum() / pos_mask.sum().clamp(min=1).to(pos_diag.dtype)
    else:
        nce_loss = sim.new_zeros(())

    # Absent: hinge-style penalty if cosine sim > -margin.
    abs_mask = ~is_present
    if abs_mask.any():
        abs_term = F.relu(sim + margin) * abs_mask.to(sim.dtype)
        abs_loss = abs_term.sum() / abs_mask.sum().clamp(min=1).to(sim.dtype)
    else:
        abs_loss = sim.new_zeros(())

    return nce_loss + abs_loss


def feature_spread_reg(enriched_query: torch.Tensor) -> torch.Tensor:
    """VICReg-variance-style hinge on the spatial std of the cross-attention output.

    The conf-map flatness diagnosis (``conf_map_std_pos ≈ 0.03``) traces back
    to the cross-attention producing near-uniform spatial features. This term
    penalises low spatial std per channel, encouraging the decoder to produce
    discriminative locations.

    Loss = ReLU(γ - sqrt(spatial_var + ε)).mean()  per channel, then averaged.
    """
    B, D, H, W = enriched_query.shape
    if H * W < 4:
        return enriched_query.new_zeros(())
    flat = enriched_query.flatten(2)                                          # (B, D, HW)
    var = flat.var(dim=2)                                                     # (B, D)
    std = (var + 1e-4).sqrt()
    target = 1.0
    return F.relu(target - std).mean()


def hardness_weighted_focal_bce(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.5,
    base_alpha: float = 0.5,
    hardness_scale: float = 1.0,
) -> torch.Tensor:
    """Class-balanced focal BCE that boosts the *currently-misclassified* class.

    Computes per-class accuracy on this batch; boosts α for the worse class.
    Acts as an automatic re-balancer when the presence head is biased toward
    one direction (which is the failure we keep seeing).
    """
    with torch.no_grad():
        p = torch.sigmoid(pred_logits)
        pred_pos = p > 0.5
        gt_pos = target > 0.5
        gt_neg = ~gt_pos
        n_pos = gt_pos.sum().clamp(min=1).to(pred_logits.dtype)
        n_neg = gt_neg.sum().clamp(min=1).to(pred_logits.dtype)
        acc_pos = (pred_pos & gt_pos).sum().to(pred_logits.dtype) / n_pos
        acc_neg = ((~pred_pos) & gt_neg).sum().to(pred_logits.dtype) / n_neg
        # Worse-class boost: α scales toward the class with lower accuracy.
        # If acc_pos < acc_neg, α_pos > 0.5 (favour pos gradient); vice versa.
        diff = (acc_neg - acc_pos).clamp(-1.0, 1.0)                           # >0 ⇒ pos worse
        alpha_pos = (base_alpha + hardness_scale * diff * 0.25).clamp(0.1, 0.9)
        alpha_neg = 1.0 - alpha_pos

    p = torch.sigmoid(pred_logits)
    pt = p * target + (1 - p) * (1 - target)
    focal_w = (1 - pt).clamp(min=1e-6).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    raw = focal_w * bce
    pos_term = (raw * (target > 0.5).to(raw.dtype)).sum() / n_pos
    neg_term = (raw * (target <= 0.5).to(raw.dtype)).sum() / n_neg
    return alpha_pos * pos_term + alpha_neg * neg_term


# ---------------------------------------------------------------------------
# Per-scale dense loss
# ---------------------------------------------------------------------------


def _dense_loss_one_scale(
    reg_logits: torch.Tensor,
    conf_logits: torch.Tensor,
    ctr_logits: torch.Tensor,
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    stride: int,
    grid_h: int,
    grid_w: int,
    pred_iou_logits: torch.Tensor | None = None,
    qfl_beta: float = 2.0,
    neg_qfl_weight: float = 0.5,
    centerness_weight: float = 1.0,
    dfl_weight: float = 0.25,
    giou_weight: float = 2.0,
    pred_iou_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute QFL + neg_qfl + centerness + DFL + GIoU + pred_iou on one FPN scale.

    ``pred_iou_logits`` is the IoU-Aware FCOS / VarifocalNet ranking head:
    a per-cell sigmoid that's trained with L1 loss against the actual
    ``IoU(decoded_pred, gt).detach()`` on positive cells. Multiplied into
    the inference score for an extra independent ranking signal.
    """
    B = conf_logits.shape[0]
    device = conf_logits.device

    # Decode current predictions for assignment (use detached for assignment).
    with torch.no_grad():
        ltrb_pred = dfl_expectation(reg_logits, dfl_bins=DFL_BINS)
        decoded_xyxy = decode_ltrb_to_xyxy(ltrb_pred, stride=stride)

        pos_mask, cell_iou, ltrb_target = task_aligned_assign(
            decoded_xyxy,
            conf_logits,
            gt_bbox,
            is_present,
            stride=stride,
            grid_h=grid_h,
            grid_w=grid_w,
        )

    num_pos = pos_mask.sum().clamp(min=1).to(conf_logits.dtype)

    # QFL on positives (target = cell_iou) and 0 elsewhere.
    qfl_target = (cell_iou * pos_mask.to(cell_iou.dtype)).unsqueeze(1)         # (B, 1, H, W)
    qfl = quality_focal_loss(conf_logits, qfl_target, beta=qfl_beta) / num_pos

    # Negative-cell focal BCE (target=0).
    neg_q = negative_qfl(conf_logits, pos_mask) * neg_qfl_weight

    # Centerness BCE on positives only.
    if pos_mask.any():
        ltrb_target_pos = ltrb_target.permute(0, 2, 3, 1)[pos_mask]            # (P, 4)
        ctr_t = centerness_target(ltrb_target_pos)
        ctr_p = ctr_logits.squeeze(1)[pos_mask]
        ctr = F.binary_cross_entropy_with_logits(ctr_p, ctr_t, reduction="mean") * centerness_weight

        # DFL on positives.
        # Reshape reg_logits to (B, H, W, 4*K) then index by pos_mask.
        rl = reg_logits.permute(0, 2, 3, 1)[pos_mask]                          # (P, 4*K)
        dfl_term = dfl_loss(rl, ltrb_target_pos) * dfl_weight

        # GIoU on the integrated decoded boxes vs GT.
        ltrb_pred_pos = dfl_expectation(reg_logits).permute(0, 2, 3, 1)[pos_mask]   # (P, 4)
        # Convert (l, t, r, b) at this cell to xyxy for the matched cell.
        # Build cell-centre positions for positive cells.
        i_idx, j_idx = torch.where(pos_mask.view(B, grid_h, grid_w).any(dim=0))
        # Easier: reuse decode_ltrb_to_xyxy on the full (B, 4, H, W) and slice.
        decoded_for_gi = decode_ltrb_to_xyxy(
            dfl_expectation(reg_logits), stride=stride
        ).permute(0, 2, 3, 1)[pos_mask]                                         # (P, 4)
        gt_for_gi = gt_bbox.view(B, 1, 1, 4).expand(B, grid_h, grid_w, 4)[pos_mask]
        giou = generalized_box_iou_loss(decoded_for_gi, gt_for_gi, reduction="mean") * giou_weight
    else:
        ctr = conf_logits.new_zeros(())
        dfl_term = conf_logits.new_zeros(())
        giou = conf_logits.new_zeros(())

    # Predicted-IoU head: L1 against detached cell_iou on positive cells.
    if pred_iou_logits is not None and pos_mask.any():
        target_iou = cell_iou[pos_mask].detach().clamp(0.0, 1.0)
        pred_iou = torch.sigmoid(pred_iou_logits.squeeze(1))[pos_mask]
        piou = F.l1_loss(pred_iou, target_iou, reduction="mean") * pred_iou_weight
    else:
        piou = conf_logits.new_zeros(())

    # Conf-map entropy regulariser.
    ent_reg = conf_entropy_reg(conf_logits, pos_mask, is_present)

    # Off-positive reg L2 prior (computed on all cells, weighted on negatives).
    reg_l2 = reg_l2_prior_off_pos(reg_logits, pos_mask)

    return {
        "qfl": qfl,
        "neg_qfl": neg_q,
        "centerness": ctr,
        "dfl": dfl_term,
        "giou": giou,
        "pred_iou": piou,
        "entropy_reg": ent_reg,
        "reg_l2_prior": reg_l2,
        "num_pos": num_pos.detach(),
        "pos_mask": pos_mask,
        "cell_iou": cell_iou,
    }


# ---------------------------------------------------------------------------
# Top-level total_loss
# ---------------------------------------------------------------------------


def total_loss(
    pred: dict[str, torch.Tensor],
    gt_bbox: torch.Tensor,
    is_present: torch.Tensor,
    support_bboxes: torch.Tensor | None = None,
    img_size: int = 224,
    presence_weight: float = 1.0,
    attn_loss_weight: float = 0.5,
    aux_weight: float = 0.5,
    entropy_reg_weight: float = 0.01,
    reg_l2_prior_weight: float = 1e-3,
    proto_norm_weight: float = 1e-3,
    # Phase 0 last-resort discrimination + anti-collapse weights.
    contrastive_presence_weight: float = 0.5,
    feature_spread_weight: float = 0.1,
    use_hardness_weighted_presence: bool = True,
    sample_loss_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combined loss across stride-16 + stride-8 + aux + presence + auxiliaries.

    Grid sizes are derived from the prediction tensors so multi-scale training
    (input 192 → P4 12×12 / P3 24×24, 256 → P4 16×16 / P3 32×32) works without
    any hard-coded 14/28 assumptions. Per-scale stride is derived as
    ``img_size / grid`` to keep cell-centre coords consistent with the GT
    bboxes (which are always in input-image px).
    """
    p4_h, p4_w = pred["conf_p4"].shape[-2:]
    p3_h, p3_w = pred["conf_p3"].shape[-2:]
    stride_p4 = img_size / float(p4_h)
    stride_p3 = img_size / float(p3_h)
    grids = [
        ("p4", pred["reg_p4_logits"], pred["conf_p4"], pred["ctr_p4"],
         pred.get("pred_iou_p4"), stride_p4, p4_h, p4_w),
        ("p3", pred["reg_p3_logits"], pred["conf_p3"], pred["ctr_p3"],
         pred.get("pred_iou_p3"), stride_p3, p3_h, p3_w),
    ]
    qfl_sum = pred["conf_p4"].new_zeros(())
    neg_qfl_sum = qfl_sum.clone()
    ctr_sum = qfl_sum.clone()
    dfl_sum = qfl_sum.clone()
    giou_sum = qfl_sum.clone()
    piou_sum = qfl_sum.clone()
    ent_sum = qfl_sum.clone()
    regl2_sum = qfl_sum.clone()
    for name, rl, cl, ctl, piou_l, stride, gh, gw in grids:
        terms = _dense_loss_one_scale(
            rl, cl, ctl, gt_bbox, is_present,
            stride=stride, grid_h=gh, grid_w=gw,
            pred_iou_logits=piou_l,
        )
        qfl_sum = qfl_sum + terms["qfl"]
        neg_qfl_sum = neg_qfl_sum + terms["neg_qfl"]
        ctr_sum = ctr_sum + terms["centerness"]
        dfl_sum = dfl_sum + terms["dfl"]
        giou_sum = giou_sum + terms["giou"]
        piou_sum = piou_sum + terms["pred_iou"]
        ent_sum = ent_sum + terms["entropy_reg"]
        regl2_sum = regl2_sum + terms["reg_l2_prior"]

    # Aux head (decoder layer-0 output).
    aux_h, aux_w = pred["conf_aux"].shape[-2:]
    aux_terms = _dense_loss_one_scale(
        pred["reg_aux_logits"],
        pred["conf_aux"],
        pred["ctr_aux"],
        gt_bbox,
        is_present,
        stride=img_size / float(aux_h),
        grid_h=aux_h,
        grid_w=aux_w,
        pred_iou_logits=pred.get("pred_iou_aux"),
    )
    aux_loss = (
        aux_terms["qfl"]
        + aux_terms["neg_qfl"]
        + aux_terms["centerness"]
        + aux_terms["dfl"]
        + aux_terms["giou"]
        + aux_terms["pred_iou"]
    )

    # Presence loss — hardness-weighted (auto-rebalances toward whichever
    # class the head is currently misclassifying) or plain class-balanced.
    presence_target = is_present.to(pred["presence_logit"].dtype)
    if use_hardness_weighted_presence:
        presence_loss = hardness_weighted_focal_bce(
            pred["presence_logit"], presence_target
        )
    else:
        presence_loss = class_balanced_focal_bce(
            pred["presence_logit"], presence_target
        )

    # Presence-aware contrastive loss + feature-spread regulariser.
    contrastive_loss = qfl_sum.new_zeros(())
    spread_loss = qfl_sum.new_zeros(())
    enriched = pred.get("enriched_p4")
    proto = pred.get("prototype")
    if enriched is not None and proto is not None and contrastive_presence_weight > 0:
        contrastive_loss = presence_aware_contrastive(
            enriched, proto, gt_bbox, is_present, img_size=img_size
        )
    if enriched is not None and feature_spread_weight > 0:
        spread_loss = feature_spread_reg(enriched)

    # Attention bbox loss.
    attn_loss = qfl_sum.new_zeros(())
    support_attn = pred.get("support_attn")
    if support_attn is not None and support_bboxes is not None:
        attn_loss = attention_bbox_loss(support_attn, support_bboxes, img_size=img_size)

    # Prototype L2 prior.
    proto = pred.get("prototype")
    pn = proto_norm_loss(proto) if proto is not None else qfl_sum.new_zeros(())

    total = (
        qfl_sum
        + neg_qfl_sum
        + ctr_sum
        + dfl_sum
        + giou_sum
        + piou_sum
        + entropy_reg_weight * ent_sum
        + reg_l2_prior_weight * regl2_sum
        + presence_weight * presence_loss
        + attn_loss_weight * attn_loss
        + aux_weight * aux_loss
        + proto_norm_weight * pn
        + contrastive_presence_weight * contrastive_loss
        + feature_spread_weight * spread_loss
    )

    # Per-source loss scaling: multiply the total by the batch-mean of the
    # per-sample weights provided by the trainer. This shifts the gradient
    # mass between sources without changing intra-batch balance — combined
    # with the source-balanced batch sampler it gives two independent levers
    # for emphasising the harder target-domain sources.
    if sample_loss_weight is not None and sample_loss_weight.numel() > 0:
        batch_w = sample_loss_weight.to(total.dtype).mean()
        total = total * batch_w

    return {
        "loss": total,
        "qfl": qfl_sum.detach(),
        "neg_qfl": neg_qfl_sum.detach(),
        "centerness": ctr_sum.detach(),
        "dfl": dfl_sum.detach(),
        "giou": giou_sum.detach(),
        "pred_iou": piou_sum.detach(),
        "entropy_reg": ent_sum.detach(),
        "reg_l2_prior": regl2_sum.detach(),
        "presence": presence_loss.detach(),
        "attn": attn_loss.detach(),
        "aux": aux_loss.detach(),
        "proto_norm": pn.detach(),
        "contrastive_presence": contrastive_loss.detach(),
        "feature_spread": spread_loss.detach(),
    }
