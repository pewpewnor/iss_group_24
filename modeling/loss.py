"""Loss stack for the OWLv2 few-shot localizer.

Components:

  - Focal loss on existence_logit (image-level binary classification).
  - L1 + GIoU on the top-1 predicted box (positive episodes only,
    gated by ``existence_prob.detach() > 0.5``).
  - Anti-collapse regularisers:
      * box-area soft prior (penalises predicted area > 0.6 of image)
      * existence-mean KL prior (penalises mean(existence_prob) > 0.85)

Bbox format conventions in this module:
  - GT box from dataset: cxcywh in [0,1] (already normalised).
  - Pred box from model: cxcywh in [0,1] (OWLv2 native).
  - GIoU computed in xyxy normalised space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Box helpers (kept tensor-only for AMP-safety)
# ---------------------------------------------------------------------------


def _cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _box_area(box_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box_xyxy.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = _box_area(a)
    area_b = _box_area(b)
    return inter / (area_a + area_b - inter + 1e-6)


def _giou_loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss = 1 - GIoU.  Vectorised over batch."""
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


def _containment_ratio(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """area(pred ∩ gt) / area(gt) — fraction of GT covered by pred."""
    inter_x1 = torch.maximum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    inter_y1 = torch.maximum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    inter_x2 = torch.minimum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    inter_y2 = torch.minimum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_g = _box_area(gt_xyxy)
    return inter / (area_g + 1e-6)


# ---------------------------------------------------------------------------
# Focal existence loss
# ---------------------------------------------------------------------------


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid-focal binary cross-entropy.

    logits  : (B,)  pre-sigmoid existence logits
    targets : (B,)  0/1 float
    """
    p = torch.sigmoid(logits)
    eps = 1e-6
    ce_pos = -targets * torch.log(p.clamp(min=eps))
    ce_neg = -(1 - targets) * torch.log((1 - p).clamp(min=eps))
    fl_pos = alpha * (1 - p).pow(gamma) * ce_pos
    fl_neg = (1 - alpha) * p.pow(gamma) * ce_neg
    return (fl_pos + fl_neg).mean()


# ---------------------------------------------------------------------------
# Anti-collapse regularisers
# ---------------------------------------------------------------------------


def box_area_prior(
    pred_box_cxcywh: torch.Tensor,
    threshold: float = 0.6,
) -> torch.Tensor:
    """Penalise predicted boxes whose area > ``threshold`` of image.

    pred_box_cxcywh is normalised cxcywh in [0,1].  Returns a scalar.
    """
    w = pred_box_cxcywh[..., 2].clamp(min=0)
    h = pred_box_cxcywh[..., 3].clamp(min=0)
    area = w * h
    excess = (area - threshold).clamp(min=0)
    return (excess.pow(2)).mean()


def existence_mean_kl(
    existence_prob: torch.Tensor,
    target: float = 0.5,
    threshold: float = 0.85,
) -> torch.Tensor:
    """KL(mean(existence_prob) || target).  Triggered only when mean > threshold.

    Soft repellent against the existence-stuck-at-1 collapse mode.
    """
    mean_p = existence_prob.mean()
    if mean_p.item() <= threshold:
        return existence_prob.new_zeros(())
    eps = 1e-6
    p = mean_p.clamp(min=eps, max=1 - eps)
    q = torch.tensor(target, device=p.device, dtype=p.dtype)
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return kl


def existence_margin_loss(
    existence_logit: torch.Tensor,
    is_present: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge-style separation loss between positive and negative logits.

    Forces ``mean_pos_logit - mean_neg_logit >= margin`` over the batch.
    Directly attacks the constant-output collapse mode where
    ``existence_head`` outputs the same value for every episode.

    Returns a scalar tensor.  Returns 0 if either pos or neg episodes
    are absent in the batch (margin is undefined in that case).
    """
    is_present_b = is_present.bool()
    pos_logits = existence_logit[is_present_b]
    neg_logits = existence_logit[~is_present_b]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return existence_logit.new_zeros(())
    gap = pos_logits.mean() - neg_logits.mean()
    return F.relu(margin - gap)


def nt_xent_prototype_loss(
    prototype: torch.Tensor,
    instance_id: list[str],
    temperature: float = 0.1,
) -> torch.Tensor:
    """SimCLR-style NT-Xent contrastive loss on prototypes.

    Treats prototypes from the same instance (different sampled support
    sets within a batch) as positive pairs; all other in-batch prototypes
    are negatives.  Forces the aggregator to produce instance-discriminative
    embeddings even when the focal loss has degenerated.

    Args:
        prototype   : (B, D) prototypes for each episode.
        instance_id : list of length B; same string ⇒ same instance.
        temperature : NT-Xent softmax temperature.

    Returns the InfoNCE loss averaged over anchor positions that have
    at least one positive partner.  Returns 0 when no positive pairs
    exist in the batch (i.e. every episode is from a different instance).
    """
    b = prototype.size(0)
    if b < 2:
        return prototype.new_zeros(())

    # L2-normalise prototypes for cosine-similarity logits.
    p_norm = F.normalize(prototype, dim=-1)
    sim = p_norm @ p_norm.t() / temperature                          # (B, B)
    # Mask self-similarity.
    eye = torch.eye(b, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # Build positive mask: same instance_id and not self.
    pos_mask = torch.zeros((b, b), dtype=torch.bool, device=sim.device)
    for i in range(b):
        for j in range(b):
            if i != j and instance_id[i] == instance_id[j]:
                pos_mask[i, j] = True

    # Loss per anchor row that has at least one positive.
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return prototype.new_zeros(())

    log_prob = F.log_softmax(sim, dim=1)                              # (B, B)
    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)
    n_pos_per_row = pos_mask.sum(dim=1).clamp(min=1)
    per_row = -(pos_log_prob / n_pos_per_row)
    return per_row[has_pos].mean()


# ---------------------------------------------------------------------------
# Combined total loss
# ---------------------------------------------------------------------------


def total_loss(
    out: dict[str, torch.Tensor],
    gt_bbox_cxcywh: torch.Tensor,
    is_present: torch.Tensor,
    *,
    use_box_loss: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    lambda_l1: float = 5.0,
    lambda_giou: float = 2.0,
    anti_collapse_weight: float = 0.1,
    box_size_threshold: float = 0.6,
    existence_kl_threshold: float = 0.85,
    # New separation / discriminative terms (added to fight existence-head
    # collapse and to push the aggregator toward instance-discriminative
    # prototypes).
    margin_weight: float = 0.5,
    margin_value: float = 1.0,
    contrastive_weight: float = 0.1,
    contrastive_temp: float = 0.1,
    instance_id: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute combined loss + per-term diagnostics.

    Args:
        out                 : model output dict.
        gt_bbox_cxcywh      : (B, 4) normalised cxcywh.
        is_present          : (B,) bool.
        use_box_loss        : disable for Stage 1.1 (existence-only).
        margin_weight       : weight on the pos/neg existence-logit margin
                              loss.  Set to 0 to disable.
        margin_value        : required gap between mean(pos_logit) and
                              mean(neg_logit).
        contrastive_weight  : weight on the NT-Xent contrastive loss applied
                              to prototypes.  Set to 0 to disable.
        contrastive_temp    : temperature for NT-Xent.
        instance_id         : optional list of instance ids per batch element.
                              Required when contrastive_weight > 0; otherwise
                              the contrastive term silently no-ops.
    """
    targets = is_present.float()
    existence_logit = out["existence_logit"]
    existence_prob = out["existence_prob"]
    pred_box = out["best_box"]                                        # (B, 4) cxcywh

    focal = focal_bce_loss(existence_logit, targets, focal_alpha, focal_gamma)

    losses: dict[str, torch.Tensor] = {"focal": focal}
    total = focal

    # Margin loss: forbid the constant-output collapse.
    if margin_weight > 0.0:
        margin = existence_margin_loss(existence_logit, is_present, margin=margin_value)
        losses["margin"] = margin
        total = total + margin_weight * margin
    else:
        losses["margin"] = focal.new_zeros(())

    # NT-Xent on prototypes: instance-discriminative pressure.
    if contrastive_weight > 0.0 and instance_id is not None and "prototype" in out:
        nt = nt_xent_prototype_loss(
            out["prototype"], instance_id, temperature=contrastive_temp
        )
        losses["nt_xent"] = nt
        total = total + contrastive_weight * nt
    else:
        losses["nt_xent"] = focal.new_zeros(())

    # Box loss only on confident-positive predictions of positive episodes.
    if use_box_loss:
        gate = (existence_prob.detach() > 0.5) & is_present
        if gate.any():
            pb = pred_box[gate]
            gb = gt_bbox_cxcywh[gate]
            l1 = F.l1_loss(pb, gb, reduction="mean")
            pb_xyxy = _cxcywh_to_xyxy(pb)
            gb_xyxy = _cxcywh_to_xyxy(gb)
            giou = _giou_loss(pb_xyxy, gb_xyxy).mean()
            box_loss = lambda_l1 * l1 + lambda_giou * giou
            losses["l1"] = l1
            losses["giou"] = giou
            losses["box_loss"] = box_loss
            total = total + box_loss
        else:
            losses["l1"] = focal.new_zeros(())
            losses["giou"] = focal.new_zeros(())
            losses["box_loss"] = focal.new_zeros(())
    else:
        losses["l1"] = focal.new_zeros(())
        losses["giou"] = focal.new_zeros(())
        losses["box_loss"] = focal.new_zeros(())

    # Anti-collapse.
    box_area = box_area_prior(pred_box, threshold=box_size_threshold)
    exist_kl = existence_mean_kl(
        existence_prob, target=0.5, threshold=existence_kl_threshold
    )
    losses["box_area_penalty"] = box_area
    losses["existence_kl"] = exist_kl
    total = total + anti_collapse_weight * (box_area + exist_kl)

    losses["loss"] = total
    return losses
