"""Evaluation pipeline for the OWLv2 few-shot localizer.

Returns the kitchen-sink metric set:

  Localisation:
    iou_mean / iou_median / iou_p25 / iou_p75
    contain_mean / contain_at_iou_50 / contain_at_iou_75

  mAP:
    map_50, map_75, map_5095, ap_per_iou (10 thresholds)
    f1_50, precision_50, recall_50

  Existence (image-level):
    existence_acc / acc_pos / acc_neg
    existence_auroc, existence_pr_auc, existence_brier, existence_f1
    false_positive_rate, false_negative_rate
    mean_score_pos / mean_score_neg

  Collapse diagnostics:
    mean_pred_box_area, frac_pred_box_too_big
    mean_existence_prob, frac_high_existence
    prototype_norm_mean / std

  per_source: same metrics bucketed by inst.source.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modeling.loss import _containment_ratio, _cxcywh_to_xyxy, _iou_xyxy
from modeling.model import OWLv2FewShotLocalizer


IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _detections_to_ap(detections: list[tuple[float, bool]], n_gt: int) -> float:
    """COCO-style 101-point AP from a flat list of (score, is_tp)."""
    if n_gt == 0 or not detections:
        return 0.0
    order = sorted(range(len(detections)), key=lambda i: -detections[i][0])
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for i in order:
        score, is_tp = detections[i]
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    for k in range(len(precisions) - 2, -1, -1):
        if precisions[k] < precisions[k + 1]:
            precisions[k] = precisions[k + 1]
    rec_thresholds = [i / 100.0 for i in range(101)]
    ap_sum = 0.0
    j = 0
    for rt in rec_thresholds:
        while j < len(recalls) and recalls[j] < rt:
            j += 1
        ap_sum += precisions[j] if j < len(precisions) else 0.0
    return ap_sum / 101.0


def _binary_auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return 0.0
    n_pos = len(pos)
    n_neg = len(neg)
    combined = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, combined) if y == 1)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _binary_pr_auc(scores: list[float], labels: list[bool]) -> float:
    if not labels or not any(labels):
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    n_pos = sum(labels)
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    auc = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        auc += p * max(0.0, r - prev_r)
        prev_r = r
    return auc


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Per-image bucket
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict[str, list]:
    return {
        "iou": [],
        "contain": [],
        "score": [],
        "is_present": [],
        "pred_box_area": [],
        "existence_prob": [],
        "prototype_norm": [],
    }


def _bucket_metrics(b: dict[str, list], score_thr: float = 0.5) -> dict[str, float]:
    n = len(b["is_present"])
    n_pos = sum(b["is_present"])
    n_neg = n - n_pos
    out: dict[str, float] = {"n": n, "n_pos": n_pos, "n_neg": n_neg}
    if n == 0:
        return out

    # Localisation (positive-only).
    pos_iou = [iou for iou, p in zip(b["iou"], b["is_present"]) if p]
    pos_cont = [c for c, p in zip(b["contain"], b["is_present"]) if p]
    out["iou_mean"] = _safe_mean(pos_iou)
    out["iou_median"] = _quantile(pos_iou, 0.5)
    out["iou_p25"] = _quantile(pos_iou, 0.25)
    out["iou_p75"] = _quantile(pos_iou, 0.75)
    out["contain_mean"] = _safe_mean(pos_cont)
    out["contain_at_iou_50"] = (
        sum(1 for iou, c in zip(pos_iou, pos_cont) if iou >= 0.5) / max(len(pos_iou), 1)
    )
    out["contain_at_iou_75"] = (
        sum(1 for iou, c in zip(pos_iou, pos_cont) if iou >= 0.75) / max(len(pos_iou), 1)
    )

    # mAP — one detection per image (top-1 box of the model).
    # TP at IoU threshold = is_present AND IoU(pred, gt) >= threshold AND existence_prob > 0.5.
    # FP = (existence_prob > 0.5 AND not present) OR (positive AND IoU < threshold).
    final_score = b["existence_prob"]                          # use existence_prob as ranking score
    ap_per_iou: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        detections = []
        for iou, score, present, ex in zip(
            b["iou"], b["score"], b["is_present"], b["existence_prob"]
        ):
            if ex < 0.05:
                continue                                       # too low to count even as FP under top-K=1 setting
            is_tp = present and (iou >= thr)
            detections.append((float(ex), bool(is_tp)))
        ap = _detections_to_ap(detections, n_gt=n_pos)
        ap_per_iou[f"{thr:.2f}"] = ap
    out["ap_per_iou"] = ap_per_iou
    out["map_50"] = ap_per_iou.get("0.50", 0.0)
    out["map_75"] = ap_per_iou.get("0.75", 0.0)
    out["map_5095"] = _safe_mean(list(ap_per_iou.values()))

    # P/R/F1 at IoU=0.50.
    tp = sum(1 for iou, p, ex in zip(b["iou"], b["is_present"], b["existence_prob"])
             if p and iou >= 0.5 and ex > score_thr)
    fp = sum(1 for p, ex in zip(b["is_present"], b["existence_prob"])
             if (not p and ex > score_thr) or
                (p and ex > score_thr and 0))                  # high-IoU pos already counted as tp
    fp += sum(1 for iou, p, ex in zip(b["iou"], b["is_present"], b["existence_prob"])
              if p and ex > score_thr and iou < 0.5)
    fn_pos = sum(1 for iou, p, ex in zip(b["iou"], b["is_present"], b["existence_prob"])
                 if p and (ex <= score_thr or iou < 0.5))
    fn = sum(1 for p, ex in zip(b["is_present"], b["existence_prob"])
             if p and ex <= score_thr)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    out["precision_50"] = precision
    out["recall_50"] = recall
    out["f1_50"] = f1

    # Existence (image-level).
    pred_pos = [ex > score_thr for ex in b["existence_prob"]]
    correct = [pp == p for pp, p in zip(pred_pos, b["is_present"])]
    out["existence_acc"] = _safe_mean([1.0 if c else 0.0 for c in correct])
    if n_pos:
        out["existence_acc_pos"] = _safe_mean(
            [1.0 if pp else 0.0 for pp, p in zip(pred_pos, b["is_present"]) if p]
        )
    else:
        out["existence_acc_pos"] = 0.0
    if n_neg:
        out["existence_acc_neg"] = _safe_mean(
            [1.0 if not pp else 0.0 for pp, p in zip(pred_pos, b["is_present"]) if not p]
        )
    else:
        out["existence_acc_neg"] = 0.0
    out["existence_auroc"] = _binary_auroc(b["existence_prob"], b["is_present"])
    out["existence_pr_auc"] = _binary_pr_auc(b["existence_prob"], b["is_present"])
    out["existence_brier"] = _safe_mean(
        [(ex - (1.0 if p else 0.0)) ** 2 for ex, p in zip(b["existence_prob"], b["is_present"])]
    )
    out["existence_f1"] = f1                                   # equal to f1_50 by construction
    out["false_positive_rate"] = (
        sum(1 for pp, p in zip(pred_pos, b["is_present"]) if pp and not p) / max(n_neg, 1)
    )
    out["false_negative_rate"] = (
        sum(1 for pp, p in zip(pred_pos, b["is_present"]) if (not pp) and p) / max(n_pos, 1)
    )
    out["mean_score_pos"] = _safe_mean(
        [ex for ex, p in zip(b["existence_prob"], b["is_present"]) if p]
    )
    out["mean_score_neg"] = _safe_mean(
        [ex for ex, p in zip(b["existence_prob"], b["is_present"]) if not p]
    )

    # Collapse diagnostics.
    out["mean_pred_box_area"] = _safe_mean(b["pred_box_area"])
    out["frac_pred_box_too_big"] = (
        sum(1 for a in b["pred_box_area"] if a > 0.4) / max(len(b["pred_box_area"]), 1)
    )
    out["mean_existence_prob"] = _safe_mean(b["existence_prob"])
    out["frac_high_existence"] = (
        sum(1 for ex in b["existence_prob"] if ex > 0.9) / max(len(b["existence_prob"]), 1)
    )
    out["prototype_norm_mean"] = _safe_mean(b["prototype_norm"])
    out["prototype_norm_std"] = _safe_std(b["prototype_norm"])
    return out


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: OWLv2FewShotLocalizer,
    loader: DataLoader,
    device: torch.device,
    score_thr: float = 0.5,
) -> dict[str, Any]:
    """Run evaluation and return the kitchen-sink metric dict.

    Returns:
        {
          "overall": {...},
          "per_source": {"hots": {...}, "insdet": {...}, "vizwiz_novel": {...}}
        }
    """
    model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)

    for batch in loader:
        support_imgs = batch["support_imgs"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)                # cxcywh normalised
        is_present = batch["is_present"].to(device)
        sources = batch["source"]

        out = model(support_imgs, query_img)

        pred_box = out["best_box"]                              # (B, 4) cxcywh
        existence_prob = out["existence_prob"]
        score = torch.sigmoid(out["best_score"])                # alternate score signal
        prototype = out["prototype"]

        pred_xyxy = _cxcywh_to_xyxy(pred_box).clamp(0, 1)
        gt_xyxy = _cxcywh_to_xyxy(gt_bbox).clamp(0, 1)
        iou = _iou_xyxy(pred_xyxy, gt_xyxy)
        contain = _containment_ratio(pred_xyxy, gt_xyxy)
        pred_area = (pred_box[..., 2].clamp(0, 1) * pred_box[..., 3].clamp(0, 1))

        b = pred_box.size(0)
        for i in range(b):
            entry_iou = float(iou[i].item())
            entry_contain = float(contain[i].item())
            entry_score = float(score[i].item())
            entry_present = bool(is_present[i].item())
            entry_area = float(pred_area[i].item())
            entry_existence = float(existence_prob[i].item())
            entry_pnorm = float(prototype[i].norm().item())
            for bucket in (overall, per_source[sources[i]]):
                bucket["iou"].append(entry_iou)
                bucket["contain"].append(entry_contain)
                bucket["score"].append(entry_score)
                bucket["is_present"].append(entry_present)
                bucket["pred_box_area"].append(entry_area)
                bucket["existence_prob"].append(entry_existence)
                bucket["prototype_norm"].append(entry_pnorm)

    return {
        "overall": _bucket_metrics(overall, score_thr),
        "per_source": {k: _bucket_metrics(v, score_thr) for k, v in per_source.items()},
    }


# ---------------------------------------------------------------------------
# Phase 0 — zero-shot OWLv2 baseline
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_phase0(
    owlv2_model,
    loader: DataLoader,
    device: torch.device,
    score_thr: float = 0.5,
) -> dict[str, Any]:
    """Zero-shot OWLv2 image-guided detection.

    For each episode, run image-guided detection 4 times (one per support)
    and average the per-patch logits.  Top-1 box → predicted bbox.
    Existence = sigmoid(max averaged logit).
    """
    from modeling.loss import _cxcywh_to_xyxy as _to_xyxy

    owlv2_model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)

    for batch in loader:
        support_imgs = batch["support_imgs"].to(device)         # (B, 4, 3, S, S)
        query_img = batch["query_img"].to(device)               # (B, 3, S, S)
        gt_bbox = batch["query_bbox"]
        is_present = batch["is_present"]
        sources = batch["source"]

        b, v, c, s, _ = support_imgs.shape

        # Compute query feature map once.
        q_feature_map, _ = owlv2_model.image_embedder(
            pixel_values=query_img, interpolate_pos_encoding=True
        )                                                        # (B, gh, gw, D)
        gh, gw, d = q_feature_map.shape[1:]
        image_feats = q_feature_map.reshape(b, gh * gw, d)
        target_pred_boxes = owlv2_model.box_predictor(
            image_feats, q_feature_map, interpolate_pos_encoding=True
        )                                                        # (B, P, 4) cxcywh

        # For each support image, run embed_image_query to get a query embed,
        # then class_predictor to get logits.
        P = image_feats.size(1)
        all_logits = []
        for vi in range(v):
            sup_v = support_imgs[:, vi]                          # (B, 3, S, S)
            sup_feature_map, _ = owlv2_model.image_embedder(
                pixel_values=sup_v, interpolate_pos_encoding=True
            )                                                    # (B, gh, gw, D)
            sup_h, sup_w = sup_feature_map.shape[1:3]
            sup_feats = sup_feature_map.reshape(b, sup_h * sup_w, d)
            q_emb, _, _ = owlv2_model.embed_image_query(
                sup_feats, sup_feature_map, interpolate_pos_encoding=True
            )                                                    # q_emb: (B, 1, D_q) or None
            if q_emb is None:
                continue
            # embed_image_query already returns (B, 1, D_q) — see modeling_owlv2.py
            # line 1334 (torch.stack of class_embeds[i][best_box_ind]).  No need
            # to unsqueeze.
            if q_emb.dim() == 2:
                q_emb = q_emb.unsqueeze(1)
            logits, _ = owlv2_model.class_predictor(image_feats, q_emb)  # (B, P, 1)
            all_logits.append(logits.squeeze(-1))                # (B, P)
        if not all_logits:
            continue
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (B, P)
        best_idx = avg_logits.argmax(dim=-1).clamp(0, P - 1)     # (B,)
        ar = torch.arange(b, device=device)
        best_box = target_pred_boxes[ar, best_idx]               # (B, 4)
        best_logit = avg_logits[ar, best_idx]
        existence_prob = torch.sigmoid(best_logit)

        pred_xyxy = _to_xyxy(best_box).clamp(0, 1)
        gt_xyxy = _to_xyxy(gt_bbox.to(device)).clamp(0, 1)
        iou = _iou_xyxy(pred_xyxy, gt_xyxy)
        contain = _containment_ratio(pred_xyxy, gt_xyxy)
        pred_area = best_box[..., 2].clamp(0, 1) * best_box[..., 3].clamp(0, 1)

        for i in range(b):
            entry = {
                "iou": float(iou[i].item()),
                "contain": float(contain[i].item()),
                "score": float(existence_prob[i].item()),
                "is_present": bool(is_present[i].item()),
                "pred_box_area": float(pred_area[i].item()),
                "existence_prob": float(existence_prob[i].item()),
                "prototype_norm": 0.0,
            }
            for bucket in (overall, per_source[sources[i]]):
                for k, v_ in entry.items():
                    bucket[k].append(v_)

    return {
        "overall": _bucket_metrics(overall, score_thr),
        "per_source": {k: _bucket_metrics(v, score_thr) for k, v in per_source.items()},
    }
