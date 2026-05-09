"""Localizer evaluation.

Reports a comprehensive metric set over POSITIVE episodes only (the localizer
trainer guarantees positive-only episodes; the evaluator filters defensively).

Categories:

  Localization quality (IoU-based):
    iou_mean / median / p25 / p75 / std
    frac_iou_at_X for X in {0.25, 0.50, 0.75, 0.90}
    map_50   (101-pt AP at IoU=0.50)
    map_75   (101-pt AP at IoU=0.75)
    map_5095 (mean over 10 thresholds 0.50:0.05:0.95, COCO-style)
    ap_per_iou: {"0.50": float, ..., "0.95": float}

  Containment (how much of GT is inside the predicted box):
    containment_mean / median / p25 / p75 / std
    frac_containment_X for X in {0.50, 0.75, 0.90, 0.99}
    map_50_containment   (101-pt AP using containment >= 0.5 as TP definition)
    map_90_containment   (101-pt AP using containment >= 0.9 as TP definition)

  Joint quality:
    contain_at_iou_50 / contain_at_iou_75
    high_contain_high_iou  (containment >= 0.9 AND IoU >= 0.5)

  Box-geometry diagnostics:
    mean_pred_box_area, std_pred_box_area
    frac_pred_box_too_big      (pred area > 0.4 of image)
    frac_pred_box_too_small    (pred area < 0.005 of image)
    mean_gt_box_area
    pred_to_gt_area_ratio_mean / median   (pred_area / gt_area)
    log_area_ratio_mean / std             (log(pred_area / gt_area))
    center_distance_mean       (||pred_center - gt_center|| in normalised coords)

  Score diagnostics:
    score_mean, score_std, score_p25, score_p75
    score_iou_correlation       (Pearson correlation between best_score and IoU)

Buckets: overall, per_source (hots/insdet, …), per_k (k1, k4, k_max).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

from localizer.loss import _cxcywh_to_xyxy
from localizer.model import MultiShotLocalizer


# COCO-style IoU thresholds (0.50, 0.55, ..., 0.95).
IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


# ---------------------------------------------------------------------------
# Box geometry helpers
# ---------------------------------------------------------------------------


def _box_area(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    return inter / (_box_area(a) + _box_area(b) - inter + 1e-6)


def _containment_ratio(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    inter_y1 = torch.maximum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    inter_x2 = torch.minimum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    inter_y2 = torch.minimum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    return inter / (_box_area(gt_xyxy) + 1e-6)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _ap_101(detections: list[tuple[float, bool]], n_gt: int) -> float:
    """COCO-style 101-point AP at a single threshold (already gated)."""
    if n_gt == 0 or not detections:
        return 0.0
    order = sorted(range(len(detections)), key=lambda i: -detections[i][0])
    tp = 0
    fp = 0
    precs: list[float] = []
    recs: list[float] = []
    for i in order:
        score, is_tp = detections[i]
        if is_tp:
            tp += 1
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / n_gt)
    # Monotone-decreasing precision envelope.
    for k in range(len(precs) - 2, -1, -1):
        if precs[k] < precs[k + 1]:
            precs[k] = precs[k + 1]
    rec_thresholds = [i / 100.0 for i in range(101)]
    ap = 0.0
    j = 0
    for rt in rec_thresholds:
        while j < len(recs) and recs[j] < rt:
            j += 1
        ap += precs[j] if j < len(precs) else 0.0
    return ap / 101.0


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


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


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    sx = sum((x - mx) ** 2 for x in xs[:n])
    sy = sum((y - my) ** 2 for y in ys[:n])
    if sx <= 0 or sy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs[:n], ys[:n]))
    return cov / math.sqrt(sx * sy)


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict[str, list]:
    return {
        "iou": [],
        "contain": [],
        "score": [],
        "pred_box_area": [],
        "gt_box_area": [],
        "center_distance": [],
    }


def _bucket_metrics(b: dict[str, list]) -> dict[str, Any]:
    n = len(b["iou"])
    if n == 0:
        return {"n": 0, "n_pos": 0}
    out: dict[str, Any] = {"n": n, "n_pos": n}

    # ── IoU (localization tightness) ──────────────────────────────────────
    out["iou_mean"]   = _safe_mean(b["iou"])
    out["iou_median"] = _quantile(b["iou"], 0.5)
    out["iou_p25"]    = _quantile(b["iou"], 0.25)
    out["iou_p75"]    = _quantile(b["iou"], 0.75)
    out["iou_std"]    = _safe_std(b["iou"])
    out["frac_iou_25"] = sum(1 for v in b["iou"] if v >= 0.25) / n
    out["frac_iou_50"] = sum(1 for v in b["iou"] if v >= 0.50) / n
    out["frac_iou_75"] = sum(1 for v in b["iou"] if v >= 0.75) / n
    out["frac_iou_90"] = sum(1 for v in b["iou"] if v >= 0.90) / n

    # ── Containment (how much of GT is inside the predicted box) ─────────
    out["containment_mean"]   = _safe_mean(b["contain"])
    out["containment_median"] = _quantile(b["contain"], 0.5)
    out["containment_p25"]    = _quantile(b["contain"], 0.25)
    out["containment_p75"]    = _quantile(b["contain"], 0.75)
    out["containment_std"]    = _safe_std(b["contain"])
    out["frac_containment_50"]   = sum(1 for c in b["contain"] if c >= 0.50) / n
    out["frac_containment_75"]   = sum(1 for c in b["contain"] if c >= 0.75) / n
    out["frac_containment_90"]   = sum(1 for c in b["contain"] if c >= 0.90) / n
    out["frac_containment_full"] = sum(1 for c in b["contain"] if c >= 0.99) / n

    # ── Joint quality ────────────────────────────────────────────────────
    out["contain_at_iou_50"]     = out["frac_iou_50"]
    out["contain_at_iou_75"]     = out["frac_iou_75"]
    out["high_contain_high_iou"] = sum(
        1 for iou, c in zip(b["iou"], b["contain"]) if iou >= 0.5 and c >= 0.9
    ) / n

    # ── mAP family (one detection per positive episode) ──────────────────
    # Each positive episode contributes one detection scored by best_score
    # (sigmoid of top-1 patch logit). TP definition varies by threshold.
    ap_per_iou: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        detections = [(s, iou >= thr) for s, iou in zip(b["score"], b["iou"])]
        ap_per_iou[f"{thr:.2f}"] = _ap_101(detections, n_gt=n)
    out["ap_per_iou"] = ap_per_iou
    out["map_50"]   = ap_per_iou["0.50"]
    out["map_75"]   = ap_per_iou["0.75"]
    out["map_5095"] = _safe_mean(list(ap_per_iou.values()))

    # Containment-mAP (TP iff containment >= 0.5 / 0.9).
    det_c50 = [(s, c >= 0.5) for s, c in zip(b["score"], b["contain"])]
    det_c90 = [(s, c >= 0.9) for s, c in zip(b["score"], b["contain"])]
    out["map_50_containment"] = _ap_101(det_c50, n_gt=n)
    out["map_90_containment"] = _ap_101(det_c90, n_gt=n)

    # ── Box-geometry diagnostics ────────────────────────────────────────
    out["mean_pred_box_area"]    = _safe_mean(b["pred_box_area"])
    out["std_pred_box_area"]     = _safe_std(b["pred_box_area"])
    out["frac_pred_box_too_big"] = sum(1 for a in b["pred_box_area"] if a > 0.4) / n
    out["frac_pred_box_too_small"] = sum(1 for a in b["pred_box_area"] if a < 0.005) / n
    out["mean_gt_box_area"]      = _safe_mean(b["gt_box_area"])
    # pred / gt area ratio.
    ratios: list[float] = []
    log_ratios: list[float] = []
    for pa, ga in zip(b["pred_box_area"], b["gt_box_area"]):
        if ga > 1e-9:
            r = pa / ga
            ratios.append(r)
            if r > 1e-9:
                log_ratios.append(math.log(r))
    out["pred_to_gt_area_ratio_mean"]   = _safe_mean(ratios)
    out["pred_to_gt_area_ratio_median"] = _quantile(ratios, 0.5) if ratios else 0.0
    out["log_area_ratio_mean"]          = _safe_mean(log_ratios)
    out["log_area_ratio_std"]           = _safe_std(log_ratios)
    out["center_distance_mean"]         = _safe_mean(b["center_distance"])
    out["center_distance_median"]       = _quantile(b["center_distance"], 0.5)

    # ── Score diagnostics ───────────────────────────────────────────────
    out["score_mean"]            = _safe_mean(b["score"])
    out["score_std"]             = _safe_std(b["score"])
    out["score_p25"]             = _quantile(b["score"], 0.25)
    out["score_p75"]             = _quantile(b["score"], 0.75)
    out["score_iou_correlation"] = _pearson(b["score"], b["iou"])

    return out


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: MultiShotLocalizer,
    loader: DataLoader,
    device: torch.device,
    *,
    progress: bool = True,
    progress_every: int = 5,
    phase0: bool = False,
) -> dict[str, Any]:
    """Run evaluation. Skips negative episodes (localizer is positive-only).

    If ``phase0=True`` uses ``model.phase0_forward`` (zero-shot OWLv2 baseline).

    Returned dict structure::

        {
          "overall":    {<metrics>},
          "per_source": {"hots": {<metrics>}, "insdet": {<metrics>}, ...},
          "per_k":      {"k1": {<metrics>}, "k4": {<metrics>}, "k10": {<metrics>}},
          "iou_thresholds": [0.50, 0.55, ..., 0.95],
        }
    """
    model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)
    per_k: dict[str, dict[str, list]] = defaultdict(_empty_bucket)

    n_batches_total = len(loader) if hasattr(loader, "__len__") else None
    t_start = time.time()
    if progress:
        print(f"  evaluating: {n_batches_total or '?'} batches", flush=True)

    n_seen = 0
    for batch in loader:
        sup = batch["support_imgs"].to(device)
        sup_mask = batch["support_mask"].to(device)
        qry = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)
        sources = batch["source"]
        ks = batch["k"]
        B = sup.size(0)

        if phase0:
            out = model.phase0_forward(sup, sup_mask, qry)
        else:
            out = model(sup, sup_mask, qry)
        pred_box = out["best_box"]
        pred_xyxy = _cxcywh_to_xyxy(pred_box)
        gt_xyxy = _cxcywh_to_xyxy(gt_bbox)
        ious = _iou_xyxy(pred_xyxy, gt_xyxy)
        contains = _containment_ratio(pred_xyxy, gt_xyxy)
        scores = out["best_score"]

        for i in range(B):
            if not bool(is_present[i].item()):
                continue
            src = sources[i]
            k_label = f"k{int(ks[i].item())}"
            iou_v = float(ious[i].item())
            cont_v = float(contains[i].item())
            sc_v = float(scores[i].item())
            pred_w = float(pred_box[i, 2].clamp(min=0).item())
            pred_h = float(pred_box[i, 3].clamp(min=0).item())
            pred_area = pred_w * pred_h
            gt_w = float(gt_bbox[i, 2].clamp(min=0).item())
            gt_h = float(gt_bbox[i, 3].clamp(min=0).item())
            gt_area = gt_w * gt_h
            pred_cx = float(pred_box[i, 0].item())
            pred_cy = float(pred_box[i, 1].item())
            gt_cx = float(gt_bbox[i, 0].item())
            gt_cy = float(gt_bbox[i, 1].item())
            cdist = math.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)
            for bucket in (overall, per_source[src], per_k[k_label]):
                bucket["iou"].append(iou_v)
                bucket["contain"].append(cont_v)
                bucket["score"].append(sc_v)
                bucket["pred_box_area"].append(pred_area)
                bucket["gt_box_area"].append(gt_area)
                bucket["center_distance"].append(cdist)

        n_seen += 1
        if progress and (n_seen % progress_every == 0 or n_seen == n_batches_total):
            elapsed = time.time() - t_start
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  [{n_seen}/{n_batches_total or '?'}]  "
                  f"elapsed={elapsed:5.1f}s  rate={rate:.2f}b/s", flush=True)

    return {
        "overall":    _bucket_metrics(overall),
        "per_source": {s: _bucket_metrics(b) for s, b in per_source.items()},
        "per_k":      {k: _bucket_metrics(b) for k, b in per_k.items()},
        "iou_thresholds": list(IOU_THRESHOLDS),
    }
