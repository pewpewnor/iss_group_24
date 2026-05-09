"""Localizer evaluation.

Reports (over POSITIVE episodes only):
    map_50          : COCO-style 101-point AP at IoU=0.5
    iou_mean / median / p25 / p75
    containment_mean         : area(pred ∩ gt) / area(gt)  — how much of GT is inside pred
    frac_containment_50/75/90/full : fraction of episodes where containment ≥ X
    contain_at_iou_50, contain_at_iou_75 : fraction of (high-IoU) cases where
                                            GT is also well-contained
    mean_pred_box_area, frac_pred_box_too_big

Buckets:
    overall, per_source (hots/insdet), per_k (k1, k4, k_max).

The localizer trainer guarantees positive-only episodes; evaluator filters
just in case (defensive — we always operate on `is_present=True` rows).
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


def _ap_50(detections: list[tuple[float, bool]], n_gt: int) -> float:
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


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict[str, list]:
    return {"iou": [], "contain": [], "score": [], "pred_box_area": []}


def _bucket_metrics(b: dict[str, list]) -> dict[str, float]:
    n = len(b["iou"])
    if n == 0:
        return {"n": 0, "n_pos": 0}
    out: dict[str, float] = {"n": n, "n_pos": n}
    out["iou_mean"] = _safe_mean(b["iou"])
    out["iou_median"] = _quantile(b["iou"], 0.5)
    out["iou_p25"] = _quantile(b["iou"], 0.25)
    out["iou_p75"] = _quantile(b["iou"], 0.75)
    # Containment: how much of GT is inside the predicted bbox.
    # 1.0 means the prediction fully encloses the GT (regardless of pred size).
    out["containment_mean"]   = _safe_mean(b["contain"])
    out["containment_median"] = _quantile(b["contain"], 0.5)
    out["containment_p25"]    = _quantile(b["contain"], 0.25)
    out["containment_p75"]    = _quantile(b["contain"], 0.75)
    out["frac_containment_50"]   = sum(1 for c in b["contain"] if c >= 0.50) / n
    out["frac_containment_75"]   = sum(1 for c in b["contain"] if c >= 0.75) / n
    out["frac_containment_90"]   = sum(1 for c in b["contain"] if c >= 0.90) / n
    out["frac_containment_full"] = sum(1 for c in b["contain"] if c >= 0.99) / n
    # Joint "well-localized AND well-contained" diagnostics.
    out["contain_at_iou_50"] = sum(1 for iou in b["iou"] if iou >= 0.5)  / n
    out["contain_at_iou_75"] = sum(1 for iou in b["iou"] if iou >= 0.75) / n
    out["high_contain_high_iou"] = sum(
        1 for iou, c in zip(b["iou"], b["contain"]) if iou >= 0.5 and c >= 0.9
    ) / n
    # mAP@50: every positive episode is one detection with score=top1 logit;
    # TP iff IoU >= 0.5; n_gt = n_positives.
    detections = [(s, iou >= 0.5) for s, iou in zip(b["score"], b["iou"])]
    out["map_50"] = _ap_50(detections, n_gt=n)
    # Containment-mAP: same thing but TP iff containment >= 0.9 (instead of
    # IoU >= 0.5). This isolates "did we surround the object" from "was the
    # box tight". Useful complement to map_50 when the user explicitly cares
    # about containment.
    detections_c = [(s, c >= 0.9) for s, c in zip(b["score"], b["contain"])]
    out["map_50_containment"] = _ap_50(detections_c, n_gt=n)
    out["mean_pred_box_area"] = _safe_mean(b["pred_box_area"])
    out["frac_pred_box_too_big"] = sum(1 for a in b["pred_box_area"] if a > 0.4) / n
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
            area_v = float((pred_box[i, 2] * pred_box[i, 3]).clamp(min=0).item())
            for bucket in (overall, per_source[src], per_k[k_label]):
                bucket["iou"].append(iou_v)
                bucket["contain"].append(cont_v)
                bucket["score"].append(sc_v)
                bucket["pred_box_area"].append(area_v)

        n_seen += 1
        if progress and (n_seen % progress_every == 0 or n_seen == n_batches_total):
            elapsed = time.time() - t_start
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  [{n_seen}/{n_batches_total or '?'}]  "
                  f"elapsed={elapsed:5.1f}s  rate={rate:.2f}b/s", flush=True)

    return {
        "overall": _bucket_metrics(overall),
        "per_source": {s: _bucket_metrics(b) for s, b in per_source.items()},
        "per_k": {k: _bucket_metrics(b) for k, b in per_k.items()},
    }
