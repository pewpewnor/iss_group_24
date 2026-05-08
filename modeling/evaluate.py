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

from modeling._tiling import detect_tiled
from modeling.loss import _containment_ratio, _cxcywh_to_xyxy, _iou_xyxy
from modeling.model import OWLv2FewShotLocalizer


IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


# Default tile-inference config — used when callers don't pass anything.
DEFAULT_TILE_CFG: dict[str, Any] = {
    "mode": "pyramid_a",                     # "off" | "pyramid_a" | "hybrid_d"
    "levels": (1, 2),                        # pyramid levels (k×k grids)
    "overlap": 0.30,                         # fractional overlap between adjacent tiles (M2)
    "for_sources": ("insdet",),              # which sources get tiled
    "nms_iou": 0.5,
    "top_k": 100,
    "score_combo": "existence_x_score",      # NMS ranking score
    "edge_score_penalty": 0.5,               # M4: down-weight detections hugging interior tile edges
    "edge_px": 4,                            # pixel tolerance for edge-hugging
    "merge_partial_boxes": True,             # M3: union-merge boundary-spanning detection pairs
    "merge_min_score": 0.2,
}


def resolve_tile_cfg(user_cfg: dict | None) -> dict:
    """Merge a user-supplied tile config with ``DEFAULT_TILE_CFG``."""
    out = dict(DEFAULT_TILE_CFG)
    if user_cfg:
        for k, v in user_cfg.items():
            out[k] = v
    # Coerce a few fields to tuples for safety.
    if isinstance(out.get("levels"), list):
        out["levels"] = tuple(out["levels"])
    if isinstance(out.get("for_sources"), list):
        out["for_sources"] = tuple(out["for_sources"])
    return out


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
    img_size: int = 768,
    tile_cfg: dict | None = None,
) -> dict[str, Any]:
    """Run evaluation and return the kitchen-sink metric dict.

    Args:
        tile_cfg: tile-inference config; merged with ``DEFAULT_TILE_CFG``.
            Set ``mode="off"`` to disable tiling entirely (single-pass eval).
            Episodes whose ``source`` is not in ``tile_cfg["for_sources"]``
            also use single-pass eval, regardless of mode.

    Returns:
        {
          "overall": {...},
          "per_source": {"hots": {...}, "insdet": {...}, "vizwiz_novel": {...}}
        }
    """
    model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)
    cfg = resolve_tile_cfg(tile_cfg)
    tile_mode = cfg["mode"]
    tile_for_sources = set(cfg["for_sources"])

    # When tiling is requested for any source, the loader must have been
    # built with ``return_native=True`` so each batch carries the native PIL.
    needs_native = tile_mode != "off"

    for batch in loader:
        support_imgs = batch["support_imgs"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox_norm = batch["query_bbox"].to(device)            # cxcywh normalised
        is_present = batch["is_present"].to(device)
        sources = batch["source"]
        b = support_imgs.size(0)
        native_size = batch.get("native_size")
        native_bbox = batch.get("native_bbox")
        query_native_list = batch.get("query_native")

        # Default: run single-pass forward for the whole batch.
        out = model(support_imgs, query_img)

        pred_box_norm = out["best_box"]                          # (B, 4) cxcywh in [0,1]
        existence_prob = out["existence_prob"]
        score_logit = out["best_score"]
        prototype = out["prototype"]

        # Per-episode loop — apply tiling for InsDet (or whatever
        # for_sources is configured) and use single-pass for the rest.
        for i in range(b):
            src = sources[i]
            present = bool(is_present[i].item())
            pnorm = float(prototype[i].norm().item())

            do_tile = (
                tile_mode != "off"
                and src in tile_for_sources
                and query_native_list is not None
                and native_size is not None
            )

            if do_tile:
                native_pil = query_native_list[i]
                tiled = detect_tiled(
                    model,
                    support_imgs[i:i + 1],
                    native_pil,
                    img_size=img_size,
                    mode=tile_mode,
                    levels=tuple(cfg["levels"]),
                    overlap=float(cfg["overlap"]),
                    nms_iou=float(cfg["nms_iou"]),
                    top_k=int(cfg["top_k"]),
                    score_combo=str(cfg["score_combo"]),
                    edge_score_penalty=float(cfg.get("edge_score_penalty", 0.5)),
                    edge_px=int(cfg.get("edge_px", 4)),
                    merge_partial_boxes=bool(cfg.get("merge_partial_boxes", True)),
                    merge_min_score=float(cfg.get("merge_min_score", 0.2)),
                )
                # Convert native xyxy → normalised xyxy of the *native* frame.
                nw, nh = int(native_size[i, 0].item()), int(native_size[i, 1].item())
                pred_xyxy_native = tiled["best_box_native_xyxy"]
                pred_xyxy_norm = pred_xyxy_native.clone().float()
                pred_xyxy_norm[0] /= max(nw, 1)
                pred_xyxy_norm[2] /= max(nw, 1)
                pred_xyxy_norm[1] /= max(nh, 1)
                pred_xyxy_norm[3] /= max(nh, 1)
                pred_xyxy_norm = pred_xyxy_norm.clamp(0, 1)
                ex = float(tiled["existence_prob"].item())
                sc = float(tiled["best_score"].item())
                # Box area in the normalised native frame.
                area = float(
                    (pred_xyxy_norm[2] - pred_xyxy_norm[0])
                    * (pred_xyxy_norm[3] - pred_xyxy_norm[1])
                )
                # GT bbox: prefer the native one (matches our tile coords).
                if native_bbox is not None and present:
                    gt_native = native_bbox[i].float()
                    gt_xyxy_norm = gt_native.clone()
                    gt_xyxy_norm[0] /= max(nw, 1)
                    gt_xyxy_norm[2] /= max(nw, 1)
                    gt_xyxy_norm[1] /= max(nh, 1)
                    gt_xyxy_norm[3] /= max(nh, 1)
                else:
                    gt_xyxy_norm = torch.zeros(4, device=pred_xyxy_norm.device)
                iou_v = float(_iou_xyxy(pred_xyxy_norm, gt_xyxy_norm).item()) if present else 0.0
                contain_v = float(_containment_ratio(pred_xyxy_norm, gt_xyxy_norm).item()) if present else 0.0
            else:
                # Single-pass path (default for HOTS, vizwiz_novel, or off).
                pred_xyxy_norm = _cxcywh_to_xyxy(pred_box_norm[i]).clamp(0, 1)
                gt_xyxy_norm = _cxcywh_to_xyxy(gt_bbox_norm[i]).clamp(0, 1)
                iou_v = float(_iou_xyxy(pred_xyxy_norm, gt_xyxy_norm).item())
                contain_v = float(_containment_ratio(pred_xyxy_norm, gt_xyxy_norm).item())
                ex = float(existence_prob[i].item())
                sc = float(torch.sigmoid(score_logit[i]).item())
                area = float(
                    pred_box_norm[i, 2].clamp(0, 1) * pred_box_norm[i, 3].clamp(0, 1)
                )

            for bucket in (overall, per_source[src]):
                bucket["iou"].append(iou_v)
                bucket["contain"].append(contain_v)
                bucket["score"].append(sc)
                bucket["is_present"].append(present)
                bucket["pred_box_area"].append(area)
                bucket["existence_prob"].append(ex)
                bucket["prototype_norm"].append(pnorm)

    return {
        "overall": _bucket_metrics(overall, score_thr),
        "per_source": {k: _bucket_metrics(v, score_thr) for k, v in per_source.items()},
        "tile_cfg": cfg,
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
    img_size: int = 768,
    tile_cfg: dict | None = None,
) -> dict[str, Any]:
    """Zero-shot OWLv2 image-guided detection.

    For each episode, run image-guided detection 4 times (one per support)
    and average the per-patch logits.  Top-1 box → predicted bbox.
    Existence = sigmoid(max averaged logit).

    Tile inference is supported via ``tile_cfg`` — when enabled for an
    episode's source, each tile of the native query is fed to the four
    supports independently and the per-patch logits averaged before the
    top-1 box is selected within that tile.  Cross-tile NMS picks the
    final detection.
    """
    from modeling.loss import _cxcywh_to_xyxy as _to_xyxy
    from modeling._tiling import (
        OWLV2_MEAN, OWLV2_STD,                                # noqa: F401 — used by helpers
        crop_and_normalize, pyramid_tiles, dilate_top_tile,
        _tile_local_cxcywh_to_native_xyxy,
    )
    from torchvision.ops import batched_nms

    owlv2_model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)
    cfg = resolve_tile_cfg(tile_cfg)
    tile_mode = cfg["mode"]
    tile_for_sources = set(cfg["for_sources"])

    def _phase0_logits_for_query(query_tensor: torch.Tensor, supports_v: torch.Tensor):
        """Run image-guided detection on a single query tensor, averaging
        over supports.  Returns (avg_logits, target_pred_boxes)."""
        b_q = query_tensor.size(0)
        q_feature_map, _ = owlv2_model.image_embedder(
            pixel_values=query_tensor, interpolate_pos_encoding=True
        )
        gh, gw, d = q_feature_map.shape[1:]
        image_feats = q_feature_map.reshape(b_q, gh * gw, d)
        target_pred_boxes = owlv2_model.box_predictor(
            image_feats, q_feature_map, interpolate_pos_encoding=True
        )
        v = supports_v.size(1)
        all_logits = []
        for vi in range(v):
            sup_v = supports_v[:, vi]
            sup_feature_map, _ = owlv2_model.image_embedder(
                pixel_values=sup_v, interpolate_pos_encoding=True
            )
            sup_h, sup_w = sup_feature_map.shape[1:3]
            sup_feats = sup_feature_map.reshape(supports_v.size(0), sup_h * sup_w, d)
            q_emb, _, _ = owlv2_model.embed_image_query(
                sup_feats, sup_feature_map, interpolate_pos_encoding=True
            )
            if q_emb is None:
                continue
            if q_emb.dim() == 2:
                q_emb = q_emb.unsqueeze(1)
            logits, _ = owlv2_model.class_predictor(image_feats, q_emb)
            all_logits.append(logits.squeeze(-1))
        if not all_logits:
            return None, None
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return avg_logits, target_pred_boxes

    for batch in loader:
        support_imgs = batch["support_imgs"].to(device)         # (B, 4, 3, S, S)
        query_img = batch["query_img"].to(device)               # (B, 3, S, S)
        gt_bbox = batch["query_bbox"]
        is_present = batch["is_present"]
        sources = batch["source"]
        native_size = batch.get("native_size")
        native_bbox = batch.get("native_bbox")
        query_native_list = batch.get("query_native")

        b = support_imgs.size(0)

        # Single-pass logits + boxes for the whole batch.
        avg_logits, target_pred_boxes = _phase0_logits_for_query(query_img, support_imgs)
        if avg_logits is None:
            continue
        P = avg_logits.size(1)

        for i in range(b):
            src = sources[i]
            present = bool(is_present[i].item())
            do_tile = (
                tile_mode != "off"
                and src in tile_for_sources
                and query_native_list is not None
                and native_size is not None
            )

            if do_tile:
                native_pil = query_native_list[i]
                nw, nh = int(native_size[i, 0].item()), int(native_size[i, 1].item())
                tiles = pyramid_tiles((nw, nh), levels=tuple(cfg["levels"]),
                                      overlap=float(cfg["overlap"]))
                tile_results: list[tuple[torch.Tensor, float, float]] = []  # (xyxy_native, score, ex)

                def _run(tile):
                    tt = crop_and_normalize(native_pil, tile, img_size).unsqueeze(0).to(device)
                    avg_l, t_boxes = _phase0_logits_for_query(tt, support_imgs[i:i + 1])
                    if avg_l is None:
                        return None
                    P_l = avg_l.size(1)
                    bi = avg_l.argmax(dim=-1).clamp(0, P_l - 1)
                    ar = torch.arange(1, device=device)
                    bb_local = t_boxes[ar, bi][0]                 # (4,) cxcywh tile-local
                    sc_logit = float(avg_l[ar, bi][0].item())
                    sc = 1.0 / (1.0 + pow(2.71828, -sc_logit))
                    bx = _tile_local_cxcywh_to_native_xyxy(bb_local, tile)
                    return bx, sc, sc                              # use sigmoid as both score and existence

                for tile in tiles:
                    r = _run(tile)
                    if r is not None:
                        tile_results.append(r)

                if tile_mode == "hybrid_d" and tile_results:
                    best_t = int(max(range(len(tile_results)),
                                     key=lambda j: tile_results[j][1]))
                    sub_tiles = dilate_top_tile(
                        tiles[best_t], (nw, nh), grid=2, overlap=float(cfg["overlap"])
                    )
                    for st in sub_tiles:
                        if st in tiles:
                            continue
                        r = _run(st)
                        if r is not None:
                            tile_results.append(r)

                if not tile_results:
                    iou_v = 0.0
                    contain_v = 0.0
                    ex = 0.0
                    sc = 0.0
                    area = 0.0
                else:
                    boxes_t = torch.stack([t[0] for t in tile_results], dim=0)
                    scores_t = torch.tensor([t[1] for t in tile_results], device=device)
                    classes = torch.zeros_like(scores_t, dtype=torch.long)
                    keep = batched_nms(boxes_t.float(), scores_t.float(), classes,
                                       iou_threshold=float(cfg["nms_iou"]))
                    keep = keep[: int(cfg["top_k"])]
                    boxes_t = boxes_t[keep]
                    scores_t = scores_t[keep]
                    bi = int(scores_t.argmax().item())
                    pred_xyxy_native = boxes_t[bi]
                    pred_xyxy_norm = pred_xyxy_native.clone().float()
                    pred_xyxy_norm[0] /= max(nw, 1)
                    pred_xyxy_norm[2] /= max(nw, 1)
                    pred_xyxy_norm[1] /= max(nh, 1)
                    pred_xyxy_norm[3] /= max(nh, 1)
                    pred_xyxy_norm = pred_xyxy_norm.clamp(0, 1)
                    if native_bbox is not None and present:
                        gt_xyxy_norm = native_bbox[i].float().clone().to(device)
                        gt_xyxy_norm[0] /= max(nw, 1)
                        gt_xyxy_norm[2] /= max(nw, 1)
                        gt_xyxy_norm[1] /= max(nh, 1)
                        gt_xyxy_norm[3] /= max(nh, 1)
                    else:
                        gt_xyxy_norm = torch.zeros(4, device=device)
                    iou_v = float(_iou_xyxy(pred_xyxy_norm, gt_xyxy_norm).item()) if present else 0.0
                    contain_v = float(_containment_ratio(pred_xyxy_norm, gt_xyxy_norm).item()) if present else 0.0
                    ex = float(scores_t[bi].item())
                    sc = ex
                    area = float(
                        (pred_xyxy_norm[2] - pred_xyxy_norm[0])
                        * (pred_xyxy_norm[3] - pred_xyxy_norm[1])
                    )
            else:
                # Single-pass branch.
                bi = int(avg_logits[i].argmax().clamp(0, P - 1).item())
                bb = target_pred_boxes[i, bi]
                pred_xyxy = _to_xyxy(bb).clamp(0, 1)
                gt_xyxy = _to_xyxy(gt_bbox[i].to(device)).clamp(0, 1)
                iou_v = float(_iou_xyxy(pred_xyxy, gt_xyxy).item())
                contain_v = float(_containment_ratio(pred_xyxy, gt_xyxy).item())
                logit = float(avg_logits[i, bi].item())
                ex = 1.0 / (1.0 + pow(2.71828, -logit))
                sc = ex
                area = float(bb[..., 2].clamp(0, 1) * bb[..., 3].clamp(0, 1))

            entry = {
                "iou": iou_v,
                "contain": contain_v,
                "score": sc,
                "is_present": present,
                "pred_box_area": area,
                "existence_prob": ex,
                "prototype_norm": 0.0,
            }
            for bucket in (overall, per_source[src]):
                for k, v_ in entry.items():
                    bucket[k].append(v_)

    return {
        "overall": _bucket_metrics(overall, score_thr),
        "per_source": {k: _bucket_metrics(v, score_thr) for k, v in per_source.items()},
        "tile_cfg": cfg,
    }
