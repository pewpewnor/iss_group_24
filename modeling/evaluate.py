"""Evaluation pipeline for the few-shot Siamese localiser.

Outputs the kitchen-sink metric set used both at training-time validation
(per epoch / per fold JSON) and at the post-training test evaluation:

  Localisation:
    iou_mean / iou_median / iou_p25 / iou_p75
    contain_mean / contain_at_iou_50 / contain_at_iou_75

  mAP:
    map_50, map_5095, ap_per_iou (10 thresholds 0.50:0.05:0.95)
    f1_50, precision_50, recall_50

  Presence (image-level):
    presence_acc / _pos / _neg
    mean_score_pos / mean_score_neg
    presence_auroc, presence_pr_auc, presence_brier

  Collapse diagnostics:
    frac_pred_near_corner, frac_pred_tiny_box
    argmax_cell_entropy, conf_map_mean/std_pos/neg
    support_proto_norm_mean / std

  per_source: same metric set bucketed by inst.source.

Inference uses ``decode_topk`` (top-K + NMS, presence-gated). Two-scale TTA
(224 + 288) merges decoded boxes via additional NMS.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

from modeling.dataset import EpisodeDataset, collate
from modeling.loss import _containment_ratio, _iou_xyxy
from modeling.model import FewShotLocalizer, decode_topk


IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


# ---------------------------------------------------------------------------
# Top-K AP utilities
# ---------------------------------------------------------------------------


def _detections_to_pr_ap(
    detections: list[tuple[float, bool]], n_gt: int
) -> float:
    """Return COCO-style 101-point AP from a flat list of (score, is_tp).

    n_gt is the total positive count. Detections is a list across all images
    (each image may contribute multiple detections after top-K + NMS).
    """
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
    """ROC-AUC by Mann-Whitney U-statistic."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return 0.0
    n_pos = len(pos)
    n_neg = len(neg)
    # Combine, rank, average ranks for ties.
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
    tp = 0
    fp = 0
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
    # Riemann-sum AUC under the PR curve.
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


def _entropy_nats(probs: list[float]) -> float:
    if not probs:
        return 0.0
    s = sum(probs)
    if s <= 0:
        return 0.0
    return -sum((p / s) * math.log((p / s) + 1e-12) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------


def _resize_query(query_img: torch.Tensor, target: int) -> torch.Tensor:
    if query_img.shape[-1] == target and query_img.shape[-2] == target:
        return query_img
    return F.interpolate(query_img, size=(target, target), mode="bilinear", align_corners=False)


def _scale_boxes(boxes: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    if src == dst or boxes.numel() == 0:
        return boxes
    s = dst / float(src)
    return boxes * s


def _hflip_boxes(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Flip xyxy boxes horizontally about an image of size ``img_size``."""
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    flipped[:, 0] = img_size - boxes[:, 2]
    flipped[:, 2] = img_size - boxes[:, 0]
    return flipped


@torch.no_grad()
def predict_topk_tta(
    model: FewShotLocalizer,
    support_imgs: torch.Tensor,
    query_img: torch.Tensor,
    base_size: int = 224,
    tta_sizes: tuple[int, ...] = (224, 288),
    top_k: int = 100,
    conf_thr: float = 0.05,
    nms_iou: float = 0.5,
    use_hflip: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
    """Multi-scale + horizontal-flip TTA. Decode at each (scale, flip) combo,
    transform boxes back to ``base_size`` un-flipped coords, NMS-merge.

    Args:
        use_hflip: if True, run a second pass with the query (and supports for
            consistency) horizontally flipped, then mirror the boxes back.
    """
    diags: dict[str, Any] = {"presence_score": None, "conf_map_p4": None,
                             "argmax_cell": None, "support_proto_norm": None}

    all_boxes: list[list[torch.Tensor]] = []
    all_scores: list[list[torch.Tensor]] = []

    for ti, sz in enumerate(tta_sizes):
        q = _resize_query(query_img, sz)
        out = model(support_imgs, q)
        if ti == 0:
            diags["presence_score"] = torch.sigmoid(out["presence_logit"]).detach().cpu()
            cp4 = out["conf_p4"].detach().sigmoid().cpu()
            diags["conf_map_p4"] = cp4
            B, _, H, W = cp4.shape
            flat = cp4.view(B, -1)
            diags["argmax_cell"] = flat.argmax(dim=1)
            diags["support_proto_norm"] = out["prototype"].detach().norm(dim=-1).cpu()
        boxes_pi, scores_pi = decode_topk(
            out, img_size=sz, top_k=top_k, conf_thr=conf_thr, nms_iou=nms_iou
        )
        boxes_pi = [_scale_boxes(b, src=sz, dst=base_size) for b in boxes_pi]
        all_boxes.append(boxes_pi)
        all_scores.append(scores_pi)

        if use_hflip:
            q_flip = torch.flip(q, dims=(-1,))
            support_flip = torch.flip(support_imgs, dims=(-1,))
            out_f = model(support_flip, q_flip)
            boxes_pi_f, scores_pi_f = decode_topk(
                out_f, img_size=sz, top_k=top_k, conf_thr=conf_thr, nms_iou=nms_iou
            )
            # Mirror boxes back to the un-flipped frame, then scale to base_size.
            boxes_pi_f = [_hflip_boxes(b, sz) for b in boxes_pi_f]
            boxes_pi_f = [_scale_boxes(b, src=sz, dst=base_size) for b in boxes_pi_f]
            all_boxes.append(boxes_pi_f)
            all_scores.append(scores_pi_f)

    B = len(all_boxes[0])
    out_boxes: list[torch.Tensor] = []
    out_scores: list[torch.Tensor] = []
    n_passes = len(all_boxes)
    for i in range(B):
        bx = torch.cat([all_boxes[t][i] for t in range(n_passes)], dim=0)
        sc = torch.cat([all_scores[t][i] for t in range(n_passes)], dim=0)
        if bx.numel() == 0:
            out_boxes.append(bx)
            out_scores.append(sc)
            continue
        cls = torch.zeros_like(sc, dtype=torch.long)
        keep = batched_nms(bx, sc, cls, iou_threshold=nms_iou)
        bx = bx[keep][:top_k]
        sc = sc[keep][:top_k]
        out_boxes.append(bx)
        out_scores.append(sc)
    return out_boxes, out_scores, diags


# ---------------------------------------------------------------------------
# Aggregation buckets
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict[str, list]:
    return {
        "iou": [],                      # IoU of top-1 box vs GT (present episodes only)
        "contain": [],                  # containment of top-1 box (present only)
        "presence_score": [],           # σ(presence_logit) per image
        "is_present": [],               # bool per image
        "topk_detections": [],          # list of (score, is_tp_at_iou_thresholds_dict)
        "frac_corner": [],
        "frac_tiny": [],
        "argmax_cell_entropy_running": [],
        "conf_mean_pos": [],
        "conf_mean_neg": [],
        "conf_std_pos": [],
        "conf_std_neg": [],
        "support_proto_norms": [],
    }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: FewShotLocalizer,
    loader: DataLoader,
    device: torch.device,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
    img_size: int = 224,
    use_tta: bool = True,
    tta_sizes: tuple[int, ...] = (224, 288),
    use_hflip: bool = True,
    top_k: int = 100,
    conf_thr: float = 0.03,
    nms_iou: float = 0.5,
    near_corner_px: int = 16,
    tiny_box_frac: float = 0.05,
) -> dict:
    """Run evaluation and return the kitchen-sink metric dict.

    The returned dict has top-level keys:
      ``overall``      — summary across all episodes.
      ``per_source``   — same summary bucketed by source.
      ``score_thr`` / ``iou_thr`` — echoed config.
    """
    model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)
    # Accumulator for the (cy, cx) histogram entropy (single across the run).
    argmax_hist: dict[int, int] = {}

    for batch in loader:
        support_imgs = batch["support_imgs"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)
        sources = batch.get("source", ["unknown"] * gt_bbox.shape[0])

        if use_tta:
            boxes_pi, scores_pi, diags = predict_topk_tta(
                model,
                support_imgs,
                query_img,
                base_size=img_size,
                tta_sizes=tta_sizes,
                top_k=top_k,
                conf_thr=conf_thr,
                nms_iou=nms_iou,
                use_hflip=use_hflip,
            )
        else:
            out = model(support_imgs, query_img)
            boxes_pi, scores_pi = decode_topk(
                out,
                img_size=img_size,
                top_k=top_k,
                conf_thr=conf_thr,
                nms_iou=nms_iou,
            )
            diags = {
                "presence_score": torch.sigmoid(out["presence_logit"]).detach().cpu(),
                "conf_map_p4": out["conf_p4"].detach().sigmoid().cpu(),
                "argmax_cell": out["conf_p4"].view(out["conf_p4"].shape[0], -1).argmax(dim=1).cpu(),
                "support_proto_norm": out["prototype"].detach().norm(dim=-1).cpu(),
            }

        B = gt_bbox.shape[0]
        presence_score = diags["presence_score"]
        conf_map = diags["conf_map_p4"]                                              # (B,1,H,W)
        argmax_cell = diags["argmax_cell"]
        proto_norms = diags["support_proto_norm"]

        for i in range(B):
            present = bool(is_present[i].item())
            score = float(presence_score[i].item())
            src = sources[i] if i < len(sources) else "unknown"

            # Top-1 box for IoU/contain metrics.
            if boxes_pi[i].numel() > 0:
                box1 = boxes_pi[i][0:1]
                iou_v = float(_iou_xyxy(box1[0], gt_bbox[i]).item()) if present else 0.0
                contain_v = float(_containment_ratio(box1[0], gt_bbox[i]).item()) if present else 0.0
            else:
                iou_v = 0.0
                contain_v = 0.0

            # Detection-list for AP: every NMS-survivor with its (score, is_tp_dict).
            det_entries: list[tuple[float, dict[float, bool]]] = []
            n_dets = boxes_pi[i].shape[0]
            for j in range(n_dets):
                sc_j = float(scores_pi[i][j].item())
                if present:
                    iou_ij = float(_iou_xyxy(boxes_pi[i][j], gt_bbox[i]).item())
                    is_tp = {tau: iou_ij >= tau for tau in IOU_THRESHOLDS}
                else:
                    is_tp = {tau: False for tau in IOU_THRESHOLDS}
                det_entries.append((sc_j, is_tp))

            # Collapse diagnostics — top-1 box only.
            if boxes_pi[i].numel() > 0:
                bx = boxes_pi[i][0]
                cx = float(((bx[0] + bx[2]) * 0.5).item())
                cy = float(((bx[1] + bx[3]) * 0.5).item())
                near_corner = (
                    cx <= near_corner_px or cy <= near_corner_px
                    or cx >= img_size - near_corner_px or cy >= img_size - near_corner_px
                )
                area = max(0.0, float((bx[2] - bx[0]).item())) * max(
                    0.0, float((bx[3] - bx[1]).item())
                )
                tiny = area < (tiny_box_frac * img_size * img_size)
            else:
                near_corner = False
                tiny = False

            cm = conf_map[i, 0]                                                       # (H, W)
            cm_mean = float(cm.mean().item())
            cm_std = float(cm.std().item())

            # Argmax cell histogram (combined across sources/folds).
            ac = int(argmax_cell[i].item())
            argmax_hist[ac] = argmax_hist.get(ac, 0) + 1

            for bucket in (overall, per_source[src]):
                bucket["iou"].append(iou_v)
                bucket["contain"].append(contain_v)
                bucket["presence_score"].append(score)
                bucket["is_present"].append(present)
                bucket["topk_detections"].append(det_entries)
                bucket["frac_corner"].append(1.0 if near_corner else 0.0)
                bucket["frac_tiny"].append(1.0 if tiny else 0.0)
                if present:
                    bucket["conf_mean_pos"].append(cm_mean)
                    bucket["conf_std_pos"].append(cm_std)
                else:
                    bucket["conf_mean_neg"].append(cm_mean)
                    bucket["conf_std_neg"].append(cm_std)
                bucket["support_proto_norms"].append(float(proto_norms[i].item()))

    def _summarise(b: dict[str, list]) -> dict:
        n = len(b["is_present"])
        if n == 0:
            return {}
        n_pos = sum(b["is_present"])
        n_neg = n - n_pos

        # Localisation
        ious_pos = [v for v, p in zip(b["iou"], b["is_present"]) if p]
        contains_pos = [v for v, p in zip(b["contain"], b["is_present"]) if p]
        iou_mean = sum(ious_pos) / max(len(ious_pos), 1)
        contain_mean = sum(contains_pos) / max(len(contains_pos), 1)
        contain_at_50 = sum(
            1 for v, c, p in zip(b["iou"], b["contain"], b["is_present"]) if p and v >= 0.5
        ) / max(n_pos, 1)
        contain_at_75 = sum(
            1 for v, c, p in zip(b["iou"], b["contain"], b["is_present"]) if p and v >= 0.75
        ) / max(n_pos, 1)

        # mAP via flattened detection list across episodes.
        ap_per_iou: dict[str, float] = {}
        for tau in IOU_THRESHOLDS:
            flat: list[tuple[float, bool]] = []
            for episode_dets in b["topk_detections"]:
                for sc, is_tp in episode_dets:
                    flat.append((sc, is_tp[tau]))
            ap_per_iou[f"{tau:.2f}"] = round(_detections_to_pr_ap(flat, n_pos), 4)
        map_5095 = sum(ap_per_iou.values()) / len(ap_per_iou)
        map_50 = ap_per_iou["0.50"]

        # F1 / precision / recall at IoU=0.5 — compute over top-1 detection per image.
        tp_50 = sum(1 for v, p in zip(b["iou"], b["is_present"]) if p and v >= 0.5)
        fp_50 = sum(1 for v, p in zip(b["iou"], b["is_present"]) if not p and False) + sum(
            1 for v, p, ps in zip(b["iou"], b["is_present"], b["presence_score"])
            if (not p) and ps >= 0.5
        )
        fn_50 = sum(1 for v, p in zip(b["iou"], b["is_present"]) if p and v < 0.5)
        prec_50 = tp_50 / max(tp_50 + fp_50, 1)
        rec_50 = tp_50 / max(n_pos, 1)
        f1_50 = 2 * prec_50 * rec_50 / max(prec_50 + rec_50, 1e-6) if (prec_50 + rec_50) > 0 else 0.0

        # Presence
        pres_correct = sum(((s >= 0.5) == p) for s, p in zip(b["presence_score"], b["is_present"]))
        pres_acc_pos = sum(
            1 for s, p in zip(b["presence_score"], b["is_present"]) if p and s >= 0.5
        ) / max(n_pos, 1)
        pres_acc_neg = sum(
            1 for s, p in zip(b["presence_score"], b["is_present"]) if not p and s < 0.5
        ) / max(n_neg, 1)
        mean_score_pos = sum(
            s for s, p in zip(b["presence_score"], b["is_present"]) if p
        ) / max(n_pos, 1)
        mean_score_neg = sum(
            s for s, p in zip(b["presence_score"], b["is_present"]) if not p
        ) / max(n_neg, 1)
        auroc = _binary_auroc(b["presence_score"], b["is_present"])
        pr_auc = _binary_pr_auc(b["presence_score"], b["is_present"])
        brier = sum(
            (s - (1.0 if p else 0.0)) ** 2 for s, p in zip(b["presence_score"], b["is_present"])
        ) / max(n, 1)

        # Collapse diags
        frac_corner = sum(b["frac_corner"]) / max(n, 1)
        frac_tiny = sum(b["frac_tiny"]) / max(n, 1)

        proto_norms = b["support_proto_norms"]
        proto_mean = sum(proto_norms) / max(len(proto_norms), 1)
        proto_std = (
            (sum((x - proto_mean) ** 2 for x in proto_norms) / len(proto_norms)) ** 0.5
            if proto_norms
            else 0.0
        )

        return {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "iou_mean": round(iou_mean, 4),
            "iou_median": round(_quantile(ious_pos, 0.5), 4),
            "iou_p25": round(_quantile(ious_pos, 0.25), 4),
            "iou_p75": round(_quantile(ious_pos, 0.75), 4),
            "contain_mean": round(contain_mean, 4),
            "contain_at_iou_50": round(contain_at_50, 4),
            "contain_at_iou_75": round(contain_at_75, 4),
            "map_50": round(map_50, 4),
            "map_5095": round(map_5095, 4),
            "ap_per_iou": ap_per_iou,
            "f1_50": round(f1_50, 4),
            "precision_50": round(prec_50, 4),
            "recall_50": round(rec_50, 4),
            "presence_acc": round(pres_correct / n, 4),
            "presence_acc_pos": round(pres_acc_pos, 4),
            "presence_acc_neg": round(pres_acc_neg, 4),
            "mean_score_pos": round(mean_score_pos, 4),
            "mean_score_neg": round(mean_score_neg, 4),
            "presence_auroc": round(auroc, 4),
            "presence_pr_auc": round(pr_auc, 4),
            "presence_brier": round(brier, 4),
            "frac_pred_near_corner": round(frac_corner, 4),
            "frac_pred_tiny_box": round(frac_tiny, 4),
            "conf_map_mean_pos": round(
                sum(b["conf_mean_pos"]) / max(len(b["conf_mean_pos"]), 1), 4
            ),
            "conf_map_mean_neg": round(
                sum(b["conf_mean_neg"]) / max(len(b["conf_mean_neg"]), 1), 4
            ),
            "conf_map_std_pos": round(
                sum(b["conf_std_pos"]) / max(len(b["conf_std_pos"]), 1), 4
            ),
            "conf_map_std_neg": round(
                sum(b["conf_std_neg"]) / max(len(b["conf_std_neg"]), 1), 4
            ),
            "support_proto_norm_mean": round(proto_mean, 4),
            "support_proto_norm_std": round(proto_std, 4),
        }

    overall_summary = _summarise(overall)
    overall_summary["argmax_cell_entropy"] = round(
        _entropy_nats(list(argmax_hist.values())), 4
    )

    return {
        "overall": overall_summary,
        "per_source": {k: _summarise(v) for k, v in per_source.items()},
        "score_thr": score_thr,
        "iou_thr": iou_thr,
        "img_size": img_size,
        "tta_sizes": list(tta_sizes) if use_tta else [img_size],
    }


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run(
    checkpoint: str | Path,
    manifest: str | Path = "dataset/cleaned/manifest.json",
    split: str = "test",
    data_root: str | Path | None = None,
    episodes: int = 600,
    batch_size: int = 8,
    num_workers: int = 2,
    neg_prob: float = 0.5,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
    seed: int = 42,
    device: str | None = None,
    report: str | Path = "analysis/test_report.json",
    use_tta: bool = True,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model = FewShotLocalizer(pretrained=False).to(device_t)
    state = torch.load(str(checkpoint), map_location=device_t, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)

    ds = EpisodeDataset(
        manifest_path=str(manifest),
        split=split,
        data_root=str(data_root) if data_root else None,
        episodes_per_epoch=episodes,
        train=False,
        augment=False,
        seed=seed,
    )
    ds.set_neg_prob(neg_prob)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device_t.type == "cuda"),
    )

    result = evaluate(
        model,
        loader,
        device_t,
        score_thr=score_thr,
        iou_thr=iou_thr,
        use_tta=use_tta,
    )
    out_path = Path(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"report -> {out_path}")
    print(json.dumps(result["overall"], indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="dataset/cleaned/manifest.json")
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", default=None)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--neg-prob", type=float, default=0.5)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--report", default="analysis/test_report.json")
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args()

    run(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        split=args.split,
        data_root=args.data_root,
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        neg_prob=args.neg_prob,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        seed=args.seed,
        device=args.device,
        report=args.report,
        use_tta=not args.no_tta,
    )


if __name__ == "__main__":
    main()
