"""Siamese evaluation: comprehensive existence-classification metrics.

Reports per (overall, per_source, per_k) bucket:

  Counts:
    n, n_pos, n_neg

  Threshold-dependent (using the user-supplied ``threshold``, default 0.5):
    accuracy, acc_pos, acc_neg
    precision, recall, f1
    fpr, fnr, tpr (= 1 - fnr), tnr (= 1 - fpr)
    youden_j  (tpr - fpr; max=1, random=0)
    mcc       (Matthews correlation coefficient — robust to class imbalance)

  Threshold-free:
    auroc                 (Mann-Whitney U)
    pr_auc                (precision-recall AUC, trapezoidal)
    avg_precision         (interpolated AP @ 101 points)
    brier                 (Brier score: mean squared error vs ground truth)

  Threshold sweep summaries:
    best_f1               (max F1 over ALL operating thresholds)
    best_f1_threshold     (threshold that achieves best_f1)
    fpr_at_recall_95      (FPR when recall = 0.95)
    recall_at_fpr_05      (recall when FPR = 0.05)
    recall_at_fpr_10      (recall when FPR = 0.10)

  Score distribution:
    mean_score_pos / neg, std_score_pos / neg
    score_gap                   (mean_score_pos - mean_score_neg)
    median_score_pos / neg
    frac_high_score             (score > 0.9)
    frac_low_score              (score < 0.1)
    frac_uncertain              (0.4 <= score < 0.6)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

from siamese.model import MultiShotSiamese


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Threshold-free classification metrics
# ---------------------------------------------------------------------------


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
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, combined) if y == 1)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _binary_pr_auc(scores: list[float], labels: list[bool]) -> float:
    """Trapezoidal area under the precision-recall curve."""
    if not labels or not any(labels):
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    precs: list[float] = []
    recs: list[float] = []
    n_pos = sum(labels)
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / n_pos)
    auc = 0.0
    prev_r = 0.0
    for p, r in zip(precs, recs):
        auc += p * max(0.0, r - prev_r)
        prev_r = r
    return auc


def _avg_precision_101(scores: list[float], labels: list[bool]) -> float:
    """101-point interpolated AP (matches the localizer's mAP convention)."""
    if not labels or not any(labels):
        return 0.0
    n_pos = sum(labels)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    precs: list[float] = []
    recs: list[float] = []
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / n_pos)
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


def _best_f1(scores: list[float], labels: list[bool]) -> tuple[float, float]:
    """Return (best_f1, threshold_at_best_f1) over all sweep points."""
    if not labels or not any(labels):
        return 0.0, 0.5
    n_pos = sum(labels)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    best_f1 = 0.0
    best_thr = 1.0
    last_score: float | None = None
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        s = scores[i]
        # Only evaluate F1 at strict score-change boundaries (correct sweep).
        if last_score is None or s != last_score:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(n_pos, 1)
            denom = prec + rec
            f1 = (2 * prec * rec / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = s
            last_score = s
    return best_f1, best_thr


def _fpr_at_recall(scores: list[float], labels: list[bool], target_recall: float) -> float:
    """FPR (= FP / N_neg) at the lowest threshold that achieves >= target_recall.

    Returns 1.0 if target_recall is unreachable.
    """
    if not labels or not any(labels):
        return 1.0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_neg == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= target_recall:
            return fp / n_neg
    return 1.0


def _recall_at_fpr(scores: list[float], labels: list[bool], target_fpr: float) -> float:
    """Highest recall achievable while FPR <= target_fpr."""
    if not labels or not any(labels):
        return 0.0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 0.0
    if n_neg == 0:
        return 1.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    best_recall = 0.0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        if fp / n_neg <= target_fpr:
            best_recall = max(best_recall, tp / n_pos)
    return best_recall


def _matthews_corrcoef(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq <= 0:
        return 0.0
    return num / math.sqrt(den_sq)


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------


def _empty_bucket() -> dict[str, list]:
    return {"score": [], "is_present": []}


def _bucket_metrics(b: dict[str, list], thr: float = 0.5) -> dict[str, Any]:
    n = len(b["score"])
    n_pos = sum(b["is_present"])
    n_neg = n - n_pos
    out: dict[str, Any] = {"n": n, "n_pos": n_pos, "n_neg": n_neg, "threshold": thr}
    if n == 0:
        return out

    scores = b["score"]
    labels = b["is_present"]

    # ── Threshold-dependent ──────────────────────────────────────────────
    pred_pos = [s > thr for s in scores]
    tp = sum(1 for pp, p in zip(pred_pos, labels) if pp and p)
    fp = sum(1 for pp, p in zip(pred_pos, labels) if pp and not p)
    fn = sum(1 for pp, p in zip(pred_pos, labels) if (not pp) and p)
    tn = sum(1 for pp, p in zip(pred_pos, labels) if (not pp) and not p)

    out["tp"] = tp
    out["fp"] = fp
    out["fn"] = fn
    out["tn"] = tn

    out["accuracy"] = (tp + tn) / max(n, 1)
    out["acc_pos"]  = tp / max(n_pos, 1) if n_pos else 0.0
    out["acc_neg"]  = tn / max(n_neg, 1) if n_neg else 0.0

    out["precision"] = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    out["recall"]    = tp / max(n_pos, 1) if n_pos else 0.0
    denom = out["precision"] + out["recall"]
    out["f1"]        = (2 * out["precision"] * out["recall"] / denom) if denom > 0 else 0.0

    out["fpr"] = fp / max(n_neg, 1) if n_neg else 0.0
    out["fnr"] = fn / max(n_pos, 1) if n_pos else 0.0
    out["tpr"] = 1.0 - out["fnr"]
    out["tnr"] = 1.0 - out["fpr"]
    out["youden_j"] = out["tpr"] - out["fpr"]
    out["mcc"] = _matthews_corrcoef(tp, fp, fn, tn)

    # ── Threshold-free ──────────────────────────────────────────────────
    out["auroc"]         = _binary_auroc(scores, labels)
    out["pr_auc"]        = _binary_pr_auc(scores, labels)
    out["avg_precision"] = _avg_precision_101(scores, labels)
    out["brier"]         = _safe_mean(
        [(s - (1.0 if p else 0.0)) ** 2 for s, p in zip(scores, labels)]
    )

    # ── Threshold sweep summaries ───────────────────────────────────────
    best_f1, best_thr = _best_f1(scores, labels)
    out["best_f1"]            = best_f1
    out["best_f1_threshold"]  = best_thr
    out["fpr_at_recall_95"]   = _fpr_at_recall(scores, labels, 0.95)
    out["recall_at_fpr_05"]   = _recall_at_fpr(scores, labels, 0.05)
    out["recall_at_fpr_10"]   = _recall_at_fpr(scores, labels, 0.10)

    # ── Score distribution ──────────────────────────────────────────────
    pos_scores = [s for s, p in zip(scores, labels) if p]
    neg_scores = [s for s, p in zip(scores, labels) if not p]
    out["mean_score_pos"]  = _safe_mean(pos_scores)
    out["mean_score_neg"]  = _safe_mean(neg_scores)
    out["std_score_pos"]   = _safe_std(pos_scores)
    out["std_score_neg"]   = _safe_std(neg_scores)
    out["score_gap"]       = out["mean_score_pos"] - out["mean_score_neg"]
    out["median_score_pos"] = _quantile(pos_scores, 0.5)
    out["median_score_neg"] = _quantile(neg_scores, 0.5)
    out["frac_high_score"] = sum(1 for s in scores if s > 0.9) / n
    out["frac_low_score"]  = sum(1 for s in scores if s < 0.1) / n
    out["frac_uncertain"]  = sum(1 for s in scores if 0.4 <= s < 0.6) / n

    return out


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: MultiShotSiamese,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float | str = 0.5,
    progress: bool = True,
    progress_every: int = 5,
    phase0: bool = False,
    save_scores: bool = True,
) -> dict[str, Any]:
    """Evaluate the siamese on ``loader``.

    threshold:
      float — pin operating-point metrics (precision/recall/f1/fpr/fnr) at
              this threshold.
      "auto" — sweep all scores, pick the OVERALL ``best_f1_threshold`` and
              re-bucket every per-source / per-k slice at that single value.
              This is the operating-point metric the user wants reported for
              the val-pinned F1 / precision / recall. ``best_f1_threshold``
              is also returned as a field so the caller can persist it.

    save_scores: when True, the per-bucket score + label arrays are returned
                 under ``confusion_matrix`` so plots / audits can rebuild the
                 ROC, PR curve, and confusion matrix at any threshold.
    """
    model.eval()
    overall = _empty_bucket()
    per_source: dict[str, dict[str, list]] = defaultdict(_empty_bucket)
    per_k: dict[str, dict[str, list]] = defaultdict(_empty_bucket)
    # Also keep per-instance score + label for confusion-matrix persistence.
    cm_records: list[dict[str, Any]] = []

    n_batches_total = len(loader) if hasattr(loader, "__len__") else None
    t_start = time.time()
    if progress:
        print(f"  evaluating: {n_batches_total or '?'} batches", flush=True)

    n_seen = 0
    for batch in loader:
        sup = batch["support_imgs"].to(device)
        sup_mask = batch["support_mask"].to(device)
        qry = batch["query_img"].to(device)
        is_present = batch["is_present"].cpu()
        sources = batch["source"]
        ks = batch["k"]
        instance_ids = batch.get("instance_id", [""] * sup.size(0))
        B = sup.size(0)

        if phase0:
            out = model.phase0_forward(sup, sup_mask, qry)
        else:
            out = model(sup, sup_mask, qry)
        scores = out["existence_prob"].cpu()

        for i in range(B):
            sc = float(scores[i])
            pres = bool(is_present[i].item())
            src = sources[i]
            k_v = int(ks[i].item())
            k_label = f"k{k_v}"
            for bucket in (overall, per_source[src], per_k[k_label]):
                bucket["score"].append(sc)
                bucket["is_present"].append(pres)
            if save_scores:
                cm_records.append({
                    "instance_id": instance_ids[i] if i < len(instance_ids) else "",
                    "source": src,
                    "k": k_v,
                    "score": sc,
                    "is_present": pres,
                })

        n_seen += 1
        if progress and (n_seen % progress_every == 0 or n_seen == n_batches_total):
            elapsed = time.time() - t_start
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  [{n_seen}/{n_batches_total or '?'}]  "
                  f"elapsed={elapsed:5.1f}s  rate={rate:.2f}b/s", flush=True)

    # Resolve threshold. "auto" ⇒ overall best_f1_threshold.
    if isinstance(threshold, str):
        if threshold.lower() != "auto":
            raise ValueError(f"threshold must be float or 'auto', got {threshold!r}")
        _, auto_thr = _best_f1(overall["score"], overall["is_present"])
        thr_used = float(auto_thr)
        thr_mode = "auto"
    else:
        thr_used = float(threshold)
        thr_mode = "fixed"

    out: dict[str, Any] = {
        "overall":    _bucket_metrics(overall, thr=thr_used),
        "per_source": {s: _bucket_metrics(b, thr=thr_used) for s, b in per_source.items()},
        "per_k":      {k: _bucket_metrics(b, thr=thr_used) for k, b in per_k.items()},
        "threshold_used": thr_used,
        "threshold_mode": thr_mode,
    }
    if save_scores:
        out["confusion_matrix"] = {
            "threshold": thr_used,
            "tp": out["overall"].get("tp", 0),
            "fp": out["overall"].get("fp", 0),
            "fn": out["overall"].get("fn", 0),
            "tn": out["overall"].get("tn", 0),
            "n": out["overall"].get("n", 0),
            "n_pos": out["overall"].get("n_pos", 0),
            "n_neg": out["overall"].get("n_neg", 0),
            # Raw per-episode records for offline curve / threshold sweeps.
            "records": cm_records,
        }
    return out
