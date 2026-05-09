"""Siamese evaluation: AUROC, FPR, accuracy, brier, etc.

Reports (overall + per_source + per_k):
    n, n_pos, n_neg
    accuracy / acc_pos / acc_neg
    fpr / fnr (at threshold 0.5)
    auroc / pr_auc / brier
    mean_score_pos / mean_score_neg

The user explicitly asked: prioritise reducing false positives. We therefore
expose ``fpr`` prominently in the priority list so the trainer's early-stopping
logic can monitor it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

from siamese.model import MultiShotSiamese


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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


def _empty_bucket() -> dict[str, list]:
    return {"score": [], "is_present": []}


def _bucket_metrics(b: dict[str, list], thr: float = 0.5) -> dict[str, float]:
    n = len(b["score"])
    n_pos = sum(b["is_present"])
    n_neg = n - n_pos
    out: dict[str, float] = {"n": n, "n_pos": n_pos, "n_neg": n_neg}
    if n == 0:
        return out
    pred_pos = [s > thr for s in b["score"]]
    correct = [pp == p for pp, p in zip(pred_pos, b["is_present"])]
    out["accuracy"] = _safe_mean([1.0 if c else 0.0 for c in correct])
    out["acc_pos"] = _safe_mean([1.0 if pp else 0.0 for pp, p in zip(pred_pos, b["is_present"]) if p]) if n_pos else 0.0
    out["acc_neg"] = _safe_mean([1.0 if not pp else 0.0 for pp, p in zip(pred_pos, b["is_present"]) if not p]) if n_neg else 0.0
    out["fpr"] = sum(1 for pp, p in zip(pred_pos, b["is_present"]) if pp and not p) / max(n_neg, 1)
    out["fnr"] = sum(1 for pp, p in zip(pred_pos, b["is_present"]) if (not pp) and p) / max(n_pos, 1)
    out["auroc"] = _binary_auroc(b["score"], b["is_present"])
    out["pr_auc"] = _binary_pr_auc(b["score"], b["is_present"])
    out["brier"] = _safe_mean([(s - (1.0 if p else 0.0)) ** 2
                                for s, p in zip(b["score"], b["is_present"])])
    out["mean_score_pos"] = _safe_mean([s for s, p in zip(b["score"], b["is_present"]) if p]) if n_pos else 0.0
    out["mean_score_neg"] = _safe_mean([s for s, p in zip(b["score"], b["is_present"]) if not p]) if n_neg else 0.0
    return out


@torch.no_grad()
def evaluate(
    model: MultiShotSiamese,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    progress: bool = True,
    progress_every: int = 5,
    phase0: bool = False,
) -> dict[str, Any]:
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
        is_present = batch["is_present"].cpu()
        sources = batch["source"]
        ks = batch["k"]
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
            k_label = f"k{int(ks[i].item())}"
            for bucket in (overall, per_source[src], per_k[k_label]):
                bucket["score"].append(sc)
                bucket["is_present"].append(pres)

        n_seen += 1
        if progress and (n_seen % progress_every == 0 or n_seen == n_batches_total):
            elapsed = time.time() - t_start
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  [{n_seen}/{n_batches_total or '?'}]  "
                  f"elapsed={elapsed:5.1f}s  rate={rate:.2f}b/s", flush=True)

    return {
        "overall": _bucket_metrics(overall, thr=threshold),
        "per_source": {s: _bucket_metrics(b, thr=threshold) for s, b in per_source.items()},
        "per_k": {k: _bucket_metrics(b, thr=threshold) for k, b in per_k.items()},
    }
