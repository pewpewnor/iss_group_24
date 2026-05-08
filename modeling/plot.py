"""Offline plotting from per-epoch / per-fold JSON files.

Reads:
  analysis/
    config.json
    folds.json
    summary.json
    phase0/results.json                            (zero-shot baseline)
    {stage_1_1,stage_1_2,stage_2_3}/
      epoch_NNN/fold_F.json                        (one per (epoch, fold))
      epoch_NNN/aggregate.json                     (mean/min/max/std across folds)
      complete.json                                (stage-completion marker)

Writes:
  analysis/plots/
    training_curves.png       — losses + val map_50/iou_mean across all stages
    map_per_iou.png           — best-epoch ap_per_iou per stage
    collapse_diagnostics.png  — mean_pred_box_area, mean_existence_prob,
                                  false_positive_rate over time
    per_source_map.png        — per-source map_50 / map_5095 curves
    cv_envelope.png           — mean ± [min, max] envelope across folds per stage
    phase0_summary.png        — bar chart of zero-shot mAP / IoU per dataset

Single public entry: ``plot_all_from_jsons(analysis_dir)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = ("stage_1_1", "stage_1_2", "stage_2_3")
STAGE_LABELS = {"stage_1_1": "Stage 1.1", "stage_1_2": "Stage 1.2", "stage_2_3": "Stage 2.3"}
STAGE_COLORS = {"stage_1_1": "#3b82f6", "stage_1_2": "#22c55e", "stage_2_3": "#f97316"}


# ---------------------------------------------------------------------------
# JSON traversal helpers
# ---------------------------------------------------------------------------


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _safe(d: Any, *keys: str, default: float = 0.0) -> float:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def _stage_epoch_dirs(analysis_dir: Path, stage: str) -> list[Path]:
    p = analysis_dir / stage
    if not p.exists():
        return []
    return sorted(d for d in p.glob("epoch_*") if d.is_dir())


def _epoch_id(d: Path) -> int:
    try:
        return int(d.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _stage_aggregates(analysis_dir: Path, stage: str) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for d in _stage_epoch_dirs(analysis_dir, stage):
        agg = _load_json(d / "aggregate.json")
        if agg is not None:
            out.append((_epoch_id(d), agg))
    return out


def _stage_fold_jsons(analysis_dir: Path, stage: str) -> list[tuple[int, list[dict]]]:
    out: list[tuple[int, list[dict]]] = []
    for d in _stage_epoch_dirs(analysis_dir, stage):
        folds: list[dict] = []
        for f in sorted(d.glob("fold_*.json")):
            j = _load_json(f)
            if j is not None:
                folds.append(j)
        if folds:
            out.append((_epoch_id(d), folds))
    return out


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------


def _plot_training_curves(analysis_dir: Path, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_loss, ax_map, ax_iou, ax_auroc = axes.flatten()

    for stage in STAGES:
        aggs = _stage_aggregates(analysis_dir, stage)
        if not aggs:
            continue
        epochs = [e for e, _ in aggs]
        loss = [_safe(a, "metrics", "train.loss", "mean") for _, a in aggs]
        map50 = [_safe(a, "metrics", "val.overall.map_50", "mean") for _, a in aggs]
        iou = [_safe(a, "metrics", "val.overall.iou_mean", "mean") for _, a in aggs]
        auroc = [_safe(a, "metrics", "val.overall.existence_auroc", "mean") for _, a in aggs]
        c = STAGE_COLORS[stage]
        lbl = STAGE_LABELS[stage]
        ax_loss.plot(epochs, loss, "-o", color=c, label=lbl)
        ax_map.plot(epochs, map50, "-o", color=c, label=lbl)
        ax_iou.plot(epochs, iou, "-o", color=c, label=lbl)
        ax_auroc.plot(epochs, auroc, "-o", color=c, label=lbl)

    for ax, title in zip(
        (ax_loss, ax_map, ax_iou, ax_auroc),
        ("Train loss", "Val mAP@50", "Val IoU mean (positives)", "Val existence AUROC"),
    ):
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def _plot_map_per_iou(analysis_dir: Path, plots_dir: Path) -> None:
    """Best-epoch (per stage) ap_per_iou bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    iou_thresholds: list[str] = []
    bars: dict[str, list[float]] = {}
    for stage in STAGES:
        aggs = _stage_aggregates(analysis_dir, stage)
        if not aggs:
            continue
        # Pick the epoch with highest val.overall.map_50.mean.
        best = max(aggs, key=lambda kv: _safe(kv[1], "metrics", "val.overall.map_50", "mean"))
        metrics = best[1].get("metrics", {})
        # Find ap_per_iou keys ─ they're flattened as val.overall.ap_per_iou.<thr>
        thr_keys = sorted(
            k for k in metrics if k.startswith("val.overall.ap_per_iou.")
        )
        if not thr_keys:
            continue
        iou_thresholds = [k.split(".")[-1] for k in thr_keys]
        bars[stage] = [_safe(metrics, k, "mean") for k in thr_keys]

    if not bars:
        plt.close(fig)
        return

    x = list(range(len(iou_thresholds)))
    for i, (stage, vals) in enumerate(bars.items()):
        ax.bar(
            [xi + (i - 1) * width for xi in x],
            vals,
            width=width,
            color=STAGE_COLORS[stage],
            label=STAGE_LABELS[stage],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(iou_thresholds)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_title("Best-epoch AP@IoU per stage")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "map_per_iou.png", dpi=120)
    plt.close(fig)


def _plot_collapse_diagnostics(analysis_dir: Path, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_area, ax_existence, ax_fpr = axes

    for stage in STAGES:
        aggs = _stage_aggregates(analysis_dir, stage)
        if not aggs:
            continue
        epochs = [e for e, _ in aggs]
        area = [_safe(a, "metrics", "val.overall.mean_pred_box_area", "mean") for _, a in aggs]
        existence = [
            _safe(a, "metrics", "val.overall.mean_existence_prob", "mean") for _, a in aggs
        ]
        fpr = [_safe(a, "metrics", "val.overall.false_positive_rate", "mean") for _, a in aggs]
        c = STAGE_COLORS[stage]
        lbl = STAGE_LABELS[stage]
        ax_area.plot(epochs, area, "-o", color=c, label=lbl)
        ax_existence.plot(epochs, existence, "-o", color=c, label=lbl)
        ax_fpr.plot(epochs, fpr, "-o", color=c, label=lbl)

    ax_area.axhline(0.4, color="red", lw=0.7, ls="--", label="collapse threshold (0.4)")
    ax_existence.axhline(0.9, color="red", lw=0.7, ls="--", label="collapse threshold (0.9)")

    for ax, title, ylim in (
        (ax_area, "Mean predicted box area", (0, 1)),
        (ax_existence, "Mean existence_prob", (0, 1)),
        (ax_fpr, "False-positive rate", (0, 1)),
    ):
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "collapse_diagnostics.png", dpi=120)
    plt.close(fig)


def _plot_per_source_map(analysis_dir: Path, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_50, ax_5095 = axes
    sources = ("hots", "insdet")
    for stage in STAGES:
        aggs = _stage_aggregates(analysis_dir, stage)
        if not aggs:
            continue
        epochs = [e for e, _ in aggs]
        c = STAGE_COLORS[stage]
        for src in sources:
            map50 = [
                _safe(a, "metrics", f"val.per_source.{src}.map_50", "mean") for _, a in aggs
            ]
            map5095 = [
                _safe(a, "metrics", f"val.per_source.{src}.map_5095", "mean") for _, a in aggs
            ]
            ls = "-" if src == "hots" else "--"
            ax_50.plot(epochs, map50, ls, marker="o", color=c,
                       label=f"{STAGE_LABELS[stage]} / {src}")
            ax_5095.plot(epochs, map5095, ls, marker="o", color=c,
                         label=f"{STAGE_LABELS[stage]} / {src}")
    for ax, title in ((ax_50, "Per-source mAP@50"), (ax_5095, "Per-source mAP@50-95")):
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_source_map.png", dpi=120)
    plt.close(fig)


def _plot_cv_envelope(analysis_dir: Path, plots_dir: Path) -> None:
    """For each stage, plot mean (line) and [min, max] envelope across folds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage in STAGES:
        aggs = _stage_aggregates(analysis_dir, stage)
        if not aggs:
            continue
        epochs = [e for e, _ in aggs]
        means = [_safe(a, "metrics", "val.overall.map_50", "mean") for _, a in aggs]
        mins = [_safe(a, "metrics", "val.overall.map_50", "min") for _, a in aggs]
        maxs = [_safe(a, "metrics", "val.overall.map_50", "max") for _, a in aggs]
        c = STAGE_COLORS[stage]
        ax.plot(epochs, means, "-o", color=c, label=f"{STAGE_LABELS[stage]} mean")
        ax.fill_between(epochs, mins, maxs, color=c, alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("Cross-fold mean ± [min, max] of val mAP@50")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_envelope.png", dpi=120)
    plt.close(fig)


def _plot_phase0(analysis_dir: Path, plots_dir: Path) -> None:
    p = analysis_dir / "phase0" / "results.json"
    res = _load_json(p)
    if res is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ("map_50", "map_75", "map_5095", "iou_mean", "existence_auroc")
    datasets = list(res.keys())
    width = 0.15
    x = list(range(len(metrics)))
    for i, ds in enumerate(datasets):
        bucket = res[ds].get("overall", {})
        vals = [float(bucket.get(m, 0.0)) for m in metrics]
        ax.bar(
            [xi + (i - len(datasets) / 2) * width for xi in x],
            vals, width=width, label=ds,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_title("Phase 0 zero-shot baseline")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "phase0_summary.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def plot_all_from_jsons(analysis_dir: str | Path) -> None:
    analysis_dir = Path(analysis_dir)
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_training_curves(analysis_dir, plots_dir)
    _plot_map_per_iou(analysis_dir, plots_dir)
    _plot_collapse_diagnostics(analysis_dir, plots_dir)
    _plot_per_source_map(analysis_dir, plots_dir)
    _plot_cv_envelope(analysis_dir, plots_dir)
    _plot_phase0(analysis_dir, plots_dir)
    print(f"plots written to {plots_dir}")
