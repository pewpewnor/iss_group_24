"""Offline plot generation from per-(epoch, fold) JSONs.

Generates plots to <analysis_root>/plots/{localizer,siamese}/<stage>/<metric>.png

Localizer plots (per stage L1 / L2 / L3):
  - mAP@50 vs epoch (mean ± std across folds, one line per source bucket)
  - IoU mean vs epoch
  - Per-K bar chart at final epoch (K=1, K=4, K=10)

Siamese plots (per stage S1 / S2):
  - AUROC vs epoch
  - FPR @ thr=0.5 vs epoch
  - Loss curves
  - Per-K bar chart of AUROC at final epoch

The plotter is robust to missing files / metrics — it just skips them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _read_fold_jsons(stage_dir: Path) -> list[tuple[int, int, dict]]:
    """Return list of (epoch, fold, payload) for every fold_*.json under epoch_*/."""
    out: list[tuple[int, int, dict]] = []
    if not stage_dir.exists():
        return out
    for ep_dir in sorted(stage_dir.glob("epoch_*")):
        try:
            epoch = int(ep_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        for fold_p in sorted(ep_dir.glob("fold_*.json")):
            try:
                fold = int(fold_p.stem.split("_")[1])
            except (ValueError, IndexError):
                continue
            try:
                with open(fold_p) as f:
                    payload = json.load(f)
                out.append((epoch, fold, payload))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def _series_per_epoch(
    fold_jsons: list[tuple[int, int, dict]],
    val_path: tuple[str, ...],
) -> tuple[list[int], list[float], list[float]]:
    """Mean ± std across folds at each epoch for a metric path."""
    by_epoch: dict[int, list[float]] = {}
    for epoch, _, p in fold_jsons:
        v = _safe_get(p, "val", *val_path)
        if isinstance(v, (int, float)):
            by_epoch.setdefault(epoch, []).append(float(v))
    epochs = sorted(by_epoch.keys())
    means = []
    stds = []
    for e in epochs:
        vs = by_epoch[e]
        m = sum(vs) / len(vs)
        v = sum((x - m) ** 2 for x in vs) / max(len(vs), 1)
        means.append(m)
        stds.append(v ** 0.5)
    return epochs, means, stds


def _plot_metric_curve(
    out_path: Path, *, title: str, ylabel: str,
    series: dict[str, tuple[list[int], list[float], list[float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (xs, ms, sds) in series.items():
        if not xs:
            continue
        ax.plot(xs, ms, marker="o", label=label)
        if any(sds):
            lo = [m - s for m, s in zip(ms, sds)]
            hi = [m + s for m, s in zip(ms, sds)]
            ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_per_k_bar(
    out_path: Path, *, title: str, ylabel: str,
    fold_jsons: list[tuple[int, int, dict]], metric_key: str,
) -> None:
    if not fold_jsons:
        return
    last_epoch = max(e for e, _, _ in fold_jsons)
    rows = [p for e, _, p in fold_jsons if e == last_epoch]
    by_k: dict[str, list[float]] = {}
    for r in rows:
        per_k = _safe_get(r, "val", "per_k", default={})
        if not isinstance(per_k, dict):
            continue
        for k_label, m in per_k.items():
            if isinstance(m, dict) and metric_key in m:
                by_k.setdefault(k_label, []).append(float(m[metric_key]))
    if not by_k:
        return
    labels = sorted(by_k.keys(), key=lambda s: int(s.lstrip("k")))
    values = [sum(by_k[l]) / len(by_k[l]) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color="steelblue")
    ax.set_xlabel("K (number of supports)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_loss_curves(
    out_path: Path, *, fold_jsons: list[tuple[int, int, dict]],
    keys: tuple[str, ...], title: str,
) -> None:
    series: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for k in keys:
        by_epoch: dict[int, list[float]] = {}
        for epoch, _, p in fold_jsons:
            v = _safe_get(p, "train", k)
            if isinstance(v, (int, float)):
                by_epoch.setdefault(epoch, []).append(float(v))
        epochs = sorted(by_epoch.keys())
        means = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]
        stds = [
            (sum((x - m) ** 2 for x in by_epoch[e]) / max(len(by_epoch[e]), 1)) ** 0.5
            for e, m in zip(epochs, means)
        ]
        series[k] = (epochs, means, stds)
    if not any(xs for xs, _, _ in series.values()):
        return
    _plot_metric_curve(out_path, title=title, ylabel="loss", series=series)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_LOCALIZER_STAGES = ("phase0", "L1", "L2", "L3")
_SIAMESE_STAGES = ("phase0", "S1", "S2")


def plot_all_from_jsons(analysis_root: str | Path) -> None:
    analysis_root = Path(analysis_root)
    plots_root = analysis_root / "plots"

    # ---- Localizer ----
    loc_root = analysis_root / "localizer"
    if loc_root.exists():
        for stage in _LOCALIZER_STAGES:
            stage_dir = loc_root / stage
            fjs = _read_fold_jsons(stage_dir)
            if not fjs:
                continue
            out_dir = plots_root / "localizer" / stage
            # mAP@50 vs epoch (overall, hots, insdet).
            series = {
                "overall": _series_per_epoch(fjs, ("overall", "map_50")),
                "hots":    _series_per_epoch(fjs, ("per_source", "hots", "map_50")),
                "insdet":  _series_per_epoch(fjs, ("per_source", "insdet", "map_50")),
            }
            _plot_metric_curve(
                out_dir / "map50_vs_epoch.png",
                title=f"Localizer {stage} — mAP@50 vs epoch",
                ylabel="mAP@50", series=series,
            )
            # IoU mean vs epoch.
            series_iou = {
                "overall": _series_per_epoch(fjs, ("overall", "iou_mean")),
                "hots":    _series_per_epoch(fjs, ("per_source", "hots", "iou_mean")),
                "insdet":  _series_per_epoch(fjs, ("per_source", "insdet", "iou_mean")),
            }
            _plot_metric_curve(
                out_dir / "iou_vs_epoch.png",
                title=f"Localizer {stage} — IoU mean vs epoch",
                ylabel="IoU mean", series=series_iou,
            )
            # Containment metrics vs epoch.
            series_contain = {
                "containment_mean":      _series_per_epoch(fjs, ("overall", "containment_mean")),
                "frac_containment_90":   _series_per_epoch(fjs, ("overall", "frac_containment_90")),
                "frac_containment_full": _series_per_epoch(fjs, ("overall", "frac_containment_full")),
            }
            _plot_metric_curve(
                out_dir / "containment_vs_epoch.png",
                title=f"Localizer {stage} — containment metrics vs epoch",
                ylabel="containment", series=series_contain,
            )
            # Per-K bar chart.
            _plot_per_k_bar(
                out_dir / "map50_by_k.png",
                title=f"Localizer {stage} — mAP@50 by K (final epoch)",
                ylabel="mAP@50", fold_jsons=fjs, metric_key="map_50",
            )
            _plot_per_k_bar(
                out_dir / "containment_by_k.png",
                title=f"Localizer {stage} — containment_mean by K (final epoch)",
                ylabel="containment", fold_jsons=fjs, metric_key="containment_mean",
            )
            # Loss curves.
            _plot_loss_curves(
                out_dir / "loss_curves.png", fold_jsons=fjs,
                keys=("loss", "l1", "giou"),
                title=f"Localizer {stage} — losses",
            )

    # ---- Siamese ----
    sia_root = analysis_root / "siamese"
    if sia_root.exists():
        for stage in _SIAMESE_STAGES:
            stage_dir = sia_root / stage
            fjs = _read_fold_jsons(stage_dir)
            if not fjs:
                continue
            out_dir = plots_root / "siamese" / stage
            series_auroc = {
                "overall": _series_per_epoch(fjs, ("overall", "auroc")),
                "hots":    _series_per_epoch(fjs, ("per_source", "hots", "auroc")),
                "insdet":  _series_per_epoch(fjs, ("per_source", "insdet", "auroc")),
            }
            _plot_metric_curve(
                out_dir / "auroc_vs_epoch.png",
                title=f"Siamese {stage} — AUROC vs epoch",
                ylabel="AUROC", series=series_auroc,
            )
            series_fpr = {
                "overall": _series_per_epoch(fjs, ("overall", "fpr")),
                "hots":    _series_per_epoch(fjs, ("per_source", "hots", "fpr")),
                "insdet":  _series_per_epoch(fjs, ("per_source", "insdet", "fpr")),
            }
            _plot_metric_curve(
                out_dir / "fpr_vs_epoch.png",
                title=f"Siamese {stage} — FPR@0.5 vs epoch",
                ylabel="FPR", series=series_fpr,
            )
            _plot_per_k_bar(
                out_dir / "auroc_by_k.png",
                title=f"Siamese {stage} — AUROC by K (final epoch)",
                ylabel="AUROC", fold_jsons=fjs, metric_key="auroc",
            )
            _plot_loss_curves(
                out_dir / "loss_curves.png", fold_jsons=fjs,
                keys=("loss", "focal", "variance", "decorrelation"),
                title=f"Siamese {stage} — losses",
            )

    print(f"plots: written under {plots_root}")
