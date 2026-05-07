"""Offline plotting from per-epoch / per-fold JSON files.

Reads:
  analysis/
    config.json
    folds.json
    stage1/epoch_*.json (+ complete.json)
    stage2/epoch_*.json (+ complete.json)
    stage3/epoch_*/fold_*.json (+ aggregate.json) (+ complete.json)
    summary.json
    test_report.json (optional, post-training)

Writes:
  analysis/plots/
    training_curves.png       — losses + val map_50/iou over the full timeline.
    map_per_iou.png           — final-epoch ap_per_iou bar chart per stage.
    collapse_diagnostics.png  — mean_score_neg / frac_corner / argmax entropy
                                 / proto norm over the timeline.
    per_source_map.png        — per-source map_50/map_5095 curves.
    cv_envelope.png           — Stage 3: mean ± min/max envelope across folds.
    eval_report.png           — final test_report.json summary.

Single entry point: ``plot_all_from_jsons(analysis_dir)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _safe_get(d: Any, *keys, default: float = 0.0) -> float:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def _stage_epoch_files(analysis_dir: Path, stage: str) -> list[tuple[int, dict]]:
    p = analysis_dir / stage
    if not p.exists():
        return []
    out = []
    for f in sorted(p.glob("epoch_*.json")):
        try:
            out.append((int(f.stem.split("_")[1]), _load_json(f)))
        except (ValueError, json.JSONDecodeError):
            continue
    return out


def _stage3_fold_files(analysis_dir: Path) -> dict[int, list[dict]]:
    p = analysis_dir / "stage3"
    if not p.exists():
        return {}
    epochs: dict[int, list[dict]] = {}
    for d in sorted(p.glob("epoch_*")):
        if not d.is_dir():
            continue
        try:
            ep = int(d.name.split("_")[1])
        except ValueError:
            continue
        for f in sorted(d.glob("fold_*.json")):
            try:
                epochs.setdefault(ep, []).append(_load_json(f))
            except json.JSONDecodeError:
                continue
    return epochs


def _stage3_aggregates(analysis_dir: Path) -> list[tuple[int, dict]]:
    p = analysis_dir / "stage3"
    if not p.exists():
        return []
    out = []
    for d in sorted(p.glob("epoch_*")):
        agg = d / "aggregate.json"
        if agg.exists():
            try:
                ep = int(d.name.split("_")[1])
                out.append((ep, _load_json(agg)))
            except (ValueError, json.JSONDecodeError):
                continue
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_training_curves(analysis_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    timeline_x: list[int] = []
    timeline_loss: list[float] = []
    timeline_map: list[float] = []
    timeline_iou: list[float] = []
    boundaries: list[tuple[int, str]] = []

    cum = 0
    for stage_name in ("stage1", "stage2"):
        files = _stage_epoch_files(analysis_dir, stage_name)
        for ep, payload in files:
            cum += 1
            timeline_x.append(cum)
            timeline_loss.append(_safe_get(payload, "train", "loss"))
            timeline_map.append(_safe_get(payload, "val", "overall", "map_50"))
            timeline_iou.append(_safe_get(payload, "val", "overall", "iou_mean"))
        if files:
            boundaries.append((cum, stage_name))

    s3_aggs = _stage3_aggregates(analysis_dir)
    s3_fold_files = _stage3_fold_files(analysis_dir)
    for ep, agg in s3_aggs:
        cum += 1
        timeline_x.append(cum)
        # Average train.loss across the K fold files of this epoch.
        folds = s3_fold_files.get(ep, [])
        if folds:
            timeline_loss.append(
                sum(_safe_get(f, "train", "loss") for f in folds) / max(len(folds), 1)
            )
        else:
            timeline_loss.append(0.0)
        timeline_map.append(
            _safe_get(agg, "metrics", "val.overall.map_50", "mean")
        )
        timeline_iou.append(
            _safe_get(agg, "metrics", "val.overall.iou_mean", "mean")
        )
    if s3_aggs:
        boundaries.append((cum, "stage3"))

    axes[0].plot(timeline_x, timeline_loss, marker="o", label="train loss")
    axes[0].set_ylabel("train loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[1].plot(timeline_x, timeline_map, marker="o", color="tab:green", label="val mAP@0.5")
    axes[1].plot(timeline_x, timeline_iou, marker="o", color="tab:orange", label="val IoU mean")
    axes[1].set_ylabel("metric")
    axes[1].set_xlabel("epoch (across all stages)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    for x, name in boundaries:
        for ax in axes:
            ax.axvline(x + 0.5, color="grey", alpha=0.3, linestyle="--")
            ax.text(x, ax.get_ylim()[1], name, fontsize=8, alpha=0.6)

    fig.suptitle("Training curves (concatenated across stages)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def plot_map_per_iou(analysis_dir: Path, out_dir: Path) -> None:
    """ap_per_iou for the last epoch of each stage."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage in ("stage1", "stage2"):
        files = _stage_epoch_files(analysis_dir, stage)
        if not files:
            continue
        _, payload = files[-1]
        ap = _safe_get_dict(payload, "val", "overall", "ap_per_iou")
        if ap:
            xs = [float(k) for k in sorted(ap)]
            ys = [ap[f"{x:.2f}"] for x in xs]
            ax.plot(xs, ys, marker="o", label=stage)
    s3_aggs = _stage3_aggregates(analysis_dir)
    if s3_aggs:
        _, agg = s3_aggs[-1]
        # Pull mean ap@0.5..0.95 from aggregate metrics.
        keys = sorted(
            k for k in agg.get("metrics", {})
            if k.startswith("val.overall.ap_per_iou.")
        )
        if keys:
            xs = [float(k.split(".")[-1]) for k in keys]
            ys = [agg["metrics"][k]["mean"] for k in keys]
            ax.plot(xs, ys, marker="o", label="stage3 (cv-mean)")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_title("AP per IoU threshold (last epoch of each stage)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "map_per_iou.png", dpi=120)
    plt.close(fig)


def _safe_get_dict(d: Any, *keys) -> dict:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return {}
        cur = cur[k]
    return cur if isinstance(cur, dict) else {}


def plot_collapse_diagnostics(analysis_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = [
        ("mean_score_neg", "presence score on absent"),
        ("frac_pred_near_corner", "fraction predicted near corner"),
        ("argmax_cell_entropy", "argmax cell entropy (nats)"),
        ("support_proto_norm_mean", "support prototype norm (mean)"),
    ]
    for ax, (k, title) in zip(axes.flat, keys):
        xs: list[int] = []
        ys: list[float] = []
        cum = 0
        for stage in ("stage1", "stage2"):
            for ep, payload in _stage_epoch_files(analysis_dir, stage):
                cum += 1
                xs.append(cum)
                ys.append(_safe_get(payload, "val", "overall", k))
        for ep, agg in _stage3_aggregates(analysis_dir):
            cum += 1
            xs.append(cum)
            ys.append(_safe_get(agg, "metrics", f"val.overall.{k}", "mean"))
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Collapse diagnostics over time")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "collapse_diagnostics.png", dpi=120)
    plt.close(fig)


def plot_per_source_map(analysis_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sources = ["vizwiz_base", "vizwiz_novel", "hots", "insdet"]
    series: dict[str, list[tuple[int, float]]] = {s: [] for s in sources}
    cum = 0
    for stage in ("stage1", "stage2"):
        for ep, payload in _stage_epoch_files(analysis_dir, stage):
            cum += 1
            for s in sources:
                v = _safe_get(payload, "val", "per_source", s, "map_50")
                series[s].append((cum, v))
    for ep, agg in _stage3_aggregates(analysis_dir):
        cum += 1
        for s in sources:
            v = _safe_get(agg, "metrics", f"val.per_source.{s}.map_50", "mean")
            series[s].append((cum, v))
    for s, pts in series.items():
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=s)
    ax.set_xlabel("epoch (concatenated)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Per-source mAP@0.5 over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "per_source_map.png", dpi=120)
    plt.close(fig)


def plot_cv_envelope(analysis_dir: Path, out_dir: Path) -> None:
    """Stage 3 only: mean ± min/max envelope per epoch on val.overall.map_50."""
    s3_aggs = _stage3_aggregates(analysis_dir)
    if not s3_aggs:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    eps = [ep for ep, _ in s3_aggs]
    means = [_safe_get(agg, "metrics", "val.overall.map_50", "mean") for _, agg in s3_aggs]
    lows = [_safe_get(agg, "metrics", "val.overall.map_50", "min") for _, agg in s3_aggs]
    highs = [_safe_get(agg, "metrics", "val.overall.map_50", "max") for _, agg in s3_aggs]
    ax.plot(eps, means, marker="o", color="tab:green", label="map_50 mean")
    ax.fill_between(eps, lows, highs, color="tab:green", alpha=0.2, label="min/max across folds")
    ax.set_xlabel("Stage 3 epoch")
    ax.set_ylabel("val mAP@0.5")
    ax.set_title("Stage 3 cross-fold envelope")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cv_envelope.png", dpi=120)
    plt.close(fig)


def plot_eval_report(analysis_dir: Path, out_dir: Path) -> None:
    p = analysis_dir / "test_report.json"
    if not p.exists():
        return
    report = _load_json(p)
    overall = report.get("overall", {})
    per_source = report.get("per_source", {})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    keys = ["map_50", "map_5095", "iou_mean", "presence_acc", "presence_auroc",
            "mean_score_neg", "frac_pred_near_corner"]
    vals = [_safe_get(overall, k) for k in keys]
    axes[0].barh(keys, vals, color="tab:blue")
    axes[0].set_title("Overall test metrics")
    axes[0].grid(True, alpha=0.3, axis="x")

    sources = list(per_source.keys())
    if sources:
        map50s = [_safe_get(per_source[s], "map_50") for s in sources]
        ious = [_safe_get(per_source[s], "iou_mean") for s in sources]
        x = range(len(sources))
        w = 0.35
        axes[1].bar([i - w / 2 for i in x], map50s, w, label="map_50")
        axes[1].bar([i + w / 2 for i in x], ious, w, label="iou_mean")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(sources, rotation=20)
        axes[1].set_title("Per-source test metrics")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "eval_report.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def plot_all_from_jsons(analysis_dir: str | Path) -> None:
    analysis_dir = Path(analysis_dir)
    out_dir = analysis_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(analysis_dir, out_dir)
    plot_map_per_iou(analysis_dir, out_dir)
    plot_collapse_diagnostics(analysis_dir, out_dir)
    plot_per_source_map(analysis_dir, out_dir)
    plot_cv_envelope(analysis_dir, out_dir)
    plot_eval_report(analysis_dir, out_dir)
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", default="analysis")
    args = p.parse_args()
    plot_all_from_jsons(args.analysis_dir)
