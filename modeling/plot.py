"""Matplotlib visualizations for training and evaluation.

All public functions write PNG files to an ``out_dir`` (typically
``analysis/``).  The directory is created if it does not exist.

Functions
---------
plot_training_curves   -- per-epoch loss / metric curves across all stages
plot_contrastive_learning -- NT-Xent + VICReg curves with stage markers
plot_lr_schedule       -- analytically simulated cosine-annealing LR curves
plot_prototype_similarity -- pairwise cosine-similarity heatmap of prototypes
plot_eval_report       -- bar chart + score-distribution histogram from eval JSON
plot_dataset_stats     -- stacked-bar and pie chart from dataset stats JSON
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STAGE_COLORS = {
    "stage1": "#4C72B0",
    "stage2": "#DD8452",
    "stage3": "#55A868",
}

_STAGE_LABELS = {
    "stage1": "Stage 1 (warmup)",
    "stage2": "Stage 2 (partial unfreeze)",
    "stage3": "Stage 3 (full unfreeze)",
}


def _ensure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _stage_boundaries(history: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Return (absolute_epoch_index, stage_name) for the first epoch of each stage."""
    seen: list[str] = []
    boundaries: list[tuple[int, str]] = []
    for i, row in enumerate(history):
        s = row["stage"]
        if s not in seen:
            seen.append(s)
            boundaries.append((i, s))
    return boundaries


def _draw_stage_vlines(ax: plt.Axes, boundaries: list[tuple[int, str]]) -> None:
    for idx, (epoch_i, _) in enumerate(boundaries):
        if idx == 0:
            continue
        ax.axvline(epoch_i, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)


def _stage_spans(ax: plt.Axes, boundaries: list[tuple[int, str]], n_epochs: int) -> None:
    """Shade background by stage."""
    alphas = [0.06, 0.10, 0.06]
    for k, (start_i, stage) in enumerate(boundaries):
        end_i = boundaries[k + 1][0] if k + 1 < len(boundaries) else n_epochs
        color = _STAGE_COLORS.get(stage, "grey")
        ax.axvspan(start_i, end_i, facecolor=color, alpha=alphas[k % len(alphas)], linewidth=0)


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(history: list[dict[str, Any]], out_dir: Path) -> None:
    """Four-subplot figure of per-epoch training/val metrics.

    Parameters
    ----------
    history:
        List of dicts with keys: stage, epoch, loss, focal, box,
        nt_xent, vicreg, val_loss, val_iou, val_presence_acc.
    out_dir:
        Directory where ``training_curves.png`` is written.
    """
    _ensure(out_dir)
    if not history:
        return

    epochs = list(range(len(history)))
    boundaries = _stage_boundaries(history)
    n = len(history)

    def _get(key: str) -> list[float]:
        return [row.get(key, 0.0) for row in history]

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    fig.suptitle("Training curves", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, _get("loss"), label="Total loss", linewidth=1.8, color="#333333")
    ax.plot(epochs, _get("focal"), label="Focal loss", linewidth=1.4, linestyle="--", color="#4C72B0")
    ax.plot(epochs, _get("box"), label="Box (GIoU) loss", linewidth=1.4, linestyle="--", color="#DD8452")
    ax.plot(epochs, _get("presence"), label="Presence BCE", linewidth=1.4, linestyle="-.", color="#E76F51")
    ax.plot(epochs, _get("attn"), label="Saliency attn KL", linewidth=1.2, linestyle=":", color="#9C27B0")
    _stage_spans(ax, boundaries, n)
    _draw_stage_vlines(ax, boundaries)
    ax.set_ylabel("Train loss")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax = axes[1]
    ax.plot(epochs, _get("nt_xent"), label="NT-Xent (contrastive)", linewidth=1.6, color="#8172B2")
    ax.plot(epochs, _get("vicreg"), label="VICReg", linewidth=1.6, linestyle="--", color="#C44E52")
    _stage_spans(ax, boundaries, n)
    _draw_stage_vlines(ax, boundaries)
    ax.set_ylabel("Contrastive loss")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax = axes[2]
    ax.plot(epochs, _get("val_loss"), label="Val loss", linewidth=1.6, color="#64B5CD")
    _stage_spans(ax, boundaries, n)
    _draw_stage_vlines(ax, boundaries)
    ax.set_ylabel("Val loss")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax = axes[3]
    ax.plot(epochs, _get("val_iou"), label="Val IoU", linewidth=1.6, color="#55A868")
    ax.plot(epochs, _get("val_map"), label="Val mAP@[0.5:0.95]", linewidth=1.6, linestyle="-.", color="#2196F3")
    ax.plot(epochs, _get("val_presence_acc"), label="Val presence acc", linewidth=1.6, linestyle="--", color="#CCB974")
    _stage_spans(ax, boundaries, n)
    _draw_stage_vlines(ax, boundaries)
    ax.set_xlabel("Epoch (absolute)")
    ax.set_ylabel("Val metric")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_ylim(0, 1.05)

    handles = [
        mpatches.Patch(facecolor=_STAGE_COLORS.get(s, "grey"), alpha=0.4, label=_STAGE_LABELS.get(s, s))
        for _, s in boundaries
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, title="Stage", title_fontsize=8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# 2. Contrastive learning plot
# ---------------------------------------------------------------------------


def plot_contrastive_learning(history: list[dict[str, Any]], out_dir: Path) -> None:
    """Focused plot of NT-Xent and VICReg losses per epoch with stage markers.

    Parameters
    ----------
    history:
        Same list of dicts as ``plot_training_curves``.
    out_dir:
        Directory where ``contrastive_learning.png`` is written.
    """
    _ensure(out_dir)
    if not history:
        return

    epochs = list(range(len(history)))
    boundaries = _stage_boundaries(history)
    n = len(history)

    nt_xent = [row.get("nt_xent", 0.0) for row in history]
    vicreg = [row.get("vicreg", 0.0) for row in history]
    val_iou = [row.get("val_iou", 0.0) for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Contrastive / regularisation losses", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, nt_xent, label="NT-Xent (uniformity)", linewidth=1.8, color="#8172B2")
    ax.plot(epochs, vicreg, label="VICReg (var+cov)", linewidth=1.8, linestyle="--", color="#C44E52")
    _stage_spans(ax, boundaries, n)
    _draw_stage_vlines(ax, boundaries)
    ax.set_ylabel("Loss value")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    val_map = [row.get("val_map", 0.0) for row in history]

    ax = axes[1]
    ax.plot(epochs, val_iou, label="Val IoU", linewidth=1.8, color="#55A868")
    ax.plot(epochs, val_map, label="Val mAP@[0.5:0.95]", linewidth=1.8, linestyle="-.", color="#2196F3")
    _stage_spans(ax, boundaries, n)
    _draw_stage_vlines(ax, boundaries)
    ax.set_xlabel("Epoch (absolute)")
    ax.set_ylabel("Val metric")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    stage_handles = [
        mpatches.Patch(facecolor=_STAGE_COLORS.get(s, "grey"), alpha=0.4, label=_STAGE_LABELS.get(s, s))
        for _, s in boundaries
    ]
    fig.legend(handles=stage_handles, loc="upper right", fontsize=8, title="Stage", title_fontsize=8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = out_dir / "contrastive_learning.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# 3. LR schedule (analytically simulated)
# ---------------------------------------------------------------------------


def _cosine_anneal(lr_start: float, lr_min: float, t: int, t_max: int) -> float:
    return lr_min + 0.5 * (lr_start - lr_min) * (1 + math.cos(math.pi * t / max(t_max, 1)))


def plot_lr_schedule(
    stage_configs: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Analytically simulate cosine-annealing LR curves for each stage.

    Parameters
    ----------
    stage_configs:
        List of dicts, one per stage:
        ``{name: str, epochs: int, steps_per_epoch: int,
           param_groups: [{label: str, lr: float}]}``.
    out_dir:
        Directory where ``lr_schedule.png`` is written.
    """
    _ensure(out_dir)
    if not stage_configs:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title("Learning-rate schedule (cosine annealing per stage)", fontsize=12, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#CCB974"]
    lr_min = 1e-7

    global_step = 0
    stage_start_steps: list[tuple[int, str]] = []

    for cfg in stage_configs:
        stage_name: str = cfg["name"]
        epochs: int = cfg["epochs"]
        steps_per_epoch: int = cfg["steps_per_epoch"]
        param_groups: list[dict] = cfg["param_groups"]
        t_max = epochs * steps_per_epoch
        stage_start_steps.append((global_step, stage_name))

        for gi, pg in enumerate(param_groups):
            label = pg["label"]
            lr_start = pg["lr"]
            steps = list(range(t_max))
            lrs = [_cosine_anneal(lr_start, lr_min, t, t_max) for t in steps]
            xs = [global_step + t for t in steps]
            color = colors[gi % len(colors)]
            linestyle = "-" if gi == 0 else "--" if gi == 1 else ":"
            ax.plot(xs, lrs, color=color, linestyle=linestyle, linewidth=1.4,
                    label=f"{label} ({stage_name})")

        global_step += t_max

    for i, (step, stage) in enumerate(stage_start_steps):
        if i > 0:
            ax.axvline(step, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(step + global_step * 0.005, ax.get_ylim()[1] * 0.95,
                _STAGE_LABELS.get(stage, stage), fontsize=7, color="grey",
                verticalalignment="top")

    ax.set_xlabel("Global training step")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "lr_schedule.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# 4. Prototype similarity heatmap
# ---------------------------------------------------------------------------


def plot_prototype_similarity(
    proto_cache: dict[str, Any],
    out_dir: Path,
    max_instances: int = 30,
) -> None:
    """Pairwise cosine-similarity heatmap of prototype vectors.

    Parameters
    ----------
    proto_cache:
        Dict mapping instance_id -> prototype tensor (shape (PROTO_DIM,)).
    out_dir:
        Directory where ``prototype_similarity.png`` is written.
    max_instances:
        Cap on instances to show (sorted alphabetically then trimmed).
    """
    _ensure(out_dir)
    if not proto_cache:
        return

    import torch
    import torch.nn.functional as F

    ids = sorted(proto_cache.keys())[:max_instances]
    vecs = torch.stack([proto_cache[i].float() for i in ids], dim=0)
    vecs = F.normalize(vecs, dim=-1)
    sim = (vecs @ vecs.T).cpu().numpy()

    n = len(ids)
    fig_size = max(6, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    tick_labels = [i.split("_", 1)[-1][:12] for i in ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=max(4, 8 - n // 10))
    ax.set_yticklabels(tick_labels, fontsize=max(4, 8 - n // 10))
    ax.set_title(f"Prototype cosine similarity ({n} instances)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "prototype_similarity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# 5. Eval report plots
# ---------------------------------------------------------------------------


def plot_eval_report(report: dict[str, Any], out_dir: Path) -> None:
    """Bar chart of key metrics and score-distribution histograms from eval.

    Parameters
    ----------
    report:
        Dict as returned by ``modeling.evaluate.evaluate`` / written to
        ``eval_report.json``.  Expected keys: ``overall``, ``by_source``,
        ``score_thr``, ``iou_thr``.
    out_dir:
        Directory where ``eval_metrics.png`` is written.
    """
    _ensure(out_dir)
    if not report:
        return

    overall = report.get("overall", {})
    by_source = report.get("by_source", {})

    metric_keys = [
        "mean_iou_pos",
        "presence_acc",
        "ap@iou=0.5",
        "ap@iou=0.75",
        "map@[0.5:0.95]",
    ]
    metric_labels = [
        "Mean IoU (pos)",
        "Presence acc",
        "AP@0.5",
        "AP@0.75",
        "mAP@[0.5:0.95]",
    ]

    sources = ["overall"] + sorted(by_source.keys())
    source_data: dict[str, dict] = {"overall": overall}
    source_data.update(by_source)

    source_display = {
        "overall": "Overall",
        "hots": "HOTS",
        "insdet": "InsDet",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evaluation metrics", fontsize=13, fontweight="bold")

    ax = axes[0]
    x = np.arange(len(metric_keys))
    bar_width = 0.8 / max(len(sources), 1)
    bar_colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]

    for si, src in enumerate(sources):
        vals = [source_data[src].get(mk, 0.0) for mk in metric_keys]
        offset = (si - len(sources) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.9,
                      label=source_display.get(src, src),
                      color=bar_colors[si % len(bar_colors)],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.set_title("Metrics by source", fontsize=10)

    ax_ap = axes[1]
    for si, src in enumerate(sources):
        ap_per_iou = source_data[src].get("ap_per_iou", {})
        if not ap_per_iou:
            continue
        taus = sorted(float(k) for k in ap_per_iou.keys())
        vals = [ap_per_iou[f"{t:.2f}"] for t in taus]
        ax_ap.plot(
            taus, vals, marker="o", linewidth=1.5,
            label=source_display.get(src, src),
            color=bar_colors[si % len(bar_colors)],
        )
        map_val = source_data[src].get("map@[0.5:0.95]", 0.0)
        ax_ap.axhline(
            map_val, linestyle="--", linewidth=0.7, alpha=0.4,
            color=bar_colors[si % len(bar_colors)],
        )
    ax_ap.set_xlabel("IoU threshold")
    ax_ap.set_ylabel("AP")
    ax_ap.set_ylim(0, 1.05)
    ax_ap.set_title("AP vs IoU threshold (dashed = mAP@[0.5:0.95])", fontsize=10)
    ax_ap.legend(fontsize=8)
    ax_ap.grid(True, linewidth=0.4, alpha=0.5)

    ax2 = axes[2]
    pos_scores = [row.get("mean_score_pos", 0.0) for row in source_data.values() if row]
    neg_scores = [row.get("mean_score_neg", 0.0) for row in source_data.values() if row]
    src_labels = [source_display.get(s, s) for s in sources]

    xb = np.arange(len(src_labels))
    bw = 0.35
    ax2.bar(xb - bw / 2, pos_scores, bw, label="Mean score (positive)", color="#55A868", alpha=0.85, edgecolor="white")
    ax2.bar(xb + bw / 2, neg_scores, bw, label="Mean score (negative)", color="#C44E52", alpha=0.85, edgecolor="white")
    for xi, (p, n_val) in enumerate(zip(pos_scores, neg_scores)):
        ax2.text(xi - bw / 2, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=7)
        ax2.text(xi + bw / 2, n_val + 0.01, f"{n_val:.3f}", ha="center", va="bottom", fontsize=7)

    score_thr = report.get("score_thr", 0.5)
    ax2.axhline(score_thr, color="grey", linestyle="--", linewidth=0.9, label=f"Score thr ({score_thr})")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(src_labels, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Predicted score")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax2.set_title("Pos vs neg score separability", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "eval_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# 6. Dataset statistics
# ---------------------------------------------------------------------------


def plot_dataset_stats(stats_path: Path, out_dir: Path) -> None:
    """Stacked-bar and pie chart from ``dataset/cleaned/stats.json``.

    The expected JSON schema is::

        {
          "total_instances": 146,
          "train": {"instances": 102, "hots": 27, "insdet": 75,
                    "support_images": 3184, "query_images": 3095},
          "val":   {...},
          "test":  {...}
        }

    Parameters
    ----------
    stats_path:
        Path to ``stats.json``.
    out_dir:
        Directory where ``dataset_stats.png`` is written.
    """
    _ensure(out_dir)
    if not stats_path.exists():
        return

    with open(stats_path) as f:
        stats = json.load(f)

    split_order = [s for s in ("train", "val", "test") if s in stats]
    if not split_order:
        return

    known_sources = ("hots", "insdet")
    sources = [s for s in known_sources if any(s in stats[sp] for sp in split_order)]

    source_colors = {
        "hots": "#4C72B0",
        "insdet": "#DD8452",
    }
    default_colors = ["#55A868", "#8172B2", "#C44E52", "#CCB974"]

    def _color(src: str, idx: int) -> str:
        return source_colors.get(src, default_colors[idx % len(default_colors)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset statistics", fontsize=13, fontweight="bold")

    ax_support = axes[0]
    ax_query = axes[1]
    ax_pie = axes[2]

    bar_width = 0.55
    x = np.arange(len(split_order))

    for ax_b, img_key, title in [
        (ax_support, "support_images", "Support images per split"),
        (ax_query, "query_images", "Query images per split"),
    ]:
        bottoms = np.zeros(len(split_order))
        for si, src in enumerate(sources):
            src_img_vals: list[float] = []
            for sp in split_order:
                sp_data = stats[sp]
                total_imgs = sp_data.get(img_key, 0)
                n_instances = sp_data.get("instances", 1)
                src_instances = sp_data.get(src, 0)
                frac = src_instances / max(n_instances, 1)
                src_img_vals.append(round(total_imgs * frac))

            vals = np.array(src_img_vals, dtype=float)
            ax_b.bar(x, vals, bar_width, bottom=bottoms,
                     label=src.upper(), color=_color(src, si), alpha=0.88, edgecolor="white")
            for xi, (v, bot) in enumerate(zip(vals, bottoms)):
                if v > 0:
                    ax_b.text(xi, bot + v / 2, str(int(v)), ha="center", va="center",
                              fontsize=8, color="white", fontweight="bold")
            bottoms = bottoms + vals

        ax_b.set_xticks(x)
        ax_b.set_xticklabels([s.capitalize() for s in split_order])
        ax_b.set_ylabel("Image count")
        ax_b.set_title(title, fontsize=10)
        ax_b.legend(fontsize=8)
        ax_b.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    instance_counts: dict[str, int] = {}
    for src in sources:
        total = sum(stats[sp].get(src, 0) for sp in split_order)
        if total > 0:
            instance_counts[src] = total

    if instance_counts:
        pie_labels = [s.upper() for s in instance_counts.keys()]
        pie_vals = list(instance_counts.values())
        pie_colors = [_color(s, i) for i, s in enumerate(instance_counts.keys())]
        _, _, autotexts = ax_pie.pie(
            pie_vals, labels=pie_labels, colors=pie_colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 10},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax_pie.set_title("Instances by source", fontsize=10)
    else:
        ax_pie.axis("off")

    plt.tight_layout()
    out_path = out_dir / "dataset_stats.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
