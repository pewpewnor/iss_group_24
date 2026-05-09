"""Per-epoch / per-fold console logging."""

from __future__ import annotations

from typing import Any


_TRAIN_PRIORITY = (
    "loss", "focal", "margin", "nt_xent",
    "l1", "giou", "box_loss",
    "box_area_penalty", "existence_kl",
    "grad_norm", "n_steps",
    # Residual aggregator gate scalar.  alpha=0 ⇒ baseline-only prototype;
    # alpha grows as the model learns to trust the aggregator's correction.
    "aggregator_alpha",
)

_VAL_PRIORITY = (
    "n", "n_pos", "n_neg",
    "map_50", "map_75", "map_5095",
    "map_50_existence_only", "map_50_score_only",
    "map_5095_existence_only", "map_5095_score_only",
    "f1_50", "precision_50", "recall_50",
    "iou_mean", "iou_median", "iou_p25", "iou_p75",
    "contain_mean", "contain_at_iou_50", "contain_at_iou_75",
    "existence_acc", "existence_acc_pos", "existence_acc_neg",
    "existence_auroc", "existence_pr_auc", "existence_brier", "existence_f1",
    "false_positive_rate", "false_negative_rate",
    "mean_score_pos", "mean_score_neg",
    # Prototype-quality diagnostic (raw class-head sigmoid; isolates the
    # support-prototype path from the existence head).
    "proto_score_pos", "proto_score_neg", "proto_score_gap",
    "mean_pred_box_area", "frac_pred_box_too_big",
    "mean_existence_prob", "frac_high_existence",
    "prototype_norm_mean", "prototype_norm_std",
)

_PER_SOURCE_KEYS = (
    "n", "n_pos", "map_50", "map_5095", "iou_mean", "f1_50",
    "existence_acc", "mean_score_pos", "mean_score_neg",
    "proto_score_gap",                                       # ← per-source
    "false_positive_rate",
)


def _format_kv(k: str, v: Any) -> str | None:
    if not isinstance(v, (int, float)):
        return None
    if k in ("n", "n_pos", "n_neg", "n_steps") or isinstance(v, int):
        return f"{k}={int(v)}"
    return f"{k}={float(v):.4f}"


def _wrap_lines(prefix: str, parts: list[str], width: int = 110) -> list[str]:
    out: list[str] = []
    cur = ""
    for chunk in parts:
        if cur and len(cur) + len(chunk) > width:
            out.append(f"{prefix}{cur}")
            cur = chunk
        else:
            cur = (cur + "  " + chunk) if cur else chunk
    if cur:
        out.append(f"{prefix}{cur}")
    return out


def print_epoch_log(
    *,
    header: str,
    train_metrics: dict[str, float],
    val_metrics: dict,
    lr_groups: dict[str, float],
    wall_clock: float,
) -> None:
    print(f"┌─ {header}  (wall={wall_clock:.1f}s)")
    if lr_groups:
        lr_str = "  ".join(f"{k}={v:.2e}" for k, v in lr_groups.items())
        print(f"│  lr      : {lr_str}")
    if train_metrics:
        ordered = [k for k in _TRAIN_PRIORITY if k in train_metrics] + sorted(
            k for k in train_metrics if k not in _TRAIN_PRIORITY
        )
        parts = [s for s in (_format_kv(k, train_metrics[k]) for k in ordered) if s]
        for line in _wrap_lines("│  train   : ", parts):
            print(line)
    overall = val_metrics.get("overall", {}) if isinstance(val_metrics, dict) else {}
    if overall:
        ordered = [k for k in _VAL_PRIORITY if k in overall] + sorted(
            k for k in overall if k not in _VAL_PRIORITY and k != "ap_per_iou"
        )
        parts = [s for s in (_format_kv(k, overall[k]) for k in ordered) if s]
        for line in _wrap_lines("│  val     : ", parts):
            print(line)
        ap = overall.get("ap_per_iou")
        if isinstance(ap, dict):
            ap_str = "  ".join(f"{k}={float(v):.3f}" for k, v in sorted(ap.items()))
            print(f"│  ap@iou  : {ap_str}")
    per_source = val_metrics.get("per_source", {}) if isinstance(val_metrics, dict) else {}
    for src in sorted(per_source.keys()):
        sm = per_source[src]
        if not isinstance(sm, dict) or not sm:
            continue
        chunks = [s for s in (_format_kv(k, sm.get(k)) for k in _PER_SOURCE_KEYS) if s]
        if chunks:
            print(f"│  {src:<13s}: {'  '.join(chunks)}")
    print("└─")


def print_aggregate(stage: str, epoch: int, aggregate: dict) -> None:
    metrics = aggregate.get("metrics", {})
    print(f"┌─ {stage} epoch {epoch} CV aggregate (n_folds={aggregate.get('n_folds')})")
    keys = (
        ("val.overall.map_50",                  "map_50"),
        ("val.overall.map_75",                  "map_75"),
        ("val.overall.map_5095",                "map_5095"),
        ("val.overall.iou_mean",                "iou_mean"),
        ("val.overall.f1_50",                   "f1_50"),
        ("val.overall.existence_acc",           "exist_acc"),
        ("val.overall.existence_auroc",         "exist_auroc"),
        ("val.overall.false_positive_rate",     "fpr"),
        ("val.overall.proto_score_gap",         "proto_gap"),
        ("val.overall.mean_pred_box_area",      "pred_area"),
        ("train.aggregator_alpha",              "agg_alpha"),
    )
    for key, label in keys:
        m = metrics.get(key)
        if not isinstance(m, dict):
            continue
        print(
            f"│  {label:>15s}: mean={m['mean']:.4f}  "
            f"min={m['min']:.4f}  max={m['max']:.4f}  std={m['std']:.4f}"
        )
    print("└─")
