"""Per-epoch / per-fold console logging.

Single public entry point: ``print_epoch_log(...)``. Dumps every train loss
key and every numeric val metric (overall + per_source) in a stable,
priority-ordered format suitable for a notebook stream. Verbose by design —
the user has explicitly asked to see every loss + every metric per epoch.
"""

from __future__ import annotations

from typing import Any


_TRAIN_PRIORITY = (
    "loss", "qfl", "neg_qfl", "centerness", "dfl", "giou",
    "presence", "attn", "aux", "entropy_reg", "reg_l2_prior",
    "proto_norm", "nt_xent", "vicreg", "barlow", "triplet",
    "grad_norm", "n_steps",
)

_VAL_PRIORITY = (
    "n", "n_pos", "n_neg",
    "map_50", "map_5095", "f1_50", "precision_50", "recall_50",
    "iou_mean", "iou_median", "iou_p25", "iou_p75",
    "contain_mean", "contain_at_iou_50", "contain_at_iou_75",
    "presence_acc", "presence_acc_pos", "presence_acc_neg",
    "mean_score_pos", "mean_score_neg",
    "presence_auroc", "presence_pr_auc", "presence_brier",
    "frac_pred_near_corner", "frac_pred_tiny_box", "argmax_cell_entropy",
    "conf_map_mean_pos", "conf_map_mean_neg",
    "conf_map_std_pos", "conf_map_std_neg",
    "support_proto_norm_mean", "support_proto_norm_std",
)

_PER_SOURCE_KEYS = (
    "n", "n_pos", "map_50", "map_5095", "iou_mean", "f1_50",
    "presence_acc", "mean_score_pos", "mean_score_neg",
    "frac_pred_near_corner",
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
    """Print a fully verbose summary block for one epoch / fold."""
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


def print_stage3_aggregate(epoch: int, aggregate: dict) -> None:
    """Pretty-print Stage 3 cross-fold mean / min / max for headline metrics."""
    metrics = aggregate.get("metrics", {})
    print(f"┌─ s3 epoch {epoch} CV aggregate (n_folds={aggregate.get('n_folds')})")
    keys = (
        ("val.overall.map_50",                  "map_50"),
        ("val.overall.map_5095",                "map_5095"),
        ("val.overall.iou_mean",                "iou_mean"),
        ("val.overall.f1_50",                   "f1_50"),
        ("val.overall.presence_acc",            "presence_acc"),
        ("val.overall.mean_score_neg",          "mean_score_neg"),
        ("val.overall.frac_pred_near_corner",   "frac_corner"),
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
