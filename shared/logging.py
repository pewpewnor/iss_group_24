"""Console logging shared by both trainers."""

from __future__ import annotations

from typing import Any


def _format_kv(k: str, v: Any) -> str | None:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    if k in ("n", "n_pos", "n_neg", "n_steps", "k", "epoch", "fold") or isinstance(v, int):
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
    train_priority: tuple[str, ...] = (),
    val_priority: tuple[str, ...] = (),
    per_source_keys: tuple[str, ...] = (),
) -> None:
    print(f"┌─ {header}  (wall={wall_clock:.1f}s)")
    if lr_groups:
        lr_str = "  ".join(f"{k}={v:.2e}" for k, v in lr_groups.items())
        print(f"│  lr      : {lr_str}")
    if train_metrics:
        ordered = [k for k in train_priority if k in train_metrics] + sorted(
            k for k in train_metrics if k not in train_priority
        )
        parts = [s for s in (_format_kv(k, train_metrics[k]) for k in ordered) if s]
        for line in _wrap_lines("│  train   : ", parts):
            print(line)
    overall = val_metrics.get("overall", {}) if isinstance(val_metrics, dict) else {}
    if overall:
        ordered = [k for k in val_priority if k in overall] + sorted(
            k for k in overall if k not in val_priority and not isinstance(overall.get(k), (dict, list))
        )
        parts = [s for s in (_format_kv(k, overall[k]) for k in ordered) if s]
        for line in _wrap_lines("│  val     : ", parts):
            print(line)
    per_source = val_metrics.get("per_source", {}) if isinstance(val_metrics, dict) else {}
    for src in sorted(per_source.keys()):
        sm = per_source[src]
        if not isinstance(sm, dict) or not sm:
            continue
        chunks = [s for s in (_format_kv(k, sm.get(k)) for k in per_source_keys) if s]
        if chunks:
            print(f"│  {src:<13s}: {'  '.join(chunks)}")
    per_k = val_metrics.get("per_k", {}) if isinstance(val_metrics, dict) else {}
    for k_label in sorted(per_k.keys(), key=lambda s: int(s.lstrip("k"))):
        km = per_k[k_label]
        if not isinstance(km, dict) or not km:
            continue
        chunks = [s for s in (_format_kv(kk, km.get(kk)) for kk in val_priority[:6]) if s]
        if chunks:
            print(f"│  {k_label:<13s}: {'  '.join(chunks)}")
    print("└─")


def print_aggregate(
    stage: str, epoch: int, aggregate: dict, keys: tuple[tuple[str, str], ...],
) -> None:
    metrics = aggregate.get("metrics", {})
    print(f"┌─ {stage} epoch {epoch} CV aggregate (n_folds={aggregate.get('n_folds')})")
    for key, label in keys:
        m = metrics.get(key)
        if not isinstance(m, dict):
            continue
        print(
            f"│  {label:>20s}: mean={m['mean']:.4f}  "
            f"min={m['min']:.4f}  max={m['max']:.4f}  std={m['std']:.4f}"
        )
    print("└─")
