"""JSON analysis writers for the trainer.

- ``write_json``         : pretty-print + atomic write.
- ``flatten_metrics``    : nested dict -> {dotted_key: float}.
- ``aggregate_folds``    : recursive mean/min/max/std across the K fold
                           JSONs of a single epoch.
- ``update_summary``     : rolling best-by-metric pointer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    os.replace(str(tmp), str(path))


def flatten_metrics(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_metrics(v, key))
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    return out


def aggregate_folds(fold_jsons: list[dict]) -> dict:
    """Recursive mean / min / max / std across folds for every numeric metric."""
    flat_per_fold = [flatten_metrics(j) for j in fold_jsons]
    keys: set[str] = set()
    for f in flat_per_fold:
        keys.update(f.keys())
    metrics: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        vals = [f[k] for f in flat_per_fold if k in f]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / max(len(vals), 1)
        metrics[k] = {
            "mean": m,
            "min": min(vals),
            "max": max(vals),
            "std": var ** 0.5,
        }
    return {
        "epoch": fold_jsons[0].get("epoch") if fold_jsons else None,
        "n_folds": len(fold_jsons),
        "metrics": metrics,
    }


def update_summary(
    analysis_dir: Path, headline: dict[str, tuple[int, float]]
) -> None:
    """Rolling best-by-metric pointer at ``analysis/summary.json``."""
    summary_path = analysis_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            current = json.load(f)
    else:
        current = {"best_by": {}}
    best_by = current.get("best_by", {})
    for k, (epoch, value) in headline.items():
        prev = best_by.get(k)
        if prev is None or value > prev["value"]:
            best_by[k] = {"epoch": epoch, "value": value}
    current["best_by"] = best_by
    write_json(summary_path, current)
