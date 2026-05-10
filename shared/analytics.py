"""JSON analysis writers for both trainers.

Uses fsync + retry semantics matching ``shared.checkpoint.atomic_save`` so
that analysis JSONs are equally durable on Google Drive's FUSE mount.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def write_json(
    path: Path,
    payload: Any,
    *,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> None:
    """Atomic, durable JSON write.

    1. Write to <path>.tmp via a regular file handle.
    2. flush() + os.fsync() to push bytes off the FUSE cache (Drive).
    3. os.replace(tmp, path) to atomically rename.
    4. fsync the parent directory (best-effort).

    Retries on transient OSError up to ``max_retries`` times.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, default=float)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(str(tmp), str(path))
            try:
                fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                pass
            last_err = None
            break
        except OSError as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt + 1 < max_retries:
                time.sleep(retry_backoff_s * (2 ** attempt))
    if last_err is not None:
        raise last_err


def flatten_metrics(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_metrics(v, key))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
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
            "mean": m, "min": min(vals), "max": max(vals), "std": var ** 0.5,
        }
    return {
        "epoch": fold_jsons[0].get("epoch") if fold_jsons else None,
        "n_folds": len(fold_jsons),
        "metrics": metrics,
    }


def update_summary(analysis_dir: Path, headline: dict[str, tuple[int, float]]) -> None:
    """Rolling best-by-metric pointer at <analysis_dir>/summary.json."""
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
