"""Stratified K-fold construction for Stage 3 cross-validation.

Each fold is built so every source's instances are partitioned in round-robin
order — guaranteeing every fold has the same source mix as the training pool.
"""

from __future__ import annotations

import random as _random
from typing import Any


def stratified_kfold(
    instances: list[dict[str, Any]], k: int, seed: int
) -> list[dict[str, list[str]]]:
    """Return K folds; each is ``{"train_ids": [...], "val_ids": [...]}``."""
    by_source: dict[str, list[str]] = {}
    for inst in instances:
        by_source.setdefault(inst.get("source", "_"), []).append(inst["instance_id"])
    rng = _random.Random(seed)
    for src in by_source:
        by_source[src].sort()
        rng.shuffle(by_source[src])

    fold_val: list[list[str]] = [[] for _ in range(k)]
    for ids in by_source.values():
        for i, iid in enumerate(ids):
            fold_val[i % k].append(iid)

    all_ids = sorted({i["instance_id"] for i in instances})
    folds: list[dict[str, list[str]]] = []
    for f in range(k):
        val_set = set(fold_val[f])
        train_ids = [iid for iid in all_ids if iid not in val_set]
        folds.append({"train_ids": train_ids, "val_ids": sorted(val_set)})
    return folds
