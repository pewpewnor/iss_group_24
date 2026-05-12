"""Manifest loader / accessor.

Manifest schema (v5) — see aggregator.py:

    {
      "schema_version": 5,
      "num_instances": int,
      "splits": {"train": [...], "test": [...]},
      "instances": [
        {
          "instance_id": str, "source": "hots"|"insdet",
          "class_name": str, "split": "train"|"test",
          "support_images": [{"path": str, "bbox": [x1,y1,x2,y2]}],
          "query_images":   [{"path": str, "bbox": [x1,y1,x2,y2], "scene_type": str}]
        }
      ]
    }

Note (v5): InsDet support images are PHYSICALLY pre-cropped on disk by the
aggregator. Their stored ``bbox`` spans the full extent of the on-disk
image (``[0, 0, W, H]``) — there is no background context. HOTS supports
are object-centred in the source dataset so they're staged unchanged.
The training + eval pipeline assumes "support image = object" universally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 6


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    p = Path(manifest_path)
    with open(p) as f:
        m = json.load(f)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version mismatch (got {m.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}). Re-run aggregator.py."
        )
    return m


def filter_instances(
    manifest: dict[str, Any],
    *,
    split: str | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    out = list(manifest["instances"])
    if split is not None:
        out = [i for i in out if i.get("split") == split]
    if sources is not None:
        ssrc = set(sources)
        out = [i for i in out if i.get("source") in ssrc]
    return out
