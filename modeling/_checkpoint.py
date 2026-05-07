"""Universal checkpoint I/O for the three-stage trainer.

- Atomic save (write tmp + os.replace).
- RNG capture / restore (torch / numpy / python / cuda).
- Resume-path resolution and arch-mismatch quarantine.
- Disk hygiene (rolling per-(epoch, fold) checkpoints capped, stage-completion /
  best / last protected).

The checkpoint payload schema is documented in PLAN.md §3.
"""

from __future__ import annotations

import os
import random as _random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def capture_rng() -> dict:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": _random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict | None) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        _random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def resolve_resume_path(resume: bool | str, out_dir: Path) -> Path | None:
    """``True`` → ``last.pt`` (if exists). String → resolved against ``out_dir``."""
    if resume is False or resume is None:
        return None
    if resume is True:
        p = out_dir / "last.pt"
        return p if p.exists() else None
    p = Path(resume)                                                          # type: ignore[arg-type]
    if not p.is_absolute():
        p = out_dir / p
    return p if p.exists() else None


def try_load_state_dict(
    model: torch.nn.Module, state: dict
) -> tuple[bool, str]:
    """Strict-by-shape load. Returns ``(ok, error_message)``.

    Refuses to load when any tensor's shape differs from the live model — the
    silent ``strict=False`` fallback would leave architecture-mismatched
    parameters at their fresh init, masking the bug.
    """
    own_state = model.state_dict()
    mismatches: list[str] = []
    for k, v in state.items():
        if k in own_state and own_state[k].shape != v.shape:
            mismatches.append(
                f"{k}: ckpt {tuple(v.shape)} vs model {tuple(own_state[k].shape)}"
            )
    if mismatches:
        head = "; ".join(mismatches[:5])
        tail = f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        return False, head + tail
    try:
        model.load_state_dict(state, strict=False)
        return True, ""
    except RuntimeError as e:                                                 # noqa: BLE001
        return False, str(e)


def quarantine_incompatible(out_dir: Path, reason: str) -> Path:
    """Move every ``*.pt`` under ``out_dir`` into ``out_dir/legacy_<ts>/``.

    Used when a loaded checkpoint's state_dict doesn't match the current
    architecture. We never delete the user's data — just step around it.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = out_dir / f"legacy_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for p in list(out_dir.glob("*.pt")):
        try:
            os.replace(str(p), str(backup / p.name))
            moved.append(p.name)
        except OSError:
            pass
    print(
        f"⚠ Quarantined {len(moved)} incompatible checkpoint(s) into {backup}\n"
        f"  reason: {reason}\n"
        f"  files : {', '.join(moved) if moved else '(none)'}\n"
        f"  Training will start fresh."
    )
    return backup


# ---------------------------------------------------------------------------
# Resume decision tree
# ---------------------------------------------------------------------------


def next_resume_point(ckpt: dict, cfg: dict) -> dict:
    """Given a loaded checkpoint, return the dict telling the trainer where
    to start.

    ``rebuild_optimizer`` is True iff resuming crosses a stage boundary —
    optimiser + scheduler must be rebuilt because parameter groups change.
    """
    stage = int(ckpt["stage"])
    epoch = int(ckpt["epoch"])
    fold = int(ckpt["fold"]) if ckpt.get("fold") is not None else None
    completed = bool(ckpt.get("stage_completed", False))

    if stage == 1:
        if completed or epoch >= cfg["stage1_epochs"]:
            return {"stage": 2, "epoch": 1, "fold": 0, "rebuild_optimizer": True}
        return {"stage": 1, "epoch": epoch + 1, "fold": 0, "rebuild_optimizer": False}
    if stage == 2:
        if completed or epoch >= cfg["stage2_epochs"]:
            return {"stage": 3, "epoch": 1, "fold": 0, "rebuild_optimizer": True}
        return {"stage": 2, "epoch": epoch + 1, "fold": 0, "rebuild_optimizer": False}
    if stage == 3:
        K = int(cfg["folds"])
        if completed:
            return {
                "stage": 3, "epoch": cfg["stage3_epochs"] + 1, "fold": 0,
                "rebuild_optimizer": False, "done": True,
            }
        if fold is None:
            fold = -1
        if fold + 1 < K:
            return {"stage": 3, "epoch": epoch, "fold": fold + 1, "rebuild_optimizer": False}
        if epoch + 1 <= cfg["stage3_epochs"]:
            return {"stage": 3, "epoch": epoch + 1, "fold": 0, "rebuild_optimizer": False}
        return {
            "stage": 3, "epoch": cfg["stage3_epochs"] + 1, "fold": 0,
            "rebuild_optimizer": False, "done": True,
        }
    raise ValueError(f"unknown stage {stage}")


# ---------------------------------------------------------------------------
# Disk hygiene
# ---------------------------------------------------------------------------


def hygiene(out_dir: Path, keep_last_n: int) -> None:
    """Delete rolling per-(epoch, fold) checkpoints older than the last ``keep_last_n``.

    Stage-completion files (``stage{1,2,3}_complete.pt``), ``last.pt``, and
    ``best.pt`` are protected.
    """
    rolling = sorted(
        [
            p for p in out_dir.glob("ckpt_s*.pt")
            if not p.name.startswith("stage")
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if len(rolling) <= keep_last_n:
        return
    for p in rolling[: len(rolling) - keep_last_n]:
        try:
            p.unlink()
        except OSError:
            pass
