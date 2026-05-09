"""Universal checkpoint I/O for both localizer and siamese.

Single payload format (single torch.save dict):

    {
      "model_kind": "localizer" | "siamese",
      "stage":      str,           # e.g. "L1" / "S2" / "phase0"
      "epoch":      int,
      "fold":       int,
      "stage_completed": bool,
      "global_step": int,
      "state_dict": flat_state_dict_of_TRAINABLE_components_only,
      "lora_state": dict | None,    # peft adapter weights, separated for clarity
      "optimizer":  optimizer.state_dict() | None,
      "scheduler":  scheduler.state_dict() | None,
      "scaler":     scaler.state_dict() | None,
      "rng":        {torch, numpy, python, cuda},
      "config":     full_resolved_cfg_dict,
      "fold_plan":  list,
      "metrics_history": list,
      "best_metric":     {"value": float, "epoch": int, "fold": int},
      "early_stop_counter": int,
    }

Resume rules:
- ``resume=True``  -> load ``out_dir/last.pt`` if present, else fresh.
- ``resume=False`` -> always fresh.
- ``resume=<str>`` -> resolve as path (absolute or relative to out_dir).

Cross-stage warm start: when the loaded ckpt is a previous stage's
``stage_complete.pt``, only the trainable component state is loaded (no
optimizer / scheduler restore).
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


def atomic_save(obj: Any, path: Path, *, quiet: bool = False, label: str | None = None) -> None:
    """Atomic save of ``obj`` to ``path``.

    Prints a single-line confirmation with the file size after writing, unless
    ``quiet=True``. The optional ``label`` is prefixed before the path (e.g.
    ``label="rolling"`` produces ``[rolling] saved …``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))
    if not quiet:
        try:
            mb = path.stat().st_size / (1024 * 1024)
            size_str = f"  ({mb:.1f} MB)"
        except OSError:
            size_str = ""
        prefix = f"[{label}] " if label else ""
        print(f"  ✓ {prefix}saved checkpoint: {path}{size_str}", flush=True)


def resolve_resume_path(resume: bool | str | None, out_dir: Path) -> Path | None:
    if resume is False or resume is None:
        return None
    if resume is True:
        p = out_dir / "last.pt"
        return p if p.exists() else None
    p = Path(resume)
    if not p.is_absolute():
        p = out_dir / p
    return p if p.exists() else None


def hygiene(out_dir: Path, keep_last_n: int) -> None:
    """Delete rolling per-(epoch, fold) checkpoints older than the last N.

    keep_last_n <= 0 disables the cull entirely.
    Stage-completion / best / last are protected.
    """
    if keep_last_n <= 0:
        return
    rolling = sorted(
        list(out_dir.glob("ckpt_fold*_epoch*.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    if len(rolling) <= keep_last_n:
        return
    for p in rolling[: len(rolling) - keep_last_n]:
        try:
            p.unlink()
        except OSError:
            pass


def quarantine_incompatible(out_dir: Path, reason: str) -> Path:
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
    print(f"⚠ Quarantined {len(moved)} incompatible ckpt(s) → {backup}\n  reason: {reason}")
    return backup


# ---------------------------------------------------------------------------
# Trainable-only state extraction / loading
# ---------------------------------------------------------------------------


def get_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a state_dict containing only currently-trainable parameters
    plus their associated buffers (per module).

    A module is considered "trainable" if any of its direct parameters has
    requires_grad=True. We also save buffers of those modules.
    """
    out: dict[str, torch.Tensor] = {}
    sd = model.state_dict()
    # Include any param whose requires_grad is True.
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    # Also include LoRA buffers / params (they may be tagged 'lora_').
    for name in trainable_param_names:
        if name in sd:
            out[name] = sd[name].detach().cpu()
    # Buffers belonging to modules that have any trainable param.
    trainable_module_prefixes = set()
    for name in trainable_param_names:
        # walk up module hierarchy
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            trainable_module_prefixes.add(".".join(parts[:i - 1]))
    for buf_name, buf in model.named_buffers():
        prefix = ".".join(buf_name.split(".")[:-1])
        if prefix in trainable_module_prefixes:
            out[buf_name] = buf.detach().cpu()
    return out


def load_trainable_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor], *, strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a flat state_dict into the model. Returns (missing, unexpected).

    Mismatched-shape keys are skipped with a warning rather than raising.
    """
    own = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape != v.shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}")
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        head = "; ".join(skipped[:5])
        tail = f" (and {len(skipped) - 5} more)" if len(skipped) > 5 else ""
        print(f"⚠ load_trainable_state skipped {len(skipped)} mismatched keys: {head}{tail}")
    return list(missing), list(unexpected)



