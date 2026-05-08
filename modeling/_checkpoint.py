"""Universal checkpoint I/O for the OWLv2 trainer."""

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
    if resume is False or resume is None:
        return None
    if resume is True:
        p = out_dir / "last.pt"
        return p if p.exists() else None
    p = Path(resume)                                                          # type: ignore[arg-type]
    if not p.is_absolute():
        p = out_dir / p
    return p if p.exists() else None


def try_load_model_state(model: torch.nn.Module, ckpt_model: dict) -> tuple[bool, str]:
    """Load the per-component state_dicts from a checkpoint.

    Strict-by-shape: returns (False, reason) on any shape mismatch so the
    caller can quarantine + start fresh rather than silently fall back to
    a fresh init for mismatched layers.
    """
    own = model.state_dict()
    state: dict[str, torch.Tensor] = {}
    if "aggregator" in ckpt_model:
        for k, v in ckpt_model["aggregator"].items():
            state[f"aggregator.{k}"] = v
    if "existence_head" in ckpt_model:
        for k, v in ckpt_model["existence_head"].items():
            state[f"existence_head.{k}"] = v
    # Top-level scalar parameter (residual aggregator gate).  Stored under
    # its own key in the ckpt dict so older checkpoints without it keep
    # loading at alpha=0 (the default init).
    if "aggregator_alpha" in ckpt_model:
        state["aggregator_alpha"] = ckpt_model["aggregator_alpha"]
    if ckpt_model.get("class_head"):
        for k, v in ckpt_model["class_head"].items():
            state[f"owlv2.class_head.{k}"] = v
    if ckpt_model.get("box_head"):
        for k, v in ckpt_model["box_head"].items():
            state[f"owlv2.box_head.{k}"] = v
    if ckpt_model.get("layer_norm"):
        for k, v in ckpt_model["layer_norm"].items():
            state[f"owlv2.layer_norm.{k}"] = v
    if ckpt_model.get("lora_state"):
        # LoRA keys are wrapped by peft — they live under their own paths
        # inside model.owlv2.  We just write them in directly.
        for k, v in ckpt_model["lora_state"].items():
            state[k] = v

    mismatches: list[str] = []
    for k, v in state.items():
        if k in own and own[k].shape != v.shape:
            mismatches.append(
                f"{k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}"
            )
    if mismatches:
        head = "; ".join(mismatches[:5])
        tail = f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        return False, head + tail
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        return True, ""
    except RuntimeError as e:
        return False, str(e)


def save_model_state(model: torch.nn.Module, lora_active: bool) -> dict:
    """Pull the trainable component state_dicts out of ``model``."""
    out: dict[str, Any] = {
        "aggregator": model.aggregator.state_dict(),
        "existence_head": model.existence_head.state_dict(),
        # Top-level residual aggregator gate (scalar tensor).
        "aggregator_alpha": model.aggregator_alpha.detach().cpu(),
    }
    # OWLv2 heads — only saved if currently unfrozen (any param requires grad).
    head_trainable = any(p.requires_grad for p in model.owlv2.class_head.parameters())
    if head_trainable:
        out["class_head"] = model.owlv2.class_head.state_dict()
        out["box_head"] = model.owlv2.box_head.state_dict()
        out["layer_norm"] = model.owlv2.layer_norm.state_dict()
    if lora_active:
        # Save only LoRA-tagged parameters from owlv2.
        out["lora_state"] = {
            n: p.detach().cpu()
            for n, p in model.owlv2.named_parameters()
            if "lora_" in n
        }
    return out


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
    print(
        f"⚠ Quarantined {len(moved)} incompatible checkpoint(s) into {backup}\n"
        f"  reason: {reason}\n"
        f"  files : {', '.join(moved) if moved else '(none)'}"
    )
    return backup


def hygiene(out_dir: Path, keep_last_n: int) -> None:
    """Delete rolling per-(epoch, fold) checkpoints older than the last
    ``keep_last_n``.  Stage-completion / best / last are protected.
    """
    rolling = sorted(
        [
            p for p in out_dir.glob("ckpt_fold*_epoch*.pt")
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
