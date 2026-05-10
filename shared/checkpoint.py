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

Google Drive durability
-----------------------
When `path` is on Google Drive (detected via the ``/content/drive`` prefix),
``atomic_save`` does extra work to make sure every checkpoint survives a
runtime reset:

  1. Write the payload to a temp file inside the same directory.
  2. ``file.flush()`` followed by ``os.fsync(file.fileno())`` to push bytes
     through the FUSE layer (Drive's FUSE caches writes aggressively; without
     fsync the bytes may sit in the runtime's RAM and disappear when the
     runtime is killed).
  3. ``os.replace(tmp, path)`` to atomically rename.
  4. ``os.fsync()`` the parent directory entry where the OS supports it.
  5. Retry the whole sequence up to 3 times on transient OSError
     (``EIO`` / ``ENOENT`` / ``ENOSPC``-like flapping is common on Colab Drive
     mounts during long-running sessions).

Effect: every per-(epoch, fold) checkpoint is FORCED to disk and visible from
the Drive web UI by the time ``atomic_save`` returns.
"""

from __future__ import annotations

import os
import random as _random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Heuristic for "this path lives on a FUSE-mounted Google Drive on Colab".
_DRIVE_PREFIXES = ("/content/drive/", "/content/gdrive/")


def _is_drive_path(p: Path) -> bool:
    s = str(p.resolve()) if p.exists() else str(p)
    return any(s.startswith(prefix) for prefix in _DRIVE_PREFIXES)


def is_drive_path(p: str | Path) -> bool:
    """Public alias of the Drive-path heuristic. Useful for notebook checks."""
    return _is_drive_path(Path(p))


def audit_drive_directory(out_dir: str | Path, *, expected_globs: tuple[str, ...]) -> dict:
    """Sanity-check what's actually present on Drive for a given output dir.

    Returns a dict ``{"missing": [...], "present": [...]}`` listing which of
    the expected file glob patterns matched on disk. Useful as a post-stage
    smoke check from the notebook.
    """
    out_dir = Path(out_dir)
    present: list[str] = []
    missing: list[str] = []
    for pat in expected_globs:
        matches = sorted(p.name for p in out_dir.glob(pat))
        if matches:
            present.extend(matches)
        else:
            missing.append(pat)
    return {"out_dir": str(out_dir), "present": present, "missing": missing}


def assert_checkpoint_root_on_drive(out_root: str | Path, *, on_colab: bool) -> None:
    """Raise if Colab is in use but ``out_root`` is NOT on Drive.

    Call this from the notebook *before* training, so a misconfigured
    OUT_ROOT (e.g. pointing at the runtime's ephemeral SSD) is caught
    early instead of silently losing every checkpoint when the runtime
    resets.
    """
    out_root = Path(out_root)
    if on_colab and not _is_drive_path(out_root):
        raise RuntimeError(
            f"Colab runtime detected, but checkpoint OUT_ROOT={out_root} "
            f"does not live on Google Drive (expected one of "
            f"{_DRIVE_PREFIXES}). Checkpoints written to the runtime SSD "
            f"will be LOST on runtime reset. Repoint OUT_ROOT into your "
            f"Drive-mounted project root."
        )


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


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory entry (POSIX-only, harmless on Windows)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # fsync on a directory is unsupported on some filesystems (incl. some
        # FUSE mounts). The file-level fsync above is the durability anchor;
        # the directory fsync is purely belt-and-suspenders.
        pass


def _torch_save_with_fsync(obj: Any, tmp_path: Path) -> None:
    """torch.save → flush → fsync, so the bytes reach disk before rename.

    On Drive's FUSE this is what actually moves the bytes off the runtime
    and onto Google's servers — without it, ``torch.save`` returns long
    before the data is durable, and a runtime kill loses the checkpoint.
    """
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some FUSE drivers throw ENOTSUP / EINVAL on fsync.  In that
            # case we have to trust the close() to do the right thing.
            pass


def _local_scratch_dir() -> Path:
    """A directory on the runtime's local SSD for staging Drive writes.

    On Colab this is ``/tmp`` (always present, fast). Elsewhere we use the
    OS tempdir. Created on demand.
    """
    base = Path(os.environ.get("ISS_CKPT_SCRATCH", "/tmp")) / "iss_ckpt_scratch"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _save_local_then_copy(
    obj: Any, dest: Path, *, max_retries: int, retry_backoff_s: float,
) -> None:
    """Stage on local SSD, then copy to Drive with delete-then-rename.

    Background: observed Colab Drive FUSE behaviour where rapid sequential
    saves to a single directory can drop earlier files. Mitigated by:
      1. Single ``torch.save`` to local SSD with full fsync.
      2. ``shutil.copy2`` local → unique partial path on Drive.
      3. Delete the existing destination if any.
      4. Rename partial → final.
      5. fsync the parent directory.
      6. Verify destination size matches local staged size.
    """
    scratch_dir = _local_scratch_dir()
    pid = os.getpid()
    fd, tmp_local_str = tempfile.mkstemp(
        suffix=".pt", prefix=f"ckpt_{pid}_", dir=str(scratch_dir),
    )
    os.close(fd)
    tmp_local = Path(tmp_local_str)
    try:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                _torch_save_with_fsync(obj, tmp_local)
                last_err = None
                break
            except OSError as e:
                last_err = e
                if attempt + 1 < max_retries:
                    time.sleep(retry_backoff_s * (2 ** attempt))
        if last_err is not None:
            raise last_err
        _copy_local_to_drive(
            tmp_local, dest,
            expected_size=tmp_local.stat().st_size,
            max_retries=max_retries, retry_backoff_s=retry_backoff_s,
        )
    finally:
        try:
            if tmp_local.exists():
                tmp_local.unlink()
        except OSError:
            pass


def _save_inplace_atomic(
    obj: Any, path: Path, *, max_retries: int, retry_backoff_s: float,
) -> None:
    """Original same-directory atomic save: tmp + os.replace, with fsync.

    Used for non-Drive destinations (local disk, runtime SSD), where the
    standard pattern is fast and reliable.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            _torch_save_with_fsync(obj, tmp)
            os.replace(str(tmp), str(path))
            _fsync_dir(path.parent)
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
                wait = retry_backoff_s * (2 ** attempt)
                print(
                    f"  ⚠ atomic_save: transient OSError on {path} "
                    f"(attempt {attempt + 1}/{max_retries}, retrying in {wait:.1f}s): {e}",
                    flush=True,
                )
                time.sleep(wait)
    if last_err is not None:
        raise last_err


def atomic_save(
    obj: Any,
    path: Path,
    *,
    quiet: bool = False,
    label: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> None:
    """Atomic, durable save of ``obj`` to ``path``.

    Two strategies depending on destination:

      - **Drive destination** (``/content/drive/...``): stage the file on
        the local SSD first with full fsync, then copy to Drive with a
        unique ``.partial.<pid>.<nstime>`` suffix, then delete-then-rename
        to the final destination. This avoids two known Drive FUSE bugs:
        (a) ``os.replace`` over an existing Drive file occasionally drops
            *another* file in the same directory; (b) rapid back-to-back
            renames in the same Drive folder can lose the earlier target.
      - **Local destination**: tmp + ``os.replace`` (atomic rename within
        the directory) with flush + fsync.

    Both paths retry up to ``max_retries`` times with exponential back-off
    on transient ``OSError``.

    Prints a single-line confirmation with the file size, unless ``quiet``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if _is_drive_path(path):
        _save_local_then_copy(
            obj, path,
            max_retries=max_retries, retry_backoff_s=retry_backoff_s,
        )
    else:
        _save_inplace_atomic(
            obj, path,
            max_retries=max_retries, retry_backoff_s=retry_backoff_s,
        )

    if not quiet:
        try:
            mb = path.stat().st_size / (1024 * 1024)
            size_str = f"  ({mb:.1f} MB)"
        except OSError:
            size_str = ""
        prefix = f"[{label}] " if label else ""
        on_drive = " [Drive]" if _is_drive_path(path) else ""
        print(f"  ✓ {prefix}saved checkpoint{on_drive}: {path}{size_str}", flush=True)


def atomic_save_multi(
    obj: Any,
    targets: list[tuple[Path, str]],
    *,
    quiet: bool = False,
    max_retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> None:
    """Save the same payload to multiple destinations using ONE local stage.

    targets: list of (destination_path, label) tuples.

    On Drive: ``torch.save`` is called exactly once (to a local SSD scratch
    file). The same staged file is then copied to every destination with a
    unique partial-name + delete-then-rename sequence. This avoids the
    back-to-back ``os.replace`` pattern that has been observed to drop
    Drive files.

    On non-Drive paths: each destination is saved via the regular
    ``atomic_save`` path.

    Verification: after each Drive write the destination's size is checked
    against the local staged file's size; if they don't match, that target
    is retried.
    """
    if not targets:
        return

    # Mkdir all parents up front.
    for dest, _ in targets:
        dest.parent.mkdir(parents=True, exist_ok=True)

    drive_targets = [(d, l) for d, l in targets if _is_drive_path(d)]
    local_targets = [(d, l) for d, l in targets if not _is_drive_path(d)]

    if drive_targets:
        # Stage once on local SSD, then copy to each Drive destination.
        scratch_dir = _local_scratch_dir()
        pid = os.getpid()
        fd, tmp_local_str = tempfile.mkstemp(
            suffix=".pt", prefix=f"ckpt_multi_{pid}_", dir=str(scratch_dir),
        )
        os.close(fd)
        tmp_local = Path(tmp_local_str)
        try:
            # Single torch.save to local SSD.
            last_err: Exception | None = None
            for attempt in range(max_retries):
                try:
                    _torch_save_with_fsync(obj, tmp_local)
                    last_err = None
                    break
                except OSError as e:
                    last_err = e
                    if attempt + 1 < max_retries:
                        time.sleep(retry_backoff_s * (2 ** attempt))
            if last_err is not None:
                raise last_err
            staged_size = tmp_local.stat().st_size

            # Copy local → each Drive destination.
            for dest, label in drive_targets:
                _copy_local_to_drive(
                    tmp_local, dest, expected_size=staged_size,
                    max_retries=max_retries, retry_backoff_s=retry_backoff_s,
                )
                if not quiet:
                    try:
                        mb = dest.stat().st_size / (1024 * 1024)
                        size_str = f"  ({mb:.1f} MB)"
                    except OSError:
                        size_str = ""
                    prefix = f"[{label}] " if label else ""
                    print(
                        f"  ✓ {prefix}saved checkpoint [Drive]: {dest}{size_str}",
                        flush=True,
                    )
        finally:
            try:
                if tmp_local.exists():
                    tmp_local.unlink()
            except OSError:
                pass

    if local_targets:
        # Local writes: do each one separately; they're cheap and reliable.
        for dest, label in local_targets:
            _save_inplace_atomic(
                obj, dest,
                max_retries=max_retries, retry_backoff_s=retry_backoff_s,
            )
            if not quiet:
                try:
                    mb = dest.stat().st_size / (1024 * 1024)
                    size_str = f"  ({mb:.1f} MB)"
                except OSError:
                    size_str = ""
                prefix = f"[{label}] " if label else ""
                print(
                    f"  ✓ {prefix}saved checkpoint: {dest}{size_str}",
                    flush=True,
                )

    # Final verification: every destination must exist.
    missing = [str(d) for d, _ in targets if not d.exists()]
    if missing:
        raise RuntimeError(
            f"atomic_save_multi: post-save verification failed; "
            f"these targets do not exist on disk: {missing}"
        )


def _copy_local_to_drive(
    src_local: Path, dest: Path,
    *, expected_size: int, max_retries: int, retry_backoff_s: float,
) -> None:
    """Copy a local file to a Drive destination using delete-then-rename.

    Verifies destination size after rename; raises if it doesn't match.
    """
    pid = os.getpid()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        # Each retry uses a fresh partial name Drive has never seen.
        unique = f"{pid}.{time.time_ns()}.{attempt}"
        partial = dest.with_suffix(dest.suffix + f".partial.{unique}")
        try:
            shutil.copy2(str(src_local), str(partial))
            # fsync the partial file on Drive.
            try:
                fdp = os.open(str(partial), os.O_RDONLY)
                try:
                    os.fsync(fdp)
                finally:
                    os.close(fdp)
            except OSError:
                pass
            # Verify partial wrote the full payload.
            partial_size = partial.stat().st_size
            if partial_size != expected_size:
                raise OSError(
                    f"short write to Drive: partial size {partial_size} "
                    f"!= expected {expected_size}"
                )
            # Delete-then-rename to avoid Drive's rename-over-existing edge cases.
            try:
                if dest.exists():
                    dest.unlink()
            except OSError:
                pass
            os.replace(str(partial), str(dest))
            _fsync_dir(dest.parent)
            # Final verification.
            final_size = dest.stat().st_size
            if final_size != expected_size:
                raise OSError(
                    f"size mismatch after rename: {final_size} != {expected_size}"
                )
            last_err = None
            break
        except OSError as e:
            last_err = e
            try:
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass
            if attempt + 1 < max_retries:
                wait = retry_backoff_s * (2 ** attempt)
                print(
                    f"  ⚠ atomic_save: Drive copy of {dest} failed "
                    f"(attempt {attempt + 1}/{max_retries}, retrying in {wait:.1f}s): {e}",
                    flush=True,
                )
                time.sleep(wait)
    if last_err is not None:
        raise last_err


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



