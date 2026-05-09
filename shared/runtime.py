"""GPU / process runtime utilities.

When a training run is interrupted (KeyboardInterrupt) or raises, Python may
keep model / optimizer / gradient references alive on the GPU until the
enclosing scope exits. In a Jupyter / Colab kernel that scope often persists
for the kernel's lifetime, leaving VRAM stuck.

``with gpu_cleanup_on_exit():`` wraps a block so that:
  - on KeyboardInterrupt: print, run aggressive gc + empty_cache, re-raise.
  - on Exception        : same.
  - on normal exit      : just empty_cache (cheap; reduces fragmentation
                          across back-to-back stages).

The trainer entry points wrap their main loop with this. In a notebook,
hitting Ctrl-C / kernel-interrupt frees VRAM without a kernel restart.

``release_gpu_memory()`` does the same cleanup explicitly. Use it from a
notebook cell after a long-running call has returned, when you want the
VRAM back without restarting.

Important caveat: there is no perfectly reliable way for this helper to
discard tensor references held in a *caller's* local variables. CPython's
fast-locals can't be mutated from outside the function. The cleanup relies
on:
  1. The exception unwind (in `gpu_cleanup_on_exit`) deleting all frames
     between the cleanup site and the raise site, dropping references those
     frames held.
  2. ``gc.collect()`` then reclaiming any cyclic references.
  3. ``torch.cuda.empty_cache()`` returning the freed VRAM to the driver.

This is sufficient when called at the top of a training entry point: by
the time the cleanup runs, the train function's frame is being unwound, so
its model / optimizer / dataloader locals are released as the `raise`
propagates.
"""

from __future__ import annotations

import contextlib
import gc
from typing import Iterator


def _vram_stats() -> tuple[float, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved() / 1024 ** 2,
        )
    except ImportError:                                                       # pragma: no cover
        return 0.0, 0.0


def release_gpu_memory(*, verbose: bool = True) -> None:
    """Run cyclic GC and return VRAM to the driver.

    Best called either:
      - automatically by ``gpu_cleanup_on_exit`` during exception unwind, or
      - manually by the user from a notebook cell when training is done.
    """
    try:
        import torch
    except ImportError:                                                       # pragma: no cover
        if verbose:
            print("  release_gpu_memory: torch not available, skipping.")
        return

    before_alloc, before_reserved = _vram_stats()

    for _ in range(3):
        gc.collect()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:                                                     # noqa: BLE001
            pass
        after_alloc, after_reserved = _vram_stats()
        if verbose:
            print(
                f"  release_gpu_memory: "
                f"allocated {before_alloc:.0f} → {after_alloc:.0f} MB, "
                f"reserved {before_reserved:.0f} → {after_reserved:.0f} MB",
                flush=True,
            )
    elif verbose:
        print("  release_gpu_memory: no CUDA device.", flush=True)


@contextlib.contextmanager
def gpu_cleanup_on_exit(*, verbose: bool = True) -> Iterator[None]:
    """Context manager that runs ``release_gpu_memory`` on any exit path.

    Use to wrap training loops::

        with gpu_cleanup_on_exit():
            run_training()

    On KeyboardInterrupt or Exception the caller's frames are still live
    when ``__exit__`` runs (Python unwinds AFTER ``__exit__`` returns), so
    a small fraction of VRAM may remain held until the exception fully
    propagates and Python's reference counting / cyclic GC collects the
    dead frames.

    To force a final reclaim after the unwind, the trainer entry points
    (which invoke this context manager) finish with a `try/finally` that
    re-invokes :func:`release_gpu_memory`. The notebook-level
    ``release_gpu_memory()`` cell at the bottom of ``modeling.ipynb`` can
    also be used as a manual final reclaim.
    """
    interrupted = False
    failed = False
    try:
        yield
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\n⏹  training interrupted by user; releasing GPU VRAM …", flush=True)
        raise
    except Exception:                                                         # noqa: BLE001
        failed = True
        if verbose:
            print("\n💥 training raised an exception; releasing GPU VRAM …", flush=True)
        raise
    finally:
        # Two-stage cleanup: first while we're still inside the trainer
        # function's frame (best-effort), then re-run via the trainer's own
        # `finally` once that frame has died. Empty cache on normal exit too
        # so back-to-back stages don't accumulate fragmentation.
        release_gpu_memory(verbose=verbose and (interrupted or failed))
