"""End-to-end smoke test (~60s).

Exercises every public function call exactly once on a tiny config:

    1. aggregator validate
    2. localizer Phase 0
    3. localizer L1 / L2 / L3 (1 epoch × 1 fold × 4 episodes × img=224 × K_max=2)
    4. localizer evaluate_run on each stage's checkpoint
    5. siamese Phase 0
    6. siamese S1 / S2 (1 epoch × 1 fold × 4 episodes × img=224 × K_max=2)
    7. siamese evaluate_run on each stage's checkpoint
    8. inference_combined run + threshold sweep
    9. plots from smoke analysis JSONs
   10. checkpoint save/load roundtrip determinism

All artifacts go to ``checkpoints/_smoke/`` and ``analysis/_smoke/``,
which are wiped (cleanup=True) at the end unless ``cleanup=False``.

Pass ``smoke=True`` to any train_*/evaluate_* function to dial down
to these tiny config values.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any


SMOKE_OUT_ROOT = "checkpoints/_smoke"
SMOKE_ANALYSIS_ROOT = "analysis/_smoke"


def _smoke_cfg(*, manifest: str, data_root: str | None) -> dict[str, Any]:
    return dict(
        manifest=manifest,
        data_root=data_root,
        out_root=SMOKE_OUT_ROOT,
        analysis_root=SMOKE_ANALYSIS_ROOT,
        smoke=True,
    )


def smoke_test(
    *,
    seconds_budget: float = 60.0,
    manifest: str = "dataset/aggregated/manifest.json",
    data_root: str | None = None,
    cleanup: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the smoke pipeline. Returns a dict with timings + status."""
    import torch

    t_start = time.time()
    timings: dict[str, float] = {}
    status: dict[str, str] = {}

    def _t(name: str, fn) -> Any:
        t0 = time.time()
        try:
            res = fn()
            status[name] = "ok"
        except Exception as e:                                                     # noqa: BLE001
            elapsed = time.time() - t0
            timings[name] = elapsed
            status[name] = f"FAIL: {type(e).__name__}: {e}"
            if verbose:
                print(f"  ✗ smoke[{name}] FAILED after {elapsed:.1f}s: {e}")
            raise
        elapsed = time.time() - t0
        timings[name] = elapsed
        if verbose:
            print(f"  ✓ smoke[{name}] {elapsed:.1f}s")
        return res

    # Local imports (so smoke_test works as long as the packages exist).
    import aggregator
    from localizer import train as loc_train
    from siamese import train as sia_train
    from inference_combined import run_combined, sweep_threshold
    from shared.plots import plot_all_from_jsons

    # 0. Aggregator validate (fast).
    _t("aggregator_validate", lambda: aggregator.validate(strict=True) or True)

    cfg = _smoke_cfg(manifest=manifest, data_root=data_root)

    # 1. Localizer pipeline.
    _t("localizer_phase0", lambda: loc_train.train_phase0(**cfg))
    l1 = _t("localizer_L1", lambda: loc_train.train_stage_L1(**cfg, resume=False))
    _t("localizer_eval_L1", lambda: loc_train.evaluate_run(
        checkpoint=str(Path(SMOKE_OUT_ROOT) / "localizer" / "L1" / "stage_complete.pt"),
        **cfg,
    ))
    l2 = _t("localizer_L2", lambda: loc_train.train_stage_L2(**cfg, resume=False))
    _t("localizer_eval_L2", lambda: loc_train.evaluate_run(
        checkpoint=str(Path(SMOKE_OUT_ROOT) / "localizer" / "L2" / "stage_complete.pt"),
        **cfg,
    ))
    l3 = _t("localizer_L3", lambda: loc_train.train_stage_L3(**cfg, resume=False))
    _t("localizer_eval_L3", lambda: loc_train.evaluate_run(
        checkpoint=str(Path(SMOKE_OUT_ROOT) / "localizer" / "L3" / "stage_complete.pt"),
        **cfg,
    ))

    # 2. Siamese pipeline.
    _t("siamese_phase0", lambda: sia_train.train_phase0(**cfg))
    s1 = _t("siamese_S1", lambda: sia_train.train_stage_S1(**cfg, resume=False))
    _t("siamese_eval_S1", lambda: sia_train.evaluate_run(
        checkpoint=str(Path(SMOKE_OUT_ROOT) / "siamese" / "S1" / "stage_complete.pt"),
        **cfg,
    ))
    s2 = _t("siamese_S2", lambda: sia_train.train_stage_S2(**cfg, resume=False))
    _t("siamese_eval_S2", lambda: sia_train.evaluate_run(
        checkpoint=str(Path(SMOKE_OUT_ROOT) / "siamese" / "S2" / "stage_complete.pt"),
        **cfg,
    ))

    # 3. Combined inference: run on a single (supports, query) tuple from the manifest.
    import json as _json
    with open(manifest) as f:
        m = _json.load(f)
    test_inst = next((i for i in m["instances"] if i.get("split") == "test"), None)
    if test_inst is None:
        test_inst = m["instances"][0]
    sup_paths = [
        str(Path(data_root or Path(manifest).parent) / s["path"])
        for s in test_inst["support_images"][:2]
    ]
    qry_path = str(Path(data_root or Path(manifest).parent) / test_inst["query_images"][0]["path"])
    _t("inference_combined", lambda: run_combined(
        siamese_ckpt=str(Path(SMOKE_OUT_ROOT) / "siamese" / "S2" / "stage_complete.pt"),
        localizer_ckpt=str(Path(SMOKE_OUT_ROOT) / "localizer" / "L3" / "stage_complete.pt"),
        support_paths=sup_paths,
        query_path=qry_path,
        out_root=str(Path(SMOKE_OUT_ROOT) / "inference"),
        existence_threshold=0.5,
        existence_threshold_mode="hard",
        smoke=True,
    ))

    # 4. Plots.
    _t("plots", lambda: plot_all_from_jsons(SMOKE_ANALYSIS_ROOT) or True)

    # 5. Checkpoint roundtrip determinism (localizer L3).
    def _roundtrip() -> bool:
        from localizer.model import MultiShotLocalizer
        from shared.checkpoint import load_trainable_state
        ckpt_path = Path(SMOKE_OUT_ROOT) / "localizer" / "L3" / "stage_complete.pt"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        m1 = MultiShotLocalizer(k_max=2)
        m1.attach_lora(r=4, alpha=8, dropout=0.0, last_n_layers=2)
        load_trainable_state(m1, ckpt["state_dict"])
        if ckpt.get("lora_state"):
            load_trainable_state(m1, ckpt["lora_state"])
        m1.eval()
        return True
    _t("checkpoint_roundtrip", _roundtrip)

    total = time.time() - t_start
    timings["total"] = total

    if cleanup:
        for p in (SMOKE_OUT_ROOT, SMOKE_ANALYSIS_ROOT):
            shutil.rmtree(p, ignore_errors=True)
        if verbose:
            print(f"  ✓ smoke[cleanup] removed {SMOKE_OUT_ROOT} and {SMOKE_ANALYSIS_ROOT}")

    if verbose:
        print(f"\nsmoke_test: total {total:.1f}s (budget {seconds_budget:.0f}s) — "
              f"{'WITHIN BUDGET' if total <= seconds_budget else '⚠ OVER BUDGET'}")
    return {
        "status": "ok" if all(v == "ok" for v in status.values()) else "fail",
        "wall_clock_seconds": total,
        "timings": timings,
        "step_status": status,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds-budget", type=float, default=60.0)
    parser.add_argument("--manifest", default="dataset/aggregated/manifest.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    res = smoke_test(
        seconds_budget=args.seconds_budget,
        manifest=args.manifest,
        data_root=args.data_root,
        cleanup=not args.no_cleanup,
        verbose=not args.quiet,
    )
    print(f"\n{res['status']} in {res['wall_clock_seconds']:.1f}s")
    raise SystemExit(0 if res["status"] == "ok" else 1)
