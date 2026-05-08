"""Training orchestrator for the OWLv2 few-shot localizer.

Public entry points (matched 1:1 with notebook cells):

    train_phase0(...)        zero-shot OWLv2 baseline on vizwiz_novel + test
    train_stage_1_1(...)     aggregator + existence head warmup
    train_stage_1_2(...)     + box / class heads
    train_stage_2_3(...)     + LoRA on last 4 vision blocks

    evaluate_phase0(...)     re-run the Phase 0 baseline on the test split
    evaluate_run(...)        load a checkpoint, evaluate on test split

Each training entry point runs ``stage_epochs × K folds`` per the
contract in PLAN.md §6.5: every epoch performs all K folds sequentially
with shared running model weights.

Checkpoints land under ``checkpoints/<stage>/`` and analytics under
``analysis/<stage>/``.
"""

from __future__ import annotations

import json
import random as _random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from modeling._analysis import (
    aggregate_folds,
    update_summary,
    write_json,
)
from modeling._checkpoint import (
    atomic_save,
    capture_rng,
    hygiene,
    quarantine_incompatible,
    resolve_resume_path,
    restore_rng,
    save_model_state,
    try_load_model_state,
)
from modeling._folds import stratified_kfold
from modeling._loaders import (
    build_phase0_loader,
    build_train_loader,
    build_val_loader,
)
from modeling._logging import print_aggregate, print_epoch_log
from modeling._optim import build_optimizer_for_stage, build_scheduler
from modeling._train_loop import train_one_pass
from modeling.dataset import collate
from modeling.evaluate import (
    DEFAULT_TILE_CFG,
    evaluate,
    evaluate_phase0 as _evaluate_phase0,
    resolve_tile_cfg,
)
from modeling.model import OWLv2FewShotLocalizer


DEFAULT_CFG: dict[str, Any] = {
    # --- data -----------------------------------------------------------
    "manifest": "dataset/aggregated/manifest.json",
    "data_root": None,
    "img_size": 768,
    "n_support": 4,
    "neg_prob": 0.5,
    "batch_size": 1,
    "num_workers": 2,
    "grad_accum_steps": 8,
    # --- folds ----------------------------------------------------------
    "folds": 5,
    "fold_seed": 42,
    # --- stage durations / volumes -------------------------------------
    "stage_1_1_epochs": 8,
    "stage_1_2_epochs": 15,
    "stage_2_3_epochs": 8,
    "episodes_per_fold_s1": 200,
    "episodes_per_fold_s2": 250,
    "episodes_per_fold_s3": 250,
    "val_episodes": 100,
    # --- optim ----------------------------------------------------------
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_frac": 0.05,
    # --- LRs (per stage; trainer picks the right key) ------------------
    "lr_aggregator_s1": 5e-4,
    "lr_existence_s1":  5e-4,
    "lr_aggregator_s2": 1e-4,
    "lr_existence_s2":  2e-4,
    "lr_box_s2":        5e-5,
    "lr_class_s2":      5e-5,
    "lr_aggregator_s3": 5e-5,
    "lr_existence_s3":  5e-5,
    "lr_box_s3":        2e-5,
    "lr_class_s3":      2e-5,
    "lr_lora_s3":       1e-4,
    # --- LoRA -----------------------------------------------------------
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_layers": 4,
    # --- loss -----------------------------------------------------------
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "lambda_l1": 5.0,
    "lambda_giou": 2.0,
    "anti_collapse_weight": 0.1,
    "box_size_threshold": 0.6,
    "existence_kl_threshold": 0.85,
    # --- AMP ------------------------------------------------------------
    "use_amp": True,
    # --- I/O ------------------------------------------------------------
    "out_root": "checkpoints",
    "analysis_root": "analysis",
    "keep_last_n": 6,
    # --- early stopping --------------------------------------------------
    "early_stop_patience_s1": 4,
    "early_stop_patience_s2": 5,
    "early_stop_patience_s3": 4,
    # --- tile inference (eval-only) -------------------------------------
    "tile_cfg": dict(DEFAULT_TILE_CFG),     # default: pyramid_a on insdet
    # In-loop val cycles use single-pass eval (cheap).  Final post-stage
    # evaluate_run() uses ``tile_cfg`` instead.
    "in_loop_val_tile_cfg": {"mode": "off"},
    # --- misc -----------------------------------------------------------
    "seed": 42,
    "device": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(cfg: dict) -> torch.device:
    dev = cfg.get("device")
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _stage_lrs(stage: str, cfg: dict) -> dict:
    """Return cfg with single-stage LR keys filled in."""
    suffix = {"1_1": "s1", "1_2": "s2", "2_3": "s3"}[stage]
    out = dict(cfg)
    for k in ("aggregator", "existence", "box", "class", "lora"):
        full_key = f"lr_{k}_{suffix}"
        if full_key in cfg:
            out[f"lr_{k}"] = cfg[full_key]
        else:
            out.setdefault(f"lr_{k}", 0.0)
    return out


def _merge_cfg(user_kwargs: dict) -> dict:
    cfg = dict(DEFAULT_CFG)
    for k, v in user_kwargs.items():
        cfg[k] = v
    return cfg


def _stage_dir_name(stage: str) -> str:
    if stage == "phase0":
        return "phase0"
    return f"stage_{stage}"


def _stage_dirs(cfg: dict, stage: str) -> tuple[Path, Path]:
    name = _stage_dir_name(stage)
    out_dir = Path(cfg["out_root"]) / name
    analysis_dir = Path(cfg["analysis_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, analysis_dir


def _make_scaler(use_amp: bool, device: torch.device) -> "torch.amp.GradScaler | None":
    if use_amp and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def _load_model_from_ckpt(
    model: OWLv2FewShotLocalizer, ckpt_path: Path, out_dir: Path, lora_active: bool
) -> bool:
    """Load model weights from ``ckpt_path``.  Returns True on success.

    On shape mismatch, quarantine the entire ``out_dir`` and return False.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ok, err = try_load_model_state(model, ckpt.get("model", {}))
    if not ok:
        quarantine_incompatible(out_dir, reason=err)
        return False
    return True


def _save_ckpt(
    *,
    out_dir: Path,
    stage: str,
    epoch: int,
    fold: int,
    model: OWLv2FewShotLocalizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: "torch.amp.GradScaler | None",
    cfg: dict,
    fold_plan: list[dict],
    best_metric: dict,
    early_stop_counter: int,
    metrics_history: list,
    stage_completed: bool = False,
    lora_active: bool = False,
) -> dict:
    ckpt = {
        "stage": stage,
        "epoch": epoch,
        "fold": fold,
        "stage_completed": stage_completed,
        "global_step": int(scheduler.last_epoch) if scheduler is not None else 0,
        "model": save_model_state(model, lora_active=lora_active),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_aggregate_map_50": best_metric["value"],
        "early_stop_counter": early_stop_counter,
        "rng": capture_rng(),
        "config": cfg,
        "fold_plan": fold_plan,
        "metrics_history": metrics_history,
    }
    return ckpt


# ---------------------------------------------------------------------------
# Phase 0 — zero-shot baseline
# ---------------------------------------------------------------------------


def train_phase0(**user_kwargs) -> dict:
    """Run the zero-shot OWLv2 baseline on phase0 (vizwiz_novel) + test.

    Returns the merged metric dict.  Writes to:
        checkpoints/phase0/results.json
        analysis/phase0/results.json
    """
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, "phase0")
    print(f"=== Phase 0 (zero-shot OWLv2) on {device} ===")

    tile_cfg = resolve_tile_cfg(cfg.get("tile_cfg"))
    needs_native = tile_cfg["mode"] != "off"

    _set_seed(int(cfg["seed"]))
    model = OWLv2FewShotLocalizer().to(device)
    owlv2 = model.owlv2

    # vizwiz_novel via Phase0Dataset (rotation synthesis).
    p0_ds, p0_loader = build_phase0_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="phase0", sources=None,
        batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]),
        return_native=needs_native,
    )
    print(f"phase0/vizwiz_novel: {len(p0_ds)} instances")

    metrics: dict[str, Any] = {}
    if len(p0_ds) > 0:
        t0 = time.time()
        metrics["vizwiz_novel"] = _evaluate_phase0(
            owlv2, p0_loader, device,
            img_size=int(cfg["img_size"]), tile_cfg=tile_cfg,
        )
        metrics["vizwiz_novel"]["wall_clock_seconds"] = round(time.time() - t0, 2)
        print(f"vizwiz_novel done in {metrics['vizwiz_novel']['wall_clock_seconds']}s")

    # HOTS + InsDet test.
    test_ds, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None,
        val_episodes=int(cfg.get("val_episodes_phase0", 200)),
        batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
        neg_prob=float(cfg["neg_prob"]),
        img_size=int(cfg["img_size"]),
        seed=int(cfg["seed"]),
        n_support=int(cfg["n_support"]),
        return_native=needs_native,
    )
    print(f"phase0/test: {len(test_ds.instances)} instances, "
          f"{len(test_ds)} episodes")
    if len(test_ds.instances) > 0:
        t0 = time.time()
        metrics["test"] = _evaluate_phase0(
            owlv2, test_loader, device,
            img_size=int(cfg["img_size"]), tile_cfg=tile_cfg,
        )
        metrics["test"]["wall_clock_seconds"] = round(time.time() - t0, 2)
        print(f"test done in {metrics['test']['wall_clock_seconds']}s")

    write_json(out_dir / "results.json", metrics)
    write_json(analysis_dir / "results.json", metrics)
    print("Phase 0 complete. Results written to checkpoints/phase0/results.json")
    return metrics


# ---------------------------------------------------------------------------
# Generic stage runner
# ---------------------------------------------------------------------------


def _run_stage(
    stage: str,
    *,
    user_kwargs: dict,
    use_box_loss: bool,
    epochs_key: str,
    episodes_key: str,
) -> dict:
    cfg = _merge_cfg(user_kwargs)
    cfg = _stage_lrs(stage, cfg)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, stage)

    print(f"\n=== Stage {stage} on {device} ===")

    _set_seed(int(cfg["seed"]))
    model = OWLv2FewShotLocalizer().to(device)

    # --- resume ---------------------------------------------------------
    resume = user_kwargs.get("resume", True)
    resume_path = resolve_resume_path(resume, out_dir)
    if resume_path is None:
        # Look for the previous stage's stage_complete.pt for warm start.
        prev_map = {"1_1": None, "1_2": "1_1", "2_3": "1_2"}
        prev = prev_map.get(stage)
        if prev:
            prev_complete = Path(cfg["out_root"]) / _stage_dir_name(prev) / "stage_complete.pt"
            if prev_complete.exists():
                print(f"warm-starting from {prev_complete}")
                _load_model_from_ckpt(model, prev_complete, out_dir, lora_active=False)
    else:
        print(f"resuming from {resume_path}")
        _load_model_from_ckpt(model, resume_path, out_dir,
                              lora_active=(stage == "2_3"))

    # --- optimiser / scheduler -----------------------------------------
    optimizer, lora_params = build_optimizer_for_stage(stage, model, cfg)
    lora_active = stage == "2_3"
    n_epochs = int(cfg[epochs_key])
    K = int(cfg["folds"])
    episodes_per_fold = int(cfg[episodes_key])
    steps_per_fold_per_epoch = max(
        1, episodes_per_fold // max(1, int(cfg["batch_size"]) * int(cfg["grad_accum_steps"]))
    )
    total_steps = steps_per_fold_per_epoch * K * n_epochs
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_frac=float(cfg["warmup_frac"])
    )
    scaler = _make_scaler(bool(cfg["use_amp"]), device)

    # If we resumed within the stage, restore opt + sched.
    if resume_path is not None:
        ckpt_full = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        if ckpt_full.get("stage") == stage and not ckpt_full.get("stage_completed"):
            try:
                optimizer.load_state_dict(ckpt_full["optimizer"])
                scheduler.load_state_dict(ckpt_full["scheduler"])
                if scaler is not None and ckpt_full.get("scaler"):
                    scaler.load_state_dict(ckpt_full["scaler"])
                restore_rng(ckpt_full.get("rng"))
                print(f"  restored optimizer + scheduler at "
                      f"epoch={ckpt_full['epoch']} fold={ckpt_full['fold']}")
            except Exception as e:
                print(f"  warning: failed to restore optimizer/scheduler ({e}); fresh")

    # --- fold plan -----------------------------------------------------
    with open(cfg["manifest"]) as f:
        manifest_obj = json.load(f)
    train_instances = [
        i for i in manifest_obj["instances"] if i.get("split") == "train"
    ]
    fold_plan_path = Path(cfg["analysis_root"]) / "folds.json"
    if fold_plan_path.exists():
        with open(fold_plan_path) as f:
            saved = json.load(f)
        if saved.get("k") == K and saved.get("seed") == int(cfg["fold_seed"]):
            fold_plan = saved["folds"]
        else:
            fold_plan = stratified_kfold(train_instances, k=K, seed=int(cfg["fold_seed"]))
            write_json(fold_plan_path,
                       {"k": K, "seed": int(cfg["fold_seed"]), "folds": fold_plan})
    else:
        fold_plan = stratified_kfold(train_instances, k=K, seed=int(cfg["fold_seed"]))
        write_json(fold_plan_path,
                   {"k": K, "seed": int(cfg["fold_seed"]), "folds": fold_plan})

    write_json(analysis_dir / "config.json", cfg)

    best_metric: dict[str, Any] = {"value": -1.0, "epoch": 0, "fold": 0}
    early_stop_counter = 0
    metrics_history: list[dict] = []

    box_freeze_epochs = int(cfg.get("stage_1_2_box_freeze_epochs", 3)) if stage == "1_2" else 0

    for epoch in range(1, n_epochs + 1):
        # Stage 1.2 box-head warmup: freeze the box_head for the first N epochs.
        if stage == "1_2":
            for p in model.owlv2.box_head.parameters():
                p.requires_grad = epoch > box_freeze_epochs

        fold_jsons: list[dict] = []
        for fold_idx in range(K):
            t0 = time.time()
            fold = fold_plan[fold_idx]
            train_ids = set(fold["train_ids"])
            val_ids = set(fold["val_ids"])

            train_ds, train_loader = build_train_loader(
                manifest=cfg["manifest"], data_root=cfg["data_root"],
                split="train", sources=None,
                episodes_per_epoch=episodes_per_fold,
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                neg_prob=float(cfg["neg_prob"]),
                img_size=int(cfg["img_size"]),
                seed=int(cfg["seed"]) + 1000 * epoch + fold_idx,
                n_support=int(cfg["n_support"]),
            )
            train_ds.set_fold(train_ids=train_ids)
            val_ds, val_loader = build_val_loader(
                manifest=cfg["manifest"], data_root=cfg["data_root"],
                split="train", sources=None,
                val_episodes=int(cfg["val_episodes"]),
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                neg_prob=float(cfg["neg_prob"]),
                img_size=int(cfg["img_size"]),
                seed=int(cfg["seed"]) + 7000 * epoch + fold_idx,
                n_support=int(cfg["n_support"]),
            )
            val_ds.set_fold(val_ids=val_ids)

            train_metrics = train_one_pass(
                model=model, optimizer=optimizer, scheduler=scheduler,
                loader=train_loader, device=device, cfg=cfg,
                use_box_loss=use_box_loss, scaler=scaler,
                use_amp=bool(cfg["use_amp"]),
            )
            val_metrics = evaluate(
                model, val_loader, device,
                img_size=int(cfg["img_size"]),
                tile_cfg=cfg.get("in_loop_val_tile_cfg"),
            )

            payload = {
                "stage": stage, "epoch": epoch, "fold": fold_idx,
                "wall_clock_seconds": round(time.time() - t0, 2),
                "lr": {g.get("name", str(i)): g["lr"]
                       for i, g in enumerate(optimizer.param_groups)},
                "train": train_metrics,
                "val": val_metrics,
            }
            write_json(
                analysis_dir / f"epoch_{epoch:03d}" / f"fold_{fold_idx}.json",
                payload,
            )
            fold_jsons.append(payload)
            metrics_history.append(payload)

            ckpt = _save_ckpt(
                out_dir=out_dir, stage=stage, epoch=epoch, fold=fold_idx,
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, cfg=cfg, fold_plan=fold_plan,
                best_metric=best_metric, early_stop_counter=early_stop_counter,
                metrics_history=metrics_history, stage_completed=False,
                lora_active=lora_active,
            )
            atomic_save(ckpt, out_dir / f"ckpt_fold{fold_idx}_epoch{epoch:03d}.pt")
            atomic_save(ckpt, out_dir / "last.pt")
            hygiene(out_dir, int(cfg["keep_last_n"]))

            print_epoch_log(
                header=f"stage {stage} epoch {epoch}/{n_epochs} fold {fold_idx}/{K - 1}",
                train_metrics=train_metrics, val_metrics=val_metrics,
                lr_groups={g.get("name", str(i)): g["lr"]
                           for i, g in enumerate(optimizer.param_groups)},
                wall_clock=payload["wall_clock_seconds"],
            )
            del train_loader, val_loader, train_ds, val_ds

        # End of epoch — aggregate folds.
        aggregate = aggregate_folds(fold_jsons)
        write_json(analysis_dir / f"epoch_{epoch:03d}" / "aggregate.json", aggregate)
        print_aggregate(stage, epoch, aggregate)
        map50 = aggregate["metrics"].get("val.overall.map_50", {}).get("mean", 0.0)
        if map50 > best_metric["value"]:
            best_metric = {"value": map50, "epoch": epoch, "fold": K - 1}
            best_ckpt = _save_ckpt(
                out_dir=out_dir, stage=stage, epoch=epoch, fold=K - 1,
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, cfg=cfg, fold_plan=fold_plan,
                best_metric=best_metric, early_stop_counter=0,
                metrics_history=metrics_history, stage_completed=False,
                lora_active=lora_active,
            )
            atomic_save(best_ckpt, out_dir / "best.pt")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        update_summary(Path(cfg["analysis_root"]), {
            f"{stage}.val.map_50_mean": (epoch, map50),
            f"{stage}.val.iou_mean_mean": (
                epoch,
                aggregate["metrics"].get("val.overall.iou_mean", {}).get("mean", 0.0),
            ),
        })

        # Early stopping.
        patience_key = {"1_1": "early_stop_patience_s1",
                        "1_2": "early_stop_patience_s2",
                        "2_3": "early_stop_patience_s3"}[stage]
        if early_stop_counter >= int(cfg[patience_key]):
            print(f"  early stop: {patience_key}={cfg[patience_key]} epochs without improvement")
            break

    # Stage-completion checkpoint.
    final_ckpt = _save_ckpt(
        out_dir=out_dir, stage=stage, epoch=epoch, fold=K - 1,
        model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, cfg=cfg, fold_plan=fold_plan,
        best_metric=best_metric, early_stop_counter=early_stop_counter,
        metrics_history=metrics_history, stage_completed=True,
        lora_active=lora_active,
    )
    atomic_save(final_ckpt, out_dir / "stage_complete.pt")
    write_json(analysis_dir / "complete.json", {
        "stage_completed": True,
        "epochs_run": epoch,
        "best_val_map_50_mean": best_metric["value"],
        "best_epoch": best_metric["epoch"],
    })
    print(f"Stage {stage} complete. Best val map_50 mean = {best_metric['value']:.4f} "
          f"at epoch {best_metric['epoch']}")
    return {"best_metric": best_metric, "config": cfg}


def train_stage_1_1(**user_kwargs) -> dict:
    return _run_stage(
        "1_1", user_kwargs=user_kwargs,
        use_box_loss=False,
        epochs_key="stage_1_1_epochs",
        episodes_key="episodes_per_fold_s1",
    )


def train_stage_1_2(**user_kwargs) -> dict:
    return _run_stage(
        "1_2", user_kwargs=user_kwargs,
        use_box_loss=True,
        epochs_key="stage_1_2_epochs",
        episodes_key="episodes_per_fold_s2",
    )


def train_stage_2_3(**user_kwargs) -> dict:
    return _run_stage(
        "2_3", user_kwargs=user_kwargs,
        use_box_loss=True,
        epochs_key="stage_2_3_epochs",
        episodes_key="episodes_per_fold_s3",
    )


# ---------------------------------------------------------------------------
# Eval-only entrypoint
# ---------------------------------------------------------------------------


def _persist_eval_results(
    *,
    metrics: dict,
    cfg: dict,
    stage: str,
    tile_cfg: dict,
    extra_provenance: dict,
) -> dict[str, str]:
    """Write the eval metrics + provenance to *all* the right places.

    Locations:
      - ``checkpoints/<stage>/test_eval_<tile_mode>.json``
            Latest result for this (stage, tile_mode) pair.  Overwritten on
            each call so the "current best view" is always at a stable path.
      - ``analysis/<stage>/test_eval_<tile_mode>_<timestamp>.json``
            Timestamped historical record — never overwritten.
      - ``analysis/eval_log.jsonl``
            Single rolling log of every eval across every stage / tile mode.
            One JSON object per line, append-only.

    Returns the dict of paths written.
    """
    payload = {
        "stage": stage,
        "tile_cfg": tile_cfg,
        "config": cfg,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra_provenance,
        "metrics": metrics,
    }
    tile_mode = str(tile_cfg.get("mode", "off"))
    ts = time.strftime("%Y%m%d_%H%M%S")

    out_dir = Path(cfg["out_root"]) / _stage_dir_name(stage)
    analysis_dir = Path(cfg["analysis_root"]) / _stage_dir_name(stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    latest_path = out_dir / f"test_eval_{tile_mode}.json"
    timestamped_path = analysis_dir / f"test_eval_{tile_mode}_{ts}.json"

    write_json(latest_path, payload)
    write_json(timestamped_path, payload)

    # Append a flat one-line summary to the rolling log.
    log_path = Path(cfg["analysis_root"]) / "eval_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    summary = {
        "timestamp": payload["timestamp"],
        "stage": stage,
        "tile_mode": tile_mode,
        "checkpoint": extra_provenance.get("checkpoint"),
        "map_50": float(overall.get("map_50", 0.0)),
        "map_75": float(overall.get("map_75", 0.0)),
        "map_5095": float(overall.get("map_5095", 0.0)),
        "iou_mean": float(overall.get("iou_mean", 0.0)),
        "existence_auroc": float(overall.get("existence_auroc", 0.0)),
        "existence_acc": float(overall.get("existence_acc", 0.0)),
        "false_positive_rate": float(overall.get("false_positive_rate", 0.0)),
        "n": int(overall.get("n", 0)),
        "results_path": str(latest_path),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(summary, default=float) + "\n")

    return {
        "latest": str(latest_path),
        "timestamped": str(timestamped_path),
        "log": str(log_path),
    }


def evaluate_run(checkpoint: str, **user_kwargs) -> dict:
    """Load ``checkpoint`` and evaluate on the held-out test split.

    Supports tiled inference via the ``tile_cfg`` kwarg — see
    ``modeling.evaluate.DEFAULT_TILE_CFG``.  Pass
    ``tile_cfg={"mode": "off"}`` to disable tiling entirely.

    Persists results to:
      - ``<checkpoint_dir>/test_eval_<tile_mode>.json``       (latest, overwritten)
      - ``<analysis_root>/<stage>/test_eval_<tile_mode>_<ts>.json``   (history)
      - ``<analysis_root>/eval_log.jsonl``                    (rolling one-liner)
    """
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    tile_cfg = resolve_tile_cfg(cfg.get("tile_cfg"))
    needs_native = tile_cfg["mode"] != "off"
    print(f"=== Evaluating {checkpoint} on test split (tile_mode={tile_cfg['mode']}) ===")

    model = OWLv2FewShotLocalizer().to(device)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", "unknown")
    lora_active = stage == "2_3"
    if lora_active:
        # Re-attach LoRA so the state_dict has matching keys.
        model.attach_lora(
            r=int(cfg["lora_r"]),
            alpha=int(cfg["lora_alpha"]),
            dropout=float(cfg["lora_dropout"]),
            last_n_layers=int(cfg["lora_layers"]),
        )
    ok, err = try_load_model_state(model, ckpt.get("model", {}))
    if not ok:
        raise RuntimeError(f"checkpoint mismatch: {err}")
    model = model.to(device)

    test_ds, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None,
        val_episodes=int(cfg.get("test_episodes", 400)),
        batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
        neg_prob=float(cfg["neg_prob"]),
        img_size=int(cfg["img_size"]),
        seed=int(cfg["seed"]),
        n_support=int(cfg["n_support"]),
        return_native=needs_native,
    )
    t0 = time.time()
    metrics = evaluate(
        model, test_loader, device,
        img_size=int(cfg["img_size"]),
        tile_cfg=tile_cfg,
    )
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)

    paths = _persist_eval_results(
        metrics=metrics,
        cfg=cfg,
        stage=str(stage),
        tile_cfg=tile_cfg,
        extra_provenance={
            "checkpoint": str(ckpt_path),
            "split": "test",
            "test_episodes": int(cfg.get("test_episodes", 400)),
        },
    )
    print(
        f"Test mAP@50 = {metrics['overall'].get('map_50', 0.0):.4f}  "
        f"existence_auroc = {metrics['overall'].get('existence_auroc', 0.0):.4f}  "
        f"({metrics['wall_clock_seconds']:.1f}s)"
    )
    print(f"  → results saved to: {paths['latest']}")
    print(f"  → history snapshot: {paths['timestamped']}")
    print(f"  → rolling log     : {paths['log']}")
    return metrics


def evaluate_phase0(**user_kwargs) -> dict:
    """Evaluate the zero-shot OWLv2 baseline on the held-out test split only.

    Supports tiled inference via the ``tile_cfg`` kwarg — see
    ``modeling.evaluate.DEFAULT_TILE_CFG``.

    Persists results to:
      - ``<out_root>/phase0/test_eval_<tile_mode>.json``       (latest, overwritten)
      - ``<analysis_root>/phase0/test_eval_<tile_mode>_<ts>.json``  (history)
      - ``<analysis_root>/eval_log.jsonl``                     (rolling one-liner)
    """
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    tile_cfg = resolve_tile_cfg(cfg.get("tile_cfg"))
    needs_native = tile_cfg["mode"] != "off"
    print(f"=== Phase 0 evaluation on test split (tile_mode={tile_cfg['mode']}) ===")
    model = OWLv2FewShotLocalizer().to(device)
    owlv2 = model.owlv2
    test_ds, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None,
        val_episodes=int(cfg.get("val_episodes_phase0", 200)),
        batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
        neg_prob=float(cfg["neg_prob"]),
        img_size=int(cfg["img_size"]),
        seed=int(cfg["seed"]),
        n_support=int(cfg["n_support"]),
        return_native=needs_native,
    )
    t0 = time.time()
    metrics = _evaluate_phase0(
        owlv2, test_loader, device,
        img_size=int(cfg["img_size"]), tile_cfg=tile_cfg,
    )
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)

    paths = _persist_eval_results(
        metrics=metrics,
        cfg=cfg,
        stage="phase0",
        tile_cfg=tile_cfg,
        extra_provenance={
            "checkpoint": None,                                    # zero-shot
            "split": "test",
            "test_episodes": int(cfg.get("val_episodes_phase0", 200)),
        },
    )
    print(
        f"Phase 0 mAP@50 = {metrics['overall'].get('map_50', 0.0):.4f}  "
        f"existence_auroc = {metrics['overall'].get('existence_auroc', 0.0):.4f}  "
        f"({metrics['wall_clock_seconds']:.1f}s)"
    )
    print(f"  → results saved to: {paths['latest']}")
    print(f"  → history snapshot: {paths['timestamped']}")
    print(f"  → rolling log     : {paths['log']}")
    return metrics
