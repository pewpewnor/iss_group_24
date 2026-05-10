"""Siamese training orchestrator.

Public entry points:
    train_phase0(...)
    train_stage_S1(...)
    train_stage_S2(...)
    evaluate_run(checkpoint=..., ...)
"""

from __future__ import annotations

import gc
import json
import random as _random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shared.analytics import aggregate_folds, update_summary, write_json
from shared.checkpoint import (
    atomic_save, atomic_save_multi, capture_rng, get_trainable_state, hygiene,
    load_trainable_state, resolve_resume_path, restore_rng,
)
from shared.folds import stratified_kfold
from shared.logging import print_aggregate, print_epoch_log
from shared.runtime import gpu_cleanup_on_exit, release_gpu_memory
from siamese.dataset import build_train_loader, build_val_loader
from siamese.evaluate import evaluate
from siamese.model import MultiShotSiamese
from siamese.optim import build_optimizer_for_stage, build_scheduler
from siamese.train_loop import train_one_pass


MODEL_KIND = "siamese"


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict[str, Any] = {
    "manifest": "dataset/aggregated/manifest.json",
    "data_root": None,
    "out_root": "checkpoints",
    "analysis_root": "analysis",
    # Hardware
    "img_size": 518,                   # DINOv2 native (37*14)
    "batch_size": 4,
    "grad_accum_steps": 2,
    "num_workers": 2,
    "use_amp": True,
    "device": None,
    # Folds
    "folds": 3,
    "fold_seed": 42,
    # K range
    "k_min": 1,
    "k_max": 10,
    # Stage durations
    "S1_epochs": 10,
    "S2_epochs": 8,
    "S1_eps_per_fold": 400,
    "S2_eps_per_fold": 400,
    "val_episodes": 200,
    "test_episodes": 400,
    # LRs
    "lr_head_S1": 1e-3,
    "lr_head_S2": 1e-4,
    "lr_lora_S2": 1e-4,
    # Optim
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_frac": 0.05,
    # Loss
    "focal_alpha": 0.25,               # lower alpha penalises FP more (negatives weighted higher)
    "focal_gamma": 2.0,
    "variance_target": 0.5,
    "variance_weight": 0.1,
    "decorr_weight": 0.05,
    # Negatives / hard-neg cache
    "neg_prob": 0.75,                  # ratio 1:3 (25% positive)
    "hard_neg_cache_frac": 0.5,
    # Architecture
    "dinov2_model_name": "facebook/dinov2-small",
    "cross_attn_heads": 6,
    "cross_attn_dropout": 0.1,
    "head_hidden_1": 256,
    "head_hidden_2": 64,
    "head_dropout": 0.2,
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_last_n_layers": 4,
    "lora_target_modules": ("query", "value"),
    # Augmentation
    "aug_color_jitter": 0.4,
    "aug_hue": 0.1,
    "aug_grayscale_prob": 0.2,
    "aug_blur_prob": 0.2,
    "aug_blur_sigma": (0.5, 2.0),
    "aug_erase_prob": 0.3,
    "aug_erase_scale": (0.05, 0.20),
    "aug_rrc_scale": (0.5, 1.0),
    "aug_hflip_prob": 0.5,
    "aug_query_color_jitter": 0.2,
    # Early stopping
    "S1_early_stop_patience": 4,
    "S2_early_stop_patience": 4,
    "early_stop_metric": "auroc",      # "auroc" | "fpr_inv" (=1-fpr)
    # Eval threshold
    "eval_threshold": 0.5,
    # Misc
    "seed": 42,
    "keep_last_n": 0,
    "smoke": False,
}


SMOKE_OVERRIDES: dict[str, Any] = {
    "img_size": 224,
    "batch_size": 1,
    "grad_accum_steps": 1,
    "num_workers": 0,
    "folds": 1,
    "k_min": 1,
    "k_max": 2,
    "S1_epochs": 1,
    "S2_epochs": 1,
    "S1_eps_per_fold": 4,
    "S2_eps_per_fold": 4,
    "val_episodes": 4,
    "test_episodes": 4,
    "cross_attn_heads": 4,
    "head_hidden_1": 64,
    "head_hidden_2": 32,
    "lora_last_n_layers": 2,
    "lora_r": 4,
    "lora_alpha": 8,
    "use_amp": False,
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


def _merge_cfg(user_kwargs: dict) -> dict:
    cfg = dict(DEFAULT_CFG)
    cfg.update(user_kwargs or {})
    if cfg.get("smoke"):
        cfg.update(SMOKE_OVERRIDES)
        cfg.update(user_kwargs or {})
    return cfg


def _stage_lrs(stage: str, cfg: dict) -> dict:
    out = dict(cfg)
    for k in ("head", "lora"):
        full = f"lr_{k}_{stage}"
        if full in cfg:
            out[f"lr_{k}"] = cfg[full]
        else:
            out.setdefault(f"lr_{k}", 0.0)
    return out


def _stage_dirs(cfg: dict, stage: str) -> tuple[Path, Path]:
    out_dir = Path(cfg["out_root"]) / MODEL_KIND / stage
    analysis_dir = Path(cfg["analysis_root"]) / MODEL_KIND / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, analysis_dir


def _make_scaler(use_amp: bool, device: torch.device):
    if use_amp and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def _build_model(cfg: dict, *, lora_active: bool = False) -> MultiShotSiamese:
    m = MultiShotSiamese(
        model_name=cfg["dinov2_model_name"],
        k_max=int(cfg["k_max"]),
        cross_attn_heads=int(cfg["cross_attn_heads"]),
        cross_attn_dropout=float(cfg["cross_attn_dropout"]),
        head_hidden_1=int(cfg["head_hidden_1"]),
        head_hidden_2=int(cfg["head_hidden_2"]),
        head_dropout=float(cfg["head_dropout"]),
    )
    if lora_active:
        m.attach_lora(
            r=int(cfg["lora_r"]),
            alpha=int(cfg["lora_alpha"]),
            dropout=float(cfg["lora_dropout"]),
            last_n_layers=int(cfg["lora_last_n_layers"]),
        )
    return m


def _augmentation_kwargs(cfg: dict) -> dict[str, Any]:
    return dict(
        aug_color_jitter=cfg["aug_color_jitter"],
        aug_hue=cfg["aug_hue"],
        aug_grayscale_prob=cfg["aug_grayscale_prob"],
        aug_blur_prob=cfg["aug_blur_prob"],
        aug_blur_sigma=tuple(cfg["aug_blur_sigma"]),
        aug_erase_prob=cfg["aug_erase_prob"],
        aug_erase_scale=tuple(cfg["aug_erase_scale"]),
        aug_rrc_scale=tuple(cfg["aug_rrc_scale"]),
        aug_hflip_prob=cfg["aug_hflip_prob"],
        aug_query_color_jitter=cfg["aug_query_color_jitter"],
    )


def _save_stage_ckpt(
    *, out_dir: Path, stage: str, epoch: int, fold: int,
    stage_completed: bool, model: MultiShotSiamese, optimizer, scheduler, scaler,
    cfg: dict, fold_plan: list, best_metric: dict,
    early_stop_counter: int, metrics_history: list,
    rolling: bool, extra_path: Path | None = None,
) -> Path:
    state = get_trainable_state(model)
    payload = {
        "model_kind": MODEL_KIND,
        "stage": stage,
        "epoch": epoch,
        "fold": fold,
        "stage_completed": stage_completed,
        "global_step": int(scheduler.last_epoch) if scheduler is not None else 0,
        "state_dict": state,
        "lora_active": bool(model.lora_attached),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng(),
        "config": cfg,
        "fold_plan": fold_plan,
        "metrics_history": metrics_history,
        "best_metric": best_metric,
        "early_stop_counter": early_stop_counter,
    }
    if rolling:
        rolling_path = out_dir / f"ckpt_fold{fold}_epoch{epoch:03d}.pt"
        targets: list[tuple[Path, str]] = [
            (rolling_path, f"rolling fold{fold} epoch{epoch:03d}"),
            (out_dir / "last.pt", "last"),
        ]
        if extra_path is not None:
            targets.append((extra_path, extra_path.stem))
        # Single torch.save → multi-mirror to all targets. On Drive this
        # avoids the back-to-back os.replace pattern that has been observed
        # to drop the earlier rolling file from Drive folders.
        atomic_save_multi(payload, targets)
        return rolling_path
    if extra_path is not None:
        atomic_save(payload, extra_path, label=extra_path.stem)
    return out_dir / "last.pt"


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


_TRAIN_PRIORITY = ("loss", "focal", "variance", "decorrelation", "grad_norm", "n_steps")
_VAL_PRIORITY = ("n", "n_pos", "n_neg",
                 "auroc", "pr_auc", "avg_precision",
                 "accuracy", "f1", "best_f1", "best_f1_threshold",
                 "precision", "recall",
                 "fpr", "fnr", "tpr", "tnr",
                 "youden_j", "mcc",
                 "acc_pos", "acc_neg",
                 "fpr_at_recall_95", "recall_at_fpr_05", "recall_at_fpr_10",
                 "mean_score_pos", "mean_score_neg", "score_gap",
                 "std_score_pos", "std_score_neg",
                 "frac_high_score", "frac_low_score", "frac_uncertain",
                 "brier",
                 "tp", "fp", "fn", "tn")
_PER_SOURCE_KEYS = ("n", "n_pos", "auroc", "pr_auc", "f1", "best_f1",
                    "fpr", "fnr", "accuracy", "mcc",
                    "mean_score_pos", "mean_score_neg", "score_gap")


def train_phase0(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _train_phase0_inner(user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def _train_phase0_inner(user_kwargs: dict) -> dict:
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, "phase0")
    print(f"=== [siamese] Phase 0 (zero-shot DINOv2 cosine) on {device} ===")
    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg).to(device)
    test_eps = int(cfg.get("test_episodes", 400))
    _, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
        neg_prob=float(cfg["neg_prob"]),
        img_size=int(cfg["img_size"]), seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
    )
    t0 = time.time()
    metrics = evaluate(
        model, test_loader, device, threshold=float(cfg["eval_threshold"]),
        phase0=True,
    )
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)
    write_json(out_dir / "results.json", metrics)
    write_json(analysis_dir / "results.json", metrics)
    o = metrics["overall"]
    print(
        f"[siamese phase0]  "
        f"AUROC={o.get('auroc', 0.0):.4f}  "
        f"PR-AUC={o.get('pr_auc', 0.0):.4f}  "
        f"AP={o.get('avg_precision', 0.0):.4f}  "
        f"acc={o.get('accuracy', 0.0):.4f}  "
        f"best_f1={o.get('best_f1', 0.0):.4f}@thr={o.get('best_f1_threshold', 0.0):.2f}  "
        f"FPR={o.get('fpr', 0.0):.4f}  FNR={o.get('fnr', 0.0):.4f}  "
        f"MCC={o.get('mcc', 0.0):.4f}"
    )
    return metrics


def evaluate_phase0(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _train_phase0_inner(user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


# ---------------------------------------------------------------------------
# Stage runner (S1 / S2)
# ---------------------------------------------------------------------------


def _run_stage(stage: str, *, user_kwargs: dict) -> dict:
    cfg = _merge_cfg(user_kwargs)
    cfg = _stage_lrs(stage, cfg)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, stage)

    print(f"\n=== [siamese] Stage {stage} on {device} ===")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _set_seed(int(cfg["seed"]))
    lora_active = (stage == "S2")
    model = _build_model(cfg, lora_active=lora_active).to(device)

    # --- resume / warm-start ----
    resume = user_kwargs.get("resume", True)
    resume_path = resolve_resume_path(resume, out_dir)
    if resume_path is None:
        prev_map = {"S1": None, "S2": "S1"}
        prev = prev_map.get(stage)
        if prev:
            prev_complete = Path(cfg["out_root"]) / MODEL_KIND / prev / "stage_complete.pt"
            if prev_complete.exists():
                print(f"  warm-starting from {prev_complete}")
                ckpt = torch.load(str(prev_complete), map_location="cpu", weights_only=False)
                load_trainable_state(model, ckpt.get("state_dict", {}))
    else:
        print(f"  resuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        load_trainable_state(model, ckpt.get("state_dict", {}))

    optimizer, lora_params = build_optimizer_for_stage(stage, model, cfg)
    n_epochs = int(cfg[f"{stage}_epochs"])
    K = int(cfg["folds"])
    eps_per_fold = int(cfg[f"{stage}_eps_per_fold"])
    steps_per_fold_per_epoch = max(
        1,
        eps_per_fold // max(1, int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])),
    )
    total_steps = steps_per_fold_per_epoch * K * n_epochs
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_frac=float(cfg["warmup_frac"]),
    )
    scaler = _make_scaler(bool(cfg["use_amp"]), device)

    resume_epoch = 1
    resume_fold = 0
    best_metric: dict[str, Any] = {"value": -1.0, "epoch": 0, "fold": 0}
    early_stop_counter = 0
    metrics_history: list[dict] = []

    if resume_path is not None:
        ckpt_full = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        if ckpt_full.get("stage") == stage and not ckpt_full.get("stage_completed"):
            try:
                if ckpt_full.get("optimizer"):
                    optimizer.load_state_dict(ckpt_full["optimizer"])
                if ckpt_full.get("scheduler"):
                    scheduler.load_state_dict(ckpt_full["scheduler"])
                if scaler is not None and ckpt_full.get("scaler"):
                    scaler.load_state_dict(ckpt_full["scaler"])
                restore_rng(ckpt_full.get("rng"))
                saved_epoch = int(ckpt_full["epoch"])
                saved_fold = int(ckpt_full["fold"])
                if saved_fold >= K - 1:
                    resume_epoch = saved_epoch + 1
                    resume_fold = 0
                else:
                    resume_epoch = saved_epoch
                    resume_fold = saved_fold + 1
                best_metric = dict(ckpt_full.get("best_metric") or best_metric)
                early_stop_counter = int(ckpt_full.get("early_stop_counter", 0))
                metrics_history = list(ckpt_full.get("metrics_history") or [])
                print(f"  restored optim+sched at epoch={saved_epoch} fold={saved_fold}; "
                      f"continuing from epoch={resume_epoch} fold={resume_fold}")
            except Exception as e:                                                 # noqa: BLE001
                print(f"  warning: failed to restore optimizer/scheduler ({e}); fresh within stage")

    # Fold plan (shared between localizer and siamese is fine, but separate file).
    with open(cfg["manifest"]) as f:
        manifest_obj = json.load(f)
    train_instances = [i for i in manifest_obj["instances"] if i.get("split") == "train"]
    if not train_instances:
        raise RuntimeError(
            f"manifest {cfg['manifest']} has no instances with split='train'. "
            "Re-run aggregator.py."
        )
    fold_plan_path = Path(cfg["analysis_root"]) / MODEL_KIND / "folds.json"
    if fold_plan_path.exists():
        with open(fold_plan_path) as f:
            saved = json.load(f)
        if saved.get("k") == K and saved.get("seed") == int(cfg["fold_seed"]):
            fold_plan = saved["folds"]
        else:
            fold_plan = stratified_kfold(train_instances, k=K, seed=int(cfg["fold_seed"]))
            write_json(fold_plan_path, {"k": K, "seed": int(cfg["fold_seed"]), "folds": fold_plan})
    else:
        fold_plan = stratified_kfold(train_instances, k=K, seed=int(cfg["fold_seed"]))
        write_json(fold_plan_path, {"k": K, "seed": int(cfg["fold_seed"]), "folds": fold_plan})

    write_json(analysis_dir / "config.json", cfg)

    # Hard-negative cache shared across folds of one stage.
    hard_neg_cache: dict[str, list[dict]] = {}
    hard_neg_frac = float(cfg["hard_neg_cache_frac"])

    epoch = resume_epoch
    for epoch in range(resume_epoch, n_epochs + 1):
        fold_jsons: list[dict] = [
            m for m in metrics_history
            if m.get("epoch") == epoch and m.get("fold", -1) < resume_fold
        ] if epoch == resume_epoch else []

        for fold_idx in range(K):
            if epoch == resume_epoch and fold_idx < resume_fold:
                continue
            t0 = time.time()
            fold = fold_plan[fold_idx]
            print(f"▶ [siamese] {stage} epoch {epoch}/{n_epochs} fold {fold_idx}/{K - 1}",
                  flush=True)

            train_ds, train_loader = build_train_loader(
                manifest=cfg["manifest"], data_root=cfg["data_root"],
                split="train", sources=None,
                episodes_per_epoch=eps_per_fold,
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                neg_prob=float(cfg["neg_prob"]),
                img_size=int(cfg["img_size"]),
                seed=int(cfg["seed"]) + 1000 * epoch + fold_idx,
                k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
                aug_kwargs=_augmentation_kwargs(cfg),
                hard_neg_cache=hard_neg_cache, hard_neg_frac=hard_neg_frac,
            )
            train_ds.set_fold(train_ids=set(fold["train_ids"]))
            val_ds, val_loader = build_val_loader(
                manifest=cfg["manifest"], data_root=cfg["data_root"],
                split="train", sources=None,
                val_episodes=int(cfg["val_episodes"]),
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                neg_prob=float(cfg["neg_prob"]),
                img_size=int(cfg["img_size"]),
                seed=int(cfg["seed"]) + 7000 * epoch + fold_idx,
                k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
            )
            val_ds.set_fold(val_ids=set(fold["val_ids"]))

            recorder: dict[str, list[dict]] = {}
            train_metrics = train_one_pass(
                model=model, optimizer=optimizer, scheduler=scheduler,
                loader=train_loader, device=device, cfg=cfg,
                scaler=scaler, use_amp=bool(cfg["use_amp"]),
                hard_neg_recorder=recorder,
            )
            val_metrics = evaluate(
                model, val_loader, device,
                threshold=float(cfg["eval_threshold"]),
                progress_every=20,
            )

            # Update shared hard-neg cache.
            for iid, items in recorder.items():
                hard_neg_cache.setdefault(iid, []).extend(items)
                hard_neg_cache[iid] = hard_neg_cache[iid][-256:]

            payload = {
                "stage": stage, "epoch": epoch, "fold": fold_idx,
                "wall_clock_seconds": round(time.time() - t0, 2),
                "lr": {g.get("name", str(i)): g["lr"]
                       for i, g in enumerate(optimizer.param_groups)},
                "train": train_metrics,
                "val": val_metrics,
            }
            write_json(analysis_dir / f"epoch_{epoch:03d}" / f"fold_{fold_idx}.json", payload)
            fold_jsons.append(payload)
            metrics_history.append(payload)

            _save_stage_ckpt(
                out_dir=out_dir, stage=stage, epoch=epoch, fold=fold_idx,
                stage_completed=False, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, cfg=cfg,
                fold_plan=fold_plan, best_metric=best_metric,
                early_stop_counter=early_stop_counter,
                metrics_history=metrics_history, rolling=True,
            )
            hygiene(out_dir, int(cfg["keep_last_n"]))

            print_epoch_log(
                header=f"siamese {stage} epoch {epoch}/{n_epochs} fold {fold_idx}/{K - 1}",
                train_metrics=train_metrics, val_metrics=val_metrics,
                lr_groups={g.get("name", str(i)): g["lr"]
                           for i, g in enumerate(optimizer.param_groups)},
                wall_clock=payload["wall_clock_seconds"],
                train_priority=_TRAIN_PRIORITY, val_priority=_VAL_PRIORITY,
                per_source_keys=_PER_SOURCE_KEYS,
            )
            del train_loader, val_loader, train_ds, val_ds

        aggregate = aggregate_folds(fold_jsons)
        write_json(analysis_dir / f"epoch_{epoch:03d}" / "aggregate.json", aggregate)
        print_aggregate(stage, epoch, aggregate, keys=(
            ("val.overall.auroc",             "auroc"),
            ("val.overall.pr_auc",            "pr_auc"),
            ("val.overall.avg_precision",     "avg_precision"),
            ("val.overall.f1",                "f1"),
            ("val.overall.best_f1",           "best_f1"),
            ("val.overall.best_f1_threshold", "best_f1_thr"),
            ("val.overall.fpr",               "fpr"),
            ("val.overall.fnr",               "fnr"),
            ("val.overall.fpr_at_recall_95",  "fpr@r95"),
            ("val.overall.recall_at_fpr_05",  "r@fpr05"),
            ("val.overall.recall_at_fpr_10",  "r@fpr10"),
            ("val.overall.mcc",               "mcc"),
            ("val.overall.youden_j",          "youden_j"),
            ("val.overall.accuracy",          "accuracy"),
            ("val.overall.score_gap",         "score_gap"),
            ("val.overall.mean_score_pos",    "score_pos"),
            ("val.overall.mean_score_neg",    "score_neg"),
            ("train.loss",                    "train_loss"),
            ("train.focal",                   "train_focal"),
            ("train.variance",                "train_var"),
        ))
        # Pick the headline metric: AUROC by default, fpr_inv (=1-fpr) optional.
        headline_key = cfg.get("early_stop_metric", "auroc")
        if headline_key == "fpr_inv":
            fpr = aggregate["metrics"].get("val.overall.fpr", {}).get("mean", 1.0)
            metric_value = 1.0 - fpr
        else:
            metric_value = aggregate["metrics"].get("val.overall.auroc", {}).get("mean", 0.0)
        if metric_value > best_metric["value"]:
            best_metric = {"value": metric_value, "epoch": epoch, "fold": K - 1}
            _save_stage_ckpt(
                out_dir=out_dir, stage=stage, epoch=epoch, fold=K - 1,
                stage_completed=False, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, cfg=cfg,
                fold_plan=fold_plan, best_metric=best_metric,
                early_stop_counter=0, metrics_history=metrics_history,
                rolling=False, extra_path=out_dir / "best.pt",
            )
            early_stop_counter = 0
            print(f"    ↳ best metric so far: {headline_key}_mean={metric_value:.4f} at epoch {epoch}")
        else:
            early_stop_counter += 1
        update_summary(Path(cfg["analysis_root"]) / MODEL_KIND, {
            f"{stage}.val.auroc_mean": (epoch,
                aggregate["metrics"].get("val.overall.auroc", {}).get("mean", 0.0)),
            f"{stage}.val.fpr_mean":  (epoch, -aggregate["metrics"].get("val.overall.fpr", {}).get("mean", 1.0)),
        })
        resume_fold = 0

        patience = int(cfg.get(f"{stage}_early_stop_patience", 4))
        if early_stop_counter >= patience:
            print(f"  early stop: {patience} epochs without {headline_key} improvement")
            break

    _save_stage_ckpt(
        out_dir=out_dir, stage=stage, epoch=epoch, fold=K - 1,
        stage_completed=True, model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, cfg=cfg,
        fold_plan=fold_plan, best_metric=best_metric,
        early_stop_counter=early_stop_counter,
        metrics_history=metrics_history, rolling=False,
        extra_path=out_dir / "stage_complete.pt",
    )
    write_json(analysis_dir / "complete.json", {
        "stage_completed": True,
        "epochs_run": epoch,
        "best_value": best_metric["value"],
        "best_epoch": best_metric["epoch"],
    })
    print(f"[siamese] {stage} complete. Best val {cfg.get('early_stop_metric', 'auroc')} = "
          f"{best_metric['value']:.4f} at epoch {best_metric['epoch']}")
    return {"best_metric": best_metric, "config": cfg}


def train_stage_S1(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _run_stage("S1", user_kwargs=user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def train_stage_S2(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _run_stage("S2", user_kwargs=user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def evaluate_run(checkpoint: str, **user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _evaluate_run_inner(checkpoint, user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def _evaluate_run_inner(checkpoint: str, user_kwargs: dict) -> dict:
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    print(f"=== [siamese] Evaluating {checkpoint} on test split ({device}) ===")
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", "unknown")
    lora_active = bool(ckpt.get("lora_active", stage == "S2"))

    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg, lora_active=lora_active)
    load_trainable_state(model, ckpt.get("state_dict", {}))
    model = model.to(device)

    test_eps = int(cfg.get("test_episodes", 400))
    _, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
        neg_prob=float(cfg["neg_prob"]),
        img_size=int(cfg["img_size"]), seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
    )
    t0 = time.time()
    metrics = evaluate(
        model, test_loader, device,
        threshold=float(cfg["eval_threshold"]),
    )
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)

    out_dir = Path(cfg["out_root"]) / MODEL_KIND / stage
    analysis_dir = Path(cfg["analysis_root"]) / MODEL_KIND / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "stage": stage, "checkpoint": str(ckpt_path),
        "split": "test", "test_episodes": test_eps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg, "metrics": metrics,
    }
    write_json(out_dir / "test_eval.json", payload)
    write_json(analysis_dir / f"test_eval_{ts}.json", payload)
    o = metrics["overall"]
    print(
        f"[siamese {stage}] test  "
        f"AUROC={o.get('auroc', 0.0):.4f}  "
        f"PR-AUC={o.get('pr_auc', 0.0):.4f}  "
        f"AP={o.get('avg_precision', 0.0):.4f}  "
        f"acc={o.get('accuracy', 0.0):.4f}  "
        f"f1={o.get('f1', 0.0):.4f}  "
        f"best_f1={o.get('best_f1', 0.0):.4f}@thr={o.get('best_f1_threshold', 0.0):.2f}  "
        f"FPR={o.get('fpr', 0.0):.4f}  FNR={o.get('fnr', 0.0):.4f}  "
        f"MCC={o.get('mcc', 0.0):.4f}  "
        f"({metrics['wall_clock_seconds']:.1f}s)"
    )
    return metrics
