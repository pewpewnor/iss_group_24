"""Localizer training orchestrator.

Public entry points:
    train_phase0(...)
    train_stage_L1(...)
    train_stage_L2(...)
    train_stage_L3(...)
    evaluate_phase0(...)
    evaluate_run(checkpoint=..., ...)

Every kwarg in DEFAULT_CFG is overridable through user_kwargs.
Pass ``smoke=True`` to dial down to a tiny config for the smoke test.
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

from localizer.dataset import (
    build_train_loader, build_val_loader,
)
from localizer.evaluate import evaluate
from localizer.model import MultiShotLocalizer
from localizer.optim import build_optimizer_for_stage, build_scheduler
from localizer.train_loop import train_one_pass
from shared.analytics import aggregate_folds, update_summary, write_json
from shared.checkpoint import (
    atomic_save, atomic_save_multi, capture_rng, get_trainable_state, hygiene,
    load_trainable_state, resolve_resume_path, restore_rng,
)
from shared.folds import stratified_kfold
from shared.logging import print_aggregate, print_epoch_log
from shared.runtime import gpu_cleanup_on_exit, release_gpu_memory


MODEL_KIND = "localizer"


# ---------------------------------------------------------------------------
# Default config — every knob is overridable via user_kwargs.
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict[str, Any] = {
    # I/O
    "manifest": "dataset/aggregated/manifest.json",
    "data_root": None,
    "out_root": "checkpoints",
    "analysis_root": "analysis",
    # Hardware
    "img_size": 768,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "num_workers": 2,
    "use_amp": True,
    "device": None,
    # Folds
    "folds": 3,
    "fold_seed": 42,
    # K range
    "k_min": 1,
    "k_max": 10,
    # Stage durations.
    # L1 is short: only the fusion (~3M params on top of mean-of-q_emb baseline)
    # is being learned, with patch-CE alone. The residual gate alpha starts at 0
    # so we begin AT zero-shot quality and the fusion learns a small correction.
    # 3 epochs × 3 folds × 400 episodes ≈ 3.6k episodes is plenty for ~3M params.
    "L1_epochs": 3,
    "L2_epochs": 12,
    "L3_epochs": 8,
    "L1_eps_per_fold": 400,
    "L2_eps_per_fold": 250,
    "L3_eps_per_fold": 250,
    "val_episodes": 100,
    "test_episodes": 400,
    # LRs.
    # L1: lower LR than before (1e-4 vs 5e-4) — the fusion has very little to
    # do at first (alpha=0) so a high LR pushes it into noisy territory.
    "lr_fusion_L1": 1e-4,
    "lr_fusion_L2": 1e-4,
    "lr_class_L2": 5e-5,
    "lr_box_L2":   5e-5,
    "lr_fusion_L3": 5e-5,
    "lr_class_L3": 2e-5,
    "lr_box_L3":   2e-5,
    "lr_lora_L3":  1e-4,
    # Optim
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_frac": 0.05,
    # Loss
    "lambda_patch_ce": 1.0,
    "lambda_l1": 5.0,
    "lambda_giou": 2.0,
    "patch_ce_label_smoothing": 0.0,
    "L2_box_warmup_epochs": 2,
    # Architecture.
    # fusion_dropout=0.0 at default — with only ~3M trainable fusion params and
    # a tiny dataset, dropout adds noise without preventing overfit.
    "fusion_layers": 2,
    "fusion_heads": 8,
    "fusion_mlp_ratio": 2,
    "fusion_dropout": 0.0,
    "owlv2_model_name": "google/owlv2-base-patch16-ensemble",
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_last_n_layers": 4,
    "lora_target_modules": ("q_proj", "v_proj"),
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
    # Early stopping (L1/L2/L3 share same patience by default; per-stage overrides accepted)
    "L1_early_stop_patience": 4,
    "L2_early_stop_patience": 4,
    "L3_early_stop_patience": 4,
    # Misc
    "seed": 42,
    "keep_last_n": 0,
    # Smoke override
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
    "L1_epochs": 1,
    "L2_epochs": 1,
    "L3_epochs": 1,
    "L1_eps_per_fold": 4,
    "L2_eps_per_fold": 4,
    "L3_eps_per_fold": 4,
    "val_episodes": 4,
    "test_episodes": 4,
    "fusion_layers": 1,
    "fusion_heads": 4,
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
        cfg.update(user_kwargs or {})  # explicit user kwargs trump smoke
    return cfg


def _stage_lrs(stage: str, cfg: dict) -> dict:
    out = dict(cfg)
    suf = stage  # "L1" / "L2" / "L3"
    for k in ("fusion", "class", "box", "lora"):
        full = f"lr_{k}_{suf}"
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


def _build_model(cfg: dict, *, lora_active: bool = False) -> MultiShotLocalizer:
    m = MultiShotLocalizer(
        model_name=cfg["owlv2_model_name"],
        k_max=int(cfg["k_max"]),
        fusion_layers=int(cfg["fusion_layers"]),
        fusion_heads=int(cfg["fusion_heads"]),
        fusion_mlp_ratio=int(cfg["fusion_mlp_ratio"]),
        fusion_dropout=float(cfg["fusion_dropout"]),
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
    *,
    out_dir: Path,
    stage: str,
    epoch: int,
    fold: int,
    stage_completed: bool,
    model: MultiShotLocalizer,
    optimizer,
    scheduler,
    scaler,
    cfg: dict,
    fold_plan: list,
    best_metric: dict,
    early_stop_counter: int,
    metrics_history: list,
    rolling: bool,
    extra_path: Path | None = None,
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
# Phase 0 — zero-shot OWLv2 baseline
# ---------------------------------------------------------------------------


_TRAIN_PRIORITY = ("loss", "patch_ce", "l1", "giou", "grad_norm", "alpha", "n_steps")
_VAL_PRIORITY = ("n", "n_pos",
                 "map_50", "map_75", "map_5095",
                 "map_50_containment", "map_90_containment",
                 "iou_mean", "iou_median", "iou_std",
                 "frac_iou_50", "frac_iou_75", "frac_iou_90",
                 "containment_mean", "containment_median", "containment_std",
                 "frac_containment_50", "frac_containment_75",
                 "frac_containment_90", "frac_containment_full",
                 "contain_at_iou_50", "contain_at_iou_75",
                 "high_contain_high_iou",
                 "mean_pred_box_area", "std_pred_box_area",
                 "frac_pred_box_too_big", "frac_pred_box_too_small",
                 "mean_gt_box_area",
                 "pred_to_gt_area_ratio_median", "log_area_ratio_mean",
                 "center_distance_mean",
                 "score_mean", "score_iou_correlation")
_PER_SOURCE_KEYS = ("n", "n_pos", "map_50", "map_5095", "map_50_containment",
                    "iou_mean", "containment_mean", "frac_containment_90",
                    "center_distance_mean")


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
    print(f"=== [localizer] Phase 0 (zero-shot OWLv2) on {device} ===")

    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg, lora_active=False).to(device)
    metrics: dict[str, Any] = {}

    # HOTS + InsDet test split.
    test_eps = int(cfg.get("test_episodes", 400))
    test_ds, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]),
        seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
    )
    if len(test_ds.instances) > 0:
        t0 = time.time()
        metrics["test"] = evaluate(
            model, test_loader, device, phase0=True, progress_every=5,
        )
        metrics["test"]["wall_clock_seconds"] = round(time.time() - t0, 2)

    write_json(out_dir / "results.json", metrics)
    write_json(analysis_dir / "results.json", metrics)
    print(f"[localizer] Phase 0 complete. Results: {out_dir / 'results.json'}")
    return metrics


def evaluate_phase0(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _evaluate_phase0_inner(user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def _evaluate_phase0_inner(user_kwargs: dict) -> dict:
    """Evaluate zero-shot OWLv2 on the held-out test split only."""
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    print(f"=== [localizer] Phase 0 evaluation on test split ({device}) ===")
    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg, lora_active=False).to(device)
    test_eps = int(cfg.get("test_episodes", 400))
    test_ds, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]), seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
    )
    t0 = time.time()
    metrics = evaluate(model, test_loader, device, phase0=True)
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)
    out_dir = Path(cfg["out_root"]) / MODEL_KIND / "phase0"
    analysis_dir = Path(cfg["analysis_root"]) / MODEL_KIND / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    write_json(out_dir / "test_eval.json", metrics)
    write_json(analysis_dir / f"test_eval_{ts}.json", metrics)
    o = metrics["overall"]
    print(
        f"[localizer phase0] test  "
        f"mAP@50={o.get('map_50', 0.0):.4f}  "
        f"mAP@75={o.get('map_75', 0.0):.4f}  "
        f"mAP@50:95={o.get('map_5095', 0.0):.4f}  "
        f"IoU={o.get('iou_mean', 0.0):.4f}  "
        f"contain={o.get('containment_mean', 0.0):.4f}  "
        f"contain>=.9={o.get('frac_containment_90', 0.0):.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Stage runner (L1 / L2 / L3)
# ---------------------------------------------------------------------------


def _run_stage(stage: str, *, user_kwargs: dict) -> dict:
    cfg = _merge_cfg(user_kwargs)
    cfg = _stage_lrs(stage, cfg)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, stage)

    print(f"\n=== [localizer] Stage {stage} on {device} ===")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _set_seed(int(cfg["seed"]))

    # --- model ----------------------------------------------------------
    lora_active = (stage == "L3")
    model = _build_model(cfg, lora_active=lora_active).to(device)

    # --- resume / warm-start -------------------------------------------
    resume = user_kwargs.get("resume", True)
    resume_path = resolve_resume_path(resume, out_dir)
    if resume_path is None:
        # Cross-stage warm start.
        prev_map = {"L1": None, "L2": "L1", "L3": "L2"}
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

    # --- optimiser / scheduler -----------------------------------------
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

    # --- restore optimizer state if mid-stage --------------------------
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
                print(f"  warning: failed to restore optimizer/scheduler ({e}); fresh start within stage")

    # --- fold plan -----------------------------------------------------
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

    box_warmup_epochs = int(cfg.get("L2_box_warmup_epochs", 2)) if stage == "L2" else 0

    epoch = resume_epoch
    for epoch in range(resume_epoch, n_epochs + 1):
        # L2 box-head warmup.
        if stage == "L2":
            if epoch <= box_warmup_epochs:
                model.freeze_box_head()
            else:
                model.unfreeze_box_head()

        fold_jsons: list[dict] = [
            m for m in metrics_history
            if m.get("epoch") == epoch and m.get("fold", -1) < resume_fold
        ] if epoch == resume_epoch else []

        for fold_idx in range(K):
            if epoch == resume_epoch and fold_idx < resume_fold:
                continue
            t0 = time.time()
            fold = fold_plan[fold_idx]
            print(f"▶ [localizer] {stage} epoch {epoch}/{n_epochs} fold {fold_idx}/{K - 1}",
                  flush=True)

            train_ds, train_loader = build_train_loader(
                manifest=cfg["manifest"], data_root=cfg["data_root"],
                split="train", sources=None,
                episodes_per_epoch=eps_per_fold,
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                img_size=int(cfg["img_size"]),
                seed=int(cfg["seed"]) + 1000 * epoch + fold_idx,
                k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
                aug_kwargs=_augmentation_kwargs(cfg),
            )
            train_ds.set_fold(train_ids=set(fold["train_ids"]))
            val_ds, val_loader = build_val_loader(
                manifest=cfg["manifest"], data_root=cfg["data_root"],
                split="train", sources=None,
                val_episodes=int(cfg["val_episodes"]),
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                img_size=int(cfg["img_size"]),
                seed=int(cfg["seed"]) + 7000 * epoch + fold_idx,
                k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
            )
            val_ds.set_fold(val_ids=set(fold["val_ids"]))

            # L1: box_head is frozen, so the box L1+GIoU terms cannot
            # produce gradient. Skip them — patch-CE alone drives the
            # fusion. L2 with box_head still in warmup also has it frozen
            # for the first N epochs, but we still keep the box loss term
            # active because the class_head IS trainable and benefits from
            # seeing the joint signal as soon as possible.
            stage_uses_box_loss = stage in ("L2", "L3")
            train_metrics = train_one_pass(
                model=model, optimizer=optimizer, scheduler=scheduler,
                loader=train_loader, device=device, cfg=cfg,
                scaler=scaler, use_amp=bool(cfg["use_amp"]),
                use_box_loss=stage_uses_box_loss,
            )
            val_metrics = evaluate(model, val_loader, device, progress_every=20)

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
                header=f"localizer {stage} epoch {epoch}/{n_epochs} fold {fold_idx}/{K - 1}",
                train_metrics=train_metrics, val_metrics=val_metrics,
                lr_groups={g.get("name", str(i)): g["lr"]
                           for i, g in enumerate(optimizer.param_groups)},
                wall_clock=payload["wall_clock_seconds"],
                train_priority=_TRAIN_PRIORITY, val_priority=_VAL_PRIORITY,
                per_source_keys=_PER_SOURCE_KEYS,
            )
            del train_loader, val_loader, train_ds, val_ds

        # End of epoch.
        aggregate = aggregate_folds(fold_jsons)
        write_json(analysis_dir / f"epoch_{epoch:03d}" / "aggregate.json", aggregate)
        print_aggregate(stage, epoch, aggregate, keys=(
            ("val.overall.map_50",                 "map_50"),
            ("val.overall.map_75",                 "map_75"),
            ("val.overall.map_5095",               "map_50:95"),
            ("val.overall.map_50_containment",     "map_50_contain"),
            ("val.overall.map_90_containment",     "map_90_contain"),
            ("val.overall.iou_mean",               "iou_mean"),
            ("val.overall.containment_mean",       "contain_mean"),
            ("val.overall.frac_containment_90",    "contain>=0.90"),
            ("val.overall.frac_containment_full",  "contain==full"),
            ("val.overall.high_contain_high_iou",  "iou>=.5 & con>=.9"),
            ("val.overall.center_distance_mean",   "center_dist"),
            ("val.overall.score_iou_correlation",  "score↔iou_corr"),
            ("train.loss",                         "train_loss"),
            ("train.patch_ce",                     "train_patch_ce"),
            ("train.l1",                           "train_l1"),
            ("train.giou",                         "train_giou"),
            ("train.alpha",                        "alpha"),
        ))
        map50 = aggregate["metrics"].get("val.overall.map_50", {}).get("mean", 0.0)
        if map50 > best_metric["value"]:
            best_metric = {"value": map50, "epoch": epoch, "fold": K - 1}
            _save_stage_ckpt(
                out_dir=out_dir, stage=stage, epoch=epoch, fold=K - 1,
                stage_completed=False, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, cfg=cfg,
                fold_plan=fold_plan, best_metric=best_metric,
                early_stop_counter=0, metrics_history=metrics_history,
                rolling=False, extra_path=out_dir / "best.pt",
            )
            early_stop_counter = 0
            print(f"    ↳ best metric so far: map_50_mean={map50:.4f} at epoch {epoch}")
        else:
            early_stop_counter += 1
        update_summary(Path(cfg["analysis_root"]) / MODEL_KIND, {
            f"{stage}.val.map_50_mean": (epoch, map50),
            f"{stage}.val.iou_mean":    (epoch,
                aggregate["metrics"].get("val.overall.iou_mean", {}).get("mean", 0.0)),
        })
        # reset within-epoch fold pointer
        resume_fold = 0

        # Early stopping.
        patience = int(cfg.get(f"{stage}_early_stop_patience", 4))
        if early_stop_counter >= patience:
            print(f"  early stop: {patience} epochs without map_50 improvement")
            break

    # Stage-completion ckpt.
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
        "best_val_map_50_mean": best_metric["value"],
        "best_epoch": best_metric["epoch"],
    })
    print(f"[localizer] {stage} complete. Best val map_50_mean = {best_metric['value']:.4f} "
          f"at epoch {best_metric['epoch']}")
    return {"best_metric": best_metric, "config": cfg}


def train_stage_L1(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _run_stage("L1", user_kwargs=user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def train_stage_L2(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _run_stage("L2", user_kwargs=user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def train_stage_L3(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _run_stage("L3", user_kwargs=user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


# ---------------------------------------------------------------------------
# Eval-only entrypoint
# ---------------------------------------------------------------------------


def evaluate_run(checkpoint: str, **user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _evaluate_run_inner(checkpoint, user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def _evaluate_run_inner(checkpoint: str, user_kwargs: dict) -> dict:
    """Load ``checkpoint`` and evaluate on the held-out test split."""
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    print(f"=== [localizer] Evaluating {checkpoint} on test split ({device}) ===")
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", "unknown")
    lora_active = bool(ckpt.get("lora_active", stage == "L3"))

    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg, lora_active=lora_active)
    load_trainable_state(model, ckpt.get("state_dict", {}))
    model = model.to(device)

    test_eps = int(cfg.get("test_episodes", 400))
    _, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]),
        seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
    )
    t0 = time.time()
    metrics = evaluate(model, test_loader, device)
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
        f"[localizer {stage}] test  "
        f"mAP@50={o.get('map_50', 0.0):.4f}  "
        f"mAP@75={o.get('map_75', 0.0):.4f}  "
        f"mAP@50:95={o.get('map_5095', 0.0):.4f}  "
        f"mAP@50_contain={o.get('map_50_containment', 0.0):.4f}  "
        f"IoU={o.get('iou_mean', 0.0):.4f}  "
        f"contain={o.get('containment_mean', 0.0):.4f}  "
        f"contain>=.9={o.get('frac_containment_90', 0.0):.4f}  "
        f"contain==full={o.get('frac_containment_full', 0.0):.4f}  "
        f"({metrics['wall_clock_seconds']:.1f}s)"
    )
    return metrics
