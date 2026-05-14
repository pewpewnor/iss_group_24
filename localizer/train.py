"""Localizer training orchestrator.

Public entry points:
    train_stage_L1(...)
    train_stage_L2(...)
    train_stage_L3(...)
    evaluate_phase0(...)
    evaluate_run(checkpoint=..., ...)

Stage curriculum (NEW):
    Each stage can run in 1..N sub-phases, each with its own source filter and
    epoch count. Configure via cfg keys like:
        "L2_curriculum": ["insdet", "hots", "mixed"]
        "L2_epochs_insdet": 4
        "L2_epochs_hots":   3
        "L2_epochs_mixed":  5

    Backwards-compat: if ``L2_curriculum`` is absent (or empty), we fall back
    to a single ``mixed`` phase with ``L2_epochs`` epochs (the old behaviour).

L1 trains positives only (fusion warm-up). L2/L3 mix in negatives so the
abstain channel learns. Set ``L<N>_neg_prob`` in cfg to override the default
per-stage neg rate.
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
# Default config
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
    # ──────────────────────────────────────────────────────────────────
    # Stage role + sizing (REBALANCED).
    #
    #   L1  fusion-only warm-up. Frozen OWLv2, no LoRA. The fusion +
    #       support-attn-pool learns to "do no harm" via patch-CE.
    #       Positive-only — abstain channel has nothing to learn yet.
    #
    #   L2  Short WARM-UP for L3. Unfreezes OWLv2 class_head + box_head +
    #       layer_norm. Bigger LRs than L3 so the heads + log-box wrapper
    #       can snap into the fused prototype. NO LoRA yet (backbone
    #       still frozen). Mixed pos+neg so the abstain channel begins.
    #
    #   L3  MAIN fine-tune. Attaches LoRA on last 4 ViT blocks. Drops
    #       fusion/head LRs ~5× from L2 (they're already converged from
    #       the L2 warm-up); LoRA gets a fresh-init LR. This is where
    #       real mAP gains come from. Longest stage.
    # ──────────────────────────────────────────────────────────────────
    "L1_epochs": 4,
    "L2_epochs": 4,
    "L3_epochs": 12,
    "L1_eps_per_fold": 400,
    "L2_eps_per_fold": 300,
    "L3_eps_per_fold": 300,
    "val_episodes": 100,
    "test_episodes": 400,
    # Curriculum (per-stage). Empty list ⇒ single "mixed" phase.
    # Order matters; phases run sequentially with their own epoch budgets.
    "L1_curriculum": ["insdet", "hots", "mixed"],
    "L2_curriculum": ["insdet", "hots", "mixed"],
    "L3_curriculum": ["insdet", "hots", "mixed"],
    # L1: short, evenly across phases.
    "L1_epochs_insdet": 1,
    "L1_epochs_hots":   1,
    "L1_epochs_mixed":  2,
    # L2: short warm-up. Bias toward "mixed" so the heads see the full
    # distribution before L3 starts touching the backbone.
    "L2_epochs_insdet": 1,
    "L2_epochs_hots":   1,
    "L2_epochs_mixed":  2,
    # L3: the workhorse. Most epochs on mixed; per-source phases give the
    # LoRA adapters time to specialise before mixing.
    "L3_epochs_insdet": 3,
    "L3_epochs_hots":   3,
    "L3_epochs_mixed":  6,
    # Per-stage negative-episode ratios. L1 stays positive-only.
    "L1_neg_prob": 0.0,
    "L2_neg_prob": 0.25,
    "L3_neg_prob": 0.30,
    # LRs (REBALANCED).
    #
    # L1 fusion gets a moderately-high LR (only ~3M params + frozen backbone).
    # L2 heads + fusion get the *biggest* LRs in the schedule — this is a
    # short warm-up so we want fast convergence of the new heads against the
    # fused prototype before L3 starts moving the backbone.
    # L3 drops everything 4-5× and adds LoRA at its own fresh-init LR.
    "lr_fusion_L1": 2e-4,
    "lr_fusion_L2": 3e-4,
    "lr_class_L2":  1e-4,
    "lr_box_L2":    1e-4,
    "lr_fusion_L3": 5e-5,
    "lr_class_L3":  2e-5,
    "lr_box_L3":    2e-5,
    "lr_lora_L3":   2e-4,
    # Optim
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_frac": 0.05,
    # Loss
    "lambda_patch_ce": 1.0,
    "lambda_l1": 2.0,
    "lambda_giou": 4.0,
    "lambda_log_area": 0.3,
    "patch_ce_label_smoothing": 0.05,
    "patch_ce_neighbour_radius": 1,
    "patch_ce_neighbour_weight": 0.30,
    # L2 is itself a warm-up, so we no longer freeze the box head for the
    # first N epochs of it. Set to 0 to disable; >0 to re-enable.
    "L2_box_warmup_epochs": 0,
    # Architecture
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
    # Early stopping
    "L1_early_stop_patience": 4,
    "L2_early_stop_patience": 4,
    "L3_early_stop_patience": 4,
    # Eval
    "abstain_threshold": 0.5,
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
    # Smoke runs one phase only to keep wall-clock small.
    "L1_curriculum": ["mixed"],
    "L2_curriculum": ["mixed"],
    "L3_curriculum": ["mixed"],
    "L1_epochs_mixed": 1,
    "L2_epochs_mixed": 1,
    "L3_epochs_mixed": 1,
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
    suf = stage
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
    phase: str | None = None,
) -> Path:
    state = get_trainable_state(model)
    payload = {
        "model_kind": MODEL_KIND,
        "stage": stage,
        "phase": phase,
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
        atomic_save_multi(payload, targets)
        return rolling_path
    if extra_path is not None:
        atomic_save(payload, extra_path, label=extra_path.stem)
    return out_dir / "last.pt"


# ---------------------------------------------------------------------------
# Phase 0 — zero-shot OWLv2 baseline
# ---------------------------------------------------------------------------


_TRAIN_PRIORITY = ("loss", "patch_ce", "l1", "giou", "log_area", "grad_norm",
                   "alpha", "bg_bias", "n_steps")
_VAL_PRIORITY = (
    "n", "n_pos", "n_neg",
    "map_50", "map_75", "map_5095",
    "map_50_containment", "map_90_containment",
    "iou_mean", "iou_median", "iou_std",
    "frac_iou_50", "frac_iou_75", "frac_iou_90",
    "containment_mean", "containment_median",
    "frac_containment_90", "frac_containment_full",
    "contain_at_iou_50", "high_contain_high_iou",
    "frac_pred_box_too_big", "frac_pred_box_too_small",
    "pred_to_gt_area_ratio_median", "log_area_ratio_mean", "log_area_ratio_std",
    "center_distance_mean",
    "score_mean", "score_iou_correlation",
    "bg_prob_pos_mean", "bg_prob_neg_mean", "abstain_gap",
    "abstain_rate_pos", "abstain_rate_neg", "fp_rate", "tn_rate",
)
_PER_SOURCE_KEYS = (
    "n", "n_pos", "n_neg",
    "map_50", "map_5095", "iou_mean", "frac_containment_90",
    "abstain_rate_pos", "abstain_rate_neg", "score_iou_correlation",
)


def evaluate_phase0(**user_kwargs) -> dict:
    try:
        with gpu_cleanup_on_exit():
            return _evaluate_phase0_inner(user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def evaluate_phase0_final_style(**user_kwargs) -> dict:
    """Phase 0 baseline evaluated under the same loader regime as the final
    (L3) eval: full K range and mixed positives + negatives at
    ``cfg['L3_neg_prob']``. Lets abstain / FP-rate / TN-rate / score-IoU
    metrics be compared apples-to-apples against the trained pipeline.
    """
    try:
        with gpu_cleanup_on_exit():
            return _evaluate_phase0_final_style_inner(user_kwargs)
    finally:
        release_gpu_memory(verbose=False)


def train_phase0(**user_kwargs) -> dict:
    """Alias for evaluate_phase0 — no training happens at Phase 0."""
    return evaluate_phase0(**user_kwargs)


def _evaluate_phase0_inner(user_kwargs: dict) -> dict:
    """Phase 0 = ONE-SHOT vanilla OWLv2 baseline (no fusion, no fine-tuning).

    Uses the first valid support per episode and runs the bare OWLv2
    image-guided detection path. This is the numerical floor the trained
    L1/L2/L3 stages must beat.
    """
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, "phase0")
    print(f"=== [localizer] Phase 0 evaluation: one-shot vanilla OWLv2 ({device}) ===")
    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg, lora_active=False).to(device)
    test_eps = int(cfg.get("test_episodes", 400))
    # Force K=1 supports per episode so the phase0_forward path that picks
    # the first valid slot sees exactly one support. This matches the
    # "one-shot vanilla OWLv2 baseline" naming.
    test_ds, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]), seed=int(cfg["seed"]),
        k_min=1, k_max=1,
        force_positive=True,
    )
    t0 = time.time()
    metrics = evaluate(model, test_loader, device, phase0=True)
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)
    metrics["baseline_kind"] = "one_shot_vanilla_owlv2"
    ts = time.strftime("%Y%m%d_%H%M%S")
    write_json(out_dir / "test_eval.json", metrics)
    write_json(analysis_dir / f"test_eval_{ts}.json", metrics)
    write_json(out_dir / "results.json", metrics)
    write_json(analysis_dir / "results.json", metrics)
    o = metrics["overall"]
    print(
        f"[localizer phase0] test  "
        f"mAP@50={o.get('map_50', 0.0):.4f}  "
        f"mAP@75={o.get('map_75', 0.0):.4f}  "
        f"mAP@50:95={o.get('map_5095', 0.0):.4f}  "
        f"IoU={o.get('iou_mean', 0.0):.4f}  "
        f"contain>=.9={o.get('frac_containment_90', 0.0):.4f}"
    )
    return metrics


def _evaluate_phase0_final_style_inner(user_kwargs: dict) -> dict:
    """Phase 0 baseline under the L3 eval loader regime (mixed pos+neg, full K).

    Same model as ``_evaluate_phase0_inner`` (vanilla OWLv2, no fusion / LoRA /
    trained heads) but mirrors the loader of ``_evaluate_run_inner`` for L3:
    ``neg_prob = cfg['L3_neg_prob']`` (default 0.30) and full
    ``k_min..k_max``. Saved under ``phase0/test_eval_final_style.json`` so the
    existing one-shot positive-only baseline is preserved.
    """
    cfg = _merge_cfg(user_kwargs)
    device = _resolve_device(cfg)
    out_dir, analysis_dir = _stage_dirs(cfg, "phase0")
    neg_prob = float(cfg.get("L3_neg_prob", 0.30))
    force_positive = (neg_prob <= 0.0)
    print(
        f"=== [localizer] Phase 0 final-style evaluation: vanilla OWLv2 "
        f"@ neg_prob={neg_prob:.2f}, K={cfg['k_min']}..{cfg['k_max']} ({device}) ==="
    )
    _set_seed(int(cfg["seed"]))
    model = _build_model(cfg, lora_active=False).to(device)
    test_eps = int(cfg.get("test_episodes", 400))
    _, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]), seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
        force_positive=force_positive, neg_prob=neg_prob,
    )
    t0 = time.time()
    metrics = evaluate(
        model, test_loader, device, phase0=True,
        abstain_threshold=float(cfg.get("abstain_threshold", 0.5)),
    )
    metrics["wall_clock_seconds"] = round(time.time() - t0, 2)
    metrics["baseline_kind"] = "phase0_final_style_mixed_neg"
    metrics["neg_prob"] = neg_prob
    ts = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "stage": "phase0",
        "variant": "final_style",
        "split": "test",
        "test_episodes": test_eps,
        "neg_prob": neg_prob,
        "k_min": int(cfg["k_min"]), "k_max": int(cfg["k_max"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg, "metrics": metrics,
    }
    write_json(out_dir / "test_eval_final_style.json", payload)
    write_json(analysis_dir / f"test_eval_final_style_{ts}.json", payload)
    o = metrics["overall"]
    print(
        f"[localizer phase0 final-style] test  "
        f"mAP@50={o.get('map_50', 0.0):.4f}  "
        f"mAP@5095={o.get('map_5095', 0.0):.4f}  "
        f"IoU={o.get('iou_mean', 0.0):.4f}  "
        f"abstain_pos={o.get('abstain_rate_pos', 0.0):.3f}  "
        f"abstain_neg={o.get('abstain_rate_neg', 0.0):.3f}  "
        f"sc↔iou={o.get('score_iou_correlation', 0.0):.3f}  "
        f"({metrics['wall_clock_seconds']:.1f}s)"
    )
    return metrics


# ---------------------------------------------------------------------------
# Stage runner (L1 / L2 / L3) with curriculum
# ---------------------------------------------------------------------------


_PHASE_TO_SOURCES: dict[str, list[str] | None] = {
    "insdet": ["insdet"],
    "hots":   ["hots"],
    "mixed":  None,
}


def _resolve_curriculum(stage: str, cfg: dict) -> list[tuple[str, int]]:
    """Return [(phase_name, n_epochs), ...] for ``stage``.

    Backwards-compat: if cfg[stage_curriculum] is empty / missing, we run a
    single ``mixed`` phase with cfg[stage_epochs] epochs.
    """
    curr = list(cfg.get(f"{stage}_curriculum", []) or [])
    if not curr:
        n = int(cfg.get(f"{stage}_epochs", 1))
        return [("mixed", n)]
    out: list[tuple[str, int]] = []
    for ph in curr:
        n = int(cfg.get(f"{stage}_epochs_{ph}", 0))
        if n <= 0:
            continue
        out.append((ph, n))
    if not out:
        # Fall through to legacy single-phase if every sub-phase budget was 0.
        n = int(cfg.get(f"{stage}_epochs", 1))
        return [("mixed", n)]
    return out


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

    lora_active = (stage == "L3")
    model = _build_model(cfg, lora_active=lora_active).to(device)

    # --- resume / warm-start --------------------------------------------
    resume = user_kwargs.get("resume", True)
    resume_path = resolve_resume_path(resume, out_dir)
    if resume_path is None:
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

    optimizer, _lora_params = build_optimizer_for_stage(stage, model, cfg)
    curriculum = _resolve_curriculum(stage, cfg)
    K = int(cfg["folds"])
    eps_per_fold = int(cfg[f"{stage}_eps_per_fold"])
    steps_per_fold_per_epoch = max(
        1,
        eps_per_fold // max(1, int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])),
    )
    total_steps = steps_per_fold_per_epoch * K * sum(n for _, n in curriculum)
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_frac=float(cfg["warmup_frac"]),
    )
    scaler = _make_scaler(bool(cfg["use_amp"]), device)

    # --- restore optimiser if mid-stage ---------------------------------
    resume_global_epoch = 1   # epoch index running across all phases (1-based)
    best_metric: dict[str, Any] = {"value": -1.0, "epoch": 0, "fold": 0, "phase": ""}
    early_stop_counter = 0
    metrics_history: list[dict] = []
    resume_fold = 0

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
                    resume_global_epoch = saved_epoch + 1
                    resume_fold = 0
                else:
                    resume_global_epoch = saved_epoch
                    resume_fold = saved_fold + 1
                best_metric = dict(ckpt_full.get("best_metric") or best_metric)
                early_stop_counter = int(ckpt_full.get("early_stop_counter", 0))
                metrics_history = list(ckpt_full.get("metrics_history") or [])
                print(f"  restored optim+sched at epoch={saved_epoch} fold={saved_fold}; "
                      f"continuing from epoch={resume_global_epoch} fold={resume_fold}")
            except Exception as e:                                                 # noqa: BLE001
                print(f"  warning: failed to restore optimizer/scheduler ({e}); fresh start within stage")

    # --- fold plan ------------------------------------------------------
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
    write_json(analysis_dir / "curriculum.json", {"phases": curriculum})

    # Walk the curriculum. ``global_epoch`` is the trainer-wide epoch counter
    # (visible in checkpoints and per-epoch logs); ``phase_idx`` indexes the
    # current curriculum entry.
    total_planned = sum(n for _, n in curriculum)
    box_warmup_epochs = int(cfg.get("L2_box_warmup_epochs", 2)) if stage == "L2" else 0
    global_epoch = 0
    epoch = 0  # for the final stage_complete ckpt

    for phase_idx, (phase_name, phase_epochs) in enumerate(curriculum):
        phase_neg_prob = (
            0.0 if phase_name == "insdet" and stage == "L1" else
            float(cfg.get(f"{stage}_neg_prob", 0.0))
        )
        force_positive = (phase_neg_prob <= 0.0)
        sources = _PHASE_TO_SOURCES[phase_name]
        print(f"\n── [localizer] {stage} curriculum phase {phase_idx + 1}/{len(curriculum)}: "
              f"name={phase_name} sources={sources} neg_prob={phase_neg_prob:.2f} "
              f"epochs={phase_epochs}", flush=True)

        for local_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            epoch = global_epoch
            if global_epoch < resume_global_epoch:
                # We already finished this whole epoch on the previous run.
                continue

            # L2 box-head warmup is measured in GLOBAL epochs to keep the
            # warm-up duration deterministic regardless of phase boundaries.
            if stage == "L2":
                if global_epoch <= box_warmup_epochs:
                    model.freeze_box_head()
                else:
                    model.unfreeze_box_head()

            fold_jsons: list[dict] = [
                m for m in metrics_history
                if m.get("epoch") == global_epoch and m.get("fold", -1) < resume_fold
            ] if global_epoch == resume_global_epoch else []

            for fold_idx in range(K):
                if global_epoch == resume_global_epoch and fold_idx < resume_fold:
                    continue
                t0 = time.time()
                fold = fold_plan[fold_idx]
                print(f"▶ [localizer] {stage}/{phase_name} epoch {global_epoch}/{total_planned} "
                      f"fold {fold_idx}/{K - 1}", flush=True)

                train_ds, train_loader = build_train_loader(
                    manifest=cfg["manifest"], data_root=cfg["data_root"],
                    split="train", sources=sources,
                    episodes_per_epoch=eps_per_fold,
                    batch_size=int(cfg["batch_size"]),
                    num_workers=int(cfg["num_workers"]),
                    img_size=int(cfg["img_size"]),
                    seed=int(cfg["seed"]) + 1000 * global_epoch + fold_idx,
                    k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
                    force_positive=force_positive,
                    neg_prob=phase_neg_prob,
                    aug_kwargs=_augmentation_kwargs(cfg),
                )
                train_ds.set_fold(train_ids=set(fold["train_ids"]))
                val_ds, val_loader = build_val_loader(
                    manifest=cfg["manifest"], data_root=cfg["data_root"],
                    split="train", sources=sources,
                    val_episodes=int(cfg["val_episodes"]),
                    batch_size=int(cfg["batch_size"]),
                    num_workers=int(cfg["num_workers"]),
                    img_size=int(cfg["img_size"]),
                    seed=int(cfg["seed"]) + 7000 * global_epoch + fold_idx,
                    k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
                    force_positive=force_positive,
                    neg_prob=phase_neg_prob,
                )
                val_ds.set_fold(val_ids=set(fold["val_ids"]))

                # Box loss off at L1 (box_head frozen).
                stage_uses_box_loss = stage in ("L2", "L3")
                train_metrics = train_one_pass(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    loader=train_loader, device=device, cfg=cfg,
                    scaler=scaler, use_amp=bool(cfg["use_amp"]),
                    use_box_loss=stage_uses_box_loss,
                )
                val_metrics = evaluate(
                    model, val_loader, device, progress_every=20,
                    abstain_threshold=float(cfg.get("abstain_threshold", 0.5)),
                )

                payload = {
                    "stage": stage, "phase": phase_name,
                    "epoch": global_epoch, "fold": fold_idx,
                    "wall_clock_seconds": round(time.time() - t0, 2),
                    "lr": {g.get("name", str(i)): g["lr"]
                           for i, g in enumerate(optimizer.param_groups)},
                    "train": train_metrics,
                    "val": val_metrics,
                }
                write_json(analysis_dir / f"epoch_{global_epoch:03d}" / f"fold_{fold_idx}.json", payload)
                fold_jsons.append(payload)
                metrics_history.append(payload)

                _save_stage_ckpt(
                    out_dir=out_dir, stage=stage, epoch=global_epoch, fold=fold_idx,
                    stage_completed=False, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, cfg=cfg,
                    fold_plan=fold_plan, best_metric=best_metric,
                    early_stop_counter=early_stop_counter,
                    metrics_history=metrics_history, rolling=True,
                    phase=phase_name,
                )
                hygiene(out_dir, int(cfg["keep_last_n"]))

                print_epoch_log(
                    header=f"localizer {stage}/{phase_name} "
                           f"epoch {global_epoch}/{total_planned} fold {fold_idx}/{K - 1}",
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
            write_json(analysis_dir / f"epoch_{global_epoch:03d}" / "aggregate.json", aggregate)
            print_aggregate(stage, global_epoch, aggregate, keys=(
                ("val.overall.map_50",                 "map_50"),
                ("val.overall.map_5095",               "map_50:95"),
                ("val.overall.iou_mean",               "iou_mean"),
                ("val.overall.frac_iou_50",            "frac_iou_50"),
                ("val.overall.frac_containment_90",    "contain>=.9"),
                ("val.overall.score_iou_correlation",  "score↔iou_corr"),
                ("val.overall.bg_prob_pos_mean",       "bg_p_pos"),
                ("val.overall.bg_prob_neg_mean",       "bg_p_neg"),
                ("val.overall.abstain_rate_pos",       "abstain_pos"),
                ("val.overall.abstain_rate_neg",       "abstain_neg"),
                ("val.overall.frac_pred_box_too_small", "frac_too_small"),
                ("val.overall.log_area_ratio_mean",     "log_area"),
                ("val.overall.center_distance_mean",    "center_dist"),
                ("train.loss",                         "train_loss"),
                ("train.patch_ce",                     "patch_ce"),
                ("train.l1",                           "l1"),
                ("train.giou",                         "giou"),
                ("train.log_area",                     "log_area_loss"),
                ("train.alpha",                        "alpha"),
                ("train.bg_bias",                      "bg_bias"),
            ))
            map50 = aggregate["metrics"].get("val.overall.map_50", {}).get("mean", 0.0)
            if map50 > best_metric["value"]:
                best_metric = {"value": map50, "epoch": global_epoch,
                               "fold": K - 1, "phase": phase_name}
                _save_stage_ckpt(
                    out_dir=out_dir, stage=stage, epoch=global_epoch, fold=K - 1,
                    stage_completed=False, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, cfg=cfg,
                    fold_plan=fold_plan, best_metric=best_metric,
                    early_stop_counter=0, metrics_history=metrics_history,
                    rolling=False, extra_path=out_dir / "best.pt",
                    phase=phase_name,
                )
                early_stop_counter = 0
                print(f"    ↳ best metric so far: map_50_mean={map50:.4f} at epoch {global_epoch} "
                      f"(phase={phase_name})")
            else:
                early_stop_counter += 1
            update_summary(Path(cfg["analysis_root"]) / MODEL_KIND, {
                f"{stage}.val.map_50_mean": (global_epoch, map50),
                f"{stage}.val.iou_mean":    (global_epoch,
                    aggregate["metrics"].get("val.overall.iou_mean", {}).get("mean", 0.0)),
            })
            resume_fold = 0

            patience = int(cfg.get(f"{stage}_early_stop_patience", 4))
            if early_stop_counter >= patience:
                print(f"  early stop: {patience} epochs without map_50 improvement "
                      f"(in phase {phase_name})")
                # Break both inner and outer loops cleanly.
                # We use a sentinel to escape the curriculum walk.
                _early_stop_outer = True
                break
        else:
            _early_stop_outer = False
            continue
        # Break-out from the epoch loop happened ⇒ break the phase loop too.
        if _early_stop_outer:
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
        phase=best_metric.get("phase"),
    )
    write_json(analysis_dir / "complete.json", {
        "stage_completed": True,
        "epochs_run": epoch,
        "best_val_map_50_mean": best_metric["value"],
        "best_epoch": best_metric["epoch"],
        "best_phase": best_metric.get("phase"),
        "curriculum": curriculum,
    })
    print(f"[localizer] {stage} complete. Best val map_50_mean = {best_metric['value']:.4f} "
          f"at epoch {best_metric['epoch']} (phase={best_metric.get('phase')})")
    return {"best_metric": best_metric, "config": cfg, "curriculum": curriculum}


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
    """Load ``checkpoint`` and evaluate on test (mixed pos+neg)."""
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
    # Eval on mixed pos+neg by default at L2/L3 (so abstain metrics are meaningful).
    # L1 stays positive-only.
    neg_prob = 0.0 if stage == "L1" else float(cfg.get(f"{stage}_neg_prob", 0.25))
    force_positive = (neg_prob <= 0.0)
    _, test_loader = build_val_loader(
        manifest=cfg["manifest"], data_root=cfg["data_root"],
        split="test", sources=None, val_episodes=test_eps,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]), seed=int(cfg["seed"]),
        k_min=int(cfg["k_min"]), k_max=int(cfg["k_max"]),
        force_positive=force_positive, neg_prob=neg_prob,
    )
    t0 = time.time()
    metrics = evaluate(
        model, test_loader, device,
        abstain_threshold=float(cfg.get("abstain_threshold", 0.5)),
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
        "neg_prob": neg_prob,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg, "metrics": metrics,
    }
    write_json(out_dir / "test_eval.json", payload)
    write_json(analysis_dir / f"test_eval_{ts}.json", payload)

    o = metrics["overall"]
    print(
        f"[localizer {stage}] test  "
        f"mAP@50={o.get('map_50', 0.0):.4f}  "
        f"mAP@5095={o.get('map_5095', 0.0):.4f}  "
        f"IoU={o.get('iou_mean', 0.0):.4f}  "
        f"abstain_pos={o.get('abstain_rate_pos', 0.0):.3f}  "
        f"abstain_neg={o.get('abstain_rate_neg', 0.0):.3f}  "
        f"sc↔iou={o.get('score_iou_correlation', 0.0):.3f}  "
        f"({metrics['wall_clock_seconds']:.1f}s)"
    )
    return metrics
