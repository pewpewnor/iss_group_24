"""Three-stage trainer (orchestrator).

Implementation split across ``modeling/_*.py`` modules; this file just wires
them together and exposes:

  - ``train(...)``         — runs all stages (default: 1 → 2 → 3).
  - ``train_stage1(...)``  — Stage 1 only (head warmup, no CV).
  - ``train_stage2(...)``  — Stage 2 only (partial unfreeze, no CV).
  - ``train_stage3(...)``  — Stage 3 only (full unfreeze + K-fold CV).

Universal resume protocol: every saved checkpoint (per-epoch, per-fold,
stage-completion, last, best) can be used as the ``resume`` argument.
"""

from __future__ import annotations

import argparse
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
    next_resume_point,
    quarantine_incompatible,
    resolve_resume_path,
    restore_rng,
    try_load_state_dict,
)
from modeling._folds import stratified_kfold
from modeling._loaders import build_train_loader, build_val_loader
from modeling._logging import print_epoch_log, print_stage3_aggregate
from modeling._optim import build_optimizer_for_stage, build_scheduler
from modeling._proto_cache import build_proto_cache
from modeling._train_loop import train_one_pass
from modeling.dataset import (
    DEFAULT_IMG_SIZE,
    DEFAULT_SOURCE_MIX,
    SourceBalancedBatchSampler,
    collate,
)
from modeling.evaluate import evaluate
from modeling.model import FewShotLocalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_cfg(local_args: dict) -> dict:
    """Snapshot every train() kwarg into a plain JSON-serialisable dict."""
    keep_types = (str, int, float, bool, list, tuple, dict, type(None))
    cfg: dict[str, Any] = {}
    for k, v in local_args.items():
        if k in ("self", "device_t"):
            continue
        if isinstance(v, keep_types):
            cfg[k] = v
    if cfg.get("source_mix") is None:
        cfg["source_mix"] = dict(DEFAULT_SOURCE_MIX)
    for k in ("multi_scale_sizes", "val_tta_sizes"):
        if isinstance(cfg.get(k), tuple):
            cfg[k] = list(cfg[k])
    return cfg


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_stage1(
    *, model, cfg: dict, manifest: str, data_root: str | None,
    out_dir: Path, analysis_dir: Path, device: torch.device,
    start: dict, loaded_opt, loaded_sched, loaded_rng, rebuild: bool,
    best_metric: dict,
) -> dict:
    if cfg["stage1_epochs"] <= 0 or start["stage"] > 1:
        return {"best_metric": best_metric, "loaded_opt": None, "loaded_sched": None,
                "loaded_rng": None, "rebuild": True}

    print(f"\n=== Stage 1 (head warmup, {cfg['stage1_epochs']} epochs) ===")
    train_ds, train_loader = build_train_loader(
        manifest=manifest, data_root=data_root, split="train", sources=None,
        episodes_per_epoch=cfg["episodes_per_epoch_s1"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], neg_prob=cfg["neg_prob_s1"],
        hard_neg_ratio=cfg["hard_neg_ratio_s1"], augment=cfg["augment"],
        augment_strength=cfg["augment_strength"], img_size=cfg["img_size"],
        seed=cfg["seed"], n_support=cfg["n_support"], source_mix=cfg["source_mix"],
    )
    val_ds, val_loader = build_val_loader(
        manifest=manifest, data_root=data_root, split="test", sources=None,
        val_episodes=cfg["val_episodes_s1"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], neg_prob=0.5, img_size=cfg["img_size"],
        seed=cfg["seed"] + 1, n_support=cfg["n_support"],
    )

    opt = build_optimizer_for_stage(1, model, cfg)
    steps_per_epoch = max(1, cfg["episodes_per_epoch_s1"] // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["stage1_epochs"]
    sched = build_scheduler(opt, total_steps=total_steps, warmup_frac=cfg["warmup_frac"])

    if start["stage"] == 1 and loaded_opt is not None and not rebuild:
        try:
            opt.load_state_dict(loaded_opt)
            if loaded_sched is not None:
                sched.load_state_dict(loaded_sched)
            restore_rng(loaded_rng)
        except Exception as e:                                                # noqa: BLE001
            print(f"  warning: failed to restore optimizer/scheduler ({e}); continuing fresh")

    start_epoch = start["epoch"] if start["stage"] == 1 else 1
    for epoch in range(start_epoch, cfg["stage1_epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_pass(
            model=model, optimizer=opt, scheduler=sched, loader=train_loader,
            device=device, cfg=cfg, multi_scale=False,
        )
        val_metrics = evaluate(
            model, val_loader, device,
            use_tta=cfg["val_use_tta"], tta_sizes=tuple(cfg["val_tta_sizes"]),
            img_size=cfg["img_size"],
        )
        epoch_payload = {
            "stage": 1, "epoch": epoch, "fold": None,
            "wall_clock_seconds": round(time.time() - t0, 2),
            "lr": {f"group_{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
            "train": train_metrics,
            "val": val_metrics,
        }
        write_json(analysis_dir / "stage1" / f"epoch_{epoch:03d}.json", epoch_payload)

        ckpt = {
            "stage": 1, "epoch": epoch, "fold": None,
            "stage_completed": (epoch == cfg["stage1_epochs"]),
            "global_step": sched.last_epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": capture_rng(),
            "config": cfg,
            "fold_plan": None,
        }
        atomic_save(ckpt, out_dir / f"ckpt_s1_epoch{epoch:03d}.pt")
        atomic_save(ckpt, out_dir / "last.pt")

        map50 = val_metrics.get("overall", {}).get("map_50", 0.0)
        if map50 > best_metric["value"]:
            best_metric = {"value": map50, "stage": 1, "epoch": epoch, "fold": 0}
            atomic_save(ckpt, out_dir / "best.pt")
        update_summary(analysis_dir, {
            "stage1.val.map_50": (epoch, map50),
            "stage1.val.iou_mean": (epoch, val_metrics.get("overall", {}).get("iou_mean", 0.0)),
        })
        hygiene(out_dir, cfg["keep_last_n"])
        print_epoch_log(
            header=f"s1 epoch {epoch}/{cfg['stage1_epochs']}",
            train_metrics=train_metrics, val_metrics=val_metrics,
            lr_groups={f"g{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
            wall_clock=epoch_payload["wall_clock_seconds"],
        )

    if cfg["save_stage_completion"]:
        ckpt = {
            "stage": 1, "epoch": cfg["stage1_epochs"], "fold": None,
            "stage_completed": True,
            "global_step": sched.last_epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": capture_rng(),
            "config": cfg,
            "fold_plan": None,
            "stage_metrics": {
                "best_val_map_50": best_metric["value"] if best_metric["stage"] == 1 else None,
                "best_epoch": best_metric["epoch"] if best_metric["stage"] == 1 else None,
            },
        }
        atomic_save(ckpt, out_dir / "stage1_complete.pt")
        write_json(analysis_dir / "stage1" / "complete.json", {
            "stage_completed": True,
            "epochs_run": cfg["stage1_epochs"],
            "best_val_map_50": best_metric["value"] if best_metric["stage"] == 1 else None,
        })

    del train_ds, val_ds, train_loader, val_loader, opt, sched
    return {"best_metric": best_metric, "loaded_opt": None,
            "loaded_sched": None, "loaded_rng": None, "rebuild": True}


def _run_stage2(
    *, model, cfg: dict, manifest: str, data_root: str | None,
    out_dir: Path, analysis_dir: Path, device: torch.device,
    start: dict, loaded_opt, loaded_sched, loaded_rng, rebuild: bool,
    best_metric: dict,
) -> dict:
    if cfg["stage2_epochs"] <= 0 or start["stage"] > 2:
        return {"best_metric": best_metric, "loaded_opt": None, "loaded_sched": None,
                "loaded_rng": None, "rebuild": True}

    print(f"\n=== Stage 2 (partial unfreeze, {cfg['stage2_epochs']} epochs) ===")
    train_ds, train_loader = build_train_loader(
        manifest=manifest, data_root=data_root, split="train", sources=None,
        episodes_per_epoch=cfg["episodes_per_epoch_s2"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], neg_prob=cfg["neg_prob_s2"],
        hard_neg_ratio=cfg["hard_neg_ratio_s2"], augment=cfg["augment"],
        augment_strength=cfg["augment_strength"], img_size=cfg["img_size"],
        seed=cfg["seed"] + 100, n_support=cfg["n_support"], source_mix=cfg["source_mix"],
    )
    val_ds, val_loader = build_val_loader(
        manifest=manifest, data_root=data_root, split="test", sources=None,
        val_episodes=cfg["val_episodes_s2"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], neg_prob=0.5, img_size=cfg["img_size"],
        seed=cfg["seed"] + 101, n_support=cfg["n_support"],
    )

    opt = build_optimizer_for_stage(2, model, cfg)
    steps_per_epoch = max(1, cfg["episodes_per_epoch_s2"] // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["stage2_epochs"]
    sched = build_scheduler(opt, total_steps=total_steps, warmup_frac=cfg["warmup_frac"])

    if start["stage"] == 2 and loaded_opt is not None and not rebuild:
        try:
            opt.load_state_dict(loaded_opt)
            if loaded_sched is not None:
                sched.load_state_dict(loaded_sched)
            restore_rng(loaded_rng)
        except Exception as e:                                                # noqa: BLE001
            print(f"  warning: failed to restore optimizer/scheduler ({e}); continuing fresh")

    start_epoch = start["epoch"] if start["stage"] == 2 else 1
    for epoch in range(start_epoch, cfg["stage2_epochs"] + 1):
        t0 = time.time()
        train_ds.hard_neg_cache = build_proto_cache(model, train_ds, device, batch_size=cfg["batch_size"])
        train_metrics = train_one_pass(
            model=model, optimizer=opt, scheduler=sched, loader=train_loader,
            device=device, cfg=cfg, multi_scale=cfg["multi_scale_s2"],
            multi_scale_sizes=tuple(cfg["multi_scale_sizes"]),
        )
        val_metrics = evaluate(
            model, val_loader, device,
            use_tta=cfg["val_use_tta"], tta_sizes=tuple(cfg["val_tta_sizes"]),
            img_size=cfg["img_size"],
        )
        epoch_payload = {
            "stage": 2, "epoch": epoch, "fold": None,
            "wall_clock_seconds": round(time.time() - t0, 2),
            "lr": {f"group_{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
            "train": train_metrics,
            "val": val_metrics,
        }
        write_json(analysis_dir / "stage2" / f"epoch_{epoch:03d}.json", epoch_payload)

        ckpt = {
            "stage": 2, "epoch": epoch, "fold": None,
            "stage_completed": (epoch == cfg["stage2_epochs"]),
            "global_step": sched.last_epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": capture_rng(),
            "config": cfg,
            "fold_plan": None,
        }
        atomic_save(ckpt, out_dir / f"ckpt_s2_epoch{epoch:03d}.pt")
        atomic_save(ckpt, out_dir / "last.pt")

        map50 = val_metrics.get("overall", {}).get("map_50", 0.0)
        if map50 > best_metric["value"]:
            best_metric = {"value": map50, "stage": 2, "epoch": epoch, "fold": 0}
            atomic_save(ckpt, out_dir / "best.pt")
        update_summary(analysis_dir, {
            "stage2.val.map_50": (epoch, map50),
            "stage2.val.iou_mean": (epoch, val_metrics.get("overall", {}).get("iou_mean", 0.0)),
        })
        hygiene(out_dir, cfg["keep_last_n"])
        print_epoch_log(
            header=f"s2 epoch {epoch}/{cfg['stage2_epochs']}",
            train_metrics=train_metrics, val_metrics=val_metrics,
            lr_groups={f"g{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
            wall_clock=epoch_payload["wall_clock_seconds"],
        )

    if cfg["save_stage_completion"]:
        ckpt = {
            "stage": 2, "epoch": cfg["stage2_epochs"], "fold": None,
            "stage_completed": True,
            "global_step": sched.last_epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": capture_rng(),
            "config": cfg,
            "fold_plan": None,
            "stage_metrics": {
                "best_val_map_50": best_metric["value"] if best_metric["stage"] == 2 else None,
                "best_epoch": best_metric["epoch"] if best_metric["stage"] == 2 else None,
            },
        }
        atomic_save(ckpt, out_dir / "stage2_complete.pt")
        write_json(analysis_dir / "stage2" / "complete.json", {
            "stage_completed": True,
            "epochs_run": cfg["stage2_epochs"],
            "best_val_map_50": best_metric["value"] if best_metric["stage"] == 2 else None,
        })

    del train_ds, val_ds, train_loader, val_loader, opt, sched
    return {"best_metric": best_metric, "loaded_opt": None,
            "loaded_sched": None, "loaded_rng": None, "rebuild": True}


def _run_stage3(
    *, model, cfg: dict, manifest: str, data_root: str | None,
    out_dir: Path, analysis_dir: Path, device: torch.device,
    start: dict, loaded_opt, loaded_sched, loaded_rng, rebuild: bool,
    best_metric: dict,
) -> dict:
    if cfg["stage3_epochs"] <= 0 or cfg["folds"] <= 0 or start["stage"] > 3:
        return {"best_metric": best_metric}

    print(
        f"\n=== Stage 3 (full unfreeze + K={cfg['folds']}-fold CV, "
        f"{cfg['stage3_epochs']} epochs) ==="
    )

    # Build / persist fold plan.
    with open(manifest) as f:
        manifest_obj = json.load(f)
    train_instances = [i for i in manifest_obj["instances"] if i.get("split") == "train"]
    fold_plan_path = analysis_dir / "folds.json"
    if fold_plan_path.exists():
        with open(fold_plan_path) as f:
            fold_plan_obj = json.load(f)
        fold_plan = fold_plan_obj["folds"]
        assert len(fold_plan) == cfg["folds"], "fold count mismatch with persisted folds.json"
    else:
        fold_plan = stratified_kfold(train_instances, k=cfg["folds"], seed=cfg["fold_seed"])
        write_json(fold_plan_path, {"k": cfg["folds"], "seed": cfg["fold_seed"], "folds": fold_plan})

    episodes_per_fold = max(1, cfg["episodes_per_epoch_s3"] // cfg["folds"])
    steps_per_fold = max(1, episodes_per_fold // cfg["batch_size"])
    total_steps = steps_per_fold * cfg["folds"] * cfg["stage3_epochs"]

    opt = build_optimizer_for_stage(3, model, cfg)
    sched = build_scheduler(opt, total_steps=total_steps, warmup_frac=cfg["warmup_frac"])

    if start["stage"] == 3 and loaded_opt is not None and not rebuild:
        try:
            opt.load_state_dict(loaded_opt)
            if loaded_sched is not None:
                sched.load_state_dict(loaded_sched)
            restore_rng(loaded_rng)
        except Exception as e:                                                # noqa: BLE001
            print(f"  warning: failed to restore optimizer/scheduler ({e}); continuing fresh")

    start_epoch = start["epoch"] if start["stage"] == 3 else 1
    start_fold = start["fold"] if start["stage"] == 3 else 0
    K = cfg["folds"]

    for epoch in range(start_epoch, cfg["stage3_epochs"] + 1):
        train_ds, _temp_loader = build_train_loader(
            manifest=manifest, data_root=data_root, split="train", sources=None,
            episodes_per_epoch=episodes_per_fold, batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"], neg_prob=cfg["neg_prob_s3"],
            hard_neg_ratio=cfg["hard_neg_ratio_s3"], augment=cfg["augment"],
            augment_strength=cfg["augment_strength"], img_size=cfg["img_size"],
            seed=cfg["seed"] + 200 + epoch, n_support=cfg["n_support"],
            source_mix=cfg["source_mix"],
        )
        del _temp_loader
        train_ds.set_fold()
        train_ds.hard_neg_cache = build_proto_cache(model, train_ds, device, batch_size=cfg["batch_size"])

        fold_jsons: list[dict] = []
        for fold_idx in range(K):
            if epoch == start_epoch and fold_idx < start_fold:
                continue
            fold = fold_plan[fold_idx]
            t0 = time.time()
            train_ids = set(fold["train_ids"])
            val_ids = set(fold["val_ids"])
            train_ds.set_fold(train_ids=train_ids)

            num_batches = max(1, episodes_per_fold // cfg["batch_size"])
            sampler = SourceBalancedBatchSampler(
                dataset=train_ds, batch_size=cfg["batch_size"],
                num_batches=num_batches, source_mix=cfg["source_mix"],
                seed=cfg["seed"] + 300 + epoch * K + fold_idx,
            )
            train_loader = DataLoader(
                train_ds, batch_sampler=sampler,
                num_workers=cfg["num_workers"], collate_fn=collate,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(cfg["num_workers"] > 0),
            )
            val_ds, val_loader = build_val_loader(
                manifest=manifest, data_root=data_root, split="train", sources=None,
                val_episodes=cfg["val_episodes_s3"], batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"], neg_prob=0.5, img_size=cfg["img_size"],
                seed=cfg["seed"] + 400 + epoch * K + fold_idx, n_support=cfg["n_support"],
            )
            val_ds.set_fold(val_ids=val_ids)

            train_metrics = train_one_pass(
                model=model, optimizer=opt, scheduler=sched, loader=train_loader,
                device=device, cfg=cfg, multi_scale=cfg["multi_scale_s3"],
                multi_scale_sizes=tuple(cfg["multi_scale_sizes"]),
            )
            val_metrics = evaluate(
                model, val_loader, device,
                use_tta=cfg["val_use_tta"], tta_sizes=tuple(cfg["val_tta_sizes"]),
                img_size=cfg["img_size"],
            )
            fold_payload = {
                "stage": 3, "epoch": epoch, "fold": fold_idx,
                "wall_clock_seconds": round(time.time() - t0, 2),
                "lr": {f"group_{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
                "train": train_metrics,
                "val": val_metrics,
            }
            write_json(
                analysis_dir / "stage3" / f"epoch_{epoch:03d}" / f"fold_{fold_idx}.json",
                fold_payload,
            )
            fold_jsons.append(fold_payload)

            ckpt = {
                "stage": 3, "epoch": epoch, "fold": fold_idx,
                "stage_completed": False,
                "global_step": sched.last_epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": capture_rng(),
                "config": cfg,
                "fold_plan": {"k": K, "seed": cfg["fold_seed"], "folds": fold_plan},
            }
            atomic_save(ckpt, out_dir / f"ckpt_s3_epoch{epoch:03d}_fold{fold_idx}.pt")
            atomic_save(ckpt, out_dir / "last.pt")
            hygiene(out_dir, cfg["keep_last_n"])
            print_epoch_log(
                header=f"s3 epoch {epoch}/{cfg['stage3_epochs']} fold {fold_idx}/{K - 1}",
                train_metrics=train_metrics, val_metrics=val_metrics,
                lr_groups={f"g{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
                wall_clock=fold_payload["wall_clock_seconds"],
            )
            del train_loader, val_loader, val_ds, sampler

        if len(fold_jsons) == K:
            aggregate = aggregate_folds(fold_jsons)
            write_json(
                analysis_dir / "stage3" / f"epoch_{epoch:03d}" / "aggregate.json",
                aggregate,
            )
            print_stage3_aggregate(epoch, aggregate)
            map50_mean = aggregate["metrics"].get("val.overall.map_50", {}).get("mean", 0.0)
            if map50_mean > best_metric["value"]:
                best_metric = {"value": map50_mean, "stage": 3, "epoch": epoch, "fold": K - 1}
                last_ckpt_path = out_dir / f"ckpt_s3_epoch{epoch:03d}_fold{K - 1}.pt"
                if last_ckpt_path.exists():
                    ck = torch.load(str(last_ckpt_path), map_location="cpu", weights_only=False)
                    atomic_save(ck, out_dir / "best.pt")
            update_summary(analysis_dir, {
                "stage3.val.map_50_mean": (epoch, map50_mean),
                "stage3.val.iou_mean_mean": (
                    epoch,
                    aggregate["metrics"].get("val.overall.iou_mean", {}).get("mean", 0.0),
                ),
            })

        start_fold = 0
        del train_ds

    if cfg["save_stage_completion"]:
        ckpt = {
            "stage": 3, "epoch": cfg["stage3_epochs"], "fold": K - 1,
            "stage_completed": True,
            "global_step": sched.last_epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": capture_rng(),
            "config": cfg,
            "fold_plan": {"k": K, "seed": cfg["fold_seed"], "folds": fold_plan},
            "stage_metrics": {
                "best_val_map_50_mean": best_metric["value"] if best_metric["stage"] == 3 else None,
                "best_epoch": best_metric["epoch"] if best_metric["stage"] == 3 else None,
            },
        }
        atomic_save(ckpt, out_dir / "stage3_complete.pt")
        write_json(analysis_dir / "stage3" / "complete.json", {
            "stage_completed": True,
            "epochs_run": cfg["stage3_epochs"],
            "folds": K,
            "best_val_map_50_mean": best_metric["value"] if best_metric["stage"] == 3 else None,
        })
    return {"best_metric": best_metric}


# ---------------------------------------------------------------------------
# Top-level train()
# ---------------------------------------------------------------------------


def train(
    manifest: str = "dataset/cleaned/manifest.json",
    data_root: str | None = None,
    out_dir: str = "model",
    analysis_dir: str = "analysis",
    # stage durations
    stage1_epochs: int = 5,
    stage2_epochs: int = 8,
    stage3_epochs: int = 35,
    # CV
    folds: int = 3,
    fold_seed: int = 42,
    # episodes / batches
    episodes_per_epoch_s1: int = 1500,
    episodes_per_epoch_s2: int = 1500,
    episodes_per_epoch_s3: int = 2000,
    val_episodes_s1: int = 400,
    val_episodes_s2: int = 400,
    val_episodes_s3: int = 240,
    batch_size: int = 16,
    num_workers: int = 2,
    n_support: int = 4,
    # LR
    lr_heads_s1: float = 3e-4,
    lr_heads_s2: float = 2e-4,
    lr_backbone_upper_s2: float = 5e-5,
    lr_heads_s3: float = 1e-4,
    lr_backbone_upper_s3: float = 5e-6,
    lr_backbone_lower_s3: float = 5e-6,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    warmup_frac: float = 0.05,
    # losses
    presence_weight: float = 1.0,
    attn_loss_weight: float = 0.5,
    aux_weight: float = 0.5,
    entropy_reg_weight: float = 0.01,
    reg_l2_prior_weight: float = 1e-3,
    proto_norm_weight: float = 1e-3,
    contrastive: bool = True,
    contrastive_weight: float = 0.1,
    contrastive_temp: float = 0.1,
    vicreg: bool = True,
    vicreg_weight: float = 0.05,
    barlow: bool = True,
    barlow_weight: float = 0.005,
    triplet: bool = False,
    triplet_weight: float = 0.1,
    # curriculum
    neg_prob_s1: float = 0.40,
    neg_prob_s2: float = 0.45,
    neg_prob_s3: float = 0.50,
    hard_neg_ratio_s1: float = 0.0,
    hard_neg_ratio_s2: float = 0.30,
    hard_neg_ratio_s3: float = 0.50,
    # sampler
    source_mix: dict[str, int] | None = None,
    # augmentation
    augment: bool = True,
    augment_strength: float = 1.0,
    multi_scale_s2: bool = True,
    multi_scale_s3: bool = True,
    multi_scale_sizes: tuple[int, ...] = (192, 224, 256),
    # checkpoints
    save_stage_completion: bool = True,
    keep_last_n: int = 6,
    # validation
    val_use_tta: bool = True,
    val_tta_sizes: tuple[int, ...] = (224, 288),
    # resume
    resume: bool | str = True,
    # misc
    seed: int = 42,
    device: str | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
) -> dict:
    """Run the three-stage training. Returns ``{"best_metric": ..., "config": cfg}``."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    out_dir_p = Path(out_dir)
    analysis_dir_p = Path(analysis_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    analysis_dir_p.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(locals())
    write_json(analysis_dir_p / "config.json", cfg)

    _set_seed(seed)
    model = FewShotLocalizer(pretrained=True).to(device_t)

    resume_path = resolve_resume_path(resume, out_dir_p)
    start = {"stage": 1, "epoch": 1, "fold": 0, "rebuild_optimizer": False, "done": False}
    loaded_opt = loaded_sched = loaded_rng = None
    rebuild = False
    if resume_path is not None:
        print(f"resuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device_t, weights_only=False)
        ok, err = try_load_state_dict(model, ckpt["model"])
        if not ok:
            quarantine_incompatible(
                out_dir_p,
                reason=(
                    f"checkpoint {resume_path.name} is not compatible with the current "
                    f"model architecture: {err}"
                ),
            )
            model = FewShotLocalizer(pretrained=True).to(device_t)
            _set_seed(seed)
        else:
            start = next_resume_point(ckpt, cfg)
            if start.get("done"):
                print("resume target already complete — nothing to train")
                return {"resumed_complete": True, "config": cfg}
            loaded_opt = ckpt.get("optimizer")
            loaded_sched = ckpt.get("scheduler")
            loaded_rng = ckpt.get("rng")
            rebuild = start["rebuild_optimizer"]

    best_metric: dict[str, Any] = {"value": -1.0, "stage": 0, "epoch": 0, "fold": 0}

    s1 = _run_stage1(
        model=model, cfg=cfg, manifest=manifest, data_root=data_root,
        out_dir=out_dir_p, analysis_dir=analysis_dir_p, device=device_t,
        start=start, loaded_opt=loaded_opt, loaded_sched=loaded_sched,
        loaded_rng=loaded_rng, rebuild=rebuild, best_metric=best_metric,
    )
    best_metric = s1["best_metric"]
    loaded_opt = s1.get("loaded_opt") if s1.get("loaded_opt") is not None else loaded_opt
    if start["stage"] == 1:
        loaded_opt = loaded_sched = loaded_rng = None
        rebuild = True

    s2 = _run_stage2(
        model=model, cfg=cfg, manifest=manifest, data_root=data_root,
        out_dir=out_dir_p, analysis_dir=analysis_dir_p, device=device_t,
        start=start, loaded_opt=loaded_opt, loaded_sched=loaded_sched,
        loaded_rng=loaded_rng, rebuild=rebuild, best_metric=best_metric,
    )
    best_metric = s2["best_metric"]
    if start["stage"] in (1, 2):
        loaded_opt = loaded_sched = loaded_rng = None
        rebuild = True

    s3 = _run_stage3(
        model=model, cfg=cfg, manifest=manifest, data_root=data_root,
        out_dir=out_dir_p, analysis_dir=analysis_dir_p, device=device_t,
        start=start, loaded_opt=loaded_opt, loaded_sched=loaded_sched,
        loaded_rng=loaded_rng, rebuild=rebuild, best_metric=best_metric,
    )
    best_metric = s3["best_metric"]

    return {"best_metric": best_metric, "config": cfg}


# ---------------------------------------------------------------------------
# Per-stage entrypoints (notebook-friendly — paired with evaluate cells)
# ---------------------------------------------------------------------------


def train_stage1(**kwargs) -> dict:
    """Run only Stage 1 (head warmup, no CV)."""
    kwargs.setdefault("stage1_epochs", 5)
    kwargs["stage2_epochs"] = 0
    kwargs["stage3_epochs"] = 0
    return train(**kwargs)


def train_stage2(**kwargs) -> dict:
    """Run only Stage 2 (partial unfreeze, no CV).

    Defaults ``resume='stage1_complete.pt'`` so the cell is self-sufficient.
    """
    kwargs["stage1_epochs"] = 0
    kwargs.setdefault("stage2_epochs", 8)
    kwargs["stage3_epochs"] = 0
    kwargs.setdefault("resume", "stage1_complete.pt")
    return train(**kwargs)


def train_stage3(**kwargs) -> dict:
    """Run only Stage 3 (full unfreeze + K-fold rotating CV).

    Defaults ``resume='stage2_complete.pt'`` so the cell is self-sufficient.
    """
    kwargs["stage1_epochs"] = 0
    kwargs["stage2_epochs"] = 0
    kwargs.setdefault("stage3_epochs", 35)
    kwargs.setdefault("folds", 3)
    kwargs.setdefault("resume", "stage2_complete.pt")
    return train(**kwargs)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="dataset/cleaned/manifest.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default="model")
    p.add_argument("--analysis-dir", default="analysis")
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--stage2-epochs", type=int, default=8)
    p.add_argument("--stage3-epochs", type=int, default=35)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--resume", default="true",
        help="'true' (auto), 'false', or a checkpoint filename / absolute path",
    )
    args = p.parse_args()

    if args.resume.lower() == "true":
        resume_arg: bool | str = True
    elif args.resume.lower() == "false":
        resume_arg = False
    else:
        resume_arg = args.resume

    train(
        manifest=args.manifest,
        data_root=args.data_root,
        out_dir=args.out_dir,
        analysis_dir=args.analysis_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        folds=args.folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        resume=resume_arg,
    )


if __name__ == "__main__":
    main()
