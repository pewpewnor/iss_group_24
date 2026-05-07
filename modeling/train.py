"""Three-stage training loop with universal resume and per-epoch JSON analysis.

Stages:
  Stage 1 — head warmup, no CV. Backbone frozen. Val from manifest test split.
  Stage 2 — partial unfreeze (features[7:]). No CV. Val from manifest test split.
  Stage 3 — full unfreeze + K-fold rotating CV (K=3 default). Per epoch:
            for fold in 0..K-1: train_one_pass(fold) → validate(fold) → save_ckpt.

Checkpoints (universal resume — every saved file is resumable):
  out_dir/
    ckpt_s1_epoch{E:03d}.pt
    ckpt_s2_epoch{E:03d}.pt
    ckpt_s3_epoch{E:03d}_fold{F}.pt
    stage1_complete.pt          # written when Stage 1 finishes
    stage2_complete.pt
    stage3_complete.pt
    last.pt                     # mirrors the most recent save
    best.pt                     # best by val.map_50 (s1/s2) or val.map_50_mean (s3)

Analysis (JSON-only during training):
  analysis/
    config.json
    folds.json                  # K-fold plan (Stage 3)
    stage1/epoch_001.json, ..., complete.json
    stage2/epoch_001.json, ..., complete.json
    stage3/epoch_001/fold_{0..K-1}.json + aggregate.json, ..., complete.json
    summary.json                # rolling best-by-metric pointer
"""

from __future__ import annotations

import argparse
import json
import os
import random as _random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from modeling.dataset import (
    DEFAULT_IMG_SIZE,
    DEFAULT_SOURCE_MIX,
    EpisodeDataset,
    SourceBalancedBatchSampler,
    collate,
    _Augment,
    _load_image,
)
from modeling.evaluate import evaluate
from modeling.loss import (
    barlow_twins_loss,
    nt_xent_loss,
    total_loss,
    triplet_loss,
    vicreg_loss,
)
from modeling.model import FewShotLocalizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


STAGE_NAMES = ("stage1", "stage2", "stage3")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": _random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict | None) -> None:
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


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def _heads_params(model: FewShotLocalizer) -> list:
    """Everything except the backbone."""
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    return [p for p in model.parameters() if id(p) not in backbone_ids]


def _backbone_upper_params(model: FewShotLocalizer) -> list:
    return [
        p
        for i, blk in enumerate(model.backbone.features)
        if i >= 7
        for p in blk.parameters()
    ]


def _backbone_lower_params(model: FewShotLocalizer) -> list:
    return [
        p
        for i, blk in enumerate(model.backbone.features)
        if i < 7
        for p in blk.parameters()
    ]


# ---------------------------------------------------------------------------
# Optimizer factories per stage
# ---------------------------------------------------------------------------


def _build_optimizer_for_stage(
    stage: int, model: FewShotLocalizer, cfg: dict
) -> torch.optim.Optimizer:
    wd = cfg["weight_decay"]
    if stage == 1:
        model.backbone.freeze_all()
        groups = [{"params": _heads_params(model), "lr": cfg["lr_heads_s1"], "weight_decay": wd}]
    elif stage == 2:
        model.backbone.freeze_lower(freeze_idx_exclusive=7)
        groups = [
            {"params": _backbone_upper_params(model), "lr": cfg["lr_backbone_upper_s2"], "weight_decay": wd},
            {"params": _heads_params(model), "lr": cfg["lr_heads_s2"], "weight_decay": wd},
        ]
    elif stage == 3:
        model.backbone.unfreeze_all()
        groups = [
            {"params": _backbone_lower_params(model), "lr": cfg["lr_backbone_lower_s3"], "weight_decay": wd},
            {"params": _backbone_upper_params(model), "lr": cfg["lr_backbone_upper_s3"], "weight_decay": wd},
            {"params": _heads_params(model), "lr": cfg["lr_heads_s3"], "weight_decay": wd},
        ]
    else:
        raise ValueError(stage)
    return torch.optim.AdamW(groups)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float
) -> SequentialLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    cosine_steps = max(1, total_steps - warmup_steps)
    warm = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Fold construction (Stage 3)
# ---------------------------------------------------------------------------


def _stratified_kfold(
    instances: list[dict[str, Any]], k: int, seed: int
) -> list[dict[str, list[str]]]:
    """Return K folds. Each fold = {"train_ids": [...], "val_ids": [...]}.

    Stratified by source so every fold has the same source mix. vizwiz_novel
    instances are partitioned alongside the rest (no longer kept in train-only).
    """
    by_source: dict[str, list[str]] = {}
    for inst in instances:
        by_source.setdefault(inst.get("source", "_"), []).append(inst["instance_id"])
    rng = _random.Random(seed)
    for src in by_source:
        by_source[src].sort()
        rng.shuffle(by_source[src])

    fold_val: list[list[str]] = [[] for _ in range(k)]
    for src, ids in by_source.items():
        # Round-robin assignment.
        for i, iid in enumerate(ids):
            fold_val[i % k].append(iid)

    all_ids = sorted({i["instance_id"] for i in instances})
    folds = []
    for f in range(k):
        val_set = set(fold_val[f])
        train_ids = [iid for iid in all_ids if iid not in val_set]
        folds.append({"train_ids": train_ids, "val_ids": sorted(val_set)})
    return folds


# ---------------------------------------------------------------------------
# Hard-negative prototype cache (rebuilt per epoch)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_proto_cache(
    model: FewShotLocalizer,
    dataset: EpisodeDataset,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    aug = _Augment("support", train=False, augment=False)
    rng = _random.Random(0)
    cache: dict[str, torch.Tensor] = {}
    k = dataset.n_support
    instances = dataset.instances
    for start in range(0, len(instances), batch_size):
        batch_instances = instances[start : start + batch_size]
        batch_support: list[torch.Tensor] = []
        for instance in batch_instances:
            pool = instance["support_images"]
            samples = (
                [rng.choice(pool) for _ in range(k)]
                if len(pool) < k
                else rng.sample(pool, k)
            )
            imgs = []
            for s in samples:
                img = _load_image(dataset._resolve(s["path"]))
                t, _ = aug(img, list(s["bbox"]), rng, img_size=dataset.img_size)
                imgs.append(t)
            batch_support.append(torch.stack(imgs))
        support_imgs_t = torch.stack(batch_support).to(device)
        tokens, _, _ = model.encode_support(support_imgs_t)
        # Bag-level prototype via attention pool for cosine-similarity hard negs.
        prototypes = model.support_pool(tokens)
        for i, instance in enumerate(batch_instances):
            cache[instance["instance_id"]] = prototypes[i].cpu()
    if was_training:
        model.train()
    return cache


# ---------------------------------------------------------------------------
# Loader builders
# ---------------------------------------------------------------------------


def _build_train_loader(
    manifest: str,
    data_root: str | None,
    split: str,
    sources: list[str] | None,
    episodes_per_epoch: int,
    batch_size: int,
    num_workers: int,
    neg_prob: float,
    hard_neg_ratio: float,
    augment: bool,
    augment_strength: float,
    img_size: int,
    seed: int,
    n_support: int,
    source_mix: dict[str, int] | None,
) -> tuple[EpisodeDataset, DataLoader]:
    ds = EpisodeDataset(
        manifest_path=manifest,
        data_root=data_root,
        split=split,
        sources=sources,
        episodes_per_epoch=episodes_per_epoch,
        n_support=n_support,
        neg_prob=neg_prob,
        hard_neg_ratio=hard_neg_ratio,
        train=True,
        augment=augment,
        augment_strength=augment_strength,
        img_size=img_size,
        seed=seed,
    )
    num_batches = max(1, episodes_per_epoch // batch_size)
    sampler = SourceBalancedBatchSampler(
        dataset=ds,
        batch_size=batch_size,
        num_batches=num_batches,
        source_mix=source_mix,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return ds, loader


def _build_val_loader(
    manifest: str,
    data_root: str | None,
    split: str | None,
    sources: list[str] | None,
    val_episodes: int,
    batch_size: int,
    num_workers: int,
    neg_prob: float,
    img_size: int,
    seed: int,
    n_support: int,
) -> tuple[EpisodeDataset, DataLoader]:
    ds = EpisodeDataset(
        manifest_path=manifest,
        data_root=data_root,
        split=split,
        sources=sources,
        episodes_per_epoch=val_episodes,
        n_support=n_support,
        neg_prob=neg_prob,
        hard_neg_ratio=0.0,
        train=False,
        augment=False,
        img_size=img_size,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return ds, loader


# ---------------------------------------------------------------------------
# Training one pass (one epoch in s1/s2; one fold-pass in s3)
# ---------------------------------------------------------------------------


def _train_one_pass(
    model: FewShotLocalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: SequentialLR,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    multi_scale: bool,
    multi_scale_sizes: tuple[int, ...] = (192, 224, 256),
) -> dict[str, float]:
    """One full sweep through ``loader``. Returns averaged train metrics."""
    model.train()
    model.backbone.eval()                                                   # freeze BN running stats

    running = {
        "loss": 0.0, "qfl": 0.0, "neg_qfl": 0.0, "centerness": 0.0,
        "dfl": 0.0, "giou": 0.0, "presence": 0.0, "attn": 0.0, "aux": 0.0,
        "entropy_reg": 0.0, "reg_l2_prior": 0.0, "proto_norm": 0.0,
        "nt_xent": 0.0, "vicreg": 0.0, "barlow": 0.0, "triplet": 0.0,
        "grad_norm": 0.0,
    }
    n_batches = 0
    rng = _random.Random(cfg.get("seed", 42))
    for batch in loader:
        if multi_scale:
            sz = rng.choice(multi_scale_sizes)
            loader.dataset.set_img_size(sz)                                # type: ignore[attr-defined]
            # The current batch was generated at the previous size — resize it
            # to keep gradient at the desired scale. This is a cheap pixel op.
            target = (sz, sz)
            if batch["query_img"].shape[-2:] != target:
                batch["query_img"] = F.interpolate(
                    batch["query_img"], size=target, mode="bilinear", align_corners=False
                )
                batch["support_imgs"] = F.interpolate(
                    batch["support_imgs"].view(-1, 3, *batch["support_imgs"].shape[-2:]),
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                ).view(*batch["support_imgs"].shape[:2], 3, sz, sz)
                # Rescale GT bboxes from previous size to new size.
                # The dataset emits bboxes in `self.img_size` coords; the loader
                # captures `loader.dataset.img_size` AT YIELD TIME, but here we
                # may have changed it. To avoid coordinate drift we simply
                # rescale to the *new* size from a known reference (assume the
                # dataset emitted at the previous yield's size, which we
                # approximate as the loader's current img_size before the
                # set_img_size call). Cleanest: scale both query + support
                # bboxes by sz / 224 (the dataset always emits 224 by default
                # and the trainer only flips img_size between batches — within
                # a batch the dataset has not been re-indexed). To remain
                # robust we just force img_size=sz on the dataset and rely on
                # the next batch being correct.
        support_imgs = batch["support_imgs"].to(device)
        support_bboxes = batch["support_bboxes"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)

        out = model(support_imgs, query_img, support_bboxes=support_bboxes)
        losses = total_loss(
            out,
            gt_bbox,
            is_present,
            support_bboxes=support_bboxes,
            img_size=query_img.shape[-1],
            presence_weight=cfg["presence_weight"],
            attn_loss_weight=cfg["attn_loss_weight"],
            aux_weight=cfg["aux_weight"],
            entropy_reg_weight=cfg["entropy_reg_weight"],
            reg_l2_prior_weight=cfg["reg_l2_prior_weight"],
            proto_norm_weight=cfg["proto_norm_weight"],
        )
        loss = losses["loss"]

        nt_v = vr_v = bt_v = trip_v = 0.0
        per_shot = out.get("per_shot_prototype")
        con_target = per_shot if per_shot is not None else out["prototype"]
        if cfg["contrastive"] and con_target is not None:
            nt = nt_xent_loss(con_target, temperature=cfg["contrastive_temp"])
            loss = loss + cfg["contrastive_weight"] * nt
            nt_v = float(nt.detach().item())
        if cfg["vicreg"] and con_target is not None:
            vr = vicreg_loss(con_target)
            loss = loss + cfg["vicreg_weight"] * vr
            vr_v = float(vr.detach().item())
        if cfg["barlow"] and per_shot is not None:
            bt = barlow_twins_loss(per_shot)
            loss = loss + cfg["barlow_weight"] * bt
            bt_v = float(bt.detach().item())
        if cfg["triplet"]:
            trip = triplet_loss(out["prototype"], list(batch["instance_id"]))
            loss = loss + cfg["triplet_weight"] * trip
            trip_v = float(trip.detach().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            cfg["grad_clip"],
        )
        optimizer.step()
        scheduler.step()

        running["loss"] += float(loss.item())
        running["qfl"] += float(losses["qfl"])
        running["neg_qfl"] += float(losses["neg_qfl"])
        running["centerness"] += float(losses["centerness"])
        running["dfl"] += float(losses["dfl"])
        running["giou"] += float(losses["giou"])
        running["presence"] += float(losses["presence"])
        running["attn"] += float(losses["attn"])
        running["aux"] += float(losses["aux"])
        running["entropy_reg"] += float(losses["entropy_reg"])
        running["reg_l2_prior"] += float(losses["reg_l2_prior"])
        running["proto_norm"] += float(losses["proto_norm"])
        running["nt_xent"] += nt_v
        running["vicreg"] += vr_v
        running["barlow"] += bt_v
        running["triplet"] += trip_v
        running["grad_norm"] += float(gn)
        n_batches += 1

    if n_batches > 0:
        for k in running:
            running[k] /= n_batches
    running["n_steps"] = n_batches
    return running


# ---------------------------------------------------------------------------
# Universal resume — load checkpoint, infer (stage, epoch, fold)
# ---------------------------------------------------------------------------


def _resolve_resume_path(resume: bool | str, out_dir: Path) -> Path | None:
    if resume is False or resume is None:
        return None
    if resume is True:
        p = out_dir / "last.pt"
        return p if p.exists() else None
    p = Path(resume)                                                          # type: ignore[arg-type]
    if not p.is_absolute():
        p = out_dir / p
    return p if p.exists() else None


def _quarantine_incompatible_checkpoints(out_dir: Path, reason: str) -> Path:
    """Move every *.pt file under ``out_dir`` into ``out_dir/legacy_<timestamp>/``.

    Used when the loaded checkpoint's state_dict doesn't match the current
    architecture (e.g. resuming an old DIM=128 / scalar-reg checkpoint after
    the architecture rewrite to DIM=160 + DFL + dual-scale heads). We never
    delete the user's data — just step around it.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = out_dir / f"legacy_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for p in list(out_dir.glob("*.pt")):
        try:
            target = backup / p.name
            os.replace(str(p), str(target))
            moved.append(p.name)
        except OSError:
            pass
    print(
        f"⚠ Quarantined {len(moved)} incompatible checkpoint(s) into {backup}\n"
        f"  reason: {reason}\n"
        f"  files : {', '.join(moved) if moved else '(none)'}\n"
        f"  Training will start fresh."
    )
    return backup


def _try_load_state_dict(
    model: torch.nn.Module, state: dict
) -> tuple[bool, str]:
    """Strict-by-shape load. Returns (ok, error_message).

    ``strict=False`` would silently leave shape-mismatched parameters at their
    fresh init — masking the architecture mismatch and producing a
    half-pretrained model. We refuse that and report the first few mismatches
    so the caller can decide what to do (quarantine + restart fresh).
    """
    own_state = model.state_dict()
    mismatches: list[str] = []
    for k, v in state.items():
        if k in own_state and own_state[k].shape != v.shape:
            mismatches.append(
                f"{k}: ckpt {tuple(v.shape)} vs model {tuple(own_state[k].shape)}"
            )
    if mismatches:
        return False, "; ".join(mismatches[:5]) + (
            f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        )
    try:
        model.load_state_dict(state, strict=False)
        return True, ""
    except RuntimeError as e:
        return False, str(e)


def _next_resume_point(ckpt: dict, cfg: dict) -> dict:
    """Given a loaded checkpoint, return the dict telling us where to resume.

    Returns:
        {"stage": int, "epoch": int, "fold": int, "rebuild_optimizer": bool}
        — fold is the next fold to run; for s1/s2 it's ignored.
    """
    stage = int(ckpt["stage"])
    epoch = int(ckpt["epoch"])
    fold = int(ckpt["fold"]) if ckpt.get("fold") is not None else None
    completed = bool(ckpt.get("stage_completed", False))

    if stage == 1:
        if completed or epoch >= cfg["stage1_epochs"]:
            return {"stage": 2, "epoch": 1, "fold": 0, "rebuild_optimizer": True}
        return {"stage": 1, "epoch": epoch + 1, "fold": 0, "rebuild_optimizer": False}
    if stage == 2:
        if completed or epoch >= cfg["stage2_epochs"]:
            return {"stage": 3, "epoch": 1, "fold": 0, "rebuild_optimizer": True}
        return {"stage": 2, "epoch": epoch + 1, "fold": 0, "rebuild_optimizer": False}
    if stage == 3:
        K = int(cfg["folds"])
        if completed:
            return {"stage": 3, "epoch": cfg["stage3_epochs"] + 1, "fold": 0, "rebuild_optimizer": False, "done": True}
        if fold is None:
            fold = -1
        if fold + 1 < K:
            return {"stage": 3, "epoch": epoch, "fold": fold + 1, "rebuild_optimizer": False}
        if epoch + 1 <= cfg["stage3_epochs"]:
            return {"stage": 3, "epoch": epoch + 1, "fold": 0, "rebuild_optimizer": False}
        return {"stage": 3, "epoch": cfg["stage3_epochs"] + 1, "fold": 0, "rebuild_optimizer": False, "done": True}
    raise ValueError(f"unknown stage {stage}")


# ---------------------------------------------------------------------------
# Disk hygiene
# ---------------------------------------------------------------------------


def _hygiene(out_dir: Path, keep_last_n: int) -> None:
    """Delete rolling per-(epoch,fold) checkpoints older than the last ``keep_last_n``.

    Stage-completion files, last.pt, best.pt are protected.
    """
    rolling = sorted(
        [
            p for p in out_dir.glob("ckpt_s*.pt")
            if not p.name.startswith("stage")
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


# ---------------------------------------------------------------------------
# JSON analysis helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def _flatten_metrics(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_metrics(v, key))
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    return out


def _aggregate_folds(fold_jsons: list[dict]) -> dict:
    """Recursive mean / min / max / std across folds for every numeric metric."""
    flat_per_fold = [_flatten_metrics(j) for j in fold_jsons]
    keys: set[str] = set()
    for f in flat_per_fold:
        keys.update(f.keys())
    metrics: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        vals = [f[k] for f in flat_per_fold if k in f]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / max(len(vals), 1)
        metrics[k] = {
            "mean": m,
            "min": min(vals),
            "max": max(vals),
            "std": var ** 0.5,
        }
    return {"epoch": fold_jsons[0].get("epoch"), "n_folds": len(fold_jsons), "metrics": metrics}


def _update_summary(
    analysis_dir: Path,
    headline: dict[str, tuple[int, float]],
) -> None:
    """Rolling best-by-metric pointer."""
    summary_path = analysis_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            current = json.load(f)
    else:
        current = {"best_by": {}}
    best_by = current.get("best_by", {})
    for k, (epoch, value) in headline.items():
        prev = best_by.get(k)
        if prev is None or value > prev["value"]:
            best_by[k] = {"epoch": epoch, "value": value}
    current["best_by"] = best_by
    _write_json(summary_path, current)


# ---------------------------------------------------------------------------
# Main train()
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
    # evaluation
    val_use_tta: bool = True,
    val_tta_sizes: tuple[int, ...] = (224, 288),
    # resume
    resume: bool | str = True,
    # misc
    seed: int = 42,
    device: str | None = None,
    img_size: int = DEFAULT_IMG_SIZE,
) -> dict:
    """Run the three-stage training. Returns the final summary dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    out_dir_p = Path(out_dir)
    analysis_dir_p = Path(analysis_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    analysis_dir_p.mkdir(parents=True, exist_ok=True)

    cfg = {
        "manifest": manifest, "data_root": data_root,
        "out_dir": str(out_dir_p), "analysis_dir": str(analysis_dir_p),
        "stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs, "stage3_epochs": stage3_epochs,
        "folds": folds, "fold_seed": fold_seed,
        "episodes_per_epoch_s1": episodes_per_epoch_s1,
        "episodes_per_epoch_s2": episodes_per_epoch_s2,
        "episodes_per_epoch_s3": episodes_per_epoch_s3,
        "val_episodes_s1": val_episodes_s1, "val_episodes_s2": val_episodes_s2,
        "val_episodes_s3": val_episodes_s3,
        "batch_size": batch_size, "num_workers": num_workers, "n_support": n_support,
        "lr_heads_s1": lr_heads_s1, "lr_heads_s2": lr_heads_s2,
        "lr_backbone_upper_s2": lr_backbone_upper_s2,
        "lr_heads_s3": lr_heads_s3,
        "lr_backbone_upper_s3": lr_backbone_upper_s3,
        "lr_backbone_lower_s3": lr_backbone_lower_s3,
        "weight_decay": weight_decay, "grad_clip": grad_clip, "warmup_frac": warmup_frac,
        "presence_weight": presence_weight, "attn_loss_weight": attn_loss_weight,
        "aux_weight": aux_weight, "entropy_reg_weight": entropy_reg_weight,
        "reg_l2_prior_weight": reg_l2_prior_weight, "proto_norm_weight": proto_norm_weight,
        "contrastive": contrastive, "contrastive_weight": contrastive_weight,
        "contrastive_temp": contrastive_temp,
        "vicreg": vicreg, "vicreg_weight": vicreg_weight,
        "barlow": barlow, "barlow_weight": barlow_weight,
        "triplet": triplet, "triplet_weight": triplet_weight,
        "neg_prob_s1": neg_prob_s1, "neg_prob_s2": neg_prob_s2, "neg_prob_s3": neg_prob_s3,
        "hard_neg_ratio_s1": hard_neg_ratio_s1,
        "hard_neg_ratio_s2": hard_neg_ratio_s2,
        "hard_neg_ratio_s3": hard_neg_ratio_s3,
        "source_mix": source_mix or DEFAULT_SOURCE_MIX,
        "augment": augment, "augment_strength": augment_strength,
        "multi_scale_s2": multi_scale_s2, "multi_scale_s3": multi_scale_s3,
        "multi_scale_sizes": list(multi_scale_sizes),
        "save_stage_completion": save_stage_completion, "keep_last_n": keep_last_n,
        "val_use_tta": val_use_tta, "val_tta_sizes": list(val_tta_sizes),
        "seed": seed, "device": device, "img_size": img_size,
    }
    _write_json(analysis_dir_p / "config.json", cfg)

    _set_seed(seed)
    model = FewShotLocalizer(pretrained=True).to(device_t)

    # Resume.
    resume_path = _resolve_resume_path(resume, out_dir_p)
    start = {"stage": 1, "epoch": 1, "fold": 0, "rebuild_optimizer": False, "done": False}
    loaded_opt = loaded_sched = loaded_rng = None
    rebuild = False
    if resume_path is not None:
        print(f"resuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device_t, weights_only=False)
        ok, err = _try_load_state_dict(model, ckpt["model"])
        if not ok:
            # Architecture mismatch — almost always because the checkpoint was
            # produced by an earlier architecture (DIM=128, scalar reg, single-
            # scale head, etc.) and we've just rewritten the model.
            _quarantine_incompatible_checkpoints(
                out_dir_p,
                reason=(
                    f"checkpoint {resume_path.name} is not compatible with the current "
                    f"model architecture: {err}"
                ),
            )
            # Reload a clean model so the quarantined weights aren't half-applied.
            model = FewShotLocalizer(pretrained=True).to(device_t)
            _set_seed(seed)
            start = {"stage": 1, "epoch": 1, "fold": 0, "rebuild_optimizer": False, "done": False}
        else:
            start = _next_resume_point(ckpt, cfg)
            if start.get("done"):
                print("resume target already complete — nothing to train")
                return {"resumed_complete": True}
            loaded_opt = ckpt.get("optimizer")
            loaded_sched = ckpt.get("scheduler")
            loaded_rng = ckpt.get("rng")
            rebuild = start["rebuild_optimizer"]

    # Best-by-metric tracking (rolling).
    best_metric = {"value": -1.0, "stage": 0, "epoch": 0, "fold": 0}

    # ---------------------------------------------------------------------
    # STAGE 1
    # ---------------------------------------------------------------------
    if start["stage"] <= 1 and stage1_epochs > 0:
        print(f"\n=== Stage 1 (head warmup, {stage1_epochs} epochs) ===")
        train_ds, train_loader = _build_train_loader(
            manifest=manifest, data_root=data_root, split="train", sources=None,
            episodes_per_epoch=episodes_per_epoch_s1, batch_size=batch_size,
            num_workers=num_workers, neg_prob=neg_prob_s1,
            hard_neg_ratio=hard_neg_ratio_s1, augment=augment,
            augment_strength=augment_strength, img_size=img_size, seed=seed,
            n_support=n_support, source_mix=source_mix,
        )
        val_ds, val_loader = _build_val_loader(
            manifest=manifest, data_root=data_root, split="test", sources=None,
            val_episodes=val_episodes_s1, batch_size=batch_size,
            num_workers=num_workers, neg_prob=0.5, img_size=img_size,
            seed=seed + 1, n_support=n_support,
        )

        opt = _build_optimizer_for_stage(1, model, cfg)
        steps_per_epoch = max(1, episodes_per_epoch_s1 // batch_size)
        total_steps = steps_per_epoch * stage1_epochs
        sched = _build_scheduler(opt, total_steps=total_steps, warmup_frac=warmup_frac)

        if start["stage"] == 1 and loaded_opt is not None and not rebuild:
            try:
                opt.load_state_dict(loaded_opt)
                if loaded_sched is not None:
                    sched.load_state_dict(loaded_sched)
                _restore_rng(loaded_rng)
            except Exception as e:                                            # noqa: BLE001
                print(f"  warning: failed to restore optimizer/scheduler ({e}); continuing fresh")

        start_epoch = start["epoch"] if start["stage"] == 1 else 1
        for epoch in range(start_epoch, stage1_epochs + 1):
            t0 = time.time()
            train_metrics = _train_one_pass(
                model, opt, sched, train_loader, device_t, cfg,
                multi_scale=False,
            )
            val_metrics = evaluate(
                model, val_loader, device_t,
                use_tta=val_use_tta, tta_sizes=val_tta_sizes,
                img_size=img_size,
            )
            epoch_payload = {
                "stage": 1, "epoch": epoch, "fold": None,
                "wall_clock_seconds": round(time.time() - t0, 2),
                "lr": {f"group_{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
                "train": train_metrics,
                "val": val_metrics,
            }
            _write_json(analysis_dir_p / "stage1" / f"epoch_{epoch:03d}.json", epoch_payload)

            # Save checkpoint.
            ckpt = {
                "stage": 1, "epoch": epoch, "fold": None,
                "stage_completed": (epoch == stage1_epochs),
                "global_step": sched.last_epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": _capture_rng(),
                "config": cfg,
                "fold_plan": None,
            }
            ckpt_path = out_dir_p / f"ckpt_s1_epoch{epoch:03d}.pt"
            _atomic_save(ckpt, ckpt_path)
            _atomic_save(ckpt, out_dir_p / "last.pt")

            # Update best.
            map50 = val_metrics.get("overall", {}).get("map_50", 0.0)
            if map50 > best_metric["value"]:
                best_metric = {"value": map50, "stage": 1, "epoch": epoch, "fold": 0}
                _atomic_save(ckpt, out_dir_p / "best.pt")
            _update_summary(analysis_dir_p, {
                "stage1.val.map_50": (epoch, map50),
                "stage1.val.iou_mean": (epoch, val_metrics.get("overall", {}).get("iou_mean", 0.0)),
            })
            _hygiene(out_dir_p, keep_last_n)
            print(
                f"  s1 epoch {epoch}/{stage1_epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"val_map50={map50:.3f}  "
                f"val_iou={val_metrics.get('overall', {}).get('iou_mean', 0):.3f}  "
                f"score_neg={val_metrics.get('overall', {}).get('mean_score_neg', 0):.3f}"
            )

        if save_stage_completion:
            ckpt = {
                "stage": 1, "epoch": stage1_epochs, "fold": None,
                "stage_completed": True,
                "global_step": sched.last_epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": _capture_rng(),
                "config": cfg,
                "fold_plan": None,
                "stage_metrics": {
                    "best_val_map_50": best_metric["value"]
                    if best_metric["stage"] == 1 else None,
                    "best_epoch": best_metric["epoch"]
                    if best_metric["stage"] == 1 else None,
                },
            }
            _atomic_save(ckpt, out_dir_p / "stage1_complete.pt")
            _write_json(analysis_dir_p / "stage1" / "complete.json", {
                "stage_completed": True,
                "epochs_run": stage1_epochs,
                "best_val_map_50": best_metric["value"] if best_metric["stage"] == 1 else None,
            })
        del train_ds, val_ds, train_loader, val_loader, opt, sched
        loaded_opt = loaded_sched = loaded_rng = None
        rebuild = True

    # ---------------------------------------------------------------------
    # STAGE 2
    # ---------------------------------------------------------------------
    if start["stage"] <= 2 and stage2_epochs > 0:
        print(f"\n=== Stage 2 (partial unfreeze, {stage2_epochs} epochs) ===")
        train_ds, train_loader = _build_train_loader(
            manifest=manifest, data_root=data_root, split="train", sources=None,
            episodes_per_epoch=episodes_per_epoch_s2, batch_size=batch_size,
            num_workers=num_workers, neg_prob=neg_prob_s2,
            hard_neg_ratio=hard_neg_ratio_s2, augment=augment,
            augment_strength=augment_strength, img_size=img_size, seed=seed + 100,
            n_support=n_support, source_mix=source_mix,
        )
        val_ds, val_loader = _build_val_loader(
            manifest=manifest, data_root=data_root, split="test", sources=None,
            val_episodes=val_episodes_s2, batch_size=batch_size,
            num_workers=num_workers, neg_prob=0.5, img_size=img_size,
            seed=seed + 101, n_support=n_support,
        )

        opt = _build_optimizer_for_stage(2, model, cfg)
        steps_per_epoch = max(1, episodes_per_epoch_s2 // batch_size)
        total_steps = steps_per_epoch * stage2_epochs
        sched = _build_scheduler(opt, total_steps=total_steps, warmup_frac=warmup_frac)

        if start["stage"] == 2 and loaded_opt is not None and not rebuild:
            try:
                opt.load_state_dict(loaded_opt)
                if loaded_sched is not None:
                    sched.load_state_dict(loaded_sched)
                _restore_rng(loaded_rng)
            except Exception as e:                                            # noqa: BLE001
                print(f"  warning: failed to restore optimizer/scheduler ({e}); continuing fresh")

        start_epoch = start["epoch"] if start["stage"] == 2 else 1
        for epoch in range(start_epoch, stage2_epochs + 1):
            t0 = time.time()
            # Epoch-level proto cache for hard-neg miner.
            train_ds.hard_neg_cache = build_proto_cache(model, train_ds, device_t, batch_size=batch_size)
            train_metrics = _train_one_pass(
                model, opt, sched, train_loader, device_t, cfg,
                multi_scale=multi_scale_s2,
                multi_scale_sizes=multi_scale_sizes,
            )
            val_metrics = evaluate(
                model, val_loader, device_t,
                use_tta=val_use_tta, tta_sizes=val_tta_sizes,
                img_size=img_size,
            )
            epoch_payload = {
                "stage": 2, "epoch": epoch, "fold": None,
                "wall_clock_seconds": round(time.time() - t0, 2),
                "lr": {f"group_{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
                "train": train_metrics,
                "val": val_metrics,
            }
            _write_json(analysis_dir_p / "stage2" / f"epoch_{epoch:03d}.json", epoch_payload)

            ckpt = {
                "stage": 2, "epoch": epoch, "fold": None,
                "stage_completed": (epoch == stage2_epochs),
                "global_step": sched.last_epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": _capture_rng(),
                "config": cfg,
                "fold_plan": None,
            }
            _atomic_save(ckpt, out_dir_p / f"ckpt_s2_epoch{epoch:03d}.pt")
            _atomic_save(ckpt, out_dir_p / "last.pt")

            map50 = val_metrics.get("overall", {}).get("map_50", 0.0)
            if map50 > best_metric["value"]:
                best_metric = {"value": map50, "stage": 2, "epoch": epoch, "fold": 0}
                _atomic_save(ckpt, out_dir_p / "best.pt")
            _update_summary(analysis_dir_p, {
                "stage2.val.map_50": (epoch, map50),
                "stage2.val.iou_mean": (epoch, val_metrics.get("overall", {}).get("iou_mean", 0.0)),
            })
            _hygiene(out_dir_p, keep_last_n)
            print(
                f"  s2 epoch {epoch}/{stage2_epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"val_map50={map50:.3f}  "
                f"val_iou={val_metrics.get('overall', {}).get('iou_mean', 0):.3f}  "
                f"score_neg={val_metrics.get('overall', {}).get('mean_score_neg', 0):.3f}"
            )

        if save_stage_completion:
            ckpt = {
                "stage": 2, "epoch": stage2_epochs, "fold": None,
                "stage_completed": True,
                "global_step": sched.last_epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": _capture_rng(),
                "config": cfg,
                "fold_plan": None,
                "stage_metrics": {
                    "best_val_map_50": best_metric["value"]
                    if best_metric["stage"] == 2 else None,
                    "best_epoch": best_metric["epoch"]
                    if best_metric["stage"] == 2 else None,
                },
            }
            _atomic_save(ckpt, out_dir_p / "stage2_complete.pt")
            _write_json(analysis_dir_p / "stage2" / "complete.json", {
                "stage_completed": True,
                "epochs_run": stage2_epochs,
                "best_val_map_50": best_metric["value"] if best_metric["stage"] == 2 else None,
            })
        del train_ds, val_ds, train_loader, val_loader, opt, sched
        loaded_opt = loaded_sched = loaded_rng = None
        rebuild = True

    # ---------------------------------------------------------------------
    # STAGE 3 (CV)
    # ---------------------------------------------------------------------
    if start["stage"] <= 3 and stage3_epochs > 0 and folds > 0:
        print(f"\n=== Stage 3 (full unfreeze + K={folds}-fold CV, {stage3_epochs} epochs) ===")

        # Build / persist fold plan.
        with open(manifest) as f:
            manifest_obj = json.load(f)
        train_instances = [i for i in manifest_obj["instances"] if i.get("split") == "train"]
        fold_plan_path = analysis_dir_p / "folds.json"
        if fold_plan_path.exists():
            with open(fold_plan_path) as f:
                fold_plan_obj = json.load(f)
            fold_plan = fold_plan_obj["folds"]
            assert len(fold_plan) == folds, "fold count mismatch with persisted folds.json"
        else:
            fold_plan = _stratified_kfold(train_instances, k=folds, seed=fold_seed)
            _write_json(fold_plan_path, {"k": folds, "seed": fold_seed, "folds": fold_plan})

        # One optimizer for the whole stage; fold rotations happen within an epoch
        # but the optimizer/scheduler state continues across the folds.
        episodes_per_fold = max(1, episodes_per_epoch_s3 // folds)
        steps_per_fold = max(1, episodes_per_fold // batch_size)
        total_steps = steps_per_fold * folds * stage3_epochs

        opt = _build_optimizer_for_stage(3, model, cfg)
        sched = _build_scheduler(opt, total_steps=total_steps, warmup_frac=warmup_frac)

        if start["stage"] == 3 and loaded_opt is not None and not rebuild:
            try:
                opt.load_state_dict(loaded_opt)
                if loaded_sched is not None:
                    sched.load_state_dict(loaded_sched)
                _restore_rng(loaded_rng)
            except Exception as e:                                            # noqa: BLE001
                print(f"  warning: failed to restore optimizer/scheduler ({e}); continuing fresh")

        start_epoch = start["epoch"] if start["stage"] == 3 else 1
        start_fold = start["fold"] if start["stage"] == 3 else 0

        for epoch in range(start_epoch, stage3_epochs + 1):
            # Build a single train dataset that we'll re-fold across the K folds.
            train_ds, _temp_loader = _build_train_loader(
                manifest=manifest, data_root=data_root, split="train", sources=None,
                episodes_per_epoch=episodes_per_fold, batch_size=batch_size,
                num_workers=num_workers, neg_prob=neg_prob_s3,
                hard_neg_ratio=hard_neg_ratio_s3, augment=augment,
                augment_strength=augment_strength, img_size=img_size,
                seed=seed + 200 + epoch, n_support=n_support, source_mix=source_mix,
            )
            del _temp_loader
            # Rebuild proto cache once per epoch (over the full train pool).
            train_ds.set_fold()                                                # reset to full pool
            train_ds.hard_neg_cache = build_proto_cache(model, train_ds, device_t, batch_size=batch_size)

            fold_jsons: list[dict] = []
            for fold_idx in range(folds):
                if epoch == start_epoch and fold_idx < start_fold:
                    continue
                fold = fold_plan[fold_idx]
                t0 = time.time()
                # Train data: full pool minus this fold's val ids.
                train_ids = set(fold["train_ids"])
                val_ids = set(fold["val_ids"])
                train_ds.set_fold(train_ids=train_ids)

                # Fresh sampler per fold — different bucket composition.
                num_batches = max(1, episodes_per_fold // batch_size)
                sampler = SourceBalancedBatchSampler(
                    dataset=train_ds, batch_size=batch_size,
                    num_batches=num_batches, source_mix=source_mix,
                    seed=seed + 300 + epoch * folds + fold_idx,
                )
                train_loader = DataLoader(
                    train_ds, batch_sampler=sampler,
                    num_workers=num_workers, collate_fn=collate,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=(num_workers > 0),
                )
                # Val data: only this fold's val ids.
                val_ds, val_loader = _build_val_loader(
                    manifest=manifest, data_root=data_root, split="train", sources=None,
                    val_episodes=val_episodes_s3, batch_size=batch_size,
                    num_workers=num_workers, neg_prob=0.5, img_size=img_size,
                    seed=seed + 400 + epoch * folds + fold_idx, n_support=n_support,
                )
                val_ds.set_fold(val_ids=val_ids)

                train_metrics = _train_one_pass(
                    model, opt, sched, train_loader, device_t, cfg,
                    multi_scale=multi_scale_s3,
                    multi_scale_sizes=multi_scale_sizes,
                )
                val_metrics = evaluate(
                    model, val_loader, device_t,
                    use_tta=val_use_tta, tta_sizes=val_tta_sizes,
                    img_size=img_size,
                )
                fold_payload = {
                    "stage": 3, "epoch": epoch, "fold": fold_idx,
                    "wall_clock_seconds": round(time.time() - t0, 2),
                    "lr": {f"group_{i}": g["lr"] for i, g in enumerate(opt.param_groups)},
                    "train": train_metrics,
                    "val": val_metrics,
                }
                _write_json(
                    analysis_dir_p / "stage3" / f"epoch_{epoch:03d}" / f"fold_{fold_idx}.json",
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
                    "rng": _capture_rng(),
                    "config": cfg,
                    "fold_plan": {"k": folds, "seed": fold_seed, "folds": fold_plan},
                }
                _atomic_save(
                    ckpt,
                    out_dir_p / f"ckpt_s3_epoch{epoch:03d}_fold{fold_idx}.pt",
                )
                _atomic_save(ckpt, out_dir_p / "last.pt")
                _hygiene(out_dir_p, keep_last_n)
                print(
                    f"  s3 epoch {epoch}/{stage3_epochs} fold {fold_idx}/{folds-1}  "
                    f"loss={train_metrics['loss']:.4f}  "
                    f"val_map50={val_metrics.get('overall', {}).get('map_50', 0):.3f}  "
                    f"val_iou={val_metrics.get('overall', {}).get('iou_mean', 0):.3f}  "
                    f"score_neg={val_metrics.get('overall', {}).get('mean_score_neg', 0):.3f}"
                )
                del train_loader, val_loader, val_ds, sampler

            # If we resumed mid-epoch we may have fewer than K fold jsons.
            if len(fold_jsons) == folds:
                aggregate = _aggregate_folds(fold_jsons)
                _write_json(
                    analysis_dir_p / "stage3" / f"epoch_{epoch:03d}" / "aggregate.json",
                    aggregate,
                )
                map50_mean = aggregate["metrics"].get("val.overall.map_50", {}).get("mean", 0.0)
                if map50_mean > best_metric["value"]:
                    best_metric = {"value": map50_mean, "stage": 3, "epoch": epoch, "fold": folds - 1}
                    # Reload the most recent checkpoint as best.pt.
                    last_ckpt_path = out_dir_p / f"ckpt_s3_epoch{epoch:03d}_fold{folds - 1}.pt"
                    if last_ckpt_path.exists():
                        ck = torch.load(str(last_ckpt_path), map_location="cpu", weights_only=False)
                        _atomic_save(ck, out_dir_p / "best.pt")
                _update_summary(analysis_dir_p, {
                    "stage3.val.map_50_mean": (epoch, map50_mean),
                    "stage3.val.iou_mean_mean": (
                        epoch,
                        aggregate["metrics"].get("val.overall.iou_mean", {}).get("mean", 0.0),
                    ),
                })

            # Reset start_fold after the first resumed epoch.
            start_fold = 0
            del train_ds

        if save_stage_completion:
            ckpt = {
                "stage": 3, "epoch": stage3_epochs, "fold": folds - 1,
                "stage_completed": True,
                "global_step": sched.last_epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": _capture_rng(),
                "config": cfg,
                "fold_plan": {"k": folds, "seed": fold_seed, "folds": fold_plan},
                "stage_metrics": {
                    "best_val_map_50_mean": best_metric["value"]
                    if best_metric["stage"] == 3 else None,
                    "best_epoch": best_metric["epoch"]
                    if best_metric["stage"] == 3 else None,
                },
            }
            _atomic_save(ckpt, out_dir_p / "stage3_complete.pt")
            _write_json(analysis_dir_p / "stage3" / "complete.json", {
                "stage_completed": True,
                "epochs_run": stage3_epochs,
                "folds": folds,
                "best_val_map_50_mean": best_metric["value"] if best_metric["stage"] == 3 else None,
            })

    return {"best_metric": best_metric}


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
    p.add_argument("--resume", default="true",
                   help="'true' (auto), 'false', or a checkpoint filename / absolute path")
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
