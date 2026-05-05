"""Five-stage episodic training entrypoint.

Phase 1 — VizWiz base pretraining (100 categories, rotation-synthesis episodes):
  Stage 1.1:  backbone fully frozen, heads LR 1e-3.          Val: VizWiz base val split.
  Stage 1.2:  features[7:] @ 1e-5, heads LR 5e-4.            Val: VizWiz base val split.

Phase 2 — target-domain fine-tuning (VizWiz novel + HOTS + InsDet):
  Stage 2.1:  backbone fully re-frozen, heads LR 5e-4.       Val: HOTS+InsDet val split.
  Stage 2.2:  features[7:] @ 1e-5, heads LR 5e-4.            Val: HOTS+InsDet val split.
  Stage 2.3:  full unfreeze, all @ 5e-6, heads LR 1e-4.       Val: HOTS+InsDet val split.

Backbone is frozen for two full warmup stages (1.1, 2.1) before any gradient flows
through MobileNetV3. This prevents catastrophic forgetting at the Phase 1→2 boundary.

Common workflows:
    # All five stages from scratch
    python -m modeling.train --no-resume

    # Skip Phase 1 (already pretrained), run Phase 2 from a checkpoint
    python -m modeling.train --resume model/stage1_2.pt --start-stage stage2_1

    # Quick smoke test (1 epoch each, 32 episodes)
    python -m modeling.train --no-resume \\
        --phase1-frozen-epochs 1 --phase1-partial-epochs 1 \\
        --phase2-frozen-epochs 1 --phase2-partial-epochs 1 \\
        --phase2-full-epochs 1 --episodes-per-epoch 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import random as _random

from modeling.dataset import EpisodeDataset, _Augment, _load_image, collate
from modeling.evaluate import IOU_THRESHOLDS, _compute_pr_ap, _iou_xyxy, run as evaluate_run
from modeling.loss import _containment_ratio
from modeling.loss import nt_xent_loss, total_loss, triplet_loss, vicreg_loss
from modeling.model import FewShotLocalizer, decode


# ---------------------------------------------------------------------------
# Data source groups
# ---------------------------------------------------------------------------

VIZWIZ_BASE_SOURCES = ["vizwiz_base"]
PHASE2_SOURCES = ["vizwiz_novel", "hots", "insdet"]
PHASE1_VAL_SOURCES = ["vizwiz_base"]
PHASE2_VAL_SOURCES = ["hots", "insdet"]


# ---------------------------------------------------------------------------
# Parameter group helpers
# ---------------------------------------------------------------------------


def _heads(model: FewShotLocalizer) -> list:
    return (
        list(model.support_tokenizer.parameters())
        + list(model.fpn.parameters())
        + list(model.p3_lat.parameters())
        + list(model.cross_attn.parameters())
        + list(model.det_head.parameters())
        + list(model.presence_head.parameters())
    )


def phase1_frozen_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 1.1: backbone fully frozen, heads at 1e-3."""
    model.backbone.freeze_all()
    return torch.optim.AdamW(
        [{"params": _heads(model), "lr": 1e-3, "weight_decay": 1e-4}]
    )


def phase1_partial_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 1.2: features[7:] @ 1e-5, heads at 5e-4."""
    model.backbone.freeze_lower(freeze_idx_exclusive=7)
    upper = [
        p
        for i, blk in enumerate(model.backbone.features)
        if i >= 7
        for p in blk.parameters()
    ]
    return torch.optim.AdamW(
        [
            {"params": upper, "lr": 1e-5, "weight_decay": 1e-4},
            {"params": _heads(model), "lr": 5e-4, "weight_decay": 1e-4},
        ]
    )


def phase2_frozen_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 2.1: re-freeze backbone, heads at 5e-4 (warmed-up heads, lower LR)."""
    model.backbone.freeze_all()
    return torch.optim.AdamW(
        [{"params": _heads(model), "lr": 5e-4, "weight_decay": 1e-4}]
    )


def phase2_partial_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 2.2: features[7:] @ 1e-5, heads at 5e-4."""
    model.backbone.freeze_lower(freeze_idx_exclusive=7)
    upper = [
        p
        for i, blk in enumerate(model.backbone.features)
        if i >= 7
        for p in blk.parameters()
    ]
    return torch.optim.AdamW(
        [
            {"params": upper, "lr": 1e-5, "weight_decay": 1e-4},
            {"params": _heads(model), "lr": 5e-4, "weight_decay": 1e-4},
        ]
    )


def phase2_full_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 2.3: full unfreeze. Lower layers at 5e-6 to preserve ImageNet features."""
    model.backbone.unfreeze_all()
    lower = [
        p
        for i, blk in enumerate(model.backbone.features)
        if i < 7
        for p in blk.parameters()
    ]
    upper = [
        p
        for i, blk in enumerate(model.backbone.features)
        if i >= 7
        for p in blk.parameters()
    ]
    return torch.optim.AdamW(
        [
            {"params": lower, "lr": 5e-6, "weight_decay": 1e-4},
            {"params": upper, "lr": 5e-6, "weight_decay": 1e-4},
            {"params": _heads(model), "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )


# ---------------------------------------------------------------------------
# Prototype cache for hard-negative mining
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_proto_cache(
    model: FewShotLocalizer,
    dataset: EpisodeDataset,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute one prototype per instance (eval augmentation, fixed seed).

    Called once per epoch before the training loop so _sample_query can
    find hard negatives by cosine similarity between prototype vectors.
    """
    was_training = model.training
    model.eval()

    aug = _Augment("support", train=False)
    rng = _random.Random(0)
    cache: dict[str, torch.Tensor] = {}

    for instance in dataset.instances:
        pool = instance["support_images"]
        k = dataset.n_support
        samples = [rng.choice(pool) for _ in range(k)] if len(pool) < k else rng.sample(pool, k)

        imgs = []
        for s in samples:
            img = _load_image(dataset._resolve(s["path"]))
            t, _ = aug(img, list(s["bbox"]), rng)
            imgs.append(t)

        support_imgs_t = torch.stack(imgs).unsqueeze(0).to(device)
        tokens, _ = model.encode_support(support_imgs_t)
        cache[instance["instance_id"]] = tokens.mean(dim=1).squeeze(0).cpu()

    if was_training:
        model.train()
        model.backbone.eval()

    return cache


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


STAGE_ORDER: tuple[str, ...] = ("stage1_1", "stage1_2", "stage2_1", "stage2_2", "stage2_3")


def _make_full_ckpt(
    model: FewShotLocalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    stage_name: str,
    epoch: int,
    stage_epochs: int,
    best_val_iou: float,
    full_history: list[dict],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stage": stage_name,
        "epoch": epoch,
        "stage_epochs": stage_epochs,
        "best_val_iou": best_val_iou,
        "full_history": full_history,
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "python": _random.getstate(),
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: FewShotLocalizer,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses, ious, contain, present_correct, n, n_pos = 0.0, 0.0, 0.0, 0, 0, 0
    all_scores: list[float] = []
    all_ious: list[float] = []
    all_present: list[bool] = []
    for batch in val_loader:
        support_imgs = batch["support_imgs"].to(device)
        support_bboxes = batch["support_bboxes"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)

        out = model(support_imgs, query_img, support_bboxes=support_bboxes)
        loss_dict = total_loss(out, gt_bbox, is_present, support_bboxes=support_bboxes)
        losses += loss_dict["loss"].item() * gt_bbox.shape[0]

        pred_box, pred_score = decode(out["reg"], out["conf"], presence_logit=out.get("presence_logit"))
        pred_present = pred_score > 0.5
        present_correct += (pred_present == is_present).sum().item()

        plain_ious = _iou_xyxy(pred_box, gt_bbox)
        for i in range(gt_bbox.shape[0]):
            present = bool(is_present[i].item())
            iou_v = float(plain_ious[i].item()) if present else 0.0
            all_scores.append(float(pred_score[i].item()))
            all_ious.append(iou_v)
            all_present.append(present)

        pos = is_present
        if pos.any():
            iou_vals = _iou_xyxy(pred_box[pos], gt_bbox[pos])
            ious += iou_vals.clamp(min=0, max=1).sum().item()
            contain_vals = _containment_ratio(pred_box[pos], gt_bbox[pos])
            contain += contain_vals.clamp(min=0, max=1).sum().item()
            n_pos += int(pos.sum().item())
        n += gt_bbox.shape[0]

    ap_vals = [
        _compute_pr_ap(
            all_scores,
            [p and iou_v >= tau for p, iou_v in zip(all_present, all_ious)],
            n_pos,
        )
        for tau in IOU_THRESHOLDS
    ]
    val_map_50 = ap_vals[0] if ap_vals else 0.0
    val_map_5095 = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0
    val_ap_per_iou = {f"{tau:.2f}": ap for tau, ap in zip(IOU_THRESHOLDS, ap_vals)}

    score_thr, iou_thr = 0.5, 0.5
    tp = fp = fn = 0
    for present, iou_v, score in zip(all_present, all_ious, all_scores):
        pred_present = score >= score_thr
        if present and pred_present and iou_v >= iou_thr:
            tp += 1
        elif pred_present:
            fp += 1
        elif present:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    val_f1_50 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    model.train()
    return {
        "val_loss": losses / max(n, 1),
        "val_iou": ious / max(n_pos, 1),
        "val_contain": contain / max(n_pos, 1),
        "val_map_50": val_map_50,
        "val_map_5095": val_map_5095,
        "val_ap_per_iou": val_ap_per_iou,
        "val_f1_50": val_f1_50,
        "val_precision_50": precision,
        "val_recall_50": recall,
        "val_presence_acc": present_correct / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Training loop (single stage)
# ---------------------------------------------------------------------------


def train_stage(
    model: FewShotLocalizer,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    stage_name: str,
    out_dir: Path,
    best_val_iou: float,
    grad_clip: float = 1.0,
    contrastive: bool = True,
    contrastive_weight: float = 0.1,
    contrastive_temp: float = 0.1,
    vicreg: bool = True,
    vicreg_weight: float = 0.05,
    triplet: bool = False,
    triplet_weight: float = 0.05,
    hard_neg_mining: bool = True,
    neg_prob_schedule: dict[int, float] | None = None,
    attn_loss_weight: float = 0.5,
    start_epoch: int = 1,
    scheduler_state: dict | None = None,
    prior_history: list[dict] | None = None,
) -> tuple[float, list[dict]]:
    total_steps = max(epochs * len(train_loader), 1)
    warmup_steps = max(1, min(200, total_steps // 20))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    prior = list(prior_history) if prior_history else []
    epoch_history: list[dict] = []

    for epoch in range(start_epoch, epochs + 1):
        if neg_prob_schedule and hasattr(train_loader.dataset, "set_neg_prob"):
            applicable = [(e, p) for e, p in neg_prob_schedule.items() if e <= epoch]
            if applicable:
                _, prob = max(applicable)
                train_loader.dataset.set_neg_prob(prob)  # type: ignore[attr-defined]

        if hard_neg_mining and hasattr(train_loader.dataset, "hard_neg_cache"):
            train_loader.dataset.hard_neg_cache = build_proto_cache(  # type: ignore[attr-defined]
                model, train_loader.dataset, device  # type: ignore[arg-type]
            )

        model.train()
        model.backbone.eval()

        t0 = time.time()
        running = {
            "loss": 0.0, "focal": 0.0, "box": 0.0,
            "presence": 0.0, "attn": 0.0,
            "nt_xent": 0.0, "vicreg": 0.0, "triplet": 0.0,
        }
        for batch in train_loader:
            support_imgs = batch["support_imgs"].to(device)
            support_bboxes = batch["support_bboxes"].to(device)
            query_img = batch["query_img"].to(device)
            gt_bbox = batch["query_bbox"].to(device)
            is_present = batch["is_present"].to(device)

            out = model(support_imgs, query_img, support_bboxes=support_bboxes)
            losses = total_loss(
                out, gt_bbox, is_present,
                support_bboxes=support_bboxes,
                attn_loss_weight=attn_loss_weight,
            )
            loss = losses["loss"]

            nt_val, vr_val, trip_val = 0.0, 0.0, 0.0
            if contrastive:
                nt = nt_xent_loss(out["prototype"], temperature=contrastive_temp)
                loss = loss + contrastive_weight * nt
                nt_val = nt.detach().item()
            if vicreg:
                vr = vicreg_loss(out["prototype"])
                loss = loss + vicreg_weight * vr
                vr_val = vr.detach().item()
            if triplet:
                trip = triplet_loss(out["prototype"], list(batch["instance_id"]))
                loss = loss + triplet_weight * trip
                trip_val = trip.detach().item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip
            )
            optimizer.step()
            scheduler.step()

            running["loss"] += loss.item()
            running["focal"] += float(losses["focal"])
            running["box"] += float(losses["box"])
            running["presence"] += float(losses.get("presence", 0.0))
            running["attn"] += float(losses.get("attn", 0.0))
            running["nt_xent"] += nt_val
            running["vicreg"] += vr_val
            running["triplet"] += trip_val

        nb = len(train_loader)
        avg = {k: v / max(nb, 1) for k, v in running.items()}
        metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0
        reg_str = ""
        if contrastive or vicreg:
            reg_str += f" nt_xent={avg['nt_xent']:.4f} vicreg={avg['vicreg']:.4f}"
        if triplet:
            reg_str += f" triplet={avg['triplet']:.4f}"
        print(
            f"[{stage_name}] epoch {epoch}/{epochs} "
            f"loss={avg['loss']:.4f} (focal={avg['focal']:.4f} box={avg['box']:.4f} "
            f"presence={avg['presence']:.4f} attn={avg['attn']:.4f}{reg_str}) "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_iou={metrics['val_iou']:.4f} "
            f"val_contain={metrics['val_contain']:.4f} "
            f"val_map50={metrics['val_map_50']:.4f} "
            f"val_map5095={metrics['val_map_5095']:.4f} "
            f"val_f1_50={metrics['val_f1_50']:.4f} "
            f"val_presence={metrics['val_presence_acc']:.4f} "
            f"time={elapsed:.1f}s"
        )

        epoch_history.append({
            "stage": stage_name,
            "epoch": epoch,
            "loss": avg["loss"],
            "focal": avg["focal"],
            "box": avg["box"],
            "presence": avg["presence"],
            "attn": avg["attn"],
            "nt_xent": avg["nt_xent"],
            "vicreg": avg["vicreg"],
            "triplet": avg["triplet"],
            "val_loss": metrics["val_loss"],
            "val_iou": metrics["val_iou"],
            "val_contain": metrics["val_contain"],
            "val_map_50": metrics["val_map_50"],
            "val_map_5095": metrics["val_map_5095"],
            "val_ap_per_iou": metrics["val_ap_per_iou"],
            "val_f1_50": metrics["val_f1_50"],
            "val_precision_50": metrics["val_precision_50"],
            "val_recall_50": metrics["val_recall_50"],
            "val_presence_acc": metrics["val_presence_acc"],
        })

        if metrics["val_iou"] > best_val_iou:
            best_val_iou = metrics["val_iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_iou": best_val_iou,
                    "val_contain": metrics["val_contain"],
                    "val_map_50": metrics["val_map_50"],
                    "val_map_5095": metrics["val_map_5095"],
                    "val_ap_per_iou": metrics["val_ap_per_iou"],
                    "val_f1_50": metrics["val_f1_50"],
                    "stage": stage_name,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )
            print(
                f"  saved best.pt (val_iou={best_val_iou:.4f} "
                f"val_map5095={metrics['val_map_5095']:.4f} "
                f"val_f1_50={metrics['val_f1_50']:.4f})"
            )

        torch.save(
            _make_full_ckpt(
                model, optimizer, scheduler, stage_name, epoch, epochs,
                best_val_iou, prior + epoch_history,
            ),
            out_dir / "last.pt",
        )
        print(f"  saved last.pt ({stage_name} epoch {epoch}/{epochs})")

    if epoch_history:
        stage_path = out_dir / f"{stage_name}.pt"
        last_metrics = epoch_history[-1]
        torch.save(
            {
                "model": model.state_dict(),
                "stage": stage_name,
                "epoch": last_metrics["epoch"],
                "val_iou": last_metrics["val_iou"],
                "val_contain": last_metrics["val_contain"],
                "val_map_50": last_metrics["val_map_50"],
                "val_map_5095": last_metrics["val_map_5095"],
                "val_ap_per_iou": last_metrics["val_ap_per_iou"],
                "val_f1_50": last_metrics["val_f1_50"],
            },
            stage_path,
        )
        print(f"  saved {stage_path.name} (final {stage_name} weights)")

    return best_val_iou, epoch_history


# ---------------------------------------------------------------------------
# Per-stage test evaluation
# ---------------------------------------------------------------------------


def _eval_stage(
    stage_name: str,
    out_dir: Path,
    analysis_dir: Path,
    manifest: str | Path,
    data_root: str | Path | None,
    val_episodes: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: str,
) -> None:
    ckpt = out_dir / f"{stage_name}.pt"
    if not ckpt.exists():
        return
    stage_dir = analysis_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Test evaluation: {stage_name} ===")
    evaluate_run(
        checkpoint=ckpt,
        manifest=manifest,
        split="test",
        data_root=data_root,
        episodes=val_episodes,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
        report=stage_dir / "test_report.json",
        analysis_dir=stage_dir,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def train(
    manifest: str | Path = "dataset/cleaned/manifest.json",
    train_split: str = "train",
    val_split: str = "val",
    data_root: str | Path | None = None,
    out_dir: str | Path = "model",
    resume: str | Path | bool | None = True,
    episodes_per_epoch: int = 2000,
    val_episodes: int = 500,
    batch_size: int = 16,
    num_workers: int = 4,
    # Phase 1: VizWiz base pretraining
    phase1_frozen_epochs: int = 15,
    phase1_partial_epochs: int = 30,
    # Phase 2: target-domain fine-tuning
    phase2_frozen_epochs: int = 15,
    phase2_partial_epochs: int = 50,
    phase2_full_epochs: int = 15,
    # Regularisation
    contrastive: bool = True,
    contrastive_weight_p1: float = 0.15,
    contrastive_weight_p2: float = 0.2,
    contrastive_temp: float = 0.1,
    vicreg: bool = True,
    vicreg_weight: float = 0.05,
    triplet: bool = False,
    triplet_weight: float = 0.05,
    hard_neg_mining: bool = True,
    start_stage: str | None = None,
    seed: int = 42,
    device: str | None = None,
    pretrained: bool = True,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "manifest": str(manifest),
        "train_split": train_split,
        "val_split": val_split,
        "data_root": str(data_root) if data_root else None,
        "out_dir": str(out_dir),
        "resume": str(resume) if isinstance(resume, (str, Path)) else bool(resume),
        "episodes_per_epoch": episodes_per_epoch,
        "val_episodes": val_episodes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "phase1_frozen_epochs": phase1_frozen_epochs,
        "phase1_partial_epochs": phase1_partial_epochs,
        "phase2_frozen_epochs": phase2_frozen_epochs,
        "phase2_partial_epochs": phase2_partial_epochs,
        "phase2_full_epochs": phase2_full_epochs,
        "contrastive": contrastive,
        "contrastive_weight_p1": contrastive_weight_p1,
        "contrastive_weight_p2": contrastive_weight_p2,
        "contrastive_temp": contrastive_temp,
        "vicreg": vicreg,
        "vicreg_weight": vicreg_weight,
        "hard_neg_mining": hard_neg_mining,
        "seed": seed,
        "device": device,
        "pretrained": pretrained,
    }
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    device_t = torch.device(device)

    def _make_val_loader(sources: list[str]) -> DataLoader:
        ds = EpisodeDataset(
            manifest_path=str(manifest),
            split=val_split,
            data_root=str(data_root) if data_root else None,
            episodes_per_epoch=val_episodes,
            neg_prob=0.3,
            train=False,
            seed=seed,
            sources=sources,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=(device_t.type == "cuda"),
        )

    def _build_train_loader(sources: list[str]) -> tuple[EpisodeDataset, DataLoader]:
        ds = EpisodeDataset(
            manifest_path=str(manifest),
            split=train_split,
            data_root=str(data_root) if data_root else None,
            episodes_per_epoch=episodes_per_epoch,
            train=True,
            sources=sources,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=(device_t.type == "cuda"),
            drop_last=True,
        )
        return ds, loader

    # Val loaders: one per phase (different domain).
    val_loader_p1 = _make_val_loader(PHASE1_VAL_SOURCES)
    val_loader_p2 = _make_val_loader(PHASE2_VAL_SOURCES)

    # Default train loader (Phase 2 target domain); rebuilt per phase below.
    train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)

    model = FewShotLocalizer(pretrained=pretrained).to(device_t)

    resume_path: Path | None = None
    if isinstance(resume, (str, Path)):
        candidate = Path(resume)
        if candidate.exists():
            resume_path = candidate
        else:
            print(f"warn: resume path {candidate} not found — starting fresh")
    elif resume is True:
        candidate = out_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate
            print(f"auto-resuming from {resume_path}")
        else:
            print(f"no checkpoint at {candidate} — starting fresh")

    resume_state: dict | None = None
    if resume_path is not None:
        loaded = torch.load(str(resume_path), map_location=device_t, weights_only=False)
        resume_state = dict(loaded) if not isinstance(loaded, dict) else loaded
        model.load_state_dict(
            resume_state["model"] if "model" in resume_state else resume_state
        )
        print(
            f"resumed from {resume_path} "
            f"(stage={resume_state.get('stage')}, epoch={resume_state.get('epoch')})"
        )
        rng = resume_state.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            try:
                cuda_states = [s.cpu().to(torch.uint8) for s in rng["torch_cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception as e:
                print(f"  warn: failed to restore cuda rng: {e}")
        if rng.get("python") is not None:
            _random.setstate(rng["python"])

    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: total={n_total/1e6:.2f}M")
    print(
        f"Phase 1 val instances: {len(val_loader_p1.dataset.instances)}  "  # type: ignore[attr-defined]
        f"Phase 2 val instances: {len(val_loader_p2.dataset.instances)}"  # type: ignore[attr-defined]
    )

    def _log_trainable(stage_name: str) -> None:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {stage_name} trainable params: {n/1e6:.2f}M")

    best = float(resume_state.get("best_val_iou", 0.0)) if resume_state else 0.0
    full_history: list[dict] = list(resume_state.get("full_history", [])) if resume_state else []

    stage_configs_for_lr: list[dict] = []

    def _resume_for(stage_name: str, configured_epochs: int):
        if (
            start_stage is not None
            and start_stage in STAGE_ORDER
            and stage_name in STAGE_ORDER
            and STAGE_ORDER.index(stage_name) < STAGE_ORDER.index(start_stage)
        ):
            return True, 1, None, None
        if not resume_state:
            return False, 1, None, None
        saved_stage = resume_state.get("stage")
        saved_epoch = int(resume_state.get("epoch", 0))
        if saved_stage == stage_name:
            if saved_epoch >= configured_epochs:
                return True, 1, None, None
            return (
                False,
                saved_epoch + 1,
                resume_state.get("optimizer"),
                resume_state.get("scheduler"),
            )
        if saved_stage in STAGE_ORDER and stage_name in STAGE_ORDER:
            if STAGE_ORDER.index(saved_stage) > STAGE_ORDER.index(stage_name):
                return True, 1, None, None
        return False, 1, None, None

    # -----------------------------------------------------------------------
    # Phase 1: VizWiz base pretraining
    # -----------------------------------------------------------------------

    if phase1_frozen_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage1_1", phase1_frozen_epochs)
        stage_configs_for_lr.append({
            "name": "stage1_1",
            "epochs": phase1_frozen_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [{"label": "heads", "lr": 1e-3}],
        })
        if skip:
            print("=== Stage 1.1: skipped ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 1.1: VizWiz base, backbone frozen{resume_str} ===")
            train_ds, train_loader = _build_train_loader(VIZWIZ_BASE_SOURCES)
            train_ds.set_hard_neg_ratio(0.0)
            opt = phase1_frozen_optimizer(model)
            _log_trainable("stage1_1")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader_p1, device_t,
                phase1_frozen_epochs, "stage1_1", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_p1,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=False,
                neg_prob_schedule={1: 0.1},
                attn_loss_weight=0.5,
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)
            _eval_stage("stage1_1", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device)

    if phase1_partial_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage1_2", phase1_partial_epochs)
        stage_configs_for_lr.append({
            "name": "stage1_2",
            "epochs": phase1_partial_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [
                {"label": "backbone upper", "lr": 1e-5},
                {"label": "heads", "lr": 5e-4},
            ],
        })
        if skip:
            print("=== Stage 1.2: skipped ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 1.2: VizWiz base, features[7:] @ 1e-5{resume_str} ===")
            train_ds, train_loader = _build_train_loader(VIZWIZ_BASE_SOURCES)
            train_ds.set_hard_neg_ratio(0.0)
            opt = phase1_partial_optimizer(model)
            _log_trainable("stage1_2")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader_p1, device_t,
                phase1_partial_epochs, "stage1_2", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_p1,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=False,
                neg_prob_schedule={1: 0.2},
                attn_loss_weight=0.5,
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)
            _eval_stage("stage1_2", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device)

    # -----------------------------------------------------------------------
    # Phase 2: target-domain fine-tuning
    # -----------------------------------------------------------------------

    if phase2_frozen_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2_1", phase2_frozen_epochs)
        stage_configs_for_lr.append({
            "name": "stage2_1",
            "epochs": phase2_frozen_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [{"label": "heads", "lr": 5e-4}],
        })
        if skip:
            print("=== Stage 2.1: skipped ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 2.1: HOTS+InsDet+VizWiz novel, backbone re-frozen{resume_str} ===")
            train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
            train_ds.set_hard_neg_ratio(0.25)
            opt = phase2_frozen_optimizer(model)
            _log_trainable("stage2_1")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader_p2, device_t,
                phase2_frozen_epochs, "stage2_1", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_p1,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=hard_neg_mining,
                neg_prob_schedule={1: 0.0, 4: 0.15},
                attn_loss_weight=1.0,
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)
            _eval_stage("stage2_1", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device)

    if phase2_partial_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2_2", phase2_partial_epochs)
        stage_configs_for_lr.append({
            "name": "stage2_2",
            "epochs": phase2_partial_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [
                {"label": "backbone upper", "lr": 1e-5},
                {"label": "heads", "lr": 5e-4},
            ],
        })
        if skip:
            print("=== Stage 2.2: skipped ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 2.2: HOTS+InsDet+VizWiz novel, features[7:] @ 1e-5{resume_str} ===")
            train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
            train_ds.set_hard_neg_ratio(0.5)
            opt = phase2_partial_optimizer(model)
            _log_trainable("stage2_2")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader_p2, device_t,
                phase2_partial_epochs, "stage2_2", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_p2,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=hard_neg_mining,
                neg_prob_schedule={1: 0.1, 8: 0.3},
                attn_loss_weight=1.0,
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)
            _eval_stage("stage2_2", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device)

    if phase2_full_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2_3", phase2_full_epochs)
        stage_configs_for_lr.append({
            "name": "stage2_3",
            "epochs": phase2_full_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [
                {"label": "backbone lower", "lr": 5e-6},
                {"label": "backbone upper", "lr": 5e-6},
                {"label": "heads", "lr": 1e-4},
            ],
        })
        if skip:
            print("=== Stage 2.3: skipped ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 2.3: full unfreeze (all @ 5e-6){resume_str} ===")
            train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
            train_ds.set_hard_neg_ratio(0.5)
            opt = phase2_full_optimizer(model)
            _log_trainable("stage2_3")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader_p2, device_t,
                phase2_full_epochs, "stage2_3", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_p2,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=hard_neg_mining,
                neg_prob_schedule={1: 0.3},
                attn_loss_weight=1.0,
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)
            _eval_stage("stage2_3", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device)

    print(f"done. best val_iou={best:.4f} (saved to {out_dir / 'best.pt'})")

    history_path = analysis_dir / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(full_history, f, indent=2)
    print(f"train history written to {history_path}")

    _generate_plots(
        full_history=full_history,
        stage_configs_for_lr=stage_configs_for_lr,
        model=model,
        train_ds=train_ds,
        device_t=device_t,
        manifest=Path(manifest),
        analysis_dir=analysis_dir,
    )

    return best


def _generate_plots(
    full_history: list[dict],
    stage_configs_for_lr: list[dict],
    model: FewShotLocalizer,
    train_ds: EpisodeDataset,
    device_t: torch.device,
    manifest: Path,
    analysis_dir: Path,
) -> None:
    from modeling.plot import (
        plot_ap_per_iou,
        plot_contrastive_learning,
        plot_dataset_stats,
        plot_lr_schedule,
        plot_prototype_similarity,
        plot_training_curves,
    )

    if full_history:
        plot_training_curves(full_history, analysis_dir)
        plot_contrastive_learning(full_history, analysis_dir)
        plot_ap_per_iou(full_history, analysis_dir)

    if stage_configs_for_lr:
        plot_lr_schedule(stage_configs_for_lr, analysis_dir)

    stats_path = manifest.parent / "stats.json"
    plot_dataset_stats(stats_path, analysis_dir)

    print("building prototype cache for similarity heatmap…")
    proto_cache = build_proto_cache(model, train_ds, device_t)
    plot_prototype_similarity(proto_cache, analysis_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="dataset/cleaned/manifest.json")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default="model")
    p.add_argument("--resume", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--episodes-per-epoch", type=int, default=1000)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    # Phase 1 epoch counts (set to 0 to skip a stage)
    p.add_argument("--phase1-frozen-epochs", type=int, default=10)
    p.add_argument("--phase1-partial-epochs", type=int, default=20)
    # Phase 2 epoch counts
    p.add_argument("--phase2-frozen-epochs", type=int, default=10)
    p.add_argument("--phase2-partial-epochs", type=int, default=35)
    p.add_argument("--phase2-full-epochs", type=int, default=10)
    p.add_argument(
        "--start-stage",
        choices=list(STAGE_ORDER),
        default=None,
        help="skip earlier stages and start fresh from this one",
    )
    p.add_argument("--no-contrastive", action="store_true")
    p.add_argument("--contrastive-weight-p1", type=float, default=0.1)
    p.add_argument("--contrastive-weight-p2", type=float, default=0.2)
    p.add_argument("--contrastive-temp", type=float, default=0.1)
    p.add_argument("--no-vicreg", action="store_true")
    p.add_argument("--vicreg-weight", type=float, default=0.05)
    p.add_argument("--no-hard-neg", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    if args.no_resume:
        resume_arg: str | bool = False
    else:
        resume_arg = args.resume if args.resume else True

    train(
        manifest=args.manifest,
        train_split=args.train_split,
        val_split=args.val_split,
        data_root=args.data_root,
        out_dir=args.out_dir,
        resume=resume_arg,
        episodes_per_epoch=args.episodes_per_epoch,
        val_episodes=args.val_episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        phase1_frozen_epochs=args.phase1_frozen_epochs,
        phase1_partial_epochs=args.phase1_partial_epochs,
        phase2_frozen_epochs=args.phase2_frozen_epochs,
        phase2_partial_epochs=args.phase2_partial_epochs,
        phase2_full_epochs=args.phase2_full_epochs,
        contrastive=not args.no_contrastive,
        contrastive_weight_p1=args.contrastive_weight_p1,
        contrastive_weight_p2=args.contrastive_weight_p2,
        contrastive_temp=args.contrastive_temp,
        vicreg=not args.no_vicreg,
        vicreg_weight=args.vicreg_weight,
        hard_neg_mining=not args.no_hard_neg,
        start_stage=args.start_stage,
        seed=args.seed,
        device=args.device,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
