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
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import random as _random

from modeling.dataset import EpisodeDataset, _Augment, _load_image, collate
from modeling.evaluate import IOU_THRESHOLDS, _compute_pr_ap, _iou_xyxy, run as evaluate_run
from modeling.loss import _containment_ratio
from modeling.loss import (
    barlow_twins_loss,
    nt_xent_loss,
    total_loss,
    triplet_loss,
    vicreg_loss,
)
from modeling.model import FewShotLocalizer, decode


# ---------------------------------------------------------------------------
# Async checkpoint helper
# ---------------------------------------------------------------------------

_CKPT_TMP_DIR = Path("/tmp/opencode/ckpts")
_ckpt_copy_lock = threading.Lock()
_pending_copies: list[threading.Thread] = []


def _save_checkpoint_async(payload: dict, drive_path: Path) -> None:
    """Write checkpoint to /tmp first (fast), then copy to Drive in background.

    The caller must invoke _flush_pending_copies() before the training function
    returns so Drive is always up-to-date before the next Colab cell starts.
    """
    _CKPT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CKPT_TMP_DIR / drive_path.name
    torch.save(payload, tmp)

    def _copy() -> None:
        try:
            drive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp, drive_path)
        except Exception as exc:
            print(f"  warn: async ckpt copy to {drive_path} failed: {exc}")

    t = threading.Thread(target=_copy, daemon=True)
    with _ckpt_copy_lock:
        _pending_copies.append(t)
    t.start()


def _flush_pending_copies(timeout: float = 120.0) -> None:
    """Wait for all in-flight Drive copy threads to finish."""
    with _ckpt_copy_lock:
        threads = list(_pending_copies)
        _pending_copies.clear()
    for t in threads:
        t.join(timeout=timeout)


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
        + list(model.support_fusion.parameters())
        + list(model.support_pool.parameters())
        + list(model.fpn.parameters())
        + list(model.p3_lat.parameters())
        + [model.p3_gate]
        + [model.query_pe]
        + [model.absent_token]
        + list(model.query_decoder.parameters())
        + list(model.det_head.parameters())
        + list(model.presence_head.parameters())
    )


# LR notes (2026-05 architecture, post query-decoder + Phase-2 K-fold CV):
#   The heads now contain SIX attention modules: SupportTokenizer slot attn,
#   2× SupportFusion encoder attn, AttentionPool query, and the QueryDecoder's
#   self-attn + cross-attn. Plus DropPath p=0.1 on three decoder branches
#   adds gradient noise. We drop stage1_1 head LR another step (5e-4 → 3e-4)
#   to avoid attention collapse on the new self-attn, and stage1_2 → 2e-4
#   once the backbone joins. Stage 2.1 heads stay at 2.5e-4 (frozen-backbone
#   re-warmup is short, 10 epochs). Stage 2.2 heads dropped 2.5e-4 → 2e-4:
#   K-fold CV trains each fold on ~10-11 HOTS+InsDet instances (+ 16 vizwiz
#   novel), so the smaller per-fold dataset wants slightly cooler heads to
#   avoid overfitting noise. Stage 2.3 backbone dropped 5e-6 → 3e-6: epoch
#   budget went 10 → 15 (+50%), so we offset the longer cosine schedule to
#   keep cumulative low-level ImageNet-feature drift bounded.


def phase1_frozen_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 1.1: backbone fully frozen, heads at 3e-4 (six attention blocks)."""
    model.backbone.freeze_all()
    return torch.optim.AdamW(
        [{"params": _heads(model), "lr": 3e-4, "weight_decay": 1e-4}]
    )


def phase1_partial_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 1.2: features[7:] @ 1e-5, heads at 2e-4."""
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
            {"params": _heads(model), "lr": 2e-4, "weight_decay": 1e-4},
        ]
    )


def phase2_frozen_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 2.1: re-freeze backbone, heads at 2.5e-4 (warmed heads)."""
    model.backbone.freeze_all()
    return torch.optim.AdamW(
        [{"params": _heads(model), "lr": 2.5e-4, "weight_decay": 1e-4}]
    )


def phase2_partial_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 2.2: features[7:] @ 1e-5, heads at 2e-4.

    Heads dropped 2.5e-4 → 2e-4: with K-fold CV active, each fold trains on
    only ~10-11 HOTS+InsDet instances (plus 16 vizwiz_novel). At that scale
    2.5e-4 over 30 epochs risks overfitting noise on the small fold; 2e-4
    gives cleaner gradients while staying productive.
    """
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
            {"params": _heads(model), "lr": 2e-4, "weight_decay": 1e-4},
        ]
    )


def phase2_full_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 2.3: full unfreeze. Backbone at 3e-6 to preserve ImageNet features.

    Backbone dropped 5e-6 → 3e-6: epoch budget went 10 → 15 (+50%), so the
    cumulative gradient flow through `features[0:7]` (low-level ImageNet
    filters) grows proportionally. Lowering the backbone LR offsets the
    longer schedule and keeps low-level feature drift bounded. Heads stay
    at 5e-5 — heads are not the catastrophic-forgetting risk and need the
    polish budget.
    """
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
            {"params": lower, "lr": 3e-6, "weight_decay": 1e-4},
            {"params": upper, "lr": 3e-6, "weight_decay": 1e-4},
            {"params": _heads(model), "lr": 5e-5, "weight_decay": 1e-4},
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
    batch_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Compute one prototype per instance (eval augmentation, fixed seed).

    Called once per epoch before the training loop so _sample_query can
    find hard negatives by cosine similarity between prototype vectors.
    Instances are processed in batches of `batch_size` to reduce GPU overhead.
    """
    was_training = model.training
    model.eval()

    aug = _Augment("support", train=False)
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
                t, _ = aug(img, list(s["bbox"]), rng)
                imgs.append(t)
            batch_support.append(torch.stack(imgs))

        support_imgs_t = torch.stack(batch_support).to(device)
        tokens, _, _ = model.encode_support(support_imgs_t)
        prototypes = model.support_pool(tokens)
        for i, instance in enumerate(batch_instances):
            cache[instance["instance_id"]] = prototypes[i].cpu()

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
    barlow: bool = True,
    barlow_weight: float = 0.005,
    triplet: bool = False,
    triplet_weight: float = 0.05,
    hard_neg_mining: bool = True,
    neg_prob_schedule: dict[int, float] | None = None,
    attn_loss_weight: float = 0.5,
    start_epoch: int = 1,
    scheduler_state: dict | None = None,
    prior_history: list[dict] | None = None,
    fold_label: str | None = None,
) -> tuple[float, list[dict]]:
    total_steps = max(epochs * len(train_loader), 1)
    # 5% warmup, capped at 500 steps (was 200). Transformer modules in the
    # support pipeline benefit from a longer warmup before LR peaks; with
    # ~125 steps/epoch (2000 episodes / B=16) the cap matters mainly for
    # multi-stage runs at the high end.
    warmup_steps = max(1, min(500, total_steps // 20))
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
            "nt_xent": 0.0, "vicreg": 0.0, "barlow": 0.0, "triplet": 0.0,
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

            nt_val, vr_val, bt_val, trip_val = 0.0, 0.0, 0.0, 0.0
            # Prefer per-shot prototypes (B, K, D) for nt_xent / vicreg / barlow —
            # supervised contrastive over B*K vectors with same-episode
            # positives is far stronger than uniformity over B prototypes,
            # and Barlow Twins genuinely needs paired views (shot 0 vs shots 1..K-1).
            per_shot = out.get("per_shot_prototype")
            con_target = per_shot if per_shot is not None else out["prototype"]
            if contrastive:
                nt = nt_xent_loss(con_target, temperature=contrastive_temp)
                loss = loss + contrastive_weight * nt
                nt_val = nt.detach().item()
            if vicreg:
                vr = vicreg_loss(con_target)
                loss = loss + vicreg_weight * vr
                vr_val = vr.detach().item()
            if barlow and per_shot is not None:
                bt = barlow_twins_loss(per_shot)
                loss = loss + barlow_weight * bt
                bt_val = bt.detach().item()
            if triplet:
                # Triplet still consumes the (B, dim) bag-level summary —
                # one anchor per episode is what its instance_id grouping expects.
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
            running["barlow"] += bt_val
            running["triplet"] += trip_val

        nb = len(train_loader)
        avg = {k: v / max(nb, 1) for k, v in running.items()}
        metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0
        reg_str = ""
        if contrastive or vicreg:
            reg_str += f" nt_xent={avg['nt_xent']:.4f} vicreg={avg['vicreg']:.4f}"
        if barlow:
            reg_str += f" barlow={avg['barlow']:.4f}"
        if triplet:
            reg_str += f" triplet={avg['triplet']:.4f}"
        prefix = f"  [{fold_label}]" if fold_label else f"[{stage_name}] epoch {epoch}/{epochs}"
        print(
            f"{prefix} "
            f"loss={avg['loss']:.4f} (focal={avg['focal']:.4f} box={avg['box']:.4f} "
            f"presence={avg['presence']:.4f} attn={avg['attn']:.4f}{reg_str}) "
            f"val_iou={metrics['val_iou']:.4f} "
            f"val_map50={metrics['val_map_50']:.4f} "
            f"val_f1_50={metrics['val_f1_50']:.4f} "
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
            "barlow": avg["barlow"],
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
            _save_checkpoint_async(
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
                f"val_map50={metrics['val_map_50']:.4f} "
                f"val_map5095={metrics['val_map_5095']:.4f} "
                f"val_f1_50={metrics['val_f1_50']:.4f})"
            )

        if not fold_label:
            # Two rolling full-state writes per epoch:
            #   last.pt          — the global "most recent" pointer (used by default
            #                       auto-resume across the whole pipeline)
            #   <stage_name>.pt  — the per-stage rolling snapshot. After this write
            #                       it always reflects the latest completed epoch of
            #                       that stage, so killing the run mid-stage and
            #                       passing --resume model/<stage>.pt restores
            #                       optimizer + scheduler + rng + history exactly.
            full_ckpt = _make_full_ckpt(
                model, optimizer, scheduler, stage_name, epoch, epochs,
                best_val_iou, prior + epoch_history,
            )
            _save_checkpoint_async(full_ckpt, out_dir / "last.pt")
            _save_checkpoint_async(full_ckpt, out_dir / f"{stage_name}.pt")
            print(
                f"  saved last.pt + {stage_name}.pt ({stage_name} epoch {epoch}/{epochs})"
            )

    _flush_pending_copies()
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
    full_history: list[dict] | None = None,
) -> None:
    """Run test eval for a stage and write per-stage analysis artefacts.

    Per-stage outputs land in ``analysis/<stage_name>/``:
      test_report.json       — eval metrics on the held-out test set
      eval_metrics.png       — bar/score-distribution from the test report
      train_history.json     — slice of full_history for this stage only
      training_curves.png    — per-epoch losses + val metrics, this stage only
      contrastive_learning.png
      ap_per_iou.png

    The global versions (covering all stages) are still produced once at the
    end of training by ``_generate_plots``.
    """
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
    # Per-stage training-history slice + plots so each stage folder is a
    # self-contained snapshot of how that stage trained.
    if full_history:
        stage_history = [r for r in full_history if r.get("stage") == stage_name]
        if stage_history:
            history_path = stage_dir / "train_history.json"
            with open(history_path, "w") as f:
                json.dump(stage_history, f, indent=2)
            print(f"saved {history_path}")
            from modeling.plot import (
                plot_ap_per_iou,
                plot_contrastive_learning,
                plot_training_curves,
            )
            plot_training_curves(stage_history, stage_dir)
            plot_contrastive_learning(stage_history, stage_dir)
            plot_ap_per_iou(stage_history, stage_dir)


# ---------------------------------------------------------------------------
# K-fold cross-validation helpers (Phase 2 only)
# ---------------------------------------------------------------------------


PHASE2_CV_SOURCES = ("hots", "insdet")


def _build_phase2_cv_folds(
    manifest_path: str | Path,
    k: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    """Pool HOTS+InsDet instances from the manifest's train+val splits and
    partition their instance_ids into K folds.

    Each fold is `{"train_ids": [...], "val_ids": [...]}` covering only the
    HOTS+InsDet instances. vizwiz_novel and Phase-1 sources are unaffected
    (vizwiz_novel always remains fully in training across all folds).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    pool: list[str] = []
    for inst in manifest["instances"]:
        if inst.get("source") in PHASE2_CV_SOURCES and inst.get("split") in ("train", "val"):
            pool.append(inst["instance_id"])
    pool = sorted(set(pool))
    rng = _random.Random(seed)
    rng.shuffle(pool)
    n = len(pool)
    if k <= 1 or n < k:
        raise ValueError(
            f"phase2_cv_folds={k} requires k>=2 and pool>=k; got pool={n}"
        )
    # Even-sized fold partition (last folds absorb the remainder).
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    folds: list[dict[str, list[str]]] = []
    cursor = 0
    for fs in fold_sizes:
        val_ids = pool[cursor : cursor + fs]
        train_ids = [x for x in pool if x not in set(val_ids)]
        folds.append({"train_ids": train_ids, "val_ids": val_ids})
        cursor += fs
    return folds


def _filter_dataset_for_fold(
    ds: EpisodeDataset,
    train_ids: set[str] | None = None,
    val_ids: set[str] | None = None,
) -> None:
    """In-place filter of `ds.instances` to a fold-specific subset.

    For training datasets pass `train_ids` — vizwiz_novel instances are kept
    automatically (they never enter CV). For validation datasets pass
    `val_ids` — the dataset is restricted to exactly those instance ids.
    """
    if val_ids is not None:
        ds.instances = [i for i in ds.instances if i["instance_id"] in val_ids]
        return
    if train_ids is not None:
        keep: list[dict[str, Any]] = []
        for inst in ds.instances:
            if inst.get("source") == "vizwiz_novel":
                keep.append(inst)
            elif inst["instance_id"] in train_ids:
                keep.append(inst)
        ds.instances = keep


def _save_cv_results(
    cv_results: list[dict],
    analysis_dir: Path,
) -> None:
    """Write per-fold + aggregated CV results to analysis/cv_results.json.

    Schema:
      {
        "n_folds": K,
        "folds": [{"fold": i, "val_ids": [...], "history": [...]}, ...],
        "aggregate": {
            "stage2_1": {"epoch_1": {"val_iou_mean": ..., ...}, ...},
            "stage2_2": {...},
            "stage2_3": {...},
        }
      }
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    metric_keys = (
        "val_loss",
        "val_iou",
        "val_contain",
        "val_map_50",
        "val_map_5095",
        "val_f1_50",
        "val_presence_acc",
    )
    # Group every (stage, epoch) pair across folds.
    by_key: dict[tuple[str, int], dict[str, list[float]]] = {}
    for fold in cv_results:
        for row in fold["history"]:
            stage = row.get("stage")
            epoch = int(row.get("epoch", 0))
            if not stage or not epoch:
                continue
            bucket = by_key.setdefault((stage, epoch), {k: [] for k in metric_keys})
            for k in metric_keys:
                v = row.get(k)
                if isinstance(v, (int, float)):
                    bucket[k].append(float(v))
    for (stage, epoch), vals in by_key.items():
        stage_dict = aggregate.setdefault(stage, {})
        epoch_dict: dict[str, float] = {}
        for k, lst in vals.items():
            if lst:
                epoch_dict[f"{k}_mean"] = sum(lst) / len(lst)
                epoch_dict[f"{k}_min"] = min(lst)
                epoch_dict[f"{k}_max"] = max(lst)
        epoch_dict["n_folds"] = len(next(iter(vals.values())))
        stage_dict[f"epoch_{epoch}"] = epoch_dict

    payload = {
        "n_folds": len(cv_results),
        "folds": cv_results,
        "aggregate": aggregate,
    }
    p = analysis_dir / "cv_results.json"
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  cv results -> {p}")

    # Pretty summary printed to console: final-epoch val_iou mean per stage.
    print("CV summary (mean across folds, final epoch of each stage):")
    for stage in ("stage2_1", "stage2_2", "stage2_3"):
        stage_agg = aggregate.get(stage)
        if not stage_agg:
            continue
        last = max(int(k.split("_")[1]) for k in stage_agg)
        m = stage_agg[f"epoch_{last}"]
        print(
            f"  {stage} (epoch {last}, n_folds={int(m['n_folds'])}): "
            f"val_iou={m.get('val_iou_mean', 0):.4f} "
            f"val_map50={m.get('val_map_50_mean', 0):.4f} "
            f"val_f1_50={m.get('val_f1_50_mean', 0):.4f}"
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
    # Phase 1: VizWiz base pretraining.
    #   Stage 1.1 set to 15: heads-only warmup with frozen backbone converges
    #     fast on the diverse VizWiz pool; extra epochs at this stage have
    #     diminishing returns since the bulk of head adaptation happens once
    #     the backbone joins in Stage 1.2.
    #   Stage 1.2 bumped 30 → 40: extended to allow fuller convergence.
    phase1_frozen_epochs: int = 15,
    phase1_partial_epochs: int = 40,
    # Phase 2: target-domain fine-tuning.
    #   Stage 2.1 set to 10: with K-fold CV in play the per-fold val pool is
    #     already noisy, so additional re-warmup epochs at frozen-backbone
    #     don't move the needle. Heads adapt fast under the new domain.
    #   Stage 2.2 set to 30 — main work; hard-neg mining is on.
    #   Stage 2.3 set to 15 — polish stage; bumped from 10 for a small
    #     extra polish budget while keeping ImageNet-feature drift in the
    #     lower backbone bounded (5e-6 × 15 epochs is still controlled).
    phase2_frozen_epochs: int = 10,
    phase2_partial_epochs: int = 30,
    phase2_full_epochs: int = 15,
    # K-fold cross-validation for Phase 2. With ~14 HOTS+InsDet instances the
    # fixed val split (~3 instances) is too small to give a stable signal —
    # CV pools the existing train+val splits back together and rotates a held-
    # out val fold across runs. Val metrics are monitored only (no early
    # stopping, no checkpoint gating). vizwiz_novel always stays in training.
    # Set to 0 to disable (single-run, current behaviour).
    phase2_cv_folds: int = 0,
    # Regularisation. nt_xent now runs SupCon over B*K=80 anchors with same-
    # episode positives instead of uniformity over B=16 prototypes — the
    # gradient density is ~5x higher per step, so the contrastive loss
    # weight is dropped accordingly to keep its share of the total comparable.
    # Temperature dropped 0.1 → 0.07 (standard SupCon).
    contrastive: bool = True,
    contrastive_weight_p1: float = 0.05,
    contrastive_weight_p2: float = 0.1,
    contrastive_temp: float = 0.07,
    vicreg: bool = True,
    vicreg_weight: float = 0.05,
    barlow: bool = True,
    barlow_weight: float = 0.005,
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
        "phase2_cv_folds": phase2_cv_folds,
        "contrastive": contrastive,
        "contrastive_weight_p1": contrastive_weight_p1,
        "contrastive_weight_p2": contrastive_weight_p2,
        "contrastive_temp": contrastive_temp,
        "vicreg": vicreg,
        "vicreg_weight": vicreg_weight,
        "barlow": barlow,
        "barlow_weight": barlow_weight,
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
            "param_groups": [{"label": "heads", "lr": 3e-4}],
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
                barlow=barlow,
                barlow_weight=barlow_weight,
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
            _eval_stage("stage1_1", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device, full_history=full_history)

    if phase1_partial_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage1_2", phase1_partial_epochs)
        stage_configs_for_lr.append({
            "name": "stage1_2",
            "epochs": phase1_partial_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [
                {"label": "backbone upper", "lr": 1e-5},
                {"label": "heads", "lr": 2e-4},
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
                barlow=barlow,
                barlow_weight=barlow_weight,
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
            _eval_stage("stage1_2", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device, full_history=full_history)

    # -----------------------------------------------------------------------
    # Phase 2: target-domain fine-tuning
    #
    # When phase2_cv_folds > 0, the HOTS+InsDet instances pooled from the
    # manifest's existing train+val splits are partitioned into K folds.
    # For each fold the model is reset to its post-Phase-1 state and all
    # three Phase-2 stages run end-to-end with that fold's val held out.
    # Val metrics are MONITORED ONLY — no early stopping, no checkpoint
    # gating on val. vizwiz_novel always stays in training across folds.
    # -----------------------------------------------------------------------

    # Stage configs registered once regardless of CV (used by the LR plot).
    if phase2_frozen_epochs > 0:
        stage_configs_for_lr.append({
            "name": "stage2_1",
            "epochs": phase2_frozen_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [{"label": "heads", "lr": 2.5e-4}],
        })
    if phase2_partial_epochs > 0:
        stage_configs_for_lr.append({
            "name": "stage2_2",
            "epochs": phase2_partial_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [
                {"label": "backbone upper", "lr": 1e-5},
                {"label": "heads", "lr": 2e-4},
            ],
        })
    if phase2_full_epochs > 0:
        stage_configs_for_lr.append({
            "name": "stage2_3",
            "epochs": phase2_full_epochs,
            "steps_per_epoch": max(episodes_per_epoch // batch_size, 1),
            "param_groups": [
                {"label": "backbone lower", "lr": 3e-6},
                {"label": "backbone upper", "lr": 3e-6},
                {"label": "heads", "lr": 5e-5},
            ],
        })

    def _run_phase2_stage_2_1(
        train_loader_local: DataLoader,
        val_loader_local: DataLoader,
        best_local: float,
        history_local: list[dict],
    ) -> tuple[float, list[dict]]:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2_1", phase2_frozen_epochs)
        if skip:
            print("=== Stage 2.1: skipped ===")
            return best_local, []
        resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
        print(f"=== Stage 2.1: HOTS+InsDet+VizWiz novel, backbone re-frozen{resume_str} ===")
        train_loader_local.dataset.set_hard_neg_ratio(0.25)  # type: ignore[attr-defined]
        opt = phase2_frozen_optimizer(model)
        _log_trainable("stage2_1")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        return train_stage(
            model, opt, train_loader_local, val_loader_local, device_t,
            phase2_frozen_epochs, "stage2_1", out_dir, best_local,
            contrastive=contrastive,
            contrastive_weight=contrastive_weight_p1,
            contrastive_temp=contrastive_temp,
            vicreg=vicreg,
            vicreg_weight=vicreg_weight,
            barlow=barlow,
            barlow_weight=barlow_weight,
            triplet=triplet,
            triplet_weight=triplet_weight,
            hard_neg_mining=hard_neg_mining,
            neg_prob_schedule={1: 0.0, 4: 0.15},
            attn_loss_weight=1.0,
            start_epoch=start_epoch,
            scheduler_state=sched_state,
            prior_history=history_local,
        )

    def _run_phase2_stage_2_2(
        train_loader_local: DataLoader,
        val_loader_local: DataLoader,
        best_local: float,
        history_local: list[dict],
    ) -> tuple[float, list[dict]]:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2_2", phase2_partial_epochs)
        if skip:
            print("=== Stage 2.2: skipped ===")
            return best_local, []
        resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
        print(f"=== Stage 2.2: HOTS+InsDet+VizWiz novel, features[7:] @ 1e-5{resume_str} ===")
        train_loader_local.dataset.set_hard_neg_ratio(0.5)  # type: ignore[attr-defined]
        opt = phase2_partial_optimizer(model)
        _log_trainable("stage2_2")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        return train_stage(
            model, opt, train_loader_local, val_loader_local, device_t,
            phase2_partial_epochs, "stage2_2", out_dir, best_local,
            contrastive=contrastive,
            contrastive_weight=contrastive_weight_p2,
            contrastive_temp=contrastive_temp,
            vicreg=vicreg,
            vicreg_weight=vicreg_weight,
            barlow=barlow,
            barlow_weight=barlow_weight,
            triplet=triplet,
            triplet_weight=triplet_weight,
            hard_neg_mining=hard_neg_mining,
            neg_prob_schedule={1: 0.1, 8: 0.3},
            attn_loss_weight=1.0,
            start_epoch=start_epoch,
            scheduler_state=sched_state,
            prior_history=history_local,
        )

    def _run_phase2_stage_2_3(
        train_loader_local: DataLoader,
        val_loader_local: DataLoader,
        best_local: float,
        history_local: list[dict],
    ) -> tuple[float, list[dict]]:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2_3", phase2_full_epochs)
        if skip:
            print("=== Stage 2.3: skipped ===")
            return best_local, []
        resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
        print(f"=== Stage 2.3: full unfreeze (all @ 5e-6){resume_str} ===")
        train_loader_local.dataset.set_hard_neg_ratio(0.5)  # type: ignore[attr-defined]
        opt = phase2_full_optimizer(model)
        _log_trainable("stage2_3")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        return train_stage(
            model, opt, train_loader_local, val_loader_local, device_t,
            phase2_full_epochs, "stage2_3", out_dir, best_local,
            contrastive=contrastive,
            contrastive_weight=contrastive_weight_p2,
            contrastive_temp=contrastive_temp,
            vicreg=vicreg,
            vicreg_weight=vicreg_weight,
            barlow=barlow,
            barlow_weight=barlow_weight,
            triplet=triplet,
            triplet_weight=triplet_weight,
            hard_neg_mining=hard_neg_mining,
            neg_prob_schedule={1: 0.3},
            attn_loss_weight=1.0,
            start_epoch=start_epoch,
            scheduler_state=sched_state,
            prior_history=history_local,
        )

    if phase2_cv_folds > 0 and (
        phase2_frozen_epochs > 0
        or phase2_partial_epochs > 0
        or phase2_full_epochs > 0
    ):
        # CV runs always start each fold from the post-Phase-1 weights —
        # mid-Phase-2 resume across folds is ill-defined (whose fold's
        # optimizer state? whose val IDs?). Disable per-stage resume so
        # _resume_for() returns "fresh" for stages 2.x in every fold.
        if resume_state is not None and resume_state.get("stage") in (
            "stage2_1",
            "stage2_2",
            "stage2_3",
        ):
            print(
                "  CV mode: discarding mid-Phase-2 resume state "
                f"(was stage={resume_state.get('stage')}, epoch={resume_state.get('epoch')})"
            )
            resume_state = None
        # Snapshot the post-Phase-1 weights so each fold starts identically.
        phase1_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        folds = _build_phase2_cv_folds(
            manifest_path=manifest, k=phase2_cv_folds, seed=seed
        )
        n_total_instances = sum(len(f["val_ids"]) for f in folds)
        print(
            f"=== Phase 2 K-fold CV: {phase2_cv_folds} folds over "
            f"{n_total_instances} HOTS+InsDet instances ==="
        )

        # Build all fold loaders up front and print a single size summary.
        fold_structs: list[dict] = []
        for fold_idx, fold in enumerate(folds):
            train_ids = set(fold["train_ids"])
            val_ids = set(fold["val_ids"])
            model.load_state_dict(phase1_state)
            fold_train_ds, fold_train_loader = _build_train_loader(PHASE2_SOURCES)
            _filter_dataset_for_fold(fold_train_ds, train_ids=train_ids)
            fold_val_loader = _make_val_loader(list(PHASE2_VAL_SOURCES))
            _filter_dataset_for_fold(fold_val_loader.dataset, val_ids=val_ids)  # type: ignore[arg-type]
            fold_structs.append({
                "fold_idx": fold_idx,
                "train_ids": sorted(train_ids),
                "val_ids": sorted(val_ids),
                "train_loader": fold_train_loader,
                "train_ds": fold_train_ds,
                "val_loader": fold_val_loader,
                "history": [],
                "best": best,
                "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            })
            n_train = len(fold_train_ds.instances)
            n_val = len(fold_val_loader.dataset.instances)  # type: ignore[attr-defined]
            print(f"  fold {fold_idx + 1}/{phase2_cv_folds}: {n_train} train, {n_val} val")

        cv_results: list[dict] = []
        best_fold_iou = -1.0
        best_fold_idx = -1

        def _run_cv_stage(
            stage_name: str,
            stage_epochs: int,
            setup_fold_fn: Any,
        ) -> None:
            """Run `stage_epochs` CV epochs for one stage.

            Each CV epoch trains every fold for exactly 1 epoch, then prints
            a summary line with the mean val metrics across folds.

            `setup_fold_fn(fold_struct)` is called once per fold before the
            first epoch to build the optimizer, configure hard-neg ratio, etc.
            It must return (optimizer, neg_prob_schedule, attn_loss_weight,
            contrastive_weight_local).
            """
            nonlocal best, best_fold_iou, best_fold_idx

            print(f"\n=== {stage_name} CV: {stage_epochs} epochs × {phase2_cv_folds} folds ===")

            # Per-fold optimizer + scheduler, built once and reused across epochs.
            fold_opts: list[dict] = []
            logged_trainable = False
            for fs in fold_structs:
                model.load_state_dict(fs["model_state"])
                opt, neg_sched, attn_w, cont_w = setup_fold_fn(fs)
                if not logged_trainable:
                    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"  trainable params: {n_train / 1e6:.2f}M")
                    logged_trainable = True
                total_steps = max(stage_epochs * len(fs["train_loader"]), 1)
                warmup_steps = max(1, min(500, total_steps // 20))
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-7
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
                )
                fold_opts.append({
                    "opt": opt,
                    "scheduler": scheduler,
                    "sched_state": None,
                    "neg_sched": neg_sched,
                    "attn_w": attn_w,
                    "cont_w": cont_w,
                })

            for cv_epoch in range(1, stage_epochs + 1):
                print(f"\n  CV epoch {cv_epoch}/{stage_epochs}")
                epoch_val_ious: list[float] = []
                epoch_val_map50s: list[float] = []

                for fi, (fs, fo) in enumerate(zip(fold_structs, fold_opts)):
                    fold_label = f"fold {fi + 1}/{phase2_cv_folds}"
                    model.load_state_dict(fs["model_state"])

                    _, hist = train_stage(
                        model=model,
                        optimizer=fo["opt"],
                        train_loader=fs["train_loader"],
                        val_loader=fs["val_loader"],
                        device=device_t,
                        epochs=stage_epochs,
                        stage_name=stage_name,
                        out_dir=out_dir,
                        best_val_iou=fs["best"],
                        contrastive=contrastive,
                        contrastive_weight=fo["cont_w"],
                        contrastive_temp=contrastive_temp,
                        vicreg=vicreg,
                        vicreg_weight=vicreg_weight,
                        barlow=barlow,
                        barlow_weight=barlow_weight,
                        triplet=triplet,
                        triplet_weight=triplet_weight,
                        hard_neg_mining=hard_neg_mining,
                        neg_prob_schedule=fo["neg_sched"],
                        attn_loss_weight=fo["attn_w"],
                        start_epoch=cv_epoch,
                        scheduler_state=fo["sched_state"],
                        prior_history=fs["history"],
                        fold_label=fold_label,
                    )
                    for _ in range(len(fs["train_loader"])):
                        fo["scheduler"].step()
                    fo["sched_state"] = fo["scheduler"].state_dict()
                    fs["model_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    if hist:
                        row = hist[0]
                        fs["history"].append(row)
                        v_iou = row["val_iou"]
                        epoch_val_ious.append(v_iou)
                        epoch_val_map50s.append(row["val_map_50"])
                        if v_iou > fs["best"]:
                            fs["best"] = v_iou

                mean_iou = sum(epoch_val_ious) / len(epoch_val_ious) if epoch_val_ious else 0.0
                mean_map50 = sum(epoch_val_map50s) / len(epoch_val_map50s) if epoch_val_map50s else 0.0
                fold_ious_str = "  ".join(f"f{i+1}:{v:.4f}" for i, v in enumerate(epoch_val_ious))
                print(
                    f"  mean val_iou={mean_iou:.4f}  mean val_map50={mean_map50:.4f}"
                    f"  ({fold_ious_str})"
                )

                # After each CV epoch, save rolling checkpoints (last fold's model state).
                model.load_state_dict(fold_structs[-1]["model_state"])
                full_ckpt = _make_full_ckpt(
                    model, fold_opts[-1]["opt"], fold_opts[-1]["scheduler"],
                    stage_name, cv_epoch, stage_epochs,
                    max(fs["best"] for fs in fold_structs),
                    full_history,
                )
                _save_checkpoint_async(full_ckpt, out_dir / "last.pt")
                _save_checkpoint_async(full_ckpt, out_dir / f"{stage_name}.pt")
                print(
                    f"  saved last.pt + {stage_name}.pt (cv epoch {cv_epoch}/{stage_epochs})"
                )

            _flush_pending_copies()

            for fi, fs in enumerate(fold_structs):
                fold_best = fs["best"]
                stage_history = list(fs["history"])
                cv_results.append({
                    "fold": fi,
                    "stage": stage_name,
                    "train_ids": fs["train_ids"],
                    "val_ids": fs["val_ids"],
                    "history": stage_history,
                })
                full_history.extend(stage_history)
                fs["history"] = []
                if fold_best > best_fold_iou:
                    best_fold_iou = fold_best
                    best_fold_idx = fi
                fold_dir = analysis_dir / "cv" / f"fold_{fi}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                with open(fold_dir / f"history_{stage_name}.json", "w") as f:
                    json.dump(stage_history, f, indent=2)

            best = max(best, best_fold_iou)

        if phase2_frozen_epochs > 0:
            def _setup_2_1(fs: dict) -> tuple:
                model.load_state_dict(fs["model_state"])
                fs["train_loader"].dataset.set_hard_neg_ratio(0.25)  # type: ignore[attr-defined]
                opt = phase2_frozen_optimizer(model)
                return opt, {1: 0.0, 4: 0.15}, 1.0, contrastive_weight_p1
            _run_cv_stage("stage2_1", phase2_frozen_epochs, _setup_2_1)

        if phase2_partial_epochs > 0:
            def _setup_2_2(fs: dict) -> tuple:
                model.load_state_dict(fs["model_state"])
                fs["train_loader"].dataset.set_hard_neg_ratio(0.5)  # type: ignore[attr-defined]
                opt = phase2_partial_optimizer(model)
                return opt, {1: 0.1, 8: 0.3}, 1.0, contrastive_weight_p2
            _run_cv_stage("stage2_2", phase2_partial_epochs, _setup_2_2)

        if phase2_full_epochs > 0:
            def _setup_2_3(fs: dict) -> tuple:
                model.load_state_dict(fs["model_state"])
                fs["train_loader"].dataset.set_hard_neg_ratio(0.5)  # type: ignore[attr-defined]
                opt = phase2_full_optimizer(model)
                return opt, {1: 0.3}, 1.0, contrastive_weight_p2
            _run_cv_stage("stage2_3", phase2_full_epochs, _setup_2_3)

        print(
            f"\n=== Phase 2 CV done. best fold = {best_fold_idx + 1}/{phase2_cv_folds} "
            f"(val_iou={best_fold_iou:.4f}) ==="
        )
        _save_cv_results(cv_results, analysis_dir)
        # Rebuild an unfiltered Phase-2 train_ds so the end-of-run prototype
        # similarity plot covers the entire instance pool, not the last fold.
        train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
    else:
        # Single-run (non-CV) Phase 2 — original behaviour. Uses the manifest's
        # fixed val split and writes test-set evaluation per stage.
        if phase2_frozen_epochs > 0:
            train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
            best_new, hist = _run_phase2_stage_2_1(
                train_loader, val_loader_p2, best, full_history
            )
            best = best_new
            full_history.extend(hist)
            _eval_stage("stage2_1", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device, full_history=full_history)

        if phase2_partial_epochs > 0:
            train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
            best_new, hist = _run_phase2_stage_2_2(
                train_loader, val_loader_p2, best, full_history
            )
            best = best_new
            full_history.extend(hist)
            _eval_stage("stage2_2", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device, full_history=full_history)

        if phase2_full_epochs > 0:
            train_ds, train_loader = _build_train_loader(PHASE2_SOURCES)
            best_new, hist = _run_phase2_stage_2_3(
                train_loader, val_loader_p2, best, full_history
            )
            best = best_new
            full_history.extend(hist)
            _eval_stage("stage2_3", out_dir, analysis_dir, manifest, data_root, val_episodes, batch_size, num_workers, seed, device, full_history=full_history)

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
    p.add_argument("--phase1-frozen-epochs", type=int, default=15)
    p.add_argument("--phase1-partial-epochs", type=int, default=40)
    # Phase 2 epoch counts
    p.add_argument("--phase2-frozen-epochs", type=int, default=10)
    p.add_argument("--phase2-partial-epochs", type=int, default=30)
    p.add_argument("--phase2-full-epochs", type=int, default=15)
    p.add_argument(
        "--phase2-cv-folds",
        type=int,
        default=0,
        help="K-fold cross-validation for Phase 2 (0 disables; e.g. 4 for 4-fold)",
    )
    p.add_argument(
        "--start-stage",
        choices=list(STAGE_ORDER),
        default=None,
        help="skip earlier stages and start fresh from this one",
    )
    p.add_argument("--no-contrastive", action="store_true")
    p.add_argument("--contrastive-weight-p1", type=float, default=0.05)
    p.add_argument("--contrastive-weight-p2", type=float, default=0.1)
    p.add_argument("--contrastive-temp", type=float, default=0.07)
    p.add_argument("--no-vicreg", action="store_true")
    p.add_argument("--vicreg-weight", type=float, default=0.05)
    p.add_argument("--no-barlow", action="store_true")
    p.add_argument("--barlow-weight", type=float, default=0.005)
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
        phase2_cv_folds=args.phase2_cv_folds,
        contrastive=not args.no_contrastive,
        contrastive_weight_p1=args.contrastive_weight_p1,
        contrastive_weight_p2=args.contrastive_weight_p2,
        contrastive_temp=args.contrastive_temp,
        vicreg=not args.no_vicreg,
        vicreg_weight=args.vicreg_weight,
        barlow=not args.no_barlow,
        barlow_weight=args.barlow_weight,
        hard_neg_mining=not args.no_hard_neg,
        start_stage=args.start_stage,
        seed=args.seed,
        device=args.device,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
