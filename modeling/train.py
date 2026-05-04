"""Three-stage episodic training entrypoint.

Stage 1 (default):       backbone fully frozen, heads at LR 1e-3.
Stage 2 (--stage2):      features[0:7] frozen, features[7:] LR 1e-5, heads LR 5e-4.
Stage 3 (--stage3):      full unfreeze, all params at very low LR.

Each stage is opt-in beyond stage 1. Common workflows:
    # Stage 1 only, from scratch
    python -m modeling.train --no-resume

    # Stage 2 from a previously-trained stage 1 checkpoint
    python -m modeling.train --resume model/stage1.pt --start-stage stage2 --stage2

    # All three stages back-to-back
    python -m modeling.train --no-resume --stage2 --stage3
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
from modeling.evaluate import IOU_THRESHOLDS, _compute_pr_ap, _iou_xyxy
from modeling.loss import _containment_ratio
from modeling.loss import nt_xent_loss, total_loss, triplet_loss, vicreg_loss
from modeling.model import FewShotLocalizer, decode


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


def stage0_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    """Stage 0 = FSOD pretrain on a frozen backbone — same shape as stage 1
    but invoked on a different data source. Kept as a separate function so
    the param-group log line in train_stage clearly identifies it.
    """
    model.backbone.freeze_all()
    return torch.optim.AdamW(
        [{"params": _heads(model), "lr": 1e-3, "weight_decay": 1e-4}]
    )


def stage1_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
    model.backbone.freeze_all()
    return torch.optim.AdamW(
        [{"params": _heads(model), "lr": 1e-3, "weight_decay": 1e-4}]
    )


def stage2_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
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


def stage3_optimizer(model: FewShotLocalizer) -> torch.optim.Optimizer:
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
        # Summarise the (1, K*M, dim) token bag to (dim,) for cosine-similarity
        # hard-negative mining. Mean over tokens is a reasonable per-instance
        # signature even though matching itself is now token-level.
        cache[instance["instance_id"]] = tokens.mean(dim=1).squeeze(0).cpu()

    if was_training:
        model.train()
        model.backbone.eval()

    return cache


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


STAGE_ORDER: tuple[str, ...] = ("stage0", "stage1", "stage2", "stage3")


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
    """Bundle everything needed to resume training mid-stage."""
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
# Loops
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
            # Plain IoU on positive episodes — directly comparable to mAP thresholds.
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
    # IOU_THRESHOLDS[0] is 0.5 → ap_vals[0] is AP@0.5
    val_map_50 = ap_vals[0] if ap_vals else 0.0
    val_map_5095 = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0
    # Persist the full per-IoU breakdown so plot.py can render the COCO-style
    # AP@τ curve over epochs (one line per τ in IOU_THRESHOLDS).
    val_ap_per_iou = {f"{tau:.2f}": ap for tau, ap in zip(IOU_THRESHOLDS, ap_vals)}

    # F1@(IoU≥0.5, score≥0.5): summarises the joint quality of presence +
    # localisation in a single number. TP = positive episode AND iou ≥ 0.5
    # AND score ≥ 0.5; FP = predicted-present but either is_present is False
    # or iou < 0.5; FN = is_present True but missed (low score or low iou).
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
    start_epoch: int = 1,
    scheduler_state: dict | None = None,
    prior_history: list[dict] | None = None,
) -> tuple[float, list[dict]]:
    """Train for ``epochs`` epochs and return ``(best_val_iou, epoch_history)``.

    ``epoch_history`` is a list of per-epoch metric dicts with keys:
    stage, epoch, loss, focal, box, nt_xent, vicreg, val_loss, val_iou,
    val_presence_acc.

    For resumption: ``start_epoch`` skips already-completed epochs in this
    stage, ``scheduler_state`` restores the cosine schedule's step counter,
    and ``prior_history`` is the across-stage history from before this stage
    so checkpoints can persist the full plotting record.
    """
    # LR schedule: linear warmup over the first ~5% of steps (capped at 200)
    # → cosine decay to 1e-7 for the rest. Warmup softens the LR transition
    # at the start of every stage, particularly helpful right after stage 0
    # where the matcher heads suddenly see new domain data and the optimiser
    # would otherwise spike at full LR with stale momentum.
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
        # Apply neg_prob schedule (epoch -> prob; latest applicable wins).
        if neg_prob_schedule and hasattr(train_loader.dataset, "set_neg_prob"):
            applicable = [(e, p) for e, p in neg_prob_schedule.items() if e <= epoch]
            if applicable:
                _, prob = max(applicable)
                train_loader.dataset.set_neg_prob(prob)  # type: ignore[attr-defined]

        # Build prototype cache for hard-negative mining at the start of each epoch.
        if hard_neg_mining and hasattr(train_loader.dataset, "hard_neg_cache"):
            train_loader.dataset.hard_neg_cache = build_proto_cache(  # type: ignore[attr-defined]
                model, train_loader.dataset, device  # type: ignore[arg-type]
            )

        model.train()
        # Keep backbone BN in eval mode — small episodic batches would
        # destabilise running stats if we let them update.
        model.backbone.eval()

        t0 = time.time()
        running = {"loss": 0.0, "focal": 0.0, "box": 0.0, "presence": 0.0, "attn": 0.0, "nt_xent": 0.0, "vicreg": 0.0, "triplet": 0.0}
        for batch in train_loader:
            support_imgs = batch["support_imgs"].to(device)
            support_bboxes = batch["support_bboxes"].to(device)
            query_img = batch["query_img"].to(device)
            gt_bbox = batch["query_bbox"].to(device)
            is_present = batch["is_present"].to(device)

            out = model(support_imgs, query_img, support_bboxes=support_bboxes)
            losses = total_loss(out, gt_bbox, is_present, support_bboxes=support_bboxes)
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

        n = len(train_loader)
        avg = {k: v / max(n, 1) for k, v in running.items()}
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
    stage0_epochs: int = 25,
    stage1_epochs: int = 15,
    stage2_epochs: int = 30,
    stage3_epochs: int = 10,
    stage0: bool = False,
    stage2: bool = False,
    stage3: bool = False,
    contrastive: bool = True,
    contrastive_weight_s1: float = 0.1,   # user: bump from 0.1 → 0.2 once backbone unfreezes
    contrastive_weight_s2: float = 0.2,
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
        "resume": (
            str(resume) if isinstance(resume, (str, Path)) else bool(resume)
        ),
        "episodes_per_epoch": episodes_per_epoch,
        "val_episodes": val_episodes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "stage0_epochs": stage0_epochs,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "stage3_epochs": stage3_epochs,
        "stage0": stage0,
        "stage3": stage3,
        "contrastive": contrastive,
        "contrastive_weight_s1": contrastive_weight_s1,
        "contrastive_weight_s2": contrastive_weight_s2,
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

    # Val loader is built once and reused across stages — val/test are always
    # HOTS+InsDet (the deployment domain). The train loader is rebuilt per
    # stage because stage 0 (FSOD pretrain) and stages 1+ (target domain)
    # need disjoint instance pools, and EpisodeDataset filters the pool at
    # construction time.
    TARGET_SOURCES = ["hots", "insdet"]

    val_ds = EpisodeDataset(
        manifest_path=str(manifest),
        split=val_split,
        data_root=str(data_root) if data_root else None,
        episodes_per_epoch=val_episodes,
        # Fixed 30% negatives in val so val_presence and val_map are meaningful
        # and stable across runs (independent of the train neg_prob schedule).
        neg_prob=0.3,
        train=False,
        seed=seed,
        sources=TARGET_SOURCES,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device_t.type == "cuda"),
    )

    def _build_train_loader(
        sources: list[str],
    ) -> tuple[EpisodeDataset, DataLoader]:
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

    # Default train pool is the target domain. Stage 0 swaps to FSOD-only
    # before training begins.
    train_ds, train_loader = _build_train_loader(TARGET_SOURCES)

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
        loaded = torch.load(
            str(resume_path), map_location=device_t, weights_only=False
        )
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
    # Per-stage trainable counts are logged separately as each stage builds its
    # optimizer (after the corresponding freeze_* call mutates requires_grad).
    print(f"params: total={n_total/1e6:.2f}M")
    print(f"train instances: {len(train_ds.instances)}  val instances: {len(val_ds.instances)}")

    def _log_trainable(stage_name: str) -> None:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {stage_name} trainable params: {n/1e6:.2f}M")

    steps_per_epoch = max(episodes_per_epoch // batch_size, 1)

    best = (
        float(resume_state.get("best_val_iou", 0.0)) if resume_state else 0.0
    )
    full_history: list[dict] = (
        list(resume_state.get("full_history", [])) if resume_state else []
    )

    stage_configs_for_lr: list[dict] = []

    def _resume_for(stage_name: str, configured_epochs: int):
        """Return (skip, start_epoch, opt_state, sched_state) for a stage."""
        # Force-skip any stage earlier than start_stage (e.g. start fresh stage 2
        # using stage 1 weights from a previous run, without re-training stage 1).
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

    if stage0 and stage0_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage0", stage0_epochs)
        stage_configs_for_lr.append({
            "name": "stage0",
            "epochs": stage0_epochs,
            "steps_per_epoch": steps_per_epoch,
            "param_groups": [{"label": "heads", "lr": 1e-3}],
        })
        if skip:
            print("=== Stage 0: skipped (already complete in checkpoint) ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 0: FSOD pretrain (backbone frozen){resume_str} ===")
            # Stage 0 trains the matcher heads on FSOD's broad category-level
            # data. Hard-neg mining off (FSOD-internal hard negs are category
            # confusion which is the wrong objective). Easy negatives only via
            # neg_prob=0.1.
            train_ds, train_loader = _build_train_loader(["fsod"])
            train_ds.set_hard_neg_ratio(0.0)
            opt = stage0_optimizer(model)
            _log_trainable("stage0")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader, device_t,
                stage0_epochs, "stage0", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_s1,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=False,
                neg_prob_schedule={1: 0.1},
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)
            # Switch back to target domain for the remaining stages.
            train_ds, train_loader = _build_train_loader(TARGET_SOURCES)

    if stage1_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage1", stage1_epochs)
        stage_configs_for_lr.append({
            "name": "stage1",
            "epochs": stage1_epochs,
            "steps_per_epoch": steps_per_epoch,
            "param_groups": [{"label": "heads", "lr": 1e-3}],
        })
        if skip:
            print("=== Stage 1: skipped (already complete in checkpoint) ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 1: warmup (backbone frozen){resume_str} ===")
            # Stage 1: ramp neg_prob 0 -> 0.15 over the warmup. The model needs to
            # first learn what objects look like before learning to reject distractors.
            train_ds.set_hard_neg_ratio(0.25)
            opt = stage1_optimizer(model)
            _log_trainable("stage1")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader, device_t,
                stage1_epochs, "stage1", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_s1,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=hard_neg_mining,
                neg_prob_schedule={1: 0.0, 3: 0.15},
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)

    if stage2 and stage2_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage2", stage2_epochs)
        stage_configs_for_lr.append({
            "name": "stage2",
            "epochs": stage2_epochs,
            "steps_per_epoch": steps_per_epoch,
            "param_groups": [
                {"label": "backbone upper", "lr": 1e-5},
                {"label": "heads", "lr": 5e-4},
            ],
        })
        if skip:
            print("=== Stage 2: skipped (already complete in checkpoint) ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 2: partial unfreeze (features[7:] @ 1e-5){resume_str} ===")
            # Stage 2: full negatives + boosted hard-negative ratio. Prototype
            # space is now meaningful, so hard negatives carry useful signal.
            train_ds.set_hard_neg_ratio(0.5)
            opt = stage2_optimizer(model)
            _log_trainable("stage2")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader, device_t,
                stage2_epochs, "stage2", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_s2,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=hard_neg_mining,
                neg_prob_schedule={1: 0.3},
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)

    if stage3 and stage3_epochs > 0:
        skip, start_epoch, opt_state, sched_state = _resume_for("stage3", stage3_epochs)
        stage_configs_for_lr.append({
            "name": "stage3",
            "epochs": stage3_epochs,
            "steps_per_epoch": steps_per_epoch,
            "param_groups": [
                {"label": "backbone lower", "lr": 5e-6},
                {"label": "backbone upper", "lr": 5e-6},
                {"label": "heads", "lr": 1e-4},
            ],
        })
        if skip:
            print("=== Stage 3: skipped (already complete in checkpoint) ===")
        else:
            resume_str = f" — resuming from epoch {start_epoch}" if start_epoch > 1 else ""
            print(f"=== Stage 3: full unfreeze (all backbone @ 5e-6){resume_str} ===")
            train_ds.set_hard_neg_ratio(0.5)
            opt = stage3_optimizer(model)
            _log_trainable("stage3")
            if opt_state is not None:
                opt.load_state_dict(opt_state)
            best, hist = train_stage(
                model, opt, train_loader, val_loader, device_t,
                stage3_epochs, "stage3", out_dir, best,
                contrastive=contrastive,
                contrastive_weight=contrastive_weight_s2,
                contrastive_temp=contrastive_temp,
                vicreg=vicreg,
                vicreg_weight=vicreg_weight,
                triplet=triplet,
                triplet_weight=triplet_weight,
                hard_neg_mining=hard_neg_mining,
                neg_prob_schedule={1: 0.3},
                start_epoch=start_epoch,
                scheduler_state=sched_state,
                prior_history=full_history,
            )
            full_history.extend(hist)

    print(f"done. best val_iou={best:.4f} (saved to {out_dir / 'best.pt'})")  # mAP also in best.pt

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
    """Generate all matplotlib plots after training completes."""
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
    p.add_argument(
        "--data-root",
        default=None,
        help="directory image paths are relative to (default: manifest parent dir)",
    )
    p.add_argument("--out-dir", default="model")
    p.add_argument(
        "--resume",
        default=None,
        help="explicit checkpoint path; default auto-detects <out-dir>/last.pt",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="disable auto-resume — train from scratch",
    )
    p.add_argument("--episodes-per-epoch", type=int, default=1000)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stage0-epochs", type=int, default=25)
    p.add_argument("--stage1-epochs", type=int, default=15)
    p.add_argument("--stage2-epochs", type=int, default=30)
    p.add_argument("--stage3-epochs", type=int, default=10)
    p.add_argument(
        "--start-stage",
        choices=["stage0", "stage1", "stage2", "stage3"],
        default=None,
        help=(
            "skip earlier stages and start fresh from this one. "
            "Pair with --resume model/stage0.pt to continue a prior run "
            "into stage 1 with new optimizer/scheduler state."
        ),
    )
    p.add_argument("--stage0", action="store_true", help="run optional Stage 0 FSOD pretrain (heads only, FSOD-only data)")
    p.add_argument("--stage2", action="store_true", help="run optional Stage 2 partial unfreeze")
    p.add_argument("--stage3", action="store_true", help="run optional Stage 3 full unfreeze")
    p.add_argument("--no-contrastive", action="store_true", help="disable NT-Xent prototype loss")
    p.add_argument("--contrastive-weight-s1", type=float, default=0.1)
    p.add_argument("--contrastive-weight-s2", type=float, default=0.2)
    p.add_argument("--contrastive-temp", type=float, default=0.1)
    p.add_argument("--no-vicreg", action="store_true", help="disable VICReg regularisation")
    p.add_argument("--vicreg-weight", type=float, default=0.05)
    p.add_argument("--no-hard-neg", action="store_true", help="disable hard-negative mining")
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
        stage0_epochs=args.stage0_epochs,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage0=args.stage0,
        stage2=args.stage2,
        stage3=args.stage3,
        contrastive=not args.no_contrastive,
        contrastive_weight_s1=args.contrastive_weight_s1,
        contrastive_weight_s2=args.contrastive_weight_s2,
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
