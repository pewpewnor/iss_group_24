"""Three-stage episodic training entrypoint.

Stage 1 (warmup): backbone fully frozen, heads at LR 1e-3.
Stage 2: features[0:7] frozen, features[7:] LR 1e-5, heads LR 5e-4.
Stage 3 (optional, --stage3): full unfreeze, all params at very low LR.

Run from repo root:
    python -m modeling.train --manifest dataset/cleaned/manifest.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import random as _random

from modeling.dataset import EpisodeDataset, _Augment, _load_image, collate
from modeling.loss import giou, nt_xent_loss, total_loss, vicreg_loss
from modeling.model import FewShotLocalizer, decode


# ---------------------------------------------------------------------------
# Parameter group helpers
# ---------------------------------------------------------------------------


def _heads(model: FewShotLocalizer) -> list:
    return (
        list(model.projection.parameters())
        + list(model.fpn.parameters())
        + list(model.det_head.parameters())
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

        imgs, bboxes = [], []
        for s in samples:
            img = _load_image(dataset._resolve(s["path"]))
            t, bb = aug(img, list(s["bbox"]), rng)
            imgs.append(t)
            bboxes.append(bb)

        support_imgs_t = torch.stack(imgs).unsqueeze(0).to(device)
        support_bboxes_t = torch.stack(bboxes).unsqueeze(0).to(device)
        proto = model.encode_support(support_imgs_t, support_bboxes_t)
        cache[instance["instance_id"]] = proto.squeeze(0).cpu()

    if was_training:
        model.train()
        model.backbone.eval()

    return cache


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: FewShotLocalizer,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses, ious, present_correct, n, n_pos = 0.0, 0.0, 0, 0, 0
    for batch in val_loader:
        support_imgs = batch["support_imgs"].to(device)
        support_bboxes = batch["support_bboxes"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)

        out = model(support_imgs, support_bboxes, query_img)
        loss_dict = total_loss(out, gt_bbox, is_present)
        losses += loss_dict["loss"].item() * gt_bbox.shape[0]

        pred_box, pred_score = decode(out["reg"], out["conf"])
        pred_present = pred_score > 0.5
        present_correct += (pred_present == is_present).sum().item()

        pos = is_present
        if pos.any():
            ious_b = giou(pred_box[pos], gt_bbox[pos])
            ious += ious_b.clamp(min=0).sum().item()
            n_pos += int(pos.sum().item())
        n += gt_bbox.shape[0]
    model.train()
    return {
        "val_loss": losses / max(n, 1),
        "val_iou": ious / max(n_pos, 1),
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
    hard_neg_mining: bool = True,
) -> tuple[float, list[dict]]:
    """Train for ``epochs`` epochs and return ``(best_val_iou, epoch_history)``.

    ``epoch_history`` is a list of per-epoch metric dicts with keys:
    stage, epoch, loss, focal, box, nt_xent, vicreg, val_loss, val_iou,
    val_presence_acc.
    """
    # Cosine LR decay over all steps in this stage — smoother than step decay,
    # prevents oscillation at the end of long stages.
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-7
    )

    epoch_history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # Build prototype cache for hard-negative mining at the start of each epoch.
        if hard_neg_mining and hasattr(train_loader.dataset, "hard_neg_cache"):
            train_loader.dataset.hard_neg_cache = build_proto_cache(
                model, train_loader.dataset, device  # type: ignore[arg-type]
            )

        model.train()
        # Keep backbone BN in eval mode — small episodic batches would
        # destabilise running stats if we let them update.
        model.backbone.eval()

        t0 = time.time()
        running = {"loss": 0.0, "focal": 0.0, "box": 0.0, "nt_xent": 0.0, "vicreg": 0.0}
        for batch in train_loader:
            support_imgs = batch["support_imgs"].to(device)
            support_bboxes = batch["support_bboxes"].to(device)
            query_img = batch["query_img"].to(device)
            gt_bbox = batch["query_bbox"].to(device)
            is_present = batch["is_present"].to(device)

            out = model(support_imgs, support_bboxes, query_img)
            losses = total_loss(out, gt_bbox, is_present)
            loss = losses["loss"]

            nt_val, vr_val = 0.0, 0.0
            if contrastive:
                nt = nt_xent_loss(out["prototype"], temperature=contrastive_temp)
                loss = loss + contrastive_weight * nt
                nt_val = nt.detach().item()
            if vicreg:
                vr = vicreg_loss(out["prototype"])
                loss = loss + vicreg_weight * vr
                vr_val = vr.detach().item()

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
            running["nt_xent"] += nt_val
            running["vicreg"] += vr_val

        n = len(train_loader)
        avg = {k: v / max(n, 1) for k, v in running.items()}
        metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0
        reg_str = (
            f" nt_xent={avg['nt_xent']:.4f} vicreg={avg['vicreg']:.4f}"
            if (contrastive or vicreg)
            else ""
        )
        print(
            f"[{stage_name}] epoch {epoch}/{epochs} "
            f"loss={avg['loss']:.4f} (focal={avg['focal']:.4f} box={avg['box']:.4f}{reg_str}) "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_iou={metrics['val_iou']:.4f} "
            f"val_presence={metrics['val_presence_acc']:.4f} "
            f"time={elapsed:.1f}s"
        )

        epoch_history.append({
            "stage": stage_name,
            "epoch": epoch,
            "loss": avg["loss"],
            "focal": avg["focal"],
            "box": avg["box"],
            "nt_xent": avg["nt_xent"],
            "vicreg": avg["vicreg"],
            "val_loss": metrics["val_loss"],
            "val_iou": metrics["val_iou"],
            "val_presence_acc": metrics["val_presence_acc"],
        })

        if metrics["val_iou"] > best_val_iou:
            best_val_iou = metrics["val_iou"]
            ckpt = {
                "model": model.state_dict(),
                "val_iou": best_val_iou,
                "stage": stage_name,
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (val_iou={best_val_iou:.4f})")

        torch.save(
            {"model": model.state_dict(), "stage": stage_name, "epoch": epoch},
            out_dir / "last.pt",
        )

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
    resume: str | Path | None = None,
    episodes_per_epoch: int = 2000,
    val_episodes: int = 500,
    batch_size: int = 16,
    num_workers: int = 4,
    stage1_epochs: int = 12,
    stage2_epochs: int = 40,
    stage3_epochs: int = 10,
    stage3: bool = False,
    contrastive: bool = True,
    contrastive_weight_s1: float = 0.1,   # user: bump from 0.1 → 0.2 once backbone unfreezes
    contrastive_weight_s2: float = 0.2,
    contrastive_temp: float = 0.1,
    vicreg: bool = True,
    vicreg_weight: float = 0.05,
    hard_neg_mining: bool = True,
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
        "resume": str(resume) if resume else None,
        "episodes_per_epoch": episodes_per_epoch,
        "val_episodes": val_episodes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "stage3_epochs": stage3_epochs,
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

    train_ds = EpisodeDataset(
        manifest_path=str(manifest),
        split=train_split,
        data_root=str(data_root) if data_root else None,
        episodes_per_epoch=episodes_per_epoch,
        train=True,
    )
    val_ds = EpisodeDataset(
        manifest_path=str(manifest),
        split=val_split,
        data_root=str(data_root) if data_root else None,
        episodes_per_epoch=val_episodes,
        train=False,
        seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device_t.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device_t.type == "cuda"),
    )

    model = FewShotLocalizer(pretrained=pretrained).to(device_t)

    if resume:
        state = torch.load(str(resume), map_location=device_t)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"resumed from {resume}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: total={n_total/1e6:.2f}M trainable(initial)={n_train/1e6:.2f}M")
    print(f"train instances: {len(train_ds.instances)}  val instances: {len(val_ds.instances)}")

    steps_per_epoch = max(episodes_per_epoch // batch_size, 1)

    best = 0.0
    full_history: list[dict] = []

    stage_configs_for_lr: list[dict] = []

    if stage1_epochs > 0:
        print("=== Stage 1: warmup (backbone frozen) ===")
        opt = stage1_optimizer(model)
        stage_configs_for_lr.append({
            "name": "stage1",
            "epochs": stage1_epochs,
            "steps_per_epoch": steps_per_epoch,
            "param_groups": [{"label": "heads", "lr": 1e-3}],
        })
        best, hist = train_stage(
            model, opt, train_loader, val_loader, device_t,
            stage1_epochs, "stage1", out_dir, best,
            contrastive=contrastive,
            contrastive_weight=contrastive_weight_s1,
            contrastive_temp=contrastive_temp,
            vicreg=vicreg,
            vicreg_weight=vicreg_weight,
            hard_neg_mining=hard_neg_mining,
        )
        full_history.extend(hist)

    if stage2_epochs > 0:
        print("=== Stage 2: partial unfreeze (features[7:] @ 1e-5) ===")
        opt = stage2_optimizer(model)
        stage_configs_for_lr.append({
            "name": "stage2",
            "epochs": stage2_epochs,
            "steps_per_epoch": steps_per_epoch,
            "param_groups": [
                {"label": "backbone upper", "lr": 1e-5},
                {"label": "heads", "lr": 5e-4},
            ],
        })
        best, hist = train_stage(
            model, opt, train_loader, val_loader, device_t,
            stage2_epochs, "stage2", out_dir, best,
            contrastive=contrastive,
            contrastive_weight=contrastive_weight_s2,
            contrastive_temp=contrastive_temp,
            vicreg=vicreg,
            vicreg_weight=vicreg_weight,
            hard_neg_mining=hard_neg_mining,
        )
        full_history.extend(hist)

    if stage3 and stage3_epochs > 0:
        print("=== Stage 3: full unfreeze (all backbone @ 5e-6) ===")
        opt = stage3_optimizer(model)
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
        best, hist = train_stage(
            model, opt, train_loader, val_loader, device_t,
            stage3_epochs, "stage3", out_dir, best,
            contrastive=contrastive,
            contrastive_weight=contrastive_weight_s2,
            contrastive_temp=contrastive_temp,
            vicreg=vicreg,
            vicreg_weight=vicreg_weight,
            hard_neg_mining=hard_neg_mining,
        )
        full_history.extend(hist)

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
    """Generate all matplotlib plots after training completes."""
    from modeling.plot import (
        plot_contrastive_learning,
        plot_dataset_stats,
        plot_lr_schedule,
        plot_prototype_similarity,
        plot_training_curves,
    )

    if full_history:
        plot_training_curves(full_history, analysis_dir)
        plot_contrastive_learning(full_history, analysis_dir)

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
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--episodes-per-epoch", type=int, default=1000)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stage1-epochs", type=int, default=8)
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--stage3-epochs", type=int, default=10)
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

    train(
        manifest=args.manifest,
        train_split=args.train_split,
        val_split=args.val_split,
        data_root=args.data_root,
        out_dir=args.out_dir,
        resume=args.resume,
        episodes_per_epoch=args.episodes_per_epoch,
        val_episodes=args.val_episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage3=args.stage3,
        contrastive=not args.no_contrastive,
        contrastive_weight_s1=args.contrastive_weight_s1,
        contrastive_weight_s2=args.contrastive_weight_s2,
        contrastive_temp=args.contrastive_temp,
        vicreg=not args.no_vicreg,
        vicreg_weight=args.vicreg_weight,
        hard_neg_mining=not args.no_hard_neg,
        seed=args.seed,
        device=args.device,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
