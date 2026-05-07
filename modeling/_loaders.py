"""Dataset + dataloader builders for the trainer.

Train loader uses ``SourceBalancedBatchSampler`` so every batch contains the
configured per-source mix. Val loader is a plain sequential dataloader.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from modeling.dataset import (
    EpisodeDataset,
    SourceBalancedBatchSampler,
    collate,
)


def build_train_loader(
    *,
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


def build_val_loader(
    *,
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
