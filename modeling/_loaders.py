"""DataLoader builders for the OWLv2 trainer."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from modeling.dataset import (
    EpisodeDataset,
    Phase0Dataset,
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
    img_size: int,
    seed: int,
    n_support: int,
) -> tuple[EpisodeDataset, DataLoader]:
    ds = EpisodeDataset(
        manifest_path=manifest,
        data_root=data_root,
        split=split,
        sources=sources,
        episodes_per_epoch=episodes_per_epoch,
        n_support=n_support,
        neg_prob=neg_prob,
        train=True,
        img_size=img_size,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=True,
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
        train=False,
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


def build_phase0_loader(
    *,
    manifest: str,
    data_root: str | None,
    split: str,
    sources: list[str] | None,
    batch_size: int,
    num_workers: int,
    img_size: int,
) -> tuple[Phase0Dataset, DataLoader]:
    ds = Phase0Dataset(
        manifest_path=manifest,
        data_root=data_root,
        split=split,
        sources=sources,
        img_size=img_size,
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
