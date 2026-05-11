"""Localizer dataset wrappers.

The localizer trains and evaluates on POSITIVE episodes only — the existence
question belongs to the siamese model.
"""

from __future__ import annotations

from pathlib import Path

from shared.dataset import EpisodeDataset, build_dataloader


def build_train_loader(
    *,
    manifest: str,
    data_root: str | None,
    split: str,
    sources: list[str] | None,
    episodes_per_epoch: int,
    batch_size: int,
    num_workers: int,
    img_size: int,
    seed: int,
    k_min: int,
    k_max: int,
    aug_kwargs: dict | None = None,
) -> tuple[EpisodeDataset, "torch.utils.data.DataLoader"]:                  # type: ignore[name-defined]
    aug_kwargs = aug_kwargs or {}
    ds = EpisodeDataset(
        manifest_path=manifest, data_root=data_root,
        split=split, sources=sources,
        episodes_per_epoch=episodes_per_epoch,
        k_min=k_min, k_max=k_max,
        force_positive=True,                      # localizer = positives only
        train=True,
        img_size=img_size, seed=seed,
        **aug_kwargs,
    )
    loader = build_dataloader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, drop_last=True,
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
    img_size: int,
    seed: int,
    k_min: int,
    k_max: int,
    return_native: bool = False,
) -> tuple[EpisodeDataset, "torch.utils.data.DataLoader"]:                  # type: ignore[name-defined]
    ds = EpisodeDataset(
        manifest_path=manifest, data_root=data_root,
        split=split, sources=sources,
        episodes_per_epoch=val_episodes,
        k_min=k_min, k_max=k_max,
        force_positive=True,                      # localizer = positives only
        train=False,
        img_size=img_size, seed=seed,
        return_native=return_native,
    )
    loader = build_dataloader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )
    return ds, loader
