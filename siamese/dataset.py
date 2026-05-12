"""Siamese dataset wrappers.

Trains and evaluates on MIXED positive + negative episodes. The siamese
predicts existence_prob ∈ [0, 1] regardless of the localizer.

Training negative-rate is controlled by ``neg_prob`` (default 0.75 to
realise a 1:3 pos:neg ratio). Hard-negative cache is supplied by the
trainer and consulted by the dataset's negative sampler.
"""

from __future__ import annotations

from typing import Any

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
    neg_prob: float,
    img_size: int,
    seed: int,
    k_min: int,
    k_max: int,
    aug_kwargs: dict | None = None,
    hard_neg_cache: dict[str, list[dict]] | None = None,
    hard_neg_frac: float = 0.0,
) -> tuple[EpisodeDataset, "torch.utils.data.DataLoader"]:                  # type: ignore[name-defined]
    aug_kwargs = aug_kwargs or {}
    ds = EpisodeDataset(
        manifest_path=manifest, data_root=data_root,
        split=split, sources=sources,
        episodes_per_epoch=episodes_per_epoch,
        k_min=k_min, k_max=k_max,
        force_positive=False,
        neg_prob=neg_prob,
        train=True,
        img_size=img_size, seed=seed,
        hard_neg_cache=hard_neg_cache, hard_neg_frac=hard_neg_frac,
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
    neg_prob: float,
    img_size: int,
    seed: int,
    k_min: int,
    k_max: int,
) -> tuple[EpisodeDataset, "torch.utils.data.DataLoader"]:                  # type: ignore[name-defined]
    """Validation / test loader.

    Note (manifest v5): every support image on disk is already object-only,
    so this loader does NO runtime bbox-crop augmentation.
    """
    ds = EpisodeDataset(
        manifest_path=manifest, data_root=data_root,
        split=split, sources=sources,
        episodes_per_epoch=val_episodes,
        k_min=k_min, k_max=k_max,
        force_positive=False, neg_prob=neg_prob,
        train=False, img_size=img_size, seed=seed,
    )
    loader = build_dataloader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )
    return ds, loader
