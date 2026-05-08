"""Per-epoch prototype cache for hard-negative mining.

Computes one bag-level prototype per instance via a deterministic forward
pass with augmentation disabled. The dataset's ``hard_neg_cache`` is set to
the result so ``EpisodeDataset._sample_query`` can pick hard negatives by
cosine similarity.
"""

from __future__ import annotations

import random as _random

import torch

from modeling.dataset import EpisodeDataset, _Augment, _load_image
from modeling.model import FewShotLocalizer


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
        # Bag-level summary (B, DIM) for cosine-similarity-based hard-neg mining.
        prototypes = model.support_pool.forward_bag(tokens)
        for i, instance in enumerate(batch_instances):
            cache[instance["instance_id"]] = prototypes[i].cpu()
    if was_training:
        model.train()
    return cache
