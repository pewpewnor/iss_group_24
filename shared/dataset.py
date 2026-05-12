"""Episode dataset shared by localizer and siamese.

Returns one episode per __getitem__:
    {
      "support_imgs":  (K_max, 3, S, S)  un-normalized RGB tensors in [0,1].
                       Real supports occupy slots 0..k-1; padding fills the rest.
      "support_mask":  (K_max,)          bool, True where the slot is real.
      "k":             int               actual number of real supports this episode.
      "query_img":     (3, S, S)         un-normalized RGB tensor in [0,1].
      "query_bbox":    (4,)              cxcywh in [0,1] (zeros for negatives).
      "is_present":    bool tensor       True for positive episodes.
      "instance_id":   str
      "source":        str
      "native_size":   (2,) int32        original (W, H) of the query image.
      "native_bbox":   (4,)              original xyxy bbox in native coords (zeros for negatives).
    }

Each model normalizes inside its own forward (OWLv2 = CLIP stats; DINOv2 = different stats),
which is why the dataset returns un-normalized tensors.

Support preprocessing: NO bbox cropping. Letterbox resize (preserve aspect ratio,
pad to img_size with mean colour) + per-view augmentation. The K supports per
episode are sampled with K ~ Uniform{k_min..k_max} at train time, and a
deterministic K from a per-source roundrobin at eval time.

Negative episode generation (for siamese only): query is sampled from a
DIFFERENT instance of the SAME source. With probability hard_neg_prob the
negative is drawn from the per-fold hard-negative cache.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from shared.manifest import filter_instances, load_manifest


DEFAULT_IMG_SIZE = 768
DEFAULT_K_MIN = 1
DEFAULT_K_MAX = 10
DEFAULT_NEG_PROB = 0.5


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------


def _xyxy_to_cxcywh_norm(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))
    # Clamp to [0, img_size] so degenerate / out-of-frame bboxes still produce
    # valid cxcywh in [0, 1].
    x1c = max(0.0, min(float(img_w), float(x1)))
    x2c = max(0.0, min(float(img_w), float(x2)))
    y1c = max(0.0, min(float(img_h), float(y1)))
    y2c = max(0.0, min(float(img_h), float(y2)))
    if x2c < x1c:
        x1c, x2c = x2c, x1c
    if y2c < y1c:
        y1c, y2c = y2c, y1c
    return [
        (x1c + x2c) / 2.0 / img_w,
        (y1c + y2c) / 2.0 / img_h,
        (x2c - x1c) / img_w,
        (y2c - y1c) / img_h,
    ]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _letterbox(
    img: Image.Image, size: int, pad_color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[Image.Image, float, int, int]:
    """Resize keeping aspect ratio, pad to (size, size).

    Returns (output_img, scale, pad_left, pad_top). The scale and pads are
    needed to map a bbox from native coordinates to letterboxed coordinates.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), pad_color)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    out.paste(img_resized, (pad_left, pad_top))
    return out, scale, pad_left, pad_top


def _letterbox_bbox(
    bbox_xyxy: list[float], scale: float, pad_left: int, pad_top: int
) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        x1 * scale + pad_left,
        y1 * scale + pad_top,
        x2 * scale + pad_left,
        y2 * scale + pad_top,
    ]


class _SupportAugment:
    """Per-view stochastic augmentation for support images. No bbox crop."""

    def __init__(self, img_size: int, train: bool, *,
                 color_jitter: float = 0.4, hue: float = 0.1,
                 grayscale_prob: float = 0.2, blur_prob: float = 0.2,
                 blur_sigma: tuple[float, float] = (0.5, 2.0),
                 erase_prob: float = 0.3,
                 erase_scale: tuple[float, float] = (0.05, 0.20),
                 rrc_scale: tuple[float, float] = (0.5, 1.0),
                 hflip_prob: float = 0.5):
        self.img_size = img_size
        self.train = train
        self.color = T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                   saturation=color_jitter, hue=hue)
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.erase_prob = erase_prob
        self.erase_scale = erase_scale
        self.rrc_scale = rrc_scale
        self.hflip_prob = hflip_prob

    def __call__(self, img: Image.Image, rng: random.Random) -> torch.Tensor:
        if self.train:
            # 1) Random horizontal flip.
            if rng.random() < self.hflip_prob:
                img = ImageOps.mirror(img)
            # 2) Random resized crop (acts as implicit weak crop).
            w, h = img.size
            scale = rng.uniform(*self.rrc_scale)
            cw = max(8, int(w * scale))
            ch = max(8, int(h * scale))
            x0 = rng.randint(0, max(w - cw, 0))
            y0 = rng.randint(0, max(h - ch, 0))
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            # 3) Letterbox to img_size.
            img, _, _, _ = _letterbox(img, self.img_size)
            # 4) Color jitter.
            img = self.color(img)
            # 5) Random grayscale.
            if rng.random() < self.grayscale_prob:
                img = ImageOps.grayscale(img).convert("RGB")
            # 6) Random blur.
            if rng.random() < self.blur_prob:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=rng.uniform(*self.blur_sigma)))
            t = TF.to_tensor(img)  # (3, S, S) in [0, 1]
            # 7) Random erasing on tensor.
            if rng.random() < self.erase_prob:
                t = _random_erase(t, self.erase_scale, rng)
            return t
        else:
            img, _, _, _ = _letterbox(img, self.img_size)
            return TF.to_tensor(img)


def _random_erase(
    t: torch.Tensor, scale_range: tuple[float, float], rng: random.Random
) -> torch.Tensor:
    """In-place random erase: replace a random-area patch with mean colour."""
    _, h, w = t.shape
    area = h * w
    target_area = rng.uniform(*scale_range) * area
    aspect = rng.uniform(0.5, 2.0)
    eh = int(round((target_area * aspect) ** 0.5))
    ew = int(round((target_area / aspect) ** 0.5))
    if eh < 1 or ew < 1 or eh > h or ew > w:
        return t
    y0 = rng.randint(0, h - eh)
    x0 = rng.randint(0, w - ew)
    fill = t.mean(dim=(1, 2), keepdim=True).expand(-1, eh, ew)
    t = t.clone()
    t[:, y0:y0 + eh, x0:x0 + ew] = fill
    return t


class _QueryTransform:
    """Mild colour jitter only — no spatial augmentation (preserves bbox)."""

    def __init__(self, img_size: int, train: bool, color_jitter: float = 0.2):
        self.img_size = img_size
        self.train = train
        self.color = T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                   saturation=color_jitter, hue=color_jitter * 0.5)

    def __call__(
        self, img: Image.Image, bbox_xyxy: list[float] | None,
        rng: random.Random | None,
    ) -> tuple[torch.Tensor, list[float] | None, float, int, int]:
        img_lb, scale, pad_left, pad_top = _letterbox(img, self.img_size)
        if self.train and rng is not None:
            img_lb = self.color(img_lb)
        bbox_lb = (
            _letterbox_bbox(bbox_xyxy, scale, pad_left, pad_top)
            if bbox_xyxy is not None else None
        )
        return TF.to_tensor(img_lb), bbox_lb, scale, pad_left, pad_top


# ---------------------------------------------------------------------------
# Episode dataset
# ---------------------------------------------------------------------------


class EpisodeDataset(Dataset):
    """Variable-K (1..10) episode dataset.

    Args:
        manifest_path : path to dataset/aggregated/manifest.json
        split         : "train" | "test" | None
        sources       : optional list of source filters (e.g. ["hots"])
        episodes_per_epoch : number of episodes the dataset reports (train mode)
        k_min, k_max  : inclusive range of supports per episode
        force_positive: if True, every episode is positive (localizer mode)
        neg_prob      : probability of generating a negative episode (siamese mode)
        train         : whether to apply augmentation
        img_size      : output square size
        seed          : RNG seed offset
        return_native : also return query as PIL + native bbox / size
        hard_neg_cache: optional dict mapping (instance_id) -> list of (path, source) tuples
                        used to draw a fraction of negatives from misclassifications
        hard_neg_frac : fraction of negatives that should be hard
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        sources: list[str] | None = None,
        data_root: str | Path | None = None,
        episodes_per_epoch: int = 200,
        k_min: int = DEFAULT_K_MIN,
        k_max: int = DEFAULT_K_MAX,
        force_positive: bool = False,
        neg_prob: float = DEFAULT_NEG_PROB,
        train: bool = True,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int = 0,
        return_native: bool = False,
        hard_neg_cache: dict[str, list[dict]] | None = None,
        hard_neg_frac: float = 0.0,
        # Augmentation knobs (forwarded into _SupportAugment / _QueryTransform).
        aug_color_jitter: float = 0.4,
        aug_hue: float = 0.1,
        aug_grayscale_prob: float = 0.2,
        aug_blur_prob: float = 0.2,
        aug_blur_sigma: tuple[float, float] = (0.5, 2.0),
        aug_erase_prob: float = 0.3,
        aug_erase_scale: tuple[float, float] = (0.05, 0.20),
        aug_rrc_scale: tuple[float, float] = (0.5, 1.0),
        aug_hflip_prob: float = 0.5,
        aug_query_color_jitter: float = 0.2,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root is not None else self.manifest_path.parent
        self.manifest = load_manifest(self.manifest_path)
        self._all_instances = filter_instances(self.manifest, split=split, sources=sources)
        self.instances = list(self._all_instances)

        self.k_min = max(1, int(k_min))
        self.k_max = max(self.k_min, int(k_max))
        self.force_positive = bool(force_positive)
        self.neg_prob = 0.0 if force_positive else float(neg_prob)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.train = bool(train)
        self.img_size = int(img_size)
        self.return_native = bool(return_native)
        self._seed = int(seed)
        self.hard_neg_cache = hard_neg_cache or {}
        self.hard_neg_frac = float(hard_neg_frac)

        self._support_aug = _SupportAugment(
            self.img_size, self.train,
            color_jitter=aug_color_jitter, hue=aug_hue,
            grayscale_prob=aug_grayscale_prob, blur_prob=aug_blur_prob,
            blur_sigma=aug_blur_sigma, erase_prob=aug_erase_prob,
            erase_scale=aug_erase_scale, rrc_scale=aug_rrc_scale,
            hflip_prob=aug_hflip_prob,
        )
        self._query_tf = _QueryTransform(
            self.img_size, self.train, color_jitter=aug_query_color_jitter,
        )

    # --- fold management ---------------------------------------------------

    def set_fold(
        self,
        train_ids: set[str] | None = None,
        val_ids: set[str] | None = None,
    ) -> None:
        if val_ids is not None:
            self.instances = [i for i in self._all_instances if i["instance_id"] in val_ids]
        elif train_ids is not None:
            self.instances = [i for i in self._all_instances if i["instance_id"] in train_ids]
        else:
            self.instances = list(self._all_instances)
        if not self.instances:
            raise RuntimeError(
                "set_fold produced an empty instance list. "
                f"requested train_ids={None if train_ids is None else len(train_ids)}, "
                f"val_ids={None if val_ids is None else len(val_ids)}, "
                f"available={len(self._all_instances)}"
            )

    # --- helpers -----------------------------------------------------------

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.data_root / path

    def _load(self, p: str) -> Image.Image:
        return Image.open(self._resolve(p)).convert("RGB")

    def _sample_k(self, rng: random.Random, n_pool: int) -> int:
        if self.k_min == self.k_max:
            k = self.k_min
        else:
            k = rng.randint(self.k_min, self.k_max)
        return min(k, max(1, n_pool))

    def _sample_supports(self, instance: dict, rng: random.Random, k: int) -> list[dict]:
        pool = instance["support_images"]
        if len(pool) >= k:
            return rng.sample(pool, k)
        return [rng.choice(pool) for _ in range(k)]

    def _sample_query_positive(self, instance: dict, rng: random.Random) -> tuple[Image.Image, list[float]]:
        q = rng.choice(instance["query_images"])
        return self._load(q["path"]), list(q["bbox"])

    def _sample_query_negative(
        self, instance: dict, rng: random.Random,
    ) -> tuple[Image.Image, str]:
        """Return (image, path_used) for a negative query.

        The returned path lets the trainer record actually-misclassified
        negative paths back into ``hard_neg_cache``, so the next epoch can
        oversample these. We skip any cache entries with an empty ``path``
        (these are stale sentinels from older recorders that did not yet
        thread the literal path through).
        """
        # Hard negative branch.
        if self.hard_neg_frac > 0.0 and self.hard_neg_cache and rng.random() < self.hard_neg_frac:
            cache = self.hard_neg_cache.get(instance["instance_id"], [])
            valid = [hn for hn in cache if hn.get("path")]
            if valid:
                hn = rng.choice(valid)
                return self._load(hn["path"]), hn["path"]
        # Same-source other-instance.
        same_source = [
            i for i in self.instances
            if i["instance_id"] != instance["instance_id"]
            and i.get("source") == instance.get("source")
        ]
        if not same_source:
            same_source = [i for i in self.instances if i["instance_id"] != instance["instance_id"]]
        if not same_source:
            other = rng.choice(instance["support_images"])
            return self._load(other["path"]), other["path"]
        other = rng.choice(same_source)
        q = rng.choice(other["query_images"])
        return self._load(q["path"]), q["path"]

    def _build_episode(
        self, instance: dict, rng: random.Random, force_k: int | None = None,
    ) -> dict[str, Any]:
        n_pool = len(instance["support_images"])
        k = force_k if force_k is not None else self._sample_k(rng, n_pool)
        k = max(1, min(k, self.k_max))

        # --- supports ---------------------------------------------------
        sup_entries = self._sample_supports(instance, rng, k)
        sup_imgs = [self._support_aug(self._load(s["path"]), rng) for s in sup_entries]
        # Pad to k_max.
        if len(sup_imgs) < self.k_max:
            pad = torch.zeros(3, self.img_size, self.img_size)
            sup_imgs = sup_imgs + [pad] * (self.k_max - len(sup_imgs))
        sup_t = torch.stack(sup_imgs, dim=0)                   # (k_max, 3, S, S)
        mask = torch.zeros(self.k_max, dtype=torch.bool)
        mask[:k] = True

        # --- query ------------------------------------------------------
        is_negative = (not self.force_positive) and rng.random() < self.neg_prob
        if not is_negative:
            q_img, q_bbox_native = self._sample_query_positive(instance, rng)
            native_w, native_h = q_img.size
            q_t, bbox_lb, _, _, _ = self._query_tf(q_img, q_bbox_native, rng if self.train else None)
            assert bbox_lb is not None
            bbox_norm = _xyxy_to_cxcywh_norm(bbox_lb, self.img_size, self.img_size)
            episode = {
                "support_imgs": sup_t,
                "support_mask": mask,
                "k": k,
                "query_img": q_t,
                "query_bbox": torch.tensor(bbox_norm, dtype=torch.float32),
                "is_present": torch.tensor(True, dtype=torch.bool),
                "instance_id": instance["instance_id"],
                "source": instance.get("source", ""),
                "native_size": torch.tensor([native_w, native_h], dtype=torch.int32),
                "native_bbox": torch.tensor(q_bbox_native, dtype=torch.float32),
                # Positives have no negative path to surface; keep the key
                # so the collate function sees a homogeneous schema.
                "query_path": "",
            }
            if self.return_native:
                episode["query_native"] = q_img
            return episode

        # Negative.
        q_img, q_path = self._sample_query_negative(instance, rng)
        native_w, native_h = q_img.size
        q_t, _, _, _, _ = self._query_tf(q_img, None, rng if self.train else None)
        episode = {
            "support_imgs": sup_t,
            "support_mask": mask,
            "k": k,
            "query_img": q_t,
            "query_bbox": torch.zeros(4, dtype=torch.float32),
            "is_present": torch.tensor(False, dtype=torch.bool),
            "instance_id": instance["instance_id"],
            "source": instance.get("source", ""),
            "native_size": torch.tensor([native_w, native_h], dtype=torch.int32),
            "native_bbox": torch.zeros(4, dtype=torch.float32),
            # ``query_path`` is the literal manifest-relative path of the
            # image that was loaded as the negative query. Negatives only —
            # positives leave this empty. Surfaced so the trainer can record
            # misclassified negative paths into ``hard_neg_cache``.
            "query_path": q_path,
        }
        if self.return_native:
            episode["query_native"] = q_img
        return episode

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.instances:
            raise RuntimeError("EpisodeDataset has no instances after filtering")
        rng = random.Random(self._seed + idx * 9973)
        if self.train:
            instance = rng.choice(self.instances)
        else:
            instance = self.instances[idx % len(self.instances)]
        # Eval mode: deterministic K via roundrobin over (1, 4, k_max) plus filler.
        force_k: int | None = None
        if not self.train:
            k_choices = sorted(set([self.k_min, 4, self.k_max]))
            force_k = k_choices[idx % len(k_choices)]
            force_k = max(1, min(force_k, len(instance["support_images"]), self.k_max))
        return self._build_episode(instance, rng, force_k=force_k)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "support_imgs": torch.stack([b["support_imgs"] for b in batch], dim=0),
        "support_mask": torch.stack([b["support_mask"] for b in batch], dim=0),
        "k":            torch.tensor([b["k"] for b in batch], dtype=torch.int32),
        "query_img":    torch.stack([b["query_img"] for b in batch], dim=0),
        "query_bbox":   torch.stack([b["query_bbox"] for b in batch], dim=0),
        "is_present":   torch.stack([b["is_present"] for b in batch], dim=0),
        "instance_id":  [b["instance_id"] for b in batch],
        "source":       [b["source"] for b in batch],
    }
    if "native_size" in batch[0]:
        out["native_size"] = torch.stack([b["native_size"] for b in batch], dim=0)
    if "native_bbox" in batch[0]:
        out["native_bbox"] = torch.stack([b["native_bbox"] for b in batch], dim=0)
    if "query_native" in batch[0]:
        out["query_native"] = [b["query_native"] for b in batch]
    if "query_path" in batch[0]:
        out["query_path"] = [b["query_path"] for b in batch]
    return out


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------


def build_dataloader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
):
    from torch.utils.data import DataLoader
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )
