"""Episode datasets for OWLv2 few-shot localizer.

Two dataset classes:

  ``EpisodeDataset`` — main training/eval dataset for HOTS+InsDet.  Each
  __getitem__ returns one episode of (4 supports, 1 query, gt bbox,
  is_present).  Negative episodes (is_present=False) are constructed
  from same-source other-instance queries.

  ``Phase0Dataset`` — special-case dataset for vizwiz_novel.  Each instance
  has only 1 image, so 4 supports are produced via rotation synthesis
  (0°/90°/180°/270°) of a tight crop around the bbox.  Always positive.

Episode contract (returned dict):
  support_imgs : (4, 3, S, S)   normalised to OWLv2 mean/std
  query_img    : (3, S, S)
  query_bbox   : (4,) normalised cxcywh in [0,1]
  is_present   : bool tensor scalar
  instance_id  : str
  source       : str
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF


# OWLv2 normalisation constants (CLIP).
OWLV2_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWLV2_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_IMG_SIZE = 768
N_SUPPORT = 4
DEFAULT_NEG_PROB = 0.5


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------


def _resize_xyxy_bbox(
    bbox: list[float] | None, src_size: tuple[int, int], dst_size: int
) -> list[float] | None:
    if bbox is None:
        return None
    sw, sh = src_size
    sx = dst_size / sw
    sy = dst_size / sh
    x1, y1, x2, y2 = bbox
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _xyxy_to_cxcywh_norm(bbox: list[float], img_size: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return [cx, cy, w, h]


def _hflip_xyxy(bbox: list[float], img_size: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [img_size - x2, y1, img_size - x1, y2]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


_NORMALIZE = T.Normalize(mean=OWLV2_MEAN, std=OWLV2_STD)


def _to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    return _NORMALIZE(TF.to_tensor(img))


class _SupportAugment:
    """Per-view stochastic augmentation for support images.

    No bbox preservation needed — we don't use support bboxes at training
    time.  Output is always a normalised (3, S, S) tensor.
    """

    def __init__(self, img_size: int, train: bool):
        self.img_size = img_size
        self.train = train
        self.color = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    def __call__(self, img: Image.Image, rng: random.Random) -> torch.Tensor:
        if self.train:
            if rng.random() < 0.5:
                img = ImageOps.mirror(img)
            # Random resized crop (scale 0.7–1.0)
            w, h = img.size
            for _ in range(5):
                scale = rng.uniform(0.7, 1.0)
                cw = int(w * scale)
                ch = int(h * scale)
                if cw <= 0 or ch <= 0:
                    continue
                x0 = rng.randint(0, max(w - cw, 0))
                y0 = rng.randint(0, max(h - ch, 0))
                img = img.crop((x0, y0, x0 + cw, y0 + ch))
                break
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            img = self.color(img)
            if rng.random() < 0.2:
                img = ImageOps.grayscale(img).convert("RGB")
            if rng.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))
        else:
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        return _to_normalized_tensor(img)


class _QueryTransform:
    """Mild colour jitter only — no spatial augmentation to preserve bbox."""

    def __init__(self, img_size: int, train: bool):
        self.img_size = img_size
        self.train = train
        self.color = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __call__(
        self,
        img: Image.Image,
        bbox_xyxy: list[float] | None,
        rng: random.Random | None = None,
    ) -> tuple[torch.Tensor, list[float] | None]:
        src_size = img.size
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        if self.train and rng is not None:
            img = self.color(img)
        bbox = _resize_xyxy_bbox(bbox_xyxy, src_size, self.img_size)
        return _to_normalized_tensor(img), bbox


# ---------------------------------------------------------------------------
# Main episodic dataset (HOTS + InsDet)
# ---------------------------------------------------------------------------


class EpisodeDataset(Dataset):
    """Episodic dataset: 4 supports + 1 query per __getitem__.

    Args:
        manifest_path : path to dataset/aggregated/manifest.json
        split         : one of {"train", "test"} (or None to use all instances)
        data_root     : directory containing the staged splits (defaults to
                        manifest's parent directory)
        episodes_per_epoch : number of episodes per epoch (random per call)
        n_support     : 4
        neg_prob      : probability of generating a negative episode
        train         : if True, apply stochastic augmentation
        img_size      : edge length S — supports/query resized to (S, S)
        seed          : optional RNG seed offset
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        data_root: str | Path | None = None,
        episodes_per_epoch: int = 200,
        n_support: int = N_SUPPORT,
        neg_prob: float = DEFAULT_NEG_PROB,
        train: bool = True,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int | None = None,
        sources: list[str] | None = None,
        return_native: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = (
            Path(data_root) if data_root is not None else self.manifest_path.parent
        )
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        all_instances: list[dict[str, Any]] = self.manifest["instances"]
        if split is not None:
            all_instances = [i for i in all_instances if i.get("split") == split]
        if sources is not None:
            ssrc = set(sources)
            all_instances = [i for i in all_instances if i.get("source") in ssrc]
        self._all_instances = all_instances[:]
        self.instances = all_instances[:]

        self.n_support = n_support
        self.neg_prob = neg_prob
        self.episodes_per_epoch = episodes_per_epoch
        self.train = train
        self.img_size = img_size
        self._support_aug = _SupportAugment(img_size, train)
        self._query_tf = _QueryTransform(img_size, train)
        self._seed = seed if seed is not None else 0
        self.return_native = return_native

    # --- fold management ----------------------------------------------------

    def set_fold(
        self,
        train_ids: set[str] | None = None,
        val_ids: set[str] | None = None,
    ) -> None:
        """Restrict ``self.instances`` to train_ids xor val_ids.  If both
        are None, restore the full set.
        """
        if val_ids is not None:
            self.instances = [
                i for i in self._all_instances if i["instance_id"] in val_ids
            ]
        elif train_ids is not None:
            self.instances = [
                i for i in self._all_instances if i["instance_id"] in train_ids
            ]
        else:
            self.instances = self._all_instances[:]

    # --- helpers ------------------------------------------------------------

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.data_root / path

    def _load(self, p: str) -> Image.Image:
        return Image.open(self._resolve(p)).convert("RGB")

    def _sample_supports(
        self, instance: dict[str, Any], rng: random.Random
    ) -> list[dict]:
        pool = instance["support_images"]
        if len(pool) >= self.n_support:
            return rng.sample(pool, self.n_support)
        return [rng.choice(pool) for _ in range(self.n_support)]

    def _sample_query_positive(
        self, instance: dict[str, Any], rng: random.Random
    ) -> tuple[Image.Image, list[float]]:
        q = rng.choice(instance["query_images"])
        return self._load(q["path"]), list(q["bbox"])

    def _sample_query_negative(
        self, instance: dict[str, Any], rng: random.Random
    ) -> Image.Image:
        same_source = [
            i for i in self.instances
            if i["instance_id"] != instance["instance_id"]
            and i.get("source") == instance.get("source")
        ]
        if not same_source:
            # Fall back to any other instance.
            same_source = [
                i for i in self.instances if i["instance_id"] != instance["instance_id"]
            ]
        if not same_source:
            # Last resort: use a different support image of the same instance.
            other = rng.choice(instance["support_images"])
            return self._load(other["path"])
        other = rng.choice(same_source)
        q = rng.choice(other["query_images"])
        return self._load(q["path"])

    def _episode(
        self, instance: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        # Supports.
        supports = self._sample_supports(instance, rng)
        s_imgs = [self._support_aug(self._load(s["path"]), rng) for s in supports]
        support_t = torch.stack(s_imgs, dim=0)                       # (4, 3, S, S)

        # Query: positive or negative.
        is_negative = rng.random() < self.neg_prob if self.train else False
        if not is_negative:
            q_img, q_bbox_xyxy = self._sample_query_positive(instance, rng)
            native_w, native_h = q_img.size
            native_bbox = list(q_bbox_xyxy)
            q_t, bbox_resized = self._query_tf(q_img, q_bbox_xyxy, rng if self.train else None)
            assert bbox_resized is not None
            bbox_norm = _xyxy_to_cxcywh_norm(bbox_resized, self.img_size)
            episode = {
                "support_imgs": support_t,
                "query_img": q_t,
                "query_bbox": torch.tensor(bbox_norm, dtype=torch.float32),
                "is_present": torch.tensor(True, dtype=torch.bool),
                "instance_id": instance["instance_id"],
                "source": instance.get("source", ""),
                "native_size": torch.tensor([native_w, native_h], dtype=torch.int32),
                "native_bbox": torch.tensor(native_bbox, dtype=torch.float32),
            }
            if self.return_native:
                episode["query_native"] = q_img
            return episode
        else:
            q_img = self._sample_query_negative(instance, rng)
            native_w, native_h = q_img.size
            q_t, _ = self._query_tf(q_img, None, rng if self.train else None)
            episode = {
                "support_imgs": support_t,
                "query_img": q_t,
                "query_bbox": torch.zeros(4, dtype=torch.float32),
                "is_present": torch.tensor(False, dtype=torch.bool),
                "instance_id": instance["instance_id"],
                "source": instance.get("source", ""),
                "native_size": torch.tensor([native_w, native_h], dtype=torch.int32),
                "native_bbox": torch.zeros(4, dtype=torch.float32),
            }
            if self.return_native:
                episode["query_native"] = q_img
            return episode

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self._seed + idx * 9973)
        if not self.instances:
            raise RuntimeError("EpisodeDataset has no instances after filtering")
        # Round-robin instance pick within the requested episode index so val
        # is deterministic; train uses random pick within the per-episode rng.
        if self.train:
            instance = rng.choice(self.instances)
        else:
            instance = self.instances[idx % len(self.instances)]
        return self._episode(instance, rng)


# ---------------------------------------------------------------------------
# Phase 0 dataset (vizwiz_novel rotation synthesis)
# ---------------------------------------------------------------------------


class Phase0Dataset(Dataset):
    """One episode per vizwiz_novel instance.  Always positive.

    Supports: 4 rotated crops (0°/90°/180°/270°) of the tight crop around
    the bbox in the single labeled image.
    Query: the original image with the original bbox.
    """

    _ROTATIONS = (0, 90, 180, 270)
    _CROP_PAD = 0.10

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        img_size: int = DEFAULT_IMG_SIZE,
        split: str = "phase0",
        sources: list[str] | None = None,
        return_native: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = (
            Path(data_root) if data_root is not None else self.manifest_path.parent
        )
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        instances: list[dict[str, Any]] = self.manifest["instances"]
        instances = [i for i in instances if i.get("split") == split]
        if sources is not None:
            ssrc = set(sources)
            instances = [i for i in instances if i.get("source") in ssrc]
        self.instances = instances
        self.img_size = img_size
        self._support_aug = _SupportAugment(img_size, train=False)
        self._query_tf = _QueryTransform(img_size, train=False)
        self.return_native = return_native

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.data_root / path

    def _load(self, p: str) -> Image.Image:
        return Image.open(self._resolve(p)).convert("RGB")

    def _rotated_supports(
        self, img: Image.Image, bbox_xyxy: list[float]
    ) -> list[Image.Image]:
        x1, y1, x2, y2 = bbox_xyxy
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(x2 - x1, y2 - y1) * (1.0 + 2 * self._CROP_PAD)
        half = side * 0.5
        w_img, h_img = img.size
        cx1 = max(0, int(cx - half))
        cy1 = max(0, int(cy - half))
        cx2 = min(w_img, int(cx + half))
        cy2 = min(h_img, int(cy + half))
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = 0, 0, w_img, h_img
        base = img.crop((cx1, cy1, cx2, cy2))
        out = []
        for angle in self._ROTATIONS:
            out.append(base if angle == 0 else base.rotate(angle, expand=True))
        return out

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        instance = self.instances[idx]
        # vizwiz_novel: 1 support entry, 1 query entry, both same image+bbox.
        s_entry = instance["support_images"][0]
        q_entry = instance["query_images"][0]
        s_img = self._load(s_entry["path"])
        s_bbox = list(s_entry["bbox"])
        rotated = self._rotated_supports(s_img, s_bbox)
        rng = random.Random(idx)
        s_t = torch.stack(
            [self._support_aug(r, rng) for r in rotated], dim=0
        )                                                            # (4, 3, S, S)

        q_img = self._load(q_entry["path"])
        native_w, native_h = q_img.size
        q_t, bbox_resized = self._query_tf(q_img, list(q_entry["bbox"]))
        assert bbox_resized is not None
        bbox_norm = _xyxy_to_cxcywh_norm(bbox_resized, self.img_size)
        episode = {
            "support_imgs": s_t,
            "query_img": q_t,
            "query_bbox": torch.tensor(bbox_norm, dtype=torch.float32),
            "is_present": torch.tensor(True, dtype=torch.bool),
            "instance_id": instance["instance_id"],
            "source": instance.get("source", ""),
            "native_size": torch.tensor([native_w, native_h], dtype=torch.int32),
            "native_bbox": torch.tensor(list(q_entry["bbox"]), dtype=torch.float32),
        }
        if self.return_native:
            episode["query_native"] = q_img
        return episode


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "support_imgs": torch.stack([b["support_imgs"] for b in batch], dim=0),
        "query_img": torch.stack([b["query_img"] for b in batch], dim=0),
        "query_bbox": torch.stack([b["query_bbox"] for b in batch], dim=0),
        "is_present": torch.stack([b["is_present"] for b in batch], dim=0),
        "instance_id": [b["instance_id"] for b in batch],
        "source": [b["source"] for b in batch],
    }
    if "native_size" in batch[0]:
        out["native_size"] = torch.stack([b["native_size"] for b in batch], dim=0)
    if "native_bbox" in batch[0]:
        out["native_bbox"] = torch.stack([b["native_bbox"] for b in batch], dim=0)
    if "query_native" in batch[0]:
        # PIL images can't be stacked — keep as a list.
        out["query_native"] = [b["query_native"] for b in batch]
    return out
