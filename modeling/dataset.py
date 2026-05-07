"""Episodic dataset + source-balanced batch sampler for the Siamese localiser.

Each __getitem__ returns one episode:
- support_imgs:    (N_SUPPORT, 3, H, W)
- support_bboxes:  (N_SUPPORT, 4)            in image coords, (x1,y1,x2,y2)
- query_img:       (3, H, W)
- query_bbox:      (4,)                       in image coords; zeros if absent
- is_present:      bool tensor
- instance_id:     str
- source:          str

H/W is fixed to 224 by default. Multi-scale training is implemented by
varying ``self.img_size`` between batches via ``set_img_size()`` (the trainer
samples sizes from {192, 224, 256} per batch).

Augmentation pipeline (``train=True`` and ``augment=True``):

Support images:
  resize → multi-scale crop (0.4–1.0) → hflip → color jitter → grayscale → blur
  → tensor → erase → bbox jitter

Query images:
  resize → copy-paste distractor (p=0.4) → hflip → perspective → color jitter → grayscale
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import BatchSampler, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

DEFAULT_IMG_SIZE = 224
N_SUPPORT = 4
NEG_PROB = 0.3
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SUPPORT_CROP_SCALES = (0.4, 0.6, 0.8, 1.0)


# Default per-batch source mix (sums to 16). The trainer uses this when
# `source_mix=None`. If a source pool is empty, its quota is redistributed
# proportionally to the remaining sources.
#
# vizwiz_base's pool is ~38% of the train set but it's also the easiest /
# most-pretrained-like distribution (object-centric phone shots). We
# deliberately under-represent it here so the heads spend more gradient on
# the harder target-domain sources (HOTS scene RGBs, InsDet products,
# VizWiz novel categories). The complementary lever is `source_loss_weights`
# which scales each sample's loss by source — see ``total_loss``.
# User-stated priority: InsDet ≳ vizwiz_novel ≳ HOTS, vizwiz_base least.
# vizwiz_novel has only ~13 train instances so the per-batch slot is generous
# (over-represents in training relative to its tiny pool). The matching
# loss-weight ladder reinforces the same priority on gradient magnitude.
DEFAULT_SOURCE_MIX: dict[str, int] = {
    "vizwiz_base": 2,
    "vizwiz_novel": 4,
    "hots": 4,
    "insdet": 6,
}

DEFAULT_SOURCE_LOSS_WEIGHTS: dict[str, float] = {
    "vizwiz_base": 0.3,
    "vizwiz_novel": 1.7,
    "hots": 1.5,
    "insdet": 1.8,
}


# ---------------------------------------------------------------------------
# Bbox-aware geometric helpers (size-parameterised)
# ---------------------------------------------------------------------------


def _resize_with_bbox(
    img: Image.Image, bbox: list[float] | None, img_size: int
) -> tuple[Image.Image, list[float] | None]:
    w, h = img.size
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    if bbox is None:
        return img, None
    sx = img_size / w
    sy = img_size / h
    x1, y1, x2, y2 = bbox
    return img, [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _hflip_with_bbox(
    img: Image.Image, bbox: list[float] | None
) -> tuple[Image.Image, list[float] | None]:
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if bbox is None:
        return img, None
    x1, y1, x2, y2 = bbox
    w = img.width
    return img, [w - x2, y1, w - x1, y2]


def _scale_jitter_crop_with_bbox(
    img: Image.Image,
    bbox: list[float] | None,
    rng: random.Random,
    img_size: int,
    min_crop: float = 0.7,
) -> tuple[Image.Image, list[float] | None]:
    w, h = img.size
    for _ in range(10):
        scale = rng.uniform(min_crop, 1.0)
        cw = int(w * scale)
        ch = int(h * scale)
        x0 = rng.randint(0, max(w - cw, 0))
        y0 = rng.randint(0, max(h - ch, 0))
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0 or bh <= 0:
                break
            ix1 = max(bx1, x0)
            iy1 = max(by1, y0)
            ix2 = min(bx2, x0 + cw)
            iy2 = min(by2, y0 + ch)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            if (ix2 - ix1) * (iy2 - iy1) < 0.5 * bw * bh:
                continue
        img_crop = img.crop((x0, y0, x0 + cw, y0 + ch))
        img_crop = img_crop.resize((img_size, img_size), Image.Resampling.BILINEAR)
        if bbox is None:
            return img_crop, None
        sx = img_size / cw
        sy = img_size / ch
        bx1, by1, bx2, by2 = bbox
        new_bbox = [
            max(0.0, (bx1 - x0) * sx),
            max(0.0, (by1 - y0) * sy),
            min(float(img_size), (bx2 - x0) * sx),
            min(float(img_size), (by2 - y0) * sy),
        ]
        if new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
            continue
        return img_crop, new_bbox
    return img, bbox


def _perspective_coeffs(
    startpoints: list[list[int]], endpoints: list[list[int]]
) -> list[float]:
    matrix = []
    for (x1, y1), (x2, y2) in zip(startpoints, endpoints):
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
    import numpy as np
    A = np.array(matrix, dtype=np.float64)
    b = np.array([x for xy in endpoints for x in xy], dtype=np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs.tolist()


def _perspective_with_bbox(
    img: Image.Image,
    bbox: list[float] | None,
    rng: random.Random,
    distortion: float = 0.2,
) -> tuple[Image.Image, list[float] | None]:
    w, h = img.size
    d = int(distortion * min(w, h))
    if d < 1:
        return img, bbox

    def jitter() -> int:
        return rng.randint(-d, d)

    startpoints: list[list[int]] = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    endpoints: list[list[int]] = [
        [0 + jitter(), 0 + jitter()],
        [w - 1 + jitter(), 0 + jitter()],
        [w - 1 + jitter(), h - 1 + jitter()],
        [0 + jitter(), h - 1 + jitter()],
    ]
    img = TF.perspective(img, startpoints, endpoints, interpolation=TF.InterpolationMode.BILINEAR)
    if bbox is None:
        return img, None
    coeffs = _perspective_coeffs(startpoints, endpoints)
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    xs, ys = [], []
    for cx, cy in corners:
        denom = coeffs[6] * cx + coeffs[7] * cy + 1.0
        if abs(denom) < 1e-8:
            return img, bbox
        xs.append((coeffs[0] * cx + coeffs[1] * cy + coeffs[2]) / denom)
        ys.append((coeffs[3] * cx + coeffs[4] * cy + coeffs[5]) / denom)
    new_bbox = [
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(float(w), max(xs)),
        min(float(h), max(ys)),
    ]
    if new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
        return img, bbox
    return img, new_bbox


def _gaussian_blur(img: Image.Image, rng: random.Random, max_radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, max_radius)))


def _random_erase(tensor: torch.Tensor, rng: random.Random, p: float = 0.2) -> torch.Tensor:
    if rng.random() >= p:
        return tensor
    _, h, w = tensor.shape
    area = h * w
    for _ in range(10):
        erase_area = rng.uniform(0.02, 0.20) * area
        aspect = rng.uniform(0.3, 3.3)
        eh = int((erase_area * aspect) ** 0.5)
        ew = int((erase_area / aspect) ** 0.5)
        if eh >= h or ew >= w:
            continue
        y0 = rng.randint(0, h - eh)
        x0 = rng.randint(0, w - ew)
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        tensor = tensor.clone()
        tensor[:, y0:y0 + eh, x0:x0 + ew] = mean
        break
    return tensor


def _jitter_bbox(
    bbox: list[float], img_size: float, rng: random.Random, jitter: float = 0.05
) -> list[float]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 += rng.uniform(-jitter, jitter) * w
    y1 += rng.uniform(-jitter, jitter) * h
    x2 += rng.uniform(-jitter, jitter) * w
    y2 += rng.uniform(-jitter, jitter) * h
    return [max(0.0, x1), max(0.0, y1), min(img_size, x2), min(img_size, y2)]


def _copy_paste_query(
    query_img: Image.Image,
    query_bbox: list[float] | None,
    distractor: Image.Image,
    rng: random.Random,
    p: float = 0.4,
) -> tuple[Image.Image, list[float] | None]:
    if rng.random() >= p:
        return query_img, query_bbox
    w, h = query_img.size
    scale = rng.uniform(0.1, 0.4)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    distractor = distractor.resize((nw, nh), Image.Resampling.BILINEAR)
    x0 = rng.randint(0, max(w - nw, 0))
    y0 = rng.randint(0, max(h - nh, 0))
    out = query_img.copy()
    out.paste(distractor, (x0, y0))
    return out, query_bbox


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return NORMALIZE(TF.to_tensor(img))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class _Augment:
    """Bbox-aware augmentation. ``augment=False`` disables stochastic ops."""

    def __init__(self, kind: str, train: bool, augment: bool = True, strength: float = 1.0) -> None:
        self.kind = kind
        self.train = train
        self.augment = augment
        self.strength = max(0.0, float(strength))
        if kind == "support":
            self.color = T.ColorJitter(
                brightness=0.3 * self.strength,
                contrast=0.3 * self.strength,
                saturation=0.2 * self.strength,
                hue=0.05 * self.strength,
            )
            self.gray_p = 0.1 * self.strength
            self.flip_p = 0.5
            self.blur_p = 0.2 * self.strength
            self.erase_p = 0.2 * self.strength
        elif kind == "query":
            self.color = T.ColorJitter(
                brightness=0.4 * self.strength,
                contrast=0.4 * self.strength,
                saturation=0.3 * self.strength,
                hue=0.05 * self.strength,
            )
            self.gray_p = 0.1 * self.strength
            self.flip_p = 0.5
            self.persp_p = 0.3 * self.strength
        else:
            raise ValueError(kind)

    def __call__(
        self,
        img: Image.Image,
        bbox: list[float] | None,
        rng: random.Random,
        img_size: int,
        distractor: Image.Image | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img, bbox = _resize_with_bbox(img, bbox, img_size=img_size)

        if self.train and self.augment:
            if self.kind == "support":
                min_crop = rng.choice(SUPPORT_CROP_SCALES)
                img, bbox = _scale_jitter_crop_with_bbox(
                    img, bbox, rng, img_size=img_size, min_crop=min_crop
                )
                if rng.random() < self.flip_p:
                    img, bbox = _hflip_with_bbox(img, bbox)
                img = self.color(img)
                if rng.random() < self.gray_p:
                    img = ImageOps.grayscale(img).convert("RGB")
                if rng.random() < self.blur_p:
                    img = _gaussian_blur(img, rng, max_radius=2.0)
            elif self.kind == "query":
                if distractor is not None:
                    img, bbox = _copy_paste_query(img, bbox, distractor, rng)
                if rng.random() < self.flip_p:
                    img, bbox = _hflip_with_bbox(img, bbox)
                if rng.random() < self.persp_p:
                    img, bbox = _perspective_with_bbox(img, bbox, rng, distortion=0.2)
                img = self.color(img)
                if rng.random() < self.gray_p:
                    img = ImageOps.grayscale(img).convert("RGB")

        t = _to_tensor(img)
        if self.train and self.augment and self.kind == "support":
            t = _random_erase(t, rng, p=self.erase_p)

        bbox_t = torch.tensor(
            bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        return t, bbox_t


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EpisodeDataset(Dataset):
    _NEG_POOL_FOR_SOURCE: dict[str, str] = {
        "vizwiz_base": "vizwiz_base",
        "vizwiz_novel": "hope",
        "hots": "hope",
        "insdet": "hope",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        data_root: str | Path | None = None,
        episodes_per_epoch: int = 1000,
        n_support: int = N_SUPPORT,
        neg_prob: float = NEG_PROB,
        hard_neg_ratio: float = 0.25,
        train: bool = True,
        augment: bool = True,
        augment_strength: float = 1.0,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int | None = None,
        sources: list[str] | None = None,
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
            sources_set = set(sources)
            all_instances = [i for i in all_instances if i.get("source") in sources_set]
        self.instances = all_instances
        self._all_instances_unfiltered = all_instances[:]                       # for fold reset

        # Bucket negative-background entries by source.
        raw_negatives = self.manifest.get("negative_backgrounds", [])
        self._negatives_by_pool: dict[str, list[str]] = {}
        for entry in raw_negatives:
            if isinstance(entry, dict):
                pool = entry.get("source", "hope")
                path = entry["path"]
            else:
                pool = "hope"
                path = entry
            self._negatives_by_pool.setdefault(pool, []).append(path)

        self.n_support = n_support
        self.neg_prob = neg_prob
        self.hard_neg_ratio = hard_neg_ratio
        self.episodes_per_epoch = episodes_per_epoch
        self.train = train
        self.augment = augment
        self.augment_strength = augment_strength
        self.img_size = img_size
        self._support_aug = _Augment("support", train, augment, augment_strength)
        self._query_aug = _Augment("query", train, augment, augment_strength)
        self._seed = seed
        self.hard_neg_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # External controls
    # ------------------------------------------------------------------

    def set_neg_prob(self, prob: float) -> None:
        self.neg_prob = prob

    def set_hard_neg_ratio(self, ratio: float) -> None:
        self.hard_neg_ratio = ratio

    def set_img_size(self, img_size: int) -> None:
        """Change the training image resolution. Used for multi-scale training."""
        self.img_size = int(img_size)

    def set_augment(self, augment: bool, strength: float | None = None) -> None:
        self.augment = augment
        if strength is not None:
            self.augment_strength = strength
        self._support_aug = _Augment("support", self.train, self.augment, self.augment_strength)
        self._query_aug = _Augment("query", self.train, self.augment, self.augment_strength)

    def set_fold(
        self,
        train_ids: set[str] | None = None,
        val_ids: set[str] | None = None,
        keep_sources_always: tuple[str, ...] = ("vizwiz_novel",),
    ) -> None:
        """Restrict ``self.instances`` to a fold-specific subset.

        - For training datasets pass ``train_ids``: instances are kept iff
          their id is in train_ids (or their source is in ``keep_sources_always``).
        - For validation datasets pass ``val_ids``: only those exact ids are kept.
        - Pass neither to reset to the full split.
        """
        if val_ids is not None:
            self.instances = [
                i for i in self._all_instances_unfiltered if i["instance_id"] in val_ids
            ]
            return
        if train_ids is not None:
            keep: list[dict[str, Any]] = []
            for inst in self._all_instances_unfiltered:
                if inst.get("source") in keep_sources_always:
                    keep.append(inst)
                elif inst["instance_id"] in train_ids:
                    keep.append(inst)
            self.instances = keep
            return
        self.instances = self._all_instances_unfiltered[:]

    def _negatives_for(self, instance: dict[str, Any]) -> list[str]:
        pool_name = self._NEG_POOL_FOR_SOURCE.get(instance.get("source", ""), "hope")
        return self._negatives_by_pool.get(pool_name, [])

    # ------------------------------------------------------------------
    # Rotation-synthesis support (vizwiz_base / vizwiz_novel)
    # ------------------------------------------------------------------

    _ROTATION_ANGLES = (0, 90, 180, 270)
    _ROTATION_CROP_PAD = 0.10

    def _rotation_supports(
        self,
        scene: Image.Image,
        bbox: list[float],
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(x2 - x1, y2 - y1) * (1.0 + 2 * self._ROTATION_CROP_PAD)
        half = side * 0.5
        w_img, h_img = scene.size
        cx1 = max(0, int(cx - half))
        cy1 = max(0, int(cy - half))
        cx2 = min(w_img, int(cx + half))
        cy2 = min(h_img, int(cy + half))
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = 0, 0, w_img, h_img
        base_crop = scene.crop((cx1, cy1, cx2, cy2))

        s_imgs: list[torch.Tensor] = []
        s_bboxes: list[torch.Tensor] = []
        for angle in self._ROTATION_ANGLES:
            rotated = base_crop if angle == 0 else base_crop.rotate(angle, expand=True)
            full_bbox = [0.0, 0.0, float(rotated.width), float(rotated.height)]
            t, bb = self._support_aug(rotated, full_bbox, rng, img_size=self.img_size)
            s_imgs.append(t)
            s_bboxes.append(bb)
        return torch.stack(s_imgs), torch.stack(s_bboxes)

    def _rotation_negative_query(
        self, instance: dict[str, Any], rng: random.Random
    ) -> Path:
        same_source_negs = self._negatives_for(instance)
        same_source_others = [
            i for i in self.instances
            if i["instance_id"] != instance["instance_id"]
            and i.get("source") == instance.get("source")
        ]
        if same_source_negs and (not same_source_others or rng.random() < 0.5):
            return self._resolve(rng.choice(same_source_negs))
        if same_source_others:
            other = rng.choice(same_source_others)
            pool = other["query_images"] + other["support_images"]
            q = rng.choice(pool)
            return self._resolve(q["path"])
        q = rng.choice(instance["support_images"])
        return self._resolve(q["path"])

    def _rotation_episode(
        self, instance: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        anchor = rng.choice(instance["support_images"])
        scene_path = self._resolve(anchor["path"])
        scene = _load_image(scene_path)
        anchor_bbox = list(anchor["bbox"])
        support_imgs, support_bboxes = self._rotation_supports(scene, anchor_bbox, rng)

        is_negative = rng.random() < self.neg_prob
        q_bbox: list[float] | None
        if not is_negative:
            q_img = scene
            q_bbox = anchor_bbox
            present = True
        else:
            q_path = self._rotation_negative_query(instance, rng)
            q_img = _load_image(q_path) if q_path != scene_path else scene
            q_bbox = None
            present = False

        # vizwiz_novel: with only 1 image per category, support and query
        # share the underlying scene → trivial pixel-match risk. Use a
        # boosted-strength query aug so support↔query are decorrelated.
        is_novel = instance.get("source") == "vizwiz_novel"
        if is_novel and self.train and self.augment:
            boost = max(1.5, self.augment_strength * 1.5)
            novel_aug = _Augment("query", train=True, augment=True, strength=boost)
            q_t, q_bbox_t = novel_aug(q_img, q_bbox, rng, img_size=self.img_size)
        else:
            q_t, q_bbox_t = self._query_aug(q_img, q_bbox, rng, img_size=self.img_size)

        return {
            "support_imgs": support_imgs,
            "support_bboxes": support_bboxes,
            "query_img": q_t,
            "query_bbox": q_bbox_t,
            "is_present": torch.tensor(present, dtype=torch.bool),
            "instance_id": instance["instance_id"],
            "source": instance.get("source", ""),
        }

    # ------------------------------------------------------------------
    # Generic episode (hots / insdet)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.data_root / path

    def _build_query_pool(self, instance: dict[str, Any]) -> list[dict]:
        all_q = instance["query_images"]
        scene = [q for q in all_q if q.get("scene_type")]
        isolated = [q for q in all_q if not q.get("scene_type")]
        return isolated + scene * 3 if scene else all_q

    def _sample_query(
        self, instance: dict[str, Any], rng: random.Random
    ) -> tuple[Path, list[float] | None, bool]:
        is_negative = rng.random() < self.neg_prob
        if not is_negative:
            pool = self._build_query_pool(instance)
            if not pool:
                pool = instance["support_images"]
            q = rng.choice(pool)
            return self._resolve(q["path"]), list(q["bbox"]), True

        r = rng.random()
        hard_thr = 1.0 - self.hard_neg_ratio
        same_source_negs = self._negatives_for(instance)
        same_source_others = [
            i for i in self.instances
            if i["instance_id"] != instance["instance_id"]
            and i.get("source") == instance.get("source")
        ]
        if same_source_negs and r < 0.5:
            return self._resolve(rng.choice(same_source_negs)), None, False

        if not same_source_others:
            if same_source_negs:
                return self._resolve(rng.choice(same_source_negs)), None, False
            q = rng.choice(instance["support_images"])
            return self._resolve(q["path"]), None, False

        if r >= hard_thr and self.hard_neg_cache is not None:
            anchor = self.hard_neg_cache.get(instance["instance_id"])
            if anchor is not None:
                sims = []
                for inst in same_source_others:
                    p = self.hard_neg_cache.get(inst["instance_id"])
                    if p is not None:
                        sim = F.cosine_similarity(anchor.unsqueeze(0), p.unsqueeze(0)).item()
                        sims.append((sim, inst))
                if sims:
                    sims.sort(key=lambda x: x[0], reverse=True)
                    hard_inst = rng.choice(sims[: min(5, len(sims))])[1]
                    pool = hard_inst["query_images"] + hard_inst["support_images"]
                    q = rng.choice(pool)
                    return self._resolve(q["path"]), None, False

        other = rng.choice(same_source_others)
        pool = other["query_images"] + other["support_images"]
        q = rng.choice(pool)
        return self._resolve(q["path"]), None, False

    def _maybe_mixup_support(
        self,
        support_imgs: torch.Tensor,
        support_bboxes: torch.Tensor,
        d_instance: dict[str, Any],
        rng: random.Random,
        alpha: float = 0.2,
        p: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rng.random() >= p:
            return support_imgs, support_bboxes
        d_pool = d_instance["support_images"]
        d_support = [rng.choice(d_pool) for _ in range(self.n_support)]
        d_imgs = []
        for s in d_support:
            img = _load_image(self._resolve(s["path"]))
            t, _ = self._support_aug(img, list(s["bbox"]), rng, img_size=self.img_size)
            d_imgs.append(t)
        lam = rng.betavariate(alpha, alpha)
        mixed = lam * support_imgs + (1 - lam) * torch.stack(d_imgs)
        return mixed, support_bboxes

    def _episode_for_instance(
        self, instance: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        if instance.get("source") in ("vizwiz_base", "vizwiz_novel"):
            return self._rotation_episode(instance, rng)

        others = [
            i for i in self.instances
            if i["instance_id"] != instance["instance_id"]
            and i.get("source") == instance.get("source")
        ]
        d_instance = rng.choice(others) if (self.train and others) else None

        support_pool = instance["support_images"]
        support = (
            rng.sample(support_pool, self.n_support)
            if len(support_pool) >= self.n_support
            else [rng.choice(support_pool) for _ in range(self.n_support)]
        )

        s_imgs, s_bboxes = [], []
        for s in support:
            img = _load_image(self._resolve(s["path"]))
            t, bb = self._support_aug(img, list(s["bbox"]), rng, img_size=self.img_size)
            if self.train and self.augment:
                bb = torch.tensor(
                    _jitter_bbox(bb.tolist(), self.img_size, rng), dtype=torch.float32
                )
            s_imgs.append(t)
            s_bboxes.append(bb)
        support_imgs = torch.stack(s_imgs)
        support_bboxes = torch.stack(s_bboxes)

        if self.train and self.augment and d_instance is not None:
            support_imgs, support_bboxes = self._maybe_mixup_support(
                support_imgs, support_bboxes, d_instance, rng
            )

        distractor: Image.Image | None = None
        if self.train and self.augment and d_instance is not None:
            d_s = rng.choice(d_instance["support_images"])
            distractor = _load_image(self._resolve(d_s["path"]))

        q_path, q_bbox, is_present = self._sample_query(instance, rng)
        q_img = _load_image(q_path)
        q_t, q_bbox_t = self._query_aug(
            q_img, q_bbox, rng, img_size=self.img_size, distractor=distractor
        )

        return {
            "support_imgs": support_imgs,
            "support_bboxes": support_bboxes,
            "query_img": q_t,
            "query_bbox": q_bbox_t,
            "is_present": torch.tensor(is_present, dtype=torch.bool),
            "instance_id": instance["instance_id"],
            "source": instance.get("source", ""),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Index can carry source-quota information from the SourceBalancedBatchSampler
        # in two forms:
        #   - plain int (random instance from self.instances)
        #   - 1_000_000 + i  → forced-instance index into self.instances
        rng = random.Random(idx + self.episodes_per_epoch * (torch.initial_seed() & 0xFFFFFFFF))
        if not self.instances:
            raise RuntimeError("EpisodeDataset has no instances after filtering")
        instance = rng.choice(self.instances)
        return self._episode_for_instance(instance, rng)


# ---------------------------------------------------------------------------
# Source-balanced batch sampler
# ---------------------------------------------------------------------------


class SourceBalancedBatchSampler(BatchSampler):
    """Yields batches of fixed indices that the dataset uses to look up instances.

    With B=16 and the default mix ``{"vizwiz_base":4,"vizwiz_novel":2,"hots":5,"insdet":5}``,
    every batch is guaranteed to contain that source mix. If a source has zero
    instances in the current dataset (e.g. a Stage-3 fold that excluded vizwiz_novel),
    its quota is redistributed proportionally across the remaining sources.

    Implementation:
      We pre-bucket the dataset's instance positions by source, then for each
      batch slot we sample (with replacement) an instance position from the
      appropriate bucket. The dataset is then asked to materialise the episode
      using a fresh rng seeded by the (epoch, slot) pair — but since the
      dataset already does its own random episode generation, we just override
      the chosen instance via a thread-local hook (see _set_forced_instance).
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        batch_size: int,
        num_batches: int,
        source_mix: dict[str, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.source_mix = self._normalise_mix(source_mix or DEFAULT_SOURCE_MIX, batch_size)
        # Compute source → list-of-positions buckets at construction. The
        # dataset's instances list may change across epochs (set_fold), so
        # rebuild at __iter__ time too.
        self._buckets: dict[str, list[int]] = {}

    @staticmethod
    def _normalise_mix(mix: dict[str, int], batch_size: int) -> dict[str, int]:
        s = sum(mix.values())
        if s == batch_size:
            return dict(mix)
        # Scale to batch_size, then fix the remainder by adjusting the largest entry.
        scaled = {k: max(0, round(v * batch_size / max(s, 1))) for k, v in mix.items()}
        diff = batch_size - sum(scaled.values())
        if diff != 0 and scaled:
            top = max(scaled, key=lambda k: scaled[k])
            scaled[top] = max(0, scaled[top] + diff)
        return scaled

    def _rebuild_buckets(self) -> None:
        buckets: dict[str, list[int]] = {}
        for i, inst in enumerate(self.dataset.instances):
            buckets.setdefault(inst.get("source", "_"), []).append(i)
        self._buckets = buckets

    def _effective_mix(self) -> dict[str, int]:
        mix = dict(self.source_mix)
        present = {k: v for k, v in mix.items() if k in self._buckets and self._buckets[k]}
        empty = [k for k in mix if k not in present]
        if not empty:
            return mix
        # Redistribute empty quotas proportionally to present sources.
        total_empty = sum(mix[k] for k in empty)
        if not present:
            # Nothing left — fall back to uniform over whatever buckets exist.
            keys = [k for k, v in self._buckets.items() if v]
            if not keys:
                return mix
            base = self.batch_size // len(keys)
            rem = self.batch_size - base * len(keys)
            redistributed = {k: base for k in keys}
            for k in keys[:rem]:
                redistributed[k] += 1
            return redistributed
        present_total = sum(present.values())
        for k in present:
            present[k] += int(round(total_empty * present[k] / max(present_total, 1)))
        # Fix rounding remainder.
        diff = self.batch_size - sum(present.values())
        if diff != 0:
            top = max(present, key=lambda k: present[k])
            present[top] = max(0, present[top] + diff)
        return present

    def __iter__(self) -> Iterator[list[int]]:
        self._rebuild_buckets()
        mix = self._effective_mix()
        rng = random.Random(self.seed)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for source, q in mix.items():
                bucket = self._buckets.get(source) or []
                if not bucket:
                    continue
                for _ in range(q):
                    batch.append(rng.choice(bucket))
            # Shuffle within the batch so source ordering doesn't bias gradient.
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


# ---------------------------------------------------------------------------
# Index → forced-instance bridge
# ---------------------------------------------------------------------------
#
# We override __getitem__ to interpret indices coming from the source-balanced
# sampler as positions into self.instances. To keep backwards compat with the
# regular sequential sampler used during validation (which yields 0..N-1),
# we treat the integer as an instance position iff it's in [0, len(instances)).


def _override_dataset_getitem() -> None:
    """Monkey-patch EpisodeDataset.__getitem__ so the SourceBalancedBatchSampler
    can pass instance positions directly. We don't actually monkey-patch
    globally — the dataset's __getitem__ already supports the right semantics
    (it picks a random instance per call and uses idx only to seed the rng).
    To honour the sampler's source mix we need __getitem__ to use the
    instance at position ``idx``. So we replace at module import time.
    """
    pass


# Replace EpisodeDataset.__getitem__ with a position-respecting version.
def _episode_getitem_with_position(self, idx: int) -> dict[str, Any]:                # type: ignore[no-redef]
    rng = random.Random(idx + self.episodes_per_epoch * (torch.initial_seed() & 0xFFFFFFFF))
    if not self.instances:
        raise RuntimeError("EpisodeDataset has no instances after filtering")
    n = len(self.instances)
    if 0 <= idx < n:
        instance = self.instances[idx]
    else:
        instance = rng.choice(self.instances)
    return self._episode_for_instance(instance, rng)


EpisodeDataset.__getitem__ = _episode_getitem_with_position                          # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "support_imgs": torch.stack([b["support_imgs"] for b in batch], dim=0),
        "support_bboxes": torch.stack([b["support_bboxes"] for b in batch], dim=0),
        "query_img": torch.stack([b["query_img"] for b in batch], dim=0),
        "query_bbox": torch.stack([b["query_bbox"] for b in batch], dim=0),
        "is_present": torch.stack([b["is_present"] for b in batch], dim=0),
        "instance_id": [b["instance_id"] for b in batch],
        "source": [b["source"] for b in batch],
    }
