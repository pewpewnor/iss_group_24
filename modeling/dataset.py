"""Episodic dataset for few-shot localization.

Each __getitem__ returns one episode:
- support_imgs:    (N_SUPPORT, 3, 224, 224)
- support_bboxes:  (N_SUPPORT, 4)            in 224x224 coords, (x1,y1,x2,y2)
- query_img:       (3, 224, 224)
- query_bbox:      (4,)                       in 224x224 coords; zeros if absent
- is_present:      bool tensor
- instance_id:     str

For NEG_PROB of training episodes the query is drawn from a different instance
(or a background image) so the model learns to output low confidence.

Augmentation pipeline (training only)
--------------------------------------
Support images:
  resize → multi-scale crop (0.4–1.0) → hflip → color jitter → grayscale → blur
  → (tensor) → erase → bbox jitter

Query images:
  resize → copy-paste distractor (p=0.4) → hflip → perspective → color jitter → grayscale
"""

from __future__ import annotations

import functools
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMG_SIZE = 224
N_SUPPORT = 4
NEG_PROB = 0.3
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Multi-scale crop: randomly select one of these as the minimum crop fraction.
# Smaller values (0.4) zoom in, training the model on objects at larger apparent scale.
SUPPORT_CROP_SCALES = (0.4, 0.6, 0.8, 1.0)


# ---------------------------------------------------------------------------
# Bbox-aware geometric helpers
# ---------------------------------------------------------------------------


def _resize_with_bbox(
    img: Image.Image, bbox: list[float] | None
) -> tuple[Image.Image, list[float] | None]:
    w, h = img.size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    if bbox is None:
        return img, None
    sx = IMG_SIZE / w
    sy = IMG_SIZE / h
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
    min_crop: float = 0.7,
) -> tuple[Image.Image, list[float] | None]:
    """Random crop to [min_crop, 1.0] of each side, then resize back to IMG_SIZE.

    When a bbox is present the crop is constrained so that at least 50% of the
    bbox area is retained. The bbox is clipped and rescaled accordingly.
    """
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
        img_crop = img_crop.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        if bbox is None:
            return img_crop, None

        sx = IMG_SIZE / cw
        sy = IMG_SIZE / ch
        bx1, by1, bx2, by2 = bbox
        new_bbox = [
            max(0.0, (bx1 - x0) * sx),
            max(0.0, (by1 - y0) * sy),
            min(float(IMG_SIZE), (bx2 - x0) * sx),
            min(float(IMG_SIZE), (by2 - y0) * sy),
        ]
        if new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
            continue
        return img_crop, new_bbox

    return img, bbox


def _perspective_with_bbox(
    img: Image.Image,
    bbox: list[float] | None,
    rng: random.Random,
    distortion: float = 0.2,
) -> tuple[Image.Image, list[float] | None]:
    """Random perspective warp. The bbox corners are transformed through the
    same homography and re-enclosed in an axis-aligned box."""
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


# ---------------------------------------------------------------------------
# Pixel-level helpers (no bbox)
# ---------------------------------------------------------------------------


def _gaussian_blur(img: Image.Image, rng: random.Random, max_radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, max_radius)))


def _random_erase(tensor: torch.Tensor, rng: random.Random, p: float = 0.2) -> torch.Tensor:
    """Erase a random rectangle on a (C, H, W) tensor with the channel mean."""
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
    bbox: list[float],
    img_size: float,
    rng: random.Random,
    jitter: float = 0.05,
) -> list[float]:
    """Perturb each bbox edge by up to ±jitter × side length.

    Trains ROI Align to be robust to imprecise support annotations, matching
    the condition at inference where user crops are never pixel-perfect.
    """
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
    """Paste a scaled distractor object onto the query image.

    The distractor comes from a different instance, training the model to
    ignore visually similar objects that are not the target.
    """
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
    return out, query_bbox  # query_bbox is unchanged — target location is unaffected


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return NORMALIZE(TF.to_tensor(img))


class _Augment:
    """Bbox-aware augmentation.

    Support pipeline (train):
      resize → multi-scale crop → hflip → color jitter → grayscale → blur → tensor → erase

    Query pipeline (train):
      resize → copy-paste distractor → hflip → perspective → color jitter → grayscale
    """

    def __init__(self, kind: str, train: bool) -> None:
        self.kind = kind
        self.train = train
        if kind == "support":
            self.color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            self.gray_p = 0.1
            self.flip_p = 0.5
            self.blur_p = 0.2
            self.erase_p = 0.2
        elif kind == "query":
            self.color = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)
            self.gray_p = 0.1
            self.flip_p = 0.5
            self.persp_p = 0.3
        else:
            raise ValueError(kind)

    def __call__(
        self,
        img: Image.Image,
        bbox: list[float] | None,
        rng: random.Random,
        distractor: Image.Image | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img, bbox = _resize_with_bbox(img, bbox)

        if self.train:
            if self.kind == "support":
                # Multi-scale crop: randomly select the minimum crop fraction.
                # Smaller min_crop → larger apparent object scale in the cropped view.
                min_crop = rng.choice(SUPPORT_CROP_SCALES)
                img, bbox = _scale_jitter_crop_with_bbox(img, bbox, rng, min_crop=min_crop)
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

        if self.train and self.kind == "support":
            t = _random_erase(t, rng, p=self.erase_p)

        bbox_t = torch.tensor(
            bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        return t, bbox_t


@functools.lru_cache(maxsize=None)
def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EpisodeDataset(Dataset):
    # Maps an instance source to the negative-background pool it draws from.
    # vizwiz_base uses a same-domain pool (VizWiz phone shots not used as
    # positives) so Phase 1 negatives can't be rejected by global image style.
    # All Phase 2 sources share the "hope" pool (HOPE + InsDet Background +
    # VizWiz query images), which is fine once the model is past the style-
    # discrimination failure mode.
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

        # Bucket negative-background entries by source. Supports both the new
        # dict-shape entries `{"path": ..., "source": ...}` and the legacy
        # flat list of strings (which all map to the hots_insdet pool — the
        # only domain that pre-VizWiz manifests covered).
        raw_negatives = self.manifest.get("negative_backgrounds", [])
        self._negatives_by_pool: dict[str, list[str]] = {}
        for entry in raw_negatives:
            if isinstance(entry, dict):
                pool = entry.get("source", "hots_insdet")
                path = entry["path"]
            else:
                pool = "hots_insdet"
                path = entry
            self._negatives_by_pool.setdefault(pool, []).append(path)

        self.n_support = n_support
        # Use neg_prob as given for both train and val. Forcing val to 0 made
        # val_presence/val_map meaningless (model trivially says "present" → 100%).
        self.neg_prob = neg_prob
        self.hard_neg_ratio = hard_neg_ratio
        self.episodes_per_epoch = episodes_per_epoch
        self.train = train
        self._support_aug = _Augment("support", train)
        self._query_aug = _Augment("query", train)
        self._seed = seed
        # Populated externally by build_proto_cache() each epoch for hard-negative mining.
        self.hard_neg_cache: dict[str, Any] | None = None

    def _negatives_for(self, instance: dict[str, Any]) -> list[str]:
        """Background-image pool for a given instance, restricted to its source."""
        pool_name = self._NEG_POOL_FOR_SOURCE.get(
            instance.get("source", ""), "hots_insdet"
        )
        return self._negatives_by_pool.get(pool_name, [])

    # ------------------------------------------------------------------
    # Rotation-synthesis support: 4 crops of the bbox at 0/90/180/270°.
    # Used by vizwiz_base and vizwiz_novel sources.
    # ------------------------------------------------------------------

    _ROTATION_ANGLES = (0, 90, 180, 270)
    _ROTATION_CROP_PAD = 0.10  # 10% bbox-side padding on each support crop

    def _rotation_supports(
        self,
        scene: Image.Image,
        bbox: list[float],
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesise 4 support tensors by cropping bbox + rotating four ways.

        The crop is square-padded around the bbox so 90° rotations don't clip
        corners. Each rotated crop runs through the standard support augmentation
        pipeline (colour jitter, blur, erase) independently.
        """
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
            # After cropping + rotating, the object spans the whole image.
            full_bbox = [0.0, 0.0, float(rotated.width), float(rotated.height)]
            t, bb = self._support_aug(rotated, full_bbox, rng)
            s_imgs.append(t)
            s_bboxes.append(bb)
        return torch.stack(s_imgs), torch.stack(s_bboxes)

    def _rotation_negative_query(
        self, instance: dict[str, Any], rng: random.Random
    ) -> Path:
        """Negative query for a rotation-synthesis episode (no double neg-roll).

        Mirrors the negative tiers of _sample_query but skips the positive
        branch since the caller has already committed to a negative episode.
        """
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
        self, idx: int, instance: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """One rotation-synthesis episode: 4 rotated crops → supports, scene → query.

        Positive episode: same (scene, bbox) entry drives both supports and
        the query — turning a category-level annotation into instance-level
        supervision (find the exact object whose crops were just shown).
        Negative episode: query comes from the per-source negative pool.
        """
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

        q_t, q_bbox_t = self._query_aug(q_img, q_bbox, rng)

        return {
            "support_imgs": support_imgs,
            "support_bboxes": support_bboxes,
            "query_img": q_t,
            "query_bbox": q_bbox_t,
            "is_present": torch.tensor(present, dtype=torch.bool),
            "instance_id": instance["instance_id"],
        }

    def set_neg_prob(self, prob: float) -> None:
        self.neg_prob = prob

    def set_hard_neg_ratio(self, ratio: float) -> None:
        self.hard_neg_ratio = ratio

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.data_root / path

    def _build_query_pool(self, instance: dict[str, Any]) -> list[dict]:
        """Oversample scene queries 3× over isolated queries.

        Scene queries (real backgrounds, multiple objects) are harder and more
        representative of inference conditions. Oversampling exposes the model
        to the hard distribution without collecting more data.
        """
        all_q = instance["query_images"]
        scene = [q for q in all_q if q.get("scene_type")]
        isolated = [q for q in all_q if not q.get("scene_type")]
        return isolated + scene * 3 if scene else all_q

    def _sample_query(
        self, instance: dict[str, Any], rng: random.Random
    ) -> tuple[Path, list[float] | None, bool]:
        """Returns (path, bbox-or-None, is_present)."""
        is_negative = rng.random() < self.neg_prob
        if not is_negative:
            pool = self._build_query_pool(instance)
            if not pool:
                pool = instance["support_images"]
            q = rng.choice(pool)
            return self._resolve(q["path"]), list(q["bbox"]), True

        # Negative episode — three tiers based on a secondary roll:
        #   < 0.50                        : easy negative (background image)
        #   0.50 – (1 - hard_neg_ratio)   : random foreign instance
        #   >= (1 - hard_neg_ratio)        : hard negative (most similar prototype)
        # All tiers stay within the instance's source domain so the model
        # can't discriminate by global image style alone.
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
                    sims.sort(reverse=True)
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
        """Linearly interpolate support images with another instance's support.

        Beta(alpha, alpha) with alpha=0.2 is bimodal — most blends are heavily
        weighted toward one instance (λ near 0 or 1), acting as mild contamination
        rather than hard confusion. Forces the projection head to learn smooth,
        well-separated prototype manifolds.
        """
        if rng.random() >= p:
            return support_imgs, support_bboxes
        d_pool = d_instance["support_images"]
        d_support = [rng.choice(d_pool) for _ in range(self.n_support)]
        d_imgs = []
        for s in d_support:
            img = _load_image(self._resolve(s["path"]))
            t, _ = self._support_aug(img, list(s["bbox"]), rng)
            d_imgs.append(t)
        lam = rng.betavariate(alpha, alpha)
        mixed = lam * support_imgs + (1 - lam) * torch.stack(d_imgs)
        return mixed, support_bboxes  # bboxes belong to the main instance

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Per-index independent seed: combines position with worker seed so every
        # (epoch, worker, idx) triple produces a unique, reproducible episode.
        rng = random.Random(idx + self.episodes_per_epoch * torch.initial_seed())

        instance = rng.choice(self.instances)

        # VizWiz base + novel episodes use the rotation-synthesis path: one
        # (scene, bbox) entry → 4 rotated crops as supports, same scene as
        # query. This bypasses the multi-shot pool sampling used by HOTS/InsDet.
        if instance.get("source") in ("vizwiz_base", "vizwiz_novel"):
            return self._rotation_episode(idx, instance, rng)

        # Distractor instance for copy-paste and support mixup — same source
        # as the anchor so we don't blend a HOTS shot with a VizWiz scene.
        others = [
            i for i in self.instances
            if i["instance_id"] != instance["instance_id"]
            and i.get("source") == instance.get("source")
        ]
        d_instance = rng.choice(others) if (self.train and others) else None

        # Sample support images
        support_pool = instance["support_images"]
        support = (
            rng.sample(support_pool, self.n_support)
            if len(support_pool) >= self.n_support
            else [rng.choice(support_pool) for _ in range(self.n_support)]
        )

        s_imgs, s_bboxes = [], []
        for s in support:
            img = _load_image(self._resolve(s["path"]))
            t, bb = self._support_aug(img, list(s["bbox"]), rng)
            if self.train:
                # Bbox jitter: simulate imprecise user-provided support crops
                bb = torch.tensor(_jitter_bbox(bb.tolist(), IMG_SIZE, rng), dtype=torch.float32)
            s_imgs.append(t)
            s_bboxes.append(bb)
        support_imgs = torch.stack(s_imgs)
        support_bboxes = torch.stack(s_bboxes)

        # Support mixup (lazy: loads distractor images only when fired)
        if self.train and d_instance is not None:
            support_imgs, support_bboxes = self._maybe_mixup_support(
                support_imgs, support_bboxes, d_instance, rng
            )

        # Distractor image for copy-paste (1 image, cheap)
        distractor: Image.Image | None = None
        if self.train and d_instance is not None:
            d_s = rng.choice(d_instance["support_images"])
            distractor = _load_image(self._resolve(d_s["path"]))

        # Sample query image
        q_path, q_bbox, is_present = self._sample_query(instance, rng)
        q_img = _load_image(q_path)
        q_t, q_bbox_t = self._query_aug(q_img, q_bbox, rng, distractor=distractor)

        return {
            "support_imgs": support_imgs,
            "support_bboxes": support_bboxes,
            "query_img": q_t,
            "query_bbox": q_bbox_t,
            "is_present": torch.tensor(is_present, dtype=torch.bool),
            "instance_id": instance["instance_id"],
        }


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
    }
