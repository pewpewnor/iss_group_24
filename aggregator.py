"""Aggregate raw datasets into a clean staging directory + manifest.

Output layout:
    dataset/aggregated/
    ├── manifest.json
    ├── stats.json
    ├── train/{support,query}/<inst_id>/<NNN>.{png,jpg}
    └── test/{support,query}/<inst_id>/<NNN>.{png,jpg}

Changes:
    * Uses rembg + ISNet for foreground extraction
    * Very tight support crops
    * Tiny edge padding
    * Offline after first rembg model download
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 6

SEED = 42
TRAIN_RATIO = 0.80

N_SUPPORT_MIN = 4

# Small bbox padding for generic bbox extraction
BBOX_PAD_FRAC = 0.01

# VERY tight crop padding for saved support images
SUPPORT_CROP_PAD_FRAC = 0.005

BBOX_MIN_SIDE = 20
BBOX_MIN_AREA_FRAC = 0.005

RMBG_MASK_THRESHOLD = 0.5

# Better than u2net
RMBG_MODEL = "isnet-general-use"

SOURCES_CROP_SUPPORTS_ON_DISK: tuple[str, ...] = ("insdet",)

BASE_DIR = Path("dataset/original")
OUT_DIR = Path("dataset/aggregated")

HOTS_OBJECT_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "object"
HOTS_SCENE_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "scene"

INSDET_DIR = BASE_DIR / "InsDet"

_RMBG_SINGLETON: "RMBGCropper | None" = None

# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------


def _largest_component_bbox(mask: np.ndarray) -> list[int] | None:
    """Return tight bbox around largest connected component."""

    if not mask.any():
        return None

    labels, n = ndimage.label(mask)

    if n == 0:
        return None

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest = int(sizes.argmax())

    ys, xs = np.where(labels == largest)

    x1 = int(xs.min())
    y1 = int(ys.min())

    # PIL crop uses exclusive max coords
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return [x1, y1, x2, y2]


def _pad_bbox(
    bbox: list[int],
    img_size: tuple[int, int],
    frac: float,
) -> list[int]:
    w, h = img_size

    x1, y1, x2, y2 = bbox

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = max(1, int(round(bw * frac)))
    py = max(1, int(round(bh * frac)))

    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    ]


def _bbox_valid(
    bbox: list[int],
    img_size: tuple[int, int],
) -> bool:
    w, h = img_size

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    if bw < BBOX_MIN_SIDE or bh < BBOX_MIN_SIDE:
        return False

    return (bw * bh) / (w * h) >= BBOX_MIN_AREA_FRAC


def _bbox_from_mask(mask_path: Path) -> list[int] | None:
    arr = (
        np.array(
            Image.open(mask_path).convert("L"),
        )
        > 127
    )

    bb = _largest_component_bbox(arr)

    if bb is None:
        return None

    h, w = arr.shape

    bb = _pad_bbox(
        bb,
        (w, h),
        frac=BBOX_PAD_FRAC,
    )

    return bb if _bbox_valid(bb, (w, h)) else None


def _bbox_from_image(
    img_path: Path,
    bg_thresh: int = 240,
) -> list[int] | None:
    img = np.array(
        Image.open(img_path).convert("RGB"),
    )

    fg = ~np.all(img > bg_thresh, axis=2)

    bb = _largest_component_bbox(fg)

    if bb is None:
        return None

    h, w = fg.shape

    bb = _pad_bbox(
        bb,
        (w, h),
        frac=BBOX_PAD_FRAC,
    )

    return bb if _bbox_valid(bb, (w, h)) else None


# ---------------------------------------------------------------------------
# XML utilities
# ---------------------------------------------------------------------------


def _int_child(el: ET.Element, tag: str) -> int | None:
    child = el.find(tag)

    if child is None or child.text is None:
        return None

    try:
        return int(float(child.text))
    except ValueError:
        return None


def _parse_voc_xml(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()

    out: list[dict] = []

    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")

        if name_el is None or name_el.text is None or bb is None:
            continue

        x1 = _int_child(bb, "xmin")
        y1 = _int_child(bb, "ymin")
        x2 = _int_child(bb, "xmax")
        y2 = _int_child(bb, "ymax")

        if None in (x1, y1, x2, y2):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        out.append(
            {
                "name": name_el.text.strip(),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return out


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Dataset collectors
# ---------------------------------------------------------------------------


def collect_hots_instances() -> list[dict]:
    train_dir = HOTS_OBJECT_DIR / "train"
    test_dir = HOTS_OBJECT_DIR / "test"

    out: list[dict] = []

    if not train_dir.exists():
        print(f"[skip] missing {train_dir}")
        return out

    for cls_dir in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        cls = cls_dir.name

        support: list[dict] = []

        for p in sorted(cls_dir.glob("*.png")):
            bbox = _bbox_from_image(p)

            if bbox is not None:
                support.append(
                    {
                        "path": str(p),
                        "bbox": bbox,
                    }
                )

        test_cls = test_dir / cls

        if test_cls.exists():
            for p in sorted(test_cls.glob("*.png")):
                bbox = _bbox_from_image(p)

                if bbox is not None:
                    support.append(
                        {
                            "path": str(p),
                            "bbox": bbox,
                        }
                    )

        if len(support) < N_SUPPORT_MIN:
            continue

        out.append(
            {
                "instance_id": f"hots_{_normalize_name(cls)}",
                "source": "hots",
                "class_name": cls,
                "support_images": support,
                "query_images": [],
            }
        )

    print(f"hots instances: {len(out)}")

    return out


def collect_insdet_instances() -> list[dict]:
    objects_dir = INSDET_DIR / "Objects"

    out: list[dict] = []

    if not objects_dir.exists():
        print(f"[skip] missing {objects_dir}")
        return out

    for obj_dir in sorted(d for d in objects_dir.iterdir() if d.is_dir()):
        images_dir = obj_dir / "images"
        masks_dir = obj_dir / "masks"

        img_files = sorted(images_dir.glob("*.jpg"))

        if len(img_files) < N_SUPPORT_MIN:
            continue

        support: list[dict] = []

        for p in img_files:
            mask = masks_dir / f"{p.stem}.png"

            if mask.exists():
                bbox = _bbox_from_mask(mask)
            else:
                bbox = _bbox_from_image(p)

            if bbox is None:
                continue

            support.append(
                {
                    "path": str(p),
                    "bbox": bbox,
                }
            )

        if len(support) < N_SUPPORT_MIN:
            continue

        out.append(
            {
                "instance_id": f"insdet_{_normalize_name(obj_dir.name)}",
                "source": "insdet",
                "class_name": obj_dir.name,
                "support_images": support,
                "query_images": [],
            }
        )

    print(f"insdet instances: {len(out)}")

    return out


# ---------------------------------------------------------------------------
# Scene queries
# ---------------------------------------------------------------------------


def collect_hots_scene_queries() -> list[dict]:
    annot_dir = HOTS_SCENE_DIR / "ObjectDetection" / "Annotations"
    rgb_dir = HOTS_SCENE_DIR / "RGB"

    out: list[dict] = []

    if not annot_dir.exists():
        return out

    for xml_p in sorted(annot_dir.glob("*.xml")):
        img_p = rgb_dir / f"{xml_p.stem}.png"

        if not img_p.exists():
            continue

        for obj in _parse_voc_xml(xml_p):
            out.append(
                {
                    "instance_id": f"hots_{_normalize_name(obj['name'])}",
                    "path": str(img_p),
                    "bbox": obj["bbox"],
                    "scene_type": "hots_scene",
                }
            )

    return out


def collect_insdet_scene_queries() -> list[dict]:
    scenes_root = INSDET_DIR / "Scenes"

    out: list[dict] = []

    if not scenes_root.exists():
        return out

    for difficulty in ("easy", "hard"):
        diff_dir = scenes_root / difficulty

        if not diff_dir.exists():
            continue

        for scene_dir in sorted(d for d in diff_dir.iterdir() if d.is_dir()):
            for xml_p in sorted(scene_dir.glob("*.xml")):
                img_p = scene_dir / f"{xml_p.stem}.jpg"

                if not img_p.exists():
                    continue

                for obj in _parse_voc_xml(xml_p):
                    out.append(
                        {
                            "instance_id": f"insdet_{_normalize_name(obj['name'])}",
                            "path": str(img_p),
                            "bbox": obj["bbox"],
                            "scene_type": f"insdet_{difficulty}",
                        }
                    )

    return out


# ---------------------------------------------------------------------------
# Query attachment
# ---------------------------------------------------------------------------


def attach_scene_queries(
    instances: list[dict],
    scene_queries: list[dict],
) -> list[dict]:
    by_id: dict[str, list[dict]] = {}

    for sq in scene_queries:
        by_id.setdefault(
            sq["instance_id"],
            [],
        ).append(
            {
                "path": sq["path"],
                "bbox": sq["bbox"],
                "scene_type": sq["scene_type"],
            }
        )

    out: list[dict] = []

    for inst in instances:
        sq = by_id.get(inst["instance_id"], [])

        out.append(
            {
                **inst,
                "query_images": inst["query_images"] + sq,
            }
        )

    return out


def filter_empty_query_instances(
    instances: list[dict],
) -> list[dict]:
    return [i for i in instances if i["query_images"]]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_train_test(
    instances: list[dict],
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)

    by_source: dict[str, list[dict]] = {}

    for inst in instances:
        by_source.setdefault(
            inst["source"],
            [],
        ).append(inst)

    train: list[dict] = []
    test: list[dict] = []

    for source in sorted(by_source):
        group = by_source[source][:]

        rng.shuffle(group)

        n = len(group)

        n_train = int(round(n * TRAIN_RATIO))

        if n > 1:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = n

        train.extend(group[:n_train])
        test.extend(group[n_train:])

        print(
            f"split[{source}] " f"train={n_train} " f"test={n - n_train}",
        )

    return train, test


# ---------------------------------------------------------------------------
# rembg cropper
# ---------------------------------------------------------------------------


class RMBGCropper:
    """Ultra tight rembg foreground cropper."""

    def __init__(
        self,
        model_name: str = RMBG_MODEL,
        mask_threshold: float = RMBG_MASK_THRESHOLD,
    ) -> None:
        from rembg import new_session

        self.mask_threshold = float(mask_threshold)

        print(
            f"loading rembg model: {model_name}",
            flush=True,
        )

        self.session = new_session(model_name)

    def _predict_mask(
        self,
        img: Image.Image,
    ) -> np.ndarray:
        from rembg import remove

        rgba = remove(
            img,
            session=self.session,
        )

        rgba = np.asarray(rgba)

        alpha = rgba[:, :, 3].astype(np.float32) / 255.0

        return alpha > self.mask_threshold

    def tight_bbox(
        self,
        img: Image.Image,
    ) -> list[int] | None:
        mask = self._predict_mask(img)

        if not mask.any():
            return None

        return _largest_component_bbox(mask)


def _get_rmbg() -> "RMBGCropper | None":
    global _RMBG_SINGLETON

    if _RMBG_SINGLETON is not None:
        return _RMBG_SINGLETON

    try:
        _RMBG_SINGLETON = RMBGCropper()

    except Exception as e:
        print(
            f"WARNING: rembg init failed " f"({type(e).__name__}: {e})",
            flush=True,
        )

        _RMBG_SINGLETON = None

    return _RMBG_SINGLETON


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def _crop_support_to_disk(
    src_path: Path,
    dst_path: Path,
    bbox: list[int],
    pad_frac: float = SUPPORT_CROP_PAD_FRAC,
    rmbg: "RMBGCropper | None" = None,
) -> tuple[int, int]:
    img = Image.open(src_path).convert("RGB")

    w, h = img.size

    if rmbg is not None:
        try:
            tight = rmbg.tight_bbox(img)
        except Exception as e:
            print(
                f"[rmbg-error] {src_path}: " f"{type(e).__name__}: {e}",
            )
            tight = None

        if tight is not None:
            x1, y1, x2, y2 = tight
        else:
            x1, y1, x2, y2 = bbox

    else:
        x1, y1, x2, y2 = bbox

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Tiny edge padding
    px = max(1, int(round(bw * pad_frac)))
    py = max(1, int(round(bh * pad_frac)))

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)

    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)

    cropped = img.crop((cx1, cy1, cx2, cy2))

    cropped.save(
        dst_path,
        quality=100,
    )

    return cropped.size


# ---------------------------------------------------------------------------
# Stage images
# ---------------------------------------------------------------------------


def stage_images(
    splits: dict[str, list[dict]],
) -> None:
    rmbg = _get_rmbg()

    n_copied = 0
    n_cropped = 0

    for split_name, insts in splits.items():
        for inst in insts:
            inst_id = inst["instance_id"]
            source = inst["source"]

            for role, img_list in (
                ("support", inst["support_images"]),
                ("query", inst["query_images"]),
            ):
                role_dir = OUT_DIR / split_name / role / inst_id

                for idx, img in enumerate(img_list, start=1):
                    src = Path(img["path"])

                    dst = role_dir / f"{idx:03d}{src.suffix}"

                    dst.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    if role == "support" and source in SOURCES_CROP_SUPPORTS_ON_DISK:
                        out_w, out_h = _crop_support_to_disk(
                            src,
                            dst,
                            img["bbox"],
                            rmbg=rmbg,
                        )

                        img["bbox"] = [
                            0,
                            0,
                            int(out_w),
                            int(out_h),
                        ]

                        n_cropped += 1

                    else:
                        shutil.copy2(src, dst)

                    img["path"] = (
                        Path(split_name) / role / inst_id / dst.name
                    ).as_posix()

                    n_copied += 1

    print(
        f"staged {n_copied} images " f"({n_cropped} cropped)",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    splits: dict[str, list[dict]],
) -> None:
    instances_with_split: list[dict] = []

    split_index: dict[str, list[str]] = {k: [] for k in splits}

    for split_name, insts in splits.items():
        for inst in insts:
            instances_with_split.append(
                {
                    **inst,
                    "split": split_name,
                }
            )

            split_index[split_name].append(
                inst["instance_id"],
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "num_instances": len(instances_with_split),
        "splits": split_index,
        "instances": instances_with_split,
    }

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("manifest written")


def write_stats(
    splits: dict[str, list[dict]],
) -> None:
    stats: dict = {
        "schema_version": SCHEMA_VERSION,
        "total_instances": sum(len(v) for v in splits.values()),
    }

    for name, insts in splits.items():
        stats[name] = {
            "instances": len(insts),
            "hots": sum(1 for i in insts if i["source"] == "hots"),
            "insdet": sum(1 for i in insts if i["source"] == "insdet"),
            "support_images": sum(len(i["support_images"]) for i in insts),
            "query_images": sum(len(i["query_images"]) for i in insts),
        }

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate() -> bool:
    manifest_path = OUT_DIR / "manifest.json"

    if not manifest_path.exists():
        print("manifest missing")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_bad = 0

    for inst in manifest["instances"]:
        for role in ("support_images", "query_images"):
            for img in inst[role]:
                p = OUT_DIR / img["path"]

                if not p.exists():
                    print(f"missing {p}")
                    n_bad += 1

    print(f"validation complete: {n_bad} bad")

    return n_bad == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(force: bool = False) -> None:
    if OUT_DIR.exists() and not force:
        if validate():
            print("dataset already exists and validates")
            return

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("collecting instances")

    instances = collect_hots_instances() + collect_insdet_instances()

    print("collecting scene queries")

    scene_queries = collect_hots_scene_queries() + collect_insdet_scene_queries()

    instances = attach_scene_queries(
        instances,
        scene_queries,
    )

    instances = filter_empty_query_instances(
        instances,
    )

    train, test = split_train_test(instances)

    splits = {
        "train": train,
        "test": test,
    }

    print("staging images")

    stage_images(splits)

    write_manifest(splits)

    write_stats(splits)

    print("validating")

    if not validate():
        print("validation failed")
        sys.exit(1)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="force rebuild",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="only validate existing dataset",
    )

    args = parser.parse_args()

    if args.validate:
        ok = validate()
        sys.exit(0 if ok else 1)

    main(force=args.force)
