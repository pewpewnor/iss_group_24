"""Aggregate raw datasets into a clean staging directory + manifest.

Output layout:
    dataset/aggregated/
    ├── manifest.json
    ├── stats.json
    ├── train/{support,query}/<inst_id>/<NNN>.{png,jpg}
    ├── test/{support,query}/<inst_id>/<NNN>.{png,jpg}
    └── phase0/{support,query}/<inst_id>/<NNN>.{png,jpg}

Sources:
    HOTS          → train + test (80/20 instance split)
    InsDet        → train + test (80/20 instance split)
    vizwiz_novel  → phase0 only (16 instances, untouched)

VizWiz base / FSOD / HOPE are NOT used. Negative episodes during training
are constructed from same-source other-instance queries.

Run:
    uv run python -m aggregator
"""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SEED = 42
TRAIN_RATIO = 0.80
N_SUPPORT = 4
BBOX_PAD_FRAC = 0.05
BBOX_MIN_SIDE = 20
BBOX_MIN_AREA_FRAC = 0.005

BASE_DIR = Path("dataset/original")
OUT_DIR = Path("dataset/aggregated")

HOTS_OBJECT_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "object"
HOTS_SCENE_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "scene"
INSDET_DIR = BASE_DIR / "InsDet"
VIZWIZ_DIR = BASE_DIR / "VizWiz"


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------


def _largest_component_bbox(mask: np.ndarray) -> list[int] | None:
    if not mask.any():
        return None
    labels, n = ndimage.label(mask)  # type: ignore[misc]
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    rows = np.any(labels == largest, axis=1)
    cols = np.any(labels == largest, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def _pad_bbox(
    bbox: list[int], img_size: tuple[int, int], frac: float = BBOX_PAD_FRAC
) -> list[int]:
    w, h = img_size
    x1, y1, x2, y2 = bbox
    px = int((x2 - x1) * frac)
    py = int((y2 - y1) * frac)
    return [max(0, x1 - px), max(0, y1 - py), min(w - 1, x2 + px), min(h - 1, y2 + py)]


def _bbox_valid(bbox: list[int], img_size: tuple[int, int]) -> bool:
    w, h = img_size
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < BBOX_MIN_SIDE or bh < BBOX_MIN_SIDE:
        return False
    return (bw * bh) / (w * h) >= BBOX_MIN_AREA_FRAC


def _bbox_from_mask(mask_path: Path) -> list[int] | None:
    mask_arr = np.array(Image.open(mask_path).convert("L")) > 127
    bbox = _largest_component_bbox(mask_arr)
    if bbox is None:
        return None
    h, w = mask_arr.shape
    bbox = _pad_bbox(bbox, (w, h))
    return bbox if _bbox_valid(bbox, (w, h)) else None


def _bbox_from_image(img_path: Path, bg_thresh: int = 240) -> list[int] | None:
    img = np.array(Image.open(img_path).convert("RGB"))
    fg = ~np.all(img > bg_thresh, axis=2)
    bbox = _largest_component_bbox(fg)
    if bbox is None:
        return None
    h, w = fg.shape
    bbox = _pad_bbox(bbox, (w, h))
    return bbox if _bbox_valid(bbox, (w, h)) else None


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
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({"name": name_el.text.strip(), "bbox": [x1, y1, x2, y2]})
    return out


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# HOTS collectors
# ---------------------------------------------------------------------------


def collect_hots_instances() -> list[dict]:
    """Pool object/{train,test}/<cls>/*.png as supports per class.

    The HOTS object split is photographer-round, not generalisation, so
    we pool both rounds for each class.
    """
    train_dir = HOTS_OBJECT_DIR / "train"
    test_dir = HOTS_OBJECT_DIR / "test"
    instances: list[dict] = []
    if not train_dir.exists():
        print(f"  [skip] hots: {train_dir} missing")
        return instances
    for cls_dir in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        cls = cls_dir.name
        support: list[dict] = []
        for p in sorted(cls_dir.glob("*.png")):
            bbox = _bbox_from_image(p)
            if bbox is not None:
                support.append({"path": str(p), "bbox": bbox})
        test_cls = test_dir / cls
        if test_cls.exists():
            for p in sorted(test_cls.glob("*.png")):
                bbox = _bbox_from_image(p)
                if bbox is not None:
                    support.append({"path": str(p), "bbox": bbox})
        if len(support) < N_SUPPORT:
            print(f"  [skip] hots/{cls}: only {len(support)} valid supports")
            continue
        instances.append(
            {
                "instance_id": f"hots_{_normalize_name(cls)}",
                "source": "hots",
                "class_name": cls,
                "support_images": support,
                "query_images": [],
            }
        )
    print(f"hots: {len(instances)} instances")
    return instances


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
        for o in _parse_voc_xml(xml_p):
            out.append(
                {
                    "instance_id": f"hots_{_normalize_name(o['name'])}",
                    "path": str(img_p),
                    "bbox": o["bbox"],
                    "scene_type": "hots_scene",
                }
            )
    print(f"hots scene annotations: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# InsDet collectors
# ---------------------------------------------------------------------------


def collect_insdet_instances() -> list[dict]:
    """Pool Objects/<cls>/images/*.jpg as supports per class.

    Bbox is computed from the matching mask under masks/<stem>.png.
    """
    objects_dir = INSDET_DIR / "Objects"
    instances: list[dict] = []
    if not objects_dir.exists():
        print(f"  [skip] insdet: {objects_dir} missing")
        return instances
    for obj_dir in sorted(d for d in objects_dir.iterdir() if d.is_dir()):
        images_dir = obj_dir / "images"
        masks_dir = obj_dir / "masks"
        img_files = sorted(images_dir.glob("*.jpg"))
        if len(img_files) < N_SUPPORT:
            print(f"  [skip] insdet/{obj_dir.name}: {len(img_files)} images")
            continue
        support: list[dict] = []
        for p in img_files:
            mask = masks_dir / f"{p.stem}.png"
            bbox = _bbox_from_mask(mask) if mask.exists() else _bbox_from_image(p)
            if bbox is None:
                continue
            support.append({"path": str(p), "bbox": bbox})
        if len(support) < N_SUPPORT:
            print(f"  [skip] insdet/{obj_dir.name}: post-bbox {len(support)} supports")
            continue
        instances.append(
            {
                "instance_id": f"insdet_{_normalize_name(obj_dir.name)}",
                "source": "insdet",
                "class_name": obj_dir.name,
                "support_images": support,
                "query_images": [],
            }
        )
    print(f"insdet: {len(instances)} instances")
    return instances


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
                for o in _parse_voc_xml(xml_p):
                    out.append(
                        {
                            "instance_id": f"insdet_{_normalize_name(o['name'])}",
                            "path": str(img_p),
                            "bbox": o["bbox"],
                            "scene_type": f"insdet_{difficulty}",
                        }
                    )
    print(f"insdet scene annotations: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# VizWiz novel collector (phase0 only)
# ---------------------------------------------------------------------------


def collect_vizwiz_novel_instances() -> list[dict]:
    """16 categories, 1 image each. Used for Phase 0 zero-shot baseline.

    The single image acts as both support source and query. The 4 supports
    are produced by rotation synthesis (0/90/180/270) at episode time —
    the manifest just stores the single (image, bbox) entry.
    """
    annot_path = VIZWIZ_DIR / "support_set.json"
    if not annot_path.exists():
        print(f"  [skip] vizwiz_novel: {annot_path} not found")
        return []
    with open(annot_path) as f:
        coco = json.load(f)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_meta = {im["id"]: im for im in coco["images"]}
    instances: list[dict] = []
    skipped = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        meta = image_meta.get(img_id)
        if meta is None:
            skipped += 1
            continue
        img_path = VIZWIZ_DIR / "support_images" / meta["file_name"]
        if not img_path.exists():
            skipped += 1
            continue
        iw, ih = int(meta["width"]), int(meta["height"])
        x, y, w_b, h_b = ann["bbox"]
        if w_b <= 0 or h_b <= 0:
            skipped += 1
            continue
        bbox = _pad_bbox([int(x), int(y), int(x + w_b), int(y + h_b)], (iw, ih))
        if not _bbox_valid(bbox, (iw, ih)):
            skipped += 1
            continue
        name = cat_name[cat_id]
        instances.append(
            {
                "instance_id": f"vizwiz_novel_{_normalize_name(name)}",
                "source": "vizwiz_novel",
                "class_name": name,
                "support_images": [{"path": str(img_path), "bbox": bbox}],
                "query_images": [{"path": str(img_path), "bbox": bbox,
                                  "scene_type": "vizwiz_novel"}],
            }
        )
    print(f"vizwiz_novel: {len(instances)} instances ({skipped} skipped)")
    return instances


# ---------------------------------------------------------------------------
# Attach scene queries
# ---------------------------------------------------------------------------


def attach_scene_queries(
    instances: list[dict], scene_queries: list[dict]
) -> list[dict]:
    by_id: dict[str, list[dict]] = {}
    for sq in scene_queries:
        by_id.setdefault(sq["instance_id"], []).append(
            {"path": sq["path"], "bbox": sq["bbox"], "scene_type": sq["scene_type"]}
        )
    out: list[dict] = []
    matched_total = 0
    matched_instances = 0
    for inst in instances:
        sq = by_id.get(inst["instance_id"], [])
        if sq:
            matched_instances += 1
            matched_total += len(sq)
        out.append({**inst, "query_images": inst["query_images"] + sq})
    unmatched = set(by_id) - {i["instance_id"] for i in instances}
    print(
        f"scene attached: {matched_total} annotations across "
        f"{matched_instances}/{len(instances)} instances "
        f"({len(unmatched)} XML names matched no instance)"
    )
    return out


def filter_empty_query_instances(instances: list[dict]) -> list[dict]:
    out = []
    for inst in instances:
        if not inst["query_images"]:
            print(f"  [skip] {inst['instance_id']}: no queries after attach")
            continue
        out.append(inst)
    return out


# ---------------------------------------------------------------------------
# Split + stage + manifest
# ---------------------------------------------------------------------------


def split_train_test(instances: list[dict]) -> tuple[list[dict], list[dict]]:
    """80/20 instance-level split, stratified by source.

    At least 1 train and 1 test per source when possible.
    """
    rng = random.Random(SEED)
    by_source: dict[str, list[dict]] = {}
    for inst in instances:
        by_source.setdefault(inst["source"], []).append(inst)
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
        print(f"  split[{source}]: train={n_train} test={n - n_train}")
    return train, test


def stage_images(splits: dict[str, list[dict]]) -> None:
    n_copied = 0

    def copy(src: Path, dst: Path) -> None:
        nonlocal n_copied
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    for split_name, insts in splits.items():
        for inst in insts:
            inst_id = inst["instance_id"]
            for role, img_list in (
                ("support", inst["support_images"]),
                ("query", inst["query_images"]),
            ):
                role_dir = OUT_DIR / split_name / role / inst_id
                for idx, img in enumerate(img_list, start=1):
                    src = Path(img["path"])
                    dst = role_dir / f"{idx:03d}{src.suffix}"
                    copy(src, dst)
                    img["path"] = (
                        Path(split_name) / role / inst_id / dst.name
                    ).as_posix()
    print(f"staged images: {n_copied} files copied")


def write_manifest(splits: dict[str, list[dict]]) -> None:
    instances_with_split: list[dict] = []
    split_index: dict[str, list[str]] = {k: [] for k in splits}
    for split_name, insts in splits.items():
        for inst in insts:
            instances_with_split.append({**inst, "split": split_name})
            split_index[split_name].append(inst["instance_id"])
    p = OUT_DIR / "manifest.json"
    with open(p, "w") as f:
        json.dump(
            {
                "num_instances": len(instances_with_split),
                "splits": split_index,
                "instances": instances_with_split,
            },
            f,
            indent=2,
        )
    counts = ", ".join(f"{k}={len(v)}" for k, v in split_index.items())
    print(f"manifest -> {p}: {len(instances_with_split)} instances ({counts})")


def write_stats(splits: dict[str, list[dict]]) -> None:
    stats: dict = {"total_instances": sum(len(v) for v in splits.values())}
    for name, insts in splits.items():
        stats[name] = {
            "instances": len(insts),
            "hots": sum(1 for i in insts if i["source"] == "hots"),
            "insdet": sum(1 for i in insts if i["source"] == "insdet"),
            "vizwiz_novel": sum(1 for i in insts if i["source"] == "vizwiz_novel"),
            "support_images": sum(len(i["support_images"]) for i in insts),
            "query_images": sum(len(i["query_images"]) for i in insts),
        }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        print(f"cleared {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)

    print("collecting HOTS + InsDet instances")
    main_instances = collect_hots_instances() + collect_insdet_instances()

    print("collecting scene queries")
    scene_queries = collect_hots_scene_queries() + collect_insdet_scene_queries()
    main_instances = attach_scene_queries(main_instances, scene_queries)
    main_instances = filter_empty_query_instances(main_instances)

    print("collecting vizwiz_novel instances (phase0)")
    phase0_instances = collect_vizwiz_novel_instances()

    train, test = split_train_test(main_instances)
    splits = {"train": train, "test": test, "phase0": phase0_instances}

    print("staging images")
    stage_images(splits)
    write_manifest(splits)
    write_stats(splits)


if __name__ == "__main__":
    main()
