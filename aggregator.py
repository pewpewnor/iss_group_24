"""Aggregate raw datasets into a clean staging directory + manifest.

Two phases of training data:
  Phase 1 (VizWiz base): 100 categories, 4229 images, episodic crop+rotation synthesis.
      source="vizwiz_base"; split 75/10/15 so Phase 1 can be validated on its own domain.

  Phase 2 (mixed):
      VizWiz novel: 16 categories, 1 support image per category (rotation synthesis).
          source="vizwiz_novel"; all in train (too few to split).
      HOTS: instance-level multi-shot supports + scene queries.
          source="hots"; split 75/10/15.
      InsDet: instance-level multi-shot supports + scene queries.
          source="insdet"; split 75/10/15.

Negatives:
  vizwiz_base pool: ~500 VizWiz base images not used as positives.
      Used by Phase 1 (same domain, prevents style-gap trivial negatives).
  hope pool: HOPE scene RGBs + InsDet Background + VizWiz query images.
      Used by Phase 2 (mixed product / real-world scenes).

Run:
    uv run python -m aggregator
"""

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
N_SUPPORT = 4
BBOX_PAD_FRAC = 0.05
BBOX_MIN_SIDE = 20
BBOX_MIN_AREA_FRAC = 0.005

BASE_DIR = Path("dataset/original")
OUT_DIR = Path("dataset/cleaned")

HOTS_OBJECT_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "object"
HOTS_SCENE_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "scene"
INSDET_DIR = BASE_DIR / "InsDet"
HOPE_DIR = BASE_DIR / "HOPE"
VIZWIZ_DIR = BASE_DIR / "VizWiz"

# VizWiz base scope controls. Each (scene, bbox) entry = one complete episode source;
# the rotation trick generates 4 support views + the original scene as query at runtime.
# Capping per-category limits staging time and disk while keeping diverse pretraining.
VIZWIZ_BASE_MAX_IMAGES_PER_CAT = 0  # 0 = no cap, use all available images per category
VIZWIZ_NEG_SAMPLES = 500  # Phase 1 negative backgrounds from VizWiz base unused images


# ---------------------------------------------------------------------------
# Bbox utilities (shared across all collectors)
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


def bbox_from_mask(mask_path: Path) -> list[int] | None:
    mask_arr = np.array(Image.open(mask_path).convert("L")) > 127
    bbox = _largest_component_bbox(mask_arr)
    if bbox is None:
        return None
    h, w = mask_arr.shape
    bbox = _pad_bbox(bbox, (w, h))
    return bbox if _bbox_valid(bbox, (w, h)) else None


def bbox_from_image(img_path: Path, bg_thresh: int = 240) -> list[int] | None:
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


def parse_voc_xml(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    out = []
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
# VizWiz base collector (Phase 1 pretraining)
# ---------------------------------------------------------------------------


def collect_vizwiz_base_instances() -> list[dict]:
    """One instance per VizWiz base category holding (scene, bbox) entries.

    Phase 1 episodes are synthesised at sample time: pick one entry, crop
    the bbox + rotate four ways → 4 supports; the same scene image becomes
    the query (with the same bbox as ground truth). This turns a category-
    level annotation into instance-level supervision — the model is asked to
    find the specific object whose own crops it just saw.

    Dedup: one entry per (image, category) pair, keeping the largest bbox if
    a scene contains multiple instances of the same category.
    """
    annot_path = VIZWIZ_DIR / "base_annotations.json"
    if not annot_path.exists():
        print(f"  [skip] vizwiz_base: {annot_path} not found")
        return []

    with open(annot_path) as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_meta = {im["id"]: im for im in coco["images"]}

    by_cat: dict[int, dict[int, dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("ignore") or ann.get("iscrowd"):
            continue
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        x, y, w_b, h_b = ann["bbox"]
        if w_b <= 0 or h_b <= 0:
            continue
        area = w_b * h_b
        cat = by_cat.setdefault(cat_id, {})
        prev = cat.get(img_id)
        if prev is None or area > prev["area"]:
            cat[img_id] = {"area": area, "x": x, "y": y, "w": w_b, "h": h_b}

    rng = random.Random(SEED)
    instances: list[dict] = []
    skipped = 0
    for cat_id in sorted(by_cat):
        per_image = by_cat[cat_id]
        entries: list[dict] = []
        for img_id, ann in per_image.items():
            meta = image_meta.get(img_id)
            if meta is None:
                continue
            img_path = VIZWIZ_DIR / "images" / meta["file_name"]
            if not img_path.exists():
                continue
            iw, ih = int(meta["width"]), int(meta["height"])
            x1 = int(ann["x"])
            y1 = int(ann["y"])
            x2 = int(ann["x"] + ann["w"])
            y2 = int(ann["y"] + ann["h"])
            bbox = _pad_bbox([x1, y1, x2, y2], (iw, ih))
            if not _bbox_valid(bbox, (iw, ih)):
                continue
            entries.append({"path": str(img_path), "bbox": bbox})

        # Minimum entries to provide episodic variety within the category.
        if len(entries) < 5:
            skipped += 1
            continue

        rng.shuffle(entries)
        if (
            VIZWIZ_BASE_MAX_IMAGES_PER_CAT > 0
            and len(entries) > VIZWIZ_BASE_MAX_IMAGES_PER_CAT
        ):
            entries = entries[:VIZWIZ_BASE_MAX_IMAGES_PER_CAT]

        name = cat_name[cat_id]
        # Deep copy entries for support/query so stage_images path mutations
        # on one list don't corrupt the other.
        instances.append(
            {
                "instance_id": f"vizwiz_base_{_normalize_name(name)}",
                "source": "vizwiz_base",
                "class_name": name,
                "support_images": [
                    {"path": e["path"], "bbox": e["bbox"]} for e in entries
                ],
                "query_images": [
                    {"path": e["path"], "bbox": e["bbox"]} for e in entries
                ],
            }
        )

    print(f"vizwiz_base: {len(instances)} categories ({skipped} skipped, < 5 entries)")
    return instances


# ---------------------------------------------------------------------------
# VizWiz novel collector (Phase 2, additional 16 instances)
# ---------------------------------------------------------------------------


def collect_vizwiz_novel_instances() -> list[dict]:
    """One instance per VizWiz novel category. 16 categories, 1 image each.

    The same rotation trick (crop + 4-way rotate) is used at sample time to
    generate 4 support views from the single labeled image. All 16 go to
    train since the set is too small to meaningfully split.
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
        entry = {"path": str(img_path), "bbox": bbox}
        instances.append(
            {
                "instance_id": f"vizwiz_novel_{_normalize_name(name)}",
                "source": "vizwiz_novel",
                "class_name": name,
                # Single entry; rotation synthesis generates 4 supports at runtime.
                "support_images": [{"path": entry["path"], "bbox": entry["bbox"]}],
                "query_images": [{"path": entry["path"], "bbox": entry["bbox"]}],
            }
        )

    print(f"vizwiz_novel: {len(instances)} instances ({skipped} skipped)")
    return instances


# ---------------------------------------------------------------------------
# HOTS + InsDet collectors (unchanged from cleaner.py)
# ---------------------------------------------------------------------------


def collect_hots_instances() -> list[dict]:
    train_dir = HOTS_OBJECT_DIR / "train"
    test_dir = HOTS_OBJECT_DIR / "test"
    instances = []
    for cls_dir in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        cls = cls_dir.name
        support = []
        for p in sorted(cls_dir.glob("*.png")):
            bbox = bbox_from_image(p)
            if bbox is not None:
                support.append({"path": str(p), "bbox": bbox})
        test_cls = test_dir / cls
        if test_cls.exists():
            for p in sorted(test_cls.glob("*.png")):
                bbox = bbox_from_image(p)
                if bbox is not None:
                    support.append({"path": str(p), "bbox": bbox})
        if len(support) < N_SUPPORT:
            print(f"  [skip] hots/{cls}: {len(support)} valid support images")
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


def collect_insdet_instances() -> list[dict]:
    objects_dir = INSDET_DIR / "Objects"
    instances = []
    for obj_dir in sorted(d for d in objects_dir.iterdir() if d.is_dir()):
        images_dir = obj_dir / "images"
        masks_dir = obj_dir / "masks"
        img_files = sorted(images_dir.glob("*.jpg"))
        if len(img_files) < N_SUPPORT + 1:
            print(f"  [skip] insdet/{obj_dir.name}: {len(img_files)} images")
            continue
        support = []
        for p in img_files:
            mask = masks_dir / f"{p.stem}.png"
            bbox = bbox_from_mask(mask) if mask.exists() else bbox_from_image(p)
            if bbox is None:
                continue
            support.append({"path": str(p), "bbox": bbox})
        if len(support) < N_SUPPORT:
            print(
                f"  [skip] insdet/{obj_dir.name}: post-bbox {len(support)} support images"
            )
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


def collect_hots_scene_queries() -> list[dict]:
    annot_dir = HOTS_SCENE_DIR / "ObjectDetection" / "Annotations"
    rgb_dir = HOTS_SCENE_DIR / "RGB"
    if not annot_dir.exists():
        return []
    out = []
    for xml_p in sorted(annot_dir.glob("*.xml")):
        img_p = rgb_dir / f"{xml_p.stem}.png"
        if not img_p.exists():
            continue
        for o in parse_voc_xml(xml_p):
            out.append(
                {
                    "instance_id": f"hots_{_normalize_name(o['name'])}",
                    "path": str(img_p),
                    "bbox": o["bbox"],
                    "scene_type": "hots_scene",
                }
            )
    print(f"hots scene: {len(out)} annotations")
    return out


def collect_insdet_scene_queries() -> list[dict]:
    scenes_root = INSDET_DIR / "Scenes"
    if not scenes_root.exists():
        return []
    out = []
    for difficulty in ("easy", "hard"):
        diff_dir = scenes_root / difficulty
        if not diff_dir.exists():
            continue
        for scene_dir in sorted(d for d in diff_dir.iterdir() if d.is_dir()):
            for xml_p in sorted(scene_dir.glob("*.xml")):
                img_p = scene_dir / f"{xml_p.stem}.jpg"
                if not img_p.exists():
                    continue
                for o in parse_voc_xml(xml_p):
                    out.append(
                        {
                            "instance_id": f"insdet_{_normalize_name(o['name'])}",
                            "path": str(img_p),
                            "bbox": o["bbox"],
                            "scene_type": f"insdet_{difficulty}",
                        }
                    )
    print(f"insdet scene: {len(out)} annotations")
    return out


def attach_scene_queries(
    instances: list[dict], scene_queries: list[dict]
) -> list[dict]:
    by_id: dict[str, list[dict]] = {}
    for sq in scene_queries:
        by_id.setdefault(sq["instance_id"], []).append(
            {"path": sq["path"], "bbox": sq["bbox"], "scene_type": sq["scene_type"]}
        )
    matched_total = 0
    matched_instances = 0
    out = []
    for inst in instances:
        sq = by_id.get(inst["instance_id"], [])
        if sq:
            matched_instances += 1
            matched_total += len(sq)
        out.append({**inst, "query_images": inst["query_images"] + sq})
    unmatched_keys = set(by_id) - {i["instance_id"] for i in instances}
    print(
        f"scene attached: {matched_total} annotations across "
        f"{matched_instances}/{len(instances)} instances "
        f"({len(unmatched_keys)} XML names matched no instance)"
    )
    return out


# ---------------------------------------------------------------------------
# Negative background collector
# ---------------------------------------------------------------------------


def collect_negative_backgrounds(
    vizwiz_base_used_paths: set[str],
) -> list[dict]:
    """Negative (no-target) backgrounds, tagged by pool.

    vizwiz_base pool — used by Phase 1 (VizWiz base episodes). Drawing from
    the same visual domain (VizWiz phone shots) prevents the model from
    rejecting negatives by global image style rather than object content.

    hope pool — used by Phase 2 (HOTS + InsDet + VizWiz novel episodes).
    Includes HOPE scene RGBs, InsDet Background images, and VizWiz query
    images (unannotated). Mixed-domain is fine for Phase 2: the model is
    already past the style-discrimination failure mode by this point.
    """
    out: list[dict] = []

    # VizWiz base negatives (Phase 1): random images not used as positives.
    n_vizwiz_base = 0
    if VIZWIZ_DIR.exists():
        rng = random.Random(SEED + 1)
        images_dir = VIZWIZ_DIR / "images"
        candidates = list(images_dir.glob("*.jpg"))
        rng.shuffle(candidates)
        for p in candidates:
            if str(p) in vizwiz_base_used_paths:
                continue
            out.append({"path": str(p), "source": "vizwiz_base"})
            n_vizwiz_base += 1
            if n_vizwiz_base >= VIZWIZ_NEG_SAMPLES:
                break

    # HOPE scenes (Phase 2).
    n_hope = 0
    if HOPE_DIR.exists():
        for p in sorted(HOPE_DIR.rglob("*_rgb.jpg")):
            out.append({"path": str(p), "source": "hope"})
            n_hope += 1

    # InsDet Background images (Phase 2).
    n_insdet_bg = 0
    bg_dir = INSDET_DIR / "Background"
    if bg_dir.exists():
        for p in sorted(bg_dir.glob("*.jpg")):
            out.append({"path": str(p), "source": "hope"})
            n_insdet_bg += 1

    # VizWiz query images (Phase 2, unannotated).
    n_vizwiz_q = 0
    query_dir = VIZWIZ_DIR / "query_images"
    if query_dir.exists():
        for p in sorted(query_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg"):
                out.append({"path": str(p), "source": "hope"})
                n_vizwiz_q += 1

    print(
        f"  negatives: vizwiz_base={n_vizwiz_base}  "
        f"hope(hope_scenes={n_hope} insdet_bg={n_insdet_bg} vizwiz_query={n_vizwiz_q})"
    )
    return out


# ---------------------------------------------------------------------------
# Split, stage, manifest
# ---------------------------------------------------------------------------


def filter_empty_instances(instances: list[dict]) -> list[dict]:
    out = []
    for inst in instances:
        if len(inst["query_images"]) == 0:
            print(f"  [skip] {inst['instance_id']}: no query images after attach")
            continue
        out.append(inst)
    return out


def split_instances(instances: list[dict]) -> tuple[list, list, list]:
    """Split instances by source:

    vizwiz_base: 75/10/15 — needs a val split for Phase 1 validation.
    vizwiz_novel: 100% train (only 16 instances).
    hots, insdet: 75/10/15 — val/test are the deployment domain.
    """
    rng = random.Random(SEED)
    by_source: dict[str, list[dict]] = {}
    for inst in instances:
        by_source.setdefault(inst["source"], []).append(inst)

    train, val, test = [], [], []
    for source in sorted(by_source):
        group = by_source[source][:]
        rng.shuffle(group)
        n = len(group)
        if source == "vizwiz_novel":
            train.extend(group)
            print(
                f"  split[vizwiz_novel]: train={n} val=0 test=0  (train-only, only {n} instances)"
            )
            continue
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
        print(
            f"  split[{source}]: train={n_train} val={n_val} test={n - n_train - n_val}"
        )
    return train, val, test


def stage_images(splits: dict[str, list[dict]], negatives: list[dict]) -> list[dict]:
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

    new_negatives: list[dict] = []
    for neg in negatives:
        src = Path(neg["path"])
        source = neg["source"]
        sub_dir = OUT_DIR / "negatives" / source
        sub_dir.mkdir(parents=True, exist_ok=True)
        dst = sub_dir / src.name
        if dst.exists():
            import hashlib

            h = hashlib.md5(str(src).encode()).hexdigest()[:8]
            dst = sub_dir / f"{src.stem}_{h}{src.suffix}"
        shutil.copy2(src, dst)
        n_copied += 1
        new_negatives.append(
            {
                "path": (Path("negatives") / source / dst.name).as_posix(),
                "source": source,
            }
        )

    print(f"staged images: {n_copied} files copied")
    return new_negatives


def write_manifest(splits: dict[str, list[dict]], negatives: list[dict]) -> None:
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
                "negative_backgrounds": negatives,
            },
            f,
            indent=2,
        )
    counts = ", ".join(f"{k}={len(v)}" for k, v in split_index.items())
    print(f"  manifest -> {p}: {len(instances_with_split)} instances ({counts})")


def write_stats(splits: dict[str, list[dict]]) -> None:
    stats: dict = {"total_instances": sum(len(v) for v in splits.values())}
    for name, insts in splits.items():
        stats[name] = {
            "instances": len(insts),
            "vizwiz_base": sum(1 for i in insts if i["source"] == "vizwiz_base"),
            "vizwiz_novel": sum(1 for i in insts if i["source"] == "vizwiz_novel"),
            "hots": sum(1 for i in insts if i["source"] == "hots"),
            "insdet": sum(1 for i in insts if i["source"] == "insdet"),
            "support_images": sum(len(i["support_images"]) for i in insts),
            "query_images": sum(len(i["query_images"]) for i in insts),
        }
    p = OUT_DIR / "stats.json"
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        print(f"cleared {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)

    print("collecting VizWiz base instances (Phase 1 pretraining pool)")
    vizwiz_base_instances = collect_vizwiz_base_instances()

    print("collecting VizWiz novel instances (Phase 2 additional training)")
    vizwiz_novel_instances = collect_vizwiz_novel_instances()

    print("collecting HOTS + InsDet instances (Phase 2 target domain)")
    target_instances = collect_hots_instances() + collect_insdet_instances()

    print("collecting scene queries for HOTS + InsDet")
    scene = collect_hots_scene_queries() + collect_insdet_scene_queries()
    target_instances = attach_scene_queries(target_instances, scene)
    target_instances = filter_empty_instances(target_instances)

    instances = vizwiz_base_instances + vizwiz_novel_instances + target_instances

    # Track VizWiz base images used as positives so the negative pool is disjoint.
    vizwiz_base_used: set[str] = set()
    for inst in vizwiz_base_instances:
        for img in inst["support_images"] + inst["query_images"]:
            vizwiz_base_used.add(img["path"])

    negatives = collect_negative_backgrounds(vizwiz_base_used_paths=vizwiz_base_used)
    print(f"negatives: {len(negatives)} background images")

    train, val, test = split_instances(instances)
    splits = {"train": train, "val": val, "test": test}

    print("staging images into train / val / test directories")
    negatives = stage_images(splits, negatives)

    write_manifest(splits, negatives)
    write_stats(splits)


if __name__ == "__main__":
    main()
