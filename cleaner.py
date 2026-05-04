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
N_SUPPORT = 5
BBOX_PAD_FRAC = 0.05      # pad each bbox by 5% on each side to avoid tight crops
BBOX_MIN_SIDE = 20         # drop bboxes whose smaller side is below this many px
BBOX_MIN_AREA_FRAC = 0.005  # drop bboxes covering < 0.5% of image area

BASE_DIR = Path("dataset/original")
OUT_DIR = Path("dataset/cleaned")

HOTS_OBJECT_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "object"
HOTS_SCENE_DIR = BASE_DIR / "HOTS" / "HOTS_v1" / "scene"
INSDET_DIR = BASE_DIR / "InsDet"
HOPE_DIR = BASE_DIR / "HOPE"
FSOD_DIR = BASE_DIR / "FSOD"

# FSOD scope controls. The dataset is huge (50k+ images, 800 categories).
# FSOD episodes are synthesised at sample time: one (scene, bbox) entry feeds
# both the supports (4 rotations of the cropped bbox region) and the query
# (the full scene), so each entry counts as one whole episode's worth of
# data. Tuning these trades stage-0 diversity vs. clean step duration / disk.
FSOD_MAX_CATEGORIES = 600
FSOD_MAX_IMAGES_PER_CAT = 50
FSOD_NEG_SAMPLES = 500


def _largest_component_bbox(mask: np.ndarray) -> list[int] | None:
    """Bbox of the largest connected component of a binary mask.

    Plain np.any-based bbox is contaminated by stray specks far from the
    object; taking the largest component gives a tight, accurate bbox.
    """
    if not mask.any():
        return None
    labels, n = ndimage.label(mask)  # type: ignore[misc]
    if n == 0:
        return None
    # bincount[0] is the background — ignore it
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    rows = np.any(labels == largest, axis=1)
    cols = np.any(labels == largest, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def _pad_bbox(bbox: list[int], img_size: tuple[int, int], frac: float = BBOX_PAD_FRAC) -> list[int]:
    """Inflate bbox by ``frac`` of each side, clamped to image bounds."""
    w, h = img_size
    x1, y1, x2, y2 = bbox
    px = int((x2 - x1) * frac)
    py = int((y2 - y1) * frac)
    return [max(0, x1 - px), max(0, y1 - py), min(w - 1, x2 + px), min(h - 1, y2 + py)]


def _bbox_valid(bbox: list[int], img_size: tuple[int, int]) -> bool:
    """Reject degenerate boxes (too thin, too small relative to image)."""
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
                f"  [skip] insdet/{obj_dir.name}: post-bbox "
                f"{len(support)} support images"
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


def collect_fsod_instances() -> list[dict]:
    """One instance per FSOD category, holding (scene, bbox) entries.

    Stage-0 episodes are synthesised at sample time: pick one entry, crop
    the bbox + rotate four ways → 4 supports; the same scene becomes the
    query (with the same bbox as ground truth). One entry = one entire
    episode, so support_images and query_images both reference the same
    pool — the dataset side knows to treat fsod episodes specially.

    Why this works as instance-level pretraining despite FSOD being
    category-level: when both supports and query come from the *same*
    scene image, the model is matching a specific object instance to its
    own context, not generalising across instances of the same category.
    The 4-way rotation gives viewpoint variation the way HOTS+InsDet
    multi-shot supports would.
    """
    annot_path = FSOD_DIR / "annotations" / "fsod_train.json"
    if not annot_path.exists():
        print(f"  [skip] fsod: {annot_path} not found")
        return []

    with open(annot_path) as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_meta = {im["id"]: im for im in coco["images"]}

    # Group annotations by category. Within a category, dedup by image_id and
    # keep the largest-area bbox per image — multiple bboxes of the same
    # category in one image would generate ambiguous queries.
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
    cat_ids = sorted(by_cat)
    rng.shuffle(cat_ids)
    if FSOD_MAX_CATEGORIES > 0:
        cat_ids = cat_ids[:FSOD_MAX_CATEGORIES]

    instances: list[dict] = []
    for cat_id in cat_ids:
        per_image = by_cat[cat_id]
        entries: list[dict] = []
        for img_id, ann in per_image.items():
            meta = image_meta.get(img_id)
            if meta is None:
                continue
            img_path = FSOD_DIR / meta["file_name"]
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

        # Each entry is a self-contained episode source, so even a single
        # entry would technically work — but keep a small minimum to ensure
        # episodic resampling actually has variety per epoch.
        if len(entries) < 5:
            continue
        rng.shuffle(entries)
        if len(entries) > FSOD_MAX_IMAGES_PER_CAT:
            entries = entries[:FSOD_MAX_IMAGES_PER_CAT]
        name = cat_name[cat_id]
        instances.append(
            {
                "instance_id": f"fsod_{_normalize_name(name)}",
                "source": "fsod",
                "class_name": name,
                # The same flat pool drives both supports and query at sample
                # time. Storing it under both keys keeps the manifest schema
                # uniform with the other sources.
                "support_images": entries,
                "query_images": entries,
            }
        )
    print(f"fsod: {len(instances)} categories kept (of {len(by_cat)} total)")
    return instances


def _normalize_name(name: str) -> str:
    """Normalize an instance name for cross-file matching (case + whitespace)."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


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


def collect_negative_backgrounds(
    fsod_used_image_ids: set[str] | None = None,
) -> list[dict]:
    """Negative (no-target) backgrounds, **tagged by source**.

    HOTS+InsDet share a "hots_insdet" pool (InsDet Background + HOPE
    scenes). HOPE ships only 6D-pose labels (no 2D bboxes), so it can't
    feed positive episodes — its scene RGBs are useful only as distractor
    backgrounds. HOTS itself has no dedicated background dir; the
    HOTS+InsDet pool is shared because the two datasets are visually
    closer to each other than either is to FSOD.

    FSOD has its own pool of random sampled images so a HOTS query is
    never asked to reject an FSOD image (and vice versa) — that domain
    gap is so large the model would learn to discriminate by global
    style alone.
    """
    out: list[dict] = []

    bg_dir = INSDET_DIR / "Background"
    n_insdet_bg = 0
    if bg_dir.exists():
        for p in sorted(bg_dir.glob("*.jpg")):
            out.append({"path": str(p), "source": "hots_insdet"})
            n_insdet_bg += 1

    n_hope = 0
    if HOPE_DIR.exists():
        for p in sorted(HOPE_DIR.rglob("*_rgb.jpg")):
            out.append({"path": str(p), "source": "hots_insdet"})
            n_hope += 1

    n_fsod = 0
    if FSOD_DIR.exists():
        rng = random.Random(SEED + 1)
        used = fsod_used_image_ids or set()
        candidates: list[Path] = []
        for part in ("part_1", "part_2"):
            part_dir = FSOD_DIR / part
            if not part_dir.exists():
                continue
            candidates.extend(part_dir.rglob("*.jpg"))
        # Shuffle to spread across categories rather than skewing to the first
        # alphabetical wnids; then drop any image already used as a positive
        # to keep negatives genuinely disjoint from positives.
        rng.shuffle(candidates)
        for p in candidates:
            if str(p) in used:
                continue
            out.append({"path": str(p), "source": "fsod"})
            n_fsod += 1
            if n_fsod >= FSOD_NEG_SAMPLES:
                break

    print(
        f"  negatives: hots_insdet(insdet_bg={n_insdet_bg} hope={n_hope})"
        f" fsod={n_fsod}"
    )
    return out


def filter_empty_instances(instances: list[dict]) -> list[dict]:
    out = []
    for inst in instances:
        if len(inst["query_images"]) == 0:
            print(f"  [skip] {inst['instance_id']}: no query images after attach")
            continue
        out.append(inst)
    return out


def split_instances(instances: list[dict]) -> tuple[list, list, list]:
    """Stratified 75/10/15 split for HOTS+InsDet; FSOD goes 100% into train.

    HOTS/InsDet are instance-level so val/test on held-out instances measures
    real generalisation. FSOD is category-level (its "instances" are just
    categories), so it has no place in val/test — keeping it train-only also
    matches the stage-0/1 boundary in the training schedule.
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
        if source == "fsod":
            train.extend(group)
            print(f"  split[fsod]: train={n} val=0 test=0  (train-only by design)")
            continue
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
        print(f"  split[{source}]: train={n_train} val={n_val} test={n - n_train - n_val}")
    return train, val, test


def stage_images(
    splits: dict[str, list[dict]], negatives: list[dict]
) -> list[dict]:
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

    # Negatives bucket by source so collisions in basenames across sources
    # (FSOD's tarball-style filenames sometimes clash with HOPE's) don't
    # silently overwrite each other.
    new_negatives: list[dict] = []
    for neg in negatives:
        src = Path(neg["path"])
        source = neg["source"]
        sub_dir = OUT_DIR / "negatives" / source
        sub_dir.mkdir(parents=True, exist_ok=True)
        dst = sub_dir / src.name
        # Disambiguate basename collisions within a source by appending a
        # short hash of the original path.
        if dst.exists():
            import hashlib
            h = hashlib.md5(str(src).encode()).hexdigest()[:8]
            dst = sub_dir / f"{src.stem}_{h}{src.suffix}"
        shutil.copy2(src, dst)
        n_copied += 1
        new_negatives.append({
            "path": (Path("negatives") / source / dst.name).as_posix(),
            "source": source,
        })

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
            "hots": sum(1 for i in insts if i["source"] == "hots"),
            "insdet": sum(1 for i in insts if i["source"] == "insdet"),
            "fsod": sum(1 for i in insts if i["source"] == "fsod"),
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

    print("collecting instances")
    target_instances = collect_hots_instances() + collect_insdet_instances()

    print("collecting scene queries")
    scene = collect_hots_scene_queries() + collect_insdet_scene_queries()
    target_instances = attach_scene_queries(target_instances, scene)
    target_instances = filter_empty_instances(target_instances)

    print("collecting fsod instances (stage-0 pretraining pool)")
    fsod_instances = collect_fsod_instances()

    instances = target_instances + fsod_instances

    # Track FSOD images already used as positives so we don't draw the same
    # file as a negative — keeps the FSOD negative pool disjoint from the
    # FSOD positive pool.
    fsod_used: set[str] = set()
    for inst in fsod_instances:
        for img in inst["support_images"] + inst["query_images"]:
            fsod_used.add(img["path"])

    negatives = collect_negative_backgrounds(fsod_used_image_ids=fsod_used)
    print(f"negatives: {len(negatives)} background images")

    train, val, test = split_instances(instances)
    splits = {"train": train, "val": val, "test": test}

    print("staging images into train / val / test directories")
    negatives = stage_images(splits, negatives)

    write_manifest(splits, negatives)
    write_stats(splits)


if __name__ == "__main__":
    main()
