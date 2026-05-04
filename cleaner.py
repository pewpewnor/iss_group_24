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


def collect_negative_backgrounds() -> list[str]:
    bg_dir = INSDET_DIR / "Background"
    if not bg_dir.exists():
        return []
    return [str(p) for p in sorted(bg_dir.glob("*.jpg"))]


def filter_empty_instances(instances: list[dict]) -> list[dict]:
    out = []
    for inst in instances:
        if len(inst["query_images"]) == 0:
            print(f"  [skip] {inst['instance_id']}: no query images after attach")
            continue
        out.append(inst)
    return out


def split_instances(instances: list[dict]) -> tuple[list, list, list]:
    """Stratified 80/10/10 split — preserves the source ratio (HOTS/InsDet)
    in each split. A pure random shuffle on a small dataset can put most of
    one source in a single split, biasing val/test metrics."""
    rng = random.Random(SEED)
    by_source: dict[str, list[dict]] = {}
    for inst in instances:
        by_source.setdefault(inst["source"], []).append(inst)

    train, val, test = [], [], []
    for source in sorted(by_source):
        group = by_source[source][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
        print(f"  split[{source}]: train={n_train} val={n_val} test={n - n_train - n_val}")
    return train, val, test


def stage_images(splits: dict[str, list[dict]], negatives: list[str]) -> list[str]:
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

    neg_dir = OUT_DIR / "negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    new_negatives = []
    for p in negatives:
        dst = neg_dir / Path(p).name
        shutil.copy2(p, dst)
        n_copied += 1
        new_negatives.append((Path("negatives") / Path(p).name).as_posix())

    print(f"staged images: {n_copied} files copied")
    return new_negatives


def write_manifest(splits: dict[str, list[dict]], negatives: list[str]) -> None:
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
    instances = collect_hots_instances() + collect_insdet_instances()

    print("collecting scene queries")
    scene = collect_hots_scene_queries() + collect_insdet_scene_queries()
    instances = attach_scene_queries(instances, scene)
    instances = filter_empty_instances(instances)

    negatives = collect_negative_backgrounds()
    print(f"negatives: {len(negatives)} background images")

    train, val, test = split_instances(instances)
    splits = {"train": train, "val": val, "test": test}

    print("staging images into train / val / test directories")
    negatives = stage_images(splits, negatives)

    write_manifest(splits, negatives)
    write_stats(splits)


if __name__ == "__main__":
    main()
