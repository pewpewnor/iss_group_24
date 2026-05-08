"""Inference script for the OWLv2 few-shot localizer.

Public function:

    run_inference(
        checkpoint        : str | Path,            # *.pt produced by train.py
        support_paths     : list[str | Path],      # exactly 4 image paths
        query_path        : str | Path,
        *,
        out_root          = "inference",           # directory under which to write
        img_size          = 768,
        device            = None,                  # auto cuda/cpu
        existence_thr     = 0.5,                   # gate the bbox draw on this
        tile_cfg          = None,                  # see modeling.evaluate.DEFAULT_TILE_CFG
        bbox_color        = (0, 255, 0),
        bbox_thickness    = 4,
        tile_for_query    = True,                  # when True, always tile the query
                                                   # regardless of tile_cfg["for_sources"]
    ) -> dict

The function:
  1. Loads the checkpoint (auto-detecting LoRA from the stage tag).
  2. Loads the 4 supports + 1 query as PIL images.
  3. Runs detection — single-pass (`tile_cfg["mode"] == "off"`)
     or tiled (`pyramid_a` / `hybrid_d`).
  4. Saves *all 5 input images* unchanged + the query annotated with the
     predicted bbox and an "EXISTS"/"NOT EXISTS" caption to
     ``inference/<NNNN>/`` where NNNN is the next free incrementing index.
  5. Writes a ``result.json`` with all the numerics.

Use as a CLI:

    uv run python inference.py \\
        --checkpoint checkpoints/stage_2_3/best.pt \\
        --supports s1.jpg s2.jpg s3.jpg s4.jpg \\
        --query    scene.jpg \\
        --tile-mode pyramid_a

    python inference.py --checkpoint checkpoints/stage_1_1/last.pt --supports dataset/aggregated/test/support/hots_banana/001.png dataset/aggregated/test/support/hots_banana/002.png dataset/aggregated/test/support/hots_banana/003.png dataset/aggregated/test/support/hots_banana/004.png --query dataset/aggregated/test/query/hots_banana/001.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from modeling._checkpoint import try_load_model_state
from modeling._tiling import detect_tiled
from modeling.dataset import OWLV2_MEAN, OWLV2_STD
from modeling.evaluate import resolve_tile_cfg
from modeling.model import OWLv2FewShotLocalizer, cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _next_run_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    used: set[int] = set()
    for p in out_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            used.add(int(p.name))
    n = 1
    while n in used:
        n += 1
    run = out_root / f"{n:04d}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def _normalize_query_for_model(pil_img: Image.Image, img_size: int) -> torch.Tensor:
    """Resize square + OWLv2-normalise.  Returns (3, S, S)."""
    resized = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    t = TF.to_tensor(resized)
    return TF.normalize(t, OWLV2_MEAN, OWLV2_STD)


def _normalize_supports_for_model(
    paths: list[Path], img_size: int
) -> tuple[torch.Tensor, list[Image.Image]]:
    """Load supports, resize, normalise, return tensor + original PILs."""
    pil_list: list[Image.Image] = []
    tensors: list[torch.Tensor] = []
    for p in paths:
        pil = Image.open(p).convert("RGB")
        pil_list.append(pil)
        resized = pil.resize((img_size, img_size), Image.Resampling.BILINEAR)
        t = TF.to_tensor(resized)
        tensors.append(TF.normalize(t, OWLV2_MEAN, OWLV2_STD))
    return torch.stack(tensors, dim=0), pil_list


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------


def _draw_caption_and_box(
    pil: Image.Image,
    bbox_xyxy_native: tuple[float, float, float, float] | None,
    caption: str,
    *,
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 4,
) -> Image.Image:
    """Returns a copy of pil with the bbox + caption drawn on top."""
    out = pil.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size

    # Pick a font that scales with image size; fall back to default if no
    # truetype is installed.
    font_size = max(16, w // 32)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    # bbox first (so caption sits on top).
    if bbox_xyxy_native is not None:
        x1, y1, x2, y2 = bbox_xyxy_native
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y2 = max(0, min(h - 1, int(round(y2))))
        if x2 > x1 and y2 > y1:
            for k in range(bbox_thickness):
                draw.rectangle(
                    [x1 - k, y1 - k, x2 + k, y2 + k],
                    outline=bbox_color,
                )

    # Caption: top-left filled rect with text on top.
    if hasattr(draw, "textbbox"):
        tb = draw.textbbox((0, 0), caption, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
    else:
        text_w, text_h = draw.textsize(caption, font=font)
    pad = 6
    pos_x, pos_y = pad, pad
    rect = [pos_x - 2, pos_y - 2, pos_x + text_w + pad, pos_y + text_h + pad]
    # Translucent dark background.
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle(rect, fill=(0, 0, 0, 180))
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.text((pos_x, pos_y), caption, fill=(255, 255, 255), font=font)
    return out


# ---------------------------------------------------------------------------
# Detection paths
# ---------------------------------------------------------------------------


@torch.no_grad()
def _detect_single_pass(
    model: OWLv2FewShotLocalizer,
    support_t: torch.Tensor,
    query_pil: Image.Image,
    img_size: int,
    device: torch.device,
) -> dict:
    """Single-pass detection (no tiling).  Returns native-coord bbox xyxy."""
    nw, nh = query_pil.size
    qt = _normalize_query_for_model(query_pil, img_size).unsqueeze(0).to(device)
    out = model(support_t.unsqueeze(0).to(device), qt)
    pred_box_norm = out["best_box"][0]                            # (4,) cxcywh in [0,1]
    xyxy_norm = cxcywh_to_xyxy(pred_box_norm).clamp(0, 1)
    bbox_native = (
        float(xyxy_norm[0]) * nw,
        float(xyxy_norm[1]) * nh,
        float(xyxy_norm[2]) * nw,
        float(xyxy_norm[3]) * nh,
    )
    return {
        "existence_prob": float(out["existence_prob"][0].item()),
        "best_score_logit": float(out["best_score"][0].item()),
        "best_score": float(torch.sigmoid(out["best_score"][0]).item()),
        "bbox_xyxy_native": bbox_native,
        "tile_mode": "off",
    }


@torch.no_grad()
def _detect_tiled_path(
    model: OWLv2FewShotLocalizer,
    support_t: torch.Tensor,
    query_pil: Image.Image,
    img_size: int,
    device: torch.device,
    tile_cfg: dict,
) -> dict:
    tiled = detect_tiled(
        model,
        support_t.unsqueeze(0).to(device),
        query_pil,
        img_size=img_size,
        mode=str(tile_cfg["mode"]),
        levels=tuple(tile_cfg["levels"]),
        overlap=float(tile_cfg["overlap"]),
        nms_iou=float(tile_cfg["nms_iou"]),
        top_k=int(tile_cfg["top_k"]),
        score_combo=str(tile_cfg["score_combo"]),
        edge_score_penalty=float(tile_cfg.get("edge_score_penalty", 0.5)),
        edge_px=int(tile_cfg.get("edge_px", 4)),
        merge_partial_boxes=bool(tile_cfg.get("merge_partial_boxes", True)),
        merge_min_score=float(tile_cfg.get("merge_min_score", 0.2)),
    )
    bbox = tiled["best_box_native_xyxy"].cpu().tolist()
    return {
        "existence_prob": float(tiled["existence_prob"].item()),
        "best_score_logit": float("nan"),                          # not directly available
        "best_score": float(tiled["best_score"].item()),
        "bbox_xyxy_native": tuple(bbox),
        "tile_mode": str(tile_cfg["mode"]),
        "n_detections_after_nms": int(tiled["all_boxes_native_xyxy"].size(0)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_inference(
    checkpoint: str | Path,
    support_paths: list[str | Path],
    query_path: str | Path,
    *,
    out_root: str | Path = "inference",
    img_size: int = 768,
    device: str | None = None,
    existence_thr: float = 0.5,
    tile_cfg: dict | None = None,
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 4,
    tile_for_query: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_layers: int = 4,
) -> dict[str, Any]:
    """Run inference on (4 supports, 1 query) and write annotated outputs.

    Outputs land in ``out_root/<NNNN>/``:
        support_1.jpg .. support_4.jpg     (originals copied)
        query.jpg                          (original copied)
        result.jpg                         (query with bbox + caption drawn)
        result.json                        (numeric outputs + config)

    Returns the result dict (also written to result.json).
    """
    if len(support_paths) != 4:
        raise ValueError(f"expected exactly 4 support paths, got {len(support_paths)}")
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device_t = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = resolve_tile_cfg(tile_cfg)

    # Build the inference output directory now so any partial run is visible.
    out_root_p = Path(out_root)
    run_dir = _next_run_dir(out_root_p)

    # Load model.
    print(f"loading checkpoint: {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    stage = ckpt.get("stage")
    print(f"  checkpoint stage = {stage}")
    model = OWLv2FewShotLocalizer()
    if stage == "2_3":
        model.attach_lora(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            last_n_layers=lora_layers,
        )
    ok, err = try_load_model_state(model, ckpt.get("model", {}))
    if not ok:
        raise RuntimeError(f"failed to load checkpoint: {err}")
    model = model.to(device_t).eval()

    # Load images.
    support_paths_p = [Path(p) for p in support_paths]
    query_path_p = Path(query_path)
    for p in (*support_paths_p, query_path_p):
        if not p.exists():
            raise FileNotFoundError(p)
    support_t, support_pils = _normalize_supports_for_model(support_paths_p, img_size)
    query_pil = Image.open(query_path_p).convert("RGB")

    # Detect.
    do_tile = (cfg["mode"] != "off") and tile_for_query
    if do_tile:
        det = _detect_tiled_path(
            model, support_t, query_pil, img_size, device_t, cfg
        )
    else:
        det = _detect_single_pass(
            model, support_t, query_pil, img_size, device_t
        )

    exists = det["existence_prob"] >= existence_thr
    bbox_to_draw = det["bbox_xyxy_native"] if exists else None
    caption = (
        f"EXISTS  (p={det['existence_prob']:.2f})"
        if exists
        else f"NOT EXISTS  (p={det['existence_prob']:.2f})"
    )

    # Save originals.
    for i, (p, pil) in enumerate(zip(support_paths_p, support_pils), start=1):
        suffix = p.suffix.lower() if p.suffix else ".jpg"
        if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
            suffix = ".jpg"
        try:
            shutil.copy2(str(p), str(run_dir / f"support_{i}{suffix}"))
        except (shutil.SameFileError, OSError):
            pil.save(run_dir / f"support_{i}{suffix}")
    qsuffix = query_path_p.suffix.lower() if query_path_p.suffix else ".jpg"
    if qsuffix not in (".jpg", ".jpeg", ".png", ".webp"):
        qsuffix = ".jpg"
    try:
        shutil.copy2(str(query_path_p), str(run_dir / f"query{qsuffix}"))
    except (shutil.SameFileError, OSError):
        query_pil.save(run_dir / f"query{qsuffix}")

    # Save annotated result.
    annotated = _draw_caption_and_box(
        query_pil,
        bbox_to_draw,
        caption,
        bbox_color=bbox_color,
        bbox_thickness=bbox_thickness,
    )
    annotated.save(run_dir / "result.jpg", quality=92)

    result = {
        "checkpoint": str(checkpoint),
        "stage": stage,
        "supports": [str(p) for p in support_paths_p],
        "query": str(query_path_p),
        "img_size": img_size,
        "existence_prob": det["existence_prob"],
        "exists": bool(exists),
        "existence_threshold": existence_thr,
        "best_score": det["best_score"],
        "best_score_logit": det["best_score_logit"],
        "bbox_xyxy_native": list(det["bbox_xyxy_native"]),
        "drew_bbox": bool(exists),
        "caption": caption,
        "tile_mode": det["tile_mode"],
        "tile_cfg": cfg,
        "out_dir": str(run_dir),
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"  → wrote {run_dir}/result.jpg  (exists={exists}, p={det['existence_prob']:.3f})")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_levels(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Run OWLv2 few-shot inference.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a *.pt checkpoint produced by train.py.")
    p.add_argument("--supports", nargs=4, required=True,
                   help="Exactly 4 support image paths.")
    p.add_argument("--query", required=True,
                   help="Query (scene) image path.")
    p.add_argument("--out-root", default="inference",
                   help="Output directory under which to create <NNNN>/ runs.")
    p.add_argument("--img-size", type=int, default=768)
    p.add_argument("--device", default=None)
    p.add_argument("--existence-thr", type=float, default=0.5)
    p.add_argument("--tile-mode", default="pyramid_a",
                   choices=["off", "pyramid_a", "hybrid_d"],
                   help="Tile inference mode (default: pyramid_a).")
    p.add_argument("--tile-levels", default="1,2",
                   help="Comma-separated pyramid levels (default: 1,2).")
    p.add_argument("--tile-overlap", type=float, default=0.30)
    p.add_argument("--tile-no-merge", action="store_true",
                   help="Disable boundary-spanning detection merge (M3).")
    p.add_argument("--tile-edge-penalty", type=float, default=0.5,
                   help="Edge-hugging detection score multiplier (M4); 1.0 disables.")
    p.add_argument("--bbox-color", default="0,255,0",
                   help="R,G,B for the predicted bbox.")
    p.add_argument("--bbox-thickness", type=int, default=4)
    args = p.parse_args()

    tile_cfg = {
        "mode": args.tile_mode,
        "levels": _parse_levels(args.tile_levels),
        "overlap": args.tile_overlap,
        "merge_partial_boxes": not args.tile_no_merge,
        "edge_score_penalty": args.tile_edge_penalty,
    }
    bbox_color = tuple(int(x) for x in args.bbox_color.split(","))
    if len(bbox_color) != 3:
        raise ValueError("--bbox-color must be R,G,B")

    run_inference(
        checkpoint=args.checkpoint,
        support_paths=args.supports,
        query_path=args.query,
        out_root=args.out_root,
        img_size=args.img_size,
        device=args.device,
        existence_thr=args.existence_thr,
        tile_cfg=tile_cfg,
        bbox_color=bbox_color,                                     # type: ignore[arg-type]
        bbox_thickness=args.bbox_thickness,
    )


if __name__ == "__main__":
    main()
