"""Localizer-only inference.

CLI:
    python inference_localizer.py \
        --checkpoint checkpoints/localizer/L3/stage_complete.pt \
        --supports s1.jpg s2.jpg s3.jpg s4.jpg \
        --query    scene.jpg

Public API:
    run_localize(checkpoint, support_paths, query_path, *, img_size=768,
                 device=None, out_root="inference/localizer", smoke=False)
        -> dict with predicted bbox in {cxcywh_norm, xyxy_native} forms.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from localizer.model import MultiShotLocalizer
from shared.checkpoint import load_trainable_state
from shared.dataset import _letterbox


def _next_run_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_root / f"{n:04d}").exists():
        n += 1
    p = out_root / f"{n:04d}"
    p.mkdir()
    return p


def _load_image(p: str | Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def _draw_bbox(img: Image.Image, bbox_xyxy: tuple[float, float, float, float],
               *, color=(0, 255, 0), thickness: int = 4,
               caption: str | None = None) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox_xyxy
    for off in range(thickness):
        draw.rectangle([x1 + off, y1 + off, x2 - off, y2 - off], outline=color)
    if caption:
        draw.rectangle([x1, max(0, y1 - 28), x1 + 8 * len(caption), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 24)), caption, fill=(0, 0, 0))
    return out


def _build_model_from_ckpt(
    ckpt: dict, *, k_max: int, img_size: int,
) -> MultiShotLocalizer:
    cfg = ckpt.get("config", {})
    lora_active = bool(ckpt.get("lora_active", ckpt.get("stage", "") == "L3"))
    m = MultiShotLocalizer(
        model_name=cfg.get("owlv2_model_name", "google/owlv2-base-patch16-ensemble"),
        k_max=k_max,
        fusion_layers=int(cfg.get("fusion_layers", 2)),
        fusion_heads=int(cfg.get("fusion_heads", 8)),
        fusion_mlp_ratio=int(cfg.get("fusion_mlp_ratio", 2)),
        fusion_dropout=float(cfg.get("fusion_dropout", 0.1)),
    )
    if lora_active:
        m.attach_lora(
            r=int(cfg.get("lora_r", 8)),
            alpha=int(cfg.get("lora_alpha", 16)),
            dropout=float(cfg.get("lora_dropout", 0.1)),
            last_n_layers=int(cfg.get("lora_last_n_layers", 4)),
        )
    load_trainable_state(m, ckpt.get("state_dict", {}))
    return m


@torch.no_grad()
def run_localize(
    checkpoint: str | Path,
    support_paths: list[str | Path],
    query_path: str | Path,
    *,
    img_size: int = 768,
    device: str | None = None,
    out_root: str | Path = "inference/localizer",
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 4,
    smoke: bool = False,
) -> dict[str, Any]:
    if smoke:
        img_size = 224
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not support_paths:
        raise ValueError("support_paths must contain at least 1 image path")
    if not Path(query_path).exists():
        raise FileNotFoundError(f"query not found: {query_path}")
    for sp in support_paths:
        if not Path(sp).exists():
            raise FileNotFoundError(f"support not found: {sp}")
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg_k_max = int(ckpt.get("config", {}).get("k_max", 10))
    if smoke:
        cfg_k_max = max(2, len(support_paths))
    K = len(support_paths)
    if K > cfg_k_max:
        raise ValueError(f"Too many supports ({K}); model trained with k_max={cfg_k_max}")

    model = _build_model_from_ckpt(ckpt, k_max=cfg_k_max, img_size=img_size).to(device_t)
    model.eval()

    # Load images.
    sup_pils = [_load_image(p) for p in support_paths]
    sup_tensors = []
    for s in sup_pils:
        lb, _, _, _ = _letterbox(s, img_size)
        sup_tensors.append(TF.to_tensor(lb))
    while len(sup_tensors) < cfg_k_max:
        sup_tensors.append(torch.zeros(3, img_size, img_size))
    sup_t = torch.stack(sup_tensors, dim=0).unsqueeze(0).to(device_t)        # (1, K_max, 3, S, S)
    mask = torch.zeros(1, cfg_k_max, dtype=torch.bool, device=device_t)
    mask[0, :K] = True

    qry_pil = _load_image(query_path)
    nw, nh = qry_pil.size
    qry_lb, q_scale, q_pad_left, q_pad_top = _letterbox(qry_pil, img_size)
    qry_t = TF.to_tensor(qry_lb).unsqueeze(0).to(device_t)

    out = model(sup_t, mask, qry_t)
    cx, cy, w, h = out["best_box"][0].cpu().tolist()
    score = float(out["best_score"][0].cpu().item())

    # Map back to native query coords.
    x1_lb = (cx - w / 2) * img_size
    y1_lb = (cy - h / 2) * img_size
    x2_lb = (cx + w / 2) * img_size
    y2_lb = (cy + h / 2) * img_size
    x1_n = (x1_lb - q_pad_left) / q_scale
    y1_n = (y1_lb - q_pad_top) / q_scale
    x2_n = (x2_lb - q_pad_left) / q_scale
    y2_n = (y2_lb - q_pad_top) / q_scale
    x1_n = max(0.0, min(nw, x1_n))
    y1_n = max(0.0, min(nh, y1_n))
    x2_n = max(0.0, min(nw, x2_n))
    y2_n = max(0.0, min(nh, y2_n))

    out_dir = _next_run_dir(Path(out_root))
    # Save inputs unchanged.
    for idx, sp in enumerate(support_paths, start=1):
        shutil.copy2(str(sp), str(out_dir / f"support_{idx:02d}{Path(sp).suffix}"))
    shutil.copy2(str(query_path), str(out_dir / f"query{Path(query_path).suffix}"))
    annotated = _draw_bbox(
        qry_pil, (x1_n, y1_n, x2_n, y2_n),
        color=bbox_color, thickness=bbox_thickness,
        caption=f"score={score:.2f}",
    )
    annotated.save(str(out_dir / "result.png"))

    payload = {
        "checkpoint": str(checkpoint),
        "stage": ckpt.get("stage"),
        "model_kind": "localizer",
        "n_support": K, "k_max": cfg_k_max,
        "img_size": img_size,
        "best_box_cxcywh_norm": [cx, cy, w, h],
        "best_box_xyxy_native": [x1_n, y1_n, x2_n, y2_n],
        "best_score": score,
        "native_size": [nw, nh],
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[localizer] result → {out_dir}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supports", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--img-size", type=int, default=768)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-root", default="inference/localizer")
    args = parser.parse_args()
    run_localize(
        checkpoint=args.checkpoint,
        support_paths=args.supports,
        query_path=args.query,
        img_size=args.img_size,
        device=args.device,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
