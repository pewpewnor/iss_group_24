"""Siamese-only inference (existence prediction).

CLI:
    python inference_siamese.py \
        --checkpoint checkpoints/siamese/S2/stage_complete.pt \
        --supports s1.jpg s2.jpg s3.jpg s4.jpg \
        --query    scene.jpg \
        --threshold 0.5

Public API:
    run_siamese(checkpoint, support_paths, query_path, *, threshold=0.5,
                img_size=518, device=None, out_root="inference/siamese",
                smoke=False)
        -> dict {existence_prob, exists}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from shared.checkpoint import load_trainable_state
from shared.dataset import _letterbox
from siamese.model import MultiShotSiamese


def _next_run_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_root / f"{n:04d}").exists():
        n += 1
    p = out_root / f"{n:04d}"
    p.mkdir()
    return p


def _build_model_from_ckpt(ckpt: dict, *, k_max: int) -> MultiShotSiamese:
    cfg = ckpt.get("config", {})
    lora_active = bool(ckpt.get("lora_active", ckpt.get("stage", "") == "S2"))
    m = MultiShotSiamese(
        model_name=cfg.get("dinov2_model_name", "facebook/dinov2-small"),
        k_max=k_max,
        cross_attn_heads=int(cfg.get("cross_attn_heads", 6)),
        cross_attn_dropout=float(cfg.get("cross_attn_dropout", 0.1)),
        head_hidden_1=int(cfg.get("head_hidden_1", 256)),
        head_hidden_2=int(cfg.get("head_hidden_2", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.2)),
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
def run_siamese(
    checkpoint: str | Path,
    support_paths: list[str | Path],
    query_path: str | Path,
    *,
    threshold: float = 0.5,
    img_size: int = 518,
    device: str | None = None,
    out_root: str | Path = "inference/siamese",
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

    model = _build_model_from_ckpt(ckpt, k_max=cfg_k_max).to(device_t)
    model.eval()

    sup_tensors = []
    for sp in support_paths:
        lb, _, _, _ = _letterbox(Image.open(sp).convert("RGB"), img_size)
        sup_tensors.append(TF.to_tensor(lb))
    while len(sup_tensors) < cfg_k_max:
        sup_tensors.append(torch.zeros(3, img_size, img_size))
    sup_t = torch.stack(sup_tensors, dim=0).unsqueeze(0).to(device_t)
    mask = torch.zeros(1, cfg_k_max, dtype=torch.bool, device=device_t)
    mask[0, :K] = True

    qry_pil = Image.open(query_path).convert("RGB")
    qry_lb, _, _, _ = _letterbox(qry_pil, img_size)
    qry_t = TF.to_tensor(qry_lb).unsqueeze(0).to(device_t)

    out = model(sup_t, mask, qry_t)
    prob = float(out["existence_prob"][0].cpu().item())
    exists = bool(prob >= threshold)

    out_dir = _next_run_dir(Path(out_root))
    for idx, sp in enumerate(support_paths, start=1):
        shutil.copy2(str(sp), str(out_dir / f"support_{idx:02d}{Path(sp).suffix}"))
    shutil.copy2(str(query_path), str(out_dir / f"query{Path(query_path).suffix}"))

    payload = {
        "checkpoint": str(checkpoint),
        "stage": ckpt.get("stage"),
        "model_kind": "siamese",
        "n_support": K, "k_max": cfg_k_max,
        "img_size": img_size,
        "existence_prob": prob,
        "threshold": threshold,
        "exists": exists,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[siamese] existence_prob={prob:.4f}  exists={exists}  →  {out_dir}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supports", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-root", default="inference/siamese")
    args = parser.parse_args()
    run_siamese(
        checkpoint=args.checkpoint,
        support_paths=args.supports,
        query_path=args.query,
        threshold=args.threshold,
        img_size=args.img_size,
        device=args.device,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
