"""Export the trained OWLv2 few-shot localizer to ONNX (and optionally TFLite).

Target formats:
  * ONNX           — primary export.  ONNX opset 17 covers all ops we use
                     (attention via MHA, ELU, GELU, sigmoid, einsum).
  * TFLite (opt.)  — produced by converting the ONNX through ``onnx2tf``,
                     since OWLv2's attention stack does not have a clean
                     direct TFLite path.  This is best-effort: TFLite has
                     reduced transformer op coverage and will fall back to
                     SELECT_TF_OPS for some operators.

Usage:

    uv run python export.py \\
        --checkpoint checkpoints/stage_2_3/best.pt \\
        --out-dir exports/ \\
        --img-size 768 \\
        --formats onnx                      # or "onnx,tflite"

What gets exported:
  Two ONNX graphs are produced because OWLv2's frozen ViT is huge (~600MB)
  and ONNX inlines all weights:

    1. ``owlv2_localizer.onnx``  — full forward graph.
       Inputs:
           support_imgs  (1, 4, 3, S, S)  fp32 (CLIP-normalised)
           query_img     (1, 3, S, S)     fp32
       Outputs:
           existence_prob (1,)
           best_box       (1, 4)          cxcywh in [0,1]
           best_score     (1,)
           pred_logits    (1, P)
           pred_boxes     (1, P, 4)
           prototype      (1, query_dim)

The script will refuse to export when the live ``transformers`` version
emits an op that ONNX exporter cannot trace; in that case run with
``--use-dynamo`` to switch to the new ``torch.onnx.dynamo_export`` path.

LoRA adapters (Stage 2.3 checkpoints) are merged into the underlying
weights via ``peft.PeftModel.merge_and_unload`` before tracing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

from modeling._checkpoint import try_load_model_state
from modeling.model import OWLv2FewShotLocalizer


# ---------------------------------------------------------------------------
# Wrapper module — flattens the dict output to a tuple for ONNX
# ---------------------------------------------------------------------------


class _ExportWrapper(nn.Module):
    """ONNX-friendly wrapper: returns a fixed tuple instead of a dict."""

    def __init__(self, model: OWLv2FewShotLocalizer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, support_imgs: torch.Tensor, query_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(support_imgs, query_img)
        return (
            out["existence_prob"],
            out["best_box"],
            out["best_score"],
            out["pred_logits"],
            out["pred_boxes"],
            out["prototype"],
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_checkpoint(
    checkpoint: Path, img_size: int, lora_cfg: dict | None
) -> OWLv2FewShotLocalizer:
    print(f"loading checkpoint: {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    stage = ckpt.get("stage")
    print(f"  stage = {stage}")
    model = OWLv2FewShotLocalizer()
    if stage == "2_3":
        cfg = lora_cfg or {}
        model.attach_lora(
            r=int(cfg.get("r", 8)),
            alpha=int(cfg.get("alpha", 16)),
            dropout=float(cfg.get("dropout", 0.1)),
            last_n_layers=int(cfg.get("last_n_layers", 4)),
        )
    ok, err = try_load_model_state(model, ckpt.get("model", {}))
    if not ok:
        raise RuntimeError(f"failed to load checkpoint: {err}")

    # Merge LoRA weights into the base model so the export graph has a
    # single frozen ViT with the merged weights.
    if stage == "2_3":
        try:
            from peft import PeftModel

            if isinstance(model.owlv2, PeftModel):
                print("  merging LoRA adapters into base weights")
                model.owlv2 = model.owlv2.merge_and_unload()
        except ImportError:
            pass

    model.eval()
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(
    model: OWLv2FewShotLocalizer,
    out_path: Path,
    img_size: int,
    use_dynamo: bool = False,
    opset: int = 17,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _ExportWrapper(model)
    wrapper.eval()

    support_dummy = torch.zeros(1, 4, 3, img_size, img_size, dtype=torch.float32)
    query_dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)

    print(f"tracing ONNX (opset={opset}, dynamo={use_dynamo}) → {out_path}")
    if use_dynamo:
        prog = torch.onnx.dynamo_export(wrapper, support_dummy, query_dummy)
        prog.save(str(out_path))
    else:
        torch.onnx.export(
            wrapper,
            (support_dummy, query_dummy),
            str(out_path),
            input_names=["support_imgs", "query_img"],
            output_names=[
                "existence_prob",
                "best_box",
                "best_score",
                "pred_logits",
                "pred_boxes",
                "prototype",
            ],
            opset_version=opset,
            dynamic_axes={
                "support_imgs": {0: "batch"},
                "query_img": {0: "batch"},
                "existence_prob": {0: "batch"},
                "best_box": {0: "batch"},
                "best_score": {0: "batch"},
                "pred_logits": {0: "batch", 1: "patches"},
                "pred_boxes": {0: "batch", 1: "patches"},
                "prototype": {0: "batch"},
            },
            do_constant_folding=True,
        )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path} ({size_mb:.1f} MB)")


def verify_onnx(onnx_path: Path) -> None:
    """Run a quick onnx.checker pass."""
    try:
        import onnx
    except ImportError:
        print("  [skip] onnx not installed — skipping verification")
        return
    print(f"verifying {onnx_path}")
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    print("  ONNX graph is valid")


# ---------------------------------------------------------------------------
# TFLite export (best-effort, via onnx2tf)
# ---------------------------------------------------------------------------


def export_tflite(onnx_path: Path, out_dir: Path) -> Path | None:
    """Convert ONNX → TFLite via the onnx2tf intermediate.

    Returns the path to the .tflite file, or None on failure.
    """
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        print(
            "  [skip] onnx2tf not installed.  Install with `pip install onnx2tf` "
            "to enable TFLite export."
        )
        return None

    tf_dir = out_dir / "tf_saved_model"
    tflite_path = out_dir / f"{onnx_path.stem}.tflite"
    if tf_dir.exists():
        shutil.rmtree(tf_dir)

    print(f"converting {onnx_path} → TFLite via onnx2tf")
    import subprocess

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(tf_dir),
        "-osd",                                      # output signaturedef
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("  onnx2tf failed:")
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        return None
    # onnx2tf emits a default tflite under tf_dir with the same stem.
    candidates = list(tf_dir.glob("*.tflite"))
    if not candidates:
        print("  no .tflite produced")
        return None
    shutil.copy2(str(candidates[0]), str(tflite_path))
    size_mb = tflite_path.stat().st_size / 1e6
    print(f"  wrote {tflite_path} ({size_mb:.1f} MB)")
    return tflite_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Export the OWLv2 few-shot localizer.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a checkpoint (e.g. checkpoints/stage_2_3/best.pt).")
    p.add_argument("--out-dir", default="exports",
                   help="Directory to write the exported files into.")
    p.add_argument("--img-size", type=int, default=768,
                   help="Input image size used during training (768 or 960).")
    p.add_argument("--formats", default="onnx",
                   help="Comma-separated formats: onnx,tflite. (Default: onnx)")
    p.add_argument("--use-dynamo", action="store_true",
                   help="Use torch.onnx.dynamo_export instead of legacy tracer.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-layers", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}

    lora_cfg = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "last_n_layers": args.lora_layers,
    }
    model = _load_checkpoint(Path(args.checkpoint), args.img_size, lora_cfg)

    onnx_path = out_dir / "owlv2_localizer.onnx"

    if "onnx" in formats or "tflite" in formats:
        export_onnx(model, onnx_path, args.img_size, args.use_dynamo, args.opset)
        verify_onnx(onnx_path)

    if "tflite" in formats:
        export_tflite(onnx_path, out_dir)

    # Sidecar JSON with provenance.
    sidecar = out_dir / "export_info.json"
    info = {
        "checkpoint": str(args.checkpoint),
        "img_size": args.img_size,
        "formats": sorted(formats),
        "opset": args.opset,
        "lora_cfg": lora_cfg,
    }
    sidecar.write_text(json.dumps(info, indent=2))
    print(f"export complete → {out_dir}")


if __name__ == "__main__":
    main()
