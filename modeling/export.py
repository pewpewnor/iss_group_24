"""Export a trained checkpoint to TFLite via litert_torch.

The exported model takes three positional tensor inputs:
  support_imgs   (1, 5, 3, 224, 224)  float32
  support_bboxes (1, 5, 4)            float32  xyxy in 224-px coords
  query_img      (1, 3, 224, 224)     float32

And returns two outputs:
  bbox   (1, 4)  float32  xyxy in 224-px coords
  score  (1,)    float32  confidence in [0, 1]

By default INT8 dynamic-range quantization is applied (weights only),
reducing model size from ~13 MB (float32) to ~4 MB.

Run:
    python -m modeling.export \
        --checkpoint model/best.pt \
        --out        model/model.tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from modeling.model import FewShotLocalizer, decode


class _ExportWrapper(nn.Module):
    """Wraps FewShotLocalizer into a flat tensor-in / tensor-out interface
    that litert_torch.convert (and torch.export) can handle."""

    def __init__(self, model: FewShotLocalizer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_bboxes: torch.Tensor,
        query_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(support_imgs, support_bboxes, query_img)
        bbox, score = decode(out["reg"], out["conf"])
        return bbox, score.unsqueeze(-1)


def export(checkpoint: str | Path, out: str | Path, quantize: bool = True) -> None:
    checkpoint = Path(checkpoint)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    base_model = FewShotLocalizer(pretrained=False)
    base_model.load_state_dict(state["model"] if "model" in state else state)
    base_model.eval()

    wrapper = _ExportWrapper(base_model).eval()

    sample_inputs = (
        torch.zeros(1, 5, 3, 224, 224),
        torch.zeros(1, 5, 4),
        torch.zeros(1, 3, 224, 224),
    )

    print("converting to LiteRT edge model...")
    try:
        import litert_torch
    except ImportError as e:
        raise ImportError(
            "litert_torch is required for export. Install it with: pip install litert-torch"
        ) from e

    if quantize:
        print("  applying INT8 dynamic-range quantization")
        try:
            import tensorflow as tf
            tfl_flags = {"optimizations": [tf.lite.Optimize.DEFAULT]}
        except ImportError:
            print("  warning: tensorflow not found, exporting without quantization")
            tfl_flags = {}
        edge_model = litert_torch.convert(
            wrapper, sample_inputs, _ai_edge_converter_flags=tfl_flags
        )
    else:
        edge_model = litert_torch.convert(wrapper, sample_inputs)

    print(f"exporting to: {out}")
    edge_model.export(str(out))

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"done. model size: {size_mb:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to best.pt")
    p.add_argument("--out", default="model/model.tflite", help="output .tflite path")
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="skip INT8 dynamic-range quantization (larger file, full float32)",
    )
    args = p.parse_args()

    export(
        checkpoint=args.checkpoint,
        out=args.out,
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
