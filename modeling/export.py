"""Export a trained checkpoint to TFLite via litert_torch.

Two artefacts are exported for the on-device pipeline:

  prototype.tflite — runs once when the user provides their 5 supports.
      input:  support_imgs (1, 5, 3, 224, 224)
      output: prototype    (1, K*M, DIM)   token bag — typically (1, 20, 128)

  detect.tflite — runs every camera frame, reusing the cached prototype.
      input:  prototype    (1, K*M, DIM)
              query_img    (1, 3, 224, 224)
      output: bbox         (1, 4)   xyxy in 224-px coords
              score        (1,)     presence confidence in [0, 1]

Splitting the pipeline avoids re-encoding 5 supports per frame at runtime —
the token bag is cached client-side and the per-frame cost drops to just
the query branch + cross-attention head + detection head.

Run:
    python -m modeling.export \
        --checkpoint model/best.pt \
        --out-dir    model/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from modeling.model import DIM, M_TOKENS, FewShotLocalizer, decode

K_SUPPORT = 4


class _PrototypeWrapper(nn.Module):
    """Encodes K support images into a (1, K*M, DIM) token bag. Run once per session."""

    def __init__(self, model: FewShotLocalizer) -> None:
        super().__init__()
        self.model = model

    def forward(self, support_imgs: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.model.encode_support(support_imgs)
        return tokens


class _DetectWrapper(nn.Module):
    """Per-frame detection: cached prototype + new query → (bbox, score)."""

    def __init__(self, model: FewShotLocalizer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, prototype: torch.Tensor, query_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.detect(prototype, query_img)
        bbox, score = decode(out["reg"], out["conf"], presence_logit=out["presence_logit"])
        return bbox, score.unsqueeze(-1)


def _convert(
    wrapper: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    out: Path,
    quantize: bool,
) -> None:
    """Run litert_torch conversion + export, with optional INT8 weight quantization."""
    try:
        import litert_torch
    except ImportError as e:
        raise ImportError(
            "litert_torch is required for export. Install it with: pip install litert-torch"
        ) from e

    if quantize:
        try:
            import tensorflow as tf  # type: ignore[import-untyped]
            tfl_flags = {"optimizations": [tf.lite.Optimize.DEFAULT]}
        except ImportError:
            print("  warning: tensorflow not found, exporting without quantization")
            tfl_flags = {}
        edge_model = litert_torch.convert(
            wrapper, sample_inputs, _ai_edge_converter_flags=tfl_flags
        )
    else:
        edge_model = litert_torch.convert(wrapper, sample_inputs)

    edge_model.export(str(out))
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  {out.name}: {size_mb:.1f} MB")


def export(checkpoint: str | Path, out_dir: str | Path, quantize: bool = True) -> None:
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_model = FewShotLocalizer(pretrained=False)
    base_model.load_state_dict(state["model"] if "model" in state else state)
    base_model.eval()

    if quantize:
        print("  applying INT8 dynamic-range quantization")

    print("exporting prototype model (one-shot, runs when user uploads supports)...")
    _convert(
        _PrototypeWrapper(base_model).eval(),
        (torch.zeros(1, 5, 3, 224, 224),),
        out_dir / "prototype.tflite",
        quantize,
    )

    print("exporting detect model (per-frame, runs on every camera frame)...")
    _convert(
        _DetectWrapper(base_model).eval(),
        (
            torch.zeros(1, K_SUPPORT * M_TOKENS, DIM),
            torch.zeros(1, 3, 224, 224),
        ),
        out_dir / "detect.tflite",
        quantize,
    )

    print("done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to best.pt")
    p.add_argument("--out-dir", default="model", help="output directory for the two .tflite files")
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="skip INT8 dynamic-range quantization (larger files, full float32)",
    )
    args = p.parse_args()

    export(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
