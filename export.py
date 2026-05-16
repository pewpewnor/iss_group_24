"""Export localizer and siamese checkpoints to ONNX (default) or TFLite.

Conversion chain
----------------
PyTorch → ONNX → TensorFlow SavedModel → TFLite flatbuffer

This is the only reliable path for these models because:
  - OWLv2 / DINOv2 use ``interpolate_pos_encoding=True`` which prevents
    pure TorchScript tracing.
  - ``torch.onnx.dynamo_export`` (torch >= 2.1) handles the dynamic calls via
    the FX-based exporter and produces a static-shape ONNX graph.
  - ``onnx-tf`` converts the ONNX graph to a TF SavedModel.
  - ``tensorflow`` lite converter finalises the TFLite flatbuffer.

Required packages (in addition to the training dependencies)::

    pip install onnx onnxscript onnx-tf tensorflow

Model inputs (identical for both models, after letterboxing)
-------------------------------------------------------------
All pixel values are **raw float32 in [0, 1]** — normalisation is baked
into the model graph so the back-end does not need to apply it.

  support_imgs : float32  (1, k_max, 3, img_size, img_size)
  support_mask : float32  (1, k_max)          1.0 = real slot, 0.0 = padding
  query_img    : float32  (1, 3, img_size, img_size)

``support_mask`` is exported as float32 rather than bool because TFLite does
not support bool input tensors; the model wrapper casts it back to bool
before the forward pass.

Pre-processing the inputs (back-end / front-end responsibility)
---------------------------------------------------------------
1. Resize the image to (img_size, img_size) with **letterbox padding**
   (preserve aspect ratio, pad with grey 114/255 ≈ 0.447):

       scale = img_size / max(orig_w, orig_h)
       new_w = round(orig_w * scale)
       new_h = round(orig_h * scale)
       pad_left = (img_size - new_w) // 2
       pad_top  = (img_size - new_h) // 2

2. Convert to float32 and divide by 255.
3. Arrange channels as CHW (channel-first) in RGB order.

Localizer outputs
-----------------
  best_box   : float32  (1, 4)  — (cx, cy, w, h) normalised to [0, 1]
                                   relative to the letterboxed img_size square.
  best_score : float32  (1,)    — foreground softmax probability ∈ [0, 1].
  bg_prob    : float32  (1,)    — background/abstain softmax probability ∈ [0, 1].
                                   Use ``bg_prob >= abstain_threshold`` (e.g. 0.5)
                                   to decide whether to suppress the box.

Converting best_box back to native-resolution pixel coordinates::

    cx_lb, cy_lb = cx * img_size, cy * img_size   # letterboxed-space centre
    w_lb,  h_lb  = w  * img_size, h  * img_size   # letterboxed-space size
    x1_lb = cx_lb - w_lb / 2 ;  x2_lb = cx_lb + w_lb / 2
    y1_lb = cy_lb - h_lb / 2 ;  y2_lb = cy_lb + h_lb / 2
    # Remove letterbox padding and scale back to native resolution:
    x1 = (x1_lb - pad_left) / scale
    y1 = (y1_lb - pad_top)  / scale
    x2 = (x2_lb - pad_left) / scale
    y2 = (y2_lb - pad_top)  / scale
    # Clamp to [0, orig_w] / [0, orig_h].

Siamese outputs
---------------
  existence_prob : float32  (1,)  — sigmoid probability ∈ [0, 1] that the
                                    object shown in the support images is
                                    present in the query image. Apply your
                                    own threshold in the back-end (trained
                                    model default ≈ 0.5; see
                                    ``checkpoints/siamese/S2/stage_complete.pt``
                                    key ``"learned_threshold"``).

CLI usage
---------
    # Export both models to ONNX (default — no TF/TFLite required):
    python export.py \\
        --localizer-ckpt checkpoints/localizer/L3/stage_complete.pt \\
        --siamese-ckpt   checkpoints/siamese/S2/stage_complete.pt

    # Opt in to TFLite (also produces ONNX as an intermediate):
    python export.py \\
        --localizer-ckpt checkpoints/localizer/L3/stage_complete.pt \\
        --siamese-ckpt   checkpoints/siamese/S2/stage_complete.pt \\
        --format tflite

    # Export only one model:
    python export.py --localizer-ckpt checkpoints/localizer/L3/stage_complete.pt
    python export.py --siamese-ckpt   checkpoints/siamese/S2/stage_complete.pt

    # Custom output directory:
    python export.py --localizer-ckpt ... --siamese-ckpt ... --out-dir exports/

Public API
----------
    export_localizer(checkpoint, out_dir="exports", format="onnx", device=None)
    export_siamese(checkpoint,   out_dir="exports", format="onnx", device=None)
    export_both(localizer_ckpt, siamese_ckpt, out_dir="exports", format="onnx",
                device=None)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from localizer.model import MultiShotLocalizer, _normalize_owlv2  # noqa: F401
from siamese.model import MultiShotSiamese
from shared.checkpoint import load_trainable_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_localizer(ckpt_path: str | Path, *, device: torch.device) -> tuple[MultiShotLocalizer, dict, int, int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    k_max = int(cfg.get("k_max", 10))
    img_size = int(cfg.get("img_size", 768))
    lora_active = bool(ckpt.get("lora_active", ckpt.get("stage", "") == "L3"))
    model = MultiShotLocalizer(
        model_name=cfg.get("owlv2_model_name", "google/owlv2-base-patch16-ensemble"),
        k_max=k_max,
        fusion_layers=int(cfg.get("fusion_layers", 2)),
        fusion_heads=int(cfg.get("fusion_heads", 8)),
        fusion_mlp_ratio=int(cfg.get("fusion_mlp_ratio", 2)),
        fusion_dropout=float(cfg.get("fusion_dropout", 0.1)),
    )
    if lora_active:
        model.attach_lora(
            r=int(cfg.get("lora_r", 8)),
            alpha=int(cfg.get("lora_alpha", 16)),
            dropout=float(cfg.get("lora_dropout", 0.1)),
            last_n_layers=int(cfg.get("lora_last_n_layers", 4)),
        )
    load_trainable_state(model, ckpt.get("state_dict", {}))
    model = model.to(device).eval()
    return model, cfg, k_max, img_size


def _load_siamese(ckpt_path: str | Path, *, device: torch.device) -> tuple[MultiShotSiamese, dict, int, int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    k_max = int(cfg.get("k_max", 10))
    img_size = int(cfg.get("img_size", 518))
    lora_active = bool(ckpt.get("lora_active", ckpt.get("stage", "") == "S2"))
    model = MultiShotSiamese(
        model_name=cfg.get("dinov2_model_name", "facebook/dinov2-small"),
        k_max=k_max,
        cross_attn_heads=int(cfg.get("cross_attn_heads", 6)),
        cross_attn_dropout=float(cfg.get("cross_attn_dropout", 0.1)),
        head_hidden_1=int(cfg.get("head_hidden_1", 256)),
        head_hidden_2=int(cfg.get("head_hidden_2", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.2)),
    )
    if lora_active:
        model.attach_lora(
            r=int(cfg.get("lora_r", 8)),
            alpha=int(cfg.get("lora_alpha", 16)),
            dropout=float(cfg.get("lora_dropout", 0.1)),
            last_n_layers=int(cfg.get("lora_last_n_layers", 4)),
        )
    load_trainable_state(model, ckpt.get("state_dict", {}))
    model = model.to(device).eval()
    learned_threshold = float(ckpt.get("learned_threshold", 0.5))
    return model, cfg, k_max, img_size, learned_threshold


# ---------------------------------------------------------------------------
# Export wrappers — thin nn.Module shims that take the 3 canonical inputs
# and return only the tensors the back-end actually needs.
# support_mask is accepted as float32 (TFLite has no bool input) and cast
# to bool internally.
# ---------------------------------------------------------------------------

class _LocalizerWrapper(nn.Module):
    """Wraps MultiShotLocalizer for export.

    Inputs  (all float32, B=1 at export time):
        support_imgs : (1, k_max, 3, img_size, img_size)  pixels in [0, 1]
        support_mask : (1, k_max)                          1.0=real, 0.0=pad
        query_img    : (1, 3, img_size, img_size)          pixels in [0, 1]

    Outputs (all float32):
        best_box   : (1, 4)  cx cy w h  normalised to [0, 1] in letterbox space
        best_score : (1,)    foreground softmax probability
        bg_prob    : (1,)    background/abstain softmax probability
    """

    def __init__(self, model: MultiShotLocalizer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_bool = support_mask.bool()
        out = self.model(support_imgs, mask_bool, query_img)
        return out["best_box"], out["best_score"], out["bg_prob"]


class _SiameseWrapper(nn.Module):
    """Wraps MultiShotSiamese for export.

    Inputs  (all float32, B=1 at export time):
        support_imgs : (1, k_max, 3, img_size, img_size)  pixels in [0, 1]
        support_mask : (1, k_max)                          1.0=real, 0.0=pad
        query_img    : (1, 3, img_size, img_size)          pixels in [0, 1]

    Outputs (all float32):
        existence_prob : (1,)  sigmoid probability in [0, 1]
    """

    def __init__(self, model: MultiShotSiamese) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        mask_bool = support_mask.bool()
        out = self.model(support_imgs, mask_bool, query_img)
        return (out["existence_prob"],)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _disable_transformer_fastpath(module: nn.Module) -> None:
    """Disable the fused TransformerEncoderLayer fast path.

    The fused kernel ``aten::_transformer_encoder_layer_fwd`` has no ONNX
    symbolic, so tracing fails. We bypass it by overriding ``forward`` on
    each ``TransformerEncoderLayer`` to skip the fast-path branch and call
    the plain ``_sa_block`` / ``_ff_block`` path directly.
    """
    import torch.nn.functional as F

    def _plain_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    for m in module.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            m.use_nested_tensor = False
        if isinstance(m, nn.TransformerEncoderLayer):
            m.forward = _plain_forward.__get__(m, type(m))


def _export_onnx(
    wrapper: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    import torch.onnx
    _disable_transformer_fastpath(wrapper)
    print(f"  Exporting ONNX → {onnx_path} …")
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ONNX written: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# TFLite export (ONNX → TF SavedModel → TFLite)
# ---------------------------------------------------------------------------

def _onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> None:
    """Convert ONNX → TFLite via TF SavedModel.

    Tries the optimized pass first (``onnxsim`` on, ``Optimize.DEFAULT`` on).
    If that fails — typically due to OOM on low-memory machines — falls back
    to a fast pass (``onnxsim`` off, no TFLite optimization), which is
    larger but much cheaper to produce.

    The final artefact is always written to ``tflite_path``.
    """
    try:
        import onnx  # noqa: F401
        import onnx2tf  # noqa: F401
        import tensorflow as tf  # noqa: F401
        import tf_keras  # noqa: F401
        import onnx_graphsurgeon  # noqa: F401
        import sng4onnx  # noqa: F401
        import onnxsim  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TFLite export requires: pip install onnx onnx2tf tensorflow "
            "tf-keras onnx-graphsurgeon sng4onnx onnxsim ai-edge-litert "
            "onnxruntime\n"
            f"Missing: {exc}"
        ) from exc

    import onnx2tf
    import tensorflow as tf

    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    def _run(*, use_onnxsim: bool, optimize: bool, label: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved_model_dir = Path(tmp) / "saved_model"
            print(f"  [{label}] Converting ONNX → TF SavedModel via onnx2tf "
                  f"(onnxsim={'on' if use_onnxsim else 'off'}) …")
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
                not_use_onnxsim=not use_onnxsim,
                verbosity="error",
            )
            print(f"  [{label}] Converting TF SavedModel → TFLite "
                  f"(optimize={'on' if optimize else 'off'}) …")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            if optimize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]
            tflite_model = converter.convert()
            tflite_path.write_bytes(tflite_model)
            print(f"  [{label}] TFLite written: {tflite_path} "
                  f"({tflite_path.stat().st_size / 1e6:.1f} MB)")

    # Pass 1: optimized (preferred).
    try:
        _run(use_onnxsim=True, optimize=True, label="optimized")
        return
    except Exception as exc:  # noqa: BLE001 - want to catch SystemExit-ish too
        print(f"  [optimized] FAILED ({type(exc).__name__}: {exc}).")
        print(f"  [optimized] Falling back to fast pass …")

    # Pass 2: fast, memory-friendly fallback.
    _run(use_onnxsim=False, optimize=False, label="fast")


# ---------------------------------------------------------------------------
# Per-model export functions
# ---------------------------------------------------------------------------

def export_localizer(
    checkpoint: str | Path,
    out_dir: str | Path = "exports",
    format: Literal["onnx", "tflite"] = "onnx",
    device: str | None = None,
) -> dict[str, Path]:
    """Export the localizer checkpoint to ONNX (default) or TFLite.

    Parameters
    ----------
    checkpoint:
        Path to a localizer ``.pt`` checkpoint (e.g. ``L3/stage_complete.pt``).
    out_dir:
        Output directory. Produced files are written here as:
        ``localizer.onnx`` and/or ``localizer.tflite``.
    format:
        ``"onnx"``   (default) — produces ONNX only.
        ``"tflite"``           — produces both ONNX and TFLite.
    device:
        PyTorch device string. Defaults to ``"cuda"`` if available else ``"cpu"``.
        Export always runs on CPU so the ONNX graph has no CUDA ops.

    Returns
    -------
    dict with keys ``"onnx"`` and (when format=="tflite") ``"tflite"``,
    each mapping to the produced file ``Path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device("cpu")

    print(f"[export] Loading localizer checkpoint: {checkpoint}")
    model, cfg, k_max, img_size = _load_localizer(checkpoint, device=device_t)
    wrapper = _LocalizerWrapper(model).eval()

    support_imgs = torch.zeros(1, k_max, 3, img_size, img_size, dtype=torch.float32)
    support_mask = torch.ones(1, k_max, dtype=torch.float32)
    query_img = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    example_inputs = (support_imgs, support_mask, query_img)

    onnx_path = out_dir / "localizer.onnx"
    with torch.no_grad():
        _export_onnx(
            wrapper, example_inputs, onnx_path,
            input_names=["support_imgs", "support_mask", "query_img"],
            output_names=["best_box", "best_score", "bg_prob"],
        )

    _write_localizer_meta(out_dir, cfg, k_max, img_size)

    result: dict[str, Path] = {"onnx": onnx_path}
    if format == "tflite":
        tflite_path = out_dir / "localizer.tflite"
        _onnx_to_tflite(onnx_path, tflite_path)
        result["tflite"] = tflite_path

    print(f"[export] Localizer export complete → {out_dir}")
    return result


def export_siamese(
    checkpoint: str | Path,
    out_dir: str | Path = "exports",
    format: Literal["onnx", "tflite"] = "onnx",
    device: str | None = None,
) -> dict[str, Path]:
    """Export the siamese checkpoint to ONNX (default) or TFLite.

    Parameters
    ----------
    checkpoint:
        Path to a siamese ``.pt`` checkpoint (e.g. ``S2/stage_complete.pt``).
    out_dir:
        Output directory. Produced files:
        ``siamese.onnx`` and/or ``siamese.tflite``.
    format:
        ``"onnx"`` (default) or ``"tflite"``.
    device:
        Export always runs on CPU regardless of this argument.

    Returns
    -------
    dict with keys ``"onnx"`` and (when format=="tflite") ``"tflite"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device("cpu")

    print(f"[export] Loading siamese checkpoint: {checkpoint}")
    model, cfg, k_max, img_size, learned_threshold = _load_siamese(checkpoint, device=device_t)
    wrapper = _SiameseWrapper(model).eval()

    support_imgs = torch.zeros(1, k_max, 3, img_size, img_size, dtype=torch.float32)
    support_mask = torch.ones(1, k_max, dtype=torch.float32)
    query_img = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    example_inputs = (support_imgs, support_mask, query_img)

    onnx_path = out_dir / "siamese.onnx"
    with torch.no_grad():
        _export_onnx(
            wrapper, example_inputs, onnx_path,
            input_names=["support_imgs", "support_mask", "query_img"],
            output_names=["existence_prob"],
        )

    _write_siamese_meta(out_dir, cfg, k_max, img_size, learned_threshold)

    result: dict[str, Path] = {"onnx": onnx_path}
    if format == "tflite":
        tflite_path = out_dir / "siamese.tflite"
        _onnx_to_tflite(onnx_path, tflite_path)
        result["tflite"] = tflite_path

    print(f"[export] Siamese export complete → {out_dir}")
    return result


def export_both(
    localizer_ckpt: str | Path | None = None,
    siamese_ckpt: str | Path | None = None,
    out_dir: str | Path = "exports",
    format: Literal["onnx", "tflite"] = "onnx",
    device: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Export both models.  Either checkpoint may be omitted to skip that model.

    Returns
    -------
    dict with keys ``"localizer"`` and/or ``"siamese"``, each containing the
    per-model result dict from ``export_localizer`` / ``export_siamese``.
    """
    results: dict[str, dict[str, Path]] = {}
    if localizer_ckpt is not None:
        results["localizer"] = export_localizer(localizer_ckpt, out_dir=out_dir, format=format, device=device)
    if siamese_ckpt is not None:
        results["siamese"] = export_siamese(siamese_ckpt, out_dir=out_dir, format=format, device=device)
    if not results:
        raise ValueError("At least one of localizer_ckpt or siamese_ckpt must be provided.")
    return results


# ---------------------------------------------------------------------------
# Metadata sidecars — machine-readable JSON describing the exported model's
# I/O contract so back-ends can read shapes, pre-processing params, and
# output semantics without parsing this source file.
# ---------------------------------------------------------------------------

def _write_localizer_meta(out_dir: Path, cfg: dict, k_max: int, img_size: int) -> None:
    meta = {
        "model": "localizer",
        "inputs": {
            "support_imgs": {
                "shape": [1, k_max, 3, img_size, img_size],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": (
                    "K support images letterboxed to img_size x img_size, "
                    "pixel values in [0,1], channels-first RGB. "
                    "Pad unused slots with zeros and set support_mask to 0."
                ),
            },
            "support_mask": {
                "shape": [1, k_max],
                "dtype": "float32",
                "values": "1.0 = real support slot, 0.0 = padding",
            },
            "query_img": {
                "shape": [1, 3, img_size, img_size],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": "Query image letterboxed to img_size x img_size, pixel values in [0,1], channels-first RGB.",
            },
        },
        "outputs": {
            "best_box": {
                "shape": [1, 4],
                "dtype": "float32",
                "format": "cxcywh_normalised",
                "description": (
                    "Top-1 predicted bounding box as (cx, cy, w, h) "
                    "normalised to [0,1] relative to the letterboxed img_size square. "
                    "Convert to native-pixel xyxy: "
                    "cx_lb=cx*img_size, cy_lb=cy*img_size, w_lb=w*img_size, h_lb=h*img_size; "
                    "x1=(cx_lb - w_lb/2 - pad_left)/scale, etc."
                ),
            },
            "best_score": {
                "shape": [1],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": "Foreground softmax probability of the top-1 detection.",
            },
            "bg_prob": {
                "shape": [1],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": (
                    "Background / abstain softmax probability. "
                    "Suppress the detection when bg_prob >= abstain_threshold (e.g. 0.5)."
                ),
            },
        },
        "preprocessing": {
            "letterbox": {
                "target_size": img_size,
                "pad_color_rgb": [114, 114, 114],
                "pad_color_float": round(114 / 255, 4),
                "description": (
                    "Resize preserving aspect ratio so the longest side == img_size, "
                    "then pad both axes with grey (114/255) to reach img_size x img_size. "
                    "Record scale = img_size / max(orig_w, orig_h), "
                    "pad_left = (img_size - new_w) // 2, "
                    "pad_top  = (img_size - new_h) // 2 "
                    "for the inverse transform above."
                ),
            },
            "normalisation": "baked into model graph — raw [0,1] inputs only",
            "channel_order": "RGB",
            "layout": "CHW (channels-first)",
        },
        "k_max": k_max,
        "img_size": img_size,
        "model_config": cfg,
    }
    path = out_dir / "localizer_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata written: {path}")


def _write_siamese_meta(
    out_dir: Path, cfg: dict, k_max: int, img_size: int, learned_threshold: float
) -> None:
    meta = {
        "model": "siamese",
        "inputs": {
            "support_imgs": {
                "shape": [1, k_max, 3, img_size, img_size],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": (
                    "K support images letterboxed to img_size x img_size, "
                    "pixel values in [0,1], channels-first RGB. "
                    "Pad unused slots with zeros and set support_mask to 0."
                ),
            },
            "support_mask": {
                "shape": [1, k_max],
                "dtype": "float32",
                "values": "1.0 = real support slot, 0.0 = padding",
            },
            "query_img": {
                "shape": [1, 3, img_size, img_size],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": "Query image letterboxed to img_size x img_size, pixel values in [0,1], channels-first RGB.",
            },
        },
        "outputs": {
            "existence_prob": {
                "shape": [1],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "description": (
                    "Sigmoid probability that the object depicted in the support "
                    "images is present in the query image. "
                    "Apply your own threshold: exists = (existence_prob >= threshold). "
                    f"Recommended threshold from training: {learned_threshold:.4f}."
                ),
            },
        },
        "preprocessing": {
            "letterbox": {
                "target_size": img_size,
                "pad_color_rgb": [114, 114, 114],
                "pad_color_float": round(114 / 255, 4),
                "description": (
                    "Resize preserving aspect ratio so the longest side == img_size, "
                    "then pad both axes with grey (114/255) to reach img_size x img_size."
                ),
            },
            "normalisation": "baked into model graph — raw [0,1] inputs only",
            "channel_order": "RGB",
            "layout": "CHW (channels-first)",
        },
        "threshold": {
            "learned_threshold": learned_threshold,
            "description": (
                "Threshold calibrated on the validation set during training. "
                "exists = (existence_prob >= learned_threshold). "
                "You may adjust this in your back-end to trade off FPR vs FNR."
            ),
        },
        "k_max": k_max,
        "img_size": img_size,
        "model_config": cfg,
    }
    path = out_dir / "siamese_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export localizer and/or siamese checkpoints to TFLite or ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--localizer-ckpt", default=None, metavar="PATH",
        help="Path to localizer checkpoint (e.g. checkpoints/localizer/L3/stage_complete.pt).",
    )
    p.add_argument(
        "--siamese-ckpt", default=None, metavar="PATH",
        help="Path to siamese checkpoint (e.g. checkpoints/siamese/S2/stage_complete.pt).",
    )
    p.add_argument(
        "--out-dir", default="exports", metavar="DIR",
        help="Output directory (default: exports/).",
    )
    p.add_argument(
        "--format", default="onnx", choices=["onnx", "tflite"],
        help="Export format: 'onnx' (default) produces ONNX only; 'tflite' produces ONNX + TFLite.",
    )
    p.add_argument(
        "--device", default=None,
        help="PyTorch device (default: auto). Export always uses CPU for the ONNX graph.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.localizer_ckpt is None and args.siamese_ckpt is None:
        parser.error("At least one of --localizer-ckpt or --siamese-ckpt must be given.")
    export_both(
        localizer_ckpt=args.localizer_ckpt,
        siamese_ckpt=args.siamese_ckpt,
        out_dir=args.out_dir,
        format=args.format,
        device=args.device,
    )


if __name__ == "__main__":
    main()
