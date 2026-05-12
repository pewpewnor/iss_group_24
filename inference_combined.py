"""Combined cascaded inference: siamese → localizer.

Modes:
    "hard":             run siamese; if existence_prob < threshold, return
                        {exists: False, bbox: None}; else run localizer.
    "soft":             always run both, return both fields; the bbox is
                        flagged as low-confidence below threshold but kept.
    "always_localize":  ignore siamese for the bbox decision; always return
                        localizer's bbox plus the siamese's existence_prob.

Public API:
    run_combined(siamese_ckpt, localizer_ckpt, support_paths, query_path,
                 *, existence_threshold=0.5, existence_threshold_mode="hard",
                 ...)
        -> dict {existence_prob, exists, bbox|None, localizer_score}

    sweep_threshold(siamese_ckpt, localizer_ckpt, *, eval_split="test",
                    thresholds=(...), **eval_kwargs)
        -> dict per threshold of {fpr, fnr, accuracy, map_50_when_exists}
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from inference_localizer import _build_model_from_ckpt as _build_localizer
from inference_siamese import _build_model_from_ckpt as _build_siamese
from shared.dataset import _letterbox
from shared.analytics import write_json
from shared.runtime import gpu_cleanup_on_exit


def _next_run_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_root / f"{n:04d}").exists():
        n += 1
    p = out_root / f"{n:04d}"
    p.mkdir()
    return p


def _draw_bbox(img: Image.Image, bbox_xyxy, *,
               color=(0, 255, 0), thickness=4, caption=None):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox_xyxy
    for off in range(thickness):
        draw.rectangle([x1 + off, y1 + off, x2 - off, y2 - off], outline=color)
    if caption:
        draw.rectangle([x1, max(0, y1 - 28), x1 + 8 * len(caption), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 24)), caption, fill=(0, 0, 0))
    return out


def _prep_inputs(
    support_paths: list[str | Path], query_path: str | Path,
    *, img_size: int, k_max: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image, dict]:
    sup_tensors = []
    for sp in support_paths:
        lb, _, _, _ = _letterbox(Image.open(sp).convert("RGB"), img_size)
        sup_tensors.append(TF.to_tensor(lb))
    K = len(sup_tensors)
    while len(sup_tensors) < k_max:
        sup_tensors.append(torch.zeros(3, img_size, img_size))
    sup_t = torch.stack(sup_tensors, dim=0).unsqueeze(0).to(device)
    mask = torch.zeros(1, k_max, dtype=torch.bool, device=device)
    mask[0, :K] = True

    qry_pil = Image.open(query_path).convert("RGB")
    nw, nh = qry_pil.size
    qry_lb, q_scale, q_pl, q_pt = _letterbox(qry_pil, img_size)
    qry_t = TF.to_tensor(qry_lb).unsqueeze(0).to(device)
    return sup_t, mask, qry_t, qry_pil, {"nw": nw, "nh": nh, "q_scale": q_scale,
                                          "q_pl": q_pl, "q_pt": q_pt, "K": K}


def _bbox_to_native(cx, cy, w, h, *, img_size, q_scale, q_pl, q_pt, nw, nh):
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return [
        max(0.0, min(nw, (x1 - q_pl) / q_scale)),
        max(0.0, min(nh, (y1 - q_pt) / q_scale)),
        max(0.0, min(nw, (x2 - q_pl) / q_scale)),
        max(0.0, min(nh, (y2 - q_pt) / q_scale)),
    ]


def run_combined(
    siamese_ckpt: str | Path,
    localizer_ckpt: str | Path,
    support_paths: list[str | Path],
    query_path: str | Path,
    *,
    existence_threshold: float | None = None,
    existence_threshold_mode: str = "hard",   # "hard" | "soft" | "always_localize"
    siamese_img_size: int = 518,
    localizer_img_size: int = 768,
    abstain_threshold: float = 0.5,
    device: str | None = None,
    out_root: str | Path = "inference/combined",
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 4,
    smoke: bool = False,
) -> dict[str, Any]:
    """Cascaded siamese → localizer inference.

    Threshold defaulting:
      - ``existence_threshold=None`` (default): read ``learned_threshold`` from
        the siamese checkpoint (median val best_f1_threshold across training).
        This is the calibrated operating point that fixes the previous
        "tp=0 because 0.5 was never crossed" bug.
      - Pass a float to override.

    The localizer's own abstain channel (``bg_prob``) is ALSO surfaced and an
    abstain decision is reported per query.
    """
    with gpu_cleanup_on_exit(verbose=False), torch.no_grad():
        return _run_combined_inner(
            siamese_ckpt=siamese_ckpt, localizer_ckpt=localizer_ckpt,
            support_paths=support_paths, query_path=query_path,
            existence_threshold=existence_threshold,
            existence_threshold_mode=existence_threshold_mode,
            siamese_img_size=siamese_img_size,
            localizer_img_size=localizer_img_size,
            abstain_threshold=abstain_threshold,
            device=device, out_root=out_root,
            bbox_color=bbox_color, bbox_thickness=bbox_thickness,
            smoke=smoke,
        )


def _run_combined_inner(
    *,
    siamese_ckpt, localizer_ckpt, support_paths, query_path,
    existence_threshold, existence_threshold_mode,
    siamese_img_size, localizer_img_size,
    abstain_threshold,
    device, out_root, bbox_color, bbox_thickness, smoke,
) -> dict[str, Any]:
    if smoke:
        siamese_img_size = 224
        localizer_img_size = 224
    if not Path(siamese_ckpt).exists():
        raise FileNotFoundError(f"siamese ckpt not found: {siamese_ckpt}")
    if not Path(localizer_ckpt).exists():
        raise FileNotFoundError(f"localizer ckpt not found: {localizer_ckpt}")
    if not support_paths:
        raise ValueError("support_paths must contain at least 1 image path")
    if not Path(query_path).exists():
        raise FileNotFoundError(f"query not found: {query_path}")
    for sp in support_paths:
        if not Path(sp).exists():
            raise FileNotFoundError(f"support not found: {sp}")
    if existence_threshold_mode not in ("hard", "soft", "always_localize"):
        raise ValueError(
            f"existence_threshold_mode must be one of 'hard'|'soft'|'always_localize', "
            f"got {existence_threshold_mode!r}"
        )
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    sia_ckpt = torch.load(str(siamese_ckpt), map_location="cpu", weights_only=False)
    loc_ckpt = torch.load(str(localizer_ckpt), map_location="cpu", weights_only=False)
    # Default existence_threshold to the val-discovered ``learned_threshold``
    # stored in the siamese checkpoint (median best_f1_threshold across
    # epochs). Falling back to 0.5 only if the ckpt did not record one.
    if existence_threshold is None:
        learned = sia_ckpt.get("learned_threshold")
        existence_threshold = float(learned) if learned is not None else 0.5
        thr_source = "ckpt.learned_threshold" if learned is not None else "default(0.5)"
    else:
        existence_threshold = float(existence_threshold)
        thr_source = "explicit"
    if not (0.0 <= existence_threshold <= 1.0):
        raise ValueError(f"existence_threshold must be in [0, 1], got {existence_threshold}")
    sia_k = int(sia_ckpt.get("config", {}).get("k_max", 10))
    loc_k = int(loc_ckpt.get("config", {}).get("k_max", 10))
    if smoke:
        sia_k = max(2, len(support_paths))
        loc_k = max(2, len(support_paths))
    K = len(support_paths)
    if K > sia_k or K > loc_k:
        raise ValueError(f"Too many supports ({K}); siamese k_max={sia_k}, localizer k_max={loc_k}")

    siamese = _build_siamese(sia_ckpt, k_max=sia_k).to(device_t).eval()
    localizer = _build_localizer(loc_ckpt, k_max=loc_k, img_size=localizer_img_size).to(device_t).eval()

    # Siamese forward.
    sia_sup, sia_mask, sia_qry, qry_pil, _ = _prep_inputs(
        support_paths, query_path, img_size=siamese_img_size,
        k_max=sia_k, device=device_t,
    )
    sia_out = siamese(sia_sup, sia_mask, sia_qry)
    existence_prob = float(sia_out["existence_prob"][0].cpu().item())
    exists = bool(existence_prob >= existence_threshold)

    bbox_native: list[float] | None = None
    loc_score: float | None = None
    bg_prob: float | None = None
    loc_abstain: bool | None = None
    skip_localizer = (existence_threshold_mode == "hard" and not exists)

    if not skip_localizer:
        loc_sup, loc_mask, loc_qry, _, geom = _prep_inputs(
            support_paths, query_path, img_size=localizer_img_size,
            k_max=loc_k, device=device_t,
        )
        loc_out = localizer(loc_sup, loc_mask, loc_qry)
        cx, cy, w, h = loc_out["best_box"][0].cpu().tolist()
        loc_score = float(loc_out["best_score"][0].cpu().item())
        bg_prob = float(loc_out.get("bg_prob", torch.zeros(1))[0].cpu().item()) \
            if "bg_prob" in loc_out else None
        loc_abstain = (bg_prob is not None and bg_prob >= abstain_threshold)
        bbox_native = _bbox_to_native(
            cx, cy, w, h,
            img_size=localizer_img_size,
            q_scale=geom["q_scale"], q_pl=geom["q_pl"], q_pt=geom["q_pt"],
            nw=geom["nw"], nh=geom["nh"],
        )

    out_dir = _next_run_dir(Path(out_root))
    for idx, sp in enumerate(support_paths, start=1):
        shutil.copy2(str(sp), str(out_dir / f"support_{idx:02d}{Path(sp).suffix}"))
    shutil.copy2(str(query_path), str(out_dir / f"query{Path(query_path).suffix}"))
    if bbox_native is not None:
        annotated = _draw_bbox(
            qry_pil, bbox_native,
            color=bbox_color, thickness=bbox_thickness,
            caption=("EXISTS" if exists else "low-conf") + f"  p={existence_prob:.2f}",
        )
        annotated.save(str(out_dir / "result.png"))
    else:
        # Save query annotated with NOT EXISTS caption.
        annotated = _draw_bbox(
            qry_pil, (5, 5, qry_pil.size[0] - 5, qry_pil.size[1] - 5),
            color=(255, 0, 0), thickness=2,
            caption=f"NOT EXISTS  p={existence_prob:.2f}",
        )
        annotated.save(str(out_dir / "result.png"))

    payload = {
        "siamese_ckpt": str(siamese_ckpt),
        "localizer_ckpt": str(localizer_ckpt),
        "n_support": K,
        "siamese_img_size": siamese_img_size,
        "localizer_img_size": localizer_img_size,
        "existence_prob": existence_prob,
        "existence_threshold": existence_threshold,
        "existence_threshold_source": thr_source,
        "existence_threshold_mode": existence_threshold_mode,
        "exists": exists,
        "bbox_xyxy_native": bbox_native,
        "localizer_score": loc_score,
        "localizer_bg_prob": bg_prob,
        "localizer_abstain": loc_abstain,
        "abstain_threshold": float(abstain_threshold),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[combined] existence_prob={existence_prob:.4f}  exists={exists}  "
          f"(thr={existence_threshold:.3f} from {thr_source})  "
          f"bg_prob={bg_prob if bg_prob is None else f'{bg_prob:.3f}'}  "
          f"→  {out_dir}")
    return payload


# ---------------------------------------------------------------------------
# Threshold sweep against the test split
# ---------------------------------------------------------------------------


def sweep_threshold(
    siamese_ckpt: str | Path,
    localizer_ckpt: str | Path,
    *,
    manifest: str = "dataset/aggregated/manifest.json",
    data_root: str | None = None,
    test_episodes: int = 400,
    thresholds: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    siamese_img_size: int = 518,
    localizer_img_size: int = 768,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    k_min: int = 1,
    k_max: int = 10,
    device: str | None = None,
    analysis_root: str = "analysis",
) -> dict[str, Any]:
    """Run BOTH models on the test split, then sweep cascade thresholds.

    Computes per-threshold:
        FPR, FNR, accuracy   (siamese-driven binary)
        map_50_at_threshold  (mAP@50 of localizer over episodes the siamese
                              passed through, gated by existence score)
    """
    with gpu_cleanup_on_exit(verbose=False), torch.no_grad():
        return _sweep_threshold_inner(
            siamese_ckpt=siamese_ckpt, localizer_ckpt=localizer_ckpt,
            manifest=manifest, data_root=data_root,
            test_episodes=test_episodes, thresholds=thresholds,
            siamese_img_size=siamese_img_size,
            localizer_img_size=localizer_img_size,
            batch_size=batch_size, num_workers=num_workers, seed=seed,
            k_min=k_min, k_max=k_max, device=device,
            analysis_root=analysis_root,
        )


def _sweep_threshold_inner(
    *,
    siamese_ckpt, localizer_ckpt,
    manifest, data_root, test_episodes, thresholds,
    siamese_img_size, localizer_img_size,
    batch_size, num_workers, seed, k_min, k_max, device, analysis_root,
) -> dict[str, Any]:
    from siamese.dataset import build_val_loader as build_sia_loader
    from localizer.dataset import build_val_loader as build_loc_loader

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sia_ckpt = torch.load(str(siamese_ckpt), map_location="cpu", weights_only=False)
    loc_ckpt = torch.load(str(localizer_ckpt), map_location="cpu", weights_only=False)
    siamese = _build_siamese(sia_ckpt, k_max=k_max).to(device_t).eval()
    localizer = _build_localizer(loc_ckpt, k_max=k_max, img_size=localizer_img_size).to(device_t).eval()

    sia_ds, sia_loader = build_sia_loader(
        manifest=manifest, data_root=data_root, split="test", sources=None,
        val_episodes=test_episodes, batch_size=batch_size, num_workers=num_workers,
        neg_prob=0.5, img_size=siamese_img_size, seed=seed,
        k_min=k_min, k_max=k_max,
    )
    loc_ds, loc_loader = build_loc_loader(
        manifest=manifest, data_root=data_root, split="test", sources=None,
        val_episodes=test_episodes, batch_size=batch_size, num_workers=num_workers,
        img_size=localizer_img_size, seed=seed,
        k_min=k_min, k_max=k_max,
    )

    # Collect per-episode predictions.
    sia_records: list[dict] = []
    for batch in sia_loader:
        sup = batch["support_imgs"].to(device_t)
        mask = batch["support_mask"].to(device_t)
        qry = batch["query_img"].to(device_t)
        out = siamese(sup, mask, qry)
        for i in range(sup.size(0)):
            sia_records.append({
                "iid": batch["instance_id"][i],
                "is_present": bool(batch["is_present"][i].item()),
                "existence_prob": float(out["existence_prob"][i].cpu().item()),
            })

    # Localizer: only compute IoU for positive episodes (positive-only training).
    from localizer.evaluate import _iou_xyxy
    from localizer.loss import _cxcywh_to_xyxy
    loc_ious: list[float] = []
    loc_scores: list[float] = []
    loc_present: list[bool] = []
    for batch in loc_loader:
        sup = batch["support_imgs"].to(device_t)
        mask = batch["support_mask"].to(device_t)
        qry = batch["query_img"].to(device_t)
        gt = batch["query_bbox"].to(device_t)
        is_present = batch["is_present"].to(device_t)
        out = localizer(sup, mask, qry)
        iou = _iou_xyxy(_cxcywh_to_xyxy(out["best_box"]), _cxcywh_to_xyxy(gt))
        for i in range(sup.size(0)):
            loc_ious.append(float(iou[i].cpu().item()))
            loc_scores.append(float(out["best_score"][i].cpu().item()))
            loc_present.append(bool(is_present[i].item()))

    # Sweep.
    results: dict[str, dict[str, float]] = {}
    n = len(sia_records)
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for r in sia_records:
            pred = r["existence_prob"] >= thr
            if r["is_present"] and pred:
                tp += 1
            elif r["is_present"] and not pred:
                fn += 1
            elif (not r["is_present"]) and pred:
                fp += 1
            else:
                tn += 1
        n_pos = tp + fn
        n_neg = fp + tn
        fpr = fp / max(n_neg, 1)
        fnr = fn / max(n_pos, 1)
        acc = (tp + tn) / max(n, 1)
        # map@50 over positives that passed siamese gating.
        kept_iou: list[tuple[float, bool]] = []
        for r, iou, sc in zip(sia_records, loc_ious, loc_scores):
            if r["is_present"] and r["existence_prob"] >= thr:
                kept_iou.append((sc, iou >= 0.5))
        kept_n = max(n_pos, 1)
        kept_iou.sort(key=lambda x: -x[0])
        tpc = fpc = 0
        precs: list[float] = []
        recs: list[float] = []
        for sc, is_tp in kept_iou:
            if is_tp:
                tpc += 1
            else:
                fpc += 1
            precs.append(tpc / (tpc + fpc))
            recs.append(tpc / kept_n)
        for k in range(len(precs) - 2, -1, -1):
            if precs[k] < precs[k + 1]:
                precs[k] = precs[k + 1]
        rec_thr = [i / 100.0 for i in range(101)]
        ap = 0.0
        j = 0
        for rt in rec_thr:
            while j < len(recs) and recs[j] < rt:
                j += 1
            ap += precs[j] if j < len(precs) else 0.0
        ap /= 101.0
        results[f"thr_{thr:.2f}"] = {
            "threshold": thr,
            "fpr": fpr, "fnr": fnr, "accuracy": acc,
            "map_50_when_exists": ap, "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }

    payload = {
        "siamese_ckpt": str(siamese_ckpt),
        "localizer_ckpt": str(localizer_ckpt),
        "test_episodes": test_episodes,
        "thresholds": list(thresholds),
        "results": results,
    }
    out_path = Path(analysis_root) / "combined" / f"threshold_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out_path, payload)
    print(f"[sweep] {out_path}")
    print(f"  threshold | FPR    | FNR    | acc    | map@50_pos")
    for thr in thresholds:
        r = results[f"thr_{thr:.2f}"]
        print(f"  {thr:9.2f} | {r['fpr']:.4f} | {r['fnr']:.4f} | {r['accuracy']:.4f} | {r['map_50_when_exists']:.4f}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siamese-ckpt", required=True)
    parser.add_argument("--localizer-ckpt", required=True)
    parser.add_argument("--supports", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mode", default="hard", choices=("hard", "soft", "always_localize"))
    parser.add_argument("--siamese-img-size", type=int, default=518)
    parser.add_argument("--localizer-img-size", type=int, default=768)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-root", default="inference/combined")
    args = parser.parse_args()
    run_combined(
        siamese_ckpt=args.siamese_ckpt,
        localizer_ckpt=args.localizer_ckpt,
        support_paths=args.supports,
        query_path=args.query,
        existence_threshold=args.threshold,
        existence_threshold_mode=args.mode,
        siamese_img_size=args.siamese_img_size,
        localizer_img_size=args.localizer_img_size,
        device=args.device,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
