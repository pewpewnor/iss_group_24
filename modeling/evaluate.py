"""Evaluate a trained checkpoint on a manifest split.

Reports:
- mean IoU on positive episodes (object actually present)
- presence accuracy: P(score > thr) matches is_present
- AP@0.5: positive episode counted correct if IoU >= 0.5 AND score > thr
- per-source breakdown (hots / insdet)

Run:
    python -m modeling.evaluate --checkpoint model/best.pt \
                                --manifest dataset/cleaned/manifest.json \
                                --split test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modeling.dataset import EpisodeDataset, collate
from modeling.loss import giou  # noqa: F401
from modeling.model import FewShotLocalizer, decode


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Plain IoU (not GIoU) for evaluation. Element-wise on matched (..., 4) boxes."""
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-6)


@torch.no_grad()
def evaluate(
    model: FewShotLocalizer,
    loader: DataLoader,
    device: torch.device,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
) -> dict:
    model.eval()
    per_source: dict[str, dict[str, list]] = defaultdict(
        lambda: {"iou": [], "score": [], "is_present": [], "correct_at_iou": []}
    )
    overall: dict[str, list] = {"iou": [], "score": [], "is_present": [], "correct_at_iou": []}

    for batch in loader:
        support_imgs = batch["support_imgs"].to(device)
        support_bboxes = batch["support_bboxes"].to(device)
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)
        ids = batch["instance_id"]

        out = model(support_imgs, support_bboxes, query_img)
        pred_box, pred_score = decode(out["reg"], out["conf"])

        ious = _iou_xyxy(pred_box, gt_bbox)
        for i in range(gt_bbox.shape[0]):
            present = bool(is_present[i].item())
            score = float(pred_score[i].item())
            iou_v = float(ious[i].item()) if present else 0.0
            correct = present and (iou_v >= iou_thr) and (score >= score_thr)
            src = ids[i].split("_", 1)[0]
            for bucket in (overall, per_source[src]):
                bucket["iou"].append(iou_v)
                bucket["score"].append(score)
                bucket["is_present"].append(present)
                bucket["correct_at_iou"].append(correct)

    def summarize(b: dict[str, list]) -> dict:
        n = len(b["is_present"])
        if n == 0:
            return {}
        n_pos = sum(b["is_present"])
        n_neg = n - n_pos
        mean_iou = (
            sum(v for v, p in zip(b["iou"], b["is_present"]) if p) / n_pos
            if n_pos
            else 0.0
        )
        presence_correct = sum(
            ((s >= score_thr) == p) for s, p in zip(b["score"], b["is_present"])
        )
        ap50 = sum(b["correct_at_iou"]) / max(n_pos, 1)
        return {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "mean_iou_pos": round(mean_iou, 4),
            "presence_acc": round(presence_correct / n, 4),
            "ap@iou=0.5": round(ap50, 4),
            "mean_score_pos": round(
                sum(s for s, p in zip(b["score"], b["is_present"]) if p) / max(n_pos, 1),
                4,
            ),
            "mean_score_neg": round(
                sum(s for s, p in zip(b["score"], b["is_present"]) if not p)
                / max(n_neg, 1),
                4,
            ),
        }

    return {
        "overall": summarize(overall),
        "by_source": {k: summarize(v) for k, v in per_source.items()},
        "score_thr": score_thr,
        "iou_thr": iou_thr,
    }


def run(
    checkpoint: str | Path,
    manifest: str | Path = "dataset/cleaned/manifest.json",
    split: str = "test",
    data_root: str | Path | None = None,
    episodes: int = 1000,
    batch_size: int = 8,
    num_workers: int = 4,
    neg_prob: float = 0.3,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
    seed: int = 42,
    device: str | None = None,
    report: str | Path = "model/eval_report.json",
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_t = torch.device(device)
    model = FewShotLocalizer(pretrained=False).to(device_t)
    state = torch.load(str(checkpoint), map_location=device_t, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    ds = EpisodeDataset(
        manifest_path=str(manifest),
        split=split,
        data_root=str(data_root) if data_root else None,
        episodes_per_epoch=episodes,
        train=False,
        seed=seed,
    )
    ds.neg_prob = neg_prob

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device_t.type == "cuda"),
    )

    result = evaluate(model, loader, device_t, score_thr, iou_thr)
    print(json.dumps(result, indent=2))

    out_path = Path(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nreport written to {out_path}")

    from modeling.plot import plot_eval_report
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plot_eval_report(result, analysis_dir)

    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="dataset/cleaned/manifest.json")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--data-root",
        default=None,
        help="directory image paths are relative to (default: manifest parent dir)",
    )
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--neg-prob", type=float, default=0.3)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--report", default="model/eval_report.json")
    args = p.parse_args()

    run(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        split=args.split,
        data_root=args.data_root,
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        neg_prob=args.neg_prob,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        seed=args.seed,
        device=args.device,
        report=args.report,
    )


if __name__ == "__main__":
    main()
