"""Evaluate a trained checkpoint on a manifest split.

Reports:
- mean IoU on positive episodes (object actually present)
- presence accuracy: P(score > thr) matches is_present
- AP@0.5, AP@0.75: PR-curve AP at fixed IoU thresholds
- mAP@[0.5:0.95]: COCO-style mean of AP at IoU 0.50, 0.55, ..., 0.95
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
from modeling.loss import _iou_xyxy
from modeling.model import FewShotLocalizer, decode


IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _compute_pr_ap(scores: list[float], is_tp: list[bool], n_gt: int) -> float:
    """COCO-style 101-point interpolated AP from a ranked list of detections.

    Each episode produces exactly one (score, is_tp) pair. ``n_gt`` is the
    number of positive episodes. Returns 0.0 if there are no positives.
    """
    if n_gt == 0 or not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for i in order:
        if is_tp[i]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)

    # Make precisions monotonically non-increasing as recall increases
    # (interpolate from the right).
    for k in range(len(precisions) - 2, -1, -1):
        if precisions[k] < precisions[k + 1]:
            precisions[k] = precisions[k + 1]

    rec_thresholds = [i / 100.0 for i in range(101)]
    ap_sum = 0.0
    j = 0
    for rt in rec_thresholds:
        while j < len(recalls) and recalls[j] < rt:
            j += 1
        ap_sum += precisions[j] if j < len(precisions) else 0.0
    return ap_sum / 101.0



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
        query_img = batch["query_img"].to(device)
        gt_bbox = batch["query_bbox"].to(device)
        is_present = batch["is_present"].to(device)
        ids = batch["instance_id"]

        out = model(support_imgs, query_img)
        pred_box, pred_score = decode(out["reg"], out["conf"], presence_logit=out.get("presence_logit"))

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
        ap_per_iou: dict[str, float] = {}
        for tau in IOU_THRESHOLDS:
            is_tp = [
                bool(p) and (iou_v >= tau)
                for p, iou_v in zip(b["is_present"], b["iou"])
            ]
            ap_per_iou[f"{tau:.2f}"] = round(
                _compute_pr_ap(b["score"], is_tp, n_pos), 4
            )
        map_5095 = sum(ap_per_iou.values()) / len(ap_per_iou)
        return {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "mean_iou_pos": round(mean_iou, 4),
            "presence_acc": round(presence_correct / n, 4),
            "ap@iou=0.5": ap_per_iou["0.50"],
            "ap@iou=0.75": ap_per_iou["0.75"],
            "map@[0.5:0.95]": round(map_5095, 4),
            "ap_per_iou": ap_per_iou,
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
