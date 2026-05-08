"""Tile-inference helpers for OWLv2 few-shot localizer.

Two modes are supported (both turned on/off purely at evaluation time;
training is unaffected):

  - "pyramid_a": a fixed pyramid of crops.  By default 1 full-image crop
                 plus a 2x2 grid with 25% overlap = 5 crops total.

  - "hybrid_d":  pyramid_a first, then re-tile the highest-scoring tile
                 into a 2x2 sub-grid for a second pass — total ≤ 9 crops.

Common pipeline per query image:
  1. Pre-compute the support prototype ONCE (model.encode_support +
     aggregator).  Reused across every tile.
  2. For each tile, run the query path (encode_query + class_predictor +
     box_predictor + existence_head).
  3. Boxes from each tile are un-projected back to the *native* image
     pixel coordinate frame.
  4. NMS-merge across tiles.  Top-1 detection becomes the final prediction.

All public entry points work in pixel coordinates of the *native* image
(no normalisation), and return:

    {
      "best_box_native_xyxy": (4,) tensor in native pixel space
      "best_score":           scalar tensor in [0,1]
      "existence_prob":       scalar tensor in [0,1]
      "all_boxes_native_xyxy": (N, 4) post-NMS detections
      "all_scores":           (N,) post-NMS scores
    }
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import batched_nms
from torchvision.transforms import functional as TF

from modeling.dataset import OWLV2_MEAN, OWLV2_STD


TileMeta = tuple[int, int, int, int]   # (x0, y0, x1, y1) in native px


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------


def pyramid_tiles(
    native_size: tuple[int, int],
    levels: tuple[int, ...] = (1, 2),
    overlap: float = 0.25,
) -> list[TileMeta]:
    """Generate tile bboxes (in native px) for the requested pyramid levels.

    Each level k yields a k×k grid of square tiles covering the entire
    image with the given fractional overlap between adjacent tiles.
    The tile size at level k is the smaller of native_w/k and native_h/k,
    multiplied by a small slack to give symmetric overlap.

    Tiles are clipped to the image bounds.  Duplicates (when the pyramid
    rounds the same region to the same crop, e.g. very small images) are
    de-duplicated.
    """
    nw, nh = native_size
    tiles: list[TileMeta] = []
    seen: set[TileMeta] = set()
    for k in levels:
        if k <= 0:
            continue
        if k == 1:
            t = (0, 0, nw, nh)
            if t not in seen:
                seen.add(t)
                tiles.append(t)
            continue
        # Tile side length so that with `overlap` between adjacent tiles
        # we cover the entire dimension:
        #   k*tw - (k-1)*overlap*tw = nw
        #   tw = nw / (k - (k-1)*overlap)
        tw = int(round(nw / (k - (k - 1) * overlap)))
        th = int(round(nh / (k - (k - 1) * overlap)))
        stride_w = max(1, int(round(tw * (1 - overlap))))
        stride_h = max(1, int(round(th * (1 - overlap))))
        for ry in range(k):
            for rx in range(k):
                x0 = min(rx * stride_w, max(0, nw - tw))
                y0 = min(ry * stride_h, max(0, nh - th))
                x1 = min(x0 + tw, nw)
                y1 = min(y0 + th, nh)
                t = (x0, y0, x1, y1)
                if t in seen:
                    continue
                seen.add(t)
                tiles.append(t)
    return tiles


def dilate_top_tile(
    tile: TileMeta,
    native_size: tuple[int, int],
    grid: int = 2,
    overlap: float = 0.25,
) -> list[TileMeta]:
    """Re-tile a single bbox into a finer ``grid x grid`` sub-grid.

    Used by hybrid_d to drill into the highest-scoring tile from pyramid_a.
    """
    x0, y0, x1, y1 = tile
    sub_w = x1 - x0
    sub_h = y1 - y0
    if sub_w <= 0 or sub_h <= 0 or grid <= 1:
        return [tile]
    tw = int(round(sub_w / (grid - (grid - 1) * overlap)))
    th = int(round(sub_h / (grid - (grid - 1) * overlap)))
    stride_w = max(1, int(round(tw * (1 - overlap))))
    stride_h = max(1, int(round(th * (1 - overlap))))
    out: list[TileMeta] = []
    seen: set[TileMeta] = set()
    nw, nh = native_size
    for ry in range(grid):
        for rx in range(grid):
            tx0 = x0 + min(rx * stride_w, max(0, sub_w - tw))
            ty0 = y0 + min(ry * stride_h, max(0, sub_h - th))
            tx1 = min(tx0 + tw, nw)
            ty1 = min(ty0 + th, nh)
            t = (tx0, ty0, tx1, ty1)
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Tile preprocessing
# ---------------------------------------------------------------------------


def crop_and_normalize(
    pil_img: Image.Image, tile: TileMeta, img_size: int
) -> torch.Tensor:
    """Crop ``pil_img`` to the tile bbox, resize to ``img_size`` and
    normalise with OWLv2 stats.  Returns a (3, S, S) tensor.
    """
    x0, y0, x1, y1 = tile
    crop = pil_img.crop((x0, y0, x1, y1))
    crop = crop.resize((img_size, img_size), Image.Resampling.BILINEAR)
    t = TF.to_tensor(crop)
    t = TF.normalize(t, OWLV2_MEAN, OWLV2_STD)
    return t


# ---------------------------------------------------------------------------
# Per-tile detection (uses cached prototype)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _detect_one_tile(
    model,
    prototype: torch.Tensor,         # (1, D_q)
    tile_tensor: torch.Tensor,       # (1, 3, S, S)
) -> dict[str, torch.Tensor]:
    """Run the query path of ``model`` on a single tile, using a cached
    prototype.  Returns a dict identical to ``model(...)`` output but with
    only the top-1 detection.
    """
    image_feats, feature_map = model.encode_query(tile_tensor)
    pred_logits, _ = model.owlv2.class_predictor(
        image_feats, prototype.unsqueeze(1)
    )                                                                # (1, P, 1)
    pred_logits = pred_logits.squeeze(-1)                            # (1, P)
    pred_boxes = model.owlv2.box_predictor(
        image_feats, feature_map, interpolate_pos_encoding=True
    )                                                                # (1, P, 4) cxcywh
    existence_prob, _ = model.existence_head(pred_logits, prototype, image_feats)

    P = pred_logits.size(1)
    best_idx = pred_logits.argmax(dim=-1).clamp(0, P - 1)
    ar = torch.arange(pred_logits.size(0), device=pred_logits.device)
    best_box = pred_boxes[ar, best_idx]                              # (1, 4)
    best_score = pred_logits[ar, best_idx]                           # (1,)
    return {
        "best_box": best_box,
        "best_score": best_score,
        "existence_prob": existence_prob,
    }


# ---------------------------------------------------------------------------
# Coordinate transforms — tile-local to native px
# ---------------------------------------------------------------------------


def _tile_local_cxcywh_to_native_xyxy(
    box_local: torch.Tensor, tile: TileMeta
) -> torch.Tensor:
    """Convert a tile-local cxcywh-normalised box to native xyxy pixels."""
    x0, y0, x1, y1 = tile
    tw = x1 - x0
    th = y1 - y0
    cx, cy, w, h = box_local.unbind(-1)
    bx1 = (cx - w / 2) * tw + x0
    by1 = (cy - h / 2) * th + y0
    bx2 = (cx + w / 2) * tw + x0
    by2 = (cy + h / 2) * th + y0
    return torch.stack([bx1, by1, bx2, by2], dim=-1)


# ---------------------------------------------------------------------------
# Public API: detect_tiled
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# M3: merge truncated boxes that straddle adjacent tile boundaries.
# ---------------------------------------------------------------------------


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter_x1 = max(float(a[0]), float(b[0]))
    inter_y1 = max(float(a[1]), float(b[1]))
    inter_x2 = min(float(a[2]), float(b[2]))
    inter_y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _box_touches_edge(box: torch.Tensor, tile: TileMeta, edge_px: int = 4) -> tuple[bool, bool, bool, bool]:
    """Return (left, top, right, bottom) flags for which tile edges the box hugs."""
    bx0, by0, bx1, by1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    tx0, ty0, tx1, ty1 = tile
    return (
        bx0 - tx0 <= edge_px,
        by0 - ty0 <= edge_px,
        tx1 - bx1 <= edge_px,
        ty1 - by1 <= edge_px,
    )


def _merge_boundary_pairs(
    boxes: list[torch.Tensor],
    scores: list[float],
    existences: list[float],
    tiles: list[TileMeta],
    edge_px: int = 4,
    min_partial_score: float = 0.2,
    range_overlap_min: float = 0.5,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    """For each pair of detections whose tiles are horizontally or vertically
    adjacent and whose boxes hug the shared edge, replace the pair with a
    single union box.

    Operates per pair greedily: we don't transitively chain merges (a target
    split across 3+ tiles is rare and would need a more careful clustering
    approach).
    """
    n = len(boxes)
    used = [False] * n
    out_boxes: list[torch.Tensor] = []
    out_scores: list[float] = []
    out_existences: list[float] = []

    for i in range(n):
        if used[i]:
            continue
        if scores[i] < min_partial_score:
            # Below merge threshold — keep as-is, don't try to merge.
            out_boxes.append(boxes[i])
            out_scores.append(scores[i])
            out_existences.append(existences[i])
            used[i] = True
            continue
        bi, ti = boxes[i], tiles[i]
        li, top_i, ri, bot_i = _box_touches_edge(bi, ti, edge_px)
        # Search for a partner.
        for j in range(i + 1, n):
            if used[j] or scores[j] < min_partial_score:
                continue
            bj, tj = boxes[j], tiles[j]
            lj, top_j, rj, bot_j = _box_touches_edge(bj, tj, edge_px)
            # Horizontal adjacency: i hugs right, j hugs left, tiles share x edge.
            horiz = ri and lj and abs(ti[2] - tj[0]) <= edge_px
            horiz |= li and rj and abs(tj[2] - ti[0]) <= edge_px
            # Vertical adjacency: i hugs bottom, j hugs top.
            vert = bot_i and top_j and abs(ti[3] - tj[1]) <= edge_px
            vert |= top_i and bot_j and abs(tj[3] - ti[1]) <= edge_px
            if not (horiz or vert):
                continue
            # Range overlap on the perpendicular axis.
            if horiz:
                # Compare y-ranges.
                a0, a1 = float(bi[1]), float(bi[3])
                b0, b1 = float(bj[1]), float(bj[3])
            else:
                a0, a1 = float(bi[0]), float(bi[2])
                b0, b1 = float(bj[0]), float(bj[2])
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            if union <= 0 or inter / union < range_overlap_min:
                continue
            # Merge.
            merged = torch.tensor([
                min(float(bi[0]), float(bj[0])),
                min(float(bi[1]), float(bj[1])),
                max(float(bi[2]), float(bj[2])),
                max(float(bi[3]), float(bj[3])),
            ], device=bi.device, dtype=bi.dtype)
            sc = max(scores[i], scores[j])
            ex = max(existences[i], existences[j])
            out_boxes.append(merged)
            out_scores.append(sc)
            out_existences.append(ex)
            used[i] = used[j] = True
            break
        if used[i]:
            continue
        # Unmerged.
        out_boxes.append(bi)
        out_scores.append(scores[i])
        out_existences.append(existences[i])
        used[i] = True

    return out_boxes, out_scores, out_existences


# ---------------------------------------------------------------------------
# Public API: detect_tiled
# ---------------------------------------------------------------------------


def detect_tiled(
    model,
    support_imgs: torch.Tensor,            # (1, V, 3, S, S) — single episode
    query_native: Image.Image,             # native-resolution PIL
    *,
    img_size: int = 768,
    mode: str = "pyramid_a",               # "pyramid_a" | "hybrid_d"
    levels: tuple[int, ...] = (1, 2),
    overlap: float = 0.30,
    nms_iou: float = 0.5,
    top_k: int = 100,
    score_combo: str = "existence_x_score",
    edge_score_penalty: float = 0.5,
    edge_px: int = 4,
    merge_partial_boxes: bool = True,
    merge_min_score: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Tile the native query image, run the model on each tile, NMS-merge.

    Args:
        model              : OWLv2FewShotLocalizer
        support_imgs       : (1, V, 3, S, S)  must already be normalised.
        query_native       : full-resolution PIL image (RGB).
        mode               : "pyramid_a" or "hybrid_d".
        levels             : pyramid levels for pyramid_a mode.
        overlap            : fractional overlap between adjacent tiles.
        nms_iou            : NMS IoU threshold for cross-tile merge.
        top_k              : maximum number of post-NMS detections kept.
        score_combo        : how to combine existence_prob and best_score.
        edge_score_penalty : multiplier (∈[0,1]) applied to a detection's
                             score when its bbox hugs an *interior* tile edge
                             (i.e. an edge that is not also an image border).
                             Setting this to 1.0 disables the penalty (M4 off);
                             0.5 (default) halves the score; 0.0 drops it.
        edge_px            : pixel tolerance for "edge-hugging".
        merge_partial_boxes: M3 — merge boundary-spanning detection pairs into
                             a single union box before NMS.  Set False to
                             disable.
        merge_min_score    : minimum score (post-edge-penalty) for a detection
                             to be considered as a merge component.

    Returns:
        {
          "best_box_native_xyxy": (4,)   in native pixel coords
          "best_score":            scalar
          "existence_prob":        scalar
          "all_boxes_native_xyxy": (N, 4)
          "all_scores":            (N,)
        }
    """
    device = support_imgs.device
    nw, nh = query_native.size

    # Encode supports once.  Mirror the residual-baseline prototype
    # construction from ``OWLv2FewShotLocalizer.forward`` so tile inference
    # uses exactly the same prototype the trained model produces.
    with torch.no_grad():
        baseline_proto = model.compute_baseline_prototype(support_imgs)  # (1, D_q)
        support_tokens = model.encode_support(support_imgs)
        correction = model.aggregator(support_tokens)                # (1, D_q)
        prototype = baseline_proto + model.aggregator_alpha * correction

    # First-pass tiles.
    tiles = pyramid_tiles((nw, nh), levels=levels, overlap=overlap)

    boxes_native: list[torch.Tensor] = []
    scores: list[float] = []
    existences: list[float] = []
    tile_of_box: list[TileMeta] = []
    per_tile_score: list[float] = []

    def _is_image_border_edge(tile: TileMeta, side: str) -> bool:
        """Return True if the named tile side coincides with the image edge.
        Detections hugging the *image* border are not penalised — only those
        hugging *interior* tile boundaries.
        """
        x0, y0, x1, y1 = tile
        if side == "left":
            return x0 <= 0
        if side == "right":
            return x1 >= nw
        if side == "top":
            return y0 <= 0
        if side == "bottom":
            return y1 >= nh
        return False

    def _run_tile(tile: TileMeta) -> tuple[torch.Tensor, float, float]:
        """Returns (box_native_xyxy, score_after_edge_penalty, existence_prob)."""
        tt = crop_and_normalize(query_native, tile, img_size).unsqueeze(0).to(device)
        out = _detect_one_tile(model, prototype, tt)
        box_native = _tile_local_cxcywh_to_native_xyxy(out["best_box"][0], tile)
        ex = float(out["existence_prob"][0].item())
        sc_logit = float(out["best_score"][0].item())
        sc_sigmoid = 1.0 / (1.0 + pow(2.71828, -sc_logit))
        if score_combo == "existence_only":
            sc = ex
        elif score_combo == "score_only":
            sc = sc_sigmoid
        elif score_combo == "existence_x_score":
            sc = ex * sc_sigmoid
        else:
            sc = ex * sc_sigmoid

        # M4: edge penalty.  Only penalise if the box hugs an *interior* edge.
        if 0.0 <= edge_score_penalty < 1.0:
            left, top, right, bottom = _box_touches_edge(box_native, tile, edge_px)
            interior_hug = (
                (left   and not _is_image_border_edge(tile, "left"))   or
                (right  and not _is_image_border_edge(tile, "right"))  or
                (top    and not _is_image_border_edge(tile, "top"))    or
                (bottom and not _is_image_border_edge(tile, "bottom"))
            )
            if interior_hug:
                sc = sc * edge_score_penalty
        return box_native, sc, ex

    for tile in tiles:
        bx, sc, ex = _run_tile(tile)
        boxes_native.append(bx)
        scores.append(sc)
        existences.append(ex)
        tile_of_box.append(tile)
        per_tile_score.append(sc)

    # Hybrid mode: dilate the highest-scoring tile.
    if mode == "hybrid_d" and tiles:
        best_tile_idx = int(max(range(len(tiles)), key=lambda i: per_tile_score[i]))
        sub_tiles = dilate_top_tile(tiles[best_tile_idx], (nw, nh), grid=2, overlap=overlap)
        for st in sub_tiles:
            if st in tiles:
                continue
            bx, sc, ex = _run_tile(st)
            boxes_native.append(bx)
            scores.append(sc)
            existences.append(ex)
            tile_of_box.append(st)

    if not boxes_native:
        return {
            "best_box_native_xyxy": torch.zeros(4, device=device),
            "best_score": torch.zeros((), device=device),
            "existence_prob": torch.zeros((), device=device),
            "all_boxes_native_xyxy": torch.zeros((0, 4), device=device),
            "all_scores": torch.zeros((0,), device=device),
        }

    # M3: merge boundary-spanning detection pairs into union boxes.
    if merge_partial_boxes and len(boxes_native) > 1:
        boxes_native, scores, existences = _merge_boundary_pairs(
            boxes_native, scores, existences, tile_of_box,
            edge_px=edge_px, min_partial_score=merge_min_score,
        )

    if not boxes_native:
        return {
            "best_box_native_xyxy": torch.zeros(4, device=device),
            "best_score": torch.zeros((), device=device),
            "existence_prob": torch.zeros((), device=device),
            "all_boxes_native_xyxy": torch.zeros((0, 4), device=device),
            "all_scores": torch.zeros((0,), device=device),
        }

    boxes_t = torch.stack(boxes_native, dim=0)                       # (N, 4)
    scores_t = torch.tensor(scores, device=device, dtype=torch.float32)
    existences_t = torch.tensor(existences, device=device, dtype=torch.float32)

    # Class-zero NMS across tiles.
    classes = torch.zeros(scores_t.numel(), dtype=torch.long, device=device)
    keep = batched_nms(boxes_t.float(), scores_t.float(), classes, iou_threshold=nms_iou)
    keep = keep[:top_k]
    boxes_kept = boxes_t[keep]
    scores_kept = scores_t[keep]
    ex_kept = existences_t[keep]

    if scores_kept.numel():
        best_idx = int(scores_kept.argmax().item())
        best_box = boxes_kept[best_idx]
        best_score = scores_kept[best_idx]
        best_existence = ex_kept[best_idx]
    else:
        best_box = torch.zeros(4, device=device)
        best_score = torch.zeros((), device=device)
        best_existence = torch.zeros((), device=device)

    return {
        "best_box_native_xyxy": best_box,
        "best_score": best_score,
        "existence_prob": best_existence,
        "all_boxes_native_xyxy": boxes_kept,
        "all_scores": scores_kept,
        "prototype_norm": float(prototype.norm().item()),
    }
