# Few-Shot Two-Model System — Implementation Plan

This document describes the codebase actually shipped under this directory.
It supersedes any prior PLAN.md and is the single source of truth for what the
code does.

---

## 1. Two-Model Architecture

### 1.1 Multi-Shot Localizer (`localizer/`)

Predicts a bounding box `(cx, cy, w, h)` conditioned on knowing the object is
present. Trained **only** on positive episodes — existence is the siamese's
job.

- **Backbone (frozen)**: `google/owlv2-base-patch16-ensemble` (ViT-B/16 vision
  encoder, 768-D vision tokens, 512-D query embedding).
- **Per-support path**: each of the K supports passes through OWLv2's native
  `image_embedder` + `embed_image_query` to get a 512-D query embedding.
- **Fusion (trainable, ~2.5M params)**: stack K embeddings with a learnable
  `[CLS]` token; pass through a 2-layer pre-LN transformer encoder with
  `key_padding_mask` over the K_max slots; the CLS output is a *correction*.
  Permutation-invariant by construction (no positional embedding).
- **Residual identity path** (key fix): the prototype is built as
  ``prototype = mean(per_support_q_emb) + alpha * fusion_correction`` with
  `alpha` initialised at **0.01** (not 0). This means epoch-0 quality
  ≈ zero-shot OWLv2 (the mean is exactly what `phase0_forward` uses), and
  the fusion learns a small correction on top. `alpha` is trainable; if the
  fusion is harmful the optimiser will drive `alpha → 0`.
- **Detection**: query image → `image_embedder` → `class_predictor(prototype)`
  → argmax patch → `box_predictor[argmax]`. Output `best_box` in `(cx, cy, w, h)`.

Loss:
| Component                    | Definition                                         | Weight |
| ---------------------------- | -------------------------------------------------- | ------ |
| `L_patch_ce`                 | cross-entropy on the patch nearest GT centre       | 1.0    |
| `L_l1`  (L2 / L3 only)       | L1 on `pred_boxes[argmax]` vs GT (cxcywh)          | 5.0    |
| `L_giou` (L2 / L3 only)      | GIoU on `pred_boxes[argmax]` vs GT (xyxy)          | 2.0    |

The patch-CE term replaces the previous "soft-box" trick. It directly demands
prototype DISCRIMINATION across patches (the prototype must produce a high
logit on the patch nearest the GT centre), so there's no fixed-point
"predict everywhere" solution. At L1 the box L1+GIoU terms produce zero
gradient (frozen box_head) and are skipped to avoid wasted compute.

Stages:
| Stage | Trainable                             | Loss                          | Epochs | Eps/fold |
| ----- | ------------------------------------- | ----------------------------- | ------ | -------- |
| L1    | fusion + CLS + alpha                  | patch_ce only                 | 3      | 400      |
| L2    | + `class_head` + `box_head` + `layer_norm` | patch_ce + L1 + GIoU     | 12     | 250      |
| L3    | + LoRA r=8 on `q_proj/v_proj` of last 4 ViT blocks | patch_ce + L1 + GIoU | 8     | 250      |

Headline metric: **mAP@50** (over positive episodes only). Per-K and per-source
breakdowns (K∈{1, 4, 10}; sources hots/insdet) reported every epoch. Plus a
full mAP@50:95 + per-IoU AP curve, IoU stats, containment family, geometry
diagnostics, and `alpha` snapshot.

### 1.2 Multi-Shot Siamese (`siamese/`)

Predicts `existence_prob ∈ [0, 1]`. Trained on mixed positive + negative
episodes at ratio 1:3.

- **Backbone (frozen)**: `facebook/dinov2-small` (ViT-S/14, 384-D, 12 layers).
- **Cross-attention pool (trainable)**: query patch tokens attend to a
  flattened bag of all K supports' patch tokens (with a `key_padding_mask` over
  padded slots) → mean-pooled to a single 384-D vector.
- **Scalar features**: 6 hand-crafted scalars derived from cosine similarities
  between query and support patches (max, top-5 mean/std, mean, CLS-cosine,
  entropy).
- **Head MLP**: concatenate `[pooled, q_cls, sup_cls_mean, scalars]` (∼1158-D)
  → LayerNorm → Linear(.,256) → GELU → Dropout → Linear(256,64) → GELU →
  Dropout → Linear(64,1) → sigmoid.

Stages:
| Stage | Trainable                                                  | Loss                          | Epochs | Eps/fold |
| ----- | ---------------------------------------------------------- | ----------------------------- | ------ | -------- |
| S1    | cross-attn + scalars + head MLP                            | focal + variance + decorrelation | 10  | 400      |
| S2    | + LoRA r=8 on `query/value` of last 4 DINOv2 blocks        | same                          | 8      | 400      |

Loss components:
- `focal_BCE(α=0.25, γ=2.0)` — α=0.25 weights NEGATIVES higher, penalising
  false positives more (per the user's stated priority).
- `variance_reg = relu(0.5 - std_of_pooled_per_dim_across_batch).mean()` —
  blocks "always output the same vector" collapse.
- `decorrelation_reg = ((corr(pooled.T) − I)**2).mean()` — blocks "all
  dimensions encode the same thing" collapse.
- Total: `focal + 0.1·variance + 0.05·decorrelation`.

**Hard-negative cache**: misclassified negatives (existence_prob > 0.5 on a
known-negative episode) are recorded into a per-stage cache; the next
epoch's negative sampler draws `hard_neg_cache_frac=0.5` of negatives from
this cache.

Headline metric: **AUROC** (`early_stop_metric="auroc"`); FPR closely watched.
`early_stop_metric="fpr_inv"` is also supported if you want to optimize FPR
directly.

---

## 2. Repository Layout

```
iss_group_24/
├── aggregator.py                 # idempotent dataset builder + --validate
├── inference_localizer.py        # CLI / API: K supports + 1 query → bbox
├── inference_siamese.py          # CLI / API: K supports + 1 query → existence_prob
├── inference_combined.py         # cascaded siamese → localizer + sweep_threshold
├── modeling.ipynb                # rewritten notebook (two parallel sections)
├── PLAN.md                       # this file
├── pyproject.toml / requirements.txt
│
├── shared/                       # used by BOTH packages
│   ├── manifest.py               # load + validate manifest.json (schema v2)
│   ├── dataset.py                # EpisodeDataset (variable K, padded+masked)
│   ├── folds.py                  # K=3 stratified folds
│   ├── checkpoint.py             # atomic save/load + RNG capture
│   ├── analytics.py              # write_json, aggregate_folds, summary
│   ├── logging.py                # console formatters
│   ├── plots.py                  # plot_all_from_jsons(...)
│   └── smoke.py                  # ~60s end-to-end smoke_test()
│
├── localizer/
│   ├── model.py                  # MultiShotLocalizer
│   ├── dataset.py                # positive-only loader builders
│   ├── loss.py                   # L1 + GIoU
│   ├── optim.py                  # per-stage param groups
│   ├── train_loop.py             # one-pass training
│   ├── train.py                  # train_phase0 / L1 / L2 / L3 / evaluate_run
│   └── evaluate.py               # mAP@50 + IoU + per-K/per-source
│
└── siamese/
    ├── model.py                  # MultiShotSiamese
    ├── dataset.py                # mixed pos+neg loader builders
    ├── loss.py                   # focal + variance + decorrelation
    ├── optim.py                  # head + (S2) LoRA param groups
    ├── train_loop.py             # one-pass training (with hard-neg recorder)
    ├── train.py                  # train_phase0 / S1 / S2 / evaluate_run
    └── evaluate.py               # AUROC, PR-AUC, FPR, FNR, accuracy, brier
```

---

## 3. Data Pipeline (`shared/dataset.py`)

The same `EpisodeDataset` serves both models. Models pull only the fields they
need from each batch.

Episode contract:
```
support_imgs : (K_max, 3, S, S)   un-normalized RGB; padded slots are zeros.
support_mask : (K_max,) bool      True for real supports.
k            : int                actual K used this episode.
query_img    : (3, S, S)          un-normalized RGB.
query_bbox   : (4,)               cxcywh in [0, 1] (zeros for negatives).
is_present   : bool tensor
instance_id, source : str
native_size, native_bbox          for inference / debugging.
```

Each model normalizes inside its own forward (CLIP vs ImageNet stats).

Support preprocessing (per the user's directive: **NO bbox cropping**):
- Letterbox resize (preserve aspect ratio, pad to square with mean colour).
- Train augmentation: hflip, random-resized-crop scale 0.5–1.0, ColorJitter,
  random grayscale, Gaussian blur, random erasing 5–20% area.
- Eval/test/phase0: letterbox only.

Query: letterbox + (train) mild ColorJitter. **No** spatial aug (preserves bbox).

K∈{1..10} sampled uniformly per episode at train time. Eval uses a deterministic
roundrobin over `(k_min, 4, k_max)` so per-K metrics are stable.

Negative episodes (siamese only): query drawn from a different instance of the
same source. Optional hard-negative cache lookup with probability
`hard_neg_cache_frac`.

---

## 4. Aggregator (`aggregator.py`)

`schema_version=2`. Idempotent — skips if `dataset/aggregated/manifest.json`
already exists and validates. Sources used:

| Source       | Role             |
| ------------ | ---------------- |
| HOTS         | train + test (80/20 stratified) |
| InsDet       | train + test (80/20 stratified) |

CLI:
```
uv run python aggregator.py            # build (idempotent)
uv run python aggregator.py --force    # rebuild from scratch
uv run python aggregator.py --validate # only validate
```

---

## 5. Checkpointing (`shared/checkpoint.py`)

Single payload format per `*.pt` file:

```python
{
  "model_kind": "localizer" | "siamese",
  "stage": "L1" | "L2" | "L3" | "S1" | "S2" | "phase0",
  "epoch": int, "fold": int, "stage_completed": bool,
  "global_step": int,
  "state_dict": flat_state_dict_of_TRAINABLE_only,   # ~10 MB, not 600 MB
  "lora_active": bool,
  "optimizer": ..., "scheduler": ..., "scaler": ...,
  "rng": {torch, numpy, python, cuda},
  "config": full_resolved_cfg_dict,
  "fold_plan": list,
  "metrics_history": list,
  "best_metric": {"value": float, "epoch": int, "fold": int},
  "early_stop_counter": int,
}
```

Resume rules:
- `resume=True` → load `<out_dir>/last.pt` if present.
- `resume=False` → fresh start.
- `resume=<str>` → resolve as path.

Cross-stage warm-start: when entering a new stage, the previous stage's
`stage_complete.pt` is auto-loaded if present. Optimizer / scheduler are
rebuilt for the new stage.

Files written per stage:
- `ckpt_fold{F}_epoch{E:03d}.pt` — every per-(epoch, fold) snapshot.
- `last.pt` — atomic alias to the latest write.
- `best.pt` — atomic alias when val metric improves.
- `stage_complete.pt` — written at stage end.

`keep_last_n=0` ⇒ never delete rolling per-(epoch, fold) checkpoints
(Drive-friendly).

### Google Drive durability (Colab)

When running on Colab the notebook pins ``OUT_ROOT`` and ``ANALYSIS_ROOT`` to
the Drive-mounted project root, and the path-setup cell now calls
``assert_checkpoint_root_on_drive(OUT_ROOT, on_colab=USE_GOOGLE_COLAB)`` —
training refuses to start if those roots are NOT on Drive, so a misconfigured
path can never silently lose checkpoints to the runtime SSD.

``atomic_save`` (and ``write_json``) implement durable writes:

  1. ``mkdir -p`` the parent directory.
  2. Write payload to ``<path>.tmp`` via a regular file handle.
  3. ``f.flush()`` then ``os.fsync(f.fileno())`` to force bytes through
     Drive's FUSE cache. (Without fsync, ``torch.save`` returns long before
     the bytes are durable; a runtime kill would lose the checkpoint.)
  4. ``os.replace(tmp, path)`` for atomic rename.
  5. ``os.fsync()`` the parent directory entry (best-effort).

The whole sequence is retried up to 3 times with exponential back-off on
transient ``OSError`` — Drive's FUSE intermittently throws EIO during long
sessions, and a single retry usually clears it.

Net effect: every per-(epoch, fold) checkpoint, every ``best.pt`` / ``last.pt``
update, and every JSON analysis file is forced to disk before the call
returns, and is visible from the Drive web UI within seconds.

---

## 6. Smoke Test (`shared/smoke.py`)

`smoke_test(seconds_budget=60)` covers:
1. `aggregator.validate(strict=True)`.
2. Localizer Phase 0 + L1/L2/L3 (1 epoch × 1 fold × 4 episodes × img=224 × K_max=2).
3. Localizer `evaluate_run` on each stage's `stage_complete.pt`.
4. Siamese Phase 0 + S1/S2 (same shape).
5. Siamese `evaluate_run` on each stage.
6. `inference_combined.run_combined(...)` on a single (supports, query) tuple.
7. `plot_all_from_jsons('analysis/_smoke')`.
8. Localizer L3 checkpoint reload roundtrip.

Every train/evaluate function accepts `smoke=True` to dial down to the tiny
config. All artifacts go under `checkpoints/_smoke/` and `analysis/_smoke/`
and are wiped at the end (unless `cleanup=False`).

CLI:
```
uv run python -m shared.smoke --seconds-budget 60
```

---

## 7. Combined Inference (`inference_combined.py`)

Cascaded siamese → localizer:

```python
run_combined(
    siamese_ckpt:           str,
    localizer_ckpt:         str,
    support_paths:          list[str],
    query_path:             str,
    *,
    existence_threshold:    float = 0.5,
    existence_threshold_mode: str = "hard",   # "hard" | "soft" | "always_localize"
    siamese_img_size:       int = 518,
    localizer_img_size:     int = 768,
    out_root:               str = "inference/combined",
    bbox_color, bbox_thickness, smoke,
) -> dict
```

Modes:
- **`"hard"`** (default) — siamese first. If `existence_prob < threshold`,
  return `{exists: False, bbox: None}` and skip the localizer. Else run the
  localizer.
- **`"soft"`** — always run both, return both fields. The bbox is annotated
  as low-confidence below threshold.
- **`"always_localize"`** — always return localizer's bbox plus the siamese's
  existence_prob, regardless of threshold.

Threshold sweep against the test split:
```python
sweep_threshold(
    siamese_ckpt, localizer_ckpt, *,
    test_episodes=400,
    thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
)
```
Writes `analysis/combined/threshold_sweep_<ts>.json` and prints the FPR / FNR
/ accuracy / mAP@50-on-passed-positives Pareto curve.

---

## 8. Notebook Layout (`modeling.ipynb`)

```
# ISS Group 24 — Two-Model Few-Shot System
%autoreload setup
USE_GOOGLE_COLAB / paths

## Step 0 — aggregator (idempotent)
## Step 1 — smoke test (RUN_SMOKE_TEST = True)
## Step 2 — imports + SHARED_KWARGS / LOC_KWARGS / SIA_KWARGS

# Localizer
## Phase 0  →  loc_train_phase0(**LOC_KWARGS)
            →  loc_evaluate_phase0(**LOC_KWARGS)
## L1       →  loc_train_L1(**LOC_KWARGS)
            →  loc_evaluate_run(checkpoint=..., **LOC_KWARGS)
## L2       →  same pattern
## L3       →  same pattern

# Siamese
## Phase 0  →  sia_train_phase0(**SIA_KWARGS)
## S1       →  sia_train_S1(**SIA_KWARGS) ; sia_evaluate_run(...)
## S2       →  same pattern

# Combined
## sweep_threshold(...)
## run_combined(...) — example call

# Plots
plot_all_from_jsons(ANALYSIS_ROOT)
```

---

## 9. Hyperparameter Surface

Every knob is exposed in `localizer/train.py::DEFAULT_CFG` and
`siamese/train.py::DEFAULT_CFG`. Both are merged with the user's kwargs in
`_merge_cfg`, so any kwarg passed to a `train_*` function overrides the
default. `smoke=True` applies a tiny-config override layer that the user's
explicit kwargs still override.

Categories of knobs (full list in the source files):
- **I/O**: manifest, data_root, out_root, analysis_root.
- **Hardware**: img_size, batch_size, grad_accum_steps, num_workers, use_amp,
  device.
- **Folds**: folds (K=3), fold_seed, val_episodes, test_episodes.
- **K range**: k_min, k_max.
- **Stage durations**: `<S>_epochs`, `<S>_eps_per_fold`.
- **LRs**: per-stage per-group (e.g. `lr_fusion_L1`, `lr_class_L2`,
  `lr_lora_L3`, `lr_head_S1`, `lr_lora_S2`).
- **Optim**: weight_decay, grad_clip, warmup_frac.
- **Loss**:
  - Localizer: `lambda_l1`, `lambda_giou`, `L2_box_warmup_epochs`.
  - Siamese: `focal_alpha`, `focal_gamma`, `variance_target`,
    `variance_weight`, `decorr_weight`, `neg_prob`, `hard_neg_cache_frac`.
- **Architecture**:
  - Localizer: `fusion_layers`, `fusion_heads`, `fusion_mlp_ratio`,
    `fusion_dropout`.
  - Siamese: `cross_attn_heads`, `cross_attn_dropout`, `head_hidden_1`,
    `head_hidden_2`, `head_dropout`.
- **LoRA**: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_last_n_layers`,
  `lora_target_modules`.
- **Augmentation**: `aug_color_jitter`, `aug_hue`, `aug_grayscale_prob`,
  `aug_blur_prob`, `aug_blur_sigma`, `aug_erase_prob`, `aug_erase_scale`,
  `aug_rrc_scale`, `aug_hflip_prob`, `aug_query_color_jitter`.
- **Eval / inference**: `eval_threshold`, `existence_threshold`,
  `existence_threshold_mode`.
- **Early stop**: `<S>_early_stop_patience`, `early_stop_metric` (siamese only:
  `"auroc"` or `"fpr_inv"`).
- **Smoke override**: `smoke=True`.

---

## 10. Success Criteria (User-Stated Targets)

| Metric                                    | Target              |
| ----------------------------------------- | ------------------- |
| Localizer InsDet mAP@50                   | ≥ 30%               |
| Localizer HOTS mAP@50                     | ≥ 30%               |
| Multi-shot scaling                        | mAP@50(K=10) > mAP@50(K=1) |
| Siamese AUROC                             | ≥ 0.80 overall      |
| Siamese FPR @ thr=0.5                     | ≤ 10%               |
| No representation collapse                | pooled std > 0.1 per dim |
| Resumability                              | full coverage       |
| Smoke test                                | < 60s wall clock    |
