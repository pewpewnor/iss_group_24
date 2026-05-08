# Few-Shot Single-Object Localization — OWLv2 Rewrite

This plan is the **single source of truth** for the rewrite. Every Python
file under `modeling/`, plus `aggregator.py` and `modeling.ipynb`, is being
rewritten from scratch. No backward compatibility with the old Siamese /
MobileNet pipeline.

---

## 1. Task Definition

Given **4 support images** of the same object (different viewpoints, no
bounding boxes) and **1 query scene image**, output:

- `existence_prob` ∈ [0, 1] — probability that the object appears in the scene.
- `bbox` (cx, cy, w, h normalized to [0, 1]) — only meaningful when the model
  predicts existence. Single-object localization (no multi-detect).

This is a **meta-learning** task: support classes at test time are unseen
during training.

Optimization priorities (from user):

1. **Reduce false positives** above all else.
2. **mAP@50 ≥ 30%, mAP@50-95 ≥ 30%** on test split.
3. **No representation collapse** — no whole-image bbox; no existence stuck near 1.0.

---

## 2. Hardware

| Target              | GPU                          | VRAM | Native res  |
| ------------------- | ---------------------------- | ---- | ----------- |
| Laptop (default)    | NVIDIA RTX 2060 Mobile       | 6 GB | 768×768     |
| Google Colab        | T4                           | 15 GB| 960×960     |
| Google Colab Pro    | A100 40GB                    | 40 GB| 960×960     |

The notebook detects `USE_GOOGLE_COLAB` and adjusts batch / image size
accordingly. The CLI / Python entry points take an explicit `device` and
`img_size`; defaults are laptop-friendly (`768, batch=1, accum=8`).

---

## 3. Datasets

### 3.1 Sources used

| Source        | Role             | Path                              |
| ------------- | ---------------- | --------------------------------- |
| HOTS          | train + test     | `dataset/original/HOTS/HOTS_v1`   |
| InsDet        | train + test     | `dataset/original/InsDet`         |
| vizwiz_novel  | Phase 0 only     | `dataset/original/VizWiz`         |

VizWiz base, FSOD, HOPE are **unused** in this rewrite (deleted from the
aggregator).

### 3.2 HOTS layout
- Support pool per class: ALL `object/{train,test}/<cls>/*.png` cutouts on
  white. Train/test refers to photographer rounds, not generalization split,
  so we pool both. Bbox computed from the largest non-white connected
  component.
- Query pool per class: `scene/RGB/<scene>.png` paired with
  `scene/ObjectDetection/Annotations/<scene>.xml` (VOC format). Fan out: one
  `(image, bbox)` query per object instance present in a scene.
- Minimum 4 supports + 1 query per instance after bbox filtering.

### 3.3 InsDet layout
- Support pool per class: `Objects/<cls>/images/*.jpg`. Bbox from
  `Objects/<cls>/masks/<stem>.png`.
- Query pool per class: `Scenes/{easy,hard}/<scene>/*.jpg` + matching VOC
  XML. Fan out per annotation.
- Minimum 4 supports + 1 query per instance after bbox filtering.

### 3.4 vizwiz_novel layout (Phase 0 only)
- 16 categories, 1 image each from `support_images/*.{jpg,jpeg}` annotated
  by `support_set.json`.
- Rotation synthesis: 4 supports = (0°, 90°, 180°, 270°) of the same image,
  cropped tightly around the bbox with 10% padding. Query = original image
  with original bbox.

### 3.5 Train / test split
- 80/20, stratified by source (HOTS, InsDet), at the **instance** level.
  Test instances are completely unseen during training (true few-shot).
- Seed 42. At least 1 test instance per source guaranteed.
- vizwiz_novel: 100% goes to `phase0` split, untouched by training stages.

### 3.6 Negative episodes
- During episode sampling, with `neg_prob=0.5`, replace the query with a
  query from a *different* instance in the *same source*.
- No external negative pool (HOPE, InsDet/Background, VizWiz query images
  are dropped).

### 3.7 Aggregator output

`dataset/aggregated/manifest.json` with three split lists:
```jsonc
{
  "splits": {
    "train":  ["hots_apple", ...],
    "test":   ["hots_zebra", ...],
    "phase0": ["vizwiz_novel_condom_box", ...]
  },
  "instances": [
    {
      "instance_id": "hots_apple",
      "source": "hots",
      "class_name": "apple",
      "split": "train",
      "support_images": [
        {"path": "train/support/hots_apple/001.png", "bbox": [x1, y1, x2, y2]}
      ],
      "query_images": [
        {"path": "train/query/hots_apple/001.png",
         "bbox": [x1, y1, x2, y2],
         "scene_type": "hots_scene"}
      ]
    }
  ]
}
```

Image staging:
```
dataset/aggregated/
├── manifest.json
├── stats.json
├── train/{support,query}/<inst_id>/<NNN>.{png,jpg}
├── test/{support,query}/<inst_id>/<NNN>.{png,jpg}
└── phase0/{support,query}/<inst_id>/<NNN>.{png,jpg}
```

---

## 4. Architecture

### 4.1 Backbone — frozen OWLv2

Model: `google/owlv2-base-patch16-ensemble` loaded via
`Owlv2ForObjectDetection.from_pretrained(...)` and
`Owlv2Processor.from_pretrained(...)`.

OWLv2-base internals (we touch these by name):
- `owlv2.vision_model` — CLIP ViT-B/16 vision encoder. Hidden dim **768**.
- `owlv2.image_embedder` — wraps `vision_model` and reshapes patch tokens
  to a 2D grid suitable for box regression.
- `owlv2.class_predictor(image_embeds, query_embeds)` — cosine-similarity
  classification head; image_embeds: (B, P, 768), query_embeds: (B, Q, 768).
- `owlv2.box_predictor(image_embeds, feature_map)` — bbox regressor returning
  (cx, cy, w, h) normalized.

Frozen except where stages explicitly unfreeze.

Patch grid at 768×768: 48×48 = **2304 patches** per image (patch16).
Patch grid at 960×960: 60×60 = 3600 patches per image.

### 4.2 Multi-View Aggregator (custom, trainable)

```
INPUT
  support_imgs : (B, 4, 3, S, S)   normalized via Owlv2Processor
       │
       ▼
  vision_model.last_hidden_state per image
       → (B*4, P, 768)             P = (S/16)^2

  ── Foreground attention pool per view ─────────────────────────
   gate_logits = Conv1d(768, 1)(tokens.transpose(1,2)) → (B*4, 1, P)
   gate        = softmax(gate_logits / sqrt(768)) → (B*4, 1, P)
   topk_idx    = top-K=128 of gate                 → (B*4, 128)
   gathered    = tokens[topk_idx]                  → (B*4, 128, 768)
       reduces token count from P (2304/3600) to 128 per view
       and filters white background / clutter.

       ▼   reshape (B, 4, 128, 768)
  ── Add view position embedding ───────────────────────────────
   tokens += nn.Embedding(4, 768)(view_idx)

       ▼   flatten to (B, 512, 768)
  ── Inter-view self-attention (2 layers × 8 heads, pre-LN) ────
       ▼
  ── ISAB (Set Transformer) ────────────────────────────────────
   M=4 inducing points, 8 heads, GELU MLP×4
       ▼
  ── PMA (single seed) ─────────────────────────────────────────
   one learnable seed → cross-attention → (B, 1, 768)
       ▼
  ── LayerNorm + Linear(768, 768) ──────────────────────────────

OUTPUT prototype : (B, 768)   permutation-invariant view-fused embedding.
```

Trainable params: ~3.5M. Permutation-invariant by ISAB+PMA construction
(view position embedding is added but pooled out by PMA).

**Why foreground gating + top-K**: support images are object cutouts on
white (HOTS, InsDet) — most patches are useless. Letting all 2304/3600
patches feed inter-view attention is wasteful and dilutes the prototype.
The gate is initialised with a slight bias toward non-uniform softmax so
gradients flow through it from the start.

### 4.3 Existence Head (custom, trainable, ~10K params)

The user explicitly prioritises **false-positive reduction**, so this head
is given more capacity than PLAN.md's original 300-param sketch:

```
INPUT features per query (computed in forward):
  f1  = max patch logit (softmax-temperature differentiable max)  scalar
  f2  = top-5 mean of patch logits                                 scalar
  f3  = top-5 std  of patch logits                                 scalar
  f4  = mean patch logit                                           scalar
  f5  = cosine_sim(prototype, GAP(query patches))                  scalar
  f6  = ||prototype||                                              scalar
  f7  = ||GAP(query)||                                             scalar
  f8  = entropy of softmax over patches                            scalar

  → (B, 8) feature vector

ARCHITECTURE
  LayerNorm(8)
  → Linear(8, 64) → GELU → Dropout(0.2)
  → Linear(64, 64) → GELU → Dropout(0.2)
  → Linear(64, 1)
  → Sigmoid

OUTPUT existence_prob : (B,) ∈ [0, 1]
```

The 8 hand-crafted features force the head to look at *both* per-patch
matching strength (f1–f4, f8) *and* global prototype/scene alignment
(f5–f7). This decouples existence from "any patch matches anything".

### 4.4 Detection head (OWLv2 built-in)

Inference path:
```python
prototype     = aggregator(support_feats)                    # (B, 768)
query_embeds  = prototype.unsqueeze(1)                       # (B, 1, 768)

image_embeds, feature_map = owlv2.image_embedder(query_imgs) # (B, P, 768), (B, gh, gw, 768)
pred_logits, _            = owlv2.class_predictor(image_embeds, query_embeds)  # (B, P, 1)
pred_boxes                = owlv2.box_predictor(image_embeds, feature_map)     # (B, P, 4)

best_patch       = argmax(pred_logits.squeeze(-1), dim=1)
predicted_box    = pred_boxes[arange(B), best_patch]         # (cx, cy, w, h)
existence_prob   = existence_head(prototype, query_embeds, pred_logits)
```

### 4.5 LoRA (Stage 2.3)

`peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
target_modules=["q_proj", "v_proj"])` applied to the **last 4 transformer
blocks** of `owlv2.vision_model`. ~150K trainable params. r=8 (vs PLAN.md's
r=4) because we have a small dataset but a very different gradient signal
than CLIP pretraining and r=4 was empirically too restrictive on similar
tasks in OWL fine-tuning literature.

---

## 5. Loss Functions

### 5.1 Existence — Focal Loss
`α=0.25, γ=2.0` per PLAN.md spec. Used in all training stages.

### 5.2 Box — L1 + GIoU
Computed on positive episodes only and gated by
`existence_prob.detach() > 0.5`.

```
L_box = λ_L1·L1(pred, gt) + λ_GIoU·GIoU(pred, gt)
λ_L1   = 5.0
λ_GIoU = 2.0
```
GIoU heavily penalises predicted boxes much larger than GT — primary
guard against whole-image collapse.

### 5.3 Anti-collapse regularisers (NEW vs PLAN.md)

```
L_box_size_prior = (max(0, area(pred) - 0.6))^2          # soft repellent above 60% image area
L_existence_kl   = KL(mean_batch(existence_prob) || 0.5)  # only when mean > 0.85
L_anti_collapse  = 0.1 * (L_box_size_prior + L_existence_kl)
```

These directly attack the named failure modes from the user brief: bbox
collapse to whole-image and existence stuck near 1.0.

### 5.4 Combined per stage

| Stage     | Loss                                              |
| --------- | ------------------------------------------------- |
| Phase 0   | none (eval-only)                                  |
| Stage 1.1 | L_focal + 0.1·L_existence_kl                      |
| Stage 1.2 | L_focal + L_box + L_anti_collapse                 |
| Stage 2.3 | L_focal + L_box + L_anti_collapse                 |

---

## 6. Training Phases

### 6.1 Phase 0 — Zero-shot baseline (no training)

Run on `vizwiz_novel ∪ HOTS_test ∪ InsDet_test`.

For each episode:
- Run `owlv2.image_guided_detection(query_image=query, support_image=support_i)`
  for each of the 4 supports independently.
- Average the 4 score maps over patches.
- `predicted_box` = decoded box of the patch with max average score.
- `existence_prob` = sigmoid of max average score (no calibration).

Decision gate: if mAP@50 < 5% on **both** datasets (HOTS, InsDet), abort
and inspect annotations. Expected: 15-25% mAP on HOTS+InsDet thanks to
COCO + Objects365 pretraining, and ~5-15% on vizwiz_novel since those
are out-of-distribution household objects.

Output:
- `checkpoints/phase0/results.json`
- `analysis/phase0/results.json`

### 6.2 Stage 1.1 — Aggregator + existence-head warmup

| Knob              | Value                                |
| ----------------- | ------------------------------------ |
| Trainable         | aggregator, existence_head           |
| Frozen            | OWLv2 vision_model, box, class heads |
| Loss              | L_focal + 0.1·L_existence_kl         |
| Epochs (per fold) | 8                                    |
| Folds             | 5                                    |
| Episodes per fold | 200                                  |
| Batch (laptop)    | 1 episode × accum 8                  |
| Batch (T4/A100)   | 4 episodes × accum 2                 |
| LR aggregator     | 5e-4                                 |
| LR existence      | 5e-4                                 |
| Schedule          | linear-warmup 5% → cosine            |
| Early stop        | val existence AUROC stagnant 4 epochs|
| Primary metric    | val existence AUROC                  |

The aggregator must learn to fuse 4 views into a coherent prototype
**before** the detection heads start moving. We freeze everything else to
prevent OWLv2's COCO prior from being clobbered by the random aggregator
output.

### 6.3 Stage 1.2 — Detection head fine-tuning

| Knob              | Value                                    |
| ----------------- | ---------------------------------------- |
| Trainable         | aggregator, existence_head, box_head, class_head |
| Frozen            | OWLv2 vision_model                       |
| Loss              | L_focal + L_box + L_anti_collapse        |
| Epochs (per fold) | 15                                       |
| Folds             | 5                                        |
| Episodes per fold | 250                                      |
| Batch (laptop)    | 1 × accum 8                              |
| LR aggregator     | 1e-4                                     |
| LR existence      | 2e-4                                     |
| LR box, class     | 5e-5                                     |
| Box-head warmup   | epochs 1-3 freeze box_head; unfreeze at 4|
| Early stop        | val mAP@50 stagnant 5 epochs             |
| Primary metric    | val mAP@50                               |

The 3-epoch box-head freeze gives the existence head time to calibrate
before the bbox loss starts firing — prevents early collapse where the
model emits whole-image boxes everywhere because every prediction is
"present".

### 6.4 Stage 2.3 — LoRA on last-4 ViT blocks

| Knob              | Value                                          |
| ----------------- | ---------------------------------------------- |
| Trainable         | aggregator, existence, box, class, LoRA        |
| Frozen            | OWLv2 vision_model except LoRA q/v adapters    |
| Loss              | same as Stage 1.2                              |
| Epochs (per fold) | 8                                              |
| Folds             | 5                                              |
| Episodes per fold | 250                                            |
| Batch (laptop)    | 1 × accum 8                                    |
| LR aggregator     | 5e-5                                           |
| LR existence      | 5e-5                                           |
| LR box, class     | 2e-5                                           |
| LR lora           | 1e-4                                           |
| Early stop        | val mAP@50 on **unseen-class subset** stagnant 4 epochs |

If val mAP on the unseen-class subset of a fold drops below the Stage 1.2
checkpoint, immediately revert to Stage 1.2 best.

### 6.5 Stage epoch loop

Per user's clarification, *each epoch performs all K folds sequentially*
with shared running weights:

```python
for epoch in 1..stage_epochs:
    for fold in 0..K-1:
        train_ds.set_fold(train_ids=fold_plan[fold]["train_ids"])
        val_ds.set_fold(val_ids=fold_plan[fold]["val_ids"])
        train_metrics = train_one_pass(model, train_loader, ...)
        val_metrics = evaluate(model, val_loader, ...)
        save_checkpoint(stage, epoch, fold, model, optimizer, scheduler, ...)
        save_analytics_json(stage, epoch, fold, train_metrics, val_metrics)
        print_epoch_log(...)            # all losses + all val metrics
    aggregate = aggregate_folds(this_epoch_fold_jsons)
    save_aggregate_json(stage, epoch, aggregate)
    if aggregate.val_map_50_mean > best_so_far:
        save best.pt
```

Note: K-fold CV is run on the *train* split only. The held-out *test*
split is never seen during training and is used solely by
`evaluate_run(...)` cells in the notebook to produce the headline
post-stage numbers.

---

## 7. Evaluation Metrics

Computed per fold during training (val set = fold-out subset of train) and
once per stage on the held-out test set after training.

### 7.1 Localisation
- `iou_mean / iou_median / iou_p25 / iou_p75` — IoU of top-1 predicted
  box vs GT, computed only on positive episodes where `existence_prob > 0.5`.
- `contain_mean` — area(GT ∩ pred) / area(GT). Measures whether the
  predicted box contains the GT.
- `contain_at_iou_50 / 75` — fraction of positive episodes where the
  predicted box contains ≥ X% of GT and IoU ≥ threshold.

### 7.2 mAP
- 10 IoU thresholds 0.50:0.05:0.95 (COCO-style).
- 101-point AP per threshold.
- `map_50`, `map_75`, `map_5095`.
- TP definition: existence_prob > 0.5 AND IoU(pred, gt) ≥ threshold.
- FP: existence_prob > 0.5 on a negative episode, or low-IoU on positive.
- FN: existence_prob ≤ 0.5 on a positive episode.

### 7.3 Existence (image-level)
- `existence_acc / acc_pos / acc_neg`
- `existence_auroc` (Mann-Whitney U)
- `existence_pr_auc`
- `existence_brier`
- `existence_f1` at threshold 0.5
- `false_positive_rate / false_negative_rate`
- `mean_score_pos / mean_score_neg`

### 7.4 Collapse diagnostics
- `mean_pred_box_area` (normalized 0–1)
- `frac_pred_box_too_big` (area > 0.4 of image)
- `mean_existence_prob`
- `frac_high_existence` (existence_prob > 0.9)
- `prototype_norm_mean / std`
- `frac_pred_near_corner` (top-left of pred box near 0,0 — degenerate signal)

### 7.5 Per-source breakdown
Same metric set bucketed by `inst.source ∈ {hots, insdet}`.

---

## 8. Data Augmentation

Applied **only during training**. Validation/test/phase0 use only resize +
OWLv2 normalisation.

### 8.1 Support augmentations (per view independently)
- Random horizontal flip, p=0.5
- ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
- Random grayscale, p=0.2
- Random resized crop, scale=0.7–1.0
- Gaussian blur, p=0.3, kernel ∈ [3, 7]
- OWLv2 normalisation

The per-view independence is critical — it forces the aggregator to
disagree on view-specific colour/lighting cues and converge on
view-invariant geometry.

### 8.2 Query augmentations
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) — mild
- OWLv2 normalisation
- **No spatial augmentation** — bbox coordinates must remain valid.

---

## 9. Optimizer & Scheduler

```python
optimizer = AdamW([
    {"params": aggregator.parameters(),     "lr": cfg.lr_aggregator,    "weight_decay": 1e-4},
    {"params": existence_head.parameters(), "lr": cfg.lr_existence,     "weight_decay": 1e-4},
    {"params": box_head.parameters(),       "lr": cfg.lr_box,           "weight_decay": 1e-4},
    {"params": class_head.parameters(),     "lr": cfg.lr_class,         "weight_decay": 1e-4},
    {"params": lora_params,                 "lr": cfg.lr_lora,          "weight_decay": 1e-4},
], betas=(0.9, 0.999))

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7),
    ],
    milestones=[warmup_steps],
)
```

`warmup_frac=0.05`. Step the scheduler **per optimiser step**, not per
epoch (works correctly with gradient accumulation by counting accumulated
steps as 1 step).

Gradient clipping: `clip_grad_norm_(params, max_norm=1.0)`.

Mixed precision: `torch.amp.autocast("cuda", dtype=torch.float16)` on RTX
2060 (no bf16); bf16 on Colab A100. T4 uses fp16. Loss scaler:
`torch.amp.GradScaler("cuda")`.

---

## 10. Checkpointing

### 10.1 Directory layout
```
checkpoints/
├── phase0/
│   └── results.json
├── stage_1_1/
│   ├── fold0_epoch001.pt   (kept for last 6 (epoch, fold) tuples per stage)
│   ├── fold0_epoch002.pt
│   ├── ...
│   ├── best.pt             (max aggregate val_map_50_mean)
│   ├── last.pt             (latest write)
│   └── stage_complete.pt
├── stage_1_2/
│   └── ...
└── stage_2_3/
    └── ...
```

### 10.2 Per-(epoch, fold) payload
```python
{
  "stage": "1_1",
  "epoch": int,
  "fold":  int,
  "stage_completed": bool,
  "global_step": int,
  "model": {
      "aggregator":     state_dict,
      "existence_head": state_dict,
      "box_head":       state_dict | None,
      "class_head":     state_dict | None,
      "lora_state":     state_dict | None,
  },
  "optimizer": optimizer.state_dict(),
  "scheduler": scheduler.state_dict(),
  "scaler":    scaler.state_dict() | None,
  "best_aggregate_map_50": float,
  "early_stop_counter":    int,
  "rng": {torch, numpy, python, cuda},
  "config": cfg,
  "fold_plan": {...},
  "metrics_history": [...],
}
```

### 10.3 Resume protocol
A single `resume: bool | str` argument:
- `True` → load `last.pt` if it exists.
- `False` → fresh start.
- str → resolved against `out_dir` then loaded.

Resume crosses stage boundaries: when the loaded checkpoint's stage is
complete, the trainer rebuilds optimizer + scheduler for the next stage
and starts at epoch 1, fold 0. The aggregator + heads state carries over.

---

## 11. Analytics JSON

Every `(epoch, fold)` writes:

`analysis/{stage}/epoch_{e:03d}/fold_{f}.json`:
```json
{
  "stage": "1_1", "epoch": 1, "fold": 0,
  "wall_clock_seconds": 124.3,
  "lr": {"aggregator": ..., "existence_head": ..., ...},
  "train": {
    "loss": ..., "focal": ..., "l1": ..., "giou": ...,
    "box_area_penalty": ..., "existence_kl": ...,
    "grad_norm": ..., "n_steps": ...
  },
  "val": {
    "overall": { ...kitchen sink... },
    "per_source": { "hots": {...}, "insdet": {...} }
  }
}
```

End-of-epoch: `analysis/{stage}/epoch_{e:03d}/aggregate.json` — recursive
mean/min/max/std across the K folds.

`analysis/summary.json` — rolling best-by-metric pointer (one entry per
metric of interest).

`analysis/folds.json` — fold plan (instance ids per fold's train/val).

`analysis/config.json` — full training config snapshot.

---

## 12. Notebook Structure

`modeling.ipynb` keeps the existing cell sequence:

1. Markdown intro
2. Autoreload setup
3. `USE_GOOGLE_COLAB` constant + `SHARED_FOLDER_LINK`
4. Drive mount (only if `USE_GOOGLE_COLAB`)
5. `setup_repo()` (only if `USE_GOOGLE_COLAB`)
6. Manifest paths
7. Repo clone + dep install (only if `USE_GOOGLE_COLAB`)
8. Markdown — Build dataset manifest
9. `import aggregator; aggregator.main()`
10. Markdown — Imports + helpers (`evaluate_run`)
11. Imports
12. Markdown — Shared kwargs
13. `SHARED_KWARGS = dict(...)` — every hyperparameter
14. `EVAL_KWARGS = dict(...)`

Then for each phase/stage, two cell pairs (markdown + code):

15. Markdown — Phase 0
16. `train_phase0(**SHARED_KWARGS)` (just runs eval, writes `checkpoints/phase0/results.json`)
17. Markdown — Evaluate Phase 0 on test split
18. `evaluate_phase0(**EVAL_KWARGS)` (runs the full eval on HOTS+InsDet test)

19. Markdown — Stage 1.1
20. `train_stage_1_1(**SHARED_KWARGS)`
21. Markdown — Evaluate Stage 1.1 on test
22. `evaluate_run(checkpoint='checkpoints/stage_1_1/best.pt', **EVAL_KWARGS)`

(repeat for 1.2 and 2.3)

29. Markdown — Plots
30. `plot_all_from_jsons('analysis')`

The Google Colab cells (drive mount, clone, pip install) all gate behind
`if USE_GOOGLE_COLAB:`. T4 vs RTX 2060 image_size auto-detected via
`torch.cuda.get_device_properties(0).total_memory > 10e9`.

---

## 13. Hyperparameter Summary

```
seed                 = 42
img_size_laptop      = 768
img_size_colab       = 960
batch_laptop         = 1
batch_colab_t4       = 4
batch_colab_a100     = 8
grad_accum_laptop    = 8
grad_accum_t4        = 2
grad_accum_a100      = 1
weight_decay         = 1e-4
grad_clip            = 1.0
warmup_frac          = 0.05
amp_dtype            = float16

aggregator_dim       = 768  (matches OWLv2 hidden)
support_topk         = 128
view_count           = 4
isab_M               = 4
attn_heads           = 8
inter_view_layers    = 2

phase0_episodes      = "all"   (one episode per test/phase0 instance)
stage_1_1_epochs     = 8
stage_1_2_epochs     = 15
stage_2_3_epochs     = 8
folds                = 5
fold_seed            = 42
episodes_per_fold_s1 = 200
episodes_per_fold_s2 = 250
episodes_per_fold_s3 = 250
val_episodes         = 100   (per fold's val set, sampled deterministically)

lr_aggregator_s1     = 5e-4
lr_existence_s1      = 5e-4
lr_aggregator_s2     = 1e-4
lr_existence_s2      = 2e-4
lr_box_s2            = 5e-5
lr_class_s2          = 5e-5
lr_aggregator_s3     = 5e-5
lr_existence_s3      = 5e-5
lr_box_s3            = 2e-5
lr_class_s3          = 2e-5
lr_lora_s3           = 1e-4

lora_r               = 8
lora_alpha           = 16
lora_dropout         = 0.1
lora_target_modules  = ["q_proj", "v_proj"]
lora_layers          = "last_4"

focal_alpha          = 0.25
focal_gamma          = 2.0
lambda_l1            = 5.0
lambda_giou          = 2.0
anti_collapse_weight = 0.1
box_size_threshold   = 0.6
existence_kl_threshold = 0.85

neg_prob             = 0.5
support_box_pad      = 0.05
keep_last_n          = 6

ema                  = OFF (small dataset, EMA on tiny aggregator hurts)
```

---

## 14. File-by-file rewrite checklist

| Path                         | Action                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `aggregator.py`              | Rewrite. HOTS+InsDet → train/test, vizwiz_novel → phase0. Drop everything else. |
| `pyproject.toml`             | Add `transformers>=4.40`, `accelerate`, `peft>=0.10`, `huggingface_hub`. Drop `litert-torch`. |
| `requirements.txt`           | Same as above.                                                                |
| `modeling/dataset.py`        | Rewrite. `EpisodeDataset` (HOTS+InsDet), `Phase0Dataset` (vizwiz_novel rotation). |
| `modeling/model.py`          | Rewrite. `OWLv2FewShotLocalizer` wrapping HF model + aggregator + existence head. |
| `modeling/loss.py`           | Rewrite. Focal + L1 + GIoU + anti-collapse only.                              |
| `modeling/evaluate.py`       | Rewrite. Existence-gated mAP + AUROC + collapse diagnostics + per-source.     |
| `modeling/train.py`          | Rewrite. `train_phase0`, `train_stage_1_1`, `train_stage_1_2`, `train_stage_2_3`, plus `evaluate_phase0`, `evaluate_run`. |
| `modeling/_folds.py`         | Update. K=5 stratified across HOTS+InsDet only.                              |
| `modeling/_loaders.py`       | Update. Plain DataLoader, no source-balanced sampler.                        |
| `modeling/_optim.py`         | Update. Per-stage param groups (aggregator/existence/box/class/lora) + cosine. |
| `modeling/_checkpoint.py`    | Update. New stage names + lora_state slot. Keep atomic-save / RNG / hygiene. |
| `modeling/_analysis.py`      | Keep verbatim (already general).                                             |
| `modeling/_logging.py`       | Update. New metric priority lists.                                           |
| `modeling/_train_loop.py`    | Rewrite. Simpler loss combo, gradient accumulation, no proto cache.          |
| `modeling/_proto_cache.py`   | DELETE.                                                                      |
| `modeling/export.py`         | DELETE.                                                                      |
| `modeling/plot.py`           | Update plot keys for new metric names; keep core plotting.                   |
| `modeling.ipynb`             | Update content of training/eval cells; keep dual-mode structure.             |

---

## 15. Architectural improvements over the spec in this file's earlier draft

1. **Foreground attention pooling per support view, top-K=128**.
   Rejects white-background tokens before view mixing. PLAN.md original
   draft did not address that support images often have empty backgrounds.

2. **Existence head: 8 hand-crafted features → 64 → 64 → 1**, ~10K params.
   Vastly more capacity than the original 1-feature 16-unit MLP. Forces the
   head to consider both per-patch matching strength and global
   prototype/scene alignment. Directly attacks the user's explicit
   false-positive-reduction priority.

3. **Anti-collapse regularisers** (box-area soft prior + existence-mean
   KL). Both named user concerns (whole-image bbox; existence stuck near 1)
   are explicitly penalised in the loss.

4. **LoRA r=8** instead of r=4. Small dataset but the aggregator gradient
   signal differs significantly from CLIP pretraining; r=4 too restrictive.

5. **Hard-negative cache removed**. With only 2 sources, same-source
   different-instance sampling already gives the necessary discrimination
   signal. The cache adds complexity for marginal benefit.

6. **K-fold CV across all stages**, not only the final one. Per user
   instruction: "each epoch performs K-fold CV". Same model carries fold-
   to-fold within an epoch.

7. **Box-head freeze first 3 epochs of Stage 1.2**. Lets the existence
   head calibrate before bbox loss fires; prevents early collapse.
