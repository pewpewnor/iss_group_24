# PLAN — Few-Shot Localization Rewrite (v5)

Scope: complete rewrite. No backwards compatibility. Total wall-clock budget: 24h on a single Colab T4. Three training stages, K=3 rotating CV in Stage 3, full per-(stage, epoch, fold) resumability, JSON-only per-epoch analysis. Architecture overhaul to escape representational collapse and unblock mAP.

## 1. Goals and non-goals

**Goals**
- Push mAP@0.5 from <0.10 to a meaningful target (>0.30) on the held-out test split.
- Eliminate the "predict near corner" / "always present" collapse modes.
- Train end-to-end in <24h on a Colab T4.
- Resume from any saved checkpoint (per-epoch, per-fold, stage-completion, last, best).
- Cross-domain generalisation: every batch sees HOTS, InsDet, VizWiz base, VizWiz novel.

**Non-goals**
- Multi-class detection (single class-agnostic detector remains).
- Live deployment changes (export path is preserved but not the focus).

## 2. Three-stage training schedule

| Stage | Trainable | LR (heads / upper / lower) | Epochs | CV | Val source |
|---|---|---|---|---|---|
| **1 — head warmup** | heads only, backbone frozen | 3e-4 / — / — | 5 | no | `test` split sample (200 episodes) |
| **2 — partial unfreeze** | heads + features[7:] | 2e-4 / 5e-5 / — | 8 | no | `test` split sample (200 episodes) |
| **3 — full unfreeze + K=3 CV** | full backbone + heads | 1e-4 / 5e-6 / 5e-6 | 35 | yes (K=3 stratified by source) | each fold's val partition |

Stage 3 protocol per epoch: `for fold in 0..K-1: train_one_pass(fold) → validate(fold) → save_ckpt(epoch, fold)`. The same model accumulates updates across all K folds within an epoch (rotating-fold validation, not classical K-fold).

Defaults (24h budget; T4):
- `episodes_per_epoch_s1 = 1500`, `val_episodes_s1 = 400`
- `episodes_per_epoch_s2 = 1500`, `val_episodes_s2 = 400`
- `episodes_per_epoch_s3 = 2000` (split across 3 folds), `val_episodes_s3 = 240` per fold
- `batch_size = 16`, `num_workers = 2`

Estimated wall-clock: Stage 1 ≈ 1h, Stage 2 ≈ 2h, Stage 3 ≈ 9h, final test eval ≈ 5min ⇒ ~12h total with headroom.

## 3. Universal resume protocol

### Checkpoint files

```
out_dir/
  ckpt_s1_epoch{E:03d}.pt              # Stage 1 per-epoch
  ckpt_s2_epoch{E:03d}.pt              # Stage 2 per-epoch
  ckpt_s3_epoch{E:03d}_fold{F}.pt      # Stage 3 per-(epoch, fold)
  stage1_complete.pt                    # written once at Stage 1 completion
  stage2_complete.pt                    # written once at Stage 2 completion
  stage3_complete.pt                    # written once at Stage 3 completion (final)
  last.pt                               # mirror of most recent save
  best.pt                               # best-by-headline-metric
```

### Checkpoint contents

```jsonc
{
  "stage": 1|2|3,
  "stage_completed": bool,
  "epoch": int,
  "fold": int|null,
  "global_step": int,
  "model": state_dict,
  "optimizer": state_dict,
  "scheduler": state_dict,
  "rng": { torch, cuda, numpy, python },
  "config": { full train() kwargs },
  "fold_plan": { k, seed, folds: [...] } | null,
  "stage_metrics": { best_val_map_50, best_epoch, final_metrics, wall_clock_seconds } | null
}
```

### Resume rules

`resume=True` → auto from `last.pt`. `resume="filename"` resolved against `out_dir`. `resume="/abs/path"` as-is.

| Loaded | Resumes at |
|---|---|
| `ckpt_s1_epoch003.pt` | Stage 1 epoch 4 |
| `ckpt_s2_epoch008.pt` (= `stage2_epochs`) | Stage 3 epoch 1 fold 0 |
| `ckpt_s3_epoch012_fold1.pt` | Stage 3 epoch 12 fold 2 |
| `ckpt_s3_epoch012_fold2.pt` (last fold) | Stage 3 epoch 13 fold 0 |
| `stage1_complete.pt` | Stage 2 epoch 1 (rebuild optimizer) |
| `stage2_complete.pt` | Stage 3 epoch 1 fold 0 (rebuild optimizer + folds) |
| `stage3_complete.pt` | training done; jump to evaluate() |

When crossing a stage boundary on resume, **model weights are kept**; optimizer + scheduler + param groups are rebuilt fresh from the new stage's config.

Disk hygiene: rolling `ckpt_*.pt` files capped at `keep_last_n` (default 6). `stage{1,2,3}_complete.pt`, `last.pt`, `best.pt` are never auto-deleted.

Resume granularity: between-folds (Stage 3) and between-epochs (Stages 1/2). Mid-fold resume is not supported.

## 4. Architecture (Siamese-preserved, transformer-grounded)

Backbone is shared between support and query streams (Siamese). Cross-attention is the only support↔query bridge. Channel widths chosen as multiples of 16 to align with MobileNetV3 channels and SIMD.

| Component | Sizing | Source / rationale |
|---|---|---|
| Backbone | MobileNetV3-Large (ImageNet pretrained), shared | siamese requirement; on-device ready |
| `DIM` | 160 | matches P5 width; 8/16-multiple |
| `M_TOKENS` (slot tokens) | 6 | "more slots than expected objects" (Slot Attention 2020) |
| `N_HEADS` | 8 | 160/8=20 per head |
| Slot-attention iterations | 2 | Locatello 2020 — >1 iter recommended |
| `SupportFusion` layers | 3, pre-LN, GELU, MLP ratio 4 | pre-LN Transformer (Xiong 2020) |
| `AttentionPool` | CaiT-style class-attention, 1 layer, 8 heads | Touvron 2021 |
| Cross-attention bridge | 2-layer DETR decoder, learnable softmax τ, DropPath 0.15 on all 6 residual branches | Carion 2020; DINO temperature trick |
| Query positional encoding | learnable (1, 160, 14, 14) + fixed 2D sinusoidal floor | DETR hybrid PE |
| `[ABSENT]` token | learnable (1, 1, 160) with token-dropout p=0.1 | DETR no-object class |
| Detection head — stride 16 (14×14) | 4× DWSep + GroupNorm; reg/conf/centerness branches; conf bias init = `-log((1-π)/π)`, π=0.01 | FCOS + GFL + RetinaNet |
| Detection head — stride 8 (28×28) | weight-shared with stride-16 head | FCOS multi-level |
| Reg branch | DFL distribution over 17 bins per coord (l, t, r, b in stride units) + integrated GIoU/CIoU | GFL (Li et al. 2020) |
| Centerness branch | 1 channel, sigmoid; FCOS centerness target | FCOS |
| `PresenceHead` | 320 → 160 → 64 → 1, GELU, LayerNorm-pre, dropout 0.2 | deeper MLP, harder to short-circuit |
| `decode_topk` | top-K cells (K=100) above τ_conf, NMS @ IoU=0.5, score = σ(conf)·σ(centerness)·σ(presence) | FCOS-X inference |

Total trainable params ≈ 6.8M. T4-friendly.

Dual-scale detection head shares weights between stride-8 and stride-16 outputs; top-K is taken across both grids' union (980 cells per image) before NMS.

Auxiliary detection head on the output of decoder layer 0 (DETR aux loss recipe) — same `total_loss`, weight 0.5.

## 5. Loss stack

Per-cell on positives (TaskAlignedAssigner-selected cells, see §6):
- **QFL** with target = `cell_iou(decoded, gt)` (detached). β=2.
- **DFL + GIoU** on the bin distribution (l, t, r, b in stride units) and integrated scalar.
- **Centerness BCE** with target `sqrt((min(l,r)/max(l,r)) · (min(t,b)/max(t,b)))`.

Per-cell on negatives:
- **`neg_qfl`**: focal BCE on `~pos_mask` cells with target=0, γ=2, weight 0.5.

Per-image:
- **Class-balanced presence focal-BCE**: `0.5 · pos_term + 0.5 · neg_term` (each normalised by its own count). γ=2.5.

Auxiliary:
- Aux head on decoder layer 0: total_loss × 0.5.
- **Conf-map entropy regulariser**: penalise high entropy on absent episodes / low entropy on multi-positive present episodes (sign-flipped). Weight 0.01.
- **Off-positive `reg` L2 prior**: weight 1e-3.
- **Attention bbox loss**: KL on aggregated slot-attention vs uniform-inside-bbox. Weight 0.5.
- **Prototype regularisation**: SupCon NT-Xent (per-shot), VICReg, Barlow Twins, optional triplet, prototype L2-norm prior.

All losses summed; coefficients are train() kwargs.

## 6. Sampling and curriculum

- **`SourceBalancedBatchSampler`** with default mix `{"vizwiz_base": 4, "vizwiz_novel": 2, "hots": 5, "insdet": 5}` (sums to 16). Within-source sampling is with-replacement so small pools work.
- **TaskAlignedAssigner (TOOD)**: per GT, pick top-q cells by `t = σ(conf)^α · IoU^β`, α=1, β=6, q = round(sum of top-10 IoU values). Conf-aware positive assignment — large mAP lever.
- **Center-sampling radius r=1.5**: positives are cells whose centre is within `r·stride` of the GT centre.
- **Multi-scale training** (Stage 2, Stage 3): input size sampled from {192, 224, 256} per batch.
- **2-scale TTA** at val/test: resize to 224 and 288, decode top-K from each, merge with NMS.
- **Adaptive hard-neg ramp** (Stage 2 epochs 4–8): if `val.iou_mean < 0.25` at end of epoch, hold ratio at previous value.

## 7. Dataset cleaner (`aggregator.py`)

- 80/20 train/test split, no val.
- `vizwiz_novel` → 100% train (only 16 instances).
- All other sources → 80/20 stratified.
- Stage directories `dataset/cleaned/{train,test}/`.
- Manifest carries only `train` and `test` splits in its `splits` index.
- `stats.json` schema mirrors the new layout.
- Negatives staging unchanged.

## 8. Per-epoch JSON analysis

Layout:

```
analysis/
  config.json
  folds.json                  # K-fold plan (Stage 3 only)
  stage1/
    epoch_001.json
    ...
    complete.json
  stage2/
    epoch_001.json
    ...
    complete.json
  stage3/
    epoch_001/
      fold_0.json
      fold_1.json
      fold_2.json
      aggregate.json
    ...
    complete.json
  summary.json
  test_report.json
```

Every per-epoch / per-fold JSON includes the kitchen-sink metric set:
- localisation: `iou_mean / median / p25 / p75`, `contain_mean / at_iou_50 / at_iou_75`
- mAP: `map_50`, `map_5095`, `ap_per_iou` (10 thresholds), `f1_50`, `precision_50`, `recall_50`
- presence: `presence_acc`, `_pos`, `_neg`, `mean_score_pos`, `mean_score_neg`, `presence_auroc`, `presence_pr_auc`, `presence_brier`
- collapse diagnostics: `frac_pred_near_corner`, `frac_pred_tiny_box`, `argmax_cell_entropy`, `conf_map_mean/std_pos/neg`, `support_proto_norm_mean/std`
- `per_source` block: same metrics per source

Stage 3 `aggregate.json` per epoch: recursive mean / min / max / std across the K fold JSONs.

`summary.json`: rolling best-by-metric pointer (epoch + value).

`stage{N}/complete.json`: mirror of in-checkpoint `stage_metrics`.

No PNGs during training. Plotting is offline via `plot_all_from_jsons(analysis_dir)`.

## 9. Notebook (`notebooks/modeling.ipynb`)

Cells 0–6 unchanged (Drive, repo, deps). Cells 7+ rewritten:

- 7: `resume_from(name_or_path)` helper accepting `last.pt`, `best.pt`, `stage{N}_complete.pt`, `ckpt_s{N}_epoch{E}_fold{F}.pt`, or absolute paths.
- 8: dataset build (80/20 only).
- 10: training overview markdown with the three-stage table.
- 11: `TRAIN_KWARGS` dict surfacing every default for editability.
- 12–13: `train(**TRAIN_KWARGS)` — single call.
- 14–15: resume-from-checkpoint demonstration cell.
- 16–18: post-training `evaluate()` on test split.
- 19–20: `plot_all_from_jsons(ANALYSIS_DIR)` for the offline report.

## 10. Tests

Per AGENTS.md scope:
- `aggregator_test.py`: 80/20 split, no val, stats schema.
- `modeling/loss_test.py`: class-balanced presence, DFL shape, neg_qfl on absent batch, entropy-reg sign.
- `modeling/model_test.py`: Siamese (single backbone), `decode_topk` NMS+gating, dual-scale outputs.
- `modeling/dataset_test.py`: source-balanced sampler mix, fold filter, augment determinism.
- `modeling/train_test.py`: cross-stage resume, fold reproducibility, stage-completion file written and survives hygiene pass.
- `modeling/evaluate_test.py`: every kitchen-sink key present, per-source breakdown sums.

## 11. Lint / format scope

Per AGENTS.md, only on touched files: `aggregator.py`, `modeling/{train,dataset,model,loss,evaluate,plot}.py` and matching `_test.py`. Notebook via `nbqa black`. No project-wide commands.

## 12. mAP unblock — Part L summary

- L.1 Top-K + NMS decode + 101-point AP (vs single-point AP)
- L.2 Centerness branch (FCOS) — score = σ(conf)·σ(centerness)·σ(presence)
- L.3 Multi-scale training + 2-scale TTA
- L.4 Stride-8 second detection head (small-object lever)
- L.5 DFL + GIoU regression
- L.6 Aux head on decoder layer 0
- L.7 TaskAlignedAssigner + center-sampling r=1.5
- L.8 Score-calibration scalar
- L.9 More val episodes (×3)
- L.10 Conf bias init, neg_qfl, class-balanced presence, off-pos reg L2 prior
- L.11 mAP-watch diagnostics in JSON
- L.12 `best.pt` headline = `val_map_50` (Stages 1/2) / `val_map_50_mean` (Stage 3)
- L.13 Adaptive hard-neg ramp gated by val.iou_mean

## 13. Build order

1. PLAN.md (this file).
2. `aggregator.py` (80/20 cleaner).
3. `modeling/model.py` (architecture overhaul, decode_topk).
4. `modeling/loss.py` (loss stack, TOOD assigner, DFL).
5. `modeling/dataset.py` (source-balanced sampler, set_fold, augment, multi-scale).
6. `modeling/evaluate.py` (top-K + TTA + per-source kitchen-sink metrics).
7. `modeling/train.py` (3-stage loop, resume, JSON output).
8. `modeling/plot.py` (offline plot_all_from_jsons).
9. `notebooks/modeling.ipynb` (full rewrite of cells 7+).
10. Tests for changed modules.
11. Targeted lint/format/test pass.
