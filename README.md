# ISS Group 24

Two-model few-shot system:
- **Localizer** (`localizer/`) — multi-shot OWLv2 + learned support fusion. Predicts a bounding box. Trained positive-only. Headline metric: mAP@50.
- **Siamese**  (`siamese/`)   — multi-shot DINOv2-small + cross-attention head. Predicts existence_prob ∈ [0, 1]. Trained on mixed pos+neg with focal-BCE + variance + decorrelation regularisers. Headline metrics: AUROC, FPR (lower is better).

The two models are fully independent — train, evaluate, save, and run inference on either alone. Combined cascaded inference (siamese → localizer with adjustable threshold) is provided in `inference_combined.py`.

## Quick start

```bash
# 1. Install deps (already pinned in pyproject.toml / requirements.txt).
uv sync

# 2. Build the staged dataset (idempotent).
uv run python aggregator.py

# 3. Smoke test (~60s end-to-end on RTX 2060).
uv run python -m shared.smoke --seconds-budget 60

# 4. Open modeling.ipynb and run cells top-to-bottom, or:
uv run python -c "from localizer.train import train_phase0, train_stage_L1, train_stage_L2, train_stage_L3; \
                  train_phase0(); train_stage_L1(); train_stage_L2(); train_stage_L3()"
```

## Datasets

- VizWiz: [VizWiz Fewshot](https://vizwiz.org/tasks-and-datasets/object-localization) — used for localizer Phase 0 only.
- HOTS:   [Household Objects in Tabletop Scenarios](https://github.com/gtziafas/HOTS) — train + test.
- InsDet: [Instance Detection](https://insdet.github.io) — train + test.

## Layout

See `PLAN.md` for the full architecture, hyperparameter surface, and stage-by-stage training spec.
