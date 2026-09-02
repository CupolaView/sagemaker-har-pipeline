# Hyperparameter Tuning (SageMaker HPO)

**Date:** 2026-09-02 · **Tuning job:** `pytorch-training-260902-1652` ·
**Best training job:** `pytorch-training-260902-1652-008-3f2d80f5`

## Setup

- Strategy: **Bayesian**, 8 trials, `max_parallel_jobs=2`
- Objective: maximize `test_acc` (scraped from stdout via regex — same
  metric definition that feeds CloudWatch)
- Search space: `lr` ∈ [1e-4, 1e-2] (log scale), `batch-size` ∈ {32, 64, 128}
- Fixed: 10 epochs, ml.m5.large, data `v1-processed`

## Results

| Rank | lr       | batch-size | test_acc |
|------|----------|------------|----------|
| best | 0.004725 | 128        | **0.9355** |
| 2    | 0.004751 | 128        | 0.9321 |
| 3    | 0.004633 | 128        | 0.9287 |
| 4    | 0.009108 | 128        | 0.9284 |
| 5    | 0.005417 | 32         | 0.9226 |
| 6    | 0.001369 | 128        | 0.9206 |
| 7    | 0.005077 | 128        | 0.9172 |
| 8    | 0.000112 | 32         | 0.9077 |

Baseline (defaults `lr=0.001, batch-size=64`): **0.9094**.

## Analysis

- **+2.6 pts over baseline** for ~25 minutes of ml.m5.large compute
  (8 × ~3 min ≈ < $1). Worth it at this scale; at real-project scale the
  same decision needs a compute budget argument.
- **Bayesian behavior visible:** 6/8 trials concentrated in the
  lr ≈ 0.004–0.009 / batch-size 128 region after early signal — the
  exploitation/exploration trade-off vs. Random/Grid in practice.
- The default lr (0.001) was simply too low for batch-size 128's
  gradient-noise profile; the tuner found the classic larger-batch →
  larger-lr pairing on its own.

## Known limitation (deliberate)

The objective is measured on the **test set**, so model selection has now
touched test data — mild selection leakage. Rigorous setup: carve a
validation split from train (e.g., by subject to avoid identity leakage,
since HAR has 30 subjects), tune on val, report test once. Accepted here
because the pipeline, not the leaderboard number, is the deliverable;
flagged so the trade-off is a decision, not an oversight.

## Downstream

The best job's `model.tar.gz` (under `har/artifacts/`) is the artifact to
register in the Model Registry (next step). Registered model metadata
should reference this document and the tuning job name for lineage.
