# Phase 2: Online Implementation Status

## Completed

- Added the `dyno-paper` W&B logger under entity `jul-guinot`.
- Added a paper trainer with:
  - full validation;
  - a 200,000-step ceiling;
  - best and last checkpoints monitored by `val/loss`;
  - early stopping after 10 validation checks without improvement.
- Added normalized MSPF behavior:
  - normalized time and values by default;
  - one `normalize` toggle;
  - compatibility for explicit legacy `absolute` callers;
  - deterministic zero output for constant semantic trajectories.
- Added streamed full-split reconstruction metrics and the content-mean
  baseline without retaining the full validation set in memory.
- Added paper metric names for:
  - trajectory reconstruction;
  - MSPF geometry using MSPF-DTW;
  - exact order sensitivity;
  - linear CKA and CKNNA dependence references;
  - chunk-shuffle displacement and severity correlation.
- Kept the original FlipFlop shuffle-severity flow intact for paper runs,
  including its sampling procedure, cosine and L2 diagnostics, FlipFlop score,
  normalized extremeness, W&B table, and Plotly scatter. Paper mode adds
  Spearman summaries and changes only the enclosing W&B section namespace.
- Added paper MuQ 1 Hz velocity configs for bottleneck dimensions 1, 2, 4, 8,
  16, 32, 64, 128, and 256, plus a trained content-only decoder config.
- Set the paper default bottleneck to 32 dimensions.
- Set the paper encoder to width 768, 12 heads, and 6 layers, and the smaller
  predictor to width 512, 8 heads, and 4 layers.
- Verified all 55,701 MuQ 1 Hz arrays against the complete MuQ 0.1 Hz
  extraction. Every file has a readable 2D float32 header with width 512 and
  the track-ID sets match exactly.
- Materialized the established split at 1 Hz: 32,859 train, 11,101 validation,
  and 11,565 test tracks. The source split leaves 176 extracted tracks
  unassigned, and the 1 Hz manifests preserve that decision.
- Added focused tests for MSPF normalization, legacy behavior, constant
  trajectories, linear CKA, standardized Euclidean distance, and DTW.

## Launch Contract

The MuQ 1 Hz paper config uses the checked MTG-Jamendo manifests under
`/gpfs/scratch/acw749/datasets/dyno/mtg-jamendo/muq/1hz/manifests`.
Regenerate and validate them with `scripts/prepare_embedding_manifests.py`
after any extraction change. Do not silently reuse validation as test data.

Example:

```text
python -m dyno.train experiment=paper_muq_1hz_velocity_d32
```

## Remaining Phase 2 Work

- Confirm and encode the no-anchor model definition before adding that
  ablation config.
- Add paper configs for the non-MuQ encoder sweep once each authoritative
  embedding rate, width, and manifest path is fixed.
- Decide whether online paper callbacks should run every validation epoch or
  at a wider interval for production runs. Offline best-checkpoint evaluation
  remains authoritative for final tables.
- Add a standalone alias/checkpoint resolver, shared with Phase 3 evaluation.

Audio-domain perturbations, cross-encoder MSPF scoring, retrieval artifacts,
and structure probing remain Phase 3 work.
