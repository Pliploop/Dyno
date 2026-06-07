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
- Added paper MuQ 1 Hz velocity configs for bottleneck dimensions 1, 4, 16,
  and 64, plus a trained content-only decoder config.
- Added focused tests for MSPF normalization, legacy behavior, constant
  trajectories, linear CKA, standardized Euclidean distance, and DTW.

## Launch Contract

Paper training configs intentionally leave `data.train_csv`, `data.val_csv`,
and `data.test_csv` unset. Supply the authoritative MTG-Jamendo manifests at
launch time; do not silently reuse validation as test data.

Example:

```text
python -m dyno.train experiment=paper_muq_1hz_velocity_d4 \
  data.train_csv=/path/to/train.csv \
  data.val_csv=/path/to/val.csv \
  data.test_csv=/path/to/test.csv
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
