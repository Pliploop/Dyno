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
- Standardized the default MSPF parameters at window 4, contrast exponent 3,
  sigma 10, and lambda `1e-3` across the API and paper callbacks.
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
- Paper FlipFlop evaluation draws 512 perturbations every five validation
  epochs to stabilize the severity scatter and correlation estimates. Legacy
  callback defaults remain unchanged.
- Added paper MuQ 1 Hz residual configs for bottleneck dimensions 1, 2, 4, 8,
  16, 32, 64, 128, and 256, plus a trained content-only decoder config.
- Residual prediction around the mean content token is the authoritative paper
  default. Velocity configs remain available as explicit ablations.
- Set the paper default bottleneck to 32 dimensions.
- Set the paper encoder to width 768, 12 heads, and 6 layers, and the smaller
  predictor to width 512, 8 heads, and 4 layers.
- Verified all 55,701 MuQ 1 Hz arrays against the complete MuQ 0.1 Hz
  extraction. Every file has a readable 2D float32 header with width 512 and
  the track-ID sets match exactly.
- Audited the original 32,859/11,101/11,565 split and a random-track
  51,701/2,000/2,000 candidate before selecting the paper split.
- Materialized a versioned,
  deterministic 51,701/2,000/2,000 split using all extracted tracks. The
  paper split is disjoint by artist and album and uses seed 142. Historical
  manifests remain untouched.
- Added focused tests for MSPF normalization, legacy behavior, constant
  trajectories, linear CKA, standardized Euclidean distance, and DTW.
- Added the no-anchor ablation as direct absolute-embedding reconstruction
  conditioned only on the temporal token. No content token is passed to the
  predictor or added to its output.
- Added explicit deterministic AE (`beta=0`) and VAE (`beta=0.01`) paper
  configs. The canonical paper config remains the VAE protocol.
- Validated all 55,701 MERT 1 Hz arrays at width 1024 and all 55,701
  Music2Latent 1 Hz arrays at width 64. Both are readable float32 arrays with
  exact track-ID parity against MuQ, and both use translated copies of the
  authoritative artist/album-disjoint paper split.
- Added launchable MERT and Music2Latent encoder-ablation configs. MATPAC and
  USAD remain unavailable because no complete 1 Hz extraction exists.
- Added a shared experiment resolver for registry aliases, W&B run IDs/URLs,
  direct checkpoint files, and checkpoint directories. Hydra runs accept the
  same references through `run_ref`.

## Launch Contract

The MuQ 1 Hz paper config uses the checked MTG-Jamendo manifests under
`/gpfs/scratch/acw749/datasets/dyno/mtg-jamendo/muq/1hz/manifests`.
Regenerate and validate them with `scripts/prepare_embedding_manifests.py`
after any extraction change. Do not silently reuse validation as test data.

Example:

```text
python -m dyno.train experiment=paper_muq_1hz_residuals_d32
python -m dyno.train experiment=paper_muq_1hz_no_anchor
python -m dyno.train experiment=paper_mert_1hz
python -m dyno.train experiment=paper_music2latent_1hz
python -m dyno.train experiment=paper_muq_1hz_ae
```

## Remaining Phase 2 Work

- Extract and validate MATPAC and USAD at the authoritative paper rate before
  adding their encoder-ablation configs.

Audio-domain perturbations, cross-encoder MSPF scoring, retrieval artifacts,
and structure probing remain Phase 3 work.
