# Phase 1: Paper Deliverable Audit

Status meanings:

- `exact-core`: the required computation exists, but logging scope or naming
  may still need adjustment.
- `proxy`: related code exists, but its statistic or protocol does not match
  the paper.
- `fixture-only`: data preparation or visualization exists without the
  required metric.
- `missing`: the paper deliverable is not implemented.
- `aggregate`: assembled across multiple runs rather than computed by one
  callback.

No behavior has been changed during this audit.

## Global Infrastructure

| Deliverable | Existing support | Status | Phase 2 action |
|---|---|---:|---|
| Paper W&B project | Logger targets `Dyno` | missing | Add paper logger config targeting `dyno-paper`. |
| Paper metric hierarchy | Human-readable callback suite names | proxy | Use `paper.<section>/<split>/<metric>` keys in paper callbacks/configs. |
| Best checkpoint | Generic `ModelCheckpoint` config exists, but monitors obsolete `loss/val/total_loss/dataloader_idx_0` | proxy | Paper callback monitors `val/loss`, saves best and last. |
| Early stopping | Generic callback exists with unresolved monitor | proxy | Paper callback monitors `val/loss`; patience is a paper-config setting. |
| Test after training | `train.py` can test the best checkpoint | exact-core | Enable only in paper configs that have a real test CSV and test-safe callbacks. |
| Standalone checkpoint evaluation | `train=false`, `test=true`, and `ckpt_path` are possible manually | proxy | Add a dedicated paper evaluation entry/config and alias resolver. |
| Stable run provenance | Dora output and W&B IDs exist | proxy | Connect ledger alias, W&B run, Dora folder, and checkpoint explicitly. |

Important infrastructure findings:

- Training logs the relevant monitor as `val/loss`.
- Default and current experiment configs disable checkpointing.
- The default trainer evaluates only 20 validation batches.
- When `data.test_csv` is absent, test silently reuses the validation dataset.
- The two current MuQ experiment configs use 0.1 Hz, 30 frames, latent sizes
  32/128, and configurations that do not represent the paper defaults.

Confirmed paper training policy:

- evaluate the full validation loader;
- checkpoint and early-stop on `val/loss`;
- use an early-stopping patience of 10 validation checks;
- cap every paper training run at 200,000 steps;
- do not run a separate fixed-wall-clock or fixed-update comparison.

Confirmed paper MSPF policy:

- normalize both time and values to `[0, 1]` by default;
- expose one `normalize=true` toggle;
- retain compatibility with the existing `absolute` API for non-paper callers.

The current `compute_mspf(..., absolute=False)` normalizes only the time axis.
Several callbacks apply value min-max scaling independently, while others use
raw values. Phase 2 must centralize this behavior so paper metrics cannot mix
normalization conventions.

## 4.2 Trajectory Reconstruction

Paper outputs:

- embedding trajectory MSE;
- MSPF reconstruction `R^2`;
- content-mean baseline;
- separately trained content-only decoder baseline;
- temporal-token models for each bottleneck dimension.

Existing support:

- `MSPFCallback` computes embedding MSE and MSPF `R^2`.
- `LightningDyno` computes reconstruction diagnostics over every batch, but
  logs embedding L1 rather than the paper MSE.
- The callback samples only four tracks from the first batch and has no test
  hooks.
- Neither paper baseline is implemented as an evaluation row/config.

Classification: `exact-core` for the two metrics; `missing` for baselines and
held-out aggregation.

Proposed keys:

```text
paper.trajectory_reconstruction/<split>/embedding_mse
paper.trajectory_reconstruction/<split>/mspf_r2
paper.trajectory_reconstruction/<split>/mspf_rmse
paper.trajectory_reconstruction/<split>/curves
```

Required Phase 2 work:

- aggregate metrics over the full selected split at track level;
- support validation and test;
- add content-mean and trained content-only baseline configs;
- keep model dimension/variant in W&B config and tags, not metric names.

## 4.3 Annotation-Free Validation Of MSPF

### Controlled Transformation Validation

Paper output: MSPF DTW for covers, pitch shift, time stretch, gain, chunk
shuffle, reverse, and section deletion.

Existing support:

- MSPF itself is implemented.
- Embedding-space shuffle, reverse, and splice transformations exist.
- Current comparison uses min-max-normalized MSPF `R^2`, not DTW.
- No DTW helper exists.
- No audio-domain pitch, stretch, gain, cover-pair, or section-deletion
  evaluation exists.

Classification: `proxy` for embedding transforms; otherwise `missing`.

Proposed keys:

```text
paper.mspf_validation/<split>/mspf_dtw/<condition>
paper.mspf_validation/<split>/curves/<condition>
```

This should be offline or test-only because preserving transformations must be
applied to audio before feature extraction.

### Representation Geometry

Paper output: Spearman agreement between representation distance and MSPF-DTW
distance for random, content, MSPF feature, temporal, and combined baselines.

Existing support:

- `AnnotationFreeTemporalCallback` computes pairwise Spearman agreement for
  all required representation families.
- It uses normalized-curve Euclidean distance instead of MSPF-DTW.
- It uses centered cosine distance for every representation. The paper
  specifies cosine for content and standardized Euclidean for temporal tokens.

Classification: `proxy`.

Proposed keys:

```text
paper.mspf_geometry/<split>/spearman/<representation>
paper.mspf_geometry/<split>/partial_spearman/<representation>
```

### Cross-Encoder MSPF

Paper output: MuQ temporal tokens scored using MuQ, MERT, USAD, and
Music2Latent MSPFs.

Existing support:

- feature extractors and structure feature layouts exist for these encoders;
- no callback joins the same held-out tracks across encoder spaces;
- no independent-encoder reconstruction MSPF `R^2` or perturbation separation
  is computed.

Classification: `fixture-only`.

Proposed keys:

```text
paper.mspf_cross_encoder/<split>/<scoring_encoder>/mspf_r2
paper.mspf_cross_encoder/<split>/<scoring_encoder>/geometry_spearman
paper.mspf_cross_encoder/<split>/<scoring_encoder>/perturbation_separation
```

## 4.4 Temporal Retrieval Case Studies

Paper output: qualitative nearest neighbours under content, temporal, and
combined representations for selected temporal forms.

Existing support:

- `CoverXCallback` performs quantitative annotation-form retrieval.
- structure visualization and general retrieval applications exist.
- no reproducible paper case-study selector, neighbour table, or MSPF
  comparison artifact exists.

Classification: `fixture-only`.

Proposed keys:

```text
paper.temporal_retrieval/<split>/cases
paper.temporal_retrieval/<split>/mspf_comparison
```

The artifact should record track IDs and distances so the paper table can be
reproduced without relying on free-text descriptions stored only in W&B.

## 4.5 Sensitivity To Temporal Perturbations

Paper outputs:

- normalized displacement for pitch, stretch, gain, and disruptive edits;
- preserving/disrupting separation ratio;
- Spearman correlation between shuffle severity and displacement.

Existing support:

- exact embedding permutations and normalized temporal displacement exist;
- `FlipFlopCallback` records custom chunk-swap severity and displacement
  samples;
- no audio-domain preserving transforms exist;
- no section deletion exists;
- no preserving/disrupting separation ratio is computed;
- no shuffle-severity Spearman value is logged.

Classification: `proxy`.

Proposed keys:

```text
paper.perturbation_sensitivity/<split>/<representation>/displacement/<condition>
paper.perturbation_sensitivity/<split>/<representation>/separation_ratio
paper.perturbation_sensitivity/<split>/<representation>/shuffle_severity_spearman
paper.perturbation_sensitivity/<split>/<representation>/shuffle_severity_samples
```

Audio-domain perturbations belong in test-only/offline evaluation. Exact
embedding permutations can run online on a bounded validation sample.

## 4.6 Structure Probing

Paper outputs: HR.5F, HR3F, PWF, and ACC for six frame-level probe inputs on
SALAMI and Harmonix.

Existing support:

- structure annotation parsing, feature manifests, and frozen Dyno token
  extraction exist;
- `CoverXCallback` probes track-level section counts and form attributes, not
  frame-level boundaries/functions;
- manifests do not contain folds or train/validation/test assignments;
- there is no 2 Hz, 30-second probe dataset;
- there is no joint linear boundary/function head, peak picking, segment
  function post-processing, `mir_eval` integration, or fold aggregation.

Classification: `fixture-only`; the required probe is `missing`.

Proposed keys:

```text
paper.structure_probing/<split>/<dataset>/<probe_input>/hr_0p5_f
paper.structure_probing/<split>/<dataset>/<probe_input>/hr_3_f
paper.structure_probing/<split>/<dataset>/<probe_input>/pwf
paper.structure_probing/<split>/<dataset>/<probe_input>/accuracy
paper.structure_probing/<split>/<dataset>/<probe_input>/<metric>_std
```

This is Phase 3 test-only work. Harmonix reproduces the referenced 8-fold
6/1/1 protocol. SALAMI uses a checked-in deterministic grouped 8-fold
manifest with the same 6/1/1 rotation.

Available SALAMI annotation choices:

- uppercase/coarse: large-scale section boundaries and recurring section
  identities;
- lowercase/fine: phrase-scale boundaries and recurring phrase identities;
- functions: free-form semantic labels such as verse, chorus, transition,
  interlude, solo, silence, and dataset-specific musical-form terms.

The local release has first-annotator versions of all three layers for 1,348
tracks and second-annotator versions for 895 tracks. It contains no official
split or fold file.

Confirmed SALAMI protocol:

- coarse/uppercase boundaries are the primary HR.5F/HR3F result;
- fine/lowercase boundaries are an additional diagnostic;
- PWF/ACC use the native SALAMI function vocabulary without mapping to
  Harmonix labels;
- use the first annotation for maximum coverage;
- do not add a second-annotator agreement evaluation;
- generate and check in deterministic grouped 8-fold assignments, with tracks
  kept intact and a 6/1/1 train/validation/test rotation;
- compare content, temporal, and other probe inputs within each dataset rather
  than treating cross-dataset function-label scores as directly comparable.

## 4.7 Content--Temporal Disentanglement

### Exact Order Sensitivity

Paper output: raw content and temporal L2 displacement for random shuffle,
reverse, half-swap, circular shift, and local shuffle.

Existing support:

- all five exact sequence transformations exist;
- temporal raw L2 is computed but only normalized L2 is logged;
- content cosine distance is logged, not raw content L2;
- deterministic posterior means are correctly used when the module is in eval
  mode.

Classification: `exact-core`.

Proposed keys:

```text
paper.order_sensitivity/<split>/<transform>/content_l2
paper.order_sensitivity/<split>/<transform>/temporal_l2
paper.order_sensitivity/<split>/<transform>/temporal_l2_normalized
```

### Representation Dependence

Paper output: CKA and CKNNA for same-content, shuffled-content, default,
no-anchor, and VAE comparisons.

Existing support:

- CKNNA is implemented for the default paired temporal/content tokens;
- a mutual-information estimate is also available but is not in the active
  table;
- linear CKA is not implemented;
- same-representation and shuffled-pair references are not logged;
- no-anchor and VAE rows require results from separate training configs.

Classification: `proxy` for default CKNNA; otherwise `missing`/`aggregate`.

Proposed keys:

```text
paper.representation_dependence/<split>/<comparison>/linear_cka
paper.representation_dependence/<split>/<comparison>/cknna
```

## 4.8 Generative Conditioning And Token Swapping

Paper outputs: content and temporal alignment, best/mean trajectory error,
sample diversity, and minimum/mean MSPF error over stochastic samples.

Existing support:

- current latent swapping uses the deterministic reconstruction decoder and
  logs only two source-classification accuracies;
- there is no trajectory diffusion or flow-matching implementation in
  `dyno/models/ldm`; only orphaned JSON configs remain;
- the legacy `test_config.yaml` references modules that are not present.

Classification: `missing`.

Proposed keys:

```text
paper.diffusion_swapping/<split>/<condition>/content_alignment
paper.diffusion_swapping/<split>/<condition>/temporal_alignment
paper.diffusion_swapping/<split>/<condition>/best_error
paper.diffusion_swapping/<split>/<condition>/mean_error
paper.diffusion_swapping/<split>/<condition>/diversity
paper.diffusion_swapping/<split>/<condition>/mspf_min_error
paper.diffusion_swapping/<split>/<condition>/mspf_mean_error
```

Implementation is reserved for Phase 4.

## 4.9 Compression Summary

Paper output: compression ratio, reconstruction MSE, MSPF `R^2`,
perturbation separation, and content CKA for each bottleneck dimension.

Existing support:

- latent dimension is configurable;
- reconstruction/MSPF and perturbation/dependence have partial support as
  described above;
- compression ratio is deterministic metadata;
- no paper sweep configs or cross-run result assembler exist;
- the current paper/results skeleton still has fixed-update and
  fixed-wall-clock columns, but the confirmed run policy supersedes them.

Classification: `aggregate`.

Proposed W&B run metadata:

```text
paper/bottleneck_dim
paper/compression_ratio
paper/max_steps
paper/early_stopping_patience
```

The table consumes the section metrics from the corresponding run aliases
rather than introducing duplicate scalar names. Remove or replace the obsolete
compute-matched columns when the paper table is next revised.

## 4.10 Encoder Receptive Field

Paper output: MSPF `R^2`, perturbation separation, and CKA across MuQ, MERT,
USAD, Music2Latent, and MATPAC.

Existing support:

- extraction configs and encoder wrappers exist for every listed encoder;
- no paper training sweep configs or standardized held-out alignment exist;
- metrics depend on Phase 2 corrections above;
- receptive-field metadata is currently prose/config knowledge rather than
  structured run metadata.

Classification: `fixture-only` plus `aggregate`.

Proposed W&B run metadata:

```text
paper/embedding_encoder
paper/embedding_rate_hz
paper/encoder_receptive_field_seconds
```

## Phase 2 Boundary

Phase 2 should implement and standardize:

1. paper logger, paper callback bundle, checkpointing, and early stopping;
2. full-split reconstruction metrics and baselines;
3. exact order-sensitivity logging;
4. CKA and dependence references;
5. corrected MSPF geometry using the paper distance definitions;
6. online-compatible shuffle severity and perturbation summaries;
7. paper experiment configs for model variants, compression dimensions, and
   encoder sweeps.

Phase 3 retains audio-domain perturbations, cross-encoder scoring, retrieval
artifacts, and frame-level structure probing. Phase 4 retains diffusion.

## Phase 1 Status

All protocol decisions needed to begin Phase 2 are confirmed. Phase 3 must
materialize and version the SALAMI fold manifest before structure-probe runs
are launched.

Phase 2 implementation progress is tracked in `.agents/PHASE2_STATUS.md`.
