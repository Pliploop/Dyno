# Phase 3: Offline Evaluation Status

## Implemented

- Added deterministic, artist-grouped 8-fold assignments for the 326 SALAMI
  and 424 Harmonix tracks with complete local MuQ 2 Hz fixtures. No artist
  group crosses folds. Each rotation uses six folds for training, one for
  validation, and one for testing.
- Added a full-track attention structure-probing protocol inspired by Toyama
  et al.:
  - frozen representations;
  - full-track features at the evaluated checkpoint's native encoder rate;
  - one Transformer layer with joint boundary BCE and function CE;
  - a native full-sequence baseline;
  - token-only probes that start from learned mask tokens plus position and
    use adaLN conditioning from content, temporal, or both global tokens;
  - padded batch positions are attention-masked, loss-masked, and removed
    before full-track metrics;
  - AdamW at `1e-4`, weight decay `0.01`;
  - five warmup epochs and 95 cosine-decay epochs;
  - validation checkpoint selection and validation-tuned peak threshold;
  - segment-averaged function probabilities;
  - HR.5F, HR3F, and PWF through `mir_eval` defaults, plus frame ACC,
    with fold mean and standard deviation.
- Added seven directly comparable probe inputs: full local sequence; content,
  temporal, or both global tokens alone; and the local sequence conditioned
  on content, temporal, or both global tokens.
- SALAMI uses first-annotator functions and uppercase/coarse boundaries as the
  primary result. Lowercase/fine boundaries are logged separately under
  `salami_fine`. Harmonix functions are collapsed to the reference
  seven-function vocabulary.
- The checkpoint's saved encoder and rate select the structure fixtures.
  Frozen content and temporal tokens are extracted once from that exact
  full-track feature sequence, with no probe-time resampling. Every probe
  predicts the complete track in one pass.
- Added both standalone Hydra execution and an opt-in Lightning callback. The
  callback runs directly on `on_test_epoch_end` for
  `train=false test=true run_ref=<reference>`, and can also run on
  `on_train_end` using the best validation checkpoint.
- Added offline temporal retrieval artifacts with exact query and neighbor
  track IDs, ranks, representation distances, and MSPF-DTW distances for
  content, temporal, and combined retrieval.
- Added cross-encoder MSPF geometry for aligned MuQ, MERT, and Music2Latent
  structure fixtures. Pair sampling is deterministic and bounded.
- Added mixed-domain perturbation evaluation. Gain, pitch shift, and time
  stretch run in the audio domain; chunk shuffle, reversal, and section
  deletion run directly on the frozen embedding sequence. Each result records
  its domain and logs content and temporal displacement, MSPF-DTW, and
  preserving/disrupting separation.

## Launch Contract

All commands accept an experiment-registry alias, W&B run ID/URL recorded in
the registry, a checkpoint file, or a checkpoint directory.

```text
python -m dyno.evaluate_structure_probe run_ref=<reference>
python -m dyno.evaluate_offline_temporal run_ref=<reference>
python -m dyno.evaluate_audio_perturbation run_ref=<reference>
```

Fast smoke overrides:

```text
python -m dyno.evaluate_structure_probe run_ref=<reference> \
  probe.max_tracks=16 probe.num_folds=2 probe.epochs=2 \
  probe.probe_inputs=[local,temporal] wandb.enabled=false
```

To run structure probing through the normal Lightning test path:

```text
python -m dyno.train experiment=paper_muq_1hz \
  callbacks=paper_structure_probe train=false test=true run_ref=<reference>
```

## Scientific Boundary

Cross-encoder MSPF geometry is well-defined because every encoder observes the
same audio track. Cross-encoder reconstruction MSPF R2 is not currently
reported: a trajectory reconstructed in MuQ embedding space cannot be scored
as a MERT or Music2Latent trajectory without an embedding-to-audio generator
or a learned cross-encoder translator. Phase 4's audio generator can make that
measurement well-defined; Phase 3 must not fabricate it by comparing
incompatible embedding spaces.

The optional unsupervised boundary diagnostic from Marmoret et al. remains
separate from the required supervised probe. It is not needed for the active
paper table and has not been made a launch prerequisite.
