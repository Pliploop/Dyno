# Dyno Paper Experiment Workflow

This file is the durable operating contract for the Dyno paper refactor.

## Sources Of Truth

- `latex/paper.tex` defines the experimental claims, sections, table layouts,
  and required deliverables.
- `latex/results.tex` is the canonical results ledger. Results are entered
  there with provenance before they are copied into the paper.
- `.agents/EXPERIMENTS.md` is the canonical experiment registry. Every
  launched training, offline evaluation, probe, or diffusion run receives a
  stable alias and records its W&B URL and local checkpoint or artifact path.
- The W&B project for paper experiments is `dyno-paper`, under entity
  `jul-guinot`.

## Experiment Logging

When the user reports a launched experiment, append or update one row in
`.agents/EXPERIMENTS.md`. Record:

- stable alias;
- launch date and status;
- phase and paper section;
- run kind and parent training alias, when applicable;
- W&B URL;
- local checkpoint or artifact directory;
- exact launch command or config overrides when available;
- short notes needed to interpret or reproduce the run.

Do not invent missing links or paths. Mark unknown fields as `pending`.
Offline evaluations and diffusion runs must explicitly reference the training
alias and checkpoint from which their frozen representations were obtained.

The preferred offline interface accepts a ledger alias or W&B run name and
resolves both W&B metadata and the local checkpoint. A direct checkpoint path
must remain supported.

## Results Ingestion

When asked to ingest metrics:

1. Read the experiment registry and the referenced local logs, W&B run, or
   explicit user-provided values.
2. Enter only observed results into `latex/results.tex`.
3. Preserve the source run alias, checkpoint, split, dataset, seed or fold
   aggregation, and metric definition needed to audit the value.
4. Do not analyze, select, average, or reinterpret results unless explicitly
   requested. If aggregation is required by the table, use the protocol fixed
   by the relevant paper experiment and record it.
5. Leave missing values as `\res`; never infer them.

When asked to ingest results into the paper, copy the corresponding recorded
values from `latex/results.tex` into `latex/paper.tex`. Do not add analysis or
change the experimental narrative unless separately requested.

## Metric Naming

Paper runs use hierarchical W&B keys:

`paper.<paper_section>/<split>/<metric>`

Use lowercase snake_case for section and metric identifiers. Additional
dataset or variant hierarchy follows the split when needed, for example:

`paper.structure_probing/test/salami/hr_0p5_f`

The first slash therefore keeps all metrics for a paper subsection together
in W&B. Artifact and figure keys follow the same section prefix.

## Experimental Configuration

- Preserve existing general-purpose defaults for backward compatibility.
- Add explicit Hydra experiment configs for paper runs under a clear
  paper-specific namespace.
- Paper configs, not global defaults, encode the paper's authoritative model,
  data, callback, checkpoint, and early-stopping settings.
- New paper training runs enable checkpointing and early stopping on validation
  loss. The checkpoint policy must retain the best validation-loss checkpoint
  and the last checkpoint.
- Paper training runs use the full validation loader, monitor `val/loss`, use
  an early-stopping patience of 10 validation checks, and set
  `trainer.max_steps=200000`.
- Do not create a separate fixed-wall-clock or fixed-update comparison
  protocol. Paper variants share the 200,000-step ceiling and may stop earlier
  according to the same validation-loss rule.
- Keep the implementation in the existing Lightning + Hydra framework.
- Prefer independently runnable offline evaluation configs that can also be
  attached to end-of-training test execution.

## MSPF Normalization

- Paper MSPFs use normalized time and normalized values by default.
- Normalized time means interpolation onto a fixed grid over `[0, 1]`.
- Normalized values means per-curve min-max scaling to `[0, 1]`, with a
  constant curve mapped deterministically to zeros.
- Expose this behavior through one toggle, `normalize=true`, in the shared
  MSPF API/config. Setting it to false preserves native temporal resolution
  and raw MSPF values.
- Keep compatibility for existing callers of the current `absolute` option
  while migrating paper callbacks to the explicit normalization toggle.

## Structure Probing Protocol

- SALAMI is the primary structure-probing benchmark.
- Harmonix is also required as a secondary benchmark.
- Reproduce the linear-probing protocol from Toyama et al.,
  arXiv:2512.17209v2: 2 Hz labels, 30-second windows, a single linear layer,
  joint boundary/function training, HR.5F, HR3F, PWF, and ACC.
- Use dataset-appropriate folds without mixing tracks across train,
  validation, and test. Harmonix uses the reported 8-fold 6/1/1 protocol.
- For SALAMI, report coarse/uppercase boundaries as the primary HR.5F/HR3F
  result and fine/lowercase boundaries as an additional diagnostic.
- Use SALAMI's native semantic function vocabulary for PWF/ACC. Do not map it
  to the Harmonix vocabulary; comparisons of interest are between probe inputs
  such as content and temporal tokens within each dataset.
- Use the first SALAMI annotation for maximum coverage. Do not add a separate
  double-annotator agreement evaluation.
- SALAMI has no official packaged split, so create and check in a deterministic
  grouped 8-fold manifest. Keep each track wholly within one fold and use the
  same 6/1/1 train/validation/test rotation as Harmonix.
- The unsupervised boundary protocol from Marmoret,
  arXiv:2603.27218v1, is a separate optional diagnostic and does not replace
  the required frame-level function probe.

## Phases

1. Identification: map each paper deliverable to an exact existing callback,
   adjacent proxy, missing implementation, fixture, and W&B key. Confirm each
   mapping with the user before changing behavior.
2. Online implementation: implement missing training-time metrics and rename
   logging under the paper hierarchy.
3. Offline metrics: implement test-only structure probing and other expensive
   evaluations as standalone Lightning/Hydra workflows tied to checkpoints.
4. Diffusion: implement frozen-token extraction, conditional trajectory
   diffusion or flow matching, token swapping, and distributional evaluation,
   with explicit provenance back to the encoder checkpoint.

## Change Discipline

- Keep changes backward compatible and scoped.
- Do not overwrite user changes or historical experiment evidence.
- Confirm ambiguous metric definitions, dataset splits, or model choices
  before launching or encoding them as authoritative.
- Commit meaningful milestones so the pre-paper baseline and each phase remain
  recoverable.
