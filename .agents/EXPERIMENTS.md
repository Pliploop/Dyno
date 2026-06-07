# Dyno Paper Experiment Registry

Use one stable alias per run. Training aliases may be referenced by offline
evaluation and diffusion child runs.

| Alias | Date | Status | Phase | Paper section | Kind | Parent | W&B | Local checkpoint or artifact | Launch/config | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| muq-1hz-residuals-d32-seed0 | 2026-06-07 | running | 2 | Main training | training | - | [W&B](https://wandb.ai/jul-guinot/dyno-paper/runs/0e33a658_1780839236) | `/gpfs/scratch/acw749/dyno/logs/xps/0e33a658_1780839236/checkpoints` | `sbatch train.sh` (`experiment=paper_muq_1hz`, `seed=0`) | Slurm 11872287; centered residuals despite historical `DynoVelocityPredictor` class name. |


## Status Values

- `planned`: specified but not launched
- `running`: launched and active
- `completed`: run finished successfully
- `failed`: run terminated unsuccessfully
- `stopped`: intentionally stopped
- `unknown`: launch was reported without current status

## Update Rules

- Append a row when a run is first reported.
- Update the existing row when its status, W&B URL, checkpoint, or notes change.
- Never reuse an alias for a different configuration.
- Use `pending` for information the user has not supplied.
- Record child evaluations separately and set `Parent` to the source training
  alias.
