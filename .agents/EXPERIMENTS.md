# Dyno Paper Experiment Registry

Use one stable alias per run. Training aliases may be referenced by offline
evaluation and diffusion child runs.

| Alias | Date | Status | Phase | Paper section | Kind | Parent | W&B | Local checkpoint or artifact | Launch/config | Notes |
|---|---|---|---|---|---|---|---|---|---|---|

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
