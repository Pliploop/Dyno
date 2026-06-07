"""Resolve a paper run alias, W&B run identifier, or path to a checkpoint."""

from __future__ import annotations

import argparse
import json

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dyno.utils.experiment_registry import resolve_experiment_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--registry", default=".agents/EXPERIMENTS.md")
    parser.add_argument("--checkpoint", choices=("best", "last", "newest"), default="best")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    resolved = resolve_experiment_reference(
        args.reference,
        registry_path=args.registry,
        checkpoint_preference=args.checkpoint,
    )
    payload = {
        "reference": resolved.reference,
        "alias": resolved.record.alias if resolved.record else None,
        "wandb": resolved.record.wandb if resolved.record else None,
        "checkpoint": str(resolved.checkpoint),
    }
    print(json.dumps(payload, indent=2) if args.json else payload["checkpoint"])


if __name__ == "__main__":
    main()
