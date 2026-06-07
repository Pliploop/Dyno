from pathlib import Path

import pytest

from dyno.utils.experiment_registry import (
    load_experiment_registry,
    resolve_experiment_reference,
    select_checkpoint,
)


HEADER = """# Registry

| Alias | Date | Status | Phase | Paper section | Kind | Parent | W&B | Local checkpoint or artifact | Launch/config | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
"""


def test_registry_resolves_alias_and_wandb_id_to_best_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    best = checkpoint_dir / "best-step=10-val_loss=0.1.ckpt"
    best.touch()
    (checkpoint_dir / "last.ckpt").touch()
    registry = tmp_path / "EXPERIMENTS.md"
    registry.write_text(
        HEADER
        + f"| muq-d32 | 2026-06-07 | completed | 2 | reconstruction | train | - | "
        f"https://wandb.ai/entity/project/runs/abc123 | {checkpoint_dir} | experiment=paper_muq_1hz | - |\n",
        encoding="utf-8",
    )

    records = load_experiment_registry(registry)
    assert records[0].alias == "muq-d32"
    assert resolve_experiment_reference("muq-d32", registry).checkpoint == best.resolve()
    assert resolve_experiment_reference("abc123", registry).checkpoint == best.resolve()


def test_direct_directory_can_select_last_checkpoint(tmp_path: Path):
    best = tmp_path / "best-step=1-val_loss=0.2.ckpt"
    last = tmp_path / "last.ckpt"
    best.touch()
    last.touch()

    assert select_checkpoint(tmp_path, "last") == last.resolve()


def test_registry_rejects_pending_checkpoint(tmp_path: Path):
    registry = tmp_path / "EXPERIMENTS.md"
    registry.write_text(
        HEADER
        + "| waiting | 2026-06-07 | running | 2 | reconstruction | train | - | pending | pending | x | - |\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no local checkpoint"):
        resolve_experiment_reference("waiting", registry)
