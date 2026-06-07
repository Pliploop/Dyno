"""Resolve paper experiment aliases to local checkpoints."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REGISTRY_COLUMNS = (
    "Alias",
    "Date",
    "Status",
    "Phase",
    "Paper section",
    "Kind",
    "Parent",
    "W&B",
    "Local checkpoint or artifact",
    "Launch/config",
    "Notes",
)
MISSING_VALUES = {"", "-", "pending", "none", "null", "n/a"}


@dataclass(frozen=True)
class ExperimentRecord:
    alias: str
    date: str
    status: str
    phase: str
    paper_section: str
    kind: str
    parent: str
    wandb: str
    local_path: str
    launch_config: str
    notes: str


@dataclass(frozen=True)
class ResolvedExperiment:
    reference: str
    checkpoint: Path
    record: ExperimentRecord | None = None


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in next(csv.reader([line.strip().strip("|")], delimiter="|"))]


def load_experiment_registry(path: str | Path) -> list[ExperimentRecord]:
    registry_path = Path(path)
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if _cells(line) == list(REGISTRY_COLUMNS)),
        None,
    )
    if header_index is None:
        raise ValueError(f"Could not find the experiment table in {registry_path}")

    records = []
    for line in lines[header_index + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = _cells(line)
        if len(cells) != len(REGISTRY_COLUMNS):
            raise ValueError(f"Malformed experiment row in {registry_path}: {line}")
        records.append(ExperimentRecord(*cells))
    return records


def _clean_path(value: str) -> str:
    value = value.strip().strip("`")
    markdown_link = re.fullmatch(r"\[[^\]]+\]\(([^)]+)\)", value)
    return markdown_link.group(1) if markdown_link else value


def _wandb_names(value: str) -> set[str]:
    value = value.strip().strip("`")
    if value.lower() in MISSING_VALUES:
        return set()
    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    return {value, path_parts[-1]} if path_parts else {value}


def _record_matches(record: ExperimentRecord, reference: str) -> bool:
    return reference == record.alias or reference in _wandb_names(record.wandb)


def select_checkpoint(path: str | Path, preference: str = "best") -> Path:
    candidate = Path(_clean_path(str(path))).expanduser()
    if candidate.is_file():
        if candidate.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got {candidate}")
        return candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)

    checkpoints = list(candidate.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {candidate}")

    last = [path for path in checkpoints if path.name == "last.ckpt"]
    best = [path for path in checkpoints if path.name.startswith("best-")]
    newest = lambda paths: max(paths, key=lambda item: item.stat().st_mtime)
    if preference == "last":
        selected = newest(last or checkpoints)
    elif preference == "newest":
        selected = newest(checkpoints)
    elif preference == "best":
        selected = newest(best or [path for path in checkpoints if path not in last] or checkpoints)
    else:
        raise ValueError("checkpoint preference must be one of: best, last, newest")
    return selected.resolve()


def resolve_experiment_reference(
    reference: str,
    registry_path: str | Path = ".agents/EXPERIMENTS.md",
    checkpoint_preference: str = "best",
) -> ResolvedExperiment:
    direct_path = Path(reference).expanduser()
    if direct_path.exists():
        return ResolvedExperiment(
            reference=reference,
            checkpoint=select_checkpoint(direct_path, checkpoint_preference),
        )

    matches = [
        record
        for record in load_experiment_registry(registry_path)
        if _record_matches(record, reference)
    ]
    if not matches:
        raise KeyError(
            f"No experiment matching {reference!r} in {registry_path}. "
            "Use a registry alias, W&B run ID/URL, or direct checkpoint path."
        )
    if len(matches) > 1:
        aliases = ", ".join(record.alias for record in matches)
        raise ValueError(f"Ambiguous experiment reference {reference!r}: {aliases}")

    record = matches[0]
    local_path = _clean_path(record.local_path)
    if local_path.lower() in MISSING_VALUES:
        raise ValueError(f"Experiment {record.alias!r} has no local checkpoint recorded yet")
    return ResolvedExperiment(
        reference=reference,
        checkpoint=select_checkpoint(local_path, checkpoint_preference),
        record=record,
    )
