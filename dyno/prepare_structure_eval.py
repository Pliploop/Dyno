"""Prepare normalized SALAMI/Harmonix manifests for CoverX.

This script does not download audio. It expects annotations under
``/gpfs/scratch/acw749/datasets/structure/annotations`` and writes manifests
that point to extracted embeddings under
``/gpfs/scratch/acw749/datasets/structure/features``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from dyno.evaluation.structure import collapse_labels_to_form, global_form_attributes


DEFAULT_ROOT = Path("/gpfs/scratch/acw749/datasets/structure")


def _safe_id(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _parse_boundary_label_file(path: Path) -> tuple[list[float], list[str]]:
    boundaries: list[float] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            label = str(parts[1]).strip()
            boundaries.append(t)
            labels.append(label)
    return boundaries, labels


def _salami_layer_match(path: Path, layer: str) -> bool:
    name = path.name.lower()
    layer = layer.lower()
    aliases = {
        "uppercase": ("uppercase", "upper", "large"),
        "lowercase": ("lowercase", "lower", "small"),
        "functions": ("function", "functions", "funct"),
    }.get(layer, (layer,))
    return any(alias in name for alias in aliases)


def find_salami_annotation_files(salami_root: Path, layer: str) -> list[Path]:
    parsed = sorted((salami_root / "annotations").glob("*/parsed/*.txt"))
    matched = [p for p in parsed if _salami_layer_match(p, layer)]
    if matched:
        return matched
    return parsed


def load_salami_rows(root: Path, encoder: str, rate: str, layer: str) -> list[dict]:
    salami_root = root / "annotations" / "salami"
    rows = []
    seen: set[str] = set()
    for path in find_salami_annotation_files(salami_root, layer):
        song_id = path.parents[1].name if path.parent.name == "parsed" else path.parent.name
        if song_id in seen:
            continue
        boundaries, labels = _parse_boundary_label_file(path)
        if not labels:
            continue
        seen.add(song_id)
        rows.append(_make_row(
            dataset="salami",
            track_id=song_id,
            boundaries=boundaries,
            labels=labels,
            encoder=encoder,
            rate=rate,
        ))
    return rows


def _load_harmonix_durations(harmonix_root: Path) -> dict[str, float]:
    metadata = harmonix_root / "dataset" / "metadata.tsv"
    if not metadata.exists():
        return {}
    df = pd.read_csv(metadata, sep="\t")
    if "File" not in df.columns or "Duration" not in df.columns:
        return {}
    return {
        str(row["File"]).replace(".mp3", "").replace(".wav", ""): float(row["Duration"])
        for _, row in df.iterrows()
        if pd.notna(row.get("Duration"))
    }


def load_harmonix_rows(root: Path, encoder: str, rate: str) -> list[dict]:
    harmonix_root = root / "annotations" / "harmonix"
    durations = _load_harmonix_durations(harmonix_root)
    segment_dir = harmonix_root / "dataset" / "segments"
    rows = []
    segment_paths = sorted(segment_dir.glob("*.tsv")) + sorted(segment_dir.glob("*.txt"))
    for path in segment_paths:
        track_id = _safe_id(path)
        boundaries, labels = _parse_boundary_label_file(path)
        if not labels:
            continue
        duration = durations.get(track_id)
        if duration is not None and (not boundaries or boundaries[-1] < duration):
            boundaries = [*boundaries, duration]
        rows.append(_make_row(
            dataset="harmonix",
            track_id=track_id,
            boundaries=boundaries,
            labels=labels,
            encoder=encoder,
            rate=rate,
        ))
    return rows


def _make_row(
    dataset: str,
    track_id: str,
    boundaries: list[float],
    labels: list[str],
    encoder: str,
    rate: str,
) -> dict:
    form = collapse_labels_to_form(labels)
    attrs = global_form_attributes(labels)
    return {
        "dataset": dataset,
        "track_id": track_id,
        "audio_path": str(Path("audio") / dataset / f"{track_id}.wav"),
        "feature_path": str(Path(dataset) / encoder / rate / f"{track_id}.npy"),
        "boundaries": "|".join(f"{t:.6f}" for t in boundaries),
        "labels": "|".join(str(label) for label in labels),
        "form_string": form,
        **attrs,
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "track_id",
        "audio_path",
        "feature_path",
        "boundaries",
        "labels",
        "form_string",
        "section_count",
        "unique_label_count",
        "repetition_rate",
        "return_a",
        "bridge",
        "through",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--rate", required=True)
    parser.add_argument("--salami-layer", default="uppercase", choices=["uppercase", "lowercase", "functions"])
    parser.add_argument("--require-features", action="store_true")
    args = parser.parse_args()

    rows = [
        *load_salami_rows(args.root, args.encoder, args.rate, args.salami_layer),
        *load_harmonix_rows(args.root, args.encoder, args.rate),
    ]
    if args.require_features:
        feature_root = args.root / "features"
        rows = [row for row in rows if (feature_root / row["feature_path"]).exists()]

    manifest_dir = args.root / "manifests" / args.encoder / args.rate
    write_manifest(manifest_dir / "all.csv", rows)
    write_manifest(manifest_dir / "salami.csv", [row for row in rows if row["dataset"] == "salami"])
    write_manifest(manifest_dir / "harmonix.csv", [row for row in rows if row["dataset"] == "harmonix"])
    print(f"Wrote {len(rows)} rows to {manifest_dir}")


if __name__ == "__main__":
    main()
