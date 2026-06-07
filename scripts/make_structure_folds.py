"""Create deterministic artist-grouped folds for structure probing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def _assign_groups(groups: dict[str, list[tuple[str, str]]], n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    rows = []
    for group, tracks in items:
        fold = int(np.argmin(fold_sizes))
        fold_sizes[fold] += len(tracks)
        rows.extend((dataset, track_id, group, fold) for dataset, track_id in tracks)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--salami-manifest", type=Path, required=True)
    parser.add_argument("--harmonix-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=142)
    args = parser.parse_args()

    salami_meta = pd.read_csv(
        args.structure_root / "annotations/salami/metadata/SALAMI_iTunes_library.csv"
    )
    salami_artist = {
        str(row.salami_id): str(row.Artist)
        for row in salami_meta.itertuples()
        if pd.notna(row.Artist)
    }
    harmonix_meta = pd.read_csv(
        args.structure_root / "annotations/harmonix/dataset/metadata.csv"
    )
    harmonix_artist = {
        str(row.File): str(row.Artist)
        for row in harmonix_meta.itertuples()
        if pd.notna(row.Artist)
    }

    groups: dict[str, list[tuple[str, str]]] = {}
    for dataset, manifest, artists in (
        ("salami", args.salami_manifest, salami_artist),
        ("harmonix", args.harmonix_manifest, harmonix_artist),
    ):
        frame = pd.read_csv(manifest, dtype={"track_id": str})
        for track_id in frame["track_id"]:
            artist = artists.get(track_id, f"track:{track_id}").strip().lower()
            group = f"{dataset}:{artist}"
            groups.setdefault(group, []).append((dataset, track_id))

    rows = _assign_groups(groups, args.folds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "track_id", "group", "fold"))
        writer.writerows(sorted(rows))
    print(f"Wrote {len(rows)} tracks to {args.output}")


if __name__ == "__main__":
    main()
