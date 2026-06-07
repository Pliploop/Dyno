"""Create deterministic, exact-size grouped train/validation/test manifests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifests", type=Path, required=True)
    parser.add_argument("--output-manifests", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--group-col", default="artist_id")
    parser.add_argument("--path-col", default="npy_path")
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=142)
    return parser.parse_args()


def exact_group_subset(
    group_sizes: list[tuple[str, int]],
    target: int,
    rng: random.Random,
) -> set[str]:
    """Select complete groups whose sizes sum exactly to target."""
    shuffled = sorted(group_sizes)
    rng.shuffle(shuffled)
    previous: dict[int, tuple[int, str] | None] = {0: None}
    for group, size in shuffled:
        for total in sorted(tuple(previous), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in previous:
                previous[new_total] = (total, group)
        if target in previous:
            break
    if target not in previous:
        raise RuntimeError(f"Could not form an exact grouped subset of {target} rows")

    selected: set[str] = set()
    total = target
    while total:
        prior = previous[total]
        if prior is None:
            raise RuntimeError("Invalid grouped subset reconstruction")
        total, group = prior
        selected.add(group)
    return selected


def load_all_manifests(manifest_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(manifest_dir / f"{split}.csv") for split in SPLITS]
    frame = pd.concat(frames, ignore_index=True)
    if frame["track_id"].duplicated().any():
        raise ValueError("Input manifests contain duplicate track IDs")
    return frame


def remap_paths(
    frame: pd.DataFrame,
    path_col: str,
    source_root: Path,
    target_root: Path,
) -> pd.DataFrame:
    frame = frame.copy()
    remapped = []
    for raw_path in frame[path_col].astype(str):
        path = Path(raw_path)
        try:
            relative = path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"{path} is not under {source_root}") from exc
        target = target_root / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        remapped.append(str(target))
    frame[path_col] = remapped
    return frame


def assert_disjoint(frames: dict[str, pd.DataFrame], column: str) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frames[left][column]) & set(frames[right][column])
        if overlap:
            raise RuntimeError(
                f"{column} leakage between {left} and {right}: {len(overlap)} values"
            )


def main() -> None:
    args = parse_args()
    frame = load_all_manifests(args.input_manifests)
    if args.group_col not in frame or args.path_col not in frame:
        raise ValueError("Input manifests are missing required grouping/path columns")

    frame = remap_paths(
        frame,
        path_col=args.path_col,
        source_root=args.source_root.resolve(),
        target_root=args.target_root.resolve(),
    )
    group_sizes = list(frame.groupby(args.group_col).size().items())
    rng = random.Random(args.seed)
    val_groups = exact_group_subset(group_sizes, args.val_size, rng)
    remaining = [(group, size) for group, size in group_sizes if group not in val_groups]
    test_groups = exact_group_subset(remaining, args.test_size, rng)

    split = pd.Series("train", index=frame.index)
    split.loc[frame[args.group_col].isin(val_groups)] = "val"
    split.loc[frame[args.group_col].isin(test_groups)] = "test"
    frames = {name: frame.loc[split == name].copy() for name in SPLITS}

    expected = {
        "train": len(frame) - args.val_size - args.test_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    for name, expected_size in expected.items():
        if len(frames[name]) != expected_size:
            raise RuntimeError(f"{name} has {len(frames[name])} rows, expected {expected_size}")
    for column in ("track_id", args.group_col, "album_id"):
        assert_disjoint(frames, column)

    args.output_manifests.mkdir(parents=True, exist_ok=True)
    for name, split_frame in frames.items():
        output = args.output_manifests / f"{name}.csv"
        split_frame.to_csv(output, index=False)
        print(
            f"{name}: tracks={len(split_frame)}, "
            f"artists={split_frame[args.group_col].nunique()}, "
            f"albums={split_frame.album_id.nunique()}"
        )
    summary = args.output_manifests / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"seed={args.seed}",
                f"total={len(frame)}",
                f"train={len(frames['train'])}",
                f"val={len(frames['val'])}",
                f"test={len(frames['test'])}",
                f"group_col={args.group_col}",
                "split_type=exact_group_disjoint",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
