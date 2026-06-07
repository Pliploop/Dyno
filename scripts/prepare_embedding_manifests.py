"""Validate an extracted embedding dataset and materialize split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-manifests", type=Path, required=True)
    parser.add_argument("--output-manifests", type=Path, default=None)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--path-col", default="npy_path")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def target_path(path: str, source_root: Path, target_root: Path) -> Path:
    source = Path(path)
    try:
        relative = source.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{source} is not under source root {source_root}") from exc
    return target_root / relative


def validate_array(path: Path, embedding_dim: int) -> None:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:
        raise ValueError(f"Could not read NumPy header: {path}") from exc
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array at {path}, got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] != embedding_dim:
        raise ValueError(
            f"Unexpected embedding shape at {path}: {array.shape}; "
            f"expected (*, {embedding_dim})"
        )
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings at {path}, got {array.dtype}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    output_dir = (args.output_manifests or target_root / "manifests").resolve()

    split_frames: dict[str, pd.DataFrame] = {}
    referenced_paths: set[Path] = set()
    for split in SPLITS:
        manifest_path = args.source_manifests / f"{split}.csv"
        frame = pd.read_csv(manifest_path)
        if args.path_col not in frame.columns:
            raise ValueError(f"{manifest_path} has no {args.path_col!r} column")
        mapped = [
            target_path(path, source_root, target_root)
            for path in frame[args.path_col].astype(str)
        ]
        for path in mapped:
            if not path.is_file():
                raise FileNotFoundError(path)
            validate_array(path, args.embedding_dim)
        frame[args.path_col] = [str(path) for path in mapped]
        split_frames[split] = frame
        referenced_paths.update(mapped)

    extracted_paths = set(target_root.rglob("*.npy"))
    source_paths = set(source_root.rglob("*.npy"))
    source_relative = {path.relative_to(source_root) for path in source_paths}
    target_relative = {path.relative_to(target_root) for path in extracted_paths}
    missing = source_relative - target_relative
    extra = target_relative - source_relative
    if missing or extra:
        raise RuntimeError(
            f"Extraction mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    for path in sorted(extracted_paths):
        validate_array(path, args.embedding_dim)

    print(
        f"Validated {len(extracted_paths)} arrays; "
        f"manifest rows={sum(len(frame) for frame in split_frames.values())}; "
        f"unreferenced arrays={len(extracted_paths - referenced_paths)}"
    )
    if args.verify_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in split_frames.items():
        output_path = output_dir / f"{split}.csv"
        frame.to_csv(output_path, index=False)
        print(f"Wrote {len(frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
