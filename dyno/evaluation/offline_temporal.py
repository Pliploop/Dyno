"""Offline cross-encoder MSPF geometry and temporal retrieval artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from dyno.callbacks.annotation_free import (
    _dtw_distance,
    _standardized_euclidean_distance_matrix,
)
from dyno.evaluation.structure import cosine_distance_matrix
from dyno.evaluation.temporal import compute_mspf


def _resample_sequence(
    features: torch.Tensor,
    source_rate: float,
    target_rate: float,
) -> torch.Tensor:
    if source_rate == target_rate:
        return features
    target_frames = max(1, int(round(features.shape[0] * target_rate / source_rate)))
    pooled = torch.nn.functional.interpolate(
        features.T.unsqueeze(0),
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return pooled.squeeze(0).T


def _load_manifest_paths(manifest_csv: str | Path, feature_root: str | Path) -> dict[str, Path]:
    frame = pd.read_csv(manifest_csv, dtype={"track_id": str})
    root = Path(feature_root)
    paths = {}
    for row in frame.itertuples(index=False):
        path = Path(str(row.feature_path))
        paths[str(row.track_id)] = path if path.is_absolute() else root / path
    return paths


def load_aligned_features(
    manifests: dict[str, str | Path],
    feature_root: str | Path,
    max_tracks: int | None = 128,
    seed: int = 142,
) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    path_maps = {
        encoder: _load_manifest_paths(manifest, feature_root)
        for encoder, manifest in manifests.items()
    }
    common_ids = sorted(set.intersection(*(set(paths) for paths in path_maps.values())))
    common_ids = [
        track_id
        for track_id in common_ids
        if all(path_maps[encoder][track_id].is_file() for encoder in path_maps)
    ]
    if max_tracks is not None and len(common_ids) > max_tracks:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(common_ids), size=max_tracks, replace=False))
        common_ids = [common_ids[index] for index in keep]
    features = {
        encoder: [
            np.asarray(np.load(paths[track_id], mmap_mode="r"), dtype=np.float32)
            for track_id in common_ids
        ]
        for encoder, paths in path_maps.items()
    }
    return common_ids, features


def extract_representations(
    model,
    sequences: list[np.ndarray],
    source_rate: float,
    model_rate: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    contents = []
    temporals = []
    model.eval()
    with torch.inference_mode():
        for array in sequences:
            sequence = torch.from_numpy(array).to(device=device, dtype=torch.float32)
            sequence = _resample_sequence(sequence, source_rate, model_rate).unsqueeze(0)
            mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=device)
            normalized = model.normalize_input(sequence)
            contents.append(model.get_content_token(normalized, mask=mask)[0].cpu().numpy())
            temporals.append(model.encode(normalized, mask=mask)[2][0].cpu().numpy())
    content = np.asarray(contents, dtype=np.float32)
    temporal = np.asarray(temporals, dtype=np.float32)
    return {"content": content, "temporal": temporal}


def _finite_upper(matrix: np.ndarray) -> np.ndarray:
    values = matrix[np.triu_indices_from(matrix, k=1)]
    return values[np.isfinite(values)]


def _sampled_mspf_dtw(
    curves: np.ndarray,
    pair_indices: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [_dtw_distance(curves[i], curves[j]) for i, j in pair_indices],
        dtype=np.float32,
    )


def cross_encoder_mspf_geometry(
    temporal: np.ndarray,
    encoder_features: dict[str, list[np.ndarray]],
    mspf_points: int = 100,
    mspf_max_frames: int = 256,
    max_pairs: int = 10000,
    seed: int = 142,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    n_tracks = temporal.shape[0]
    pairs = np.asarray(
        [(i, j) for i in range(n_tracks) for j in range(i + 1, n_tracks)],
        dtype=np.int64,
    )
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = pairs[np.sort(rng.choice(len(pairs), size=max_pairs, replace=False))]

    temporal_distance = _standardized_euclidean_distance_matrix(temporal)
    temporal_pairs = temporal_distance[pairs[:, 0], pairs[:, 1]]
    metrics = {}
    curves_by_encoder = {}
    for encoder, sequences in encoder_features.items():
        curves = np.asarray(
            [
                compute_mspf(
                    torch.from_numpy(
                        sequence[
                            np.linspace(
                                0,
                                len(sequence) - 1,
                                min(len(sequence), mspf_max_frames),
                            ).round().astype(np.int64)
                        ]
                    ),
                    n_points=mspf_points,
                    normalize=True,
                )
                for sequence in sequences
            ],
            dtype=np.float32,
        )
        curves_by_encoder[encoder] = curves
        mspf_distance = _sampled_mspf_dtw(curves, pairs)
        metrics[
            f"paper.mspf_cross_encoder/test/{encoder}/geometry_spearman"
        ] = float(spearmanr(temporal_pairs, mspf_distance).statistic)
    return metrics, curves_by_encoder


def retrieval_artifact(
    track_ids: list[str],
    representations: dict[str, np.ndarray],
    reference_mspf: np.ndarray,
    top_k: int = 5,
    n_queries: int = 8,
) -> list[dict[str, float | int | str]]:
    content_distance = cosine_distance_matrix(representations["content"])
    temporal_distance = _standardized_euclidean_distance_matrix(representations["temporal"])
    content_scale = np.nanmedian(_finite_upper(content_distance))
    temporal_scale = np.nanmedian(_finite_upper(temporal_distance))
    combined_distance = (
        content_distance / max(float(content_scale), 1e-8)
        + temporal_distance / max(float(temporal_scale), 1e-8)
    )
    distances = {
        "content": content_distance,
        "temporal": temporal_distance,
        "combined": combined_distance,
    }
    query_indices = np.linspace(0, len(track_ids) - 1, min(n_queries, len(track_ids))).round().astype(int)
    rows = []
    for query_index in query_indices:
        for representation, matrix in distances.items():
            order = np.argsort(matrix[query_index])
            order = [index for index in order if index != query_index][:top_k]
            for rank, neighbor_index in enumerate(order, start=1):
                rows.append(
                    {
                        "query_id": track_ids[query_index],
                        "representation": representation,
                        "rank": rank,
                        "neighbor_id": track_ids[neighbor_index],
                        "representation_distance": float(matrix[query_index, neighbor_index]),
                        "mspf_dtw": _dtw_distance(
                            reference_mspf[query_index],
                            reference_mspf[neighbor_index],
                        ),
                    }
                )
    return rows


def run_offline_temporal_evaluation(
    model,
    manifests: dict[str, str | Path],
    feature_root: str | Path,
    token_encoder: str = "muq",
    source_rate: float = 2.0,
    model_rate: float = 1.0,
    max_tracks: int | None = 128,
    max_pairs: int = 10000,
    mspf_points: int = 100,
    mspf_max_frames: int = 256,
    top_k: int = 5,
    n_queries: int = 8,
    seed: int = 142,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    track_ids, features = load_aligned_features(manifests, feature_root, max_tracks, seed)
    if token_encoder not in features:
        raise KeyError(f"Token encoder {token_encoder!r} is not present in manifests")
    representations = extract_representations(
        model,
        features[token_encoder],
        source_rate,
        model_rate,
        torch.device(device),
    )
    metrics, curves = cross_encoder_mspf_geometry(
        representations["temporal"],
        features,
        mspf_points,
        mspf_max_frames,
        max_pairs,
        seed,
    )
    rows = retrieval_artifact(
        track_ids,
        representations,
        curves[token_encoder],
        top_k,
        n_queries,
    )
    metrics["paper.temporal_retrieval/test/track_count"] = float(len(track_ids))
    return metrics, rows
