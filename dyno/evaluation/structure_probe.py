"""Linear frame-level probing for music structure analysis."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mir_eval
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset


PROBE_INPUTS = (
    "local",
    "local_content",
    "local_temporal",
    "local_content_temporal",
    "content_position",
    "temporal_position",
)
HARMONIX_FUNCTIONS = ("intro", "verse", "chorus", "bridge", "inst", "outro", "silence")


def parse_frame_rate(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized.endswith("hz"):
        normalized = normalized[:-2]
    rate = float(normalized)
    if rate <= 0:
        raise ValueError(f"Frame rate must be positive, got {value!r}")
    return rate


@dataclass
class ProbeTrack:
    dataset: str
    track_id: str
    features: np.ndarray
    boundary_times: np.ndarray
    function_times: np.ndarray
    function_labels: list[str]
    fold: int
    content: np.ndarray | None = None
    temporal: np.ndarray | None = None


def _read_annotation(path: Path) -> tuple[np.ndarray, list[str]]:
    times: list[float] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
            labels.append(parts[1].strip())
    return np.asarray(times, dtype=np.float32), labels


def _salami_annotation(annotation_root: Path, track_id: str, layer: str) -> tuple[np.ndarray, list[str]]:
    parsed = annotation_root / "salami" / "annotations" / str(track_id) / "parsed"
    path = parsed / f"textfile1_{layer}.txt"
    if not path.is_file():
        matches = sorted(parsed.glob(f"textfile1*_{layer}.txt"))
        if not matches:
            raise FileNotFoundError(path)
        path = matches[0]
    return _read_annotation(path)


def normalize_harmonix_function(label: str) -> str:
    value = re.sub(r"[\s_-]+", "", str(label).strip().lower())
    value = re.sub(r"\d+$", "", value)
    if value in {"silence", "end", "fadein", "fadeout"}:
        return "silence"
    if "intro" in value:
        return "intro"
    if "prechor" in value or "chorus" in value or value in {"hook", "refrain"}:
        return "chorus"
    if "verse" in value or value == "rap":
        return "verse"
    if "bridge" in value or value in {"break", "transition", "preverse"}:
        return "bridge"
    if "outro" in value or "coda" in value:
        return "outro"
    return "inst"


def normalize_salami_function(label: str) -> str:
    value = re.sub(r"\s+", "_", str(label).strip().lower())
    return value or "unknown"


def _segment_labels(
    times: np.ndarray,
    labels: list[str],
    n_frames: int,
    frame_rate: float,
) -> np.ndarray:
    if not labels:
        return np.asarray(["unknown"] * n_frames, dtype=object)
    starts = times[: len(labels)]
    centers = (np.arange(n_frames, dtype=np.float32) + 0.5) / frame_rate
    indices = np.searchsorted(starts, centers, side="right") - 1
    indices = np.clip(indices, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[indices]


def _boundary_targets(times: np.ndarray, n_frames: int, frame_rate: float) -> np.ndarray:
    target = np.zeros(n_frames, dtype=np.float32)
    duration = n_frames / frame_rate
    internal = times[(times > 0.0) & (times < duration)]
    indices = np.rint(internal * frame_rate - 0.5).astype(np.int64)
    indices = indices[(indices >= 0) & (indices < n_frames)]
    target[indices] = 1.0
    return target


def _position_encoding(n_frames: int, dim: int = 32) -> np.ndarray:
    position = np.arange(n_frames, dtype=np.float32)[:, None]
    scale = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000.0) / dim))
    encoding = np.zeros((n_frames, dim), dtype=np.float32)
    encoding[:, 0::2] = np.sin(position * scale)
    encoding[:, 1::2] = np.cos(position * scale)
    return encoding


def extract_track_tokens(
    model,
    tracks: Iterable[ProbeTrack],
    device: torch.device,
) -> None:
    model.eval()
    with torch.inference_mode():
        for track in tracks:
            sequence = (
                torch.from_numpy(track.features)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(0)
            )
            mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=device)
            normalized = model.normalize_input(sequence)
            track.content = model.get_content_token(normalized, mask=mask)[0].cpu().float().numpy()
            track.temporal = model.encode(normalized, mask=mask)[2][0].cpu().float().numpy()


def load_probe_tracks(
    manifest_csv: str | Path,
    folds_csv: str | Path,
    feature_root: str | Path,
    annotation_root: str | Path,
    dataset: str,
    frame_rate: str | float = 1.0,
    salami_boundary_layer: str = "uppercase",
    max_tracks: int | None = None,
) -> list[ProbeTrack]:
    manifest = pd.read_csv(manifest_csv)
    folds = pd.read_csv(folds_csv, dtype={"track_id": str})
    folds = folds[folds["dataset"].str.lower() == dataset.lower()]
    fold_by_id = dict(zip(folds["track_id"], folds["fold"]))
    feature_root = Path(feature_root)
    annotation_root = Path(annotation_root)
    tracks: list[ProbeTrack] = []

    for row in manifest.itertuples(index=False):
        track_id = str(row.track_id)
        if track_id not in fold_by_id:
            continue
        feature_path = Path(str(row.feature_path))
        if not feature_path.is_absolute():
            feature_path = feature_root / feature_path
        if not feature_path.is_file():
            continue
        features = np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] < 2:
            continue

        if dataset.lower() == "salami":
            boundary_times, _ = _salami_annotation(
                annotation_root,
                track_id,
                salami_boundary_layer,
            )
            function_times, function_labels = _salami_annotation(
                annotation_root,
                track_id,
                "functions",
            )
            function_labels = [normalize_salami_function(label) for label in function_labels]
        else:
            boundary_times = np.asarray(
                [float(value) for value in str(row.boundaries).split("|") if value],
                dtype=np.float32,
            )
            raw_labels = [value for value in str(row.labels).split("|") if value]
            function_times = boundary_times
            function_labels = [normalize_harmonix_function(label) for label in raw_labels]

        tracks.append(
            ProbeTrack(
                dataset=dataset.lower(),
                track_id=track_id,
                features=features,
                boundary_times=boundary_times,
                function_times=function_times,
                function_labels=function_labels,
                fold=int(fold_by_id[track_id]),
            )
        )
        if max_tracks is not None and len(tracks) >= max_tracks:
            break
    if not tracks:
        raise RuntimeError(f"No probe tracks could be loaded from {manifest_csv}")
    return tracks


class ProbeWindowDataset(Dataset):
    def __init__(
        self,
        tracks: list[ProbeTrack],
        track_indices: list[int],
        label_to_index: dict[str, int],
        probe_input: str,
        frame_rate: float = 1.0,
        window_seconds: float = 30.0,
        hop_seconds: float = 30.0,
        position_dim: int = 32,
    ):
        if probe_input not in PROBE_INPUTS:
            raise ValueError(f"Unknown probe input {probe_input!r}")
        self.tracks = tracks
        self.label_to_index = label_to_index
        self.probe_input = probe_input
        self.frame_rate = frame_rate
        self.window_frames = int(round(window_seconds * frame_rate))
        self.hop_frames = int(round(hop_seconds * frame_rate))
        self.position_dim = position_dim
        self.windows: list[tuple[int, int]] = []
        for track_index in track_indices:
            n_frames = tracks[track_index].features.shape[0]
            starts = list(range(0, max(n_frames - self.window_frames + 1, 1), self.hop_frames))
            final_start = max(0, n_frames - self.window_frames)
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            self.windows.extend((track_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def _input(self, track: ProbeTrack, start: int, stop: int) -> np.ndarray:
        local = track.features[start:stop]
        n_frames = local.shape[0]
        content = np.repeat(track.content[None, :], n_frames, axis=0)
        temporal = np.repeat(track.temporal[None, :], n_frames, axis=0)
        position = _position_encoding(n_frames, self.position_dim)
        parts = {
            "local": (local,),
            "local_content": (local, content),
            "local_temporal": (local, temporal),
            "local_content_temporal": (local, content, temporal),
            "content_position": (content, position),
            "temporal_position": (temporal, position),
        }[self.probe_input]
        return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_index, start = self.windows[index]
        track = self.tracks[track_index]
        stop = min(start + self.window_frames, track.features.shape[0])
        x = self._input(track, start, stop)
        labels = _segment_labels(
            track.function_times,
            track.function_labels,
            track.features.shape[0],
            self.frame_rate,
        )[start:stop]
        y_function = np.asarray(
            [self.label_to_index.get(str(label), self.label_to_index["unknown"]) for label in labels],
            dtype=np.int64,
        )
        y_boundary = _boundary_targets(
            track.boundary_times,
            track.features.shape[0],
            self.frame_rate,
        )[start:stop]
        valid = x.shape[0]
        if valid < self.window_frames:
            x = np.pad(x, ((0, self.window_frames - valid), (0, 0)))
            y_function = np.pad(y_function, (0, self.window_frames - valid))
            y_boundary = np.pad(y_boundary, (0, self.window_frames - valid))
        return {
            "x": torch.from_numpy(x),
            "boundary": torch.from_numpy(y_boundary),
            "function": torch.from_numpy(y_function),
            "mask": torch.arange(self.window_frames) < valid,
            "track_index": torch.tensor(track_index),
            "start": torch.tensor(start),
        }


class LinearStructureProbe(nn.Module):
    def __init__(self, input_dim: int, n_functions: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_functions + 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.linear(x)
        return output[..., 0], output[..., 1:]


def _function_vocabulary(tracks: list[ProbeTrack], train_indices: list[int]) -> dict[str, int]:
    labels = sorted(
        {
            label
            for index in train_indices
            for label in tracks[index].function_labels
        }
    )
    labels = [label for label in labels if label != "unknown"]
    return {label: index for index, label in enumerate([*labels, "unknown"])}


def _joint_loss(
    boundary_logits: torch.Tensor,
    function_logits: torch.Tensor,
    boundary: torch.Tensor,
    function: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    boundary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        boundary_logits[mask],
        boundary[mask],
    )
    function_loss = torch.nn.functional.cross_entropy(
        function_logits[mask],
        function[mask],
    )
    return boundary_loss + function_loss


def _train_one_fold(
    tracks: list[ProbeTrack],
    train_indices: list[int],
    val_indices: list[int],
    probe_input: str,
    frame_rate: float,
    window_seconds: float,
    hop_seconds: float,
    position_dim: int,
    batch_size: int,
    epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
) -> tuple[LinearStructureProbe, dict[str, int]]:
    vocabulary = _function_vocabulary(tracks, train_indices)
    train_data = ProbeWindowDataset(
        tracks,
        train_indices,
        vocabulary,
        probe_input,
        frame_rate,
        window_seconds,
        hop_seconds,
        position_dim,
    )
    val_data = ProbeWindowDataset(
        tracks,
        val_indices,
        vocabulary,
        probe_input,
        frame_rate,
        window_seconds,
        hop_seconds,
        position_dim,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    input_dim = train_data[0]["x"].shape[-1]
    probe = LinearStructureProbe(input_dim, len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def lr_factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best_state = copy.deepcopy(probe.state_dict())
    best_loss = float("inf")
    for _ in range(epochs):
        probe.train()
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            boundary_logits, function_logits = probe(batch["x"])
            loss = _joint_loss(
                boundary_logits,
                function_logits,
                batch["boundary"],
                batch["function"],
                batch["mask"],
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

        probe.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                boundary_logits, function_logits = probe(batch["x"])
                loss = _joint_loss(
                    boundary_logits,
                    function_logits,
                    batch["boundary"],
                    batch["function"],
                    batch["mask"],
                )
                total += float(loss) * int(batch["mask"].sum())
                count += int(batch["mask"].sum())
        val_loss = total / max(count, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    return probe, vocabulary


def _predict_tracks(
    probe: LinearStructureProbe,
    tracks: list[ProbeTrack],
    indices: list[int],
    vocabulary: dict[str, int],
    probe_input: str,
    frame_rate: float,
    window_seconds: float,
    hop_seconds: float,
    position_dim: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = ProbeWindowDataset(
        tracks,
        indices,
        vocabulary,
        probe_input,
        frame_rate,
        window_seconds,
        hop_seconds,
        position_dim,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sums: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    probe.eval()
    with torch.inference_mode():
        for batch in loader:
            boundary_logits, function_logits = probe(batch["x"].to(device))
            boundary = torch.sigmoid(boundary_logits).cpu().numpy()
            function = torch.softmax(function_logits, dim=-1).cpu().numpy()
            for row in range(boundary.shape[0]):
                track_index = int(batch["track_index"][row])
                start = int(batch["start"][row])
                valid = int(batch["mask"][row].sum())
                n_frames = tracks[track_index].features.shape[0]
                if track_index not in sums:
                    sums[track_index] = (
                        np.zeros(n_frames, dtype=np.float64),
                        np.zeros((n_frames, function.shape[-1]), dtype=np.float64),
                        np.zeros(n_frames, dtype=np.float64),
                    )
                boundary_sum, function_sum, count = sums[track_index]
                stop = min(start + valid, n_frames)
                width = stop - start
                boundary_sum[start:stop] += boundary[row, :width]
                function_sum[start:stop] += function[row, :width]
                count[start:stop] += 1.0
    return {
        index: (
            boundary_sum / np.maximum(count, 1.0),
            function_sum / np.maximum(count[:, None], 1.0),
        )
        for index, (boundary_sum, function_sum, count) in sums.items()
    }


def _peak_times(
    probabilities: np.ndarray,
    frame_rate: float,
    threshold: float,
    min_distance_seconds: float = 1.0,
) -> np.ndarray:
    peaks, _ = find_peaks(
        probabilities,
        height=threshold,
        distance=max(1, int(round(min_distance_seconds * frame_rate))),
    )
    return (peaks.astype(np.float64) + 0.5) / frame_rate


def _boundary_f1(
    reference: np.ndarray,
    estimated: np.ndarray,
    window: float,
    trim: bool = False,
) -> float:
    duration = float(reference[-1]) if reference.size else 0.0
    if duration <= 0.0:
        return float("nan")
    reference = reference[(reference > 0.0) & (reference < duration)]
    estimated = estimated[(estimated > 0.0) & (estimated < duration)]
    reference_intervals = mir_eval.util.boundaries_to_intervals(
        np.concatenate([[0.0], reference, [duration]])
    )
    estimated_intervals = mir_eval.util.boundaries_to_intervals(
        np.concatenate([[0.0], estimated, [duration]])
    )
    return float(
        mir_eval.segment.detection(
            reference_intervals,
            estimated_intervals,
            window=window,
            trim=trim,
        )[2]
    )


def _select_threshold(
    tracks: list[ProbeTrack],
    predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_rate: float,
    thresholds: Iterable[float],
    trim_boundaries: bool,
) -> float:
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in thresholds:
        scores = [
            _boundary_f1(
                tracks[index].boundary_times,
                _peak_times(boundary, frame_rate, threshold),
                3.0,
                trim=trim_boundaries,
            )
            for index, (boundary, _) in predictions.items()
        ]
        score = float(np.nanmean(scores))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _segment_function_predictions(
    probabilities: np.ndarray,
    boundary_times: np.ndarray,
    frame_rate: float,
) -> np.ndarray:
    n_frames = probabilities.shape[0]
    frame_boundaries = np.rint(boundary_times * frame_rate - 0.5).astype(np.int64)
    frame_boundaries = frame_boundaries[(frame_boundaries > 0) & (frame_boundaries < n_frames)]
    edges = np.concatenate([[0], frame_boundaries, [n_frames]])
    output = np.zeros(n_frames, dtype=np.int64)
    for start, stop in zip(edges[:-1], edges[1:]):
        output[start:stop] = int(np.argmax(probabilities[start:stop].mean(axis=0)))
    return output


def _frame_labels_to_segments(
    labels: np.ndarray,
    frame_rate: float,
) -> tuple[np.ndarray, list[str]]:
    if labels.size == 0:
        return np.empty((0, 2), dtype=np.float64), []
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    edges = np.concatenate([[0], changes, [labels.size]])
    intervals = np.column_stack([edges[:-1], edges[1:]]).astype(np.float64) / frame_rate
    segment_labels = [str(labels[start]) for start in edges[:-1]]
    return intervals, segment_labels


def _pairwise_f1(
    reference: np.ndarray,
    estimated: np.ndarray,
    frame_rate: float,
) -> float:
    if reference.size == 0:
        return float("nan")
    reference_intervals, reference_labels = _frame_labels_to_segments(reference, frame_rate)
    estimated_intervals, estimated_labels = _frame_labels_to_segments(estimated, frame_rate)
    return float(
        mir_eval.segment.pairwise(
            reference_intervals,
            reference_labels,
            estimated_intervals,
            estimated_labels,
        )[2]
    )


def _evaluate_predictions(
    tracks: list[ProbeTrack],
    predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    vocabulary: dict[str, int],
    frame_rate: float,
    threshold: float,
    trim_boundaries: bool,
) -> dict[str, float]:
    hr_0p5: list[float] = []
    hr_3: list[float] = []
    pwf: list[float] = []
    accuracy: list[float] = []
    for index, (boundary_probability, function_probability) in predictions.items():
        track = tracks[index]
        estimated_boundaries = _peak_times(boundary_probability, frame_rate, threshold)
        hr_0p5.append(
            _boundary_f1(
                track.boundary_times,
                estimated_boundaries,
                0.5,
                trim=trim_boundaries,
            )
        )
        hr_3.append(
            _boundary_f1(
                track.boundary_times,
                estimated_boundaries,
                3.0,
                trim=trim_boundaries,
            )
        )
        reference_labels = _segment_labels(
            track.function_times,
            track.function_labels,
            track.features.shape[0],
            frame_rate,
        )
        reference = np.asarray(
            [vocabulary.get(str(label), vocabulary["unknown"]) for label in reference_labels],
            dtype=np.int64,
        )
        estimated = _segment_function_predictions(
            function_probability,
            estimated_boundaries,
            frame_rate,
        )
        pwf.append(_pairwise_f1(reference, estimated, frame_rate))
        accuracy.append(float(np.mean(reference == estimated)))
    return {
        "hr_0p5_f": float(np.nanmean(hr_0p5)),
        "hr_3_f": float(np.nanmean(hr_3)),
        "pwf": float(np.nanmean(pwf)),
        "accuracy": float(np.nanmean(accuracy)),
    }


def run_structure_probe(
    model,
    manifest_csv: str | Path,
    folds_csv: str | Path,
    feature_root: str | Path,
    annotation_root: str | Path,
    dataset: str,
    probe_inputs: Iterable[str] = PROBE_INPUTS,
    frame_rate: float = 1.0,
    window_seconds: float = 30.0,
    hop_seconds: float = 30.0,
    position_dim: int = 32,
    batch_size: int = 8,
    epochs: int = 100,
    warmup_epochs: int = 5,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    threshold_grid: Iterable[float] = tuple(np.linspace(0.1, 0.9, 9)),
    num_folds: int = 8,
    num_workers: int = 0,
    max_tracks: int | None = None,
    salami_boundary_layer: str = "uppercase",
    trim_boundaries: bool = False,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    frame_rate = parse_frame_rate(frame_rate)
    device = torch.device(device)
    tracks = load_probe_tracks(
        manifest_csv,
        folds_csv,
        feature_root,
        annotation_root,
        dataset,
        frame_rate,
        salami_boundary_layer,
        max_tracks,
    )
    extract_track_tokens(model, tracks, device)
    fold_rows: list[dict[str, float | int | str]] = []

    for probe_input in probe_inputs:
        for test_fold in range(num_folds):
            val_fold = (test_fold - 1) % num_folds
            test_indices = [index for index, track in enumerate(tracks) if track.fold == test_fold]
            val_indices = [index for index, track in enumerate(tracks) if track.fold == val_fold]
            train_indices = [
                index
                for index, track in enumerate(tracks)
                if track.fold not in {test_fold, val_fold}
            ]
            if not train_indices or not val_indices or not test_indices:
                continue
            probe, vocabulary = _train_one_fold(
                tracks,
                train_indices,
                val_indices,
                probe_input,
                frame_rate,
                window_seconds,
                hop_seconds,
                position_dim,
                batch_size,
                epochs,
                warmup_epochs,
                learning_rate,
                weight_decay,
                device,
                num_workers,
            )
            val_predictions = _predict_tracks(
                probe,
                tracks,
                val_indices,
                vocabulary,
                probe_input,
                frame_rate,
                window_seconds,
                hop_seconds,
                position_dim,
                batch_size,
                device,
                num_workers,
            )
            threshold = _select_threshold(
                tracks,
                val_predictions,
                frame_rate,
                threshold_grid,
                trim_boundaries,
            )
            test_predictions = _predict_tracks(
                probe,
                tracks,
                test_indices,
                vocabulary,
                probe_input,
                frame_rate,
                window_seconds,
                hop_seconds,
                position_dim,
                batch_size,
                device,
                num_workers,
            )
            metrics = _evaluate_predictions(
                tracks,
                test_predictions,
                vocabulary,
                frame_rate,
                threshold,
                trim_boundaries,
            )
            fold_rows.append(
                {
                    "dataset": dataset.lower(),
                    "boundary_layer": salami_boundary_layer if dataset.lower() == "salami" else "functions",
                    "probe_input": probe_input,
                    "fold": test_fold,
                    "threshold": threshold,
                    "train_tracks": len(train_indices),
                    "val_tracks": len(val_indices),
                    "test_tracks": len(test_indices),
                    **metrics,
                }
            )

    frame = pd.DataFrame(fold_rows)
    metrics: dict[str, float] = {}
    for probe_input in probe_inputs:
        subset = frame[frame["probe_input"] == probe_input]
        if subset.empty:
            continue
        prefix = f"paper.structure_probing/test/{dataset.lower()}/{probe_input}"
        for metric in ("hr_0p5_f", "hr_3_f", "pwf", "accuracy"):
            metrics[f"{prefix}/{metric}"] = float(subset[metric].mean())
            metrics[f"{prefix}/{metric}_std"] = float(subset[metric].std(ddof=0))
    return metrics, fold_rows
