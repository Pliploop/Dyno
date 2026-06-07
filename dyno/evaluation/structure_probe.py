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
    "content",
    "temporal",
    "content_temporal",
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


class FullTrackProbeDataset(Dataset):
    def __init__(
        self,
        tracks: list[ProbeTrack],
        track_indices: list[int],
        label_to_index: dict[str, int],
        frame_rate: float = 1.0,
    ):
        self.tracks = tracks
        self.track_indices = track_indices
        self.label_to_index = label_to_index
        self.frame_rate = frame_rate

    def __len__(self) -> int:
        return len(self.track_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_index = self.track_indices[index]
        track = self.tracks[track_index]
        n_frames = track.features.shape[0]
        labels = _segment_labels(
            track.function_times,
            track.function_labels,
            n_frames,
            self.frame_rate,
        )
        y_function = np.asarray(
            [self.label_to_index.get(str(label), self.label_to_index["unknown"]) for label in labels],
            dtype=np.int64,
        )
        y_boundary = _boundary_targets(
            track.boundary_times,
            n_frames,
            self.frame_rate,
        )
        return {
            "local": torch.from_numpy(track.features),
            "content": torch.from_numpy(track.content),
            "temporal": torch.from_numpy(track.temporal),
            "boundary": torch.from_numpy(y_boundary),
            "function": torch.from_numpy(y_function),
            "track_index": torch.tensor(track_index),
        }


def collate_full_tracks(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_frames = max(item["local"].shape[0] for item in items)
    batch_size = len(items)
    feature_dim = items[0]["local"].shape[-1]
    local = torch.zeros(batch_size, max_frames, feature_dim, dtype=torch.float32)
    boundary = torch.zeros(batch_size, max_frames, dtype=torch.float32)
    function = torch.zeros(batch_size, max_frames, dtype=torch.long)
    mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    for row, item in enumerate(items):
        n_frames = item["local"].shape[0]
        local[row, :n_frames] = item["local"]
        boundary[row, :n_frames] = item["boundary"]
        function[row, :n_frames] = item["function"]
        mask[row, :n_frames] = True
    return {
        "local": local,
        "content": torch.stack([item["content"] for item in items]),
        "temporal": torch.stack([item["temporal"] for item in items]),
        "boundary": boundary,
        "function": function,
        "mask": mask,
        "track_index": torch.stack([item["track_index"] for item in items]),
    }


class AdaLNProbeBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if condition is None:
            shift_sa = scale_sa = shift_ffn = scale_ffn = 0.0
            gate_sa = gate_ffn = 1.0
        else:
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = (
                self.modulation(condition).chunk(6, dim=-1)
            )
            shift_sa = shift_sa.unsqueeze(1)
            scale_sa = scale_sa.unsqueeze(1)
            gate_sa = gate_sa.unsqueeze(1)
            shift_ffn = shift_ffn.unsqueeze(1)
            scale_ffn = scale_ffn.unsqueeze(1)
            gate_ffn = gate_ffn.unsqueeze(1)

        normalized = (1.0 + scale_sa) * self.norm1(x) + shift_sa
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x = x + gate_sa * attended
        normalized = (1.0 + scale_ffn) * self.norm2(x) + shift_ffn
        return x + gate_ffn * self.ffn(normalized)


class AttentionStructureProbe(nn.Module):
    def __init__(
        self,
        probe_input: str,
        local_dim: int,
        content_dim: int,
        temporal_dim: int,
        n_functions: int,
        model_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if probe_input not in PROBE_INPUTS:
            raise ValueError(f"Unknown probe input {probe_input!r}")
        self.probe_input = probe_input
        self.local_projection = nn.Linear(local_dim, model_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        condition_dim = {
            "local": 0,
            "content": content_dim,
            "temporal": temporal_dim,
            "content_temporal": content_dim + temporal_dim,
        }[probe_input]
        self.condition_projection = (
            None if condition_dim == 0 else nn.Linear(condition_dim, model_dim)
        )
        self.block = AdaLNProbeBlock(model_dim, num_heads, ffn_dim, dropout)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, n_functions + 1)

    def forward(
        self,
        local: torch.Tensor,
        content: torch.Tensor,
        temporal: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.probe_input == "local":
            x = self.local_projection(local)
            condition = None
        else:
            x = self.mask_token.expand(local.shape[0], local.shape[1], -1)
            raw_condition = {
                "content": content,
                "temporal": temporal,
                "content_temporal": torch.cat([content, temporal], dim=-1),
            }[self.probe_input]
            condition = self.condition_projection(raw_condition)
        position = torch.from_numpy(
            _position_encoding(local.shape[1], x.shape[-1])
        ).to(device=x.device, dtype=x.dtype)
        x = x + position.unsqueeze(0)
        x = self.block(x, mask, condition)
        output = self.output(self.output_norm(x))
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
    batch_size: int,
    epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    weight_decay: float,
    model_dim: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    device: torch.device,
    num_workers: int,
) -> tuple[AttentionStructureProbe, dict[str, int]]:
    vocabulary = _function_vocabulary(tracks, train_indices)
    train_data = FullTrackProbeDataset(
        tracks,
        train_indices,
        vocabulary,
        frame_rate,
    )
    val_data = FullTrackProbeDataset(
        tracks,
        val_indices,
        vocabulary,
        frame_rate,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_full_tracks,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_full_tracks,
    )
    example = train_data[0]
    probe = AttentionStructureProbe(
        probe_input=probe_input,
        local_dim=example["local"].shape[-1],
        content_dim=example["content"].shape[-1],
        temporal_dim=example["temporal"].shape[-1],
        n_functions=len(vocabulary),
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    ).to(device)
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
            boundary_logits, function_logits = probe(
                batch["local"],
                batch["content"],
                batch["temporal"],
                batch["mask"],
            )
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
                boundary_logits, function_logits = probe(
                    batch["local"],
                    batch["content"],
                    batch["temporal"],
                    batch["mask"],
                )
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
    probe: AttentionStructureProbe,
    tracks: list[ProbeTrack],
    indices: list[int],
    vocabulary: dict[str, int],
    frame_rate: float,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = FullTrackProbeDataset(
        tracks,
        indices,
        vocabulary,
        frame_rate,
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_full_tracks,
    )
    predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    probe.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            boundary_logits, function_logits = probe(
                batch["local"],
                batch["content"],
                batch["temporal"],
                batch["mask"],
            )
            boundary = torch.sigmoid(boundary_logits).cpu().numpy()
            function = torch.softmax(function_logits, dim=-1).cpu().numpy()
            for row in range(boundary.shape[0]):
                track_index = int(batch["track_index"][row])
                valid = int(batch["mask"][row].sum())
                predictions[track_index] = (
                    boundary[row, :valid],
                    function[row, :valid],
                )
    return predictions


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
    batch_size: int = 8,
    epochs: int = 100,
    warmup_epochs: int = 5,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    model_dim: int = 128,
    num_heads: int = 4,
    ffn_dim: int = 256,
    dropout: float = 0.1,
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
                batch_size,
                epochs,
                warmup_epochs,
                learning_rate,
                weight_decay,
                model_dim,
                num_heads,
                ffn_dim,
                dropout,
                device,
                num_workers,
            )
            val_predictions = _predict_tracks(
                probe,
                tracks,
                val_indices,
                vocabulary,
                frame_rate,
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
                frame_rate,
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
                    "boundary_layer": (
                        salami_boundary_layer
                        if dataset.lower() == "salami"
                        else "functions"
                    ),
                    "probe_type": "full_track_attention",
                    "probe_input": probe_input,
                    "model_dim": model_dim,
                    "num_heads": num_heads,
                    "ffn_dim": ffn_dim,
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
