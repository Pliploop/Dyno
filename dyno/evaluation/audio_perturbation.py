"""Controlled audio- and latent-domain perturbations for frozen checkpoints."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from dyno.callbacks.annotation_free import _dtw_distance
from dyno.evaluation.temporal import compute_mspf


PRESERVING_TRANSFORMS = ("gain", "pitch_shift", "time_stretch")
DISRUPTIVE_TRANSFORMS = ("chunk_shuffle", "reverse", "section_delete")


def transform_audio(
    audio: np.ndarray,
    sample_rate: int,
    condition: str,
    seed: int,
    gain_db: float = -6.0,
    pitch_steps: float = 2.0,
    stretch_rate: float = 1.1,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if condition == "gain":
        return audio * np.float32(10.0 ** (gain_db / 20.0))
    if condition == "pitch_shift":
        return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=pitch_steps).astype(np.float32)
    if condition == "time_stretch":
        return librosa.effects.time_stretch(y=audio, rate=stretch_rate).astype(np.float32)
    raise ValueError(f"{condition!r} is not an audio-domain perturbation")


def transform_embedding_sequence(
    sequence: np.ndarray,
    condition: str,
    seed: int,
    chunk_frames: int = 15,
) -> np.ndarray:
    """Apply order and deletion interventions directly to a frozen sequence."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected [time, feature] sequence, got {sequence.shape}")
    if condition == "reverse":
        return sequence[::-1].copy()

    rng = np.random.default_rng(seed)
    chunk_frames = max(1, int(chunk_frames))
    chunks = [
        sequence[start : start + chunk_frames]
        for start in range(0, len(sequence), chunk_frames)
    ]
    if condition == "chunk_shuffle":
        order = rng.permutation(len(chunks))
        return np.concatenate([chunks[index] for index in order]).astype(np.float32)
    if condition == "section_delete":
        if len(chunks) <= 2:
            start = len(sequence) // 3
            stop = 2 * len(sequence) // 3
            return np.concatenate([sequence[:start], sequence[stop:]]).astype(np.float32)
        delete_index = int(rng.integers(1, len(chunks) - 1))
        return np.concatenate([chunk for index, chunk in enumerate(chunks) if index != delete_index]).astype(
            np.float32
        )
    raise ValueError(f"{condition!r} is not a latent-domain perturbation")


def extract_embedding_sequence(
    encoder,
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    window = int(round(window_seconds * sample_rate))
    hop = int(round(hop_seconds * sample_rate))
    if len(audio) < window:
        audio = np.pad(audio, (0, window - len(audio)))
    tensor = torch.from_numpy(audio).unfold(0, window, hop)
    parts = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device=device, dtype=torch.float32)
            output = encoder.extract_features(batch, return_dict=False)
            if output.ndim == 3:
                output = output.mean(dim=1)
            parts.append(output.cpu().float().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _dyno_tokens(model, sequence: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(sequence).to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(tensor.shape[:2], dtype=torch.bool, device=device)
    with torch.inference_mode():
        normalized = model.normalize_input(tensor)
        content = model.get_content_token(normalized, mask=mask)[0]
        temporal = model.encode(normalized, mask=mask)[2][0]
    return content.cpu().numpy(), temporal.cpu().numpy()


def _normalized_displacement(reference: np.ndarray, changed: np.ndarray) -> float:
    return float(np.linalg.norm(changed - reference) / max(np.linalg.norm(reference), 1e-8))


def _mspf(sequence: np.ndarray, n_points: int, max_frames: int) -> np.ndarray:
    indices = np.linspace(
        0,
        len(sequence) - 1,
        min(len(sequence), max_frames),
    ).round().astype(np.int64)
    return compute_mspf(
        torch.from_numpy(sequence[indices]),
        n_points=n_points,
        normalize=True,
    )


def run_audio_perturbation_evaluation(
    model,
    encoder,
    manifest_csv: str | Path,
    structure_root: str | Path,
    conditions: tuple[str, ...] = PRESERVING_TRANSFORMS + DISRUPTIVE_TRANSFORMS,
    max_tracks: int = 32,
    sample_rate: int = 24000,
    window_seconds: float = 10.0,
    hop_seconds: float = 1.0,
    batch_size: int = 32,
    latent_chunk_frames: int = 15,
    mspf_points: int = 100,
    mspf_max_frames: int = 256,
    seed: int = 142,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    device = torch.device(device)
    frame = pd.read_csv(manifest_csv, dtype={"track_id": str})
    rows: list[dict[str, float | str]] = []
    for row_index, row in enumerate(frame.head(max_tracks).itertuples(index=False)):
        audio_path = Path(str(row.audio_path))
        if not audio_path.is_absolute():
            audio_path = Path(structure_root) / audio_path
        if not audio_path.is_file():
            continue
        audio, source_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if source_rate != sample_rate:
            audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
        original_sequence = extract_embedding_sequence(
            encoder,
            audio,
            sample_rate,
            window_seconds,
            hop_seconds,
            batch_size,
            device,
        )
        original_content, original_temporal = _dyno_tokens(model, original_sequence, device)
        original_mspf = _mspf(original_sequence, mspf_points, mspf_max_frames)
        for condition_index, condition in enumerate(conditions):
            transform_seed = seed + row_index * 100 + condition_index
            if condition in PRESERVING_TRANSFORMS:
                changed_audio = transform_audio(
                    audio,
                    sample_rate,
                    condition,
                    transform_seed,
                )
                changed_sequence = extract_embedding_sequence(
                    encoder,
                    changed_audio,
                    sample_rate,
                    window_seconds,
                    hop_seconds,
                    batch_size,
                    device,
                )
                domain = "audio"
            elif condition in DISRUPTIVE_TRANSFORMS:
                changed_sequence = transform_embedding_sequence(
                    original_sequence,
                    condition,
                    transform_seed,
                    chunk_frames=latent_chunk_frames,
                )
                domain = "latent"
            else:
                raise ValueError(f"Unknown perturbation {condition!r}")
            changed_content, changed_temporal = _dyno_tokens(model, changed_sequence, device)
            changed_mspf = _mspf(changed_sequence, mspf_points, mspf_max_frames)
            rows.append(
                {
                    "track_id": str(row.track_id),
                    "condition": condition,
                    "domain": domain,
                    "content_displacement": _normalized_displacement(
                        original_content,
                        changed_content,
                    ),
                    "temporal_displacement": _normalized_displacement(
                        original_temporal,
                        changed_temporal,
                    ),
                    "mspf_dtw": _dtw_distance(original_mspf, changed_mspf),
                }
            )

    result = pd.DataFrame(rows)
    metrics: dict[str, float] = {}
    for condition in conditions:
        subset = result[result["condition"] == condition]
        if subset.empty:
            continue
        for representation in ("content", "temporal"):
            metrics[
                f"paper.perturbation_sensitivity/test/{representation}/displacement/{condition}"
            ] = float(subset[f"{representation}_displacement"].mean())
        metrics[f"paper.mspf_validation/test/mspf_dtw/{condition}"] = float(subset["mspf_dtw"].mean())

    for representation in ("content", "temporal"):
        preserving = result[result["condition"].isin(PRESERVING_TRANSFORMS)][
            f"{representation}_displacement"
        ].mean()
        disruptive = result[result["condition"].isin(DISRUPTIVE_TRANSFORMS)][
            f"{representation}_displacement"
        ].mean()
        metrics[
            f"paper.perturbation_sensitivity/test/{representation}/separation_ratio"
        ] = float(disruptive / max(float(preserving), 1e-8))
    return metrics, rows
