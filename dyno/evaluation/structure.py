"""Structural annotation evaluation utilities for CoverX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructureTargets:
    track_ids: list[str]
    datasets: list[str]
    form_strings: list[str]
    section_count: np.ndarray
    unique_label_count: np.ndarray
    repetition_rate: np.ndarray
    return_a: np.ndarray
    bridge: np.ndarray
    through: np.ndarray
    boundaries: list[np.ndarray]
    labels: list[list[str]]
    feature_paths: list[str]


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance without external dependencies."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_form_distance(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return float(levenshtein(a, b) / denom)


def pairwise_form_distance(forms: list[str]) -> np.ndarray:
    n = len(forms)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = normalized_form_distance(forms[i], forms[j])
            out[i, j] = out[j, i] = d
    return out


def collapse_labels_to_form(labels: Iterable[str]) -> str:
    """Collapse consecutive duplicate labels and map symbols to A, B, C, ..."""
    collapsed: list[str] = []
    for label in labels:
        label = str(label).strip()
        if label and (not collapsed or collapsed[-1] != label):
            collapsed.append(label)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mapping: dict[str, str] = {}
    form = []
    for label in collapsed:
        if label not in mapping:
            idx = len(mapping)
            mapping[label] = alphabet[idx] if idx < len(alphabet) else f"X{idx}"
        form.append(mapping[label])
    return "".join(form)


def normalize_structure_label(label: str, dataset: str = "unknown") -> str | None:
    """Dataset-aware cleanup for section labels used in form metrics.

    This deliberately targets coarse musical form rather than literal annotation
    tokens. Non-musical padding labels are dropped; SALAMI variants like A' are
    folded into A; Harmonix function labels are lowercased and obvious numbered
    or compound variants are folded.
    """
    raw = str(label).strip()
    if not raw:
        return None
    dataset = dataset.lower()
    low = raw.lower().strip()
    low = low.replace("_", "").replace("-", "").replace(" ", "")
    if low in {"silence", "silent", "end", "fadein", "fadeout"}:
        return None

    if dataset == "salami":
        base = re.sub(r"['’`]+$", "", raw.strip())
        base = re.sub(r"\d+$", "", base)
        return base.upper() if base else None

    if dataset == "harmonix":
        base = re.sub(r"\d+$", "", low)
        aliases = {
            "introverse": "verse",
            "miniverse": "verse",
            "altchorus": "chorus",
            "quietchorus": "chorus",
            "chorusinst": "chorus",
            "chorusinstrumental": "chorus",
            "instchorus": "chorus",
            "postchorus": "postchorus",
            "prechorus": "prechorus",
            "prechor": "prechorus",
            "bre": "bridge",
            "gtr": "solo",
            "inst": "instrumental",
        }
        return aliases.get(base, base) or None

    return low


def clean_structure_segments(
    boundaries: np.ndarray,
    labels: list[str],
    dataset: str = "unknown",
    mode: str = "dataset",
) -> tuple[np.ndarray, list[str]]:
    """Clean labels and compact boundaries for structure metrics."""
    if mode in (None, "none", "raw", False):
        return np.asarray(boundaries, dtype=np.float32), labels
    cleaned_labels: list[str] = []
    keep_indices: list[int] = []
    for i, label in enumerate(labels):
        cleaned = normalize_structure_label(label, dataset=dataset)
        if cleaned is not None:
            keep_indices.append(i)
            cleaned_labels.append(cleaned)
    if not cleaned_labels:
        return np.zeros(0, dtype=np.float32), []

    boundaries = np.asarray(boundaries, dtype=np.float32)
    if boundaries.size == len(labels) + 1:
        durations = np.maximum(boundaries[1:] - boundaries[:-1], 1e-6)
        kept_durations = durations[np.asarray(keep_indices, dtype=np.int64)]
        compact = np.concatenate([[0.0], np.cumsum(kept_durations)]).astype(np.float32)
        return compact, cleaned_labels
    if boundaries.size >= len(labels):
        starts = boundaries[: len(labels)]
        if len(starts) >= 2:
            diffs = np.diff(starts)
            fallback = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        else:
            fallback = 1.0
        ends = np.concatenate([starts[1:], [starts[-1] + fallback]])
        durations = np.maximum(ends - starts, 1e-6)
        kept_durations = durations[np.asarray(keep_indices, dtype=np.int64)]
        compact = np.concatenate([[0.0], np.cumsum(kept_durations)]).astype(np.float32)
        return compact, cleaned_labels
    return np.arange(len(cleaned_labels) + 1, dtype=np.float32), cleaned_labels


def global_form_attributes(labels: Iterable[str]) -> dict[str, float]:
    form = collapse_labels_to_form(labels)
    n_sections = len(form)
    n_unique = len(set(form))
    repetition_rate = 0.0 if n_sections == 0 else 1.0 - (n_unique / n_sections)
    return {
        "section_count": float(n_sections),
        "unique_label_count": float(n_unique),
        "repetition_rate": float(repetition_rate),
        "return_a": float(n_sections >= 3 and form[0] == "A" and "A" in form[2:]),
        "bridge": float(n_sections >= 3 and any(ch not in {"A", "B"} for ch in form)),
        "through": float(n_sections > 0 and n_unique == n_sections),
    }


def parse_sequence_field(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value)
    if not text:
        return []
    if "|" in text:
        return [part for part in text.split("|") if part != ""]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def parse_float_sequence_field(value) -> np.ndarray:
    vals = parse_sequence_field(value)
    if not vals:
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(v) for v in vals], dtype=np.float32)


def same_section_matrix(boundaries: np.ndarray, labels: list[str], n_frames: int) -> np.ndarray:
    """Build an annotation same-section matrix on a normalized frame grid."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if len(labels) == 0:
        return np.eye(n_frames, dtype=np.float32)

    boundaries = np.asarray(boundaries, dtype=np.float32)
    if boundaries.size == 0:
        section_idx = np.zeros(n_frames, dtype=np.int64)
    else:
        duration = float(max(boundaries[-1], 1e-6))
        starts = boundaries[:-1] if boundaries.size == len(labels) + 1 else boundaries[: len(labels)]
        starts = np.clip(starts / duration, 0.0, 1.0)
        grid = np.linspace(0.0, 1.0, n_frames, endpoint=False, dtype=np.float32)
        section_idx = np.searchsorted(starts, grid, side="right") - 1
        section_idx = np.clip(section_idx, 0, len(labels) - 1)
    return (section_idx[:, None] == section_idx[None, :]).astype(np.float32)


def pairwise_ssm_distance(
    boundaries: list[np.ndarray],
    labels: list[list[str]],
    n_frames: int = 64,
) -> np.ndarray:
    matrices = [same_section_matrix(b, l, n_frames) for b, l in zip(boundaries, labels)]
    n = len(matrices)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(matrices[i] - matrices[j], ord="fro"))
            out[i, j] = out[j, i] = d
    return out


def cosine_distance_matrix(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, eps)
    d = 1.0 - (x @ x.T)
    np.fill_diagonal(d, np.inf)
    return d.astype(np.float32)


def retrieval_metrics(
    rep_distance: np.ndarray,
    form_distance: np.ndarray,
    ks: tuple[int, ...] = (1, 5),
    epsilon: float = 0.25,
    candidate_mask: np.ndarray | None = None,
) -> dict[str, float]:
    n = rep_distance.shape[0]
    out: dict[str, float] = {}
    ranks = {}
    for i in range(n):
        d = rep_distance[i].copy()
        d[i] = np.inf
        if candidate_mask is not None:
            d[~candidate_mask[i]] = np.inf
        ranks[i] = np.argsort(d)

    for k in ks:
        d_vals = []
        recalls = []
        for i in range(n):
            nn = ranks[i][:k]
            nn = nn[np.isfinite(rep_distance[i, nn])]
            if nn.size == 0:
                continue
            fd = form_distance[i, nn]
            d_vals.append(float(fd.mean()))
            recalls.append(float(fd.min() < epsilon))
        out[f"D@{k}"] = float(np.mean(d_vals)) if d_vals else float("nan")
        out[f"FR@{k}"] = float(np.mean(recalls)) if recalls else float("nan")
    return out


def content_control_mask(content_distance: np.ndarray, pool_size: int) -> np.ndarray:
    n = content_distance.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        d = content_distance[i].copy()
        d[i] = np.inf
        nn = np.argsort(d)[: min(pool_size, n - 1)]
        mask[i, nn] = True
    return mask


def spearman_from_distance_matrices(a: np.ndarray, b: np.ndarray) -> float:
    iu = np.triu_indices_from(a, k=1)
    x = a[iu]
    y = b[iu]
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x[keep], y[keep]).correlation)
    except Exception:
        return float(pd.Series(x[keep]).corr(pd.Series(y[keep]), method="spearman"))


def probe_scores(
    reps: np.ndarray,
    targets: StructureTargets,
    seed: int = 0,
    test_size: float = 0.25,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    out: dict[str, float] = {}
    n = reps.shape[0]
    if n < 8:
        return out

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(reps[train_idx])
    x_test = scaler.transform(reps[test_idx])

    continuous = {
        "section_count_R2": targets.section_count,
        "unique_label_count_R2": targets.unique_label_count,
        "repetition_rate_R2": targets.repetition_rate,
    }
    for name, y in continuous.items():
        model = Ridge(alpha=1.0)
        model.fit(x_train, y[train_idx])
        out[name] = float(r2_score(y[test_idx], model.predict(x_test)))

    categorical = {
        "return_a_F1": targets.return_a,
        "bridge_F1": targets.bridge,
        "through_F1": targets.through,
    }
    for name, y in categorical.items():
        y = y.astype(int)
        if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
            out[name] = float("nan")
            continue
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x_train, y[train_idx])
        out[name] = float(f1_score(y[test_idx], model.predict(x_test), average="macro"))
    return out


def load_structure_targets(manifest_csv: str | Path, clean: str = "dataset") -> StructureTargets:
    df = pd.read_csv(manifest_csv)
    required = {"track_id", "dataset", "feature_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Structure manifest is missing required columns: {sorted(missing)}")

    raw_labels = [parse_sequence_field(v) for v in df.get("labels", pd.Series([""] * len(df)))]
    raw_boundaries = [parse_float_sequence_field(v) for v in df.get("boundaries", pd.Series([""] * len(df)))]
    datasets = df["dataset"].astype(str).tolist()
    cleaned = [
        clean_structure_segments(boundary, label, dataset=dataset, mode=clean)
        for boundary, label, dataset in zip(raw_boundaries, raw_labels, datasets)
    ]
    boundaries = [row[0] for row in cleaned]
    labels = [row[1] for row in cleaned]
    forms = [collapse_labels_to_form(seq) for seq in labels]
    attrs = [global_form_attributes(seq) for seq in labels]

    def attr_col(col: str) -> np.ndarray:
        return np.asarray([row[col] for row in attrs], dtype=np.float32)

    return StructureTargets(
        track_ids=df["track_id"].astype(str).tolist(),
        datasets=datasets,
        form_strings=forms,
        section_count=attr_col("section_count"),
        unique_label_count=attr_col("unique_label_count"),
        repetition_rate=attr_col("repetition_rate"),
        return_a=attr_col("return_a"),
        bridge=attr_col("bridge"),
        through=attr_col("through"),
        boundaries=boundaries,
        labels=labels,
        feature_paths=df["feature_path"].astype(str).tolist(),
    )
