#!/usr/bin/env python
"""Gradio app for interactive structure MSPF inspection."""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dyno-mspf-matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/dyno-mspf-numba-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dyno.evaluation.structure import clean_structure_segments, parse_float_sequence_field, parse_sequence_field
from dyno.evaluation.temporal import compute_mspf
from scripts.build_structure_mspf_viz import (
    DEFAULT_MSPF_LAM,
    DEFAULT_MSPF_POINTS,
    DEFAULT_MSPF_POWER,
    DEFAULT_MSPF_SIGMA,
    DEFAULT_MSPF_WINDOW,
    DEFAULT_ROOT,
    DEFAULT_UMAP_COMPONENTS,
    FeatureRef,
    Track,
    _encoder_key,
    _label_colors,
    _rate_key,
    _semantic_percent,
)


BRIGHT = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]

PAPER_LINE_COLORS = ["#000000", "#3a3a3a", "#666666", "#8a8a8a", "#1f1f1f", "#575757"]
PAPER_LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1.5)), (0, (2.5, 1, 1, 1))]
SECTION_COLORS = [
    "#ffd166",
    "#06d6a0",
    "#4cc9f0",
    "#f72585",
    "#b8f35a",
    "#ff9f1c",
    "#c77dff",
    "#70e000",
    "#ffadad",
    "#9bf6ff",
    "#fdffb6",
    "#a0c4ff",
]


def _paper_label_colors(labels: list[str]) -> dict[str, str]:
    unique: list[str] = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return {label: SECTION_COLORS[i % len(SECTION_COLORS)] for i, label in enumerate(unique)}


def _dmspf_peak_times(series: list[tuple[np.ndarray, np.ndarray]], max_peaks: int = 5, min_distance: float = 0.045) -> list[float]:
    candidates: list[tuple[float, float]] = []
    for t, y in series:
        if len(y) < 3:
            continue
        for idx in range(1, len(y) - 1):
            if t[idx] < 0.05 or t[idx] > 0.95:
                continue
            if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1] and y[idx] >= 0.08:
                candidates.append((float(y[idx]), float(t[idx])))
    peaks: list[float] = []
    for _, x in sorted(candidates, reverse=True):
        if all(abs(x - seen) >= min_distance for seen in peaks):
            peaks.append(x)
        if len(peaks) >= max_peaks:
            break
    return sorted(peaks)


def _track_intervals(track: Track) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = list(track.labels)
    boundaries = np.asarray(track.boundaries, dtype=np.float32)
    if not labels:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, []
    if boundaries.size == 0:
        starts = np.linspace(0.0, 1.0, len(labels), endpoint=False, dtype=np.float32)
        ends = np.concatenate([starts[1:], [1.0]]).astype(np.float32)
        return starts, ends, labels

    duration = float(max(boundaries[-1], 1e-6))
    starts = boundaries[:-1] if boundaries.size == len(labels) + 1 else boundaries[: len(labels)]
    starts = np.clip(starts / duration, 0.0, 1.0).astype(np.float32)
    ends = np.concatenate([starts[1:], [1.0]]).astype(np.float32)
    return starts, ends, labels


def _interior_track(track: Track) -> tuple[Track, float, float]:
    starts, ends, labels = _track_intervals(track)
    valid = [
        idx
        for idx, (start, end) in enumerate(zip(starts, ends))
        if float(end - start) > 1e-5 and labels[idx].strip().lower() != "end"
    ]
    if len(valid) < 3:
        return track, 0.0, 1.0

    keep = valid[1:-1]
    trim_start = float(ends[valid[0]])
    trim_end = float(starts[valid[-1]])
    if trim_end <= trim_start:
        return track, 0.0, 1.0

    kept_labels: list[str] = []
    kept_starts: list[float] = []
    for idx in keep:
        if starts[idx] < trim_end and ends[idx] > trim_start:
            kept_labels.append(labels[idx])
            kept_starts.append(max(float(starts[idx]), trim_start))
    if not kept_labels:
        return track, 0.0, 1.0

    scale = max(trim_end - trim_start, 1e-6)
    new_boundaries = np.asarray([(start - trim_start) / scale for start in kept_starts] + [1.0], dtype=np.float32)
    trimmed = Track(dataset=track.dataset, track_id=track.track_id, boundaries=new_boundaries, labels=kept_labels)
    return trimmed, trim_start, trim_end


def _crop_feature_frames(arr: np.ndarray, trim_start: float, trim_end: float) -> np.ndarray:
    trim_start = float(np.clip(trim_start, 0.0, 1.0))
    trim_end = float(np.clip(trim_end, 0.0, 1.0))
    if trim_end <= trim_start or arr.shape[0] < 3:
        return arr
    start_idx = int(np.floor(trim_start * arr.shape[0]))
    end_idx = int(np.ceil(trim_end * arr.shape[0]))
    start_idx = max(0, min(start_idx, arr.shape[0] - 2))
    end_idx = max(start_idx + 2, min(end_idx, arr.shape[0]))
    return arr[start_idx:end_idx]


CSS = """
:root {
  color-scheme: light;
  --bg: #f5f7f4;
  --panel: rgba(255, 255, 255, 0.78);
  --panel-strong: rgba(255, 255, 255, 0.92);
  --text: #171917;
  --muted: #626b62;
  --line: rgba(110, 130, 112, 0.24);
  --green: #286145;
  --green-soft: #e7f3ec;
  --shadow: rgba(20, 28, 22, 0.07);
}
body, .gradio-container {
  background:
    radial-gradient(circle at 12% 0%, rgba(212, 229, 216, 0.8), transparent 32rem),
    linear-gradient(180deg, #f9fbf8 0%, var(--bg) 58%, #eef3ee 100%) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: 1480px !important;
  padding: 18px !important;
}
.app-title {
  border-bottom: 1px solid var(--line);
  margin-bottom: 12px;
  padding-bottom: 10px;
}
.app-title h1 {
  font-size: 1.42rem;
  line-height: 1.2;
  margin: 0;
  letter-spacing: 0;
}
.app-title p {
  color: var(--muted);
  font-size: 0.92rem;
  margin: 5px 0 0;
}
.panel, .plot-panel {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  box-shadow: 0 18px 42px var(--shadow) !important;
  backdrop-filter: blur(16px);
}
.plot-panel {
  padding: 12px !important;
}
button.primary {
  background: var(--green) !important;
  border-color: var(--green) !important;
  color: white !important;
}
button.secondary, button {
  border-radius: 8px !important;
}
label, .wrap label {
  color: var(--muted) !important;
  font-size: 0.82rem !important;
}
input, textarea, select {
  border-radius: 8px !important;
}
.status {
  color: var(--muted);
  font-size: 0.9rem;
}
"""


def _empty_update():
    return gr.update(choices=[], value=None)


@lru_cache(maxsize=8)
def discover_index(root: str) -> tuple[dict[tuple[str, str], tuple[FeatureRef, ...]], tuple[tuple[str, str], ...]]:
    root_path = Path(root).expanduser()
    feature_root = root_path / "features"
    features: dict[tuple[str, str], list[FeatureRef]] = {}
    for manifest in sorted((root_path / "manifests").glob("*/*/all.csv")):
        encoder = manifest.parent.parent.name
        rate = manifest.parent.name
        try:
            df = pd.read_csv(
                manifest,
                usecols=lambda col: col in {"dataset", "track_id", "feature_path"},
                dtype={"dataset": str, "track_id": str, "feature_path": str},
            )
        except Exception:
            continue
        if not {"dataset", "track_id", "feature_path"} <= set(df.columns):
            continue
        for row in df.itertuples(index=False):
            dataset = str(getattr(row, "dataset"))
            track_id = str(getattr(row, "track_id"))
            feature_path = Path(str(getattr(row, "feature_path")))
            if not feature_path.is_absolute():
                feature_path = feature_root / feature_path
            key = (dataset, track_id)
            features.setdefault(key, []).append(
                FeatureRef(dataset=dataset, encoder=encoder, rate=rate, track_id=track_id, path=feature_path)
            )
    frozen = {
        key: tuple(sorted(refs, key=lambda ref: (_encoder_key(ref.encoder), _rate_key(ref.rate))))
        for key, refs in features.items()
    }
    keys = tuple(sorted(frozen, key=lambda key: (key[0], key[1])))
    return frozen, keys


@lru_cache(maxsize=512)
def load_track(root: str, dataset: str, track_id: str, clean_labels: str) -> Track:
    root_path = Path(root).expanduser()
    for manifest in sorted((root_path / "manifests").glob("*/*/all.csv")):
        try:
            df = pd.read_csv(
                manifest,
                usecols=lambda col: col in {"dataset", "track_id", "boundaries", "labels"},
                dtype={"dataset": str, "track_id": str, "boundaries": str, "labels": str},
            )
        except Exception:
            continue
        required = {"dataset", "track_id", "boundaries", "labels"}
        if not required <= set(df.columns):
            continue
        hit = df[(df["dataset"] == dataset) & (df["track_id"] == track_id)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        raw_boundaries = parse_float_sequence_field(row.get("boundaries", ""))
        raw_labels = parse_sequence_field(row.get("labels", ""))
        boundaries, labels = clean_structure_segments(raw_boundaries, raw_labels, dataset=dataset, mode=clean_labels)
        return Track(dataset=dataset, track_id=track_id, boundaries=boundaries, labels=labels)
    raise ValueError(f"No annotations found for {dataset}/{track_id}")


@lru_cache(maxsize=128)
def load_feature(path: str, max_frames: int | None, trim_start: float = 0.0, trim_end: float = 1.0) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float32)
    arr = _crop_feature_frames(arr, trim_start, trim_end)
    if arr.ndim != 2:
        raise ValueError(f"Expected feature array with shape (T, D), got {arr.shape} at {path}")
    if max_frames is not None and arr.shape[0] > max_frames:
        idx = np.linspace(0, arr.shape[0] - 1, max_frames).round().astype(np.int64)
        arr = arr[idx]
    return arr.copy()


@lru_cache(maxsize=512)
def mspf_curve(path: str, max_frames: int, window: int, power: float, sigma: float, lam: float, points: int, dmspf_contrast: float, trim_start: float, trim_end: float):
    x = torch.from_numpy(load_feature(path, max_frames, trim_start, trim_end))
    curve = compute_mspf(
        x,
        window=min(window, max(1, x.shape[0] - 1)),
        sigma=sigma,
        lam=lam,
        power=power,
        absolute=False,
        n_points=points,
    )
    mspf = _semantic_percent(curve)
    t = np.linspace(0.0, 1.0, len(mspf), dtype=np.float32)
    dmspf = _semantic_percent(np.gradient(mspf, t).astype(np.float32))
    dmspf = np.power(dmspf, max(float(dmspf_contrast), 1e-6)).astype(np.float32)
    return t, mspf, dmspf


@lru_cache(maxsize=256)
def umap_strip(path: str, max_frames: int, components: int, seed: int, trim_start: float, trim_end: float):
    x_np = load_feature(path, max_frames, trim_start, trim_end)
    if x_np.shape[0] < 3:
        return np.full((1, max(1, x_np.shape[0]), 3), 127, dtype=np.uint8)
    from umap import UMAP

    n_components = max(3, min(int(components), x_np.shape[0] - 1))
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=min(15, x_np.shape[0] - 1),
        min_dist=0.05,
        metric="cosine",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")
        z = reducer.fit_transform(x_np)[:, :3]

    rgb = np.zeros_like(z, dtype=np.float32)
    for dim in range(3):
        col = z[:, dim]
        lo, hi = np.nanpercentile(col, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            rgb[:, dim] = 0.5
        else:
            rgb[:, dim] = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    return (rgb[None, :, :] * 255).astype(np.uint8)


def refs_for_track(root: str, dataset: str, track_id: str) -> list[FeatureRef]:
    features, _ = discover_index(root)
    return list(features.get((dataset, track_id), ()))


def _plot_annotations(ax, track: Track) -> None:
    labels = track.labels
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("Labels", fontsize=8)
    if not labels:
        return
    colors = _paper_label_colors(labels)
    boundaries = np.asarray(track.boundaries, dtype=np.float32)
    if boundaries.size == 0:
        starts = np.linspace(0.0, 1.0, len(labels), endpoint=False, dtype=np.float32)
    else:
        duration = float(max(boundaries[-1], 1e-6))
        starts = boundaries[:-1] if boundaries.size == len(labels) + 1 else boundaries[: len(labels)]
        starts = np.clip(starts / duration, 0.0, 1.0)
    ends = np.concatenate([starts[1:], [1.0]]).astype(np.float32)
    for start, end, label in zip(starts, ends, labels):
        width = max(float(end - start), 1e-6)
        ax.broken_barh([(float(start), width)], (0.08, 0.84), facecolors=colors[label], edgecolors="white", linewidth=0.6)
        if width >= 0.045:
            ax.text(float(start + width / 2), 0.5, label, ha="center", va="center", fontsize=7.2, color="black", clip_on=True)


def build_figure(track: Track, selected_refs: list[FeatureRef], window: int, power: float, dmspf_contrast: float, sigma: float, lam: float, points: int, max_frames: int, umap_frames: int, umap_components: int, trim_start: float, trim_end: float):
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.2, 4.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 0.9, 0.28, 0.24], "hspace": 0.08},
    )
    legend_handles: list[Line2D] = []
    umap_rows = []
    umap_labels = []
    dmspf_series: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, ref in enumerate(selected_refs):
        color = PAPER_LINE_COLORS[idx % len(PAPER_LINE_COLORS)]
        linestyle = PAPER_LINE_STYLES[idx % len(PAPER_LINE_STYLES)]
        t, mspf, dmspf = mspf_curve(str(ref.path), max_frames, window, power, sigma, lam, points, dmspf_contrast, trim_start, trim_end)
        name = f"{ref.encoder} @ {ref.rate}"
        axes[0].plot(t, mspf, color=color, linestyle=linestyle, linewidth=1.35)
        axes[1].plot(t, dmspf, color=color, linestyle=linestyle, linewidth=1.2)
        legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=name))
        dmspf_series.append((t, dmspf))
        umap_rows.append(umap_strip(str(ref.path), umap_frames, umap_components, seed=0, trim_start=trim_start, trim_end=trim_end)[0])
        umap_labels.append(name)

    peak_times = _dmspf_peak_times(dmspf_series)
    for peak_t in peak_times:
        for ax in axes:
            ax.axvline(peak_t, color="#8d8d8d", linewidth=0.75, linestyle=(0, (3, 3)), alpha=0.78, zorder=0)

    if umap_rows:
        width = max(row.shape[0] for row in umap_rows)
        image_rows = []
        for row in umap_rows:
            xp = np.linspace(0.0, 1.0, row.shape[0], dtype=np.float32)
            xq = np.linspace(0.0, 1.0, width, dtype=np.float32)
            image_rows.append(np.stack([np.interp(xq, xp, row[:, dim]) for dim in range(3)], axis=-1))
        image = np.stack(image_rows, axis=0).astype(np.uint8)
        axes[2].imshow(image, aspect="auto", extent=[0.0, 1.0, len(image_rows), 0.0], interpolation="nearest")
        axes[2].set_yticks(np.arange(len(umap_labels)) + 0.5)
        axes[2].set_yticklabels(umap_labels, fontsize=6.8)
    else:
        axes[2].set_yticks([])
    axes[2].set_ylabel("UMAP", fontsize=8)
    _plot_annotations(axes[3], track)

    for ax, ylabel in zip(axes[:2], ["MSPF (%)", "dMSPF"]):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", color="#bdbdbd", linestyle=(0, (2.5, 3)), linewidth=0.55, alpha=0.75)
        ax.grid(axis="y", color="#e2e2e2", linestyle=(0, (2.5, 3)), linewidth=0.45, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#1a1a1a")
        ax.spines["bottom"].set_color("#1a1a1a")
        ax.tick_params(length=2.5, width=0.7, colors="#1a1a1a")
    for ax in axes[2:]:
        ax.set_xlim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#1a1a1a")
        ax.tick_params(length=2.5, width=0.7, colors="#1a1a1a")
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="upper left", ncol=min(3, len(legend_handles)), frameon=False, handlelength=2.5, columnspacing=1.1)
    axes[3].set_xlabel("Normalized time", fontsize=8.5)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.105, right=0.992, top=0.985, bottom=0.19, hspace=0.08)
    return fig


def _save_figure(fig, fmt: str) -> str:
    handle = tempfile.NamedTemporaryFile(prefix="dyno_mspf_", suffix=f".{fmt}", delete=False)
    handle.close()
    fig.savefig(handle.name, format=fmt, bbox_inches="tight", dpi=160)
    return handle.name


def _choices(root: str):
    features, keys = discover_index(root)
    datasets = sorted({dataset for dataset, _ in keys})
    return features, keys, datasets


def load_index(root: str):
    discover_index.cache_clear()
    load_track.cache_clear()
    load_feature.cache_clear()
    mspf_curve.cache_clear()
    umap_strip.cache_clear()
    try:
        _, keys, datasets = _choices(root)
    except Exception as exc:
        return _empty_update(), _empty_update(), _empty_update(), _empty_update(), f"Index load failed: {exc}"
    if not datasets:
        return _empty_update(), _empty_update(), _empty_update(), _empty_update(), "No manifest-backed features found."
    dataset = datasets[0]
    tracks = sorted(track_id for ds, track_id in keys if ds == dataset)
    track = tracks[0] if tracks else None
    encoders, rates, encoder, rate = selection_options(root, dataset, track)
    return (
        gr.update(choices=datasets, value=dataset),
        gr.update(choices=tracks, value=track),
        gr.update(choices=encoders, value=encoder),
        gr.update(choices=rates, value=rate),
        f"Loaded {len(keys)} tracks from manifests.",
    )


def update_tracks(root: str, dataset: str):
    if not dataset:
        return _empty_update(), _empty_update(), _empty_update()
    _, keys, _ = _choices(root)
    tracks = sorted(track_id for ds, track_id in keys if ds == dataset)
    track = tracks[0] if tracks else None
    encoders, rates, encoder, rate = selection_options(root, dataset, track)
    return gr.update(choices=tracks, value=track), gr.update(choices=encoders, value=encoder), gr.update(choices=rates, value=rate)


def selection_options(root: str, dataset: str | None, track_id: str | None, encoder: str | None = None):
    if not dataset or not track_id:
        return [], [], None, None
    refs = refs_for_track(root, dataset, track_id)
    encoders = sorted({ref.encoder for ref in refs}, key=_encoder_key)
    chosen_encoder = encoder if encoder in encoders else (encoders[0] if encoders else None)
    rates = sorted({ref.rate for ref in refs if ref.encoder == chosen_encoder}, key=_rate_key)
    chosen_rate = rates[0] if rates else None
    return encoders, rates, chosen_encoder, chosen_rate


def update_track_selection(root: str, dataset: str, track_id: str):
    encoders, rates, encoder, rate = selection_options(root, dataset, track_id)
    return gr.update(choices=encoders, value=encoder), gr.update(choices=rates, value=rate)


def update_rates(root: str, dataset: str, track_id: str, encoder: str):
    _, rates, _, rate = selection_options(root, dataset, track_id, encoder)
    return gr.update(choices=rates, value=rate)


def render_plot(root: str, clean_labels: str, dataset: str, track_id: str, encoder: str, rate: str, compare: str, window: int, power: float, dmspf_contrast: float, sigma: float, lam: float, points: int, max_frames: int, umap_frames: int, umap_components: int, trim_edges: bool):
    if not dataset or not track_id:
        raise gr.Error("Load the dataset index and choose a track first.")
    refs = refs_for_track(root, dataset, track_id)
    if compare == "encoders @ rate":
        selected_refs = [ref for ref in refs if ref.rate == rate]
    elif compare == "rates @ encoder":
        selected_refs = [ref for ref in refs if ref.encoder == encoder]
    else:
        selected_refs = [ref for ref in refs if ref.encoder == encoder and ref.rate == rate]
    if not selected_refs:
        raise gr.Error("No matching feature files for this selection.")

    track = load_track(root, dataset, track_id, clean_labels)
    plot_track, trim_start, trim_end = _interior_track(track) if trim_edges else (track, 0.0, 1.0)
    fig = build_figure(
        track=plot_track,
        selected_refs=selected_refs,
        window=int(window),
        power=float(power),
        dmspf_contrast=float(dmspf_contrast),
        sigma=float(sigma),
        lam=float(lam),
        points=int(points),
        max_frames=int(max_frames),
        umap_frames=int(umap_frames),
        umap_components=int(umap_components),
        trim_start=float(trim_start),
        trim_end=float(trim_end),
    )
    png_path = _save_figure(fig, "png")
    pdf_path = _save_figure(fig, "pdf")
    plt.close(fig)
    suffix = f" trimmed to {trim_start:.3f}-{trim_end:.3f}" if trim_edges else ""
    status = f"Rendered {dataset}/{track_id} with {len(selected_refs)} line(s){suffix}."
    return png_path, pdf_path, status


def build_app(default_root: str):
    with gr.Blocks(title="MSPF Structure Lab") as demo:
        gr.HTML(
            """
            <div class="app-title">
              <h1>MSPF Structure Lab</h1>
              <p>Interactive semantic progress inspection for structure embeddings.</p>
            </div>
            """
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=330, elem_classes=["panel"]):
                root = gr.Textbox(label="Structure root", value=default_root)
                clean_labels = gr.Dropdown(label="Labels", choices=["dataset", "none", "raw"], value="dataset")
                load = gr.Button("Load dataset index", variant="primary")
                status = gr.Markdown("<span class='status'>Ready.</span>")
                dataset = gr.Dropdown(label="Dataset", choices=[], value=None)
                track_id = gr.Dropdown(label="Track", choices=[], value=None)
                encoder = gr.Dropdown(label="Encoder", choices=[], value=None)
                rate = gr.Dropdown(label="Rate", choices=[], value=None)
                compare = gr.Radio(label="Compare", choices=["single", "encoders @ rate", "rates @ encoder"], value="single")
                render = gr.Button("Render plot", variant="primary")
                pdf = gr.File(label="PDF")
            with gr.Column(scale=3, elem_classes=["plot-panel"]):
                plot = gr.Image(label="MSPF figure", type="filepath", height=760)
                with gr.Accordion("Parameters", open=False):
                    with gr.Row():
                        window = gr.Slider(1, 64, value=DEFAULT_MSPF_WINDOW, step=1, label="SPF window")
                        power = gr.Slider(0.25, 8.0, value=float(DEFAULT_MSPF_POWER), step=0.25, label="Contrast p")
                    dmspf_contrast = gr.Slider(0.25, 8.0, value=1.0, step=0.25, label="dMSPF contrast")
                    with gr.Row():
                        sigma = gr.Number(value=float(DEFAULT_MSPF_SIGMA), minimum=0.1, maximum=100.0, label="Sigma")
                        lam = gr.Number(value=float(DEFAULT_MSPF_LAM), minimum=1e-6, maximum=1.0, label="Lambda")
                    with gr.Row():
                        points = gr.Slider(32, 1024, value=DEFAULT_MSPF_POINTS, step=32, label="MSPF points")
                        max_frames = gr.Slider(64, 4096, value=1024, step=64, label="Max feature frames")
                    with gr.Row():
                        umap_frames = gr.Slider(32, 2048, value=500, step=32, label="Max UMAP frames")
                        umap_components = gr.Slider(3, 16, value=DEFAULT_UMAP_COMPONENTS, step=1, label="UMAP components")
                    trim_edges = gr.Checkbox(label="Trim edge sections", value=False)

        load.click(load_index, inputs=root, outputs=[dataset, track_id, encoder, rate, status])
        dataset.change(update_tracks, inputs=[root, dataset], outputs=[track_id, encoder, rate])
        track_id.change(update_track_selection, inputs=[root, dataset, track_id], outputs=[encoder, rate])
        encoder.change(update_rates, inputs=[root, dataset, track_id, encoder], outputs=rate)
        render.click(
            render_plot,
            inputs=[root, clean_labels, dataset, track_id, encoder, rate, compare, window, power, dmspf_contrast, sigma, lam, points, max_frames, umap_frames, umap_components, trim_edges],
            outputs=[plot, pdf, status],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_app(args.root)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, css=CSS, theme=gr.themes.Base())


if __name__ == "__main__":
    main()
