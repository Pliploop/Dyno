#!/usr/bin/env python
"""Interactive Streamlit app for structure MSPF inspection."""

from __future__ import annotations

import os
import sys
import warnings
from io import BytesIO
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

import numpy as np
import streamlit as st
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandas as pd
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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          color-scheme: light;
          --ji-bg: #f5f7f4;
          --ji-panel: #ffffff;
          --ji-panel-soft: #f9fbf8;
          --ji-text: #171917;
          --ji-muted: #626b62;
          --ji-line: #dfe6df;
          --ji-green: #286145;
          --ji-green-bg: #e7f3ec;
          --ji-blue: #245982;
          --ji-blue-bg: #e7f0f7;
        }
        .stApp {
          background: var(--ji-bg);
          color: var(--ji-text);
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
          background: var(--ji-bg) !important;
          color: var(--ji-text) !important;
        }
        [data-testid="stSidebar"] {
          background: #eef3ee;
          border-right: 1px solid var(--ji-line);
        }
        .block-container {
          max-width: 1480px;
          padding-top: 1.4rem;
          padding-bottom: 3rem;
        }
        .ji-page-title {
          margin: 0 0 12px;
          padding-bottom: 8px;
          border-bottom: 1px solid var(--ji-line);
        }
        .ji-page-title h1 {
          margin: 0;
          color: var(--ji-text);
          font-size: 1.35rem;
          line-height: 1.2;
          letter-spacing: 0;
        }
        .ji-page-title p {
          margin: 4px 0 0;
          color: var(--ji-muted);
          font-size: 0.9rem;
        }
        .ji-card {
          background: var(--ji-panel);
          border: 1px solid var(--ji-line);
          border-radius: 8px;
          box-shadow: 0 10px 24px rgba(20, 28, 22, 0.05);
          padding: 14px;
          margin-bottom: 12px;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stMetric"] {
          background: var(--ji-panel-soft);
          border: 1px solid var(--ji-line);
          border-radius: 8px;
          padding: .65rem .75rem;
          box-shadow: none;
        }
        [data-testid="stMetricValue"] {
          font-size: 1.05rem;
          color: var(--ji-text) !important;
        }
        .stButton > button {
          background: var(--ji-panel) !important;
          color: var(--ji-text) !important;
          border: 1px solid #b9cabb !important;
          border-radius: 8px !important;
          box-shadow: none !important;
        }
        .stButton > button:hover {
          background: var(--ji-green-bg) !important;
          color: var(--ji-green) !important;
          border-color: #9fc9af !important;
        }
        .stSelectbox [data-baseweb="select"],
        .stNumberInput input,
        .stSlider,
        .stTextInput input {
          color: var(--ji-text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def discover_index(root: str):
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
    for refs in features.values():
        refs.sort(key=lambda ref: (_encoder_key(ref.encoder), _rate_key(ref.rate)))
    keys = sorted(features, key=lambda key: (key[0], key[1]))
    return features, keys


@st.cache_data(show_spinner=False)
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
        if not {"dataset", "track_id", "boundaries", "labels"} <= set(df.columns):
            continue
        hit = df[(df["dataset"].astype(str) == dataset) & (df["track_id"].astype(str) == track_id)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        raw_boundaries = parse_float_sequence_field(row.get("boundaries", ""))
        raw_labels = parse_sequence_field(row.get("labels", ""))
        boundaries, labels = clean_structure_segments(raw_boundaries, raw_labels, dataset=dataset, mode=clean_labels)
        return Track(dataset=dataset, track_id=track_id, boundaries=boundaries, labels=labels)
    raise ValueError(f"No annotations found for {dataset}/{track_id}")


@st.cache_data(show_spinner=False)
def load_feature(path: str, max_frames: int | None, trim_start: float = 0.0, trim_end: float = 1.0) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float32)
    arr = _crop_feature_frames(arr, trim_start, trim_end)
    if max_frames is not None and arr.shape[0] > max_frames:
        idx = np.linspace(0, arr.shape[0] - 1, max_frames).round().astype(np.int64)
        arr = arr[idx]
    return arr.copy()


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


def refs_for_track(features: dict, key: tuple[str, str]) -> list[FeatureRef]:
    return sorted(features.get(key, []), key=lambda ref: (_encoder_key(ref.encoder), _rate_key(ref.rate)))


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
        path = str(ref.path)
        t, mspf, dmspf = mspf_curve(path, max_frames, window, power, sigma, lam, points, dmspf_contrast, trim_start, trim_end)
        name = f"{ref.encoder} @ {ref.rate}"
        axes[0].plot(t, mspf, color=color, linestyle=linestyle, linewidth=1.35)
        axes[1].plot(t, dmspf, color=color, linestyle=linestyle, linewidth=1.2)
        legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=name))
        dmspf_series.append((t, dmspf))
        umap_rows.append(umap_strip(path, umap_frames, umap_components, seed=0, trim_start=trim_start, trim_end=trim_end)[0])
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


def figure_bytes(fig, fmt: str) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=160)
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title="MSPF Structure Lab", layout="wide")
    inject_css()
    st.markdown(
        '<div class="ji-page-title"><h1>MSPF Structure Lab</h1>'
        '<p>Interactive semantic progress inspection for structure embeddings.</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        root = st.text_input("Structure root", value=str(DEFAULT_ROOT))
        clean_labels = st.selectbox("Labels", ["dataset", "none", "raw"], index=0)
        index_signature = (root,)
        if st.session_state.get("index_signature") != index_signature:
            st.session_state.index_signature = index_signature
            st.session_state.index_loaded = False
        load_index = st.button(
            "Load dataset index" if not st.session_state.get("index_loaded") else "Reload dataset index",
            use_container_width=True,
        )
        if load_index:
            discover_index.clear()
            st.session_state.index_loaded = True
        max_frames = st.number_input("Max feature frames", min_value=64, max_value=4096, value=1024, step=64)
        umap_frames = st.number_input("Max UMAP frames", min_value=32, max_value=2048, value=500, step=32)
        umap_components = st.slider("UMAP components", min_value=3, max_value=16, value=DEFAULT_UMAP_COMPONENTS)
        st.divider()
        window = st.slider("SPF window", min_value=1, max_value=64, value=DEFAULT_MSPF_WINDOW)
        power = st.slider("Contrast p", min_value=0.25, max_value=8.0, value=float(DEFAULT_MSPF_POWER), step=0.25)
        dmspf_contrast = st.slider("dMSPF contrast", min_value=0.25, max_value=8.0, value=1.0, step=0.25)
        sigma = st.number_input("Sigma", min_value=0.1, max_value=100.0, value=float(DEFAULT_MSPF_SIGMA), step=0.5)
        lam = st.number_input("Lambda", min_value=1e-6, max_value=1.0, value=float(DEFAULT_MSPF_LAM), format="%.6f")
        points = st.number_input("MSPF points", min_value=32, max_value=1024, value=DEFAULT_MSPF_POINTS, step=32)
        trim_edges = st.checkbox("Trim edge sections", value=False)
        auto_render = st.checkbox("Auto render", value=False)

    if not st.session_state.get("index_loaded"):
        st.info("Click Load dataset index in the sidebar to scan manifests. The app shell is ready before any dataset crawl runs.")
        return

    try:
        with st.spinner("Loading manifest index..."):
            features, keys = discover_index(root)
    except Exception as exc:
        st.error(f"Could not discover structure dataset: {exc}")
        return
    if not keys:
        st.warning("No matching structure features/manifests found.")
        return

    datasets = sorted({dataset for dataset, _ in keys})
    c1, c2, c3, c4 = st.columns([1.0, 1.7, 1.0, 1.0])
    with c1:
        dataset = st.selectbox("Dataset", datasets)
    dataset_tracks = sorted([track_id for ds, track_id in keys if ds == dataset])
    with c2:
        track_id = st.selectbox("Track", dataset_tracks)
    key = (dataset, track_id)
    refs = refs_for_track(features, key)
    encoders = sorted({ref.encoder for ref in refs}, key=_encoder_key)
    with c3:
        encoder = st.selectbox("Encoder", encoders)
    rates = sorted({ref.rate for ref in refs if ref.encoder == encoder}, key=_rate_key)
    with c4:
        rate = st.selectbox("Rate", rates)

    compare = st.radio("Compare", ["single", "encoders @ rate", "rates @ encoder"], horizontal=True)
    if compare == "encoders @ rate":
        selected_refs = [ref for ref in refs if ref.rate == rate]
    elif compare == "rates @ encoder":
        selected_refs = [ref for ref in refs if ref.encoder == encoder]
    else:
        selected_refs = [ref for ref in refs if ref.encoder == encoder and ref.rate == rate]

    try:
        track = load_track(root, dataset, track_id, clean_labels)
    except Exception as exc:
        st.error(f"Could not load annotations for this track: {exc}")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Available encoders", len(encoders))
    m2.metric("Available rates", len(sorted({ref.rate for ref in refs}, key=_rate_key)))
    m3.metric("Sections", len(track.labels))
    plot_track, trim_start, trim_end = _interior_track(track) if trim_edges else (track, 0.0, 1.0)

    render = st.button("Render plot", type="primary", use_container_width=True)
    if not (render or auto_render):
        st.info("Choose a track/configuration, then click Render plot. Enable Auto render for live recompute.")
        return

    with st.spinner("Computing plot..."):
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

    st.markdown('<div class="ji-card">', unsafe_allow_html=True)
    st.pyplot(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    png = figure_bytes(fig, "png")
    pdf = figure_bytes(fig, "pdf")
    d1, d2, d3 = st.columns([1, 1, 4])
    d1.download_button("Download PNG", data=png, file_name=f"{dataset}_{track_id}_mspf.png", mime="image/png")
    d2.download_button("Download PDF", data=pdf, file_name=f"{dataset}_{track_id}_mspf.pdf", mime="application/pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
