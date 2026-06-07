#!/usr/bin/env python
"""Build MSPF structure visualizations for extracted structure embeddings.

The script discovers structure feature arrays laid out as:

    features/<dataset>/<encoder>/<rate>/<track_id>.npy

and writes PDF figures under:

    viz/<dataset>/<track_id>/

It is CPU-only and does not instantiate any model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dyno-mspf-matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/dyno-mspf-numba-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from dyno.evaluation.structure import clean_structure_segments, parse_float_sequence_field, parse_sequence_field
from dyno.evaluation.temporal import compute_mspf


DEFAULT_ROOT = Path("/gpfs/scratch/acw749/datasets/structure")
DEFAULT_WINDOWS = (2, 3, 5, 8, 13)
DEFAULT_POWERS = (0.75, 1.0, 1.4, 2.0, 5.0)
DEFAULT_MSPF_POINTS = 256
DEFAULT_MSPF_SIGMA = 10.0
DEFAULT_MSPF_LAM = 1e-3
DEFAULT_MSPF_WINDOW = 3
DEFAULT_MSPF_POWER = 5.0
DEFAULT_UMAP_COMPONENTS = 8
DEFAULT_WORKERS = 4


@dataclass(frozen=True)
class Track:
    dataset: str
    track_id: str
    boundaries: np.ndarray
    labels: list[str]


@dataclass(frozen=True)
class FeatureRef:
    dataset: str
    encoder: str
    rate: str
    track_id: str
    path: Path


@dataclass(frozen=True)
class Series:
    label: str
    feature: FeatureRef
    window: int
    power: float
    color_key: str
    style_key: str


def _slug(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown"


def _rate_key(rate: str) -> tuple[float, str]:
    match = re.search(r"[-+]?\d*\.?\d+", rate)
    return (float(match.group(0)) if match else float("inf"), rate)


def _encoder_key(encoder: str) -> str:
    return encoder.lower()


def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _semantic_percent(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize an SPF to percentage of total semantic progress."""
    return np.clip(_minmax01(x, eps=eps), 0.0, 1.0).astype(np.float32)


def _load_feature(path: Path, max_frames: int | None) -> torch.Tensor:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected feature array with shape (T, D), got {arr.shape} at {path}")
    if max_frames is not None and arr.shape[0] > max_frames:
        idx = np.linspace(0, arr.shape[0] - 1, max_frames).round().astype(np.int64)
        arr = arr[idx]
    return torch.from_numpy(arr.copy())


def _discover_features(feature_root: Path) -> dict[tuple[str, str], list[FeatureRef]]:
    features: dict[tuple[str, str], list[FeatureRef]] = {}
    if not feature_root.exists():
        return features
    for path in sorted(feature_root.glob("*/*/*/*.npy")):
        rel = path.relative_to(feature_root)
        dataset, encoder, rate = rel.parts[:3]
        track_id = path.stem
        ref = FeatureRef(dataset=dataset, encoder=encoder, rate=rate, track_id=track_id, path=path)
        features.setdefault((dataset, track_id), []).append(ref)
    for refs in features.values():
        refs.sort(key=lambda r: (_encoder_key(r.encoder), _rate_key(r.rate)))
    return features


def _load_tracks(manifest_root: Path, clean_labels: str) -> dict[tuple[str, str], Track]:
    tracks: dict[tuple[str, str], Track] = {}
    for manifest in sorted(manifest_root.glob("*/*/all.csv")):
        try:
            df = pd.read_csv(manifest)
        except Exception:
            continue
        required = {"dataset", "track_id", "boundaries", "labels"}
        if not required <= set(df.columns):
            continue
        for _, row in df.iterrows():
            dataset = str(row["dataset"])
            track_id = str(row["track_id"])
            key = (dataset, track_id)
            if key in tracks:
                continue
            boundaries = parse_float_sequence_field(row.get("boundaries", ""))
            labels = parse_sequence_field(row.get("labels", ""))
            boundaries, labels = clean_structure_segments(boundaries, labels, dataset=dataset, mode=clean_labels)
            tracks[key] = Track(dataset=dataset, track_id=track_id, boundaries=boundaries, labels=labels)
    return tracks


def _label_colors(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    palette = list(plt.get_cmap("tab20").colors)
    unique: list[str] = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return {label: (*palette[i % len(palette)], 1.0) for i, label in enumerate(unique)}


def _style_maps(series: list[Series]) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, str]]:
    palette = list(plt.get_cmap("Set1").colors) + list(plt.get_cmap("tab10").colors)
    color_keys = []
    style_keys = []
    for item in series:
        if item.color_key not in color_keys:
            color_keys.append(item.color_key)
        if item.style_key not in style_keys:
            style_keys.append(item.style_key)
    colors = {key: (*palette[i % len(palette)], 1.0) for i, key in enumerate(color_keys)}
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    styles = {key: linestyles[i % len(linestyles)] for i, key in enumerate(style_keys)}
    return colors, styles


def _rgb01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    for dim in range(x.shape[1]):
        col = x[:, dim]
        lo, hi = np.nanpercentile(col, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
            out[:, dim] = 0.5
        else:
            out[:, dim] = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    return out


def _umap_rgb_timeline(x: torch.Tensor, max_frames: int, seed: int, n_components: int) -> np.ndarray:
    x_np = x.detach().cpu().float().numpy()
    if max_frames > 0 and x_np.shape[0] > max_frames:
        idx = np.linspace(0, x_np.shape[0] - 1, max_frames).round().astype(np.int64)
        x_np = x_np[idx]
    if x_np.shape[0] < 3:
        return np.full((max(1, x_np.shape[0]), 3), 0.5, dtype=np.float32)

    from umap import UMAP

    n_components = max(3, min(int(n_components), x_np.shape[0] - 1))
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=min(15, x_np.shape[0] - 1),
        min_dist=0.05,
        metric="cosine",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")
        return _rgb01(reducer.fit_transform(x_np)[:, :3])


def _plot_annotations(ax, boundaries: np.ndarray, labels: list[str]) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("Labels")
    if not labels:
        return
    colors = _label_colors(labels)
    boundaries = np.asarray(boundaries, dtype=np.float32)
    if boundaries.size == 0:
        starts = np.linspace(0.0, 1.0, len(labels), endpoint=False, dtype=np.float32)
    else:
        duration = float(max(boundaries[-1], 1e-6))
        starts = boundaries[:-1] if boundaries.size == len(labels) + 1 else boundaries[: len(labels)]
        starts = np.clip(starts / duration, 0.0, 1.0)
    ends = np.concatenate([starts[1:], [1.0]]).astype(np.float32)
    legend_labels: list[str] = []
    for start, end, label in zip(starts, ends, labels):
        width = max(float(end - start), 1e-6)
        ax.broken_barh([(float(start), width)], (0.0, 1.0), facecolors=colors[label])
        if width >= 0.055:
            ax.text(
                float(start + width / 2),
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                clip_on=True,
            )
        if label not in legend_labels:
            legend_labels.append(label)
    handles = [Patch(facecolor=colors[label], label=label) for label in legend_labels[:12]]
    if handles:
        ax.legend(handles=handles, ncol=min(6, len(handles)), loc="upper center", bbox_to_anchor=(0.5, -0.58), fontsize=7)


def _make_figure(
    *,
    track: Track,
    series: list[Series],
    feature_cache: dict[Path, torch.Tensor],
    umap_cache: dict[Path, np.ndarray],
    max_frames: int | None,
    max_umap_frames: int,
    umap_components: int,
    mspf_points: int,
    sigma: float,
    lam: float,
    title: str,
) -> plt.Figure | None:
    if not series:
        return None
    colors, styles = _style_maps(series)
    fig = plt.figure(figsize=(10.5, 7.1))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 0.75, 0.75, 0.55], hspace=0.22)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    legend_handles: list[Line2D] = []
    umap_strips: list[np.ndarray] = []
    umap_labels: list[str] = []
    umap_seen: set[Path] = set()
    for item in series:
        if item.feature.path not in feature_cache:
            feature_cache[item.feature.path] = _load_feature(item.feature.path, max_frames=max_frames)
        x = feature_cache[item.feature.path]
        if x.shape[0] < 2:
            continue
        mspf = _semantic_percent(
            compute_mspf(
                x,
                window=min(item.window, max(1, x.shape[0] - 1)),
                sigma=sigma,
                lam=lam,
                power=item.power,
                absolute=False,
                n_points=mspf_points,
            )
        )
        t_mspf = np.linspace(0.0, 1.0, len(mspf), dtype=np.float32)
        dmspf = _minmax01(np.gradient(mspf, t_mspf).astype(np.float32))
        color = colors[item.color_key]
        style = styles[item.style_key]
        axes[0].plot(t_mspf, mspf, color=color, linestyle=style, linewidth=1.5, alpha=0.95)
        axes[1].plot(t_mspf, dmspf, color=color, linestyle=style, linewidth=1.2, alpha=0.95)
        legend_handles.append(Line2D([0], [0], color=color, linestyle=style, linewidth=1.7, label=item.label))

        if item.feature.path not in umap_seen:
            umap_seen.add(item.feature.path)
            if item.feature.path not in umap_cache:
                umap_cache[item.feature.path] = _umap_rgb_timeline(
                    x,
                    max_frames=max_umap_frames,
                    seed=0,
                    n_components=umap_components,
                )
            umap_strips.append(umap_cache[item.feature.path])
            umap_labels.append(f"{item.feature.encoder} @ {item.feature.rate}")

    for ax, ylabel in zip(axes[:2], ["MSPF (%)", "dMSPF"]):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if umap_strips:
        width = max(strip.shape[0] for strip in umap_strips)
        rows = []
        for strip in umap_strips:
            xp = np.linspace(0.0, 1.0, strip.shape[0], dtype=np.float32)
            xq = np.linspace(0.0, 1.0, width, dtype=np.float32)
            row = np.stack([np.interp(xq, xp, strip[:, dim]) for dim in range(3)], axis=-1)
            rows.append(row.astype(np.float32))
        image = np.stack(rows, axis=0)
        axes[2].imshow(image, aspect="auto", extent=[0.0, 1.0, len(rows), 0.0], interpolation="nearest")
        axes[2].set_yticks(np.arange(len(rows)) + 0.5)
        axes[2].set_yticklabels(umap_labels, fontsize=7)
    else:
        axes[2].set_yticks([])
        axes[2].text(0.5, 0.5, "UMAP unavailable", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylabel("UMAP")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    _plot_annotations(axes[3], track.boundaries, track.labels)
    axes[3].set_xlim(0.0, 1.0)
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)

    axes[3].set_xlabel("normalised time")
    if legend_handles:
        axes[0].legend(
            handles=legend_handles,
            loc="upper left",
            ncol=min(4, len(legend_handles)),
            fontsize=7,
            frameon=False,
        )
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.2, hspace=0.26)
    return fig


def _save_figure(fig: plt.Figure, path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        plt.close(fig)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _refs_by_encoder_rate(refs: Iterable[FeatureRef]) -> dict[tuple[str, str], FeatureRef]:
    out = {}
    for ref in refs:
        out[(ref.encoder, ref.rate)] = ref
    return out


def _build_jobs(
    track: Track,
    refs: list[FeatureRef],
    *,
    figure_ext: str,
    base_window: int,
    base_power: float,
    windows: tuple[int, ...],
    powers: tuple[float, ...],
) -> list[tuple[str, str, list[Series]]]:
    jobs: list[tuple[str, str, list[Series]]] = []
    by_pair = _refs_by_encoder_rate(refs)
    encoders = sorted({ref.encoder for ref in refs}, key=_encoder_key)
    rates = sorted({ref.rate for ref in refs}, key=_rate_key)
    track_slug = _slug(track.track_id)
    suffix = f".{figure_ext.lstrip('.')}"

    for rate in rates:
        items = [
            Series(
                label=f"{encoder} @ {rate}",
                feature=by_pair[(encoder, rate)],
                window=base_window,
                power=base_power,
                color_key=encoder,
                style_key=encoder,
            )
            for encoder in encoders
            if (encoder, rate) in by_pair
        ]
        jobs.append(("encoders", f"track_{track_slug}_encoders_rate_{_slug(rate)}{suffix}", items))

    for encoder in encoders:
        items = [
            Series(
                label=f"{encoder} @ {rate}",
                feature=by_pair[(encoder, rate)],
                window=base_window,
                power=base_power,
                color_key=rate,
                style_key=rate,
            )
            for rate in rates
            if (encoder, rate) in by_pair
        ]
        jobs.append(("rates", f"track_{track_slug}_rates_encoder_{_slug(encoder)}{suffix}", items))

    for rate in rates:
        items = []
        for encoder in encoders:
            ref = by_pair.get((encoder, rate))
            if ref is None:
                continue
            for window in windows:
                items.append(
                    Series(
                        label=f"{encoder}, w={window}",
                        feature=ref,
                        window=window,
                        power=base_power,
                        color_key=encoder,
                        style_key=f"w={window}",
                    )
                )
        jobs.append(("window", f"track_{track_slug}_window_rate_{_slug(rate)}{suffix}", items))

    for rate in rates:
        items = []
        for encoder in encoders:
            ref = by_pair.get((encoder, rate))
            if ref is None:
                continue
            for power in powers:
                items.append(
                    Series(
                        label=f"{encoder}, p={power:g}",
                        feature=ref,
                        window=base_window,
                        power=power,
                        color_key=encoder,
                        style_key=f"p={power:g}",
                    )
                )
        jobs.append(("contrast", f"track_{track_slug}_contrast_rate_{_slug(rate)}{suffix}", items))

    return [(xp, name, items) for xp, name, items in jobs if items]


def _parse_csv_values(values: str, cast):
    out = []
    for value in values.split(","):
        value = value.strip()
        if value:
            out.append(cast(value))
    return tuple(out)


def _output_path(output_dir: Path, track: Track, xp: str, filename: str) -> Path:
    return output_dir / _slug(track.dataset) / _slug(track.track_id) / _slug(xp) / filename


def _render_track_jobs(payload: tuple[Track, list[tuple[str, str, list[Series]]], dict]) -> dict:
    track, jobs, cfg = payload
    output_dir = Path(cfg["output_dir"])
    feature_cache: dict[Path, torch.Tensor] = {}
    umap_cache: dict[Path, np.ndarray] = {}
    messages = []
    saved = 0
    skipped = 0
    failed = 0

    for xp, filename, series in jobs:
        output_path = _output_path(output_dir, track, xp, filename)
        if output_path.exists() and not cfg["overwrite"]:
            skipped += 1
            continue
        try:
            fig = _make_figure(
                track=track,
                series=series,
                feature_cache=feature_cache,
                umap_cache=umap_cache,
                max_frames=cfg["max_frames"],
                max_umap_frames=cfg["max_umap_frames"],
                umap_components=cfg["umap_components"],
                mspf_points=cfg["mspf_points"],
                sigma=cfg["mspf_sigma"],
                lam=cfg["mspf_lam"],
                title=f"{track.dataset} / {track.track_id} / {Path(filename).stem}",
            )
            if fig is None:
                skipped += 1
            elif _save_figure(fig, output_path, overwrite=cfg["overwrite"]):
                saved += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            messages.append(f"Skipping {output_path}: {exc}")

    return {
        "dataset": track.dataset,
        "track_id": track.track_id,
        "total": len(jobs),
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Structure dataset root.")
    parser.add_argument("--feature-root", type=Path, default=None, help="Feature root. Defaults to <root>/features.")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Manifest root. Defaults to <root>/manifests.")
    parser.add_argument("--output-dir", type=Path, default=Path("viz"), help="Output directory.")
    parser.add_argument("--datasets", default="salami,harmonix", help="Comma-separated dataset names.")
    parser.add_argument("--encoders", default=None, help="Optional comma-separated encoder filter.")
    parser.add_argument("--rates", default=None, help="Optional comma-separated rate filter.")
    parser.add_argument("--tracks", default=None, help="Optional comma-separated track_id filter.")
    parser.add_argument("--limit-tracks", type=int, default=None, help="Limit tracks for a smoke test.")
    parser.add_argument("--clean-labels", default="dataset", choices=["dataset", "none", "raw"], help="Structure label cleanup mode.")
    parser.add_argument("--max-frames", type=int, default=1024, help="Uniformly downsample long tracks before plotting.")
    parser.add_argument("--max-umap-frames", type=int, default=500, help="Max frames used in each UMAP strip.")
    parser.add_argument("--umap-components", type=int, default=DEFAULT_UMAP_COMPONENTS, help="UMAP dimensions to compute before RGB mapping.")
    parser.add_argument("--max-pca-points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mspf-window", type=int, default=DEFAULT_MSPF_WINDOW)
    parser.add_argument("--mspf-power", type=float, default=DEFAULT_MSPF_POWER)
    parser.add_argument("--mspf-sigma", type=float, default=DEFAULT_MSPF_SIGMA)
    parser.add_argument("--mspf-lam", type=float, default=DEFAULT_MSPF_LAM)
    parser.add_argument("--mspf-points", type=int, default=DEFAULT_MSPF_POINTS)
    parser.add_argument("--windows", default=",".join(str(v) for v in DEFAULT_WINDOWS), help="Comma-separated SPF windows to sweep.")
    parser.add_argument("--powers", default=",".join(str(v) for v in DEFAULT_POWERS), help="Comma-separated contrast powers to sweep.")
    parser.add_argument("--figure-format", default="pdf", choices=["pdf", "png"], help="Figure file format.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of worker processes for track-level rendering.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_pca_points is not None:
        args.max_umap_frames = args.max_pca_points
    console = Console()
    feature_root = args.feature_root or args.root / "features"
    manifest_root = args.manifest_root or args.root / "manifests"
    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    encoder_filter = {value.strip() for value in args.encoders.split(",")} if args.encoders else None
    rate_filter = {value.strip() for value in args.rates.split(",")} if args.rates else None
    track_filter = {value.strip() for value in args.tracks.split(",")} if args.tracks else None
    windows = _parse_csv_values(args.windows, int)
    powers = _parse_csv_values(args.powers, float)

    console.print(f"[bold]Discovering structure features[/bold] in {feature_root}")
    features = _discover_features(feature_root)
    tracks = _load_tracks(manifest_root, clean_labels=args.clean_labels)

    track_keys = sorted(set(features) & set(tracks), key=lambda key: (key[0], key[1]))
    if datasets:
        track_keys = [key for key in track_keys if key[0] in datasets]
    if track_filter:
        track_keys = [key for key in track_keys if key[1] in track_filter]
    if args.limit_tracks is not None:
        track_keys = track_keys[: args.limit_tracks]

    def filtered_refs(key: tuple[str, str]) -> list[FeatureRef]:
        refs = features[key]
        if encoder_filter:
            refs = [ref for ref in refs if ref.encoder in encoder_filter]
        if rate_filter:
            refs = [ref for ref in refs if ref.rate in rate_filter]
        return refs

    track_jobs: list[tuple[Track, list[tuple[str, str, list[Series]]]]] = []
    for key in track_keys:
        refs = filtered_refs(key)
        if not refs:
            continue
        track = tracks[key]
        jobs = _build_jobs(
            track,
            refs,
            figure_ext=args.figure_format,
            base_window=args.mspf_window,
            base_power=args.mspf_power,
            windows=windows,
            powers=powers,
        )
        if jobs:
            track_jobs.append((track, jobs))

    total_figures = sum(len(jobs) for _, jobs in track_jobs)
    console.print(f"Found {len(track_jobs)} tracks and {total_figures} figures to build.")
    if not track_jobs:
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    saved = 0
    skipped = 0
    failed = 0
    cfg = {
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
        "max_frames": args.max_frames,
        "max_umap_frames": args.max_umap_frames,
        "umap_components": args.umap_components,
        "mspf_points": args.mspf_points,
        "mspf_sigma": args.mspf_sigma,
        "mspf_lam": args.mspf_lam,
    }
    workers = max(1, int(args.workers))
    with progress:
        task = progress.add_task("Building MSPF visualizations", total=total_figures)
        if workers == 1:
            for payload in ((track, jobs, cfg) for track, jobs in track_jobs):
                result = _render_track_jobs(payload)
                saved += result["saved"]
                skipped += result["skipped"]
                failed += result["failed"]
                for message in result["messages"]:
                    console.print(f"[yellow]{message}[/yellow]")
                progress.update(task, description=f"{result['dataset']}/{result['track_id']}")
                progress.advance(task, result["total"])
        else:
            max_workers = min(workers, len(track_jobs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_render_track_jobs, (track, jobs, cfg))
                    for track, jobs in track_jobs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    saved += result["saved"]
                    skipped += result["skipped"]
                    failed += result["failed"]
                    for message in result["messages"]:
                        console.print(f"[yellow]{message}[/yellow]")
                    progress.update(task, description=f"{result['dataset']}/{result['track_id']}")
                    progress.advance(task, result["total"])

    console.print(f"[bold]Done.[/bold] saved={saved} skipped={skipped} failed={failed} output={args.output_dir}")


if __name__ == "__main__":
    main()
