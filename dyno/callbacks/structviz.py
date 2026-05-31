"""Structural visualization callback for SALAMI/Harmonix examples."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from dyno.callbacks.utils import BaseCallback
from dyno.evaluation.structure import StructureTargets, load_structure_targets
from dyno.evaluation.temporal import compute_mspf

log = logging.getLogger(__name__)


def _get_wandb_logger(trainer):
    try:
        from lightning.pytorch.loggers import WandbLogger

        for lg in (trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]):
            if isinstance(lg, WandbLogger):
                return lg
    except Exception:
        pass
    return None


def _fig_to_wandb_image(fig):
    import io

    import wandb
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    image = Image.open(buf).copy()
    return wandb.Image(image)


def _resolve_feature_path(feature_path: str, feature_root: str | None) -> Path:
    path = Path(feature_path)
    if path.is_absolute() or feature_root is None:
        return path
    return Path(feature_root) / path


def _load_feature_sequence(path: Path) -> torch.Tensor:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected feature array with shape (T, D), got {arr.shape} at {path}")
    return torch.from_numpy(arr.copy())


def _uniform_limit(x: torch.Tensor, max_frames: int | None) -> torch.Tensor:
    if max_frames is None or x.shape[0] <= max_frames:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, max_frames).round().long()
    return x[idx]


def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


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


def _embedding_rgb_timeline(x: torch.Tensor, seed: int) -> tuple[np.ndarray, str]:
    x_np = x.detach().cpu().float().numpy()
    from umap import UMAP

    reducer = UMAP(
        n_components=3,
        n_neighbors=min(15, x_np.shape[0] - 1),
        min_dist=0.05,
        metric="cosine",
        random_state=seed,
    )
    return _rgb01(reducer.fit_transform(x_np)), "UMAP"


def _label_colors(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    palette = list(plt.get_cmap("tab20").colors)
    unique = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return {
        label: (*palette[i % len(palette)], 1.0)
        for i, label in enumerate(unique)
    }


class StructVizCallback(BaseCallback):
    """Log MSPF, embedding UMAP timeline, and structure labels for examples."""

    def __init__(
        self,
        manifest_csv: str,
        feature_root: str | None = None,
        every_n_epochs: int = 20,
        test_only: bool = False,
        n_samples_per_dataset: int = 4,
        datasets: tuple[str, ...] = ("salami", "harmonix"),
        max_frames: int | None = 1024,
        evaluation_suite: str = "Structure visualization",
        clean_labels: str = "dataset",
        seed: int = 0,
        mspf_window: int = 30,
        mspf_sigma: float = 10.0,
        mspf_lam: float = 1e-3,
        mspf_power: float = 1.4,
        mspf_points: int = 256,
    ):
        super().__init__(every_n_epochs=every_n_epochs)
        self.manifest_csv = manifest_csv
        self.feature_root = feature_root
        self.test_only = test_only
        self.n_samples_per_dataset = n_samples_per_dataset
        self.datasets = tuple(d.lower() for d in datasets)
        self.max_frames = max_frames
        self.evaluation_suite = evaluation_suite
        self.clean_labels = clean_labels
        self.seed = seed
        self.mspf_kw = dict(
            window=mspf_window,
            sigma=mspf_sigma,
            lam=mspf_lam,
            power=mspf_power,
            absolute=False,
            n_points=mspf_points,
        )
        self._targets: StructureTargets | None = None

    def _load_targets(self) -> StructureTargets:
        if self._targets is None:
            self._targets = load_structure_targets(self.manifest_csv, clean=self.clean_labels)
        return self._targets

    def _sample_indices(self, targets: StructureTargets, stage: str, epoch: int) -> list[int]:
        rng = np.random.default_rng(self.seed + epoch + (100_000 if stage == "test" else 0))
        datasets = np.asarray([d.lower() for d in targets.datasets])
        indices: list[int] = []
        for dataset in self.datasets:
            candidates = np.flatnonzero(datasets == dataset)
            if candidates.size == 0:
                log.warning("StructViz found no %s rows in %s", dataset, self.manifest_csv)
                continue
            n = min(self.n_samples_per_dataset, int(candidates.size))
            indices.extend(rng.choice(candidates, size=n, replace=False).tolist())
        return indices

    def _feature_for_index(self, targets: StructureTargets, idx: int, pl_module) -> torch.Tensor:
        path = _resolve_feature_path(targets.feature_paths[idx], self.feature_root)
        x = _uniform_limit(_load_feature_sequence(path), self.max_frames)
        x = x.to(device=pl_module.device)
        with torch.inference_mode():
            if hasattr(pl_module, "normalize_input"):
                x = pl_module.normalize_input(x.unsqueeze(0)).squeeze(0)
        return x.detach().cpu()

    def _make_figure(
        self,
        x: torch.Tensor,
        track_id: str,
        dataset: str,
        boundaries: np.ndarray,
        labels: list[str],
        seed: int,
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        mspf = _minmax01(compute_mspf(x, **self.mspf_kw))
        t_mspf = np.linspace(0.0, 1.0, len(mspf), dtype=np.float32)
        dmspf = np.gradient(mspf, t_mspf).astype(np.float32)
        dmspf = _minmax01(dmspf)
        rgb, method = _embedding_rgb_timeline(x, seed=seed)
        strip = rgb[None, :, :]

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(9, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 0.75, 0.55, 0.55], "hspace": 0.18},
        )

        axes[0].plot(t_mspf, mspf, color="#111111", linewidth=1.8)
        axes[0].fill_between(t_mspf, 0.0, mspf, color="#111111", alpha=0.08)
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_ylabel("MSPF")
        axes[0].grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)

        axes[1].plot(t_mspf, dmspf, color="#C2410C", linewidth=1.4)
        axes[1].fill_between(t_mspf, 0.0, dmspf, color="#C2410C", alpha=0.08)
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_ylabel("dMSPF")
        axes[1].grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)

        axes[2].imshow(strip, aspect="auto", extent=[0.0, 1.0, 0.0, 1.0], interpolation="nearest")
        axes[2].set_yticks([])
        axes[2].set_ylabel(method)

        axes[3].set_ylim(0.0, 1.0)
        axes[3].set_yticks([])
        axes[3].set_ylabel("Labels")
        colors = _label_colors(labels)
        boundaries = np.asarray(boundaries, dtype=np.float32)
        if labels and boundaries.size > 0:
            duration = float(max(boundaries[-1], 1e-6))
            starts = boundaries[:-1] if boundaries.size == len(labels) + 1 else boundaries[: len(labels)]
            starts = np.clip(starts / duration, 0.0, 1.0)
            ends = np.concatenate([starts[1:], [1.0]]).astype(np.float32)
            legend_labels: list[str] = []
            for start, end, label in zip(starts, ends, labels):
                width = max(float(end - start), 1e-6)
                axes[3].broken_barh([(float(start), width)], (0.0, 1.0), facecolors=colors[label])
                if width >= 0.055:
                    axes[3].text(
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
            if legend_labels:
                handles = [Patch(facecolor=colors[label], label=label) for label in legend_labels[:12]]
                axes[3].legend(handles=handles, ncol=min(6, len(handles)), loc="upper center", bbox_to_anchor=(0.5, -0.45), fontsize=7)

        for ax in axes:
            ax.set_xlim(0.0, 1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[3].set_xlabel("normalised time")
        fig.suptitle(f"{dataset} / {track_id}", fontsize=11)
        fig.tight_layout()
        return fig

    def _run(self, stage: str, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        if not Path(self.manifest_csv).exists():
            log.warning("Skipping StructViz: manifest does not exist: %s", self.manifest_csv)
            return

        wandb_logger = _get_wandb_logger(trainer)
        if wandb_logger is None:
            log.warning("Skipping StructViz image logging because no WandbLogger is attached.")
            return

        targets = self._load_targets()
        figures = []
        epoch = int(getattr(trainer, "current_epoch", 0))
        was_training = pl_module.training
        pl_module.eval()
        try:
            for sample_i, idx in enumerate(self._sample_indices(targets, stage, epoch)):
                x = self._feature_for_index(targets, idx, pl_module)
                fig = self._make_figure(
                    x=x,
                    track_id=targets.track_ids[idx],
                    dataset=targets.datasets[idx],
                    boundaries=targets.boundaries[idx],
                    labels=targets.labels[idx],
                    seed=self.seed + epoch + sample_i,
                )
                figures.append(fig)
        finally:
            if was_training:
                pl_module.train()

        if figures:
            key = f"{self.evaluation_suite}/{stage}/MSPF-UMAP-structure timelines"
            wandb_logger.experiment.log(
                {key: [_fig_to_wandb_image(fig) for fig in figures]},
                step=trainer.global_step,
            )

        import matplotlib.pyplot as plt

        for fig in figures:
            plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.test_only:
            return
        if self._check_epoch(trainer, pl_module):
            self._run("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_only:
            self._run("test", trainer, pl_module)
