"""CoverX structural agreement callback."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dyno.callbacks.utils import BaseCallback
from dyno.evaluation.structure import (
    StructureTargets,
    content_control_mask,
    cosine_distance_matrix,
    load_structure_targets,
    pairwise_form_distance,
    pairwise_ssm_distance,
    probe_scores,
    retrieval_metrics,
    spearman_from_distance_matrices,
)
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


def _resolve_feature_path(feature_path: str, feature_root: str | None) -> Path:
    path = Path(feature_path)
    if path.is_absolute() or feature_root is None:
        return path
    return Path(feature_root) / path


def _load_feature_sequence(path: Path) -> torch.Tensor:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected feature array with shape (T, D), got {arr.shape} at {path}")
    return torch.from_numpy(arr.copy())


def _uniform_limit(x: torch.Tensor, max_frames: int | None) -> torch.Tensor:
    if max_frames is None or x.shape[0] <= max_frames:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, max_frames).round().long()
    return x[idx]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _subset_targets(targets: StructureTargets, idx: np.ndarray) -> StructureTargets:
    return StructureTargets(
        track_ids=[targets.track_ids[i] for i in idx],
        datasets=[targets.datasets[i] for i in idx],
        form_strings=[targets.form_strings[i] for i in idx],
        section_count=targets.section_count[idx],
        unique_label_count=targets.unique_label_count[idx],
        repetition_rate=targets.repetition_rate[idx],
        return_a=targets.return_a[idx],
        bridge=targets.bridge[idx],
        through=targets.through[idx],
        boundaries=[targets.boundaries[i] for i in idx],
        labels=[targets.labels[i] for i in idx],
        feature_paths=[targets.feature_paths[i] for i in idx],
    )


def _retrieval_metric_name(key: str, content_controlled: bool = False) -> str:
    if key.startswith("D@"):
        k = key.split("@", 1)[1]
        label = f"Mean annotation form distance at {k} nearest embedding neighbors"
    elif key.startswith("FR@"):
        k = key.split("@", 1)[1]
        label = f"Form recall at {k}: embedding neighbors match annotation form"
    else:
        label = key
    if content_controlled:
        return f"Content-controlled {label[0].lower()}{label[1:]}"
    return label


def _probe_metric_name(key: str) -> str:
    names = {
        "section_count_R2": "Probe R2: section count from representation",
        "unique_label_count_R2": "Probe R2: unique section labels from representation",
        "repetition_rate_R2": "Probe R2: repetition rate from representation",
        "return_a_F1": "Probe F1: return-to-A form from representation",
        "bridge_F1": "Probe F1: bridge section from representation",
        "through_F1": "Probe F1: through-composed form from representation",
    }
    return names.get(key, key)


class CoverXCallback(BaseCallback):
    """Evaluate agreement between Dyno representations and structural annotations.

    The callback expects a normalized manifest with SALAMI and Harmonix rows
    already aligned to the same schema. It computes the same metrics either at
    validation time during training or at test time when ``test_only=True``.
    """

    def __init__(
        self,
        manifest_csv: str,
        feature_root: str | None = None,
        encoder: str = "unknown",
        rate: str = "unknown",
        every_n_epochs: int = 5,
        test_only: bool = False,
        max_tracks: int | None = 512,
        max_frames: int | None = 1024,
        batch_size: int = 64,
        form_epsilon: float = 0.25,
        content_control_pool: int = 50,
        ssm_frames: int = 64,
        mspf_points: int = 64,
        random_dim: int = 32,
        seed: int = 0,
        log_tables: bool = False,
        evaluation_suite: str = "Structure",
        clean_labels: str = "dataset",
        per_dataset: bool = True,
    ):
        super().__init__(every_n_epochs=every_n_epochs)
        self.manifest_csv = manifest_csv
        self.feature_root = feature_root
        self.encoder = encoder
        self.rate = rate
        self.test_only = test_only
        self.max_tracks = max_tracks
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.form_epsilon = form_epsilon
        self.content_control_pool = content_control_pool
        self.ssm_frames = ssm_frames
        self.mspf_points = mspf_points
        self.random_dim = random_dim
        self.seed = seed
        self.log_tables = log_tables
        self.evaluation_suite = evaluation_suite
        self.clean_labels = clean_labels
        self.per_dataset = per_dataset

        self._targets: StructureTargets | None = None
        self._features: list[torch.Tensor] | None = None
        self._form_distance: np.ndarray | None = None
        self._ssm_distance: np.ndarray | None = None

    def _load_eval_data(self) -> tuple[StructureTargets, list[torch.Tensor]]:
        if self._targets is not None and self._features is not None:
            return self._targets, self._features

        manifest = Path(self.manifest_csv)
        targets = load_structure_targets(manifest, clean=self.clean_labels)
        n = len(targets.track_ids)
        if self.max_tracks is not None and n > self.max_tracks:
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(n, size=self.max_tracks, replace=False))
            targets = StructureTargets(
                track_ids=[targets.track_ids[i] for i in keep],
                datasets=[targets.datasets[i] for i in keep],
                form_strings=[targets.form_strings[i] for i in keep],
                section_count=targets.section_count[keep],
                unique_label_count=targets.unique_label_count[keep],
                repetition_rate=targets.repetition_rate[keep],
                return_a=targets.return_a[keep],
                bridge=targets.bridge[keep],
                through=targets.through[keep],
                boundaries=[targets.boundaries[i] for i in keep],
                labels=[targets.labels[i] for i in keep],
                feature_paths=[targets.feature_paths[i] for i in keep],
            )

        features = []
        kept_rows = []
        for i, feature_path in enumerate(targets.feature_paths):
            path = _resolve_feature_path(feature_path, self.feature_root)
            try:
                features.append(_uniform_limit(_load_feature_sequence(path), self.max_frames))
                kept_rows.append(i)
            except Exception as exc:
                log.warning("Skipping CoverX row %s because feature loading failed: %s", feature_path, exc)

        if not features:
            raise RuntimeError(f"CoverX found no loadable features in {self.manifest_csv}")

        if len(kept_rows) != len(targets.track_ids):
            keep = np.asarray(kept_rows, dtype=np.int64)
            targets = StructureTargets(
                track_ids=[targets.track_ids[i] for i in keep],
                datasets=[targets.datasets[i] for i in keep],
                form_strings=[targets.form_strings[i] for i in keep],
                section_count=targets.section_count[keep],
                unique_label_count=targets.unique_label_count[keep],
                repetition_rate=targets.repetition_rate[keep],
                return_a=targets.return_a[keep],
                bridge=targets.bridge[keep],
                through=targets.through[keep],
                boundaries=[targets.boundaries[i] for i in keep],
                labels=[targets.labels[i] for i in keep],
                feature_paths=[targets.feature_paths[i] for i in keep],
            )

        self._targets = targets
        self._features = features
        return targets, features

    def _annotation_distances(self, targets: StructureTargets) -> tuple[np.ndarray, np.ndarray]:
        if self._form_distance is None:
            self._form_distance = pairwise_form_distance(targets.form_strings)
        if self._ssm_distance is None:
            self._ssm_distance = pairwise_ssm_distance(
                targets.boundaries,
                targets.labels,
                n_frames=self.ssm_frames,
            )
        return self._form_distance, self._ssm_distance

    def _model_representations(
        self,
        pl_module,
        features: list[torch.Tensor],
    ) -> dict[str, np.ndarray]:
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        zc_parts = []
        ztau_parts = []
        combo_parts = []
        mspf_parts = []

        with torch.inference_mode():
            for start in range(0, len(features), self.batch_size):
                seqs = [f.to(device=device, non_blocking=True) for f in features[start : start + self.batch_size]]
                lengths = torch.tensor([s.shape[0] for s in seqs], device=device)
                x = pad_sequence(seqs, batch_first=True)
                mask = torch.arange(x.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
                x_norm = pl_module.normalize_input(x) if hasattr(pl_module, "normalize_input") else x
                zc = (
                    pl_module.get_content_token(x_norm, mask=mask)
                    if hasattr(pl_module, "get_content_token")
                    else _masked_mean(x_norm, mask)
                )
                ztau = pl_module.encode(x_norm, mask=mask)[2]
                zc_cpu = zc.detach().cpu().float()
                ztau_cpu = ztau.detach().cpu().float()
                zc_parts.append(zc_cpu.numpy())
                ztau_parts.append(ztau_cpu.numpy())
                combo_parts.append(torch.cat([zc_cpu, ztau_cpu], dim=-1).numpy())
                x_norm_cpu = x_norm.detach().cpu()

                for i, length in enumerate(lengths.detach().cpu().tolist()):
                    curve = compute_mspf(
                        x_norm_cpu[i, : int(length)],
                        absolute=False,
                        n_points=self.mspf_points,
                    )
                    mspf_parts.append(curve)

        if was_training:
            pl_module.train()

        n = len(features)
        rng = np.random.default_rng(self.seed)
        return {
            "Random": rng.standard_normal((n, self.random_dim), dtype=np.float32),
            "zC": np.concatenate(zc_parts, axis=0),
            "MSPF_feat": np.asarray(mspf_parts, dtype=np.float32),
            "z_tau": np.concatenate(ztau_parts, axis=0),
            "zC_z_tau": np.concatenate(combo_parts, axis=0),
        }

    def _compute_metrics(self, pl_module) -> tuple[dict[str, float], list[dict[str, float | str]]]:
        targets, features = self._load_eval_data()
        reps = self._model_representations(pl_module, features)

        metrics: dict[str, float] = {}
        rows: list[dict[str, float | str]] = []

        split_specs = [("all", np.arange(len(targets.track_ids), dtype=np.int64))]
        if self.per_dataset:
            datasets = np.asarray([d.lower() for d in targets.datasets])
            for dataset in sorted(set(datasets.tolist())):
                idx = np.flatnonzero(datasets == dataset)
                if idx.size >= 3:
                    split_specs.append((dataset, idx))

        for split_name, idx in split_specs:
            split_targets = _subset_targets(targets, idx)
            split_reps = {name: rep[idx] for name, rep in reps.items()}
            split_metrics, split_rows = self._compute_split_metrics(split_targets, split_reps)
            prefix = "" if split_name == "all" else f"{split_name}/"
            metrics.update({f"{prefix}{key}": value for key, value in split_metrics.items()})
            for row in split_rows:
                rows.append({"dataset": split_name, **row})

        return metrics, rows

    def _compute_split_metrics(
        self,
        targets: StructureTargets,
        reps: dict[str, np.ndarray],
    ) -> tuple[dict[str, float], list[dict[str, float | str]]]:
        form_distance = pairwise_form_distance(targets.form_strings)
        ssm_distance = pairwise_ssm_distance(
            targets.boundaries,
            targets.labels,
            n_frames=self.ssm_frames,
        )
        content_distance = cosine_distance_matrix(reps["zC"])
        cc_mask = content_control_mask(content_distance, self.content_control_pool)
        metrics: dict[str, float] = {}
        rows: list[dict[str, float | str]] = []

        for name, rep in reps.items():
            rep_distance = cosine_distance_matrix(rep)
            retrieval = retrieval_metrics(
                rep_distance,
                form_distance,
                ks=(1, 5),
                epsilon=self.form_epsilon,
            )
            retrieval_cc = retrieval_metrics(
                rep_distance,
                form_distance,
                ks=(1, 5),
                epsilon=self.form_epsilon,
                candidate_mask=cc_mask,
            )
            corr_form = spearman_from_distance_matrices(rep_distance, form_distance)
            corr_ssm = spearman_from_distance_matrices(rep_distance, ssm_distance)
            probes = probe_scores(rep, targets, seed=self.seed)

            prefix = name
            for key, value in retrieval.items():
                metrics[f"{prefix}/{_retrieval_metric_name(key)}"] = value
            for key, value in retrieval_cc.items():
                metrics[f"{prefix}/{_retrieval_metric_name(key, content_controlled=True)}"] = value
            metrics[f"{prefix}/Spearman rho: embedding distances vs annotation form distances"] = corr_form
            metrics[f"{prefix}/Spearman rho: embedding distances vs annotation section-SSM distances"] = corr_ssm
            for key, value in probes.items():
                metrics[f"{prefix}/{_probe_metric_name(key)}"] = value

            rows.append({
                "rep": name,
                "D@1": retrieval.get("D@1", float("nan")),
                "D@5": retrieval.get("D@5", float("nan")),
                "FR@1": retrieval.get("FR@1", float("nan")),
                "FR@5": retrieval.get("FR@5", float("nan")),
                "Form rho": corr_form,
                "SSM rho": corr_ssm,
                **probes,
            })

        metrics["Track count"] = float(len(targets.track_ids))
        metrics["Unique annotation form count"] = float(len(set(targets.form_strings)))
        metrics["Mean annotated section count"] = float(np.mean(targets.section_count))
        metrics["Mean unique section label count"] = float(np.mean(targets.unique_label_count))
        metrics["Mean annotation repetition rate"] = float(np.mean(targets.repetition_rate))
        metrics["Fraction of tracks with return-to-A form"] = float(np.mean(targets.return_a))
        metrics["Fraction of tracks with bridge section"] = float(np.mean(targets.bridge))
        metrics["Fraction of tracks with through-composed form"] = float(np.mean(targets.through))
        return metrics, rows

    def _run(self, stage: str, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        if not Path(self.manifest_csv).exists():
            log.warning("Skipping CoverX: manifest does not exist: %s", self.manifest_csv)
            return

        metrics, _ = self._compute_metrics(pl_module)
        scalar_metrics = {f"{self.evaluation_suite}/{stage}/{key}": value for key, value in metrics.items()}
        pl_module.log_dict(
            scalar_metrics,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

        wandb_logger = _get_wandb_logger(trainer)
        if wandb_logger is not None:
            try:
                wandb_logger.experiment.log(scalar_metrics, step=trainer.global_step)
            except Exception as exc:
                log.warning("Skipping CoverX direct scalar log: %s", exc)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.test_only:
            return
        if self._check_epoch(trainer, pl_module):
            self._run("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_only:
            self._run("test", trainer, pl_module)
