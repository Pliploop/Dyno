"""Test-only callback for the paper structure-probing protocol."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from lightning import Callback

from dyno.evaluation.structure_probe import run_structure_probe


log = logging.getLogger(__name__)


class StructureProbeCallback(Callback):
    def __init__(
        self,
        datasets: dict,
        folds_csv: str,
        feature_root: str,
        annotation_root: str,
        output_dir: str,
        probe_inputs: list[str],
        frame_rate: str | float = 1.0,
        window_seconds: float = 30.0,
        hop_seconds: float = 30.0,
        position_dim: int = 32,
        batch_size: int = 8,
        epochs: int = 100,
        warmup_epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        threshold_grid: list[float] | None = None,
        num_folds: int = 8,
        num_workers: int = 0,
        max_tracks: int | None = None,
        salami_boundary_layers: list[str] | None = None,
        trim_boundaries: bool = False,
        run_on_train_end: bool = True,
        run_on_test_end: bool = True,
        run_once: bool = True,
        use_best_checkpoint_on_train_end: bool = True,
    ):
        self.datasets = datasets
        self.folds_csv = folds_csv
        self.feature_root = feature_root
        self.annotation_root = annotation_root
        self.output_dir = output_dir
        self.probe_inputs = probe_inputs
        self.frame_rate = frame_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.position_dim = position_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold_grid = threshold_grid or [index / 10 for index in range(1, 10)]
        self.num_folds = num_folds
        self.num_workers = num_workers
        self.max_tracks = max_tracks
        self.salami_boundary_layers = salami_boundary_layers or ["uppercase", "lowercase"]
        self.trim_boundaries = trim_boundaries
        self.run_on_train_end = run_on_train_end
        self.run_on_test_end = run_on_test_end
        self.run_once = run_once
        self.use_best_checkpoint_on_train_end = use_best_checkpoint_on_train_end
        self._has_run = False

    def _load_best_checkpoint(self, trainer, pl_module) -> None:
        if not self.use_best_checkpoint_on_train_end:
            return
        callback = getattr(trainer, "checkpoint_callback", None)
        checkpoint_path = callback.best_model_path if callback is not None else ""
        if not checkpoint_path:
            log.warning("No best checkpoint is available; probing the final in-memory weights")
            return
        payload = torch.load(checkpoint_path, map_location=pl_module.device, weights_only=False)
        pl_module.load_state_dict(payload["state_dict"], strict=True)
        log.info("Loaded best checkpoint for end-of-training structure probe: %s", checkpoint_path)

    def _run(self, stage: str, trainer, pl_module) -> None:
        if not trainer.is_global_zero or (self.run_once and self._has_run):
            return
        self._has_run = True
        all_metrics: dict[str, float] = {}
        all_rows: list[dict] = []
        for dataset, manifest_csv in self.datasets.items():
            layers = self.salami_boundary_layers if dataset.lower() == "salami" else ["functions"]
            for layer in layers:
                metrics, rows = run_structure_probe(
                    pl_module,
                    manifest_csv=manifest_csv,
                    folds_csv=self.folds_csv,
                    feature_root=self.feature_root,
                    annotation_root=self.annotation_root,
                    dataset=dataset,
                    probe_inputs=self.probe_inputs,
                    frame_rate=self.frame_rate,
                    window_seconds=self.window_seconds,
                    hop_seconds=self.hop_seconds,
                    position_dim=self.position_dim,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    warmup_epochs=self.warmup_epochs,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                    threshold_grid=self.threshold_grid,
                    num_folds=self.num_folds,
                    num_workers=self.num_workers,
                    max_tracks=self.max_tracks,
                    salami_boundary_layer=layer,
                    trim_boundaries=self.trim_boundaries,
                    device=pl_module.device,
                )
                if layer == "lowercase":
                    metrics = {
                        key.replace("/salami/", "/salami_fine/"): value
                        for key, value in metrics.items()
                    }
                all_metrics.update(metrics)
                all_rows.extend(rows)

        if stage == "test":
            pl_module.log_dict(all_metrics, on_epoch=True, sync_dist=False)
        loggers = getattr(trainer, "loggers", None) or [getattr(trainer, "logger", None)]
        for logger in loggers:
            if logger is not None:
                logger.log_metrics(all_metrics, step=trainer.global_step)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"structure_probe_{stage}_folds.csv"
        if all_rows:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)
        log.info("Wrote structure-probe fold results to %s", output_path)

    def on_train_end(self, trainer, pl_module) -> None:
        if not self.run_on_train_end:
            return
        self._load_best_checkpoint(trainer, pl_module)
        self._run("train_end", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if self.run_on_test_end:
            self._run("test", trainer, pl_module)
