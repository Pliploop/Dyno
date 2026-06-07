"""Test-only callback for the paper structure-probing protocol."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import subprocess
import sys

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
        frame_rate: float = 2.0,
        model_rate: float = 1.0,
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
        run_in_subprocess: bool = True,
        probe_encoder: str = "muq",
    ):
        self.datasets = datasets
        self.folds_csv = folds_csv
        self.feature_root = feature_root
        self.annotation_root = annotation_root
        self.output_dir = output_dir
        self.probe_inputs = probe_inputs
        self.frame_rate = frame_rate
        self.model_rate = model_rate
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
        self.run_in_subprocess = run_in_subprocess
        self.probe_encoder = probe_encoder

    def _checkpoint_path(self, trainer) -> str | None:
        checkpoint_path = getattr(trainer, "ckpt_path", None)
        if checkpoint_path:
            return str(checkpoint_path)
        callback = getattr(trainer, "checkpoint_callback", None)
        if callback is not None and callback.best_model_path:
            return str(callback.best_model_path)
        return None

    def _run_subprocess(self, trainer) -> bool:
        checkpoint_path = self._checkpoint_path(trainer)
        if checkpoint_path is None:
            log.warning("Cannot launch structure probe subprocess: no checkpoint path is available")
            return False
        command = [
            sys.executable,
            "-m",
            "dyno.evaluate_structure_probe",
            f"run_ref={checkpoint_path}",
            f"probe_encoder={self.probe_encoder}",
            f"probe.epochs={self.epochs}",
            f"probe.warmup_epochs={self.warmup_epochs}",
            f"probe.batch_size={self.batch_size}",
            f"probe.num_workers={self.num_workers}",
            f"probe.max_tracks={'null' if self.max_tracks is None else self.max_tracks}",
        ]
        project_root = os.environ.get("PROJECT_ROOT", str(Path.cwd()))
        log.info("Launching isolated structure probe: %s", " ".join(command))
        subprocess.run(command, cwd=project_root, check=True)
        return True

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if self.run_in_subprocess and self._run_subprocess(trainer):
            return
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
                    model_rate=self.model_rate,
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
                    device=pl_module.device,
                )
                if layer == "lowercase":
                    metrics = {
                        key.replace("/salami/", "/salami_fine/"): value
                        for key, value in metrics.items()
                    }
                all_metrics.update(metrics)
                all_rows.extend(rows)

        pl_module.log_dict(all_metrics, on_epoch=True, sync_dist=False)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "structure_probe_folds.csv"
        if all_rows:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)
        log.info("Wrote structure-probe fold results to %s", output_path)
