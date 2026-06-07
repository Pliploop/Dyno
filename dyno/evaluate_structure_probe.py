"""Standalone Hydra entry point for paper structure probing."""

from __future__ import annotations

import csv
from pathlib import Path

import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra

from dyno.evaluation.checkpoint import load_checkpoint_model, load_checkpoint_run_config
from dyno.evaluation.structure_probe import run_structure_probe
from dyno.utils import register_resolvers
from dyno.utils.experiment_registry import resolve_experiment_reference


register_resolvers()


@hydra.main(version_base="1.3", config_path="../configs", config_name="structure_probe.yaml")
def main(cfg: DictConfig) -> None:
    resolved = resolve_experiment_reference(
        cfg.run_ref,
        registry_path=cfg.experiment_registry,
        checkpoint_preference=cfg.checkpoint_preference,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    saved_cfg = load_checkpoint_run_config(resolved.checkpoint)
    if saved_cfg is None or saved_cfg.get("data") is None:
        raise ValueError(
            "Could not recover the checkpoint training data signature. "
            "The structure probe requires the original embedding encoder and rate."
        )
    training_encoder = str(saved_cfg.data.embedding_encoder)
    training_rate = str(saved_cfg.data.embedding_rate)
    requested_encoder = str(cfg.probe_encoder)
    requested_rate = str(cfg.probe_rate)
    if requested_encoder not in {"auto", training_encoder}:
        raise ValueError(
            f"Probe encoder {requested_encoder!r} does not match checkpoint "
            f"encoder {training_encoder!r}"
        )
    if requested_rate not in {"auto", training_rate}:
        raise ValueError(
            f"Probe rate {requested_rate!r} does not match checkpoint "
            f"rate {training_rate!r}"
        )
    with open_dict(cfg):
        cfg.probe_encoder = training_encoder
        cfg.probe_rate = training_rate
    model = load_checkpoint_model(cfg, resolved.checkpoint).to(device)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    all_metrics: dict[str, float] = {}
    all_rows: list[dict] = []

    for dataset, manifest_csv in cfg.probe.datasets.items():
        layers = cfg.probe.salami_boundary_layers if dataset.lower() == "salami" else ["functions"]
        for layer in layers:
            metrics, rows = run_structure_probe(
                model,
                manifest_csv=manifest_csv,
                folds_csv=cfg.probe.folds_csv,
                feature_root=cfg.probe.feature_root,
                annotation_root=cfg.probe.annotation_root,
                dataset=dataset,
                probe_inputs=cfg.probe.probe_inputs,
                frame_rate=cfg.probe.frame_rate,
                batch_size=cfg.probe.batch_size,
                epochs=cfg.probe.epochs,
                warmup_epochs=cfg.probe.warmup_epochs,
                learning_rate=cfg.probe.learning_rate,
                weight_decay=cfg.probe.weight_decay,
                model_dim=cfg.probe.model_dim,
                num_heads=cfg.probe.num_heads,
                ffn_dim=cfg.probe.ffn_dim,
                dropout=cfg.probe.dropout,
                threshold_grid=cfg.probe.threshold_grid,
                num_folds=cfg.probe.num_folds,
                num_workers=cfg.probe.num_workers,
                max_tracks=cfg.probe.max_tracks,
                salami_boundary_layer=layer,
                trim_boundaries=cfg.probe.trim_boundaries,
                device=device,
            )
            if layer == "lowercase":
                metrics = {
                    key.replace("/salami/", "/salami_fine/"): value
                    for key, value in metrics.items()
                }
            all_metrics.update(metrics)
            all_rows.extend(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(all_metrics), output_dir / "structure_probe_metrics.yaml")
    if all_rows:
        with (output_dir / "structure_probe_folds.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)

    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            job_type="structure-probe",
            tags=list(cfg.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(output_dir),
        )
        run.log(all_metrics)
        run.log({"structure_probe_folds": wandb.Table(dataframe=__import__("pandas").DataFrame(all_rows))})
        run.finish()

    for key, value in sorted(all_metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
