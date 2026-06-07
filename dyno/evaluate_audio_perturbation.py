"""Standalone controlled perturbation evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dyno.evaluation.checkpoint import load_checkpoint_model
from dyno.evaluation.audio_perturbation import run_audio_perturbation_evaluation
from dyno.utils import register_resolvers
from dyno.utils.experiment_registry import resolve_experiment_reference


register_resolvers()


@hydra.main(version_base="1.3", config_path="../configs", config_name="audio_perturbation.yaml")
def main(cfg: DictConfig) -> None:
    resolved = resolve_experiment_reference(
        cfg.run_ref,
        cfg.experiment_registry,
        cfg.checkpoint_preference,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(cfg, resolved.checkpoint).to(device)
    encoder = hydra.utils.instantiate(cfg.audio_encoder).to(device)
    metrics, rows = run_audio_perturbation_evaluation(
        model,
        encoder,
        manifest_csv=cfg.evaluation.manifest_csv,
        structure_root=cfg.paths.structure_root,
        conditions=tuple(cfg.evaluation.conditions),
        max_tracks=cfg.evaluation.max_tracks,
        sample_rate=cfg.evaluation.sample_rate,
        window_seconds=cfg.evaluation.window_seconds,
        hop_seconds=cfg.evaluation.hop_seconds,
        batch_size=cfg.evaluation.batch_size,
        latent_chunk_frames=cfg.evaluation.latent_chunk_frames,
        mspf_points=cfg.evaluation.mspf_points,
        mspf_max_frames=cfg.evaluation.mspf_max_frames,
        seed=cfg.seed,
        device=device,
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(metrics), output_dir / "audio_perturbation_metrics.yaml")
    if not rows:
        raise RuntimeError("Audio perturbation evaluation produced no track rows")
    with (output_dir / "audio_perturbation_tracks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if cfg.wandb.enabled:
        import pandas as pd
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            job_type="audio-perturbation",
            tags=list(cfg.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(output_dir),
        )
        run.log(metrics)
        run.log(
            {
                "paper.perturbation_sensitivity/test/tracks": wandb.Table(
                    dataframe=pd.DataFrame(rows)
                )
            }
        )
        run.finish()

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
