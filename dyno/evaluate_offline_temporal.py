"""Standalone cross-encoder MSPF and retrieval evaluation."""

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
from dyno.evaluation.offline_temporal import run_offline_temporal_evaluation
from dyno.utils import register_resolvers
from dyno.utils.experiment_registry import resolve_experiment_reference


register_resolvers()


@hydra.main(version_base="1.3", config_path="../configs", config_name="offline_temporal.yaml")
def main(cfg: DictConfig) -> None:
    resolved = resolve_experiment_reference(
        cfg.run_ref,
        cfg.experiment_registry,
        cfg.checkpoint_preference,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(cfg, resolved.checkpoint).to(device)
    metrics, rows = run_offline_temporal_evaluation(
        model,
        manifests=cfg.evaluation.manifests,
        feature_root=cfg.evaluation.feature_root,
        token_encoder=cfg.evaluation.token_encoder,
        source_rate=cfg.evaluation.source_rate,
        model_rate=cfg.evaluation.model_rate,
        max_tracks=cfg.evaluation.max_tracks,
        max_pairs=cfg.evaluation.max_pairs,
        mspf_points=cfg.evaluation.mspf_points,
        mspf_max_frames=cfg.evaluation.mspf_max_frames,
        top_k=cfg.evaluation.top_k,
        n_queries=cfg.evaluation.n_queries,
        seed=cfg.seed,
        device=device,
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(metrics), output_dir / "offline_temporal_metrics.yaml")
    if not rows:
        raise RuntimeError("Offline temporal evaluation produced no retrieval rows")
    with (output_dir / "temporal_retrieval.csv").open(
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
            job_type="offline-temporal",
            tags=list(cfg.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(output_dir),
        )
        run.log(metrics)
        run.log({"paper.temporal_retrieval/test/cases": wandb.Table(dataframe=pd.DataFrame(rows))})
        run.finish()

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
