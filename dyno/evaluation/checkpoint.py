"""Checkpoint-backed model loading for standalone evaluations."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def checkpoint_run_config(checkpoint: Path) -> Path | None:
    for parent in checkpoint.parents:
        for relative in (".hydra/config.yaml", "config.yaml"):
            candidate = parent / relative
            if candidate.is_file():
                return candidate
    return None


def load_checkpoint_run_config(checkpoint: Path) -> DictConfig | None:
    path = checkpoint_run_config(checkpoint)
    return OmegaConf.load(path) if path is not None else None


def load_checkpoint_model(cfg: DictConfig, checkpoint: Path):
    model_cfg = cfg.model
    if cfg.get("use_checkpoint_config", True):
        saved = load_checkpoint_run_config(checkpoint)
        if saved is not None:
            if saved.get("model") is not None:
                model_cfg = saved.model
    model = hydra.utils.instantiate(model_cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    return model
