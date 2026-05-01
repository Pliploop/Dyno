#!/usr/bin/env python3
"""
Script for generating audio from a prompt and retrieving nearest elements from a pre-extracted dataset.

This script generates audio from a prompt, extracts its embedding, and retrieves the top-k
most similar items from a pre-extracted dataset.
"""

import os
import argparse
import torch
import rootutils
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dora import hydra_main
import hydra
from dyno.utils import RankedLogger, register_resolvers, extras
from dyno.dataloading.dataloaders import TextAudioDataModule

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()
console = Console()


def load_config_from_source(config_source: str) -> DictConfig:
    """Load configuration from wandb or local file.
    
    :param config_source: Either a wandb run path (entity/project/run_id or entity/project/run_name)
                          prefixed with 'wandb://', or a local file path (yaml file).
    :return: DictConfig loaded from the source.
    """
    # Check if it's a wandb path (explicit prefix) or a local file
    if config_source.startswith('wandb://'):
        # Load from wandb
        try:
            import wandb
            api = wandb.Api()
            
            # Parse wandb path: entity/project/run_id or entity/project/run_name
            path = config_source[8:]  # Remove 'wandb://' prefix
            
            parts = path.split('/')
            if len(parts) == 3:
                entity, project, run_identifier = parts
            elif len(parts) == 2:
                # Assume default entity or project
                project, run_identifier = parts
                entity = None
            else:
                raise ValueError(f"Invalid wandb path format: {config_source}. Expected format: wandb://entity/project/run_id or wandb://entity/project/run_name")
            
            # Try to get run by ID first, then by name
            if entity:
                runs = api.runs(f'{entity}/{project}')
            else:
                runs = api.runs(project)
            
            run = None
            for r in runs:
                if r.id == run_identifier or r.name == run_identifier:
                    run = r
                    break
            
            if run is None:
                raise ValueError(f"Could not find wandb run: {run_identifier} in {entity}/{project if entity else project}")
            
            # Convert wandb config to OmegaConf
            config_dict = dict(run.config)
            fetched_config = OmegaConf.create(config_dict)
            log.info(f"Loaded config from wandb run: {run.name} ({run.id})")
            return fetched_config
        except ImportError:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        except Exception as e:
            raise ValueError(f"Failed to load config from wandb: {e}")
    else:
        # Load from local file
        if not os.path.exists(config_source):
            raise FileNotFoundError(f"Config file not found: {config_source}")
        
        if config_source.endswith('.yaml') or config_source.endswith('.yml'):
            fetched_config = OmegaConf.load(config_source)
            log.info(f"Loaded config from local file: {config_source}")
            return fetched_config
        else:
            raise ValueError(f"Unsupported config file format: {config_source}. Expected .yaml or .yml file")


def apply_config_overrides(cfg: DictConfig, fetched_config: DictConfig, override_keys: list) -> DictConfig:
    """Apply hard replacements from fetched_config to cfg based on override_keys.
    
    :param cfg: The original Hydra config.
    :param fetched_config: The config loaded from wandb or local file.
    :param override_keys: List of keys to override in cfg from fetched_config.
    :return: Modified cfg with overrides applied.
    """
    for key in override_keys:
        if key in fetched_config:
            # Hard replace: completely replace cfg[key] with fetched_config[key]
            cfg[key] = fetched_config[key]
            log.info(f"Replaced cfg.{key} with value from fetched config")
        else:
            log.warning(f"Key '{key}' not found in fetched config, skipping override")
    
    return cfg


def generate_and_extract_embedding(model, prompt, negative_prompt=None, guidance_scale=1.0, 
                                   num_steps=50, device='cuda:0', num_samples_per_prompt=1,
                                   audio_embedding=None, audio_guidance_scale=1.0):
    """Generate audio from prompt and extract its embedding.
    
    This follows the same processing as GenerationCallback:
    - Generate latents using model.inference()
    - Permute: (batch, channels, time) -> (batch, time, channels)
    - Mean across temporal dimension: (batch, time, channels) -> (batch, channels)
    
    :param audio_embedding: Optional audio embedding for audio similarity conditioning.
        Should be shaped (1, 1, D) where D is the embedding dimension.
    :param audio_guidance_scale: Scale for audio conditioning guidance.
    """
    log.info(f"Generating audio from prompt: '{prompt}'")

    if prompt == '':
        prompt = None
    
    with torch.no_grad():
        # Generate audio latents using model inference
        generated_latents = model.inference(
            prompt=[prompt],
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            negative_prompt=[negative_prompt] if negative_prompt else None,
            disable_progress=False,
            num_samples_per_prompt=num_samples_per_prompt,
            audio_embedding=audio_embedding,
            audio_guidance_scale=audio_guidance_scale,
        )
    
    # Process predictions: permute and take mean across temporal dimension
    # This matches GenerationCallback._generate_and_store()
    preds = generated_latents.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)
    print(f"preds shape: {preds.shape}")
    preds = preds.mean(dim=1)  # (batch, time, channels) -> (batch, channels)
    
    # Remove batch dimension
    generated_embedding = preds[0]  # [channels]
    
    log.info(f"Generated embedding shape: {generated_embedding.shape}")
    return generated_embedding.cpu()


def load_dataset_embeddings(model, dataset, preextracted_features=True, device='cuda:0'):
    """Load embeddings from pre-extracted dataset.
    
    This follows the same processing as GenerationCallback for ground truth:
    - Extract latents from audio
    - Mean across temporal dimension: (batch, time, channels) -> (batch, channels)
    """
    log.info(f"Loading embeddings from dataset with {len(dataset)} samples")
    
    embeddings = []
    file_paths = []

    #checkt the dataset annotations and remove the duplicate file_paths (get unique indicides)
    unique_indices = []
    for i in range(len(dataset.annotations)):
        if dataset.annotations[i]['file_path'] not in unique_indices:
            unique_indices.append(i)
    
    for i in tqdm(unique_indices, desc="Loading dataset embeddings"):
        item = dataset[i]
        
        if preextracted_features:
            # Use pre-extracted features (latents)
            audio_latents = item['audio']
            audio_embedding = audio_latents
        else:
            # Extract on-the-fly (not recommended for large datasets)
            audio = item['audio']
            audio_encoder = model.audio_encoder if hasattr(model, 'audio_encoder') else model.encoder_pair
            if hasattr(audio_encoder, 'get_audio_embedding_from_data'):
                audio_embedding = audio_encoder.get_audio_embedding_from_data(audio)
                if isinstance(audio_embedding, dict):
                    audio_embedding = audio_embedding.get('embedding_proj', 
                        audio_embedding.get('last_hidden_state'))
            # Process: mean across temporal dimension
            if len(audio_embedding.shape) > 1:
                audio_embedding = audio_embedding.mean(dim=1) if audio_embedding.shape[1] > 1 else audio_embedding.squeeze(1)
        
        embeddings.append(audio_embedding.cpu().mean(0))
        file_paths.append(item.get('file_path', f'item_{i}'))
    
    embeddings = torch.stack(embeddings)  # [N, D]
    log.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings, file_paths


def retrieve_top_k(query_embedding, dataset_embeddings, file_paths, k=10):
    """Retrieve top-k most similar items.
    
    This follows GenRetrieval callback:
    - Normalize embeddings
    - Compute cosine similarity via matrix multiplication
    """
    log.info(f"Computing similarities")
    
    # Normalize embeddings (like GenRetrieval callback)
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    dataset_embeddings = dataset_embeddings / dataset_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix: query to dataset (like GenRetrieval)
    # query_embedding: [D], dataset_embeddings: [N, D]
    similarities = query_embedding.unsqueeze(0) @ dataset_embeddings.t()  # [1, N]
    similarities = similarities.squeeze(0)  # [N]
    
    # Get top-k indices (higher is better for cosine similarity)
    top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)), largest=True)
    
    # Convert to lists
    top_k_indices = top_k_indices.tolist()
    top_k_values = top_k_values.tolist()
    top_k_paths = [file_paths[idx] for idx in top_k_indices]
    
    return top_k_paths, top_k_values, top_k_indices


def pretty_print_results(prompt, top_k_paths, top_k_values, k):
    """Pretty print retrieval results."""
    table = Table(title=f"Top-{k} Retrieval Results for Prompt: '{prompt}'")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("File Path", style="magenta")
    table.add_column("Similarity Score", style="green", justify="right")
    
    for i, (path, score) in enumerate(zip(top_k_paths, top_k_values), 1):
        table.add_row(str(i), path, f"{score:.4f}")
    
    console.print(table)


def inference_retrieval(cfg: DictConfig) -> Optional[dict]:
    """Main inference retrieval function.
    
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: Optional dict with results.
    """
    # Get parameters from config
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError("ckpt_path must be specified in config")
    
    prompt = cfg.get("prompt")
    if prompt is None:
        raise ValueError("prompt must be specified in config")
    
    negative_prompt = cfg.get("negative_prompt", None)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    num_inference_steps = cfg.get("num_inference_steps", 50)
    top_k = cfg.get("top_k", 10)
    device = cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    preextracted_features = cfg.get("preextracted_features", True)
    num_samples_per_prompt = cfg.get("num_samples_per_prompt", 1)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint if provided
    if ckpt_path:
        log.info(f"Loading checkpoint from: {ckpt_path}")
        if 's3://' in ckpt_path:
            from s3torchconnector import S3Checkpoint
            checkpoint = S3Checkpoint(region='us-east-1')
            with checkpoint.reader(ckpt_path) as f:
                ckpt = torch.load(f, map_location=device)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
        
        model.load_state_dict(ckpt, strict=True)
    
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Override engines if specified
    if cfg.get("_guidance_engine") and cfg._guidance_engine is not None:
        guidance_engine_cfg = cfg._guidance_engine
        # Handle case where it's a string path (config group reference)
        if isinstance(guidance_engine_cfg, str):
            # Try to resolve as config group reference (e.g., "inference/engines/cfg_guidance")
            try:
                from pathlib import Path
                # Find configs directory relative to project root
                project_root = Path(__file__).parent.parent
                config_dir = project_root / "configs"
                parts = guidance_engine_cfg.split('/')
                if len(parts) >= 2:
                    # Navigate to the config group
                    group_path = '/'.join(parts[:-1])
                    config_name = parts[-1]
                    # Load the config file directly
                    config_path = config_dir / group_path / f"{config_name}.yaml"
                    if config_path.exists():
                        guidance_engine_cfg = OmegaConf.load(config_path)
                        log.info(f"Loaded guidance engine config from: {config_path}")
                    else:
                        log.warning(f"Config file not found: {config_path}. Ignoring _guidance_engine.")
                        guidance_engine_cfg = None
                else:
                    log.warning(f"Invalid config group path: {guidance_engine_cfg}. Ignoring.")
                    guidance_engine_cfg = None
            except Exception as e:
                log.warning(f"Failed to resolve guidance engine config group '{guidance_engine_cfg}': {e}. Ignoring.")
                guidance_engine_cfg = None
        
        if guidance_engine_cfg is not None and not isinstance(guidance_engine_cfg, str):
            log.info(f"Overriding guidance engine with: {guidance_engine_cfg.get('_target_', 'unknown')}")
            guidance_engine = hydra.utils.instantiate(guidance_engine_cfg)
            # Set backbone from model
            if hasattr(guidance_engine, 'set_backbone'):
                guidance_engine.set_backbone(model.backbone)
            elif hasattr(guidance_engine, '_backbone'):
                guidance_engine._backbone = model.backbone
            model.guidance_engine = guidance_engine
    
    if cfg.get("_edit_engine") and cfg._edit_engine is not None:
        edit_engine_cfg = cfg._edit_engine
        # Handle case where it's a string path (config group reference)
        if isinstance(edit_engine_cfg, str):
            # Try to resolve as config group reference (e.g., "inference/engines/ddim_inversion")
            try:
                from pathlib import Path
                # Find configs directory relative to project root
                project_root = Path(__file__).parent.parent
                config_dir = project_root / "configs"
                parts = edit_engine_cfg.split('/')
                if len(parts) >= 2:
                    # Navigate to the config group
                    group_path = '/'.join(parts[:-1])
                    config_name = parts[-1]
                    # Load the config file directly
                    config_path = config_dir / group_path / f"{config_name}.yaml"
                    if config_path.exists():
                        edit_engine_cfg = OmegaConf.load(config_path)
                        log.info(f"Loaded edit engine config from: {config_path}")
                    else:
                        log.warning(f"Config file not found: {config_path}. Ignoring _edit_engine.")
                        edit_engine_cfg = None
                else:
                    log.warning(f"Invalid config group path: {edit_engine_cfg}. Ignoring.")
                    edit_engine_cfg = None
            except Exception as e:
                log.warning(f"Failed to resolve edit engine config group '{edit_engine_cfg}': {e}. Ignoring.")
                edit_engine_cfg = None
        
        if edit_engine_cfg is not None and not isinstance(edit_engine_cfg, str):
            log.info(f"Overriding edit engine with: {edit_engine_cfg.get('_target_', 'unknown')}")
            edit_engine = hydra.utils.instantiate(edit_engine_cfg)
            # Set dependencies from model
            if hasattr(edit_engine, 'set_backbone'):
                edit_engine.set_backbone(model.backbone)
            elif hasattr(edit_engine, '_backbone'):
                edit_engine._backbone = model.backbone
            
            if hasattr(edit_engine, 'set_text_encoder'):
                edit_engine.set_text_encoder(model.text_encoder)
            elif hasattr(edit_engine, '_text_encoder'):
                edit_engine._text_encoder = model.text_encoder
            
            # Set guidance engine if available
            if hasattr(edit_engine, 'guidance_engine'):
                if model.guidance_engine is not None:
                    edit_engine.guidance_engine = model.guidance_engine
                elif hasattr(model, '_guidance_engine') and model._guidance_engine is not None:
                    edit_engine.guidance_engine = model._guidance_engine
            
            model.edit_engine = edit_engine
    
    # Load dataset
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(None)
    # test dataset
    dataset = datamodule.test_datasets[0]
    
    if dataset is None:
        raise ValueError("Could not load dataset")
    
    log.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Generate and extract embedding
    query_embedding = generate_and_extract_embedding(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_steps=num_inference_steps,
        device=device,
        num_samples_per_prompt=num_samples_per_prompt
    )
    
    # Load dataset embeddings
    dataset_embeddings, file_paths = load_dataset_embeddings(
        model=model,
        dataset=dataset,
        preextracted_features=preextracted_features,
        device=device
    )
    
    # Retrieve top-k
    top_k_paths, top_k_values, top_k_indices = retrieve_top_k(
        query_embedding=query_embedding,
        dataset_embeddings=dataset_embeddings,
        file_paths=file_paths,
        k=top_k
    )
    
    # Pretty print results
    pretty_print_results(prompt, top_k_paths, top_k_values, top_k)
    
    return {
        'top_k_paths': top_k_paths,
        'top_k_values': top_k_values,
        'top_k_indices': top_k_indices
    }


@hydra_main(version_base="1.3", config_path="../configs", config_name="inference_retrieval.yaml")
def main(cfg: DictConfig) -> Optional[dict]:
    """Main entry point for inference retrieval.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional dict with results.
    """
    # handle A100 GPUs
    if torch.cuda.is_available() and ("A100" in torch.cuda.get_device_name() or "A5000" in torch.cuda.get_device_name()):
        torch.set_float32_matmul_precision("high")

    # avoid annoying multiprocessing errors
    torch.multiprocessing.set_sharing_strategy('file_system')

    # prevent annoying warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load and apply config overrides if specified
    if cfg.get("_fetch_config"):
        log.info(f"Fetching config from: {cfg._fetch_config}")
        fetched_config = load_config_from_source(cfg._fetch_config)
        
        if cfg.get("_yaml_overrides"):
            override_keys = cfg._yaml_overrides if isinstance(cfg._yaml_overrides, list) else [cfg._yaml_overrides]
            log.info(f"Applying config overrides for keys: {override_keys}")
            cfg = apply_config_overrides(cfg, fetched_config, override_keys)
        else:
            log.warning("_fetch_config specified but _yaml_overrides is not set. No overrides will be applied.")

    # apply extra utilities
    extras(cfg)

    # perform inference retrieval
    result = inference_retrieval(cfg)

    return result


if __name__ == '__main__':
    main()
