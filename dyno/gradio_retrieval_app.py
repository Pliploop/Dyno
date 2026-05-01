#!/usr/bin/env python3
"""
Gradio app for text-to-audio retrieval on pre-extracted datasets.

Features:
- Fetch runs from WandB filtered by tags (default: ['gradio'])
- Load full Hydra config from WandB for a selected run and override `model`
- Discover checkpoints from `sagemaker.estimator.ckpt_dir` (or fallback to `paths.ckpt_dir`)
- Instantiate datamodule from Hydra `data` config and expose test datasets as options
- Run retrieval using the existing inference_retrieval utilities
"""

import boto3
from botocore.exceptions import ClientError
import soundfile as sf


import os
import json
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import rootutils
from omegaconf import OmegaConf
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import gradio as gr
import hydra
from dora import hydra_main
from omegaconf import DictConfig, OmegaConf
import logging

from dyno.utils import RankedLogger, extras, register_resolvers
from dyno.dataloading.dataloaders import TextAudioDataModule
from dyno.inference_retrieval import (
    load_config_from_source,
    apply_config_overrides,
    generate_and_extract_embedding,
)

# S3 base path - the dataset subdirectory is already in the file_path
S3_AUDIO_BASE = 's3://maml-aimcdt/datasets/spamr-training/audio'

# Legacy mapping kept for reference (not used directly anymore)
DATASET_TO_S3 = {
    'song_describer_val': f'{S3_AUDIO_BASE}/song-describer',
    'musiccaps_val': f'{S3_AUDIO_BASE}/musiccaps',
    'maxcaps': f'{S3_AUDIO_BASE}/maxcaps',
    'yt8m_val': f'{S3_AUDIO_BASE}/yt8m',
}

# Embedding cache settings
CACHE_DIR = ".temp/cache"
EMBEDDING_CHUNK_SIZE = 25000

log = RankedLogger(__name__, rank_zero_only=True)


# ============================================================================
# Embedding Caching Functions
# ============================================================================

def get_cache_dir() -> str:
    """Get or create the cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_cache_paths(dataset_name: str) -> Tuple[str, str]:
    """
    Get paths for cached embeddings and metadata.
    
    :param dataset_name: Name of the dataset
    :return: Tuple of (embeddings_path, metadata_path)
    """
    cache_dir = get_cache_dir()
    embeddings_path = os.path.join(cache_dir, f"{dataset_name}_embeddings.npy")
    metadata_path = os.path.join(cache_dir, f"{dataset_name}_metadata.json")
    return embeddings_path, metadata_path


def is_dataset_cached(dataset_name: str) -> bool:
    """Check if a dataset's embeddings are already cached."""
    embeddings_path, metadata_path = get_cache_paths(dataset_name)
    return os.path.exists(embeddings_path) and os.path.exists(metadata_path)


def cache_dataset_embeddings(
    dataset,
    dataset_name: str,
    preextracted_features: bool = True,
    progress: Optional[gr.Progress] = None,
    progress_offset: float = 0.0,
    progress_scale: float = 1.0,
) -> Tuple[str, str]:
    """
    Cache dataset embeddings to a .npy file for memory-mapped access.
    
    :param dataset: The dataset to extract embeddings from
    :param dataset_name: Name of the dataset (used for cache filename)
    :param preextracted_features: Whether to use pre-extracted features
    :param progress: Optional Gradio progress tracker
    :param progress_offset: Offset for progress bar (for multi-dataset caching)
    :param progress_scale: Scale for progress bar (for multi-dataset caching)
    :return: Tuple of (embeddings_path, metadata_path)
    """
    embeddings_path, metadata_path = get_cache_paths(dataset_name)
    
    # Skip if already cached
    if is_dataset_cached(dataset_name):
        log.info(f"Dataset '{dataset_name}' already cached at {embeddings_path}")
        if progress:
            progress(progress_offset + progress_scale, desc=f"✓ {dataset_name} (cached)")
        return embeddings_path, metadata_path
    
    log.info(f"Caching embeddings for dataset '{dataset_name}' ({len(dataset)} samples)")
    
    embeddings = []
    file_paths = []
    
    # Get unique indices (avoiding duplicate file paths)
    unique_indices = []
    seen_paths = set()
    for i in range(len(dataset.annotations) if hasattr(dataset, 'annotations') else len(dataset)):
        if hasattr(dataset, 'annotations'):
            fp = dataset.annotations[i].get('file_path', f'item_{i}')
        else:
            fp = f'item_{i}'
        if fp not in seen_paths:
            seen_paths.add(fp)
            unique_indices.append(i)
    
    total = len(unique_indices)
    
    # Extract embeddings with progress
    for idx, i in enumerate(unique_indices):
        item = dataset[i]
        
        if preextracted_features:
            audio_latents = item['audio']
            audio_embedding = audio_latents
        else:
            audio_embedding = item['audio']
        
        # Take mean across temporal dimension if needed
        if isinstance(audio_embedding, torch.Tensor):
            audio_embedding = audio_embedding.cpu().float().mean(0)
        elif isinstance(audio_embedding, np.ndarray):
            audio_embedding = audio_embedding.mean(0)
        
        embeddings.append(audio_embedding.numpy() if isinstance(audio_embedding, torch.Tensor) else audio_embedding)
        file_paths.append(item.get('file_path', f'item_{i}'))
        
        # Update progress
        if progress and idx % 100 == 0:  # Update every 100 items to avoid overhead
            pct = progress_offset + (idx / total) * progress_scale
            progress(pct, desc=f"Caching {dataset_name}: {idx}/{total}")
    
    # Stack and save embeddings as .npy
    embeddings_array = np.stack(embeddings).astype(np.float32)  # [N, D]
    np.save(embeddings_path, embeddings_array)
    
    # Save metadata (file paths, shape info)
    metadata = {
        'file_paths': file_paths,
        'shape': list(embeddings_array.shape),
        'dtype': str(embeddings_array.dtype),
        'num_samples': len(file_paths),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    if progress:
        progress(progress_offset + progress_scale, desc=f"✓ {dataset_name} cached ({total} samples)")
    
    log.info(f"Cached {len(embeddings)} embeddings to {embeddings_path} (shape: {embeddings_array.shape})")
    return embeddings_path, metadata_path


def cache_all_datasets(
    datamodule: TextAudioDataModule,
    preextracted_features: bool = True,
    progress: Optional[gr.Progress] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Cache embeddings for all test datasets at app instantiation.
    
    :param datamodule: The datamodule containing test datasets
    :param preextracted_features: Whether to use pre-extracted features
    :param progress: Optional Gradio progress tracker
    :return: Dict mapping dataset_name -> (embeddings_path, metadata_path)
    """
    cache_paths = {}
    
    if not hasattr(datamodule, 'test_datasets') or not datamodule.test_datasets:
        log.warning("No test datasets found in datamodule")
        return cache_paths
    
    # Get dataset names
    if hasattr(datamodule, 'test_dataset_names'):
        dataset_names = datamodule.test_dataset_names
    else:
        dataset_names = [f"dataset_{i}" for i in range(len(datamodule.test_datasets))]
    
    num_datasets = len(dataset_names)
    
    for ds_idx, (name, dataset) in enumerate(zip(dataset_names, datamodule.test_datasets)):
        # Calculate progress offset and scale for this dataset
        progress_offset = ds_idx / num_datasets
        progress_scale = 1.0 / num_datasets
        
        if progress:
            progress(progress_offset, desc=f"Processing dataset {ds_idx + 1}/{num_datasets}: {name}")
        
        embeddings_path, metadata_path = cache_dataset_embeddings(
            dataset=dataset,
            dataset_name=name,
            preextracted_features=preextracted_features,
            progress=progress,
            progress_offset=progress_offset,
            progress_scale=progress_scale,
        )
        cache_paths[name] = (embeddings_path, metadata_path)
    
    if progress:
        progress(1.0, desc=f"✓ All {num_datasets} datasets cached")
    
    return cache_paths


def load_cached_metadata(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Load cached metadata for a dataset.
    
    :param dataset_name: Name of the dataset
    :return: Metadata dict or None if not cached
    """
    _, metadata_path = get_cache_paths(dataset_name)
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def retrieve_top_k_chunked(
    query_embedding: torch.Tensor,
    dataset_name: str,
    k: int = 10,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
) -> Tuple[List[str], List[float], List[int]]:
    """
    Retrieve top-k results using memory-mapped embeddings with chunked processing.
    
    This avoids loading the entire embedding matrix into memory by:
    1. Memory-mapping the .npy file
    2. Processing in chunks of `chunk_size`
    3. Maintaining a running top-k across all chunks
    
    :param query_embedding: Query embedding tensor [D]
    :param dataset_name: Name of the dataset to search
    :param k: Number of top results to return
    :param chunk_size: Number of embeddings to process per chunk
    :return: Tuple of (top_k_paths, top_k_values, top_k_indices)
    """
    embeddings_path, metadata_path = get_cache_paths(dataset_name)
    
    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        raise ValueError(f"Dataset '{dataset_name}' is not cached. Run cache_all_datasets first.")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    file_paths = metadata['file_paths']
    num_samples = metadata['num_samples']
    
    # Memory-map the embeddings file
    embeddings_mmap = np.load(embeddings_path, mmap_mode='r')  # [N, D]
    
    # Normalize query embedding
    query_np = query_embedding.cpu().numpy().astype(np.float32)
    query_np = query_np / np.linalg.norm(query_np)
    
    # Track global top-k across all chunks
    all_scores = []
    all_indices = []
    
    # Process in chunks
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        
        # Load chunk from memmap (this actually reads into memory)
        chunk_embeddings = embeddings_mmap[start_idx:end_idx].copy()  # [chunk_size, D]
        
        # Normalize chunk embeddings
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        chunk_embeddings = chunk_embeddings / norms
        
        # Compute similarities: query @ embeddings.T
        similarities = chunk_embeddings @ query_np  # [chunk_size]
        
        # Get top-k from this chunk
        chunk_k = min(k, len(similarities))
        top_chunk_indices = np.argpartition(similarities, -chunk_k)[-chunk_k:]
        top_chunk_indices = top_chunk_indices[np.argsort(similarities[top_chunk_indices])[::-1]]
        
        # Convert to global indices and store
        for local_idx in top_chunk_indices:
            global_idx = start_idx + local_idx
            all_scores.append(similarities[local_idx])
            all_indices.append(global_idx)
    
    # Get final global top-k from all chunk results
    all_scores = np.array(all_scores)
    all_indices = np.array(all_indices)
    
    final_k = min(k, len(all_scores))
    top_k_positions = np.argpartition(all_scores, -final_k)[-final_k:]
    top_k_positions = top_k_positions[np.argsort(all_scores[top_k_positions])[::-1]]
    
    top_k_indices = all_indices[top_k_positions].tolist()
    top_k_values = all_scores[top_k_positions].tolist()
    top_k_paths = [file_paths[idx] for idx in top_k_indices]
    
    log.info(f"Retrieved top-{k} from {num_samples} samples using chunked memmap (chunk_size={chunk_size})")
    
    return top_k_paths, top_k_values, top_k_indices


# ============================================================================
# Label formatting and other utilities
# ============================================================================

def format_label_value(rank: int, score: float, tags: Optional[str] = None) -> dict:
    """
    Format the score as a dict for gr.Label component.
    
    :param rank: Result rank (1-10)
    :param score: Similarity score
    :param tags: Optional tags string
    :return: Dict for gr.Label with label and confidence
    """
    # Create a label string with rank and optional tags
    tag_str = f" ({tags})" if tags else ""
    label = f"#{rank}{tag_str}"
    
    # Return dict format for gr.Label - shows confidence bar
    return {label: float(score)}


def _get_s3_client():
    """Get boto3 S3 client."""
    import boto3
    return boto3.client("s3")


def download_s3_to_temp(s3_uri: str, temp_dir: str) -> Optional[str]:
    """
    Download an S3 file to a local temp directory.
    
    :param s3_uri: Full S3 URI (s3://bucket/key)
    :param temp_dir: Local temp directory to save the file
    :return: Local path to downloaded file, or None on error
    """
    if not s3_uri or not s3_uri.startswith("s3://"):
        log.warning(f"Invalid S3 URI: {s3_uri}")
        return None
    
    try:
        s3 = _get_s3_client()
        # Parse s3://bucket/key
        path = s3_uri[5:]  # Remove "s3://"
        bucket, key = path.split("/", 1)
        
        # Create local path preserving filename
        filename = os.path.basename(key)
        local_path = os.path.join(temp_dir, filename)
        
        # Avoid re-downloading if exists
        if os.path.exists(local_path):
            return local_path
        
        log.info(f"Downloading {s3_uri} -> {local_path}")
        s3.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        log.warning(f"Failed to download {s3_uri}: {e}")
        return None


def build_s3_audio_uri(file_path: str, dataset_name: str, cfg: DictConfig) -> Optional[str]:
    """
    Convert a local file path to an S3 URI.
    
    The file_path already contains the dataset subdirectory (e.g., /opt/ml/data/maxcaps/1234.npy).
    We replace data_dir with S3_AUDIO_BASE to get the correct S3 path.
    
    :param file_path: Local file path (e.g., /opt/ml/data/maxcaps/audio/file.npy)
    :param dataset_name: The currently selected dataset name (for logging)
    :param cfg: Config containing paths.data_dir
    :return: S3 URI or None if no match
    """
    # Replace local data_dir with S3 audio base path
    # The file_path already includes the dataset subdirectory (e.g., maxcaps/, musiccaps/)
    data_dir = cfg.get("paths", {}).get("data_dir", "/opt/ml/processing/input")
    s3_uri = file_path.replace(data_dir, S3_AUDIO_BASE).replace('.npy', '.mp3')
    log.info(f"S3 URI: {s3_uri}")
    return s3_uri


def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """
    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name,
                    'Key': object_name},
            ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None
    
    # The response contains the presigned URL
    return response


def _get_wandb_api():
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` to use the Gradio app."
        ) from e
    return wandb.Api()


def list_wandb_runs(entity: str, project: str, tags: List[str]) -> List[Tuple[str, str]]:
    """
    List WandB runs matching all given tags.

    Returns list of (display_name, wandb_path) where wandb_path is
    'wandb://entity/project/run_id'.
    """
    api = _get_wandb_api()
    runs = api.runs(f"{entity}/{project}")
    options: List[Tuple[str, str]] = []

    for run in runs:
        for tag in tags:
            if tag in run.tags:
                path = f"wandb://{entity}/{project}/{run.id}"
                options.append((run.name, path))
                
    unique_options = list(set(options))
    return unique_options


# S3 base path for checkpoints: s3://maml-aimcdt/gdr2/{id}/checkpoints
# {id} is extracted from wandb config.id e.g. "gdr2-df7af1c6-20260128154553" -> "df7af1c6"
GDR2_CKPT_BASE = "s3://maml-aimcdt/gdr2/logs"


def resolve_ckpt_dir_from_config(cfg: DictConfig) -> Optional[str]:
    """
    Resolve checkpoint directory from a fetched WandB config.

    Checkpoints are assumed at s3://maml-aimcdt/gdr2/{id}/checkpoints where {id}
    is the middle segment of config.id (e.g. config.id = "gdr2-df7af1c6-20260128154553" -> id = "df7af1c6").
    """
    try:
        config_id = cfg.get("sagemaker_job_name")
        if isinstance(config_id, str) and config_id:
            parts = config_id.split("-")
            # e.g. "gdr2-df7af1c6-20260128154553" -> id = "df7af1c6"
            id_segment = parts[1] if len(parts) >= 2 else config_id
            ckpt_dir = f"{GDR2_CKPT_BASE}/{id_segment}/checkpoints"
            log.info(f"Resolved checkpoint directory from config.sagemaker_job_name={config_id} -> {ckpt_dir}")
            return ckpt_dir
    except Exception as e:
        log.warning(f"Could not resolve ckpt dir from config.sagemaker_job_name: {e}")

    return None


def list_checkpoints(ckpt_dir: str) -> List[str]:
    """
    List available checkpoints in a directory or S3 URI.
    Returns a list of checkpoint paths (full paths).
    """

    log.info(f"Listing checkpoints in {ckpt_dir}")

    if ckpt_dir is None:
        return []

    # S3 path
    if ckpt_dir.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse

        parsed = urlparse(ckpt_dir, allow_fragments=False)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        ckpts: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".ckpt"):
                    ckpts.append(f"s3://{bucket}/{key}")
        ckpts.sort(reverse=True)
        return ckpts

    # Local / mounted path
    if not os.path.exists(ckpt_dir):
        log.warning(f"Checkpoint directory does not exist: {ckpt_dir}")
        return []

    files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt")
    ]
    files.sort()
    return files


def build_datamodule(
    cfg: DictConfig,
    cache_embeddings: bool = True,
    progress: Optional[gr.Progress] = None,
) -> Tuple[TextAudioDataModule, Optional[Dict[str, Tuple[str, str]]]]:
    """
    Build datamodule and optionally cache all dataset embeddings.
    
    :param cfg: Hydra config
    :param cache_embeddings: Whether to cache embeddings for all test datasets
    :param progress: Optional Gradio progress tracker
    :return: Tuple of (datamodule, cache_paths_dict)
    """
    if progress:
        progress(0, desc="Instantiating datamodule...")
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: TextAudioDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("eval")
    
    cache_paths = None
    if cache_embeddings:
        log.info("Caching embeddings for all test datasets...")
        cache_paths = cache_all_datasets(
            datamodule=datamodule,
            preextracted_features=cfg.get("preextracted_features", True),
            progress=progress,
        )
        log.info(f"Cached {len(cache_paths)} datasets")
    
    return datamodule, cache_paths


def prepare_state_from_run(
    base_cfg: DictConfig, run_path: str
) -> Tuple[DictConfig, List[str], str]:
    """
    Given a base Hydra cfg and a WandB run path ('wandb://entity/project/run_id'),
    fetch the full config, override `model`, and discover checkpoints.

    Returns:
        updated_cfg, checkpoint_options, ckpt_dir
    """
    log.info(f"Fetching config from {run_path}")
    fetched_config = load_config_from_source(run_path)

    # Override only the `model` key from fetched config
    updated_cfg = apply_config_overrides(
        base_cfg, fetched_config, override_keys=["model"]
    )

    logging.info(f"Updated config: {updated_cfg}")

    # Resolve checkpoint directory and list checkpoints
    ckpt_dir = resolve_ckpt_dir_from_config(fetched_config)
    log.info(f"Resolved checkpoint directory: {ckpt_dir}")
    ckpts = list_checkpoints(ckpt_dir) if ckpt_dir else []

    return updated_cfg, ckpts, ckpt_dir or ""


def _load_model_into_state(
    state: Dict[str, Any],
    cfg: DictConfig,
    ckpt_path: str,
) -> str:
    """
    Load model and checkpoint into state. Called when checkpoint dropdown changes.
    Returns a short status message for the UI.
    """
    if not ckpt_path:
        state.pop("model", None)
        state.pop("model_key", None)
        state.pop("embeddings", None)
        state.pop("file_paths", None)
        state.pop("embeddings_key", None)
        state.pop("dataset_name", None)
        return "No checkpoint selected."

    device = cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    model_key = f"{ckpt_path}"

    if state.get("model_key") == model_key:
        return f"Model already loaded for this checkpoint."

    try:
        log.info(f"Loading model <{cfg.model._target_}> for checkpoint {ckpt_path}")
        model = hydra.utils.instantiate(cfg.model)

        if "s3://" in ckpt_path:
            from s3torchconnector import S3Checkpoint

            checkpoint = S3Checkpoint(region="us-east-1")
            with checkpoint.reader(ckpt_path) as f:
                ckpt = torch.load(f, map_location=device, weights_only=False)
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["state_dict"], strict=True)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        state["model"] = model
        state["model_key"] = model_key
        state.pop("embeddings", None)
        state.pop("file_paths", None)
        state.pop("embeddings_key", None)
        state.pop("dataset_name", None)
        return f"Model loaded: {os.path.basename(ckpt_path)}"
    except Exception as e:
        log.exception("Failed to load model")
        return f"Failed to load model: {e}"


def extract_audio_embedding_from_file(
    audio_path: str,
    model,
    device: str = "cuda:0",
) -> Optional[torch.Tensor]:
    """
    Extract audio embedding from an uploaded audio file using the model's audio encoder.
    
    Loads the audio, runs it through the audio encoder, and averages across the
    temporal dimension. Returns a tensor shaped (1, 1, D) suitable for passing
    to model.inference() as audio_embedding.
    
    :param audio_path: Path to the audio file
    :param model: The model (must have .audio_encoder attribute)
    :param device: Device to run on
    :return: Audio embedding tensor of shape (1, 1, D), or None on failure
    """
    import torchaudio
    
    try:
        log.info(f"Extracting audio embedding from: {audio_path}")
        
        # Load audio
        # with soundfile
        waveform, sr = sf.read(audio_path, always_2d=True, dtype='float32')
        waveform = torch.tensor(waveform.T)
        
        # Resample to 48kHz if needed (expected by most audio encoders)
        target_sr = 48000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Move to device
        waveform = waveform.to(device)
        
        # Get audio encoder
        audio_encoder = model.audio_encoder if hasattr(model, 'audio_encoder') else model.encoder_pair
        
        # Extract features
        with torch.no_grad():
            if hasattr(audio_encoder, 'get_audio_embedding_from_data'):
                result = audio_encoder.get_audio_embedding_from_data(waveform)
                if isinstance(result, dict):
                    embedding = result.get('embedding_proj', result.get('last_hidden_state'))
                else:
                    embedding = result
            else:
                embedding = audio_encoder(waveform)
        
        log.info(f"Raw audio encoder output shape: {embedding.shape}")

        embedding = embedding.permute(1, 0).unsqueeze(0)
        
        return embedding
    

        
    except Exception as e:
        log.exception(f"Failed to extract audio embedding from {audio_path}: {e}")
        return None


def run_single_retrieval(
    cfg: DictConfig,
    state: Dict[str, Any],
    prompt: str,
    dataset_name: str,
    ckpt_path: str,
    guidance_scale: float = 1.0,
    negative_prompt: str = "",
    num_steps: int = 50,
    audio_conditioning_path: Optional[str] = None,
    audio_guidance_scale: float = 1.0,
    progress: Optional[gr.Progress] = None,
) -> List[Tuple[Optional[str], Optional[dict]]]:
    """
    Core retrieval routine. Uses the model already in state (loaded on checkpoint change).
    
    Returns list of (audio_path, label_dict) tuples for top-k results.
    label_dict is formatted for gr.Label component.
    
    :param audio_conditioning_path: Optional path to uploaded audio for similarity conditioning.
    :param audio_guidance_scale: Scale for audio conditioning guidance.
    """
    def _error_result(msg: str) -> List[Tuple[Optional[str], Optional[dict]]]:
        """Return error as first result, rest empty."""
        # Use gr.Label format with error message as label
        error_label = {msg: 0.0}
        return [(None, error_label)] + [(None, None) for _ in range(9)]

    if not prompt and not audio_conditioning_path:
        return _error_result("Please enter a prompt or upload an audio file.")

    if not ckpt_path:
        return _error_result("Please select a checkpoint.")

    model = state.get("model")
    if model is None or state.get("model_key") != f"{ckpt_path}":
        return _error_result("Please select a checkpoint and wait for it to load.")

    device = cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"

    # Build datamodule if not cached (also caches embeddings on first run)
    if "datamodule" not in state:
        if progress:
            progress(0, desc="Building datamodule and caching embeddings...")
        datamodule, cache_paths = build_datamodule(cfg, cache_embeddings=True, progress=progress)
        state["datamodule"] = datamodule
        state["embedding_cache_paths"] = cache_paths
    else:
        datamodule = state["datamodule"]

    # Resolve dataset by name from test_datasets
    if not hasattr(datamodule, "test_datasets") or not datamodule.test_datasets:
        return _error_result("No test datasets found in datamodule.")

    if hasattr(datamodule, "test_dataset_names"):
        test_names = datamodule.test_dataset_names
    else:
        # Fallback: use indices as names
        test_names = [f"dataset_{i}" for i in range(len(datamodule.test_datasets))]

    if dataset_name not in test_names:
        return _error_result(f"Unknown dataset '{dataset_name}'.")

    # Verify dataset is cached
    if not is_dataset_cached(dataset_name):
        return _error_result(f"Dataset '{dataset_name}' embeddings not cached. Please restart the app.")

    # Get dataset for annotations lookup
    dataset_idx = test_names.index(dataset_name)
    dataset = datamodule.test_datasets[dataset_idx]

    # Extract audio conditioning embedding if audio file provided
    audio_emb = None
    if audio_conditioning_path:
        audio_emb = extract_audio_embedding_from_file(
            audio_path=audio_conditioning_path,
            model=model,
            device=device,
        )
        if audio_emb is not None:
            log.info(f"Audio conditioning enabled with shape {audio_emb.shape}")
        else:
            log.warning("Audio conditioning file provided but feature extraction failed")

    # Generate query embedding
    neg_prompt = negative_prompt.strip() if negative_prompt else None
    query_embedding = generate_and_extract_embedding(
        model=model,
        prompt=prompt,
        negative_prompt=neg_prompt,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        device=device,
        num_samples_per_prompt=cfg.get("num_samples_per_prompt", 1),
        audio_embedding=audio_emb,
        audio_guidance_scale=audio_guidance_scale,
    )

    # Retrieve using chunked memmap processing
    top_k = cfg.get("top_k", 10)
    top_k_paths, top_k_values, top_k_indices = retrieve_top_k_chunked(
        query_embedding=query_embedding,
        dataset_name=dataset_name,
        k=top_k,
        chunk_size=EMBEDDING_CHUNK_SIZE,
    )

    # Clean up previous temp directory
    old_temp_dir = state.get("temp_dir")
    if old_temp_dir and os.path.exists(old_temp_dir):
        try:
            shutil.rmtree(old_temp_dir)
            log.info(f"Cleaned up temp directory: {old_temp_dir}")
        except Exception as e:
            log.warning(f"Failed to clean up temp dir {old_temp_dir}: {e}")

    # Create new temp directory for this retrieval
    temp_dir = tempfile.mkdtemp(prefix="gdr_retrieval_")
    state["temp_dir"] = temp_dir
    log.info(f"Created temp directory: {temp_dir}")

    # Download audio files from S3 and build results
    # Returns list of (audio_path, score_label) - one per top-k result
    audio_results: List[Tuple[Optional[str], str]] = []
    for rank, (file_path, score, idx) in enumerate(
        zip(top_k_paths, top_k_values, top_k_indices), start=1
    ):
        # Build S3 URI using selected dataset name and download
        s3_uri = build_s3_audio_uri(file_path, dataset_name, cfg) if file_path else None
        presigned_url = create_presigned_url(s3_uri.split("/")[2], "/".join(s3_uri.split("/")[3:])) if s3_uri else None

        # Build label dict for gr.Label component
        annot = dataset.annotations[idx] if hasattr(dataset, "annotations") else {}
        tags = annot.get("tags", None)
        label_value = format_label_value(rank, score, tags)

        audio_results.append((presigned_url, label_value))

    # Pad to 10 results if fewer
    while len(audio_results) < 10:
        audio_results.append((None, None))

    return audio_results


def build_interface(cfg: DictConfig):
    """
    Build the Gradio Blocks interface.
    """
    state: Dict[str, Any] = {}

    entity = cfg.wandb.entity
    project = cfg.wandb.project
    tags = cfg.wandb.tags or []

    # Initial run list
    initial_runs = list_wandb_runs(entity, project, tags)
    run_labels = [name for name, _ in initial_runs]
    run_values = [path for _, path in initial_runs]
    run_choices = list(zip(run_labels, run_values)) if initial_runs else []

    # Custom purple accent color (following Gradio theming guide)
    # https://www.gradio.app/guides/theming-guide
    purple = gr.themes.Color(
        c50="#faf5ff",
        c100="#f3e8ff",
        c200="#e9d5ff",
        c300="#d8b4fe",
        c400="#c084fc",
        c500="#c266ff",  # Main accent
        c600="#a855f7",
        c700="#9333ea",
        c800="#7e22ce",
        c900="#581c87",
        c950="#3b0764",
    )
    
    # Light theme with purple accent and Helvetica Neue font
    # Using .set() method to customize CSS variables as per docs
    theme = gr.themes.Soft(
        primary_hue=purple,
        secondary_hue=purple,
        neutral_hue="gray",
        radius_size="md",  # Medium border radius
        font=[
            "Helvetica Neue",
            "Helvetica", 
            "Arial",
            "ui-sans-serif",
            "sans-serif",
        ],
        font_mono=[
            "ui-monospace",
            "monospace",
        ],
    ).set(
        # Body/background - force light mode
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#fafafa",
        background_fill_secondary_dark="#fafafa",
        
        # Text colors - all black
        body_text_color="#000000",
        body_text_color_dark="#000000",
        block_label_text_color="#000000",
        block_label_text_color_dark="#000000",
        block_title_text_color="#000000",
        block_title_text_color_dark="#000000",
        
        # Block styling - no borders
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_width="0px",
        block_shadow="none",
        block_shadow_dark="none",
        block_title_text_weight="300",
        
        # Input styling - subtle, no heavy borders
        input_background_fill="#fafafa",
        input_background_fill_dark="#fafafa",
        input_border_width="1px",
        input_border_color_focus="*primary_800",
        input_border_color_focus_dark="*primary_800",
        input_shadow_focus="0 0 0 2px rgba(194, 102, 255, 0.2)",
        input_shadow_focus_dark="0 0 0 2px rgba(194, 102, 255, 0.2)",
        
        # Button styling - primary buttons: accent background with white text
        button_primary_background_fill="*primary_300",
        button_primary_background_fill_hover="*primary_500",
        button_primary_background_fill_dark="*primary_300",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="*primary_300",
        button_primary_border_color_dark="*primary_300",
        button_border_width="0px",
        button_large_radius="24px",
        
        # Secondary button styling - same approach
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_background_fill_hover="#fafafa",
        button_secondary_border_color="*primary_400",
        button_secondary_border_color_dark="*primary_400",
        button_secondary_text_color="*primary_500",
        button_secondary_text_color_dark="*primary_500",
        
        # Slider styling
        slider_color="*primary_900",
        slider_color_dark="*primary_900",
    )
    
    # Minimal custom CSS for elements that can't be styled via theme API
    custom_css = """
    /* Header styling - white background, accent text */
    .main-header {
        background: #ffffff;
        color: #000000;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    
    .main-header h1 {
        margin: 0 0 8px 0;
        font-weight: 300;
        font-size: 1.75rem;
        color: #c266ff;
    }
    
    .main-header p {
        margin: 0;
        color: #000000;
        font-size: 0.95rem;
    }
    
    /* Result row container spacing */
    .result-row-container {
        margin-bottom: 8px !important;
        align-items: center !important;
        border-radius: 6px;
    }
    
    /* Section headers - black text with accent underline */
    .section-header {
        color: #000000;
        font-weight: 300;
        font-size: 1rem;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #c266ff;
    }
    
    /* Status text */
    .status-text {
        font-size: 0.875rem;
        color: #000000;
        padding: 8px 12px;
        background-color: #fafafa;
        border-radius: 6px;
        border-left: 3px solid #c266ff;
    }
    
    /* Label component styling - white background with accent text */
    .label-class {
        background: #ffffff !important;
    }
    .label-class .output-class {
        background: #ffffff !important;
    }
    .label-class .output-class .label {
        color: #c266ff !important;
        font-weight: 600 !important;
    }
    .label-class .confidence-bar {
        background: linear-gradient(90deg, #ef4444 0%, #eab308 50%, #22c55e 100%) !important;
    }
    
    /* Gradio Label specific selectors */
    div[data-testid="label"] {
        background: #ffffff !important;
    }
    div[data-testid="label"] .label {
        color: #c266ff !important;
        font-weight: 600 !important;
    }
    div[data-testid="label"] .confidences {
        background: #ffffff !important;
    }
    div[data-testid="label"] .confidence-bar {
        background: linear-gradient(90deg, #ef4444 0%, #eab308 50%, #22c55e 100%) !important;
    }
    
    /* Primary button - larger radius, accent background */
    button.primary {
        border-radius: 24px !important;
        background: #d8b4fe !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        transition: background 0.2s ease !important;
    }
    button.primary:hover {
        background: #c266ff !important;
    }
    /* === Kill the purple "pill" label backgrounds; keep purple bold text === */
    .gr-label {
        background: none !important;
        padding: 0 !important;
        color: #c266ff !important;
        font-weight: 700 !important;
    }

    /* Some Gradio builds wrap labels like this */
    .gr-form .label, .gr-block-label, label {
        background: none !important;
    }
    
    /* Audio player buttons */
    .audio-player button {
        color: #9ca3af !important;
    }
    .audio-player button:hover,
    .audio-player.playing button {
        color: #000000 !important;
    }
    """
    
    with gr.Blocks(title="GDR Retrieval", theme=theme, css=custom_css) as demo:
        # Sidebar with model/dataset controls
        with gr.Sidebar():
            gr.HTML("""
                <div style="text-align: center; padding: 16px 0;">
                    <h2 style="margin: 0; color: #c266ff; font-weight: 600;">GDR</h2>
                    <p style="margin: 4px 0 0 0; color: #000000; font-size: 0.85rem;">Audio Retrieval</p>
                </div>
            """)
            
            gr.HTML('<div class="section-header">Model & Checkpoint</div>')
            
            run_dropdown = gr.Dropdown(
                label="WandB Run",
                choices=[c[0] for c in run_choices] if run_choices else [],
                value='---- Select a run ----',
                container=True,
            )
            run_hidden_values = {
                name: path for name, path in initial_runs
            }  # label -> wandb_path

            refresh_button = gr.Button("↻ Refresh Runs", size="sm")

            ckpt_dropdown = gr.Dropdown(
                label="Checkpoint",
                choices=[],
                value=None,
            )

            ckpt_dir_text = gr.Textbox(
                label="Checkpoint Directory",
                interactive=False,
                show_label=True,
            )
            
            model_status_text = gr.Textbox(
                label="Status",
                value="Select a checkpoint to load the model.",
                interactive=False,
                elem_classes=["status-text"],
            )

            gr.HTML('<div class="section-header" style="margin-top: 24px;">Dataset</div>')
            
            dataset_dropdown = gr.Dropdown(
                label="Target Dataset (test split)",
                choices=[],
                value=None,
            )

            # Expandable config viewers using gr.JSON for pretty display
            with gr.Accordion("📄 Model Config", open=False):
                model_config_display = gr.JSON(
                    value=None,
                    open=False,
                    show_indices=False,
                    max_height=300,
                )
            
            with gr.Accordion("📊 Data Config", open=False):
                data_config_display = gr.JSON(
                    value=None,
                    open=False,
                    show_indices=False,
                    max_height=300,
                )

        # Main content area
        gr.HTML("""
            <div class="main-header">
                <h1>🐶 GDR</h1>
                <p>Retrieve music with generative flow matching.</p>
            </div>
        """)
        
        # Retrieval controls - no group wrapper to avoid internal borders
        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="Describe the audio you're looking for...",
                scale=3,
            )
            audio_conditioning_upload = gr.Audio(
                label="Audio Conditioning (optional)",
                type="filepath",
                scale=1,
            )
        negative_prompt_box = gr.Textbox(
            label="Negative Prompt (optional)",
            lines=2,
            placeholder="What to avoid in the results...",
        )
        
        with gr.Row():
            guidance_scale_slider = gr.Slider(
                label="CFG Scale",
                minimum=1.0,
                maximum=15.0,
                value=1.0,
                step=0.5,
                info="Classifier-free guidance strength",
            )
            audio_guidance_scale_slider = gr.Slider(
                label="Audio Guidance Scale",
                minimum=0.0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                info="Audio similarity conditioning strength",
            )
            num_steps_slider = gr.Slider(
                label="Inference Steps",
                minimum=10,
                maximum=100,
                value=50,
                step=5,
                info="Number of diffusion steps",
            )
        
        run_button = gr.Button("🔍 Run Retrieval", variant="primary", size="lg")
        
        gr.HTML('<div class="section-header" style="margin-top: 32px;">Results</div>')
        
        # 10 audio components with gr.Label for confidence scores
        audio_components = []
        label_components = []
        for i in range(10):
            with gr.Row(elem_classes=["result-row-container"]):
                score_label = gr.Label(
                    value=None,
                    num_top_classes=1,
                    show_label=False,
                    container=False,
                    scale=1,
                )
                audio = gr.Audio(
                    label=None,
                    type="filepath",
                    interactive=False,
                    scale=3,
                    waveform_options={
                        # dark grey for progress
                        "waveform_progress_color": "#9ca3af",
                    },
                )
                label_components.append(score_label)
                audio_components.append(audio)

        # Helper functions for UI callbacks
        def _refresh_runs_cb():
            runs = list_wandb_runs(entity, project, tags)
            labels = [name for name, _ in runs]
            # Update mapping
            nonlocal run_hidden_values
            run_hidden_values = {name: path for name, path in runs}
            first_label = labels[0] if labels else None
            return gr.update(choices=labels, value='---- Select a run ----')

        def _on_run_change_cb(selected_label: str, progress=gr.Progress()):
            if not selected_label:
                return (
                    gr.update(choices=[]),
                    "",
                    "",
                    gr.update(choices=[]),
                    None,
                    None,
                )

            wandb_path = run_hidden_values.get(selected_label)
            if not wandb_path:
                return (
                    gr.update(choices=[]),
                    "",
                    "",
                    gr.update(choices=[]),
                    None,
                    None,
                )

            progress(0, desc="Loading run configuration...")
            
            # Update cfg with model + list checkpoints
            updated_cfg, ckpts, ckpt_dir = prepare_state_from_run(cfg, wandb_path)
            state["cfg"] = updated_cfg
            # Invalidate all cached state when run changes
            state.clear()
            state["cfg"] = updated_cfg

            progress(0.1, desc="Building datamodule and caching embeddings...")
            
            # Refresh dataset list from datamodule and cache embeddings
            datamodule, cache_paths = build_datamodule(updated_cfg, cache_embeddings=True, progress=progress)
            state["datamodule"] = datamodule
            state["embedding_cache_paths"] = cache_paths
            if hasattr(datamodule, "test_dataset_names"):
                ds_names = datamodule.test_dataset_names
            else:
                ds_names = [f"dataset_{i}" for i in range(len(datamodule.test_datasets))]

            # Extract model and data configs as dict for gr.JSON display
            model_cfg = updated_cfg.get("model", {})
            data_cfg = updated_cfg.get("data", {})
            # Convert OmegaConf to plain dict for gr.JSON
            model_dict = OmegaConf.to_container(model_cfg, resolve=True) if model_cfg else None
            data_dict = OmegaConf.to_container(data_cfg, resolve=True) if data_cfg else None

            progress(1.0, desc="✓ Ready")
            
            return (
                gr.update(choices=ckpts, value=ckpts[0] if ckpts else None),
                ckpt_dir or "",
                "Select a checkpoint to load the model.",
                gr.update(choices=ds_names, value=ds_names[0] if ds_names else None),
                model_dict,
                data_dict,
            )

        def _on_ckpt_change_cb(ckpt_path: str):
            """Load model when checkpoint selection changes. Model is stored in state."""
            current_cfg = state.get("cfg")
            if not current_cfg:
                return "Select a WandB run first."
            return _load_model_into_state(state, current_cfg, ckpt_path)

        def _on_run_button_cb(
            prompt: str,
            negative_prompt: str,
            guidance_scale: float,
            audio_guidance_scale: float,
            num_steps: int,
            dataset_name: str,
            ckpt_path: str,
            audio_conditioning_path: Optional[str] = None,
            progress=gr.Progress(),
        ):
            progress(0, desc="Running retrieval...")
            current_cfg = state.get("cfg", cfg)
            results = run_single_retrieval(
                cfg=current_cfg,
                state=state,
                prompt=prompt,
                dataset_name=dataset_name,
                ckpt_path=ckpt_path,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_steps=int(num_steps),
                audio_conditioning_path=audio_conditioning_path,
                audio_guidance_scale=audio_guidance_scale,
                progress=progress,
            )
            progress(1.0, desc="✓ Done")
            # Unpack results: interleave labels and audio paths for Gradio outputs
            # Order: label_0, audio_0, label_1, audio_1, ...
            outputs = []
            for audio_path, score_label in results:
                outputs.append(score_label)  # label textbox
                outputs.append(audio_path)   # audio component
            return outputs

        # Wire callbacks
        refresh_button.click(
            _refresh_runs_cb,
            inputs=None,
            outputs=run_dropdown,
        )

        run_dropdown.change(
            _on_run_change_cb,
            inputs=run_dropdown,
            outputs=[
                ckpt_dropdown,
                ckpt_dir_text,
                model_status_text,
                dataset_dropdown,
                model_config_display,
                data_config_display,
            ],
        )

        ckpt_dropdown.change(
            _on_ckpt_change_cb,
            inputs=ckpt_dropdown,
            outputs=model_status_text,
        )

        # Build outputs list: interleave labels and audio components
        retrieval_outputs = []
        for label, audio in zip(label_components, audio_components):
            retrieval_outputs.append(label)
            retrieval_outputs.append(audio)

        run_button.click(
            _on_run_button_cb,
            inputs=[
                prompt_box,
                negative_prompt_box,
                guidance_scale_slider,
                audio_guidance_scale_slider,
                num_steps_slider,
                dataset_dropdown,
                ckpt_dropdown,
                audio_conditioning_upload,
            ],
            outputs=retrieval_outputs,
        )

    return demo



@hydra_main(version_base="1.3", config_path="../configs", config_name="gradio.yaml")
def main(cfg: DictConfig):
    # Print loaded config for inspection in logs / stdout
    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    log.info("Loaded config:\n%s", cfg_yaml)
    print("Loaded config:\n", cfg_yaml)

    # handle A100 GPUs
    if torch.cuda.is_available() and (
        "A100" in torch.cuda.get_device_name() or "A5000" in torch.cuda.get_device_name()
    ):
        torch.set_float32_matmul_precision("high")

    # avoid annoying multiprocessing errors
    torch.multiprocessing.set_sharing_strategy("file_system")



    def tree_str(
        path=".",
        max_depth=None,
        max_files=2,
        ignore={".git", "__pycache__"}
    ):
        lines = []
        from pathlib import Path
        import os
        path = Path(path)

        def _walk(p, prefix="", level=0):
            if max_depth is not None and level > max_depth:
                return

            entries = [e for e in p.iterdir() if e.name not in ignore]

            dirs = sorted((e for e in entries if e.is_dir()), key=lambda x: x.name.lower())
            files = sorted((e for e in entries if e.is_file()), key=lambda x: x.name.lower())

            shown_files = files[:max_files]
            omitted_files = len(files) - len(shown_files)

            combined = dirs + shown_files

            for i, entry in enumerate(combined):
                is_last = i == len(combined) - 1
                connector = "└── " if is_last else "├── "
                lines.append(prefix + connector + entry.name)

                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    _walk(entry, prefix + extension, level + 1)

            if omitted_files > 0:
                lines.append(prefix + f"└── … ({omitted_files} more files)")

        _walk(path)
        return "\n".join(lines)



    # prevent annoying warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # apply extra utilities (logging, seeding, etc.)
    extras(cfg)

    # recursively show the first two files in the data diredctory as a tree
    data_dir = cfg.paths.data_dir
    tree_dir = tree_str(data_dir, max_depth=2, max_files=2)
    log.info(f"Data directory tree:\n{tree_dir}")
    print(f"Data directory tree:\n{tree_dir}")

    demo = build_interface(cfg)
    demo.launch(share=True)

    # get the public URL
    public_url = demo.launch_url
    log.info(f"Gradio app is available at: {public_url}")


if __name__ == "__main__":
    main()


