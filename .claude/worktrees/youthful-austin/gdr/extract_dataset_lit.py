import os
from typing import Any, Dict, Optional
import numpy as np

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
from lightning.pytorch.callbacks import Callback
import rootutils
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from litdata import StreamingRawDataset
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from gdr import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from dora import hydra_main

from gdr.utils import (
    RankedLogger,
    extras,
    register_resolvers,
)
from gdr.dataloading.loading_utils import load_full_and_split, load_audio_chunk

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


class FeatureExtractionModule(LightningModule):
    """Lightning module for feature extraction."""
    
    def __init__(self, 
                 audio_encoder,
                 extract_method='extract_features',
                 out_key='embedding',
                 hop=None,
                 return_full_audio=True,
                 target_n_samples=96000,
                 target_sr=48000,
                 save_dir=None,
                 root_path=None):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.extract_method = extract_method
        self.out_key = out_key
        self.hop = hop
        self.return_full_audio = return_full_audio
        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.save_dir = save_dir
        self.root_path = root_path
        
        # Set model to eval mode
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Process a batch for feature extraction."""
        features_list = []
        file_paths = []
        
        for i in range(len(batch['file_path'])):
            features = batch['features'][i]
            file_path = batch['file_path'][i]
            
            # Save features if save_dir is provided
            if self.save_dir:
                self._save_features(features, file_path)
            
            features_list.append(features)
            file_paths.append(file_path)
        
        return {
            'features': features_list,
            'file_paths': file_paths
        }
    
    def _save_features(self, features, file_path):
        """Save features to S3 or local filesystem."""
        if self.root_path is not None:
            file_path = file_path.replace(self.root_path + '/', '')
        
        save_path = os.path.join(self.save_dir, file_path) if not self.save_dir.startswith('s3://') else f"{self.save_dir}/{file_path}"
        
        if self.save_dir.startswith('s3://'):
            import s3fs
            import io
            fs = s3fs.S3FileSystem()
            buffer = io.BytesIO()
            np.save(buffer, features.numpy())
            buffer.seek(0)
            with fs.open(save_path, 'wb') as f:
                f.write(buffer.read())
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                os.remove(save_path)
            np.save(save_path, features.numpy())


class AudioFeatureExtractionDataset(StreamingRawDataset):
    """StreamingRawDataset for extracting audio features, streaming raw audio files directly from S3 or local directory."""
    
    def __init__(self, 
                 input_dir,  # S3 path (s3://bucket/path) or local directory containing audio files
                 model,
                 extract_method='extract_features',
                 out_key='embedding',
                 hop=None,
                 return_full_audio=True,
                 target_n_samples=96000,
                 target_sr=48000,
                 device='cuda:0',
                 save_dir=None,
                 root_path=None,
                 cache_dir=None,
                 storage_options=None,
                 cache_files=False,
                 recompute_index=False,
                 **kwargs):
        """
        Args:
            input_dir: Directory path (S3 or local) containing raw audio files to process
            model: The audio encoder model
            extract_method: Method name to call on model
            out_key: Key to extract from model output
            hop: Hop size for audio processing
            return_full_audio: Whether to return full audio
            target_n_samples: Target number of audio samples
            target_sr: Target sample rate
            device: Device to run model on
            save_dir: Directory to save extracted features
            root_path: Root path to remove from file paths when saving
            cache_dir: Optional cache directory for litdata
            storage_options: Optional storage options for S3 (e.g., credentials)
            cache_files: Whether to cache files locally
            recompute_index: Whether to recompute the index
        """
        self.model = model
        self.extract_method = extract_method
        self.out_key = out_key
        self.hop = hop
        self.return_full_audio = return_full_audio
        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.device = device
        self.save_dir = save_dir
        self.root_path = root_path
        
        # Handle DDP-wrapped models
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.model = model
        else:
            # Move model to device and set to eval
            self.model = model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Initialize StreamingRawDataset with the input directory
        # This will stream all raw audio files from the directory
        super().__init__(
            input_dir=input_dir,
            cache_dir=cache_dir,
            storage_options=storage_options,
            cache_files=cache_files,
            recompute_index=recompute_index,
            **kwargs
        )
    
    def __getitem__(self, index):
        """Extract features for a single item, processing raw audio bytes from S3 or local."""
        # Get raw bytes from StreamingRawDataset
        # StreamingRawDataset returns raw bytes for the file at this index
        raw_data = super().__getitem__(index)
        
        # Try to get file info from StreamingRawDataset
        # StreamingRawDataset stores file metadata in self.files
        original_file_path = None
        file_ext = '.wav'  # Default extension
        
        try:
            # Check if StreamingRawDataset has files attribute with metadata
            if hasattr(self, 'files') and index < len(self.files):
                file_metadata = self.files[index]
                # FileMetadata might have path or name attribute
                if hasattr(file_metadata, 'path'):
                    original_file_path = file_metadata.path
                elif hasattr(file_metadata, 'name'):
                    original_file_path = file_metadata.name
                elif hasattr(file_metadata, 'file_path'):
                    original_file_path = file_metadata.file_path
                elif isinstance(file_metadata, str):
                    original_file_path = file_metadata
        except Exception as e:
            log.debug(f"Could not get file metadata for index {index}: {e}")
        
        # Generate output file path
        import os
        if original_file_path:
            # Use original file path to determine output path
            base_name = os.path.basename(original_file_path)
            output_file_path = base_name.replace('.mp3', '.npy').replace('.wav', '.npy').replace('.flac', '.npy').replace('.m4a', '.npy')
            # Determine file extension from original path
            file_ext = os.path.splitext(original_file_path)[1].lower() or '.wav'
        else:
            # Fallback: use index-based naming
            output_file_path = f"audio_{index}.npy"
        
        try:
            # Process raw bytes to load audio
            # raw_data is bytes, we need to write to temp file or use in-memory loading
            import tempfile
            import os
            
            # Create temporary file with correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(raw_data)
                tmp_path = tmp_file.name
            
            try:
                # Load audio from temporary file
                if self.return_full_audio:
                    audio = load_full_and_split(
                        tmp_path,
                        self.target_sr,
                        self.target_n_samples,
                        hop=self.hop,
                        verbose=False
                    )
                    audio = audio.mean(1, keepdim=True)
                else:
                    audio = load_audio_chunk(
                        tmp_path,
                        target_sr=self.target_sr,
                        target_n_samples=self.target_n_samples,
                        verbose=False
                    )
                    audio = audio.mean(0, keepdim=True)
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            # Move to device
            audio = audio.squeeze(1).to(self.device)
            
            # Get the actual model (unwrap DDP if needed)
            model_to_use = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            
            # Process in chunks if audio is too long
            if audio.shape[0] > 200:
                chunks = torch.split(audio, 200, dim=0)
                chunks = list(chunks)
                audio_features = []
                for chunk in chunks:
                    with torch.no_grad():
                        feat = getattr(model_to_use, self.extract_method)(chunk)[self.out_key]
                        audio_features.append(feat)
                audio_features = torch.cat(audio_features, dim=0)
            else:
                with torch.no_grad():
                    audio_features = getattr(model_to_use, self.extract_method)(audio)[self.out_key]
            
            # Save features if save_dir is provided
            if self.save_dir:
                self._save_features(audio_features.detach().cpu(), output_file_path)
            
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            return {
                'features': audio_features.detach().cpu(),
                'file_path': output_file_path,
                'index': index
            }
        except Exception as e:
            log.error(f"Error processing item {index}: {e}")
            raise e
    
    def _save_features(self, features, file_path):
        """Save features to S3 or local filesystem."""
        if self.root_path is not None:
            file_path = file_path.replace(self.root_path + '/', '')
        
        save_path = os.path.join(self.save_dir, file_path) if not self.save_dir.startswith('s3://') else f"{self.save_dir}/{file_path}"
        
        if self.save_dir.startswith('s3://'):
            import s3fs
            import io
            fs = s3fs.S3FileSystem()
            buffer = io.BytesIO()
            np.save(buffer, features.numpy())
            buffer.seek(0)
            with fs.open(save_path, 'wb') as f:
                f.write(buffer.read())
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                os.remove(save_path)
            np.save(save_path, features.numpy())


def extract_features(cfg: DictConfig) -> Dict[str, Any]:
    """Extracts features from audio datasets using pre-trained encoders with litdata streaming.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A dict with extraction metadata.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # import rich printing and print the config as a tree
    from gdr.utils.rich_utils import print_config_tree
    print_config_tree(cfg)

    # Get extract parameters from config
    input_dir = cfg.get("input_dir")  # Directory containing raw audio files (S3 or local)
    save_dir = cfg.get("save_dir")
    root_path = cfg.get("root_path")
    extract_method = cfg.get("extract_method", "get_audio_embedding_from_data")
    out_key = cfg.get("out_key", "embedding_proj")
    hop = cfg.get("hop", 48000)
    limit_n = cfg.get("limit_n")
    save = cfg.get("save", False)
    batch_size = cfg.get("batch_size", 1)
    num_workers = cfg.get("num_workers", os.cpu_count())
    state_file = cfg.get("state_file", "dataloader_state.pt")
    target_n_samples = cfg.get("target_n_samples", 96000)
    target_sr = cfg.get("target_sr", 48000)
    return_full_audio = cfg.get("return_full_audio", True)
    cache_dir = cfg.get("cache_dir")
    storage_options = cfg.get("storage_options")
    cache_files = cfg.get("cache_files", False)
    recompute_index = cfg.get("recompute_index", False)

    # Validate input_dir
    if input_dir is None:
        raise ValueError("input_dir must be specified in config (S3 path like s3://bucket/path or local directory)")

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # Get audio encoder
    audio_encoder = model.audio_encoder if hasattr(model, 'audio_encoder') else model

    # Handle save_dir - default based on input_dir if not specified
    if save_dir is None:
        if input_dir.startswith('s3://'):
            # For S3, create output path in same bucket
            save_dir = input_dir.rstrip('/') + '_extracted_features'
        else:
            # For local, create subdirectory
            save_dir = os.path.join(input_dir, 'extracted_features')

    log.info(f"Extracting features with {extract_method} method")
    log.info(f"Saving to: {save_dir}")
    log.info(f"out_key: {out_key}, hop: {hop}, limit_n: {limit_n}, save: {save}")

    # Save config to save_dir
    use_s3 = 's3://' in save_dir
    if use_s3:
        try:
            import s3fs
            fs = s3fs.S3FileSystem()
            config_path = f"{save_dir}/config.yaml"
            with fs.open(config_path, "w") as config_file:
                config_file.write(OmegaConf.to_yaml(cfg, resolve=True))
            log.info(f"Uploaded config to {config_path}")
        except Exception as e:
            log.warning(f"Failed to upload config to S3: {e}")
    else:
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as config_file:
            config_file.write(OmegaConf.to_yaml(cfg, resolve=True))
        log.info(f"Saved config to {config_path}")

    # Setup Lightning Trainer for distributed inference
    # Get trainer config or use defaults
    trainer_cfg = cfg.get("trainer", {})
    
    # Determine number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    
    # Create trainer
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Create streaming dataset directly from input directory
    log.info(f"Streaming raw audio files from: {input_dir}")
    
    # Create streaming dataset
    streaming_dataset = AudioFeatureExtractionDataset(
        input_dir=input_dir,
        model=audio_encoder,
        extract_method=extract_method,
        out_key=out_key,
        hop=hop,
        return_full_audio=return_full_audio,
        target_n_samples=target_n_samples,
        target_sr=target_sr,
        device="cuda" if torch.cuda.is_available() else "cpu",  # Device will be handled by Lightning
        save_dir=None,  # Don't save in dataset, let Lightning module handle it
        root_path=root_path,
        cache_dir=cache_dir,
        storage_options=storage_options,
        cache_files=cache_files,
        recompute_index=recompute_index
    )
    
    # Create dataloader (using standard DataLoader with StreamingRawDataset)
    # Use persistent_workers to avoid multiprocessing import issues
    dataloader = DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,  # Don't shuffle for feature extraction
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False  # Speed up worker initialization
    )
    
    # Restore dataloader state if it exists
    state_file_path = f"{state_file}.pt"
    if os.path.isfile(state_file_path):
        log.info(f"Loading dataloader state from {state_file_path}")
        state_dict = torch.load(state_file_path, map_location='cpu')
        dataloader.load_state_dict(state_dict)
        log.info(f"Resumed from saved state")
    
    # Create Lightning module for feature extraction
    pl_module = FeatureExtractionModule(
        audio_encoder=audio_encoder,
        extract_method=extract_method,
        out_key=out_key,
        hop=hop,
        return_full_audio=return_full_audio,
        target_n_samples=target_n_samples,
        target_sr=target_sr,
        save_dir=save_dir if save else None,
        root_path=root_path
    )
    
    dataset_size = len(streaming_dataset)
    if limit_n:
        dataset_size = min(dataset_size, limit_n)
        log.info(f"Limiting to {limit_n} items (dataset has {len(streaming_dataset)} items)")
    
    log.info(f"Processing {dataset_size} audio files from {input_dir}")
    
    # Use Lightning's predict for distributed inference
    # This automatically handles multi-GPU distribution
    results = trainer.predict(pl_module, dataloaders=dataloader)
    
    # Limit results if needed (after processing)
    if limit_n:
        limited_results = []
        count = 0
        for r in results:
            if r is None:
                continue
            batch_size_actual = len(r.get('file_paths', []))
            if count + batch_size_actual <= limit_n:
                limited_results.append(r)
                count += batch_size_actual
            else:
                # Take only what we need from this batch
                needed = limit_n - count
                if needed > 0:
                    limited_result = {
                        'features': r['features'][:needed],
                        'file_paths': r['file_paths'][:needed]
                    }
                    limited_results.append(limited_result)
                break
        results = limited_results
    
    # Count processed items
    processed_count = sum(len(r['file_paths']) for r in results if r is not None)
    log.info(f"Completed extraction: {processed_count} items processed")
    
    # Save final state
    state_dict = dataloader.state_dict()
    torch.save(state_dict, state_file_path)
    
    # Clean up
    del streaming_dataset
    del dataloader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log.info("Feature extraction completed!")
    
    return {
        "save_dir": save_dir,
        "extract_method": extract_method,
        "out_key": out_key,
    }


@hydra_main(version_base="1.3", config_path="../configs", config_name="extract_feature_lit_local.yaml")
def main(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Main entry point for feature extraction with litdata streaming.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Dict with extraction metadata.
    """
    # handle A100 GPUs
    if torch.cuda.is_available() and ("A100" in torch.cuda.get_device_name() or "A5000" in torch.cuda.get_device_name()):
        torch.set_float32_matmul_precision("high")

    # avoid annoying multiprocessing errors
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Set multiprocessing start method early to avoid issues with worker processes
    # This is especially important on macOS where 'spawn' is the default
    try:
        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    # prevent annoying warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # extract features
    result = extract_features(cfg)

    return result


if __name__ == "__main__":
    main()
