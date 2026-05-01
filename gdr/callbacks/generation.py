from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
import torch
import torch.distributed as dist
from typing import Any, Mapping, Optional

from gdr.callbacks.utils import BaseCallback
import logging

log = logging.getLogger(__name__)

class GenerationCallback(BaseCallback):
    """
    Callback that generates audio samples during validation/test and stores them
    in the Lightning module's dictionaries, organized by dataloader_idx.
    This allows other callbacks to access the generated samples and ground truth.
    """
    
    def __init__(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        disable_progress: bool = True,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_epochs: int = 1,
        every_n_steps: int = None,
        max_samples_global: Optional[int] = None,
        max_batches_global: Optional[int] = None,
        max_samples_global_by_dataloader: Optional[Mapping[Any, int]] = None,
        max_batches_global_by_dataloader: Optional[Mapping[Any, int]] = None,
    ):
        """
        Args:
            num_inference_steps: Number of diffusion steps for generation
            guidance_scale: Guidance scale for classifier-free guidance during inference
            disable_progress: Whether to disable progress bar during generation
            enable_on_validation: Whether to generate samples during validation
            enable_on_test: Whether to generate samples during test
            max_samples_global: Optional cap on gathered samples per dataloader
            max_batches_global: Optional cap on gathered batches per dataloader
            max_samples_global_by_dataloader: Optional per-dataloader sample caps
            max_batches_global_by_dataloader: Optional per-dataloader batch caps
        """
        super().__init__(every_n_epochs=every_n_epochs, every_n_steps=every_n_steps)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.disable_progress = disable_progress
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.max_samples_global = max_samples_global
        self.max_batches_global = max_batches_global
        self.max_samples_global_by_dataloader = dict(max_samples_global_by_dataloader or {})
        self.max_batches_global_by_dataloader = dict(max_batches_global_by_dataloader or {})
    
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Generate audio samples after each validation batch and store them."""
        
        if not self._should_run_on_validation(trainer, pl_module):
            return

        
        self._generate_and_store(
            pl_module=pl_module,
            batch=batch,
            dataloader_idx=dataloader_idx,
            mode='val'
        )
    
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Generate audio samples after each test batch and store them."""
        
        if not self._should_run_on_test(trainer, pl_module):
            return

        
        self._generate_and_store(
            pl_module=pl_module,
            batch=batch,
            dataloader_idx=dataloader_idx,
            mode='test'
        )
    
    def _generate_and_store(
        self,
        pl_module: LightningModule,
        batch: dict,
        dataloader_idx: int,
        mode: str = 'val',
    ):
        """
        Generate audio samples and store predictions and ground truth in the module.
        
        Args:
            pl_module: The Lightning module (should be LightningGDR)
            batch: Batch containing 'audio' and 'prompt'
            dataloader_idx: Index of the dataloader
            mode: 'val' or 'test'
        """
        audio = batch['audio']
        prompt = batch['prompt']
        
        # Extract latents from audio
        latents = audio

        # Generate audio samples using inference
        preds = pl_module.inference(
            prompt=prompt,
            inference_scheduler=pl_module.inference_scheduler,
            num_steps=self.num_inference_steps,
            disable_progress=self.disable_progress,
            guidance_scale=self.guidance_scale
        )
        
        # Process predictions: permute and take mean across temporal dimension
        preds = preds.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)
        preds = preds.mean(dim=1)  # (batch, time, channels) -> (batch, channels)
        
        # Process ground truth latents: take mean across temporal dimension
        latents = latents.mean(dim=1)  # (batch, time, channels) -> (batch, channels)
        
        # Get text embeddings
        text_dict = pl_module.text_encoder.get_text_embedding(
            prompt, use_tensor=True, return_dict=True
        )
        text_embedding = text_dict.get(
            'projected_pooler_output',
            text_dict['last_hidden_state'].mean(1)
        )
        
        # Get or initialize dictionaries for this dataloader_idx
        if mode == 'val':
            preds_dict, gt_dict = pl_module._get_or_init_dataloader_dict(
                dataloader_idx, mode='val'
            )
        elif mode == 'test':
            preds_dict, gt_dict = pl_module._get_or_init_dataloader_dict(
                dataloader_idx, mode='test'
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'val' or 'test'")
        
        # Store predictions and ground truth organized by dataloader
        preds_dict['audio'].append(preds.detach().cpu())
        preds_dict['prompt_text'].append(prompt)
        preds_dict['prompt'].append(text_embedding.detach().cpu())
        preds_dict['batch_sizes'].append(len(prompt))
        
        gt_dict['audio'].append(latents.detach().cpu())
        gt_dict['prompt'].append(text_embedding.detach().cpu())
        gt_dict['prompt_text'].append(prompt)
        gt_dict['batch_sizes'].append(len(prompt))
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Gather all predictions and ground truth, then synchronize all ranks."""
        if not self._should_run_on_validation(trainer, pl_module):
            return
        
        # Gather all tensors once for all callbacks to use
        log.info(
            f"[GenerationCallback] Rank {trainer.global_rank}/{trainer.world_size} "
            f"gathering validation predictions",
            extra={"sync_dist": True}
        )
        self._gather_all_predictions(trainer, pl_module, mode='val')
        
        # Log before barrier
        log.info(
            f"[GenerationCallback] Rank {trainer.global_rank}/{trainer.world_size} "
            f"reached validation epoch end, waiting at barrier",
            extra={"sync_dist": True}
        )
        
        # Synchronize all processes
        if trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(
                f"[GenerationCallback] Rank {trainer.global_rank}/{trainer.world_size} "
                f"passed validation barrier",
                extra={"sync_dist": True}
            )
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Gather all predictions and ground truth, then synchronize all ranks."""
        if not self._should_run_on_test(trainer, pl_module):
            return
        
        # Gather all tensors once for all callbacks to use
        log.info(
            f"[GenerationCallback] Rank {trainer.global_rank}/{trainer.world_size} "
            f"gathering test predictions",
            extra={"sync_dist": True}
        )
        self._gather_all_predictions(trainer, pl_module, mode='test')
        
        # Log before barrier
        log.info(
            f"[GenerationCallback] Rank {trainer.global_rank}/{trainer.world_size} "
            f"reached test epoch end, waiting at barrier",
            extra={"sync_dist": True}
        )
        
        # Synchronize all processes
        if trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(
                f"[GenerationCallback] Rank {trainer.global_rank}/{trainer.world_size} "
                f"passed test barrier",
                extra={"sync_dist": True}
            )
    
    def _gather_all_predictions(self, trainer: Trainer, pl_module: LightningModule, mode: str = 'val'):
        """
        Gather all predictions and ground truth from all processes and store them back
        in the module's dictionaries. This ensures all callbacks see the same gathered data.
        """
        # Import here to avoid circular dependency
        from gdr.callbacks.retrieval import gather_tensor_if_distributed
        
        if mode == 'val':
            preds_dict = pl_module.val_preds
            gt_dict = pl_module.val_gt
        elif mode == 'test':
            preds_dict = pl_module.test_preds
            gt_dict = pl_module.test_gt
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get device for gathering
        device = next(pl_module.parameters()).device if trainer.world_size > 1 else torch.device('cpu')
        
        for dataloader_idx in preds_dict.keys():
            # Skip if no predictions for this dataloader
            if len(preds_dict[dataloader_idx]['audio']) == 0:
                continue
            
            log.info(
                f"[GenerationCallback] Rank {trainer.global_rank} gathering dataloader {dataloader_idx}",
                extra={"sync_dist": True}
            )
            
            # Concatenate all batches first
            for key in ['audio', 'prompt']:
                if key in preds_dict[dataloader_idx] and len(preds_dict[dataloader_idx][key]) > 0:
                    tensor = torch.cat(preds_dict[dataloader_idx][key], dim=0)
                    log.info(
                        f"[GenerationCallback] Rank {trainer.global_rank} preds[{key}] before gather: {tensor.shape}",
                        extra={"sync_dist": True}
                    )
                    
                    # Gather across all processes
                    if trainer.world_size > 1:
                        tensor = tensor.to(device)
                        tensor = gather_tensor_if_distributed(tensor, trainer)
                        log.info(
                            f"[GenerationCallback] Rank {trainer.global_rank} preds[{key}] after gather: {tensor.shape}",
                            extra={"sync_dist": True}
                        )
                    
                    # Replace the list with the gathered tensor (as a single-element list for compatibility)
                    preds_dict[dataloader_idx][key] = [tensor.cpu()]
                
                if key in gt_dict[dataloader_idx] and len(gt_dict[dataloader_idx][key]) > 0:
                    tensor = torch.cat(gt_dict[dataloader_idx][key], dim=0)
                    log.info(
                        f"[GenerationCallback] Rank {trainer.global_rank} gt[{key}] before gather: {tensor.shape}",
                        extra={"sync_dist": True}
                    )
                    
                    # Gather across all processes
                    if trainer.world_size > 1:
                        tensor = tensor.to(device)
                        tensor = gather_tensor_if_distributed(tensor, trainer)
                        log.info(
                            f"[GenerationCallback] Rank {trainer.global_rank} gt[{key}] after gather: {tensor.shape}",
                            extra={"sync_dist": True}
                        )
                    
                    # Replace the list with the gathered tensor (as a single-element list for compatibility)
                    gt_dict[dataloader_idx][key] = [tensor.cpu()]
            
            # Handle prompt_text separately (it's a list of strings, not tensors)
            # Gather objects explicitly so prompt text and batch boundaries stay aligned
            # with the gathered tensors in multi-GPU runs.
            if 'prompt_text' in preds_dict[dataloader_idx]:
                all_texts = self._flatten_text_batches(preds_dict[dataloader_idx]['prompt_text'])
                all_texts = self._gather_objects_if_distributed(all_texts, trainer)
                preds_dict[dataloader_idx]['prompt_text'] = [all_texts]
            
            if 'prompt_text' in gt_dict[dataloader_idx]:
                all_texts = self._flatten_text_batches(gt_dict[dataloader_idx]['prompt_text'])
                all_texts = self._gather_objects_if_distributed(all_texts, trainer)
                gt_dict[dataloader_idx]['prompt_text'] = [all_texts]

            for current_dict in [preds_dict[dataloader_idx], gt_dict[dataloader_idx]]:
                batch_sizes = current_dict.get('batch_sizes', [])
                batch_sizes = self._gather_objects_if_distributed(batch_sizes, trainer)
                current_dict['batch_sizes'] = batch_sizes

            self._apply_global_cap(preds_dict[dataloader_idx], gt_dict[dataloader_idx], dataloader_idx)

    def _flatten_text_batches(self, text_batches):
        all_texts = []
        for text_batch in text_batches:
            if isinstance(text_batch, (list, tuple)):
                all_texts.extend(text_batch)
            else:
                all_texts.append(text_batch)
        return all_texts

    def _gather_objects_if_distributed(self, values, trainer: Trainer):
        if trainer.world_size <= 1 or not dist.is_initialized():
            return list(values)

        gathered_values = [None for _ in range(trainer.world_size)]
        dist.all_gather_object(gathered_values, list(values))

        flattened = []
        for rank_values in gathered_values:
            if rank_values is None:
                continue
            flattened.extend(rank_values)
        return flattened

    def _resolve_cap(self, dataloader_idx: int, per_dataloader_caps, global_cap: Optional[int]) -> Optional[int]:
        if dataloader_idx in per_dataloader_caps:
            return per_dataloader_caps[dataloader_idx]

        str_key = str(dataloader_idx)
        if str_key in per_dataloader_caps:
            return per_dataloader_caps[str_key]

        return global_cap

    def _truncate_batch_sizes_to_num_samples(self, batch_sizes, num_samples: int):
        if num_samples is None:
            return list(batch_sizes)

        remaining = max(int(num_samples), 0)
        truncated = []
        for batch_size in batch_sizes:
            if remaining <= 0:
                break
            kept = min(int(batch_size), remaining)
            truncated.append(kept)
            remaining -= kept
        return truncated

    def _apply_sample_cap_to_dict(self, data_dict, sample_cap: Optional[int], batch_cap: Optional[int]):
        if sample_cap is None and batch_cap is None:
            return

        batch_sizes = [int(size) for size in data_dict.get('batch_sizes', [])]

        if batch_cap is not None:
            batch_cap = max(int(batch_cap), 0)
            batch_sizes_from_batch_cap = batch_sizes[:batch_cap]
            batch_sample_cap = sum(batch_sizes_from_batch_cap)
        else:
            batch_sizes_from_batch_cap = batch_sizes
            batch_sample_cap = None

        if sample_cap is not None:
            sample_cap = max(int(sample_cap), 0)

        effective_sample_cap = sample_cap
        if batch_sample_cap is not None:
            effective_sample_cap = batch_sample_cap if effective_sample_cap is None else min(effective_sample_cap, batch_sample_cap)

        if effective_sample_cap is None:
            return

        available_lengths = []
        if 'audio' in data_dict and len(data_dict['audio']) > 0 and isinstance(data_dict['audio'][0], torch.Tensor):
            available_lengths.append(data_dict['audio'][0].shape[0])
        if 'prompt' in data_dict and len(data_dict['prompt']) > 0 and isinstance(data_dict['prompt'][0], torch.Tensor):
            available_lengths.append(data_dict['prompt'][0].shape[0])
        if 'prompt_text' in data_dict and len(data_dict['prompt_text']) > 0:
            available_lengths.append(len(data_dict['prompt_text'][0]))
        if available_lengths:
            effective_sample_cap = min(effective_sample_cap, min(available_lengths))

        if 'audio' in data_dict and len(data_dict['audio']) > 0 and isinstance(data_dict['audio'][0], torch.Tensor):
            data_dict['audio'] = [data_dict['audio'][0][:effective_sample_cap]]
        if 'prompt' in data_dict and len(data_dict['prompt']) > 0 and isinstance(data_dict['prompt'][0], torch.Tensor):
            data_dict['prompt'] = [data_dict['prompt'][0][:effective_sample_cap]]
        if 'prompt_text' in data_dict and len(data_dict['prompt_text']) > 0:
            data_dict['prompt_text'] = [data_dict['prompt_text'][0][:effective_sample_cap]]

        truncated_batch_sizes = self._truncate_batch_sizes_to_num_samples(batch_sizes_from_batch_cap, effective_sample_cap)
        data_dict['batch_sizes'] = truncated_batch_sizes

    def _apply_global_cap(self, preds_dict, gt_dict, dataloader_idx: int):
        sample_cap = self._resolve_cap(
            dataloader_idx,
            self.max_samples_global_by_dataloader,
            self.max_samples_global,
        )
        batch_cap = self._resolve_cap(
            dataloader_idx,
            self.max_batches_global_by_dataloader,
            self.max_batches_global,
        )

        self._apply_sample_cap_to_dict(preds_dict, sample_cap, batch_cap)
        self._apply_sample_cap_to_dict(gt_dict, sample_cap, batch_cap)


    
