from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
import torch
import torch.distributed as dist

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
    ):
        """
        Args:
            num_inference_steps: Number of diffusion steps for generation
            guidance_scale: Guidance scale for classifier-free guidance during inference
            disable_progress: Whether to disable progress bar during generation
            enable_on_validation: Whether to generate samples during validation
            enable_on_test: Whether to generate samples during test
        """
        super().__init__(every_n_epochs=every_n_epochs, every_n_steps=every_n_steps)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.disable_progress = disable_progress
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
    
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
        
        gt_dict['audio'].append(latents.detach().cpu())
        gt_dict['prompt'].append(text_embedding.detach().cpu())
        gt_dict['prompt_text'].append(prompt)
    
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
            # We don't need to gather text, just keep the first rank's data
            if 'prompt_text' in preds_dict[dataloader_idx]:
                all_texts = []
                for text_batch in preds_dict[dataloader_idx]['prompt_text']:
                    if isinstance(text_batch, (list, tuple)):
                        all_texts.extend(text_batch)
                    else:
                        all_texts.append(text_batch)
                preds_dict[dataloader_idx]['prompt_text'] = [all_texts]
            
            if 'prompt_text' in gt_dict[dataloader_idx]:
                all_texts = []
                for text_batch in gt_dict[dataloader_idx]['prompt_text']:
                    if isinstance(text_batch, (list, tuple)):
                        all_texts.extend(text_batch)
                    else:
                        all_texts.append(text_batch)
                gt_dict[dataloader_idx]['prompt_text'] = [all_texts]


    