from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
import torch
import torch.nn.functional as F
import logging

from gdr.callbacks.utils import BaseCallback

log = logging.getLogger(__name__)


def gather_tensor_if_distributed(tensor: torch.Tensor, trainer: Trainer) -> torch.Tensor:
    """
    Gather tensor from all processes if distributed training is enabled.
    Uses PyTorch Lightning's strategy for distributed operations.
    
    Args:
        tensor: Tensor to gather (on CPU or GPU)
        trainer: PyTorch Lightning trainer
        
    Returns:
        Gathered tensor concatenated across all processes, or original tensor if not distributed
    """
    # Check if distributed training is enabled
    if trainer.world_size <= 1:
        return tensor
    
    # Use PyTorch Lightning's strategy for all_gather
    # The strategy handles different backends (DDP, DeepSpeed, FSDP, etc.)
    try:
        # Try using the strategy's all_gather method
        # This is the recommended way in PyTorch Lightning 2.0+
        if hasattr(trainer.strategy, 'all_gather'):
            # all_gather may return a list or a tensor depending on the strategy
            gathered = trainer.strategy.all_gather(tensor, sync_grads=False)
            if isinstance(gathered, (list, tuple)):
                gathered = torch.cat(gathered, dim=0)
            elif isinstance(gathered, torch.Tensor):
                # If all_gather returns a tensor, it might be (world_size, batch_size, ...)
                # We need to reshape it to (world_size * batch_size, ...)
                if gathered.dim() > tensor.dim():
                    # New dimension was added (typically at dim=0)
                    # Reshape to concatenate along the batch dimension
                    gathered = gathered.view(-1, *gathered.shape[2:])
            return gathered
    except (AttributeError, NotImplementedError):
        pass
    
    # Fallback: use torch.distributed directly
    # This handles variable-sized tensors by padding
    import torch.distributed as dist
    if not dist.is_initialized():
        return tensor
    
    device = tensor.device
    
    # Get the size of the tensor on each process
    local_size = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(trainer.world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]
    max_size = max(sizes)
    
    # Pad tensors to the same size if needed (for all_gather)
    if tensor.shape[0] < max_size:
        # Pad with zeros along the first dimension
        padding_shape = list(tensor.shape)
        padding_shape[0] = max_size - tensor.shape[0]
        padding = torch.zeros(padding_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)
    
    # Gather tensors from all processes (now all have the same shape)
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(trainer.world_size)]
    dist.all_gather(gathered_tensors, tensor)
    
    # Remove padding and concatenate
    gathered_list = []
    for i, gathered_tensor in enumerate(gathered_tensors):
        gathered_list.append(gathered_tensor[:sizes[i]])
    
    gathered = torch.cat(gathered_list, dim=0)
    
    return gathered


def compute_recall(similarities: torch.Tensor):
    """
    Compute recall metrics from similarity matrix.
    
    Args:
        similarities: (num_src_embeddings, num_tgt_embeddings) similarity matrix
        
    Returns:
        recalls: dict mapping threshold -> recall value
        precisions: dict mapping threshold -> precision value
        ranks: tensor of ranks for each query
        normalized_ranks: normalized ranks
    """
    num_src_embeddings, num_tgt_embeddings = similarities.size()
    device = similarities.device

    true_indices = torch.arange(num_src_embeddings, device=device).unsqueeze(1)
    sorted_indices = similarities.argsort(descending=True)

    if num_src_embeddings < num_tgt_embeddings:
        tgt_per_src, r = divmod(num_tgt_embeddings, num_src_embeddings)
        assert r == 0
        sorted_indices = sorted_indices.div_(tgt_per_src, rounding_mode="floor")
    else:
        src_per_tgt, r = divmod(num_src_embeddings, num_tgt_embeddings)
        assert r == 0
        true_indices.div_(src_per_tgt, rounding_mode="floor")

    ranks = (sorted_indices == true_indices).long().argmax(dim=1)

    recalls = torch.zeros(num_tgt_embeddings + 1, dtype=torch.long, device=device)
    precisions = torch.zeros(num_tgt_embeddings + 1, dtype=torch.long, device=device)
    values, counts = torch.unique(ranks, return_counts=True)
    recalls[values + 1] = counts
    precisions[values + 1] = counts.cumsum(dim=0)
    recalls = recalls.cumsum(dim=0).float().div_(num_src_embeddings)
    precisions = precisions.cumsum(dim=0).float().div_(num_src_embeddings)
    
    if ranks.numel() > 0 and ranks.max() > 0:
        normalized_ranks = ranks.float().div_(ranks.max())
    else:
        normalized_ranks = ranks.float()
    
    # Convert to dict format for easy access
    recalls_dict = {k: recalls[k].item() if k < len(recalls) else 0.0 for k in [1, 5, 10]}
    precisions_dict = {k: precisions[k].item() if k < len(precisions) else 0.0 for k in [1, 5, 10]}
    
    return recalls_dict, precisions_dict, ranks, normalized_ranks


class TeacherRetrieval(BaseCallback):
    """
    Callback that computes retrieval metrics (R@1, R@5, R@10) using ground truth
    audio and text embeddings. Computes both audio-to-text and text-to-audio retrieval.
    """
    
    def __init__(
        self,
        thresholds: list = [1, 5, 10],
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        text_encoder = None,
    ):
        """
        Args:
            thresholds: List of k values for R@k metrics
            enable_on_validation: Whether to compute metrics during validation
            enable_on_test: Whether to compute metrics during test
            every_n_steps: Compute metrics every N steps (None to disable step-based checking)
            every_n_epochs: Compute metrics every N epochs (default: 1, i.e., every epoch)
            text_encoder: Optional text encoder to use. If None, uses pl_module.text_encoder
        """
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.thresholds = thresholds
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.text_encoder = text_encoder
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute retrieval metrics at the end of validation epoch."""
        if not self.enable_on_validation:
            return
        
        # Check if we should compute metrics based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_retrieval_metrics(trainer, pl_module, mode='val')
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute retrieval metrics at the end of test epoch."""
        if not self.enable_on_test:
            return
        
        self._compute_retrieval_metrics(trainer, pl_module, mode='test')
    
    def _compute_retrieval_metrics(self, trainer: Trainer, pl_module: LightningModule, mode: str = 'val'):
        """Compute retrieval metrics for all dataloaders."""
        if mode == 'val':
            preds_dict = pl_module.val_preds
            gt_dict = pl_module.val_gt
        elif mode == 'test':
            preds_dict = pl_module.test_preds
            gt_dict = pl_module.test_gt
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get dataloader names if available
        dataloader_names = getattr(trainer.datamodule, 'dataloader_names', {})
        
        for dataloader_idx in preds_dict.keys():
            if len(preds_dict[dataloader_idx]['audio']) == 0:
                continue
            
            # Concatenate all batches (if already gathered by GenerationCallback, this is a single tensor)
            gt_audio = torch.cat(gt_dict[dataloader_idx]['audio'], dim=0)
            
            # Prefer stored text embeddings because GenerationCallback gathers them
            # across ranks. Recomputing from prompt_text would only see local-rank
            # strings in DDP and can mismatch the gathered audio pool.
            if len(gt_dict[dataloader_idx]['prompt']) > 0 and isinstance(gt_dict[dataloader_idx]['prompt'][0], torch.Tensor):
                gt_text = torch.cat(gt_dict[dataloader_idx]['prompt'], dim=0)
            else:
                prompt_texts = []
                for prompt_list in gt_dict[dataloader_idx].get('prompt_text', []):
                    if isinstance(prompt_list, list):
                        prompt_texts.extend(prompt_list)
                    else:
                        prompt_texts.append(prompt_list)
                
                text_encoder = self.text_encoder if self.text_encoder is not None else pl_module.text_encoder
                logging.info(f'using text encoder: {text_encoder}')
                with torch.no_grad():
                    text_dict = text_encoder.get_text_embedding(
                        prompt_texts, use_tensor=True, return_dict=True
                    )
                    gt_text = text_dict.get(
                        'projected_pooler_output',
                        text_dict['last_hidden_state'].mean(1)
                    )
                    logging.info(f'gt_text shape: {gt_text.shape}')
                gt_text = gt_text.cpu()  # Ensure on CPU to match other tensors
        
            # Check if gathering is needed (GenerationCallback may have already gathered audio)
            needs_gather = (trainer.world_size > 1 and 
                          len(gt_dict[dataloader_idx]['audio']) > 1)
            
            if needs_gather:
                log.info(f"[TeacherRetrieval] Rank {trainer.global_rank} - tensors not pre-gathered, gathering now", extra={"sync_dist": True})
                device = next(pl_module.parameters()).device
                gt_audio = gt_audio.to(device)
                gt_text = gt_text.to(device)
                
                # Gather from all processes
                gt_audio = gather_tensor_if_distributed(gt_audio, trainer)
                gt_text = gather_tensor_if_distributed(gt_text, trainer)
            else:
                log.info(f"[TeacherRetrieval] Rank {trainer.global_rank} - using pre-gathered tensors for audio", extra={"sync_dist": True})
                # Still need to move to same device
                device = gt_audio.device
                gt_text = gt_text.to(device)
            
            # Normalize embeddings
            gt_audio = gt_audio / gt_audio.norm(dim=1, keepdim=True)
            gt_text = gt_text / gt_text.norm(dim=1, keepdim=True)

            
            
            # Compute similarity matrices
            # Audio-to-Text: for each audio, find matching text
            a2t_sim = gt_audio @ gt_text.t() if gt_audio.shape[-1] == gt_text.shape[-1] else torch.zeros(
                (gt_audio.shape[0], gt_text.shape[0]), device=gt_audio.device
            )
            
            # Text-to-Audio: for each text, find matching audio
            t2a_sim = gt_text @ gt_audio.t() if gt_text.shape[-1] == gt_audio.shape[-1] else torch.zeros(
                (gt_text.shape[0], gt_audio.shape[0]), device=gt_text.device
            )
            
            # Compute retrieval metrics
            a2t_recall, a2t_precision, a2t_ranks, a2t_normalized_ranks = compute_recall(a2t_sim)
            t2a_recall, t2a_precision, t2a_ranks, t2a_normalized_ranks = compute_recall(t2a_sim)
            
            # Get dataloader name
            dataloader_name = dataloader_names.get(dataloader_idx, f'dataloader_{dataloader_idx}')
            
            # Log metrics (only on rank 0 to avoid duplicate logging)
            if trainer.is_global_zero:
                for threshold in self.thresholds:
                    if threshold in a2t_recall:
                        pl_module.log(
                            f'{dataloader_name}_TeacherRetrieval/{mode}/A->T_R@{threshold}',
                            a2t_recall[threshold],
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False
                        )
                    if threshold in t2a_recall:
                        pl_module.log(
                            f'{dataloader_name}_TeacherRetrieval/{mode}/T->A_R@{threshold}',
                            t2a_recall[threshold],
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False
                        )
                
                # Log median ranks
                pl_module.log(
                    f'{dataloader_name}_TeacherRetrieval/{mode}/A->T_median_rank',
                    a2t_normalized_ranks.median().item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )
                pl_module.log(
                    f'{dataloader_name}_TeacherRetrieval/{mode}/T->A_median_rank',
                    t2a_normalized_ranks.median().item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )


class TeacherScore(BaseCallback):
    """
    Callback that computes cosine similarity scores using ground truth
    audio and text embeddings. Computes both audio-to-text and text-to-audio similarities.
    """
    
    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        text_encoder = None,
    ):
        """
        Args:
            enable_on_validation: Whether to compute scores during validation
            enable_on_test: Whether to compute scores during test
            every_n_steps: Compute scores every N steps (None to disable step-based checking)
            every_n_epochs: Compute scores every N epochs (default: 1, i.e., every epoch)
            text_encoder: Optional text encoder to use. If None, uses pl_module.text_encoder
        """
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.text_encoder = text_encoder
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute cosine similarity scores at the end of validation epoch."""
        if not self.enable_on_validation:
            return
        
        # Check if we should compute scores based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_scores(trainer, pl_module, mode='val')
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute cosine similarity scores at the end of test epoch."""
        if not self.enable_on_test:
            return
        
        # Check if we should compute scores based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_scores(trainer, pl_module, mode='test')
    
    def _compute_scores(self, trainer: Trainer, pl_module: LightningModule, mode: str = 'val'):
        """Compute cosine similarity scores for all dataloaders."""
        if mode == 'val':
            preds_dict = pl_module.val_preds
            gt_dict = pl_module.val_gt
        elif mode == 'test':
            preds_dict = pl_module.test_preds
            gt_dict = pl_module.test_gt
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get dataloader names if available
        dataloader_names = getattr(trainer.datamodule, 'dataloader_names', {})
        
        for dataloader_idx in preds_dict.keys():
            if len(preds_dict[dataloader_idx]['audio']) == 0:
                continue
            
            gt_audio = torch.cat(gt_dict[dataloader_idx]['audio'], dim=0)
            
            # Get text embeddings - use stored embeddings if available, otherwise compute from text
            if len(gt_dict[dataloader_idx]['prompt']) > 0 and isinstance(gt_dict[dataloader_idx]['prompt'][0], torch.Tensor):
                # Already computed embeddings
                gt_text = torch.cat(gt_dict[dataloader_idx]['prompt'], dim=0)
            else:
                # Need to compute embeddings from text strings using the model's text encoder
                prompt_texts = []
                for prompt_list in gt_dict[dataloader_idx].get('prompt_text', []):
                    if isinstance(prompt_list, list):
                        prompt_texts.extend(prompt_list)
                    else:
                        prompt_texts.append(prompt_list)
                
                # Get text embeddings using the model's text encoder
                text_encoder = self.text_encoder if self.text_encoder is not None else pl_module.text_encoder
                with torch.no_grad():
                    text_dict = text_encoder.get_text_embedding(
                        prompt_texts, use_tensor=True, return_dict=True
                    )
                    gt_text = text_dict.get(
                        'projected_pooler_output',
                        text_dict['last_hidden_state'].mean(1)
                    )
                gt_text = gt_text.cpu()  # Ensure on CPU to match other tensors
            
            # Check if gathering is needed (GenerationCallback may have already gathered audio)
            needs_gather = (trainer.world_size > 1 and 
                          len(gt_dict[dataloader_idx]['audio']) > 1)
            
            if needs_gather:
                log.info(f"[TeacherScore] Rank {trainer.global_rank} - tensors not pre-gathered, gathering now", extra={"sync_dist": True})
                device = next(pl_module.parameters()).device
                gt_audio = gt_audio.to(device)
                gt_text = gt_text.to(device)
                
                # Gather from all processes
                gt_audio = gather_tensor_if_distributed(gt_audio, trainer)
                gt_text = gather_tensor_if_distributed(gt_text, trainer)
            else:
                log.info(f"[TeacherScore] Rank {trainer.global_rank} - using pre-gathered tensors for audio", extra={"sync_dist": True})
                # Still need to move to same device
                device = gt_audio.device
                gt_text = gt_text.to(device)
            
            # Normalize embeddings
            gt_audio = gt_audio / gt_audio.norm(dim=1, keepdim=True)
            gt_text = gt_text / gt_text.norm(dim=1, keepdim=True)
            
            # Compute similarity matrices
            # Audio-to-Text: for each audio, find matching text
            a2t_sim = gt_audio @ gt_text.t() if gt_audio.shape[-1] == gt_text.shape[-1] else torch.zeros(
                (gt_audio.shape[0], gt_text.shape[0]), device=gt_audio.device
            )
            
            # Text-to-Audio: for each text, find matching audio
            t2a_sim = gt_text @ gt_audio.t() if gt_text.shape[-1] == gt_audio.shape[-1] else torch.zeros(
                (gt_text.shape[0], gt_audio.shape[0]), device=gt_text.device
            )
            
            # Compute diagonal mean (cosine similarity between matching pairs)
            a2t_score = a2t_sim.diag().mean().item()
            t2a_score = t2a_sim.diag().mean().item()
            
            # Get dataloader name
            dataloader_name = dataloader_names.get(dataloader_idx, f'dataloader_{dataloader_idx}')
            
            # Log scores (only on rank 0 to avoid duplicate logging)
            if trainer.global_rank == 0:
                pl_module.log(
                    f'{dataloader_name}_TeacherScore/{mode}/A->T_cosine_sim',
                    a2t_score,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )
                pl_module.log(
                    f'{dataloader_name}_TeacherScore/{mode}/T->A_cosine_sim',
                    t2a_score,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )


class GenRetrieval(BaseCallback):
    """
    Callback that computes retrieval metrics (R@1, R@5, R@10) for generated audio
    embeddings against ground truth audio embeddings. Only computes audio-to-audio retrieval.
    """
    
    def __init__(
        self,
        thresholds: list = [1, 5, 10],
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
    ):
        """
        Args:
            thresholds: List of k values for R@k metrics
            enable_on_validation: Whether to compute metrics during validation
            enable_on_test: Whether to compute metrics during test
            every_n_steps: Compute metrics every N steps (None to disable step-based checking)
            every_n_epochs: Compute metrics every N epochs (default: 1, i.e., every epoch)
        """
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.thresholds = thresholds
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute retrieval metrics at the end of validation epoch."""
        if not self.enable_on_validation:
            return
        
        # Check if we should compute metrics based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_retrieval_metrics(trainer, pl_module, mode='val')
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute retrieval metrics at the end of test epoch."""
        if not self.enable_on_test:
            return
        
        # Check if we should compute metrics based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_retrieval_metrics(trainer, pl_module, mode='test')
    
    def _compute_retrieval_metrics(self, trainer: Trainer, pl_module: LightningModule, mode: str = 'val'):
        """Compute retrieval metrics for all dataloaders."""
        if mode == 'val':
            preds_dict = pl_module.val_preds
            gt_dict = pl_module.val_gt
        elif mode == 'test':
            preds_dict = pl_module.test_preds
            gt_dict = pl_module.test_gt
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get dataloader names if available
        dataloader_names = getattr(trainer.datamodule, 'dataloader_names', {})
        
        for dataloader_idx in preds_dict.keys():
            if len(preds_dict[dataloader_idx]['audio']) == 0:
                continue
            
            # Concatenate all batches (if already gathered by GenerationCallback, this is a single tensor)
            gen_audio = torch.cat(preds_dict[dataloader_idx]['audio'], dim=0)
            gt_audio = torch.cat(gt_dict[dataloader_idx]['audio'], dim=0)
            
            # Log shapes
            log.info(f"[GenRetrieval] gen_audio shape: {gen_audio.shape}, gt_audio shape: {gt_audio.shape}", extra={"sync_dist": True})
            
            # Check if gathering is needed (GenerationCallback may have already gathered)
            # If len == 1, it's likely already gathered; if world_size > len(audio_list), we need to gather
            needs_gather = (trainer.world_size > 1 and 
                          len(preds_dict[dataloader_idx]['audio']) > 1)
            
            if needs_gather:
                log.info(f"[GenRetrieval] Rank {trainer.global_rank} - tensors not pre-gathered, gathering now", extra={"sync_dist": True})
                device = next(pl_module.parameters()).device
                gen_audio = gen_audio.to(device)
                gt_audio = gt_audio.to(device)
                
                gen_audio = gather_tensor_if_distributed(gen_audio, trainer)
                gt_audio = gather_tensor_if_distributed(gt_audio, trainer)
                
                log.info(f"[GenRetrieval] After gather - gen_audio shape: {gen_audio.shape}, gt_audio shape: {gt_audio.shape}", extra={"sync_dist": True})
            else:
                log.info(f"[GenRetrieval] Rank {trainer.global_rank} - using pre-gathered tensors", extra={"sync_dist": True})
            
            # Normalize embeddings
            gen_audio = gen_audio / gen_audio.norm(dim=1, keepdim=True)
            gt_audio = gt_audio / gt_audio.norm(dim=1, keepdim=True)
            
            # Compute similarity matrix: generated audio to ground truth audio
            gen2gt_sim = gen_audio @ gt_audio.t() if gen_audio.shape[-1] == gt_audio.shape[-1] else torch.zeros(
                (gen_audio.shape[0], gt_audio.shape[0]), device=gen_audio.device
            )
            
            # Compute retrieval metrics
            gen2gt_recall, gen2gt_precision, gen2gt_ranks, gen2gt_normalized_ranks = compute_recall(gen2gt_sim)
            
            # Get dataloader name
            dataloader_name = dataloader_names.get(dataloader_idx, f'dataloader_{dataloader_idx}')
            
            # Log metrics (only on rank 0 to avoid duplicate logging)
            if trainer.is_global_zero:
                for threshold in self.thresholds:
                    if threshold in gen2gt_recall:
                        pl_module.log(
                            f'{dataloader_name}_GenRetrieval/{mode}/Gen->GT_R@{threshold}',
                            gen2gt_recall[threshold],
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False
                        )
                
                # Log median rank
                pl_module.log(
                    f'{dataloader_name}_GenRetrieval/{mode}/Gen->GT_median_rank',
                    gen2gt_normalized_ranks.median().item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )


class GenScore(BaseCallback):
    """
    Callback that computes cosine similarity scores for generated audio
    embeddings against ground truth audio embeddings. Only computes audio-to-audio similarity.
    """
    
    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
    ):
        """
        Args:
            enable_on_validation: Whether to compute scores during validation
            enable_on_test: Whether to compute scores during test
            every_n_steps: Compute scores every N steps (None to disable step-based checking)
            every_n_epochs: Compute scores every N epochs (default: 1, i.e., every epoch)
        """
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute cosine similarity scores at the end of validation epoch."""
        if not self.enable_on_validation:
            return
        
        # Check if we should compute scores based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_scores(trainer, pl_module, mode='val')
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Compute cosine similarity scores at the end of test epoch."""
        if not self.enable_on_test:
            return
        
        # Check if we should compute scores based on step/epoch filters
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        
        self._compute_scores(trainer, pl_module, mode='test')
    
    def _compute_scores(self, trainer: Trainer, pl_module: LightningModule, mode: str = 'val'):
        """Compute cosine similarity scores for all dataloaders."""
        if mode == 'val':
            preds_dict = pl_module.val_preds
            gt_dict = pl_module.val_gt
        elif mode == 'test':
            preds_dict = pl_module.test_preds
            gt_dict = pl_module.test_gt
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Get dataloader names if available
        dataloader_names = getattr(trainer.datamodule, 'dataloader_names', {})
        
        for dataloader_idx in preds_dict.keys():
            if len(preds_dict[dataloader_idx]['audio']) == 0:
                continue
            
            # Concatenate all batches (if already gathered by GenerationCallback, this is a single tensor)
            gen_audio = torch.cat(preds_dict[dataloader_idx]['audio'], dim=0)
            gt_audio = torch.cat(gt_dict[dataloader_idx]['audio'], dim=0)
            
            # Check if gathering is needed (GenerationCallback may have already gathered)
            needs_gather = (trainer.world_size > 1 and 
                          len(preds_dict[dataloader_idx]['audio']) > 1)
            
            if needs_gather:
                log.info(f"[GenScore] Rank {trainer.global_rank} - tensors not pre-gathered, gathering now", extra={"sync_dist": True})
                device = next(pl_module.parameters()).device
                gen_audio = gen_audio.to(device)
                gt_audio = gt_audio.to(device)
                
                gen_audio = gather_tensor_if_distributed(gen_audio, trainer)
                gt_audio = gather_tensor_if_distributed(gt_audio, trainer)
            else:
                log.info(f"[GenScore] Rank {trainer.global_rank} - using pre-gathered tensors", extra={"sync_dist": True})
            
            # Normalize embeddings
            gen_audio = gen_audio / gen_audio.norm(dim=1, keepdim=True)
            gt_audio = gt_audio / gt_audio.norm(dim=1, keepdim=True)
            
            # Compute similarity matrix: generated audio to ground truth audio
            gen2gt_sim = gen_audio @ gt_audio.t() if gen_audio.shape[-1] == gt_audio.shape[-1] else torch.zeros(
                (gen_audio.shape[0], gt_audio.shape[0]), device=gen_audio.device
            )
            
            # Compute diagonal mean (cosine similarity between matching pairs)
            gen2gt_score = gen2gt_sim.diag().mean().item()
            
            # Get dataloader name
            dataloader_name = dataloader_names.get(dataloader_idx, f'dataloader_{dataloader_idx}')
            
            # Log score (only on rank 0 to avoid duplicate logging)
            if trainer.global_rank == 0:
                pl_module.log(
                    f'{dataloader_name}_GenScore/{mode}/Gen->GT_cosine_sim',
                    gen2gt_score,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False
                )
