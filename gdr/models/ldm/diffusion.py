from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from gdr.utils.subject_processing import *
import lightning as L
from lightning import LightningModule

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FlowMatchEulerDiscreteScheduler
from .unet import UNet
from pytorch_lightning.cli import OptimizerCallable
from torch import optim
from gdr.models.utils.schedulers import *
import numpy as np
import yaml
import scipy
import copy

import torch

from gdr.models.utils.base import BaseModule

from torchaudio.functional import frechet_distance
from gdr.utils.instantiators import instantiate
from .engines import (
    BaseGuidanceEngine,
    BaseEditEngine,
    CFGGuidanceEngine,
    DDIMInversionEditEngine,
    DDPSteer,
    SpecifyGradient,
)

logger = logging.getLogger(__name__)


def _broadcast_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reshape a (B,) tensor into (B, 1, 1, ...) so it broadcasts over `target`.
    """
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x



            
class GDR(BaseModule):
    """
    Generative Diffusion Retriever model.
    
    Supports custom guidance and editing engines by accepting implementations
    of BaseGuidanceEngine and BaseEditEngine. This allows swapping different
    algorithms (e.g., different CFG methods, different inversion algorithms)
    without modifying the core model code.
    """
    def __init__(
        self,
        audio_encoder = None,
        text_encoder = None,
        noise_scheduler = None,
        backbone = None,
        ckpt_path = None,
        cfg = 0.1,
        freeze = False,
        guidance_engine = None,
        edit_engine = None,
        frames = 64,
        **kwargs
    ):
        """
        Args:
            guidance_engine: Optional BaseGuidanceEngine instance. If None, a default
                           CFGGuidanceEngine will be created. Can be swapped with custom
                           implementations for different guidance algorithms.
            edit_engine: Optional BaseEditEngine instance. If None, a default 
                        DDIMInversionEditEngine will be created. Can be swapped with custom 
                        implementations for different editing/inversion algorithms.
        """
        super(GDR, self).__init__(ckpt_path = ckpt_path, freeze = freeze)

        # Register encoders and backbone as submodules so they contribute to model parameters
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.backbone = backbone
        self.frames = frames
        ## DIFFUSION SCHEDULERS
        # Instantiate scheduler (diffusion scheduler, not LR scheduler)
        # This should be instantiated here since it doesn't need model parameters
        self.noise_scheduler = noise_scheduler
        
        # Handle inference_scheduler from kwargs or parameter
        inference_scheduler = kwargs.get('inference_scheduler', None)
        if inference_scheduler is not None:
            self.inference_scheduler = inference_scheduler
        else:
            self.inference_scheduler = copy.deepcopy(self.noise_scheduler)
        self.cfg = cfg

        if self.text_encoder is not None:
            test_embedding = self.text_encoder.get_text_embedding('test', use_tensor=True, return_dict=True)['last_hidden_state']
            size = test_embedding.shape
            
            self.null_text_embedding = nn.Parameter(torch.randn(size))

        else:
            self.null_text_embedding = None

        if self.backbone is not None:
            self.backbone.null_text_embedding = self.null_text_embedding

        
        
        self._guidance_engine = guidance_engine
        self._edit_engine = edit_engine

        if self.audio_encoder is not None:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # If edit_engine was provided without guidance_engine, update the reference
        # If both are None, they'll be created lazily
        if self._edit_engine is not None and self._guidance_engine is None:
            # Extract guidance engine from edit engine if available
            if hasattr(self._edit_engine, 'guidance_engine') and self._edit_engine.guidance_engine is not None:
                self._guidance_engine = self._edit_engine.guidance_engine
        
        # If engines were provided but don't have backbone/text_encoder set, set them now
        # This handles the case where engines are instantiated via config without dependencies
        if self._guidance_engine is not None and hasattr(self._guidance_engine, 'set_backbone'):
            if self._guidance_engine._backbone is None:
                self._guidance_engine.set_backbone(self.backbone)
        
        if self._edit_engine is not None:
            if hasattr(self._edit_engine, 'set_backbone') and self._edit_engine._backbone is None:
                self._edit_engine.set_backbone(self.backbone)
            if hasattr(self._edit_engine, 'set_text_encoder') and self._edit_engine._text_encoder is None:
                self._edit_engine.set_text_encoder(self.text_encoder)
            # Also ensure guidance_engine in edit_engine has backbone set
            if (hasattr(self._edit_engine, 'guidance_engine') and 
                self._edit_engine.guidance_engine is not None and
                hasattr(self._edit_engine.guidance_engine, 'set_backbone')):
                if self._edit_engine.guidance_engine._backbone is None:
                    self._edit_engine.guidance_engine.set_backbone(self.backbone)


    def _apply_cfg_masking(self, embedding, mask_proba, device):
        """Apply CFG masking using learned null embedding."""
        if not self.training or mask_proba <= 0.0 or embedding is None:
            return embedding


        bsz = embedding.shape[0]
        mask = (torch.rand(bsz, device=device) < mask_proba)

        if not mask.any():
            return embedding

        out = embedding.clone()

        if self.null_text_embedding is not None:
            null = self.null_text_embedding.to(device)
            out[mask] = null
        else:
            out[mask] = 0

        assert out.shape == embedding.shape, "Output and embedding shape mismatch"


        return out


    @property
    def guidance_engine(self):
        """Lazy initialization of guidance engine for backwards compatibility."""
        if self._guidance_engine is None:
            self._guidance_engine = CFGGuidanceEngine(self.backbone)
        return self._guidance_engine
    
    @guidance_engine.setter
    def guidance_engine(self, value):
        """Setter for guidance engine."""
        self._guidance_engine = value
    
    @property
    def edit_engine(self):
        """Lazy initialization of edit engine for backwards compatibility."""
        if self._edit_engine is None:
            self._edit_engine = DDIMInversionEditEngine(self.backbone, self.text_encoder, self.guidance_engine)
        return self._edit_engine
    
    @edit_engine.setter
    def edit_engine(self, value):
        """Setter for edit engine."""
        self._edit_engine = value

    def _apply_cfg_guidance(self, x, time, t, embedding, negative_embedding, guidance_scale, inference_scheduler = None, audio_embedding = None, audio_guidance_scale = 1.0, **kwargs):
        """
        Apply classifier-free guidance during inference.
        
        This method is kept for backwards compatibility and delegates to the guidance engine.
        
        Args:
            x: Input latents
            time: Timestep tensor
            embedding: Positive text embeddings
            negative_embedding: Negative text embeddings (or None)
            guidance_scale: Guidance scale factor
            
        Returns:
            Combined output with CFG applied
        """

        return self.guidance_engine.apply_guidance(
            x = x,
            time = time,
            t = t,
            embedding = embedding,
            negative_embedding = negative_embedding,
            guidance_scale = guidance_scale,
            inference_scheduler = inference_scheduler,
            backbone = self.backbone,
            audio_embedding = audio_embedding,
            audio_guidance_scale = audio_guidance_scale,
            **kwargs
        )

    def forward(self, latents, prompt, validation_mode=False):
        
            
        bsz = latents.shape[0]
        
        device = next(self.parameters()).device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True)
        
        encoder_hidden_states = text_dict['last_hidden_state']

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
                
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        latents_dtype = latents.dtype
        encoder_hidden_states = encoder_hidden_states.to(latents_dtype)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        bsz, length, device = *encoder_hidden_states.shape[0:2], encoder_hidden_states.device
        
        assert latents.shape == noisy_latents.shape, "Latents and noisy latents shape mismatch"
        assert latents.shape == target.shape, "Latents and target shape mismatch"
        
        # Apply CFG masking during training
        masked_embedding = self._apply_cfg_masking(encoder_hidden_states, self.cfg, device)

        model_pred = self.backbone(
            noisy_latents, time=timesteps, embedding=masked_embedding
        )

        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
            
        loss_dict = {
            'mse_loss': mse_loss,
        }
        
        assert model_pred.shape == target.shape, "Model prediction and target shape mismatch"
        return mse_loss, loss_dict


    @torch.no_grad()
    def inference(self, prompt, inference_scheduler = None, num_steps=20, guidance_scale=1, num_samples_per_prompt=1, 
                  disable_progress=True, slider = None, slider_scale = 0, negative_prompt = None, return_all_latents = False, audio_embedding = None, audio_guidance_scale = 1.0):
        
        self.eval()
        self.backbone.null_text_embedding = self.null_text_embedding
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt
        
        inference_scheduler = self.noise_scheduler if inference_scheduler is None else inference_scheduler 
        
        encoded_text = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True)
        prompt_embeds = encoded_text['last_hidden_state']
        boolean_prompt_mask = encoded_text['attention_mask']
        
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = (boolean_prompt_mask == 1).to(device)
        
        
        if negative_prompt:
            encoded_negative_text = self.text_encoder.get_text_embedding(negative_prompt, use_tensor = True, return_dict = True)
            negative_prompt_embeds = encoded_negative_text['last_hidden_state']
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            negative_prompt_mask = encoded_negative_text['attention_mask']
            negative_prompt_mask = negative_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
            negative_prompt_mask = (negative_prompt_mask == 1).to(device)
        else:
            # Use learned null embedding for unconditional guidance
            negative_prompt_embeds = self.null_text_embedding.to(device).repeat_interleave(prompt_embeds.shape[0], 0).repeat_interleave(num_samples_per_prompt, 0)

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.backbone.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)
    
        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress, leave=False)


        all_latents = []

        for i, t in enumerate(timesteps):
            latent_model_input = latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t) if hasattr(inference_scheduler, 'scale_model_input') else latent_model_input

            # expand t to batch size
            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), t, dtype=torch.long, device=device)

            # Apply CFG guidance during inference
            latents = self._apply_cfg_guidance(
                latent_model_input, time, t, prompt_embeds, negative_prompt_embeds, guidance_scale, inference_scheduler=inference_scheduler, audio_embedding=audio_embedding, audio_guidance_scale=audio_guidance_scale
            )

            # compute the previous noisy sample x_t -> x_t-1
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

            all_latents.append(latents)

        all_latents = torch.stack(all_latents, dim=0)
        if return_all_latents:
            return latents, all_latents
        else:
            return latents
    
    
    @torch.no_grad()
    def invert(
        self,
        start_latents,
        original_prompt = '',
        edit_prompt = '',
        guidance_scale=3,
        invert_steps=20,
        inference_steps=50,
        num_images_per_prompt=1,
        negative_prompt = None,
        return_intermediate_latents = False,
        inference_scheduler = None,
        verbose= False
    ):
        """
        Perform DDIM inversion and editing.
        
        This method is kept for backwards compatibility and delegates to the edit engine.
        """
        self.eval()
        self.backbone.null_text_embedding = self.null_text_embedding ## important to set this here, otherwise the edit engine will not have the null text embedding


        inference_scheduler = self.inference_scheduler if inference_scheduler is None else inference_scheduler 
        
        return self.edit_engine.invert(
            start_latents=start_latents,
            original_prompt=original_prompt,
            edit_prompt=edit_prompt,
            guidance_scale=guidance_scale,
            invert_steps=invert_steps,
            inference_steps=inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            return_intermediate_latents=return_intermediate_latents,
            inference_scheduler=inference_scheduler,
            verbose=verbose
        )
    
    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, self.frames)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents

class FMR(GDR):
    def __init__(self,
                 audio_encoder = None,
                 text_encoder = None,
                 noise_scheduler = None,
                 backbone = None,
                 ckpt_path = None,
                 freeze = False,
                 do_contrastive_loss = 0,
                 **kwargs):
        super(FMR, self).__init__(audio_encoder = audio_encoder, text_encoder = text_encoder, noise_scheduler = noise_scheduler, backbone = backbone, ckpt_path = ckpt_path, freeze = freeze, **kwargs)
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps if noise_scheduler is not None else None
        self.do_contrastive_loss = do_contrastive_loss

    @staticmethod
    def _sample_neg_indices(bsz: int, device: torch.device) -> torch.Tensor:
        """
        Sample a "negative" index for each element in the batch, ensuring j != i.
        Uses a random permutation, then fixes any accidental fixed points.
        """
        if bsz <= 1:
            # No valid negatives
            return torch.zeros((bsz,), device=device, dtype=torch.long)

        perm = torch.randperm(bsz, device=device)
        fixed = perm == torch.arange(bsz, device=device)

        # If any fixed points exist, rotate those positions by 1 (among themselves)
        if fixed.any():
            idx = torch.nonzero(fixed, as_tuple=False).flatten()
            if idx.numel() == 1:
                # Single fixed point: just swap with next (cyclic)
                k = idx.item()
                perm[k] = (k + 1) % bsz
            else:
                # Cycle the fixed indices
                perm[idx] = perm[idx.roll(1)]
        return perm
    
    def get_target_from_gt(self, gt, noise):
        return noise - gt


    def forward(self, latents, prompt, validation_mode=False):
        """Flow matching forward pass. Predicts velocity field instead of noise."""
        bsz = latents.shape[0]
        device = next(self.parameters()).device
        
        text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor=True, return_dict=True)
        encoder_hidden_states = text_dict['last_hidden_state']
        attention_mask = text_dict['attention_mask']
        
        x1 = latents
        x0 = torch.randn_like(x1)

        
        if validation_mode:
            timesteps_int = torch.full((bsz,), self.num_train_timesteps // 2, device=device, dtype=torch.long)
        else:
            timesteps_int = torch.randint(0, self.num_train_timesteps, (bsz,), device=device)

        
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=x1.device, dtype=torch.float32)
        timestep = schedule_timesteps[timesteps_int]  # exact elements of schedule_timesteps
        phi_t = self.noise_scheduler.scale_noise(sample=x1, timestep=timestep, noise=x0)

        
        flow = x0 - x1
        
        # Compute flow and predict
        masked_embedding = self._apply_cfg_masking(encoder_hidden_states, self.cfg, device)
        model_pred = self.backbone(phi_t, time=timestep, embedding=masked_embedding, attention_mask=attention_mask)  # float time
        
        reduce_dims = tuple(range(1, model_pred.ndim))
        pos_mse_per = (model_pred.float() - flow.float()).pow(2).mean(dim=reduce_dims)
        mse_loss = pos_mse_per.mean()

        loss_dict = {
            'mse_loss': mse_loss.detach(),
        }

        total_loss = mse_loss

        lam = float(self.do_contrastive_loss) if self.do_contrastive_loss is not None else 0.0
        if lam > 0.0 and bsz > 1:
            neg_idx = self._sample_neg_indices(bsz, device=x1.device)

            flow_neg = flow[neg_idx]  # "foreign" target flow from another sample in the batch

            neg_mse_per = (model_pred.float() - flow_neg.float()).pow(2).mean(dim=reduce_dims)
            neg_mse = neg_mse_per.mean()

            total_loss = mse_loss - lam * neg_mse

            loss_dict.update({
                'mse_neg': neg_mse.detach(),
                'lambda_contrastive': torch.tensor(lam, device=device),
                'contrastive_term': (-lam * neg_mse).detach(),
            })

        return total_loss, loss_dict

    def sample_timesteps(self, bsz, device, validation_mode=False):
        """Sample integer timesteps using sigmoid(N(0,1)) distribution."""
        if validation_mode:
            return torch.full((bsz,), self.num_train_timesteps // 2, device=device, dtype=torch.long)
        
        t = torch.randn(bsz, device=device)
        t = torch.sigmoid(t)
        timesteps = (t * self.num_train_timesteps).long()
        return timesteps.clamp(0, self.num_train_timesteps - 1)
    
    @torch.no_grad()
    def inference(self, prompt, inference_scheduler = None, num_steps=20, guidance_scale=1, num_samples_per_prompt=1, 
                  disable_progress=True, slider = None, slider_scale = 0, negative_prompt = None, return_all_latents = False, audio_embedding = None, audio_guidance_scale = 1.0):
        """Flow matching inference using FlowMatchEulerDiscreteScheduler."""
        self.eval()

        self.backbone.null_text_embedding = self.null_text_embedding
        
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt
        
        inference_scheduler = copy.deepcopy(self.noise_scheduler) if inference_scheduler is None else inference_scheduler
        
        encoded_text = self.text_encoder.get_text_embedding(prompt, use_tensor=True, return_dict=True)
        if prompt[0] is not None:
            prompt_embeds = encoded_text['last_hidden_state'].repeat_interleave(num_samples_per_prompt, 0)
        else:
            prompt_embeds = None
        
        if negative_prompt:
            encoded_negative_text = self.text_encoder.get_text_embedding(negative_prompt, use_tensor=True, return_dict=True)
            negative_prompt_embeds = encoded_negative_text['last_hidden_state'].repeat_interleave(num_samples_per_prompt, 0)
        else:
            # use learned null embedding - match shape of prompt_embeds (which already has num_samples_per_prompt applied)
            negative_prompt_embeds = self.null_text_embedding.to(device).repeat_interleave(batch_size, 0).repeat_interleave(num_samples_per_prompt, 0)
        
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps
        
        num_channels_latents = self.backbone.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, negative_prompt_embeds.dtype, device)

        
        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress, leave=False)
        all_latents = []
        num_train_timesteps = inference_scheduler.config.num_train_timesteps

        if audio_embedding is not None:
            # truncate or copy to match latents shape
            logging.info(f"Audio embedding shape: {audio_embedding.shape}")
            logging.info(f"Latents shape: {latents.shape}")
            if audio_embedding.shape[0] == 1:
                audio_embedding = audio_embedding.repeat(latents.shape[0], 1, 1)

            logging.info(f"Audio embedding shape: {audio_embedding.shape}")
            audio_embedding = self.get_target_from_gt(latents, audio_embedding)
            logging.info(f"Audio embedding shape: {audio_embedding.shape}")
        
        for i, t in enumerate(timesteps):
            # Convert scheduler timestep to integer for backbone
            if isinstance(t, torch.Tensor):
                t_value = t.item() if t.numel() == 1 else float(t)
            else:
                t_value = float(t)
            
            bsz = latents.shape[0]
            timesteps_for_backbone = torch.full((bsz,), t_value, device=device)
            
            latents = inference_scheduler.scale_model_input(latents, timesteps_for_backbone) if hasattr(inference_scheduler, 'scale_model_input') else latents
            latents = self._apply_cfg_guidance(
                latents, timesteps_for_backbone, t, prompt_embeds, negative_prompt_embeds, guidance_scale, inference_scheduler=inference_scheduler, audio_embedding=audio_embedding, audio_guidance_scale=audio_guidance_scale
            )
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)
            
            all_latents.append(latents)
        
        all_latents = torch.stack(all_latents, dim=0)
        if return_all_latents:
            return latents, all_latents
        else:
            return latents
    
    def invert(self, start_latents, original_prompt = '', edit_prompt = '', guidance_scale=3, invert_steps=20, inference_steps=50, num_images_per_prompt=1, negative_prompt = None, return_intermediate_latents = False, inference_scheduler = None, verbose= False):
        """Flow matching inversion. Delegates to edit engine or returns placeholder."""
        self.eval()
        inference_scheduler = self.inference_scheduler if inference_scheduler is None else inference_scheduler
        
        self.backbone.null_text_embedding = self.null_text_embedding
        if self._edit_engine is not None:
            return self.edit_engine.invert(
                start_latents=start_latents,
                original_prompt=original_prompt,
                edit_prompt=edit_prompt,
                guidance_scale=guidance_scale,
                invert_steps=invert_steps,
                inference_steps=inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                negative_prompt=negative_prompt,
                return_intermediate_latents=return_intermediate_latents,
                inference_scheduler=inference_scheduler,
                verbose=verbose
            )
        else:
            if return_intermediate_latents:
                return start_latents, [start_latents]
            return start_latents

    
class LightningGDR(GDR, LightningModule):
    
    def __init__(self,
                audio_encoder = None,
                text_encoder = None,
                noise_scheduler = None,
                backbone = None,
                ckpt_path = None,
                freeze = False,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                inference_scheduler = None,
                **kwargs
                ):
        
        LightningModule.__init__(self)
        super(LightningGDR, self).__init__(
            audio_encoder = audio_encoder,
            text_encoder = text_encoder,
            noise_scheduler = noise_scheduler,
            backbone = backbone,
            ckpt_path = ckpt_path,
            freeze = freeze,
            inference_scheduler = inference_scheduler,
            **kwargs)
        
        self.first_run = True
        # Store optimizer config as dict (don't instantiate yet - needs model parameters)
        # If it's already a dict, keep it; if Hydra instantiated it, we'll handle it in configure_optimizers
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.reset_preds_gt_dicts()

    def reset_preds_gt_dicts(self):
        # Organize by dataloader_idx to support multiple validation/test dataloaders
        # Initialize empty dicts; will be populated per dataloader_idx
        self.val_preds = {}
        self.val_gt = {}
        self.test_preds = {}
        self.test_gt = {}
    
    def _get_or_init_dataloader_dict(self, dataloader_idx, mode='val'):
        """Get or initialize the dictionary for a specific dataloader_idx"""
        if mode == 'val':
            if dataloader_idx not in self.val_preds:
                self.val_preds[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            if dataloader_idx not in self.val_gt:
                self.val_gt[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            return self.val_preds[dataloader_idx], self.val_gt[dataloader_idx]
        elif mode == 'test':
            if dataloader_idx not in self.test_preds:
                self.test_preds[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            if dataloader_idx not in self.test_gt:
                self.test_gt[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            return self.test_preds[dataloader_idx], self.test_gt[dataloader_idx]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        
        
    def training_step(self, batch, batch_idx):
        import logging
        import torch.distributed as dist
        log = logging.getLogger(__name__)
        
        # Barrier 1: Entry to training_step
        if self.trainer.world_size > 1 and dist.is_initialized():
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - entering training_step", extra={"sync_dist": True})
            dist.barrier()
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - passed entry barrier", extra={"sync_dist": True})
        
        latents = batch['audio']
        prompt = batch['prompt']
        
        b = latents.shape[0]
        log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - batch_size={b}", extra={"sync_dist": True})
        
        # Barrier 2: Before forward pass
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - starting forward pass", extra={"sync_dist": True})
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt)
        
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        # Barrier 3: After forward pass, before logging
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - forward pass done, loss={loss.item():.4f}, loss_dict_keys={list(loss_dict.keys())}", extra={"sync_dist": True})
        
        # Log metrics - use reduce_fx='mean' to ensure consistent aggregation across ranks
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        
        # Barrier 3b: After main loss log
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - logged main loss", extra={"sync_dist": True})
        
        # Log each component of loss_dict separately to ensure consistency
        for i, (key, value) in enumerate(loss_dict.items()):
            self.log(f'{key}', value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
            # Barrier after each loss_dict component
            if self.trainer.world_size > 1 and dist.is_initialized() and i == 0:  # Only log for first item to reduce spam
                dist.barrier()
                log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - logged loss_dict['{key}']", extra={"sync_dist": True})
        
        # Barrier 4: After all loss_dict logging
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - logged all loss_dict components", extra={"sync_dist": True})
        
        # Log learning rate without sync (it's the same on all ranks)
        if self.trainer.is_global_zero:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)
        
        # PyTorch Lightning handles scheduler stepping automatically when configure_optimizers
        # returns the proper format with interval='step'. No manual stepping needed.
        # Lightning accounts for gradient accumulation automatically.
        
        # Barrier 5: Before return
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[TrainingStep] Rank {self.trainer.global_rank} batch {batch_idx} - exiting training_step", extra={"sync_dist": True})

        return loss
        
    
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        """Validation step that computes loss. Generation is handled by GenerationCallback."""
        import logging
        import torch.distributed as dist
        log = logging.getLogger(__name__)
        
        # Barrier: Entry to validation_step
        if self.trainer.world_size > 1 and dist.is_initialized():
            log.info(f"[ValidationStep] Rank {self.trainer.global_rank} dataloader {dataloader_idx} batch {batch_idx} - entering", extra={"sync_dist": True})
            dist.barrier()
        
        latents = batch['audio']
        prompt = batch['prompt']
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt, validation_mode=True)
        
        # Barrier: Before logging
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        # Log each component separately
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        
        # Barrier: Before return
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[ValidationStep] Rank {self.trainer.global_rank} dataloader {dataloader_idx} batch {batch_idx} - exiting", extra={"sync_dist": True})
        
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        """Test step that computes loss. Generation is handled by GenerationCallback."""
        latents = batch['audio']
        prompt = batch['prompt']
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt, validation_mode=True)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        # Log each component separately
        for key, value in loss_dict.items():
            self.log(f'test_{key}', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        
        return loss
    
    def on_validation_epoch_end(self):
        self.reset_preds_gt_dicts()

    def on_test_epoch_end(self):
        self.reset_preds_gt_dicts()


class LightningFMR(FMR, LightningModule):
    
    def __init__(self,
                audio_encoder = None,
                text_encoder = None,
                noise_scheduler = None,
                backbone = None,
                ckpt_path = None,
                freeze = False,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                inference_scheduler = None,
                do_contrastive_loss = 0,
                **kwargs
                ):
        LightningModule.__init__(self)
        super(LightningFMR, self).__init__(
            audio_encoder = audio_encoder,
            text_encoder = text_encoder,
            noise_scheduler = noise_scheduler,
            backbone = backbone,
            ckpt_path = ckpt_path,
            freeze = freeze,
            inference_scheduler = inference_scheduler,
            do_contrastive_loss = do_contrastive_loss,
            **kwargs)
        
        self.first_run = True
        # Store optimizer config as dict (don't instantiate yet - needs model parameters)
        # If it's already a dict, keep it; if Hydra instantiated it, we'll handle it in configure_optimizers
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.reset_preds_gt_dicts()

    def reset_preds_gt_dicts(self):
        # Organize by dataloader_idx to support multiple validation/test dataloaders
        # Initialize empty dicts; will be populated per dataloader_idx
        self.val_preds = {}
        self.val_gt = {}
        self.test_preds = {}
        self.test_gt = {}
    
    def _get_or_init_dataloader_dict(self, dataloader_idx, mode='val'):
        """Get or initialize the dictionary for a specific dataloader_idx"""
        if mode == 'val':
            if dataloader_idx not in self.val_preds:
                self.val_preds[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            if dataloader_idx not in self.val_gt:
                self.val_gt[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            return self.val_preds[dataloader_idx], self.val_gt[dataloader_idx]
        elif mode == 'test':
            if dataloader_idx not in self.test_preds:
                self.test_preds[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            if dataloader_idx not in self.test_gt:
                self.test_gt[dataloader_idx] = {
                    'audio': [],
                    'prompt': [],
                    'prompt_text': [],
                    'batch_sizes': []
                }
            return self.test_preds[dataloader_idx], self.test_gt[dataloader_idx]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def training_step(self, batch, batch_idx):
        latents = batch['audio']
        prompt = batch['prompt']
        
        b = latents.shape[0]
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt)
        
        if latents.dtype == torch.float16:
            latents = latents.float()
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        
        # Note: PyTorch Lightning handles scheduler stepping automatically
        # Manual stepping is removed to avoid potential rank desynchronization
        # if hasattr(self, 'scheduler') and self.scheduler is not None and not isinstance(self.scheduler, dict) and not hasattr(self.scheduler, 'func'):
        #     self.scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        """Validation step that computes loss. Generation is handled by GenerationCallback."""
        import logging
        import torch.distributed as dist
        log = logging.getLogger(__name__)
        
        # Barrier: Entry to validation_step
        if self.trainer.world_size > 1 and dist.is_initialized():
            log.info(f"[ValidationStep] Rank {self.trainer.global_rank} dataloader {dataloader_idx} batch {batch_idx} - entering", extra={"sync_dist": True})
            dist.barrier()
        
        latents = batch['audio']
        prompt = batch['prompt']
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt, validation_mode=True)
        
        # Barrier: Before logging
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        # Log each component separately
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        
        # Barrier: Before return
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.barrier()
            log.info(f"[ValidationStep] Rank {self.trainer.global_rank} dataloader {dataloader_idx} batch {batch_idx} - exiting", extra={"sync_dist": True})
        
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        """Test step that computes loss. Generation is handled by GenerationCallback."""
        latents = batch['audio']
        prompt = batch['prompt']
        
        loss, loss_dict = self(latents.permute(0,2,1), prompt, validation_mode=True)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        # Log each component separately
        for key, value in loss_dict.items():
            self.log(f'test_{key}', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx='mean')
        
        return loss
    
    def on_validation_epoch_end(self):
        self.reset_preds_gt_dicts()

    def on_test_epoch_end(self):
        self.reset_preds_gt_dicts()
