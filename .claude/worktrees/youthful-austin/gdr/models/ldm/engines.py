"""
Engine classes for handling guidance and editing mechanisms in diffusion models.
These engines can be swapped out or customized while maintaining backwards compatibility.
"""

import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
from diffusers import DDIMInverseScheduler
import logging
import torch.nn.functional as F

# Try to import custom_fwd/custom_bwd for SpecifyGradient, fallback if not available
try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ImportError:
    # Fallback decorators if custom_fwd/custom_bwd not available
    def custom_fwd(fn):
        return fn
    def custom_bwd(fn):
        return fn


class BaseGuidanceEngine(ABC):
    """
    Abstract base class for guidance engines.
    
    This defines the interface that all guidance algorithms must implement,
    allowing different guidance strategies to be swapped in.
    """
    
    @abstractmethod
    def apply_guidance(self, x, time, embedding, negative_embedding, guidance_scale, backbone=None, inference_scheduler=None, step = True, **kwargs):
        """
        Apply guidance during inference.
        
        Args:
            x: Input latents
            time: Timestep tensor
            embedding: Positive text embeddings
            negative_embedding: Negative text embeddings (or None)
            guidance_scale: Guidance scale factor
            backbone: Optional backbone to use (if None, implementation should use its own)
            
        Returns:
            Combined output with guidance applied
        """
        pass


class CFGGuidanceEngine(BaseGuidanceEngine):
    """
    Classifier-free guidance (CFG) implementation.
    
    This engine handles the guidance mechanism that allows controlling
    the strength of conditioning on text prompts.
    
    To create a custom guidance algorithm, subclass BaseGuidanceEngine
    and implement the apply_guidance method.
    
    Example:
        class CustomGuidanceEngine(BaseGuidanceEngine):
            def apply_guidance(self, x, time, embedding, negative_embedding, guidance_scale):
                # Your custom guidance algorithm here
                ...
    """
    
    def __init__(self, backbone=None):
        """
        Args:
            backbone: The UNet backbone model to use for predictions.
                     Can be None and set later via set_backbone() to avoid duplication
                     when engines are instantiated via config.
        """
        self._backbone = backbone
    
    def set_backbone(self, backbone):
        """Set the backbone model (useful when engine is instantiated before backbone is available)."""
        self._backbone = backbone
    
    @property
    def backbone(self):
        """Get the backbone model."""
        if self._backbone is None:
            raise ValueError("Backbone not set. Call set_backbone() or provide it in __init__.")
        return self._backbone
    
    def apply_guidance(self, x, time, t, embedding, negative_embedding, guidance_scale, inference_scheduler=None, backbone=None, step = True, audio_embedding = None, audio_guidance_scale = 1.0, **kwargs):
        """
        Apply classifier-free guidance during inference using standard formulation.
        
        Uses efficient batched forward pass instead of multiple separate calls.
        
        Args:
            x: Input latents [batch_size, ...]
            time: Timestep tensor [batch_size] or scalar
            embedding: Positive (conditional) text embeddings
            negative_embedding: Negative text embeddings. Can be:
                - None: Uses unconditional (empty) embedding for CFG (requires backbone.null_text_embedding)
                - Tensor: Uses as negative prompt (e.g., "bad quality, blurry")
            guidance_scale: Guidance scale factor. 
                - 1.0: No guidance, returns conditional output only
                - >1.0: Amplifies difference from negative/unconditional
            inference_scheduler: Optional scheduler for stepping (if step=True)
            backbone: Optional backbone to use (if None, uses self.backbone)
            step: If True and inference_scheduler provided, perform scheduler step
            
        Returns:
            - If step=False or inference_scheduler=None: noise prediction tensor
            - If step=True and inference_scheduler provided: latents after scheduler step
        """
        backbone = backbone if backbone is not None else self.backbone

        if audio_guidance_scale is None:
            audio_guidance_scale = guidance_scale

        if guidance_scale == 1.0:
            # No CFG, just return conditional output
            noise_pred = backbone(x, time=time, embedding=embedding)
            if inference_scheduler is not None and step:
                # Extract scalar timestep from time tensor
                return inference_scheduler.step(noise_pred, t, x).prev_sample
            return noise_pred
        
        batch_size = x.shape[0]
        
        # Handle None negative_embedding - use learned null embedding if available
        if negative_embedding is None:
            if hasattr(backbone, 'null_text_embedding'):
                # Use learned null embedding
                null_emb = backbone.null_text_embedding
                if null_emb.dim() == 2:
                    null_emb = null_emb.unsqueeze(0)
                if null_emb.shape[0] == 1 and batch_size > 1:
                    negative_embedding = null_emb.expand(batch_size, *null_emb.shape[1:]).contiguous()
                else:
                    negative_embedding = null_emb
            else:
                # Fallback: use empty string embedding (requires text_encoder, but we don't have it here)
                # This will fail - negative_embedding must be provided or backbone must have null_text_embedding
                raise ValueError(
                    "negative_embedding is None but backbone.null_text_embedding is not available. "
                    "Either provide negative_embedding or ensure backbone has null_text_embedding attribute."
                )
        
        # Concatenate inputs for batched forward pass (2x more efficient)
        # [negative, positive]
        if embedding is not None:
            x_combined = torch.cat([x, x], dim=0)
            time_combined = torch.cat([time, time], dim=0) if isinstance(time, torch.Tensor) else time
            
            # Batched forward pass with concatenated embeddings
            embedding_combined = torch.cat([negative_embedding, embedding], dim=0)

            combined_output = backbone(x_combined, time=time_combined, embedding=embedding_combined)
            
            # Split outputs
            negative_output, positive_output = combined_output.chunk(2, dim=0)
        else:
            negative_output = backbone(x, time=time, embedding=negative_embedding)
            positive_output = None
    
        # Apply CFG formula: neg + guidance_scale * (pos - neg)
        output = negative_output - guidance_scale * negative_output
        if positive_output is not None:
            output = output + guidance_scale * positive_output
        if audio_embedding is not None:
            output = output + audio_guidance_scale * audio_embedding

        if inference_scheduler is not None and step:
            # Validate that scheduler is not CFG++-only (CFG++ schedulers should not be used with CFGGuidanceEngine)
            if hasattr(inference_scheduler, 'cfgpp') and getattr(inference_scheduler, 'cfgpp', False):
                raise ValueError(
                    "CFGGuidanceEngine cannot be used with CFG++-compatible schedulers (e.g., DDPMSchedulerPlusPlus, FlowMatchEulerDiscretePlusPlus). "
                    "Use CFGPlusPlusGuidanceEngine with CFG++ schedulers, or use standard schedulers with CFGGuidanceEngine."
                )
            
            return inference_scheduler.step(output, t, x).prev_sample
        
        return output


class BaseEditEngine(ABC):
    """
    Abstract base class for edit engines.
    
    This defines the interface that all editing algorithms must implement,
    allowing different editing strategies (e.g., DDIM, PNDM, etc.) to be swapped in.
    """
    
    @abstractmethod
    def invert(
        self,
        start_latents,
        original_prompt='',
        edit_prompt='',
        guidance_scale=3,
        invert_steps=20,
        inference_steps=50,
        num_images_per_prompt=1,
        negative_prompt=None,
        return_intermediate_latents=False,
        inference_scheduler=None,
        verbose=False
    ):
        """
        Perform inversion and editing.
        
        Args:
            start_latents: Starting latents to invert
            original_prompt: Original prompt for inversion
            edit_prompt: Edit prompt for generation
            guidance_scale: Guidance scale for CFG
            invert_steps: Number of inversion steps
            inference_steps: Number of inference steps
            num_images_per_prompt: Number of images per prompt
            negative_prompt: Optional negative prompt
            return_intermediate_latents: Whether to return intermediate latents
            inference_scheduler: Scheduler to use for inference
            verbose: Whether to show progress bars
            
        Returns:
            Final edited latents, and optionally intermediate latents
        """
        pass


class DDIMInversionEditEngine(BaseEditEngine):
    """
    DDIM inversion and editing implementation.
    
    This engine performs inversion of latents back to noise space
    and then generates new latents conditioned on edit prompts.
    
    To create a custom editing algorithm (e.g., using PNDM, DPM-Solver, etc.),
    subclass BaseEditEngine and implement the invert method.
    
    Example:
        class CustomEditEngine(BaseEditEngine):
            def invert(self, start_latents, original_prompt='', edit_prompt='', ...):
                # Your custom editing algorithm here
                ...
    """
    
    def __init__(self, backbone=None, text_encoder=None, guidance_engine=None):
        """
        Args:
            backbone: The UNet backbone model to use for predictions.
                     Can be None and set later via set_backbone() to avoid duplication
                     when engines are instantiated via config.
            text_encoder: The text encoder to use for encoding prompts.
                         Can be None and set later via set_text_encoder() to avoid duplication
                         when engines are instantiated via config.
            guidance_engine: Optional BaseGuidanceEngine instance (or any object with apply_guidance method).
                           If None, will not use guidance during editing.
        """
        self._backbone = backbone
        self._text_encoder = text_encoder
        self.guidance_engine = guidance_engine
    
    def set_backbone(self, backbone):
        """Set the backbone model (useful when engine is instantiated before backbone is available)."""
        self._backbone = backbone
    
    def set_text_encoder(self, text_encoder):
        """Set the text encoder (useful when engine is instantiated before text_encoder is available)."""
        self._text_encoder = text_encoder
    
    @property
    def backbone(self):
        """Get the backbone model."""
        if self._backbone is None:
            raise ValueError("Backbone not set. Call set_backbone() or provide it in __init__.")
        return self._backbone
    
    @property
    def text_encoder(self):
        """Get the text encoder."""
        if self._text_encoder is None:
            raise ValueError("Text encoder not set. Call set_text_encoder() or provide it in __init__.")
        return self._text_encoder
    
    def invert(
        self,
        start_latents,
        original_prompt='',
        edit_prompt='',
        guidance_scale=3,
        invert_steps=20,
        inference_steps=50,
        num_images_per_prompt=1,
        negative_prompt=None,
        return_intermediate_latents=False,
        inference_scheduler=None,
        verbose=False
    ):
        """
        Perform DDIM inversion and editing.
        
        Args:
            start_latents: Starting latents to invert
            original_prompt: Original prompt for inversion
            edit_prompt: Edit prompt for generation
            guidance_scale: Guidance scale for CFG
            invert_steps: Number of inversion steps
            inference_steps: Number of inference steps
            num_images_per_prompt: Number of images per prompt
            negative_prompt: Optional negative prompt
            return_intermediate_latents: Whether to return intermediate latents
            inference_scheduler: Scheduler to use for inference
            verbose: Whether to show progress bars
            
        Returns:
            Final edited latents, and optionally intermediate latents
        """
        device = next(self.backbone.parameters()).device
        
        # Create inversion scheduler
        inversion_scheduler = DDIMInverseScheduler.from_pretrained(
            inference_scheduler.config.scheduler_name,
            subfolder="scheduler",
            prediction_type=inference_scheduler.config.prediction_type
        )
        inversion_scheduler.config.prediction_type = inference_scheduler.config.prediction_type
        
        # Encode prompts
        edit_prompt_embedding = self.text_encoder.get_text_embedding(
            edit_prompt, use_tensor=True, return_dict=True
        )['last_hidden_state'].repeat_interleave(num_images_per_prompt, 0)
        
        if negative_prompt:
            negative_prompt_embeddings = self.text_encoder.get_text_embedding(
                negative_prompt, use_tensor=True, return_dict=True
            )['last_hidden_state'].repeat_interleave(num_images_per_prompt, 0)
        else:
            negative_prompt_embeddings = None
            
        original_prompt_embeddings = self.text_encoder.get_text_embedding(
            original_prompt, use_tensor=True, return_dict=True
        )['last_hidden_state'].repeat_interleave(num_images_per_prompt, 0)
    
        latents = start_latents.to(device)
        intermediate_latents = [latents]


        latents_dtype = latents.dtype
        edit_prompt_embedding = edit_prompt_embedding.to(latents_dtype)
        negative_prompt_embeddings = negative_prompt_embeddings.to(latents_dtype)
        original_prompt_embeddings = original_prompt_embeddings.to(latents_dtype)
        
        inversion_scheduler.set_timesteps(inference_steps, device=device)
        inference_scheduler.set_timesteps(inference_steps, device=device)

        # Inversion phase: go back to noise space
        ts = []
        for i in tqdm(range(invert_steps), disable=not verbose, leave=False, desc="Inverting"):
            t = inversion_scheduler.timesteps[i]
            ts.append(t)
            latent_model_input = latents
            latent_model_input = inversion_scheduler.scale_model_input(latent_model_input, t) if hasattr(inversion_scheduler, 'scale_model_input') else latent_model_input
            
            time = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)
            
            # Apply CFG guidance during inversion
            if self.guidance_engine is not None:
                # For inversion, we typically don't use negative prompts, so pass None
                # CFGGuidanceEngine will use backbone.null_text_embedding if available
                latents = self.guidance_engine.apply_guidance(
                    latent_model_input, time, original_prompt_embeddings, None, guidance_scale, 
                    inference_scheduler=inversion_scheduler, backbone=self.backbone, step=True
                )
            else:
                # Fallback to direct backbone call if no guidance engine
                noise_pred = self.backbone(latent_model_input, time=time, embedding=original_prompt_embeddings)
                latents = inversion_scheduler.step(noise_pred, t, latents).prev_sample
            
            intermediate_latents.append(latents)
        
        # Editing phase: generate from inverted latents with edit prompt
        edit_latents = latents.clone()
        latent_model_input = edit_latents
        intermediate_edit_latents = []
        
        for i, t in tqdm(enumerate(ts[::-1]), disable=not verbose, leave=False, desc="Reverting with edit"):
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t) if hasattr(inference_scheduler, 'scale_model_input') else latent_model_input
            
            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), t, dtype=torch.long, device=device)
            
            # Apply CFG guidance during inversion revert
            if self.guidance_engine is not None:
                latents = self.guidance_engine.apply_guidance(
                    latent_model_input, time, edit_prompt_embedding, negative_prompt_embeddings, guidance_scale, 
                    inference_scheduler=inference_scheduler, backbone=self.backbone, step=True
                )
            else:
                # Fallback to direct backbone call if no guidance engine
                pred = self.backbone(latent_model_input, time=time, embedding=edit_prompt_embedding)
                latents = inference_scheduler.step(pred, t, latents).prev_sample
            intermediate_edit_latents.append(latents)
            latent_model_input = latents
            
        intermediate_edit_latents = torch.stack(intermediate_edit_latents, dim=0)
        intermediate_latents = torch.stack(intermediate_latents, dim=0)
        
        if return_intermediate_latents:
            return intermediate_edit_latents[-1], intermediate_latents, intermediate_edit_latents
        else:
            return intermediate_edit_latents[-1]


class SpecifyGradient(torch.autograd.Function):
    """
    Custom gradient function for manual gradient manipulation.
    
    This allows fine-grained control over gradients in a deep learning model that relies
    on automatic differentiation. The class is particularly useful for gradient clipping,
    applying noise to gradients, or implementing custom gradient-based optimization.
    
    The forward method returns a dummy value that will be scaled by amp's scaler,
    allowing us to get the scale in the backward pass.
    """
    
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        """
        Forward pass that stores the ground truth gradient.
        
        Args:
            ctx: Context object for storing information needed for backward computation
            input_tensor: Input tensor to this layer
            gt_grad: Ground truth gradient to use in backward pass
            
        Returns:
            Dummy tensor of ones with same device and dtype as input_tensor
        """
        ctx.save_for_backward(gt_grad)
        # Return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        """
        Backward pass that applies the stored gradient scaled by grad_scale.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_scale: Gradient scaling factor from automatic differentiation
            
        Returns:
            Scaled ground truth gradient and None (no gradient for gt_grad itself)
        """
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class DDPSteer(BaseEditEngine):
    """
    Delta Denoising Score (DDP) Steer editing engine.
    
    This engine implements SteerMusic-style editing using delta denoising score.
    It optimizes latents by computing the difference between noise predictions
    for edit and reference prompts, enabling controllable audio editing.
    
    Based on the SteerMusic approach, this engine:
    1. Encodes the original audio into latents
    2. Optimizes latents using delta denoising score between edit and reference prompts
    3. Uses a custom gradient function to apply the computed gradients
    """
    
    def __init__(self, backbone=None, text_encoder=None, guidance_engine=None, 
                 weight_aug=2, validation_step=400, latent_size=None):
        """
        Args:
            backbone: The UNet backbone model to use for predictions.
                     Can be None and set later via set_backbone() to avoid duplication
                     when engines are instantiated via config.
            text_encoder: The text encoder to use for encoding prompts.
                         Can be None and set later via set_text_encoder() to avoid duplication
                         when engines are instantiated via config.
            guidance_engine: Optional BaseGuidanceEngine instance (or any object with apply_guidance method).
                           If None, will not use guidance during editing.
            weight_aug: Weight augmentation factor for gradient computation (default: 2)
            validation_step: Step interval for validation/saving intermediate results (default: 400)
            latent_size: Optional latent size. If None, will be inferred from start_latents
        """
        self._backbone = backbone
        self._text_encoder = text_encoder
        self.guidance_engine = guidance_engine
        self.weight_aug = weight_aug
        self.validation_step = validation_step
    
    def set_backbone(self, backbone):
        """Set the backbone model (useful when engine is instantiated before backbone is available)."""
        self._backbone = backbone
    
    def set_text_encoder(self, text_encoder):
        """Set the text encoder (useful when engine is instantiated before text_encoder is available)."""
        self._text_encoder = text_encoder
    
    @property
    def backbone(self):
        """Get the backbone model."""
        if self._backbone is None:
            raise ValueError("Backbone not set. Call set_backbone() or provide it in __init__.")
        return self._backbone
    
    @property
    def text_encoder(self):
        """Get the text encoder."""
        if self._text_encoder is None:
            raise ValueError("Text encoder not set. Call set_text_encoder() or provide it in __init__.")
        return self._text_encoder
    
    def _get_alpha(self, scheduler, t):
        """
        Get alpha value for a given timestep from the scheduler.
        
        Args:
            scheduler: Diffusion scheduler
            t: Timestep tensor or value
            
        Returns:
            Alpha value(s) for the given timestep(s)
        """
        if hasattr(scheduler, 'alphas_cumprod'):
            # Convert timestep to index if needed
            if isinstance(t, torch.Tensor):
                t_idx = t.long()
            else:
                t_idx = int(t)
            # Ensure index is within bounds
            t_idx = torch.clamp(t_idx, 0, len(scheduler.alphas_cumprod) - 1)
            if isinstance(t_idx, torch.Tensor):
                return scheduler.alphas_cumprod[t_idx]
            else:
                return scheduler.alphas_cumprod[t_idx]
        elif hasattr(scheduler, 'alphas'):
            if isinstance(t, torch.Tensor):
                t_idx = t.long()
            else:
                t_idx = int(t)
            t_idx = torch.clamp(t_idx, 0, len(scheduler.alphas) - 1)
            if isinstance(t_idx, torch.Tensor):
                return scheduler.alphas[t_idx]
            else:
                return scheduler.alphas[t_idx]
        else:
            # Fallback: approximate alpha from timestep
            # This is a simple approximation - may need adjustment based on scheduler type
            if isinstance(t, torch.Tensor):
                t_normalized = t.float() / scheduler.config.num_train_timesteps if hasattr(scheduler.config, 'num_train_timesteps') else t.float() / 1000.0
            else:
                t_normalized = float(t) / (scheduler.config.num_train_timesteps if hasattr(scheduler.config, 'num_train_timesteps') else 1000.0)
            return 1.0 - t_normalized
    
    def _predict_noise(self, x, time, prompt_embeddings, guidance_scale, scheduler, as_latent=True):
        """
        Predict noise using the backbone model.
        
        Args:
            x: Input latents
            time: Timestep tensor
            prompt_embeddings: Text embeddings for conditioning
            guidance_scale: Guidance scale for CFG
            scheduler: Diffusion scheduler
            as_latent: Whether input is already in latent space
            
        Returns:
            Predicted noise, timestep, and noise tensor
        """
        device = next(self.backbone.parameters()).device
        
        # Sample noise if needed (for training/inference consistency)
        noise = torch.randn_like(x)
        
        # Get timestep value
        if isinstance(time, torch.Tensor):
            t = time[0].item() if time.numel() > 0 else 0
        else:
            t = time
        
        # Apply guidance if available
        if self.guidance_engine is not None:
            noise_pred = self.guidance_engine.apply_guidance(
                x, time, prompt_embeddings, None, guidance_scale, backbone=self.backbone, step = False
            )
        else:
            # Direct backbone call
            noise_pred = self.backbone(x, time=time, embedding=prompt_embeddings)
        

        return noise_pred, t, noise
    
    def invert(
        self,
        start_latents,
        original_prompt='',
        edit_prompt='',
        guidance_scale=15,
        invert_steps=400,
        inference_steps=50,
        num_images_per_prompt=1,
        negative_prompt=None,
        return_intermediate_latents=False,
        inference_scheduler=None,
        verbose=False,
        prompt_ref=None
    ):
        """
        Perform DDP Steer editing using delta denoising score.
        
        Args:
            start_latents: Starting latents to edit (already encoded audio)
            original_prompt: Original prompt (used as reference if prompt_ref not provided)
            edit_prompt: Edit prompt for generation
            guidance_scale: Guidance scale for CFG (default: 15)
            invert_steps: Number of optimization steps (default: 400)
            inference_steps: Number of inference steps (unused in this engine, kept for compatibility)
            num_images_per_prompt: Number of images per prompt
            negative_prompt: Optional negative prompt (unused in this engine)
            return_intermediate_latents: Whether to return intermediate latents
            inference_scheduler: Scheduler to use (required for alpha values)
            verbose: Whether to show progress bars
            prompt_ref: Reference prompt for delta denoising. If None, uses original_prompt
            
        Returns:
            Final edited latents, and optionally intermediate latents
        """
        device = next(self.backbone.parameters()).device
        
        if inference_scheduler is None:
            raise ValueError("inference_scheduler is required for DDPSteer engine")
        
        # Use prompt_ref if provided, otherwise use original_prompt
        if prompt_ref is None:
            prompt_ref = original_prompt
        
        # Encode prompts
        if prompt_ref:
            text_z_ref = self.text_encoder.get_text_embedding(
                prompt_ref, use_tensor=True, return_dict=True
            )['last_hidden_state'].repeat_interleave(num_images_per_prompt, 0)
        else:
            # Use null/empty embedding if no reference prompt
            text_z_ref = self.text_encoder.get_text_embedding(
                '', use_tensor=True, return_dict=True
            )['last_hidden_state'].repeat_interleave(num_images_per_prompt, 0)
        
        text_z = self.text_encoder.get_text_embedding(
            edit_prompt, use_tensor=True, return_dict=True
        )['last_hidden_state'].repeat_interleave(num_images_per_prompt, 0)
        
        # Initialize latent from start_latents
        latent = start_latents.to(device)
        
        
        # Freeze model parameters
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Make latent trainable
        latent = latent.clone().detach().requires_grad_(True)
        latent_ref = latent.clone().detach()
        
        # Setup optimizer
        optim = torch.optim.SGD([latent], lr=0.02)
        scheduler_opt = torch.optim.lr_scheduler.StepLR(optim, 20, 0.9)
        
        intermediate_latents = [latent.detach().clone()]
        
        if verbose:
            print(f'[INFO] Starting DDP Steer editing with {invert_steps} steps...')
        
        # Optimization loop
        for i in tqdm(range(invert_steps + 1), disable=not verbose, leave=False, desc="DDP Steer editing"):
            optim.zero_grad()
            
            x = latent
            
            # Sample a random timestep for this optimization step
            # Use scheduler's timesteps if available
            if hasattr(inference_scheduler, 'timesteps') and len(inference_scheduler.timesteps) > 0:
                # Sample from available timesteps
                t_idx = torch.randint(0, len(inference_scheduler.timesteps), (1,), device=device)
                t = inference_scheduler.timesteps[t_idx].item()
            else:
                # Fallback: sample from full range
                max_t = inference_scheduler.config.num_train_timesteps if hasattr(inference_scheduler.config, 'num_train_timesteps') else 1000
                t = torch.randint(0, max_t, (1,), device=device).item()
            
            time_tensor = torch.full((latent.shape[0],), t, dtype=torch.long, device=device)
            
            # Predict noise for edit prompt
            noise_pred, t_val, noise = self._predict_noise(
                x, time_tensor, text_z, guidance_scale, inference_scheduler, as_latent=True
            )
            
            # Predict noise for reference prompt (with same timestep and noise)
            with torch.no_grad():
                noise_pred_ref, _, _ = self._predict_noise(
                    latent_ref, time_tensor, text_z_ref, guidance_scale, inference_scheduler, as_latent=True
                )
            
            # Validation/saving
            if i % self.validation_step == 0:
                if return_intermediate_latents:
                    intermediate_latents.append(x.detach().clone())
                if verbose:
                    print(f'[INFO] Step {i}: Latent shape {x.shape}')
            
            # Compute weight based on alpha
            alpha = self._get_alpha(inference_scheduler, t)
            if isinstance(alpha, torch.Tensor):
                alpha_val = alpha.item() if alpha.numel() == 1 else alpha
            else:
                alpha_val = alpha
            
            w = self.weight_aug * (1 - alpha_val)
            
            # Compute gradient: delta denoising score
            grad = w * (noise_pred - noise_pred_ref)
            grad = torch.nan_to_num(grad)
            
            # Apply custom gradient
            loss = SpecifyGradient.apply(x, grad)
            loss.backward()
            
            optim.step()
            scheduler_opt.step()
        
        if return_intermediate_latents:
            intermediate_latents = torch.stack(intermediate_latents, dim=0)
            return latent.detach(), intermediate_latents
        else:
            return latent.detach()

class CFGPlusPlusGuidanceEngine(BaseGuidanceEngine):
    def __init__(self, backbone=None, clamp_lambda=True, allow_extrapolation=False):
        self._backbone = backbone
        self.clamp_lambda = clamp_lambda
        self.allow_extrapolation = allow_extrapolation

    def set_backbone(self, backbone):
        self._backbone = backbone

    @property
    def backbone(self):
        if self._backbone is None:
            raise ValueError("Backbone not set. Call set_backbone() or provide it in __init__.")
        return self._backbone

    @torch.no_grad()
    def apply_guidance(
        self,
        x,
        time,
        embedding,
        negative_embedding,
        guidance_scale,
        inference_scheduler=None,
        backbone=None,
        step = True,
        **kwargs,
    ):
        backbone = backbone if backbone is not None else self.backbone

        lam = float(guidance_scale)
        if self.clamp_lambda and not self.allow_extrapolation:
            lam = max(0.0, min(1.0, lam))

        if negative_embedding is None:
            raise ValueError("CFG++ requires negative/uncond embedding (null prompt embedding).")

        # Batched forward: [uncond, cond]
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([time, time], dim=0)
        e2 = torch.cat([negative_embedding, embedding], dim=0)

        out2 = backbone(x2, time=t2, embedding=e2)
        out_u, out_c = out2.chunk(2, dim=0)

        # guided output for x0_hat computation
        out_guided = out_u + lam * (out_c - out_u)

        # If no scheduler passed, return model output(s) (caller does stepping)
        if inference_scheduler is None:
            return (out_guided, out_u) if kwargs.get("return_uncond", False) else out_guided

        # Scheduler must implement CFG++ stepping
        if not hasattr(inference_scheduler, "step_cfgpp"):
            raise ValueError(
                "CFG++ requires a CFG++-compatible scheduler (e.g., DDPMSchedulerPlusPlus, FlowMatchEulerDiscretePlusPlus) "
                "that implements step_cfgpp. Got scheduler type: {}. "
                "Non-CFG++ schedulers cannot be used with CFGPlusPlusGuidanceEngine.".format(type(inference_scheduler).__name__)
            )

        # IMPORTANT: timestep must be a scalar in the scheduler’s expected format
        # - For DDPM: int index
        # - For FlowMatch Euler: float timestep from scheduler.timesteps
        if isinstance(time, torch.Tensor):
            t_scalar = time[0]
        else:
            t_scalar = time

        # If it’s a 0-d tensor, convert cleanly
        if isinstance(t_scalar, torch.Tensor) and t_scalar.numel() == 1:
            # keep dtype as-is; scheduler may expect float vs int
            t_scalar = t_scalar.item()

        if step:
            return inference_scheduler.step_cfgpp(out_guided, out_u, t_scalar, x).prev_sample
        else:
            return out_guided
        

class DDIMInversionPlusPlus(DDIMInversionEditEngine):
    """
    DDIM inversion + editing engine that performs CFG++-compatible inversion.

    Requirements / assumptions:
      - `self.guidance_engine` is CFG++-capable and supports:
            apply_guidance(..., step=False, return_uncond=True) -> (out_guided, out_uncond)
      - The *learned* null text embedding is available as:
            self.backbone.null_text_embedding
        (as you described: stored in the backbone's parent / shared base)
      - Inversion uses DDIMInversePlusPlus.step_cfgpp(...) to implement CFG++ inversion update.
      - Reverting (generation) uses whatever `inference_scheduler` you pass:
            - if it has step_cfgpp and you use CFG++ guidance, it will use CFG++ stepping
            - otherwise it will fall back to vanilla scheduler.step()

    Important:
      - `guidance_scale` is interpreted as CFG++ lambda (typically in [0,1]) by your CFG++ guidance engine.
        If you pass classic CFG scales (e.g., 3, 7.5, 15) and you clamp, you'll effectively get lambda=1.
    """

    def __init__(self, backbone=None, text_encoder=None, guidance_engine=None, inversion_scheduler_cls=None):
        super().__init__(backbone=backbone, text_encoder=text_encoder, guidance_engine=guidance_engine)
        # allow DI via config, default to DDIMInversePlusPlus if available
        self.inversion_scheduler_cls = inversion_scheduler_cls  # e.g., DDIMInversePlusPlus
        
        # Validate that guidance engine supports CFG++ if provided
        if self.guidance_engine is not None:
            # Check if guidance engine supports return_uncond parameter
            import inspect
            sig = inspect.signature(self.guidance_engine.apply_guidance)
            if 'return_uncond' not in sig.parameters:
                raise ValueError(
                    "DDIMInversionPlusPlus requires a CFG++-compatible guidance engine "
                    "(e.g., CFGPlusPlusGuidanceEngine) that supports return_uncond parameter. "
                    "Got guidance engine type: {}. Non-CFG++ guidance engines cannot be used with DDIMInversionPlusPlus.".format(
                        type(self.guidance_engine).__name__
                    )
                )

    def _get_learned_null_embedding(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Fetch and batch the learned null text embedding.

        Expected shapes:
          - [1, seq, dim]  -> expanded to [B, seq, dim]
          - [seq, dim]     -> unsqueezed and expanded to [B, seq, dim]
          - [B, seq, dim]  -> returned as-is if B matches
        """
        if not hasattr(self.backbone, "null_text_embedding"):
            raise AttributeError(
                "Backbone must expose `null_text_embedding` (learned null text embedding) for CFG++ inversion."
            )

        null_emb = self.backbone.null_text_embedding
        if not isinstance(null_emb, torch.Tensor):
            raise TypeError("backbone.null_text_embedding must be a torch.Tensor")

        null_emb = null_emb.to(device=device, dtype=dtype)

        if null_emb.dim() == 2:
            # [seq, dim] -> [1, seq, dim]
            null_emb = null_emb.unsqueeze(0)

        if null_emb.shape[0] == 1 and batch_size > 1:
            null_emb = null_emb.expand(batch_size, *null_emb.shape[1:]).contiguous()
        elif null_emb.shape[0] != batch_size:
            raise ValueError(
                f"null_text_embedding batch dim mismatch: got {null_emb.shape[0]} but expected {batch_size}. "
                "Provide [1, seq, dim] or [B, seq, dim]."
            )

        return null_emb

    def invert(
        self,
        start_latents,
        original_prompt="",
        edit_prompt="",
        guidance_scale=3,
        invert_steps=20,
        inference_steps=50,
        num_images_per_prompt=1,
        negative_prompt=None,
        return_intermediate_latents=False,
        inference_scheduler=None,
        verbose=False,
    ):
        device = next(self.backbone.parameters()).device
        if inference_scheduler is None:
            raise ValueError("inference_scheduler is required")

        # --- create inversion scheduler (CFG++ aware) ---
        inv_cls = self.inversion_scheduler_cls
        if inv_cls is None:
            raise ValueError(
                "DDIMInversionPlusPlus requires `inversion_scheduler_cls` to be a CFG++-compatible inversion scheduler "
                "(e.g., DDIMInversePlusPlus) that implements step_cfgpp. "
                "Non-CFG++ inversion schedulers cannot be used with DDIMInversionPlusPlus."
            )

        inversion_scheduler = inv_cls.from_pretrained(
            inference_scheduler.config.scheduler_name,
            subfolder="scheduler",
            prediction_type=inference_scheduler.config.prediction_type,
        )
        inversion_scheduler.config.prediction_type = inference_scheduler.config.prediction_type

        # --- encode prompts ---
        edit_prompt_embedding = self.text_encoder.get_text_embedding(
            edit_prompt, use_tensor=True, return_dict=True
        )["last_hidden_state"].repeat_interleave(num_images_per_prompt, 0)

        if negative_prompt:
            negative_prompt_embeddings = self.text_encoder.get_text_embedding(
                negative_prompt, use_tensor=True, return_dict=True
            )["last_hidden_state"].repeat_interleave(num_images_per_prompt, 0)
        else:
            negative_prompt_embeddings = None

        original_prompt_embeddings = self.text_encoder.get_text_embedding(
            original_prompt, use_tensor=True, return_dict=True
        )["last_hidden_state"].repeat_interleave(num_images_per_prompt, 0)

        # --- init latents ---
        latents = start_latents.to(device)
        intermediate_latents = [latents]

        latents_dtype = latents.dtype
        edit_prompt_embedding = edit_prompt_embedding.to(latents_dtype)
        if negative_prompt_embeddings is not None:
            negative_prompt_embeddings = negative_prompt_embeddings.to(latents_dtype)
        original_prompt_embeddings = original_prompt_embeddings.to(latents_dtype)

        # learned null embedding (batched)
        null_embeddings = self._get_learned_null_embedding(
            batch_size=latents.shape[0], dtype=latents_dtype, device=device
        )

        # set timesteps
        inversion_scheduler.set_timesteps(inference_steps, device=device)
        inference_scheduler.set_timesteps(inference_steps, device=device)

        # -------------------------
        # Inversion phase (CFG++ DDIM inversion)
        # -------------------------
        # We walk forward along inversion_scheduler.timesteps (0 -> larger),
        # using inversion_scheduler.step_cfgpp(...).
        # Use only the first `invert_steps` steps (must leave room for t_next).
        inv_steps = min(invert_steps, len(inversion_scheduler.timesteps) - 1)

        ts = []
        for i in tqdm(range(inv_steps), disable=not verbose, leave=False, desc="Inverting (CFG++ DDIM)"):
            t = inversion_scheduler.timesteps[i]
            ts.append(t)

            latent_model_input = latents
            if hasattr(inversion_scheduler, "scale_model_input"):
                latent_model_input = inversion_scheduler.scale_model_input(latent_model_input, t)

            time = torch.full((latents.shape[0],), int(t), dtype=torch.long, device=device)

            if self.guidance_engine is not None:
                # IMPORTANT: step=False so we get model outputs, not latents
                out = self.guidance_engine.apply_guidance(
                    latent_model_input,
                    time,
                    embedding=original_prompt_embeddings,
                    negative_embedding=null_embeddings,
                    guidance_scale=guidance_scale,
                    backbone=self.backbone,
                    inference_scheduler=None,
                    step=False,
                    return_uncond=True,
                )
                if not (isinstance(out, (tuple, list)) and len(out) == 2):
                    raise ValueError(
                        "CFG++ inversion requires guidance_engine.apply_guidance(..., return_uncond=True, step=False) "
                        "to return (out_guided, out_uncond)."
                    )
                out_guided, out_uncond = out

                # CFG++ inversion update
                latents = inversion_scheduler.step_cfgpp(out_guided, out_uncond, t, latents).prev_sample
            else:
                # fallback: unconditional DDIM inversion (no CFG++)
                noise_pred = self.backbone(latent_model_input, time=time, embedding=original_prompt_embeddings)
                latents = inversion_scheduler.step(noise_pred, t, latents).prev_sample

            intermediate_latents.append(latents)

        # -------------------------
        # Editing / Reverting phase (normal generation)
        # -------------------------
        edit_latents = latents.clone()
        latent_model_input = edit_latents
        intermediate_edit_latents = []

        for t in tqdm(ts[::-1], disable=not verbose, leave=False, desc="Reverting with edit"):
            if hasattr(inference_scheduler, "scale_model_input"):
                latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), int(t), dtype=torch.long, device=device)

            if self.guidance_engine is not None:
                # If scheduler supports CFG++ stepping, your guidance engine can step internally.
                # Otherwise it will just return model output and we'll call scheduler.step below.
                latents_or_pred = self.guidance_engine.apply_guidance(
                    latent_model_input,
                    time,
                    embedding=edit_prompt_embedding,
                    negative_embedding=negative_prompt_embeddings if negative_prompt_embeddings is not None else null_embeddings,
                    guidance_scale=guidance_scale,
                    inference_scheduler=inference_scheduler,
                    backbone=self.backbone,
                    step=True,
                )

                # If guidance stepped, it returns latents. If not, it returns a prediction tensor.
                if isinstance(latents_or_pred, torch.Tensor) and latents_or_pred.shape == latents.shape:
                    latents = latents_or_pred
                else:
                    latents = inference_scheduler.step(latents_or_pred, int(t), latents).prev_sample
            else:
                pred = self.backbone(latent_model_input, time=time, embedding=edit_prompt_embedding)
                latents = inference_scheduler.step(pred, int(t), latents).prev_sample

            intermediate_edit_latents.append(latents)
            latent_model_input = latents

        intermediate_edit_latents = torch.stack(intermediate_edit_latents, dim=0)
        intermediate_latents = torch.stack(intermediate_latents, dim=0)

        if return_intermediate_latents:
            return intermediate_edit_latents[-1], intermediate_latents, intermediate_edit_latents
        return intermediate_edit_latents[-1]


