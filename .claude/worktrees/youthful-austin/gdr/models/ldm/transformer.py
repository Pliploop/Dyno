

from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.transformers.transformer_flux2 import Flux2Attention, Flux2TransformerBlock, Flux2SingleTransformerBlock, Flux2Modulation, Flux2TimestepGuidanceEmbeddings, Flux2PosEmbed
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_dispatch import dispatch_attention_fn

from gdr.models.utils.base import BaseModule

import logging


def _get_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)



class TransformerBackbone(BaseModule):
    """
    Transformer backbone adapted from Flux2Transformer2DModel for 1D audio.
    Compatible with the UNet interface for seamless integration.
    
    This transformer handles:
    - Time conditioning via timesteps
    - Text embeddings via cross-attention
    - Classifier-free guidance (CFG) with negative embeddings
    - Embedding scale and mask probability for CFG
    - Native RoPE positional embeddings
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 6,
        num_single_layers: int = 6,
        num_attention_heads: int = 8,
        inner_dim: int = 768,
        cond_dim: int = 768,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (64,),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        use_classifier_free_guidance: bool = False,
        embedding_max_length: int = 256,
        ckpt_path: Optional[str] = None,
        freeze: bool = False,
        **kwargs
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (defaults to in_channels)
            num_layers: Number of dual stream transformer layers
            num_single_layers: Number of single stream transformer layers
            attention_head_dim: Dimension of each attention head
            num_attention_heads: Number of attention heads
            joint_attention_dim: Embedding dimension for joint text-audio attention
            timestep_guidance_channels: Channels for timestep/guidance embedding
            mlp_ratio: MLP ratio for feedforward layers
            axes_dims_rope: Dimensions for RoPE (for 1D, use [32] or similar)
            rope_theta: Theta parameter for RoPE
            eps: Epsilon for normalization
            use_classifier_free_guidance: Whether to use CFG
            embedding_max_length: Maximum length of embeddings for CFG
            ckpt_path: Path to checkpoint for loading weights
            freeze: Whether to freeze model parameters
        """
        
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.use_classifier_free_guidance = use_classifier_free_guidance
        self.embedding_max_length = embedding_max_length
        self.use_time = kwargs.get("use_time", True)
        self.inner_dim = inner_dim
        attention_head_dim = self.inner_dim // num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads

        # 1. Sinusoidal positional embedding for RoPE on audio and text tokens (1D)
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim, bias=False
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(cond_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        # For 1D, output is just out_channels (no patch_size * patch_size)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)

        self.gradient_checkpointing = False
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    @staticmethod
    def _prep_ids(
        # NOTE: if we condition on other modalities later we will need to make an _id function that takes in the modality and returns the ids
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)
    
    @classmethod
    def from_config(cls, config):
        """Create instance from config dictionary."""
        return cls(**config)
    
    def forward(
        self,
        x: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch, in_channels, sequence_len)
            time: Timestep tensor of shape (batch,) or None
            embedding: Text embeddings of shape (batch, seq_len, joint_attention_dim)
            guidance: Guidance tensor for timestep embedding (optional)
            joint_attention_kwargs: Joint attention kwargs
        Returns:
            Output tensor of shape (batch, out_channels, sequence_len)
        """
        batch_size = x.shape[0]
        device = x.device

        # Handle time conditioning
        if not self.use_time:
            timestep = None
            guidance = None
        else:
            if time is None:
                timestep = torch.zeros((batch_size,), dtype=torch.long, device=device)
            else:
                timestep = time
            # If guidance is not provided, use zeros (no guidance)
            if guidance is None:
                guidance = torch.zeros((batch_size,), dtype=torch.long, device=device)

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        # Single forward pass
        output = self._forward_single(
            x, timestep, embedding, guidance, joint_attention_kwargs, attention_mask
        )
        
        return output
    
    def _forward_single(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        guidance: Optional[torch.Tensor],
        joint_attention_kwargs: Optional[Dict[str, Any]],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Single forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch, in_channels, sequence_len)
            timestep: Timestep tensor of shape (batch,)
            encoder_hidden_states: Text embeddings
            guidance: Guidance tensor for timestep embedding
            joint_attention_kwargs: Joint attention kwargs  
        Returns:
            Output tensor of shape (batch, out_channels, sequence_len)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        device = x.device
        dtype = x.dtype

        # Reshape latents: (batch, in_channels, sequence_len) -> (batch, sequence_len, in_channels)
        hidden_states = x.transpose(1, 2)  # (batch, sequence_len, in_channels)

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        num_txt_tokens = encoder_hidden_states.shape[1]
        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        # img_ids and txt_ids are of shape (batch, seq_len)
        img_ids = self._prep_ids(hidden_states)
        txt_ids = self._prep_ids(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]



        if is_torch_npu_available():
            freqs_cos_image, freqs_sin_image = self.pos_embed(img_ids.cpu())
            image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
            freqs_cos_text, freqs_sin_text = self.pos_embed(txt_ids.cpu())
            text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
        else:
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 4. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_params_img=double_stream_mod_img,
                    temb_mod_params_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 5. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod_params=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Reshape back: (batch, sequence_len, out_channels) -> (batch, out_channels, sequence_len)
        output = output.transpose(1, 2)  # (batch, out_channels, sequence_len)

        return output
