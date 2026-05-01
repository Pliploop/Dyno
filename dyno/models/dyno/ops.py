import torch.nn as nn
import xformers.ops as xops
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def attend(q, k, v, attn_bias=None):
    B, Tq, H, Dh = q.shape
    return xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias).reshape(B, Tq, H * Dh)


def make_ffn(model_dim: int, ffn_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        xops.SwiGLU(in_features=model_dim, hidden_features=ffn_dim, out_features=model_dim),
        nn.Dropout(dropout),
    )


def make_norm(dim: int) -> nn.Module:
    return LlamaRMSNorm(dim)
