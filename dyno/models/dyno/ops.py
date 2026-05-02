import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

try:
    import xformers.ops as xops
except ModuleNotFoundError:
    xops = None


def attend(q, k, v, attn_bias=None):
    B, Tq, H, Dh = q.shape
    if xops is not None:
        return xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias).reshape(B, Tq, H * Dh)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    return out.transpose(1, 2).reshape(B, Tq, H * Dh)


class SwiGLU(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int):
        super().__init__()
        self.w12 = nn.Linear(model_dim, 2 * ffn_dim)
        self.w3 = nn.Linear(ffn_dim, model_dim)

    def forward(self, x):
        x, gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * x)


def make_ffn(model_dim: int, ffn_dim: int, dropout: float) -> nn.Module:
    swiglu = (
        xops.SwiGLU(in_features=model_dim, hidden_features=ffn_dim, out_features=model_dim)
        if xops is not None
        else SwiGLU(model_dim, ffn_dim)
    )
    return nn.Sequential(
        swiglu,
        nn.Dropout(dropout),
    )


def make_norm(dim: int) -> nn.Module:
    return LlamaRMSNorm(dim)
