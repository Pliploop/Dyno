import torch
import torch.nn as nn

from .rope import RotaryEmbedding, apply_rotary_emb
from .ops import attend, make_ffn, make_norm


class AggregatorBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.norm1 = make_norm(model_dim)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.norm2 = make_norm(model_dim)
        self.ffn = make_ffn(model_dim, ffn_dim, dropout)

    def forward(self, x, cos, sin, attn_bias=None):
        B, T, _ = x.shape
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, cos, sin)
        x = residual + self.out_proj(attend(q, k, v, attn_bias=attn_bias))
        return x + self.ffn(self.norm2(x))


class DynoAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert (model_dim // num_heads) % 2 == 0
        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=model_dim // num_heads, max_seq_len=max_seq_len + 1, theta=rope_theta)
        self.blocks = nn.ModuleList([AggregatorBlock(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.norm = make_norm(model_dim)

    def forward(self, x, mask=None):
        """
        x:    (B, T, input_dim)
        mask: (B, T) bool — True for valid frames, False for padding.
              When None, all positions are treated as valid.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # (B, T+1, D)
        x = self.dropout(x)
        cos, sin = self.rope(seq_len=T + 1)

        attn_bias = None
        if mask is not None:
            # Prepend True for CLS token (always valid)
            full_mask = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
            )  # (B, T+1)
            # Additive bias: (B, 1, 1, T+1) — broadcast over heads and query positions
            # Padding key positions get -inf → zero weight after softmax
            attn_bias = torch.zeros(B, 1, 1, T + 1, dtype=x.dtype, device=x.device)
            attn_bias.masked_fill_(~full_mask[:, None, None, :], float("-inf"))

        for block in self.blocks:
            x = block(x, cos, sin, attn_bias=attn_bias)
        return self.norm(x)[:, 0, :]
