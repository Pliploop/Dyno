import torch
import torch.nn as nn

from .rope import RotaryEmbedding, apply_rotary_emb
from .ops import attend, make_ffn, make_norm

CONDITIONING_TYPES = ("cross_attention", "film", "adaLN_zero", "prefix")


class FiLMModulation(nn.Module):
    def __init__(self, model_dim: int, context_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(context_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, 2 * model_dim))

    def forward(self, x, c):
        gamma, beta = self.mlp(c).chunk(2, dim=-1)
        return (1.0 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)


class AdaLNZeroModulation(nn.Module):
    def __init__(self, model_dim: int, context_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.proj = nn.Linear(context_dim, 6 * model_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c):
        return self.proj(self.silu(c)).chunk(6, dim=-1)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        conditioning_dim: int | None = None,
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.cross_norm = make_norm(model_dim)
        self.cross_q = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_k = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_v = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_out = nn.Linear(model_dim, model_dim)

        self.self_norm = make_norm(model_dim)
        self.self_q = nn.Linear(model_dim, model_dim, bias=False)
        self.self_k = nn.Linear(model_dim, model_dim, bias=False)
        self.self_v = nn.Linear(model_dim, model_dim, bias=False)
        self.self_out = nn.Linear(model_dim, model_dim)

        self.ffn_norm = make_norm(model_dim)
        self.ffn = make_ffn(model_dim, ffn_dim, dropout)

    def forward(self, x, cond, cos, sin):
        B, T, _ = x.shape
        C = cond.shape[1]

        residual = x
        x_n = self.cross_norm(x)
        q = self.cross_q(x_n).view(B, T, self.num_heads, self.head_dim)
        k = self.cross_k(cond).view(B, C, self.num_heads, self.head_dim)
        v = self.cross_v(cond).view(B, C, self.num_heads, self.head_dim)
        x = residual + self.cross_out(attend(q, k, v))

        residual = x
        x_n = self.self_norm(x)
        q = self.self_q(x_n).view(B, T, self.num_heads, self.head_dim)
        k = self.self_k(x_n).view(B, T, self.num_heads, self.head_dim)
        v = self.self_v(x_n).view(B, T, self.num_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, cos, sin)
        x = residual + self.self_out(attend(q, k, v))

        return x + self.ffn(self.ffn_norm(x))


class FiLMBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        conditioning_dim: int | None = None,
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        context_dim = conditioning_dim or 2 * model_dim

        self.norm1 = make_norm(model_dim)
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(model_dim, model_dim, bias=False)
        self.v = nn.Linear(model_dim, model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim)
        self.film_attn = FiLMModulation(model_dim, context_dim)

        self.norm2 = make_norm(model_dim)
        self.ffn = make_ffn(model_dim, ffn_dim, dropout)
        self.film_ffn = FiLMModulation(model_dim, context_dim)

    def forward(self, x, cond, cos, sin):
        B, T, _ = x.shape
        residual = x
        x_n = self.norm1(x)
        q = self.q(x_n).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x_n).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x_n).view(B, T, self.num_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, cos, sin)
        x = residual + self.film_attn(self.out(attend(q, k, v)), cond)
        return x + self.film_ffn(self.ffn(self.norm2(x)), cond)


class AdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        conditioning_dim: int | None = None,
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.temporal_only = conditioning_dim == model_dim

        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(model_dim, model_dim, bias=False)
        self.v = nn.Linear(model_dim, model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim)

        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ffn = make_ffn(model_dim, ffn_dim, dropout)
        self.z_modulation = AdaLNZeroModulation(model_dim, context_dim=model_dim)
        self.content_modulation = None if self.temporal_only else AdaLNZeroModulation(model_dim, context_dim=model_dim)

    def forward(self, x, cond, cos, sin):
        B, T, _ = x.shape
        if self.temporal_only:
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = self.z_modulation(cond)
        else:
            z_cond, content_cond = cond.chunk(2, dim=-1)
            z_mod = self.z_modulation(z_cond)
            content_mod = self.content_modulation(content_cond)
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = (
                z_part + content_part for z_part, content_part in zip(z_mod, content_mod)
            )

        x_n = (1.0 + scale_sa.unsqueeze(1)) * self.norm1(x) + shift_sa.unsqueeze(1)
        q = self.q(x_n).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x_n).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x_n).view(B, T, self.num_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, cos, sin)
        x = x + gate_sa.unsqueeze(1) * self.out(attend(q, k, v))

        x_n = (1.0 + scale_ffn.unsqueeze(1)) * self.norm2(x) + shift_ffn.unsqueeze(1)
        return x + gate_ffn.unsqueeze(1) * self.ffn(x_n)


class PrefixBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        conditioning_dim: int | None = None,
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.norm1 = make_norm(model_dim)
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(model_dim, model_dim, bias=False)
        self.v = nn.Linear(model_dim, model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim)

        self.norm2 = make_norm(model_dim)
        self.ffn = make_ffn(model_dim, ffn_dim, dropout)

    def forward(self, x, cond, cos, sin):
        B, T, D = x.shape
        C = cond.shape[1]

        combined_n = self.norm1(torch.cat([cond, x], dim=1))
        q = self.q(combined_n).view(B, C + T, self.num_heads, self.head_dim)
        k = self.k(combined_n).view(B, C + T, self.num_heads, self.head_dim)
        v = self.v(combined_n).view(B, C + T, self.num_heads, self.head_dim)

        q_rope, k_rope = apply_rotary_emb(q[:, C:], k[:, C:], cos, sin)
        q = torch.cat([q[:, :C], q_rope], dim=1)
        k = torch.cat([k[:, :C], k_rope], dim=1)

        x = x + self.out(attend(q, k, v)[:, C:])
        return x + self.ffn(self.norm2(x))


_BLOCK_CLS = {
    "cross_attention": CrossAttentionBlock,
    "film":            FiLMBlock,
    "adaLN_zero":      AdaLNZeroBlock,
    "prefix":          PrefixBlock,
}


class DynoVelocityPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
        max_frames: int = 1024,
        rope_theta: float = 10000.0,
        conditioning_type: str = "adaLN_zero",
        temporal_only_conditioning: bool = False,
    ):
        if conditioning_type not in _BLOCK_CLS:
            raise ValueError(f"conditioning_type must be one of {list(_BLOCK_CLS)}, got {conditioning_type!r}")
        super().__init__()
        assert (model_dim // num_heads) % 2 == 0
        self.conditioning_type = conditioning_type
        self.temporal_only_conditioning = temporal_only_conditioning

        self.position_queries = nn.Parameter(torch.randn(max_frames, model_dim) * 0.02)
        self.z_tau_proj = nn.Linear(latent_dim, model_dim)
        self.content_proj = None if temporal_only_conditioning else nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=model_dim // num_heads, max_seq_len=max_frames, theta=rope_theta)
        conditioning_dim = model_dim if temporal_only_conditioning else 2 * model_dim
        self.blocks = nn.ModuleList(
            [
                _BLOCK_CLS[conditioning_type](
                    model_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    conditioning_dim=conditioning_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = make_norm(model_dim)
        self.out_proj = nn.Linear(model_dim, input_dim)

    def _build_cond(self, z_tau, content):
        z = self.z_tau_proj(z_tau)
        if self.temporal_only_conditioning:
            if self.conditioning_type in ("cross_attention", "prefix"):
                return z.unsqueeze(1)
            return z
        c = self.content_proj(content)
        if self.conditioning_type in ("cross_attention", "prefix"):
            return torch.stack([z, c], dim=1)
        return torch.cat([z, c], dim=-1)

    def forward(self, z_tau, content, num_frames):
        B = z_tau.shape[0]
        cond = self._build_cond(z_tau, content)
        queries = self.dropout(self.position_queries[:num_frames].unsqueeze(0).expand(B, -1, -1))
        cos, sin = self.rope(seq_len=num_frames)
        for block in self.blocks:
            queries = block(queries, cond, cos, sin)
        return self.out_proj(self.norm(queries))
