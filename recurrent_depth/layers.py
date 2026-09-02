"""Transformer sub-layers for the recurrent-depth architecture.

Implements the "Microscopic Design" of Section 3.2 of

    Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning:
    A Recurrent Depth Approach", arXiv:2502.05171v2.

Paper, Sec. 3.2 (verbatim):

    "Within each group, we broadly follow standard transformer layer design.
     Each block contains multiple layers, and each layer contains a standard,
     causal self-attention block using RoPE (Su et al., 2021) with a base of
     50000, and a gated SiLU MLP (Shazeer, 2020). We use RMSNorm (Zhang and
     Sennrich, 2019) as our normalization function. The model has learnable
     biases on queries and keys, and nowhere else."

    "To stabilize the recurrence, we order all layers in the following
     'sandwich' format, using norm layers n_i [...]:

        x_hat_l = n2( x_{l-1} + Attn(n1(x_{l-1})) )
        x_l     = n4( x_hat_l + MLP(n3(x_hat_l)) )"

    Footnote 2: "Note also that technically n3 is superfluous, but we report
    here the exact norm setup with which we trained the final model."

We keep n3 for fidelity. `SandwichBlock(norm_style="pre")` reproduces the
"Bad Run 2" pre-norm configuration of Sec. 4.3, which is one of our ablations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learnable scale.

    Sec. 4.3 reports that *parameter-free* RMSNorm was part of the failed
    "Bad Run 1" configuration, so the final model's norms are parameterised.
    `elementwise_affine=False` is available to reproduce that failure mode.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(dtype)
        return x if self.weight is None else x * self.weight


def build_rope_cache(head_dim: int, max_seq: int, base: float, device, dtype):
    """RoPE (Su et al. 2021). Paper uses base 50000 for the from-scratch model."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # (S, hd/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, S, hd). cos/sin: (S, hd/2). Interleaved-pair convention."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Causal MHA with RoPE and learnable q/k biases only (Sec. 3.2)."""

    def __init__(self, dim: int, n_heads: int, rope_base: float = 50000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim, self.rope_base = n_heads, dim // n_heads, rope_base
        # "learnable biases on queries and keys, and nowhere else"
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)   # out-projection: sigma_out init

    def forward(self, x, rope, kv_cache=None):
        B, S, D = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        cos, sin = rope
        # When decoding with a cache the new tokens sit at absolute offset len(cache).
        off = 0 if kv_cache is None else kv_cache[0].shape[2]
        q = apply_rope(q, cos[off:off + S], sin[off:off + S])
        k = apply_rope(k, cos[off:off + S], sin[off:off + S])

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        is_causal = kv_cache is None or S > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).reshape(B, S, D)
        return self.o_proj(y), new_cache


class GatedSiLUMLP(nn.Module):
    """Gated SiLU MLP / SwiGLU (Shazeer 2020), Sec. 3.2."""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.up_proj = nn.Linear(dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)  # out-projection

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SandwichBlock(nn.Module):
    """One transformer layer in the paper's sandwich-norm ordering (Sec. 3.2).

        x_hat = n2( x + Attn(n1(x)) )
        x_out = n4( x_hat + MLP(n3(x_hat)) )

    Sec. 4.3 is explicit that this ordering is what made the large run work:
    "In a third, and final run ('Main', blue), we fix this issue by reverting
     back to the sandwich block format". `norm_style="pre"` gives the standard
     pre-norm block used by the failed "Bad Run 2", kept here as an ablation.
    """

    def __init__(self, dim, n_heads, inner, rope_base=50000.0, norm_eps=1e-6,
                 norm_style="sandwich", norm_affine=True):
        super().__init__()
        assert norm_style in ("sandwich", "pre")
        self.norm_style = norm_style
        self.attn = CausalSelfAttention(dim, n_heads, rope_base)
        self.mlp = GatedSiLUMLP(dim, inner)
        nrm = lambda: RMSNorm(dim, norm_eps, norm_affine)
        self.n1, self.n3 = nrm(), nrm()
        if norm_style == "sandwich":
            self.n2, self.n4 = nrm(), nrm()

    def forward(self, x, rope, kv_cache=None):
        a, new_cache = self.attn(self.n1(x), rope, kv_cache)
        if self.norm_style == "sandwich":
            x = self.n2(x + a)
            x = self.n4(x + self.mlp(self.n3(x)))
        else:                                   # pre-norm ("Bad Run 2")
            x = x + a
            x = x + self.mlp(self.n3(x))
        return x, new_cache
