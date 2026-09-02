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


class RMSNorm(nn.Module):                                                                           # +-- RMSNORM, AND WHY ITS SCALE IS LEARNABLE --------------------
    """Root-mean-square norm with a learnable scale.

    Sec. 4.3 reports that *parameter-free* RMSNorm was part of the failed
    "Bad Run 1" configuration, so the final model's norms are parameterised.
    `elementwise_affine=False` is available to reproduce that failure mode.
    """
                                                                                                    # | This divides each token's hidden vector by its own root-mean-
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):               # | square, so every vector leaves with length 1 regardless of
        super().__init__()                                                                          # | what came in, then multiplies by a learned per-channel scale.
        self.eps = eps                                                                              # | The mean is computed in float32 even when the activations are
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None                 # | float16, because squaring float16 numbers near the top of
                                                                                                    # | their range overflows and near the bottom flushes to zero, and
    def forward(self, x: torch.Tensor) -> torch.Tensor:                                             # | one bad row would poison the whole vector.
        dtype = x.dtype                                                                             # | elementwise_affine=False removes the learned scale entirely.
        x = x.float()                                                                               # | That option is not a convenience: parameter-free norms were
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)                             # | part of the configuration that made the paper's first large
        x = x.to(dtype)                                                                             # | run collapse, so it is kept so that failure can be reproduced
        return x if self.weight is None else x * self.weight                                        # | on purpose rather than argued about.


def build_rope_cache(head_dim: int, max_seq: int, base: float, device, dtype):                      # +-- ROTARY POSITION EMBEDDING, PRECOMPUTED ---------------------
    """RoPE (Su et al. 2021). Paper uses base 50000 for the from-scratch model."""                  # | Position is encoded by rotating pairs of channels in a query
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))     # | or key vector by an angle proportional to the token's
    t = torch.arange(max_seq, device=device).float()                                                # | position. Channel pair j rotates at frequency
    freqs = torch.outer(t, inv_freq)                      # (S, hd/2)                               # | base^(-2j/head_dim), so early pairs turn fast and late pairs
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)                                   # | turn slowly, and the dot product between two rotated vectors
                                                                                                    # | ends up depending on the gap between their positions rather
                                                                                                    # | than their absolute values. build_rope_cache tabulates every
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:              # | angle once up to max_seq. apply_rope splits the vector into
    """x: (B, H, S, hd). cos/sin: (S, hd/2). Interleaved-pair convention."""                        # | even and odd channels, treats each pair as a point in a plane,
    x1, x2 = x[..., 0::2], x[..., 1::2]                                                             # | and rotates it. The offset argument at the call site matters
    c, s = cos[None, None], sin[None, None]                                                         # | during decoding: a cached token sits at absolute position
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)                                   # | len(cache), not at zero, and reading the table from the wrong
    return out.flatten(-2)                                                                          # | row silently misplaces every token.


class CausalSelfAttention(nn.Module):                                                               # +-- CAUSAL ATTENTION WITH A KEY/VALUE CACHE --------------------
    """Causal MHA with RoPE and learnable q/k biases only (Sec. 3.2)."""                            # | Queries, keys and values are projected, split into heads,
                                                                                                    # | rotated by position, then attention is taken with each
    def __init__(self, dim: int, n_heads: int, rope_base: float = 50000.0):                         # | position allowed to see only itself and everything before it.
        super().__init__()                                                                          # | Only the query and key projections carry a bias, which is the
        assert dim % n_heads == 0                                                                   # | paper's own choice and is why v_proj and o_proj are built
        self.n_heads, self.head_dim, self.rope_base = n_heads, dim // n_heads, rope_base            # | without one. o_proj is named so that the initialiser can find
        # "learnable biases on queries and keys, and nowhere else"                                  # | it: out projections are deliberately started much smaller than
        self.q_proj = nn.Linear(dim, dim, bias=True)                                                # | other weights, because in a looped model the same residual
        self.k_proj = nn.Linear(dim, dim, bias=True)                                                # | addition happens r times and a normal-sized start compounds.
        self.v_proj = nn.Linear(dim, dim, bias=False)                                               # | The cache holds keys and values already computed for earlier
        self.o_proj = nn.Linear(dim, dim, bias=False)   # out-projection: sigma_out init            # | tokens. When it is present the new tokens are appended to it
                                                                                                    # | and the causal mask is switched off, because a single new
    def forward(self, x, rope, kv_cache=None):                                                      # | token attending over a cache of strictly older tokens is
        B, S, D = x.shape                                                                           # | already causal, and asking for a triangular mask over a one-
        H, hd = self.n_heads, self.head_dim                                                         # | row query would mask out everything.
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


class GatedSiLUMLP(nn.Module):                                                                      # +-- THE SANDWICH BLOCK, AND ITS PRE-NORM TWIN ------------------
    """Gated SiLU MLP / SwiGLU (Shazeer 2020), Sec. 3.2."""                                         # | The MLP multiplies two separate projections of the input, one
                                                                                                    # | passed through SiLU as a gate, then projects back down;
    def __init__(self, dim: int, inner: int):                                                       # | down_proj is the second out projection the initialiser scales
        super().__init__()                                                                          # | down. The block itself is where the paper's stability argument
        self.gate_proj = nn.Linear(dim, inner, bias=False)                                          # | lives. A standard pre-norm block normalises before each
        self.up_proj = nn.Linear(dim, inner, bias=False)                                            # | sublayer and lets the residual stream pass through untouched,
        self.down_proj = nn.Linear(inner, dim, bias=False)  # out-projection                        # | so its magnitude grows with every layer. The sandwich ordering
                                                                                                    # | normalises again after each residual addition, so the stream
    def forward(self, x):                                                                           # | is renormalised four times per layer and cannot grow across
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))                          # | turns of the loop. At a small scale both work. At the paper's
                                                                                                    # | scale only the sandwich form trained, and reverting to it is
                                                                                                    # | what separated their successful run from the second failed
class SandwichBlock(nn.Module):                                                                     # | one. n3 is mathematically redundant given n2 immediately
    """One transformer layer in the paper's sandwich-norm ordering (Sec. 3.2).

        x_hat = n2( x + Attn(n1(x)) )
        x_out = n4( x_hat + MLP(n3(x_hat)) )

    Sec. 4.3 is explicit that this ordering is what made the large run work:
    "In a third, and final run ('Main', blue), we fix this issue by reverting
     back to the sandwich block format". `norm_style="pre"` gives the standard
     pre-norm block used by the failed "Bad Run 2", kept here as an ablation.
    """
                                                                                                    # | before it, but it is kept because it is what the released
    def __init__(self, dim, n_heads, inner, rope_base=50000.0, norm_eps=1e-6,                       # | model was actually trained with, and removing it would make
                 norm_style="sandwich", norm_affine=True):                                          # | this a different architecture. The pre-norm path is retained
        super().__init__()                                                                          # | as an ablation arm, not as an option.
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
