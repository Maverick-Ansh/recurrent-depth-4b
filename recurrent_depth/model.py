"""The latent recurrent-depth language model, Sec. 3 of arXiv:2502.05171.

Macroscopic design, Sec. 3.1 (verbatim):

    "the prelude P, which embeds the input data into a latent space using
     multiple transformer layers, then the core recurrent block R, which is the
     central unit of recurrent computation modifying states s in R^{n x h}, and
     finally the coda C, which un-embeds from latent space using several layers
     and also contains the prediction head of the model."

    "Given a number of recurrent iterations r, and a sequence of input tokens
     x in V^n these groups are used in the following way to produce output
     probabilities p in R^{n x |V|}

        e  = P(x)
        s0 ~ N(0, sigma^2 I_{n.h})
        si = R(e, s_{i-1})  for i in {1, ..., r}
        p  = C(sr)"

Sec. 3.2, the core block (verbatim):

    "Our core recurrent block R starts with an adapter matrix A : R^{2h} -> R^h
     mapping the concatenation of si and e into the hidden dimension h
     (Bansal et al., 2022). While re-incorporation of initial embedding features
     via addition rather than concatenation works equally well for smaller
     models, we find that concatenation works best at scale. This is then fed
     into lR transformer layers. At the end of the core block the output is
     again rescaled with an RMSNorm nc."

    "The coda contains lC layers, normalization by nc, and projection into the
     vocabulary using tied embeddings E^T."

Sec. 3.3, truncated backpropagation (verbatim):

    "To keep computation and memory low at train time, we backpropagate through
     only the last k iterations of the recurrent unit. This enables us to train
     with the heavy-tailed Poisson distribution Lambda, as maximum activation
     memory and backward compute is now independent of r. We fix k = 8 in our
     main experiments. [...] Note that the prelude block still receives gradient
     updates in every step, as its output e is injected in every step."

That last sentence is why `e` is computed OUTSIDE the no-grad region below: the
*state* chain is truncated to the final k steps, but the *injection* of e into
each of those k steps carries gradient into the prelude k times per forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RecurrentDepthConfig
from .layers import RMSNorm, SandwichBlock, build_rope_cache
from .init import takase_init_, sample_s0


class RecurrentDepthLM(nn.Module):
    def __init__(self, cfg: RecurrentDepthConfig, init_weights: bool = True):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        self.embed = nn.Embedding(cfg.vocab_size, h)
        # Sec. 3.2: "the prelude block first embeds input tokens x as gamma*E(x)"
        # Sec. 4.1: "The output of the embedding layer is scaled by sqrt(h)."
        self.gamma = h ** 0.5

        def blk():
            return SandwichBlock(h, cfg.n_heads, cfg.mlp_inner, cfg.rope_base,
                                 cfg.norm_eps, cfg.norm_style, cfg.norm_affine)

        self.prelude = nn.ModuleList([blk() for _ in range(cfg.l_prelude)])
        self.core = nn.ModuleList([blk() for _ in range(cfg.l_core)])
        self.coda = nn.ModuleList([blk() for _ in range(cfg.l_coda)])

        # adapter A : R^{2h} -> R^h applied to concat(s_i, e)
        self.adapter = nn.Linear(2 * h, h, bias=False) if cfg.injection == "concat" else None

        self.core_norm = RMSNorm(h, cfg.norm_eps, cfg.norm_affine)  # n_c at end of core
        self.coda_norm = RMSNorm(h, cfg.norm_eps, cfg.norm_affine)  # n_c in the coda
        self.lm_head = nn.Linear(h, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Sec. 4.1: l = lP + rbar*lR + lC, "the number of effective layers"
        eff = cfg.l_prelude + cfg.mean_recurrence * cfg.l_core + cfg.l_coda
        self.effective_layers = int(eff)
        if init_weights:
            takase_init_(self, h, self.effective_layers)
        self._rope = None

    # ------------------------------------------------------------------ utils
    def rope(self, device, dtype):
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype:
            self._rope = build_rope_cache(self.cfg.hidden // self.cfg.n_heads,
                                          self.cfg.max_seq, self.cfg.rope_base, device, dtype)
        return self._rope

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def materialized_params(self, r: int) -> int:
        """Sec. 5.1 / Fig. 1 second x-axis: parameters actually *executed* at depth r.

        The core block's parameters are re-used r times, so the executed compute
        (and the depth of the computation graph) matches a fixed-depth model of
        this many parameters. This is the axis on which the paper claims a 3.5B
        model reaches "a computation load equivalent to 50 billion parameters".
        """
        core = sum(p.numel() for p in self.core.parameters())
        if self.adapter is not None:
            core += self.adapter.weight.numel()
        return self.n_params() - core + r * core

    # --------------------------------------------------------------- the model
    def prelude_forward(self, idx, kv=None):
        """e = P(x).  Sec. 3.1."""
        x = self.embed(idx) * self.gamma
        rope = self.rope(x.device, x.dtype)
        new = []
        for i, blk in enumerate(self.prelude):
            x, c = blk(x, rope, None if kv is None else kv[i])
            new.append(c)
        return x, new

    def inject(self, s, e):
        """The adapter A of Sec. 3.2."""
        mode = self.cfg.injection
        if mode == "concat":
            return self.adapter(torch.cat([s, e], dim=-1))
        if mode == "add":
            return s + e            # "re-incorporation ... via addition"
        return s                    # "none": no per-step injection (ablation)

    def core_forward(self, s, e, kv=None):
        """s_i = R(e, s_{i-1}).  Adapter -> lR layers -> RMSNorm n_c."""
        x = self.inject(s, e)
        rope = self.rope(x.device, x.dtype)
        new = []
        for i, blk in enumerate(self.core):
            x, c = blk(x, rope, None if kv is None else kv[i])
            new.append(c)
        return self.core_norm(x), new

    def coda_forward(self, s, kv=None):
        """p = C(s_r).  lC layers -> n_c -> tied unembedding."""
        x = s
        rope = self.rope(x.device, x.dtype)
        new = []
        for i, blk in enumerate(self.coda):
            x, c = blk(x, rope, None if kv is None else kv[i])
            new.append(c)
        return self.lm_head(self.coda_norm(x)), new

    def init_state(self, e, generator=None):
        return sample_s0(e.shape, e.device, e.dtype,
                         deterministic=not self.cfg.random_s0, generator=generator)

    # -------------------------------------------------------------- forward API
    def forward(self, idx, r: int, targets=None, k: int | None = None, s0=None,
                generator=None, return_states: bool = False, ignore_index: int = -100):
        """One unrolled forward pass at recurrence depth `r`.

        Truncated backprop (Sec. 3.3): the first r-k steps run under no_grad and
        the state is detached; the last k steps carry gradient. Pass k >= r for
        full backpropagation-through-unrolling (our C4 control arm).
        """
        cfg = self.cfg
        k = cfg.backprop_depth if k is None else k
        k = max(1, min(k, r))

        e, _ = self.prelude_forward(idx)
        s = self.init_state(e, generator) if s0 is None else s0
        states = [s] if return_states else None

        n_nograd = r - k
        if n_nograd > 0:
            with torch.no_grad():
                for _ in range(n_nograd):
                    s, _ = self.core_forward(s, e)
                    if return_states:
                        states.append(s)
            s = s.detach()
        for _ in range(k):
            s, _ = self.core_forward(s, e)
            if return_states:
                states.append(s)

        logits, _ = self.coda_forward(s)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=ignore_index)
        out = {"logits": logits, "loss": loss, "state": s}
        if return_states:
            out["states"] = torch.stack(states)      # (r+1, B, S, h)
        return out

    @torch.no_grad()
    def trajectory(self, idx, r: int, s0=None, generator=None):
        """Full latent trajectory {s_i}_{i=0..r} for the Sec. 7 analyses."""
        return self.forward(idx, r, s0=s0, generator=generator, return_states=True)["states"]
