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


class RecurrentDepthLM(nn.Module):                                                                  # +-- THREE STACKS, ONE OF THEM REUSED ---------------------------
    def __init__(self, cfg: RecurrentDepthConfig, init_weights: bool = True):                       # | The model holds three stacks of transformer layers and uses
        super().__init__()                                                                          # | them differently. The prelude runs once and turns token ids
        self.cfg = cfg                                                                              # | into a latent vector e. The core is the only stack that runs
        h = cfg.hidden                                                                              # | more than once: the same weights are applied again and again,
                                                                                                    # | and how many times is decided when the model is called, not
        self.embed = nn.Embedding(cfg.vocab_size, h)                                                # | when it is built. The coda runs once at the end and produces
        # Sec. 3.2: "the prelude block first embeds input tokens x as gamma*E(x)"                   # | vocabulary logits. Because the core is one list of layers
        # Sec. 4.1: "The output of the embedding layer is scaled by sqrt(h)."                       # | reused r times rather than r separate stacks, buying more
        self.gamma = h ** 0.5                                                                       # | depth at test time costs no extra parameters. gamma is the
                                                                                                    # | square root of the hidden size. The embedding table is
        def blk():                                                                                  # | initialised with variance 2/(5h), so scaling its output by
            return SandwichBlock(h, cfg.n_heads, cfg.mlp_inner, cfg.rope_base,                      # | sqrt(h) lands e at variance 2/5, which is exactly the variance
                                 cfg.norm_eps, cfg.norm_style, cfg.norm_affine)                     # | the random start state is drawn at; the two are placed on the
                                                                                                    # | same scale on purpose so the adapter sees comparable
        self.prelude = nn.ModuleList([blk() for _ in range(cfg.l_prelude)])                         # | magnitudes on both halves of its input. The adapter is the
        self.core = nn.ModuleList([blk() for _ in range(cfg.l_core)])                               # | only genuinely new matrix the recurrence needs.
        self.coda = nn.ModuleList([blk() for _ in range(cfg.l_coda)])                               # | effective_layers is what the initialiser wants: out
                                                                                                    # | projections are scaled down by how many layers the signal
        # adapter A : R^{2h} -> R^h applied to concat(s_i, e)                                       # | actually passes through, and for a looped core that is lP +
        self.adapter = nn.Linear(2 * h, h, bias=False) if cfg.injection == "concat" else None       # | rbar*lR + lC, not the number of distinct layers, which is 8.

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

    # ------------------------------------------------------------------ utils                      # +-- PARAMETERS THAT EXIST VS PARAMETERS THAT RUN ---------------
    def rope(self, device, dtype):                                                                  # | rope builds the sine and cosine tables for rotary position
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype:    # | embedding once and rebuilds them only if the device or dtype
            self._rope = build_rope_cache(self.cfg.hidden // self.cfg.n_heads,                      # | changes, because every layer in all three stacks asks for the
                                          self.cfg.max_seq, self.cfg.rope_base, device, dtype)      # | same table. n_params counts weights that exist.
        return self._rope                                                                           # | materialized_params counts weights that run: the core's
                                                                                                    # | weights are executed r times, so at r=32 a model whose core
    def n_params(self, non_embedding: bool = False) -> int:                                         # | holds 1.5B parameters performs the arithmetic of a model
        n = sum(p.numel() for p in self.parameters())                                               # | holding 32 times that. Those two numbers are the paper's
        return n - self.embed.weight.numel() if non_embedding else n                                # | entire framing, a 3.5B model doing the work of a 50B one, and
                                                                                                    # | they are what the upper axis of its Figure 1 plots. The count
    def materialized_params(self, r: int) -> int:                                                   # | is written as total minus core plus r times core so that r=1
        """Sec. 5.1 / Fig. 1 second x-axis: parameters actually *executed* at depth r.

        The core block's parameters are re-used r times, so the executed compute
        (and the depth of the computation graph) matches a fixed-depth model of
        this many parameters. This is the axis on which the paper claims a 3.5B
        model reaches "a computation load equivalent to 50 billion parameters".
        """
        core = sum(p.numel() for p in self.core.parameters())                                       # | returns the ordinary parameter count unchanged.
        if self.adapter is not None:
            core += self.adapter.weight.numel()
        return self.n_params() - core + r * core

    # --------------------------------------------------------------- the model                     # +-- THE PRELUDE RUNS ONCE, AND e NEVER CHANGES -----------------
    def prelude_forward(self, idx, kv=None):                                                        # | prelude_forward is the only place token ids are looked up. It
        """e = P(x).  Sec. 3.1."""                                                                  # | scales the embedding, runs lP layers, and hands back e along
        x = self.embed(idx) * self.gamma                                                            # | with the key/value caches those layers produced. e is computed
        rope = self.rope(x.device, x.dtype)                                                         # | once and reused for every turn of the loop, however many turns
        new = []                                                                                    # | there are. inject is the adapter, and it is where the design's
        for i, blk in enumerate(self.prelude):                                                      # | central decision lives. concat lays the current state and e
            x, c = blk(x, rope, None if kv is None else kv[i])                                      # | side by side into a 2h vector and multiplies by a learned
            new.append(c)                                                                           # | h-by-2h matrix. add simply sums them and needs no parameters
        return x, new                                                                               # | at all. none drops e after the state is first formed, which is
                                                                                                    # | the ablation that asks whether re-injection matters: with no
    def inject(self, s, e):                                                                         # | data entering the loop, iterating can only push the state
        """The adapter A of Sec. 3.2."""                                                            # | around according to its own dynamics, and there is nothing for
        mode = self.cfg.injection                                                                   # | it to converge toward.
        if mode == "concat":
            return self.adapter(torch.cat([s, e], dim=-1))
        if mode == "add":
            return s + e            # "re-incorporation ... via addition"
        return s                    # "none": no per-step injection (ablation)

    def core_forward(self, s, e, kv=None):                                                          # +-- ONE TURN OF THE LOOP, AND WHERE IT STARTS ------------------
        """s_i = R(e, s_{i-1}).  Adapter -> lR layers -> RMSNorm n_c."""                            # | core_forward is a single application of R. It folds e into the
        x = self.inject(s, e)                                                                       # | state, runs lR layers, then renormalises. That final
        rope = self.rope(x.device, x.dtype)                                                         # | renormalisation is what stops the state growing without bound
        new = []                                                                                    # | over many turns: every layer adds into a residual stream, so
        for i, blk in enumerate(self.core):                                                         # | after 32 turns of 4 layers the coda would otherwise be handed
            x, c = blk(x, rope, None if kv is None else kv[i])                                      # | vectors far larger than anything it saw in training.
            new.append(c)                                                                           # | coda_forward turns a final state into logits and shares its
        return self.core_norm(x), new                                                               # | matrix with the embedding table, so the same vectors that
                                                                                                    # | encode a token also decode it. init_state draws s0. Drawing it
    def coda_forward(self, s, kv=None):                                                             # | at random rather than fixing it at zero forces the loop to
        """p = C(s_r).  lC layers -> n_c -> tied unembedding."""                                    # | arrive somewhere that does not depend on where it began; the
        x = s                                                                                       # | deterministic option exists only so that assumption can be
        rope = self.rope(x.device, x.dtype)                                                         # | tested rather than trusted.
        new = []
        for i, blk in enumerate(self.coda):
            x, c = blk(x, rope, None if kv is None else kv[i])
            new.append(c)
        return self.lm_head(self.coda_norm(x)), new

    def init_state(self, e, generator=None):
        return sample_s0(e.shape, e.device, e.dtype,
                         deterministic=not self.cfg.random_s0, generator=generator)

    # -------------------------------------------------------------- forward API                    # +-- TRUNCATED BACKPROP: MEMORY TRACKS k, NOT r -----------------
    def forward(self, idx, r: int, targets=None, k: int | None = None, s0=None,                     # | This is the training objective. r is how many times the core
                generator=None, return_states: bool = False, ignore_index: int = -100):             # | runs; k is how many of those turns keep a computation graph.
        """One unrolled forward pass at recurrence depth `r`.

        Truncated backprop (Sec. 3.3): the first r-k steps run under no_grad and
        the state is detached; the last k steps carry gradient. Pass k >= r for
        full backpropagation-through-unrolling (our C4 control arm).
        """
        cfg = self.cfg                                                                              # | The first r minus k turns run with gradients switched off and
        k = cfg.backprop_depth if k is None else k                                                  # | the state is detached afterwards, so their activations are
        k = max(1, min(k, r))                                                                       # | released the moment they have been used. Only the last k turns
                                                                                                    # | are differentiated. That is why activation memory depends on k
        e, _ = self.prelude_forward(idx)                                                            # | alone and not on r: measured here, the same 4.78 GB whether r
        s = self.init_state(e, generator) if s0 is None else s0                                     # | is 8 or 32, and the same 10.22 GB on the 4B retrofit whether r
        states = [s] if return_states else None                                                     # | is 4 or 8. e is computed outside the no-gradient region
                                                                                                    # | deliberately. It still carries a gradient, and because it is
        n_nograd = r - k                                                                            # | fed into each of the last k turns, the prelude collects k
        if n_nograd > 0:                                                                            # | separate gradient contributions from one forward pass rather
            with torch.no_grad():                                                                   # | than one. Passing k greater than or equal to r differentiates
                for _ in range(n_nograd):                                                           # | the whole unrolling and is the control arm that checks
                    s, _ = self.core_forward(s, e)                                                  # | truncation costs nothing. return_states keeps every
                    if return_states:                                                               # | intermediate state so a trajectory can be inspected
                        states.append(s)                                                            # | afterwards; it stays off during training because holding r+1
            s = s.detach()                                                                          # | full states is exactly the memory this scheme exists to avoid.
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
