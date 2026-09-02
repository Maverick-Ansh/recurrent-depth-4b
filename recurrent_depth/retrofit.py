"""Track B: retrofit recurrent depth onto a pretrained fixed-depth 4B model.

Why this is a reproduction and not a detour
-------------------------------------------
The paper pretrains its recurrent model from scratch for 800B tokens, and is
explicit that the alternative is open.  Sec. 8 (verbatim):

    "the works of Hao et al. (2024); Cheng and Durme (2024) and Liu et al.
     (2024a) discuss how to finetune existing fixed-depth transformers with this
     capability. These works have a similar aim to ours, enabling reasoning in
     latent space, but approach this goal from separate directions."

and Sec. 6.3 frames the distinction as the central open question:

    "in this way the main distinction between both approaches is whether to
     pretrain from scratch for recurrence, or whether to finetune existing
     fixed-depth models to have this capability"

Sec. 9 lists post-training schemes for recurrent models as future work.  So
"can the Sec. 3 architecture be *installed* on a pretrained model rather than
pretrained into one?" is the paper's own question, and it is the only version of
this paper testable at 4B on two T4s.

The surgery
-----------
A 36-layer Qwen3-4B-Base is cut into the paper's triplet (lP, lR, lC), the core
is looped, and the paper's adapter is inserted in front of it:

    e   = embed(x) -> layers[0 : lP]
    s0  ~ N(0, sigma_s^2)
    s_i = core_norm( layers[lP : lP+lR]( A([s_{i-1} ; e]) ) )
    p   = lm_head( final_norm( layers[lP+lR : ]( s_r ) ) )

Two things must be calibrated or the retrofit starts from a broken model, and
both are places where the paper's constants do not transfer:

  * sigma_s.  Sec. 4.1 fixes sigma_s^2 = 2/5 because *their* embedding output has
    variance 2/5 by construction (2/(5h) scaled by sqrt(h)).  Qwen's hidden
    states have their own scale, so we measure the RMS of e and match it.

  * core_norm.  The paper's n_c at the end of the core keeps the state bounded
    over many iterations.  Inserting a fresh RMSNorm mid-stack would renormalise
    activations the coda was trained to receive at a different scale, so we
    initialise its weight to the base model's own measured RMS at the cut point.
    It is then near-identity at r=1 and still bounds the state at large r.

Adapter initialisation is the experiment
----------------------------------------
  identity : A = [0 | I], so A([s,e]) = e.  At r=1 the retrofit is EXACTLY the
             original Qwen3-4B (up to core_norm), and recurrence is a no-op the
             model must learn to use.  This starts life inside the exact failure
             mode Sec. 4.3 describes for "Bad Run 2" -- "the model has learned
             early to ignore the incoming state s, preventing further
             improvements" -- so whether training escapes it is a real question.
  paper    : both halves randomly initialised, as in the paper.  Faithful, but
             destroys the pretrained function at step 0.
  sum      : A = [I | I], the paper's cheaper "addition rather than
             concatenation" variant, which Sec. 3.2 says "works equally well for
             smaller models".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def chunked_cross_entropy(logits, targets, chunk: int = 64, ignore_index: int = -100):              # +-- LOSS WITHOUT A 300 MB FLOAT32 COPY -------------------------
    """Cross entropy without materialising an fp32 (B, T, 151936) tensor.

    Qwen's vocabulary is 151936, so at B=2, T=256 the fp32 upcast of the logits
    alone is 300 MB and it is what pushed a 15.6 GB T4 over the edge. Summing
    the loss over sequence chunks keeps only one chunk upcast at a time.
    """
    B, T, V = logits.shape                                                                          # | Cross entropy needs float32 to be numerically safe, but Qwen's
    total = logits.new_zeros((), dtype=torch.float32)                                               # | vocabulary is 151936 entries wide, so upcasting the whole
    n = 0                                                                                           # | logit tensor at batch 2 and length 256 allocates 300 MB in one
    for i in range(0, T, chunk):                                                                    # | go. That single allocation is what pushed a 15.6 GB T4 over
        lg = logits[:, i:i + chunk].reshape(-1, V)                                                  # | its limit. Summing the loss over slices of the sequence keeps
        tg = targets[:, i:i + chunk].reshape(-1)                                                    # | only one slice upcast at a time and gives the identical
        cnt = int((tg != ignore_index).sum())                                                       # | number, because cross entropy summed over positions and then
        if cnt == 0:                                                                                # | divided by the count is exactly the mean. Positions marked
            continue                                                                                # | ignore_index are excluded from both the sum and the count, so
        total = total + F.cross_entropy(lg.float(), tg, ignore_index=ignore_index,                  # | padded batches do not shift the average.
                                        reduction="sum")
        n += cnt
    return total / max(n, 1)


# ------------------------------------------------------------------ minimal LoRA                   # +-- LOW-RANK UPDATE, SHARED ACROSS ALL TURNS -------------------
class LoRALinear(nn.Module):                                                                        # | The frozen weight stays; a small correction is added on top,
    """y = W x + (alpha/rank) * B(A x).  W frozen, A/B trained.

    Written out rather than imported so that the weight sharing is unambiguous:
    the core block's LoRA parameters are the SAME tensors at every recurrence
    step, which is what makes this a recurrent model rather than a deep one.
    """
                                                                                                    # | built from two thin matrices whose product has rank at most
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.0):     # | 16. lora_B starts at zero, so at step zero the correction is
        super().__init__()                                                                          # | exactly zero and the model is bit-for-bit the pretrained one.
        self.base = base                                                                            # | These parameters are float32 while the base model is float16,
        for p in self.base.parameters():                                                            # | because the gradient scaler used for mixed precision refuses
            p.requires_grad_(False)                                                                 # | to unscale float16 gradients, so anything being optimised
        self.rank, self.scale = rank, alpha / rank                                                  # | needs a float32 master copy. Casting them to the activation
        # Trainable params are fp32 while the frozen base stays fp16: a                             # | dtype on each call is free at this size. The reason this
        # GradScaler refuses to unscale fp16 gradients, so the master copy of                       # | matters more here than in ordinary fine-tuning: these are the
        # anything we optimise has to be fp32. The rank-r factors are tiny, so                      # | same tensors on every turn of the loop, so a rank-16 update
        # casting them to the activation dtype per call costs nothing.                              # | learned once is applied r times. That weight sharing is what
        dv = base.weight.device                                                                     # | makes this a recurrent model rather than a deep one.
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32, device=dv))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dv))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # B stays zero: dW = 0 at init      # | inject_lora walks the module tree and swaps matching linear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()                        # | layers in place, without recursing into the ones it has just
                                                                                                    # | replaced.
    def forward(self, x):
        h = self.dropout(x)
        h = F.linear(h, self.lora_A.to(h.dtype))
        h = F.linear(h, self.lora_B.to(h.dtype))
        return self.base(x) + self.scale * h


def inject_lora(module: nn.Module, targets, rank=16, alpha=32, dropout=0.0) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            n += 1
        else:
            n += inject_lora(child, targets, rank, alpha, dropout)
    return n


# ------------------------------------------------------------------ the retrofit                   # +-- CUTTING A 36-LAYER MODEL INTO THREE PIECES -----------------
class RecurrentDepthRetrofit(nn.Module):                                                            # | Nothing is copied or rebuilt. The pretrained layer list is
    ATTN_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")        # | sliced into three views: the first lP layers become the
                                                                                                    # | prelude, the next lR become the core that will be looped, the
    def __init__(self, hf_model, l_prelude: int, l_core: int, l_coda: int,                          # | rest become the coda. The embedding, the final norm and the
                 injection: str = "concat", adapter_init: str = "identity",                         # | output head are reused as they are, and every pretrained
                 core_norm: bool = True, backprop_depth: int = 2, random_s0: bool = True,           # | weight is frozen. The only new tensors are the adapter and the
                 grad_checkpoint: bool = True):                                                     # | core norm. How the adapter starts is the whole experiment.
        super().__init__()                                                                          # | identity sets the state half to zero and the embedding half to
        self.hf = hf_model                                                                          # | the identity matrix, so the adapter returns e and ignores the
        inner = hf_model.model                                                                      # | state: at r=1 the model computes exactly what the original
        L = len(inner.layers)                                                                       # | network computed, and the recurrence does nothing at all until
        assert l_prelude + l_core + l_coda <= L, f"{l_prelude}+{l_core}+{l_coda} > {L}"             # | training teaches it to. That is a deliberate choice to begin
        self.lP, self.lR, self.lC = l_prelude, l_core, l_coda                                       # | inside the failure the paper describes for its second failed
        self.h = hf_model.config.hidden_size                                                        # | run, where the model learned early to ignore the incoming
        self.injection, self.backprop_depth, self.random_s0 = injection, backprop_depth, random_s0  # | state. paper initialises both halves at random, which is
        self.grad_checkpoint = grad_checkpoint                                                      # | faithful but starts from a destroyed model: measured, its loss
                                                                                                    # | before training is 11.3 against the base model's 0.71. sum is
        self.embed_tokens = inner.embed_tokens                                                      # | the cheaper addition variant.
        self.rotary = inner.rotary_emb
        self.prelude = inner.layers[:l_prelude]
        self.core = inner.layers[l_prelude:l_prelude + l_core]
        self.coda = inner.layers[l_prelude + l_core:l_prelude + l_core + l_coda]
        self.final_norm = inner.norm
        self.lm_head = hf_model.lm_head

        # adapter and core_norm are trained, so they are fp32 (see LoRALinear)
        if injection == "concat":
            self.adapter = nn.Linear(2 * self.h, self.h, bias=False, dtype=torch.float32)
            self._init_adapter(adapter_init)
        else:
            self.adapter = None

        # n_c: calibrated in `calibrate` below, identity-ish until then
        self.core_norm = _RMSNorm(self.h, hf_model.config.rms_norm_eps) if core_norm else None
        self.sigma_s = 1.0                       # calibrated
        for p in self.hf.parameters():
            p.requires_grad_(False)

    def _init_adapter(self, mode: str):
        h = self.h
        with torch.no_grad():
            W = self.adapter.weight             # (h, 2h) = [A_s | A_e]
            if mode == "identity":              # A([s,e]) = e  -> r=1 is the base model
                W.zero_()
                W[:, h:].copy_(torch.eye(h, dtype=W.dtype))
            elif mode == "sum":                 # A([s,e]) = s + e
                W[:, :h].copy_(torch.eye(h, dtype=W.dtype))
                W[:, h:].copy_(torch.eye(h, dtype=W.dtype))
            elif mode == "paper":               # Sec. 4.1 sigma_h^2 = 2/(5h)
                nn.init.trunc_normal_(W, std=math.sqrt(2.0 / (5.0 * h)),
                                      a=-3 * math.sqrt(2.0 / (5.0 * h)),
                                      b=3 * math.sqrt(2.0 / (5.0 * h)))
            else:
                raise ValueError(mode)
        self.adapter_init = mode

    # ------------------------------------------------------------------ pieces
    def _rope(self, x, position_ids):                                                               # +-- RUNNING THE PIECES, AND THE ADAPTER ------------------------
        return self.rotary(x, position_ids)                                                         # | _run drives a slice of pretrained layers. It passes the rotary
                                                                                                    # | tables in explicitly rather than letting the model build them,
    def _run(self, layers, x, pos, rope):                                                           # | because the core is being called many times over the same
        for lyr in layers:                                                                          # | positions and rebuilding would be wasted work. The layers
            x = lyr(x, attention_mask=None, position_ids=pos, position_embeddings=rope)             # | return a bare tensor in current transformers versions and a
            if isinstance(x, tuple):                                                                # | tuple in older ones, so both are handled. prelude_forward
                x = x[0]                                                                            # | computes e once. inject is the adapter, and it casts the
        return x                                                                                    # | float32 adapter matrix down to the activation dtype rather
                                                                                                    # | than casting the activations up, because the activations are
    def prelude_forward(self, idx):                                                                 # | the large tensor. core_forward is one turn. init_state draws
        x = self.embed_tokens(idx)                                                                  # | the start state at the calibrated scale, or returns zeros if
        pos = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)                            # | the random start is being ablated.
        rope = self._rope(x, pos)
        return self._run(self.prelude, x, pos, rope), pos, rope

    def inject(self, s, e):
        """The adapter A of Sec. 3.2, applied to concat(s_i, e)."""
        if self.injection == "concat":
            # A is fp32 (it is trained); cast it, not the activations
            cat = torch.cat([s, e], dim=-1)
            return F.linear(cat, self.adapter.weight.to(cat.dtype))
        if self.injection == "add":
            return s + e            # "re-incorporation ... via addition"
        return s                    # "none": no per-step injection (ablation)

    def core_forward(self, s, e, pos, rope):
        x = self._run(self.core, self.inject(s, e), pos, rope)
        return self.core_norm(x) if self.core_norm is not None else x

    def coda_forward(self, s, pos, rope):
        return self.lm_head(self.final_norm(self._run(self.coda, s, pos, rope)))

    def init_state(self, e, generator=None):
        if not self.random_s0:
            return torch.zeros_like(e)
        t = torch.empty(e.shape, device=e.device, dtype=torch.float32)
        t.normal_(0.0, self.sigma_s, generator=generator).clamp_(-3 * self.sigma_s, 3 * self.sigma_s)
        return t.to(e.dtype)

    # ----------------------------------------------------------------- forward
    def forward(self, idx, r: int, targets=None, k: int | None = None, s0=None,                     # +-- THE SAME TRUNCATED OBJECTIVE, PLUS CHECKPOINTING -----------
                generator=None, return_states: bool = False, ignore_index: int = -100):             # | Identical to the from-scratch model: the first r minus k turns
        k = self.backprop_depth if k is None else k                                                 # | run without gradients and the state is detached, only the last
        k = max(1, min(k, r))                                                                       # | k turns are differentiated. The addition here is that each
        e, pos, rope = self.prelude_forward(idx)                                                    # | differentiated turn is wrapped in a checkpoint, so its
        s = self.init_state(e, generator) if s0 is None else s0                                     # | intermediate activations are thrown away during the forward
        states = [s] if return_states else None                                                     # | pass and recomputed during the backward one. Without it, k=2
                                                                                                    # | turns of 18 pretrained layers each, at hidden size 2560, does
        if r - k > 0:                                                                               # | not fit alongside 8 GB of frozen weights. With it, peak memory
            with torch.no_grad():                                                                   # | is one turn's worth: measured at 10.22 GB, and identical
                for _ in range(r - k):                                                              # | whether r is 4 or 8. Checkpointing is skipped when states are
                    s = self.core_forward(s, e, pos, rope)                                          # | being collected, because recomputation would produce a second
                    if return_states:                                                               # | set of them.
                        states.append(s)
            s = s.detach()
        for _ in range(k):
            if self.grad_checkpoint and self.training and not return_states:
                # App. A.2: "gradient checkpointing on a per-iteration granularity".
                # One checkpoint per recurrence step, so activation memory is
                # O(one core step) instead of O(k core steps).
                s = torch.utils.checkpoint.checkpoint(
                    self.core_forward, s, e, pos, rope, use_reentrant=False)
            else:
                s = self.core_forward(s, e, pos, rope)
            if return_states:
                states.append(s)

        logits = self.coda_forward(s, pos, rope)
        loss = None
        if targets is not None:
            loss = chunked_cross_entropy(logits, targets, ignore_index=ignore_index)
        out = {"logits": logits, "loss": loss, "state": s}
        if return_states:
            out["states"] = torch.stack(states)
        return out

    @torch.no_grad()
    def trajectory(self, idx, r, s0=None, generator=None):
        return self.forward(idx, r, s0=s0, generator=generator, return_states=True)["states"]

    # --------------------------------------------------------------- calibrate
    @torch.no_grad()                                                                                # +-- MEASURE THE MODEL, DO NOT ASSUME ITS SCALES ----------------
    def calibrate(self, idx) -> dict:                                                               # | The paper fixes the start state's variance at 2/5 and that
        """Measure the base model's own scales at the cut point and match them.

        Sets sigma_s to RMS(e) and core_norm.weight to RMS(h_{lP+lR}), so that
        with the identity adapter the r=1 retrofit reproduces the base model
        instead of feeding the coda activations at an unfamiliar scale.
        """
        e, pos, rope = self.prelude_forward(idx)                                                    # | number is correct for its own model, where the embedding is
        rms_e = e.float().pow(2).mean().sqrt().item()                                               # | initialised at variance 2/(5h) and scaled by sqrt(h). Qwen was
        h_core = self._run(self.core, e, pos, rope)         # what the coda normally receives       # | trained differently and its activations live somewhere else
        rms_core = h_core.float().pow(2).mean().sqrt().item()                                       # | entirely: measured here, e has RMS 6.66 and what the coda
        self.sigma_s = rms_e                                                                        # | expects to receive has RMS 10.08. Copying the paper's constant
        if self.core_norm is not None:                                                              # | would have fed the coda vectors about sixteen times too small.
            self.core_norm.weight.fill_(rms_core)                                                   # | So calibrate runs the base model once, reads both scales off
        return {"rms_e": rms_e, "rms_core_out": rms_core, "sigma_s": self.sigma_s}                  # | it, and sets the start state and the core norm's weight to
                                                                                                    # | match. That makes the new norm close to a no-op at r=1 while
    # ---------------------------------------------------------------- training                     # | still bounding the state when the loop runs long.
    def add_lora(self, rank=16, alpha=32, dropout=0.0, where=("core",)) -> int:                     # | mark_trainable turns on exactly the tensors that must learn:
        n = 0                                                                                       # | the adapter, the core norm, and the low-rank factors, which
        for part in where:                                                                          # | came to 29.6M of 4.05B, or 0.73 percent.
            n += inject_lora(getattr(self, part), self.ATTN_MLP, rank, alpha, dropout)
        return n

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def mark_trainable(self):
        """Adapter + core_norm are new modules and must train; LoRA already does."""
        if self.adapter is not None:
            self.adapter.weight.requires_grad_(True)
        if self.core_norm is not None:
            self.core_norm.weight.requires_grad_(True)
        n = sum(p.numel() for p in self.trainable_parameters())
        tot = sum(p.numel() for p in self.parameters())
        return {"trainable": n, "total": tot, "pct": 100.0 * n / tot}

    def materialized_params(self, r: int) -> int:
        core = sum(p.numel() for p in self.core.parameters())
        if self.adapter is not None:
            core += self.adapter.weight.numel()
        tot = sum(p.numel() for p in self.parameters())
        return tot - core + r * core


class _RMSNorm(nn.Module):                                                                          # +-- BUILD ORDER MATTERS ----------------------------------------
    def __init__(self, dim, eps=1e-6):                                                              # | The retrofit's own modules are created after the pretrained
        super().__init__()                                                                          # | model has already been moved to the GPU, so they start life on
        self.weight = nn.Parameter(torch.ones(dim))                                                 # | the CPU and the whole object has to be moved again. Forgetting
        self.eps = eps                                                                              # | that second move is a silent construction-time error that only
                                                                                                    # | surfaces as a device mismatch deep inside a matrix multiply.
    def forward(self, x):                                                                           # | materialized_params above counts the same thing as in the
        dt = x.dtype                                                                                # | from-scratch model: total weights minus the core, plus the
        x = x.float()                                                                               # | core counted r times. For this split it reads 4.05B at r=1,
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)                             # | 16.98B at r=8 and 61.29B at r=32.
        return (x * self.weight.float()).to(dt)


def build_retrofit(model_id="Qwen/Qwen3-4B-Base", split=(9, 18, 9), adapter_init="identity",
                   dtype=torch.float16, device="cuda", lora_rank=16, lora_alpha=32,
                   backprop_depth=2, injection="concat", core_norm=True, attn="sdpa",
                   grad_checkpoint=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    hf.to(device).eval()
    m = RecurrentDepthRetrofit(hf, *split, injection=injection, adapter_init=adapter_init,
                               core_norm=core_norm, backprop_depth=backprop_depth,
                               grad_checkpoint=grad_checkpoint)
    m.to(device)          # the adapter and core_norm are new modules, still on CPU
    return m, tok
