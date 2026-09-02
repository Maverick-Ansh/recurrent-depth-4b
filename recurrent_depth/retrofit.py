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


# ------------------------------------------------------------------ minimal LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha/rank) * B(A x).  W frozen, A/B trained.

    Written out rather than imported so that the weight sharing is unambiguous:
    the core block's LoRA parameters are the SAME tensors at every recurrence
    step, which is what makes this a recurrent model rather than a deep one.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        dt, dv = base.weight.dtype, base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=dt, device=dv))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # B stays zero: dW = 0 at init
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)


def inject_lora(module: nn.Module, targets, rank=16, alpha=32, dropout=0.0) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            n += 1
        else:
            n += inject_lora(child, targets, rank, alpha, dropout)
    return n


# ------------------------------------------------------------------ the retrofit
class RecurrentDepthRetrofit(nn.Module):
    ATTN_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, hf_model, l_prelude: int, l_core: int, l_coda: int,
                 injection: str = "concat", adapter_init: str = "identity",
                 core_norm: bool = True, backprop_depth: int = 2, random_s0: bool = True):
        super().__init__()
        self.hf = hf_model
        inner = hf_model.model
        L = len(inner.layers)
        assert l_prelude + l_core + l_coda <= L, f"{l_prelude}+{l_core}+{l_coda} > {L}"
        self.lP, self.lR, self.lC = l_prelude, l_core, l_coda
        self.h = hf_model.config.hidden_size
        self.injection, self.backprop_depth, self.random_s0 = injection, backprop_depth, random_s0

        self.embed_tokens = inner.embed_tokens
        self.rotary = inner.rotary_emb
        self.prelude = inner.layers[:l_prelude]
        self.core = inner.layers[l_prelude:l_prelude + l_core]
        self.coda = inner.layers[l_prelude + l_core:l_prelude + l_core + l_coda]
        self.final_norm = inner.norm
        self.lm_head = hf_model.lm_head

        dt = self.embed_tokens.weight.dtype
        if injection == "concat":
            self.adapter = nn.Linear(2 * self.h, self.h, bias=False).to(dt)
            self._init_adapter(adapter_init)
        else:
            self.adapter = None

        # n_c: calibrated in `calibrate` below, identity-ish until then
        self.core_norm = _RMSNorm(self.h, hf_model.config.rms_norm_eps).to(dt) if core_norm else None
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
    def _rope(self, x, position_ids):
        return self.rotary(x, position_ids)

    def _run(self, layers, x, pos, rope):
        for lyr in layers:
            x = lyr(x, attention_mask=None, position_ids=pos, position_embeddings=rope)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def prelude_forward(self, idx):
        x = self.embed_tokens(idx)
        pos = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
        rope = self._rope(x, pos)
        return self._run(self.prelude, x, pos, rope), pos, rope

    def inject(self, s, e):
        if self.injection == "concat":
            return self.adapter(torch.cat([s, e], dim=-1))
        if self.injection == "add":
            return s + e
        return s

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
    def forward(self, idx, r: int, targets=None, k: int | None = None, s0=None,
                generator=None, return_states: bool = False, ignore_index: int = -100):
        k = self.backprop_depth if k is None else k
        k = max(1, min(k, r))
        e, pos, rope = self.prelude_forward(idx)
        s = self.init_state(e, generator) if s0 is None else s0
        states = [s] if return_states else None

        if r - k > 0:
            with torch.no_grad():
                for _ in range(r - k):
                    s = self.core_forward(s, e, pos, rope)
                    if return_states:
                        states.append(s)
            s = s.detach()
        for _ in range(k):
            s = self.core_forward(s, e, pos, rope)
            if return_states:
                states.append(s)

        logits = self.coda_forward(s, pos, rope)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=ignore_index)
        out = {"logits": logits, "loss": loss, "state": s}
        if return_states:
            out["states"] = torch.stack(states)
        return out

    @torch.no_grad()
    def trajectory(self, idx, r, s0=None, generator=None):
        return self.forward(idx, r, s0=s0, generator=generator, return_states=True)["states"]

    # --------------------------------------------------------------- calibrate
    @torch.no_grad()
    def calibrate(self, idx) -> dict:
        """Measure the base model's own scales at the cut point and match them.

        Sets sigma_s to RMS(e) and core_norm.weight to RMS(h_{lP+lR}), so that
        with the identity adapter the r=1 retrofit reproduces the base model
        instead of feeding the coda activations at an unfamiliar scale.
        """
        e, pos, rope = self.prelude_forward(idx)
        rms_e = e.float().pow(2).mean().sqrt().item()
        h_core = self._run(self.core, e, pos, rope)         # what the coda normally receives
        rms_core = h_core.float().pow(2).mean().sqrt().item()
        self.sigma_s = rms_e
        if self.core_norm is not None:
            self.core_norm.weight.fill_(rms_core)
        return {"rms_e": rms_e, "rms_core_out": rms_core, "sigma_s": self.sigma_s}

    # ---------------------------------------------------------------- training
    def add_lora(self, rank=16, alpha=32, dropout=0.0, where=("core",)) -> int:
        n = 0
        for part in where:
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


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_retrofit(model_id="Qwen/Qwen3-4B-Base", split=(9, 18, 9), adapter_init="identity",
                   dtype=torch.float16, device="cuda", lora_rank=16, lora_alpha=32,
                   backprop_depth=2, injection="concat", core_norm=True, attn="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    hf.to(device).eval()
    m = RecurrentDepthRetrofit(hf, *split, injection=injection, adapter_init=adapter_init,
                               core_norm=core_norm, backprop_depth=backprop_depth)
    m.to(device)          # the adapter and core_norm are new modules, still on CPU
    return m, tok
