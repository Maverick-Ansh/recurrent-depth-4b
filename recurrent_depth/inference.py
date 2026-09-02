"""Zero-shot inference behaviours that recurrent depth gets for free (Sec. 6).

6.1 Zero-Shot Adaptive Compute (verbatim):

    "To test our model's zero-shot exit abilities, we choose a simple exit
     criterion to evaluate convergence, the KL-divergence between two successive
     steps. If this divergence falls below 5 x 10^-4, we stop iterating, sample
     the output token, and move to generate the next token."

6.2 Zero-Shot KV-cache Sharing (verbatim):

    "We set a fixed KV-cache budget for the recurrence at every token k, and at
     iteration i, read and write the cache entry i mod k. For example, we set a
     maximum KV-cache budget of 16 steps, overwriting the KV-cache of the 1st
     step when executing the 17th step, and so forth."

6.3 Zero-Shot Continuous Chain-of-Thought (verbatim):

    "Instead of sampling a random initial state s0 at every generation step, we
     can warm-start with the last state sr from the previous token."

Two notes where the paper underspecifies, and what we chose:

* The KL is "between two successive steps" but not between *which* quantities.
  We use the coda's next-token distributions, KL(p_i || p_{i-1}), since that is
  the only per-step distribution the model defines and it is what "stop
  iterating, sample the output token" implies.  Running the coda every step
  costs lC extra layers per step; we report the exit-step histogram (the paper's
  Fig. 10 quantity) and the accuracy, and separately note that the compute saved
  is in *core* steps.

* KV-cache sharing is only observable in cached autoregressive decoding -- in a
  full-sequence teacher-forced pass there is no cache to share.  So C7b is
  measured through the real cached `generate` path below, not simulated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 6.1  Per-position adaptive exit, teacher-forced (the Figure 10 measurement)
# --------------------------------------------------------------------------
@torch.no_grad()
def adaptive_exit_forward(model, idx, r_max: int = 64, threshold: float = 5e-4,
                          s0=None, generator=None):
    """Iterate the core, freezing each sequence position's logits at the step
    where KL(p_i || p_{i-1}) first falls below `threshold`.

    Every position is computed in parallel, so this needs no cache and gives an
    exact per-token exit-step distribution -- the histogram of the paper's
    Fig. 10 -- together with the logits the model would actually have emitted.
    """
    e, _ = model.prelude_forward(idx)
    s = model.init_state(e, generator) if s0 is None else s0
    B, S = idx.shape
    exit_step = torch.full((B, S), r_max, device=idx.device, dtype=torch.long)
    done = torch.zeros(B, S, dtype=torch.bool, device=idx.device)
    frozen, prev_logp = None, None

    for i in range(1, r_max + 1):
        s, _ = model.core_forward(s, e)
        logits, _ = model.coda_forward(s)
        if frozen is None:
            frozen = logits.clone()
        else:                                   # positions still running get updated
            frozen = torch.where(done.unsqueeze(-1), frozen, logits)
        logp = F.log_softmax(logits.float(), dim=-1)
        if prev_logp is not None:
            kl = (logp.exp() * (logp - prev_logp)).sum(-1)        # KL(p_i || p_{i-1})
            newly = (~done) & (kl < threshold)
            exit_step[newly] = i
            done |= newly
        prev_logp = logp
        if bool(done.all()):
            break

    return {"logits": frozen, "exit_step": exit_step,
            "mean_steps": exit_step.float().mean().item()}


# --------------------------------------------------------------------------
# 6.2 / 6.3  Cached autoregressive decoding with a recurrence KV budget
# --------------------------------------------------------------------------
class RecurrenceKVCache:
    """Per-recurrence-step KV cache with the Sec. 6.2 modular budget.

    Without a budget, recurrence step i owns its own cache; memory grows like
    r * lR * seq.  With budget b, step i reads and writes slot (i-1) mod b, so
    memory is capped at b * lR * seq.

    The read/write split matters and is easy to get wrong.  A cache slot holds
    KV entries for tokens *already emitted*.  Within one block of new tokens (the
    prompt during prefill, or one token during decode) every recurrence step must
    read the same committed state -- if step 3 read what step 1 wrote during the
    same block, the new tokens would be attended to twice and the prefill would
    silently differ from decoding.  So reads come from `committed`, writes land
    in `pending`, and `end_block` publishes them.  A slot revisited inside one
    block (steps i and i+b) is simply overwritten in `pending`, which is exactly
    the paper's "overwriting the KV-cache of the 1st step when executing the
    17th step".
    """

    def __init__(self, n_layers: int, budget: int | None):
        self.budget = budget
        self.n_layers = n_layers
        self.committed: dict[int, list] = {}
        self.pending: dict[int, list] = {}

    def slot_of(self, step: int) -> int:
        return (step - 1) % self.budget if self.budget else step - 1

    def begin_block(self):
        self.pending = {}

    def read(self, step: int):
        return self.committed.get(self.slot_of(step))

    def write(self, step: int, layer_caches: list):
        self.pending[self.slot_of(step)] = layer_caches

    def end_block(self):
        self.committed.update(self.pending)
        self.pending = {}


@torch.no_grad()
def generate(model, prompt, max_new_tokens: int = 8, r: int = 8, kv_budget: int | None = None,
             kl_threshold: float | None = None, r_max: int = 64, warm_start: bool = False,
             greedy: bool = True, eos_id: int | None = None, generator=None):
    """Cached decoding.  Supports Sec. 6.1 (kl_threshold), 6.2 (kv_budget) and
    6.3 (warm_start: carry s_r into the next token's s_0 instead of resampling).

    Returns the generated ids and, per generated token, the recurrence steps used.
    """
    device = prompt.device
    B = prompt.shape[0]
    prelude_kv, coda_kv = None, None
    core_cache = RecurrenceKVCache(len(model.core), kv_budget)
    ids = prompt
    steps_used, generated = [], []
    carry = None
    cur = prompt

    for t in range(max_new_tokens + 1):
        e, prelude_kv = model.prelude_forward(cur, prelude_kv)
        if warm_start and carry is not None:
            # Sec. 6.3: "warm-start with the last state sr from the previous token"
            s = carry[:, -1:].expand(-1, e.shape[1], -1).contiguous()
        else:
            s = model.init_state(e, generator)

        core_cache.begin_block()
        limit = r if kl_threshold is None else r_max
        prev_logp, used = None, limit
        for i in range(1, limit + 1):
            s, new_kv = model.core_forward(s, e, core_cache.read(i))
            core_cache.write(i, new_kv)
            if kl_threshold is not None:
                # coda_forward returns fresh cache tensors and does not mutate
                # coda_kv, so probing here is side-effect free.
                probe, _ = model.coda_forward(s, coda_kv)
                logp = F.log_softmax(probe[:, -1].float(), -1)
                if prev_logp is not None:
                    kl = (logp.exp() * (logp - prev_logp)).sum(-1)
                    if bool((kl < kl_threshold).all()):
                        used = i
                        break
                prev_logp = logp
        core_cache.end_block()

        logits, coda_kv = model.coda_forward(s, coda_kv)
        carry = s
        steps_used.append(used)
        if t == max_new_tokens:
            break
        nxt = (logits[:, -1].argmax(-1, keepdim=True) if greedy
               else torch.multinomial(logits[:, -1].softmax(-1), 1))
        generated.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)
        cur = nxt
        if eos_id is not None and bool((nxt == eos_id).all()):
            break

    gen = torch.cat(generated, dim=1) if generated else torch.zeros(B, 0, dtype=torch.long, device=device)
    return {"ids": ids, "generated": gen, "steps_used": steps_used}
