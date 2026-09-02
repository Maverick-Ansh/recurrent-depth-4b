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


# --------------------------------------------------------------------------                        # +-- STOP LOOPING WHEN THE ANSWER STOPS MOVING ------------------
# 6.1  Per-position adaptive exit, teacher-forced (the Figure 10 measurement)                       # | The loop runs, and after each turn the coda is applied to get
# --------------------------------------------------------------------------                        # | a next-token distribution. When that distribution stops
@torch.no_grad()                                                                                    # | changing, more turns cannot change the answer, so the position
def adaptive_exit_forward(model, idx, r_max: int = 64, threshold: float = 5e-4,                     # | is finished. The measure of change is the KL divergence
                          s0=None, generator=None):                                                 # | between this turn's distribution and the previous turn's;
    """Iterate the core, freezing each sequence position's logits at the step
    where KL(p_i || p_{i-1}) first falls below `threshold`.

    Every position is computed in parallel, so this needs no cache and gives an
    exact per-token exit-step distribution -- the histogram of the paper's
    Fig. 10 -- together with the logits the model would actually have emitted.
    """
    e, _ = model.prelude_forward(idx)                                                               # | below the threshold, that position's logits are frozen and it
    s = model.init_state(e, generator) if s0 is None else s0                                        # | stops being updated while other positions keep going. Every
    B, S = idx.shape                                                                                # | position is computed in parallel here, so this needs no cache
    exit_step = torch.full((B, S), r_max, device=idx.device, dtype=torch.long)                      # | at all and produces an exact per-token distribution of how
    done = torch.zeros(B, S, dtype=torch.bool, device=idx.device)                                   # | many turns each token needed. That distribution is the
    frozen, prev_logp = None, None                                                                  # | interesting object: it says which tokens the model found hard
                                                                                                    # | without anyone labelling them. The cost is running the coda
    for i in range(1, r_max + 1):                                                                   # | every turn instead of once, which is why the saving is counted
        s, _ = model.core_forward(s, e)                                                             # | in core turns rather than wall clock. torch.where keeps
        logits, _ = model.coda_forward(s)                                                           # | finished positions frozen rather than breaking out of the
        if frozen is None:                                                                          # | loop, because the batch shares one loop and different
            frozen = logits.clone()                                                                 # | positions finish at different times.
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


# --------------------------------------------------------------------------                        # +-- ONE CACHE SLOT PER TURN, OR FEWER --------------------------
# 6.2 / 6.3  Cached autoregressive decoding with a recurrence KV budget                             # | Every turn of the core writes its own keys and values, so an
# --------------------------------------------------------------------------                        # | unbudgeted cache costs r times what a normal transformer's
class RecurrenceKVCache:                                                                            # | does. The budget caps that: turn i reads and writes slot (i-1)
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
                                                                                                    # | mod budget, so turn 1 and turn budget+1 land in the same place
    def __init__(self, n_layers: int, budget: int | None):                                          # | and the later one overwrites the earlier. The split between
        self.budget = budget                                                                        # | committed and pending is the part that is easy to get wrong,
        self.n_layers = n_layers                                                                    # | and getting it wrong here produced a real bug. A slot holds
        self.committed: dict[int, list] = {}                                                        # | keys and values for tokens already emitted. Within one block
        self.pending: dict[int, list] = {}                                                          # | of new tokens, which is the whole prompt during prefill or a
                                                                                                    # | single token during decoding, every turn must read the same
    def slot_of(self, step: int) -> int:                                                            # | committed state. When turn 3 was allowed to read what turn 1
        return (step - 1) % self.budget if self.budget else step - 1                                # | had written during the same prefill, the prompt's own tokens
                                                                                                    # | were attended to twice and a cached decode silently disagreed
    def begin_block(self):                                                                          # | with the equivalent full-sequence forward. Reads now come from
        self.pending = {}                                                                           # | committed, writes land in pending, and end_block publishes
                                                                                                    # | them once the whole block is done.
    def read(self, step: int):
        return self.committed.get(self.slot_of(step))

    def write(self, step: int, layer_caches: list):
        self.pending[self.slot_of(step)] = layer_caches

    def end_block(self):
        self.committed.update(self.pending)
        self.pending = {}


@torch.no_grad()                                                                                    # +-- DECODING, WITH BOTH ZERO-SHOT TRICKS WIRED IN --------------
def generate(model, prompt, max_new_tokens: int = 8, r: int = 8, kv_budget: int | None = None,      # | One token at a time. The prelude runs on the new token and
             kl_threshold: float | None = None, r_max: int = 64, warm_start: bool = False,          # | appends to its own cache; the core turns as many times as
             greedy: bool = True, eos_id: int | None = None, generator=None):                       # | asked, or until the KL criterion says the distribution has
    """Cached decoding.  Supports Sec. 6.1 (kl_threshold), 6.2 (kv_budget) and
    6.3 (warm_start: carry s_r into the next token's s_0 instead of resampling).

    Returns the generated ids and, per generated token, the recurrence steps used.
    """
    device = prompt.device                                                                          # | settled; the coda produces the logits. warm_start is the third
    B = prompt.shape[0]                                                                             # | trick: instead of drawing a fresh random start state for each
    prelude_kv, coda_kv = None, None                                                                # | new token, it carries the previous token's final state in as
    core_cache = RecurrenceKVCache(len(model.core), kv_budget)                                      # | the new starting point, which chains the computation across
    ids = prompt                                                                                    # | tokens and makes the effective depth larger than r without
    steps_used, generated = [], []                                                                  # | running more turns. The KL probe calls the coda with the
    carry = None                                                                                    # | committed cache, which is safe because coda_forward builds and
    cur = prompt                                                                                    # | returns new cache tensors rather than modifying the ones
                                                                                                    # | handed to it, so probing every turn leaves the real cache
    for t in range(max_new_tokens + 1):                                                             # | untouched. steps_used records how many turns each generated
        e, prelude_kv = model.prelude_forward(cur, prelude_kv)                                      # | token actually took, which is the number the adaptive-exit
        if warm_start and carry is not None:                                                        # | claim is measured on.
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
