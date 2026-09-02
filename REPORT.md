# Reproducing *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*

Geiping et al., [arXiv:2502.05171v2](https://arxiv.org/abs/2502.05171).
Reproduction on 2×NVIDIA T4 (16 GB, sm_75) against a paper trained on 4096 AMD
MI250X GPUs for 800B tokens.

---

## 1. Claims, stated so they can be falsified

| # | Claim | Where | What would confirm it |
|---|---|---|---|
| **C1** | Test-time recurrence *r* monotonically improves task performance, and harder tasks saturate at larger *r* | Fig. 1, Fig. 7, Fig. 9, Tab. 1 | accuracy rising in *r*; saturation point ordered by the problem's sequential depth |
| **C2** | A recurrent model beats its non-recurrent twin, by more on reasoning than on recall | Tab. 4 | Δ(rec, fixed-depth) larger on depth-hard than on memory-hard tasks |
| **C3** | The gains live in the recurrence, not in the prelude/coda | Tab. 4, "Ours (r=1)" rows | model@r=1 flat over training while model@r=r̄ improves |
| **C4** | Truncated backprop through the last *k* steps suffices, and makes memory independent of *r* | §3.3 | k≪r matches k=r in loss; peak memory equal at r=8 and r=32 |
| **C5** | Input injection, sandwich norm, learned adapter and random s₀ are load-bearing; too high a learning rate collapses the recurrence | §3.1, §3.2, §4.3, Fig. 5 | ablations flatten the r-curve; token correlation → 1.0 on the collapsed arms |
| **C6** | The model extrapolates past its training depth, and its trajectories are path-independent | §7 | accuracy holds at r > max train r; agreement across s₀ seeds |
| **C7** | Zero-shot per-token adaptive exit and KV-cache sharing cost ~nothing | §6.1, §6.2 | steps saved at unchanged accuracy; sharing at budget ≪ r unchanged |
| **C8** | *(ours)* Recurrent depth can be **retrofitted** onto a pretrained fixed-depth 4B model, not only pretrained into one | §6.3, §8, §9 — the paper's own open question | after surgery + short random-*r* training, loss/accuracy improves with *r* |

C8 is not a departure from the paper. §6.3 states the distinction explicitly —
*"the main distinction between both approaches is whether to pretrain from
scratch for recurrence, or whether to finetune existing fixed-depth models to
have this capability"* — and §9 lists post-training schemes as future work. It is
also the only version of a 4B recurrent-depth experiment that fits on two T4s.

---

## 2. How it was resized

Two tracks, because 3.5B parameters × 800B tokens is roughly 10⁶× our budget.

**Track A — the architecture from scratch, at the paper's own small shape.**
`(lP, lR, lC) = (1, 4, 1)` is what §3.2 says the authors used for their own
small-scale ablations, so the ablations here are run at the scale they were
originally decided at. The substrate is a **depth-controlled task suite** rather
than a shrunken benchmark: running ARC or GSM8K against a model this size
measures noise around chance. Instead:

| task | what it requires | difficulty knob | prompt space |
|---|---|---|---|
| `perm` | S₅ word problem over a fixed generating set. NC¹-complete; a constant-depth (TC⁰) transformer cannot solve it for growing *n* unless TC⁰ = NC¹ | number of generators composed, *n* ∈ {2,4,8,16,24} | 5ⁿ |
| `add` | multi-operand addition — the paper's own App. A.1 / Fig. 14 study | operands × digits, 8 cells | 10²…10⁸ |
| `recall` | single-hop associative recall. Memory-hard, **depth-easy** — the control | key/value pairs ∈ {4,8,16,24} | unbounded |

The control is the point. Without a task that recurrence should *not* help,
"recurrence helps" is unfalsifiable, because everything improves with training.
And because depth is a knob we set, C1's saturation-ordering claim becomes
quantitative rather than anecdotal — sharper than the paper states it, which
tests it via few-shot count (Fig. 9) and operand count (Fig. 14).

**Track B — the 4B.** `Qwen3-4B-Base` (36 layers, h=2560, GQA 32/8, vocab 151936)
cut into `(9, 18, 9)` — the paper's 25/50/25 proportions — the core looped, the
§3.2 concat adapter installed, and the §3.3 random-*r* objective applied with
LoRA on the core.

### Deviations

| | paper | here | why it is still a test of the claim |
|---|---|---|---|
| model | 3.5B, (2,4,2), h=5280 | **A:** 19.6M, (1,4,1), h=512 · **B:** 4.05B Qwen3, (9,18,9), h=2560 | (1,4,1) is the paper's own ablation shape; Track B is at the paper's parameter scale |
| tokens | 800B | **A:** 24.6M · **B:** ~1M retrofit | claims C1–C7 are about the *shape* of the r-curve, not absolute benchmark numbers |
| r̄ , k | 32 , 8 | **A:** 8 , 4 · **B:** 4 , 2 | the whole r-sweep including extrapolation must fit the budget; ratios r̄/k preserved (4:1 vs 4:1) |
| precision | bf16 on MI250X | **fp16 + GradScaler** | T4 is sm_75; bf16 is emulated there and ~4× slower |
| learning rate | 4×10⁻⁵ peak, warmup 4096 | **A:** 3×10⁻⁴, warmup 200 | 4×10⁻⁵ is tuned for 3.5B; the *ordering* claim of §4.3 (too high → collapse) is run as its own arm rather than assumed |
| data | 800B tokens web/code/math, 65536-token BPE | **A:** byte-level task suite · **B:** FineWeb-Edu + open-web-math + the-stack | byte-level means no tokenizer artifact to lose; Track B keeps Qwen's tokenizer |
| tokenizer | trained on instruction data | raw bytes (Track A) | removes a confound and a reproducibility hazard |
| eval | lm-eval-harness, full test sets | 17 exact-ground-truth cells (A); 300-item subsamples (B) | every number is reported with its floor and standard error |
| **weights** | **all trained** | **Track B: base frozen, LoRA r=16 on the core + adapter + n_c = 29.6M / 4.05B (0.73%)** | **material.** Track B tests whether recurrence can be *installed*, not whether a 4B can be pretrained |

The last row is the one to hold onto when reading Track B.

---

## 3. Fidelity checks that cost nothing and caught real things

`scripts/smoke.py` — 21 assertions, all passing — checks the paper's *rules*, not
tensor shapes. Three worth quoting:

**The unrolling distribution is pinned against the paper's own Figure 3.** §3.3
writes *"a variance that we set to σ = ½"*, then samples
`τ ~ N(log r̄ − ½σ², σ)`. Whether σ is the standard deviation or the variance
changes the distribution. Only the σ-as-std reading reproduces the moments
annotated on Fig. 3:

| | paper Fig. 3 (r̄=32) | σ as std | σ as variance |
|---|---|---|---|
| mean | 33.0 | **32.98** | 37.3 |
| median | 29.0 | **29.0** | 33.0 |

**Our parameter layout reproduces the paper's own Figure 1 axis.** Building the
exact `(2,4,2)`, h=5280, 55 heads × 96, MLP 17920, vocab 65536 config on the meta
device gives 3.56B total and 1.64B in the core (paper: "3.5B", "about 1.5B"), and
the materialized-parameter axis of Fig. 1 — 3.6B, 8.3B, 11.5B, 14.6B, 21.0B,
33.6B, 52.6B, 77.9B, 103B at r = 1, 4, 6, 8, 12, 20, 32, 48, 64 — is reproduced
to a maximum relative error of **3.6%**. A wrong adapter shape or a missing norm
would not land on that axis.

> One inconsistency surfaced: §4.1 says *"0.5B in the tied input embedding"*, but
> the stated vocabulary and hidden size give 65536 × 5280 = **0.35B**. The
> 3.5B total in the abstract is consistent with our 3.56B build; the 0.5B figure
> is not consistent with the other numbers in the same sentence.

**s₀'s scale is not a free constant.** §4.1 fixes σ_s² = 2/5 without derivation.
It is forced: the embedding matrix has variance 2/(5h) and its output is scaled
by γ = √h, so γE(x) has variance (2/(5h))·h = 2/5 — *exactly* σ_s². The random
initial state is deliberately placed on the same scale as the injected
embedding. Measured: var(γE(x)) = 0.392, var(s₀) = 0.388. This matters for
Track B, where Qwen's activations are on a completely different scale
(RMS(e) = 6.66) and the constant must be recomputed rather than copied.

---

## 4. What broke — the evaluation, three times, before the model once

This is the section worth reading.

### 4.1 The validation loss was flat while accuracy was climbing

First measurement of C1 used held-out loss over the packed byte stream, the
analogue of the paper's Fig. 6. It read:

```
val loss by r:  1:2.313  2:2.298  4:2.298  8:2.298  16:2.298  32:2.298
```

Flat — an apparent refutation of the paper's central claim. But the task grid on
the same checkpoint read:

```
add/(3,1)   floor 0.085 |  r1=0.16  r2=0.23  r4=0.31  r8=0.30  r16=0.30
```

which is clearly *not* flat. The loss was the broken instrument. Most bytes in
the stream are prompts — random operands, random generator symbols, random
key/value pairs — and they are irreducibly unpredictable no matter how much the
model reasons. Averaging over them buries the handful of answer positions where
reasoning can show up at all. Fix: `answer_loss_vs_r`, restricted to answer
tokens. Both are now reported; the full-stream loss is kept precisely because
its flatness is the interesting artifact.

**If this reproduction had stopped at the first plot, it would have reported
"C1 does not reproduce".**

### 4.2 Half the task cells were solvable by table lookup

The first `perm` formulation sampled uniformly from all 120 elements of S₅, and
the model never got off the floor at any difficulty in 600 steps — it had to
memorise a 120×120 multiplication table before it could compose anything. The
fix is also the theoretically correct statement: the word problem is defined
over a **fixed generating set**, so the model reads 5 distinct symbols while
tracking a 120-state automaton. NC¹-hardness is unchanged.

That introduced a second problem, which the gate caught and refused to run past:
with 5 generators, `perm/2` has only 25 possible prompts and every eval item
appears verbatim in training. So the gate now *computes* each cell's prompt space
and classifies it. The measured leak fraction tracks the computed space exactly:

| cell | prompt space | measured leak | class |
|---|---|---|---|
| `perm/2` | 2.5e+01 | 1.00 | tabulable |
| `perm/4` | 6.2e+02 | 1.00 | tabulable |
| `perm/8` | 3.9e+05 | 0.03 | tabulable |
| `perm/16` | 1.5e+11 | 0.00 | **must compute** |
| `add/(2,1)` | 1.0e+02 | 1.00 | tabulable |
| `add/(3,1)` | 1.0e+03 | 0.73 | tabulable |
| `add/(4,1)` | 1.0e+04 | 0.03 | tabulable |
| `add/(3,3)` | 7.3e+08 | 0.00 | **must compute** |
| `recall/*` | ∞ | 0.00 | **must compute** |

Eight of seventeen cells can carry a reasoning claim. The other nine are kept as
the genuinely easy end of the difficulty axis — and being memorisable is itself
informative, since C1 predicts easy tasks saturate at small *r*.

### 4.3 The KV cache double-counted its own prefill

Sec. 6.2's cache sharing is only observable in real cached decoding, so it was
implemented for real rather than simulated. The first version had recurrence step
3 read the cache slot that step 1 had written *during the same prefill* — so the
prompt tokens were attended to twice, and a cached decode silently disagreed with
the equivalent teacher-forced forward. Reads now come from a committed snapshot
and writes land in a pending map published at the end of each token block. The
regression test is now the load-bearing one in the suite:

```
cached decoding == teacher-forced forward   max |diff| = 1.4e-06  (logit std 0.63)
KV budget >= r is a no-op                   bitwise identical
KV budget < r genuinely perturbs            0.013 logit shift
```

Without that last assertion, "KV sharing costs nothing" would have been
vacuously true, because the sharing was doing nothing.

### 4.4 Two engineering failures worth one line each

The 4B retrofit OOMed on a T4 in the step-time test. Two fixes, both of which the
paper itself uses: per-iteration gradient checkpointing (App. A.2 —
*"gradient checkpointing on a per-iteration granularity"*), and not materialising
an fp32 (B, T, 151936) logit tensor. And `GradScaler` refuses to unscale fp16
gradients, so the trainable parameters (LoRA factors, adapter, n_c) are fp32
while the frozen base stays fp16.

---

## 5. Results

*(This section is filled from `results/*.json` by `scripts/analyze.py`; the
Track-A sweep and Track-B retrofit runs are in progress at time of writing.)*

### 5.1 The surgery is exact (Track B gate)

Before any training, on Qwen3-4B-Base cut into (9, 18, 9) with the identity
adapter `A = [0 | I]`:

| quantity | value |
|---|---|
| base Qwen3-4B loss | 0.705967 |
| retrofit @ r=1, **n_c removed** | **0.705967** |
| max abs logit difference | **0.00e+00** |
| argmax agreement | **1.0000** |
| retrofit @ r=1, with n_c | 0.7288 (+0.0228) |
| argmax agreement with n_c | 0.9490 |

The cut is bit-exact. The entire deviation is the core RMSNorm `n_c`, which is a
deliberate addition (it bounds the state over many iterations) whose cost is now
measured rather than assumed: **+0.023 nats, 5.1% argmax disagreement**.

Calibration mattered. The paper's σ_s² = 2/5 is derived from *its* embedding
scale; Qwen's is RMS(e) = 6.66, and the activations the coda expects have
RMS = 10.08. Copying the paper's constants here would have fed the coda
activations ~16× too small.

For contrast, the **paper-faithful random adapter** on the same surgery, before
any training:

| adapter init | loss @ r=1 | loss @ r=8 |
|---|---|---|
| identity `[0 \| I]` | 0.729 | 0.729 |
| paper (random) | **11.32** | **12.13** |
| base model | 0.706 | — |

This is what "destroys the pretrained function" means quantitatively, and it is
why the identity init is the interesting arm — it starts *inside* the failure
mode §4.3 describes for their second failed run (*"the model has learned early to
ignore the incoming state s"*), so whether training escapes that basin is a real
question rather than a foregone one.

### 5.2 Cost model, measured

| config | step time | peak memory |
|---|---|---|
| Track A, h=512, B=32, T=256, r=8, k=4 | 558 ms | 4.78 GB |
| Track A, h=512, B=32, T=256, **r=32**, k=4 | 1322 ms | **4.78 GB** |
| Track A, h=512, B=32, T=256, r=1, k=1 | 163 ms | 2.32 GB |
| Track B, 4B, B=1, T=256, r=4, k=2 | 991 ms | 10.22 GB |
| Track B, 4B, B=1, T=256, **r=8**, k=2 | 1358 ms | **10.22 GB** |

**C4's memory claim reproduces directly and at both scales.** §3.3: *"maximum
activation memory and backward compute is now independent of r"*. Peak memory is
identical at r=8 and r=32 (Track A) and at r=4 and r=8 (Track B), while step time
grows with *r* as it must. Memory tracks *k*, not *r*: dropping k from 4 to 1
takes Track A from 4.78 GB to 2.32 GB.

Materialized parameters for the 4B retrofit — the paper's Fig. 1 upper axis,
computed for our model:

| r | 1 | 8 | 32 |
|---|---|---|---|
| materialized params | 4.05B | 16.98B | **61.29B** |

### 5.3 Learning rate and the §4.3 collapse

A two-point learning-rate probe (600 steps, Track A, otherwise identical):

| lr | final train loss | token correlation | val loss r=1 → r≥2 |
|---|---|---|---|
| 3×10⁻⁴ | 2.21 | +0.52 | 2.313 → 2.298 |
| 1×10⁻³ | 4.38 | **+0.76** | 2.631 → **3.86** |

The high-lr run collapses: training loss rises, token correlation climbs toward
the 1.0 that Fig. 5 (middle panel) uses to diagnose representation collapse, and
recurrence becomes actively harmful — running the core more makes the model
worse. That is the ordering §4.3 reports at 3.5B (*"a third, and final run
('Main', blue), we fix this issue by … dropping the peak learning rate"*),
reproduced at 19.6M. It is now an arm (`hi_lr`) rather than an anecdote.

### 5.4 Track A: accuracy vs test-time recurrence

*Pending — see `figures/c1_accuracy_vs_r.png` and the tables printed by
`scripts/analyze.py`.*

### 5.5 Track B: the 4B retrofit

*Pending.*

---

## 6. Verdicts

*Pending completion of the sweep.*

---

## 7. What was **not** tested

Stated exhaustively, because a reproduction that only lists its successes is not
worth much.

- **Scale.** Nothing here speaks to whether the claims hold at 3.5B pretrained
  from scratch on 800B tokens. Track A is 19.6M parameters; Track B is 4.05B but
  with 99.3% of its weights frozen and ~1M tokens of adaptation.
- **The paper's actual benchmarks.** No ARC, HellaSwag, MMLU, GSM8K, MBPP or
  HumanEval number here is comparable to Table 1, 2 or 3. Track B's benchmark
  numbers are 300-item subsamples reported with standard errors.
- **Emergent latent behaviours (§7).** The paper's orbits, sliders and
  context-dependent convergence are reported as emerging *with scale*. We plot
  the same PCA projections and distance-to-limit maps, but a 19.6M model showing
  or not showing them is evidence about neither.
- **Weight averaging (§5.4).** The EMA-over-75-checkpoints cooldown is not run.
- **Self-speculative decoding (§6.4).** Implemented nowhere; only C7a and C7b are
  measured.
- **The full data-mixture question (§4.1).** The paper could not ablate its
  mixture either; neither can we.
- **Locked-step sampling's effect on convergence.** We sample one *r* per
  micro-batch as §4.1 does, but with a single process there are no workers to
  synchronise, so the claim that this "improves compute utilization without
  impacting convergence speed" is inherited, not tested.
- **bf16 vs fp16.** The paper trains bf16; sm_75 forces fp16 + GradScaler. Any
  stability difference between the two is confounded with everything else.
- **Longer contexts.** Track A uses 256-byte blocks, Track B 256 tokens, against
  the paper's 4096.

---

## 8. Reproducing this

```bash
python scripts/smoke.py                     # 21 paper-rule assertions
python scripts/check_eval.py                # gate the instrument
python scripts/run_sweep.py --steps 3000 --gpus 0,1
python scripts/analyze.py
python scripts/analyze_mechanisms.py --ckpt results/rec_s0.pt
python scripts/prep_retrofit_data.py
python scripts/verify_retrofit.py           # gate the 4B surgery
python scripts/train_retrofit.py --adapter-init identity --tag retro_identity
python scripts/train_retrofit.py --adapter-init paper    --tag retro_paper
python scripts/eval_benchmarks.py --ckpt results/retro_identity_trainable.pt
```

Seeds are fixed and the byte corpus is cached per (size, seed), so every arm sees
an identical stream and differs only in architecture and objective.
