# recurrent-depth-4b

A from-scratch reproduction of **"Scaling up Test-Time Compute with Latent Reasoning:
A Recurrent Depth Approach"** (Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson,
Kailkhura, Bhatele, Goldstein — [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)),
plus a **retrofit of the architecture onto a pretrained 4B model**, which is the paper's
own open question and the only version of it that fits on two T4s.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/recurrent-depth-4b/blob/main/recurrent_depth_4b.ipynb)

Every module in `recurrent_depth/` carries a right-margin comment rail explaining what
the code does and why, and quotes the section of the paper it implements verbatim.

---

## 1. The problem, from the start

A transformer with `L` layers does a fixed amount of work per token. Whether the next
token is the second half of a common word or the answer to a multi-step arithmetic
problem, the same `L` matrix multiplies run. The amount of computation is a property of
the architecture, not of the question.

There are two established ways to buy more computation.

**Make the model bigger.** More parameters per layer, more layers. This costs training
compute, it costs memory at inference, and the extra capacity is spent on *everything* —
including the tokens that never needed it.

**Make the model write more.** Chain-of-thought: the model emits intermediate steps as
text, and each emitted token buys another full forward pass. This works, and it is what
most current reasoning models do. But it has three costs that are structural rather than
incidental:

- Every intermediate result has to be squeezed through a single token from a fixed
  vocabulary. Whatever the model was holding in a 5280-dimensional vector at layer 40, it
  must project down to one of 65536 discrete choices before it can carry it forward.
- Those intermediate steps live in the context window, so more thinking means longer
  sequences, and attention over longer sequences costs more.
- The model has to be *taught* to produce them. That means chain-of-thought training data
  in the domain you care about, constructed by someone.

## 2. The third axis: iterate instead of emit

If the goal is more computation before the next token, there is a direct way to get it:
run part of the network more than once.

Take a block of layers `R`. Apply it to a state. Apply it to the result. Keep going. After
`r` applications the computation is `r` times as deep, and no parameters were added,
nothing was written to the context window, and no special training data was needed. `r`
can be chosen at inference time, per query, and it does not have to be the value used
during training.

This is the paper's proposal. The model has 8 real layers; run with `r = 32` it unrolls to
132, and executes the arithmetic of a 52B-parameter fixed-depth transformer.

The rest of this section derives why the architecture looks the way it does. Almost every
piece is there to fix a specific failure that shows up when you actually try this.

### 2.1 Three groups, not one loop

You cannot simply loop the whole network. The first few layers of a language model do
something qualitatively different from the middle: they turn sub-word pieces into a
representation of a concept. The last few layers do something different again: they turn a
representation back into a distribution over the vocabulary. Neither of those jobs is
iterative — doing them thirty-two times would be meaningless.

So the layers are split into three groups:

```
e  = P(x)              prelude — runs once, embeds tokens into latent space
s0 ~ N(0, sigma_s^2)   a random starting state
si = R(e, s_{i-1})     core — the only part that loops, r times
p  = C(s_r)            coda — runs once, un-embeds and predicts
```

`(lP, lR, lC)` — the number of layers in each group — plus `r` describes the whole model.
The paper's main run is `(2, 4, 2)` at hidden size 5280.

### 2.2 Why `e` is fed in at *every* step, not just the first

The obvious design is `s0 = e`: start the loop from the embedded input and let it run.
That does not work, and the reason is not empirical.

An iterative process that only sees its data through its initial condition is a map
`s → R(s)` applied repeatedly. Where it ends up is determined by where it started. There
is nothing pulling it toward an answer that depends on the input, because after the first
step the input is gone.

Compare gradient descent on some function `E(x, y)`, where `x` is what you are solving for
and `y` is the data. Each step uses `y` again. If you only used `y` to pick `x0` and then
iterated on `x` alone, you would not be minimising anything data-dependent.

So `e` is re-injected into every turn. `R` takes two arguments, always. Formally, without
that dependence `R` cannot be a monotone operator and so cannot represent gradient descent
on a strictly convex, data-dependent function.

This repository ships that as a switchable ablation (`injection="none"`) precisely because
it is the load-bearing claim, and a claim you can't turn off is a claim you can't test.

### 2.3 Why the starting state is random

Given that `e` enters at every step, what should `s0` be?

Setting it to zero would work, but it hides a question you want answered: *does the loop
converge to something that depends on the input, or to something that depends on where it
started?* If the answer must be the former — and it must, for the model to be usable at any
`r` — then the honest way to build that in is to start from noise and force the model to
reach the same place regardless. This property is called **path independence**.

It also gives you a free diagnostic. Run the same input from four different random starts.
If the four runs agree, the loop is doing what it should. If they don't, more recurrence at
test time is not safe. `recurrent_depth/diagnostics.py` measures exactly this.

The variance of `s0` is not a free knob. The embedding matrix is initialised at variance
`2/(5h)` and its output is scaled by `gamma = sqrt(h)`, so `e` has variance `2/5`. The
paper sets `sigma_s^2 = 2/5` — the same number — so the state and the injected embedding
arrive at the adapter on the same scale. This matters enormously in Track B, where the
model being retrofitted has entirely different activation scales (measured: RMS 6.66, not
0.63) and the constant has to be *recomputed*, not copied.

### 2.4 The adapter

`R` needs to take two things — the current state `s` and the embedding `e` — and produce
one hidden vector for its layers. Two options:

- **Add them**: `A(s, e) = s + e`. No parameters. Works at small scale.
- **Concatenate and project**: `A(s, e) = W · [s ; e]` with `W` of shape `h × 2h`. The model
  learns how much of each to keep, and can keep different mixtures per channel.

The paper found the second necessary at scale. It is also the single new matrix the whole
architecture requires — everything else is ordinary transformer layers.

### 2.5 Why the norms are arranged the way they are

A standard pre-norm transformer normalises *before* each sublayer and lets the residual
stream pass through untouched. The residual stream therefore grows with depth. In a fixed
20-layer model that is fine. In a model where the same 4 layers run 32 times, the stream is
being added to 128 times, and it grows without bound.

The paper's block normalises after each residual addition as well:

```
x_hat = n2( x     + Attn(n1(x)) )
x     = n4( x_hat + MLP(n3(x_hat)) )
```

Four norms per layer. This is the "sandwich" arrangement, and §4.3 of the paper is a report
of what happens without it: their first large training run collapsed. Not diverged —
*collapsed*, in a specific and diagnosable way. Every recurrence step increased the
correlation between different tokens' hidden states, until the model was predicting the
same hidden state for every token in the sequence. The metric that catches this
(`token_correlation` in `diagnostics.py`) goes to 1.0.

Their second attempt fixed the collapse but landed somewhere else: the model learned to
*ignore* the incoming state entirely. Validation perplexity was the same at `r = 1` and
`r = 32`. Recurrence was present in the architecture and absent from the function. The fix
was reverting to the sandwich norm and cutting the peak learning rate by 10×.

Both failure modes are reproducible here as arms (`prenorm`, `nonormparams`, `hi_lr`), and
the second one turns out to matter a great deal for the 4B retrofit — see §4.

## 3. The training objective

### 3.1 `r` is sampled, not fixed

If the model is trained at one depth it will only work at that depth. So a depth is drawn
per micro-batch from a log-normal Poisson:

```
tau ~ N(log(rbar) − sigma^2/2, sigma)
r   ~ Poisson(e^tau) + 1
```

The `+1` guarantees at least one turn. The `−sigma^2/2` correction makes the mean come out
at `rbar`. The distribution is deliberately heavy-tailed: most batches see fewer than
`rbar` turns, and a few see many more, which is what teaches the model to remain sensible
at depths it rarely visits.

> The paper writes "a variance that we set to `sigma = 1/2`", which is ambiguous — is
> `sigma` the standard deviation or the variance? It changes the distribution. Only the
> standard-deviation reading reproduces the moments printed on the paper's own Figure 3
> (mean 33.0, median 29.0 at `rbar = 32`); we get 32.98 and 29.0. `sampling.py` asserts
> this, so the ambiguity is pinned rather than assumed.

### 3.2 Truncated backpropagation, and why memory is the point

Backpropagating through 32 turns of a 4-layer block means storing activations for 128
layers. That is not affordable, and worse, it makes memory a function of `r` — so the
heavy tail of the sampling distribution would blow up the largest batch.

So only the last `k` turns are differentiated. The earlier ones run with gradients off and
the state is detached. Activation memory then depends on `k` alone.

This is not a hand-wave; it is directly measurable, and we measured it:

| model | r | k | peak memory |
|---|---|---|---|
| Track A, h=512, B=32 | 8 | 4 | 4.78 GB |
| Track A, h=512, B=32 | **32** | 4 | **4.78 GB** |
| Track A, h=512, B=32 | 1 | 1 | 2.32 GB |
| Track B, Qwen3-4B, B=1 | 4 | 2 | 10.22 GB |
| Track B, Qwen3-4B, B=1 | **8** | 2 | **10.22 GB** |

Identical at both depths, at both scales. Memory tracks `k`, never `r`.

One subtlety worth stating because it is easy to get wrong: `e` is computed **outside** the
no-gradient region. The *state chain* is truncated, but the *injection* of `e` into each of
the last `k` turns is not, so the prelude receives `k` separate gradient contributions per
forward pass rather than one.

## 4. What this repository does

Two tracks, because 3.5B parameters × 800B tokens is roughly 10⁶ times our budget.

### Track A — the architecture from scratch

The paper's own small shape `(1, 4, 1)`, byte-level, ~19.6M parameters. Small enough to
train many arms and ablations, at the same scale the paper used to make these same design
decisions.

The substrate is the part worth explaining. Running ARC or GSM8K against a model this size
measures noise around chance. So instead the tasks are built so that **the sequential depth
a problem requires is a knob we set**:

| task | what it needs | knob |
|---|---|---|
| `perm` | Compose `n` permutations of 5 elements. The running state after `k` symbols depends on all `k`, and the group is non-abelian so they cannot be reordered. S₅ is the smallest symmetric group that is not solvable, which makes its word problem NC¹-complete — a constant-depth transformer computes only TC⁰ functions, so it *provably cannot* solve this for growing `n` unless those classes collapse. A model that can iterate can. | `n` |
| `add` | Multi-operand addition — the paper's own App. A.1 study. More operands chains more carries. | operands × digits |
| `recall` | Find one key among many, report its value. Needs attention and memory; one induction head does it in a single pass. Depth should buy **nothing**. | pairs |

`recall` is not filler. Without a task where recurrence should *not* help, "recurrence
helps" is unfalsifiable — everything improves with training. It is the control.

Arms, each named for the claim it exists to test: `rec` (the model), `fixed1` (the paper's
non-recurrent twin), `fixed1_flop` (that twin given `rbar`× more steps so the two match in
FLOPs — a control the paper does not run), `fixedr`, `noinject`, `prenorm`, `dets0`,
`hi_lr`, `fullbp`.

### Track B — the 4B

`Qwen3-4B-Base`: 36 layers, hidden 2560, GQA 32/8, vocabulary 151936. Cut into `(9, 18, 9)`
— the paper's 25/50/25 proportions — the middle 18 layers looped, the concat adapter
installed in front of them, and the random-`r` objective applied with LoRA on the core.

This is not a detour from the paper. §6.3 states the distinction explicitly: *"the main
distinction between both approaches is whether to pretrain from scratch for recurrence, or
whether to finetune existing fixed-depth models to have this capability"*, and §9 lists
post-training schemes as future work. It is the paper's own question.

**The initialisation is the experiment.** Set the adapter to `A = [0 | I]` — zero on the
state half, identity on the embedding half — and the adapter returns `e` and ignores the
state. Then at `r = 1` the surgery computes *exactly* what the original network computed.
Measured:

| | value |
|---|---|
| base Qwen3-4B-Base loss | 0.705967 |
| retrofit at `r=1`, core norm removed | **0.705967** |
| max absolute logit difference | **0.00e+00** |
| argmax agreement | **1.0000** |
| cost of adding the core RMSNorm | +0.023 nats, 94.9% agreement |
| **paper-style random adapter, untrained** | **11.32** |

The cut is bit-exact, so anything the r-curve gains later, it gained from recurrence. And
the last row is why the identity initialisation is interesting rather than merely
convenient: the faithful random adapter destroys the pretrained function outright.

But the identity initialisation starts life *inside* the failure mode of §4.3's second
run — the adapter literally ignores the state, so the untrained loss is flat across every
`r`. Whether training escapes that basin is a real question, and it is run against
`--adapter-init paper` as the contrast arm.

At `r = 32` this 4B model executes **61.3B materialized parameters**.

## 5. The measurement is the hard part

On a resized reproduction, the evaluation breaks more often than the model does. It broke
three times here before anything worked. In full, with numbers, in
[REPORT.md §4](REPORT.md). In short:

**The validation loss was flat in `r` while accuracy was climbing.** The first C1
measurement read `2.313, 2.298, 2.298, 2.298 …` — an apparent refutation of the paper's
central claim. On the same checkpoint, task accuracy read `0.16, 0.23, 0.31`. The loss was
the broken instrument: most bytes in a packed stream are prompts — random operands, random
symbols, random keys — which are unpredictable no matter how well the model reasons, and
averaging over them buries the few answer positions where reasoning shows up. Stopping at
that first plot would have reported "does not reproduce".

**Half the task cells were solvable by table lookup.** With 5 generators, `perm/2` has 25
possible prompts and every held-out item appears in training. Rather than guess which cells
were safe, the gate now *computes* each cell's prompt space and classifies it — and the
measured leak fraction tracks the computed space exactly (1.00 at 25 prompts, 0.73 at 10³,
0.00 above 10⁶). Eight of seventeen cells can carry a reasoning claim; the rest are kept as
the genuinely easy end of the axis.

**The KV cache double-counted its own prefill.** A later recurrence step read the slot an
earlier step had written *during the same block*, so prompt tokens were attended to twice.
The regression test that catches it is now the load-bearing one: cached decoding must equal
the teacher-forced forward (it does, to 1.4e-06).

Everything is reported against a floor. `scripts/check_eval.py` refuses to run the sweep if
the instrument isn't fit.

## 6. Repository map

```
recurrent_depth/
  config.py       the (lP, lR, lC) triplet and every ablation switch
  layers.py       sandwich-norm block, RoPE attention with learnable q/k biases, SwiGLU
  model.py        prelude / core / coda, the concat adapter, truncated backprop through k
  sampling.py     the log-normal Poisson, pinned against the paper's Figure 3
  init.py         Takase init, sigma_h^2 = 2/(5h), sigma_out^2 = 1/(5hl), the s0 scale
  diagnostics.py  token correlation, the App. A.2 recurrence statistics, path independence
  inference.py    Sec. 6: KL adaptive exit, KV-cache sharing, continuous-CoT warm start
  evaluate.py     strict exact match, answer-only loss, val loss vs r
  retrofit.py     Track B: the surgery on a pretrained HF model, plus a minimal LoRA
data/tasks.py     the depth-controlled suite and its tabulability model
scripts/
  smoke.py            21 assertions on the paper's RULES, not tensor shapes
  check_eval.py       Phase-4 gate: ceiling, floor, tabulability, leakage
  train_scratch.py    Track A, one arm at a time
  run_sweep.py        the arm queue, one process per GPU
  verify_retrofit.py  Phase-4 gate for the 4B surgery
  train_retrofit.py   Track B
  analyze.py          tables and figures for C1–C5
  analyze_mechanisms.py   C6, C7 and the Sec. 7 latent-trajectory plots
  eval_benchmarks.py  lm-eval-style multiple-choice scoring vs r
nbsrc/build_notebook.py   the notebook is generated, never hand-edited
```

Two fidelity checks in `smoke.py` are worth calling out because they cost nothing and
would catch a genuinely wrong implementation:

- Building the paper's exact `(2,4,2)`, h=5280 config on the meta device reproduces
  3.56B total / 1.64B core, **and the upper x-axis of the paper's Figure 1** (3.6B, 8.3B,
  11.5B … 103B at r = 1, 4, 6 … 64) to a maximum relative error of 3.6%. A wrong adapter
  shape or a missing norm would not land on that axis.
- `var(gamma·E(x)) = 0.392` and `var(s0) = 0.388` — the paper's `sigma_s^2 = 2/5` is not a
  free constant, it is forced by the embedding scale, and the assertion says so.

## 7. Running it

```bash
python scripts/smoke.py                                  # the paper's rules, as assertions
python scripts/check_eval.py                             # gate the instrument first
python scripts/run_sweep.py --steps 3000 --gpus 0,1      # Track A, 11 arms
python scripts/analyze.py
python scripts/analyze_mechanisms.py --ckpt results/rec_s0.pt

python scripts/prep_retrofit_data.py                     # Track B corpus
python scripts/verify_retrofit.py                        # gate the 4B surgery
python scripts/train_retrofit.py --adapter-init identity --tag retro_identity
python scripts/train_retrofit.py --adapter-init paper    --tag retro_paper
python scripts/eval_benchmarks.py --ckpt results/retro_identity_trainable.pt
```

Seeds are fixed and the byte corpus is cached per (size, seed), so every arm sees an
identical stream and differs only in architecture and objective. Runs are skippable, so a
killed sweep resumes rather than restarting.

Hardware notes: T4 is sm_75, where bf16 is emulated and roughly 4× slower, so everything
runs fp16 with a GradScaler where the paper used bf16. Trainable parameters in Track B are
fp32 because a GradScaler cannot unscale fp16 gradients.

## 8. Results

**[REPORT.md](REPORT.md)** — claims stated so they can be falsified, how the reproduction
was resized with a full deviations table, what broke in the evaluation before anything
worked, results bracketed by their floors, a verdict per claim, and an exhaustive list of
what was *not* tested.

Picking the work back up on a fresh machine: **[RESUME.md](RESUME.md)** — what is
permanent, what was ephemeral, the cost of regenerating each piece, and the two
open questions stated so they can be resumed cold.

## 9. Citation

```bibtex
@article{geiping2025recurrentdepth,
  title  = {Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach},
  author = {Geiping, Jonas and McLeish, Sean and Jain, Neel and Kirchenbauer, John and
            Singh, Siddharth and Bartoldson, Brian R. and Kailkhura, Bhavya and
            Bhatele, Abhinav and Goldstein, Tom},
  journal= {arXiv preprint arXiv:2502.05171},
  year   = {2025}
}
```

Original code and model: [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining),
[tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125).
This repository is an independent reproduction and is not affiliated with the authors.
