# recurrent-depth-4b

A from-scratch reproduction of **"Scaling up Test-Time Compute with Latent Reasoning:
A Recurrent Depth Approach"** (Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson,
Kailkhura, Bhatele, Goldstein — [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)),
plus a **retrofit of the architecture onto a pretrained 4B model** — the paper's own
open question, and the only version of it that fits on two T4s.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/recurrent-depth-4b/blob/main/recurrent_depth_4b.ipynb)

## What the paper does

A decoder-only transformer split into three functional groups — a **prelude** `P`, a
**core** `R` that is looped, and a **coda** `C`:

```
e  = P(x)                    prelude embeds the input into latent space
s0 ~ N(0, sigma_s^2 I)       random initial latent state
si = R(e, s_{i-1})           core block, applied r times, with e re-injected every step
p  = C(sr)                   coda un-embeds and predicts the next token
```

`r` is drawn per micro-batch from a log-normal Poisson during training, so at test time
the model can be run at **any** depth. The 3.5B model has 8 real layers; at `r = 32` it
unrolls to 132, executing the FLOPs of a ~52B fixed-depth transformer.

## What is here

```
recurrent_depth/
  layers.py       sandwich-norm block, RoPE attention with learnable q/k biases, SwiGLU
  model.py        prelude / core / coda, concat adapter, truncated backprop through k
  sampling.py     the log-normal Poisson Lambda, pinned against the paper's Figure 3
  init.py         Takase init, sigma_h^2 = 2/(5h), sigma_out^2 = 1/(5hl), s0 scale
  diagnostics.py  token correlation and the App. A.2 recurrence statistics
  inference.py    Sec. 6: KL adaptive exit, KV-cache sharing, continuous CoT warm start
  evaluate.py     strict exact match, answer-only loss, val loss vs r
  retrofit.py     Track B: surgery on a pretrained HF model + minimal LoRA
data/tasks.py     the depth-controlled task suite (see below)
scripts/          smoke tests, eval gates, training, sweeps, analysis
```

Every module quotes the section of the paper it implements, verbatim, and says where we
deviated.

## The two tracks

**Track A — the architecture from scratch.** The paper's own small shape `(1, 4, 1)`,
byte-level, trained on a suite where the sequential depth a problem requires is a knob
we set exactly:

| task | what it needs | difficulty knob |
|---|---|---|
| `perm` | S₅ word problem over a fixed generating set — NC¹-complete, so a constant-depth (TC⁰) transformer provably cannot do it for growing *n* | number of generators composed |
| `add` | multi-operand addition, the paper's own App. A.1 study | operands × digits |
| `recall` | single-hop associative recall — memory-hard, depth-easy. **The control**: without it, "recurrence helps" is unfalsifiable | number of key/value pairs |

**Track B — the 4B.** `Qwen3-4B-Base` (36 layers, h=2560) cut into `(9, 18, 9)`, the core
looped, the paper's concat adapter installed, trained with the random-*r* objective.
With the identity adapter `A = [0 | I]` the retrofit at `r = 1` is *exactly* the base
model, so anything the r-curve gains, it gained from recurrence.

## Running it

```bash
python scripts/smoke.py            # the paper's rules, as assertions
python scripts/check_eval.py       # gate the instrument before spending GPU time
python scripts/run_sweep.py --steps 2500 --gpus 0,1
python scripts/analyze.py
python scripts/verify_retrofit.py  # gate the 4B surgery
python scripts/train_retrofit.py --adapter-init identity
```

The notebook is generated, never hand-edited: `python nbsrc/build_notebook.py`.

## Results

See **[REPORT.md](REPORT.md)** — claims, how the reproduction was resized, what broke in
the evaluation before anything worked, results bracketed by their floors, a verdict per
claim, and an explicit list of what was *not* tested.

## Citation

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
