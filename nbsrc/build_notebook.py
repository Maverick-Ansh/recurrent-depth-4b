"""Generate recurrent_depth_4b.ipynb from this file.

The notebook is a BUILD ARTIFACT, never edited by hand: that way the prose and
the code cannot drift apart and stale outputs are never committed.  Run:

    python nbsrc/build_notebook.py
"""

from __future__ import annotations

import json
import os

REPO = "https://github.com/Maverick-Ansh/recurrent-depth-4b"

CELLS: list[tuple[str, str]] = [
("md", f"""# Recurrent Depth: reproducing arXiv:2502.05171, and retrofitting it onto a 4B model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/recurrent-depth-4b/blob/main/recurrent_depth_4b.ipynb)

Geiping et al., *"Scaling up Test-Time Compute with Latent Reasoning: A Recurrent
Depth Approach"* trains a 3.5B model that iterates a shared **core block** at
test time, unrolling to arbitrary depth without emitting a single extra token.

    e  = P(x)                     prelude embeds the input
    s0 ~ N(0, sigma^2 I)          random initial latent state
    si = R(e, s_{{i-1}})            core block, run r times, e re-injected every step
    p  = C(sr)                    coda un-embeds and predicts

This notebook runs two tracks on 2x T4:

* **Track A** -- the architecture from scratch at the paper's own small shape
  `(lP, lR, lC) = (1, 4, 1)`, trained on a **depth-controlled task suite** where
  the sequential depth a problem requires is a knob with exact ground truth.
* **Track B** -- surgery on **Qwen3-4B-Base**: 36 layers cut into
  prelude / looped core / coda, the paper's concat adapter installed, and the
  random-*r* objective used to teach a pretrained model to use a recurrence it
  never had. That retrofit-vs-pretrain question is the paper's own (Sec. 6.3).

Full method and results: [`REPORT.md`]({REPO}/blob/main/REPORT.md)."""),

("code", f"""!git clone -q {REPO}.git /content/recurrent-depth-4b 2>/dev/null || (cd /content/recurrent-depth-4b && git pull -q)
%cd /content/recurrent-depth-4b
import torch, subprocess
print("torch", torch.__version__, "| gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{{i}}] {{p.name}} sm_{{p.major}}{{p.minor}} {{p.total_memory/1e9:.1f}} GB")
# T4 is sm_75: bf16 is emulated there, so everything below runs fp16 + GradScaler."""),

("md", """## 1. The paper's rules, as executable assertions

`scripts/smoke.py` asserts the *rules* of the paper rather than tensor shapes.
The one worth pointing at: we build the paper's own `(2,4,2)`, `h=5280` config on
the meta device and check that our parameter layout reproduces the 3.5B / 1.5B /
0.5B split **and the upper x-axis of the paper's Figure 1** (materialized
parameters at r = 1, 4, 6, 8, 12, 20, 32, 48, 64). If the adapter or the norms
were wired wrongly, that axis would not land."""),

("code", "!python scripts/smoke.py"),

("md", """## 2. Gate the evaluation before spending any GPU time

On a resized reproduction the *evaluation* breaks more often than the model.
This gate measures, for every (task x difficulty) cell: the grader's ceiling
against gold answers, the floor of the best constant policy, and how much of the
prompt space is small enough to be memorised rather than computed."""),

("code", "!python scripts/check_eval.py"),

("md", """## 3. Track A -- pretrain the architecture and its ablations

Twelve arms across two GPUs. Each names the claim it exists to test: the
non-recurrent twin of Table 4, a FLOP-matched version of that twin (a control
the paper does not run), input-injection off, pre-norm instead of sandwich norm,
fixed `s0`, full backprop instead of truncated, and a high-learning-rate arm
that tries to reproduce the Sec. 4.3 collapse."""),

("code", """import subprocess, sys, os
os.makedirs("logs", exist_ok=True)
p = subprocess.Popen([sys.executable, "scripts/run_sweep.py", "--steps", "2500",
                      "--lr", "3e-4", "--gpus", "0,1", "--out", "results"],
                     stdout=open("logs/sweep.log", "w"), stderr=subprocess.STDOUT,
                     env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
print("sweep pid", p.pid, "-- poll logs/sweep.log; do not block this kernel")"""),

("code", """# poll in a SHORT cell -- a long-running cell blocks the Colab kernel with no interrupt
print(open("logs/sweep.log").read()[-2000:])"""),

("code", "!python scripts/analyze.py --results results --figures figures"),

("md", """## 4. Sections 6 and 7 -- what recurrence gives you for free

Path independence, extrapolation past the training depth, the zero-shot KL
adaptive exit of Sec. 6.1, KV-cache sharing of Sec. 6.2, and the latent-space
trajectories of Sec. 7 (distance-to-limit map and PCA projections)."""),

("code", "!python scripts/analyze_mechanisms.py --ckpt results/rec_s0.pt"),

("md", """## 5. Track B -- retrofit recurrent depth onto Qwen3-4B-Base

First the gate: with the identity adapter `A = [0 | I]` the core ignores `s`, so
at `r=1` the retrofit must be **exactly** the base model. That is what makes the
later r-curve interpretable -- anything it gains, it gained from recurrence.

It also starts life inside the failure mode Sec. 4.3 describes for their second
failed run: *"the model has learned early to ignore the incoming state s"*. So
`--adapter-init paper` is run as the contrast arm."""),

("code", "!python scripts/prep_retrofit_data.py --tokens 6000000 --val-tokens 250000"),
("code", "!python scripts/verify_retrofit.py --split 9,18,9 --seq 256 --batch 2 --k 2"),

("code", """import subprocess, sys, os
ENV = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
jobs = [("0", ["--adapter-init", "identity", "--tag", "retro_identity"]),
        ("1", ["--adapter-init", "paper",    "--tag", "retro_paper"])]
for gpu, extra in jobs:
    subprocess.Popen([sys.executable, "scripts/train_retrofit.py", "--steps", "600",
                      "--rbar", "4", "--k", "2", "--lr", "1e-4"] + extra,
                     stdout=open(f"logs/{extra[-1]}.log", "w"), stderr=subprocess.STDOUT,
                     env=dict(ENV, CUDA_VISIBLE_DEVICES=gpu), start_new_session=True)
print("launched both retrofit arms")"""),

("code", """for t in ["retro_identity", "retro_paper"]:
    print(f"===== {t} =====")
    print("".join(l for l in open(f"logs/{t}.log") if l.startswith("["))[-1500:])"""),

("code", """!python scripts/eval_benchmarks.py --base-only --n 300 --out results/bench_base.json
!python scripts/eval_benchmarks.py --ckpt results/retro_identity_trainable.pt \\
    --r 1,2,4,8,16 --n 300 --out results/bench_retro_identity.json"""),

("md", f"""## 6. Results

Every claim, its verdict, what broke in the evaluation before it worked, and what
was **not** tested, are written up in [`REPORT.md`]({REPO}/blob/main/REPORT.md).

The one thing to carry away if you read nothing else: on a resized reproduction,
budget for the evaluation breaking before the model does. Three of our
measurements were wrong before any of them were right -- a val loss averaged over
irreducibly-random prompt bytes that was flat while accuracy climbed, a task
whose easy cells were memorisable rather than computable, and a KV cache whose
prefill double-counted its own tokens."""),
]


def main():
    nb = {
        "cells": [
            {"cell_type": "markdown" if k == "md" else "code",
             "metadata": {}, "source": v.splitlines(keepends=True),
             **({} if k == "md" else {"outputs": [], "execution_count": None})}
            for k, v in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "recurrent_depth_4b.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
