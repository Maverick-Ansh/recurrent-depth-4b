"""Track A: pretrain a recurrent-depth model from scratch, one arm at a time.

Optimizer setup follows Sec. 4.1 (verbatim):

    "We train using the Adam optimizer with decoupled weight regularization
     (beta1 = 0.9, beta2 = 0.95, eta = 5 x 10^-4) [...] We clip gradients above 1.
     We train with warm-up and a constant learning rate [...] warming up to our
     maximal learning rate within the first 4096 steps."

Sec. 4.1, locked-step sampling (verbatim):

    "we sample a single depth r for each micro-batch of training, which we
     synchronize across workers"

so one r is drawn per micro-batch, not per sequence.

Deviations from the paper, and why each still tests the claim:
  * scale     (1,4,1) h=512 rather than (2,4,2) h=5280 -- the paper's own small
              ablation shape, which is what it used to make these same choices.
  * rbar=8    rather than 32, k=4 rather than 8, so the whole r-sweep including
              extrapolation past training depth fits in the compute budget.
  * lr        the paper's 4e-5 is tuned for a 3.5B model; at h=512 that is far
              below the stable band. We set it from a short probe (--lr) and
              report the value. Sec. 4.3's *ordering* claim (too-high lr collapses
              the recurrence) is reproduced as its own arm, not assumed.
  * data      byte-level depth-controlled tasks + optional text, not 800B tokens
              of web/code/math. See data/tasks.py for why.
  * fp16      T4s are sm_75: bf16 is emulated and ~4x slower, so we use fp16 with
              a GradScaler where the paper used bf16 on MI250X.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import tasks
from recurrent_depth.config import RecurrentDepthConfig
from recurrent_depth.model import RecurrentDepthLM
from recurrent_depth.sampling import sample_r
from recurrent_depth.diagnostics import token_correlation, recurrence_stats
from recurrent_depth.evaluate import sweep, val_loss_vs_r, answer_loss_vs_r

# --------------------------------------------------------------------- the arms
# Each arm names the claim it exists to test.
ARMS = {
    # main model: random-r unrolling, truncated backprop, full architecture
    "rec":            dict(claim="C1/C6", cfg={}, train_r="random"),
    # Table 4's non-recurrent twin: same architecture, one pass through the core
    "fixed1":         dict(claim="C2", cfg={}, train_r=1),
    # our addition: the twin given rbar x MORE STEPS so the two match in FLOPs.
    # The paper's Table 4 is token-matched, which lets the recurrent model spend
    # ~32x the compute; this arm is the control that objection demands.
    "fixed1_flop":    dict(claim="C2-control", cfg={}, train_r=1, step_mult=None),
    # trained at a single fixed depth: isolates *recurrence* from *random unrolling*
    "fixedr":         dict(claim="C1-mechanism", cfg={}, train_r="fixed"),
    # Sec. 3.1: input injection every step. "none" => s0 carries e, never re-injected
    "noinject":       dict(claim="C5a", cfg=dict(injection="none"), train_r="random"),
    "addinject":      dict(claim="C5a", cfg=dict(injection="add"), train_r="random"),
    # Sec. 4.3 Bad Run 2: pre-norm instead of the sandwich ordering
    "prenorm":        dict(claim="C5b", cfg=dict(norm_style="pre"), train_r="random"),
    # Sec. 3.1: random s0 promotes path independence
    "dets0":          dict(claim="C5c", cfg=dict(random_s0=False), train_r="random"),
    # Sec. 4.3 Bad Run 1: parameter-free norms
    "nonormparams":   dict(claim="C5b", cfg=dict(norm_affine=False), train_r="random"),
    # Sec. 3.3: k = r, i.e. full backprop through the unrolling
    "fullbp":         dict(claim="C4", cfg={}, train_r="random", full_bp=True),
    # Sec. 4.3: the peak learning rate is what separated their Bad Run 2 from the
    # main run ("dropping the peak learning rate to 4e-5"). Same arm as `rec`,
    # 3.3x the lr, to see whether the collapse ordering reproduces at small scale.
    "hi_lr":          dict(claim="C5-lr", cfg={}, train_r="random", lr_mult=10/3),
}


def build_data(cfg_tokens: int, seed: int, text_path: str | None, text_frac: float,
               cache_dir: str = "data_cache"):
    """Generate the byte corpus once per (size, seed, text) and cache it.

    Building it is a pure-Python loop over ~1M examples; every arm draws from the
    SAME stream, so caching is both faster and a stronger control -- the arms
    differ only in architecture and objective, never in the data they saw.
    """
    os.makedirs(cache_dir, exist_ok=True)
    stem = f"{cache_dir}/trackA_{cfg_tokens}_{seed}_{'t' if text_path else 'n'}{text_frac}"
    tr_p, va_p = stem + "_train.npy", stem + "_val.npy"
    if os.path.exists(tr_p) and os.path.exists(va_p):
        return np.load(tr_p), np.load(va_p)
    text = np.load(text_path) if (text_path and os.path.exists(text_path)) else None
    tr = tasks.build_train_corpus(cfg_tokens, seed=seed, text=text, text_frac=text_frac)
    va = tasks.build_train_corpus(300_000, seed=10_000 + seed, text=text, text_frac=text_frac)
    np.save(tr_p, tr.tokens); np.save(va_p, va.tokens)
    return tr.tokens, va.tokens


def get_batch(tokens, batch_size, block, device, rng):
    ix = rng.integers(0, len(tokens) - block - 1, size=batch_size)
    x = np.stack([tokens[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--rbar", type=float, default=8.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--train-tokens", type=int, default=25_000_000)
    p.add_argument("--text", type=str, default="")
    p.add_argument("--text-frac", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    spec = ARMS[args.arm]
    tag = f"{args.arm}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(res_path):
        print(f"[skip] {res_path} exists")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device

    steps = args.steps * (int(args.rbar) if args.arm == "fixed1_flop" else 1)
    lr = args.lr * spec.get("lr_mult", 1.0)

    cfg = RecurrentDepthConfig(
        vocab_size=tasks.VOCAB_SIZE, hidden=args.hidden, n_heads=args.heads,
        mlp_inner=int(args.hidden * 2.6875) // 32 * 32,
        l_prelude=1, l_core=4, l_coda=1,
        mean_recurrence=args.rbar, backprop_depth=args.k, max_seq=args.block + 8,
        **spec["cfg"])
    model = RecurrentDepthLM(cfg).to(dev)
    if args.compile:
        model = torch.compile(model)

    train_tok, val_tok = build_data(args.train_tokens, args.seed,
                                    args.text or None, args.text_frac)

    # Sec. 4.1 optimizer: Adam, decoupled weight decay, betas (0.9, 0.95), clip 1.0
    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    nodecay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda")      # fp16 on sm_75, see module docstring

    gen = torch.Generator().manual_seed(args.seed + 777)
    hist = {"step": [], "loss": [], "r": [], "token_corr": [], "rel_step": [], "lr": []}
    t0 = time.time()

    def lr_at(it):
        # "warm-up and a constant learning rate" (Sec. 4.1)
        return lr * min(1.0, (it + 1) / args.warmup)

    for it in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        # locked-step sampling: ONE r for the whole micro-batch
        if spec["train_r"] == "random":
            r = int(sample_r(args.rbar, cfg.sigma, generator=gen, n=1).item())
        elif spec["train_r"] == "fixed":
            r = int(args.rbar)
        else:
            r = int(spec["train_r"])
        k = r if spec.get("full_bp") else args.k

        x, y = get_batch(train_tok, args.batch_size, args.block, dev, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, r=r, targets=y, k=k)
            loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # "clip gradients above 1"
        scaler.step(opt)
        scaler.update()

        if it % 50 == 0 or it == steps - 1:
            with torch.no_grad():
                tc = token_correlation(out["state"].detach().float())
            lv = loss.detach().item()
            hist["step"].append(it); hist["loss"].append(lv)
            hist["r"].append(r); hist["token_corr"].append(tc)
            hist["rel_step"].append(float("nan")); hist["lr"].append(lr_at(it))
            if it % 500 == 0:
                el = time.time() - t0
                print(f"[{tag}] it {it:5d}/{steps}  loss {lv:.4f}  r={r:2d}  "
                      f"tok_corr {tc:+.3f}  {el:.0f}s", flush=True)
                if not math.isfinite(lv):
                    print(f"[{tag}] DIVERGED at it {it}", flush=True)
                    break

    # ------------------------------------------------------------------- eval
    r_values = [1, 2, 4, 8, 12, 16, 24, 32, 48]
    model.eval()
    grid = sweep(model, r_values, device=dev, n_per_cell=256, amp_dtype=torch.float16)
    vl = val_loss_vs_r(model, val_tok, r_values, block=args.block, n_blocks=48,
                       device=dev, amp_dtype=torch.float16)
    al = answer_loss_vs_r(model, r_values, device=dev, amp_dtype=torch.float16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        probe = torch.from_numpy(np.stack(
            [val_tok[i * args.block:(i + 1) * args.block] for i in range(4)]
        ).astype(np.int64)).to(dev)
        traj = model.trajectory(probe, r=48)
        stats = recurrence_stats(traj.float())

    result = {
        "arm": args.arm, "claim": spec["claim"], "seed": args.seed,
        "steps": steps, "lr": lr, "rbar": args.rbar, "k": args.k,
        "config": cfg.to_dict(), "n_params": model.n_params(),
        "tokens_seen": steps * args.batch_size * args.block,
        "wall_s": time.time() - t0,
        "history": hist, "val_loss_vs_r": vl, "answer_loss_vs_r": al, "grid": grid,
        "recurrence_stats": stats,
        "materialized_params": {int(r): model.materialized_params(int(r)) for r in r_values},
    }
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    torch.save({"cfg": cfg.to_dict(), "state_dict": model.state_dict()},
               os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] done in {result['wall_s']:.0f}s -> {res_path}")
    print(f"[{tag}] val loss by r: " + "  ".join(f"{r}:{v:.3f}" for r, v in vl.items()))
    for t, d_ in al.items():
        print(f"[{tag}] ANSWER loss {t:<7}: " + "  ".join(f"{r}:{v:.3f}" for r, v in d_.items()))


if __name__ == "__main__":
    main()
