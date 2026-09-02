"""Track B: train the recurrent-depth retrofit of a pretrained 4B model.

Objective is the paper's, unchanged (Sec. 3.3): plain next-token loss, one
depth r drawn per micro-batch from the log-normal Poisson Lambda, gradient
truncated to the last k recurrence steps.  What differs is that only the adapter,
the core RMSNorm and LoRA parameters inside the looped core are trainable -- the
pretrained weights are frozen.  That is the whole point: we are asking whether
recurrence can be *installed*, not whether a 4B model can be pretrained on 2xT4.

Deviations, all material and all reported:
  * ~6M tokens of retrofit training vs 800B tokens of pretraining.
  * rbar = 4 (not 32) and k = 2 (not 8), set by what fits on a 15.6 GB T4.
  * LoRA rank 16 on the core rather than full fine-tuning.
  * fp16 + GradScaler, since T4 is sm_75 (bf16 is emulated there).
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

from recurrent_depth.retrofit import build_retrofit
from recurrent_depth.sampling import sample_r
from recurrent_depth.diagnostics import token_correlation, recurrence_stats


def get_batch(tokens, B, T, device, rng):
    ix = rng.integers(0, len(tokens) - T - 1, size=B)
    x = np.stack([tokens[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def val_loss_vs_r(m, val, r_values, T, device, n_blocks=24, B=2):
    m.eval()
    n = min(n_blocks, (len(val) - 1) // T)
    X = torch.from_numpy(np.stack([val[i * T:(i + 1) * T] for i in range(n)]).astype(np.int64))
    Y = torch.from_numpy(np.stack([val[i * T + 1:(i + 1) * T + 1] for i in range(n)]).astype(np.int64))
    out = {}
    for r in r_values:
        tot, cnt = 0.0, 0
        for b in range(0, n, B):
            with torch.autocast("cuda", dtype=torch.float16):
                l = m(X[b:b + B].to(device), r=int(r), targets=Y[b:b + B].to(device))["loss"]
            tot += float(l) * len(X[b:b + B]); cnt += len(X[b:b + B])
        out[int(r)] = tot / cnt
    m.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--split", default="9,18,9")
    ap.add_argument("--adapter-init", default="identity", choices=["identity", "paper", "sum"])
    ap.add_argument("--injection", default="concat", choices=["concat", "add", "none"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--rbar", type=float, default=4.0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-where", default="core")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data_cache")
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tag = args.tag or f"retro_{args.adapter_init}_r{int(args.rbar)}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    res_path = f"{args.out}/{tag}.json"
    if os.path.exists(res_path):
        print(f"[skip] {res_path} exists")
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    split = tuple(int(x) for x in args.split.split(","))

    train = np.load(f"{args.data}/retrofit_train.npy")
    val = np.load(f"{args.data}/retrofit_val.npy")
    print(f"[{tag}] data: {len(train):,} train / {len(val):,} val tokens", flush=True)

    m, tok = build_retrofit(args.model, split=split, adapter_init=args.adapter_init,
                            injection=args.injection, backprop_depth=args.k, device=dev)
    cal_ids, _ = get_batch(val, 2, args.seq, dev, np.random.default_rng(0))
    cal = m.calibrate(cal_ids)
    print(f"[{tag}] calibrated sigma_s={cal['sigma_s']:.3f} core_rms={cal['rms_core_out']:.3f}",
          flush=True)

    n_lora = m.add_lora(rank=args.lora_rank, where=tuple(args.lora_where.split(",")))
    tstat = m.mark_trainable()
    print(f"[{tag}] LoRA on {n_lora} linears; trainable {tstat['trainable']/1e6:.1f}M "
          f"({tstat['pct']:.3f}%)", flush=True)

    # base model reference: the ceiling the retrofit must not fall below
    with torch.no_grad():
        Xv, Yv = get_batch(val, 2, args.seq, dev, np.random.default_rng(1))
        with torch.autocast("cuda", dtype=torch.float16):
            bl = torch.nn.functional.cross_entropy(
                m.hf(Xv).logits.float().view(-1, m.hf.config.vocab_size), Yv.reshape(-1)).item()
    print(f"[{tag}] base Qwen3-4B loss on this val batch: {bl:.4f}", flush=True)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator().manual_seed(args.seed + 777)

    R_EVAL = [1, 2, 4, 6, 8, 12, 16]
    hist = {"step": [], "loss": [], "r": [], "token_corr": [], "grad_norm": []}
    curves = {}
    t0 = time.time()
    m.train()

    for it in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (it + 1) / args.warmup)
        r = int(sample_r(args.rbar, 0.5, generator=gen, n=1).item())   # locked-step
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            x, y = get_batch(train, args.batch, args.seq, dev, rng)
            with torch.autocast("cuda", dtype=torch.float16):
                out = m(x, r=r, targets=y, k=args.k)
                loss = out["loss"] / args.accum
            scaler.scale(loss).backward()
            tot += loss.detach().item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()

        if it % 10 == 0 or it == args.steps - 1:
            tc = token_correlation(out["state"].detach().float())
            hist["step"].append(it); hist["loss"].append(tot)
            hist["r"].append(r); hist["token_corr"].append(tc); hist["grad_norm"].append(gn)
        if it % 50 == 0:
            print(f"[{tag}] it {it:4d}/{args.steps} loss {tot:.4f} r={r} "
                  f"tok_corr {tc:+.3f} |g| {gn:.2f} {time.time()-t0:.0f}s", flush=True)
        if (it + 1) % args.eval_every == 0 or it == args.steps - 1:
            vl = val_loss_vs_r(m, val, R_EVAL, args.seq, dev)
            curves[it + 1] = vl
            print(f"[{tag}] step {it+1} val loss vs r: " +
                  "  ".join(f"{r_}:{v:.3f}" for r_, v in vl.items()), flush=True)

    res = {
        "tag": tag, "args": vars(args), "split": split,
        "calibration": cal, "trainable": tstat, "n_lora": n_lora,
        "base_loss_ref": bl, "history": hist, "val_curves": curves,
        "wall_s": time.time() - t0,
        "tokens_seen": args.steps * args.accum * args.batch * args.seq,
        "materialized_params": {r: m.materialized_params(r) for r in R_EVAL},
    }
    json.dump(res, open(res_path, "w"), indent=1)
    torch.save({k: v for k, v in m.state_dict().items()
                if ("lora_" in k or "adapter" in k or "core_norm" in k)},
               f"{args.out}/{tag}_trainable.pt")
    print(f"[{tag}] done {res['wall_s']:.0f}s -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
