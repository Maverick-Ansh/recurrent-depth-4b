"""PHASE-4 GATE for Track B. Run before training the 4B retrofit.

Three things must be true before the retrofit is worth training, and each has a
number attached so the report can quote it:

  1. SURGERY IS LOSSLESS.  With the identity adapter (A = [0 | I]) and a
     calibrated core_norm, the retrofit at r=1 must reproduce the base model.
     If it does not, every later result is measured against a model we broke,
     not against Qwen3-4B.

  2. THE BRACKET.  We record base-model loss (the CEILING the retrofit starts
     from) and the loss at each r BEFORE any training (the FLOOR that r-scaling
     has to beat).  With the identity adapter the untrained r-curve is flat by
     construction -- the adapter ignores s -- and that flatness is exactly the
     Sec. 4.3 "Bad Run 2" failure the training has to escape.  Reporting it up
     front stops us from later mistaking "it did not move" for "we broke it".

  3. IT FITS.  Peak memory and step time at the intended r and k on one T4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recurrent_depth.retrofit import build_retrofit
from recurrent_depth.diagnostics import token_correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--split", default="9,18,9")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--out", default="results/retrofit_gate.json")
    args = ap.parse_args()
    split = tuple(int(x) for x in args.split.split(","))

    dev = "cuda"
    torch.manual_seed(0)
    print(f"building retrofit  split={split}  from {args.model}", flush=True)
    m, tok = build_retrofit(args.model, split=split, adapter_init="identity",
                            backprop_depth=args.k, device=dev)
    base = m.hf

    text = ("The recurrent block is applied repeatedly to the latent state. "
            "Question: if a train leaves at 3pm and travels 60 km per hour for "
            "two and a half hours, how far does it go? Answer: 150 km. "
            "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n") * 8
    ids = tok(text, return_tensors="pt").input_ids[:, :args.seq].to(dev)
    ids = ids.repeat(args.batch, 1)
    x, y = ids[:, :-1], ids[:, 1:]

    report = {"model": args.model, "split": split}

    # ---------------------------------------------------------------- calibrate
    cal = m.calibrate(x)
    report["calibration"] = cal
    print(f"[calibrate] RMS(e)={cal['rms_e']:.3f}  RMS(core out)={cal['rms_core_out']:.3f}"
          f"  -> sigma_s={cal['sigma_s']:.3f}", flush=True)

    # ------------------------------------------------- 1. surgery is lossless
    with torch.no_grad():
        base_logits = base(x).logits.float()
        base_loss = torch.nn.functional.cross_entropy(
            base_logits.view(-1, base_logits.size(-1)), y.reshape(-1)).item()
        r1 = m(x, r=1, targets=y)
        r1_logits = r1["logits"].float()
        # identity adapter ignores s, so s0 must not matter at all at r=1
        r1b = m(x, r=1, targets=y, s0=torch.zeros_like(r1["state"]))

    dlogit = (base_logits - r1_logits).abs().max().item()
    scale = base_logits.std().item()
    agree = (base_logits.argmax(-1) == r1_logits.argmax(-1)).float().mean().item()
    report["surgery"] = {
        "base_loss": base_loss,
        "retrofit_r1_loss": r1["loss"].item(),
        "loss_delta": r1["loss"].item() - base_loss,
        "max_abs_logit_delta": dlogit,
        "logit_std": scale,
        "argmax_agreement": agree,
        "s0_invariance_max_delta": (r1_logits - r1b["logits"].float()).abs().max().item(),
    }
    print(f"[surgery] base loss {base_loss:.4f} -> retrofit@r=1 {r1['loss'].item():.4f} "
          f"(delta {r1['loss'].item()-base_loss:+.4f})", flush=True)
    print(f"[surgery] argmax agreement {agree:.4f}, max|dlogit| {dlogit:.3f} "
          f"(logit std {scale:.2f})", flush=True)

    # ----------------------------------------------------- 2. untrained bracket
    print("[bracket] untrained loss vs r (identity adapter -> expected flat):", flush=True)
    untrained = {}
    for r in [1, 2, 4, 8, 16]:
        with torch.no_grad():
            o = m(x, r=r, targets=y)
        untrained[r] = o["loss"].item()
        tc = token_correlation(o["state"].float())
        print(f"   r={r:2d}  loss {o['loss'].item():.4f}   token_corr {tc:+.3f}", flush=True)
    report["untrained_loss_vs_r"] = untrained

    # The paper-faithful init, for contrast. Re-initialise the adapter in place
    # rather than building a second 4B model -- two of them do not fit on a T4.
    saved = m.adapter.weight.detach().clone()
    m._init_adapter("paper")
    with torch.no_grad():
        paper_r1 = m(x, r=1, targets=y)["loss"].item()
        paper_r8 = m(x, r=8, targets=y)["loss"].item()
    m.adapter.weight.data.copy_(saved)
    m.adapter_init = "identity"
    del saved
    report["paper_init_untrained"] = {"r1": paper_r1, "r8": paper_r8}
    print(f"[bracket] paper-style random adapter, untrained: r=1 {paper_r1:.3f}, r=8 {paper_r8:.3f} "
          f"(vs base {base_loss:.3f}) -- this is what 'destroys the pretrained function' means",
          flush=True)
    with torch.no_grad():
        chk = m(x, r=1, targets=y)["loss"].item()
    assert abs(chk - r1["loss"].item()) < 1e-3, "adapter restore failed"
    torch.cuda.empty_cache()

    # -------------------------------------------------------------- 3. it fits
    n = m.add_lora(rank=args.lora_rank, where=("core",))
    stats = m.mark_trainable()
    report["lora_modules"] = n
    report["trainable"] = stats
    print(f"[fit] LoRA on {n} core linears; trainable {stats['trainable']/1e6:.1f}M "
          f"of {stats['total']/1e9:.2f}B ({stats['pct']:.3f}%)", flush=True)

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    for r in [4, 8]:
        t0 = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(x, r=r, targets=y, k=args.k)["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        mem = torch.cuda.max_memory_allocated() / 1e9
        report[f"step_r{r}"] = {"s": dt, "peak_gb": mem}
        print(f"[fit] r={r} k={args.k} B={args.batch} T={args.seq}: "
              f"{dt*1000:.0f} ms/step, peak {mem:.2f} GB", flush=True)

    report["materialized_params"] = {r: m.materialized_params(r) for r in [1, 2, 4, 8, 16, 32]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)

    # --------------------------------------------------------------- verdict
    ok = report["surgery"]["argmax_agreement"] > 0.98 and abs(report["surgery"]["loss_delta"]) < 0.05
    print("\n" + "=" * 70)
    print("VERDICT:", "surgery is lossless -- proceed" if ok else
          "SURGERY BROKE THE MODEL -- fix before training")
    print(f"  materialized params: r=1 {m.materialized_params(1)/1e9:.2f}B  "
          f"r=8 {m.materialized_params(8)/1e9:.2f}B  r=32 {m.materialized_params(32)/1e9:.2f}B")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
