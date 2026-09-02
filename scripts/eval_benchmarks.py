"""Multiple-choice benchmark evaluation vs recurrence, for the 4B retrofit.

This is the paper's own headline measurement (Table 1, Fig. 1): zero-shot
lm-eval-harness tasks scored at r = 1, 4, 8, 16, 32.  We score the same way the
harness does -- sum the log-likelihood of each continuation under the model and
take the argmax -- reporting both `acc` (raw sum) and `acc_norm` (sum divided by
continuation length in characters), because the paper says "We report normalized
accuracy when provided."

Deviation: we subsample each benchmark (default 300 items).  A full ARC-C sweep
at 7 values of r on a T4 is ~45 minutes per arm; 300 items gives a standard
error of about +-2.7pp at 50% accuracy, which we report next to every number so
no difference smaller than the noise gets called a result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARKS = {
    # name: (hf path, config, split, builder)
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "sciq": ("allenai/sciq", None, "test"),
}


def load_items(name, n, seed=0):
    from datasets import load_dataset
    path, cfg, split = BENCHMARKS[name]
    ds = load_dataset(path, cfg, split=split)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n]
    items = []
    for i in idx:
        row = ds[int(i)]
        if name.startswith("arc"):
            q, ch = row["question"], row["choices"]
            opts, labels = ch["text"], ch["label"]
            gold = labels.index(row["answerKey"]) if row["answerKey"] in labels else None
            ctx = f"Question: {q}\nAnswer:"
        elif name == "openbookqa":
            q, ch = row["question_stem"], row["choices"]
            opts, labels = ch["text"], ch["label"]
            gold = labels.index(row["answerKey"]) if row["answerKey"] in labels else None
            ctx = q
        else:  # sciq
            opts = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
            gold = 3
            ctx = f"{row['support']}\nQuestion: {row['question']}\nAnswer:"
        if gold is None:
            continue
        items.append((ctx, [" " + o.strip() for o in opts], gold))
    return items


@torch.no_grad()
def score(model, tok, items, r, device="cuda", max_len=512, is_retrofit=True):
    """Returns (acc, acc_norm). One forward per (item, option)."""
    correct = correct_norm = 0
    for ctx, opts, gold in items:
        lls, lls_norm = [], []
        ctx_ids = tok(ctx, add_special_tokens=False).input_ids
        for opt in opts:
            opt_ids = tok(opt, add_special_tokens=False).input_ids
            ids = (ctx_ids + opt_ids)[-max_len:]
            n_opt = min(len(opt_ids), len(ids) - 1)
            x = torch.tensor([ids], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = (model(x[:, :-1], r=r)["logits"] if is_retrofit
                          else model(x[:, :-1]).logits)
            logp = F.log_softmax(logits.float(), -1)[0]
            tgt = x[0, 1:]
            tok_lp = logp[torch.arange(len(tgt), device=device), tgt][-n_opt:]
            lls.append(tok_lp.sum().item())
            lls_norm.append(tok_lp.sum().item() / max(len(opt), 1))
        correct += int(int(np.argmax(lls)) == gold)
        correct_norm += int(int(np.argmax(lls_norm)) == gold)
    n = len(items)
    return correct / n, correct_norm / n


def stderr(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--split", default="9,18,9")
    ap.add_argument("--ckpt", default="", help="trainable-params .pt from train_retrofit")
    ap.add_argument("--adapter-init", default="identity")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--benchmarks", default="arc_easy,arc_challenge,openbookqa")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--r", default="1,2,4,8,16")
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--out", default="results/bench.json")
    args = ap.parse_args()

    from recurrent_depth.retrofit import build_retrofit
    dev = "cuda"
    split = tuple(int(x) for x in args.split.split(","))
    m, tok = build_retrofit(args.model, split=split, adapter_init=args.adapter_init, device=dev)

    if args.base_only:
        res = {"model": "base_" + args.model, "n": args.n, "scores": {}}
        for b in args.benchmarks.split(","):
            items = load_items(b, args.n)
            a, an = score(m.hf, tok, items, r=None, device=dev, is_retrofit=False)
            res["scores"][b] = {"acc": a, "acc_norm": an, "stderr": stderr(an, len(items)),
                                "n": len(items)}
            print(f"[base] {b:<16} acc {a:.4f}  acc_norm {an:.4f} +-{stderr(an,len(items)):.4f}",
                  flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        return

    cal = m.calibrate(torch.tensor([tok("The quick brown fox jumps over the lazy dog. " * 12,
                                        add_special_tokens=False).input_ids[:256]], device=dev))
    if args.ckpt:
        m.add_lora(rank=args.lora_rank, where=("core",))
        sd = torch.load(args.ckpt, map_location=dev)
        missing = m.load_state_dict(sd, strict=False)
        print(f"[ckpt] loaded {len(sd)} tensors from {args.ckpt}; "
              f"unexpected={len(missing.unexpected_keys)}", flush=True)
    m.eval()

    res = {"model": args.model, "ckpt": args.ckpt, "split": split, "n": args.n,
           "calibration": cal, "scores": {}}
    for b in args.benchmarks.split(","):
        items = load_items(b, args.n)
        res["scores"][b] = {}
        for r in [int(x) for x in args.r.split(",")]:
            a, an = score(m, tok, items, r=r, device=dev)
            se = stderr(an, len(items))
            res["scores"][b][r] = {"acc": a, "acc_norm": an, "stderr": se, "n": len(items)}
            print(f"{b:<16} r={r:<3} acc {a:.4f}  acc_norm {an:.4f} +-{se:.4f}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
