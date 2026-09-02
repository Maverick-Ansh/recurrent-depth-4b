"""Tokenise a small corpus for the Track-B retrofit and cache it as .npy.

The paper's mixture is "heavily skewed towards code and mathematical reasoning
data with (hopefully) just enough general webtext" (Sec. 4.1).  We cannot
reproduce 800B tokens, and we are not trying to: the retrofit's job is to teach
an already-pretrained model to USE a recurrence it did not have, not to teach it
language.  So we use a small slice of FineWeb-Edu plus open math/code text, in
roughly the paper's proportions, and report the token count honestly.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--tokens", type=int, default=6_000_000)
    ap.add_argument("--val-tokens", type=int, default=250_000)
    ap.add_argument("--out", default="data_cache")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)

    # (dataset, config, split, text field, share of the mixture)
    SOURCES = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.45),
        ("open-web-math/open-web-math", None, "train", "text", 0.35),
        ("bigcode/the-stack-smol", None, "train", "content", 0.20),
    ]

    buf: list[int] = []
    per_source = {}
    target_total = args.tokens + args.val_tokens
    for name, cfg, split, field, share in SOURCES:
        want = int(target_total * share)
        got = 0
        try:
            ds = load_dataset(name, cfg, split=split, streaming=True)
        except Exception as e:                        # a source may be gated or moved
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            per_source[name] = 0
            continue
        for row in ds:
            t = row.get(field) or ""
            if not t:
                continue
            ids = tok(t, add_special_tokens=False).input_ids
            buf.extend(ids)
            got += len(ids)
            if got >= want:
                break
        per_source[name] = got
        print(f"[ok] {name}: {got:,} tokens", flush=True)

    arr = np.array(buf, dtype=np.int32)
    rng = np.random.default_rng(0)
    # shuffle in 2048-token chunks so the mixture is interleaved, not blocked
    ch = 2048
    n = len(arr) // ch
    idx = rng.permutation(n)
    arr = arr[:n * ch].reshape(n, ch)[idx].reshape(-1)
    val, train = arr[:args.val_tokens], arr[args.val_tokens:]
    np.save(f"{args.out}/retrofit_train.npy", train)
    np.save(f"{args.out}/retrofit_val.npy", val)
    print(f"train {len(train):,} tokens, val {len(val):,} tokens -> {args.out}")
    print("per-source:", per_source)


if __name__ == "__main__":
    main()
