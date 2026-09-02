"""Scoring for the Track-A task suite.

Metric: strict exact match under teacher forcing.  For an item (prompt, answer)
we feed [prompt || answer || EOS] once and require the model's argmax to be
correct at EVERY answer position AND at the terminating EOS.  Requiring the EOS
closes the loophole where a model that emits "1275" for the gold "127" would
otherwise score a partial credit; requiring all positions makes this a strict
upper bound on free-running greedy exact match (they differ only on items the
model already gets wrong).  One forward pass per batch, no generation -- which
is what makes a 9-point r-sweep across 16 difficulty cells affordable.

We report, next to every cell, the score of the best constant policy
(`data.tasks.shortcut_baseline`), so no number in the report floats free of its
floor.
"""

from __future__ import annotations

import torch
import numpy as np

from data import tasks


def _pack(items, pad=tasks.PAD):
    """items: list of (prompt_ids, answer_ids). Returns padded ids + answer mask."""
    seqs, masks = [], []
    for prompt, answer in items:
        full = prompt + answer + [tasks.EOS]
        seqs.append(full)
        m = [0] * len(prompt) + [1] * (len(answer) + 1)
        masks.append(m)
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad, dtype=np.int64)
    msk = np.zeros((len(seqs), L), dtype=bool)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = s
        msk[i, :len(m)] = m
    return torch.from_numpy(ids), torch.from_numpy(msk)


@torch.no_grad()
def exact_match(model, items, r: int, device="cuda", batch_size: int = 64, amp_dtype=None):
    """Fraction of items where every answer token (and the EOS) is argmax-correct."""
    model.eval()
    correct = 0
    for b in range(0, len(items), batch_size):
        ids, mask = _pack(items[b:b + batch_size])
        ids, mask = ids.to(device), mask.to(device)
        ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
               else torch.autocast("cuda", enabled=False))
        with ctx:
            logits = model(ids[:, :-1], r=r)["logits"]
        pred = logits.float().argmax(-1)                    # predicts ids[:, 1:]
        tgt, m = ids[:, 1:], mask[:, 1:]
        ok = ((pred == tgt) | ~m).all(dim=1)                # all masked positions right
        correct += int(ok.sum())
    return correct / len(items)


@torch.no_grad()
def sweep(model, r_values, device="cuda", n_per_cell: int = 256, seed: int = 1234,
          amp_dtype=None, batch_size: int = 64):
    """Full (task x difficulty x r) accuracy grid, plus each cell's shortcut floor."""
    out = {}
    for task in tasks.TASKS:
        for level in tasks.levels_for(task):
            items = tasks.build_eval_set(task, level, n=n_per_cell, seed=seed)
            key = f"{task}/{level}"
            out[key] = {
                "floor": tasks.shortcut_baseline(task, level),
                "acc": {int(r): exact_match(model, items, r, device, batch_size, amp_dtype)
                        for r in r_values},
            }
    return out


@torch.no_grad()
def val_loss_vs_r(model, val_tokens: np.ndarray, r_values, block: int = 256,
                  n_blocks: int = 32, device="cuda", amp_dtype=None, batch_size: int = 8):
    """Held-out next-token loss at each recurrence depth -- the paper's Fig. 6 right panel.

    Fig. 6: "Plot of val ppl at recurrent depths 1, 4, 8, 16, 32, 64. During
    training, the model improves in perplexity on all levels of recurrence."
    """
    model.eval()
    n = min(n_blocks, (len(val_tokens) - 1) // block)
    xs = np.stack([val_tokens[i * block:(i + 1) * block] for i in range(n)]).astype(np.int64)
    ys = np.stack([val_tokens[i * block + 1:(i + 1) * block + 1] for i in range(n)]).astype(np.int64)
    X, Y = torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)
    res = {}
    for r in r_values:
        tot, cnt = 0.0, 0
        for b in range(0, len(X), batch_size):
            ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
                   else torch.autocast("cuda", enabled=False))
            with ctx:
                loss = model(X[b:b + batch_size], r=int(r), targets=Y[b:b + batch_size])["loss"]
            tot += float(loss) * len(X[b:b + batch_size])
            cnt += len(X[b:b + batch_size])
        res[int(r)] = tot / cnt
    return res
