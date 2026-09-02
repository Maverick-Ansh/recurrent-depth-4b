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


def _pack(items, pad=tasks.PAD):                                                                    # +-- STRICT EXACT MATCH IN ONE FORWARD PASS ---------------------
    """items: list of (prompt_ids, answer_ids). Returns padded ids + answer mask."""                # | The prompt, the answer and a terminating end marker are laid
    seqs, masks = [], []                                                                            # | out as one sequence and a mask marks which positions are the
    for prompt, answer in items:                                                                    # | answer. The model sees the whole thing once and its prediction
        full = prompt + answer + [tasks.EOS]                                                        # | at each position is compared to what actually follows. An item
        seqs.append(full)                                                                           # | counts only if every masked position is right, the end marker
        m = [0] * len(prompt) + [1] * (len(answer) + 1)                                             # | included. Requiring the end marker closes a real loophole: a
        masks.append(m)                                                                             # | model emitting 1275 where the answer was 127 would otherwise
    L = max(len(s) for s in seqs)                                                                   # | score correct on all three digits it did produce. Requiring
    ids = np.full((len(seqs), L), pad, dtype=np.int64)                                              # | all positions makes this an upper bound on free-running greedy
    msk = np.zeros((len(seqs), L), dtype=bool)                                                      # | decoding rather than a different metric, since the two can
    for i, (s, m) in enumerate(zip(seqs, masks)):                                                   # | only diverge on items the model already gets wrong. Scoring
        ids[i, :len(s)] = s                                                                         # | this way costs one forward pass per batch instead of one per
        msk[i, :len(m)] = m                                                                         # | generated token, which is what makes a nine-point depth sweep
    return torch.from_numpy(ids), torch.from_numpy(msk)                                             # | across seventeen difficulty cells affordable at all. Padding
                                                                                                    # | goes at the end of each sequence, and attention is causal, so
                                                                                                    # | a real position never attends to a pad.
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


@torch.no_grad()                                                                                    # +-- THE INSTRUMENT THAT HAS THE EFFECT IN RANGE ----------------
def sweep(model, r_values, device="cuda", n_per_cell: int = 256, seed: int = 1234,                  # | sweep walks every task and difficulty and carries each cell's
          amp_dtype=None, batch_size: int = 64):                                                    # | constant-guess floor alongside its accuracy, so no number is
    """Full (task x difficulty x r) accuracy grid, plus each cell's shortcut floor."""              # | ever read in isolation. answer_loss_vs_r exists because the
    out = {}                                                                                        # | first attempt at measuring the central claim used loss over
    for task in tasks.TASKS:                                                                        # | the whole packed byte stream and read flat in r while accuracy
        for level in tasks.levels_for(task):                                                        # | was visibly climbing. The reason is that the stream is mostly
            items = tasks.build_eval_set(task, level, n=n_per_cell, seed=seed)                      # | prompts, and prompts here are random operands, random
            key = f"{task}/{level}"                                                                 # | generator symbols and random keys, which cannot be predicted
            out[key] = {                                                                            # | however well the model reasons. Averaging over them buries the
                "floor": tasks.shortcut_baseline(task, level),                                      # | handful of answer positions that are the only place reasoning
                "acc": {int(r): exact_match(model, items, r, device, batch_size, amp_dtype)         # | can show up. Restricting the loss to answer tokens puts the
                        for r in r_values},                                                         # | effect back in range. The full-stream version is kept anyway,
            }                                                                                       # | because its flatness against a rising accuracy curve is itself
    return out                                                                                      # | the finding.


@torch.no_grad()
def answer_loss_vs_r(model, r_values, device="cuda", n_per_task: int = 128,
                     seed: int = 4321, amp_dtype=None, batch_size: int = 32):
    """Held-out loss on ANSWER tokens only, per task, at each recurrence depth.

    Why this exists: our first measurement was loss over the whole packed byte
    stream, and it was flat in r while task accuracy was clearly rising.  The
    reason is that most bytes in the stream are the *prompts* -- random operands,
    random generator symbols, random key/value pairs -- which are irreducibly
    unpredictable.  Averaging over them buries the answer tokens, which are the
    only positions where reasoning can show up.  Loss restricted to answer
    positions is the instrument that actually has the effect in range.
    """
    model.eval()
    out = {}
    for task in tasks.TASKS:
        items = []
        for level in tasks.levels_for(task):
            items += tasks.build_eval_set(task, level, n=n_per_task, seed=seed)
        per_r = {}
        for r in r_values:
            tot, cnt = 0.0, 0
            for b in range(0, len(items), batch_size):
                ids, mask = _pack(items[b:b + batch_size])
                ids, mask = ids.to(device), mask.to(device)
                ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
                       else torch.autocast("cuda", enabled=False))
                with ctx:
                    logits = model(ids[:, :-1], r=int(r))["logits"]
                lp = torch.log_softmax(logits.float(), -1)
                tgt, m = ids[:, 1:], mask[:, 1:]
                nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                tot += float((nll * m).sum())
                cnt += int(m.sum())
            per_r[int(r)] = tot / max(cnt, 1)
        out[task] = per_r
    return out


@torch.no_grad()                                                                                    # +-- THE PERPLEXITY-AGAINST-DEPTH CURVE -------------------------
def val_loss_vs_r(model, val_tokens: np.ndarray, r_values, block: int = 256,                        # | This is the paper's own Figure 6 measurement: held-out loss
                  n_blocks: int = 32, device="cuda", amp_dtype=None, batch_size: int = 8):          # | evaluated at each recurrence depth, on a model trained with
    """Held-out next-token loss at each recurrence depth -- the paper's Fig. 6 right panel.

    Fig. 6: "Plot of val ppl at recurrent depths 1, 4, 8, 16, 32, 64. During
    training, the model improves in perplexity on all levels of recurrence."
    """
    model.eval()                                                                                    # | depths drawn at random. Blocks are cut from a fixed held-out
    n = min(n_blocks, (len(val_tokens) - 1) // block)                                               # | stream at fixed offsets, so the same text is scored at every
    xs = np.stack([val_tokens[i * block:(i + 1) * block] for i in range(n)]).astype(np.int64)       # | depth and the only thing varying is r. It reports whatever the
    ys = np.stack([val_tokens[i * block + 1:(i + 1) * block + 1] for i in range(n)]).astype(np.int64)
    X, Y = torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)                         # | packed stream contains, prompts included, which is exactly why
    res = {}                                                                                        # | the answer-only version above exists next to it rather than
    for r in r_values:                                                                              # | instead of it.
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
