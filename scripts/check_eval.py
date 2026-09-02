"""PHASE-4 GATE. Run this BEFORE spending any GPU time on the sweep.

The lesson this file exists for: on a resized reproduction the *evaluation*
breaks more often than the model does.  A task whose best constant guess already
scores 0.9 cannot show you a recurrence effect; a grader that scores 0.0 on the
gold answers will make every arm look equally dead.  So we measure the bracket
each cell lives inside first, and refuse the sweep if any cell is unusable.

Prints a VERDICT and exits non-zero if the instrument is not fit to run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import tasks
from recurrent_depth.evaluate import exact_match, _pack

MAX_FLOOR = 0.35        # a cell whose constant-guess baseline beats this has no headroom
N_ITEMS = 256

problems, rows = [], []


class GoldModel(torch.nn.Module):
    """CEILING probe: a 'model' that reads the gold answer off the input.

    Because the target sits in the teacher-forced input at position i-1..i, an
    oracle that copies input[t+1] scores 1.0 iff the grader is correct.  If this
    does not read 1.0, the bug is in `exact_match`, not in any trained model.
    """

    def forward(self, idx, r=1, targets=None, **kw):
        V = tasks.VOCAB_SIZE
        oracle = torch.nn.functional.one_hot(self._gold[:, 1:idx.shape[1] + 1], V).float() * 30.0
        return {"logits": oracle, "loss": None}

    def eval(self):
        return self


class FloorModel(torch.nn.Module):
    """FLOOR probe: a no-information model that always predicts token 0."""

    def forward(self, idx, r=1, targets=None, **kw):
        lg = torch.zeros(idx.shape[0], idx.shape[1], tasks.VOCAB_SIZE)
        lg[..., 0] = 10.0
        return {"logits": lg, "loss": None}

    def eval(self):
        return self


print("=" * 78)
print("PHASE-4 EVALUATION GATE  --  bracketing every cell before the sweep")
print("=" * 78)

# ---------------------------------------------------------------- grader ceiling
print("\n[1] Grader ceiling: an oracle that reads the gold answer must score 1.000")
gold_ok = True
for task in tasks.TASKS:
    for level in tasks.levels_for(task):
        items = tasks.build_eval_set(task, level, n=64, seed=1234)
        ids, _ = _pack(items)
        gm = GoldModel()
        gm._gold = ids
        acc = exact_match(gm, items, r=1, device="cpu", batch_size=64)
        if acc < 0.999:
            gold_ok = False
            problems.append(f"grader scores {acc:.3f} on GOLD answers for {task}/{level}")
print(f"    {'PASS' if gold_ok else 'FAIL'} -- grader recovers gold answers on all cells")

# ------------------------------------------------------------------ cell floors
print("\n[2] Per-cell floor: best constant policy, and a no-information model")
print(f"    {'cell':<18} {'const-guess':>11} {'zero-model':>11} {'headroom':>9}  status")
for task in tasks.TASKS:
    for level in tasks.levels_for(task):
        items = tasks.build_eval_set(task, level, n=N_ITEMS, seed=1234)
        floor = tasks.shortcut_baseline(task, level)
        zero = exact_match(FloorModel(), items, r=1, device="cpu", batch_size=64)
        head = 1.0 - floor
        bad = floor > MAX_FLOOR
        if bad:
            problems.append(f"{task}/{level} floor {floor:.3f} > {MAX_FLOOR} (no headroom)")
        rows.append((f"{task}/{level}", floor, zero, head))
        print(f"    {task+'/'+str(level):<18} {floor:>11.3f} {zero:>11.3f} {head:>9.3f}"
              f"  {'TOO EASY' if bad else 'ok'}")

# ------------------------------------------------- difficulty is actually ordered
print("\n[3] Difficulty ordering: token-level entropy of the answer must rise with level")
for task in tasks.TASKS:
    ents = []
    for level in tasks.levels_for(task):
        f = tasks.shortcut_baseline(task, level, n=4000)
        ents.append((str(level), f))
    print(f"    {task:<8} const-guess by level: " +
          "  ".join(f"{lv}={f:.3f}" for lv, f in ents))

# --------------------------------------------------- eval set is not memorisable
print("\n[4] Held-out check: eval prompts must not appear in the training stream")
train = tasks.build_train_corpus(400_000, seed=0, text=None)
train_str = train.tokens.tobytes()
leaks = 0
for task in tasks.TASKS:
    for level in tasks.levels_for(task):
        for prompt, _ in tasks.build_eval_set(task, level, n=32, seed=1234)[:32]:
            if len(prompt) >= 8 and np.array(prompt[1:], dtype=np.uint16).tobytes() in train_str:
                leaks += 1
print(f"    {leaks} of {sum(32 for t in tasks.TASKS for _ in tasks.levels_for(t))} "
      f"eval prompts also occur verbatim in a 400k-token training stream")
if leaks > 5:
    problems.append(f"{leaks} eval prompts leak into the training distribution")

# --------------------------------------------------------------------- verdict
print("\n" + "=" * 78)
if problems:
    print("VERDICT: DO NOT RUN THE SWEEP")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
print("VERDICT: instrument is fit for the sweep.")
print(f"  {len(rows)} cells, floors in [{min(r[1] for r in rows):.3f},"
      f" {max(r[1] for r in rows):.3f}], grader ceiling 1.000 everywhere.")
print("  Every accuracy in the report must be read against its cell's floor above.")
