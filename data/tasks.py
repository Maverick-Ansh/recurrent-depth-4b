"""A depth-controlled task suite for Track A.

Why a synthetic suite instead of a shrunken lm-eval-harness
-----------------------------------------------------------
The paper's headline claim (Fig. 1, Fig. 7, Fig. 9) is that *more test-time
recurrence buys more accuracy, and harder tasks saturate later*.  Running ARC /
GSM8K against a model we can afford to pretrain on 2xT4 would measure noise
around the random baseline.  So we instead build tasks where the intrinsic
sequential depth of the problem is a knob we control exactly, and where ground
truth is exact.  That makes the claim sharper than the paper states it, not
weaker: we can ask whether the recurrence needed *scales with the depth we dialed
in*, which the paper only shows indirectly (Fig. 9 few-shot count, Fig. 14
operand count).

Three tasks, chosen so that two are depth-hard and one is a memory-hard control:

  perm    S_5 permutation composition.  Compose n permutations of 5 elements and
          emit the result.  The word problem for a non-solvable group is
          NC^1-complete, and a constant-depth transformer lives in TC^0, so a
          fixed-depth model provably cannot solve this for growing n unless
          TC^0 = NC^1 (Merrill & Sabharwal, 2023).  A depth-recurrent model can,
          by iterating.  Difficulty knob: n.

  add     Multi-operand addition, the paper's own App. A.1 / Fig. 14 study:
          "we explore whether our model can leverage increased test-time compute
          via recurrence to solve verbalized addition problems of increased
          difficulty".  Difficulty knobs: number of operands, number of digits.

  recall  Single-hop associative recall over k key/value pairs.  This needs
          *memory and attention*, not depth -- one induction head suffices.
          It is the CONTROL for claim C2: Table 4 of the paper reports that
          recurrence helps most on hard reasoning (ARC-C) and least on
          straightforward recall (SciQ).  Without a memory-hard task in the mix,
          "recurrence helps" is unfalsifiable -- everything would just improve.

Tokenisation is raw bytes (vocab 0..255) plus four specials, so there is no
tokenizer artifact to lose or version.  Text stays ASCII (<128); permutation
symbols live at 128..247, so the ranges never collide.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- vocabulary                       # +-- ONE BYTE VOCABULARY FOR EVERYTHING -------------------------
PAD, BOS, EOS, SEP = 256, 257, 258, 259                                                             # | Tokens are raw bytes plus four control ids, so there is no
VOCAB_SIZE = 260                                                                                    # | tokenizer file to train, version, or lose. The three tasks
                                                                                                    # | share the single vocabulary, which means one model can be
PERM_BASE = 128                                    # 120 symbols -> bytes 128..247                  # | trained on all of them at once and evaluated per task.
S5 = list(itertools.permutations(range(5)))        # 120 elements                                   # | Permutations need their own symbols, and they are placed at
S5_INDEX = {p: i for i, p in enumerate(S5)}                                                         # | 128 and above while every task that writes text stays inside
assert len(S5) == 120 and PERM_BASE + 120 <= 248                                                    # | ASCII, so the two ranges can never be confused for one
                                                                                                    # | another. compose is the group product, written so that
                                                                                                    # | applying p after q means looking up q's output in p.
def compose(p, q):
    """Group product (p . q)(i) = p(q(i))."""
    return tuple(p[q[i]] for i in range(5))


def encode(s: str) -> list[int]:
    return list(s.encode("ascii"))


# ------------------------------------------------------------------- perm task                     # +-- A WORD PROBLEM WITH A DEPTH KNOB ---------------------------
# The word problem is stated over a FIXED GENERATING SET, and that is also what                     # | Each input symbol is one permutation of five elements, drawn
# makes it learnable at small scale: the model has to read only |GENERATORS|                        # | from a fixed set of five generators, and the answer is what
# distinct input symbols while tracking a 120-state automaton, rather than first                    # | you get by applying them all in order. There is no shortcut:
# memorising a 120 x 120 multiplication table.  The hardness result is unchanged                    # | the running state after k symbols depends on all k of them,
# -- S5 is the smallest non-solvable symmetric group, so its word problem over a                    # | and the group is non-abelian so they cannot be reordered or
# fixed generating set is NC^1-complete and a constant-depth (TC^0) transformer                     # | grouped cheaply. S5 is the smallest symmetric group that is
# cannot solve it for growing n unless TC^0 = NC^1.                                                 # | not solvable, which is what makes its word problem
GENERATORS = [                                                                                      # | NC1-complete; a transformer of fixed depth computes only TC0
    (1, 0, 2, 3, 4),      # transposition (0 1)                                                     # | functions, so it cannot solve this for growing n unless those
    (0, 2, 1, 3, 4),      # transposition (1 2)                                                     # | two classes collapse. A model that can iterate can. Stating
    (1, 2, 3, 4, 0),      # 5-cycle                                                                 # | the problem over a fixed generating set rather than over all
    (4, 0, 1, 2, 3),      # its inverse                                                             # | 120 elements is both the theoretically standard form and the
    (0, 1, 2, 3, 4),      # identity, so the answer is not a function of n alone                    # | practical fix: the model reads five distinct input symbols
]                                                                                                   # | instead of 120, so it does not have to memorise a
# There are len(GENERATORS)**n distinct prompts at level n: 25 at n=2, 625 at n=4, 390k             # | multiplication table before it can start composing. n is the
# at n=8, 1.5e11 at n=16. The two smallest levels are therefore memorisable by                      # | difficulty knob and it is exactly the number of sequential
# table lookup, which `scripts/check_eval.py` measures per cell rather than                         # | steps the answer requires.
# hiding -- they are the easy end of the difficulty axis and the claim is carried
# by n >= 8, where memorisation is not available.


def make_perm(rng: random.Random, n: int) -> tuple[list[int], list[int]]:
    """'<generator symbols> = <composed state>'."""
    ps = [GENERATORS[rng.randrange(len(GENERATORS))] for _ in range(n)]
    acc = (0, 1, 2, 3, 4)
    for p in ps:
        acc = compose(acc, p)
    prompt = [PERM_BASE + S5_INDEX[p] for p in ps] + encode("=")
    answer = [PERM_BASE + S5_INDEX[acc]]
    return prompt, answer


# -------------------------------------------------------------------- add task                     # +-- ADDITION, AND THE CONTROL THAT MUST NOT IMPROVE ------------
def make_add(rng: random.Random, n_operands: int, n_digits: int):                                   # | Addition is the paper's own study, and it has two knobs: more
    """Paper App. A.1 formulation, compressed to 'a+b+c=' with no chat template."""                 # | operands means more carries chained together, more digits
    lo, hi = (0, 9) if n_digits == 1 else (10 ** (n_digits - 1), 10 ** n_digits - 1)                # | means longer carries. recall is the control, and the suite
    xs = [rng.randint(lo, hi) for _ in range(n_operands)]                                           # | would prove nothing without it. It asks the model to find one
    prompt = encode("+".join(str(x) for x in xs) + "=")                                             # | key among many and report its value, which needs attention and
    answer = encode(str(sum(xs)))                                                                   # | memory but no sequential computation at all: one induction
    return prompt, answer                                                                           # | head does it in a single pass, and adding depth should buy
                                                                                                    # | nothing. If extra recurrence improved every task equally, the
                                                                                                    # | claim being tested would be unfalsifiable, because everything
# ----------------------------------------------------------------- recall task                     # | improves with training. The prediction being checked is that
def make_recall(rng: random.Random, n_pairs: int):                                                  # | recurrence helps perm and add and not recall.
    """'k1:v1 k2:v2 ... ?kq=' -> vq.   Single hop: memory-hard, depth-easy."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    keys = rng.sample([a + b for a in letters for b in letters], n_pairs)
    vals = [rng.choice(letters) for _ in keys]
    j = rng.randrange(n_pairs)
    body = " ".join(f"{k}:{v}" for k, v in zip(keys, vals))
    prompt = encode(f"{body} ?{keys[j]}=")
    answer = encode(vals[j])
    return prompt, answer


# ---------------------------------------------------------------- difficulties                     # +-- WHICH CELLS CAN BE MEMORISED, COMPUTED NOT ASSUMED ---------
# Each entry is (kwargs, difficulty_label).  `difficulty` orders tasks by the                       # | A cell whose set of possible prompts is small enough gets
# sequential depth we believe they require -- the axis C1 predicts saturation                       # | covered completely by the training stream, and a model can
# should track.                                                                                     # | then score on it by looking the answer up rather than working
PERM_LEVELS = [2, 4, 8, 16, 24]                                                                     # | it out. That is not a flaw to hide; it is the easy end of the
ADD_LEVELS = [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 2)]                       # | difficulty axis, and it is exactly where the claim predicts
RECALL_LEVELS = [4, 8, 16, 24]                                                                      # | saturation at small r. But it cannot carry a claim about
                                                                                                    # | reasoning, so which cells are which is computed from the size
TASKS = ("perm", "add", "recall")                                                                   # | of the prompt space rather than guessed. Five generators give
                                                                                                    # | 25 possible prompts at n=2 and 1.5e11 at n=16; two single
# How many distinct prompts a cell can produce.  A cell whose space is small                        # | digits give 100 and three three-digit numbers give 7.3e8;
# compared to the training stream will be covered exhaustively, so a model can                      # | recall's key sets are unbounded. The gate script then measures
# score on it by table lookup instead of by computing -- which is a legitimate                      # | the actual overlap between eval prompts and a training stream,
# easy end of the difficulty axis, but cannot carry a claim about reasoning.                        # | and the measured leak matches the computed space closely: 1.00
# We compute this rather than assume it, and `scripts/check_eval.py` prints it                      # | at 25 prompts, 0.73 at 1e3, and 0.00 above 1e6. Eight of
# so every number in the report can be read as "computed" or "possibly recalled".                   # | seventeen cells survive as places where a score has to be
TABULABLE_THRESHOLD = 1_000_000                                                                     # | computed.


def prompt_space(task: str, level) -> float:
    if task == "perm":
        return float(len(GENERATORS)) ** level
    if task == "add":
        n_ops, n_dig = level
        span = 10.0 if n_dig == 1 else 9.0 * 10 ** (n_dig - 1)
        return span ** n_ops
    if task == "recall":                       # C(676, k) * 26^k * k, astronomically large
        return float("inf")
    raise ValueError(task)


def is_tabulable(task: str, level) -> bool:
    return prompt_space(task, level) < TABULABLE_THRESHOLD


def claim_cells():
    """The cells on which a reasoning claim can actually be made."""
    return [f"{t}/{l}" for t in TASKS for l in levels_for(t) if not is_tabulable(t, l)]


def sample_example(rng: random.Random, task: str, level):
    if task == "perm":
        return make_perm(rng, level)
    if task == "add":
        return make_add(rng, *level)
    if task == "recall":
        return make_recall(rng, level)
    raise ValueError(task)


def levels_for(task: str):
    return {"perm": PERM_LEVELS, "add": ADD_LEVELS, "recall": RECALL_LEVELS}[task]


# ------------------------------------------------------------------- packing                       # +-- PACKING, HELD-OUT SETS, AND EVERY CELL'S FLOOR -------------
@dataclass                                                                                          # | Examples are concatenated into one continuous byte stream and
class Corpus:                                                                                       # | random windows are cut from it, which is what the paper does
    tokens: np.ndarray          # 1-D uint16 stream, ready for packing into blocks                  # | with its own corpus. The model therefore sees answers only as
    n_examples: int                                                                                 # | ordinary continuations, never flagged as targets, and the loss
                                                                                                    # | is plain next byte prediction over everything. Held-out sets
                                                                                                    # | are generated from a different seed, so they are drawn from
def build_train_corpus(n_tokens: int, seed: int = 0, text: np.ndarray | None = None,                # | the same distribution but are not the same items.
                       text_frac: float = 0.30, task_weights=(0.40, 0.35, 0.25)) -> Corpus:         # | shortcut_baseline is the number that keeps the results honest:
    """Build one packed byte stream, mixing the three tasks (and optional text).

    Sec. 4.1: the paper packs tokenised documents into fixed-length sequences and
    trains a plain next-token loss over everything; we do the same, so the model
    sees the answers only as ordinary continuations, never as a special target.
    """
    rng = random.Random(seed)                                                                       # | it is what an unconditional guesser scores by always answering
    out: list[int] = []                                                                             # | with the most common answer for that cell. Reporting an
    n_ex = 0                                                                                        # | accuracy without it invites reading a number near zero as a
    n_task_tokens = int(n_tokens * (1.0 - (text_frac if text is not None else 0.0)))                # | failure when it is actually the ceiling of the trivial policy,
    while len(out) < n_task_tokens:                                                                 # | or a number well above zero as success when the constant guess
        task = rng.choices(TASKS, weights=task_weights)[0]                                          # | already gets there. Measured floors here run from 0.004 to
        level = rng.choice(levels_for(task))                                                        # | 0.199 depending on the cell, so no single baseline would have
        prompt, answer = sample_example(rng, task, level)                                           # | done.
        out.extend([BOS] + prompt + answer + [EOS])
        n_ex += 1
    if text is not None and text_frac > 0:
        take = int(n_tokens * text_frac)
        start = rng.randrange(max(1, len(text) - take))
        out.extend(text[start:start + take].tolist())
    arr = np.array(out, dtype=np.uint16)
    rngn = np.random.default_rng(seed)
    # shuffle at document granularity would need offsets; the stream is already
    # interleaved at random so a plain stream is fine for packing.
    return Corpus(arr, n_ex)


def build_eval_set(task: str, level, n: int = 256, seed: int = 1234):
    """Held-out prompts + gold answers for one (task, difficulty) cell."""
    rng = random.Random(hash((task, str(level), seed)) & 0xFFFFFFFF)
    items = []
    for _ in range(n):
        prompt, answer = sample_example(rng, task, level)
        items.append(([BOS] + prompt, answer))
    return items


def shortcut_baseline(task: str, level, n: int = 2000, seed: int = 99) -> float:
    """Phase-4 gate: the score of the best DEGENERATE constant policy.

    For every eval cell we compute what an unconditional most-frequent-answer
    guesser scores, so the report can state each result against its real floor
    rather than against 0.
    """
    from collections import Counter
    rng = random.Random(seed)
    c = Counter()
    for _ in range(n):
        _, answer = sample_example(rng, task, level)
        c[tuple(answer)] += 1
    return c.most_common(1)[0][1] / n
