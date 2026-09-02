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

# ---------------------------------------------------------------- vocabulary
PAD, BOS, EOS, SEP = 256, 257, 258, 259
VOCAB_SIZE = 260

PERM_BASE = 128                                    # 120 symbols -> bytes 128..247
S5 = list(itertools.permutations(range(5)))        # 120 elements
S5_INDEX = {p: i for i, p in enumerate(S5)}
assert len(S5) == 120 and PERM_BASE + 120 <= 248


def compose(p, q):
    """Group product (p . q)(i) = p(q(i))."""
    return tuple(p[q[i]] for i in range(5))


def encode(s: str) -> list[int]:
    return list(s.encode("ascii"))


# ------------------------------------------------------------------- perm task
# The word problem is stated over a FIXED GENERATING SET, and that is also what
# makes it learnable at small scale: the model has to read only |GENERATORS|
# distinct input symbols while tracking a 120-state automaton, rather than first
# memorising a 120 x 120 multiplication table.  The hardness result is unchanged
# -- S5 is the smallest non-solvable symmetric group, so its word problem over a
# fixed generating set is NC^1-complete and a constant-depth (TC^0) transformer
# cannot solve it for growing n unless TC^0 = NC^1.
GENERATORS = [
    (1, 0, 2, 3, 4),      # transposition (0 1)
    (0, 2, 1, 3, 4),      # transposition (1 2)
    (1, 2, 3, 4, 0),      # 5-cycle
    (4, 0, 1, 2, 3),      # its inverse
    (0, 1, 2, 3, 4),      # identity, so the answer is not a function of n alone
]
# |GENERATORS|^n distinct prompts exist at level n: 25 at n=2, 625 at n=4, 390k
# at n=8, 1.5e11 at n=16. The two smallest levels are therefore memorisable by
# table lookup, which `scripts/check_eval.py` measures per cell rather than
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


# -------------------------------------------------------------------- add task
def make_add(rng: random.Random, n_operands: int, n_digits: int):
    """Paper App. A.1 formulation, compressed to 'a+b+c=' with no chat template."""
    lo, hi = (0, 9) if n_digits == 1 else (10 ** (n_digits - 1), 10 ** n_digits - 1)
    xs = [rng.randint(lo, hi) for _ in range(n_operands)]
    prompt = encode("+".join(str(x) for x in xs) + "=")
    answer = encode(str(sum(xs)))
    return prompt, answer


# ----------------------------------------------------------------- recall task
def make_recall(rng: random.Random, n_pairs: int):
    """'k1:v1 k2:v2 ... ?kq=' -> vq.   Single hop: memory-hard, depth-easy."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    keys = rng.sample([a + b for a in letters for b in letters], n_pairs)
    vals = [rng.choice(letters) for _ in keys]
    j = rng.randrange(n_pairs)
    body = " ".join(f"{k}:{v}" for k, v in zip(keys, vals))
    prompt = encode(f"{body} ?{keys[j]}=")
    answer = encode(vals[j])
    return prompt, answer


# ---------------------------------------------------------------- difficulties
# Each entry is (kwargs, difficulty_label).  `difficulty` orders tasks by the
# sequential depth we believe they require -- the axis C1 predicts saturation
# should track.
PERM_LEVELS = [2, 4, 8, 16, 24]
ADD_LEVELS = [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 2)]
RECALL_LEVELS = [4, 8, 16, 24]

TASKS = ("perm", "add", "recall")

# How many distinct prompts a cell can produce.  A cell whose space is small
# compared to the training stream will be covered exhaustively, so a model can
# score on it by table lookup instead of by computing -- which is a legitimate
# easy end of the difficulty axis, but cannot carry a claim about reasoning.
# We compute this rather than assume it, and `scripts/check_eval.py` prints it
# so every number in the report can be read as "computed" or "possibly recalled".
TABULABLE_THRESHOLD = 1_000_000


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


# ------------------------------------------------------------------- packing
@dataclass
class Corpus:
    tokens: np.ndarray          # 1-D uint16 stream, ready for packing into blocks
    n_examples: int


def build_train_corpus(n_tokens: int, seed: int = 0, text: np.ndarray | None = None,
                       text_frac: float = 0.30, task_weights=(0.40, 0.35, 0.25)) -> Corpus:
    """Build one packed byte stream, mixing the three tasks (and optional text).

    Sec. 4.1: the paper packs tokenised documents into fixed-length sequences and
    trains a plain next-token loss over everything; we do the same, so the model
    sees the answers only as ordinary continuations, never as a special target.
    """
    rng = random.Random(seed)
    out: list[int] = []
    n_ex = 0
    n_task_tokens = int(n_tokens * (1.0 - (text_frac if text is not None else 0.0)))
    while len(out) < n_task_tokens:
        task = rng.choices(TASKS, weights=task_weights)[0]
        level = rng.choice(levels_for(task))
        prompt, answer = sample_example(rng, task, level)
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
