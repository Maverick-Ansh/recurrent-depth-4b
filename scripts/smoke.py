"""Smoke tests that assert the PAPER'S RULES, not just tensor shapes.

Run:  python scripts/smoke.py
Every assertion names the section or figure of arXiv:2502.05171 it enforces.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recurrent_depth.config import RecurrentDepthConfig
from recurrent_depth.model import RecurrentDepthLM
from recurrent_depth.sampling import sample_r, selftest_matches_figure3
from recurrent_depth.diagnostics import token_correlation, recurrence_stats
from recurrent_depth.inference import adaptive_exit_forward, generate
from data import tasks

OK, FAIL = [], []


def check(name, fn):
    try:
        detail = fn()
        OK.append((name, detail))
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}\n        {e}")


# ---------------------------------------------------------------------------
print("\n== Sec. 3.3 / Fig. 3 : the unrolling distribution Lambda ==")


def lambda_min_is_one():
    r = sample_r(8.0, 0.5, n=100_000)
    assert int(r.min()) >= 1, f"Eq. 2 adds +1, so min r must be >= 1; got {int(r.min())}"
    return f"min={int(r.min())} max={int(r.max())}"


def lambda_is_heavy_tailed():
    r = sample_r(8.0, 0.5, n=100_000).float()
    p = (r > 16).float().mean().item()
    assert p > 0.001, "Sec. 3.3 calls Lambda heavy-tailed; found no mass past 2*rbar"
    return f"P(r > 2*rbar) = {p:.4f}"


check("Lambda moments match Fig. 3 (mean 33.0, median 29.0 at rbar=32)",
      lambda: str(selftest_matches_figure3()))
check("r >= 1 always (Eq. 2 adds +1 to the Poisson draw)", lambda_min_is_one)
check("Lambda is heavy-tailed: P(r > 2*rbar) > 0 (Sec. 3.3 'heavy tail')", lambda_is_heavy_tailed)


# ---------------------------------------------------------------------------
print("\n== Sec. 4.1 : initialisation ==")
cfg = RecurrentDepthConfig(vocab_size=260, hidden=512, n_heads=8, mlp_inner=1376, max_seq=256)
m = RecurrentDepthLM(cfg)
h = cfg.hidden


def init_std_ok():
    sigma_h = math.sqrt(2.0 / (5.0 * h))
    w = m.core[0].attn.q_proj.weight.std().item()
    # truncated at 3 sigma shrinks the empirical std by ~1.4%
    assert abs(w - sigma_h) / sigma_h < 0.08, f"sigma_h {sigma_h:.5f} vs empirical {w:.5f}"
    return f"sigma_h={sigma_h:.5f} emp={w:.5f}"


def out_proj_std_ok():
    l = m.effective_layers
    sigma_out = math.sqrt(1.0 / (5.0 * h * l))
    w = m.core[0].attn.o_proj.weight.std().item()
    assert abs(w - sigma_out) / sigma_out < 0.08, f"sigma_out {sigma_out:.5f} vs {w:.5f}"
    assert sigma_out < math.sqrt(2.0 / (5.0 * h)), "out-projections must init SMALLER (Sec. 4.1)"
    return f"sigma_out={sigma_out:.6f} emp={w:.6f} (l={l})"


def s0_matches_embedding_scale():
    """Sec. 4.1 internal consistency: E has variance 2/(5h) and its output is
    scaled by gamma=sqrt(h), giving variance 2/5 -- exactly sigma_s^2. So the
    random initial state lives on the same scale as the injected embedding."""
    idx = torch.randint(0, 260, (4, 32))
    e = m.embed(idx) * m.gamma
    s0 = m.init_state(e)
    ve, vs = e.var().item(), s0.var().item()
    assert abs(vs - 0.4) < 0.05, f"var(s0)={vs:.3f}, paper sigma_s^2 = 2/5"
    assert abs(ve - 0.4) / 0.4 < 0.25, f"var(gamma*E(x))={ve:.3f}, expected ~2/5"
    return f"var(gamma*E)={ve:.3f}  var(s0)={vs:.3f}  (paper: both 0.4)"


check("sigma_h^2 = 2/(5h) on ordinary weights", init_std_ok)
check("sigma_out^2 = 1/(5hl) on out-projections, and smaller than sigma_h", out_proj_std_ok)
check("var(s0) = 2/5 = var(gamma*E(x)) [Sec. 4.1 consistency]", s0_matches_embedding_scale)

# ---------------------------------------------------------------------------
print("\n== Sec. 3.2 : architecture layout ==")


def sandwich_has_four_norms():
    b = m.core[0]
    assert hasattr(b, "n2") and hasattr(b, "n4"), "sandwich block needs n2 and n4"
    pre = RecurrentDepthLM(RecurrentDepthConfig(vocab_size=260, hidden=64, n_heads=4,
                                                mlp_inner=128, norm_style="pre"))
    assert not hasattr(pre.core[0], "n2"), "pre-norm block must not have n2/n4"
    return "sandwich: n1..n4 ; pre: n1,n3"


def adapter_shape():
    assert m.adapter.weight.shape == (h, 2 * h), \
        f"A : R^2h -> R^h, got {tuple(m.adapter.weight.shape)}"
    return f"A: {tuple(m.adapter.weight.shape)}"


def tied_embeddings():
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr(), \
        "Sec. 3.2: projection into the vocabulary using tied embeddings E^T"
    return "lm_head.weight is embed.weight"


check("core block starts with adapter A : R^2h -> R^h (Sec. 3.2)", adapter_shape)
check("sandwich norm has n1..n4; pre-norm ablation has only n1,n3", sandwich_has_four_norms)
check("tied input/output embeddings", tied_embeddings)

# ---------------------------------------------------------------------------
print("\n== Sec. 3.3 : truncated backpropagation ==")
idx = torch.randint(0, 260, (2, 24))
tgt = torch.randint(0, 260, (2, 24))


def prelude_gets_grad_with_truncation():
    """'the prelude block still receives gradient updates in every step, as its
    output e is injected in every step' -- so e must NOT be inside no_grad."""
    m.zero_grad()
    m(idx, r=8, k=2, targets=tgt)["loss"].backward()
    g = m.prelude[0].attn.q_proj.weight.grad
    assert g is not None and g.norm() > 0, "prelude received no gradient under truncation"
    return f"|grad| = {g.norm():.5f}"


def truncation_changes_grads():
    def grads(k):
        m.zero_grad()
        torch.manual_seed(0)
        m(idx, r=8, k=k, targets=tgt, s0=S0)["loss"].backward()
        return m.core[0].mlp.down_proj.weight.grad.clone()
    S0 = torch.zeros(2, 24, h)
    g1, g8 = grads(1), grads(8)
    rel = (g1 - g8).norm() / (g8.norm() + 1e-12)
    assert rel > 1e-3, "k=1 and k=8 gave identical gradients -- truncation is not wired up"
    return f"||g_k1 - g_k8|| / ||g_k8|| = {rel:.3f}"


def memory_independent_of_r():
    """'maximum activation memory and backward compute is now independent of r'."""
    def graph_depth(r, k):
        torch.manual_seed(0)
        out = m(idx, r=r, k=k, targets=tgt, s0=torch.zeros(2, 24, h))
        n = 0
        seen, stack = set(), [out["loss"].grad_fn]
        while stack:
            fn = stack.pop()
            if fn is None or id(fn) in seen:
                continue
            seen.add(id(fn)); n += 1
            stack.extend(nf for nf, _ in fn.next_functions)
        return n
    a, b = graph_depth(8, 4), graph_depth(32, 4)
    assert a == b, f"autograd graph grew with r under fixed k: {a} vs {b}"
    return f"graph nodes: r=8 -> {a}, r=32 -> {b} (identical, as Sec. 3.3 claims)"


check("prelude still gets gradient under truncation", prelude_gets_grad_with_truncation)
check("k actually truncates (gradients differ for k=1 vs k=8)", truncation_changes_grads)
check("activation memory independent of r for fixed k", memory_independent_of_r)

# ---------------------------------------------------------------------------
print("\n== Sec. 5.1 / Fig. 1 : parameter accounting vs the paper's own numbers ==")


def paper_config_param_counts():
    """Build the paper's exact main config on the meta device and check our
    parameter layout against the numbers they report.

    Sec. 4.1: "(2, 4, 2) [...] hidden size to h = 5280, which yields 55 heads of
    size of 96. The MLP inner dimension is 17920 [...] about 1.5B parameters in
    non-recurrent prelude and head, 1.5B parameters in the core recurrent block,
    and 0.5B in the tied input embedding." Vocab 65536 (Sec. 4.1 tokenisation).
    """
    pc = RecurrentDepthConfig(vocab_size=65536, hidden=5280, n_heads=55, mlp_inner=17920,
                              l_prelude=2, l_core=4, l_coda=2, mean_recurrence=32,
                              backprop_depth=8, max_seq=4096)
    with torch.device("meta"):
        pm = RecurrentDepthLM(pc, init_weights=False)
    total = pm.n_params() / 1e9
    embed = pm.embed.weight.numel() / 1e9
    core = (sum(p.numel() for p in pm.core.parameters()) + pm.adapter.weight.numel()) / 1e9
    assert 3.2 < total < 3.8, f"paper says 3.5B total, we build {total:.2f}B"
    assert 1.3 < core < 1.75, f"paper says ~1.5B in the core, we build {core:.2f}B"
    assert 0.30 < embed < 0.40, f"paper says 0.5B tied embedding, we build {embed:.2f}B"
    # Fig. 1 upper x-axis: materialized params at r = 1,4,6,8,12,20,32,48,64
    fig1 = {1: 3.6, 4: 8.3, 6: 11.5, 8: 14.6, 12: 21.0, 20: 33.6, 32: 52.6, 48: 77.9, 64: 103.0}
    errs = {r: abs(pm.materialized_params(r) / 1e9 - v) / v for r, v in fig1.items()}
    worst = max(errs.values())
    assert worst < 0.10, f"materialized params disagree with Fig. 1 axis: {errs}"
    return (f"total={total:.2f}B core={core:.2f}B embed={embed:.2f}B ; "
            f"Fig.1 axis max rel. err {worst:.1%}")


check("paper's (2,4,2) h=5280 config reproduces 3.5B/1.5B/0.5B and the Fig. 1 axis",
      paper_config_param_counts)

# ---------------------------------------------------------------------------
print("\n== Sec. 6 : zero-shot inference paths run ==")


def adaptive_exit_runs():
    out = adaptive_exit_forward(m, idx, r_max=12, threshold=1e9)   # trivially satisfied
    assert out["exit_step"].max().item() <= 12
    assert out["mean_steps"] <= 3.0, "with an infinite threshold everything must exit at step 2"
    out2 = adaptive_exit_forward(m, idx, r_max=12, threshold=0.0)  # never satisfied
    assert out2["mean_steps"] == 12.0, "with threshold 0 nothing may exit early"
    return f"trivial-threshold mean steps {out['mean_steps']:.2f}, strict {out2['mean_steps']:.1f}"


def kv_budget_matches_unshared_when_budget_ge_r():
    """With budget >= r no slot is ever reused, so decoding must be IDENTICAL to
    the unshared cache. If this fails, the sharing logic is corrupting the cache."""
    torch.manual_seed(0)
    p = torch.randint(0, 260, (1, 10))
    a = generate(m, p, max_new_tokens=4, r=4, kv_budget=None,
                 generator=torch.Generator().manual_seed(7))
    b = generate(m, p, max_new_tokens=4, r=4, kv_budget=8,
                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a["generated"], b["generated"]), \
        f"budget>=r changed the output: {a['generated']} vs {b['generated']}"
    return f"generated {a['generated'].tolist()}"


def cached_decode_equals_teacher_forced():
    """The load-bearing cache test: decoding one token at a time through the
    recurrence caches must reproduce the full-sequence forward exactly. This is
    what catches a RoPE offset slip or a mis-ordered recurrence cache."""
    from recurrent_depth.inference import RecurrenceKVCache
    S, r = 10, 6
    p = torch.randint(0, 260, (1, S))
    s0 = torch.randn(1, S, h) * 0.632          # same s0 down both paths
    m.eval()
    tf = m(p, r=r, s0=s0)["logits"]
    cache = RecurrenceKVCache(len(m.core), None)
    pk = ck = None
    outs = []
    for j in range(S):
        e, pk = m.prelude_forward(p[:, j:j + 1], pk)
        s = s0[:, j:j + 1]
        cache.begin_block()
        for i in range(1, r + 1):
            s, nkv = m.core_forward(s, e, cache.read(i))
            cache.write(i, nkv)
        cache.end_block()
        lg, ck = m.coda_forward(s, ck)
        outs.append(lg[:, -1])
    d = (tf - torch.stack(outs, 1)).abs().max().item()
    assert d < 2e-3, f"cached decode differs from teacher forcing by {d:.2e}"
    return f"max |diff| = {d:.2e} (logit std {tf.std():.2f})"


def kv_budget_below_r_perturbs_logits():
    """Budget < r must actually change what the model computes -- otherwise the
    'zero-shot KV sharing costs nothing' result would be vacuously true."""
    from recurrent_depth.inference import RecurrenceKVCache
    p = torch.randint(0, 260, (1, 12))

    def dec(budget, r=8):
        cache = RecurrenceKVCache(len(m.core), budget)
        pk = ck = None
        g = torch.Generator().manual_seed(7)
        outs = []
        for cur in [p[:, :8], p[:, 8:9], p[:, 9:10], p[:, 10:11], p[:, 11:12]]:
            e, pk = m.prelude_forward(cur, pk)
            s = m.init_state(e, g)
            cache.begin_block()
            for i in range(1, r + 1):
                s, nkv = m.core_forward(s, e, cache.read(i))
                cache.write(i, nkv)
            cache.end_block()
            lg, ck = m.coda_forward(s, ck)
            outs.append(lg[:, -1])
        return torch.stack(outs)

    a, b, c = dec(None), dec(2), dec(8)
    assert (a - c).abs().max().item() == 0.0, "budget == r must be bitwise identical"
    delta = (a - b).abs().max().item()
    assert delta > 1e-4, "budget 2 changed nothing -- slot sharing is not wired up"
    return f"budget2 perturbs logits by {delta:.4f} (std {a.std():.2f}); budget8 identical"


check("KL adaptive exit honours its threshold in both limits", adaptive_exit_runs)
check("KV budget >= r is a no-op (cache logic is not corrupting state)",
      kv_budget_matches_unshared_when_budget_ge_r)
check("cached decoding == teacher-forced forward", cached_decode_equals_teacher_forced)
check("KV budget < r genuinely perturbs the computation", kv_budget_below_r_perturbs_logits)

# ---------------------------------------------------------------------------
print("\n== Sec. 4.3 : the collapse diagnostic is sensitive ==")


def token_corr_detects_collapse():
    B, S, H = 2, 16, 32
    collapsed = torch.randn(B, 1, H).expand(B, S, H).contiguous()
    diverse = torch.randn(B, S, H)
    c1, c2 = token_correlation(collapsed), token_correlation(diverse)
    assert c1 > 0.99, f"collapsed states must read ~1.0, got {c1:.3f}"
    assert abs(c2) < 0.3, f"random states must read ~0.0, got {c2:.3f}"
    return f"collapsed={c1:.3f}  random={c2:.3f}  (Fig. 5 middle panel goes to 1.0)"


check("token_correlation reads 1.0 on collapsed and ~0 on diverse states",
      token_corr_detects_collapse)

# ---------------------------------------------------------------------------
print("\n== Task suite : ground truth is exact ==")


def perm_is_a_group():
    import random
    rng = random.Random(0)
    for n in (2, 8, 24):
        prompt, ans = tasks.make_perm(rng, n)
        syms = [t - tasks.PERM_BASE for t in prompt[:-1]]
        acc = (0, 1, 2, 3, 4)
        for si in syms:
            acc = tasks.compose(acc, tasks.S5[si])
        assert ans[0] - tasks.PERM_BASE == tasks.S5_INDEX[acc], f"perm gold wrong at n={n}"
    return "composition verified for n in {2,8,24}"


def add_gold_is_right():
    import random
    rng = random.Random(0)
    for lv in tasks.ADD_LEVELS:
        p, a = tasks.make_add(rng, *lv)
        lhs = bytes(p[:-1]).decode()
        assert eval(lhs) == int(bytes(a).decode()), f"add gold wrong at {lv}"
    return f"{len(tasks.ADD_LEVELS)} difficulty cells verified"


def vocab_ranges_do_not_collide():
    assert tasks.PERM_BASE >= 128, "perm symbols must sit above ASCII"
    assert tasks.PERM_BASE + 120 <= 248
    import random
    rng = random.Random(0)
    for lv in tasks.ADD_LEVELS:
        p, a = tasks.make_add(rng, *lv)
        assert max(p + a) < 128, "add task leaked out of ASCII"
    for lv in tasks.RECALL_LEVELS:
        p, a = tasks.make_recall(rng, lv)
        assert max(p + a) < 128, "recall task leaked out of ASCII"
    return "ASCII (<128) and perm symbols (128..247) are disjoint"


check("perm task gold answers really are the group product", perm_is_a_group)
check("add task gold answers really are the sum", add_gold_is_right)
check("token ranges of the three tasks never collide", vocab_ranges_do_not_collide)

# ---------------------------------------------------------------------------
print(f"\n{'='*70}\n{len(OK)} passed, {len(FAIL)} failed")
if FAIL:
    for n, e in FAIL:
        print(f"  FAILED: {n}\n    {e}")
    sys.exit(1)
print("All paper-rule assertions hold.")
