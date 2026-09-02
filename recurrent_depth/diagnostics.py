"""The paper's own model-health metrics.

App. A.2, "Pretraining Metrics" (verbatim):

    "During the pretraining run, we run a careful tracking of optimizer and model
     health metrics, tracking effective Adam learning rates per layer, optimizer
     RMS (Wortsman et al., 2023a), L2 and L1 parameter and gradient norms,
     recurrence statistics such as ||sk-sk-1||/||sk||, ||sk||, ||s0 - sk||. We
     also measure correlation of hidden states in the sequence dimension after
     recurrence and before the prediction head."

Sec. 4.3 makes the token-correlation metric the diagnostic for the failure mode
that killed their first large run:

    "we find that this stall is due to the model's representation collapsing
     (Noci et al., 2022). The correlation of hidden states in the token dimension
     quickly goes to 1.0 (middle plot), meaning the model predicts the same
     hidden state for every token in the sequence. [...] Every iteration of the
     recurrence block increases token correlation, mixing the sequence until
     collapse."

We reproduce these because they are how you tell a *failed* recurrent run from a
merely undertrained one, and because "Bad Run 2" is a failure our Track-B
retrofit is at direct risk of: a model that has learned to ignore the incoming
state s and therefore shows a flat perplexity-vs-r curve.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def token_correlation(s: torch.Tensor) -> float:
    """Mean pairwise correlation of hidden states across the sequence dimension.

    s: (B, S, h).  Returns the average off-diagonal correlation; -> 1.0 is the
    representation collapse of Sec. 4.3 / Fig. 5 (middle panel).
    """
    x = s.float()
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    g = x @ x.transpose(1, 2)                       # (B, S, S) cosine sims
    B, S, _ = g.shape
    if S < 2:
        return float("nan")
    off = (g.sum(dim=(1, 2)) - g.diagonal(dim1=1, dim2=2).sum(-1)) / (S * (S - 1))
    return off.mean().item()


@torch.no_grad()
def recurrence_stats(states: torch.Tensor) -> dict:
    """states: (r+1, B, S, h) from `RecurrentDepthLM.trajectory`.

    Returns the three App. A.2 recurrence statistics plus the distance-to-limit
    curve used for Fig. 11 ("the norm distance ||si - s*|| between each si in a
    trajectory and an approximate limit point s* computed with 128 iterations").
    """
    s = states.float()
    norms = s.norm(dim=-1).mean(dim=(1, 2))                      # ||s_k||
    d = (s[1:] - s[:-1]).norm(dim=-1).mean(dim=(1, 2))           # ||s_k - s_{k-1}||
    rel = d / (norms[1:] + 1e-6)                                 # relative step
    d0 = (s - s[0:1]).norm(dim=-1).mean(dim=(1, 2))              # ||s_0 - s_k||
    dstar = (s - s[-1:]).norm(dim=-1).mean(dim=(1, 2))           # ||s_k - s*|| (Fig. 11)
    return {
        "state_norm": norms.tolist(),
        "step_norm": d.tolist(),
        "rel_step": rel.tolist(),
        "dist_from_s0": d0.tolist(),
        "dist_to_limit": dstar.tolist(),
        "token_corr_final": token_correlation(states[-1]),
    }


@torch.no_grad()
def path_independence(model, idx, r: int, n_seeds: int = 4, device="cuda") -> dict:
    """Sec. 7, "Path Independence" (verbatim):

        "We verify that our models maintain path independence, in the sense of
         Anil et al. (2022) [...] When re-initializing from multiple starting
         states s0, the model moves in similar trajectories, exhibiting
         consistent behavior. The same orbital patterns, fixed points, or
         directional drifts emerge regardless of initialization."

    We turn that qualitative statement into two numbers:
      argmax_agree  fraction of positions where two runs from different s0 pick
                    the same next token (1.0 = perfectly path independent);
      state_cos     mean cosine similarity of the FINAL latent states s_r across
                    seeds (1.0 = converged to the same point).
    """
    finals, preds = [], []
    for seed in range(n_seeds):
        g = torch.Generator(device=device).manual_seed(1000 + seed)
        out = model(idx, r=r, generator=g)
        finals.append(out["state"].float())
        preds.append(out["logits"].argmax(-1))
    agree, cos, n = 0.0, 0.0, 0
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            agree += (preds[i] == preds[j]).float().mean().item()
            a, b = finals[i], finals[j]
            cos += torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
            n += 1
    return {"argmax_agree": agree / n, "state_cos": cos / n, "r": r, "n_seeds": n_seeds}
