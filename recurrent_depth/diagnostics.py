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


@torch.no_grad()                                                                                    # +-- THE NUMBER THAT CATCHES A COLLAPSED MODEL ------------------
def token_correlation(s: torch.Tensor) -> float:                                                    # | Each token's hidden vector is centred and scaled to unit
    """Mean pairwise correlation of hidden states across the sequence dimension.

    s: (B, S, h).  Returns the average off-diagonal correlation; -> 1.0 is the
    representation collapse of Sec. 4.3 / Fig. 5 (middle panel).
    """
    x = s.float()                                                                                   # | length, then every pair of tokens has its dot product taken,
    x = x - x.mean(dim=-1, keepdim=True)                                                            # | which is their cosine similarity. The average over pairs says
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)                                                   # | how much the sequence has homogenised. At zero the tokens
    g = x @ x.transpose(1, 2)                       # (B, S, S) cosine sims                         # | carry different information. At one the model produces the
    B, S, _ = g.shape                                                                               # | same hidden state at every position, so the sequence dimension
    if S < 2:                                                                                       # | has stopped mattering and the model can emit only one thing
        return float("nan")                                                                         # | regardless of context. That is the specific way the paper's
    off = (g.sum(dim=(1, 2)) - g.diagonal(dim1=1, dim2=2).sum(-1)) / (S * (S - 1))                  # | first large run died, and every turn of the loop pushes the
    return off.mean().item()                                                                        # | number higher, so a recurrent model reaches it faster than a
                                                                                                    # | plain one. It tells a collapsed run apart from a merely
                                                                                                    # | undertrained one, which the loss curve alone does not.
@torch.no_grad()                                                                                    # +-- HOW THE STATE MOVES OVER A TRAJECTORY ----------------------
def recurrence_stats(states: torch.Tensor) -> dict:                                                 # | Given every state the loop passed through, these four curves
    """states: (r+1, B, S, h) from `RecurrentDepthLM.trajectory`.

    Returns the three App. A.2 recurrence statistics plus the distance-to-limit
    curve used for Fig. 11 ("the norm distance ||si - s*|| between each si in a
    trajectory and an approximate limit point s* computed with 128 iterations").
    """
    s = states.float()                                                                              # | say what kind of motion it was. The step size relative to the
    norms = s.norm(dim=-1).mean(dim=(1, 2))                      # norm of s_k                      # | state's own norm says whether the loop is still moving or has
    d = (s[1:] - s[:-1]).norm(dim=-1).mean(dim=(1, 2))           # step from s_k-1 to s_k           # | settled. The distance from the start says how far it
    rel = d / (norms[1:] + 1e-6)                                 # relative step                    # | travelled. The distance to the last state treats that as an
    d0 = (s - s[0:1]).norm(dim=-1).mean(dim=(1, 2))              # distance travelled from s_0      # | approximate fixed point and measures the approach to it. A
    dstar = (s - s[-1:]).norm(dim=-1).mean(dim=(1, 2))           # distance to the limit point (Fig. 11)
    return {                                                                                        # | curve falling monotonically is convergence. One levelling off
        "state_norm": norms.tolist(),                                                               # | above zero is an orbit: the state keeps moving but stops
        "step_norm": d.tolist(),                                                                    # | getting closer, which the paper reports on tokens carrying
        "rel_step": rel.tolist(),                                                                   # | numeric reasoning. One drifting steadily in a single direction
        "dist_from_s0": d0.tolist(),                                                                # | is a counter. None of this is trained for; the objective only
        "dist_to_limit": dstar.tolist(),                                                            # | ever asks for a correct next token at whatever depth was
        "token_corr_final": token_correlation(states[-1]),                                          # | sampled.
    }


@torch.no_grad()                                                                                    # +-- DOES THE ANSWER DEPEND ON WHERE IT STARTED -----------------
def path_independence(model, idx, r: int, n_seeds: int = 4, device="cuda") -> dict:                 # | The same input is run several times from different random
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
    finals, preds = [], []                                                                          # | start states and the results compared two ways. Argmax
    for seed in range(n_seeds):                                                                     # | agreement is the fraction of positions where two runs emit the
        g = torch.Generator(device=device).manual_seed(1000 + seed)                                 # | same token; it is the number that matters in practice, because
        out = model(idx, r=r, generator=g)                                                          # | a model whose output depends on its random start cannot safely
        finals.append(out["state"].float())                                                         # | be run at a depth it was not tuned at. Cosine similarity
        preds.append(out["logits"].argmax(-1))                                                      # | between final states is the stricter version: it asks whether
    agree, cos, n = 0.0, 0.0, 0                                                                     # | the runs converged to the same point, not merely to the same
    for i in range(n_seeds):                                                                        # | decision. Both are needed. A model can agree on every token
        for j in range(i + 1, n_seeds):                                                             # | while its states sit far apart, which means the decision is
            agree += (preds[i] == preds[j]).float().mean().item()                                   # | robust but the loop has not converged, and running it longer
            a, b = finals[i], finals[j]                                                             # | might still change something.
            cos += torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
            n += 1
    return {"argmax_agree": agree / n, "state_cos": cos / n, "r": r, "n_seeds": n_seeds}
