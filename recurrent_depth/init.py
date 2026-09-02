"""Initialisation, Sec. 4.1 "Architecture and Initialization".

Paper (verbatim):

    "At small scales, most sensible initialization schemes work. However, at
     larger scales, we use the initialization of Takase et al. (2024) which
     prescribes a variance of sigma_h^2 = 2/(5h). We initialize all parameters
     from a truncated normal distribution (truncated at 3 sigma) with this
     variance, except all out-projection layers, where the variance is set to
     sigma_out^2 = 1/(5 h l), for l = lP + rbar*lR + lC the number of effective
     layers, which is 132 for this model. As a result, the out-projection layers
     are initialized with fairly small values (Goyal et al., 2018). The output of
     the embedding layer is scaled by sqrt(h). To match this initialization, the
     state s0 is also sampled from a truncated normal distribution, here with
     variance sigma_s^2 = 2/5."

Note the internal consistency that pins sigma_s: the embedding matrix has
per-entry variance 2/(5h) and its output is scaled by gamma = sqrt(h), so
gamma*E(x) has per-entry variance (2/(5h)) * h = 2/5 = sigma_s^2.  The random
initial state is therefore drawn on the same scale as the injected embedding.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

OUT_PROJ_NAMES = ("o_proj", "down_proj")   # attention out-proj and MLP down-proj                   # +-- SMALLER STARTING WEIGHTS WHERE THINGS ADD UP ---------------
                                                                                                    # | Every weight starts from a normal cut off at three standard
                                                                                                    # | deviations, at variance 2/(5h), except the two projections
def truncated_normal_(t: torch.Tensor, std: float) -> torch.Tensor:                                 # | that write back into the residual stream. Those start far
    return nn.init.trunc_normal_(t, mean=0.0, std=std, a=-3 * std, b=3 * std)                       # | smaller, scaled down by the number of layers the signal
                                                                                                    # | actually passes through. The reason is additive: each layer
                                                                                                    # | adds its output into a stream that is carried forward, so with
def takase_init_(model: nn.Module, hidden: int, effective_layers: int) -> None:                     # | L layers the stream accumulates L contributions, and starting
    """sigma_h^2 = 2/(5h) everywhere; sigma_out^2 = 1/(5 h l) on out-projections."""                # | each one at full size makes the stream grow with depth before
    sigma_h = math.sqrt(2.0 / (5.0 * hidden))                                                       # | a single gradient step. For a looped core the count is not the
    sigma_out = math.sqrt(1.0 / (5.0 * hidden * max(effective_layers, 1)))                          # | number of distinct layers but lP + rbar*lR + lC, which for the
    for name, module in model.named_modules():                                                      # | paper's model is 132 rather than 8. The names of the two
        if isinstance(module, (nn.Linear, nn.Embedding)):                                           # | projections are matched by string here because that is the
            std = sigma_out if name.split(".")[-1] in OUT_PROJ_NAMES else sigma_h                   # | only thing distinguishing them from the other linear layers.
            truncated_normal_(module.weight, std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


def sample_s0(shape, device, dtype, sigma_s: float = math.sqrt(0.4),                                # +-- THE START STATE SITS ON THE EMBEDDING SCALE ----------------
              generator: torch.Generator | None = None, deterministic: bool = False):               # | The initial latent state is drawn at variance 2/5, and that
    """s0 ~ truncated N(0, sigma_s^2), truncated at 3 sigma (Sec. 4.1).

    `deterministic=True` returns zeros, reproducing the "fixed s0" ablation that
    Sec. 3.1 argues against ("initializing the latent vector with a random state
    stabilizes the recurrence and promotes convergence to a steady state
    independent of initialization, i.e. path independence").
    """
    if deterministic:                                                                               # | number is not chosen. The embedding table starts at variance
        return torch.zeros(shape, device=device, dtype=dtype)                                       # | 2/(5h) and its output is multiplied by the square root of h,
    if generator is not None:                                                                       # | which lands the embedding at variance 2/5 exactly. Drawing the
        # trunc_normal_ takes no generator; sample via a clamped seeded normal.                     # | state at the same variance means the adapter receives both
        # torch requires the generator and the tensor to share a device, so sample                  # | halves of its input at comparable magnitudes; drawing it at
        # on the generator's device and move -- callers should not have to care.                    # | some other scale would make one half dominate before any
        t = torch.empty(shape, device=generator.device, dtype=torch.float32)                        # | learning happened. The deterministic option returns zeros and
        t.normal_(0.0, sigma_s, generator=generator).clamp_(-3 * sigma_s, 3 * sigma_s)              # | exists only so that the claim about path independence can be
        t = t.to(device)                                                                            # | tested rather than assumed. A seeded run samples a plain
    else:                                                                                           # | normal and clamps it, because torch's truncated normal takes
        t = torch.empty(shape, device=device, dtype=torch.float32)                                  # | no generator. The generator may live on a different device
        truncated_normal_(t, sigma_s)                                                               # | than the target, so callers seeding on the CPU need not know
    return t.to(dtype)                                                                              # | where the tensor ends up.
