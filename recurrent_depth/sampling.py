"""The random-unrolling distribution Lambda, Sec. 3.3.

Paper (verbatim):

    "We optimize the expectation of the loss function L over random samples x
     from distribution X and random iteration counts r from distribution Lambda.

        L(theta) = E_{x in X} E_{r ~ Lambda} L( m_theta(x, r), x' ).

     [...] We choose Lambda to be a log-normal Poisson distribution. Given a
     targeted mean recurrence rbar + 1 and a variance that we set to sigma = 1/2,
     we can sample from this distribution via

        tau ~ N( log(rbar) - 1/2 sigma^2, sigma )          (1)
        r   ~ P( e^tau ) + 1                               (2)"

The paper writes "variance ... sigma = 1/2" but the standard log-normal
parametrisation with the -sigma^2/2 correction makes sigma the *standard
deviation*.  We verify this reading against the paper's own Figure 3, which is
drawn for rbar = 32 and annotated Mean = 33.0, Median = 29.0, Mode = 24.0:

    sigma as std : E[r] = rbar + 1 = 33.0, median = exp(log 32 - 0.125)+1 = 29.2
    sigma as var : E[r] = 32*exp(0.125)+1 = 37.3, median = 33.0

Only the "sigma is the std" reading reproduces the caption, so that is what we
implement.  `selftest_matches_figure3()` asserts it numerically.

Sec. 4.1, "Locked-Step Sampling":

    "To enable synchronization between parallel workers, we sample a single
     depth r for each micro-batch of training, which we synchronize across
     workers."

so `sample_r` returns ONE integer per micro-batch, not one per sequence.
"""

from __future__ import annotations
import math
import torch


def sample_r(rbar: float, sigma: float = 0.5, generator: torch.Generator | None = None,
             n: int = 1, device="cpu") -> torch.Tensor:
    """Draw n samples from Lambda: tau ~ N(log(rbar) - sigma^2/2, sigma); r ~ Poisson(e^tau) + 1."""
    tau = torch.normal(
        mean=torch.full((n,), math.log(rbar) - 0.5 * sigma ** 2, device=device),
        std=sigma, generator=generator,
    )
    return torch.poisson(torch.exp(tau), generator=generator).long() + 1


def selftest_matches_figure3(tol_mean=0.4, tol_median=1.0) -> dict:
    """Assert our Lambda reproduces the moments annotated on the paper's Figure 3."""
    g = torch.Generator().manual_seed(0)
    r = sample_r(32.0, 0.5, generator=g, n=2_000_000).float()
    stats = {"mean": r.mean().item(), "median": r.median().item(), "mode": r.mode().values.item()}
    assert abs(stats["mean"] - 33.0) < tol_mean, f"Fig.3 says Mean=33.0, got {stats['mean']:.2f}"
    assert abs(stats["median"] - 29.0) < tol_median, f"Fig.3 says Median=29.0, got {stats['median']}"
    return stats


if __name__ == "__main__":
    print(selftest_matches_figure3())
