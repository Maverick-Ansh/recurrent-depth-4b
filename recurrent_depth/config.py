"""Architecture config.

Sec. 3.2: "we can summarize the architecture by the triplet (lP, lR, lC),
describing the number of layers in each stage, and by the number of recurrences
r, which may vary in each forward pass."

Paper configs:
    small ablations : (lP, lR, lC) = (1, 4, 1), h = 1024
    main model      : (lP, lR, lC) = (2, 4, 2), h = 5280, 55 heads of size 96,
                      MLP inner 17920, RMSNorm eps 1e-6, RoPE base 50000,
                      vocab 65536, tied embeddings, rbar = 32, k = 8.
"""

from dataclasses import dataclass, asdict


@dataclass
class RecurrentDepthConfig:
    vocab_size: int = 260
    hidden: int = 512
    n_heads: int = 8
    mlp_inner: int = 1376        # ~2.7x hidden, multiple of 32 (SwiGLU convention)
    l_prelude: int = 1           # lP
    l_core: int = 4              # lR
    l_coda: int = 1              # lC
    rope_base: float = 50000.0   # Sec. 3.2
    norm_eps: float = 1e-6       # Sec. 4.1
    max_seq: int = 512

    # --- training objective (Sec. 3.3) ---
    mean_recurrence: float = 8.0  # rbar  (paper: 32)
    sigma: float = 0.5            # Lambda spread (paper: 1/2)
    backprop_depth: int = 4       # k     (paper: 8)

    # --- ablation switches (each names the paper claim it probes) ---
    norm_style: str = "sandwich"  # "pre"  reproduces Bad Run 2 (Sec. 4.3)
    norm_affine: bool = True      # False  reproduces Bad Run 1 parameter-free norms
    injection: str = "concat"     # "concat" | "add" | "none"  (Sec. 3.2 adapter)
    random_s0: bool = True        # False = fixed s0 (Sec. 3.1 path independence)
    tie_embeddings: bool = True   # Sec. 3.2 "projection ... using tied embeddings E^T"

    def to_dict(self):
        return asdict(self)
