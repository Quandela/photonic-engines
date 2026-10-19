"""Exact output probabilities of a PURE Gaussian input (no loss): one loop hafnian per pattern.

    P(n) = |N|^2 |lhaf(B[n], zeta[n])|^2 / prod_i n_i!

with (B, zeta, N) from `GaussianInput.bargmann(U)`. This is the per-pattern strong side, used as a
second oracle for the sampler (the first being `GaussianInput.truncated` through
`slos.ProductSLOSKernel`, which shares no code with this). A lossy (mixed) input needs the
2N x 2N form or the Fock-space route; it is not implemented here (docs/GAUSSIAN.md §5).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import numpy as np

from ..states import GaussianInput
from .hafnian import expand, lhaf


def pure_probability(state: GaussianInput, U, pattern: Sequence[int], device=None) -> float:
    """P(pattern) for the pure output state of `state` through U (no loss). Cost 2^(N/2) poly."""
    B, zeta, logN = state.bargmann(U)
    pattern = np.asarray(pattern, dtype=np.int64)
    A, D = expand(B, zeta, pattern)
    h = lhaf(A, D, device=device) if A.shape[0] else 1.0
    return float(abs(h) ** 2 * math.exp(2 * logN.real) / math.prod(math.factorial(int(k)) for k in pattern))


def pure_probabilities(state: GaussianInput, U, cutoff: int, device=None) -> dict[tuple[int, ...], float]:
    """{pattern: P} for every pattern with at most `cutoff` photons in total. Exponential in m: tests."""
    out = {}
    for pat in product(range(cutoff + 1), repeat=state.m):
        if sum(pat) <= cutoff:
            out[pat] = pure_probability(state, U, pat, device=device)
    return out
