"""ONE amplitude of a `states.ProductSuperposition` input, without building the basis.

`kernel.ProductSLOSKernel` produces every output amplitude at once and is bounded by the graded
basis. This module answers the other question - one output pattern, for as many sources as a
2^k Glynn sum allows - and is the inner loop any sampler for this input type will need.

Derivation (d = 2, so every factor is linear; a[i] = (c_i, s_i) for source i, feeding input mode
`modes[i]`). The output state is

    prod_i ( c_i + s_i sum_j U[j, modes[i]] b_j^dag ) |0> ,

so the amplitude of an output occupation t with k = |t| photons is the coefficient of
prod_j z_j^{t_j} in prod_i (c_i + s_i sum_j U[j, modes[i]] z_j), times prod_j sqrt(t_j!).

Give every photon its own variable, z_j -> sum_{s : r_s = j} x_s over the k slots (r_s lists the
output rows with multiplicity), and mark the emitting factors with lambda:

    Q(lambda, x) = prod_i ( c_i + lambda s_i sum_s x_s U[r_s, modes[i]] ) .

The lambda^k part is homogeneous of degree k in x, and the only degree-k monomial with every
exponent odd is x_1...x_k, so Glynn's sign extraction is exact on it:

    amp(t) = 1 / (2^k prod_j sqrt(t_j!))
             * sum_{delta in {+-1}^k} (prod_s delta_s) [lambda^k] prod_i (c_i + lambda a_i(delta)),
      a_i(delta) = s_i sum_s delta_s U[r_s, modes[i]] .

The lambda^k coefficient is one truncated polynomial product - n steps on a (k+1)-vector - so the
cost is O(2^k n k): exponential in the DETECTED photon count, polynomial in the source count. The
sum runs over all 2^k signs and not 2^{k-1}: the global flip is a symmetry of the summand only
when every c_i is zero, i.e. in the Fock case.

WHY THE lambda FILTER AND NOT A PERMANENT. Homogenising with one extra variable instead makes the
coefficient a permanent: with M the n x n matrix whose rows are the k output rows followed by
n - k copies of the constant row (c_1 ... c_n), amp(t) = per(M) / ((n-k)! prod_j sqrt(t_j!)), and
Glynn's sum collapses those identical rows to a binomial weight over their n - k + 1 sign counts.
That identity is exact and `tests/slos/test_product.py` still checks against it at small n - but
it is NOT how to evaluate it. The collapse multiplies the constant row by (g - 2v), g = n - k, so
each summand grows like (g |c|)^n while the answer stays tiny: measured on a GPU (LOG 2026-09-19),
the relative error passes 1 at n = 32 and reaches 1e19 at n = 48. The lambda filter never forms
that amplified quantity; on the U = identity case, where the amplitude is analytic at any size, it
holds 5e-16 out to n = 48.

Plain batched torch (CPU or CUDA), chunked over the sign vectors. Folding this into the Triton
cofactor kernels is docs/SUPERPOSITION.md §7.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _sign_block(start: int, stop: int, k: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Sign vectors `start`..`stop` of {+-1}^k as a [stop - start, k] tensor (bit s = sign s)."""
    idx = torch.arange(start, stop, device=device, dtype=torch.int64).unsqueeze(1)     # [C, 1]
    bits = torch.arange(k, device=device, dtype=torch.int64).unsqueeze(0)              # [1, k]
    return (1 - 2 * ((idx >> bits) & 1)).to(dtype)                                     # [C, k]


def product_amplitude(state, U: torch.Tensor, occupation: Sequence[int],
                      chunk: int | None = None, device=None,
                      dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Amplitude <t| phi(U) |state> for one output occupation `t`, by the sum above.

    state       : a `states.ProductSuperposition` with d = 2 (the derivation needs linear factors)
    U           : [m, m], any dtype/device (moved to `dtype` / `device`)
    occupation  : length-m output pattern; k = sum(t) photons, 0 if k > sources
    chunk       : sign vectors evaluated per batch (peak memory ~ chunk * (n + k) complex);
                  default: whatever keeps that near 2^24 elements
    Returns a 0-dim complex tensor. Cost O(2^k n k) time, no basis and no state vector. Use
    complex128: complex64 loses the amplitude to rounding well before k = 20.
    """
    if state.d != 2:
        raise ValueError(f"product_amplitude needs d = 2 (linear factors), got d = {state.d}")
    t = [int(x) for x in occupation]
    if len(t) != state.m or any(x < 0 for x in t):
        raise ValueError(f"occupation must be {state.m} non-negative ints")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    n, k = state.sources, sum(t)
    if k > n:                                        # more photons than the sources can emit
        return torch.zeros((), dtype=dtype, device=dev)
    U = torch.as_tensor(U, dtype=dtype, device=dev)
    cols = torch.as_tensor(list(state.modes), dtype=torch.int64, device=dev)
    amps = torch.tensor(state.amps, dtype=dtype, device=dev)                           # [n, 2] (copy: read-only)
    c, s = amps[:, 0], amps[:, 1]
    rows = [j for j, x in enumerate(t) for _ in range(x)]
    V = s.unsqueeze(0) * U[torch.as_tensor(rows, dtype=torch.int64, device=dev)][:, cols] if rows \
        else torch.zeros((0, n), dtype=dtype, device=dev)                              # [k, n]
    if chunk is None:
        chunk = max(1, min(1 << k, (1 << 24) // max(1, n + k)))
    total = torch.zeros((), dtype=dtype, device=dev)
    for start in range(0, 1 << k, chunk):
        delta = _sign_block(start, min(start + chunk, 1 << k), k, dtype, dev)          # [C, k]
        a = delta @ V                                                                  # [C, n]
        poly = torch.zeros((delta.shape[0], k + 1), dtype=dtype, device=dev)           # [C, k+1]
        poly[:, 0] = 1
        for i in range(n):                           # poly_d <- c_i poly_d + a_i poly_{d-1}
            poly = poly * c[i] + torch.nn.functional.pad(poly[:, :-1], (1, 0)) * a[:, i:i + 1]
        total = total + (poly[:, k] * delta.prod(dim=1)).sum()
    norm = math.sqrt(math.prod(math.factorial(x) for x in t))
    return total / (2.0 ** k) / norm


def product_probability(state, U: torch.Tensor, occupation: Sequence[int], **kw) -> torch.Tensor:
    """|`product_amplitude`|^2 for one output pattern (same arguments)."""
    a = product_amplitude(state, U, occupation, **kw)
    return (a.conj() * a).real
