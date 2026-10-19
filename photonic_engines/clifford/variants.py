"""
The algorithm set of the T1 / F2a comparison (docs/ALGORITHMS.md), as plain-torch REFERENCE steps.

Every exact Fock sampler in the literature that is not strong-simulation-then-sample is the same
chain rule (Clifford & Clifford 2018, arXiv:1706.01260, algorithm B): draw a uniform order sigma
of the n input columns, grow the output rows r_1..r_n one photon at a time, the weight of the
candidate rho at step k being |per(A[(r_1..r_{k-1}, rho), sigma(1..k)])|^2. What differs between
the published algorithms is ONLY how the k candidate-independent quantities of the step are
computed from the block B = A[r_{<k}, sigma(1..k)] ([k-1, k], repeated rows allowed), and how many
Gray/Guan addends that costs. This module holds one function per algorithm, written for one
stream on the CPU with the addend count returned next to the result, so the GPU kernels can be
checked against them and the counts along a real chain can be compared with the published ones
(bench/algorithm_terms.py). The GPU implementations are in sampler.py (algo=...) and the kernel
modules; these functions are the oracles, not the engine.

    "2018"    row-signed Glynn sweep, all k Laplace cofactors per(B \\ c) in one sweep, 2^(k-2) addends.
    "2020"    the same sweep collapsed on the OUTPUT multiplicities (Clifford & Clifford 2020,
              arXiv:2005.04214): prod_nu (s_nu + 1) / 2 addends, rho(c)^n on average.
    "input"   the sweep signs the INPUT columns and collapses on their multiplicities (the
              generalised algorithm B of Brod & Oszmaniec, arXiv:1906.06696, as `theboss` runs it):
              one sweep gives Q_mu = per([B; e_mu]) per distinct input column mu, prod_mu (s_mu + 1)
              / 2 addends, i.e. 2^(k-1) on a collision-free input, whatever the output collisions.
    "percof"  the k cofactors computed SEPARATELY, each by its own output-collapsed BB/FG sweep
              (PiquassoBoost's generalised C&C-B, Rakyta & Morse arXiv:2309.07027: one
              BBFGPermanentCalculatorRepeated call per input column per step): k x the "2020" count.
    "strong"  strong simulation then sampling (SLOS, then inverse-CDF): DeepQuantum's path, the
              memory-bound branch (docs/MEMORY_FRONTIER.md), `strong_then_sample` below.

Conventions: complex128 throughout; B[i, j] = A[rows[i], cols[j]]; a permanent of an empty block is 1.
"""
from __future__ import annotations

import math
from itertools import product as _product

import torch

ALGOS = ("2018", "2020", "input", "percof")


# ------------------------------------------------------------------------------- helpers

def _groups(ids):
    """Distinct values of `ids` (a list of ints, order of first appearance) and their multiplicities."""
    seen: dict = {}
    for x in ids:
        seen[x] = seen.get(x, 0) + 1
    return list(seen.keys()), list(seen.values())


def _sign_grid(mults, halve: bool):
    """All v in prod_nu {0..s_nu} as an int64 tensor [T, D] with the weight w(v) = prod (-1)^v C(s, v) [T].
    With `halve`, the complement symmetry v -> s - v is used (only valid when it is a symmetry of
    the summand, which every caller below guarantees): the LAST group must have odd multiplicity
    and its digit is restricted to 0..(s-1)/2, the weight is doubled.
    """
    ranges = [range(s + 1) for s in mults]
    if halve:
        assert mults[-1] % 2 == 1
        ranges[-1] = range((mults[-1] + 1) // 2)
    v = torch.tensor(list(_product(*ranges)), dtype=torch.int64).reshape(-1, len(mults))
    w = torch.ones(v.shape[0], dtype=torch.float64)
    for nu, s in enumerate(mults):
        w = w * torch.tensor([math.comb(s, int(x)) * (-1) ** int(x) for x in v[:, nu]], dtype=torch.float64)
    if halve:
        w = 2.0 * w
    return v, w


def _allbutone(s):
    """All-but-one products of s [T, K] complex -> [T, K] prod_{j != c} s_j, exactly (no division)."""
    T, K = s.shape
    out = torch.empty_like(s)
    for c in range(K):
        keep = [j for j in range(K) if j != c]
        out[:, c] = s[:, keep].prod(1) if keep else torch.ones(T, dtype=s.dtype)
    return out


# ------------------------------------------------------------------------------- the four steps

def cofactors_2018(B: torch.Tensor):
    """Clifford & Clifford 2018: per(B \\ c) for every column c of B [K1, K] in ONE Glynn sweep over
    delta in {+-1}^K1 with delta_0 = +1, 2^(K1-1) addends; K1 = 0 gives ones. Returns (P [K], addends).
    """
    K1, K = B.shape
    if K1 == 0:
        return torch.ones(K, dtype=torch.complex128), 1
    v, w = _sign_grid([1] * K1, halve=True)                                    # v_i = 1 <=> delta_i = -1; v_0 = 0
    delta = (1 - 2 * v).to(torch.complex128)                                   # [T, K1]
    s = delta @ B.to(torch.complex128)                                         # [T, K]
    P = (w.to(torch.complex128)[:, None] * _allbutone(s)).sum(0) / 2.0 ** K1
    return P, v.shape[0]


def cofactors_2020(B: torch.Tensor, rows):
    """Clifford & Clifford 2020: the same cofactors with the Glynn sum collapsed on the repeated
    rows of B (rows[i] = output mode of row i; equal ids = equal rows), prod_nu (s_nu + 1) addends,
    halved by complement symmetry when some multiplicity is odd. Returns (P [K], addends).
    """
    K1, K = B.shape
    if K1 == 0:
        return torch.ones(K, dtype=torch.complex128), 1
    rows = [int(r) for r in rows]
    ids, mults = _groups(rows)
    odd = [nu for nu, s in enumerate(mults) if s % 2 == 1]
    if odd:                                                                     # put an odd group last
        nu = odd[-1]; ids.append(ids.pop(nu)); mults.append(mults.pop(nu))
    b = torch.stack([B[rows.index(i)] for i in ids]).to(torch.complex128)      # [D, K] distinct rows
    v, w = _sign_grid(mults, halve=bool(odd))
    coef = (torch.tensor(mults, dtype=torch.int64)[None, :] - 2 * v).to(torch.complex128)   # [T, D]
    s = coef @ b                                                               # [T, K]
    P = (w.to(torch.complex128)[:, None] * _allbutone(s)).sum(0) / 2.0 ** K1
    return P, v.shape[0]


def qvec_input(B: torch.Tensor, cols):
    """Input-side collapse (Brod & Oszmaniec generalised algorithm B): sign the K columns of
    B [K1, K] with their multiplicities (cols[j] = input mode of column j) and take the FULL
    product over the K1 rows. One sweep gives, per distinct input column mu,
        Q_mu = 2^-K sum_v w(v) (s_mu - 2 v_mu) prod_i (sum_nu (s_nu - 2 v_nu) B[i, nu]) = per([B; e_mu]),
    and the candidate amplitude is amp_rho = sum_mu A[rho, mu] Q_mu = sum_c A[rho, cols[c]] per(B \\ c).
    prod_mu (s_mu + 1) addends, halved when a multiplicity is odd (the summand is symmetric under
    v -> s - v: sign (-1)^K from w, (-1)^K1 from the rows, -1 from the coefficient, K1 = K - 1).
    Returns (Q [D], dcol [D] the distinct input modes, addends).
    """
    K1, K = B.shape
    cols = [int(c) for c in cols]
    ids, mults = _groups(cols)
    odd = [nu for nu, s in enumerate(mults) if s % 2 == 1]
    if odd:
        nu = odd[-1]; ids.append(ids.pop(nu)); mults.append(mults.pop(nu))
    b = torch.stack([B[:, cols.index(i)] for i in ids], 1).to(torch.complex128)   # [K1, D] distinct columns
    v, w = _sign_grid(mults, halve=bool(odd))
    coef = (torch.tensor(mults, dtype=torch.int64)[None, :] - 2 * v).to(torch.complex128)   # [T, D]
    rs = coef @ b.T                                                            # [T, K1] row sums
    P = rs.prod(1) if K1 else torch.ones(v.shape[0], dtype=torch.complex128)   # [T] full product
    Q = (w.to(torch.complex128)[:, None] * coef * P[:, None]).sum(0) / 2.0 ** K
    return Q, ids, v.shape[0]


def cofactors_percof(B: torch.Tensor, rows):
    """PiquassoBoost form: every cofactor per(B \\ c) by its OWN output-collapsed sweep of the
    [K1, K1] block B \\ c (BBFGPermanentCalculatorRepeated, one call per input column), so K
    sweeps per step instead of one. Returns (P [K], addends = sum over the K sweeps).
    """
    K1, K = B.shape
    if K1 == 0:
        return torch.ones(K, dtype=torch.complex128), K
    P = torch.empty(K, dtype=torch.complex128)
    total = 0
    for c in range(K):
        keep = [j for j in range(K) if j != c]
        Pc, T = _permanent_collapsed(B[:, keep], rows)
        P[c] = Pc; total += T
    return P, total


def _permanent_collapsed(M: torch.Tensor, rows):
    """per(M) of a square block with repeated rows by the row-collapsed Glynn/BB-FG sum (the
    n-ary Gray code formula of Rakyta & Morse, eq. 8, on the row side): prod (s+1) addends, halved.
    """
    K1 = M.shape[0]
    rows = [int(r) for r in rows]
    ids, mults = _groups(rows)
    odd = [nu for nu, s in enumerate(mults) if s % 2 == 1]
    if odd:
        nu = odd[-1]; ids.append(ids.pop(nu)); mults.append(mults.pop(nu))
    b = torch.stack([M[rows.index(i)] for i in ids]).to(torch.complex128)      # [D, K1]
    v, w = _sign_grid(mults, halve=bool(odd))
    coef = (torch.tensor(mults, dtype=torch.int64)[None, :] - 2 * v).to(torch.complex128)
    s = coef @ b                                                               # [T, K1]
    return (w.to(torch.complex128) * s.prod(1)).sum() / 2.0 ** K1, v.shape[0]


def step_amplitudes(A: torch.Tensor, rows, cols, algo: str):
    """The m candidate amplitudes amp[rho] = per(A[rows + [rho], cols]) of one chain step, by `algo`,
    and the addends the step cost. A [m, n] complex128 (input columns of the circuit), rows the
    k-1 output modes chosen so far, cols the k input columns sigma(1..k).
    """
    rows = [int(r) for r in rows]; cols = [int(c) for c in cols]
    B = A[rows][:, cols]
    if algo == "2018":
        P, T = cofactors_2018(B)
    elif algo == "2020":
        P, T = cofactors_2020(B, rows)
    elif algo == "percof":
        P, T = cofactors_percof(B, rows)
    elif algo == "input":
        Q, dcol, T = qvec_input(B, cols)
        return A[:, dcol] @ Q, T
    else:
        raise ValueError(f"unknown algo {algo!r}: one of {ALGOS}")
    return A[:, cols] @ P, T


def addends(algo: str, rows, cols) -> int:
    """The addend count of one step WITHOUT computing it (closed forms, the GPU kernels' counts):
    2018: 2^(k-2); 2020: the halved mixed-radix product over the output multiplicities (exactly
    `multiplicity_tables`); input: the same over the input multiplicities of sigma(1..k); percof:
    k x the 2020 count of the [k-1, k-1] blocks (their row multiplicities are those of B).
    """
    from .cofactors_triton import multiplicity_tables  # lazy: the module stays importable without triton

    K1, K = len(rows), len(cols)
    if algo == "2018":
        return 1 if K1 == 0 else 1 << (K1 - 1)
    if algo in ("2020", "percof"):
        if K1 == 0:
            return 1 if algo == "2020" else K
        N = int(multiplicity_tables(torch.tensor([rows], dtype=torch.int64), K1)[3].item())
        return N if algo == "2020" else K * N
    if algo == "input":
        return int(multiplicity_tables(torch.tensor([cols], dtype=torch.int64), K)[3].item())
    raise ValueError(f"unknown algo {algo!r}: one of {ALGOS}")


def fock_sample_reference(U, in_modes, shots: int, algo: str = "2020", generator=None, stats: dict | None = None):
    """The chain rule on the CPU with the step of `algo`: occupations [shots, m] int64. Slow (Python
    per step, all addends materialised): n <= ~10. `stats["addends"]` receives one total per shot,
    so the count along a real chain can be compared between algorithms and with published counts.
    """
    U = torch.as_tensor(U, dtype=torch.complex128)
    m = U.shape[-1]
    n = len(in_modes)
    A = U[:, list(in_modes)]
    occ = torch.zeros(shots, m, dtype=torch.int64)
    for s in range(shots):
        sig = torch.randperm(n, generator=generator).tolist()
        rows: list[int] = []
        total = 0
        for k in range(1, n + 1):
            amp, T = step_amplitudes(A, rows, sig[:k], algo)
            total += T
            w = amp.real ** 2 + amp.imag ** 2
            rows.append(int(torch.multinomial(w / w.sum(), 1, generator=generator).item()))
        for r in rows:
            occ[s, r] += 1
        if stats is not None:
            stats.setdefault("addends", []).append(total)
    return occ


# ------------------------------------------------------------------------------- strong-then-sample

def strong_then_sample(U, in_modes, shots: int, generator=None, device: str = "cpu", dtype=torch.complex64,
                       stats: dict | None = None):
    """Strong simulation then sampling: the full output distribution by SLOS (photonic_engines.slos),
    then `shots` inverse-CDF draws from it. Occupations [shots, m] int64. The cost is the build
    (C(m+n-1, n) amplitudes, memory-bound: docs/MEMORY_FRONTIER.md) amortised over the run plus one
    lookup per sample; `stats` receives "build_s", "draw_s" and "states" so a run-of-S cost
    t_S = build/S + draw can be formed. dtype complex64 is the memory-frontier convention.
    """
    import time as _time

    from ..slos import FockSLOSKernel
    U = torch.as_tensor(U, dtype=torch.complex128)
    m = U.shape[-1]
    n = len(in_modes)
    occ_in = [0] * m
    for j in in_modes:
        occ_in[j] += 1
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = _time.perf_counter()
    kern = FockSLOSKernel(n, m, occ_in, device=device, dtype=dtype)
    p = kern.probs(U.to(device=device, dtype=dtype))
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = _time.perf_counter()
    gen = generator
    if device != "cpu" and generator is not None and generator.device.type != device:
        gen = torch.Generator(device=device).manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
    # inverse-CDF draw on the device, then ONLY the drawn states are turned into occupations:
    # `sample_from_probs` materialises every final occupation ([S, m] for 1e8 states, minutes),
    # which is a per-run cost no lookup sampler pays.
    from ..imperfections import draw_indices
    # the KERNEL's own layers (built on the device): a second `get_layers` on the CPU would rebuild
    # the whole numpy basis, minutes at 1e8 states
    idx = draw_indices(p, shots, gen)
    st = torch.as_tensor(kern.layers.states[n])                                # [S, n] photon positions, int32/int64
    t = st[idx.to(st.device)].to(torch.int64).cpu()                            # [shots, n]
    out = torch.zeros(shots, m, dtype=torch.int64)
    out.scatter_add_(1, t, torch.ones(shots, n, dtype=torch.int64))
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = _time.perf_counter()
    if stats is not None:
        stats.update(build_s=t1 - t0, draw_s=t2 - t1, states=int(p.numel()))
    return out
