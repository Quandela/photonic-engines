"""Loop hafnians for the Gaussian chain rule (docs/GAUSSIAN.md §3).

Three evaluations of the same quantity, from slow and obvious to the one the sampler uses:

  lhaf_brute      recursion over the first vertex (loop, or pair with any other vertex):
                  (N-1)!! terms. Tests only - the oracle for the other two.
  lhaf            the pair sieve below, one value.
  lhaf_poly       the pair sieve with a loop vector that is LINEAR in a variable z,
                  D(z) = D0 + z D1, returning the coefficients of the polynomial lhaf(A, D(z)).
                  That is the object the chain rule needs: z is the current mode's Bargmann
                  variable, so one call prices every photon number of that mode at once.

THE PAIR SIEVE (our derivation, docs/GAUSSIAN.md §3.1). Pad N to even 2n with one vertex of loop
weight 1 and no edges when N is odd (lhaf unchanged). Pair vertex a with X(a) = a + n (mod 2n),
C = A X, i.e. C[a, b] = A[a, X(b)]. A loop-perfect matching M (edges of A, loops of D) together
with the n pairing links decomposes uniquely into disjoint cycles and paths whose two ends carry
loops. A cycle through k pairs is a closed walk of C of length k (2k orientations and rotations),
a path through k pairs is (XD)^T C^(k-1) D (2 orientations). So, with lambda counting pairs,

    lhaf(A, D) = sum_{S subset [n]} (-1)^(n-|S|) [lambda^n] exp( sum_{k=1..n} lambda^k g_k(S) ),
    g_k(S) = tr(C_S^k) / (2k) + 1/2 (X D_S)^T C_S^(k-1) D_S ,

C_S, D_S being C and D with the rows/columns of the pairs outside S zeroed. The inclusion-exclusion
over S keeps exactly the configurations that visit every pair, and [lambda^n] then visits each one
once. 2^n terms of O(n^4) (v1: powers by repeated products; v2 will use n/2 products, docs §6).
The diagonal of A is never used (a step a -> X(a) revisits a pair and is sieved out).

With D = D0 + z D1 the path term is quadratic in z,
    1/2 (XD0)^T C^(k-1) D0 + z (XD0)^T C^(k-1) D1 + z^2 1/2 (XD1)^T C^(k-1) D1
(C^T = X C X makes the cross term symmetric), and the exponential is carried as a series in lambda
whose coefficients are polynomials in z, truncated at degree 2n.

Everything is complex128. No collision (repeated-row) discount yet: a row repeated t times is t
separate vertices, so the cost is 2^(N/2) whatever the pattern (docs/GAUSSIAN.md §6, v2 item).
"""
from __future__ import annotations

import math

import numpy as np
import torch

EPS = float(np.finfo(np.float64).eps)


# ---- oracle -------------------------------------------------------------------------------------
def lhaf_brute(A: np.ndarray, D: np.ndarray) -> complex:
    """Loop hafnian by recursion over the first vertex, sum over all loop-perfect matchings of the
    product of edge weights A[i, j] and loop weights D[i]. (N-1)!!-type cost: tests only (N <= 12).
    The diagonal of A is ignored; loops come from D.
    """
    A = np.asarray(A, dtype=np.complex128); D = np.asarray(D, dtype=np.complex128)

    def rec(idx: tuple[int, ...]) -> complex:
        if not idx:
            return 1.0 + 0j
        a, rest = idx[0], idx[1:]
        tot = D[a] * rec(rest)
        for t, b in enumerate(rest):
            tot += A[a, b] * rec(rest[:t] + rest[t + 1:])
        return tot

    return complex(rec(tuple(range(A.shape[0]))))


def expand(B: np.ndarray, gamma: np.ndarray, counts) -> tuple[np.ndarray, np.ndarray]:
    """Repeat row/column i of B `counts[i]` times and the loop vector likewise: the matrix whose
    loop hafnian is Π counts! times the coefficient of Π w_i^counts_i in exp(1/2 w^T B w + gamma^T w).
    Returns (A [N, N], D [N]); the diagonal of A is the (unused) diagonal of B repeated.
    """
    idx = np.repeat(np.arange(len(counts)), np.asarray(counts, dtype=np.int64))
    return B[np.ix_(idx, idx)], gamma[idx]


# ---- the pair sieve -----------------------------------------------------------------------------
def _pad(A: torch.Tensor, D0: torch.Tensor, D1: torch.Tensor):
    """Pad an odd vertex count with one isolated vertex of loop weight 1 (z-independent)."""
    N = A.shape[0]
    if N % 2 == 0:
        return A, D0, D1
    Ap = torch.zeros((N + 1, N + 1), dtype=A.dtype, device=A.device); Ap[:N, :N] = A
    one = torch.ones(1, dtype=A.dtype, device=A.device); zero = torch.zeros(1, dtype=A.dtype, device=A.device)
    return Ap, torch.cat([D0, one]), torch.cat([D1, zero])


def _chunk_size(n: int, budget_bytes: int) -> int:
    """Subsets per batch so that ~4 [chunk, 2n, 2n] complex128 temporaries fit the budget."""
    per = 4 * (2 * n) ** 2 * 16 + 64 * (2 * n + 1) * (n + 1)
    return max(1, min(1 << n, budget_bytes // max(per, 1)))


def lhaf_poly(A, D0, D1=None, device: str | torch.device | None = None,
              budget_bytes: int | None = None, return_abs: bool = False):
    """Coefficients c_0..c_N of the polynomial z -> lhaf(A, D0 + z D1), complex128 [N + 1].

    A [N, N] symmetric (diagonal ignored), D0, D1 [N]. D1 = None means a constant loop vector
    (then only c_0 is non-zero). Runs on `device` (default: the device of A when it is a tensor,
    else CPU); the 2^(N/2) subsets are processed in batches bounded by `budget_bytes` (default
    256 MiB on CPU, 4 GiB on CUDA).
    Cost: 2^(N/2) x (N/2 products of N x N matrices + N/2 x N^2 series work).

    `return_abs=True` also returns sum_S |term_S| per coefficient, float64 [N + 1]: the sieve is an
    alternating sum, so max(abs) / max|c| is its condition number and eps x that ratio is the
    rounding error to expect (docs/GAUSSIAN.md §6; measured 4e7 at N = 26 on random O(1) matrices).
    """
    if device is None:
        device = A.device if isinstance(A, torch.Tensor) else "cpu"
    dev = torch.device(device)
    if budget_bytes is None:
        budget_bytes = (1 << 32) if dev.type == "cuda" else (1 << 28)
    A = torch.as_tensor(A, dtype=torch.complex128).to(dev)
    D0 = torch.as_tensor(D0, dtype=torch.complex128).to(dev).reshape(-1)
    D1 = torch.zeros_like(D0) if D1 is None else torch.as_tensor(D1, dtype=torch.complex128).to(dev).reshape(-1)
    N = A.shape[0]
    out = torch.zeros(N + 1, dtype=torch.complex128, device=dev)
    if N == 0:
        out[0] = 1.0
        return (out, out.real.abs().clone()) if return_abs else out
    A, D0, D1 = _pad(A, D0, D1)
    n = A.shape[0] // 2
    L = 2 * n + 1                                                     # z-degree bound + 1
    X = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
    C = A[:, X]                                                       # C[a, b] = A[a, X(b)]
    XD0, XD1 = D0[X], D1[X]
    bits = torch.arange(n, device=dev)
    chunk = _chunk_size(n, budget_bytes)
    acc = torch.zeros(L, dtype=torch.complex128, device=dev)
    acc_abs = torch.zeros(L, dtype=torch.float64, device=dev)
    for start in range(0, 1 << n, chunk):
        stop = min(1 << n, start + chunk)
        S = torch.arange(start, stop, device=dev)
        s = ((S.unsqueeze(1) >> bits) & 1).to(torch.complex128)     # [c, n]
        s = torch.cat([s, s], dim=1)                                  # [c, 2n] both vertices of a pair
        CS = s.unsqueeze(2) * C.unsqueeze(0) * s.unsqueeze(1)         # [c, 2n, 2n]
        d0, d1 = s * D0, s * D1                                       # [c, 2n]
        x0, x1 = s * XD0, s * XD1
        # g_k(z) = tr(C^k)/(2k) + g0 + g1 z + g2 z^2, k = 1..n        # [c, n+1, 3]
        g = torch.zeros((stop - start, n + 1, 3), dtype=torch.complex128, device=dev)
        P = CS; w0, w1 = d0, d1                                       # C^(k-1) D at k = 1
        for k in range(1, n + 1):
            if k > 1:
                P = P @ CS
                w0 = (CS @ w0.unsqueeze(2)).squeeze(2); w1 = (CS @ w1.unsqueeze(2)).squeeze(2)
            tr = torch.diagonal(P, dim1=1, dim2=2).sum(1)
            g[:, k, 0] = tr / (2 * k) + 0.5 * (x0 * w0).sum(1)
            g[:, k, 1] = (x0 * w1).sum(1)
            g[:, k, 2] = 0.5 * (x1 * w1).sum(1)
        # F = exp(sum_k lambda^k g_k): F_k = (1/k) sum_i i g_i F_{k-i}, F_k a z-polynomial   [c, L]
        F = [torch.zeros((stop - start, L), dtype=torch.complex128, device=dev) for _ in range(n + 1)]
        F[0][:, 0] = 1.0
        for k in range(1, n + 1):
            t = torch.zeros_like(F[0])
            for i in range(1, k + 1):
                f = F[k - i]
                gi = g[:, i]
                t += i * gi[:, 0:1] * f
                t[:, 1:] += i * gi[:, 1:2] * f[:, :-1]
                t[:, 2:] += i * gi[:, 2:3] * f[:, :-2]
            F[k] = t / k
        sign = 1.0 - 2.0 * ((n - s[:, :n].real.sum(1).round().long()) % 2).to(torch.float64)
        acc += (sign.to(torch.complex128).unsqueeze(1) * F[n]).sum(0)
        if return_abs:
            acc_abs += F[n].abs().sum(0)
    out[:] = acc[: N + 1]
    return (out, acc_abs[: N + 1]) if return_abs else out


# ---- v2: one matrix, many loop vectors, subsets grouped by size ---------------------------------
_BUCKET_FRACTIONS = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_TABLE_MAX = 16                  # up to this many free pairs the subset lists are tabulated (<= 16 MB each)
_TABLES: dict = {}


def _bucket_sizes(n: int, dev: torch.device) -> list[int]:
    """Padded subset sizes.

    On a GPU, few buckets while the sieve is small (launch-bound: one bucket up to 12 pairs, i.e. the
    masked layout of v1, three up to 16); the finer split where the flops dominate, and always on a CPU
    (measured there: 8 buckets 0.13 s against 3 buckets 0.24 s at N = 24).
    """
    if dev.type == "cpu":
        fr = _BUCKET_FRACTIONS
    else:
        fr = (1.0,) if n <= 12 else (0.5, 0.75, 1.0) if n <= _TABLE_MAX else _BUCKET_FRACTIONS
    return sorted({max(1, math.ceil(n * f)) for f in fr})


def _subset_table(n: int, dev: torch.device):
    """Tabulated `_free_subsets` for n <= _TABLE_MAX, cached per (n, device).

    No device-side enumeration and no host synchronisation per call. The first GPU run measured v2 at
    0.2-0.3x of v1 below N ~ 28 (run 20260924_071956); the per-chunk `bool(sel.any())` syncs and
    boolean indexing of the on-the-fly path are the suspected cause, not yet isolated.
    """
    key = (n, str(dev))
    if key not in _TABLES:
        from itertools import combinations
        out, lo = [], 0
        for pb in _bucket_sizes(n, dev):
            idx, sgn = [], []
            for p in range(lo + 1, pb + 1):
                cb = np.fromiter((x for c in combinations(range(n), p) for x in c), dtype=np.int64).reshape(-1, p)
                pad = np.full((cb.shape[0], pb - p), 2 * n, dtype=np.int64)
                idx.append(np.concatenate([cb, pad, cb + n, pad], 1))
                sgn.append(np.full(cb.shape[0], -1.0 if (n - p) % 2 else 1.0))
            out.append((pb, torch.as_tensor(np.concatenate(idx), device=dev),
                        torch.as_tensor(np.concatenate(sgn), device=dev)))
            lo = pb
        _TABLES[key] = out
    return _TABLES[key]


def _free_subsets(n: int, dev: torch.device, chunk_of):
    """Every non-empty subset of n pairs, regrouped into a few padded sizes pb. Yields (pb, idx [c, 2 pb]
    int64, sign [c] float64 = (-1)^(n - |T|)), with c <= chunk_of(pb) (the batch size is set per bucket, so
    a small bucket is not cut into small batches by a chunk sized for the largest). Pair i is vertices i and
    i + n; padding uses the dummy vertex 2n (a zero row and column of the extended C, zero loops), so a
    padded subset contributes exactly what the unpadded one does. Tabulated up to _TABLE_MAX pairs; above,
    the subsets of each size p are generated on the device by colexicographic unranking of their ranks
    0 .. C(n, p) - 1 (`_unrank`): no table (it would be 2^30 x 60 entries at n = 30), no host sync.

    Until 2026-09-24 the on-device path cut a range of bitmasks into the buckets, which left each bucket
    ~1/8 of a batch and issued every bucket's kernels once per range: 60939 launches for one sieve at
    N = 36 with the GPU 20 % busy (run 20260924_124226).
    """
    if n <= _TABLE_MAX:
        for pb, idx_all, sgn_all in _subset_table(n, dev):
            chunk = chunk_of(pb)
            for start in range(0, idx_all.shape[0], chunk):
                yield pb, idx_all[start:start + chunk], sgn_all[start:start + chunk]
        return
    binom = _binom_table(n, dev)
    lo = 0
    for pb in _bucket_sizes(n, dev):
        chunk = chunk_of(pb)
        pieces, filled = [], 0
        for p in range(lo + 1, pb + 1):
            total, start = math.comb(n, p), 0
            while start < total:                                        # fill batches across sizes p
                stop = min(total, start + chunk - filled)
                pieces.append((p, start, stop))
                filled += stop - start
                start = stop
                if filled == chunk:
                    yield pb, *_assemble(pieces, n, pb, binom, dev)
                    pieces, filled = [], 0
        if pieces:
            yield pb, *_assemble(pieces, n, pb, binom, dev)
        lo = pb


def _binom_table(n: int, dev: torch.device) -> torch.Tensor:
    """T[i, c] = C(c, i), i = 0..n, c = 0..n - 1, int64 on `dev` (cached): the colex unranking table."""
    key = ("binom", n, str(dev))
    if key not in _TABLES:
        T = np.array([[math.comb(c, i) for c in range(n)] for i in range(n + 1)], dtype=np.int64)
        _TABLES[key] = torch.as_tensor(T, device=dev)
    return _TABLES[key]


def _unrank(ranks: torch.Tensor, p: int, binom: torch.Tensor) -> torch.Tensor:
    """The p-subsets {c_1 < .. < c_p} of 0..n-1 with colex rank r = sum_i C(c_i, i), for ranks [c] int64:
    c_i = the largest c with C(c, i) <= r, taken for i = p .. 1, r -= C(c_i, i). Returns [c, p] int64.
    """
    r = ranks.clone()
    out = torch.empty((r.shape[0], p), dtype=torch.int64, device=r.device)
    for i in range(p, 0, -1):
        row = binom[i]                                                  # C(c, i), non-decreasing in c
        c = torch.searchsorted(row, r, right=True) - 1
        out[:, i - 1] = c
        r -= row[c]
    return out


def _assemble(pieces, n: int, pb: int, binom: torch.Tensor, dev: torch.device):
    """(idx [c, 2 pb], sign [c]) for rank ranges pieces = [(p, start, stop)] of the p-subsets, padded to pb."""
    idx, sgn = [], []
    for p, start, stop in pieces:
        pos = _unrank(torch.arange(start, stop, device=dev, dtype=torch.int64), p, binom)
        pad = torch.full((stop - start, pb - p), 2 * n, dtype=torch.int64, device=dev)
        idx.append(torch.cat([pos, pad, pos + n, pad], 1))
        sgn.append(torch.full((stop - start,), -1.0 if (n - p) % 2 else 1.0, dtype=torch.float64, device=dev))
    return (idx[0], sgn[0]) if len(idx) == 1 else (torch.cat(idx), torch.cat(sgn))


_FUSED = None                    # None: fused layout of `_terms` on CUDA up to _FUSED_MAX pairs, else loops; True / False: force
_FUSED_MAX = 16


def _fused(dev, n: int) -> bool:
    """Whether `_terms` uses the fused layout on `dev` for a sieve of n pairs (see `_FUSED`).

    Measured (H100, run 20260924_141521, one sieve, v2 V = 1): the fused layout wins while the sieve is
    launch-bound, 2.6 vs 4.5 ms at N = 20 and 11.2 vs 17.3 ms at N = 28, and loses once it is compute-bound,
    277 vs 251 ms at N = 36 and 1367 vs 1176 ms at N = 40 (its concatenations are the top kernel). N = 32 was
    measured only fused. On two CPU cores the fused layout is x0.4-0.9 of the loops at every size.
    """
    if _FUSED is not None:
        return bool(_FUSED)
    return dev is not None and torch.device(dev).type == "cuda" and n <= _FUSED_MAX


_TRITON = None                   # None: the Triton kernels on CUDA from _TRITON_FROM pairs; True / False: force
_TRITON_FROM = 1                 # every size: faster than the fused torch layout at N = 16-28 too (0.8 vs 2.2 ms at
                                 # N = 16, 3.7 vs 11.2 ms at N = 28; v2.6, run 20260924_190851)


def _use_triton(dev, n: int) -> bool:
    """Whether `_terms` computes the powers, traces and paths with `sieve_triton.powers_paths` (see `_TRITON`).
    Forced True on a CPU device it runs under the Triton interpreter (TRITON_INTERPRET=1; tests only). Never past
    n = 32 pairs unless forced (the kernel holds subset matrices of at most 64 vertices).
    """
    if _TRITON is not None:
        return bool(_TRITON)
    if dev is None or torch.device(dev).type != "cuda" or n < _TRITON_FROM:
        return False
    if n > 32:                                  # subset matrices up to 2n vertices: the kernel's limit is 64 (2026-09-29:
        return False                            # before, CUDA sieves with n > 32 raised in powers_paths; torch path instead)
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _chunk_size_multi(n: int, p: int, V: int, budget_bytes: int, dev=None) -> int:
    """Subsets per batch for `lhaf_poly_multi` at (bucket) subset size p.

    ~h + 2 [c, 2p, 2p] matrices, the path vectors [c, 2p, 2V] (x2), g [c, V, n+1, 3] and the series
    F [c, V, n+1, 2n+1], complex128; in the fused layout (`_fused`) the n path vectors stacked
    [c, n, 2p, 2V] (x2 with the copy), F [c, V, n+1, 2n+3] and its [c, V, 3n, 2n+1] operand.
    """
    if _use_triton(dev, n):                                            # no subset matrices off chip
        per = 16 * (2 * p * 2 + (n + 1) + 3 * V * (n + 1) + V * (n + 1) * 3 + 2 * V * (n + 1) * (2 * n + 1))
    elif not _fused(dev, n):
        per = 16 * (((n + 1) // 2 + 2) * (2 * p) ** 2 + 4 * (2 * p) * 2 * V + V * (n + 1) * 3 + 2 * V * (n + 1) * (2 * n + 1))
    else:
        per = 16 * (((n + 1) // 2 + 2) * (2 * p) ** 2 + 2 * (n + 1) * (2 * p) * 2 * V + V * (n + 1) * 3
                    + 2 * V * (n + 1) * (2 * n + 3) + 3 * V * n * (2 * n + 1))
    return max(1, budget_bytes // max(per, 1))


def _terms(Ce, Ye, XYe, idx, weight, n: int, V: int, acc, acc_abs, return_abs: bool, rows=None):
    """Add sum_S weight_S [lambda^n] exp(sum_k lambda^k g_k(S)) over the subsets whose (padded) vertex
    lists are the rows of idx [c, 2p] into acc [V, 2n+1] (and sum |.| into acc_abs). With `rows` (a list),
    append |term_S| [c, V, 2n+1] float64 instead and leave acc alone (`_choose_pairing`).
    """
    dev = Ce.device
    c = idx.shape[0]; h = (n + 1) // 2; Lz = 2 * n + 1
    ks = torch.arange(1, n + 1, device=dev, dtype=torch.float64).to(torch.complex128)
    tri = _use_triton(dev, n)
    fused = not tri and _fused(dev, n)
    g = torch.empty((c, V, n + 1, 3), dtype=torch.complex128, device=dev)
    if tri:                                                             # matrix part on chip (sieve_triton)
        from .sieve_triton import powers_paths
        tr, pa = powers_paths(Ce, Ye, XYe, idx, n, V)
        g[:, :, 1:, 0] = (tr[:, 1:] / (2 * ks)).unsqueeze(1) + 0.5 * pa[:, :, 1:, 0]
        g[:, :, 1:, 1] = pa[:, :, 1:, 1]
        g[:, :, 1:, 2] = 0.5 * pa[:, :, 1:, 2]
    else:
        CS = Ce[idx.unsqueeze(2), idx.unsqueeze(1)]                     # [c, 2p, 2p]  restricted C_S
        xy = XYe[idx]
        x0, x1 = xy[:, :, :V], xy[:, :, V:]
        tr = torch.empty((c, n + 1), dtype=torch.complex128, device=dev)
        P = [None, CS]
        for _ in range(2, h + 1):
            P.append(P[-1] @ CS)
        PT = [None] + [q.transpose(1, 2) for q in P[1:]]
        for k in range(1, n + 1):
            a, b = (k + 1) // 2, k // 2
            tr[:, k] = torch.diagonal(P[a], dim1=1, dim2=2).sum(1) if b == 0 else (P[a] * PT[b]).sum((1, 2))
    if not fused:                                                       # per-k loops: fewer copies
        w = None if tri else Ye[idx]                                    # [c, 2p, 2V]  C_S^(k-1) Y_S
        for k in range(1, n + 1) if not tri else ():
            if k > 1:
                w = CS @ w
            w0, w1 = w[:, :, :V], w[:, :, V:]
            g[:, :, k, 0] = (tr[:, k] / (2 * k)).unsqueeze(1) + 0.5 * (x0 * w0).sum(1)
            g[:, :, k, 1] = (x0 * w1).sum(1)
            g[:, :, k, 2] = 0.5 * (x1 * w1).sum(1)
        gi = g[:, :, 1:, :] * ks.view(1, 1, n, 1)                       # i g_i, i = 1..n
        if tri:
            from .sieve_triton import series
            Fn = series(gi)                                             # [c, V, Lz]
        else:
            F = torch.zeros((c, V, n + 1, Lz), dtype=torch.complex128, device=dev)
            F[:, :, 0, 0] = 1.0
        for k in range(1, n + 1) if not tri else ():
            dk = min(2 * k, Lz - 1) + 1                                 # F_k has z-degree <= 2k
            Fr = F[:, :, :k, :dk]                                       # F_0 .. F_{k-1}
            G = gi[:, :, :k].flip(2)                                    # i = k .. 1, paired with F_{k-i}
            t = torch.einsum("cvi,cviz->cvz", G[..., 0], Fr)
            t[..., 1:] += torch.einsum("cvi,cviz->cvz", G[..., 1], Fr[..., :-1])
            t[..., 2:] += torch.einsum("cvi,cviz->cvz", G[..., 2], Fr[..., :-2])
            F[:, :, k, :dk] = t / k
        if not tri:
            Fn = F[:, :, n]
        if rows is not None:
            rows.append(Fn.abs()); return
        acc += (weight.to(torch.complex128).view(c, 1, 1) * Fn).sum(0)
        if return_abs:
            acc_abs += (weight.abs().view(c, 1, 1) * Fn.abs()).sum(0)
        return
    # fused layout (GPU): the n path vectors stacked, the g terms as three contractions, and each series
    # step as ONE contraction - fewer, larger launches (the loops above issue ~10 kernels per k)
    W = [Ye[idx]]                                                       # C_S^(k-1) Y_S, k = 1..n  [c, 2p, 2V]
    for _ in range(2, n + 1):
        W.append(CS @ W[-1])
    W = torch.stack(W, 1)                                               # [c, n, 2p, 2V]
    g[:, :, 1:, 0] = (tr[:, 1:] / (2 * ks)).unsqueeze(1) + 0.5 * torch.einsum("cpv,ckpv->cvk", x0, W[..., :V])
    g[:, :, 1:, 1] = torch.einsum("cpv,ckpv->cvk", x0, W[..., V:])
    g[:, :, 1:, 2] = 0.5 * torch.einsum("cpv,ckpv->cvk", x1, W[..., V:])
    del W
    # F_k(z), z-degree <= 2k, stored with two leading zero columns so that z F and z^2 F are views:
    # Fp[..., 2 + j] = [z^j] F_k, and [z^j] (z^s F) = Fp[..., 2 + j - s]
    Fp = torch.zeros((c, V, n + 1, Lz + 2), dtype=torch.complex128, device=dev)
    Fp[:, :, 0, 2] = 1.0
    gi = g[:, :, 1:, :] * ks.view(1, 1, n, 1)                           # i g_i, i = 1..n
    for k in range(1, n + 1):
        dk = min(2 * k, Lz - 1) + 1                                     # F_k has z-degree <= 2k
        G = gi[:, :, :k].flip(2)                                        # i = k .. 1, paired with F_{k-i}
        Fs = torch.cat([Fp[:, :, :k, 2 - s:2 - s + dk] for s in range(3)], 2)   # [c, V, 3k, dk]: F, zF, z^2F
        Gs = torch.cat([G[..., 0], G[..., 1], G[..., 2]], 2)            # [c, V, 3k]
        Fp[:, :, k, 2:2 + dk] = torch.einsum("cvi,cviz->cvz", Gs, Fs) / k
    Fn = Fp[:, :, n, 2:]
    if rows is not None:
        rows.append(Fn.abs()); return
    acc += (weight.to(torch.complex128).view(c, 1, 1) * Fn).sum(0)
    if return_abs:
        acc_abs += (weight.abs().view(c, 1, 1) * Fn.abs()).sum(0)


_DD_MEM = 48                     # a dd term's working set against an fp64 one: hi / lo, the Ozaki slices and their per-order
                                 # concatenations of the stacked powers (~20 J (2q)^2 doubles per subset); 12 ran out of the
                                 # H100's 80 GB at N = 40 (job 27)
_DD_TRITON = None                # None: the dd series in its Triton kernel on CUDA (sieve_dd_triton); True / False: force


def _dd_series_triton(dev) -> bool:
    if _DD_TRITON is not None:
        return bool(_DD_TRITON)
    if dev.type != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _series_dd_torch(gih, gil):
    """(F_n hi, lo) [c, V, 2n+1] of the dd series recursion from i g_i = (gih, gil) [c, V, n, 3] complex128, in torch
    (`sieve_dd_triton.series_dd` is the kernel of the same recursion).
    """
    from . import dd
    c, V, n, _ = gih.shape
    Lz = 2 * n + 1
    Fh = torch.zeros((c, V, n + 1, Lz), dtype=torch.complex128, device=gih.device); Fl = torch.zeros_like(Fh)
    Fh[:, :, 0, 0] = 1.0
    for k in range(1, n + 1):
        dk = min(2 * k, Lz - 1) + 1
        Gh = gih[:, :, :k].flip(2); Gl = gil[:, :, :k].flip(2)           # i = k .. 1, paired with F_(k-i)
        Frh, Frl = Fh[:, :, :k, :dk], Fl[:, :, :k, :dk]
        sh, sl = dd.dsum(*dd.cmul(Gh[..., 0:1], Gl[..., 0:1], Frh, Frl), 2)
        for sft in (1, 2):
            if dk > sft:
                ah, al = dd.dsum(*dd.cmul(Gh[..., sft:sft + 1], Gl[..., sft:sft + 1], Frh[..., :-sft], Frl[..., :-sft]), 2)
                a2 = dd.add(sh[..., sft:], sl[..., sft:], ah, al)
                sh = torch.cat([sh[..., :sft], a2[0]], -1); sl = torch.cat([sl[..., :sft], a2[1]], -1)
        Fh[:, :, k, :dk], Fl[:, :, k, :dk] = dd.div_scalar(sh, sl, float(k))
    return Fh[:, :, n], Fl[:, :, n]


# largest dd batch (subsets) that fitted, per (n, subset width, V, device), learned from out-of-memory splits (`_terms_dd`)
_DD_CAP: dict = {}


def _terms_dd(Ce, Ye, XYe, idx, weight, n: int, V: int, acc, acc_abs, return_abs: bool):
    """`_terms_dd_batch` on the batch, split in halves while it runs out of device memory (the dd working set past
    n = 32 pairs is not covered by the `_DD_MEM` sizing: job 31, N ~ 70, a 15.8 GB concatenation of slices). The
    halving is remembered in `_DD_CAP` for later batches of the same shape. acc / acc_abs are only updated at the end
    of a batch, so a failed batch leaves them untouched; the split changes the summation order only (dd rounding).
    """
    key = (n, idx.shape[1], V, str(Ce.device))
    cap = _DD_CAP.get(key)
    c = idx.shape[0]
    if cap is not None and c > cap:
        for b0 in range(0, c, cap):
            _terms_dd(Ce, Ye, XYe, idx[b0:b0 + cap], weight[b0:b0 + cap], n, V, acc, acc_abs, return_abs)
        return
    try:
        _terms_dd_batch(Ce, Ye, XYe, idx, weight, n, V, acc, acc_abs, return_abs)
        return
    except torch.OutOfMemoryError:
        if c == 1:
            raise
    # retry OUTSIDE the except block: inside it the exception's traceback keeps the failed batch's tensors alive, and
    # the halved retries ran out of memory on 2 MiB with 78.5 GiB still allocated (job 31, run 20260929_210408)
    if Ce.device.type == "cuda":
        torch.cuda.empty_cache()
    _DD_CAP[key] = min(_DD_CAP.get(key, c), max(1, c // 2))
    _terms_dd(Ce, Ye, XYe, idx, weight, n, V, acc, acc_abs, return_abs)


def _terms_dd_batch(Ce, Ye, XYe, idx, weight, n: int, V: int, acc, acc_abs, return_abs: bool):
    """`_terms` in double-double: the same terms (powers P^1..P^h, traces <P^a, (P^b)^T>, path vectors C_S^(k-1) Y_S, g_k,
    the series F_k) with every product and sum in dd (`gaussian.dd`), weights exact, the batch sum added to acc = [hi, lo]
    in dd. Inputs Ce, Ye, XYe complex128 (exact as dd); acc_abs (the round-off estimate) stays float64.

    Layout (v2.11b, 2026-09-28): every product is one grouped Ozaki product of the complex block forms (5 GEMMs), and the
    ~3n/2 sequential products of v2.11 are ~log2(n) + 2 stacked ones: the powers by doubling (P^(m+j) = P^j P^m for all
    j <= m at once), the path vectors from them (w_(j+1) = P^j Y, then w_(h+1+j) = P^h w_(j+1)); the n traces and the 3n
    path dot-products are each ONE stacked dd product-and-sum; the series is the Triton kernel of `sieve_dd_triton` on
    CUDA (the torch recursion below elsewhere). v2.11 issued ~25 000 kernels per batch, x710-1990 of fp64 (job 26).
    """
    from . import dd
    dev = Ce.device
    c = idx.shape[0]; h = (n + 1) // 2

    def Z(t):
        return torch.zeros_like(t)
    rf = torch.profiler.record_function                                 # stage labels for bench/gbs_dd_profile.py (job 28)
    CS = Ce[idx.unsqueeze(2), idx.unsqueeze(1)]                         # [c, 2p, 2p]  exact
    xy = XYe[idx]; x0, x1 = xy[:, :, :V], xy[:, :, V:]                  # exact
    # powers P^1..P^h by doubling: P^(m + j) = P^j P^m for j = 1..min(m, h - m), one stacked product per level
    P = [None, (CS, Z(CS))]
    with rf("dd.powers"):
        while len(P) - 1 < h:
            m = len(P) - 1; J = min(m, h - m)
            Lh = torch.stack([P[j][0] for j in range(1, J + 1)]); Ll = torch.stack([P[j][1] for j in range(1, J + 1)])
            Rh, Rl = dd.cmatmul(Lh, Ll, *P[m])                          # [J, c, 2p, 2p]
            P += [(Rh[j], Rl[j]) for j in range(J)]
    # traces tr_k = <P^a, (P^b)^T>, a = ceil(k/2), b = floor(k/2): k = 1 is the diagonal of P^1, k >= 2 one stacked product
    trh = torch.zeros((c, n + 1), dtype=torch.complex128, device=dev); trl = Z(trh)
    d1h, d1l = dd.dsum(torch.diagonal(CS, dim1=1, dim2=2), torch.zeros_like(torch.diagonal(CS, dim1=1, dim2=2)), 1)
    trh[:, 1], trl[:, 1] = d1h, d1l
    if n >= 2:
        with rf("dd.traces"):
            ks = list(range(2, n + 1))
            Ah = torch.stack([P[(k + 1) // 2][0] for k in ks]); Al = torch.stack([P[(k + 1) // 2][1] for k in ks])
            Bh = torch.stack([P[k // 2][0].transpose(1, 2) for k in ks])
            Bl = torch.stack([P[k // 2][1].transpose(1, 2) for k in ks])
            th, tl = dd.dsum(*(t.flatten(2) for t in dd.cmul(Ah, Al, Bh, Bl)), 2)      # [n - 1, c]
            trh[:, 2:], trl[:, 2:] = th.T, tl.T
    # path vectors w_k = C_S^(k-1) Y_S, k = 1..n, from the powers: w_(j+1) = P^j Y (j = 1..h, one stacked product), then
    # w_(h+1+j) = P^h w_(j+1) (j = 1..n-h-1, a second one); then the three dot products for all k at once
    Y = Ye[idx]
    W = [(Y, Z(Y))]
    with rf("dd.paths"):
        if n >= 2:
            jj = range(1, min(h, n - 1) + 1)
            Lh = torch.stack([P[j][0] for j in jj]); Ll = torch.stack([P[j][1] for j in jj])
            Rh, Rl = dd.cmatmul(Lh, Ll, Y, Z(Y))                         # [|jj|, c, 2p, 2V]
            W += [(Rh[t], Rl[t]) for t in range(len(jj))]
        if n - 1 > h:
            jj = range(1, n - h)
            Bh = torch.stack([W[j][0] for j in jj]); Bl = torch.stack([W[j][1] for j in jj])
            Rh, Rl = dd.cmatmul(*P[h], Bh, Bl)
            W += [(Rh[t], Rl[t]) for t in range(len(jj))]
    with rf("dd.dots"):
        Wh = torch.stack([w[0] for w in W], 1); Wl = torch.stack([w[1] for w in W], 1)   # [c, n, 2p, 2V]
        x0e, x1e = x0.unsqueeze(1), x1.unsqueeze(1)
        d0 = dd.dsum(*dd.cmul(x0e, Z(x0e), Wh[..., :V], Wl[..., :V]), 2)                # [c, n, V]
        dy = dd.dsum(*dd.cmul(x0e, Z(x0e), Wh[..., V:], Wl[..., V:]), 2)
        d2 = dd.dsum(*dd.cmul(x1e, Z(x1e), Wh[..., V:], Wl[..., V:]), 2)
    kk = torch.arange(1, n + 1, device=dev, dtype=torch.float64)
    tqh, tql = dd.div_scalar(trh[:, 1:], trl[:, 1:], (2.0 * kk).view(1, n))             # [c, n]  tr_k / (2k)
    g0 = dd.add(tqh.unsqueeze(2).expand(-1, -1, V), tql.unsqueeze(2).expand(-1, -1, V), *dd.scale(*d0, 0.5))
    g2 = dd.scale(*d2, 0.5)
    gh = torch.stack([g0[0], dy[0], g2[0]], -1).transpose(1, 2)                          # [c, V, n, 3]
    gl = torch.stack([g0[1], dy[1], g2[1]], -1).transpose(1, 2)
    gih, gil = dd.scale(gh, gl, kk.view(1, 1, n, 1))                                    # i g_i, exact integer scale
    with rf("dd.series"):
        if _dd_series_triton(dev):
            from .sieve_dd_triton import series_dd
            Fnh, Fnl = series_dd(gih.contiguous(), gil.contiguous())
        else:
            Fnh, Fnl = _series_dd_torch(gih, gil)
    wv = weight.to(torch.float64).view(c, 1, 1)
    sh, sl = dd.dsum(*dd.scale(Fnh, Fnl, wv), 0)
    acc[0], acc[1] = dd.add(acc[0], acc[1], sh, sl)
    if return_abs:
        acc_abs += (weight.abs().view(c, 1, 1) * (Fnh + Fnl).abs()).sum(0)


def _pairing(labels, N: int):
    """Vertex order for the pair sieve that pairs identical vertices with each other.

    labels [N]: equal labels mark identical vertices (same row, column and loop entries - repeated
    output modes). Returns (perm [2n] into the padded vertex set, groups): pair i is vertices perm[i]
    and perm[i + n]; the first pairs are self-pairs of one label, `groups` = [(first pair, count)] per
    label with count >= 2 identical self-pairs, the others are singletons. Vertex N is the padding
    vertex when N is odd.
    """
    labels = np.asarray(labels)
    firsts, seconds, groups, left = [], [], [], []
    for lab in dict.fromkeys(labels.tolist()):
        vs = np.nonzero(labels == lab)[0].tolist()
        r = len(vs) // 2
        if r:
            if r >= 2:
                groups.append((len(firsts), r))
            firsts += vs[0:2 * r:2]; seconds += vs[1:2 * r:2]
        if len(vs) % 2:
            left.append(vs[-1])
    if N % 2:
        left.append(N)
    firsts += left[0::2]; seconds += left[1::2]
    return np.array(firsts + seconds, dtype=np.int64), groups


def _pairing_cross(labels, N: int, rng=None):
    """Vertex order for the pair sieve that groups repeated pairs ACROSS modes (Bulmer et al.'s term count).

    All vertices of one label are identical, so k pairs (u_i, v_i) with every u_i of label a and every v_i of label
    b != a are k identical pairs: they enter the sieve through how many of them S contains, (k + 1) terms instead of
    2^k (the same grouping as `_pairing`'s self-pairs, which reach only (floor(t/2) + 1) for a mode of t photons). The
    two labels with the most vertices left are paired min(t_a, t_b) times, repeatedly; when one label is left alone
    its vertices are self-paired (a group from 2 self-pairs), and the leftover singles are matched in order (by `rng`
    permutation when given). Returns (perm, groups) as `_pairing`. Ties between labels are broken by first
    appearance, or at random with `rng` (a numpy Generator): the term count depends on the counts only.
    Measured (bench, CPU, N = 26-28 vertices cut from Borealis fig4 samples, against a long-double sieve): x2.7-55
    fewer terms than a matching without grouping and x2-10 fewer than `_pairing`, with a condition number between the
    two (x3-5e4 above the best random matching, x10-1e5 below the self-pairing).
    """
    labels = np.asarray(labels)
    rem = {lab: np.nonzero(labels == lab)[0].tolist() for lab in dict.fromkeys(labels.tolist())}
    tie = {lab: (rng.random() if rng is not None else i) for i, lab in enumerate(rem)}
    firsts, seconds, groups = [], [], []
    while True:
        live = sorted((lab for lab in rem if rem[lab]), key=lambda lab: (-len(rem[lab]), tie[lab]))
        if len(live) < 2 or len(rem[live[1]]) < 2:
            break
        a, b = live[0], live[1]
        k = min(len(rem[a]), len(rem[b]))
        groups.append((len(firsts), k))
        for _ in range(k):
            firsts.append(rem[a].pop()); seconds.append(rem[b].pop())
    for lab in [lab for lab in rem if len(rem[lab]) >= 4]:           # one label left with many vertices: self-pairs
        vs = rem[lab]; r = len(vs) // 2
        groups.append((len(firsts), r))
        firsts += vs[0:2 * r:2]; seconds += vs[1:2 * r:2]
        rem[lab] = vs[2 * r:]
    singles = [v for lab in rem for v in rem[lab]] + ([N] if N % 2 else [])
    if rng is not None:
        singles = [int(v) for v in rng.permutation(singles)]
    h = len(singles) // 2
    firsts += singles[:h]; seconds += singles[h:]
    return np.array(firsts + seconds, dtype=np.int64), groups


def _pairing_terms(groups, n: int) -> float:
    """Subset terms of a sieve of n pairs with these collision groups: prod (r + 1) x 2^(free pairs)."""
    return math.prod(r + 1 for _, r in groups) * 2.0 ** (n - sum(r for _, r in groups))


# Launch floor of one v2 sieve, in vertices: below it a call costs about the same whatever its size (H100: 0.82 ms from
# 14 to 20 vertices, x1.3-2.8 per pair above, run 20260924_234127). The grouped sieve batches its count vectors by
# prefix length (`_sieve_multi`), so each prefix length is one sieve of (count vectors) x 2^n_f terms.
_LAUNCH_FLOOR = {"cuda": 22, "cpu": 8}


def _prefix_counts(groups) -> list[int]:
    """Number of count vectors with |s| = L, L = 0 .. sum r: the coefficients of prod_g (1 + x + .. + x^r_g)."""
    poly = [1]
    for _, r in groups:
        new = [0] * (len(poly) + r)
        for i, c in enumerate(poly):
            for j in range(r + 1):
                new[i + j] += c
        poly = new
    return poly


def _pairing_cost(groups, n: int, floor: int) -> float:
    """Relative cost of a sieve of n pairs with these groups: per prefix length L, one batched sieve of
    (count vectors with |s| = L) x 2^n_f terms, max(1, terms / 2^(floor/2)) floor units (the ungrouped sieve: groups = []).
    """
    n_f = n - sum(r for _, r in groups)
    unit = 2.0 ** (floor / 2)
    return sum(max(1.0, c * 2.0 ** n_f / unit) for c in _prefix_counts(groups) if c)


def _pick_grouping(labels, N: int, dev_type: str):
    """(perm, groups) below the pairing search: the cheapest by `_pairing_cost` of the ungrouped sieve on
    `_pairing`'s order (v2.8 when it has no groups), the self-pairing and the cross pairing; ties keep the earlier.
    """
    n = (N + 1) // 2
    floor = _LAUNCH_FLOOR.get(dev_type, 0)
    ps = _pairing(labels, N)
    opts = [(ps[0], []), ps, _pairing_cross(labels, N)]
    return min(opts, key=lambda o: _pairing_cost(o[1], n, floor))


def _setup(A, D0, D1, perm, dev):
    """(Ce, Ye, XYe) of the sieve for the vertex order `perm` (pair i = vertices perm[i], perm[i + n]) of the
    padded A [2n, 2n], D0 / D1 [V, 2n]: C = A X in the permuted order, Y = [D0 | D1]^T, XY = Y[X], each extended
    with the dummy vertex 2n (zero row / column, zero loop values) used for padding.
    """
    pt = torch.as_tensor(perm, device=dev)
    A = A[pt][:, pt]; D0 = D0[:, pt]; D1 = D1[:, pt]
    n = A.shape[0] // 2; V = D0.shape[0]
    X = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
    Y = torch.cat([D0, D1], 0).T.contiguous()                          # [2n, 2V]: D0 columns then D1
    Ce = torch.zeros((2 * n + 1, 2 * n + 1), dtype=torch.complex128, device=dev); Ce[:2 * n, :2 * n] = A[:, X]
    Ye = torch.cat([Y, torch.zeros((1, 2 * V), dtype=Y.dtype, device=dev)], 0)
    XYe = torch.cat([Y[X], torch.zeros((1, 2 * V), dtype=Y.dtype, device=dev)], 0)
    return Ce, Ye, XYe


# The sieve's rounding error is ~0.01-0.6 eps kappa, kappa = sum_S |term_S| / max|c|, and kappa depends on the
# pairing by 1-3 orders of magnitude (bench/gbs_precision.py against a long-double reference, N = 12-28,
# 2026-09-24); the collision self-pairs were x3-500 (median x66) above the best of 8 random matchings on a mode holding 4
# photons. Since the value is the same for every pairing, candidates are ranked by sum_S |term_S| alone, estimated
# from _PAIR_SAMPLES random subsets (Spearman 0.96-0.99 with the exact kappa over 24 matchings, N = 16-24).
LOG: list | None = None                     # bench only: a list gets (N, V, pairs, searched, terms) per lhaf_poly_multi call
_PAIR_SEARCH = {"cuda": 18, "cpu": 14}     # pairs from which "auto" searches (GPU: ~2 % of a sieve at N = 36)
_PAIR_CANDIDATES = 8                        # the self-pairing, 3 cross pairings (`_pairing_cross`) and 4 random matchings
_PAIR_SAMPLES = 128
_PAIR_PREFER_SELF = 2.0                     # "self" candidates: kept within this factor of the least estimate (v2.8 rule)
# Cost-aware choice (2026-09-27): the candidate with the fewest terms whose predicted round-off eps kappa stays below
# _ROUND_TOL, kappa predicted as the kappa fit of the least-estimate candidate (log10 kappa = _KAPPA_FIT[0] + _KAPPA_FIT[1] N,
# H100, runs to N = 56, docs/GAUSSIAN.md §2.4) times the candidate's estimate / the least estimate; checked after the
# sieve on the actual kappa (sum_S |term_S| / max |c|) and redone with the least-estimate candidate when it misses.
# The round-off is eps kappa, not N eps kappa (2026-09-29): the actual error of 275 fp64 sieves (N = 24-48, job 27, against
# the dd result) was at most 0.74 eps kappa, median 0.031; the N factor (used until then) made the rule x24-48 stricter.
# _ROUND_TOL = None: the v2.8 rule (self-pairing within _PAIR_PREFER_SELF of the least estimate, else the least).
# 1e-3 is the criterion that sets the fp64 tier in the device comparison (fp64 to N ~ 62, dd to N ~ 142 by the kappa fit,
# docs/GAUSSIAN.md §2.4); v2.8 itself runs with no control (up to 6e-5 measured, job 27). A grouped pairing is never taken past it.
_ROUND_TOL = 1e-3
_KAPPA_FIT = (0.40, 0.197)
EPS_DD = 2.0 ** -104
# precision of `lhaf_poly_multi` when its `precision` is None: "fp64", "dd" (double-double, `_terms_dd`) or "auto" (fp64,
# escalated to dd when the round-off misses _ROUND_TOL, or taken in dd directly when the cheapest dd-admissible pairing is
# _DD_FACTOR times cheaper in terms than the cheapest fp64-admissible one). _DD_FACTOR: a dd term (`_terms_dd`) against an
# fp64 Triton one, measured on the H100: v2.11 (job 26, run 20260928_092453) x320 at n = 12 pairs, x1100 at 24, x1000 at 32
# (V = 1; launch-bound: ~25 000 small kernels per batch); v2.11b (job 27, run 20260928_204415) x116-253 (V = 1-4, n = 12-32),
# hence 250, the upper end (it only steers "auto"; the timing model's tiers use the measured rates themselves). The first
# value, 100, from the product costs alone (job 24), sent heavy N = 48 sieves to dd at x3.7-5.8 of their fp64 time.
_PRECISION = "fp64"
_DD_FACTOR = 250.0


def _choose_pairing(A, D0, D1, labels, N: int, dev):
    """(candidates [(perm, groups)], estimated sum_S |term_S| per candidate, index of the least) (see above).

    Candidates: the self-pairing of `_pairing(labels)`, three cross pairings of `_pairing_cross` (collision groups
    across modes; one with ties in order, two at random), and random perfect matchings of the 2n padded vertices
    without grouping; the random generator is seeded by N so the choice is reproducible (`_select_pairing` picks
    among them for a given arithmetic). Every candidate is sampled on
    the same number of (count vector, free subset) terms, all in ONE `_terms` call on the block-diagonal union of the
    candidates' matrices. A [2n, 2n], D0 / D1 [V, 2n], padded.
    """
    rng = np.random.default_rng(1234567 + N)
    n = A.shape[0] // 2; V = D0.shape[0]; E = 2 * n + 1
    cross = [_pairing_cross(labels, N)] + [_pairing_cross(labels, N, rng) for _ in range(2)]
    cands = [_pairing(labels, N), *cross]
    cands += [(rng.permutation(2 * n), []) for _ in range(_PAIR_CANDIDATES - len(cands))]
    Ces, Yes, XYes, rows, scale = [], [], [], [], []
    for c, (perm, groups) in enumerate(cands):
        Ce, Ye, XYe = _setup(A, D0, D1, perm, dev)
        Ces.append(Ce); Yes.append(Ye); XYes.append(XYe)
        grouped = {i for f, r in groups for i in range(f, f + r)}
        free = [i for i in range(n) if i not in grouped]
        total = math.prod(r + 1 for _, r in groups) * 2.0 ** len(free)
        for _ in range(_PAIR_SAMPLES):
            sg = [int(rng.integers(0, r + 1)) for _, r in groups]
            pairs = [f + j for (f, _), k in zip(groups, sg) for j in range(k)]
            pairs += [i for i in free if rng.random() < 0.5]
            w = math.prod(math.comb(r, k) for (_, r), k in zip(groups, sg))
            vs = pairs + [2 * n] * (n - len(pairs)) + [i + n for i in pairs] + [2 * n] * (n - len(pairs))
            rows.append([c * E + v for v in vs]); scale.append(w * total / _PAIR_SAMPLES)
    K = len(cands)
    Cb = torch.zeros((K * E, K * E), dtype=torch.complex128, device=dev)
    for c in range(K):
        Cb[c * E:(c + 1) * E, c * E:(c + 1) * E] = Ces[c]
    idx = torch.as_tensor(np.array(rows, dtype=np.int64), device=dev)
    out = []
    _terms(Cb, torch.cat(Yes), torch.cat(XYes), idx, None, n, V, None, None, False, rows=out)
    est = (out[0] * torch.as_tensor(scale, dtype=torch.float64, device=dev).view(-1, 1, 1))   # [K M, V, Lz]
    est = est.view(K, _PAIR_SAMPLES, V, -1).sum(1).amax(2).sum(1).cpu().numpy()             # per candidate
    return cands, est, int(np.argmin(est))


def _select_pairing(cands, est, best: int, N: int, n: int, eps: float, dev) -> tuple[int, bool]:
    """(index, predicted within tolerance) of the candidate `lhaf_poly_multi` runs in an arithmetic of unit round-off
    eps: the cheapest by `_pairing_cost` whose predicted eps kappa <= `_ROUND_TOL` (kappa = the kappa fit x est /
    est[best]), else the least estimate; `_ROUND_TOL = None`: the v2.8 rule (self within _PAIR_PREFER_SELF of the least).
    """
    if _ROUND_TOL is None:
        return (0 if est[0] <= _PAIR_PREFER_SELF * est[best] else best), True
    kappa0 = 10.0 ** (_KAPPA_FIT[0] + _KAPPA_FIT[1] * N)
    ok = [c for c in range(len(cands)) if eps * kappa0 * est[c] / est[best] <= _ROUND_TOL]
    floor = _LAUNCH_FLOOR.get(torch.device(dev).type, 0)
    if not ok:
        return best, False
    return min(ok, key=lambda c: (_pairing_cost(cands[c][1], n, floor), est[c])), True


def lhaf_poly_multi(A, D0, D1, device=None, budget_bytes: int | None = None, return_abs: bool = False,
                    labels=None, pairing: str = "auto", precision: str | None = None):
    """`lhaf_poly` for V loop-vector pairs sharing one matrix: coefficients [V, N + 1] of
    z -> lhaf(A, D0[v] + z D1[v]) for v = 0..V-1 (D0, D1 [V, N]).

    v2 of the sieve (docs/GAUSSIAN.md §3.4), same formula as `lhaf_poly`, four changes:
      * subsets grouped by size (a few buckets, `_free_subsets`) and their matrices GATHERED as [2pb, 2pb]
        instead of masked [2n, 2n]: the matrix work falls towards E[p^3]/n^3 ~ 1/8 for p ~ Binomial(n, 1/2);
      * the matrix part (C_S and its traces) is built once for all V vectors; only the path terms
        (XD)^T C^(k-1) D and the series are per vector (the chain rule batches the steps of a run of
        empty modes, which share A);
      * traces from half the products, tr C^k = <C^a, (C^b)^T>, a = ceil(k/2), b = floor(k/2), and the
        series F_k = (1/k) sum_i i g_i F_(k-i) as one contraction per k, z-degree capped at 2k;
      * collisions (`labels`: equal labels = identical vertices, i.e. a repeated output mode): identical
        vertices are paired with each other, and r identical pairs enter the sieve only through how many
        of them S contains, s = 0..r with weight C(r, s) — (r + 1) terms instead of 2^r.
    `return_abs` as in `lhaf_poly` (then a second [V, N + 1] float64 tensor: sum of |weight x term|).
    `pairing`: "self" = the collision self-pairs of `_pairing`; "cross" = the cross-mode groups of `_pairing_cross`;
    "auto" (default) = below `_PAIR_SEARCH` pairs the cheapest of ungrouped / self / cross by `_pairing_cost`
    (`_pick_grouping`: the launch floor makes grouping a loss for small free sieves), and from there the cheapest
    candidate of `_choose_pairing` predicted to meet `_ROUND_TOL` (any perfect matching gives the same value, not the
    same rounding). In "auto", a grouped pairing whose actual round-off eps kappa (kappa =
    sum_S |term_S| / max |c|, per loop vector) exceeds `_ROUND_TOL` is redone with the best-conditioned candidate
    (below `_PAIR_SEARCH`: with a random matching, seeded by N), so the discount never costs precision past it.
    `precision`: "fp64", "dd" (every term and the sum over subsets in double-double, `_terms_dd`: the torch path,
    x116-253 of an fp64 term on the H100; the pairing choice and the round-off check then use eps_dd = 2^-104) or "auto" (fp64,
    redone in dd when the fp64 round-off still misses `_ROUND_TOL` after the pairing fallback, or run in dd from the start
    when that is `_DD_FACTOR` cheaper by `_pairing_cost`); None = the module default `_PRECISION`. The result is
    complex128 either way (the dd sum rounded to it).
    """
    if device is None:
        device = A.device if isinstance(A, torch.Tensor) else "cpu"
    dev = torch.device(device)
    if budget_bytes is None:
        budget_bytes = (1 << 32) if dev.type == "cuda" else (1 << 28)
    A = torch.as_tensor(A, dtype=torch.complex128).to(dev)
    D0 = torch.as_tensor(D0, dtype=torch.complex128).to(dev)
    D1 = torch.as_tensor(D1, dtype=torch.complex128).to(dev)
    if D0.dim() != 2 or D0.shape != D1.shape or D0.shape[1] != A.shape[0]:
        raise ValueError("D0 and D1 must be [V, N] with N = A.shape[0]")
    V, N = D0.shape
    out = torch.zeros((V, N + 1), dtype=torch.complex128, device=dev)
    if N == 0:
        out[:, 0] = 1.0
        return (out, out.real.abs().clone()) if return_abs else out
    labels = np.arange(N) if labels is None else np.asarray(labels)
    if labels.shape != (N,):
        raise ValueError("labels must have one entry per vertex")
    if N % 2:                                                          # pad: isolated vertex, loop 1
        Ap = torch.zeros((N + 1, N + 1), dtype=A.dtype, device=dev); Ap[:N, :N] = A; A = Ap
        D0 = torch.cat([D0, torch.ones((V, 1), dtype=D0.dtype, device=dev)], 1)
        D1 = torch.cat([D1, torch.zeros((V, 1), dtype=D1.dtype, device=dev)], 1)
    n = A.shape[0] // 2
    search = _PAIR_SEARCH.get(dev.type, 10 ** 9) if pairing == "auto" else 10 ** 9
    if pairing not in ("auto", "self", "cross"):
        raise ValueError("pairing must be 'auto', 'self' or 'cross'")
    prec = _PRECISION if precision is None else precision
    if prec not in ("fp64", "dd", "auto"):
        raise ValueError("precision must be 'fp64', 'dd' or 'auto'")
    floor = _LAUNCH_FLOOR.get(dev.type, 0)
    fallback = None
    use_dd = prec == "dd"
    if n >= search:
        cands, est, best = _choose_pairing(A, D0, D1, labels, N, dev)
        fallback = cands[best]
        pick, ok = _select_pairing(cands, est, best, N, n, EPS_DD if use_dd else EPS, dev)
        if prec == "auto":
            pdd, okdd = _select_pairing(cands, est, best, N, n, EPS_DD, dev)
            if okdd and (not ok or _pairing_cost(cands[pick][1], n, floor)
                         > _DD_FACTOR * _pairing_cost(cands[pdd][1], n, floor)):
                pick, use_dd = pdd, True
        perm, groups = cands[pick]
    elif pairing == "self":
        perm, groups = _pairing(labels, N)
    elif pairing == "cross":
        perm, groups = _pairing_cross(labels, N)
    else:
        perm, groups = _pick_grouping(labels, N, dev.type)
        fallback = (np.random.default_rng(7654321 + N).permutation(2 * n), [])
    check = pairing == "auto" and _ROUND_TOL is not None and fallback is not None

    def run(perm, groups, dd):
        return _sieve_multi(A, D0, D1, perm, groups, n, V, dev, budget_bytes, return_abs or check, dd=dd)

    def misses(acc, acc_abs, dd):
        kappa = (acc_abs.amax(1) / acc.abs().amax(1).clamp_min(1e-300)).max().item()
        return (EPS_DD if dd else EPS) * kappa > _ROUND_TOL
    acc, acc_abs = run(perm, groups, use_dd)
    redo = False
    if check and misses(acc, acc_abs, use_dd):
        if groups and fallback[0] is not perm:                         # the grouping cost the precision: ungroup
            redo = True
            perm, groups = fallback
            acc, acc_abs = run(perm, groups, use_dd)
        if prec == "auto" and not use_dd and misses(acc, acc_abs, False):   # fp64 itself is not enough: dd
            redo, use_dd = True, True
            acc, acc_abs = run(perm, groups, True)
    if LOG is not None:                                                # bench only: what the sieve enumerated
        LOG.append((N, V, n, n >= search, _pairing_terms(groups, n), redo, "dd" if use_dd else "fp64"))
    out[:] = acc[:, : N + 1]
    return (out, acc_abs[:, : N + 1]) if return_abs else out


def _sieve_multi(A, D0, D1, perm, groups, n: int, V: int, dev, budget_bytes: int, return_abs: bool, dd: bool = False):
    """(acc [V, 2n+1] complex128, acc_abs [V, 2n+1] float64) of the v2 sieve for one pairing (`lhaf_poly_multi`):
    the count vectors of the collision groups times the free subsets, in `_terms` batches. A, D0, D1 padded to 2n.
    `dd`: every term and the accumulation in double-double (`_terms_dd`, batches /_DD_MEM of the fp64 size); acc is then
    the dd sum rounded to complex128.
    """
    Ce, Ye, XYe = _setup(A, D0, D1, perm, dev)
    Lz = 2 * n + 1
    acc = torch.zeros((V, Lz), dtype=torch.complex128, device=dev)
    acc_abs = torch.zeros((V, Lz), dtype=torch.float64, device=dev)
    if dd:
        acc = [acc, torch.zeros_like(acc)]                             # (hi, lo)
        budget_bytes = max(1, budget_bytes // _DD_MEM)
    terms = _terms_dd if dd else _terms
    # pairs in groups of identical self-pairs (enumerated by count) and the free pairs (subsets)
    grouped = [i for f, r in groups for i in range(f, f + r)]
    free = np.array([i for i in range(n) if i not in set(grouped)], dtype=np.int64)
    n_f = free.size
    # map of the local vertex ids of _free_subsets(n_f) to global ones: local pair o -> global free[o]
    loc = np.concatenate([free, free + n, [2 * n]])
    loc_t = torch.as_tensor(loc, device=dev)
    from itertools import product as iproduct
    # count vectors grouped by prefix length |s| (their prefix rows have one width), each group batched with the free
    # subsets in the same `_terms` calls: one call per (prefix length, free batch) instead of one per count vector,
    # which left the grouped sieve launch-bound (x0.4-1.3 of v2.8 on CPU at N = 24-28 with x30 fewer terms, 2026-09-27)
    n_g = sum(r for _, r in groups)
    by_len: dict = {}
    for counts in iproduct(*[range(r + 1) for _, r in groups]):
        pre = [f + j for (f, _), sg in zip(groups, counts) for j in range(sg)]
        wgt = float(np.prod([math.comb(r, sg) for (_, r), sg in zip(groups, counts)]))
        base = wgt * (-1.0 if (n_g - len(pre)) % 2 else 1.0)          # sign of the grouped part, (-1)^(n_g - |s|)
        by_len.setdefault(len(pre), []).append((pre + [i + n for i in pre], base))
    for n_pre, items in by_len.items():
        P = torch.as_tensor(np.array([v for v, _ in items], dtype=np.int64).reshape(len(items), 2 * n_pre), device=dev)
        Bw = torch.as_tensor([b for _, b in items], dtype=torch.float64, device=dev)
        if n_pre:                                                      # free subset empty: sign (-1)^(n_f)
            terms(Ce, Ye, XYe, P, Bw * (-1.0 if n_f % 2 else 1.0), n, V, acc, acc_abs, return_abs)
        if not n_f:
            continue
        def chunk_of(pb, n_pre=n_pre):
            return _chunk_size_multi(n, n_pre + pb, V, budget_bytes, dev)   # per bucket: small buckets, big batches
        for pb, idx, sgn in _free_subsets(n_f, dev, chunk_of):        # non-empty free subsets
            idx = loc_t[idx]
            c = idx.shape[0]
            cap = max(1, chunk_of(pb) // c)                            # count vectors per call with this free batch
            for b0 in range(0, len(items), cap):
                Pb, Bb = P[b0:b0 + cap], Bw[b0:b0 + cap]
                nb = Pb.shape[0]
                rows = idx.repeat(nb, 1)
                if n_pre:
                    rows = torch.cat([Pb.repeat_interleave(c, 0), rows], 1)
                terms(Ce, Ye, XYe, rows, Bb.repeat_interleave(c) * sgn.repeat(nb), n, V, acc, acc_abs, return_abs)
    return (acc[0] + acc[1] if dd else acc), acc_abs


_SMALL_MAX = 20                  # vertices: `lhaf_poly_small` handles N <= this (2^10 subsets, all at once)


def lhaf_poly_small(A, D0, D1, return_abs: bool = False):
    """`lhaf_poly_multi` for small sieves in numpy: coefficients [V, N + 1] of z -> lhaf(A, D0[v] + z D1[v]),
    complex128 numpy (and sum_S |term_S| [V, N + 1] float64 with `return_abs`). D0, D1 [V, N]; N <= _SMALL_MAX.

    The v1 pairing (i, i + n) and masked [2^n, 2n, 2n] subset matrices, every subset in one batch, and the
    series as one contraction per k: ~6n numpy calls in all. For the chain rule's many small steps, which
    ran on the CPU through torch at 2-3.5 ms per call (n = 40: 849 calls under 16 vertices per 16 shots,
    35 % of the time, run 20260924_141521), where the time was per-operation dispatch, not arithmetic.
    Collisions are not grouped (no gain at this size) and the pairing is not searched (kappa <= ~1e5 here).
    """
    A = np.asarray(A, dtype=np.complex128); D0 = np.atleast_2d(np.asarray(D0, dtype=np.complex128))
    D1 = np.atleast_2d(np.asarray(D1, dtype=np.complex128))
    V, N = D0.shape
    if N > _SMALL_MAX:
        raise ValueError(f"lhaf_poly_small is for N <= {_SMALL_MAX}")
    if N == 0:
        out = np.zeros((V, 1), dtype=np.complex128); out[:, 0] = 1.0
        return (out, np.abs(out)) if return_abs else out
    if N % 2:                                                          # pad: isolated vertex, loop 1
        Ap = np.zeros((N + 1, N + 1), dtype=np.complex128); Ap[:N, :N] = A; A = Ap
        D0 = np.concatenate([D0, np.ones((V, 1))], 1); D1 = np.concatenate([D1, np.zeros((V, 1))], 1)
    n = A.shape[0] // 2; L = 2 * n + 1
    X = np.concatenate([np.arange(n, 2 * n), np.arange(0, n)])
    C = A[:, X]
    S = np.arange(1 << n)
    s = ((S[:, None] >> np.arange(n)) & 1).astype(np.float64)
    s = np.concatenate([s, s], 1)                                       # [S, 2n]
    CS = s[:, :, None] * C[None] * s[:, None, :]                        # [S, 2n, 2n]
    W = s[:, None, :] * np.concatenate([D0, D1], 0)[None]               # [S, 2V, 2n]: C_S^(k-1) D, k = 1
    x0 = s[:, None, :] * D0[:, X][None]; x1 = s[:, None, :] * D1[:, X][None]   # [S, V, 2n]
    g = np.zeros((len(S), V, n + 1, 3), dtype=np.complex128)
    P = CS
    for k in range(1, n + 1):
        if k > 1:
            P = P @ CS
            W = W @ CS.transpose(0, 2, 1)                               # (C_S w)^T for every row of W
        tr = np.trace(P, axis1=1, axis2=2)
        w0, w1 = W[:, :V], W[:, V:]
        g[:, :, k, 0] = (tr / (2 * k))[:, None] + 0.5 * (x0 * w0).sum(2)
        g[:, :, k, 1] = (x0 * w1).sum(2)
        g[:, :, k, 2] = 0.5 * (x1 * w1).sum(2)
    gi = g[:, :, 1:, :] * np.arange(1, n + 1)[None, None, :, None]      # i g_i
    Fp = np.zeros((len(S), V, n + 1, L + 2), dtype=np.complex128)      # two zero columns: z F, z^2 F as views
    Fp[:, :, 0, 2] = 1.0
    for k in range(1, n + 1):
        dk = min(2 * k, L - 1) + 1
        G = gi[:, :, :k][:, :, ::-1]                                    # i = k .. 1, paired with F_{k-i}
        t = np.einsum("svi,sviz->svz", G[..., 0], Fp[:, :, :k, 2:2 + dk])
        t += np.einsum("svi,sviz->svz", G[..., 1], Fp[:, :, :k, 1:1 + dk])
        t += np.einsum("svi,sviz->svz", G[..., 2], Fp[:, :, :k, 0:dk])
        Fp[:, :, k, 2:2 + dk] = t / k
    Fn = Fp[:, :, n, 2:]                                                # [S, V, L]
    sign = np.where((n - s[:, :n].sum(1).astype(np.int64)) % 2 == 1, -1.0, 1.0)
    out = (sign[:, None, None] * Fn).sum(0)[:, : N + 1]
    if return_abs:
        return out, np.abs(Fn).sum(0)[:, : N + 1]
    return out


def lhaf(A, D, device=None) -> complex:
    """Loop hafnian of A (diagonal ignored) with loop weights D, by the pair sieve."""
    return complex(lhaf_poly(A, D, None, device=device)[0].item())


def hafnian(A, device=None) -> complex:
    """Hafnian (no loops) of a symmetric matrix of even size; 0 for odd size."""
    A = torch.as_tensor(A, dtype=torch.complex128)
    if A.shape[0] % 2:
        return 0j
    return lhaf(A, torch.zeros(A.shape[0], dtype=torch.complex128), device=device)


def double_factorial(n: int) -> int:
    """(n-1)!! pairings of n points (n even), used by tests to size brute-force checks."""
    return math.prod(range(n - 1, 0, -2)) if n > 0 else 1
