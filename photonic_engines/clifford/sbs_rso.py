"""Random-source-order chain rule for superposed sources (SBS): Clifford & Clifford with an offset start set.

2026-10-02. The replica / residual chain of
`product_sampler` evaluates the true marginal at every step; this one does what Clifford & Clifford do for a
Fock input -- draw a random order first and condition on it -- which an earlier analysis (2026-09-19) had ruled out
on the strength of a pointwise lemma that is false for Fock inputs too.

The algorithm (k photons from n live sources, A = U[:, source modes], m x n, orthonormal columns):

    sigma  <- uniform random order of the n sources
    for j = 1..k:
        T_j  = sigma[1 .. n-k+j]                          (|T_j| = n-k+j; T_0 has n-k sources)
        P_x  = Per_rect(A[r_1..r_{j-1}, T_j \\ x])         for every x in T_j   ((j-1) x (n-k+j-1))
        w(r) = | sum_{x in T_j} A[r, x] P_x |^2           = |Per_rect(A[r_<j + (r), T_j])|^2 (Laplace)
        r_j  <- w / sum(w)

Exact because (i) the true marginal of r_<=j is the uniform average over (n-k+j)-sets T of
|Per_rect(A[r_<=j, T])|^2, (ii) sum_r |Per_rect(A[r_<j + (r), T])|^2 = sum_{x in T} |Per_rect(A[r_<j, T\\x])|^2
by column orthonormality, so (iii) the posterior over the active SET given r_<=j stays proportional to
|Per_rect|^2 when T grows by one uniform element. At k = n it is Clifford & Clifford 2018 verbatim.

The rectangular cofactors, all x at once, by Glynn over the j-1 placed rows:

    Per_rect(B[:, T\\x]) = 2^-(j-1) sum_delta (prod delta) e_{j-1}( a(delta) without a_x ),
    a_i(delta) = sum_t delta_t B[t, i]   (j-1 placed rows, so e is of degree j-1)

e_{j-1} is homogeneous, so the global flip delta -> -delta is a symmetry and half the signs suffice. The
leave-one-out elementary symmetric polynomial e_{j-1}(a without a_x) is the degree-(j-1) coefficient of
prefix_x(lam) * suffix_x(lam), prefix/suffix being the truncated products of (1 + lam a_i) over i < x and
i > x (`_loo_prefix`, the reference), or -- the default, `_loo_divide` -- the full product divided by (1 + lam a_x),
forward where |a_x| <= 1 and backward from the top degree where |a_x| > 1 (forward alone is NOT stable: 1e-10 at
j = 20, 3e-5 at j = 40, 30 at j = 50 relative to the largest value, n = 2j; bench/methods_cpu_checks.py --only division).

Cost per step, per stream: 2^(j-2) signs x ~(n-k+j)^2 / 2 + (n-k+j) j  +  m (n-k+j) for the Laplace sweep. Per shot
~ 2^k n k: the exponent is the Fock one in the DETECTED photons, and the photon law is that of the
incoherent twin (Fock at eta = sin^2 alpha), so E[cost] ~ (1 - p + 2p)^n n k with p = eta sin^2 alpha.
The output-side collision collapse of Clifford & Clifford 2020 (prod(mu + 1) terms instead of 2^(j-1), halved
when some mu is odd) applies unchanged to this sum: from q = `collapse_from` placed rows on, each stream's cofactors
are the collapsed sum (backend "collapsed": torch, leave-one-out by `_loo_divide`; "triton": `sbs_rso_triton`, the
leave-one-out e_q read off a discrete Fourier transform on a circle so that every (term, point) pair is a Fock-type
all-but-one product, cost per term ~ N (D + L), L = N points; 2026-10-03). The CUDA backend (`cuda_rect`,
`backend="auto"` on a CUDA device with cupy) applies the collapse through the Fock kernel's odometer.

The pseudo-code above evolves with this file.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch

from ..states import ProductSuperposition


def _budget_bytes(dev: torch.device) -> int:
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        return max(1 << 28, int(0.5 * free))
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return max(1 << 27, int(0.3 * int(line.split()[1]) * 1024))
    except OSError:
        pass
    return 1 << 30


def _signs(nbits: int, start: int, stop: int, dtype, dev) -> torch.Tensor:
    """Sign vectors start..stop of {+-1}^(nbits+1) with the FIRST sign fixed to +1: [D, nbits + 1]."""
    idx = torch.arange(start, stop, device=dev, dtype=torch.int64).unsqueeze(1)
    bits = torch.arange(nbits, device=dev, dtype=torch.int64).unsqueeze(0)
    rest = (1 - 2 * ((idx >> bits) & 1)).to(dtype)
    return torch.cat([torch.ones((rest.shape[0], 1), dtype=dtype, device=dev), rest], dim=1)


def _loo_prefix(a: torch.Tensor, q: int) -> torch.Tensor:
    """e_q(a without a_x) for every x, by prefix x suffix truncated products. a: [..., N] -> [..., N].
    Stable, O(3 N q) and memory [..., N + 1, q + 1]: the reference for `_loo_divide`.
    """
    N = a.shape[-1]
    pre = torch.zeros((*a.shape[:-1], N + 1, q + 1), dtype=a.dtype, device=a.device)
    suf = torch.zeros_like(pre)
    pre[..., 0, 0] = 1
    suf[..., N, 0] = 1
    for i in range(N):
        pre[..., i + 1, :] = pre[..., i, :]
        pre[..., i + 1, 1:] += a[..., i, None] * pre[..., i, :-1]
        r = N - 1 - i
        suf[..., r, :] = suf[..., r + 1, :]
        suf[..., r, 1:] += a[..., r, None] * suf[..., r + 1, :-1]
    return (pre[..., :N, :] * suf[..., 1:, :].flip(-1)).sum(-1)


def _loo_minform(a: torch.Tensor, q: int) -> torch.Tensor:
    """e_q(a without a_x) for every x, division-free, at cost O(3 N d), d = min(q, N - 1 - q). a: [..., N] -> [..., N].

    q <= N - 1 - q: `_loo_prefix` (prefix x suffix products of prod (1 + lam a_i) truncated at degree q). Otherwise the
    reversed form: e_q(a without a_x) is the coefficient of mu^s, s = N - 1 - q, in prod_{i != x} (mu + a_i), and the
    prefix / suffix products of (mu + a_i) are truncated at degree s. At s = 0 (square: q = N - 1, an empty offset set)
    it is the all-but-one product of the Fock payload, so the rectangular cost interpolates between O(N) at either end
    and O(N^2) at q = (N - 1) / 2 -- the mu-form payload of cuda_rect. Reference / prototype.
    """
    N = a.shape[-1]
    s = N - 1 - q
    if q <= s:
        return _loo_prefix(a, q)
    pre = torch.zeros((*a.shape[:-1], N + 1, s + 1), dtype=a.dtype, device=a.device)
    suf = torch.zeros_like(pre)
    pre[..., 0, 0] = 1
    suf[..., N, 0] = 1
    for i in range(N):
        pre[..., i + 1, :] = a[..., i, None] * pre[..., i, :]
        pre[..., i + 1, 1:] += pre[..., i, :-1]
        r = N - 1 - i
        suf[..., r, :] = a[..., r, None] * suf[..., r + 1, :]
        suf[..., r, 1:] += suf[..., r + 1, :-1]
    return (pre[..., :N, :] * suf[..., 1:, :].flip(-1)).sum(-1)


def _loo_divide(a: torch.Tensor, q: int) -> torch.Tensor:
    """e_q(a without a_x) for every x: the FULL product F(lam) = prod_i (1 + lam a_i) once, then F / (1 + lam a_x)
    by forward substitution where |a_x| <= 1 (c_d = F_d - a_x c_(d-1), errors damped by |a_x| per step) and by
    backward substitution from the top degree where |a_x| > 1 (c_(d-1) = (F_d - c_d) / a_x, damped by 1/|a_x|).
    Forward alone loses everything at large q (relative 1e-10 / 3e-5 / 30 at q = 20 / 40 / 50, N = 2q; mixed 2e-13 /
    5e-10 / 3e-8: bench/methods_cpu_checks.py --only division). O(N^2 + N q) per vector, memory [..., N + 1].
    """
    N = a.shape[-1]
    F = torch.zeros((*a.shape[:-1], N + 1), dtype=a.dtype, device=a.device)
    F[..., 0] = 1
    for i in range(N):
        F[..., 1:i + 2] += a[..., i:i + 1] * F[..., 0:i + 1]
    big = a.abs() > 1
    fw = F[..., :1].expand_as(a).clone()
    for d in range(1, q + 1):
        fw = F[..., d:d + 1] - a * fw
    if not bool(big.any()):
        return fw
    ainv = torch.where(big, 1 / torch.where(big, a, torch.ones_like(a)), torch.zeros_like(a))
    bw = F[..., N:N + 1] * ainv                                                  # c_(N-1)
    for d in range(N - 1, q, -1):                                               # c_(d-1) from c_d
        bw = (F[..., d:d + 1] - bw) * ainv
    return torch.where(big, bw, fw)


def rect_cofactors(B: torch.Tensor, sign_chunk: int | None = None, loo: str = "divide") -> torch.Tensor:
    """P[s, x] = Per_rect(B[s][:, all columns but x]) for B [S, q, N] (q rows, N > q columns). -> [S, N].

    q = 0 gives ones. Glynn over the q rows with the first sign fixed (2^(q-1) vectors), the leave-one-out
    elementary symmetric polynomials e_q by `_loo_divide` (default) or `_loo_prefix` (loo="prefix", the
    reference), chunked over the signs.
    """
    S, q, N = B.shape
    dtype, dev = B.dtype, B.device
    if q == 0:
        return torch.ones((S, N), dtype=dtype, device=dev)
    full = 1 << (q - 1)
    esz = torch.empty(0, dtype=dtype).element_size()
    per_sign = (3 * (N + 1) * (q + 1)) if loo == "prefix" else (N + 1 + 4 * N)
    room = max(1, _budget_bytes(dev) // (per_sign * esz * S))
    dc = min(full, room if sign_chunk is None else min(room, sign_chunk))
    dc = 1 << max(0, max(1, dc).bit_length() - 1)
    fn = _loo_prefix if loo == "prefix" else _loo_divide
    P = torch.zeros((S, N), dtype=dtype, device=dev)
    for d0 in range(0, full, dc):
        sd = _signs(q - 1, d0, min(d0 + dc, full), dtype, dev)                 # [D, q]
        a = torch.einsum("dt,stn->sdn", sd, B)                                 # [S, D, N]
        P += torch.einsum("sdn,d->sn", fn(a, q), sd.prod(dim=1))
    return P * (2.0 / 2 ** q)


def collapse_tables(mu: list[int]):
    """(radix, stride, wtab, n_terms, factor) of the collapsed Glynn sum for multiplicities mu (python ints).

    The symmetry v -> mu - v halves the sum when some mu_nu is odd: that group runs over v < (mu+1)/2 and the result is
    doubled. With every mu_nu even there is a fixed point and no halving.
    """
    D = len(mu)
    order = list(range(D))
    odd = [i for i in order if mu[i] % 2 == 1]
    radix = [m_ + 1 for m_ in mu]
    factor = 1.0
    if odd:
        h = odd[-1]
        radix[h] = (mu[h] + 1) // 2
        factor = 2.0
    stride, s = [], 1
    for r_ in radix:
        stride.append(s)
        s *= r_
    mu_pad = max(mu) + 1 if mu else 1
    wtab = [[(-1) ** v * math.comb(m_, v) if v <= m_ else 0.0 for v in range(mu_pad)] for m_ in mu]
    return radix, stride, wtab, s, factor


def _circle(N: int, q: int, sigma: float, dev):
    L = N
    r = q / ((N - q) * sigma) if (q < N and sigma > 0) else 1.0
    t = torch.arange(L, dtype=torch.float64, device=dev)
    lam = r * torch.exp(2j * math.pi * t / L)
    coef = lam ** (-q) / L
    return lam, coef


def rect_cofactors_collapsed_torch(b: torch.Tensor, mu: list[int], term_chunk: int | None = None,
                                   loo: str = "divide") -> torch.Tensor:
    """Collapsed Glynn cofactors in torch (CPU or CUDA). b: [D, N] complex distinct rows, mu: their multiplicities
    (sum = q). Returns P [N] complex128.

    loo="dft" is the twin of the Triton kernel (leave-one-out e_q on the circle, memory [term_chunk, N, N]);
    loo="divide" (default, the torch production path) uses `_loo_divide` per term, memory [term_chunk, 5 N].
    """
    D, N = b.shape
    q = int(sum(mu))
    dev = b.device
    if q == 0:
        return torch.ones(N, dtype=b.dtype, device=dev)
    radix, stride, wtab, n_terms, factor = collapse_tables(mu)
    sigma = math.sqrt(float((torch.tensor(mu, dtype=torch.float64, device=dev)[:, None] * b.abs() ** 2).sum(0).mean()))
    lam, coef = _circle(N, q, sigma, dev)
    rad = torch.tensor(radix, dtype=torch.int64, device=dev)
    strd = torch.tensor(stride, dtype=torch.int64, device=dev)
    mut = torch.tensor(mu, dtype=torch.int64, device=dev)
    wt = torch.tensor(wtab, dtype=torch.float64, device=dev)                        # [D, MU_PAD]
    if term_chunk is None:
        term_chunk = (1 << 12) if loo == "dft" else (1 << 15)
    P = torch.zeros(N, dtype=torch.complex128, device=dev)
    for g0 in range(0, n_terms, term_chunk):
        g = torch.arange(g0, min(g0 + term_chunk, n_terms), dtype=torch.int64, device=dev)
        v = (g[:, None] // strd[None, :]) % rad[None, :]                             # [T, D]
        c = (mut[None, :] - 2 * v).to(torch.complex128)
        w = wt.gather(1, v.T).T.prod(1)                                              # [T]
        a = c @ b.to(torch.complex128)                                               # [T, N]
        if loo == "divide":
            P += torch.einsum("t,tn->n", w.to(torch.complex128), _loo_divide(a, q))
            continue
        y = 1 + lam[None, :, None] * a[:, None, :]                                    # [T, L, N]
        tot = y.prod(2, keepdim=True)
        ab = tot / y                                                                  # all-but-one (no exact zeros here)
        P += torch.einsum("t,l,tln->n", w.to(torch.complex128), coef, ab)
    return P * (factor / 2 ** q)


def resolve_backend(backend: str, dev: torch.device) -> str:
    """"auto" -> "cuda" on a CUDA device with cupy (the thread-per-lane kernel, clifford/cuda_rect.py),
    else "torch"; "emul" is the kernel code compiled for the CPU (tests). "torch" is the plain (uncollapsed) Glynn sum of
    this file; "collapsed" / "triton" the C&C 2020 collapse over repeated placed rows in torch / Triton.
    """
    if backend == "auto":
        if dev.type == "cuda":
            from . import cuda_rect
            return "cuda" if cuda_rect.available() else "torch"
        return "torch"
    if backend not in ("torch", "cuda", "emul", "collapsed", "triton"):
        raise ValueError(f"sbs_rso: backend {backend!r} (auto | torch | cuda | emul | collapsed | triton)")
    return backend


def _collapsed_streams(AT: torch.Tensor, placed: torch.Tensor, fn) -> torch.Tensor:
    """Per-stream collapsed cofactors. AT: [S, m, N], placed: [S, q] row indices (q >= 1). Returns P [S, N]."""
    S, _, N = AT.shape
    dev = AT.device
    P = torch.empty((S, N), dtype=AT.dtype, device=dev)
    rc = placed.cpu()
    for s in range(S):
        uq, mu = torch.unique(rc[s], return_counts=True)
        b = AT[s, uq.to(dev), :].to(torch.complex128)                          # [D, N] distinct placed rows
        P[s] = fn(b, [int(x) for x in mu.tolist()]).to(AT.dtype)
    return P


def _fock_step_chain(A: torch.Tensor, generator, kernel: str) -> torch.Tensor:
    """The chain of a k = n stratum (empty offset set: every step square, N = q + 1) on the production Fock STEP kernels.

    A: [S, m, n] columns already in each stream's uniform random source order. Each step is one call of the Fock
    engine's 2020 cofactor kernel (`cuda_cofactors.cofactor_partials_cuda` for kernel="cuda",
    `cofactors_triton.cofactor_partials_multi` for "triton": collapse of repeated placed rows, all-but-one payload)
    and of its fused `draw_step`, on THIS chain's order and placed rows -- the rso chain at s0 = 0 with the Fock
    payload (check (b) of `bench/sbs_rso_validate.py`), not a hand-off to `fock_sample`. Returns rows [S, n] int64.
    """
    from .cofactors_triton import cofactor_partials_multi, draw_step
    S, m, n = A.shape
    dev = A.device
    if kernel == "cuda":
        from .cuda_cofactors import cofactor_partials_cuda as multi
    else:
        multi = cofactor_partials_multi
    A2 = torch.view_as_real(A.to(torch.complex128).contiguous())                  # [S, m, n, 2]
    sig = torch.arange(n, device=dev).expand(S, n).contiguous()
    dp = torch.arange(S, device=dev, dtype=torch.int64)
    rows = torch.zeros((S, n), dtype=torch.int64, device=dev)
    seeds = torch.randint(0, 2 ** 31 - 1, (n,), generator=generator).tolist()
    ones = torch.zeros(S, 1, 2, dtype=torch.float64, device=dev)
    ones[:, :, 0] = 1.0
    one_lane = torch.ones(S, dtype=torch.int64, device=dev)
    unit = torch.ones(S, dtype=torch.float64, device=dev)
    for K in range(1, n + 1):
        if K == 1:
            part, lstart, nl, scale, mx = ones, dp, one_lane, unit, 1
        else:
            part, lstart, nl, scale, mx, _ = multi(A2, rows, sig, dp, S, K - 1, K)
        draw_step(A2, part, rows, sig, dp, S, m, K, lstart, nl, scale, K - 1, seeds[K - 1], max_lanes=mx)
    return rows


def _chain(A: torch.Tensor, k: int, generator, sign_chunk=None, backend: str = "torch",
           collapse_from: int = 12, draw: str = "host") -> torch.Tensor:
    """Lockstep chain for S streams with the SAME (n, k). A: [S, m, n] columns already in each stream's
    random source order. Returns rows [S, k] int64 (output modes, in placement order).

    backend "torch": `rect_cofactors` (this file); "cuda" / "emul": `cuda_rect.rect_cofactors_cuda` / `_emul`
    (the production thread-per-lane cofactor kernel with the 2020 collapse of repeated placed rows; steps wider than its
    largest tier, N > 64, use the collapsed sum below, Triton on cuda); "collapsed" /
    "triton": steps with q = j - 1 < `collapse_from` placed rows use the batched plain sum, from q = `collapse_from` on
    each stream's cofactors are the Glynn sum collapsed over its repeated rows (`rect_cofactors_collapsed_torch` /
    `sbs_rso_triton.rect_cofactors_collapsed_triton`), one stream at a time.
    draw "host": amplitudes by einsum, row drawn by torch.multinomial on the host (one sync per step); "device": the
    Fock sampler's fused draw (`cofactors_triton.draw_step`: amplitudes, weights and an inverse-CDF Philox draw in one
    Triton launch per step, no host round trip). Same law, a different random stream.
    """
    S, m, n = A.shape
    dev = A.device
    rows = torch.zeros((S, k), dtype=torch.int64, device=dev)
    ar = torch.arange(S, device=dev)
    fn = None
    if backend == "collapsed":
        fn = rect_cofactors_collapsed_torch
    elif backend == "triton":
        from .sbs_rso_triton import rect_cofactors_collapsed_triton as fn
    wide_n, wide_fn = None, None
    if backend in ("cuda", "emul"):
        # cuda_rect's tier ladder stops at N = 64 columns: wider steps go to the collapsed sum (Triton on cuda, else torch)
        from . import cuda_rect
        wide_n = cuda_rect._TIERS_LADDER[-1]
        wide_fn = rect_cofactors_collapsed_torch
        if dev.type == "cuda":
            with contextlib.suppress(ImportError):
                from .sbs_rso_triton import rect_cofactors_collapsed_triton as wide_fn
    if draw not in ("host", "device"):
        raise ValueError(f"sbs_rso: draw {draw!r} (host | device)")
    if draw == "device":
        from .cofactors_triton import draw_step
        A2 = torch.view_as_real(A.to(torch.complex128).contiguous())          # [S, m, n, 2]
        sig = torch.arange(n, device=dev).expand(S, n).contiguous()            # columns already in source order
        one = torch.ones(S, dtype=torch.int64, device=dev)
        unit = torch.ones(S, dtype=torch.float64, device=dev)
        seeds = torch.randint(0, 2 ** 31 - 1, (k,), generator=generator).tolist()
    for j in range(1, k + 1):
        N = n - k + j
        q = j - 1
        AT = A[:, :, :N]                                                       # [S, m, N]
        if j == 1:
            P = torch.ones((S, N), dtype=A.dtype, device=dev)
        elif wide_n is not None and wide_n < N:
            P = _collapsed_streams(AT, rows[:, :q], wide_fn)
        elif backend == "cuda":
            from .cuda_rect import rect_cofactors_cuda
            P = rect_cofactors_cuda(AT.contiguous(), rows[:, :q], q)
        elif backend == "emul":
            from .cuda_rect import rect_cofactors_emul
            P = rect_cofactors_emul(AT.contiguous(), rows[:, :q], q)
        elif fn is None or q < collapse_from:
            B = AT[ar[:, None], rows[:, :q], :]                                # [S, j-1, N]
            P = rect_cofactors(B, sign_chunk)
        else:
            P = _collapsed_streams(AT, rows[:, :q], fn)
        if draw == "device":
            part = torch.view_as_real(P.to(torch.complex128).contiguous())       # [S, N, 2]: one lane per stream
            draw_step(A2, part, rows, sig, ar, S, m, N, ar, one, unit, j - 1, seeds[j - 1])
            continue
        amp = torch.einsum("smn,sn->sm", AT, P)
        w = (amp.real ** 2 + amp.imag ** 2).to(torch.float64)
        w = w / w.sum(1, keepdim=True)
        rows[:, j - 1] = torch.multinomial(w.cpu(), 1, generator=generator)[:, 0].to(dev)
    return rows


def _square_fock(U: torch.Tensor, cols: torch.Tensor, generator, fock_backend: str) -> torch.Tensor:
    """Occupations [S, m] of S Fock instances: stream s sends one photon into each source mode cols[s] (an n-subset).

    The k = n_live strata of the chain: with an empty offset set the rso chain is Clifford & Clifford on the live
    sources, so they are drawn by the Fock engine itself (`fock_sample`, lossless, per-circuit stream counts): one
    circuit per distinct live subset, its source columns moved to the front of U. The device generator is seeded from
    `generator` (deterministic per seed, a different stream of draws than the chain's).
    """
    from .sampler import fock_sample
    S, n = cols.shape
    m = U.shape[0]
    dev = U.device
    live = cols.sort(dim=1).values.cpu()
    sets, inv = torch.unique(live, dim=0, return_inverse=True)                  # [B, n], [S]
    Ub = []
    for b in range(sets.shape[0]):
        rest = torch.ones(m, dtype=torch.bool)
        rest[sets[b]] = False
        perm = torch.cat([sets[b], rest.nonzero(as_tuple=True)[0]]).to(dev)
        Ub.append(U[:, perm])
    Ub = torch.stack(Ub)                                                          # [B, m, m]
    reps = torch.bincount(inv, minlength=sets.shape[0])
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator))
    gen = torch.Generator(device=dev).manual_seed(seed)
    occ_sorted = fock_sample(Ub, list(range(n)), reps.to(dev), generator=gen, backend=fock_backend)   # by circuit
    occ = torch.empty((S, m), dtype=torch.int64)
    occ[torch.argsort(inv, stable=True)] = occ_sorted.cpu().to(torch.int64)
    return occ


def _uniform_alpha(state: ProductSuperposition) -> float:
    amps = torch.as_tensor(np.array(state.amps), dtype=torch.complex128)
    if not torch.allclose(amps, amps[:1].expand_as(amps)):
        raise ValueError("sbs_rso: uniform alpha only (non-uniform sources weight the Dicke terms by prod tan alpha_i)")
    if state.d != 2:
        raise ValueError("sbs_rso: d = 2 only")
    return math.atan2(float(amps[0, 1].abs()), float(amps[0, 0].abs()))


_WARMED: set = set()


def _warm_cuda(square_: str) -> None:
    """Compile every CUDA kernel family the chain can dispatch to, once per process, on the first call: the lambda form
    (`cuda_rect.warmup`), the templated mu lane (`cuda_rect.warmup_mu`) and, for square="kernel", the Fock step kernel
    (`cuda_cofactors.warmup`). Each family otherwise compiles on its own first call, which a caller's untimed warm-up
    sample reaches only if its random photon counts happen to visit that form: run 20261006_000908 timed a 58 s mu
    compile as the first shot at n = 35 (alpha = pi/4).
    """
    want = {"lambda", "mu"} | ({"fock"} if square_ == "kernel" else set())
    todo = want - _WARMED
    if not todo:
        return
    from . import cuda_rect
    if "lambda" in todo:
        cuda_rect.warmup()
    if "mu" in todo:
        cuda_rect.warmup_mu()
    if "fock" in todo:
        from . import cuda_cofactors
        cuda_cofactors.warmup()
    _WARMED.update(todo)


def sample(state: ProductSuperposition, U: torch.Tensor, shots: int, generator: torch.Generator | None = None,
           photons: int | None = None, loss: float | None = None, indist=None,
           dtype: torch.dtype = torch.complex128, device=None, sign_chunk: int | None = None,
           max_streams: int = 1 << 12, stats: dict | None = None, backend: str = "auto",
           collapse_from: int = 12, square: str = "auto", draw: str = "host") -> torch.Tensor:
    """Exact samples [shots, m] (int64 occupations) of a uniform-alpha `ProductSuperposition` through U.

    Per shot: which sources are live (`loss`, product_sampler.lossy_source: a lossy superposed source is a
    mixture of vacuum and a source at a smaller alpha'), which live ones interfere (`indist` x-model; a lone
    source's photon is routed classically), the photon number k ~ Bin(n_live, sin^2 alpha') (or `photons`,
    lossless only), and a uniformly random source order. Shots are then grouped by (n_live, k) and each group
    runs the chain above in lockstep (`max_streams` per pass). `stats["passes"]` receives (n, k, streams,
    seconds) per pass, synchronised on cuda (the stratum costs the run-of-S statistic needs).
    backend: "auto" (the CUDA thread-per-lane kernel `cuda_rect` when cupy and a CUDA device are there, else the
    torch cofactors of this file), "torch", "cuda", "emul" (the kernel code on the CPU, for tests), "collapsed" /
    "triton" (the 2020 collapse in torch / Triton from q = `collapse_from` placed rows on); see `_chain`.
    square: the strata with k = n_live (every live source fired: empty offset set, the chain IS Clifford & Clifford
    on a Fock instance, e.g. alpha = pi/2 or its lossy twin) go to the Fock engine itself (`_square_fock`,
    `fock_sample` with backend "cuda" / "triton") instead of the rectangular chain; "kernel": they stay on the rso chain
    (its own order and draws) but every step runs on the Fock engine's step kernels (`_fock_step_chain`);
    "auto" = "cuda" when the backend is cuda, else "chain" (the rso chain everywhere).
    """
    from ..imperfections import spec
    from .product_sampler import lossy_source
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    dev = torch.device(device) if device is not None else (U.device if isinstance(U, torch.Tensor) else torch.device("cpu"))
    U = torch.as_tensor(U).to(dev, dtype)
    backend_ = resolve_backend(backend, dev)
    if square not in ("auto", "chain", "cuda", "triton", "kernel"):
        raise ValueError(f"sbs_rso: square {square!r} (auto | chain | cuda | triton | kernel)")
    square_ = ("cuda" if backend_ == "cuda" else "chain") if square == "auto" else square
    if backend_ == "cuda" and dev.type == "cuda":
        _warm_cuda(square_)
    m = state.m
    modes = torch.as_tensor(list(state.modes), dtype=torch.int64)
    n0 = modes.shape[0]
    d = spec(indist)
    if d is not None and d.x is None:
        raise NotImplementedError("sbs_rso: the x-model only")
    alpha = _uniform_alpha(state)
    lossy = loss is not None and float(loss) < 1.0
    if photons is not None and (lossy or d is not None):
        raise ValueError("photons= with loss / indist: post-selecting a detected count conditions the masks too")
    w1, ap = lossy_source(alpha, float(loss)) if lossy else (1.0, alpha)
    p1 = math.sin(ap) ** 2
    out = torch.zeros((shots, m), dtype=torch.int64)
    live = torch.rand((shots, n0), generator=generator) < w1
    good = torch.rand((shots, n0), generator=generator) < (1.0 if d is None else d.x)
    lone = live & ~good
    if bool(lone.any()):                                       # lone sources: photon w.p. sin^2 alpha', row by |U[:, mode]|^2
        emit = lone & (torch.rand((shots, n0), generator=generator) < p1)
        s_, i_ = emit.nonzero(as_tuple=True)
        if s_.numel():
            Uabs2 = (U.abs() ** 2).to("cpu", torch.float64)[:, modes[i_]].T          # [E, m]
            r_ = torch.multinomial(Uabs2, 1, generator=generator)[:, 0]
            out.index_put_((s_, r_), torch.ones_like(s_), accumulate=True)
    inter = live & good                                                             # [shots, n0]
    nl = inter.sum(1)
    if photons is not None:
        if not 0 <= photons <= n0:
            raise ValueError(f"photons must be in 0..{n0}")
        ks = torch.full((shots,), int(photons), dtype=torch.int64)
    else:
        ks = torch.binomial(nl.to(torch.float64), torch.full((shots,), p1, dtype=torch.float64), generator=generator).to(torch.int64)
    # a uniformly random order of each shot's interfering sources: sort random keys, dead sources last
    keys = torch.rand((shots, n0), generator=generator) + (~inter).to(torch.float64) * 2.0
    order = modes[keys.argsort(dim=1)]                                              # [shots, n0] source modes
    if stats is not None:
        stats.setdefault("passes", [])
    for n_l, k in sorted({(int(a), int(b)) for a, b in zip(nl.tolist(), ks.tolist())}):
        if k == 0:
            continue
        ids = ((nl == n_l) & (ks == k)).nonzero(as_tuple=True)[0]
        for c0 in range(0, ids.shape[0], max_streams):
            sid = ids[c0:c0 + max_streams]
            cols = order[sid, :n_l].to(dev)                                       # [S, n_l]
            if stats is not None and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            if k == n_l and square_ == "kernel":                                   # rso chain, Fock step kernels
                A = U[:, cols].permute(1, 0, 2).contiguous()
                rows = _fock_step_chain(A, generator, "cuda" if dev.type == "cuda" else "triton")
                occ = torch.zeros((sid.shape[0], m), dtype=torch.int64)
                occ.scatter_add_(1, rows.cpu(), torch.ones_like(rows.cpu()))
            elif k == n_l and square_ != "chain":                                  # empty offset set: a Fock instance
                occ = _square_fock(U, cols, generator, square_)
            else:
                A = U[:, cols].permute(1, 0, 2).contiguous()                       # [S, m, n_l]
                rows = _chain(A, k, generator, sign_chunk, backend_, collapse_from, draw)
                occ = torch.zeros((sid.shape[0], m), dtype=torch.int64)
                occ.scatter_add_(1, rows.cpu(), torch.ones_like(rows.cpu()))
            if stats is not None:
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                stats["passes"].append((n_l, k, int(sid.shape[0]), time.perf_counter() - t0))
            out[sid] += occ.cpu()
    return out
