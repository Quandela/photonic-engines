"""Exact photon-number-resolving sampler for Gaussian inputs: the chain rule over conditional pure
Gaussian states with loop hafnians (docs/GAUSSIAN.md §3; algorithm of Quesada et al.
arXiv:2010.15595 and Bulmer et al. arXiv:2108.01622, reimplemented from the derivation, not from
their code).

One shot, for a pure output state |psi> = N exp(1/2 a^T B a + zeta^T a)|0> (a = creation operators):

 1. Draw heterodyne outcomes alpha [m] of EVERY mode from the Q function (a Gaussian).
 2. For j = 0 .. m-1: modes < j hold their drawn photon numbers n_<j, modes > j their heterodyne
    outcomes alpha_>j. Replace the heterodyne of mode j by photon counting: draw
        n_j ~ P(n_j | n_<j, alpha_>j).
    Valid because the marginal of (n_<j, alpha_>j) does not depend on what is measured on mode j
    (no signalling): the pair (n_<j, alpha_>j) of the previous step is a sample of it.
 3. Projecting modes > j on coherent states leaves a pure Gaussian state on modes <= j,
        exp(1/2 w^T B_<=j w + gamma^T w),   gamma = zeta_<=j + B_<=j,>j conj(alpha_>j),
    so, with z the Bargmann variable of mode j and n_<j fixed,
        G(z) = Q(z) exp(b z^2 / 2 + g z),   Q(z) = lhaf(B_<j[n_<j], gamma_<j[n_<j] + z B_<j,j[n_<j]]),
    b = B[j, j], g = gamma[j]; P(n_j = k | .) = k! |[z^k] G|^2 / Z with Z = sum_k k! |[z^k] G|^2.
    Q is a polynomial of degree <= N_<j (`hafnian.lhaf_poly`: one sieve prices every k), and
    Z = (1/pi) int |G|^2 exp(-|z|^2) d^2z is a Gaussian moment sum, closed form (`_norm`).
    k is drawn by inverse CDF against that Z, so no photon-number cutoff enters anywhere.

Mixed states (loss): a lossy squeezed vacuum is a pure squeezed vacuum at a smaller r plus a random
classical displacement (`imperfections.lossy_squeezing`); each shot draws the displacement, then
samples the pure displaced state. Exact, no truncation. A general lossy transfer matrix T (non-uniform loss inside the
interferometer) takes the same route through the Williamson decomposition of the output covariance
(`transfer_decomposition`, `sample_transfer`).

Cost of a shot: sum_j 2^(N_<j / 2) x poly, i.e. O(m N^3 2^(N/2)) with N the detected photon number;
no collision discount yet (v1). Everything is complex128. Per-shot loop on the host; the loop
hafnian runs on `device` (CPU or CUDA), batched over its 2^(N/2) subsets.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..imperfections import lossy_squeezing
from ..states import GaussianInput
from .hafnian import expand, lhaf_poly, lhaf_poly_multi, lhaf_poly_small

EPS = float(np.finfo(np.float64).eps)
V2_FROM = 14          # impl "v2" uses the batched, collision-aware sieve from this many vertices (was 16, the CPU crossover of
                      # 2026-09-23; with the Triton kernels the 14-15-vertex sieves went through torch v1 on the GPU at ~5 ms per
                      # call against ~1 ms for the kernels at N = 16, run 20260924_202647)
SMALL_BELOW = 14      # impl "v2": sieves under this many vertices by the numpy `lhaf_poly_small`, with windows (2026-09-24:
                      # x3-5 over torch per call at N <= 11, x1.5 at 15, none from 17 on two CPU cores)


@dataclass
class Profile:
    """Wall-time breakdown of `sample_pure`, filled while `sampler.PROFILE` is set to one (bench only).

    `sieve[(N, V, device)] = [calls, seconds]`: the loop-hafnian call and its copy back to the host (the copy
    synchronises the device, so the time includes the kernels); `draw` = the normaliser and inverse-CDF draw
    per step; `prep` = the host work that builds the sieve inputs (gamma / expand / loop vectors).
    `events` (a list, when not None): every call in order, ("prep", seconds), ("sieve", N, V, kind, device,
    seconds) with kind "none" (N = 0, no sieve), "plain" (torch v1), "small" (numpy) or "multi"
    (`lhaf_poly_multi`), and ("draw", N, seconds) per step; `timing.plan_calls` replays the sieve events.
    """
    sieve: dict = field(default_factory=dict)
    draw: list = field(default_factory=lambda: [0, 0.0])
    prep: list = field(default_factory=lambda: [0, 0.0])
    events: list | None = None

    def add(self, key, dt):
        """Accumulate one call of `dt` seconds under `key` ('draw', 'prep' or a sieve tuple)."""
        slot = getattr(self, key) if isinstance(key, str) else self.sieve.setdefault(key, [0, 0.0])
        slot[0] += 1; slot[1] += dt

    def event(self, *ev):
        """Append one event to `events` when recording (see the class docstring)."""
        if self.events is not None:
            self.events.append(ev)


PROFILE: Profile | None = None


@dataclass
class ShotStats:
    """Per-shot bookkeeping of `sample(..., return_stats=True)`: detected photons, wall time, and
    the largest loop hafnian (vertex count) evaluated. `censored[i]` is True when shot i was aborted
    at the `max_photons` cap; its `photons` entry is then the count reached (> cap), a lower bound.
    """
    photons: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    max_lhaf: list[int] = field(default_factory=list)
    norm_rel_err: list[float] = field(default_factory=list)
    round_off: list[float] = field(default_factory=list)
    censored: list[bool] = field(default_factory=list)


# ---- the single-mode step ---------------------------------------------------------------------
def _gauss_coeffs(b: complex, g: complex, K: int) -> np.ndarray:
    """H_i = sqrt(i!) [z^i] exp(b z^2/2 + g z), i = 0..K (scaled Hermite recurrence, no overflow):
    (i+1) h_{i+1} = g h_i + b h_{i-1}  ->  H_{i+1} = (g H_i + b sqrt(i) H_{i-1}) / sqrt(i+1).
    """
    H = np.zeros(K + 1, dtype=np.complex128); H[0] = 1.0
    if K >= 1:
        H[1] = g
    for i in range(1, K):
        H[i + 1] = (g * H[i] + b * math.sqrt(i) * H[i - 1]) / math.sqrt(i + 1)
    return H


def _amps(q: np.ndarray, b: complex, g: complex, K: int) -> np.ndarray:
    """c_k = sqrt(k!) [z^k] Q(z) exp(b z^2/2 + g z), k = 0..K, for Q with coefficients q:
    c_k = sum_l q_l H_{k-l} sqrt(k! / (k-l)!), as one [K+1, L] matrix-vector product.
    """
    H = _gauss_coeffs(b, g, K)
    L = min(len(q), K + 1)
    k = np.arange(K + 1)[:, None]; l = np.arange(L)[None, :]
    d = k - l
    ok = d >= 0
    lg = np.array([math.lgamma(i + 1) for i in range(K + 1)])
    f = np.where(ok, np.exp(0.5 * (lg[k] - lg[np.where(ok, d, 0)])), 0.0)
    T = np.where(ok, H[np.where(ok, d, 0)], 0.0) * f
    return T @ q[:L]


def _norm(q: np.ndarray, b: complex, g: complex) -> float:
    """Z = (1/pi) int |Q(z)|^2 |exp(b z^2/2 + g z)|^2 exp(-|z|^2) d^2z = sum_k |c_k|^2, closed form.

    The weight is a real 2D Gaussian in z (needs |b| < 1): with v = (z, conj z),
    E[z] = (conj(b) g + conj(g)) / (1 - |b|^2), E[dz dz] = conj(b)/(1-|b|^2), E[dz dz*] = 1/(1-|b|^2),
    mass Z0 = exp((Re(conj(b) g^2) + |g|^2) / (1 - |b|^2)) / sqrt(1 - |b|^2) (docs/GAUSSIAN.md §3.3), and the
    moments m_{l,l'} = E[z^l conj(z)^l'] follow from Stein's recurrence, one row (vectorised in l') at a time.
    """
    ab2 = abs(b) ** 2
    if ab2 >= 1.0:
        raise ValueError("|B_jj| >= 1: not a normalisable Gaussian state")
    s = 1.0 / (1.0 - ab2)
    mu = (b.conjugate() * g + g.conjugate()) * s
    czz, czc = b.conjugate() * s, s
    logZ0 = 0.5 * s * (b.conjugate() * g * g + 2 * abs(g) ** 2 + b * g.conjugate() ** 2).real - 0.5 * math.log(1.0 - ab2)
    L = len(q)
    M = np.zeros((L, L), dtype=np.complex128)
    # row 0: m_{0,l'} = conj(mu) m_{0,l'-1} + (l'-1) conj(czz) m_{0,l'-2}
    M[0, 0] = 1.0
    for lp in range(1, L):
        M[0, lp] = mu.conjugate() * M[0, lp - 1] + ((lp - 1) * czz.conjugate() * M[0, lp - 2] if lp >= 2 else 0.0)
    lps = np.arange(1, L)
    for l in range(1, L):                                       # m_{l,l'} = mu m_{l-1,l'} + (l-1) czz m_{l-2,l'} + l' czc m_{l-1,l'-1}
        row = mu * M[l - 1]
        if l >= 2:
            row = row + (l - 1) * czz * M[l - 2]
        row[1:] += lps * czc * M[l - 1, :-1]
        M[l] = row
    val = (q @ M @ q.conj()).real
    return float(val) * math.exp(logZ0)


def _draw_count(q: np.ndarray, b: complex, g: complex, u: float) -> tuple[int, float]:
    """Inverse-CDF draw of k from P(k) = |c_k|^2 / Z against the closed-form Z of `_norm`.

    The amplitudes are generated to a growing K until the partial sum passes u Z. Returns (k, gap),
    gap = 1 - (partial sum at the cap) / Z when rounding left the partial sums below u Z at the cap
    (then k is the cap, a draw from a tail of relative mass < gap); gap = 0 otherwise.
    """
    Z = _norm(q, b, g)
    target = u * Z
    ab = max(abs(b), 1e-300)
    kmax = len(q) + 64 + int(80.0 / max(-math.log(ab), 1e-3))   # tail below ~e^-80 relative by then
    K = min(kmax, max(16, 2 * len(q) + 8))
    while True:
        cum = np.cumsum(np.abs(_amps(q, b, g, K)) ** 2)
        k = int(np.searchsorted(cum, target, side="left"))
        if k <= K:
            return k, 0.0
        if kmax <= K:
            return K, max(0.0, 1.0 - float(cum[-1]) / Z)
        K = min(2 * K, kmax)


# ---- one shot -----------------------------------------------------------------------------------
def _heterodyne(r: np.ndarray, U: np.ndarray, modes, beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Alpha ~ Q function of D(beta) U prod S(r)|0>: alpha = beta + U eps with eps independent per
    INPUT mode: a squeezed source has Var Re = cosh(r) e^-r / 2, Var Im = cosh(r) e^r / 2, a vacuum
    input mode Var Re = Var Im = 1/2 (E|eps|^2 = 1, the Q function of |0>).
    """
    m = U.shape[0]
    sx = np.full(m, math.sqrt(0.5)); sy = np.full(m, math.sqrt(0.5))
    idx = list(modes); c = np.cosh(r)
    sx[idx] = np.sqrt(c * np.exp(-r) / 2); sy[idx] = np.sqrt(c * np.exp(r) / 2)
    eps = rng.normal(size=m) * sx + 1j * rng.normal(size=m) * sy
    return beta + U @ eps


def sample_pure(B: np.ndarray, zeta: np.ndarray, alpha: np.ndarray, rng: np.random.Generator,
                device=None, stats: ShotStats | None = None, cpu_below: int = 14, impl: str = "v2",
                window: int = 4, max_photons: int | None = None, precision: str | None = None) -> np.ndarray:
    """The chain of steps 2-3 for one shot of the pure state (B, zeta) given heterodyne outcomes
    alpha [m] drawn from its Q function. Returns the photon numbers [m] (int64). `precision`: of the v2 sieves
    (`hafnian.lhaf_poly_multi`: "fp64", "dd" or "auto"; None = its module default); the small numpy sieves stay fp64.

    impl "v1": one `lhaf_poly` per mode. impl "v2": speculative windows over runs of empty modes. While
    the next modes draw 0 photons the matrix B_<j[n_<j] does not change; only the loop vectors do. So
    the loop hafnians of the next `window` steps are computed in ONE `lhaf_poly_multi` call (the matrix
    work shared), as if those modes were empty, and the draws then proceed in order; the first nonzero
    draw ends the window and the rest of it is discarded. v2 also uses the collision-aware sieve (a mode
    holding t photons gives floor(t/2) identical pairs, enumerated by count). The draws consume the
    generator exactly as v1 does, so v1 and v2 give the same samples up to rounding.

    `max_photons`: abort the shot as soon as the running count exceeds it (the modes not reached stay 0,
    and `stats.censored` records True). The count only grows along the chain, so a shot whose final
    count is <= max_photons is never aborted and is drawn exactly as without the cap; this is what makes
    the cap exact for the cost estimator stratified by N (gaussian/cost.py). A censored row is NOT a sample.

    impl "v2" computes the sieves under `SMALL_BELOW` vertices with the numpy `lhaf_poly_small` (windows too),
    the others with `lhaf_poly_multi`.

    Loop hafnians with fewer than `cpu_below` vertices run on the CPU whatever `device` is: below that size a
    GPU call is launch-bound and the chain spends most of its steps there. Default 14 since 2026-09-24 (was 18,
    the v1 crossover of run 20260923_224054): with v2.2 a GPU sieve at N = 16 takes 2.2 ms against ~30 ms per
    call for the CPU sieves of 16-17 vertices inside a shot (run 20260924_162756), and the numpy small sieve
    takes over below 14.
    """
    if impl not in ("v1", "v2"):
        raise ValueError("impl must be 'v1' or 'v2'")
    m = B.shape[0]
    n = np.zeros(m, dtype=np.int64)
    ac = alpha.conj()
    worst, biggest, rnd = 0.0, 0, 0.0
    cap = math.inf if max_photons is None else max_photons
    j, cut = 0, False
    while j < m and not cut:
        N = int(n[:j].sum())
        biggest = max(biggest, N)
        small = impl == "v2" and 0 < N < min(SMALL_BELOW, cpu_below)   # numpy small sieve, windows as below
        plain = impl == "v1" or (N < V2_FROM and not small)         # torch v1 (less overhead than v2 there)
        P = PROFILE
        if N == 0 or plain or window <= 1:
            t0 = time.perf_counter()
            gamma = zeta[: j + 1] + B[: j + 1, j + 1:] @ ac[j + 1:]
            if N == 0:
                q = np.ones(1, dtype=np.complex128)
                if P is not None:
                    t1 = time.perf_counter(); P.add("prep", t1 - t0); P.event("prep", t1 - t0)
                    P.event("sieve", 0, 1, "none", "cpu", 0.0); t0 = t1
            else:
                A, D0 = expand(B[:j, :j], gamma[:j], n[:j])
                D1 = np.repeat(B[:j, j], n[:j])
                dev = device if cpu_below <= N else "cpu"
                if P is not None:
                    t1 = time.perf_counter(); P.add("prep", t1 - t0); P.event("prep", t1 - t0); t0 = t1
                if plain:
                    q, qa = lhaf_poly(A, D0, D1, device=dev, return_abs=True)
                    q, qa = q.cpu().numpy(), qa.cpu().numpy()
                elif small:
                    q, qa = lhaf_poly_small(A, D0[None], D1[None], return_abs=True)
                    q, qa = q[0], qa[0]
                else:
                    q, qa = lhaf_poly_multi(A, D0[None], D1[None], device=dev, return_abs=True,
                                            labels=np.repeat(np.arange(j), n[:j]), precision=precision)
                    q, qa = q[0].cpu().numpy(), qa[0].cpu().numpy()
                if P is not None:
                    t1 = time.perf_counter(); P.add((N, 1, str(dev)), t1 - t0)
                    P.event("sieve", N, 1, "plain" if plain else "small" if small else "multi", str(dev), t1 - t0); t0 = t1
                rnd = max(rnd, EPS * float(qa.max()) / max(float(np.abs(q).max()), 1e-300))
            k, gap = _draw_count(q, complex(B[j, j]), complex(gamma[j]), float(rng.random()))
            if P is not None:
                dt = time.perf_counter() - t0; P.add("draw", dt); P.event("draw", N, dt)
            worst = max(worst, gap)
            n[j] = k
            j += 1
            cut = n.sum() > cap
            continue
        # v2: window over steps j .. j + V - 1, assuming the modes in between stay empty
        t0 = time.perf_counter()
        J = list(range(j, min(m, j + window)))
        idx = np.repeat(np.arange(j), n[:j])                        # the N occupied vertices (modes < j)
        A = B[np.ix_(idx, idx)]
        D0 = np.stack([zeta[idx] + B[idx, jj + 1:] @ ac[jj + 1:] for jj in J])
        D1 = np.stack([B[idx, jj] for jj in J])
        dev = device if cpu_below <= N else "cpu"
        if P is not None:
            t1 = time.perf_counter(); P.add("prep", t1 - t0); P.event("prep", t1 - t0); t0 = t1
        if small:
            Q, QA = lhaf_poly_small(A, D0, D1, return_abs=True)
        else:
            Q, QA = lhaf_poly_multi(A, D0, D1, device=dev, return_abs=True, labels=idx, precision=precision)
            Q, QA = Q.cpu().numpy(), QA.cpu().numpy()
        if P is not None:
            t1 = time.perf_counter(); P.add((N, len(J), str(dev)), t1 - t0)
            P.event("sieve", N, len(J), "small" if small else "multi", str(dev), t1 - t0); t0 = t1
        for v, jj in enumerate(J):
            q = Q[v]
            rnd = max(rnd, EPS * float(QA[v].max()) / max(float(np.abs(q).max()), 1e-300))
            g = complex(zeta[jj] + B[jj, jj + 1:] @ ac[jj + 1:])
            k, gap = _draw_count(q, complex(B[jj, jj]), g, float(rng.random()))
            if P is not None:
                t1 = time.perf_counter(); P.add("draw", t1 - t0); P.event("draw", N, t1 - t0); t0 = t1
            worst = max(worst, gap)
            n[jj] = k
            j = jj + 1
            if k > 0:
                cut = n.sum() > cap
                break
    if stats is not None:
        stats.max_lhaf.append(biggest); stats.norm_rel_err.append(worst); stats.round_off.append(rnd)
        stats.censored.append(bool(cut))
    return n


def sample(state: GaussianInput, U, shots: int, loss=None, seed: int | None = None,
           device=None, return_stats: bool = False, schmidt=None, cpu_below: int = 14, impl: str = "v2",
           window: int | None = None, max_photons: int | None = None, order: str = "natural",
           precision: str | None = None):
    """Exact PNR samples [shots, m] (int64, CPU) of `state` through the interferometer U [m, m].

    `loss=eta`: uniform transmission, applied per source as the pure-plus-classical decomposition of
    `imperfections.lossy_squeezing` (and sqrt(eta) on any displacement); exact.
    `schmidt=[lam_1, ..., lam_K]` (sum 1): spectral impurity (docs/GAUSSIAN.md §4.2). Every squeezer
    emits into K orthogonal Schmidt modes with squeezing r_k = r sqrt(lam_k) (the mapping of an
    undisplaced multimode squeezer without time-ordering corrections); photons of different Schmidt
    modes do not interfere, so a shot is the sum of K independent GBS shots through the same U. Any
    displacement stays in the first (leading) Schmidt mode. None = one mode, pure.
    `impl` / `window`: "v2" (default) batches the loop hafnians of up to `window` consecutive steps over a
    run of empty modes (`sample_pure`); "v1" is the plain one-sieve-per-step chain, kept for comparison.
    window None: 4 (on the H100 a window of 8 is no faster than 4 on whole shots after the subset-table
    change: 0.494 vs 0.460 s/shot at n = 40, run 20260924_084540).
    `max_photons`: the cost-benchmark cap of `sample_pure`, on the total over the Schmidt modes; a
    censored shot's row is set to -1 (not a sample) and `stats.censored` is True for it.
    `order`: "natural" = the chain visits the output modes 0..m-1; "mean" = in increasing exact mean photon
    number (as The Walrus's sampler does): the running count then stays low for more of the steps, so fewer of
    them pay the large sieves. A relabelling of the outputs fixed before any draw, so the law of the samples is the
    same (tested); the samples for a given seed are not. No expected gain for a Haar U (equal means).
    `precision`: arithmetic of the v2 sieves, "fp64", "dd" or "auto" (`hafnian.lhaf_poly_multi`; None = its default).
    `seed` seeds numpy's generator (every random draw is host-side). `device` is where the loop
    hafnians of >= `cpu_below` vertices run. With `return_stats`, also returns a `ShotStats` (one
    entry per shot; with K Schmidt modes, photons / time summed and max_lhaf maximised over them).
    """
    if not isinstance(state, GaussianInput):
        raise TypeError("gaussian.sample takes a states.GaussianInput")
    U = np.asarray(U.detach().cpu().numpy() if hasattr(U, "detach") else U, dtype=np.complex128)
    if window is None:
        window = 4
    if U.shape != (state.m, state.m):
        raise ValueError(f"U must be [{state.m}, {state.m}]")
    rng = np.random.default_rng(seed)
    if schmidt is None:
        parts = [(state.r, state.disp)]
    else:
        lam = np.asarray(schmidt, dtype=np.float64).reshape(-1)
        if (lam < 0).any() or abs(lam.sum() - 1) > 1e-12:
            raise ValueError("schmidt weights must be >= 0 and sum to 1")
        parts = [(state.r * math.sqrt(lk), state.disp if k == 0 else None) for k, lk in enumerate(lam) if lk > 0]
    e = 1.0 if loss is None else float(loss)
    if order not in ("natural", "mean"):
        raise ValueError("order must be 'natural' or 'mean'")
    perm = None
    if order == "mean":                                   # <n_j> = eta sum_s |U_js|^2 sum_k sinh^2 r_sk + |beta_j|^2
        Uc0 = U[:, list(state.modes)]
        src = sum(np.sinh(r) ** 2 for r, _ in parts)
        mean = e * (np.abs(Uc0) ** 2) @ src
        if state.disp is not None:
            mean = mean + e * np.abs(Uc0 @ state.disp) ** 2
        perm = np.argsort(mean, kind="stable")
        U = U[perm]                                       # the chain's mode i is output mode perm[i]
    Uc = U[:, list(state.modes)]
    prep = []
    for r, disp in parts:
        r_pure, var_p = lossy_squeezing(r, loss)
        B, _, _ = state.bargmann(U, r=r_pure, disp=np.zeros(state.sources))
        prep.append((r_pure, var_p, B, disp))
    out = np.zeros((shots, state.m), dtype=np.int64)
    stats = ShotStats() if return_stats else None
    for s in range(shots):
        t0 = time.perf_counter()
        sub = ShotStats()
        cut = False
        for r_pure, var_p, B, disp in prep:
            d = np.zeros(state.sources, dtype=np.complex128) if disp is None else math.sqrt(e) * disp
            if np.any(var_p > 0):
                d = d + 1j * rng.normal(size=state.sources) * np.sqrt(var_p)
            beta = Uc @ d
            zeta = beta - B @ beta.conj()
            alpha = _heterodyne(r_pure, U, state.modes, beta, rng)
            left = None if max_photons is None else max_photons - int(out[s].sum())
            out[s] += sample_pure(B, zeta, alpha, rng, device=device, stats=sub, cpu_below=cpu_below, impl=impl,
                                  window=window, max_photons=left, precision=precision)
            if sub.censored[-1]:
                cut = True
                break
        if stats is not None:
            stats.photons.append(int(out[s].sum())); stats.seconds.append(time.perf_counter() - t0)
            stats.censored.append(cut)
            stats.max_lhaf.append(max(sub.max_lhaf)); stats.norm_rel_err.append(max(sub.norm_rel_err))
            stats.round_off.append(max(sub.round_off))
        if cut:
            out[s] = -1
    if perm is not None:
        o2 = np.empty_like(out); o2[:, perm] = out; out = o2
    return (out, stats) if return_stats else out


# ---- a general linear-optical channel: a lossy transfer matrix T ----------------------------------
def _quadrature_matrix(T: np.ndarray) -> np.ndarray:
    """The real [2m, 2m] action of a = x + i p -> T a on (x, p) in xxpp order."""
    return np.block([[T.real, -T.imag], [T.imag, T.real]])


def transfer_decomposition(state: GaussianInput, T) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-plus-classical decomposition of the Gaussian state `state` sent through the lossy transfer matrix T.

    T [m, m] complex (m = state.m), a contraction (singular values <= 1): a = x + i p -> T a, the missing
    transmission going to environment modes that are traced out. Output quadrature covariance (vacuum I/4, xxpp):
        V = T_R V_in T_R^T + (I - T_R T_R^T) / 4.
    Williamson, V = S D S^T with S symplectic and D = diag(d, d) >= 1/4, splits V into the PURE state S S^T / 4 plus
    the classical noise W = S (D - I/4) S^T >= 0: the mixed state is the pure one displaced by a Gaussian random
    (x, p) ~ N(0, W). The pure part with zero mean is U' prod S(r')|0>: its <da da^T> = Vxx - Vpp + i (Vxp + Vpx)
    equals -U' diag(sinh 2 r') U'^T / 2, a Takagi factorisation, computed from the positive eigenvectors of the
    real symmetric [[A, B], [B, -A]] of -2 <da da^T> = A + i B (eigenvector [u; v] -> Takagi column u + i v).
    Returns (r_pure [k], U_pure [m, m] unitary whose first k columns carry the squeezers, W [2m, 2m],
    beta0 [m] = T @ the input displacement). k counts the squeezers above 1e-12 in sinh 2r; the other
    columns complete the unitary (they carry vacuum, so their choice does not matter). Exact, no truncation; the
    uniform-loss decomposition `imperfections.lossy_squeezing` is the special case T = sqrt(eta) U.
    """
    T = np.asarray(T.detach().cpu().numpy() if hasattr(T, "detach") else T, dtype=np.complex128)
    if T.shape != (state.m, state.m):
        raise ValueError(f"T must be [{state.m}, {state.m}]")
    if np.linalg.norm(T, 2) > 1 + 1e-10:
        raise ValueError("T must be a contraction (largest singular value <= 1)")
    mi, mo = state.m, T.shape[0]
    vin = np.full(2 * mi, 0.25)
    idx = np.asarray(state.modes)
    vin[idx] = np.exp(-2 * state.r) / 4; vin[mi + idx] = np.exp(2 * state.r) / 4
    TR = _quadrature_matrix(T)
    V = (TR * vin) @ TR.T + (np.eye(2 * mo) - TR @ TR.T) / 4
    V = (V + V.T) / 2
    # Williamson: A = V^-1/2 Omega V^-1/2 is antisymmetric; iA is Hermitian with eigenvalues +-1/d. For a +lambda
    # eigenvector z, e1 = sqrt2 Re z and e2 = -sqrt2 Im z satisfy A e1 = -lambda e2, A e2 = lambda e1, so
    # O = [e1..., e2...] gives O^T A O = [[0, L], [-L, 0]] and S = V^1/2 O D^-1/2 (S Omega S^T = Omega, S D S^T = V)
    w, Q = np.linalg.eigh(V)
    if w.min() <= 0:
        raise ValueError("output covariance is not positive definite")
    Vh = (Q * np.sqrt(w)) @ Q.T; Vmh = (Q / np.sqrt(w)) @ Q.T
    Om = np.block([[np.zeros((mo, mo)), np.eye(mo)], [-np.eye(mo), np.zeros((mo, mo))]])
    lam, Z = np.linalg.eigh(1j * (Vmh @ Om @ Vmh))
    lam, Z = lam[mo:], Z[:, mo:]                                    # the m positive ones (ascending)
    O = np.concatenate([math.sqrt(2) * Z.real, -math.sqrt(2) * Z.imag], axis=1)
    d = np.concatenate([1 / lam, 1 / lam])
    S = (Vh @ O) / np.sqrt(d)
    W = (S * np.maximum(d - 0.25, 0.0)) @ S.T
    Vp = S @ S.T / 4
    Mx = Vp[:mo, :mo] - Vp[mo:, mo:] + 1j * (Vp[:mo, mo:] + Vp[mo:, :mo])
    M2 = -2 * Mx
    M2 = (M2 + M2.T) / 2
    ev, E = np.linalg.eigh(np.block([[M2.real, M2.imag], [M2.imag, -M2.real]]))
    keep = ev > 1e-12
    sig = ev[keep]; Wt = E[:mo, keep] + 1j * E[mo:, keep]           # [m, k] Takagi columns, orthonormal
    r_pure = np.arcsinh(sig) / 2
    k = sig.size
    if k < mo:                                                     # complete the unitary on the vacuum columns
        rng = np.random.default_rng(0)
        X = rng.normal(size=(mo, mo - k)) + 1j * rng.normal(size=(mo, mo - k))
        X -= Wt @ (Wt.conj().T @ X)
        Qc, _ = np.linalg.qr(X)
        Up = np.concatenate([Wt, Qc], axis=1)
    else:
        Up = Wt
    beta0 = np.zeros(mo, dtype=np.complex128)
    if state.disp is not None:
        dfull = np.zeros(mi, dtype=np.complex128); dfull[idx] = state.disp
        beta0 = T @ dfull
    return r_pure, Up, W, beta0


def sample_transfer(state: GaussianInput, T, shots: int, seed: int | None = None, device=None,
                    return_stats: bool = False, cpu_below: int = 14, impl: str = "v2", window: int | None = None,
                    max_photons: int | None = None, decomposition=None, precision: str | None = None):
    """Exact PNR samples [shots, m] (int64, CPU) of `state` through the lossy transfer matrix T [m, m] (m = state.m).

    The general linear-optical channel (a published device transfer matrix: non-uniform loss, loss inside the
    interferometer), where `sample(loss=eta)` only takes uniform loss at the sources. `transfer_decomposition` gives
    the pure part (squeezers r' through U') and the classical noise W; each shot draws a displacement
    beta = beta0 + (x + i p), (x, p) ~ N(0, W), then runs the chain of `sample_pure` on the displaced pure state.
    Exact, no truncation. The chain visits the output modes in their order (no `order`), one spectral mode (no
    `schmidt`); `impl`, `window`, `max_photons`, `device`, `return_stats`, `precision` as for `sample`. `decomposition`: the
    result of `transfer_decomposition(state, T)`, to reuse it across calls (0.1-0.2 s at m = 216); T is then unused.
    """
    if not isinstance(state, GaussianInput):
        raise TypeError("gaussian.sample_transfer takes a states.GaussianInput")
    if window is None:
        window = 4
    r_pure, Up, W, beta0 = transfer_decomposition(state, T) if decomposition is None else decomposition
    mo = Up.shape[0]; k = r_pure.size
    pure = GaussianInput(r_pure, tuple(range(k)), mo)
    B, _, _ = pure.bargmann(Up, disp=np.zeros(k))
    wv, Qw = np.linalg.eigh(W)
    Lw = Qw * np.sqrt(np.maximum(wv, 0.0))                          # W = Lw Lw^T
    rng = np.random.default_rng(seed)
    out = np.zeros((shots, mo), dtype=np.int64)
    stats = ShotStats() if return_stats else None
    for s in range(shots):
        t0 = time.perf_counter()
        sub = ShotStats()
        xp = Lw @ rng.normal(size=2 * mo)
        beta = beta0 + xp[:mo] + 1j * xp[mo:]
        zeta = beta - B @ beta.conj()
        alpha = _heterodyne(r_pure, Up, range(k), beta, rng)
        out[s] = sample_pure(B, zeta, alpha, rng, device=device, stats=sub, cpu_below=cpu_below, impl=impl,
                             window=window, max_photons=max_photons, precision=precision)
        cut = bool(sub.censored[-1])
        if stats is not None:
            stats.photons.append(int(out[s].sum())); stats.seconds.append(time.perf_counter() - t0)
            stats.censored.append(cut)
            stats.max_lhaf.append(max(sub.max_lhaf)); stats.norm_rel_err.append(max(sub.norm_rel_err))
            stats.round_off.append(max(sub.round_off))
        if cut:
            out[s] = -1
    return (out, stats) if return_stats else out
