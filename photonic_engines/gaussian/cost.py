"""The expected cost of a GBS sample, stratified by the detected photon number (docs/GAUSSIAN.md §2).

The cost of one exact shot is heavy-tailed: it grows ~2^(N/2) with the detected photon number N while
P(N) decays only geometrically, so a plain mean over a few hundred shots is dominated by whether the
rare bright shots happened to be drawn (run 20260924_071956: the same frontier search gave n* = 38.0
and 31.4 on two runs). But P(N) is known EXACTLY and does not depend on U — it is the convolution of
the per-squeezer laws, thinned by the loss — so the mean can be stratified:

    E[cost] = sum_N P(N) E[cost | N],

with E[cost | N] estimated per stratum from the shots that landed there, and from a fit
log t = a + b N over the upper strata where a stratum has too few shots (in particular above the
largest N drawn). The estimator reports the share of E[cost] that rests on the fit rather than on
measured strata (`extrapolated`), which is the honest error bar of the tail: a frontier point whose
extrapolated share is large is a cost-model number, not a measurement.

It IS large at the headline point: with a cost ~ rho^N (rho = sqrt 2 for the loop-hafnian chain rule)
the cost-weighted photon number law is P(N) rho^N, i.e. exactly the law of squeezers with
tanh r' = rho tanh r — 2.0 photons per squeezer instead of 0.5 at r = 0.658 — so E[cost] rests on
shots four times brighter than the typical one, which a few hundred shots at r never contain.

IMPORTANCE SAMPLING, exact for lossless undisplaced inputs (`importance_cost`). For B = -U diag(tanh r) U^T
the amplitude of a pattern with N photons is proportional to prod tanh(r)^.. only through a common
factor when every tanh r_i is scaled by the same s: B -> s B multiplies every N-photon amplitude by
s^(N/2). So the conditional law P(pattern | N) does not depend on s (checked to 2e-16 against
`probs.pure_probabilities`), and neither does the cost of a shot given its pattern. Hence

    E_r[cost] = E_{r'}[ cost x P_r(N) / P_{r'}(N) ],   tanh r'_i = s tanh r_i,

unbiased with exact weights, and with s = rho the weight cancels the rho^N growth: a near
zero-variance estimator of the mean cost from shots at the tilted squeezing (`tilted`). Not exact
under loss or with Schmidt modes (the mixture breaks the common scaling); those use `stratified_cost`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..states import GaussianInput


def squeezed_law(r: float, cutoff: int) -> np.ndarray:
    """P(k), k = 0..cutoff, of one squeezed vacuum: C(k, k/2) 2^-k tanh^k r / cosh r on even k."""
    p = np.zeros(cutoff + 1)
    t2 = math.tanh(r) ** 2
    c = 1.0 / math.cosh(r)
    for j in range(cutoff // 2 + 1):
        p[2 * j] = c
        c *= t2 * (2 * j + 1) / (2 * j + 2)
    return p


def thin(p: np.ndarray, eta: float | None) -> np.ndarray:
    """Binomial thinning of a photon-number law by a transmission eta (None / 1: unchanged)."""
    if eta is None or eta >= 1.0:
        return p.copy()
    K = len(p) - 1
    out = np.zeros(K + 1)
    for n, pn in enumerate(p):
        if pn == 0.0:
            continue
        k = np.arange(n + 1)
        logc = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
        with np.errstate(divide="ignore"):
            w = np.exp(logc + k * math.log(eta) + (n - k) * (math.log1p(-eta) if eta < 1 else -np.inf))
        out[: n + 1] += pn * w
    return out


def detected_law(state: GaussianInput, cutoff: int, loss=None, schmidt=None) -> np.ndarray:
    """Exact law of the DETECTED photon number, N = 0..cutoff, for undisplaced squeezed vacua through any
    passive U: each squeezer (and each Schmidt mode r sqrt(lam_k)) independently, thinned by `loss`,
    convolved. The mass above `cutoff` is 1 - sum.
    """
    if state.disp is not None and np.any(state.disp != 0):
        raise NotImplementedError("detected_law is for undisplaced inputs")
    lams = [1.0] if schmidt is None else [float(x) for x in schmidt]
    law = np.zeros(cutoff + 1); law[0] = 1.0
    for r in state.r:
        for lk in lams:
            q = thin(squeezed_law(r * math.sqrt(lk), cutoff), loss)
            law = np.convolve(law, q)[: cutoff + 1]
    return law


def binomial_law(n: int, p: float, cutoff: int | None = None) -> np.ndarray:
    """Law of the surviving photon count of n single photons kept independently with probability p —
    Binomial(n, p) on k = 0..cutoff (default n), zero above n: the stratum law of every engine whose
    per-shot cost is set by how many photons interfere (Fock, emitter, NLBS and adaptive under loss
    with p = eta, or eta x with partial distinguishability; sbs with p = eta sin^2 alpha). It is what
    `stratified_cost` / `run_cutoff` take in place of `detected_law` for those engines (docs/GAUSSIAN.md §2:
    the Fock law Phi = z and the sbs law Phi = 1 - p + pz are the same statement).
    """
    top = n if cutoff is None else cutoff
    law = np.zeros(top + 1)
    delta = np.zeros(n + 1); delta[n] = 1.0
    law[: min(n, top) + 1] = thin(delta, p)[: min(n, top) + 1]
    return law


@dataclass
class CostEstimate:
    """Result of `stratified_cost`: seconds per sample (mean), the plain mean for comparison, the share
    of the mean resting on the fit (`extrapolated`), the fitted growth per photon exp(b), the largest N
    drawn, the N at which the sum was stopped, and whether the fitted tail converges at all.
    """
    seconds: float
    raw_mean: float
    extrapolated: float
    growth_per_photon: float
    max_drawn: int
    summed_to: int
    converges: bool
    shots: int


def stratified_cost(photons, seconds, law: np.ndarray, min_count: int = 5, rel_tol: float = 1e-6,
                    max_N: int | None = None) -> CostEstimate:
    """E[seconds per shot] = sum_N law[N] t(N), t(N) the stratum mean where the stratum holds >= min_count
    shots, else the fit log t = a + b N over the shots in the upper half of the drawn N range (at least
    two distinct N). `law` must extend far enough for law[N] t(N) to become negligible (rel_tol of the
    sum); if the fitted tail does not converge within it, `converges` is False and the sum is a lower
    bound. `max_N`: sum only N <= max_N (the cost of a run of S samples, `run_cutoff`); the law is then NOT
    renormalised, so the result is E[cost 1{N <= max_N}].
    """
    N = np.asarray(photons, dtype=np.int64); t = np.asarray(seconds, dtype=np.float64)
    if N.size == 0:
        raise ValueError("no shots")
    raw = float(t.mean())
    ns = np.unique(N)
    lo = ns[len(ns) // 2] if len(ns) > 3 else ns[0]
    sel = lo <= N
    if len(np.unique(N[sel])) >= 2:
        b, a = np.polyfit(N[sel].astype(float), np.log(np.maximum(t[sel], 1e-9)), 1)
    else:                                                        # one stratum only: flat model at its mean
        a, b = math.log(max(t[sel].mean(), 1e-9)), 0.0
    b = max(b, 0.0)                                              # cost never falls with N
    counts = {int(k): int((k == N).sum()) for k in ns}
    means = {int(k): float(t[k == N].mean()) for k in ns}
    tot, ext, summed_to, conv = 0.0, 0.0, len(law) - 1, False
    top = len(law) if max_N is None else min(len(law), max_N + 1)
    for k in range(top):
        if law[k] == 0.0:
            continue
        measured = counts.get(k, 0) >= min_count
        tk = means[k] if measured else math.exp(a + b * k)
        term = law[k] * tk
        tot += term
        if not measured:
            ext += term
        if max_N is None and k > N.max() and term < rel_tol * tot:
            summed_to, conv = k, True
            break
    if max_N is not None:
        summed_to, conv = top - 1, True
    return CostEstimate(seconds=tot, raw_mean=raw, extrapolated=ext / tot if tot > 0 else 0.0,
                        growth_per_photon=math.exp(b), max_drawn=int(N.max()), summed_to=summed_to,
                        converges=conv, shots=int(N.size))


def tilted(state: GaussianInput, s: float, cap: float = 0.95) -> tuple[GaussianInput, float]:
    """The squeezing tilt of `importance_cost`: tanh r'_i = s tanh r_i, with s lowered if needed so that
    max tanh r' <= cap (at s tanh r >= 1 the collision-free cost-weighted law does not exist). Returns
    (tilted state, the s actually used).
    """
    if state.disp is not None and np.any(state.disp != 0):
        raise NotImplementedError("the tilt is exact for undisplaced inputs only")
    t = np.tanh(state.r)
    s_used = min(float(s), cap / float(t.max())) if t.max() > 0 else float(s)
    return GaussianInput(np.arctanh(s_used * t), state.modes, state.m), s_used


@dataclass
class ISEstimate:
    """Result of `importance_cost`: mean seconds per sample at the TARGET squeezing, its standard error,
    the effective sample size of the weights, the number of shots and the tilt s.
    """
    seconds: float
    stderr: float
    ess: float
    shots: int
    s: float


def importance_cost(photons, seconds, law_target: np.ndarray, law_tilted: np.ndarray, s: float) -> ISEstimate:
    """E_target[seconds] from shots drawn at the tilted squeezing: mean of seconds x w(N), w(N) =
    law_target[N] / law_tilted[N] (exact weights, see the module docstring); stderr from the sample
    variance of the weighted costs; ess = (sum w)^2 / sum w^2.
    """
    N = np.asarray(photons, dtype=np.int64); t = np.asarray(seconds, dtype=np.float64)
    if N.max() >= len(law_target) or N.max() >= len(law_tilted):
        raise ValueError("laws must extend past the largest photon number drawn")
    w = law_target[N] / law_tilted[N]
    x = t * w
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("inf")
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return ISEstimate(seconds=m, stderr=se, ess=ess, shots=len(x), s=s)


def mixture_cost(photons, seconds, phases, law_target: np.ndarray, laws: list[np.ndarray]) -> ISEstimate:
    """Multiple importance sampling (balance heuristic) over shots drawn at several squeezings: shot i
    came from law laws[phases[i]], a fraction alpha_p of the shots from each; the weight is
    law_target[N] / sum_p alpha_p laws[p][N]. With laws = [law_target] it is the plain mean; with the
    target and one tilt it is a DEFENSIVE importance sampler — never worse than the plain mean by more
    than 1 / alpha_target, and near zero-variance when the tilt matches the cost growth. `s` in the
    result is left at 0 (several tilts).
    """
    N = np.asarray(photons, dtype=np.int64); t = np.asarray(seconds, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.int64)
    alpha = np.array([(ph == p).mean() for p in range(len(laws))])
    if N.max() >= min(len(law_target), *(len(q) for q in laws)):
        raise ValueError("laws must extend past the largest photon number drawn")
    mix = sum(a * q[N] for a, q in zip(alpha, laws))
    w = law_target[N] / mix
    x = t * w
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("inf")
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return ISEstimate(seconds=m, stderr=se, ess=ess, shots=len(x), s=0.0)


def cost_growth(photons, seconds) -> float:
    """exp(b) of the fit log t = a + b N over the upper half of the drawn photon numbers (at least two
    distinct N; 1.0 otherwise, and never below 1): the per-photon cost growth that sets the tilt.
    """
    N = np.asarray(photons, dtype=np.int64); t = np.asarray(seconds, dtype=np.float64)
    ns = np.unique(N)
    if len(ns) < 2:
        return 1.0
    lo = ns[len(ns) // 2] if len(ns) > 3 else ns[0]
    sel = lo <= N
    if len(np.unique(N[sel])) < 2:
        sel = np.ones_like(N, dtype=bool)
    b = np.polyfit(N[sel].astype(float), np.log(np.maximum(t[sel], 1e-9)), 1)[0]
    return float(math.exp(max(b, 0.0)))


def run_cutoff(law: np.ndarray, samples: float) -> int:
    """N_S: the smallest N with P(N > N_S) <= 1 / samples — above it lie the shots a run of `samples`
    samples meets less than once on average. E[cost 1{N <= N_S}] is the mean cost per sample of such a
    run with its once-in-a-run excursions left out; the untruncated mean of GBS is dominated by shots of
    roughly four times the typical photon number (module docstring), which no finite run contains.
    """
    tail = 1.0 - np.cumsum(law)
    idx = np.nonzero(tail <= 1.0 / samples)[0]
    if idx.size == 0:
        raise ValueError("law too short for this many samples")
    return int(idx[0])
