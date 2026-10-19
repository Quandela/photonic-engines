"""A calibrated timing model of the exact GBS sampler, for the frontier (docs/GAUSSIAN.md §2.2).

The cost of a shot of `sampler.sample_pure` is a deterministic function of its photon pattern (in the
chain's mode order) and of the sampler settings (window, cpu_below, impl, cap): which loop-hafnian sieves
it calls, of what size and with how many loop vectors, does not depend on the matrix entries. The one
exception is the pairing that `hafnian.lhaf_poly_multi` searches from `_PAIR_SEARCH` pairs on, which can keep
or drop the collision discount; `plan_calls` reports the self-paired term count and the model prices both.
So

    cost(shot) = sum over the calls of `plan_calls(pattern)` of t_call(N, V, kind, collisions)
                 + sum over the steps of t_draw(N),

and E[cost | N] = the mean of that over the patterns of total N. `plan_calls` is tested to reproduce the
call sequence the sampler records (`sampler.Profile.events`) shot by shot; the per-call times come from a
calibration run on the target GPU (bench/gbs_timing_model.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import sampler as _smp


@dataclass(frozen=True)
class Call:
    """One loop-hafnian call of the chain: N vertices (occupied photons of the modes before the step), V loop
    vectors (1, or the window length), kind "none" / "plain" / "small" / "multi" as in `sampler.Profile`,
    device "cpu" or the sampler's device, `draws` = the steps this call priced (1 .. V), and `groups` = the
    photon counts >= 2 of the modes before the step (the collision groups of the self-pairing; each count t
    gives floor(t/2) identical self-pairs).
    """
    N: int
    V: int
    kind: str
    device: str
    draws: int
    groups: tuple


def plan_calls(pattern, window: int = 4, cpu_below: int = 14, impl: str = "v2", max_photons: int | None = None,
               device="cuda") -> list[Call]:
    """The sieve calls `sampler.sample_pure` makes for a shot whose drawn counts are `pattern` [m] (chain order).

    Mirrors the control flow of `sample_pure` step for step (the thresholds `sampler.SMALL_BELOW` and
    `sampler.V2_FROM` are read from the sampler); a censored shot (`max_photons`) stops where the sampler stops.
    """
    n = np.asarray(pattern, dtype=np.int64)
    m = n.size
    cap = math.inf if max_photons is None else max_photons
    out: list[Call] = []
    j, total = 0, 0
    while j < m:
        N = total
        small = impl == "v2" and 0 < N < min(_smp.SMALL_BELOW, cpu_below)
        plain = impl == "v1" or (N < _smp.V2_FROM and not small)
        grp = tuple(sorted(int(t) for t in n[:j] if t >= 2))
        dev = str(device) if cpu_below <= N else "cpu"          # as the sampler records it (device None -> "None")
        if N == 0 or plain or window <= 1:
            kind = "none" if N == 0 else "plain" if plain else "small" if small else "multi"
            out.append(Call(N, 1, kind, "cpu" if N == 0 else dev, 1, grp))
            total += int(n[j]); j += 1
            if total > cap:
                break
            continue
        J = min(m, j + window) - j
        used = J
        for v in range(J):
            if n[j + v] > 0:
                used = v + 1
                break
        out.append(Call(N, J, "small" if small else "multi", dev, used, grp))
        total += int(n[j:j + used].sum()); j += used
        if total > cap:
            break
    return out


def events_to_calls(events) -> list[tuple]:
    """The ("sieve", N, V, kind, device) part of `sampler.Profile.events`, for comparison with `plan_calls`."""
    return [(e[1], e[2], e[3], e[4]) for e in events if e[0] == "sieve"]


# ---- patterns of a given photon number -------------------------------------------------------
def surrogate_patterns(N: int, m: int, count: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """`count` patterns [count, m] of total N in a random mode order, for E[cost | N] where no measured shots exist.

    kind "free": N distinct modes (no collisions); "be": uniform over the multisets of size N on m modes
    (Bose-Einstein counting, the Haar-averaged law of Fock boson sampling). The mode order is uniform in both,
    which is exact for a Haar-random U (the Haar measure is invariant under output permutations); the collision
    statistics are the approximation, checked against measured patterns (bench/gbs_timing_model.py).
    """
    out = np.zeros((count, m), dtype=np.int64)
    for s in range(count):
        if kind == "free":
            if m < N:
                raise ValueError("free patterns need N <= m")
            out[s, rng.choice(m, size=N, replace=False)] = 1
        elif kind == "be":                                    # stars and bars: N + m - 1 slots, m - 1 bars
            bars = np.sort(rng.choice(N + m - 1, size=m - 1, replace=False))
            edges = np.concatenate([[-1], bars, [N + m - 1]])
            out[s] = np.diff(edges) - 1
        else:
            raise ValueError("kind must be 'free' or 'be'")
    return out


# ---- the per-call time -----------------------------------------------------------------------
def self_terms(N: int, groups) -> int:
    """Subset terms of the self-paired v2 sieve of N vertices with collision `groups` (counts >= 2): each count t
    gives r = floor(t/2) identical pairs enumerated by count (r + 1 terms) instead of 2^r subsets.
    """
    n = (N + 1) // 2
    rs = [t // 2 for t in groups if t // 2 >= 2]
    return math.prod(r + 1 for r in rs) * 2 ** (n - sum(rs))


def group_sizes(N: int, groups, pairing: str = "cross", dev_type: str = "cuda") -> list[int]:
    """Sizes r of the collision groups the v2 sieve enumerates by count for a call of N vertices whose repeated modes
    hold the photon counts `groups` (>= 2; the other vertices are distinct). "self": `hafnian._pairing` (v2.8, a
    count t gives floor(t/2) self-pairs, grouped from 2); "cross": the pairing `lhaf_poly_multi` takes below the
    search since 2026-09-27, `hafnian._pick_grouping` on the device `dev_type` (ungrouped, self or cross, by cost);
    "cross_all": the cross pairing whatever it costs. With `discount="self_all"` the chosen grouping is kept at every
    size (the counterfactual that ignores the round-off limit `hafnian._ROUND_TOL` above the search).
    """
    if pairing == "self":
        return [t // 2 for t in groups if t // 2 >= 2]
    from .hafnian import _pairing_cross, _pick_grouping
    counts = list(groups) + [1] * (N - sum(groups))
    lab = np.repeat(np.arange(len(counts)), counts)
    best = _pairing_cross(lab, N) if pairing == "cross_all" else _pick_grouping(lab, N, dev_type)
    return [r for _, r in best[1]]


@dataclass
class TimingModel:
    """Per-call and per-step times of the sampler on one machine, from a calibration run.

    `multi[(N, V)]`: seconds of a collision-free `lhaf_poly_multi` call (all 2^n terms) at N vertices, V vectors,
    measured (N <= `multi_max` in the calibration, plus any whole calls timed later), extrapolated above by `rate`
    (see `t_multi`). `small[(N, V)]` / `plain[N]`: the
    numpy and torch v1 calls. `draw[N]` / `prep[N]`: one normaliser + draw, and the host preparation of one call,
    at N (host; the last calibrated size is used above the data).
    `rate[(n, V)]`: seconds per subset term of the v2 sieve pipeline at n pairs (bench/gbs_timing_model.py
    measures it on random subsets, so it reaches n where a whole sieve is too long to time); `rate_power`: see
    `rate_at`.
    `shot[m]`: the rest of a shot on m modes (heterodyne, bookkeeping; measured shot time minus its events).
    `discount`: "none" prices every multi call at 2^n terms; "self" prices a collided call below the pairing-search
    threshold (where the sieve keeps the self-pairing) as prod(r + 1) free calls on the ungrouped pairs; "self_all"
    does so at every size (the collision-discounted term count v2.8 does not implement above `search_from`; used to
    price device samples, not the frontier). `pairing`: which grouping prices a collided call, "cross" (default, the
    sieve since 2026-09-27: `group_sizes`) or "self" (v2.8; what the frontier grid of 2026-09-25 used). Above
    `search_from` the sieve may now keep a grouped pairing when its predicted round-off allows (`hafnian._ROUND_TOL`);
    the model prices those calls ungrouped, an upper bound.
    `tiers` (2026-09-28, v2.11b): price a multi call as the sieve with precision "auto" can run it (`_tiered`): the
    cheapest of fp64 / double-double / quad-double, ungrouped / cross-grouped (Bulmer's term count), among the options
    whose round-off eps kappa fits `tol` (the error is bounded by eps kappa: at most 0.74 eps kappa on 275 sieves against
    dd, job 27). kappa(N) = 10^(kappa_fit[0] + kappa_fit[1] N) (H100 fit, heavy patterns), plus `cross_penalty` decades
    for the cross grouping (+4.6..+5.2 measured at N = 36-48, job 25). An option that needs quad-double (ungrouped past
    N ~ 141, grouped past N ~ 117) is priced at `qd_factor` x dd (1 by default: a lower bound, no quad-double kernel
    exists); `arith_max` = "dd" restricts the options to fp64 and dd (the sieve as built: no qd kernel), a call that
    fits neither then priced ungrouped in dd, a lower bound. `tier_choice` "cost" takes the cheapest admissible option
    (dd with the grouping may beat fp64 without it); "needed" the cheapest in the lowest arithmetic that admits any
    (the precision cost alone). dd time = `dd_factor` x fp64 time: `dd_rate[(n, V)]` the
    measured dd per-term rate (job 27, n <= 32), `dd_ratio_beyond[V]` dd / fp64 past it (held at its value at the last
    measured n against the torch fp64 rate the model uses there: an assumption). Off: v2.8-v2.11 pricing, all fp64,
    grouping regardless of round-off (every number before 2026-09-28).
    """
    multi: dict
    small: dict
    plain: dict
    draw: dict
    prep: dict
    rate: dict
    multi_max: int
    shot: dict | None = None
    search_from: int = 18
    discount: str = "self"
    rate_power: float | None = None
    pairing: str = "cross"
    tiers: bool = False
    tol: float = 1e-3
    kappa_fit: tuple = (0.40, 0.197)
    dd_rate: dict | None = None
    dd_ratio_beyond: dict | None = None
    cross_penalty: float = 5.0
    qd_factor: float = 1.0
    arith_max: str = "qd"
    tier_choice: str = "cost"
    last_arith: str = "fp64"

    def tier(self, N: int) -> str:
        """The arithmetic a sieve of N vertices needs for round-off eps kappa(N) <= tol: "fp64", "dd" or "qd"."""
        lk = self.kappa_fit[0] + self.kappa_fit[1] * N
        if self._fits(lk, 2.0 ** -52):
            return "fp64"
        return "dd" if self._fits(lk, 2.0 ** -104) else "qd"

    def dd_factor(self, n: int, V: int) -> float:
        """Double-double / fp64 time of a term at n pairs: measured to the largest dd n (against `rate_at`), then `dd_ratio_beyond`."""
        ks = sorted(k[0] for k in self.dd_rate if k[1] == V)
        if n <= ks[-1]:
            return self._interp(self.dd_rate, n, V) / self.rate_at(n, V)
        return self.dd_ratio_beyond[V]

    def _fits(self, lk: float, eps: float) -> bool:
        return lk + math.log10(eps) <= math.log10(self.tol)

    def _tiered(self, c, n: int, s_fp64: float) -> float:
        """A multi call as the sieve with precision "auto" can run it: the cheapest (pairing, arithmetic) whose round-off
        eps kappa fits `tol` - fp64, dd or qd, ungrouped or cross-grouped (kappa + `cross_penalty` decades). qd is
        priced at `qd_factor` x the dd time (1: a lower bound, no quad-double kernel exists). `s_fp64` (the untiered
        price) is kept only when its pairing fits fp64. The arithmetic of the chosen option is left in `last_arith`.
        """
        lk = self.kappa_fit[0] + self.kappa_fit[1] * c.N
        rc = group_sizes(c.N, c.groups, "cross", c.device) if c.groups else []
        t_un = self.t_multi(c.N, c.V)
        t_gr = self._grouped(rc, n, c.V) if rc else math.inf
        if rc:
            # a grouped term is a term of the full n (its subset matrices carry the count-vector prefix): at least
            # terms x rate(n). _grouped prices each prefix length as a free sieve of fewer vertices, at that smaller
            # sieve's per-term rate: x5-10 low on job 31's dd-grouped calls (N = 53-61, 2^16-2^20 terms)
            terms = math.prod(r + 1 for r in rc) * 2.0 ** (n - sum(rc))
            t_gr = max(t_gr, terms * self.rate_at(n, c.V))
        f = self.dd_factor(n, min(c.V, 4))
        opts = []
        ariths = (("fp64", 2.0 ** -52, 1.0), ("dd", 2.0 ** -104, f), ("qd", 2.0 ** -208, f * self.qd_factor))
        for arith, eps, x in ariths[:1 + ("fp64", "dd", "qd").index(self.arith_max)]:
            if self._fits(lk, eps):
                opts.append((x * t_un, arith))
            if rc and self._fits(lk + self.cross_penalty, eps):
                opts.append((x * (min(t_gr, s_fp64) if arith == "fp64" else t_gr), arith))
        if self.tier_choice == "needed" and opts:          # the lowest arithmetic with an admissible option only
            low = min(("fp64", "dd", "qd").index(a) for _, a in opts)
            opts = [o for o in opts if o[1] == ("fp64", "dd", "qd")[low]]
        if not opts:                                        # past the best arithmetic: ungrouped in it, a lower bound
            opts.append(({a: x for a, _, x in ariths}[self.arith_max] * t_un, ">" + self.arith_max))
        t, self.last_arith = min(opts)
        return t

    def _grouped(self, rs, n: int, V: int) -> float:
        from .hafnian import _prefix_counts
        n_f = n - sum(rs)
        s = 0.0
        for cnt in _prefix_counts([(0, r) for r in rs]):
            if cnt:
                Ne = max(2 * n_f + 2 * math.log2(cnt), 14.0)
                lo = 2 * math.floor(Ne / 2); f = (Ne - lo) / 2
                s += math.exp((1 - f) * math.log(self.t_multi(lo, V)) + f * math.log(self.t_multi(lo + 2, V)))
        return s

    def _interp(self, table: dict, key_N: int, V: int | None = None) -> float:
        keys = sorted(k for k in (table if V is None else [k[0] for k in table if k[1] == V]))
        if not keys:
            raise KeyError("empty calibration table")
        get = (lambda x: table[x]) if V is None else (lambda x: table[(x, V)])
        if key_N <= keys[0]:
            return get(keys[0])
        if key_N >= keys[-1]:
            return get(keys[-1])
        hi = next(k for k in keys if k >= key_N); lo = max(k for k in keys if k <= key_N)
        if hi == lo:
            return get(lo)
        f = (key_N - lo) / (hi - lo)                          # log-linear between calibrated sizes
        return math.exp((1 - f) * math.log(max(get(lo), 1e-15)) + f * math.log(max(get(hi), 1e-15)))

    def t_multi(self, N: int, V: int) -> float:
        """A collision-free multi call. Measured sizes are used as they are; an odd N takes the measured N + 1 (same
        pair count n = ceil(N/2), measured equal to 1e-3 at N = 15..47); between measured sizes, log-linear. Above the
        largest measured N0: t(N) = t(N0) 2^(n - n0) rate(n, V) / rate(n0, V), the term count times the per-term rate.
        """
        if (N, V) in self.multi:
            return self.multi[(N, V)]
        if N % 2 and (N + 1, V) in self.multi:
            return self.multi[(N + 1, V)]
        N0 = max(k[0] for k in self.multi if k[1] == V)
        if N0 >= N:
            return self._interp(self.multi, N, V)
        n, n0 = (N + 1) // 2, (N0 + 1) // 2
        return self.multi[(N0, V)] * 2.0 ** (n - n0) * self.rate_at(n, V) / self.rate_at(n0, V)

    def rate_at(self, n: int, V: int) -> float:
        """Seconds per term at n pairs. Past the calibrated pairs: log-linear in n from the last two points (default,
        what the frontier grid of 2026-09-25 used), or, with `rate_power` = a, rate(n_max) (n / n_max)^a — the flop
        scaling of the power route (n products of padded q x q matrices per term, q ~ n: a = 4) for extrapolations far
        past the data (device sizes), where two noisy points give no usable slope.
        """
        ks = sorted(k[0] for k in self.rate if k[1] == V)
        if n <= ks[-1]:
            return self._interp(self.rate, n, V)
        if self.rate_power is not None:
            return self.rate[(ks[-1], V)] * (n / ks[-1]) ** self.rate_power
        a, b = ks[-2], ks[-1]
        slope = (math.log(self.rate[(b, V)]) - math.log(self.rate[(a, V)])) / (b - a)
        return self.rate[(b, V)] * math.exp(slope * (n - b))

    def t_call(self, c: Call) -> float:
        """Seconds of one call, sieve plus host preparation (not the draws)."""
        if c.kind == "none":
            s = 0.0
        elif c.kind == "small":
            s = self._interp(self.small, c.N, min(c.V, max(k[1] for k in self.small)))
        elif c.kind == "plain":
            s = self._interp(self.plain, c.N)
        else:
            n = (c.N + 1) // 2
            rs = group_sizes(c.N, c.groups, self.pairing, c.device) if c.groups else []
            if rs and (self.discount == "self_all" or (self.discount == "self" and n < self.search_from)):
                # the grouped sieve runs one free sieve of the n_f = n - sum r ungrouped pairs per count vector
                # (lhaf_poly_multi), each with its own launches: prod(r + 1) x t_free(2 n_f). Scaling one call by
                # terms / 2^n instead priced the m = n shots of job 22 at x0.70 (the fixed cost is per count vector).
                # "self_all" keeps the grouping at every n (Bulmer et al.'s prod(eta_i + 1) term count), which v2.8
                # does not do above the pairing search: a counterfactual kernel, priced for device samples only.
                # since 2026-09-27 the count vectors of one prefix length L share the calls (`hafnian._sieve_multi`):
                # each L is priced as one free sieve of (count vectors) x 2^n_f terms, i.e. of 2 n_f + 2 log2(count)
                # vertices, log-linear between even sizes
                s = self._grouped(rs, n, c.V)
            else:
                s = self.t_multi(c.N, c.V)
            if self.tiers:
                s = self._tiered(c, n, s)
        return s + self._interp(self.prep, c.N)

    def shot_time(self, pattern, **plan) -> float:
        """Predicted seconds of one shot with this pattern (chain order), `plan` as for `plan_calls`."""
        t = self._interp(self.shot, len(pattern)) if self.shot else 0.0
        for c in plan_calls(pattern, **plan):
            t += self.t_call(c) + c.draws * self._interp(self.draw, c.N)
        return t


# ---- E[cost | N] and the run-of-S cost ----------------------------------------------------------
def conditional_cost(model: TimingModel, N: int, m: int, patterns=None, kind: str = "be", count: int = 256,
                     seed: int = 0, **plan) -> float:
    """E[cost | N]: the mean `shot_time` over `patterns` [k, m] of total N (e.g. measured ones), or over `count`
    `surrogate_patterns(N, m, kind)`. `plan` as for `plan_calls` (the cap is left out: the frontier statistic
    prices whole shots).
    """
    if patterns is None:
        patterns = surrogate_patterns(N, m, count, kind, np.random.default_rng(seed + 7919 * N))
    return float(np.mean([model.shot_time(p, **plan) for p in patterns]))


def run_cost(law: np.ndarray, cond, S: float) -> tuple[float, int]:
    """(t_S, N_S): t_S = sum_(N <= N_S) P(N) E[cost | N] with N_S = `cost.run_cutoff(law, S)` (docs/GAUSSIAN.md §2.1);
    `cond` maps N to E[cost | N] (a callable).
    """
    from .cost import run_cutoff
    NS = run_cutoff(law, S)
    return float(sum(law[N] * cond(N) for N in range(NS + 1) if law[N] > 0)), NS


def run_weights(laws, S: float) -> tuple[np.ndarray, int]:
    """(w, N_S) with t_S = sum_a w[a] E[cost | a] for a shot made of independent instances (Schmidt modes).

    `laws` = the detected photon-number law of each instance (one law: an ordinary shot). A shot of K instances
    is K independent chains through the same U, so its cost is sum_k cost_k(N_k), and E[cost | N_k = a] is the same
    function of a for every instance (P(pattern | N) does not depend on the squeezing, `cost` module docstring).
    With N = sum_k N_k and N_S = `cost.run_cutoff` of the law of N,
        t_S = E[sum_k cost_k 1{N <= N_S}] = sum_k sum_a P_k(a) E[cost | a] P(N - N_k <= N_S - a),
    so w[a] = sum_k P_k(a) P(sum_(j != k) N_j <= N_S - a). One instance: w[a] = P(a) for a <= N_S.
    """
    from .cost import run_cutoff
    laws = [np.asarray(p, dtype=np.float64) for p in laws]
    L = min(len(p) for p in laws)
    laws = [p[:L] for p in laws]
    total = np.zeros(L); total[0] = 1.0
    for p in laws:
        total = np.convolve(total, p)[:L]
    NS = run_cutoff(total, S)
    w = np.zeros(NS + 1)
    for k, pk in enumerate(laws):
        rest = np.zeros(L); rest[0] = 1.0
        for j, p in enumerate(laws):
            if j != k:
                rest = np.convolve(rest, p)[:L]
        cdf = np.cumsum(rest)
        for a in range(NS + 1):
            w[a] += pk[a] * cdf[NS - a]
    return w, NS
