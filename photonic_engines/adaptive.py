"""Adaptive linear optics on a Fock input: mid-circuit photon counting with feed-forward on the interferometer.

The model of Chabaud, Markham & Sohbi (arXiv:2102.04579); notation of docs/ADAPTIVE.md §1:

    n, m, t        photons, modes, input modes (one photon per input mode)
    levels         Lambda lists of output modes, measured in this order; level lam has l_lam modes,
                   K_lam = l_1 + ... + l_lam measured so far, K = K_Lambda adaptive modes in all
    p              the K photon counts read on the adaptive modes (PNR), r = |p|; p_{<=lam} its prefix
    V_lam(p_{<=lam})  feed-forward unitary on the m - K_lam modes still unmeasured after level lam
    U^{p_{<=lam}}  = E(V_lam) ... E(V_1) U_0, the cumulative m x m unitary (E embeds with the identity
                   on the measured modes)
    s              the counts on the m - K final modes, n - r photons; omega = (p, s) the full occupation

One permanent per outcome: <omega|...|t> = Per(U^p[omega, t]) / sqrt(omega!). Three answers:

  strong   `AdaptiveCircuit.probs`: the joint distribution of omega, one SLOS pass per branch p.
  marginal `mode_marginals` (Gurvits' second algorithm: the permanent of identity + rank L, exact and
           polynomial for a fixed number L of modes) and, on top of it, `outcome_probs` (the law of
           p_{<=lam}) and `final_marginal` (counts on a few final modes, summed over the branches).
  weak     `AdaptiveCircuit.sample`: exact samples of omega. The adaptive outcomes are a chain over
           LEVELS - P(p_{<=lam}) does not depend on V_lam, V_{lam+1}, ... because those act on other
           modes - drawn from the marginals (or by rejection); the final stage s | p runs the Clifford &
           Clifford chain on U^p with its first r photons FORCED onto the adaptive modes and rejects only
           when another photon joins them: acceptance P(p) / T[p], E[2^r] chains per sample on average
           instead of one per branch (docs/ADAPTIVE.md §4b).

Restrictions: one photon per input mode, photon-number-resolving detectors. Uniform loss and partial
distinguishability on the sampler only (`sample(loss=, indist=)`, docs/ADAPTIVE.md §7).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import product

import numpy as np
import torch

MAX_MARGINAL_ENTRIES = 1 << 28        # graded table of `binomial_moments`, sum_d C(d+L-1, L-1)^2 complex128: 4 GB


def _binomial_inverse(n: int, device) -> torch.Tensor:
    """[n+1, n+1] float64, M[q, c] = (-1)^(c-q) C(c, q): binomial moments E[C(N, c)] -> P(N = q)."""
    M = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    for q in range(n + 1):
        for c in range(q, n + 1):
            M[q, c] = (-1.0) ** (c - q) * math.comb(c, q)
    return M.to(device)


@lru_cache(maxsize=8)
def _graded_index(n: int, L: int):
    """Monomials of L variables by total degree d = 0..n: (exps, up) with exps[d] int64 [M_d, L] the exponent
    vectors of degree d (M_d = C(d+L-1, L-1), lexicographic) and up[d] int64 [L, M_d] the position in layer
    d + 1 of each of them times x_l. Host tensors, cached per (n, L).
    """
    exps, up = [], []
    layers = [[e for e in product(range(d + 1), repeat=L) if sum(e) == d] if L <= 1 or d == 0 else None for d in range(n + 1)]
    for d in range(n + 1):
        if layers[d] is None:                                                       # compositions of d into L parts
            prev = layers[d - 1]
            layers[d] = sorted({tuple(e[k] + (k == l) for k in range(L)) for e in prev for l in range(L)})
    pos = [{e: k for k, e in enumerate(lay)} for lay in layers]
    for d in range(n + 1):
        exps.append(torch.tensor(layers[d], dtype=torch.int64).reshape(-1, L))
        if d < n:
            up.append(torch.tensor([[pos[d + 1][tuple(e[k] + (k == l) for k in range(L))] for e in layers[d]]
                                    for l in range(L)], dtype=torch.int64).reshape(L, -1))
    return exps, up


def graded_entries(n: int, L: int) -> int:
    """Entries of the graded table of `binomial_moments`: sum over d of C(d+L-1, L-1)^2 (dense would be (n+1)^(2L))."""
    return sum(math.comb(d + L - 1, L - 1) ** 2 for d in range(n + 1))


def binomial_moments(U: torch.Tensor, in_modes: Sequence[int], modes: Sequence[int]) -> torch.Tensor:
    """T[c_1..c_L] = E[prod_l C(N_l, c_l)], N_l the photon count of output mode modes[l]; real [(n+1)]*L.

    The generating function E[prod z_l^N_l] is Per(I + V^dag (Z - 1) V), V = U[modes, t] ([L, n]), a
    rank-L perturbation of the identity, and its expansion in y = z - 1 has the coefficients
        T[c] = c! [x^c w^c] prod_i (1 + (sum_l conj(V_li) x_l) (sum_l V_li w_l)) .
    Every factor adds one x and one w, so only monomials x^a w^b with |a| = |b| occur: the table is GRADED,
    one [M_d, M_d] block per total degree d (M_d = C(d+L-1, L-1) monomials), and a photon maps block d into
    block d + 1 by X_d T_d Y_d^T with X_d, Y_d the sparse "times a linear form" maps. sum_d M_d^2 entries,
    ~ n^(2L-1) / ((2L-1) (L-1)!^2), against (n+1)^(2L) for a dense table: 4e6 against 3.5e9 at n = 38, L = 3
    (`graded_entries`; MAX_MARGINAL_ENTRIES guards it). Entries with |c| > n are zero. One instance;
    `binomial_moments_blocks` is the batched form.
    """
    n, L = len(in_modes), len(modes)
    if len(set(in_modes)) != n:
        raise ValueError("mode marginals: one photon per input mode only (the identity + low-rank form needs it)")
    if len(set(modes)) != L:
        raise ValueError("mode marginals: repeated output mode")
    if L == 0:
        return torch.ones((), dtype=torch.float64, device=U.device)
    V = U.to(torch.complex128)[list(modes)][:, list(in_modes)]                       # [L, n]
    return binomial_moments_blocks(V[None])[0]


def binomial_moments_blocks(V: torch.Tensor, max_entries: int | None = None) -> torch.Tensor:
    """The tables of `binomial_moments` for B blocks V [B, L, n] at once: real [B, (n+1)]*L.

    V[b] = U_b[modes_b, t_b] - each block may come from a different unitary, different output modes and
    different input columns, as the instances of one level of `AdaptiveCircuit._sample_prefix` do under
    imperfections (every stream its own species columns). A block with fewer photons is padded with zero
    columns: a zero column is the factor (1 + 0) and leaves the table of its block unchanged, so one
    call serves mixed photon numbers exactly. The graded recursion runs on [B, M_d, M_d] blocks - the same
    index_add launches as for one instance, B times the memory - in chunks of at most `max_entries`
    (default MAX_MARGINAL_ENTRIES) graded entries per chunk.
    """
    B, L, n = V.shape
    dev = V.device
    if L == 0:
        return torch.ones((B,), dtype=torch.float64, device=dev)
    ent = graded_entries(n, L)
    cap = MAX_MARGINAL_ENTRIES if max_entries is None else max_entries
    if ent > cap:
        raise MemoryError(f"mode marginals: {ent:.3g} graded table entries for n = {n}, L = {L}")
    chunk = max(1, cap // ent)
    if chunk < B:
        return torch.cat([binomial_moments_blocks(V[c0:c0 + chunk], max_entries) for c0 in range(0, B, chunk)])
    V = V.to(torch.complex128)
    exps, up = _graded_index(n, L)
    up = [u.to(dev) for u in up]
    T = [torch.ones(B, 1, 1, dtype=torch.complex128, device=dev)]                    # block d: [B, M_d, M_d], rows x, cols w
    for i in range(n):
        T.append(torch.zeros(B, exps[i + 1].shape[0], exps[i + 1].shape[0], dtype=torch.complex128, device=dev))
        for d in range(i, -1, -1):                                                  # descending: block d is still the old one
            rows = torch.zeros(B, T[d + 1].shape[1], T[d].shape[2], dtype=torch.complex128, device=dev)
            for l in range(L):
                rows.index_add_(1, up[d][l], V[:, l, i].conj()[:, None, None] * T[d])   # x_l (conj(V_li))
            for l in range(L):
                T[d + 1].index_add_(2, up[d][l], V[:, l, i][:, None, None] * rows)      # w_l (V_li)
    out = torch.zeros((B,) + (n + 1,) * L, dtype=torch.float64, device=dev)
    fact = torch.tensor([math.factorial(c) for c in range(n + 1)], dtype=torch.float64, device=dev)
    for d in range(n + 1):
        e = exps[d].to(dev)                                                          # [M_d, L]
        out[(slice(None), *tuple(e.T))] = torch.diagonal(T[d], dim1=1, dim2=2).real * fact[e].prod(1)
    return out


def mode_marginals(U: torch.Tensor, in_modes: Sequence[int], modes: Sequence[int]) -> torch.Tensor:
    """Exact joint distribution of the photon counts on `modes`, every other mode traced: float64 [(n+1)]*L.

    Entry [q_1..q_L] is P(N_modes[0] = q_1, ..., N_modes[L-1] = q_L) for one photon in each of
    `in_modes` through U (rows = outputs, columns = inputs). Binomial inversion of `binomial_moments`
    along every axis. The inversion is an alternating sum: it is well conditioned while the moments
    decay (mean occupation per mode below ~1, the case of a spread-out interferometer) and loses
    digits when many photons pile into the measured modes; a result below -1e-9 raises.
    """
    return marginals_from_moments(binomial_moments(U, in_modes, modes))


def marginals_from_moments(T: torch.Tensor) -> torch.Tensor:
    """Binomial inversion of a `binomial_moments` table along every axis: the count distribution, same shape."""
    P, L = T, T.dim()
    if L == 0:
        return P
    Minv = _binomial_inverse(P.shape[0] - 1, P.device)
    for l in range(L):
        P = torch.movedim(torch.tensordot(Minv, torch.movedim(P, l, 0), dims=([1], [0])), 0, l)
    if L and float(P.min()) < -1e-9:
        raise FloatingPointError(f"mode marginals: binomial inversion lost precision (min {float(P.min()):.2e})")
    return P.clamp_min(0.0)


class AdaptiveCircuit:
    """U_0, then Lambda levels of photon counting, each followed by a feed-forward unitary on the unmeasured modes.

    U0        [m, m] unitary (rows = outputs, columns = inputs)
    in_modes  the n input modes t, one photon each
    levels    list of lists of output modes; level lam (1-based) measures levels[lam-1]. A flat list
              of modes is one mode per level.
    policy    policy(lam, p) -> [m - K_lam, m - K_lam] unitary V_lam on `rest(lam)` (the unmeasured
              modes in increasing order) or None for the identity; p is the tuple of ALL K_lam counts
              read so far, in measurement order. Must be deterministic: it is called once per
              distinct p and cached.
    Everything is complex128 on U0's device.
    """

    def __init__(self, U0: torch.Tensor, in_modes: Sequence[int], levels: Sequence, policy: Callable | None = None):
        self.U0 = U0.to(torch.complex128)
        self.m = int(U0.shape[0])
        self.in_modes = [int(j) for j in in_modes]
        self.n = len(self.in_modes)
        self.levels = [[int(a)] if np.isscalar(a) else [int(x) for x in a] for a in levels]
        self.adaptive_modes = [a for lev in self.levels for a in lev]                # a_1..a_K
        self.K = len(self.adaptive_modes)
        if len(set(self.adaptive_modes)) != self.K or any(not 0 <= a < self.m for a in self.adaptive_modes):
            raise ValueError("adaptive modes must be distinct modes of the circuit")
        if any(len(lev) == 0 for lev in self.levels):
            raise ValueError("empty level")
        if len(set(self.in_modes)) != self.n:
            raise ValueError("one photon per input mode only")
        self.K_upto = [0] + [int(x) for x in np.cumsum([len(lev) for lev in self.levels])]       # K_0 = 0, K_1, ..., K_Lambda
        self.policy = policy
        self._cache: dict = {(): self.U0}
        self._tables: dict = {}
        self._kernel = None
        self.stats: dict = {}

    @property
    def n_levels(self) -> int:
        """Lambda, the number of measurement levels."""
        return len(self.levels)

    def rest(self, lam: int) -> list[int]:
        """The m - K_lam modes still unmeasured after level lam, increasing; V_lam acts on them in this order."""
        done = set(self.adaptive_modes[:self.K_upto[lam]])
        return [j for j in range(self.m) if j not in done]

    def _level_of(self, p: Sequence[int]) -> int:
        if len(p) not in self.K_upto:
            raise ValueError(f"outcome prefix of length {len(p)}: not a whole number of levels {self.K_upto}")
        return self.K_upto.index(len(p))

    def unitary(self, p: Sequence[int] = ()) -> torch.Tensor:
        """U^{p}: the cumulative [m, m] unitary after the feed-forward of the last level in `p` (a whole-level prefix)."""
        p = tuple(int(x) for x in p)
        if p in self._cache:
            return self._cache[p]
        lam = self._level_of(p)
        U = self.unitary(p[:self.K_upto[lam - 1]])
        V = self.policy(lam, p) if self.policy is not None else None
        if V is not None:
            rest = self.rest(lam)
            V = torch.as_tensor(V).to(torch.complex128).to(U.device)
            if V.shape != (len(rest), len(rest)):
                raise ValueError(f"policy({lam}, {p}) returned {tuple(V.shape)}, expected {(len(rest), len(rest))}")
            eye = torch.eye(len(rest), dtype=torch.complex128, device=U.device)
            if float((V.conj().T @ V - eye).abs().max()) > 1e-8:
                raise ValueError(f"policy({lam}, {p}) is not unitary")
            U = U.clone()
            U[rest] = V @ U[rest]                                                    # E(V_lam) U: rows of the rest
        self._cache[p] = U
        return U

    # ---- marginals ---------------------------------------------------------------------------

    def level_probs(self, lam: int, prefix: Sequence[int]) -> torch.Tensor:
        """Unnormalised law of the level-lam counts given the earlier ones: [(n+1)]*l_lam, entry = P(p_{<=lam}).

        One `mode_marginals` on the K_lam modes measured so far, under U^{prefix} (V_lam and later
        do not touch them), sliced at `prefix`. Its sum is P(prefix).
        """
        prefix = tuple(int(x) for x in prefix)
        if len(prefix) != self.K_upto[lam - 1]:
            raise ValueError(f"level {lam} needs a prefix of {self.K_upto[lam - 1]} counts")
        return self._level_tables(lam, prefix)[0]

    def _level_tables(self, lam: int, prefix: tuple, cols: tuple | None = None,
                      inst: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(P, T) of level lam sliced at `inst`, both [(n'+1)]*l_lam: P(p_{<=lam}) and the binomial moment
        T[p_{<=lam}] = E[prod_l C(N_l, p_l)] that the forced-prefix acceptance P / T needs, for the n'
        photons of the input columns `cols` (default: all of `in_modes`) under U^{prefix}. `inst` (default
        `prefix`) is the prefix of THIS instance's counts: with imperfections a stream is several
        interfering instances whose counts add up to `prefix`. The slices are cached per
        (lam, prefix, cols, inst); the full table is not kept (1e8 entries at n = 38, K = 4).
        """
        cols = tuple(self.in_modes) if cols is None else tuple(cols)
        inst = prefix if inst is None else tuple(int(x) for x in inst)
        key = (lam, prefix, cols, inst)
        if key not in self._tables:
            modes = self.adaptive_modes[:self.K_upto[lam]]
            U = self.unitary(prefix)
            T = binomial_moments(U, list(cols), modes)                               # one table serves both
            P = marginals_from_moments(T)
            self._tables[key] = (P[inst].clone(), T[inst].clone()) if inst else (P, T)
        return self._tables[key]

    def _level_tables_many(self, lam: int, keys: Sequence[tuple]) -> None:
        """Fill the `_level_tables` cache for many (prefix, cols, inst) keys of level lam in one batched call.

        One `binomial_moments_blocks` on the stacked blocks U^{prefix}[modes, cols] (padded with zero columns
        to the largest photon number: exact, see there) instead of one table per key - what makes the
        per-stream tables of the imperfect sampler affordable (every shot its own species columns) and
        removes the launch overhead per branch of the lossless one.
        """
        modes = self.adaptive_modes[:self.K_upto[lam]]
        todo = []
        for prefix, cols, inst in keys:
            cols = tuple(cols); inst = tuple(int(x) for x in inst)
            if (lam, prefix, cols, inst) not in self._tables and (prefix, cols, inst) not in todo:
                todo.append((prefix, cols, inst))
        if not todo:
            return
        n_max = max(len(cols) for _, cols, _ in todo)
        V = torch.zeros(len(todo), len(modes), n_max, dtype=torch.complex128, device=self.U0.device)
        for b, (prefix, cols, _) in enumerate(todo):
            V[b, :, :len(cols)] = self.unitary(prefix)[modes][:, list(cols)]
        T = binomial_moments_blocks(V)                                              # [B, (n_max+1)]*l
        for b, (prefix, cols, inst) in enumerate(todo):
            P = marginals_from_moments(T[b])
            self._tables[(lam, prefix, cols, inst)] = (P[inst].clone(), T[b][inst].clone()) if inst else (P, T[b])

    def outcome_probs(self, lam: int | None = None, tol: float = 0.0) -> dict:
        """{p_{<=lam}: P(p_{<=lam})} over the adaptive outcomes with probability > tol (lam = Lambda by default).

        Walks the branch tree level by level with `level_probs`; `tol` prunes it.
        """
        lam = self.n_levels if lam is None else lam
        out = {(): 1.0}
        for l in range(1, lam + 1):
            nxt = {}
            for prefix in out:
                law = self.level_probs(l, prefix)
                for idx in torch.nonzero(law > tol).tolist():
                    if sum(prefix) + sum(idx) <= self.n:
                        nxt[prefix + tuple(idx)] = float(law[tuple(idx)])
            out = nxt
        return out

    def final_marginal(self, modes: Sequence[int], tol: float = 0.0) -> torch.Tensor:
        """Exact distribution of the counts on a few FINAL modes, summed over the adaptive outcomes: [(n+1)]*len(modes).

        sum_p P(p, N_modes = q): one (K + len(modes))-mode marginal under U^p per branch p - the
        probability-estimation task of arXiv:2102.04579, polynomial for K + len(modes) = O(1).
        `tol` drops the branches with P(p) <= tol (the result then sums to 1 minus the dropped mass).
        """
        modes = [int(j) for j in modes]
        if set(modes) & set(self.adaptive_modes):
            raise ValueError("final_marginal: a requested mode is an adaptive mode")
        out = torch.zeros((self.n + 1,) * len(modes), dtype=torch.float64, device=self.U0.device)
        for p in self.outcome_probs(tol=tol):
            out += mode_marginals(self.unitary(p), self.in_modes, self.adaptive_modes + modes)[p]
        return out

    # ---- strong ------------------------------------------------------------------------------

    def _slos(self):
        if self._kernel is None:
            from .slos import FockSLOSKernel
            occ = [0] * self.m
            for j in self.in_modes:
                occ[j] += 1
            self._kernel = FockSLOSKernel(self.n, self.m, occ, device=str(self.U0.device), dtype=torch.complex128)
            self._occ = torch.as_tensor(self._kernel.layers.final_occupations(), device=self.U0.device)   # [S, m]
        return self._kernel

    def occupations(self) -> torch.Tensor:
        """The output basis of `probs`: int64 [S, m] occupations omega (the SLOS final basis, adaptive modes included)."""
        self._slos()
        return self._occ

    def probs(self, chunk: int = 64) -> torch.Tensor:
        """Joint distribution of omega = (p, s) over `occupations()`: float64 [S], sums to 1.

        Entry omega is |Per(U^{p(omega)}[omega, t])|^2 / omega!: one batched SLOS pass per distinct
        adaptive outcome p (C(n + K, K) of them), each contributing the basis states that show p on
        the adaptive modes. Strong simulation: stops where SLOS stops (memory C(n + m - 1, n)).
        """
        kernel = self._slos()
        pa = self._occ[:, self.adaptive_modes]                                       # [S, K]
        branches, inv = torch.unique(pa, dim=0, return_inverse=True)                 # [Bp, K], [S]
        out = torch.zeros(self._occ.shape[0], dtype=torch.float64, device=self.U0.device)
        for c0 in range(0, branches.shape[0], chunk):
            Us = torch.stack([self.unitary(tuple(b.tolist())) for b in branches[c0:c0 + chunk]])
            pr = kernel.probs(Us).to(torch.float64)                                  # [B, S]
            for b in range(Us.shape[0]):
                sel = inv == c0 + b
                out[sel] = pr[b, sel]
        return out

    # ---- weak --------------------------------------------------------------------------------

    def _base_sampler(self, backend: str) -> Callable:
        def base(U, in_modes, shots, generator):
            from .clifford import fock_sample
            return fock_sample(U, in_modes, shots, generator=generator, backend=backend)
        return base

    def _prefix_sampler(self, backend: str) -> Callable:
        def base(U, in_modes, shots, generator, prefix, avoid):
            from .clifford import fock_sample
            return fock_sample(U, in_modes, shots, generator=generator, backend=backend, prefix=prefix, avoid=avoid,
                               keep_ids=isinstance(shots, torch.Tensor))
        return base

    def _forced(self, items: list, avoid: Sequence[int], base: Callable, generator, max_batch: int, max_draws: int,
                first_reps: int = 2) -> list:
        """One accepted forced-prefix chain per item: [occ [m] int64] in the order of `items`.

        An item is (U [m, m], cols, prefix, p_acc): the chain of the len(cols) photons entering the
        columns `cols` of U, its first len(prefix) rows forced to `prefix` (docs/ADAPTIVE.md §4b), aborted
        when a later photon lands on a mode of `avoid`; p_acc is the acceptance P / T when known (None:
        a running estimate). Identical items (same U, cols and prefix: the streams of one branch) are
        pooled; the pools are grouped by (photon number, prefix length) - a lockstep pass needs one chain
        length and one step to start at - and every group is ONE call of `base` on the stacked circuits
        U[:, cols] with per-circuit prefixes and per-circuit stream counts, a pool of w streams getting
        ceil(w / p_acc) chains (the expected need); the streams still without a survivor go into the next
        round (their count doubled when p_acc is a guess). A round is capped at `max_batch` chains (the
        counts are scaled down, never below 1) and the whole call at `max_draws`. Pooling the streams and
        batching the branches is what removes the per-branch launch overhead of the earlier
        one-branch-at-a-time rejection (`_reject`).
        """
        out: list = [None] * len(items)
        avoid = list(avoid)
        pools: dict = {}                                                            # (U, cols, prefix) -> [item ids]
        for i, (U, cols, prefix, _) in enumerate(items):
            pools.setdefault((U.data_ptr(), tuple(cols), tuple(prefix)), []).append(i)
        groups: dict = {}
        for key, ids in pools.items():
            groups.setdefault((len(key[1]), len(key[2])), []).append(ids)
        for (k, r), plist in groups.items():
            pending = [(list(ids), first_reps) for ids in plist]                    # (stream items, chains per stream if p_acc is None)
            drawn = 0
            while pending:
                need = []
                for ids, g in pending:
                    acc = items[ids[0]][3]
                    need.append(math.ceil(len(ids) / max(acc, 1e-300)) if acc else len(ids) * g)
                scale = min(1.0, max_batch / sum(need))
                counts = [max(1, int(c * scale)) for c in need]
                if drawn + sum(counts) > max_draws:
                    raise RuntimeError(f"forced prefix: {drawn} chains started, {sum(len(ids) for ids, _ in pending)} "
                                       f"streams still without an accepted one; raise max_draws")
                dev = items[pending[0][0][0]][0].device
                A = torch.stack([items[ids[0]][0][:, list(items[ids[0]][1])] for ids, _ in pending])   # [G, m, k]
                pre = torch.tensor([list(items[ids[0]][2]) for ids, _ in pending], dtype=torch.int64,
                                   device=dev).reshape(len(pending), r)
                res = base(A, list(range(k)), torch.tensor(counts, device=dev), generator, pre if r else None, avoid)
                if isinstance(res, tuple):
                    occ, sid = res
                else:                                                               # a base without ids: every stream survived
                    occ, sid = res, torch.arange(res.shape[0], device=res.device)
                drawn += sum(counts)
                starts = np.cumsum([0, *counts])
                circ = np.searchsorted(starts, sid.cpu().numpy(), side="right") - 1        # pool of each survivor
                rows: dict = {}
                for row, c in enumerate(circ.tolist()):
                    rows.setdefault(c, []).append(row)
                nxt = []
                for c, (ids, g) in enumerate(pending):
                    got = rows.get(c, [])
                    for i, row in zip(ids, got, strict=False):
                        out[i] = occ[row]
                    left = ids[len(got):]
                    if left:
                        nxt.append((left, g if items[ids[0]][3] else 2 * g))
                pending = nxt
            self.stats["base_shots"] += drawn
        return out

    def _reject(self, U, modes, counts, want, base, generator, p_accept, max_batch, max_draws):
        """`want` samples of the boson sampler on U conditioned on `counts` in `modes`: int64 [want, m].

        Draws batches from `base` and keeps the rows that match; the first `want` kept rows of an
        i.i.d. stream are exact conditional samples. `p_accept` (the exact acceptance when known)
        sizes the batches; without it the running acceptance rate does. A batch is the EXPECTED need
        and no more: at frontier sizes a draw costs a second, so the overshoot is what is minimised.
        The full-sample rejection of `final="clifford"` (the forced prefix goes through `_forced`).
        """
        if not modes:
            self.stats["base_shots"] += want
            return base(U, self.in_modes, want, generator)
        target = torch.as_tensor(counts, device=U.device)
        kept, got, drawn = [], 0, 0
        while got < want:
            rate = p_accept if p_accept else (got + 1) / (drawn + 2) if drawn else 0.25
            B = int(min(max_batch, max(1, math.ceil((want - got) / max(rate, 1e-12)))))      # the expected need, no margin
            if drawn + B > max_draws:
                raise RuntimeError(f"rejection on outcome {tuple(counts)}: {drawn} draws, {got}/{want} accepted "
                                   f"(acceptance {p_accept}); raise max_draws, or route rare branches with min_accept")
            occ = base(U, self.in_modes, B, generator)
            ok = occ[(occ[:, list(modes)] == target).all(1)]
            kept.append(ok); got += ok.shape[0]; drawn += B
        self.stats["base_shots"] += drawn
        return torch.cat(kept)[:want]

    def _sample_prefix(self, shots: int, generator, chain: str, base: Callable, loss, indist,
                       max_batch: int, max_draws: int) -> torch.Tensor:
        """`sample(final="prefix")`, with or without imperfections: int64 [shots, m].

        Every shot draws its partition (`imperfections.partition`: lost photons, interfering species,
        singletons). Species evolve independently through the same feed-forward, so the adaptive outcome
        is drawn level by level PER INSTANCE (a species of >= 2 photons: from its own marginal table under
        U^{p_{<lam}}, `chain="marginal"`, or a forced chain, `chain="rejection"`), the singletons landing on
        the level's modes with |U^{p_{<lam}}[a, col]|^2 conditioned on not having landed before, and the
        level's counts added over instances and singletons before the policy sees them. The final stage is
        one forced-prefix chain per instance (`_forced`, batched over streams and branches), the singletons
        still in flight being routed classically among the final modes. Lossless, indistinguishable
        photons: one instance per stream, the same code.
        """
        from .imperfections import partition
        dev = self.U0.device
        m, A = self.m, self.adaptive_modes
        gen_cpu = generator if generator is None or generator.device.type == "cpu" else \
            torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, device=dev).item()))
        inst_cols: list = [[] for _ in range(shots)]                                  # per stream: [cols tuple] (>= 2 photons)
        singles: list = [[] for _ in range(shots)]                                    # per stream: [input column]
        for subsets, single, ids in partition(self.n, shots, loss, indist, generator=gen_cpu):
            for s in ids.tolist():
                for sub in subsets:
                    cols = tuple(self.in_modes[j] for j in sub)
                    (inst_cols[s] if len(cols) >= 2 else singles[s]).append(cols if len(cols) >= 2 else cols[0])
                singles[s].extend(self.in_modes[j] for j in single)
        inst_p: list = [[() for _ in c] for c in inst_cols]                          # per instance: its counts so far
        inst_P: list = [[1.0 for _ in c] for c in inst_cols]                         # P(p^inst_{<=lam}) under the tables
        inst_T: list = [[1.0 for _ in c] for c in inst_cols]
        landed: list = [[-1 for _ in c] for c in singles]                            # adaptive mode of each singleton, or -1
        tot: list = [() for _ in range(shots)]                                       # p_{<=lam} per stream
        U_of = {(): self.U0}
        for lam in range(1, self.n_levels + 1):
            lev = self.levels[lam - 1]
            prev = A[:self.K_upto[lam - 1]]
            new = torch.zeros(shots, len(lev), dtype=torch.int64, device=dev)
            if chain == "marginal":
                keys: dict = {}
                for s in range(shots):
                    for i, cols in enumerate(inst_cols[s]):
                        keys.setdefault((tot[s], cols, inst_p[s][i]), []).append((s, i))
                self._level_tables_many(lam, list(keys))                            # all the level's tables in one batch
                for (prefix, cols, inst), members in keys.items():
                    law, mom = self._level_tables(lam, prefix, cols, inst)          # [(n'+1)]*l_lam
                    flat = torch.multinomial(law.reshape(-1), len(members), replacement=True, generator=generator)
                    cnt = torch.stack(torch.unravel_index(flat, law.shape), dim=1)  # [members, l_lam]
                    pl, tl = law.reshape(-1)[flat].tolist(), mom.reshape(-1)[flat].tolist()
                    for k, (s, i) in enumerate(members):
                        new[s] += cnt[k]
                        inst_p[s][i] = inst_p[s][i] + tuple(cnt[k].tolist())
                        inst_P[s][i], inst_T[s][i] = pl[k], tl[k]
            else:
                items, where = [], []
                for s in range(shots):
                    for i, cols in enumerate(inst_cols[s]):
                        pre = [a for a, c in zip(prev, inst_p[s][i], strict=True) for _ in range(c)]
                        items.append((self.unitary(tot[s]), cols, tuple(pre), None)); where.append((s, i))
                for (s, i), occ in zip(where, self._forced(items, prev, base, generator, max_batch, max_draws), strict=True):
                    cnt = occ[lev]
                    new[s] += cnt
                    inst_p[s][i] = inst_p[s][i] + tuple(cnt.tolist())
            for s in range(shots):                                                  # singletons: classical routing
                if not singles[s]:
                    continue
                U = self.unitary(tot[s])
                for j, col in enumerate(singles[s]):
                    if landed[s][j] >= 0:
                        continue
                    w = (U[:, col].abs() ** 2)
                    rest = 1.0 - float(w[prev].sum()) if prev else 1.0
                    q = w[lev] / max(rest, 1e-300)
                    u = float(torch.rand(1, generator=gen_cpu))
                    acc = 0.0
                    for l, a in enumerate(lev):
                        acc += float(q[l])
                        if u < acc:
                            landed[s][j] = a; new[s, l] += 1
                            break
            for s in range(shots):
                tot[s] = tot[s] + tuple(new[s].tolist())
                if tot[s] not in U_of:
                    U_of[tot[s]] = self.unitary(tot[s])
        out = torch.zeros(shots, m, dtype=torch.int64, device=dev)
        items, where = [], []
        for s in range(shots):
            for i, cols in enumerate(inst_cols[s]):
                pre = [a for a, c in zip(A, inst_p[s][i], strict=True) for _ in range(c)]
                acc = min(1.0, inst_P[s][i] / inst_T[s][i]) if chain == "marginal" and self.K else None
                items.append((U_of[tot[s]], cols, tuple(pre), acc)); where.append(s)
        self.stats["branches"] = len({tot[s] for s in range(shots)})
        for s, occ in zip(where, self._forced(items, A, base, generator, max_batch, max_draws), strict=True):
            out[s] += occ
        final = [j for j in range(m) if j not in set(A)]
        for s in range(shots):
            for j, col in enumerate(singles[s]):
                if landed[s][j] >= 0:
                    out[s, landed[s][j]] += 1
                else:
                    w = (U_of[tot[s]][final, col].abs() ** 2).cpu()
                    out[s, final[int(torch.multinomial(w / w.sum(), 1, generator=gen_cpu))]] += 1
        return out

    def _groups(self, P: torch.Tensor):
        """Distinct rows of the per-stream outcome table P [S, K_lam] -> [(tuple, stream ids)]."""
        if P.shape[1] == 0:
            return [((), torch.arange(P.shape[0], device=P.device))]
        uniq, inv = torch.unique(P, dim=0, return_inverse=True)
        return [(tuple(u.tolist()), torch.nonzero(inv == g)[:, 0]) for g, u in enumerate(uniq)]

    def sample(self, shots: int, generator: torch.Generator | None = None, chain: str = "marginal",
               final: str = "prefix", backend: str = "triton", base_sampler: Callable | None = None,
               min_accept: float = 0.0, max_batch: int = 1 << 16, max_draws: int = 1 << 26,
               loss=None, indist=None) -> torch.Tensor:
        """Exact samples of the full occupation omega = (p, s): int64 [shots, m] (fewer photons per row with loss).

        chain  how the adaptive outcomes p are drawn, level by level:
               "marginal"   from `level_probs` (exact marginals, no rejection; table (n+1)^(2 K_lam))
               "rejection"  a boson sample of U^{p_{<lam}} kept when it reproduces p_{<lam}, its
                            level-lam modes read off (any K; cost 1 / P(p_{<lam}) per level)
        final  how s | p is drawn:
               "prefix"     the chain of `fock_sample(prefix=)` on U^p with its first r photons forced onto
                            the adaptive modes, kept when no other photon joins them: acceptance
                            P(p) / T[p], mean cost E[2^r] chains STARTED per sample, no rare-branch tail; a chain
                            is aborted as soon as a photon lands on an adaptive mode, so the rejected ones
                            cost ~1/((n-r) ln 2) of a full one. Also conditions the levels of
                            chain="rejection". `base_sampler` then takes two more arguments, the prefix
                            (list of output modes) and the modes to avoid, and may return fewer rows.
               "clifford"   rejection on the full sample of the Clifford & Clifford sampler run on U^p
                            (`fock_sample`, `backend`), acceptance P(p): one chain per BRANCH on average;
                            or on `base_sampler(U, in_modes, shots, generator) -> [shots, m]` when given
               "slos"       directly from the strong distribution of U^p restricted to p (small n)
        min_accept  with chain="marginal" and final="clifford": branches with P(p) < min_accept take
               the "slos" route instead of the rejection (rare branches carry many photons in p,
               so few are left for SLOS).
        loss, indist  uniform transmission eta and partial distinguishability (docs/LOSS.md,
               docs/DISTINGUISHABILITY.md), input side as in `clifford.fock_sample`: every shot draws
               which photons survive and which interfere; final="prefix" only. The policy sees the
               detected counts. With `final="prefix"` the streams and the branches of the final stage
               are batched together (`_forced`), lossless or not.
        `generator` must live on U0's device (the chain draws with torch.multinomial there).
        `self.stats` afterwards: base_shots (chains started), branches (distinct p), slos_branches.
        """
        if chain not in ("marginal", "rejection"):
            raise ValueError(f'unknown chain {chain!r}: "marginal" or "rejection"')
        if final not in ("prefix", "clifford", "slos"):
            raise ValueError(f'unknown final {final!r}: "prefix", "clifford" or "slos"')
        dev = self.U0.device
        forced = final == "prefix"
        if base_sampler is not None:
            base = base_sampler
        else:
            base = self._prefix_sampler(backend) if forced else self._base_sampler(backend)
        self.stats = {"base_shots": 0, "branches": 0, "slos_branches": 0}
        if forced:
            return self._sample_prefix(shots, generator, chain, base, loss, indist, max_batch, max_draws)
        from .imperfections import spec
        if spec(indist) is not None or not (loss is None or float(loss) >= 1.0):
            raise NotImplementedError('loss / indist: final="prefix" only')
        Tp = torch.ones(shots, dtype=torch.float64, device=dev)                      # T[p_{<=lam}] per stream
        P = torch.zeros(shots, 0, dtype=torch.int64, device=dev)                     # [S, K_lam] outcomes so far
        Pp = torch.ones(shots, dtype=torch.float64, device=dev)                      # P(p_{<=lam}) per stream
        for lam in range(1, self.n_levels + 1):
            lev = self.levels[lam - 1]
            new = torch.zeros(shots, len(lev), dtype=torch.int64, device=dev)
            for prefix, ids in self._groups(P):
                if chain == "marginal":
                    law, mom = self._level_tables(lam, prefix)                       # [(n+1)]*l_lam each
                    flat = torch.multinomial(law.reshape(-1), ids.shape[0], replacement=True, generator=generator)
                    new[ids] = torch.stack(torch.unravel_index(flat, law.shape), dim=1)
                    Pp[ids] = law.reshape(-1)[flat]; Tp[ids] = mom.reshape(-1)[flat]
                else:
                    occ = self._reject(self.unitary(prefix), self.adaptive_modes[:len(prefix)], prefix, ids.shape[0],
                                       base, generator, None, max_batch, max_draws)
                    new[ids] = occ[:, lev]
            P = torch.cat([P, new], dim=1)
        out = torch.zeros(shots, self.m, dtype=torch.int64, device=dev)
        for p, ids in self._groups(P):
            self.stats["branches"] += 1
            U = self.unitary(p)
            p_acc = float(Pp[ids[0]]) if chain == "marginal" and self.K else None
            if p_acc is not None and forced:
                p_acc = min(1.0, p_acc / float(Tp[ids[0]]))
            if final == "slos" or (p_acc is not None and not forced and p_acc < min_accept):
                self.stats["slos_branches"] += 1
                pr = self._slos().probs(U).to(torch.float64)                         # [S]
                pr = pr * (self._occ[:, self.adaptive_modes] == torch.as_tensor(p, device=dev)).all(1)
                out[ids] = self._occ[torch.multinomial(pr, ids.shape[0], replacement=True, generator=generator)]
            else:
                out[ids] = self._reject(U, self.adaptive_modes, p, ids.shape[0], base, generator, p_acc,
                                        max_batch, max_draws)
        return out
