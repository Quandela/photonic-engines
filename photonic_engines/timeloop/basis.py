"""Register Fock basis for the native time-loop / periodic-network kernels: occupations of
R = n_rail + L modes (modes 0..n_rail-1 = the rail wires of the current period, in transit;
modes n_rail..R-1 = delay-line slots) with at most N photons, plus the index tables the kernels
need:

  pair tables (a, b): every basis state belongs to one group (the occupations of the other
      modes fixed, k = n_a + n_b fixed) at position p = n_a; a two-mode gate is the (k+1)x(k+1)
      Fock block B_k on each group. Groups are sorted by k.
  rail patterns: the Q tuples of rail occupations with sum <= N (for n_rail = 1: 0..N in order);
      rail_of[i] is the pattern id of state i.
  inject[q] : index of the state with the rail cleared, for states with rail pattern q (else -1)
  exit[q]   : index of the state with rail pattern q (same slots), for states with empty rail
  loss tables (mode w, j photons lost): index of the state with n_w + j photons and the factor
      sqrt(C(n_w + j, j)) (the eta powers are applied by the kernel)
"""
from __future__ import annotations

from math import comb, sqrt

import numpy as np


def _fock_rows(R: int, N: int) -> np.ndarray:
    """All occupation rows of R modes with total <= N, [C(N+R, R), R], lexicographic
    (vectorised: one mode appended at a time).
    """
    rows = np.arange(N + 1, dtype=np.int64)[:, None]
    for _ in range(R - 1):
        rem = N - rows.sum(1)
        counts = rem + 1
        rep = np.repeat(rows, counts, axis=0)
        tot = int(counts.sum())
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        col = np.arange(tot, dtype=np.int64) - starts
        rows = np.concatenate([rep, col[:, None]], 1)
    return rows


class RegisterBasis:
    """Truncated Fock basis of the R = n_rail + L register modes with at most N photons, with the kernels' index tables.

    States are the lexicographic occupation rows `occ` [D, R], D = C(N + R, R); lookup goes through radix-(N+1) codes and
    a sorted-code `searchsorted`. Pair, loss and rail-pattern tables are built on demand and cached per (a, b) / mode.
    All tables are numpy int64 on the host; the kernels move them to the device.
    """

    def __init__(self, L: int, N: int, n_rail: int = 1):
        self.L, self.N, self.n_rail = int(L), int(N), int(n_rail)
        self.R = self.n_rail + self.L
        self.occ = _fock_rows(self.R, self.N)                                   # [D, R]
        self.D = self.occ.shape[0]
        self.total = self.occ.sum(1)
        base = self.N + 1
        self._radix = base ** np.arange(self.R, dtype=np.int64)
        self._codes = self.occ @ self._radix
        order = np.argsort(self._codes)
        self._sorted_codes = self._codes[order]
        self._order = order
        # rail patterns
        self.rail_patterns = _fock_rows(self.n_rail, self.N)                     # [Q, n_rail]
        self.Q = self.rail_patterns.shape[0]
        rradix = base ** np.arange(self.n_rail, dtype=np.int64)
        rcodes = self.rail_patterns @ rradix
        lut = -np.ones(int(rcodes.max()) + 1, dtype=np.int64)
        lut[rcodes] = np.arange(self.Q)
        self.rail_of = lut[self.occ[:, :self.n_rail] @ rradix]                  # [D]
        self._pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._loss_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def index(self, occ: np.ndarray) -> np.ndarray:
        """Indices of occupation rows [..., R]; -1 for rows outside the basis."""
        occ = np.asarray(occ, dtype=np.int64)
        codes = occ @ self._radix
        pos = np.searchsorted(self._sorted_codes, codes)
        pos = np.minimum(pos, self.D - 1)
        found = (self._sorted_codes[pos] == codes) & (occ.min(-1) >= 0) & (occ.sum(-1) <= self.N)
        return np.where(found, self._order[pos], -1)

    # ---- two-mode gate tables --------------------------------------------------------------
    def pair_tables(self, a: int, b: int):
        """(member [G, N+1] basis index per (group, p = n_a) or -1, group_k [G] sorted ascending,
        inv [D] flat position group*(N+1)+p of every basis state) for the mode pair (a, b).
        """
        if (a, b) in self._pair_cache:
            return self._pair_cache[(a, b)]
        occ = self.occ
        k = occ[:, a] + occ[:, b]
        rest = occ.copy()
        rest[:, a] = 0
        rest[:, b] = 0
        key = rest @ self._radix + k * (self._radix[-1] * (self.N + 1))
        uniq, gid = np.unique(key, return_inverse=True)
        gk = np.zeros(len(uniq), dtype=np.int64)
        gk[gid] = k
        order = np.argsort(gk, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        gid = rank[gid]
        gk = gk[order]
        G = len(uniq)
        member = -np.ones((G, self.N + 1), dtype=np.int64)
        member[gid, occ[:, a]] = np.arange(self.D)
        inv = gid * (self.N + 1) + occ[:, a]
        self._pair_cache[(a, b)] = (member, gk, inv)
        return self._pair_cache[(a, b)]

    # ---- inject / exit ---------------------------------------------------------------------
    def _rail_cleared(self) -> np.ndarray:
        src = self.occ.copy()
        src[:, :self.n_rail] = 0
        return self.index(src)

    def inject_map(self, q: int) -> np.ndarray:
        """[D] source index (same slots, empty rail) for states with rail pattern q, else -1."""
        return np.where(self.rail_of == q, self._rail_cleared(), -1)

    def exit_map(self, q: int) -> np.ndarray:
        """[D] source index (same slots, rail pattern q) for states with empty rail, else -1."""
        src = self.occ.copy()
        src[:, :self.n_rail] = self.rail_patterns[q]
        idx = self.index(src)
        return np.where(self.rail_of == 0, idx, -1)

    def pattern_id(self, rail: np.ndarray) -> np.ndarray:
        """Pattern ids of rail occupation rows [..., n_rail] (-1 if outside)."""
        rail = np.asarray(rail, dtype=np.int64)
        base = self.N + 1
        rradix = base ** np.arange(self.n_rail, dtype=np.int64)
        codes = self.rail_patterns @ rradix
        lut = -np.ones(int(codes.max()) + 1, dtype=np.int64)
        lut[codes] = np.arange(self.Q)
        c = rail @ rradix
        ok = (rail.min(-1) >= 0) & (rail.sum(-1) <= self.N) & (rail.max(-1) <= self.N)
        return np.where(ok, lut[np.clip(c, 0, len(lut) - 1)], -1)

    # ---- single-mode loss ------------------------------------------------------------------
    def loss_tables(self, mode: int):
        """Kraus map E_j (j photons lost in mode w) as a gather:
        (E_j psi)[i] = coef[j, i] * eta^{n_w(i)/2} * (1-eta)^{j/2} * psi[src[j, i]] with
        src[j, i] the state with n_w + j photons in w (-1 if outside the basis) and
        coef[j, i] = sqrt(C(n_w(i) + j, j)). Returns (src [N+1, D], coef [N+1, D]).
        """
        if mode in self._loss_cache:
            return self._loss_cache[mode]
        src = -np.ones((self.N + 1, self.D), dtype=np.int64)
        coef = np.zeros((self.N + 1, self.D), dtype=np.float64)
        nw = self.occ[:, mode]
        ctab = np.array([[sqrt(comb(a + j, j)) for j in range(self.N + 1)] for a in range(self.N + 1)])
        for j in range(self.N + 1):
            tgt = self.occ.copy()
            tgt[:, mode] = nw + j
            src[j] = self.index(tgt)
            coef[j] = np.where(src[j] >= 0, ctab[np.minimum(nw, self.N), j], 0.0)
        self._loss_cache[mode] = (src, coef)
        return self._loss_cache[mode]

    def dimension_table(self) -> list[int]:
        """Basis size C(N + r, r) for r = 1..R modes — the growth of the register dimension, for sizing reports."""
        return [comb(self.N + r, r) for r in range(1, self.R + 1)]
