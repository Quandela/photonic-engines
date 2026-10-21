"""Mask-pruned Fock basis as photon-position layers, plus the raising edges between layers —
built with vectorized numpy, never one Python object per state or per edge.

A k-photon state is the sorted tuple of its photons' modes (length k, not m). Layer k holds
all such states, in lexicographic order, as an int8 (int16 when m > 127) array `states[k]` of shape [S_k, k].
The k -> k+1 raising edges are four flat arrays (parent_pos, child_pos as the narrowest of
int32/int64 that holds the layer size, mode int8/int16 (`mode_dtype(m)`), factor float32 = sqrt(parent occupation of
mode + 1)), child-major; consumers widen them to int64 on the device.

Construction (per layer, all vectorized):
  * children of layer k = every parent followed by one admissible mode a >= last(parent)
    (per-mode cap and block constraints checked against the trailing run only, since the
    parent is already valid). Every (k+1)-state has exactly one such (prefix, last) split, and
    lexicographic order of children = parent order then a, so no sort is needed.
  * edges: for every child and every FIRST occurrence of a mode in it, the parent is the child
    with that entry removed; its position is found by packing states into 64-bit keys (the
    key order is the lexicographic order) and `searchsorted` on the parent layer's keys.
    Packing needs k * ceil(log2 m) <= 64 bits, which holds for every (n, m) whose basis fits in
    memory; beyond that a Python fallback builds the layers as before.

Cost: O(S_k * m) for the admissibility matrix and O(E log S) for the edge lookup, in numpy —
n = 10, m = 20 (20 M states, 138 M edges in the last layer) builds in ~2 min on two slow cores
with a 4.3 GB peak, against minutes-to-hours and tens of GB for the tuple/dict construction.
`layers[k]` still yields position tuples for inspection (materialized on
demand); `final_states` / `final_occupations()` give the output basis as occupation vectors.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

import numpy as np

from ..masks import Mask

State = tuple[int, ...]


def _per_mode_max(mask: str, n: int, m: int) -> list[int]:
    """Upper bound on each mode's occupation (prunes the bunching dimension up front)."""
    if mask == Mask.FOCK.value:
        return [n] * m
    if mask in (Mask.UNBUNCHED.value, Mask.DUAL_RAIL.value, Mask.SPARSE_1IN4.value):
        return [1] * m
    if mask == Mask.TAIL_EMPTY.value:
        return [n] * (m - 2) + [0, 0]
    raise ValueError(f"unknown mask {mask!r}")


def _block_shift(mask: str) -> int:
    """Modes are grouped in blocks of 2**shift holding at most one photon (0 = no blocks)."""
    if mask == Mask.DUAL_RAIL.value:
        return 1
    if mask == Mask.SPARSE_1IN4.value:
        return 2
    return 0


def positions_to_occupation(t: Sequence[int], m: int) -> tuple[int, ...]:
    """Sorted position tuple -> length-m occupation vector."""
    occ = [0] * m
    for p in t:
        occ[int(p)] += 1
    return tuple(occ)


class _TupleView:
    """Lazy list-of-tuples view of a [S, k] state array (for inspection and small tests)."""

    def __init__(self, arr: np.ndarray):
        self._a = arr

    def __len__(self):
        return self._a.shape[0]

    def __iter__(self):
        for row in self._a.tolist():
            yield tuple(row)

    def __getitem__(self, i):
        return tuple(self._a[i].tolist())


def mode_dtype(m: int):
    """Numpy dtype for mode positions: int8 up to 127 modes, int16 beyond (m <= 32767)."""
    if m > 32767:
        raise ValueError(f"m={m}: at most 32767 modes")
    return np.int8 if m <= 127 else np.int16


class FockLayers:
    """Precomputed, mask-pruned Fock basis for 0..n photons plus the raising edges.

    Attributes:
        states[k] : int8 (int16 for m > 127) array [S_k, k], k-photon states as sorted mode positions, lex order
        layers[k] : the same as a lazy sequence of tuples (len / iter / index)
        edges[k]  : (parent_pos, child_pos, mode, factor) flat arrays for k -> k+1
        final_states : output basis as occupation tuples (list, materialized on demand)
    """

    def __init__(self, n: int, m: int, mask: str = Mask.FOCK.value):
        self.n, self.m, self.mask = n, m, mask
        per_max = np.asarray(_per_mode_max(mask, n, m), dtype=np.int64)
        shift = _block_shift(mask)
        self.mode_dtype = mode_dtype(m)
        self.states: list[np.ndarray] = [np.zeros((1, 0), dtype=self.mode_dtype)]
        self.edges: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        bits = max(1, math.ceil(math.log2(max(m, 2))))
        for k in range(n):
            child = self._children(self.states[k], per_max, shift, self.mode_dtype)
            self.states.append(child)
            if (k + 1) * bits <= 64:
                self.edges.append(self._edges(self.states[k], child, bits, self.mode_dtype))
            else:                                   # states too wide to pack: dict lookup
                self.edges.append(self._edges_python(self.states[k], child, self.mode_dtype))
        self.layers: list[_TupleView] = [_TupleView(a) for a in self.states]

    # ---- layer generation ------------------------------------------------------------------
    @staticmethod
    def _children(parents: np.ndarray, per_max: np.ndarray, shift: int, dtype=np.int8) -> np.ndarray:
        S, k = parents.shape
        m = per_max.shape[0]
        modes = np.arange(m, dtype=np.int64)
        if k == 0:
            adm = (per_max >= 1)[None, :]                              # [1, m]
        else:
            last = parents[:, -1].astype(np.int64)                     # [S]
            # trailing run length of `last` (occupation of the last mode)
            run = np.ones(S, dtype=np.int64)
            for j in range(k - 2, -1, -1):
                same = parents[:, j] == parents[:, -1]
                run = np.where(same & (run == k - 1 - j), run + 1, run)
            adm = modes[None, :] > last[:, None]                       # a > last: new mode
            adm &= (per_max >= 1)[None, :]
            # a == last: one more photon in the same mode, if the cap allows
            same_ok = run < per_max[last]                              # [S]
            adm[np.arange(S), last] = same_ok
            if shift:
                # block constraint: the new mode must not share its block with the last mode
                # (earlier modes are in strictly smaller blocks, so only the last one matters)
                adm &= (modes[None, :] >> shift) != (last[:, None] >> shift)
        rows, a = np.nonzero(adm)                                      # row-major = lex order
        out = np.empty((rows.shape[0], k + 1), dtype=dtype)
        if k:
            out[:, :k] = parents[rows]
        out[:, k] = a
        return out

    # ---- edges -----------------------------------------------------------------------------
    @staticmethod
    def _pack(states: np.ndarray, bits: int) -> np.ndarray:
        """[S, k] int8/int16 -> uint64 keys whose numeric order is the lexicographic state order."""
        S, k = states.shape
        key = np.zeros(S, dtype=np.uint64)
        for j in range(k):
            key = (key << np.uint64(bits)) | states[:, j].astype(np.uint64)
        return key

    @classmethod
    def _edges(cls, parents: np.ndarray, children: np.ndarray, bits: int, dtype=np.int8, chunk: int = 1 << 22):
        S1, k1 = children.shape
        k = k1 - 1
        idx_dtype = np.int32 if max(S1, parents.shape[0]) < 2 ** 31 else np.int64
        parent_keys = cls._pack(parents, bits)                         # sorted (lex order)
        # first occurrence of each mode within a child -> one edge; child-major order
        first = np.ones((S1, k1), dtype=bool)
        if k1 > 1:
            first[:, 1:] = children[:, 1:] != children[:, :-1]
        per_child = first.sum(axis=1).astype(np.int64)
        cols = np.nonzero(first)[1].astype(np.int8)
        del first
        child_pos = np.repeat(np.arange(S1, dtype=idx_dtype), per_child)
        E = child_pos.shape[0]
        del per_child
        keep_idx = np.array([[j for j in range(k1) if j != c] for c in range(k1)], dtype=np.int64)
        parent_pos = np.empty(E, dtype=idx_dtype)
        mode = np.empty(E, dtype=dtype)
        factor = np.empty(E, dtype=np.float32)
        for s in range(0, E, chunk):
            r, c = child_pos[s:s + chunk].astype(np.int64), cols[s:s + chunk].astype(np.int64)
            sub = children[r]                                          # [e, k1]
            ar = np.arange(r.shape[0])
            md = sub[ar, c]
            mode[s:s + chunk] = md
            factor[s:s + chunk] = np.sqrt((sub == md[:, None]).sum(axis=1, dtype=np.int64))
            if k:
                par = sub[ar[:, None], keep_idx[c]]                    # [e, k]
                parent_pos[s:s + chunk] = np.searchsorted(parent_keys, cls._pack(par, bits))
            else:
                parent_pos[s:s + chunk] = 0
            del sub
        return (parent_pos, child_pos, mode, factor)

    @staticmethod
    def _edges_python(parents: np.ndarray, children: np.ndarray, dtype=np.int8):
        index = {tuple(row): i for i, row in enumerate(parents.tolist())}
        pp, cp, md, fc = [], [], [], []
        for c_pos, child in enumerate(children.tolist()):
            L = len(child)
            i = 0
            while i < L:
                a = child[i]
                j = i + 1
                while j < L and child[j] == a:
                    j += 1
                pp.append(index[tuple(child[:i] + child[i + 1:])])
                cp.append(c_pos)
                md.append(a)
                fc.append(math.sqrt(j - i))
                i = j
        return (np.asarray(pp, dtype=np.int64), np.asarray(cp, dtype=np.int64),
                np.asarray(md, dtype=dtype), np.asarray(fc, dtype=np.float32))

    # ---- edges grouped by mode ------------------------------------------------------------
    def edges_by_mode(self, k: int):
        """The k -> k+1 edges regrouped by raised mode: (parent_pos, child_pos, offsets) with
        offsets int64[m + 1] such that edges of mode j are the slab [offsets[j], offsets[j+1]).
        Within a slab the multiplier U[i, j] is a constant, so a kernel processes the slab with
        one scalar (or one chi x chi matrix) instead of a per-edge gather. Cached.
        """
        cache = self.__dict__.setdefault("_by_mode", {})
        if k not in cache:
            parent, child, mode, _ = self.edges[k]
            order = np.argsort(mode, kind="stable")
            offsets = np.searchsorted(mode[order], np.arange(self.m + 1, dtype=mode.dtype)).astype(np.int64)
            cache[k] = (parent[order], child[order], offsets)
        return cache[k]

    # ---- output basis ----------------------------------------------------------------------
    def size(self, k: int) -> int:
        """Number of states in layer k."""
        return int(self.states[k].shape[0])

    def output_size(self) -> int:
        """Number of states in the final (n-photon) layer."""
        return self.size(self.n)

    @property
    def final_states(self) -> list[tuple[int, ...]]:
        """Output basis as a list of length-m occupation tuples, in layer order (materialized on demand)."""
        return [tuple(r) for r in self.final_occupations().tolist()]

    def final_occupations(self, dtype=np.int64) -> np.ndarray:
        """Output basis as an [S, m] occupation array (vectorized)."""
        t = self.states[self.n]
        S = t.shape[0]
        occ = np.zeros((S, self.m), dtype=dtype)
        ar = np.arange(S)
        for j in range(self.n):
            occ[ar, t[:, j]] += 1
        return occ

    def output_prob_norms(self) -> np.ndarray:
        """Per final state, prod_j (t_j!) as an int64 array — the deferred normalization factor
        (|raw amplitude|^2 * this restores the bosonic magnitude). All 1 for unbunched masks.
        """
        return self.prob_norms(self.n)

    def prob_norms(self, k: int) -> np.ndarray:
        """`output_prob_norms` for an INTERMEDIATE layer k — the deferred normalization of a graded
        engine, which reads every layer and not only the last (`slos.ProductSLOSKernel`).
        """
        t = self.states[k]
        S, k = t.shape
        out = np.ones(S, dtype=np.int64)
        pos = np.ones(S, dtype=np.int64)          # position within the current run
        for j in range(1, k):
            same = t[:, j] == t[:, j - 1]
            pos = np.where(same, pos + 1, 1)
            out *= pos                            # a run of length L contributes 1*2*...*L
        return out


def get_layers(n: int, m: int, mask: str = Mask.FOCK.value, device: str = "cpu",
               backend: str = "auto"):
    """Cached basis. device="cpu" builds with numpy (this module); a CUDA device builds on the
    device with torch (`basis_torch.TorchFockLayers`). backend="numpy" | "torch" forces one.
    Arguments are canonicalized so keyword and positional calls share one cache entry.
    """
    dev = str(device)
    if backend == "auto":
        backend = "numpy" if dev == "cpu" else "torch"
    return _get_layers_cached(int(n), int(m), str(mask), dev, backend)


@cache
def _get_layers_cached(n: int, m: int, mask: str, dev: str, backend: str):
    if backend == "numpy":
        return FockLayers(n, m, mask)
    from .basis_torch import TorchFockLayers
    return TorchFockLayers(n, m, mask, device=dev)


def clear_layers_cache() -> None:
    """Drop every cached basis (frees their device memory once no kernel references them), including the fine
    (spatial x internal) stages built on them for internal photon modes.
    """
    _get_layers_cached.cache_clear()
    from .internal import _STAGES
    _STAGES.clear()


def device_bytes(layers) -> int:
    """Bytes held on the device by a TorchFockLayers (states + mode-grouped edges)."""
    tot = sum(s.numel() * s.element_size() for s in getattr(layers, "states", []))
    for k in range(getattr(layers, "n", 0)):
        p, c, _ = layers.edges_by_mode(k)
        if hasattr(p, "element_size"):
            tot += p.numel() * p.element_size() + c.numel() * c.element_size()
    return tot
