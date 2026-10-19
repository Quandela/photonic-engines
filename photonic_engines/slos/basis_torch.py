"""The Fock basis and its raising edges built ON THE DEVICE with torch — same construction as
`basis.py` (lex-ordered position layers, packed-key lookup), no host-side enumeration.

Differences from the numpy `FockLayers`:
  * edges are produced directly GROUPED BY RAISED MODE (the layout the kernels consume): for
    every mode j, the children containing j are found with one reduction, the parent of each
    is the child with one j removed, its position by `searchsorted` on the parent layer's packed
    keys. No child-major edge list, no sort. `edges[k]` (child-major, with mode and factor) is
    derived lazily for the edge-layout arms.
  * positions are stored as int32 (layers < 2^31 states); slabs are widened to int64 when used.
  * packed keys are int64: k * ceil(log2 m) must be <= 63 (n = 12 at m = 24 is 60 bits; beyond
    that the numpy builder's Python fallback applies).

Memory on the device for the last layer at n = 12, m = 24: states 1.7 GB, edges ~1.15 G x 8 B
= 9 GB (parent + child int32).
"""
from __future__ import annotations

import math
import os

import torch

from ..masks import Mask
from .basis import _block_shift, _per_mode_max


class TorchFockLayers:
    """Device-resident twin of `FockLayers`: mask-pruned Fock layers and raising edges built with torch.

    Layers are built eagerly in `__init__` (one per photon), each as an int8/int16 [S_k, k] tensor of sorted photon
    positions in lexicographic order, together with the k -> k+1 edges already grouped by raised mode (int32 positions
    plus an m + 1 offset list). Everything lives on `device`; the child-major `edges` layout is derived lazily and cached.
    """

    def __init__(self, n: int, m: int, mask: str = Mask.FOCK.value, device="cpu"):
        self.n, self.m, self.mask = n, m, mask
        self.device = torch.device(device)
        bits = max(1, math.ceil(math.log2(max(m, 2))))
        if n * bits > 63:
            raise ValueError(f"n={n}, m={m}: packed keys need {n * bits} > 63 bits; use the numpy builder")
        self.bits = bits
        self.mode_dtype = torch.int8 if m <= 127 else torch.int16
        per_max = torch.as_tensor(_per_mode_max(mask, n, m), dtype=torch.int64, device=self.device)
        shift = _block_shift(mask)
        self.states: list[torch.Tensor] = [torch.zeros((1, 0), dtype=self.mode_dtype, device=self.device)]
        self._by_mode: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
        self._edges_cache = {}
        debug = os.environ.get("PE_MEM_DEBUG") and self.device.type == "cuda"
        for k in range(n):
            child = self._children(self.states[k], per_max, shift)
            self.states.append(child)
            if debug:
                torch.cuda.synchronize()
                print(f"  [basis] layer {k + 1}: {child.shape[0]:,} states; allocated "
                      f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB, "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
            self._by_mode.append(self._edges_by_mode(self.states[k], child))
            if debug:
                torch.cuda.synchronize()
                print(f"  [basis] edges {k}->{k + 1}: {self._by_mode[-1][0].shape[0]:,}; allocated "
                      f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB, "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)

    # ---- layers ----------------------------------------------------------------------------
    def _children(self, parents: torch.Tensor, per_max: torch.Tensor, shift: int) -> torch.Tensor:
        S, k = parents.shape
        m = self.m
        dev = self.device
        modes = torch.arange(m, dtype=torch.int64, device=dev)
        if k == 0:
            adm = (per_max >= 1).unsqueeze(0)
        else:
            last = parents[:, -1].to(torch.int64)
            run = torch.ones(S, dtype=torch.int64, device=dev)
            for j in range(k - 2, -1, -1):
                same = parents[:, j] == parents[:, -1]
                run = torch.where(same & (run == k - 1 - j), run + 1, run)
            adm = modes.unsqueeze(0) > last.unsqueeze(1)
            adm &= (per_max >= 1).unsqueeze(0)
            adm[torch.arange(S, device=dev), last] = run < per_max[last]
            if shift:
                adm &= (modes.unsqueeze(0) >> shift) != (last.unsqueeze(1) >> shift)
        rows, a = adm.nonzero(as_tuple=True)                     # row-major = lex order
        del adm
        out = torch.empty((rows.shape[0], k + 1), dtype=self.mode_dtype, device=dev)
        if k:
            out[:, :k] = parents[rows]
        out[:, k] = a.to(self.mode_dtype)
        return out

    def _pack(self, states: torch.Tensor) -> torch.Tensor:
        S, k = states.shape
        key = torch.zeros(S, dtype=torch.int64, device=self.device)
        for j in range(k):
            key = (key << self.bits) | states[:, j].to(torch.int64)
        return key

    # ---- edges grouped by raised mode ------------------------------------------------------
    def _edges_by_mode(self, parents: torch.Tensor, children: torch.Tensor, chunk: int = 1 << 24):
        S1, k1 = children.shape
        k = k1 - 1
        dev = self.device
        parent_keys = self._pack(parents) if k else None
        par_list, chi_list, offsets = [], [], [0]
        keep = torch.tensor([[jj for jj in range(k1) if jj != c] for c in range(k1)],
                            dtype=torch.int64, device=dev)          # [k1, k]
        for j in range(self.m):
            is_j = children == j                                       # [S1, k1]
            has = is_j.any(dim=1)
            cpos = has.nonzero(as_tuple=True)[0]                       # children containing j
            if cpos.numel() == 0:
                offsets.append(offsets[-1])
                continue
            if k:
                ppos = torch.empty(cpos.shape[0], dtype=torch.int64, device=dev)
                for a in range(0, cpos.shape[0], chunk):          # bound the int64 temporaries
                    cp = cpos[a:a + chunk]
                    sub = children[cp]                                 # [e, k1]
                    col = is_j[cp].to(torch.int8).argmax(dim=1)        # first occurrence of j
                    par = sub[torch.arange(sub.shape[0], device=dev).unsqueeze(1), keep[col]]
                    ppos[a:a + chunk] = torch.searchsorted(parent_keys, self._pack(par))
                    del sub, par, col
            else:
                ppos = torch.zeros_like(cpos)
            par_list.append(ppos.to(torch.int32))
            chi_list.append(cpos.to(torch.int32))
            offsets.append(offsets[-1] + int(cpos.shape[0]))
            del is_j, has
        parent = torch.cat(par_list) if par_list else torch.zeros(0, dtype=torch.int32, device=dev)
        child = torch.cat(chi_list) if chi_list else torch.zeros(0, dtype=torch.int32, device=dev)
        return parent, child, offsets

    def edges_by_mode(self, k: int):
        """The k -> k+1 edges grouped by raised mode: (parent_pos, child_pos, offsets).

        Positions are int32 tensors on the device; `offsets` is a Python list of m + 1 ints, so the edges raising mode j
        are the slab [offsets[j], offsets[j + 1]). Built during `__init__`, not on demand.
        """
        return self._by_mode[k]

    @property
    def edges(self):
        """Child-major (parent, child, mode, factor) tensors per layer, derived on demand."""
        if "edges" not in self._edges_cache:
            out = []
            for k in range(self.n):
                parent, child, off = self._by_mode[k]
                E = parent.shape[0]
                mode = torch.empty(E, dtype=self.mode_dtype, device=self.device)
                for j in range(self.m):
                    mode[off[j]:off[j + 1]] = j
                order = torch.argsort(child.to(torch.int64), stable=True)
                p, c, md = parent[order].to(torch.int64), child[order].to(torch.int64), mode[order]
                # factor = sqrt(occupation of mode in the child)
                ch = self.states[k + 1][c]                              # [E, k+1]
                cnt = (ch == md.unsqueeze(1)).sum(dim=1)
                out.append((p, c, md.to(torch.int64), torch.sqrt(cnt.to(torch.float64))))
            self._edges_cache["edges"] = out
        return self._edges_cache["edges"]

    # ---- output basis ----------------------------------------------------------------------
    def size(self, k: int) -> int:
        """Number of states in layer k."""
        return int(self.states[k].shape[0])

    def output_size(self) -> int:
        """Number of states in the final (n-photon) layer."""
        return self.size(self.n)

    @property
    def layers(self):
        """All layers as lists of photon-position tuples — copies every layer to host, for inspection only."""
        return [[tuple(r) for r in s.cpu().tolist()] for s in self.states]

    def final_occupations(self) -> torch.Tensor:
        """Output basis as an int64 [S, m] occupation tensor on the device."""
        t = self.states[self.n].to(torch.int64)
        S = t.shape[0]
        occ = torch.zeros((S, self.m), dtype=torch.int64, device=self.device)
        ar = torch.arange(S, device=self.device)
        for j in range(self.n):
            occ[ar, t[:, j]] += 1
        return occ

    @property
    def final_states(self):
        """Output basis as a list of length-m occupation tuples (copies the final layer to host)."""
        return [tuple(r) for r in self.final_occupations().cpu().tolist()]

    def output_prob_norms(self) -> torch.Tensor:
        """Per final state, prod_j (t_j!) as an int64 device tensor — the deferred bosonic normalization factor.

        Multiplying |raw amplitude|^2 by this restores the bosonic magnitude; all 1 for unbunched masks.
        """
        return self.prob_norms(self.n)

    def prob_norms(self, k: int) -> torch.Tensor:
        """`output_prob_norms` for an INTERMEDIATE layer k — the deferred normalization of a graded
        engine, which reads every layer and not only the last (`slos.ProductSLOSKernel`).
        """
        t = self.states[k]
        S, k = t.shape
        out = torch.ones(S, dtype=torch.int64, device=self.device)
        pos = torch.ones(S, dtype=torch.int64, device=self.device)
        for j in range(1, k):
            same = t[:, j] == t[:, j - 1]
            pos = torch.where(same, pos + 1, torch.ones_like(pos))
            out *= pos
        return out
