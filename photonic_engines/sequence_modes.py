"""SequenceKernel with internal (temporal / frequency) photon modes — docs/SEQUENCE.md §14.

A photon of emitter j leaves bin b with label l in internal mode k (`ModeBinEmitter.ops[b, l, k]`); the interferometer
acts as U (x) 1_K and the detectors sum over k, so photons interfere only within the same internal mode and the outcome
of a bin is the spatial pattern t (internal index summed out). `SequenceKernel(emitters, m)` returns this class when
some emitter is a `ModeBinEmitter`; a problem without internal modes never reaches this module.

The spatial basis of a bin (`layers`, `patterns`, `S`, `pattern_index`) is the one of `SequenceKernel`: k photons over
m modes. `kraus(U, b)` / `coarse(U, b)` give the channel of a bin as the fine Kraus set on the populated internal span
(route F, `slos.internal`), with the contract of the partial-distinguishability path: physical operators [Q, chi, chi]
and the spatial pattern index of each; `compress=rtol` returns the per-outcome Choi-rank set, by compressing the fine
set or, with route "R" (automatic when the fine set does not fit), without building it (`slos.internal.range_kraus`:
range samples + a core from pairing applies, with an a-posteriori error estimate; outcomes above tol recomputed by F).
`outcome_law(U, b, rho)`
and `condition(U, b, rho, outcomes)` apply the channel to a register (state, density matrix or batch) by route F or by
the pairing route P' (`slos.internal.pairing_apply`, cost independent of K, with an a-posteriori certificate; results
above `tol` are recomputed by F), chosen per call by a device cost model (`COST`). The other `SequenceKernel` methods
raise NotImplementedError.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .masks import Mask
from .sequence import ModeBinEmitter, SequenceKernel

_NOT_YET = ("sequence_probability", "sequence_amplitude", "transfer", "bin_marginals", "joint", "sample", "sample_diag",
            "label_amplitudes", "fock_sample")
# public members inherited unchanged: they depend on the spatial basis of a bin (k photons over m modes) only
# (`patterns` and `S` are overridden: with `lost` operators they cover 0..k detected photons)
SPATIAL = ("layers", "pattern_index", "clear_cache", "from_mps")

# Cost model of the route dispatcher, seconds per unit, per device type (measured; an uncalibrated device keeps the
# channel route F unless its fine set does not fit): F_build per (photon x fine pattern x D^3), F_apply per
# (fine operator x batch x D^3), P per (outcome x 4^(n-1) x n^2 d^2 D^2 x batch); optional F0, P0: seconds per call
# (launch / Python overhead, which dominates small ticks).
# cuda: H100 PCIe, run 20261020_182236 (k = 2-6, K = 1, 3, 8; results/keep/internal_modes_bench_20261020_h100.csv),
# relative least squares over its 14 points; the P model is within 20 %, the F model up to 2.8x low at the largest
# F points; the dispatcher takes the faster route at every point but k = 3, K = 3 and 8 (F at most 1.3x slower, inside
# its factor-2 margin). Calibrated at B = 1 on the full outcome set only: batches and classes are the model's linear
# extrapolation, to be checked before a batched-register use relies on it. F0 and P0 are fits to the overhead-bound
# points (k = 2-3), not device constants.
# cpu: Apple Silicon Mac (arm64, 10 cores, torch 2.14.1 with 8 threads), 2026-10-20, the same fit over 9 points
# (k = 2-4, K = 1, 3, 8; results/keep/internal_modes_bench_20261020_mac_cpu.csv):
# the dispatcher takes the faster route at 8 of 9 (k = 2, K = 8: F, 1.8x slower than P', inside the factor-2 margin);
# the P model is up to 2.3x low at k = 3 (per-einsum overhead), the F model 14x low at k = 4, K = 8 (a 3 GB fine set,
# 93 s measured), where P' is taken anyway. Another CPU is uncalibrated in spirit: the constants are per machine.
# The weak route (`sample_apply`): t = CC0 + CCk2 k^2 + CC S units, units per stream = sum_j C(k, j) j L K_eff
# D X (d (1 + k - j) + j L) (`sample_apply_cost`); relative least squares, k^2 being the per-call Python / launch
# overhead of the chain (k steps over the subset tables). cuda: H100 PCIe, run 20261021_073232 (k = 2-12, K = 1, 3, 8,
# 1-10^4 streams, 92 rows; results/keep/internal_modes_ccspin_20261021_h100.csv): within 2.9x of every row, median
# 1.25x. cpu: the same Mac, 2026-10-21, k = 2-7, 1-1000 streams, 54 rows (results/keep/internal_modes_ccspin_20261021_mac_cpu.csv):
# within 3.0x, median 1.23x. A prediction, not a dispatch: sample_apply has no alternative route.
COST: dict[str, dict[str, float]] = {
    "cuda": {"F0": 1.29e-3, "F_build": 3.58e-12, "F_apply": 2.50e-12, "P0": 1.60e-3, "P": 2.25e-11,
             "CC0": 2.96e-3, "CCk2": 2.02e-4, "CC": 1.54e-11},
    "cpu": {"F0": 3.35e-4, "F_build": 3.93e-10, "F_apply": 5.46e-10, "P0": 3.75e-4, "P": 1.61e-9,
            "CC0": 5.6e-4, "CCk2": 7.34e-5, "CC": 1.05e-9},
}


class ModeSequenceKernel(SequenceKernel):
    """k emitters, some with internal modes, x n bins through one static m-mode interferometer (or one per bin).

    `K` (upper case) is the number of internal modes shared by the kernel (the largest of its emitters; a plain
    `BinEmitter` and an emitter with fewer modes use the first ones); `k` (lower case, inherited) is the number of
    emitters. `stage_ops(b)` gives every emitter's operators of bin b on the shared K modes.

    span_rtol: None (default) keeps every internal mode on which some operator of the bin is non-zero (exact); a
    float rotates to the singular basis of the bin's stacked operators and drops singular values below
    span_rtol * s_max (the channel changes by O((span_rtol s_max)^2)). max_bytes: the largest fine Kraus set the
    channel route builds (larger ticks: `outcome_law` / `condition` by the pairing route). chunk_bytes: working set of one
    propagation chunk.
    """

    def __init__(self, emitters, m: int, device: str = "cpu", dtype: torch.dtype = torch.complex128, mask=None,
                 indist=None, order: int | None = None, span_rtol: float | None = None, max_bytes: int = 4 << 30,
                 chunk_bytes: int = 1 << 30):
        kw = {} if mask is None else {"mask": mask}
        super().__init__(emitters, m, device=device, dtype=dtype, indist=indist, order=order, **kw)
        self.K = max(e.K if isinstance(e, ModeBinEmitter) else 1 for e in self.emitters)
        self.span_rtol, self.max_bytes, self.chunk_bytes = span_rtol, int(max_bytes), int(chunk_bytes)
        self._check_complete()
        self.lossy = any(isinstance(e, ModeBinEmitter) and e.lost is not None and e.lost.shape[1] > 0
                         for e in self.emitters)
        self.R = [e.R if isinstance(e, ModeBinEmitter) else 1 for e in self.emitters]
        self.multi_r = max(self.R) > 1
        self._span_m: dict = {}                       # R > 1: the joint span per bin (the ops are fixed)
        self._pats_lossy: np.ndarray | None = None
        self._in_modes = [torch.tensor(e.modes, dtype=torch.long, device=self.device) for e in self.emitters]
        self._stages: dict[tuple[int, int, int], object] = {}
        self._kraus_m: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._fine_m: dict[tuple, object] = {}
        self._kb_m: dict[tuple, int] = {}

    def _check_complete(self, atol: float = 1e-10):
        """Every emitter, every bin: sum_{l,k} A^dag A + sum_q N^dag N = 1 (the precision bounds of the channel route
        assume a trace-preserving stage).
        """
        for j, e in enumerate(self.emitters):
            a = e.ops.reshape(e.n, -1, e.d, e.d)                                                 # [n, L K, d, d]
            s = np.einsum("bqji,bqjk->bik", a.conj(), a)
            if isinstance(e, ModeBinEmitter) and e.lost is not None:
                s = s + np.einsum("bqji,bqjk->bik", e.lost.conj(), e.lost)
            dev = np.abs(s - np.eye(e.d)).max()
            if dev > atol:
                raise ValueError(f"emitter {j} is not trace-preserving (max |sum A^dag A - 1| = {dev:.2e} > {atol}); "
                                 "emitters with internal modes must be complete: an emission that is never detected "
                                 "goes into `lost` (source efficiency: `sequence_modes.source_loss`)")

    @property
    def patterns(self) -> np.ndarray:
        """The patterns of one bin, [S, m]: k photons over m modes; with `lost` operators (photons that can go
        undetected) the patterns of k, k - 1, ..., 0 detected photons, the k-photon block first (its indices are the
        lossless kernel's) and the all-zero pattern last.
        """
        if not getattr(self, "lossy", False):                    # (also during the base constructor)
            return super().patterns
        if self._pats_lossy is None:
            from .slos.basis import get_layers
            blocks = [super().patterns]
            for n in range(self.k - 1, 0, -1):
                occ = get_layers(n, self.m, self.mask, device="cpu").final_occupations()
                blocks.append(occ.cpu().numpy() if isinstance(occ, torch.Tensor) else np.asarray(occ))
            blocks.append(np.zeros((1, self.m), dtype=blocks[0].dtype))
            self._pats_lossy = np.concatenate(blocks)
        return self._pats_lossy

    @property
    def S(self) -> int:
        """Number of patterns of one bin (`patterns`): C(m + k - 1, k), or the sum over 0..k photons with `lost`."""
        return super().S if not getattr(self, "lossy", False) else self.patterns.shape[0]

    def _lost(self, b: int) -> list[torch.Tensor]:
        """Bin b's `lost` operators per emitter [Q_j, d_j, d_j] (Q_j = 0 for an emitter without them)."""
        out = []
        for e in self.emitters:
            if isinstance(e, ModeBinEmitter) and e.lost is not None:
                out.append(torch.as_tensor(e.lost[b], dtype=self.cdtype, device=self.device))
            else:
                out.append(torch.zeros((0, e.d, e.d), dtype=self.cdtype, device=self.device))
        return out

    def _presence(self, b: int):
        """The presence subsets of the no-photon branch: (present emitters, absent emitters) for every subset whose
        absent emitters all have `lost` operators (an emitter without them always emits into the network).
        """
        lost = self._lost(b)
        out = []
        for mask in range(2 ** self.k):
            P = [j for j in range(self.k) if mask >> j & 1]
            Ab = [j for j in range(self.k) if not mask >> j & 1]
            if all(lost[j].shape[0] for j in Ab):
                out.append((P, Ab))
        return out

    def stage_ops(self, b: int, rw: tuple[int, ...] | None = None) -> list[torch.Tensor]:
        """Operators of bin b, one [L_j, K, d_j, d_j] tensor per emitter on the shared K internal modes (zero-padded);
        for emitters with R > 1 unobserved emission outcomes, those of the r-word rw (one r per emitter; default
        all 0).
        """
        out = []
        for j, e in enumerate(self.emitters):
            r = 0 if rw is None else rw[j]
            a = e.ops_r(r)[b] if isinstance(e, ModeBinEmitter) else e.ops[b][:, None]          # [L, K_j, d, d]
            full = np.zeros((a.shape[0], self.K, *a.shape[2:]), dtype=np.complex128)
            full[:, :a.shape[1]] = a
            out.append(torch.as_tensor(full, dtype=self.cdtype, device=self.device))
        return out

    def _rwords(self, emitters=None) -> list[tuple[int, ...]]:
        """The r-words (one unobserved outcome per emitter; only the listed emitters vary, the others stay 0)."""
        js = range(self.k) if emitters is None else emitters
        words = [()]
        for j in range(self.k):
            words = [(*w, r) for w in words for r in (range(self.R[j]) if j in js else (0,))]
        return words

    def _ops(self, b: int, private: bool = True, rw: tuple[int, ...] | None = None) -> list[torch.Tensor]:
        """Bin b's operators on the populated internal span (`span`, span_rtol), per emitter [L_j, K_eff, d_j, d_j]
        (r-word rw, default all 0; the span is that of every r together, so K_eff does not depend on rw).
        With `indist` (x-model, docs/DISTINGUISHABILITY.md §3.4) and private=True, the internal space is extended by
        one private block per emitter: photon j is sqrt(x_j) A_j in the common modes (K_b) plus sqrt(1 - x_j) A_j in
        its own block (its own span, r_j modes; none when x_j = 1), so K_eff = K_b + sum_j r_j — the Gram
        S_jj' = sqrt(x_j x_j') of `SequenceKernel`'s species route, here as internal modes.
        """
        from .slos.internal import span
        rw = (0,) * self.k if rw is None else rw
        if not self.multi_r:
            ops = span(self.stage_ops(b), self.span_rtol)
            allr = [[a] for a in ops]
        else:                                         # one span over every (emitter, r) slice, then the word's slices
            if b not in self._span_m:                 # cached per bin: the R^k words reuse it
                sl = [(j, r) for j in range(self.k) for r in range(self.R[j])]
                red = span([self.stage_ops(b, tuple(r if i == j else 0 for i in range(self.k)))[j] for j, r in sl],
                           self.span_rtol)
                self._span_m[b] = [[red[i] for i, (jj, _) in enumerate(sl) if jj == j] for j in range(self.k)]
            allr = self._span_m[b]
            ops = [allr[j][rw[j]] for j in range(self.k)]
        if self.indist is None or not private:
            return ops
        key = ("priv", b)
        if not self.multi_r or key not in self._span_m:
            priv = []
            for j, x in enumerate(self.indist):       # private block: emitter j's own span (all its r together)
                if x >= 1.0:
                    priv.append([a[:, :0] for a in allr[j]])
                else:
                    priv.append(span(allr[j], self.span_rtol))
            if self.multi_r:
                self._span_m[key] = priv
        else:
            priv = self._span_m[key]
        priv = [p[rw[j]] for j, p in enumerate(priv)]
        out = []
        for j, (a, x) in enumerate(zip(ops, self.indist)):
            blocks = [math.sqrt(x) * a]
            for i, p in enumerate(priv):
                blocks.append(math.sqrt(1.0 - x) * p if i == j else a.new_zeros(a.shape[0], p.shape[1], *a.shape[2:]))
            out.append(torch.cat(blocks, 1))
        return out

    def _x(self) -> torch.Tensor | None:
        """The per-emitter overlaps x_j [k] of `indist` on the kernel's device (cached), or None."""
        if self.indist is None:
            return None
        if not hasattr(self, "_xs"):
            self._xs = torch.tensor(self.indist, dtype=torch.float64, device=self.device)
        return self._xs

    def _embed(self, j: int, A: torch.Tensor) -> torch.Tensor:
        """Emitter j's operators [..., d_j, d_j] on the product register [..., chi, chi] (1 on the other spins)."""
        left = int(np.prod(self.dims[:j])) if j else 1
        right = int(np.prod(self.dims[j + 1:])) if j + 1 < self.k else 1
        Il = torch.eye(left, dtype=A.dtype, device=A.device)
        Ir = torch.eye(right, dtype=A.dtype, device=A.device)
        out = torch.einsum("xy,...ij,uv->...xiuyjv", Il, A, Ir)
        return out.reshape(*A.shape[:-2], self.chi, self.chi)

    def _route_f_checks(self):
        if self.indist is not None and self.order is not None:
            raise NotImplementedError("internal modes: `order` (truncation at b distinguishable photons) is not "
                                      "implemented with internal-mode emitters")
        if self.mask != Mask.FOCK.value:
            raise NotImplementedError("internal modes: only the full Fock mask")

    def kraus(self, U, b: int, compress: float | None = None, route: str = "auto", tol: float = 1e-14,
              classes=None) -> torch.Tensor:
        """Physical Kraus operators of bin b, [Q, chi, chi] (final x initial): the fine set over (spatial x internal)
        patterns on the populated internal span, `coarse(U, b)` giving each operator's spatial pattern; the register
        update of pattern t is rho -> sum_{coarse[q] = t} K_q rho K_q^dag, and sum_q K_q^dag K_q = 1.
        compress=rtol: per outcome the Choi-rank set (singular values above rtol * s_max of the outcome).
        route (with compress): "F" compresses the fine set; "R" builds the compressed set without it, from range
        samples and the pairing route (`slos.internal.range_kraus`; cost ~ the Choi rank, independent of K; outcomes
        whose error estimate (a-posteriori, not a bound: measured >= 0.58 x the true error when the range misses a
        direction) exceeds tol on the normalised Choi matrix J_t / chi are recomputed by F, or raise when F does not
        fit max_bytes; deterministic for a given chunk_bytes, but equal to F's set only up to a unitary mixing of each
        outcome's operators; R returns the Choi-rank set when the spectrum's gap at rtol is above ~1e-15 lam_max (the
        normal case), otherwise fewer operators within tol); "auto": F when the fine set fits max_bytes, else R (R is
        not yet in the device cost model).
        classes [S] (int class id per pattern, -1 = dropped): one Kraus group per class instead of per pattern (route
        F; with compress, the class's Choi-rank set); `coarse(..., classes=)` then gives each operator's class id.
        """
        if classes is not None:
            return self._channel_classes(U, b, compress, route, classes)[0]
        return self._channel(U, b, compress, route, tol)[0]

    def coarse(self, U, b: int, compress: float | None = None, route: str = "auto", tol: float = 1e-14,
               classes=None) -> torch.Tensor:
        """Spatial pattern index [Q] (class id with `classes`) of every operator of `kraus(U, b, compress, route, tol,
        classes)`.
        """
        if classes is not None:
            return self._channel_classes(U, b, compress, route, classes)[1]
        return self._channel(U, b, compress, route, tol)[1]

    def _channel_classes(self, U, b, compress, route, classes):
        """Route F's Kraus set grouped by class (-1 dropped), compressed per class with compress."""
        from .slos.internal import compress_by_outcome
        if route not in ("auto", "F"):
            raise NotImplementedError("kraus(classes=): route F only")
        c = torch.as_tensor(np.asarray(classes), dtype=torch.long).reshape(-1)
        if c.numel() != self.S:
            raise ValueError(f"classes: one class id per pattern (S = {self.S})")
        bk = 0 if self._same_bins(U) else b
        key = (bk, compress, self.span_rtol, "classes", tuple(c.tolist()))
        K, coarse = self._channel(U, b, None, "F")
        if key not in self._kraus_m:
            cls = c.to(coarse.device)[coarse]
            keep = cls >= 0
            K, cls = K[keep], cls[keep]
            if compress is not None:
                K, cls = compress_by_outcome(K, cls, compress)
            self._kraus_m[key] = (K, cls)
        return self._kraus_m[key]

    def _channel(self, U, b, compress, route="auto", tol=1e-14):
        from .slos.internal import compress_by_outcome, fine_kraus, fine_stage
        self._route_f_checks()
        Ut = self._as_tensor(U).detach()
        if self._cache_U is None or self._cache_U.shape != Ut.shape or not torch.equal(self._cache_U, Ut.to(self._cache_U)):
            self._kraus_m, self._cache_U = {}, Ut.clone()
        bk = 0 if self._same_bins(U) else b
        if route not in ("auto", "F", "R"):
            raise ValueError("route: 'auto', 'F' or 'R'")
        if route == "R" and compress is None:
            raise ValueError("route 'R' builds the compressed set: pass compress=rtol")
        if route == "auto":
            fits = self._s_fine(self._kb(U, b)) * self.chi ** 2 * 16 * (4 if compress is not None else 1) <= self.max_bytes
            route = "R" if compress is not None and not fits and len(set(self.dims)) == 1 else "F"
        if route == "R":
            return self._channel_r(U, bk, compress, tol)
        key = (bk, compress, self.span_rtol)
        if key not in self._kraus_m:
            ops = self._ops(key[0])
            Kb = ops[0].shape[1]
            need = self._s_fine(Kb) * self.chi ** 2 * 16 * (4 if compress is not None else 1)  # compression: ~4 copies
            if need > self.max_bytes:                 # checked before the stage is built: its tables scale as S_fine
                raise NotImplementedError(f"internal modes: the fine Kraus set of this bin needs {need / 2 ** 30:.1f} GB "
                                          f"(> max_bytes); outcome_law / condition reach it by the pairing route")
            if self.lossy:
                K, coarse = self._fine_presence(U, key[0], Kb)
            else:
                stage = fine_stage(self.k, self.m, Kb, self.device)
                self._stages[(self.k, self.m, Kb)] = stage
                Ks = []
                for rw in self._rwords():                # R > 1: the fine set of every r-word (R = 1: one word)
                    ops = [self._embed(j, A) for j, A in enumerate(self._ops(key[0], rw=rw))]
                    Ks.append(fine_kraus(self._U(U, key[0]), ops, self._in_modes, stage, chunk_bytes=self.chunk_bytes))
                K = Ks[0] if len(Ks) == 1 else torch.cat(Ks)
                coarse = self.pattern_index(stage.spatial_occ).repeat(len(Ks))
            if self.indist is not None or self.lossy or self.multi_r:   # exact-zero operators dropped
                nz = K.flatten(1).ne(0).any(1)
                K, coarse = K[nz], coarse[nz]
            if compress is not None:
                K, coarse = compress_by_outcome(K, coarse, compress)
            if K.requires_grad:                       # keep the graph fresh: no caching under autograd (plain path)
                return K, coarse
            self._kraus_m[key] = (K, coarse)
        return self._kraus_m[key]

    def _fine_presence(self, U, bk, Kb):
        """The fine set with the no-photon branch (route F): for every presence subset P and every r-word of the
        present emitters, the fine operators of the |P| detected photons (a |P|-photon fine stage) times the absent
        emitters' `lost` operators (on their own spins, so the order of the products is immaterial; an absent emitter
        has no r). Returns (K [Q, chi, chi], coarse [Q] into the lossy `patterns`).
        """
        from .slos.internal import fine_kraus, fine_stage
        lost = [self._embed(j, N) for j, N in enumerate(self._lost(bk))]
        Ub = self._U(U, bk)
        eye = torch.eye(self.chi, dtype=self.cdtype, device=self.device).unsqueeze(0)
        zero = torch.as_tensor(self.pattern_index(np.zeros((1, self.m), dtype=np.int64)), device=self.device)
        Ks, cs = [], []
        for P, Ab in self._presence(bk):
            for rw in self._rwords(P):
                if P:
                    stage = fine_stage(len(P), self.m, Kb, self.device)
                    self._stages[(len(P), self.m, Kb)] = stage
                    ops = self._ops(bk, rw=rw)
                    K = fine_kraus(Ub, [self._embed(j, ops[j]) for j in P], [self._in_modes[j] for j in P], stage,
                                   chunk_bytes=self.chunk_bytes)
                    c = self.pattern_index(stage.spatial_occ)
                else:
                    K, c = eye, zero
                for j in Ab:
                    N = lost[j]
                    K = torch.einsum("qab,rbc->qrac", K, N).reshape(-1, self.chi, self.chi)
                    c = c.repeat_interleave(N.shape[0])
                Ks.append(K)
                cs.append(c)
        return torch.cat(Ks), torch.cat(cs)

    def _channel_r(self, U, bk, compress, tol):
        """The compressed set by route R (range samples + pairing core), outcomes above tol recomputed by F."""
        from .slos.internal import fine_counts, range_kraus
        if len(set(self.dims)) != 1:
            raise NotImplementedError("route R: emitters of equal spin dimension only")
        if self.lossy or self.multi_r:
            raise NotImplementedError("route R with `lost` operators or R > 1 unobserved emission outcomes: not "
                                      "implemented yet (it needs z-graded / r-word range samples); kraus(route='F') or "
                                      "outcome_law / condition by the pairing route")
        key = (bk, compress, self.span_rtol, "R", tol)
        if key not in self._kraus_m:
            A, inm, rows, tf = self._pairing_inputs(bk, private=True)            # x-model: private internal modes
            F_t = fine_counts(torch.as_tensor(np.asarray(self.patterns)), A.shape[2])
            K, t, cert, failed = range_kraus(self._U(U, bk).detach(), A, inm, rows, tf, F_t, compress, tol=tol,
                                             chunk_bytes=self.chunk_bytes)
            if bool(failed.any()):
                need = self._s_fine(A.shape[2]) * self.chi ** 2 * 16 * 4
                if need > self.max_bytes:
                    raise ValueError(f"route R: certificate {float(cert.max()):.1e} (over {int(failed.sum())} "
                                     f"outcome(s)) above tol {tol:.0e}, and the channel route needs "
                                     f"{need / 2 ** 30:.1f} GB > max_bytes; raise tol or max_bytes")
                KF, cF = self._channel(U, bk, compress, "F")       # one full F compression, for the failed few
                sel = failed[cF]
                K, t = torch.cat([K, KF[sel]]), torch.cat([t, cF[sel]])
                o = torch.argsort(t, stable=True)
                K, t = K[o], t[o]
            self._kraus_m[key] = (K, t)
        return self._kraus_m[key]

    # ---- outcome law and conditioned register (route F or the pairing route P') --------------------------------
    def _rho(self, rho):
        """[D] pure state, [D, D] density matrix or [B, D, D] batch -> ([B, D, D], batched)."""
        r = torch.as_tensor(rho).to(self.device, self.cdtype)
        if r.dim() == 1:
            return torch.outer(r, r.conj()).unsqueeze(0), False
        if r.dim() == 2:
            return r.unsqueeze(0), False
        if r.dim() == 3:
            return r, True
        raise ValueError("rho: a state [D], a density matrix [D, D] or a batch [B, D, D]")

    def _class(self, outcomes) -> torch.Tensor:
        """An outcome class -> sorted unique pattern indices: an index, a sequence of indices or a boolean mask [S]
        (occupation patterns: `pattern_index` first).
        """
        o = np.asarray(outcomes.cpu() if torch.is_tensor(outcomes) else outcomes)
        if o.dtype == bool:
            if o.shape != (self.S,):
                raise ValueError(f"a boolean outcome mask has S = {self.S} entries")
            idx = np.nonzero(o)[0]
        else:
            idx = o.reshape(-1).astype(np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.S):
            raise ValueError("outcome index out of range")
        return torch.as_tensor(np.unique(idx), dtype=torch.long, device=self.device)

    def _pairing_inputs(self, b: int, private: bool = False, rw: tuple[int, ...] | None = None):
        """Local operators A [n, L, K_b, d, d] (labels zero-padded), input columns [n, L], event rows [S, n], t! [S].
        With `indist`, private=False leaves the x-model to the pair Gram's scaling (`pairing_apply(x=)`), private=True
        extends the internal space by the private modes (`_ops`). rw: the r-word (R > 1), default all 0.
        """
        if len(set(self.dims)) != 1:
            raise NotImplementedError("pairing route: emitters of equal spin dimension only")
        ops = self._ops(b, private, rw)
        L = max(a.shape[0] for a in ops)
        A = torch.stack([torch.cat([a, a.new_zeros((L - a.shape[0], *a.shape[1:]))]) for a in ops])
        inm = torch.stack([torch.cat([im, im.new_zeros(L - im.shape[0])]) for im in self._in_modes])
        if self.lossy:                                # rows of unequal length: grouped by photon number in _apply_p
            return A, inm, None, None
        if not hasattr(self, "_rows"):
            pats = self.patterns
            self._rows = torch.as_tensor(np.stack([np.repeat(np.arange(self.m), p) for p in pats]), device=self.device)
            self._tfact = torch.as_tensor([math.prod(math.factorial(int(x)) for x in p) for p in pats],
                                          dtype=torch.float64, device=self.device)
        return A, inm, self._rows, self._tfact

    def _s_fine(self, Kb: int) -> int:
        """S_fine = C(m K_b + k - 1, k), without building the stage (whose host tables scale as S_fine); with `lost`,
        the sum over the presence subsets of C(m K_b + |P| - 1, |P|) x the absent emitters' `lost` counts (taken from
        bin 0: `lost` is [n, Q, d, d], the same Q in every bin).
        """
        if not self.lossy:
            return math.comb(self.m * Kb + self.k - 1, self.k) * math.prod(self.R)           # x the r-words
        q = [N.shape[0] for N in self._lost(0)]
        return sum(math.comb(self.m * Kb + len(P) - 1, len(P)) * math.prod(q[j] for j in Ab)
                   * math.prod(self.R[j] for j in P) for P, Ab in self._presence(0))

    def _kb(self, U, b) -> int:
        """K_b of bin b's populated internal span, with the private modes of `indist` (route F's K_eff; cached per bin
        key and span_rtol: the ops are fixed).
        """
        key = (0 if self._same_bins(U) else b, self.span_rtol)
        if key not in self._kb_m:
            self._kb_m[key] = self._ops(key[0])[0].shape[1]
        return self._kb_m[key]

    def _fine(self, U, b):
        """The fine stage of bin b's populated internal span (cached per bin key and span_rtol: the ops are fixed).
        Callers check max_bytes on _s_fine first.
        """
        from .slos.internal import fine_stage
        key = (0 if self._same_bins(U) else b, self.span_rtol)
        if key not in self._fine_m:
            self._fine_m[key] = fine_stage(self.k, self.m, self._kb(U, b), self.device)
        return self._fine_m[key]

    def _f_fallback(self, U, b, rho, idx, cert, tol, law=False, summed=False, weights=None):
        """Route F for outcomes whose pairing certificate exceeded tol; ValueError when F does not fit max_bytes."""
        need = self._s_fine(self._kb(U, b)) * self.chi ** 2 * 16
        if need > self.max_bytes:
            raise ValueError(f"pairing route: certificate {float(cert.max()):.1e} (over {idx.numel()} outcome(s), "
                             f"summed for a class) above tol {tol:.0e}, and the channel route needs "
                             f"{need / 2 ** 30:.1f} GB > max_bytes; raise tol or max_bytes (a class's certificates "
                             "add up: large classes need a larger tol)")
        return self._apply_f(U, b, rho, idx, law, summed, weights)

    def _route(self, U, b, n_out: int, B: int) -> str:
        """The cheaper route for n_out outcomes of a B-batch: the pairing route only when its modelled time is below
        half the channel route's (device constants of `COST`; an uncalibrated device keeps the channel route unless
        the fine set does not fit).
        """
        S_fine = self._s_fine(self._kb(U, b))
        D, n = self.chi, self.k
        f_fits = S_fine * D * D * 16 <= self.max_bytes
        if len(set(self.dims)) != 1:
            return "F"
        if not f_fits:
            return "P"
        c = COST.get(self.device.type)
        if c is None:
            return "F"
        cached = (0 if self._same_bins(U) else b, None, self.span_rtol) in self._kraus_m
        t_f = (c.get("F0", 0.0) + (0.0 if cached else c["F_build"] * n * S_fine * D ** 3)
               + c["F_apply"] * S_fine * B * D ** 3 * n_out / self.S)
        d = self.dims[0]
        t_p = (c.get("P0", 0.0) + c["P"] * n_out * 4 ** (n - 1) * n * n * d * d * D * D * B
               * (n + 1 if self.lossy else 1) * math.prod(self.R))                         # degree axis, r-words
        return "P" if 2 * t_p < t_f else "F"

    def _apply_f(self, U, b, rho, idx, law=False, summed=False, weights=None):
        """E_t(rho) [B, len(idx), D, D] for the outcomes idx through the fine Kraus set (law: Tr E_t(rho) [B, len(idx)],
        without forming the matrices; summed: the class sum [B, 1, D, D] in one contraction, no per-operator terms;
        weights [len(idx)] with summed: the weighted class sum sum_t w_t E_t(rho)).
        """
        K, coarse = self._channel(U, b, None)
        pos = torch.full((self.S,), -1, dtype=torch.long, device=self.device)
        pos[idx] = torch.arange(idx.numel(), device=self.device)
        sel = pos[coarse] >= 0
        Ks, ps = K[sel], pos[coarse[sel]]
        if summed and weights is not None:
            wq = weights.to(device=self.device, dtype=K.dtype)[ps]
            return torch.einsum("q,qab,zbc,qdc->zad", wq, Ks, rho, Ks.conj()).unsqueeze(1)
        if summed:
            return torch.einsum("qab,zbc,qdc->zad", Ks, rho, Ks.conj()).unsqueeze(1)
        if law:
            p = torch.einsum("qab,zbc,qac->zq", Ks, rho, Ks.conj()).real
            return torch.zeros((rho.shape[0], idx.numel()), dtype=p.dtype, device=p.device).index_add_(1, ps, p)
        E = torch.einsum("qab,zbc,qdc->zqad", Ks, rho, Ks.conj())
        return torch.zeros((rho.shape[0], idx.numel(), self.chi, self.chi), dtype=E.dtype, device=E.device).index_add_(1, ps, E)

    def _apply_p(self, U, b, rho, idx):
        """(E_t(rho) [B, len(idx), D, D], certificate [B, len(idx)]) by the pairing route. With R > 1 unobserved
        emission outcomes: the sum over the r-words (prod_j R_j pairing evaluations; certificates add up), the
        absent-photon superoperator Lam_j scaled by 1 / R_j so that the sum over r_j of a word with photon j absent
        counts Lam_j once (r_j appears nowhere else in such a term).
        """
        from .slos.internal import lost_superop
        Lam = None
        if self.lossy:                                # the same for every r-word: built once
            bk = 0 if self._same_bins(U) else b
            Lam = torch.stack([lost_superop(N) / R for N, R in zip(self._lost(bk), self.R)])   # [n, d, d, d, d]
        if not self.multi_r:
            return self._apply_p_word(U, b, rho, idx, None, Lam)
        E, cert = None, None
        for rw in self._rwords():
            Ew, cw = self._apply_p_word(U, b, rho, idx, rw, Lam)
            E, cert = (Ew, cw) if E is None else (E + Ew, cert + cw)
        return E, cert

    def _emission_A(self, bk) -> torch.Tensor:
        """Local operators [n, L', K_b, d, d] for the emission channel: every r of an emitter stacked along the label
        axis (the emission channel sums over labels, internal modes and r alike); zero-padded.
        """
        if not self.multi_r:
            return self._pairing_inputs(bk)[0]
        per = [torch.cat([self._ops(bk, False, tuple(r if i == j else 0 for i in range(self.k)))[j]
                          for r in range(self.R[j])]) for j in range(self.k)]
        L = max(a.shape[0] for a in per)
        return torch.stack([torch.cat([a, a.new_zeros((L - a.shape[0], *a.shape[1:]))]) for a in per])

    def _apply_p_word(self, U, b, rho, idx, rw, Lam=None):
        """`_apply_p` for one r-word (None: R = 1); Lam: the absent-photon superoperators (with `lost`)."""
        from .slos.internal import pairing_apply, pairing_apply_lost
        bk = 0 if self._same_bins(U) else b
        A, inm, rows, tf = self._pairing_inputs(bk, rw=rw)
        if not self.lossy:
            return pairing_apply(self._U(U, b), A, inm, rows[idx], tf[idx], rho, chunk_bytes=self.chunk_bytes,
                                 x=self._x())
        groups = self._lossy_groups()
        cnt = groups["count"][idx]                                                               # |t| of each outcome
        E = torch.empty((rho.shape[0], idx.numel(), self.chi, self.chi), dtype=rho.dtype, device=rho.device)
        cert = torch.empty((rho.shape[0], idx.numel()), dtype=rho.real.dtype, device=rho.device)
        for npr in torch.unique(cnt).tolist():                                                   # one group per |t|
            p_t = torch.nonzero(cnt == npr).reshape(-1)
            loc = groups["local"][idx[p_t]]                                                      # row in the group
            rows_g, tf_g = groups[npr]
            Eg, cg = pairing_apply_lost(self._U(U, b), A, inm, rows_g[loc], tf_g[loc], rho, Lam,
                                        chunk_bytes=self.chunk_bytes, x=self._x())
            E[:, p_t], cert[:, p_t] = Eg, cg
        return E, cert

    def _lossy_groups(self) -> dict:
        """The lossy patterns grouped by detected photon number (built once): "count" [S] |t| of each pattern,
        "local" [S] its row within its group, and per |t| the event rows [S_|t|, |t|] and t! [S_|t|].
        """
        if not hasattr(self, "_groups"):
            pats = self.patterns
            cnt = pats.sum(1)
            local = np.zeros(len(pats), dtype=np.int64)
            g = {"count": torch.as_tensor(cnt, dtype=torch.long, device=self.device)}
            for npr in np.unique(cnt).tolist():
                pos = np.nonzero(cnt == npr)[0]
                local[pos] = np.arange(len(pos))
                rows = np.stack([np.repeat(np.arange(self.m), p) for p in pats[pos]]).reshape(len(pos), npr)
                tf = [math.prod(math.factorial(int(x)) for x in p) for p in pats[pos]]
                g[npr] = (torch.as_tensor(rows, dtype=torch.long, device=self.device),
                          torch.as_tensor(tf, dtype=torch.float64, device=self.device))
            g["local"] = torch.as_tensor(local, device=self.device)
            self._groups = g
        return self._groups

    @staticmethod
    def _over(cert, tol, rho, ref):
        """Whether a class certificate [B] exceeds tol: against tol Tr rho per batch row (ref None), or, for the
        X^2 blocks of registers with a passive rest (ref = (Tr of each whole register [B0], X^2)), the blocks'
        certificates of one register combined in Frobenius norm (the error matrix of the whole register) against tol
        x its trace.
        """
        if ref is None:
            return bool((cert > tol * torch.einsum("bii->b", rho).real.abs()).any())
        tr_g, G = ref
        return bool((cert.reshape(-1, G).pow(2).sum(1).sqrt() > tol * tr_g).any())

    def _outcomes(self, U, b, rho, idx, tol, route, summed=False, law=False, ref=None):
        """E_t(rho) for the outcomes idx with the requested / chosen route; pairing results whose certificate exceeds
        tol (trace norm, sqrt(D) x the Frobenius bound, against tol Tr rho) are recomputed by the channel route.
        summed: the caller sums the results over idx (a class), so the certificates add up, the sum is compared with
        tol Tr rho in Frobenius norm (>= the operator norm of the criterion) and the class falls back as a whole.
        law: return Tr E_t(rho) [B, len(idx)] instead (route F then never forms the matrices).
        """
        self._route_f_checks()
        r = self._route(U, b, idx.numel(), rho.shape[0]) if route == "auto" else route
        if r == "F":                                  # a class: one contraction to its sum [B, 1, D, D]
            return self._apply_f(U, b, rho, idx, law, summed and not law), "F"
        if r != "P":
            raise ValueError("route: 'auto', 'F' or 'P'")
        E, cert = self._apply_p(U, b, rho, idx)
        if law:
            E = torch.einsum("btii->bt", E).real
        tr = torch.einsum("bii->b", rho).real.abs().unsqueeze(1)
        if summed:                                    # a class (condition): Frobenius >= operator norm, the criterion
            cert = cert.sum(1, keepdim=True)
            if self._over(cert[:, 0], tol, rho, ref):  # the class falls back as a whole, to F's direct class sum
                return self._f_fallback(U, b, rho, idx, cert, tol, law, not law), "F"
            return E, "P"
        bad = (math.sqrt(self.chi) * cert > tol * tr).any(0)
        if bool(bad.any()):
            E[:, bad] = self._f_fallback(U, b, rho, idx[bad], cert, tol, law)
        return E, "P"

    def outcome_law(self, U, b: int, rho, tol: float = 1e-14, route: str = "auto") -> torch.Tensor:
        """p(t) = Tr E_t(rho) over the S spatial patterns of bin b: [S] (or [B, S] for a batch [B, D, D]).
        rho: a register state [D], a density matrix [D, D] or a batch [B, D, D]. route: "auto" (the cheaper of the
        channel route F and the pairing route P', by the device cost model), "F" or "P"; pairing results above tol
        are recomputed by F. Gradients: route F builds and caches its Kraus set from a detached U (no d/dU, as
        `kraus`); the pairing route is differentiable in U — with route="auto" the dispatcher decides, so pass
        route="P" when d/dU is needed.
        """
        r, batched = self._rho(rho)
        idx = torch.arange(self.S, device=self.device)
        p, _ = self._outcomes(U, b, r, idx, tol, route, law=True)
        return p if batched else p[0]

    def condition(self, U, b: int, rho, outcomes, tol: float = 1e-14, route: str = "auto", rest: int = 1,
                  weights=None) -> torch.Tensor:
        """E_C(rho) = sum_{t in C} E_t(rho), the register conditioned on the outcome class C of bin b (not normalised:
        its trace is the class probability). outcomes: a pattern index, a sequence of indices or a boolean mask [S]
        (occupation patterns: `pattern_index` first). The class is certified as a whole: its certificates add up and
        their sum (Frobenius norm, which bounds the operator norm of the criterion) is compared with tol Tr rho — so
        here tol bounds the operator (Frobenius) norm of the error, while `outcome_law` bounds each outcome's trace
        norm (sqrt(D) x Frobenius), a stricter test per outcome by up to sqrt(D); above
        tol it is recomputed by F, directly. So a class of more than about tol / (per-outcome certificate, ~1e-16 to
        1e-15) outcomes leaves the pairing route — and raises when F does not fit `max_bytes`: pass a larger tol.
        Route F sums the class directly. By the pairing route, a class larger than half the patterns is computed as
        the emission channel minus its complement (the channels of all patterns sum to the emission channel when U is
        unitary; a non-unitary U takes the direct sum). rho, tol, route as in `outcome_law`.
        rest=X: rho is a register [D X, D X] (or a batch [B, D X, D X]) whose X passive dimensions belong to spins
        outside the kernel (kernel spins first, most significant): returns (E_C (x) id_X)(rho). The X^2 blocks
        rho[(., x), (., y)] go through the routes as a batch; their certificates are combined per register (Frobenius)
        and compared with tol Tr rho of the whole register (an off-diagonal block has no trace of its own).
        weights (w >= 0; one per entry of `outcomes`, or [S]; a length equal to both is read per entry): the weighted
        class E_w(rho) = sum_t w_t E_t(rho) (e.g. a
        detector-noise predecessor sum_o T(o'|o) E_o); route F contracts the weighted operators, the pairing route sums
        the weighted results with the certificate sum_t w_t cert_t; the complement path is used for unweighted classes
        only.
        """
        idx, w = self._class_weights(outcomes, weights)
        if rest > 1:
            R0 = torch.as_tensor(rho).to(self.device, self.cdtype)
            one = R0.dim() == 2
            R0 = R0.unsqueeze(0) if one else R0
            B0, D, X = R0.shape[0], self.chi, int(rest)
            if R0.shape[1:] != (D * X, D * X):
                raise ValueError(f"rest={X}: rho must be [{D * X}, {D * X}] (or a batch of them)")
            blocks = R0.reshape(B0, D, X, D, X).permute(0, 2, 4, 1, 3).reshape(B0 * X * X, D, D)
            ref = (torch.einsum("bii->b", R0).real.abs(), X * X)
            out = self._condition(U, b, blocks, idx, tol, route, ref, w)
            out = out.reshape(B0, X, X, D, D).permute(0, 3, 1, 4, 2).reshape(B0, D * X, D * X)
            return out[0] if one else out
        r, batched = self._rho(rho)
        out = self._condition(U, b, r, idx, tol, route, None, w)
        return out if batched else out[0]

    def _class_weights(self, outcomes, weights):
        """(sorted unique indices [T], weights [T] float64 or None) of a class and its optional weights (per entry of
        `outcomes`, repeated entries adding up, or one per pattern [S]).
        """
        if weights is None:
            return self._class(outcomes), None
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).cpu()
        if bool((w < 0).any()):
            raise ValueError("weights must be >= 0")
        full = torch.zeros(self.S, dtype=torch.float64)
        o = np.asarray(outcomes.cpu() if torch.is_tensor(outcomes) else outcomes)
        if o.dtype == bool:
            if o.shape != (self.S,):
                raise ValueError(f"a boolean outcome mask has S = {self.S} entries")
            o = np.nonzero(o)[0]
        o = o.reshape(-1).astype(np.int64)
        self._class(o)                                                                           # range check
        if w.numel() == o.shape[0]:                   # one weight per entry (repeated entries add up)
            full.index_add_(0, torch.as_tensor(o), w)
        elif w.numel() == self.S:                     # one weight per pattern, restricted to the class
            full[torch.as_tensor(o)] = w[torch.as_tensor(o)]
        else:
            raise ValueError(f"weights: one per outcome ({o.shape[0]}) or one per pattern ({self.S})")
        idx = torch.nonzero(full).reshape(-1)
        return idx.to(self.device), full[idx].to(self.device)

    def _condition(self, U, b, r, idx, tol, route, ref, w=None):
        """`condition` on a batch r [B, D, D] (ref: `_over`; w: class weights, or None)."""
        from .slos.internal import emission_channel
        self._route_f_checks()
        if w is not None:                             # a weighted class: the direct (weighted) sum, no complement
            rt = self._route(U, b, idx.numel(), r.shape[0]) if route == "auto" else route
            if rt == "F":
                return self._apply_f(U, b, r, idx, summed=True, weights=w)[:, 0]
            if rt != "P":
                raise ValueError("route: 'auto', 'F' or 'P'")
            E, cert = self._apply_p(U, b, r, idx)
            out = torch.einsum("t,btxy->bxy", w.to(E.dtype), E)
            cert = (cert * w.to(cert.dtype)).sum(1)
            if self._over(cert, tol, r, ref):
                out = self._f_fallback(U, b, r, idx, cert, tol, summed=True, weights=w)[:, 0]
            return out
        comp = idx.numel() > self.S // 2 and len(set(self.dims)) == 1
        if comp:
            Ub = self._U(U, b).detach()
            eye = torch.eye(self.m, dtype=Ub.dtype, device=Ub.device)
            comp = bool(torch.allclose(Ub.conj().T @ Ub, eye, atol=self._unit_tol(), rtol=0))
        n_out = self.S - idx.numel() if comp else idx.numel()
        rt = self._route(U, b, n_out, r.shape[0]) if route == "auto" else route
        if rt == "P" and comp:
            keep = torch.ones(self.S, dtype=torch.bool, device=self.device)
            keep[idx] = False
            other = torch.nonzero(keep).reshape(-1)
            bk = 0 if self._same_bins(U) else b
            E_tot, cert = emission_channel(self._emission_A(bk), r, self._lost(bk) if self.lossy else None)
            out = E_tot
            if other.numel():
                E_rest, c_rest = self._apply_p(U, b, r, other)
                out, cert = E_tot - E_rest.sum(1), cert + c_rest.sum(1)
            if self._over(cert, tol, r, ref):
                out = self._f_fallback(U, b, r, idx, cert, tol, summed=True)[:, 0]   # the class directly, exact F
        else:
            E, _ = self._outcomes(U, b, r, idx, tol, rt, summed=True, ref=ref)
            out = E.sum(1)
        return out

    # ---- the weak route: one tick of pure-state trajectories (CC-spin, docs/SEQUENCE.md) -----------------------
    def sample_apply_cost(self, U, b: int, streams: int, X: int = 1) -> float | None:
        """Predicted seconds of one `sample_apply` tick of `streams` streams with a rest X on this device (`COST`
        "CC*" entries; None on an uncalibrated device): CC0 + CCk2 k^2 + CC streams units, units = sum_j C(k, j) j L
        K_eff D X (d (1 + k - j) + j L) with all k photons present (the R = 1, no-loss case the bench measured).
        With `lost` or R > 1 (batch sizing): the all-present units bound the arithmetic of a stream (fewer present
        photons cost less), but every distinct branch word in the batch is its own chain call, each paying the
        per-call terms CC0 + CCk2 k'^2 (k' its present photons).
        """
        c = COST.get(self.device.type)
        if c is None or "CC" not in c:
            return None
        K = self._kb(U, b)
        L = max(len(e.modes) for e in self.emitters)
        d = max(self.dims)
        units = sum(math.comb(self.k, j) * j * L * K * self.chi * X * (d * (1 + self.k - j) + j * L)
                    for j in range(1, self.k + 1))
        return c["CC0"] + c["CCk2"] * self.k ** 2 + c["CC"] * streams * units

    def _unit_tol(self) -> float:
        """The unitarity tolerance of the kernel's dtype, 1e3 eps (2.2e-13 for complex128): the identities that need U
        unitary (the complement path of `condition`, the chain of `sample_apply`).
        """
        return 1e3 * torch.finfo(torch.empty(0, dtype=self.cdtype).real.dtype).eps

    @property
    def patterns_all(self) -> np.ndarray:
        """The patterns of k, k - 1, ..., 0 detected photons (k-photon block first): `patterns` for a kernel with
        `lost` operators; for any kernel, the outcome list of `sample_apply(..., eta_out=)` (detection loss).
        """
        if self.lossy:
            return self.patterns
        if not hasattr(self, "_pats_all"):
            from .slos.basis import get_layers
            blocks = [self.patterns]
            for n in range(self.k - 1, 0, -1):
                occ = get_layers(n, self.m, self.mask, device="cpu").final_occupations()
                blocks.append(occ.cpu().numpy() if isinstance(occ, torch.Tensor) else np.asarray(occ))
            blocks.append(np.zeros((1, self.m), dtype=blocks[0].dtype))
            self._pats_all = np.concatenate(blocks)
        return self._pats_all

    def _branch_ops(self, bk):
        """Per emitter: the branch operators G [R_j + Q_j, d, d] (M_r = sum_{l,k} A_r^dag A_r for each unobserved
        outcome r, then N_q^dag N_q for each `lost` operator q) and the M_r themselves [R_j, d, d].
        """
        lost = self._lost(bk)
        G, Ms = [], []
        for j in range(self.k):
            Mj = []
            for r in range(self.R[j]):
                A = self._ops(bk, rw=tuple(r if i == j else 0 for i in range(self.k)))[j]           # [L, K_eff, d, d]
                Mj.append(torch.einsum("lkba,lkbc->ac", A.conj(), A))
            Mj = torch.stack(Mj)
            Nj = lost[j]
            G.append(torch.cat([Mj, torch.einsum("qba,qbc->qac", Nj.conj(), Nj)]) if Nj.shape[0] else Mj)
            Ms.append(Mj)
        return G, Ms

    def _on_spin(self, j, ops, x):
        """Per-stream operators ops [S, d_j, d_j] on spin j of x [S, D, X]."""
        left = int(np.prod(self.dims[:j])) if j else 1
        right = int(np.prod(self.dims[j + 1:])) if j + 1 < self.k else 1
        y = x.reshape(x.shape[0], left, self.dims[j], right, x.shape[2])
        return torch.einsum("sab,sxbyz->sxayz", ops, y).reshape(x.shape)

    def _draw_branch(self, bk, p, unif, force):
        """The branch word per stream (drawn before the chain): sequentially over the emitters, b_j ~
        <phi| G_{j,b} |phi> / <phi|phi> on spin j, phi <- G_{j,b_j}^(1/2) phi (operators on different spins commute,
        so the product of the conditionals is <psi| (x)_j G_{j,b_j} |psi>); then psi_b = (x)_{lost j} N_{j,q_j} psi.
        Returns (branch [S, k]: r < R_j, or R_j + q for a lost photon; psi_b [S, D, X]; prob [S]; M_r; the largest
        relative deviation of sum_b <phi| G_{j,b} |phi> = <phi|phi> over the emitters (completeness)).
        """
        G, Ms = self._branch_ops(bk)
        S = p.shape[0]
        phi = p
        branch = torch.empty((S, self.k), dtype=torch.long)
        prob = torch.ones(S, dtype=torch.float64)
        bdev = 0.0
        for j in range(self.k):
            Gj = G[j]                                                                            # [B, d, d]
            Bj = Gj.shape[0]
            w = torch.stack([(phi.conj() * self._on_spin(j, Gj[bb].expand(S, -1, -1), phi)).real.sum((1, 2))
                             for bb in range(Bj)], 1).to(torch.float64).cpu()                    # [S, B]
            nphi = phi.abs().pow(2).sum((1, 2)).to(torch.float64).cpu()                          # completeness:
            bdev = max(bdev, float(((w.sum(1) - nphi).abs() / nphi.clamp_min(1e-300)).max()))  # sum_b G_b = 1
            pr = w / w.sum(1, keepdim=True).clamp_min(1e-300)
            if force is not None:
                bj = force[:, j].cpu()
            else:
                cdf = pr.clamp_min(0).cumsum(1)
                bj = torch.searchsorted(cdf, (unif[:, j] * cdf[:, -1]).unsqueeze(1).contiguous(), right=True)[:, 0]
                bj = bj.clamp(max=Bj - 1)
            branch[:, j] = bj
            prob *= pr[torch.arange(S), bj]
            lam, V = torch.linalg.eigh(Gj)                                                       # G^(1/2), PSD
            sq = (V * lam.clamp_min(0).sqrt().unsqueeze(-2)) @ V.conj().transpose(-1, -2)
            phi = self._on_spin(j, sq[bj.to(sq.device)], phi)
        psi_b = p
        lost = self._lost(bk)
        for j in range(self.k):
            gone = torch.nonzero(branch[:, j] >= self.R[j]).reshape(-1)
            if gone.numel():
                ops = lost[j][(branch[gone, j] - self.R[j]).to(lost[j].device)]
                psi_b = psi_b.clone() if psi_b is p else psi_b
                psi_b[gone.to(p.device)] = self._on_spin(j, ops, psi_b[gone.to(p.device)])
        return branch, psi_b, prob, Ms, bdev

    def sample_apply(self, U, b: int, psi, generator: torch.Generator | None = None, tol_draw: float = 1e-10,
                     U_streams: torch.Tensor | None = None, chunk_bytes: int | None = None, eta_out=None,
                     index: bool = True, _force=None):
        """Draw the detected pattern of bin b for every stream and apply its fine Kraus operator: psi [S, D] or
        [S, D, X] (the kernel's spins, first emitter most significant, then a passive rest X), normalised per stream.
        Returns (pattern index [S] into `patterns`, psi' [S, D, X] normalised, info) — an exact unravelling of the
        tick's channel (the internal mode of each event is recorded: info["events"] [S, k, 2] (row, internal mode) in
        draw order, -1 for photons not detected). The operator-valued Clifford-Clifford chain over the detected
        photons: cost linear in the populated internal modes (with `indist`, the private modes of the x-model), memory
        C(k, j) D X per stream and step.
        Branches: with `lost` operators or R > 1 unobserved outcomes, each emitter's branch (outcome r, or
        lost q) is drawn first (info["branch"] [S, k]: r, or R_j + q), the absent photons' N_q applied, and the chain
        runs over the present photons with the trace weight M_{j,r} of the photons not yet detected.
        eta_out [m] (optional): a detection efficiency per output mode — U is dilated to [[diag(sqrt(eta)) U],
        [diag(sqrt(1 - eta)) U]] (2m candidate rows, an isometry on the inputs) and events on the loss rows are not
        reported (info["events"] keeps them: rows m..2m-1); the pattern index then refers to `patterns_all` (0..k
        detected photons). info["occ"] [S, m]: the detected occupations themselves; index=False (or a basis whose key
        (k + 1)^m exceeds int64, e.g. k = 10, m = 20, where the pattern list is large anyway) returns None as the
        index — use info["occ"].
        generator: the CPU generator of the draws (float64 uniforms drawn first, k per stream, 2k with branches; each
        draw an inverse CDF of the float64 weights on the host: the same seed gives the same draws whatever the
        chunking, and on CUDA to the rare flip at a weight boundary). U_streams [S, m, m]: one unitary per stream
        instead of the bin's. tol_draw: the per-step identity sum_e W_e = (n' - j + 1) x the previous total (U
        unitary, emitters complete) is checked to this relative tolerance (info["deviation"] [k]).
        """
        from .slos.internal import ccspin_draw
        self._route_f_checks()
        p = torch.as_tensor(psi).to(self.device, self.cdtype)
        p = p.unsqueeze(-1) if p.dim() == 2 else p
        if p.dim() != 3 or p.shape[1] != self.chi:
            raise ValueError(f"psi: [S, {self.chi}] or [S, {self.chi}, X]")
        nrm = p.reshape(p.shape[0], -1).norm(dim=1)
        if float((nrm - 1).abs().max()) > 1e-10:
            raise ValueError("sample_apply: psi must be normalised per stream (| |psi| - 1 | <= 1e-10)")
        Ub = (torch.as_tensor(U_streams) if U_streams is not None else self._U(U, b)).to(self.device, self.cdtype)
        Uc = Ub.detach() if Ub.dim() == 3 else Ub.detach().unsqueeze(0)
        eye = torch.eye(self.m, dtype=Uc.dtype, device=Uc.device)
        if float((Uc.conj().transpose(1, 2) @ Uc - eye).abs().max()) > self._unit_tol():
            raise ValueError(f"sample_apply: U must be unitary (|U^dag U - 1| <= {self._unit_tol():.1e}): the chain traces the "
                             "photons not yet detected over every output")
        if eta_out is not None:                       # detection loss: the dilated isometry, loss rows m..2m-1
            e = torch.as_tensor(eta_out, dtype=torch.float64).reshape(-1)
            if e.numel() != self.m or bool(((e < 0) | (e > 1)).any()):
                raise ValueError(f"eta_out: {self.m} efficiencies in [0, 1]")
            se = e.sqrt().to(self.device, self.cdtype).view(-1, 1)
            sl = (1 - e).sqrt().to(self.device, self.cdtype).view(-1, 1)
            Ub = torch.cat([se * Ub, sl * Ub], -2)
        bk = 0 if self._same_bins(U) else b
        S, k, cb = p.shape[0], self.k, chunk_bytes or self.chunk_bytes
        branchy = self.lossy or self.multi_r
        if branchy:
            unif = torch.rand((S, 2 * k), generator=generator, dtype=torch.float64)          # branches, then the chain
            fb, fe = (None, None) if _force is None else _force
            branch, psi_b, bprob, Ms, bdev = self._draw_branch(bk, p, unif[:, :k], fb)
            u_chain = unif[:, k:]
            words, inv = torch.unique(branch, dim=0, return_inverse=True)
        else:
            unif = torch.rand((S, k), generator=generator, dtype=torch.float64)              # one per stream and step
            fe, psi_b, bprob, u_chain, bdev = _force, p, torch.ones(S, dtype=torch.float64), unif, 0.0
            branch = torch.zeros((S, k), dtype=torch.long)
            words, inv = branch[:1], torch.zeros(S, dtype=torch.long)
        events = torch.full((S, k), -1, dtype=torch.long)
        out = torch.empty_like(p)
        prob = bprob.clone()
        dev = torch.zeros(k, dtype=torch.float64)
        K = None
        for w_i in range(words.shape[0]):
            w = words[w_i].tolist()
            sel = torch.nonzero(inv == w_i).reshape(-1)
            present = [j for j in range(k) if w[j] < self.R[j]]
            rw = tuple(w[j] if w[j] < self.R[j] else 0 for j in range(k))
            ops_w = self._ops(bk, rw=rw)
            K = ops_w[0].shape[1]
            psel = psi_b[sel.to(p.device)]
            if not present:                           # every photon lost: no event, the register is psi_b
                out[sel.to(p.device)] = psel / psel.reshape(sel.numel(), -1).norm(dim=1).clamp_min(1e-300).view(-1, 1, 1)
                continue
            ops_p = [ops_w[j] for j in present]
            M_p = [Ms[j][w[j]] for j in present] if branchy else None
            L = max(a.shape[0] for a in ops_p)
            n1 = len(present)
            per_stream = max(math.comb(n1, j) * j for j in range(1, n1 + 1)) * L * K * self.chi * p.shape[2] * 64
            Sc = max(1, int(cb // max(1, per_stream)))                                           # streams per chunk
            for s0 in range(0, sel.numel(), Sc):
                ss = sel[s0:s0 + Sc]
                Us = Ub[ss.to(Ub.device)] if Ub.dim() == 3 else Ub
                ev, o, pr, dv = ccspin_draw(Us, ops_p, [self._in_modes[j] for j in present], list(self.dims),
                                            psel[s0:s0 + Sc], uniforms=u_chain[ss, :n1],
                                            force=None if fe is None else fe[ss, :n1], chunk_bytes=cb,
                                            spins=present if branchy else None, M=M_p)
                events[ss, :n1] = ev
                out[ss.to(p.device)] = o
                prob[ss] *= pr
                dev[:n1] = torch.maximum(dev[:n1], dv)
        if max(float(dev.max()), bdev) > tol_draw:
            raise ValueError(f"sample_apply: per-step identity off by {max(float(dev.max()), bdev):.1e} > tol_draw "
                             f"{tol_draw:.0e} (U not unitary to rounding, or an emitter not complete)")
        K = K or self._kb(U, b)
        rows = torch.where(events >= 0, events // K, -1)
        kk = torch.where(events >= 0, events % K, -1)
        det = (rows >= 0) & (rows < self.m)                                                      # loss rows: unreported
        occ = torch.zeros((S, self.m), dtype=torch.long).scatter_add_(1, torch.where(det, rows, 0), det.long())
        if not index or (self.k + 1) ** self.m >= 2 ** 62:   # the pattern key of `pattern_index` would overflow
            idx = None
        elif eta_out is not None and not self.lossy:
            pats = self.patterns_all
            base = self.k + 1
            keys = (pats * base ** np.arange(self.m)).sum(1)
            order = np.argsort(keys)
            q = (occ.numpy() * base ** np.arange(self.m)).sum(-1)
            idx = torch.as_tensor(order[np.searchsorted(keys[order], q)], device=self.device)
        else:
            idx = self.pattern_index(occ.numpy())
        info = {"events": torch.stack([rows, kk], -1), "deviation": dev, "branch_deviation": bdev, "prob": prob,
                "branch": branch, "occ": occ}
        return idx, out, info

    def clear_cache(self):
        """Drop the cached Kraus operators and the unitary they were built for."""
        super().clear_cache()
        self._kraus_m = {}
        self._fine_m, self._kb_m, self._span_m = {}, {}, {}

    def bin_mps(self, b: int):
        """No EmitterMPS form: the stage of a bin with internal modes lives on (spatial x internal) modes."""
        raise NotImplementedError("internal modes: a bin with internal modes has no EmitterMPS form on the spatial "
                                  "modes (use stage_ops)")

    def __repr__(self):
        return f"ModeSequenceKernel(k={self.k}, n={self.n}, m={self.m}, chi={self.chi}, K={self.K})"


def source_loss(emitter, eta) -> ModeBinEmitter:
    """The emitter with a source efficiency eta (a float, or one per bin [n]): its photon reaches the network with
    amplitude sqrt(eta), ops -> sqrt(eta) A, and is otherwise never detected, `lost` += sqrt(1 - eta) A[l, k] for every
    label and internal mode (the internal mode of an undetected photon is traced out). A uniform detection efficiency
    moves to the sources the same way (eta = eta_source * eta_detection). A plain `BinEmitter` becomes a one-mode
    `ModeBinEmitter`. The `lost` set (with the emitter's own) is reduced per bin to its Choi rank, at most d^2
    operators (SVD of the stacked operators, singular values above 1e-14 s_max: the channel moves by <= ~1e-28
    relative), since F's presence branches multiply the absent emitters' `lost` counts.
    """
    ops = emitter.ops if isinstance(emitter, ModeBinEmitter) else emitter.ops[:, :, None]           # [n, L, K, d, d]
    n, d = ops.shape[0], ops.shape[-1]
    e = np.broadcast_to(np.asarray(eta, dtype=np.float64), (n,))
    if np.any((e < 0) | (e > 1)):
        raise ValueError("eta must lie in [0, 1]")
    lost = [np.sqrt(1.0 - e).reshape(n, 1, 1, 1) * ops.reshape(n, -1, d, d)]
    old = emitter.lost if isinstance(emitter, ModeBinEmitter) else None
    if old is not None:
        lost.append(old)
    lost = np.concatenate(lost, 1)                                                               # [n, Q, d, d]
    red = []
    for b in range(n):                                                                           # Choi-rank set
        _, s, vh = np.linalg.svd(lost[b].reshape(lost.shape[1], d * d), full_matrices=False)
        red.append((s[:, None] * vh)[s > 1e-14 * s[0]] if s.size and s[0] > 0 else np.zeros((0, d * d)))
    q = max(len(r) for r in red)
    out = np.zeros((n, q, d, d), dtype=np.complex128)
    for b, r in enumerate(red):
        out[b, :len(r)] = r.reshape(-1, d, d)
    return ModeBinEmitter(np.sqrt(e).reshape(n, *([1] * (ops.ndim - 1))) * ops, emitter.modes, emitter.s0,
                          emitter.final, out if q else None)


def _not_yet(name):
    def method(self, *args, **kwargs):
        raise NotImplementedError(f"internal modes: SequenceKernel.{name} with internal-mode emitters is not "
                                  "implemented yet")
    method.__name__ = name
    method.__doc__ = f"Not implemented yet with internal-mode emitters; see SequenceKernel.{name}."
    return method


for _name in _NOT_YET:
    setattr(ModeSequenceKernel, _name, _not_yet(_name))
