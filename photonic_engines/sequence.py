"""Multi-emitter time-bin sequences through a static interferometer — docs/SEQUENCE.md.

k emitters (spins / qudits of dimension d_j) each emit one photon per time bin into their own
input modes of one m-mode interferometer U that contains no delay line. Photons of different
bins never interfere: the unrolled unitary on the m·n (mode, bin) slots is block-diagonal,
U ⊕ U ⊕ … ⊕ U, and the only correlation between bins is the product spin register
chi = prod d_j. The k·n-photon output — the sequence of the n per-bin patterns — is therefore
a matrix product along the bin axis with the register as the bond:

    p(t_1 … t_n) = Tr[ K_{t_n} … K_{t_1} rho_0 K_{t_1}^† … K_{t_n}^† ],
    K_t = sqrt(N_t) <t| Phi   (one per k-photon pattern t of bin b; EmitterSLOSKernel.kraus on the
                               k-emission stage MPS of bin b, chi = prod d_j)

(the emitter "stage" of a spin-optical processor, iterated over the bins, without feed-forward).

  strong side   the per-bin Kraus sets (one SLOS pass per distinct bin, chi x the k-photon Fock
                cost), then everything is a chain on the register: sequence probabilities /
                amplitudes, bin marginals by marching the register density matrix through the
                transfer channel E_b(rho) = sum_t K_t rho K_t^† (differentiable in U and in the
                spin gates), joint distributions of a few bins. The full k·n-photon distribution
                has S_bin^n entries and is never enumerated (S_bin = C(m+k-1, k)).
  weak side     pure quantum trajectories: the register starts pure, at bin b draw t from
                ||K_t psi||^2 over the S_bin patterns, set psi <- K_t psi / ||.||, next bin. Exact,
                cost n · shots · S_bin · chi^2 by enumeration of the bin patterns (the chain-rule
                route for large S_bin needs the chi > 2 sampling kernel, docs/EMITTER_SAMPLER.md).
  loss          uniform eta per photon commutes with U inside a bin and bins are independent
                given the register, so marginals and samples are thinned bin by bin
                (photonic_engines.imperfections); the trajectory keeps the lossless register update,
                which is a valid unravelling of the lossy channel.

k (emitters interfering in one bin) is the exponent — chi = 2^k for spins, 4^k per Kraus
operator, S_bin patterns per bin; n (length of the graph states) is linear.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .masks import Mask
from .states import EmitterMPS

C128 = torch.complex128


@dataclass(frozen=True)
class BinEmitter:
    """One emitter of the sequence: ops[b, l] (d x d) applied at bin b when label l is emitted,
    with any spin gate preceding the emission folded in (ops[b, l] = P_l G_b); modes[l] the
    interferometer input mode of label l (the same at every bin); s0 / final on its own spin.
    """
    ops: np.ndarray                      # [n, L, d, d]
    modes: tuple[int, ...]               # [L]
    s0: np.ndarray                       # [d]
    final: np.ndarray | None = None   # [d] or None (traced)

    def __post_init__(self):
        ops = np.asarray(self.ops, dtype=np.complex128)
        if ops.ndim != 4 or ops.shape[2] != ops.shape[3]:
            raise ValueError("ops must have shape [n, L, d, d]")
        modes = tuple(int(x) for x in self.modes)
        if len(modes) != ops.shape[1] or len(set(modes)) != len(modes):
            raise ValueError("modes must list one distinct input mode per label")
        s0 = np.asarray(self.s0, dtype=np.complex128).reshape(-1)
        final = None if self.final is None else np.asarray(self.final, dtype=np.complex128).reshape(-1)
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "s0", s0)
        object.__setattr__(self, "final", final)

    @property
    def n(self) -> int:
        """Number of bins, i.e. of emissions."""
        return self.ops.shape[0]

    @property
    def d(self) -> int:
        """Spin dimension of this emitter."""
        return self.ops.shape[2]

    @staticmethod
    def from_mps(state: EmitterMPS, m: int) -> BinEmitter:
        """An EmitterMPS written on the unrolled (mode, bin) slots, modes[b][l] = b*m + mode_l,
        i.e. the same spatial modes at every bin.
        """
        modes0 = tuple(x % m for x in state.modes[0])
        for b, mb in enumerate(state.modes):
            if tuple(x - b * m for x in mb) != modes0 or any(x // m != b for x in mb):
                raise ValueError("EmitterMPS must use the same spatial modes at every bin (modes[b][l] = b*m + mode_l)")
        return BinEmitter(state.ops, modes0, state.s0, state.final)

    @staticmethod
    def caterpillar(n: int, spin_gates: Sequence[np.ndarray], modes: tuple[int, int],
                    s0=(1.0, 0.0), final=None) -> BinEmitter:
        """Single-spin caterpillar emitter (see EmitterMPS.caterpillar) into the two modes `modes`."""
        st = EmitterMPS.caterpillar(n, spin_gates, modes=[(2 * b, 2 * b + 1) for b in range(n)], s0=s0, final=final)
        return BinEmitter(st.ops, tuple(modes), st.s0, st.final)


def _embed(op: np.ndarray, i: int, dims: Sequence[int]) -> np.ndarray:
    out = np.ones((1, 1), dtype=complex)
    for j, d in enumerate(dims):
        out = np.kron(out, op if j == i else np.eye(d, dtype=complex))
    return out


class SequenceKernel:
    """k emitters x n bins through one static m-mode interferometer (or one unitary per bin)."""

    def __init__(self, emitters: Sequence[BinEmitter], m: int, device: str = "cpu", dtype: torch.dtype = C128,
                 mask: str = Mask.FOCK.value, indist=None, order: int | None = None):
        """indist: partial distinguishability of the emitters' photons (docs/DISTINGUISHABILITY.md
        §3.4) — one pairwise overlap x for every emitter, or one per emitter (a sequence of k
        values: emitter a's photon is in the common internal mode with probability x_a, i.e. the
        Gram matrix S_ab = sqrt(x_a x_b)); photons of ONE emitter never meet (different bins), so
        only the inter-emitter distinguishability enters. With it every bin's Kraus set is
        species-resolved (`EmitterSLOSKernel.kraus_species`, `coarse` maps each operator to its
        pattern), the strong side is a channel chain, `sample` draws refined outcomes, and
        `sample_diag` / `sequence_amplitude` are unavailable. `order=b` truncates the set at b
        distinguishable photons per bin (`imperfections.truncation_bound` with n = k).
        """
        self.emitters = list(emitters)
        if not self.emitters:
            raise ValueError("at least one emitter")
        self.k, self.m = len(self.emitters), int(m)
        self.n = self.emitters[0].n
        if any(e.n != self.n for e in self.emitters):
            raise ValueError("every emitter must emit at every bin (same n)")
        used = [x for e in self.emitters for x in e.modes]
        if len(set(used)) != len(used) or max(used) >= self.m:
            raise ValueError("emitters must use distinct input modes below m")
        self.dims = [e.d for e in self.emitters]
        self.chi = int(np.prod(self.dims))
        self.device, self.cdtype = torch.device(device), dtype
        self.mask = mask
        self.s0 = torch.as_tensor(self._kron([e.s0 for e in self.emitters]), dtype=dtype, device=self.device)
        finals = [e.final for e in self.emitters]
        self.final = None if any(f is None for f in finals) else torch.as_tensor(self._kron(finals), dtype=dtype, device=self.device)
        if any(f is None for f in finals) and not all(f is None for f in finals):
            raise ValueError("either every emitter is projected (final) or every emitter is traced")
        from .imperfections import spec
        if indist is None or (not isinstance(indist, (list, tuple)) and spec(indist) is None):
            self.indist = None
        elif isinstance(indist, (list, tuple)):
            if len(indist) != self.k:
                raise ValueError(f"indist: one overlap per emitter ({self.k})")
            xs = tuple(float(v) for v in indist)
            self.indist = None if all(v >= 1.0 for v in xs) else xs
        else:
            self.indist = (float(spec(indist).x),) * self.k
        self.order = order
        self._coarse: dict[int, torch.Tensor] = {}      # refined operator -> coarse pattern index, per bin (indist)
        self._kraus: dict[int, torch.Tensor] = {}       # per bin (0 only when all bins are alike), for the last U seen
        self._cache_U: torch.Tensor | None = None
        self._layers = None
        self._rank1: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._subset_cache = None
        self._fock_cache: dict[tuple, object] = {}

    @staticmethod
    def _kron(vs):
        out = np.ones(1, dtype=complex)
        for v in vs:
            out = np.kron(out, np.asarray(v, dtype=complex))
        return out

    # ---- constructors ----------------------------------------------------------------------
    @classmethod
    def from_mps(cls, states: Sequence[EmitterMPS], m: int, **kw) -> SequenceKernel:
        """Build the kernel from one EmitterMPS per emitter, each written on the unrolled (mode, bin) slots."""
        return cls([BinEmitter.from_mps(s, m) for s in states], m, **kw)

    # ---- per-bin stage -----------------------------------------------------------------------
    def bin_mps(self, b: int) -> EmitterMPS:
        """The k-emission stage MPS of bin b on the product register (spatial modes)."""
        ops = np.stack([np.stack([_embed(e.ops[b, l], j, self.dims) for l in range(e.ops.shape[1])])
                        for j, e in enumerate(self.emitters)])
        L = ops.shape[1]
        if any(e.ops.shape[1] != L for e in self.emitters):
            raise ValueError("all emitters must have the same number of labels")
        return EmitterMPS(ops=ops, modes=tuple(e.modes for e in self.emitters), s0=self._kron([e.s0 for e in self.emitters]),
                          m=self.m, final=None)

    @staticmethod
    def _as_tensor(U) -> torch.Tensor:
        """U: [m, m] (static) or [n, m, m] (one unitary per bin), numpy / torch, or a Perceval
        circuit (or a list of them, one per bin).
        """
        if isinstance(U, (list, tuple)) and U and hasattr(U[0], "compute_unitary"):
            return torch.stack([torch.as_tensor(np.asarray(c.compute_unitary())) for c in U])
        if hasattr(U, "compute_unitary"):
            U = np.asarray(U.compute_unitary())
        return torch.as_tensor(U)

    def _U(self, U, b: int) -> torch.Tensor:
        U = self._as_tensor(U).to(self.device, self.cdtype)
        return U[b] if U.dim() == 3 else U

    def _same_bins(self, U) -> bool:
        return self._as_tensor(U).dim() == 2 and all(np.array_equal(e.ops[0], e.ops[b]) for e in self.emitters for b in range(self.n))

    @property
    def layers(self):
        """SLOS basis layers of one bin (k photons over m modes under `mask`), built once and cached."""
        if self._layers is None:
            from .slos.basis import get_layers
            self._layers = get_layers(self.k, self.m, self.mask, device=str(self.device))
        return self._layers

    @property
    def patterns(self) -> np.ndarray:
        """The S_bin = C(m+k-1, k) patterns of one bin, [S, m] (the same basis for every bin)."""
        occ = self.layers.final_occupations()
        return occ.cpu().numpy() if isinstance(occ, torch.Tensor) else np.asarray(occ)

    @property
    def S(self) -> int:
        """Number of output patterns of one bin, S_bin = C(m+k-1, k) for the full Fock mask."""
        return self.layers.output_size()

    def kraus(self, U, b: int) -> torch.Tensor:
        """Physical Kraus operators of bin b, [S, chi, chi] (final x initial), K_t = sqrt(N_t) <t|Phi_b>.
        With `indist` the species-resolved set [Q, chi, chi] (Q >= S; `coarse(b)` gives each
        operator's pattern index): the register update of pattern t is the channel over its
        operators, and sum_q K_q^dag K_q = 1 still.
        """
        from .slos.emitter import EmitterSLOSKernel
        Ut = self._as_tensor(U).detach()
        if self._cache_U is None or self._cache_U.shape != Ut.shape or not torch.equal(self._cache_U, Ut.to(self._cache_U)):
            self._kraus, self._coarse, self._cache_U = {}, {}, Ut.clone()
        key = 0 if self._same_bins(U) else b
        if key not in self._kraus:
            st = self.bin_mps(key)
            kern = EmitterSLOSKernel(st, mask=self.mask, device=str(self.device), dtype=self.cdtype, layers=self.layers)
            if self.indist is not None:
                K, coarse = kern.kraus_species(self._U(U, key), list(self.indist), order=self.order)
                self._kraus[key], self._coarse[key] = K, coarse
                return K
            # working set of the propagation: chunk * chi * S amplitudes, twice; keep it near 1 GB
            chunk = max(1, min(self.chi, int(2 ** 26 // max(1, self.chi * self.S))))
            K = kern.kraus(self._U(U, key), chunk=chunk)
            norm = torch.sqrt(torch.as_tensor(self.layers.output_prob_norms()).to(K.device, torch.float64))
            K = K * norm.view(-1, 1, 1).to(K.dtype)
            if K.requires_grad:                       # keep the graph fresh: no caching under autograd
                return K
            self._kraus[key] = K
        return self._kraus[key]

    def coarse(self, U, b: int) -> torch.Tensor:
        """Pattern index [Q] of every operator of `kraus(U, b)` (arange(S) without `indist`)."""
        K = self.kraus(U, b)
        if self.indist is None:
            return torch.arange(K.shape[0], device=K.device)
        return self._coarse[0 if self._same_bins(U) else b]

    def _coarse_probs(self, U, b: int, p: torch.Tensor) -> torch.Tensor:
        """Sum per-operator probabilities [Q] into the S patterns."""
        if self.indist is None:
            return p
        return torch.zeros(self.S, dtype=p.dtype, device=p.device).index_add(0, self.coarse(U, b), p)

    def clear_cache(self):
        """Drop the cached Kraus operators and the unitary they were built for."""
        self._kraus, self._coarse, self._cache_U = {}, {}, None

    def pattern_index(self, occ) -> torch.Tensor:
        """Occupations [..., m] -> indices into `patterns`."""
        pats = self.patterns
        base = self.k + 1
        keys = (pats * base ** np.arange(self.m)).sum(1)
        order = np.argsort(keys)
        occ = np.asarray(occ, dtype=np.int64)
        q = (occ * base ** np.arange(self.m)).sum(-1)
        pos = np.searchsorted(keys[order], q)
        if np.any(pos >= keys.shape[0]) or np.any(keys[order][np.minimum(pos, keys.shape[0] - 1)] != q):
            raise ValueError("a pattern is not a k-photon pattern of this bin basis")
        return torch.as_tensor(order[pos], device=self.device)

    # ---- strong side ---------------------------------------------------------------------------
    def _chain(self, U, seq_idx: torch.Tensor) -> torch.Tensor:
        """K_{t_n} ... K_{t_1} for sequences seq_idx [B, n] -> [B, chi, chi]."""
        B = seq_idx.shape[0]
        M = torch.eye(self.chi, dtype=self.cdtype, device=self.device).expand(B, -1, -1)
        for b in range(self.n):
            M = self.kraus(U, b)[seq_idx[:, b]] @ M
        return M

    def _indices(self, sequences) -> tuple[torch.Tensor, bool]:
        """Sequences as occupations [n, m] / [B, n, m] or pattern indices [n] / [B, n] ->
        (indices [B, n], single).
        """
        seq = np.asarray(sequences)
        if seq.ndim == 1:
            return torch.as_tensor(seq, dtype=torch.long, device=self.device).reshape(1, self.n), True
        if seq.ndim == 2 and seq.shape == (self.n, self.m):
            return self.pattern_index(seq).reshape(1, self.n), True
        if seq.ndim == 2:
            return torch.as_tensor(seq, dtype=torch.long, device=self.device).reshape(-1, self.n), False
        if seq.ndim == 3:
            return self.pattern_index(seq).reshape(-1, self.n), False
        raise ValueError("sequences: [n], [B, n], [n, m] or [B, n, m]")

    def sequence_probability(self, U, sequences) -> torch.Tensor:
        """p(t_1..t_n) for sequences given as occupations [n, m] / [B, n, m] or pattern indices
        [n] / [B, n]; traced register: Tr[M rho_0 M^dag]; projected: |<f|M|s0>|^2.
        """
        idx, single = self._indices(sequences)
        if self.indist is not None:                                                # a channel chain, not an operator chain
            rho = torch.outer(self.s0, self.s0.conj()).expand(idx.shape[0], -1, -1)
            for b in range(self.n):
                K, coarse = self.kraus(U, b), self.coarse(U, b)
                nxt = torch.zeros_like(rho)
                for i in range(idx.shape[0]):
                    Ki = K[coarse == idx[i, b]]
                    nxt[i] = torch.einsum("qab,bc,qdc->ad", Ki, rho[i], Ki.conj())
                rho = nxt
            if self.final is None:
                p = torch.einsum("baa->b", rho).real
            else:
                p = torch.einsum("a,bac,c->b", self.final.conj(), rho, self.final).real
            return p[0] if single else p
        M = self._chain(U, idx)
        v = M @ self.s0
        p = (v.abs() ** 2).sum(-1) if self.final is None else (torch.einsum("s,bs->b", self.final.conj(), v).abs() ** 2)
        return p[0] if single else p

    def sequence_amplitude(self, U, sequences) -> torch.Tensor:
        """<f| K_{t_n}..K_{t_1} |s0> (projected register only)."""
        if self.final is None:
            raise ValueError("amplitudes need projected emitters (final)")
        if self.indist is not None:
            raise NotImplementedError("sequence_amplitude: undefined under partial distinguishability (mixed output)")
        idx, single = self._indices(sequences)
        a = torch.einsum("s,bs->b", self.final.conj(), self._chain(U, idx) @ self.s0)
        return a[0] if single else a

    def _pattern_chunk(self) -> int:
        """Patterns per chunk so that a [chunk, chi, chi] temporary stays near 1 GB."""
        return max(1, int(2 ** 26 // max(1, self.chi * self.chi)))

    def transfer(self, U, b: int, rho: torch.Tensor) -> torch.Tensor:
        """E_b(rho) = sum_t K_t rho K_t^dag (the register after bin b is traced over its outcome)."""
        K = self.kraus(U, b)
        return torch.einsum("tab,bc,tdc->ad", K, rho, K.conj())

    def bin_marginals(self, U, loss=None, return_register: bool = False):
        """Distribution of every bin's pattern, [n, S] over `patterns` (traced register; for
        projected emitters the unnormalised joint with the projection, renormalise by the
        projection probability). Differentiable. loss=eta: each bin thinned (LossyDistribution
        per bin, list of n).
        """
        rho = torch.outer(self.s0, self.s0.conj())
        out = []
        pc = self._pattern_chunk()
        for b in range(self.n):
            K = self.kraus(U, b)
            p = torch.empty(K.shape[0], dtype=rho.real.dtype, device=self.device)
            nxt = torch.zeros_like(rho)
            for t0 in range(0, K.shape[0], pc):                                   # chunks of patterns
                Kc = K[t0:t0 + pc]
                X = torch.matmul(Kc, rho)                                          # K_t rho, [c, chi, chi]
                if self.final is None:
                    p[t0:t0 + pc] = torch.einsum("tab,tab->t", X, Kc.conj()).real   # Tr[K_t rho K_t^dag]
                else:
                    p[t0:t0 + pc] = torch.einsum("a,tab,cb,c->t", self.final.conj(), X, Kc.conj(), self.final).real
                Xp = X.permute(1, 0, 2).reshape(self.chi, -1)                      # [a, (t b)]
                Kp = Kc.conj().permute(1, 0, 2).reshape(self.chi, -1)              # [c, (t b)]
                nxt = nxt + Xp @ Kp.T                                              # sum_t K_t rho K_t^dag
                del X, Xp, Kp
            out.append(self._coarse_probs(U, b, p))
            rho = nxt
        marg = torch.stack(out)
        if loss is not None and float(loss) < 1.0:
            from .imperfections import thin_patterns
            marg = [thin_patterns(self.patterns, marg[b], loss) for b in range(self.n)]
        return (marg, rho) if return_register else marg

    def joint(self, U, bins: Sequence[int]) -> torch.Tensor:
        """Joint distribution of a few bins, tensor [S]*len(bins) (traced register), by
        marching the register with the transfer channel between them. Exponential in
        len(bins) — meant for 2 or 3 bins.
        """
        bins = sorted(int(b) for b in bins)
        rho = torch.outer(self.s0, self.s0.conj()).unsqueeze(0)                     # [1, chi, chi]
        shape = []
        for b in range(self.n):
            if b in bins:
                K = self.kraus(U, b)
                new = torch.einsum("tab,xbc,tdc->xtad", K, rho, K.conj())                # [x, Q, chi, chi]
                if self.indist is not None:
                    agg = torch.zeros((new.shape[0], self.S, self.chi, self.chi), dtype=new.dtype, device=new.device)
                    new = agg.index_add(1, self.coarse(U, b), new)
                rho = new.reshape(-1, self.chi, self.chi)
                shape.append(self.S)
            else:
                rho = torch.stack([self.transfer(U, b, r) for r in rho]) if rho.shape[0] < 64 else \
                    torch.einsum("tab,xbc,tdc->xad", self.kraus(U, b), rho, self.kraus(U, b).conj())
            if b >= bins[-1]:
                break
        if self.final is None:
            p = torch.einsum("xaa->x", rho).real
        else:
            p = torch.einsum("a,xab,b->x", self.final.conj(), rho, self.final).real
        return p.reshape(shape)

    # ---- weak side -----------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, U, shots: int, generator: torch.Generator | None = None, loss=None,
               return_register: bool = False, chunk: int = 1 << 12):
        """Exact samples [shots, n, m] (pattern of every bin) by pure quantum trajectories on the
        register: draw t_b from ||K_t psi||^2 over the S_bin patterns, psi <- K_t psi. Projected
        emitters are rejection-sampled at the end (fewer rows come back). loss=eta thins each
        bin's pattern (the register keeps the lossless update: a valid unravelling).
        """
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        if shots > chunk:                                                          # bound the [shots, S_bin] weights
            parts = [self.sample(U, c, generator=generator, loss=None, return_register=True, chunk=chunk)
                     for c in [chunk] * (shots // chunk) + ([shots % chunk] if shots % chunk else [])]
            pats, psi = torch.cat([p for p, _ in parts]), torch.cat([q for _, q in parts])
            if loss is not None and float(loss) < 1.0:
                from .imperfections import thin_counts
                pats = thin_counts(pats, loss, generator=generator)
            return (pats, psi) if return_register else pats
        psi = self.s0.unsqueeze(0).expand(shots, -1).clone()                       # [S, chi]
        idx = torch.zeros((shots, self.n), dtype=torch.long)
        Ms: dict[int, torch.Tensor] = {}
        for b in range(self.n):
            K = self.kraus(U, b)                                                   # [Sb, chi, chi]
            key = 0 if self._same_bins(U) else b
            if key not in Ms:
                M = torch.empty((K.shape[0], self.chi * self.chi), dtype=K.dtype, device=K.device)
                pc = self._pattern_chunk()
                for t0 in range(0, K.shape[0], pc):
                    Kc = K[t0:t0 + pc]
                    M[t0:t0 + pc] = torch.matmul(Kc.conj().transpose(1, 2), Kc).reshape(Kc.shape[0], -1)   # K_t^dag K_t
                Ms[key] = M                                                        # [Sb, chi^2]
            M = Ms[key]
            outer = (psi.conj().unsqueeze(2) * psi.unsqueeze(1)).reshape(shots, -1)  # conj(psi_a) psi_b
            w = (outer @ M.T).real.clamp_min(0).to(torch.float64)                  # [shots, Sb]
            t = torch.multinomial((w / w.sum(1, keepdim=True)).cpu(), 1, generator=generator)[:, 0]
            idx[:, b] = t if self.indist is None else self.coarse(U, b).cpu()[t]     # refined operator -> its pattern
            psi = torch.bmm(K[t.to(K.device)], psi.unsqueeze(-1)).squeeze(-1)     # [shots, chi]
            psi = psi / psi.norm(dim=1, keepdim=True)
        keep = torch.ones(shots, dtype=torch.bool)
        if self.final is not None:
            acc = (torch.einsum("s,xs->x", self.final.conj(), psi).abs() ** 2).cpu()
            keep = torch.rand(shots, generator=generator, dtype=torch.float64) < acc
            idx, psi = idx[keep], psi[keep.to(psi.device)]
        pats = torch.as_tensor(self.patterns)[idx]                                 # [shots, n, m]
        if loss is not None and float(loss) < 1.0:
            from .imperfections import thin_counts
            pats = thin_counts(pats, loss, generator=generator)
        return (pats, psi) if return_register else pats

    # ---- label-diagonal trajectory sampler (no Kraus set) --------------------------------------
    def _rank1_ops(self, b: int):
        """Emitters whose ops[b, l] are rank one, ops[b, l] = |v_l><u_l| (a spin gate then a
        label-preserving emission, e.g. the caterpillar: ops = |l><l| G). Then the bin's Kraus
        operator factors through the label basis,

            K_t = V diag_x(<t|U|x>) Uop,   V = (x) [v_0 .. v_{L-1}],  Uop = (x) [<u_0|; ..; <u_{L-1}|],

        with <t|U|x> the k-photon amplitude of the input |x> (emitter j in mode modes_j[x_j]).
        Returns (V [k, d, L], Uop [k, L, d]) as torch tensors; raises if an emitter is not of that
        form or if V / Uop are not isometries (needed for the label mixture below).
        """
        if b in self._rank1:
            return self._rank1[b]
        Vs, Us = [], []
        for e in self.emitters:
            d, L = e.d, e.ops.shape[1]
            V = np.zeros((d, L), dtype=complex); Uo = np.zeros((L, d), dtype=complex)
            for l in range(L):
                u, s, vh = np.linalg.svd(e.ops[b, l])
                if s.size > 1 and s[1] > 1e-12 * max(s[0], 1e-300):
                    raise ValueError("sample_diag needs rank-one emission operators ops[b, l] = |v><u|")
                V[:, l], Uo[l] = u[:, 0] * s[0], vh[0]
            if not (np.allclose(V.conj().T @ V, np.eye(L), atol=1e-9) and np.allclose(Uo @ Uo.conj().T, np.eye(L), atol=1e-9)):
                raise ValueError("sample_diag needs isometric emission (unitary spin gate, orthogonal labels)")
            Vs.append(V); Us.append(Uo)
        out = (torch.as_tensor(np.stack(Vs), dtype=self.cdtype, device=self.device),
               torch.as_tensor(np.stack(Us), dtype=self.cdtype, device=self.device))
        self._rank1[b] = out
        return out

    @staticmethod
    def _apply_local(z: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        """Z [S, a^k] -> (x)_j mats[j] z, mats [k, b, a] (emitter 0 the most significant index)."""
        S, k = z.shape[0], mats.shape[0]
        a, b = mats.shape[2], mats.shape[1]
        z = z.reshape(S, *([a] * k))
        for j in range(k):
            z = torch.movedim(torch.tensordot(z, mats[j], dims=([j + 1], [1])), -1, j + 1)   # [S, .., b, ..]
        return z.reshape(S, b ** k)

    def _subsets(self):
        """Laplace structure of the label amplitudes over the k output photon positions: for every
        step j (subsets of size j -> j+1), the parent tables par[j] [n_{j+1}, j+1] (index of T \\ p in
        the size-j list) and pos[j] [n_{j+1}, j+1] (the removed position p), plus the peak memory per
        stream (complex entries) of the recurrence.
        """
        if self._subset_cache is None:
            from itertools import combinations
            k = self.k
            index = [{c: i for i, c in enumerate(combinations(range(k), j))} for j in range(k + 1)]
            steps, peak = [], 0
            for j in range(k):
                par = torch.empty((len(index[j + 1]), j + 1), dtype=torch.long)
                pos = torch.empty_like(par)
                for c, i in index[j + 1].items():
                    for q, p in enumerate(c):
                        par[i, q] = index[j][tuple(y for y in c if y != p)]
                        pos[i, q] = p
                steps.append((par.to(self.device), pos.to(self.device)))
                n0, n1, L = len(index[j]), len(index[j + 1]), self.emitters[0].ops.shape[1]
                peak = max(peak, n0 * L ** j + n1 * L ** (j + 1) + n1 * L ** j)          # f, g, one gathered parent block
            self._subset_cache = (steps, peak)
        return self._subset_cache

    def label_amplitudes(self, U, counts: torch.Tensor, b: int = 0, chunk_bytes: int = 1 << 30,
                         U_streams: torch.Tensor | None = None, method: str = "glynn",
                         emitters: Sequence[int] | None = None, pad_tol: float = 1.25) -> torch.Tensor:
        """<t|U|x> / sqrt(t!) for every label word x, [S, L^k], for the output patterns
        `counts` [S, m] (k photons each), batched over the patterns, no Kraus set. `U_streams`
        [S, m, m]: one unitary per stream (routing feed-forward) instead of the bin's U.

        method "glynn" (default): the 2^k permanents at once from Glynn's formula. With
        c_j(delta, l) = sum_p delta_p U[r_p, modes[j][l]] the column sums of the stream's k rows,
            <t|U|x> = 2^{1-k} sum_{delta in {+-1}^k, delta_0 = 1} (prod delta) prod_j c_j(delta, x_j),
        so the vector over x is a sum of 2^{k-1} outer products (x)_j (c_j(delta, 0), c_j(delta, 1)).
        Split the emitters in two halves: V_a[delta, x_a] = prod_{j < k/2} c_j(delta, x_j),
        V_b[delta, x_b] = (prod delta) prod_{j >= k/2} c_j(delta, x_j), and the result is the matrix
        product V_a^T V_b [2^{k/2}, 2^{k/2}] summed over delta — a batched GEMM over blocks of
        delta, no state to gather: 4^k / 2 flops per stream at GEMM speed, memory
        O(D (2k + 2^{k/2})) per stream for a block of D signs. All streams of a chunk share every
        launch; tens of launches per tick instead of tens of thousands (docs/SEQUENCE.md §12).
        Since 2026-09-26 the sign sum is COLLAPSED over the repeated output rows of the pattern
        (Clifford & Clifford 2020, docs/SEQUENCE.md §12b): a row of multiplicity t contributes
        t + 1 sign counts v (weight (-1)^v C(t, v), column-sum coefficient t - 2v) instead of 2^t
        signs, and the complement symmetry halves the count once when some t is odd — the same
        tables as `clifford.cofactors_triton.multiplicity_tables`, rebuilt here in plain torch so
        that this module does not import triton. prod (t_r + 1) terms per stream instead of
        2^{k-1}: on Haar patterns at m = 2k the work drops 3x at k = 12, 6.8x at k = 20, 8.3x at
        k = 22 (Bose-Einstein surrogate, matched by interpreter-sampled patterns at k = 8, 12).
        Streams are sorted by their term count and cut into groups whose counts lie within
        `pad_tol` of each other; a group is padded to its largest count with zero-weight terms.
        method "glynn_full": the previous, uncollapsed sum over all 2^{k-1} signs (the A/B
        reference; identical values to the collapsed sum up to rounding). method "laplace": the
        earlier expansion over the output photon positions, O(k 3^k) but state-bound (kept as
        the reference). `emitters`: a subset of emitter indices — the amplitudes of a g-photon
        pattern over THEIR label words only, [S, L^g] (glynn methods only; `sample_diag` under
        partial distinguishability, the other emitters' photons being routed separately).
        """
        if method == "laplace":
            if emitters is not None:
                raise NotImplementedError("label_amplitudes(emitters=): glynn only")
            return self._label_amplitudes_laplace(U, counts, b, chunk_bytes, U_streams)
        if method == "glynn":
            return self._label_amplitudes_collapsed(U, counts, b, chunk_bytes, U_streams, emitters, pad_tol)
        if method != "glynn_full":
            raise ValueError(f'unknown method {method!r}: "glynn", "glynn_full" or "laplace"')
        Ut = self._U(U, b) if U_streams is None else None
        sel = list(range(self.k)) if emitters is None else [int(a) for a in emitters]
        k, m, L = len(sel), self.m, self.emitters[0].ops.shape[1]
        if k == 0:
            return torch.ones((counts.shape[0], 1), dtype=self.cdtype, device=self.device)
        modes = torch.as_tensor([self.emitters[a].modes for a in sel], dtype=torch.long, device=self.device)   # [k, L]
        counts = counts.to(self.device)
        S = counts.shape[0]
        rows = torch.repeat_interleave(torch.arange(m, device=self.device).repeat(S), counts.reshape(-1)).view(S, k)
        norm = torch.exp(-0.5 * torch.lgamma(counts.to(torch.float64) + 1).sum(1))                # 1/sqrt(t!)
        ka = k // 2
        kb = k - ka
        nd = 1 << (k - 1)                                                                          # signs with delta_0 = +1
        per = 16 * (2 * k * L + 3 * (L ** ka + L ** kb))                                           # bytes per (stream, sign)
        D = max(1, min(nd, chunk_bytes // (per * max(1, min(S, 256)))))
        D = 1 << (D.bit_length() - 1)
        sc = max(1, min(S, chunk_bytes // (per * D)))
        signs = self._glynn_signs(k, D)                                                             # list of (Delta [D, k], prod [D])
        out = torch.empty((S, L ** k), dtype=self.cdtype, device=self.device)
        for s0 in range(0, S, sc):
            r = rows[s0:s0 + sc]
            Sg = r.shape[0]
            if U_streams is None:
                Ug = Ut[r]                                                                      # [Sg, k, m]
            else:
                Ug = U_streams[s0:s0 + sc][torch.arange(Sg, device=self.device)[:, None], r]    # rows of each stream's U
            A = Ug[:, :, modes].reshape(Sg, k, k * L)                                           # [Sg, k (row p), (j, l)]
            acc = torch.zeros((Sg, L ** ka, L ** kb), dtype=self.cdtype, device=self.device)
            for Delta, prod in signs:
                C = torch.matmul(Delta.unsqueeze(0), A).reshape(Sg, -1, k, L)                   # [Sg, D, k, L] column sums
                Dn = C.shape[1]
                Va = torch.ones((Sg, Dn, 1), dtype=self.cdtype, device=self.device)
                for j in range(ka):
                    Va = (Va.unsqueeze(-1) * C[:, :, j, :].unsqueeze(2)).reshape(Sg, Dn, -1)
                Vb = prod.view(1, -1, 1).expand(Sg, Dn, 1)
                for j in range(ka, k):
                    Vb = (Vb.unsqueeze(-1) * C[:, :, j, :].unsqueeze(2)).reshape(Sg, Dn, -1)
                acc += torch.matmul(Va.transpose(1, 2), Vb)                                     # [Sg, L^ka, L^kb]
            out[s0:s0 + sc] = acc.reshape(Sg, -1) * (2.0 ** (1 - k)) * norm[s0:s0 + sc].unsqueeze(1).to(self.cdtype)
        return out

    def _glynn_signs(self, k: int, D: int):
        """Blocks of D sign vectors delta in {+-1}^k with delta_0 = +1 (all 2^{k-1}), as complex
        matrices [D, k] with their products [D]; cached per (k, D).
        """
        key = ("glynn", k, D)
        if key not in self._fock_cache:
            nd = 1 << (k - 1)
            idx = torch.arange(nd, device=self.device)
            bits = ((idx.unsqueeze(1) >> torch.arange(k - 1, device=self.device)) & 1)         # [nd, k-1]
            delta = torch.cat([torch.ones((nd, 1), dtype=torch.float64, device=self.device),
                               1.0 - 2.0 * bits.to(torch.float64)], dim=1)                         # [nd, k]
            prod = delta.prod(dim=1)
            self._fock_cache[key] = [(delta[i:i + D].to(self.cdtype), prod[i:i + D].to(self.cdtype)) for i in range(0, nd, D)]
        return self._fock_cache[key]

    @staticmethod
    def _multiplicity_tables(rows: torch.Tensor):
        """Distinct output rows of every pattern, their multiplicities and the mixed radices of the
        collapsed sign sum: rows [S, k] int64 -> drow [S, k] (distinct rows, padded with row 0 and
        multiplicity 0), mult [S, k], radix [S, k] (t + 1; a padded slot has radix 1; one odd row is
        moved last and gets radix (t + 1) / 2 — the complement halving), N [S] (terms per stream)
        and scale [S] float64 (2^{-k}, times 2 when halved). Same contract and same values as
        `clifford.cofactors_triton.multiplicity_tables`, without importing triton.
        """
        S, k = rows.shape
        dev = rows.device
        rs, _ = torch.sort(rows, dim=1)
        ar = torch.arange(k, device=dev)
        newgrp = torch.ones(S, k, dtype=torch.bool, device=dev)
        if k > 1:
            newgrp[:, 1:] = rs[:, 1:] != rs[:, :-1]
        gid = newgrp.long().cumsum(1) - 1                                          # group id per sorted position
        D = gid[:, -1] + 1                                                         # distinct rows per stream
        mult = torch.zeros(S, k, dtype=torch.int64, device=dev).scatter_add_(1, gid, torch.ones_like(gid))
        drow = torch.zeros(S, k, dtype=torch.int64, device=dev).scatter_(1, gid, rs)
        active = ar[None, :] < D[:, None]
        odd = (mult % 2 == 1) & active
        has_odd = odd.any(1)
        pos = torch.where(odd, ar[None, :], -1).max(1).values                      # an odd-multiplicity group
        top = D - 1
        perm = ar[None, :].expand(S, k).clone()                                    # swap group `pos` with the top group
        sidx = torch.arange(S, device=dev)
        perm[sidx[has_odd], top[has_odd]] = pos[has_odd]
        perm[sidx[has_odd], pos[has_odd]] = top[has_odd]
        mult = torch.gather(mult, 1, perm)
        drow = torch.gather(drow, 1, perm)
        radix = torch.where(active, mult + 1, torch.ones_like(mult))
        top_r = torch.where(has_odd, (mult.gather(1, top[:, None])[:, 0] + 1) // 2, radix.gather(1, top[:, None])[:, 0])
        radix.scatter_(1, top[:, None], top_r[:, None])
        N = radix.prod(1)
        scale = torch.where(has_odd, 2.0, 1.0).to(torch.float64) * (2.0 ** (-k))
        return drow, mult, radix, N, scale

    def _label_amplitudes_collapsed(self, U, counts, b, chunk_bytes, U_streams, emitters, pad_tol):
        """The Glynn / GEMM label amplitudes with the sign sum collapsed over repeated output rows
        (docs/SEQUENCE.md §12b). For stream s with distinct rows r_nu of multiplicity t_nu,

            <t|U|x> = scale_s sum_v [prod_nu (-1)^{v_nu} C(t_nu, v_nu)] prod_j c_j(v, x_j),
            c_j(v, l) = sum_nu (t_nu - 2 v_nu) U[r_nu, modes[j][l]],   v_nu = 0..t_nu (mixed radix),

        i.e. the sum of §12 with the per-stream sign matrix Delta[s, e, nu] = t_nu - 2 v_nu(e) and the
        weight w[s, e] in place of the shared {+-1} rows and their product; the two half-products
        and the [L^ka, L^kb] GEMM per block of terms are unchanged. Streams are grouped by term count
        (sorted, cut where N_max > pad_tol N_min) and padded to the group's N_max with w = 0.
        """
        Ut = self._U(U, b) if U_streams is None else None
        sel = list(range(self.k)) if emitters is None else [int(a) for a in emitters]
        k, m, L = len(sel), self.m, self.emitters[0].ops.shape[1]
        dev, cd = self.device, self.cdtype
        if k == 0:
            return torch.ones((counts.shape[0], 1), dtype=cd, device=dev)
        modes = torch.as_tensor([self.emitters[a].modes for a in sel], dtype=torch.long, device=dev)   # [k, L]
        counts = counts.to(dev)
        S = counts.shape[0]
        rows = torch.repeat_interleave(torch.arange(m, device=dev).repeat(S), counts.reshape(-1)).view(S, k)
        norm = torch.exp(-0.5 * torch.lgamma(counts.to(torch.float64) + 1).sum(1))                # 1/sqrt(t!)
        drow, mult, radix, N, scale = self._multiplicity_tables(rows)
        binom = self._binomial_table(k)                                                              # [k+1, k+1] float64
        place = torch.cumprod(radix.flip(1), 1).flip(1)                                              # prod_{nu' >= nu} radix
        place = torch.cat([place[:, 1:], torch.ones((S, 1), dtype=torch.long, device=dev)], 1)      # prod_{nu' > nu}
        ka = k // 2
        kb = k - ka
        per = 16 * (2 * k * L + 3 * (L ** ka + L ** kb)) + 24 * k                                    # bytes per (stream, term)
        order = torch.argsort(N)
        Ns = N[order].tolist()
        out = torch.empty((S, L ** k), dtype=cd, device=dev)
        g0 = 0
        while g0 < S:                                                                                # groups by term count
            g1 = g0 + 1
            while g1 < S and Ns[g1] <= pad_tol * Ns[g0]:
                g1 += 1
            ids = order[g0:g1]
            Sg, Nmax = ids.shape[0], Ns[g1 - 1]
            D = max(1, min(Nmax, chunk_bytes // (per * max(1, min(Sg, 256)))))
            sc = max(1, min(Sg, chunk_bytes // (per * D)))
            for s0 in range(0, Sg, sc):
                sid = ids[s0:s0 + sc]
                Sc = sid.shape[0]
                dr, mu, ra, pl = drow[sid], mult[sid], radix[sid], place[sid]                       # [Sc, k]
                if U_streams is None:
                    Ug = Ut[dr]                                                                      # [Sc, k (nu), m]
                else:
                    Ug = U_streams[sid][torch.arange(Sc, device=dev)[:, None], dr]
                A = Ug[:, :, modes].reshape(Sc, k, k * L)                                            # [Sc, k (nu), (j, l)]
                acc = torch.zeros((Sc, L ** ka, L ** kb), dtype=cd, device=dev)
                for e0 in range(0, Nmax, D):
                    e = torch.arange(e0, min(e0 + D, Nmax), device=dev)                             # term indices
                    v = (e[None, :, None] // pl[:, None, :]) % ra[:, None, :]                       # [Sc, Dn, k] sign counts
                    w = torch.where(e[None, :] < N[sid][:, None], scale[sid][:, None], 0.0)          # padded terms weigh 0
                    w = w * binom[mu[:, None, :], v].prod(2) * (1.0 - 2.0 * (v.sum(2) % 2))         # prod C(t, v) (-1)^{sum v}
                    Delta = (mu[:, None, :] - 2 * v).to(cd)                                          # [Sc, Dn, k]
                    C = torch.bmm(Delta, A).reshape(Sc, -1, k, L)                                    # [Sc, Dn, k, L] column sums
                    Dn = C.shape[1]
                    Va = torch.ones((Sc, Dn, 1), dtype=cd, device=dev)
                    for j in range(ka):
                        Va = (Va.unsqueeze(-1) * C[:, :, j, :].unsqueeze(2)).reshape(Sc, Dn, -1)
                    Vb = w.to(cd).unsqueeze(-1)
                    for j in range(ka, k):
                        Vb = (Vb.unsqueeze(-1) * C[:, :, j, :].unsqueeze(2)).reshape(Sc, Dn, -1)
                    acc += torch.matmul(Va.transpose(1, 2), Vb)                                      # [Sc, L^ka, L^kb]
                out[sid] = acc.reshape(Sc, -1) * norm[sid].unsqueeze(1).to(cd)
            g0 = g1
        return out

    def _binomial_table(self, k: int) -> torch.Tensor:
        """C(a, b) for a, b <= k as float64 [k+1, k+1] (0 above the diagonal); cached per k."""
        key = ("binom", k)
        if key not in self._fock_cache:
            t = torch.zeros(k + 1, k + 1, dtype=torch.float64)
            for a in range(k + 1):
                for c in range(a + 1):
                    t[a, c] = math.comb(a, c)
            self._fock_cache[key] = t.to(self.device)
        return self._fock_cache[key]

    def _label_amplitudes_laplace(self, U, counts, b, chunk_bytes, U_streams):
        """Laplace expansion over the k output photon positions, states (subset of positions,
        label prefix), O(k 3^k) per pattern at L = 2; step j is a loop over the j+1 parents of
        every new subset (a gather of the state and a fused multiply-add each).
        """
        Ut = self._U(U, b) if U_streams is None else None
        k, m, L = self.k, self.m, self.emitters[0].ops.shape[1]
        modes = torch.as_tensor([e.modes for e in self.emitters], dtype=torch.long, device=self.device)   # [k, L]
        counts = counts.to(self.device)
        S = counts.shape[0]
        rows = torch.repeat_interleave(torch.arange(m, device=self.device).repeat(S), counts.reshape(-1)).view(S, k)
        norm = torch.exp(-0.5 * torch.lgamma(counts.to(torch.float64) + 1).sum(1))                # 1/sqrt(t!)
        steps, peak = self._subsets()
        sc = max(1, min(S, chunk_bytes // (peak * 16)))
        out = torch.empty((S, L ** k), dtype=self.cdtype, device=self.device)
        for s0 in range(0, S, sc):
            r = rows[s0:s0 + sc]
            Sg = r.shape[0]
            if U_streams is None:
                Ug = Ut[r]                                                                      # [Sg, k, m]
            else:
                Ug = U_streams[s0:s0 + sc][torch.arange(Sg, device=self.device)[:, None], r]    # rows of each stream's U
            A = Ug[:, :, modes]                                                                 # [Sg, k (position p), k (emitter j), L]
            f = torch.ones((Sg, 1, 1), dtype=self.cdtype, device=self.device)                   # [Sg, n_j, L^j]
            for j, (par, pos) in enumerate(steps):
                n1 = par.shape[0]
                g = torch.zeros((Sg, n1, L ** j, L), dtype=self.cdtype, device=self.device)
                Aj = A[:, :, j, :]                                                              # [Sg, k, L]
                for q in range(j + 1):
                    g.addcmul_(f[:, par[:, q]].unsqueeze(-1), Aj[:, pos[:, q]].unsqueeze(2))     # [Sg, n1, L^j, L]
                f = g.reshape(Sg, n1, L ** (j + 1))
            out[s0:s0 + sc] = f[:, 0] * norm[s0:s0 + sc].unsqueeze(1).to(self.cdtype)
        return out

    def fock_sample(self, U, cols: torch.Tensor, generator: torch.Generator, b: int = 0,
                    U_streams: torch.Tensor | None = None, backend: str = "triton") -> torch.Tensor:
        """One exact k-photon output pattern per stream, for single photons entering the input modes cols.

        Public API (used by applications built on the sequence engine, e.g. a spin-register machine that draws
        the photon pattern of a tick and then conditions its own register on it).

        Args:
            U: the bin's unitary [m, m], or [B, m, m] with `b` selecting the bin.
            cols: int64 [S, k], the input modes of the k photons of each stream (a mode may repeat: several photons).
            generator: torch.Generator on the CPU (the draws are reproducible from it; on cuda a device generator
                is seeded from it).
            b: bin index when U is [B, m, m].
            U_streams: optional complex [S, m, m], one unitary per stream (overrides U).
            backend: the Clifford sampler's kernel on cuda, "triton" | "cuda"; ignored on the cpu.

        Returns:
            int64 counts [S, m] on the kernel's device; each row sums to k.

        On cuda: the Clifford GPU sampler, one circuit per stream. On the cpu: the strong Fock SLOS law of each
        distinct input, sampled exactly (small k only: the law is enumerated, and cached per input).
        """
        Ut = self._U(U, b) if U_streams is None else None
        S, k, m = cols.shape[0], cols.shape[1], self.m                            # k = photons of THIS instance (<= self.k)
        if k == 0:
            return torch.zeros((S, m), dtype=torch.long, device=self.device)
        if self.device.type == "cuda":
            from .clifford import fock_sample as clifford_fock_sample
            Ub = torch.zeros((S, m, m), dtype=self.cdtype, device=self.device)
            if U_streams is None:
                Ub[:, :, :k] = Ut[:, cols].permute(1, 0, 2)
            else:
                Ub[:, :, :k] = torch.gather(U_streams, 2, cols[:, None, :].expand(S, m, k))
            # the Clifford sampler draws on the device: a cuda generator seeded from the host one
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator))
            g = torch.Generator(device=self.device).manual_seed(seed)
            return clifford_fock_sample(Ub, list(range(k)), 1, generator=g, backend=backend).to(self.device)
        from .slos.kernel import FockSLOSKernel
        occ = torch.zeros((S, m), dtype=torch.long, device=self.device)
        uniq, inv = torch.unique(cols, dim=0, return_inverse=True)
        for i, c in enumerate(uniq.tolist()):
            key = tuple(c)
            key = (b if self._as_tensor(U).dim() == 3 else 0, *key)
            if key not in self._fock_cache:
                inp = [0] * m
                for x in c:
                    inp[x] += 1
                self._fock_cache[key] = FockSLOSKernel(k, m, inp, mask=self.mask, device=str(self.device),
                                                       dtype=self.cdtype, layers=self.layers if k == self.k else None)
            pats = torch.as_tensor(self._fock_cache[key].layers.final_occupations(), device=self.device)
            ids = (inv == i).nonzero()[:, 0]
            if U_streams is None:
                p = self._fock_cache[key].probs(Ut).detach().to(torch.float64).cpu()
                occ[ids] = pats[torch.multinomial(p, ids.numel(), replacement=True, generator=generator).to(self.device)]
            else:                                                                   # one distribution per stream
                p = self._fock_cache[key].probs(U_streams[ids]).detach().to(torch.float64).cpu()   # [n_ids, S_bin]
                occ[ids] = pats[torch.multinomial(p, 1, generator=generator)[:, 0].to(self.device)]
        return occ

    @torch.no_grad()
    def sample_diag(self, U, shots: int, generator: torch.Generator | None = None,
                    return_register: bool = False, hook=None, chunk_bytes: int | None = None, policy=None,
                    amp_method: str = "glynn", fock_backend: str = "triton", draw: str = "auto",
                    timing: dict | None = None):
        """Trajectory sampler without the Kraus set, for label-preserving emitters (`_rank1_ops`).
        Per bin: phi = Uop psi (label basis), draw the label word x ~ |phi(x)|^2 (the bin's
        distribution is the mixture over x of the k-photon Fock instances |x>), draw the pattern t
        from the Fock instance (Clifford sampler on cuda), then psi <- V (<t|U|.> * phi) — the
        2^k amplitudes <t|U|x> by `label_amplitudes` (O(k 3^k)). Exactly K_t psi / ||K_t psi||, so
        the same distribution as `sample`, at O(k 3^k + Fock(k)) per bin and 2^k memory instead
        of the S_bin chi^2 Kraus set. Traced register only (no rejection step). hook(b, psi,
        counts) is called after every bin. Returns patterns [shots, n, m] (and psi [shots, chi]).

        Feed-forward (docs/SEQUENCE.md §10): `policy(b, counts, psi)` is called after bin b with
        the bin's patterns [S, m] and the updated register [S, chi] and returns, for bin b + 1 and
        per stream, the spin gates [S, k, d, d] (label-diagonal emitters, d = L: the bin's Uop
        becomes the per-stream G), or the interferometer [S, m, m] (routing feed-forward), or a
        pair (gates, U) with either None; None keeps the emitters' own gates and the bin's U. The
        classical outcome -> setting loop is what the trajectory picture handles at no extra
        cost; the strong side (bin_marginals, kraus) does not take a policy. `amp_method` is
        passed to `label_amplitudes` ("glynn" | "glynn_full" | "laplace"; benchmarks only).

        The tick's floor (docs/SEQUENCE.md §12b): `fock_backend` is the Clifford sampler's kernel on
        cuda ("triton" | "cuda"; cuda = the thread-per-lane cofactor kernel, cupy; measured
        irrelevant, 8-22 ms per tick); `draw` is where the label word is drawn — "device" with a cuda
        generator seeded from `generator` (the default on cuda since 2026-09-26: "auto"), "host" with
        `generator` itself after moving the 2^k weights of every stream over (the pre-2026-09-26
        behaviour, 0.44 s of a 0.68 s tick at k = 16 with 256 shots). Both keep the law; the
        trajectories at a given seed differ between the two draws. `timing`: a dict that accumulates
        synchronised seconds per phase ("apply", "draw", "fock", "amp", "update", "hook") for the
        tick bench.
        """
        if self.final is not None:
            raise NotImplementedError("sample_diag: traced register only")
        xs = None if self.indist is None else torch.tensor(self.indist, dtype=torch.float64)      # per-emitter purities
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        if chunk_bytes is None:
            chunk_bytes = (4 << 30) if self.device.type == "cuda" else (1 << 30)              # label-amplitude work space
        L = self.emitters[0].ops.shape[1]
        modes = torch.as_tensor([e.modes for e in self.emitters], dtype=torch.long, device=self.device)
        psi = self.s0.unsqueeze(0).expand(shots, -1).clone()
        pats = torch.zeros((shots, self.n, self.m), dtype=torch.long)
        powers = L ** torch.arange(self.k - 1, -1, -1, device=self.device)
        gates, U_streams = None, None                                                            # per-stream settings of the next bin
        if draw not in ("auto", "host", "device"):
            raise ValueError('draw: "auto", "host" or "device"')
        dev_gen = None
        if draw != "host" and self.device.type == "cuda":
            dev_gen = torch.Generator(device=self.device).manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        import time as _time

        def _lap(key, t0):                                                                       # timing: synchronised seconds
            if timing is None:
                return t0
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = _time.perf_counter(); timing[key] = timing.get(key, 0.0) + t1 - t0
            return t1

        for b in range(self.n):
            t0 = _time.perf_counter()
            V, Uop = self._rank1_ops(b)
            if gates is not None:
                if self.dims[0] != L or any(d != L for d in self.dims):
                    raise ValueError("feed-forward policies need label-diagonal emitters with d = L")
                phi = self._apply_local_batched(psi, gates)                                      # Uop = G per stream
                V = torch.eye(L, dtype=self.cdtype, device=self.device).expand(self.k, L, L)      # ops = |l><l| G: V = identity
            else:
                phi = self._apply_local(psi, Uop)                                                # [S, L^k]
            w = (phi.abs() ** 2).to(torch.float64)
            t0 = _lap("apply", t0)
            if dev_gen is not None:
                x = torch.multinomial(w / w.sum(1, keepdim=True), 1, generator=dev_gen)[:, 0]
            else:
                x = torch.multinomial((w / w.sum(1, keepdim=True)).cpu(), 1, generator=generator)[:, 0].to(self.device)
            labels = (x.unsqueeze(1) // powers) % L                                             # [S, k]
            cols = modes[torch.arange(self.k, device=self.device), labels]                      # [S, k]
            t0 = _lap("draw", t0)
            if xs is None:
                counts = self.fock_sample(U, cols, generator, b=b, U_streams=U_streams, backend=fock_backend)
                t0 = _lap("fock", t0)
                amp = self.label_amplitudes(U, counts, b=b, chunk_bytes=chunk_bytes, U_streams=U_streams, method=amp_method)
                t0 = _lap("amp", t0)
            else:
                counts, amp = self._tick_species(U, cols, labels, xs, generator, b, chunk_bytes, U_streams, amp_method,
                                                 fock_backend)
                t0 = _lap("fock+amp", t0)
            psi = self._apply_local(amp * phi, V)
            psi = psi / psi.norm(dim=1, keepdim=True)
            pats[:, b] = counts.cpu()
            t0 = _lap("update", t0)
            if hook is not None:
                hook(b, psi, counts)
                t0 = _lap("hook", t0)
            if policy is not None and b + 1 < self.n:
                out = policy(b, counts, psi)
                g, Us = (out if isinstance(out, tuple) else (None, None))
                if not isinstance(out, tuple) and out is not None:
                    out = torch.as_tensor(out)
                    g, Us = (out, None) if out.dim() == 4 else (None, out)
                gates = None if g is None else torch.as_tensor(g, dtype=self.cdtype, device=self.device)
                U_streams = None if Us is None else torch.as_tensor(Us, dtype=self.cdtype, device=self.device)
                if gates is not None and gates.shape != (shots, self.k, L, L):
                    raise ValueError(f"policy gates must be [shots, k, d, d] = {(shots, self.k, L, L)}, got {tuple(gates.shape)}")
                if U_streams is not None and U_streams.shape != (shots, self.m, self.m):
                    raise ValueError(f"policy unitaries must be [shots, m, m] = {(shots, self.m, self.m)}, got {tuple(U_streams.shape)}")
        return (pats, psi) if return_register else pats

    def _tick_species(self, U, cols: torch.Tensor, labels: torch.Tensor, xs: torch.Tensor, generator, b: int,
                      chunk_bytes: int, U_streams: torch.Tensor | None, amp_method: str = "glynn",
                      fock_backend: str = "triton"):
        """One tick of `sample_diag` under partial distinguishability (per-emitter purities xs):
        draw which emitters' photons are in an internal mode of their own (probability 1 - x_a),
        Fock-sample the others' photons together (their columns, `fock_sample`), route each lone
        photon classically by |U[r, modes[a][x_a]]|^2, and return the bin's counts [S, m] with the
        register amplitudes [S, L^k] over ALL label words: <t_good|U|x'_good> over the interfering
        emitters (`label_amplitudes(emitters=)`) times prod_a U[r_a, modes[a][x'_a]] for the lone
        ones — the operator stays diagonal in the label basis, so the update is exact
        (docs/SEQUENCE.md §13). Streams are grouped by their set of lone emitters.
        """
        S, k, m = cols.shape[0], self.k, self.m
        L = self.emitters[0].ops.shape[1]
        modes = torch.as_tensor([e.modes for e in self.emitters], dtype=torch.long, device=self.device)   # [k, L]
        lone = torch.rand((S, k), generator=generator) >= xs.unsqueeze(0)                     # [S, k] cpu
        counts = torch.zeros((S, m), dtype=torch.long, device=self.device)
        amp = torch.empty((S, L ** k), dtype=self.cdtype, device=self.device)
        Ub = self._U(U, b) if U_streams is None else None
        uniq, inv = torch.unique(lone, dim=0, return_inverse=True)
        for i, pattern in enumerate(uniq.tolist()):
            ids = (inv == i).nonzero()[:, 0]
            ids_d = ids.to(self.device)
            good = [a for a in range(k) if not pattern[a]]
            bad = [a for a in range(k) if pattern[a]]
            Us = None if U_streams is None else U_streams[ids_d]
            cg = self.fock_sample(U, cols[ids_d][:, good], generator, b=b, U_streams=Us, backend=fock_backend)   # [n_ids, m], g photons
            ag = self.label_amplitudes(U, cg, b=b, chunk_bytes=chunk_bytes, U_streams=Us, emitters=good,
                                       method=amp_method)                                             # [n_ids, L^g]
            full = ag.reshape(len(ids), *([L] * len(good)), *([1] * len(bad)))                 # good axes first, then singletons
            perm = [0] + [1 + good.index(a) if a in good else 1 + len(good) + bad.index(a) for a in range(k)]
            full = full.permute(perm)                                                         # emitter order
            for a in bad:
                col = modes[a, labels[ids_d, a]]                                              # the lone photon's column
                Urow = Ub[:, col].T if Us is None else Us[:, :, :].gather(2, col.view(-1, 1, 1).expand(-1, m, 1))[:, :, 0]   # [n_ids, m]
                p = (Urow.abs() ** 2).to("cpu", torch.float64)
                r = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=generator)[:, 0].to(self.device)
                counts[ids_d, r] += 1
                fac = (Ub[r][:, modes[a]] if Us is None else Us[torch.arange(len(ids), device=self.device), r][:, modes[a]])   # [n_ids, L]
                shape = [len(ids)] + [1] * k
                shape[1 + a] = L
                full = full * fac.reshape(shape)
            counts[ids_d] += cg
            amp[ids_d] = full.reshape(len(ids), -1)
        return counts, amp

    @staticmethod
    def _apply_local_batched(z: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        """Z [S, a^k] -> (x)_j mats[s, j] z_s, one matrix per stream and emitter, mats [S, k, b, a]."""
        S, k = z.shape[0], mats.shape[1]
        a, b = mats.shape[3], mats.shape[2]
        for j in range(k):
            left, right = b ** j, a ** (k - 1 - j)
            zj = z.reshape(S, left, a, right)
            z = torch.einsum("sba,slar->slbr", mats[:, j], zj)
        return z.reshape(S, b ** k)

    def __repr__(self):
        return (f"SequenceKernel(k={self.k} emitters, n={self.n} bins, m={self.m} modes, chi={self.chi}, "
                f"S_bin={self.S})")


__all__ = ["BinEmitter", "SequenceKernel"]
