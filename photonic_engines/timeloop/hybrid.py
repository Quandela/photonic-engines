"""Carried-photon engine: k spin emitters -> m rails -> per-rail loop ladders + a per-tick rail mixer,
marched as one register spins (x) Fock (docs/HYBRID.md).

Per tick t (= one period of `PeriodicNetworkKernel`):
    1. emissions: emitter j applies sum_l M_{j,t,l} (x) a†_{rail(j,l)} (its spin factor only — the
       spin register is the product of the k emitter spaces, chi = prod d_j, never a dense chi x chi op);
    2. the loop couplers: for every rail r and loop i of its ladder, amplitude damping on the head slot
       (one round trip, `loss["loop"]`) then the beam splitter G(theta, phi) on (rail r, head slot);
    3. the rail mixer U_t (m x m) as a fixed mesh of two-mode gates (`mesh_decompose`);
       `mixer="before"` swaps 2 and 3;
    4. detector damping on the rails, then the rails exit: projected (amplitudes), sampled
       (trajectories) or traced (density matrix, per-tick marginals).

Register = rail wires in transit + the slots of every delay line, <= p photons: D = C(p + R, R) Fock
states, R = m + sum of the delays, times chi = 2^k spin states. p = None is exact (p = every photon
the trains and the classical inputs can put in the register); a smaller p truncates — components
that would exceed p are dropped at emission and the dropped mass is reported (`leaked_mass`,
`sample(return_weights=True)`).

Limits (tests/timeloop/test_hybrid.py): no loops -> `SequenceKernel`; no emitter, one rail ->
`LoopRegisterKernel`; T = 1 -> the k-emission stage; oracle = the unrolled unitary over T ticks
fed with the product emitter MPS (`EmitterSLOSKernel`).
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from math import comb

import numpy as np
import torch

from .circuit import TimeLoopCircuit
from .kernel import LoopRegisterKernel, PeriodicNetworkKernel


def bs_matrix(theta, phi) -> torch.Tensor:
    """G(theta, phi) [..., 2, 2] = exp(-i theta [[0, e^{-i phi}], [e^{i phi}, 0]]) — the TimeLoopCircuit convention."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    ph = torch.as_tensor(phi, dtype=torch.float64)
    th, ph = torch.broadcast_tensors(th, ph)
    c, s = torch.cos(th).to(torch.complex128), torch.sin(th).to(torch.complex128)
    e = torch.exp(1j * ph.to(torch.complex128))
    return torch.stack([torch.stack([c, -1j * e.conj() * s], -1), torch.stack([-1j * e * s, c], -1)], -2)


def mesh_pairs(m: int) -> list[tuple[int, int]]:
    """Mode pairs of `mesh_decompose`'s gate list for an m-mode unitary (fixed for every U):
    the phase gates first (applied first), then the Reck elimination gates in application order.
    """
    phase = [(a, a + 1) for a in range(0, m - 1, 2)]
    if m % 2 and m > 1:
        phase.append((m - 2, m - 1))
    elim = [(i - 1, i) for j in range(m - 1) for i in range(m - 1, j, -1)]
    return phase + elim[::-1]


def mesh_decompose(U) -> np.ndarray:
    """Two-mode gates [n_gates, 2, 2] on `mesh_pairs(m)` whose product (first gate applied first) is U.

    Reck-type elimination T_N .. T_1 U = diag(d) (T on modes (i-1, i)), so U = T_1† .. T_N† diag(d): the phases are
    applied first as diag gates on disjoint pairs (an odd last mode as diag(1, d) on (m-2, m-1)), then T_N† .. T_1†.
    Every elimination step emits a gate (the identity when the column entry is already zero), so the structure does
    not depend on U — required for per-tick unitaries sharing one program. Host numpy, complex128. Mode-matrix
    convention: a†_b -> sum_a G[a, b] a†_a, the same as `fock_blocks`.
    """
    V = np.array(U, dtype=np.complex128, copy=True)
    m = V.shape[0]
    elim = []
    for j in range(m - 1):
        for i in range(m - 1, j, -1):
            x, y = V[i - 1, j], V[i, j]
            r = math.hypot(abs(x), abs(y))
            T = np.eye(2, dtype=np.complex128) if r == 0.0 else np.array([[x.conjugate() / r, y.conjugate() / r], [-y / r, x / r]])
            V[i - 1:i + 1] = T @ V[i - 1:i + 1]
            elim.append(T.conj().T)
    d = np.diagonal(V).copy()
    phase = [np.diag([d[a], d[a + 1]]) for a in range(0, m - 1, 2)]
    if m % 2 and m > 1:
        phase.append(np.diag([1.0, d[m - 1]]).astype(np.complex128))
    gates = phase + elim[::-1]
    return np.stack(gates) if gates else np.zeros((0, 2, 2), dtype=np.complex128)


def hybrid_register_dimension(k: int, p: int, rails: int, slots: int, d: int = 2) -> int:
    """d^k * C(p + R, R), R = rails + slots: the amplitudes the carried-photon register holds (the paper's knob)."""
    R = int(rails) + int(slots)
    return int(d) ** int(k) * comb(int(p) + R, R)


class HybridRegisterKernel(PeriodicNetworkKernel):
    """k spin emitters feeding m rails, a loop ladder on every rail, a rail mixer per tick: spins (x) Fock register.

    The spin axis has chi = prod d_j (each emitter's operators act on its own factor, applied locally); the Fock part
    is the `RegisterBasis` of the m rail wires + every delay-line slot with <= p photons. State [..., chi, D] complex.
    """

    def __init__(self, emitters: Sequence, rails: int, loops, T: int | None = None, carry: int | None = None,
                 inputs=None, mixer: str = "after", device: str = "cpu", dtype: torch.dtype = torch.complex128):
        """emitters: `sequence.BinEmitter` list (ops [T, Lab, d, d], modes = the rails of the labels, s0, final —
        all projected or all traced); may be empty (then T and/or inputs give the length).
        rails: m. loops: delays in ticks, one tuple for every rail or a list of tuples (one per rail; () = none).
        carry: register photon cap p (None = exact). inputs: classical photons [T, m] injected on the rails at the
        start of each tick (before the emissions). mixer: "after" (loops, then U) or "before" (U, then loops).
        """
        self.emitters = list(emitters)
        m = int(rails)
        if T is None:
            if self.emitters:
                T = self.emitters[0].n
            elif inputs is not None:
                T = int(np.asarray(inputs).reshape(-1, m).shape[0])
            else:
                raise ValueError("T: no emitter and no inputs")
        T = int(T)
        if any(e.n != T for e in self.emitters):
            raise ValueError("every emitter emits once per tick (BinEmitter.n == T)")
        used = [r for e in self.emitters for r in e.modes]
        if any(r < 0 or r >= m for r in used) or len(set(used)) != len(used):
            raise ValueError("emitter label rails must be distinct and < rails (as in SequenceKernel)")
        if mixer not in ("after", "before"):
            raise ValueError("mixer: 'after' or 'before'")
        if loops and not isinstance(loops[0], (tuple, list)):
            ladders = [tuple(int(x) for x in loops)] * m
        else:
            ladders = [tuple(int(x) for x in lad) for lad in loops] if loops else [()] * m
        if len(ladders) != m:
            raise ValueError("loops: one ladder per rail")
        self.ladders, self.m, self.k, self.mixer = ladders, m, len(self.emitters), mixer
        self.dl_rail = [r for r, lad in enumerate(ladders) for _ in lad]                         # delay line -> rail
        delays = [x for lad in ladders for x in lad]
        inp = np.zeros((T, m), dtype=np.int64) if inputs is None else np.asarray(inputs, dtype=np.int64).reshape(T, m)
        n_total = int(inp.sum()) + self.k * T
        self.n_total = n_total
        self.carry = None if carry is None or int(carry) >= n_total else int(carry)
        # program of one tick
        self.pairs = mesh_pairs(m)
        n_dl = len(delays)
        loop_ops = []
        for dl in range(n_dl):
            loop_ops.append(("loss", ("slot", dl), f"loop{dl}"))
            loop_ops.append(("gate", self.dl_rail[dl], ("slot", dl), dl))
        mix_ops = [("gate", a, b, n_dl + g) for g, (a, b) in enumerate(self.pairs)]
        program = (loop_ops + mix_ops) if mixer == "after" else (mix_ops + loop_ops)
        program += [("loss", r, "detector") for r in range(m)]
        self.n_dl = n_dl
        super().__init__(m, delays, program, T, inp, n_gates=n_dl + len(self.pairs), emitter=None,
                         device=device, dtype=dtype, n_max=n_total if self.carry is None else self.carry)
        # spin register (product of the emitters), emissions applied locally per emitter
        dev = self.device
        self.dims = [e.d for e in self.emitters]
        self.chi = int(np.prod(self.dims)) if self.emitters else 1
        if self.emitters:
            s0 = np.ones(1, dtype=complex)
            for e in self.emitters:
                s0 = np.kron(s0, e.s0)
            self._s0 = torch.as_tensor(s0, dtype=dtype, device=dev)
            finals = [e.final for e in self.emitters]
            if any(f is None for f in finals) and not all(f is None for f in finals):
                raise ValueError("either every emitter is projected (final) or every emitter is traced")
            if finals[0] is None:
                self._final = None
            else:
                f = np.ones(1, dtype=complex)
                for x in finals:
                    f = np.kron(f, x)
                self._final = torch.as_tensor(f, dtype=dtype, device=dev)
            self._em_local = [torch.as_tensor(e.ops, dtype=dtype, device=dev) for e in self.emitters]   # [T, Lab, d, d]
            self._emit_at = {t: list(range(self.k)) for t in range(T)}
        self._track: list | None = None

    # ---- sizes ------------------------------------------------------------------------------
    @property
    def slots(self) -> int:
        """Total delay-line slots (sum of the delays over every rail's ladder)."""
        return self.L

    def register_size(self) -> int:
        """Chi * D: complex amplitudes per trajectory (the paper's 2^k C(p + R, R))."""
        return self.chi * self.D

    # ---- emissions (local on the spin factor) ----------------------------------------------
    def _emit(self, psi: torch.Tensor, t: int, conj: bool = False) -> torch.Tensor:
        """Emissions of tick t on psi [..., chi, D]: for each emitter j, sum_l (M_{j,t,l} on spin factor j) (x)
        a†_{rail l}. Components that would exceed the photon cap are dropped (the raise table has no target);
        under `_track` the surviving squared norm per leading row is recorded (sampler weights).
        """
        if t not in self._emit_at:
            return psi
        before = self._norm2(psi, psi.dim() - 2) if self._track is not None else None
        lead = psi.shape[:-2]
        for j, e in enumerate(self.emitters):
            pre, dj = int(np.prod(self.dims[:j])), self.dims[j]
            post = self.chi // (pre * dj)
            ext = self._ext(psi)
            acc = 0
            for l, rail in enumerate(e.modes):
                src, coef = self._raise_tables(int(rail))
                x = (ext[..., src] * coef).reshape(*lead, pre, dj, post, self.D)
                M = self._em_local[j][t, l]
                if conj:
                    M = M.conj()
                acc = acc + torch.einsum("ab,...pbqd->...paqd", M, x)
            psi = acc.reshape(*lead, self.chi, self.D)
        if before is not None:
            after = self._norm2(psi, psi.dim() - 2)
            self._track.append(after / before)
            dead = after <= 0
            if dead.any():                            # every component dropped: weight 0 from here on; the row
                psi = psi.clone()                     # restarts from the vacuum so the sampler can continue
                psi[dead] = self._init_psi()           # (its outcomes are meaningless — w = 0 flags them)
        return psi

    # ---- density-matrix rail trace, per rail pattern block -----------------------------------
    def _trace_tables(self):
        """Per rail pattern q: (states with rail q [n_q], their cleared-rail indices [n_q]); cached."""
        if not hasattr(self, "_trace_tab"):
            occ = self.basis.occ.copy()
            occ[:, :self.m] = 0
            cleared = self.basis.index(occ)
            order = np.argsort(self.basis.rail_of, kind="stable")
            bounds = np.searchsorted(self.basis.rail_of[order], np.arange(self.Q + 1))
            self._trace_tab = [(torch.as_tensor(order[bounds[q]:bounds[q + 1]], device=self.device),
                                torch.as_tensor(cleared[order[bounds[q]:bounds[q + 1]]], device=self.device))
                               for q in range(self.Q) if bounds[q + 1] > bounds[q]]
        return self._trace_tab

    def _trace_rail(self, rho: torch.Tensor):
        """(pattern probabilities [Q], rho with the rail measured and cleared) — the same map as
        PeriodicNetworkKernel._trace_rail, sum_q P_q rho P_q moved to the cleared states, computed block by block on
        the states of each rail pattern (sum_q n_q^2 chi^2 entries instead of Q full passes over rho).
        """
        d = torch.einsum("sisi->i", rho).real
        p = torch.zeros((self.Q,), dtype=self.rdtype, device=rho.device).index_add(0, self._rail_of, d)
        out = torch.zeros_like(rho)
        for idx, cl in self._trace_tables():
            blk = rho[:, idx][:, :, :, idx]                                             # [chi, n, chi, n]
            z = torch.zeros((*blk.shape[:3], self.D), dtype=rho.dtype, device=rho.device)
            z[..., cl] = blk
            out.index_add_(1, cl, z)
        return p, out

    # ---- settings -> gate tensor ------------------------------------------------------------
    def gates(self, theta=None, phi=None, U=None) -> torch.Tensor:
        """[n_gates, T, 2, 2]: loop couplers G(theta, phi) (theta, phi [n_dl, T] or broadcastable; None = bar,
        theta = 0) then the mixer mesh of U ([m, m] static or [T, m, m] per tick; None = identity).
        """
        T = self.T
        th = torch.zeros((self.n_dl, T)) if theta is None else torch.as_tensor(theta, dtype=torch.float64).cpu()
        ph = torch.zeros(()) if phi is None else torch.as_tensor(phi, dtype=torch.float64).cpu()
        G = bs_matrix(th, ph).expand(self.n_dl, T, 2, 2) if self.n_dl else torch.zeros((0, T, 2, 2), dtype=torch.complex128)
        if U is None:
            U = np.eye(self.m, dtype=np.complex128)
        U = U.detach().cpu().numpy() if torch.is_tensor(U) else np.asarray(U)
        if self.m == 1 and not np.allclose(U, 1.0):
            raise ValueError("one rail: the mixer is a phase on the only rail, which the mesh does not carry")
        mix = np.stack([mesh_decompose(Ut) for Ut in U], 1) if U.ndim == 3 \
            else np.broadcast_to(mesh_decompose(U)[:, None], (len(self.pairs), T, 2, 2))
        mix = torch.as_tensor(np.array(mix, dtype=np.complex128), dtype=torch.complex128)
        return torch.cat([G, mix], 0).to(dtype=self.cdtype, device=self.device)

    @staticmethod
    def _loss_dict(loss, n_dl: int) -> dict | None:
        """dict(loop=eta or [eta per delay line], detector=eta) -> the program's channel names."""
        if not loss:
            return None
        loop = loss.get("loop")
        loop = [None] * n_dl if loop is None else ([float(loop)] * n_dl if np.isscalar(loop) else list(loop))
        out = {f"loop{i}": e for i, e in enumerate(loop)}
        out["detector"] = loss.get("detector")
        return out

    # ---- public API -------------------------------------------------------------------------
    def amplitudes(self, patterns, theta=None, phi=None, U=None) -> torch.Tensor:
        """Residual register amplitudes [P, (chi,) D] of output patterns [P, T, m] (spin projected on `final`
        when the emitters are projected). Lossless.
        """
        return super().amplitudes(patterns, self.gates(theta, phi, U))

    def probabilities(self, patterns, theta=None, phi=None, U=None) -> torch.Tensor:
        """[P] probabilities of output patterns [P, T, m] (leftover register photons and a traced spin summed).
        Under a photon cap these are the truncated (sub-normalised) probabilities.
        """
        return super().probabilities(patterns, self.gates(theta, phi, U))

    def distribution(self, theta=None, phi=None, U=None, tol: float = 1e-14):
        """(patterns [P, T, m], probs [P]) by branching every tick — small sizes only."""
        return super().distribution(self.gates(theta, phi, U), tol=tol)

    def tick_marginals(self, theta=None, phi=None, U=None, loss=None, return_register: bool = False):
        """[T, Q] distribution of the rail pattern exiting at every tick (`basis.rail_patterns`), by marching the
        density matrix [chi, D, chi, D] (traced spin only). Rows sum to the trace kept so far: 1 when exact,
        1 - leaked mass under a photon cap. loss: dict(loop=..., detector=...) Kraus channels.
        """
        return super().period_marginals(self.gates(theta, phi, U), loss=self._loss_dict(loss, self.n_dl),
                                        return_register=return_register)

    def leaked_mass(self, theta=None, phi=None, U=None, loss=None) -> float:
        """Exact probability mass dropped by the photon cap over the whole train (0 when carry is None)."""
        if self.carry is None:
            return 0.0
        marg = self.tick_marginals(theta, phi, U, loss=loss)
        return float(1.0 - marg[-1].sum())

    def sample_sectors(self, shots: int, theta=None, phi=None, U=None, generator: torch.Generator | None = None,
                       loss=None, return_stats: bool = False):
        """`sample` on photon-number sectors (`sector_sample`): same law, exact register only (carry None); returns
        [shots', T, m] (and the sector statistics).
        """
        if self.carry is not None:
            raise NotImplementedError("sector sampler: exact register only (carry=None)")
        pats, _, stats = sector_sample(self, shots, self.gates(theta, phi, U), chan=self._loss_dict(loss, self.n_dl),
                                       generator=generator)
        return (pats, stats) if return_stats else pats

    def sample(self, shots: int, theta=None, phi=None, U=None, generator: torch.Generator | None = None,
               loss=None, return_weights: bool = False, return_register: bool = False):
        """Trajectory samples [shots, T, m] (photon counts per rail per tick; see PeriodicNetworkKernel.sample).
        Under a photon cap each tick is renormalised after the dropped components, so the rows are draws from the
        per-step-renormalised truncated process, not from the exact distribution; return_weights=True also returns
        w [shots] = prod_t (surviving squared norm at emission t): E[w] = 1 - leaked mass (`leaked_mass`), and
        w = 0 marks a row whose register lost every component (its later outcomes are not meaningful). Exact
        (w = 1) when carry is None. Traced spin only for the weights.
        """
        if return_weights and self._final is not None:
            raise NotImplementedError("sampler weights with a projected spin (rejection sampling)")
        self._track = [] if (return_weights or self.carry is not None) else None
        try:
            out = super().sample(shots, self.gates(theta, phi, U), generator=generator,
                                 loss=self._loss_dict(loss, self.n_dl), return_register=return_register)
            track = self._track
        finally:
            self._track = None
        if not return_weights:
            return out
        w = torch.ones(shots, dtype=self.rdtype, device=self.device)
        for s in track:
            w = w * s.reshape(shots)
        return (*out, w) if return_register else (out, w)

    def __repr__(self):
        return (f"HybridRegisterKernel(k={self.k}, rails={self.m}, ladders={self.ladders}, T={self.T}, "
                f"carry={self.carry}, chi={self.chi}, D={self.D})")



class SuperpositionLoopKernel(LoopRegisterKernel):
    """Superposition boson sampling into an ORCA-type loop device: one rail, source i in bin `state.modes[i]` emitting
    sum_q a[i, q] |q> coherently, then the loops of `circuit` (docs/HYBRID.md §5).

    The photon number is indefinite, so the register carries every photon number up to sources * (d - 1) at once
    (the dense `RegisterBasis` is graded already): D = C(N + L + 1, L + 1), N = sources * (d - 1), the same as a Fock
    input of N photons. Injection at bin b: psi -> sum_q a[i, q] / sqrt(q!) (a†)^q psi on the (empty) rail. Everything
    else — gates, loss channels, uniform loss by thinning, the three modes — is LoopRegisterKernel's. Partial
    distinguishability is refused (the x-model of the loop kernel is for Fock inputs).
    """

    def __init__(self, circuit: TimeLoopCircuit, state, device: str = "cpu", dtype: torch.dtype = torch.complex128):
        """state: a `ProductSuperposition` whose modes are bins of `circuit` (distinct, < T)."""
        amps = np.asarray(state.amps, dtype=np.complex128)
        modes = [int(x) for x in state.modes]
        if max(modes) >= circuit.T:
            raise ValueError("source bins must be < T")
        d = amps.shape[1]
        occ = [0] * circuit.T
        for b in modes:
            occ[b] = d - 1                                       # sizes the register: N = sources * (d - 1)
        super().__init__(circuit, occ, device=device, dtype=dtype)
        self.state = state
        q0 = int(self.basis.pattern_id(np.zeros((1, 1), dtype=np.int64))[0])
        self._input_q = [q0] * self.T                             # nothing injected as Fock: the sources act in _emit
        inv_sqrt_fact = np.exp(-0.5 * np.array([math.lgamma(q + 1) for q in range(d)]))   # float for any d (q! > int64 at 21)
        self._sbs = {b: torch.as_tensor(amps[i] * inv_sqrt_fact, dtype=dtype, device=self.device) for i, b in enumerate(modes)}
        self._emit_at = {b: [i] for i, b in enumerate(modes)}

    def _emit(self, psi: torch.Tensor, t: int, conj: bool = False) -> torch.Tensor:
        """Source of bin t on psi [..., chi, D]: sum_q a_q / sqrt(q!) (a†_rail)^q psi (conj: conjugated amplitudes)."""
        if t not in self._sbs:
            return psi
        a = self._sbs[t].conj() if conj else self._sbs[t]
        src, coef = self._raise_tables(0)
        acc, cur = a[0] * psi, psi
        for q in range(1, a.shape[0]):
            cur = self._ext(cur)[..., src] * coef
            acc = acc + a[q] * cur
        return acc

    def bin_marginals(self, theta, phi=None, loss=None, return_register: bool = False, indist=None, **kw):
        """LoopRegisterKernel.bin_marginals for the superposed sources (no indist)."""
        if indist is not None:
            raise NotImplementedError("partial distinguishability: Fock inputs only on the loop kernel")
        return super().bin_marginals(theta, phi, loss=loss, return_register=return_register)

    def sample(self, shots: int, theta, phi=None, generator: torch.Generator | None = None,
               loss=None, return_register: bool = False, threshold: bool = False, indist=None):
        """LoopRegisterKernel.sample for the superposed sources (no indist)."""
        if indist is not None:
            raise NotImplementedError("partial distinguishability: Fock inputs only on the loop kernel")
        return super().sample(shots, theta, phi, generator=generator, loss=loss, return_register=return_register,
                              threshold=threshold)


# ---------------------------------------------------------------------------------------------------
# Photon-number sectors: in a trajectory (and for a fixed output pattern) the register photon number is
# definite — an emission adds exactly one photon per emitter, gates conserve it, an exit or a loss jump
# removes a known number — so a shot only needs the basis of its current sector (docs/HYBRID.md §6).
# ---------------------------------------------------------------------------------------------------
def _compositions(R: int, n: int) -> np.ndarray:
    """All occupation rows of R modes with total exactly n, [C(n + R - 1, R - 1), R] int64 (R = 0: one empty row iff n = 0)."""
    from .basis import _fock_rows
    if R == 0:
        return np.zeros((1 if n == 0 else 0, 0), dtype=np.int64)
    if R == 1:
        return np.array([[n]], dtype=np.int64)
    head = _fock_rows(R - 1, n)                                                   # sum <= n
    return np.concatenate([head, n - head.sum(1, keepdims=True)], 1)


class _Lookup:
    """Row -> index over a fixed set of occupation rows (byte keys; occupations < 256)."""

    def __init__(self, occ: np.ndarray):
        self.R, self.D = occ.shape[1], occ.shape[0]
        if self.R:
            v = self._key(occ)
            self.order = np.argsort(v, kind="stable")
            self.sorted = v[self.order]

    @staticmethod
    def _key(occ: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(np.clip(occ, 0, 255).astype(np.uint8))
        return a.view(np.dtype((np.void, a.shape[-1]))).reshape(a.shape[:-1])

    def index(self, occ: np.ndarray) -> np.ndarray:
        """Indices of rows [..., R] (-1 where absent: negative, >= 256 or not in the set)."""
        occ = np.asarray(occ, dtype=np.int64)
        if self.R == 0:
            return np.zeros(occ.shape[:-1], dtype=np.int64) if self.D else -np.ones(occ.shape[:-1], dtype=np.int64)
        ok = (occ.min(-1) >= 0) & (occ.max(-1) < 256)
        v = self._key(occ)
        pos = np.minimum(np.searchsorted(self.sorted, v), self.D - 1)
        found = ok & (self.sorted[pos] == v)
        return np.where(found, self.order[pos], -1)


class _SectorTables:
    """Per-sector bases and index tables of the hybrid register, built on demand and cached (host numpy, device copies).

    full(n): the R = m + L register modes (rails first) with exactly n photons — the state during a tick.
    slot(n): the L delay-line slots with exactly n photons — the state between ticks (rails empty).
    """

    def __init__(self, m: int, L: int, device):
        self.m, self.L, self.R, self.dev = m, L, m + L, device
        self._full: dict[int, dict] = {}
        self._slot: dict[int, tuple] = {}
        self.seconds, self._depth = 0.0, 0            # host time spent building tables (one-time cost, reported apart)

    def _t(self, x):
        return torch.as_tensor(x, device=self.dev)

    def slot(self, n: int):
        """(occ [Ds, L], lookup) of the slot sector n."""
        if n not in self._slot:
            occ = _compositions(self.L, n)
            self._slot[n] = (occ, _Lookup(occ))
        return self._slot[n]

    def full(self, n: int) -> dict:
        """occ, lookup, rail patterns, per-state rail id / slot index, embed map of slot(n)."""
        if n not in self._full:
            from .basis import _fock_rows
            occ = _compositions(self.R, n)
            lk = _Lookup(occ)
            pats = _fock_rows(self.m, n)                                          # [Q, m], lexicographic, sum <= n
            rid = _Lookup(pats).index(occ[:, :self.m])                            # [D]
            rtot = occ[:, :self.m].sum(1)
            sidx = np.empty(occ.shape[0], dtype=np.int64)
            for s in np.unique(n - rtot):
                sel = (n - rtot) == s
                sidx[sel] = self.slot(int(s))[1].index(occ[sel, self.m:])
            socc = self.slot(n)[0]
            emb = lk.index(np.concatenate([np.zeros((socc.shape[0], self.m), dtype=np.int64), socc], 1))
            self._full[n] = {"occ": occ, "lk": lk, "D": occ.shape[0], "pats": pats, "Q": pats.shape[0],
                             "ptot": pats.sum(1), "rid": rid, "sidx": sidx, "emb": emb,
                             "rid_t": self._t(rid), "pats_t": self._t(pats), "ptot_t": self._t(pats.sum(1)),
                             "emb_t": self._t(emb), "pair": {}, "raise": {}, "loss": {}, "exit": {}}
        return self._full[n]

    def pair(self, n: int, a: int, b: int):
        """Gate tables of pair (a, b) in full(n): (member [G, n+1] or None, gk, inv, bounds, per-k flat members, inv_c)."""
        F = self.full(n)
        if (a, b) not in F["pair"]:
            occ = F["occ"]
            k = occ[:, a] + occ[:, b]
            rest = occ.copy()
            rest[:, a] = 0
            rest[:, b] = k
            _, gid = np.unique(_Lookup._key(rest), return_inverse=True)
            gid = gid.reshape(-1)
            G = int(gid.max()) + 1 if gid.size else 0
            gk = np.zeros(G, dtype=np.int64)
            gk[gid] = k
            order = np.argsort(gk, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(G)
            gid = rank[gid]
            gk = gk[order]
            member = -np.ones((G, n + 1), dtype=np.int64)
            member[gid, occ[:, a]] = np.arange(F["D"])
            bounds = [int(x) for x in np.searchsorted(gk, np.arange(n + 2))]
            mk, inv_c, off = [], np.empty(F["D"], dtype=np.int64), 0
            for kk in range(n + 1):
                flat = member[bounds[kk]:bounds[kk + 1], :kk + 1].reshape(-1)
                inv_c[flat] = off + np.arange(flat.size)
                mk.append(self._t(flat))
                off += flat.size
            fused = G * (n + 1) <= 2 * F["D"] and G * (n + 1) ** 2 * 16 <= (1 << 28)
            F["pair"][(a, b)] = (self._t(np.where(member < 0, F["D"], member)) if fused else None,
                                 self._t(gk) if fused else None, self._t(gid * (n + 1) + occ[:, a]) if fused else None,
                                 bounds, mk, self._t(inv_c))
        return F["pair"][(a, b)]

    def raise_(self, n: int, w: int):
        """a† on mode w from full(n-1) to full(n): (src [D_n] index into full(n-1) or D_{n-1}, coef [D_n])."""
        F = self.full(n)
        if w not in F["raise"]:
            tgt = F["occ"].copy()
            tgt[:, w] -= 1
            P = self.full(n - 1)
            src = P["lk"].index(tgt)
            coef = np.where(src >= 0, np.sqrt(np.maximum(F["occ"][:, w], 0)), 0.0)
            F["raise"][w] = (self._t(np.where(src < 0, P["D"], src)), self._t(coef))
        return F["raise"][w]

    def loss(self, n: int, w: int, j: int):
        """Kraus E_j on mode w from full(n) to full(n-j): (src index into full(n) or D_n, sqrt C(n_w + j, j), n_w)."""
        F = self.full(n)
        if (w, j) not in F["loss"]:
            T_ = self.full(n - j)
            tgt = T_["occ"].copy()
            nw = tgt[:, w].copy()
            tgt[:, w] += j
            src = F["lk"].index(tgt)
            coef = np.where(src >= 0, np.sqrt([comb(int(a) + j, j) for a in nw]), 0.0) if nw.size else np.zeros(0)
            F["loss"][(w, j)] = (self._t(np.where(src < 0, F["D"], src)), self._t(np.asarray(coef, dtype=np.float64)),
                                 self._t(nw.astype(np.float64)))
        return F["loss"][(w, j)]

    # ---- ext-form tables of the fast path (`fast=True`): every state carries one trailing zero column (index D),
    # so a gather reads "absent" from the sentinel without a torch.cat copy, and every table ends on the sentinel
    # (its output is again in ext form). -------------------------------------------------------------------------
    def emb_x(self, n: int):
        """slot(n) ext [.., Ds+1] -> full(n) ext [.., D+1] gather index (D+1)."""
        F = self.full(n)
        if "emb_x" not in F:
            Ds = self.slot(n)[0].shape[0]
            tab = np.full(F["D"] + 1, Ds, dtype=np.int64)
            tab[F["emb"]] = np.arange(Ds)
            F["emb_x"] = self._t(tab)
        return F["emb_x"]

    def raise_x(self, n: int, w: int):
        """a† on mode w, full(n-1) ext -> full(n) ext: (src [D_n+1], coef [D_n+1])."""
        F = self.full(n)
        key = ("x", w)
        if key not in F["raise"]:
            src, coef = self.raise_(n, w)
            P = self.full(n - 1)
            F["raise"][key] = (torch.cat([src, self._t([P["D"]])]), torch.cat([coef, coef.new_zeros(1)]))
        return F["raise"][key]

    def pair_x(self, n: int, a: int, b: int):
        """Gate tables of pair (a, b) on full(n) ext: ("fused", member [G+1, n+1], gk [G+1], inv [D+1]) with one
        all-sentinel group whose (zero) output is the trailing zero, or ("k", bounds, mk, inv_c [D+1]).
        """
        F = self.full(n)
        key = ("x", a, b)
        if key not in F["pair"]:
            member, gk, inv, bounds, mk, inv_c = self.pair(n, a, b)
            if member is not None and n >= 2:
                G = member.shape[0]
                F["pair"][key] = ("fused", torch.cat([member, member.new_full((1, n + 1), F["D"])]),
                                  torch.cat([gk, gk.new_zeros(1)]), torch.cat([inv, self._t([G * (n + 1)])]))
            else:
                F["pair"][key] = ("k", bounds, mk, torch.cat([inv_c, self._t([F["D"]])]))
        return F["pair"][key]

    def loss_x(self, n: int, w: int, j: int):
        """Kraus E_j on mode w, full(n) ext -> full(n-j) ext: (src, coef, n_w), each [D_{n-j}+1]."""
        F = self.full(n)
        key = ("x", w, j)
        if key not in F["loss"]:
            src, coef, nw = self.loss(n, w, j)
            F["loss"][key] = (torch.cat([src, self._t([F["D"]])]), torch.cat([coef, coef.new_zeros(1)]),
                              torch.cat([nw, nw.new_zeros(1)]))
        return F["loss"][key]

    def exit_x(self, n: int):
        """Rail exit of full(n) ext, all photon counts at once: (rid [D+1] pattern of each state (sentinel -> 0),
        table [Q, Ds(n)+1] of full(n) indices padded with the sentinel D: row q holds the states with rails = pattern q,
        in slot(n - d) order, d = |q| — its first Ds(n-d)+1 columns are that slot sector in ext form).
        """
        F = self.full(n)
        if "x" not in F["exit"]:
            Ds = self.slot(n)[0].shape[0]
            tab = np.full((F["Q"], Ds + 1), F["D"], dtype=np.int64)
            tab[F["rid"], F["sidx"]] = np.arange(F["D"])
            F["exit"]["x"] = (self._t(np.concatenate([F["rid"], [0]])), self._t(tab))
        return F["exit"]["x"]


def _timed(method):
    """Accumulate the wall time of the outermost table-building call in `self.seconds` (cache hits included: ~1 us)."""
    def wrapper(self, *a):
        if self._depth:
            return method(self, *a)
        self._depth += 1
        t0 = time.perf_counter()
        try:
            return method(self, *a)
        finally:
            self._depth -= 1
            self.seconds += time.perf_counter() - t0
    wrapper.__doc__ = method.__doc__
    return wrapper


for _name in ("slot", "full", "pair", "raise_", "loss", "emb_x", "raise_x", "pair_x", "loss_x", "exit_x"):
    setattr(_SectorTables, _name, _timed(getattr(_SectorTables, _name)))


def _sector_gate_x(psi: torch.Tensor, tab, B: torch.Tensor, n: int) -> torch.Tensor:
    """Two-mode gate on ext-form states psi [S, chi, D_n+1] of sector n with the `pair_x` tables (output in ext
    form); B [N+1, N+1, N+1] padded Fock blocks (N >= n). Fused path: gather [G+1, S chi, n+1], one bmm with the
    blocks B[k] of each group, scatter back; per-k path otherwise.
    """
    lead = psi.shape[:-1]
    if tab[0] == "fused":
        _, member, gk, inv = tab
        G1 = member.shape[0]
        x = psi[..., member].reshape(-1, G1, n + 1).transpose(0, 1)                         # [G+1, S chi, n+1]
        new = torch.bmm(x, B[gk][:, :n + 1, :n + 1].transpose(-1, -2))
        return new.transpose(0, 1).reshape((*lead, G1 * (n + 1)))[..., inv]
    _, bounds, mk, inv_c = tab
    parts = []
    for k in range(n + 1):
        g0, g1 = bounds[k], bounds[k + 1]
        if g1 == g0:
            continue
        blk = psi[..., mk[k]].reshape((*lead, g1 - g0, k + 1)) @ B[k, :k + 1, :k + 1].transpose(-1, -2)
        parts.append(blk.reshape((*lead, -1)))
    parts.append(psi.new_zeros((*lead, 1)))
    return torch.cat(parts, -1)[..., inv_c]


def _split_by(items):
    """Regroup rows by a device-side outcome with ONE host sync for all groups: items [(n, rows, key)], key [S] long
    in [0, n]. Returns [(i, n, v, rows', idx)]: group i, outcome v, its rows and their device indices in group i
    (stable order). Counts come from scatter_add (CUDA bincount would sync once per call).
    """
    orders, cnts = [], []
    for n, _, key in items:
        orders.append(torch.argsort(key, stable=True))
        cnts.append(torch.zeros(n + 1, dtype=torch.long, device=key.device).scatter_add_(0, key, torch.ones_like(key)))
    host = torch.cat(cnts).cpu().tolist() if cnts else []                               # the one sync
    out, off = [], 0
    for i, ((n, rows, _), order) in enumerate(zip(items, orders)):
        s = 0
        for v, cv in enumerate(host[off:off + n + 1]):
            if cv:
                idx = order[s:s + cv]
                out.append((i, n, v, rows[idx], idx))
                s += cv
        off += n + 1
    return out


def _sector_march(K: PeriodicNetworkKernel, shots: int, gates, chan: dict | None = None,
                  generator: torch.Generator | None = None, force=None, chain: bool = True):
    """The sector march of `sector_sample`, built to keep the per-tick launch count low
    (validated in run 20261012_075702):
    (1) the gates' Fock blocks are built for a chunk of ticks at once (recursion, padded to a running photon cap);
    (2) the rail exit and the loss jumps regroup the shots with one host sync per stage for all photon-number groups
    (device-side sort + counts), and the exit gathers all outcomes of a group at once from a padded table; (3) states
    are kept in ext form (one trailing zero column), so no gather pays a `torch.cat` copy. chain (default): hand
    one-loop Fock-input kernels to `_loop_chain_sample`; chain=False forces the march (tests compare the two).
    """
    if isinstance(K, SuperpositionLoopKernel):
        raise NotImplementedError("sector sampler: the photon number of a superposed source is indefinite")
    prog = _head_program(K) if chain else None
    if prog is not None:
        return _loop_chain_sample(K, shots, gates, chan=chan, generator=generator, force=force, prog=prog)
    from .kernel import _GenSpec, fock_blocks
    dev, cdt, chi = K.device, K.cdtype, K.chi
    tabs = K.__dict__.get("_sector_tables")
    if tabs is None:
        tabs = K.__dict__["_sector_tables"] = _SectorTables(K.n_rail, K.L, dev)
    if isinstance(gates, _GenSpec):
        V, h = None, gates.h.to(dtype=cdt, device=dev).expand(K.n_gates, K.T, 2, 2)
    else:
        V, h = torch.as_tensor(gates).to(dtype=cdt, device=dev).expand(K.n_gates, K.T, 2, 2), None
    local = getattr(K, "_em_local", None)
    dims = getattr(K, "dims", None)
    T, nr = K.T, K.n_rail
    n_tot = int(np.asarray(K.inputs).sum()) + sum(len(v) for v in K._emit_at.values())     # photons ever injected
    pats = torch.zeros((shots, T, nr), dtype=torch.long, device=dev)
    logp = torch.zeros(shots, dtype=torch.float64, device=dev) if force is not None else None
    if force is not None:
        force = torch.as_tensor(np.asarray(force), dtype=torch.long, device=dev).reshape(shots, T, nr)
    psi = torch.zeros((shots, chi, 2), dtype=cdt, device=dev)                             # slot(0) ext: [vac, 0]
    psi[:, :, 0] = K._s0
    groups = {0: (torch.arange(shots, device=dev), psi)}
    stats = {"max_amplitudes": 0, "mean_n": [], "table_seconds": 0.0}
    t_tab = tabs.seconds
    blk = {"t0": 0, "t1": 0, "cap": -1, "B": None}

    def blocks(t, n):                                     # fix 1: Fock blocks of a chunk of ticks, cap >= n
        if not (blk["t0"] <= t < blk["t1"]) or n > blk["cap"]:
            cap = min(max(n + 4, blk["cap"]), max(n, n_tot))
            C = max(1, min(T - t, (1 << 28) // (K.n_gates * (cap + 1) ** 3 * 16)))
            # contiguous: with bpp > 1 (emitter) h is a transposed view, and matrix_exp rejects a mid-range slice of it
            Vc = V[:, t:t + C] if V is not None else torch.linalg.matrix_exp((-1j * h[:, t:t + C]).contiguous())
            blk.update(t0=t, t1=t + C, cap=cap, B=fock_blocks(Vc, cap))
        return blk["B"][:, t - blk["t0"]]                                               # [n_gates, cap+1, ...]

    def raise_(x, n, w):
        src, coef = tabs.raise_x(n, w)
        return x[..., src] * coef

    def merge(parts):
        out = {}
        for n, rows, x in parts:
            out.setdefault(n, []).append((rows, x))
        return {n: (v[0] if len(v) == 1 else (torch.cat([r for r, _ in v]), torch.cat([x for _, x in v])))
                for n, v in out.items()}

    for t in range(T):
        items = []
        for n, (rows, x0) in groups.items():
            x = x0[..., tabs.emb_x(n)]
            for r, c in enumerate(K.inputs[t]):
                for _ in range(int(c)):
                    n += 1
                    x = raise_(x, n, r)
                if c > 1:
                    x = x / math.sqrt(math.factorial(int(c)))
            for kk in K._emit_at.get(t, ()):
                n += 1
                acc = 0
                if local is not None:
                    for l, rail in enumerate(K.emitters[kk].modes):
                        acc = acc + _local(local[kk][t, l], raise_(x, n, int(rail)), dims, kk)
                else:
                    for l in range(K._em_rails.shape[1]):
                        acc = acc + torch.einsum("ab,sbd->sad", K._em_ops[kk, l], raise_(x, n, int(K._em_rails[kk, l])))
                x = acc
            items.append((n, rows, x))
        Bt = blocks(t, max(n for n, _, _ in items))
        for op in K.program:
            if op[0] == "gate":
                a, b = K.mode(op[1], t), K.mode(op[2], t)
                B = Bt[op[3]]
                items = [(n, rows, _sector_gate_x(x, tabs.pair_x(n, a, b), B, n)) for n, rows, x in items]
                continue
            eta = K._eta(chan, op[2])
            if eta is None:
                continue
            if force is not None:
                raise NotImplementedError("sector sampler: forced outcomes are lossless only")
            w_mode = K.mode(op[1], t)
            keyed, outs = [], []
            for n, rows, x in items:
                ys, ws = [], []
                for j in range(n + 1):
                    src, coef, nw = tabs.loss_x(n, w_mode, j)
                    y = x[..., src] * (coef * eta ** (nw / 2) * (1 - eta) ** (j / 2))
                    ys.append(y)
                    ws.append((y.real ** 2 + y.imag ** 2).sum((-1, -2)))
                W = torch.stack(ws, 1)
                keyed.append((n, rows, torch.multinomial(W / W.sum(-1, keepdim=True), 1, generator=generator)[:, 0]))
                outs.append(ys)
            items = [(n, r_, x_) for n, (r_, x_) in merge([(n - j, r_, outs[i][j][idx])
                                                            for i, n, j, r_, idx in _split_by(keyed)]).items()]
        # exit the rails (fix 2: all outcomes of a group in one gather, one sync for the regrouping)
        keyed, ys = [], []
        for n, rows, x in items:
            F = tabs.full(n)
            stats["max_amplitudes"] = max(stats["max_amplitudes"], chi * F["D"])
            rid, tab = tabs.exit_x(n)
            w = (x.real ** 2 + x.imag ** 2).sum(-2)                                             # [S, D+1]
            p = torch.zeros((x.shape[0], F["Q"]), dtype=w.dtype, device=dev).index_add_(1, rid, w)
            tot = p.sum(-1, keepdim=True)
            if force is None:
                q = torch.multinomial(p / tot, 1, generator=generator)[:, 0]
            else:
                fq = force[rows, t]
                q = torch.as_tensor(_Lookup(F["pats"]).index(fq.cpu().numpy()), device=dev)
                ok = q >= 0
                qq = q.clamp(min=0)
                pq = torch.where(ok, p.gather(1, qq[:, None])[:, 0], torch.zeros_like(tot[:, 0]))
                logp[rows] += torch.log(pq / tot[:, 0])
                q = qq
            pats[rows, t] = F["pats_t"][q]
            y = x.gather(-1, tab[q][:, None, :].expand(-1, chi, -1))                             # [S, chi, Ds(n)+1]
            nrm = torch.sqrt((y.real ** 2 + y.imag ** 2).sum((-1, -2))).clamp_min(1e-300)
            ys.append(y / nrm[:, None, None])
            keyed.append((n, rows, F["ptot_t"][q]))
        groups = merge([(n - d, r_, ys[i][idx][..., :tabs.slot(n - d)[0].shape[0] + 1])
                        for i, n, d, r_, idx in _split_by(keyed)])
        stats["mean_n"].append(sum(n * r_.numel() for n, (r_, _) in groups.items()) / shots)
    keep = None
    if K._final is not None:
        acc = torch.zeros(shots, dtype=torch.float64, device=dev)
        for rows, x in groups.values():
            f = torch.einsum("a,sad->sd", K._final.conj(), x)
            acc[rows] = (f.real ** 2 + f.imag ** 2).sum(-1).to(torch.float64)
        if force is not None:
            logp = logp + torch.log(acc)
        else:
            keep = torch.rand(shots, generator=generator, device=dev, dtype=torch.float64) < acc
            pats = pats[keep]
    stats["table_seconds"] = tabs.seconds - t_tab
    return pats, logp, stats


def _head_program(K) -> tuple | None:
    """(gate index, loop-loss channel or None, detector channel or None) when K is a one-rail, one-loop, stride-1
    register with Fock input only and the program [loss(head)?, gate(rail, head), loss(rail)?] — the case of
    `_loop_chain_sample`; else None.
    """
    if K.n_rail != 1 or len(K.delays) != 1 or tuple(K.strides) != (1,) or K.chi != 1 or K._emit_at \
            or getattr(K, "_em_local", None) is not None or K._final is not None:
        return None
    head, rail = ("slot", 0, 0), 0
    gate, lloss, dloss = None, None, None
    for op in K.program:
        if op[0] == "gate" and gate is None and op[1] == rail and op[2] == head:
            gate = op[3]
        elif op[0] == "loss" and op[1] == head and gate is None and lloss is None:
            lloss = op[2]
        elif op[0] == "loss" and op[1] == rail and gate is not None and dloss is None:
            dloss = op[2]
        else:
            return None
    return None if gate is None else (gate, lloss, dloss)


def gate_column(V: torch.Tensor, c, s_max: int, d_max: int, c_max: int | None = None) -> torch.Tensor:
    """A[..., s, d] = <d, c+s-d| V |c, s> (mode a first, `fock_blocks` convention: V a† V† = V00 a† + V10 b†) for
    gates V [..., 2, 2], input c [...] photons on mode a, s = 0..s_max on mode b, d = 0..d_max on mode a out
    (zero for d > c + s). Closed form: sqrt(d! (k-d)! / (c! s!)) sum_i C(c, i) C(s, d-i) V00^i V10^(c-i) V01^(d-i)
    V11^(s-d+i), k = c + s, summed in log-magnitude + phase (no factorial overflow up to k ~ 10^3). c_max: max(c) if
    the caller knows it (saves a device sync).
    """
    c = torch.as_tensor(c, device=V.device).to(torch.float64)
    rdt = torch.float64
    s = torch.arange(s_max + 1, device=V.device, dtype=rdt)[:, None]
    d = torch.arange(d_max + 1, device=V.device, dtype=rdt)[None, :]
    lg = torch.lgamma
    lead = c.shape
    c_ = c.reshape(*lead, 1, 1)
    k = c_ + s
    pref = 0.5 * (lg(d + 1) + lg((k - d).clamp(min=0) + 1) - lg(c_ + 1) - lg(s + 1))
    v = [V[..., i, j].reshape(*lead, 1, 1) for i, j in ((0, 0), (1, 0), (0, 1), (1, 1))]
    lm = [torch.log(x.abs()) for x in v]
    ph = [torch.angle(x) for x in v]
    out = torch.zeros((*lead, s_max + 1, d_max + 1), dtype=V.dtype, device=V.device)
    if c_max is None:
        c_max = int(c.max().item()) if c.numel() else 0
    for i in range(c_max + 1):
        ex = [torch.full_like(k, float(i)), c_ - i, d - i, s - d + i]                        # exponents of v00..v11
        ok = (c_ >= i) & (d - i >= 0) & (d - i <= s) & (s - d + i >= 0)
        lc = lg(c_ + 1) - lg(torch.full_like(c_, float(i)) + 1) - lg((c_ - i).clamp(min=0) + 1) \
            + lg(s + 1) - lg((d - i).clamp(min=0) + 1) - lg((s - d + i).clamp(min=0) + 1)
        logm = pref + lc
        phase = torch.zeros_like(logm)
        for e, a, p in zip(ex, lm, ph):
            e = e.clamp(min=0)
            logm = logm + torch.where(e > 0, e * a, torch.zeros_like(logm))                 # 0^0 = 1
            phase = phase + e * p
        term = torch.exp(torch.complex(logm, phase))
        out = out + torch.where(ok & (d <= k), term, torch.zeros_like(term)).to(V.dtype)
    return out


def _loop_chain_sample(K: PeriodicNetworkKernel, shots: int, gates, chan: dict | None = None,
                       generator: torch.Generator | None = None, force=None, prog: tuple | None = None):
    """Exact sampler of a one-rail, one-loop register with Fock input (`_head_program`; docs/HYBRID.md §6d): the
    loop register is ONE Fock basis state between ticks, so a shot is its L slot occupations (the property ORCA's
    progressive simulator relies on; docs/HYBRID.md).

    Induction: the loop starts empty; at a tick the rail holds exactly c photons (the input) and the head slot a
    definite s; the gate conserves k = c + s and the rail is measured at once, so the head is left with the definite
    k - d. Outcome d has probability |<d, k-d| V |c, s>|^2 (`gate_column`). Loop loss on the head before the gate is a
    binomial thinning of s (a Fock state through a loss channel); detector loss a binomial thinning of d. Per tick:
    O(1) batched ops over all shots and O(s + c) work per shot; no host sync except one per chunk of ticks (photon cap).
    Same law and returns as `sector_sample` (max_amplitudes = 1, the register is one basis state). prog: the
    `_head_program(K)` tuple when the caller has it.
    """
    from .kernel import _GenSpec
    g_idx, lch, dch = prog if prog is not None else _head_program(K)
    dev, cdt = K.device, K.cdtype
    rdt = torch.float64 if cdt == torch.complex128 else torch.float32
    if isinstance(gates, _GenSpec):
        V, h = None, gates.h.to(dtype=cdt, device=dev).expand(K.n_gates, K.T, 2, 2)
    else:
        V, h = torch.as_tensor(gates).to(dtype=cdt, device=dev).expand(K.n_gates, K.T, 2, 2), None
    eta_l = K._eta(chan, lch) if lch is not None else None
    eta_d = K._eta(chan, dch) if dch is not None else None
    if force is not None and (eta_l is not None or eta_d is not None):
        raise NotImplementedError("sector sampler: forced outcomes are lossless only")
    T, L = K.T, K.L
    cin = np.asarray(K.inputs, dtype=np.int64).reshape(T)
    ccum = np.concatenate([[0], np.cumsum(cin)])
    pats = torch.zeros((shots, T, 1), dtype=torch.long, device=dev)
    logp = torch.zeros(shots, dtype=torch.float64, device=dev) if force is not None else None
    if force is not None:
        force = torch.as_tensor(np.asarray(force), dtype=torch.long, device=dev).reshape(shots, T)
    occ = torch.zeros((shots, L), dtype=torch.long, device=dev)                            # slot occupations
    tot = torch.zeros(shots, dtype=torch.long, device=dev)                                 # photons in the loop
    tot_t = torch.zeros((T, shots), dtype=torch.long, device=dev)
    ar = torch.arange(shots, device=dev)
    blk = {"t0": 0, "t1": 0, "A": None}
    chunk = 64

    for t in range(T):
        if t >= blk["t1"]:                    # amplitude table of the next chunk; cap = a bound on any head count in it
            C = min(chunk, T - t)
            s_max = int(tot.max().item()) + int(ccum[t + C] - ccum[t])                      # the one host sync per chunk
            c_max = int(cin[t:t + C].max())
            # contiguous: as in _sector_march (h may be a strided view; matrix_exp rejects such a slice)
            Vc = V[g_idx, t:t + C] if V is not None else torch.linalg.matrix_exp((-1j * h[g_idx, t:t + C]).contiguous())
            blk.update(t0=t, t1=t + C, A=gate_column(Vc, torch.as_tensor(cin[t:t + C], device=dev), s_max,
                                                     s_max + c_max, c_max=c_max))
        c = int(cin[t])
        hd = t % L                                                                         # head slot (stride 1)
        s = occ[:, hd]
        if eta_l is not None:
            s = torch.binomial(s.to(rdt), torch.full_like(s, eta_l, dtype=rdt), generator=generator).long()
        a = blk["A"][t - blk["t0"]][s]                                                     # [S, d]
        p = a.real ** 2 + a.imag ** 2
        if force is None:
            d = torch.multinomial(p, 1, generator=generator)[:, 0]
        else:
            d = force[:, t]
            ok = d < p.shape[1]
            dd = d.clamp(max=p.shape[1] - 1)
            logp += torch.log(torch.where(ok, p[ar, dd], torch.zeros_like(p[:, 0])) / p.sum(1))
            d = torch.minimum(dd, s + c)
        new = s + c - d
        tot = tot + new - occ[:, hd]
        occ[:, hd] = new
        tot_t[t] = tot
        pats[:, t, 0] = d if eta_d is None else torch.binomial(d.to(rdt), torch.full_like(d, eta_d, dtype=rdt),
                                                               generator=generator).long()
    stats = {"max_amplitudes": 1, "mean_n": tot_t.double().mean(1).tolist(), "table_seconds": 0.0}
    return pats, logp, stats


def _local(M: torch.Tensor, x: torch.Tensor, dims, j: int) -> torch.Tensor:
    """M (d_j x d_j) on spin factor j of x [S, chi, D] (chi = prod dims)."""
    pre, dj = int(np.prod(dims[:j])), dims[j]
    post = x.shape[1] // (pre * dj)
    y = torch.einsum("ab,spbqd->spaqd", M, x.reshape(x.shape[0], pre, dj, post, x.shape[-1]))
    return y.reshape(x.shape)


def sector_sample(K: PeriodicNetworkKernel, shots: int, gates, chan: dict | None = None,
                  generator: torch.Generator | None = None, force=None):
    """Exact trajectory sampler of a register kernel on photon-number sectors (docs/HYBRID.md §6), run by
    `_sector_march`; one-loop Fock-input kernels go to the Fock-chain sampler `_loop_chain_sample` (§6d: their
    register is one basis state).

    K: a `PeriodicNetworkKernel` (LoopRegisterKernel with Fock and/or emitter input, HybridRegisterKernel). Each shot
    carries only the slot sector of its current photon number between periods and the full (rails + slots) sector
    during a period, so its state is chi * C(n + R - 1, R - 1) amplitudes instead of chi * C(N + R, R). Same law as
    `PeriodicNetworkKernel.sample` (inputs, emissions, the program's gates and loss jumps, rail exit, spin projection
    by rejection). gates: what K.blocks() takes; chan: channel name -> transmission (None = lossless).
    force [shots, T, n_rail] (lossless only): follow these outcomes instead of drawing them; then the returned
    log-probabilities are those of the patterns (-inf if impossible) — the exactness check against `probabilities`.
    Returns (patterns [shots', T, n_rail] long, logp [shots'] float64 (force only, else None), stats dict:
    max_amplitudes = the largest chi * D of a full sector touched, mean_n [T] = mean register photon number after exit,
    table_seconds = host time spent building sector tables in this call — a one-time cost per kernel, cached on K).
    Not for indefinite photon numbers (SuperpositionLoopKernel).
    """
    return _sector_march(K, shots, gates, chan=chan, generator=generator, force=force)


class _NoBasis(PeriodicNetworkKernel):
    """Builds the PeriodicNetworkKernel with a register basis of at most `_n_max` photons (the program, emitter and
    input bookkeeping without the dense basis) — the sector sampler builds its own bases.
    """

    _n_max = 1

    def __init__(self, *a, **kw):
        kw["n_max"] = self._n_max
        super().__init__(*a, **kw)


def light_loop_kernel(circuit: TimeLoopCircuit, input_occupation, device: str = "cpu",
                      dtype: torch.dtype = torch.complex128, emitter=None, bins_per_period: int | None = None):
    """A LoopRegisterKernel for `loop_sector_sample` only: same program, inputs, emitter and settings handling, but its
    dense register basis is capped at the largest per-period input (so building it costs nothing at any N). Its
    dense entry points (sample, amplitudes, marginals) are NOT exact and must not be used.
    """
    occ = np.asarray(input_occupation, dtype=np.int64)
    bpp = bins_per_period or (2 if emitter is not None else 1)
    per = occ.reshape(-1, bpp).sum(1).max() if occ.size % bpp == 0 else occ.max()
    cls = type("LightLoopKernel", (LoopRegisterKernel, _NoBasis), {"_n_max": max(1, int(per))})
    return cls(circuit, input_occupation, device=device, dtype=dtype, emitter=emitter, bins_per_period=bins_per_period)


def light_hybrid_kernel(emitters: Sequence, rails: int, loops, T: int | None = None, inputs=None, mixer: str = "after",
                        device: str = "cpu", dtype: torch.dtype = torch.complex128):
    """A HybridRegisterKernel (exact register, carry None) for `sample_sectors` only: same program, emitters, inputs
    and settings handling, but its dense register basis is capped at the largest per-tick input (k emissions plus
    the injected photons), so building it costs nothing at any T. Its dense entry points (sample, amplitudes,
    probabilities, marginals) are NOT exact and must not be used.
    """
    per = len(emitters) + (0 if inputs is None else int(np.asarray(inputs).reshape(-1, int(rails)).sum(1).max()))
    cls = type("LightHybridKernel", (HybridRegisterKernel, _NoBasis), {"_n_max": max(1, per)})
    return cls(emitters, rails, loops, T=T, inputs=inputs, mixer=mixer, device=device, dtype=dtype)


def loop_sector_sample(lk: LoopRegisterKernel, shots: int, theta, phi=None, generator: torch.Generator | None = None,
                       loss=None, return_stats: bool = False):
    """`LoopRegisterKernel.sample` on photon-number sectors (Fock and/or emitter input; no indist, no threshold):
    samples [shots', T_bins] with the same law; loss as there (dict of Kraus channels as jumps, scalar = uniform
    thinning of the output). With return_stats also the sector statistics of `sector_sample`. One loop with Fock
    input runs the Fock chain (§6d).
    """
    eta = lk._uniform(loss)
    chan = None if eta is not None else lk._loss_dict(loss, lk.circuit.n_loops)
    pats, _, stats = sector_sample(lk, shots, lk._gates(theta, phi), chan=chan, generator=generator)
    pats = pats.reshape(pats.shape[0], lk.T_bins)
    if eta is not None:
        from ..imperfections import thin_counts
        pats = thin_counts(pats, eta, generator=generator if generator is None or generator.device.type == "cpu" else None)
    return (pats, stats) if return_stats else pats
