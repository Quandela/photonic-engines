"""Input-state types shared by every engine.

Five kinds of input over m modes (n photons, except where the photon number is indefinite):

  FockState          one occupation vector |s_1 ... s_m>
  FockSuperposition  a coherent sum  sum_c w_c |s^(c)>  of Fock states with the same n
  ProductSuperposition  independent sources, each in a photon-number superposition of its own
                     mode: prod_i (sum_q a[i,q] |q>_{mode_i}).  The photon number is INDEFINITE;
                     d = 2 is superposition boson sampling (docs/SUPERPOSITION.md).
  GaussianInput      independent single-mode squeezed (optionally displaced) sources, one per input
                     mode: Gaussian boson sampling (docs/GAUSSIAN.md). Indefinite, unbounded photon number.
  EmitterMPS         the state emitted by ONE stationary emitter (spin / qudit of dimension chi)
                     that emits one photon per event into one of L possible modes, with a
                     chi x chi operator per (event, label):

                         |psi> = sum_x  <f| M^(n)_{x_n} ... M^(1)_{x_1} |s0>  |x_1 ... x_n>

                     x_k in {0..L-1} selects the mode receiving photon k (modes[k][x_k]).
                     This is a matrix product state of bond dimension chi; caterpillar graph
                     states (Nature Comm. 16, 2025, s41467-025-59693-3) are the chi = 2, L = 2
                     case with M_R = |up><up| U_k, M_L = |down><down| U_k.  The final spin is
                     either projected on a vector `final` (pure photonic state) or traced
                     (`final=None`: a rank-chi mixture, one pure component per final basis state).

Engines consume these directly; `EmitterMPS.expand()` unrolls to FockSuperposition(s) — the
2^n-component form — and exists only as a test oracle for small n.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np


def _modes_from_occupation(occ: Sequence[int]) -> list[int]:
    return [j for j, c in enumerate(occ) for _ in range(int(c))]


@dataclass(frozen=True)
class FockState:
    """One n-photon Fock state |s_1 ... s_m> over m modes.

    `occupation` is the per-mode photon count, a tuple of m non-negative ints (normalised to ints and
    checked non-empty in __post_init__). Immutable and hashable.
    """

    occupation: tuple[int, ...]

    def __post_init__(self):
        occ = tuple(int(x) for x in self.occupation)
        if any(x < 0 for x in occ) or not occ:
            raise ValueError("occupation must be a non-empty tuple of non-negative ints")
        object.__setattr__(self, "occupation", occ)

    @property
    def n(self) -> int:
        """Total photon number."""
        return sum(self.occupation)

    @property
    def m(self) -> int:
        """Number of modes."""
        return len(self.occupation)

    def modes(self) -> list[int]:
        """Input modes with multiplicity (the columns of U used by the engines)."""
        return _modes_from_occupation(self.occupation)

    def norm_factor(self) -> float:
        """prod_i s_i!"""
        return math.prod(math.factorial(x) for x in self.occupation)


class FockSuperposition:
    """A coherent sum  sum_c w_c |s^(c)>  of K Fock states with the same photon number n and mode
    count m, stored as arrays: `occupations` int64 [K, m], `amplitudes` complex128 [K] — usable
    for K up to ~10^6 (the engines' cost is K SLOS passes / K pure terms per sampling step, so
    that is the practical limit, not the storage). `components` (FockState tuple) is built on
    demand for small K. Construct from a list of occupations or FockStates, or from arrays.
    """

    __slots__ = ("_components", "amplitudes", "occupations")

    def __init__(self, components, amplitudes):
        occ = np.asarray([tuple(c.occupation) if isinstance(c, FockState) else tuple(c) for c in components]
                         if not isinstance(components, np.ndarray) else components, dtype=np.int64)
        amps = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
        if occ.ndim != 2 or occ.shape[0] == 0 or occ.shape[1] == 0:
            raise ValueError("components must be a non-empty [K, m] array of occupations")
        if occ.shape[0] != amps.shape[0]:
            raise ValueError("components and amplitudes must have equal length")
        if (occ < 0).any():
            raise ValueError("occupations must be non-negative")
        ns = occ.sum(1)
        if (ns != ns[0]).any():
            raise ValueError("all components must share the same photon number")
        if np.unique(occ, axis=0).shape[0] != occ.shape[0]:
            raise ValueError("components must be distinct occupations")
        occ.setflags(write=False); amps.setflags(write=False)
        object.__setattr__(self, "occupations", occ)
        object.__setattr__(self, "amplitudes", amps)
        object.__setattr__(self, "_components", None)

    def __setattr__(self, k, v):
        raise AttributeError("FockSuperposition is immutable")

    @property
    def components(self) -> tuple[FockState, ...]:
        """The K components as FockState objects, built on first access and cached."""
        if self._components is None:
            object.__setattr__(self, "_components", tuple(FockState(tuple(int(x) for x in row)) for row in self.occupations))
        return self._components

    @property
    def K(self) -> int:
        """Number of components."""
        return int(self.occupations.shape[0])

    @property
    def n(self) -> int:
        """Photon number, shared by every component."""
        return int(self.occupations[0].sum())

    @property
    def m(self) -> int:
        """Number of modes."""
        return int(self.occupations.shape[1])

    def norm(self) -> float:
        """l2 norm of the amplitude vector (1 for a normalised state)."""
        return float(np.sqrt((np.abs(self.amplitudes) ** 2).sum()))

    def normalized(self) -> FockSuperposition:
        """A copy with the amplitudes divided by `norm()`."""
        return FockSuperposition(self.occupations, self.amplitudes / self.norm())

    def __len__(self):
        return self.K

    def __repr__(self):
        head = ", ".join(f"{a:.3g}*|{','.join(map(str, o))}>" for o, a in zip(self.occupations[:3], self.amplitudes[:3]))
        return f"FockSuperposition(K={self.K}, n={self.n}, m={self.m}: {head}{', ...' if self.K > 3 else ''})"

    # ---- Perceval ------------------------------------------------------------------------
    @staticmethod
    def from_perceval(sv) -> list[FockSuperposition]:
        """Perceval StateVector -> one FockSuperposition per photon number (Perceval allows mixed n
        in a StateVector; the engines evaluate each photon-number block separately and the
        adapter recombines them). Amplitudes are taken as they are (not renormalized).
        """
        groups: dict = {}
        for bs, amp in sv:
            occ = tuple(int(x) for x in bs)
            groups.setdefault(sum(occ), ([], []))
            groups[sum(occ)][0].append(occ); groups[sum(occ)][1].append(complex(amp))
        return [FockSuperposition(o, a) for _, (o, a) in sorted(groups.items())]

    def to_perceval(self):
        """Build the equivalent Perceval StateVector (amplitudes as they are; requires perceval)."""
        from perceval.utils import BasicState, StateVector
        sv = StateVector()
        for occ, amp in zip(self.occupations.tolist(), self.amplitudes.tolist()):
            sv += complex(amp) * StateVector(BasicState(occ))
        return sv


@dataclass(frozen=True)
class ProductSuperposition:
    """Independent sources, each in a photon-number superposition of its OWN input mode:

        |psi> = prod_{i<sources} ( sum_{q<d} a[i, q] |q>_{mode_i} )   (x) |0> on the other modes

    d = 2 with a[i] = (cos alpha, sin alpha) for every source is Renema's superposition boson
    sampling (arXiv:1911.10112) - `ProductSuperposition.sbs`. Physically, a quantum dot driven
    with pulse area theta emits cos(theta/2)|0> + sin(theta/2)|1> coherently; this is the
    coherent counterpart of the incoherent per-source loss of `imperfections.py`, not the same state.

    The photon number is INDEFINITE. A passive unitary conserves it, so the detected total is the
    convolution of the per-source |a[i]|^2 (binomial at d = 2) whatever U is - `photon_number_probs`.
    Engines therefore return one block per photon number (`loss.LossyDistribution`) rather than a
    single fixed-n vector. Post-selected on k photons at d = 2 the input is the Dicke state D^n_k
    on the source modes: a coherent sum of C(sources, k) terms, so `blocks()` is a test oracle
    only - the engines consume this type directly (docs/SUPERPOSITION.md).

    amps   : complex array [sources, d]; amps[i, q] is the amplitude of q photons from source i
    modes  : modes[i] is the input mode fed by source i (distinct)
    m      : number of interferometer modes (> max mode index)
    """
    amps: np.ndarray
    modes: tuple[int, ...]
    m: int

    def __post_init__(self):
        amps = np.asarray(self.amps, dtype=np.complex128)
        if amps.ndim != 2 or amps.shape[0] == 0 or amps.shape[1] < 1:
            raise ValueError("amps must have shape [sources, d] with sources >= 1 and d >= 1")
        modes = tuple(int(x) for x in self.modes)
        if len(modes) != amps.shape[0]:
            raise ValueError("one mode per source")
        if len(set(modes)) != len(modes):
            raise ValueError("every source must use a distinct input mode")
        if min(modes) < 0 or max(modes) >= self.m:
            raise ValueError("mode index out of range")
        amps.setflags(write=False)
        object.__setattr__(self, "amps", amps)
        object.__setattr__(self, "modes", modes)

    @property
    def sources(self) -> int:
        """Number of independent sources, i.e. of occupied input modes."""
        return int(self.amps.shape[0])

    @property
    def d(self) -> int:
        """Local dimension: a source emits 0 .. d-1 photons."""
        return int(self.amps.shape[1])

    @property
    def n(self) -> int:
        """Largest photon number the state can carry, sources * (d - 1) - the top basis layer."""
        return self.sources * (self.d - 1)

    def norm(self) -> float:
        """Norm of the product state, prod_i ||a[i]||."""
        return float(np.prod(np.linalg.norm(self.amps, axis=1)))

    def normalized(self) -> ProductSuperposition:
        """The same state with every source normalized to unit norm."""
        rows = np.linalg.norm(self.amps, axis=1, keepdims=True)
        if (rows == 0).any():
            raise ValueError("a source with zero norm cannot be normalized")
        return ProductSuperposition(self.amps / rows, self.modes, self.m)

    def photon_number_probs(self) -> np.ndarray:
        """P(k photons), k = 0..n: the convolution of the per-source |a[i, q]|^2. Exact and
        independent of the interferometer (a passive unitary conserves photon number), so it is
        also the distribution of the detected total. Binomial(sources, sin^2 alpha) for `sbs`.
        """
        p = np.array([1.0])
        for row in self.amps:
            p = np.convolve(p, np.abs(row) ** 2)
        return p

    # ---- oracle ---------------------------------------------------------------------------
    def blocks(self, photons: int | None = None):
        """Unroll to explicit Fock superpositions: {k: FockSuperposition} over the photon numbers
        with non-zero weight, or the single block when `photons` is given (None if it is empty).

        d^sources components (C(sources, k) per block at d = 2) - exponential, tests only, like
        `EmitterMPS.expand()`. Component amplitudes are prod_i a[i, q_i] with no factorials: the
        bunching normalization of |q> is the kernels' input norm, not part of the coefficient.
        """
        acc: dict = {}
        for qs in product(range(self.d), repeat=self.sources):
            k = sum(qs)
            if photons is not None and k != photons:
                continue
            w = complex(np.prod([self.amps[i, q] for i, q in enumerate(qs)]))
            if w == 0:
                continue
            occ = [0] * self.m
            for i, q in enumerate(qs):
                occ[self.modes[i]] = q
            acc.setdefault(k, ([], []))
            acc[k][0].append(tuple(occ)); acc[k][1].append(w)
        out = {k: FockSuperposition(o, a) for k, (o, a) in sorted(acc.items())}
        return out if photons is None else out.get(photons)

    def to_perceval(self):
        """Perceval `StateVector` of the whole product state (every block at once). Exponential -
        it goes through `blocks()`. There is no `from_perceval`: factorizing a StateVector into
        independent sources is not this type's job, see `FockSuperposition.from_perceval`.
        """
        from perceval.utils import BasicState, StateVector
        sv = StateVector()
        for block in self.blocks().values():
            for occ, amp in zip(block.occupations.tolist(), block.amplitudes.tolist()):
                sv += complex(amp) * StateVector(BasicState(occ))
        return sv

    # ---- constructors ---------------------------------------------------------------------
    @staticmethod
    def sbs(sources: int, m: int, alpha: float = math.pi / 4,
            modes: Sequence[int] | None = None) -> ProductSuperposition:
        """Superposition boson sampling: `sources` copies of cos(alpha)|0> + sin(alpha)|1>, one per
        input mode (default: modes 0..sources-1).

        alpha = pi/4 (the default) is the hardest case at fixed source count: the mean photon
        number sin^2(alpha) * sources sits at sources/2, which is where the bond dimension
        min(k, sources - k) + 1 of the post-selected Dicke state saturates at its maximum and
        where the component count C(sources, k) is largest (docs/SUPERPOSITION.md §2).
        """
        if sources < 1:
            raise ValueError("need at least one source")
        row = np.array([math.cos(alpha), math.sin(alpha)], dtype=np.complex128)
        modes = tuple(range(sources)) if modes is None else tuple(int(x) for x in modes)
        return ProductSuperposition(np.tile(row, (sources, 1)), modes, m)


@dataclass(frozen=True)
class EmitterMPS:
    """Photons emitted by one emitter of dimension chi; see module docstring.

    ops    : complex array [n, L, chi, chi]; ops[k, l] is applied at event k when label l is emitted
    modes  : modes[k][l] is the input mode receiving photon k for label l (all distinct across k)
    s0     : initial emitter vector [chi]
    final  : final emitter vector [chi] (projection), or None to trace the emitter out
    m      : number of interferometer modes (>= max mode index + 1)
    """
    ops: np.ndarray
    modes: tuple[tuple[int, ...], ...]
    s0: np.ndarray
    m: int
    final: np.ndarray | None = None

    def __post_init__(self):
        ops = np.asarray(self.ops, dtype=np.complex128)
        if ops.ndim != 4 or ops.shape[2] != ops.shape[3]:
            raise ValueError("ops must have shape [n, L, chi, chi]")
        n, L, chi, _ = ops.shape
        modes = tuple(tuple(int(x) for x in mk) for mk in self.modes)
        if len(modes) != n or any(len(mk) != L for mk in modes):
            raise ValueError("modes must have shape [n][L]")
        flat = [x for mk in modes for x in mk]
        if len(set(flat)) != len(flat):
            raise ValueError("every (event, label) must use a distinct input mode")
        if min(flat) < 0 or max(flat) >= self.m:
            raise ValueError("mode index out of range")
        s0 = np.asarray(self.s0, dtype=np.complex128).reshape(-1)
        if s0.shape != (chi,):
            raise ValueError("s0 must have length chi")
        final = None if self.final is None else np.asarray(self.final, dtype=np.complex128).reshape(-1)
        if final is not None and final.shape != (chi,):
            raise ValueError("final must have length chi")
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "s0", s0)
        object.__setattr__(self, "final", final)

    @property
    def n(self) -> int:
        """Number of emission events, i.e. of photons."""
        return self.ops.shape[0]

    @property
    def num_labels(self) -> int:
        """Number L of labels (candidate modes) per emission event."""
        return self.ops.shape[1]

    @property
    def chi(self) -> int:
        """Emitter dimension, i.e. the MPS bond dimension."""
        return self.ops.shape[2]

    def is_pure(self) -> bool:
        """Whether the emitter is projected on `final` (pure photonic state) rather than traced out."""
        return self.final is not None

    def fold(self, folds: dict) -> tuple[EmitterMPS | None, np.ndarray]:
        """The shorter chain in which the slots of `folds` {slot: chi x chi operator} are
        replaced by a fixed operator instead of an emission: a folded slot's operator multiplies
        (on the right, chain order) the operators of the next unfolded slot, trailing folded
        slots go into a tail operator returned separately (to be applied after the chain, before
        `final`). Returns (reduced chain over the unfolded slots with `final=None`, tail [chi, chi]);
        the chain is None when every slot is folded. Two uses: a lost photon whose label x was
        drawn (`folds[j] = ops[j, x]`, docs/LOSS.md §3b) and a distinguishable photon whose output
        row r was drawn (`folds[j] = sum_l U[r, modes[j][l]] ops[j, l]`, docs/DISTINGUISHABILITY.md).
        """
        n, chi = self.n, self.chi
        acc = np.eye(chi, dtype=np.complex128)
        new_ops, kept = [], []
        for j in range(n):
            if j in folds:
                acc = np.asarray(folds[j], dtype=np.complex128) @ acc
            else:
                new_ops.append(np.stack([self.ops[j, l] @ acc for l in range(self.num_labels)]))
                kept.append(j)
                acc = np.eye(chi, dtype=np.complex128)
        if not kept:
            return None, acc
        red = EmitterMPS(ops=np.stack(new_ops), modes=tuple(self.modes[j] for j in kept), s0=self.s0, m=self.m, final=None)
        return red, acc

    # ---- oracle ---------------------------------------------------------------------------
    def expand(self) -> list[FockSuperposition]:
        """Unroll to the 2^n-component (L^n) form. Returns one FockSuperposition when `final`
        is given, else chi of them (one per final basis vector; the photonic state is their
        incoherent mixture, each with weight = its squared norm). Exponential — tests only.
        """
        finals = [self.final] if self.final is not None else list(np.eye(self.chi, dtype=np.complex128))
        out = []
        for f in finals:
            comps, amps = [], []
            for x in product(range(self.num_labels), repeat=self.n):
                v = self.s0
                for k, lab in enumerate(x):
                    v = self.ops[k, lab] @ v
                a = complex(np.vdot(f, v))
                if abs(a) < 1e-14:
                    continue
                occ = [0] * self.m
                for k, lab in enumerate(x):
                    occ[self.modes[k][lab]] += 1
                comps.append(FockState(tuple(occ)))
                amps.append(a)
            if comps:
                out.append(FockSuperposition(tuple(comps), tuple(amps)))
        return out

    # ---- constructors ---------------------------------------------------------------------
    @staticmethod
    def caterpillar(n: int, spin_gates: Sequence[np.ndarray], m: int | None = None,
                    modes: Sequence[tuple[int, int]] | None = None,
                    s0: Sequence[complex] = (1.0, 0.0),
                    final: Sequence[complex] | None = None) -> EmitterMPS:
        """Single-spin emitter with polarization-entangled emission: |up> emits R (label 0),
        |down> emits L (label 1). spin_gates[k] (2x2) is applied to the spin BEFORE emission k
        (k = 0..n-1), so spin_gates[0] is the spin preparation (Ry(pi/2) on |up> gives the
        |+> of the paper) and spin_gates[k >= 1] = U(theta_k, phi_k) the gate between emissions
        k-1 and k: theta = pi/2, phi = 0 gives the linear cluster state, theta = pi/2, phi = pi
        the GHZ state, mixed choices the redundantly-encoded / caterpillar family. Photon k
        goes to modes[k] = (R mode, L mode), default (2k, 2k+1) with m = 2n.
        """
        if len(spin_gates) != n:
            raise ValueError("one spin gate per emission")
        if modes is None:
            modes = [(2 * k, 2 * k + 1) for k in range(n)]
        if m is None:
            m = max(x for mk in modes for x in mk) + 1
        P_up = np.diag([1.0, 0.0]).astype(np.complex128)
        P_dn = np.diag([0.0, 1.0]).astype(np.complex128)
        ops = np.empty((n, 2, 2, 2), dtype=np.complex128)
        for k in range(n):
            G = np.asarray(spin_gates[k], dtype=np.complex128)
            ops[k, 0] = P_up @ G
            ops[k, 1] = P_dn @ G
        return EmitterMPS(ops=ops, modes=tuple(tuple(mk) for mk in modes), s0=np.asarray(s0),
                          m=m, final=None if final is None else np.asarray(final))


@dataclass(frozen=True)
class GaussianInput:
    """Independent single-mode squeezed sources, optionally displaced, one per input mode:

        |psi> = prod_{i<sources} D(disp[i]) S(r[i]) |0>_{mode_i}   (x) |0> on the other modes,

    S(r) = exp((r a^2 - r a^dag^2) / 2), so S(r)|0> = sech(r)^(1/2) exp(-tanh(r) a^dag^2 / 2)|0>
    squeezes the x = (a + a^dag)/2 quadrature (variance e^(-2r)/4, vacuum 1/4) and carries
    sinh^2 r photons on average, even numbers only. Gaussian boson sampling (docs/GAUSSIAN.md).
    Two-mode squeezing is two single-mode squeezers at the same r through a 50:50 beam splitter,
    i.e. an interferometer choice, not a separate type.

    The photon number is INDEFINITE and unbounded: P(2k) = C(2k, k) 4^-k tanh^2k r / cosh r per
    source (`photon_number_probs`). After a passive U the state stays Gaussian and pure,

        |psi_out> = N exp(1/2 a^dag^T B a^dag + zeta^T a^dag) |0>,   B = -U diag(tanh r) U^T
        (restricted to the source columns),  zeta = beta - B conj(beta),  beta = U disp,

    which `bargmann(U)` returns; every Fock amplitude is a loop hafnian of B and zeta.

    r      : float array [sources], r >= 0 (r = 0 is vacuum, allowed)
    modes  : modes[i] is the input mode fed by source i (distinct)
    m      : number of interferometer modes
    disp   : complex array [sources] or None, displacement applied after the squeezer
    """
    r: np.ndarray
    modes: tuple[int, ...]
    m: int
    disp: np.ndarray | None = None

    def __post_init__(self):
        r = np.asarray(self.r, dtype=np.float64).reshape(-1)
        if r.size == 0 or (r < 0).any() or not np.isfinite(r).all():
            raise ValueError("r must be a non-empty array of finite values >= 0")
        modes = tuple(int(x) for x in self.modes)
        if len(modes) != r.size:
            raise ValueError("one mode per source")
        if len(set(modes)) != len(modes):
            raise ValueError("every source must use a distinct input mode")
        if min(modes) < 0 or max(modes) >= self.m:
            raise ValueError("mode index out of range")
        disp = None
        if self.disp is not None:
            disp = np.asarray(self.disp, dtype=np.complex128).reshape(-1)
            if disp.size != r.size:
                raise ValueError("one displacement per source")
            disp.setflags(write=False)
        r.setflags(write=False)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "disp", disp)

    @property
    def sources(self) -> int:
        """Number of squeezed sources (the frontier axis n, docs/GAUSSIAN.md §2)."""
        return int(self.r.size)

    def mean_photons(self) -> float:
        """Mean total photon number, sum_i sinh^2 r_i + |disp_i|^2 (conserved by a passive U)."""
        tot = float(np.sum(np.sinh(self.r) ** 2))
        return tot + (0.0 if self.disp is None else float(np.sum(np.abs(self.disp) ** 2)))

    def photon_number_probs(self, cutoff: int) -> np.ndarray:
        """P(N) for N = 0..cutoff of the total photon number (undisplaced input only): the
        convolution of the per-source squeezed-vacuum laws, U-independent. The tail above
        `cutoff` is 1 - sum.
        """
        if self.disp is not None and np.any(self.disp != 0):
            raise NotImplementedError("photon-number law of a displaced input is not implemented")
        p = np.zeros(cutoff + 1); p[0] = 1.0
        for r in self.r:
            t2 = math.tanh(r) ** 2
            q = np.zeros(cutoff + 1)
            c = 1.0 / math.cosh(r)
            for k in range(cutoff // 2 + 1):
                q[2 * k] = c
                c *= t2 * (2 * k + 1) / (2 * k + 2)                   # C(2k,k)/4^k recurrence
            p = np.convolve(p, q)[: cutoff + 1]
        return p

    def truncated(self, d: int) -> ProductSuperposition:
        """The input truncated at d - 1 photons per source, NOT renormalised, as a
        `ProductSuperposition` (amplitude of |q> = sech^(1/2) r (-tanh r)^(q/2) sqrt(q!) / (2^(q/2) (q/2)!)
        on even q). Every lossless output pattern with at most d - 1 photons then has its exact
        probability under `slos.ProductSLOSKernel`: no source configuration feeding it is truncated.
        The independent oracle of the Gaussian engines. Undisplaced input only.
        """
        if self.disp is not None and np.any(self.disp != 0):
            raise NotImplementedError("truncated() is for undisplaced squeezed vacuum")
        amps = np.zeros((self.sources, d), dtype=np.complex128)
        for i, r in enumerate(self.r):
            t = -math.tanh(r); pre = 1.0 / math.sqrt(math.cosh(r))
            for q in range(0, d, 2):
                k = q // 2
                amps[i, q] = pre * t ** k * math.sqrt(math.factorial(q)) / (2 ** k * math.factorial(k))
        return ProductSuperposition(amps, self.modes, self.m)

    def bargmann(self, U, r=None, disp=None) -> tuple[np.ndarray, np.ndarray, complex]:
        """(B [m, m], zeta [m], log N) of the pure output state exp(1/2 a^T B a + zeta^T a)|0> x N
        after the interferometer U [m, m] (numpy or torch, complex). `r` / `disp` override the
        state's own (the lossy sampler passes the pure part and a drawn displacement).
        log N = -1/2 sum log cosh r - 1/2 |beta|^2 + 1/2 conj(beta)^T B conj(beta).
        """
        U = np.asarray(U.detach().cpu().numpy() if hasattr(U, "detach") else U, dtype=np.complex128)
        r = self.r if r is None else np.asarray(r, dtype=np.float64)
        d = self.disp if disp is None else np.asarray(disp, dtype=np.complex128)
        Uc = U[:, list(self.modes)]
        B = -(Uc * np.tanh(r)) @ Uc.T
        beta = np.zeros(self.m, dtype=np.complex128) if d is None else Uc @ d
        zeta = beta - B @ beta.conj()
        logN = -0.5 * float(np.sum(np.log(np.cosh(r)))) - 0.5 * float(np.vdot(beta, beta).real) \
            + 0.5 * complex(beta.conj() @ B @ beta.conj())
        return B, zeta, logN

    @staticmethod
    def smsv(sources: int, m: int, r: float, modes: Sequence[int] | None = None) -> GaussianInput:
        """`sources` identical squeezed vacua at squeezing r on modes 0..sources-1 (default)."""
        if sources < 1:
            raise ValueError("need at least one source")
        modes = tuple(range(sources)) if modes is None else tuple(int(x) for x in modes)
        return GaussianInput(np.full(sources, float(r)), modes, m)

    @staticmethod
    def r_for_mean(mean_photons_per_source: float) -> float:
        """Squeezing r with sinh^2 r = the given mean photon number per source."""
        return math.asinh(math.sqrt(mean_photons_per_source))


def spin_gate(theta: float, phi: float) -> np.ndarray:
    """U(theta, phi) = Ry(theta/2) Rz(phi) Ry(theta/2): Larmor precession (Ry) around an optical
    spin-rotation pulse (Rz), the unitary between two emissions in the caterpillar protocol.
    """
    def ry(t):
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    def rz(p):
        return np.diag([np.exp(-1j * p / 2), np.exp(1j * p / 2)]).astype(np.complex128)

    return ry(theta / 2) @ rz(phi) @ ry(theta / 2)
