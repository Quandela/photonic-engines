"""Non-linear boson sampling: interferometers interleaved with single-mode photon-number phases.

The model of Spagnolo, Brod, Galvao & Sciarrino (arXiv:2110.13788), generalised; notation of docs/NONLINEAR.md §1:

    n, m, t        photons, modes, input modes (one photon per input mode: the source axis)
    U_0 .. U_L     the interferometers, [m, m], rows = outputs, columns = inputs
    site           (layer, x, f): exp(-i f(N_x)) on mode x, between U_{layer-1} and U_layer (layer = 1..L);
                   f a phase per photon number 0..n. Kerr: f(j) = phi j^2 (`kerr`). Several sites of one layer
                   sit on distinct modes.
    g(j)           exp(-i f(j)), the response of a site
    branch         one frequency theta per site: the LINEAR interferometer U_b = U_L D_L ... D_1 U_0, D_l the
                   phases exp(i theta) on the sites of layer l

The key identity (§2): N_x <= n, so exp(-i f(N_x)) = sum_l c_l exp(i theta_l N_x) exactly for a finite set of
frequencies (`branches`), and the circuit is a coherent sum of linear interferometers,

    <s| C |t> = sum_b c_b Per(U_b[s, t]) / sqrt(s! t!),        c_b = product over the sites of their c_l.

  strong  `NonlinearCircuit.amplitudes / probs`: route "branch" (one SLOS pass per branch, coherent sum of the
          amplitude vectors) or "gate" (SLOS for U_0, then the site phases and U_1.. applied to the general
          n-photon state by nearest-neighbour two-mode gates, `TwoModeNetwork`); "auto" takes the cheaper.
  weak    `NonlinearCircuit.sample`: rejection from the INCOHERENT branch mixture (§5). The Clifford & Clifford
          random-order chain does not hold for this circuit (§4: the k-photon marginal has coherences between
          different input subsets); the mixture sum_b (|c_b| / |c|_1) |Per(U_b[s, t])|^2 / s! is a mixture of
          linear boson samplers, drawn with `clifford.fock_sample`, and Cauchy-Schwarz bounds the target by
          |c|_1^2 times it: exact, acceptance 1 / |c|_1^2, all branch permanents of the proposal per draw.
          method="strong" draws from `probs()` instead (small n).

Restrictions: photon-number-resolving detectors; one photon per input mode for the sampler.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

GATE_WEIGHT = 4.0                     # a TwoModeNetwork entry / a SLOS edge, measured on one CPU core (NONLINEAR.md §3)
BRANCH_DROP = 1e-14                   # branch coefficients below this (absolute) are numerical zeros of the DFT


def kerr(phi: float, n: int) -> np.ndarray:
    """The Kerr phase vector f(j) = phi j^2, j = 0..n: float64 [n+1]."""
    return float(phi) * np.arange(n + 1, dtype=np.float64) ** 2


def site_phases(f, n: int) -> np.ndarray:
    """A site's phase vector on 0..n: float64 [n+1]. A number is the Kerr strength phi (f(j) = phi j^2); a sequence
    is f itself and must cover 0..n (longer ones are cut).
    """
    if np.isscalar(f):
        return kerr(float(f), n)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    if f.shape[0] < n + 1:
        raise ValueError(f"site phase vector of length {f.shape[0]}: needs one phase per photon number 0..{n}")
    return f[:n + 1].copy()


def branches(f, n: int, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies theta [P] (float64) and coefficients c [P] (complex128) with
    sum_l c_l exp(i theta_l j) = exp(-i f(j)) for every j = 0..n: the site as a coherent sum of P phase shifters.

    g(j) = exp(-i f(j)) is written e^{i a j} h(j) with h periodic of the smallest period q on 0..n (a twisted
    period: g(j+q) / g(j) constant; q = n+1 always works), and h is expanded on its q-point DFT:
    theta_l = a + 2 pi l / q. Kerr at phi = 2 pi p / q' has q <= q' whatever n (phi = pi: one branch, a phase
    shifter; phi = pi/2: two). Coefficients below BRANCH_DROP are dropped. |c|_1^2 is the sampler's
    rejection factor (docs/NONLINEAR.md §5). The reconstruction is checked to `tol` on 0..n.
    """
    fv = site_phases(f, n)
    g = np.exp(-1j * fv)
    q, a = n + 1, 0.0
    for p in range(1, n + 1):
        r = g[p:] / g[:-p]
        if np.all(np.abs(r - r[0]) < tol):
            q, a = p, float(np.angle(r[0])) / p
            break
    h = g[:q] * np.exp(-1j * a * np.arange(q))
    c = np.fft.fft(h) / q
    theta = a + 2 * np.pi * np.arange(q) / q
    keep = np.abs(c) > BRANCH_DROP
    theta, c = theta[keep], c[keep]
    j = np.arange(n + 1)
    err = np.abs(np.exp(1j * np.outer(j, theta)) @ c - g).max()
    if err > tol:
        raise FloatingPointError(f"branch expansion of the site: reconstruction error {err:.2e}")
    return theta, c


class TwoModeNetwork:
    """A unitary applied to a general n-photon state on the SLOS final basis, as nearest-neighbour two-mode gates.

    occ   int64 [S, m] the basis (every n-photon occupation of m modes; `FockLayers.final_occupations()` order)
    The decomposition is the Reck-type elimination T_N .. T_1 U = D (each T on modes (a, a+1)), so
    U = T_1^dag .. T_N^dag D and the state is multiplied by D, then T_N^dag .. T_1^dag. A gate on modes (a, a+1)
    mixes the states that differ only there: per pair, the (source, target) index pairs are built once
    (O(S (1 + 2n/m)) entries) and cached; a gate is one gather + index_add over them, O(S (1 + 2n/m)).
    m (m - 1) / 2 gates per unitary. complex128 on occ's device. Needs n ceil(log2 m) <= 63 (the packed key).
    """

    def __init__(self, occ: torch.Tensor):
        self.occ = occ.to(torch.int64)
        self.S, self.m = self.occ.shape
        self.n = int(self.occ[0].sum()) if self.S else 0
        self.dev = self.occ.device
        self.bits = max(1, math.ceil(math.log2(max(self.m, 2))))
        if self.n * self.bits > 63:
            raise ValueError(f"TwoModeNetwork: n = {self.n}, m = {self.m} does not pack into 63 bits")
        self.keys = self._key(self.occ)
        self.order = torch.argsort(self.keys)
        self._pairs: dict = {}

    def _key(self, occ: torch.Tensor) -> torch.Tensor:
        """Packed sorted photon positions [R] int64 (lexicographic order of the positions = numeric order)."""
        cs = torch.cumsum(occ, dim=1)                                                # [R, m]
        key = torch.zeros(occ.shape[0], dtype=torch.int64, device=self.dev)
        for k in range(self.n):
            key = (key << self.bits) | (cs <= k).sum(1)                              # position of photon k
        return key

    def _index(self, occ: torch.Tensor) -> torch.Tensor:
        key = self._key(occ)
        pos = torch.searchsorted(self.keys[self.order], key)
        return self.order[pos.clamp_max(self.S - 1)]

    def _pair(self, a: int):
        """Cached (src, tgt, N, j, k) int64 [E] for the gate on modes (a, a+1)."""
        if a not in self._pairs:
            N = self.occ[:, a] + self.occ[:, a + 1]                                  # [S]
            src = torch.repeat_interleave(torch.arange(self.S, device=self.dev), N + 1)
            start = torch.cumsum(N + 1, 0) - (N + 1)
            k = torch.arange(src.shape[0], device=self.dev) - start[src]             # 0..N_src
            tgt_occ = self.occ[src].clone()
            tgt_occ[:, a] = k; tgt_occ[:, a + 1] = N[src] - k
            tgt = self._index(tgt_occ)
            self._pairs[a] = (src, tgt, N[src], self.occ[src, a], k)
        return self._pairs[a]

    def _gate_table(self, u: torch.Tensor) -> torch.Tensor:
        """[n+1, n+1, n+1] complex128: T[N, k, j] = <k, N-k| u |j, N-j> on modes (a, a+1), u the 2 x 2 mode matrix
        (a^dag -> u00 a^dag + u10 b^dag, b^dag -> u01 a^dag + u11 b^dag): the coefficient of x^k in
        (u00 x + u10)^j (u01 x + u11)^(N-j), times sqrt(k! (N-k)! / (j! (N-j)!)).
        """
        n = self.n
        u = u.detach().to(torch.complex128).cpu().numpy()
        T = np.zeros((n + 1, n + 1, n + 1), dtype=np.complex128)
        lf = np.array([math.lgamma(i + 1) for i in range(n + 1)])
        binom = [np.array([math.comb(j, p) for p in range(j + 1)], dtype=np.float64) for j in range(n + 1)]
        pa = [binom[j] * u[0, 0] ** np.arange(j + 1) * u[1, 0] ** (j - np.arange(j + 1)) for j in range(n + 1)]
        pb = [binom[j] * u[0, 1] ** np.arange(j + 1) * u[1, 1] ** (j - np.arange(j + 1)) for j in range(n + 1)]
        for N in range(n + 1):
            k = np.arange(N + 1)
            for j in range(N + 1):
                T[N, :N + 1, j] = np.convolve(pa[j], pb[N - j]) * np.exp(0.5 * (lf[k] + lf[N - k] - lf[j] - lf[N - j]))
        return torch.as_tensor(T, device=self.dev)

    @staticmethod
    def decompose(U: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """(d [m], gates [(a, G [2, 2])]) with U = G_1 .. G_N diag(d), G_k on modes (a_k, a_k + 1) (host, complex128)."""
        V = U.detach().to(torch.complex128).cpu().numpy().copy()
        m = V.shape[0]
        gates = []
        for j in range(m - 1):
            for i in range(m - 1, j, -1):
                x, y = V[i - 1, j], V[i, j]
                if abs(y) < 1e-300:
                    continue
                r = math.hypot(abs(x), abs(y))
                T = np.array([[x.conjugate() / r, y.conjugate() / r], [-y / r, x / r]])
                V[i - 1:i + 1] = T @ V[i - 1:i + 1]
                gates.append((i - 1, torch.as_tensor(T.conj().T)))
        return torch.as_tensor(np.diagonal(V).copy()), gates

    def apply(self, U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """U applied to the state psi [S] (or [B, S], every row the same U) over the basis: same shape."""
        d, gates = self.decompose(U)
        d = d.to(self.dev)
        psi = psi * torch.prod(d[None, :] ** self.occ, dim=1)                        # diag(d): prod_i d_i^{occ_i}
        for a, G in reversed(gates):
            src, tgt, N, j, k = self._pair(a)
            T = self._gate_table(G)
            val = T[N, k, j] * psi[..., src]
            out = torch.zeros_like(psi)
            out.index_add_(psi.dim() - 1, tgt, val)
            psi = out
        return psi


class NonlinearCircuit:
    """U_0, then L layers of (single-mode photon-number phases, interferometer U_l), on a Fock input.

    unitaries  [U_0, .., U_L], each [m, m] (rows = outputs, columns = inputs)
    in_modes   the n input modes t (one photon each)
    sites      [(layer, mode, f)]: exp(-i f(N_mode)) between U_{layer-1} and U_layer, layer = 1..L, f a number
               (Kerr phi, f(j) = phi j^2) or a phase per photon number 0..n. The sites of one layer are on
               distinct modes.
    Everything is complex128 on U_0's device. `stats` holds the counters of the last `sample`.
    """

    def __init__(self, unitaries: Sequence[torch.Tensor], in_modes: Sequence[int], sites: Sequence):
        if len(unitaries) < 1:
            raise ValueError("at least one interferometer")
        self.dev = unitaries[0].device
        self.unitaries = [torch.as_tensor(U).to(device=self.dev, dtype=torch.complex128) for U in unitaries]
        self.m = int(self.unitaries[0].shape[0])
        if any(tuple(U.shape) != (self.m, self.m) for U in self.unitaries):
            raise ValueError("every interferometer is m x m")
        self.in_modes = [int(j) for j in in_modes]
        self.n = len(self.in_modes)
        if any(not 0 <= j < self.m for j in self.in_modes):
            raise ValueError("input modes must be modes of the circuit")
        self.L = len(self.unitaries) - 1
        self.sites = []
        for layer, mode, f in sites:
            if not 1 <= int(layer) <= self.L:
                raise ValueError(f"site layer {layer}: between two interferometers, 1..{self.L}")
            if not 0 <= int(mode) < self.m:
                raise ValueError(f"site mode {mode}: not a mode of the circuit")
            self.sites.append((int(layer), int(mode), site_phases(f, self.n)))
        self.sites.sort(key=lambda s: s[0])
        for layer in range(1, self.L + 1):
            modes = [x for lay, x, _ in self.sites if lay == layer]
            if len(set(modes)) != len(modes):
                raise ValueError(f"two sites on one mode in layer {layer}: merge their phase vectors")
        self._kernel = None
        self._network = None
        self._tables: dict = {}
        self.stats: dict = {}

    # ---- the branch expansion ---------------------------------------------------------------

    def branch_table(self, photons: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(U_b [B, m, m], c_b [B]) complex128: the linear interferometers of the branch expansion and their
        weights, B = product over the sites of their branch counts. Cached per photon number.

        `photons` (default n) is the number of photons that can reach the sites: with loss on the input side a
        stream carries k <= n of them, and the k-photon expansion (exact on 0..k) is never larger than the
        n-photon one - for a generic response it is the reason the cost follows the same eta law as the other
        engines (docs/NONLINEAR.md §6).
        """
        photons = self.n if photons is None else int(photons)
        if photons not in self._tables:
            per_site = [branches(f, photons) for _, _, f in self.sites]
            T = self.unitaries[0][None]                                              # [B, m, m]
            c = torch.ones(1, dtype=torch.complex128, device=self.dev)
            for layer in range(1, self.L + 1):
                for i in [i for i, site in enumerate(self.sites) if site[0] == layer]:
                    th, cs = per_site[i]
                    x = self.sites[i][1]
                    ph = torch.as_tensor(np.exp(1j * th), device=self.dev)            # [P]
                    T = T[:, None].repeat(1, len(th), 1, 1)                          # [B, P, m, m]
                    T[:, :, x, :] = T[:, :, x, :] * ph[None, :, None]
                    T = T.reshape(-1, self.m, self.m)
                    c = (c[:, None] * torch.as_tensor(cs, device=self.dev)[None, :]).reshape(-1)
                T = self.unitaries[layer] @ T
            self._tables[photons] = (T, c)
        return self._tables[photons]

    @property
    def n_branches(self) -> int:
        """B, the number of linear branches."""
        return int(self.branch_table()[1].shape[0])

    def rejection_factor(self, photons: int | None = None) -> float:
        """|c|_1^2 = (sum_b |c_b|)^2: the expected proposals per accepted sample of `sample` (1 for a linear
        circuit), for a stream carrying `photons` photons (default n).
        """
        return float(self.branch_table(photons)[1].abs().sum()) ** 2

    # ---- strong -------------------------------------------------------------------------------

    def _slos(self):
        if self._kernel is None:
            from .slos import FockSLOSKernel
            occ = [0] * self.m
            for j in self.in_modes:
                occ[j] += 1
            self._kernel = FockSLOSKernel(self.n, self.m, occ, device=str(self.dev), dtype=torch.complex128)
            self._occ = torch.as_tensor(self._kernel.layers.final_occupations(), device=self.dev)   # [S, m]
            self._amp_norm = self._kernel.output_prob_norm.to(torch.float64).sqrt()                 # sqrt(s!) [S]
        return self._kernel

    def occupations(self) -> torch.Tensor:
        """The output basis of `amplitudes` / `probs`: int64 [S, m] (the SLOS final basis)."""
        self._slos()
        return self._occ

    def route_costs(self) -> dict:
        """Cost model of the two strong routes (docs/NONLINEAR.md §3): {"branch": B E, "gate": E + 4 G}, E = SLOS
        edges (sum over the layers of states x photons), G = L m (m-1)/2 gates x S (1 + 2n/m) entries; 4 is the
        measured cost of a gate entry over a SLOS edge (one CPU core, complex128; not measured on a GPU).
        """
        k = self._slos()
        E = sum(k.layers.size(i) * i for i in range(1, self.n + 1))
        S = k.layers.size(self.n)
        G = self.L * self.m * (self.m - 1) / 2 * S * (1 + 2 * self.n / self.m)
        return {"branch": self.n_branches * E, "gate": E + GATE_WEIGHT * G}

    def amplitudes(self, route: str = "auto", chunk: int | None = None) -> torch.Tensor:
        """Output amplitudes <s| U_L N_L .. N_1 U_0 |t> over `occupations()`: complex128 [S], normalised (sum |.|^2 = 1).

        route  "branch"  sum_b c_b SLOS(U_b): B passes of the linear kernel, `chunk` branches at a time (memory
                         chunk x S complex128; default: as many as fit 2^27 entries)
               "gate"    SLOS(U_0), then per layer the diagonal site phases and U_l by `TwoModeNetwork`
               "auto"    the smaller of `route_costs()`
        """
        kernel = self._slos()
        if route == "auto":
            cost = self.route_costs()
            route = "branch" if cost["branch"] <= cost["gate"] else "gate"
        if route == "branch":
            Ub, c = self.branch_table()
            S = kernel.output_size()
            chunk = chunk or max(1, (1 << 27) // max(S, 1))
            acc = torch.zeros(S, dtype=torch.complex128, device=self.dev)
            for b0 in range(0, Ub.shape[0], chunk):
                acc += (c[b0:b0 + chunk, None] * kernel.amplitudes(Ub[b0:b0 + chunk])).sum(0)
            return acc * self._amp_norm
        if route == "gate":
            psi = kernel.amplitudes(self.unitaries[0]) * self._amp_norm
            if self._network is None and self.L:
                self._network = TwoModeNetwork(self._occ)
            for layer in range(1, self.L + 1):
                for lay, x, f in self.sites:
                    if lay == layer:
                        g = torch.as_tensor(np.exp(-1j * f), device=self.dev)
                        psi = psi * g[self._occ[:, x]]
                psi = self._network.apply(self.unitaries[layer], psi)
            return psi
        raise ValueError(f'unknown route {route!r}: "branch", "gate" or "auto"')

    def probs(self, route: str = "auto") -> torch.Tensor:
        """Output distribution over `occupations()`: float64 [S], sums to 1."""
        a = self.amplitudes(route)
        return a.real ** 2 + a.imag ** 2

    # ---- weak ---------------------------------------------------------------------------------

    def _base_sampler(self, backend: str) -> Callable:
        def base(U, in_modes, shots, generator):
            from .clifford import fock_sample
            return fock_sample(U, in_modes, shots, generator=generator, backend=backend)
        return base

    def _rows(self, occ: torch.Tensor, p: int) -> torch.Tensor:
        """Occupations [S, m] of p photons -> their sorted output rows [S, p] int64."""
        S = occ.shape[0]
        flat = torch.arange(self.m, device=occ.device).expand(S, -1).reshape(-1)
        return torch.repeat_interleave(flat, occ.reshape(-1)).reshape(S, p)

    def _permanents(self, rows: torch.Tensor, cols, photons: int, backend: str,
                    max_entries: int = 1 << 24) -> torch.Tensor:
        """Per(U_b[rows, cols]) for every stream and every branch of the `photons` table: complex128 [S, B].

        rows [S, p] int64 output rows (repeats allowed); `cols` the p input columns, either shared by every
        stream (a sequence) or one column set per stream (int64 [S, p] - the species of a lossy / partially
        distinguishable batch, whose columns differ from stream to stream). backend "torch": batched Glynn
        with a Gray code, 2^(p-1) steps of O(S B p) (CPU / tests); "triton" / "cuda": the Clifford cofactor
        kernels (`clifford.cofactors_triton.laplace_cofactors_rows`, repeated rows exploited) and one Laplace
        step on the last row. p = 1 is the amplitude U_b[r, col] itself. The (stream, branch) axis is cut into
        chunks of at most `max_entries` gathered complex128 entries: S B m p for the kernels is 13 GB at
        S = 256, B = 256, m = 42 (the CUDA OOM of job 3 at 8 sites, run 20260922_211901).
        """
        Ub, _ = self.branch_table(photons)
        rows = rows.to(self.dev)
        per_stream = isinstance(cols, torch.Tensor) and cols.dim() == 2
        S, B = rows.shape[0], Ub.shape[0]
        p = cols.shape[1] if per_stream else len(cols)
        if p == 0 or S == 0:
            return torch.ones(S, B, dtype=torch.complex128, device=self.dev)
        step = max(1, int(max_entries // max(B * max(self.m if backend != "torch" else p, p) * p, 1)))
        if step < S:
            return torch.cat([self._permanents(rows[i:i + step], cols[i:i + step] if per_stream else cols,
                                               photons, backend, max_entries) for i in range(0, S, step)])
        if per_stream:                                                               # [S, B, m, p]: one gather
            cols = cols.to(self.dev)
            A = Ub[None].expand(S, -1, -1, -1).gather(3, cols[:, None, None, :].expand(S, B, self.m, p))
        else:
            A = Ub[:, :, list(cols)][None].expand(S, -1, -1, -1)                      # [S, B, m, p] (a view)
        if p == 1:
            return A[:, :, :, 0].gather(2, rows[:, None, :].expand(S, B, 1))[:, :, 0]
        if backend == "torch":
            M = A.gather(2, rows[:, None, :, None].expand(S, B, p, p))                # [S, B, p, p]
            return _glynn(M)
        from .clifford.cofactors_triton import laplace_cofactors_rows
        Ab = A.reshape(S * B, self.m, p).contiguous()                                 # stream s*B + b
        rb = rows[:, None, :].expand(-1, B, -1).reshape(S * B, p)
        P = laplace_cofactors_rows(Ab, rb[:, :p - 1].contiguous(), backend=backend)    # [S*B, p]
        last = Ab.gather(1, rb[:, p - 1:, None].expand(-1, 1, p))[:, 0]               # [S*B, p]
        return (last * P).sum(1).reshape(S, B)

    def branch_amplitudes(self, occ: torch.Tensor, backend: str = "torch") -> torch.Tensor:
        """Per(U_b[s, t]) for every row s of occ [S, m] (n photons each) and every branch b: complex128 [S, B]."""
        return self._permanents(self._rows(occ.to(self.dev), self.n), self.in_modes, self.n, backend)

    def _reject_batch(self, groups, need: torch.Tensor, base: Callable, generator, backend: str,
                      amp_backend: str, max_batch: int) -> list:
        """One proposal pass over EVERY partition at once: `groups` [(cols_list, photons, ids)], `need[g]` the
        samples group g is still short of. Returns the accepted occupations per group ([n_g, m] each).

        Nothing in the pass is per partition. The streams of every group are laid out contiguously, and the
        work is grouped by (species size k, photon count, species slot): one `fock_sample` call per class, the
        distinct (columns, branch) pairs being its circuits (the column set moved into the first k columns of
        the matrix, the way `fock_sample` treats the surviving subsets of a lossy stream), and ONE batched
        permanent call per class over the per-stream column sets. Doing either per partition prices launch
        latency instead of the algorithm: with loss almost every stream draws its own partition (runs
        20260922_211901 and 20260923_080913, docs/NONLINEAR.md §6c).

        A group is proposed `need * |c|_1^2 * 1.15` times and no more: a fixed slack per group is paid per
        STREAM under loss - the +8 of the first version drew 4.6x the proposals it needed at eta = 0.9. A group
        still short simply reappears in the next pass.
        """
        plan = [(g, max(1, math.ceil(int(need[g]) * self.rejection_factor(gr[1]) * 1.15)))
                for g, gr in enumerate(groups) if int(need[g]) > 0]
        tot = sum(d for _, d in plan)
        if tot > max_batch:                                                          # keep one pass in memory
            plan = [(g, max(1, d * max_batch // tot)) for g, d in plan]
        offs, off = [], 0
        for _, d in plan:
            offs.append(off); off += d
        S = off
        self.stats["proposals"] += S
        br = torch.cat([torch.multinomial((lambda c1: (c1 / c1.sum()).to(torch.float64))(
            self.branch_table(groups[g][1])[1].abs()), d, replacement=True, generator=generator)
            for g, d in plan])                                                       # [S] branch per stream
        photons = torch.tensor([groups[g][1] for g, _ in plan], device=self.dev).repeat_interleave(
            torch.tensor([d for _, d in plan], device=self.dev))                      # [S]
        Bmax = max(self.branch_table(groups[g][1])[1].shape[0] for g, _ in plan)
        occ = torch.zeros(S, self.m, dtype=torch.int64, device=self.dev)
        amp = torch.ones(S, Bmax, dtype=torch.complex128, device=self.dev)
        classes: dict = {}                                                           # (k, photons, slot) -> [(off, d, cols)]
        for (g, d), o in zip(plan, offs, strict=True):
            cols_list, pho, _ = groups[g]
            slots: dict = {}
            for cols in cols_list:
                k = len(cols)
                slots[k] = slots.get(k, 0) + 1                                       # a stream may hold two species of one size
                classes.setdefault((k, pho, slots[k]), []).append((o, d, cols))
        for (k, pho, _slot), lst in classes.items():
            Ub, c = self.branch_table(pho)
            B = c.shape[0]
            ids = torch.cat([torch.arange(o, o + d, device=self.dev) for o, d, _ in lst])            # [R], distinct
            cm = torch.cat([torch.tensor(cols, device=self.dev).expand(d, k) for _, d, cols in lst])  # [R, k]
            uniq, inv = torch.unique(cm, dim=0, return_inverse=True)                  # [C, k], [R]
            circ = inv * B + br[ids]                                                 # circuit = (column set, branch)
            V = torch.zeros(uniq.shape[0] * B, self.m, self.m, dtype=torch.complex128, device=self.dev)
            V[:, :, :k] = Ub[:, :, uniq].permute(2, 0, 1, 3).reshape(-1, self.m, k)   # [C*B, m, k]
            order = torch.argsort(circ, stable=True)                                 # fock_sample groups by circuit
            counts = torch.bincount(circ, minlength=V.shape[0]).cpu()
            piece = torch.zeros(ids.shape[0], self.m, dtype=torch.int64, device=self.dev)
            piece[order] = base(V, list(range(k)), counts, generator).to(self.dev)
            occ.index_add_(0, ids, piece)
            amp[ids, :B] *= self._permanents(self._rows(piece, k), cm, pho, amp_backend)
        keep = torch.zeros(S, dtype=torch.bool, device=self.dev)
        for pho in {p for p, _ in ((groups[g][1], g) for g, _ in plan)}:
            sel = torch.nonzero(photons == pho)[:, 0]
            c = self.branch_table(pho)[1]
            c1 = c.abs()
            a = amp[sel][:, :c.shape[0]]
            num = (a @ c).abs() ** 2
            den = c1.sum() * ((a.real ** 2 + a.imag ** 2) @ c1.to(torch.float64))
            u = torch.rand(sel.shape[0], generator=generator, device=self.dev, dtype=torch.float64)
            keep[sel] = u * den < num
        kept_by_group = [occ[:0]] * len(groups)
        for (g, d), o in zip(plan, offs, strict=True):
            k = occ[o:o + d][keep[o:o + d]]
            kept_by_group[g] = k[torch.randperm(k.shape[0], generator=generator, device=self.dev)]
            self.stats["accepted"] += k.shape[0]
        return kept_by_group

    def sample(self, shots: int, generator: torch.Generator | None = None, method: str = "rejection",
               backend: str = "triton", amp_backend: str | None = None, base_sampler: Callable | None = None,
               max_batch: int = 1 << 16, max_draws: int = 1 << 26, loss=None, eta_pre=None, eta_post=None,
               indist=None) -> torch.Tensor:
        """Exact samples of the output occupation: int64 [shots, m] (fewer photons per row with loss).

        method  "rejection"  proposal: a branch b with probability |c_b| / |c|_1, then a boson sample of the
                             linear U_b (`clifford.fock_sample(backend=)`, or `base_sampler(U [B, m, m],
                             in_modes, shots int64 [B], generator) -> [sum shots, m]` rows grouped by circuit);
                             accepted with probability |sum_b c_b a_b|^2 / (|c|_1 sum_b |c_b| |a_b|^2), a_b the
                             product of the species permanents Per(U_b[s_mu, t_mu]) over ALL branches
                             (`amp_backend`; default "torch" on CPU, `backend` on CUDA). Acceptance exactly
                             1 / |c|_1^2 (`rejection_factor`); docs/NONLINEAR.md §5.
                "strong"     from `probs()` (small n, lossless and indistinguishable only).
        loss    uniform transmission eta BEFORE the sites (`eta_pre` is the same argument under the name of
                docs/NONLINEAR.md §6, and wins when both are given): it commutes with U_0, so every stream draws which
                photons enter (`imperfections.partition`) and runs the expansion of its own photon count. Loss
                AFTER the last site is `eta_post` (binomial thinning of the counts, `imperfections.thin_counts`);
                the two compose, eta = eta_pre eta_post. Loss BETWEEN two sites is not covered (it couples the
                branches through the traced photon: docs/NONLINEAR.md §6).
        indist  partial distinguishability, x-model or `imperfections.Distinguishability`: the site acts on the
                TOTAL photon number of its mode, so in the branch basis the species are independent given the
                branch and share it - the probability is coherent over branches and incoherent over the species
                partition, and the same Cauchy-Schwarz bound holds species by species (docs/NONLINEAR.md §6).
        Every loss / species partition is proposed in the SAME batch, one `fock_sample` call per species size
        (`_reject_batch`), so a lossy run keeps the kernels busy. `generator` lives on U_0's device. `stats`:
        proposals, accepted, rejection_factor, branches, groups (distinct loss / species partitions).
        """
        from .imperfections import partition, spec, thin_counts
        loss = loss if eta_pre is None else eta_pre
        if len(set(self.in_modes)) != self.n:
            raise ValueError("sample: one photon per input mode")
        lossless = loss is None or float(loss) >= 1.0
        self.stats = {"proposals": 0, "accepted": 0, "rejection_factor": self.rejection_factor(),
                      "branches": self.n_branches, "groups": 1}
        if method == "strong":
            if not lossless or spec(indist) is not None or eta_post is not None:
                raise NotImplementedError('loss / indist: method="rejection" only (the strong side is lossless)')
            occ = self.occupations()
            idx = torch.multinomial(self.probs().to(torch.float64), shots, replacement=True, generator=generator)
            self.stats.update(proposals=shots, accepted=shots)
            return occ[idx]
        if method != "rejection":
            raise ValueError(f'unknown method {method!r}: "rejection" or "strong"')
        base = base_sampler if base_sampler is not None else self._base_sampler(backend)
        amp_backend = amp_backend or ("torch" if self.dev.type == "cpu" else backend)
        out = torch.zeros(shots, self.m, dtype=torch.int64, device=self.dev)
        gen_cpu = generator if generator is None or generator.device.type == "cpu" else None
        parts = [((tuple(range(self.n)),), (), torch.arange(shots))] if lossless and spec(indist) is None \
            else partition(self.n, shots, loss, indist, generator=gen_cpu)
        self.stats["groups"] = len(parts)
        groups, done = [], []
        for subsets, singles, ids in parts:
            cols_list = [tuple(self.in_modes[j] for j in sub) for sub in subsets]
            cols_list += [(self.in_modes[j],) for j in singles]
            photons = sum(len(cols) for cols in cols_list)
            if photons == 0:                                                         # every photon lost: the vacuum
                continue
            if len(cols_list) == 1 and self.branch_table(photons)[1].shape[0] == 1:   # linear: nothing to reject
                cnt = torch.tensor([ids.shape[0]], dtype=torch.int64)
                out[ids.to(self.dev)] = base(self.branch_table(photons)[0], list(cols_list[0]), cnt, generator).to(self.dev)
                self.stats["proposals"] += ids.shape[0]; self.stats["accepted"] += ids.shape[0]
                continue
            groups.append((cols_list, photons, ids.to(self.dev))); done.append(0)
        need = torch.tensor([gr[2].shape[0] for gr in groups], dtype=torch.int64)
        while bool((need > 0).any()):
            if self.stats["proposals"] >= max_draws:
                raise RuntimeError(f"non-linear sampler: {max_draws} proposals drawn, {int(need.sum())} samples short")
            for g, kept in enumerate(self._reject_batch(groups, need, base, generator, backend, amp_backend, max_batch)):
                take = min(kept.shape[0], int(need[g]))
                if take:
                    out[groups[g][2][done[g]:done[g] + take]] = kept[:take]
                    done[g] += take; need[g] -= take
        if eta_post is not None and float(eta_post) < 1.0:
            out = thin_counts(out, float(eta_post), generator=gen_cpu)
        return out


def _glynn(M: torch.Tensor) -> torch.Tensor:
    """Permanents of the trailing [n, n] blocks of M by Glynn's formula with a Gray code: [...] complex128.
    2^(n-1) steps, each O(n) per matrix; n = 0 gives 1.
    """
    n = M.shape[-1]
    if n == 0:
        return torch.ones(M.shape[:-2], dtype=M.dtype, device=M.device)
    M = M.to(torch.complex128)
    s = M.sum(-2)                                                                    # delta = all +1: row sums [..., n]
    total = s.prod(-1)
    delta = [1] * n
    for g in range(1, 1 << (n - 1)):
        i = (g & -g).bit_length()                                                    # flipped row 1..n-1 (row 0 fixed)
        delta[i] = -delta[i]
        s = s + 2 * delta[i] * M[..., i, :]
        total = total + (-1 if g & 1 else 1) * s.prod(-1)
    return total / (1 << (n - 1))
