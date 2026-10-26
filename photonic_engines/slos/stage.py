"""Stage kernel: an emitter chain next to Fock photons (`StageChain` + a batch of Fock inputs).

The input of one stage of a hybrid spin-photon program is the emitter chain of `StageChain` (events
E_k = sum_l M_{k,l} (x) a_{p_{k,l}}^dag on the register) applied to a Fock state |s> of the stage's modes
(heralded or carried photons). For input s with N = |s| photons the photonic state after the chain is

    |Phi> = sum_x <.| M_{x_n} ... M_{x_1} |.>  prod_k A_{p_{k,x_k}}  prod_p A_p^{s_p} / sqrt(s_p!) |0>,
    A_p = sum_i U[i, p] a_i^dagger,

so the SLOS recurrence runs in two parts on one basis `get_layers(N + n, m)`:

    prefix   f = prod_p A_p^{s_p} / sqrt(s!) |0>       N scalar raises (FockSLOSKernel's step), no bond index
    chain    Phi_0 = psi0 (x) f; event k raises layer N + k -> N + k + 1 with the chi x chi coefficient
             C_j = sum_l U[j, p_{k,l}] M_{k,l}        (EmitterSLOSKernel's step, layer index shifted by N)

Raises are creation operators in the monomial basis (deferred normalisation, as in every SLOS kernel), so an
emission into a mode that s occupies carries its sqrt(s_p + 1) by construction; the output norm stays deferred
(`prob_norms` = prod t_j!, `StageGroup.physical()`). Inputs are grouped by N; a group is one recurrence with the bond axis carrying
(input, initial register vector), Q = B_N * chi for `kraus` — no Python loop over inputs.

Dispatch (no regression): a single-photon chain with n >= 1 and no Fock input (vacuum, B = 1) is today's
`EmitterSLOSKernel` on `chain.to_mps()`; `StageKernel` forwards to it, so tensors and arithmetic are today's.
complex128 by default. Differentiable in U and in `ops_tensor`. Not supported: loss / distinguishability parameters
(write loss as photon-less labels instead).

Labels of any photon number (graded recurrence): with a photon-less label (empty word, the emitter acts and no photon
enters: internal modes' `lost` operator) or a multiphoton one, every photon number N + T the chain reaches is live,
one layer each on `get_layers(N + N_max, m)`. Per event and word group (the labels with one word add):

    |w| = 0   layer T -> T      psi <- M psi                      (bond only)
    |w| = 1   layer T -> T + 1  one slab pass, C_j = sum U[j, p] M  (today's step, all single-photon groups at once)
    |w| >= 2  layer T -> T + |w|  |w| scalar raises A_p (coefficients U[:, p]), then M / sqrt(w!)

The outputs list every reachable photon number, largest first. Labels with different `env` records are incoherent:
the recurrence runs once per env word and each env word is one more Kraus operator per outcome (`StageGroup.envs`),
so time and memory scale with E = prod_k (distinct env values of event k) — prod_j (R_j + Q_j) for a tick of
internal-mode emitters with R_j unobserved outcomes and Q_j lost operators (`StageChain.from_bin_emitters`).
A single-photon chain with one env word never takes this path.

Partial detection (carry-over, route B): with detected modes D and carried modes C (the complement), every full
output pattern o splits uniquely as (q, c) = (o on D, o on C). The block operator of a detected pattern q,

    K_q [C_q chi, B chi],   K_q[(c, f), (b, i)] = <q + c| Phi^(b)>[f, i]   (physical normalisation),

is an index regroup of the full-output Kraus set (no amplitude computed twice); its rows run over the carried-out
basis of q (the distinct c reached, occupations over all m modes, zero on D), zero where q + c has another photon
number than input b's outputs. `blocks(U)` returns them, `apply(U, rho)` the blockwise update of a stage simulator,
rho_q = (K_q (x) 1_X) rho (K_q (x) 1_X)^dag, with rho over (input b, chain register, passive rest X), b major. The
regroup tables depend only on (m, the inputs' photon numbers, D) and are built once per kernel. A photons-only stage
(`StageChain.empty`, chi = 1) has scalar blocks, and a source register that does not emit is the rest X: no
Kronecker product is formed.
"""
from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import torch

from ..states import StageChain
from .basis import get_layers
from .emitter import EmitterSLOSKernel


class StageGroup(NamedTuple):
    """The outputs of the inputs with N photons: `inputs` [B_N] (positions in the kernel's input batch),
    `patterns` [S, m] (output occupations: N + n photons for a single-photon chain; with photon-less or multiphoton
    labels every reachable N + T, largest first), `prob_norms` [S] (prod t_j!, in the kernel's real dtype), `value`
    (`kind` "kraus": [(Bu,) (E,) B_N, S, chi, chi] raw Kraus operators; "probs": [(Bu,) B_N, S] probabilities;
    "bond": [(Bu,) (E,) B_N, chi, S] raw bond amplitudes). The E axis (env words, `envs` [E, n]) is present only for
    a chain whose labels carry different env records: each env word is one more Kraus operator per outcome (the
    channel sums over it; probabilities are summed already). Raw = deferred normalisation; `physical()` gives the
    physical operators / amplitudes (raw * sqrt(prob_norms)).
    """
    inputs: torch.Tensor
    patterns: torch.Tensor
    prob_norms: torch.Tensor
    value: torch.Tensor
    kind: str
    envs: torch.Tensor | None = None

    def physical(self) -> torch.Tensor:
        """`value` in physical normalisation: Kraus operators / bond amplitudes times sqrt(prod t_j!) (probabilities
        are returned as they are, already physical).
        """
        root = torch.sqrt(self.prob_norms).to(self.value.dtype)                    # real dtype of the kernel
        if self.kind == "kraus":
            return self.value * root[:, None, None]
        if self.kind == "bond":
            return self.value * root
        return self.value


class CarryClass(NamedTuple):
    """The detected patterns whose carried-out basis has the same size C. `q` [nq] indices into
    `StageCarry.detected`; `basis` [nq, C] indices into `StageCarry.carried` (the carried-out basis of each q, in the
    order of the rows of its block); `value`: `blocks` -> [(Bu,) nq, C chi, B chi] physical block operators K_q;
    `apply` -> [(Bu,) (R,) nq, C chi X, C chi X] the updated blocks rho_q (unnormalised, Tr rho_q = p(q) Tr rho);
    `probs` [(Bu,) (R,) nq] = Tr rho_q (`apply` only, None for `blocks`).
    """
    q: torch.Tensor
    basis: torch.Tensor
    value: torch.Tensor
    probs: torch.Tensor | None


class StageCarry(NamedTuple):
    """Partial detection output. `detected` [Qd, m] the detected patterns q (occupations over all modes, zero on
    the carried modes); `carried` [T, m] every carried-out occupation reached (zero on the detected modes);
    `classes` the per-q data grouped by carried-basis size; `deviation` (`apply` only) [(Bu,) (R,)] =
    sum_q Tr rho_q - Tr rho, zero (to rounding) for a complete chain, a unitary U and inputs the chain does not emit
    into (then sum_q K_q^dag K_q = 1).
    """
    detected: torch.Tensor
    carried: torch.Tensor
    classes: list
    deviation: torch.Tensor | None


def _layer_occupations(layers, k: int, m: int) -> np.ndarray:
    """Occupations [S_k, m] of layer k of a basis (numpy or torch states: k sorted photon positions per state)."""
    t = layers.states[k]
    t = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    occ = np.zeros((t.shape[0], m), dtype=np.int64)
    ar = np.arange(t.shape[0])
    for j in range(k):
        occ[ar, t[:, j].astype(np.int64)] += 1
    return occ


def _unique_rows(a: np.ndarray):
    """Distinct rows of a (non-negative ints) in order of first appearance, and each row's index into them. The rows
    are packed into one int64 mixed-radix key when it fits (a 1-D unique), else `np.unique(axis=0)`.
    """
    base = int(a.max()) + 1 if a.size else 1
    if a.shape[1] * math.log2(max(base, 2)) < 62:
        key = (a * (base ** np.arange(a.shape[1] - 1, -1, -1, dtype=np.int64))).sum(axis=1)
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    else:
        _, first, inv = np.unique(a, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return a[first[order]], rank[inv.reshape(-1)]


class StageKernel:
    """Precompiled strong evaluator of one stage: `StageChain` x a batch of Fock inputs, full or partial detection.

    inputs : int [B, m] carried-in / heralded Fock occupations (None: the vacuum, B = 1)
    detected : the detected modes (None: all modes); the others are carried (`blocks`, `apply`)

    `kraus(U)`, `probs(U)`, `bond_amplitudes(U)` return one `StageGroup` per photon number N of the inputs (inputs
    with N photons share one recurrence), over the full output patterns; U is [m, m] or [Bu, m, m] (a leading Bu
    axis on `value`). `blocks(U)` / `apply(U, rho)` regroup them by detected pattern (`StageCarry`). Built once,
    called with many U.
    """

    def __init__(self, chain: StageChain, inputs=None, detected=None, device: str = "cpu",
                 dtype: torch.dtype = torch.complex128, ops_tensor: torch.Tensor | None = None):
        if not isinstance(chain, StageChain):
            raise TypeError("chain must be a StageChain (StageChain.from_mps for an EmitterMPS)")
        m = chain.m
        det = list(range(m)) if detected is None else sorted({int(x) for x in detected})
        if det and (det[0] < 0 or det[-1] >= m):
            raise ValueError(f"detected modes must lie in 0..{m - 1}")
        self.detected = tuple(det)
        self.carried_modes = tuple(p for p in range(m) if p not in set(det))
        self._carry = None
        self.chain, self.n, self.m, self.chi = chain, chain.n, m, chain.chi
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        if inputs is None:
            occ = np.zeros((1, m), dtype=np.int64)
        else:
            occ = np.asarray(inputs, dtype=np.int64)
            if occ.ndim == 1:
                occ = occ[None]
            if occ.ndim != 2 or occ.shape[1] != m or occ.shape[0] < 1 or (occ < 0).any():
                raise ValueError(f"inputs must be a non-negative int array [B, {m}]")
        self.inputs = occ
        self._dispatch = None
        # graded (R2): a label emitting 0 or >= 2 photons, or labels with different env records
        self._graded = (not chain.is_single_photon) or chain.env_words.shape[0] > 1
        self._envw = chain.env_words if self._graded else None
        if self._envw is not None and self._envw.shape[0] > 1024:
            warnings.warn(f"StageKernel: {self._envw.shape[0]} env words (prod over events of their env values): one "
                          "recurrence each, so time and memory scale with that count", RuntimeWarning, stacklevel=2)
        if not self._graded and self.n >= 1 and occ.shape[0] == 1 and not occ.any():
            # today's case: forward to EmitterSLOSKernel (same tensors, same arithmetic)
            self._dispatch = EmitterSLOSKernel(chain.to_mps(), device=str(self.device), dtype=dtype,
                                               ops_tensor=self._mps_ops_tensor(ops_tensor))
            return

        # the chain as an EmitterMPS-like operator set: labels merged per mode (to_mps), padded slots on modes[k][0]
        # with a zero operator, ops masked once so padded slots carry no gradient (as EmitterSLOSKernel, §R1)
        if self._graded:
            # the chain's own labels; padded slots are never read (each event's groups list its real labels only)
            if ops_tensor is not None and tuple(ops_tensor.shape) != tuple(chain.ops.shape):
                raise ValueError("ops_tensor must have the chain's ops shape [n, L, chi, chi]")
            ops = torch.as_tensor(chain.ops) if ops_tensor is None else ops_tensor
            self._ops = ops.to(device=self.device, dtype=dtype)
            self._in_mode = None
            self._events = self._word_groups()
        elif self.n >= 1:
            mps = chain.to_mps()
            ops = self._mps_ops_tensor(ops_tensor)
            ops = (torch.as_tensor(mps.ops) if ops is None else ops).to(device=self.device, dtype=dtype)
            if mps.is_ragged:
                ops = ops * torch.as_tensor(mps.label_mask(), device=self.device)[..., None, None]
            self._ops = ops                                                         # [n, L, chi, chi]
            self._in_mode = torch.as_tensor(mps.modes_padded(), device=self.device)  # [n, L]
        else:
            if ops_tensor is not None:
                raise ValueError("ops_tensor: an empty chain has no operators")
            self._ops, self._in_mode = None, None
        self.s0 = torch.as_tensor(chain.s0).to(device=self.device, dtype=dtype)
        self.final = None if chain.final is None else torch.as_tensor(chain.final).to(device=self.device, dtype=dtype)

        # inputs grouped by photon number N; per group: its basis, the input photons' modes (one row per input,
        # the N modes with multiplicity), 1/sqrt(s!) and the raising edges of every layer
        self._groups = []
        Ns = occ.sum(axis=1)
        emitted = chain.photons_max if self._graded else self.n
        self._totals = chain.photon_numbers if self._graded else (self.n,)    # emitted photon numbers, largest first
        for N in sorted({int(x) for x in Ns}):
            idx = np.nonzero(Ns == N)[0]
            layers = get_layers(N + emitted, m, device=str(self.device))
            photons = np.array([np.repeat(np.arange(m), occ[b]) for b in idx], dtype=np.int64).reshape(len(idx), N)
            inv_norm = [1.0 / math.sqrt(math.prod(math.factorial(int(x)) for x in occ[b])) for b in idx]
            edges = []
            for k in range(N + emitted):
                p, c, off = layers.edges_by_mode(k)
                edges.append((torch.as_tensor(p).to(self.device).long(), torch.as_tensor(c).to(self.device).long(),
                              [int(x) for x in off], layers.size(k + 1)))
            if self._graded:
                # outputs of every reachable photon number N + T, largest first (internal modes' `patterns` order)
                patterns = torch.cat([torch.as_tensor(_layer_occupations(layers, N + T, m)) for T in self._totals])
                norms = torch.cat([torch.as_tensor(layers.prob_norms(N + T)).cpu() for T in self._totals])
            else:
                patterns, norms = torch.as_tensor(layers.final_occupations()), torch.as_tensor(layers.output_prob_norms())
            self._groups.append({
                "idx": torch.as_tensor(idx, device=self.device), "N": N, "layers": layers, "edges": edges,
                "photons": torch.as_tensor(photons, device=self.device),
                "inv_norm": torch.as_tensor(inv_norm, dtype=self.rdtype, device=self.device),
                "patterns": patterns.to(self.device),
                "norms": norms.to(device=self.device, dtype=self.rdtype),
            })

    def _word_groups(self) -> list[dict]:
        """Per event, per env value: the labels grouped by word (labels with the same (word, env) add coherently):
        (|w|, the word's photon modes with multiplicity, 1/sqrt(w!), the label indices).
        """
        out = []
        for w, e, P in zip(self.chain.words, self.chain.env, self.chain.event_modes):
            per_env = {}
            for v in sorted(set(e.tolist())):
                groups = {}
                for lab in np.nonzero(e == v)[0].tolist():
                    groups.setdefault(tuple(w[lab].tolist()), []).append(lab)
                items = []
                for word, labs in groups.items():
                    modes = [int(p) for p, c in zip(P, word) for _ in range(c)]
                    inv = 1.0 / math.sqrt(math.prod(math.factorial(c) for c in word))
                    items.append((sum(word), modes, inv, torch.as_tensor(labs, device=self.device)))
                per_env[v] = items
            out.append(per_env)
        return out

    def _raise_scalar(self, g, k: int, x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        """One photon raised from layer k: x [Bu, Q, chi, size(k)] -> [Bu, Q, chi, size(k + 1)], the raised mode j
        weighted by coeff[:, j] (A_p = sum_j U[j, p] a_j^dag: coeff = U[:, :, p]).
        """
        parent, child, off, size = g["edges"][k]
        nxt = torch.zeros((*x.shape[:-1], size), dtype=self.cdtype, device=self.device)
        for j in range(self.m):
            a, b_ = off[j], off[j + 1]
            if b_ <= a:
                continue
            nxt.index_add_(3, child[a:b_], coeff[:, j, None, None, None] * x[..., parent[a:b_]])
        return nxt

    def _graded_chain(self, g, U: torch.Tensor, psi: torch.Tensor, envw) -> dict:
        """The graded recurrence of one env word: psi [Bu, Q, chi, size(N)] -> {T: [Bu, Q, chi, size(N + T)]} for
        every emitted photon number T. Per event, per word group (labels of the env value envw[k] with one word):
        |w| = 0 applies M in place; the |w| = 1 groups share one slab pass with C_j = sum U[j, p] M (as today's
        kernel); |w| >= 2 raises its photons one by one (scalar coefficients) and applies M / sqrt(w!).
        """
        def add(new: dict, T: int, x: torch.Tensor) -> None:
            new[T] = x if T not in new else new[T] + x

        state = {0: psi}
        for k in range(self.n):
            items = self._events[k].get(int(envw[k]), [])
            singles = [it for it in items if it[0] == 1]
            C = None
            if singles:                                     # per event, shared by every layer T
                Ms = torch.stack([self._ops[k, it[3]].sum(0) for it in singles])              # [ns, chi, chi]
                cols = torch.as_tensor([it[1][0] for it in singles], device=self.device)
                C = torch.einsum("bjl,lts->bjts", U[:, :, cols], Ms)                           # [Bu, m, chi, chi]
            others = [(w, modes, inv, self._ops[k, labs].sum(0)) for w, modes, inv, labs in items if w != 1]
            new: dict = {}
            for T, x in state.items():
                L = g["N"] + T
                if C is not None:
                    parent, child, off, size = g["edges"][L]
                    nxt = torch.zeros((*x.shape[:-1], size), dtype=self.cdtype, device=self.device)
                    for j in range(self.m):
                        a, b_ = off[j], off[j + 1]
                        if b_ <= a:
                            continue
                        nxt.index_add_(3, child[a:b_], torch.matmul(C[:, j].unsqueeze(1), x[..., parent[a:b_]]))
                    add(new, T + 1, nxt)
                for w, modes, inv, M in others:
                    y = x
                    for r, p in enumerate(modes):
                        y = self._raise_scalar(g, L + r, y, U[:, :, p])
                    y = torch.einsum("st,bqtx->bqsx", M, y)
                    add(new, T + w, y * inv if w >= 2 else y)
            state = new
        return state

    def _graded_phi(self, g, U: torch.Tensor, vecs: torch.Tensor, envw) -> torch.Tensor:
        """Raw amplitudes [Bu, B_N, R, chi, S] of every reachable output (largest photon number first) for one env
        word, from initial register vectors vecs [R, chi].
        """
        f = self._prefix(g, U)
        Bu, BN, R = U.shape[0], f.shape[1], vecs.shape[0]
        psi = (vecs.to(self.cdtype)[None, None, :, :, None] * f[:, :, None, None, :]).reshape(Bu, BN * R, self.chi, -1)
        state = self._graded_chain(g, U, psi, envw)
        parts = []
        for T in self._totals:
            x = state.get(T)
            if x is None:
                x = torch.zeros((Bu, BN * R, self.chi, g["layers"].size(g["N"] + T)), dtype=self.cdtype,
                                device=self.device)
            parts.append(x)
        return torch.cat(parts, dim=-1).reshape(Bu, BN, R, self.chi, -1)

    def _phi_any(self, g, U, vecs, e=None):
        return self._graded_phi(g, U, vecs, self._envw[e]) if self._graded else self._phi(g, U, vecs)

    @property
    def env_count(self) -> int:
        """Number of env words E (1 for a coherent chain; then `value` has no env axis)."""
        return 1 if self._envw is None else int(self._envw.shape[0])

    def _mps_ops_tensor(self, ops_tensor):
        """A caller's [n, L, chi, chi] tensor for the chain's operators; refused when `to_mps` merges labels."""
        if ops_tensor is None:
            return None
        if self.n == 0 or tuple(ops_tensor.shape) != tuple(self.chain.ops.shape):
            raise ValueError("ops_tensor must have the chain's ops shape [n, L, chi, chi]")
        if tuple(len(mk) for mk in self.chain.to_mps().modes) != self.chain.labels:
            raise ValueError("ops_tensor: not supported for a chain whose labels share a mode (merged by to_mps)")
        return ops_tensor

    # ---- recurrence ------------------------------------------------------------------------------------------------
    def _prefix(self, g, U: torch.Tensor) -> torch.Tensor:
        """The Fock prefix of group g: [Bu, B_N, size(N)] raw amplitudes of prod_p A_p^{s_p} / sqrt(s!) |0>."""
        Bu, BN = U.shape[0], g["photons"].shape[0]
        f = torch.ones((Bu, BN, 1), dtype=self.cdtype, device=self.device)
        for r in range(g["N"]):
            parent, child, off, size = g["edges"][r]
            Ur = U[:, :, g["photons"][:, r]]                                       # [Bu, m, B_N]: U[j, p_{b,r}]
            nxt = torch.zeros((Bu, BN, size), dtype=self.cdtype, device=self.device)
            for j in range(self.m):
                a, b_ = off[j], off[j + 1]
                if b_ <= a:
                    continue
                nxt.index_add_(2, child[a:b_], Ur[:, j, :, None] * f[:, :, parent[a:b_]])
            f = nxt
        return f * g["inv_norm"].to(self.cdtype)[None, :, None]

    def _chain(self, g, U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Events 1..n on group g's layers N..N+n: psi [Bu, Q, chi, size(N)] -> [Bu, Q, chi, size(N + n)]."""
        Bu, Q, chi = psi.shape[0], psi.shape[1], self.chi
        for k in range(self.n):
            parent, child, off, size = g["edges"][g["N"] + k]
            nxt = torch.zeros((Bu, Q, chi, size), dtype=self.cdtype, device=self.device)
            C = torch.einsum("bjl,lts->bjts", U[:, :, self._in_mode[k]], self._ops[k])   # [Bu, m, chi, chi]
            for j in range(self.m):
                a, b_ = off[j], off[j + 1]
                if b_ <= a:
                    continue
                slab = torch.matmul(C[:, j].unsqueeze(1), psi[:, :, :, parent[a:b_]])   # [Bu, Q, chi, e]
                nxt.index_add_(3, child[a:b_], slab)
            psi = nxt
        return psi

    def _phi(self, g, U: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
        """Final raw amplitudes [Bu, B_N, R, chi, S] from initial register vectors vecs [R, chi]."""
        f = self._prefix(g, U)                                                     # [Bu, B_N, size(N)]
        Bu, BN, R = U.shape[0], f.shape[1], vecs.shape[0]
        psi = vecs.to(self.cdtype)[None, None, :, :, None] * f[:, :, None, None, :]   # [Bu, B_N, R, chi, size(N)]
        psi = self._chain(g, U, psi.reshape(Bu, BN * R, self.chi, -1))
        return psi.reshape(Bu, BN, R, self.chi, -1)

    def _prep(self, U):
        U = torch.as_tensor(U)
        batched = U.dim() == 3
        U = (U if batched else U.unsqueeze(0)).to(device=self.device, dtype=self.cdtype)
        if U.shape[-1] != self.m or U.shape[-2] != self.m:
            raise ValueError(f"U must be [m, m] or [B, m, m] with m = {self.m}")
        return U, batched

    def _group(self, g, value, kind):
        envs = None
        if self.env_count > 1 and kind != "probs":
            envs = torch.as_tensor(self._envw, device=self.device)
        return StageGroup(g["idx"], g["patterns"], g["norms"], value, kind, envs)

    def _dispatch_group(self, value, kind):
        layers = self._dispatch.layers
        return [StageGroup(torch.zeros(1, dtype=torch.long, device=self.device),
                           torch.as_tensor(layers.final_occupations()).to(self.device),
                           torch.as_tensor(layers.output_prob_norms()).to(device=self.device, dtype=self.rdtype),
                           value, kind)]

    # ---- public API ------------------------------------------------------------------------------------------------
    def kraus(self, U: torch.Tensor, chunk: int | None = None) -> list[StageGroup]:
        """Stage Kraus operators K_t^(b) = <t| Phi^(b)> (chi x chi, final x initial) per input b and output pattern t,
        deferred normalisation (`StageGroup.physical()` for the physical operators). `value`: [B_N, S, chi, chi]
        (or [Bu, B_N, S, chi, chi]); an env axis E before B_N when the chain has several env words (`envs`).
        chunk: propagate that many inputs of a group at a time (memory).
        """
        U, batched = self._prep(U)
        if self._dispatch is not None:
            K = self._dispatch.kraus(U if batched else U[0], chunk=chunk)
            return self._dispatch_group(K.unsqueeze(-4), "kraus")
        eye = torch.eye(self.chi, dtype=self.cdtype, device=self.device)
        E = self.env_count
        out = []
        for g in self._groups:
            BN = g["photons"].shape[0]
            step = BN if chunk is None else max(1, int(chunk))
            parts = []
            for b0 in range(0, BN, step):
                sub = {**g, "photons": g["photons"][b0:b0 + step], "inv_norm": g["inv_norm"][b0:b0 + step]}
                if not self._graded:
                    phi = self._phi(sub, U, eye)                                   # [Bu, b, init, final, S]
                    parts.append(phi.permute(0, 1, 4, 3, 2))                       # [Bu, b, S, final, init]
                    continue
                ks = [self._graded_phi(sub, U, eye, self._envw[e]).permute(0, 1, 4, 3, 2) for e in range(E)]
                parts.append(torch.stack(ks, dim=1) if E > 1 else ks[0])           # [Bu, (E,) b, S, final, init]
            K = torch.cat(parts, dim=2 if E > 1 else 1)
            out.append(self._group(g, K if batched else K[0], "kraus"))
        return out

    def bond_amplitudes(self, U: torch.Tensor) -> list[StageGroup]:
        """Raw amplitudes from `s0` with the register attached, `value` [B_N, chi, S] (index = final register state);
        an env axis E before B_N when the chain has several env words.
        """
        U, batched = self._prep(U)
        if self._dispatch is not None:
            phi = self._dispatch.bond_amplitudes(U if batched else U[0])
            return self._dispatch_group(phi.unsqueeze(-3), "bond")
        E = self.env_count
        out = []
        for g in self._groups:
            if not self._graded:
                phi = self._phi(g, U, self.s0[None])[:, :, 0]                      # [Bu, B_N, chi, S]
            else:
                ph = [self._graded_phi(g, U, self.s0[None], self._envw[e])[:, :, 0] for e in range(E)]
                phi = torch.stack(ph, dim=1) if E > 1 else ph[0]                   # [Bu, (E,) B_N, chi, S]
            out.append(self._group(g, phi if batched else phi[0], "bond"))
        return out

    def probs(self, U: torch.Tensor) -> list[StageGroup]:
        """Output probabilities per input, `value` [B_N, S]: the register traced (`final` None) or projected on
        `final` (then unnormalised, weight = the projection probability), as `EmitterSLOSKernel.probs`.
        """
        U, batched = self._prep(U)
        if self._dispatch is not None:
            p = self._dispatch.probs(U if batched else U[0])
            return self._dispatch_group(p.unsqueeze(-2), "probs")
        out = []
        for g, ga in zip(self._groups, self.bond_amplitudes(U if batched else U[0])):
            phi = ga.value
            if self.final is not None:
                amp = torch.einsum("s,...sj->...j", self.final.conj(), phi)
                p = amp.real ** 2 + amp.imag ** 2
            else:
                p = (phi.real ** 2 + phi.imag ** 2).sum(dim=-2)
            if self.env_count > 1:                                                 # incoherent: sum the env words
                p = p.sum(dim=1 if batched else 0)
            out.append(self._group(g, p * g["norms"], "probs"))
        return out

    # ---- partial detection (route B) -------------------------------------------------------------------------------
    def _carry_tables(self) -> dict:
        """Regroup tables, built once: every full output pattern (group by group, in `kraus` order) -> its detected
        pattern q, its class (carried-basis size), its index in the class and its row slot in K_q. q and the carried
        occupations are numbered by first appearance, so with full detection the q follow the patterns of `kraus`.
        """
        if self._carry is not None:
            return self._carry
        if self._dispatch is not None:
            occ = self._dispatch.layers.final_occupations()                  # a device tensor on CUDA
            occ = occ.cpu().numpy() if isinstance(occ, torch.Tensor) else np.asarray(occ)
            meta = [(np.zeros(1, dtype=np.int64), occ)]
        else:
            meta = [(g["idx"].cpu().numpy(), g["patterns"].cpu().numpy()) for g in self._groups]
        P = np.concatenate([p for _, p in meta]).astype(np.int64)
        on_d = np.zeros(self.m, dtype=bool)
        on_d[list(self.detected)] = True
        qtab, qid = _unique_rows(P * on_d)
        ctab, cid = _unique_rows(P * ~on_d)
        # one row of K_q per distinct (q, c) pair, its slot = its rank among q's pairs (first appearance). On a
        # single-photon chain each pair is one output pattern. On a graded chain (photon-less or multiphoton
        # labels) inputs of different photon numbers can reach the same full pattern: those patterns share the
        # row, each filling its own input columns.
        nq = qtab.shape[0]
        ptab, pid = _unique_rows(np.stack([qid, cid], axis=1))
        pq = ptab[:, 0]
        order = np.argsort(pq, kind="stable")
        size = np.bincount(pq, minlength=nq)
        start = np.concatenate([[0], np.cumsum(size)[:-1]])
        pslot = np.empty_like(pq)
        pslot[order] = np.arange(pq.size) - start[pq[order]]
        basis = np.empty(pq.size, dtype=np.int64)
        basis[start[pq] + pslot] = ptab[:, 1]
        slot = pslot[pid]
        sizes = sorted({int(x) for x in size})
        cls_of = np.searchsorted(np.asarray(sizes), size)
        local = np.empty(nq, dtype=np.int64)
        classes = []
        for ci, C in enumerate(sizes):
            qs = np.nonzero(cls_of == ci)[0]
            local[qs] = np.arange(qs.size)
            rows = basis[(start[qs][:, None] + np.arange(C)[None, :])]
            if C > 1 and not np.all(np.diff(np.sort(rows, axis=1), axis=1)):
                raise RuntimeError("StageKernel: a carried-out state appears twice in one block (internal error)")
            classes.append((C, torch.as_tensor(qs, device=self.device), torch.as_tensor(rows, device=self.device)))
        dev = self.device
        groups, r0 = [], 0
        for idx, p in meta:
            r1 = r0 + p.shape[0]
            groups.append({"idx": torch.as_tensor(idx, device=dev),
                           "cls": torch.as_tensor(cls_of[qid[r0:r1]], device=dev),
                           "ql": torch.as_tensor(local[qid[r0:r1]], device=dev),
                           "slot": torch.as_tensor(slot[r0:r1], device=dev)})
            r0 = r1
        self._carry = {"detected": torch.as_tensor(qtab, device=dev), "carried": torch.as_tensor(ctab, device=dev),
                       "classes": classes, "groups": groups, "B": int(self.inputs.shape[0])}
        return self._carry

    def _physical_kraus(self, U: torch.Tensor) -> list[list[torch.Tensor]]:
        """Physical full-output Kraus operators, per env word a list per group of [Bu, B_N, S, chi, chi] (U already
        prepared, batched).
        """
        Ks = [g.physical() for g in self.kraus(U)]
        if self.env_count == 1:
            return [Ks]
        return [[K[:, e] for K in Ks] for e in range(self.env_count)]

    def _class_blocks(self, Ks, ci: int, q0: int, q1: int) -> torch.Tensor:
        """K_q of class ci for its q in [q0, q1): [Bu, q1 - q0, C chi, B chi], scattered from the groups' Kraus sets."""
        t = self._carry
        C, chi, B = t["classes"][ci][0], self.chi, t["B"]
        Bu = Ks[0].shape[0]
        Z = torch.zeros((Bu, q1 - q0, C, B, chi, chi), dtype=self.cdtype, device=self.device)
        for K, gt in zip(Ks, t["groups"]):
            sel = torch.nonzero((gt["cls"] == ci) & (gt["ql"] >= q0) & (gt["ql"] < q1)).reshape(-1)
            if sel.numel() == 0:
                continue
            vals = K[:, :, sel].transpose(1, 2)                                     # [Bu, ns, B_N, chi, chi]
            Z[:, (gt["ql"][sel] - q0)[:, None], gt["slot"][sel][:, None], gt["idx"][None, :]] = vals
        return Z.permute(0, 1, 2, 4, 3, 5).reshape(Bu, q1 - q0, C * chi, B * chi)

    def blocks(self, U: torch.Tensor) -> StageCarry:
        """The block operators K_q of every detected pattern q (`CarryClass.value` [(Bu,) nq, C chi, B chi], physical
        normalisation; rows (carried-out basis state, final register), columns (input b, initial register)); with
        several env words an env axis [(Bu,) E, nq, ...]: one block operator per env word (the channel sums over it).
        """
        U, batched = self._prep(U)
        t = self._carry_tables()
        Ks = self._physical_kraus(U)
        classes = []
        for ci, (_, qs, basis) in enumerate(t["classes"]):
            Kl = [self._class_blocks(Ke, ci, 0, qs.numel()) for Ke in Ks]
            K = torch.stack(Kl, dim=1) if len(Kl) > 1 else Kl[0]
            classes.append(CarryClass(qs, basis, K if batched else K[0], None))
        return StageCarry(t["detected"], t["carried"], classes, None)

    def apply(self, U: torch.Tensor, rho: torch.Tensor, rest: int = 1, chunk: int | None = None,
              chunk_bytes: int = 2 ** 30, tol: float | None = None, strict: bool = False) -> StageCarry:
        """The blockwise stage update rho_q = (K_q (x) 1_rest) rho (K_q (x) 1_rest)^dag for every detected pattern q.

        rho  : [B chi rest, B chi rest] or a batch [R, ...]; index (input b, chain register, passive rest), b major
               (a stage simulator's carried-photon blocks rho[b, b'] on the register, the register = chain (x) rest)
        rest : dimension of a passive register next to the chain's (sources that do not emit in this stage; for a
               photons-only stage, chi = 1, the whole register)
        chunk: number of q per pass; None: sized from `chunk_bytes` (the blocks of a pass and the update's working
               tensors, per q: C chi x B chi blocks + R (C chi rest) (B chi rest) + R (C chi rest)^2 entries)
        tol  : bound on |deviation| relative to Tr rho (None: 1e3 eps of the kernel's dtype); above it `apply` warns,
               or raises with `strict=True`. A stage simulator should run strict (or assert on `deviation`).

        Returns `StageCarry` with `CarryClass.value` = rho_q [(Bu,) (R,) nq, C chi rest, C chi rest], `probs` = Tr
        rho_q and `deviation` = sum_q Tr rho_q - Tr rho (an a-posteriori check of the update). The deviation is zero
        to rounding only for a complete chain, a unitary U and inputs the chain does not emit into: an input that
        occupies an emission mode is defined (creation operators, sqrt(s_p + 1)) but not an isometry, and then the
        deviation is legitimately non-zero (pass strict=False and read it).
        """
        U, batched = self._prep(U)
        t = self._carry_tables()
        X, B, chi = int(rest), t["B"], self.chi
        rho = torch.as_tensor(rho).to(device=self.device, dtype=self.cdtype)
        single = rho.dim() == 2
        rho = rho[None] if single else rho
        D = B * chi * X
        if rho.dim() != 3 or rho.shape[-2:] != (D, D):
            raise ValueError(f"rho must be [{D}, {D}] or [R, {D}, {D}] (B = {B}, chi = {chi}, rest = {X})")
        R = rho.shape[0]
        r5 = rho.reshape(R, B * chi, X, B * chi, X)
        Ks = self._physical_kraus(U)
        Bu = U.shape[0]
        total = torch.zeros((Bu, R), dtype=self.rdtype, device=self.device)
        classes = []
        for ci, (C, qs, basis) in enumerate(t["classes"]):
            nq = qs.numel()
            if chunk is None:
                per_q = 16 * Bu * (C * chi * B * chi + R * C * chi * X * B * chi * X + R * (C * chi * X) ** 2)
                step = max(1, int(chunk_bytes) // per_q)
            else:
                step = max(1, int(chunk))
            parts, probs = [], []
            for q0 in range(0, nq, step):
                q1 = min(nq, q0 + step)
                out = None
                for Ke in Ks:                                                       # env words: incoherent sum
                    Kq = self._class_blocks(Ke, ci, q0, q1)                         # [Bu, nqc, C chi, B chi]
                    T = torch.einsum("uqab,rbxcy->uqraxcy", Kq, r5)
                    o = torch.einsum("uqraxcy,uqdc->urqaxdy", T, Kq.conj())
                    out = o if out is None else out + o
                out = out.reshape(Bu, R, q1 - q0, C * chi * X, C * chi * X)
                parts.append(out)
                probs.append(torch.diagonal(out, dim1=-2, dim2=-1).real.sum(-1))
            val, p = torch.cat(parts, dim=2), torch.cat(probs, dim=2)
            total = total + p.sum(-1)
            if single:
                val, p = val[:, 0], p[:, 0]
            classes.append(CarryClass(qs, basis, val if batched else val[0], p if batched else p[0]))
        tr = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(-1)
        dev = total - tr[None]
        bound = (1e3 * torch.finfo(self.rdtype).eps if tol is None else float(tol)) * tr.abs()[None]
        if bool((dev.abs() > bound).any()):
            msg = (f"StageKernel.apply: sum_q Tr rho_q - Tr rho = {float(dev.abs().max()):.2e} above tol x Tr rho "
                   "(an incomplete chain, a non-unitary U, or inputs that occupy emission modes)")
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        dev = dev[:, 0] if single else dev
        return StageCarry(t["detected"], t["carried"], classes, dev if batched else dev[0])


__all__ = ["CarryClass", "StageCarry", "StageGroup", "StageKernel"]
