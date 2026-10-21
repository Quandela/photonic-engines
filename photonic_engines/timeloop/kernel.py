"""Native engine for periodic linear-optical networks with delay lines: the delay-line register
marched through time.

State = amplitudes over the register basis (`RegisterBasis`: the n_rail wires of the current
period + the L delay-line slots, <= N photons). Per period t: inject the period's input photons
on the rail wires, run the period's program — two-mode gates on any pair of (rail wire, head
slot of a delay line) as block-diagonal Fock matrices (`fock_blocks`, one (k+1)x(k+1) block per
pair photon number k, one batched matmul per gate), optional single-mode losses — then the rail
wires exit: projected on a given pattern (amplitudes), sampled (trajectory sampler) or traced
(density matrix, per-period marginals). Everything is a gather / batched matmul on torch
tensors, so `amplitudes`, `probabilities` and `period_marginals` are differentiable in the
gate matrices and run batched on the GPU.

`PeriodicNetworkKernel` is the general engine (a Perceval processor with TD components, a
multi-rail machine); `LoopRegisterKernel` is the ORCA-type special case: one rail, loops of
delay l_i closed by programmable beam splitters (theta, phi) on (bin, head slot of loop i),
losses per round trip and per detector.

Program ops (one list per period; `("slot", k)` means the head slot of delay line k at the
current period, an int a rail wire):
    ("gate", a, b, g)   two-mode gate g (index into the gates tensor [n_gates, T, 2, 2]) on (a, b)
    ("loss", a, name)   amplitude damping on mode a with efficiency loss[name] (call-time dict)
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from math import comb

import numpy as np
import torch

from .basis import RegisterBasis
from .circuit import TimeLoopCircuit


def fock_blocks(V: torch.Tensor, N: int) -> torch.Tensor:
    """Fock representation of two-mode unitaries V [..., 2, 2] (mode 0 = a, mode 1 = b):
    [..., N+1, N+1, N+1] with out[..., k, p', p] = <p', k-p'| V |p, k-p> in the top-left corner
    of block k (rest zero: never acts, padded register slots are zero). Recursive in k
    (V a† V† = V00 a† + V10 b†), differentiable in V.
    """
    lead = V.shape[:-2]
    dt, dev = V.dtype, V.device
    out = torch.zeros((*lead, N + 1, N + 1, N + 1), dtype=dt, device=dev)
    B = torch.ones((*lead, 1, 1), dtype=dt, device=dev)
    out[..., 0, 0, 0] = 1.0
    v00, v01, v10, v11 = (V[..., i, j][..., None, None] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    for k in range(1, N + 1):
        ar = torch.arange(k, dtype=torch.float64, device=dev)
        RA = torch.nn.functional.pad(torch.sqrt(ar + 1)[:, None] * B, (0, 0, 1, 0))     # a† : row p' -> p'+1
        RB = torch.nn.functional.pad(torch.sqrt(k - ar)[:, None] * B, (0, 0, 0, 1))     # b† : slot count +1
        cols = (v00 * RA + v10 * RB) / torch.sqrt(ar + 1)[None, :]                       # inputs p = 1..k
        col0 = (v01 * RA + v11 * RB)[..., :, :1] / math.sqrt(k)                          # input p = 0
        B = torch.cat([col0, cols], -1)                                                  # [..., k+1, k+1]
        out[..., k, :k + 1, :k + 1] = B
    return out


def two_mode_generator(V: torch.Tensor) -> torch.Tensor:
    """H [..., 2, 2] with V = exp(-i h) for unitary V (eigendecomposition; h hermitian)."""
    lam, W = torch.linalg.eig(V)
    logl = torch.log(lam)
    h = 1j * (W * logl[..., None, :]) @ torch.linalg.inv(W)
    return 0.5 * (h + h.transpose(-1, -2).conj())


def fock_blocks_exp(h: torch.Tensor, N: int) -> torch.Tensor:
    """Same blocks as `fock_blocks` from the generator h (V = exp(-i h)): one batched
    `matrix_exp` of the padded second-quantised generators J_k = h_aa n_a + h_bb n_b +
    h_ab a†b + h_ba ab† (identity in the padding, harmless). Faster than the recursion on a
    GPU at large N (one launch instead of ~12 N).
    """
    dev = h.device
    Np1 = N + 1
    k = torch.arange(Np1, dtype=torch.float64, device=dev)[:, None]
    p = torch.arange(Np1, dtype=torch.float64, device=dev)[None, :]
    valid = (p <= k).to(h.dtype)
    off = (torch.sqrt((p + 1) * (k - p).clamp(min=0)) * (p + 1 <= k)).to(h.dtype)      # (p+1, p) magnitude
    diag = (h[..., 0, 0][..., None, None] * p + h[..., 1, 1][..., None, None] * (k - p)) * valid
    J = torch.diag_embed(diag) \
        + torch.diag_embed((h[..., 0, 1][..., None, None] * off)[..., :-1], offset=-1) \
        + torch.diag_embed((h[..., 1, 0][..., None, None] * off)[..., :-1], offset=1)
    return torch.linalg.matrix_exp(-1j * J)


class _Blocks:
    """Lazily computed padded Fock blocks of the gates: blocks(g, t) -> [N+1, N+1, N+1].
    From the gate matrices V [n_gates, T, 2, 2] (recursion) or their generators h (matrix_exp,
    chosen on CUDA for N >= EXP_MIN_N).
    """
    EXP_MIN_N = 24

    def __init__(self, N: int, V: torch.Tensor | None = None, h: torch.Tensor | None = None):
        self.N, self.V, self.h = N, V, h
        ref = V if V is not None else h
        self.cache: dict[int, torch.Tensor] = {}
        self.keep_all = ref.shape[0] * ref.shape[1] * (N + 1) ** 3 * 16 <= 2 ** 30
        self.use_exp = ref.is_cuda and N >= self.EXP_MIN_N

    def step(self, t: int) -> torch.Tensor:
        if t in self.cache:
            return self.cache[t]
        if self.use_exp:
            h = self.h[:, t] if self.h is not None else two_mode_generator(self.V[:, t])
            B = fock_blocks_exp(h, self.N)
        else:
            V = self.V[:, t] if self.V is not None else torch.linalg.matrix_exp(-1j * self.h[:, t])
            B = fock_blocks(V, self.N)
        if self.keep_all:
            self.cache[t] = B
        return B

    def __call__(self, g: int, t: int) -> torch.Tensor:
        return self.step(t)[g]


class _ExitRows:
    """The [Q, D] exit gather table of a PeriodicNetworkKernel, built row by row on first use (cached): a multi-rail
    register has Q = C(N + n_rail, n_rail) rail patterns, and the eager [Q, D] table outgrows the device long before
    the state does (2 rails, L = 1, N = 121: 7 503 x 310 124 int64 = 18.6 GB). Indexing as the eager tensor: an int
    gives the row [D], a long tensor of pattern ids [...] gives [..., D].
    """

    def __init__(self, kernel):
        self.k = kernel
        self.rows: dict[int, torch.Tensor] = {}

    def _row(self, q: int) -> torch.Tensor:
        if q not in self.rows:
            self.rows[q] = self.k._gather_index(self.k.basis.exit_map(q))
        return self.rows[q]

    def __getitem__(self, q):
        if isinstance(q, int):
            return self._row(q)
        uq = q.unique()
        table = torch.stack([self._row(int(x)) for x in uq.tolist()])                    # [U, D]
        return table[torch.searchsorted(uq, q.contiguous())]


class PeriodicNetworkKernel:
    """General native engine for a periodic linear-optical network with delay lines.

    Marches a `RegisterBasis` state vector [..., D] through T periods: inject the period's input photons on the rail
    wires, apply the period's program (two-mode gates as block-diagonal Fock matrices, single-mode losses, optional
    emitter events), then exit the rail wires. Everything is torch gathers and batched matmuls on `device`, so the
    amplitude, probability and marginal entry points are differentiable in the gate tensors.
    """

    def __init__(self, n_rail: int, delays: Sequence[int], program: Sequence[tuple], T: int,
                 inputs, n_gates: int | None = None, strides: Sequence[int] | None = None,
                 emitter: dict | None = None,
                 device: str = "cpu", dtype: torch.dtype = torch.complex128, n_max: int | None = None):
        """inputs: [T, n_rail] photons injected on each rail wire at each period (or [T] when
        n_rail == 1). strides: head-slot advance per period of each delay line (default 1; a
        period of b bins on a 1-bin-clocked loop has stride b and slot specs ("slot", k, j)).
        emitter: a spin emitter feeding photons into the rails —
            dict(ops [n_em, Lab, chi, chi], periods [n_em], rails [n_em, Lab], s0 [chi],
                 final [chi] or None)
        emission k happens at the start of period periods[k]: psi -> sum_l ops[k, l] (x)
        a†_{rails[k, l]} psi (labels may share a rail). `final` projects the spin at the end
        (post-selection; `amplitudes` and rejection sampling), None traces it out.
        n_max: photon cap of the register basis (default: every photon the input and the emitter
        can put in it, i.e. exact). Below that, components that would exceed the cap are dropped
        at injection / emission and the state becomes sub-normalised (the truncated register of
        `timeloop.hybrid`; gates and exits conserve or lower the count, so nothing else changes).
        """
        self.n_rail, self.T = int(n_rail), int(T)
        self.delays = tuple(int(d) for d in delays)
        self.L = sum(self.delays)
        self.strides = tuple(int(x) for x in strides) if strides is not None else (1,) * len(self.delays)
        self.offsets = np.concatenate([[0], np.cumsum(self.delays)[:-1]]).astype(int) if self.delays else np.zeros(0, int)
        inp = np.asarray(inputs, dtype=np.int64).reshape(self.T, self.n_rail)
        self.inputs = inp
        self.emitter = emitter
        n_em = 0 if emitter is None else int(np.asarray(emitter["ops"]).shape[0])
        self.N = int(inp.sum()) + n_em if n_max is None else int(n_max)
        self.program = list(program)
        for op in self.program:
            if op[0] not in ("gate", "loss"):
                raise ValueError(f"unknown program op {op!r}")
        self.n_gates = n_gates if n_gates is not None else (1 + max([op[3] for op in self.program if op[0] == "gate"], default=-1))
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.basis = RegisterBasis(self.L, self.N, self.n_rail)
        self.D, self.Q = self.basis.D, self.basis.Q
        dev = self.device
        self._pair: dict[tuple[int, int], tuple] = {}
        self._rail_of = torch.as_tensor(self.basis.rail_of, device=dev)
        self._input_q = [int(q) for q in self.basis.pattern_id(inp)]
        self._inject = {q: self._gather_index(self.basis.inject_map(q)) for q in set(self._input_q)}
        self._exit = _ExitRows(self)                                                     # [Q, D], rows built on first use
        self._vac = int(self.basis.index(np.zeros((1, self.basis.R), dtype=np.int64))[0])
        self._loss: dict[int, tuple] = {}
        self._raise: dict[int, tuple] = {}
        # spin emitter (chi = 1, trivial, when absent: the state always carries a spin axis)
        if emitter is None:
            self.chi = 1
            self._s0 = torch.ones(1, dtype=dtype, device=dev)
            self._final = None
            self._emit_at: dict[int, list[int]] = {}
            self._em_ops = self._em_rails = None
        else:
            ops = torch.as_tensor(np.asarray(emitter["ops"], dtype=np.complex128), dtype=dtype, device=dev)
            self.chi = int(ops.shape[2])
            self._em_ops = ops                                                              # [n_em, Lab, chi, chi]
            self._em_rails = np.asarray(emitter["rails"], dtype=np.int64).reshape(n_em, -1)
            self._s0 = torch.as_tensor(np.asarray(emitter["s0"], dtype=np.complex128), dtype=dtype, device=dev)
            fin = emitter.get("final")
            self._final = None if fin is None else torch.as_tensor(np.asarray(fin, dtype=np.complex128), dtype=dtype, device=dev)
            self._emit_at = {}
            for k, t in enumerate(emitter["periods"]):
                self._emit_at.setdefault(int(t), []).append(k)

    # ---- tables ----------------------------------------------------------------------------
    def _gather_index(self, m: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.where(m < 0, self.D, m), device=self.device)

    def _pair_tables(self, a: int, b: int):
        """Per pair: padded tables (member [G, N+1] with D for missing, gk [G], inv [D]) for the
        fused path and compact per-k tables (member_k [G_k (k+1)] flat, inv_c [D]) for the
        per-k path (no padding: the concatenation of the k-blocks has exactly D entries).
        """
        if (a, b) not in self._pair:
            member, gk, inv = self.basis.pair_tables(a, b)
            bounds = [int(x) for x in np.searchsorted(gk, np.arange(self.N + 2))]
            mk, inv_c, off = [], np.empty(self.basis.D, dtype=np.int64), 0
            for k in range(self.N + 1):
                blk = member[bounds[k]:bounds[k + 1], :k + 1]                     # all valid
                flat = blk.reshape(-1)
                inv_c[flat] = off + np.arange(flat.size)
                mk.append(torch.as_tensor(flat, device=self.device))
                off += flat.size
            G = member.shape[0]
            fused = G * (self.N + 1) <= 2 * self.basis.D                          # padding waste <= 2x
            self._pair[(a, b)] = (torch.as_tensor(np.where(member < 0, self.D, member), device=self.device) if fused else None,
                                  bounds, torch.as_tensor(inv, device=self.device) if fused else None,
                                  torch.as_tensor(gk, device=self.device) if fused else None,
                                  mk, torch.as_tensor(inv_c, device=self.device), fused)
        return self._pair[(a, b)]

    def _loss_tables(self, mode: int):
        if mode not in self._loss:
            src, coef = self.basis.loss_tables(mode)
            self._loss[mode] = (self._gather_index(src), torch.as_tensor(coef, dtype=self.rdtype, device=self.device),
                                torch.as_tensor(self.basis.occ[:, mode], dtype=self.rdtype, device=self.device))
        return self._loss[mode]

    def _raise_tables(self, mode: int):
        """a† on mode w as a gather: (a† psi)[i] = sqrt(n_w(i)) psi[src[i]] (src: one photon fewer)."""
        if mode not in self._raise:
            tgt = self.basis.occ.copy()
            tgt[:, mode] -= 1
            src = self.basis.index(tgt)
            coef = np.where(src >= 0, np.sqrt(self.basis.occ[:, mode]), 0.0)
            self._raise[mode] = (self._gather_index(src), torch.as_tensor(coef, dtype=self.cdtype, device=self.device))
        return self._raise[mode]

    def mode(self, spec, t: int) -> int:
        """Resolve a program mode spec at period t: int = rail wire, ("slot", k[, j]) = slot
        offset_k + ((t * stride_k + j) mod l_k) of delay line k (j = 0: its head).
        """
        if isinstance(spec, tuple):
            k = spec[1]
            j = spec[2] if len(spec) > 2 else 0
            return self.n_rail + int(self.offsets[k]) + ((t * self.strides[k] + j) % self.delays[k])
        return int(spec)

    def dimension(self) -> int:
        """Register basis dimension D — the length of the state vector carried through the periods."""
        return self.D

    def blocks(self, gates=None, generators=None) -> _Blocks:
        """gates: [n_gates, T, 2, 2] two-mode unitaries (or broadcastable); or generators h with
        gate = exp(-i h).
        """
        if isinstance(gates, _GenSpec):
            generators, gates = gates.h, None
        if generators is not None:
            h = torch.as_tensor(generators).to(dtype=self.cdtype, device=self.device).expand(self.n_gates, self.T, 2, 2)
            return _Blocks(self.N, h=h)
        V = torch.as_tensor(gates).to(dtype=self.cdtype, device=self.device).expand(self.n_gates, self.T, 2, 2)
        return _Blocks(self.N, V=V)

    # ---- elementary operations on psi [..., D] ---------------------------------------------
    def _ext(self, psi: torch.Tensor) -> torch.Tensor:
        return torch.cat([psi, torch.zeros((*psi.shape[:-1], 1), dtype=psi.dtype, device=psi.device)], -1)

    def _apply_gate(self, psi: torch.Tensor, a: int, b: int, B: torch.Tensor) -> torch.Tensor:
        """Psi' = B psi on the pair (a, b); B [N+1, N+1, N+1] padded blocks (index k). Fused
        (one bmm over all groups, padded to N+1) when the padding waste is small, else per-k
        blocks on compact tables (memory = the state itself, N+1 launches).
        """
        member, bounds, inv, gk, mk, inv_c, fused = self._pair_tables(a, b)
        lead = psi.shape[:-1]
        if fused and self.N >= 2:
            blocked = self._ext(psi)[..., member]                                # [..., G, N+1]
            G = member.shape[0]
            Bg = B[gk]                                                           # [G, N+1, N+1]
            x = blocked.reshape(-1, G, self.N + 1).transpose(0, 1)               # [G, batch, N+1]
            new = torch.bmm(x, Bg.transpose(-1, -2)).transpose(0, 1).reshape((*lead, G * (self.N + 1)))
            return new[..., inv]
        parts = []
        for k in range(self.N + 1):
            g0, g1 = bounds[k], bounds[k + 1]
            if g1 == g0:
                continue
            blk = psi[..., mk[k]].reshape((*lead, g1 - g0, k + 1)) @ B[k, :k + 1, :k + 1].transpose(-1, -2)
            parts.append(blk.reshape((*lead, -1)))
        return torch.cat(parts, -1)[..., inv_c]

    # density matrices are [chi, D, chi, D] (ket spin, ket register, bra spin, bra register);
    # `_ket` applies a last-axis operation to the ket register axis, the bra side is the same
    # operation (with conjugated matrices) applied directly on the last axis.
    @staticmethod
    def _ket(fn, rho: torch.Tensor) -> torch.Tensor:
        return fn(rho.movedim(1, -1)).movedim(-1, 1)

    def _apply_gate_rho(self, rho: torch.Tensor, a: int, b: int, B: torch.Tensor) -> torch.Tensor:
        r = self._ket(lambda x: self._apply_gate(x, a, b, B), rho)                 # B rho
        return self._apply_gate(r, a, b, B.conj())                                 # (B rho) B†

    def _inject_psi(self, psi: torch.Tensor, q: int) -> torch.Tensor:
        if q not in self._inject:                                                 # an injection the input never used
            self._inject[q] = self._gather_index(self.basis.inject_map(q))         # (a partial injection, `inputs=`)
        return self._ext(psi)[..., self._inject[q]]

    def _inject_streams(self, psi: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        """Per-stream injection: qs [S] rail-pattern ids, psi [S, chi, D] (the sampler's batch)."""
        out = psi
        for q in qs.unique().tolist():
            mask = qs == q
            out = out.clone() if out is psi else out
            out[mask] = self._inject_psi(psi[mask], q)
        return out

    def _inject_rho(self, rho: torch.Tensor, q: int) -> torch.Tensor:
        r = self._ket(lambda x: self._inject_psi(x, q), rho)
        return self._inject_psi(r, q)

    def _emit(self, psi: torch.Tensor, t: int, conj: bool = False) -> torch.Tensor:
        """Emissions scheduled at period t on psi [..., chi, D]: sum_l M_l (x) a†_{rail l}."""
        for k in self._emit_at.get(t, ()):
            ext = self._ext(psi)
            acc = 0
            for l in range(self._em_rails.shape[1]):
                src, coef = self._raise_tables(int(self._em_rails[k, l]))
                x = ext[..., src] * coef                                            # [..., chi, D]
                M = self._em_ops[k, l]
                if conj:
                    M = M.conj()
                acc = acc + torch.einsum("ab,...bd->...ad", M, x)
            psi = acc
        return psi

    def _emit_rho(self, rho: torch.Tensor, t: int) -> torch.Tensor:
        if t not in self._emit_at:
            return rho
        r = self._emit(rho.permute(2, 3, 0, 1), t).permute(2, 3, 0, 1)              # ket side
        return self._emit(r, t, conj=True)                                          # bra side

    def _rail_probs(self, psi: torch.Tensor) -> torch.Tensor:
        """[..., Q] probability of each rail pattern (spin summed)."""
        w = (psi.real ** 2 + psi.imag ** 2).sum(-2)
        out = torch.zeros((*w.shape[:-1], self.Q), dtype=self.rdtype, device=psi.device)
        return out.index_add(-1, self._rail_of, w)

    def _project(self, psi: torch.Tensor, q) -> torch.Tensor:
        """Keep the components with rail pattern q, clear the rail. q: int, or a tensor of
        per-row pattern ids shaped like psi.shape[:-2] (psi [..., chi, D]).
        """
        ext = self._ext(psi)
        if isinstance(q, int):
            return ext[..., self._exit[q]]
        idx = self._exit[q].unsqueeze(-2).expand(*q.shape, ext.shape[-2], self.D)
        return ext.gather(-1, idx)

    def _trace_rail(self, rho: torch.Tensor):
        """(pattern probabilities [Q], rho with the rail traced out and cleared)."""
        d = torch.einsum("sisi->i", rho).real
        p = torch.zeros((self.Q,), dtype=self.rdtype, device=rho.device).index_add(0, self._rail_of, d)
        out = 0
        for q in range(self.Q):
            r = self._ket(lambda x, q=q: self._project(x, q), rho)
            out = out + self._project(r, q)
        return p, out

    # ---- losses ----------------------------------------------------------------------------
    def _kraus_amp(self, mode: int, eta: float, device):
        src, coef, nw = self._loss_tables(mode)
        eta_t = torch.as_tensor(eta, dtype=self.rdtype, device=device)
        j = torch.arange(self.N + 1, dtype=self.rdtype, device=device)
        return src, coef * eta_t ** (nw / 2)[None, :] * (1 - eta_t) ** (j / 2)[:, None]    # [N+1, D]

    def _kraus_all(self, psi: torch.Tensor, mode: int, eta: float) -> torch.Tensor:
        """Stack of E_j psi, j = 0..N: [N+1, ..., D]."""
        src, amp = self._kraus_amp(mode, eta, psi.device)
        ext = self._ext(psi)
        shape = (self.N + 1,) + (1,) * (psi.dim() - 1) + (self.D,)
        return ext[..., src].movedim(-2, 0) * amp.reshape(shape) if psi.dim() > 1 else ext[src] * amp

    def _loss_rho(self, rho: torch.Tensor, mode: int, eta: float) -> torch.Tensor:
        src, amp = self._kraus_amp(mode, eta, rho.device)
        amp = amp.to(self.cdtype)
        out = 0
        for jj in range(self.N + 1):
            idx, a = src[jj], amp[jj]
            r = self._ket(lambda x, idx=idx, a=a: self._ext(x)[..., idx] * a, rho)
            out = out + self._ext(r)[..., idx] * a
        return out

    def _loss_sample(self, psi: torch.Tensor, mode: int, eta: float, generator) -> torch.Tensor:
        """Monte-Carlo jump: pick j with probability ||E_j psi||^2 / ||psi||^2 (psi [S, chi, D])."""
        stack = self._kraus_all(psi, mode, eta)                                   # [N+1, S, chi, D]
        w = (stack.real ** 2 + stack.imag ** 2).sum((-1, -2)).transpose(0, 1)    # [S, N+1]
        j = torch.multinomial(w / w.sum(-1, keepdim=True), 1, generator=generator)[:, 0]
        return stack[j, torch.arange(psi.shape[0], device=psi.device)]

    @staticmethod
    def _eta(loss: dict | None, name) -> float | None:
        if not loss or name not in loss or loss[name] is None or float(loss[name]) >= 1.0:
            return None
        return float(loss[name])

    # ---- one period ------------------------------------------------------------------------
    def _period(self, state, t: int, blocks: _Blocks, loss: dict | None, mode: str, generator=None):
        """Apply period t's program to a pure state ([..., D], mode "psi" or "sample") or a
        density matrix ([D, D], mode "rho").
        """
        for op in self.program:
            if op[0] == "gate":
                a, b = self.mode(op[1], t), self.mode(op[2], t)
                B = blocks(op[3], t)
                state = self._apply_gate_rho(state, a, b, B) if mode == "rho" else self._apply_gate(state, a, b, B)
            else:
                eta = self._eta(loss, op[2])
                if eta is None:
                    continue
                a = self.mode(op[1], t)
                if mode == "rho":
                    state = self._loss_rho(state, a, eta)
                elif mode == "sample":
                    state = self._loss_sample(state, a, eta, generator)
                else:
                    raise ValueError("losses need the density-matrix (period_marginals) or sampling path")
        return state

    # ---- public API ------------------------------------------------------------------------
    def _init_psi(self, batch=()) -> torch.Tensor:
        """[*batch, chi, D]: spin s0, register in vacuum."""
        psi = torch.zeros((*tuple(batch), self.chi, self.D), dtype=self.cdtype, device=self.device)
        psi[..., :, self._vac] = self._s0
        return psi

    def _finish(self, psi: torch.Tensor) -> torch.Tensor:
        """Project the spin on `final` ([..., chi, D] -> [..., D]) or keep it (traced later)."""
        if self._final is None:
            return psi
        return torch.einsum("a,...ad->...d", self._final.conj(), psi)

    @staticmethod
    def _norm2(psi: torch.Tensor, keep: int = 1) -> torch.Tensor:
        """||psi||^2 over all axes after the first `keep` ones."""
        w = psi.real ** 2 + psi.imag ** 2
        return w.reshape((*w.shape[:keep], -1)).sum(-1)

    def _pattern_ids(self, patterns) -> torch.Tensor:
        pat = np.asarray(patterns, dtype=np.int64).reshape(-1, self.T, self.n_rail)
        q = self.basis.pattern_id(pat)                                           # [P, T]
        if (q < 0).any():
            raise ValueError("pattern with more than N photons in a period")
        return torch.as_tensor(q, dtype=torch.long, device=self.device)

    def amplitudes(self, patterns, gates) -> torch.Tensor:
        """Residual register amplitudes for output patterns [P, T, n_rail] ([P, T] when
        n_rail == 1): [P, D] with support on the states whose rail is empty; the leftover
        delay-line photons are the slot occupations. probability = ||row||^2. Differentiable
        in the gates.
        """
        q = self._pattern_ids(patterns)
        blocks = self.blocks(gates)
        psi = self._init_psi((q.shape[0],))
        for t in range(self.T):
            psi = self._inject_psi(psi, self._input_q[t])
            psi = self._emit(psi, t)
            psi = self._period(psi, t, blocks, None, "psi")
            psi = self._project(psi, q[:, t])
        psi = self._finish(psi)
        return psi[:, 0] if self.chi == 1 else psi

    def probabilities(self, patterns, gates) -> torch.Tensor:
        """[P] probabilities of the output patterns (leftover delay-line photons marginalised;
        a traced spin summed, a projected spin's weight included).
        """
        a = PeriodicNetworkKernel.amplitudes(self, patterns, gates)
        return self._norm2(a)

    def distribution(self, gates, tol: float = 1e-14):
        """All output patterns with probability > tol: (patterns [P, T, n_rail] numpy — [P, T]
        when n_rail == 1 —, probs [P]). Branches the pure state on every period — exponential,
        for small N only.
        """
        blocks = self.blocks(gates)
        with torch.no_grad():
            psi = self._init_psi((1,))
            qs = torch.zeros((1, 0), dtype=torch.long, device=self.device)
            for t in range(self.T):
                psi = self._inject_psi(psi, self._input_q[t])
                psi = self._emit(psi, t)
                psi = self._period(psi, t, blocks, None, "psi")
                cand, cq = [], []
                for q in range(self.Q):
                    c = self._project(psi, q)
                    keep = self._norm2(c) > tol
                    if keep.any():
                        cand.append(c[keep])
                        cq.append(torch.cat([qs[keep], torch.full((int(keep.sum()), 1), q, dtype=torch.long, device=self.device)], 1))
                psi, qs = torch.cat(cand), torch.cat(cq)
            probs = self._norm2(self._finish(psi))
        pats = self.basis.rail_patterns[qs.cpu().numpy()]                        # [P, T, n_rail]
        if self.n_rail == 1:
            pats = pats[..., 0]
        return pats, probs

    def period_marginals(self, gates, loss: dict | None = None, return_register: bool = False, inputs=None):
        """Distribution of the rail pattern leaving at every period, [T, Q] (Q = N+1 photon
        counts when n_rail == 1; `basis.rail_patterns` lists the patterns otherwise), by marching
        the register density matrix (D x D). Differentiable in the gates. With return_register
        the final register density matrix (leftover photons) is returned too. `inputs` [T]:
        rail-pattern ids injected per period instead of the constructor's input (a subset of
        the photons — the interfering ones of a partially distinguishable input).
        """
        if self._final is not None:
            raise NotImplementedError("period marginals with a projected spin (post-selection) need the "
                                      "backward (Heisenberg) sweep; use final=None or amplitudes()")
        blocks = self.blocks(gates)
        psi = self._init_psi()                                                    # [chi, D]
        rho = torch.einsum("ad,be->adbe", psi, psi.conj())                        # [chi, D, chi, D]
        out = []
        inj = self._input_q if inputs is None else [int(q) for q in inputs]
        for t in range(self.T):
            rho = self._inject_rho(rho, inj[t])
            rho = self._emit_rho(rho, t)
            rho = self._period(rho, t, blocks, loss, "rho")
            p, rho = self._trace_rail(rho)
            out.append(p)
        marg = torch.stack(out)
        if return_register:
            return marg, (rho[0, :, 0, :] if self.chi == 1 else rho)
        return marg

    def sample(self, shots: int, gates, generator: torch.Generator | None = None,
               loss: dict | None = None, return_register: bool = False, threshold: bool = False,
               inputs: torch.Tensor | None = None) -> torch.Tensor:
        """Exact samples [shots, T, n_rail] ([shots, T] when n_rail == 1) of photon counts per
        rail wire and period (quantum trajectories: the rail is measured as it exits; losses as
        Monte-Carlo jumps). threshold=True clips counts to 0/1 (threshold detectors). With
        return_register the per-shot residual register states [shots, (chi,) D] are returned
        too. A projected spin (emitter final) is rejection-sampled: fewer than `shots` rows
        come back (the acceptance rate is the projection probability). `inputs` [shots, T]
        (long): per-stream rail-pattern ids injected at every period instead of the
        constructor's input — the interfering photons of a stream under partial
        distinguishability (`LoopRegisterKernel.sample(indist=)`).
        """
        with torch.no_grad():
            blocks = self.blocks(gates)
            psi = self._init_psi((shots,))
            qs = torch.zeros((shots, self.T), dtype=torch.long, device=self.device)
            for t in range(self.T):
                psi = self._inject_psi(psi, self._input_q[t]) if inputs is None else self._inject_streams(psi, inputs[:, t].to(self.device))
                psi = self._emit(psi, t)
                psi = self._period(psi, t, blocks, loss, "sample", generator)
                p = self._rail_probs(psi)
                q = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=generator)[:, 0]
                qs[:, t] = q
                psi = self._project(psi, q)
                psi = psi / torch.sqrt(self._norm2(psi))[:, None, None]
            if self._final is not None:
                acc = self._norm2(self._finish(psi))                                     # projection probability
                keep = torch.rand(shots, generator=generator, device=acc.device, dtype=acc.dtype) < acc
                qs, psi = qs[keep], psi[keep]
            pats = torch.as_tensor(self.basis.rail_patterns, device=self.device)[qs]     # [S, T, n_rail]
            if self.n_rail == 1:
                pats = pats[..., 0]
            if threshold:
                pats = pats.clamp(max=1)
            if self.chi == 1:
                psi = psi[:, 0]
        return (pats, psi) if return_register else pats


class _GenSpec:
    """Marker: gates given by their generators (see LoopRegisterKernel._gates)."""

    def __init__(self, h: torch.Tensor):
        self.h = h


class LoopRegisterKernel(PeriodicNetworkKernel):
    """ORCA-type device: one rail, loops of delay l_i closed by programmable beam splitters
    G(theta_{i,t}, phi_{i,t}) on (bin, head slot of loop i), loops in series. Public methods take
    (theta, phi) [n_loops, T] (or explicit gates [n_loops, T, 2, 2], see
    TimeLoopCircuit.gate_matrices) and `loss=dict(loop=[eta_i, ...], detector=eta_d)`:
    amplitude damping on the head slot before each beam splitter (one round trip in loop i
    survives with probability eta_i) and on the bin before detection. A photon circulating k
    times in loop i pays eta_i^k, so loop loss is path-dependent and must run on the register;
    `loss=eta` (scalar, or the `uniform` key of the dict) is the uniform end-to-end part
    (brightness, transmission, detector), applied by output thinning at no register cost
    (docs/LOSS.md) and folded into the detector efficiency when loop channels are present.
    """

    def __init__(self, circuit: TimeLoopCircuit, input_occupation: Sequence[int],
                 device: str = "cpu", dtype: torch.dtype = torch.complex128,
                 emitter=None, bins_per_period: int | None = None):
        """emitter: an EmitterMPS whose modes are bins of this circuit (photon k, label l goes to
        bin emitter.modes[k][l]); all labels of one emission must fall in the same period, so
        the period is `bins_per_period` bins (default: the span of the label bins, 2 for the
        dual-rail caterpillar (2k, 2k+1)); T must be a multiple of it. Classical photons
        (`input_occupation`, one entry per bin) can be combined with the emitter.
        """
        if len(input_occupation) != circuit.T:
            raise ValueError(f"input_occupation must have T={circuit.T} entries (photons per bin)")
        self.circuit = circuit
        bpp = 1
        em = None
        if emitter is not None:
            from ..sequence import ModeBinEmitter
            if isinstance(emitter, ModeBinEmitter):
                raise NotImplementedError("internal modes: not supported through loops")
            emitter.require_uniform_labels("LoopRegisterKernel(emitter=)")
            modes = np.asarray(emitter.modes, dtype=np.int64)                                # [n_em, Lab]
            if bins_per_period is None:
                bins_per_period = max(1, int((modes.max(1) - modes.min(1)).max()) + 1)
            bpp = int(bins_per_period)
            if circuit.T % bpp:
                raise ValueError(f"T={circuit.T} must be a multiple of bins_per_period={bpp}")
            periods = modes // bpp
            if (periods != periods[:, :1]).any():
                raise ValueError("all labels of an emission must emit in the same period (bins_per_period too small)")
            em = {"ops": emitter.ops, "periods": periods[:, 0], "rails": modes % bpp, "s0": emitter.s0, "final": emitter.final}
        elif bins_per_period is not None:
            bpp = int(bins_per_period)
            if circuit.T % bpp:
                raise ValueError(f"T={circuit.T} must be a multiple of bins_per_period={bpp}")
        self.bpp = bpp
        self.T_bins = circuit.T
        program = []
        for r in range(bpp):
            for i in range(circuit.n_loops):
                program.append(("loss", ("slot", i, r), f"loop{i}"))
                program.append(("gate", r, ("slot", i, r), i * bpp + r))
            program.append(("loss", r, "detector"))
        inputs = np.asarray(input_occupation, dtype=np.int64).reshape(circuit.T // bpp, bpp)
        super().__init__(bpp, circuit.loops, program, circuit.T // bpp, inputs,
                         n_gates=circuit.n_loops * bpp, strides=[bpp] * circuit.n_loops, emitter=em,
                         device=device, dtype=dtype)
        self.input = [int(c) for c in input_occupation]
        # rail pattern -> per-bin photon counts, for bin marginals: [Q, bpp, N+1] one-hot
        oh = np.zeros((self.Q, bpp, self.N + 1))
        for q, pat in enumerate(self.basis.rail_patterns):
            for r in range(bpp):
                oh[q, r, pat[r]] = 1.0
        self._bin_onehot = torch.as_tensor(oh, dtype=self.rdtype, device=self.device)

    def _split_gates(self, gates: torch.Tensor) -> torch.Tensor:
        """[n_loops, T_bins, ...] -> [n_loops * bpp, T_periods, ...] with gate i*bpp + r at period
        t = gate i at bin t*bpp + r.
        """
        n, Tb = gates.shape[0], gates.shape[1]
        g = gates.reshape(n, Tb // self.bpp, self.bpp, *gates.shape[2:]).transpose(1, 2)     # [n, bpp, Tp, ...]
        return g.reshape(n * self.bpp, Tb // self.bpp, *gates.shape[2:])

    def _gates(self, theta, phi):
        """Explicit gate matrices, or a _GenSpec carrying the generators h = theta [[0, e^{-i phi}],
        [e^{i phi}, 0]] (so large-N blocks can use the batched matrix_exp path).
        """
        if isinstance(theta, _GenSpec):
            return theta
        if torch.is_tensor(theta) and theta.dim() >= 2 and theta.shape[-2:] == (2, 2):
            return self._split_gates(self.circuit.gate_matrices(theta, phi, self.cdtype, self.device))
        th, ph = self.circuit._params(theta, 0.0 if phi is None else phi, self.cdtype, self.device)
        e = torch.exp(-1j * ph).to(self.cdtype)
        z = torch.zeros_like(e)
        h = th.to(self.cdtype)[..., None, None] * torch.stack([torch.stack([z, e], -1), torch.stack([e.conj(), z], -1)], -2)
        return _GenSpec(self._split_gates(h))

    @staticmethod
    def _loss_dict(loss: dict | None, n_loops: int) -> dict | None:
        if not loss:
            return None
        loop = loss.get("loop")
        if loop is None:
            loop = [None] * n_loops
        elif np.isscalar(loop):
            loop = [float(loop)] * n_loops
        out = {f"loop{i}": e for i, e in enumerate(loop)}
        det, uni = loss.get("detector"), loss.get("uniform")
        if uni is not None:                       # uniform loss commutes with everything: fold it into the detector
            det = float(uni) * (1.0 if det is None else float(det))
        out["detector"] = det
        return out

    def bs_blocks(self, theta, phi=None) -> _Blocks:
        """Fock blocks of the loop beam splitters for settings theta/phi ([n_loops, T], or broadcastable to it)."""
        return self.blocks(self._gates(theta, phi))

    def amplitudes(self, patterns, theta, phi=None) -> torch.Tensor:
        """Residual amplitudes for bin patterns [P, T_bins] (see PeriodicNetworkKernel.amplitudes)."""
        return super().amplitudes(np.asarray(patterns).reshape(-1, self.T, self.bpp), self._gates(theta, phi))

    def probabilities(self, patterns, theta, phi=None) -> torch.Tensor:           # noqa: D102
        return super().probabilities(np.asarray(patterns).reshape(-1, self.T, self.bpp), self._gates(theta, phi))

    @staticmethod
    def _uniform(loss):
        """A scalar `loss`, or a dict with only a `uniform` key, is the uniform end-to-end
        transmission eta (docs/LOSS.md): applied on the output side by thinning, which costs
        nothing on the register. Returns eta or None (None also when loop / detector channels
        are present: those are path-dependent and run as Kraus maps on the register, with
        `uniform` folded into the detector efficiency by `_loss_dict`).
        """
        if loss is None:
            return None
        if isinstance(loss, dict):
            if any(loss.get(k) is not None for k in ("loop", "detector")):
                return None
            loss = loss.get("uniform")
            if loss is None:
                return None
        e = float(loss)
        return None if e >= 1.0 else e

    def distribution(self, theta, phi=None, tol: float = 1e-14, loss=None):
        """(patterns [P, T_bins], probs [P]); with a uniform `loss=eta` a `LossyDistribution`
        over 0..N detected photons (photonic_engines.imperfections.thin_patterns).
        """
        pats, probs = super().distribution(self._gates(theta, phi), tol=tol)
        pats = pats.reshape(pats.shape[0], self.T_bins)
        eta = self._uniform(loss)
        if eta is None:
            return pats, probs
        from ..imperfections import thin_patterns
        return thin_patterns(pats, probs, eta)

    def bin_marginals(self, theta, phi=None, loss=None, return_register: bool = False, indist=None,
                      mc_shots: int | None = None, generator: torch.Generator | None = None):
        """Photon-number distribution of every output bin, [T_bins, N+1] (see period_marginals).
        loss: dict(loop=..., detector=...) for the Kraus channels on the density matrix, or a
        scalar eta for uniform end-to-end loss (pure-state marginals, then thinned per bin).
        indist=x (x-model): the mixture over the interfering subsets of the N input photons —
        for each subset the register marginals of that injection schedule, convolved bin by bin
        with the independent one-photon marginals of the other photons; enumerated exactly when
        2^N <= 256 (N <= 8), else a Monte-Carlo average over `mc_shots` subsets (default 64; the
        result is then an estimate). Not differentiable in that case; return_register refused.
        """
        from ..imperfections import spec
        d = spec(indist)
        eta = self._uniform(loss)
        chan = None if eta is not None else self._loss_dict(loss, self.circuit.n_loops)
        gates = self._gates(theta, phi)
        if d is not None:
            if return_register:
                raise NotImplementedError("bin_marginals(indist=): no register (a mixture over subsets)")
            if getattr(self, "_em_ops", None) is not None and self._emit_at:
                raise NotImplementedError("partial distinguishability with an emitter-driven loop: classical inputs only")
            if d.x is None:
                raise NotImplementedError("time loops: the x-model only")
            bins = self._photon_bins()
            if 2 ** self.N <= 256:
                subsets = np.array(list(itertools.product([True, False], repeat=self.N)), dtype=bool)   # [2^N, N]
                w = torch.as_tensor([d.x ** g.sum() * (1 - d.x) ** (self.N - g.sum()) for g in subsets], dtype=self.rdtype)
            else:
                if generator is None:
                    generator = torch.Generator().manual_seed(0)
                subsets = (torch.rand((mc_shots or 64, self.N), generator=generator) < d.x).numpy()
                w = torch.full((subsets.shape[0],), 1.0 / subsets.shape[0], dtype=self.rdtype)
            sched = self._schedule(subsets)                                                     # [K, T]
            singles = {b: self._single_kernel(b).bin_marginals(theta, phi, loss=chan) for b in set(bins)}   # [T_bins, 2]
            total = torch.zeros((self.T_bins, self.N + 1), dtype=self.rdtype, device=self.device)
            for good, wk, sch in zip(subsets, w, sched):
                if good.any():
                    m = super().period_marginals(gates, loss=chan, inputs=sch)
                    m = torch.einsum("tq,qrn->trn", m, self._bin_onehot).reshape(self.T_bins, self.N + 1)
                else:
                    m = torch.zeros((self.T_bins, self.N + 1), dtype=self.rdtype, device=self.device); m[:, 0] = 1
                for j in range(self.N):
                    if not good[j]:
                        one = singles[bins[j]].to(m.device)                                  # [T_bins, 2]
                        m = m * one[:, :1] + torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1) * one[:, 1:2]
                total = total + wk * m
            if eta is not None:
                from ..imperfections import thin_marginals
                total = thin_marginals(total, eta)
            return total
        out = super().period_marginals(gates, loss=chan, return_register=return_register)
        marg, rho = out if return_register else (out, None)
        marg = torch.einsum("tq,qrn->trn", marg, self._bin_onehot).reshape(self.T_bins, self.N + 1)
        if eta is not None:
            from ..imperfections import thin_marginals
            marg = thin_marginals(marg, eta)
        return (marg, rho) if return_register else marg

    # ---- partial distinguishability (docs/DISTINGUISHABILITY.md §3.6) --------------------------
    def _photon_bins(self) -> list[int]:
        """The input photons as a list of bins (with multiplicity), the order `draw_roles` uses."""
        return [b for b, c in enumerate(self.input) for _ in range(int(c))]

    def _single_kernel(self, b: int) -> LoopRegisterKernel:
        """The one-photon register of a photon in a mode of its own, injected at bin b: it
        interferes with nothing else but still with itself across the delay lines, so it is a
        one-photon evolution through the same gates and loss channels, not a random walk. Cached
        per bin.
        """
        cache = self.__dict__.setdefault("_singles", {})
        if b not in cache:
            occ = [0] * self.T_bins; occ[b] = 1
            cache[b] = LoopRegisterKernel(self.circuit, occ, device=str(self.device), dtype=self.cdtype, bins_per_period=self.bpp)
        return cache[b]

    def _schedule(self, good: np.ndarray) -> np.ndarray:
        """Rail-pattern ids [.., T_periods] of the injection schedule that keeps the photons
        flagged in `good` [.., N] (bool, photon order of `_photon_bins`).
        """
        bins = np.asarray(self._photon_bins())
        lead = good.shape[:-1]
        counts = np.zeros((*lead, self.T_bins), dtype=np.int64)
        for j, b in enumerate(bins):
            counts[..., b] += good[..., j]
        return self.basis.pattern_id(counts.reshape(*lead, self.T, self.bpp))

    def sample(self, shots: int, theta, phi=None, generator: torch.Generator | None = None,
               loss=None, return_register: bool = False, threshold: bool = False, indist=None):
        """Exact samples [shots, T_bins] of photon counts per output bin (see PeriodicNetworkKernel.sample).
        loss: dict for the Monte-Carlo jump channels, or a scalar eta for uniform loss (lossless
        trajectories, counts thinned per shot; the returned register states are then those of
        the lossless trajectories). indist=x (x-model, docs/DISTINGUISHABILITY.md §3.6): each
        input photon is, independently, in a mode of its own with probability 1 - x; the
        interfering photons of a shot run the register together (a per-stream injection
        schedule), every other photon runs its own one-photon register through the same gates
        and channels, and the counts add. Classical (Fock) inputs only.
        """
        from ..imperfections import spec
        d = spec(indist)
        eta = self._uniform(loss)
        chan = None if eta is not None else self._loss_dict(loss, self.circuit.n_loops)
        gates = self._gates(theta, phi)
        inputs = None
        if d is not None:
            if getattr(self, "_em_ops", None) is not None and self._emit_at:
                raise NotImplementedError("partial distinguishability with an emitter-driven loop: classical inputs only")
            if d.x is None:
                raise NotImplementedError("time loops: the x-model only")
            if generator is None:
                generator = torch.Generator().manual_seed(0)
            good = (torch.rand((shots, self.N), generator=generator) < d.x).numpy()      # [shots, N] interfering photons
            inputs = torch.as_tensor(self._schedule(good), dtype=torch.long)
        out = super().sample(shots, gates, generator=generator, loss=chan,
                             return_register=return_register, threshold=threshold and eta is None and d is None, inputs=inputs)
        pats, psi = out if return_register else (out, None)
        pats = pats.reshape(pats.shape[0], self.T_bins)
        if d is not None:
            bins = self._photon_bins()
            for j in range(self.N):                                                # the photons in a mode of their own
                ids = torch.as_tensor(np.nonzero(~good[:, j])[0], dtype=torch.long)
                if ids.numel() == 0:
                    continue
                one = self._single_kernel(bins[j]).sample(int(ids.numel()), theta, phi, generator=generator, loss=chan)
                pats[ids.to(pats.device)] += one.to(pats.device, pats.dtype)
            if threshold and eta is None:
                pats = pats.clamp(max=1)
        if eta is not None:
            from ..imperfections import thin_counts
            pats = thin_counts(pats, eta, generator=generator if generator is None or generator.device.type == "cpu" else None)
            if threshold:
                pats = pats.clamp(max=1)
        return (pats, psi) if return_register else pats


def register_dimension(N: int, L: int, n_rail: int = 1) -> int:
    """Size of the register basis for N photons over n_rail rail wires and L delay-line slots, without building it."""
    return comb(N + L + n_rail, L + n_rail)
