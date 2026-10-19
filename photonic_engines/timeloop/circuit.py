"""Programmable time-bin loop circuits (ORCA PT-series): one spatial rail carrying T time bins,
closed on itself by fibre loops of delay l_i bins through programmable beam splitters.

Wires of the unrolled network: bins 0..T-1, then the loop slots (loop i occupies L_i = l_i
consecutive wires; at time t its head slot is wire T + off_i + (t mod l_i)). At time t the bin
wire t meets the head slot of loop 0, then of loop 1, ... (loops in series along the rail).
The beam splitter is the two-mode gate

    G(theta, phi) = exp(-i theta [[0, e^{-i phi}], [e^{i phi}, 0]])
                  = [[cos theta, -i e^{-i phi} sin theta], [-i e^{i phi} sin theta, cos theta]]

on (bin, slot): theta = 0 lets the bin pass and the loop content circulate, theta = pi/2 swaps
them (the bin photon enters the loop, the loop photon leaves). `unitary(theta, phi)` returns the
[T + L, T + L] matrix (rows = output wires, columns = input wires) as a differentiable torch
function of theta, phi (each [n_loops, T], or broadcastable to it).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


class TimeLoopCircuit:
    """Unrolled geometry of a time-bin loop circuit: T bins on one rail plus fibre loops of given delays.

    Holds the wire layout only — T bin wires followed by L = sum(loops) loop slots, one slot block per loop at
    `offsets[i]` — and builds the [m, m] unrolled unitary from the per-gate (theta, phi) settings.
    """

    def __init__(self, T: int, loops: Sequence[int]):
        if T < 1 or any(int(l) < 1 for l in loops) or len(loops) == 0:
            raise ValueError("TimeLoopCircuit needs T >= 1 and at least one loop of delay >= 1")
        self.T = int(T)
        self.loops: tuple[int, ...] = tuple(int(l) for l in loops)
        self.L = sum(self.loops)
        self.m = self.T + self.L                       # unrolled mode count
        self.offsets: list[int] = []
        off = 0
        for l in self.loops:
            self.offsets.append(off)
            off += l

    @property
    def n_loops(self) -> int:
        """Number of fibre loops."""
        return len(self.loops)

    def slot(self, i: int, t: int) -> int:
        """Index (0..L-1) of the head slot of loop i at time t."""
        return self.offsets[i] + (t % self.loops[i])

    def slot_wire(self, i: int, t: int) -> int:
        """Unrolled wire index (T..T+L-1) of the head slot of loop i at time t."""
        return self.T + self.slot(i, t)

    def gates(self) -> list[tuple[int, int, int]]:
        """Time-ordered list of (t, loop i, slot index) — the gate sequence."""
        return [(t, i, self.slot(i, t)) for t in range(self.T) for i in range(self.n_loops)]

    def _params(self, theta, phi, dtype, device):
        rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        th = torch.as_tensor(theta, dtype=rdtype, device=device).expand(self.n_loops, self.T)
        ph = torch.as_tensor(phi, dtype=rdtype, device=device).expand(self.n_loops, self.T)
        return th, ph

    @staticmethod
    def bs_matrix(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """G(theta, phi) for tensors of parameters: [...] -> [..., 2, 2]."""
        c, s = torch.cos(theta), torch.sin(theta)
        e = torch.exp(1j * phi)
        z = torch.zeros_like(c)
        g = torch.stack([torch.stack([c + 0j, -1j * s / e], -1),
                         torch.stack([-1j * s * e, c + 0j], -1)], -2)
        return g + z[..., None, None]

    def gate_matrices(self, theta, phi=None, dtype: torch.dtype = torch.complex128, device="cpu") -> torch.Tensor:
        """[n_loops, T, 2, 2] two-mode unitaries (bin, slot) of every gate: G(theta, phi), or
        `theta` itself when it is a [n_loops, T, 2, 2] (or broadcastable) tensor of explicit
        unitaries (any U(2) element, e.g. converted from a Perceval circuit).
        """
        if torch.is_tensor(theta) and theta.dim() >= 2 and theta.shape[-2:] == (2, 2):
            return theta.to(dtype=dtype, device=device).expand(self.n_loops, self.T, 2, 2)
        th, ph = self._params(theta, 0.0 if phi is None else phi, dtype, device)
        return self.bs_matrix(th, ph).to(dtype)

    def unitary(self, theta, phi=None, dtype: torch.dtype = torch.complex128, device="cpu") -> torch.Tensor:
        """Unrolled [T+L, T+L] unitary, differentiable in theta and phi (or in explicit gates,
        see gate_matrices).
        """
        G = self.gate_matrices(theta, phi, dtype, device)   # [n_loops, T, 2, 2]
        U = torch.eye(self.m, dtype=dtype, device=device)
        for t, i, s in self.gates():
            idx = torch.tensor([t, self.T + s], device=device)
            rows = G[i, t] @ U[idx]                          # gate acts on the two output wires
            U = U.index_copy(0, idx, rows)
        return U

    def flush_settings(self, theta, phi):
        """Copy of (theta, phi) whose last L steps empty the loops (no photon left in a slot):
        loops are flushed last-first — loop i swaps (theta=pi/2) for l_i steps while every other
        loop is set to bar (theta=0), so the photons it releases reach the detector. The last L
        input bins must be empty. Requires T >= L; the active settings occupy steps 0..T-L-1.
        """
        th, ph = self._params(theta, phi, torch.complex128, "cpu")
        th, ph = th.clone(), ph.clone()
        t0 = self.T - self.L
        if t0 < 0:
            raise ValueError("flush_settings needs T >= L")
        th[:, t0:] = 0.0
        ph[:, t0:] = 0.0
        t = t0
        for i in reversed(range(self.n_loops)):
            th[i, t:t + self.loops[i]] = torch.pi / 2
            t += self.loops[i]
        return th, ph
