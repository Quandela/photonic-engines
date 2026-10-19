"""Fast, GPU-resident, vectorized circuit -> unitary build (SLOScore's #1 measured win).

The benchmark showed the circuit->unitary conversion - not the SLOS kernel - dominates the
backend task (84-98%), and that the existing converters are host-bound Python (merlin's ~5x
slower than dperceval's). SLOScore builds the unitary directly on-device:

  * a *plan* is compiled once (setup): the circuit's 2-mode (BS) and 1-mode (PS) gates are
    greedily packed into parallel layers of non-overlapping (hence commuting) gates;
  * `convert(device)` (the timed step) applies each layer as a single BATCHED 2x2 row update
    on the device - O(m) work per gate, vectorized across a layer, no per-component m x m
    matmul and no host round-trip.

Verified against perceval's `compute_unitary()` in tests/adapters/test_perceval_converter.py.
NOTE on precision: the plan stores gate matrices as complex64 and `convert` builds in complex64,
so the agreement with perceval's complex128 unitary is at single precision (~1e-7), not exact.
There is no dtype argument yet; add one here and in `build_plan` if a caller needs complex128.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class _Layer:
    a: list[int] = field(default_factory=list)      # top modes of pair gates
    b: list[int] = field(default_factory=list)      # bottom modes of pair gates
    g: list[np.ndarray] = field(default_factory=list)  # 2x2 matrices
    s: list[int] = field(default_factory=list)      # single-mode (PS) modes
    ph: list[complex] = field(default_factory=list)  # PS scalars


@dataclass
class ConversionPlan:
    """Compiled circuit schedule: mode count `m` plus the ordered list of layers of commuting gates to replay."""

    m: int
    layers: list[_Layer]


def build_plan(circuit) -> ConversionPlan:
    """Compile a perceval circuit into parallel layers of commuting gates (host-side, once).

    ASAP list-scheduling: each gate is placed in the earliest layer after every layer that
    already touched one of its modes (`1 + max(last_touch[mode])`). Disjoint (commuting) gates
    share a layer, so the schedule reaches the mesh's true parallel depth (~4 x columns), which
    minimizes the number of sequential batched updates - i.e. GPU launches - at run time.
    """
    m = circuit.m
    layers: list[_Layer] = []
    last_touch: list[int] = [-1] * m                # latest layer index that used each mode

    for r, comp in circuit:
        name = type(comp).__name__
        if name == "Barrier":
            continue
        modes = tuple(int(x) for x in r)
        mat = np.array(comp.compute_unitary(), dtype=np.complex64)
        li = 1 + max((last_touch[x] for x in modes), default=-1)
        while len(layers) <= li:
            layers.append(_Layer())
        if len(modes) == 2:
            L = layers[li]
            L.a.append(modes[0]); L.b.append(modes[1]); L.g.append(mat)
        elif len(modes) == 1:
            L = layers[li]
            L.s.append(modes[0]); L.ph.append(complex(mat[0, 0]))
        else:                                       # rare >2-mode block: give it its own layer
            li = len(layers)
            layers.append(_DenseLayer(modes, mat))  # type: ignore
        for x in modes:
            last_touch[x] = li
    return ConversionPlan(m, [L for L in layers if not _empty(L)])


def _empty(layer) -> bool:
    if isinstance(layer, _DenseLayer):
        return False
    return not (layer.a or layer.s)


class _DenseLayer:
    """Fallback for a general k-mode block (not used by the reference MZI mesh)."""
    def __init__(self, modes: tuple[int, ...], mat: np.ndarray):
        self.modes = list(modes)
        self.mat = mat


class CircuitConverter:
    """Applies a compiled plan to produce the m x m unitary on a target device.

    Left-multiplication in circuit order (U <- G_embedded @ U), matching perceval's
    `compute_unitary()` composition. Gate tensors are cached per device.
    """

    def __init__(self, circuit):
        self.plan = build_plan(circuit)
        self.m = self.plan.m
        self._cache: dict = {}                       # device -> prepared tensors

    def _prepare(self, device: torch.device):
        key = str(device)
        if key in self._cache:
            return self._cache[key]
        prepared = []
        for layer in self.plan.layers:
            if isinstance(layer, _DenseLayer):
                prepared.append(("dense",
                                 torch.tensor(layer.modes, dtype=torch.long, device=device),
                                 torch.as_tensor(layer.mat, dtype=torch.complex64, device=device)))
                continue
            item = {"has_pair": bool(layer.a), "has_single": bool(layer.s)}
            if layer.a:
                item["a"] = torch.tensor(layer.a, dtype=torch.long, device=device)
                item["b"] = torch.tensor(layer.b, dtype=torch.long, device=device)
                item["g"] = torch.as_tensor(np.stack(layer.g), dtype=torch.complex64, device=device)
            if layer.s:
                item["s"] = torch.tensor(layer.s, dtype=torch.long, device=device)
                item["ph"] = torch.as_tensor(np.array(layer.ph), dtype=torch.complex64, device=device)
            prepared.append(("layer", item))
        self._cache[key] = prepared
        return prepared

    def convert(self, device: str = "cpu") -> torch.Tensor:
        """Build and return the m x m unitary on `device` (the timed conversion)."""
        dev = torch.device(device)
        prepared = self._prepare(dev)
        U = torch.eye(self.m, dtype=torch.complex64, device=dev)
        for kind, *rest in prepared:
            if kind == "dense":
                modes, mat = rest
                U[modes] = mat @ U[modes]
                continue
            item = rest[0]
            if item["has_pair"]:
                a, b, g = item["a"], item["b"], item["g"]
                ra, rb = U[a], U[b]                              # [K, m]
                U[a] = g[:, 0, 0:1] * ra + g[:, 0, 1:2] * rb
                U[b] = g[:, 1, 0:1] * ra + g[:, 1, 1:2] * rb
            if item["has_single"]:
                s, ph = item["s"], item["ph"]
                U[s] = ph.unsqueeze(1) * U[s]
        return U


class ParametrizedMesh:
    """The TRAINABLE conversion path: U(theta[, x]) built by the same row-update
    pattern as `CircuitConverter`, but with gate entries computed from a LIVE parameter tensor
    inside autograd — so gradients flow through the conversion to the circuit parameters, the
    property a trainable layer (merlin's QuantumLayer contract) requires. `CircuitConverter`
    above stays the as-shipped benchmark converter (constant replay of a fixed circuit).

    Structure — fixed at init, like any neural layer (structure is never differentiated):
    `layers` rows of 2-mode BS(theta) rotations on alternating-offset disjoint pairs (the
    rectangular Reck/Clements-family mesh). Two parameter roles, matching the QuantumLayer
    shape:
      * ``theta`` [L, m//2] — SHARED trainable weights (a leaf tensor; expose via
        ``.parameters()`` to an optimizer);
      * ``x``   [B, m]     — optional PER-SAMPLE angle encoding: a phase screen
        diag(e^{i x_bj}) applied MID-MESH (after floor(L/2) layers), so
        ``U_b = Mesh2(theta) @ diag(e^{i x_b}) @ Mesh1(theta)`` differs OBSERVABLY across
        the batch. PHYSICS CONSTRAINT (caught by the gates): a phase screen BEFORE the mesh
        is unobservable for a Fock input (it contributes only e^{i sum_j s_j x_j} — a global
        phase per basis state), and AFTER the mesh it is equally unobservable in the output
        probabilities — the encoding must sit between interfering layers to change the
        distribution. Requires L >= 2. Gradients flow to theta AND x (through probabilities,
        not just amplitudes — gated in tests/adapters/test_perceval_converter.py).

    Memory note (documented, deliberate): autograd requires a fresh row-updated tensor per
    layer (`U.clone()`), so training-mode conversion holds O(L) layer intermediates of
    [B, m, m] for the backward pass — the price merlin's converter also pays. Checkpointing is
    the known lever if this ever dominates; profile the conversion before reaching for it.
    """

    def __init__(self, m: int, layers: int | None = None, device: str = "cpu",
                 dtype: torch.dtype = torch.complex64, seed: int = 7):
        self.m = m
        self.L = layers if layers is not None else m
        self.device = torch.device(device)
        self.dtype = dtype
        self.pairs = []
        for l in range(self.L):
            off = l % 2
            self.pairs.append([(a, a + 1) for a in range(off, m - 1, 2)])
        g = torch.Generator().manual_seed(seed)
        theta = torch.rand(self.L, max(1, m // 2), generator=g) * math.pi
        self.theta = theta.to(self.device).requires_grad_(True)

    def parameters(self):
        """Return the trainable leaf tensors, i.e. the single shared angle tensor theta [L, m//2]."""
        return [self.theta]

    def unitary(self, x: torch.Tensor = None, batch: int = 1) -> torch.Tensor:
        """[B, m, m] unitary from the live parameters. `x` [B, m] (real angles) enables the
        per-sample MID-MESH encoding; without it the batch replicates the shared-theta
        unitary.
        """
        if x is not None and self.L < 2:
            raise ValueError("per-sample encoding needs L >= 2 mesh layers - a phase screen "
                             "at the boundary is unobservable for Fock inputs (see docstring)")
        B = x.shape[0] if x is not None else batch
        U = torch.eye(self.m, dtype=self.dtype, device=self.device
                      ).unsqueeze(0).repeat(B, 1, 1)
        rdtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
        xr = x.to(device=self.device, dtype=rdtype) if x is not None else None
        encode_after = self.L // 2 - 1                 # interior position (layers exist on both sides)
        for l, pr in enumerate(self.pairs):
            if pr:
                a = torch.tensor([p[0] for p in pr], device=self.device)
                b = torch.tensor([p[1] for p in pr], device=self.device)
                th = self.theta[l, : len(pr)].to(rdtype)
                c = torch.cos(th).to(self.dtype).view(1, -1, 1)
                sin = torch.sin(th)
                s = torch.complex(torch.zeros_like(sin), sin).to(self.dtype).view(1, -1, 1)  # i*sin
                ra, rb = U[:, a, :], U[:, b, :]
                U = U.clone()      # autograd-safe row assignment (see memory note above)
                U[:, a, :] = c * ra + s * rb
                U[:, b, :] = s * ra + c * rb
            if xr is not None and l == encode_after:
                # mid-mesh phase screen: row j scaled by e^{i x_bj} (polar keeps grad to x)
                U = torch.polar(torch.ones_like(xr), xr).to(self.dtype).unsqueeze(-1) * U
        return U
