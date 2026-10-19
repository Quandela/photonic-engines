"""The Perceval -> torch circuit converter: plan structure, exactness vs `compute_unitary()`,
and the trainable `ParametrizedMesh` (shapes, unitarity, mid-mesh encoding physics, gradients).
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pcvl = pytest.importorskip("perceval")

from photonic_engines.adapters.perceval.converter import (
    CircuitConverter,
    ConversionPlan,
    ParametrizedMesh,
    _DenseLayer,
    _empty,
    _Layer,
    build_plan,
)
from photonic_engines.slos import FockSLOSKernel

# `convert()` builds in complex64 (hard-coded in the module), so agreement with perceval's
# complex128 `compute_unitary()` is at single-precision level, not 1e-12.
C64_TOL = 1e-6


# ---------------------------------------------------------------------------- circuits under test

def _bs() -> pcvl.Circuit:
    return pcvl.Circuit(2).add(0, pcvl.BS(theta=0.7, phi_tl=0.3))


def _mesh() -> pcvl.Circuit:
    """A 4-mode rectangular BS/PS mesh: two full columns plus phase screens."""
    c = pcvl.Circuit(4)
    for col, off in enumerate((0, 1, 0, 1)):
        for a in range(off, 3, 2):
            c.add(a, pcvl.BS(theta=0.4 + 0.11 * col + 0.07 * a))
            c.add(a, pcvl.PS(0.2 + 0.13 * col + 0.05 * a))
    return c


def _perm() -> pcvl.Circuit:
    """Non-trivial mode permutations: a 2-mode swap and a 3-mode cycle (the `_DenseLayer` path)."""
    c = pcvl.Circuit(5)
    c.add(0, pcvl.BS(theta=1.1))
    c.add(1, pcvl.PERM([2, 0, 1]))
    c.add(3, pcvl.BS.H(theta=0.6))
    c.add(0, pcvl.PERM([1, 0]))
    c.add(2, pcvl.PS(1.4))
    c.add(2, pcvl.BS(theta=0.25, phi_bl=0.8))
    return c


CIRCUITS = {"bs": _bs, "mesh": _mesh, "perm": _perm}


def _components(circuit):
    """(modes, 2-D unitary) of every non-Barrier component, in circuit order."""
    out = []
    for r, comp in circuit:
        if type(comp).__name__ == "Barrier":
            continue
        out.append((tuple(int(x) for x in r), np.array(comp.compute_unitary(), dtype=np.complex128)))
    return out


def _layer_modes(layer) -> list[int]:
    if isinstance(layer, _DenseLayer):
        return list(layer.modes)
    return list(layer.a) + list(layer.b) + list(layer.s)


# ------------------------------------------------------------------------------------- build_plan

@pytest.mark.parametrize("name", list(CIRCUITS))
def test_plan_places_every_component_exactly_once(name):
    circuit = CIRCUITS[name]()
    plan = build_plan(circuit)
    assert isinstance(plan, ConversionPlan)
    assert plan.m == circuit.m
    placed = sum(len(L.modes) > 0 if isinstance(L, _DenseLayer) else len(L.a) + len(L.s) for L in plan.layers)
    assert placed == len(_components(circuit))
    # the gate matrices themselves survive the compilation, in schedule order
    pair_mats = [g for L in plan.layers if not isinstance(L, _DenseLayer) for g in L.g]
    assert all(g.shape == (2, 2) for g in pair_mats)


@pytest.mark.parametrize("name", list(CIRCUITS))
def test_plan_layers_are_disjoint_and_non_empty(name):
    plan = build_plan(CIRCUITS[name]())
    for layer in plan.layers:
        modes = _layer_modes(layer)
        assert len(modes) == len(set(modes)), "gates sharing a mode cannot commute within a layer"
        assert all(0 <= x < plan.m for x in modes)
        assert not _empty(layer)
    assert all(_layer_modes(L) for L in plan.layers)


@pytest.mark.parametrize("name", list(CIRCUITS))
def test_plan_is_asap_schedule(name):
    """The mode groups per layer match an independent ASAP list-scheduler on the same circuit."""
    circuit = CIRCUITS[name]()
    last_touch = [-1] * circuit.m
    expected: dict[int, list[tuple[int, ...]]] = {}
    for modes, _ in _components(circuit):
        li = 1 + max(last_touch[x] for x in modes)
        if len(modes) > 2:                                   # dense blocks get a layer of their own
            li = max(expected, default=-1) + 1
            while li in expected:
                li += 1
        expected.setdefault(li, []).append(modes)
        for x in modes:
            last_touch[x] = li
    got = [sorted(_layer_modes(L)) for L in build_plan(circuit).layers]
    want = [sorted(x for g in expected[i] for x in g) for i in sorted(expected)]
    assert got == want


def test_plan_skips_barriers_and_isolates_dense_blocks():
    c = pcvl.Circuit(4)
    c.add(0, pcvl.BS(theta=0.7))
    c.add(1, pcvl.PERM([2, 0, 1]))
    c.barrier()
    c.add(0, pcvl.PS(0.5))
    plan = build_plan(c)
    dense = [L for L in plan.layers if isinstance(L, _DenseLayer)]
    assert len(dense) == 1 and dense[0].modes == [1, 2, 3] and dense[0].mat.shape == (3, 3)
    assert len(plan.layers) == len(_components(c))           # BS | PERM3 | PS, barrier dropped
    assert _empty(_Layer()) and not _empty(_Layer(a=[0], b=[1], g=[np.eye(2)]))
    assert not _empty(_Layer(s=[0], ph=[1j])) and not _empty(dense[0])


# --------------------------------------------------------------------------------- convert()

@pytest.mark.parametrize("name", list(CIRCUITS))
def test_convert_matches_perceval_compute_unitary(name):
    """The load-bearing test: the converted matrix reproduces perceval's own unitary."""
    circuit = CIRCUITS[name]()
    ref = np.array(circuit.compute_unitary(), dtype=np.complex128)
    got = CircuitConverter(circuit).convert("cpu")
    assert got.dtype == torch.complex64 and tuple(got.shape) == (circuit.m, circuit.m)
    got = got.numpy().astype(np.complex128)
    assert np.abs(got - ref).max() < C64_TOL, f"{name}: max |dU| = {np.abs(got - ref).max():.2e}"
    assert np.abs(got @ got.conj().T - np.eye(circuit.m)).max() < C64_TOL


def test_convert_matches_perceval_on_random_meshes():
    """Same claim over randomly parametrized meshes of several widths and depths."""
    rng = np.random.default_rng(20240917)
    for m, depth in ((2, 3), (3, 4), (5, 6), (6, 4)):
        c = pcvl.Circuit(m)
        for col in range(depth):
            for a in range(col % 2, m - 1, 2):
                c.add(a, pcvl.BS(theta=float(rng.uniform(0, math.pi))))
                c.add(a + 1, pcvl.PS(float(rng.uniform(0, 2 * math.pi))))
        ref = np.array(c.compute_unitary(), dtype=np.complex128)
        got = CircuitConverter(c).convert().numpy().astype(np.complex128)
        assert np.abs(got - ref).max() < C64_TOL, f"m={m} depth={depth}"


def test_convert_is_repeatable_and_caches_per_device():
    circuit = _mesh()
    conv = CircuitConverter(circuit)
    assert conv.m == circuit.m and not conv._cache
    first = conv.convert("cpu")
    assert set(conv._cache) == {"cpu"}
    cached = conv._cache["cpu"]
    second = conv.convert("cpu")
    assert conv._cache["cpu"] is cached                      # prepared tensors reused, not rebuilt
    assert torch.equal(first, second)                        # and the replay is side-effect free


def test_convert_on_empty_and_single_component_circuits():
    empty = pcvl.Circuit(3)
    assert build_plan(empty).layers == []
    assert torch.equal(CircuitConverter(empty).convert(), torch.eye(3, dtype=torch.complex64))
    ps = pcvl.Circuit(2).add(1, pcvl.PS(0.9))
    got = CircuitConverter(ps).convert().numpy().astype(np.complex128)
    assert np.abs(got - np.array(ps.compute_unitary(), dtype=np.complex128)).max() < C64_TOL


# ------------------------------------------------------------------------- ParametrizedMesh shapes

@pytest.mark.parametrize(("m", "layers"), [(2, 2), (4, 4), (5, 3), (6, 5)])
def test_mesh_parameters_are_the_single_theta_leaf(m, layers):
    mesh = ParametrizedMesh(m, layers, dtype=torch.complex128, seed=11)
    params = mesh.parameters()
    assert len(params) == 1
    theta = params[0]
    assert theta is mesh.theta
    assert tuple(theta.shape) == (layers, m // 2)
    assert theta.requires_grad and theta.is_leaf and theta.grad is None
    assert float(theta.detach().min()) >= 0.0 and float(theta.detach().max()) <= math.pi
    assert len(mesh.pairs) == layers
    for l, pr in enumerate(mesh.pairs):                      # alternating-offset disjoint pairs
        modes = [x for p in pr for x in p]
        assert len(modes) == len(set(modes))
        assert all(p[1] == p[0] + 1 and p[0] % 2 == l % 2 for p in pr)


@pytest.mark.parametrize(("m", "layers"), [(2, 2), (3, 3), (4, 4), (6, 5)])
def test_mesh_unitary_is_unitary(m, layers):
    mesh = ParametrizedMesh(m, layers, dtype=torch.complex128, seed=5)
    for U in (mesh.unitary(), mesh.unitary(batch=3)):
        eye = torch.eye(m, dtype=torch.complex128).expand_as(U)
        assert (U @ U.conj().transpose(-1, -2) - eye).abs().max() < 1e-12
        assert (U.conj().transpose(-1, -2) @ U - eye).abs().max() < 1e-12


def test_mesh_default_layers_and_batch_replicates():
    mesh = ParametrizedMesh(4, dtype=torch.complex128, seed=2)
    assert mesh.L == 4                                       # `layers=None` -> m
    U = mesh.unitary(batch=4)
    assert tuple(U.shape) == (4, 4, 4)
    assert torch.allclose(U, U[0].unsqueeze(0).expand_as(U))
    assert torch.allclose(mesh.unitary()[0], U[0])           # batch=1 default


def test_mesh_x_gives_per_sample_unitaries():
    m, B = 4, 3
    mesh = ParametrizedMesh(m, 4, dtype=torch.complex128, seed=8)
    g = torch.Generator().manual_seed(4)
    x = torch.rand(B, m, generator=g, dtype=torch.float64) * math.pi
    U = mesh.unitary(x)
    assert tuple(U.shape) == (B, m, m) and U.dtype == torch.complex128
    eye = torch.eye(m, dtype=torch.complex128).expand_as(U)
    assert (U @ U.conj().transpose(-1, -2) - eye).abs().max() < 1e-12
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert (U[i] - U[j]).abs().max() > 1e-3, f"rows {i},{j} of x produced the same unitary"
    # a repeated row must reproduce its unitary exactly: the map is a function of (theta, x_b)
    U2 = mesh.unitary(torch.stack([x[0], x[0]]))
    assert torch.allclose(U2[0], U2[1]) and torch.allclose(U2[0], U[0])


def test_mesh_rejects_encoding_with_fewer_than_two_layers():
    mesh = ParametrizedMesh(4, 1, dtype=torch.complex128, seed=1)
    x = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        mesh.unitary(x)
    assert tuple(mesh.unitary(batch=2).shape) == (2, 4, 4)   # without `x` a single layer is fine


# ----------------------------------------------------------- the physics claim in the docstring

def _kernel(n, m):
    from helpers import occupation_list
    return FockSLOSKernel(n, m, occupation_list(n, m), dtype=torch.complex128)


def test_mid_mesh_encoding_changes_output_probabilities():
    """The reason the phase screen sits mid-mesh: it must move the Fock output DISTRIBUTION,
    not just the amplitudes."""
    m, n, B = 4, 2, 3
    mesh = ParametrizedMesh(m, 4, dtype=torch.complex128, seed=3)
    g = torch.Generator().manual_seed(17)
    x = torch.rand(B, m, generator=g, dtype=torch.float64) * math.pi
    p = _kernel(n, m).probs(mesh.unitary(x))
    assert p.shape[0] == B
    assert torch.allclose(p.sum(-1), torch.ones(B, dtype=torch.float64), atol=1e-12)
    assert (p >= -1e-15).all()
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert (p[i] - p[j]).abs().max() > 1e-4, f"x rows {i},{j} gave the same distribution"
    p0 = _kernel(n, m).probs(mesh.unitary(batch=B))          # x = 0 screen is the identity
    assert torch.allclose(_kernel(n, m).probs(mesh.unitary(torch.zeros(B, m, dtype=torch.float64))), p0, atol=1e-12)
    assert (p - p0).abs().max() > 1e-4                       # and a non-zero screen is observable


def test_boundary_phase_screens_are_unobservable():
    """The constraint the mid-mesh placement works around: a screen at either END of the mesh
    leaves the Fock output probabilities untouched (global phase per basis state)."""
    m, n = 4, 2
    mesh = ParametrizedMesh(m, 4, dtype=torch.complex128, seed=6)
    kernel = _kernel(n, m)
    U = mesh.unitary()[0]
    g = torch.Generator().manual_seed(23)
    phase = torch.polar(torch.ones(m, dtype=torch.float64), torch.rand(m, generator=g, dtype=torch.float64) * math.pi)
    base = kernel.probs(U.unsqueeze(0))
    before = kernel.probs((U @ torch.diag(phase)).unsqueeze(0))
    after = kernel.probs((torch.diag(phase) @ U).unsqueeze(0))
    assert (before - base).abs().max() < 1e-12
    assert (after - base).abs().max() < 1e-12
    assert (torch.diag(phase) @ U - U).abs().max() > 1e-3    # the amplitudes DID change


# ------------------------------------------------------------------------------------- gradients

def test_probabilities_backpropagate_to_theta_and_x():
    m, n, B = 4, 2, 2
    mesh = ParametrizedMesh(m, 4, dtype=torch.complex128, seed=13)
    theta = mesh.parameters()[0]
    g = torch.Generator().manual_seed(31)
    x = (torch.rand(B, m, generator=g, dtype=torch.float64) * math.pi).requires_grad_(True)
    p = _kernel(n, m).probs(mesh.unitary(x))
    target = torch.zeros_like(p)
    target[:, 0] = 1.0
    loss = ((p - target) ** 2).sum()
    loss.backward()
    for name, grad in (("theta", theta.grad), ("x", x.grad)):
        assert grad is not None, f"no gradient reached {name}"
        assert grad.shape == (theta if name == "theta" else x).shape
        assert torch.isfinite(grad).all(), f"non-finite gradient on {name}"
        assert grad.abs().max() > 1e-8, f"vanishing gradient on {name}"


def test_theta_gradient_matches_finite_differences():
    m, n = 4, 2
    kernel = _kernel(n, m)

    def loss_of(theta):
        mesh = ParametrizedMesh(m, 4, dtype=torch.complex128, seed=19)
        mesh.theta = theta
        return kernel.probs(mesh.unitary()).pow(2).sum()

    base = ParametrizedMesh(m, 4, dtype=torch.complex128, seed=19).theta.detach().to(torch.float64)
    theta = base.clone().requires_grad_(True)
    loss_of(theta).backward()
    eps = 1e-5
    for idx in ((0, 0), (2, 1)):
        bump = torch.zeros_like(base); bump[idx] = eps
        with torch.no_grad():
            up, down = float(loss_of(base + bump)), float(loss_of(base - bump))
        assert abs((up - down) / (2 * eps) - float(theta.grad[idx])) < 1e-5
