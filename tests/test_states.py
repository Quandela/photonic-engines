"""Input-state types: validation, and the caterpillar constructor against the paper's states."""
import math

import numpy as np
import pytest

from photonic_engines import EmitterMPS, FockState, FockSuperposition, spin_gate


def test_fock_state_basics():
    s = FockState((1, 0, 2))
    assert (s.n, s.m, s.modes(), s.norm_factor()) == (3, 3, [0, 2, 2], 2)
    with pytest.raises(ValueError):
        FockState((1, -1))


def test_superposition_validation():
    FockSuperposition(((1, 0), (0, 1)), (1, 1j))
    with pytest.raises(ValueError):
        FockSuperposition(((1, 0), (1, 1)), (1, 1))          # photon numbers differ
    with pytest.raises(ValueError):
        FockSuperposition(((1, 0), (1, 0)), (1, 1))          # duplicate component


def _caterpillar(n, phis, final=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final)


def test_ghz_expands_to_two_components():
    n = 4
    st = _caterpillar(n, [math.pi] * (n - 1))
    comps = st.expand()                     # traced spin: one superposition per final spin
    assert len(comps) == 2
    occs = {c.components[0].occupation for c in comps}
    assert occs == {tuple([1, 0] * n), tuple([0, 1] * n)}
    for c in comps:
        assert len(c.components) == 1 and abs(abs(c.amplitudes[0]) - 1 / math.sqrt(2)) < 1e-12


def test_linear_cluster_is_uniform_superposition():
    n = 4
    st = _caterpillar(n, [0.0] * (n - 1))
    comps = st.expand()
    assert len(comps) == 2
    total = sum(c.norm() ** 2 for c in comps)
    assert abs(total - 1) < 1e-12
    for c in comps:
        assert len(c.components) == 2 ** (n - 1)
        assert all(abs(abs(a) - 2 ** (-n / 2)) < 1e-12 for a in c.amplitudes)


def test_projected_final_is_pure_and_normalized_by_projection_weight():
    n = 3
    st = _caterpillar(n, [0.0] * (n - 1), final=[1, 0])
    (c,) = st.expand()
    assert abs(c.norm() ** 2 - 0.5) < 1e-12      # <up| picks half the weight of the traced state


def test_emitter_mps_validation():
    ops = np.zeros((2, 2, 2, 2), dtype=complex)
    with pytest.raises(ValueError):
        EmitterMPS(ops, ((0, 1), (1, 2)), np.array([1, 0]), m=4)     # mode 1 reused
    with pytest.raises(ValueError):
        EmitterMPS(ops, ((0, 1), (2, 3)), np.array([1, 0, 0]), m=4)  # s0 wrong length
