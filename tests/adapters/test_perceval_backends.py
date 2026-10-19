"""Perceval backend wrappers vs Perceval's own SLOSBackend on the same circuit and input."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pcvl = pytest.importorskip("perceval")

from helpers import haar_unitary
from perceval.backends import SLOSBackend

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.adapters.perceval.backends import EmitterSLOSBackend, SLOSTorchBackend
from photonic_engines.slos.emitter import EmitterSLOSKernel


def _circuit(m, seed):
    U = haar_unitary(m, seed).numpy()
    return pcvl.Unitary(pcvl.Matrix(U)), U


@pytest.mark.parametrize("occ", [[1, 0, 1, 0, 1, 0], [2, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [3, 0, 0, 0, 0, 0]])
def test_slos_torch_matches_perceval(occ):
    circ, _ = _circuit(6, sum(occ) + 3)
    ref = SLOSBackend(); ref.set_circuit(circ); ref.set_input_state(pcvl.BasicState(occ))
    b = SLOSTorchBackend(dtype=torch.complex128); b.set_circuit(circ); b.set_input_state(pcvl.BasicState(occ))
    assert np.allclose(ref.all_prob(), b.all_prob(), atol=1e-10)
    for s in list(ref._get_iterator(ref._input_state))[:5]:
        assert abs(ref.prob_amplitude(s) - b.prob_amplitude(s)) < 1e-10
    assert abs(sum(b.all_prob()) - 1) < 1e-10
    sv_ref, sv = ref.evolve(), b.evolve()
    for s, a in sv_ref:
        assert abs(a - sv[s]) < 1e-10


def test_emitter_backend_matches_kernel_and_normalizes():
    n = 3
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n)
    circ, U = _circuit(st.m, 5)
    b = EmitterSLOSBackend(dtype=torch.complex128); b.set_circuit(circ); b.set_emitter_state(st)
    p = b.all_prob()
    assert abs(sum(p) - 1) < 1e-10
    k = EmitterSLOSKernel(st, dtype=torch.complex128)
    pk = k.probs(torch.as_tensor(U))
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    for s, ps in zip(b._get_iterator(b._input_state), p):
        assert abs(pk[idx[tuple(s)]].item() - ps) < 1e-10
    dist = b.prob_distribution()
    assert abs(sum(dist.values()) - 1) < 1e-10
    with pytest.raises(NotImplementedError):
        b.evolve()                                       # traced spin: mixed, no amplitudes
    stp = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, final=[1, 0])
    b2 = EmitterSLOSBackend(dtype=torch.complex128); b2.set_circuit(circ); b2.set_emitter_state(stp)
    sv = b2.evolve()                                     # Perceval normalizes StateVectors
    p2 = dict(zip((tuple(s) for s in b2._get_iterator(b2._input_state)), b2.all_prob()))
    tot = sum(p2.values())
    assert abs(tot - 0.5) < 1e-10                        # projection weight of <up|
    for s, a in sv:
        assert abs(abs(a) ** 2 - p2[tuple(s)] / tot) < 1e-10


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_clifford_gpu_backend_samples():
    from photonic_engines.adapters.perceval.backends import CliffordGPUBackend
    circ, _ = _circuit(6, 1)
    b = CliffordGPUBackend(seed=0); b.set_circuit(circ); b.set_input_state(pcvl.BasicState([1, 0, 1, 0, 1, 0]))
    smp = b.samples(200)
    assert len(smp) == 200 and all(s.n == 3 for s in smp)


def test_statevector_input_matches_perceval_simulator():
    """Coherent superposition inputs (Perceval StateVector, mixed photon numbers) against
    Perceval's Simulator(SLOSBackend)."""
    from perceval.simulators import Simulator
    from perceval.utils import StateVector
    circ, _ = _circuit(5, 21)
    sv = (StateVector(pcvl.BasicState([1, 0, 1, 0, 0])) + 1j * StateVector(pcvl.BasicState([0, 1, 0, 1, 0]))
          + 0.5 * StateVector(pcvl.BasicState([2, 0, 0, 0, 0])) + 0.3 * StateVector(pcvl.BasicState([0, 0, 0, 0, 1])))
    sv.normalize()
    sim = Simulator(SLOSBackend()); sim.set_circuit(circ)
    ref_p = sim.probs(sv); ref_sv = sim.evolve(sv)
    b = SLOSTorchBackend(dtype=torch.complex128); b.set_circuit(circ); b.set_input_state(sv)
    p = b.prob_distribution()
    assert abs(sum(p.values()) - 1) < 1e-10
    for s, v in ref_p.items():
        assert abs(p[s] - v) < 1e-10, s
    out = b.evolve()
    for s, a in ref_sv:
        assert abs(out[s] - a) < 1e-10, s
    # the same through FockSuperposition (single photon-number block) and SuperpositionSLOSKernel.from_state
    from photonic_engines import FockSuperposition
    from photonic_engines.slos import SuperpositionSLOSKernel
    fs = FockSuperposition(((1, 0, 1, 0, 0), (0, 1, 0, 1, 0), (2, 0, 0, 0, 0)), (1, 1j, 0.5))
    b.set_input_state(fs)
    k = SuperpositionSLOSKernel.from_state(fs, dtype=torch.complex128)
    pk = k.probs(torch.as_tensor(_circuit(5, 21)[1]))
    assert abs(float(pk.sum()) - 1) < 1e-10
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    for s, v in b.prob_distribution().items():
        assert abs(float(pk[idx[tuple(s)]]) - v) < 1e-10
