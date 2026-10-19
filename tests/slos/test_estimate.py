"""EmitterGurvitsEstimator (docs/ESTIMATE.md) against the bond-carrying strong simulation:
unbiased amplitudes / probabilities for projected and traced emitters, collision-free and
bunched outputs, both sampling laws; second moment at its theoretical value ||psi_f||^2 prod t_j!
(Gaussian) and below it (Rademacher)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.slos.estimate import EmitterGurvitsEstimator

C128 = torch.complex128
SHOTS = 60_000


def _caterpillar(n, phis, final=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final)


def _random_emitter(n, L, chi, m, seed, final=True):
    """Random emitter, normalised so that the traced photonic state has unit trace."""
    rng = np.random.default_rng(seed)
    ops = rng.normal(size=(n, L, chi, chi)) + 1j * rng.normal(size=(n, L, chi, chi))
    modes = tuple(tuple(range(k * L, (k + 1) * L)) for k in range(n))
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    s0 /= np.linalg.norm(s0)
    tot = sum(float(np.vdot(sp.amplitudes, sp.amplitudes).real)
              for sp in EmitterMPS(ops=ops, modes=modes, s0=s0, m=m).expand())
    ops = ops / tot ** (0.5 / n)
    f = None
    if final:
        f = rng.normal(size=chi) + 1j * rng.normal(size=chi)
        f /= np.linalg.norm(f)
    return EmitterMPS(ops=ops, modes=modes, s0=s0, m=m, final=f)


def _outputs(layers):
    fs = layers.final_states
    free = next(i for i, t in enumerate(fs) if max(t) == 1)
    bunched = next(i for i, t in enumerate(fs) if max(t) == 2)
    return [(free, fs[free]), (bunched, fs[bunched])]


STATES = {
    "cluster5_proj": lambda: _caterpillar(5, [0] * 4, final=[1, 0]),
    "ghz5_proj": lambda: _caterpillar(5, [math.pi] * 4, final=[1 / math.sqrt(2), 1 / math.sqrt(2)]),
    "cluster5_traced": lambda: _caterpillar(5, [0] * 4),
    "rand_n4_L3_chi3_proj": lambda: _random_emitter(4, 3, 3, 12, 1),
    "rand_n4_L2_chi2_traced": lambda: _random_emitter(4, 2, 2, 8, 2, final=False),
}


@pytest.mark.parametrize("sampling", ["gaussian", "rademacher"])
@pytest.mark.parametrize("name", list(STATES))
def test_probability_unbiased(name, sampling):
    st = STATES[name]()
    U = haar_unitary(st.m, 11)
    K = EmitterSLOSKernel(st, dtype=C128)
    P = K.probs(U)
    est = EmitterGurvitsEstimator(st, sampling=sampling, seed=7)
    for idx, t in _outputs(K.layers):
        e = est.probability(U, t, shots=SHOTS)
        assert abs(e.value - P[idx].item()) < 5 * e.stderr + 1e-12, (name, t, e, P[idx].item())


@pytest.mark.parametrize("sampling", ["gaussian", "rademacher"])
@pytest.mark.parametrize("name", ["cluster5_proj", "ghz5_proj", "rand_n4_L3_chi3_proj"])
def test_amplitude_unbiased_and_second_moment(name, sampling):
    st = STATES[name]()
    U = haar_unitary(st.m, 12)
    K = EmitterSLOSKernel(st, dtype=C128)
    raw = K.amplitudes(U)                                   # raw = physical / sqrt(prod t!)
    norm2 = float(sum(np.vdot(sp.amplitudes, sp.amplitudes).real for sp in st.expand()))
    est = EmitterGurvitsEstimator(st, sampling=sampling, seed=3)
    for idx, t in _outputs(K.layers):
        tfact = math.prod(math.factorial(c) for c in t)
        A = raw[idx].item() * math.sqrt(tfact)
        e = est.amplitude(U, t, shots=SHOTS)
        assert abs(e.value - A) < 5 * e.stderr + 1e-12, (name, t, e, A)
        if sampling == "gaussian":                          # E|Z|^2 = ||psi_f||^2 exactly (unitary invariance)
            assert abs(e.second_moment - norm2) < 0.15 * norm2, (name, t, e.second_moment, norm2)
        else:                                               # no theorem; measured <= the Gaussian value
            assert e.second_moment < 1.2 * norm2


def test_traced_equals_mixture_of_projections():
    """Traced estimate = sum over an orthonormal final basis of projected estimates (same U, t)."""
    st = _caterpillar(4, [0.3, 1.1, 2.0])
    U = haar_unitary(st.m, 5)
    K = EmitterSLOSKernel(st, dtype=C128)
    idx, t = _outputs(K.layers)[0]
    e_tr = EmitterGurvitsEstimator(st, seed=1).probability(U, t, shots=SHOTS)
    parts = []
    for f in np.eye(2):
        stf = EmitterMPS(ops=st.ops, modes=st.modes, s0=st.s0, m=st.m, final=f)
        parts.append(EmitterGurvitsEstimator(stf, seed=2).probability(U, t, shots=SHOTS))
    tot = sum(p.value for p in parts)
    err = math.sqrt(e_tr.stderr ** 2 + sum(p.stderr ** 2 for p in parts))
    assert abs(e_tr.value - tot) < 5 * err
    assert abs(tot - K.probs(U)[idx].item()) < 5 * err


def test_bad_inputs():
    st = _caterpillar(3, [0, 0])
    U = haar_unitary(st.m, 1)
    with pytest.raises(ValueError):
        EmitterGurvitsEstimator(st, sampling="uniform")
    with pytest.raises(ValueError):                         # traced emitter has no amplitude
        EmitterGurvitsEstimator(st).amplitude(U, (1, 1, 1, 0, 0, 0), 10)
    with pytest.raises(ValueError):                         # wrong photon number
        EmitterGurvitsEstimator(st).probability(U, (1, 1, 0, 0, 0, 0), 10)
