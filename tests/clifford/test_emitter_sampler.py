"""Batched torch sampler (clifford/emitter_sampler.py): histogram vs EmitterSLOSKernel on the
cases of the reference tests, and agreement with the reference sampler's per-step weights."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_sampler import sample_torch
from photonic_engines.slos.emitter import EmitterSLOSKernel

C128 = torch.complex128


def _caterpillar(n, phis, final=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final)


def _random_emitter(n, L, chi, m, seed, final=True):
    rng = np.random.default_rng(seed)
    ops = rng.normal(size=(n, L, chi, chi)) + 1j * rng.normal(size=(n, L, chi, chi))
    modes = tuple(tuple(range(k * L, (k + 1) * L)) for k in range(n))
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    f = (rng.normal(size=chi) + 1j * rng.normal(size=chi)) if final else None
    return EmitterMPS(ops=ops, modes=modes, s0=s0, m=m, final=f)


def _tvd(state, U, shots, seed, **kw):
    k = EmitterSLOSKernel(state, dtype=C128)
    p = k.probs(U)
    p = p / p.sum()
    occ = sample_torch(state, U, shots, generator=torch.Generator().manual_seed(seed), **kw)
    assert occ.shape == (shots, state.m) and (occ.sum(1) == state.n).all()
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= shots
    tvd = 0.5 * float((emp - p).abs().sum())
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / shots).sum()) * math.sqrt(2 / math.pi)
    return tvd, floor


@pytest.mark.parametrize("state,seed", [
    (_caterpillar(3, [0, 0]), 1),
    (_caterpillar(4, [math.pi, 0, 0.9]), 3),
    (_caterpillar(4, [math.pi] * 3, final=(1, 0)), 4),
    (_caterpillar(5, [0.3, 1.1, 2.0, 0.5], final=(0.6, 0.8j)), 5),
    (_random_emitter(3, 2, 2, m=7, seed=4), 6),
    (_random_emitter(3, 3, 2, m=9, seed=6, final=False), 7),
    (_random_emitter(3, 2, 3, m=7, seed=8, final=False), 9),      # chi = 3
])
def test_batched_sampler_matches_strong_simulation(state, seed):
    U = haar_unitary(state.m, seed)
    tvd, floor = _tvd(state, U, 4000, seed)
    assert tvd < 2.0 * floor, (tvd, floor)


def test_delta_chunking_does_not_change_samples():
    state = _caterpillar(5, [0, 0, 0, 0])
    U = haar_unitary(state.m, 2)
    a = sample_torch(state, U, 300, generator=torch.Generator().manual_seed(3), delta_chunk=1 << 12)
    b = sample_torch(state, U, 300, generator=torch.Generator().manual_seed(3), delta_chunk=3)
    assert torch.equal(a, b)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_batched_sampler_cuda():
    state = _caterpillar(4, [math.pi, 0, 0.9])
    U = haar_unitary(state.m, 3)
    tvd, floor = _tvd(state, U, 4000, 3, device="cuda")
    assert tvd < 2.0 * floor, (tvd, floor)
