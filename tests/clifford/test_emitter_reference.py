"""Generalized Clifford chain rule for EmitterMPS inputs (clifford/emitter_reference.py):
the sampler's histogram must match the strong simulation (EmitterSLOSKernel) - this is the
numerical check of the marginal lemma for entangled inputs - and the kernel-style computation
(gap Kraus terms + matrix Glynn cofactors) must equal the explicit reduced-density weights."""
import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_reference import (
    candidate_weights_reference,
    emitter_sample_reference,
    gap_kraus_terms,
    matrix_cofactor_weights,
    pure_coefficients,
    reduced_density,
)
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


def _tvd_check(state, U, shots, seed, **kw):
    k = EmitterSLOSKernel(state, dtype=C128)
    p = k.probs(U)
    p = p / p.sum()
    occ = emitter_sample_reference(state, U, shots, generator=torch.Generator().manual_seed(seed), **kw)
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= shots
    tvd = 0.5 * float((emp - p).abs().sum())
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / shots).sum()) * math.sqrt(2 / math.pi)
    return tvd, floor


@pytest.mark.parametrize("state,seed,shots", [
    (_caterpillar(3, [0, 0]), 1, 3000),                              # linear cluster, spin traced
    (_caterpillar(3, [math.pi, math.pi], final=(1, 0)), 2, 3000),    # GHZ, spin projected
    (_caterpillar(4, [math.pi, 0, 0.9]), 3, 800),                    # caterpillar, traced (slow reference: fewer shots)
    (_random_emitter(3, 2, 2, m=7, seed=4), 5, 3000),                # random chi=2 emitter, projected
    (_random_emitter(3, 3, 2, m=9, seed=6, final=False), 7, 1000),   # L=3, traced
])
def test_sampler_matches_strong_simulation(state, seed, shots):
    U = haar_unitary(state.m, seed)
    tvd, floor = _tvd_check(state, U, shots, seed)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_matrix_method_matches_density_method_samples():
    state = _caterpillar(3, [0, math.pi])
    U = haar_unitary(state.m, 11)
    tvd, floor = _tvd_check(state, U, 2000, 11, method="matrix")
    assert tvd < 2.5 * floor, (tvd, floor)


def test_fixed_photon_order_is_wrong():
    """With sigma = identity the chain rule does not reproduce the distribution: the uniform
    random order is what makes the subset-averaged marginals telescope."""
    state = _caterpillar(3, [0, 0])
    U = haar_unitary(state.m, 21)
    tvd, floor = _tvd_check(state, U, 3000, 21, order="fixed")
    assert tvd > 3 * floor, (tvd, floor)          # measured: 4.2x the floor (0.216 vs 0.051)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matrix_cofactor_weights_equal_reduced_density_weights(seed):
    rng = np.random.default_rng(seed)
    n = 5
    state = _random_emitter(n, 2, 2, m=2 * n + 1, seed=seed)
    U = haar_unitary(state.m, seed + 30)
    c = pure_coefficients(state)
    for k in range(1, n + 1):
        T = rng.permutation(n)[:k].tolist()
        rows = rng.integers(0, state.m, size=k - 1).tolist()
        w1 = candidate_weights_reference(U, state, c, T, rows)
        w2, R = matrix_cofactor_weights(U, state, T, rows)
        assert torch.allclose(w1, w2, atol=1e-12 * max(1.0, float(w1.max()))), (k, T, rows)


def test_gap_kraus_terms_reconstruct_reduced_density():
    n = 6
    state = _random_emitter(n, 2, 2, m=12, seed=9)
    c = pure_coefficients(state)
    rng = np.random.default_rng(3)
    for _ in range(6):
        k = int(rng.integers(1, n + 1))
        T = rng.permutation(n)[:k].tolist()
        kept, terms = gap_kraus_terms(state, T)
        rho = reduced_density(c, kept)                                # index order = sorted T
        acc = torch.zeros_like(rho)
        for s0_e, mats, f_e in terms:
            psi = torch.zeros(2 ** k, dtype=C128)
            for xi, x in enumerate(itertools.product(range(2), repeat=k)):
                v = s0_e
                for j, lab in enumerate(x):
                    v = mats[j, lab] @ v
                psi[xi] = torch.vdot(f_e, v)
            acc += torch.outer(psi, psi.conj())
        assert torch.allclose(acc, rho, atol=1e-12 * max(1.0, float(rho.abs().max()))), (k, T)
        # rank bound: prod over gaps of min(4, 2**len)
        gaps = [b - a - 1 for a, b in itertools.pairwise(kept)]
        bound = math.prod(min(4, 2 ** g) for g in gaps if g > 0) * min(2, 2 ** kept[0]) * min(2, 2 ** (n - 1 - kept[-1]))
        assert len(terms) <= bound


def test_ghz_has_rank_two_reduced_states():
    """Structured states collapse R: GHZ reduced states are rank 2 whatever the gaps."""
    state = _caterpillar(6, [math.pi] * 5, final=(1, 1))
    for T in ([0, 5], [1, 3, 5], [0, 2, 4]):
        _, terms = gap_kraus_terms(state, T)
        assert len(terms) <= 2, (T, len(terms))


@pytest.mark.parametrize("final", [None, (1, 0)])
def test_caterpillar_segment_weights_match_matrix_cofactors(final):
    """Rank-one (measure-and-prepare) gap operators: same R, same weights as the SVD purification."""
    from photonic_engines.clifford.emitter_caterpillar import caterpillar_gates, caterpillar_step_weights
    n = 6
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, 0.3 * j) for j in range(1, n)]
    st = EmitterMPS.caterpillar(n, gates, final=final)
    assert caterpillar_gates(st) is not None
    assert caterpillar_gates(_random_emitter(3, 2, 2, m=7, seed=1)) is None
    U = haar_unitary(st.m, 5)
    fv = torch.eye(2, dtype=C128)[1] if final is None else torch.tensor(final, dtype=C128)
    for T, rows in [([2], []), ([0, 3], [1]), ([1, 2, 4], [0, 7]), ([0, 1, 2, 3, 4, 5], [3, 1, 4, 6, 2]),
                    ([1, 3, 5], [2, 9]), ([0, 5], [1])]:
        w1, R1 = matrix_cofactor_weights(U, st, T, rows, final=fv)
        w2, R2 = caterpillar_step_weights(U, st, T, rows, final=fv)
        assert R1 == R2
        assert float((w1 - w2).abs().max()) < 1e-12 * float(w1.max())


def test_caterpillar_segment_sampler_matches_strong_simulation():
    from photonic_engines.clifford.emitter_caterpillar import sample_caterpillar
    n = 4
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, 0.4 * j) for j in range(1, n)]
    st = EmitterMPS.caterpillar(n, gates)
    U = haar_unitary(st.m, n)
    k = EmitterSLOSKernel(st, dtype=C128)
    p = k.probs(U)
    p = p / p.sum()
    occ = sample_caterpillar(st, U, 3000, generator=torch.Generator().manual_seed(1))
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= 3000
    tvd = 0.5 * float((emp - p).abs().sum())
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / 3000).sum()) * math.sqrt(2 / math.pi)
    assert tvd < 2.5 * floor, (tvd, floor)
