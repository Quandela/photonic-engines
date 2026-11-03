"""Partial distinguishability on the register modalities (docs/DISTINGUISHABILITY.md §3.4–3.6):
the time-loop kernel and the multi-emitter sequence kernel.
Oracles: the species-resolved permanent sum on the unrolled unitary (time loops, sequences) and
the emitter strong side (`EmitterSLOSKernel.probs(indist=)`, itself checked against the
expanded chain in tests/test_imperfections.py).
"""
import itertools
import math

import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.imperfections import subset_weights
from photonic_engines.slos.reference import species_probs

C128 = torch.complex128


def _empirical(occ):
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / occ.shape[0]
    return emp


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _noise_floor(p, shots):
    return 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in p.values())


# ---- time loops ----------------------------------------------------------------------------------

def _loop_case(seed=0):
    from photonic_engines.timeloop import TimeLoopCircuit
    T, loops = 6, (1, 2)
    c = TimeLoopCircuit(T, loops)
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(c.n_loops, T, dtype=torch.float64, generator=g) * 3
    phi = torch.rand(c.n_loops, T, dtype=torch.float64, generator=g) * 6
    return c, theta, phi


def _unrolled_x_mixture(U, T, cols, x):
    """x-model on the unrolled unitary, leftover loop modes summed out (as the timeloop tests do)."""
    ref = {}
    for G, w in subset_weights(len(cols), x):
        sp = [0 if j in G else 10 + j for j in range(len(cols))]
        for t, p in species_probs(U, [(1.0, cols)], sp).items():
            ref[t[:T]] = ref.get(t[:T], 0.0) + w * p
    return ref


def test_timeloop_indist_marginals_and_samples_match_the_unrolled_unitary():
    from photonic_engines.timeloop import LoopRegisterKernel
    c, theta, phi = _loop_case(0)
    T, x = c.T, 0.6
    U = c.unitary(theta, phi)
    inp = [1, 1, 1, 0, 0, 0]
    k = LoopRegisterKernel(c, inp)
    ref = _unrolled_x_mixture(U, T, [0, 1, 2], x)
    marg_ref = torch.zeros(T, 4, dtype=torch.float64)
    for t, p in ref.items():
        for b in range(T):
            marg_ref[b, t[b]] += p
    marg = k.bin_marginals(theta, phi, indist=x)                                  # exact enumeration (2^3 subsets)
    assert float((marg - marg_ref).abs().max()) < 1e-13
    assert torch.allclose(k.bin_marginals(theta, phi, indist=1.0), k.bin_marginals(theta, phi))
    mc = k.bin_marginals(theta, phi, indist=x, mc_shots=200, generator=torch.Generator().manual_seed(1))
    assert float((mc - marg_ref).abs().max()) < 1e-13                              # N = 3: still enumerated
    shots = 20000
    occ = k.sample(shots, theta, phi, generator=torch.Generator().manual_seed(1), indist=x)
    assert occ.shape == (shots, T) and (occ.sum(1) <= 3).all()                    # unflushed loops keep photons
    assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)
    # x = 0: three one-photon registers, no interference at all — the classical-transport oracle
    ref0 = _unrolled_x_mixture(U, T, [0, 1, 2], 0.0)
    occ0 = k.sample(shots, theta, phi, generator=torch.Generator().manual_seed(2), indist=0.0)
    assert _tvd(_empirical(occ0), ref0) < 2.0 * _noise_floor(ref0, shots)


def test_timeloop_indist_with_loop_loss_channels():
    """Loop and detector channels run on every register (the interfering one and the one-photon
    ones): the marginals stay normalised and the thresholded sampler agrees with them."""
    from photonic_engines.timeloop import LoopRegisterKernel
    c, theta, phi = _loop_case(3)
    k = LoopRegisterKernel(c, [1, 1, 1, 0, 0, 0])
    loss = {"loop": [0.9, 0.8], "detector": 0.95}
    marg = k.bin_marginals(theta, phi, indist=0.7, loss=loss)
    assert torch.allclose(marg.sum(1), torch.ones(c.T, dtype=marg.dtype), atol=1e-12)
    assert float(marg[:, 0].sum()) > float(k.bin_marginals(theta, phi, indist=0.7)[:, 0].sum())   # loss empties bins
    shots = 20000
    occ = k.sample(shots, theta, phi, generator=torch.Generator().manual_seed(0), indist=0.7, loss=loss)
    assert (occ.sum(1) <= 3).all()
    for b in range(c.T):
        emp = torch.bincount(occ[:, b], minlength=4).double() / shots
        assert float((emp - marg[b]).abs().max()) < 4 * float((marg[b] * (1 - marg[b]) / shots).sqrt().max()) + 1e-3
    with pytest.raises(NotImplementedError):
        k.bin_marginals(theta, phi, indist=0.7, return_register=True)
    uni = k.bin_marginals(theta, phi, indist=0.7, loss=0.8)                        # uniform loss: thinned after
    assert torch.allclose(uni.sum(1), torch.ones(c.T, dtype=uni.dtype), atol=1e-12)


# ---- multi-emitter sequences -------------------------------------------------------------------------

def _sequence_case(k, n, m):
    from photonic_engines.sequence import BinEmitter
    return [BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(0.5 + 0.3 * a, 0.7 * i) for i in range(1, n)],
                                   (2 * a, 2 * a + 1)) for a in range(k)]


@pytest.mark.slow                                      # ~10 min on CPU (625 s, 8-core host)
def test_sequence_indist_kraus_marginals_joint_and_samples():
    from photonic_engines.sequence import SequenceKernel
    from photonic_engines.slos import EmitterSLOSKernel
    k, n, m, xs = 3, 3, 6, [0.9, 0.6, 0.8]
    U = haar_unitary(m, 2)
    sk = SequenceKernel(_sequence_case(k, n, m), m, indist=xs)
    K, coarse = sk.kraus(U, 0), sk.coarse(U, 0)
    assert K.shape[0] == coarse.shape[0] > sk.S
    assert torch.allclose(torch.einsum("qts,qtu->su", K.conj(), K), torch.eye(sk.chi, dtype=C128), atol=1e-10)
    p0 = EmitterSLOSKernel(sk.bin_mps(0), dtype=C128).probs(U, indist=xs)        # per-slot overlaps
    marg = sk.bin_marginals(U)
    assert float((marg[0] - p0).abs().max()) < 1e-13 and torch.allclose(marg.sum(1), torch.ones(n, dtype=marg.dtype), atol=1e-12)
    J = sk.joint(U, [0, 1])
    ps = torch.zeros(sk.S, sk.S, dtype=torch.float64)
    for i in range(sk.S):
        for j in range(sk.S):
            ps[i, j] = sk.sequence_probability(U, np.array([[i, j, t] for t in range(sk.S)])).sum()
    assert float((J - ps).abs().max()) < 1e-13 and abs(float(J.sum()) - 1) < 1e-12
    shots = 20000
    pats = sk.sample(U, shots, generator=torch.Generator().manual_seed(0))
    idx = sk.pattern_index(pats.reshape(-1, m)).reshape(shots, n).cpu()
    for b in range(n):
        emp = torch.bincount(idx[:, b], minlength=sk.S).double() / shots
        assert float((emp - marg[b]).abs().max()) < 4 * float((marg[b] * (1 - marg[b]) / shots).sqrt().max())
    assert SequenceKernel(_sequence_case(k, n, m), m, indist=[1, 1, 1]).indist is None
    assert sk.sample_diag(U, 4).shape == (4, n, m)
    with pytest.raises(ValueError):
        SequenceKernel(_sequence_case(k, n, m), m, indist=[0.5, 0.5])


def test_sequence_indist_two_bins_against_the_unrolled_species_oracle():
    """Two emitters, two bins: the 4-photon sequence through U (+) U as a species-resolved
    permanent sum over the expanded 2-bin chain, with each emitter's photon independently in
    the common mode (x_a) — photons of different bins never meet, so only inter-emitter
    distinguishability within a bin shows, which is what the kernel computes."""
    from photonic_engines.sequence import SequenceKernel
    k, n, m, xs = 2, 2, 4, [0.8, 0.5]
    ems = _sequence_case(k, n, m)
    U = haar_unitary(m, 7)
    sk = SequenceKernel(ems, m, indist=xs)
    # the 2-bin chain on the product register: slots (bin 0: emitters 0, 1), (bin 1: emitters 0, 1)
    st0, st1 = sk.bin_mps(0), sk.bin_mps(1)
    ops = np.concatenate([st0.ops, st1.ops])                                             # [4, L, chi, chi]
    modes = tuple(st0.modes) + tuple(tuple(v + m for v in mk) for mk in st1.modes)
    chain = EmitterMPS(ops=ops, modes=modes, s0=st0.s0, m=2 * m, final=None)
    from photonic_engines.clifford.emitter_reference import coefficient_tensor
    c = coefficient_tensor(chain)
    Ubig = torch.block_diag(U, U)
    x_slot = xs + xs
    ref = {}
    for f in range(chain.chi):
        comps = []
        for x in itertools.product(range(2), repeat=4):
            a = complex(c[(*x, f)])
            if abs(a) > 1e-14:
                comps.append((a, [modes[j][x[j]] for j in range(4)]))
        for good in itertools.product([True, False], repeat=4):
            w = math.prod(x_slot[j] if good[j] else 1 - x_slot[j] for j in range(4))
            sp = [0 if good[j] else 10 + j for j in range(4)]
            for t, p in species_probs(Ubig, comps, sp).items():
                ref[t] = ref.get(t, 0.0) + w * p
    got = {}
    for i in range(sk.S):
        for j in range(sk.S):
            t = tuple(int(v) for v in sk.patterns[i]) + tuple(int(v) for v in sk.patterns[j])
            got[t] = float(sk.sequence_probability(U, np.array([i, j])))
    assert max(abs(got.get(t, 0.0) - ref.get(t, 0.0)) for t in set(got) | set(ref)) < 1e-12
    assert abs(sum(got.values()) - 1) < 1e-12


def test_sequence_sample_diag_under_indist():
    """`sample_diag` with per-emitter purities: lone photons routed, the register updated with the
    good emitters' Laplace amplitudes times the lone photons' U factors. Bin marginals and the
    joint of bins 0, 1 (which sees the register update) against the strong side; the
    feed-forward policies still run."""
    from photonic_engines.sequence import SequenceKernel
    k, n, m, xs = 3, 3, 6, [0.9, 0.6, 0.8]
    U = haar_unitary(m, 2)
    sk = SequenceKernel(_sequence_case(k, n, m), m, indist=xs)
    marg, J = sk.bin_marginals(U), sk.joint(U, [0, 1])
    shots = 20000
    pats, psi = sk.sample_diag(U, shots, generator=torch.Generator().manual_seed(0), return_register=True)
    assert pats.shape == (shots, n, m) and (pats.sum(2) == k).all() and float((psi.norm(dim=1) - 1).abs().max()) < 1e-12
    idx = sk.pattern_index(pats.reshape(-1, m)).reshape(shots, n).cpu()
    for b in range(n):
        emp = torch.bincount(idx[:, b], minlength=sk.S).double() / shots
        assert float((emp - marg[b]).abs().max()) < 4 * float((marg[b] * (1 - marg[b]) / shots).sqrt().max())
    ones = torch.ones(shots, dtype=torch.float64)
    Jemp = torch.zeros(sk.S, sk.S, dtype=torch.float64).index_put_((idx[:, 0], idx[:, 1]), ones, accumulate=True) / shots
    assert float(0.5 * (Jemp - J).abs().sum()) < 1.5 * float(0.5 * math.sqrt(2 / math.pi) * (J * (1 - J) / shots).sqrt().sum())
    U2 = haar_unitary(m, 4)
    route = lambda b, c, psi: torch.stack([U if int(r[0]) else U2 for r in c])
    p = sk.sample_diag(U, 50, generator=torch.Generator().manual_seed(1), policy=route)
    assert p.shape == (50, n, m)
    G = torch.eye(2, dtype=C128)
    p = sk.sample_diag(U, 50, generator=torch.Generator().manual_seed(1), policy=lambda b, c, psi: G.expand(c.shape[0], k, 2, 2))
    assert p.shape == (50, n, m)


def test_product_sample_indist_matches_the_strong_side():
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import product_sample
    from photonic_engines.slos import ProductSLOSKernel
    st = ProductSuperposition.sbs(4, 6)
    U = haar_unitary(6, 3)
    shots = 6000
    for kw in ({"indist": 0.6}, {"indist": 0.6, "loss": 0.8}, {"indist": 0.0}):
        ref = ProductSLOSKernel(st, dtype=C128).probs(U, **kw).as_dict()
        occ = product_sample(st, U, shots, generator=torch.Generator().manual_seed(0), **kw)
        assert occ.shape == (shots, 6) and (occ.sum(1) <= 4).all()
        assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)
    with pytest.raises(ValueError):
        product_sample(st, U, 2, indist=0.5, photons=2)
