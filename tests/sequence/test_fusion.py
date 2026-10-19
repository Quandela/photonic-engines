"""Local-optics fusion blocks and programs (bench/esbs_fusion.py): the fusion blocks act on the spin register as
claimed (type-II: even-parity projector or collapse; xfusion: ZZ parity measurement or a Pauli
byproduct), the parity-measurement CNOT gadget is a unitary on the data spins for every outcome,
the trajectory sampler with per-bin unitaries agrees with the Kraus sampler, and a failed
xfusion attempt leaves the register's Schmidt spectrum untouched."""
import itertools
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))
from esbs_fusion import CircuitPolicy, bin_unitary, brickwork, fusion_block, make_emitters, pair_success, schmidt_entropy

from photonic_engines.sequence import BinEmitter, SequenceKernel

C128 = torch.complex128
I2 = np.eye(2, dtype=complex)


def _kraus_of_block(kind):
    """Kraus operators on two spins (chi = 4) of one bin: identity gates, the block on modes 0-3."""
    ems = [BinEmitter.caterpillar(1, [I2], (0, 1)), BinEmitter.caterpillar(1, [I2], (2, 3))]
    sk = SequenceKernel(ems, 4)
    U = torch.as_tensor(fusion_block(kind), dtype=C128)
    K = sk.kraus(U, 0)                                                                   # [S_bin, 4, 4]
    return sk, K


def test_typeII_even_parity_projector_or_collapse():
    sk, K = _kraus_of_block("typeII")
    pats = sk.patterns
    for t, Kt in zip(pats, K):
        d = torch.diagonal(Kt)
        assert torch.allclose(Kt, torch.diag(d), atol=1e-12)                             # diagonal in the label basis
        if d.abs().max() < 1e-12:
            continue                                                                     # pattern of zero probability
        if (t[0] + t[1]) == 1:                                                           # coincidence: diag(1, 0, 0, +-1)/2
            assert torch.allclose(d.abs(), torch.tensor([0.5, 0, 0, 0.5], dtype=torch.float64), atol=1e-12)
        else:                                                                            # bunching: one odd-parity label word survives
            assert int((d.abs() > 1e-12).sum()) == 1 and d.abs().max().item() == pytest.approx(1 / math.sqrt(2))


def test_xfusion_parity_measurement_or_pauli():
    sk, K = _kraus_of_block("xfusion")
    pats = sk.patterns
    for t, Kt in zip(pats, K):
        d = torch.diagonal(Kt)
        assert torch.allclose(Kt, torch.diag(d), atol=1e-12)
        if d.abs().max() < 1e-12:
            continue
        if (t[0] + t[1]) == 1:                                                           # ZZ parity projector, either parity
            assert torch.allclose(d.abs(), torch.tensor([0.5, 0, 0, 0.5], dtype=torch.float64), atol=1e-12) or \
                torch.allclose(d.abs(), torch.tensor([0, 0.5, 0.5, 0], dtype=torch.float64), atol=1e-12)
        else:                                                                            # proportional to a unitary (Pauli Z's)
            assert torch.allclose(d.abs(), torch.full((4,), 1 / math.sqrt(8), dtype=torch.float64), atol=1e-12)
    p_success = sum(float((Kt.conj().T @ Kt).trace().real) / 4 for t, Kt in zip(pats, K) if t[0] + t[1] == 1)
    assert p_success == pytest.approx(0.5)


def test_parity_cnot_gadget_is_unitary_and_entangling():
    """ZZ(c, a) with a = |+>, XX(a, t), then a measured in Z: for every outcome the map on (c, t)
    is proportional to a unitary of operator-Schmidt rank 2 (Beenakker et al. 2004)."""
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    Z = np.diag([1.0, -1.0]); X = np.array([[0.0, 1.0], [1.0, 0.0]])

    def kron(*ops):
        out = np.eye(1)
        for o in ops:
            out = np.kron(out, o)
        return out

    def proj(P, s):
        return (np.eye(8) + s * P) / 2

    plus = np.array([1, 1]) / math.sqrt(2)
    for s1, s2, s3 in itertools.product([1, -1], repeat=3):
        M = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            v = np.zeros(4); v[i] = 1
            st = np.einsum("ct,a->cat", v.reshape(2, 2), plus).reshape(8)
            st = proj(kron(np.eye(2), X, X), s2) @ proj(kron(Z, Z, np.eye(2)), s1) @ st
            st = proj(kron(np.eye(2), Z, np.eye(2)), s3) @ st
            M[:, i] = st.reshape(2, 2, 2)[:, 0 if s3 == 1 else 1, :].reshape(4)
        M = M / np.linalg.norm(M[:, 0])
        assert np.allclose(M.conj().T @ M, np.eye(4), atol=1e-12)
        R = M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
        assert np.linalg.matrix_rank(R, 1e-9) == 2
    del H


def test_brick_trajectories_match_kraus_sampler():
    """k = 3, T = 3, xfusion brickwork with per-bin unitaries: the last bin's pattern marginal from
    sample_diag agrees with the exact bin marginal (strong side) within shot noise."""
    k, T = 3, 3
    rng = np.random.default_rng(1)
    sk = SequenceKernel(make_emitters(k, T, "random", rng, "brick"), 2 * k)
    U = torch.stack([bin_unitary(k, brickwork(k, b), "xfusion", "x", rng) for b in range(T)])
    exact = sk.bin_marginals(U)[T - 1]                                                   # [S_bin]
    shots = 3000
    pats = sk.sample_diag(U, shots, generator=torch.Generator().manual_seed(2))
    idx = sk.pattern_index(pats[:, T - 1])
    emp = torch.bincount(idx, minlength=exact.shape[0]).double() / shots
    tvd = 0.5 * float((emp - exact.to(emp)).abs().sum())
    support = int((exact > 1e-12).sum())
    assert tvd < 3 * math.sqrt(support / shots)                                          # a few sigma of the multinomial noise
    ref = sk.sample(U, 400, generator=torch.Generator().manual_seed(3), return_register=True)[1]
    e_ref = schmidt_entropy(ref, 1)[0].mean()
    e_diag = schmidt_entropy(sk.sample_diag(U, 400, generator=torch.Generator().manual_seed(4), return_register=True)[1], 1)[0].mean()
    assert abs(float(e_ref - e_diag)) < 0.15


def test_failed_xfusion_keeps_schmidt_spectrum_typeII_destroys_it():
    """k = 3, cut 0 | 12, bin 0 fuses (0, 1) and bin 1 fuses (1, 2): streams whose bin-1 attempt
    fails keep the bin-0 entanglement across the cut under xfusion (Pauli byproduct) and lose it
    under type-II (both spins collapsed)."""
    k, T = 3, 2
    for kind, keep in (("xfusion", True), ("typeII", False)):
        rng = np.random.default_rng(3)
        sk = SequenceKernel(make_emitters(k, T, "random", rng, "brick"), 2 * k)
        U = torch.stack([bin_unitary(k, brickwork(k, b), kind, "x", rng) for b in range(T)])
        ents, fails = {}, {}

        def hook(b, psi, counts):
            ents[b] = schmidt_entropy(psi, 1)[0]
            fails[b] = ~pair_success(counts, brickwork(k, b)[0][0])

        sk.sample_diag(U, 300, generator=torch.Generator().manual_seed(5), hook=hook)
        f = fails[1]
        assert f.sum() > 20
        if keep:
            assert torch.allclose(ents[1][f], ents[0][f], atol=1e-9)
        else:
            assert float(ents[1][f].max()) < 1e-9 and float(ents[0].mean()) > 0.2


def test_circuit_policy_runs_and_entangles_data_spins():
    """k = 3 (data 0, 2; ancilla 1): the gadget completes layers, the data cut reaches entropy > 0."""
    k, T, shots = 3, 24, 64
    rng = np.random.default_rng(0)
    sk = SequenceKernel(make_emitters(k, T, "random", rng, "circuit"), 2 * k)
    U = bin_unitary(k, [], "xfusion", "x")
    pol = CircuitPolicy(k, shots, "cpu", seed=1)
    last = {}

    def hook(b, psi, counts):
        last["ent"] = schmidt_entropy(psi, 2)[0]                                        # cut {0, 1} | {2}

    sk.sample_diag(U, shots, generator=torch.Generator().manual_seed(1), hook=hook, policy=pol)
    assert float(pol.layers.mean()) > 1.0
    assert 0.35 < float(pol.successes / pol.attempts) < 0.65
    assert float(last["ent"].mean()) > 0.2
