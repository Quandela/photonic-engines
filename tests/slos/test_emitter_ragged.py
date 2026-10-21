"""Ragged EmitterMPS chains (a label count L_k per emission event) through EmitterSLOSKernel, against two
oracles: the dummy-mode padding workaround (the same chain padded to L_max labels with zero operators on extra
modes, outputs restricted to the patterns with no photon there) and the expanded L_1...L_n components through
coherent permanents. Plus validation, fold / expand, loss, the x-model Kraus set, gradients and the samplers'
refusal."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.slos.reference import _modes_from_occupation, permanent

C128 = torch.complex128


def _isometry_blocks(L, chi, rng):
    """L operators chi x chi with sum_l M_l^dag M_l = I (blocks of a random isometry)."""
    z = rng.normal(size=(L * chi, chi)) + 1j * rng.normal(size=(L * chi, chi))
    q, _ = np.linalg.qr(z)
    return q.reshape(L, chi, chi)


def _ragged_chain(counts, chi, seed, final=False, spare=1):
    """Random isometric ragged chain, event k with counts[k] labels on shuffled distinct modes."""
    rng = np.random.default_rng(seed)
    ops = [_isometry_blocks(c, chi, rng) for c in counts]
    m = sum(counts) + spare
    perm = [int(x) for x in rng.permutation(m)]
    modes, i = [], 0
    for c in counts:
        modes.append(tuple(perm[i:i + c]))
        i += c
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    f = (rng.normal(size=chi) + 1j * rng.normal(size=chi)) if final else None
    return EmitterMPS.ragged(ops, modes, s0 / np.linalg.norm(s0), m=m, final=f)


def _padded_twin(st):
    """The padding workaround: every event padded to L_max labels, the pads on fresh dummy modes m, m+1, ..."""
    L, extra, modes = st.num_labels, st.m, []
    for k in range(st.n):
        pads = L - st.labels[k]
        modes.append(tuple(st.modes[k]) + tuple(range(extra, extra + pads)))
        extra += pads
    return EmitterMPS(ops=st.ops, modes=tuple(modes), s0=st.s0, m=extra, final=st.final)


def _embed(U, m2):
    out = torch.eye(m2, dtype=C128)
    out[:U.shape[0], :U.shape[0]] = U
    return out


def _restrict(k_pad, m):
    """Indices of the padded kernel's output patterns with no photon on the dummy modes, keyed by the pattern."""
    occ = k_pad.layers.final_occupations()
    return {tuple(int(x) for x in row[:m]): i for i, row in enumerate(occ) if not row[m:].any()}


def _physical(k, T):
    """Deferred -> physical normalisation along the output axis (axis 0 of T)."""
    norm = torch.sqrt(torch.as_tensor(k.layers.output_prob_norms(), dtype=torch.float64)).to(T.dtype)
    return T * norm.view(-1, *([1] * (T.dim() - 1)))


def _permanent_kraus(st, U, k):
    """K[t][f, i] = <t| chain from s0 = e_i, projected on e_f>, physical normalisation, from expand()."""
    occ = k.layers.final_occupations()
    out = torch.zeros((len(occ), st.chi, st.chi), dtype=C128)
    for i in range(st.chi):
        for f in range(st.chi):
            e_i, e_f = np.eye(st.chi)[i], np.eye(st.chi)[f]
            sub = EmitterMPS(ops=st.ops, modes=st.modes, s0=e_i, m=st.m, final=e_f)
            for sup in sub.expand():
                for c, w in zip(sup.components, sup.amplitudes):
                    for idx, t in enumerate(occ):
                        t = tuple(int(x) for x in t)
                        if sum(t) != st.n:
                            continue
                        a = permanent(U[_modes_from_occupation(t)][:, c.modes()]).item()
                        out[idx, f, i] += w * a / math.sqrt(math.prod(math.factorial(x) for x in t))
    return out


CASES = [((1, 3, 2), 2), ((2, 1, 1, 2), 2), ((3, 1, 2), 3), ((1, 2), 4)]


def test_validation_and_helpers():
    st = _ragged_chain((1, 3, 2), 2, seed=1)
    assert st.is_ragged and st.labels == (1, 3, 2) and st.num_labels == 3
    assert not np.any(st.ops[0, 1:]) and not np.any(st.ops[2, 2:])
    mp = st.modes_padded()
    assert mp.shape == (3, 3) and mp.dtype == np.int64
    assert list(mp[0]) == [st.modes[0][0]] * 3 and list(mp[2]) == [*st.modes[2], st.modes[2][0]]
    assert st.label_mask().tolist() == [[True, False, False], [True, True, True], [True, True, False]]
    # a uniform chain through `ragged` is today's object
    rng = np.random.default_rng(2)
    ops = [_isometry_blocks(2, 2, rng) for _ in range(3)]
    u = EmitterMPS.ragged(ops, [(0, 1), (2, 3), (4, 5)], (1, 0))
    ref = EmitterMPS(ops=np.stack(ops), modes=((0, 1), (2, 3), (4, 5)), s0=(1, 0), m=6)
    assert not u.is_ragged and not ref.is_ragged and u.labels == (2, 2, 2)
    assert np.array_equal(u.ops, ref.ops) and u.modes == ref.modes and u.m == ref.m
    assert np.array_equal(ref.modes_padded(), np.array(ref.modes)) and ref.label_mask().all()
    with pytest.raises(ValueError, match=r"ops\[0, 2\] must be zero"):
        bad = np.array(st.ops); bad[0, 2, 0, 0] = 1e-3
        EmitterMPS(ops=bad, modes=st.modes, s0=st.s0, m=st.m)
    with pytest.raises(ValueError, match="1 <= L_k <= L"):
        EmitterMPS(ops=st.ops, modes=((), st.modes[1], st.modes[2]), s0=st.s0, m=st.m)
    with pytest.raises(ValueError, match="1 <= L_k <= L"):
        EmitterMPS(ops=st.ops, modes=(st.modes[1] + (st.m - 1,), st.modes[1], st.modes[2]), s0=st.s0, m=st.m)
    with pytest.raises(ValueError, match="distinct"):
        EmitterMPS(ops=st.ops, modes=(st.modes[1][:1], st.modes[1], st.modes[2]), s0=st.s0, m=st.m)
    with pytest.raises(ValueError, match="one mode per label"):
        EmitterMPS.ragged([st.ops[0, :1], st.ops[1]], [(0, 1), (2, 3, 4)], st.s0)
    # the kernel refuses an ops_tensor with a non-zero padded slot
    bad_t = torch.as_tensor(st.ops).clone(); bad_t[0, 1, 1, 1] = 1
    with pytest.raises(ValueError, match="padded slots"):
        EmitterSLOSKernel(st, dtype=C128, ops_tensor=bad_t)


def test_expand_and_fold():
    st = _ragged_chain((1, 3, 2), 2, seed=3, final=True)
    comps = sum(len(s.occupations) for s in st.expand())
    assert comps == 1 * 3 * 2                                   # one component per label word (none vanish here)
    red, tail = st.fold({1: st.ops[1, 2]})
    assert red.labels == (1, 2) and red.modes == (st.modes[0], st.modes[2])
    red0, _ = st.fold({0: st.ops[0, 0]})                       # a folded d = 1 slot
    assert red0.labels == (3, 2) and red0.is_ragged
    red2, _ = st.fold({2: st.ops[2, 0]})
    assert red2.labels == (1, 3) and red2.is_ragged


@pytest.mark.parametrize("counts,chi", CASES)
@pytest.mark.parametrize("final", [False, True])
def test_matches_padding_workaround_and_permanents(counts, chi, final):
    st = _ragged_chain(counts, chi, seed=sum(counts) * 7 + chi, final=final)
    U = haar_unitary(st.m, seed=len(counts) + chi)
    k = EmitterSLOSKernel(st, dtype=C128)
    pad = _padded_twin(st)
    kp = EmitterSLOSKernel(pad, dtype=C128)
    Up = _embed(U, pad.m)
    sel = _restrict(kp, st.m)
    idx = torch.tensor([sel[tuple(int(x) for x in row)] for row in k.layers.final_occupations()])
    assert len(sel) == k.output_size()
    # probabilities, bond amplitudes, Kraus operators vs the padding workaround
    assert torch.allclose(k.probs(U), kp.probs(Up)[idx], atol=1e-12, rtol=0)
    assert torch.allclose(k.bond_amplitudes(U), kp.bond_amplitudes(Up)[:, idx], atol=1e-12, rtol=0)
    K = k.kraus(U)
    assert torch.allclose(K, kp.kraus(Up)[idx], atol=1e-12, rtol=0)
    assert torch.allclose(k.kraus(U, chunk=1), K, atol=0, rtol=0)
    if final:
        assert torch.allclose(k.amplitudes(U), kp.amplitudes(Up)[idx], atol=1e-12, rtol=0)
    # vs the expanded label words through coherent permanents
    assert torch.allclose(_physical(k, K), _permanent_kraus(st, U, k), atol=1e-12, rtol=0)
    # isometric emission, unitary U, full detection: trace preserving
    Kp = _physical(k, K)
    assert torch.allclose(torch.einsum("sfi,sfj->ij", Kp.conj(), Kp), torch.eye(chi, dtype=C128), atol=1e-12)


def test_product_register_single_photon_source_next_to_spin():
    """d = 1 source (one label, identity on the spin) and a spin (caterpillar, two labels) on one register:
    the stage that motivated R1. Traced spin: probabilities vs the padding workaround and permanents."""
    G = [spin_gate(math.pi / 2, 0.0), spin_gate(math.pi / 2, 0.7), spin_gate(math.pi / 2, 1.9)]
    up, dn = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    ops = [np.eye(2)[None], np.stack([up @ G[0], dn @ G[0]]), np.stack([up @ G[1], dn @ G[1]]),
           np.eye(2)[None], np.stack([up @ G[2], dn @ G[2]])]
    modes = [(0,), (1, 2), (3, 4), (5,), (6, 7)]
    st = EmitterMPS.ragged(ops, modes, (1, 0), m=9)
    assert st.labels == (1, 2, 2, 1, 2)
    U = haar_unitary(9, seed=21)
    k = EmitterSLOSKernel(st, dtype=C128)
    pad = _padded_twin(st)
    kp = EmitterSLOSKernel(pad, dtype=C128)
    sel = _restrict(kp, st.m)
    idx = torch.tensor([sel[tuple(int(x) for x in row)] for row in k.layers.final_occupations()])
    p = k.probs(U)
    assert torch.allclose(p, kp.probs(_embed(U, pad.m))[idx], atol=1e-12, rtol=0)
    Kp = _physical(k, k.kraus(U))
    p_perm = torch.einsum("sfi,i->sf", _permanent_kraus(st, U, k), torch.tensor([1, 0], dtype=C128)).abs().pow(2).sum(1)
    assert torch.allclose(p, p_perm, atol=1e-12, rtol=0)
    assert torch.allclose(torch.einsum("sfi,sfj->ij", Kp.conj(), Kp), torch.eye(2, dtype=C128), atol=1e-12)
    assert abs(float(p.sum()) - 1) < 1e-12


def _permanent_probs(st, U, k):
    """Traced / projected output probabilities from expand() through coherent permanents (physical normalisation)."""
    occ = k.layers.final_occupations()
    total = torch.zeros(len(occ), dtype=torch.float64)
    for sup in st.expand():
        amp = torch.zeros(len(occ), dtype=C128)
        for idx, t in enumerate(occ):
            t = tuple(int(x) for x in t)
            for c, w in zip(sup.components, sup.amplitudes):
                amp[idx] += w * permanent(U[_modes_from_occupation(t)][:, c.modes()]).item()
            amp[idx] /= math.sqrt(math.prod(math.factorial(x) for x in t))
        total += amp.abs() ** 2
    return total


def test_product_register_qutrit_next_to_qubit():
    """chi = 6 register (qutrit x qubit): the qutrit emits with three labels, the qubit with two."""
    rng = np.random.default_rng(23)

    def unitary(d):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
        return q

    g3, g2 = unitary(3), unitary(2)
    qutrit = np.stack([np.kron(np.outer(e, e) @ g3, np.eye(2)) for e in np.eye(3)])        # [3, 6, 6]
    qubit = np.stack([np.kron(np.eye(3), np.outer(e, e) @ g2) for e in np.eye(2)])         # [2, 6, 6]
    st = EmitterMPS.ragged([qutrit, qubit, qutrit], [(0, 1, 2), (3, 4), (5, 6, 7)], np.eye(6)[0], m=9)
    assert st.labels == (3, 2, 3) and st.chi == 6
    U = haar_unitary(9, seed=24)
    k = EmitterSLOSKernel(st, dtype=C128)
    pad = _padded_twin(st)
    kp = EmitterSLOSKernel(pad, dtype=C128)
    sel = _restrict(kp, st.m)
    idx = torch.tensor([sel[tuple(int(x) for x in row)] for row in k.layers.final_occupations()])
    K = k.kraus(U)
    assert torch.allclose(K, kp.kraus(_embed(U, pad.m))[idx], atol=1e-12, rtol=0)
    p = k.probs(U)
    assert torch.allclose(p, _permanent_probs(st, U, k), atol=1e-12, rtol=0)
    Kp = _physical(k, K)
    assert torch.allclose(torch.einsum("sfi,sfj->ij", Kp.conj(), Kp), torch.eye(6, dtype=C128), atol=1e-12)


def test_loss_and_x_model_match_padding_workaround():
    st = _ragged_chain((2, 1, 2), 2, seed=11)
    U = haar_unitary(st.m, seed=5)
    k = EmitterSLOSKernel(st, dtype=C128)
    pad = _padded_twin(st)
    kp = EmitterSLOSKernel(pad, dtype=C128)
    Up = _embed(U, pad.m)
    lossy, lossy_p = k.probs(U, loss=0.7), kp.probs(Up, loss=0.7)
    for j in range(st.n + 1):                                         # every surviving-photon layer
        occ_p = lossy_p.occupations(j)
        sel_j = {tuple(int(x) for x in row[:st.m]): i for i, row in enumerate(occ_p) if not row[st.m:].any()}
        idx_j = torch.tensor([sel_j[tuple(int(x) for x in row)] for row in lossy.occupations(j)])
        assert len(sel_j) == len(idx_j)
        assert torch.allclose(lossy.probs[j], lossy_p.probs[j][idx_j], atol=1e-12, rtol=0)
    # x-model: probabilities and the species-resolved channel, coarse pattern by coarse pattern
    sel = _restrict(kp, st.m)
    idx = torch.tensor([sel[tuple(int(x) for x in row)] for row in k.layers.final_occupations()])
    assert torch.allclose(k.probs(U, indist=0.8), kp.probs(Up, indist=0.8)[idx], atol=1e-12, rtol=0)
    K, coarse = k.kraus_species(U, 0.8)
    Kq, coarse_q = kp.kraus_species(Up, 0.8)
    rho = torch.tensor([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=C128)
    out = torch.zeros((k.output_size(), 2, 2), dtype=C128).index_add(0, coarse, K @ rho @ K.mH)
    out_q = torch.zeros((kp.output_size(), 2, 2), dtype=C128).index_add(0, coarse_q, Kq @ rho @ Kq.mH)
    assert torch.allclose(out, out_q[idx], atol=1e-12, rtol=0)
    assert abs(float(torch.einsum("tii->", out).real) - 1) < 1e-12
    # uniform loss: the lossless part of the thinned distribution is the lossless one scaled by eta^n
    assert torch.allclose(lossy.probs[-1], k.probs(U) * 0.7 ** st.n, atol=1e-12, rtol=0)


@pytest.mark.parametrize("final", [False, True])
def test_gradients_match_padding_workaround(final):
    st = _ragged_chain((1, 3, 2), 2, seed=13, final=final)
    pad = _padded_twin(st)
    sel = None
    grads = []
    for s in (st, pad):
        U = haar_unitary(st.m, seed=6).requires_grad_(True)
        ops = torch.as_tensor(st.ops, dtype=C128).requires_grad_(True)
        k = EmitterSLOSKernel(s, dtype=C128, ops_tensor=ops)
        p = k.probs(_embed(U, s.m) if s is pad else U)
        if sel is None:
            occ = k.layers.final_occupations()
            out = p
        else:
            look = _restrict(k, st.m)
            out = p[torch.tensor([look[tuple(int(x) for x in row)] for row in occ])]
        sel = True
        (out * torch.linspace(0.5, 2.0, len(out), dtype=torch.float64)).sum().backward()
        grads.append((U.grad, ops.grad))
    (gU, gops), (gU_p, gops_p) = grads
    assert torch.allclose(gU, gU_p, atol=1e-12, rtol=0)
    assert not gops[~torch.as_tensor(st.label_mask())].any()        # padded slots: exactly zero gradient
    m = torch.as_tensor(st.label_mask())
    assert torch.allclose(gops[m], gops_p[m], atol=1e-12, rtol=0)
    # finite difference on one real label's operator entry
    eps = 1e-6
    with torch.no_grad():
        ops = torch.as_tensor(st.ops, dtype=C128)
        U = haar_unitary(st.m, seed=6)
        ops2 = ops.clone(); ops2[1, 1, 0, 1] += eps
        w = torch.linspace(0.5, 2.0, len(occ), dtype=torch.float64)
        fd = ((EmitterSLOSKernel(st, dtype=C128, ops_tensor=ops2).probs(U) * w).sum()
              - (EmitterSLOSKernel(st, dtype=C128, ops_tensor=ops).probs(U) * w).sum()) / eps
    assert abs(float(gops[1, 1, 0, 1].real) - float(fd)) < 1e-4


def _refusals():
    """Every public entry point that refuses ragged chains — the single list to extend (or shrink, when a sampler
    learns ragged chains)."""
    from photonic_engines.clifford import emitter_sample, sample
    from photonic_engines.clifford.emitter_caterpillar import sample_caterpillar
    from photonic_engines.clifford.emitter_reference import emitter_sample_reference
    from photonic_engines.clifford.emitter_sampler import sample_torch
    from photonic_engines.sequence import BinEmitter
    from photonic_engines.slos.estimate import EmitterGurvitsEstimator
    from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
    return {
        "emitter_sample/torch": lambda st, U: emitter_sample(st, U, 4, backend="torch", device="cpu"),
        "emitter_sample/triton": lambda st, U: emitter_sample(st, U, 4, backend="triton"),
        "emitter_sample/segments": lambda st, U: emitter_sample(st, U, 4, backend="segments"),
        "sample": lambda st, U: sample(st, U, 4, backend="torch"),
        "sample_torch": lambda st, U: sample_torch(st, U, 4, device="cpu"),
        "sample_caterpillar": lambda st, U: sample_caterpillar(st, U, 4, device="cpu"),
        "emitter_sample_reference": lambda st, U: emitter_sample_reference(st, U, 4),
        "EmitterGurvitsEstimator": lambda st, U: EmitterGurvitsEstimator(st),
        "BinEmitter.from_mps": lambda st, U: BinEmitter.from_mps(st, st.m),
        "LoopRegisterKernel": lambda st, U: LoopRegisterKernel(TimeLoopCircuit(st.m, (1,)), [0] * st.m, emitter=st),
    }


@pytest.mark.parametrize("entry", list(_refusals()))
def test_entry_points_refuse_ragged_chains(entry):
    st = _ragged_chain((1, 2), 2, seed=17)
    U = haar_unitary(st.m, seed=1)
    with pytest.raises(NotImplementedError, match="ragged label counts"):
        _refusals()[entry](st, U)
