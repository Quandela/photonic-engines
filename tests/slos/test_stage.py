"""StageKernel (an emitter chain next to Fock inputs) against an independent permanent oracle: every label word of
the chain is expanded, the joint input (Fock occupation + the word's photons) goes through coherent permanents. Plus
the dispatch to EmitterSLOSKernel (bit-identical), batch = per-input calls, completeness, chunking, batched U,
gradients, the photons-only stage and the refusals."""
import math
from itertools import product

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, StageChain, spin_gate
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.slos.reference import _modes_from_occupation, permanent, reference_probs
from photonic_engines.slos.stage import StageKernel

C128 = torch.complex128


def _isometry_blocks(L, chi, rng):
    """L operators chi x chi with sum_l M_l^dag M_l = I."""
    z = rng.normal(size=(L * chi, chi)) + 1j * rng.normal(size=(L * chi, chi))
    q, _ = np.linalg.qr(z)
    return q.reshape(L, chi, chi)


def _chain(counts, chi, m, seed, final=False):
    """Random isometric (ragged) single-photon chain on the first sum(counts) modes of a random permutation of m."""
    rng = np.random.default_rng(seed)
    perm = [int(x) for x in rng.permutation(m)]
    ops = [_isometry_blocks(c, chi, rng) for c in counts]
    modes, i = [], 0
    for c in counts:
        modes.append(tuple(perm[i:i + c]))
        i += c
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    f = (rng.normal(size=chi) + 1j * rng.normal(size=chi)) if final else None
    return StageChain.from_mps(EmitterMPS.ragged(ops, modes, s0 / np.linalg.norm(s0), m=m, final=f))


def _oracle_kraus(chain, s, U, patterns):
    """Physical K_t[f, i] = sum_x <f| M_x |i> perm(U[t, s + occ(x)]) / sqrt(s! t!), from the label words."""
    chi = chain.chi
    K = torch.zeros((len(patterns), chi, chi), dtype=C128)
    s_fact = math.prod(math.factorial(int(v)) for v in s)
    for x in product(*(range(L) for L in chain.labels)):
        M = np.eye(chi, dtype=complex)
        occ = np.array(s, dtype=np.int64)
        for k, lab in enumerate(x):
            M = chain.ops[k, lab] @ M
            occ = occ + np.asarray(chain.words[k][lab]) @ np.eye(chain.m, dtype=np.int64)[list(chain.event_modes[k])]
        if not np.any(M):
            continue
        cols = _modes_from_occupation(occ)
        for it, t in enumerate(patterns):
            t = [int(v) for v in t]
            if sum(t) != len(cols):
                continue
            a = permanent(U[_modes_from_occupation(t)][:, cols]).item()
            K[it] += torch.as_tensor(M, dtype=C128) * a / math.sqrt(s_fact * math.prod(math.factorial(v) for v in t))
    return K


INPUTS = [
    [[1, 0, 0, 0, 0, 0, 0, 0]],                       # one heralded photon
    [[0, 2, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],   # N = 2, 2, 1 (two groups)
]


@pytest.mark.parametrize("counts,chi", [((2, 2), 2), ((1, 2, 1), 2), ((2, 3), 3)])
@pytest.mark.parametrize("inputs", INPUTS)
def test_kraus_matches_permanent_oracle(counts, chi, inputs):
    ch = _chain(counts, chi, m=8, seed=sum(counts) + chi)
    U = haar_unitary(8, seed=chi + len(counts))
    groups = StageKernel(ch, inputs=inputs).kraus(U)
    seen = []
    for g in groups:
        for row, b in enumerate(g.inputs.tolist()):
            seen.append(b)
            ref = _oracle_kraus(ch, inputs[b], U, g.patterns.tolist())
            assert torch.allclose(g.physical()[row], ref, atol=1e-12, rtol=0)
    assert sorted(seen) == list(range(len(inputs)))


def test_emission_into_an_occupied_mode():
    """The chain emits into modes the Fock input occupies: the creation-operator sqrt factors (spec R3)."""
    rng = np.random.default_rng(4)
    ops = [_isometry_blocks(2, 2, rng), _isometry_blocks(2, 2, rng)]
    ch = StageChain.from_mps(EmitterMPS.ragged(ops, [(0, 1), (2, 3)], (1, 0), m=5))
    U = haar_unitary(5, seed=9)
    inputs = [[2, 1, 0, 0, 0], [0, 0, 3, 0, 1]]
    for g in StageKernel(ch, inputs=inputs).kraus(U):
        for row, b in enumerate(g.inputs.tolist()):
            ref = _oracle_kraus(ch, inputs[b], U, g.patterns.tolist())
            assert torch.allclose(g.physical()[row], ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("final", [False, True])
def test_vacuum_dispatches_to_emitter_kernel_bit_identically(final):
    ch = _chain((2, 2, 2), 2, m=7, seed=5, final=final)
    U = haar_unitary(7, seed=3)
    Us = torch.stack([U, haar_unitary(7, seed=4)])
    for inputs in (None, [[0] * 7]):
        sk = StageKernel(ch, inputs=inputs, dtype=C128)
        ek = EmitterSLOSKernel(ch.to_mps(), dtype=C128)
        assert sk._dispatch is not None
        for u in (U, Us):
            (gk,), (gp,), (gb,) = sk.kraus(u), sk.probs(u), sk.bond_amplitudes(u)
            assert torch.equal(gk.value.squeeze(-4), ek.kraus(u))
            assert torch.equal(gp.value.squeeze(-2), ek.probs(u))
            assert torch.equal(gb.value.squeeze(-3), ek.bond_amplitudes(u))
        assert torch.equal(gk.patterns, torch.as_tensor(ek.layers.final_occupations()))
        assert (gk.kind, gp.kind, gb.kind) == ("kraus", "probs", "bond") and gk.prob_norms.dtype == torch.float64
        Kp = sk.kraus(U)[0].physical()[0]                                      # complete chain, public fields
        assert torch.allclose(torch.einsum("sfi,sfj->ij", Kp.conj(), Kp), torch.eye(2, dtype=C128), atol=1e-13)
        assert torch.equal(sk.probs(U)[0].physical(), sk.probs(U)[0].value)
    # a vacuum row inside a batch takes the general path (no dispatch): the same operators
    (g0, _) = StageKernel(ch, inputs=[[0] * 7, [1] + [0] * 6]).kraus(U)
    assert g0.inputs.tolist() == [0]
    assert torch.allclose(g0.value[0], EmitterSLOSKernel(ch.to_mps(), dtype=C128).kraus(U), atol=1e-15, rtol=0)


def test_batch_equals_per_input_calls_and_completeness():
    ch = _chain((2, 1, 2), 2, m=8, seed=7)
    U = haar_unitary(8, seed=11)
    # inputs on the modes the chain does not emit into: then the emission is an isometry and the stage complete
    # (an emission into an occupied mode multiplies the norm by s_p + 1, test_emission_into_an_occupied_mode)
    a, b, c = sorted(set(range(8)) - {p for pk in ch.event_modes for p in pk})
    inputs = [np.eye(8, dtype=int)[a] + np.eye(8, dtype=int)[c], np.eye(8, dtype=int)[b], 2 * np.eye(8, dtype=int)[a],
              np.eye(8, dtype=int)[c], np.eye(8, dtype=int)[a] + np.eye(8, dtype=int)[b] + np.eye(8, dtype=int)[c]]
    inputs = [list(map(int, r)) for r in inputs]
    sk = StageKernel(ch, inputs=inputs)
    groups, pgroups = sk.kraus(U), sk.probs(U)
    assert [g.patterns.sum(1)[0].item() for g in groups] == [1 + 3, 2 + 3, 3 + 3]
    for g, pg in zip(groups, pgroups):
        for row, b in enumerate(g.inputs.tolist()):
            (single,) = StageKernel(ch, inputs=[inputs[b]]).kraus(U)
            assert torch.equal(single.value[0], g.value[row])
            # complete chain, unitary U: sum_t K_t^dag K_t = 1 for every input
            # (from the public fields, as a caller does: StageGroup.physical(), prob_norms in float64)
            assert g.prob_norms.dtype == torch.float64
            Kp = g.physical()[row]
            assert torch.allclose(torch.einsum("sfi,sfj->ij", Kp.conj(), Kp), torch.eye(2, dtype=C128), atol=1e-13)
            # traced probabilities from s0 = the Kraus action on s0
            s0 = torch.as_tensor(ch.s0, dtype=C128)
            p = (torch.einsum("sfi,i->sf", Kp, s0).abs() ** 2).sum(1)
            assert torch.allclose(pg.value[row], p, atol=1e-13, rtol=0)
            assert abs(float(pg.value[row].sum()) - 1) < 1e-12


def test_chunk_and_batched_unitaries():
    ch = _chain((2, 2), 2, m=6, seed=8, final=True)
    inputs = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]]
    sk = StageKernel(ch, inputs=inputs)
    Us = torch.stack([haar_unitary(6, seed=s) for s in (1, 2, 3)])
    full = sk.kraus(Us)
    for g, gc in zip(full, sk.kraus(Us, chunk=1)):
        assert torch.equal(g.value, gc.value)
    for b in range(3):
        for g, g1 in zip(full, sk.kraus(Us[b])):
            assert torch.equal(g.value[b], g1.value)
        for g, g1 in zip(sk.probs(Us), sk.probs(Us[b])):                      # the projection's einsum may
            assert torch.allclose(g.value[b], g1.value, atol=1e-15, rtol=1e-14)  # reduce in another order


def test_photons_only_stage():
    """StageChain.empty: no register (chi = 1); the Kraus 'operators' are the Fock amplitudes."""
    U = haar_unitary(5, seed=2)
    inputs = [[1, 1, 0, 0, 0], [2, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    groups = StageKernel(StageChain.empty(5), inputs=inputs).probs(U)
    for g in groups:
        for row, b in enumerate(g.inputs.tolist()):
            N = sum(inputs[b])
            if N == 0:
                assert torch.allclose(g.value[row], torch.ones(1, dtype=torch.float64))
                continue
            ref = reference_probs(U, N, 5, inputs[b])
            assert torch.allclose(g.value[row], torch.as_tensor(ref, dtype=torch.float64), atol=1e-12, rtol=0)


def test_gradients_against_finite_differences():
    ch = _chain((2, 2), 2, m=5, seed=12, final=True)
    inputs = [[1, 0, 0, 0, 1], [0, 0, 1, 0, 0]]
    U = haar_unitary(5, seed=13).requires_grad_(True)
    ops = torch.as_tensor(ch.ops, dtype=C128).requires_grad_(True)
    groups = StageKernel(ch, inputs=inputs, ops_tensor=ops).probs(U)
    w = [torch.linspace(0.5, 2.0, g.value.shape[-1], dtype=torch.float64) for g in groups]
    loss = sum((g.value * wi).sum() for g, wi in zip(groups, w))
    loss.backward()
    eps = 1e-6
    with torch.no_grad():
        def f(o, u):
            return sum((g.value * wi).sum() for g, wi in zip(StageKernel(ch, inputs=inputs, ops_tensor=o).probs(u), w))

        o2 = ops.detach().clone(); o2[1, 0, 1, 0] += eps
        assert abs(float(ops.grad[1, 0, 1, 0].real) - float((f(o2, U.detach()) - loss.detach()) / eps)) < 1e-4
        u2 = U.detach().clone(); u2[2, 0] += eps
        assert abs(float(U.grad[2, 0].real) - float((f(ops.detach(), u2) - loss.detach()) / eps)) < 1e-4


def test_stage_chain_model_and_refusals():
    rng = np.random.default_rng(1)
    st = EmitterMPS.caterpillar(3, [spin_gate(math.pi / 2, 0.2 * i) for i in range(3)])
    ch = StageChain.from_mps(st)
    assert ch.is_single_photon and ch.photons_max == 3 and ch.labels == (2, 2, 2)
    back = ch.to_mps()
    assert np.array_equal(back.ops, st.ops) and back.modes == st.modes and back.m == st.m
    # two labels of one event into the same mode merge into one label (operators add)
    o = _isometry_blocks(2, 2, rng)
    merged = StageChain(ops=o[None], event_modes=((0, 1),), s0=(1, 0), m=3, words=(np.array([[1, 0], [1, 0]]),))
    mm = merged.to_mps()
    assert mm.modes == ((0,),) and np.allclose(mm.ops[0, 0], o[0] + o[1])
    U = haar_unitary(3, seed=5)
    (g,) = StageKernel(merged, inputs=[[0, 0, 1]]).kraus(U)
    assert torch.allclose(g.physical()[0], _oracle_kraus(merged, [0, 0, 1], U, g.patterns.tolist()),
                          atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="distinct"):
        StageChain(ops=o[None], event_modes=((0, 0),), s0=(1, 0), m=3)
    with pytest.raises(ValueError, match="must be zero"):
        StageChain(ops=o[None], event_modes=((0,),), s0=(1, 0), m=3)
    two_photon = StageChain(ops=o[None], event_modes=((0, 1),), s0=(1, 0), m=3, words=(np.array([[2, 0], [0, 1]]),))
    assert not two_photon.is_single_photon and two_photon.photons_max == 2
    assert two_photon.photon_numbers == (2, 1)                                  # multiphoton words: test_stage_words
    with pytest.raises(ValueError, match="env records"):
        StageChain(ops=o[None], event_modes=((0, 1),), s0=(1, 0), m=3, env=(np.array([0, 1]),)).to_mps()
    with pytest.raises(ValueError, match="detected modes"):                   # partial detection: test_stage_carry
        StageKernel(ch, detected=[0, 1, ch.m])
    with pytest.raises(ValueError, match="inputs"):
        StageKernel(ch, inputs=[[1, 0]])
    with pytest.raises(ValueError, match="merged"):
        StageKernel(merged, inputs=[[0, 0, 1]], ops_tensor=torch.as_tensor(merged.ops))
