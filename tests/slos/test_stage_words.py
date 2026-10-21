"""StageKernel with labels of any photon number (R2): photon-less (empty word) and multiphoton labels by the graded
recurrence, and env records (labels with different env never interfere: one Kraus set per env word). Against an
independent permanent oracle on the expanded label words (with the 1/sqrt(w!) of each word), completeness, the
single-photon recurrence, the internal-modes kernel's `lost` / R > 1 channel (one concept), probabilities, carry-over
and gradients."""
import math
from itertools import product

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary
from test_stage import _chain, _isometry_blocks

from photonic_engines import StageChain
from photonic_engines.sequence import ModeBinEmitter, SequenceKernel
from photonic_engines.slos.reference import _modes_from_occupation, permanent
from photonic_engines.slos.stage import StageKernel

C128 = torch.complex128


def _word_chain(events, chi, m, seed, env=None, s0=None):
    """events: per event (modes P_k, words [L_k, |P_k|]); complete operators: one isometry over the labels."""
    rng = np.random.default_rng(seed)
    L = max(len(w) for _, w in events)
    ops = np.zeros((len(events), L, chi, chi), dtype=complex)
    for k, (_, w) in enumerate(events):
        ops[k, :len(w)] = _isometry_blocks(len(w), chi, rng)
    if s0 is None:
        s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
        s0 = s0 / np.linalg.norm(s0)
    return StageChain(ops=ops, event_modes=tuple(P for P, _ in events), s0=s0, m=m,
                      words=tuple(np.asarray(w) for _, w in events), env=env)


def _oracle(chain, s, U, patterns, envw):
    """Physical K_t[f, i] = sum_x <f| M_x |i> perm(U[t, s + occ(x)]) / sqrt(s! t! prod w!) over the label choices x
    whose env matches the env word envw (one label per event)."""
    chi = chain.chi
    K = torch.zeros((len(patterns), chi, chi), dtype=C128)
    s_fact = math.prod(math.factorial(int(v)) for v in s)
    labels = [[lab for lab in range(len(w)) if chain.env[k][lab] == envw[k]] for k, w in enumerate(chain.words)]
    for x in product(*labels):
        M = np.eye(chi, dtype=complex)
        occ = np.array(s, dtype=np.int64)
        w_fact = 1
        for k, lab in enumerate(x):
            M = chain.ops[k, lab] @ M
            w = chain.words[k][lab]
            w_fact *= math.prod(math.factorial(int(c)) for c in w)
            occ = occ + np.asarray(w) @ np.eye(chain.m, dtype=np.int64)[list(chain.event_modes[k])]
        if not np.any(M):
            continue
        cols = _modes_from_occupation(occ)
        for it, t in enumerate(patterns):
            t = [int(v) for v in t]
            if sum(t) != len(cols):
                continue
            a = 1.0 if not cols else permanent(U[_modes_from_occupation(t)][:, cols]).item()
            t_fact = math.prod(math.factorial(v) for v in t)
            K[it] += torch.as_tensor(M, dtype=C128) * a / math.sqrt(s_fact * t_fact * w_fact)
    return K


# a vacuum, two single-photon and a two-photon label; then single photons and a two-mode pair
WORDS = [((0, 1), [[0, 0], [1, 0], [0, 1], [2, 0]]), ((2, 3), [[1, 0], [0, 1], [1, 1]])]
# env records: two lost operators (empty words, env 1, 2) next to the emission; R = 2 unobserved outcomes
ENV_WORDS = [((0, 1), [[1, 0], [0, 1], [0, 0], [0, 0]]), ((2, 3), [[1, 0], [0, 1], [1, 0], [0, 1]])]
ENV = (np.array([0, 0, 1, 2]), np.array([0, 0, 1, 1]))
INPUTS = [None, [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0] * 6, [0, 0, 0, 0, 0, 2]]]


def _cases():
    return {"words": _word_chain(WORDS, 2, 6, seed=1),
            "env": _word_chain(ENV_WORDS, 2, 6, seed=2, env=ENV),
            "chi3": _word_chain([((0, 1, 2), [[0, 0, 0], [3, 0, 0], [0, 1, 1]]), ((3,), [[1], [0]])], 3, 6, seed=3)}


@pytest.mark.parametrize("name", ["words", "env", "chi3"])
@pytest.mark.parametrize("inputs", INPUTS)
def test_kraus_against_the_word_oracle(name, inputs):
    ch = _cases()[name]
    U = haar_unitary(6, seed=7)
    sk = StageKernel(ch, inputs=inputs)
    assert sk._dispatch is None
    E = sk.env_count
    inp = [[0] * 6] if inputs is None else inputs
    for g in sk.kraus(U):
        Kp = g.physical()
        if E > 1:
            assert g.envs.shape == (E, ch.n)
        for row, b in enumerate(g.inputs.tolist()):
            for e in range(E):
                got = Kp[e, row] if E > 1 else Kp[row]
                ref = _oracle(ch, inp[b], U, g.patterns.tolist(), ch.env_words[e])
                assert torch.allclose(got, ref, atol=1e-12, rtol=0)
        # every reachable photon number, largest first
        tot = g.patterns.sum(1).tolist()
        N = sum(inp[g.inputs[0]])
        assert sorted(set(tot), reverse=True) == [N + T for T in ch.photon_numbers]
        assert tot == sorted(tot, reverse=True)


@pytest.mark.parametrize("name", ["words", "env", "chi3"])
def test_completeness_over_patterns_and_env_words(name):
    ch = _cases()[name]
    assert ch.is_complete()
    U = haar_unitary(6, seed=8)
    for g in StageKernel(ch, inputs=[[0] * 6, [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 2]]).kraus(U):
        Kp = g.physical()
        if g.envs is None:
            Kp = Kp[None]
        for row in range(Kp.shape[1]):
            G = torch.einsum("esfi,esfj->ij", Kp[:, row].conj(), Kp[:, row])
            assert torch.allclose(G, torch.eye(ch.chi, dtype=C128), atol=1e-13, rtol=0)


def test_single_photon_chain_through_the_graded_recurrence():
    """A single-photon chain with an extra zero-operator vacuum label takes the graded path: its n-photon operators
    equal the single-photon kernel's (R3), the other photon numbers carry nothing."""
    base = _chain((2, 2, 1), 2, m=7, seed=4)
    ops = np.zeros((base.n, base.ops.shape[1] + 1, 2, 2), dtype=complex)
    for k, w in enumerate(base.words):                 # the real labels first, the zero vacuum label after them
        ops[k, :len(w)] = base.ops[k, :len(w)]
    words = tuple(np.concatenate([w, np.zeros((1, w.shape[1]), dtype=np.int64)]) for w in base.words)
    graded = StageChain(ops=ops, event_modes=base.event_modes, s0=base.s0, m=7, words=words)
    U = haar_unitary(7, seed=5)
    inputs = [[0] * 7, [0, 0, 0, 0, 0, 0, 0]]
    free = sorted(set(range(7)) - {p for pk in base.event_modes for p in pk})
    inputs[1][free[0]] = 1
    sk = StageKernel(graded, inputs=inputs)
    assert sk._graded and graded.photon_numbers == (3, 2, 1, 0)
    for gr, gb in zip(sk.kraus(U), StageKernel(base, inputs=inputs).kraus(U)):
        top = gr.patterns.sum(1) == gb.patterns.sum(1)[0]
        assert torch.equal(gr.patterns[top], gb.patterns)
        assert torch.allclose(gr.value[:, top], gb.value, atol=1e-15, rtol=0)
        assert not gr.value[:, ~top].any()


def _r_emitter(L, R, Q, d, modes, seed):
    """A complete K = 1 internal-mode emitter with R unobserved outcomes and Q lost operators (ops [1, L, 1, R, d, d])."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=((L * R + Q) * d, d)) + 1j * rng.normal(size=((L * R + Q) * d, d))
    q, _ = np.linalg.qr(z)
    A = q[:L * R * d].reshape(R, L, d, d).transpose(1, 0, 2, 3)             # [L, R, d, d]
    ops = A[None, :, None]                                                  # [1, L, 1, R, d, d]
    lost = q[L * R * d:].reshape(1, Q, d, d) if Q else None
    s0 = np.zeros(d, dtype=complex)
    s0[0] = 1
    return ModeBinEmitter(ops[..., 0, :, :] if R == 1 else ops, modes, s0, lost=lost)


@pytest.mark.parametrize("R,Q", [(1, 1), (1, 2), (2, 0), (2, 1)])
def test_same_channel_as_the_internal_mode_kernel(R, Q):
    """One concept: an empty-word label with its own env = `ModeBinEmitter.lost`, env r = an unobserved outcome. A
    two-emitter tick through SequenceKernel (route F) and through StageKernel (`StageChain.from_bin_emitters`): the
    same superoperator per detected pattern (0..2 photons)."""
    m = 5
    ems = [_r_emitter(2, R, Q, 2, (0, 1), seed=10 + R + Q), _r_emitter(2, R, Q, 2, (2, 3), seed=20 + R + Q)]
    mk = SequenceKernel(ems, m)
    U = haar_unitary(m, seed=9)
    K, coarse = mk.kraus(U, 0, route="F"), mk.coarse(U, 0)
    chi = 4
    J_seq = torch.zeros((mk.S, chi * chi, chi * chi), dtype=C128).index_add(
        0, coarse, torch.einsum("qab,qcd->qacbd", K, K.conj()).reshape(-1, chi * chi, chi * chi))
    ch = StageChain.from_bin_emitters(ems, b=0, m=m)
    assert ch.is_complete() and ch.env_words.shape[0] == (R + Q) ** 2
    (g,) = StageKernel(ch).kraus(U)
    Kp = g.physical()
    Kp = Kp if g.envs is not None else Kp[None]
    J = torch.einsum("esab,escd->sacbd", Kp[:, 0], Kp[:, 0].conj()).reshape(-1, chi * chi, chi * chi)
    idx = mk.pattern_index(g.patterns.numpy())
    assert sorted(idx.tolist()) == list(range(mk.S))
    assert float((J - J_seq[torch.as_tensor(idx)]).abs().max()) < 1e-13
    with pytest.raises(NotImplementedError, match="K > 1"):
        StageChain.from_bin_emitters([ModeBinEmitter(np.zeros((1, 2, 2, 2, 2)), (0, 1), (1, 0))], m=m)


def test_probabilities_and_bond_amplitudes_sum_the_env_words():
    ch = _cases()["env"]
    U = haar_unitary(6, seed=11)
    sk = StageKernel(ch, inputs=[[0, 0, 0, 0, 1, 0]])
    (gk,), (gb,), (gp,) = sk.kraus(U), sk.bond_amplitudes(U), sk.probs(U)
    s0 = torch.as_tensor(ch.s0, dtype=C128)
    Kp = gk.physical()[:, 0]                                                # [E, S, chi, chi]
    p = (torch.einsum("esfi,i->esf", Kp, s0).abs() ** 2).sum((0, 2))
    assert torch.allclose(gp.value[0], p, atol=1e-14, rtol=0) and abs(float(p.sum()) - 1) < 1e-13
    assert torch.allclose(gb.physical()[:, 0], torch.einsum("esfi,i->efs", Kp, s0), atol=1e-14, rtol=0)
    final = np.array([0.6, 0.8j])
    chf = StageChain(ops=ch.ops, event_modes=ch.event_modes, s0=ch.s0, m=6, words=ch.words, env=ch.env, final=final)
    (gf,) = StageKernel(chf, inputs=[[0, 0, 0, 0, 1, 0]]).probs(U)
    amp = torch.einsum("f,esfi,i->es", torch.as_tensor(final, dtype=C128).conj(), Kp, s0)
    assert torch.allclose(gf.value[0], (amp.abs() ** 2).sum(0), atol=1e-14, rtol=0)


def test_carry_over_with_words_and_env():
    """R4 on an R2 chain: the blocks get an env axis, sum_e sum_q K^dag K = 1, apply = the env-summed update."""
    ch = _cases()["env"]
    U = haar_unitary(6, seed=12)
    inputs = [[0] * 6, [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    sk = StageKernel(ch, inputs=inputs, detected=(0, 2, 4))
    res = sk.blocks(U)
    E = sk.env_count
    G = sum(torch.einsum("eqai,eqaj->ij", cl.value.conj(), cl.value) for cl in res.classes)
    assert torch.allclose(G, torch.eye(len(inputs) * 2, dtype=C128), atol=1e-13, rtol=0)
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    rho = torch.as_tensor(a @ a.conj().T / np.trace(a @ a.conj().T).real, dtype=C128)
    out = sk.apply(U, rho, strict=True)
    for cb, ca in zip(res.classes, out.classes):
        assert cb.value.shape[0] == E
        ref = torch.einsum("eqab,bc,eqdc->qad", cb.value, rho, cb.value.conj())
        assert torch.allclose(ca.value, ref, atol=1e-15, rtol=0)
    assert abs(float(out.deviation)) < 1e-13


def test_gradients_through_the_graded_recurrence():
    ch = _cases()["words"]
    inputs = [[0, 0, 0, 0, 1, 0], [0] * 6]
    U = haar_unitary(6, seed=13).requires_grad_(True)
    ops = torch.as_tensor(ch.ops, dtype=C128).requires_grad_(True)

    def loss(o, u):
        gs = StageKernel(ch, inputs=inputs, ops_tensor=o).probs(u)
        return sum((g.value * torch.linspace(0.5, 2.0, g.value.shape[-1], dtype=torch.float64)).sum() for g in gs)

    f = loss(ops, U)
    f.backward()
    eps = 1e-6
    with torch.no_grad():
        o2 = ops.detach().clone(); o2[0, 3, 1, 0] += eps                    # the two-photon label
        assert abs(float(ops.grad[0, 3, 1, 0].real) - float((loss(o2, U.detach()) - f.detach()) / eps)) < 1e-4
        o2 = ops.detach().clone(); o2[0, 0, 0, 1] += eps                    # the vacuum label
        assert abs(float(ops.grad[0, 0, 0, 1].real) - float((loss(o2, U.detach()) - f.detach()) / eps)) < 1e-4
        u2 = U.detach().clone(); u2[2, 0] += eps
        assert abs(float(U.grad[2, 0].real) - float((loss(ops.detach(), u2) - f.detach()) / eps)) < 1e-4


def test_model_validation():
    ch = _cases()["env"]
    assert ch.photon_numbers == (2, 1) and ch.env_words.shape == (6, 2)        # event 2 always emits
    assert _cases()["words"].photon_numbers == (4, 3, 2, 1)
    with pytest.raises(ValueError, match="env"):
        StageChain(ops=ch.ops, event_modes=ch.event_modes, s0=ch.s0, m=6, words=ch.words, env=(np.array([0, 0]),))
    # two labels with one word and one env add coherently: isometric label blocks are then not a complete event
    same = [((0, 1), [[1, 0], [1, 0]])]
    assert not _word_chain(same, 2, 6, seed=1).is_complete()
    assert _word_chain(same, 2, 6, seed=1, env=(np.array([0, 1]),)).is_complete()    # different env: incoherent
    # 5 env values on each of 5 events: E = 5^5 recurrences, warned at construction
    many = _word_chain([((p,), [[1], [1], [0], [0], [0]]) for p in range(5)], 2, 5, seed=4,
                       env=tuple(np.arange(5) for _ in range(5)))
    assert many.env_words.shape[0] == 5 ** 5
    with pytest.warns(RuntimeWarning, match="3125 env words"):
        StageKernel(many)
