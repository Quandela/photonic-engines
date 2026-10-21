"""StageKernel partial detection (carry-over, route B): block operators per detected pattern and the blockwise
update, against the full-output Kraus set (exact regroup), the permanent oracle, an explicit loop over the stage
simulator's formula, a brute-force carried basis, completeness, and a two-stage composition against one-shot full
detection of the unrolled problem."""
import math
from itertools import product

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary
from test_stage import _chain, _isometry_blocks, _oracle_kraus

from photonic_engines import EmitterMPS, StageChain
from photonic_engines.slos.stage import StageKernel

C128 = torch.complex128


def _free_inputs(ch, rows):
    """Fock inputs on the modes the chain does not emit into (rows of mode lists, one photon per entry)."""
    free = sorted(set(range(ch.m)) - {p for pk in ch.event_modes for p in pk})
    out = []
    for r in rows:
        occ = [0] * ch.m
        for i in r:
            occ[free[i]] += 1
        out.append(occ)
    return out


def _full_ops(sk, U):
    """{(b, pattern): physical K} from the full-output Kraus set."""
    ops = {}
    for g in sk.kraus(U):
        K = g.physical()
        for row, b in enumerate(g.inputs.tolist()):
            for s, t in enumerate(g.patterns.tolist()):
                ops[(b, tuple(t))] = K[row, s]
    return ops


def _per_q(res):
    """{q pattern: (basis occupations, value[, prob])} from a StageCarry."""
    det, car = res.detected.tolist(), res.carried.tolist()
    out = {}
    for cl in res.classes:
        for i, q in enumerate(cl.q.tolist()):
            basis = [tuple(car[j]) for j in cl.basis[i].tolist()]
            out[tuple(det[q])] = (basis, cl.value[..., i, :, :], None if cl.probs is None else cl.probs[..., i])
    return out


def _random_rho(D, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    r = a @ a.conj().T
    return torch.as_tensor(r / np.trace(r).real, dtype=C128)


@pytest.mark.parametrize("inputs", [None, "batch"])
def test_full_detection_is_the_kraus_set(inputs):
    """C = {} : one carried state (the vacuum) per q, K_q = the full-output operators, bit for bit; with the vacuum
    input (dispatch to EmitterSLOSKernel) the q are its patterns in order."""
    ch = _chain((2, 2, 1), 2, m=7, seed=3)
    U = haar_unitary(7, seed=4)
    inp = None if inputs is None else _free_inputs(ch, [[0], [], [0, 1], [1]])
    sk = StageKernel(ch, inputs=inp)
    res = sk.blocks(U)
    B = 1 if inp is None else len(inp)
    assert res.carried.tolist() == [[0] * 7]
    assert [cl.value.shape[-2] for cl in res.classes] == [2]                      # C = 1 for every q
    ops = _full_ops(sk, U)
    (cl,) = res.classes
    for i, q in enumerate(cl.q.tolist()):
        t = tuple(res.detected[q].tolist())
        Kq = cl.value[i].reshape(2, B, 2)
        for b in range(B):
            if (b, t) in ops:
                assert torch.equal(Kq[:, b], ops[(b, t)])
            else:
                assert not Kq[:, b].any()
    if inp is None:
        (g,) = sk.kraus(U)
        assert torch.equal(res.detected, g.patterns)
        assert torch.equal(cl.value.reshape(-1, 2, 2), g.physical()[0])


@pytest.mark.parametrize("counts,chi,det", [((2, 2), 2, (0, 3, 5)), ((1, 2, 2), 2, (1, 2)), ((2, 3), 3, (4,))])
def test_blocks_against_the_oracle_and_the_brute_force_basis(counts, chi, det):
    ch = _chain(counts, chi, m=7, seed=sum(counts) + chi)
    U = haar_unitary(7, seed=chi + 1)
    inputs = _free_inputs(ch, [[], [0], [1, 1], [0, 1]])
    sk = StageKernel(ch, inputs=inputs, detected=det)
    res = sk.blocks(U)
    n = ch.n
    carried = [p for p in range(7) if p not in det]
    per_q = _per_q(res)
    # every output pattern of every input group is one (q, c) with q on D and c on C
    Ns = sorted({sum(r) for r in inputs})
    expect = {}
    for N in Ns:
        for occ in product(range(N + n + 1), repeat=7):
            if sum(occ) != N + n:
                continue
            q = tuple(v if p in det else 0 for p, v in enumerate(occ))
            c = tuple(v if p not in det else 0 for p, v in enumerate(occ))
            expect.setdefault(q, set()).add(c)
    assert set(per_q) == set(expect)
    for q, (basis, Kq, _) in per_q.items():
        assert len(basis) == len(set(basis)) and set(basis) == expect[q]
        assert all(all(c[p] == 0 for p in det) for c in basis)
        Kq = Kq.reshape(len(basis), chi, len(inputs), chi)
        for j, c in enumerate(basis):
            o = [q[p] + c[p] for p in range(7)]
            for b, s in enumerate(inputs):
                if sum(o) != sum(s) + n:
                    assert not Kq[j, :, b].any()                                # wrong photon number: zero
                    continue
                ref = _oracle_kraus(ch, s, U, [o])[0]
                assert torch.allclose(Kq[j, :, b], ref, atol=1e-12, rtol=0)
    assert all(p not in det for p in carried)


def test_completeness_of_the_blocks():
    """sum_q K_q^dag K_q = 1 on (input, register): complete chain, unitary U, distinct inputs on chain-free modes."""
    ch = _chain((2, 1, 2), 2, m=8, seed=7)
    U = haar_unitary(8, seed=11)
    inputs = _free_inputs(ch, [[], [0], [2], [0, 2], [1, 1], [0, 1, 2]])
    for det in [(0, 1, 2), (5,), (), tuple(range(8))]:
        res = StageKernel(ch, inputs=inputs, detected=det).blocks(U)
        G = sum(torch.einsum("qai,qaj->ij", cl.value.conj(), cl.value) for cl in res.classes)
        assert torch.allclose(G, torch.eye(len(inputs) * 2, dtype=C128), atol=1e-13, rtol=0)


@pytest.mark.parametrize("rest", [1, 3])
def test_apply_against_the_explicit_contraction(rest):
    """apply = the stage simulator's loop rho_out[f(o_i), f(o_j)] += K_{o_i}^(a) rho[a, b] K_{o_j}^(b)dag, with the
    register = chain (x) a passive rest."""
    ch = _chain((2, 2), 2, m=6, seed=8)
    U = haar_unitary(6, seed=2)
    inputs = _free_inputs(ch, [[0], [1], [0, 1], []])
    det = (0, 2, 4)
    sk = StageKernel(ch, inputs=inputs, detected=det)
    d = 2 * rest
    rho = _random_rho(len(inputs) * d, seed=rest)
    res = sk.apply(U, rho, rest=rest)
    ops = {k: np.kron(v.numpy(), np.eye(rest)) for k, v in _full_ops(sk, U).items()}
    blocks = rho.numpy().reshape(len(inputs), d, len(inputs), d).transpose(0, 2, 1, 3)
    per_q = _per_q(res)
    total = 0.0
    for q, (basis, val, p) in per_q.items():
        ref = np.zeros((len(basis), len(basis), d, d), dtype=complex)
        for a in range(len(inputs)):
            for b in range(len(inputs)):
                for i, ci in enumerate(basis):
                    ki = ops.get((a, tuple(x + y for x, y in zip(q, ci))))
                    if ki is None:
                        continue
                    for j, cj in enumerate(basis):
                        kj = ops.get((b, tuple(x + y for x, y in zip(q, cj))))
                        if kj is None:
                            continue
                        ref[i, j] += ki @ blocks[a, b] @ kj.conj().T
        ref = ref.transpose(0, 2, 1, 3).reshape(len(basis) * d, len(basis) * d)
        assert np.abs(val.numpy() - ref).max() < 1e-14
        assert abs(float(p) - np.trace(ref).real) < 1e-14
        total += float(p)
    assert abs(total - 1) < 1e-13 and abs(float(res.deviation)) < 1e-13


def test_photons_only_stage():
    """StageChain.empty with a register that does not emit (rest = d): K_q = scalar amplitudes, no Kronecker
    product; against the scalar formula and the permanents."""
    m, d = 5, 3
    U = haar_unitary(m, seed=6)
    inputs = [[1, 0, 1, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    sk = StageKernel(StageChain.empty(m), inputs=inputs, detected=(0, 3))
    blocks = _per_q(sk.blocks(U))
    rho = _random_rho(len(inputs) * d, seed=9)
    res = _per_q(sk.apply(U, rho, rest=d))
    r4 = rho.reshape(len(inputs), d, len(inputs), d)
    from photonic_engines.slos.reference import _modes_from_occupation, permanent
    for q, (basis, A, _) in blocks.items():
        assert A.shape == (len(basis), len(inputs))                               # chi = 1: scalar blocks
        for j, c in enumerate(basis):
            o = [x + y for x, y in zip(q, c)]
            for b, s in enumerate(inputs):
                if sum(o) != sum(s):
                    assert A[j, b] == 0
                    continue
                if sum(s) == 0:
                    ref = 1.0
                else:
                    ref = permanent(U[_modes_from_occupation(o)][:, _modes_from_occupation(s)]).item()
                    ref /= math.sqrt(math.prod(math.factorial(v) for v in s) * math.prod(math.factorial(v) for v in o))
                assert abs(complex(A[j, b]) - complex(ref)) < 1e-12
        out = torch.einsum("ia,axby,jb->ixjy", A, r4, A.conj()).reshape(len(basis) * d, len(basis) * d)
        assert torch.allclose(res[q][1], out, atol=1e-15, rtol=0)


def test_two_stages_equal_one_shot_detection():
    """Detect part, carry, emit more, detect the rest = full detection of the unrolled problem: stage 1 emits into
    modes 0-3 with U1 on 0-3 and detects 0, 1; stage 2 takes the carried photons as inputs, emits into 4, 5 and U2
    acts on 2-5 (identity on the detected modes, which U1 leaves alone too)."""
    rng = np.random.default_rng(5)
    chi, m = 2, 6
    ops1 = [_isometry_blocks(2, chi, rng), _isometry_blocks(2, chi, rng)]
    ops2 = [_isometry_blocks(2, chi, rng)]
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    s0 = s0 / np.linalg.norm(s0)
    ch1 = StageChain.from_mps(EmitterMPS.ragged(ops1, [(0, 1), (2, 3)], s0, m=m))
    ch2 = StageChain.from_mps(EmitterMPS.ragged(ops2, [(4, 5)], s0, m=m))
    one = StageChain.from_mps(EmitterMPS.ragged(ops1 + ops2, [(0, 1), (2, 3), (4, 5)], s0, m=m))
    U1 = torch.eye(m, dtype=C128)
    U1[:4, :4] = haar_unitary(4, seed=1)
    U2 = torch.eye(m, dtype=C128)
    U2[2:, 2:] = haar_unitary(4, seed=2)
    rho0 = torch.outer(torch.as_tensor(s0, dtype=C128), torch.as_tensor(s0, dtype=C128).conj())

    (g,) = StageKernel(one).kraus(U2 @ U1)
    K = g.physical()[0]
    want = {tuple(t): K[s] @ rho0 @ K[s].conj().T for s, t in enumerate(g.patterns.tolist())}

    got = {}
    for q1, (basis, rho_q1, _) in _per_q(StageKernel(ch1, detected=(0, 1)).apply(U1, rho0)).items():
        for t2, (b2, rho_t, _) in _per_q(StageKernel(ch2, inputs=basis).apply(U2, rho_q1)).items():
            assert b2 == [(0,) * m]
            if t2[0] or t2[1]:                          # modes already detected: stage 2 leaves them empty
                assert rho_t.abs().max() < 1e-15
                continue
            t = tuple(x + y for x, y in zip(q1, t2))
            assert t not in got
            got[t] = rho_t
    for t, r in want.items():
        assert torch.allclose(got.get(t, torch.zeros_like(r)), r, atol=1e-13, rtol=0)
    assert all(t in want or not r.abs().max() > 1e-15 for t, r in got.items())


def test_chunk_batches_dispatch_and_refusals():
    ch = _chain((2, 2), 2, m=6, seed=12)
    inputs = _free_inputs(ch, [[0], [1], []])
    sk = StageKernel(ch, inputs=inputs, detected=(1, 3))
    Us = torch.stack([haar_unitary(6, seed=s) for s in (1, 2)])
    rhos = torch.stack([_random_rho(6, seed=s) for s in (3, 4, 5)])
    full = sk.apply(Us, rhos)
    for cl, cc in zip(full.classes, sk.apply(Us, rhos, chunk=1).classes):   # the einsums reduce per chunk shape:
        assert torch.allclose(cl.value, cc.value, atol=1e-16, rtol=0)        # equal to rounding, not bit for bit
        assert torch.allclose(cl.probs, cc.probs, atol=1e-16, rtol=0)
    for cl, cc in zip(sk.blocks(Us).classes, StageKernel(ch, inputs=inputs, detected=(1, 3)).blocks(Us).classes):
        assert torch.equal(cl.value, cc.value)                               # the regroup itself is exact
    for u in range(2):
        for r in range(3):
            one = sk.apply(Us[u], rhos[r])
            for cl, c1 in zip(full.classes, one.classes):
                assert torch.allclose(cl.value[u, r], c1.value, atol=1e-15, rtol=0)
            assert abs(float(full.deviation[u, r] - one.deviation)) < 1e-15
    assert full.deviation.abs().max() < 1e-13
    for cl, cb in zip(sk.blocks(Us).classes, sk.blocks(Us[1]).classes):
        assert torch.equal(cl.value[1], cb.value)
    # the vacuum input of a single-photon chain dispatches to EmitterSLOSKernel; its blocks regroup the same set
    vac = StageKernel(ch, detected=(1, 3))
    assert vac._dispatch is not None
    gen = StageKernel(ch, inputs=[[0] * 6, inputs[0]], detected=(1, 3)).blocks(Us[0])
    pv, pg = _per_q(vac.blocks(Us[0])), _per_q(gen)
    for q, (basis, Kq, _) in pv.items():
        bg, Kg, _ = pg[q]
        rows = [bg.index(c) for c in basis]
        Kg = Kg.reshape(len(bg), 2, 2, 2)[rows][:, :, 0]
        assert torch.allclose(Kq.reshape(len(basis), 2, 2), Kg, atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="detected"):
        StageKernel(ch, detected=[0, 6])
    with pytest.raises(ValueError, match="rho must be"):
        sk.apply(Us[0], torch.eye(5, dtype=C128))


def test_deviation_tolerance_byte_budget_and_row_keys():
    """apply warns (or raises with strict) when sum_q Tr rho_q departs from Tr rho: an input occupying an emission mode
    is defined but not an isometry. chunk from a byte budget = unchunked. The mixed-radix row key = the row unique."""
    ch = _chain((2, 2), 2, m=6, seed=15)
    emit = ch.event_modes[0][0]
    occupied = [[1 if p == emit else 0 for p in range(6)]]
    sk = StageKernel(ch, inputs=occupied, detected=(0, 1, 2))
    U = haar_unitary(6, seed=16)
    rho = _random_rho(2, seed=2)
    with pytest.warns(RuntimeWarning, match="Tr rho"):
        res = sk.apply(U, rho)
    assert abs(float(res.deviation)) > 1e-3
    with pytest.raises(ValueError, match="Tr rho"):
        sk.apply(U, rho, strict=True)
    sk.apply(U, rho, tol=float("inf"), strict=True)                          # tol: the caller's bound
    free = StageKernel(ch, inputs=_free_inputs(ch, [[0], [1], []]), detected=(1, 3))
    rho3 = _random_rho(6, seed=3)
    full = free.apply(U, rho3, strict=True, chunk_bytes=2 ** 40)
    small = free.apply(U, rho3, strict=True, chunk_bytes=1)                  # one q per pass
    for a, b in zip(full.classes, small.classes):
        assert torch.allclose(a.value, b.value, atol=1e-16, rtol=0)
    from photonic_engines.slos.stage import _unique_rows
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, size=(500, 6))
    tab, idx = _unique_rows(a)
    wide = np.concatenate([a, np.zeros((500, 40), dtype=a.dtype)], axis=1)  # 46 x 2 bits: the key path
    wide[:, -1] = 2 ** 40                                                    # base 2^40 + 1: the row-unique fallback
    tab_w, idx_w = _unique_rows(wide)
    assert np.array_equal(tab_w[:, :6], tab) and np.array_equal(idx_w, idx)
    assert np.array_equal(tab[idx], a) and len({tuple(r) for r in a}) == len(tab)
    first = [next(i for i in range(500) if tuple(a[i]) == tuple(r)) for r in tab]
    assert first == sorted(first)                                            # numbered by first appearance


@pytest.mark.gpu
@pytest.mark.parametrize("inputs", [None, "batch"])
def test_carry_over_cuda_matches_cpu(inputs):
    """On CUDA, with the vacuum dispatch to EmitterSLOSKernel (whose patterns live on the device) and with a batch
    (general path): the tables, blocks and update equal the CPU's."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ch = _chain((2, 2, 1), 2, m=7, seed=21)
    inp = None if inputs is None else _free_inputs(ch, [[0], [], [0, 1]])
    B = 1 if inp is None else len(inp)
    U = haar_unitary(7, seed=22)
    rho = _random_rho(B * 2 * 2, seed=23)
    kc, kd = (StageKernel(ch, inputs=inp, detected=(0, 2, 4), device=d) for d in ("cpu", "cuda"))
    assert (kd._dispatch is not None) == (inputs is None)
    rc, rd = kc.apply(U, rho, rest=2, strict=True), kd.apply(U.cuda(), rho.cuda(), rest=2, strict=True)
    assert torch.equal(rc.detected, rd.detected.cpu()) and torch.equal(rc.carried, rd.carried.cpu())
    for a, b in zip(rc.classes, rd.classes):
        assert torch.equal(a.basis, b.basis.cpu())
        assert torch.allclose(a.value, b.value.cpu(), atol=1e-13, rtol=0)       # device check's cuda = cpu bound
    for a, b in zip(kc.blocks(U).classes, kd.blocks(U.cuda()).classes):
        assert torch.allclose(a.value, b.value.cpu(), atol=1e-13, rtol=0)


def test_gradients_of_the_detected_probabilities():
    ch = _chain((2, 2), 2, m=5, seed=13)
    inputs = _free_inputs(ch, [[0], []])
    rho = _random_rho(4, seed=1)
    U = haar_unitary(5, seed=14).requires_grad_(True)
    ops = torch.as_tensor(ch.ops, dtype=C128).requires_grad_(True)

    def loss(o, u):
        # finite differences perturb the operators off completeness: the deviation check is not the point here
        res = StageKernel(ch, inputs=inputs, detected=(0, 2), ops_tensor=o).apply(u, rho, tol=float("inf"))
        return sum((cl.probs * torch.linspace(0.5, 2.0, cl.probs.numel(), dtype=torch.float64)).sum()
                   for cl in res.classes)

    f = loss(ops, U)
    f.backward()
    eps = 1e-6
    with torch.no_grad():
        o2 = ops.detach().clone(); o2[0, 1, 0, 1] += eps
        assert abs(float(ops.grad[0, 1, 0, 1].real) - float((loss(o2, U.detach()) - f.detach()) / eps)) < 1e-4
        u2 = U.detach().clone(); u2[1, 3] += eps
        assert abs(float(U.grad[1, 3].real) - float((loss(ops.detach(), u2) - f.detach()) / eps)) < 1e-4
