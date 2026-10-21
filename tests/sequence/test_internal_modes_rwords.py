"""Internal photon modes with R > 1 unobserved emission outcomes per emitter (ops [n, L, K, R, d, d]; docs/SEQUENCE.md
§14): the channel is the sum over the r-words. Oracle independent of that sum: each spin dilated by an
environment register of dimension R, A' = sum_r A_r (x) |r + e mod R><e| (complete), `lost` (x) 1, the register
started in rho (x) |0><0| and the environment traced out, through the R = 1 path."""
import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines.sequence import ModeBinEmitter, SequenceKernel
from photonic_engines.sequence_modes import source_loss


def _emitter_r(n, L, K, R, Q, d, modes, seed):
    """A complete emitter with R unobserved outcomes and Q `lost` operators: a random isometry [(L K R + Q) d, d]."""
    rng = np.random.default_rng(seed)
    ops = np.empty((n, L, K, R, d, d), dtype=complex)
    lost = np.empty((n, Q, d, d), dtype=complex) if Q else None
    for b in range(n):
        z = rng.normal(size=((L * K * R + Q) * d, d)) + 1j * rng.normal(size=((L * K * R + Q) * d, d))
        q, _ = np.linalg.qr(z)
        ops[b] = q[:L * K * R * d].reshape(L, K, R, d, d)
        if Q:
            lost[b] = q[L * K * R * d:].reshape(Q, d, d)
    s0 = np.zeros(d, dtype=complex)
    s0[0] = 1
    return ModeBinEmitter(ops, modes, s0, lost=lost)


def _dilated(e):
    """The emitter on spin (x) environment (dimension d R, spin major): ops A' = sum_r A_r (x) |r><0| (an isometry on
    the environment-|0> subspace, where the register starts: one emission per environment, n = 1), lost N (x) 1."""
    n, L, K, R, d, _ = e.ops.shape
    lift = [np.outer(np.eye(R)[r], np.eye(R)[0]) for r in range(R)]              # |r><0|
    ops = sum(np.einsum("nlkab,xy->nlkaxby", e.ops[:, :, :, r], lift[r]) for r in range(R)).reshape(n, L, K, d * R,
                                                                                                  d * R)
    lost = None
    if e.lost is not None:
        lost = np.einsum("nqab,xy->nqaxby", e.lost, np.eye(R)).reshape(n, e.lost.shape[1], d * R, d * R)
    s0 = np.kron(e.s0, np.eye(R)[0])
    return ModeBinEmitter(ops, e.modes, s0, lost=lost)


def _rho(D, B, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, D, generator=g, dtype=torch.complex128)
    r = z @ z.conj().transpose(1, 2)
    return r / torch.einsum("bii->b", r).view(-1, 1, 1)


def _oracle_condition(ems, m, U, rho, cls, indist=None, b=0):
    """E_C(rho) by the dilated R = 1 kernel: rho (x) |0><0| per environment, route F, environments traced. The dilated
    emitters are complete only on the environment-|0> subspace (all the register visits), so the kernel's global
    completeness check is skipped for this oracle (route F's operators do not depend on it)."""
    from unittest import mock

    from photonic_engines.sequence_modes import ModeSequenceKernel
    dil = [_dilated(e) for e in ems]
    with mock.patch.object(ModeSequenceKernel, "_check_complete", return_value=None):
        big = SequenceKernel(dil, m, indist=indist)
    k = len(ems)
    d, R = ems[0].d, ems[0].R
    B = rho.shape[0]
    env0 = torch.zeros(R, R, dtype=rho.dtype)
    env0[0, 0] = 1
    r = rho.reshape(B, *([d] * k), *([d] * k))
    full = r
    for _ in range(k):                                                           # attach |0><0| per spin, spin-major
        full = torch.einsum("b...,xy->b...xy", full, env0)
    # axes: b, s_1..s_k, s'_1..s'_k, then (e_j, e'_j) pairs; reorder to (s_1 e_1 .. s_k e_k), (s'_1 e'_1 ..)
    ket = [1 + j for j in range(k)]
    bra = [1 + k + j for j in range(k)]
    env_k = [1 + 2 * k + 2 * j for j in range(k)]
    env_b = [2 + 2 * k + 2 * j for j in range(k)]
    order = [0] + [a for j in range(k) for a in (ket[j], env_k[j])] + [a for j in range(k) for a in (bra[j], env_b[j])]
    D2 = (d * R) ** k
    rho_d = full.permute(order).reshape(B, D2, D2)
    out = big.condition(U, b, rho_d, cls, route="F").reshape(B, *([d, R] * k), *([d, R] * k))
    letters = "acegikmoqs"                                                       # (spin, environment) per emitter
    sub_in = "Z" + "".join(letters[2 * j] + letters[2 * j + 1] for j in range(k)) \
        + "".join(letters[2 * j].upper() + letters[2 * j + 1] for j in range(k))    # bra env = ket env: traced
    sub_out = "Z" + "".join(letters[2 * j] for j in range(k)) + "".join(letters[2 * j].upper() for j in range(k))
    return torch.einsum(f"{sub_in}->{sub_out}", out).reshape(B, d ** k, d ** k)


@pytest.mark.parametrize("k,K,R,Q,indist", [(2, 2, 2, 0, None), (2, 2, 3, 1, None), (2, 3, 2, 1, [0.8, 0.5]),
                                            (3, 2, 2, 1, None)])
def test_rwords_against_the_dilated_spin(k, K, R, Q, indist):
    """F (fine sets per r-word) and P' (sum over r-words, Lam_j / R_j) against the dilated R = 1 construction: the
    conditioned register of single outcomes of every photon number and a class of more than half the patterns
    (the complement, whose emission channel stacks every r), < 1e-14; the law sums to 1."""
    m = 2 * k
    ems = [_emitter_r(1, 2, K, R, Q, 2, (2 * j, 2 * j + 1), seed=10 * k + R + j) for j in range(k)]
    mk = SequenceKernel(ems, m, indist=indist)
    assert mk.multi_r and [R] * k == mk.R
    U = haar_unitary(m, 7 + k)
    rho = _rho(mk.chi, 2, k + R)
    cnt = mk.patterns.sum(1)
    one_each = [int(np.nonzero(cnt == n)[0][0]) for n in sorted(set(cnt.tolist()))]
    classes = [[t] for t in one_each] + [list(range(mk.S // 2 + 1))]
    for cls in classes:
        ref = _oracle_condition(ems, m, U, rho, cls, indist)
        for route in ("F", "P"):
            got = mk.condition(U, 0, rho, cls, route=route)
            assert float((got - ref).abs().max()) < 1e-14, (cls, route)
    p = mk.outcome_law(U, 0, rho, route="P")
    assert float((p.sum(1) - 1).abs().max()) < 1e-13
    assert float((p - mk.outcome_law(U, 0, rho, route="F")).abs().max()) < 1e-15


def test_rwords_per_bin():
    """Two bins with different operators (the span and its cache are per bin): bin 1 against the oracle, both routes,
    after bin 0 was evaluated."""
    k, m = 2, 4
    ems = [_emitter_r(2, 2, 2, 2, 1, 2, (2 * j, 2 * j + 1), seed=60 + j) for j in range(k)]
    mk = SequenceKernel(ems, m)
    U = haar_unitary(m, 61)
    rho = _rho(mk.chi, 2, 62)
    cls = [1, mk.S - 1]
    mk.condition(U, 0, rho, cls, route="P")
    mk.condition(U, 0, rho, cls, route="F")
    ref = _oracle_condition(ems, m, U, rho, cls, b=1)
    ref0 = _oracle_condition(ems, m, U, rho, cls, b=0)
    assert float((ref - ref0).abs().max()) > 1e-3                                # the bins differ
    for route in ("F", "P"):
        assert float((mk.condition(U, 1, rho, cls, route=route) - ref).abs().max()) < 1e-14


def test_one_outcome_axis_is_the_plain_form():
    """ops [n, L, K, 1, d, d] is the same emitter as [n, L, K, d, d]: the same operators (torch.equal) and law."""
    from test_internal_modes_route_f import _mode_emitter
    ems = [_mode_emitter(2, 2, 3, 2, (2 * j, 2 * j + 1), seed=20 + j) for j in range(2)]
    six = [ModeBinEmitter(e.ops[:, :, :, None], e.modes, e.s0) for e in ems]
    U = haar_unitary(4, 21)
    a, b = SequenceKernel(ems, 4), SequenceKernel(six, 4)
    assert not b.multi_r and b.R == [1, 1]
    assert torch.equal(a.kraus(U, 0), b.kraus(U, 0)) and torch.equal(a.coarse(U, 0), b.coarse(U, 0))
    rho = _rho(a.chi, 1, 22)
    assert torch.equal(a.outcome_law(U, 0, rho, route="P"), b.outcome_law(U, 0, rho, route="P"))


def test_completeness_over_r_route_r_refused_and_source_loss():
    """Completeness is checked over (l, k, r) (a single r slice is refused); route R refuses R > 1; source_loss keeps
    the r axis (sqrt(eta) per r, the absent photon's Choi-rank `lost` over every r)."""
    e = _emitter_r(1, 2, 2, 2, 0, 2, (0, 1), seed=30)
    f = _emitter_r(1, 2, 2, 2, 0, 2, (2, 3), seed=31)
    with pytest.raises(ValueError, match="trace-preserving"):
        SequenceKernel([ModeBinEmitter(e.ops[:, :, :, :1], e.modes, e.s0), f], 4)
    mk = SequenceKernel([e, f], 4)
    U = haar_unitary(4, 32)
    with pytest.raises(NotImplementedError, match="R > 1"):
        mk.kraus(U, 0, compress=1e-12, route="R")
    K = mk.kraus(U, 0)
    tot = torch.einsum("qba,qbc->ac", K.conj(), K)
    assert float((tot - torch.eye(mk.chi, dtype=tot.dtype)).abs().max()) < 1e-13
    el = source_loss(e, 0.6)
    assert el.R == 2 and el.lost.shape[1] <= 4
    ml = SequenceKernel([el, source_loss(f, 0.8)], 4)
    rho = _rho(ml.chi, 1, 33)
    ref = _oracle_condition(ml.emitters, 4, U, rho, list(range(ml.S)))
    assert float((ml.condition(U, 0, rho, list(range(ml.S)), route="P") - ref).abs().max()) < 1e-14


@pytest.mark.gpu
def test_rwords_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ems = [_emitter_r(1, 2, 2, 2, 1, 2, (2 * j, 2 * j + 1), seed=40 + j) for j in range(2)]
    mk, gk = SequenceKernel(ems, 4), SequenceKernel(ems, 4, device="cuda")
    U = haar_unitary(4, 41)
    rho = _rho(mk.chi, 2, 42)
    for route in ("P", "F"):
        a = mk.outcome_law(U, 0, rho, route=route)
        g = gk.outcome_law(U, 0, rho.cuda(), route=route).cpu()
        assert float((a - g).abs().max()) < 1e-14
