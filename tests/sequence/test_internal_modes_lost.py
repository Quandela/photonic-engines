"""Internal photon modes with the no-photon branch (docs/SEQUENCE.md §14): `lost` operators (an
emission that is never detected, possibly spin-dependent) and source efficiency (`source_loss`), by route F (presence
branches) and the pairing route P' (z-grading), against an oracle independent of both: the lossless problem with one
dummy spatial mode per (emitter, lost operator), U (+) 1, the dummy occupations summed out."""
import math

import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_route_f import _superops

from photonic_engines.sequence import ModeBinEmitter, SequenceKernel
from photonic_engines.sequence_modes import source_loss


def _lossy_emitter(n, L, K, Q, d, modes, seed):
    """A complete emitter with Q spin-dependent `lost` operators: a random isometry [(L K + Q) d, d] split into
    A[l, k] and N_q."""
    rng = np.random.default_rng(seed)
    ops = np.empty((n, L, K, d, d), dtype=complex)
    lost = np.empty((n, Q, d, d), dtype=complex)
    for b in range(n):
        z = rng.normal(size=((L * K + Q) * d, d)) + 1j * rng.normal(size=((L * K + Q) * d, d))
        q, _ = np.linalg.qr(z)
        ops[b] = q[:L * K * d].reshape(L, K, d, d)
        lost[b] = q[L * K * d:].reshape(Q, d, d)
    s0 = np.zeros(d, dtype=complex)
    s0[0] = 1
    return ModeBinEmitter(ops, modes, s0, lost=lost)


def _rho(D, B, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, D, generator=g, dtype=torch.complex128)
    r = z @ z.conj().transpose(1, 2)
    return r / torch.einsum("bii->b", r).view(-1, 1, 1)


def _oracle(mk, U, b=0):
    """Per lossy pattern of mk, sum K (x) conj K [S, chi^2, chi^2] from the dummy-mode construction: emitter j's lost
    operator q becomes a label emitting into its own extra mode (internal mode 0), U (+) 1 on the extra modes, a
    lossless internal-mode kernel, outcomes restricted to the m real modes."""
    m, K = mk.m, mk.K
    ems, extra = [], m
    for e in mk.emitters:
        ops = e.ops if isinstance(e, ModeBinEmitter) else e.ops[:, :, None]
        n, L, Kj, d, _ = ops.shape
        full = np.zeros((n, L, K, d, d), dtype=complex)
        full[:, :, :Kj] = ops
        Q = 0 if not isinstance(e, ModeBinEmitter) or e.lost is None else e.lost.shape[1]
        ext = np.zeros((n, L + Q, K, d, d), dtype=complex)
        ext[:, :L] = full
        if Q:
            ext[:, L:, 0] = e.lost
        ems.append(ModeBinEmitter(ext, tuple(e.modes) + tuple(range(extra, extra + Q)), e.s0))
        extra += Q
    big = SequenceKernel(ems, extra)
    Ub = torch.block_diag(torch.as_tensor(U), torch.eye(extra - m, dtype=torch.complex128))
    Kx, cx = big.kraus(Ub, b), big.coarse(Ub, b)
    occ = np.asarray(big.patterns)[cx.numpy()][:, :m]
    return _superops(Kx, mk.pattern_index(occ), mk.S)


def _law(J, rho, chi):
    Jr = J.reshape(J.shape[0], chi, chi, chi, chi)
    return torch.einsum("tacbd,zbd,ac->zt", Jr, rho, torch.eye(chi, dtype=J.dtype)).real


@pytest.mark.parametrize("k,K,Q,plain_last", [(2, 2, 1, False), (2, 3, 2, True), (3, 2, 1, False)])
def test_lost_matches_the_dummy_mode_oracle(k, K, Q, plain_last):
    """Route F's presence branches and P''s z-grading against the dummy-mode construction: per-outcome
    superoperators (F) and the outcome law (P', F) < 1e-14 over every pattern of 0..k detected photons; one emitter
    without `lost` (always detected) when plain_last."""
    m = 2 * k
    ems = [_lossy_emitter(2, 2, K, Q, 2, (2 * j, 2 * j + 1), seed=10 * k + j) for j in range(k)]
    if plain_last:
        e = ems[-1]
        rng = np.random.default_rng(5)
        z = rng.normal(size=(2 * K * 2, 2)) + 1j * rng.normal(size=(2 * K * 2, 2))
        ops = np.stack([np.linalg.qr(z)[0].reshape(2, K, 2, 2)] * 2)
        ems[-1] = ModeBinEmitter(ops, e.modes, e.s0)
    mk = SequenceKernel(ems, m)
    U = haar_unitary(m, 3 + k)
    n_pat = sum(math.comb(m + n - 1, n) for n in range(k + 1))                 # 0..k detected photons
    assert mk.lossy and n_pat == mk.S
    J_or = _oracle(mk, U)
    J_f = _superops(mk.kraus(U, 0), mk.coarse(U, 0), mk.S)
    assert float((J_or - J_f).abs().max()) < 1e-14
    rho = _rho(mk.chi, 2, k)
    ref = _law(J_or, rho, mk.chi)
    for route in ("P", "F"):
        assert float((mk.outcome_law(U, 0, rho, route=route) - ref).abs().max()) < 1e-14
    assert float((mk.outcome_law(U, 0, rho, route="P").sum(1) - 1).abs().max()) < 1e-13


def test_pattern_order_and_zero_photon_outcome():
    """The k-photon block comes first with the lossless kernel's indices, the all-zero pattern last; the zero-photon
    outcome is the product of the absent-photon channels (x)_j sum_q N_q . N_q^dag."""
    k, K, m = 2, 2, 4
    ems = [_lossy_emitter(1, 2, K, 2, 2, (2 * j, 2 * j + 1), seed=40 + j) for j in range(k)]
    mk = SequenceKernel(ems, m)
    U = haar_unitary(m, 42)
    S_k = math.comb(m + k - 1, k)
    assert np.array_equal(mk.patterns[:S_k], SequenceKernel.patterns.fget(mk))     # the base kernel's k-photon basis
    assert mk.patterns[:S_k].sum(1).tolist() == [k] * S_k and mk.patterns[-1].sum() == 0
    n_pat = S_k + m + 1                                                          # k, 1 and 0 photons (k = 2)
    assert n_pat == mk.S
    rho = _rho(mk.chi, 1, 41)
    zero = mk.S - 1
    N = [torch.as_tensor(e.lost[0]) for e in ems]
    # (N0 (x) N1) rho (N0 (x) N1)^dag summed over q, r: spin 0 = the left tensor factor
    ref = torch.einsum("qab,rcd,bdeh,qfe,rgh->acfg", N[0], N[1], rho[0].reshape(2, 2, 2, 2), N[0].conj(),
                       N[1].conj()).reshape(1, 4, 4)
    for route in ("P", "F"):
        got = mk.condition(U, 0, rho, [zero], route=route)
        assert float((got - ref).abs().max()) < 1e-15


def test_pairing_equals_presence_branches_with_classes_and_indist():
    """P' = F on batches with `lost` and `indist`: the law, single outcomes of every photon number, a class mixing
    photon numbers and a class of more than half the patterns (the complement, whose emission channel carries the
    `lost` terms)."""
    k, K, m = 2, 3, 4
    ems = [_lossy_emitter(1, 2, K, 1, 2, (2 * j, 2 * j + 1), seed=50 + j) for j in range(k)]
    mk = SequenceKernel(ems, m, indist=[0.8, 0.6])
    U = haar_unitary(m, 51)
    rho = _rho(mk.chi, 3, 52)
    assert float((mk.outcome_law(U, 0, rho, route="F") - mk.outcome_law(U, 0, rho, route="P")).abs().max()) < 1e-15
    for cls in ([0], [11], [mk.S - 1], [1, 12, mk.S - 1], list(range(mk.S // 2 + 1))):
        cF, cP = mk.condition(U, 0, rho, cls, route="F"), mk.condition(U, 0, rho, cls, route="P")
        assert float((cF - cP).abs().max()) < 1e-14


def test_conditioned_register_at_three_photons():
    """k = 3 (the first size where the event and bra-slot sign groups differ in length at every degree): the
    conditioned register by P' against F and against the dummy-mode oracle, on a batch, for one outcome of each
    detected photon number 0..3 and a class mixing them."""
    k, K, m = 3, 2, 6
    ems = [_lossy_emitter(1, 2, K, 1, 2, (2 * j, 2 * j + 1), seed=90 + j) for j in range(k)]
    mk = SequenceKernel(ems, m)
    U = haar_unitary(m, 91)
    rho = _rho(mk.chi, 3, 92)
    cnt = mk.patterns.sum(1)
    one_each = [int(np.nonzero(cnt == n)[0][3 % int((cnt == n).sum())]) for n in range(k + 1)]
    J = _oracle(mk, U).reshape(mk.S, mk.chi, mk.chi, mk.chi, mk.chi)            # K (x) conj K, axes (a, c, b, d)
    assert 1 not in one_each                                                     # the mixed class: distinct outcomes
    for cls in [*[[t] for t in one_each], [*one_each, 1]]:
        ref = torch.einsum("tacbd,zbd->zac", J[cls], rho)
        cP, cF = mk.condition(U, 0, rho, cls, route="P"), mk.condition(U, 0, rho, cls, route="F")
        assert float((cP - ref).abs().max()) < 1e-14 and float((cF - ref).abs().max()) < 1e-14


def test_source_loss_helper():
    """source_loss: eta = 1 returns the emitter without `lost` (the lossless kernel, the same operators); the `lost`
    set is reduced to its Choi rank (<= d^2) with the same absent-photon channel; the channel is complete; eta per
    bin; a plain BinEmitter is accepted; against the dummy-mode oracle at eta = 0.7 and (0.9, 0.5)."""
    from test_internal_modes_route_f import _mode_emitter
    k, K, m = 2, 2, 4
    base = [_mode_emitter(2, 2, K, 2, (2 * j, 2 * j + 1), seed=60 + j) for j in range(k)]
    U = haar_unitary(m, 61)
    ref = SequenceKernel(base, m)
    one = SequenceKernel([source_loss(e, 1.0) for e in base], m)
    assert not one.lossy and torch.equal(ref.kraus(U, 0), one.kraus(U, 0))
    e7 = source_loss(base[0], 0.7)
    assert e7.lost.shape[1] <= 4                                                 # L K = 4 operators -> rank <= d^2
    full = np.sqrt(0.3) * base[0].ops.reshape(2, -1, 2, 2)                      # the unreduced absent operators
    lam = np.einsum("bqac,bqde->bacde", full, full.conj())
    lam7 = np.einsum("bqac,bqde->bacde", e7.lost, e7.lost.conj())
    assert np.abs(lam - lam7).max() < 1e-15
    mk = SequenceKernel([e7, source_loss(base[1], [0.9, 0.5])], m)
    for b in range(2):
        Kb = mk.kraus(U, b)
        tot = torch.einsum("qba,qbc->ac", Kb.conj(), Kb)
        assert float((tot - torch.eye(mk.chi, dtype=tot.dtype)).abs().max()) < 1e-13
        assert float((_oracle(mk, U, b) - _superops(Kb, mk.coarse(U, b), mk.S)).abs().max()) < 1e-14
    plain = _mode_emitter(2, 2, 1, 2, (0, 1), seed=62)                           # K = 1: a plain BinEmitter
    pl = source_loss(plain, 0.5)
    assert isinstance(pl, ModeBinEmitter) and pl.ops.shape == (2, 2, 1, 2, 2) and pl.lost.shape == (2, 2, 2, 2)
    with pytest.raises(ValueError, match="eta"):
        source_loss(base[0], 1.5)


def test_route_r_refused_with_lost():
    mk = SequenceKernel([_lossy_emitter(1, 2, 2, 1, 2, (2 * j, 2 * j + 1), seed=70 + j) for j in range(2)], 4)
    with pytest.raises(NotImplementedError, match="lost"):
        mk.kraus(haar_unitary(4, 71), 0, compress=1e-12, route="R")


@pytest.mark.gpu
def test_lost_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ems = [_lossy_emitter(1, 2, 3, 2, 2, (2 * j, 2 * j + 1), seed=80 + j) for j in range(3)]
    mk, gk = SequenceKernel(ems, 6), SequenceKernel(ems, 6, device="cuda")
    U = haar_unitary(6, 81)
    rho = _rho(mk.chi, 2, 82)
    for route in ("P", "F"):
        a = mk.outcome_law(U, 0, rho, route=route)
        g = gk.outcome_law(U, 0, rho.cuda(), route=route).cpu()
        assert float((a - g).abs().max()) < 1e-14
