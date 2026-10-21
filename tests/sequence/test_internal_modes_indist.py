"""Internal photon modes with partial distinguishability (docs/SEQUENCE.md §14, design test 6): the x-model as private
internal modes in route F and as the scaled pair Gram in the pairing route P', against each other and, at one internal
mode, against today's species-resolved path (`SequenceKernel(indist=...)`, docs/DISTINGUISHABILITY.md §3.4)."""
import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_route_f import _mode_emitter, _superops

from photonic_engines.sequence import BinEmitter, ModeBinEmitter, SequenceKernel


def _rho(D, B, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, D, generator=g, dtype=torch.complex128)
    r = z @ z.conj().transpose(1, 2)
    return r / torch.einsum("bii->b", r).view(-1, 1, 1)


def _emitters(k, K, seed):
    return [_mode_emitter(2, 2, K, 2, (2 * j, 2 * j + 1), seed=seed + j) for j in range(k)]


@pytest.mark.parametrize("k,indist", [(2, 0.8), (2, [0.6, 0.95]), (3, [0.5, 1.0, 0.8]), (3, 0.0)])
def test_one_internal_mode_is_the_species_path(k, indist):
    """K = 1 through the internal-mode kernel with indist: route F (private modes) and P' (scaled Gram) give today's
    species-resolved channel per outcome (< 1e-14 on the superoperators; a different algorithm, not bitwise)."""
    m = 2 * k
    plain = _emitters(k, 1, 10 * k)                                          # K = 1: the factory gives BinEmitters
    assert all(type(e) is BinEmitter for e in plain)
    one = [ModeBinEmitter(e.ops[:, :, None], e.modes, e.s0) for e in plain]
    U = haar_unitary(m, 7 + k)
    ref = SequenceKernel(plain, m, indist=indist)
    mk = SequenceKernel(one, m, indist=indist)
    assert type(mk).__name__ == "ModeSequenceKernel" and type(ref) is SequenceKernel
    for b in range(2):
        J_ref = _superops(ref.kraus(U, b), ref.coarse(U, b), ref.S)
        J_f = _superops(mk.kraus(U, b), mk.coarse(U, b), mk.S)
        assert float((J_ref - J_f).abs().max()) < 1e-14
    rho = _rho(mk.chi, 2, k)
    J = J_ref.reshape(ref.S, mk.chi, mk.chi, mk.chi, mk.chi)                 # sum K (x) conj K, axes (a, c, b, d)
    law_ref = torch.einsum("tacbd,zbd,ac->zt", J, rho, torch.eye(mk.chi, dtype=J.dtype)).real
    assert float((mk.outcome_law(U, 1, rho, route="P") - law_ref).abs().max()) < 1e-14
    assert float((mk.outcome_law(U, 1, rho, route="F") - law_ref).abs().max()) < 1e-14


def _class_cert(mk, U, rho, cls):
    """P''s Frobenius error estimate of a class's conditioned register [B], as `condition` forms it: the sum of its
    outcomes' certificates, or for a class of more than S/2 patterns the emission channel's bound plus the
    complement's certificates."""
    from photonic_engines.slos.internal import emission_channel
    if len(cls) > mk.S // 2:
        rest = torch.as_tensor(sorted(set(range(mk.S)) - set(cls)), dtype=torch.long)
        _, c_tot = emission_channel(mk._pairing_inputs(0)[0], rho)
        return c_tot + mk._apply_p(U, 0, rho, rest)[1].sum(1)
    return mk._apply_p(U, 0, rho, torch.as_tensor(cls))[1].sum(1)


def _f_rounding(mk, U, E):
    """Route F's a-priori rounding bound on a conditioned register (2 n gamma_w relative, w = L K_eff + n
    terms per accumulated entry per layer), times its Frobenius norm [B]."""
    L = max(len(e.modes) for e in mk.emitters)
    w = L * mk._kb(U, 0) + mk.k
    return 2 * mk.k * w * torch.finfo(torch.float64).eps * E.flatten(1).norm(dim=1)


@pytest.mark.parametrize("k,K,indist", [(2, 3, 0.7), (2, 8, [0.9, 0.3]), (3, 3, [0.8, 1.0, 0.6])])
def test_pairing_scaling_equals_private_modes(k, K, indist):
    """K > 1: the outcome law (< 1e-15) and the conditioned register of classes by P' (Gram scaled by
    sqrt(x_j x_j')) equal route F's (private internal modes) on a batch; a class's difference is bounded by P''s class
    estimate + F's rounding bound and by the criterion 1e-14 (a class of S/2 outcomes sums rounding errors of ~1e-16
    each: measured up to 1.4e-15 on the H100 host's CPU, run 20261020_221250, above a flat 1e-15); route R
    (private modes, pairing core) gives F's compressed channel."""
    m = 2 * k
    mk = SequenceKernel(_emitters(k, K, 20 * k + K), m, indist=indist)
    U = haar_unitary(m, 30 + k)
    rho = _rho(mk.chi, 3, K)
    pF, pP = mk.outcome_law(U, 0, rho, route="F"), mk.outcome_law(U, 0, rho, route="P")
    assert float((pF - pP).abs().max()) < 1e-15 and float((pP.sum(1) - 1).abs().max()) < 1e-13
    for cls in ([1], list(range(mk.S // 2)), list(range(mk.S // 2 + 1))):             # last: via the complement
        cF, cP = mk.condition(U, 0, rho, cls, route="F"), mk.condition(U, 0, rho, cls, route="P")
        err = (cF - cP).flatten(1).norm(dim=1)
        assert bool((err <= _class_cert(mk, U, rho, cls) + _f_rounding(mk, U, cF)).all())
        assert float((cF - cP).abs().max()) < 1e-14
    JF = _superops(mk.kraus(U, 0, compress=1e-12, route="F"), mk.coarse(U, 0, compress=1e-12, route="F"), mk.S)
    JR = _superops(mk.kraus(U, 0, compress=1e-12, route="R"), mk.coarse(U, 0, compress=1e-12, route="R"), mk.S)
    assert float((JF - JR).abs().max()) < 1e-14


@pytest.mark.parametrize("K,indist", [(2, 0.6), (3, [0.8, 0.4])])
def test_expanded_construction_with_indist(K, indist):
    """K > 1, an oracle independent of the private-mode derivation: the expanded construction
    (labels (l, c) on m K modes, U (x) 1_K) through the plain kernel's species route with the same indist, outcomes
    summed over c, per-outcome superoperators against route F and route P' (law), < 1e-14."""
    from test_internal_modes_route_f import _expanded
    k, m = 2, 4
    ems = _emitters(k, K, 70 + K)
    U = haar_unitary(m, 71)
    mk = SequenceKernel(ems, m, indist=indist)
    ex = SequenceKernel(_expanded(ems, m, K).emitters, m * K, indist=indist)
    Ub = torch.block_diag(*[U] * K)                                          # mode c m + i
    Kx, cx = ex.kraus(Ub, 0), ex.coarse(Ub, 0)
    occ = torch.as_tensor(np.asarray(ex.patterns))[cx].reshape(-1, K, m).sum(1)
    J_ex = _superops(Kx, mk.pattern_index(occ), mk.S)
    J_f = _superops(mk.kraus(U, 0), mk.coarse(U, 0), mk.S)
    assert float((J_ex - J_f).abs().max()) < 1e-14
    rho = _rho(mk.chi, 2, K)
    J = J_ex.reshape(mk.S, mk.chi, mk.chi, mk.chi, mk.chi)
    law = torch.einsum("tacbd,zbd,ac->zt", J, rho, torch.eye(mk.chi, dtype=J.dtype)).real
    assert float((mk.outcome_law(U, 0, rho, route="P") - law).abs().max()) < 1e-14


def test_indistinguishable_limit_and_private_block_sizes():
    """x = 1 for every emitter is the kernel without indist (no private modes, the same operators); an emitter with
    x = 1 gets no private block; the fine set drops the exact-zero operators of patterns no photon can fill."""
    k, K, m = 2, 3, 4
    ems = _emitters(k, K, 50)
    U = haar_unitary(m, 51)
    a, b = SequenceKernel(ems, m), SequenceKernel(ems, m, indist=[1.0, 1.0])
    assert b.indist is None and torch.equal(a.kraus(U, 0), b.kraus(U, 0))
    c = SequenceKernel(ems, m, indist=[1.0, 0.5])
    ops = c._ops(0)
    assert [o.shape[1] for o in ops] == [2 * K, 2 * K]                         # K common + K private (emitter 1)
    assert float(ops[0][:, K:].abs().max()) == 0.0
    Kc = c.kraus(U, 0)
    assert bool(Kc.flatten(1).ne(0).any(1).all())
    tot = torch.einsum("qba,qbc->ac", Kc.conj(), Kc)
    assert float((tot - torch.eye(c.chi, dtype=tot.dtype)).abs().max()) < 1e-13


@pytest.mark.gpu
def test_indist_cuda_matches_cpu():
    """The scaled pair Gram (P') and the private modes (F) on CUDA give the CPU results to rounding."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    k, K, m = 3, 3, 6
    ems = _emitters(k, K, 60)
    U = haar_unitary(m, 61)
    mk, gk = SequenceKernel(ems, m, indist=0.7), SequenceKernel(ems, m, indist=0.7, device="cuda")
    rho = _rho(mk.chi, 2, 62)
    for route in ("P", "F"):
        a = mk.outcome_law(U, 0, rho, route=route)
        g = gk.outcome_law(U, 0, rho.cuda(), route=route).cpu()
        assert float((a - g).abs().max()) < 1e-14


def test_distinguishability_lowers_interference():
    """Fully distinguishable photons (x = 0) on a balanced beam splitter: no two-photon interference, so the
    coincidence probability of two single-rail photons is 1/2 whatever K (HOM dip filled); x = 1 and K = 1: 0."""
    m = 2
    ops = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    ops[0, 0, 0] = 1.0
    ems = [ModeBinEmitter(ops, (j,), [1.0]) for j in range(2)]
    bs = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
    coinc = next(i for i, p in enumerate(SequenceKernel(ems, m).patterns) if p.max() == 1)
    rho = torch.ones(1, 1, dtype=torch.complex128)
    p_ind = SequenceKernel(ems, m).outcome_law(bs, 0, rho, route="P")[coinc]
    for route in ("F", "P"):
        p_dis = SequenceKernel(ems, m, indist=0.0).outcome_law(bs, 0, rho, route=route)[coinc]
        assert abs(float(p_dis) - 0.5) < 1e-15
    assert abs(float(p_ind)) < 1e-15
