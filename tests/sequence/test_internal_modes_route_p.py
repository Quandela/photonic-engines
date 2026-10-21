"""Internal photon modes, the pairing route (docs/SEQUENCE.md §14): the outcome law and the conditioned register by the
double Glynn sum over the pair Gram, checked against the channel route F (itself checked against the expanded
construction and a 50-digit reference in test_internal_modes_route_f.py)."""
import math

import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_route_f import _mode_emitter

from photonic_engines import sequence_modes
from photonic_engines.sequence import SequenceKernel
from photonic_engines.slos.internal import pairing_apply, span


def _kernel(k, K, seed=0, **kw):
    m = 2 * k
    ems = [_mode_emitter(2, 2, K, 2, (2 * j, 2 * j + 1), seed=seed + j) for j in range(k)]
    return SequenceKernel(ems, m, **kw), haar_unitary(m, seed + 50)


def _rho(D, B, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, D, generator=g, dtype=torch.complex128)
    r = z @ z.conj().transpose(1, 2)
    return r / torch.einsum("bii->b", r).view(-1, 1, 1)


@pytest.mark.parametrize("k,K", [(1, 3), (2, 2), (2, 8), (3, 3)])
def test_pairing_route_equals_channel_route(k, K):
    """outcome_law and condition: P' against F on a batch of density matrices, < 1e-15; the law sums to Tr rho."""
    mk, U = _kernel(k, K, seed=10 * k + K)
    rho = _rho(mk.chi, 3, k)
    for b in range(2):
        pF = mk.outcome_law(U, b, rho, route="F")
        pP = mk.outcome_law(U, b, rho, route="P")
        assert pF.shape == (3, mk.S) and float((pF - pP).abs().max()) < 1e-15
        assert float((pP.sum(1) - 1).abs().max()) < 1e-13
        for cls in ([0], sorted({1 % mk.S, mk.S - 1}), list(range(mk.S // 2 + 1))):        # last: via the complement
            cF = mk.condition(U, b, rho, cls, route="F")
            cP = mk.condition(U, b, rho, cls, route="P")
            assert float((cF - cP).abs().max()) < 1e-15


def test_classes_masks_states_and_complement():
    """A class equals the sum of its members; a boolean mask equals its index list; a pure state equals its density
    matrix; the complement path (class of more than half the patterns) equals the direct sum."""
    mk, U = _kernel(2, 3, seed=7)
    psi = torch.randn(mk.chi, dtype=torch.complex128, generator=torch.Generator().manual_seed(2))
    psi = psi / psi.norm()
    rho = torch.outer(psi, psi.conj())
    single = torch.stack([mk.condition(U, 0, rho, [t], route="P") for t in range(mk.S)])
    big = list(range(1, mk.S))
    assert float((mk.condition(U, 0, rho, big, route="P") - single[1:].sum(0)).abs().max()) < 1e-15
    mask = np.zeros(mk.S, dtype=bool)
    mask[[0, 2, 5]] = True
    assert torch.allclose(mk.condition(U, 0, rho, mask, route="P"), single[[0, 2, 5]].sum(0), atol=1e-15)
    assert torch.allclose(mk.condition(U, 0, psi, 4, route="P"), single[4], atol=1e-16)
    assert torch.allclose(mk.outcome_law(U, 0, psi, route="P"), torch.einsum("tii->t", single).real, atol=1e-15)


@pytest.mark.parametrize("k,K", [(2, 3), (3, 2), (4, 2)])
def test_certificate_bounds_the_measured_error(k, K):
    """The pairing route's certificate is at least its measured distance to F (n = 2..4, bunched outcomes included),
    and stays below 1e-13 here."""
    mk, U = _kernel(k, K, seed=30 + k)
    rho = _rho(mk.chi, 2, 31)
    A = torch.stack(span(mk.stage_ops(0)))
    inm = torch.stack(mk._in_modes)
    _, _, rows, tf = mk._pairing_inputs(0)
    E, cert = pairing_apply(U, A, inm, rows, tf, rho)
    EF = mk._apply_f(U, 0, rho, torch.arange(mk.S))
    err = (E - EF).flatten(2).norm(dim=-1)
    assert bool((err <= cert + 1e-30).all()) and float(cert.max()) < 1e-13


def test_tolerance_fallback_and_its_limit():
    """tol = 0 sends every pairing result back to the channel route (= F exactly); with F over max_bytes it raises
    rather than return a result above tol."""
    mk, U = _kernel(2, 3, seed=40)
    rho = _rho(mk.chi, 1, 41)[0]
    assert torch.equal(mk.outcome_law(U, 0, rho, tol=0.0, route="P"), mk.outcome_law(U, 0, rho, route="F"))
    small, U2 = _kernel(2, 3, seed=40, max_bytes=1000)
    with pytest.raises(ValueError, match="certificate"):
        small.outcome_law(U2, 0, rho, tol=0.0, route="P")
    assert small.outcome_law(U2, 0, rho, route="auto").shape == (small.S,)                     # F does not fit: P


def test_class_is_certified_as_a_whole():
    """condition sums its outcomes, so their certificates add up: a tol above every single certificate but below
    their sum sends the whole class back to F (= F exactly); a tol above the sum keeps the pairing result."""
    mk, U = _kernel(2, 3, seed=45)
    rho = _rho(mk.chi, 1, 46)
    cls = list(range(mk.S // 2))
    _, cert = mk._apply_p(U, 0, rho, torch.as_tensor(cls))
    lo, hi = float(cert.max()), float(cert.sum())                    # a class: Frobenius certificate against tol
    assert hi > 2 * lo
    tol = (lo * hi) ** 0.5
    exact_f = mk.condition(U, 0, rho[0], cls, route="F")
    assert torch.equal(mk.condition(U, 0, rho[0], cls, tol=tol, route="P"), exact_f)
    assert not torch.equal(mk.condition(U, 0, rho[0], cls, tol=2 * hi, route="P"), exact_f)


def test_large_class_routes():
    """A class of more than S/2 patterns: route F is the direct F class sum (one contraction, no complement), equal to
    the sum of its outcomes' channels; the pairing route's complement falling back (tol = 0) gives that same direct F
    sum exactly; a non-unitary U takes the direct sum, not the complement identity (which needs a unitary U)."""
    mk, U = _kernel(2, 3, seed=49)
    rho = _rho(mk.chi, 2, 50)
    cls = list(range(1, mk.S))
    direct_f = mk._apply_f(U, 0, rho, torch.as_tensor(cls), summed=True)[:, 0]
    per_outcome = mk._apply_f(U, 0, rho, torch.as_tensor(cls)).sum(1)
    assert float((direct_f - per_outcome).abs().max()) < 1e-15
    assert torch.equal(mk.condition(U, 0, rho, cls, route="F"), direct_f)
    assert torch.equal(mk.condition(U, 0, rho, cls, tol=0.0, route="P"), direct_f)
    V = 0.9 * U
    direct_p, _ = mk._apply_p(V, 0, rho, torch.as_tensor(cls))
    assert torch.equal(mk.condition(V, 0, rho, cls, route="P"), direct_p.sum(1))


def test_large_class_at_four_photons_fallback_and_raise():
    """k = 4: a class of S/2 outcomes has a summed certificate well above any single one. With F over max_bytes, a tol
    below the sum raises (naming the class size); a tol above it returns the pairing result, which agrees with F."""
    mk, U = _kernel(4, 2, seed=90, max_bytes=1000)
    rho = _rho(mk.chi, 1, 91)[0]
    cls = list(range(mk.S // 2))
    _, cert = mk._apply_p(U, 0, rho.unsqueeze(0), torch.as_tensor(cls))
    total = float(cert.sum())
    with pytest.raises(ValueError, match=f"over {len(cls)} outcome"):
        mk.condition(U, 0, rho, cls, tol=total / 2, route="auto")
    got = mk.condition(U, 0, rho, cls, tol=2 * total, route="auto")
    ref_k, _ = _kernel(4, 2, seed=90)
    ref = ref_k.condition(U, 0, rho, cls, route="F")
    assert float((got - ref).abs().max()) < 1e-14


@pytest.mark.parametrize("lossy", [False, True])
def test_condition_with_a_passive_rest(lossy):
    """condition(rest=X) on a register [D X, D X] whose X dimensions belong to spins outside the kernel (kernel spins
    first): (E_C (x) id_X)(rho), by P' and F, against E_C applied block by block (route F), for a single outcome, a
    class and a class over half the patterns (the complement); the pairing result is kept (the blocks' certificates
    are judged against Tr rho of the whole register, so off-diagonal blocks of trace ~0 do not force F)."""
    from test_internal_modes_lost import _lossy_emitter
    if lossy:
        ems = [_lossy_emitter(1, 2, 3, 1, 2, (2 * j, 2 * j + 1), seed=95 + j) for j in range(2)]
        mk, U = SequenceKernel(ems, 4), haar_unitary(4, 96)
    else:
        mk, U = _kernel(2, 3, seed=95)
    D, X = mk.chi, 3
    full = _rho(D * X, 2, 97)
    blocks = full.reshape(2, D, X, D, X).permute(0, 2, 4, 1, 3).reshape(2 * X * X, D, D)
    for cls in ([1], [0, 2, 5], list(range(mk.S // 2 + 1))):
        ref = mk._apply_f(U, 0, blocks, mk._class(cls), summed=True)[:, 0]
        ref = ref.reshape(2, X, X, D, D).permute(0, 3, 1, 4, 2).reshape(2, D * X, D * X)
        got_p = mk.condition(U, 0, full, cls, route="P", rest=X)
        got_f = mk.condition(U, 0, full, cls, route="F", rest=X)
        assert float((got_p - ref).abs().max()) < 1e-14 and torch.equal(got_f, ref)   # class sums: the criterion
        assert not torch.equal(got_p, got_f)                                   # the pairing result, not a fallback
    one = mk.condition(U, 0, full[0], [1], route="P", rest=X)
    assert one.shape == (D * X, D * X)
    with pytest.raises(ValueError, match="rest"):
        mk.condition(U, 0, full[0, :D, :D], [1], rest=X)


def test_emission_channel_is_the_sum_of_all_outcomes():
    """The complement path's emission channel equals the sum of every outcome's channel (F), within its certificate."""
    from photonic_engines.slos.internal import emission_channel
    mk, U = _kernel(3, 3, seed=47)
    rho = _rho(mk.chi, 2, 48)
    A = mk._pairing_inputs(0)[0]
    E, cert = emission_channel(A, rho)
    ref = mk._apply_f(U, 0, rho, torch.arange(mk.S)).sum(1)
    err = (E - ref).flatten(1).norm(dim=1)
    assert bool((cert > 0).all()) and float(cert.max()) < 1e-13
    assert float(err.max()) < 1e-14


def test_dispatcher(monkeypatch):
    """Uncalibrated device: the channel route; calibrated: the pairing route only below half the channel route's cost;
    the fine set over max_bytes: the pairing route."""
    mk, U = _kernel(3, 8, seed=60)
    monkeypatch.setattr(sequence_modes, "COST", {})
    assert mk._route(U, 0, mk.S, 1) == "F"
    monkeypatch.setattr(sequence_modes, "COST", {"cpu": {"F_build": 1.0, "F_apply": 1.0, "P": 1e-9}})
    assert mk._route(U, 0, mk.S, 1) == "P"
    monkeypatch.setattr(sequence_modes, "COST", {"cpu": {"F_build": 1e-9, "F_apply": 1e-9, "P": 1.0}})
    assert mk._route(U, 0, mk.S, 1) == "F"
    tiny, U2 = _kernel(3, 8, seed=60, max_bytes=1000)
    assert tiny._route(U2, 0, tiny.S, 1) == "P"


def test_fine_set_over_max_bytes_never_builds_the_stage(monkeypatch):
    """S_fine is C(m K_b + k - 1, k) in closed form: the dispatcher, route F's refusal and the fallback's refusal
    decide without building the fine stage (its host tables scale as S_fine; at k = 6, K = 8 they exhaust host RAM)."""
    from photonic_engines.slos import internal
    mk, U = _kernel(3, 8, seed=62, max_bytes=1000)
    Kb = mk._kb(U, 0)
    assert mk._s_fine(Kb) == internal.fine_stage(3, mk.m, Kb, "cpu").S_fine

    def no_stage(*a, **kw):
        raise AssertionError("fine stage built although the fine set is over max_bytes")
    monkeypatch.setattr(internal, "fine_stage", no_stage)
    monkeypatch.setattr(internal, "_STAGES", {})
    mk.clear_cache()
    assert mk._route(U, 0, mk.S, 1) == "P"
    with pytest.raises(NotImplementedError, match="max_bytes"):
        mk.kraus(U, 0)
    rho = _rho(mk.chi, 1, 63)[0]
    with pytest.raises(ValueError, match="certificate"):
        mk.outcome_law(U, 0, rho, tol=0.0, route="P")
    assert mk.outcome_law(U, 0, rho, route="auto").shape == (mk.S,)


@pytest.mark.gpu
def test_pairing_route_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    mk, U = _kernel(3, 3, seed=70)
    gk = SequenceKernel(mk.emitters, mk.m, device="cuda")
    rho = _rho(mk.chi, 2, 71)
    a = mk.outcome_law(U, 0, rho, route="P")
    g = gk.outcome_law(U, 0, rho.cuda(), route="P").cpu()
    assert float((a - g).abs().max()) < 1e-14


def test_glynn_normalisation_on_a_bunched_pair():
    """Two photons in one row (t! = 2): the pairing route against the channel route for every bunched outcome."""
    mk, U = _kernel(2, 2, seed=80)
    pats = mk.patterns
    bunched = [i for i, p in enumerate(pats) if p.max() == 2]
    rho = _rho(mk.chi, 1, 81)
    a = mk.condition(U, 0, rho, bunched, route="P")
    b = mk.condition(U, 0, rho, bunched, route="F")
    assert bunched and float((a - b).abs().max()) < 1e-15
    assert math.isclose(float(mk._tfact[bunched[0]]), 2.0)
