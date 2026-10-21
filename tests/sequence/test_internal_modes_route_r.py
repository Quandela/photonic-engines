"""Internal photon modes, the range route R (docs/SEQUENCE.md §14): the compressed channel of every outcome without the
fine set (range samples K(w) + a least-squares core from pairing applies), checked against route F's compression of
the fine set (itself checked against the expanded construction in test_internal_modes_route_f.py)."""
import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_route_f import _mode_emitter, _superops

from photonic_engines.sequence import SequenceKernel, bin_emitter
from photonic_engines.slos import internal


def _structured(n, L, K, d, modes, seed):
    """A label-preserving emitter of low internal rank: A[l, k] = f_l(k) |l mod d><l mod d| (complete for L = d)."""
    rng = np.random.default_rng(seed)
    ops = np.zeros((n, L, K, d, d), dtype=complex)
    for b in range(n):
        f = rng.normal(size=(L, K)) + 1j * rng.normal(size=(L, K))
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        for l in range(L):
            ops[b, l, :, l % d, l % d] = f[l]
    s0 = np.zeros(d, dtype=complex)
    s0[0] = 1
    return bin_emitter(ops, modes, s0)


def _weak(n, L, K, d, modes, seed):
    """Weak internal modes (H's probe): modes 2, 3, 4 scaled by 1e-4, 1e-7, 1e-9 before the isometry — a spread Choi
    spectrum (a wavepacket with a small distinguishable tail)."""
    rng = np.random.default_rng(seed)
    ops = np.empty((n, L, K, d, d), dtype=complex)
    for b in range(n):
        z = rng.normal(size=(L, K, d, d)) + 1j * rng.normal(size=(L, K, d, d))
        z *= np.array([1.0, 1e-4, 1e-7, 1e-9][:K]).reshape(1, K, 1, 1)
        q, _ = np.linalg.qr(z.reshape(L * K * d, d))
        ops[b] = q.reshape(L, K, d, d)
    s0 = np.zeros(d, dtype=complex)
    s0[0] = 1
    return bin_emitter(ops, modes, s0)


def _kernel(k, K, seed=0, structured=False, weak=False, **kw):
    m = 2 * k
    make = _weak if weak else _structured if structured else _mode_emitter
    ems = [make(2, 2, K, 2, (2 * j, 2 * j + 1), seed=seed + j) for j in range(k)]
    return SequenceKernel(ems, m, **kw), haar_unitary(m, seed + 50)


@pytest.mark.parametrize("k,K,structured", [(1, 3, False), (2, 3, False), (2, 8, False), (2, 8, True),
                                            (3, 3, False), (3, 4, True)])
def test_route_r_equals_the_compressed_fine_set(k, K, structured):
    """Per outcome: the same channel as F's compressed set (< 1e-14 on the Choi matrices), the same operator count,
    and a certificate at least the measured error of J_t / chi (bunched outcomes included)."""
    mk, U = _kernel(k, K, seed=10 * k + K, structured=structured)
    for b in range(2):
        KF, cF = mk.kraus(U, b, compress=1e-12, route="F"), mk.coarse(U, b, compress=1e-12, route="F")
        KR, cR = mk.kraus(U, b, compress=1e-12, route="R"), mk.coarse(U, b, compress=1e-12, route="R")
        JF, JR = _superops(KF, cF, mk.S), _superops(KR, cR, mk.S)
        assert float((JF - JR).abs().max()) < 1e-14
        assert torch.equal(torch.bincount(cF, minlength=mk.S), torch.bincount(cR, minlength=mk.S))
    A, inm, rows, tf = mk._pairing_inputs(0)
    F_t = internal.fine_counts(torch.as_tensor(np.asarray(mk.patterns)), A.shape[2])
    Kr, tr, cert, failed = internal.range_kraus(U, A, inm, rows, tf, F_t, 1e-12)
    JF = _superops(mk.kraus(U, 0, compress=1e-12, route="F"), mk.coarse(U, 0, compress=1e-12, route="F"), mk.S)
    err = (JF - _superops(Kr, tr, mk.S)).flatten(1).norm(dim=1) / mk.chi
    assert not bool(failed.any()) and bool((err <= cert).all()) and float(cert.max()) < 1e-14


@pytest.mark.parametrize("rtol", [1e-6, 1e-12])
def test_weak_modes_spread_spectrum(monkeypatch, rtol):
    """Weak internal modes (k = 3, K = 3): no failure; per outcome the estimate bounds |J_R - J_F| / chi; at most F's
    operator count (equal at rtol 1e-6, where the spectrum's gap at rtol is resolved); fewer pairing applies than the
    matrix units (the range is cut at tol, not rtol)."""
    mk, U = _kernel(3, 3, seed=40, weak=True)
    applies = [0]
    orig = internal.pairing_apply

    def counting(U_, A_, inm_, rows_, tf_, rho_, chunk_bytes=1 << 30):
        applies[0] += rows_.shape[0] * rho_.shape[0]
        return orig(U_, A_, inm_, rows_, tf_, rho_, chunk_bytes=chunk_bytes)
    monkeypatch.setattr(internal, "pairing_apply", counting)
    A, inm, rows, tf = mk._pairing_inputs(0)
    F_t = internal.fine_counts(torch.as_tensor(np.asarray(mk.patterns)), A.shape[2])
    Kr, tr, cert, failed = internal.range_kraus(U, A, inm, rows, tf, F_t, rtol)
    KF, cF = mk.kraus(U, 0, compress=rtol, route="F"), mk.coarse(U, 0, compress=rtol, route="F")
    err = (_superops(KF, cF, mk.S) - _superops(Kr, tr, mk.S)).flatten(1).norm(dim=1) / mk.chi
    nF, nR = torch.bincount(cF, minlength=mk.S), torch.bincount(tr, minlength=mk.S)
    assert not bool(failed.any()) and bool((err <= cert).all()) and float(cert.max()) <= 1e-14
    assert bool((nR <= nF).all()) and (rtol < 1e-9 or torch.equal(nR, nF))
    assert applies[0] < mk.S * mk.chi ** 2


def test_low_internal_rank_is_revealed():
    """k = 2, K = 8, label-preserving emitters: 64 fine operators per collision-free outcome, Choi rank 4."""
    mk, U = _kernel(2, 8, seed=3, structured=True)
    cR = mk.coarse(U, 0, compress=1e-12, route="R")
    free = [i for i, p in enumerate(mk.patterns) if p.max() == 1]
    F_t = internal.fine_counts(torch.as_tensor(np.asarray(mk.patterns)), 8)
    assert bool((F_t[free] == 64).all())
    assert torch.equal(torch.bincount(cR, minlength=mk.S)[free], torch.full((len(free),), 4))


def test_fine_counts_match_the_fine_stage():
    """F_t in closed form equals the number of fine patterns of the stage mapping to each spatial pattern."""
    mk, U = _kernel(3, 3, seed=5)
    Kb = mk._kb(U, 0)
    stage = internal.fine_stage(3, mk.m, Kb, "cpu")
    counts = torch.bincount(mk.pattern_index(stage.spatial_occ), minlength=mk.S)
    assert torch.equal(internal.fine_counts(torch.as_tensor(np.asarray(mk.patterns)), Kb), counts)


def test_incomplete_range_is_caught(monkeypatch):
    """Range samples confined to one direction: the core's residual exposes the missing range, the outcomes are
    recomputed from the matrix units, and the channel is still F's."""
    mk, U = _kernel(2, 3, seed=7)
    JF = _superops(mk.kraus(U, 0, compress=1e-12, route="F"), mk.coarse(U, 0, compress=1e-12, route="F"), mk.S)
    orig = internal.range_samples

    def one_direction(*a, **kw):
        S = orig(*a, **kw)
        return S[:, :, :1].expand_as(S).clone()
    monkeypatch.setattr(internal, "range_samples", one_direction)
    A, inm, rows, tf = mk._pairing_inputs(0)
    F_t = internal.fine_counts(torch.as_tensor(np.asarray(mk.patterns)), A.shape[2])
    Kr, tr, cert, failed = internal.range_kraus(U, A, inm, rows, tf, F_t, 1e-12)
    assert not bool(failed.any())
    assert float((JF - _superops(Kr, tr, mk.S)).abs().max()) < 1e-14


def test_weak_missed_direction_is_estimated():
    """Samples projected off the weakest eigen-direction of J_t (k = 3, K = 3; missed eigenvalue ~1e-3 of the largest):
    the first-pass estimate is at least half the true error (measured margin >= 0.58 over 396 misses, docs §14) and
    above tol, so the outcome is recomputed from the matrix units — the result is F's channel."""
    mk, U = _kernel(3, 3, seed=21)
    D, t = mk.chi, 7
    Ks = mk.kraus(U, 0)[mk.coarse(U, 0) == t].reshape(-1, D * D)
    J = Ks.T @ Ks.conj()                                                     # sum vec K vec K^dag, vec row-major
    lam, V = torch.linalg.eigh(J)
    i = int(torch.nonzero(lam > 1e-12 * lam[-1])[0])
    assert float(lam[i] / lam[-1]) < 1e-2
    Pq = torch.eye(D * D, dtype=J.dtype) - torch.outer(V[:, i], V[:, i].conj())
    orig = internal.range_samples
    A, inm, rows, tf = mk._pairing_inputs(0)
    F_t = internal.fine_counts(torch.as_tensor(np.asarray(mk.patterns)), A.shape[2])
    sl = slice(t, t + 1)

    def projected(*a, **kw):
        return Pq @ orig(*a, **kw)
    internal.range_samples = projected
    try:
        K1, _, c1, _ = internal.range_kraus(U, A, inm, rows[sl], tf[sl], F_t[sl], 1e-12, tol=float("inf"))
        K2, _, c2, f2 = internal.range_kraus(U, A, inm, rows[sl], tf[sl], F_t[sl], 1e-12)
    finally:
        internal.range_samples = orig
    err = float((K1.reshape(-1, D * D).T @ K1.reshape(-1, D * D).conj() - J).norm()) / D
    assert float(c1[0]) >= 0.5 * err and float(c1[0]) > 1e-14
    assert not bool(f2.any()) and float(c2[0]) < 1e-14
    assert float((K2.reshape(-1, D * D).T @ K2.reshape(-1, D * D).conj() - J).abs().max()) < 1e-14


def test_chunking_keeps_the_channel():
    """A tiny chunk budget (one outcome per chunk in both stages) gives the same channel and operator counts."""
    mk, U = _kernel(3, 3, seed=6)
    small, _ = _kernel(3, 3, seed=6, chunk_bytes=1)
    a = _superops(mk.kraus(U, 0, compress=1e-12, route="R"), mk.coarse(U, 0, compress=1e-12, route="R"), mk.S)
    cb = small.coarse(U, 0, compress=1e-12, route="R")
    b = _superops(small.kraus(U, 0, compress=1e-12, route="R"), cb, mk.S)
    assert float((a - b).abs().max()) < 1e-14
    assert torch.equal(torch.bincount(mk.coarse(U, 0, compress=1e-12, route="R"), minlength=mk.S),
                       torch.bincount(cb, minlength=mk.S))


def test_zero_outcomes_have_no_operator():
    """U = 1: patterns no photon can reach have E_t = 0 exactly; route R gives them no operator, as F does."""
    mk, _ = _kernel(2, 3, seed=8)
    U = torch.eye(mk.m, dtype=torch.complex128)
    cF, cR = mk.coarse(U, 0, compress=1e-12, route="F"), mk.coarse(U, 0, compress=1e-12, route="R")
    nF, nR = torch.bincount(cF, minlength=mk.S), torch.bincount(cR, minlength=mk.S)
    assert bool((nF == 0).any()) and torch.equal(nF, nR)
    JF = _superops(mk.kraus(U, 0, compress=1e-12, route="F"), cF, mk.S)
    assert float((JF - _superops(mk.kraus(U, 0, compress=1e-12, route="R"), cR, mk.S)).abs().max()) < 1e-14


def test_route_r_fallback_raise_and_auto():
    """tol = 0: every outcome is recomputed by F (= F's compressed set exactly); with F over max_bytes it raises; auto
    takes R when the fine set does not fit (F would refuse), and R needs compress."""
    mk, U = _kernel(2, 3, seed=40)
    KF, cF = mk.kraus(U, 0, compress=1e-12, route="F"), mk.coarse(U, 0, compress=1e-12, route="F")
    assert torch.equal(mk.kraus(U, 0, compress=1e-12, route="R", tol=0.0), KF)
    assert torch.equal(mk.coarse(U, 0, compress=1e-12, route="R", tol=0.0), cF)
    with pytest.raises(ValueError, match="compress"):
        mk.kraus(U, 0, route="R")
    small, U2 = _kernel(2, 3, seed=40, max_bytes=1000)
    with pytest.raises(ValueError, match="certificate"):
        small.kraus(U2, 0, compress=1e-12, route="R", tol=0.0)
    with pytest.raises(NotImplementedError, match="max_bytes"):
        small.kraus(U2, 0)
    Ka, ca = small.kraus(U2, 0, compress=1e-12), small.coarse(U2, 0, compress=1e-12)              # auto: R
    assert float((_superops(KF, cF, mk.S) - _superops(Ka, ca, mk.S)).abs().max()) < 1e-14


@pytest.mark.gpu
def test_route_r_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    mk, U = _kernel(3, 3, seed=70)
    gk = SequenceKernel(mk.emitters, mk.m, device="cuda")
    a = _superops(mk.kraus(U, 0, compress=1e-12, route="R"), mk.coarse(U, 0, compress=1e-12, route="R"), mk.S)
    Kg, cg = gk.kraus(U, 0, compress=1e-12, route="R"), gk.coarse(U, 0, compress=1e-12, route="R")
    g = _superops(Kg.cpu(), cg.cpu(), mk.S)
    assert float((a - g).abs().max()) < 1e-14
