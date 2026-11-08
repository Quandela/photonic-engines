"""Gaussian inputs (docs/GAUSSIAN.md): loop hafnians, the pure-state probabilities, the chain-rule
sampler, loss.

Oracles, independent of each other:
  * `lhaf_brute` (recursion over matchings) for every loop-hafnian evaluation;
  * `GaussianInput.truncated(d)` through `slos.ProductSLOSKernel` - exact for every lossless pattern
    with <= d-1 photons, and for lossy patterns up to the truncated tail, which is bounded here;
  * The Walrus / Piquasso when installed (importorskip), for displaced and lossy states.
Sampling tests use chi-square against the exact distribution with the threshold derived from the
chi-square law (dof + 5 sqrt(2 dof)), fixed seeds.
"""
from __future__ import annotations

import math
import os
from collections import Counter

import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines.gaussian import lhaf, lhaf_brute, lhaf_poly, pure_probability, sample
from photonic_engines.gaussian.hafnian import expand
from photonic_engines.gaussian.sampler import _amps, _norm
from photonic_engines.imperfections import lossy_squeezing
from photonic_engines.slos.kernel import ProductSLOSKernel
from photonic_engines.states import GaussianInput

# The Triton kernels forced on below run on the device when there is one (a CUDA host does not set TRITON_INTERPRET,
# and a compiled Triton kernel cannot take CPU tensors), under the CPU interpreter otherwise.
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") != "1" else "cpu"

EPS = np.finfo(np.float64).eps


def _rand_sym(N, rng):
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    return (A + A.T) / 2


# ---- loop hafnian ---------------------------------------------------------------------------------
@pytest.mark.parametrize("N", list(range(11)))
def test_pair_sieve_matches_brute_force(N):
    rng = np.random.default_rng(100 + N)
    A = _rand_sym(N, rng); D = rng.normal(size=N) + 1j * rng.normal(size=N)
    ref = lhaf_brute(A, D)
    # (N-1)!! terms of O(1) each: relative rounding ~ N eps sqrt(#terms) is generous
    assert abs(lhaf(A, D) - ref) <= 1e3 * EPS * max(1.0, abs(ref)) * (N + 1)


@pytest.mark.parametrize("N", [1, 2, 5, 8, 9])
def test_lhaf_poly_is_the_polynomial_in_z(N):
    rng = np.random.default_rng(7 + N)
    A = _rand_sym(N, rng); D0 = rng.normal(size=N) + 1j * rng.normal(size=N); D1 = rng.normal(size=N) + 0.3j
    c = lhaf_poly(A, D0, D1).numpy()
    assert c.shape == (N + 1,)
    for z in (0.0, 0.4 - 0.3j, -1.2j, 0.9):
        ref = lhaf_brute(A, D0 + z * D1)
        assert abs(np.polynomial.polynomial.polyval(z, c) - ref) <= 1e4 * EPS * max(1.0, abs(ref))


def test_lhaf_ignores_the_diagonal_and_vanishes_without_loops_at_odd_size():
    rng = np.random.default_rng(3)
    A = _rand_sym(5, rng)
    assert abs(lhaf(A, np.zeros(5))) < 1e-13
    B = A.copy(); np.fill_diagonal(B, 7.0)
    D = rng.normal(size=5) + 0j
    assert abs(lhaf(A, D) - lhaf(B, D)) < 1e-12 * abs(lhaf(A, D))


def test_lhaf_against_thewalrus():
    tw = pytest.importorskip("thewalrus")
    rng = np.random.default_rng(11)
    for N in (6, 11, 16):
        A = _rand_sym(N, rng) / math.sqrt(N); D = rng.normal(size=N) + 1j * rng.normal(size=N)
        A2 = A.copy(); np.fill_diagonal(A2, D)
        ref = tw.hafnian(A2, loop=True)
        # 2^(N/2) sieve terms against a different algorithm: tolerance grows with the term count
        assert abs(lhaf(A, D) - ref) <= 1e2 * EPS * 2 ** (N / 2) * max(1.0, abs(ref))


# ---- the single-mode step --------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 2, 6, 12])
def test_closed_form_norm_equals_the_series(L):
    rng = np.random.default_rng(L)
    q = rng.normal(size=L) + 1j * rng.normal(size=L)
    b, g = 0.55 * np.exp(0.7j), 0.9 - 0.4j
    c = _amps(q, b, g, 600)                                  # |b|^k tail at k = 600 is < 1e-150
    assert abs(_norm(q, b, g) - np.sum(np.abs(c) ** 2)) <= 1e3 * EPS * np.sum(np.abs(c) ** 2)


# ---- states and probabilities -------------------------------------------------------------------
def test_photon_number_law_and_truncation():
    st = GaussianInput(np.array([0.3, 0.8, 1.1]), (0, 2, 3), 4)
    p = st.photon_number_probs(300)                          # tanh^2(1.1) = 0.64 per pair: tail ~ 1e-28
    assert abs(p.sum() - 1) < 1e-12
    assert abs(np.dot(np.arange(301), p) - st.mean_photons()) < 1e-10
    ps = st.truncated(11)
    assert np.allclose(ps.photon_number_probs()[:11], p[:11], atol=1e-15)   # N <= d-1 untouched


def test_pure_probability_matches_truncated_slos():
    st = GaussianInput(np.array([0.5, 0.9]), (0, 2), 3)
    U = haar_unitary(3, 21)
    d = 8
    dist = ProductSLOSKernel(st.truncated(d), dtype=torch.complex128).probs(U)
    for N in range(d):
        for occ, p in zip(dist.occupations(N), dist.probs[N].numpy()):
            assert abs(pure_probability(st, U, occ) - p) <= 1e-13 * max(p, 1e-3)


def test_displaced_pure_probability_against_thewalrus():
    pytest.importorskip("thewalrus")
    from thewalrus import symplectic as sy
    from thewalrus.quantum import probabilities
    m = 3
    st = GaussianInput(np.array([0.4, 0.7]), (0, 1), m, disp=np.array([0.3 + 0.2j, -0.5j]))
    U = haar_unitary(m, 4).numpy()
    hbar = 2.0
    cov = np.eye(2 * m) * hbar / 2; mu = np.zeros(2 * m)
    for i, (r, dd) in enumerate(zip(st.r, st.disp)):
        S = sy.expand(sy.squeezing(r), i, m); cov = S @ cov @ S.T
        mu[i] += math.sqrt(2 * hbar) * dd.real; mu[i + m] += math.sqrt(2 * hbar) * dd.imag
    Sint = sy.interferometer(U); cov = Sint @ cov @ Sint.T; mu = Sint @ mu
    P = probabilities(mu, cov, cutoff=4, hbar=hbar)
    for pat in [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 0, 1), (0, 3, 0), (1, 2, 1)]:
        assert abs(pure_probability(st, U, pat) - P[pat]) <= 1e-12


# ---- loss -------------------------------------------------------------------------------------------
def test_lossy_squeezing_reproduces_the_lossy_covariance():
    r = np.array([0.0, 0.3, 1.1, 1.8])
    for eta in (1.0, 0.9, 0.5, 0.2):
        rp, w = lossy_squeezing(r, eta)
        vx = (eta * np.exp(-2 * r) + 1 - eta) / 4; vp = (eta * np.exp(2 * r) + 1 - eta) / 4
        assert np.allclose(np.exp(-2 * rp) / 4, vx, rtol=1e-14)
        assert np.allclose(np.exp(2 * rp) / 4 + w, vp, rtol=1e-13)
        assert (w >= 0).all() and (rp <= r + 1e-15).all()
    rp, w = lossy_squeezing(r, None)
    assert np.array_equal(rp, r) and not w.any()


def test_eta_one_is_the_lossless_path_exactly():
    st = GaussianInput.smsv(2, 3, 0.6)
    U = haar_unitary(3, 2)
    a = sample(st, U, 50, seed=9); b = sample(st, U, 50, loss=1.0, seed=9)
    assert np.array_equal(a, b)


# ---- the sampler ------------------------------------------------------------------------------------
def _chi2(samples, exact, min_expected=15):
    c = Counter(map(tuple, samples)); shots = len(samples)
    chi, dof = 0.0, 0
    for o, p in exact.items():
        e = p * shots
        if e >= min_expected:
            chi += (c.get(tuple(o), 0) - e) ** 2 / e; dof += 1
    return chi, dof


def _exact_from(dist, top):
    out = {}
    for N in range(top + 1):
        for occ, p in zip(dist.occupations(N), dist.probs[N].numpy()):
            out[tuple(int(x) for x in occ)] = float(p)
    return out


def test_sampler_lossless_matches_truncated_slos():
    st = GaussianInput(np.array([0.6, 0.45]), (0, 1), 3)
    U = haar_unitary(3, 31)
    d = 9
    exact = _exact_from(ProductSLOSKernel(st.truncated(d), dtype=torch.complex128).probs(U), d - 1)
    S = sample(st, U, 3000, seed=5)
    chi, dof = _chi2(S, exact)
    assert dof >= 8
    assert chi < dof + 5 * math.sqrt(2 * dof)


def test_sampler_photon_number_law_is_u_independent():
    st = GaussianInput(np.array([0.7, 0.7, 0.3]), (0, 2, 3), 4)
    U = haar_unitary(4, 8)
    S = sample(st, U, 3000, seed=6)
    law = st.photon_number_probs(60)
    c = np.bincount(S.sum(1), minlength=61)[:61]
    chi, dof = 0.0, 0
    for N, p in enumerate(law):
        if p * len(S) >= 15:
            chi += (c[N] - p * len(S)) ** 2 / (p * len(S)); dof += 1
    assert chi < dof + 5 * math.sqrt(2 * dof)


def test_sampler_lossy_matches_thinned_truncated_slos():
    st = GaussianInput(np.array([0.8, 0.5]), (0, 1), 3)
    U = haar_unitary(3, 12); eta = 0.6
    d = 21
    # tail of the truncation: P(a source has >= d photons), per source, bounds every lossy error
    tail = 1 - np.prod([GaussianInput(np.array([r]), (0,), 1).photon_number_probs(d - 1).sum() for r in st.r])
    assert tail < 1e-4
    exact = _exact_from(ProductSLOSKernel(st.truncated(d), dtype=torch.complex128).probs(U, loss=eta), 8)
    S = sample(st, U, 3000, loss=eta, seed=7)
    chi, dof = _chi2(S, exact)
    assert dof >= 8
    assert chi < dof + 5 * math.sqrt(2 * dof)


def _psd_sqrt(H):
    w, Q = np.linalg.eigh(H)
    return (Q * np.sqrt(np.maximum(w, 0.0))) @ Q.conj().T


def _lossy_T(seed):
    # a contraction with non-uniform loss inside the interferometer: U1 diag(sqrt eta_i) U2
    return haar_unitary(3, seed).numpy() @ np.diag([0.95, 0.5, 0.2]) @ haar_unitary(3, seed + 1).numpy()


def test_transfer_decomposition_rebuilds_the_covariance():
    # pure part U' prod S(r')|0> plus noise W >= 0 must give back T_R V_in T_R^T + (I - T_R T_R^T)/4, and a unitary T
    # must give no noise and the Bargmann matrix of the lossless path
    from photonic_engines.gaussian.sampler import _quadrature_matrix, transfer_decomposition
    m = 5; rng = np.random.default_rng(4)
    st = GaussianInput(np.array([0.8, 0.5, 0.3]), (0, 2, 3), m)
    U1 = haar_unitary(m, 50).numpy(); U2 = haar_unitary(m, 51).numpy()
    T = U1 @ np.diag(rng.uniform(0.2, 1.0, m)) @ U2
    rp, Up, W, _ = transfer_decomposition(st, T)
    k = rp.size
    vin = np.full(2 * m, 0.25); vin[:k] = np.exp(-2 * rp) / 4; vin[m:m + k] = np.exp(2 * rp) / 4
    UR = _quadrature_matrix(Up)
    v0 = np.full(2 * m, 0.25); v0[list(st.modes)] = np.exp(-2 * st.r) / 4; v0[[m + i for i in st.modes]] = np.exp(2 * st.r) / 4
    TR = _quadrature_matrix(T)
    V = (TR * v0) @ TR.T + (np.eye(2 * m) - TR @ TR.T) / 4
    tol = 100 * m * EPS                                          # O(m) sums of O(1) entries, a few eigendecompositions
    assert np.abs(Up.conj().T @ Up - np.eye(m)).max() <= tol
    assert np.abs((UR * vin) @ UR.T + W - V).max() <= tol
    assert np.linalg.eigvalsh(W).min() >= -tol
    rp, Up, W, _ = transfer_decomposition(st, U1)
    assert np.abs(W).max() <= tol
    B0, _, _ = st.bargmann(U1)
    Uc = Up[:, :rp.size]
    assert np.abs(-(Uc * np.tanh(rp)) @ Uc.T - B0).max() <= tol
    with pytest.raises(ValueError, match="contraction"):
        transfer_decomposition(st, 1.1 * U1)


def test_sample_transfer_matches_dilated_truncated_slos():
    # Independent oracle: T is the top-left block of the unitary dilation [[T, (I - TT^dag)^1/2],
    # [(I - T^dag T)^1/2, -T^dag]] on 2m modes; the lossless truncated SLOS on the dilation, marginalised over the
    # m environment modes, is the law of the lossy output. Uniform loss at the mean transmission fails this test
    # (chi2 = 152 on 7 dof at these settings), so the non-uniform part is really tested.
    from collections import Counter

    from photonic_engines.gaussian import sample_transfer
    m = 3; st = GaussianInput(np.array([0.6, 0.35]), (0, 1), m)
    T = _lossy_T(5)
    Ud = np.block([[T, _psd_sqrt(np.eye(m) - T @ T.conj().T)], [_psd_sqrt(np.eye(m) - T.conj().T @ T), -T.conj().T]])
    assert np.abs(Ud.conj().T @ Ud - np.eye(2 * m)).max() < 1e-13
    d = 13
    tail = 1 - np.prod([GaussianInput(np.array([r]), (0,), 1).photon_number_probs(d - 1).sum() for r in st.r])
    assert tail < 1e-4
    dist = ProductSLOSKernel(GaussianInput(st.r, st.modes, 2 * m).truncated(d), dtype=torch.complex128).probs(
        torch.as_tensor(Ud))
    exact = Counter()
    for N in range(d):
        for occ, p in zip(dist.occupations(N), dist.probs[N].numpy()):
            exact[tuple(int(x) for x in occ[:m])] += float(p)
    S = sample_transfer(st, T, 3000, seed=3)
    from photonic_engines.gaussian import transfer_decomposition
    assert np.array_equal(S[:50], sample_transfer(st, None, 50, seed=3, decomposition=transfer_decomposition(st, T)))
    chi, dof = _chi2(S, exact)
    assert dof >= 6
    assert chi < dof + 5 * math.sqrt(2 * dof)


def test_sample_transfer_displaced_unitary_matches_pure_probabilities():
    # T unitary: no noise, the displaced pure state; exact law from `pure_probability` (loop hafnians of `bargmann`)
    from itertools import product

    from photonic_engines.gaussian import sample_transfer
    st = GaussianInput(np.array([0.5, 0.3]), (0, 1), 3, disp=np.array([0.4, -0.2j]))
    U = haar_unitary(3, 61).numpy()
    exact = {pat: pure_probability(st, U, pat) for pat in product(range(9), repeat=3) if sum(pat) <= 8}
    assert 1 - sum(exact.values()) < 1e-3
    S, stats = sample_transfer(st, U, 3000, seed=2, return_stats=True)
    assert len(stats.photons) == 3000 and not any(stats.censored)
    chi, dof = _chi2(S, exact)
    assert dof >= 8
    assert chi < dof + 5 * math.sqrt(2 * dof)


def test_mean_ordered_chain_has_the_same_law():
    # order="mean" visits the outputs in increasing mean photon number: a fixed relabelling, so the same law.
    # A non-uniform U (outputs with different means) so that the order is not the natural one
    st = GaussianInput(np.array([0.6, 0.35]), (0, 1), 3, disp=np.array([0.3j, 0.0]))
    U = haar_unitary(3, 41)[[2, 1, 0]]
    Uc = np.asarray(U)[:, [0, 1]]
    means = (np.abs(Uc) ** 2) @ np.sinh([0.6, 0.35]) ** 2 + np.abs(Uc @ np.array([0.3j, 0.0])) ** 2
    assert list(np.argsort(means)) != [0, 1, 2]                          # the chain order really changes
    exact = {}
    from itertools import product
    for pat in product(range(9), repeat=3):
        if sum(pat) <= 8:
            exact[pat] = pure_probability(st, U, pat)
    assert sum(exact.values()) > 0.995
    S = sample(st, U, 3000, seed=9, order="mean")
    chi, dof = _chi2(S, exact)
    assert dof >= 8
    assert chi < dof + 5 * math.sqrt(2 * dof)
    with pytest.raises(ValueError):
        sample(st, U, 1, order="bright")


def test_sampler_displaced_matches_pure_probabilities():
    st = GaussianInput(np.array([0.5, 0.3]), (0, 1), 3, disp=np.array([0.4 - 0.1j, 0.2j]))
    U = haar_unitary(3, 41)
    exact = {}
    from itertools import product
    for pat in product(range(7), repeat=3):
        if sum(pat) <= 6:
            exact[pat] = pure_probability(st, U, pat)
    assert sum(exact.values()) > 0.995
    S = sample(st, U, 3000, seed=8)
    chi, dof = _chi2(S, exact)
    assert dof >= 8
    assert chi < dof + 5 * math.sqrt(2 * dof)


def test_sampler_vacuum_modes_and_stats():
    st = GaussianInput(np.array([0.9]), (1,), 4)            # one source among 4 modes
    U = haar_unitary(4, 13)
    S, stats = sample(st, U, 400, seed=3, return_stats=True)
    assert S.shape == (400, 4) and (S >= 0).all()
    assert stats.photons == S.sum(1).tolist()
    assert max(stats.norm_rel_err) == 0.0
    assert all(k % 2 == 0 for k in S.sum(1))                 # one squeezer, no loss: even totals


def test_expand_is_the_generating_function_coefficient():
    # coefficient of w0^2 w1 in exp(1/2 w^T B w + g^T w) = lhaf(expand) / (2! 1!)
    B = np.array([[0.3, 0.2], [0.2, -0.1]], dtype=complex); g = np.array([0.5, -0.4j])
    A, D = expand(B, g, [2, 1])
    # direct: terms with w0^2 w1: (1/2 B00 w0^2)(g1 w1) + (g0 w0)(B01 w0 w1) + (g0 w0)^2 (g1 w1) / 2
    direct = 0.5 * B[0, 0] * g[1] + g[0] * B[0, 1] + g[0] ** 2 * g[1] / 2
    assert abs(lhaf_brute(A, D) / 2 - direct) < 1e-15


def _piquasso_state(pq, r, U, eta=None):
    m = U.shape[0]
    with pq.Program() as prog:
        for i, ri in enumerate(r):
            pq.Q(i) | pq.Squeezing(r=float(ri))
        pq.Q(*range(m)) | pq.Interferometer(U)
        if eta is not None:
            for i in range(m):
                pq.Q(i) | pq.Attenuator(theta=math.acos(math.sqrt(eta)))   # amplitude cos(theta) = sqrt(eta)
    return pq.GaussianSimulator(d=m, config=pq.Config(cutoff=8)).execute(prog).state


def test_pure_and_lossy_probabilities_against_piquasso():
    pq = pytest.importorskip("piquasso")
    m = 3; r = np.array([0.4, 0.7]); U = haar_unitary(m, 4).numpy()
    st = GaussianInput(r, (0, 1), m)
    pure = _piquasso_state(pq, r, U)
    pats = [(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, 1, 2), (0, 2, 2), (3, 1, 0)]
    for pat in pats:
        assert abs(pure_probability(st, U, pat) - pure.get_particle_detection_probability(pat)) <= 1e-13
    eta = 0.55
    lossy = _piquasso_state(pq, r, U, eta)
    d = 25                                                   # truncation tail < 1e-9 at r = 0.7
    dist = _exact_from(ProductSLOSKernel(st.truncated(d), dtype=torch.complex128).probs(torch.as_tensor(U), loss=eta), 5)
    for pat in pats:
        assert abs(dist[pat] - lossy.get_particle_detection_probability(pat)) <= 1e-8


def test_schmidt_modes_are_independent_instances():
    # two Schmidt modes: the output is the convolution of two GBS distributions through the same U
    st = GaussianInput(np.array([0.8, 0.6]), (0, 1), 3)
    U = haar_unitary(3, 17); lam = (0.75, 0.25)
    d = 11
    parts = []
    for lk in lam:
        sk = GaussianInput(st.r * math.sqrt(lk), st.modes, st.m)
        parts.append(_exact_from(ProductSLOSKernel(sk.truncated(d), dtype=torch.complex128).probs(U), d - 1))
    exact = {}
    for a, pa in parts[0].items():
        for b, pb in parts[1].items():
            if sum(a) + sum(b) <= d - 1:
                key = tuple(x + y for x, y in zip(a, b))
                exact[key] = exact.get(key, 0.0) + pa * pb
    S = sample(st, U, 3000, seed=4, schmidt=lam)
    chi, dof = _chi2(S, exact)
    assert dof >= 8
    assert chi < dof + 5 * math.sqrt(2 * dof)
    with pytest.raises(ValueError):
        sample(st, U, 1, schmidt=(0.5, 0.4))


@pytest.mark.gpu
def test_lhaf_poly_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    rng = np.random.default_rng(5)
    for N in (7, 16, 21):
        A = _rand_sym(N, rng) / math.sqrt(N); D0 = rng.normal(size=N) + 0j; D1 = rng.normal(size=N) + 1j
        a = lhaf_poly(A, D0, D1, device="cpu"); b = lhaf_poly(A, D0, D1, device="cuda").cpu()
        assert torch.allclose(a, b, rtol=1e-11, atol=1e-11 * float(a.abs().max()))


# ---- strong simulation on the SLOS layers ---------------------------------------------------------
def test_gaussian_slos_equals_the_truncated_product_slos_and_the_hafnians():
    from photonic_engines.slos import GaussianSLOSKernel
    st = GaussianInput(np.array([0.5, 0.9, 0.3]), (0, 2, 3), 5)
    U = haar_unitary(5, 3)
    d = GaussianSLOSKernel(st, 7).probs(U)
    ref = ProductSLOSKernel(st.truncated(8), dtype=torch.complex128).probs(U)
    law = st.photon_number_probs(7)
    for k in range(8):
        assert torch.allclose(d.probs[k], ref.probs[k], rtol=0, atol=1e-15)
        assert abs(float(d.mass(k)) - law[k]) < 1e-14                     # U-independent photon-number law
    sd = GaussianInput(np.array([0.5, 0.7]), (0, 1), 3, disp=np.array([0.3, -0.2j]))
    U3 = haar_unitary(3, 1)
    d2 = GaussianSLOSKernel(sd, 5).probs(U3)
    for k in range(6):
        for occ, p in zip(d2.occupations(k), d2.probs[k].numpy()):
            assert abs(p - pure_probability(sd, U3, occ)) < 1e-15


def test_gaussian_slos_under_loss_is_short_by_at_most_the_tail():
    from photonic_engines.slos import GaussianSLOSKernel
    st = GaussianInput(np.array([0.6, 0.4]), (0, 1), 3)
    U = haar_unitary(3, 9); eta = 0.7
    k = GaussianSLOSKernel(st, 6)
    lo = k.probs(U, loss=eta)
    ref = ProductSLOSKernel(st.truncated(31), dtype=torch.complex128).probs(U, loss=eta)   # tail < 1e-12
    tail = k.tail_mass()
    for j in range(7):
        diff = ref.probs[j] - lo.probs[j]
        assert float(diff.min()) > -1e-15 and float(diff.sum()) <= tail + 1e-12


def test_lhaf_poly_condition_number_bounds_the_coefficients():
    rng = np.random.default_rng(12)
    A = _rand_sym(12, rng); D0 = rng.normal(size=12) + 0j; D1 = rng.normal(size=12) + 1j
    c, a = lhaf_poly(A, D0, D1, return_abs=True)
    assert torch.equal(c, lhaf_poly(A, D0, D1))
    assert bool((a >= c.abs() * (1 - 1e-12)).all())                      # triangle inequality per coefficient
    st = GaussianInput.smsv(3, 5, 0.6)
    _, stats = sample(st, haar_unitary(5, 1), 20, seed=0, return_stats=True)
    assert len(stats.round_off) == 20 and max(stats.round_off) < 1e-12


# ---- v2 sieve -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["fused", "loops"])
@pytest.mark.parametrize("N", [1, 2, 3, 5, 8, 11, 14])
def test_lhaf_poly_multi_equals_one_sieve_per_vector(monkeypatch, N, layout):
    import photonic_engines.gaussian.hafnian as H
    from photonic_engines.gaussian.hafnian import lhaf_poly_multi
    monkeypatch.setattr(H, "_FUSED", layout == "fused")
    rng = np.random.default_rng(40 + N)
    A = _rand_sym(N, rng); D0 = rng.normal(size=(3, N)) + 1j * rng.normal(size=(3, N)); D1 = rng.normal(size=(3, N)) + 0j
    M = lhaf_poly_multi(A, D0, D1)
    for v in range(3):
        ref = lhaf_poly(A, D0[v], D1[v])
        assert torch.allclose(M[v], ref, rtol=0, atol=1e4 * EPS * float(ref.abs().max()))


def test_generated_subsets_are_every_subset_once_with_its_sign():
    # the on-device path of _free_subsets (colex unranking, used above _TABLE_MAX pairs), against itertools
    import itertools
    n, dev = 9, torch.device("cpu")
    got = []
    for pb, idx, sgn in _gen_subsets(n, dev):
        assert idx.shape[0] <= 37 and idx.shape[1] == 2 * pb
        for row, sg in zip(idx.tolist(), sgn.tolist()):
            first = [x for x in row[:pb] if x != 2 * n]; second = [x for x in row[pb:] if x != 2 * n]
            assert second == [x + n for x in first] and sg == (-1.0 if (n - len(first)) % 2 else 1.0)
            got.append(tuple(sorted(first)))
    ref = [c for p in range(1, n + 1) for c in itertools.combinations(range(n), p)]
    assert sorted(got) == sorted(ref)


def _gen_subsets(n, dev):
    import photonic_engines.gaussian.hafnian as H
    old = H._TABLE_MAX
    H._TABLE_MAX = 0
    try:
        yield from H._free_subsets(n, dev, lambda pb: 37)
    finally:
        H._TABLE_MAX = old


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("counts", [[1] * 9, [2, 1, 1, 3, 1, 1], [4, 4, 2, 1]])
def test_multi_sieve_on_the_generated_path_equals_v1(monkeypatch, counts, fused):
    # the same sieve with the subsets generated on the device (as above _TABLE_MAX, i.e. N > 32) instead of
    # tabulated, small batches so that one bucket spans several batches and one batch several sizes p
    import photonic_engines.gaussian.hafnian as H
    monkeypatch.setattr(H, "_TABLE_MAX", 0)
    monkeypatch.setattr(H, "_FUSED", fused)                             # the GPU layout of _terms, run on the CPU
    monkeypatch.setattr(H, "_chunk_size_multi", lambda n, p, V, b, dev=None: 11)
    rng = np.random.default_rng(len(counts))
    m = len(counts)
    B = _rand_sym(m, rng); gam = rng.normal(size=(3, m)) + 1j * rng.normal(size=(3, m)); col = rng.normal(size=(3, m)) + 0j
    A, _ = expand(B, gam[0], counts); lab = np.repeat(np.arange(m), counts)
    M = H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab)
    for v in range(3):
        ref = lhaf_poly(A, gam[v, lab], col[v, lab])
        assert torch.allclose(M[v], ref, rtol=0, atol=1e4 * EPS * float(ref.abs().max()))


@pytest.mark.parametrize("rule", [(None, 2.0), (None, 0.0), (1.0, 2.0), (0.0, 2.0)])
@pytest.mark.parametrize("counts", [[1] * 9, [4, 1, 1, 1, 1, 1], [2, 2, 2, 1, 1], [3, 1, 1, 1, 1, 1, 1], [3, 3, 2, 2]])
def test_searched_pairing_gives_the_same_sieve(monkeypatch, counts, rule):
    # pairing="auto" with the search forced on (it starts at 14 pairs on the CPU): the value must not depend on
    # the pairing chosen and the choice must be reproducible. rule = (_ROUND_TOL, _PAIR_PREFER_SELF): None = the v2.8
    # rule (prefer_self 0: the least estimate wins, often a random matching); 1.0: every candidate meets the
    # tolerance, the cheapest (a cross pairing when there are collisions) wins; 0.0: none does, the least estimate
    import photonic_engines.gaussian.hafnian as H
    monkeypatch.setattr(H, "_PAIR_SEARCH", {"cpu": 1})
    monkeypatch.setattr(H, "_ROUND_TOL", rule[0])
    monkeypatch.setattr(H, "_PAIR_PREFER_SELF", rule[1])
    rng = np.random.default_rng(3 + sum(counts))
    m = len(counts)
    B = _rand_sym(m, rng); gam = rng.normal(size=(2, m)) + 1j * rng.normal(size=(2, m)); col = rng.normal(size=(2, m)) + 0j
    A, _ = expand(B, gam[0], counts); lab = np.repeat(np.arange(m), counts)
    M = H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab)
    assert torch.equal(M, H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab))
    for v in range(2):
        ref = lhaf_poly(A, gam[v, lab], col[v, lab])
        assert torch.allclose(M[v], ref, rtol=0, atol=1e4 * EPS * float(ref.abs().max()))
    with pytest.raises(ValueError):
        H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab, pairing="best")


@pytest.mark.parametrize("counts", [[3, 3], [4, 4, 2, 1], [5], [6, 5, 4], [2, 2, 2], [5, 2], [13, 1], [3, 3, 3, 1, 1]])
def test_cross_pairing_sieve_equals_the_plain_one(counts):
    # repeated pairs across modes (`_pairing_cross`, Bulmer et al.'s term count) give the same loop-hafnian polynomial
    # as the plain v1 sieve, with no more terms than the self-pairing; exact counts on shapes worked by hand
    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(17 + sum(counts))
    m = len(counts)
    B = _rand_sym(m, rng) / math.sqrt(m); gam = rng.normal(size=(2, m)) + 1j * rng.normal(size=(2, m))
    col = rng.normal(size=(2, m)) + 0j
    A, _ = expand(B, gam[0], counts); lab = np.repeat(np.arange(m), counts)
    N = lab.size; n = (N + 1) // 2
    M = H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab, pairing="cross")
    for v in range(2):
        ref = lhaf_poly(A, gam[v, lab], col[v, lab])
        # round-off: 2^n terms of O(|ref| kappa); kappa of these small instances stays below 1e4
        assert torch.allclose(M[v], ref, rtol=0, atol=1e6 * EPS * float(ref.abs().max()))
    tc = H._pairing_terms(H._pairing_cross(lab, N)[1], n)
    assert tc <= H._pairing_terms(H._pairing(lab, N)[1], n)
    hand = {(3, 3): 4, (4, 4, 2, 1): 5 * 4, (5,): 3 * 2, (5, 2): 3 * 4, (13, 1): 7 * 2}
    if tuple(counts) in hand:
        assert tc == hand[tuple(counts)]
    # random tie-breaks change the pairs, not the count
    assert H._pairing_terms(H._pairing_cross(lab, N, np.random.default_rng(1))[1], n) == tc


def test_grouped_sieve_redone_when_its_round_off_misses(monkeypatch):
    # "auto" below the search: the grouped pairing is checked on its actual kappa and redone with a random matching
    # when N eps kappa > _ROUND_TOL; the value is the same either way, and the LOG records the redo
    import photonic_engines.gaussian.hafnian as H
    counts = [4, 3, 2, 1, 1]
    rng = np.random.default_rng(5)
    m = len(counts)
    B = _rand_sym(m, rng); gam = rng.normal(size=(1, m)) + 1j * rng.normal(size=(1, m)); col = rng.normal(size=(1, m)) + 0j
    A, _ = expand(B, gam[0], counts); lab = np.repeat(np.arange(m), counts)
    ref = lhaf_poly(A, gam[0, lab], col[0, lab])
    for tol, redo in ((1.0, False), (0.0, True)):
        monkeypatch.setattr(H, "_ROUND_TOL", tol)
        monkeypatch.setattr(H, "LOG", [])
        M = H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab)
        assert H.LOG[-1][5] is redo
        assert torch.allclose(M[0], ref, rtol=0, atol=1e4 * EPS * float(ref.abs().max()))


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("counts", [[2, 0, 1], [4, 1], [3, 3], [5], [4, 4, 2, 1], [2, 2, 2]])
def test_collision_aware_sieve_equals_the_plain_one(monkeypatch, counts, fused):
    import photonic_engines.gaussian.hafnian as H
    from photonic_engines.gaussian.hafnian import lhaf_poly_multi
    monkeypatch.setattr(H, "_FUSED", fused)
    rng = np.random.default_rng(sum(counts))
    m = len(counts)
    B = _rand_sym(m, rng); gam = rng.normal(size=(2, m)) + 1j * rng.normal(size=(2, m)); col = rng.normal(size=(2, m)) + 0j
    A, _ = expand(B, gam[0], counts); lab = np.repeat(np.arange(m), counts)
    M = lhaf_poly_multi(A, gam[:, lab], col[:, lab], labels=lab)
    for v in range(2):
        ref = torch.as_tensor([lhaf_brute(A, gam[v, lab] + z * col[v, lab]) for z in (0.0, 0.5, -0.3j)])
        got = torch.as_tensor([complex(np.polynomial.polynomial.polyval(z, M[v].numpy())) for z in (0.0, 0.5, -0.3j)])
        assert torch.allclose(got, ref, rtol=1e-11, atol=0)


def _triton_multi_vs_v1(monkeypatch, counts, V, seed, pairing="auto"):
    pytest.importorskip("triton")
    import photonic_engines.gaussian.hafnian as H
    monkeypatch.setattr(H, "_TRITON", True)                               # on CPU: the Triton interpreter
    rng = np.random.default_rng(seed)
    m = len(counts)
    B = _rand_sym(m, rng); gam = rng.normal(size=(V, m)) + 1j * rng.normal(size=(V, m)); col = rng.normal(size=(V, m)) + 0j
    A, _ = expand(B, gam[0], counts); lab = np.repeat(np.arange(m), counts)
    M, Ma = H.lhaf_poly_multi(A, gam[:, lab], col[:, lab], device=DEV, labels=lab, return_abs=True, pairing=pairing)
    M, Ma = M.cpu(), Ma.cpu()
    for v in range(V):
        ref, ra = lhaf_poly(A, gam[v, lab], col[v, lab], return_abs=True)
        # both round-offs: the plain sieve's sum |term| and the grouped one's (a cross pairing can be worse conditioned)
        assert torch.allclose(M[v], ref, rtol=0, atol=1e4 * EPS * max(float(ra.max()), float(Ma[v].max())))


@pytest.mark.parametrize("counts,V", [([1, 1, 1], 1), ([2, 1, 1, 1], 2), ([1] * 5, 3)])
def test_triton_matrix_kernel_equals_v1(monkeypatch, counts, V):
    _triton_multi_vs_v1(monkeypatch, counts, V, 11 + len(counts))


@pytest.mark.slow
@pytest.mark.parametrize("counts,V,pairing", [([1] * 8, 2, "auto"), ([2, 1, 3, 1, 1], 3, "auto"), ([4, 1, 1, 1, 1, 1], 1, "auto"),
                                              ([1] * 11, 1, "auto"), ([3, 3, 2, 1], 2, "cross"), ([4, 4, 1], 1, "cross")])
def test_triton_matrix_kernel_equals_v1_larger(monkeypatch, counts, V, pairing):
    # "cross": the batched count vectors of the cross grouping (rows = prefix + free subset) through the kernel
    _triton_multi_vs_v1(monkeypatch, counts, V, 17 + len(counts), pairing)


@pytest.mark.parametrize("q,n,V", [(2, 1, 1), (5, 3, 1), (8, 4, 1), (7, 5, 2), (12, 6, 3)])
def test_triton_powers_paths_equal_numpy_powers(q, n, V):
    """tr C_S^k and x^T C_S^(k-1) y of the matrix kernel against numpy matrix powers, k = 1..n; the tolerance is the
    float64 bound of a k-fold product chain, 4 k Q eps times the same quantity with |C_S|, |x|, |y| (Q = 16 here).
    """
    pytest.importorskip("triton")
    from photonic_engines.gaussian.sieve_triton import powers_paths
    rng = np.random.default_rng(90 + 10 * q + V)
    E = q + 4
    Ce = (rng.normal(size=(E, E)) + 1j * rng.normal(size=(E, E))) / np.sqrt(q); Ce[-1] = 0; Ce[:, -1] = 0
    Ye = rng.normal(size=(E, 2 * V)) + 1j * rng.normal(size=(E, 2 * V)); Xe = rng.normal(size=(E, 2 * V)) + 1j * rng.normal(size=(E, 2 * V))
    Ye[-1] = 0; Xe[-1] = 0                                               # E - 1: the dummy vertex that pads to Q
    idx = np.stack([rng.permutation(E - 1)[:q] for _ in range(3)])
    tr, pa = powers_paths(*(torch.tensor(a, device=DEV) for a in (Ce, Ye, Xe, idx)), n, V)
    assert tr.shape == (3, n + 1) and pa.shape == (3, V, n + 1, 3)
    for s in range(3):
        C = Ce[np.ix_(idx[s], idx[s])]; Y = Ye[idx[s]]; X = Xe[idx[s]]
        for k in range(1, n + 1):
            Pk = np.linalg.matrix_power(C, k); Pa = np.linalg.matrix_power(np.abs(C), k)
            tol = 4 * k * 16 * EPS
            assert abs(tr[s, k].item() - np.trace(Pk)) <= tol * np.trace(Pa)
            Pm = np.linalg.matrix_power(C, k - 1); Pma = np.linalg.matrix_power(np.abs(C), k - 1)
            for v in range(V):
                for j, (xc, yc) in enumerate([(v, v), (v, V + v), (V + v, V + v)]):
                    ref = X[:, xc] @ Pm @ Y[:, yc]; bound = np.abs(X[:, xc]) @ Pma @ np.abs(Y[:, yc])
                    assert abs(pa[s, v, k, j].item() - ref) <= tol * bound


@pytest.mark.parametrize("N", [0, 1, 2, 5, 8, 13])
def test_small_numpy_sieve_equals_v1(N):
    from photonic_engines.gaussian.hafnian import lhaf_poly_small
    rng = np.random.default_rng(70 + N)
    A = _rand_sym(N, rng); D0 = rng.normal(size=(3, N)) + 1j * rng.normal(size=(3, N)); D1 = rng.normal(size=(3, N)) + 0j
    q, qa = lhaf_poly_small(A, D0, D1, return_abs=True)
    assert q.shape == (3, N + 1)
    for v in range(3):
        ref, ra = lhaf_poly(A, D0[v], D1[v], return_abs=True)
        assert np.allclose(q[v], ref.numpy(), rtol=0, atol=1e4 * EPS * float(ref.abs().max()))
        assert np.allclose(qa[v], ra.numpy(), rtol=1e-12, atol=0)          # same pairing: same |terms|


def test_v2_sampler_draws_the_v1_samples(monkeypatch):
    import photonic_engines.gaussian.sampler as smp
    monkeypatch.setattr(smp, "V2_FROM", 1)                          # exercise the batched sieve at small N
    st = GaussianInput.smsv(5, 9, 0.7)
    U = haar_unitary(9, 23)
    a = sample(st, U, 150, seed=12, impl="v1")
    for w in (1, 4, 9):                                              # (every sieve here is under SMALL_BELOW: numpy)
        assert np.array_equal(a, sample(st, U, 150, seed=12, impl="v2", window=w))



@pytest.mark.parametrize("impl,schmidt,loss", [("v1", None, None), ("v2", None, None), ("v2", [0.7, 0.3], 0.8)])
def test_the_photon_cap_censors_exactly_the_shots_above_it(monkeypatch, impl, schmidt, loss):
    # a shot whose final count is <= cap must come out identical to the uncapped draw (same seed), and a
    # shot is censored iff its uncapped count exceeds the cap: what makes the cap exact for stratified_cost
    import photonic_engines.gaussian.sampler as smp
    monkeypatch.setattr(smp, "V2_FROM", 1)
    st = GaussianInput.smsv(4, 8, 0.8)
    U = haar_unitary(8, 5)
    cap, seen = 3, set()
    for seed in range(60):
        free = sample(st, U, 1, seed=seed, impl=impl, schmidt=schmidt, loss=loss)[0]
        got, stats = sample(st, U, 1, seed=seed, impl=impl, schmidt=schmidt, loss=loss, max_photons=cap,
                            return_stats=True)
        over = int(free.sum()) > cap
        seen.add(over)
        assert stats.censored == [over]
        if over:
            assert (got[0] == -1).all() and cap < stats.photons[0] <= free.sum()
        else:
            assert np.array_equal(got[0], free)
    assert seen == {True, False}                                        # both branches exercised

# ---- the frontier statistic (gaussian/cost.py) ---------------------------------------------------------
def test_detected_law_is_the_thinned_convolution():
    from photonic_engines.gaussian.cost import detected_law
    st = GaussianInput(np.array([0.4, 0.66, 0.9]), (0, 1, 2), 4)
    for eta in (None, 0.7):
        law = detected_law(st, 300, loss=eta)
        assert abs(law.sum() - 1) < 1e-12
        assert abs(np.dot(np.arange(301), law) - (eta or 1.0) * st.mean_photons()) < 1e-10
    assert np.allclose(detected_law(st, 40), st.photon_number_probs(40), atol=1e-15)
    lam = (0.7, 0.3)
    lawk = detected_law(st, 300, schmidt=lam)
    assert abs(np.dot(np.arange(301), lawk) - sum(np.sinh(r * np.sqrt(lk)) ** 2 for r in st.r for lk in lam)) < 1e-10


def test_binomial_law_is_the_thinned_fock_law():
    """`binomial_law(n, p)` is Binomial(n, p): matches math.comb term by term, sums to one, has mean n p;
    p = 1 is the point mass at n; a cutoff below n truncates, above n pads with zeros; and it is what
    `run_cutoff` / `stratified_cost` accept — the S = 1e4 cut of Bin(64, 0.5) sits where P(K > k) <= 1e-4."""
    from photonic_engines.gaussian.cost import binomial_law, run_cutoff, stratified_cost
    for n, p in ((7, 0.3), (24, 0.9), (64, 0.5)):
        law = binomial_law(n, p)
        ref = np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])
        assert law.shape == (n + 1,) and np.allclose(law, ref, atol=1e-15, rtol=1e-12)
        assert abs(law.sum() - 1) < 1e-12 and abs(np.dot(np.arange(n + 1), law) - n * p) < 1e-10
    one = binomial_law(10, 1.0)
    assert one[10] == 1.0 and one[:10].sum() == 0.0
    assert binomial_law(10, 0.5, cutoff=4).shape == (5,) and binomial_law(10, 0.5, cutoff=12)[11:].sum() == 0.0
    law = binomial_law(64, 0.5)
    kS = run_cutoff(law, 1e4)
    assert law[kS + 1:].sum() <= 1e-4 < law[kS:].sum()
    # a k-photon instance costing 2^k: t_S from the law equals the direct truncated sum
    ks, secs = np.repeat(np.arange(65), 5), np.repeat(2.0 ** np.arange(65), 5)
    est = stratified_cost(ks, secs, law, max_N=kS)
    assert abs(est.seconds - float((law[: kS + 1] * 2.0 ** np.arange(kS + 1)).sum())) < 1e-9 * est.seconds


def test_the_tilt_leaves_the_conditional_law_given_N_unchanged():
    from itertools import product as iproduct

    from photonic_engines.gaussian.cost import tilted
    st = GaussianInput(np.array([0.4, 0.7]), (0, 1), 3)
    st2, s = tilted(st, 1.3)
    assert s == pytest.approx(1.3) and np.allclose(np.tanh(st2.r), 1.3 * np.tanh(st.r))
    U = haar_unitary(3, 4)
    pats = [p for p in iproduct(range(7), repeat=3) if sum(p) <= 6]
    for N in (2, 4, 6):
        ps = [p for p in pats if sum(p) == N]
        a = np.array([pure_probability(st, U, p) for p in ps]); b = np.array([pure_probability(st2, U, p) for p in ps])
        assert np.abs(a / a.sum() - b / b.sum()).max() < 1e-13


def test_defensive_importance_sampling_is_unbiased_on_a_known_cost():
    from photonic_engines.gaussian.cost import detected_law, mixture_cost, tilted
    st = GaussianInput.smsv(12, 24, GaussianInput.r_for_mean(0.5))
    st2, s = tilted(st, np.sqrt(2))
    K = 400
    lt, l2 = detected_law(st, K), detected_law(st2, K)
    cost = 1e-3 * 2.0 ** (np.arange(K + 1) / 2)
    true = float((lt * cost).sum())
    rng = np.random.default_rng(3)
    N = np.concatenate([rng.choice(K + 1, 300, p=lt / lt.sum()), rng.choice(K + 1, 300, p=l2 / l2.sum())])
    ph = np.repeat([0, 1], 300)
    e = mixture_cost(N, cost[N], ph, lt, [lt, l2])
    assert abs(e.seconds - true) < 5 * e.stderr and e.stderr < 0.1 * true
    plain = mixture_cost(N[:300], cost[N[:300]], ph[:300], lt, [lt])
    assert plain.seconds == pytest.approx(cost[N[:300]].mean())


def test_stratified_cost_is_exact_when_every_stratum_is_measured():
    from photonic_engines.gaussian.cost import stratified_cost
    law = np.array([0.5, 0.0, 0.3, 0.0, 0.15, 0.0, 0.05])
    N = np.repeat([0, 2, 4, 6], 6); t = 0.01 * 2.0 ** (N / 2)
    e = stratified_cost(N, t, law)
    assert e.seconds == pytest.approx(float((law * 0.01 * 2.0 ** (np.arange(7) / 2)).sum()))
    assert e.extrapolated == 0.0


@pytest.mark.parametrize("impl,window,cap", [("v2", 4, None), ("v2", 1, None), ("v2", 3, 7), ("v1", 4, None)])
def test_timing_plan_replays_the_sampler_calls(impl, window, cap):
    """`timing.plan_calls(pattern)` gives the sieve calls the sampler recorded for that shot, in order, for every
    shot (windows, the small / plain / multi thresholds, and the cap: a capped shot replays the uncapped pattern
    of the same seed up to where it stopped)."""
    import photonic_engines.gaussian.sampler as smp
    from photonic_engines.gaussian.timing import events_to_calls, plan_calls
    st = GaussianInput.smsv(8, 16, GaussianInput.r_for_mean(1.5))
    U = haar_unitary(16, seed=5)
    old = smp.PROFILE
    kinds = set()
    try:
        for seed in range(8):
            pat = sample(st, U, 1, seed=seed, impl=impl, window=window)[0]
            smp.PROFILE = smp.Profile(events=[])
            sample(st, U, 1, seed=seed, impl=impl, window=window, max_photons=cap)
            got = events_to_calls(smp.PROFILE.events)
            smp.PROFILE = None
            want = [(c.N, c.V, c.kind, c.device) for c in plan_calls(pat, window=window, impl=impl, max_photons=cap,
                                                                     device=None)]
            assert got == want, (seed, pat.tolist())
            kinds |= {(k, v > 1) for _, v, k, _ in got}
    finally:
        smp.PROFILE = old
    if cap is None:                                          # every kind of call was exercised (the cap stops below 14)
        assert {"none", "small" if impl == "v2" else "plain", "multi" if impl == "v2" else "plain"} <= {k for k, _ in kinds}


def test_timing_surrogates_and_self_terms():
    """Surrogate patterns have total N on m modes ("free": N distinct modes); self_terms counts the collision
    discount of `hafnian._pairing` (a mode of t photons: floor(t/2) self-pairs, grouped from 2 of them)."""
    from photonic_engines.gaussian.hafnian import _pairing
    from photonic_engines.gaussian.timing import self_terms, surrogate_patterns
    rng = np.random.default_rng(3)
    for kind in ("free", "be"):
        P = surrogate_patterns(9, 12, 50, kind, rng)
        assert P.shape == (50, 12) and (P.sum(1) == 9).all() and (P >= 0).all()
        if kind == "free":
            assert (P.max(1) == 1).all()
    for counts in ([1, 1, 1], [2, 1, 1], [4, 1], [5, 3, 1], [6, 4, 2]):
        N = sum(counts)
        _, groups = _pairing(np.repeat(np.arange(len(counts)), counts), N)
        n = (N + 1) // 2
        want = math.prod(r + 1 for _, r in groups) * 2 ** (n - sum(r for _, r in groups))
        assert self_terms(N, [t for t in counts if t >= 2]) == want


def test_timing_run_weights_by_enumeration():
    """t_S = sum_a w[a] c(a) equals E[sum_k c(N_k) 1{N_1 + N_2 <= N_S}] by direct enumeration of two instances, and
    reduces to `run_cost` for one instance."""
    from photonic_engines.gaussian.cost import run_cutoff
    from photonic_engines.gaussian.timing import run_cost, run_weights
    p1 = np.array([0.5, 0.0, 0.3, 0.0, 0.15, 0.0, 0.05]); p2 = np.array([0.7, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0])
    c = lambda a: 1.0 + a ** 2
    w, NS = run_weights([p1, p2], 20.0)
    tot = np.convolve(p1, p2)[:7]
    assert run_cutoff(tot, 20.0) == NS
    want = sum(p1[a] * p2[b] * (c(a) + c(b)) for a in range(7) for b in range(7) if a + b <= NS)
    assert abs(sum(w[a] * c(a) for a in range(NS + 1)) - want) <= 1e-14 * want
    w1, NS1 = run_weights([p1], 20.0)
    assert abs(sum(w1[a] * c(a) for a in range(NS1 + 1)) - run_cost(p1, c, 20.0)[0]) <= 1e-14


def test_timing_grouped_call_price():
    """A collided call below the pairing search is priced per prefix length L as one free sieve of (count vectors with
    |s| = L) x 2^n_f terms (`lhaf_poly_multi` batches the count vectors of one L since 2026-09-27); on this exactly
    exponential table that equals prod(r + 1) free sieves of 2 n_f vertices (up to the log-linear interpolation's
    rounding). A free call is taken from the table as it is."""
    from photonic_engines.gaussian.timing import Call, TimingModel
    multi = {(N, 1): 1e-3 * 2 ** max(0, (N - 14) // 2) for N in range(14, 41)}
    M = TimingModel(multi=multi, small={}, plain={}, draw={0: 0.0}, prep={0: 0.0}, rate={(n, 1): 1e-6 for n in (10, 20)},
                    multi_max=40, pairing="self")                      # v2.8's self-pairing; "cross" below
    assert M.t_call(Call(30, 1, "multi", "cuda", 1, ())) == multi[(30, 1)]
    # counts 4 and 6 -> r = 2, 3 identical pairs: (2 + 1)(3 + 1) free sieves of n_f = 15 - 5 = 10 pairs (20 vertices)
    assert M.t_call(Call(30, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(12 * multi[(20, 1)], rel=1e-12)
    # a count of 2 or 3 is one self-pair, not grouped: priced as free
    assert M.t_call(Call(30, 1, "multi", "cuda", 1, (2, 3))) == multi[(30, 1)]
    # at or above the pairing search (n = 20 >= 18) v2.8 drops the grouping; "self_all" keeps it at every size
    assert M.t_call(Call(40, 1, "multi", "cuda", 1, (4, 6))) == multi[(40, 1)]
    M.discount = "self_all"
    assert M.t_call(Call(40, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(12 * multi[(30, 1)], rel=1e-12)
    # the ungrouped sub-sieve past the table is the extrapolated free call (term count x rate), not the last entry:
    # N = 80, n_f = 40 - 5 = 35 pairs, 2^(35 - 20) x t(40) at a flat rate
    assert M.t_call(Call(80, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(12 * 2 ** 15 * multi[(40, 1)], rel=1e-14)
    # the cross pairing (default since 2026-09-27): counts 4 and 6 in N = 30 pair the two modes 4 times (one group of 4,
    # 5 terms) and leave 22 distinct vertices = 11 free pairs: 5 free sieves of 22 vertices, against 12 of 20 self-paired
    M.pairing = "cross"; M.discount = "self"
    from photonic_engines.gaussian.timing import group_sizes
    assert group_sizes(30, (4, 6)) == [4] and group_sizes(30, (4, 6), "self") == [2, 3]
    assert M.t_call(Call(30, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(5 * multi[(22, 1)], rel=1e-12)
    # counts 2 and 3: no self group (floor(t/2) < 2), but 2 cross pairs: 3 free sieves of 26 vertices
    assert M.t_call(Call(30, 1, "multi", "cuda", 1, (2, 3))) == pytest.approx(3 * multi[(26, 1)], rel=1e-12)


def test_timing_precision_tiers():
    """`TimingModel.tiers`: fp64 while eps kappa(N) <= tol (N <= 62 on the H100 fit), dd to N = 141, then qd priced as
    dd; a dd call costs dd_factor x its fp64 time (measured dd rate / fp64 rate to the last dd n, the fixed ratio
    beyond); the cross grouping only where kappa + cross_penalty decades fits the arithmetic; the cheapest admissible
    option wins."""
    from photonic_engines.gaussian.timing import Call, TimingModel
    multi = {(N, 1): 1e-3 * 2 ** max(0, (N - 14) // 2) for N in range(14, 41)}
    M = TimingModel(multi=multi, small={}, plain={}, draw={0: 0.0}, prep={0: 0.0}, rate={(n, 1): 1e-6 for n in (10, 20)},
                    multi_max=40, pairing="cross", discount="self_all", tiers=True,
                    dd_rate={(n, 1): 1e-4 for n in (10, 20)}, dd_ratio_beyond={1: 50.0})
    assert [M.tier(N) for N in (40, 62, 63, 141, 142)] == ["fp64", "fp64", "dd", "dd", "qd"]
    assert M.dd_factor(15, 1) == pytest.approx(100.0) and M.dd_factor(30, 1) == 50.0
    # N = 30, counts 4 and 6: kappa + 5 decades fits fp64 (0.40 + 5.91 + 5 - 15.65 <= -3): the untiered price
    grouped = M.t_call(Call(30, 1, "multi", "cuda", 1, (4, 6)))
    M.tiers = False
    assert grouped == pytest.approx(M.t_call(Call(30, 1, "multi", "cuda", 1, (4, 6))), rel=1e-12)
    M.tiers = True
    # N = 50: the grouping no longer fits fp64 (15.25 decades); fp64 ungrouped (t_multi) vs dd grouped (x50 of 5
    # sieves of 42 vertices): here fp64 ungrouped is cheaper
    t50 = M.t_call(Call(50, 1, "multi", "cuda", 1, (4, 6)))
    assert t50 == pytest.approx(min(M.t_multi(50, 1), 50 * 5 * M.t_multi(42, 1)), rel=1e-12)
    # N = 80: dd only; grouped if 0.40 + 15.76 + 5 - 31.31 <= -3 (yes): 50 x the cheaper of free and grouped
    t80 = M.t_call(Call(80, 1, "multi", "cuda", 1, (4, 6)))
    assert t80 == pytest.approx(50 * min(M.t_multi(80, 1), M._grouped([4], 40, 1)), rel=1e-12)
    # a free call past fp64 is dd x its fp64 time
    assert M.t_call(Call(70, 1, "multi", "cuda", 1, ())) == pytest.approx(50 * M.t_multi(70, 1), rel=1e-12)
    assert M.last_arith == "dd"
    # N = 130: the grouping needs qd (0.40 + 25.61 + 5 - 31.31 > -3), priced at qd_factor x dd: 1 (lower bound) or 10
    t130 = M.t_call(Call(130, 1, "multi", "cuda", 1, (4, 6)))
    assert t130 == pytest.approx(50 * min(M.t_multi(130, 1), M._grouped([4], 65, 1)), rel=1e-12)
    M.qd_factor = 10.0
    assert M.t_call(Call(130, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(
        50 * min(M.t_multi(130, 1), 10 * M._grouped([4], 65, 1)), rel=1e-12)
    # arith_max "dd" (no qd kernel): N = 130 ungrouped in dd; N = 150 fits nothing, priced ungrouped in dd (">dd")
    M.arith_max = "dd"
    assert M.t_call(Call(130, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(50 * M.t_multi(130, 1), rel=1e-12)
    assert M.last_arith == "dd"
    assert M.t_call(Call(150, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(50 * M.t_multi(150, 1), rel=1e-12)
    assert M.last_arith == ">dd"
    # tier_choice "needed": the lowest admissible arithmetic only; at N = 50 fp64 ungrouped, even if dd grouped is cheaper
    M.arith_max = "qd"; M.tier_choice = "needed"
    M.dd_ratio_beyond = {1: 1e-6}                                    # make dd grouped the cheapest option
    assert M.t_call(Call(50, 1, "multi", "cuda", 1, (4, 6))) == pytest.approx(M.t_multi(50, 1), rel=1e-12)
    M.tier_choice = "cost"
    assert M.t_call(Call(50, 1, "multi", "cuda", 1, (4, 6))) < M.t_multi(50, 1) and M.last_arith == "dd"


# ---- double-double (v2.11) ------------------------------------------------------------------------
def test_dd_arithmetic_against_exact_rationals():
    # gaussian.dd element-wise ops, pairwise sum and Ozaki product against Python fractions: each within a few
    # eps_dd = 2^-104 of the exact value (relative to the sum of |terms| for the sums and products)
    from fractions import Fraction as F

    from photonic_engines.gaussian import dd
    rng = np.random.default_rng(0)

    def rdd(shape):
        h = torch.tensor(rng.normal(size=shape)); lo = torch.tensor(rng.normal(size=shape)) * h.abs() * 2 ** -54
        hr, lr = dd.quick_two_sum(h, lo)
        h2 = torch.tensor(rng.normal(size=shape)); l2 = torch.tensor(rng.normal(size=shape)) * h2.abs() * 2 ** -54
        hi, li = dd.quick_two_sum(h2, l2)
        return torch.complex(hr, hi), torch.complex(lr, li)

    def cf(h, lo):
        return F(float(h.real)) + F(float(lo.real)), F(float(h.imag)) + F(float(lo.imag))
    eps = F(2) ** -104
    a = rdd((40,)); b = rdd((40,))
    m = dd.cmul(*a, *b); s = dd.add(*a, *b); q = dd.div_scalar(*a, 7.0)
    for i in range(40):
        x, y = cf(a[0][i], a[1][i]), cf(b[0][i], b[1][i])
        z = (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]); g = cf(m[0][i], m[1][i])
        scale = abs(x[0] * y[0]) + abs(x[1] * y[1]) + abs(x[0] * y[1]) + abs(x[1] * y[0])
        assert abs(g[0] - z[0]) + abs(g[1] - z[1]) <= 4 * eps * scale
        g = cf(s[0][i], s[1][i])
        assert abs(g[0] - x[0] - y[0]) + abs(g[1] - x[1] - y[1]) <= 4 * eps * (abs(x[0]) + abs(y[0]) + abs(x[1]) + abs(y[1]))
        g = cf(q[0][i], q[1][i])
        assert abs(g[0] - x[0] / 7) + abs(g[1] - x[1] / 7) <= 4 * eps * (abs(x[0]) + abs(x[1])) / 7
    h, lo = rdd((301,)); S = dd.dsum(h, lo, 0); g = cf(S[0], S[1])
    ex = [sum(cf(h[i], lo[i])[t] for i in range(301)) for t in (0, 1)]
    tot = sum(abs(cf(h[i], lo[i])[0]) + abs(cf(h[i], lo[i])[1]) for i in range(301))
    assert abs(g[0] - ex[0]) + abs(g[1] - ex[1]) <= (2 + 9) * eps * tot            # 2 + log2(301) levels
    A = rdd((3, 13)); B = rdd((13, 2)); C = dd.cmatmul(*A, *B)
    for i in range(3):
        for j in range(2):
            z = [F(0), F(0)]; scale = F(0)
            for k in range(13):
                x, y = cf(A[0][i, k], A[1][i, k]), cf(B[0][k, j], B[1][k, j])
                z[0] += x[0] * y[0] - x[1] * y[1]; z[1] += x[0] * y[1] + x[1] * y[0]
                scale += abs(x[0] * y[0]) + abs(x[1] * y[1]) + abs(x[0] * y[1]) + abs(x[1] * y[0])
            g = cf(C[0][i, j], C[1][i, j])
            assert abs(g[0] - z[0]) + abs(g[1] - z[1]) <= 8 * eps * scale


def _lhaf_poly_mp(A, D0, D1):
    from functools import lru_cache

    import mpmath
    N = A.shape[0]
    A = [[mpmath.mpc(complex(A[i, j])) for j in range(N)] for i in range(N)]
    D0 = [mpmath.mpc(complex(x)) for x in D0]; D1 = [mpmath.mpc(complex(x)) for x in D1]

    @lru_cache(None)
    def rec(rem):
        if not rem:
            return (mpmath.mpc(1),)
        i, rest = rem[0], rem[1:]
        out = [mpmath.mpc(0)] * (len(rem) + 1)
        for k, c in enumerate(rec(rest)):                               # the loop on i: D0 + z D1
            out[k] += D0[i] * c; out[k + 1] += D1[i] * c
        for t, j in enumerate(rest):
            for k, c in enumerate(rec(rest[:t] + rest[t + 1:])):
                out[k] += A[i][j] * c
        return tuple(out)
    return rec(tuple(range(N)))


@pytest.mark.parametrize("counts", [[1] * 9, [3, 2, 2, 1], [4, 4, 1]])
def test_dd_sieve_against_50_digits(counts):
    # precision="dd": the loop-hafnian polynomial against a brute-force sum over matchings in 50-digit arithmetic;
    # every pairing gives the value to the complex128 rounding of the result (2^-53 of max |c|, x2 for the complex
    # modulus), where fp64 carries eps kappa
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(2 + sum(counts))
    m = len(counts)
    X = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); B = (X + X.T) / (2 * math.sqrt(m))
    g = rng.normal(size=m) + 1j * rng.normal(size=m); col = rng.normal(size=m) + 1j * rng.normal(size=m)
    A, _ = expand(B, g, counts); lab = np.repeat(np.arange(m), counts)
    ref = _lhaf_poly_mp(A, g[lab], col[lab]); scale = max(abs(x) for x in ref)
    for pairing in ("auto", "cross", "self"):
        v = H.lhaf_poly_multi(A, g[lab][None], col[lab][None], labels=lab, pairing=pairing, precision="dd")[0].numpy()
        err = max(abs(mpmath.mpc(complex(v[k])) - ref[k]) for k in range(len(v))) / scale
        assert float(err) <= 2 * 2.0 ** -53


def test_dd_sieve_resolves_an_ill_conditioned_pairing(monkeypatch):
    # Borealis-like squeezing (r = 1.1) with heavy collisions: the self-pairing has kappa ~1e5, so fp64 is off by
    # ~eps kappa; in dd the self and cross pairings agree to the result's rounding. "auto" with a tolerance fp64 cannot
    # meet (1e-14) escalates to dd, and the LOG records it
    from helpers import haar_unitary_np

    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(5)
    st = GaussianInput.smsv(12, 12, 1.1)
    B, _, _ = st.bargmann(haar_unitary_np(12, rng))
    counts = [6, 5, 4, 3]
    Bs = B[:4, :4]; g = rng.normal(size=4) + 1j * rng.normal(size=4); col = Bs[:, 0]
    A, _ = expand(Bs, g, counts); lab = np.repeat(np.arange(4), counts)
    D0, D1 = g[lab][None], col[lab][None]
    f_self, ab = H.lhaf_poly_multi(A, D0, D1, labels=lab, pairing="self", return_abs=True)
    d_self = H.lhaf_poly_multi(A, D0, D1, labels=lab, pairing="self", precision="dd")
    d_cross = H.lhaf_poly_multi(A, D0, D1, labels=lab, pairing="cross", precision="dd")
    scale = float(d_cross.abs().max())
    kappa = float(ab.max()) / scale
    assert kappa > 1e4                                                  # really ill-conditioned for fp64
    assert float((d_self - d_cross).abs().max()) <= 4 * EPS * scale
    assert float((f_self - d_cross).abs().max()) > 100 * EPS * scale    # the fp64 error is real, and dd removes it
    monkeypatch.setattr(H, "_ROUND_TOL", 1e-14)
    monkeypatch.setattr(H, "LOG", [])
    a = H.lhaf_poly_multi(A, D0, D1, labels=lab, precision="auto")
    assert H.LOG[-1][6] == "dd" and H.LOG[-1][5] is True
    assert float((a - d_cross).abs().max()) <= 4 * EPS * scale
    with pytest.raises(ValueError):
        H.lhaf_poly_multi(A, D0, D1, labels=lab, precision="qd")


def test_sampler_in_dd_draws_the_same_samples(monkeypatch):
    # precision="dd" changes only the rounding of the conditional probabilities: on a well-conditioned instance the
    # inverse-CDF draws land on the same counts for the same seed (and the law is the exact one either way). The v2
    # sieve is forced from 4 vertices (sampler.V2_FROM / SMALL_BELOW) so that the chain runs it; the LOG shows dd
    import photonic_engines.gaussian.hafnian as H
    import photonic_engines.gaussian.sampler as smp
    monkeypatch.setattr(smp, "V2_FROM", 4); monkeypatch.setattr(smp, "SMALL_BELOW", 4)
    st = GaussianInput.smsv(6, 6, 1.0)
    U = haar_unitary(6, 71)
    a = sample(st, U, 3, seed=4, max_photons=18)
    monkeypatch.setattr(H, "LOG", [])
    b = sample(st, U, 3, seed=4, max_photons=18, precision="dd")
    assert np.array_equal(a, b)
    assert H.LOG and all(e[6] == "dd" for e in H.LOG)


@pytest.mark.slow
@pytest.mark.parametrize("counts", [[1] * 7, [3, 2, 1, 1], [4, 4, 1]])
def test_dd_series_triton_kernel_equals_the_torch_recursion(monkeypatch, counts):
    # the dd series in its Triton kernel (interpreter: Dekker split) against the torch dd recursion, and both against
    # a 50-digit brute force: the same value to the result's complex128 rounding
    pytest.importorskip("triton")
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(31 + sum(counts))
    m = len(counts)
    X = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); B = (X + X.T) / (2 * math.sqrt(m))
    g = rng.normal(size=(2, m)) + 1j * rng.normal(size=(2, m)); col = rng.normal(size=(2, m)) + 1j * rng.normal(size=(2, m))
    A, _ = expand(B, g[0], counts); lab = np.repeat(np.arange(m), counts)
    out = {}
    for tri in (False, True):
        monkeypatch.setattr(H, "_DD_TRITON", tri)
        out[tri] = H.lhaf_poly_multi(A, g[:, lab], col[:, lab], labels=lab, precision="dd")
    for v in range(2):
        ref = _lhaf_poly_mp(A, g[v, lab], col[v, lab]); scale = max(abs(x) for x in ref)
        for tri in (False, True):
            err = max(abs(mpmath.mpc(complex(out[tri][v, k])) - ref[k]) for k in range(len(ref))) / scale
            assert float(err) <= 2 * 2.0 ** -53


def test_dd_series_triton_kernel_small(monkeypatch):
    # the fast-tier counterpart of the slow test above: one small instance through the dd series kernel (interpreter)
    # against the torch dd recursion, to the result's complex128 rounding
    pytest.importorskip("triton")
    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(8)
    counts = [2, 1, 1]
    X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); B = (X + X.T) / 2
    g = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3)); col = rng.normal(size=(1, 3)) + 0j
    A, _ = expand(B, g[0], counts); lab = np.repeat(np.arange(3), counts)
    out = {}
    for tri in (False, True):
        monkeypatch.setattr(H, "_DD_TRITON", tri)
        out[tri] = H.lhaf_poly_multi(A, g[:, lab], col[:, lab], device=DEV, labels=lab, precision="dd")
    assert float((out[True] - out[False]).abs().max()) <= 4 * EPS * float(out[False].abs().max())


@pytest.mark.parametrize("shape", [(3, 5, 7), (2, 4, 6, 6), (1, 1, 1)])
def test_dd_fused_slices_are_bit_identical(shape):
    # the fused Ozaki slicing kernel (interpreter) against dd.slices of the embedded block forms, left and right,
    # with a zero row / column and entries over ~10 decades: the same float64 slices bit for bit
    pytest.importorskip("triton")
    import torch

    from photonic_engines.gaussian import dd
    from photonic_engines.gaussian.sieve_dd_triton import slices_complex
    g = torch.Generator().manual_seed(sum(shape))

    def rn():
        return torch.randn(*shape, generator=g, dtype=torch.float64)
    hi = torch.complex(rn(), rn()) * torch.exp(3 * rn()); lo = hi * 1e-17 * rn()
    hi, lo = hi.to(DEV), lo.to(DEV)
    hi[..., 0, :] = 0; lo[..., 0, :] = 0
    for left in (True, False):
        beta, s = dd.plan(2 * shape[-1] if left else 2 * shape[-2])
        ref = dd.slices(*(dd.embed_left(hi, lo) if left else dd.embed_right(hi, lo)), beta, s, -1 if left else -2)
        got = slices_complex(hi, lo, beta, s, left)
        assert len(got) == s and all(torch.equal(a, b) for a, b in zip(ref, got))


def test_dd_sieve_with_fused_slices_is_unchanged(monkeypatch):
    # the dd sieve with the fused slicing (forced on, interpreter) returns exactly the unfused result
    pytest.importorskip("triton")
    import photonic_engines.gaussian.dd as D
    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(12)
    counts = [2, 2, 1, 1]
    X = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); B = (X + X.T) / 2
    g = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4)); col = rng.normal(size=(2, 4)) + 0j
    A, _ = expand(B, g[0], counts); lab = np.repeat(np.arange(4), counts)
    monkeypatch.setattr(H, "_DD_TRITON", False)
    out = {}
    for fu in (False, True):
        monkeypatch.setattr(D, "_FUSED_SLICES", fu)
        out[fu] = H.lhaf_poly_multi(A, g[:, lab], col[:, lab], device=DEV, labels=lab, precision="dd").cpu()
    assert np.array_equal(np.asarray(out[True]), np.asarray(out[False]))


def test_triton_matrix_kernel_not_chosen_past_64_vertices(monkeypatch):
    # the Triton matrix kernel holds subset matrices of <= 64 vertices: on CUDA the automatic choice stops at n = 32 pairs
    # (a CUDA sieve with n > 32 raised in powers_paths before 2026-09-29); forcing it still forces it
    pytest.importorskip("triton")
    import torch

    import photonic_engines.gaussian.hafnian as H
    monkeypatch.setattr(H, "_TRITON", None)
    cuda = torch.device("cuda")
    assert H._use_triton(cuda, 32) and not H._use_triton(cuda, 33) and not H._use_triton(cuda, 48)
    assert not H._use_triton(torch.device("cpu"), 20)
    monkeypatch.setattr(H, "_TRITON", True)
    assert H._use_triton(cuda, 48)


def test_dd_batches_split_on_out_of_memory(monkeypatch):
    # _terms_dd halves a batch that raises torch.OutOfMemoryError (simulated here above 5 subsets), remembers the cap,
    # and returns the unsplit result to the complex128 rounding of the dd sum
    import torch

    import photonic_engines.gaussian.hafnian as H
    rng = np.random.default_rng(21)
    counts = [2, 1, 1, 1, 1, 1]
    X = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)); B = (X + X.T) / 2
    g = rng.normal(size=(1, 6)) + 1j * rng.normal(size=(1, 6)); col = rng.normal(size=(1, 6)) + 0j
    A, _ = expand(B, g[0], counts); lab = np.repeat(np.arange(6), counts)
    monkeypatch.setattr(H, "_DD_TRITON", False)
    ref = H.lhaf_poly_multi(A, g[:, lab], col[:, lab], labels=lab, precision="dd")
    real, sizes = H._terms_dd_batch, []

    def tight(Ce, Ye, XYe, idx, *a):
        if idx.shape[0] > 5:
            raise torch.OutOfMemoryError("simulated")
        sizes.append(idx.shape[0])
        return real(Ce, Ye, XYe, idx, *a)
    monkeypatch.setattr(H, "_terms_dd_batch", tight)
    monkeypatch.setattr(H, "_DD_CAP", {})
    out = H.lhaf_poly_multi(A, g[:, lab], col[:, lab], labels=lab, precision="dd")
    assert sizes and max(sizes) <= 5 and H._DD_CAP
    assert float((torch.as_tensor(out) - torch.as_tensor(ref)).abs().max()) <= 4 * EPS * float(torch.as_tensor(ref).abs().max())


def test_timing_tiers_grouped_price_floor():
    # with tiers, a cross-grouped call costs at least (its term count) x (the per-term rate at the full n) x the
    # arithmetic's factor: a grouped term is a term of the full n (job 31: _grouped alone priced dd-grouped calls x5-10 low)
    import math

    from photonic_engines.gaussian.timing import Call, TimingModel, group_sizes
    multi = {(N, 1): 1e-9 * 2 ** max(0, (N - 14) // 2) for N in range(14, 41)}   # whole calls far cheaper than
    rate = {(10, 1): 1e-6, (20, 1): 1e-6}                                          # terms x rate: the floor binds
    M = TimingModel(multi=multi, small={}, plain={}, draw={0: 0.0}, prep={0: 0.0}, rate=rate, multi_max=40,
                    pairing="cross", tiers=True, dd_rate={(n, 1): 1e-4 for n in (10, 20)}, dd_ratio_beyond={1: 50.0})
    c = Call(80, 1, "multi", "cuda", 1, (4, 6))
    rs = group_sizes(80, (4, 6), "cross", "cuda")
    terms = math.prod(r + 1 for r in rs) * 2.0 ** (40 - sum(rs))
    assert M.t_call(c) == pytest.approx(min(50 * M.t_multi(80, 1), 50 * max(M._grouped(rs, 40, 1), terms * 1e-6)),
                                        rel=1e-12)
    assert terms * 1e-6 > M._grouped(rs, 40, 1)                              # the floor is the binding term here


def test_gaussian_slos_basis_backend_past_63_bit_keys(monkeypatch):
    # GaussianSLOSKernel asks for the numpy basis builder when the torch builder's packed int64 keys would overflow
    # (N_cut x ceil(log2 m) > 63 bits: N_cut = 16 at m = 10 failed on CUDA in job 8, 2026-10-01), "auto" otherwise
    import photonic_engines.slos.gaussian as G
    from photonic_engines import GaussianInput
    seen = []

    class Stop(Exception):
        pass

    def fake(n, m, mask, device="cpu", backend="auto"):
        seen.append((n, m, device, backend)); raise Stop
    monkeypatch.setattr(G, "get_layers", fake)
    st = GaussianInput.smsv(5, 10, 0.6)
    for cut in (16, 15):
        with pytest.raises(Stop):
            G.GaussianSLOSKernel(st, cut, device="cpu")
    assert seen == [(16, 10, "cpu", "numpy"), (15, 10, "cpu", "auto")]


def test_gbs_loop_number_law_and_cap_sampler():
    # photonic_engines/gaussian/loop.py (GBS-L): the Gaussian photon-number law is the SMSV law; the capped
    # SuperpositionLoopKernel's probabilities are the loop-hafnian probabilities of the unrolled circuit (loops flushed,
    # so nothing is left in a slot) up to the dropped mass; the leak model brackets the sampler's own weights
    from photonic_engines.gaussian.loop import cap, gbs_loop_kernel, number_law, smsv_amps
    from photonic_engines.timeloop import TimeLoopCircuit
    r = math.asinh(math.sqrt(0.5))
    sh, ch = math.sinh(r), math.cosh(r)
    P = number_law(np.array([[sh * sh]], complex), np.array([[-sh * ch]], complex), 20)
    ref_p = np.asarray(GaussianInput.smsv(1, 1, r).photon_number_probs(20)).reshape(-1)[:21]
    assert np.allclose(P, ref_p, atol=1e-14) and np.allclose(P, np.abs(smsv_amps(r, 21)) ** 2, atol=1e-14)
    L, n_src, T, p = 2, 2, 6, 16
    c = TimeLoopCircuit(T, (L,))
    g = torch.Generator().manual_seed(3)
    th, ph = c.flush_settings(torch.rand(1, T, generator=g, dtype=torch.float64) * 2,
                              torch.rand(1, T, generator=g, dtype=torch.float64) * 6)
    k = gbs_loop_kernel(c, n_src, r, p)
    st = GaussianInput(r=np.full(n_src, r), modes=tuple(range(n_src)), m=c.m)
    U = c.unitary(th, ph).numpy()
    from itertools import product
    pats = [list(q) for q in product(range(3), repeat=T) if sum(q) % 2 == 0 and sum(q) <= 4]
    got = k.probabilities(torch.as_tensor(pats), th, ph).numpy()
    ref = np.array([pure_probability(st, U, q + [0] * L) for q in pats])
    assert ref.sum() > 0.9 and np.abs(got - ref).max() < 1e-10            # the cap drops ~1e-7 here
    # leak: the model (first order, conservative) vs the weights, at a cap where it is ~1e-2
    th2 = torch.rand(1, 12, generator=g, dtype=torch.float64) * 2
    ph2 = torch.rand(1, 12, generator=g, dtype=torch.float64) * 6
    _, _, leak, _ = cap((1,), 6, 2.0, r, 1e-4, 0, 40, th2.numpy(), ph2.numpy())
    k2 = gbs_loop_kernel(TimeLoopCircuit(12, (1,)), 6, r, 8)
    _, lw = k2.sample_weighted(2000, th2, ph2, generator=g)
    assert 0.9 * float(lw.mean()) < leak[8] < 1.25 * float(lw.mean())
