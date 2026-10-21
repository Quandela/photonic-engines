"""CUDA thread-per-lane rectangular cofactor kernel (clifford/cuda_rect.py: the SBS random-source-order chain on the
production cofactor substrate), run as its CPU emulation — the kernel code itself, each lane's TPL threads
being std::threads joined by a barrier. Needs a C++20 compiler; skipped otherwise."""
import itertools
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import ProductSuperposition
from photonic_engines.clifford import cuda_rect
from photonic_engines.clifford.sbs_rso import rect_cofactors, sample
from photonic_engines.slos.reference import permanent

if not cuda_rect.emul_available():
    pytest.skip("no C++20 compiler for the kernel emulation", allow_module_level=True)


def _per_rect(B):
    q, N = B.shape
    if q == 0:
        return 1.0 + 0j
    return sum(complex(permanent(B[:, list(S)])) for S in itertools.combinations(range(N), q))


@pytest.mark.parametrize(("q", "N", "rowsets"), [
    (0, 3, [[], []]), (1, 2, [[0], [3]]), (2, 5, [[0, 3], [2, 2]]), (3, 6, [[0, 3, 3], [2, 2, 2]]),
    (4, 7, [[0, 3, 3, 5], [2, 2, 7, 7]]), (5, 9, [[0, 3, 3, 5, 1], [2, 2, 2, 7, 7]]), (6, 11, [[1] * 6, [0, 1, 2, 3, 4, 5]]),
])
def test_matches_brute_force(q, N, rowsets):
    """Every leave-one-out rectangular permanent, repeated rows (the 2020 collapse) included; TPL 1, 2, 4; one lane
    per stream and many segments."""
    U = haar_unitary(12, seed=q * 10 + N)
    A = torch.stack([U[:, :N] for _ in rowsets])
    rows = torch.tensor(rowsets, dtype=torch.long).reshape(len(rowsets), q)
    ref = torch.stack([torch.tensor([_per_rect(A[s][rows[s]][:, [c for c in range(N) if c != x]]) for x in range(N)],
                                    dtype=torch.complex128) for s in range(len(rowsets))])
    for tpl in (1, 2, 4):
        for ttl in (1, 1 << 17):
            got = cuda_rect.rect_cofactors_emul(A, rows, q, target_total_lanes=ttl, tpl=tpl)
            assert torch.allclose(got, ref, rtol=0, atol=1e-13 * max(1.0, float(ref.abs().max()))), (tpl, ttl)


def test_matches_the_torch_cofactors_with_both_division_branches():
    """N = 20, q = 12, rows drawn from 8 values (deep collapse): the kernel vs `sbs_rso.rect_cofactors` (torch, the
    uncollapsed Glynn sum) and vs its own long-double build; TPL 1 / 2 / 4 agree with each other."""
    rng = np.random.default_rng(0)
    U = haar_unitary(30, seed=5)
    N, q, S = 20, 12, 3
    A = torch.stack([U[:, :N]] * S)
    rows = torch.tensor(rng.integers(0, 8, size=(S, q)))
    B = torch.stack([A[s][rows[s]] for s in range(S)])
    ref = rect_cofactors(B, loo="prefix")
    scale = float(ref.abs().max())
    a = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=1)
    b = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=4, target_total_lanes=64)
    c = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=2, real="long double")
    assert torch.allclose(a, ref, rtol=0, atol=1e-12 * scale)
    assert torch.allclose(b, ref, rtol=0, atol=1e-12 * scale)
    assert torch.allclose(c, ref, rtol=0, atol=1e-12 * scale)
    assert float((a - c).abs().max()) < 1e-13 * scale


def test_wide_tiers_above_64_columns():
    """N = 70 (tier 72, TPL 8 by default) and N = 90 (tier 96): the lossy SBS curve at eta = 0.5 runs the chain with
    N ~ 85 active columns (run 20261004_121405 stopped at the old ladder's end, N = 64)."""
    rng = np.random.default_rng(1)
    U = haar_unitary(96, seed=9)
    for N, q in ((70, 6), (90, 5)):
        S = 2
        A = torch.stack([U[:, :N]] * S)
        rows = torch.tensor(rng.integers(0, 12, size=(S, q)))
        B = torch.stack([A[s][rows[s]] for s in range(S)])
        ref = rect_cofactors(B, loo="prefix")
        scale = float(ref.abs().max())
        assert cuda_rect.tier(N)[1] == 8
        got = cuda_rect.rect_cofactors_emul(A, rows, q)
        assert torch.allclose(got, ref, rtol=0, atol=1e-12 * scale), N
        got2 = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=4, target_total_lanes=64)
        assert torch.allclose(got2, ref, rtol=0, atol=1e-12 * scale), N


def test_sampler_draws_the_same_samples_on_both_backends():
    """`sample(backend="emul")` reproduces `sample(backend="torch")` from the same generator (equal weights to
    rounding), lossless and lossy."""
    st = ProductSuperposition.sbs(5, 10, alpha=0.9, modes=(0, 2, 4, 6, 8))
    U = haar_unitary(10, seed=3)
    for kw in ({}, {"loss": 0.7}):
        a = sample(st, U, 12, generator=torch.Generator().manual_seed(2), backend="torch", **kw)
        b = sample(st, U, 12, generator=torch.Generator().manual_seed(2), backend="emul", **kw)
        assert torch.equal(a, b), kw
    with pytest.raises(ValueError, match="backend"):
        sample(st, U, 2, backend="nope")


def test_tiers_and_config():
    assert cuda_rect.tier(1)[0] == 8 and cuda_rect.tier(8)[0] == 8 and cuda_rect.tier(9)[0] == 16 and cuda_rect.tier(64)[0] == 64
    assert cuda_rect.tier(65) == (72, 8) and cuda_rect.tier(96) == (96, 8)
    with pytest.raises(StopIteration):
        cuda_rect.tier(97)
    cuda_rect.set_config({8: (4, None)})
    try:
        assert cuda_rect.tier(5) == (8, 4)
    finally:
        cuda_rect.set_config(None)


@pytest.mark.parametrize("form", ["mu", "mu_generic"])
@pytest.mark.parametrize(("N", "q", "tpl", "repeats"), [(7, 6, 1, False), (9, 6, 2, False), (12, 8, 4, True), (15, 8, 2, False),
                                                       (13, 12, 4, True), (20, 14, 2, False)])
def test_mu_form_matches_the_lambda_form(N, q, tpl, repeats, form):
    """The division-free mu-form payloads -- "mu_generic" (runtime degree s0 = N - 1 - q <= DMAX, Hillis-
    Steele prefix / suffix across the lane's threads) and "mu" (the lane templated on D = s0, xor butterfly) -- against
    the lambda-form's long-double build and the torch cofactors; s0 = 0 (the Fock limit), 2, 3, 4, 6; repeated rows;
    TPL 1 / 2 / 4."""
    rng = np.random.default_rng(N * 100 + q)
    U = haar_unitary(2 * N, seed=N + q)
    S = 2
    A = torch.stack([U[:, torch.randperm(2 * N, generator=torch.Generator().manual_seed(s))[:N]] for s in range(S)])
    rows = torch.tensor(rng.integers(0, 5 if repeats else 2 * N, size=(S, q)))
    s0 = N - 1 - q
    assert 0 <= s0 <= cuda_rect.DMAX
    ref = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=1, real="long double", form="lambda")
    scale = float(ref.abs().max())
    got = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form=form)
    assert torch.allclose(got, ref, rtol=0, atol=1e-13 * scale)
    B = torch.stack([A[s][rows[s]] for s in range(S)])
    assert torch.allclose(got, rect_cofactors(B, loo="prefix"), rtol=0, atol=1e-12 * scale)
    got2 = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form=form, target_total_lanes=64)   # many segments
    assert torch.allclose(got2, ref, rtol=0, atol=1e-13 * scale)


# test_choose_form and test_mu_dmax_matches_the_kernel_source: tests/clifford/test_cuda_rect_dispatch.py (no compiler
# needed there, so they run on every host).


def _mu_lane_atol(N, D, scale):
    """fp64 error bound of the mu lane, absolute: c N (D + 1) eps max|P|. One Glynn term carries N columns times D + 1
    polynomial coefficients through its prefix/suffix products; c = 4 covers complex arithmetic and the reference
    itself (on hosts where long double is double, e.g. arm64, the reference is a second fp64 run). Measured by
    bench/mu_lane_error.py (exact integer-DP reference, every case of test_templated_mu_lane_every_degree, with and
    without --nofma): max|err| / (N (D + 1) eps max|P|) <= 1.22 (worst N = 18, D = 4; the N = 23, D = 6 case is
    2.1e-14 relative)."""
    return 4 * N * (D + 1) * 2.0 ** -53 * scale


@pytest.mark.parametrize(("N", "q", "tpls", "repeats"), [
    (5, 4, (1, 2, 4), False), (8, 6, (1, 2, 8), True), (11, 8, (1, 4, 16), False), (14, 10, (2, 8), True),
    (18, 13, (4, 16), False), (21, 14, (8,), False), (23, 16, (1, 4), True)])
def test_templated_mu_lane_every_degree(N, q, tpls, repeats):
    """The templated mu lane (mud_lane, D = s0 = 0 .. 6) against its own long-double build, the generic mu-form, the
    torch cofactors and the brute-force-checked lambda-form; TPL from 1 to 16 (the xor butterfly at every depth), tiers
    not divisible by TPL (NH = ceil(NP / TPL), idle threads), repeated rows, one and many segments per stream. The lane
    is held to its fp64 error bound (_mu_lane_atol)."""
    rng = np.random.default_rng(N * 31 + q)
    U = haar_unitary(2 * N, seed=3 * N + q)
    S = 2
    A = torch.stack([U[:, torch.randperm(2 * N, generator=torch.Generator().manual_seed(7 + s))[:N]] for s in range(S)])
    rows = torch.tensor(rng.integers(0, 4 if repeats else 2 * N, size=(S, q)))
    s0 = N - 1 - q
    ref = cuda_rect.rect_cofactors_emul(A, rows, q, real="long double", form="mu")
    scale = float(ref.abs().max())
    gen = cuda_rect.rect_cofactors_emul(A, rows, q, form="mu_generic")
    lam = cuda_rect.rect_cofactors_emul(A, rows, q, real="long double", form="lambda")
    B = torch.stack([A[s][rows[s]] for s in range(S)])
    tor = rect_cofactors(B, loo="prefix")
    assert torch.allclose(lam, ref, rtol=0, atol=1e-13 * scale), s0
    assert torch.allclose(tor, ref, rtol=0, atol=1e-12 * scale), s0
    assert torch.allclose(gen, ref, rtol=0, atol=1e-13 * scale), s0
    for tpl in tpls:
        for ttl in (1 << 17, 16):
            got = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form="mu", target_total_lanes=ttl)
            assert torch.allclose(got, ref, rtol=0, atol=_mu_lane_atol(N, s0, scale)), (s0, tpl, ttl)


@pytest.mark.parametrize(("N", "q", "tpls", "repeats"), [
    (12, 4, (1, 4), False), (16, 6, (2, 8), True), (20, 9, (4, 16), False), (24, 10, (8,), True), (25, 8, (1, 16), False),
    (26, 5, (4, 16), True), (29, 4, (8, 16), False)])
def test_templated_mu_lane_high_degrees(N, q, tpls, repeats):
    """The templated mu lane at D = s0 = 7 .. 24 (beyond the generic lane's DMAX; the 10-05 / 10-06 extensions, runs
    20261005_222151 / 20261006_120404) against its own long-double build, the long-double lambda-form and the torch
    cofactors; TPL 1 to 16, repeated rows, one and many segments. The fixed 2e-14 below is tighter than _mu_lane_atol
    here (small q): max|err| / max|P| <= 6.6e-15 on these cases (bench/mu_lane_error.py --high)."""
    rng = np.random.default_rng(N * 37 + q)
    U = haar_unitary(2 * N, seed=5 * N + q)
    S = 2
    A = torch.stack([U[:, torch.randperm(2 * N, generator=torch.Generator().manual_seed(11 + s))[:N]] for s in range(S)])
    rows = torch.tensor(rng.integers(0, 4 if repeats else 2 * N, size=(S, q)))
    s0 = N - 1 - q
    assert cuda_rect.DMAX < s0 <= cuda_rect.MU_DMAX
    ref = cuda_rect.rect_cofactors_emul(A, rows, q, real="long double", form="mu")
    scale = float(ref.abs().max())
    lam = cuda_rect.rect_cofactors_emul(A, rows, q, real="long double", form="lambda")
    B = torch.stack([A[s][rows[s]] for s in range(S)])
    tor = rect_cofactors(B, loo="prefix")
    assert torch.allclose(lam, ref, rtol=0, atol=1e-13 * scale), s0
    assert torch.allclose(tor, ref, rtol=0, atol=1e-12 * scale), s0
    for tpl in tpls:
        for ttl in (1 << 17, 16):
            got = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form="mu", target_total_lanes=ttl)
            assert torch.allclose(got, ref, rtol=0, atol=2e-14 * scale), (s0, tpl, ttl)


def test_templated_mu_lane_is_the_fock_cofactor_at_s0_zero():
    """D = 0: the division-free all-but-one product equals the square leave-one-out permanents (brute force)."""
    U = haar_unitary(10, seed=4)
    N, q = 6, 5
    A = torch.stack([U[:, :N], U[:, 2:2 + N]])
    rows = torch.tensor([[0, 3, 3, 5, 7], [1, 1, 1, 2, 9]])
    ref = torch.stack([torch.tensor([_per_rect(A[s][rows[s]][:, [c for c in range(N) if c != x]]) for x in range(N)],
                                    dtype=torch.complex128) for s in range(2)])
    for tpl in (1, 2, 4, 8):
        got = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form="mu")
        assert torch.allclose(got, ref, rtol=0, atol=1e-13 * float(ref.abs().max())), tpl
        assert torch.allclose(cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form="fock"), got, rtol=0, atol=0), tpl


def test_mu_tier_rule_and_config():
    """Tier ladder in steps of 4 to 64 (8 above), power-of-two TPL growing with the degree (the stored prefixes);
    overrides merge and reset; illegal TPL refused."""
    assert cuda_rect.mu_tier(34, 1)[0] == 36 and cuda_rect.mu_tier(65, 2)[0] == 72 and cuda_rect.mu_tier(96, 6)[0] == 96
    for NP in cuda_rect._MU_TIERS:
        tpls = [cuda_rect.mu_tier(NP, D)[1] for D in range(cuda_rect.MU_DMAX + 1)]
        assert tpls == sorted(tpls), NP
    for N in (5, 20, 36, 60, 90):
        for D in range(cuda_rect.MU_DMAX + 1):
            NP, tpl, _ = cuda_rect.mu_tier(N, D)
            assert tpl & (tpl - 1) == 0 and tpl <= 16 and NP >= N
    assert cuda_rect.mu_tier(34, 1)[1] <= cuda_rect.mu_tier(34, 6)[1]
    cuda_rect.set_mu_config({(36, 1): (8, 128)})
    try:
        assert cuda_rect.mu_tier(34, 1) == (36, 8, 128)
        with pytest.raises(ValueError):
            cuda_rect.set_mu_config({(36, 2): (3, None)})
    finally:
        cuda_rect.set_mu_config(None)
    assert cuda_rect.mu_tier(34, 1)[1:] == cuda_rect._mu_default(36, 1)
    with pytest.raises(ValueError):
        cuda_rect.rect_cofactors_emul(torch.ones(1, 3, 4, dtype=torch.complex128), torch.zeros(1, 2, dtype=torch.long), 2,
                                      tpl=3, form="mu")


def test_phantom_cofactors_exact_at_small_s0():
    """The phantom ones-row route through the production Fock kernel (Triton interpreter) equals the rectangular
    cofactors at s0 = 0 and 2 to rounding; at larger s0 it cancels catastrophically (docstring), which is why it is
    not used."""
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") != "1" else "cpu"
    rng = np.random.default_rng(3)
    U = haar_unitary(20, seed=11)
    S = 2
    for N, q in ((7, 6), (9, 6)):
        A = torch.stack([U[:, :N]] * S)
        rows = torch.tensor(rng.integers(0, 20, size=(S, q)))
        ref = cuda_rect.rect_cofactors_emul(A, rows, q, real="long double", form="lambda")
        got = cuda_rect.phantom_cofactors(A.to(dev), rows.to(dev), q, backend="triton").cpu()   # Triton: device tensors
        assert torch.allclose(got, ref, rtol=0, atol=1e-11 * float(ref.abs().max())), (N, q)
