"""Random-source-order chain rule for superposed sources (`clifford.sbs_rso`, docs/ALGORITHMS.md).

Oracles, none of which shares code with the sampler: brute-force rectangular permanents from
`slos.reference.permanent`; the EXACT law of the algorithm (every source order, every branch) against the
Dicke-input law; `slos.ProductSLOSKernel.probs` (graded SLOS) for the samples, lossless and lossy, with and
without distinguishability; and Perceval's SLOS backend on the coherent product StateVector (the independent
public oracle of the methods paper).
"""
import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import ProductSuperposition
from photonic_engines.clifford import sbs_rso
from photonic_engines.clifford.sbs_rso import rect_cofactors, sample
from photonic_engines.slos import ProductSLOSKernel
from photonic_engines.slos.reference import permanent

EPS = np.finfo(np.float64).eps


def _per_rect(B):
    """sum over the q-subsets of columns of the square permanents, q = rows."""
    q, N = B.shape
    if q == 0:
        return 1.0 + 0j
    return sum(complex(permanent(B[:, list(S)])) for S in itertools.combinations(range(N), q))


def _tvd_and_floor(occ, ref, shots):
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / shots
    tvd = 0.5 * sum(abs(emp.get(k, 0.0) - ref.get(k, 0.0)) for k in set(emp) | set(ref))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in ref.values())
    return tvd, floor


@pytest.mark.parametrize(("q", "N"), [(1, 2), (2, 5), (3, 6), (4, 7), (5, 9)])
def test_rect_cofactors_match_brute_force(q, N):
    """Every leave-one-out rectangular permanent, bunched rows included, two streams at once."""
    U = haar_unitary(12, seed=q * 10 + N)
    rows = [[0, 3, 3, 5, 1][:q], [2, 2, 2, 7, 7][:q]]
    B = torch.stack([U[r][:, :N] for r in rows])                                  # [2, q, N]
    got = rect_cofactors(B)
    for s in range(2):
        ref = torch.tensor([_per_rect(B[s][:, [c for c in range(N) if c != x]]) for x in range(N)], dtype=torch.complex128)
        assert torch.allclose(got[s], ref, rtol=0, atol=1e3 * EPS * float(ref.abs().max())), s


def test_sign_chunking_and_a_tiny_budget_change_nothing(monkeypatch):
    U = haar_unitary(14, seed=3)
    B = U[[0, 4, 4, 9, 2, 11]][:, :10].unsqueeze(0)
    full = rect_cofactors(B)
    for ch in (1, 2, 4, 8):
        assert torch.allclose(full, rect_cofactors(B, sign_chunk=ch), rtol=0, atol=1e3 * EPS * float(full.abs().max()))
    monkeypatch.setattr(sbs_rso, "_budget_bytes", lambda dev: 1)
    assert torch.allclose(full, rect_cofactors(B), rtol=0, atol=1e3 * EPS * float(full.abs().max()))


@pytest.mark.parametrize(("n", "m", "k"), [(3, 4, 2), (4, 5, 2), (4, 6, 3), (5, 6, 3)])
def test_exact_law_of_the_algorithm(n, m, k):
    """The theorem itself: average the chain's branch probabilities over EVERY source order and compare
    with |Per_rect(A[r, :])|^2 (n-k)!/n! on every ordered tuple. Uses the module's cofactors, so this
    checks the offset-start chain, not the arithmetic (covered above)."""
    A = haar_unitary(m, seed=n + 7 * k)[:, :n]
    true = {r: abs(_per_rect(A[list(r)])) ** 2 * math.factorial(n - k) / math.factorial(n)
            for r in itertools.product(range(m), repeat=k)}
    law = dict.fromkeys(true, 0.0)
    perms = list(itertools.permutations(range(n)))
    for sig in perms:
        As = A[:, list(sig)]

        def rec(prefix, p, As=As):
            j = len(prefix) + 1
            if j > k:
                law[prefix] += p / len(perms)
                return
            AT = As[:, :n - k + j]
            P = rect_cofactors(AT[list(prefix)].unsqueeze(0))[0] if prefix else torch.ones(n - k + j, dtype=As.dtype)
            w = (AT @ P).abs() ** 2
            w = w / w.sum()
            for r in range(m):
                if float(w[r]) > 0:
                    rec((*prefix, r), p * float(w[r]))
        rec((), 1.0)
    assert sum(true.values()) == pytest.approx(1.0, abs=1e-12)
    assert max(abs(true[r] - law[r]) for r in true) < 1e-13


def test_k_equals_n_is_clifford_clifford():
    """alpha = pi/2: every source fires, the offset set is empty, the law is the Fock law."""
    n, m, shots = 3, 6, 3000
    st, U = ProductSuperposition.sbs(n, m, alpha=math.pi / 2), haar_unitary(m, seed=9)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U).as_dict()
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(2))
    assert (occ.sum(1) == n).all()
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


@pytest.mark.parametrize(("sources", "m", "alpha"), [(4, 8, math.pi / 4), (5, 6, 1.1), (6, 6, math.pi / 4)])
def test_samples_match_the_strong_simulation(sources, m, alpha):
    st, U = ProductSuperposition.sbs(sources, m, alpha=alpha), haar_unitary(m, seed=sources + 1)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U).as_dict()
    shots = 4000
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(0))
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


@pytest.mark.parametrize(("alpha", "eta", "x"), [(math.pi / 4, 0.6, None), (1.1, 0.8, None),
                                                 (math.pi / 4, 1.0, 0.7), (1.0, 0.7, 0.8)])
def test_lossy_and_distinguishable_samples_match_the_strong_simulation(alpha, eta, x):
    sources, m, shots = 4, 7, 4000
    st, U = ProductSuperposition.sbs(sources, m, alpha=alpha), haar_unitary(m, seed=44)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U, loss=eta, indist=x).as_dict()
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(1), loss=eta, indist=x)
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_post_selected_sector_matches_the_conditional():
    sources, m, k, shots = 6, 10, 3, 3000
    st, U = ProductSuperposition.sbs(sources, m), haar_unitary(m, seed=5)
    dist = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U)
    ref = {tuple(int(v) for v in o): float(q) for o, q in zip(dist.occupations(k), dist.conditional(k).numpy())}
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(3), photons=k)
    assert (occ.sum(1) == k).all()
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_photon_number_is_the_incoherent_twin_law():
    """k ~ Bin(n, eta sin^2 alpha): the photon law of Fock at eta' = eta sin^2 alpha, whatever U."""
    sources, m, shots, alpha, eta = 8, 16, 4000, 1.0, 0.7
    st = ProductSuperposition.sbs(sources, m, alpha=alpha)
    occ = sample(st, haar_unitary(m, seed=7), shots, generator=torch.Generator().manual_seed(1), loss=eta)
    p = eta * math.sin(alpha) ** 2
    expected = np.array([math.comb(sources, k) * p ** k * (1 - p) ** (sources - k) for k in range(sources + 1)])
    counts = np.bincount(occ.sum(1).numpy(), minlength=sources + 1) / shots
    tol = 4 * sum(math.sqrt(q * (1 - q) / shots) for q in expected)
    assert np.abs(counts - expected).sum() < tol, (counts, expected)


def test_eta_one_is_the_lossless_sampler():
    st, U = ProductSuperposition.sbs(5, 8), haar_unitary(8, seed=4)
    a = sample(st, U, 200, generator=torch.Generator().manual_seed(8))
    b = sample(st, U, 200, generator=torch.Generator().manual_seed(8), loss=1.0)
    assert torch.equal(a, b)


def test_samples_match_perceval_slos():
    """The independent public oracle: Perceval SLOS on the coherent product StateVector."""
    pcvl = pytest.importorskip("perceval")
    sources, m, alpha, shots = 5, 8, math.pi / 4, 6000
    st, U = ProductSuperposition.sbs(sources, m, alpha=alpha), haar_unitary(m, seed=21)
    c, s = float(math.cos(alpha)), float(math.sin(alpha))
    sv = None
    for i in range(m):
        f = (pcvl.StateVector([0]) * c + pcvl.StateVector([1]) * s) if i < sources else pcvl.StateVector([0])
        sv = f if sv is None else sv * f
    sim = pcvl.Simulator(pcvl.SLOSBackend())
    sim.set_circuit(pcvl.Unitary(pcvl.Matrix(U.numpy())))
    ref = {tuple(o): float(p) for o, p in sim.probs(sv).items()}
    assert sum(ref.values()) == pytest.approx(1.0, abs=1e-12)
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(5))
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_errors():
    st, U = ProductSuperposition.sbs(4, 8), haar_unitary(8, seed=1)
    with pytest.raises(ValueError, match="conditions the masks"):
        sample(st, U, 2, loss=0.5, photons=2)
    amps = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=complex)
    with pytest.raises(ValueError, match="uniform alpha only"):
        sample(ProductSuperposition(amps, (0, 1), 6), haar_unitary(6, seed=2), 2)


@pytest.mark.parametrize(("N", "q"), [(12, 5), (30, 15), (40, 20)])
def test_mixed_division_matches_prefix_suffix(N, q):
    """The default leave-one-out (full product, forward / backward division by |a_x|) against the
    prefix x suffix reference, on Glynn vectors with |a_x| on both sides of 1."""
    U = haar_unitary(2 * N, seed=N + q)
    B = U[torch.arange(q) * 2][:, :N].to(torch.complex128)
    sd = 1 - 2 * torch.randint(0, 2, (256, q), generator=torch.Generator().manual_seed(q)).to(torch.complex128)
    a = sd @ B
    assert bool((a.abs() > 1).any()) or N < 30
    ref, got = sbs_rso._loo_prefix(a, q), sbs_rso._loo_divide(a, q)
    assert float((ref - got).abs().max()) < 1e-12 * float(ref.abs().max())
    B1 = B.unsqueeze(0)
    assert torch.allclose(rect_cofactors(B1[:, :8]), rect_cofactors(B1[:, :8], loo="prefix"), rtol=0, atol=1e-13)


def _distinct(B):
    """[q, N] rows -> (distinct rows [D, N], multiplicities) in first-seen order."""
    keys, rows, mu = [], [], []
    for r in B:
        hit = next((i for i, k in enumerate(keys) if torch.equal(k, r)), None)
        if hit is None:
            keys.append(r)
            rows.append(r)
            mu.append(1)
        else:
            mu[hit] += 1
    return torch.stack(rows), mu


@pytest.mark.parametrize(("rows", "N"), [([3], 4), ([1, 1], 5), ([0, 3, 3, 5, 1], 9), ([2, 2, 2, 7, 7], 8),
                                         ([4, 4, 6, 6], 7), ([5] * 6, 10), ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 14),
                                         ([0, 0, 1, 1, 1, 2, 9, 9, 9, 9, 4, 4], 20)])
def test_collapsed_cofactors_match_the_plain_glynn_sum(rows, N):
    """C&C 2020 collapse over repeated rows + DFT leave-one-out (torch twin of the Triton kernel) against the
    uncollapsed prefix x suffix reference: odd and all-even multiplicities, a single repeated row, no repeats."""
    U = haar_unitary(2 * N, seed=len(rows) + N)
    B = U[rows][:, :N].to(torch.complex128)
    ref = rect_cofactors(B.unsqueeze(0), loo="prefix")[0]
    b, mu = _distinct(B)
    for loo in ("dft", "divide"):
        got = sbs_rso.rect_cofactors_collapsed_torch(b, mu, term_chunk=7, loo=loo)
        # rounding of a sum of prod(mu+1) terms, each a ratio of N-factor products (at L = N points for dft)
        assert float((got - ref).abs().max()) < 1e2 * N * EPS * float(ref.abs().max()), loo


@pytest.mark.parametrize("backend", ["collapsed", "torch"])
def test_collapsed_chain_matches_the_strong_simulation(backend):
    """The whole sampler with the collapsed cofactors forced from the second step on (collapse_from=1)."""
    st, U = ProductSuperposition.sbs(5, 6, alpha=1.1), haar_unitary(6, seed=6)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U).as_dict()
    shots = 4000
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(0), backend=backend, collapse_from=1)
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_collapsed_and_plain_chains_draw_the_same_samples():
    """Same generator, weights equal to rounding: the collapsed and plain chains return the same draws."""
    st, U = ProductSuperposition.sbs(7, 9, alpha=1.2), haar_unitary(9, seed=4)
    a = sample(st, U, 300, generator=torch.Generator().manual_seed(5), backend="torch")
    b = sample(st, U, 300, generator=torch.Generator().manual_seed(5), backend="collapsed", collapse_from=1)
    assert torch.equal(a, b)
    c = sample(st, U, 300, generator=torch.Generator().manual_seed(5), backend="collapsed", collapse_from=12)
    assert torch.equal(a, c)


@pytest.mark.slow
@pytest.mark.parametrize(("rows", "N"), [([0, 3, 3, 5, 1], 9), ([4, 4, 6, 6], 7),
                                         ([0, 0, 1, 1, 1, 2, 9, 9, 9, 9, 4, 4], 20)])
def test_triton_collapsed_cofactors_match_the_torch_twin(rows, N):
    """The Triton kernel (interpreter on CPU, compiled on cuda) against its torch twin."""
    pytest.importorskip("triton")
    from photonic_engines.clifford.sbs_rso_triton import rect_cofactors_collapsed_triton
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    U = haar_unitary(2 * N, seed=len(rows) + N)
    b, mu = _distinct(U[rows][:, :N].to(torch.complex128))
    ref = sbs_rso.rect_cofactors_collapsed_torch(b, mu, loo="dft")
    got = rect_cofactors_collapsed_triton(b.to(dev), mu, BT=4).cpu()
    assert float((got - ref).abs().max()) < 1e2 * N * EPS * float(ref.abs().max())


@pytest.mark.slow
def test_triton_backend_draws_the_same_samples():
    """The whole chain on the Triton backend (interpreter on CPU, compiled on cuda) reproduces the plain chain's draws."""
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st, U = ProductSuperposition.sbs(6, 8, alpha=1.2), haar_unitary(8, seed=7)
    a = sample(st, U, 24, generator=torch.Generator().manual_seed(3), backend="torch", device=dev)
    b = sample(st, U, 24, generator=torch.Generator().manual_seed(3), backend="triton", collapse_from=2, device=dev)
    assert torch.equal(a, b)


def test_steps_wider_than_the_cuda_tiers_fall_back_to_the_collapsed_sum(monkeypatch):
    """backend="emul" with the tier ladder cut at N = 8: the wider steps take the collapsed path, draws unchanged."""
    from photonic_engines.clifford import cuda_rect
    if not cuda_rect.emul_available():
        pytest.skip("no C++20 compiler for the kernel emulation")
    st, U = ProductSuperposition.sbs(10, 12, alpha=1.0), haar_unitary(12, seed=8)
    a = sample(st, U, 40, generator=torch.Generator().manual_seed(4), backend="torch")
    monkeypatch.setattr(cuda_rect, "_TIERS_LADDER", (8,))
    b = sample(st, U, 40, generator=torch.Generator().manual_seed(4), backend="emul")
    assert torch.equal(a, b)


@pytest.mark.slow
@pytest.mark.parametrize(("alpha", "eta"), [(math.pi / 2, 1.0), (math.pi / 2, 0.6), (1.1, 0.7)])
def test_square_strata_on_the_fock_engine_match_the_strong_simulation(alpha, eta):
    """square="triton": the k = n_live strata go to fock_sample (Triton, interpreter on CPU). alpha = pi/2 lossless
    is all-square, its lossy twin is all-square on the live subsets, alpha = 1.1 mixes square and offset strata."""
    pytest.importorskip("triton")
    st, U = ProductSuperposition.sbs(4, 6, alpha=alpha), haar_unitary(6, seed=11)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U, **({} if eta >= 1 else {"loss": eta})).as_dict()
    shots = 1200
    dev = "cuda" if torch.cuda.is_available() else "cpu"                          # compiled Triton needs device tensors
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(1), loss=None if eta >= 1 else eta, square="triton",
                 device=dev)
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_square_option_errors():
    st, U = ProductSuperposition.sbs(3, 6), haar_unitary(6, seed=1)
    with pytest.raises(ValueError, match="square"):
        sample(st, U, 2, square="fock")


@pytest.mark.parametrize(("N", "q"), [(6, 5), (10, 7), (12, 3), (30, 25), (40, 39)])
def test_min_degree_leave_one_out_matches_prefix_suffix(N, q):
    """`_loo_minform` (truncation at min(q, N - 1 - q), the reversed form above the middle) against `_loo_prefix`."""
    g = torch.Generator().manual_seed(N + q)
    a = (torch.randn(32, N, generator=g, dtype=torch.float64) + 1j * torch.randn(32, N, generator=g, dtype=torch.float64)) * 0.7
    ref, got = sbs_rso._loo_prefix(a, q), sbs_rso._loo_minform(a, q)
    # both division-free sums of the same products, in different orders: rounding level of an N-term product sum
    assert float((ref - got).abs().max()) < 1e2 * N * EPS * float(ref.abs().max())


@pytest.mark.slow
@pytest.mark.parametrize(("backend", "alpha", "eta"), [("torch", math.pi / 4, 1.0), ("torch", 1.1, 0.7), ("collapsed", 1.2, 1.0)])
def test_device_draw_matches_the_strong_simulation(backend, alpha, eta):
    """draw="device" (the Fock sampler's fused Triton draw, interpreter on CPU) on the rso chain proper."""
    pytest.importorskip("triton")
    st, U = ProductSuperposition.sbs(4, 6, alpha=alpha), haar_unitary(6, seed=12)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U, **({} if eta >= 1 else {"loss": eta})).as_dict()
    shots = 1500
    dev = "cuda" if torch.cuda.is_available() else "cpu"                          # compiled Triton needs device tensors
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(2), loss=None if eta >= 1 else eta,
                 backend=backend, collapse_from=2, square="chain", draw="device", device=dev)
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)
    with pytest.raises(ValueError, match="draw"):
        sample(st, U, 2, draw="gpu")


@pytest.mark.slow
@pytest.mark.parametrize(("alpha", "eta"), [(math.pi / 2, 1.0), (math.pi / 2, 0.6), (1.1, 0.7)])
def test_square_strata_on_the_fock_step_kernels_match_the_strong_simulation(alpha, eta):
    """square="kernel": the k = n_live strata stay on the rso chain with the Fock engine's step kernels (Triton multi
    + draw_step; interpreter on CPU, compiled on cuda)."""
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"                          # compiled Triton needs device tensors
    st, U = ProductSuperposition.sbs(4, 6, alpha=alpha), haar_unitary(6, seed=13)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U, **({} if eta >= 1 else {"loss": eta})).as_dict()
    shots = 1200
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(4), loss=None if eta >= 1 else eta,
                 square="kernel", device=dev)
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_warm_cuda_compiles_each_family_once(monkeypatch):
    """`_warm_cuda` calls the lambda, mu and (square="kernel") Fock warm-ups once per process each."""
    from photonic_engines.clifford import cuda_cofactors, cuda_rect
    calls = []
    monkeypatch.setattr(cuda_rect, "warmup", lambda: calls.append("lambda"))
    monkeypatch.setattr(cuda_rect, "warmup_mu", lambda: calls.append("mu"))
    monkeypatch.setattr(cuda_cofactors, "warmup", lambda: calls.append("fock"))
    monkeypatch.setattr(sbs_rso, "_WARMED", set())
    sbs_rso._warm_cuda("cuda")
    assert sorted(calls) == ["lambda", "mu"]
    sbs_rso._warm_cuda("kernel")
    sbs_rso._warm_cuda("kernel")
    assert sorted(calls) == ["fock", "lambda", "mu"]
