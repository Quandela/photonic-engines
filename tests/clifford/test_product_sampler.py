"""Chain-rule sampler for ProductSuperposition inputs (`clifford.product_sampler`).

The oracles: brute-force residual sums built from `slos.reference.permanent` for the per-step
marginals, and `slos.ProductSLOSKernel.probs` — a completely different engine — for the samples.
"""
import math
from itertools import combinations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import ProductSuperposition, clifford
from photonic_engines.clifford.product_sampler import lossy_source, marginals, sample
from photonic_engines.slos import ProductSLOSKernel
from photonic_engines.slos.reference import permanent

EPS = np.finfo(np.float64).eps


def _brute(U, rows, n, k):
    r"""M_j(r) = sum_{|R| = k-j} |sum_{S subset [n]\R, |S| = j} Per(U[r, S])|^2, written out."""
    j = len(rows)
    tot = 0.0
    for R in combinations(range(n), k - j):
        keep = [i for i in range(n) if i not in R]
        amp = sum(complex(permanent(U[list(rows)][:, list(S)])) for S in combinations(keep, j)) if j else 1.0 + 0j
        tot += abs(amp) ** 2
    return tot


def _tvd_and_floor(occ, ref, shots):
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / shots
    tvd = 0.5 * sum(abs(emp.get(k, 0.0) - ref.get(k, 0.0)) for k in set(emp) | set(ref))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in ref.values())
    return tvd, floor


@pytest.mark.parametrize("method", ["replica", "residual"])
@pytest.mark.parametrize(("sources", "k", "m"), [(5, 3, 8), (6, 4, 10)])
def test_marginals_match_brute_force(method, sources, k, m):
    """Every step of the chain, including a bunched prefix — both evaluations against the sum."""
    st, U = ProductSuperposition.sbs(sources, m), haar_unitary(m, seed=sources)
    for j in range(1, k + 1):
        rows = [0, 2, 2][:j - 1]
        got = marginals(st, U, rows, list(range(m)), k, method=method)
        ref = torch.tensor([_brute(U, [*rows, c], sources, k) for c in range(m)], dtype=torch.float64)
        assert torch.allclose(got, ref, rtol=0, atol=4096 * EPS * float(ref.max())), (j, method)


@pytest.mark.parametrize("method", ["replica", "residual"])
def test_chunking_is_exact(method):
    """The delta sum must complete before it is squared; chunking it must change nothing."""
    st, U = ProductSuperposition.sbs(6, 10), haar_unitary(10, seed=6)
    for rows in ([0], [0, 3], [0, 3, 3]):
        full = marginals(st, U, rows, list(range(10)), 4, method=method)
        for ch in (1, 2, 3):
            part = marginals(st, U, rows, list(range(10)), 4, method=method, chunk=ch)
            assert torch.allclose(full, part, rtol=0, atol=4096 * EPS * float(full.max())), (rows, ch)


@pytest.mark.parametrize("method", ["replica", "residual"])
def test_a_tiny_memory_budget_changes_nothing(method, monkeypatch):
    """The batching that bounds the polynomial tables must be invisible in the answer.

    Without a bound those tables reach 96 GiB at j = 8 with 126 candidates and 256 sign vectors
    per batch, which is what killed the sbs arm of bench/find_threshold.py (run 20260919_195403).
    Forcing the smallest possible batches exercises every split at once.
    """
    from photonic_engines.clifford import product_sampler as ps
    st, U = ProductSuperposition.sbs(6, 10), haar_unitary(10, seed=11)
    for rows in ([], [0], [0, 3], [0, 3, 3]):
        full = marginals(st, U, rows, list(range(10)), 4, method=method)
        monkeypatch.setattr(ps, "_budget_bytes", lambda dev: 1)
        try:
            tight = marginals(st, U, rows, list(range(10)), 4, method=method)
        finally:
            monkeypatch.undo()
        assert torch.allclose(full, tight, rtol=0, atol=4096 * EPS * float(full.max())), (rows, method)


def test_the_two_evaluations_agree():
    st, U = ProductSuperposition.sbs(6, 10), haar_unitary(10, seed=2)
    for j, rows in ((1, []), (2, [1]), (3, [1, 4]), (4, [1, 4, 4])):
        a = marginals(st, U, rows, list(range(10)), 4, method="replica")
        b = marginals(st, U, rows, list(range(10)), 4, method="residual")
        assert torch.allclose(a, b, rtol=0, atol=4096 * EPS * float(a.max())), j


@pytest.mark.slow
def test_the_two_evaluations_agree_with_a_deep_prefix_tree():
    """n = 12, k = 8: the residual sum divides up to 7 factors out of the full product.

    `test_the_two_evaluations_agree` only reaches k - j = 3, where the division that the prefix
    tree is built on barely runs. This drives it to depth 7 against an evaluation that shares no
    code with it, which is the only oracle that covers the division at that depth.

    j stops at 4 because the depth is k - j: the deep end of the tree is the SHALLOW end of the
    chain, so j = 1..4 already covers depths 7, 6, 5 and 4, and every further j only re-checks a
    shallower tree against a replica oracle whose 4^j has meanwhile become the whole cost.
    """
    st, U = ProductSuperposition.sbs(12, 20), haar_unitary(20, seed=77)
    for j in range(1, 5):
        rows = [0, 3, 3, 7, 1, 1, 9][:j - 1]
        a = marginals(st, U, rows, list(range(20)), 8, method="replica")
        b = marginals(st, U, rows, list(range(20)), 8, method="residual")
        assert torch.allclose(a, b, rtol=0, atol=65536 * EPS * float(a.max())), (j, float((a - b).abs().max()))


@pytest.mark.parametrize(("sources", "m"), [(4, 8), pytest.param(5, 10, marks=pytest.mark.slow)])
def test_samples_match_the_strong_simulation(sources, m):
    """The whole chain against a different engine: histogram vs `ProductSLOSKernel.probs`."""
    st, U = ProductSuperposition.sbs(sources, m), haar_unitary(m, seed=sources + 1)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U).as_dict()
    shots = 1500
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(0))
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)
    assert occ.shape == (shots, m)


@pytest.mark.parametrize(("alpha", "eta"), [(math.pi / 4, 0.6), (1.1, 0.8),
                                            pytest.param(math.pi / 2, 0.5, id="alpha=pi/2-is-lossy-Fock")])
def test_lossy_samples_match_the_thinned_strong_simulation(alpha, eta):
    """`loss=eta` against `ProductSLOSKernel.probs(U, loss=eta)`, which thins a different engine.

    The lossy source is a mixture of vacuum and a source at a smaller alpha (`lossy_source`), so
    this also checks that decomposition end to end. alpha = pi/2 is the degenerate corner where the
    source is a deterministic single photon and the whole thing IS the Fock modality under loss.
    """
    sources, m, shots = 4, 8, 4000
    st, U = ProductSuperposition.sbs(sources, m, alpha=alpha), haar_unitary(m, seed=44)
    ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U, loss=eta).as_dict()
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(0), loss=eta)
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)
    mean = float(occ.sum(1).double().mean())
    assert abs(mean - sources * eta * math.sin(alpha) ** 2) < 0.1, mean


def test_lossy_source_is_the_documented_mixture():
    """(P(live), alpha') against the Kraus decomposition, and the two limits that name the axis."""
    for alpha, eta in ((math.pi / 4, 0.6), (1.1, 0.8), (0.3, 0.25)):
        c, s = math.cos(alpha), math.sin(alpha)
        w1, ap = lossy_source(alpha, eta)
        assert w1 == pytest.approx(c * c + eta * s * s)
        assert w1 * math.sin(ap) ** 2 == pytest.approx(eta * s * s)          # brightness
        assert w1 * math.cos(ap) * math.sin(ap) == pytest.approx(c * s * math.sqrt(eta))   # coherence
    assert lossy_source(math.pi / 2, 0.4) == pytest.approx((0.4, math.pi / 2))  # eta = p: no coherence
    assert lossy_source(0.7, 1.0) == pytest.approx((1.0, 0.7))                  # eta = 1: untouched


def test_loss_rejects_what_it_cannot_do():
    st, U = ProductSuperposition.sbs(4, 8), haar_unitary(8, seed=1)
    with pytest.raises(ValueError, match="conditions the source mask"):
        sample(st, U, 2, loss=0.5, photons=2)
    amps = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=complex)                 # non-uniform alpha
    with pytest.raises(ValueError, match="uniform alpha only"):
        sample(ProductSuperposition(amps, (0, 1), 6), haar_unitary(6, seed=2), 2, loss=0.5)


def test_post_selected_sector_matches_the_conditional():
    """`photons=k` samples the k-photon sector, i.e. the Dicke input, not the binomial mixture."""
    sources, m, k, shots = 5, 10, 3, 1200
    st, U = ProductSuperposition.sbs(sources, m), haar_unitary(m, seed=5)
    dist = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U)
    occ_k, p_k = dist.occupations(k), dist.conditional(k).numpy()
    ref = {tuple(int(v) for v in o): float(q) for o, q in zip(occ_k, p_k)}
    occ = sample(st, U, shots, generator=torch.Generator().manual_seed(3), photons=k)
    assert (occ.sum(1) == k).all()
    tvd, floor = _tvd_and_floor(occ, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_photon_number_follows_the_binomial():
    sources, m, shots = 6, 12, 1200
    st = ProductSuperposition.sbs(sources, m)
    occ = sample(st, haar_unitary(m, seed=7), shots, generator=torch.Generator().manual_seed(1))
    counts = np.bincount(occ.sum(1).numpy(), minlength=sources + 1) / shots
    expected = st.photon_number_probs()
    # binomial counting noise, summed over the sources + 1 outcomes
    tol = 4 * sum(math.sqrt(q * (1 - q) / shots) for q in expected)
    assert np.abs(counts - expected).sum() < tol, (counts, expected)


def test_dispatch_and_errors():
    st, U = ProductSuperposition.sbs(4, 8), haar_unitary(8, seed=1)
    out = clifford.sample(st, U, 4, generator=torch.Generator().manual_seed(0))
    assert out.shape == (4, 8)
    with pytest.raises(ValueError, match="d = 2 only"):
        amps = np.array([[0.6, 0.6, 0.4], [0.8, 0.5, 0.2]], dtype=complex)
        marginals(ProductSuperposition(amps / np.linalg.norm(amps, axis=1, keepdims=True), (0, 2), 4),
                  haar_unitary(4, seed=1), [], [0], 2)
    with pytest.raises(ValueError, match="k must be in"):
        marginals(st, U, [], [0], 9)
    with pytest.raises(ValueError, match="unknown method"):
        marginals(st, U, [], [0], 2, method="glynn")
