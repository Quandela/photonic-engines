"""The rso cost model of `bench/sbs_run_cost.py`: collapsed Haar-averaged term count, form dispatch, fit.

Oracles: brute-force enumeration of the output multisets with `sbs_rso.collapse_tables` (the sampler's own term
count) for the closed form; `cuda_rect.choose_form` for the dispatch; synthetic data with known constants for the
non-negative fit; the uncollapsed formula of runs <= 20261004 for the legacy features.
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import sbs_run_cost as rc

from photonic_engines.clifford import cuda_rect
from photonic_engines.clifford.sbs_rso import collapse_tables


def _brute_terms(q: int, m: int) -> float:
    tot = 0
    count = 0
    for ms in itertools.combinations_with_replacement(range(m), q):
        mu = [x for x in np.bincount(ms, minlength=m).tolist() if x]
        tot += collapse_tables(mu)[3]
        count += 1
    assert count == math.comb(m + q - 1, q)
    return tot / count


@pytest.mark.parametrize("q,m", [(1, 3), (2, 3), (3, 4), (4, 4), (5, 3), (6, 5), (4, 8)])
def test_terms_closed_form_matches_enumeration(q, m):
    assert rc.rso_terms(q, m) == pytest.approx(_brute_terms(q, m), rel=1e-12)


def test_terms_plain_and_limits():
    assert rc.rso_terms(0, 10) == 1.0
    assert rc.rso_terms(7, 10, collapsed=False) == 2.0 ** 6
    assert rc.rso_terms(7, 10 ** 6) == pytest.approx(2.0 ** 6, rel=1e-4)       # no collisions: plain count
    # growth along m = c n reproduces the Fock engine's rho(c): C(3n,n)/C(2n,n) and C(5n,n)/C(3n,n)
    for c, rho in [(1, 27 / 16), (2, 5 ** 5 / 4 ** 4 / (27 / 4))]:
        n1, n2 = 200, 400
        g = (rc.rso_terms(n2, c * n2) / rc.rso_terms(n1, c * n1)) ** (1 / (n2 - n1))
        assert g == pytest.approx(rho, rel=5e-3)


def test_form_dispatch_matches_kernel():
    for N in range(2, cuda_rect._TIERS_LADDER[-1] + 1):
        for q in range(N):
            assert rc.rso_form(N - 1 - q, N) == cuda_rect.choose_form(q, N)
    assert rc.rso_form(0, cuda_rect._TIERS_LADDER[-1] + 1) == "lambda"


def test_legacy_features_unchanged():
    n, m, k = 30, 60, 20
    sw = lw = 0.0
    for j in range(1, k + 1):
        N = n - k + j
        sw += 2 ** max(j - 2, 0) * (N * N / 2 + N * j) if j > 1 else 0.0
        lw += m * N
    assert rc.rso_features(n, m, k, legacy=True) == (float(k), sw, lw)
    price = rc.rso_model((1e-3, 2e-11, 3e-8))
    assert price(n, m, k) == pytest.approx(1e-3 * k + 2e-11 * sw + 3e-8 * lw)


def test_features_split_by_form():
    m = 40
    assert rc.rso_features(20, m, 20)[2:4] == (0.0, 0.0)                     # s0 = 0: fock only
    x = rc.rso_features(20, m, 18)                                          # s0 = 2: mu only
    assert x[1] == 0.0 and x[3] == 0.0 and x[2] > 0
    x = rc.rso_features(20, m, 10)                                          # s0 = 10: lambda only
    assert x[1] == 0.0 and x[2] == 0.0 and x[3] > 0
    # collapsed count below the plain one at m = n
    assert rc.rso_features(20, 20, 10)[3] < rc.rso_features(20, 20, 10, collapse_from=None)[3]


def test_fit_recovers_constants_and_grid_covers_forms():
    pts = rc.rso_grid([1, 2, 4], kmeas=20)
    forms = {rc.rso_form(n - k, n) for n, m, k in pts}
    assert forms == {"fock", "mu", "lambda"}
    true = np.array([1e-3, 3e-12, 2e-11, 5e-11, 2e-8])
    X = np.array([rc.rso_features(n, m, k) for n, m, k in pts])
    y = X @ true
    assert rc.nnls_relative(X, y) == pytest.approx(true, rel=1e-6)
    # a form absent from the data keeps a zero constant
    X0 = X.copy()
    X0[:, 2] = 0.0
    assert rc.nnls_relative(X0, X0 @ true)[2] == 0.0


def test_cost_reproduces_the_final_block():
    """`cost` (the callable behind the rate maps) reproduces the n* of the final block computed from the same fit
    (results/keep/sbs_rso_final_run_cost_r22_20261006_h100.csv: m/n = 2, lossless, untruncated 49.77)."""
    if not rc.FINAL_FIT.exists():
        pytest.skip("kept fit CSV not in this checkout")
    a, b = rc.cost(49, 98), rc.cost(50, 100)
    assert isinstance(a, float) and a < 1.0 < b
    assert 49 + (0.0 - math.log(a)) / (math.log(b) - math.log(a)) == pytest.approx(49.77, abs=0.01)
    t, ext = rc.cost(49, 98, detail=True)
    assert t == a and 0.9 < ext < 1.0                         # most of t from strata above k = 22 (LOG: 98-99.5 %)


def test_collapse_from_per_backend():
    assert rc.rso_collapse_from("cuda", 12) == 1
    assert rc.rso_collapse_from("torch", 12) is None
    assert rc.rso_collapse_from("triton", 12) == 12
