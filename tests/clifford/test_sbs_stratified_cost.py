"""`bench/sbs_stratified_cost.py`: the exact joint law of (n_live, k) and the stratified estimator.

Oracles: the law sums to 1 and its k-marginal is Binomial(n, eta sin^2 alpha) (thin-first loss); with every stratum
priced, the estimator is the exact expectation; a missing tail is extrapolated geometrically and reported.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import sbs_stratified_cost as sc
from sbs_run_cost import photon_law


@pytest.mark.parametrize("n,eta,alpha", [(10, 0.7, math.pi / 4), (12, 0.5, math.pi / 2), (8, 1.0, 1.1)])
def test_joint_law_marginal_is_binomial(n, eta, alpha):
    j = sc.joint_law(n, eta, alpha)
    assert j.sum() == pytest.approx(1.0, abs=1e-12)
    assert j.sum(0) == pytest.approx(photon_law(n, eta * math.sin(alpha) ** 2), abs=1e-12)


def test_stratified_exact_when_every_stratum_is_priced():
    n, eta, alpha = 10, 0.7, math.pi / 4
    j = sc.joint_law(n, eta, alpha)
    cost = {(l, k): 1e-3 * 2.0 ** k * (1 + 0.1 * l) for l in range(n + 1) for k in range(1, l + 1) if j[l, k] > 0}
    exact = sum(j[l, k] * c for (l, k), c in cost.items())
    est, covered, ext = sc.stratified(j, cost)
    assert est == pytest.approx(exact, rel=1e-12)
    assert ext == 0.0 and covered == pytest.approx(1.0 - j[:, 0].sum(), abs=1e-12)


def test_stratified_extrapolates_a_missing_geometric_tail():
    n = 14
    j = sc.joint_law(n, 1.0, math.pi / 4)
    full = {(n, k): 1e-3 * 2.0 ** k for k in range(1, n + 1)}
    exact = sum(j[n, k] * c for (_, k), c in full.items())
    cut = {key: c for key, c in full.items() if key[1] <= 10}
    est, covered, ext = sc.stratified(j, cut)
    assert est == pytest.approx(exact, rel=1e-9) and 0 < ext < 1 and covered < 1
    assert np.isfinite(est)
