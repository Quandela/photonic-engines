"""Detector dark counts, the generic layer (photonic_engines/imperfections.py, DARK COUNTS):
brute-force convolution, closed forms, sampling against the strong law, conditioning weights."""
import itertools
import math

import numpy as np
import pytest
import torch

from photonic_engines.imperfections import add_dark_counts, dark_distribution, dark_predecessors


def _pois(j, d):
    return math.exp(-d) * d ** j / math.factorial(j)


def _random_law(rng, m, n, P):
    """P distinct n-photon patterns on m modes with random probabilities summing to 1."""
    pats = sorted({tuple(np.bincount(rng.integers(0, m, n), minlength=m).tolist()) for _ in range(4 * P)})[:P]
    p = rng.random(len(pats))
    return np.array(pats, dtype=np.int64), torch.as_tensor(p / p.sum())


def _as_dict(occ, p):
    return {tuple(int(v) for v in o): float(q) for o, q in zip(occ, p)}


def _brute_pnr(occ, p, dark, K):
    out = {}
    for o, q in zip(occ.tolist(), p.tolist()):
        for n in itertools.product(range(K + 1), repeat=len(o)):
            if sum(n) > K:
                continue
            w = math.prod(_pois(nj, dj) for nj, dj in zip(n, dark))
            key = tuple(a + b for a, b in zip(o, n))
            out[key] = out.get(key, 0.0) + q * w
    return out


def _brute_threshold(occ, p, dark):
    out = {}
    for o, q in zip(occ.tolist(), p.tolist()):
        for b in itertools.product((0, 1), repeat=len(o)):
            w = math.prod(-math.expm1(-dj) if bj else math.exp(-dj) for bj, dj in zip(b, dark))
            key = tuple(int(oj > 0 or bj) for oj, bj in zip(o, b))
            out[key] = out.get(key, 0.0) + q * w
    return out


def _close(a, b, tol=1e-14):
    assert {k for k, v in a.items() if v > tol} == {k for k, v in b.items() if v > tol}
    for k in a:
        assert a[k] == pytest.approx(b.get(k, 0.0), abs=tol)


def test_dark_zero_is_the_identity():
    occ = np.array([[1, 0, 1], [0, 2, 0]])
    p = torch.tensor([0.25, 0.75], dtype=torch.float64)
    for dark in (0, 0.0, [0, 0, 0], None, np.zeros(3), torch.zeros(3)):
        o2, p2, dropped = dark_distribution(occ, p, dark, max_dark=None)
        assert p2 is p and float(dropped) == 0.0 and np.array_equal(o2, occ)
        counts = torch.tensor([[1, 0, 3]])
        g = torch.Generator().manual_seed(0)
        state = g.get_state()
        assert add_dark_counts(counts, dark, generator=g) is counts
        assert torch.equal(g.get_state(), state)                          # no draw at all
        assert dark_predecessors([1, 0, 3], dark) == [((1, 0, 3), 1.0)]
    assert torch.equal(add_dark_counts(torch.tensor([[2, 0, 1]]), 0.0, threshold=True), torch.tensor([[1, 0, 1]]))


@pytest.mark.parametrize("K", [0, 1, 3])
def test_pnr_matches_bruteforce(K):
    rng = np.random.default_rng(1)
    occ, p = _random_law(rng, 3, 2, 5)
    dark = [0.3, 0.0, 0.05]
    o2, p2, dropped = dark_distribution(occ, p, dark, max_dark=K)
    _close(_as_dict(o2, p2), _brute_pnr(occ, p, dark, K))
    tail = 1.0 - sum(_pois(j, sum(dark)) for j in range(K + 1))
    assert float(dropped) == pytest.approx(tail, abs=1e-15)
    assert float(p2.sum()) + float(dropped) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("K", [None, 0, 1])
def test_threshold_matches_bruteforce_and_no_click_law(K):
    rng = np.random.default_rng(2)
    occ, p = _random_law(rng, 3, 2, 5)
    occ = np.concatenate([occ, np.zeros((1, 3), dtype=np.int64)])
    p = torch.cat([p * 0.8, torch.tensor([0.2], dtype=torch.float64)])
    dark = [0.2, 0.05, 0.4]
    o2, p2, dropped = dark_distribution(occ, p, dark, max_dark=K, threshold=True)
    assert o2.max() <= 1
    got = _as_dict(o2, p2)
    assert got[(0, 0, 0)] == pytest.approx(math.exp(-sum(dark)) * 0.2, abs=1e-15)   # P(no click) = e^-sum(dark) P_true(vacuum)
    if K is None:
        _close(got, _brute_threshold(occ, p, dark))
        assert float(dropped) == 0.0
    else:                                                                # every visible dark-click set of size <= K kept
        assert all(sum(o) <= 2 + K for o in got)
        assert float(p2.sum()) + float(dropped) == pytest.approx(1.0, abs=1e-14)
        assert float(dropped) > 0


def test_one_mode_closed_forms():
    d = 0.7
    occ = np.array([[0], [2]])
    p = torch.tensor([0.4, 0.6], dtype=torch.float64)
    o2, p2, dropped = dark_distribution(occ, p, d, max_dark=5)
    got = _as_dict(o2, p2)
    for j in range(8):
        exp = 0.4 * (_pois(j, d) if j <= 5 else 0) + 0.6 * (_pois(j - 2, d) if 2 <= j <= 7 else 0)
        assert got.get((j,), 0.0) == pytest.approx(exp, abs=1e-15)
    assert float(dropped) == pytest.approx(1 - sum(_pois(j, d) for j in range(6)), rel=1e-12)
    o3, p3, _ = dark_distribution(occ, p, d, threshold=True)
    assert _as_dict(o3, p3) == pytest.approx({(0,): 0.4 * math.exp(-d), (1,): 0.6 + 0.4 * (1 - math.exp(-d))})


def test_dropped_mass_is_the_poisson_tail_even_when_tiny():
    occ = np.array([[1, 0, 0, 1]])
    p = torch.ones(1, dtype=torch.float64)
    dark = [1e-4, 2e-4, 3e-4, 4e-4]
    lam = sum(dark)
    _, p2, dropped = dark_distribution(occ, p, dark, max_dark=2)
    tail = sum(_pois(j, lam) for j in range(3, 12))
    assert float(dropped) == pytest.approx(tail, rel=1e-9)                # not 1 - (1 - tail): no cancellation
    assert float(p2.sum()) == pytest.approx(1 - tail, abs=1e-15)


@pytest.mark.parametrize("threshold,K", [(False, 2), (True, None), (True, 1)])
def test_predecessors_reproduce_the_convolved_law(threshold, K):
    rng = np.random.default_rng(3)
    occ, p = _random_law(rng, 4, 2, 7)
    if threshold:
        occ, p = np.minimum(occ, 1), p                                    # true click patterns (may repeat)
    dark = [0.1, 0.0, 0.3, 0.02]
    o2, p2, _ = dark_distribution(occ, p, dark, max_dark=K, threshold=threshold)
    true = {}
    for o, q in zip(occ.tolist(), p.tolist()):
        true[tuple(o)] = true.get(tuple(o), 0.0) + q
    for o, q in zip(o2.tolist(), p2.tolist()):
        preds = dark_predecessors(o, dark, max_dark=K, threshold=threshold)
        assert sum(w * true.get(t, 0.0) for t, w in preds) == pytest.approx(q, abs=1e-15)
        assert all(all(a <= b for a, b in zip(t, o)) for t, _ in preds)
        assert all(t[1] == o[1] for t, _ in preds)                       # no dark counts on mode 1


@pytest.mark.parametrize("threshold", [False, True])
def test_samples_match_the_strong_law(threshold):
    rng = np.random.default_rng(4)
    occ, p = _random_law(rng, 3, 2, 4)
    dark = torch.tensor([0.15, 0.4, 0.05], dtype=torch.float64)
    shots = 200_000
    g = torch.Generator().manual_seed(11)
    idx = torch.multinomial(p, shots, replacement=True, generator=g)
    noisy = add_dark_counts(torch.as_tensor(occ)[idx], dark, generator=g, threshold=threshold)
    if not threshold:
        mean_true = (torch.as_tensor(occ, dtype=torch.float64) * p[:, None]).sum(0)
        assert torch.allclose(noisy.double().mean(0), mean_true + dark, atol=5e-3)
    o2, p2, dropped = dark_distribution(occ, p, dark, max_dark=None if threshold else 8, threshold=threshold)
    assert float(dropped) < 1e-7
    emp = {}
    for o in noisy.tolist():
        emp[tuple(o)] = emp.get(tuple(o), 0) + 1 / shots
    exact = _as_dict(o2, p2)
    assert set(emp) <= set(exact)
    tvd = 0.5 * sum(abs(exact[k] - emp.get(k, 0.0)) for k in exact)
    assert tvd < 0.01


def test_batched_and_differentiable():
    rng = np.random.default_rng(5)
    occ, p = _random_law(rng, 3, 2, 4)
    pb = torch.stack([p, p.flip(0)])
    o2, p2, dropped = dark_distribution(occ, pb, [0.1, 0.2, 0.3], max_dark=2)
    for b in range(2):
        ob, pb_, db = dark_distribution(occ, pb[b], [0.1, 0.2, 0.3], max_dark=2)
        assert np.array_equal(ob, o2) and torch.allclose(pb_, p2[b], atol=0) and float(db) == float(dropped[b])
    for threshold, K in ((False, 2), (True, 1)):
        def f(probs, dark):
            q, dr = dark_distribution(occ, probs, dark, max_dark=K, threshold=threshold)[1:]
            return torch.cat([q, dr.reshape(1)])
        d0 = torch.tensor([0.1, 0.05, 0.3], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (p.clone().requires_grad_(), d0))


def test_validation():
    occ = np.array([[1, 0]])
    p = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="max_dark"):
        dark_distribution(occ, p, 0.1)
    with pytest.raises(ValueError, match=">= 0"):
        dark_distribution(occ, p, [-0.1, 0.0], max_dark=1)
    with pytest.raises(ValueError, match="one value per mode"):
        dark_distribution(occ, p, [0.1, 0.1, 0.1], max_dark=1)
    with pytest.raises(ValueError, match="one value per mode"):
        add_dark_counts(torch.zeros((2, 2), dtype=torch.long), torch.tensor([0.1, 0.1, 0.1]))
    with pytest.raises(ValueError, match="clicks"):
        dark_predecessors([2, 0], 0.1, threshold=True)
