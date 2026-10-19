"""Perceval processors with TD components vs the native periodic-network kernel."""
import numpy as np
import pytest

pcvl = pytest.importorskip("perceval")
torch = pytest.importorskip("torch")

from photonic_engines.adapters.perceval import last_period_distribution, periodic_kernel_from_perceval


def _compare(build, inp):
    p = pcvl.Processor("SLOS", len(inp))
    build(p)
    p.with_input(pcvl.BasicState(inp))
    p.min_detected_photons_filter(0)
    ref = {tuple(k): v for k, v in p.probs()["results"].items()}
    k, g = periodic_kernel_from_perceval(p, pcvl.BasicState(inp))
    got = last_period_distribution(k, g)
    assert abs(sum(got.values()) - 1) < 1e-12
    assert max(abs(got.get(s, 0.0) - v) for s, v in ref.items()) < 1e-12
    assert max(abs(ref.get(s, 0.0) - v) for s, v in got.items()) < 1e-12
    return k


def test_bs_td_ps():
    k = _compare(lambda p: (p.add(0, pcvl.BS(theta=1.0)), p.add(1, pcvl.TD(1)), p.add(0, pcvl.PS(0.3))), [1, 0])
    assert (k.T, k.delays) == (2, (1,))


def test_td2_phases_bs_after():
    _compare(lambda p: (p.add(0, pcvl.PS(0.4)), p.add(0, pcvl.BS(theta=0.7, phi_tl=0.2, phi_br=1.1)),
                        p.add(1, pcvl.PS(0.9)), p.add(1, pcvl.TD(2)), p.add(1, pcvl.PS(0.5)),
                        p.add(0, pcvl.BS(theta=0.5))), [1, 0])


def test_three_modes_two_delays():
    _compare(lambda p: (p.add(0, pcvl.BS(theta=1.2)), p.add(1, pcvl.TD(1)), p.add(0, pcvl.PS(0.3)),
                        p.add(0, pcvl.BS.H(theta=0.8)), p.add(2, pcvl.TD(2))), [1, 0, 0])


def test_bunched_input_and_hom():
    _compare(lambda p: (p.add(0, pcvl.BS(theta=1.2)), p.add(1, pcvl.TD(1)), p.add(0, pcvl.BS(theta=0.8)),
                        p.add(2, pcvl.TD(1))), [2, 0, 0])
    _compare(lambda p: (p.add(0, pcvl.TD(1)), p.add(0, pcvl.BS())), [1, 1])


def test_perm():
    _compare(lambda p: (p.add(0, pcvl.BS(theta=1.2)), p.add(0, pcvl.PERM([1, 2, 0])), p.add(1, pcvl.TD(1)),
                        p.add(1, pcvl.BS(theta=0.8))), [1, 1, 0])


def test_sampler_matches_marginals():
    p = pcvl.Processor("SLOS", 2)
    p.add(0, pcvl.BS(theta=1.0))
    p.add(1, pcvl.TD(1))
    p.add(0, pcvl.BS(theta=0.6))
    k, g = periodic_kernel_from_perceval(p, pcvl.BasicState([1, 0]), periods=4)
    marg = k.period_marginals(g).numpy()                                  # [T, Q]
    s = k.sample(20000, g, generator=torch.Generator().manual_seed(0))     # [S, T, 2]
    q = k.basis.pattern_id(s.numpy().reshape(-1, 2)).reshape(20000, k.T)
    emp = np.stack([np.bincount(q[:, t], minlength=k.Q) / 20000 for t in range(k.T)])
    assert np.abs(emp - marg).max() < 0.012
