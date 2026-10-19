"""Continuity check (a) of `bench/sbs_rso_validate.py` with the stratified estimator: strata, errors, verdicts,
the 4x re-run on an ambiguous comparison, and one CPU run of the whole check on the kernel's emulation."""
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import sbs_rso_validate as v


def test_strata_cover_the_mass_at_every_p():
    n = 34
    smax = v.strata_needed(n, v.PS, 0.999)
    for p in v.PS:
        mass = sum(math.comb(n, s) * (1 - p) ** s * p ** (n - s) for s in range(smax + 1))
        assert mass >= 0.999
    p = min(v.PS)
    assert sum(math.comb(n, s) * (1 - p) ** s * p ** (n - s) for s in range(smax)) < 0.999   # smallest such s


def test_stratified_sum_and_error():
    tse = {0: (1.0, 0.1), 1: (10.0, 1.0)}
    t, se, mass = v.stratified(2, 1.0, tse)
    assert (t, se, mass) == (1.0, 0.1, 1.0)
    t, se, mass = v.stratified(2, 0.5, tse)                      # Bin(2, 0.5): 1/4, 1/2 on s0 = 0, 1
    assert t == pytest.approx(0.25 + 5.0)
    assert se == pytest.approx(math.hypot(0.025, 0.5))
    assert mass == pytest.approx(0.75)


def test_mean_se():
    mu, se = v.mean_se([1.0, 3.0])
    assert mu == 2.0 and se == pytest.approx(1.0)


def test_judge_states():
    ts = [1.0, 1.0, 1.0, 1.0, 1.0]
    c = {name: state for name, _, _, state in v.judge_a(ts, [0.01] * 5, 1.0, 0.01)}
    assert set(c.values()) == {"GREEN"}
    c = {name: state for name, _, _, state in v.judge_a([1.0, 1.0, 1.0, 2.0, 1.0], [0.01] * 5, 1.0, 0.01)}
    assert c["mono_0.99_1.0"] == "RED" and c["level_0.99"] == "RED"
    c = {name: state for name, _, _, state in v.judge_a([1.0, 1.0, 1.0, 1.29, 1.0], [0.01] * 5, 1.0, 0.01)}
    assert c["level_0.99"] == "AMBIGUOUS"                        # margin 0.01, s.e. ~ 0.016


def _args(**kw):
    a = {"batches": 2, "mass": 0.999, "stratum_shots": 4, "fock_shots": 4}
    a.update(kw)
    return types.SimpleNamespace(**a)


def _fake(monkeypatch, cost):
    """Replace the timing by cost(kind, round) per batch; count calls. Sampling calls are stubbed out."""
    calls = []

    def timed(fn, nb, dev):
        calls.append(("batch", nb))
        return [cost(len(calls))] * nb
    monkeypatch.setattr(v, "_timed_batches", timed)
    import photonic_engines.clifford as cl
    from photonic_engines.clifford import sbs_rso
    monkeypatch.setattr(sbs_rso, "sample", lambda *a, **k: None)
    monkeypatch.setattr(cl, "fock_sample", lambda *a, **k: None)
    return calls


def test_gate_green_single_round(monkeypatch):
    calls = _fake(monkeypatch, lambda i: 1.0)                   # every stratum costs t_Fock: flat in p
    rows = []
    ok, tf = v.gate_a(_args(), 10, 20, None, torch.device("cpu"), {}, "triton", lambda **kw: rows.append(kw))
    assert ok and tf == 1.0
    smax = v.strata_needed(10, v.PS, 0.999)
    assert len(calls) == 1 + smax + 1                            # t_Fock + one call per stratum, no re-run
    assert {r["round"] for r in rows} == {"first"}
    assert [r["p"] for r in rows if r["check"] == "a"] == list(v.PS)


def test_gate_ambiguous_reruns_4x(monkeypatch):
    # within one batch-to-batch spread of the level threshold: t(0.99) ~ 1.3 t_Fock with noisy batches
    state = {"i": 0}
    fock_calls = (1, 1 + v.strata_needed(10, v.PS, 0.999) + 1 + 1)        # t_Fock first in each round

    def timed(fn, nb, dev):
        state["i"] += 1
        return [1.0 + (0.2 if b % 2 else -0.2) for b in range(nb)] if state["i"] in fock_calls else [1.3] * nb
    monkeypatch.setattr(v, "_timed_batches", timed)
    import photonic_engines.clifford as cl
    from photonic_engines.clifford import sbs_rso
    monkeypatch.setattr(sbs_rso, "sample", lambda *a, **k: None)
    monkeypatch.setattr(cl, "fock_sample", lambda *a, **k: None)
    rows = []
    ok, tf = v.gate_a(_args(), 10, 20, None, torch.device("cpu"), {}, "triton", lambda **kw: rows.append(kw))
    assert {r["round"] for r in rows} == {"pooled_4x"}
    fock = next(r for r in rows if r["check"] == "a_fock")
    assert fock["shots"] == 4 * 4                                # 2 batches of 2, then 6 more: 4x the shots
    assert not ok                                                # still ambiguous / red: not a verdict


def test_gate_a_runs_on_the_cpu_emulation(monkeypatch):
    """End to end on the CPU: the rso chain on cuda_rect's emulation (backend emul), t_Fock from the rso chain at
    k = n (the Fock engine needs a GPU); checks the plumbing, not timings."""
    import photonic_engines.clifford as cl
    from photonic_engines.clifford import sbs_rso
    n, m = 6, 12
    U = v.haar(m, seed=3, dev="cpu")
    st = __import__("photonic_engines").ProductSuperposition.sbs(n, m)
    monkeypatch.setattr(cl, "fock_sample", lambda U_, modes, shots, generator=None, backend=None:
                        sbs_rso.sample(st, U_, shots, generator=torch.Generator().manual_seed(0), photons=n,
                                       backend="emul", square="chain"))
    rows = []
    ok, tf = v.gate_a(_args(), n, m, U, torch.device("cpu"), {"backend": "emul", "square": "chain",
                      "device": torch.device("cpu"), "draw": "host"}, "triton", lambda **kw: rows.append(kw))
    assert tf > 0 and isinstance(ok, bool)
    a = [r for r in rows if r["check"] == "a"]
    assert len(a) == 5 and all(r["value"] > 0 and r["se"] >= 0 for r in a)
    assert all(r["mass"] >= 0.999 for r in a)
