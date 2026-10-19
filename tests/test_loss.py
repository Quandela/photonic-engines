"""Uniform loss on every engine (photonic_engines.imperfections, docs/LOSS.md): the thinned strong
simulation against Perceval's lossy simulation (LC on every mode), the input-side samplers
against the thinned distribution, time-loop uniform loss against its detector Kraus channel."""
import math

import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.imperfections import (
    LossyDistribution,
    binomial_pmf,
    clicks,
    split_shots,
    thin_counts,
    thin_distribution,
    thin_marginals,
    thin_patterns,
)
from photonic_engines.slos import EmitterSLOSKernel, FockSLOSKernel, SuperpositionSLOSKernel, sample_from_probs

C128 = torch.complex128


def _perceval_lossy(U, occ, eta):
    pcvl = pytest.importorskip("perceval")
    m = len(occ)
    p = pcvl.Processor("SLOS", m)
    p.add(0, pcvl.Unitary(pcvl.Matrix(np.asarray(U))))
    for i in range(m):
        p.add(i, pcvl.LC(1 - eta))
    p.min_detected_photons_filter(0)
    p.with_input(pcvl.BasicState(list(occ)))
    return {tuple(k): float(v) for k, v in p.probs()["results"].items()}


def _empirical(occ):
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / occ.shape[0]
    return emp


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _noise_floor(p, shots):
    """Expected TVD of an empirical distribution: sum_i sqrt(p_i (1-p_i) / shots) sqrt(2/pi) / 2."""
    return 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in p.values())


# ---- core --------------------------------------------------------------------------------------

def test_binomial_and_thin_counts():
    p = binomial_pmf(5, 0.3)
    assert abs(float(p.sum()) - 1) < 1e-12 and abs(float(p[5]) - 0.3 ** 5) < 1e-12
    c = torch.full((20000, 3), 2)
    t = thin_counts(c, 0.25, generator=torch.Generator().manual_seed(0))
    assert t.shape == c.shape and t.max() <= 2 and abs(float(t.float().mean()) - 0.5) < 0.02
    assert (clicks(t) == (t > 0).long()).all()
    assert thin_counts(c, None) is c and thin_counts(c, 1.0) is c


def test_split_shots_groups():
    n, shots, eta = 4, 5000, 0.8
    groups = split_shots(n, shots, eta, generator=torch.Generator().manual_seed(1))
    ids = torch.cat([g for _, g in groups])
    assert sorted(ids.tolist()) == list(range(shots))                  # a partition of the streams
    ks = torch.zeros(n + 1)
    for sub, g in groups:
        assert list(sub) == sorted(set(sub))
        ks[len(sub)] += g.shape[0]
    assert ((ks / shots - binomial_pmf(n, eta)).abs() < 0.03).all()
    assert split_shots(n, 7, None)[0][0] == (0, 1, 2, 3)


@pytest.mark.parametrize("occ,eta", [((1, 1, 1, 0), 0.7), ((2, 0, 1, 0, 0), 0.9), ((1, 1, 0, 0), 0.5)])
def test_thin_distribution_matches_perceval(occ, eta):
    m, n = len(occ), sum(occ)
    U = haar_unitary(m, 11 + n)
    ref = _perceval_lossy(U.numpy(), occ, eta)
    k = FockSLOSKernel(n, m, list(occ), dtype=C128)
    ld = k.probs(U, loss=eta)
    assert isinstance(ld, LossyDistribution) and ld.n == n
    d = ld.as_dict()
    assert max(abs(d[t] - ref.get(t, 0.0)) for t in d) < 1e-9
    assert abs(sum(d.values()) - 1) < 1e-9
    # the explicit-pattern route gives the same numbers, and the binomial masses
    d2 = thin_patterns(k.layers.final_occupations(), k.probs(U), eta).as_dict()
    assert max(abs(d[t] - d2.get(t, 0.0)) for t in d) < 1e-12
    pk = binomial_pmf(n, eta)
    assert all(abs(float(ld.mass(j)) - float(pk[j])) < 1e-9 for j in range(n + 1))
    assert k.probs(U, loss=1.0).shape == (k.output_size(),)


def test_thin_distribution_batched_and_differentiable():
    m, occ = 4, [1, 1, 0, 0]
    U = torch.stack([haar_unitary(m, 1), haar_unitary(m, 2)])
    k = FockSLOSKernel(2, m, occ, dtype=C128)
    eta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    ld = k.probs(U, loss=eta)
    assert ld.probs[2].shape == (2, k.output_size())
    ld.mass(2).sum().backward()
    assert abs(float(eta.grad) - 2 * 2 * 0.8) < 1e-9                    # d/d eta of 2 x eta^2
    # per-batch distribution equals the unbatched one
    single = k.probs(U[1], loss=0.8).as_dict()
    assert _tvd(ld.as_dict(batch=1), single) < 1e-12


def test_superposition_and_emitter_probs_loss():
    m = 4
    U = haar_unitary(m, 5)
    sk = SuperpositionSLOSKernel(2, m, [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=C128, weights=[1 / math.sqrt(2)] * 2)
    d = sk.probs(U, loss=0.6).as_dict()
    assert abs(sum(d.values()) - 1) < 1e-9
    st = EmitterMPS.caterpillar(3, [spin_gate(math.pi / 2, 0), spin_gate(math.pi / 2, 0.4), spin_gate(1.0, 0.2)])
    ek = EmitterSLOSKernel(st, dtype=C128)
    U6 = haar_unitary(6, 5)
    ld = ek.probs(U6, loss=0.6)
    assert abs(sum(ld.as_dict().values()) - 1) < 1e-9                 # traced emitter: mixture thinned
    assert _tvd(ld.as_dict(), thin_distribution(ek.layers, ek.probs(U6), 0.6).as_dict()) < 1e-12


def test_sample_from_probs_loss():
    m, occ = 4, [1, 1, 1, 0]
    U = haar_unitary(m, 3)
    k = FockSLOSKernel(3, m, occ, dtype=C128)
    ld = k.probs(U, loss=0.7)
    s = sample_from_probs(k.layers, ld, 4000, generator=torch.Generator().manual_seed(0))
    assert s.shape == (4000, m)
    assert _tvd(_empirical(s), ld.as_dict()) < 2.5 * _noise_floor(ld.as_dict(), 4000)
    s2 = sample_from_probs(k.layers, k.probs(U), 4000, generator=torch.Generator().manual_seed(0), loss=0.7)
    assert _tvd(_empirical(s2), ld.as_dict()) < 2.5 * _noise_floor(ld.as_dict(), 4000)


def test_thin_marginals_and_clicks():
    marg = torch.tensor([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]], dtype=torch.float64)
    t = thin_marginals(marg, 0.5)
    assert abs(float(t[0, 2]) - 0.3 * 0.25) < 1e-12 and abs(float(t[0, 0]) - (0.2 + 0.5 * 0.5 + 0.3 * 0.25)) < 1e-12
    assert (t.sum(1) - 1).abs().max() < 1e-12 and (t[1] == marg[1]).all()
    ld = thin_patterns(np.array([[2, 0], [1, 1]]), torch.tensor([0.5, 0.5], dtype=torch.float64), 0.5)
    cl, p = ld.clicks()
    assert abs(float(p.sum()) - 1) < 1e-12 and cl.shape[1] == 2 and set(map(tuple, cl.tolist())) <= {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_thin_patterns_mixed_photon_numbers():
    occ = np.array([[2, 0, 0], [1, 0, 0], [0, 1, 1]])
    p = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64)
    ld = thin_patterns(occ, p, 0.5)
    d = ld.as_dict()
    assert abs(sum(d.values()) - 1) < 1e-12
    assert abs(d[(1, 0, 0)] - (0.5 * 2 * 0.25 + 0.2 * 0.5)) < 1e-12
    assert abs(d[(0, 0, 0)] - (0.5 * 0.25 + 0.2 * 0.5 + 0.3 * 0.25)) < 1e-12


# ---- samplers (input-side loss) -------------------------------------------------------------

@pytest.mark.parametrize("backend", ["torch", "segments"])
def test_emitter_sample_loss_matches_thinned_probs(backend):
    from photonic_engines.clifford import emitter_sample
    n, m, eta, shots = 4, 8, 0.7, 12000
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in (0.3, 1.1, 2.0)])
    U = haar_unitary(m, 3)
    ref = EmitterSLOSKernel(st, dtype=C128).probs(U, loss=eta).as_dict()
    occ = emitter_sample(st, U, shots, backend=backend, loss=eta, generator=torch.Generator().manual_seed(2))
    assert occ.shape == (shots, m) and occ.sum(1).max() <= n
    assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)
    ks = occ.sum(1).bincount(minlength=n + 1).double() / shots
    assert (ks - binomial_pmf(n, eta)).abs().max() < 0.03


def test_lossy_samplers_report_one_pass_per_surviving_count():
    """`stats["passes"]` of `emitter_sample(return_stats=True)` and `fock_sample(stats=)` under loss: one
    (k, streams, seconds) triple per surviving count k > 0 drawn, streams summing to the shots that kept a
    photon; the samples are the same with and without the record (torch backend / interpreter, slow for
    the Fock kernels: the emitter side here, the Fock side in the slow tier)."""
    from photonic_engines.clifford import emitter_sample
    n, m, eta, shots = 4, 8, 0.7, 400
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in (0.3, 1.1, 2.0)])
    U = haar_unitary(m, 3)
    occ, stats = emitter_sample(st, U, shots, backend="torch", loss=eta, generator=torch.Generator().manual_seed(2),
                                return_stats=True)
    occ2 = emitter_sample(st, U, shots, backend="torch", loss=eta, generator=torch.Generator().manual_seed(2))
    assert torch.equal(occ, occ2)
    ks = sorted(k for k, _, _ in stats["passes"])
    assert ks == sorted(set(occ.sum(1).tolist()) - {0}) and len(ks) == len(set(ks))
    assert sum(s for _, s, _ in stats["passes"]) == int((occ.sum(1) > 0).sum())
    assert all(t >= 0 for _, _, t in stats["passes"])


@pytest.mark.slow
def test_fock_sample_stats_passes_under_loss():
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, eta, shots = 4, 0.7, 300
    U = haar_unitary(m, 4).to(dev)
    st = {}
    s = fock_sample(U, [0, 1, 2], shots, generator=torch.Generator(device=dev).manual_seed(0), loss=eta, stats=st).cpu()
    s2 = fock_sample(U, [0, 1, 2], shots, generator=torch.Generator(device=dev).manual_seed(0), loss=eta).cpu()
    assert torch.equal(s, s2)
    ks = sorted(k for k, _, _ in st["passes"])                                # k = 1 is routed classically, no pass
    assert ks == sorted(set(s.sum(1).tolist()) - {0, 1})
    assert sum(c for _, c, _ in st["passes"]) == int((s.sum(1) > 1).sum())


def test_emitter_sample_keep_is_the_reduced_state():
    """Tracing out emission slots by `keep` equals the SLOS marginal on the kept photons."""
    from photonic_engines.clifford.emitter_sampler import sample_torch
    n, m, shots = 3, 6, 6000
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0), spin_gate(math.pi / 2, 0.5), spin_gate(0.8, 1.9)])
    U = haar_unitary(m, 9)
    keep = torch.tensor([[0, 2]]).expand(shots, 2)
    occ = sample_torch(st, U, shots, generator=torch.Generator().manual_seed(0), keep=keep)
    assert (occ.sum(1) == 2).all()
    # reference: the reduced state on slots {0, 2} as its pure-term decomposition, each term
    # through the SLOS emitter kernel
    from photonic_engines.clifford.emitter_reference import gap_kraus_terms
    terms = [t for fv in torch.eye(2, dtype=C128) for t in gap_kraus_terms(st, [0, 2], final=fv)[1]]
    ref = {}
    for s0, mats, f in terms:
        red = EmitterMPS(ops=mats.numpy(), modes=(st.modes[0], st.modes[2]), s0=s0.numpy(), m=m, final=f.numpy())
        p = EmitterSLOSKernel(red, dtype=C128).probs(U)
        for t, q in zip(EmitterSLOSKernel(red, dtype=C128).layers.final_states, p.tolist()):
            ref[t] = ref.get(t, 0.0) + q
    assert abs(sum(ref.values()) - 1) < 1e-9
    assert _tvd(_empirical(occ), ref) < 2.5 * _noise_floor(ref, shots)


@pytest.mark.slow
def test_fock_sample_loss_matches_perceval():
    """The Triton kernels under the CPU interpreter (slow); the same check runs on a GPU in
    seconds.

    U and the generator follow the device: with a real CUDA device the Triton kernels want device
    tensors (a cpu one gives "Pointer argument cannot be accessed from Triton"), and without one
    they run under the interpreter on cpu tensors. Before 2026-09-19 this only ever ran the second
    way, because `--runslow` had not been run on a CUDA box.
    """
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, occ, eta, shots = 4, (1, 1, 1, 0), 0.7, 3000
    U = haar_unitary(m, 4)
    ref = _perceval_lossy(U.numpy(), occ, eta)
    s = fock_sample(U.to(dev), [0, 1, 2], shots,
                    generator=torch.Generator(device=dev).manual_seed(0), loss=eta).cpu()
    assert s.shape == (shots, m) and s.sum(1).max() <= 3
    assert _tvd(_empirical(s), ref) < 2.5 * _noise_floor(ref, shots)


# ---- time loop --------------------------------------------------------------------------------

def test_timeloop_uniform_loss():
    from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
    c = TimeLoopCircuit(T=5, loops=[1, 2])
    k = LoopRegisterKernel(c, [1, 1, 0, 0, 0], dtype=C128)
    g = torch.Generator().manual_seed(0)
    th, ph = torch.rand(2, 5, generator=g, dtype=torch.float64) * 3, torch.rand(2, 5, generator=g, dtype=torch.float64) * 6
    m_uni = k.bin_marginals(th, ph, loss=0.8)
    m_det = k.bin_marginals(th, ph, loss={"detector": 0.8})                  # the Kraus route
    assert (m_uni - m_det).abs().max() < 1e-12
    assert (k.bin_marginals(th, ph, loss={"uniform": 0.8}) - m_uni).abs().max() < 1e-12
    # uniform folded into the detector when loop channels are present
    m_a = k.bin_marginals(th, ph, loss={"loop": [0.9, 0.95], "detector": 0.8})
    m_b = k.bin_marginals(th, ph, loss={"loop": [0.9, 0.95], "uniform": 0.8})
    assert (m_a - m_b).abs().max() < 1e-12
    ld = k.distribution(th, ph, loss=0.8)
    occ, p = ld.flat()
    mb = torch.zeros(3, dtype=p.dtype)
    for o, q in zip(occ, p):
        mb[int(o[0])] += q
    assert (mb - m_uni[0]).abs().max() < 1e-12 and abs(float(p.sum()) - 1) < 1e-12
    s = k.sample(4000, th, ph, loss=0.8, generator=torch.Generator().manual_seed(1))
    mean = m_uni @ torch.arange(3, dtype=m_uni.dtype)
    assert (s.double().mean(0) - mean).abs().max() < 0.05
    assert k.sample(50, th, ph, loss=0.8, threshold=True).max() <= 1


# ---- adapters ---------------------------------------------------------------------------------

def test_perceval_backend_loss():
    pcvl = pytest.importorskip("perceval")
    from photonic_engines.adapters.perceval.backends import SLOSTorchBackend
    m, occ, eta = 4, (1, 1, 1, 0), 0.7
    U = haar_unitary(m, 8)
    ref = _perceval_lossy(U.numpy(), occ, eta)
    b = SLOSTorchBackend(dtype=C128, loss=eta)
    b.set_circuit(pcvl.Unitary(pcvl.Matrix(U.numpy())))
    b.set_input_state(pcvl.BasicState(list(occ)))
    d = b.prob_distribution()
    assert max(abs(v - ref.get(tuple(k), 0.0)) for k, v in d.items()) < 1e-9
    assert abs(b.probability(pcvl.BasicState([1, 0, 0, 0])) - ref[(1, 0, 0, 0)]) < 1e-9
    assert len(b.samples(5)) == 5
    with pytest.raises(NotImplementedError):
        b.all_prob()


def test_lost_label_unravelling():
    """docs/LOSS.md §3b: the lost slots' labels are drawn from their exact joint marginal (kept
    slots and final spin traced), every gap then carries one term, and the lossy sample
    distribution is unchanged (unravel=True vs the traced gap terms)."""
    from photonic_engines.clifford import emitter_sample
    from photonic_engines.clifford.emitter_reference import coefficient_tensor
    from photonic_engines.clifford.emitter_terms import batched_gap_terms, sample_lost_labels
    n = 5
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(0.7 * i + 0.3, 1.1 * i) for i in range(1, n)], m=2 * n)
    N = 100000
    keep = torch.tensor([[1, 3]]).expand(N, 2)
    labels, fid = sample_lost_labels(st, keep, generator=torch.Generator().manual_seed(0))
    assert (labels[:, [1, 3]] == -1).all() and (labels[:, [0, 2, 4]] >= 0).all()
    c = coefficient_tensor(st)                                                      # [2]*n + [chi]
    p = (c.abs() ** 2).sum(dim=(1, 3)).reshape(-1)                                  # joint of (x0, x2, x4, s_f)
    p = p / p.sum()
    key = labels[:, 0] * 8 + labels[:, 2] * 4 + labels[:, 4] * 2 + fid
    emp = torch.bincount(key, minlength=16).double() / N
    assert float((emp - p).abs().max()) < 4 * float((p * (1 - p) / N).sqrt().max())
    S = 6
    finals = torch.eye(2, dtype=C128)[fid[:S]]
    _, _, _, t2s = batched_gap_terms(st, keep[:S], finals, labels=labels[:S])
    assert torch.bincount(t2s, minlength=S).tolist() == [1] * S
    _, _, _, t2s = batched_gap_terms(st, keep[:S], finals)
    assert int(torch.bincount(t2s, minlength=S).min()) > 1
    # partial chains (pending kept slots traced, lost slots at their labels): the batched terms
    # reproduce the reference decomposition's probabilities (the bug of run 20260918_160033 masked
    # pending kept slots to label 0)
    import itertools

    from photonic_engines.clifford.emitter_reference import gap_kraus_terms
    for T in ([1], [3], [1, 3]):
        Tt = torch.tensor([T] * S)
        s0, mats, f, t2s = batched_gap_terms(st, Tt, finals, labels=labels[:S])
        for s_ in range(S):
            _, terms = gap_kraus_terms(st, T, final=finals[s_], labels=labels[s_].tolist())

            def amps(s0v, m, fv):
                out = []
                for x in itertools.product(range(2), repeat=len(T)):
                    v = s0v.clone()
                    for j, l in enumerate(x):
                        v = m[j, l] @ v
                    out.append(fv.conj() @ v)
                return torch.stack(out)
            p_ref = sum(amps(*t).abs() ** 2 for t in terms)
            idx = (t2s == s_).nonzero()[:, 0]
            p_bat = sum(amps(s0[i], mats[i], f[i]).abs() ** 2 for i in idx)
            assert torch.allclose(p_ref, p_bat, atol=1e-12)
    n, m, eta, shots = 4, 8, 0.6, 12000
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, q) for q in (0.3, 1.1, 2.0)])
    U = haar_unitary(m, 3)
    ref = EmitterSLOSKernel(st, dtype=C128).probs(U, loss=eta).as_dict()
    for unravel in (True, False):
        occ = emitter_sample(st, U, shots, backend="torch", loss=eta, unravel=unravel, generator=torch.Generator().manual_seed(2))
        assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)
