"""The CPU examples of docs/TUTORIAL.md, run as written (keeps the tutorial in sync with the API)."""
import math

import pytest

torch = pytest.importorskip("torch")

from photonic_engines import EmitterMPS, FockState, FockSuperposition, Mask, spin_gate, state_space_size
from photonic_engines.slos import EmitterSLOSKernel, FockSLOSKernel, SuperpositionSLOSKernel, reference_probs, sample_from_probs


def haar(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=g))[0]


def tvd_check(layers, p, occ):
    idx = {s: i for i, s in enumerate(layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= occ.shape[0]
    tvd = 0.5 * float((emp - p / p.sum()).abs().sum())
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / occ.shape[0]).sum()) * math.sqrt(2 / math.pi)
    return tvd, floor


def test_section_2_states():
    fock = FockState((1, 0, 1, 0, 1, 0))
    assert (fock.n, fock.m, fock.modes()) == (3, 6, [0, 2, 4])
    FockSuperposition(((1, 0, 1, 0), (0, 1, 0, 1)), (1 / math.sqrt(2), 1j / math.sqrt(2)))
    assert state_space_size(6, 12, Mask.FOCK) == 12376
    assert state_space_size(6, 12, Mask.UNBUNCHED) == 924
    assert state_space_size(6, 12, Mask.DUAL_RAIL) == 64
    prep = spin_gate(math.pi / 2, 0)
    n = 4
    lc = EmitterMPS.caterpillar(n, [prep] + [spin_gate(math.pi / 2, 0)] * (n - 1))
    EmitterMPS.caterpillar(n, [prep] + [spin_gate(math.pi / 2, math.pi)] * (n - 1))
    assert (lc.n, lc.m, lc.chi, lc.num_labels) == (4, 8, 2, 2)
    assert lc.modes == ((0, 1), (2, 3), (4, 5), (6, 7))
    EmitterMPS.caterpillar(n, [prep] * n, final=(1, 0))
    assert len(lc.expand()) == 2


def test_section_3_slos():
    U = haar(8)
    k = FockSLOSKernel(n=3, m=8, input_occupation=[1, 0, 1, 0, 1, 0, 0, 0], mask="fock",
                   device="cpu", dtype=torch.complex128)
    p = k.probs(U)
    assert p.shape == (120,) and abs(float(p.sum()) - 1) < 1e-10
    states = k.layers.final_states
    best = max(range(len(p)), key=lambda i: p[i])
    assert sum(states[best]) == 3
    assert torch.allclose(p, reference_probs(U, 3, 8, [1, 0, 1, 0, 1, 0, 0, 0]), atol=1e-10)
    Ub = torch.stack([haar(8, s) for s in range(4)])
    assert k.probs(Ub).shape == (4, 120)
    Ug = U.clone().requires_grad_(True)
    loss = -k.probs(Ug)[best].log()
    loss.backward()
    assert Ug.grad.shape == (8, 8)
    amp = k.amplitudes(U) * torch.sqrt(torch.as_tensor(k.layers.output_prob_norms()))
    assert torch.allclose(amp.abs() ** 2, p, atol=1e-10)
    occ = sample_from_probs(k.layers, p, shots=1000, generator=torch.Generator().manual_seed(0))
    assert occ.shape == (1000, 8) and occ.sum(1).unique().tolist() == [3]


def test_section_4_multi_input():
    mk = SuperpositionSLOSKernel(n=2, m=4, input_occupations=[[1, 0, 1, 0], [0, 1, 0, 1]],
                              device="cpu", dtype=torch.complex128)
    U4 = haar(4)
    w = [1 / math.sqrt(2), 1j / math.sqrt(2)]
    amp = mk.amplitudes(U4, weights=w)
    p = mk.probs(U4, weights=w)
    assert amp.shape == (10,) and abs(float(p.sum()) - 1) < 1e-10
    sup = FockSuperposition(((1, 0, 1, 0), (0, 1, 0, 1)), (1 / math.sqrt(2), 1j / math.sqrt(2)))
    assert torch.allclose(SuperpositionSLOSKernel.from_state(sup, dtype=torch.complex128).probs(U4), p)


def test_section_4b_superposition_boson_sampling():
    from photonic_engines import ProductSuperposition, graded_state_space_size
    from photonic_engines.slos import ProductSLOSKernel, product_amplitude
    sbs = ProductSuperposition.sbs(sources=4, m=8)
    assert (sbs.n, sbs.d, sbs.sources) == (4, 2, 4)
    assert graded_state_space_size(sbs.n, sbs.m) == math.comb(12, 4) == 495
    U8 = haar(8)
    pk = ProductSLOSKernel(sbs, device="cpu", dtype=torch.complex128)
    dist = pk.probs(U8)
    assert abs(float(dist.mass(2)) - 0.375) < 1e-12
    assert abs(float(dist.conditional(2).sum()) - 1) < 1e-12
    occ = pk.sample(U8, 100, generator=torch.Generator().manual_seed(0))
    assert occ.shape == (100, 8)
    assert abs(float(sum(b.sum() for b in pk.probs(U8, loss=0.8).probs)) - 1) < 1e-12
    amp = product_amplitude(sbs, U8, [1, 1, 0, 0, 0, 0, 0, 0])
    assert amp.shape == ()
    from photonic_engines import clifford
    g = torch.Generator().manual_seed(2)
    assert clifford.sample(sbs, U8, 5, generator=g).shape == (5, 8)
    assert (clifford.sample(sbs, U8, 5, generator=g, photons=2).sum(1) == 2).all()


def test_section_4_perceval_round_trip():
    """The `to_perceval` / `from_perceval` line of tutorial S4; needs the optional perceval dependency."""
    pytest.importorskip("perceval")
    sup = FockSuperposition(((1, 0, 1, 0), (0, 1, 0, 1)), (1 / math.sqrt(2), 1j / math.sqrt(2)))
    assert FockSuperposition.from_perceval(sup.to_perceval())[0].K == 2


def test_section_5_emitter_and_section_7_tvd():
    prep = spin_gate(math.pi / 2, 0)
    n = 4
    lc = EmitterMPS.caterpillar(n, [prep] + [spin_gate(math.pi / 2, 0)] * (n - 1))
    proj = EmitterMPS.caterpillar(n, [prep] * n, final=(1, 0))
    U8 = haar(8)
    ek = EmitterSLOSKernel(lc, device="cpu", dtype=torch.complex128)
    p = ek.probs(U8)
    assert p.shape == (330,) and abs(float(p.sum()) - 1) < 1e-10
    assert ek.bond_amplitudes(U8).shape == (2, 330)
    ekp = EmitterSLOSKernel(proj, device="cpu", dtype=torch.complex128)
    amp = ekp.amplitudes(U8)
    pp = ekp.probs(U8)
    assert amp.shape == (330,) and 0 < float(pp.sum()) <= 1 + 1e-10
    K = ek.kraus(U8)
    K = K * torch.sqrt(torch.as_tensor(ek.layers.output_prob_norms(), dtype=torch.float64)).view(-1, 1, 1)
    assert torch.allclose(torch.einsum("tji,tjk->ik", K.conj(), K), torch.eye(2, dtype=K.dtype), atol=1e-8)
    comps = lc.expand()
    mk2 = SuperpositionSLOSKernel(lc.n, lc.m, [c.occupation for c in comps[0].components], dtype=torch.complex128)
    p_up = mk2.probs(U8, weights=comps[0].amplitudes)
    assert torch.allclose(p_up, (ek.bond_amplitudes(U8)[0].abs() ** 2) * torch.as_tensor(ek.layers.output_prob_norms()), atol=1e-10)
    # section 5b (additive-error estimator)
    from photonic_engines.slos.estimate import EmitterGurvitsEstimator
    est = EmitterGurvitsEstimator(lc, sampling="gaussian", seed=0)
    t = ek.layers.final_states[5]
    e = est.probability(U8, t, shots=20_000)
    assert abs(e.value - p[5].item()) < 5 * e.stderr + 1e-9
    estp = EmitterGurvitsEstimator(proj, seed=0)
    a = estp.amplitude(U8, t, shots=20_000)
    tfact = math.prod(math.factorial(int(c)) for c in t)
    assert abs(a.value - amp[5].item() * math.sqrt(tfact)) < 5 * a.stderr + 1e-9
    # section 6 (CPU sampler) + section 7
    from photonic_engines.clifford import emitter_sample, sample
    occ = emitter_sample(lc, U8, shots=200, generator=torch.Generator().manual_seed(0), backend="torch")
    occ2 = sample(lc, U8, 200, generator=torch.Generator().manual_seed(0), backend="torch")
    assert torch.equal(occ, occ2)
    assert occ.shape == (200, 8)
    tvd, floor = tvd_check(ek.layers, p, occ)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_section_9b_timeloop():
    from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit, register_dimension
    T, loops = 8, (1, 2)
    c = TimeLoopCircuit(T, loops)
    theta = torch.rand(c.n_loops, T, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    phi = torch.zeros(c.n_loops, T, dtype=torch.float64)
    theta, phi = c.flush_settings(theta, phi)
    U = c.unitary(theta, phi)
    inp = [1, 1, 1, 0, 0, 0, 0, 0]
    lk = LoopRegisterKernel(c, inp)
    assert lk.D == register_dimension(3, 3) == 35
    pats, probs = lk.distribution(theta, phi)
    assert abs(float(probs.sum()) - 1) < 1e-12 and (pats.sum(1) == 3).all()
    p = lk.probabilities([[1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0]], theta, phi)
    assert p.shape == (2,)
    marg = lk.bin_marginals(theta, phi, loss={"loop": [0.9, 0.95], "detector": 0.8})
    assert marg.shape == (T, 4) and torch.allclose(marg.sum(1), torch.ones(T, dtype=torch.float64))
    samples = lk.sample(1000, theta, phi, generator=torch.Generator().manual_seed(0))
    assert samples.shape == (1000, T) and (samples.sum(1) == 3).all()
    k = FockSLOSKernel(3, c.m, inp + [0] * c.L, dtype=torch.complex128)
    pr = k.probs(U)
    occ = k.layers.final_occupations()
    ref = {}
    for o, q in zip(occ, pr.tolist()):
        ref[tuple(int(x) for x in o[:T])] = ref.get(tuple(int(x) for x in o[:T]), 0.0) + q
    got = {tuple(int(x) for x in a): float(b) for a, b in zip(pats, probs)}
    assert max(abs(got.get(kk, 0.0) - v) for kk, v in ref.items()) < 1e-12
    # sector sampler, superposed sources (docs/TUTORIAL.md §9b, HYBRID.md)
    from photonic_engines import ProductSuperposition
    from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel, light_loop_kernel, loop_sector_sample, sector_sample
    s = loop_sector_sample(lk, 1000, theta, phi, generator=torch.Generator().manual_seed(0))
    assert s.shape == (1000, T) and (s.sum(1) == 3).all()
    big = TimeLoopCircuit(400, (4,))
    lk_big = light_loop_kernel(big, [1] * 200 + [0] * 200)
    s_big, stats = loop_sector_sample(lk_big, 4, torch.rand(1, 400, dtype=torch.float64), return_stats=True)
    assert s_big.shape == (4, 400) and stats["max_amplitudes"] < 10_000
    _, logp, _ = sector_sample(lk, 2, lk._gates(theta, phi), force=pats[:2].reshape(2, T, 1))
    assert torch.allclose(torch.exp(logp), probs[:2], atol=1e-12)
    sl = SuperpositionLoopKernel(c, ProductSuperposition.sbs(3, c.m, modes=(0, 1, 2)))
    assert sl.bin_marginals(theta, phi).shape == (T, 4)


def test_section_9d_losses():
    from photonic_engines.clifford import emitter_sample
    from photonic_engines.imperfections import LossyDistribution
    U6 = haar(6, 3)
    k = FockSLOSKernel(3, 6, [1, 1, 1, 0, 0, 0], dtype=torch.complex128)
    ld = k.probs(U6, loss=0.8)
    assert isinstance(ld, LossyDistribution)
    assert abs(float(ld.mass(2)) - 3 * 0.8 ** 2 * 0.2) < 1e-12
    assert ld.probs[2].shape[0] == ld.occupations(2).shape[0] == 21
    assert torch.allclose(ld.conditional(3), k.probs(U6))
    d, (cl, pc) = ld.as_dict(), ld.clicks()
    assert abs(sum(d.values()) - 1) < 1e-12 and abs(float(pc.sum()) - 1) < 1e-12 and cl.max() == 1
    occ = sample_from_probs(k.layers, ld, shots=1000)
    assert occ.shape == (1000, 6) and occ.sum(1).max() <= 3
    U8 = haar(8, 4)
    lc = EmitterMPS.caterpillar(4, [spin_gate(math.pi / 2, 0)] * 4)
    occ = emitter_sample(lc, U8, 200, backend="torch", loss=0.8)
    assert occ.shape == (200, 8) and occ.sum(1).max() <= 4
    # partial distinguishability (the fock_sample lines of the tutorial need triton: tests/test_imperfections.py)
    from photonic_engines.imperfections import Distinguishability
    occ = emitter_sample(lc, U8, 200, backend="torch", loss=0.8, indist=0.95)
    assert occ.shape == (200, 8) and occ.sum(1).max() <= 4
    occ = emitter_sample(lc, U8, 200, backend="torch", indist=0.0)
    assert occ.shape == (200, 8) and (occ.sum(1) == 4).all()
    assert abs(Distinguishability.from_hom_visibility(0.9).x - math.sqrt(0.9)) < 1e-15
    Distinguishability(species=(0.7, 0.2))
    from photonic_engines.imperfections import truncation_bound
    p = k.probs(U6, indist=0.9)
    assert abs(float(p.sum()) - 1) < 1e-12
    p2 = k.probs(U6, indist=0.9, order=1)
    assert abs(1 - float(p2.sum()) - truncation_bound(3, 0.9, 1)) < 1e-12
    ld = k.probs(U6, indist=0.9, loss=0.8)
    assert isinstance(ld, LossyDistribution)
    pe = EmitterSLOSKernel(lc, dtype=torch.complex128).probs(U8, indist=0.9)
    assert abs(float(pe.sum()) - 1) < 1e-12
    K, coarse = EmitterSLOSKernel(lc, dtype=torch.complex128).kraus_species(U8, 0.9)
    assert K.shape[0] == coarse.shape[0] and K.shape[1:] == (2, 2)
    from photonic_engines.sequence import BinEmitter, SequenceKernel
    sk = SequenceKernel([BinEmitter.caterpillar(2, [spin_gate(math.pi / 2, 0), spin_gate(1.0, 0.5)], (2 * a, 2 * a + 1)) for a in range(2)],
                        4, indist=[0.9, 0.8])
    assert abs(float(sk.bin_marginals(haar(4, 1)).sum(1).min()) - 1) < 1e-12 and sk.sample(haar(4, 1), 3).shape == (3, 2, 4)
    from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
    c6 =TimeLoopCircuit(6, (1,))
    lk = LoopRegisterKernel(c6, [1, 1, 0, 0, 0, 0])
    th = torch.rand(1, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert lk.sample(5, th, torch.zeros(1, 6, dtype=torch.float64), indist=0.95).shape == (5, 6)
    assert abs(float(lk.bin_marginals(th, torch.zeros(1, 6, dtype=torch.float64), indist=0.95).sum(1).min()) - 1) < 1e-12
    from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
    c = TimeLoopCircuit(6, (1,))
    kernel = LoopRegisterKernel(c, [1, 1, 0, 0, 0, 0])
    theta = torch.rand(1, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    phi = torch.zeros(1, 6, dtype=torch.float64)
    m1 = kernel.bin_marginals(theta, phi, loss=0.8)
    m2 = kernel.bin_marginals(theta, phi, loss={"loop": [0.9], "uniform": 0.8})
    assert m1.shape == m2.shape == (6, 3)
    eta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    k.probs(U6, loss=eta).mass(1).backward()
    assert eta.grad is not None


def test_section_9e_sequence():
    from photonic_engines.sequence import BinEmitter, SequenceKernel
    k, n, m = 3, 4, 6
    U6 = haar(6, 9)
    ems = [BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, modes=(2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m)
    assert sk.chi == 8 and sk.S == 56
    K = sk.kraus(U6, 0)
    assert K.shape == (56, 8, 8)
    marg = sk.bin_marginals(U6)
    assert marg.shape == (n, 56) and torch.allclose(marg.sum(1), torch.ones(n, dtype=torch.float64))
    occ = sk.sample(U6, 200)
    assert occ.shape == (200, n, m) and (occ.sum(2) == k).all()
    p = sk.sequence_probability(U6, occ[:5].numpy())
    assert p.shape == (5,) and (p > 0).all()
    J = sk.joint(U6, [0, 3])
    assert J.shape == (56, 56) and abs(float(J.sum()) - 1) < 1e-12
    assert sk.sample(U6, 50, loss=0.8).sum(2).max() <= k
    assert len(sk.bin_marginals(U6, loss=0.8)) == n
    assert (SequenceKernel(ems, m).bin_marginals(torch.stack([U6] * n)) - marg).abs().max() < 1e-12


def test_section_9c_distributed():
    pytest.importorskip("mpi4py")
    pytest.importorskip("cffi")
    pcvl = pytest.importorskip("perceval")
    import shutil
    if shutil.which("mpicc") is None or shutil.which("mpirun") is None:
        pytest.skip("MPI not installed")
    from photonic_engines.adapters.perceval import DistributedSLOSBackend, DistributedSLOSSampler
    from photonic_engines.slos.distributed import plan
    assert plan(13, 26, 48).per_rank_bytes < 3 * 2 ** 30
    circuit = pcvl.Unitary(pcvl.Matrix.random_unitary(6))
    occ = [1, 1, 1, 0, 0, 0]
    strong = DistributedSLOSBackend(workers=2)
    try:
        p = pcvl.Processor(strong, circuit)
        p.with_input(pcvl.BasicState(occ))
        p.min_detected_photons_filter(0)
        dist = p.probs()["results"]
        assert abs(sum(dist.values()) - 1) < 1e-12
        s = pcvl.Processor(DistributedSLOSSampler(strong=strong, seed=0), circuit)
        s.with_input(pcvl.BasicState(occ))
        samples = pcvl.algorithm.Sampler(s).samples(100)["results"]
        assert len(samples) == 100
    finally:
        strong.close()


def test_section_9f_nonlinear():
    from photonic_engines.nonlinear import NonlinearCircuit, branches
    U0, U1 = haar(6, 1), haar(6, 2)
    C = NonlinearCircuit([U0, U1], [0, 1, 2], [(1, 3, math.pi / 2)])   # Kerr pi/2 on mode 3, between U0 and U1
    theta, c = branches(math.pi / 2, 3)       # 2 phase shifters reproduce exp(-i phi j^2) on j = 0..3
    p = C.probs()                             # [S] over C.occupations(); route "branch" | "gate" | "auto"
    occ = C.sample(2000, generator=torch.Generator().manual_seed(0), method="strong")     # small n
    C.rejection_factor()                      # 2.0 proposals per sample of the exact sampler (method="rejection")
    assert len(theta) == 2 and abs(float(p.sum()) - 1) < 1e-12
    assert p.shape == (C.occupations().shape[0],) and occ.shape == (2000, 6)
    assert C.rejection_factor() == pytest.approx(2.0, abs=1e-12)
    tvd, floor = tvd_check(C._slos().layers, p, occ)
    assert tvd < 2.5 * floor


def test_section_9g_adaptive():
    from photonic_engines.adaptive import AdaptiveCircuit, mode_marginals
    U0 = haar(6, 1)
    policy = lambda lam, p: haar(5, 10 + p[0])     # V_1 on the 5 unmeasured modes, picked by the count on mode 1
    C = AdaptiveCircuit(U0, [0, 2, 4], [[1]], policy)   # 3 photons; one level: count mode 1, then feed forward
    p = C.probs()                                  # [S] joint law of the full occupation, over C.occupations()
    q = C.outcome_probs()                          # {(count on mode 1,): probability}
    mm = mode_marginals(U0, [0, 2, 4], [1, 3])     # [n+1, n+1] exact photon-count law of modes 1 and 3 behind U0
    occ = C.sample(2000, generator=torch.Generator().manual_seed(0), final="slos")   # small n, CPU
    occs = C.occupations()
    assert p.shape == (occs.shape[0],) and abs(float(p.sum()) - 1) < 1e-12
    assert mm.shape == (4, 4) and abs(float(mm.sum()) - 1) < 1e-12
    for (c,), w in q.items():                      # the outcome law is the joint summed over the other modes
        assert float(w) == pytest.approx(float(p[occs[:, 1] == c].sum()), abs=1e-12)
    assert torch.allclose(mm.sum(1), torch.stack([p[occs[:, 1] == c].sum() for c in range(4)]).to(mm.dtype), atol=1e-12)
    assert occ.shape == (2000, 6) and bool((occ.sum(1) == 3).all())
    idx = {tuple(o): i for i, o in enumerate(occs.tolist())}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= occ.shape[0]
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / occ.shape[0]).sum()) * math.sqrt(2 / math.pi)
    assert 0.5 * float((emp - p).abs().sum()) < 2.5 * floor


def test_section_9h_gaussian():
    from photonic_engines import GaussianInput, gaussian
    st = GaussianInput.smsv(3, 6, r=GaussianInput.r_for_mean(0.5))   # 3 squeezers, 0.5 photons each on average
    U = haar(6, 1)
    occ = gaussian.sample(st, U, 500, seed=0)                  # [500, 6] int64 counts; CPU (device="cuda" for the hafnians)
    lossy = gaussian.sample(st, U, 500, loss=0.7, seed=1)      # exact: pure squeezing + classical noise per shot
    impure = gaussian.sample(st, U, 500, schmidt=(0.9, 0.1), seed=2)   # two Schmidt modes per squeezer
    p = gaussian.pure_probability(st, U, (1, 1, 0, 0, 0, 0))   # one loop hafnian (lossless input)
    law = st.photon_number_probs(20)                           # exact law of the total photon number, U-independent
    assert occ.shape == lossy.shape == impure.shape == (500, 6)
    assert bool((occ.sum(1) % 2 == 0).all())                  # lossless squeezed vacua: even totals
    assert 0 < p < 1 and 0.99 < law.sum() < 1           # the tail above 20 photons is ~2e-3 here
    # mean photon number within 5 sigma of 1.5 (sampled N has variance sum 2 sinh^2 cosh^2 = 3 x 1.5)
    assert abs(occ.sum(1).mean() - 1.5) < 5 * math.sqrt(4.5 / 500)


def test_section_6_algorithm_variants():
    pytest.importorskip("triton")       # clifford.variants imports cofactors_triton at module level (no-optional-deps job)
    from photonic_engines.clifford.variants import fock_sample_reference, strong_then_sample
    U6, U8 = haar(6, 3), haar(8, 1)
    stats = {}
    occ = fock_sample_reference(U6, [0, 1, 2], 50, algo="input", stats=stats, generator=torch.Generator().manual_seed(0))
    assert occ.shape == (50, 6) and bool((occ.sum(1) == 3).all()) and len(stats["addends"]) == 50
    st = {}
    occ = strong_then_sample(U8, [0, 1, 2, 3], 4000, stats=st, generator=torch.Generator().manual_seed(1))
    assert occ.shape == (4000, 8) and st["states"] == 330 and st["build_s"] > 0
    k = FockSLOSKernel(n=4, m=8, input_occupation=[1, 1, 1, 1, 0, 0, 0, 0], mask="fock", dtype=torch.complex128)
    p = k.probs(U8)
    tvd, floor = tvd_check(k.layers, p, occ)
    assert tvd < 2.5 * floor


def test_section_9e_sample_diag():
    from photonic_engines.sequence import BinEmitter, SequenceKernel
    U6 = haar(6, 3)
    k, n, m = 3, 8, 6
    ems = [BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, modes=(2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m)
    occ = sk.sample_diag(U6, 200, generator=torch.Generator().manual_seed(0))
    assert occ.shape == (200, n, m) and bool((occ.sum(2) == k).all())
    occ2 = sk.sample_diag(U6, 200, generator=torch.Generator().manual_seed(0), policy=lambda b, counts, psi: None)
    assert torch.equal(occ, occ2)                  # a policy returning None changes nothing


def test_section_9h_transfer_and_precision():
    from photonic_engines import GaussianInput, gaussian
    st = GaussianInput.smsv(3, 6, r=GaussianInput.r_for_mean(0.5))
    T = 0.8 * haar(6, 1)
    occ = gaussian.sample_transfer(st, T, 200, seed=0)
    dec = gaussian.transfer_decomposition(st, T)
    occ2 = gaussian.sample_transfer(st, T, 200, seed=0, decomposition=dec)
    assert tuple(occ.shape) == (200, 6) and (occ == occ2).all()
    occ = gaussian.sample(st, haar(6, 1), 50, seed=0, precision="dd")
    assert tuple(occ.shape) == (50, 6)
