"""Carried-photon engine (spins (x) Fock register, docs/HYBRID.md) vs its limits and the generic oracle:
EmitterSLOSKernel on the unrolled unitary of the T ticks fed with the product emitter MPS."""
import itertools
import math

import numpy as np
import pytest
import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
from photonic_engines.timeloop.hybrid import HybridRegisterKernel, hybrid_register_dimension, mesh_decompose, mesh_pairs
from tests.helpers import haar_unitary_np

TOL = 1e-12          # complex128 identities over a few hundred gate applications


def _G(th, ph):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -1j * np.exp(-1j * ph) * s], [-1j * np.exp(1j * ph) * s, c]])


def _emitters(k, T, rails, seed, final=None):
    rng = np.random.default_rng(seed)
    out = []
    for j in range(k):
        gates = [spin_gate(rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)) for _ in range(T)]
        out.append(BinEmitter.caterpillar(T, gates, modes=rails[j], final=final))
    return out


def _settings(n_dl, T, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_dl, T, generator=g, dtype=torch.float64) * 2, torch.rand(n_dl, T, generator=g, dtype=torch.float64) * 6


def _unrolled(K, th, ph, U):
    """Independent construction of the [T m + L, T m + L] unrolled unitary: tick t's rail r is wire t m + r, slot s of
    delay line dl is wire T m + offset_dl + s; per tick the loop gates on (rail wire, head slot) and the full U on the
    tick's rail wires, in the kernel's order."""
    T, m = K.T, K.m
    M = T * m + K.L
    W = np.eye(M, dtype=np.complex128)
    Us = np.asarray(U) if np.asarray(U).ndim == 3 else np.broadcast_to(np.asarray(U), (T, m, m))
    th, ph = np.asarray(th), np.asarray(ph)
    for t in range(T):
        def loops(W, t=t):
            for dl, ell in enumerate(K.delays):
                a = t * m + K.dl_rail[dl]
                b = T * m + int(K.offsets[dl]) + t % ell
                W[[a, b]] = _G(th[dl, t], ph[dl, t]) @ W[[a, b]]
            return W
        def mix(W, t=t):
            idx = np.arange(t * m, (t + 1) * m)
            W[idx] = Us[t] @ W[idx]
            return W
        W = mix(loops(W)) if K.mixer == "after" else loops(mix(W))
    return W


def _product_mps(emitters, T, m, M):
    """All k T emissions as one EmitterMPS on the product spin, tick-major, modes t m + rail (the generic input)."""
    seq = SequenceKernel(emitters, m)
    ops, modes = [], []
    for t in range(T):
        st = seq.bin_mps(t)
        ops.append(st.ops)
        modes += [tuple(t * m + r for r in mk) for mk in st.modes]
    fin = emitters[0].final
    final = None
    if fin is not None:
        final = np.ones(1, dtype=complex)
        for e in emitters:
            final = np.kron(final, e.final)
    return EmitterMPS(ops=np.concatenate(ops), modes=tuple(modes), s0=seq.s0.numpy(), m=M, final=final)


def _oracle(K, emitters, th, ph, U):
    W = _unrolled(K, th, ph, U)
    st = _product_mps(emitters, K.T, K.m, W.shape[0])
    ref_k = EmitterSLOSKernel(st, dtype=torch.complex128)
    pr = ref_k.probs(torch.as_tensor(W)).numpy()
    occ = ref_k.layers.final_occupations()
    ref = {}
    for o, q in zip(occ, pr):
        key = tuple(int(x) for x in o[:K.T * K.m])
        ref[key] = ref.get(key, 0.0) + q
    return ref


def _as_dict(pats, probs):
    return {tuple(int(x) for x in a.reshape(-1)): float(b) for a, b in zip(pats, probs)}


def _max_diff(a, b):
    return max(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in set(a) | set(b))


def _probs_on(K, ref, *args, chunk=128, n_max=400):
    """The kernel's probabilities on the oracle's patterns — all of them, or the n_max/2 largest + n_max/2 random
    ones (the projected path, in chunks; `distribution` branches every tick and is exponential in memory)."""
    keys = sorted(ref, key=ref.get, reverse=True)
    if len(keys) > n_max:
        rng = np.random.default_rng(0)
        rest = keys[n_max // 2:]
        keys = keys[:n_max // 2] + [rest[i] for i in rng.choice(len(rest), n_max // 2, replace=False)]
    pats = np.asarray(keys, dtype=np.int64).reshape(len(keys), K.T, K.m)
    out = torch.cat([K.probabilities(pats[i:i + chunk], *args) for i in range(0, len(keys), chunk)])
    return dict(zip(keys, out.tolist()))


def test_mesh_decompose():
    for m in (1, 2, 3, 4, 5):
        U = haar_unitary_np(m, np.random.default_rng(m))
        G = mesh_decompose(U)
        W = np.eye(m, dtype=complex)
        for (a, b), g in zip(mesh_pairs(m), G):
            W[[a, b]] = g @ W[[a, b]]
        if m == 1:
            continue                                     # a 1-mode U is a phase the mesh does not carry (see doc)
        assert np.abs(W - U).max() < 1e-13
    assert len(mesh_decompose(np.eye(4))) == len(mesh_pairs(4))           # fixed structure, also for U = I


CASES = [  # k, rails, label rails per emitter, ladders, T, mixer  (chi D <= ~1300: the density-matrix marginals)
    (1, 2, [(0, 1)], (1,), 3, "after"),
    (2, 4, [(0, 1), (2, 3)], [(1,), (), (1,), ()], 2, "after"),
    (2, 4, [(0, 1), (2, 3)], [(1,), (), (2,), ()], 2, "before"),
    (2, 4, [(0, 1), (2, 3)], [(1, 2), (), (), ()], 2, "after"),
    (1, 2, [(0, 1)], (2, 1), 4, "before"),
]


@pytest.mark.parametrize("k,m,rails,loops,T,mixer", CASES)
@pytest.mark.parametrize("final", [None, "proj"])
def test_hybrid_matches_unrolled_oracle(k, m, rails, loops, T, mixer, final):
    fin = None if final is None else [0.6, 0.8j]
    ems = _emitters(k, T, rails, seed=10 + k + T, final=fin)
    K = HybridRegisterKernel(ems, m, loops, mixer=mixer)
    th, ph = _settings(K.n_dl, T, 3)
    U = haar_unitary_np(m, np.random.default_rng(7))
    ref = _oracle(K, ems, th, ph, U)
    got = _probs_on(K, ref, th, ph, U)
    assert max(abs(v - ref[x]) for x, v in got.items()) < TOL
    if K.D * K.chi <= 200:                                                          # branching enumeration: tiny cases
        pats, probs = K.distribution(th, ph, U)
        assert _max_diff(_as_dict(pats, probs), ref) < TOL
    if final is None:                                                               # completeness + norm: the marginals
        marg = K.tick_marginals(th, ph, U).numpy()
        mref = np.zeros_like(marg)
        qid = {tuple(int(x) for x in r): i for i, r in enumerate(K.basis.rail_patterns)}
        for key, v in ref.items():
            for t in range(T):
                mref[t, qid[key[t * m:(t + 1) * m]]] += v
        assert np.abs(marg - mref).max() < TOL
        assert np.abs(marg.sum(1) - 1).max() < TOL


GEOM = [(1,), (), (), ()]                    # 2 emitters, 4 rails, one unit loop on rail 0: chi D = 4 * 462


def test_per_tick_unitaries():
    T, m = 3, 4
    ems = _emitters(2, T, [(0, 1), (2, 3)], seed=5)
    K = HybridRegisterKernel(ems, m, GEOM)
    th, ph = _settings(K.n_dl, T, 4)
    U = np.stack([haar_unitary_np(m, np.random.default_rng(20 + t)) for t in range(T)])
    ref = _oracle(K, ems, th, ph, U)
    assert max(abs(v - ref[x]) for x, v in _probs_on(K, ref, th, ph, U).items()) < TOL


def test_limit_no_loops_is_sequence_kernel():
    T, m = 3, 4
    ems = _emitters(2, T, [(0, 1), (2, 3)], seed=6)
    U = haar_unitary_np(m, np.random.default_rng(8))
    K = HybridRegisterKernel(ems, m, ())
    assert K.L == 0
    pats, probs = K.distribution(U=U)
    seq = SequenceKernel(ems, m)
    ref = seq.sequence_probability(torch.as_tensor(U), torch.as_tensor(np.asarray(pats)))
    assert torch.allclose(probs, ref.to(probs.dtype), atol=TOL)
    assert abs(float(probs.sum()) - 1) < TOL


@pytest.mark.parametrize("loops,T,inp", [((1,), 4, [1, 1, 0, 0]), ((1, 2), 6, [1, 1, 1, 0, 0, 0]), ((2, 1), 6, [2, 0, 1, 0, 0, 0])])
def test_limit_no_emitter_is_loop_register(loops, T, inp):
    c = TimeLoopCircuit(T, loops)
    g = torch.Generator().manual_seed(2)
    th = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 6
    ref = LoopRegisterKernel(c, inp)
    K = HybridRegisterKernel([], 1, loops, T=T, inputs=np.asarray(inp)[:, None])
    assert K.D == ref.D and K.chi == 1
    p1, q1 = ref.distribution(th, ph)
    p2, q2 = K.distribution(th, ph)
    assert _max_diff(_as_dict(p1, q1), _as_dict(p2, q2)) < TOL
    m1 = ref.bin_marginals(th, ph, loss={"loop": [0.8, 0.9][:len(loops)], "detector": 0.7})
    m2 = K.tick_marginals(th, ph, loss={"loop": [0.8, 0.9][:len(loops)], "detector": 0.7})
    assert torch.allclose(m1, m2, atol=TOL)


def test_limit_one_tick_is_the_stage():
    m = 4
    ems = _emitters(2, 1, [(0, 1), (2, 3)], seed=9)
    U = haar_unitary_np(m, np.random.default_rng(1))
    K = HybridRegisterKernel(ems, m, ())
    pats, probs = K.distribution(U=U)
    st = SequenceKernel(ems, m).bin_mps(0)
    ref_k = EmitterSLOSKernel(st, dtype=torch.complex128)
    pr = ref_k.probs(torch.as_tensor(U)).numpy()
    ref = {tuple(int(x) for x in o): float(q) for o, q in zip(ref_k.layers.final_occupations(), pr)}
    assert _max_diff(_as_dict(pats, probs), ref) < TOL


def test_sampler_vs_strong():
    T, m, S = 2, 4, 20000
    ems = _emitters(2, T, [(0, 1), (2, 3)], seed=11)
    K = HybridRegisterKernel(ems, m, GEOM)
    th, ph = _settings(K.n_dl, T, 12)
    U = haar_unitary_np(m, np.random.default_rng(13))
    marg = K.tick_marginals(th, ph, U).numpy()                                     # [T, Q]
    s = K.sample(S, th, ph, U, generator=torch.Generator().manual_seed(0))
    assert s.shape == (S, T, m)
    qid = {tuple(int(x) for x in r): i for i, r in enumerate(K.basis.rail_patterns)}
    emp = np.zeros_like(marg)
    for row in s.numpy():
        for t in range(T):
            emp[t, qid[tuple(int(x) for x in row[t])]] += 1.0 / S
    sigma = np.sqrt(marg * (1 - marg) / S)
    assert (np.abs(emp - marg) <= 5 * sigma + 1.0 / S).all()                       # 5 sigma per entry


def test_photon_cap_leak():
    """carry = p truncates the register: the exact leaked mass is 1 - trace, it vanishes at the exact cap, the
    truncated distribution's mass is 1 - leak, and the sampler's weights estimate the leak without bias."""
    T, m = 4, 2
    ems = _emitters(1, T, [(0, 1)], seed=14)
    th, ph = _settings(2, T, 15)
    th = th * 0 + 0.7                                      # keep photons circulating: a real leak
    U = haar_unitary_np(m, np.random.default_rng(16))
    exact = HybridRegisterKernel(ems, m, [(2,), (1,)])
    assert exact.carry is None and exact.leaked_mass(th, ph, U) == 0.0
    leaks = []
    for p in (1, 2, 3, 4):
        K = HybridRegisterKernel(ems, m, [(2,), (1,)], carry=p)
        leak = K.leaked_mass(th, ph, U)
        leaks.append(leak)
        _, probs = K.distribution(th, ph, U)
        assert abs(float(probs.sum()) - (1 - leak)) < TOL
        if p >= 2:
            S = 20000
            _, w = K.sample(S, th, ph, U, generator=torch.Generator().manual_seed(p), return_weights=True)
            est, se = 1 - float(w.mean()), float(w.std()) / math.sqrt(S)
            assert abs(est - leak) < 5 * se + 1e-12
    assert all(a >= b - TOL for a, b in itertools.pairwise(leaks)) and leaks[0] > 1e-3
    assert leaks[-1] < TOL                                                          # p = 4 = every photon: exact


def test_register_dimension():
    ems = _emitters(2, 3, [(0, 1), (2, 3)], seed=1)
    K = HybridRegisterKernel(ems, 4, (1, 2), carry=3)
    assert K.register_size() == hybrid_register_dimension(2, 3, 4, K.slots) == 4 * math.comb(3 + 4 + 12, 4 + 12)


def test_eta_one_is_lossless():
    T, m = 2, 4
    ems = _emitters(2, T, [(0, 1), (2, 3)], seed=17)
    K = HybridRegisterKernel(ems, m, GEOM)
    th, ph = _settings(K.n_dl, T, 18)
    U = haar_unitary_np(m, np.random.default_rng(19))
    a = K.tick_marginals(th, ph, U)
    b = K.tick_marginals(th, ph, U, loss={"loop": 1.0, "detector": 1.0})
    assert torch.equal(a, b)
    s1 = K.sample(64, th, ph, U, generator=torch.Generator().manual_seed(3))
    s2 = K.sample(64, th, ph, U, generator=torch.Generator().manual_seed(3), loss={"loop": 1.0, "detector": 1.0})
    assert torch.equal(s1, s2)


def test_lossy_marginals_vs_unrolled_with_environment():
    """Round-trip and detector damping as Kraus maps on the register vs the unrolled network with one environment
    wire per loss event (a beam splitter of transmission eta), environment and slots marginalised."""
    T, m, eta_l, eta_d = 3, 2, 0.8, 0.7
    ems = _emitters(1, T, [(0, 1)], seed=21)
    K = HybridRegisterKernel(ems, m, [(1,), (2,)])
    th, ph = _settings(K.n_dl, T, 22)
    U = haar_unitary_np(m, np.random.default_rng(23))
    base = T * m + K.L
    n_env = T * (K.n_dl + m)
    W = np.eye(base + n_env, dtype=np.complex128)
    env = iter(range(base, base + n_env))

    def damp(W, a, eta):
        e = next(env)
        B = np.array([[math.sqrt(eta), -math.sqrt(1 - eta)], [math.sqrt(1 - eta), math.sqrt(eta)]])
        W[[a, e]] = B @ W[[a, e]]
        return W
    for t in range(T):
        for dl, ell in enumerate(K.delays):
            a, b = t * m + K.dl_rail[dl], T * m + int(K.offsets[dl]) + t % ell
            W = damp(W, b, eta_l)
            W[[a, b]] = _G(float(th[dl, t]), float(ph[dl, t])) @ W[[a, b]]
        idx = np.arange(t * m, (t + 1) * m)
        W[idx] = U @ W[idx]
        for r in range(m):
            W = damp(W, t * m + r, eta_d)
    st = _product_mps(ems, T, m, W.shape[0])
    ref_k = EmitterSLOSKernel(st, dtype=torch.complex128)
    pr = ref_k.probs(torch.as_tensor(W)).numpy()
    qid = {tuple(int(x) for x in r): i for i, r in enumerate(K.basis.rail_patterns)}
    mref = np.zeros((T, K.Q))
    for o, q in zip(ref_k.layers.final_occupations(), pr):
        for t in range(T):
            mref[t, qid[tuple(int(x) for x in o[t * m:(t + 1) * m])]] += q
    marg = K.tick_marginals(th, ph, U, loss={"loop": eta_l, "detector": eta_d}).numpy()
    assert np.abs(marg - mref).max() < TOL
    S = 20000
    s = K.sample(S, th, ph, U, generator=torch.Generator().manual_seed(4), loss={"loop": eta_l, "detector": eta_d})
    emp = np.zeros_like(marg)
    for row in s.numpy():
        for t in range(T):
            emp[t, qid[tuple(int(x) for x in row[t])]] += 1.0 / S
    assert (np.abs(emp - marg) <= 5 * np.sqrt(np.clip(marg * (1 - marg), 0, None) / S) + 1.0 / S).all()


# ---- superposition boson sampling through the loops -------------------------------------------------------------
@pytest.mark.parametrize("loops,T,bins,alpha", [((1,), 4, (0, 1), math.pi / 4), ((1, 2), 6, (0, 1, 2), math.pi / 4),
                                                ((2,), 5, (0, 2, 3), 0.4)])
def test_sbs_through_loops_vs_unrolled(loops, T, bins, alpha):
    """Coherent per-bin sources cos(a)|0> + sin(a)|1> into the loops vs ProductSLOSKernel on the unrolled unitary
    (slots marginalised): every photon-number block of the distribution, the bin marginals and the sampler."""
    from photonic_engines import ProductSuperposition
    from photonic_engines.slos.kernel import ProductSLOSKernel
    from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel
    c = TimeLoopCircuit(T, loops)
    g = torch.Generator().manual_seed(31)
    th = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 6
    st = ProductSuperposition.sbs(len(bins), c.m, alpha=alpha, modes=bins)
    dist = ProductSLOSKernel(st, dtype=torch.complex128).probs(c.unitary(th, ph))
    ref = {}
    for k in range(dist.n + 1):
        for o, p in zip(dist.occupations(k), dist.probs[k].numpy()):
            key = tuple(int(x) for x in o[:T])
            ref[key] = ref.get(key, 0.0) + float(p)
    K = SuperpositionLoopKernel(c, st)
    assert len(bins) == K.N
    pats, probs = K.distribution(th, ph)
    got = _as_dict(pats, probs)
    assert _max_diff(got, ref) < TOL
    assert abs(sum(got.values()) - 1) < TOL
    marg = K.bin_marginals(th, ph).numpy()
    mref = np.zeros_like(marg)
    for key, v in ref.items():
        for t, b in enumerate(key):
            mref[t, b] += v
    assert np.abs(marg - mref).max() < TOL
    S = 20000
    s = K.sample(S, th, ph, generator=torch.Generator().manual_seed(5))
    emp = np.stack([np.bincount(s[:, t].numpy(), minlength=K.N + 1) / S for t in range(T)])
    assert (np.abs(emp - marg) <= 5 * np.sqrt(np.clip(marg * (1 - marg), 0, None) / S) + 1.0 / S).all()


def test_sbs_alpha_half_pi_is_fock():
    """a = (0, 1): every source emits one photon — the Fock input on the same bins, bit for bit."""
    from photonic_engines import ProductSuperposition
    from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel
    T, loops, bins = 5, (1, 2), (0, 1, 3)
    c = TimeLoopCircuit(T, loops)
    g = torch.Generator().manual_seed(32)
    th = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 6
    st = ProductSuperposition(np.tile([0.0, 1.0], (3, 1)), bins, c.m)
    occ = [1 if b in bins else 0 for b in range(T)]
    a = SuperpositionLoopKernel(c, st).bin_marginals(th, ph)
    b = LoopRegisterKernel(c, occ).bin_marginals(th, ph)
    assert torch.allclose(a, b, atol=TOL)


@pytest.mark.gpu
def test_cuda_matches_cpu():
    """The three loop inputs of part (b) and the multi-emitter kernel on CUDA: marginals equal to the CPU ones
    (complex128, 1e-10), samplers within 5 sigma of them."""
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    from photonic_engines import ProductSuperposition
    from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel
    T, loops, n = 10, (1, 2), 4
    c = TimeLoopCircuit(T, loops)
    g = torch.Generator().manual_seed(40)
    th = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 6
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.0)] * n, m=c.m, modes=[(2 * b, 2 * b + 1) for b in range(n)])
    sbs = ProductSuperposition.sbs(n, c.m, modes=range(n))
    makers = [lambda dev: LoopRegisterKernel(c, [0] * T, emitter=st, device=dev),
              lambda dev: SuperpositionLoopKernel(c, sbs, device=dev)]
    S = 20000
    for make in makers:
        m_cpu = make("cpu").bin_marginals(th, ph).numpy()
        K = make("cuda")
        assert np.abs(K.bin_marginals(th.cuda(), ph.cuda()).cpu().numpy() - m_cpu).max() < 1e-10
        s = K.sample(S, th.cuda(), ph.cuda(), generator=torch.Generator(device="cuda").manual_seed(1)).cpu().numpy()
        emp = np.stack([np.bincount(s[:, t], minlength=m_cpu.shape[1]) / S for t in range(T)])
        assert (np.abs(emp - m_cpu) <= 5 * np.sqrt(np.clip(m_cpu * (1 - m_cpu), 0, None) / S) + 1.0 / S).all()
    ems = _emitters(2, 2, [(0, 1), (2, 3)], seed=41)
    U = haar_unitary_np(4, np.random.default_rng(42))
    th2, ph2 = _settings(1, 2, 43)
    a = HybridRegisterKernel(ems, 4, GEOM).tick_marginals(th2, ph2, U).numpy()
    b = HybridRegisterKernel(ems, 4, GEOM, device="cuda").tick_marginals(th2.cuda(), ph2.cuda(), U).cpu().numpy()
    assert np.abs(a - b).max() < 1e-10


# ---- photon-number sector sampler ------------------------------------------------------------------------------
def _loop_emitter(case, c):
    if case == 0:
        return None
    from photonic_engines import EmitterMPS as _E
    return _E.caterpillar(3, [spin_gate(math.pi / 2, 0.3 * k) for k in range(3)], m=c.m, modes=[(2 * b, 2 * b + 1) for b in range(3)],
                          final=None if case == 1 else [0.6, 0.8j])


def _loop_cases():
    from photonic_engines import EmitterMPS as _E
    out = []
    c = TimeLoopCircuit(6, (1, 2))
    out.append(("fock", c, LoopRegisterKernel(c, [1, 1, 0, 1, 0, 0])))
    c2 = TimeLoopCircuit(8, (1, 2))
    st = _E.caterpillar(3, [spin_gate(math.pi / 2, 0.3 * k) for k in range(3)], m=c2.m, modes=[(2 * b, 2 * b + 1) for b in range(3)])
    out.append(("emitter", c2, LoopRegisterKernel(c2, [0] * 8, emitter=st)))
    stp = _E.caterpillar(3, [spin_gate(math.pi / 2, 0.3 * k) for k in range(3)], m=c2.m, modes=[(2 * b, 2 * b + 1) for b in range(3)],
                         final=[0.6, 0.8j])
    out.append(("emitter-projected", c2, LoopRegisterKernel(c2, [0] * 8, emitter=stp)))
    return out


@pytest.mark.parametrize("case", range(3))
def test_sector_sampler_forced_patterns_are_exact(case):
    """Forcing the sector march along given outcomes yields the patterns' probabilities: equal to the dense path's
    `probabilities` (complex128, 1e-12) on every pattern of the distribution, and the masses sum to the total."""
    from photonic_engines.timeloop.hybrid import light_loop_kernel, sector_sample
    _, c, lk = _loop_cases()[case]
    g = torch.Generator().manual_seed(50 + case)
    th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
    pats, probs = lk.distribution(th, ph)
    P = pats.shape[0]
    _, logp, stats = sector_sample(lk, P, lk._gates(th, ph), force=np.asarray(pats).reshape(P, lk.T, lk.bpp))
    assert torch.allclose(torch.exp(logp), probs, atol=TOL)
    assert stats["max_amplitudes"] < lk.chi * lk.D
    light = light_loop_kernel(c, lk.input, emitter=_loop_emitter(case, c))
    assert light.D < lk.D
    _, logp2, _ = sector_sample(light, P, light._gates(th, ph), force=np.asarray(pats).reshape(P, lk.T, lk.bpp))
    assert torch.allclose(torch.exp(logp2), probs, atol=TOL)                    # no dense basis needed


def test_sector_sampler_forced_many_photons_march_and_chain():
    """A packed 7-photon input through a delay-3 loop: the sector march forced on it (chain=False; sectors up to 7
    photons, the chunked Fock-block cap grows mid-march, fused and per-k gate paths both used) and the Fock chain it
    ships with give the dense `probabilities` on the 300 likeliest patterns."""
    from photonic_engines.timeloop.hybrid import _sector_march, light_loop_kernel, sector_sample
    T, N = 12, 7
    c = TimeLoopCircuit(T, (3,))
    inp = [1] * N + [0] * (T - N)
    lk = LoopRegisterKernel(c, inp)
    g = torch.Generator().manual_seed(90)
    th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
    pats, probs = lk.distribution(th, ph)
    top = torch.argsort(probs, descending=True)[:300]
    pats, probs = np.asarray(pats)[top.numpy()], probs[top]
    light = light_loop_kernel(c, inp)
    f = pats.reshape(len(top), T, 1)
    _, lp0, st0 = _sector_march(light, len(top), light._gates(th, ph), force=f, chain=False)
    _, lp1, st = sector_sample(light, len(top), light._gates(th, ph), force=f)
    assert torch.allclose(torch.exp(lp0), probs, atol=TOL)
    assert torch.allclose(torch.exp(lp1), torch.exp(lp0), atol=TOL)
    assert st0["max_amplitudes"] > 1 and st["max_amplitudes"] == 1 and max(st["mean_n"]) > 2


def test_sector_sampler_emitter_generator_gates_cap_growth():
    """Emitter (two bins per period: the generator gates are a transposed view) through a delay-3 loop, 7 photons: the
    chunked Fock-block cap grows mid-march, so matrix_exp sees a mid-range slice of the generators (crashed before
    the .contiguous() fix). Forced probabilities from generator gates equal those from explicit gate matrices and
    the dense `probabilities` on the 200 likeliest patterns."""
    from photonic_engines import EmitterMPS as _E
    from photonic_engines.timeloop.hybrid import light_loop_kernel, sector_sample
    from photonic_engines.timeloop.kernel import _GenSpec
    n = 7
    c = TimeLoopCircuit(2 * n, (3,))
    st = _E.caterpillar(n, [spin_gate(math.pi / 2, 0.2 * k) for k in range(n)], m=c.m, modes=[(2 * b, 2 * b + 1) for b in range(n)])
    lk = LoopRegisterKernel(c, [0] * c.T, emitter=st)
    g = torch.Generator().manual_seed(130)
    th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
    pats, probs = lk.distribution(th, ph)
    top = torch.argsort(probs, descending=True)[:200]
    pats, probs = np.asarray(pats)[top.numpy()], probs[top]
    light = light_loop_kernel(c, [0] * c.T, emitter=st)
    gen = light._gates(th, ph)
    assert isinstance(gen, _GenSpec) and not gen.h.is_contiguous()
    explicit = light._gates(c.gate_matrices(th, ph, torch.complex128, "cpu"), None)
    f = pats.reshape(len(top), light.T, light.bpp)
    _, lp_gen, stats = sector_sample(light, len(top), gen, force=f)
    _, lp_exp, _ = sector_sample(light, len(top), explicit, force=f)
    assert torch.allclose(torch.exp(lp_gen), probs, atol=TOL)
    assert torch.allclose(torch.exp(lp_gen), torch.exp(lp_exp), atol=TOL)
    assert max(stats["mean_n"]) > 1


def test_gate_column_matches_fock_blocks():
    """The closed-form one-column Fock amplitudes <d, c+s-d|V|c, s> equal the `fock_blocks` recursion (c = 0..3)."""
    from photonic_engines.timeloop.hybrid import gate_column
    from photonic_engines.timeloop.kernel import fock_blocks
    V = torch.as_tensor(haar_unitary_np(2, np.random.default_rng(120)), dtype=torch.complex128)
    V = torch.stack([V, torch.eye(2, dtype=torch.complex128), torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)])
    N = 14
    B = fock_blocks(V, N)
    for c in range(4):
        A = gate_column(V, torch.full((3,), c), N - c, N)
        for s in range(N - c + 1):
            assert torch.allclose(A[:, s, :c + s + 1], B[:, c + s, :c + s + 1, c], atol=1e-12)
            assert torch.all(A[:, s, c + s + 1:] == 0)
    # unitarity of every column up to s = 400 (log-domain sum, no overflow), random loop settings
    g = torch.Generator().manual_seed(121)
    th, ph = torch.rand(8, generator=g, dtype=torch.float64) * 2, torch.rand(8, generator=g, dtype=torch.float64) * 6
    ct, st = torch.cos(th) + 0j, torch.sin(th) + 0j
    Vr = torch.stack([torch.stack([ct, -st * torch.exp(-1j * ph)], -1), torch.stack([st * torch.exp(1j * ph), ct], -1)], -2)
    for c in range(3):
        A = gate_column(Vr, torch.full((8,), c), 400, 402)
        assert torch.isfinite(A).all() and ((A.abs() ** 2).sum(-1) - 1).abs().max() < 1e-10


@pytest.mark.parametrize("L,inp", [(1, [1, 1, 1, 0, 1, 0]), (2, [1, 0, 2, 1, 0, 1, 0, 0]),
                                   (5, [1, 1, 1, 1, 0, 0, 1, 0, 0, 0])])
def test_sector_sampler_chain_path_forced_exact(L, inp):
    """One loop with Fock input takes the Fock-chain path (`_loop_chain_sample`: the register is one basis state):
    forced probabilities of every pattern equal the dense `probabilities` (packed, spaced and 2-photon bins;
    L = 1, 2, 5)."""
    from photonic_engines.timeloop.hybrid import _head_program, light_loop_kernel, sector_sample
    T = len(inp)
    c = TimeLoopCircuit(T, (L,))
    lk = LoopRegisterKernel(c, inp)
    g = torch.Generator().manual_seed(100 + L)
    th = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 6
    pats, probs = lk.distribution(th, ph)
    light = light_loop_kernel(c, inp)
    assert _head_program(light) is not None and _head_program(LoopRegisterKernel(TimeLoopCircuit(6, (1, 2)), [1] * 6)) is None
    f = np.asarray(pats).reshape(len(probs), T, 1)
    _, lp, st = sector_sample(light, len(probs), light._gates(th, ph), force=f)
    assert st["max_amplitudes"] == 1
    assert torch.allclose(torch.exp(lp), probs, atol=TOL)
    assert abs(torch.exp(lp).sum().item() - 1) < 1e-10


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_sector_sampler_chain_path_law_with_loss(device):
    """Fock-chain path with loop and detector loss (binomial thinning of the head and of the rail) vs the dense
    density-matrix bin marginals (5 sigma per entry), one loop of delay 3; on the CPU and on CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda")
    from photonic_engines.timeloop.hybrid import light_loop_kernel, loop_sector_sample
    T, inp, S = 8, [1, 1, 1, 0, 1, 1, 0, 0], 20000
    c = TimeLoopCircuit(T, (3,))
    lk = LoopRegisterKernel(c, inp)
    g = torch.Generator().manual_seed(110)
    th = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, dtype=torch.float64) * 6
    K = lk if device == "cpu" else light_loop_kernel(c, inp, device="cuda")
    for loss in ({"loop": [0.7], "detector": 0.8}, {"loop": [0.9]}, {"detector": 0.6}):
        marg = lk.bin_marginals(th, ph, loss=loss).numpy()
        s = loop_sector_sample(K, S, th.to(device), ph.to(device), generator=torch.Generator(device=device).manual_seed(9),
                               loss=loss).cpu().numpy()
        emp = np.stack([np.bincount(s[:, t], minlength=marg.shape[1])[:marg.shape[1]] / S for t in range(T)])
        assert (np.abs(emp - marg) <= 5 * np.sqrt(np.clip(marg * (1 - marg), 0, None) / S) + 1.0 / S).all()


def test_sector_sampler_law_with_loss():
    """Sampled bin marginals of the sector sampler vs the dense density-matrix marginals, with loop and detector
    jumps, for a Fock and an emitter input (5 sigma per entry)."""
    from photonic_engines.timeloop.hybrid import loop_sector_sample
    loss = {"loop": [0.8, 0.9], "detector": 0.85}
    S = 20000
    for case in (0, 1):
        _, c, lk = _loop_cases()[case]
        g = torch.Generator().manual_seed(60 + case)
        th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
        ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
        marg = lk.bin_marginals(th, ph, loss=loss).numpy()
        s = loop_sector_sample(lk, S, th, ph, generator=torch.Generator().manual_seed(7), loss=loss).numpy()
        emp = np.stack([np.bincount(s[:, t], minlength=marg.shape[1]) / S for t in range(c.T)])
        assert (np.abs(emp - marg) <= 5 * np.sqrt(np.clip(marg * (1 - marg), 0, None) / S) + 1.0 / S).all()


def test_sector_sampler_law_lossless():
    """Sampled bin marginals vs the dense marginals, lossless, Fock and projected-emitter inputs."""
    from photonic_engines.timeloop.hybrid import loop_sector_sample
    S = 20000
    for case in (0, 2):
        _, c, lk = _loop_cases()[case]
        g = torch.Generator().manual_seed(95 + case)
        th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
        ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
        pats, probs = lk.distribution(th, ph)
        pats, probs = np.asarray(pats).reshape(len(probs), -1), probs.numpy() / probs.sum().item()
        marg = np.stack([np.bincount(pats[:, t], weights=probs, minlength=4)[:4] for t in range(pats.shape[1])])
        s = loop_sector_sample(lk, S, th, ph, generator=torch.Generator().manual_seed(8)).numpy()
        emp = np.stack([np.bincount(s[:, t], minlength=4)[:4] / len(s) for t in range(s.shape[1])])
        n = len(s)
        assert (np.abs(emp - marg) <= 5 * np.sqrt(np.clip(marg * (1 - marg), 0, None) / n) + 1.0 / n).all()


def test_sector_sampler_hybrid_kernel():
    """The multi-emitter kernel on sectors: forced patterns reproduce the dense probabilities on oracle patterns."""
    from photonic_engines.timeloop.hybrid import sector_sample
    T, m = 2, 4
    ems = _emitters(2, T, [(0, 1), (2, 3)], seed=70)
    K = HybridRegisterKernel(ems, m, GEOM)
    th, ph = _settings(K.n_dl, T, 71)
    U = haar_unitary_np(m, np.random.default_rng(72))
    ref = _oracle(K, ems, th, ph, U)
    keys = sorted(ref, key=ref.get, reverse=True)[:200]
    pats = np.asarray(keys).reshape(len(keys), T, m)
    _, logp, _ = sector_sample(K, len(keys), K.gates(th, ph, U), force=pats)
    assert np.abs(np.exp(logp.numpy()) - np.array([ref[k] for k in keys])).max() < TOL
    s = K.sample_sectors(64, th, ph, U, generator=torch.Generator().manual_seed(1))
    assert s.shape == (64, T, m) and (s.sum((1, 2)) <= 2 * T).all()
    # the light kernel (no dense basis; what bench/hybrid_frontier.py --input hybrid builds) gives the same law
    from photonic_engines.timeloop.hybrid import light_hybrid_kernel
    Kl = light_hybrid_kernel(ems, m, GEOM)
    _, logp_l, _ = sector_sample(Kl, len(keys), Kl.gates(th, ph, U), force=pats)
    assert np.abs(np.exp(logp_l.numpy()) - np.array([ref[k] for k in keys])).max() < TOL
    sl = Kl.sample_sectors(64, th, ph, U, generator=torch.Generator().manual_seed(1))
    assert torch.equal(sl, s)


@pytest.mark.gpu
def test_cuda_sector_sampler():
    """The sector sampler on CUDA (the loop column's sampler of record): Fock and emitter inputs, with loop and
    detector jumps, against the CPU dense density-matrix marginals (5 sigma per entry)."""
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    from photonic_engines.timeloop.hybrid import light_loop_kernel, loop_sector_sample
    loss = {"loop": [0.8, 0.9], "detector": 0.85}
    S = 20000
    for case in (0, 1):
        _, c, lk = _loop_cases()[case]
        g = torch.Generator().manual_seed(80 + case)
        th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
        ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
        marg = lk.bin_marginals(th, ph, loss=loss).numpy()
        K = light_loop_kernel(c, lk.input, device="cuda", emitter=_loop_emitter(case, c))
        s = loop_sector_sample(K, S, th.cuda(), ph.cuda(), generator=torch.Generator(device="cuda").manual_seed(2),
                               loss=loss).cpu().numpy()
        emp = np.stack([np.bincount(s[:, t], minlength=marg.shape[1]) / S for t in range(c.T)])
        assert (np.abs(emp - marg) <= 5 * np.sqrt(np.clip(marg * (1 - marg), 0, None) / S) + 1.0 / S).all()


def test_superposition_loop_many_levels():
    """A per-bin source with d >= 22 photon-number levels (q! beyond int64, e.g. a truncated squeezed vacuum): the
    amplitudes are scaled by 1/sqrt(q!) in floating point. Two sources of 25 levels, the top level only on the first:
    marginal photon-number mass of the first source's photons conserved (norm 1, one loop)."""
    from photonic_engines import ProductSuperposition
    from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel
    d = 25
    a = np.zeros((2, d)); a[0, 0] = a[0, d - 1] = 1 / math.sqrt(2); a[1, 1] = 1.0
    c = TimeLoopCircuit(4, (1,))
    K = SuperpositionLoopKernel(c, ProductSuperposition(a, (0, 1), c.m))
    th = torch.full((1, 4), 0.4, dtype=torch.float64)
    marg = K.bin_marginals(th, torch.zeros_like(th), return_register=True)
    m, rho = marg
    total = float(m.sum(1)[-1])
    assert abs(total - 1.0) < 1e-10
    left = torch.einsum("ii,i->", rho, torch.as_tensor(K.basis.total, dtype=rho.dtype)).real        # still in the loop
    mean = float((m * torch.arange(m.shape[1], dtype=m.dtype)).sum()) + float(left)
    assert abs(mean - (0.5 * (d - 1) + 1)) < 1e-9                                  # photon number conserved: E[n] = 13
