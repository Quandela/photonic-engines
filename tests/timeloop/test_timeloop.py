"""Native time-loop engine vs the generic path (FockSLOSKernel on the unrolled unitary)."""
import math

import numpy as np
import pytest
import torch

from photonic_engines.slos.kernel import FockSLOSKernel
from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit, register_dimension


def _params(c, seed):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, c.T, generator=g, dtype=torch.float64) * 6
    return th, ph


def _generic_marginals(U, T, inp, m):
    N = sum(inp)
    k = FockSLOSKernel(N, m, list(inp) + [0] * (m - T), dtype=torch.complex128)
    probs = k.probs(U).numpy()
    occ = k.layers.final_occupations()
    ref, marg = {}, np.zeros((T, N + 1))
    for o, p in zip(occ, probs):
        key = tuple(int(x) for x in o[:T])
        ref[key] = ref.get(key, 0.0) + p
        for t in range(T):
            marg[t, o[t]] += p
    return ref, marg


CASES = [((1,), 4, [1, 1, 0, 0]), ((2,), 5, [1, 0, 1, 0, 0]), ((1, 2), 6, [1, 1, 1, 0, 0, 0]),
         ((2, 1), 6, [2, 0, 1, 0, 0, 0]), ((3,), 6, [1, 1, 1, 1, 0, 0])]


@pytest.mark.parametrize("loops,T,inp", CASES)
def test_native_matches_unrolled(loops, T, inp):
    c = TimeLoopCircuit(T, loops)
    th, ph = _params(c, 1)
    U = c.unitary(th, ph)
    assert torch.allclose(U @ U.conj().T, torch.eye(c.m, dtype=U.dtype), atol=1e-12)
    ref, mref = _generic_marginals(U, T, inp, c.m)
    lk = LoopRegisterKernel(c, inp)
    assert register_dimension(sum(inp), c.L) == lk.D
    pats, pr = lk.distribution(th, ph)
    got = {tuple(int(x) for x in p): float(q) for p, q in zip(pats, pr)}
    assert abs(sum(got.values()) - 1) < 1e-12
    assert max(abs(got.get(k, 0.0) - v) for k, v in ref.items()) < 1e-12
    assert max(abs(ref.get(k, 0.0) - v) for k, v in got.items()) < 1e-12
    assert torch.allclose(lk.probabilities(pats, th, ph), pr, atol=1e-13)
    assert np.abs(lk.bin_marginals(th, ph).numpy() - mref).max() < 1e-12


def test_sampler_marginals():
    c = TimeLoopCircuit(6, (1, 2))
    inp = [1, 1, 1, 0, 0, 0]
    th, ph = _params(c, 2)
    lk = LoopRegisterKernel(c, inp)
    mref = lk.bin_marginals(th, ph).numpy()
    s = lk.sample(20000, th, ph, generator=torch.Generator().manual_seed(3))
    emp = np.stack([np.bincount(s[:, t].numpy(), minlength=lk.N + 1) / 20000 for t in range(c.T)])
    assert np.abs(emp - mref).max() < 0.012


def test_losses_match_loss_modes():
    """Loop and detector losses as beam splitters to fresh vacuum wires in the unrolled network."""
    loops, T, inp = (1, 2), 5, [1, 1, 1, 0, 0]
    c = TimeLoopCircuit(T, loops)
    th, ph = _params(c, 4)
    etas, etad = [0.7, 0.85], 0.9
    m = c.m + T * c.n_loops + T
    U = torch.eye(m, dtype=torch.complex128)

    def bs(U, a, b, G):
        idx = torch.tensor([a, b])
        U = U.clone()
        U[idx] = G @ U[idx]
        return U

    def damp(eta):
        s, r = math.sqrt(eta), math.sqrt(1 - eta)
        return torch.tensor([[s, r], [-r, s]], dtype=torch.complex128)

    w = c.m
    for t in range(T):
        for i in range(c.n_loops):
            s = c.slot_wire(i, t)
            U = bs(U, s, w, damp(etas[i]))
            w += 1
            U = bs(U, t, s, TimeLoopCircuit.bs_matrix(th[i, t], ph[i, t]).to(torch.complex128))
        U = bs(U, t, w, damp(etad))
        w += 1
    _, mref = _generic_marginals(U, T, inp, m)
    lk = LoopRegisterKernel(c, inp)
    loss = {"loop": etas, "detector": etad}
    assert np.abs(lk.bin_marginals(th, ph, loss=loss).numpy() - mref).max() < 1e-12
    s = lk.sample(40000, th, ph, generator=torch.Generator().manual_seed(5), loss=loss)
    emp = np.stack([np.bincount(s[:, t].numpy(), minlength=lk.N + 1) / 40000 for t in range(T)])
    assert np.abs(emp - mref).max() < 0.012


def test_gradients():
    c = TimeLoopCircuit(5, (1, 2))
    inp = [1, 1, 1, 0, 0]
    th, ph = _params(c, 6)
    lk = LoopRegisterKernel(c, inp)
    th.requires_grad_(True)
    pats = [[1, 1, 1, 0, 0], [0, 1, 0, 1, 1]]
    p = lk.probabilities(pats, th, ph).sum()
    g = torch.autograd.grad(p, th)[0]
    eps = 1e-6
    th2 = th.detach().clone()
    th2[0, 1] += eps
    fd = (lk.probabilities(pats, th2, ph).sum() - p.detach()) / eps
    assert abs(float(g[0, 1]) - float(fd)) < 1e-5
    mg = lk.bin_marginals(th, ph, loss={"loop": [0.8, 0.9]})[2, 1]
    g = torch.autograd.grad(mg, th)[0]
    fd = (lk.bin_marginals(th2, ph, loss={"loop": [0.8, 0.9]})[2, 1] - mg.detach()) / eps
    assert abs(float(g[0, 1]) - float(fd)) < 1e-5


def test_flush_settings_empty_loops():
    c = TimeLoopCircuit(8, (1, 2))
    th, ph = _params(c, 7)
    th, ph = c.flush_settings(th, ph)
    inp = [1, 1, 1, 0, 0, 0, 0, 0]
    lk = LoopRegisterKernel(c, inp)
    _, rho = lk.bin_marginals(th, ph, return_register=True)
    d = torch.diagonal(rho).real
    assert float(d[lk._vac]) > 1 - 1e-12
    s = lk.sample(200, th, ph, generator=torch.Generator().manual_seed(8))
    assert (s.sum(1) == 3).all()


@pytest.mark.gpu
def test_cuda_sampler():
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    c = TimeLoopCircuit(12, (1, 2))
    inp = [1] * 6 + [0] * 6
    th, ph = _params(c, 9)
    lk = LoopRegisterKernel(c, inp, device="cuda")
    m_cpu = LoopRegisterKernel(c, inp).bin_marginals(th, ph).numpy()
    m_gpu = lk.bin_marginals(th, ph).cpu().numpy()
    assert np.abs(m_cpu - m_gpu).max() < 1e-10
    s = lk.sample(20000, th, ph, generator=torch.Generator(device="cuda").manual_seed(1))
    emp = np.stack([np.bincount(s[:, t].cpu().numpy(), minlength=lk.N + 1) / 20000 for t in range(c.T)])
    assert np.abs(emp - m_cpu).max() < 0.012


def test_module_training_step():
    from photonic_engines.timeloop import TimeLoopModule
    c = TimeLoopCircuit(6, (1, 2))
    layer = TimeLoopModule(c, [1, 1, 1, 0, 0, 0], loss={"loop": [0.95, 0.9], "detector": 0.9})
    y = layer()
    assert y.shape == (6,) and (y >= 0).all() and (y <= 1).all()
    target = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.5, 0.5], dtype=y.dtype)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    l0 = float(((y - target) ** 2).sum().detach())
    for _ in range(15):
        opt.zero_grad()
        l = ((layer() - target) ** 2).sum()
        l.backward()
        opt.step()
    assert float(l.detach()) < l0
    p = layer(readout="patterns", patterns=[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], loss={})
    assert p.shape == (2,)
    s = layer.sample(50, generator=torch.Generator().manual_seed(0), threshold=True)
    assert s.shape == (50, 6) and int(s.max()) <= 1


def test_periodic_multi_rail_vs_unrolled():
    """Two rails, one delay line on rail 1 (Perceval TD semantics) vs FockSLOSKernel on the
    hand-unrolled network."""
    from photonic_engines.timeloop import PeriodicNetworkKernel
    T, m, t = 3, 2, 2
    g = torch.Generator().manual_seed(11)
    U1 = torch.linalg.qr(torch.randn(2, 2, dtype=torch.complex128, generator=g))[0]
    U2 = torch.linalg.qr(torch.randn(2, 2, dtype=torch.complex128, generator=g))[0]
    swap = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    program = [("gate", 0, 1, 0), ("gate", 1, ("slot", 0), 1), ("gate", 0, 1, 2)]
    gates = torch.stack([U1, swap, U2])[:, None].expand(3, T, 2, 2)
    inp = [[1, 0]] * T
    k = PeriodicNetworkKernel(m, [t], program, T, inp)
    pats, probs = k.distribution(gates)
    # unrolled: wires = T*m rail wires + t slot wires; per period: U1 on (2p, 2p+1), swap (2p+1, slot),
    # U2 on (2p, 2p+1)
    M = T * m + t
    W = torch.eye(M, dtype=torch.complex128)

    def apply(W, a, b, G):
        idx = torch.tensor([a, b])
        W = W.clone()
        W[idx] = G @ W[idx]
        return W

    for p in range(T):
        W = apply(W, 2 * p, 2 * p + 1, U1)
        W = apply(W, 2 * p + 1, T * m + (p % t), swap)
        W = apply(W, 2 * p, 2 * p + 1, U2)
    occ_in = [1, 0] * T + [0] * t
    ref = FockSLOSKernel(T, M, occ_in, dtype=torch.complex128)
    pr = ref.probs(W).numpy()
    occ = ref.layers.final_occupations()
    refd = {}
    for o, q in zip(occ, pr):
        key = tuple(int(x) for x in o[:T * m])
        refd[key] = refd.get(key, 0.0) + q
    got = {tuple(int(x) for x in a.reshape(-1)): float(b) for a, b in zip(pats, probs)}
    assert max(abs(got.get(kk, 0.0) - v) for kk, v in refd.items()) < 1e-12
    assert max(abs(refd.get(kk, 0.0) - v) for kk, v in got.items()) < 1e-12


@pytest.mark.parametrize("n,loops,final", [(2, (1,), None), (3, (1, 2), None), (3, (2,), [1, 0]), (3, (1,), [0.6, 0.8j])])
def test_emitter_into_loops(n, loops, final):
    """Caterpillar state (dual-rail bins (2k, 2k+1)) feeding the loops vs EmitterSLOSKernel on
    the unrolled unitary; traced spin: marginals + sampler; projected spin: rejection sampler."""
    from photonic_engines import EmitterMPS, spin_gate
    from photonic_engines.slos.emitter import EmitterSLOSKernel
    T = 2 * n + 2
    c = TimeLoopCircuit(T, loops)
    th, ph = _params(c, 20 + n)
    U = c.unitary(th, ph)
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.3 * k) for k in range(n)], m=c.m, final=final)
    ref_k = EmitterSLOSKernel(st, dtype=torch.complex128)
    pr = ref_k.probs(U).numpy()
    occ = ref_k.layers.final_occupations()
    ref = {}
    for o, q in zip(occ, pr):
        key = tuple(int(x) for x in o[:T])
        ref[key] = ref.get(key, 0.0) + q
    lk = LoopRegisterKernel(c, [0] * T, emitter=st)
    assert (lk.bpp, lk.chi, lk.N) == (2, 2, n)
    pats, probs = lk.distribution(th, ph)
    got = {tuple(int(x) for x in a): float(b) for a, b in zip(pats, probs)}
    assert max(abs(got.get(k, 0.0) - v) for k, v in ref.items()) < 1e-12
    assert max(abs(ref.get(k, 0.0) - v) for k, v in got.items()) < 1e-12
    assert torch.allclose(lk.probabilities(pats, th, ph), probs, atol=1e-13)
    g = torch.Generator().manual_seed(1)
    if final is None:
        marg = lk.bin_marginals(th, ph).numpy()
        mref = np.zeros((T, n + 1))
        for k, v in ref.items():
            for t, b in enumerate(k):
                mref[t, b] += v
        assert np.abs(marg - mref).max() < 1e-12
        s = lk.sample(20000, th, ph, generator=g)
        emp = np.stack([np.bincount(s[:, t].numpy(), minlength=n + 1) / 20000 for t in range(T)])
        assert np.abs(emp - mref).max() < 0.015
    else:
        s = lk.sample(20000, th, ph, generator=g)
        tot = sum(ref.values())                                   # projection probability
        assert abs(len(s) / 20000 - tot) < 0.02
        emp = {}
        for row in s.numpy():
            key = tuple(int(x) for x in row)
            emp[key] = emp.get(key, 0) + 1 / len(s)
        assert 0.5 * sum(abs(emp.get(k, 0) - v / tot) for k, v in ref.items()) < 0.05
