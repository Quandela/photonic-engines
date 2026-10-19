"""SequenceKernel (docs/SEQUENCE.md): k emitters x n bins through a static interferometer, checked
against the brute-force k*n-photon strong simulation on the block-diagonal unrolled unitary."""
import math

import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel
from photonic_engines.slos.emitter import EmitterSLOSKernel

C128 = torch.complex128


def _embed(op, i, dims):
    out = np.ones((1, 1), dtype=complex)
    for j, d in enumerate(dims):
        out = np.kron(out, op if j == i else np.eye(d, dtype=complex))
    return out


def _setup(k=2, n=3, m=4, seed=0, final=False):
    rng = np.random.default_rng(seed)
    ems = []
    for j in range(k):
        gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, rng.uniform(0, 6)) for _ in range(n - 1)]
        f = None if not final else np.array([1.0, np.exp(1j * rng.uniform(0, 6))]) / math.sqrt(2)
        ems.append(BinEmitter.caterpillar(n, gates, (2 * j, 2 * j + 1), final=f))
    U = haar_unitary(m, seed + 5)
    return ems, U


def _brute(ems, U, n, m):
    """Full k*n-photon EmitterMPS on the unrolled modes, block-diagonal unitary -> (sequences [B, n, m], probs)."""
    k = len(ems)
    dims = [e.d for e in ems]
    ops, modes = [], []
    for b in range(n):
        for j, e in enumerate(ems):
            ops.append(np.stack([_embed(e.ops[b, l], j, dims) for l in range(e.ops.shape[1])]))
            modes.append(tuple(b * m + x for x in e.modes))
    s0 = np.ones(1, dtype=complex)
    for e in ems:
        s0 = np.kron(s0, e.s0)
    final = None
    if all(e.final is not None for e in ems):
        final = np.ones(1, dtype=complex)
        for e in ems:
            final = np.kron(final, e.final)
    full = EmitterMPS(ops=np.stack(ops), modes=tuple(modes), s0=s0, m=n * m, final=final)
    kern = EmitterSLOSKernel(full, dtype=C128)
    p = kern.probs(torch.block_diag(*[U] * n))
    occ = kern.layers.final_occupations()
    seqs, probs = [], []
    for o, q in zip(occ, p.tolist()):
        blocks = o.reshape(n, m)
        if (blocks.sum(1) == k).all():
            seqs.append(blocks); probs.append(q)
    return np.array(seqs), torch.tensor(probs, dtype=torch.float64)


def test_sequence_probability_matches_brute_force():
    k, n, m = 2, 3, 4
    ems, U = _setup(k, n, m)
    sk = SequenceKernel(ems, m)
    assert sk.chi == 4 and sk.S == 10 and sk.n == n
    seqs, ref = _brute(ems, U, n, m)
    p = sk.sequence_probability(U, seqs)
    assert (p - ref).abs().max() < 1e-9 and abs(float(p.sum()) - 1) < 1e-9
    assert abs(float(sk.sequence_probability(U, seqs[3]) - ref[3])) < 1e-9
    idx = sk.pattern_index(seqs)
    assert (sk.sequence_probability(U, idx.numpy()) - ref).abs().max() < 1e-9


def test_marginals_joint_and_gradient():
    k, n, m = 2, 3, 4
    ems, U = _setup(k, n, m, seed=1)
    sk = SequenceKernel(ems, m)
    seqs, ref = _brute(ems, U, n, m)
    marg = sk.bin_marginals(U)
    assert marg.shape == (n, sk.S) and (marg.sum(1) - 1).abs().max() < 1e-12
    for b in range(n):
        mb = torch.zeros(sk.S, dtype=torch.float64).index_add_(0, sk.pattern_index(seqs[:, b]), ref)
        assert (marg[b] - mb).abs().max() < 1e-12
    J = sk.joint(U, [0, 2])
    j2 = torch.zeros(sk.S, sk.S, dtype=torch.float64)
    j2.index_put_((sk.pattern_index(seqs[:, 0]), sk.pattern_index(seqs[:, 2])), ref, accumulate=True)
    assert (J - j2).abs().max() < 1e-12
    Ug = U.clone().requires_grad_(True)
    SequenceKernel(ems, m).bin_marginals(Ug)[2, 0].backward()
    assert float(Ug.grad.abs().sum()) > 0


def test_projected_emitters_amplitudes():
    k, n, m = 2, 2, 4
    ems, U = _setup(k, n, m, seed=2, final=True)
    sk = SequenceKernel(ems, m)
    seqs, ref = _brute(ems, U, n, m)
    p = sk.sequence_probability(U, seqs)
    a = sk.sequence_amplitude(U, seqs)
    assert (p - ref).abs().max() < 1e-9 and (a.abs() ** 2 - ref).abs().max() < 1e-9
    s = sk.sample(U, 500, generator=torch.Generator().manual_seed(0))
    assert s.shape[1:] == (n, m) and 0 < s.shape[0] <= 500        # rejection on the projection


def test_sampling_and_loss():
    k, n, m = 2, 3, 4
    ems, U = _setup(k, n, m, seed=3)
    sk = SequenceKernel(ems, m)
    seqs, ref = _brute(ems, U, n, m)
    shots = 20000
    s = sk.sample(U, shots, generator=torch.Generator().manual_seed(1))
    assert s.shape == (shots, n, m) and (s.sum(2) == k).all()
    emp = {}
    for r in s.reshape(shots, -1).tolist():
        emp[tuple(r)] = emp.get(tuple(r), 0) + 1 / shots
    refd = {tuple(sq.reshape(-1).tolist()): float(q) for sq, q in zip(seqs, ref)}
    tvd = 0.5 * sum(abs(emp.get(x, 0) - refd.get(x, 0)) for x in set(emp) | set(refd))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in refd.values())
    assert tvd < 2.0 * floor
    ml = sk.bin_marginals(U, loss=0.7)
    sl = sk.sample(U, shots, generator=torch.Generator().manual_seed(2), loss=0.7)
    assert sl.sum(2).max() <= k and sl.sum(2).min() >= 0
    d = ml[1].as_dict()
    emp = {}
    for r in sl[:, 1].tolist():
        emp[tuple(r)] = emp.get(tuple(r), 0) + 1 / shots
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in d.values())
    assert 0.5 * sum(abs(emp.get(x, 0) - d.get(x, 0)) for x in set(emp) | set(d)) < 2.5 * floor


def test_per_bin_unitaries_and_from_mps():
    k, n, m = 2, 2, 4
    ems, U = _setup(k, n, m, seed=4)
    Us = torch.stack([U, haar_unitary(m, 11)])
    sk = SequenceKernel(ems, m)
    marg = sk.bin_marginals(Us)
    # bin 0 marginal does not depend on the second unitary
    assert (marg[0] - SequenceKernel(ems, m).bin_marginals(U)[0]).abs().max() < 1e-12
    # from_mps: EmitterMPS on the unrolled modes
    mps = [EmitterMPS(ops=e.ops, modes=tuple(tuple(b * m + x for x in e.modes) for b in range(n)), s0=e.s0, m=n * m)
           for e in ems]
    sk2 = SequenceKernel.from_mps(mps, m)
    assert (sk2.bin_marginals(U) - sk.bin_marginals(U)).abs().max() < 1e-12
    with pytest.raises(ValueError):
        BinEmitter.from_mps(EmitterMPS(ops=ems[0].ops, modes=((0, 1), (m + 2, m + 3)), s0=ems[0].s0, m=2 * m), m)


def test_label_diagonal_sampler_matches_kraus_route():
    """K_t psi == V diag(<t|U|x>) Uop psi for every pattern t (label-diagonal factorisation),
    and sample_diag reproduces the bin marginals and a two-bin joint of the Kraus-route sampler."""
    k, n, m = 3, 2, 6
    gates = [spin_gate(math.pi / 2, 0.3), spin_gate(1.1, 0.2)]
    ems = [BinEmitter.caterpillar(n, gates, (2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m)
    U = torch.as_tensor(haar_unitary(m, seed=5))
    K = sk.kraus(U, 0)
    amp = sk.label_amplitudes(U, torch.as_tensor(sk.patterns))
    V, Uop = sk._rank1_ops(0)
    psi = torch.randn(3, sk.chi, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    lhs = torch.einsum("tab,sb->sta", K, psi)
    rhs = torch.stack([sk._apply_local(amp[t][None] * sk._apply_local(psi, Uop), V) for t in range(sk.S)], 1)
    assert (lhs - rhs).abs().max() < 1e-12
    shots = 20000
    s, reg = sk.sample_diag(U, shots, generator=torch.Generator().manual_seed(1), return_register=True)
    assert s.shape == (shots, n, m) and reg.shape == (shots, sk.chi)
    assert torch.allclose(reg.norm(dim=1), torch.ones(shots, dtype=torch.float64))
    marg = sk.bin_marginals(U)
    for b in range(n):
        emp = torch.zeros(sk.S, dtype=torch.float64).index_add_(0, sk.pattern_index(s[:, b].numpy()).cpu(),
                                                                torch.full((shots,), 1 / shots, dtype=torch.float64))
        floor = 0.5 * math.sqrt(2 / math.pi) * torch.sqrt(marg[b] * (1 - marg[b]) / shots).sum()
        assert 0.5 * (emp - marg[b]).abs().sum() < 3 * floor
    J = sk.joint(U, [0, 1])
    i0, i1 = sk.pattern_index(s[:, 0].numpy()).cpu(), sk.pattern_index(s[:, 1].numpy()).cpu()
    emp = torch.zeros(sk.S, sk.S, dtype=torch.float64).index_put_(
        (i0, i1), torch.full((shots,), 1 / shots, dtype=torch.float64), accumulate=True)
    floor = 0.5 * math.sqrt(2 / math.pi) * torch.sqrt(J * (1 - J) / shots).sum()
    assert 0.5 * (emp - J).abs().sum() < 1.5 * floor
    # a non-label-preserving emitter is refused
    bad = BinEmitter(np.stack([np.stack([np.eye(2)] * 2)] * n), (0, 1), (1.0, 0.0))
    with pytest.raises(ValueError):
        SequenceKernel([bad], 2).sample_diag(torch.eye(2, dtype=torch.complex128), 2)


def test_label_amplitudes_glynn_matches_laplace():
    """The Glynn / GEMM evaluation of the 2^k label amplitudes equals the Laplace recurrence,
    with the bin's U and with per-stream unitaries (k = 5, collisions included)."""
    import math

    import numpy as np

    from photonic_engines import spin_gate
    from photonic_engines.sequence import BinEmitter, SequenceKernel
    k, m = 5, 10
    rng = np.random.default_rng(4)
    ems = [BinEmitter.caterpillar(2, [spin_gate(math.pi / 2, 0)] * 2, (2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m)

    def haar(seed):
        r_ = np.random.default_rng(seed)
        z = r_.normal(size=(m, m)) + 1j * r_.normal(size=(m, m))
        q, r = np.linalg.qr(z)
        return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))))

    S = 12
    counts = torch.zeros((S, m), dtype=torch.long)
    for s in range(S):
        for i in rng.choice(m, k, replace=True):
            counts[s, i] += 1
    U = haar(1)
    a_g = sk.label_amplitudes(U, counts)
    a_l = sk.label_amplitudes(U, counts, method="laplace")
    assert torch.allclose(a_g, a_l, atol=1e-13)
    Us = torch.stack([haar(10 + s) for s in range(S)])
    a_g = sk.label_amplitudes(U, counts, U_streams=Us)
    a_l = sk.label_amplitudes(U, counts, U_streams=Us, method="laplace")
    assert torch.allclose(a_g, a_l, atol=1e-13)
    assert a_g.shape == (S, 2 ** k)
    # the uncollapsed sum over all 2^{k-1} signs (the A/B reference) gives the same values
    a_f = sk.label_amplitudes(U, counts, U_streams=Us, method="glynn_full")
    assert torch.allclose(a_g, a_f, atol=1e-13)
    with pytest.raises(ValueError):
        sk.label_amplitudes(U, counts, method="ryser")


def test_sample_diag_timing_and_floor_options_on_cpu():
    """`timing` accumulates one entry per phase of every tick; `draw="device"` and `fock_backend`
    are accepted and change nothing on the cpu (the device draw needs cuda, the backend is the
    cuda sampler's); the trajectories at a seed are those of the plain call."""
    k, n, m = 3, 2, 6
    gates = [spin_gate(math.pi / 2, 0.3), spin_gate(1.1, 0.2)]
    ems = [BinEmitter.caterpillar(n, gates, (2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m)
    U = torch.as_tensor(haar_unitary(m, seed=5))
    T = {}
    s = sk.sample_diag(U, 200, generator=torch.Generator().manual_seed(1), timing=T)
    assert set(T) == {"apply", "draw", "fock", "amp", "update"} and all(v >= 0 for v in T.values())
    s2 = sk.sample_diag(U, 200, generator=torch.Generator().manual_seed(1), draw="device", fock_backend="cuda")
    s3 = sk.sample_diag(U, 200, generator=torch.Generator().manual_seed(1), draw="host")
    assert torch.equal(s, s2) and torch.equal(s, s3)
    with pytest.raises(ValueError):
        sk.sample_diag(U, 2, draw="gpu")


def test_label_amplitudes_collapsed_sum_bunched_patterns():
    """The collapsed Glynn sum (docs/SEQUENCE.md §12b) on every multiplicity shape of k = 6
    photons — fully bunched, all-even multiplicities (no complement halving), one odd row,
    collision-free — equals the Laplace recurrence, whatever the term grouping (`pad_tol`) and
    the chunking (`chunk_bytes` down to one term per block), and the term counts are
    prod (t_r + 1), halved once when some t_r is odd."""
    import math

    import numpy as np

    from photonic_engines import spin_gate
    from photonic_engines.sequence import BinEmitter, SequenceKernel
    k, m = 6, 12
    ems = [BinEmitter.caterpillar(2, [spin_gate(math.pi / 2, 0)] * 2, (2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m)
    r_ = np.random.default_rng(3)
    z = r_.normal(size=(m, m)) + 1j * r_.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    U = torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))))
    shapes = [(6,), (4, 2), (2, 2, 2), (3, 3), (5, 1), (1, 1, 1, 1, 1, 1), (2, 1, 1, 1, 1), (4, 1, 1), (3, 2, 1)]
    counts = torch.zeros((len(shapes), m), dtype=torch.long)
    for s, shape in enumerate(shapes):
        for i, t in enumerate(shape):
            counts[s, 2 * i + 1] = t
    rows = torch.repeat_interleave(torch.arange(m).repeat(len(shapes)), counts.reshape(-1)).view(-1, k)
    _, _, _, N, scale = sk._multiplicity_tables(rows)
    for s, shape in enumerate(shapes):
        full = math.prod(t + 1 for t in shape)
        odd = any(t % 2 for t in shape)
        assert int(N[s]) == (full // 2 if odd else full)
        assert float(scale[s]) == (2.0 if odd else 1.0) * 2.0 ** (-k)
    a_l = sk.label_amplitudes(U, counts, method="laplace")
    for tol in (1.0, 1.25, 1e9):
        for cb in (1 << 30, 4000, 600):
            a_c = sk.label_amplitudes(U, counts, chunk_bytes=cb, pad_tol=tol)
            assert torch.allclose(a_c, a_l, atol=1e-14), (tol, cb)


def _tvd_to_law(samples, pats, p):
    """Total variation distance between the empirical law of samples [S, m] and p over the patterns pats [K, m]."""
    index = {tuple(o): i for i, o in enumerate(pats.tolist())}
    counts = torch.zeros(len(p), dtype=torch.float64)
    for row in samples.tolist():
        counts[index[tuple(row)]] += 1
    return 0.5 * float((counts / samples.shape[0] - p).abs().sum())


def test_fock_sample_public_api_matches_the_strong_law():
    """SequenceKernel.fock_sample (public): exact k-photon patterns per stream, distributed as the Fock SLOS law of
    each stream's input (repeated input modes included), one unitary per stream with U_streams, and k = 0 is the
    vacuum. The TVD bound is three times the expected shot-noise TVD, 0.5 sqrt(K / S), K patterns, S shots."""
    from photonic_engines.slos import FockSLOSKernel
    ems, U = _setup(k=2, n=2, m=4)
    sk = SequenceKernel(ems, 4)
    Ut = torch.as_tensor(U, dtype=C128)
    g = torch.Generator().manual_seed(3)
    shots = 20000
    for inp in ((0, 2), (1, 1)):
        cols = torch.tensor([inp] * shots)
        occ = sk.fock_sample(Ut, cols, g)
        assert occ.shape == (shots, 4) and occ.dtype == torch.int64 and bool((occ.sum(1) == 2).all())
        kern = FockSLOSKernel(2, 4, [inp.count(x) for x in range(4)], dtype=C128)
        p = kern.probs(Ut).to(torch.float64)
        pats = torch.as_tensor(kern.layers.final_occupations())
        assert _tvd_to_law(occ, pats, p) < 3 * 0.5 * math.sqrt(len(p) / shots)
    U2 = torch.as_tensor(haar_unitary(4, 77), dtype=C128)                          # two unitaries, one per half
    Us = torch.stack([Ut] * (shots // 2) + [U2] * (shots // 2))
    occ = sk.fock_sample(Ut, torch.tensor([(0, 2)] * shots), g, U_streams=Us)
    kern = FockSLOSKernel(2, 4, [1, 0, 1, 0], dtype=C128)
    pats = torch.as_tensor(kern.layers.final_occupations())
    for half, Uh in ((occ[: shots // 2], Ut), (occ[shots // 2:], U2)):
        p = kern.probs(Uh).to(torch.float64)
        assert _tvd_to_law(half, pats, p) < 3 * 0.5 * math.sqrt(len(p) / (shots // 2))
    assert torch.equal(sk.fock_sample(Ut, torch.zeros((5, 0), dtype=torch.long), g), torch.zeros((5, 4), dtype=torch.long))
