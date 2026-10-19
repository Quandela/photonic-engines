"""Non-linear boson sampling (photonic_engines/nonlinear.py, docs/NONLINEAR.md): the two oracles, the branch
identity and its reductions, the two strong routes, and the rejection sampler against the strong distribution."""
import math
from itertools import permutations

import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines.nonlinear import NonlinearCircuit, TwoModeNetwork, _glynn, branches, kerr, site_phases
from photonic_engines.slos import FockSLOSKernel
from photonic_engines.slos.reference import (
    fock_space_unitary,
    nonlinear_branch_amplitudes,
    nonlinear_state_amplitudes,
    permanent,
)

EPS = 1e-13                                            # complex128 identities on a few hundred terms


def _occ(in_modes, m):
    o = [0] * m
    for j in in_modes:
        o[j] += 1
    return o


def _vec_sites(sites, n):
    return [(lay, x, site_phases(f, n)) for lay, x, f in sites]


CASES = [                                              # (m, in_modes, number of unitaries, sites)
    (4, [0, 1, 2], 2, [(1, 2, 0.7)]),
    (5, [0, 1, 3], 2, [(1, 2, math.pi / 2), (1, 4, [0.1, 0.5, -0.3, 2.0])]),              # same layer
    (5, [0, 1, 3], 3, [(1, 2, 0.7), (2, 2, math.pi / 4)]),                                 # sequential, same mode
    (4, [0, 0, 2], 2, [(1, 0, 1.3)]),                                                     # bunched input
]


@pytest.mark.parametrize(("m", "in_modes", "nu", "sites"), CASES)
def test_oracles_agree(m, in_modes, nu, sites):
    """Oracle (a) (dense Fock space, gate by gate) equals oracle (b) (branch sum of permanents on the n+1 grid)."""
    Us = [haar_unitary(m, 10 + i) for i in range(nu)]
    vs = _vec_sites(sites, len(in_modes))
    a = nonlinear_state_amplitudes(Us, _occ(in_modes, m), vs)
    b = nonlinear_branch_amplitudes(Us, _occ(in_modes, m), vs)
    assert float((a - b).abs().max()) < EPS
    assert float((a.abs() ** 2).sum()) == pytest.approx(1.0, abs=EPS)


@pytest.mark.parametrize(("m", "in_modes", "nu", "sites"), CASES)
@pytest.mark.parametrize("route", ["branch", "gate", "auto"])
def test_strong_routes_equal_the_oracle(m, in_modes, nu, sites, route):
    Us = [haar_unitary(m, 10 + i) for i in range(nu)]
    C = NonlinearCircuit(Us, in_modes, sites)
    ref = nonlinear_state_amplitudes(Us, _occ(in_modes, m), _vec_sites(sites, len(in_modes)))
    assert float((C.amplitudes(route) - ref).abs().max()) < EPS
    assert float(C.probs(route).sum()) == pytest.approx(1.0, abs=EPS)


def test_branch_chunks_do_not_change_the_sum():
    Us = [haar_unitary(5, 1), haar_unitary(5, 2)]
    C = NonlinearCircuit(Us, [0, 1, 3], [(1, 2, 0.7), (1, 4, 0.3)])
    assert C.n_branches > 4
    assert float((C.amplitudes("branch", chunk=3) - C.amplitudes("branch")).abs().max()) < EPS


def test_linear_reductions():
    """phi = 0 is the identity, phi = pi the phase shifter pi on the site (n^2 = n mod 2): one branch, and the
    circuit equals the linear SLOS run on U_1 D U_0 to machine precision."""
    m, t = 5, [0, 1, 2, 3]
    U0, U1 = haar_unitary(m, 3), haar_unitary(m, 4)
    k = FockSLOSKernel(4, m, _occ(t, m), dtype=torch.complex128)
    for phi, ph in ((0.0, 1.0), (math.pi, -1.0), (2 * math.pi, 1.0)):
        C = NonlinearCircuit([U0, U1], t, [(1, 2, phi)])
        assert C.n_branches == 1 and C.rejection_factor() == pytest.approx(1.0, abs=EPS)
        D = torch.ones(m, dtype=torch.complex128); D[2] = ph
        ref = k.probs(U1 @ (D[:, None] * U0)).to(torch.float64)
        for route in ("branch", "gate"):
            assert float((C.probs(route) - ref).abs().max()) < EPS


def test_idle_site_is_the_identity():
    """A site on a mode no photon can reach: U_0 leaves mode x alone and x is not an input."""
    m, x, t = 5, 4, [0, 1, 2]
    U0 = torch.eye(m, dtype=torch.complex128)
    U0[:x, :x] = haar_unitary(x, 5)
    U1 = haar_unitary(m, 6)
    C = NonlinearCircuit([U0, U1], t, [(1, x, [0.3, 1.1, -2.0, 0.4])])
    ref = FockSLOSKernel(3, m, _occ(t, m), dtype=torch.complex128).probs(U1 @ U0).to(torch.float64)
    assert float((C.probs("branch") - ref).abs().max()) < EPS
    assert float((C.probs("gate") - ref).abs().max()) < EPS


@pytest.mark.parametrize("n", [1, 4, 9])
def test_branch_expansion(n):
    rng = np.random.default_rng(n)
    j = np.arange(n + 1)
    two, four = (1, 1) if n == 1 else (2, 4)             # one photon: every site is a phase shifter
    cases = {                                          # f -> expected number of branches (None: any <= n+1)
        0.0: 1, math.pi: 1, math.pi / 2: two, math.pi / 4: four,
        0.37: None, "linear": 1, "random": None, "chi n(n-1)": two}     # chi n(n-1) = Kerr pi/2 up to a linear phase
    for f, want in cases.items():
        fv = {"linear": 0.81 * j, "random": rng.uniform(-3, 3, n + 1), "chi n(n-1)": math.pi / 2 * j * (j - 1)}.get(f, f)
        th, c = branches(fv, n)
        g = np.exp(-1j * site_phases(fv, n))
        assert np.abs(np.exp(1j * np.outer(j, th)) @ c - g).max() < 1e-12
        assert len(th) <= n + 1 and (want is None or len(th) == want), (f, len(th))
        assert np.abs(c).sum() ** 2 <= n + 1 + 1e-9        # |c|_1 <= sqrt(P) |c|_2 = sqrt(P) (Parseval)
    assert np.allclose(kerr(0.5, 3), [0, 0.5, 2.0, 4.5])


def test_two_mode_network_equals_the_dense_representation():
    m, n = 4, 3
    k = FockSLOSKernel(n, m, [1, 1, 1, 0], dtype=torch.complex128)
    net = TwoModeNetwork(torch.as_tensor(k.layers.final_occupations()))
    g = torch.Generator().manual_seed(0)
    psi = torch.randn(net.S, generator=g, dtype=torch.float64) + 1j * torch.randn(net.S, generator=g, dtype=torch.float64)
    U = haar_unitary(m, 7)
    assert float((net.apply(U, psi) - fock_space_unitary(U, n) @ psi).abs().max()) < EPS
    d, gates = TwoModeNetwork.decompose(U)                   # U = G_1 .. G_N diag(d)
    R = torch.eye(m, dtype=torch.complex128)
    for a, G in gates:
        E = torch.eye(m, dtype=torch.complex128); E[a:a + 2, a:a + 2] = G
        R = R @ E
    assert float((R @ torch.diag(d) - U).abs().max()) < EPS


def test_glynn_and_branch_amplitudes():
    g = torch.Generator().manual_seed(1)
    M = torch.randn(3, 5, 5, generator=g, dtype=torch.float64) + 1j * torch.randn(3, 5, 5, generator=g, dtype=torch.float64)
    ref = torch.stack([permanent(M[i]) for i in range(3)])
    assert float((_glynn(M) - ref).abs().max()) < 1e-12 * float(ref.abs().max())
    assert float(_glynn(M[:, :0, :0]).abs().sub(1).abs().max()) == 0.0
    C = NonlinearCircuit([haar_unitary(5, 1), haar_unitary(5, 2)], [0, 1, 3], [(1, 2, 0.7)])
    occ = torch.tensor([[1, 1, 1, 0, 0], [0, 3, 0, 0, 0], [2, 0, 0, 0, 1]])
    a = C.branch_amplitudes(occ)
    Ub, _ = C.branch_table()
    for s in range(3):
        rows = [r for r, v in enumerate(occ[s].tolist()) for _ in range(v)]
        for b in range(Ub.shape[0]):
            assert abs(complex(a[s, b]) - complex(permanent(Ub[b][rows][:, [0, 1, 3]]))) < EPS


def _slos_base(n, m, in_modes=None):
    """A linear boson sampler from the SLOS table: the proposal of the rejection sampler without the kernels.
    Any column set (the species of a partition carry their own), one cached kernel per input occupation."""
    cache: dict = {}

    def base(U, cols, shots, generator):
        key = tuple(cols)
        if key not in cache:
            k = FockSLOSKernel(len(cols), m, _occ(cols, m), dtype=torch.complex128)
            cache[key] = (k, torch.as_tensor(k.layers.final_occupations()))
        k, occ = cache[key]
        out = [occ[torch.multinomial(k.probs(U[b]).to(torch.float64), int(shots[b]), replacement=True, generator=generator)]
               for b in range(U.shape[0]) if int(shots[b])]
        return torch.cat(out) if out else torch.zeros(0, m, dtype=torch.int64)
    return base


def _tvd_and_floor(smp, J, occ, shots):
    idx = {tuple(o): i for i, o in enumerate(occ.tolist())}
    emp = torch.zeros_like(J)
    rows, cnt = torch.unique(smp, dim=0, return_counts=True)
    for o, c in zip(rows.tolist(), cnt.tolist(), strict=True):
        emp[idx[tuple(o)]] = c / shots
    floor = 0.5 * math.sqrt(2 / math.pi) * float((J * (1 - J) / shots).sqrt().sum())
    return 0.5 * float((emp - J).abs().sum()), floor


@pytest.mark.parametrize(("nu", "sites"), [(2, [(1, 2, 0.9)]), (2, [(1, 2, math.pi / 2), (1, 4, 0.4)]),
                                           (3, [(1, 1, math.pi / 2), (2, 1, 0.6)])])
def test_rejection_sampler_matches_the_strong_distribution(nu, sites):
    """TVD within 2.5x its shot-noise expectation, and the acceptance equals 1 / |c|_1^2 within 4 sigma."""
    m, t, shots = 5, [0, 1, 3], 40000
    C = NonlinearCircuit([haar_unitary(m, 20 + i) for i in range(nu)], t, sites)
    g = torch.Generator().manual_seed(3)
    smp = C.sample(shots, generator=g, base_sampler=_slos_base(3, m))
    assert smp.shape == (shots, m) and bool((smp.sum(1) == 3).all())
    tvd, floor = _tvd_and_floor(smp, C.probs(), C.occupations(), shots)
    assert tvd < 2.5 * floor, (tvd, floor)
    acc, P = 1 / C.rejection_factor(), C.stats["proposals"]
    assert abs(C.stats["accepted"] / P - acc) < 4 * math.sqrt(acc * (1 - acc) / P)


def test_strong_method_and_linear_shortcut():
    m, t = 5, [0, 1, 3]
    C = NonlinearCircuit([haar_unitary(m, 1), haar_unitary(m, 2)], t, [(1, 2, 0.9)])
    g = torch.Generator().manual_seed(4)
    tvd, floor = _tvd_and_floor(C.sample(20000, generator=g, method="strong"), C.probs(), C.occupations(), 20000)
    assert tvd < 2.5 * floor
    lin = NonlinearCircuit([haar_unitary(m, 1), haar_unitary(m, 2)], t, [(1, 2, math.pi)])
    smp = lin.sample(10, generator=g, base_sampler=_slos_base(3, m))
    assert smp.shape == (10, m) and lin.stats["proposals"] == 10


def test_random_order_chain_is_not_exact():
    """The docs/NONLINEAR.md §4 verdict on a 3-photon instance: the Clifford & Clifford chain with random photon
    order, run on the branch sum (step k weighs |sum_b c_b per(U_b[r_<=k, sigma_<=k])|^2), does not reproduce the
    circuit - while it does for the linear circuit (the check of the enumeration itself)."""
    m, t = 4, [0, 1, 2]
    for phi, exact in ((0.9, False), (0.0, True)):
        C = NonlinearCircuit([haar_unitary(m, 30), haar_unitary(m, 31)], t, [(1, 2, phi)])
        Ub, c = C.branch_table()
        q: dict = {}
        for sig in permutations(range(3)):
            def rec(r, pr, sig=sig):
                k = len(r)
                if k == 3:
                    key = tuple(np.bincount(r, minlength=m).tolist()); q[key] = q.get(key, 0.0) + pr / 6
                    return
                cols = [t[j] for j in sig[:k + 1]]
                w = [abs(sum(complex(c[b]) * complex(permanent(Ub[b][[*r, i]][:, cols])) for b in range(len(c)))) ** 2
                     for i in range(m)]
                for i in range(m):
                    if w[i] > 0:
                        rec([*r, i], pr * w[i] / sum(w))
            rec([], 1.0)
        P = C.probs()
        tvd = 0.5 * sum(abs(q.get(tuple(o), 0.0) - float(P[i])) for i, o in enumerate(C.occupations().tolist()))
        assert (tvd < 1e-12) if exact else (tvd > 1e-2), (phi, tvd)


def test_validation():
    U = haar_unitary(4, 1)
    with pytest.raises(ValueError, match="layer"):
        NonlinearCircuit([U, U], [0, 1], [(2, 1, 0.3)])
    with pytest.raises(ValueError, match="two sites"):
        NonlinearCircuit([U, U], [0, 1], [(1, 1, 0.3), (1, 1, 0.2)])
    with pytest.raises(ValueError, match="phase vector"):
        NonlinearCircuit([U, U], [0, 1, 2], [(1, 1, [0.0, 1.0])])
    with pytest.raises(ValueError, match="mode"):
        NonlinearCircuit([U, U], [0, 1], [(1, 7, 0.3)])
    with pytest.raises(ValueError, match="route"):
        NonlinearCircuit([U, U], [0, 1], [(1, 1, 0.3)]).amplitudes("nope")
    C = NonlinearCircuit([U, U], [0, 0], [(1, 1, 0.3)])
    with pytest.raises(ValueError, match="one photon per input mode"):
        C.sample(3)
    with pytest.raises(NotImplementedError, match="strong side is lossless"):
        NonlinearCircuit([U, U], [0, 1], [(1, 1, 0.3)]).sample(3, method="strong", loss=0.5)


@pytest.mark.slow
def test_kernel_path_under_the_interpreter():
    """The GPU path on the Triton CPU interpreter: branch permanents by the cofactor kernel, proposals by fock_sample."""
    pytest.importorskip("triton")
    m, t = 5, [0, 1, 3]
    C = NonlinearCircuit([haar_unitary(m, 1), haar_unitary(m, 2)], t, [(1, 2, 0.9)])
    occ = torch.tensor([[1, 1, 1, 0, 0], [0, 3, 0, 0, 0], [2, 0, 0, 0, 1]])
    assert float((C.branch_amplitudes(occ, "triton") - C.branch_amplitudes(occ, "torch")).abs().max()) < EPS
    g = torch.Generator().manual_seed(5)
    smp = C.sample(30, generator=g, backend="triton", amp_backend="triton")
    assert smp.shape == (30, m) and bool((smp.sum(1) == 3).all())


def test_true_marginal_has_subset_coherences():
    """docs/NONLINEAR.md §4: the law of the first k outputs (random photon order) is
    (n-k)!/n! sum_{b,b'} c_b conj(c_b') sum_{|S|=|T|=k} per(A_b[r, S]) conj(per(A_b'[r, T])) per(G_bb'[T^c, S^c]),
    G_bb' = (U_b'^dag U_b)[t, t]; the S != T terms (absent for a linear circuit, G = 1) are needed."""
    from itertools import combinations
    m, t, n = 4, [0, 1, 2], 3
    C = NonlinearCircuit([haar_unitary(m, 30), haar_unitary(m, 31)], t, [(1, 2, 0.9)])
    Ub, c = C.branch_table()
    P, occ = C.probs(), C.occupations().tolist()
    ordered: dict = {}                                                    # exchangeable ordered law, then its marginals
    for pr, o in zip(P.tolist(), occ, strict=True):
        rows = [r for r, v in enumerate(o) for _ in range(v)]
        w = pr * math.prod(math.factorial(v) for v in o) / math.factorial(n)
        for r in set(permutations(rows)):
            ordered[r] = w
    for k in (1, 2):
        marg: dict = {}
        for r, w in ordered.items():
            marg[r[:k]] = marg.get(r[:k], 0.0) + w
        worst = {True: 0.0, False: 0.0}
        for r, want in marg.items():
            for diag in (True, False):
                tot = 0.0 + 0.0j
                for b in range(len(c)):
                    for bp in range(len(c)):
                        G = (Ub[bp].conj().T @ Ub[b])[t][:, t]
                        for S in combinations(range(n), k):
                            for T in combinations(range(n), k):
                                if diag and S != T:
                                    continue
                                Sc, Tc = [j for j in range(n) if j not in S], [j for j in range(n) if j not in T]
                                tot += (complex(c[b] * c[bp].conj()) * complex(permanent(Ub[b][list(r)][:, [t[j] for j in S]]))
                                        * complex(permanent(Ub[bp][list(r)][:, [t[j] for j in T]])).conjugate()
                                        * complex(permanent(G[Tc][:, Sc])))
                val = tot.real * math.factorial(n - k) / math.factorial(n)
                worst[diag] = max(worst[diag], abs(val - want))
        assert worst[False] < EPS and worst[True] > 1e-3, (k, worst)


def _species_probs(C, cols_list, photons):
    """{occupation: probability} for one species structure (docs/NONLINEAR.md §6): the amplitude of a
    species-resolved output is sum_b c_b prod_mu per(U_b[s_mu, t_mu]) / sqrt(prod s_mu!), squared and summed
    into the total occupation. Enumerates every species-resolved output: tiny n only."""
    from itertools import product as iproduct

    from photonic_engines.slos.reference import _multisets
    Ub, c = C.branch_table(photons)
    tabs = []
    for cols in cols_list:
        ts = list(_multisets(C.m, len(cols)))
        tab = torch.stack([torch.stack([permanent(Ub[b][[r for r, v in enumerate(t) for _ in range(v)]][:, list(cols)])
                                        for b in range(len(c))]) for t in ts])          # [T, B]
        tabs.append((ts, tab, [math.prod(math.factorial(v) for v in t) for t in ts]))
    out: dict = {}
    for choice in iproduct(*[range(len(ts)) for ts, _, _ in tabs]):
        amp = torch.ones(len(c), dtype=torch.complex128)
        nrm, total = 1.0, np.zeros(C.m, dtype=np.int64)
        for (ts, tab, norms), ti in zip(tabs, choice, strict=True):
            amp = amp * tab[ti]; nrm *= norms[ti]; total += np.asarray(ts[ti])
        key = tuple(int(v) for v in total)
        out[key] = out.get(key, 0.0) + float(abs(complex(amp @ c)) ** 2) / nrm
    return out


def _imperfect_probs(C, eta, x, eta_post=1.0):
    """The enumerated mixture over loss / species partitions, then binomial thinning of the counts: the oracle
    of `sample(loss=, indist=, eta_post=)`."""
    from itertools import product as iproduct
    n = C.n
    out: dict = {}
    for roles in iproduct(["lost", "good", "single"], repeat=n):                        # x-model: good w.p. x
        w = 1.0
        for r in roles:
            w *= (1 - eta) if r == "lost" else eta * (x if r == "good" else 1 - x)
        if w == 0.0:
            continue
        good = tuple(C.in_modes[j] for j in range(n) if roles[j] == "good")
        cols_list = ([good] if good else []) + [(C.in_modes[j],) for j in range(n) if roles[j] == "single"]
        photons = sum(len(cols) for cols in cols_list)
        if photons == 0:
            out[(0,) * C.m] = out.get((0,) * C.m, 0.0) + w
            continue
        for occ, p in _species_probs(C, cols_list, photons).items():
            out[occ] = out.get(occ, 0.0) + w * p
    if eta_post >= 1.0:
        return out
    thinned: dict = {}
    for occ, p in out.items():
        for keep in iproduct(*[range(v + 1) for v in occ]):
            k, tot = sum(keep), sum(occ)
            wk = math.prod(math.comb(occ[i], keep[i]) for i in range(len(occ))) * eta_post ** k * (1 - eta_post) ** (tot - k)
            thinned[keep] = thinned.get(keep, 0.0) + p * wk
    return thinned


@pytest.mark.parametrize(("eta", "x", "eta_post"), [(1.0, 1.0, 1.0), (0.7, 1.0, 1.0), (1.0, 0.8, 1.0),
                                                    (0.8, 0.7, 1.0), (1.0, 1.0, 0.8), (0.9, 0.9, 0.9)])
def test_sampler_with_loss_and_distinguishability(eta, x, eta_post):
    """The sampler against the enumerated mixture over partitions; eta = x = 1 is the lossless path."""
    m, t, shots = 4, [0, 1, 2], 30000
    C = NonlinearCircuit([haar_unitary(m, 40), haar_unitary(m, 41)], t, [(1, 2, 0.9)])
    ref = _imperfect_probs(C, eta, x, eta_post)
    assert sum(ref.values()) == pytest.approx(1.0, abs=1e-12)
    if (eta, x, eta_post) == (1.0, 1.0, 1.0):
        P, occ = C.probs(), C.occupations().tolist()
        assert max(abs(ref.get(tuple(o), 0.0) - float(P[i])) for i, o in enumerate(occ)) < EPS
    g = torch.Generator().manual_seed(7)
    smp = C.sample(shots, generator=g, base_sampler=_slos_base(3, m), loss=None if eta >= 1 else eta,
                   indist=None if x >= 1 else x, eta_post=None if eta_post >= 1 else eta_post)
    keys = sorted(set(ref) | {tuple(r) for r in smp.tolist()})
    J = torch.tensor([ref.get(k, 0.0) for k in keys], dtype=torch.float64)
    tvd, floor = _tvd_and_floor(smp, J, torch.tensor(keys, dtype=torch.int64), shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_loss_shrinks_the_expansion():
    """With loss a stream uses the expansion of its own photon count: never more branches than the n-photon one."""
    C = NonlinearCircuit([haar_unitary(6, 1), haar_unitary(6, 2)], [0, 1, 2, 3], [(1, 4, 0.7)])
    sizes = [C.branch_table(k)[1].shape[0] for k in range(1, 5)]
    assert sizes == sorted(sizes) and sizes[-1] == C.n_branches
    assert C.rejection_factor(1) == pytest.approx(1.0, abs=1e-12)       # one photon: the site is a phase shifter


def test_partitions_are_proposed_in_one_batch():
    """The lossy / distinguishable path works per (species size, photon count, slot) class, never per partition:
    the calls follow the classes, not the partitions, and one call carries many circuits (docs/NONLINEAR.md §6c
    - doing either per partition prices launch latency instead of the algorithm)."""
    m, t = 8, [0, 1, 2, 3, 4]
    C = NonlinearCircuit([haar_unitary(m, 40), haar_unitary(m, 41)], t, [(1, 5, 0.9)])
    inner, calls = _slos_base(5, m), []

    def counting(U, cols, shots, generator):
        calls.append((U.shape[0], len(cols), int(shots.sum())))
        return inner(U, cols, shots, generator)
    g = torch.Generator().manual_seed(11)
    C.sample(400, generator=g, base_sampler=counting, loss=0.7, indist=0.8)
    assert C.stats["groups"] > 100                                  # many distinct partitions
    assert len(calls) < C.stats["groups"] / 2, (len(calls), C.stats)         # calls follow the classes, not the partitions
    assert max(c[0] for c in calls) > 5, calls                      # one call covers many partitions at once
    assert sum(c[2] for c in calls) > C.stats["proposals"]          # at least one species drawn per proposal


@pytest.mark.parametrize(("eta", "x"), [(None, None), (0.7, None), (0.7, 0.8)])
def test_proposals_per_sample_is_the_rejection_factor(eta, x):
    """A partition is proposed need x |c|_1^2 times and no more: a fixed slack per group is paid per STREAM
    under loss, and cost the measured lossy frontier ~5 photons (docs/NONLINEAR.md §6c)."""
    m, t, shots = 4, [0, 1, 2], 2000
    C = NonlinearCircuit([haar_unitary(m, 40), haar_unitary(m, 41)], t, [(1, 2, 0.9)])
    g = torch.Generator().manual_seed(5)
    C.sample(shots, generator=g, base_sampler=_slos_base(3, m), loss=eta, indist=x)
    assert C.stats["proposals"] / shots < 1.25 * C.stats["rejection_factor"], C.stats
