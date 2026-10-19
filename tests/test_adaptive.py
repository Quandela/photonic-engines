"""Adaptive linear optics (`photonic_engines.adaptive`, docs/ADAPTIVE.md).

Oracles, all independent of the code under test: `slos.reference.permanent` on a cumulative
unitary composed HERE (not by `AdaptiveCircuit.unitary`), and the non-adaptive SLOS distribution
summed over the traced modes for the marginals. The samplers are checked against the strong joint
once that one is pinned to the permanents; the rejection logic runs on an SLOS-backed base sampler
(exact, fast on CPU), the Triton Clifford sampler only for a structural check. The forced-prefix final
stage has its own CPU oracle-backed sampler, `_chain`: the Clifford & Clifford chain written with Ryser
permanents, prefix entry included - the same construction as `clifford.sampler`, not the same code.
"""
import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines.adaptive import AdaptiveCircuit, binomial_moments, binomial_moments_blocks, mode_marginals
from photonic_engines.clifford.sampler import _ryser, prefix_columns, prefix_columns_many, prefix_subset_weights
from photonic_engines.slos import FockSLOSKernel
from photonic_engines.slos.reference import permanent

EPS = torch.finfo(torch.float64).eps
N, M, T_IN = 3, 6, [0, 2, 4]
LEVELS = [[1], [0, 3]]                                     # Lambda = 2, l = (1, 2), K = 3


def _policy(lam, p):
    """A different Haar unitary on the unmeasured modes for every outcome prefix."""
    size = M - (1 if lam == 1 else 3)
    return haar_unitary(size, 100 + 7 * lam + sum(x * 11 ** i for i, x in enumerate(p)))


def _circuit(policy=_policy):
    return AdaptiveCircuit(haar_unitary(M, 1), T_IN, LEVELS, policy)


def _slos(n=N, m=M, t=T_IN):
    k = FockSLOSKernel(n, m, [1 if j in t else 0 for j in range(m)], dtype=torch.complex128)
    return k, torch.as_tensor(k.layers.final_occupations())


def _tol(n):
    return 4096 * EPS * 2 ** n                             # a Ryser / SLOS sum of 2^n terms of order one


@pytest.mark.parametrize("modes", [[1], [0, 3], [5, 2, 1]])
def test_mode_marginals_equal_the_slos_distribution_summed_over_the_other_modes(modes):
    U = haar_unitary(M, 2)
    k, occ = _slos()
    ref = torch.zeros((N + 1,) * len(modes), dtype=torch.float64)
    for w, o in zip(k.probs(U).to(torch.float64), occ, strict=True):
        ref[tuple(o[modes].tolist())] += w
    got = mode_marginals(U, T_IN, modes)
    assert torch.allclose(got, ref, rtol=0, atol=_tol(N))
    assert float(got.sum()) == pytest.approx(1.0, abs=_tol(N))


def test_binomial_moments_first_order_is_the_mean_photon_number():
    U = haar_unitary(M, 3)
    T = binomial_moments(U, T_IN, [4, 1])
    mean = (U[[4, 1]][:, T_IN].abs() ** 2).sum(1)          # E[N_l] = sum_i |U[l, t_i]|^2
    assert float(T[0, 0]) == pytest.approx(1.0, abs=64 * EPS)
    assert float(T[1, 0]) == pytest.approx(float(mean[0]), abs=64 * EPS)
    assert float(T[0, 1]) == pytest.approx(float(mean[1]), abs=64 * EPS)


def test_the_graded_table_reaches_three_modes_at_the_frontier_size():
    """n = 38, L = 3: 5e6 graded entries where the dense table had 3.5e9; and a (L+1)-mode table marginalises to it."""
    from photonic_engines.adaptive import graded_entries
    assert graded_entries(38, 3) < 1e7 < 39 ** 6
    U = haar_unitary(24, 21)
    t = list(range(12))
    three, four = mode_marginals(U, t, [23, 22, 21]), mode_marginals(U, t, [23, 22, 21, 5])
    assert float(three.sum()) == pytest.approx(1.0, abs=1e-12)
    assert torch.allclose(four.sum(3), three, rtol=0, atol=1e-12)
    assert torch.allclose(three.sum((1, 2)), mode_marginals(U, t, [23]), rtol=0, atol=1e-12)


def test_mode_marginals_refuse_a_bunched_input():
    with pytest.raises(ValueError):
        mode_marginals(haar_unitary(4, 0), [0, 0], [1])


def test_joint_equals_one_permanent_per_outcome_on_the_branch_unitary():
    C = _circuit()
    U0 = haar_unitary(M, 1)
    J, occ = C.probs(), C.occupations()
    assert float(J.sum()) == pytest.approx(1.0, abs=_tol(N))
    for w, o in zip(J, occ.tolist(), strict=True):
        p = (o[1], o[0], o[3])
        U = U0.clone()
        r1 = [0, 2, 3, 4, 5]; U[r1] = _policy(1, p[:1]) @ U[r1]
        r2 = [2, 4, 5]; U[r2] = _policy(2, p) @ U[r2]
        rows = [j for j, c in enumerate(o) for _ in range(c)]
        amp = permanent(U[rows][:, T_IN]) / math.sqrt(math.prod(math.factorial(c) for c in o))
        assert float(w) == pytest.approx(float(amp.abs() ** 2), abs=_tol(N))


def test_identity_policy_reduces_to_plain_boson_sampling():
    k, _ = _slos()
    C = AdaptiveCircuit(haar_unitary(M, 1), T_IN, LEVELS, policy=None)
    assert torch.allclose(C.probs(), k.probs(haar_unitary(M, 1)).to(torch.float64), rtol=0, atol=_tol(N))


def test_outcome_probs_and_final_marginal_equal_the_joint_summed():
    C = _circuit()
    J, occ = C.probs(), C.occupations().tolist()
    ref_p, ref_f = {}, torch.zeros(N + 1, N + 1, dtype=torch.float64)
    for w, o in zip(J, occ, strict=True):
        ref_p[(o[1], o[0], o[3])] = ref_p.get((o[1], o[0], o[3]), 0.0) + float(w)
        ref_f[o[2], o[5]] += w
    got = C.outcome_probs()
    assert set(got) == {p for p, w in ref_p.items() if w > 0}
    assert max(abs(got[p] - ref_p[p]) for p in got) < _tol(N)
    assert torch.allclose(C.final_marginal([2, 5]), ref_f, rtol=0, atol=len(got) * _tol(N))
    first = C.outcome_probs(lam=1)                         # the level-1 law does not see V_1, V_2
    assert first[(1,)] == pytest.approx(sum(w for p, w in ref_p.items() if p[0] == 1), abs=_tol(N))


def _tvd_and_floor(smp, J, occ, shots):
    idx = {tuple(o): i for i, o in enumerate(occ.tolist())}
    emp = torch.zeros_like(J)
    rows, cnt = torch.unique(smp, dim=0, return_counts=True)
    for o, c in zip(rows.tolist(), cnt.tolist(), strict=True):
        emp[idx[tuple(o)]] = c / shots
    floor = 0.5 * math.sqrt(2 / math.pi) * float((J * (1 - J) / shots).sqrt().sum())
    return 0.5 * float((emp - J).abs().sum()), floor


def _chain(U, in_modes, shots, generator, prefix, avoid=None):
    """Clifford & Clifford, algorithm B, on the CPU with explicit permanents; `prefix` forces the first rows.
    Step k draws row k with weight |per(A[rows_{<=k}, sig_{<=k}])|^2; a forced prefix of c rows draws the
    column set of the first c steps from its posterior |per(A[prefix, S])|^2 and starts at step c + 1.
    The library's batched base protocol (`AdaptiveCircuit._forced`): U [G, m, k] with `shots` an int tensor
    [G] and `prefix` [G, c] -> (occ of the surviving streams, their stream ids), a stream surviving when no
    photon after the prefix landed on `avoid` (every chain runs to the end here; the library aborts).
    """
    if U.dim() == 3:
        occs, ids, start = [], [], 0
        for g in range(U.shape[0]):
            reps = int(shots[g])
            pre = [] if prefix is None else prefix[g].tolist()
            occ = _chain(U[g], in_modes, reps, generator, pre)
            if avoid is not None:
                extra = occ[:, list(avoid)].sum(1) - len([r for r in pre if r in set(avoid)])
                ok = extra == 0
            else:
                ok = torch.ones(reps, dtype=torch.bool)
            occs.append(occ[ok]); ids.append(start + torch.nonzero(ok)[:, 0]); start += reps
        return torch.cat(occs), torch.cat(ids)
    A = U[:, list(in_modes)].to(torch.complex128)                                   # [m, n]
    m, n = A.shape
    prefix = list(prefix or [])
    c = len(prefix)
    key = torch.rand(shots, n, generator=generator)
    if c:
        sub, w = prefix_subset_weights(A[None], prefix)
        pick = torch.multinomial(w[0], shots, replacement=True, generator=generator)
        key[torch.arange(shots)[:, None], sub[pick]] = -1.0
    sig = torch.argsort(key, dim=1)                                                 # [S, n]
    rows = torch.zeros(shots, n, dtype=torch.int64)
    rows[:, :c] = torch.tensor(prefix, dtype=torch.int64)
    for k in range(c + 1, n + 1):
        cand = torch.cat([rows[:, None, :k - 1].expand(shots, m, k - 1),
                          torch.arange(m)[None, :, None].expand(shots, m, 1)], dim=2)      # [S, m, k] candidate row sets
        blocks = A[cand[:, :, :, None], sig[:, None, None, :k]]                     # [S, m, k, k]
        rows[:, k - 1] = torch.multinomial(_ryser(blocks).abs() ** 2, 1, generator=generator)[:, 0]
    return torch.zeros(shots, m, dtype=torch.int64).scatter_add(1, rows, torch.ones_like(rows))


def test_chain_oracle_is_a_boson_sampler():
    """The CPU chain used below as base sampler, without prefix, against SLOS."""
    U = haar_unitary(M, 12)
    k, kocc = _slos()
    shots = 20000
    smp = _chain(U, T_IN, shots, torch.Generator().manual_seed(2), None)
    tvd, floor = _tvd_and_floor(smp, k.probs(U).to(torch.float64), kocc, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_prefix_weights_sum_to_the_binomial_moment():
    """sum_S |per(A[a^p, S])|^2 = prod_l p_l! * T[p]: two independent routes to E[prod_l C(N_l, p_l)]."""
    U = haar_unitary(M, 13)
    T = binomial_moments(U, T_IN, [1, 3])
    for p in [(1, 0), (2, 0), (1, 1), (2, 1), (0, 3)]:
        prefix = [1] * p[0] + [3] * p[1]
        _, w = prefix_subset_weights(U[:, T_IN][None], prefix)
        want = float(T[p]) * math.factorial(p[0]) * math.factorial(p[1])
        assert float(w.sum()) == pytest.approx(want, abs=_tol(N))


@pytest.mark.parametrize(("prefix", "modes", "counts"), [([1], [1], (1,)), ([1, 1], [1], (2,)), ([1, 3], [1, 3], (1, 1)),
                                                         ([3, 3, 1], [1, 3], (1, 2))])
def test_forced_prefix_gives_the_conditional_law_at_acceptance_P_over_T(prefix, modes, counts):
    """The claim the adaptive sampler rests on, in isolation: force the first photons of the chain onto `modes`,
    keep the draws where no other photon joins them -> the SLOS law conditioned on exactly `counts`, accepted
    with probability P(counts) / T[counts]."""
    U = haar_unitary(M, 14)
    k, kocc = _slos()
    shots = 20000
    smp = _chain(U, T_IN, shots, torch.Generator().manual_seed(4), prefix)
    keep = (smp[:, modes] == torch.tensor(counts)).all(1)
    ratio = float(mode_marginals(U, T_IN, modes)[counts] / binomial_moments(U, T_IN, modes)[counts])
    sigma = math.sqrt(max(ratio * (1 - ratio), 1e-12) / shots)
    assert abs(float(keep.double().mean()) - ratio) < 4 * sigma + 1 / shots
    ref = k.probs(U).to(torch.float64) * (kocc[:, modes] == torch.tensor(counts)).all(1)
    tvd, floor = _tvd_and_floor(smp[keep], ref / ref.sum(), kocc, int(keep.sum()))
    assert tvd <= 2.5 * floor, (tvd, floor)                # equality: all n photons forced, one outcome, both zero


@pytest.mark.parametrize("chain", ["marginal", "rejection"])
def test_forced_prefix_sampler_reproduces_the_joint_and_costs_the_moment_sum(chain):
    C = _circuit()
    J, occ = C.probs(), C.occupations()
    shots = 20000
    smp = C.sample(shots, generator=torch.Generator().manual_seed(8), chain=chain, final="prefix", base_sampler=_chain)
    assert smp.sum(1).tolist() == [N] * shots
    tvd, floor = _tvd_and_floor(smp, J, occ, shots)
    assert tvd < 2.5 * floor, (tvd, floor)
    if chain == "marginal":                                # mean chains per sample = sum_p T[p] = E[2^r], not the branch count
        mean = sum(float(C._level_tables(C.n_levels, p[:C.K_upto[-2]])[1][p[C.K_upto[-2]:]]) for p in C.outcome_probs())
        assert C.stats["base_shots"] / shots == pytest.approx(mean, rel=0.05)
        assert mean < 0.5 * len(C.outcome_probs())


@pytest.mark.parametrize("prefix", [[5], [5, 5], [5, 2], [2, 5, 5, 11], [11, 11, 11, 2, 2]])
def test_prefix_columns_draws_the_enumerated_posterior(prefix):
    """The polynomial column-by-column sampler against the C(n, c) permanents it replaces."""
    A = haar_unitary(12, 3)[:, [0, 1, 3, 4, 6, 8, 9]]
    sub, w = prefix_subset_weights(A[None], prefix)
    ref = w[0] / w[0].sum()
    shots = 40000
    chosen = prefix_columns(A, prefix, shots, torch.Generator().manual_seed(1))
    assert chosen.sum(1).tolist() == [len(prefix)] * shots
    index = {tuple(c): i for i, c in enumerate(sub.tolist())}
    emp = torch.zeros_like(ref)
    rows, cnt = torch.unique(chosen, dim=0, return_counts=True)
    for r, c in zip(rows, cnt.tolist(), strict=True):
        emp[index[tuple(torch.nonzero(r)[:, 0].tolist())]] = c / shots
    floor = 0.5 * math.sqrt(2 / math.pi) * float((ref * (1 - ref) / shots).sqrt().sum())
    assert 0.5 * float((emp - ref).abs().sum()) < 2.5 * floor


def test_prefix_columns_is_polynomial_in_the_prefix_length():
    """13 forced photons on 38: C(38, 13) = 5e9 subsets, never enumerated."""
    chosen = prefix_columns(haar_unitary(76, 5)[:, :38], [75] * 7 + [74] * 6, 4, torch.Generator().manual_seed(0))
    assert chosen.sum(1).tolist() == [13] * 4


@pytest.mark.parametrize(("chain", "final"), [("marginal", "slos"), ("marginal", "clifford"), ("rejection", "clifford")])
def test_sampler_reproduces_the_joint_distribution(chain, final):
    C = _circuit()
    J, occ = C.probs(), C.occupations()
    k, kocc = _slos()

    def base(U, in_modes, shots, generator):               # an exact boson sampler, from the strong side
        return kocc[torch.multinomial(k.probs(U).to(torch.float64), shots, replacement=True, generator=generator)]

    shots = 20000
    smp = C.sample(shots, generator=torch.Generator().manual_seed(5), chain=chain, final=final, base_sampler=base)
    assert smp.shape == (shots, M) and smp.sum(1).tolist() == [N] * shots
    tvd, floor = _tvd_and_floor(smp, J, occ, shots)
    assert tvd < 2.5 * floor, (tvd, floor)
    if final == "clifford":                                # expected draws per sample = number of branches
        assert C.stats["base_shots"] > shots and C.stats["slos_branches"] == 0
    else:
        assert C.stats["base_shots"] == 0 and C.stats["slos_branches"] == C.stats["branches"]


def test_rare_branches_are_routed_to_the_strong_side():
    C = _circuit()
    k, kocc = _slos()
    base = lambda U, in_modes, shots, generator: kocc[torch.multinomial(
        k.probs(U).to(torch.float64), shots, replacement=True, generator=generator)]
    shots = 20000
    smp = C.sample(shots, generator=torch.Generator().manual_seed(6), final="clifford", base_sampler=base, min_accept=0.05)
    assert 0 < C.stats["slos_branches"] < C.stats["branches"]
    tvd, floor = _tvd_and_floor(smp, C.probs(), C.occupations(), shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_rejection_gives_up_loudly():
    C = _circuit()
    calls = []

    def never(U, in_modes, shots, generator):              # reads p_1 = 1 once, then never shows it again
        calls.append(shots)
        row = [0, 1, 1, 0, 1, 0] if len(calls) == 1 else [0, 0, 1, 0, 1, 1]
        return torch.tensor([row]).repeat(shots, 1)

    with pytest.raises(RuntimeError):
        C.sample(64, generator=torch.Generator().manual_seed(0), chain="rejection", final="clifford", base_sampler=never,
                 max_draws=4096)


def test_bad_policies_and_prefixes_are_refused():
    U0 = haar_unitary(M, 1)
    with pytest.raises(ValueError):
        AdaptiveCircuit(U0, T_IN, [[1], [1]])
    with pytest.raises(ValueError):
        AdaptiveCircuit(U0, [0, 0, 2], [[1]])
    with pytest.raises(ValueError):
        AdaptiveCircuit(U0, T_IN, [[1]], lambda lam, p: torch.eye(3)).unitary((0,))             # wrong size
    with pytest.raises(ValueError):
        AdaptiveCircuit(U0, T_IN, [[1]], lambda lam, p: 2 * torch.eye(5)).unitary((0,))         # not unitary
    with pytest.raises(ValueError):
        _circuit().unitary((0, 1))                                                              # half a level
    with pytest.raises(ValueError):
        _circuit().sample(1, chain="nope")


def test_early_abort_keeps_the_conditional_law():
    """fock_sample(prefix=, avoid=) on the interpreted kernels: every survivor respects the constraint and the
    survival fraction is P / T (n = 2, m = 4: 150 shots at ~40 ms each on the CPU interpreter)."""
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    U, t, modes, counts = haar_unitary(4, 31).to(dev), [0, 2], [3], (1,)
    from photonic_engines.clifford import fock_sample
    shots = 150
    kept = fock_sample(U, t, shots, generator=torch.Generator(device=dev).manual_seed(2), backend="triton", prefix=[3], avoid=modes)
    assert kept.shape[1] == 4 and kept.sum(1).eq(2).all() and kept[:, 3].eq(1).all()
    ratio = float(mode_marginals(U, t, modes)[counts] / binomial_moments(U, t, modes)[counts])
    assert abs(kept.shape[0] / shots - ratio) < 4 * math.sqrt(ratio * (1 - ratio) / shots) + 1 / shots


def test_clifford_backend_runs_end_to_end():
    """The Triton sampler under the CPU interpreter is ~40 ms per shot: structure only, tiny instance.
    On a CUDA host the kernels are compiled and refuse CPU tensors, so the circuit lives on the GPU there
    (the distribution test on the real sampler is `bench/adaptive_frontier.py --check`)."""
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = AdaptiveCircuit(haar_unitary(4, 8).to(dev), [0, 1], [[2]], lambda lam, p: haar_unitary(3, 40 + p[0]).to(dev))
    smp = C.sample(6, generator=torch.Generator(device=dev).manual_seed(1))
    assert smp.shape == (6, 4) and smp.sum(1).tolist() == [2] * 6


# ---- imperfections: the sampler against a mixture over partitions, enumerated here ------------------------------

def _partition_joint(C, good, singles):
    """{omega: P} for one partition of the photons: `good` (input columns) interfere as one instance, `singles` are
    each in a mode of their own, the rest are lost. Enumerated level by level with the SLOS joint of the good
    photons under the cumulative unitary (its marginal on the measured modes is the physical one: later
    feed-forward acts elsewhere) and |U[a, col]|^2 for a single, conditioned on where it has not landed yet.
    """
    A, m = C.adaptive_modes, C.m
    k, kocc = (_slos(len(good), m, list(good)) if good else (None, None))

    def good_law(U, on, prefix):                            # P(counts on `on` = prefix + x) over x, from the SLOS joint
        pr = k.probs(U).to(torch.float64)
        law = {}
        for o, q in zip(kocc.tolist(), pr.tolist(), strict=True):
            key = tuple(o[a] for a in on)
            if key[:len(prefix)] == tuple(prefix):
                law[key[len(prefix):]] = law.get(key[len(prefix):], 0.0) + q
        return law

    # state: (tot prefix, good prefix, landed tuple) -> weight
    states = {((), (), tuple(-1 for _ in singles)): 1.0}
    for lam in range(1, C.n_levels + 1):
        lev, prev = C.levels[lam - 1], A[:C.K_upto[lam - 1]]
        nxt = {}
        for (tot, gp, landed), w in states.items():
            U = C.unitary(tot)
            glaw = good_law(U, A[:C.K_upto[lam]], gp) if good else {tuple(0 for _ in lev): 1.0}
            norm = sum(glaw.values())
            opts = []                                       # per single: [(landing mode or -1, prob)]
            for j, col in enumerate(singles):
                if landed[j] >= 0:
                    opts.append([(landed[j], 1.0)]); continue
                w2 = (U[:, col].abs() ** 2).tolist()
                rest = 1.0 - sum(w2[a] for a in prev)
                o = [(a, w2[a] / rest) for a in lev]
                opts.append([*o, (-1, 1.0 - sum(q for _, q in o))])
            for gx, gq in glaw.items():
                for choice in itertools.product(*opts):
                    cnt = list(gx)
                    for a, _ in choice:
                        if a >= 0 and a in lev:
                            cnt[lev.index(a)] += 1
                    key = (tot + tuple(cnt), gp + gx, tuple(a for a, _ in choice))
                    q = w * gq / norm * math.prod(pq for _, pq in choice)
                    nxt[key] = nxt.get(key, 0.0) + q
        states = nxt
    final = [j for j in range(m) if j not in set(A)]
    out = {}
    for (tot, gp, landed), w in states.items():
        U = C.unitary(tot)
        if good:
            pr = k.probs(U).to(torch.float64)
            sel = [(o, q) for o, q in zip(kocc.tolist(), pr.tolist(), strict=True) if tuple(o[a] for a in A) == gp]
            norm = sum(q for _, q in sel)
            glaw = [(o, q / norm) for o, q in sel]
        else:
            glaw = [([0] * m, 1.0)]
        opts = []
        for j, col in enumerate(singles):
            if landed[j] >= 0:
                opts.append([(landed[j], 1.0)]); continue
            w2 = (U[final, col].abs() ** 2)
            w2 = (w2 / w2.sum()).tolist()
            opts.append(list(zip(final, w2, strict=True)))
        for o, gq in glaw:
            for choice in itertools.product(*opts):
                om = list(o)
                for a, _ in choice:
                    om[a] += 1
                q = w * gq * math.prod(pq for _, pq in choice)
                out[tuple(om)] = out.get(tuple(om), 0.0) + q
    return out


def _imperfect_joint(C, eta, x):
    """The x-model with loss: every photon lost (1 - eta), interfering (eta x) or a singleton (eta (1 - x))."""
    out = {}
    for roles in itertools.product(("lost", "good", "single"), repeat=C.n):
        w = math.prod({"lost": 1 - eta, "good": eta * x, "single": eta * (1 - x)}[r] for r in roles)
        if w == 0:
            continue
        good = [c for c, r in zip(C.in_modes, roles, strict=True) if r == "good"]
        singles = [c for c, r in zip(C.in_modes, roles, strict=True) if r == "single"]
        for om, q in _partition_joint(C, good, singles).items():
            out[om] = out.get(om, 0.0) + w * q
    return out


def _tvd_dict(smp, ref, shots):
    keys = sorted(set(ref) | {tuple(r) for r in smp.tolist()})
    J = torch.tensor([ref.get(k, 0.0) for k in keys], dtype=torch.float64)
    occ = torch.tensor(keys, dtype=torch.int64)
    return _tvd_and_floor(smp, J, occ, shots)


def test_partition_oracle_reduces_to_the_lossless_joint():
    """Sanity of the enumeration: all photons good = the strong joint; and the mixture sums to one."""
    C = _circuit()
    J, occ = C.probs(), C.occupations()
    ref = _partition_joint(C, list(C.in_modes), [])
    assert sum(ref.values()) == pytest.approx(1.0, abs=1e-12)
    for o, q in zip(occ.tolist(), J.tolist(), strict=True):
        assert ref.get(tuple(o), 0.0) == pytest.approx(q, abs=1e-12)
    mix = _imperfect_joint(C, 0.8, 0.7)
    assert sum(mix.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(("eta", "x", "chain"), [(0.7, 1.0, "marginal"), (1.0, 0.6, "marginal"), (0.8, 0.7, "marginal"),
                                                 (0.8, 0.7, "rejection")])
def test_sampler_with_loss_and_distinguishability_reproduces_the_mixture(eta, x, chain):
    """Uniform loss and the x-model on the adaptive sampler, against the enumerated mixture over partitions."""
    C = _circuit()
    ref = _imperfect_joint(C, eta, x)
    shots = 20000
    smp = C.sample(shots, generator=torch.Generator().manual_seed(9), chain=chain, final="prefix", base_sampler=_chain,
                   loss=eta if eta < 1 else None, indist=x if x < 1 else None)
    assert smp.shape == (shots, M)
    if eta < 1:                                             # the photon number is binomial
        mean = float(smp.sum(1).to(torch.float64).mean())
        assert abs(mean - N * eta) < 4 * math.sqrt(N * eta * (1 - eta) / shots)
    else:
        assert smp.sum(1).tolist() == [N] * shots
    tvd, floor = _tvd_dict(smp, ref, shots)
    assert tvd < 2.5 * floor, (tvd, floor)


def test_imperfections_need_the_forced_prefix():
    with pytest.raises(NotImplementedError):
        _circuit().sample(4, generator=torch.Generator().manual_seed(0), final="slos", loss=0.5)


def test_batched_tables_match_the_single_ones_with_padding_and_chunks():
    """`binomial_moments_blocks`: blocks from different unitaries, output modes and photon numbers (padded with zero
    columns) against one `binomial_moments` each, through the chunking path (max_entries forces 2 blocks per chunk)."""
    cases = [(haar_unitary(8, 21), [0, 2, 4, 6], [1, 5]), (haar_unitary(8, 22), [1, 3], [0, 7]), (haar_unitary(8, 23), [0, 1, 2], [4, 6]),
             (haar_unitary(8, 24), [5], [2, 3]), (haar_unitary(8, 25), [0, 3, 5, 7], [1, 2])]
    n_max = max(len(c) for _, c, _ in cases)
    V = torch.zeros(len(cases), 2, n_max, dtype=torch.complex128)
    for b, (U, cols, modes) in enumerate(cases):
        V[b, :, :len(cols)] = U[modes][:, cols]
    from photonic_engines.adaptive import graded_entries
    T = binomial_moments_blocks(V, max_entries=2 * graded_entries(n_max, 2))
    assert T.shape == (len(cases), n_max + 1, n_max + 1)
    for b, (U, cols, modes) in enumerate(cases):
        single = binomial_moments(U, cols, modes)                                    # [(n_b+1)]*2
        n_b = len(cols)
        assert float((T[b, :n_b + 1, :n_b + 1] - single).abs().max()) < _tol(n_b)
        if n_b < n_max:                                                             # the padded rows and columns stay zero
            assert float(T[b, n_b + 1:].abs().max()) == 0.0 and float(T[b, :, n_b + 1:].abs().max()) == 0.0


def test_batched_prefix_columns_draw_each_circuits_posterior():
    """`prefix_columns_many`: three circuits with different prefix modes of one multiplicity shape (2, 1), different
    unitaries, streams interleaved; every circuit's draws against its enumerated posterior."""
    A = torch.stack([haar_unitary(12, 40 + g)[:, [0, 1, 3, 4, 6, 8, 9]] for g in range(3)])           # [3, 12, 7]
    prefixes = [[5, 5, 2], [11, 3, 11], [0, 7, 7]]
    per = 12000
    dp = torch.arange(3).repeat(per)                                                 # interleaved streams
    chosen = prefix_columns_many(A, prefixes, dp, torch.Generator().manual_seed(3))
    assert chosen.sum(1).tolist() == [3] * (3 * per)
    for g in range(3):
        sub, w = prefix_subset_weights(A[g:g + 1], prefixes[g])
        ref = w[0] / w[0].sum()
        index = {tuple(c): i for i, c in enumerate(sub.tolist())}
        emp = torch.zeros_like(ref)
        rows, cnt = torch.unique(chosen[dp == g], dim=0, return_counts=True)
        for r, c in zip(rows, cnt.tolist(), strict=True):
            emp[index[tuple(torch.nonzero(r)[:, 0].tolist())]] = c / per
        floor = 0.5 * math.sqrt(2 / math.pi) * float((ref * (1 - ref) / per).sqrt().sum())
        assert 0.5 * float((emp - ref).abs().sum()) < 2.5 * floor
