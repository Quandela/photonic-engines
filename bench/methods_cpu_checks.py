"""CPU checks of the chain-rule samplers, printed with their values. Each mirrors a test that asserts it (named
below); this script prints the numbers the tests only bound. CPU, about a minute.
Needs tests/ on disk: it imports `haar_unitary` (tests/helpers.py) and the circuits of tests/test_nonlinear.py.

    python3 bench/methods_cpu_checks.py [--only nlbs,exact,division,lossmask,oracles]

nlbs      Why the non-linear sampler is not a chain rule: the random-order C&C chain on the branch sum
          |sum_b c_b per(U_b[r, sigma])|^2,
          enumerated over every photon order and every step, against the circuit's law: TVD at Kerr 0.9 (n = 3,
          m = 4) and, as the check of the enumeration, at Kerr 0 (a linear circuit).
          Test: tests/test_nonlinear.py::test_random_order_chain_is_not_exact (same instance).
exact     Exactness of the superposed-source chain: its law, averaged over every source order and every branch,
          against |Per_rect(A[r, :])|^2 (n-k)!/n! on every ordered k-tuple; max absolute difference per (n, m, k).
          Test: tests/clifford/test_sbs_rso.py::test_exact_law_of_the_algorithm (same instances and unitaries).
division  Leave-one-out values by division: e_q(a without a_x) by forward division alone and by
          the mixed rule (`sbs_rso._loo_divide`) against the prefix x suffix reference, relative max error, on Glynn
          vectors of Haar rows (256 random sign vectors).
          Test: tests/clifford/test_sbs_rso.py::test_mixed_division_matches_prefix_suffix.
lossmask  Loss as a mask mixture (docs/LOSS.md): the lossy superposed source as the mixture
          over Bernoulli masks of lossless sources at alpha' (`product_sampler.lossy_source`), each mask a lossless
          strong simulation on the live modes, against `ProductSLOSKernel.probs(U, loss=eta)` (graded thinning, a
          different engine); TVD of the two exact laws.
          Test (sampling side): tests/clifford/test_product_sampler.py::test_lossy_samples_match_the_thinned_strong_simulation.
oracles   Two dense references for the non-linear sampler: the non-linear circuit's amplitudes, dense Fock
          space gate by gate vs the branch sum of permanents, and NonlinearCircuit's three routes vs the dense one.
          Tests: tests/test_nonlinear.py::test_oracles_agree, ::test_strong_routes_equal_the_oracle (same circuits).
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from helpers import haar_unitary

from photonic_engines.clifford import sbs_rso
from photonic_engines.clifford.sbs_rso import rect_cofactors
from photonic_engines.nonlinear import NonlinearCircuit
from photonic_engines.slos.reference import permanent

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="nlbs,exact,division,lossmask,oracles")
args = ap.parse_args()
only = set(args.only.split(","))


def nlbs():
    m, t = 4, [0, 1, 2]
    for phi in (0.9, 0.0):
        C = NonlinearCircuit([haar_unitary(m, 30), haar_unitary(m, 31)], t, [(1, 2, phi)])
        Ub, c = C.branch_table()
        q: dict = {}
        for sig in itertools.permutations(range(3)):
            def rec(r, pr, sig=sig):
                k = len(r)
                if k == 3:
                    key = tuple(np.bincount(r, minlength=m).tolist())
                    q[key] = q.get(key, 0.0) + pr / 6
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
        print(f"nlbs: n = 3, m = 4, Kerr {phi}: TVD random-order chain vs exact law = {tvd:.3g}")


def per_rect(B):
    q, N = B.shape
    if q == 0:
        return 1.0 + 0j
    return sum(complex(permanent(B[:, list(S)])) for S in itertools.combinations(range(N), q))


def exact():
    worst = 0.0
    for n, m, k in [(3, 4, 1), (3, 4, 2), (4, 5, 2), (4, 5, 3), (4, 6, 2), (4, 6, 3), (5, 6, 3)]:
        A = haar_unitary(m, seed=n + 7 * k)[:, :n]
        true = {r: abs(per_rect(A[list(r)])) ** 2 * math.factorial(n - k) / math.factorial(n)
                for r in itertools.product(range(m), repeat=k)}
        law = dict.fromkeys(true, 0.0)
        perms = list(itertools.permutations(range(n)))
        for sig in perms:
            As = A[:, list(sig)]

            def rec(prefix, p, As=As):
                j = len(prefix) + 1
                if j > k:
                    law[prefix] += p / len(perms)
                    return
                AT = As[:, :n - k + j]
                P = (rect_cofactors(AT[list(prefix)].unsqueeze(0))[0] if prefix
                     else torch.ones(n - k + j, dtype=As.dtype))
                w = (AT @ P).abs() ** 2
                w = w / w.sum()
                for r in range(m):
                    if float(w[r]) > 0:
                        rec((*prefix, r), p * float(w[r]))
            rec((), 1.0)
        err = max(abs(true[r] - law[r]) for r in true)
        worst = max(worst, err)
        print(f"exact: (n, m, k) = ({n}, {m}, {k}): max |law - q| = {err:.1e}")
    print(f"exact: worst over the instances {worst:.1e}")


def forward_only(a, q):
    N = a.shape[-1]
    F = torch.zeros((*a.shape[:-1], N + 1), dtype=a.dtype)
    F[..., 0] = 1
    for i in range(N):
        F[..., 1:i + 2] += a[..., i:i + 1] * F[..., 0:i + 1]
    fw = F[..., :1].expand_as(a).clone()
    for d in range(1, q + 1):
        fw = F[..., d:d + 1] - a * fw
    return fw


def division():
    for N, q in [(40, 20), (80, 40), (100, 50)]:
        U = haar_unitary(2 * N, seed=N + q)
        B = U[torch.arange(q) * 2][:, :N].to(torch.complex128)
        sd = 1 - 2 * torch.randint(0, 2, (256, q), generator=torch.Generator().manual_seed(q)).to(torch.complex128)
        a = sd @ B
        ref = sbs_rso._loo_prefix(a, q)
        scale = float(ref.abs().max())
        fw = float((ref - forward_only(a, q)).abs().max()) / scale
        mx = float((ref - sbs_rso._loo_divide(a, q)).abs().max()) / scale
        print(f"division: (N, q) = ({N}, {q}): relative error forward only {fw:.1e}, mixed {mx:.1e}")


def lossmask():
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford.product_sampler import lossy_source
    from photonic_engines.slos import ProductSLOSKernel

    def probs(st, U, **kw):
        return ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(U, **kw).as_dict()

    for n, m, alpha, eta in [(4, 6, math.pi / 4, 0.6), (4, 6, 1.1, 0.8), (5, 8, math.pi / 4, 0.7)]:
        U = haar_unitary(m, seed=n + m)
        ref = probs(ProductSuperposition.sbs(n, m, alpha=alpha), U, loss=eta)
        w1, a1 = lossy_source(alpha, eta)
        mix: dict = {}
        for mask in itertools.product((0, 1), repeat=n):
            live = [i for i in range(n) if mask[i]]
            w = math.prod(w1 if b else 1 - w1 for b in mask)
            law = probs(ProductSuperposition.sbs(len(live), m, alpha=a1, modes=live), U) if live else {(0,) * m: 1.0}
            for o, p in law.items():
                mix[o] = mix.get(o, 0.0) + w * p
        tvd = 0.5 * sum(abs(mix.get(k, 0.0) - ref.get(k, 0.0)) for k in set(mix) | set(ref))
        print(f"lossmask: n = {n}, m = {m}, alpha = {alpha:.4f}, eta = {eta}: TVD mask mixture vs thinned SLOS = {tvd:.1e}")


def oracles():
    from test_nonlinear import CASES, _occ, _vec_sites

    from photonic_engines.slos.reference import nonlinear_branch_amplitudes, nonlinear_state_amplitudes
    worst_ab = worst_route = 0.0
    for m, in_modes, nu, sites in CASES:
        Us = [haar_unitary(m, 10 + i) for i in range(nu)]
        a = nonlinear_state_amplitudes(Us, _occ(in_modes, m), _vec_sites(sites, len(in_modes)))
        b = nonlinear_branch_amplitudes(Us, _occ(in_modes, m), _vec_sites(sites, len(in_modes)))
        C = NonlinearCircuit(Us, in_modes, sites)
        worst_ab = max(worst_ab, float((a - b).abs().max()))
        worst_route = max(worst_route, *(float((C.amplitudes(r) - a).abs().max()) for r in ("branch", "gate", "auto")))
    print(f"oracles: {len(CASES)} circuits, dense gate-by-gate vs branch sum of permanents max |diff| {worst_ab:.1e}; "
          f"NonlinearCircuit amplitudes (branch / gate / auto) vs the dense oracle {worst_route:.1e}")


for name, fn in (("nlbs", nlbs), ("exact", exact), ("division", division), ("lossmask", lossmask),
                 ("oracles", oracles)):
    if name in only:
        fn()
