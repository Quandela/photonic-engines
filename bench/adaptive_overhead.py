"""Rejection overhead of the adaptive sampler's final stage (docs/ADAPTIVE.md §5), from the exact outcome law.

For Haar U, m = c n, K adaptive modes: P(p) by `mode_marginals` (no sampling, CPU). Per (n, K), averaged
over `--seeds` unitaries:
    H_bits      entropy of p;  2^H is the typical-branch overhead 1 / P(p)
    branches    C(n + K, K): the MEAN number of boson samples drawn per accepted sample when every
                branch goes through the rejection (E[1 / P(p)] = number of outcomes), exact sampling
    kept(tau)   mean draws per sample when the branches with P(p) < tau do NOT go through the rejection
                (= number of branches kept), and their mass: the total-variation error if they are
                dropped, the share of samples that needs another route if they are not
    dn_*        the frontier shift  -log X / log rho  at rho = 2 for X = 2^H, kept(tau), branches

    python bench/adaptive_overhead.py --n 10 20 30 --K 1 2 --out results/adaptive_overhead.csv
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from helpers import haar_unitary

from photonic_engines.adaptive import mode_marginals

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[10, 20, 30])
ap.add_argument("--K", type=int, nargs="+", default=[1, 2])
ap.add_argument("--c", type=int, default=2)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--tau", type=float, default=1e-2)
ap.add_argument("--out", default="results/adaptive_overhead.csv")
a = ap.parse_args()

rows = []
for n in a.n:
    for K in a.K:
        if (n + 1) ** (2 * K) > 1 << 27:
            continue
        acc = {"H_bits": 0.0, "kept": 0.0, "mass_below_tau": 0.0}
        for seed in range(a.seeds):
            m = a.c * n
            law = mode_marginals(haar_unitary(m, 1000 * n + seed), list(range(0, m, a.c)), list(range(K))).reshape(-1)
            nz = law[law > 0]
            acc["H_bits"] += float(-(nz * nz.log2()).sum()) / a.seeds
            acc["kept"] += float((law >= a.tau).sum()) / a.seeds
            acc["mass_below_tau"] += float(law[law < a.tau].sum()) / a.seeds
        branches = math.comb(n + K, K)
        row = {"n": n, "m": a.c * n, "K": K, "tau": a.tau, "branches": branches, **{k: round(v, 4) for k, v in acc.items()},
               "dn_typical": round(-acc["H_bits"], 2), "dn_kept": round(-math.log2(max(acc["kept"], 1.0)), 2),
               "dn_exact": round(-math.log2(branches), 2)}
        rows.append(row); print(row)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
