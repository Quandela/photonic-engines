"""Addends per exact sample for every algorithm of the T1 set, along the SAME chains (docs/ALGORITHMS.md §4).

One addend = one rank-1 update of the k column (or row) sums of a step's Gray/Guan walk, the unit
`bench/theboss_workcount.py` counts; it is the machine-independent quantity the exponent claims are
about. For a chain that chose the output rows r_1..r_n in that order from the input columns
sigma(1..n), step k costs (photonic_engines.clifford.variants.addends):

    2018      2^(k-2)                                   Clifford & Clifford 2018, one shared sweep
    2020      prod_nu (s_nu + 1) / 2   over r_{<k}      Clifford & Clifford 2020, output collapse (this engine)
    input     prod_mu (s_mu + 1) / 2   over sigma(1..k) Brod & Oszmaniec / theboss, input collapse
    percof    k x [2020 count]                          PiquassoBoost: one sweep per cofactor
    theboss   2^k                                       the measured theboss count (sum = 2^(n+1) - 4), for reference
    pqb_lit   sum over the k columns of t_min prod_{nu != min} (t_nu + 1)   PiquassoBoost's own halving (row of
              minimal multiplicity fixed): what its BBFGPermanentCalculatorRepeated walks, for reference

The chains are drawn by the plain-torch reference sampler (`fock_sample_reference`, n <= ~12) or, with
--engine triton|cuda, by `fock_sample` followed by a uniform ordering of the sample's photons: the
chain's row sequence is exchangeable (the C&C symmetrisation), so a random ordering of the final
occupation IS a chain path, which is what lets the counts be taken at frontier sizes without
instrumenting the kernels.

    python3 bench/algorithm_terms.py --ns 4,6,8,10,12 --ratios 1,2 --samples 40
    python3 bench/algorithm_terms.py --ns 16,20,24,28,32,36,40 --ratios 1,2 --samples 100 --engine cuda

CSV columns: ratio, n, m, samples, then mean addends per sample for each algorithm and its fitted
base (addends ~ b^n over the n of the run, printed at the end).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford.variants import addends

ALGOS = ["2018", "2020", "input", "percof", "theboss", "pqb_lit"]


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def pqb_literal(rows):
    """PiquassoBoost's addends for ONE cofactor at a step whose chosen rows are `rows`: the n-ary Gray
    code over the output multiplicities with the row of minimal (non-zero) multiplicity fixed
    (BBFGPermanentCalculatorRepeated_Tasks: row_mult[minelem]--, radix = mult + 1 on the rest)."""
    if not rows:
        return 1
    mult: dict = {}
    for r in rows:
        mult[r] = mult.get(r, 0) + 1
    s = sorted(mult.values())
    return s[0] * math.prod(x + 1 for x in s[1:])


def chain_addends(path, sig):
    n = len(path)
    out = dict.fromkeys(ALGOS, 0)
    for k in range(1, n + 1):
        rows, cols = path[:k - 1], sig[:k]
        for a in ("2018", "2020", "input", "percof"):
            out[a] += addends(a, rows, cols)
        out["theboss"] += (1 << k) if k >= 2 else 1
        out["pqb_lit"] += k * pqb_literal(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="4,6,8,10,12")
    ap.add_argument("--ratios", default="1,2")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--engine", default="reference", choices=["reference", "triton", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="results/algorithm_terms.csv")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    rows_out = []
    for c in ratios:
        per_n = {a: [] for a in ALGOS}
        for n in ns:
            m = round(c * n)
            U = haar(m, args.seed + n)
            in_modes = list(range(n))
            if args.engine == "reference":
                from photonic_engines.clifford.variants import fock_sample_reference
                # the reference sampler draws sigma itself; re-draw an equivalent one here: the
                # count of "input" on a collision-free input does not depend on sigma at all
                occ = fock_sample_reference(U, in_modes, args.samples, algo="2020", generator=gen)
            else:
                from photonic_engines.clifford import fock_sample
                dev = "cuda" if args.engine == "cuda" else "cpu"
                occ = fock_sample(U.to(dev), in_modes, args.samples, generator=torch.Generator(device=dev).manual_seed(args.seed + n),
                                  algo="auto", backend=args.engine).cpu()
            tot = dict.fromkeys(ALGOS, 0.0)
            for o in occ.tolist():
                photons = [j for j, x in enumerate(o) for _ in range(x)]
                path = [photons[i] for i in torch.randperm(n, generator=gen).tolist()]
                sig = torch.randperm(n, generator=gen).tolist()
                for a, v in chain_addends(path, sig).items():
                    tot[a] += v
            mean = {a: tot[a] / args.samples for a in ALGOS}
            for a in ALGOS:
                per_n[a].append(mean[a])
            rows_out.append({"ratio": c, "n": n, "m": m, "samples": args.samples, **mean})
            print(f"c={c:g} n={n:2d} m={m:3d}: " + "  ".join(f"{a} {mean[a]:12.1f}" for a in ALGOS), flush=True)
        if len(ns) >= 2:
            print(f"c={c:g} fitted base b (addends ~ b^n): " + "  ".join(
                f"{a} {math.exp(np.polyfit(ns, np.log(per_n[a]), 1)[0]):.3f}" for a in ALGOS))
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", args.csv)


if __name__ == "__main__":
    main()
