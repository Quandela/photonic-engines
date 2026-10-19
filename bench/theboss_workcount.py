"""theboss against this engine, counted rather than timed: Guan-code addends per exact sample.

`theboss` (PyPI, Apache-2.0, Rybotycki) is the only public codebase that implements the same
ALGORITHM GENERATION as this engine -- C&C 2020's submatrices speedup
(`GeneralizedCliffordsBSimulationStrategy`) with a Guan-code (mixed-radix Gray) walk carrying a
running binomial product (`BSGuanBasedSubmatricesPermanentCalculatorBase`).  It is single-threaded
Python; we are CUDA.  So a wall-clock comparison measures the language, not the algorithm, and
tells us nothing about whether their implementation contains an optimisation ours does not.

This measures the thing that IS comparable: **the number of Gray/Guan addends evaluated to draw
one exact sample**.  One addend is one rank-1 column-sum update, from which the step's permanents
are read off, in both codebases.  It is machine-independent, language-independent, and it is the
quantity the rho(c)^n exponent is a statement about.

The decisive question it answers: theboss's submatrices calculator sets its Guan radix from
`input_state` (docstring: "we can reverse the role of input and output ... not desired for the
standard BS experiment, wherein the input is not binned").  For a standard Fock input every input
multiplicity is 1, so that radix would give 2^k per step -- the 2018 exponent -- while this engine
puts the radix on the ALREADY-CHOSEN OUTPUT rows, where the collisions actually are, which is what
produces rho(c)^n.  If that reading is right the counts diverge as n grows, and theboss's
repeated-row machinery, though real, does not reach the 2020 exponent on this problem.
**The count settles it; the reading does not.**

    python3 bench/theboss_workcount.py --ns 4,6,8,10,12 --ratio 2 --samples 20
    python3 bench/theboss_workcount.py --ns 6,8 --ratio 1 --samples 20 --check   # + TVD agreement

Reported per n: mean addends per sample for each side, the ratio, and a fitted base b in
addends ~ b^n -- which is the exponent claim, measured.
"""
import argparse
import csv
import itertools
import math
import os
import statistics

import numpy as np


# ---------------------------------------------------------------- our per-step term count
def our_terms(rows):
    """Terms this engine's mixed-radix Gray walk visits for a step whose already-chosen OUTPUT
    rows are `rows` (a multiset).  Exactly `multiplicity_tables` in
    photonic_engines/clifford/cofactors_triton.py: radix_nu = s_nu + 1 over the distinct rows,
    with one odd-multiplicity group's factor replaced by (s+1)//2 (the sign-symmetry halving).
    Validated against that function on the GPU host by --validate.
    """
    if len(rows) == 0:
        return 1
    mult = {}
    for r in rows:
        mult[r] = mult.get(r, 0) + 1
    s = sorted(mult.values())
    radix = [x + 1 for x in s]
    odd = [i for i, x in enumerate(s) if x % 2 == 1]
    if odd:
        i = odd[-1]
        radix[i] = (s[i] + 1) // 2
    return math.prod(radix)


def our_addends(path):
    """Total addends to draw the sample whose chain chose output rows `path` in that order:
    step k = 1..n sees the k-1 rows already chosen."""
    return sum(our_terms(path[:k]) for k in range(len(path)))


# ------------------------------------------------------------------- theboss, instrumented
_COUNTER = {"guan": 0}          # module level: the patch is installed once and must outlive it
_PATCHED = []


def run_theboss(U, n, m, samples, seed):
    """Draw `samples` exact samples with GeneralizedCliffordsBSimulationStrategy, counting every
    Guan-code update and recording the chain path, so both sides can be counted on the SAME path."""
    import numpy.random as npr
    npr.seed(seed)       # noqa: NPY002 -- theboss draws from numpy's global legacy RNG, so it is seeded here
    from theboss.boson_sampling_utilities.permanent_calculators import (
        bs_permanent_calculator_base as perm_base,
    )
    from theboss.boson_sampling_utilities.permanent_calculators import (
        bs_submatrices_permanent_calculator_base as sub_base,
    )
    from theboss.boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_factory import (
        BSPermanentCalculatorFactory,
    )
    from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
        GeneralizedCliffordsBSimulationStrategy as GCB,
    )

    counter = _COUNTER

    def patch(mod):
        hits = []
        for name in dir(mod):
            cls = getattr(mod, name)
            if isinstance(cls, type) and hasattr(cls, "_update_guan_code"):
                orig = cls._update_guan_code
                if getattr(orig, "_counted", False):
                    continue

                def wrapped(self, _o=orig):
                    counter["guan"] += 1
                    return _o(self)

                wrapped._counted = True
                cls._update_guan_code = wrapped
                hits.append(name)
        return hits

    if not _PATCHED:
        _PATCHED.extend(patch(sub_base) + patch(perm_base))
    patched = _PATCHED

    inp = [1] * n + [0] * (m - n)
    pc = BSPermanentCalculatorFactory(U, inp, None).generate_calculator()
    strat = GCB(pc)

    # record the chain path: the output mode drawn at each step
    paths, cur = [], []
    orig_draw = GCB._sample_from_pmf

    def draw(self):
        before = list(self.r_sample)
        orig_draw(self)
        for i, (a, b) in enumerate(zip(self.r_sample, before)):
            if a > b:
                cur.append(i)
                break

    GCB._sample_from_pmf = draw

    per_sample, out = [], []
    try:
        for _ in range(samples):
            counter["guan"] = 0
            cur.clear()
            s = strat.simulate(inp, 1)[0]
            per_sample.append(counter["guan"])
            paths.append(list(cur))
            out.append(s)
    finally:
        GCB._sample_from_pmf = orig_draw
    return per_sample, paths, out, patched


# ------------------------------------------------------------------------- agreement check
def perm(A):
    k = A.shape[0]
    if k == 0:
        return 1.0 + 0j
    tot = 0j
    for msk in range(1, 1 << k):
        cols = [j for j in range(k) if msk >> j & 1]
        tot += (-1) ** len(cols) * np.prod(A[:, cols].sum(1))
    return tot * (-1) ** k


def exact_distribution(U, n, m):
    out = {}
    for combo in itertools.combinations_with_replacement(range(m), n):
        t = [0] * m
        for c in combo:
            t[c] += 1
        rows = [r for r in range(m) for _ in range(t[r])]
        out[tuple(t)] = abs(perm(U[np.ix_(rows, list(range(n)))])) ** 2 / math.prod(
            math.factorial(x) for x in t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="4,6,8,10,12")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=0, metavar="SHOTS",
                    help="also draw SHOTS extra samples and compare the distribution against brute "
                         "force (only where C(m+n-1, n) <= --check/20, so the TVD is meaningful)")
    ap.add_argument("--validate", action="store_true",
                    help="check our_terms against multiplicity_tables (needs torch)")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    if a.validate:
        import torch

        from photonic_engines.clifford.cofactors_triton import multiplicity_tables
        rng = np.random.default_rng(0)
        bad = 0
        for _ in range(300):
            K1 = int(rng.integers(1, 12))
            rows = rng.integers(0, 6, size=K1)
            t = torch.tensor(rows, dtype=torch.int64)[None]
            N = int(multiplicity_tables(t, K1)[3][0])
            if our_terms(list(rows)) != N:
                bad += 1
                print(f"  mismatch rows={list(rows)}: table {N}, ours {our_terms(list(rows))}")
        print(f"our_terms vs multiplicity_tables: {'OK' if not bad else str(bad) + ' MISMATCHES'} "
              f"over 300 random steps\n")

    rows_out = []
    print(f"{a.samples} samples per n, m = {a.ratio:g}n, Haar, seed {a.seed}\n")
    print(f"{'n':>3} {'m':>4} {'theboss guan':>14} {'ours gray':>12} {'theboss/ours':>13} "
          f"{'2^n scale':>10} {'note':>22}")
    for n in [int(x) for x in a.ns.split(",")]:
        m = round(a.ratio * n)
        rng = np.random.default_rng(a.seed + n)
        z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        q, r = np.linalg.qr(z)
        U = q * (np.diag(r) / abs(np.diag(r)))

        tb, paths, out, patched = run_theboss(U, n, m, a.samples, a.seed)
        ours = [our_addends(p) for p in paths]
        mtb, mo = statistics.fmean(tb), statistics.fmean(ours)
        # how much of 2^n the two sides spend, as a scale-free read of the exponent
        ref = sum(2 ** k for k in range(n))
        note = ""
        n_out = math.comb(m + n - 1, n)
        if a.check and n_out * 20 <= a.check:
            from collections import Counter
            _, _, big, _ = run_theboss(U, n, m, a.check, a.seed + 977)
            emp = {k: v / a.check for k, v in Counter(big).items()}
            ex = exact_distribution(U, n, m)
            d = 0.5 * sum(abs(emp.get(k, 0) - ex.get(k, 0)) for k in set(emp) | set(ex))
            # sampling floor: E[TVD] ~ sqrt(n_out / (2 pi shots)) for a flat-ish distribution
            floor = math.sqrt(n_out / (2 * math.pi * a.check))
            note = f"TVD {d:.3f} (floor {floor:.3f})"
        print(f"{n:>3} {m:>4} {mtb:>14,.0f} {mo:>12,.0f} {mtb / max(mo, 1):>13.2f} "
              f"{mtb / ref:>10.2f} {note:>22}")
        rows_out.append({"n": n, "m": m, "ratio": a.ratio, "samples": a.samples,
                             "theboss_guan": mtb, "ours_gray": mo, "ratio_tb_ours": mtb / max(mo, 1),
                             "sum_2k": ref, "note": note})

    if len(rows_out) >= 2:
        def base(key):
            xs = np.array([r["n"] for r in rows_out], dtype=float)
            ys = np.log(np.array([max(r[key], 1.0) for r in rows_out]))
            return math.exp(np.polyfit(xs, ys, 1)[0])
        print(f"\nfitted base b in addends ~ b^n:   theboss {base('theboss_guan'):.3f}   "
              f"ours {base('ours_gray'):.3f}")
        print("A base near 2.0 is the C&C 2018 exponent; rho(c) < 2 is the 2020 collapse.\n"
              "rho(c) at this m/n is the theory value in results/keep/frontier.csv.")
    print(f"\ninstrumented classes: {', '.join(patched) if rows_out else '-'}")
    print("Counting convention: one Guan/Gray update = one addend = one rank-1 column-sum update,\n"
          "in both codebases.  Our count is taken along the SAME chain path theboss drew, so the\n"
          "comparison has no sampling-path bias.")

    if a.csv and rows_out:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows_out)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
