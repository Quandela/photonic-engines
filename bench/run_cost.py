"""The lossy frontiers under the run-of-S statistic, priced from the lossless map (docs/GAUSSIAN.md
§2.1 for the statistic). No GPU run: a calibrated-model number, labelled so.

    python3 bench/run_cost.py --engine fock                    # eta = 0.9 .. 0.5 and the (eta, V) points at m = 2n
    python3 bench/run_cost.py --engine emitter
    python3 bench/run_cost.py --engine fock --check           # model E[cost] against the measured lossy points

The statistic (docs/GAUSSIAN.md §2.1): t_S = sum_{k <= k_S} P(k) E[cost | k], k_S the smallest k with P(K > k) <= 1/S.
For Fock, emitter, NLBS and adaptive inputs under uniform loss eta and the x-model the stratum variable is the
number g of interfering photons, exactly Binomial(n, eta x) and U-independent (`gaussian.cost.binomial_law`), and
a stratum's cost is the lossless cost of a g-photon instance in the same m modes (LOSS.md §3: one lockstep pass
per surviving count; DISTINGUISHABILITY.md §3.1). That cost is not in the lossy CSVs (batch rates only), so it is
MODELLED from the lossless map: for every measured mode ratio r the points (n, seconds per sample) of the
canonical run give a fit log t = a_r + b_r n, and cost(g, m) is the fit at the ratio m/g, (a, b) interpolated
linearly in log r between the measured ratios (clamped at the ends). The untruncated mean of the same model is
the check: with `--check` it is compared with every measured lossy point (ms per sample), which says how far
the model is from the machine before the truncation is read off it.

Reads (results/keep and the canonical Fock run 20260915_222632, see --lossless): the per-ratio lossless point
files, the lossy point files `frontier_loss_<engine>*_eta*.csv`, the crossings `frontier_loss_run*.csv` and
`frontier_indist_run2.csv`. Writes $RESULTS_DIR/run_cost_<engine>.csv.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys
from itertools import pairwise

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.gaussian.cost import binomial_law, run_cutoff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_points(path):
    """(n, m, seconds per sample) rows of a find_threshold point file."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["n"]), int(r["m"]), float(r["ms_per_sample"]) / 1e3))
    return out


class LosslessMap:
    """cost(g, m): seconds of a lossless g-photon sample in m modes, from per-ratio log-linear fits."""

    def __init__(self, files):
        self.fits = {}                                                         # ratio -> (a, b)
        for f in files:
            pts = read_points(f)
            if len(pts) < 2:
                continue
            ratio = pts[0][1] / pts[0][0]
            ns = np.array([p[0] for p in pts], dtype=float)
            ts = np.array([p[2] for p in pts])
            b, a = np.polyfit(ns, np.log(ts), 1)
            self.fits[round(ratio, 3)] = (a, b, ns.min(), ns.max())
        self.ratios = sorted(self.fits)
        if not self.ratios:
            raise SystemExit("no lossless point files matched")
        self.clamped = 0

    def coeffs(self, r):
        rs = self.ratios
        if r <= rs[0]:
            self.clamped += r < rs[0] * 0.999
            return self.fits[rs[0]][:2]
        if r >= rs[-1]:
            self.clamped += r > rs[-1] * 1.001
            return self.fits[rs[-1]][:2]
        i = max(j for j in range(len(rs)) if rs[j] <= r)
        r0, r1 = rs[i], rs[i + 1]
        w = (math.log(r) - math.log(r0)) / (math.log(r1) - math.log(r0))
        a0, b0 = self.fits[r0][:2]
        a1, b1 = self.fits[r1][:2]
        return (1 - w) * a0 + w * a1, (1 - w) * b0 + w * b1

    def cost(self, g, m):
        if g < 2:
            return 0.0                                                         # a lone photon is routed on the host
        a, b = self.coeffs(m / g)
        return math.exp(a + b * g)

    def describe(self):
        return ", ".join(f"r={r:g}: rho={math.exp(self.fits[r][1]):.3f} (n {int(self.fits[r][2])}-{int(self.fits[r][3])})"
                         for r in self.ratios)


def expected_costs(lmap, n, m, p, S_list):
    """(untruncated mean, {S: t_S}, {S: k_S}) of the model at n sources, m modes, interfering probability p."""
    law = binomial_law(n, p)
    costs = np.array([lmap.cost(g, m) for g in range(n + 1)])
    full = float((law * costs).sum())
    tS, kS = {}, {}
    for S in S_list:
        k = run_cutoff(law, S)
        kS[S] = k
        tS[S] = float((law[: k + 1] * costs[: k + 1]).sum())
    return full, tS, kS


def crossing(fn, lo, hi, target=1.0):
    """The real n where the log-linear interpolation of fn(n) (seconds) crosses `target`, scanning integers lo..hi."""
    prev = None
    for n in range(lo, hi + 1):
        t = fn(n)
        if prev is not None and prev[1] <= target < t:
            n0, t0 = prev
            return n0 + (math.log(target) - math.log(t0)) / (math.log(t) - math.log(t0))
        prev = (n, t)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="fock", choices=["fock", "emitter"])
    ap.add_argument("--lossless", default=None, help="glob of the lossless per-ratio point files (default: the canonical "
                    "run 20260915_222632 for fock, results/keep/frontier_emitter_r*.csv for emitter)")
    ap.add_argument("--S", default="1e4,1e6,1e8")
    ap.add_argument("--check", action="store_true", help="compare the model's untruncated mean with the measured lossy points")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eng = args.engine
    S_list = [float(x) for x in args.S.split(",")]
    if args.lossless:
        files = sorted(glob.glob(os.path.join(ROOT, args.lossless)))
    elif eng == "fock":
        files = sorted(glob.glob(os.path.join(ROOT, "results/20260915_222632/frontier_v2_fock_r*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(ROOT, "results/keep/frontier_fock_r*.csv")))
    else:
        files = sorted(glob.glob(os.path.join(ROOT, "results/keep/frontier_emitter_r*.csv")))
    lmap = LosslessMap(files)
    print(f"{eng}: lossless map from {len(files)} ratio files — {lmap.describe()}")

    # the measured lossy crossings (eta axis, then the (eta, V) grid)
    meas = {}                                                                  # (eta, x) -> measured n*
    for f in (["results/keep/frontier_loss_run1.csv"] if eng == "fock" else
              ["results/keep/frontier_loss_emitter.csv"]):
        path = os.path.join(ROOT, f)
        if os.path.exists(path):
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    if r["engine"] == eng and r["n_star"] not in ("", "nan"):
                        meas[(float(r["eta"]), 1.0)] = float(r["n_star"])
    path = os.path.join(ROOT, "results/keep/frontier_indist_run2.csv")
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["engine"] == eng:
                    meas[(float(r["eta"]), float(r["x"]))] = float(r["n_star"])
    # emitter: the unravelled lossy frontier, per-eta point files (run 20261003_103957, kernel="auto";
    # the seq-kernel run emitter3 of 2026-09-18 is in git history)
    for f in sorted(glob.glob(os.path.join(ROOT, f"results/keep/frontier_loss_{eng}_auto_{eng}_eta*.csv"))):
        pts = read_points(f)
        eta = float(re.search(r"eta([0-9.]+)\.csv", f).group(1))
        pts = sorted(pts)
        for (n0, _, t0), (_n1, _, t1) in pairwise(pts):
            if t0 <= 1.0 < t1:
                meas.setdefault((eta, 1.0), n0 + (0 - math.log(t0)) / (math.log(t1) - math.log(t0)))

    rows = []
    c = 2.0                                                                    # every lossy point is at m = 2n
    print(f"\n{eng}, m = 2n, 1 sample/s. n* measured (bracket) | model untruncated | model t_S for S = "
          + " / ".join(f"{S:.0e}" for S in S_list) + " | shift n*(S) - n*(untruncated) | k_S at n*")
    for (eta, x) in sorted(meas, key=lambda k: (-k[0], -k[1])):
        p = eta * x

        def f_full(n):
            return expected_costs(lmap, n, round(c * n), p, S_list)[0]

        n_full = crossing(f_full, 10, 120)
        n_S = {}
        for S in S_list:
            n_S[S] = crossing(lambda n, S=S: expected_costs(lmap, n, round(c * n), p, [S])[1][S], 10, 120)
        n_ref = round(n_S[S_list[0]]) if not math.isnan(n_S[S_list[0]]) else round(n_full)
        _, _, kS = expected_costs(lmap, n_ref, round(c * n_ref), p, S_list)
        row = {"engine": eng, "ratio": c, "eta": eta, "x": x, "p": p, "nstar_measured": meas[(eta, x)],
               "nstar_model_untruncated": round(n_full, 2)}
        for S in S_list:
            row[f"nstar_model_S{S:.0e}"] = round(n_S[S], 2)
            row[f"shift_S{S:.0e}"] = round(n_S[S] - n_full, 2)
            row[f"kS_S{S:.0e}"] = kS[S]
        rows.append(row)
        print(f"  eta={eta:<4g} x={x:<6g} p={p:.3f}: measured {meas[(eta, x)]:5.1f} | model {n_full:5.1f} | "
              + " / ".join(f"{n_S[S]:5.1f}" for S in S_list) + " | "
              + " / ".join(f"{n_S[S] - n_full:+.2f}" for S in S_list) + " | "
              + " / ".join(str(kS[S]) for S in S_list))
    print(f"  (ratio m/g outside the measured range on {lmap.clamped} model evaluations: clamped to the end ratio)")

    if args.check:
        print(f"\n{eng}: model untruncated mean against the measured lossy points (model / measured, seconds per sample)")
        pat = f"results/keep/frontier_loss_{eng}_eta*.csv" if eng == "fock" else f"results/keep/frontier_loss_{eng}_auto_{eng}_eta*.csv"
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            eta = float(re.search(r"eta([0-9.]+)\.csv", f).group(1))
            for n, m, t in read_points(f):
                full, tS, kS = expected_costs(lmap, n, m, eta, S_list)
                print(f"  eta={eta:<4g} n={n:3d} m={m:3d}: measured {t:9.3f} s  model {full:9.3f} s  ratio {full / t:5.2f}"
                      f"   t_S(1e4)/untr {tS[S_list[0]] / full:.3f}")

    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results")), f"run_cost_{eng}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    main()
