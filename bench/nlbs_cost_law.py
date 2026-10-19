"""The cost law of the non-linear sampler (docs/NONLINEAR.md §5b), recomputed from the frontier scans.

    python3 bench/nlbs_cost_law.py --ratio-csv results/keep/nlbs_ratio_20260922_h100.csv \
        --phi-csv results/keep/nlbs_phi_20260922_h100.csv --log logs/run_20260922_211901.log \
        --acceptance-csv $OUT/nlbs_acceptance.csv

A sample costs |c|_1^2 proposals, each one Fock sample plus B branch permanents, so
    Delta n* = n*_fock(c) - n*_nlbs = log(|c|_1^2 (1 + kappa B)) / log rho(c),
rho(c) the C&C 2020 collapsed growth at m = c n. Per point, kappa = (rho^Delta / |c|_1^2 - 1) / B. The quoted
kappa is the least-squares fit of the law in n* over the ratio scan (m/n = 1 excluded: Delta n* is smallest
there, kappa least constrained) and the phi scan (B > 1); the +- is the spread (sample standard deviation) of
the per-point kappa over the same points.

Acceptance: `find_threshold --engine nlbs` prints "accept <a> of <k> proposals, B=<B>, 1/|c|_1^2=<p>" per
measured n. --log reads those lines from the stdout of `bench/nlbs_frontier.py` (the run log) and writes them
to --acceptance-csv; without --log, --acceptance-csv is read back (the kept file). Ranges are per B.
CPU only, seconds.
"""
import argparse
import csv
import math
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--ratio-csv", required=True, help="nlbs_frontier.py --scan ratio output")
ap.add_argument("--phi-csv", required=True, help="nlbs_frontier.py --scan phi output (m = 2n)")
ap.add_argument("--fock", default="1:42.4,1.5:40.0,2:38.5,3:36.8,4:36.2,6:35.3,8:34.9,12:34.4",
                help="ratio:n* of the Fock row the scans were measured against (NONLINEAR.md §5b table)")
ap.add_argument("--log", nargs="*", default=[], help="stdout of the nlbs_frontier.py runs (acceptance lines)")
ap.add_argument("--acceptance-csv", default=None)
args = ap.parse_args()

fock = {float(c): float(n) for c, n in (kv.split(":") for kv in args.fock.split(","))}


def rho(c):
    return (2 * c + 1) ** (2 * c + 1) / ((4 * c) ** c * (c + 1) ** (c + 1))


pts = []  # (label, ratio, B, |c|_1^2, n*_fock, n*_nlbs)
for path, scan in ((args.ratio_csv, "ratio"), (args.phi_csv, "phi")):
    for r in csv.DictReader(open(path)):
        c, B = float(r["ratio"]), int(r["branches"])
        if c not in fock or B < 2 or not r["n_star"]:
            continue
        label = f"m/n={c:g}" if scan == "ratio" else f"phi={r['phi']}"
        if scan == "phi" and any(p[0] == f"m/n={c:g}" and p[2] == B for p in pts):
            continue  # phi = pi/2 at m = 2n is the ratio scan's m/n = 2 point
        pts.append((label, c, B, float(r["rejection"]), fock[c], float(r["n_star"])))


def kappa_point(p):
    _, c, B, one, nf, nn = p
    return (rho(c) ** (nf - nn) / one - 1) / B


print(f"{'point':>12} {'B':>3} {'Fock':>5} {'NLBS':>5} {'dn*':>4} {'rho(c)':>7} {'kappa':>6}")
for p in pts:
    print(f"{p[0]:>12} {p[2]:>3} {p[4]:>5.1f} {p[5]:>5.1f} {p[4] - p[5]:>4.1f} {rho(p[1]):>7.4f} "
          f"{kappa_point(p):>6.2f}")

fit = [p for p in pts if p[1] != 1.0]


def resid(k):
    return [nf - math.log(one * (1 + k * B)) / math.log(rho(c)) - nn for _, c, B, one, nf, nn in fit]


lo, hi = 0.0, 5.0  # golden-section on the sum of squares (one parameter, convex here)
g = (math.sqrt(5) - 1) / 2
for _ in range(200):
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    if sum(x * x for x in resid(a)) < sum(x * x for x in resid(b)):
        hi = b
    else:
        lo = a
k_fit = (lo + hi) / 2
ks = [kappa_point(p) for p in fit]
rms = math.sqrt(sum(x * x for x in resid(k_fit)) / len(fit))
print(f"\nkappa = {k_fit:.3f} (least squares in n*, {len(fit)} points, m/n = 1 excluded; rms residual "
      f"{rms:.2f} source) +- {statistics.stdev(ks):.3f} (sd of the per-point kappa); "
      f"all {len(pts)} points: mean {statistics.mean(kappa_point(p) for p in pts):.3f}")

pat = re.compile(r"n=\s*(\d+) m=\s*(\d+):.*accept ([0-9.]+) of (\d+) proposals, B=(\d+), 1/\|c\|_1\^2=([0-9.]+)")
rows = []
if args.log:
    for path in args.log:
        job, setting = None, None
        for line in open(path, errors="replace"):
            if (m := re.match(r"=== job (\d+)", line)):
                job = int(m.group(1))
            if (m := re.match(r"=== phi=(\S+) sites=(\d+) (\S+) m/n=([0-9.]+) eta=([0-9.]+) x=([0-9.]+)", line)):
                setting = m.groups()
            if (m := pat.search(line)) and setting is not None:
                rows.append({"log": path.split("/")[-1], "job": job, "phi": setting[0], "sites": setting[1],
                             "layout": setting[2], "ratio": setting[3], "eta": setting[4], "x": setting[5],
                             "n": m.group(1), "m": m.group(2), "accept": m.group(3), "proposals": m.group(4),
                             "B": m.group(5), "inv_c1sq": m.group(6)})
    if not rows:
        print("no acceptance lines (=== phi=... header + 'accept ... proposals') in --log")
    elif args.acceptance_csv:
        with open(args.acceptance_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.acceptance_csv} ({len(rows)} lines)")
elif args.acceptance_csv:
    rows = list(csv.DictReader(open(args.acceptance_csv)))

lossless = [r for r in rows if float(r["eta"]) == 1.0 and float(r["x"]) == 1.0]
for B in sorted({int(r["B"]) for r in lossless}):
    acc = [float(r["accept"]) for r in lossless if int(r["B"]) == B]
    inv = float(next(r["inv_c1sq"] for r in lossless if int(r["B"]) == B))
    print(f"acceptance B = {B:>3}: {min(acc):.3f} - {max(acc):.3f} over {len(acc)} points (1/|c|_1^2 = {inv:.4f})")
