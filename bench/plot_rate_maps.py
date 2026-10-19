"""The frontier at other rates: n*(m/n) at 1e-3, 1 and 1e3 samples/s, one H100.

    python3 bench/plot_rate_maps.py [--csv results/keep/frontier.csv] [--rates 1e-3,1,1e3]
                                   [--out img/frontier_rate_maps] [--table results/keep/frontier_rate_maps_<stamp>.csv]

The rate enters the frontier as a translation: a sampler whose cost per sample is t(n) = t0 rho^n
crosses R samples/s at n*(R) = n*(1 Hz) - ln R / ln rho. Every curve here is the kept 1 Hz frontier
(results/keep/frontier.csv: measured n* and the rho_eff fitted at the bracket, per engine and
ratio) shifted by that amount with the row's own rho_eff — no new measurement. The 1e3 Hz map is
checked against direct measurements at that rate: the production envelope points of
results/keep/algorithm_envelope_c{1,2}_20260930_h100.csv (n = 20..44, where 1e3 samples/s is
bracketed between consecutive n) and the cost-table columns of results/keep/cost_table_20261011_h100.csv
(m = 32 / 48 / 96, bracketed in k). Those crossings are drawn as hollow markers; the 1e-3 Hz map
has no measurement behind it (1000 s per sample is past every budget) and is a pure extrapolation,
labelled so. Where a translated curve enters the precision regime noted in ALGORITHMS.md §6 (fp64
cofactor error ~1e-6 near n = 43, ~1e-3 near n = 50 on the Fock chain) or past the GBS fp64 tier
(N ~ 55), the point is still drawn — the caveat is the table's, not the figure's.

Superposed sources (sbs) are the exception: their cost per sample is heavy-tailed, so a
ramp-fitted rho_eff is noise (1.21 at m/n = 12). With --engines ...,sbs their 1e-3 / 1e3 cells are shifted by the run-cost
model instead — n_model(R) - n_model(1 Hz) from bench/sbs_run_cost.cost (fitted on strata k <= 22) — drawn dashed and
labelled model-priced, the 1 Hz points staying measured. Allowed only if the model reproduces the measured 1 Hz n* within
0.5 source at all nine ratios (on the 10-06 re-run); otherwise sbs stays out of the 1e-3 / 1e3 maps (1 Hz only, stated).
On the 10-06 fit (run 20261006_132352) the test failed (up to 1.48 low at m/n >= 3), so the default build leaves sbs out. The 1e3 Hz
model cells are nan by construction: the model's per-sample floor (> 1 ms) means 1e3 samples/s is never reached at any n.

Writes <out>.png and --table (engine, ratio, rate, n_star, rho_eff, shift, measured — the
measured column holds the direct crossing where one exists).
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", default="results/keep/frontier.csv")
ap.add_argument("--rates", default="1e-3,1,1e3")
ap.add_argument("--engines", default="fock,emitter,gbs",
                help="rows of the frontier CSV to translate; sbs (the rso chain) only once the model test passes "
                     "(cost model within 0.5 source of the measured 1 Hz n* at all nine ratios) — else sbs is quoted at 1 Hz only")
ap.add_argument("--sbs-price", default="model", choices=["model", "rho"],
                help="sbs rows: shift by the run-cost model, n_model(R) - n_model(1 Hz) (bench/sbs_run_cost.cost; "
                     "the ramp-fitted rho_eff of heavy-tailed sbs shots is noise) or, like the other rows, by rho_eff")
ap.add_argument("--envelopes", default="results/keep/algorithm_envelope_c2_20260930_h100.csv,"
                "results/keep/algorithm_envelope_c1_20260930_h100.csv")
ap.add_argument("--cost-table", default="results/keep/cost_table_20261011_h100.csv")
ap.add_argument("--out", default="img/frontier_rate_maps", help="figure, .png appended")
ap.add_argument("--table", default="results/frontier_rate_maps.csv", help="the translated / measured points")
args = ap.parse_args()
rates = [float(x) for x in args.rates.split(",")]

engines = args.engines.split(",")
rows = [r for r in csv.DictReader(open(args.csv)) if r["n_star"] not in ("nan", "") and r["engine"] in engines]
eng = {}
for r in rows:
    eng.setdefault(r["engine"], []).append((float(r["ratio"]), float(r["n_star"]), float(r["rho_eff"])))
for k in eng:
    eng[k].sort()


def model_n(ratio, n_start, R, price):
    """n at which the model's seconds per sample crosses 1/R, log-linear between consecutive n."""
    target = 1.0 / R
    n = max(4, math.floor(n_start))
    t = price(n, max(2, round(ratio * n)))
    step = 1 if t < target else -1
    while 4 <= n + step <= 200:
        n1 = n + step
        t1 = price(n1, max(2, round(ratio * n1)))
        if (t - target) * (t1 - target) <= 0:
            (na, ta), (nb, tb) = sorted([(n, t), (n1, t1)])
            return na + (math.log(target) - math.log(ta)) / (math.log(tb) - math.log(ta))
        n, t = n1, t1
    return float("nan")


# sbs shifts from the cost model (mean statistic, alpha = pi/4, eta = 1 — the conditions of the nine-ratio row)
sbs_shift = {}
if "sbs" in eng and args.sbs_price == "model":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sbs_run_cost import cost as sbs_cost
    for ratio, nmeas, _ in eng["sbs"]:
        n1 = model_n(ratio, nmeas, 1.0, sbs_cost)
        for R in rates:
            sbs_shift[(ratio, R)] = model_n(ratio, n1, R, sbs_cost) - n1 if R != 1 else 0.0


def shift(name, p, R):
    """n*(R) - n*(1 Hz) for one row: the model's for sbs, -ln R / ln rho_eff otherwise."""
    if name == "sbs" and sbs_shift:
        return sbs_shift[(p[0], R)]
    return -math.log(R) / math.log(p[2])


def crossing(ns, rates_meas, R):
    """n at which a measured samples/s series (decreasing in n) crosses R, log-linear between neighbours."""
    for (n0, r0), (n1, r1) in zip(zip(ns, rates_meas), zip(ns[1:], rates_meas[1:])):
        if r0 >= R > r1:
            return n0 + (n1 - n0) * (math.log(r0) - math.log(R)) / (math.log(r0) - math.log(r1))
    return None


# direct Fock measurements at each rate: (ratio, n) from the envelopes (fixed ratio) and the cost table (fixed m)
measured = {R: [] for R in rates}
for path in [p for p in args.envelopes.split(",") if p and os.path.exists(p)]:
    pts = [r for r in csv.DictReader(open(path)) if r["algo"] == "auto"]
    pts.sort(key=lambda r: int(r["n"]))
    ns = [int(r["n"]) for r in pts]; rs = [float(r["samples_per_s"]) for r in pts]
    for R in rates:
        n = crossing(ns, rs, R)
        if n is not None:
            measured[R].append((float(pts[0]["ratio"]), n, "envelope"))
if os.path.exists(args.cost_table):
    tab = {}
    for r in csv.DictReader(open(args.cost_table)):
        tab.setdefault(int(r["m"]), []).append((int(r["k"]), float(r["samples_per_s"])))
    for m, pts in tab.items():
        pts.sort()
        ks = [p[0] for p in pts]; rs = list(np.maximum.accumulate([p[1] for p in pts][::-1])[::-1])   # monotone envelope
        for R in rates:
            k = crossing(ks, rs, R)
            if k is not None and m / k <= 12.5:
                measured[R].append((m / k, k, f"cost table m = {m}"))

style = {"fock": ("tab:blue", "Fock input"), "emitter": ("tab:red", "emitter chain (linear cluster)"),
         "sbs": ("tab:green", "superposition BS"), "gbs": ("tab:orange", "Gaussian BS (timing model, t_S)")}
out = []
fig, axes = plt.subplots(1, len(rates), figsize=(4.2 * len(rates), 4.6), sharey=True)
for ax, R in zip(np.atleast_1d(axes), sorted(rates)):
    for name, pts in eng.items():
        color, label = style.get(name, ("k", name))
        xs = [p[0] for p in pts]
        ys = [p[1] + shift(name, p, R) for p in pts]
        model = name == "sbs" and sbs_shift and R != 1
        ax.plot(xs, ys, "o-" if R == 1 else ("--" if model else "-"), color=color, lw=2, ms=4,
                label=label + (" (model-priced)" if model else ""))
        for p, y in zip(pts, ys):
            out.append([name, p[0], R, round(y, 1), "model" if model else p[2], round(shift(name, p, R), 2),
                        "model-priced: sbs_run_cost fit (k <= 22, model/measured 0.60-1.43)" if model else ""])
    if measured[R]:
        h, = ax.plot([p[0] for p in measured[R]], [p[1] for p in measured[R]], "o", mfc="none", mec="tab:blue", ms=8, mew=1.5,
                     label="Fock, measured crossing (envelope / cost table)")
        ax.legend(handles=[h], fontsize=7.5, loc="lower left")
        for p in measured[R]:
            out.append(["fock", round(p[0], 2), R, "", "", "", f"{p[1]:.1f} ({p[2]})"])
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4, 6, 8, 12]); ax.set_xticklabels(["1", "1.5", "2", "3", "4", "6", "8", "12"])
    ax.set_xlabel("m / n (modes per source)")
    tag = "measured frontier" if R == 1 else ("translated" if measured[R] else "translated, extrapolated")
    ax.set_title(f"{R:g} samples/s  ({tag})", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel("n* (sources)")
np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="lower left")   # engines once, on the extrapolated panel
fig.suptitle("The frontier at three rates, one H100: n*(R) = n*(1 Hz) - ln R / ln rho_eff", fontsize=11)
fig.tight_layout()
fig.savefig(args.out + ".png", dpi=150)
os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
with open(args.table, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["engine", "ratio", "rate", "n_star", "rho_eff", "shift", "measured"])
    w.writerows(out)
print("wrote", args.out + ".png", args.table)

# the table for the paper: n* per engine at the three rates, m/n = 1, 2, 4, 12 (where present)
print(f"{'engine':8} {'m/n':>5}  " + "  ".join(f"{R:>9g} Hz" for R in sorted(rates)) + "   rho_eff")
for name, pts in eng.items():
    for p in pts:
        if p[0] in (1.0, 2.0, 4.0, 12.0):
            cells = "  ".join(f"{p[1] + shift(name, p, R):12.1f}" for R in sorted(rates))
            print(f"{name:8} {p[0]:5g}  {cells}   " + ("model" if name == "sbs" and sbs_shift else f"{p[2]:.3f}"))
for R, pts in measured.items():
    for p in sorted(pts):
        print(f"measured fock at {R:g} Hz: m/n = {p[0]:.2f}  n = {p[1]:.1f}  ({p[2]})")
