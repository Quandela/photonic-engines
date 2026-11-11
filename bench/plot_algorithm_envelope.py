"""Figure F2a: the algorithm envelope on one instance family (docs/ALGORITHMS.md §5).

    python3 bench/plot_algorithm_envelope.py [--csv results/keep/algorithm_envelope_<stamp>.csv] [--out img/algorithm_envelope]
    python3 bench/plot_algorithm_envelope.py --csv a.csv,b.csv           # several runs merged (later files win on a repeated key)

One panel per mode ratio: seconds per exact sample against n for every algorithm of the T1 set,
measured on the same Haar instances and the same GPU, log scale, the 1 sample/s line, and the lower
envelope (the frontier is the minimum over algorithms). Every curve gets a log-linear fit over its
measured points (slope = ln rho_eff) and its crossing of the 1 Hz line, printed as the T1 column
"n* at 1 Hz (fitted)"; the bracketed n* of `find_threshold --algo` is the number the table quotes
when it exists, this fit is the fallback. "strong" is drawn as the run-of-S cost (build / S + draw)
and stops where its table no longer fits — the memory frontier, not a rate.
"""
import argparse
import csv
import math
from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/keep/algorithm_envelope.csv")
ap.add_argument("--out", default="img/algorithm_envelope")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--fit-min-ms", type=float, default=10.0,
                help="the log-linear fit uses the points at or above this cost only (the small-n points are latency-bound)")
ap.add_argument("--title", default="Exact single-photon samplers on one instance family (Haar U, one H100 PCIe)")
args = ap.parse_args()

LABEL = {
    "2018": "C&C 2018 (shared cofactors, 2^k)",
    "2020": "C&C 2020 (output collapse, this engine, Triton)",
    "auto": "this engine, production path (CUDA)",
    "input": "input-side collapse (Brod–Oszmaniec / theboss form)",
    "percof": "one sweep per cofactor (PiquassoBoost form)",
    "strong": "strong simulation then sample (SLOS, run of S)",
}
STYLE = {"2018": ("C3", "o"), "2020": ("C0", "s"), "auto": ("C0", "^"), "input": ("C1", "D"), "percof": ("C2", "v"), "strong": ("C4", "x")}

pts: dict = {}                                                  # (algo, ratio) -> {n: ms}
for path in args.csv.split(","):
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["samples_per_s"]:
                continue
            pts.setdefault((r["algo"], float(r["ratio"])), {})[int(r["n"])] = float(r["ms_per_sample"]) / 1e3
ratios = sorted({c for _, c in pts})
fig, axes = plt.subplots(1, len(ratios), figsize=(6.2 * len(ratios), 4.8), squeeze=False)
print(f"{'algo':8s} {'c':>4s} {'rho_eff':>8s} {'n* (fit)':>9s} {'n* (bracket)':>13s}  n measured (fit over n >= {args.fit_min_ms:g} ms)")
for ax, c in zip(axes[0], ratios):
    env: dict = {}
    for algo in ("2018", "2020", "auto", "input", "percof", "strong"):
        d = pts.get((algo, c))
        if not d:
            continue
        ns = sorted(d)
        t = np.array([d[n] for n in ns])
        col, mk = STYLE[algo]
        ax.plot(ns, t, marker=mk, color=col, ls="-" if algo != "auto" else "--", ms=5, label=LABEL[algo])
        for n in ns:
            env[n] = min(env.get(n, math.inf), d[n])
        sel = [i for i, n in enumerate(ns) if t[i] >= args.fit_min_ms / 1e3]
        if len(sel) < 2:
            sel = list(range(len(ns)))
        # bracket: the largest n at or under the target cost and the next measured n above it
        tt = 1.0 / args.target
        lo = [i for i, n in enumerate(ns) if t[i] <= tt]
        brk = ""
        if lo and lo[-1] + 1 < len(ns):
            i = lo[-1]
            brk = f"{ns[i] + (math.log(tt) - math.log(t[i])) / (math.log(t[i + 1]) - math.log(t[i])) * (ns[i + 1] - ns[i]):.1f}"
        if len(ns) >= 2 and algo != "strong":
            b, a = np.polyfit([ns[i] for i in sel], np.log(t[sel]), 1)
            n_star = (math.log(tt) - a) / b if b > 0 else float("nan")
            xs = np.linspace(ns[sel[0]], max(max(ns), n_star if n_star == n_star else max(ns)), 50)
            ax.plot(xs, np.exp(a + b * xs), color=col, lw=0.8, alpha=0.5)
            print(f"{algo:8s} {c:4g} {math.exp(b):8.3f} {n_star:9.1f} {brk:>13s}  {ns}")
        else:
            print(f"{algo:8s} {c:4g} {'':>8s} {'':>9s} {brk:>13s}  {ns}")
    if env:
        ne = sorted(env)
        # one segment per contiguous stretch of measured n (the strong and chain-rule grids do not meet)
        segs, cur = [], [ne[0]]
        for a, b in pairwise(ne):
            if b - a > 3:
                segs.append(cur); cur = []
            cur.append(b)
        segs.append(cur)
        for i, seg in enumerate(segs):
            ax.plot(seg, [env[n] for n in seg], color="k", lw=2.2, alpha=0.35, zorder=0, label="lower envelope" if i == 0 else None)
    ax.axhline(1.0 / args.target, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(f"photonic inputs n (single photons), m = {c:g} n")
    ax.set_ylabel("seconds per exact sample")
    ax.set_title(f"m = {c:g} n")
    ax.grid(alpha=0.3, which="both")
axes[0][0].legend(fontsize=7.5, loc="upper left")
fig.suptitle(args.title, fontsize=11)
fig.tight_layout()
fig.savefig(f"{args.out}.png", dpi=160)
print("wrote", f"{args.out}.png")
