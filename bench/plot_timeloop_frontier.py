"""Plot the time-loop frontier from a timeloop_frontier.py CSV.

    python3 bench/plot_timeloop_frontier.py [--csv results/keep/timeloop_frontier.csv] [--out img/timeloop_frontier]
Left: samples/s vs N for every loop length L at --plot-ratio (log-log; the 1 sample/s line).
Right: N*(L) at 1 sample/s (interpolated between the bracketing points), one curve per bin
ratio T/N, against the generic Fock frontier (n* = 37.8 at m = 2n, one H100); the register
dimension D at N* is annotated on the --plot-ratio curve.
"""
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/keep/timeloop_frontier.csv")
ap.add_argument("--out", default="img/timeloop_frontier")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--fock", type=float, default=37.8, help="generic Fock frontier at m = 2n")
ap.add_argument("--plot-ratio", type=float, default=2.0)
ap.add_argument("--title", default="Native time-loop sampler, one NVIDIA H100 PCIe")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
data = {}                                                         # ratio -> L -> [(N, rate, D)]
for r in rows:
    ratio = float(r.get("ratio", 2.0))
    data.setdefault(ratio, {}).setdefault(int(r["L"]), []).append((int(r["N"]), float(r["samples_per_s"]), int(r["D"])))
for byL in data.values():
    for L in byL:
        byL[L].sort()


def frontier(byL):
    out = {}
    for L, pts in byL.items():
        lo = [p for p in pts if p[1] >= args.target]
        hi = [p for p in pts if p[1] < args.target]
        if lo and hi:
            a, b = max(lo), min(hi, key=lambda p: p[0])
            n_star = a[0] + (math.log(args.target) - math.log(a[1])) / (math.log(b[1]) - math.log(a[1])) * (b[0] - a[0])
            out[L] = (n_star, a[2])
        elif lo:
            out[L] = (max(lo)[0], max(lo)[2])
    return out


plot_ratio = args.plot_ratio if args.plot_ratio in data else sorted(data)[0]
byL = data[plot_ratio]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
cmap = plt.get_cmap("viridis")
Ls = sorted(byL)
for i, L in enumerate(Ls):
    pts = byL[L]
    color = cmap(i / max(1, len(Ls) - 1))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, label=f"L = {L}", ms=4)
stars = frontier(byL)
ax.axhline(args.target, color="k", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("photons N")
ax.set_ylabel("samples / s")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
ax.set_title(f"throughput vs N per total loop length L (T = {plot_ratio:g} N bins)")

reds = plt.get_cmap("autumn")
for j, ratio in enumerate(sorted(data)):
    st = frontier(data[ratio])
    xs = sorted(st)
    ax2.plot(xs, [st[L][0] for L in xs], "s-", color=reds(0.75 * j / max(1, len(data) - 1)),
             label=f"native loop sampler, T = {ratio:g} N")
xs = sorted(stars)
ax2.axhline(args.fock, color="tab:blue", ls="--", label=f"generic Fock sampler n* = {args.fock} (m = 2n)")
for L in xs:
    ax2.annotate(f"D={stars[L][1]:,}", (L, stars[L][0]), textcoords="offset points", xytext=(4, 4), fontsize=7)
ax2.set_xlabel("total loop length L (bins)")
ax2.set_ylabel("N* at 1 sample/s")
ax2.set_yscale("log")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=8)
ax2.set_title("frontier N*(L)")
fig.suptitle(args.title, fontsize=11)
fig.tight_layout()
fig.savefig(args.out + ".png", dpi=150)
fig.savefig(args.out + ".pdf")
for ratio in sorted(data):
    print(f"T = {ratio:g} N: N* per L:", {L: round(v[0], 1) for L, v in frontier(data[ratio]).items()})
print(f"wrote {args.out}.png/.pdf")
