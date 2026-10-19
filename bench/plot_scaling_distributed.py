"""Plot a scaling_distributed.sh CSV: total time and speedup vs ranks per size, and the
build / comm / compute breakdown.   python3 bench/plot_scaling_distributed.py --csv ... --out img/cpu_scaling"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/keep/cpu48_scaling_small.csv")
ap.add_argument("--out", default="img/cpu_scaling")
ap.add_argument("--title", default="Distributed SLOS_full on one 48-core node (mpich, complex128)")
args = ap.parse_args()
rows = list(csv.DictReader(open(args.csv)))
sizes = sorted({r["size"] for r in rows}, key=lambda s: int(s.split(":")[0]))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for s in sizes:
    pts = sorted([r for r in rows if r["size"] == s], key=lambda r: int(r["P"]))
    P = [int(r["P"]) for r in pts]
    tot = [float(r["total_s"]) for r in pts]
    n, m = s.split(":")
    lab = f"n={n} m={m} (M={int(pts[0]['M_n']):.2g})"
    axes[0].plot(P, tot, "o-", label=lab)
    axes[1].plot(P, [tot[0] / t for t in tot], "o-", label=lab)
    if s == sizes[-1]:
        axes[2].stackplot(P, [float(r["build_s"]) for r in pts], [float(r["comm_s"]) for r in pts],
                          [float(r["compute_s"]) for r in pts],
                          labels=["build (index arithmetic + halo bookkeeping)",
                                  "comm (Alltoallv halo, incl. waiting)", "compute (gather)"], alpha=0.8)
        axes[2].plot(P, [float(r["build_min_s"]) for r in pts], "k--", lw=0.8, label="fastest rank's build")
        axes[2].set_title(f"breakdown, {lab}")
axes[1].plot([1, 48], [1, 48], "k:", lw=0.8, label="ideal")
for ax, yl in zip(axes, ["total time (s)", "speedup vs 1 rank", "seconds (max over ranks)"]):
    ax.set_xlabel("ranks P")
    ax.set_ylabel(yl)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
axes[0].set_yscale("log")
axes[1].set_yscale("log")
fig.suptitle(args.title, fontsize=11)
fig.tight_layout()
fig.savefig(args.out + ".png", dpi=150)
fig.savefig(args.out + ".pdf")
print("wrote", args.out + ".png")
