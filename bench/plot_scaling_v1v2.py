"""v1 (equal chunks, 1 thread per rank) vs v2 (pattern ownership, interval halo, cores/P threads
per rank) on the 48-core node: total time vs ranks per size, and the v2 breakdown.
    python3 bench/plot_scaling_v1v2.py [--v1 results/keep/cpu48_scaling_small.csv] [--v2 results/keep/cpu48_scaling_v2.csv]"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--v1", default="results/keep/cpu48_scaling_small.csv")
ap.add_argument("--v2", default="results/keep/cpu48_scaling_v2.csv")
ap.add_argument("--out", default="img/cpu_scaling_v2")
args = ap.parse_args()
v1 = list(csv.DictReader(open(args.v1)))
v2 = list(csv.DictReader(open(args.v2)))
sizes = sorted({r["size"] for r in v2}, key=lambda s: int(s.split(":")[0]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
cols = plt.get_cmap("tab10")
for i, s in enumerate(sizes):
    n, m = s.split(":")
    for rows, ls, tag in ((v1, "--", "v1, 1 thread/rank"), (v2, "-", "v2, 48/P threads/rank")):
        pts = sorted([r for r in rows if r["size"] == s], key=lambda r: int(r["P"]))
        if pts:
            axes[0].plot([int(r["P"]) for r in pts], [float(r["total_s"]) for r in pts], "o" + ls, color=cols(i),
                         label=f"n={n} m={m} {tag}", ms=4)
s = sizes[-1]
pts = sorted([r for r in v2 if r["size"] == s], key=lambda r: int(r["P"]))
P = [int(r["P"]) for r in pts]
axes[1].stackplot(P, [float(r["build_s"]) for r in pts], [float(r["comm_s"]) for r in pts], [float(r["compute_s"]) for r in pts],
                  labels=["build (parent indices)", "comm (interval halo, incl. waiting)", "compute (gather)"], alpha=0.8)
ax2 = axes[1].twinx()
ax2.plot(P, [float(r["halo_gb"]) for r in pts], "k.-", label="halo volume (GB, right axis)")
ax2.set_ylabel("halo GB")
ax2.legend(loc="lower right", fontsize=7)
axes[1].set_title(f"v2 breakdown, n={s.split(':')[0]} m={s.split(':')[1]} (max over ranks)")
for ax in axes:
    ax.set_xlabel("ranks P (v2: threads = 48 / P)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_yscale("log")
axes[0].set_ylabel("total time (s)")
axes[1].set_ylabel("seconds")
axes[0].legend(fontsize=7)
axes[1].legend(loc="upper left", fontsize=7)
fig.suptitle("Distributed SLOS_full on one 48-core node: equal-chunk layout (v1) "
             "vs pattern ownership + interval halo + threads (v2)", fontsize=10)
fig.tight_layout()
fig.savefig(args.out + ".png", dpi=150)
fig.savefig(args.out + ".pdf")
print("wrote", args.out)
