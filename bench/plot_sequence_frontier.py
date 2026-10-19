"""Plot bench/sequence_frontier.py results (docs/SEQUENCE.md §5).

    python3 bench/plot_sequence_frontier.py results/keep/sequence_frontier_h100.csv [--out img/sequence_frontier]

Left: ms per bin-shot against k for each m/k ratio (exponential in k, ∝ S_bin chi^2). Middle:
ms per n-bin sequence against n, one line per k (linear in n), with the 1 sequence/s line and
the bins reachable at 1 Hz. Right: the same instances by the generic route (emitter chain-rule
sampler on the unrolled block-diagonal unitary, exponential in k·n) against the chain.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="img/sequence_frontier")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.csv)) if r.get("ms_per_sequence")]
    rows = [r for r in rows if not (r["k"] == "1" and r["n"] == "4")]      # first point of the run: CUDA warm-up
    for r in rows:
        for key in ("k", "n", "S_bin", "chi"):
            r[key] = int(r[key])
        r["ratio"] = int(r.get("ratio") or 2)
        r["ms"] = float(r["ms_per_sequence"]); r["bs"] = float(r["ms_per_bin_shot"])
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    # (a) per bin-shot vs k, per ratio (median over n)
    for ratio, mk in ((2, "o-"), (4, "s--")):
        byk = defaultdict(list)
        for r in rows:
            if r["ratio"] == ratio:
                byk[r["k"]].append(r["bs"])
        ks = sorted(byk)
        med = [sorted(byk[k])[len(byk[k]) // 2] for k in ks]
        ax[0].semilogy(ks, med, mk, label=f"m = {ratio}k")
        for k, v in zip(ks, med):
            r = next(x for x in rows if x["k"] == k and x["ratio"] == ratio)
            ax[0].annotate(f"{r['S_bin']}", (k, v), textcoords="offset points", xytext=(4, -10 if ratio == 2 else 4), fontsize=6)
    ax[0].set_xlabel("k emitters per bin"); ax[0].set_ylabel("ms per bin-shot"); ax[0].legend(fontsize=8)
    ax[0].set_title("exponential in k (labels: S_bin patterns per bin)", fontsize=9); ax[0].grid(alpha=0.3)
    # (b) per sequence vs n, per k (ratio 2), 1 s line
    for k in sorted({r["k"] for r in rows if r["ratio"] == 2}):
        pts = sorted((r["n"], r["ms"]) for r in rows if r["k"] == k and r["ratio"] == 2)
        ax[1].loglog([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, label=f"k = {k}")
        bs = sorted(r["bs"] for r in rows if r["k"] == k and r["ratio"] == 2)[len(pts) // 2]
        nstar = 1000 / bs
        if nstar < 1e5:
            ax[1].annotate(f"n* = {nstar:,.0f}", (pts[-1][0], pts[-1][1]), textcoords="offset points", xytext=(4, 0), fontsize=6)
    ax[1].axhline(1000, color="k", ls="--", lw=0.8); ax[1].text(4.2, 1200, "1 sequence / s", fontsize=7)
    ax[1].set_xlabel("n bins (log-log: slope 1 = linear)"); ax[1].set_ylabel("ms per n-bin sequence"); ax[1].legend(fontsize=7, ncol=2)
    ax[1].set_title("linear in n; n* = bins at 1 Hz (m = 2k)", fontsize=9); ax[1].grid(alpha=0.3, which="both")
    # (c) generic vs chain against k*n
    for k, mk in ((1, "o"), (2, "s"), (3, "^")):
        pts = sorted((r["k"] * r["n"], float(r["generic_ms_per_sequence"]), r["ms"]) for r in rows
                     if r["k"] == k and r["ratio"] == 2 and r.get("generic_ms_per_sequence"))
        if pts:
            ax[2].semilogy([p[0] for p in pts], [p[1] for p in pts], mk + "--", color=f"C{k}", label=f"unrolled sampler, k = {k}")
            ax[2].semilogy([p[0] for p in pts], [p[2] for p in pts], mk + "-", color=f"C{k}", label=f"chain, k = {k}")
    ax[2].set_xlabel("k·n photons in the sequence"); ax[2].set_ylabel("ms per sequence"); ax[2].legend(fontsize=6)
    ax[2].set_title("generic route (chain rule on the unrolled unitary) vs chain", fontsize=9); ax[2].grid(alpha=0.3)
    fig.suptitle("Time-bin trains through a static interferometer, one H100: k emitters × n bins, dual-rail, 1024 shots", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150); fig.savefig(args.out + ".pdf")
    print("wrote", args.out + ".png/.pdf")


if __name__ == "__main__":
    main()
