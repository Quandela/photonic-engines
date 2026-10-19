"""Plot bench/sequence_ticks.py results (docs/SEQUENCE.md §9).

    python3 bench/plot_sequence_ticks.py results/keep/sequence_ticks_haar_h100.csv
        [results/keep/sequence_ticks_band2_h100.csv ...] [--out img/sequence_ticks]

Left: middle-cut entanglement entropy of the spin register against the tick, one line per k
(fraction of the maximum k/2), Haar solid / banded dashed. Middle: bond dimension holding
99.9 % of the Schmidt weight against the tick, with the exact maximum 2^{k/2}. Right: ms per
tick-shot against k (the sampling cost of one tick once the register is generic), with the
1 tick/s line.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="img/sequence_ticks")
    args = ap.parse_args()
    rows = []
    for path in args.csv:
        rows += list(csv.DictReader(open(path)))
    for r in rows:
        for key in ("k", "tick", "max_rank"):
            r[key] = int(r[key])
        for key in ("entropy", "max_entropy", "bond999", "rank", "ms_per_tick_shot"):
            r[key] = float(r[key])
    unis = sorted({r["unitary"] for r in rows}, key=lambda u: (u != "haar", u))
    styles = {u: ("-" if u == "haar" else "--") for u in unis}
    cmap = plt.get_cmap("viridis")
    ks = sorted({r["k"] for r in rows})
    color = {k: cmap(i / max(1, len(ks) - 1)) for i, k in enumerate(ks)}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for u in unis:
        for k in ks:
            rs = sorted([r for r in rows if r["unitary"] == u and r["k"] == k], key=lambda r: r["tick"])
            if not rs:
                continue
            t = [r["tick"] for r in rs]
            ax[0].plot(t, [r["entropy"] / r["max_entropy"] for r in rs], styles[u], marker="o", ms=3, color=color[k],
                       label=f"k = {k}" if u == "haar" else None)
            ax[1].semilogy(t, [r["bond999"] for r in rs], styles[u], marker="o", ms=3, color=color[k])
            ax[1].axhline(rs[0]["max_rank"], color=color[k], lw=0.5, alpha=0.5)
    ax[0].set(xlabel="tick", ylabel="S(middle cut) / (k/2)", ylim=(0, 1.05), title="register entanglement, entropy")
    ax[1].set(xlabel="tick", ylabel="bond for 99.9 % of the weight", title="register bond dimension (lines: 2^{k/2})")
    for u in unis:
        byk = defaultdict(list)
        for r in rows:
            if r["unitary"] == u and r["tick"] >= 2:
                byk[r["k"]].append(r["ms_per_tick_shot"])
        kk = sorted(byk)
        ax[2].semilogy(kk, [sorted(byk[k])[len(byk[k]) // 2] for k in kk], styles[u], marker="o", ms=4, color="k",
                       label=("Haar U" if u == "haar" else f"banded U ({u})"))
    ax[2].axhline(1e3, color="gray", lw=0.8, ls=":")
    ax[2].text(kk[0] if kk else 0, 1.2e3, "1 tick / s", color="gray", fontsize=8)
    ax[2].set(xlabel="k emitters (m = 2k)", ylabel="ms per tick-shot (tick ≥ 2)", title="cost of one tick, label-diagonal sampler")
    ax[0].legend(fontsize=8, ncol=2)
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
