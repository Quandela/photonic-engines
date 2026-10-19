"""Plot bench/esbs_fusion.py results: register entropy vs tick per k (one panel per CSV = one
(optics, program)), and the final entropy vs k across panels (volume law: ∝ k; area law: flat).

    python3 bench/plot_esbs_fusion.py results/<stamp>/esbs_fusion_*.csv --out results/<stamp>/esbs_fusion.png
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = [(os.path.basename(p)[len("esbs_fusion_"):-4], load(p)) for p in args.csvs]
    files = [(n, r) for n, r in files if r]
    n = len(files)
    fig, axs = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.8))
    cmap = plt.get_cmap("viridis")
    finals = {}
    for ax, (name, rows) in zip(axs, files):
        ks = sorted({int(r["k"]) for r in rows})
        for i, k in enumerate(ks):
            rk = [r for r in rows if int(r["k"]) == k]
            x = [int(r["tick"]) for r in rk]
            y = [float(r["entropy"]) for r in rk]
            ax.plot(x, y, "-", color=cmap(i / max(1, len(ks) - 1)), label=f"k={k} (max {rk[0]['max_entropy']})")
            finals.setdefault(name, []).append((k, y[-1], float(rk[0]["max_entropy"])))
        ax.set_title(name.replace("_", " "), fontsize=9); ax.set_xlabel("tick"); ax.set_ylabel("register entropy at the cut (bits)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
    ax = axs[-1]
    for name, pts in finals.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name.replace("_", " "))
    ks_all = sorted({p[0] for pts in finals.values() for p in pts})
    ax.plot(ks_all, [k // 2 for k in ks_all], "k:", label="k/2 (Page bound, middle cut)")
    ax.set_xlabel("k emitters"); ax.set_ylabel("entropy at the last tick (bits)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
    ax.set_title("final entropy vs k", fontsize=9)
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(args.csvs[0]), "esbs_fusion.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
