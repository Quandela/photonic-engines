"""Plot bench/esbs_design_plane.py outputs: one panel per quantity, x = k, one line per unitary
family, values at the last tick averaged over settings (register entropy fraction, bin flatness,
inter-bin training signal against its shot noise, tick cost).

    python3 bench/plot_esbs_design_plane.py results/<stamp>/esbs_design_plane_*.csv --out img/esbs_design_plane.png
"""
import argparse
import collections
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="img/esbs_design_plane.png")
    args = ap.parse_args()
    rows = load(args.csv)
    last = collections.defaultdict(list)                                  # (unitary, k) -> rows at the last tick
    for r in rows:
        last[(r["unitary"], int(r["k"]))].append(r)
    keys = sorted(last)
    unis = sorted({u for u, _ in keys}, key=lambda u: (u != "haar", u))
    panels = [
        ("entropy / (k/2)", lambda rs: sum(fnum(r["entropy"]) for r in rs) / len(rs) / max(int(rs[0]["max_entropy"]), 1), None),
        ("bond_999 / 2^(k/2)", lambda rs: sum(fnum(r["bond999"]) for r in rs) / len(rs) / int(rs[0]["max_rank"]), "log"),
        ("Z2 of the 4-mode marginal", lambda rs: sum(fnum(r["Z2_4mode"]) for r in rs) / len(rs), None),
        ("inter-bin signal: sig_std / sig_err", lambda rs: fnum(rs[0]["sig_std"]) / max(fnum(rs[0]["sig_err"]), 1e-12), "log"),
        ("Z2 of the bin (exact, k <= strong-kmax)", lambda rs: fnum(rs[0]["Z2_bin_exact"]), None),
        ("ms per tick-shot", lambda rs: sum(fnum(r["ms_per_tick_shot"]) for r in rs) / len(rs), "log"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (title, f, scale) in zip(axes.flat, panels):
        for u in unis:
            ks, ys = [], []
            for (uu, k) in keys:
                if uu != u:
                    continue
                rs = last[(uu, k)]
                tmax = max(int(r["tick"]) for r in rs)
                rs = [r for r in rs if int(r["tick"]) == tmax]
                y = f(rs)
                if not math.isnan(y):
                    ks.append(k); ys.append(y)
            if ks:
                ax.plot(ks, ys, "o-", label=u)
        ax.set_title(title); ax.set_xlabel("k emitters"); ax.grid(alpha=0.3)
        if scale:
            ax.set_yscale(scale)
        if title.startswith("inter-bin"):
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
    axes.flat[0].legend()
    fig.suptitle("Emitter-state sequence generator: design plane (last tick, mean over settings)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
