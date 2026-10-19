"""F7 — the feed-forward cost table of the emitter-train sampler.

Tabulates, from the local-optics fusion runs kept in results/keep/esbs_fusion_*.csv (run 20260918_084620 for the
Laplace label pass, 20260919_084900 for the Glynn/GEMM pass; H100, 256 shots, T = 40 ticks brick /
rus, 120 ticks circuit), the per-tick-shot cost of `SequenceKernel.sample_diag` with and without a
feed-forward policy, and what a sample of the whole train costs:

  brick    no feed-forward (static brickwork)                        ms per tick-shot, T = 40
  rus      routing feed-forward (optics of bin b+1 chosen per stream)  same measurement, same T
  circuit  gates + routing feed-forward (parity-CNOT random circuit)   T = 120, ~8.4-11.6 layers

The cost of a tick is the median of ms_per_tick_shot over ticks >= 2 (tick 1 carries the Triton
compile of the run, 3 ms at small k) — the per-tick number is what the SEQUENCE.md §9 register
frontier is quoted on. ms per sample = T x that number (one sample = one stream through the T bins);
circuit also gives ms per completed layer = ms per sample / mean completed layers.

    python3 bench/f7_feedforward_cost.py                      # prints the table, writes the CSV
    python3 bench/f7_feedforward_cost.py --csv results/keep/f7_feedforward_cost_20261002.csv --png img/f7_feedforward_cost.png

The ratio columns (rus / brick, circuit / brick at the same k) are the measured price of the
policy: 1.0 within the tick noise for k >= 12, 1.1-1.2 at k <= 10 where a tick is launch-bound and
the per-stream gather of the policy is visible. The exponent is the register's (3^k per tick for
the trajectory sampler, GEMM 2^k x 2^k for the labels), not the policy's.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--glob", default="results/keep/esbs_fusion_*.csv")
ap.add_argument("--csv", default="results/f7_feedforward_cost.csv")
ap.add_argument("--png", default=None, help="figure: ms per tick-shot vs k, one line per program (and the Glynn pass)")
ap.add_argument("--skip-ticks", type=int, default=1, help="ticks dropped at the start (compile warm-up)")
args = ap.parse_args()


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# key = (label_pass, optics, program, gates) -> {k: (T, median ms/tick-shot, mean layers)}
series = {}
for path in sorted(glob.glob(args.glob)):
    pas = "glynn" if "glynn" in os.path.basename(path) else "laplace"
    rows = load(path)
    by_k = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(r)
    for k, rs in sorted(by_k.items()):
        rs = sorted(rs, key=lambda r: int(r["tick"]))
        T = int(rs[-1]["tick"])
        ms = np.array([float(r["ms_per_tick_shot"]) for r in rs[args.skip_ticks:]])
        lay = [float(r["layers"]) for r in rs if r["layers"] not in ("", None)]
        key = (pas, rs[0]["optics"], rs[0]["program"], rs[0]["gates"])
        series.setdefault(key, {})[k] = (T, float(np.median(ms)), float(np.max(ms)), max(lay) if lay else None, int(rs[0]["m"]))

def ratio_to_brick(base, k, med):
    """Policy / brick per-tick cost at the same k; the circuit runs at odd k, where the brick point is the
    geometric mean of its k - 1 and k + 1 neighbours (the brick cost is close to geometric in k)."""
    if k in base:
        return med / base[k][1]
    if k - 1 in base and k + 1 in base:
        return med / np.sqrt(base[k - 1][1] * base[k + 1][1])
    return None


head = ["label_pass", "optics", "program", "gates", "feed_forward", "k", "m", "ticks", "ms_per_tick_shot",
        "ms_per_tick_shot_max", "ms_per_sample", "layers", "ms_per_layer", "ratio_to_brick"]
FF = {"brick": "none", "rus": "routing", "circuit": "gates+routing"}
out = []
for (pas, opt, prog, gates), d in sorted(series.items()):
    base = series.get((pas, opt, "brick", "random"), {})
    for k, (T, med, mx, lay, m) in sorted(d.items()):
        per_sample = T * med
        ratio = ratio_to_brick(base, k, med)
        out.append([pas, opt, prog, gates, FF[prog], k, m, T, round(med, 4), round(mx, 4), round(per_sample, 3),
                    lay, round(per_sample / lay, 3) if lay else None, round(ratio, 3) if ratio else None])

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
with open(args.csv, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(head)
    w.writerows(out)

# the paper table: xfusion, random gates, Laplace pass — brick / rus / circuit side by side, + the Glynn brick
lap = {p: series.get(("laplace", "xfusion", p, "random"), {}) for p in ("brick", "rus", "circuit")}
gly = series.get(("glynn", "xfusion", "brick", "random"), {})
ks = sorted(set(lap["brick"]) | set(lap["rus"]) | set(lap["circuit"]) | set(gly))
print(f"{'k':>3} | {'brick (no FF)':>14} {'rus (routing FF)':>17} {'ratio':>6} | "
      f"{'circuit (gates+routing FF)':>27} {'ratio':>6} {'layers':>7} {'ms/layer':>9} | {'Glynn brick':>12}")
print("    |   ms per tick-shot (median over ticks >= 2); ratio = policy / brick at the same k (odd k: geometric mean of k -+ 1)")
for k in ks:
    b = lap["brick"].get(k); r = lap["rus"].get(k); c = lap["circuit"].get(k); g = gly.get(k)
    f = lambda x: f"{x[1]:8.4f}" if x else "       -"
    rb = f"{r[1] / b[1]:6.2f}" if (r and b) else "     -"
    rc_v = ratio_to_brick(lap["brick"], k, c[1]) if c else None
    rc = f"{rc_v:6.2f}" if rc_v else "     -"
    lay = f"{c[3]:7.2f}" if c else "      -"
    mpl = f"{c[0] * c[1] / c[3]:9.3f}" if c else "        -"
    print(f"{k:3d} | {f(b):>14} {f(r):>17} {rb} | {f(c):>27} {rc} {lay} {mpl} | {f(g):>12}")
print("wrote", args.csv)

if args.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style = {"brick": ("o-", "no feed-forward (brickwork)"), "rus": ("s--", "routing feed-forward (RUS)"),
             "circuit": ("^-.", "gates + routing feed-forward (circuit)")}
    for p, (st, lab) in style.items():
        d = lap[p]
        if d:
            ax.semilogy(sorted(d), [d[k][1] for k in sorted(d)], st, label=lab + ", Laplace labels", ms=5)
    if gly:
        ax.semilogy(sorted(gly), [gly[k][1] for k in sorted(gly)], "d:", color="k", label="no feed-forward, Glynn/GEMM labels", ms=5)
    ax.set_xlabel("emitters k (m = 2k rails)")
    ax.set_ylabel("ms per tick-shot (H100, 256 streams)")
    ax.set_title("Emitter-train sampler: the feed-forward policy costs nothing per tick")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.png, dpi=150)
    print("wrote", args.png)
