"""Plot the loop-interferometer frontier N*(L) of bench/hybrid_frontier.py: one curve per CSV (input, loop loss).

    python3 bench/plot_hybrid_frontier.py results/<stamp>/hybrid_frontier_fock.csv,results/<stamp>/hybrid_frontier_emitter.csv,\
        results/<stamp>/hybrid_frontier_sbs.csv --out img/hybrid_frontier.png [--target 1] [--min-shots 64]

"a.csv+b.csv" merges two runs into one curve (a later file replaces the earlier one's points at the same (ratio, L)).
A CSV of bench/gbs_loop.py (no `input` column; the GBS-L frontier) is drawn as the squeezed-light curve.

N* per (ratio, L) is the log-rate interpolation between the last point at or above the target and the first below it,
exactly as the bench prints it; a bracket counts only if both its points have >= --min-shots shots (t_S
from >= 64-shot brackets), otherwise its upper side is dropped and the point is a lower bound. Unbracketed L
(search cap or memory stop) are left out; a curve whose small-L points all
hit the search cap names it in the legend ("N* > cap at L <= ..."). --show-bounds draws them as open markers instead.
"""
import argparse
import csv
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csvs", help="comma-separated CSV files written by bench/hybrid_frontier.py")
ap.add_argument("--out", default="img/hybrid_frontier.png")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--min-shots", type=int, default=64, help="shots both bracket points need (R1); fewer = lower bound")
ap.add_argument("--show-bounds", action="store_true", help="also draw the unbracketed points (lower bounds) as open markers;"
                                                           " by default they are left out and the search cap is named in"
                                                           " the legend")
args = ap.parse_args()


def n_star(points, target, shots=None):
    """(N*, bracketed) from {N: samples/s}; shots {N: shots} (sector rows only): a bracket counts only if both of its
    points have >= --min-shots shots, otherwise the lower side is a lower bound."""
    ok = (lambda n: True) if shots is None else (lambda n: shots[n] >= args.min_shots)
    lo = [n for n in sorted(points) if points[n] >= target and ok(n)]
    if not lo:
        return None, False
    a = max(lo)
    hi = [n for n in sorted(points) if n > a and points[n] < target]
    if hi and ok(min(hi)):
        b = min(hi)
        return a + (math.log(target) - math.log(points[a])) / (math.log(points[b]) - math.log(points[a])) * (b - a), True
    return a, False


fig, ax = plt.subplots(figsize=(6.4, 5.4))
for item in args.csvs.split(","):
    curves, nshots = defaultdict(dict), defaultdict(dict)
    label = None
    for path in item.split("+"):
        file_curves = defaultdict(dict)
        for r in csv.DictReader(open(path)):
            if "input" not in r:                                          # bench/gbs_loop.py frontier
                r = dict(r, input="GBS-L (squeezed)", ratio="2", sampler="capped register")
            eta = r.get("loop_eta") or ""
            sampler = r.get("sampler") or "dense"
            label = r["input"] + (f", loop eta {eta}" if eta else "") + ", " + {"sector": "sectors"}.get(sampler, sampler) \
                + (f", every {r['spacing']} bins" if r.get("spacing", "1") not in ("", "1") else "")
            key = (float(r["ratio"]), int(r["L"]))
            file_curves[key][int(r["N"])] = (float(r["samples_per_s"]), int(r.get("shots") or 0) if sampler == "sector" else None)
        for key, pts in file_curves.items():                              # a later file replaces the curve at this L
            curves[key] = {n: v[0] for n, v in pts.items()}
            nshots[key] = {n: v[1] for n, v in pts.items()}
    by_ratio = defaultdict(list)
    for (ratio, L), pts in sorted(curves.items()):
        sh = nshots[(ratio, L)]
        ns, ok = n_star(pts, args.target, None if None in sh.values() else sh)
        if ns is not None:
            by_ratio[ratio].append((L, ns, ok))
    for ratio, pts in by_ratio.items():
        tag = ""
        capped = [(L, n) for L, n, o in pts if not o]
        if not args.show_bounds:
            cap = max((n for _, n in capped), default=None)
            at_cap = [L for L, n in capped if n == cap]
            if at_cap and cap is not None and cap >= 1000:              # the search cap, not a memory stop
                tag = f"; N* > {cap} at L <= {max(at_cap)}"
            pts = [p for p in pts if p[2]]
            if not pts:
                continue
        Ls, ns, ok = zip(*pts)
        line, = ax.plot(Ls, ns, "-", label=f"{label} (T/N = {ratio:g}){tag}")
        ax.scatter([l for l, o in zip(Ls, ok) if o], [n for n, o in zip(ns, ok) if o], color=line.get_color(), zorder=3)
        ax.scatter([l for l, o in zip(Ls, ok) if not o], [n for n, o in zip(ns, ok) if not o], facecolors="none",
                   edgecolors=line.get_color(), zorder=3)
ax.set_yscale("log")
ax.set_xlabel("total loop delay L (bins)")
ax.set_ylabel("photons / sources N* at 1 sample/s")
ax.grid(alpha=.3, which="both")
ax.set_title("Loop interferometers: exact sampling frontier on one GPU")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print("wrote", args.out)
