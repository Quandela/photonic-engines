"""Plot the simulability frontier n*(m/n) of both samplers from a map_threshold.py CSV.

    python3 bench/plot_frontier.py [--csv results/keep/frontier.csv] [--out img/frontier]
Left: n* at 1 sample/s on one H100 for the Fock sampler and the emitter (linear-cluster) sampler,
with the 8- and 64-GPU extrapolations n* + log(G)/log(rho_eff). Right: the gap in photons and
the per-photon factors rho_eff against the Fock theory rho(c).
"""
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/keep/frontier.csv")
ap.add_argument("--out", default="img/frontier")
ap.add_argument("--title", default="Simulability frontier at 1 sample/s, one NVIDIA H100 PCIe")
ap.add_argument("--panel", default="both", choices=["both", "left"], help="'left' writes the n*(m/n) "
                "panel alone, which is what the deck embeds as Images/frontier_left.png — it had no "
                "generator before and went stale")
args = ap.parse_args()


def rho(c):
    return (2 * c + 1) ** (2 * c + 1) / ((4 * c) ** c * (c + 1) ** (c + 1))


rows = [r for r in csv.DictReader(open(args.csv)) if r["n_star"] not in ("nan", "")]
eng = {}
for r in rows:
    eng.setdefault(r["engine"], []).append((float(r["ratio"]), float(r["n_star"]), float(r["rho_eff"])))
for k in eng:
    eng[k].sort()
style = {"fock": ("tab:blue", "Fock input (Clifford & Clifford 2020)"),
         "emitter": ("tab:red", "linear-cluster caterpillar state (emitter chain rule)"),
         "sbs": ("tab:green", "superposition BS, alpha = pi/4 (n sources, n/2 photons on average)"),
         # GBS: n squeezers at 0.5 photons each; n* from the calibrated timing model and the run-of-S statistic t_S
         # (S = 1e4), not a bracketed mean over unconditioned shots (docs/GAUSSIAN.md §2.1-2.3)
         "gbs": ("tab:orange", "Gaussian BS, 0.5 photons/squeezer (timing model, t_S, S = 1e4)")}
ratios = sorted({p[0] for v in eng.values() for p in v})

if args.panel == "left":
    fig, ax = plt.subplots(figsize=(7, 4.67))
    ax2 = None
else:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
for name, pts in eng.items():
    color, label = style[name]
    xs = [p[0] for p in pts]
    ax.plot(xs, [p[1] for p in pts], "o-", color=color, lw=2, label=f"{label}, 1 GPU")
    for G, ls in ((8, "--"), (64, ":")):
        ys = [p[1] + math.log(G) / math.log(p[2] if 1.3 <= p[2] <= 3.5 else rho(p[0])) for p in pts]
        ax.plot(xs, ys, ls, color=color, alpha=0.7, label="_nolegend_")   # style, not engine: proxies below
    dy = {"fock": 6, "emitter": -12}.get(name, 6)      # sbs sits far below the other two: label above
    if "gbs" in eng:                                    # gbs runs 0.5-4 above fock: fock's labels go below
        dy = {"fock": -12, "emitter": -12}.get(name, 6)
    for x, y, _ in pts:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=7, color=color)
ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator(ratios)); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("mode ratio m / n"); ax.set_ylabel("photon number n* at 1 sample / s")
ax.set_title(args.title, fontsize=10); ax.grid(True, which="both", alpha=0.3)
# one entry per engine plus two for the GPU-count styles: three engines x three lines each is a
# legend box that covers whichever curve it is placed over, whatever the corner
proxy = [Line2D([], [], color="0.4", ls=ls, label=f"{G} GPUs (extrapolated)") for G, ls in ((8, "--"), (64, ":"))]
h, l = ax.get_legend_handles_labels()
if len(eng) > 3:                        # four engines: no strip is wide enough, the legend goes under the axes
    ax.legend(h + proxy, l + [q.get_label() for q in proxy], fontsize=6.5, ncol=1, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
else:
    if len(eng) > 2:                    # sbs is ~15 photons against 30 and 38: every corner is taken
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.34 * (hi - lo), hi)   # open a strip under the curves
    ax.legend(h + proxy, l + [q.get_label() for q in proxy], fontsize=7, ncol=2 if len(eng) > 2 else 1,
              loc="lower center" if len(eng) > 2 else "upper right")

# right: gap and rho
if ax2 is not None and "fock" in eng and "emitter" in eng:
    fock = {p[0]: p[1] for p in eng["fock"]}
    gap = [(p[0], fock[p[0]] - p[1]) for p in eng["emitter"] if p[0] in fock]
    ax2.plot([g[0] for g in gap], [g[1] for g in gap], "s-", color="tab:purple", label="n*(Fock) − n*(emitter)  [photons]")
    ax2.set_ylabel("gap in photons", color="tab:purple"); ax2.set_ylim(0, max(g[1] for g in gap) * 1.6)
    ax3 = ax2.twinx()
    for name, pts in eng.items():
        ax3.plot([p[0] for p in pts], [p[2] for p in pts], "o--", color=style[name][0], alpha=0.8, label=f"rho_eff, {name}")
    cs = [c / 10 for c in range(10, 125)]
    ax3.plot(cs, [rho(c) for c in cs], "-", color="gray", alpha=0.6, label="rho(c), Fock theory (arXiv:2005.04214)")
    # sbs's per-SOURCE factor is ~2.3-3.2, well off the Fock/emitter band: widen rather than clip it
    hi = max(2.2, max(p[2] for v in eng.values() for p in v) * 1.05)
    lo3 = min(1.6, min(p[2] for v in eng.values() for p in v) * 0.95)   # gbs: ~1.38 per squeezer
    ax3.set_ylabel("per-photon cost factor rho"); ax3.set_ylim(lo3, hi)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    if len(eng) > 3:                    # the gbs rho line runs along the bottom: legend under the axes
        ax2.legend(h1 + h2, l1 + l2, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    else:
        ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
if ax2 is not None:
    ax2.set_xscale("log")
    ax2.xaxis.set_major_locator(FixedLocator(ratios)); ax2.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("mode ratio m / n"); ax2.grid(True, which="both", alpha=0.3)
    ax2.set_title("gap between the samplers and per-photon factors", fontsize=10)
fig.tight_layout()
fig.savefig(f"{args.out}.png", dpi=160, **({"bbox_inches": "tight"} if len(eng) > 3 else {}))   # legends below
print(f"wrote {args.out}.png")
