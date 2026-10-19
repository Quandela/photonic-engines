"""The lossy simulability frontier n*(eta) of every engine with a loss curve, from the kept CSVs (no measurement).

    python3 bench/plot_frontier_loss.py [--out img/frontier_loss_all]

Left: n* (sources: photons for Fock and the emitter, sources for SBS, squeezers for GBS) at 1 sample/s on one H100,
m = 2n, against the transmission eta, with each engine's loss law as a thin line:
  Fock      n*(eta) = n*(1) log rho / log(1 - eta + eta rho), rho = its lossless rho_eff (docs/LOSS.md);
  SBS       the same with the survival probability eta/2 of a source that emits half the time (bench/frontier_loss.py);
  GBS       n*(1) log Phi(1) / log Phi(eta), Phi(eta) = sech r (1 - (1 - eta + eta sqrt 2)^2 tanh^2 r)^(-1/2): the
            collision-free, UNtruncated per-squeezer factor E[sqrt(2)^N] (docs/GAUSSIAN_SURVEY.md §6.1), against the
            t_S points of the timing model (docs/GAUSSIAN.md §2.3), so the gap between them is the truncation at S = 1e4.
The emitter engine has no simple law (host work per surviving-photon group, bench/bench_loss.py): points only.
Right: the mean number of detected photons at n* (eta n*; eta n*/2 for SBS; eta 0.5 n* for GBS at 0.5 photons per
squeezer).
Sources: results/keep/frontier_loss_run1.csv (fock), frontier_loss_run2_emitter.csv (emitter),
sbs_20260920_frontier_loss.csv with eta = 0.5 from the rerun sbs_20260921_frontier_loss_eta05.csv, and
gbs_frontier_grid_20260925_h100.csv (S = 1e4, ratio 2, 0.5 photons per squeezer, one Schmidt mode).
"""
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--keep", default="results/keep")
ap.add_argument("--out", default="img/frontier_loss_all")
args = ap.parse_args()
K = args.keep


def rows(name):
    return list(csv.DictReader(open(f"{K}/{name}")))


def pts(rs, engine):
    out = {}
    for r in rs:
        if r["engine"] == engine and r["n_star"] not in ("nan", ""):
            out[float(r["eta"])] = (float(r["n_star"]), float(r.get("rho_eff") or "nan"))
    return out


fock = pts(rows("frontier_loss_run1.csv"), "fock")
emit = pts(rows("frontier_loss_run2_emitter.csv"), "emitter")
sbs = pts(rows("sbs_20260920_frontier_loss.csv"), "sbs")
sbs.update(pts(rows("sbs_20260921_frontier_loss_eta05.csv"), "sbs"))           # the eta = 0.5 rerun
gbs = {}
for r in rows("gbs_frontier_grid_20260925_h100.csv"):
    if (float(r["S"]) == 1e4 and float(r["ratio"]) == 2.0 and float(r["mean"]) == 0.5 and r["schmidt"] == "1"):
        gbs[float(r["eta"])] = (float(r["n_star"]), float("nan"))

etas = [e / 100 for e in range(50, 101)]                                   # the measured range


def fock_law(n1, rho, p_of_eta):
    return [n1 * math.log(rho) / math.log(1 - p_of_eta(e) + p_of_eta(e) * rho) for e in etas]


r0 = math.asinh(math.sqrt(0.5))                                            # sinh^2 r = 0.5


def phi(eta):
    t2 = math.tanh(r0) ** 2 * (1 - eta + eta * math.sqrt(2)) ** 2
    return 1 / (math.cosh(r0) * math.sqrt(1 - t2))


engines = [("fock", fock, "tab:blue", "Fock input", lambda e: e, 1.0),
           ("emitter", emit, "tab:red", "linear-cluster caterpillar (emitter)", None, 1.0),
           ("sbs", sbs, "tab:green", "superposition BS, alpha = pi/4", None, 0.5),
           ("gbs", gbs, "tab:orange", "Gaussian BS, 0.5 photons/squeezer (timing model, t_S, S = 1e4)", None, 0.5)]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
for name, P, color, label, _, yield_ in engines:
    xs = sorted(P)
    ax.plot(xs, [P[x][0] for x in xs], "o-", color=color, lw=2, label=label)
    for x in xs:
        ax.annotate(f"{P[x][0]:.1f}", (x, P[x][0]), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7, color=color)
    ax2.plot(xs, [x * yield_ * P[x][0] for x in xs], "o-", color=color, lw=2, label=label)
    if name == "fock":
        ax.plot(etas, fock_law(P[1.0][0], P[1.0][1], lambda e: e), "-", color=color, lw=0.8, alpha=0.6,
                label=f"Fock law, rho_eff(1) = {P[1.0][1]:.2f}")
    elif name == "sbs" and P[1.0][1] == P[1.0][1]:
        ax.plot(etas, [P[1.0][0] * math.log(1 - 0.5 + 0.5 * P[1.0][1]) / math.log(1 - e / 2 + e / 2 * P[1.0][1])
                       for e in etas], "-", color=color, lw=0.8, alpha=0.6, label="SBS law (survival eta/2)")
    elif name == "gbs":
        ax.plot(etas, [P[1.0][0] * math.log(phi(1.0)) / math.log(phi(e)) for e in etas], "-", color=color, lw=0.8,
                alpha=0.6, label="GBS collision-free law, untruncated E[sqrt2^N]")
for a in (ax, ax2):
    a.invert_xaxis(); a.grid(True, alpha=0.3); a.set_xlabel("transmission eta")
ax.set_ylabel("n* (sources) at 1 sample / s, m = 2n")
ax2.set_ylabel("mean detected photons at n*")
ax.set_title("Lossy simulability frontier, one NVIDIA H100 PCIe", fontsize=10)
ax2.set_title("detected photons at the frontier", fontsize=10)
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax2.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.tight_layout(); fig.savefig(f"{args.out}.png", dpi=160, bbox_inches="tight"); print(f"wrote {args.out}.png")
for name, P, *_ in engines:
    print(name, {e: P[e][0] for e in sorted(P)})
