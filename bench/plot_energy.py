"""
Energy per exact sample: measured on one H100 against the classical flop lower bound and the
quantum-device figure of Soret et al. (arXiv:2601.08068).

Ported from gpu-clifford (bench/plot_energy.py, 2026-09) and re-pointed at the
photonic-engines CSVs, because both curves move with the kernel and the frontier:

  * the measured GPU curve is P / R_GPU(n), so it drops by whatever the kernel gained
    (1.35x at n = 24 to 1.93x at n = 50 with the step-4 tier ladder);
  * beyond the measured range it is extrapolated as rho_eff^(n* - n), so it also moves with
    the frontier CSV;
  * the vertical 1-Hz frontier line moves with n*.

Note one invariant worth stating rather than recomputing: the energy *at* the frontier is
P x 1 s = 280 W / 3600 = 7.8e-2 Wh whatever n* turns out to be, by the definition of n*.
What the kernel changes is the energy per sample at a FIXED n, which is the quantity the
Soret et al. comparison is about, so the crossing moves but the frontier point does not.

    python3 bench/plot_energy.py --csv results/<stamp>/frontier_v2.csv \\
            --rates results/<stamp>/rate_r2.csv --out img/energy

With no --rates the built-in pre-2026-09-16 (old-kernel) plateau rates are used and the run
says so loudly: those understate the current kernel by 1.3-1.9x above n = 22.
"""
import argparse
import csv
import math

import numpy as np

# Old-kernel plateau rates at m = 2n (superseded by --rates).
MEAS_FALLBACK = {8: 40e6, 12: 4.9e6, 16: 808e3, 20: 61.7e3, 22: 16.2e3, 24: 4.2e3, 30: 105, 34: 9.3}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="img/energy")
ap.add_argument("--csv", default="", help="map_threshold.py frontier CSV, for n* and rho_eff at "
                                          "--ratio (default: the pre-2026-09-16 37.8 / 1.843)")
ap.add_argument("--rates", default="", help="comma list of bench/rate_series.py CSVs (plateau "
                                            "column) replacing the built-in measured rates")
ap.add_argument("--ratio", type=float, default=2.0, help="mode ratio m/n of the curve")
ap.add_argument("--watts", type=float, default=280.0, help="measured board power during sampling")
ap.add_argument("--eff", type=float, default=72.7e9, help="FLOPS/W of the classical bound "
                                                          "(Green500 JEDI, Soret et al.)")
ap.add_argument("--device-w", type=float, default=3.8e3, help="quantum device power (Soret et al.)")
ap.add_argument("--device-s", type=float, default=12e-9, help="seconds per sample (1 GHz / 12)")
ap.add_argument("--table", action="store_true", help="also print the paper's 8 table")
args = ap.parse_args()

# ---- measured rates ---------------------------------------------------------------------
meas, src = {}, "built-in (PRE-2026-09-16, OLD KERNEL)"
for p in [x for x in args.rates.split(",") if x]:
    for r in csv.DictReader(open(p)):
        if abs(float(r["ratio"]) - args.ratio) < 1e-9:
            rate = r.get("plateau_per_s") or r.get("samples_per_s")
            if rate:
                meas[int(r["n"])] = float(rate)
if meas:
    src = args.rates
else:
    meas = dict(MEAS_FALLBACK)
    print("!! no --rates given: using the built-in OLD-KERNEL rates.")
    print("!! they understate the current kernel by 1.3-1.9x above n = 22; pass --rates for a")
    print("!! curve that matches the frontier in the same figure.")
print(f"rates: {src}  ({len(meas)} points, m/n = {args.ratio:g})")

# ---- frontier ---------------------------------------------------------------------------
n_star, rho_eff = 37.8, 1.843
if args.csv:
    rows = [r for r in csv.DictReader(open(args.csv))
            if r.get("engine", "fock") == "fock" and r["n_star"] not in ("", "nan")
            and abs(float(r["ratio"]) - args.ratio) < 1e-9]
    if not rows:
        raise SystemExit(f"{args.csv}: no fock row at ratio {args.ratio:g}")
    n_star, rho_eff = float(rows[0]["n_star"]), float(rows[0]["rho_eff"])
print(f"frontier: n* = {n_star:g}, rho_eff = {rho_eff:g} at m/n = {args.ratio:g}"
      + (f"  ({args.csv})" if args.csv else "  (built-in default)"))

ks = sorted(meas)
n_max = ks[-1]


def rate(n):
    """samples/s: log-interpolated through the measured points, then rho_eff^(n*-n) beyond."""
    if n <= n_max:
        return float(np.exp(np.interp(n, ks, np.log([meas[k] for k in ks]))))
    return rho_eff ** (n_star - n)


def flops(m, n):
    """Clifford & Clifford 2018 flop count used as the classical lower bound by Soret et al."""
    return sum(j * 2 ** j + m * j for j in range(2, n + 1))


ns = np.arange(8, 49)
E_gpu = np.array([args.watts / rate(n) / 3600 for n in ns])
E_bound = np.array([flops(math.ceil(2.1 * n), n) / args.eff / 3600 for n in ns])
E_q = args.device_w * args.device_s / 3600

print(f"\nenergy at the frontier (n* = {n_star:g}): {args.watts / 3600:.3g} Wh per sample "
      f"-- invariant by construction, one sample per second at {args.watts:g} W")
if args.table:
    print(f"\n{'n':>4} {'measured Wh':>13} {'bound Wh':>12} {'ratio':>7} {'s/sample (GPU)':>15}")
    for n in sorted({12, 15, 18, 20, 24, 30, 34, 38, round(n_star), 42, 48}):
        if n < 8 or n > 48:
            continue
        i = int(n - 8)
        print(f"{n:>4} {E_gpu[i]:>13.2e} {E_bound[i]:>12.2e} {E_gpu[i] / E_bound[i]:>7.1f} "
              f"{1 / rate(n):>15.3g}"
              + ("   <- measured" if n <= n_max else "   (extrapolated)"))

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogy(ns, E_gpu, "o-", color="tab:blue", ms=3,
            label=f"classical, measured: one H100 at {args.watts:g} W (this work, m = {args.ratio:g}n)")
ax.semilogy(ns, E_bound, "--", color="tab:gray",
            label="classical, flop lower bound of Soret et al. (2018 form at 72.7 GFLOPS/W)")
ax.axhline(E_q, color="tab:green", ls="-.",
           label=f"quantum device, Soret et al.: {args.device_w / 1e3:g} kW x {args.device_s * 1e9:g} ns "
                 f"= {E_q:.1e} Wh per (lossy) sample")
ax.axvline(n_star, color="tab:red", ls=":",
           label=f"1-Hz simulability frontier, one H100 (m = {args.ratio:g}n): n* = {n_star:g}")
ax.axvspan(15, 18, color="tab:orange", alpha=0.15,
           label="Soret et al.: energetic (M = 15) to computational (M = 18) advantage")
ax.axvline(n_max, color="tab:blue", ls="-", lw=0.6, alpha=0.5)
ax.annotate("measured  |  extrapolated", (n_max, 1e-9), fontsize=6.5, color="tab:blue",
            rotation=90, va="bottom", ha="right")
ax.set_xlabel(f"photons n (or effective metric M), m = {args.ratio:g}n")
ax.set_ylabel("energy per sample (Wh)")
ax.set_title("Energy per sample: measured GPU vs the energetic-advantage analysis", fontsize=10)
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, loc="upper left"); ax.set_ylim(1e-10, 10)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.tight_layout(); fig.savefig(f"{args.out}.png", dpi=150); print(f"\nwrote {args.out}.png")
