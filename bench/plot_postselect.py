"""
Where a lossy photonic device stands against the classical sampler (docs/HARDWARE_TARGET.md).
v2 (2026-10-01): devices described by their parameters only, the measured cost table t(k, m) of
bench/cost_table.py where it exists, the run-of-S statistic for the whole stream, two panels.

Two comparisons, one rate model:

  (A) POST-SELECTED exactly k.  A device sends n photons per train and keeps the trains with exactly k
      detected. Under uniform loss that is the distribution of k photons at uniformly random input
      modes, so the classical side is one k-photon sample at m modes, t(k, m). The device delivers
          R_hw(n, k, eta) = (S f / n) C(n, k) eta^k (1 - eta)^(n-k)   such trains per second,
      and the smallest eta at which that ties G H100 producing k-photon samples, G / t(k, m), is
      plotted against k. k where even the binomial peak (eta = k / n) is below the target are not
      reachable: one H100 outruns the source there at any transmission.

  (B) THE WHOLE LOSSY STREAM.  The device emits f / ceil(n / S) trains per second, each carrying k ~ Bin(n, eta)
      detected photons. A competent classical simulator does NOT pay t(n, m) per train: uniform loss
      commutes with the interferometer, so it draws the surviving input subset first and runs the
      k-photon sampler. Its cost per train is the binomial average of t(k, m), dominated by the rare
      high-k trains, and the statistic is the run-of-S one (D2, docs/GAUSSIAN.md §2.1): a run of
      S_run = (f / ceil(n / S)) x --run-seconds trains meets the trains beyond the 1 - 1/S_run quantile of k
      less than once, so they are left out of the mean. The device ties G H100 when
          G / <t(k, m)>_{k <= k_S}  =  f / ceil(n / S).
      The right panel shows the H100-equivalents needed to reproduce the stream in real time,
      (S f / n) <t>, against eta, with the 1 / 8 / 64 lines; the table prints the crossings.

Rate model t(k, m): a measured point of the cost table at exactly (k, m) if there is one; beyond a
column's last measured k, that point continued with the frontier's rho_eff at each mode ratio; where
no column exists, the frontier extrapolation rho_eff^(n*(m/k) - k) capped at --cap (launch-bound).
Every number is tagged with which of the three it rests on.

    python bench/plot_postselect.py [--frontier results/keep/frontier.csv] [--cost results/keep/cost_table_*.csv]
                                    [--device "label:S:f:n:m" ...] [--run-seconds 3600] [--out img/postselect]

Devices: described by their characteristics only, no device or vendor name in any output — A (1 source,
80 MHz, 12 photons / 24 modes), B (2 sources, 80 MHz, 24 / 48), C (3 entangled sources, 1 GHz, 48 / 96),
D (1 source, 80 MHz, 6 / 12; a published system). For C the Fock cost is the reading conservative for the
device: entangled emission is priced by the emitter engine, ~8 photons dearer at equal cost (docs/HARDWARE_TARGET.md).
"""
import argparse
import csv
import math
import os
from math import comb

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--frontier", default="results/keep/frontier.csv", help="map_threshold.py output (frontier v2)")
ap.add_argument("--engine", default="fock")
ap.add_argument("--cost", default="", help="comma list of bench/cost_table.py CSVs (measured t(k, m))")
ap.add_argument("--device", action="append", default=[], help='"label:S:f:n:m" (sources, clock Hz, photons per train, modes); repeatable')
ap.add_argument("--no-default-devices", action="store_true")
ap.add_argument("--run-seconds", type=float, default=3600.0, help="run length of the device for the run-of-S statistic (B)")
ap.add_argument("--gpus", default="1,8,64")
ap.add_argument("--cap", type=float, default=5e7, help="max GPU rate (launch-bound), samples/s")
ap.add_argument("--out", default="img/postselect")
ap.add_argument("--table", default="", help="write the per-device numbers to this CSV")
args = ap.parse_args()
GPUS = [int(x) for x in args.gpus.split(",")]

# ------------------------------------------------------------------ the frontier (v2) and its extrapolation
_rows = [r for r in csv.DictReader(open(args.frontier))
         if r.get("mode", "throughput") == "throughput" and r.get("engine", args.engine) == args.engine
         and r["n_star"] not in ("", "nan")]
if not _rows:
    raise SystemExit(f"no usable rows in {args.frontier}")
cs = np.array([float(r["ratio"]) for r in _rows]); ns = np.array([float(r["n_star"]) for r in _rows])
rh = np.array([float(r["rho_eff"]) for r in _rows])
o = np.argsort(cs); cs, ns, rh = cs[o], ns[o], rh[o]
print(f"frontier: {args.frontier}  ({len(_rows)} ratios, engine={args.engine}, n*(2) = {float(np.interp(2.0, cs, ns)):.1f})")


def rho_th(c):
    return math.exp((2 * c + 1) * math.log(2 * c + 1) - c * math.log(4 * c) - (c + 1) * math.log(c + 1))


def nstar_rho(c):
    """(n*, rho_eff) at mode ratio c: interpolated in log c inside the measured ratios; beyond them
    rho -> the theory value (-> 2) at the same total cost."""
    c = max(c, cs[0])
    if c <= cs[-1]:
        return float(np.interp(math.log(c), np.log(cs), ns)), float(np.interp(math.log(c), np.log(cs), rh))
    r = max(rho_th(c), rh[-1])
    return ns[-1] * math.log(rh[-1]) / math.log(r), r


# ------------------------------------------------------------------ the measured cost table
COST: dict = {}                                  # m -> {k: seconds per sample}
for _p in [x for x in args.cost.split(",") if x]:
    for _r in csv.DictReader(open(_p)):
        if _r["samples_per_s"]:
            COST.setdefault(int(_r["m"]), {})[int(_r["k"])] = 1.0 / float(_r["samples_per_s"])
for _m, _d in COST.items():
    # a k-photon sample is never cheaper than a (k-1)-photon one: the measured rates are made monotone by
    # taking, at each k, the best rate measured at any k' >= k. The dips this removes are ramp artefacts
    # (Triton recompiles of the draw kernel at the lane-count steps of the ramp, landing inside the timed
    # runs at small k; run 20261001_005657, e.g. m = 12 k = 6, m = 96 k = 8, 9, 16, 17 — full list in
    # HARDWARE_TARGET.md), not kernel behaviour. The envelope only half-corrects a dip (it substitutes the
    # slower rate of a larger k): those 10-01 dips raised C at 20 % from 29 to 38 H100. Table of record since
    # 10-11: run 20261011_205906 (one dip left, m = 96 k = 9, <= 3 % on any number).
    ks = sorted(_d, reverse=True); best = math.inf
    for k in ks:
        best = min(best, _d[k]); _d[k] = best
if COST:
    print("cost table: " + ", ".join(f"m = {m}: k = {min(d)}..{max(d)}" for m, d in sorted(COST.items())) + "  (monotone envelope)")


def cost(k, m):
    """(seconds per k-photon sample at m modes, source): 'measured' | 'continued' | 'extrapolated'."""
    if k <= 0:
        return 1e-10, "trivial"
    col = COST.get(m)
    if col and k in col:
        return col[k], "measured"
    if col and k > max(col):
        k0 = max(col); t = col[k0]
        for j in range(k0 + 1, k + 1):                              # one photon at a time, rho_eff of that ratio
            t *= nstar_rho(m / j)[1]
        return t, "continued"
    n_, r = nstar_rho(m / k)
    return max(1.0 / args.cap, r ** (k - n_)), "extrapolated"


# ------------------------------------------------------------------ devices
def parse_device(s):
    label, S, f, n, m = s.rsplit(":", 4)
    return label, int(S), float(f), int(n), int(m)


DEFAULT = [
    "A (1 source, 80 MHz, 12 photons / 24 modes):1:80e6:12:24",
    "B (2 sources, 80 MHz, 24 / 48):2:80e6:24:48",
    "C (3 entangled sources, 1 GHz, 48 / 96):3:1e9:48:96",
    "D (1 source, 80 MHz, 6 / 12; published):1:80e6:6:12",
]


def train_rate(S, f, n):
    """Trains per second: one photon per source per tick, S sources in parallel -> ceil(n / S) ticks per train."""
    return f / math.ceil(n / S)
devices = [parse_device(s) for s in (([] if args.no_default_devices else DEFAULT) + args.device)]


def binom_law(n, eta):
    return np.array([comb(n, k) * eta ** k * (1 - eta) ** (n - k) for k in range(n + 1)])


def run_cutoff(law, samples):
    """k_S: the smallest k with P(k > k_S) <= 1 / samples (gaussian/cost.run_cutoff, same definition)."""
    tail = 1.0 - np.cumsum(law)
    idx = np.nonzero(tail <= 1.0 / samples)[0]
    return int(idx[0]) if idx.size else len(law) - 1


def stream_cost(n, m, eta, S_run):
    """Seconds of one H100 per train of the lossy stream, run-of-S statistic: sum_{k <= k_S} P(k) t(k, m)."""
    law = binom_law(n, eta)
    kS = run_cutoff(law, S_run)
    return sum(law[k] * cost(k, m)[0] for k in range(kS + 1)), kS


def eta_post(n, k, S, f, target):
    """Smallest eta at which the exactly-k train rate reaches `target` samples/s, or None."""
    pref = train_rate(S, f, n) * comb(n, k)
    grid = np.linspace(1e-4, 1, 20000)
    rate = pref * grid ** k * (1 - grid) ** (n - k)
    ok = np.where(rate >= target)[0]
    return float(grid[ok[0]]) if len(ok) else None


def eta_stream(n, m, S, f, G, S_run):
    """Smallest eta at which G H100 no longer keep up with the stream: G / <t> <= S f / n."""
    hw = train_rate(S, f, n)
    lo, hi = 0.002, 0.999
    if G / stream_cost(n, m, hi, S_run)[0] > hw:
        return None
    if G / stream_cost(n, m, lo, S_run)[0] <= hw:
        return lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if G / stream_cost(n, m, mid, S_run)[0] <= hw:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------ numbers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = ["tab:orange", "tab:green", "tab:blue", "tab:red", "tab:purple", "tab:brown"]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.4))
table_rows = []
print(f"\nrun-of-S statistic: run = {args.run_seconds:g} s of the device's stream")
for di, (label, S, f, n, m) in enumerate(devices):
    col = colors[di % len(colors)]
    hw = train_rate(S, f, n)
    S_run = hw * args.run_seconds
    print(f"\n=== {label}: {hw:.2e} trains/s, run of {S_run:.1e} trains")
    # (A) post-selected
    print(f"  {'k':>3} {'m/k':>5} {'t(k,m)':>10} {'src':>12} " + " ".join(f"{'eta ' + str(G) + ' H100':>12}" for G in GPUS))
    best = dict.fromkeys(GPUS, (None, None))
    curves = {G: ([], []) for G in GPUS}
    k_front = None
    for k in range(1, n + 1):
        t, src = cost(k, m)
        if k_front is None and t >= 1.0:
            k_front = k
        es = {G: eta_post(n, k, S, f, G / t) for G in GPUS}
        for G in GPUS:
            if es[G] is not None:
                curves[G][0].append(k); curves[G][1].append(100 * es[G])
                if best[G][1] is None or es[G] < best[G][1]:
                    best[G] = (k, es[G])
        if k in (max(1, n // 4), n // 3, n // 2, (2 * n) // 3, n) or (k_front == k):
            print(f"  {k:>3} {m / k:>5.1f} {t:>10.3e} {src:>12} " + " ".join(
                f"{100 * es[G]:>11.1f}%" if es[G] is not None else f"{'unreachable':>12}" for G in GPUS))
    for G, ls in zip(GPUS, ("-", "--", ":")):
        ks, es = curves[G]
        if ks:
            axA.plot(ks, es, ls, color=col, marker="o" if GPUS[0] == G else None, ms=3,
                     label=f"{label}, {G} H100" if GPUS[0] == G or di == 0 else None)
    kmin_reach = curves[GPUS[0]][0][0] if curves[GPUS[0]][0] else None
    # (B) whole stream
    etas = np.linspace(0.02, 0.98, 97)
    heq = []
    for e in etas:
        c_, kS = stream_cost(n, m, e, S_run)
        heq.append(hw * c_)
    axB.plot(100 * etas, heq, color=col, label=label)
    ties = {}
    for G in GPUS:
        e = eta_stream(n, m, S, f, G, S_run)
        if e is None:
            ties[G] = None
            continue
        c_, kS = stream_cost(n, m, e, S_run)
        ties[G] = (e, n * e, kS)
    print("  whole stream, eta at which G H100 stop keeping up (<k>, k_S = largest k a run of that length meets):")
    for G in GPUS:
        v = ties[G]
        txt = f"{100 * v[0]:5.1f} %   <k> = {v[1]:4.1f}   k_S = {v[2]}" if v else "never (one H100 outruns the source at every eta)"
        print(f"    {G:>3} H100: " + txt)
    opt = ", ".join(f"{G} H100: k = {best[G][0]} at {100 * best[G][1]:.1f} %" if best[G][0] else f"{G} H100: unreachable" for G in GPUS)
    reach = f";  first k unreachable below: {kmin_reach}" if kmin_reach and kmin_reach > 1 else ""
    front = (f";  1 Hz frontier at k = {k_front} (t = {cost(k_front, m)[0]:.2f} s, {cost(k_front, m)[1]})" if k_front
             else "; every k inside the 1 Hz frontier")
    print("  post-selected optimum (smallest eta over k): " + opt + reach + front)
    table_rows.append({"device": label, "S": S, "f_Hz": f, "n": n, "m": m, "trains_per_s": hw, "run_trains": S_run,
                       **{f"eta_stream_{G}": (ties[G][0] if ties[G] else "") for G in GPUS},
                       **{f"mean_k_stream_{G}": (ties[G][1] if ties[G] else "") for G in GPUS},
                       **{f"kS_stream_{G}": (ties[G][2] if ties[G] else "") for G in GPUS},
                       **{f"post_k_opt_{G}": (best[G][0] or "") for G in GPUS},
                       **{f"post_eta_opt_{G}": (best[G][1] if best[G][0] else "") for G in GPUS},
                       "k_at_1Hz": k_front or "", "kmin_reachable_1": kmin_reach or ""})

axA.set_xlabel("k photons detected (post-selected) out of n sent per train")
axA.set_ylabel("min transmission to tie G H100 at (k, m)  [%]")
axA.set_title("(A) post-selected on exactly k detected photons", fontsize=10)
axA.set_ylim(0, 100); axA.grid(alpha=0.3); axA.legend(fontsize=6.5, loc="upper right")
axB.set_yscale("log")
for G in GPUS:
    axB.axhline(G, color="k", lw=0.8, ls=":" if G > 1 else "-")
    axB.text(3, G * 1.15, f"{G} H100", fontsize=7)
axB.set_xlabel("end-to-end transmission per photon  [%]")
axB.set_ylabel("H100 needed to reproduce the whole stream in real time")
axB.set_title(f"(B) the whole lossy stream, run of {args.run_seconds:g} s", fontsize=10)
axB.grid(alpha=0.3, which="both"); axB.legend(fontsize=6.5, loc="upper left")
fig.suptitle("Lossy photonic devices against the exact classical sampler (frontier v2, one H100 PCIe)", fontsize=10)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(f"{args.out}.png", dpi=150); print(f"\nwrote {args.out}.png")
if args.table:
    with open(args.table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys())); w.writeheader(); w.writerows(table_rows)
    print("wrote", args.table)
