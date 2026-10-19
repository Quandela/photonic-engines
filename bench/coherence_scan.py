"""The lossy advantage of a COHERENT source: n*(coherence) at fixed brightness.

Uniform transmission eta on a partially excited source cos(alpha)|0> + sin(alpha)|1> leaves a
mixture of vacuum and a source at a smaller alpha (docs/LOSS.md §8). Its brightness is
p = eta sin^2(alpha) but its coherence <a> = cos(alpha) sin(alpha) sqrt(eta) only falls as
sqrt(eta), so at FIXED p the pair (alpha, eta) has one free parameter left:

    sin^2(alpha) = p / eta ,   chi = sqrt( p (1 - p/eta) ) ,   eta in [p, 1]

    eta = p  -> alpha = pi/2, chi = 0            : a DETERMINISTIC source behind a lossy channel,
                                                   i.e. exactly the Fock modality at transmission p
    eta = 1  -> sin^2 alpha = p, chi maximal     : partial excitation with no loss, i.e. SBS

Every point on that line has the same photon-number statistics, Binomial(n, p), and the same
source count. What changes is only how much of the missing brightness is coherent. So this scan
asks the lossy-advantage question directly: at a detected photon rate a device can actually
deliver, does coherence make the output harder to simulate?

    python3 bench/coherence_scan.py --brightness 0.5 --eta 0.5,0.6,0.7,0.8,0.9,1 --ratio 2
    python3 bench/coherence_scan.py --brightness 0.5 --eta 0.5 --with-fock   # the calibration below

**The eta = p point is a calibration, not a data point.** There the sbs engine and the Fock engine
simulate the SAME STATE, so running both prices the two ENGINES on identical physics and measures
the implementation gap that every sbs-vs-fock comparison otherwise has to argue about. `--with-fock`
adds it.

Output: <out>.csv (eta, alpha, brightness, coherence, n_star, detected = p n_star, rho_eff,
bracket, seconds), one find_threshold CSV per point, and <out>.png.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--brightness", type=float, default=0.5, help="p = eta sin^2(alpha), detected photons per source")
ap.add_argument("--eta", default="0.5,0.6,0.7,0.8,0.9,1", help="transmissions; each must be >= brightness")
ap.add_argument("--ratio", type=float, default=2.0, help="m / sources")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--budget", type=float, default=600.0, help="seconds per n")
ap.add_argument("--start", type=int, default=15)
ap.add_argument("--max-n", type=int, default=40)
ap.add_argument("--with-fock", action="store_true", help="also price the fock engine at eta = p (same state)")
ap.add_argument("--anchor-fock", default="2:38.5", help="ratio:n* of the lossless Fock threshold; seeds the fock "
                                                        "point's search window, which is far above the sbs one")
ap.add_argument("--out", default="coherence")
ap.add_argument("--extra", default="")
args = ap.parse_args()

p = args.brightness
out_dir = os.environ.get("RESULTS_DIR", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
here = os.path.dirname(os.path.abspath(__file__))
etas = [float(x) for x in args.eta.split(",")]
if any(e < p - 1e-12 for e in etas):
    raise SystemExit(f"every eta must be >= brightness {p}: sin^2(alpha) = p/eta cannot exceed 1")


def measure(engine: str, eta: float, alpha: float | None, tag: str, start: int | None = None,
            max_n: int | None = None, step: int = 2) -> dict:
    csv_path = f"{out}_{tag}.csv"
    cmd = [sys.executable, "-u", os.path.join(here, "find_threshold.py"), "--engine", engine,
           "--ratio", str(args.ratio), "--target", str(args.target), "--start", str(start or args.start),
           "--step", str(step), "--max-n", str(max_n or args.max_n), "--budget", str(args.budget), "--csv", csv_path]
    if eta < 1.0:
        cmd += ["--loss", str(eta)]
    if alpha is not None:
        cmd += ["--alpha", f"{alpha:.12f}"]
    cmd += args.extra.split()
    print(f"=== {tag}: {engine}  eta {eta:g}  alpha {'-' if alpha is None else f'{alpha:.4f}'} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], flush=True)
    m_rho = re.search(r"rho_eff = ([\d.]+)", proc.stdout)
    m_br = re.search(r"n=(\d+) at ([\d.]+)/s, n=(\d+) at ([\d.]+)/s  ->  n\* = ([\d.]+)", proc.stdout)
    m_fit = re.search(r"fitted n\* = ([\d.]+)", proc.stdout)
    n_star = float(m_br.group(5)) if m_br else (float(m_fit.group(1)) if m_fit else float("nan"))
    return {"engine": engine, "eta": eta, "alpha": alpha if alpha is not None else float("nan"),
            "brightness": p, "coherence": math.sqrt(max(0.0, p * (1 - p / eta))),
            "n_star": n_star, "detected": round(p * n_star, 2),
            "rho_eff": float(m_rho.group(1)) if m_rho else float("nan"),
            "bracket": m_br.group(0) if m_br else ("fitted only" if m_fit else "not bracketed"),
            "seconds": round(time.time() - t0)}


rows = []
for eta in etas:
    rows.append(measure("sbs", eta, math.asin(math.sqrt(min(1.0, p / eta))), f"sbs_eta{eta:g}"))
    with open(f"{out}.csv", "w", newline="") as f:      # rewritten per point: a kill keeps the rest
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
if args.with_fock:
    # The fock engine on the same state reaches ~3x the sbs n*, so it gets its own window: the first
    # version reused --start/--max-n of the sbs points (15 / 40), never got near its ~64, and reported
    # an extrapolation to 141 from a slope of 1.06 (run 20260920_193154). Seeded from the Fock lossy law
    # n*(eta) = n*(1) log rho / log(1 - eta + eta rho) at the anchor ratio (docs/LOSS.md §7).
    a_ratio, a_n = (float(x) for x in args.anchor_fock.split(":"))
    rho_a = (2 * a_ratio + 1) ** (2 * a_ratio + 1) / ((4 * a_ratio) ** a_ratio * (a_ratio + 1) ** (a_ratio + 1))
    pred = a_n * math.log(rho_a) / math.log(1 - p + p * rho_a)
    rows.append(measure("fock", p, None, f"fock_eta{p:g}", start=max(12, int(pred) - 6),
                        max_n=int(pred) + 20, step=3))
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

print("=== coherence scan ===")
print(f"brightness p = {p:g} detected photons per source, m/n = {args.ratio:g}")
print(f"{'engine':>7} {'eta':>5} {'alpha':>7} {'chi':>7} {'n*':>6} {'detected':>9} {'rho_eff':>8}   bracket")
for r in rows:
    print(f"{r['engine']:>7} {r['eta']:>5g} {r['alpha']:>7.4f} {r['coherence']:>7.4f} {r['n_star']:>6.1f} "
          f"{r['detected']:>9.1f} {r['rho_eff']:>8.3f}   {r['bracket']}  [{r['seconds']} s]")
print(f"wrote {out}.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sb = [r for r in rows if r["engine"] == "sbs" and r["n_star"] == r["n_star"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["coherence"] for r in sb], [r["n_star"] for r in sb], "o-", color="tab:green",
            label="superposition BS engine, n* (sources)")
    ax.plot([r["coherence"] for r in sb], [r["detected"] for r in sb], "s--", color="tab:green",
            alpha=0.6, label=f"the same, detected photons (p n*, p = {p:g})")
    for r in rows:
        if r["engine"] == "fock" and r["n_star"] == r["n_star"]:
            ax.axhline(r["n_star"], color="tab:blue", ls=":",
                       label=f"fock engine on the SAME state at eta = {p:g}: n* = {r['n_star']:.1f}")
    ax.set_xlabel(f"source coherence |<a>| at fixed brightness p = {p:g}   (0 = lossy Fock, max = lossless SBS)")
    ax.set_ylabel(f"n* at {args.target:g} sample / s")
    ax.set_title("Does coherence make a dim source harder to simulate?", fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150); print(f"wrote {out}.png")
except Exception as e:
    print("no plot:", e)
