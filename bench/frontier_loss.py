"""The lossy simulability frontier n*(eta): for each engine (fock, emitter, sbs) and transmission
eta, run find_threshold.py --loss eta and collect the photon number at which the *lossy* sampler
drops to --target samples/s on this GPU.

    python3 bench/frontier_loss.py --engines fock,emitter --eta 1,0.9,0.8,0.7,0.5 --ratio 2

For sbs the transmission multiplies a source that is ALREADY partial: brightness becomes
eta sin^2(alpha), so at the default alpha = pi/4 an eta scan is a brightness scan from 0.5 down,
and the source stays coherent throughout (docs/LOSS.md §8). The scan that holds brightness fixed
and trades loss against coherence instead is bench/coherence_scan.py, and that is the one that
answers the lossy-advantage question; this one is the device curve, "how does my sbs frontier
degrade as the chip gets lossier".

Loss is applied on the input side (docs/LOSS.md): a stream with k surviving photons costs what a
k-photon lossless sample costs, so the expected cost per sample is E[rho^k], k ~ Bin(n, eta),
i.e. rho_eff(eta) ~ 1 - eta + eta rho and n*(eta) ~ n*(1) log(rho) / log(1 - eta + eta rho)
(plus, for the emitter engine, the per-k-group host work that bench_loss.py measured). The
start point of each search is seeded from that prediction with the anchors n*(1) of
results/keep/frontier.csv (fock 2:37.8, emitter 2:29.0, one H100, 1 Hz).

Output: $RESULTS_DIR/<out>.csv (engine, ratio, eta, n_star, eta_n_star = mean detected photons
(for sbs that is eta sin^2(alpha) n* = eta n*/2 at the default alpha: its sources already emit half
the time; runs 20260920_193154 and 20260921_080421 printed eta n* and so doubled it)
at the threshold, m_star, rho_eff, bracket, seconds), one find_threshold CSV per point, and
<out>.png: n*(eta) and eta n*(eta) for both engines. The hardware reading goes to docs/LOSS.md §7 once measured.
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
ap.add_argument("--engines", default="fock,emitter")
ap.add_argument("--eta", default="1,0.9,0.8,0.7,0.5")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--budget", type=float, default=None, help="seconds per n (find_threshold default per engine)")
ap.add_argument("--max-n", type=int, default=80)
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--out", default="frontier_loss")
ap.add_argument("--anchor-fock", default="2:37.8:2.0", help="ratio:n*:rho_eff of the lossless Fock threshold")
ap.add_argument("--anchor-emitter", default="2:29.0:2.1", help="ratio:n*:rho_eff of the lossless emitter threshold")
ap.add_argument("--anchor-sbs", default="2:20.0:2.05", help="ratio:n*:rho_eff of the lossless sbs threshold")
ap.add_argument("--extra", default="", help="extra arguments passed to find_threshold.py")
args = ap.parse_args()

out_dir = os.environ.get("RESULTS_DIR", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
etas = [float(x) for x in args.eta.split(",")]
engines = args.engines.split(",")
c = args.ratio


def rho(cc):
    return (2 * cc + 1) ** (2 * cc + 1) / ((4 * cc) ** cc * (cc + 1) ** (cc + 1))


def predicted_n(engine, eta):
    """Same total cost as the lossless anchor: rho_a^{n_a} = (1 - eta + eta rho_a)^n, ratio corrected
    as in map_threshold.py (Fock theory), shifted for the GPU count.

    sbs is off that ratio correction, as in map_threshold.py: the exponent of both marginal
    evaluations is free of m. Its loss correction uses the BRIGHTNESS eta sin^2(alpha) = eta/2 at
    the default alpha, not eta -- a transmission eta on a source that already emits half the time
    takes a survivor with probability eta/2, so the per-source factor is 1 - p + p rho_a with
    p = eta/2, referred to the lossless anchor which already sits at p = 1/2."""
    anchor = (args.anchor_fock if engine == "fock" else
              args.anchor_sbs if engine in ("sbs", "sbsfixed") else args.anchor_emitter)
    a_ratio, a_n, a_rho = (float(x) for x in anchor.split(":"))
    if engine in ("sbs", "sbsfixed"):
        base = (a_rho - 0.5) / 0.5                                  # rho_a = 1 - 1/2 + 1/2 base
        return a_n * math.log(a_rho) / math.log(1 - eta / 2 + eta / 2 * base)
    n1 = (a_n * math.log(rho(a_ratio)) + math.log(args.gpus)) / math.log(rho(c))       # lossless n* at this ratio
    return n1 * math.log(a_rho) / math.log(1 - eta + eta * a_rho)


rows = []
here = os.path.dirname(os.path.abspath(__file__))
for engine in engines:
    for eta in etas:
        if engine == "emitter" and c < 2:
            print(f"=== {engine}: skipped (needs m >= 2n) ===", flush=True)
            break
        n_pred = predicted_n(engine, eta)
        start = max(6 if engine in ("sbs", "sbsfixed") else 12, int(n_pred) - (6 if engine == "fock" else 5))
        csv_path = f"{out}_{engine}_eta{eta:g}.csv"
        cmd = [sys.executable, "-u", os.path.join(here, "find_threshold.py"), "--engine", engine, "--ratio", str(c),
               "--target", str(args.target), "--start", str(start),
               "--step", "2" if engine in ("sbs", "sbsfixed") else "3", "--gpus", str(args.gpus),
               "--max-n", str(args.max_n), "--csv", csv_path]
        if eta < 1:
            cmd += ["--loss", str(eta)]
        if args.budget:
            cmd += ["--budget", str(args.budget)]
        cmd += args.extra.split()
        print(f"=== {engine}  ratio {c:g}  eta {eta:g}  (predicted n* ~ {n_pred:.1f}, start {start}) ===", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout, flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], flush=True)
        m_rho = re.search(r"rho_eff = ([\d.]+)", proc.stdout)
        m_br = re.search(r"n=(\d+) at ([\d.]+)/s, n=(\d+) at ([\d.]+)/s  ->  n\* = ([\d.]+)", proc.stdout)
        m_fit = re.search(r"fitted n\* = ([\d.]+)", proc.stdout)
        n_star = float(m_br.group(5)) if m_br else (float(m_fit.group(1)) if m_fit else float("nan"))
        brightness = 0.5 if engine in ("sbs", "sbsfixed") else 1.0          # sin^2(alpha) at the default alpha
        rows.append({"engine": engine, "ratio": c, "eta": eta, "n_star": n_star, "eta_n_star": round(eta * n_star * brightness, 2),
                         "m_star": round(c * n_star, 1), "rho_eff": float(m_rho.group(1)) if m_rho else float("nan"),
                         "rho_pred": 1 - eta + eta * rho(c), "n_pred": round(n_pred, 1),
                         "bracket": (m_br.group(0) if m_br else "not bracketed"), "seconds": round(time.time() - t0)})
        with open(f"{out}.csv", "w", newline="") as f:          # rewritten after every point (partial results survive a kill)
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)

print("=== lossy frontier ===")
print(f"{'engine':>8} {'eta':>5} {'n*':>6} {'eta n*':>7} {'m*':>6} {'rho_eff':>8} {'pred':>6}   bracket")
for r in rows:
    print(f"{r['engine']:>8} {r['eta']:>5g} {r['n_star']:>6.1f} {r['eta_n_star']:>7.1f} {r['m_star']:>6.1f} {r['rho_eff']:>8.3f} "
          f"{r['n_pred']:>6.1f}   {r['bracket']}  [{r['seconds']} s]")
print(f"wrote {out}.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for engine, color, label in (("fock", "tab:blue", "Fock input"), ("emitter", "tab:red", "linear cluster (emitter)")):
        pts = sorted((r["eta"], r["n_star"]) for r in rows if r["engine"] == engine and r["n_star"] == r["n_star"])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        axs[0].plot(xs, [p[1] for p in pts], "o-", color=color, label=label)
        axs[1].plot(xs, [p[0] * p[1] for p in pts], "o-", color=color, label=label)
    axs[0].set_xlabel("transmission eta"); axs[0].set_ylabel(f"n* (emitted photons) at {args.target:g} sample/s")
    axs[1].set_xlabel("transmission eta"); axs[1].set_ylabel("eta n* (mean detected photons at n*)")
    for ax in axs:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8); ax.invert_xaxis()
    fig.suptitle(f"Lossy simulability frontier, one GPU, m = {c:g} n")
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150); print(f"wrote {out}.png")
except Exception as e:
    print("no plot:", e)
