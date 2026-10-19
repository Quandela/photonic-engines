"""The simulability frontier under partial distinguishability n*(eta, x): for each engine (fock,
emitter), transmission eta and pairwise overlap x (x-model, V_HOM = x^2), run find_threshold.py
--loss eta --indist x and collect the photon number at which the sampler drops to --target
samples/s on this GPU (docs/DISTINGUISHABILITY.md §5).

    python3 bench/frontier_indist.py --engines fock,emitter --eta 1,0.7 --vis 1,0.95,0.9,0.8 --ratio 2

A photon interferes when it is kept (eta) AND in the common species (x), so a shot with g
interfering photons costs a g-photon lossless sample, g ~ Bin(n, eta x): the expected cost is
E[rho^g] = (1 - eta x + eta x rho)^n and the prediction is the loss law with eta -> eta x,
n*(eta, x) = n*(1) log rho / log(1 - eta x + eta x rho). The start point of each search is seeded
from it with the anchors of results/keep/frontier.csv (same convention as frontier_loss.py).

Output: $RESULTS_DIR/<out>.csv (engine, ratio, eta, V, x, n_star, det_n_star = eta n*, ideal_n_star
= eta x n* — the "ideal-equivalent" photon number, m_star, rho_eff, prediction, bracket,
seconds), one find_threshold CSV per point, and <out>.png: n*(V) at each eta for both engines.
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
ap.add_argument("--eta", default="1,0.7")
ap.add_argument("--vis", default="1,0.95,0.9,0.8", help="HOM visibilities V; x = sqrt(V)")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--budget", type=float, default=None, help="seconds per n (find_threshold default per engine)")
ap.add_argument("--max-n", type=int, default=80)
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--out", default="frontier_indist")
ap.add_argument("--anchor-fock", default="2:38.6:1.81", help="ratio:n*:rho_eff of the lossless Fock threshold")
ap.add_argument("--anchor-emitter", default="2:30.9:1.95", help="ratio:n*:rho_eff of the lossless emitter threshold")
ap.add_argument("--extra", default="", help="extra arguments passed to find_threshold.py")
args = ap.parse_args()

out_dir = os.environ.get("RESULTS_DIR", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
etas = [float(v) for v in args.eta.split(",")]
viss = [float(v) for v in args.vis.split(",")]
engines = args.engines.split(",")
c = args.ratio


def rho(cc):
    return (2 * cc + 1) ** (2 * cc + 1) / ((4 * cc) ** cc * (cc + 1) ** (cc + 1))


def predicted_n(engine, q):
    """Same total cost as the lossless anchor: rho_a^{n_a} = (1 - q + q rho_a)^n with q = eta x,
    ratio corrected as in map_threshold.py (Fock theory), shifted for the GPU count."""
    a_ratio, a_n, a_rho = (float(v) for v in (args.anchor_fock if engine == "fock" else args.anchor_emitter).split(":"))
    n1 = (a_n * math.log(rho(a_ratio)) + math.log(args.gpus)) / math.log(rho(c))       # lossless n* at this ratio
    return n1 * math.log(a_rho) / math.log(1 - q + q * a_rho)


rows = []
here = os.path.dirname(os.path.abspath(__file__))
for engine in engines:
    for eta in etas:
        for V in viss:
            if engine == "emitter" and c < 2:
                print(f"=== {engine}: skipped (needs m >= 2n) ===", flush=True)
                break
            x = math.sqrt(V)
            n_pred = predicted_n(engine, eta * x)
            start = max(12, int(n_pred) - (6 if engine == "fock" else 5))
            csv_path = f"{out}_{engine}_eta{eta:g}_V{V:g}.csv"
            cmd = [sys.executable, "-u", os.path.join(here, "find_threshold.py"), "--engine", engine, "--ratio", str(c),
                   "--target", str(args.target), "--start", str(start), "--step", "3", "--gpus", str(args.gpus),
                   "--max-n", str(args.max_n), "--csv", csv_path]
            if eta < 1:
                cmd += ["--loss", str(eta)]
            if V < 1:
                cmd += ["--indist", str(x)]
            if args.budget:
                cmd += ["--budget", str(args.budget)]
            cmd += args.extra.split()
            print(f"=== {engine}  ratio {c:g}  eta {eta:g}  V {V:g} (x {x:.4f})  (predicted n* ~ {n_pred:.1f}, start {start}) ===",
                  flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print(proc.stdout, flush=True)
            if proc.returncode != 0:
                print(proc.stderr[-2000:], flush=True)
            m_rho = re.search(r"rho_eff = ([\d.]+)", proc.stdout)
            m_br = re.search(r"n=(\d+) at ([\d.]+)/s, n=(\d+) at ([\d.]+)/s  ->  n\* = ([\d.]+)", proc.stdout)
            m_fit = re.search(r"fitted n\* = ([\d.]+)", proc.stdout)
            n_star = float(m_br.group(5)) if m_br else (float(m_fit.group(1)) if m_fit else float("nan"))
            rows.append({"engine": engine, "ratio": c, "eta": eta, "V": V, "x": round(x, 4), "n_star": n_star,
                         "det_n_star": round(eta * n_star, 2), "ideal_n_star": round(eta * x * n_star, 2),
                         "m_star": round(c * n_star, 1), "rho_eff": float(m_rho.group(1)) if m_rho else float("nan"),
                         "rho_pred": 1 - eta * x + eta * x * rho(c), "n_pred": round(n_pred, 1),
                         "bracket": (m_br.group(0) if m_br else "not bracketed"), "seconds": round(time.time() - t0)})
            with open(f"{out}.csv", "w", newline="") as f:      # rewritten after every point (partial results survive a kill)
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                w.writeheader(); w.writerows(rows)

print("=== frontier under partial distinguishability ===")
print(f"{'engine':>8} {'eta':>5} {'V':>5} {'n*':>6} {'eta n*':>7} {'eta x n*':>9} {'rho_eff':>8} {'pred':>6}   bracket")
for r in rows:
    print(f"{r['engine']:>8} {r['eta']:>5g} {r['V']:>5g} {r['n_star']:>6.1f} {r['det_n_star']:>7.1f} {r['ideal_n_star']:>9.1f} "
          f"{r['rho_eff']:>8.3f} {r['n_pred']:>6.1f}   {r['bracket']}  [{r['seconds']} s]")
print(f"wrote {out}.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    styles = ["-", "--", ":", "-."]
    for engine, color, label in (("fock", "tab:blue", "Fock input"), ("emitter", "tab:red", "linear cluster (emitter)")):
        for ei, eta in enumerate(etas):
            pts = sorted((r["V"], r["n_star"]) for r in rows if r["engine"] == engine and r["eta"] == eta and r["n_star"] == r["n_star"])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            axs[0].plot(xs, [p[1] for p in pts], "o" + styles[ei % 4], color=color, label=f"{label}, eta = {eta:g}")
            axs[1].plot(xs, [eta * math.sqrt(p[0]) * p[1] for p in pts], "o" + styles[ei % 4], color=color, label=f"{label}, eta = {eta:g}")
    axs[0].set_xlabel("HOM visibility V"); axs[0].set_ylabel(f"n* (emitted photons) at {args.target:g} sample/s")
    axs[1].set_xlabel("HOM visibility V"); axs[1].set_ylabel("eta x n* (ideal-equivalent photons at n*)")
    for ax in axs:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7); ax.invert_xaxis()
    fig.suptitle(f"Simulability frontier under loss and partial distinguishability, one GPU, m = {c:g} n")
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150); print(f"wrote {out}.png")
except Exception as e:
    print("no plot:", e)
