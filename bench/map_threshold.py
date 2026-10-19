"""Map the simulability frontier n*(m/n) for the Fock, emitter (caterpillar) and superposition
samplers: run find_threshold.py per (engine, ratio), collect n*, rho_eff and the bracketing points,
and write a table + CSV (+ a plot if matplotlib is available). n is the SOURCE count for all three
and m = round(ratio*n) is modes per source, which is what makes the three comparable.

    python3 bench/map_threshold.py --engines fock,emitter,sbs --ratios 2,3,4,6 [--target 1] [--budget 120]
    python3 bench/map_threshold.py --engines fock --ratios 1,1.5,2,3,4,6,8

The emitter engine needs ratio >= 2 (dual-rail pairs); other ratios are skipped for it. Each
(engine, ratio) point takes 5-15 min (the emitter one more: the first n compiles ~n kernels).
Results go to $RESULTS_DIR (or ./results) as <out>.csv, <out>_<engine>_r<c>.csv, <out>.png.
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
ap.add_argument("--ratios", default="2,3,4,6", help="m/n values, non-integer allowed (m = round(ratio*n))")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--budget", type=float, default=None, help="seconds per n (find_threshold default per engine)")
ap.add_argument("--budget-sbs", type=float, default=600.0, help="seconds per n for sbs alone: its cost per shot is "
                                                                "E[cost(k)] over k ~ Binomial(n, 1/2), tail-dominated, "
                                                                "so a short ramp measures the mode and not the mean")
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--out", default="frontier")
# all three measured at m = 2n on one H100 in run 20260919_224559; they only seed the search
ap.add_argument("--anchor-fock", default="2:38.5", help="ratio:n* seeding the Fock start points (H100, 1 Hz)")
ap.add_argument("--anchor-emitter", default="2:30.9", help="ratio:n* seeding the emitter start points")
ap.add_argument("--anchor-sbs", default="2:17.2", help="n* seeding the sbs start points; flat in the ratio (see predicted_n)")
ap.add_argument("--extra", default="", help="extra arguments passed to find_threshold.py")
args = ap.parse_args()

out_dir = os.environ.get("RESULTS_DIR", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
ratios = [float(x) for x in args.ratios.split(",")]
engines = args.engines.split(",")


def rho(c):
    return (2 * c + 1) ** (2 * c + 1) / ((4 * c) ** c * (c + 1) ** (c + 1))


def predicted_n(engine, c):
    """Start point: same total cost as the anchor, rho(c)^n = rho(a)^a_n (Fock theory; the
    emitter engine's extra factor is assumed to be ratio-independent), shifted for the GPU count.

    The sbs chain rule is NOT on this model and must not borrow it: its cost per source is set by
    the 4^j of the middle of the chain, measured at ~2.3 per source against the Fock rho(2) = 1.81,
    and n is the source count while only n/2 sources emit. Extrapolating it with rho(c) put the
    start at n = 24 sources, an order of magnitude past its reach, and all four points died on the
    first evaluation (run 20260919_195403). So the sbs anchor is taken literally, flat in c -- the
    ratio dependence of a curve we have two CPU points for is not worth modelling, and
    find_threshold.py's own fit takes over after two measured n anyway.
    """
    if engine in ("sbs", "sbsfixed"):
        return float(args.anchor_sbs.split(":")[-1])
    a_ratio, a_n = (float(x) for x in (args.anchor_fock if engine == "fock" else args.anchor_emitter).split(":"))
    return (a_n * math.log(rho(a_ratio)) + math.log(args.gpus)) / math.log(rho(c))


rows = []
here = os.path.dirname(os.path.abspath(__file__))
for engine in engines:
    for c in ratios:
        if engine == "emitter" and c < 2:
            print(f"=== {engine} ratio {c:g}: skipped (needs m >= 2n) ===", flush=True)
            continue
        n_pred = predicted_n(engine, c)
        sbs = engine in ("sbs", "sbsfixed")
        start = max(6 if sbs else 12, int(n_pred) - (6 if engine == "fock" else 5))
        csv_path = f"{out}_{engine}_r{c:g}.csv"
        cmd = [sys.executable, "-u", os.path.join(here, "find_threshold.py"), "--engine", engine, "--ratio", str(c),
               "--target", str(args.target), "--start", str(start), "--step", "2" if sbs else "3",
               "--gpus", str(args.gpus), "--csv", csv_path]
        b = (args.budget_sbs if sbs else args.budget) or args.budget
        if b:
            cmd += ["--budget", str(b)]
        cmd += args.extra.split()
        print(f"=== {engine}  ratio {c:g}  (predicted n* ~ {n_pred:.1f}, start {start}, "
              f"budget {b or 'default'} s) ===", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout, flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], flush=True)
        m_rho = re.search(r"rho_eff = ([\d.]+)", proc.stdout)
        m_br = re.search(r"n=(\d+) at ([\d.]+)/s, n=(\d+) at ([\d.]+)/s  ->  n\* = ([\d.]+)", proc.stdout)
        m_fit = re.search(r"fitted n\* = ([\d.]+)", proc.stdout)
        n_star = float(m_br.group(5)) if m_br else (float(m_fit.group(1)) if m_fit else float("nan"))
        rows.append({"engine": engine, "ratio": c, "n_star": n_star, "rho_eff": float(m_rho.group(1)) if m_rho else float("nan"),
                         "rho_theory": rho(c), "bracket": (m_br.group(0) if m_br else "not bracketed"), "seconds": round(time.time() - t0)})
        with open(f"{out}.csv", "w", newline="") as f:          # rewritten after every point (partial results survive a kill)
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)

print("=== frontier ===")
print(f"{'engine':>8} {'m/n':>5} {'n*':>6} {'rho_eff':>8} {'rho(c)':>7}   bracket")
for r in rows:
    print(f"{r['engine']:>8} {r['ratio']:>5g} {r['n_star']:>6.1f} {r['rho_eff']:>8.3f} {r['rho_theory']:>7.3f}   {r['bracket']}  "
          f"[{r['seconds']} s]")
print(f"wrote {out}.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for engine, color, label in (("sbs", "tab:green", "superposition BS, alpha = pi/4 (n sources, n/2 photons on average)"),
                                 ("fock", "tab:blue", "Fock input (Clifford & Clifford 2020)"),
                                 ("emitter", "tab:red", "caterpillar / linear cluster (emitter chain rule)")):
        pts = sorted((r["ratio"], r["n_star"], r["rho_eff"]) for r in rows if r["engine"] == engine and r["n_star"] == r["n_star"])
        if not pts:
            continue
        xs_ = [p[0] for p in pts]
        ax.plot(xs_, [p[1] for p in pts], "o-", color=color, label=f"{label}, 1 GPU")
        for G, ls in ((8, "--"), (64, ":")):
            ys_ = [p[1] + math.log(G) / math.log(p[2] if 1.3 <= p[2] <= 3.5 else rho(p[0])) for p in pts]
            ax.plot(xs_, ys_, ls, color=color, alpha=0.7, label=f"{engine}, {G} GPUs (extrapolated)")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ratios)); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("m / n"); ax.set_ylabel(f"n* at {args.target:g} sample/s")
    ax.set_title("Simulability frontier, one H100"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150); print(f"wrote {out}.png")
except Exception as e:
    print("no plot:", e)
