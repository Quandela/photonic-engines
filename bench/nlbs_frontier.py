"""The simulability frontier of non-linear boson sampling n*(phi, L, m/n, eta, x): for each setting, run
find_threshold.py --engine nlbs and collect the photon number at which the exact rejection sampler drops to
--target samples/s on this GPU (docs/NONLINEAR.md §5b, §7).

    python3 bench/nlbs_frontier.py --scan ratio --ratios 1,1.5,2,3,4,6,8,12,16 --phi pi/2
    python3 bench/nlbs_frontier.py --scan phi   --phis 0,pi/16,pi/8,pi/4,3pi/8,pi/2,5pi/8,3pi/4,pi --ratio 2
    python3 bench/nlbs_frontier.py --scan sites --sites 1,2,4,8 --layout same-layer,sequential --phi pi/2
    python3 bench/nlbs_frontier.py --scan imperfections --eta 1,0.9,0.7 --vis 1,0.9 --phi pi/2

Cost model behind the seeded start point (docs/NONLINEAR.md §5b): a sample costs |c|_1^2 proposals, each one
Fock sample plus B branch permanents, so with the Fock anchor n*(1) and rho(c),
    n*(phi) ~ n*_fock - log(|c|_1^2 (1 + kappa B)) / log rho ,
kappa the cost of one permanent in Fock samples (--kappa, 0.5 by default until measured). |c|_1^2 and B come
from `nonlinear.branches` at the anchor n, so phi = 0 and phi = pi predict the Fock frontier exactly.

Output: $RESULTS_DIR/<out>.csv (phi, sites, layout, ratio, eta, x, eta_post, n_star, m_star, rho_eff, branches,
rejection factor, predicted n*, bracket, seconds), one find_threshold CSV per point, and <out>.png.
The hardware reading goes to docs/NONLINEAR.md §5b once measured.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from photonic_engines.nonlinear import branches

ap = argparse.ArgumentParser()
ap.add_argument("--scan", default="phi", choices=["phi", "ratio", "sites", "imperfections"])
ap.add_argument("--phi", default="pi/2", help="Kerr strength as an expression in pi (exact: 'pi/4', not 0.785)")
ap.add_argument("--phis", default="0,pi/16,pi/8,pi/4,3*pi/8,pi/2,5*pi/8,3*pi/4,pi")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--ratios", default="1,1.5,2,3,4,6,8,12,16")
ap.add_argument("--sites", default="1,2,4,8")
ap.add_argument("--layout", default="same-layer")
ap.add_argument("--eta", default="1")
ap.add_argument("--vis", default="1", help="HOM visibilities V; x = sqrt(V)")
ap.add_argument("--eta-post", type=float, default=None)
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--budget", type=float, default=None)
ap.add_argument("--max-n", type=int, default=60)
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--kappa", type=float, default=0.5, help="cost of one branch permanent in Fock samples (prediction only)")
ap.add_argument("--anchor-fock", default="2:38.5:1.81", help="ratio:n*:rho_eff of the Fock threshold at that ratio")
ap.add_argument("--out", default="nlbs_frontier")
ap.add_argument("--extra", default="", help="extra arguments passed to find_threshold.py")
args = ap.parse_args()

out_dir = os.environ.get("RESULTS_DIR", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
val = lambda e: float(eval(e, {"__builtins__": {}}, {"pi": math.pi}))
a_ratio, a_n, a_rho = (float(v) for v in args.anchor_fock.split(":"))


def rho(c):
    return (2 * c + 1) ** (2 * c + 1) / ((4 * c) ** c * (c + 1) ** (c + 1))


def predicted(phi, sites, c, eta, x):
    """n* of the cost model: the Fock anchor moved by the ratio (Fock theory), by the loss / species law
    (eta x as in frontier_indist.py) and by the toll of the non-linearity |c|_1^2 (1 + kappa B)."""
    n_fock = a_n * math.log(a_rho) / math.log(rho(c)) if c != a_ratio else a_n
    q = eta * x
    n_fock *= math.log(rho(c)) / math.log(1 + q * (rho(c) - 1)) if q < 1 else 1.0
    th, cc = branches(phi, max(1, round(n_fock)))
    B, one = len(th), float(abs(cc).sum() ** 2)
    toll = one ** sites * (1 + args.kappa * B ** sites)
    return max(1.0, n_fock - math.log(toll) / math.log(rho(c))), B ** sites, one ** sites


def run(phi_expr, sites, layout, c, eta, x, tag):
    phi = val(phi_expr)
    pred, B, one = predicted(phi, sites, c, eta, x)
    csv_path = f"{out}_{tag}.csv"
    cmd = [sys.executable, "bench/find_threshold.py", "--engine", "nlbs", "--ratio", str(c), "--phi", phi_expr,
           "--sites", str(sites), "--layout", layout, "--target", str(args.target), "--max-n", str(args.max_n),
           "--gpus", str(args.gpus), "--start", str(max(2, round(pred) - 1)), "--csv", csv_path]
    if args.budget:
        cmd += ["--budget", str(args.budget)]
    if eta < 1:
        cmd += ["--loss", str(eta)]
    if x < 1:
        cmd += ["--indist", str(x)]
    if args.eta_post:
        cmd += ["--eta-post", str(args.eta_post)]
    cmd += args.extra.split() if args.extra else []
    print(f"\n=== phi={phi_expr} sites={sites} {layout} m/n={c} eta={eta} x={x}: B={B}, |c|_1^2={one:.2f}, "
          f"predicted n*={pred:.1f} ===", flush=True)
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(p.stdout + p.stderr, flush=True)
    sec = time.perf_counter() - t
    n_star = m_star = rho_eff = None
    for line in p.stdout.splitlines():
        if (mm := re.search(r"n\* = ([0-9.]+)", line)):
            n_star = float(mm.group(1))
        if (mm := re.search(r"rho_eff = ([0-9.]+)", line)):
            rho_eff = float(mm.group(1))
    if n_star is not None:
        m_star = round(c * n_star)
    return {"phi": phi_expr, "phi_value": phi, "sites": sites, "layout": layout, "ratio": c, "eta": eta, "x": x,
            "eta_post": args.eta_post or 1.0, "branches": B, "rejection": one, "predicted": round(pred, 1),
            "n_star": n_star, "m_star": m_star, "rho_eff": rho_eff, "seconds": round(sec, 1)}


rows = []
etas = [float(v) for v in args.eta.split(",")]
xs = [math.sqrt(float(v)) for v in args.vis.split(",")]
if args.scan == "phi":
    for i, ph in enumerate(args.phis.split(",")):
        rows.append(run(ph, 1, args.layout, args.ratio, etas[0], xs[0], f"phi{i}"))
elif args.scan == "ratio":
    for i, c in enumerate(float(v) for v in args.ratios.split(",")):
        rows.append(run(args.phi, 1, args.layout, c, etas[0], xs[0], f"ratio{i}"))
elif args.scan == "sites":
    for lay in args.layout.split(","):
        for L in (int(v) for v in args.sites.split(",")):
            rows.append(run(args.phi, L, lay, args.ratio, etas[0], xs[0], f"L{L}_{lay}"))
else:
    for eta in etas:
        for x in xs:
            rows.append(run(args.phi, 1, args.layout, args.ratio, eta, x, f"eta{eta}_x{x:.3f}"))
    for r in rows:
        print(f"  eta={r['eta']} x={r['x']:.3f}: n* = {r['n_star']}", flush=True)

with open(f"{out}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out}.csv", flush=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs_p = {"phi": [r["phi_value"] for r in rows], "ratio": [r["ratio"] for r in rows],
            "sites": [r["sites"] for r in rows], "imperfections": [r["eta"] * r["x"] for r in rows]}[args.scan]
    lab = {"phi": "Kerr strength phi", "ratio": "mode ratio m/n", "sites": "non-linear sites L",
           "imperfections": "eta x"}[args.scan]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if args.scan == "sites":
        for lay in sorted({r["layout"] for r in rows}):
            sel = [r for r in rows if r["layout"] == lay]
            ax.plot([r["sites"] for r in sel], [r["n_star"] for r in sel], "o-", label=lay)
            ax.plot([r["sites"] for r in sel], [r["predicted"] for r in sel], "--", alpha=.5, label=f"{lay}, model")
    else:
        ax.plot(xs_p, [r["n_star"] for r in rows], "o-", label="measured")
        ax.plot(xs_p, [r["predicted"] for r in rows], "--", alpha=.5, label="cost model")
    if args.scan == "phi":
        ax.axhline(a_n, ls=":", color="grey", label=f"Fock n* = {a_n:g}")
    ax.set_xlabel(lab); ax.set_ylabel("sources n*"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"Non-linear boson sampling: n* at {args.target:g} sample/s")
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=200)
    print(f"wrote {out}.png", flush=True)
except ImportError:
    pass
