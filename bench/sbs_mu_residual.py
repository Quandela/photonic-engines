"""Residual of the superposed (alpha = pi/4) sampler against the best known cofactor kernel, in sources
(the frontier paper's parity audit: <= 0.07 source, headroom
+0.14-0.18 at n = 46, +0.35-0.43 at n = 60). CPU model, measured kernel costs only, seconds.

    python3 bench/sbs_mu_residual.py \
        --mu-csv results/keep/cuda_rect_mu_extend_96db3fd_20261005_h100.csv \
        --r22-csv results/keep/cuda_rect_mu_extend_r22_833d953_20261006_h100.csv

Model: lossless alpha = pi/4, n sources, k ~ Bin(n, 1/2) photons, s0 = n - k constant along a shot, shot cost
proportional to 2^k times the per-unit cost of the form that runs stratum s0. Cost weight of stratum s0:
C(n, k) 2^k u(s0). Units (ps per Glynn unit, H100):
  s0 = 0         Fock kernel, 4.5 ps;
  s0 = 1..6      mu at its D = 7 cost (cost mass < 0.1 %);
  s0 = 7..16     best mu over TPL, q = 24, 64 streams (run 20261005_215026); D = 16 = mean of q = 24 and q = 28;
  s0 = 17, 18    best mu, q = 24, unpooled (run 20261006_120404) -- the build of record dispatches mu up to 18;
  s0 >= 19       lambda, --lambda-ps (540: q = 24 measured 468-587).
Scenarios against that build: (a) mu at D = 19, cost interpolated between the measured D = 18 and D = 20;
(b) mu at D = 19 at D = 18's cost (optimistic); (c) headroom: a hypothetical kernel running s0 = 17..30 at D = 18's
cost (where cheaper than today). Delta n* = ln(speed-up) / ln rho_local, rho_local = 1.9 / 2.2 per source (the
measured local slope of the rate at n*, run 20261006_132352).
"""
import argparse
import csv
import math

ap = argparse.ArgumentParser()
ap.add_argument("--mu-csv", required=True, help="cuda_rect_mu_extend.py output, D <= 16 (run 20261005_215026)")
ap.add_argument("--r22-csv", required=True, help="cuda_rect_mu_extend.py output, D = 17..24 (run 20261006_120404)")
ap.add_argument("--n", default="40,46,49,52,60")
ap.add_argument("--fock-ps", type=float, default=4.5)
ap.add_argument("--lambda-ps", type=float, default=540.0)
ap.add_argument("--rho-local", default="1.9,2.2")
args = ap.parse_args()


def best_mu(path, q, pool=None):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["form"] != "mu" or int(r["q"]) != q or (pool is not None and int(r["pool"]) != pool):
            continue
        s0, ps = int(r["s0"]), float(r["ps_per_unit"])
        out[s0] = min(out.get(s0, math.inf), ps)
    return out


mu = best_mu(args.mu_csv, 24)
mu[16] = (mu[16] + best_mu(args.mu_csv, 28)[16]) / 2
r22 = best_mu(args.r22_csv, 24, pool=0)
mu[17], mu[18] = r22[17], r22[18]
d19 = (r22[18] + r22[20]) / 2
print("unit costs (ps): " + ", ".join(f"{s}: {mu[s]:.0f}" for s in sorted(mu)) +
      f"; D = 19 interpolated {d19:.0f}; D = 20 {r22[20]:.0f}; lambda {args.lambda_ps:g}")


def unit(s0):
    if s0 == 0:
        return args.fock_ps
    if s0 < 7:
        return mu[7]
    return mu.get(s0, args.lambda_ps)


rhos = [float(v) for v in args.rho_local.split(",")]
scen = {"(a) mu D=19 interp.": {19: d19}, "(b) mu D=19 at D=18": {19: mu[18]},
        "(c) s0 17..30 at D=18": {s: min(mu[18], unit(s)) for s in range(17, 31)}}
print(f"\n{'n':>3} {'mass 19..30':>11} " + " ".join(f"{k:>24}" for k in scen))
for n in (int(v) for v in args.n.split(",")):
    w = {n - k: math.comb(n, k) * 2.0 ** k for k in range(n + 1)}
    mass = sum(v for s, v in w.items() if 19 <= s <= 30) / sum(w.values())
    base = sum(v * unit(s) for s, v in w.items())
    cells = []
    for o in scen.values():
        gain = base / sum(v * o.get(s, unit(s)) for s, v in w.items())
        cells.append(" / ".join(f"+{math.log(gain) / math.log(r):.2f}" for r in rhos))
    print(f"{n:>3} {100 * mass:>10.0f}% " + " ".join(f"{c:>24}" for c in cells))
print(f"\n(Delta n* at rho_local = {' / '.join(map(str, rhos))})")
