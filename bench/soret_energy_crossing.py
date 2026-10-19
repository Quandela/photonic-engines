"""
Energetic crossing of Soret et al. (arXiv:2601.08068v1) recomputed on their own metric M.

Steps: (2) reproduce M_e = 17 (fixed l = 12, their appendix VI.3) and M_e* = 15
(binomial averaging, their Fig. 2a) from their equations; (3) change only the classical side: (a) C&C 2020 work instead
of C&C 2018, (b) measured H100 energy per exact sample, (a + b).

Their model, as read from the v1 HTML (equation numbers are theirs):
  m = ceil(2.1 n)                         n = photons emitted by the device, m = modes = detectors
  c_S(n, m)     = (m/n)/(m/n + 1) n                                   (6)
  c_S^l(n, m)   = (m/(n-l))/(m/(n-l) + 1) (n-l)                       (7)
  k_eff         = ln[(eps delta/2)(1 - I^2 c_S^l/c_S)] / ln[I^2 c_S^l/c_S] - 1,  eps delta = 1e-3   (9)
  M             = min(n - l, k_eff)                                   (10)
  L(m, n)       = sum_{j=2}^{n} (j 2^j + m j)                         (11)  (C&C 2018)
  E^C           = L(m, M) / eta_e,  eta_e = 72.733 GFLOPS/W           (12)
  P_cryo(T, m)  = P0/eta_c(T) (1 + floor(m/s_max)),  eta_c = x T/(T_ext - T), x = 0.4 %, P0 = 50 mW, s_max = 26  (2)
  P_tot         = P_fix + P_cryo,  P_fix = 1 kW                       (3)
  nbar          = ceil(n / (1 + floor(m/s_max))),  r_sample = r_SPS / nbar,  r_SPS = 1 GHz   (13)
  E^Q           = P_tot / r_sample                                    (14)
  eta(m)        = eta_of eta_d eta_dmx 10^(-2 c_coup/10) 10^(-m c_MZI/10)   (17), Table 1 'energetic' column
Not given in the paper, assumed here (printed in the output):
  * I(T): only "I = 95 %" at the optimal temperature (their "33 K", read as 3 K); eta_ZPL(T) is not given, so I and T
    are fixed (--I, --T).
  * L at non-integer M: not specified. 2018: closed form at real M ('cont') or floor(M); 2020 and GPU: log-linear
    between floor(M) and ceil(M).
  * Averaging (Fig. 2): N = 10^4 draws of the detected count ~ Binomial(n, p), <E^C> over draws, x-axis <M>.
    p = eta(m) of eq. (17) ('device', reproduces their appendix crossing point n = 24, m = 51, 60 %) or (n - l)/n.

Classical sides (--classical, comma list):
  cc2018      eq. (11)-(12) as theirs.
  cc2020      C&C 2020 (arXiv:2005.04214) Haar-averaged work: the 2^j of step j becomes E[prod_nu (s_nu + 1)] over the
              j rows drawn, which for the Haar-averaged (uniform-multiset) law of j photons in m modes is
              C(2m+j-1, j) / C(m+j-1, j)  ([x^j] (1-x)^(-2m) over the number of multisets). Same eta_e.
  gpu         measured: W / R(M) per sample, R log-interpolated in the detected count through --rates
              (bench/rate_series.py plateau CSV at m/n = --ratio) or the built-in old-kernel plateau rates of
              bench/plot_energy.py; below the smallest measured n the smallest-n rate is used (an over-estimate).

    python3 bench/soret_energy_crossing.py [--I 0.95] [--T 3] [--classical cc2018,cc2020,gpu] [--rates CSV]
"""
import argparse
import csv
import math

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--I", type=float, default=0.95)
ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--l", type=int, default=12)
ap.add_argument("--draws", type=int, default=10000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--classical", default="cc2018,cc2020,gpu")
ap.add_argument("--rates", default="", help="comma list of plateau-rate CSVs (columns n, ratio, plateau_per_s or "
                                            "samples_per_s)")
ap.add_argument("--ratio", type=float, default=2.0, help="m/n of the rate CSV rows to use")
ap.add_argument("--cost-table", default="", help="bench/cost_table.py CSV (m, k, samples_per_s): rates at fixed m "
                                                 "(--cost-m), monotone envelope; replaces --rates")
ap.add_argument("--cost-m", type=int, default=48)
ap.add_argument("--energy", default="", help="measured-energy CSV (n, wh_per_sample[, wh_per_sample_above_idle]) at "
                                             "m = 2n, e.g. results/keep/energy_r2_20261011_h100.csv; replaces "
                                             "--watts x rates")
ap.add_argument("--above-idle", action="store_true", help="with --energy: use wh_per_sample_above_idle")
ap.add_argument("--watts", type=float, default=280.0, help="H100 board power during sampling")
ap.add_argument("--table", action="store_true", help="print the per-n table")
ap.add_argument("--x-points", default="", help="m:M list, e.g. 51:14,51:15,59:16,63:17,66:18: print X = L_2018 / "
                                               "L_2020, the per-photon growth of L_2020 and ln X / ln rho, then exit")
args = ap.parse_args()

EPSD, ETA_E = 1e-3, 72.733e9
P0, X, T_EXT, S_MAX, P_FIX, R_SPS = 0.05, 0.004, 300.0, 26, 1000.0, 1e9
WH = 3600.0
# Old-kernel plateau rates at m = 2n (bench/plot_energy.py MEAS_FALLBACK).
RATES_FALLBACK = {8: 40e6, 12: 4.9e6, 16: 808e3, 20: 61.7e3, 22: 16.2e3, 24: 4.2e3, 30: 105, 34: 9.3}


def modes(n):
    return math.ceil(2.1 * n - 1e-9)


def eta_dev(m, of=0.75, d=0.99, dmx=0.92, coup=0.06, mzi=0.0085):
    """Eq. (17), Table 1 energetic-advantage column; d(m) = m."""
    return of * d * dmx * 10 ** (-2 * coup / 10) * 10 ** (-m * mzi / 10)


def k_eff(n, det, m):
    if det <= 0:
        return 0.0
    x = args.I ** 2 * (m * det / (m + det)) / (m * n / (m + n))
    return math.log(EPSD / 2 * (1 - x)) / math.log(x) - 1


def metric(n, det):
    return max(0.0, min(det, k_eff(n, det, modes(n))))


def E_quantum(n):
    m = modes(n)
    ncryo = 1 + m // S_MAX
    p_tot = P_FIX + P0 / (X * args.T / (T_EXT - args.T)) * ncryo
    return p_tot * math.ceil(n / ncryo) / R_SPS / WH


def log_binom(a, b):
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def flops(m, M, algo):
    """Integer M: eq. (11) with 2^j replaced by the C&C 2020 Haar-averaged count when algo == 'cc2020'."""
    tot = 0.0
    for j in range(2, M + 1):
        w = 2.0 ** j if algo == "cc2018" else math.exp(log_binom(2 * m + j - 1, j) - log_binom(m + j - 1, j))
        tot += j * w + m * j
    return tot


if args.x_points:
    print(f"{'m':>4} {'M':>3} {'X = L18/L20':>11} {'rho_2020':>8} {'ln X/ln rho':>11}")
    for m, M in (tuple(int(v) for v in p.split(":")) for p in args.x_points.split(",")):
        x = flops(m, M, "cc2018") / flops(m, M, "cc2020")
        rho = flops(m, M, "cc2020") / flops(m, M - 1, "cc2020")
        print(f"{m:>4} {M:>3} {x:>11.2f} {rho:>8.2f} {math.log(x) / math.log(rho):>11.2f}")
    raise SystemExit(0)

rates = {}
for p in [x for x in args.rates.split(",") if x]:
    for r in csv.DictReader(open(p)):
        if abs(float(r["ratio"]) - args.ratio) < 1e-9:
            v = r.get("plateau_per_s") or r.get("samples_per_s")
            if v:
                rates[int(r["n"])] = float(v)
rate_src = args.rates or "built-in old-kernel plateau rates (bench/plot_energy.py)"
if args.cost_table:
    raw = {int(r["k"]): float(r["samples_per_s"]) for r in csv.DictReader(open(args.cost_table))
           if int(r["m"]) == args.cost_m}
    # Monotone envelope: the dips are Triton recompiles in the timed runs.
    rates = {k: max(raw[q] for q in raw if q >= k) for k in raw}
    rate_src = f"{args.cost_table}, m = {args.cost_m} (k photons, fixed m), monotone envelope"
if args.energy:
    col = "wh_per_sample_above_idle" if args.above_idle else "wh_per_sample"
    # Stored as equivalent rates at --watts so that gpu_energy() is unchanged: R = W / (3600 E).
    rates = {int(r["n"]): args.watts / (WH * float(r[col])) for r in csv.DictReader(open(args.energy))}
    rate_src = f"{args.energy} ({col}, measured board energy)"
if not rates:
    rates = dict(RATES_FALLBACK)
rk = sorted(rates)


def gpu_energy(M):
    """Wh per sample at integer M detected photons; log-interpolated, flat below the smallest measured n,
    extrapolated with the last measured slope above the largest."""
    lr = np.log([rates[k] for k in rk])
    if rk[0] >= M:
        lrate = lr[0]
    elif rk[-1] <= M:
        lrate = lr[-1] + (M - rk[-1]) * (lr[-1] - lr[-2]) / (rk[-1] - rk[-2])
    else:
        lrate = np.interp(M, rk, lr)
    return args.watts / math.exp(lrate) / WH


_E_CACHE = {}


def E_classical(m, M, side):
    """Wh per sample with metric M (real): log-linear between floor and ceil."""
    key = (m, side)
    if key not in _E_CACHE:
        _E_CACHE[key] = {}
    tab = _E_CACHE[key]

    def at(k):
        if k not in tab:
            if k < 2:
                tab[k] = 0.0 if side != "gpu" else gpu_energy(max(k, 1))
            elif side == "gpu":
                tab[k] = gpu_energy(k)
            else:
                tab[k] = flops(m, k, side) / ETA_E / WH
        return tab[k]

    if side == "cc2018cont":
        return 0.0 if M < 2 else ((M - 1) * 2 ** (M + 1) + m * (M * (M + 1) / 2 - 1)) / ETA_E / WH
    lo, hi = math.floor(M), math.ceil(M)
    a, b = at(lo), at(hi)
    if lo == hi or a <= 0:
        return b if lo < M else a
    return math.exp(math.log(a) + (M - lo) * (math.log(b) - math.log(a)))


sides = args.classical.split(",")
if "cc2018" in sides:
    sides.insert(sides.index("cc2018") + 1, "cc2018cont")
print(f"assumed I = {args.I}, T = {args.T} K, l = {args.l}, eps delta = {EPSD}, eta_e = {ETA_E / 1e9:g} GFLOPS/W,"
      f" draws = {args.draws}, seed = {args.seed}")
if "gpu" in sides:
    print(f"gpu: {args.watts:g} W / R(M), R from {rate_src}, m/n = {args.ratio:g}: "
          + ", ".join(f"{k}: {rates[k]:.3g}/s" for k in rk))
print()

rng = np.random.default_rng(args.seed)
res = {}  # (draw, side) -> list of (n, x, EC, EQ)
for n in range(args.l + 2, 60):
    m = modes(n)
    det = n - args.l
    eq = E_quantum(n)
    M = metric(n, det)
    for s in sides:
        res.setdefault(("fixed", s), []).append((n, M, E_classical(m, M, s), eq))
    for draw, p in (("device", eta_dev(m)), ("n-l", det / n)):
        Ms = [metric(n, int(d)) for d in rng.binomial(n, p, size=args.draws)]
        for s in sides:
            res.setdefault((draw, s), []).append((n, float(np.mean(Ms)), float(np.mean([E_classical(m, x, s) for x in Ms])),
                                                  eq))

if args.table:
    for key, rows in res.items():
        print(f"-- {key[0]} / {key[1]}")
        print(f"{'n':>3} {'m':>3} {'M':>6} {'E^C Wh':>10} {'E^Q Wh':>10}")
        for n, x, ec, eq in rows:
            print(f"{n:3d} {modes(n):3d} {x:6.2f} {ec:10.3e} {eq:10.3e}")
        print()


def crossing(rows):
    """Crossings of E^C with E^Q (sign changes of ln E^C/E^Q), log-linear in x between neighbouring n."""
    out, prev = [], None
    for n, x, ec, eq in rows:
        r = math.log(ec / eq) if ec > 0 else -math.inf
        if prev and (prev[1] < 0) != (r < 0) and math.isfinite(prev[1]):
            x0, r0, n0 = prev
            out.append((x0 + (x - x0) * (-r0) / (r - r0), n0, n, "up" if r > r0 else "down"))
        prev = (x, r, n)
    return out


print(f"{'averaging':>9} {'classical':>11}  crossings (M, between device n, direction of E^C - E^Q)")
for (draw, s), rows in res.items():
    cs = crossing(rows)
    txt = "; ".join(f"M = {x:.2f} (n {a}-{b}, {d})" for x, a, b, d in cs) or \
        ("none: E^C > E^Q everywhere" if rows[0][2] > rows[0][3] else "none: E^C < E^Q everywhere")
    print(f"{draw:>9} {s:>11}  {txt}")
print("\nSoret et al.: M_e = 17 (fixed l = 12, appendix VI.3), M_e* = 15 (averaged, Fig. 2a); "
      "E^C = E^Q = 1.0e-8 Wh at n = 24, m = 51, eta = 60 %")
