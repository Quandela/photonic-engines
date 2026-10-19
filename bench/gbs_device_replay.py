"""Price a device's own samples through the exact sampler's timing model (docs/GAUSSIAN.md §2.2).

    python3 bench/gbs_device_replay.py --samples experiments/qca-data/fig4/samples.npy --calib gbs_timing_calib.json \
        --extra rate.json,sieves.json --per-n 40 --discount self_all --csv borealis_replay.csv

(the Borealis runs: github.com/XanaduAI/xanadu-qca-data, Apache-2.0; samples.npy is [shots, 1, modes] int8).
--per-n stratifies by photon number: the mean is carried by the few highest-N samples of the run, which a uniform
subset misses; every N with at most --per-n samples is priced in full. --discount self_all prices the
collision-discounted term count at every size (a kernel v2.8 is not), the default "self" prices v2.8 as built.

A GBS sample's exact cost is a function of its photon pattern only (`timing.plan_calls`), so the cost of producing a
device's run with our exact PNR sampler is the mean replayed cost over the run's own samples: t_run = mean_s
shot_time(pattern_s), the run-of-S statistic with S = the number of samples the experiment took and the device's own
pattern law (no P(N) model, no collision surrogate). What the replay cannot know: whether the device's distribution
is the ideal one (it is not — loss, noise), which changes how many photons an ideal sample would carry, not how a
pattern is priced.

Per sample, every call is tagged
  measured  N <= the largest whole call timed (the calibration and --extra sieves),
  priced    above it and N <= 2 x the largest pair count with a measured per-term rate (term count x measured
            rate: n <= 32 with the Triton kernel, n <= 64 with --extra from `calibrate --kernel torch`, job 24),
  beyond    above that: per-term rate extrapolated as rate(n_max) (n / n_max)^--rate-power (4 by default: the flop
            scaling of the power route; the torch rates of job 24 fit n^2.9-3.2 over n = 48-64),
and its precision tier from the condition-number fit log10 kappa = --kappa-a + --kappa-b N (runs to N = 48, H100) with
the requirement eps kappa <= --tol (the measured error law, job 27; N eps kappa before 2026-09-28): fp64, double-double
(eps 2^-104), quad-double (eps 2^-208). Without --dd-rate the costs reported are fp64 costs and the tier shares say
which part needs a slower arithmetic; with --dd-rate (and --dd-ref) every call is priced in the arithmetic and pairing
the sieve with precision "auto" would use (TimingModel.tiers; quad-double priced as dd: a lower bound).
"""
import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gbs_timing_model as gtm

TIERS = (("fp64", 2.0 ** -52), ("dd", 2.0 ** -104), ("qd", 2.0 ** -208))


def tier(N, a, b, tol):
    for name, eps in TIERS:
        if eps * 10 ** (a + b * N) <= tol:            # eps kappa: the error law measured in job 27 (was N eps kappa)
            return name
    return ">qd"


def main():
    from photonic_engines.gaussian.timing import plan_calls
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help=".npy [shots, (1,) m] photon counts, or a JSON {'patterns': [...]}")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--discount", default="self", choices=["self", "self_all", "none"])
    ap.add_argument("--rate-power", type=float, default=4.0)
    ap.add_argument("--pairing", default="cross", choices=["cross", "self"],
                    help="grouping of collided calls: cross (the sieve since 2026-09-27) or self (v2.8)")
    ap.add_argument("--kappa-a", type=float, default=0.40)
    ap.add_argument("--kappa-b", type=float, default=0.197)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--max-shots", type=int, default=20000, help="random subset of the run (the mean converges fast)")
    ap.add_argument("--per-n", type=int, default=0, help="stratified: at most this many samples per photon number N, "
                    "each N weighted by its count in the whole run (the tail carries the mean; a uniform subset misses it)")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dd-rate", default=None, help="dd rate JSON: price each call in the arithmetic its round-off "
                    "needs (TimingModel.tiers, kappa fit and tol below); off: fp64 costs with the tier shares only")
    ap.add_argument("--dd-ref", default=None, help="fp64 torch rate JSON for the dd / fp64 ratio past the dd rates")
    ap.add_argument("--cross-penalty", type=float, default=5.0)
    ap.add_argument("--qd-factor", type=float, default=1.0, help="qd / dd time (1: a lower bound; not measured)")
    ap.add_argument("--arith-max", default="qd", choices=["dd", "qd"], help="dd: no qd kernel (the sieve as built)")
    ap.add_argument("--json", default=None, help="summary (mean, quantiles, shares, tiers) as JSON")
    args = ap.parse_args()

    class A:
        pass
    a = A(); a.calib = args.calib; a.extra = args.extra; a.discount = args.discount
    cal, M = gtm.load_model(a)
    M.rate_power = args.rate_power; M.pairing = args.pairing
    if args.dd_rate:
        gtm.add_tiers(M, args.dd_rate, args.dd_ref)
        M.kappa_fit = (args.kappa_a, args.kappa_b); M.tol = args.tol; M.cross_penalty = args.cross_penalty
        M.qd_factor = args.qd_factor; M.arith_max = args.arith_max
    measured_max = max(k[0] for k in M.multi if k[1] == 1)
    priced_max = 2 * max(k[0] for k in M.rate)
    if args.samples.endswith(".json"):
        P = np.array(json.load(open(args.samples))["patterns"], dtype=np.int64)
    else:
        P = np.load(args.samples)
        P = P.reshape(P.shape[0], -1).astype(np.int64)
    S_exp, m = P.shape
    rng = np.random.default_rng(args.seed)
    Ns = P.sum(1)
    if args.per_n:
        idx, wts = [], []
        for N in np.unique(Ns):
            at = np.flatnonzero(Ns == N); k = min(len(at), args.per_n)
            idx += list(rng.choice(at, k, replace=False)); wts += [len(at) / k / S_exp] * k
        sel, w = P[idx], np.array(wts)
    else:
        sel = P if S_exp <= args.max_shots else P[rng.choice(S_exp, args.max_shots, replace=False)]
        w = np.full(len(sel), 1.0 / len(sel))
    print(f"{args.samples}: {S_exp} samples, m = {m}, N mean {Ns.mean():.1f}, max {Ns.max()}, pricing {len(sel)} "
          f"(measured whole calls to N = {measured_max}, per-term rate measured to N = {priced_max}, beyond ~ "
          f"n^{args.rate_power:g})", flush=True)
    rows, tot = [], {"measured": 0.0, "priced": 0.0, "beyond": 0.0}
    tiers = {}
    for wi, pat in zip(w, sel):
        cost = {"measured": 0.0, "priced": 0.0, "beyond": 0.0}
        for c in plan_calls(pat, window=args.window):
            t = M.t_call(c) + c.draws * M._interp(M.draw, c.N)
            k = "measured" if measured_max >= c.N else "priced" if priced_max >= c.N else "beyond"
            cost[k] += t
            # with --dd-rate: the arithmetic of the option the model chose (a grouped call can need qd below N = 141)
            tr = (M.last_arith if c.kind == "multi" else "fp64") if M.tiers else tier(c.N, args.kappa_a, args.kappa_b, args.tol)
            tiers[tr] = tiers.get(tr, 0.0) + wi * t
        t = sum(cost.values()) + (M._interp(M.shot, m) if M.shot else 0.0)
        for k in tot:
            tot[k] += wi * cost[k]
        rows.append((int(pat.sum()), t))
    ts = np.array([r[1] for r in rows]); mean = float(w @ ts); total = sum(tot.values())
    o = np.argsort(ts); cw = np.cumsum(w[o])
    q = {p: ts[o][min(np.searchsorted(cw, p), len(ts) - 1)] for p in (.5, .9, .99)}
    arith = "precision tiers" if M.tiers else "fp64 kernels"
    print(f"t_run (mean replayed cost per sample, {arith}): {mean:.3e} s on one H100;  median {q[.5]:.3e},"
          f" 90 % {q[.9]:.3e}, 99 % {q[.99]:.3e}, max {ts.max():.3e}")
    print("cost shares: " + ", ".join(f"{k} {100 * v / total:.1f} %" for k, v in tot.items())
          + ";  precision tiers: " + ", ".join(f"{k} {100 * v / total:.1f} %" for k, v in sorted(tiers.items())))
    print(f"H100s for 1 sample/s: {mean:.2e};  seconds for the whole run of {S_exp} samples on one H100: "
          f"{mean * S_exp:.2e}")
    if args.json:
        json.dump({"samples": args.samples, "S_exp": int(S_exp), "m": int(m), "N_mean": float(Ns.mean()), "N_max": int(Ns.max()),
                   "priced_samples": len(sel), "discount": args.discount, "pairing": args.pairing, "tiers": bool(M.tiers),
                   "rate_power": args.rate_power,
                   "qd_factor": args.qd_factor, "arith_max": args.arith_max, "mean_s": mean, "median_s": float(q[.5]),
                   "p90_s": float(q[.9]),
                   "p99_s": float(q[.99]), "max_s": float(ts.max()), "run_s": mean * S_exp,
                   "shares": {k: v / total for k, v in tot.items()},
                   "tier_shares": {k: v / total for k, v in tiers.items()}}, open(args.json, "w"), indent=1)
    byN = {}
    for N, t in rows:
        byN.setdefault(N, []).append(t)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            cw_ = csv.writer(f, lineterminator="\n")
            cw_.writerow(["N", "count_run", "priced", "mean_s", "tier"])
            for N in sorted(byN):
                cw_.writerow([N, int((Ns == N).sum()), len(byN[N]), float(np.mean(byN[N])),
                              tier(N, args.kappa_a, args.kappa_b, args.tol)])


if __name__ == "__main__":
    main()
