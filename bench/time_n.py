"""Time single samples at a given (n, m/n): the control points of the frontier extrapolation,
and the per-sample cost distribution.

Two things the throughput map cannot give:

1. **Control points beyond the 1-GPU frontier.**  The mean wall time T of one sample at (n, c)
   *is* the number of GPUs that would reach 1 sample/s there (independent samples on
   independent GPUs), so timing single samples directly tests
   n*(G) = n*(1) + ln G / ln rho_eff without owning G GPUs.  Deviations say the extrapolation
   is wrong; agreement is the only evidence the multi-GPU curves in the paper rest on.

2. **The cost distribution.**  The cost of a sample varies with its collision pattern (0.3x-10x
   around the mean; two consecutive samples at n = 48, m = n took 542 ms and 5825 ms).  The
   mean is carried by the rare low-collision instances, so a throughput point measured on a
   handful of samples has a much wider error bar than its wall time suggests.  This prints
   mean / median / min / max / mean-over-median and writes every individual time, so the tail
   can be fitted rather than assumed.

    python3 bench/time_n.py --points 2:41,2:45,1:46,1:50 --samples 8
    python3 bench/time_n.py --points 2:34 --samples 40 --csv tail_34.csv   # distribution only

A point is `ratio:n`.  Each point stops at --samples samples or --budget seconds, whichever
comes first, so an over-ambitious list degrades gracefully instead of running all night.
With --n-star / --rho given, the predicted GPU count is printed next to the measured one.
"""
import argparse
import csv
import os
import statistics
import time

import torch

from photonic_engines.clifford import fock_sample


def rho_theory(c):
    return (2 * c + 1) ** (2 * c + 1) / ((4 * c) ** c * (c + 1) ** (c + 1))


def haar(m, dev, gen):
    X = torch.randn(m, m, 2, generator=gen, dtype=torch.float64, device=dev)
    return torch.linalg.qr(torch.view_as_complex(X))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="2:41,2:45,1:46,1:50",
                    help="comma list of ratio:n (m = round(ratio*n))")
    ap.add_argument("--samples", type=int, default=8, help="max samples per point")
    ap.add_argument("--budget", type=float, default=900.0, help="max seconds per point")
    ap.add_argument("--n-star", default="1:42.5,2:38.6,4:36.2",
                    help="ratio:n* from the throughput map, for the predicted GPU count")
    ap.add_argument("--rho", default="1:1.750,2:1.806,4:1.911", help="ratio:rho_eff, same source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    dev = torch.device(a.device)
    nstar = {float(k): float(v) for k, v in (p.split(":") for p in a.n_star.split(",") if p)}
    rhos = {float(k): float(v) for k, v in (p.split(":") for p in a.rho.split(",") if p)}
    out = a.csv or os.path.join(os.environ.get("RESULTS_DIR", "results"), "time_n.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    try:                                            # compile every tier before anything is timed
        from photonic_engines.clifford.cuda_cofactors import warmup
        warmup()
    except Exception:
        pass

    print(f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev}   "
          f"max {a.samples} samples or {a.budget:.0f} s per point")
    print(f"{'m/n':>5} {'n':>4} {'m':>5} {'N':>3} {'mean s':>9} {'median':>9} {'min':>8} {'max':>9} "
          f"{'mean/med':>8} {'GPUs meas':>10} {'pred':>7}")
    rows = []
    for spec in a.points.split(","):
        if not spec:
            continue
        c, n = spec.split(":")
        c, n = float(c), int(n)
        m = round(c * n)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        U = haar(m, dev, gen)
        in_modes = list(range(n))
        fock_sample(U, in_modes, 1, generator=gen, algo="auto", backend="cuda")   # warm-up sample
        torch.cuda.synchronize()

        ts, t_start = [], time.perf_counter()
        while len(ts) < a.samples and time.perf_counter() - t_start < a.budget:
            t0 = time.perf_counter()
            fock_sample(U, in_modes, 1, generator=gen, algo="auto", backend="cuda")
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        if not ts:
            continue
        mean, med = statistics.fmean(ts), statistics.median(ts)
        pred = ""
        if c in nstar and c in rhos:
            # rate at n predicted by the map: 1 Hz at n*, divided by rho_eff per extra photon
            pred = f"{rhos[c] ** (n - nstar[c]):7.1f}"
        print(f"{c:>5g} {n:>4} {m:>5} {len(ts):>3} {mean:>9.2f} {med:>9.2f} {min(ts):>8.2f} "
              f"{max(ts):>9.2f} {mean / med:>8.2f} {mean:>10.1f} {pred:>7}")
        for i, t in enumerate(ts):
            rows.append({"ratio": c, "n": n, "m": m, "sample": i, "seconds": t, "mean": mean, "median": med,
                             "gpus_measured": mean,
                             "gpus_predicted": (rhos[c] ** (n - nstar[c])) if (c in nstar and c in rhos) else ""})

    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} samples)")
        print("Reading: `GPUs meas` is the mean seconds per sample = the number of independent GPUs\n"
              "that reach 1 sample/s at this (n, m/n); `pred` is what the throughput map's n* and\n"
              "rho_eff predict.  They should agree within the sampling noise (a factor ~1.5 at 8\n"
              "samples).  A systematic excess means the extrapolation is optimistic.")


if __name__ == "__main__":
    main()
