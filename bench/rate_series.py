"""Sustained throughput at a list of photon numbers — the series quoted in the paper's
"Results" paragraph and on the deck, measured with exactly the protocol of
`bench/find_threshold.py` so the points sit on the same curve as the frontier.

    python3 bench/rate_series.py --ratio 2 --ns 12,16,20,24,30,34
    python3 bench/rate_series.py --ratio 2 --ns 8,12,16,20,24,30,34 --csv results/rate_r2.csv

Two rates are reported, because the two existing number sets use two different definitions and
they only coincide once one sample already fills the GPU (n >~ 22 at m = 2n):

  sustained  total samples / total time over the ramp runs with at least --min-shots shots.
             This is `find_threshold.measure` verbatim, so it is the definition behind n* and
             the frontier table -- use it for anything that has to sit on that curve.
  plateau    the best single run of the ramp, i.e. the rate at the largest batch reached.
             This is the definition behind the reference rates below and the paper's
             results series, and it is what a user batching properly actually gets.

At small n the ramp is launch-bound, so the small batches drag the average down and `sustained`
sits up to 2x below `plateau` (measured: 0.48x at n = 8, 0.94x at n = 20, 1.00x from n ~ 22).
Quoting one against the other is a definition mismatch, not a speed difference.

Ramp: 1, 2, 4, ... up to --max-shots; it stops when one run exceeds --budget/3 or the point
exceeds --budget.  A warm-up sample is run and timed separately, so kernel compilation never
lands inside a measured region.

Also prints the ratio against these CPU and GPU reference points (plateau rates at m = 2n),
which do not depend on our kernel and so need no re-measurement:

    Perceval CPU (24 cores):  2.7 M/s (n=8), 136 k/s (12), 6.2 k/s (16), 302/s (20), 14.8/s (24)
    exqalibur GPU (H100):     131 k/s (12), 13.8 k/s (16), 779/s (20), 40.6/s (24)
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

# Reference rates -- Perceval Clifford2017Backend on 24 EPYC
# cores and the exqalibur Clifford2017GPU runner on the same H100.  They are m = 2n
# measurements and do not depend on this kernel, so they are quoted, not re-measured; the
# comparison columns are printed only at ratio 2.
REF_CPU = {8: 2.7e6, 12: 1.36e5, 16: 6.2e3, 20: 302.0, 22: 67.0, 24: 14.8}
REF_GPU = {12: 1.31e5, 16: 1.38e4, 20: 779.0, 22: 184.0, 24: 40.6}


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--ns", default="8,12,16,20,24,28,30,34")
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per n")
    ap.add_argument("--max-shots", type=int, default=1 << 18,
                    help="ramp ceiling; the sampler runs passes of max_streams = 2^18, and the "
                         "small-n plateau is only reached at that order of batch size")
    ap.add_argument("--min-shots", type=int, default=8)
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    from photonic_engines.clifford import fock_sample
    try:
        from photonic_engines.clifford.cuda_cofactors import warmup
        warmup()
    except Exception:
        pass

    dev = a.device
    out = a.csv or os.path.join(os.environ.get("RESULTS_DIR", "results"), f"rate_r{a.ratio:g}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"{torch.cuda.get_device_name(0) if dev == 'cuda' else dev}  m/n={a.ratio:g}  "
          f"backend={a.backend}  budget={a.budget:g} s per n")
    print(f"{'n':>4} {'m':>5} {'plateau/s':>12} {'sustained/s':>12} {'pl/su':>6} {'ms/sample':>10} "
          f"{'shots':>7} {'s':>5}   {'x Perceval':>11} {'x exqalibur':>12}")

    rows = []
    for n in [int(x) for x in a.ns.split(",")]:
        m = round(a.ratio * n)
        Ut = torch.tensor(haar(m, a.seed + n), device=dev)
        in_modes = list(range(n))
        rid = [0]

        def run(shots):
            rid[0] += 1
            gen = torch.Generator(device=dev).manual_seed(a.seed + 1000 * rid[0])
            torch.cuda.synchronize()
            t = time.perf_counter()
            occ = fock_sample(Ut, in_modes, shots, generator=gen, algo="auto", backend=a.backend)
            torch.cuda.synchronize()
            return time.perf_counter() - t, occ

        run(1)                                                   # warm-up, not timed
        shots, spent, t0 = 1, 0.0, time.perf_counter()
        tot_shots, tot_time, best, last, plateau = 0, 0.0, 0.0, 1, 0.0
        while shots <= a.max_shots:
            dt, occ = run(shots)
            last, spent = shots, spent + dt
            if shots >= a.min_shots:
                tot_shots, tot_time = tot_shots + shots, tot_time + dt
                plateau = max(plateau, shots / dt)          # best single run = largest batch
            best = shots / dt if tot_shots == 0 else tot_shots / tot_time
            if dt > a.budget / 3 or spent > a.budget:
                break
            shots *= 2
        plateau = plateau or best
        # first-moment sanity check, as in find_threshold
        X = occ.cpu().numpy().astype(np.float64)
        U = Ut.cpu().numpy()
        err = float(np.abs(X.mean(0) - (np.abs(U[:, :n]) ** 2).sum(1)).max())
        ref = abs(a.ratio - 2.0) < 1e-9                    # the references are m = 2n only
        # the references are plateau rates, so compare plateau with plateau
        rc = f"{plateau / REF_CPU[n]:10.0f}x" if (ref and n in REF_CPU) else "          -"
        rg = f"{plateau / REF_GPU[n]:11.0f}x" if (ref and n in REF_GPU) else "           -"
        print(f"{n:>4} {m:>5} {plateau:>12.4g} {best:>12.4g} {plateau / best:>6.2f} "
              f"{1e3 / plateau:>10.4g} {last:>7} {time.perf_counter() - t0:>5.0f}   {rc} {rg}   "
              f"max|dE[n_j]|={err:.3f}")
        rows.append({"n": n, "m": m, "ratio": a.ratio, "plateau_per_s": plateau, "sustained_per_s": best,
                         "ms_per_sample_plateau": 1e3 / plateau, "shots": last,
                         "seconds": round(time.perf_counter() - t0),
                         "x_perceval_cpu": (plateau / REF_CPU[n]) if (ref and n in REF_CPU) else "",
                         "x_exqalibur_gpu": (plateau / REF_GPU[n]) if (ref and n in REF_GPU) else "",
                         "moment_err": err})

    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}")
        print("The `x Perceval` / `x exqalibur` columns divide by the reference rates of "
              "this script's docstring;\nthose references do not depend on this kernel, so they "
              "are quoted, not re-measured.")


if __name__ == "__main__":
    main()
