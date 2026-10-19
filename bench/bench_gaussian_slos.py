"""Reach of the Gaussian strong simulation (`slos.GaussianSLOSKernel`, docs/GAUSSIAN.md §7): n squeezers
at --mean photons each on modes 0..n-1, m = round(ratio n), every lossless layer up to the cutoff that
leaves --eps of the photon-number mass out. Prints and writes (n, m, N_cut, states, seconds, peak GiB,
captured mass, max |layer mass - law|).

    python3 bench/bench_gaussian_slos.py --device cuda --n 4,6,8,10 --ratio 2 --csv gaussian_slos.csv
"""
import argparse
import csv
import math
import time

import numpy as np
import torch

from photonic_engines import GaussianInput
from photonic_engines.slos import GaussianSLOSKernel

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="4,6,8")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--mean", type=float, default=0.5, help="sinh^2 r per squeezer")
ap.add_argument("--eps", type=float, default=1e-3, help="uncaptured photon-number mass")
ap.add_argument("--device", default="cpu")
ap.add_argument("--csv", default=None)
args = ap.parse_args()

r = GaussianInput.r_for_mean(args.mean)
rows = []
for n in (int(x) for x in args.n.split(",")):
    m = round(args.ratio * n)
    st = GaussianInput.smsv(n, m, r)
    law = st.photon_number_probs(40 * n + 40)
    cut = int(np.nonzero(1 - np.cumsum(law) <= args.eps)[0][0])
    rng = np.random.default_rng(n)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, rr = np.linalg.qr(z)
    U = q * (np.diag(rr) / np.abs(np.diag(rr)))
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.perf_counter()
    k = GaussianSLOSKernel(st, cut, device=args.device)
    d = k.probs(U)
    if args.device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else float("nan")
    states = sum(math.comb(m + j - 1, j) for j in range(cut + 1))
    err = max(abs(float(d.mass(j)) - law[j]) for j in range(cut + 1))
    mass = sum(float(d.mass(j)) for j in range(cut + 1))
    rows.append((n, m, cut, states, dt, peak, mass, err))
    print(f"n={n:2d} m={m:3d} N_cut={cut:2d} states={states:.3e} {dt:8.2f} s peak {peak:.2f} GiB "
          f"captured {mass:.6f}  max|mass - law| {err:.1e}", flush=True)
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "m", "N_cut", "states", "seconds", "peak_gib", "captured", "mass_err"])
        w.writerows(rows)
