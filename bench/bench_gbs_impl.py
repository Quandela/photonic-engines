"""v1 vs v2 of the Gaussian sampler (docs/GAUSSIAN.md §3.4), same seeds, same device.

    python3 bench/bench_gbs_impl.py --device cuda --n 20,30,40 --shots 16 --windows 1,4,8 --csv gbs_impl.csv

A. one loop-hafnian polynomial: `lhaf_poly` (v1) against `lhaf_poly_multi` (v2) with V = 1 and V = --vmulti
   loop vectors, N = --lhaf-n; reports seconds and the per-vector speedup, and the max relative difference.
B. whole shots at m = ratio n, --mean photons per squeezer: v1 and v2 at each window; the draws consume the
   generator identically, so the samples must agree (reported as the fraction of identical shots) and the
   speedup is per shot on the same shots.
"""
import argparse
import csv
import time

import numpy as np
import torch

from photonic_engines import GaussianInput
from photonic_engines.gaussian import sample
from photonic_engines.gaussian.hafnian import lhaf_poly, lhaf_poly_multi

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--lhaf-n", default="16,20,24,28,32,36")
ap.add_argument("--vmulti", type=int, default=8)
ap.add_argument("--n", default="20,30")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--mean", type=float, default=0.5)
ap.add_argument("--shots", type=int, default=16)
ap.add_argument("--windows", default="1,4,8")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default=None)
args = ap.parse_args()
dev = args.device


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


rows = []
print(f"{torch.cuda.get_device_name(0) if dev == 'cuda' else dev}  torch {torch.__version__}", flush=True)
print("\nA. one loop-hafnian polynomial (random GBS-like matrix B[n] of a squeezed state, no collisions)", flush=True)
for N in (int(x) for x in args.lhaf_n.split(",")):
    m = 2 * N
    st = GaussianInput.smsv(N, m, GaussianInput.r_for_mean(args.mean))
    B, zeta, _ = st.bargmann(haar(m, N))
    rng = np.random.default_rng(N)
    A = B[:N, :N]
    D0 = rng.normal(size=(args.vmulti, N)) * 0.5 + 0j; D1 = B[:N, N:N + args.vmulti].T.copy()
    lhaf_poly(A, D0[0], D1[0], device=dev); lhaf_poly_multi(A, D0[:1], D1[:1], device=dev); sync()
    t = time.perf_counter(); a = lhaf_poly(A, D0[0], D1[0], device=dev); sync(); t1 = time.perf_counter() - t
    t = time.perf_counter(); b = lhaf_poly_multi(A, D0[:1], D1[:1], device=dev); sync(); t2 = time.perf_counter() - t
    t = time.perf_counter(); lhaf_poly_multi(A, D0, D1, device=dev); sync(); t3 = time.perf_counter() - t
    rel = float((a - b[0]).abs().max() / a.abs().max())
    print(f"  N={N:2d}: v1 {t1:8.4f} s   v2 V=1 {t2:8.4f} s ({t1 / t2:4.1f}x)   v2 V={args.vmulti} {t3:8.4f} s "
          f"({args.vmulti * t1 / t3:4.1f}x per vector)   rel diff {rel:.1e}", flush=True)
    rows.append(["lhaf", N, "", "", t1, t2, t3, rel])

print(f"\nB. whole shots, m = {args.ratio:g} n, {args.mean} photons per squeezer, {args.shots} shots", flush=True)
for n in (int(x) for x in args.n.split(",")):
    m = round(args.ratio * n)
    st = GaussianInput.smsv(n, m, GaussianInput.r_for_mean(args.mean))
    U = haar(m, args.seed + n)
    sample(st, U, 1, seed=1, device=dev)                             # warm-up
    t = time.perf_counter(); S1, s1 = sample(st, U, args.shots, seed=args.seed + 7 * n, device=dev, impl="v1",
                                             return_stats=True); T1 = time.perf_counter() - t
    for w in (int(x) for x in args.windows.split(",")):
        t = time.perf_counter()
        S2, s2 = sample(st, U, args.shots, seed=args.seed + 7 * n, device=dev, impl="v2", window=w, return_stats=True)
        T2 = time.perf_counter() - t
        same = float(np.mean(np.all(S1 == S2, axis=1)))
        print(f"  n={n:2d} m={m:3d}: v1 {T1 / args.shots:8.3f} s/shot   v2 window={w} {T2 / args.shots:8.3f} s/shot "
              f"({T1 / T2:4.1f}x)   identical shots {same:.3f}   photons {S1.sum(1).tolist()}", flush=True)
        rows.append(["shot", n, m, w, T1 / args.shots, T2 / args.shots, "", same])
if args.csv:
    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["kind", "N_or_n", "m", "window", "v1_s", "v2_s", "v2_multi_s", "rel_or_identical"])
        wr.writerows(rows)
