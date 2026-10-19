"""Simulability frontier of the native time-loop sampler: for each bin ratio T/N and total loop
length L, the photon number N* at which `LoopRegisterKernel.sample` sustains --target samples/s
(default 1 Hz) with T = ratio*N bins (N photons in the first N bins, one loop of delay L,
random settings).

    python3 bench/timeloop_frontier.py --loops 1,2,3,4,6,8 [--ratios 1,2,4] [--target 1] [--csv ...]

The cost per sample is O(T * D * N) with D = C(N+L+1, L+1), polynomial in N for fixed L, so the
frontier is a curve N*(L) rather than a single number; it is memory-bound before it is
time-bound at large L (the shot batch shrinks with D; the run stops when even one shot does not
fit in --mem-frac of the device memory). Method: measure the sustained rate at increasing N
(fit log rate vs log N, jump towards the prediction, at most --max-jump), stop once two
consecutive N bracket the target and interpolate. Writes --csv (L, N, T, D, shots, samples/s,
ms/sample) and a result block per L.
"""
import argparse
import csv
import math
import time

import numpy as np
import torch

from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit, register_dimension

ap = argparse.ArgumentParser()
ap.add_argument("--loops", default="1,2,3,4,6,8", help="total loop lengths L (single loop each)")
ap.add_argument("--ratios", default="2", help="bin ratios T/N, comma separated (T = round(ratio * N))")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--start", type=int, default=8)
ap.add_argument("--max-n", type=int, default=400)
ap.add_argument("--max-jump", type=float, default=1.6, help="max multiplicative jump in N per step")
ap.add_argument("--budget", type=float, default=120.0, help="seconds per N (shot ramp stops beyond budget/3 per run)")
ap.add_argument("--max-shots", type=int, default=256)
ap.add_argument("--mem-frac", type=float, default=0.4, help="fraction of device memory the shot batch may use")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="timeloop_frontier.csv")
args = ap.parse_args()

dev = args.device
cdtype = getattr(torch, args.dtype)
bytes_per = 16 if cdtype == torch.complex128 else 8
if dev == "cuda":
    mem = torch.cuda.get_device_properties(0).total_memory
    name = torch.cuda.get_device_name(0)
else:
    mem = 32 * 2 ** 30
    name = "cpu"
print(f"{name}  native time-loop sampler  ratios T/N={args.ratios}  target={args.target} samples/s  dtype={args.dtype}", flush=True)
rows = []


def measure(N, L, ratio):
    T = round(ratio * N)
    D = register_dimension(N, L)
    per_shot = D * bytes_per * 6                                  # state, gathered blocks, products (~6 copies, no padding)
    shots_fit = int(args.mem_frac * mem // per_shot)
    if shots_fit < 1:
        print(f"  N={N:3d} L={L} T={T:3d} D={D:>12,d}: does not fit ({per_shot / 2**30:.1f} GB per shot)", flush=True)
        return None
    c = TimeLoopCircuit(T, (L,))
    inp = [1] * N + [0] * (T - N)
    t0 = time.perf_counter()
    lk = LoopRegisterKernel(c, inp, device=dev, dtype=cdtype)
    build = time.perf_counter() - t0
    g = torch.Generator(device=dev).manual_seed(args.seed)
    th = torch.rand(1, T, generator=g, device=dev, dtype=torch.float64) * 2
    ph = torch.rand(1, T, generator=g, device=dev, dtype=torch.float64) * 6
    shots, spent, tot_s, tot_t, last = 1, 0.0, 0, 0.0, 1
    lk.sample(1, th, ph, generator=g)                               # warm-up
    if dev == "cuda":
        torch.cuda.synchronize()
    while shots <= min(args.max_shots, shots_fit):
        t = time.perf_counter()
        lk.sample(shots, th, ph, generator=g)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        spent += dt
        last = shots
        if shots >= 4 or shots == shots_fit:
            tot_s += shots
            tot_t += dt
        if dt > args.budget / 3 or spent > args.budget:
            break
        shots *= 2
    if tot_s == 0:
        tot_s, tot_t = last, dt
    rate = tot_s / tot_t
    print(f"  N={N:3d} L={L} T={T:3d} D={D:>12,d}: {rate:9.3f} samples/s  ({1e3 / rate:9.2f} ms/sample, "
          f"up to {last} shots, build {build:.1f} s)", flush=True)
    rows.append([ratio, L, N, T, D, last, rate, 1e3 / rate])
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ratio", "L", "N", "T", "D", "shots", "samples_per_s", "ms_per_sample"])
        w.writerows(rows)
    return rate


results = {}
cases = [(float(r), int(L)) for r in args.ratios.split(",") for L in args.loops.split(",")]
for ratio, L in cases:
    pts = {}
    N = args.start
    reason = "bracketed"
    while True:
        r = measure(N, L, ratio)
        if r is None:
            reason = "memory"
            break
        pts[N] = r
        ns = sorted(pts)
        lo = [k for k in ns if pts[k] >= args.target]
        hi = [k for k in ns if pts[k] < args.target]
        if lo and hi and min(hi) - max(lo) <= 1:
            break
        if lo and hi:
            nxt = (max(lo) + min(hi)) // 2
        else:
            if len(ns) >= 2:
                slope = np.polyfit(np.log(ns[-2:]), np.log([pts[k] for k in ns[-2:]]), 1)[0]
                slope = min(slope, -1.0)
            else:
                slope = -(L + 3.0)
            n_pred = N * math.exp((math.log(args.target) - math.log(pts[N])) / slope)
            n_pred = min(max(n_pred, N / args.max_jump), N * args.max_jump)
            nxt = math.floor(n_pred) if not hi else math.ceil(n_pred)
            nxt = max(nxt, N + 1) if not hi else min(nxt, min(hi) - 1)
        if nxt in pts or nxt > args.max_n or nxt < 1:
            reason = "max-n"
            break
        N = nxt
    ns = sorted(pts)
    lo = [k for k in ns if pts[k] >= args.target]
    hi = [k for k in ns if pts[k] < args.target]
    print("=== result ===")
    if lo and hi:
        a, b = max(lo), min(hi)
        n_int = a + (math.log(args.target) - math.log(pts[a])) / (math.log(pts[b]) - math.log(pts[a])) * (b - a)
        print(f"  ratio = {ratio:g}  L = {L}: bracket N={a} at {pts[a]:.2f}/s, N={b} at {pts[b]:.2f}/s  ->  N* = {n_int:.1f}")
        results[(ratio, L)] = n_int
    elif lo:
        print(f"  ratio = {ratio:g}  L = {L}: not bracketed ({reason}): last N = {max(lo)} at {pts[max(lo)]:.2f}/s, D = "
              f"{register_dimension(max(lo), L):,d}")
        results[(ratio, L)] = max(lo)
    else:
        print(f"  ratio = {ratio:g}  L = {L}: below target already at N = {ns[0]}")
print("=== summary ===")
for (ratio, L), n in results.items():
    print(f"  ratio={ratio:g} L={L}: N*={n:.1f}")
print(f"  wrote {args.csv}")
