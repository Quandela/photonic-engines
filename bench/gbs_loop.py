"""GBS through time-bin loops at small delay (the Gaussian row of the loop frontier): the source
number N* at which a cutoff-register sampler sustains --target samples/s, as a curve N*(L), on the axis and with the
settings of bench/hybrid_frontier.py (one rail, one loop of delay L, T = round(ratio * N) bins, loops start empty,
random settings theta in [0, 2), phi in [0, 6)), and with the GBS headline source: N single-mode squeezed vacua of
0.5 photons each (r = 0.658) in bins 0..N-1, lossless (docs/GAUSSIAN.md §2).

    python3 bench/gbs_loop.py count --loops 1,2,3,4,6,8 --n 16,64,256,1024        # register cap p and size D (CPU)
    python3 bench/gbs_loop.py count --ladder 1,6,36 --n 216 --ratio 1              # Borealis' ladder (register count)
    python3 bench/gbs_loop.py check --loops 1,2 --n 6 --cap 4,6,8 --shots 4000     # leak model vs sampler weights
    python3 bench/gbs_loop.py frontier --loops 1,2,3,4,5,6,8 --csv gbs_loop.csv    # N*(L) on the GPU

A squeezed train has no finite Fock register: the register (the rail wire plus the L loop slots) is capped at p photons
(the graded basis of `SuperpositionLoopKernel`, D = C(p + L + 1, L + 1)) and every source is truncated at p photons.
The cutoff convention is the GBS one: at most eps = 1/S of the probability mass of a sample is left out, S = 10^4.

The cap and the sampler: photonic_engines/gaussian/loop.py (leak model, capped `SuperpositionLoopKernel`). The sampler
renormalises after every injection, so the rows are draws of the per-step renormalised truncated process; the weights
give the leak, which is reported next to the model's. Loss: none (the headline is lossless; uniform loss would be output
thinning at no register cost).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")
from photonic_engines.gaussian.loop import R_HEAD, cap, gbs_loop_kernel, settings


def cmd_count(a):
    ladders = [tuple(int(x) for x in a.ladder.split(","))] if a.ladder else [(int(L),) for L in a.loops.split(",")]
    rows = []
    for ladder in ladders:
        L = sum(ladder)
        for N in (int(x) for x in a.n.split(",")):
            t0 = time.perf_counter()
            p, lk, _, nmax = cap(ladder, N, a.ratio, a.r, a.eps, a.seed, a.pmax)
            D = comb(p + L + 1, L + 1) if p is not None else None
            rows.append({"ladder": "-".join(map(str, ladder)), "L": L, "N": N, "T": round(a.ratio * N), "p": p, "leak": lk, "D": D,
                             "mean_reg_max": nmax})
            print(f"  L={L:2d} ({'-'.join(map(str, ladder))}) N={N:5d} T={round(a.ratio * N):5d}: p = {p}, leak {lk:.2e}, "
                  f"D = C(p+L+1, L+1) = {D:.3e}, max mean register photons {nmax:.2f}  ({time.perf_counter() - t0:.1f} s)",
                  flush=True)
    if a.json:
        json.dump({"args": vars(a), "rows": rows}, open(a.json, "w"), indent=1)


def cmd_check(a):
    import torch

    from photonic_engines.timeloop import TimeLoopCircuit
    dt = getattr(torch, a.dtype)
    for L in (int(x) for x in a.loops.split(",")):
        for N in (int(x) for x in a.n.split(",")):
            T = round(a.ratio * N)
            c = TimeLoopCircuit(T, (L,))
            th, ph, g = settings(1, T, a.seed, a.device)
            _, _, leak, _ = cap((L,), N, a.ratio, a.r, a.eps, a.seed, a.pmax, th.cpu().numpy(), ph.cpu().numpy())
            for p in (int(x) for x in a.cap.split(",")):
                k = gbs_loop_kernel(c, N, a.r, p, a.device, dt)
                _, lw = k.sample_weighted(a.shots, th, ph, generator=g)
                m, se = float(lw.mean()), float(lw.std()) / math.sqrt(a.shots)
                print(f"  L={L} N={N} T={T} p={p} D={k.D}: leak model {leak[p]:.4e}, sampler weights {m:.4e} +- {se:.1e}",
                      flush=True)


def cmd_frontier(a):
    import torch

    from photonic_engines.timeloop import TimeLoopCircuit
    dev, dt = a.device, getattr(torch, a.dtype)
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"{name}  GBS loop frontier  r={a.r:.4f} ({math.sinh(a.r) ** 2:.2f} photons per squeezer)  ratio T/N={a.ratio}  "
          f"eps={a.eps:g}  target={a.target} samples/s  max-shots {a.max_shots}  dtype={a.dtype}", flush=True)
    rows, results = [], {}

    def free():
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    def measure(N, ladder):
        free()
        L = sum(ladder)
        T = round(a.ratio * N)
        th, ph, g = settings(len(ladder), T, a.seed, dev)
        t0 = time.perf_counter()
        p, lk_model, _, nmax = cap(ladder, N, a.ratio, a.r, a.eps, a.seed, a.pmax, th.cpu().numpy(), ph.cpu().numpy())
        t_cap = time.perf_counter() - t0
        if p is None:
            print(f"  N={N} L={L}: no cap <= {a.pmax} meets eps", flush=True)
            return None
        D = comb(p + L + 1, L + 1)
        if a.mem_frac * (torch.cuda.get_device_properties(0).total_memory if dev == "cuda" else 32 << 30) < D * 16 * 8:
            print(f"  N={N} L={L} p={p} D={D:,d}: does not fit", flush=True)
            return None
        t0 = time.perf_counter()
        c = TimeLoopCircuit(T, ladder)
        k = gbs_loop_kernel(c, N, a.r, p, dev, dt)
        build = time.perf_counter() - t0
        try:
            k.sample_weighted(1, th, ph, generator=g)                       # warm-up
        except torch.cuda.OutOfMemoryError:
            print(f"  N={N} L={L} p={p} D={D:,d}: out of memory at one shot", flush=True)
            return None
        shots, spent, tot_s, tot_t, last, leaks, d_t, d_last = 1, 0.0, 0, 0.0, 1, [], 0.0, 0.0
        while shots <= a.max_shots:
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                _, lw = k.sample_weighted(shots, th, ph, generator=g)
            except torch.cuda.OutOfMemoryError:
                free()
                break
            if dev == "cuda":
                torch.cuda.synchronize()
            d_t = time.perf_counter() - t
            spent += d_t
            last, d_last = shots, d_t
            leaks.append(lw.double().cpu())
            if shots >= 4:
                tot_s += shots
                tot_t += d_t
            if d_t > a.budget / 3 or spent > a.budget:
                break
            shots *= 2
        if tot_s == 0 or a.rate_at_max:                    # --rate-at-max: the largest batch reached alone
            tot_s, tot_t = last, d_last
        rate = tot_s / tot_t
        lw = torch.cat(leaks)
        lk_s, lk_se = float(lw.mean()), float(lw.std()) / math.sqrt(max(1, lw.numel()))
        print(f"  N={N:5d} L={L} ({'-'.join(map(str, ladder))}) T={T:5d} p={p:2d} D={D:>11,d}: {rate:9.3f} samples/s "
              f"({1e3 / rate:9.2f} ms/sample, {'at the batch of' if a.rate_at_max else 'up to'} {last} shots; build "
              f"{build:.1f} s, cap {t_cap:.1f} s); leak model {lk_model:.2e}, weights {lk_s:.2e} +- "
              f"{lk_se:.0e}; max mean register photons {nmax:.2f}", flush=True)
        rows.append({"L": L, "ladder": "-".join(map(str, ladder)), "N": N, "T": T, "p": p, "D": D, "shots": last,
                     "rate_basis": "largest batch" if a.rate_at_max else "batches >= 4 pooled",
                     "samples_per_s": rate, "ms_per_sample": 1e3 / rate,
                         "leak_model": lk_model, "leak_weights": lk_s, "leak_weights_se": lk_se, "mean_reg_max": nmax,
                         "build_s": build, "cap_s": t_cap})
        if a.csv:
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                w.writeheader(); w.writerows(rows)
        return rate

    ladders = [tuple(int(x) for x in a.ladder.split(","))] if a.ladder else [(int(x),) for x in a.loops.split(",")]
    for ladder in ladders:
        L = "-".join(map(str, ladder)) if len(ladder) > 1 else ladder[0]      # the label of this curve
        pts, N, reason = {}, a.start, "bracketed"
        while True:
            rt = measure(N, ladder)
            if rt is None:
                reason = "memory"
                break
            pts[N] = rt
            lo = [n for n in sorted(pts) if pts[n] >= a.target]
            hi = [n for n in sorted(pts) if pts[n] < a.target]
            if lo and hi and min(hi) - max(lo) <= max(1, int(a.rel_bracket * max(lo))):
                break
            if lo and hi:
                nxt = (max(lo) + min(hi)) // 2
            else:
                ns = sorted(pts)
                slope = np.polyfit(np.log(ns[-2:]), np.log([pts[n] for n in ns[-2:]]), 1)[0] if len(ns) >= 2 else -2.0
                slope = min(slope, -0.5)
                pred = N * math.exp((math.log(a.target) - math.log(pts[N])) / slope)
                nxt = int(min(max(pred, N / a.max_jump), N * a.max_jump))
                nxt = max(nxt, N + 1) if not hi else min(nxt, min(hi) - 1)
            if nxt > a.max_n and a.max_n > N and not hi:
                nxt = a.max_n
            if nxt in pts or nxt > a.max_n or nxt < 1:
                reason = "max-n"
                break
            N = nxt
        lo = [n for n in sorted(pts) if pts[n] >= a.target]
        hi = [n for n in sorted(pts) if pts[n] < a.target]
        print("=== result ===", flush=True)
        if lo and hi:
            x, y = max(lo), min(hi)
            n_int = x + (math.log(a.target) - math.log(pts[x])) / (math.log(pts[y]) - math.log(pts[x])) * (y - x)
            print(f"  GBS L = {L}: bracket N={x} at {pts[x]:.2f}/s, N={y} at {pts[y]:.2f}/s  ->  N* = {n_int:.1f}", flush=True)
            results[L] = ("N*", n_int)
        elif lo:
            print(f"  GBS L = {L}: not bracketed ({reason}): last N = {max(lo)} at {pts[max(lo)]:.2f}/s", flush=True)
            results[L] = (">=", max(lo))
        else:
            print(f"  GBS L = {L}: below target already at N = {min(pts) if pts else a.start}", flush=True)
            results[L] = ("<", min(pts) if pts else a.start)
    print("=== summary ===")
    for L, (kind, n) in results.items():
        print(f"  GBS L={L}: N* {kind if kind != 'N*' else '='} {n:.1f}")
    if a.json:
        json.dump({"args": vars(a), "rows": rows, "results": {str(L): v for L, v in results.items()}}, open(a.json, "w"),
                  indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("count", "check", "frontier"):
        s = sub.add_parser(name)
        s.add_argument("--loops", default="1,2,3,4,6,8")
        s.add_argument("--ratio", type=float, default=2.0)
        s.add_argument("--r", type=float, default=R_HEAD)
        s.add_argument("--eps", type=float, default=1e-4, help="mass left out per sample (1/S, S = 10^4)")
        s.add_argument("--pmax", type=int, default=120)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--json", default=None)
        s.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
    sub.choices["count"].add_argument("--ladder", default=None)
    sub.choices["frontier"].add_argument("--ladder", default=None,
                                         help="one ladder of delays instead of --loops, e.g. 1,5,25 (L = its sum)")
    sub.choices["frontier"].add_argument("--rate-at-max", action="store_true",
                                         help="rate from the largest batch reached alone (P 10-12: S = 10^4 quoting; "
                                              "a lower bound on the rate at S = 10^4 when the ramp stops earlier)")
    sub.choices["count"].add_argument("--n", default="16,64,256,1024")
    sub.choices["check"].add_argument("--n", default="6")
    sub.choices["check"].add_argument("--cap", default="4,6,8")
    sub.choices["check"].add_argument("--shots", type=int, default=4000)
    sub.choices["check"].add_argument("--device", default="cpu")
    f = sub.choices["frontier"]
    import torch
    f.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    f.add_argument("--target", type=float, default=1.0)
    f.add_argument("--start", type=int, default=8)
    f.add_argument("--max-n", type=int, default=4000)
    f.add_argument("--max-jump", type=float, default=2.0)
    f.add_argument("--rel-bracket", type=float, default=0.02, help="stop when the bracket is within this share of N")
    f.add_argument("--budget", type=float, default=120.0)
    f.add_argument("--max-shots", type=int, default=256)
    f.add_argument("--mem-frac", type=float, default=0.4)
    f.add_argument("--csv", default=None)
    a = ap.parse_args()
    {"count": cmd_count, "check": cmd_check, "frontier": cmd_frontier}[a.cmd](a)


if __name__ == "__main__":
    main()
