"""Cost split of the Triton matrix kernel (gaussian/sieve_triton.py): how much of a sieve's matrix part is the power
products (tl.dot), the trace reductions and the path terms, to decide whether an O(q^3) characteristic-polynomial
route (no products, more reductions; docs/GAUSSIAN.md §8) can pay on the GPU.

    python3 bench/bench_sieve_kernel.py --n 12,18,24 --V 1,4 --p-frac 0.5,0.9 --subsets 65536 --csv k.csv

For each (n pairs, V loop vectors, subset size p = n/2 -> Q), random subsets of the right size are drawn and the kernel
is timed (CUDA events, best of --reps) in five variants: all, no paths, no traces, products only, and paths + traces
without products (P stays C). CUDA only. Before 2026-09-24 22:00 "products only" was dead-code eliminated and "no
products" at V > 1 folded (P stayed the identity), so those two columns of run 20260924_214308 and of the v2.8 rows of
run 20260924_221455 are not measurements; the switches now store a result (see the kernel).
"""
import argparse
import csv

import numpy as np
import torch

from photonic_engines.gaussian.sieve_triton import powers_paths, warmup

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="12,18,24")
ap.add_argument("--V", default="1,4")
ap.add_argument("--subsets", type=int, default=65536)
ap.add_argument("--p-frac", default="0.5", help="subset sizes p = round(frac * n) pairs, comma-separated")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--csv", default=None)
args = ap.parse_args()
dev = "cuda"
warmup(dev, 8)
variants = {"all": {}, "no paths": {"DO_PATHS": False}, "no traces": {"DO_TRACE": False},
            "products only": {"DO_PATHS": False, "DO_TRACE": False}, "no products": {"DO_POWER": False}}
rows = []
print(torch.cuda.get_device_name(0), flush=True)
for n, pf in ((int(x), float(f)) for x in args.n.split(",") for f in args.p_frac.split(",")):
    p = max(1, round(pf * n)); q = 2 * p; E = 2 * n + 1
    rng = np.random.default_rng(n)
    C = rng.normal(size=(E, E)) + 1j * rng.normal(size=(E, E)); C[-1] = 0; C[:, -1] = 0
    Ce = torch.tensor(C / np.sqrt(E), device=dev)
    for V in (int(x) for x in args.V.split(",")):
        Y = torch.tensor(rng.normal(size=(E, 2 * V)) + 0j, device=dev); Y[-1] = 0
        pairs = np.stack([rng.choice(n, size=p, replace=False) for _ in range(args.subsets)])
        idx = torch.tensor(np.concatenate([pairs, pairs + n], 1), device=dev)
        times = {}
        for name, sw in variants.items():
            powers_paths(Ce, Y, Y, idx[:16], n, V, **sw); torch.cuda.synchronize()
            best = float("inf")
            for _ in range(args.reps):
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record(); powers_paths(Ce, Y, Y, idx, n, V, **sw); b.record(); torch.cuda.synchronize()
                best = min(best, a.elapsed_time(b) / 1e3)
            times[name] = best
        print(f"  n={n:2d} p={p:2d} q={q:2d} V={V}: " + "  ".join(f"{k} {1e6 * t / args.subsets:7.2f} us" for k, t in times.items())
              + "  (per subset)", flush=True)
        rows.append([n, p, q, V, args.subsets] + [times[k] for k in variants])
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "p", "q", "V", "subsets"] + [f"s_{k.replace(' ', '_')}" for k in variants])
        w.writerows(rows)
