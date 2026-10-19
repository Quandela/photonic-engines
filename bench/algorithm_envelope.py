"""The algorithm envelope on one instance family (docs/ALGORITHMS.md §5, figure F2a, table T1).

Seconds per exact sample of every algorithm of the T1 set on the SAME instances (Haar-random U,
n single photons in modes 0..n-1, m = round(c n), seed = --seed + n as `find_threshold`), on the
same substrate (backend triton for the four chain-rule sums; the production `auto`/cuda path is
recorded as the reference curve), at the shot-ramp discipline of `find_threshold` (ramp 1, 2, 4, ...
shots until a run exceeds budget / 3; sustained rate = samples / seconds over the runs with at
least --min-shots shots). "strong" is strong simulation then sampling: the SLOS build of the whole
distribution (complex64, docs/MEMORY_FRONTIER.md) plus --run-samples inverse-CDF draws, reported
per sample of that run, t_S = (build + draw) / S — the run-of-S statistic of docs/GAUSSIAN.md §2.1
with S = 1e4 by default; it stops at the first n whose table does not fit the device.

    python3 bench/algorithm_envelope.py --algos 2018,2020,input,percof,auto --ratios 2 --ns 20,24,28,32,36,40
    python3 bench/algorithm_envelope.py --algos strong --ratios 1,2 --ns 8,10,12,14,16,18
    python3 bench/algorithm_envelope.py --algos percof --ratios 2 --ns 20,22,24,26,28,30   # the k x sweeps are dear

A point past the budget (one shot longer than --budget seconds) is not measured: the row is
written with shots = 0 and the algorithm's curve stops there, so the sweep never spends more than
--budget per (algo, c, n). CSV: algo, backend, ratio, n, m, samples_per_s, ms_per_sample, seconds,
shots, run_samples, build_s, states. `bench/plot_algorithm_envelope.py` draws F2a from it and
reads n* per algorithm off the fitted curves; the bracketed n* of T1 is `find_threshold --algo`.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--algos", default="2018,2020,input,percof,auto")
ap.add_argument("--ratios", default="2")
ap.add_argument("--ns", default="20,24,28,32,36,40")
ap.add_argument("--budget", type=float, default=120.0,
                help="seconds per (algo, c, n): the ramp stops when a run exceeds budget/3")
ap.add_argument("--max-shots", type=int, default=256)
ap.add_argument("--min-shots", type=int, default=8)
ap.add_argument("--run-samples", type=float, default=1e4, help="strong: samples S of the run whose per-sample cost is reported")
ap.add_argument("--backend", default=None,
                help="chain-rule algos: triton (default, the common substrate) or cuda (2018/2020/percof/auto only)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="results/algorithm_envelope.csv")
args = ap.parse_args()

dev = args.device
algos = args.algos.split(",")
ratios = [float(x) for x in args.ratios.split(",")]
ns = [int(x) for x in args.ns.split(",")]
name = torch.cuda.get_device_name(0) if dev == "cuda" and torch.cuda.is_available() else dev
print(f"{name}  algos={algos}  ratios={ratios}  ns={ns}  budget={args.budget:g} s", flush=True)


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def measure_chain(algo, backend, n, m, U):
    from photonic_engines.clifford import fock_sample
    in_modes = list(range(n))
    Ut = U.to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed + n)

    def run(shots):
        sync(); t = time.perf_counter()
        occ = fock_sample(Ut, in_modes, shots, generator=gen, algo=algo, backend=backend)
        sync()
        return time.perf_counter() - t, occ

    t_w = time.perf_counter()
    dt, _ = run(1)                                                       # warm-up (compiles) — and the budget gate
    warm = time.perf_counter() - t_w
    if dt > args.budget:
        print(f"  {algo:7s} n={n:2d} m={m:3d}: one shot took {dt:.0f} s > budget, stopping this curve", flush=True)
        return None
    shots, spent, tot_shots, tot_time, best, last = 1, 0.0, 0, 0.0, 0.0, 1
    while shots <= args.max_shots:
        dt, occ = run(shots)
        last = shots; spent += dt
        if shots >= args.min_shots:
            tot_shots += shots; tot_time += dt
        best = shots / dt if tot_shots == 0 else tot_shots / tot_time
        if dt > args.budget / 3 or spent > args.budget:
            break
        shots *= 2
    X = occ.cpu().numpy().astype(np.float64)
    err = float(np.abs(X.mean(0) - (np.abs(U.numpy()[:, :n]) ** 2).sum(1)).max())
    print(f"  {algo:7s} n={n:2d} m={m:3d}: {best:9.3f} samples/s  ({1e3 / best:10.2f} ms/sample, up to {last} shots, "
          f"{spent:.0f} s + {warm:.0f} s warm-up, max|dE[n_j]|={err:.3f})", flush=True)
    return best, spent, last, "", ""


def measure_strong(n, m, U):
    from photonic_engines.clifford.variants import strong_then_sample
    S = int(args.run_samples)
    st = {}
    try:
        strong_then_sample(U, list(range(n)), S, generator=torch.Generator().manual_seed(args.seed + n), device=dev, stats=st)
    except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError, ValueError) as e:
        if isinstance(e, ValueError) and "packed keys" in str(e):           # the torch basis builder's 63-bit key limit
            print(f"  strong  n={n:2d} m={m:3d}: basis builder limit ({e}), stopping this curve", flush=True)
            return None
        if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError):
            raise
        print(f"  strong  n={n:2d} m={m:3d}: table does not fit ({type(e).__name__}), stopping this curve", flush=True)
        return None
    t = (st["build_s"] + st["draw_s"]) / S
    print(f"  strong  n={n:2d} m={m:3d}: {1 / t:9.3f} samples/s of a run of {S} ({1e3 * t:10.4f} ms/sample: build {st['build_s']:.2f} s "
          f"over {st['states']:,} states + draws {st['draw_s']:.2f} s)", flush=True)
    return 1 / t, st["build_s"] + st["draw_s"], S, round(st["build_s"], 4), st["states"]


os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
rows = []
for c in ratios:
    for algo in algos:
        backend = "torch" if algo == "strong" else (args.backend or "triton")
        if algo == "auto" and args.backend is None:
            backend = "cuda"                                            # the production frontier curve
        for n in ns:
            m = round(c * n)
            if algo == "strong" and math.comb(m + n - 1, n) > 4e10:
                print(f"  strong  n={n:2d} m={m:3d}: {math.comb(m + n - 1, n):.2e} states, not attempted", flush=True)
                r = None
            else:
                U = haar(m, args.seed + n)
                r = measure_strong(n, m, U) if algo == "strong" else measure_chain(algo, backend, n, m, U)
            if r is None:
                rows.append([algo, backend, c, n, m, "", "", "", 0, "", "", ""])
                break
            rate, sec, shots, build, states = r
            rows.append([algo, backend, c, n, m, rate, 1e3 / rate, round(sec, 2), shots,
                         int(args.run_samples) if algo == "strong" else "", build, states])
            with open(args.csv, "w", newline="") as f:                   # rewritten after every point
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["algo", "backend", "ratio", "n", "m", "samples_per_s", "ms_per_sample", "seconds", "shots",
                            "run_samples", "build_s", "states"])
                w.writerows(rows)
print("wrote", args.csv)
