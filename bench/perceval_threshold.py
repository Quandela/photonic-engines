"""n* of Perceval's Clifford & Clifford 2017 back-end (exqalibur) on the T1 instance family and protocol.

2026-10-07. The row sits next to the C&C 2018 reimplementation (32.5 at m = 2n, run 20260930_082501) as a
*public-implementation* row, not a parity row (same algorithm level, same kernel substrate).

Instance: n photons in modes 0..n-1, m = round(c n), U = haar(m, seed + n) — verbatim bench/find_threshold.py l.234-238.
Protocol (find_threshold's measure()): one untimed warm-up sample; shot ramp 1, 2, 4, ... up to --max-shots per batch;
rate = (shots) / (seconds) over the batches with >= --min-shots shots; stop when a batch takes > budget/3 or the total
exceeds the budget (120 s). Search n upward from --start (downward if the start is already below 1/s) until two
consecutive n straddle 1 sample/s; n* = a + ln(r_a) / ln(r_a / r_b).

    python3 bench/perceval_threshold.py --ratio 2 --threads 0 --start 28 --csv out.csv

--threads sets exqalibur's Config.compute_max_thread_count: 0 = all cores (with a WITH_CUDA build: one GPU runner + the
remaining cores, "as shipped"), 1 = a single runner (the GPU runner on a WITH_CUDA build).
"""
import argparse
import csv
import math
import os
import subprocess
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--start", type=int, default=28)
ap.add_argument("--budget", type=float, default=120.0)
ap.add_argument("--min-shots", type=int, default=8)
ap.add_argument("--max-shots", type=int, default=256)
ap.add_argument("--max-n", type=int, default=44)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--points", default="", help="extra n measured after the bracket (C asked for 26, 28, 30-34)")
ap.add_argument("--csv", required=True)
args = ap.parse_args()

import exqalibur as xq
import perceval as pcvl


def haar(m, seed):  # verbatim bench/find_threshold.py l.234-238
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


xq.Config.compute_max_thread_count = args.threads
doc = (xq.Clifford2017.__doc__ or "").lower()
GPU_BUILD = "gpu" in doc  # the WITH_CUDA docstring; the device is confirmed below by nvidia-smi on this pid


def on_gpu():
    """True if nvidia-smi lists this process among the GPU compute apps (exact pid match)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return str(os.getpid()) in {line.strip() for line in out.splitlines()}


DEVICE = ("GPU+CPU" if args.threads != 1 else "GPU") if GPU_BUILD else "CPU"
THREADS = args.threads or os.cpu_count()
PRECISION = "fp32" if xq.Config.use_single_precision else "fp64"
try:
    from importlib.metadata import version as _ver
    _xqv = _ver("exqalibur")
except Exception:
    _xqv = "?"
VERSION = f"perceval {pcvl.__version__}; exqalibur {_xqv}"
IMPL = "exqalibur Clifford2017 (" + ("WITH_CUDA build" if GPU_BUILD else "public CPU build") + ")"
print(f"{IMPL}; {VERSION}; device {DEVICE}; threads {THREADS}; {PRECISION}", flush=True)


def measure(n):
    m = round(args.ratio * n)
    U = haar(m, args.seed + n)
    b = pcvl.BackendFactory.get_backend("CliffordClifford2017")
    b.set_circuit(pcvl.Unitary(pcvl.Matrix(U)))
    b.set_input_state(pcvl.BasicState([1] * n + [0] * (m - n)))
    b.samples(1)  # warm-up, untimed (runners and prepared matrix are built here)
    seen_gpu.append(on_gpu())
    shots, spent, tot_shots, tot_time = 1, 0.0, 0, 0.0
    while shots <= args.max_shots:
        t0 = time.perf_counter()
        b.samples(shots)
        dt = time.perf_counter() - t0
        spent += dt
        if shots >= args.min_shots:
            tot_shots += shots
            tot_time += dt
        if dt > args.budget / 3 or spent > args.budget:
            break
        shots *= 2
    if tot_shots == 0:  # the ramp stopped before reaching min_shots: use the last batch
        tot_shots, tot_time = shots, dt
    rate = tot_shots / tot_time
    print(f"  n={n} m={m}: {rate:.4f} samples/s ({tot_shots} shots in {tot_time:.1f} s)", flush=True)
    return m, rate, tot_time, tot_shots


rows = {}
seen_gpu = []


def point(n):
    if n not in rows:
        rows[n] = measure(n)
    return rows[n][1]


n = args.start
r = point(n)
step = 1 if r >= 1.0 else -1
bracket = None
while True:
    n2 = n + step
    if n2 < 2 or n2 > args.max_n:
        break
    r2 = point(n2)
    if (r - 1.0) * (r2 - 1.0) <= 0:
        bracket = (min(n, n2), max(n, n2))
        break
    n, r = n2, r2
n_star = float("nan")
if bracket is not None:  # n* from the search's own straddling pair, as find_threshold does
    a, b = bracket
    ra, rb = rows[a][1], rows[b][1]
    n_star = a + math.log(ra) / math.log(ra / rb)
print(f"n* = {n_star:.2f} at m = {args.ratio:g} n ({IMPL}, {DEVICE}); bracket {bracket}", flush=True)
for p in [int(v) for v in args.points.split(",") if v.strip()]:  # per-point comparison set (C), after the bracket
    point(p)
GPU_SEEN = any(seen_gpu)
print(f"process seen on the GPU by nvidia-smi: {'YES' if GPU_SEEN else 'no'} ({sum(seen_gpu)}/{len(seen_gpu)} points)",
      flush=True)

with open(args.csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["engine", "ratio", "n", "m", "samples_per_s", "ms_per_sample", "seconds", "shots",
                "impl", "version", "device", "gpu_seen", "threads", "precision", "n_star"])
    for k in sorted(rows):
        m, rate, secs, shots = rows[k]
        w.writerow(["perceval_cc2017", args.ratio, k, m, f"{rate:.6g}", f"{1000 / rate:.6g}", f"{secs:.3f}", shots,
                    IMPL, VERSION, DEVICE, int(GPU_SEEN), THREADS, PRECISION, f"{n_star:.3f}"])
