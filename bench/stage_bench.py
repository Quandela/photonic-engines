"""Timing of the stage kernel (`slos.stage.StageKernel`, docs/DESIGN.md §2.4): Kraus operators of an emitter chain next
to a batch of Fock inputs, per input and output pattern, on one device.

Two sweeps, one CSV row per point (complex128, median of --repeats calls after two warm-ups, device synchronised):
  --sweep batch   : B = 1, 4, 16, 64, 256, ... carried-in inputs of N photons next to an n-event caterpillar (chi = 2),
                    one batched call vs B single-input kernels called in a loop (the per-input route, B x launches);
  --sweep photons : n events (photons emitted per stage) x N input photons at m = --modes, B = --batch inputs.

    python3 bench/stage_bench.py --device cuda --sweep batch --n 4 --N 2 --modes 16 --batches 1,4,16,64,256 --csv out.csv
    python3 bench/stage_bench.py --device cuda --sweep photons --ns 2,4,6,8 --Ns 0,1,2,4 --modes 16 --batch 16 --csv out.csv

CSV: sweep, device, n, N, m, chi, B, S (output patterns of N + n photons), batched_ms (one call, all B inputs),
per_input_ms (B single-input calls; batch sweep only), speedup, ms_per_input, peak_mib (device peak of the batched
call; 0 on the CPU), note. The CSV is rewritten after every point. A point whose estimated device memory exceeds
--max-gib is skipped (a row with the note); a CUDA out-of-memory is caught and recorded. A point slower than --max-s
ends the batch sweep (larger B are only slower) and, in the photons sweep, skips the larger N of the same n.
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from photonic_engines import EmitterMPS, StageChain, spin_gate
from photonic_engines.slos.stage import StageKernel

C128 = torch.complex128


def haar(m, seed):
    """Haar-random unitary [m, m] complex128."""
    z = np.random.default_rng(seed).normal(size=(m, m, 2)) @ [1, 1j]
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / abs(np.diag(r))), dtype=C128)


def chain(n, m):
    """n-event caterpillar (chi = 2) on modes 0 .. 2n-1 of m."""
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.37 * i) for i in range(n)], m=m,
                                modes=[(2 * b, 2 * b + 1) for b in range(n)])
    return StageChain.from_mps(st)


def inputs_of(B, N, n, m, seed):
    """B random N-photon occupations on the modes the chain does not use (2n .. m-1)."""
    rng = np.random.default_rng(seed)
    free = m - 2 * n
    occ = np.zeros((B, m), dtype=np.int64)
    for b in range(B):
        for p in rng.integers(0, free, size=N):
            occ[b, 2 * n + p] += 1
    return occ


def timed(fn, dev, repeats):
    """Median wall time (s) of `repeats` calls after two warm-ups."""
    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
    fn(); fn(); sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def peak_mib(fn, dev):
    if not dev.startswith("cuda"):
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", choices=["batch", "photons"], required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--ns", default="2,4,6,8")
    ap.add_argument("--Ns", default="0,1,2,4")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--batches", default="1,4,16,64,256")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--max-gib", type=float, default=40.0, help="skip a point whose estimated device memory is above")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    dev, m = args.device, args.modes
    U = haar(m, 1).to(dev)
    rows = []
    if args.sweep == "batch":
        points = [(args.n, args.N, int(B)) for B in args.batches.split(",")]
    else:
        points = [(int(n), int(N), args.batch) for n in args.ns.split(",") for N in args.Ns.split(",")]
    fields = ["sweep", "device", "n", "N", "m", "chi", "B", "S", "batched_ms", "per_input_ms", "speedup",
              "ms_per_input", "peak_mib", "note"]
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    def save():
        """Rewrite the CSV after every point, so a timeout or a crash keeps the rows already measured."""
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    t_start = time.time()
    slow_n = None                                    # photons sweep: a slow point skips the larger N of the same n
    for n, N, B in points:
        if 2 * n > m or (N > 0 and 2 * n == m):
            print(f"skip n={n} N={N}: m = {m} too small", flush=True)
            continue
        if slow_n == n:
            print(f"skip n={n} N={N}: a smaller N of this n was over --max-s", flush=True)
            continue
        S = math.comb(m + N + n - 1, N + n)
        est_gib = 3 * S * B * 4 * 16 / 2 ** 30       # result + two working layers, chi = 2 (Q = 2B, chi x chi blocks)
        base = {"sweep": args.sweep, "device": dev, "n": n, "N": N, "m": m, "chi": 2, "B": B, "S": S}
        if est_gib > args.max_gib:
            print(f"skip n={n} N={N} B={B}: ~{est_gib:.0f} GiB > --max-gib {args.max_gib}", flush=True)
            rows.append({**base, "note": f"skipped, ~{est_gib:.0f} GiB"})
            save()
            continue
        ch = chain(n, m)
        occ = inputs_of(B, N, n, m, seed=1000 * n + N)
        try:
            sk = StageKernel(ch, inputs=occ, device=dev)
            t_b = timed(lambda: sk.kraus(U), dev, args.repeats)
            mem = peak_mib(lambda: sk.kraus(U), dev)
            t_p = float("nan")
            if args.sweep == "batch":
                singles = [StageKernel(ch, inputs=occ[b:b + 1], device=dev) for b in range(B)]
                t_p = timed(lambda: [s.kraus(U) for s in singles], dev, max(1, args.repeats // 2))
        except torch.cuda.OutOfMemoryError as e:
            print(f"n={n} N={N} B={B}: out of device memory ({str(e).splitlines()[0][:80]})", flush=True)
            rows.append({**base, "note": "out of memory"})
            save()
            sk = singles = None
            torch.cuda.empty_cache()
            if args.sweep == "batch":
                break
            slow_n = n
            continue
        rows.append({**base, "batched_ms": 1e3 * t_b, "per_input_ms": 1e3 * t_p, "speedup": t_p / t_b,
                     "ms_per_input": 1e3 * t_b / B, "peak_mib": mem, "note": ""})
        save()
        print(f"[{time.time() - t_start:7.1f} s] n={n} N={N} B={B} S={S}: batched {1e3 * t_b:.2f} ms "
              f"({1e3 * t_b / B:.3f} ms/input), per-input {1e3 * t_p:.2f} ms, peak {mem:.0f} MiB", flush=True)
        sk = singles = None
        if t_b > args.max_s:
            if args.sweep == "batch":
                print("stop: single call over --max-s (larger B only slower)", flush=True)
                break
            print(f"n={n}: single call over --max-s, larger N of this n skipped", flush=True)
            slow_n = n
    print("saved", args.csv, flush=True)


if __name__ == "__main__":
    main()
