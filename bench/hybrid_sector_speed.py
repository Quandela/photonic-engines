"""Per-sample cost of the loop sampler on two paths, "default" and "fast", at fixed (N, L) and batch size; plus a
two-sample law check of the two paths and a host-sync count per tick (2026-10-12). The CSV column `fast` is 0 / 1.
Paths by build (the run log's `=== source` line says which): in the first builds default = the v2.11c march, fast =
fixes 1-3; later fast also sends one-loop Fock input to the Fock-chain sampler; since the flag was dropped (10-12)
default = the sector march v3 forced (`_sector_march(chain=False)`) and fast = `loop_sector_sample` as shipped (the
chain where it applies).

Instance = the ORCA loop comparison (bench/orca_loop_bench.py): one loop of delay L, T = 2N bins, N single
photons packed in bins 0..N-1, theta = 2 rand, phi = 6 rand from a generator seeded 0 on the device, shape (1, T).
Per path and point: one untimed warm-up batch of 16 (builds the sector tables), then one timed batch per batch size;
the sector-table time of a timed call (normally 0) is subtracted, as in bench/hybrid_frontier.py.

  python bench/hybrid_sector_speed.py --device cuda --points 96:1,96:6,96:8 --batches 4,64,256 --csv out.csv
  python bench/hybrid_sector_speed.py --device cuda --law 96:8 --law-shots 4096 --syncs 96:8
"""
import argparse
import csv
import time
import warnings

import numpy as np
import torch

from photonic_engines.timeloop import TimeLoopCircuit
from photonic_engines.timeloop.hybrid import _head_program, _sector_march, light_loop_kernel, loop_sector_sample


def algo(lk, fast):
    """The algorithm a path runs on this kernel (CSV column `algo`)."""
    return "chain" if fast and _head_program(lk) is not None else "march"


def draw(lk, shots, th, ph, g, fast, stats=False):
    """fast: `loop_sector_sample` as shipped; else the sector march with the chain disabled (lossless)."""
    if fast:
        return loop_sector_sample(lk, shots, th, ph, generator=g, return_stats=stats)
    pats, _, st = _sector_march(lk, shots, lk._gates(th, ph), generator=g, chain=False)
    pats = pats.reshape(pats.shape[0], lk.T_bins)
    return (pats, st) if stats else pats


def instance(N: int, L: int, dev):
    T = 2 * N
    occ = np.zeros(T, dtype=np.int64)
    occ[:N] = 1
    c = TimeLoopCircuit(T, (L,))
    lk = light_loop_kernel(c, occ, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    th = torch.rand(c.n_loops, T, generator=g, device=dev, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, device=dev, dtype=torch.float64) * 6
    return lk, th, ph


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(lk, th, ph, shots, fast, g, dev):
    sync(dev)
    t0 = time.perf_counter()
    out, st = draw(lk, shots, th, ph, g, fast, stats=True)
    sync(dev)
    return time.perf_counter() - t0 - st["table_seconds"], out, st


def parse_points(s):
    return [tuple(int(v) for v in p.split(":")) for p in s.split(",") if p]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points", default="", help="N:L,... timing points")
    ap.add_argument("--batches", default="4,64,256")
    ap.add_argument("--paths", default="default,fast")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--law", default="", help="N:L,... two-sample law check default vs fast")
    ap.add_argument("--law-shots", type=int, default=4096)
    ap.add_argument("--law-batch", type=int, default=256)
    ap.add_argument("--syncs", default="", help="N:L,... count host syncs per tick (CUDA sync debug mode)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"device {dev} {torch.cuda.get_device_name() if dev.type == 'cuda' else ''} torch {torch.__version__}", flush=True)

    for N, L in parse_points(args.law):
        lk, th, ph = instance(N, L, dev)
        res = {}
        for fast in (False, True):
            g = torch.Generator(device=dev).manual_seed(11 + fast)
            chunks, left = [], args.law_shots
            while left:
                s = min(left, args.law_batch)
                chunks.append(draw(lk, s, th, ph, g, fast).double().cpu())
                left -= s
            res[fast] = torch.cat(chunks).numpy()
        a, b = res[False], res[True]
        z_mean = (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / len(a) + 1e-300)
        pa, pb = (a > 0).mean(0), (b > 0).mean(0)
        z_click = (pa - pb) / np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / len(a) + 1e-300)
        ta, tb = a.sum(1), b.sum(1)
        print(f"LAW N={N} L={L} shots={len(a)} per path: max |z| per-bin mean {np.abs(z_mean).max():.2f}, "
              f"per-bin click {np.abs(z_click).max():.2f} over {a.shape[1]} bins; detected photons "
              f"{ta.mean():.3f} vs {tb.mean():.3f} (z {(ta.mean() - tb.mean()) / np.sqrt((ta.var() + tb.var()) / len(a)):.2f})",
              flush=True)

    for N, L in parse_points(args.syncs):
        if dev.type != "cuda":
            break
        lk, th, ph = instance(N, L, dev)
        for fast in (False, True):
            g = torch.Generator(device=dev).manual_seed(3)
            draw(lk, 64, th, ph, g, fast)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode(1)
                draw(lk, 64, th, ph, g, fast)
                torch.cuda.set_sync_debug_mode(0)
            n = sum("synchroniz" in str(x.message) for x in w)
            print(f"SYNCS N={N} L={L} fast={fast}: {n} host syncs in one 64-shot batch = {n / (2 * N):.1f} per tick",
                  flush=True)

    rows = []
    paths = [p == "fast" for p in args.paths.split(",")]
    for N, L in parse_points(args.points):
        lk, th, ph = instance(N, L, dev)
        for fast in paths:
            g = torch.Generator(device=dev).manual_seed(1)
            t0 = time.perf_counter()
            _, _, st = timed(lk, th, ph, args.warmup, fast, g, dev)
            print(f"  N={N} L={L} fast={fast}: warm-up {time.perf_counter() - t0:.1f} s (tables {st['table_seconds']:.1f} s)",
                  flush=True)
            for s in [int(x) for x in args.batches.split(",")]:
                try:
                    dt, out, st = timed(lk, th, ph, s, fast, g, dev)
                except torch.OutOfMemoryError:
                    print(f"POINT N={N} L={L} fast={fast} batch={s}: out of GPU memory", flush=True)
                    torch.cuda.empty_cache()
                    break
                row = {"N": N, "L": L, "fast": int(fast), "algo": algo(lk, fast), "batch": s, "seconds": dt, "per_sample": dt / s,
                       "max_amplitudes": st["max_amplitudes"], "mean_loop_n": float(np.mean(st["mean_n"])),
                       "photons": float(out.sum(1).double().mean())}
                rows.append(row)
                print(f"POINT N={N} L={L} fast={fast} ({row['algo']}) batch={s}: {dt:.2f} s, {dt / s:.5f} s/sample, "
                      f"max amp {st['max_amplitudes']}, mean loop n {row['mean_loop_n']:.2f}", flush=True)
        del lk
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if args.csv and rows:
            with open(args.csv, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    by = {(r["N"], r["L"], r["batch"], r["fast"]): r["per_sample"] for r in rows}
    for (N, L, s, fast), v in sorted(by.items()):
        if fast and (N, L, s, 0) in by:
            print(f"SPEEDUP N={N} L={L} batch={s}: default {by[(N, L, s, 0)]:.5f} / fast {v:.5f} s/sample = "
                  f"{by[(N, L, s, 0)] / v:.2f}x", flush=True)


if __name__ == "__main__":
    main()
