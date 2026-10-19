"""
Cost of the graded SLOS recurrence (`ProductSLOSKernel`) against the Fock one at the same size.

docs/SUPERPOSITION.md §3 predicts a CONSTANT factor, not a factor n: the graded basis is
C(m + n, n) = (m + n)/m times the n-photon layer (1.5x at m = 2n), and although each of the n
sources touches every populated layer, the work sum_k (n - k + 1) |E_k| is dominated by the last
layers because the layer sizes grow geometrically. This measures both.

    python3 bench/bench_product_slos.py --device cuda --n 4,6,8,10,11 --ratio 2
    python3 bench/bench_product_slos.py --device cpu --n 3,4,5 --repeats 3

Per n: basis build, kernel build, `probs(U)` time and peak memory, for the Fock kernel (one photon
per source mode) and the graded kernel on the alpha = pi/4 SBS state over the same modes, plus the
two state-space sizes. Writes a CSV (`--csv`) with one row per (engine, n).
"""
import argparse
import csv
import math
import time

import torch

from photonic_engines import ProductSuperposition, graded_state_space_size, state_space_size
from photonic_engines.slos import (
    FockSLOSKernel,
    ProductSLOSKernel,
    clear_layers_cache,
    get_layers,
    sample_from_probs,
)


def _haar(m: int, seed: int, device, dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return (q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)).to(device=device, dtype=dtype)


def _peak(device) -> float:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")


def _time(fn, device, repeats: int) -> tuple[float, float]:
    """(median ms, peak GiB) of `fn`, one warm-up call first."""
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2], _peak(device)


def run(n: int, m: int, device, dtype, repeats: int, shots: int = 0) -> list[dict]:
    U = _haar(m, seed=n * 100 + m, device=device, dtype=dtype)
    rows = []
    occ = [1] * n + [0] * (m - n)
    for engine in ("fock", "sbs"):
        clear_layers_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        get_layers(n, m, "fock", device=str(device))
        build_basis = time.perf_counter() - t0
        t0 = time.perf_counter()
        ker = (FockSLOSKernel(n, m, occ, device=str(device), dtype=dtype) if engine == "fock"
               else ProductSLOSKernel(ProductSuperposition.sbs(n, m), device=str(device), dtype=dtype))
        size = state_space_size(n, m) if engine == "fock" else graded_state_space_size(n, m)
        build_kernel = time.perf_counter() - t0
        ms, peak = _time(lambda k=ker: k.probs(U), device, repeats)
        peak = round(peak, 6) if peak == peak else peak          # keep small peaks non-zero
        draw_ms = float("nan")
        if shots:                                    # strong simulation THEN sample: build once, draw cheaply
            dist = ker.probs(U)
            gen = torch.Generator().manual_seed(0)
            t0 = time.perf_counter()
            out = dist.sample(shots, generator=gen) if engine == "sbs" else \
                sample_from_probs(ker.layers, dist, shots, generator=gen)
            if device.type == "cuda":
                torch.cuda.synchronize()
            draw_ms = (time.perf_counter() - t0) * 1e3
            assert out.shape == (shots, m)
            del dist, out
        rows.append({"engine": engine, "n": n, "m": m, "states": size, "basis_s": round(build_basis, 3),
                     "kernel_s": round(build_kernel, 3), "ms": round(ms, 3), "peak_gib": peak,
                     "draw_ms": round(draw_ms, 3), "shots": shots})
        print(f"  {engine:5s} n={n:3d} m={m:3d} states={size:>12,}  basis {build_basis:6.2f} s  "
              f"kernel {build_kernel:6.2f} s  probs {ms:9.2f} ms  peak {peak:6.2f} GiB"
              + (f"  draw {shots} {draw_ms:8.1f} ms ({draw_ms / max(shots, 1) * 1e3:.2f} us/sample)" if shots else ""),
              flush=True)
        del ker
    a, b = rows
    def ratio(key):                                  # peak_gib is nan on CPU, and tiny at small n
        num, den = b[key], a[key]
        ok = den == den and num == num and den != 0   # nan-safe: nan != nan
        return f"{num / den:.2f}x" if ok else "n/a"
    print(f"  ratio graded/Fock: states {ratio('states')}  time {ratio('ms')}  "
          f"memory {ratio('peak_gib')}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="4,6,8,10", help="comma-separated source counts")
    ap.add_argument("--ratio", type=float, default=2.0, help="m / n")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", default="complex64", choices=["complex64", "complex128"])
    ap.add_argument("--shots", type=int, default=0,
                    help="also draw this many samples from the computed distribution (strong sim then sample)")
    ap.add_argument("--csv", default="", help="write the rows here")
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={args.dtype} ratio={args.ratio} repeats={args.repeats}")
    rows = []
    for n in [int(x) for x in args.n.split(",")]:
        try:                                         # one n running out of memory must not lose the rest
            rows += run(n, math.ceil(args.ratio * n), device, dtype, args.repeats, args.shots)
        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
            print(f"  n={n}: SKIPPED -- {type(e).__name__}: {str(e)[:200]}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if not rows:
        raise SystemExit("every point failed")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
