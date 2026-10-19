"""EmitterSLOSKernel vs the Fock FockSLOSKernel: time and peak memory per (n, m) on one device.

    python3 bench/bench_emitter_slos.py --device cuda --n 6,8,10,12 [--ratio 2] [--dtype complex64]

Writes a CSV to $RESULTS_DIR (or results/).
"""
import argparse
import csv
import math
import os
import time

import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos import EmitterSLOSKernel, FockSLOSKernel


def timed(fn, device, repeats=3):
    """(ms per call, resident MiB before the call, peak MiB during the calls)."""
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated() / 2 ** 20
        torch.cuda.reset_peak_memory_stats()
    else:
        base = float("nan")
    t = time.time()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t) / repeats
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else float("nan")
    return dt * 1e3, base, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", default="6,8,10,12")
    ap.add_argument("--ratio", type=float, default=2.0, help="m = ratio * n")
    ap.add_argument("--dtype", default="complex64", choices=["complex64", "complex128"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "bench_emitter_slos.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for n in [int(x) for x in args.n.split(",")]:
        m = round(args.ratio * n)
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
        g = torch.Generator(device="cpu").manual_seed(n)
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=g))[0].to(args.device, dtype)
        from photonic_engines.slos.basis import clear_layers_cache, device_bytes, get_layers
        if args.device == "cuda":
            import gc
            gc.collect(); torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 2 ** 20
            print(f"  [mem] before basis {mem_before:.0f} MiB", flush=True)
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); layers = get_layers(n, m, "fock", device=args.device); build_s = time.time() - t0
        basis_mib = device_bytes(layers) / 2 ** 20 if args.device == "cuda" else float("nan")
        if args.device == "cuda":
            print(f"  [mem] after basis {torch.cuda.memory_allocated() / 2 ** 20:.0f} MiB (basis tensors {basis_mib:.0f} MiB), "
                  f"build peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)

        def run(make, fn_name="probs"):
            try:
                k = make()
                with torch.no_grad():
                    t, base, mem = timed(lambda k=k: getattr(k, fn_name)(U), args.device)
                    mass = float(getattr(k, fn_name)(U).sum())
                del k
                return t, base, mem, mass
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return float("nan"), float("nan"), float("nan"), float("nan")

        te, be, me, mass = run(lambda: EmitterSLOSKernel(st, device=args.device, dtype=dtype))
        tf, bf, mf, _ = run(lambda: FockSLOSKernel(n, m, [1, 0] * n + [0] * (m - 2 * n), device=args.device, dtype=dtype))
        S = layers.output_size()
        row = {"n": n, "m": m, "S": S, "build_s": round(build_s, 1), "basis_MiB": round(basis_mib, 0),
                   "emitter_ms": round(te, 2), "fock_ms": round(tf, 2), "ratio": round(te / tf, 2),
                   "emitter_resident_MiB": round(be, 1), "emitter_peak_MiB": round(me, 1),
                   "fock_resident_MiB": round(bf, 1), "fock_peak_MiB": round(mf, 1),
                   "mass": round(mass, 6)}   # nan = OOM; resident = basis + kernel arrays before the call
        print(" ".join(f"{a}={b}" for a, b in row.items()), flush=True)
        rows.append(row)
        del layers
        clear_layers_cache()                     # one basis alive at a time
        if args.device == "cuda":
            torch.cuda.empty_cache()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
