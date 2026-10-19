"""
Cost of ONE superposition-boson-sampling amplitude (`slos.glynn.product_amplitude`).

docs/SUPERPOSITION.md §4: the amplitude is a single permanent whose n - k identical constant rows
collapse to a binomial weight, so the sum is O(2^k (n - k) n) — exponential in the DETECTED photon
count k, polynomial in the source count n. This measures the two exponents separately:

    python3 bench/bench_product_amp.py --device cuda --k 10,14,18,22 --sources 2k
    python3 bench/bench_product_amp.py --device cuda --k 16 --sources 20,32,48,64   # the n factor

`--sources 2k` means n = 2k, the alpha = pi/4 point where the mean detected count is n/2. The
pattern is unbunched (one photon in each of the first k output modes) unless `--bunched`.
"""
import argparse
import csv
import math
import time

import torch

from photonic_engines import ProductSuperposition
from photonic_engines.slos.glynn import product_amplitude


def _haar(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)


def analytic_error(sources: int, k: int, m: int, device, dtype, chunk: int | None) -> float:
    """Relative error on U = identity, where the amplitude is cos^(n-k) sin^k exactly at any size.

    The evaluation is a signed sum, so its accuracy has to be measured, not assumed: the first
    version of this path (the homogenised permanent with the constant row collapsed) was exact in
    exact arithmetic and lost every digit by n = 32 (LOG 2026-09-19).
    """
    st = ProductSuperposition.sbs(sources, m)
    occ = [0] * m
    for j in range(k):
        occ[j] = 1
    got = complex(product_amplitude(st, torch.eye(m, dtype=torch.complex128), occ,
                                    device=device, dtype=dtype, chunk=chunk))
    exact = math.cos(math.pi / 4) ** (sources - k) * math.sin(math.pi / 4) ** k
    return abs(got - exact) / exact


def run(sources: int, k: int, device, dtype, repeats: int, bunched: bool, chunk: int | None,
        selfcheck: bool = False) -> dict:
    m = max(2 * sources, k + 2)
    st = ProductSuperposition.sbs(sources, m)
    U = _haar(m, seed=sources * 97 + k)
    occ = [0] * m
    if bunched:
        occ[0] = k
    else:
        for j in range(k):
            occ[j] = 1
    def fn():
        return product_amplitude(st, U, occ, device=device, dtype=dtype, chunk=chunk)

    val = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    ms = times[len(times) // 2]
    err = analytic_error(sources, k, m, device, dtype, chunk) if selfcheck else float("nan")
    print(f"  sources={sources:4d} k={k:3d} m={m:4d}  {ms:10.2f} ms   |amp| = {float(val.abs()):.3e}"
          f"   identity rel.err = {err:.1e}", flush=True)
    if float(val.abs()) > 1.0:
        print("    WARNING: |amp| > 1 for a normalised state -- the sum has lost its digits", flush=True)
    return {"sources": sources, "photons": k, "m": m, "ms": round(ms, 3),
            "amp_abs": float(val.abs()), "identity_rel_err": err}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", default="8,12,16", help="comma-separated detected photon counts")
    ap.add_argument("--sources", default="2k", help='"2k" (alpha = pi/4) or comma-separated counts')
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bunched", action="store_true", help="all k photons in one output mode")
    ap.add_argument("--chunk", type=int, default=0, help="sign vectors per batch (0 = automatic)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="also evaluate the U = identity case, whose amplitude is analytic, and report its error")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    chunk = args.chunk or None
    ks = [int(x) for x in args.k.split(",")]
    print(f"device={device} dtype={args.dtype} bunched={args.bunched} chunk={chunk or 'auto'}")
    rows = []
    if args.sources == "2k":
        for k in ks:
            rows.append(run(2 * k, k, device, dtype, args.repeats, args.bunched, chunk, args.selfcheck))
    else:
        for n in [int(x) for x in args.sources.split(",")]:
            for k in ks:
                rows.append(run(n, k, device, dtype, args.repeats, args.bunched, chunk, args.selfcheck))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
