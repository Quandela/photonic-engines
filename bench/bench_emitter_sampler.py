"""Emitter samplers (batched torch, Triton kernel) vs the Fock Clifford sampler (Triton/CUDA kernels):
ms per sample and the mean number of pure terms per step.

    python3 bench/bench_emitter_sampler.py --device cuda --n 8,10,12,14 --shots 512

Writes $RESULTS_DIR/bench_emitter_sampler.csv. The Fock comparison uses the same U and n
photons in modes 0, 2, 4, ... (one photon per dual-rail pair) with `fock_sample`
(backend cuda if cupy is present, else triton); skipped on CPU.
"""
import argparse
import csv
import math
import os
import time

import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_sampler import sample_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", default="8,10,12")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--fock-shots", type=int, default=4096)
    ap.add_argument("--impl", default="torch,triton", help="comma list of: torch, triton")
    ap.add_argument("--kernel", default="seq", choices=["seq", "scan"], help="Triton kernel implementation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "bench_emitter_sampler.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    from photonic_engines.clifford import emitter_triton as et
    et.set_impl(args.kernel)
    for n in [int(x) for x in args.n.split(",")]:
        m = round(args.ratio * n)
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
        g = torch.Generator().manual_seed(n)
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=g))[0]
        row = {"n": n, "m": m, "shots": args.shots}
        for impl in args.impl.split(","):
            if impl == "triton":
                from photonic_engines.clifford.emitter_triton import sample_triton as fn
            else:
                fn = sample_torch
            # warm-up (term cache, kernels), then timed
            if impl == "torch" and n >= 16:
                continue                      # the torch oracle exceeds device memory at n >= 16 with thousands of shots
            fn(st, U, min(16, args.shots), generator=torch.Generator().manual_seed(0), device=args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            occ, stats = fn(st, U, args.shots, generator=torch.Generator().manual_seed(1),
                            device=args.device, return_stats=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            te = (time.time() - t) / args.shots * 1e3
            row[f"{impl}_ms_per_sample"] = round(te, 4)
            row["R_mean"] = round(sum(stats["R"]) / n, 2); row["R_max_step"] = round(max(stats["R"]), 1)
            if "kernel_s" in stats:
                tot = max(sum(stats["step_s"]), 1e-9)
                row.update(triton_terms_pct=round(100 * stats["terms_s"] / tot), triton_kernel_pct=round(100 * stats["kernel_s"] / tot),
                           triton_post_pct=round(100 * stats["post_s"] / tot), lanes_last=stats["lanes"][-1])
        te = row.get("triton_ms_per_sample", row.get("torch_ms_per_sample"))
        if args.device == "cuda":
            from photonic_engines.clifford import fock_sample
            backend = "cuda"
            try:
                import cupy  # noqa: F401
            except Exception:
                backend = "triton"
            Uc = U.to("cuda")
            in_modes = [2 * k for k in range(n)]
            gen = torch.Generator(device="cuda").manual_seed(0)
            fock_sample(Uc, in_modes, 64, generator=gen, backend=backend)
            torch.cuda.synchronize(); t = time.time()
            fock_sample(Uc, in_modes, args.fock_shots, generator=gen, backend=backend)
            torch.cuda.synchronize()
            tf = (time.time() - t) / args.fock_shots * 1e3
            row.update(fock_ms_per_sample=round(tf, 4), fock_backend=backend, ratio_last_impl=round(te / tf, 1))
        print(" ".join(f"{a}={b}" for a, b in row.items()), flush=True)
        rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
