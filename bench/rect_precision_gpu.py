"""fp64 rounding of the SBS rso step kernels at the frontier sizes, on the GPU: a LOWER BOUND from disagreement between
evaluations of the same cofactors that round differently (Supplement C, method 2) -- the templated mu lane at two
threads-per-lane (two summation orders) and mu vs the lambda-form (two algorithms). Complements bench/rect_precision.py
(true error in long double, CPU, small q). Agent H, 2026-10-06 (P: SBS precision row).

    python3 bench/rect_precision_gpu.py --points 47:28,49:30,51:32 --s0 18 --streams 1

A point N:q is one step (N active columns, q placed rows, s0 = N - 1 - q); rows uniform over m = 2N (distinct rows).
Per point: max over streams of max_x |P_a - P_b| / max_x |P_b| for (mu TPL 4, mu TPL 8) and (mu, lambda).
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford import cuda_rect


def instance(N, q, S, seed):
    m = 2 * N
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)))[0].cuda()
    A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(1000 * seed + s))[:N]] for s in range(S)])
    rows = torch.stack([torch.randperm(m, generator=torch.Generator().manual_seed(7 * seed + s))[:q] for s in range(S)]).cuda()
    return A, rows


def rel(a, b):
    return float(((a - b).abs().max(1).values / b.abs().max(1).values).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="47:28,49:30,51:32")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--csv", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "rect_precision_gpu.csv"))
    args = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}", flush=True)
    cuda_rect.warmup()
    cuda_rect.warmup_mu()
    out = []
    for pt in args.points.split(","):
        N, q = (int(x) for x in pt.split(":"))
        s0 = N - 1 - q
        A, rows = instance(N, q, args.streams, seed=N + 100 * q)
        NP = cuda_rect.mu_tier(N, s0)[0]
        res = {}
        t0 = time.time()
        for tpl in (4, 8):
            cuda_rect.set_mu_config({(NP, s0): (tpl, None)})
            try:
                res[tpl] = cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu")
            finally:
                cuda_rect.set_mu_config(None)
        lam = cuda_rect.rect_cofactors_cuda(A, rows, q, form="lambda")
        torch.cuda.synchronize()
        r_tpl, r_lam = rel(res[4], res[8]), rel(res[4], lam)
        print(f"N={N} q={q} s0={s0}: mu TPL4 vs TPL8 {r_tpl:.2e}   mu vs lambda {r_lam:.2e}   ({time.time() - t0:.0f} s)",
              flush=True)
        out.append({"N": N, "q": q, "s0": s0, "streams": args.streams, "mu_tpl4_vs_tpl8": r_tpl, "mu_vs_lambda": r_lam,
                    "seconds": time.time() - t0})
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    print("wrote", args.csv, flush=True)


if __name__ == "__main__":
    main()
