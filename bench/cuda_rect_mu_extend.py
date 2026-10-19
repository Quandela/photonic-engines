"""Extension of the mu lane: the templated mu lane of clifford/cuda_rect.py at D = s0 = 7 .. 16
against the lambda-form that "auto" dispatched there before (2026-10-05), on the same instances.

    python3 bench/cuda_rect_mu_extend.py --points 32:24,35:24,38:24,41:24 --streams 64 --tpls 4,8,16 --csv out.csv

A point N:q is one chain step with N active columns and q placed rows (s0 = N - 1 - q silent sources), rows uniform over
m = 2N, `--streams` streams (a point may carry its own count as N:q:S). Per point: the lambda-form, then the mu lane at
every TPL of the sweep (one NVRTC compile each, outside the timed calls); seconds (best of --reps after a first call),
ps per unit (unit = one collapsed Gray term x one column, `cuda_rect._plan`), mu vs lambda relative difference (both
exact; lambda is the reference here), and the speed-up lambda / best mu. The printed table is the input of the "auto"
crossover rule (MU_AUTO_MAX per tier) and of the Delta n* table of the extension.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford import cuda_rect


def instance(N, q, S, seed, pool=0):
    """pool > 0: rows drawn from the first `pool` output modes only -- the 2020 collapse then shrinks the term count
    (prod (mult + 1) instead of ~2^(q-1)), which reaches large q (wide steps at low s0) in seconds; the per-unit cost is
    the same per-term work (checked against an unpooled point)."""
    m = 2 * N
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)))[0].cuda()
    A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(1000 * seed + s))[:N]] for s in range(S)])
    rows = torch.tensor(np.random.default_rng(seed + q).integers(0, pool or m, size=(S, q))).cuda()
    return A, rows


def timed(fn, reps):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    first = time.time() - t0
    ts = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return out, first, min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="32:24,35:24,38:24,41:24")
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--tpls", default="4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-s", type=float, default=60.0, help="a TPL slower than this (s) ends the point's sweep")
    ap.add_argument("--csv", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "cuda_rect_mu_extend.csv"))
    args = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}  MU_DMAX {cuda_rect.MU_DMAX}  MU_AUTO_MAX {cuda_rect.MU_AUTO_MAX}",
          flush=True)
    t0 = time.time()
    cuda_rect.warmup()
    cuda_rect.warmup_mu()
    print(f"warmup (lambda tiers, mu D = 1..{cuda_rect.MU_AUTO_MAX}): {time.time() - t0:.1f} s", flush=True)
    rows_out = []
    for pt in args.points.split(","):
        f = [int(x) for x in pt.split(":")]
        N, q = f[0], f[1]
        S = f[2] if len(f) > 2 else args.streams
        pool = f[3] if len(f) > 3 else 0
        s0 = N - 1 - q
        A, rows = instance(N, q, S, seed=N, pool=pool)
        units = float(cuda_rect._plan(torch.view_as_real(A.contiguous()), rows, q, 1 << 17)[3].sum()) * N
        lam, first, dt = timed(lambda: cuda_rect.rect_cofactors_cuda(A, rows, q, form="lambda"), args.reps)
        ps_l = dt / units * 1e12
        print(f"N={N} q={q} s0={s0} streams={S} pool={pool or 'm'} units {units:.3e}: lambda {dt:8.3f} s  "
              f"{ps_l:8.1f} ps/unit", flush=True)
        rows_out.append({"N": N, "q": q, "s0": s0, "streams": S, "pool": pool, "form": "lambda", "tpl": cuda_rect.tier(N)[1],
                         "seconds": dt, "units": units, "ps_per_unit": ps_l, "rel_vs_lambda": 0.0})
        best = None
        if s0 <= cuda_rect.MU_DMAX:
            NP = cuda_rect.mu_tier(N, s0)[0]
            for tpl in [int(x) for x in args.tpls.split(",")]:
                cuda_rect.set_mu_config({(NP, s0): (tpl, None)})
                try:
                    got, first, dt = timed(lambda: cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu"), args.reps)
                except Exception as e:
                    print(f"   mu TPL={tpl:2d}: failed ({type(e).__name__}: {str(e)[:160]})", flush=True)
                    continue
                finally:
                    cuda_rect.set_mu_config(None)
                err = float((got - lam).abs().max() / lam.abs().max())
                ps = dt / units * 1e12
                best = ps if best is None else min(best, ps)
                print(f"   mu TPL={tpl:2d}: {dt:8.3f} s (first, incl. compile, {first:6.1f} s)  {ps:8.1f} ps/unit  "
                      f"x{ps_l / ps:5.2f} vs lambda  rel {err:.1e}", flush=True)
                rows_out.append({"N": N, "q": q, "s0": s0, "streams": S, "pool": pool, "form": "mu", "tpl": tpl, "seconds": dt,
                                 "units": units, "ps_per_unit": ps, "rel_vs_lambda": err})
                if dt > args.skip_s:
                    break
        if best is not None:
            print(f"   best mu {best:.1f} ps/unit -> lambda / mu = {ps_l / best:.2f}", flush=True)
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print("wrote", args.csv, f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
