"""Tune the CUDA rectangular cofactor kernel (clifford/cuda_rect.py, the SBS rso chain's step) against the torch
cofactors of sbs_rso, at the chain's frontier step sizes.

    python3 bench/bench_cuda_rect.py --points 42:21:21,42:28:28,42:33:33 --streams 64 --tpls 1,2,4,8 --maxreg 0,128,255

A point n:k:j is step j of a chain with n live sources and k photons: N = n - k + j active columns, q = j - 1 placed
rows (rows drawn uniformly from m = 2n: few repeats, as on the chain). Every (TPL, maxrregcount) of the grid is timed
(3 calls), checked against the torch path, and the best per tier is printed as the `_CFG_BY_ARCH` entry. Work unit =
sum over streams of the collapsed term count x N (column-updates). Writes $RESULTS_DIR/bench_cuda_rect<tag>.csv.
"""
import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford import cuda_rect
from photonic_engines.clifford.cofactors_triton import multiplicity_tables
from photonic_engines.clifford.sbs_rso import rect_cofactors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="42:21:21,42:28:28,42:33:33,48:24:24")
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--tpls", default="1,2,4,8")
    ap.add_argument("--maxreg", default="0,128,255")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--torch-max-q", type=int, default=27,
                    help="skip the torch reference and timing above this many placed rows (576 s per call at q = 27 "
                         "for 64 streams, x2 per row: the q = 32 point would cost 4 x 5 h; run 20261004_065629)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--form", default="lambda", help="lambda | mu (silent-source form, for s0 = N - 1 - q <= DMAX) | auto")
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(1)
    rows_out, table = [], {}
    for pt in args.points.split(","):
        n, k, j = (int(x) for x in pt.split(":"))
        m, S = 2 * n, args.streams
        N, q = n - k + j, j - 1
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].to(dev)
        A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(s))[:N]] for s in range(S)])
        rows = torch.tensor(rng.integers(0, m, size=(S, q))).to(dev)
        Nt = multiplicity_tables(rows, q)[3] if q else torch.ones(S)
        units = float(Nt.sum()) * N
        NP = cuda_rect.tier(N)[0]
        print(f"point n={n} k={k} j={j}: N={N} q={q} (tier {NP}) streams={S} work units={units:.3e}", flush=True)
        dt_t, ref = float("nan"), None
        if args.no_torch or q > args.torch_max_q:
            print(f"  torch rect_cofactors            : skipped ({'--no-torch' if args.no_torch else f'q > {args.torch_max_q}'})",
                  flush=True)
        else:
            B = torch.stack([A[s][rows[s]] for s in range(S)])
            ref = rect_cofactors(B); torch.cuda.synchronize()
            t = time.time()
            for _ in range(args.reps):
                rect_cofactors(B)
            torch.cuda.synchronize(); dt_t = (time.time() - t) / args.reps
            print(f"  torch rect_cofactors            : {dt_t * 1e3:9.1f} ms  {units / dt_t / 1e9:7.2f} G units/s", flush=True)
        results = []
        for tpl, reg in itertools.product([int(x) for x in args.tpls.split(",")], [int(x) for x in args.maxreg.split(",")]):
            label = f"TPL={tpl} maxreg={reg or '-':>4}"
            try:
                cuda_rect.set_config({NP: (tpl, reg or None)})
                t0 = time.time()
                got = cuda_rect.rect_cofactors_cuda(A, rows, q, form=args.form); torch.cuda.synchronize()
                warm = time.time() - t0
                t = time.time()
                for _ in range(args.reps):
                    cuda_rect.rect_cofactors_cuda(A, rows, q, form=args.form)
                torch.cuda.synchronize(); dt = (time.time() - t) / args.reps
                err = float((got - ref).abs().max() / ref.abs().max()) if ref is not None else float("nan")
                results.append((dt, tpl, reg))
                rows_out.append({"n": n, "k": k, "j": j, "N": N, "q": q, "tier": NP, "streams": S, "units": units, "form": args.form,
                                 "tpl": tpl,
                                 "maxreg": reg, "ms": dt * 1e3, "g_units_s": units / dt / 1e9, "ms_torch": dt_t * 1e3,
                                 "speedup_vs_torch": dt_t / dt, "max_rel_err_vs_torch": err})
                print(f"  {label}: {dt * 1e3:9.1f} ms  {units / dt / 1e9:7.2f} G units/s  x{dt_t / dt:6.1f} vs torch  err {err:.1e}"
                      f"   (first call incl. compile {warm:.1f} s)", flush=True)
            except Exception as e:
                print(f"  {label}: failed ({type(e).__name__}: {str(e)[:100]})", flush=True)
        cuda_rect.set_config(None)
        if results:
            dt, tpl, reg = min(results)
            table[NP] = (tpl, reg or None)
            print(f"  best: TPL={tpl} maxreg={reg or 'default'} ({dt * 1e3:.1f} ms, x{dt_t / dt:.1f} vs torch)", flush=True)
    print("_CFG_BY_ARCH entry:", table)
    out = os.path.join(os.environ.get("RESULTS_DIR", "results"), f"bench_cuda_rect{args.tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows_out:
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            wr.writeheader(); wr.writerows(rows_out)
        print("wrote", out)


if __name__ == "__main__":
    main()
