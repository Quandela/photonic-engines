"""Continuity check (a) of the SBS validation (bench/sbs_rso_validate.py), kernel side: validate the templated mu lane of
clifford/cuda_rect.py on the device and measure the per-s0 step costs c(s) that predict check (a) (2026-10-05).

    python3 bench/cuda_rect_mu_validate.py --n 34 --s0 0,1,2,3,4,5,6 --streams 64 --tpls 4,8,16 --csv out.csv

1. correctness (small, q <= 21): the templated lane ("mu") against the torch cofactors (`sbs_rso.rect_cofactors`) and the
   generic lane ("mu_generic") at n_check = 22, s0 = 0 .. 6, every TPL of the sweep. Pass: <= 1e-11 relative.
2. c(s): the chain's last step at n = 34 (N = 34 columns, q = 33 - s0 placed rows, rows uniform over m = 2n, 64 streams),
   seconds per call. s0 = 0: the production Fock kernel (form "fock") and the templated lane at D = 0 (its kappa_0, and
   their agreement); s0 = 1 .. 6: the templated lane at each TPL of the sweep (the TPLs must agree: summation orders);
   s0 = 7, 8: the lambda-form (what "auto" dispatches there). A TPL slower than --skip-s seconds ends that s0's sweep.
3. the prediction: kappa_s = c(s) / c(0) / ((s + 1) / 2^s), and t(p) / t_Fock = r0 sum_s Bin(n, 1 - p)(s) c(s) / c(0) at
   p = 0.8, 0.9, 0.95, 0.99, 1, uncalibrated and with the x1.5 residual of run 20261005_125116 on every s0 >= 1 stratum;
   gate (a) = t(0.99) <= 1.3 t_Fock and t(p_i) <= 1.15 t(p_i+1). s0 > 8: c(8) halved per unit of s0.

The torch reference only runs at q <= 21 (never time it above q = 27); no n >= 42, s0 = 0 point (~6 h per call).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford import cuda_rect
from photonic_engines.clifford.sbs_rso import rect_cofactors

P_LIST = (0.8, 0.9, 0.95, 0.99, 1.0)
R0 = 0.0617 / 0.0682        # t_rso(p = 1) / t_Fock, run 20261005_125116 (gate 29, square = kernel)


def instance(n, s0, S, seed):
    m = 2 * n
    q = n - 1 - s0
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)))[0].cuda()
    A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(1000 * seed + s))[:n]] for s in range(S)])
    rng = np.random.default_rng(seed + s0)
    rows = torch.tensor(rng.integers(0, m, size=(S, q))).cuda()
    return A, rows, q


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
    return out, first, (min(ts) if ts else first)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def predict(c, n, cal):
    def cost(s):
        if s in c:
            return c[s]
        top = max(c)
        return c[top] * 2.0 ** -(s - top)
    out = []
    for p in P_LIST:
        tot = 0.0
        for s in range(n + 1):
            w = math.comb(n, s) * (1 - p) ** s * p ** (n - s)
            tot += w * (cost(s) / c[0]) * (cal if s >= 1 else 1.0)
        out.append(R0 * tot)
    mono = all(out[i] <= 1.15 * out[i + 1] for i in range(len(out) - 1))
    return out, mono and out[3] <= 1.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=34)
    ap.add_argument("--s0", default="0,1,2,3,4,5,6")
    ap.add_argument("--lambda-s0", default="7,8")
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--tpls", default="4,8,16")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--skip-s", type=float, default=60.0)
    ap.add_argument("--n-check", type=int, default=22)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    tpls = [int(x) for x in args.tpls.split(",")]
    s0s = [int(x) for x in args.s0.split(",")]
    rows_out = []
    t_start = time.time()
    print(f"device {torch.cuda.get_device_name()}  cuda_rect MU_AUTO_MAX={cuda_rect.MU_AUTO_MAX}", flush=True)
    t0 = time.time()
    cuda_rect.warmup_mu(degrees=range(cuda_rect.MU_DMAX + 1), n_max=max(args.n, args.n_check))
    print(f"warmup_mu (default TPL, D = 0..{cuda_rect.MU_DMAX}): {time.time() - t0:.1f} s", flush=True)

    # 1. correctness at small q
    ok_all = True
    nc = args.n_check
    for s0 in s0s:
        A, rows, q = instance(nc, s0, 16, seed=7)
        B = torch.stack([A[s][rows[s]] for s in range(A.shape[0])])
        ref = rect_cofactors(B, loo="prefix")
        gen = cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu_generic")
        errs = []
        NP = cuda_rect.mu_tier(nc, s0)[0]
        for tpl in tpls:
            cuda_rect.set_mu_config({(NP, s0): (tpl, None)})
            got = cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu")
            errs.append(rel(got, ref))
        cuda_rect.set_mu_config(None)
        e_gen = rel(gen, ref)
        ok = max(errs) <= 1e-11 and e_gen <= 1e-11
        ok_all &= ok
        print(f"check n={nc} s0={s0} q={q}: mu vs torch {' '.join(f'{e:.1e}' for e in errs)} (TPL {tpls})  "
              f"generic vs torch {e_gen:.1e}  {'ok' if ok else 'FAIL'}", flush=True)
        rows_out.append({"part": "check", "n": nc, "s0": s0, "q": q, "form": "mu", "tpl": "/".join(map(str, tpls)),
                         "err": max(errs), "err_generic": e_gen, "ok": ok})
    print(f"CHECK {'GREEN' if ok_all else 'RED'}", flush=True)

    # 2. c(s) at n
    n, S = args.n, args.streams
    c, best_tpl = {}, {}
    for s0 in s0s:
        A, rows, q = instance(n, s0, S, seed=n)
        NP = cuda_rect.mu_tier(n, s0)[0]
        units = float(cuda_rect._plan(torch.view_as_real(A.contiguous()), rows, q, 1 << 17)[3].sum()) * n
        if s0 == 0:
            ref, first, dt = timed(lambda A=A, rows=rows, q=q: cuda_rect.rect_cofactors_cuda(A, rows, q, form="fock"), args.reps)
            c[0] = dt
            print(f"s0=0 q={q}: Fock kernel {dt:8.2f} s  ({units / dt / 1e9:.2f} G col-updates/s)", flush=True)
            rows_out.append({"part": "cost", "n": n, "s0": 0, "q": q, "form": "fock", "tpl": "",
                             "seconds": dt, "units": units})
        else:
            ref = None
        res = {}
        for tpl in tpls:
            cuda_rect.set_mu_config({(NP, s0): (tpl, None)})
            got, first, dt = timed(lambda A=A, rows=rows, q=q: cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu"), args.reps)
            if ref is None:
                ref = got
            e = rel(got, ref)
            res[tpl] = dt
            ideal = (s0 + 1) / 2 ** s0
            kap = dt / c[0] / ideal
            print(f"s0={s0} q={q} TPL={tpl:2d}: mu {dt:8.2f} s  (first {first:7.2f} s)  {units / dt / 1e9:6.2f} G col-updates/s  "
                  f"c/c0 {dt / c[0]:6.3f}  kappa {kap:5.2f}  {'vs Fock' if s0 == 0 else 'vs first TPL'} {e:.1e}", flush=True)
            rows_out.append({"part": "cost", "n": n, "s0": s0, "q": q, "form": "mu", "tpl": tpl, "seconds": dt, "units": units,
                             "kappa": kap, "err": e})
            if s0 == 0:
                assert e < 1e-8, ("templated mu D = 0 vs the Fock kernel", e)
            else:
                assert e < 1e-8, ("templated mu: two TPLs disagree", s0, e)
            if dt > args.skip_s:
                print(f"  (> {args.skip_s} s: rest of the TPL sweep skipped)", flush=True)
                break
        cuda_rect.set_mu_config(None)
        if s0 >= 1:
            best_tpl[s0] = min(res, key=res.get)
            c[s0] = res[best_tpl[s0]]
    for s0 in [int(x) for x in args.lambda_s0.split(",") if x]:
        A, rows, q = instance(n, s0, S, seed=n)
        _, first, dt = timed(lambda A=A, rows=rows, q=q: cuda_rect.rect_cofactors_cuda(A, rows, q, form="lambda"), args.reps)
        c[s0] = dt
        print(f"s0={s0} q={q}: lambda {dt:8.2f} s  c/c0 {dt / c[0]:6.3f}", flush=True)
        rows_out.append({"part": "cost", "n": n, "s0": s0, "q": q, "form": "lambda", "tpl": cuda_rect.tier(n)[1], "seconds": dt})

    # 3. prediction
    print("measured TPL optimum per s0:", best_tpl, flush=True)
    for cal in (1.0, 1.5):
        out, green = predict(c, n, cal)
        print(f"PREDICTION x{cal}: t/t_Fock at p = {', '.join(map(str, P_LIST))}: {' / '.join(f'{x:.3f}' for x in out)}  "
              f"-> gate (a) {'GREEN' if green else 'RED'}", flush=True)
        rows_out.append({"part": "prediction", "n": n, "cal": cal, "pred": " / ".join(f"{x:.4f}" for x in out), "ok": green})
    print(f"total {time.time() - t_start:.0f} s", flush=True)
    if args.csv:
        keys = sorted({k for r in rows_out for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows_out)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
