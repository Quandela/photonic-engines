"""Cofactor backends of the rso SBS chain on the GPU: plain Glynn vs the C&C 2020 collapse (torch / Triton / CUDA), pick.

2026-10-03 (implementation parity: same algorithm level, same kernel substrate). Two parts.

A. One cofactor call P[x] = Per_rect(B[:, T \\ x]) at (q placed rows, N active columns), rows drawn uniformly from
   m = 2 N output modes of a Haar unitary (so the repeats are those of an m = 2n run): plain Glynn
   (`sbs_rso.rect_cofactors`, 2^(q-1) signs), the collapsed sum in torch (`rect_cofactors_collapsed_torch`) and the
   Triton kernel (`sbs_rso_triton`) over a grid of launch shapes (BT, num_warps). Error relative to max|P| against the
   collapsed torch value; seconds = median of 3 after a warm-up.
B. Whole post-selected shots (`sbs_rso.sample(photons=k)`, m = 2n), seconds per shot, for the backends "torch" (plain),
   "collapsed" and "triton" x collapse_from, and "cuda" (clifford/cuda_rect.py, the CUDA port) when cupy is there.

The Triton launch shape is the fastest one whose part-A error stays below 1e-8 at every point (two fp64 summation
orders differ by ~2e-10 at q = 30: check_cuda_rect.py); the pick is the fastest part-B configuration (summed
over the part-B points). It is written as a shell snippet (`--choice`) that the frontier
runs source:

    RSO_KW="--rso-backend triton --rso-collapse-from 12"; export PE_RSO_BT=16 PE_RSO_WARPS=4

    python3 bench/sbs_rso_kernels.py --device cuda --csv out.csv --choice choice.sh
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
import traceback

import torch


def haar(m: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return (q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)).to(dev)


def timed(fn, dev, reps: int = 3):
    out = fn()                                                     # warm-up (and Triton compile)
    ts = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)


def main():
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import sbs_rso
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points", default="12:24,16:28,20:32,23:36,25:40", help="part A, q:N list")
    ap.add_argument("--glynn-max-q", type=int, default=25)
    ap.add_argument("--bt", default="4,8,16,32")
    ap.add_argument("--warps", default="2,4,8")
    ap.add_argument("--shots-at", default="36:18:8,44:22:4", help="part B, n:k:shots list (m = 2n)")
    ap.add_argument("--from", dest="cfrom", default="8,12,16")
    ap.add_argument("--csv", default="")
    ap.add_argument("--choice", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows_out = []
    tri = None
    if dev.type == "cuda":
        try:
            from photonic_engines.clifford.sbs_rso_triton import rect_cofactors_collapsed_triton as tri
        except Exception:  # report and fall back to torch
            traceback.print_exc()
    shapes = [(bt, w) for bt in map(int, args.bt.split(",")) for w in map(int, args.warps.split(","))]
    bad_shapes = set()
    best_shape_time = {}

    def rec(**kw):
        rows_out.append(kw)
        print("  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)

    # ---- part A
    for pt in args.points.split(","):
        q, N = map(int, pt.split(":"))
        U = haar(2 * N, seed=q * 100 + N, dev=dev)
        r = torch.randint(0, 2 * N, (q,), generator=torch.Generator().manual_seed(q))
        B = U[r.to(dev)][:, :N].contiguous()
        uq, mu = torch.unique(r, return_counts=True)
        b, mu = U[uq.to(dev)][:, :N].contiguous(), [int(x) for x in mu.tolist()]
        ref, t_t = timed(lambda b=b, mu=mu: sbs_rso.rect_cofactors_collapsed_torch(b, mu), dev)
        scale = float(ref.abs().max())
        rec(part="A", q=q, N=N, D=len(mu), kernel="collapsed", bt=0, warps=0, collapse_from=0, seconds=t_t, rel_err=0.0)
        if q <= args.glynn_max_q:
            got, t_g = timed(lambda B=B: sbs_rso.rect_cofactors(B.unsqueeze(0))[0], dev)
            rec(part="A", q=q, N=N, D=len(mu), kernel="torch", bt=0, warps=0, collapse_from=0, seconds=t_g,
                rel_err=float((got - ref).abs().max()) / scale)
        if tri is not None:
            for bt, w in shapes:
                if (bt, w) in bad_shapes:
                    continue
                try:
                    got, t_r = timed(lambda b=b, mu=mu, bt=bt, w=w: tri(b, mu, BT=bt, num_warps=w), dev)
                    err = float((got - ref).abs().max()) / scale
                except Exception:
                    traceback.print_exc()
                    bad_shapes.add((bt, w))
                    continue
                if not err < 1e-8:
                    bad_shapes.add((bt, w))
                best_shape_time[(bt, w)] = best_shape_time.get((bt, w), 0.0) + t_r
                rec(part="A", q=q, N=N, D=len(mu), kernel="triton", bt=bt, warps=w, collapse_from=0, seconds=t_r, rel_err=err)
    good = {s: t for s, t in best_shape_time.items() if s not in bad_shapes}
    shape = min(good, key=good.get) if good else None
    print(f"triton launch shape: {shape}", flush=True)

    # ---- part B
    if shape is not None:
        os.environ["PE_RSO_BT"], os.environ["PE_RSO_WARPS"] = str(shape[0]), str(shape[1])
    configs = [("torch", 0)] + [("collapsed", int(f)) for f in args.cfrom.split(",")]
    if shape is not None:
        configs += [("triton", int(f)) for f in args.cfrom.split(",")]
    if dev.type == "cuda":
        try:
            from photonic_engines.clifford import cuda_rect
            if cuda_rect.available():
                configs.append(("cuda", 0))
        except Exception:
            traceback.print_exc()
    totals = dict.fromkeys(configs, 0.0)
    for spec in args.shots_at.split(","):
        n, k, shots = map(int, spec.split(":"))
        st = ProductSuperposition.sbs(n, 2 * n)
        U = haar(2 * n, seed=n, dev=dev)
        for kern, cf in configs:
            try:
                kw = {"backend": kern, "collapse_from": max(cf, 1)}
                sbs_rso.sample(st, U, 1, generator=torch.Generator().manual_seed(1), device=dev, photons=min(k, 13), **kw)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                sbs_rso.sample(st, U, shots, generator=torch.Generator().manual_seed(2), device=dev, photons=k, **kw)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                sec = (time.perf_counter() - t0) / shots
            except Exception:
                traceback.print_exc()
                sec = float("inf")
            totals[(kern, cf)] += sec
            rec(part="B", q=k, N=n, D=0, kernel=kern, bt=shape[0] if (kern == "triton") else 0,
                warps=shape[1] if kern == "triton" else 0, collapse_from=cf, seconds=sec, rel_err=0.0)
    pick = min(totals, key=totals.get)
    print(f"pick: {pick}  (summed s/shot {totals[pick]:.3e}; plain torch {totals[('torch', 0)]:.3e})", flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows_out[0]), lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows_out)
    if args.choice:
        kern, cf = pick
        line = f'RSO_KW="--rso-backend {kern}' + (f' --rso-collapse-from {cf}"' if kern in ("collapsed", "triton") else '"')
        if kern == "triton":
            line += f"; export PE_RSO_BT={shape[0]} PE_RSO_WARPS={shape[1]}"
        with open(args.choice, "w") as f:
            f.write(line + "\n")
        print("choice:", line, flush=True)


if __name__ == "__main__":
    main()
