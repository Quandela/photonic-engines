"""Whole post-selected SBS shots per silent-source stratum at chosen n (2026-10-06): the per-shot cost the
rso chain pays on the current cuda_rect dispatch -- run it twice (e.g. PE_MU_R22=0 / 1) to compare two dispatches on the
same instances. Same sampler settings as the SBS validation (bench/sbs_rso_validate.py: backend cuda, square kernel, draw device).

    python3 bench/sbs_rso_strata.py --points 50:14:32,50:18:32,56:18:8 --csv out.csv

A point n:s0:shots draws `shots` streams of k = n - s0 photons from n alpha = pi/4 sources (post-selected: alpha does not
enter), m = 2n, Haar U seeded by n; seconds per shot = best of --reps after a 2-shot warm-up (compiles outside timing).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def haar(m: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return (q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)).to(dev)


def main():
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import cuda_rect, sbs_rso
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="50:14:32,50:18:32")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "sbs_rso_strata.csv"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    kw = {"backend": "cuda", "square": "kernel", "device": dev, "draw": "device"}
    print(f"device {torch.cuda.get_device_name()}  MU_AUTO_MAX {cuda_rect.MU_AUTO_MAX}  MU_AUTO_NMAX {cuda_rect.MU_AUTO_NMAX}"
          f"  tag {args.tag}", flush=True)
    t0 = time.perf_counter()
    cuda_rect.warmup()
    cuda_rect.warmup_mu()
    print(f"warmup: {time.perf_counter() - t0:.1f} s", flush=True)
    out = []
    for pt in args.points.split(","):
        n, s0, S = (int(x) for x in pt.split(":"))
        m, k = 2 * n, n - s0
        U = haar(m, seed=n, dev=dev)
        st = ProductSuperposition.sbs(n, m, alpha=math.pi / 4)
        sbs_rso.sample(st, U, 2, generator=torch.Generator().manual_seed(1), photons=k, **kw)          # warm-up
        torch.cuda.synchronize()
        best = math.inf
        for r in range(args.reps):
            t = time.perf_counter()
            sbs_rso.sample(st, U, S, generator=torch.Generator().manual_seed(100 + r), photons=k, **kw)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        form = cuda_rect.choose_form(k - 1, n)
        print(f"n={n} s0={s0} k={k} shots={S}: {best / S:.5f} s/shot  (last step form {form})", flush=True)
        out.append({"tag": args.tag, "n": n, "s0": s0, "k": k, "shots": S, "sec_per_shot": best / S, "last_form": form,
                    "mu_auto_max": cuda_rect.MU_AUTO_MAX, "mu_auto_nmax": cuda_rect.MU_AUTO_NMAX})
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    print("wrote", args.csv, f"total {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
