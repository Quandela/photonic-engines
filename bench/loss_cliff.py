"""Diagnostic for the lossy Fock sampler cliffs seen in run 20260917_182540:
1.08 -> 0.007 samples/s between n = 50 and 51 at eta = 0.7 (m = 100 -> 102) and 1.01 -> 0.24
between n = 64 and 65 at eta = 0.5 (m = 128 -> 130), against a smooth lossy factor elsewhere.
Times `fock_sample(loss=eta)` at fixed shot counts across each cliff and, for reference, the
lossless sampler on k = round(eta n) photons in the same m (one lockstep pass, the cost the lossy
run should average over), so that a jump in one column and not the other locates the cause.

    python3 bench/loss_cliff.py --device cuda --cases 0.7:49,50,51,52 0.5:63,64,65,66 --shots 2,8,32
Writes $RESULTS_DIR/loss_cliff.csv.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from photonic_engines.clifford import fock_sample
from photonic_engines.imperfections import split_shots


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--cases", nargs="+", default=["0.7:49,50,51,52", "0.5:63,64,65,66"], help="eta:n1,n2,...")
    ap.add_argument("--shots", default="2,8,32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "loss_cliff.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = []

    def timed(fn):
        torch.cuda.synchronize(); t = time.perf_counter(); occ = fn(); torch.cuda.synchronize()
        return time.perf_counter() - t, occ

    for case in args.cases:
        eta, ns = case.split(":")
        eta = float(eta)
        for n in [int(x) for x in ns.split(",")]:
            m = round(args.ratio * n)
            U = torch.tensor(haar(m, args.seed + n), device=dev)
            k_ref = round(eta * n)
            for shots in [int(x) for x in args.shots.split(",")]:
                gen = torch.Generator(device=dev).manual_seed(args.seed + shots)
                groups = split_shots(n, shots, eta, generator=torch.Generator().manual_seed(args.seed + shots))
                ks = sorted(len(sub) for sub, _ in groups)
                timed(lambda: fock_sample(U, list(range(n)), shots, generator=gen, loss=eta))            # warm-up
                gen = torch.Generator(device=dev).manual_seed(args.seed + shots)
                dt, occ = timed(lambda: fock_sample(U, list(range(n)), shots, generator=gen, loss=eta))
                mean_n = float(occ.sum(1).double().mean())
                gen = torch.Generator(device=dev).manual_seed(args.seed + shots)
                timed(lambda: fock_sample(U, list(range(k_ref)), shots, generator=gen))
                gen = torch.Generator(device=dev).manual_seed(args.seed + shots)
                dt_ref, _ = timed(lambda: fock_sample(U, list(range(k_ref)), shots, generator=gen))
                row = {"eta": eta, "n": n, "m": m, "shots": shots, "lossy_s": round(dt, 3), "ms_per_sample": round(1e3 * dt / shots, 1),
                           "groups": len(groups), "k_min": ks[0], "k_max": ks[-1], "mean_detected": round(mean_n, 2),
                           "k_ref": k_ref, "lossless_kref_s": round(dt_ref, 3),
                           "lossless_kref_ms_per_sample": round(1e3 * dt_ref / shots, 1)}
                rows.append(row); print(row, flush=True)
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
