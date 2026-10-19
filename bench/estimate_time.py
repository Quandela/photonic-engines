"""Cost of the Gurvits-type estimator (photonic_engines.slos.estimate) against n and chi, where the
strong and sampling engines stop: ms per shot, second moment, and the shots needed for a target
additive error on p(t) (docs/ESTIMATE.md).

    python3 bench/estimate_time.py --device cuda --n 10,20,30,40 --chi 2,16,256 --shots 100000
Writes $RESULTS_DIR/estimate_time.csv.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos.estimate import EmitterGurvitsEstimator


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def state(n, chi, rng):
    if chi == 2:                                                   # linear cluster, projected
        gates = [spin_gate(math.pi / 2, 0)] * n
        return EmitterMPS.caterpillar(n, gates, final=[1, 0])
    ops = np.empty((n, 2, chi, chi), dtype=np.complex128)          # random unitary emitter, projected
    for k in range(n):
        z = rng.normal(size=(2 * chi, 2 * chi)) + 1j * rng.normal(size=(2 * chi, 2 * chi))
        V, r = np.linalg.qr(z)
        V = (V * (np.diag(r) / np.abs(np.diag(r)))).reshape(2, chi, 2, chi)
        for l in range(2):
            ops[k, l] = V[l, :, 0, :]
    s0 = np.zeros(chi, complex); s0[0] = 1
    f = rng.normal(size=chi) + 1j * rng.normal(size=chi); f /= np.linalg.norm(f)
    return EmitterMPS(ops=ops, modes=tuple((2 * k, 2 * k + 1) for k in range(n)), s0=s0, m=2 * n, final=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", default="10,20,30,40")
    ap.add_argument("--chi", default="2,16,256")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "estimate_time.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for n in [int(x) for x in args.n.split(",")]:
        m = 2 * n
        U = haar(m, n).to(args.device)
        t = [1, 0] * n                                             # a collision-free output
        for chi in [int(x) for x in args.chi.split(",")]:
            st = state(n, chi, rng)
            for sampling in ("gaussian", "rademacher"):
                est = EmitterGurvitsEstimator(st, sampling=sampling, device=args.device, seed=1)
                est.probability(U, t, shots=min(2000, args.shots), batch=args.batch)        # warm-up
                if args.device == "cuda":
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                e = est.probability(U, t, shots=args.shots, batch=args.batch)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if args.device == "cuda" else None
                row = {"n": n, "m": m, "chi": chi, "sampling": sampling, "shots": args.shots,
                       "ms_per_shot": round(1e3 * dt / args.shots, 5),
                           "p_est": e.value, "stderr": e.stderr, "stderr_bound": e.stderr_bound, "second_moment": float(e.second_moment),
                           "shots_for_err_1e3": int(e.second_moment / 1e-6) if e.second_moment > 0 else None,   # shots for +-1e-3 on p
                           "peak_GB": peak}
                rows.append(row); print(row, flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
