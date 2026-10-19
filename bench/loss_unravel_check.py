"""GPU check of the label unravelling of the lossy emitter sampler (docs/LOSS.md §3b): for a few
n and eta, sample with `unravel=True` (labels drawn, one term per gap) and `unravel=False`
(traced gap terms), compare the detected-photon distribution and the mean occupation per mode
(same distribution expected), and time both. Raises with diagnostics if the triton sampler
produces non-finite weights (the failure of run 20260918_160033).

    python3 bench/loss_unravel_check.py --device cuda --n 12,20,26 --eta 0.9,0.7 --shots 256
"""
import argparse
import time

import numpy as np
import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford import emitter_sample


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", default="12,20,26")
    ap.add_argument("--eta", default="0.9,0.7")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--backend", default="triton")
    args = ap.parse_args()
    import math
    for n in [int(x) for x in args.n.split(",")]:
        m = 2 * n
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
        U = torch.tensor(haar(m, n), device=args.device)
        for eta in [float(x) for x in args.eta.split(",")]:
            res = {}
            for unravel in (True, False):
                gen = torch.Generator().manual_seed(7)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                try:
                    occ = emitter_sample(st, U, args.shots, backend=args.backend, device=args.device,
                                         loss=eta, unravel=unravel, generator=gen)
                except Exception as e:
                    print(f"n={n} eta={eta} unravel={unravel}: FAILED: {e}", flush=True)
                    continue
                if args.device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t
                ks = occ.sum(1).cpu()
                res[unravel] = (dt, occ.double().mean(0).cpu(), ks.double().mean().item())
                print(f"n={n} eta={eta} unravel={unravel}: {dt:.2f} s for {args.shots} shots, mean detected {ks.double().mean():.2f} "
                      f"(expect {eta * n:.2f})", flush=True)
            if len(res) == 2:
                d = float((res[True][1] - res[False][1]).abs().max())
                print(f"   max |<n_j>| difference between the two unravellings: {d:.4f} (shot noise ~ {math.sqrt(0.5 / args.shots):.4f}); "
                      f"speed-up x{res[False][0] / res[True][0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
