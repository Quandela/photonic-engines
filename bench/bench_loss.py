"""Uniform loss on the GPU samplers (docs/LOSS.md): correctness of the input-side split against
the thinned strong simulation at small n, then the time per sample of the lossy vs lossless
samplers at large n (Fock `fock_sample`, emitter `emitter_sample(backend="triton")`).

    python3 bench/bench_loss.py --device cuda --n 16,20,24 --eta 0.9,0.7,0.5 --shots 4096

Writes $RESULTS_DIR/bench_loss.csv with one row per (sampler, n, eta): seconds, ms per sample,
the group count and the empirical / expected photon-count histogram; and prints the small-n
TVD checks. Expected: lossy time ≈ sum_k P(k) t(k) ≤ lossless time (docs/LOSS.md §3).
"""
import argparse
import csv
import math
import os
import time

import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.imperfections import binomial_pmf, split_shots


def haar(m, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _emp(occ):
    d = {}
    for r in occ.tolist():
        d[tuple(r)] = d.get(tuple(r), 0) + 1.0 / occ.shape[0]
    return d


def _floor(p, shots):
    return 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in p.values())


def check_small(device, shots=20000):
    from photonic_engines.clifford import emitter_sample, fock_sample
    from photonic_engines.slos import EmitterSLOSKernel, FockSLOSKernel
    C128 = torch.complex128
    for n, m, eta in [(3, 6, 0.7), (5, 10, 0.8)]:
        U = haar(m, 3 + n).to(device)
        occ_in = [0] * m
        for j in range(n):
            occ_in[2 * j] = 1
        ref = FockSLOSKernel(n, m, occ_in, dtype=C128, device=device).probs(U, loss=eta).as_dict()
        s = fock_sample(U, [2 * j for j in range(n)], shots, generator=torch.Generator(device=device).manual_seed(0), loss=eta)
        tvd = _tvd(_emp(s.cpu()), ref)
        print(f"fock    n={n} m={m} eta={eta}: TVD {tvd:.4f} (noise floor {_floor(ref, shots):.4f})")
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, 0.3 * j) for j in range(1, n)], m=m)
        ref = EmitterSLOSKernel(st, dtype=C128, device=device).probs(U, loss=eta).as_dict()
        s = emitter_sample(st, U, shots, backend="triton", device=device, loss=eta, generator=torch.Generator().manual_seed(0))
        tvd = _tvd(_emp(s.cpu()), ref)
        print(f"emitter n={n} m={m} eta={eta}: TVD {tvd:.4f} (noise floor {_floor(ref, shots):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", default="16,20,24")
    ap.add_argument("--eta", default="1.0,0.9,0.7,0.5")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--which", default="fock,emitter")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "bench_loss.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not args.no_check:
        check_small(dev)
    from photonic_engines.clifford import emitter_sample, fock_sample
    rows = []
    for n in [int(x) for x in args.n.split(",")]:
        m = round(args.ratio * n)
        U = haar(m, 100 + n).to(dev)
        in_modes = [2 * j for j in range(n)]
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
        for eta in [float(x) for x in args.eta.split(",")]:
            for which in args.which.split(","):
                gen = torch.Generator(device=dev).manual_seed(1) if which == "fock" else torch.Generator().manual_seed(1)
                loss = None if eta >= 1.0 else eta
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.time()
                if which == "fock":
                    occ = fock_sample(U, in_modes, args.shots, generator=gen, loss=loss)
                else:
                    occ = emitter_sample(st, U, args.shots, backend="triton", device=dev, loss=loss, generator=gen)
                torch.cuda.synchronize() if dev == "cuda" else None
                dt = time.time() - t0
                ks = occ.sum(1).bincount(minlength=n + 1).double().cpu() / args.shots
                groups = len(split_shots(n, args.shots, loss, generator=torch.Generator().manual_seed(1))) if loss else 1
                exp = binomial_pmf(n, eta)
                row = {"sampler": which, "n": n, "m": m, "eta": eta, "shots": args.shots, "seconds": round(dt, 3),
                           "ms_per_sample": round(1e3 * dt / args.shots, 4), "groups": groups,
                           "mean_k": round(float((ks * torch.arange(n + 1)).sum()), 3), "expected_k": round(n * eta, 3),
                           "hist_maxdev": round(float((ks - exp).abs().max()), 4)}
                rows.append(row)
                print(row, flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
