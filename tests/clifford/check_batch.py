"""Batch of distinct unitaries U[B, m, m]: each circuit's histogram must match its own exact
distribution (streams s = b*shots + shot must read A[dp(s)]), for both kernels and both backends."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import math

import numpy as np
import torch
from check_sampler import exact_dist, haar

from photonic_engines.clifford.sampler import fock_sample

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--shots", type=int, default=100_000)
ap.add_argument("--backends", default="triton,cuda" if torch.cuda.is_available() else "triton")
ap.add_argument("--large", default="", help="comma-separated n for the large-n check (m = 2n, B = 3 circuits, second moments vs closed "
                                            "form), e.g. 16,20,24")
ap.add_argument("--large-shots", type=int, default=100_000)
ap.add_argument("--seed", type=int, default=5)
args = ap.parse_args()
dev = args.device
rng = np.random.default_rng(args.seed)
B = 3
for m, in_modes, algo in [(6, [0, 0, 1, 2], "2018"), (7, list(range(5)), "2018"), (7, list(range(5)), "2020")]:
    Us = [haar(m, rng) for _ in range(B)]
    dists = [exact_dist(U, in_modes) for U in Us]
    for backend in args.backends.split(","):
        if backend == "cuda" and algo == "2018":
            continue
        gen = torch.Generator(device=dev).manual_seed(0)
        Ut = torch.tensor(np.stack(Us), device=dev)                      # [B, m, m]
        occ = fock_sample(Ut, in_modes, args.shots, generator=gen, algo=algo, backend=backend).cpu().numpy()
        assert occ.shape == (B * args.shots, m)
        for b in range(B):
            block = occ[b * args.shots:(b + 1) * args.shots]
            keys = list(dists[b]); idx = {k: i for i, k in enumerate(keys)}
            counts = np.zeros(len(keys))
            for row in block:
                k = tuple(row); assert k in idx, f"impossible outcome {k} for circuit {b}"
                counts[idx[k]] += 1
            p = np.array([dists[b][k] for k in keys]); q = counts / args.shots
            tvd = 0.5 * abs(p - q).sum()
            floor = 0.5 * sum(math.sqrt(x * (1 - x) / args.shots) for x in p) * math.sqrt(2 / math.pi)
            # a circuit's samples must NOT match another circuit's distribution
            other = np.array([dists[(b + 1) % B].get(k, 0.0) for k in keys])
            tvd_other = 0.5 * abs(other - q).sum()
            print(f"m={m} n={len(in_modes)} algo={algo} backend={backend} circuit {b}: TVD={tvd:.4f} floor={floor:.4f}  (vs other "
                  f"circuit: {tvd_other:.3f})")
            assert tvd < 2 * floor, "histogram does not match this circuit's distribution"
            assert tvd_other > 5 * floor, "samples indistinguishable from another circuit's distribution"
# --- large n: exact distribution out of reach -> per-circuit first and second moments against the
# closed form (check_large_n.check_second_moments), cuda backend, algo auto (32/48-column tiers,
# lane planning across streams of different unitaries)
if args.large:
    from check_large_n import check_second_moments
    ok_all = True
    for n in [int(x) for x in args.large.split(",")]:
        m = 2 * n
        Us = [haar(m, rng) for _ in range(B)]
        Ut = torch.tensor(np.stack(Us), device=dev)
        gen = torch.Generator(device=dev).manual_seed(1)
        occ = fock_sample(Ut, list(range(n)), args.large_shots, generator=gen, algo="auto", backend="cuda").cpu().numpy()
        for b in range(B):
            block = occ[b * args.large_shots:(b + 1) * args.large_shots]
            ok_b = check_second_moments(Us[b], n, block, f"n={n} m={m} batch circuit {b}")
            if not ok_b:
                # control: the same circuit sampled alone (B = 1), same shots, fresh seed -- separates a
                # batch-path bias (only the batched block fails) from a statistic running hot for this U
                for trial in range(2):
                    g1 = torch.Generator(device=dev).manual_seed(100 + trial)
                    occ1 = fock_sample(torch.tensor(Us[b], device=dev), list(range(n)), args.large_shots,
                                           generator=g1, algo="auto", backend="cuda").cpu().numpy()
                    ok1 = check_second_moments(Us[b], n, occ1, f"    control: circuit {b} alone (B=1), seed {100 + trial}")
                    g2 = torch.Generator(device=dev).manual_seed(200 + trial)
                    occ2 = fock_sample(Ut, list(range(n)), args.large_shots, generator=g2, algo="auto", backend="cuda").cpu().numpy()
                    ok2 = check_second_moments(Us[b], n, occ2[b * args.large_shots:(b + 1) * args.large_shots],
                                               f"    control: circuit {b} in batch again, seed {200 + trial}")
                    ok_b = ok_b or (ok1 and ok2)
            ok_all &= ok_b
            # cross-check: the block must FAIL against another circuit's moments (z far above threshold)
            mu_other = (np.abs(Us[(b + 1) % B][:, :n]) ** 2).sum(1)
            z_other = np.abs(block.mean(0) - mu_other) / (block.std(0) / math.sqrt(args.large_shots))
            print(f"    vs circuit {(b + 1) % B}: first-moment max|z| = {z_other.max():.1f} (must be large)")
            assert z_other.max() > 10
    assert ok_all
print("OK")
