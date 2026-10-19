"""Single-sample path: must reproduce the batched sampler (S = 1, same sigma and seeds) exactly,
and its histogram must match the exact distribution at small n."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import time

import numpy as np
import torch
from check_sampler import exact_dist, haar

from photonic_engines.clifford.sampler import fock_sample
from photonic_engines.clifford.single import fock_sample_single

dev = "cuda"
rng = np.random.default_rng(21)
# --- exact agreement with the batched sampler (same sigma, same seeds) at several n
for n, m, trials in [(6, 12, 50), (14, 28, 20), (20, 40, 10), (26, 52, 3)]:
    same = 0
    for t in range(trials):
        U = torch.tensor(haar(m, rng), device=dev)
        g = torch.Generator(device=dev).manual_seed(1000 * n + t)
        sig = torch.argsort(torch.rand(n, generator=g, device=dev))
        seeds = torch.randint(0, 2 ** 31 - 1, (n,), generator=g, device=dev).tolist()
        # batched path with the same sigma/seeds: patch its generator-driven choices by re-seeding identically
        g2 = torch.Generator(device=dev).manual_seed(1000 * n + t)
        occ_b = fock_sample(U, list(range(n)), 1, generator=g2, algo="2020", backend="triton")[0]   # batched path
        occ_s = fock_sample_single(U, list(range(n)), sig=sig, seeds=seeds)
        same += int(torch.equal(occ_b, occ_s))
    print(f"n={n} m={m}: single == batched(S=1) in {same}/{trials} trials")
    assert same == trials
# --- histogram vs exact distribution
for m, in_modes, shots in [(5, [0, 0, 1], 4000), (6, [0, 1, 2, 3], 4000)]:
    U = haar(m, rng); dist = exact_dist(U, in_modes)
    Ut = torch.tensor(U, device=dev)
    keys = list(dist); idx = {k: i for i, k in enumerate(keys)}; counts = np.zeros(len(keys))
    t0 = time.time()
    for s in range(shots):
        occ = tuple(fock_sample(Ut, in_modes, 1, generator=torch.Generator(device=dev).manual_seed(s), backend="cuda")[0].cpu().tolist())
        assert occ in idx, occ
        counts[idx[occ]] += 1
    p = np.array([dist[k] for k in keys]); q = counts / shots
    tvd = 0.5 * abs(p - q).sum(); floor = 0.5 * sum(math.sqrt(x * (1 - x) / shots) for x in p) * math.sqrt(2 / math.pi)
    print(f"m={m} n={len(in_modes)} shots={shots}: TVD={tvd:.4f} floor={floor:.4f}  ({(time.time()-t0)/shots*1e3:.2f} ms/sample)")
    assert tvd < 2 * floor
print("OK")
