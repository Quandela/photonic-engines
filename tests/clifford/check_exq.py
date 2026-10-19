"""exqalibur GPU port (exq_sampler.py): sampled histogram vs exact distribution, then a
first-moment check against the cuda2020 path at n = 16 (same U, distinct seeds)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import time

import numpy as np
import torch
from check_sampler import exact_dist, haar

from photonic_engines.clifford.exq_sampler import exq_sample
from photonic_engines.clifford.sampler import fock_sample

dev = "cuda"
rng = np.random.default_rng(11)
for m, in_modes, shots in [(6, [0, 0, 1, 2], 200_000), (7, list(range(7)), 300_000), (8, list(range(6)), 300_000)]:
    U = haar(m, rng)
    dist = exact_dist(U, in_modes)
    gen = torch.Generator(device=dev).manual_seed(0)
    torch.cuda.synchronize(); t = time.time()
    occ = exq_sample(torch.tensor(U, device=dev), in_modes, shots, generator=gen).cpu().numpy()
    torch.cuda.synchronize(); dt = time.time() - t
    keys = list(dist); idx = {k: i for i, k in enumerate(keys)}
    counts = np.zeros(len(keys))
    for row in occ:
        k = tuple(row); assert k in idx, f"impossible outcome {k}"
        counts[idx[k]] += 1
    p = np.array([dist[k] for k in keys]); q = counts / shots
    tvd = 0.5 * abs(p - q).sum()
    floor = 0.5 * sum(math.sqrt(x * (1 - x) / shots) for x in p) * math.sqrt(2 / math.pi)
    mask = p * shots >= 5
    chi2 = (((counts - p * shots) ** 2 / (p * shots))[mask]).sum(); dof = mask.sum() - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    print(f"m={m} n={len(in_modes)} in={in_modes} exq: outcomes={len(keys)} shots={shots}: TVD={tvd:.4f} floor={floor:.4f} chi2 "
          f"z={z:+.2f} ({dt:.2f}s)")
    assert tvd < 2 * floor and abs(z) < 4

n, m, shots = 16, 32, 4096
U = haar(m, rng); Ut = torch.tensor(U, device=dev); modes = list(range(n))
mu = (np.abs(U[:, :n]) ** 2).sum(1)
a = exq_sample(Ut, modes, shots, generator=torch.Generator(device=dev).manual_seed(1)).cpu().numpy().astype(float)
b = fock_sample(Ut, modes, shots, generator=torch.Generator(device=dev).manual_seed(2), backend="cuda").cpu().numpy().astype(float)
sd = np.sqrt(mu / shots)
za = np.abs(a.mean(0) - mu) / sd; zb = np.abs(b.mean(0) - mu) / sd
print(f"n={n} m={m} shots={shots}: max |z| of E[n_j] vs exact: exq {za.max():.2f}, cuda2020 {zb.max():.2f} (expect < ~4)")
assert za.max() < 5 and zb.max() < 5
print("OK")
