"""Device-side distribution test: sampled histogram vs exact distribution (independent Ryser),
for both kernels (2018 Glynn/Gray, 2020 mixed-radix Gray with repeated rows)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import time

import numpy as np
import torch
from check_sampler import exact_dist, haar

from photonic_engines.clifford.sampler import fock_sample

dev = "cuda"
rng = np.random.default_rng(11)
CASES = [(6, [0, 0, 1, 2], 200_000, "2018"), (7, list(range(7)), 300_000, "2018"),
         (8, list(range(6)), 300_000, "2018"), (6, [0, 0, 1, 2], 200_000, "2020"),
         (7, list(range(7)), 300_000, "2020"), (8, list(range(6)), 300_000, "2020")]
for m, in_modes, shots, algo in CASES:
    U = haar(m, rng)
    dist = exact_dist(U, in_modes)
    gen = torch.Generator(device=dev).manual_seed(0)
    torch.cuda.synchronize(); t = time.time()
    occ = fock_sample(torch.tensor(U, device=dev), in_modes, shots, generator=gen, algo=algo).cpu().numpy()
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
    print(f"m={m} n={len(in_modes)} in={in_modes} algo={algo} outcomes={len(keys)} shots={shots}: TVD={tvd:.4f} floor={floor:.4f} chi2 "
          f"z={z:+.2f} ({dt:.2f}s on GPU)")
    assert tvd < 2 * floor and abs(z) < 4
print("OK")
