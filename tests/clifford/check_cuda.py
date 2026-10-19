"""CUDA thread-per-lane 2020 kernel: vs Ryser (small k, all collision patterns) and vs the
Triton 2020 kernel at large k (random collision patterns), then the sampler end-to-end."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import time

import numpy as np
import torch
from check_cofactors import ref_cofactors
from check_sampler import exact_dist, haar

from photonic_engines.clifford.cofactors_triton import laplace_cofactors_rows
from photonic_engines.clifford.sampler import fock_sample

DEV = "cuda"
rng = np.random.default_rng(9)
# --- small k vs Ryser, same patterns as test_multi
cases = [(3, [[0, 0, 0], [1, 1, 0], [2, 1, 0]]),
         (5, [[0]*5, [1, 1, 0, 0, 0], [2, 2, 1, 1, 0], [4, 3, 2, 1, 0], [1, 1, 1, 1, 0]]),
         (9, [[0, 0, 1, 1, 2, 2, 3, 3, 4], [8, 7, 6, 5, 4, 3, 2, 1, 0], [1]*9, [0, 0, 0, 0, 1, 1, 1, 2, 3]])]
for K1, pats in cases:
    K = K1 + 1; S = len(pats); m = 9
    A = rng.normal(size=(S, m, K)) + 1j * rng.normal(size=(S, m, K))
    rows = np.array(pats, dtype=np.int64)
    for ttl in [1 << 20, 4, 1]:
        P = laplace_cofactors_rows(torch.tensor(A, device=DEV), torch.tensor(rows, device=DEV),
                                   target_total_lanes=ttl, backend="cuda").cpu().numpy()
        R = np.stack([ref_cofactors(A[s][rows[s]]) for s in range(S)])
        err = np.abs(P - R).max() / np.abs(R).max()
        print(f"K1={K1} target_lanes={ttl:7d}: rel.err vs Ryser={err:.2e}")
        assert err < 1e-10
# --- large k vs Triton 2020 kernel, random collision patterns (m = 2n-ish)
for K1 in [12, 16, 20, 23, 27]:
    K = K1 + 1; S = 64; m = 2 * K
    A = rng.normal(size=(S, m, K)) + 1j * rng.normal(size=(S, m, K))
    rows = rng.integers(0, m, size=(S, K1))
    At, rt = torch.tensor(A, device=DEV), torch.tensor(rows, device=DEV)
    torch.cuda.synchronize(); t = time.time()
    Pc = laplace_cofactors_rows(At, rt, backend="cuda")
    torch.cuda.synchronize(); tc = time.time() - t
    t = time.time(); Pt = laplace_cofactors_rows(At, rt, backend="triton"); torch.cuda.synchronize(); tt = time.time() - t
    err = (Pc - Pt).abs().max().item() / Pt.abs().max().item()
    print(f"K1={K1:2d} S={S}: cuda vs triton rel.diff={err:.2e}   cuda {tc*1e3:8.1f} ms  triton {tt*1e3:8.1f} ms")
    assert err < 1e-8
# --- 48 / 64 tiers (4 threads per lane) on heavily collided blocks (few terms) vs Triton 2020
for K1, m in [(36, 12), (50, 10)]:
    K = K1 + 1; S = 16
    A = rng.normal(size=(S, m, K)) + 1j * rng.normal(size=(S, m, K))
    rows = rng.integers(0, m, size=(S, K1))
    At, rt = torch.tensor(A, device=DEV), torch.tensor(rows, device=DEV)
    Pc = laplace_cofactors_rows(At, rt, backend="cuda"); Pt = laplace_cofactors_rows(At, rt, backend="triton")
    err = (Pc - Pt).abs().max().item() / Pt.abs().max().item()
    print(f"K1={K1:2d} m={m} (heavy collisions): cuda vs triton rel.diff={err:.2e}")
    assert err < 1e-8
# --- sampler end-to-end (cuda backend forced on every step)
for m, in_modes, shots in [(6, [0, 0, 1, 2], 200_000), (7, list(range(7)), 300_000)]:
    U = haar(m, rng); dist = exact_dist(U, in_modes)
    occ = fock_sample(torch.tensor(U, device=DEV), in_modes, shots, generator=torch.Generator(device=DEV).manual_seed(0),
                          algo="2020", backend="cuda").cpu().numpy()
    keys = list(dist); idx = {k: i for i, k in enumerate(keys)}; counts = np.zeros(len(keys))
    for row in occ:
        k = tuple(row); assert k in idx; counts[idx[k]] += 1
    p = np.array([dist[k] for k in keys]); q = counts / shots
    tvd = 0.5 * abs(p - q).sum(); floor = 0.5 * sum(math.sqrt(x * (1 - x) / shots) for x in p) * math.sqrt(2 / math.pi)
    print(f"sampler cuda m={m} n={len(in_modes)}: TVD={tvd:.4f} floor={floor:.4f}")
    assert tvd < 2 * floor
print("OK")
