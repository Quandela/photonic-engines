import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import os

import numpy as np
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
from photonic_engines.clifford.cofactors_triton import laplace_cofactors

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") != "1" else "cpu"

def per_ryser(M):
    M = np.asarray(M); n = M.shape[0]
    if n == 0: return 1.0+0j
    tot = 0j
    for r in range(1, n+1):
        for cols in itertools.combinations(range(n), r):
            tot += (-1)**r * np.prod(M[:, list(cols)].sum(1))
    return tot * (-1)**n

def ref_cofactors(B):
    K1, K = B.shape
    return np.array([per_ryser(np.delete(B, c, axis=1)) for c in range(K)])

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for K1 in [1, 2, 3, 5, 8, 11]:
        K = K1 + 1; S = 3
        B = rng.normal(size=(S, K1, K)) + 1j*rng.normal(size=(S, K1, K))
        for max_lanes in [1, 4, 1<<16]:
            P = laplace_cofactors(torch.tensor(B, device=DEV), max_lanes=max_lanes, seg=4, target_total_lanes=1<<20).cpu().numpy()
            R = np.stack([ref_cofactors(B[s]) for s in range(S)])
            err = np.abs(P - R).max() / np.abs(R).max()
            print(f"K1={K1:2d} lanes={max_lanes:6d}  rel.err={err:.2e}")
            assert err < 1e-10
    # structured blocks: zero columns (exact-zero handling), ones (per = (K-1)!)
    import math
    for K1 in [3, 6]:
        K = K1 + 1
        B = rng.normal(size=(4, K1, K)) + 1j*rng.normal(size=(4, K1, K))
        B[1][:, 0] = 0                                   # one zero column
        B[2][:, 0] = 0; B[2][:, 2] = 0                   # two zero columns
        B[3][:, :] = 1                                   # all ones
        P = laplace_cofactors(torch.tensor(B, device=DEV), max_lanes=4, seg=4, target_total_lanes=1<<20).cpu().numpy()
        R = np.stack([ref_cofactors(B[s]) for s in range(4)])
        err = np.abs(P - R).max() / np.abs(R).max()
        assert err < 1e-10, (K1, err)
        assert abs(P[3, 0] - math.factorial(K1)) < 1e-9
        print(f"K1={K1:2d} structured (zero columns / ones): rel.err={err:.2e}")
    # device-only: large K (register-pressure tier), default heuristics, vs Ryser where affordable
    if DEV == "cuda":
        for K1 in [12, 16, 20, 24]:
            B = rng.normal(size=(2, K1, K1 + 1)) + 1j*rng.normal(size=(2, K1, K1 + 1))
            P = laplace_cofactors(torch.tensor(B, device=DEV)).cpu().numpy()
            if K1 <= 12:
                R = np.stack([ref_cofactors(B[s]) for s in range(2)])
                print(f"K1={K1:2d} default lanes: rel.err={np.abs(P-R).max()/np.abs(R).max():.2e}")
            else:
                print(f"K1={K1:2d} default lanes: ran, finite={np.isfinite(P).all()}, |P|max={np.abs(P).max():.3e}")
    print("OK")
