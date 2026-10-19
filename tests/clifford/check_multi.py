"""2020 kernel (repeated rows) vs independent Ryser, incl. segmentation and halving cases."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os

import numpy as np
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
from check_cofactors import ref_cofactors

from photonic_engines.clifford.cofactors_triton import laplace_cofactors_rows, multiplicity_tables

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") != "1" else "cpu"

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    cases = [  # (K1, rows patterns)
        (1, [[0]]),
        (2, [[0, 0], [0, 1]]),
        (3, [[0, 0, 0], [1, 1, 0], [2, 1, 0]]),
        (4, [[0, 0, 0, 0], [1, 1, 0, 0], [3, 2, 1, 0], [2, 2, 1, 1]]),
        (5, [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 1, 1, 0], [4, 3, 2, 1, 0], [1, 1, 1, 1, 0]]),
        (7, [[0]*7, [1, 1, 1, 0, 0, 0, 2], [3, 3, 2, 2, 1, 1, 0], [6, 5, 4, 3, 2, 1, 0], [1, 1, 2, 2, 3, 3, 4]]),
        (9, [[0, 0, 1, 1, 2, 2, 3, 3, 4], [8, 7, 6, 5, 4, 3, 2, 1, 0], [1]*9, [0, 0, 0, 0, 1, 1, 1, 2, 3]]),
    ]
    for K1, pats in cases:
        K = K1 + 1; S = len(pats); m = 9
        A = rng.normal(size=(S, m, K)) + 1j * rng.normal(size=(S, m, K))
        rows = np.array(pats, dtype=np.int64)
        for ttl in [1 << 20, 4, 1]:          # segment lengths: L=1 (many lanes), medium, one lane
            P = laplace_cofactors_rows(torch.tensor(A, device=DEV), torch.tensor(rows, device=DEV),
                                       target_total_lanes=ttl).cpu().numpy()
            R = np.stack([ref_cofactors(A[s][rows[s]]) for s in range(S)])
            err = np.abs(P - R).max() / np.abs(R).max()
            print(f"K1={K1} target_lanes={ttl:7d}: rel.err={err:.2e}")
            assert err < 1e-10, (K1, ttl, err)
        d, mu, rad, N, sc = multiplicity_tables(torch.tensor(rows, device=DEV), K1)
        print("   terms per stream:", N.tolist(), " (2^(K1-1) =", 1 << (K1 - 1), ") scale*2^K1:", (sc * 2 ** K1).tolist())
    print("OK")
