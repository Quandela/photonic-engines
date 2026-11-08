import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import math
import os

import numpy as np
import torch

# The GPU check scripts import exact_dist / haar from here: an unconditional TRITON_INTERPRET=1 would run
# their kernels under the Triton CPU interpreter on a CUDA host too.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
from check_cofactors import per_ryser

from photonic_engines.clifford.sampler import fock_sample

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") != "1" else "cpu"


def haar(m, rng):
    z = rng.normal(size=(m,m)) + 1j*rng.normal(size=(m,m))
    q, r = np.linalg.qr(z); return q * (np.diag(r)/np.abs(np.diag(r)))

def exact_dist(U, in_modes):
    m = U.shape[0]; n = len(in_modes)
    A = U[:, in_modes]
    s_fact = np.prod([math.factorial(c) for c in np.bincount(in_modes, minlength=m)])
    dist = {}
    for t in itertools.combinations_with_replacement(range(m), n):
        occ = np.bincount(t, minlength=m)
        At = A[list(t), :]
        p = abs(per_ryser(At))**2 / (s_fact * np.prod([math.factorial(c) for c in occ]))
        dist[tuple(occ)] = p
    assert abs(sum(dist.values()) - 1) < 1e-10
    return dist

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    dev = DEV
    # deterministic cases
    occ = fock_sample(torch.eye(3, dtype=torch.complex128, device=dev), [0, 2, 2], 50,
                      generator=torch.Generator(device=dev).manual_seed(0)).cpu().numpy()
    assert (occ == np.array([1, 0, 2])).all(), "identity must return the input occupation"
    hom = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128, device=dev) / math.sqrt(2)
    occ = fock_sample(hom, [0, 1], 300, generator=torch.Generator(device=dev).manual_seed(0)).cpu().numpy()
    assert not (occ == 1).all(1).any(), "HOM: |1,1> must never occur"
    print("identity / HOM: OK")
    for (m, in_modes, shots, algo) in [(4, [0, 1, 2], 600, "2018"), (5, [0, 0, 1], 600, "2018"),
                                       (4, [0, 1, 2, 3], 600, "2020"), (5, [0, 0, 1], 600, "2020")]:
        U = haar(m, rng)
        dist = exact_dist(U, in_modes)
        gen = torch.Generator(device=dev).manual_seed(0)
        occ = fock_sample(torch.tensor(U, device=dev), in_modes, shots, generator=gen, algo=algo).cpu().numpy()
        emp = {}
        for row in occ:
            assert tuple(row) in dist, f"impossible outcome {row}"
            emp[tuple(row)] = emp.get(tuple(row), 0) + 1
        tvd = 0.5*sum(abs(dist[k] - emp.get(k, 0)/shots) for k in dist)
        exp_tvd = 0.5*sum(math.sqrt(p*(1-p)/shots) for p in dist.values())*math.sqrt(2/math.pi)
        print(f"m={m} in={in_modes} shots={shots} algo={algo}: TVD={tvd:.4f}  (statistical floor ~{exp_tvd:.4f}, #outcomes={len(dist)})")
        assert tvd < 2.5*exp_tvd
    print("OK")
