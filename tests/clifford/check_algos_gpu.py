"""GPU gate of the T1 algorithm set (docs/ALGORITHMS.md §3): the four chain-rule sums of `fock_sample`
draw the SAME samples from the same seed (same sigma, same Philox draws, exact weights that agree
to rounding), on the Triton kernels compiled for real and on the CUDA production kernel, at a size
where every collapse and every lane plan is exercised; plus the Triton "input" kernel against the
plain-torch reference sum on random blocks with repeated rows and columns up to k = 14.

    python3 tests/clifford/check_algos_gpu.py [--n 12 --m 24 --shots 512]

Exit status 1 on any disagreement; run it on the GPU before anything is timed.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from helpers import haar_unitary

from photonic_engines.clifford import fock_sample
from photonic_engines.clifford.cofactors_input import qvec_input_rows
from photonic_engines.clifford.variants import qvec_input

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--m", type=int, default=24)
ap.add_argument("--shots", type=int, default=512)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
dev = args.device
ok = True

# 1. the kernel against the reference sum, repeated rows and columns
U = haar_unitary(16, 1).to(dev)
g = torch.Generator().manual_seed(0)
for k in (4, 6, 8, 11, 14):
    for rep in range(3):
        cols = torch.randint(0, 6 if rep else 16, (k,), generator=g).tolist()
        rows = torch.randint(0, 5 if rep else 16, (k - 1,), generator=g).tolist()
        Q, dcol = qvec_input_rows(U[None], torch.tensor([rows], device=dev), torch.tensor([cols], device=dev))
        Qr, ids, _ = qvec_input(U.cpu()[rows][:, cols], cols)
        ref = dict(zip(ids, Qr.tolist()))
        D = len(ids)
        scale = max(abs(v) for v in ref.values())
        err = max(abs(q - ref[mu]) for mu, q in zip(dcol[0, :D].tolist(), Q[0, :D].tolist())) / scale
        pad = float(Q[0, D:].abs().max()) if k > D else 0.0
        flag = err < 1e-12 and pad == 0.0
        ok &= flag
        print(f"input kernel k={k:2d} rows={rows} cols={cols}: rel err {err:.1e} pad {pad:.0e} {'ok' if flag else '*** MISMATCH ***'}")

# 2. the sampler: same seed, same samples, every algorithm, both backends
U = haar_unitary(args.m, 3).to(dev)
in_modes = list(range(args.n))
out = {}
try:
    import cupy  # noqa: F401
    have_cupy = True
except ImportError:                                                        # no cupy: the CUDA rows are skipped
    have_cupy = False
    print("cupy absent: the CUDA rows are skipped")
for algo, backend in (("2018", "triton"), ("2020", "triton"), ("input", "triton"), ("percof", "triton"),
                      ("2020", "cuda"), ("auto", "cuda"), ("percof", "cuda")):
    if backend == "cuda" and not have_cupy:
        continue
    out[(algo, backend)] = fock_sample(U, in_modes, args.shots, backend=backend, algo=algo,
                                       generator=torch.Generator(device=dev).manual_seed(11))
base = out[("2018", "triton")]
for (algo, backend), o in out.items():
    same = torch.equal(o, base)
    ok &= same
    print(f"{algo:6s} {backend:6s}: photons {o.sum(1).unique().tolist()}  same samples as 2018/triton: {same}")
print("GATE OK" if ok else "GATE FAILED")
sys.exit(0 if ok else 1)
