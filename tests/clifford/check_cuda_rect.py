"""CUDA rectangular cofactor kernel (clifford/cuda_rect.py) on a GPU: against the torch cofactors of sbs_rso at the
chain's step sizes up to N = 44, identical end-to-end SBS draws, and the per-step time against the torch path.

    python3 tests/clifford/check_cuda_rect.py [--streams 64]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from photonic_engines import ProductSuperposition
from photonic_engines.clifford import cuda_rect
from photonic_engines.clifford.sbs_rso import rect_cofactors, sample

ap = argparse.ArgumentParser()
ap.add_argument("--streams", type=int, default=64)
ap.add_argument("--torch-max-q", type=int, default=27,
                help="above this many placed rows the torch reference is skipped in the timing section (it costs "
                     "576 s at q = 27 for 64 streams and doubles per row: 5 h at q = 32; run 20261004_065629)")
args = ap.parse_args()
dev = "cuda"
assert cuda_rect.available(), "needs cupy and a CUDA device"
cuda_rect.warmup()
rng = np.random.default_rng(1)

print("rectangular cofactors, cuda vs torch (max |dP| / max |P|), S = 4:")
for N, q in ((16, 8), (24, 16), (32, 20), (40, 24), (44, 30), (72, 12), (90, 10)):
    m = 2 * N
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(N)))[0].to(dev)
    S = 4
    A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(s))[:N]] for s in range(S)])
    rows = torch.tensor(rng.integers(0, m, size=(S, q))).to(dev)
    torch.cuda.synchronize(); t0 = time.time()
    B = torch.stack([A[s][rows[s]] for s in range(S)])
    ref = rect_cofactors(B)
    torch.cuda.synchronize(); t1 = time.time()
    got = cuda_rect.rect_cofactors_cuda(A, rows, q)
    torch.cuda.synchronize(); t2 = time.time()
    err = float((got - ref).abs().max() / ref.abs().max())
    NP, tpl = cuda_rect.tier(N)
    # two fp64 summation orders of a 2^(q-1)-term alternating sum: 2e-15 (q = 8) .. 3e-12 (24) .. 2e-10 (30), ~x2 per
    # row (run 20261003_225335); the kernel's own error vs its long-double build is 4e-11 at q = 26 (bench/rect_precision.py)
    print(f"  N={N:2d} q={q:2d} tier {NP} TPL={tpl}: {err:.2e} {'ok' if err < 1e-8 else 'MISMATCH'}   "
          f"torch {t1 - t0:7.2f} s  cuda {t2 - t1:7.2f} s  x{(t1 - t0) / (t2 - t1):.1f}", flush=True)
    assert err < 1e-8

n, m = 12, 24
st = ProductSuperposition.sbs(n, m, alpha=math.pi / 4)
U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(7)))[0].to(dev)
a = sample(st, U, 256, generator=torch.Generator().manual_seed(5), backend="torch", device=dev)
b = sample(st, U, 256, generator=torch.Generator().manual_seed(5), backend="cuda", device=dev)
print("end-to-end draws identical (n = 12, 256 shots):", bool(torch.equal(a, b)))
assert torch.equal(a, b)

S = args.streams
print(f"one chain step at the frontier sizes, {S} streams (seconds, torch vs cuda):")
for n, k, j in ((42, 21, 21), (42, 28, 28), (42, 33, 33), (48, 24, 24)):
    m = 2 * n
    N, q = n - k + j, j - 1
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].to(dev)
    A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(s))[:N]] for s in range(S)])
    rows = torch.tensor(rng.integers(0, m, size=(S, q))).to(dev)
    torch.cuda.synchronize(); t0 = time.time()
    if q <= args.torch_max_q:
        B = torch.stack([A[s][rows[s]] for s in range(S)])
        ref = rect_cofactors(B)
    torch.cuda.synchronize(); t1 = time.time()
    got = cuda_rect.rect_cofactors_cuda(A, rows, q)
    torch.cuda.synchronize(); t2 = time.time()
    if q <= args.torch_max_q:
        err = float((got - ref).abs().max() / ref.abs().max())
        print(f"  n={n} k={k} step j={j} (N={N}, q={q}): torch {t1 - t0:8.2f} s  cuda {t2 - t1:8.2f} s  "
              f"x{(t1 - t0) / (t2 - t1):5.1f}  err {err:.1e}", flush=True)
    else:
        print(f"  n={n} k={k} step j={j} (N={N}, q={q}): torch  (skipped, q > {args.torch_max_q})  cuda {t2 - t1:8.2f} s",
              flush=True)

# The division-free step payloads at the Fock limit and the continuity region (s0 = 0 .. DMAX): the chain's last step
# at n = 34, k = 34 - s0 (q = k - 1, N = n), 64 streams. s0 = 0: the production Fock kernel is the reference AND the
# dispatched payload (form "fock"); the mu-form must agree with it. s0 = 1, 2: no exact reference at this size (the
# lambda-form loses ~10 digits at q ~ N -- 1.8e-6 at s0 = 0, run 20261005_111300 -- and the phantom route cancels,
# 4e-5 at s0 = 1, run 20261005_120751), so the mu-form is checked against itself in another summation order (TPL
# changed: a lower bound of its error, expected ~1e-9 as the Fock kernel's own at k = 34). s0 = 4, 6: mu vs lambda.
print(f"step payloads at the Fock limit, n = 34, last step (N = 34, q = 33 - s0), {S} streams:")
n = 34
m = 2 * n
U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(34)))[0].to(dev)
A = torch.stack([U[:, torch.randperm(m, generator=torch.Generator().manual_seed(s))[:n]] for s in range(S)])
for s0 in (0, 1, 2, 4, 6):
    q = n - 1 - s0
    rows = torch.tensor(rng.integers(0, m, size=(S, q))).to(dev)
    torch.cuda.synchronize(); t0 = time.time()
    la = cuda_rect.rect_cofactors_cuda(A, rows, q, form="lambda")
    torch.cuda.synchronize(); t1 = time.time()
    mu = cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu")
    torch.cuda.synchronize(); t2 = time.time()
    if s0 == 0:
        ref = cuda_rect.rect_cofactors_cuda(A, rows, q, form="fock")
        torch.cuda.synchronize(); t3 = time.time()
        err = float((mu - ref).abs().max() / ref.abs().max())
        print(f"  s0={s0} q={q}: lambda {t1 - t0:8.2f} s  mu {t2 - t1:8.2f} s  Fock kernel {t3 - t2:8.2f} s  mu err {err:.1e}  "
              f"lambda err {float((la - ref).abs().max() / ref.abs().max()):.1e}  auto -> {cuda_rect.choose_form(q, n)}", flush=True)
    elif s0 <= 2:
        NPm, TPLm, _ = cuda_rect.mu_tier(n, s0)                 # the templated mu lane's own tier / TPL table
        cuda_rect.set_mu_config({(NPm, s0): (8 if TPLm != 8 else 4, None)})
        mu2 = cuda_rect.rect_cofactors_cuda(A, rows, q, form="mu")
        torch.cuda.synchronize(); t3 = time.time()
        cuda_rect.set_mu_config(None)
        err = float((mu - mu2).abs().max() / mu.abs().max())
        err_la = float((mu - la).abs().max() / mu.abs().max())
        print(f"  s0={s0} q={q}: lambda {t1 - t0:8.2f} s  mu {t2 - t1:8.2f} s  x{(t1 - t0) / (t2 - t1):5.2f}  "
              f"mu (other TPL) {t3 - t2:8.2f} s  mu two orders {err:.1e}  mu vs lambda {err_la:.1e}  "
              f"auto -> {cuda_rect.choose_form(q, n)}", flush=True)
        err = err * 1e-2                                    # two orders of a 2^32-term sum: assert at 1e-6 (lower bound, not a reference)
    else:
        err = float((mu - la).abs().max() / la.abs().max())
        print(f"  s0={s0} q={q}: lambda {t1 - t0:8.2f} s  mu {t2 - t1:8.2f} s  x{(t1 - t0) / (t2 - t1):5.2f}  "
              f"mu vs lambda {err:.1e}  auto -> {cuda_rect.choose_form(q, n)}", flush=True)
    assert err < 1e-8, (s0, err)
