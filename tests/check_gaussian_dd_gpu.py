"""GPU checks of the double-double sieve (docs/GAUSSIAN.md §3.4 v2.11; job 27). Run on a CUDA host:

    python3 tests/check_gaussian_dd_gpu.py

A. gaussian.dd on CUDA tensors against exact rationals (element-wise ops relative to the operands; the grouped Ozaki
   product of the complex block forms normwise, relative to q x row max x column max, the scheme's own bound): the
   error-free transformations must survive the GPU's arithmetic (no fused multiply-add in
   eager torch kernels; the GEMMs exact on the tensor cores).
C. the dd series kernel alone on CUDA against the torch dd recursion on CUDA (isolates the kernel's fma path).
B. the dd sieve on CUDA (the Triton series kernel with fma-based two_prod, stacked Ozaki products) against the dd sieve on
   the CPU (torch recursion, Dekker split): the same polynomial to the complex128 rounding of the result, for collision-free
   and collided instances, N = 12 .. 24, V = 1 and 3.
D. (v2.11c) the fused Ozaki slicing kernel against the torch slicing of the embedded block forms on CUDA: bit for bit.
   A and B run with the fused slicing (the default on CUDA).
Exit status 1 on any failure.
"""
import math
import sys
from fractions import Fraction as F

import numpy as np
import torch

from photonic_engines.gaussian import dd
from photonic_engines.gaussian.hafnian import expand, lhaf_poly_multi

dev = torch.device("cuda")
rng = np.random.default_rng(0)
ok = True
EPS_DD = 2.0 ** -104


def rdd(shape):
    h = torch.tensor(rng.normal(size=shape) * np.exp2(rng.integers(-20, 20, size=shape)))
    lo = torch.tensor(rng.normal(size=shape)) * h.abs() * 2 ** -54
    hr, lr = dd.quick_two_sum(h, lo)
    h2 = torch.tensor(rng.normal(size=shape)); l2 = torch.tensor(rng.normal(size=shape)) * h2.abs() * 2 ** -54
    hi, li = dd.quick_two_sum(h2, l2)
    return torch.complex(hr, hi).to(dev), torch.complex(lr, li).to(dev)


def cf(h, lo):
    return F(float(h.real)) + F(float(lo.real)), F(float(h.imag)) + F(float(lo.imag))


print("A. gaussian.dd on CUDA against exact rationals", flush=True)
a = rdd((64,)); b = rdd((64,))
m = dd.cmul(*a, *b); s = dd.add(*a, *b); q = dd.div_scalar(*a, 7.0)
worst = {"cmul": 0.0, "add": 0.0, "div": 0.0}
for i in range(64):
    x, y = cf(a[0][i].cpu(), a[1][i].cpu()), cf(b[0][i].cpu(), b[1][i].cpu())
    z = (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]); g = cf(m[0][i].cpu(), m[1][i].cpu())
    sc = abs(x[0] * y[0]) + abs(x[1] * y[1]) + abs(x[0] * y[1]) + abs(x[1] * y[0])
    worst["cmul"] = max(worst["cmul"], float((abs(g[0] - z[0]) + abs(g[1] - z[1])) / sc / F(EPS_DD)))
    g = cf(s[0][i].cpu(), s[1][i].cpu())
    worst["add"] = max(worst["add"], float((abs(g[0] - x[0] - y[0]) + abs(g[1] - x[1] - y[1]))
                                           / (abs(x[0]) + abs(y[0]) + abs(x[1]) + abs(y[1])) / F(EPS_DD)))
    g = cf(q[0][i].cpu(), q[1][i].cpu())
    worst["div"] = max(worst["div"], float((abs(g[0] - x[0] / 7) + abs(g[1] - x[1] / 7)) * 7 / (abs(x[0]) + abs(x[1])) / F(EPS_DD)))
for qd in (16, 64, 128):
    A = rdd((2, 3, qd)); B = rdd((2, qd, 2)); C = dd.cmatmul(*A, *B); w = 0.0
    for bb in range(2):
        for i in range(3):
            for j in range(2):
                re = im = F(0)
                for k in range(qd):
                    x, y = cf(A[0][bb, i, k].cpu(), A[1][bb, i, k].cpu()), cf(B[0][bb, k, j].cpu(), B[1][bb, k, j].cpu())
                    re += x[0] * y[0] - x[1] * y[1]; im += x[0] * y[1] + x[1] * y[0]
                # the Ozaki bound is normwise: relative to q x (max of the row of A) x (max of the column of B), both parts
                ra = max(max(abs(float(A[0][bb, i, k].real)), abs(float(A[0][bb, i, k].imag))) for k in range(qd))
                cb = max(max(abs(float(B[0][bb, k, j].real)), abs(float(B[0][bb, k, j].imag))) for k in range(qd))
                g = cf(C[0][bb, i, j].cpu(), C[1][bb, i, j].cpu())
                w = max(w, float((abs(g[0] - re) + abs(g[1] - im)) / (F(qd) * F(ra) * F(cb)) / F(EPS_DD)))
    worst[f"cmatmul q={qd}"] = w
for k, v in worst.items():
    good = v <= 8
    ok &= good
    print(f"  {k:14s} {v:6.2f} eps_dd  {'ok' if good else 'FAIL'}", flush=True)

print("C. the dd series alone: Triton kernel (fma) vs the torch recursion, both on CUDA, same inputs", flush=True)
from photonic_engines.gaussian.hafnian import _series_dd_torch
from photonic_engines.gaussian.sieve_dd_triton import series_dd

for n_, V_ in ((4, 1), (12, 2), (24, 3)):
    gh = torch.tensor(rng.normal(size=(5, V_, n_, 3)) + 1j * rng.normal(size=(5, V_, n_, 3)), device=dev) / n_
    gl = gh * 2 ** -60
    gh, gl = dd.quick_two_sum(gh, gl)
    kh, kl = series_dd(gh.contiguous(), gl.contiguous())
    th, tl = _series_dd_torch(gh, gl)
    num = float(((kh - th) + (kl - tl)).abs().max()); den = float(th.abs().max())
    good = num <= 64 * EPS_DD * den
    ok &= good
    print(f"  n={n_:2d} V={V_}: |kernel - torch| / max {num / den:.1e} ({num / den / EPS_DD:.0f} eps_dd)  {'ok' if good else 'FAIL'}",
          flush=True)

print("D. fused Ozaki slicing (v2.11c) vs the torch slicing of the block forms, both on CUDA: bit for bit", flush=True)
from photonic_engines.gaussian.sieve_dd_triton import slices_complex

for shp in ((7, 5, 9), (3, 32, 32), (2, 4, 64, 64), (5, 128, 2)):
    hs = torch.tensor(rng.normal(size=shp) + 1j * rng.normal(size=shp), device=dev) * torch.exp(
        3 * torch.tensor(rng.normal(size=shp), device=dev))
    ls = hs * 1e-17 * torch.tensor(rng.normal(size=shp), device=dev)
    hs[..., 0, :] = 0; ls[..., 0, :] = 0
    for left in (True, False):
        beta, s_ = dd.plan(2 * shp[-1] if left else 2 * shp[-2])
        ref = dd.slices(*(dd.embed_left(hs, ls) if left else dd.embed_right(hs, ls)), beta, s_, -1 if left else -2)
        got = slices_complex(hs, ls, beta, s_, left)
        good = len(got) == s_ and all(torch.equal(a, b) for a, b in zip(ref, got))
        ok &= good
        print(f"  {shp!s:16s} {'left ' if left else 'right'}: {'identical' if good else 'DIFFERENT'}  "
              f"{'ok' if good else 'FAIL'}", flush=True)

print("B. dd sieve: CUDA (Triton series, fma) vs CPU (torch, Dekker)", flush=True)
for counts, V in (([1] * 12, 1), ([1] * 16, 3), ([3, 2, 2, 1, 1, 1, 1, 1], 1), ([4, 4, 3, 2, 1, 1, 1], 3), ([1] * 24, 1),
                  ([5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1], 3)):
    mm = len(counts)
    X = rng.normal(size=(mm, mm)) + 1j * rng.normal(size=(mm, mm)); Bm = (X + X.T) / (2 * math.sqrt(mm))
    gam = rng.normal(size=(V, mm)) + 1j * rng.normal(size=(V, mm)); col = rng.normal(size=(V, mm)) + 1j * rng.normal(size=(V, mm))
    Am, _ = expand(Bm, gam[0], counts); lab = np.repeat(np.arange(mm), counts)
    r_cpu = lhaf_poly_multi(Am, gam[:, lab], col[:, lab], labels=lab, precision="dd", device="cpu")
    r_gpu = lhaf_poly_multi(Am, gam[:, lab], col[:, lab], labels=lab, precision="dd", device=dev).cpu()
    r_64 = lhaf_poly_multi(Am, gam[:, lab], col[:, lab], labels=lab, device=dev).cpu()
    scale = float(r_cpu.abs().max())
    diff = float((r_gpu - r_cpu).abs().max()) / scale; d64 = float((r_64 - r_cpu).abs().max()) / scale
    good = diff <= 4 * 2.0 ** -53
    ok &= good
    print(f"  N={len(lab):2d} V={V} counts {str(counts)[:28]:28s}: |gpu dd - cpu dd| / max {diff:.1e}, fp64 {d64:.1e}  "
          f"{'ok' if good else 'FAIL'}", flush=True)
print("all passed" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
