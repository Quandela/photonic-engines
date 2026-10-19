"""GPU validation of the Gaussian engine (docs/GAUSSIAN.md §6). Runnable script, not collected.

    python3 tests/check_gaussian_gpu.py [--max-n 36] [--shots 3000]

A. lhaf_poly on CUDA against the CPU path, N = 6..28: relative error of the coefficient vector
   (A2: the v2 sieve lhaf_poly_multi on CUDA, with collision groups, against v1 on the CPU)
   (both complex128, same algorithm, different reduction order: agreement to ~1e-12 expected).
B. The sampler with every loop hafnian forced onto the GPU (cpu_below=0): chi-square against the
   truncated-ProductSuperposition oracle (exact for patterns with <= d-1 photons), lossless and lossy.
C. Timing of one lhaf_poly (degree-N polynomial) on CPU and GPU, N = 8..max-n: the table the GPU
   optimisation starts from.
D. Per-shot profile at m = 2n, 0.5 photons per squeezer: seconds per shot against the detected N, and the
   worst eps x sieve condition number met in the shot (the precision question at frontier sizes).
Exit code 1 if A or B fails.
"""
import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import haar_unitary

from photonic_engines import GaussianInput
from photonic_engines.gaussian import lhaf_poly, sample
from photonic_engines.gaussian.hafnian import expand, lhaf_poly_multi
from photonic_engines.slos.kernel import ProductSLOSKernel

ap = argparse.ArgumentParser()
ap.add_argument("--max-n", type=int, default=36, help="largest loop hafnian timed in C")
ap.add_argument("--shots", type=int, default=3000)
ap.add_argument("--profile-n", default="20,30,40,50", help="squeezer counts for D")
ap.add_argument("--profile-shots", type=int, default=8)
args = ap.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("needs CUDA")
dev = "cuda"
print(torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
ok = True

print("\nA. lhaf_poly cuda vs cpu", flush=True)
# The sieve is an alternating sum: with kappa = sum_S |term_S| / max|c| its rounding error is ~ eps kappa per
# path, and each term carries ~N eps from its N/2 matrix products. Tolerance: N eps kappa (derived, not picked;
# the first run of this check used a fixed 1e-10 and "failed" at N = 26 with kappa = 4e7, measured 8.7e-10).
rng = np.random.default_rng(1)
for N in range(6, 29, 2):
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); A = (A + A.T) / (2 * math.sqrt(N))
    D0 = rng.normal(size=N) + 1j * rng.normal(size=N); D1 = rng.normal(size=N) + 0j
    a, a_abs = lhaf_poly(A, D0, D1, device="cpu", return_abs=True); b = lhaf_poly(A, D0, D1, device=dev).cpu()
    scale = float(a.abs().max())
    kappa = float(a_abs.max()) / scale
    rel = float((a - b).abs().max()) / scale
    tol = N * float(np.finfo(np.float64).eps) * kappa
    flag = rel <= tol
    ok &= flag
    print(f"  N={N:2d}: max rel diff {rel:.2e}  (kappa {kappa:.1e}, tolerance N eps kappa = {tol:.1e})  {'ok' if flag else 'FAIL'}",
          flush=True)


print("\nA2. lhaf_poly_multi (v2: gathered buckets, 4 loop vectors, collision groups) cuda vs cpu v1", flush=True)
for counts in ([1] * 20, [2, 2, 1, 1, 3] + [1] * 13, [4, 4, 2] + [1] * 14, [1] * 26):
    N = sum(counts); mm = len(counts)
    Bm = rng.normal(size=(mm, mm)) + 1j * rng.normal(size=(mm, mm)); Bm = (Bm + Bm.T) / (2 * math.sqrt(N))
    lab = np.repeat(np.arange(mm), counts)
    A, _ = expand(Bm, np.zeros(mm, dtype=complex), counts)
    D0 = (rng.normal(size=(4, mm)) + 1j * rng.normal(size=(4, mm)))[:, lab]; D1 = rng.normal(size=(4, mm))[:, lab] + 0j
    ref = torch.stack([lhaf_poly(A, D0[v], D1[v], device="cpu", return_abs=True)[0] for v in range(4)])
    kap = max(float(lhaf_poly(A, D0[v], D1[v], device="cpu", return_abs=True)[1].max() /
                    lhaf_poly(A, D0[v], D1[v], device="cpu").abs().max()) for v in range(4))
    got = lhaf_poly_multi(A, D0, D1, device=dev, labels=lab).cpu()
    rel = float((got - ref).abs().max() / ref.abs().max())
    tol = N * float(np.finfo(np.float64).eps) * kap
    flag = rel <= tol
    ok &= flag
    print(f"  N={N:2d} pattern max {max(counts)}: max rel diff {rel:.2e} (tolerance {tol:.1e})  {'ok' if flag else 'FAIL'}",
          flush=True)


def chi2(S, exact, min_expected=15):
    c = Counter(map(tuple, S)); chi, dof = 0.0, 0
    for o, p in exact.items():
        e = p * len(S)
        if e >= min_expected:
            chi += (c.get(o, 0) - e) ** 2 / e; dof += 1
    return chi, dof


print("\nB. sampler on the GPU vs the truncated SLOS oracle", flush=True)
for label, eta in (("lossless", None), ("eta=0.6", 0.6)):
    st = GaussianInput(np.array([0.7, 0.5, 0.6]), (0, 1, 2), 4)
    U = haar_unitary(4, 3)
    d = 21 if eta else 11
    dist = ProductSLOSKernel(st.truncated(d), dtype=torch.complex128).probs(U, loss=eta)
    exact = {}
    for k in range(d if eta is None else 10):
        for o, p in zip(dist.occupations(k), dist.probs[k].numpy()):
            exact[tuple(int(x) for x in o)] = float(p)
    t = time.perf_counter()
    S = sample(st, U, args.shots, loss=eta, seed=11, device=dev, cpu_below=0)
    dt = time.perf_counter() - t
    chi, dof = chi2(S, exact)
    thr = dof + 5 * math.sqrt(2 * dof)
    flag = chi < thr
    ok &= flag
    print(f"  {label}: chi2 = {chi:.1f} over {dof} dof (threshold {thr:.1f})  {'ok' if flag else 'FAIL'}  "
          f"({dt:.1f} s, {args.shots} shots)", flush=True)

print("\nC. one lhaf_poly: seconds on cpu / cuda", flush=True)
for N in range(8, args.max_n + 1, 2):
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); A = (A + A.T) / (2 * math.sqrt(N))
    D0 = rng.normal(size=N) + 0j; D1 = rng.normal(size=N) + 0j
    tc = float("nan")
    if N <= 26:
        t = time.perf_counter(); lhaf_poly(A, D0, D1, device="cpu"); tc = time.perf_counter() - t
    lhaf_poly(A, D0, D1, device=dev); torch.cuda.synchronize()
    t = time.perf_counter(); lhaf_poly(A, D0, D1, device=dev); torch.cuda.synchronize(); tg = time.perf_counter() - t
    print(f"  N={N:2d}: cpu {tc:9.4f} s   cuda {tg:9.4f} s   ({2 ** (N / 2):.0f} subsets)", flush=True)

print("\nD. per-shot profile, m = 2n, sinh^2 r = 0.5", flush=True)
r = GaussianInput.r_for_mean(0.5)
for n in (int(x) for x in args.profile_n.split(",")):
    st = GaussianInput.smsv(n, 2 * n, r)
    U = haar_unitary(2 * n, 100 + n)
    S, stats = sample(st, U, args.profile_shots, seed=n, device=dev, return_stats=True)
    rows = sorted(zip(stats.photons, stats.seconds))
    print(f"  n={n:2d}: mean {np.mean(stats.seconds):.3f} s/shot, worst expected round-off {max(stats.round_off):.1e}; (N, s): "
          + ", ".join(f"({p}, {s:.2f})" for p, s in rows), flush=True)

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
