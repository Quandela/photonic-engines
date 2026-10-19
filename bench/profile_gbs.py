"""Where the time of a GBS shot goes (docs/GAUSSIAN.md §3.4, the v3 plan): the benchmark for sampling
optimisation, deterministic (fixed matrices, fixed seeds, capped shots) so that versions compare on the
same work.

    python3 bench/profile_gbs.py --device cuda --csv gbs_profile.csv [--trace-n 20,36]

A. one sieve per size N (vertices) and shape: v1 `lhaf_poly` (V = 1) and v2 `lhaf_poly_multi` with V = 1 and
   --vmulti loop vectors, collision-free and with collision groups (2x2: two modes holding 2 photons;
   1x4: one mode holding 4). Seconds per call (min of --reps below --slow-from vertices), and the relative
   difference v2 vs v1. Sizes stop once a call exceeds --max-sieve-s.
B. torch-profiler trace of one v2 sieve at each N of --trace-n: wall time, summed CUDA kernel time, kernel
   launches and the top kernels — how launch-bound a size is (GPU busy fraction = kernel time / wall).
C. whole shots at m = ratio n, fixed seeds, the photon cap of `sample(max_photons=)` (--cap): per version,
   seconds per shot and the breakdown of `sampler.Profile` — sieve time per size bucket, the draws
   (normaliser + inverse CDF), the host preparation, and the rest (heterodyne, bookkeeping). Censored shots
   are counted and their time included (the cap bounds it). The versions must draw identical samples.
CSV (one file): sieve rows a/b/c = v1, v2 V=1, v2 V=vmulti seconds, d = relative difference, e / censored
column = kappa v1 / v2; trace rows
a = wall, b = summed kernel seconds, c = launches; shot rows a = s/shot, b/c/d/e = sieve / draw / prep / rest
seconds per shot, then censored shots, identical fraction, sieve calls and draws per shot.
"""
import argparse
import csv
import time

import numpy as np
import torch

import photonic_engines.gaussian.hafnian as haf
import photonic_engines.gaussian.sampler as smp
from photonic_engines import GaussianInput
from photonic_engines.gaussian import sample
from photonic_engines.gaussian.hafnian import expand, lhaf_poly, lhaf_poly_multi

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--sieve-n", default="16,20,24,28,32,36,40,44,48")
ap.add_argument("--shapes", default="free,2x2,1x4")
ap.add_argument("--vmulti", type=int, default=4)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--slow-from", type=int, default=32, help="one repetition from this many vertices")
ap.add_argument("--max-sieve-s", type=float, default=60.0)
ap.add_argument("--trace-n", default="20,36")
ap.add_argument("--n", default="20,30,40,50")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--mean", type=float, default=0.5)
ap.add_argument("--shots", type=int, default=16)
ap.add_argument("--cap", type=int, default=44, help="photon cap per shot in C (-1: none)")
ap.add_argument("--versions", default="v1,v2w1,v2w4", help="v1 | v2w<window>")
ap.add_argument("--fused", default="auto", choices=["auto", "0", "1"],
                help="layout of the v2 sieve's per-subset work (hafnian._FUSED): auto = fused on CUDA, loops on CPU")
ap.add_argument("--kernel", default="auto", choices=["auto", "torch", "triton"],
                help="v2 matrix part: the Triton kernel (hafnian._TRITON; auto = CUDA from _TRITON_FROM pairs) or torch")
ap.add_argument("--sections", default="A,B,C")
ap.add_argument("--also-self", action="store_true",
                help="A: also run v2 with pairing='self' (the collision self-pairs, before 2026-09-24) and print its "
                     "kappa and error against v1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default=None)
args = ap.parse_args()
dev = args.device
EPS = float(np.finfo(np.float64).eps)
haf._FUSED = None if args.fused == "auto" else args.fused == "1"
haf._TRITON = None if args.kernel == "auto" else args.kernel == "triton"
rows = []


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def counts_for(N, shape):
    if shape == "free":
        return [1] * N
    if shape == "2x2":
        return [2, 2] + [1] * (N - 4)
    if shape == "1x4":
        return [4] + [1] * (N - 4)
    raise SystemExit(f"unknown shape {shape}")


def timed(f, reps):
    best = float("inf"); out = None
    for _ in range(reps):
        sync(); t = time.perf_counter(); out = f(); sync(); best = min(best, time.perf_counter() - t)
    return best, out


print(f"{torch.cuda.get_device_name(0) if dev == 'cuda' else dev}  torch {torch.__version__}  "
      f"v2 layout: fused {args.fused} (auto: up to {haf._FUSED_MAX} pairs on CUDA), kernel {args.kernel} "
      f"(auto: Triton from {haf._TRITON_FROM} pairs on CUDA)", flush=True)
if dev == "cuda" and args.kernel != "torch":                      # compile every Triton variant before timing
    from photonic_engines.gaussian.sieve_triton import warmup
    t_w = time.perf_counter(); warmup(dev, max(args.vmulti, 8)); print(f"Triton warm-up {time.perf_counter() - t_w:.1f} s", flush=True)
SEC = set(args.sections.split(","))
if "A" in SEC:
    print(f"\nA. one sieve (s per call; v2 V={args.vmulti} also per vector)", flush=True)
stopped = set()
for N in (int(x) for x in args.sieve_n.split(",") if "A" in SEC):
    for shape in args.shapes.split(","):
        if shape in stopped:
            continue
        cnt = counts_for(N, shape); k = len(cnt); m = 2 * k
        st = GaussianInput.smsv(k, m, GaussianInput.r_for_mean(args.mean))
        B, _, _ = st.bargmann(haar(m, N))
        rng = np.random.default_rng(N)
        gam = rng.normal(size=(args.vmulti, k)) * 0.5 + 0j
        A, D00 = expand(B[:k, :k], gam[0], cnt)
        lab = np.repeat(np.arange(k), cnt)
        D0 = gam[:, lab]; D1 = B[:k, k:k + args.vmulti].T[:, lab].copy()
        reps = 1 if args.slow_from <= N else args.reps
        lhaf_poly(A[:2, :2], D0[0, :2], D1[0, :2], device=dev)          # warm-up (allocator, kernels)
        lhaf_poly_multi(A[:2, :2], D0[:1, :2], D1[:1, :2], device=dev, labels=lab[:2])
        # v1 on the expanded matrix is the oracle for every shape; return_abs gives each sieve's condition
        # number kappa, and a difference is accounted for by rounding when rel <= N eps max(kappa)
        t1, (a, aa) = timed(lambda: lhaf_poly(A, D0[0], D1[0], device=dev, return_abs=True), reps)
        t2, (b, ba) = timed(lambda: lhaf_poly_multi(A, D0[:1], D1[:1], device=dev, labels=lab, return_abs=True), reps)
        t3, _ = timed(lambda: lhaf_poly_multi(A, D0, D1, device=dev, labels=lab), reps)
        rel = float((a - b[0]).abs().max() / a.abs().max())
        k1 = float(aa.max() / a.abs().max()); k2 = float(ba[0].max() / b[0].abs().max())
        bound = N * EPS * max(k1, k2)
        print(f"  N={N:2d} {shape:4s}: v1 {t1:9.4f}  v2 V=1 {t2:9.4f}  V={args.vmulti} {t3:9.4f} "
              f"({t3 / args.vmulti:9.4f}/vector)  rel {rel:.1e}  kappa v1 {k1:.1e} v2 {k2:.1e}  "
              f"{'within' if rel <= bound else 'ABOVE'} N eps kappa = {bound:.1e}", flush=True)
        rows.append(["sieve", N, shape, args.vmulti, t1, t2, t3, rel, k1, k2, "", "", ""])
        if args.also_self:
            t4, (c4, ca4) = timed(lambda: lhaf_poly_multi(A, D0[:1], D1[:1], device=dev, labels=lab, return_abs=True,
                                                           pairing="self"), 1)
            rel4 = float((a - c4[0]).abs().max() / a.abs().max()); k4 = float(ca4[0].max() / c4[0].abs().max())
            print(f"         v2 pairing=self {t4:9.4f} s  rel {rel4:.1e}  kappa {k4:.1e}  "
                  f"{'within' if rel4 <= N * EPS * max(k1, k4) else 'ABOVE'} N eps kappa = {N * EPS * max(k1, k4):.1e}",
                  flush=True)
            rows.append(["sieve_self", N, shape, 1, "", t4, "", rel4, k1, k4, "", "", ""])
        if max(t1, t2, t3) > args.max_sieve_s:
            stopped.add(shape)

if dev == "cuda" and args.trace_n and "B" in SEC:
    from torch.profiler import ProfilerActivity, profile
    print("\nB. torch-profiler trace of one v2 sieve (V = 1, collision-free)", flush=True)
    for N in (int(x) for x in args.trace_n.split(",")):
        st = GaussianInput.smsv(N, 2 * N, GaussianInput.r_for_mean(args.mean))
        B, _, _ = st.bargmann(haar(2 * N, N))
        A = B[:N, :N]; D0 = np.full((1, N), 0.3 + 0j); D1 = B[:N, N:N + 1].T.copy()
        lhaf_poly_multi(A, D0, D1, device=dev); sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t = time.perf_counter(); lhaf_poly_multi(A, D0, D1, device=dev); sync(); wall = time.perf_counter() - t
        ev = [e for e in prof.key_averages() if getattr(e, "device_type", None) is not None
              and str(e.device_type).endswith("CUDA")]
        kern_t = sum(e.self_device_time_total for e in ev) / 1e6 if ev else float("nan")
        launches = sum(e.count for e in ev)
        print(f"  N={N}: wall {wall:.4f} s, CUDA kernels {kern_t:.4f} s ({kern_t / wall:.0%} busy), {launches} launches",
              flush=True)
        for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:8]:
            print(f"      {e.self_device_time_total / 1e3:9.2f} ms  x{e.count:6d}  {e.key[:90]}", flush=True)
        rows.append(["trace", N, "free", 1, wall, kern_t, launches, "", "", "", "", "", ""])

if "C" in SEC:
    print(f"\nC. whole shots, m = {args.ratio:g} n, {args.mean} photons per squeezer, {args.shots} shots, "
          f"cap {args.cap}", flush=True)
cap = None if args.cap < 0 else args.cap
for n in (int(x) for x in args.n.split(",") if "C" in SEC):
    m = round(args.ratio * n)
    st = GaussianInput.smsv(n, m, GaussianInput.r_for_mean(args.mean))
    U = haar(m, args.seed + n)
    sample(st, U, 1, seed=1, device=dev, max_photons=cap)              # warm-up
    ref = None
    for ver in args.versions.split(","):
        impl, window = ("v1", 1) if ver == "v1" else ("v2", int(ver.split("w")[1]))
        smp.PROFILE = P = smp.Profile()
        sync(); t = time.perf_counter()
        S, stats = sample(st, U, args.shots, seed=args.seed + 7 * n, device=dev, impl=impl, window=window,
                          return_stats=True, max_photons=cap)
        sync(); T = time.perf_counter() - t
        smp.PROFILE = None
        ref = S if ref is None else ref
        same = float(np.mean(np.all(ref == S, axis=1)))
        sieve_t = sum(v[1] for v in P.sieve.values())
        rest = T - sieve_t - P.draw[1] - P.prep[1]
        buckets = {}
        for (N, _V, d), (c_, s_) in P.sieve.items():
            b = f"{d[:4]}:{(N // 8) * 8:02d}-{(N // 8) * 8 + 7:02d}"
            x = buckets.setdefault(b, [0, 0.0]); x[0] += c_; x[1] += s_
        bk = "  ".join(f"{b} {s_:.3f}s/{c_}" for b, (c_, s_) in sorted(buckets.items()))
        print(f"  n={n:2d} m={m:3d} {ver:5s}: {T / args.shots:8.3f} s/shot  sieve {sieve_t / T:4.0%}  draw {P.draw[1] / T:4.0%}"
              f"  prep {P.prep[1] / T:4.0%}  rest {rest / T:4.0%}  censored {sum(stats.censored)}  identical {same:.3f}",
              flush=True)
        print(f"        sieves: {bk}", flush=True)
        print(f"        photons {stats.photons}", flush=True)
        rows.append(["shot", n, ver, m, T / args.shots, sieve_t / args.shots, P.draw[1] / args.shots,
                     P.prep[1] / args.shots, rest / args.shots, sum(stats.censored), same,
                     sum(v[0] for v in P.sieve.values()) / args.shots, P.draw[0] / args.shots])
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "N_or_n", "shape_or_version", "V_or_m", "a_s", "b_s", "c_s", "d", "e_s",
                    "censored", "identical", "sieve_calls_per_shot", "draws_per_shot"])
        w.writerows(rows)
