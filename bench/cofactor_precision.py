"""Rounding error of the fp64 cofactor sweeps at frontier sizes (docs/ALGORITHMS.md §6, Supplement C).

No exact reference exists at k ~ 38 (2^37 addends; a double-double sweep is the GBS engine's tool,
not this one's), so the error is ESTIMATED from disagreement between evaluations of the same
cofactors that round differently: the CUDA thread-per-lane kernel, the Triton 2020 kernel with two
lane segmentations (the per-lane partial sums are added in a different order), and the Triton 2018
kernel (different walk, same sum). The spread is a lower bound of the fp64 error of each; the paper
states it as such. Blocks: Haar U, m = 2n, n = k, distinct rows (the collision-free case is the one
with the most addends and the worst cancellation) and, with --collide, one row repeated twice.

    python3 bench/cofactor_precision.py --ks 30,34,38 --reps 2 --csv results/cof_precision.csv

CSV: k, rows, mode, rel_max (max over the k cofactors of |a - b| / max|cofactor|), rel_amp (the same on
the m candidate amplitudes, which is what the draw sees), seconds. Modes: cuda_vs_triton, triton_L_vs_l
(two segmentations), 2018_vs_2020 (k <= --k2018 only: the 2018 sweep has no collapse, 2^(k-2) addends).

--ref ld (2026-10-11, B's precision request): adds the TRUE fp64 error of each evaluation against an extended-precision
reference — the 2020 Glynn sum of the same cofactors computed by the CPU emulation of the CUDA kernel code compiled with
real = x87 long double (64-bit mantissa, eps 1.1e-19; `cuda_rect.rect_cofactors_emul`, form "fock" = the templated lane
at s0 = 0, every addend of the collapsed sum), split over --ld-workers concurrent calls of 8 lane threads each. Modes
<name>_vs_ld for name in cuda / triton_L / triton_l / 2018. Needs an x86-64 host (long double = double on arm64);
--ld-max-k skips the reference above that k. Floor: the lane partials are stored and summed in fp64, so the reference is good
to ~eps of the result (an upper bound on each evaluation's error at that level), far below the kernels' errors at k >= 30.
Cost ~2^(k-2) k long-double column updates (measured per k in the log).
"""
import argparse
import csv
import ctypes
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford.cofactors_triton import laplace_cofactors, laplace_cofactors_rows

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ks", default="30,34,38", help="block sizes k (the step: k-1 rows, k columns)")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--k2018", type=int, default=30, help="largest k at which the 2018 sweep is run (2^(k-2) addends)")
ap.add_argument("--collide", action="store_true", help="repeat one chosen row (exercises the collapse)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="results/cof_precision.csv")
ap.add_argument("--device", default="cuda", help="cpu runs the Triton pairs under the interpreter (small k only)")
ap.add_argument("--ref", default="none", choices=["none", "ld"], help="ld: long-double reference (x86-64 host CPU)")
ap.add_argument("--ld-workers", type=int, default=3, help="concurrent emulation calls (each runs 8 lane threads)")
ap.add_argument("--ld-max-k", type=int, default=64, help="largest k given the long-double reference")
ap.add_argument("--ld-lanes", type=int, default=1 << 12, help="lane plan of the reference (target total lanes)")
ap.add_argument("--ld-check", action="store_true", help="also run the reference with 1 worker: must agree bit-exactly")
args = ap.parse_args()
dev = args.device
if args.ref == "ld" and ctypes.sizeof(ctypes.c_longdouble) <= 8:
    raise SystemExit("long double is not extended precision on this host (arm64?): --ref ld needs x86-64")


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def rel(a, b):
    return float(((a - b).abs().max() / a.abs().max()).item())


def ld_reference(A, rows, workers, lanes):
    """P[s, x] = per(A[s][rows[s]] without column x) in x87 long double on the CPU: the kernel code's emulation
    (`cuda_rect.rect_cofactors_emul(form="fock", real="long double", tpl=1)`), its lanes split into `workers`
    concurrent calls. A [S, m, K] complex128, rows [S, K - 1] (CPU) -> [S, K] complex128.
    Each lane's partial is stored in fp64 and the partials are summed in fp64 (`_reduce`): the reference's floor is
    ~eps of the result, not 1e-19 — still far below the kernels' errors at k >= 30 (R, 10-11). `lanes` (default 2^12)
    is a different plan from the Triton runs' 2^17 / 2^11, so no lane split is shared with an evaluation under test."""
    from photonic_engines.clifford import cuda_rect as cr
    from photonic_engines.clifford.cofactors_triton import _binom_table
    A2 = torch.view_as_real(A.contiguous())
    S, N = A.shape[0], A.shape[2]
    q = rows.shape[1]
    Bs2, mult, radix, Nt, scale, L, nl, lstart, lane2s, total = cr._plan(A2, rows, q, lanes)
    lane2s, lstart, Nt = lane2s.contiguous(), lstart.contiguous(), Nt.contiguous()
    part = torch.zeros(total, N, 2, dtype=torch.float64)
    binom = _binom_table(torch.device("cpu"))
    lib = cr._emul_lib(cr.mu_tier(N, 0)[0], 1, "long double")

    def p(t, a=0):
        return ctypes.c_void_p(t.data_ptr() + a * t.stride(0) * t.element_size())

    def run(ab):
        # lane2s / part are per lane (pointer offset); lstart is per STREAM and the C takes li = g - lstart[s] with g
        # local to the call, so a call over global lanes [a, b) gets the VALUES lstart - a
        a, b = ab
        ls = (lstart - a).contiguous()
        lib.mud_emul(p(Bs2), p(mult), p(radix), p(lane2s, a), p(ls), p(Nt), p(binom), p(part, a), q, N, 0, L, b - a)

    cuts = np.linspace(0, total, workers + 1).astype(np.int64)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, [(int(a), int(b)) for a, b in itertools.pairwise(cuts) if b > a]))
    return cr._reduce(part, lane2s, scale, S, N)


rows_out = []
for k in [int(x) for x in args.ks.split(",")]:
    for rep in range(args.reps):
        m = 2 * k
        U = haar(m, args.seed + 100 * k + rep).to(dev)
        A = U[:, :k][None]                                                   # [1, m, k]
        g = torch.Generator().manual_seed(args.seed + rep)
        rows = torch.randperm(m, generator=g)[: k - 1]
        if args.collide:
            rows[1] = rows[0]
        rows = rows[None].to(dev)
        res = {}
        t = {}
        def sync():
            if dev == "cuda":
                torch.cuda.synchronize()
        evals = [("triton_L", lambda: laplace_cofactors_rows(A, rows, backend="triton", target_total_lanes=1 << 17)),
                 ("triton_l", lambda: laplace_cofactors_rows(A, rows, backend="triton", target_total_lanes=1 << 11))]
        if dev == "cuda":
            evals.insert(0, ("cuda", lambda: laplace_cofactors_rows(A, rows, backend="cuda")))
        for name, fn in evals:
            sync(); t0 = time.perf_counter()
            res[name] = fn()[0]
            sync(); t[name] = time.perf_counter() - t0
        if k <= args.k2018 and not args.collide:
            B = A[0][rows[0]][None]                                            # [1, k-1, k]
            sync(); t0 = time.perf_counter()
            res["2018"] = laplace_cofactors(B)[0]
            sync(); t["2018"] = time.perf_counter() - t0
        pairs = [("triton_L_vs_l", "triton_L", "triton_l")]
        if "cuda" in res:
            pairs.insert(0, ("cuda_vs_triton", "cuda", "triton_L"))
        if "2018" in res:
            pairs.append(("2018_vs_2020", "2018", "triton_L"))
        if args.ref == "ld" and k <= args.ld_max_k:
            names = list(res)
            t0 = time.perf_counter()
            res["ld"] = ld_reference(A.cpu(), rows.cpu(), args.ld_workers, args.ld_lanes)[0].to(dev)
            t["ld"] = time.perf_counter() - t0
            if args.ld_check and rep == 0 and k <= 24:
                one = ld_reference(A.cpu(), rows.cpu(), 1, args.ld_lanes)[0].to(dev)
                d = float((one - res["ld"]).abs().max().item())
                print(f"k={k:2d} split check: workers 1 vs {args.ld_workers} max |diff| = {d:.3e} (must be 0)", flush=True)
                if d != 0.0:
                    raise SystemExit("long-double reference depends on the lane split: stop")
            print(f"k={k:2d} rep={rep} long-double reference {t['ld']:.1f} s "
                  f"({t['ld'] / (2 ** (k - 2) * k) * 1e9:.2f} ns per term-column, {args.ld_workers} x 8 threads)", flush=True)
            # reference first: rel() normalises by its max
            pairs += [(f"{n}_vs_ld", "ld", n) for n in names]
        amp = {n: A[0] @ P for n, P in res.items()}
        for mode, a, b in pairs:
            r1, r2 = rel(res[a], res[b]), rel(amp[a], amp[b])
            rows_out.append([k, rows[0].tolist(), mode, r1, r2, round(t[a] + t[b], 2)])
            print(f"k={k:2d} rep={rep} {mode:16s} rel_max={r1:.2e}  rel_amp={r2:.2e}  ({t[a]:.1f} s + {t[b]:.1f} s)", flush=True)
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["k", "rows", "mode", "rel_max", "rel_amp", "seconds"])
            w.writerows(rows_out)
print("wrote", args.csv)
