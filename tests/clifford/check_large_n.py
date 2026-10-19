"""
Overnight exactness checks at LARGE n (where the full distribution is out of reach),
exercising the large-k kernel paths (many lanes per stream, L > 1, K_PAD = 32, LCHUNK
loop in the draw kernel).

  A. cofactor kernel vs an independent Ryser permanent (numpy, chunked over subsets)
     at k-1 = 14, 18, 20, 23: single-lane (L = 2^(k-2)) and many-lane plans, full
     cofactor vectors up to k-1 = 20, two cofactors at k-1 = 23.       [exact, 1e-9]
  B. second moments E[n_a n_b] vs the closed form (distinct input modes)
        E[n_a n_b] = mu_a mu_b + |M_ab|^2 - 2 W_ab + delta_ab mu_a,
        mu = rowsum |C|^2,  M = conj(C) C^T,  W = |C|^2 |C|^2T,  C = U[:, in_modes]
     (the interference term |M_ab|^2 - W_ab is what first moments cannot see);
     the closed form is itself self-checked against the exact distribution at n = 5.
     Bonferroni-corrected z over all m(m+1)/2 pairs.                   [statistical]
  C. two-sample agreement with Perceval Clifford2017Backend on the same U: per-mode
     occupation classes (0 / 1 / >=2), pair coincidences, bunching count, and a hashed
     full-outcome histogram (chi-square homogeneity).                  [statistical]

Usage:  python tests/clifford/check_large_n.py [--ns 16,20,24] [--shots 1000000,300000,100000]
        [--perceval-shots 1000000,100000,5000] [--no-perceval] [--seed 0]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import itertools
import math
import os
import sys
import time
from statistics import NormalDist

import numpy as np
import torch

from photonic_engines.clifford.cofactors_triton import laplace_cofactors
from photonic_engines.clifford.sampler import fock_sample

DEV = "cuda"


def haar(m, rng):
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def sync():
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------- A
def ryser_permanent(M, chunk=1 << 16):
    """Independent permanent: Ryser over column subsets, vectorised in chunks."""
    M = np.asarray(M)
    k = M.shape[0]
    if k == 0:
        return 1.0 + 0j
    total = 0j
    nsub = 1 << k
    bits = np.arange(k)
    for s0 in range(0, nsub, chunk):
        sub = np.arange(s0, min(nsub, s0 + chunk), dtype=np.int64)
        ind = ((sub[:, None] >> bits[None, :]) & 1).astype(np.float64)   # [chunk, k]
        rs = ind @ M.T                                                   # [chunk, k] row sums
        sign = np.where(ind.sum(1) % 2 == 0, 1.0, -1.0)
        total += (sign * np.prod(rs, axis=1)).sum()
    return total * (-1) ** k


def check_kernel(rng):
    print("=== A. cofactor kernel vs independent Ryser (large k paths) ===")
    ok = True
    for K1, ncof in [(14, None), (18, None), (20, None), (23, 2)]:
        K = K1 + 1
        B = rng.normal(size=(1, K1, K)) + 1j * rng.normal(size=(1, K1, K))
        Bt = torch.tensor(B, device=DEV)
        t = time.time()
        P1 = laplace_cofactors(Bt, max_lanes=1).cpu().numpy()[0]              # one lane: L = 2^(K1-1)
        P2 = laplace_cofactors(Bt).cpu().numpy()[0]                           # default plan (up to 2^16 lanes)
        P3 = laplace_cofactors(Bt, max_lanes=64).cpu().numpy()[0]             # intermediate
        sync(); dt = time.time() - t
        cols = range(K) if ncof is None else [0, K - 1]
        t = time.time()
        ref = {c: ryser_permanent(np.delete(B[0], c, axis=1)) for c in cols}
        dref = time.time() - t
        scale = max(abs(v) for v in ref.values())
        e1 = max(abs(P1[c] - ref[c]) for c in cols) / scale
        e2 = max(abs(P2[c] - ref[c]) for c in cols) / scale
        e3 = max(abs(P3[c] - ref[c]) for c in cols) / scale
        e12 = np.abs(P1 - P2).max() / np.abs(P2).max()
        good = max(e1, e2, e3) < 1e-9          # Glynn cancellation grows ~4x per +2 in k (8e-14 at k-1=14, 4e-11 at 23)
        ok &= good
        print(f"  k-1={K1:2d}: rel.err vs Ryser  1-lane {e1:.1e}  default {e2:.1e}  64-lanes {e3:.1e}"
              f"   (1-lane vs default {e12:.1e})  [{len(list(cols))} cofactors, kernel {dt:.2f}s, ref {dref:.1f}s]"
              f"  {'OK' if good else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- B
def second_moments_exact(U, in_modes):
    C = np.asarray(U)[:, list(in_modes)]
    A2 = np.abs(C) ** 2
    mu = A2.sum(1)
    M = np.conj(C) @ C.T
    W = A2 @ A2.T
    return np.outer(mu, mu) + np.abs(M) ** 2 - 2 * W + np.diag(mu), mu


def _exact_dist(U, in_modes):
    from check_cofactors import per_ryser
    m = U.shape[0]; n = len(in_modes)
    A = U[:, in_modes]
    dist = {}
    for t in itertools.combinations_with_replacement(range(m), n):
        occ = np.bincount(t, minlength=m)
        p = abs(per_ryser(A[list(t), :])) ** 2 / np.prod([math.factorial(c) for c in occ])
        dist[tuple(occ)] = p
    return dist


def selfcheck_second_moments(rng):
    U = haar(8, rng); in_modes = [0, 1, 2, 3, 4]
    dist = _exact_dist(U, in_modes)
    states = np.array(list(dist.keys()), dtype=float); p = np.array(list(dist.values()))
    E2 = (states * p[:, None]).T @ states
    G, mu = second_moments_exact(U, in_modes)
    err = np.abs(E2 - G).max()
    print(f"  closed form vs exact distribution (n=5, m=8): max|diff| = {err:.1e}  {'OK' if err < 1e-10 else 'FAIL'}")
    return err < 1e-10


def check_second_moments(U, n, occ, label):
    m = U.shape[0]
    G, mu = second_moments_exact(U, list(range(n)))
    N = occ.shape[0]
    X = occ.astype(np.float64)
    E2 = X.T @ X / N
    # std of the estimator of E[n_a n_b] from the empirical variance of n_a n_b,
    # var = E[(n_a n_b)^2] - E[n_a n_b]^2 with E[(n_a n_b)^2] = (X^2)^T (X^2) / N  (no [N, m, m] tensor)
    X2 = X * X
    var = X2.T @ X2 / N - E2 * E2
    sd = np.sqrt(np.maximum(var, 0) / N)                             # [m, m]
    sd = np.where(sd > 0, sd, 1e-300)
    z = (E2 - G) / sd
    iu = np.triu_indices(m)
    zz = z[iu]
    npairs = len(zz)
    zmax = np.abs(zz).max()
    zthr = NormalDist().inv_cdf(1 - 0.005 / npairs)
    # first moments as well
    z1 = (X.mean(0) - mu) / (X.std(0) / math.sqrt(N))
    z1max = np.abs(z1).max(); z1thr = NormalDist().inv_cdf(1 - 0.005 / m)
    # interference share: how big is the exchange term relative to the classical part
    inter = np.abs(np.abs(np.conj(U[:, :n]) @ U[:, :n].T) ** 2 - (np.abs(U[:, :n]) ** 2 @ (np.abs(U[:, :n]) ** 2).T))[iu].max()
    good = zmax < zthr and z1max < z1thr
    print(f"  {label}: N={N}  first moments max|z|={z1max:.2f} (thr {z1thr:.2f})   "
          f"second moments max|z|={zmax:.2f} over {npairs} pairs (thr {zthr:.2f}), "
          f"mean z^2={np.mean(zz**2):.3f} (expect 1)   [max interference term {inter:.2e}, "
          f"resolution sd~{np.median(sd[iu]):.1e}]  {'OK' if good else 'FAIL'}")
    return good


# ----------------------------------------------------------------------------- C
def perceval_samples(U, n, shots):
    import exqalibur as xq
    import perceval as pcvl
    from perceval.backends import Clifford2017Backend
    xq.Config.compute_max_thread_count = os.cpu_count() or 1
    m = U.shape[0]
    b = Clifford2017Backend()
    b.set_circuit(pcvl.Unitary(pcvl.Matrix(U)))
    b.set_input_state(pcvl.BasicState([1] * n + [0] * (m - n)))
    step = 1 << 16
    out = [np.array([list(st) for st in b.samples(min(step, shots - s0))], dtype=np.int64)
           for s0 in range(0, shots, step)]
    return np.concatenate(out)


def feature_stats(occ):
    """(mean, var) over samples of: per-mode classes (0 / 1 / >=2), pair products n_a n_b (a<b),
    bunching count, max occupation — pair products via matrix products (no [N, m, m] tensor)."""
    m = occ.shape[1]; N = occ.shape[0]
    X = occ.astype(np.float64)
    direct = np.concatenate([(occ == 0), (occ == 1), (occ >= 2),
                             (occ >= 2).sum(1, keepdims=True), occ.max(1, keepdims=True)], axis=1).astype(np.float64)
    iu = np.triu_indices(m, 1)
    E2 = (X.T @ X / N)[iu]
    X2 = X * X
    V2 = (X2.T @ X2 / N)[iu] - E2 * E2
    mean = np.concatenate([direct.mean(0), E2])
    var = np.concatenate([direct.var(0), np.maximum(V2, 0)])
    return mean, var, N


def two_sample_z(a, b):
    (mx, vx, nx), (my, vy, ny) = a, b
    sd = np.sqrt(vx / nx + vy / ny)
    sd = np.where(sd > 0, sd, 1e-300)
    return (mx - my) / sd


def check_vs_perceval(occ_t, occ_p, label):
    m = occ_t.shape[1]
    z = two_sample_z(feature_stats(occ_t), feature_stats(occ_p))
    nf = z.shape[0]
    zthr = NormalDist().inv_cdf(1 - 0.005 / nf)
    zmax = np.nanmax(np.abs(z))
    # hashed full-outcome histogram, chi-square homogeneity
    nb = 2048
    def hashes(occ):
        h = np.zeros(occ.shape[0], dtype=np.int64)
        for j in range(m):
            h = (h * 1000003 + occ[:, j] * 7919) % 2147483647
        return h % nb
    ht = np.bincount(hashes(occ_t), minlength=nb).astype(float)
    hp = np.bincount(hashes(occ_p), minlength=nb).astype(float)
    Nt, Np = occ_t.shape[0], occ_p.shape[0]
    tot = ht + hp
    mask = tot > 0
    et = tot[mask] * Nt / (Nt + Np); ep = tot[mask] * Np / (Nt + Np)
    chi2 = ((ht[mask] - et) ** 2 / et + (hp[mask] - ep) ** 2 / ep).sum()
    dof = mask.sum() - 1
    zchi = (chi2 - dof) / math.sqrt(2 * dof)
    good = zmax < zthr and abs(zchi) < 4
    print(f"  {label}: N_triton={Nt} N_perceval={Np}  features max|z|={zmax:.2f} over {nf} (thr {zthr:.2f}), "
          f"mean z^2={np.nanmean(z**2):.3f}   hashed-outcome chi2 z={zchi:+.2f} (dof {dof})  {'OK' if good else 'FAIL'}")
    return good


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="16,20,24")
    ap.add_argument("--shots", default="1000000,300000,100000")
    ap.add_argument("--perceval-shots", default="1000000,100000,5000")
    ap.add_argument("--no-perceval", action="store_true")
    ap.add_argument("--skip-kernel", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algos", default="2018,2020", help="sampler kernels to check (comma list)")
    ap.add_argument("--backend", default="triton", help="triton | cuda (kernel for the 2020 steps)")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    shots = [int(x) for x in args.shots.split(",")]
    pshots = [int(x) for x in args.perceval_shots.split(",")]
    rng = np.random.default_rng(args.seed)
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cpu_count {os.cpu_count()}")
    results = {}

    if not args.skip_kernel:
        results["A.kernel"] = check_kernel(rng)
        sys.stdout.flush()

    print("=== B/C. sampler at large n: second moments (closed form) and Perceval cross-check ===")
    results["B.selfcheck"] = selfcheck_second_moments(rng)
    algos = args.algos.split(",")
    for n, N, Np in zip(ns, shots, pshots):
        m = 2 * n
        U = haar(m, rng)
        occs = {}
        for ai, algo in enumerate(algos):
            # distinct seeds per algorithm: with the same seed both kernels reproduce the same
            # samples exactly (their weights agree to ~1e-14), which the two-sample test would flag
            gen = torch.Generator(device=DEV).manual_seed(args.seed + n + 7919 * ai)
            sync(); t = time.time()
            occs[algo] = fock_sample(torch.tensor(U, device=DEV), list(range(n)), N, generator=gen, algo=algo,
                                         backend=args.backend).cpu().numpy()
            sync(); dt = time.time() - t
            print(f"n={n} m={m}: {args.backend}[{algo}] {N} samples in {dt:.1f}s ({dt/N*1e3:.3f} ms/sample)")
            results[f"B.n{n}.{algo}"] = check_second_moments(U, n, occs[algo], f"n={n} {args.backend}[{algo}] vs closed form")
            sys.stdout.flush()
        if len(algos) == 2:
            a, b = algos
            results[f"C.n{n}.{a}-vs-{b}"] = check_vs_perceval(occs[a], occs[b], f"n={n} {args.backend}[{a}] vs {args.backend}[{b}]")
        if not args.no_perceval and Np > 0:
            t = time.time()
            occ_p = perceval_samples(U, n, Np)
            dp = time.time() - t
            print(f"n={n} m={m}: perceval {Np} samples in {dp:.1f}s ({dp/Np*1e3:.3f} ms/sample)")
            results[f"B.n{n}.perceval"] = check_second_moments(U, n, occ_p, f"n={n} perceval vs closed form")
            for algo in algos:
                results[f"C.n{n}.{algo}"] = check_vs_perceval(occs[algo], occ_p, f"n={n} {args.backend}[{algo}] vs perceval")
            sys.stdout.flush()

    print("=== summary ===")
    for k, v in results.items():
        print(f"  {k:18s} {'OK' if v else 'FAIL'}")
    print("ALL OK" if all(results.values()) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
