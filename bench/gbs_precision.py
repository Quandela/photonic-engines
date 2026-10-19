"""Rounding error of the pair sieve against an extended-precision reference, and how the choice of vertex
pairing sets it (docs/GAUSSIAN.md §3.4, "Precision, open"). CPU only.

    python3 bench/gbs_precision.py --n 16,20,24 --shapes free,2x2,1x4,2x2x2 --pairings v1,self,random:8 --csv prec.csv

The sieve lhaf(A, D) = sum_S (-1)^(n - |S|) [lambda^n] exp(...) over subsets S of a perfect matching of the
vertices into n pairs; any matching gives the same value, not the same rounding. v1 (`lhaf_poly`) pairs
vertex i with i + n, so running it on A[perm][:, perm] realises any pairing. For each matrix and pairing this
reports
  err   = max_k |c_k - ref_k| / max|ref|, the actual float64 error, ref = the same sieve in long double
          (numpy clongdouble, 64-bit mantissa: eps 1.1e-19 on x86-64; checked against lhaf_brute at N <= 10);
  kappa = sum_S |term_S| / max|c| of that pairing (`return_abs`), and err / (eps kappa).
Pairings: v1 (i with i + n on the given order), self (identical vertices of a repeated mode paired together,
as v2's collision grouping does), random:K (K random matchings: min / median / max), and v2 (`lhaf_poly_multi`
with the labels, its own pairing and collision grouping).
Matrices: B[N] of m = 2k squeezers at 0.5 photons each through a Haar U, expanded with the shape's counts, loop
vectors as in bench/profile_gbs.py.
"""
import argparse
import csv

import numpy as np

from photonic_engines import GaussianInput
from photonic_engines.gaussian.hafnian import _pairing, expand, lhaf_brute, lhaf_poly, lhaf_poly_multi

EPS = float(np.finfo(np.float64).eps)
LD = np.clongdouble


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def counts_for(N, shape):
    if shape == "free":
        return [1] * N
    table = {"2x2": [2, 2], "1x4": [4], "2x2x2": [2, 2, 2], "1x6": [6], "1x3": [3]}
    head = table.get(shape)
    if head is None:
        raise SystemExit(f"unknown shape {shape}")
    return head + [1] * (N - sum(head))


def lhaf_poly_ld(A, D0, D1, chunk=4096):
    """The v1 sieve (pairs i, i + n) in numpy clongdouble: coefficients [N + 1] of z -> lhaf(A, D0 + z D1)."""
    A = np.asarray(A, dtype=LD); D0 = np.asarray(D0, dtype=LD); D1 = np.asarray(D1, dtype=LD)
    N = A.shape[0]
    if N % 2:
        Ap = np.zeros((N + 1, N + 1), dtype=LD); Ap[:N, :N] = A; A = Ap
        D0 = np.concatenate([D0, [LD(1)]]); D1 = np.concatenate([D1, [LD(0)]])
    n = A.shape[0] // 2; L = 2 * n + 1
    X = np.concatenate([np.arange(n, 2 * n), np.arange(0, n)])
    C = A[:, X]; XD0, XD1 = D0[X], D1[X]
    acc = np.zeros(L, dtype=LD)
    bits = np.arange(n)
    for start in range(0, 1 << n, chunk):
        S = np.arange(start, min(1 << n, start + chunk))
        s = ((S[:, None] >> bits) & 1).astype(np.longdouble)
        s = np.concatenate([s, s], 1).astype(LD)                           # [c, 2n]
        CS = s[:, :, None] * C[None] * s[:, None, :]
        d0, d1, x0, x1 = s * D0, s * D1, s * XD0, s * XD1
        c = len(S)
        g = np.zeros((c, n + 1, 3), dtype=LD)
        P = CS; w0, w1 = d0, d1
        for k in range(1, n + 1):
            if k > 1:
                P = np.matmul(P, CS)
                w0 = np.einsum("cij,cj->ci", CS, w0); w1 = np.einsum("cij,cj->ci", CS, w1)
            tr = np.einsum("cii->c", P)
            g[:, k, 0] = tr / (2 * k) + 0.5 * (x0 * w0).sum(1)
            g[:, k, 1] = (x0 * w1).sum(1)
            g[:, k, 2] = 0.5 * (x1 * w1).sum(1)
        F = [np.zeros((c, L), dtype=LD) for _ in range(n + 1)]
        F[0][:, 0] = 1
        for k in range(1, n + 1):
            t = np.zeros((c, L), dtype=LD)
            for i in range(1, k + 1):
                f = F[k - i]; gi = g[:, i]
                t += i * gi[:, 0:1] * f
                t[:, 1:] += i * gi[:, 1:2] * f[:, :-1]
                t[:, 2:] += i * gi[:, 2:3] * f[:, :-2]
            F[k] = t / k
        sign = np.where((n - s[:, :n].real.sum(1).round().astype(np.int64)) % 2 == 1, -1, 1).astype(np.longdouble)
        acc += (sign[:, None] * F[n]).sum(0)
    return acc[: N + 1]


def matching_perm(pairs, N):
    """Vertex order that makes v1 (pairs i, i + n) realise the matching `pairs` [(u, v), ...] of N vertices."""
    firsts = [u for u, _ in pairs]; seconds = [v for _, v in pairs]
    return np.array(firsts + seconds, dtype=np.int64)


def self_pairs(lab):
    perm, _ = _pairing(lab, len(lab))
    n = len(perm) // 2
    return [(int(perm[i]), int(perm[i + n])) for i in range(n)]


def run_pairing(A, D0, D1, pairs, ref):
    N = A.shape[0]
    if N % 2:                                                              # pad as the sieve does
        Ap = np.zeros((N + 1, N + 1), dtype=A.dtype); Ap[:N, :N] = A; A = Ap
        D0 = np.concatenate([D0, [1.0]]); D1 = np.concatenate([D1, [0.0]])
    perm = matching_perm(pairs, A.shape[0])
    c, a = lhaf_poly(A[np.ix_(perm, perm)], D0[perm], D1[perm], return_abs=True)
    c = c[: N + 1].numpy(); a = a[: N + 1].numpy()
    return err_of(c, ref), float(a.max() / np.abs(c).max())


def err_of(c, ref):
    r = np.asarray(ref, dtype=LD)
    return float(np.max(np.abs(np.asarray(c, dtype=LD) - r)) / np.max(np.abs(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="12,16,20,24")
    ap.add_argument("--shapes", default="free,2x2,1x4,2x2x2")
    ap.add_argument("--pairings", default="v1,self,v2,random:8")
    ap.add_argument("--matrices", type=int, default=2, help="random matrices per (N, shape)")
    ap.add_argument("--mean", type=float, default=0.5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    print(f"long double eps {float(np.finfo(np.longdouble).eps):.1e}, float64 eps {EPS:.1e}", flush=True)

    # the reference against brute force first
    rng0 = np.random.default_rng(1)
    for N in (4, 7, 10):
        M = rng0.normal(size=(N, N)) + 1j * rng0.normal(size=(N, N)); M = (M + M.T) / 2
        d0 = rng0.normal(size=N) + 1j * rng0.normal(size=N); d1 = rng0.normal(size=N) + 0j
        ref = lhaf_poly_ld(M, d0, d1)
        for z in (0.0, 0.7, -0.4j):
            val = complex(np.polynomial.polynomial.polyval(z, ref.astype(np.complex128)))
            bru = lhaf_brute(M, d0 + z * d1)
            assert abs(val - bru) <= 1e-11 * max(1.0, abs(bru)), (N, z, val, bru)
    print("reference sieve == lhaf_brute at N = 4, 7, 10", flush=True)

    rows = []
    for N in (int(x) for x in args.n.split(",")):
        for shape in args.shapes.split(","):
            cnt = counts_for(N, shape); k = len(cnt); m = 2 * k
            for mi in range(args.matrices):
                st = GaussianInput.smsv(k, m, GaussianInput.r_for_mean(args.mean))
                B, _, _ = st.bargmann(haar(m, 1000 * mi + N))
                rng = np.random.default_rng(1000 * mi + N)
                gam = rng.normal(size=k) * 0.5 + 0j
                A, D0 = expand(B[:k, :k], gam, cnt); lab = np.repeat(np.arange(k), cnt)
                D1 = B[:k, k][lab].copy()
                ref = lhaf_poly_ld(A, D0, D1)
                out = []
                for pz in args.pairings.split(","):
                    if pz == "v1":
                        pairs = [(i, i + (N + N % 2) // 2) for i in range((N + N % 2) // 2)]
                        e, kap = run_pairing(A, D0, D1, pairs, ref); res = [(pz, e, kap)]
                    elif pz == "self":
                        e, kap = run_pairing(A, D0, D1, self_pairs(lab), ref); res = [(pz, e, kap)]
                    elif pz == "v2":
                        c, a = lhaf_poly_multi(A, D0[None], D1[None], labels=lab, return_abs=True)
                        c = c[0].numpy(); a = a[0].numpy()
                        res = [(pz, err_of(c, ref), float(a.max() / np.abs(c).max()))]
                    elif pz.startswith("random"):
                        K = int(pz.split(":")[1]); prng = np.random.default_rng(7 + mi)
                        vals = []
                        for _ in range(K):
                            order = prng.permutation(N + N % 2); h = len(order) // 2
                            vals.append(run_pairing(A, D0, D1, list(zip(order[:h], order[h:])), ref))
                        vals.sort(key=lambda t: t[1])
                        res = [("random-min", *vals[0]), ("random-med", *vals[len(vals) // 2]), ("random-max", *vals[-1])]
                    for name, e, kap in res:
                        out.append(f"{name} err {e:.1e} kappa {kap:.1e} ({e / (EPS * kap):.2f})")
                        rows.append([N, shape, mi, name, e, kap, e / (EPS * kap)])
                print(f"  N={N:2d} {shape:5s} #{mi}: " + " | ".join(out), flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["N", "shape", "matrix", "pairing", "err", "kappa", "err_over_eps_kappa"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
