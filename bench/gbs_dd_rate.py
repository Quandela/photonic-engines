"""The cost of double-double (dd, ~106-bit) arithmetic in the sieve's matrix products, measured against fp64 (part of
the device comparison, docs/GAUSSIAN.md §2.4).

    python3 bench/gbs_dd_rate.py check --device cuda              # dd products against exact rationals
    python3 bench/gbs_dd_rate.py rate --device cuda --q 32,64,96,128 --n 16,32,48,64 --csv gbs_dd_rate.csv

Why: the device samples need more than fp64 once a sieve has N >~ 55 vertices (the condition-number fit log10 kappa =
0.40 + 0.197 N and N eps kappa <= 1e-3; fp64 to N = 54, dd to N ~ 130). The timing model prices fp64 kernels; this
script measures the factor a dd kernel costs.

dd matrix product (Ozaki scheme): a dd matrix A = A_hi + A_lo is cut into s slices A = sum_i A_i, each A_i holding
beta = floor((53 - log2 q) / 2) bits relative to its row's maximum (B: per column), so that every fp64 product
A_i B_j over the inner dimension q is exact whatever the summation order (2 beta + log2 q <= 53 bits of one common
unit), on tensor cores too. The dd product is sum_{i + j <= s + 1} A_i B_j (the dropped terms are below 2^(-beta s)
relative), accumulated with error-free sums (TwoSum) into (hi, lo). s = ceil(107 / beta) slices: 5 at q <= 128.
Complex: four real products (re re - im im, re im + im re).

check: random dd inputs, q = 8 .. 64, compared with the exact product in Python fractions: relative error per entry
(target ~2^-104; fp64 gives ~2^-53).
rate: per q, a batch of c matrices [c, q, q]: fp64 complex batched product (`torch.bmm`, what the torch sieve path
runs) against the dd product; and, per n, the share of `hafnian._terms` (torch matrix path) spent in its batched
products, so that the dd factor of a whole term follows as share x (dd / fp64 product) + (1 - share) x (dd / fp64
element-wise), the element-wise factor measured on a complex multiply-add (TwoProd by Dekker splitting, TwoSum).
"""
import argparse
import csv
import math
import time
from fractions import Fraction

import numpy as np
import torch

SPLIT = 134217729.0                                                     # 2^27 + 1, Dekker/Veltkamp split


# ---- error-free transformations (element-wise, float64 tensors) ------------------------------
def two_sum(a, b):
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def two_prod(a, b):
    p = a * b
    ca = SPLIT * a; ah = ca - (ca - a); al = a - ah
    cb = SPLIT * b; bh = cb - (cb - b); bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    e = e + (al + bl)
    return quick_two_sum(s, e)


def dd_mul(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e = e + (ah * bl + al * bh)
    return quick_two_sum(p, e)


# ---- Ozaki dd matrix product -------------------------------------------------------------------
def _slices(hi, lo, beta, s, dim):
    """s fp64 slices of the dd matrix (hi, lo), each of beta bits relative to the max along `dim` (-1: rows of A,
    -2: columns of B). Exact: sum of slices + final remainder = hi + lo, the remainder below 2^(-beta s)."""
    out = []
    r_hi, r_lo = hi, lo
    for _ in range(s):
        m = torch.amax(torch.abs(r_hi), dim=dim, keepdim=True)
        e = torch.where(m > 0, torch.ceil(torch.log2(torch.where(m > 0, m, 1.0))), torch.zeros_like(m))
        sig = 0.75 * torch.exp2(e + (53 - beta))
        sl = (r_hi + sig) - sig                                         # top beta bits of r_hi (per row / column)
        out.append(sl)
        r_hi, r_lo = quick_two_sum(r_hi - sl, r_lo)                     # r_hi - sl is exact
    return out


def dd_matmul(ah, al, bh, bl):
    """(hi, lo) of the dd product of real dd matrices [.., p, q] x [.., q, r] (batched as torch.matmul)."""
    q = ah.shape[-1]
    beta = int((53 - math.ceil(math.log2(max(q, 2)))) // 2)
    s = math.ceil(107 / beta)
    A = _slices(ah, al, beta, s, -1); B = _slices(bh, bl, beta, s, -2)
    hi = torch.zeros(ah.shape[:-1] + bh.shape[-1:], dtype=ah.dtype, device=ah.device); lo = torch.zeros_like(hi)
    for t in range(s + 1, 1, -1):                                       # smallest terms first
        for i in range(max(1, t - s), min(s, t - 1) + 1):
            hi, lo = dd_add(hi, lo, A[i - 1] @ B[t - i - 1], 0.0)
    return hi, lo


def dd_cmatmul(a, b):
    """Complex dd product: a, b = (re_hi, re_lo, im_hi, im_lo) tuples of real tensors."""
    rr = dd_matmul(a[0], a[1], b[0], b[1]); ii = dd_matmul(a[2], a[3], b[2], b[3])
    ri = dd_matmul(a[0], a[1], b[2], b[3]); ir = dd_matmul(a[2], a[3], b[0], b[1])
    re = dd_add(rr[0], rr[1], -ii[0], -ii[1]); im = dd_add(ri[0], ri[1], ir[0], ir[1])
    return re[0], re[1], im[0], im[1]


def rand_dd(shape, rng, dev):
    hi = torch.as_tensor(rng.normal(size=shape), device=dev)
    lo = torch.as_tensor(rng.normal(size=shape), device=dev) * torch.abs(hi) * 2.0 ** -54
    return quick_two_sum(hi, lo)


# ---- check -------------------------------------------------------------------------------------
def check(args):
    dev = torch.device(args.device); rng = np.random.default_rng(1)
    worst_all = 0.0
    for q in (8, 16, 33, 64):
        ah, al = rand_dd((3, q), rng, dev); bh, bl = rand_dd((q, 4), rng, dev)
        hi, lo = dd_matmul(ah, al, bh, bl)
        f64 = (ah @ bh).cpu().numpy()
        A = [[Fraction(float(x)) + Fraction(float(y)) for x, y in zip(r1, r2)] for r1, r2 in zip(ah.cpu().numpy(), al.cpu().numpy())]
        B = [[Fraction(float(x)) + Fraction(float(y)) for x, y in zip(r1, r2)] for r1, r2 in zip(bh.cpu().numpy(), bl.cpu().numpy())]
        H, L = hi.cpu().numpy(), lo.cpu().numpy()
        worst = worst64 = 0.0
        for i in range(3):
            for j in range(4):
                ex = sum(A[i][k] * B[k][j] for k in range(q))
                scale = sum(abs(A[i][k] * B[k][j]) for k in range(q))       # relative to sum |a b|: the condition-free bound
                worst = max(worst, float(abs(Fraction(float(H[i, j])) + Fraction(float(L[i, j])) - ex) / scale))
                worst64 = max(worst64, float(abs(Fraction(float(f64[i, j])) - ex) / scale))
        worst_all = max(worst_all, worst)
        print(f"q = {q:3d}: dd error / sum|ab| {worst:.2e} (2^{math.log2(max(worst, 1e-300)):.0f}); fp64 {worst64:.2e}",
              flush=True)
    ok = worst_all < 2.0 ** -100
    print("check", "passed" if ok else "FAILED", f"(worst {worst_all:.2e}, bound 2^-100)", flush=True)
    return ok


# ---- rate --------------------------------------------------------------------------------------
def rate(args):
    import photonic_engines.gaussian.hafnian as haf
    dev = torch.device(args.device); rng = np.random.default_rng(0)
    cuda = dev.type == "cuda"

    def timed(f, reps=3):
        best = float("inf")
        for _ in range(reps):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter(); f()
            if cuda:
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return best
    rows = []
    for q in [int(x) for x in args.q.split(",")]:
        c = max(1, args.batch_elems // (q * q))
        Z = torch.as_tensor(rng.normal(size=(c, q, q)) + 1j * rng.normal(size=(c, q, q)), device=dev)
        t64 = timed(lambda Z=Z: torch.bmm(Z, Z))
        a = rand_dd((c, q, q), rng, dev) + rand_dd((c, q, q), rng, dev)
        tdd = timed(lambda a=a: dd_cmatmul(a, a))
        rows.append({"what": "product", "q": q, "n": "", "V": "", "batch": c, "fp64_s": t64, "dd_s": tdd, "ratio": tdd / t64})
        print(f"product q = {q:3d}, batch {c}: fp64 {t64 / c * 1e6:9.3f} us, dd {tdd / c * 1e6:10.3f} us, x{tdd / t64:6.1f}",
              flush=True)
    shp = (args.elem_rows, 64)
    xr, xi = (torch.as_tensor(rng.normal(size=shp), device=dev) for _ in range(2))
    Zc = torch.complex(xr, xi)
    t64 = timed(lambda: Zc * Zc + Zc)
    X = [rand_dd(shp, rng, dev) for _ in range(2)]

    def dd_cmuladd():
        (rh, rl), (ih, il) = X
        a = dd_mul(rh, rl, rh, rl); b = dd_mul(ih, il, ih, il); d = dd_mul(rh, rl, ih, il)
        re = dd_add(*dd_add(a[0], a[1], -b[0], -b[1]), rh, rl); im = dd_add(*dd_add(d[0], d[1], d[0], d[1]), ih, il)
        return re, im
    tdd = timed(dd_cmuladd)
    rows.append({"what": "elementwise", "q": "", "n": "", "V": "", "batch": shp[0] * shp[1], "fp64_s": t64, "dd_s": tdd,
                 "ratio": tdd / t64})
    print(f"element-wise complex multiply-add: fp64 {t64:.3e} s, dd {tdd:.3e} s, x{tdd / t64:.1f}", flush=True)
    # share of the torch sieve's term spent in its batched products (P^1 .. P^h, h = ceil(n/2)), at the largest bucket
    haf._TRITON = False
    for n in [int(x) for x in args.n.split(",")]:
        for V in (1, 4):
            N = 2 * n
            A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); A = (A + A.T) / (2 * math.sqrt(N))
            D0 = rng.normal(size=(V, N)) + 0j; D1 = rng.normal(size=(V, N)) + 0j
            At, D0t, D1t = (torch.as_tensor(x, device=dev) for x in (A, D0, D1))
            Ce, Ye, XYe = haf._setup(At, D0t, D1t, np.arange(N), dev)
            pb = n
            chunk = max(1, min(haf._chunk_size_multi(n, pb, V, args.budget, dev), args.max_chunk))
            pos = np.stack([np.sort(rng.choice(n, size=pb, replace=False)) for _ in range(chunk)])
            idx = torch.as_tensor(np.concatenate([pos, pos + n], 1), device=dev)
            w = torch.ones(chunk, dtype=torch.float64, device=dev)
            acc = torch.zeros((V, 2 * n + 1), dtype=torch.complex128, device=dev); acc_abs = torch.zeros_like(acc.real)
            haf._terms(Ce, Ye, XYe, idx[:2], w[:2], n, V, acc, acc_abs, True)
            tt = timed(lambda idx=idx, w=w: haf._terms(Ce, Ye, XYe, idx, w, n, V, acc, acc_abs, True))
            CS = Ce[idx.unsqueeze(2), idx.unsqueeze(1)]

            def prods(CS=CS, n=n):
                P = CS
                for _ in range(2, (n + 1) // 2 + 1):
                    P = P @ CS
            tp = timed(prods)
            rows.append({"what": "term_share", "q": 2 * pb, "n": n, "V": V, "batch": chunk, "fp64_s": tt, "dd_s": tp,
                         "ratio": tp / tt})
            print(f"term n = {n:2d} V = {V}, {chunk} subsets of {2 * pb}: {tt / chunk * 1e6:9.2f} us per term, products "
                  f"{100 * tp / tt:5.1f} %", flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            wr.writeheader(); wr.writerows(rows)
        print(f"wrote {args.csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--device", default="cpu")
    r = sub.add_parser("rate")
    r.add_argument("--device", default="cuda")
    r.add_argument("--q", default="32,64,96,128")
    r.add_argument("--n", default="16,32,48,64", help="pairs for the product share of a term (torch path)")
    r.add_argument("--batch-elems", type=int, default=1 << 22, help="matrix entries per timed batch")
    r.add_argument("--elem-rows", type=int, default=1 << 16)
    r.add_argument("--budget", type=int, default=1 << 32)
    r.add_argument("--max-chunk", type=int, default=4096)
    r.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.cmd == "check":
        raise SystemExit(0 if check(args) else 1)
    rate(args)


if __name__ == "__main__":
    main()
