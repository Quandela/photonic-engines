"""Depth below which the treewidth sampler (Oh et al. 2022) beats the chain rule: D* at one (n, m).

Their exact per-sample cost on a depth-D local mesh is t_TW = M N^2 w^2 2^w / R (one permanent of width w per
candidate) or N^2 w^2 2^w / R when the M candidates share one pass, with treewidth w = 2D/c + 1 at m = cn
(supplement, Sec. samplers). R is calibrated on our own kernel, generous to them:
R = n rho^n / t_ours, so R t_ours = n rho^n and the crossing does not depend on the measured rate, only on rho.
Gaussian inputs (squeezed point): theirs / ours = M N w^2 2^w / (N^3 2^(N/2)), window density N/M,
w = (2D + 1) N/M.

    python3 bench/treewidth_crossover.py                            # Fock n = 38, m = 76, rho 1.816; squeezed N = 58, M = 82
    python3 bench/treewidth_crossover.py --n 38 --m 76 --rho 1.804
"""
import argparse
import math


def solve(f, lo=1.0, hi=80.0):
    assert f(lo) < 0 < f(hi), "no crossing in the bracket"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=38)
    ap.add_argument("--m", type=int, default=76)
    ap.add_argument("--rho", type=float, default=1.816, help="per-photon cost factor of the chain rule (m = 2n fit of the 256-shot runs)")
    ap.add_argument("--gbs-N", type=int, default=58, help="detected photons of the squeezed tail stratum")
    ap.add_argument("--gbs-M", type=int, default=82)
    a = ap.parse_args()
    n, m, c = a.n, a.m, a.m / a.n
    target = n * a.rho ** n
    print(f"Fock n = {n}, m = {m}, rho = {a.rho}: R t_ours = n rho^n = {target:.3g}")
    for label, pref in (("one pass per candidate", m * n * n), ("candidates share one pass", n * n)):
        w = solve(lambda w: math.log(pref * w * w) + w * math.log(2) - math.log(target))
        print(f"  {label:26s} w* = {w:5.1f}  D* = {(w - 1) * c / 2:5.1f} layers")
    w = n * math.log2(a.rho)
    print(f"  exponent only (2^w < rho^n)  w* = {w:5.1f}  D* = {(w - 1) * c / 2:5.1f} layers")
    N, M = a.gbs_N, a.gbs_M
    rhs = math.log(N * N) + (N / 2) * math.log(2) - math.log(M)
    w = solve(lambda w: math.log(w * w) + w * math.log(2) - rhs)
    print(f"squeezed N = {N}, M = {M}: w* = {w:5.1f}  D* = {(w * M / N - 1) / 2:5.1f} layers;"
          f" exponent only (w < N/2): D* = {(N / 2 * M / N - 1) / 2:5.1f}")


if __name__ == "__main__":
    main()
