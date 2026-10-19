"""CPU prototype of the O(q^3)-per-subset sieve (docs/GAUSSIAN.md §8): the trace part of every term from
det(I - lam C_S)^(-1/2) (batched Householder Hessenberg + the Hessenberg characteristic-polynomial recurrence) instead
of matrix powers, the paths by matrix-vector products. Precision against the long-double sieve of gbs_precision.py.

    python3 bench/gbs_charpoly_proto.py 8,12,16,20,24,28
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbs_precision import counts_for, haar, lhaf_poly_ld

from photonic_engines import GaussianInput
from photonic_engines.gaussian.hafnian import expand, lhaf_poly


def hessenberg(H):                       # batched Householder, H [c, q, q] complex, in place
    c, q, _ = H.shape
    for j in range(q - 2):
        x = H[:, j + 1:, j].copy()                                  # [c, q-j-1]
        nx = np.linalg.norm(x, axis=1)
        a0 = x[:, 0]
        ph = np.where(np.abs(a0) > 0, a0 / np.where(np.abs(a0) > 0, np.abs(a0), 1), 1.0)
        v = x.copy(); v[:, 0] += ph * nx
        nv = np.linalg.norm(v, axis=1)
        ok = nv > 1e-300
        v[ok] /= nv[ok, None]; v[~ok] = 0
        # H <- (I - 2 v v^H) H (I - 2 v v^H) on the trailing block rows / cols
        sub = H[:, j + 1:, :]
        H[:, j + 1:, :] = sub - 2 * v[:, :, None] * np.einsum("ci,cij->cj", v.conj(), sub)[:, None, :]
        sub = H[:, :, j + 1:]
        H[:, :, j + 1:] = sub - 2 * np.einsum("cij,cj->ci", sub, v)[:, :, None] * v.conj()[:, None, :]
    return H

def charpoly_hess(H):                    # det(t I - H) coefficients [c, q+1] (index = power), batched
    c, q, _ = H.shape
    P = [np.ones((c, 1), dtype=complex)]
    for k in range(1, q + 1):
        hk = H[:, k - 1, k - 1]
        pk = np.zeros((c, k + 1), dtype=complex)
        pk[:, 1:] += P[k - 1]; pk[:, :k] -= hk[:, None] * P[k - 1]
        prod = np.ones(c, dtype=complex)
        for i in range(k - 1, 0, -1):                               # i = k-1 .. 1 (1-based rows)
            prod = prod * H[:, i, i - 1]                            # h_{i+1, i}
            coef = H[:, i - 1, k - 1] * prod
            pk[:, :i] -= coef[:, None] * P[i - 1]
        P.append(pk)
    return P[q]

def lhaf_charpoly(A, D0, D1):
    A = np.asarray(A, complex); D0 = np.asarray(D0, complex); D1 = np.asarray(D1, complex)
    N = A.shape[0]
    if N % 2:
        Ap = np.zeros((N + 1, N + 1), complex); Ap[:N, :N] = A; A = Ap
        D0 = np.concatenate([D0, [1]]); D1 = np.concatenate([D1, [0]])
    n = A.shape[0] // 2; L = 2 * n + 1
    X = np.concatenate([np.arange(n, 2 * n), np.arange(n)]); C = A[:, X]; XD0, XD1 = D0[X], D1[X]
    S = np.arange(1 << n); s = ((S[:, None] >> np.arange(n)) & 1).astype(float); s = np.concatenate([s, s], 1)
    CS = s[:, :, None] * C[None] * s[:, None, :]
    c = len(S)
    # trace part: T(lam) = det(I - lam C)^(-1/2), degree <= n
    cp = charpoly_hess(hessenberg(CS.copy()))                      # det(t - C) = sum cp_j t^j, q = 2n
    Pl = cp[:, ::-1][:, :n + 1]                                     # det(I - lam C) coefficients a_0..a_n
    T = np.zeros((c, n + 1), complex); T[:, 0] = 1
    for mm in range(1, n + 1):
        acc = np.zeros(c, complex)
        for i in range(1, mm + 1):
            acc += (-0.5 * i - (mm - i)) * Pl[:, i] * T[:, mm - i]
        T[:, mm] = acc / mm
    # loop part g_k(z) = 0.5 (x0.w0 + z (x0.w1 + x1.w0) + z^2 x1.w1) with w = C^(k-1) (d), as the power-sieve code
    d0, d1, x0, x1 = s * D0, s * D1, s * XD0, s * XD1
    g = np.zeros((c, n + 1, 3), complex); w0, w1 = d0, d1
    for k in range(1, n + 1):
        if k > 1:
            w0 = np.einsum("cij,cj->ci", CS, w0); w1 = np.einsum("cij,cj->ci", CS, w1)
        g[:, k, 0] = 0.5 * (x0 * w0).sum(1); g[:, k, 1] = (x0 * w1).sum(1); g[:, k, 2] = 0.5 * (x1 * w1).sum(1)
    F = [np.zeros((c, L), complex) for _ in range(n + 1)]; F[0][:, 0] = 1
    for k in range(1, n + 1):
        t = np.zeros((c, L), complex)
        for i in range(1, k + 1):
            f = F[k - i]; gi = g[:, i]
            t += i * gi[:, 0:1] * f; t[:, 1:] += i * gi[:, 1:2] * f[:, :-1]; t[:, 2:] += i * gi[:, 2:3] * f[:, :-2]
        F[k] = t / k
    term = np.zeros((c, L), complex)                                # [lam^n] T(lam) * exp(loop)(lam, z)
    for i in range(n + 1):
        term += T[:, i:i + 1] * F[n - i]
    sign = np.where((n - s[:, :n].sum(1).astype(int)) % 2 == 1, -1.0, 1.0)
    return (sign[:, None] * term).sum(0)[:N + 1], np.abs(term).sum(0)[:N + 1]

if __name__ != "__main__":
    raise SystemExit
for N in [int(x) for x in sys.argv[1].split(",")]:
    for shape in ("free", "1x4"):
        for mi in range(2):
            cnt = counts_for(N, shape); k = len(cnt); m = 2 * k
            st = GaussianInput.smsv(k, m, GaussianInput.r_for_mean(0.5)); B, _, _ = st.bargmann(haar(m, 1000 * mi + N))
            rng = np.random.default_rng(1000 * mi + N); gam = rng.normal(size=k) * 0.5 + 0j
            A, D0 = expand(B[:k, :k], gam, cnt); lab = np.repeat(np.arange(k), cnt); D1 = B[:k, k][lab].copy()
            ref = lhaf_poly_ld(A, D0, D1).astype(np.clongdouble)
            e = lambda v: float(np.max(np.abs(np.asarray(v, dtype=np.clongdouble) - ref)) / np.max(np.abs(ref)))
            pw, pa = lhaf_poly(A, D0, D1, return_abs=True)
            ch, ca = lhaf_charpoly(A, D0, D1)
            kp = float(pa.max() / pw.abs().max()); kc = float(ca.max() / np.abs(ch).max())
            print(f"N={N} {shape} #{mi}: powers err {e(pw.numpy()):.1e} (kappa {kp:.1e})   "
                  f"charpoly err {e(ch):.1e} (kappa {kc:.1e})", flush=True)
