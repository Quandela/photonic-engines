"""Triton kernel for the matrix part of the v2 pair sieve (docs/GAUSSIAN.md §3.4, v3 step "fused kernel").

One program per subset S: gather C_S = Ce[idx_S, idx_S] once into registers (complex as two float64 planes,
padded to Q = 16 / 32 / 64 with the dummy vertex, whose row and column are zero), then for k = 1 .. n

    paths  a_k = x0^T C^(k-1) y0,  b_k = x0^T C^(k-1) y1,  c_k = x1^T C^(k-1) y1     (per loop vector v)
    power  C^k = C^(k-1) C_S           (four real tl.dot per step, float64)
    trace  tr C^k

and write tr [c, n+1] and the paths [c, V, n+1, 3] (complex, interleaved re / im). `hafnian._terms` turns them
into g_k, and `series` (a second kernel, one program per subset and loop vector) runs the recursion
F_k = (1/k) sum_i i g_i(z) F_(k-i) in a per-program scratch row block and returns F_n. Nothing of C_S or its powers goes back to global
memory: in the torch layouts every one of the ~n matrix-vector products re-read C_S from it, which made the
large sieves memory-bound (gemv the top kernel at N = 36, run 20260924_162756).

Cost per subset: n complex Q x Q x Q products and n complex Q x Q x VP ones for the paths (v2.8; see the kernel).
v2.9 (consecutive powers, ceil(n/2) - 1 products) was measured slower and removed (docs/GAUSSIAN.md §3.4).
Imported only when the sieve is asked to use it.
"""
import torch
import triton
import triton.language as tl


# do_not_specialize: Triton recompiles for every integer argument that is 1 or divisible by 16, which cost ~1 s at
# the first sieve of N = 32 / 36 / 48 in run 20260924_180802; one compilation per (V, Q) instead
# do_not_specialize: see above. v2.8: the paths of all V loop vectors as ONE [Q, Q] x [Q, VP] tl.dot per step (VP =
# columns for the 3V (x, y) pairs, padded to 16) and one column-wise reduction, instead of 2V matrix-vector
# reductions over the rows. v2.7 (even powers only, fewer products but more matrix-vector reductions) was slower on the
# H100 - one sieve at N = 48, V = 4: 42.7 s vs 27.4 s for v2.6 (run 20260924_195027) - so the products were not the cost.
# v2.8 measured (run 20260924_202647): V = 4 x1.4 over v2.6 (19.2 vs 27.4 s at N = 48), V = 1 x0.9 (12.9 vs 11.8 s; three
# useful columns of 16), hence the V = 1 branch keeps v2.6's two matrix-vector reductions.
@triton.jit(do_not_specialize=["E", "q", "n"])
def _powers_paths_kernel(cr_ptr, ci_ptr,              # C extended, [E, E] float64 planes
                         yr_ptr, yi_ptr,              # Ye [E, 2V] planes: D0 columns then D1
                         xr_ptr, xi_ptr,              # XYe [E, 2V] planes
                         idx_ptr,                     # [c, q] int64 vertex lists (padded with E - 1)
                         tr_ptr,                      # out [c, n + 1, 2] float64
                         pa_ptr,                      # out [c, V, n + 1, 3, 2] float64
                         E, q, n,
                         V: tl.constexpr, Q: tl.constexpr, VP: tl.constexpr,
                         DO_POWER: tl.constexpr = True, DO_TRACE: tl.constexpr = True, DO_PATHS: tl.constexpr = True):
    # DO_*: cost-split switches for bench/bench_sieve_kernel.py only (the sieve always runs with all three).
    # DO_POWER = False starts P at C, not at the identity (which the compiler folded), and with DO_TRACE = DO_PATHS =
    # False the sum of P is stored at k = 0 so that the products are not dead code (run 20260924_214308 had neither).
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, Q)
    ids = tl.load(idx_ptr + pid * q + offs, mask=offs < q, other=E - 1).to(tl.int64)
    cptr = ids[:, None] * E + ids[None, :]
    cr = tl.load(cr_ptr + cptr)
    ci = tl.load(ci_ptr + cptr)
    # column c < 3V: pair kind c // V (0: x0^T . y0, 1: x0^T . y1, 2: x1^T . y1) of loop vector c % V
    cols = tl.arange(0, VP)
    kind = cols // V
    vv = cols % V
    cm = cols < 3 * V
    W2 = 2 * V
    ycol = tl.where(kind == 0, vv, V + vv)
    xcol = tl.where(kind == 2, V + vv, vv)
    yptr = ids[:, None] * W2 + ycol[None, :]
    xptr = ids[:, None] * W2 + xcol[None, :]
    Yr = tl.load(yr_ptr + yptr, mask=cm[None, :], other=0.0)
    Yi = tl.load(yi_ptr + yptr, mask=cm[None, :], other=0.0)
    Xr = tl.load(xr_ptr + xptr, mask=cm[None, :], other=0.0)
    Xi = tl.load(xi_ptr + xptr, mask=cm[None, :], other=0.0)
    eye = offs[:, None] == offs[None, :]
    if DO_POWER:
        pr = tl.where(eye, 1.0, 0.0).to(tl.float64)                # C^0
        pi = tl.zeros([Q, Q], dtype=tl.float64)
    else:
        pr = cr
        pi = ci
    for k in range(1, n + 1):
        if not DO_PATHS:
            pass
        elif V == 1:                                                # two matrix-vector reductions beat a 16-wide product
            y0r = tl.load(yr_ptr + ids * W2); y0i = tl.load(yi_ptr + ids * W2)
            y1r = tl.load(yr_ptr + ids * W2 + 1); y1i = tl.load(yi_ptr + ids * W2 + 1)
            x0r = tl.load(xr_ptr + ids * W2); x0i = tl.load(xi_ptr + ids * W2)
            x1r = tl.load(xr_ptr + ids * W2 + 1); x1i = tl.load(xi_ptr + ids * W2 + 1)
            u0r = tl.sum(pr * y0r[None, :] - pi * y0i[None, :], axis=1)
            u0i = tl.sum(pr * y0i[None, :] + pi * y0r[None, :], axis=1)
            u1r = tl.sum(pr * y1r[None, :] - pi * y1i[None, :], axis=1)
            u1i = tl.sum(pr * y1i[None, :] + pi * y1r[None, :], axis=1)
            base = ((pid * (n + 1) + k) * 3) * 2
            tl.store(pa_ptr + base + 0, tl.sum(x0r * u0r - x0i * u0i))
            tl.store(pa_ptr + base + 1, tl.sum(x0r * u0i + x0i * u0r))
            tl.store(pa_ptr + base + 2, tl.sum(x0r * u1r - x0i * u1i))
            tl.store(pa_ptr + base + 3, tl.sum(x0r * u1i + x0i * u1r))
            tl.store(pa_ptr + base + 4, tl.sum(x1r * u1r - x1i * u1i))
            tl.store(pa_ptr + base + 5, tl.sum(x1r * u1i + x1i * u1r))
        else:
            ur = tl.dot(pr, Yr, out_dtype=tl.float64) - tl.dot(pi, Yi, out_dtype=tl.float64)   # C^(k-1) y, all columns
            ui = tl.dot(pr, Yi, out_dtype=tl.float64) + tl.dot(pi, Yr, out_dtype=tl.float64)
            sr = tl.sum(Xr * ur - Xi * ui, axis=0)                 # [VP]: x^T C^(k-1) y per column
            si = tl.sum(Xr * ui + Xi * ur, axis=0)
            po = ((((pid * V + vv) * (n + 1) + k) * 3) + kind) * 2
            tl.store(pa_ptr + po, sr, mask=cm)
            tl.store(pa_ptr + po + 1, si, mask=cm)
        if DO_POWER:
            nr = tl.dot(pr, cr, out_dtype=tl.float64) - tl.dot(pi, ci, out_dtype=tl.float64)
            ni = tl.dot(pr, ci, out_dtype=tl.float64) + tl.dot(pi, cr, out_dtype=tl.float64)
            pr = nr
            pi = ni
        if DO_TRACE:
            tb = (pid * (n + 1) + k) * 2
            tl.store(tr_ptr + tb, tl.sum(tl.sum(tl.where(eye, pr, 0.0), axis=1), axis=0))
            tl.store(tr_ptr + tb + 1, tl.sum(tl.sum(tl.where(eye, pi, 0.0), axis=1), axis=0))
    if not DO_TRACE and not DO_PATHS:
        tl.store(tr_ptr + pid * (n + 1) * 2, tl.sum(tl.sum(pr + pi, axis=1), axis=0))


def _planes(t: torch.Tensor):
    return t.real.contiguous(), t.imag.contiguous()


def powers_paths(Ce: torch.Tensor, Ye: torch.Tensor, XYe: torch.Tensor, idx: torch.Tensor, n: int, V: int,
                 **switches):
    """(tr [c, n+1], paths [c, V, n+1, 3]) complex128 for the subsets idx [c, q] (see the module docstring).

    Ce [E, E], Ye / XYe [E, 2V] complex128 on the device of idx; E - 1 is the dummy vertex. Index 0 of the
    k axis is left zero. Q = the next power of two >= max(q, 16). `switches` (DO_POWER / DO_TRACE / DO_PATHS = False)
    drop parts of the work for cost measurements only; the results are then not the sieve's.
    """
    c, q = idx.shape
    E = Ce.shape[0]
    Q = max(16, 1 << (max(q, 1) - 1).bit_length())
    if Q > 64:
        raise ValueError(f"subset matrices of {q} vertices exceed the kernel's 64")
    cr, ci = _planes(Ce); yr, yi = _planes(Ye); xr, xi = _planes(XYe)
    tr = torch.zeros((c, n + 1, 2), dtype=torch.float64, device=idx.device)
    pa = torch.zeros((c, V, n + 1, 3, 2), dtype=torch.float64, device=idx.device)
    if c:
        VP = max(16, 1 << (3 * V - 1).bit_length())
        _powers_paths_kernel[(c,)](cr, ci, yr, yi, xr, xi, idx.contiguous(), tr, pa, E, q, n, V=V, Q=Q, VP=VP,
                                   num_warps=4 if Q <= 32 else 8, **switches)
    return torch.view_as_complex(tr), torch.view_as_complex(pa)


@triton.jit(do_not_specialize=["n", "L"])
def _series_kernel(gi_ptr,                            # [c V, n, 3, 2] float64: i g_i (z^0, z^1, z^2), i = 1..n
                   f_ptr,                             # scratch [c V, n + 1, L, 2] float64
                   out_ptr,                           # out [c V, L, 2] float64: F_n
                   n, L, LP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    z = tl.arange(0, LP)
    zm = z < L
    fb = f_ptr + pid * (n + 1) * L * 2
    gb = gi_ptr + pid * n * 6
    one = tl.where(z == 0, 1.0, 0.0).to(tl.float64)
    tl.store(fb + z * 2, one, mask=zm)
    tl.store(fb + z * 2 + 1, one * 0.0, mask=zm)
    tl.debug_barrier()
    for k in range(1, n + 1):
        tr_ = tl.zeros([LP], dtype=tl.float64)
        ti_ = tl.zeros([LP], dtype=tl.float64)
        for i in range(1, k + 1):
            rb = fb + (k - i) * L * 2
            f0r = tl.load(rb + z * 2, mask=zm, other=0.0)
            f0i = tl.load(rb + z * 2 + 1, mask=zm, other=0.0)
            f1r = tl.load(rb + (z - 1) * 2, mask=zm & (z >= 1), other=0.0)
            f1i = tl.load(rb + (z - 1) * 2 + 1, mask=zm & (z >= 1), other=0.0)
            f2r = tl.load(rb + (z - 2) * 2, mask=zm & (z >= 2), other=0.0)
            f2i = tl.load(rb + (z - 2) * 2 + 1, mask=zm & (z >= 2), other=0.0)
            gp = gb + (i - 1) * 6
            ar = tl.load(gp); ai = tl.load(gp + 1)
            br = tl.load(gp + 2); bi = tl.load(gp + 3)
            cr = tl.load(gp + 4); ci = tl.load(gp + 5)
            tr_ += ar * f0r - ai * f0i + br * f1r - bi * f1i + cr * f2r - ci * f2i
            ti_ += ar * f0i + ai * f0r + br * f1i + bi * f1r + cr * f2i + ci * f2r
        kb = fb + k * L * 2
        tl.store(kb + z * 2, tr_ / k, mask=zm)
        tl.store(kb + z * 2 + 1, ti_ / k, mask=zm)
        tl.debug_barrier()                            # row k is read by other lanes at the next k
    nb = fb + n * L * 2
    ob = out_ptr + pid * L * 2
    tl.store(ob + z * 2, tl.load(nb + z * 2, mask=zm, other=0.0), mask=zm)
    tl.store(ob + z * 2 + 1, tl.load(nb + z * 2 + 1, mask=zm, other=0.0), mask=zm)


def series(gi: torch.Tensor) -> torch.Tensor:
    """F_n [c, V, 2n+1] complex128 of the recursion F_0 = 1, F_k = (1/k) sum_(i=1..k) (i g_i)(z) F_(k-i), where
    gi [c, V, n, 3] complex128 holds i g_i as the coefficients of z^0, z^1, z^2 (i = 1..n): [lambda^n] of
    exp(sum_k lambda^k g_k(z)), the sieve's per-subset term. Scratch: c V (n + 1) (2n + 1) complex.
    """
    c, V, n, _ = gi.shape
    L = 2 * n + 1
    LP = max(16, 1 << (L - 1).bit_length())
    g2 = torch.view_as_real(gi.contiguous()).contiguous()
    scratch = torch.empty((c * V, n + 1, L, 2), dtype=torch.float64, device=gi.device)
    out = torch.zeros((c * V, L, 2), dtype=torch.float64, device=gi.device)
    if c * V and n:
        _series_kernel[(c * V,)](g2, scratch, out, n, L, LP=LP, num_warps=1 if LP <= 32 else 2)
    elif n == 0:
        out[:, 0, 0] = 1.0
    return torch.view_as_complex(out).view(c, V, L)


def warmup(device, V_max: int = 8):
    """Compile both kernels for every (V <= V_max, Q) and series width once, on `device` (a few seconds on first
    use; then from Triton's disk cache), so that no compilation lands inside a timed sieve or shot.
    """
    dev = torch.device(device)
    for Q in (16, 32, 64):
        n = Q // 2
        E = Q + 1
        Ce = torch.zeros((E, E), dtype=torch.complex128, device=dev)
        idx = torch.zeros((1, Q), dtype=torch.int64, device=dev)
        for V in range(1, V_max + 1):
            Ye = torch.zeros((E, 2 * V), dtype=torch.complex128, device=dev)
            powers_paths(Ce, Ye, Ye, idx, n, V)
    for n in (7, 15, 31):
        series(torch.zeros((1, 1, n, 3), dtype=torch.complex128, device=dev))
