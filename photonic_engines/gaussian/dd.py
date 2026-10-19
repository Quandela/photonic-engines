"""Double-double (dd) arithmetic on torch tensors, for the pair sieve beyond float64 (docs/GAUSSIAN.md §3.4 v2.11).

A dd number is an unevaluated sum hi + lo of two float64 with |lo| <= ulp(hi) / 2: ~106 bits, eps_dd = 2^-104 (the
error bounds below are relative to that). Complex dd values are (hi, lo) pairs of complex128 tensors: addition is
component-wise, so the real error-free transformations apply to complex tensors as they are; products are formed from
the real and imaginary parts.

  two_sum / quick_two_sum / two_prod     error-free transformations (Knuth; Dekker's split, no fused multiply-add)
  add / mul / cmul / scale               dd element-wise (Bailey's double-double formulas, error <= ~4 eps_dd)
  dsum                                   dd sum along a dimension (pairwise, error <= ~(2 + log2 n) eps_dd sum |x|)
  matmul / cmatmul                       dd matrix products by the Ozaki scheme: each operand cut into beta-bit slices
                                         (beta = floor((53 - log2 q) / 2), q the inner size) so that every fp64 product of
                                         two slices is exact in any summation order - on GPU tensor cores too, checked on
                                         the H100 (run 20260927_201539) - then the slice products summed in dd.
Reference implementation and the H100 cost: bench/gbs_dd_rate.py (x249 of an fp64 complex product at q = 128 as written:
the slicing and the dd sums dominate, the GEMM count alone is x15).
"""
from __future__ import annotations

import math

import torch

_SPLIT = 134217729.0                                                   # 2^27 + 1: Dekker / Veltkamp split


# ---- real error-free transformations -------------------------------------------------------------
def two_sum(a, b):
    """(s, e) with s = fl(a + b) and s + e = a + b exactly."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def quick_two_sum(a, b):
    """(s, e) with s + e = a + b exactly, assuming |a| >= |b| (or a = 0)."""
    s = a + b
    return s, b - (s - a)


def _split(a):
    c = _SPLIT * a
    h = c - (c - a)
    return h, a - h


def two_prod(a, b):
    """(p, e) with p = fl(a b) and p + e = a b exactly (real float64 tensors, |a|, |b| < 2^996)."""
    p = a * b
    ah, al = _split(a); bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


# ---- dd element-wise (real, or complex for add) ---------------------------------------------------
def add(ah, al, bh, bl):
    """Double-double a + b (real or complex tensors: component-wise)."""
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    e = e + t
    s, e = quick_two_sum(s, e)
    e = e + f
    return quick_two_sum(s, e)


def mul(ah, al, bh, bl):
    """Double-double a b, real tensors."""
    p, e = two_prod(ah, bh)
    e = e + (ah * bl + al * bh)
    return quick_two_sum(p, e)


def cmul(ah, al, bh, bl):
    """Double-double a b, complex128 tensors (hi, lo each): (ar br - ai bi) + i (ar bi + ai br), every product and sum in dd."""
    ar, ai = ah.real, ah.imag; arl, ail = al.real, al.imag
    br, bi = bh.real, bh.imag; brl, bil = bl.real, bl.imag
    rr = mul(ar, arl, br, brl); ii = mul(ai, ail, bi, bil)
    ri = mul(ar, arl, bi, bil); ir = mul(ai, ail, br, brl)
    re = add(rr[0], rr[1], -ii[0], -ii[1]); im = add(ri[0], ri[1], ir[0], ir[1])
    return torch.complex(re[0], im[0]), torch.complex(re[1], im[1])


def scale(ah, al, s):
    """Double-double a x s for a float64 s (a scalar or a real tensor broadcasting against a), a real or complex: s is exact."""
    if torch.is_complex(ah):
        re = _scale_real(ah.real, al.real, s); im = _scale_real(ah.imag, al.imag, s)
        return torch.complex(re[0], im[0]), torch.complex(re[1], im[1])
    return _scale_real(ah, al, s)


def _scale_real(ah, al, s):
    s = torch.as_tensor(s, dtype=torch.float64, device=ah.device)
    p, e = two_prod(ah, s.expand_as(ah) if s.dim() else s * torch.ones_like(ah))
    return quick_two_sum(p, e + al * s)


def div_scalar(ah, al, d):
    """Double-double a / d for an exact float64 divisor d (a scalar or a real tensor broadcasting against a; the series'
    1 / k): one Newton correction on the fp64 quotient.
    """
    d = torch.as_tensor(d, dtype=torch.float64, device=ah.device)
    q1 = ah / d
    if torch.is_complex(ah):
        dr = d.expand_as(q1.real)
        pr = two_prod(q1.real, dr); pi = two_prod(q1.imag, dr)
        rr = add(ah.real, al.real, -pr[0], -pr[1]); ri = add(ah.imag, al.imag, -pi[0], -pi[1])
        q2 = torch.complex(rr[0], ri[0]) / d
    else:
        p = two_prod(q1, d.expand_as(q1)); r = add(ah, al, -p[0], -p[1]); q2 = r[0] / d
    return quick_two_sum(q1, q2)


def dsum(h, l, dim: int):
    """Double-double sum of (h, l) along `dim` (real or complex): pairwise halving, each level one dd add."""
    h = h.movedim(dim, 0); l = l.movedim(dim, 0)
    while h.shape[0] > 1:
        n = h.shape[0]
        if n % 2:
            h = torch.cat([h, torch.zeros_like(h[:1])]); l = torch.cat([l, torch.zeros_like(l[:1])]); n += 1
        h, l = add(h[0::2], l[0::2], h[1::2], l[1::2])
    return h[0], l[0]


# ---- Ozaki dd matrix products -------------------------------------------------------------------
# Fixed exponent schedule (Ozaki scheme, grouped): slice i of a row with max |x| < 2^e holds the bits between 2^(e - (i-1) beta)
# and the unit 2^(e + 1 - i beta), so every product A_i B_j with i + j = t is a multiple of the same unit U_t (row exponent +
# column exponent - t beta), bounded by 2^(2 beta - 2) U_t: the t-th group is ONE fp64 GEMM over the concatenated inner
# dimension, exact in any summation order while 2 beta - 2 + log2(inner size) <= 53. s slices (s beta >= 107) leave a rest
# below 2^-106 of the row max; groups t > s + 1 are dropped (below 2^-(s beta) relative). s GEMMs per real product instead
# of s (s + 1) / 2, each on a wider inner dimension.
def plan(q: int) -> tuple[int, int]:
    """(beta, s) for an inner dimension q: the largest beta with s = ceil(107 / beta) groups exact at inner size s q, with
    one bit of margin (the slices can exceed 2^(beta - 1) units by the lo parts and the rounding of the remainder).
    """
    beta = 26
    while True:
        s = math.ceil(107 / beta)
        if 2 * beta - 2 + math.ceil(math.log2(max(s * q, 2))) <= 52:     # one bit of margin under 53
            return beta, s
        beta -= 1


def _pow2(k):
    """2^k exactly for an int64 tensor k in [-1022, 1023], from the bit pattern (log2 / exp2 are not exact on every
    device: on the H100 ceil(log2(m)) and exp2 put the slice exponents one off and broke the GEMMs' exactness, job 27).
    """
    return ((k.clamp(-1022, 1023) + 1023) << 52).view(torch.float64)


def slices(hi, lo, beta: int, s: int, dim: int):
    """The s float64 slices of the real dd matrix (hi, lo) on the fixed schedule, relative to the max along `dim` (-1: the
    rows of a left operand, -2: the columns of a right one). Exact: hi + lo = sum of the slices + a rest < 2^(e - s beta).
    """
    m = torch.amax(torch.abs(hi), dim=dim, keepdim=True)
    e = torch.frexp(m)[1].to(torch.int64)                              # m < 2^e exactly (frexp: m = f 2^e, f in [0.5, 1))
    out = []
    rh, rl = hi, lo
    for i in range(1, s + 1):
        sig = 0.75 * _pow2(e + (53 - i * beta))
        sl = (rh + sig) - sig
        out.append(sl)
        rh, rl = two_sum(rh - sl, rl)                                   # rh - sl is exact
    return out


def matmul_sliced(A, B):
    """Double-double product from the slices of a left and a right operand (`slices`, same s): the groups t = s + 1 .. 2,
    one GEMM each over the concatenated inner dimension, summed in dd from the smallest.
    """
    s = len(A)
    hi = lo = None
    for t in range(s + 1, 1, -1):
        ii = range(max(1, t - s), min(s, t - 1) + 1)
        a = torch.cat([A[i - 1] for i in ii], -1) if len(ii) > 1 else A[ii[0] - 1]
        b = torch.cat([B[t - i - 1] for i in ii], -2) if len(ii) > 1 else B[t - ii[0] - 1]
        p = a @ b
        if hi is None:
            hi, lo = p, torch.zeros_like(p)
        else:
            hi, lo = add(hi, lo, p, torch.zeros_like(p))
    return hi, lo


def matmul(ah, al, bh, bl):
    """Double-double product of real dd matrices [.., p, q] x [.., q, r] (batched as torch.matmul), Ozaki grouped: s exact
    fp64 GEMMs; relative error ~2^-104 of sum |a| |b|.
    """
    beta, s = plan(ah.shape[-1])
    return matmul_sliced(slices(ah, al, beta, s, -1), slices(bh, bl, beta, s, -2))


def embed_left(ah, al):
    """The real [.., 2p, 2q] block form [[re, -im], [im, re]] of a complex dd left operand (hi, lo)."""
    def blk(x):
        return torch.cat([torch.cat([x.real, -x.imag], -1), torch.cat([x.imag, x.real], -1)], -2)
    return blk(ah), blk(al)


def embed_right(bh, bl):
    """The real [.., 2q, r] form [re; im] of a complex dd right operand (hi, lo)."""
    return torch.cat([bh.real, bh.imag], -2), torch.cat([bl.real, bl.imag], -2)


# fused slicing (v2.11c): the embedding and all s slices of a complex operand in one Triton kernel
# (`sieve_dd_triton.slices_complex`, bit-identical) instead of ~10 s torch element-wise kernels over the block form.
# None: on CUDA only; True: everywhere (the Triton interpreter on CPU, for the tests); False: never.
_FUSED_SLICES = None


def _fused(t) -> bool:
    return _FUSED_SLICES if _FUSED_SLICES is not None else t.device.type == "cuda"


def left_slices(ah, al):
    """Slices of the embedded complex left operand, for `cmatmul(left=...)` (reuse across products)."""
    beta, s = plan(2 * ah.shape[-1])
    if _fused(ah):
        from .sieve_dd_triton import slices_complex
        return slices_complex(ah, al, beta, s, True)
    return slices(*embed_left(ah, al), beta, s, -1)


def right_slices(bh, bl):
    """Slices of the embedded complex right operand, for `cmatmul(right=...)`."""
    beta, s = plan(2 * bh.shape[-2])
    if _fused(bh):
        from .sieve_dd_triton import slices_complex
        return slices_complex(bh, bl, beta, s, False)
    return slices(*embed_right(bh, bl), beta, s, -2)


def cmatmul(ah, al, bh, bl, left=None, right=None):
    """Double-double product of complex dd matrices (complex128 hi, lo) [.., p, q] x [.., q, r]: one real dd product of the
    block forms, [[Ar, -Ai], [Ai, Ar]] [Br; Bi] = [re; im]. `left` / `right`: precomputed `left_slices` / `right_slices`
    (then the corresponding hi, lo may be None; p is read from the other operand's shape when needed).
    """
    A = left if left is not None else left_slices(ah, al)
    B = right if right is not None else right_slices(bh, bl)
    hi, lo = matmul_sliced(A, B)
    p = hi.shape[-2] // 2
    return torch.complex(hi[..., :p, :], hi[..., p:, :]), torch.complex(lo[..., :p, :], lo[..., p:, :])
