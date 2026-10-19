"""Triton kernel for the series of the double-double sieve term (`hafnian._terms_dd`, docs/GAUSSIAN.md §3.4 v2.11).

The recursion F_0 = 1, F_k = (1/k) sum_(i=1..k) (i g_i)(z) F_(k-i)(z), with (i g_i)(z) = a_i + b_i z + c_i z^2, per subset and
loop vector - the same as `sieve_triton._series_kernel`, every product and sum in double-double. One program per (subset,
loop vector); the z coefficients on the lanes; F rows in a per-program scratch block (hi / lo, re / im planes). The torch
version (`_terms_dd` on CPU) issued ~3 n (log2 n + 10) kernels per k for this; here it is one launch per batch.

Double-double in registers. On CUDA (ASM) every add, subtract, multiply, divide and fma of the error-free
transformations is one inline PTX instruction with an explicit rounding mode (`add.rn.f64`, ...), which the optimizer can
neither reorder nor contract, and two_prod is exact by fma; the first GPU run (job 27) used plain operators and tl.fma and
came out at fp64 accuracy. Under the interpreter the plain operators (numpy, exact IEEE) and Dekker's split.
"""
import math

import torch
import triton
import triton.language as tl


# IEEE operations the optimizer may not touch: on CUDA each is one PTX instruction with an explicit rounding mode, which
# ptxas neither reorders nor contracts into a multiply-add (the first GPU run, job 27, computed this kernel at fp64 accuracy
# with plain Python operators and tl.fma: the error-free transformations did not survive compilation). Under the
# interpreter (ASM = False) the plain operators are numpy's, which are exact IEEE.
@triton.jit
def _add(a, b, ASM: tl.constexpr):
    if ASM:
        return tl.inline_asm_elementwise("add.rn.f64 $0, $1, $2;", "=d,d,d", [a, b], dtype=tl.float64, is_pure=True, pack=1)
    return a + b


@triton.jit
def _sub(a, b, ASM: tl.constexpr):
    if ASM:
        return tl.inline_asm_elementwise("sub.rn.f64 $0, $1, $2;", "=d,d,d", [a, b], dtype=tl.float64, is_pure=True, pack=1)
    return a - b


@triton.jit
def _mul(a, b, ASM: tl.constexpr):
    if ASM:
        return tl.inline_asm_elementwise("mul.rn.f64 $0, $1, $2;", "=d,d,d", [a, b], dtype=tl.float64, is_pure=True, pack=1)
    return a * b


@triton.jit
def _div(a, b, ASM: tl.constexpr):
    if ASM:
        return tl.inline_asm_elementwise("div.rn.f64 $0, $1, $2;", "=d,d,d", [a, b], dtype=tl.float64, is_pure=True, pack=1)
    return a / b


@triton.jit
def _two_sum(a, b, ASM: tl.constexpr):
    s = _add(a, b, ASM)
    bb = _sub(s, a, ASM)
    return s, _add(_sub(a, _sub(s, bb, ASM), ASM), _sub(b, bb, ASM), ASM)


@triton.jit
def _quick_two_sum(a, b, ASM: tl.constexpr):
    s = _add(a, b, ASM)
    return s, _sub(b, _sub(s, a, ASM), ASM)


@triton.jit
def _two_prod(a, b, ASM: tl.constexpr):
    p = _mul(a, b, ASM)
    if ASM:                                           # exact: p + fma(a, b, -p) = a b
        e = tl.inline_asm_elementwise("fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d", [a, b, -p], dtype=tl.float64,
                                      is_pure=True, pack=1)
    else:                                             # Dekker's split (numpy has no fma)
        ca = 134217729.0 * a
        ah = ca - (ca - a)
        al = a - ah
        cb = 134217729.0 * b
        bh = cb - (cb - b)
        bl = b - bh
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@triton.jit
def _dd_add(ah, al, bh, bl, ASM: tl.constexpr):
    s, e = _two_sum(ah, bh, ASM)
    t, f = _two_sum(al, bl, ASM)
    e = _add(e, t, ASM)
    s, e = _quick_two_sum(s, e, ASM)
    e = _add(e, f, ASM)
    return _quick_two_sum(s, e, ASM)


@triton.jit
def _dd_mul(ah, al, bh, bl, ASM: tl.constexpr):
    p, e = _two_prod(ah, bh, ASM)
    e = _add(e, _add(_mul(ah, bl, ASM), _mul(al, bh, ASM), ASM), ASM)
    return _quick_two_sum(p, e, ASM)


@triton.jit
def _cdd_mul(xrh, xrl, xih, xil, yrh, yrl, yih, yil, ASM: tl.constexpr):
    """(x y) for complex dd x = (xr, xi), y = (yr, yi), each part a (hi, lo) pair."""
    rrh, rrl = _dd_mul(xrh, xrl, yrh, yrl, ASM)
    iih, iil = _dd_mul(xih, xil, yih, yil, ASM)
    rih, ril = _dd_mul(xrh, xrl, yih, yil, ASM)
    irh, irl = _dd_mul(xih, xil, yrh, yrl, ASM)
    reh, rel = _dd_add(rrh, rrl, -iih, -iil, ASM)
    imh, iml = _dd_add(rih, ril, irh, irl, ASM)
    return reh, rel, imh, iml


@triton.jit
def _dd_div(h, l, d, ASM: tl.constexpr):
    q1 = _div(h, d, ASM)
    p, e = _two_prod(q1, d, ASM)
    rh, rl = _dd_add(h, l, -p, -e, ASM)
    return _quick_two_sum(q1, _div(rh, d, ASM), ASM)


@triton.jit(do_not_specialize=["n", "L"])
def _series_dd_kernel(g_ptr,                          # [cV, n, 3, 4] float64: i g_i (z^0, z^1, z^2) as (re hi, re lo, im hi, im lo)
                      f_ptr,                          # scratch [cV, n + 1, L, 4] float64
                      out_ptr,                        # out [cV, L, 4] float64: F_n
                      n, L, LP: tl.constexpr, ASM: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    z = tl.arange(0, LP)
    zm = z < L
    fb = f_ptr + pid * (n + 1) * L * 4
    gb = g_ptr + pid * n * 12
    zero = tl.zeros([LP], dtype=tl.float64)
    one = tl.where(z == 0, 1.0, 0.0).to(tl.float64)
    tl.store(fb + z * 4, one, mask=zm)
    tl.store(fb + z * 4 + 1, zero, mask=zm)
    tl.store(fb + z * 4 + 2, zero, mask=zm)
    tl.store(fb + z * 4 + 3, zero, mask=zm)
    tl.debug_barrier()
    for k in range(1, n + 1):
        trh = tl.zeros([LP], dtype=tl.float64); trl = tl.zeros([LP], dtype=tl.float64)
        tih = tl.zeros([LP], dtype=tl.float64); til = tl.zeros([LP], dtype=tl.float64)
        for i in range(1, k + 1):
            rb = fb + (k - i) * L * 4
            gp = gb + (i - 1) * 12
            for sft in tl.static_range(3):
                m = zm & (z >= sft)
                frh = tl.load(rb + (z - sft) * 4, mask=m, other=0.0)
                frl = tl.load(rb + (z - sft) * 4 + 1, mask=m, other=0.0)
                fih = tl.load(rb + (z - sft) * 4 + 2, mask=m, other=0.0)
                fil = tl.load(rb + (z - sft) * 4 + 3, mask=m, other=0.0)
                grh = tl.load(gp + sft * 4); grl = tl.load(gp + sft * 4 + 1)
                gih = tl.load(gp + sft * 4 + 2); gil = tl.load(gp + sft * 4 + 3)
                prh, prl, pih, pil = _cdd_mul(grh + zero, grl + zero, gih + zero, gil + zero, frh, frl, fih, fil, ASM)
                trh, trl = _dd_add(trh, trl, prh, prl, ASM)
                tih, til = _dd_add(tih, til, pih, pil, ASM)
        kd = k * 1.0
        trh, trl = _dd_div(trh, trl, kd + zero, ASM)
        tih, til = _dd_div(tih, til, kd + zero, ASM)
        kb = fb + k * L * 4
        tl.store(kb + z * 4, trh, mask=zm)
        tl.store(kb + z * 4 + 1, trl, mask=zm)
        tl.store(kb + z * 4 + 2, tih, mask=zm)
        tl.store(kb + z * 4 + 3, til, mask=zm)
        tl.debug_barrier()                            # row k is read by other lanes at the next k
    nb = fb + n * L * 4
    ob = out_ptr + pid * L * 4
    for c in tl.static_range(4):
        tl.store(ob + z * 4 + c, tl.load(nb + z * 4 + c, mask=zm, other=0.0), mask=zm)


def series_dd(gih: torch.Tensor, gil: torch.Tensor):
    """(F_n hi, F_n lo) [c, V, 2n+1] complex128 of the dd recursion, from i g_i = (gih, gil) [c, V, n, 3] complex128 (the
    coefficients of z^0, z^1, z^2, i = 1..n). Scratch: c V (n + 1) (2n + 1) x 4 float64.
    """
    c, V, n, _ = gih.shape
    L = 2 * n + 1
    LP = max(16, 1 << (L - 1).bit_length())
    g4 = torch.stack([gih.real, gil.real, gih.imag, gil.imag], -1).contiguous()      # [c, V, n, 3, 4]
    scratch = torch.empty((c * V, n + 1, L, 4), dtype=torch.float64, device=gih.device)
    out = torch.zeros((c * V, L, 4), dtype=torch.float64, device=gih.device)
    if c * V and n:
        _series_dd_kernel[(c * V,)](g4, scratch, out, n, L, LP=LP, ASM=gih.device.type == "cuda",
                                    num_warps=1 if LP <= 32 else 2)
    elif n == 0:
        out[:, 0, 0] = 1.0
    out = out.view(c, V, L, 4)
    return torch.complex(out[..., 0], out[..., 2]), torch.complex(out[..., 1], out[..., 3])


# ---- fused Ozaki slicing of a complex dd operand (v2.11c, 2026-09-28) ------------------------------------------------
@triton.jit
def _slices_kernel(x_ptr,                            # [M, 4] float64: (re hi, im hi, re lo, im lo) per complex entry
                   e_ptr,                            # int64 exponents: per row (LEFT) or per column (right) of the block form
                   out_ptr,                          # [s, B, 2P, 2Q] (LEFT) or [s, B, 2P, Q] float64
                   M, P, Q, per, beta, S: tl.constexpr, LEFT: tl.constexpr, ASM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    msk = i < M
    rh = tl.load(x_ptr + 4 * i, mask=msk, other=0.0)
    ih = tl.load(x_ptr + 4 * i + 1, mask=msk, other=0.0)
    rl = tl.load(x_ptr + 4 * i + 2, mask=msk, other=0.0)
    il = tl.load(x_ptr + 4 * i + 3, mask=msk, other=0.0)
    b = i // (P * Q); r = (i // Q) % P; c = i % Q
    if LEFT:
        e = tl.load(e_ptr + b * P + r, mask=msk, other=0)
        W = 2 * Q
        base = b * (2 * P) * W
        o_rr = base + r * W + c                      # [[re, -im], [im, re]]
        o_ri = base + r * W + Q + c
        o_ir = base + (P + r) * W + c
        o_ii = base + (P + r) * W + Q + c
    else:
        e = tl.load(e_ptr + b * Q + c, mask=msk, other=0)
        base = b * (2 * P) * Q
        o_rr = base + r * Q + c                      # [re; im]
        o_ir = base + (P + r) * Q + c
    for k in tl.static_range(1, S + 1):
        kk = tl.minimum(tl.maximum(e + 53 - k * beta, -1022), 1023)
        sig = 0.75 * ((kk + 1023) << 52).to(tl.float64, bitcast=True)
        sr = _sub(_add(rh, sig, ASM), sig, ASM)
        si = _sub(_add(ih, sig, ASM), sig, ASM)
        rh, rl = _two_sum(_sub(rh, sr, ASM), rl, ASM)
        ih, il = _two_sum(_sub(ih, si, ASM), il, ASM)
        ob = out_ptr + (k - 1) * per
        tl.store(ob + o_rr, sr, mask=msk)
        tl.store(ob + o_ir, si, mask=msk)
        if LEFT:
            tl.store(ob + o_ri, -si, mask=msk)
            tl.store(ob + o_ii, sr, mask=msk)


def slices_complex(hi: torch.Tensor, lo: torch.Tensor, beta: int, s: int, left: bool):
    """`dd.slices` of the block form of a complex dd operand (hi, lo) [.., P, Q] complex128 in one kernel: left, the rows of
    [[re, -im], [im, re]] ([.., 2P, 2Q]); right, the columns of [re; im] ([.., 2P, Q]). Bit-identical to
    `dd.slices(*dd.embed_left(hi, lo), beta, s, -1)` / `dd.slices(*dd.embed_right(hi, lo), beta, s, -2)`: the block row max
    is max(|re|, |im|) over the complex row (column), and the slicing is odd in its argument (round to nearest even
    about a positive sig), so the slices of -im are minus those of im. Returns the list of s slices.
    """
    *bs, P, Q = hi.shape
    B = math.prod(bs)
    a = torch.maximum(hi.real.abs(), hi.imag.abs())
    m = torch.amax(a, dim=-1) if left else torch.amax(a, dim=-2)
    e = torch.frexp(m)[1].to(torch.int64).contiguous()
    x = torch.stack([hi.real, hi.imag, lo.real, lo.imag], -1).contiguous()
    shape = (s, *bs, 2 * P, 2 * Q) if left else (s, *bs, 2 * P, Q)
    out = torch.empty(shape, dtype=torch.float64, device=hi.device)
    M = B * P * Q
    if M:
        BLOCK = 256
        _slices_kernel[(triton.cdiv(M, BLOCK),)](x, e, out, M, P, Q, out[0].numel(), beta, S=s, LEFT=left,
                                                 ASM=hi.device.type == "cuda", BLOCK=BLOCK)
    return list(out.unbind(0))
