"""Sequential-chain Triton kernel for the EmitterMPS sampler, generated per (K, L).

Same lane decomposition and Gray walk as `emitter_triton._emitter_cofactors_kernel`, but every
lane is ONE thread that holds all K columns (column index static, code fully unrolled):

    per Gray step:  s_{c,l} += fct * U[r, mode(c,l)]                 (4 loads per column, L = 2)
                    G_c = sum_l s_{c,l} M~_{c,l}                        (8 complex mults)
                    p_0 = s0,  p_{c+1} = G_c p_c                        (4 per column, stored)
                    q_{K-1}^T = f^dag,  q_{c-1}^T = q_c^T G_c           (4 per column)
                    acc_c += sign * p_c q_c^T                           (4 per column)

i.e. O(K) 2x2 products instead of the two O(K log K) associative matrix scans of the scan kernel,
no shifted copies of the column sums, and exact K (no padding). Per-lane state is 16 K doubles
(sums, accumulators, stored p_c); the term matrices are re-read from L1 every step, and since a
warp holds 32 consecutive lanes of the same term (segments of one Glynn sum) those loads and the
U row loads are warp-uniform broadcasts. Register pressure grows with K, so SEG = 32 * num_warps
(one lane per thread) and the compiler spills for large K; tune with bench/bench_emitter_tune.py.

The kernel source is generated as text (Triton's compiler rejects Python lists/comprehensions,
and tuples of K tensors would need static indexing tricks), written to a cache directory and
imported, so both the CPU interpreter and the GPU compiler see an ordinary module.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile

_KERNELS = {}


def _cmul(ar, ai, br, bi):
    """(re, im) expression strings of (ar + i ai) * (br + i bi)."""
    return f"({ar} * {br} - {ai} * {bi})", f"({ar} * {bi} + {ai} * {br})"


def _cmac(acc_r, acc_i, ar, ai, br, bi, sign="+"):
    re, im = _cmul(ar, ai, br, bi)
    return [f"{acc_r} {sign}= {re}", f"{acc_i} {sign}= {im}"]


def generate_source(K: int, L: int, B: int, mat_mode: str = "scalar") -> str:
    """Triton source of the kernel for K kept slots, L labels per slot (1 or 2) and B column
    blocks per lane.

    Lane = (term, Gray segment) as before; the K columns are split into B blocks of CB =
    ceil(K / B) consecutive columns (columns >= K are padding with G = I). A block is one thread:
    it holds its CB columns as [SEG, B] tensors with a STATIC in-block index, so the chains inside
    a block are plain sequential 2x2 products; the B blocks of a lane are B adjacent threads and
    are joined by two associative scans over the block products H_b = G_{last} ... G_{first}
    (prefix E_b = H_{b-1} ... H_0 and suffix F_b = H_{B-1} ... H_{b+1}, obtained directly with
    pair combines (P, E) -> no shifted copies). Per step and column: G_c (8 complex mults), block
    product (8), forward p (4), backward q (4), accumulate (4); the scans add 2 log2(B) x 16 per
    lane-step. Per-thread state ~ 24 CB + 40 doubles, so B is chosen to keep CB <= 2-3.

    A program is SEG consecutive segments of ONE term (host guarantees SEG <= segments per term),
    so the term data (stream, rows, modes, matrices, boundary vectors) are program-uniform: rows
    and boundary vectors are scalar loads; modes and matrices are [SEG, B] loads whose addresses
    differ only across blocks (L1 broadcasts). mat_mode "scalar" re-reads the matrices every
    step, "hoist" keeps them in registers (16 CB doubles).
    """
    assert K >= 1 and L in (1, 2) and mat_mode in ("scalar", "hoist") and B >= 1 and (B & (B - 1)) == 0
    CB = -(-K // B)
    o = []
    w = o.append
    w("import triton")
    w("import triton.language as tl")
    w("")
    if B > 1:
        w("@triton.jit")
        w("def _fwd(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i,")
        w("         b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, f0r, f0i, f1r, f1i, f2r, f2i, f3r, f3i):")
        w('    """forward pair combine: (P_A, E_A) lower, (P_B, E_B) higher -> (P_B P_A, E_B P_A)."""')
        for name, (X, Y) in (("p", ("b", "a")), ("e", ("f", "a"))):
            for r, (ri, ci) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                # (X Y)[ri, ci] = X[ri,0] Y[0,ci] + X[ri,1] Y[1,ci]
                x0, x1 = f"{X}{ri * 2}", f"{X}{ri * 2 + 1}"
                y0, y1 = f"{Y}{ci}", f"{Y}{2 + ci}"
                r1, i1 = _cmul(x0 + "r", x0 + "i", y0 + "r", y0 + "i")
                r2, i2 = _cmul(x1 + "r", x1 + "i", y1 + "r", y1 + "i")
                w(f"    {name}{r}r = {r1} + {r2}; {name}{r}i = {i1} + {i2}")
        w("    return p0r, p0i, p1r, p1i, p2r, p2i, p3r, p3i, e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i")
        w("")
        w("@triton.jit")
        w("def _bwd(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i,")
        w("         b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, f0r, f0i, f1r, f1i, f2r, f2i, f3r, f3i):")
        w('    """reverse pair combine: (P_A, F_A) higher, (P_B, F_B) lower -> (P_A P_B, P_A F_B)."""')
        for name, (X, Y) in (("p", ("a", "b")), ("e", ("a", "f"))):
            for r, (ri, ci) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                x0, x1 = f"{X}{ri * 2}", f"{X}{ri * 2 + 1}"
                y0, y1 = f"{Y}{ci}", f"{Y}{2 + ci}"
                r1, i1 = _cmul(x0 + "r", x0 + "i", y0 + "r", y0 + "i")
                r2, i2 = _cmul(x1 + "r", x1 + "i", y1 + "r", y1 + "i")
                w(f"    {name}{r}r = {r1} + {r2}; {name}{r}i = {i1} + {i2}")
        w("    return p0r, p0i, p1r, p1i, p2r, p2i, p3r, p3i, e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i")
        w("")
    K1 = K - 1                      # drawn rows at this step
    H_PAD = K1 // 2 + 1             # max repeated-row groups (mult >= 2) + the halved top digit
    w('@triton.jit(do_not_specialize=["LSEG", "TOTAL"])')
    w(f"def kernel_k{K}_l{L}_b{B}(u_ptr, drow_ptr, mult_ptr, radix_ptr, n2_ptr, nterm_ptr, cm_ptr,")
    w("        lane2term_ptr, term2stream_ptr, lstart_ptr, mats_ptr, s0_ptr, f_ptr, binom_ptr, part_ptr, ctz_ptr,")
    w("        stride_um, stride_ds, stride_cs, stride_mt, stride_mj, LSEG, TOTAL, SEG: tl.constexpr):")
    I = "    "
    w(I + "pid = tl.program_id(0)")
    w(I + "g = pid * SEG + tl.arange(0, SEG)")
    w(I + "valid = g < TOTAL")
    w(I + "lam = tl.load(lane2term_ptr + g, mask=valid, other=0)")
    w(I + "s = tl.load(term2stream_ptr + lam, mask=valid, other=0)")
    w(I + "li = g - tl.load(lstart_ptr + lam, mask=valid, other=0)")
    w(I + "base = li.to(tl.int64) * LSEG")
    w(I + "n2 = tl.load(n2_ptr + s, mask=valid, other=0)                    # binary digits (distinct single rows)")
    w(I + "Ns = tl.load(nterm_ptr + s, mask=valid, other=1)                 # terms of the stream")
    w(I + "drow_lane = drow_ptr + s * stride_ds")
    w(I + "mult_lane = mult_ptr + s * stride_ds")
    w(I + "radix_lane = radix_ptr + s * stride_ds")
    w(I + "cm_lane = (cm_ptr + s * stride_cs)[:, None]")
    w(I + "mat_lane = (mats_ptr + lam * stride_mt)[:, None]")
    w(I + f"bb = tl.zeros([SEG, {B}], tl.int32) + tl.arange(0, {B})[None, :]")
    E = ("00", "01", "10", "11")
    # per in-block column i: column index, validity, modes
    for i in range(CB):
        w(I + f"col{i} = bb * {CB} + {i}")
        w(I + f"cm{i} = (col{i} < {K}) & valid[:, None]")
        for l in range(L):
            w(I + f"md{i}_{l} = tl.load(cm_lane + col{i} * 2 + {l}, mask=cm{i}, other=0) * 2")
    # ---- mixed-radix Gray walk state (Clifford & Clifford 2020 collapse of repeated rows) ----
    # positions 0..n2-1: distinct single rows (radix 2, implicit binary digits of the term index);
    # positions n2..K1-1: repeated-row groups (radix s+1) and the halved top digit; H_PAD of them
    # are held as explicit digits d_h / reflected digits v_h (int32 [SEG]).
    w(I + "th = base >> n2")
    for h in range(H_PAD):
        w(I + f"hp{h} = n2 + {h}")
        w(I + f"R{h} = tl.load(radix_lane + hp{h}, mask=valid & (hp{h} < {K1}), other=1)")
        w(I + f"d{h} = (th % R{h}).to(tl.int32); th = th // R{h}; R{h} = R{h}.to(tl.int32)")
    w(I + "par = tl.zeros([SEG], tl.int32)")
    for h in range(H_PAD - 1, -1, -1):
        w(I + f"v{h} = tl.where((par & 1) == 0, d{h}, R{h} - 1 - d{h}); par += v{h}")
    w(I + "parh = par & 1")
    w(I + "w0 = tl.zeros([SEG], tl.float64) + 1.0")
    for h in range(H_PAD):
        w(I + f"Mu{h} = tl.load(mult_lane + hp{h}, mask=valid & (hp{h} < {K1}), other=0).to(tl.int32)")
        w(I + f"bn = tl.load(binom_ptr + Mu{h} * 32 + v{h}, mask=valid, other=1.0)")
        w(I + f"w0 = w0 * tl.where((v{h} & 1) == 1, -bn, bn)")
    # column sums at the segment start: sum_i coef_i U[r_i, mode]
    for i in range(CB):
        for l in range(L):
            w(I + f"sr{i}_{l} = tl.zeros([SEG, {B}], tl.float64); si{i}_{l} = tl.zeros([SEG, {B}], tl.float64)")
    # dynamic loop (an unrolled one would cost K1 x 4 CB tensor loads of compile time)
    w(I + f"for i in range(0, {K1}):")
    w(I + "    isb = i < n2")
    w(I + "    bt = ((base >> i) & 1).to(tl.int32); bt1 = ((base >> (i + 1)) & 1).to(tl.int32)")
    w(I + "    vb = bt ^ tl.where(i == n2 - 1, parh, bt1)")
    w(I + "    cb = (1 - 2 * vb).to(tl.float64)")
    w(I + "    ch = tl.zeros([SEG], tl.float64)")
    for h in range(H_PAD):
        w(I + f"    ch = tl.where(hp{h} == i, (Mu{h} - 2 * v{h}).to(tl.float64), ch)")
    w(I + "    coef = tl.where(isb, cb, ch)[:, None]")
    w(I + "    w0 = w0 * tl.where(isb, cb, 1.0)")
    w(I + "    rowp = u_ptr + (tl.load(drow_lane + i, mask=valid, other=0) * stride_um)[:, None]")
    for c in range(CB):
        for l in range(L):
            w(I + f"    sr{c}_{l} += coef * tl.load(rowp + md{c}_{l}, mask=cm{c}, other=0.0)")
            w(I + f"    si{c}_{l} += coef * tl.load(rowp + md{c}_{l} + 1, mask=cm{c}, other=0.0)")
    w(I + "w = w0")
    for nm, off, sg in (("s0a_r", 0, ""), ("s0a_i", 1, ""), ("s0b_r", 2, ""), ("s0b_i", 3, ""),
                        ("fa_r", 0, ""), ("fa_i", 1, "-"), ("fb_r", 2, ""), ("fb_i", 3, "-")):
        ptr = "s0_ptr" if nm.startswith("s0") else "f_ptr"
        w(I + f"{nm} = {sg}tl.load({ptr} + lam * 4 + {off}, mask=valid, other=0.0)[:, None]")
    for i in range(CB):
        for e in E:
            w(I + f"acc{i}_{e}r = tl.zeros([SEG, {B}], tl.float64); acc{i}_{e}i = tl.zeros([SEG, {B}], tl.float64)")

    def mat_loads(ind):
        for i in range(CB):
            for ei, e in enumerate(E):
                for l in range(L):
                    off = l * 8 + ei * 2
                    w(ind + f"mr{i}_{l}_{e} = tl.load(mat_lane + col{i} * stride_mj + {off}, mask=cm{i}, other=0.0); "
                            f"mi{i}_{l}_{e} = tl.load(mat_lane + col{i} * stride_mj + {off + 1}, mask=cm{i}, other=0.0)")

    if mat_mode == "hoist":
        mat_loads(I)
    # ---- walk: t' = 1..L-1 within the segment; the changing digit is ctz(t') (base is a multiple
    # of L): a binary flip when ctz(t') < n2, else an odometer step on the explicit high digits
    # (rare, taken by the whole program under a scalar `if` on ctz(t') >= min n2). Lanes past N_s only idle.
    w(I + "rem = tl.where(valid, tl.minimum(Ns - base, LSEG), 0)")
    w(I + "Lp = tl.max(rem, axis=0)")
    w(I + "n2min = tl.min(tl.where(valid, n2, 1 << 30), axis=0)")     # lanes needing a high-digit step at t: ctz(t) >= n2
    w(I + "for t in range(0, Lp):")
    J = I + "    "
    w(J + "tt = base + t")
    w(J + "inN = valid & (tt < Ns)")
    w(J + "ok = inN & (t > 0)")
    w(J + "cc = tl.load(ctz_ptr + t).to(tl.int64)")
    w(J + "isb = cc < n2")
    w(J + "bt = ((tt >> cc) & 1).to(tl.int32); bt1 = ((tt >> (cc + 1)) & 1).to(tl.int32)")
    w(J + "vnew = bt ^ tl.where(cc == n2 - 1, parh, bt1)")
    w(J + "fct = tl.where(isb, (2 - 4 * vnew).to(tl.float64), 0.0)")
    w(J + "ratio = tl.where(isb, -1.0, 1.0).to(tl.float64)")
    w(J + "rowi = tl.where(isb, cc, 0)")
    w(J + "nb = tl.where(isb, 0, 1)")
    w(J + "if cc >= n2min:")                                      # scalar test: no per-step reduction
    Q = J + "    "
    w(Q + f"nu = tl.zeros([SEG], tl.int32) + {H_PAD}")
    for h in range(H_PAD - 1, -1, -1):
        w(Q + f"nu = tl.where(d{h} < R{h} - 1, {h}, nu)")
    w(Q + "suf = tl.zeros([SEG], tl.int32); dirn = tl.zeros([SEG], tl.int32) + 1")
    w(Q + "vold = tl.zeros([SEG], tl.int32); mnu = tl.zeros([SEG], tl.int32)")
    for h in range(H_PAD - 1, -1, -1):
        w(Q + f"sel = nu == {h}")
        w(Q + "dirn = tl.where(sel, tl.where((suf & 1) == 0, 1, -1), dirn)")
        w(Q + f"vold = tl.where(sel, v{h}, vold)")
        w(Q + f"mnu = tl.where(sel, Mu{h}, mnu)")
        w(Q + f"suf += v{h}")
    w(Q + "upd = ok & (nb == 1)")
    for h in range(H_PAD):
        w(Q + f"d{h} = tl.where(upd, tl.where(nu == {h}, d{h} + 1, tl.where({h} < nu, 0, d{h})), d{h})")
        w(Q + f"v{h} = tl.where(upd & (nu == {h}), v{h} + dirn, v{h})")
    w(Q + "up = dirn > 0")
    w(Q + "num = tl.where(up, -(mnu - vold), -vold).to(tl.float64)")
    w(Q + "den = tl.where(up, vold + 1, mnu - vold + 1).to(tl.float64)")
    w(Q + "ratio = tl.where(upd, num / den, ratio)")
    w(Q + "fct = tl.where(upd, (-2 * dirn).to(tl.float64), fct)")
    w(Q + "rowi = tl.where(upd, n2 + nu.to(tl.int64), rowi)")
    w(Q + "parh = tl.where(upd, parh ^ 1, parh)")
    w(J + "w = w * tl.where(ok, ratio, 1.0)")
    w(J + "fct = tl.where(ok, fct, 0.0)[:, None]")
    w(J + "rowp = u_ptr + (tl.load(drow_lane + rowi, mask=ok, other=0) * stride_um)[:, None]")
    for i in range(CB):
        for l in range(L):
            w(J + f"sr{i}_{l} += fct * tl.load(rowp + md{i}_{l}, mask=cm{i}, other=0.0)")
            w(J + f"si{i}_{l} += fct * tl.load(rowp + md{i}_{l} + 1, mask=cm{i}, other=0.0)")
    w(J + "sign = tl.where(inN, w, 0.0)[:, None]")
    if mat_mode == "scalar":
        mat_loads(J)
    # slot matrices G_i = sum_l s_{i,l} M_{i,l}; identity on padded columns
    for i in range(CB):
        for _ei, e in enumerate(E):
            for l in range(L):
                re, im = _cmul(f"sr{i}_{l}", f"si{i}_{l}", f"mr{i}_{l}_{e}", f"mi{i}_{l}_{e}")
                if l == 0:
                    w(J + f"g{i}_{e}r = {re}; g{i}_{e}i = {im}")
                else:
                    w(J + f"g{i}_{e}r += {re}; g{i}_{e}i += {im}")
        if CB * B > K:
            w(J + f"g{i}_00r = tl.where(cm{i}, g{i}_00r, 1.0); g{i}_11r = tl.where(cm{i}, g{i}_11r, 1.0)")

    def matmul(out, X, Y, ind):
        """Out = X @ Y on names <X>_<e>r/i (2x2 complex)."""
        for ri in range(2):
            for ci in range(2):
                r1, i1 = _cmul(f"{X}_{ri}0r", f"{X}_{ri}0i", f"{Y}_0{ci}r", f"{Y}_0{ci}i")
                r2, i2 = _cmul(f"{X}_{ri}1r", f"{X}_{ri}1i", f"{Y}_1{ci}r", f"{Y}_1{ci}i")
                w(ind + f"{out}_{ri}{ci}r = {r1} + {r2}; {out}_{ri}{ci}i = {i1} + {i2}")

    if B > 1:
        # block product H = G_{CB-1} ... G_0 and the two scans
        if CB == 1:
            for e in E:
                w(J + f"h_{e}r = g0_{e}r; h_{e}i = g0_{e}i")
        else:
            matmul("h", "g1", "g0", J)
            for i in range(2, CB):
                matmul("hn", f"g{i}", "h", J)
                for e in E:
                    w(J + f"h_{e}r = hn_{e}r; h_{e}i = hn_{e}i")
        w(J + f"one = tl.zeros([SEG, {B}], tl.float64) + 1.0; zero = tl.zeros([SEG, {B}], tl.float64)")
        hs = ", ".join(f"h_{e}r, h_{e}i" for e in E)
        w(J + "_, _, _, _, _, _, _, _, E_00r, E_00i, E_01r, E_01i, E_10r, E_10i, E_11r, E_11i = tl.associative_scan(")
        w(J + f"    ({hs}, one, zero, zero, zero, zero, zero, one, zero), 1, _fwd)")
        w(J + "_, _, _, _, _, _, _, _, F_00r, F_00i, F_01r, F_01i, F_10r, F_10i, F_11r, F_11i = tl.associative_scan(")
        w(J + f"    ({hs}, one, zero, zero, zero, zero, zero, one, zero), 1, _bwd, reverse=True)")
        # p_start = E s0 ; q_end^T = f^dag F
        r1, i1 = _cmul("E_00r", "E_00i", "s0a_r", "s0a_i"); r2, i2 = _cmul("E_01r", "E_01i", "s0b_r", "s0b_i")
        w(J + f"p0a_r = {r1} + {r2}; p0a_i = {i1} + {i2}")
        r1, i1 = _cmul("E_10r", "E_10i", "s0a_r", "s0a_i"); r2, i2 = _cmul("E_11r", "E_11i", "s0b_r", "s0b_i")
        w(J + f"p0b_r = {r1} + {r2}; p0b_i = {i1} + {i2}")
        r1, i1 = _cmul("fa_r", "fa_i", "F_00r", "F_00i"); r2, i2 = _cmul("fb_r", "fb_i", "F_10r", "F_10i")
        w(J + f"qa_r = {r1} + {r2}; qa_i = {i1} + {i2}")
        r1, i1 = _cmul("fa_r", "fa_i", "F_01r", "F_01i"); r2, i2 = _cmul("fb_r", "fb_i", "F_11r", "F_11i")
        w(J + f"qb_r = {r1} + {r2}; qb_i = {i1} + {i2}")
    else:
        w(J + "p0a_r = s0a_r; p0a_i = s0a_i; p0b_r = s0b_r; p0b_i = s0b_i")
        w(J + "qa_r = fa_r; qa_i = fa_i; qb_r = fb_r; qb_i = fb_i")
    # forward chain inside the block: p_{i+1} = G_i p_i (stored)
    for i in range(CB - 1):
        n = i + 1
        r1, i1 = _cmul(f"g{i}_00r", f"g{i}_00i", f"p{i}a_r", f"p{i}a_i"); r2, i2 = _cmul(f"g{i}_01r", f"g{i}_01i", f"p{i}b_r", f"p{i}b_i")
        w(J + f"p{n}a_r = {r1} + {r2}; p{n}a_i = {i1} + {i2}")
        r1, i1 = _cmul(f"g{i}_10r", f"g{i}_10i", f"p{i}a_r", f"p{i}a_i"); r2, i2 = _cmul(f"g{i}_11r", f"g{i}_11i", f"p{i}b_r", f"p{i}b_i")
        w(J + f"p{n}b_r = {r1} + {r2}; p{n}b_i = {i1} + {i2}")
    # backward chain with accumulation: q_{i-1}^T = q_i^T G_i
    for i in range(CB - 1, -1, -1):
        for e, (pv, qv) in (("00", ("a", "a")), ("01", ("a", "b")), ("10", ("b", "a")), ("11", ("b", "b"))):
            re, im = _cmul(f"p{i}{pv}_r", f"p{i}{pv}_i", f"q{qv}_r", f"q{qv}_i")
            w(J + f"acc{i}_{e}r += sign * {re}; acc{i}_{e}i += sign * {im}")
        if i > 0:
            r1, i1 = _cmul("qa_r", "qa_i", f"g{i}_00r", f"g{i}_00i"); r2, i2 = _cmul("qb_r", "qb_i", f"g{i}_10r", f"g{i}_10i")
            w(J + f"na_r = {r1} + {r2}; na_i = {i1} + {i2}")
            r1, i1 = _cmul("qa_r", "qa_i", f"g{i}_01r", f"g{i}_01i"); r2, i2 = _cmul("qb_r", "qb_i", f"g{i}_11r", f"g{i}_11i")
            w(J + f"nb_r = {r1} + {r2}; nb_i = {i1} + {i2}")
            w(J + "qa_r = na_r; qa_i = na_i; qb_r = nb_r; qb_i = nb_i")
    # stores: part[lane, c, a, b, re/im], lane = lam * (PPT * SEG) + li ; one [SEG, B, 8] store per i
    w(I + f"optr = part_ptr + g.to(tl.int64)[:, None, None] * {K * 8} + tl.arange(0, 8)[None, None, :]")
    for i in range(CB):
        # tl.join appends the selector as the LAST dim: join(join(00, 10), join(01, 11)) is [.., a, b]
        w(I + f"jr = tl.join(tl.join(acc{i}_00r, acc{i}_10r), tl.join(acc{i}_01r, acc{i}_11r))")
        w(I + f"ji = tl.join(tl.join(acc{i}_00i, acc{i}_10i), tl.join(acc{i}_01i, acc{i}_11i))")
        w(I + f"tl.store(optr + col{i}[:, :, None] * 8, tl.reshape(tl.join(jr, ji), [SEG, {B}, 8]), mask=cm{i}[:, :, None])")
    w("")
    return "\n".join(o)


def _cache_dir():
    d = os.environ.get("PE_KERNEL_CACHE") or os.path.join(tempfile.gettempdir(), "photonic_engines_kernels")
    os.makedirs(d, exist_ok=True)
    return d


def get_kernel(K: int, L: int, B: int, mat_mode: str = "scalar"):
    """Compiled-on-demand JIT function for (K, L, B) (cached per process; source cached on disk)."""
    key = (K, L, B, mat_mode)
    if key in _KERNELS:
        return _KERNELS[key]
    src = generate_source(K, L, B, mat_mode)
    h = hashlib.sha1(src.encode()).hexdigest()[:10]
    name = f"pe_emitter_seq_k{K}_l{L}_b{B}_{mat_mode}_{h}"
    path = os.path.join(_cache_dir(), name + ".py")
    if not os.path.exists(path):
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(src)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, f"kernel_k{K}_l{L}_b{B}")
    _KERNELS[key] = fn
    return fn
