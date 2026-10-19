"""Triton kernel for the EmitterMPS sampler (docs/EMITTER_SAMPLER.md §3, §6).

Per step k (K = k kept slots, K1 = k-1 drawn rows), every LANE is (pure term λ, Gray segment)
and walks a contiguous segment of the Glynn sum over delta in {+1} x {+-1}^(K1-1) exactly as
`cofactors_triton._cofactors_kernel` does for Fock inputs (same Gray-code facts, same row
loads), with the scalar column sums replaced by chi x chi matrix chains:

    column sums   s_{j,l}(delta) = sum_i delta_i U[r_i, colmode[j, l]]           (L per slot)
    slot matrices G_j(delta) = sum_l s_{j,l} M~_{j,l}                             (2 x 2, complex)
    prefixes      P_c = G_{c-1} ... G_0,  suffixes Q_c = G_{K-1} ... G_{c+1}      (associative scans)
    p_c = P_c s0,  q_c^T = f^dag Q_c,   acc_c += sign(delta) p_c q_c^T           (2 x 2 per column)

The scans avoid any dynamic indexing along the chain: the forward scan runs on the shifted
sequence F_j = G_{j-1} (F_0 = I) and the backward scan on B_j = G_{j+1} (B_{K-1} = I), so two
shifted copies of the column sums are maintained (shifted *loads*, masked at the ends).
chi = 2 and L <= 2 are fixed in this first version.

Output: partial accumulators part[lane, c, a, b] (complex). The host reduces segments per
term, forms Tr(M~_{c,l} K_c), the candidate amplitudes and w = sum_terms |amp|^2 (torch),
and draws. Term decompositions come from `emitter_terms.batched_gap_terms` (all streams at once).
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from ..states import EmitterMPS
from .cofactors_triton import _binom_table, _ctz_table, _pow2_floor, plan_multi
from .emitter_reference import final_spin_distribution
from .emitter_terms import batched_gap_terms
from .emitter_triton_seq import get_kernel as get_seq_kernel


@triton.jit
def _cmm(a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
         b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i):
    """Scan combine: returns b @ a (complex 2x2, re/im split)."""
    c00r = b00r * a00r - b00i * a00i + b01r * a10r - b01i * a10i
    c00i = b00r * a00i + b00i * a00r + b01r * a10i + b01i * a10r
    c01r = b00r * a01r - b00i * a01i + b01r * a11r - b01i * a11i
    c01i = b00r * a01i + b00i * a01r + b01r * a11i + b01i * a11r
    c10r = b10r * a00r - b10i * a00i + b11r * a10r - b11i * a10i
    c10i = b10r * a00i + b10i * a00r + b11r * a10i + b11i * a10r
    c11r = b10r * a01r - b10i * a01i + b11r * a11r - b11i * a11i
    c11i = b10r * a01i + b10i * a01r + b11r * a11i + b11i * a11r
    return c00r, c00i, c01r, c01i, c10r, c10i, c11r, c11i


@triton.jit
def _cmm_rev(a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
             b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i):
    """reverse-scan combine: returns a @ b, so the suffix at c is G_{K-1} ... G_{c+1}."""
    return _cmm(b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i,
                a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i)


@triton.jit
def _slot_entry(s0r, s0i, s1r, s1i, mptr, mmask, e: tl.constexpr, diag: tl.constexpr):
    """Entry e of G = s_0 M_0 + s_1 M_1 (re, im); identity where mmask is False."""
    m0r = tl.load(mptr + e * 2, mask=mmask, other=0.0)
    m0i = tl.load(mptr + e * 2 + 1, mask=mmask, other=0.0)
    m1r = tl.load(mptr + 8 + e * 2, mask=mmask, other=0.0)
    m1i = tl.load(mptr + 8 + e * 2 + 1, mask=mmask, other=0.0)
    gr = s0r * m0r - s0i * m0i + s1r * m1r - s1i * m1i
    gi = s0r * m0i + s0i * m0r + s1r * m1i + s1i * m1r
    if diag:
        gr = tl.where(mmask, gr, 1.0)
    else:
        gr = tl.where(mmask, gr, 0.0)
    gi = tl.where(mmask, gi, 0.0)
    return gr, gi


@triton.jit
def _slot_matrices(s0r, s0i, s1r, s1i, mptr, mmask):
    """G = s_0 M_0 + s_1 M_1 for [SEG, K_PAD] column sums (label 0 and 1) and matrices at
    mptr (+ l*8 + e*2 : entry e of label l, re/im). Where mmask is False, G = I.
    """
    g00r, g00i = _slot_entry(s0r, s0i, s1r, s1i, mptr, mmask, 0, True)
    g01r, g01i = _slot_entry(s0r, s0i, s1r, s1i, mptr, mmask, 1, False)
    g10r, g10i = _slot_entry(s0r, s0i, s1r, s1i, mptr, mmask, 2, False)
    g11r, g11i = _slot_entry(s0r, s0i, s1r, s1i, mptr, mmask, 3, True)
    return g00r, g00i, g01r, g01i, g10r, g10i, g11r, g11i


@triton.jit(do_not_specialize=["K1", "K", "L", "TOTAL", "NLAB"])
def _emitter_cofactors_kernel(
    u_ptr,                             # U [m, m, 2] float64
    rows_ptr, cm_ptr,                  # rows [S, n] int64 ; colmode [S, K, 2] int64
    lane2term_ptr, term2stream_ptr, lstart_ptr,   # int64 tables
    mats_ptr, s0_ptr, f_ptr,           # mats [Λ, K, 2, 2, 2, 2] ; s0 [Λ, 2, 2] ; f [Λ, 2, 2]  (float64, last dim re/im)
    part_ptr,                          # out [TOTAL, K, 2, 2, 2]
    ctz_ptr,
    stride_um, stride_rs, stride_cs, stride_mt, stride_mj,
    K1, K, L, TOTAL, NLAB,
    K_PAD: tl.constexpr, SEG: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid * SEG + tl.arange(0, SEG)
    valid = g < TOTAL
    lam = tl.load(lane2term_ptr + g, mask=valid, other=0)
    s = tl.load(term2stream_ptr + lam, mask=valid, other=0)
    li = g - tl.load(lstart_ptr + lam, mask=valid, other=0)
    base = li.to(tl.int64) * L
    jj = tl.arange(0, K_PAD)
    jmask = jj < K
    fmask = valid[:, None] & (jj[None, :] >= 1) & jmask[None, :]          # F_j = G_{j-1}
    bmask = valid[:, None] & (jj[None, :] <= K - 2) & jmask[None, :]      # B_j = G_{j+1}
    lab1 = NLAB > 1

    rows_lane = rows_ptr + s * stride_rs
    cm_lane = cm_ptr + s * stride_cs                                        # [SEG]
    # column-mode pointers of the shifted slots (label 0 and 1)
    cmF = cm_lane[:, None] + (jj[None, :] - 1) * 2
    cmB = cm_lane[:, None] + (jj[None, :] + 1) * 2
    modF0 = tl.load(cmF, mask=fmask, other=0)
    modF1 = tl.load(cmF + 1, mask=fmask & lab1, other=0)
    modB0 = tl.load(cmB, mask=bmask, other=0)
    modB1 = tl.load(cmB + 1, mask=bmask & lab1, other=0)
    # matrices of the shifted slots for this lane's term
    mat_lane = mats_ptr + lam * stride_mt
    mF = mat_lane[:, None] + (jj[None, :] - 1) * stride_mj
    mB = mat_lane[:, None] + (jj[None, :] + 1) * stride_mj

    # ---- column sums at delta(base) --------------------------------------------
    gray0 = base ^ (base >> 1)
    sF0r = tl.zeros([SEG, K_PAD], tl.float64); sF0i = tl.zeros([SEG, K_PAD], tl.float64)
    sF1r = tl.zeros([SEG, K_PAD], tl.float64); sF1i = tl.zeros([SEG, K_PAD], tl.float64)
    sB0r = tl.zeros([SEG, K_PAD], tl.float64); sB0i = tl.zeros([SEG, K_PAD], tl.float64)
    sB1r = tl.zeros([SEG, K_PAD], tl.float64); sB1i = tl.zeros([SEG, K_PAD], tl.float64)
    for i in range(K1):
        sh = tl.maximum(i - 1, 0)
        bit = ((gray0 >> sh) & 1).to(tl.float64)
        d = tl.where(i == 0, 1.0, 1.0 - 2.0 * bit)
        r_i = tl.load(rows_lane + i, mask=valid, other=0)
        rowp = u_ptr + r_i[:, None] * stride_um
        sF0r += d[:, None] * tl.load(rowp + modF0 * 2, mask=fmask, other=0.0)
        sF0i += d[:, None] * tl.load(rowp + modF0 * 2 + 1, mask=fmask, other=0.0)
        sF1r += d[:, None] * tl.load(rowp + modF1 * 2, mask=fmask & lab1, other=0.0)
        sF1i += d[:, None] * tl.load(rowp + modF1 * 2 + 1, mask=fmask & lab1, other=0.0)
        sB0r += d[:, None] * tl.load(rowp + modB0 * 2, mask=bmask, other=0.0)
        sB0i += d[:, None] * tl.load(rowp + modB0 * 2 + 1, mask=bmask, other=0.0)
        sB1r += d[:, None] * tl.load(rowp + modB1 * 2, mask=bmask & lab1, other=0.0)
        sB1i += d[:, None] * tl.load(rowp + modB1 * 2 + 1, mask=bmask & lab1, other=0.0)

    # boundary vectors of the lane's term
    s0a_r = tl.load(s0_ptr + lam * 4 + 0, mask=valid, other=0.0); s0a_i = tl.load(s0_ptr + lam * 4 + 1, mask=valid, other=0.0)
    s0b_r = tl.load(s0_ptr + lam * 4 + 2, mask=valid, other=0.0); s0b_i = tl.load(s0_ptr + lam * 4 + 3, mask=valid, other=0.0)
    # f^dag = conj(f) as a row vector
    fa_r = tl.load(f_ptr + lam * 4 + 0, mask=valid, other=0.0); fa_i = -tl.load(f_ptr + lam * 4 + 1, mask=valid, other=0.0)
    fb_r = tl.load(f_ptr + lam * 4 + 2, mask=valid, other=0.0); fb_i = -tl.load(f_ptr + lam * 4 + 3, mask=valid, other=0.0)

    acc00r = tl.zeros([SEG, K_PAD], tl.float64); acc00i = tl.zeros([SEG, K_PAD], tl.float64)
    acc01r = tl.zeros([SEG, K_PAD], tl.float64); acc01i = tl.zeros([SEG, K_PAD], tl.float64)
    acc10r = tl.zeros([SEG, K_PAD], tl.float64); acc10i = tl.zeros([SEG, K_PAD], tl.float64)
    acc11r = tl.zeros([SEG, K_PAD], tl.float64); acc11i = tl.zeros([SEG, K_PAD], tl.float64)
    vsign = valid.to(tl.float64)

    for t in range(L):
        if t > 0:
            c = tl.load(ctz_ptr + t)
            g_t = base + t
            gp = g_t - 1
            grayp = gp ^ (gp >> 1)
            fct = -2.0 * (1.0 - 2.0 * (((grayp >> c) & 1).to(tl.float64)))
            r_c = tl.load(rows_lane + (c + 1), mask=valid, other=0)
            rowp = u_ptr + r_c[:, None] * stride_um
            sF0r += fct[:, None] * tl.load(rowp + modF0 * 2, mask=fmask, other=0.0)
            sF0i += fct[:, None] * tl.load(rowp + modF0 * 2 + 1, mask=fmask, other=0.0)
            sF1r += fct[:, None] * tl.load(rowp + modF1 * 2, mask=fmask & lab1, other=0.0)
            sF1i += fct[:, None] * tl.load(rowp + modF1 * 2 + 1, mask=fmask & lab1, other=0.0)
            sB0r += fct[:, None] * tl.load(rowp + modB0 * 2, mask=bmask, other=0.0)
            sB0i += fct[:, None] * tl.load(rowp + modB0 * 2 + 1, mask=bmask, other=0.0)
            sB1r += fct[:, None] * tl.load(rowp + modB1 * 2, mask=bmask & lab1, other=0.0)
            sB1i += fct[:, None] * tl.load(rowp + modB1 * 2 + 1, mask=bmask & lab1, other=0.0)
            sign = vsign * (1.0 - 2.0 * ((g_t & 1).to(tl.float64)))
        else:
            sign = vsign * (1.0 - 2.0 * ((base & 1).to(tl.float64)))
        # slot matrices of the shifted sequences (identity where masked)
        F00r, F00i, F01r, F01i, F10r, F10i, F11r, F11i = _slot_matrices(sF0r, sF0i, sF1r, sF1i, mF, fmask)
        B00r, B00i, B01r, B01i, B10r, B10i, B11r, B11i = _slot_matrices(sB0r, sB0i, sB1r, sB1i, mB, bmask)
        P0, P1, P2, P3, P4, P5, P6, P7 = tl.associative_scan(
            (F00r, F00i, F01r, F01i, F10r, F10i, F11r, F11i), 1, _cmm)                 # P_c = G_{c-1} ... G_0
        Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7 = tl.associative_scan(
            (B00r, B00i, B01r, B01i, B10r, B10i, B11r, B11i), 1, _cmm_rev, reverse=True)  # Q_c = G_{K-1} ... G_{c+1}
        # p_c = P_c s0
        pa_r = P0 * s0a_r[:, None] - P1 * s0a_i[:, None] + P2 * s0b_r[:, None] - P3 * s0b_i[:, None]
        pa_i = P0 * s0a_i[:, None] + P1 * s0a_r[:, None] + P2 * s0b_i[:, None] + P3 * s0b_r[:, None]
        pb_r = P4 * s0a_r[:, None] - P5 * s0a_i[:, None] + P6 * s0b_r[:, None] - P7 * s0b_i[:, None]
        pb_i = P4 * s0a_i[:, None] + P5 * s0a_r[:, None] + P6 * s0b_i[:, None] + P7 * s0b_r[:, None]
        # q_c^T = f^dag Q_c :  q_b = sum_a fdag_a Q[a, b]
        qa_r = fa_r[:, None] * Q0 - fa_i[:, None] * Q1 + fb_r[:, None] * Q4 - fb_i[:, None] * Q5
        qa_i = fa_r[:, None] * Q1 + fa_i[:, None] * Q0 + fb_r[:, None] * Q5 + fb_i[:, None] * Q4
        qb_r = fa_r[:, None] * Q2 - fa_i[:, None] * Q3 + fb_r[:, None] * Q6 - fb_i[:, None] * Q7
        qb_i = fa_r[:, None] * Q3 + fa_i[:, None] * Q2 + fb_r[:, None] * Q7 + fb_i[:, None] * Q6
        sg = sign[:, None]
        acc00r += sg * (pa_r * qa_r - pa_i * qa_i); acc00i += sg * (pa_r * qa_i + pa_i * qa_r)
        acc01r += sg * (pa_r * qb_r - pa_i * qb_i); acc01i += sg * (pa_r * qb_i + pa_i * qb_r)
        acc10r += sg * (pb_r * qa_r - pb_i * qa_i); acc10i += sg * (pb_r * qa_i + pb_i * qa_r)
        acc11r += sg * (pb_r * qb_r - pb_i * qb_i); acc11i += sg * (pb_r * qb_i + pb_i * qb_r)

    m2 = valid[:, None] & jmask[None, :]
    optr = part_ptr + (g[:, None].to(tl.int64) * K + jj[None, :]) * 8
    tl.store(optr + 0, acc00r, mask=m2); tl.store(optr + 1, acc00i, mask=m2)
    tl.store(optr + 2, acc01r, mask=m2); tl.store(optr + 3, acc01i, mask=m2)
    tl.store(optr + 4, acc10r, mask=m2); tl.store(optr + 5, acc10i, mask=m2)
    tl.store(optr + 6, acc11r, mask=m2); tl.store(optr + 7, acc11i, mask=m2)


# ---------------------------------------------------------------- host side

# Launch configuration per K_PAD tier and kernel implementation; tune with bench/bench_emitter_tune.py.
#   scan: (SEG, num_warps). The kernel holds ~60 [SEG, K_PAD] float64 tensors, so SEG shrinks with
#         K_PAD (tuned on H100: 8:(16,4), 16:(4,2), 32:(2,2)).
#   seq:  (SEG, B) — B column blocks (threads) per lane, num_warps = SEG * B / 32; B keeps the
#         columns per thread (ceil(K / B)) at ~2 so the per-thread state fits the register file.
#         SEG is a compile-time constant: one kernel per (K, L, B, SEG).
# Per-K_PAD tier.  seq entries are (SEG, B) or (SEG, B, mat_mode); without a third element the
# process-wide SEQ_MAT_MODE is used.  The best (SEG, B, mat_mode) varies much more here than the
# launch knobs do -- best/worst spread over one K grows 1.24x at K = 10 to 5.7x at K = 22
# (run 20260916_155718) -- so a wrong entry is the dominant implementation cost on this kernel.
_CFG = {
    "scan": {2: (32, 2), 4: (32, 2), 8: (16, 4), 16: (4, 2), 32: (2, 2), 64: (2, 2)},
    "seq": {2: (32, 2), 4: (16, 4), 8: (8, 4), 16: (8, 4), 32: (4, 8), 64: (4, 16)},
}   # values unchanged: the 2026-09-16 candidates did not reproduce, see below.
#
# Two sweeps (runs 20260916_155718 and _191446) agree on what does NOT matter and disagree on
# the exact winner, so nothing here was changed on their basis:
#   * mat_mode (scalar vs hoist) is NOISE -- the two always land adjacent in the ranking,
#     within 0.2-2.7 % at every K.  The 1.01-1.02x "gain" the first sweep reported did not
#     reproduce (at K = 22 hoist won by 0.9 % in one run, scalar by 0.2 % in the other).
#   * B is the dominant knob (worst config 2.0x off at K = 18, 2.1x at K = 22) and its optimum
#     MOVES INSIDE tier 32: at 32 streams, K = 18 wants B = 4 (3.74 ms against 4.65 for B = 8)
#     and K = 22 wants B = 8 (44.6 against 51.8).  That is what _CFG_BY_K below is for.
#   * the optimum does NOT depend on the stream count: at K = 22 the same (SEG 4, B 8) wins at
#     8, 32, 128 and 512 streams, and the rate is flat at 18.5-21.5 G units/s across that 64x
#     range (run 20260916_200342).  An earlier reading of a stream dependence was wrong -- the
#     two sweeps used different grids, see below.
# Exact-K overrides, tried before the tier, for when the optimum moves inside a tier.
_CFG_BY_K = {
    "scan": {},
    # K = 18 wants B = 4 (3.74 ms) where the tier-32 entry gives B = 8 (4.65 ms): 1.24x.
    # This entry was proposed, withdrawn and re-applied.  It was withdrawn because two sweeps
    # disagreed; the cause was that bench_emitter_tune's default --blocks grid was
    # {K_PAD/4, K_PAD/2, K_PAD} = {8, 16, 32}, so B = 4 was never tried at tier 32 in the first
    # sweep.  The grid now includes K_PAD/8.  Verified at 32 streams with both B values present;
    # stream-independence is established at K = 22, not directly at K = 18.
    "seq": {18: (8, 4)},
}
_CFG_OVERRIDE = {"scan": {}, "seq": {}}
_CFG_BY_K_OVERRIDE = {"scan": {}, "seq": {}}
IMPL = "seq"
# impl "auto": the CUDA kernel (cuda_emitter) for launches with >= AUTO_LANES lanes per term (the
# frontier steps, where it is 1.2-1.5x the seq kernel), the seq kernel below (the middle of the
# chain at 64-256 shots, where the CUDA kernel is 0.5-0.6x; docs/EMITTER_SAMPLER.md §8, runs
# 20260927_153353 / _205524).  Requires cupy; falls back to "seq" without it.
AUTO_LANES = 1.5
SEQ_MAT_MODE = "scalar"      # or "hoist": term matrices kept in registers across the Gray walk
# Launch knobs the (SEG, B) table cannot reach: num_warps is DERIVED here (SEG*B/32), so the
# register/occupancy trade has never been swept independently.  On the Fock kernel that trade was
# worth 1.32x at the widest tier and the wrong value cost 1.9x (H100 sweep, 2026-09),
# so it is the first thing to try here.  None = let Triton choose; set via set_launch().
SEQ_MAXNREG = None
SEQ_NUM_STAGES = None


def set_launch(maxnreg=..., num_stages=...):
    """Override the register cap / pipeline depth for this process (... = leave unchanged)."""
    global SEQ_MAXNREG, SEQ_NUM_STAGES
    if maxnreg is not ...:
        SEQ_MAXNREG = maxnreg
    if num_stages is not ...:
        SEQ_NUM_STAGES = num_stages


def _launch_kw():
    """Extra kernel launch arguments, dropped silently on a Triton that does not accept them."""
    kw = {}
    if SEQ_MAXNREG is not None:
        kw["maxnreg"] = int(SEQ_MAXNREG)
    if SEQ_NUM_STAGES is not None:
        kw["num_stages"] = int(SEQ_NUM_STAGES)
    return kw


def set_impl(impl):
    """Default kernel implementation: "seq" or "scan"."""
    global IMPL
    assert impl in ("seq", "scan")
    IMPL = impl


def set_config(cfg, impl=None):
    """Override the {K_PAD: (SEG, num_warps)} table for this process (merged);
    passing cfg=None resets it.
    """
    impl = impl or IMPL
    _CFG_OVERRIDE[impl] = {} if cfg is None else {int(k): tuple(v) for k, v in cfg.items()}


def set_config_k(cfg, impl=None):
    """Override the exact-K table {K: (SEG, B[, mat_mode])} for this process; None resets."""
    impl = impl or IMPL
    _CFG_BY_K_OVERRIDE[impl] = {} if cfg is None else {int(k): tuple(v) for k, v in cfg.items()}


def get_config(K_PAD, impl=None, K=None):
    """(SEG, B[, mat_mode]) for this step: the exact-K entry if there is one, else the tier."""
    impl = impl or IMPL
    if K is not None:
        byk = dict(_CFG_BY_K[impl]); byk.update(_CFG_BY_K_OVERRIDE[impl])
        if K in byk:
            return byk[K]
    cfg = dict(_CFG[impl])
    cfg.update(_CFG_OVERRIDE[impl])
    return cfg[K_PAD]


_MODES_CACHE = {}


def _modes_tensor(state, dev):
    key = (state.modes, str(dev))               # by value: ids of freed states get reused
    t = _MODES_CACHE.get(key)
    if t is None:
        t = torch.tensor([list(mk) for mk in state.modes], dtype=torch.long, device=dev)
        _MODES_CACHE[key] = t
    return t


def emitter_multiplicity_tables(rows, K1):
    """Clifford & Clifford 2020 tables for the seq kernel. rows [S, >= K1] int64 ->
    drow / mult / radix [S, K1] int64 (positions: distinct single rows first, radix 2 = implicit
    binary digits of the term index; then repeated-row groups, radix s + 1; then the halved top
    digit: a single row with radix 1 if there is one, else an odd-multiplicity group with radix
    (s + 1) / 2, else nothing), n2 [S] int64 (number of binary digits), N [S] int64 (terms per
    stream), scale [S] float64 (2^-K1, x2 when halved).
    """
    S = rows.shape[0]
    dev = rows.device
    if K1 == 0:
        z = torch.zeros((S, 1), dtype=torch.long, device=dev)
        return z, z, z + 1, torch.zeros(S, dtype=torch.long, device=dev), torch.ones(S, dtype=torch.long, device=dev), \
            torch.ones(S, dtype=torch.float64, device=dev)
    rs, _ = torch.sort(rows[:, :K1], dim=1)
    ar = torch.arange(K1, device=dev)
    newgrp = torch.ones(S, K1, dtype=torch.bool, device=dev)
    if K1 > 1:
        newgrp[:, 1:] = rs[:, 1:] != rs[:, :-1]
    gid = newgrp.long().cumsum(1) - 1
    D = gid[:, -1] + 1
    mult = torch.zeros(S, K1, dtype=torch.long, device=dev).scatter_add_(1, gid, torch.ones_like(gid))
    drow = torch.zeros(S, K1, dtype=torch.long, device=dev).scatter_(1, gid, rs)
    active = ar[None, :] < D[:, None]
    is_bin = active & (mult == 1)
    is_high = active & (mult >= 2)
    key = torch.where(is_bin, 0, torch.where(is_high, 1, 3))                     # 3: padding
    radix = torch.where(is_bin, 2, torch.where(is_high, mult + 1, 1))
    # top (halved) digit
    has_bin = is_bin.any(1)
    top_bin = torch.where(is_bin, ar[None, :], -1).max(1).values                 # last single row
    odd_high = is_high & (mult % 2 == 1)
    has_odd = odd_high.any(1) & ~has_bin
    top_odd = torch.where(odd_high, ar[None, :], K1).min(1).values
    top = torch.where(has_bin, top_bin, torch.where(has_odd, top_odd, -1))
    sidx = torch.arange(S, device=dev)
    sel = top >= 0
    key[sidx[sel], top[sel]] = 2
    radix[sidx[sel], top[sel]] = torch.where(has_bin, 1, (mult[sidx, top.clamp_min(0)] + 1) // 2)[sel]
    perm = torch.argsort(key * K1 + ar[None, :], dim=1)
    drow, mult, radix = drow.gather(1, perm), mult.gather(1, perm), radix.gather(1, perm)
    mult = torch.where(radix > 1, mult, torch.where(key.gather(1, perm) == 2, mult, 0))   # padding: 0
    n2 = (key == 0).sum(1)
    N = radix.prod(1)
    scale = torch.where(sel, 2.0, 1.0).to(torch.float64) * (2.0 ** (-K1))
    return drow.contiguous(), mult.contiguous(), radix.contiguous(), n2, N, scale


def emitter_cofactor_partials(U2, rows, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
                              K1, K, nlab, L, total, seg=None, num_warps=None):
    """Scan kernel launch (binary Gray walk over 2^(K1-1) terms per pure term)."""
    dev = U2.device
    K_PAD = max(2, triton.next_power_of_2(K))
    cseg, cwarps = get_config(K_PAD, "scan")
    seg = seg or cseg
    num_warps = num_warps or cwarps
    part = torch.empty(total, K, 2, 2, 2, dtype=torch.float64, device=dev)
    ctz = _ctz_table(L, dev)
    grid = (triton.cdiv(total, seg),)
    _emitter_cofactors_kernel[grid](
        U2, rows, colmode, lane2term, term2stream, lstart, mats2, s02, f2, part, ctz,
        U2.stride(0), rows.stride(0), colmode.stride(0), mats2.stride(0), mats2.stride(1),
        K1, K, L, total, nlab,
        K_PAD=K_PAD, SEG=seg, num_warps=num_warps, **_launch_kw(),
    )
    return part


def emitter_cofactor_partials_seq(U2, tables, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
                                  K, nlab, L, total, seg=None, num_warps=None):
    """Seq kernel launch (mixed-radix walk over the collapsed terms; tables from
    `emitter_multiplicity_tables`).
    """
    dev = U2.device
    drow, mult, radix, n2, N = tables
    K_PAD = max(2, triton.next_power_of_2(K))
    ccfg = get_config(K_PAD, "seq", K)
    cseg, cB = ccfg[0], ccfg[1]
    mat_mode = ccfg[2] if len(ccfg) > 2 else SEQ_MAT_MODE
    seg = seg or cseg
    B = num_warps or cB                            # second config entry is the block count
    num_warps = max(1, seg * B // 32)
    part = torch.empty(total, K, 2, 2, 2, dtype=torch.float64, device=dev)
    ctz = _ctz_table(L, dev)
    grid = (triton.cdiv(total, seg),)
    kern = get_seq_kernel(K, nlab, B, mat_mode)
    kern[grid](
        U2, drow, mult, radix, n2, N, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
        _binom_table(dev), part, ctz,
        U2.stride(0), drow.stride(0), colmode.stride(0), mats2.stride(0), mats2.stride(1),
        L, total, SEG=seg, num_warps=num_warps, **_launch_kw(),
    )
    return part


_BMM_BUDGET = 2 * 2 ** 30      # bytes for the padded candidate product, see emitter_step_weights


def emitter_step_weights(U, state, Tsorted, rows_k, s0, mats, f, term2stream, S,
                         target_total_lanes=1 << 17, max_lanes=1 << 16, seg=None, num_warps=None, timing=None, impl=None,
                         seq_kw=None, cof=None):
    """W [S, m] for one step from the term tensors (s0 [Λ, chi], mats [Λ, K, L, chi, chi],
    f [Λ, chi]) and the drawn rows rows_k [S, K1]. Returns (w, mean lanes per term).
    impl: "seq" | "scan" (Triton), "cuda" (`cuda_emitter`, thread-per-lane NVRTC kernel), "auto"
    (cuda when the launch has >= AUTO_LANES lanes per term, else seq) or "cuda_emul" (the same
    kernel code compiled for the CPU; seq_kw = {tpl, gs} overrides).
    """
    dev = U.device
    impl = impl or IMPL
    Lam = s0.shape[0]
    K = Tsorted.shape[1]
    K1 = rows_k.shape[1]
    Lab = mats.shape[2]
    m = U.shape[0]
    term2stream = term2stream.contiguous()
    modes = _modes_tensor(state, dev)                                                     # [n, L]
    cm = modes[Tsorted]                                                                   # [S, K, L]
    if Lab == 2:
        colmode = cm
        mats2 = torch.view_as_real(mats.to(torch.complex128).contiguous())               # [Λ, K, 2, 2, 2, 2]
    else:
        colmode = torch.zeros((S, K, 2), dtype=torch.long, device=dev)
        colmode[:, :, :Lab] = cm
        m2 = torch.zeros((Lam, K, 2, 2, 2), dtype=torch.complex128, device=dev)
        m2[:, :, :Lab] = mats
        mats2 = torch.view_as_real(m2)
    colmode = colmode.contiguous()
    s02 = torch.view_as_real(s0.to(torch.complex128).contiguous())
    f2 = torch.view_as_real(f.to(torch.complex128).contiguous())
    U2 = torch.view_as_real(U.to(torch.complex128).contiguous())
    rows = rows_k.contiguous() if K1 else torch.zeros((S, 1), dtype=torch.long, device=dev)
    if impl in ("seq", "cuda", "cuda_emul", "auto"):
        drow, mult, radix, n2, N, scale = emitter_multiplicity_tables(rows, K1)
        tables = (drow, mult, radix, n2, N)
        Nt = N[term2stream]                                                               # terms per pure term
        Lseg, nl, lstart, lane2term, total = plan_multi(Nt, target_total_lanes)
        if impl == "auto":
            impl = "cuda" if total / max(1, Nt.shape[0]) >= AUTO_LANES else "seq"
        if impl != "seq":                                # docs/EMITTER_SAMPLER.md §8: the Fock-style tables (same N)
            from .cuda_emitter import emitter_tables
            tables = emitter_tables(rows, K1)
            scale = tables[-1]
        if timing is not None and dev.type == "cuda":
            torch.cuda.synchronize(); timing["_t0"] = time.time()
        if impl == "seq":
            part = emitter_cofactor_partials_seq(U2, tables, colmode, lane2term, term2stream,
                                                 lstart, mats2, s02, f2, K, Lab, Lseg, total, seg, num_warps)
        elif impl == "cuda":
            from .cuda_emitter import emitter_cofactor_partials_cuda
            part = emitter_cofactor_partials_cuda(U2, tables, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
                                                  K1, K, Lab, Lseg, total)
        else:
            from .cuda_emitter import emitter_cofactor_partials_emul
            part = emitter_cofactor_partials_emul(U2, tables, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
                                                  K1, K, Lab, Lseg, total, **(seq_kw or {}))
        if timing is not None and dev.type == "cuda":
            torch.cuda.synchronize(); timing["kernel"] += time.time() - timing.pop("_t0")
        acc = torch.zeros((Lam, K, 2, 2, 2), dtype=torch.float64, device=dev)
        acc.index_add_(0, lane2term, part)
        Kc = torch.view_as_complex(acc) * scale[term2stream][:, None, None, None]
        lanes = float(nl.double().mean())
    else:
        nterms = (1 << max(K1 - 1, 0)) if K1 else 1
        lanes = _pow2_floor(max(1, min(nterms, max_lanes, target_total_lanes // max(Lam, 1))))
        Lseg = nterms // lanes
        total = Lam * lanes
        lane2term = torch.repeat_interleave(torch.arange(Lam, device=dev), lanes)
        lstart = torch.arange(Lam, device=dev) * lanes
        if timing is not None and dev.type == "cuda":
            torch.cuda.synchronize(); timing["_t0"] = time.time()
        part = emitter_cofactor_partials(U2, rows, colmode, lane2term, term2stream,
                                         lstart, mats2, s02, f2, K1, K, Lab, Lseg, total, seg, num_warps)
        if timing is not None and dev.type == "cuda":
            torch.cuda.synchronize(); timing["kernel"] += time.time() - timing.pop("_t0")
        Kc = torch.view_as_complex(part.view(Lam, lanes, K, 2, 2, 2).sum(1)) * (2.0 ** (-(K1 - 1)) if K1 >= 1 else 1.0)
    # Tr(M~_{c,l} K_c) per term and column, then the candidate amplitudes stream by stream:
    # amp[x, r] = sum_{c,l} U[r, cm[s(x), c, l]] tr[x, c, l]  as a padded bmm over the terms of
    # each stream ([S, R_max, K L] @ [S, K L, m]) - never the [m, Λ, K, L] gather.
    tr = (mats * Kc.transpose(-1, -2).unsqueeze(2)).sum((-1, -2))                          # [Λ, K, L]
    if cof is not None:
        cof["Kc"], cof["tr"] = Kc, tr
    Rs = torch.bincount(term2stream, minlength=S)
    start = torch.cumsum(Rs, 0) - Rs
    pos = torch.arange(Lam, device=dev) - start[term2stream]
    tr_flat = tr.reshape(Lam, K * Lab)
    U128 = U.to(torch.complex128)
    m = U.shape[-1]
    Rs_l, start_l = Rs.tolist(), start.tolist()
    w = torch.empty((S, m), dtype=torch.float64, device=dev)
    # chunks of streams so that the padded [chunk, R_max, m] product and the [chunk, K L, m]
    # gather stay below ~2 GB (n = 27, m = 324, 256 shots, R_max in the thousands went OOM)
    budget = _BMM_BUDGET
    s0i = 0
    while s0i < S:
        s1i, rmax = s0i, 0
        while s1i < S:
            rmax_new = max(rmax, Rs_l[s1i])
            need = (s1i + 1 - s0i) * (rmax_new * (m + K * Lab) + K * Lab * m) * 16
            if s1i > s0i and need > budget:
                break
            rmax = rmax_new
            s1i += 1
        cs = s1i - s0i
        t0i, t1i = start_l[s0i], start_l[s1i - 1] + Rs_l[s1i - 1]
        tr_pad = torch.zeros((cs, rmax, K * Lab), dtype=torch.complex128, device=dev)
        tr_pad[term2stream[t0i:t1i] - s0i, pos[t0i:t1i]] = tr_flat[t0i:t1i]
        Ucm = U128[:, cm[s0i:s1i].reshape(cs, K * Lab)].permute(1, 2, 0)                    # [cs, K L, m]
        amp = torch.bmm(tr_pad, Ucm)                                                      # [cs, R_max, m]
        w[s0i:s1i] = (amp.real ** 2 + amp.imag ** 2).sum(1)
        s0i = s1i
    return w, lanes


def _rng(t: torch.Tensor, mx: bool = False) -> str:
    """min..max (or max) of |t| for an error message; 'empty' when no term survived."""
    if t.numel() == 0:
        return "empty"
    a = t.abs()
    return f"{float(a.max()):.3e}" if mx else f"{float(a.min()):.3e}..{float(a.max()):.3e}"


def sample_triton(state: EmitterMPS, U: torch.Tensor, shots: int,
                          generator: torch.Generator | None = None, device: str = "cuda",
                          target_total_lanes=1 << 17, seg=None, num_warps=None, return_stats=False, impl=None,
                          keep=None, lost_labels=None, final_ids=None):
    """Exact samples [shots, m] with the Triton cofactor kernel (chi = 2, L <= 2); impl "seq" | "scan".
    keep [shots, k]: surviving emission slots per stream; lost_labels [shots, n] / final_ids
    [shots]: the label unravelling of the traced slots (see `emitter_sampler.sample_torch`).
    """
    import time
    dev = torch.device(device)
    assert state.chi == 2 and state.num_labels <= 2, "Triton version: chi = 2, L <= 2"
    if impl == "cuda":
        from . import cuda_emitter
        if not cuda_emitter.available():
            raise RuntimeError("emitter kernel impl='cuda' needs cupy and a CUDA device")
    if impl == "auto":
        from . import cuda_emitter
        if not cuda_emitter.available():
            impl = "seq"
    U = torch.as_tensor(U).to(dev, torch.complex128)
    m, n = U.shape[0], state.n
    S = shots
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    if final_ids is not None:
        fid = torch.as_tensor(final_ids, dtype=torch.long).cpu()
    elif state.final is None:
        pf = final_spin_distribution(state)
        fid = torch.multinomial(pf / pf.sum(), S, replacement=True, generator=generator)
    else:
        fid = torch.zeros(S, dtype=torch.long)
    lab = None if lost_labels is None else torch.as_tensor(lost_labels, dtype=torch.long).to(dev)
    from .emitter_sampler import photon_order
    sigma = photon_order(S, n, keep, generator)
    n_k = sigma.shape[1]
    rows = torch.zeros((S, n_k), dtype=torch.long, device=dev)
    stats = {"R": [], "step_s": [], "terms_s": 0.0, "kernel_s": 0.0, "post_s": 0.0, "lanes": []}
    timing = {"kernel": 0.0}
    for k in range(1, n_k + 1):
        t0 = time.time()
        Tsorted = torch.sort(sigma[:, :k], dim=1).values
        finals = (torch.eye(2, dtype=torch.complex128, device=dev)[fid.to(dev)] if state.final is None
                  else torch.as_tensor(state.final, dtype=torch.complex128, device=dev).unsqueeze(0).expand(S, 2))
        s0, mats, f, term2stream = batched_gap_terms(state, Tsorted.to(dev), finals, labels=lab, U=U)
        stats["R"].append(s0.shape[0] / S)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time(); stats["terms_s"] += t1 - t0
        kb = timing["kernel"]
        w, lanes = emitter_step_weights(U, state, Tsorted.to(dev), rows[:, :k - 1], s0, mats, f, term2stream, S,
                                        target_total_lanes, seg=seg, num_warps=num_warps, timing=timing, impl=impl)
        stats["lanes"].append(lanes)
        w = w.clamp_min(0)
        bad = ~torch.isfinite(w).all(dim=1) | (w.sum(dim=1) <= 0)
        if bad.any():
            cnt = torch.bincount(term2stream, minlength=S)
            bs = bad.nonzero()[:, 0]
            raise RuntimeError(f"sample_triton: step k={k}/{n_k}: {int(bad.sum())} of {S} streams have non-finite or zero weights; "
                               f"terms/stream of bad ones {cnt[bs][:8].tolist()}, "
                               f"|s0| range {_rng(s0)}, |f| range {_rng(f)}, |mats| max {_rng(mats, True)}, "
                               f"finite w rows {int(torch.isfinite(w).all(dim=1).sum())}, "
                               f"kept slots of first bad stream {Tsorted[bs[0]].tolist()}"
                               + (f", lost labels {lab[bs[0]].tolist()}" if lab is not None else ""))
        w = w / w.sum(dim=1, keepdim=True)
        rho = torch.multinomial(w.cpu(), 1, generator=generator).squeeze(1)
        rows[:, k - 1] = rho.to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        stats["post_s"] += (time.time() - t1) - (timing["kernel"] - kb)
        stats["step_s"].append(time.time() - t0)
    stats["kernel_s"] = timing["kernel"]
    occ = torch.zeros((S, m), dtype=torch.long, device=dev)
    occ.scatter_add_(1, rows, torch.ones_like(rows))
    return (occ, stats) if return_stats else occ
