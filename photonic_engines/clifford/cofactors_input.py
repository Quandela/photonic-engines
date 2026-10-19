"""
Triton kernel for the INPUT-side collapse (docs/ALGORITHMS.md, algo "input"): the generalised
Clifford & Clifford algorithm B of Brod & Oszmaniec (arXiv:1906.06696) in this engine's vocabulary.

Per photon step k (K = k columns sigma(1..k) with D distinct input modes of multiplicities s_mu,
K1 = k-1 chosen output rows): the Glynn sum signs the COLUMNS instead of the rows,
    Q_mu = 2^-K sum_v [prod_nu (-1)^v_nu C(s_nu, v_nu)] (s_mu - 2 v_mu) prod_{i<K} (sum_nu (s_nu - 2 v_nu) A[dp, rows_i, dcol_nu])
over v in prod_nu {0..s_nu} (prod (s_nu + 1) terms, halved by complement symmetry when a
multiplicity is odd), and the candidate amplitudes are amp[rho] = sum_mu A[dp, rho, dcol_mu] Q_mu.
Same reflected mixed-radix Gray walk, lane plan, binomial-ratio weight update and flat lane grid as
`_cofactors_multi_kernel` (cofactors_triton.py); what changes is the axis that is signed (the
distinct input columns, `dcol`/`mult`/`radix` from `multiplicity_tables` applied to sigma) and the
accumulator (the FULL product over the K1 rows times the signed coefficient of column mu, one
value per distinct column, instead of the all-but-one products over the K columns). The output
part [TOTAL, K, 2] is laid out like the other kernels' so `draw_step` consumes it unchanged, with
`dcol` (padded to K) passed as the column table: entries mu >= D hold Q = 0.

On a collision-free input every s_mu = 1, so the count is 2^(K-1) whatever the output collisions:
twice the 2018 sweep (2^(K1-1)) and never the 2020 collapse. That is the measured content of the
"input" row of T1; the form is what pays off on binned inputs (Brod & Oszmaniec's lossy setting).
"""
import torch
import triton
import triton.language as tl

from .cofactors_triton import _binom_table, _cmul, _imul, get_config, multiplicity_tables, plan_multi


@triton.jit(do_not_specialize=["K1", "K", "L", "TOTAL"])
def _qvec_input_kernel(
    a_ptr,                          # A [B, m, n, 2] float64
    dcol_ptr, mult_ptr, radix_ptr,  # per stream [S, K]: distinct input columns (int64), multiplicities, radices (int32)
    rows_ptr, dp_ptr,               # rows [S, n] int64 (the K1 chosen output rows), dp [S] int64
    lane2s_ptr, lstart_ptr, n_ptr,  # lane -> stream, first lane of stream, terms per stream
    binom_ptr,                      # [32, 32] float64 C(s, v)
    part_ptr,                       # out [TOTAL, K, 2]: per-lane partial Q_mu, mu < K (zero beyond D)
    stride_ab, stride_am, stride_an, stride_ds, stride_rs,
    K1, K, L, TOTAL,
    K_PAD: tl.constexpr, SEG: tl.constexpr,
):
    """Input-side collapsed Glynn sweep (module docstring). jj indexes both the signed axis
    (distinct columns, mult/radix padded with 0/1 beyond D) and the product axis (the K1 rows,
    row sums padded with 1 beyond K1).
    """
    pid = tl.program_id(0)
    g = pid * SEG + tl.arange(0, SEG)
    valid = g < TOTAL
    s = tl.load(lane2s_ptr + g, mask=valid, other=0)
    li = g - tl.load(lstart_ptr + s, mask=valid, other=0)
    Ns = tl.load(n_ptr + s, mask=valid, other=1)
    base = li * L                                              # int64 [SEG]
    jj = tl.arange(0, K_PAD)
    jmask = jj < K                                             # signed axis (distinct columns, padded)
    imask = jj < K1                                            # product axis (rows)
    m2 = valid[:, None] & jmask[None, :]
    mi = valid[:, None] & imask[None, :]

    dp = tl.load(dp_ptr + s, mask=valid, other=0)
    rws = tl.load(rows_ptr + s[:, None] * stride_rs + jj[None, :], mask=mi, other=0)   # [SEG, K_PAD] output rows
    a_lane = a_ptr + dp * stride_ab
    rowofs = rws * stride_am
    dcol_lane = dcol_ptr + s * stride_ds

    R = tl.load(radix_ptr + s[:, None] * stride_ds + jj[None, :], mask=m2, other=1)      # int32 [SEG, K_PAD]
    Mu = tl.load(mult_ptr + s[:, None] * stride_ds + jj[None, :], mask=m2, other=0)      # int32
    Pinc = tl.associative_scan(R.to(tl.float64), 1, _imul)
    Pexc = (Pinc / R.to(tl.float64)).to(tl.int64)
    d = ((base[:, None] // Pexc) % R.to(tl.int64)).to(tl.int32)
    v = tl.zeros([SEG, K_PAD], tl.int32)
    par0 = tl.zeros([SEG], tl.int32)
    for i in range(K):
        nu0 = K - 1 - i
        isnu0 = jj[None, :] == nu0
        d_nu0 = tl.sum(tl.where(isnu0, d, 0), axis=1)
        r_nu0 = tl.sum(tl.where(isnu0, R, 0), axis=1)
        v_nu0 = tl.where((par0 & 1) == 0, d_nu0, r_nu0 - 1 - d_nu0)
        v = tl.where(isnu0, v_nu0[:, None], v)
        par0 += v_nu0
    bin = tl.load(binom_ptr + Mu * 32 + v, mask=m2, other=1.0)
    w = tl.reduce(tl.where((v & 1) == 1, -bin, bin), 1, _imul)                          # [SEG] float64
    # row sums over the signed columns: rs_i = sum_nu coef_nu A[dp, rows_i, dcol_nu]; padded rows hold 1
    coef = (Mu - 2 * v).to(tl.float64)                                                   # [SEG, K_PAD], 0 beyond D
    rs_re = tl.where(imask[None, :], 0.0, 1.0) + tl.zeros([SEG, K_PAD], tl.float64)
    rs_im = tl.zeros([SEG, K_PAD], tl.float64)
    for nu1 in range(K):
        c_nu1 = tl.sum(tl.where(jj[None, :] == nu1, coef, 0.0), axis=1)
        col1 = tl.load(dcol_lane + nu1, mask=valid, other=0)
        ptr1 = a_lane[:, None] + rowofs + col1[:, None] * stride_an
        a_re1 = tl.load(ptr1, mask=mi, other=0.0)
        a_im1 = tl.load(ptr1 + 1, mask=mi, other=0.0)
        rs_re += c_nu1[:, None] * a_re1
        rs_im += c_nu1[:, None] * a_im1

    acc_re = tl.zeros([SEG, K_PAD], tl.float64)
    acc_im = tl.zeros([SEG, K_PAD], tl.float64)
    ok = valid & (base < Ns)
    wt = tl.where(ok, w, 0.0)
    p_re, p_im = tl.reduce((rs_re, rs_im), 1, _cmul)                                     # [SEG] full product
    acc_re += (wt * p_re)[:, None] * coef
    acc_im += (wt * p_im)[:, None] * coef

    rem = tl.where(valid, tl.minimum(Ns - base, L), 0)
    Lp = tl.max(rem, axis=0)
    for t in range(1, Lp):
        ok = valid & (base + t < Ns)
        can = (d < R - 1) & jmask[None, :]
        nu = tl.min(tl.where(can, jj[None, :], K_PAD), axis=1)                          # [SEG] int32
        isnu = jj[None, :] == nu[:, None]
        par = tl.sum(tl.where(jj[None, :] > nu[:, None], v, 0), axis=1) & 1
        dirn = tl.where(par == 0, 1, -1)
        v_old = tl.sum(tl.where(isnu, v, 0), axis=1)
        m_nu = tl.sum(tl.where(isnu, Mu, 0), axis=1)
        d = tl.where(isnu, d + 1, tl.where(jj[None, :] < nu[:, None], 0, d))
        v = tl.where(isnu, v + dirn[:, None], v)
        up = (dirn > 0)
        num = tl.where(up, -(m_nu - v_old).to(tl.float64), -v_old.to(tl.float64))
        den = tl.where(up, (v_old + 1).to(tl.float64), (m_nu - v_old + 1).to(tl.float64))
        w = w * (num / den)
        col = tl.load(dcol_lane + nu, mask=ok & (nu < K), other=0)
        ptr = a_lane[:, None] + rowofs + col[:, None] * stride_an
        mm = ok[:, None] & imask[None, :]
        a_re = tl.load(ptr, mask=mm, other=0.0)
        a_im = tl.load(ptr + 1, mask=mm, other=0.0)
        f = (-2 * dirn).to(tl.float64)
        rs_re += f[:, None] * a_re
        rs_im += f[:, None] * a_im
        coef = tl.where(isnu, coef + f[:, None], coef)
        wt = tl.where(ok, w, 0.0)
        p_re, p_im = tl.reduce((rs_re, rs_im), 1, _cmul)
        acc_re += (wt * p_re)[:, None] * coef
        acc_im += (wt * p_im)[:, None] * coef

    optr = part_ptr + (g[:, None].to(tl.int64) * K + jj[None, :]) * 2
    tl.store(optr, acc_re, mask=m2)
    tl.store(optr + 1, acc_im, mask=m2)


def input_tables(sig, K):
    """The input-side tables of a step: `multiplicity_tables` applied to the K columns sigma(1..k)
    of every stream. Returns dcol [S, K] int64 (distinct input columns, padded with the first
    one), mult [S, K] int32, radix [S, K] int32, N [S] int64, scale [S] float64 (2^-K, x2 when
    halved). The padded entries have multiplicity 0 and radix 1, so they contribute nothing.
    """
    dcol, mult, radix, N, scale = multiplicity_tables(sig[:, :K].contiguous(), K)
    pad = mult == 0
    dcol = torch.where(pad, dcol[:, :1].expand_as(dcol), dcol)
    return dcol, mult, radix, N, scale


def qvec_partials_input(A2, rows, sig, dp, S, K1, K, target_total_lanes=1 << 17, seg=None, num_warps=None):
    """Kernel 1 of the "input" algorithm: per-lane partial Q [TOTAL, K, 2]. Returns
    (part, lstart, nl, scale, max_lanes, lane2s, dcol): the same tuple as `cofactor_partials_multi`
    plus the column table `dcol` [S, K] to hand to `draw_step` in place of sigma.
    """
    dev = A2.device
    dcol, mult, radix, N, scale = input_tables(sig, K)
    L, nl, lstart, lane2s, total = plan_multi(N, target_total_lanes)
    K_PAD = max(2, triton.next_power_of_2(max(K, K1)))
    cseg, cwarps = get_config(dev)[K_PAD]
    SEG = seg or cseg
    num_warps = num_warps or cwarps
    part = torch.empty(total, K, 2, dtype=torch.float64, device=dev)
    grid = (triton.cdiv(total, SEG),)
    _qvec_input_kernel[grid](
        A2, dcol, mult, radix, rows, dp, lane2s, lstart, N, _binom_table(dev), part,
        A2.stride(0), A2.stride(1), A2.stride(2), dcol.stride(0), rows.stride(0),
        K1, K, L, total,
        K_PAD=K_PAD, SEG=SEG, num_warps=num_warps,
    )
    return part, lstart, nl, scale, int(nl.max().item()), lane2s, dcol


def qvec_input_rows(A, rows, cols, target_total_lanes=1 << 17, seg=None, num_warps=None):
    """Test entry point: A [S, m, n] complex128, rows [S, K1] int64 (the chosen output rows), cols
    [S, K] int64 (input columns sigma(1..k), repeats allowed). Returns (Q [S, K] complex128, dcol
    [S, K]) with Q[s, mu] = per([A[s][rows[s]][:, cols[s]] ; e_mu]) for the distinct columns mu of
    stream s, 0 on the padding.
    """
    S, m, n = A.shape
    K1, K = rows.shape[1], cols.shape[1]
    dev = A.device
    A2 = torch.view_as_real(A.contiguous())
    dp = torch.arange(S, dtype=torch.int64, device=dev)
    part, lstart, nl, scale, _, lane2s, dcol = qvec_partials_input(A2, rows.contiguous(), cols.contiguous(), dp, S, K1, K,
                                                                    target_total_lanes, seg, num_warps)
    out = torch.zeros(S, K, 2, dtype=torch.float64, device=dev)
    out.index_add_(0, lane2s, part)
    return torch.view_as_complex(out) * scale[:, None], dcol
