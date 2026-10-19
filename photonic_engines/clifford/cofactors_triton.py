"""
Triton kernels for the Clifford & Clifford boson sampler (arXiv:1706.01260).

Per photon step k (K = k columns sigma(1..k), K1 = k-1 chosen rows):

  kernel 1  _cofactors_kernel
      all K Laplace cofactors  P_c = per(B \\ column c),  B[i, j] = A[dp(s), rows[s, i], sigma[s, j]]
      in ONE Gray-code sweep of Glynn's formula (2^(K1-1) terms), cut into `lanes`
      contiguous segments per stream; lanes of all streams form one flat grid so that
      small k (few terms, many streams) and large k (one stream, many lanes) both fill
      the GPU.  Output: per-lane partial sums  part[S*lanes, K] (complex, interleaved).

  kernel 2  _draw_kernel
      per stream: P = sum_lanes(part) / 2^(K1-1);  amp[rho] = sum_c A[dp, rho, sigma_c] P_c
      for all m candidate modes; w = |amp|^2; draw rho ~ w (Philox uniform, inverse-CDF
      on a cumsum); write rows[s, k-1].

Glynn over delta in {+-1}^K1 (delta_0 = +1 fixed):
      per(B \\ c) = 2^-(K1-1) sum_delta sign(delta) prod_{j != c} s_j(delta),
      s_j(delta) = sum_i delta_i B[i, j]                       (shared by all c)

Gray-code facts (g = global term index within a stream, 0-based):
  * gray(g) = g ^ (g >> 1); bit b of gray(g) <-> delta_{b+1} = -1
  * gray(g) and gray(g-1) differ in bit ctz(g); for g = base + t with base a multiple
    of the segment length L and 0 < t < L, ctz(g) = ctz(t): the flipped row is the same
    for every lane at step t (one scalar table load), no divergence.
  * sign(delta) = (-1)^popcount(gray(g)) = (-1)^(g & 1).

All-but-one products  prod_{j != c} s_j  are obtained from the full product P = prod_j s_j
as P / s_c, with exact handling of zeros (z = number of zero s_j: z = 0 -> P/s_c,
z = 1 -> P' for the zero column and 0 elsewhere, z >= 2 -> 0).  Division is relatively
well-conditioned, so the accuracy matches the prefix/suffix formulation while needing a
single reduction per term instead of two scans and three shifted copies of s.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _allbutone(s_re, s_im, jmask):
    """[SEG, K_PAD] -> [SEG, K_PAD]: prod_{j != c} s_j with exact zero handling.
    Padded columns (jmask False) must hold 1+0i.
    """
    iszero = (s_re == 0.0) & (s_im == 0.0) & jmask[None, :]
    z = tl.sum(iszero.to(tl.int32), axis=1)                    # [SEG]
    sn_re = tl.where(iszero, 1.0, s_re)
    sn_im = tl.where(iszero, 0.0, s_im)
    p_re, p_im = tl.reduce((sn_re, sn_im), 1, _cmul)           # [SEG]
    d = tl.where(iszero, 1.0, s_re * s_re + s_im * s_im)
    q_re = (p_re[:, None] * s_re + p_im[:, None] * s_im) / d   # P / s_c
    q_im = (p_im[:, None] * s_re - p_re[:, None] * s_im) / d
    z2 = z[:, None]
    ab_re = tl.where(z2 == 0, q_re, tl.where((z2 == 1) & iszero, p_re[:, None], 0.0))
    ab_im = tl.where(z2 == 0, q_im, tl.where((z2 == 1) & iszero, p_im[:, None], 0.0))
    return ab_re, ab_im


@triton.jit(do_not_specialize=["K1", "K", "L", "LANES", "TOTAL"])
def _cofactors_kernel(
    a_ptr,                          # A  [B, m, n, 2] float64 (re, im interleaved)
    rows_ptr, sig_ptr, dp_ptr,      # rows [S, n] int64, sigma [S, n] int64, dp [S] int64
    part_ptr,                       # out [S*LANES, K, 2] float64
    ctz_ptr,                        # int32 table, ctz[t] = ctz(t), t < L
    stride_ab, stride_am, stride_an, stride_rs, stride_ss,
    K1, K, L, LANES, TOTAL,         # runtime ints, TOTAL = S * LANES
    K_PAD: tl.constexpr, SEG: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid * SEG + tl.arange(0, SEG)                          # global lane  [SEG]
    valid = g < TOTAL
    s = g // LANES                                             # stream
    li = g % LANES                                             # lane within stream
    base = li.to(tl.int64) * L                                 # first Gray index of the lane
    jj = tl.arange(0, K_PAD)
    jmask = jj < K
    m2 = valid[:, None] & jmask[None, :]

    dp = tl.load(dp_ptr + s, mask=valid, other=0)              # [SEG] int64
    sig = tl.load(sig_ptr + s[:, None] * stride_ss + jj[None, :], mask=m2, other=0)   # [SEG, K_PAD]
    a_lane = a_ptr + dp * stride_ab                            # [SEG] pointers
    colofs = sig * stride_an                                   # [SEG, K_PAD]
    rows_lane = rows_ptr + s * stride_rs

    # ---- s = sum_i delta_i B[i, :] at g = base;  padded columns hold 1 ----------
    gray0 = base ^ (base >> 1)
    s_re = tl.where(jmask[None, :], 0.0, 1.0) + tl.zeros([SEG, K_PAD], tl.float64)
    s_im = tl.zeros([SEG, K_PAD], tl.float64)
    for i in range(K1):
        sh = tl.maximum(i - 1, 0)
        bit = ((gray0 >> sh) & 1).to(tl.float64)
        d = tl.where(i == 0, 1.0, 1.0 - 2.0 * bit)             # [SEG]
        r_i = tl.load(rows_lane + i, mask=valid, other=0)      # [SEG] int64
        ptr = a_lane[:, None] + r_i[:, None] * stride_am + colofs
        row_re = tl.load(ptr, mask=m2, other=0.0)
        row_im = tl.load(ptr + 1, mask=m2, other=0.0)
        s_re += d[:, None] * row_re
        s_im += d[:, None] * row_im

    acc_re = tl.zeros([SEG, K_PAD], tl.float64)
    acc_im = tl.zeros([SEG, K_PAD], tl.float64)
    vsign = valid.to(tl.float64)

    sign = vsign * (1.0 - 2.0 * ((base & 1).to(tl.float64)))
    ab_re, ab_im = _allbutone(s_re, s_im, jmask)
    acc_re += sign[:, None] * ab_re
    acc_im += sign[:, None] * ab_im

    for t in range(1, L):
        c = tl.load(ctz_ptr + t)
        g_t = base + t
        gp = g_t - 1
        grayp = gp ^ (gp >> 1)
        f = -2.0 * (1.0 - 2.0 * (((grayp >> c) & 1).to(tl.float64)))   # -2 * delta_old  [SEG]
        r_c = tl.load(rows_lane + (c + 1), mask=valid, other=0)
        ptr = a_lane[:, None] + r_c[:, None] * stride_am + colofs
        row_re = tl.load(ptr, mask=m2, other=0.0)
        row_im = tl.load(ptr + 1, mask=m2, other=0.0)
        s_re += f[:, None] * row_re
        s_im += f[:, None] * row_im
        sign = vsign * (1.0 - 2.0 * ((g_t & 1).to(tl.float64)))
        ab_re, ab_im = _allbutone(s_re, s_im, jmask)
        acc_re += sign[:, None] * ab_re
        acc_im += sign[:, None] * ab_im

    optr = part_ptr + (g[:, None].to(tl.int64) * K + jj[None, :]) * 2
    tl.store(optr, acc_re, mask=m2)
    tl.store(optr + 1, acc_im, mask=m2)


@triton.jit
def _imul(a, b):
    return a * b


@triton.jit(do_not_specialize=["K1", "K", "L", "TOTAL"])
def _cofactors_multi_kernel(
    a_ptr,                          # A [B, m, n, 2] float64
    drow_ptr, mult_ptr, radix_ptr,  # per stream [S, K1]: distinct rows (int64), multiplicities, enumeration radices (int32)
    sig_ptr, dp_ptr,                # sigma [S, n] int64, dp [S] int64
    lane2s_ptr, lstart_ptr, n_ptr,  # lane -> stream (int64 [TOTAL]), first lane of stream, terms per stream (int64 [S])
    binom_ptr,                      # [32, 32] float64 binomial table C(s, v)
    part_ptr,                       # out [TOTAL, K, 2]
    stride_ab, stride_am, stride_an, stride_ds, stride_ss,
    K1, K, L, TOTAL,
    K_PAD: tl.constexpr, SEG: tl.constexpr,
):
    """Clifford & Clifford 2020: cofactors of B with repeated rows.  With D distinct rows
    b_nu of multiplicity s_nu (sum = K1) and v_nu = number of -1 signs among the copies,
        per(B \\ c) = 2^-K1 * sum_v  [prod_nu (-1)^v_nu C(s_nu, v_nu)] * prod_{j != c} (sum_nu (s_nu - 2 v_nu) b_nu,j)
    over v in prod_nu {0..s_nu} — prod(s_nu + 1) terms instead of 2^K1 — walked in a reflected
    mixed-radix Gray code (one v_nu changes by +-1 per term => rank-1 update of the column sums):
    with plain digits d of the term index, v_nu = d_nu if sum_{mu>nu} v_mu is even else R_nu-1-d_nu;
    stepping = odometer on d (lowest digit nu that can increase), v_nu moves by +1 if
    sum_{mu>nu} v_mu is even, -1 otherwise, lower v's unchanged.
    The host halves the enumeration when some s_nu is odd (that row is placed last with radix
    (s+1)/2, complement symmetry) and passes the matching scale to the draw kernel.
    Each lane walks terms [li*L, (li+1)*L) of its stream (masked beyond N_s).
    """
    pid = tl.program_id(0)
    g = pid * SEG + tl.arange(0, SEG)
    valid = g < TOTAL
    s = tl.load(lane2s_ptr + g, mask=valid, other=0)
    li = g - tl.load(lstart_ptr + s, mask=valid, other=0)
    Ns = tl.load(n_ptr + s, mask=valid, other=1)
    base = li * L                                              # int64 [SEG]
    jj = tl.arange(0, K_PAD)
    jmask = jj < K
    kmask = jj < K1
    m2 = valid[:, None] & jmask[None, :]
    mk = valid[:, None] & kmask[None, :]

    dp = tl.load(dp_ptr + s, mask=valid, other=0)
    sig = tl.load(sig_ptr + s[:, None] * stride_ss + jj[None, :], mask=m2, other=0)
    a_lane = a_ptr + dp * stride_ab
    colofs = sig * stride_an
    drow_lane = drow_ptr + s * stride_ds

    R = tl.load(radix_ptr + s[:, None] * stride_ds + jj[None, :], mask=mk, other=1)      # int32 [SEG, K_PAD]
    Mu = tl.load(mult_ptr + s[:, None] * stride_ds + jj[None, :], mask=mk, other=0)      # int32
    # mixed-radix digits of base: d_nu = (base // P_nu) % R_nu, P_nu = prod_{mu < nu} R_mu (exact in f64)
    Pinc = tl.associative_scan(R.to(tl.float64), 1, _imul)
    Pexc = (Pinc / R.to(tl.float64)).to(tl.int64)
    d = ((base[:, None] // Pexc) % R.to(tl.int64)).to(tl.int32)
    # reflected Gray value, top-down: v_nu = d_nu if sum_{mu > nu} v_mu even else R_nu - 1 - d_nu
    v = tl.zeros([SEG, K_PAD], tl.int32)
    par0 = tl.zeros([SEG], tl.int32)
    for i in range(K1):
        nu0 = K1 - 1 - i
        isnu0 = jj[None, :] == nu0
        d_nu0 = tl.sum(tl.where(isnu0, d, 0), axis=1)
        r_nu0 = tl.sum(tl.where(isnu0, R, 0), axis=1)
        v_nu0 = tl.where((par0 & 1) == 0, d_nu0, r_nu0 - 1 - d_nu0)
        v = tl.where(isnu0, v_nu0[:, None], v)
        par0 += v_nu0
    # weight w = prod_nu (-1)^v_nu C(s_nu, v_nu)
    bin = tl.load(binom_ptr + Mu * 32 + v, mask=mk, other=1.0)
    w = tl.reduce(tl.where((v & 1) == 1, -bin, bin), 1, _imul)                          # [SEG] float64
    # column sums s_j = sum_nu (s_nu - 2 v_nu) b_{nu, j};  padded columns hold 1
    coef = (Mu - 2 * v).to(tl.float64)
    s_re = tl.where(jmask[None, :], 0.0, 1.0) + tl.zeros([SEG, K_PAD], tl.float64)
    s_im = tl.zeros([SEG, K_PAD], tl.float64)
    for nu1 in range(K1):
        c_nu1 = tl.sum(tl.where(jj[None, :] == nu1, coef, 0.0), axis=1)
        r_nu1 = tl.load(drow_lane + nu1, mask=valid, other=0)
        ptr1 = a_lane[:, None] + r_nu1[:, None] * stride_am + colofs
        row_re1 = tl.load(ptr1, mask=m2, other=0.0)
        row_im1 = tl.load(ptr1 + 1, mask=m2, other=0.0)
        s_re += c_nu1[:, None] * row_re1
        s_im += c_nu1[:, None] * row_im1

    acc_re = tl.zeros([SEG, K_PAD], tl.float64)
    acc_im = tl.zeros([SEG, K_PAD], tl.float64)
    ok = valid & (base < Ns)
    wt = tl.where(ok, w, 0.0)
    ab_re, ab_im = _allbutone(s_re, s_im, jmask)
    acc_re += wt[:, None] * ab_re
    acc_im += wt[:, None] * ab_im

    # terms left for each lane; the program loops to the longest one (lanes are grouped by
    # stream size on the host, so this wastes little)
    rem = tl.where(valid, tl.minimum(Ns - base, L), 0)
    Lp = tl.max(rem, axis=0)
    for t in range(1, Lp):
        ok = valid & (base + t < Ns)
        # odometer on the plain digits: lowest digit that can still increase
        can = (d < R - 1) & kmask[None, :]
        nu = tl.min(tl.where(can, jj[None, :], K_PAD), axis=1)                          # [SEG] int32
        isnu = jj[None, :] == nu[:, None]
        par = tl.sum(tl.where(jj[None, :] > nu[:, None], v, 0), axis=1) & 1
        dirn = tl.where(par == 0, 1, -1)                                                 # +-1 change of v_nu
        v_old = tl.sum(tl.where(isnu, v, 0), axis=1)
        m_nu = tl.sum(tl.where(isnu, Mu, 0), axis=1)
        d = tl.where(isnu, d + 1, tl.where(jj[None, :] < nu[:, None], 0, d))
        v = tl.where(isnu, v + dirn[:, None], v)
        up = (dirn > 0)
        num = tl.where(up, -(m_nu - v_old).to(tl.float64), -v_old.to(tl.float64))
        den = tl.where(up, (v_old + 1).to(tl.float64), (m_nu - v_old + 1).to(tl.float64))
        w = w * (num / den)                                                              # C(s, v+-1) / C(s, v) with sign
        r_nu = tl.load(drow_lane + nu, mask=ok & (nu < K1), other=0)
        ptr = a_lane[:, None] + r_nu[:, None] * stride_am + colofs
        mm = ok[:, None] & jmask[None, :]
        row_re = tl.load(ptr, mask=mm, other=0.0)
        row_im = tl.load(ptr + 1, mask=mm, other=0.0)
        f = (-2 * dirn).to(tl.float64)
        s_re += f[:, None] * row_re
        s_im += f[:, None] * row_im
        wt = tl.where(ok, w, 0.0)
        ab_re, ab_im = _allbutone(s_re, s_im, jmask)
        acc_re += wt[:, None] * ab_re
        acc_im += wt[:, None] * ab_im

    optr = part_ptr + (g[:, None].to(tl.int64) * K + jj[None, :]) * 2
    tl.store(optr, acc_re, mask=m2)
    tl.store(optr + 1, acc_im, mask=m2)


@triton.jit(do_not_specialize=["M", "K", "kidx", "seed"])
def _draw_kernel(
    a_ptr, part_ptr, rows_ptr, sig_ptr, dp_ptr,
    lstart_ptr, nl_ptr, scale_ptr,  # per stream: first lane, lane count, normalisation
    stride_ab, stride_am, stride_an, stride_rs, stride_ss,
    M, K, kidx, seed,
    M_PAD: tl.constexpr, K_PAD: tl.constexpr, LCHUNK: tl.constexpr,
):
    s = tl.program_id(0)
    jj = tl.arange(0, K_PAD)
    jmask = jj < K
    # ---- reduce lane partials -> cofactors P[K] ----------------------------------
    l_start = tl.load(lstart_ptr + s)
    n_lanes = tl.load(nl_ptr + s)
    scale = tl.load(scale_ptr + s)
    P_re = tl.zeros([K_PAD], tl.float64)
    P_im = tl.zeros([K_PAD], tl.float64)
    ll = tl.arange(0, LCHUNK)
    for l0 in range(0, n_lanes, LCHUNK):
        lm = (l0 + ll) < n_lanes
        idx = ((l_start + l0 + ll)[:, None].to(tl.int64) * K + jj[None, :]) * 2
        mm = lm[:, None] & jmask[None, :]
        P_re += tl.sum(tl.load(part_ptr + idx, mask=mm, other=0.0), axis=0)
        P_im += tl.sum(tl.load(part_ptr + idx + 1, mask=mm, other=0.0), axis=0)
    P_re = P_re * scale
    P_im = P_im * scale
    # ---- amplitudes of the M candidate modes -------------------------------------
    rr = tl.arange(0, M_PAD)
    rmask = rr < M
    dp = tl.load(dp_ptr + s)
    sig = tl.load(sig_ptr + s * stride_ss + jj, mask=jmask, other=0)
    ptr = a_ptr + dp * stride_ab + rr[:, None] * stride_am + sig[None, :] * stride_an
    mm = rmask[:, None] & jmask[None, :]
    A_re = tl.load(ptr, mask=mm, other=0.0)
    A_im = tl.load(ptr + 1, mask=mm, other=0.0)
    amp_re = tl.sum(A_re * P_re[None, :] - A_im * P_im[None, :], axis=1)
    amp_im = tl.sum(A_re * P_im[None, :] + A_im * P_re[None, :], axis=1)
    w = tl.where(rmask, amp_re * amp_re + amp_im * amp_im, 0.0)
    tot = tl.sum(w, axis=0)
    # ---- inverse-CDF draw, 48-bit uniform from two Philox floats -------------------
    two = tl.arange(0, 2)
    r = tl.rand(seed, s * 2 + two).to(tl.float64)
    u = tl.sum(tl.where(two == 0, r, r * 5.9604644775390625e-08), axis=0) / (1.0 + 5.9604644775390625e-08)
    cw = tl.cumsum(w, axis=0)
    idx = tl.sum(((cw < u * tot) & rmask).to(tl.int32), axis=0)
    idx = tl.minimum(idx, M - 1)
    tl.store(rows_ptr + s * stride_rs + kidx, idx.to(tl.int64))


# ------------------------------------------------------------------ host side

_CTZ_CACHE = {}


def _ctz_table(L, dev):
    key = (L, str(dev))
    tab = _CTZ_CACHE.get(key)
    if tab is None:
        t = torch.arange(L, dtype=torch.int64)
        low = t & (-t)
        tab = torch.zeros(L, dtype=torch.int32)
        if L > 1:
            tab[1:] = torch.log2(low[1:].double()).round().to(torch.int32)
        tab = tab.to(dev)
        _CTZ_CACHE[key] = tab
    return tab


def _pow2_floor(x):
    return 1 << (max(int(x), 1).bit_length() - 1)


# (SEG, num_warps) of the cofactor kernel per K_PAD tier, keyed by CUDA compute capability.
# Values measured on that GPU (a sweep of SEG and num_warps per tier).
_CFG_BY_ARCH = {
    (9, 0): {2: (128, 2), 4: (16, 2), 8: (16, 2), 16: (16, 4), 32: (16, 4), 64: (8, 4)},   # H100 PCIe, triton 3.8 (64: untuned)
}
_CFG_DEFAULT = {2: (128, 2), 4: (16, 2), 8: (16, 2), 16: (16, 4), 32: (16, 4), 64: (8, 4)}
_CFG_OVERRIDE = None


def set_config(cfg):
    """Override the (SEG, num_warps) table for this process: cfg = {K_PAD: (SEG, num_warps)}
    (partial tables are merged over the architecture default). set_config(None) resets.
    """
    global _CFG_OVERRIDE
    _CFG_OVERRIDE = None if cfg is None else {int(k): tuple(v) for k, v in cfg.items()}


def get_config(dev):
    """Effective table for device `dev`."""
    base = _CFG_DEFAULT
    if isinstance(dev, torch.device) and dev.type == "cuda":
        base = _CFG_BY_ARCH.get(torch.cuda.get_device_capability(dev), _CFG_DEFAULT)
    if _CFG_OVERRIDE:
        base = {**base, **_CFG_OVERRIDE}
    return base



def _pow2_ceil(x):
    x = max(int(x), 1)
    return 1 << (x - 1).bit_length()


def _plan(S, K1, max_lanes, target_total_lanes):
    nterms = 1 << (K1 - 1)
    lanes = _pow2_floor(max(1, min(nterms, max_lanes, target_total_lanes // max(S, 1))))
    return nterms, lanes, nterms // lanes


def cofactor_partials(A2, rows, sig, dp, S, K1, K, lanes, L, seg=None, num_warps=None):
    """Launch kernel 1 (2018 form, distinct rows assumed).  Returns part [S*lanes, K, 2]."""
    dev = A2.device
    K_PAD = max(2, triton.next_power_of_2(K))
    cseg, cwarps = get_config(dev)[K_PAD]
    SEG = seg or cseg
    num_warps = num_warps or cwarps
    total = S * lanes
    part = torch.empty(total, K, 2, dtype=torch.float64, device=dev)
    ctz = _ctz_table(L, dev)
    grid = (triton.cdiv(total, SEG),)
    _cofactors_kernel[grid](
        A2, rows, sig, dp, part, ctz,
        A2.stride(0), A2.stride(1), A2.stride(2), rows.stride(0), sig.stride(0),
        K1, K, L, lanes, total,
        K_PAD=K_PAD, SEG=SEG, num_warps=num_warps,
    )
    return part


# ---------------------------------------------------------------- 2020 form (repeated rows)

_BINOM_CACHE = {}


def _binom_table(dev):
    key = str(dev)
    t = _BINOM_CACHE.get(key)
    if t is None:
        import math
        t = torch.zeros(32, 32, dtype=torch.float64)
        for a in range(32):
            for b in range(a + 1):
                t[a, b] = math.comb(a, b)
        t = t.to(dev)
        _BINOM_CACHE[key] = t
    return t


def multiplicity_tables(rows, K1):
    """Rows [S, >=K1] int64 -> distinct rows / multiplicities / enumeration radices, per stream.
    Returns drow [S,K1] int64, mult [S,K1] int32, radix [S,K1] int32, N [S] int64 (terms), scale [S] f64.
    """
    S = rows.shape[0]
    dev = rows.device
    rs, _ = torch.sort(rows[:, :K1], dim=1)
    ar = torch.arange(K1, device=dev)
    newgrp = torch.ones(S, K1, dtype=torch.bool, device=dev)
    if K1 > 1:
        newgrp[:, 1:] = rs[:, 1:] != rs[:, :-1]
    gid = newgrp.long().cumsum(1) - 1                                           # group id per sorted position
    D = gid[:, -1] + 1                                                          # distinct rows per stream
    mult = torch.zeros(S, K1, dtype=torch.int64, device=dev).scatter_add_(1, gid, torch.ones_like(gid))
    drow = torch.zeros(S, K1, dtype=torch.int64, device=dev).scatter_(1, gid, rs)
    active = ar[None, :] < D[:, None]
    odd = (mult % 2 == 1) & active
    has_odd = odd.any(1)
    pos = torch.where(odd, ar[None, :], -1).max(1).values                       # an odd-multiplicity group
    top = D - 1
    # swap group `pos` with the top group (only where has_odd)
    perm = ar[None, :].expand(S, K1).clone()
    sidx = torch.arange(S, device=dev)
    sel = has_odd
    perm[sidx[sel], top[sel]] = pos[sel]
    perm[sidx[sel], pos[sel]] = top[sel]
    mult = torch.gather(mult, 1, perm)
    drow = torch.gather(drow, 1, perm)
    radix = torch.where(active, mult + 1, torch.ones_like(mult))
    top_r = torch.where(has_odd, (mult.gather(1, top[:, None])[:, 0] + 1) // 2, radix.gather(1, top[:, None])[:, 0])
    radix.scatter_(1, top[:, None], top_r[:, None])
    N = radix.prod(1)
    ones = torch.ones(S, dtype=torch.float64, device=dev)
    scale = torch.where(has_odd, torch.full_like(N, 2, dtype=torch.float64), ones) * (2.0 ** (-K1))
    return drow, mult.to(torch.int32), radix.to(torch.int32), N, scale


def plan_multi(N, target_total_lanes=1 << 17):
    """Common segment length L (power of 2) and per-stream lane ranges for term counts N [S].
    Lanes are handed out in decreasing order of N so that the lanes sharing a program have
    similar amounts of work (the kernel loops to the longest lane of the program).
    """
    sumN = int(N.sum().item())
    L = _pow2_ceil(max(1, sumN // target_total_lanes))
    nl = (N + L - 1) // L                                                       # lanes per stream
    order = torch.argsort(N, descending=True)
    nl_o = nl[order]
    lstart_o = torch.cumsum(nl_o, 0) - nl_o
    lstart = torch.empty_like(lstart_o)
    lstart[order] = lstart_o
    total = int(nl_o.sum().item())
    lane2s = torch.repeat_interleave(order, nl_o, output_size=total)
    return L, nl, lstart, lane2s, total


def cofactor_partials_multi(A2, rows, sig, dp, S, K1, K, target_total_lanes=1 << 17, seg=None, num_warps=None):
    """2020 form: kernel 1 on B with repeated rows.  Returns (part [TOTAL, K, 2], lstart, nl, scale)."""
    dev = A2.device
    drow, mult, radix, N, scale = multiplicity_tables(rows, K1)
    L, nl, lstart, lane2s, total = plan_multi(N, target_total_lanes)
    K_PAD = max(2, triton.next_power_of_2(K))
    cseg, cwarps = get_config(dev)[K_PAD]
    SEG = seg or cseg
    num_warps = num_warps or cwarps
    part = torch.empty(total, K, 2, dtype=torch.float64, device=dev)
    grid = (triton.cdiv(total, SEG),)
    _cofactors_multi_kernel[grid](
        A2, drow, mult, radix, sig, dp, lane2s, lstart, N, _binom_table(dev), part,
        A2.stride(0), A2.stride(1), A2.stride(2), drow.stride(0), sig.stride(0),
        K1, K, L, total,
        K_PAD=K_PAD, SEG=SEG, num_warps=num_warps,
    )
    return part, lstart, nl, scale, int(nl.max().item()), lane2s


def laplace_cofactors(B, max_lanes=1 << 16, seg=None, target_total_lanes=1 << 17, num_warps=None):
    """
    B : [S, K1, K] complex128 (K = K1 + 1).  Returns P : [S, K] complex128 with
        P[s, c] = per(B[s] without column c).   (2018 kernel; test / benchmark entry point)
    """
    assert B.dtype == torch.complex128 and B.dim() == 3
    S, K1, K = B.shape
    assert K == K1 + 1
    dev = B.device
    if K1 == 0:
        return torch.ones(S, 1, dtype=B.dtype, device=dev)
    A2 = torch.view_as_real(B.contiguous())                                  # [S, K1, K, 2]
    rows = torch.arange(K1, dtype=torch.int64, device=dev)[None, :].expand(S, K1).contiguous()
    sig = torch.arange(K, dtype=torch.int64, device=dev)[None, :].expand(S, K).contiguous()
    dp = torch.arange(S, dtype=torch.int64, device=dev)
    nterms, lanes, L = _plan(S, K1, max_lanes, target_total_lanes)
    part = cofactor_partials(A2, rows, sig, dp, S, K1, K, lanes, L, seg, num_warps)
    P = torch.view_as_complex(part.view(S, lanes, K, 2).sum(1)) / float(nterms)
    return P


def laplace_cofactors_rows(A, rows, target_total_lanes=1 << 17, seg=None, num_warps=None, backend="triton"):
    """
    2020 kernel entry point: A [S, m, K] complex128, rows [S, K1] int64 (may repeat).
    Returns P [S, K] with P[s, c] = per(A[s][rows[s]] without column c).
    backend: "triton" (this module) or "cuda" (cuda_cofactors, thread-per-lane, needs cupy).
    """
    S, m, K = A.shape
    K1 = rows.shape[1]
    assert K == K1 + 1
    dev = A.device
    A2 = torch.view_as_real(A.contiguous())
    sig = torch.arange(K, dtype=torch.int64, device=dev)[None, :].expand(S, K).contiguous()
    dp = torch.arange(S, dtype=torch.int64, device=dev)
    if backend == "cuda":
        from .cuda_cofactors import cofactor_partials_cuda
        part, lstart, nl, scale, _, lane2s = cofactor_partials_cuda(A2, rows.contiguous(), sig, dp, S, K1, K, target_total_lanes)
    else:
        part, lstart, nl, scale, _, lane2s = cofactor_partials_multi(A2, rows.contiguous(), sig, dp, S, K1, K,
                                                                     target_total_lanes, seg, num_warps)
    out = torch.zeros(S, K, 2, dtype=torch.float64, device=dev)
    out.index_add_(0, lane2s, part)
    return torch.view_as_complex(out) * scale[:, None]


def draw_step(A2, part, rows, sig, dp, S, M, K, lstart, nl, scale, kidx, seed, max_lanes=1, num_warps=4):
    """Launch kernel 2: writes rows[:, kidx].  lstart/nl/scale: per-stream lane range and normalisation;
    max_lanes: an upper bound on nl (sizes the reduction tile).
    """
    M_PAD = max(2, triton.next_power_of_2(M))
    K_PAD = max(2, triton.next_power_of_2(K))
    lchunk = min(64, max(2, triton.next_power_of_2(int(max_lanes))))
    _draw_kernel[(S,)](
        A2, part, rows, sig, dp, lstart, nl, scale,
        A2.stride(0), A2.stride(1), A2.stride(2), rows.stride(0), sig.stride(0),
        M, K, kidx, seed,
        M_PAD=M_PAD, K_PAD=K_PAD, LCHUNK=lchunk, num_warps=num_warps,
    )
