"""
CUDA (NVRTC via cupy) thread-per-lane version of the 2020 cofactor kernel.

One thread = one Gray lane: the k column sums, the accumulators and the odometer state
live in that thread's registers / L1 local memory; no cross-thread reductions, no
layout constraints.  Same maths and same host planning as the Triton kernel
(cofactors_triton._cofactors_multi_kernel): mixed-radix reflected Gray code over the
multiplicities of the already-chosen rows, halving when a multiplicity is odd,
all-but-one products from the full product with exact zero handling.

Input block: Bs [S, K1, K] complex128 (rows = distinct chosen rows, columns = sigma(1..k)),
gathered once per step by the host, so the per-term row read is contiguous.
Output: per-lane partial sums part [TOTAL, K, 2] — the draw kernel is unchanged.

Tiers K <= 8/12/16/20/24 use one thread per lane; 32 and 48 use two threads per lane
(16 / 24 columns each), 64 uses four, the threads of a lane exchanging their partial
product and zero count with __shfl_xor_sync once per term — no register spills up to
n = 63.  Requires `cupy` (cupy-cuda13x for torch cu13).  Compiled once per tier.
"""
import torch

_SRC = r"""
#include <cuda_runtime.h>
#define LMIN_TERMS 256   // single-stream path: minimum Gray terms per lane (amortises the per-lane initialisation)

// KP = padded column count of the tier, TPL = threads per lane (1, 2 or 4): the TPL
// consecutive threads of a lane each own KH = KP / TPL columns (column sums and
// accumulators in registers), run the same odometer redundantly, and exchange their
// partial product / zero count with __shfl_xor_sync once per term.
// The register arrays are KHP = KH rounded up to the 4 product chains, and every access
// past KH is removed at compile time, so ANY (KP, TPL) is legal -- which is what lets the
// tier ladder run in steps of 4 instead of jumping 24 -> 32 -> 48 -> 64.  Measured on an
// H100: 1.3-1.9x over the old ladder for K = 26..62,
// identical results to 1e-14.
template <int KP, int TPL>
__global__ void cof_multi(const double2* __restrict__ Bs,     // [S, K1, K]
                          const int*  __restrict__ mult,     // [S, K1]
                          const int*  __restrict__ radix,    // [S, K1]
                          const long long* __restrict__ lane2s,   // [TOTAL lanes]
                          const long long* __restrict__ lstart,   // [S]
                          const long long* __restrict__ nterms,   // [S]
                          const double* __restrict__ binom,       // [32*32]
                          double2* __restrict__ part,             // [TOTAL, K]
                          int K1, int K, long long L, long long TOTAL)
{
    constexpr int KH = KP / TPL;                  // columns this thread owns
    constexpr int KHP = ((KH + 3) / 4) * 4;       // register array length (4 product chains)
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // global thread
    const long long g = gt / TPL;                                            // lane
    const int h = (int)(gt % TPL);                                           // half / quarter
    const int c0 = h * KH;                                                   // first column of this thread
    if (g >= TOTAL) return;
    const unsigned grp = ((1u << TPL) - 1u) << ((threadIdx.x & 31) & ~(TPL - 1));   // shuffle mask of the lane's threads
    const long long s = lane2s[g];
    const long long li = g - lstart[s];
    const long long Ns = nterms[s];
    double2* out = part + g * K;
    if (L <= 0) {
        // fixed lane grid (single-stream path): use only enough lanes for >= LMIN terms each,
        // the others write zeros and leave before the O(K1*K) initialisation
        long long le = Ns / LMIN_TERMS; if (le < 1) le = 1; if (le > TOTAL) le = TOTAL;
        if (li >= le) { for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) out[c0 + j] = make_double2(0.0, 0.0); return; }
        L = (Ns + le - 1) / le;
    }
    const long long base = li * L;
    if (base >= Ns) { for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) out[c0 + j] = make_double2(0.0, 0.0); return; }

    const double2* B = Bs + s * (long long)K1 * K + c0;
    const int* Ms = mult + s * K1;
    const int* Rs = radix + s * K1;

    // odometer state (all values < 64): bytes, so the per-thread local footprint stays L1-resident
    unsigned char d[KP], v[KP], R[KP], M[KP], pp[KP];      // pp[nu] = parity of sum_{mu > nu} v_mu
    for (int nu = 0; nu < K1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < K1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = K1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1); v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    double w = 1.0;
    for (int nu = 0; nu < K1; ++nu) { double b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    double sr[KHP], si[KHP], ar[KHP], ai[KHP];
    #pragma unroll
    for (int j = 0; j < KHP; ++j) { sr[j] = 0.0; si[j] = 0.0; ar[j] = 0.0; ai[j] = 0.0; }
    for (int nu = 0; nu < K1; ++nu) {
        const double c = (double)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + nu * K;
        #pragma unroll
        for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) { double2 b = row[j]; sr[j] += c * b.x; si[j] += c * b.y; }
    }

    const long long Lp = (L < Ns - base) ? L : (Ns - base);
    for (long long t = 0; t < Lp; ++t) {
        if (t > 0) {
            int nu = 0;
            while (d[nu] == R[nu] - 1) ++nu;
            const int dirn = pp[nu] ? -1 : 1;
            const int vold = v[nu], mnu = M[nu];
            for (int mu = 0; mu < nu; ++mu) { d[mu] = 0; pp[mu] ^= 1; }
            d[nu] += 1; v[nu] = (unsigned char)(vold + dirn);
            w *= (dirn > 0) ? -(double)(mnu - vold) / (double)(vold + 1)
                            : -(double)vold / (double)(mnu - vold + 1);
            const double f = -2.0 * dirn;
            const double2* row = B + nu * K;
            #pragma unroll
            for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) { double2 b = row[j]; sr[j] += f * b.x; si[j] += f * b.y; }
        }
        // partial product over this thread's columns, 4 interleaved chains; zero count
        int z = 0;
        double p0r = 1.0, p0i = 0.0, p1r = 1.0, p1i = 0.0, p2r = 1.0, p2i = 0.0, p3r = 1.0, p3i = 0.0;
        #pragma unroll
        for (int j = 0; j < KHP; j += 4) {
            if (j < KH && c0 + j < K) { const bool zr = (sr[j] == 0.0) && (si[j] == 0.0); z += zr ? 1 : 0;
                         if (!zr) { const double nr = p0r * sr[j] - p0i * si[j]; p0i = p0r * si[j] + p0i * sr[j]; p0r = nr; } }
            if (j + 1 < KH && c0 + j + 1 < K) { const bool zr = (sr[j+1] == 0.0) && (si[j+1] == 0.0); z += zr ? 1 : 0;
                         if (!zr) { const double nr = p1r * sr[j+1] - p1i * si[j+1]; p1i = p1r * si[j+1] + p1i * sr[j+1]; p1r = nr; } }
            if (j + 2 < KH && c0 + j + 2 < K) { const bool zr = (sr[j+2] == 0.0) && (si[j+2] == 0.0); z += zr ? 1 : 0;
                         if (!zr) { const double nr = p2r * sr[j+2] - p2i * si[j+2]; p2i = p2r * si[j+2] + p2i * sr[j+2]; p2r = nr; } }
            if (j + 3 < KH && c0 + j + 3 < K) { const bool zr = (sr[j+3] == 0.0) && (si[j+3] == 0.0); z += zr ? 1 : 0;
                         if (!zr) { const double nr = p3r * sr[j+3] - p3i * si[j+3]; p3i = p3r * si[j+3] + p3i * sr[j+3]; p3r = nr; } }
        }
        double pr, pi;
        { const double ar01 = p0r * p1r - p0i * p1i, ai01 = p0r * p1i + p0i * p1r;
          const double ar23 = p2r * p3r - p2i * p3i, ai23 = p2r * p3i + p2i * p3r;
          pr = ar01 * ar23 - ai01 * ai23; pi = ar01 * ai23 + ai01 * ar23; }
        // combine the TPL threads of the lane (butterfly)
        #pragma unroll
        for (int o = 1; o < TPL; o <<= 1) {
            const double qr = __shfl_xor_sync(grp, pr, o);
            const double qi = __shfl_xor_sync(grp, pi, o);
            const int    qz = __shfl_xor_sync(grp, z, o);
            const double nr = pr * qr - pi * qi; pi = pr * qi + pi * qr; pr = nr;
            z += qz;
        }
        const double wpr = w * pr, wpi = w * pi;
        if (z == 0) {
            #pragma unroll
            for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) {
                const double inv = 1.0 / (sr[j] * sr[j] + si[j] * si[j]);
                ar[j] += (wpr * sr[j] + wpi * si[j]) * inv;
                ai[j] += (wpi * sr[j] - wpr * si[j]) * inv;
            }
        } else if (z == 1) {
            #pragma unroll
            for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) {
                if (sr[j] == 0.0 && si[j] == 0.0) { ar[j] += wpr; ai[j] += wpi; }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < KHP; ++j) if (j < KH && c0 + j < K) out[c0 + j] = make_double2(ar[j], ai[j]);
}

// Single-stream step preparation, one block: multiplicities of the chosen rows, radices
// (odd-multiplicity row last, halved), term count, normalisation, and the gathered block
// Bs[nu, j] = A[drow[nu], sig[j]].  Everything stays on the device: no host synchronisation.
extern "C" __global__ void prep_single(const long long* __restrict__ rows, const long long* __restrict__ sig,
                                       const double2* __restrict__ A, int K1, int K, long long stride_am,
                                       long long* __restrict__ drow, int* __restrict__ mult, int* __restrict__ radix,
                                       long long* __restrict__ nout, double* __restrict__ scale,
                                       double2* __restrict__ Bs)
{
    __shared__ long long dr[64];
    if (threadIdx.x == 0) {
        long long r[64]; int mc[64]; long long dv[64];
        for (int i = 0; i < K1; ++i) r[i] = rows[i];
        for (int i = 1; i < K1; ++i) { long long x = r[i]; int j = i - 1; while (j >= 0 && r[j] > x) { r[j + 1] = r[j]; --j; } r[j + 1] = x; }
        int D = 0;
        for (int i = 0; i < K1; ++i) { if (D > 0 && dv[D - 1] == r[i]) mc[D - 1]++; else { dv[D] = r[i]; mc[D] = 1; ++D; } }
        int pos = -1;
        for (int q = 0; q < D; ++q) if (mc[q] & 1) pos = q;
        if (pos >= 0 && pos != D - 1) { long long tv = dv[pos]; dv[pos] = dv[D - 1]; dv[D - 1] = tv; int tm = mc[pos]; mc[pos] = mc[D - 1]; mc[D - 1] = tm; }
        long long N = 1;
        for (int nu = 0; nu < K1; ++nu) {
            if (nu < D) { int R = (nu == D - 1 && pos >= 0) ? (mc[nu] + 1) / 2 : mc[nu] + 1;
                          drow[nu] = dv[nu]; mult[nu] = mc[nu]; radix[nu] = R; N *= R; }
            else        { drow[nu] = 0; mult[nu] = 0; radix[nu] = 1; }
            dr[nu] = drow[nu];
        }
        *nout = N;
        *scale = (pos >= 0 ? 2.0 : 1.0) * ldexp(1.0, -K1);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < K1 * K; idx += blockDim.x) {
        const int nu = idx / K, j = idx % K;
        Bs[idx] = A[dr[nu] * stride_am + sig[j]];
    }
}
"""

# K_PAD tier -> (threads per lane, --maxrregcount).  Measured, not guessed: the kernel is
# occupancy-bound, so what decides the rate is (a) how few of the KP columns are padding and
# (b) whether the register cap sits on the right side of a cliff.  Per-COLUMN throughput is
# flat at ~200-250 G column-updates/s whatever the tier, so a ladder in steps of 4 is worth
# exactly its padding fraction (three H100 sweeps).
#
# The cliffs are steep and tier-specific -- 128 is right at tier 48 (5.01 G terms/s) and
# catastrophic at tier 64 (1.67 against 3.18) -- so an unmeasured GPU must fall back to the
# compiler's own choice (None) rather than inherit these.  Regenerate with
# `python bench/bench_cofactor_variants.py --fine-tiers --modes 0 --tpls 1,2,4 --maxreg ...`.
_CFG_BY_ARCH = {
    (9, 0): {                                  # NVIDIA H100 (PCIe, measured 2026-09-15)
        8:  (1, None), 12: (1, None), 16: (1, None), 20: (1, None), 24: (2, 128),
        28: (4, 128), 32: (4, 168), 36: (4, None), 40: (4, None), 44: (4, 128),
        48: (4, 128), 52: (4, 168), 56: (4, 168), 60: (4, 168), 64: (4, 168),
    },
}
# Fallback for an untuned GPU: the same ladder, TPL by the no-spill rule, no register cap.
_CFG_DEFAULT = {k: ((1 if k <= 24 else (2 if k <= 32 else 4)), None) for k in range(8, 65, 4)}

_TIERS_LADDER = tuple(range(8, 65, 4))
_CFG = None
_MODULES = {}
_KERNELS = {}


def _cfg():
    """(tier -> (TPL, maxrregcount)) for the current device."""
    global _CFG
    if _CFG is None:
        import torch as _t
        _CFG = _CFG_BY_ARCH.get(_t.cuda.get_device_capability(), _CFG_DEFAULT)
    return _CFG


def set_config(cfg):
    """Override the (tier -> (TPL, maxrregcount)) table for this process; None resets."""
    global _CFG, _MODULES, _KERNELS
    _CFG = None if cfg is None else {**_cfg(), **cfg}
    _MODULES, _KERNELS = {}, {}


# `_TIERS` kept for the tests and benchmarks that read it as tier -> threads per lane.
class _TierView(dict):
    def __getitem__(self, k):
        return _cfg()[k][0]


_TIERS = _TierView(dict.fromkeys(_TIERS_LADDER))   # iteration yields the ladder, [k] the TPL


def warmup():
    """Compile every module up front (~1 s per distinct register cap).

    The register cap is an NVRTC option, so it is per compilation unit: the tier ladder needs
    one module per distinct cap instead of the single module the old fixed table used.  Those
    compilations are done here, eagerly on first use, rather than lazily when a run first
    reaches a tier in a new cap group -- otherwise a timed region can swallow one (seen in
    tests/clifford/check_cuda.py: 827 ms at k-1 = 20, the first step to reach the cap-128
    group, against 10 ms for the same tier once compiled).  `bench/find_threshold.py` and the
    sampler benchmarks already run a warm-up sample, so this only makes the cost explicit.
    """
    import cupy
    cfg = _cfg()
    for cap in dict.fromkeys(cfg[k][1] for k in _TIERS_LADDER):
        if cap not in _MODULES:
            tiers = [k for k in _TIERS_LADDER if cfg[k][1] == cap]
            opts = ("-std=c++17",) if cap is None else ("-std=c++17", f"--maxrregcount={cap}")
            _MODULES[cap] = cupy.RawModule(
                code=_SRC, options=opts,
                name_expressions=[f"cof_multi<{k}, {cfg[k][0]}>" for k in tiers])


def _get_kernel(KP):
    """One module per distinct register cap (an NVRTC option is per compilation unit)."""
    cfg = _cfg()
    if KP not in _KERNELS:
        if not _MODULES:
            warmup()
        _KERNELS[KP] = _MODULES[cfg[KP][1]].get_function(f"cof_multi<{KP}, {cfg[KP][0]}>")
    return _KERNELS[KP]


def _cp(t):
    import cupy
    return cupy.asarray(t)


def cofactor_partials_cuda(A2, rows, sig, dp, S, K1, K, target_total_lanes=1 << 17, block=128):
    """Same contract as cofactors_triton.cofactor_partials_multi (returns part, lstart, nl, scale, max_nl, lane2s)."""
    import cupy

    from .cofactors_triton import _binom_table, multiplicity_tables, plan_multi
    dev = A2.device
    drow, mult, radix, N, scale = multiplicity_tables(rows, K1)
    L, nl, lstart, lane2s, total = plan_multi(N, target_total_lanes)
    # gather the block Bs[s, nu, j] = A[dp[s], drow[s, nu], sig[s, j]]   (float64 pairs)
    A = torch.view_as_complex(A2)                                                  # [B, m, n]
    Bs = A[dp[:, None, None], drow[:, :, None], sig[:, None, :K]]                  # [S, K1, K] complex128
    Bs2 = torch.view_as_real(Bs.contiguous())
    part = torch.empty(total, K, 2, dtype=torch.float64, device=dev)
    KP = next(k for k in _TIERS_LADDER if k >= K)
    kern = _get_kernel(KP)
    nthreads = total * _cfg()[KP][0]
    grid = ((nthreads + block - 1) // block,)
    with cupy.cuda.Device(dev.index or 0):
        kern(grid, (block,), (
            _cp(Bs2), _cp(mult), _cp(radix), _cp(lane2s), _cp(lstart), _cp(N), _cp(_binom_table(dev)), _cp(part),
            cupy.int32(K1), cupy.int32(K), cupy.int64(L), cupy.int64(total)))
    return part, lstart, nl, scale, int(nl.max().item()), lane2s


def get_prep_kernel():
    """Return the compiled `prep_single` kernel, compiling the module for tier 8 first if needed."""
    _get_kernel(8)
    return _MODULES[_cfg()[8][1]].get_function("prep_single")


def tier(K):
    """Return (padded column count, threads per lane) of the compiled tier serving K columns."""
    KP = next(k for k in _TIERS_LADDER if k >= K)
    return KP, _cfg()[KP][0]


def available():
    """True when this backend can run: `cupy` is importable and a CUDA device is visible to torch."""
    try:
        import cupy  # noqa
        return torch.cuda.is_available()
    except Exception:
        return False
