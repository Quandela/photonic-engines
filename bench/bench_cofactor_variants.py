"""
Cofactor-kernel variant sweep: where the 2020 CUDA kernel loses time, measured.

Micro-benchmark of the inner Gray loop alone (no sampler, no draw kernel), on a synthetic
step: K1 distinct rows of multiplicity 1 (the collision-free case, which dominates the mean
cost of a sample) and a capped term count, so every point costs the same wall time whatever
K.  The figure of merit is Gray-terms/s -- the hardware constant the whole frontier rests on.

Variants (MODE), all numerically identical to the production kernel:

  0  baseline          the production body, re-compiled here (sanity: must match `prod`)
  1  padded            columns K..KP-1 carry s_j = 1 instead of a per-column `c0+j < K`
                       predicate -> the three inner loops are predicate-free; zero handling
                       by select instead of branch
  2  padded + fastrcp  1/|s_j|^2 from an FP32 reciprocal + 2 Newton steps (full FP64
                       accuracy) instead of the compiler's FP64 divide
  3  padded + pfx/sfx  no reciprocal at all: all-but-one products from blocked prefix/suffix
                       scans (4 independent chains, so the latency is the same as the 4
                       interleaved product chains it replaces).  Costs 2*KH extra registers,
                       so only meaningful for KH <= 16.
  5  generalised 0     the production body EXACTLY (runtime predicate kept, branch zero
                       handling kept, no extra state), but with the register arrays padded to
                       a multiple of 4 and a compile-time `j < KH` guard, so any (tier, TPL)
                       works.  Round 2 showed mode 0 sustains 225-240 G column-updates/s while
                       mode 4 only reaches 160-200: the tier ladder should be unlocked WITHOUT
                       touching the body.  This is the production candidate.
  4  padded, branch    the production body with the runtime `c0+j < K` predicate replaced by
                       padding ONLY -- zero handling stays a branch, which round 1 showed is
                       cheaper than the select of mode 1.  Also pads KH up to a multiple of 4
                       internally, so ANY (tier, TPL) works: this is what makes the tiers
                       28/36/44/52/56/60 measurable at all.

Swept alongside: threads per lane TPL (the production table is 1 up to K = 24, 2 at 32, 4 at
48/64 -- chosen to avoid spills, never measured against occupancy), block size, and
--maxrregcount (occupancy hypothesis).

Usage
    python bench/bench_cofactor_variants.py                     # default sweep, ~4 min
    python bench/bench_cofactor_variants.py --ks 32,40,48,56,64 --csv out.csv
    python bench/bench_cofactor_variants.py --quick

Writes a CSV with one row per (K, tier, TPL, MODE, block, maxreg): Gray-terms/s and the
relative error against the production kernel.
"""
import argparse
import itertools
import math
import time

import torch

# ---------------------------------------------------------------------------------------
# tiers.  Same padding tiers as production, plus the intermediate ones the frontier needs:
# at n = 50 the last step runs K = 50 in the 64 tier -> 28 % of the work is padding, and the
# last step is ~45 % of the cost of the whole sample.
TIERS_PROD = (8, 12, 16, 20, 24, 32, 48, 64)
TIERS_FINE = (8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)

_SRC = r"""
#include <cuda_runtime.h>

__device__ __forceinline__ double frcp(double x) {
    // 1/x to full double accuracy from an FP32 seed: 2 Newton steps (2^-24 -> 2^-48 -> 2^-96).
    float xf = (float)x;
    if (!(xf > 1.17549435e-38f && xf < 3.40282347e+38f)) return 1.0 / x;   // out of FP32 range
    double r = (double)__frcp_rn(xf);
    r = r * (2.0 - x * r);
    r = r * (2.0 - x * r);
    return r;
}

// KP  padded column count of the tier
// TPL threads per lane (1, 2, 4); each of them owns KH = KP / TPL columns
// MODE 0 baseline / 1 padded / 2 padded + fast reciprocal / 3 padded + prefix-suffix
// Row stride of Bs is KP in every mode (the production kernel uses K; the difference is one
// index computation per row read and is reported by the `prod` reference row).
template <int KP, int TPL, int MODE>
__global__ void cof_v(const double2* __restrict__ Bs,          // [S, K1, KP]
                      const int* __restrict__ mult,            // [S, K1]
                      const int* __restrict__ radix,           // [S, K1]
                      const long long* __restrict__ lane2s,    // [TOTAL]
                      const long long* __restrict__ lstart,    // [S]
                      const long long* __restrict__ nterms,    // [S]
                      const double* __restrict__ binom,        // [32*32]
                      double2* __restrict__ part,              // [TOTAL, KP]
                      int K1, int K, long long L, long long TOTAL)
{
    constexpr int KH = KP / TPL;                 // columns this thread owns
    constexpr int KHP = ((KH + 3) / 4) * 4;      // rounded up to the 4 product chains
    constexpr int KA = (MODE == 4 || MODE == 5) ? KHP : KH;   // register array length
    constexpr bool PRED = (MODE == 0 || MODE == 5);           // keep the runtime column predicate
    constexpr int Q4 = KH / 4;
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long g = gt / TPL;
    const int h = (int)(gt % TPL);
    const int c0 = h * KH;
    if (g >= TOTAL) return;
    const unsigned grp = ((1u << TPL) - 1u) << ((threadIdx.x & 31) & ~(TPL - 1));
    const long long s = lane2s[g];
    const long long li = g - lstart[s];
    const long long Ns = nterms[s];
    double2* out = part + g * KP;
    const long long base = li * L;
    if (base >= Ns) {
        #pragma unroll
        for (int j = 0; j < KA; ++j) if (j < KH && c0 + j < K) out[c0 + j] = make_double2(0.0, 0.0);
        return;
    }

    const double2* B = Bs + s * (long long)K1 * KP + c0;
    const int* Ms = mult + s * K1;
    const int* Rs = radix + s * K1;

    unsigned char d[KP], v[KP], R[KP], M[KP], pp[KP];
    for (int nu = 0; nu < K1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < K1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = K1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1);
          v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    double w = 1.0;
    for (int nu = 0; nu < K1; ++nu) { double b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    double sr[KA], si[KA], ar[KA], ai[KA];
    #pragma unroll
    for (int j = 0; j < KA; ++j) { sr[j] = 0.0; si[j] = 0.0; ar[j] = 0.0; ai[j] = 0.0; }
    for (int nu = 0; nu < K1; ++nu) {
        const double c = (double)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + nu * KP;
        #pragma unroll
        for (int j = 0; j < KA; ++j) if (j < KH) {          // `j < KH` is compile-time
            if (PRED) { if (c0 + j < K) { double2 b = row[j]; sr[j] += c * b.x; si[j] += c * b.y; } }
            else      { double2 b = row[j]; sr[j] += c * b.x; si[j] += c * b.y; }
        }
    }
    if (!PRED) {   // padded columns carry s_j = 1: neutral in the product, never zero
        #pragma unroll
        for (int j = 0; j < KA; ++j) if (j >= KH || c0 + j >= K) { sr[j] = 1.0; si[j] = 0.0; }
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
            const double2* row = B + nu * KP;
            #pragma unroll
            for (int j = 0; j < KA; ++j) if (j < KH) {      // `j < KH` is compile-time
                if (PRED) { if (c0 + j < K) { double2 b = row[j]; sr[j] += f * b.x; si[j] += f * b.y; } }
                else      { double2 b = row[j]; sr[j] += f * b.x; si[j] += f * b.y; }
            }
        }

        int z = 0, jz = -1;
        double pr, pi;
        double pfr[(MODE == 3) ? KH : 1], pfi[(MODE == 3) ? KH : 1];
        double ckr[4], cki[4];

        if (MODE == 3) {
            // four independent prefix chains, one per chunk of KH/4 columns
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                double rr = 1.0, ri = 0.0;
                #pragma unroll
                for (int jj = 0; jj < Q4; ++jj) {
                    const int j = q * Q4 + jj;
                    const bool zz = (sr[j] == 0.0) && (si[j] == 0.0);
                    if (zz) { ++z; jz = j; }
                    const double yr = zz ? 1.0 : sr[j], yi = zz ? 0.0 : si[j];
                    pfr[j] = rr; pfi[j] = ri;
                    const double nr = rr * yr - ri * yi; ri = rr * yi + ri * yr; rr = nr;
                }
                ckr[q] = rr; cki[q] = ri;
            }
            const double a01r = ckr[0] * ckr[1] - cki[0] * cki[1], a01i = ckr[0] * cki[1] + cki[0] * ckr[1];
            const double a23r = ckr[2] * ckr[3] - cki[2] * cki[3], a23i = ckr[2] * cki[3] + cki[2] * ckr[3];
            pr = a01r * a23r - a01i * a23i; pi = a01r * a23i + a01i * a23r;
        } else if (MODE == 4) {
            double p0r = 1.0, p0i = 0.0, p1r = 1.0, p1i = 0.0, p2r = 1.0, p2i = 0.0, p3r = 1.0, p3i = 0.0;
            #pragma unroll
            for (int j = 0; j < KA; j += 4) {
                { const bool zr = (sr[j] == 0.0) && (si[j] == 0.0); if (zr) { ++z; jz = j; }
                  else { const double nr = p0r * sr[j] - p0i * si[j]; p0i = p0r * si[j] + p0i * sr[j]; p0r = nr; } }
                { const bool zr = (sr[j+1] == 0.0) && (si[j+1] == 0.0); if (zr) { ++z; jz = j + 1; }
                  else { const double nr = p1r * sr[j+1] - p1i * si[j+1]; p1i = p1r * si[j+1] + p1i * sr[j+1]; p1r = nr; } }
                { const bool zr = (sr[j+2] == 0.0) && (si[j+2] == 0.0); if (zr) { ++z; jz = j + 2; }
                  else { const double nr = p2r * sr[j+2] - p2i * si[j+2]; p2i = p2r * si[j+2] + p2i * sr[j+2]; p2r = nr; } }
                { const bool zr = (sr[j+3] == 0.0) && (si[j+3] == 0.0); if (zr) { ++z; jz = j + 3; }
                  else { const double nr = p3r * sr[j+3] - p3i * si[j+3]; p3i = p3r * si[j+3] + p3i * sr[j+3]; p3r = nr; } }
            }
            const double ar01 = p0r * p1r - p0i * p1i, ai01 = p0r * p1i + p0i * p1r;
            const double ar23 = p2r * p3r - p2i * p3i, ai23 = p2r * p3i + p2i * p3r;
            pr = ar01 * ar23 - ai01 * ai23; pi = ar01 * ai23 + ai01 * ar23;
        } else if (PRED) {
            double p0r = 1.0, p0i = 0.0, p1r = 1.0, p1i = 0.0, p2r = 1.0, p2i = 0.0, p3r = 1.0, p3i = 0.0;
            #pragma unroll
            for (int j = 0; j < KA; j += 4) {
                if (j < KH && c0 + j < K)     { const bool zr = (sr[j] == 0.0) && (si[j] == 0.0); z += zr ? 1 : 0;
                    if (!zr) { const double nr = p0r * sr[j] - p0i * si[j]; p0i = p0r * si[j] + p0i * sr[j]; p0r = nr; } }
                if (j + 1 < KH && c0 + j + 1 < K) { const bool zr = (sr[j+1] == 0.0) && (si[j+1] == 0.0); z += zr ? 1 : 0;
                    if (!zr) { const double nr = p1r * sr[j+1] - p1i * si[j+1]; p1i = p1r * si[j+1] + p1i * sr[j+1]; p1r = nr; } }
                if (j + 2 < KH && c0 + j + 2 < K) { const bool zr = (sr[j+2] == 0.0) && (si[j+2] == 0.0); z += zr ? 1 : 0;
                    if (!zr) { const double nr = p2r * sr[j+2] - p2i * si[j+2]; p2i = p2r * si[j+2] + p2i * sr[j+2]; p2r = nr; } }
                if (j + 3 < KH && c0 + j + 3 < K) { const bool zr = (sr[j+3] == 0.0) && (si[j+3] == 0.0); z += zr ? 1 : 0;
                    if (!zr) { const double nr = p3r * sr[j+3] - p3i * si[j+3]; p3i = p3r * si[j+3] + p3i * sr[j+3]; p3r = nr; } }
            }
            const double ar01 = p0r * p1r - p0i * p1i, ai01 = p0r * p1i + p0i * p1r;
            const double ar23 = p2r * p3r - p2i * p3i, ai23 = p2r * p3i + p2i * p3r;
            pr = ar01 * ar23 - ai01 * ai23; pi = ar01 * ai23 + ai01 * ar23;
        } else {
            double p0r = 1.0, p0i = 0.0, p1r = 1.0, p1i = 0.0, p2r = 1.0, p2i = 0.0, p3r = 1.0, p3i = 0.0;
            #pragma unroll
            for (int j = 0; j < KH; j += 4) {
                { const bool zz = (sr[j] == 0.0) && (si[j] == 0.0); if (zz) { ++z; jz = j; }
                  const double yr = zz ? 1.0 : sr[j], yi = zz ? 0.0 : si[j];
                  const double nr = p0r * yr - p0i * yi; p0i = p0r * yi + p0i * yr; p0r = nr; }
                { const bool zz = (sr[j+1] == 0.0) && (si[j+1] == 0.0); if (zz) { ++z; jz = j + 1; }
                  const double yr = zz ? 1.0 : sr[j+1], yi = zz ? 0.0 : si[j+1];
                  const double nr = p1r * yr - p1i * yi; p1i = p1r * yi + p1i * yr; p1r = nr; }
                { const bool zz = (sr[j+2] == 0.0) && (si[j+2] == 0.0); if (zz) { ++z; jz = j + 2; }
                  const double yr = zz ? 1.0 : sr[j+2], yi = zz ? 0.0 : si[j+2];
                  const double nr = p2r * yr - p2i * yi; p2i = p2r * yi + p2i * yr; p2r = nr; }
                { const bool zz = (sr[j+3] == 0.0) && (si[j+3] == 0.0); if (zz) { ++z; jz = j + 3; }
                  const double yr = zz ? 1.0 : sr[j+3], yi = zz ? 0.0 : si[j+3];
                  const double nr = p3r * yr - p3i * yi; p3i = p3r * yi + p3i * yr; p3r = nr; }
            }
            const double ar01 = p0r * p1r - p0i * p1i, ai01 = p0r * p1i + p0i * p1r;
            const double ar23 = p2r * p3r - p2i * p3i, ai23 = p2r * p3i + p2i * p3r;
            pr = ar01 * ar23 - ai01 * ai23; pi = ar01 * ai23 + ai01 * ar23;
        }

        // cross-thread: Q = product over the OTHER threads of the lane, ztot = total zero count
        double qr = 1.0, qi = 0.0;
        int ztot = z;
        #pragma unroll
        for (int o = 1; o < TPL; ++o) {
            const double ur = __shfl_xor_sync(grp, pr, o);
            const double ui = __shfl_xor_sync(grp, pi, o);
            ztot += __shfl_xor_sync(grp, z, o);
            const double nr = qr * ur - qi * ui; qi = qr * ui + qi * ur; qr = nr;
        }
        const double pallr = pr * qr - pi * qi, palli = pr * qi + pi * qr;   // full product, zeros -> 1

        if (MODE == 3) {
            if (ztot == 0) {
                // G[q] = (product of the other chunks) * (product of the other threads)
                double er[4], ei[4], fr[4], fi[4];
                er[0] = 1.0; ei[0] = 0.0;
                #pragma unroll
                for (int q = 1; q < 4; ++q) { er[q] = er[q-1] * ckr[q-1] - ei[q-1] * cki[q-1];
                                              ei[q] = er[q-1] * cki[q-1] + ei[q-1] * ckr[q-1]; }
                fr[3] = 1.0; fi[3] = 0.0;
                #pragma unroll
                for (int q = 2; q >= 0; --q) { fr[q] = fr[q+1] * ckr[q+1] - fi[q+1] * cki[q+1];
                                               fi[q] = fr[q+1] * cki[q+1] + fi[q+1] * ckr[q+1]; }
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    double gr = er[q] * fr[q] - ei[q] * fi[q], gi = er[q] * fi[q] + ei[q] * fr[q];
                    const double tr = gr * qr - gi * qi; gi = gr * qi + gi * qr; gr = tr;
                    gr *= w; gi *= w;
                    double br = 1.0, bi = 0.0;                      // suffix inside the chunk
                    #pragma unroll
                    for (int jj = Q4 - 1; jj >= 0; --jj) {
                        const int j = q * Q4 + jj;
                        const double tr2 = pfr[j] * br - pfi[j] * bi, ti2 = pfr[j] * bi + pfi[j] * br;
                        ar[j] += tr2 * gr - ti2 * gi;
                        ai[j] += tr2 * gi + ti2 * gr;
                        const double nr = br * sr[j] - bi * si[j]; bi = br * si[j] + bi * sr[j]; br = nr;
                    }
                }
            } else if (ztot == 1 && jz >= 0) {
                ar[jz] += w * pallr; ai[jz] += w * palli;
            }
        } else {
            const double wpr = w * pallr, wpi = w * palli;
            if (ztot == 0) {
                #pragma unroll
                for (int j = 0; j < KA; ++j) {
                    if (PRED && !(j < KH && c0 + j < K)) continue;
                    const double q2 = sr[j] * sr[j] + si[j] * si[j];
                    const double inv = (MODE == 2) ? frcp(q2) : (1.0 / q2);
                    ar[j] += (wpr * sr[j] + wpi * si[j]) * inv;
                    ai[j] += (wpi * sr[j] - wpr * si[j]) * inv;
                }
            } else if (ztot == 1) {
                if (PRED) {
                    #pragma unroll
                    for (int j = 0; j < KA; ++j) if (j < KH && c0 + j < K)
                        if (sr[j] == 0.0 && si[j] == 0.0) { ar[j] += wpr; ai[j] += wpi; }
                } else if (jz >= 0) { ar[jz] += wpr; ai[jz] += wpi; }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < KA; ++j) if (j < KH && c0 + j < K) out[c0 + j] = make_double2(ar[j], ai[j]);
}
"""


def _instantiations(tiers, tpls, modes):
    names = []
    for kp, tpl, mode in itertools.product(tiers, tpls, modes):
        if kp % tpl:
            continue
        kh = kp // tpl
        if kh < 4 or kh > 24:             # one thread never holds more than 24 columns
            continue
        if kh % 4 and mode not in (4, 5):  # only modes 4 and 5 pad KH up to a multiple of 4
            continue
        if mode == 3 and kh > 16:         # prefix array costs 2*KH extra registers
            continue
        names.append((kp, tpl, mode))
    return names


_MODULES = {}


def _module(names, maxreg):
    import cupy
    key = (tuple(names), maxreg)
    if key not in _MODULES:
        opts = ["-std=c++17"]
        if maxreg:
            opts.append(f"--maxrregcount={maxreg}")
        _MODULES[key] = cupy.RawModule(
            code=_SRC, options=tuple(opts),
            name_expressions=[f"cof_v<{kp}, {tpl}, {mode}>" for kp, tpl, mode in names])
    return _MODULES[key]


def _binom32(dev):
    b = torch.zeros(32, 32, dtype=torch.float64)
    for n in range(32):
        for k in range(n + 1):
            b[n, k] = math.comb(n, k)
    return b.reshape(-1).to(dev)


def make_instance(K, dev, lanes, terms_per_lane, seed=0):
    """Collision-free step: K1 = K-1 rows of multiplicity 1 (radix 2, the last one halved).

    This is the case that dominates the *mean* cost of a sample (the mean is carried by the
    rare low-collision instances), and the one the 2018 and 2020 kernels agree on.
    """
    K1 = K - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = (torch.randn(1, K1, K, 2, generator=g, dtype=torch.float64) / math.sqrt(2 * K)).to(dev)
    mult = torch.ones(1, K1, dtype=torch.int32, device=dev)
    radix = torch.full((1, K1), 2, dtype=torch.int32, device=dev)
    radix[0, K1 - 1] = 1                                    # halving on the odd-multiplicity row
    nterms = torch.tensor([lanes * terms_per_lane], dtype=torch.int64, device=dev)
    lstart = torch.zeros(1, dtype=torch.int64, device=dev)
    lane2s = torch.zeros(lanes, dtype=torch.int64, device=dev)
    return B, mult, radix, nterms, lstart, lane2s, K1


def _pad(B2, KP):
    """[1, K1, K, 2] -> [1, K1, KP, 2] with zeros in the padded columns."""
    S, K1, K, _ = B2.shape
    out = torch.zeros(S, K1, KP, 2, dtype=B2.dtype, device=B2.device)
    out[:, :, :K] = B2
    return out.contiguous()


def time_kernel(fn, warmup=1, reps=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def run(args):
    import cupy
    dev = torch.device("cuda")
    lanes = args.lanes
    tpl_list = tuple(int(x) for x in args.tpls.split(","))
    mode_list = tuple(int(x) for x in args.modes.split(","))
    block_list = tuple(int(x) for x in args.blocks.split(","))
    maxreg_list = tuple(int(x) if x != "0" else 0 for x in args.maxreg.split(","))
    tiers = TIERS_FINE if args.fine_tiers else TIERS_PROD
    ks = [int(x) for x in args.ks.split(",")]
    binom = _binom32(dev)
    rows = []

    def tiers_for(K):
        cand = [kp for kp in tiers if kp >= K]
        return cand if args.all_tiers else cand[:1]

    all_names = _instantiations(sorted({kp for K in ks for kp in tiers_for(K)}), tpl_list, mode_list)
    if not all_names:
        raise SystemExit("no valid (tier, TPL, mode) combination for this sweep")

    print(f"lanes={lanes} terms/lane={args.terms_per_lane} "
          f"total terms={lanes * args.terms_per_lane:.3e}  tiers={'fine' if args.fine_tiers else 'prod'}")
    print(f"{'K':>4} {'tier':>5} {'TPL':>4} {'mode':>5} {'blk':>4} {'rreg':>5} "
          f"{'Gterms/s':>9} {'rel.err':>9}  note")

    from photonic_engines.clifford.cuda_cofactors import _TIERS as PROD_TIERS
    from photonic_engines.clifford.cuda_cofactors import _get_kernel as prod_kernel

    for K in ks:
        B2, mult, radix, nterms, lstart, lane2s, K1 = make_instance(K, dev, lanes, args.terms_per_lane)
        total_terms = float(nterms.item())
        L = args.terms_per_lane

        # --- production kernel, same workload, as the reference value and the baseline time
        KPp = next(k for k in PROD_TIERS if k >= K)
        ref = None
        try:
            partp = torch.empty(lanes, K, 2, dtype=torch.float64, device=dev)
            kern = prod_kernel(KPp)
            tplp = PROD_TIERS[KPp]
            nth = lanes * tplp
            blk = 128
            cargs = (cupy.asarray(B2.contiguous()), cupy.asarray(mult), cupy.asarray(radix),
                     cupy.asarray(lane2s), cupy.asarray(lstart), cupy.asarray(nterms),
                     cupy.asarray(binom), cupy.asarray(partp),
                     cupy.int32(K1), cupy.int32(K), cupy.int64(L), cupy.int64(lanes))
            t = time_kernel(lambda: kern(((nth + blk - 1) // blk,), (blk,), cargs), reps=args.reps)
            ref = torch.view_as_complex(partp).sum(0).cpu()
            gt = total_terms / t / 1e9
            print(f"{K:>4} {KPp:>5} {tplp:>4} {'prod':>5} {blk:>4} {0:>5} {gt:>9.3f} {0.0:>9.1e}  production")
            rows.append({"K": K, "tier": KPp, "tpl": tplp, "mode": "prod", "block": blk, "maxreg": 0,
                             "gterms": gt, "relerr": 0.0})
        except Exception as e:
            print(f"{K:>4} production kernel failed: {e}")

        keep = set(tiers_for(K))
        names = [nm for nm in all_names if nm[0] in keep]
        if not names:
            continue
        for maxreg in maxreg_list:
            try:
                mod = _module(all_names, maxreg)
            except Exception as e:
                print(f"  module compile failed (maxrregcount={maxreg}): {e}")
                continue
            for (KP, tpl, mode), blk in itertools.product(names, block_list):
                try:
                    fn = mod.get_function(f"cof_v<{KP}, {tpl}, {mode}>")
                except Exception as e:
                    print(f"{K:>4} {KP:>5} {tpl:>4} {mode:>5} {blk:>4} {maxreg:>5}  get_function: {e}")
                    continue
                Bp = _pad(B2, KP)
                part = torch.empty(lanes, KP, 2, dtype=torch.float64, device=dev)
                nth = lanes * tpl
                cargs = (cupy.asarray(Bp), cupy.asarray(mult), cupy.asarray(radix),
                         cupy.asarray(lane2s), cupy.asarray(lstart), cupy.asarray(nterms),
                         cupy.asarray(binom), cupy.asarray(part),
                         cupy.int32(K1), cupy.int32(K), cupy.int64(L), cupy.int64(lanes))
                try:
                    t = time_kernel(lambda: fn(((nth + blk - 1) // blk,), (blk,), cargs), reps=args.reps)
                except Exception as e:
                    print(f"{K:>4} {KP:>5} {tpl:>4} {mode:>5} {blk:>4} {maxreg:>5}  launch: {e}")
                    continue
                got = torch.view_as_complex(part).sum(0)[:K].cpu()
                err = 0.0 if ref is None else float(
                    (got - ref).abs().max() / max(ref.abs().max().item(), 1e-300))
                gt = total_terms / t / 1e9
                note = "" if err < 1e-10 else "*** MISMATCH ***"
                print(f"{K:>4} {KP:>5} {tpl:>4} {mode:>5} {blk:>4} {maxreg:>5} "
                      f"{gt:>9.3f} {err:>9.1e}  {note}")
                rows.append({"K": K, "tier": KP, "tpl": tpl, "mode": mode, "block": blk, "maxreg": maxreg,
                                 "gterms": gt, "relerr": err})

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default="16,24,32,40,48,56,64",
                   help="column counts K (= photon index k of the step) to measure")
    p.add_argument("--tpls", default="1,2,4", help="threads per lane to try")
    p.add_argument("--modes", default="0,4,5", help="kernel variants to try")
    p.add_argument("--blocks", default="128", help="block sizes to try")
    p.add_argument("--maxreg", default="0", help="--maxrregcount values (0 = compiler default)")
    p.add_argument("--lanes", type=int, default=1 << 17)
    p.add_argument("--terms-per-lane", type=int, default=2048)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--fine-tiers", action="store_true",
                   help="use the fine tier ladder (28/36/40/44/52/56/60) instead of the production one")
    p.add_argument("--all-tiers", action="store_true",
                   help="also time K in every tier >= K (measures the cost of padding)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--csv", default="")
    a = p.parse_args()
    if a.quick:
        a.ks, a.terms_per_lane = "24,40,64", 512
    run(a)
