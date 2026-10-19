"""
CUDA (NVRTC via cupy) thread-per-lane version of the emitter cofactor kernel
(docs/EMITTER_SAMPLER.md §8; cross-engine review item #5).

Same lane decomposition and host planning as `emitter_triton.emitter_cofactor_partials_seq`
(lane = (pure term λ, Gray segment), mixed-radix reflected Gray walk over the Clifford & Clifford
2020 collapse of repeated rows), and the same per-step mathematics as the seq kernel:

    column sums   s_{c,l} += f · U[r, mode(c, l)]                 (NL labels per slot)
    slot matrices G_c = Σ_l s_{c,l} M̃_{c,l}                         (2 × 2 complex)
    p_c = G_{c−1} … G_0 s0,   q_cᵀ = f† G_{K−1} … G_{c+1},   acc_c += w · p_c q_cᵀ

but written as ONE family with the Fock kernel `cuda_cofactors.cof_multi`: the odometer (byte
arrays, `d/v/R/M/pp`), the lane -> (stream, segment) bookkeeping, the tier ladder in steps of 4
and the per-tier (threads per lane, register cap) table are the Fock kernel's; only the payload
differs (2 × 2 chains instead of scalar all-but-one products, no zero handling).  A lane is TPL
consecutive threads, each owning KH = ⌈KP / TPL⌉ consecutive columns (column sums, stored p_c and
accumulators in registers); the block products H_h = G_last … G_first of the TPL threads are
joined once per term by two Kogge–Stone scans over `__shfl_up_sync` / `__shfl_down_sync`
(exclusive prefix E_h and suffix F_h), so p starts at E_h s0 and qᵀ at f† F_h in every block.

Input block: Bs [S, K1, K, NL] complex128 (rows = distinct chosen rows of the stream, columns =
(kept slot, label) modes), gathered once per step by the host as the Fock path gathers its Bs,
so the per-term row read is contiguous; term matrices mats2 [Λ, K, 2, 2, 2] (complex), boundary
vectors s02 / f2 [Λ, 2].  Output: part [TOTAL, K, 2, 2] complex — the host reduction of
`emitter_step_weights` is unchanged.

The kernel source is plain C++ apart from the exchange primitive: compiled by NVRTC as
`emit_multi<KP, TPL, NL, GS>` on a GPU, and by the host C++ compiler as a CPU emulation
(`emitter_cofactor_partials_emul`: the TPL threads of a lane are std::threads joined by a
barrier, so the SAME kernel code runs -- including the shared-accumulator path PE_ACCSH, emulated with a per-thread
buffer of the same layout; only PE_MSH, which needs a block-wide __syncthreads, is GPU-only and checked on the device by
tests/clifford/check_cuda_emitter.py) for the CPU test-suite.  GS = 1 keeps the K slot matrices
in registers across the three passes of a step, GS = 0 recomputes them (fewer registers, more
FMAs) — a tuning knob per tier like TPL.  Requires `cupy` for the GPU path.

Since 2026-10-05 (runs 20261005_213240 / _221051 / _223132): the production build folds w into the backward seed (PE_WQ:
every q_j is linear in its seed, so the per-column scaling of the accumulation goes) and keeps the 2 x 2 accumulators in
shared memory (PE_ACCSH: [column][entry][thread], 64-thread blocks up to 10 columns per thread, 32 up to 20) -- spills
halved, 1.27-1.31x.  PE_MSH (term matrices in shared memory) and PE_SYM (TPL = 2 lanes without the block scan: forward on
thread 0 and backward on thread 1, one vector exchange, then the other chain -- the minimum op count, but 18 columns of
state per thread spill and it runs 2-6x slower at K = 36) are kept as measured, non-default variants.
"""
import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile

import torch

_SRC = r"""
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define PE_FN __device__ __forceinline__
#else
#include <cmath>
#include <barrier>
#include <thread>
#include <vector>
#define PE_FN static inline
struct double2 { double x, y; };
static inline double2 make_double2(double x, double y) { double2 r; r.x = x; r.y = y; return r; }
#endif
// Arithmetic type of the sums, chains and accumulators: double on the GPU; the CPU emulation can be
// compiled with -DPE_REAL="long double" (x87 80-bit: 64-bit mantissa) as a higher-precision
// reference for the rounding error of the fp64 kernels (bench/emitter_precision.py). Inputs and
// outputs stay double.
#ifdef __CUDACC__
typedef double real_t;
#else
#ifndef PE_REAL
#define PE_REAL double
#endif
typedef PE_REAL real_t;
#endif
// Kernel variants, compile-time, all 0 = the kernel of 2026-09-27 (production sets PE_WQ and PE_ACCSH):
// PE_WQ    w folded into the backward seed q once per term (q is linear in its seed), no per-column w scaling;
// PE_MSH   the term matrices of the block's first lane copied once into shared memory, read from there by every
//          lane of the block on that term (other lanes read global memory as before) -- GPU only;
// PE_ACCSH the 2 x 2 accumulators in shared memory (64-thread blocks), freeing 8 doubles per column -- GPU only.
#ifndef PE_WQ
#define PE_WQ 0
#endif
#ifndef PE_MSH
#define PE_MSH 0
#endif
#ifndef PE_ACCSH
#define PE_ACCSH 0
#endif
#define PE_ACCSH_KH 20                                     // tiers with more columns per thread keep registers (48 KB)
#ifndef PE_SYM
#define PE_SYM 0                                           // TPL = 2 lanes without the block scan (emit_lane_sym)
#endif
// shared accumulator stride = threads per block: 64 up to 10 columns per thread, 32 up to PE_ACCSH_KH (<= 41 KB)
#define PE_ACCSH_B(KH) ((KH) <= 10 ? 64 : 32)
#ifndef __CUDACC__
#undef PE_MSH                                              // needs a block-wide __syncthreads: GPU only
#define PE_MSH 0
#endif                                                     // PE_ACCSH is emulated (a per-thread buffer, same layout)

// ---- 2 x 2 complex algebra on plain doubles (row-major e[a*2+b]) ----
struct C2 { real_t r, i; };
struct M2 { C2 e[4]; };
PE_FN C2 cmul(C2 a, C2 b) { C2 c; c.r = a.r * b.r - a.i * b.i; c.i = a.r * b.i + a.i * b.r; return c; }
PE_FN C2 cadd(C2 a, C2 b) { C2 c; c.r = a.r + b.r; c.i = a.i + b.i; return c; }
PE_FN C2 cscale(real_t w, C2 a) { C2 c; c.r = w * a.r; c.i = w * a.i; return c; }
PE_FN M2 mident() { M2 m; m.e[0].r = 1.0; m.e[0].i = 0.0; m.e[1].r = 0.0; m.e[1].i = 0.0;
                          m.e[2].r = 0.0; m.e[2].i = 0.0; m.e[3].r = 1.0; m.e[3].i = 0.0; return m; }
PE_FN M2 mmul(const M2& X, const M2& Y) {                 // X Y
    M2 Z;
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < 2; ++b)
            Z.e[a * 2 + b] = cadd(cmul(X.e[a * 2], Y.e[b]), cmul(X.e[a * 2 + 1], Y.e[2 + b]));
    return Z;
}
PE_FN void mvec(const M2& X, C2& pa, C2& pb) {            // p <- X p
    C2 na = cadd(cmul(X.e[0], pa), cmul(X.e[1], pb));
    C2 nb = cadd(cmul(X.e[2], pa), cmul(X.e[3], pb));
    pa = na; pb = nb;
}
PE_FN void vecm(C2& qa, C2& qb, const M2& X) {            // q^T <- q^T X
    C2 na = cadd(cmul(qa, X.e[0]), cmul(qb, X.e[2]));
    C2 nb = cadd(cmul(qa, X.e[1]), cmul(qb, X.e[3]));
    qa = na; qb = nb;
}

// ---- exchange primitive between the TPL threads of a lane ----
#ifdef __CUDACC__
template <int TPL> struct Xchg {
    unsigned grp;
    __device__ M2 up(const M2& x, int o) const {          // value of thread h - o (own value if h < o)
        M2 y;
        #pragma unroll
        for (int j = 0; j < 4; ++j) { y.e[j].r = __shfl_up_sync(grp, x.e[j].r, o, TPL); y.e[j].i = __shfl_up_sync(grp, x.e[j].i, o, TPL); }
        return y;
    }
    __device__ M2 down(const M2& x, int o) const {        // value of thread h + o (own value if h + o >= TPL)
        M2 y;
        #pragma unroll
        for (int j = 0; j < 4; ++j) { y.e[j].r = __shfl_down_sync(grp, x.e[j].r, o, TPL);
                                      y.e[j].i = __shfl_down_sync(grp, x.e[j].i, o, TPL); }
        return y;
    }
};
#else
template <int TPL> struct Xchg {                          // CPU emulation: TPL std::threads and a barrier
    M2* slots; std::barrier<>* bar; int h;
    M2 up(const M2& x, int o) const {
        slots[h] = x; bar->arrive_and_wait();
        M2 y = (h >= o) ? slots[h - o] : x; bar->arrive_and_wait();
        return y;
    }
    M2 down(const M2& x, int o) const {
        slots[h] = x; bar->arrive_and_wait();
        M2 y = (h + o < TPL) ? slots[h + o] : x; bar->arrive_and_wait();
        return y;
    }
};
#endif

// KP = padded column count of the tier, TPL = threads per lane (power of 2, <= 32), NL = labels
// per slot (1 or 2), GS = 1: the KH slot matrices are kept across the three passes of a step.
// Column j of this thread is c0 + j; accesses past KH are removed at compile time and columns
// >= K idle (G = I), so any (KP, TPL) is legal.
template <int KP, int TPL, int NL, int GS>
PE_FN void emit_lane(const double2* __restrict__ Bs,       // [S, K1, K, NL]
                     const int*  __restrict__ mult,        // [S, K1]
                     const int*  __restrict__ radix,       // [S, K1]
                     const long long* __restrict__ lane2term,   // [TOTAL]
                     const long long* __restrict__ term2stream, // [LAMBDA]
                     const long long* __restrict__ lstart,      // [LAMBDA]
                     const long long* __restrict__ nterms,      // [S]
                     const double* __restrict__ binom,          // [32*32]
                     const double2* __restrict__ mats,          // [LAMBDA, K, 2(L), 2, 2]
                     const double2* __restrict__ s0v,           // [LAMBDA, 2]
                     const double2* __restrict__ fv,            // [LAMBDA, 2]
                     double2* __restrict__ part,                // [TOTAL, K, 2, 2]
                     int K1, int K, long long stride_mt, long long stride_mj, long long L,
                     long long g, int h, const Xchg<TPL>& xc,
                     const double2* msh = nullptr, long long lam0 = -1,    // PE_MSH: block's term matrices, its term
                     double2* ash = nullptr)                               // PE_ACCSH: this thread's slot, stride 64
{
    constexpr int KH = (KP + TPL - 1) / TPL;
    constexpr bool ACCS = PE_ACCSH && KH <= PE_ACCSH_KH;
    constexpr int AB = PE_ACCSH_B(KH);
    const int c0 = h * KH;
    const long long lam = lane2term[g];
    const long long s = term2stream[lam];
    const long long li = g - lstart[lam];
    const long long Ns = nterms[s];
    double2* out = part + g * (long long)K * 4;
    const long long base = li * L;
    if (base >= Ns) {
        #pragma unroll
        for (int j = 0; j < KH; ++j) if (c0 + j < K) for (int e = 0; e < 4; ++e) out[(c0 + j) * 4 + e] = make_double2(0.0, 0.0);
        return;
    }
    const double2* B = Bs + (s * (long long)K1 * K + c0) * NL;
    const double2* Mt = (PE_MSH && lam == lam0) ? msh + c0 * stride_mj : mats + lam * stride_mt + c0 * stride_mj;
    const int* Ms = mult + s * K1;
    const int* Rs = radix + s * K1;

    // odometer state (all values < 64): bytes, so the per-thread local footprint stays L1-resident
    unsigned char d[KP], v[KP], R[KP], M[KP], pp[KP];      // pp[nu] = parity of sum_{mu > nu} v_mu
    for (int nu = 0; nu < K1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < K1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = K1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1);
                                             v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    real_t w = 1.0;
    for (int nu = 0; nu < K1; ++nu) { real_t b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    // column sums at the segment start, accumulators, stored p_c
    C2 sm[KH][NL];
    M2 acc[ACCS ? 1 : KH];
    #pragma unroll
    for (int j = 0; j < KH; ++j) {
        #pragma unroll
        for (int l = 0; l < NL; ++l) { sm[j][l].r = 0.0; sm[j][l].i = 0.0; }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            if constexpr (ACCS) ash[(j * 4 + e) * AB] = make_double2(0.0, 0.0);
            else { acc[j].e[e].r = 0.0; acc[j].e[e].i = 0.0; }
        }
    }
    for (int nu = 0; nu < K1; ++nu) {
        const real_t c = (real_t)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + (long long)nu * K * NL;
        #pragma unroll
        for (int j = 0; j < KH; ++j) if (c0 + j < K) {
            #pragma unroll
            for (int l = 0; l < NL; ++l) { double2 b = row[j * NL + l]; sm[j][l].r += c * b.x; sm[j][l].i += c * b.y; }
        }
    }
    // boundary vectors: p_start = s0, q_end^T = f^dag
    C2 s0a, s0b, fa, fb;
    { double2 t = s0v[lam * 2]; s0a.r = t.x; s0a.i = t.y; t = s0v[lam * 2 + 1]; s0b.r = t.x; s0b.i = t.y;
      t = fv[lam * 2]; fa.r = t.x; fa.i = -t.y; t = fv[lam * 2 + 1]; fb.r = t.x; fb.i = -t.y; }

    // slot matrix of column j from the current sums (identity on idle columns)
    auto slot = [&](int j) -> M2 {
        M2 G;
        if (c0 + j < K) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                C2 acc_e; acc_e.r = 0.0; acc_e.i = 0.0;
                #pragma unroll
                for (int l = 0; l < NL; ++l) { double2 m = Mt[j * stride_mj + l * 4 + e]; C2 me; me.r = m.x; me.i = m.y;
                                               acc_e = cadd(acc_e, cmul(sm[j][l], me)); }
                G.e[e] = acc_e;
            }
        } else {
            G = mident();
        }
        return G;
    };

    const long long Lp = (L < Ns - base) ? L : (Ns - base);
    for (long long t = 0; t < Lp; ++t) {
        if (t > 0) {
            int nu = 0;
            while (d[nu] == R[nu] - 1) ++nu;
            const int dirn = pp[nu] ? -1 : 1;
            const int vold = v[nu], mnu = M[nu];
            for (int mu = 0; mu < nu; ++mu) { d[mu] = 0; pp[mu] ^= 1; }
            d[nu] += 1; v[nu] = (unsigned char)(vold + dirn);
            w *= (dirn > 0) ? -(real_t)(mnu - vold) / (real_t)(vold + 1)
                            : -(real_t)vold / (real_t)(mnu - vold + 1);
            const real_t f = -2.0 * dirn;
            const double2* row = B + (long long)nu * K * NL;
            #pragma unroll
            for (int j = 0; j < KH; ++j) if (c0 + j < K) {
                #pragma unroll
                for (int l = 0; l < NL; ++l) { double2 b = row[j * NL + l]; sm[j][l].r += f * b.x; sm[j][l].i += f * b.y; }
            }
        }
        M2 Gs[GS ? KH : 1];
        // pass 1 (TPL > 1 only): block product H = G_{KH-1} ... G_0, exclusive prefix / suffix over the lane
        C2 pa = s0a, pb = s0b, qa = fa, qb = fb;
        if (TPL > 1) {
            M2 H = mident();
            #pragma unroll
            for (int j = 0; j < KH; ++j) { M2 G = slot(j); if (GS) Gs[j] = G; H = mmul(G, H); }
            M2 P = H, Sf = H;
            #pragma unroll
            for (int o = 1; o < TPL; o <<= 1) {                    // Kogge-Stone: P_h = H_h ... H_0, S_h = H_{TPL-1} ... H_h
                M2 Q = xc.up(P, o);   if (h >= o)      P = mmul(P, Q);
                M2 T = xc.down(Sf, o); if (h + o < TPL) Sf = mmul(T, Sf);
            }
            M2 E = xc.up(P, 1);   if (h == 0)       E = mident();   // E_h = H_{h-1} ... H_0
            M2 F = xc.down(Sf, 1); if (h == TPL - 1) F = mident();  // F_h = H_{TPL-1} ... H_{h+1}
            mvec(E, pa, pb);
            vecm(qa, qb, F);
        }
        if constexpr (PE_WQ) { qa = cscale(w, qa); qb = cscale(w, qb); }    // every q_j = w x (unscaled q_j): linear chain
        // pass 2: forward chain, p_j stored
        C2 pj[KH][2];
        #pragma unroll
        for (int j = 0; j < KH; ++j) {
            pj[j][0] = pa; pj[j][1] = pb;
            if (j + 1 < KH) {
                M2 G;
                if (GS && TPL > 1) G = Gs[j]; else { G = slot(j); if (GS) Gs[j] = G; }
                mvec(G, pa, pb);
            }
        }
        // pass 3: backward chain with accumulation acc_j += w p_j q_j^T
        #pragma unroll
        for (int j = KH - 1; j >= 0; --j) {
            if (c0 + j < K) {
                const real_t ws = PE_WQ ? (real_t)1.0 : w;
                C2 t[4] = {cmul(pj[j][0], qa), cmul(pj[j][0], qb), cmul(pj[j][1], qa), cmul(pj[j][1], qb)};
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if constexpr (!PE_WQ) t[e] = cscale(ws, t[e]);
                    if constexpr (ACCS) { double2 a = ash[(j * 4 + e) * AB]; a.x += t[e].r; a.y += t[e].i; ash[(j * 4 + e) * AB] = a; }
                    else acc[j].e[e] = cadd(acc[j].e[e], t[e]);
                }
            }
            if (j > 0) {
                M2 G;
                if (GS && (TPL > 1 || j + 1 < KH)) G = Gs[j]; else G = slot(j);
                vecm(qa, qb, G);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < KH; ++j) if (c0 + j < K) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            if constexpr (ACCS) out[(c0 + j) * 4 + e] = ash[(j * 4 + e) * AB];
            else out[(c0 + j) * 4 + e] = make_double2(acc[j].e[e].r, acc[j].e[e].i);
        }
    }
}

// PE_SYM: a TPL = 2 lane WITHOUT the 2 x 2 block scan (measured, not the default).  Thread 0 owns the
// first KH columns, thread 1 the last KH, in REVERSED register order (slot j <-> column c0 + KH - 1 - j), so both run
// the same instructions:
//   phase A (slots ascending):  a_j stored, a <- X_j a   from a = s0 (thread 0: X = G, the forward chain p)
//                                                       from a = conj(f) (thread 1: X = G^T, the backward chain q)
//   exchange the end vectors (thread 0's = p at thread 1's first column, thread 1's = q at thread 0's last column);
//   phase B (slots descending): t = a_j b^T (transposed on thread 1), acc_j += t, b <- Y_j b   from b = w x received
//                                                       (thread 0: Y = G^T, backward; thread 1: Y = G, forward)
// Per column and term: the slot matrix, one mat-vec in each phase and the outer product -- C's algorithmic minimum
// (ALGORITHMS §9.2 row 4) with nothing else (no H = G...G products, no Kogge-Stone).  GS = 1 keeps the slot matrices
// from phase A for phase B; GS = 0 recomputes them.
PE_FN M2 mtrans(const M2& X) { M2 Y = X; Y.e[1] = X.e[2]; Y.e[2] = X.e[1]; return Y; }

template <int KP, int NL, int GS>
PE_FN void emit_lane_sym(const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix,
                         const long long* __restrict__ lane2term, const long long* __restrict__ term2stream,
                         const long long* __restrict__ lstart, const long long* __restrict__ nterms,
                         const double* __restrict__ binom, const double2* __restrict__ mats,
                         const double2* __restrict__ s0v, const double2* __restrict__ fv, double2* __restrict__ part,
                         int K1, int K, long long stride_mt, long long stride_mj, long long L,
                         long long g, int h, const Xchg<2>& xc,
                         const double2* msh = nullptr, long long lam0 = -1, double2* ash = nullptr)
{
    constexpr int KH = (KP + 1) / 2;
    constexpr bool ACCS = PE_ACCSH && KH <= PE_ACCSH_KH;
    constexpr int AB = PE_ACCSH_B(KH);
    const int c0 = h * KH;
    auto col = [&](int j) -> int { return h ? c0 + KH - 1 - j : c0 + j; };
    const long long lam = lane2term[g];
    const long long s = term2stream[lam];
    const long long li = g - lstart[lam];
    const long long Ns = nterms[s];
    double2* out = part + g * (long long)K * 4;
    const long long base = li * L;
    if (base >= Ns) {
        #pragma unroll
        for (int j = 0; j < KH; ++j) if (col(j) < K) for (int e = 0; e < 4; ++e) out[col(j) * 4 + e] = make_double2(0.0, 0.0);
        return;
    }
    const double2* B = Bs + s * (long long)K1 * K * NL;
    const double2* Mt = (PE_MSH && lam == lam0) ? msh : mats + lam * stride_mt;
    const int* Ms = mult + s * K1;
    const int* Rs = radix + s * K1;

    unsigned char d[KP], v[KP], R[KP], M[KP], pp[KP];
    for (int nu = 0; nu < K1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < K1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = K1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1);
                                             v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    real_t w = 1.0;
    for (int nu = 0; nu < K1; ++nu) { real_t b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    C2 sm[KH][NL];
    M2 acc[ACCS ? 1 : KH];
    #pragma unroll
    for (int j = 0; j < KH; ++j) {
        #pragma unroll
        for (int l = 0; l < NL; ++l) { sm[j][l].r = 0.0; sm[j][l].i = 0.0; }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            if constexpr (ACCS) ash[(j * 4 + e) * AB] = make_double2(0.0, 0.0);
            else { acc[j].e[e].r = 0.0; acc[j].e[e].i = 0.0; }
        }
    }
    for (int nu = 0; nu < K1; ++nu) {
        const real_t c = (real_t)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + (long long)nu * K * NL;
        #pragma unroll
        for (int j = 0; j < KH; ++j) if (col(j) < K) {
            #pragma unroll
            for (int l = 0; l < NL; ++l) { double2 b = row[col(j) * NL + l]; sm[j][l].r += c * b.x; sm[j][l].i += c * b.y; }
        }
    }
    // phase-A start: s0 on thread 0, conj(f) on thread 1
    C2 a0, a1;
    { const double2* src = h ? fv : s0v; const real_t sg = h ? -1.0 : 1.0;
      double2 t = src[lam * 2]; a0.r = t.x; a0.i = sg * t.y; t = src[lam * 2 + 1]; a1.r = t.x; a1.i = sg * t.y; }

    auto slot = [&](int j) -> M2 {                        // G of column col(j) (identity on idle columns)
        M2 G;
        if (col(j) < K) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                C2 acc_e; acc_e.r = 0.0; acc_e.i = 0.0;
                #pragma unroll
                for (int l = 0; l < NL; ++l) { double2 m = Mt[col(j) * stride_mj + l * 4 + e]; C2 me; me.r = m.x; me.i = m.y;
                                               acc_e = cadd(acc_e, cmul(sm[j][l], me)); }
                G.e[e] = acc_e;
            }
        } else {
            G = mident();
        }
        return G;
    };

    const long long Lp = (L < Ns - base) ? L : (Ns - base);
    for (long long t = 0; t < Lp; ++t) {
        if (t > 0) {
            int nu = 0;
            while (d[nu] == R[nu] - 1) ++nu;
            const int dirn = pp[nu] ? -1 : 1;
            const int vold = v[nu], mnu = M[nu];
            for (int mu = 0; mu < nu; ++mu) { d[mu] = 0; pp[mu] ^= 1; }
            d[nu] += 1; v[nu] = (unsigned char)(vold + dirn);
            w *= (dirn > 0) ? -(real_t)(mnu - vold) / (real_t)(vold + 1)
                            : -(real_t)vold / (real_t)(mnu - vold + 1);
            const real_t f = -2.0 * dirn;
            const double2* row = B + (long long)nu * K * NL;
            #pragma unroll
            for (int j = 0; j < KH; ++j) if (col(j) < K) {
                #pragma unroll
                for (int l = 0; l < NL; ++l) { double2 b = row[col(j) * NL + l]; sm[j][l].r += f * b.x; sm[j][l].i += f * b.y; }
            }
        }
        M2 Gs[GS ? KH : 1];
        C2 av[KH][2];
        C2 xa = a0, xb = a1;
        #pragma unroll
        for (int j = 0; j < KH; ++j) {                    // phase A
            av[j][0] = xa; av[j][1] = xb;
            M2 G = slot(j);
            if (GS) Gs[j] = G;
            mvec(h ? mtrans(G) : G, xa, xb);
        }
        M2 X = mident(); X.e[0] = xa; X.e[1] = xb;        // exchange the end vectors (packed in an M2)
        const M2 Yu = xc.up(X, 1), Yd = xc.down(X, 1);
        const M2& Y = h ? Yu : Yd;
        C2 ba = cscale(w, Y.e[0]), bb = cscale(w, Y.e[1]);
        #pragma unroll
        for (int j = KH - 1; j >= 0; --j) {               // phase B
            if (col(j) < K) {
                C2 tt[4] = {cmul(av[j][0], ba), cmul(av[j][0], bb), cmul(av[j][1], ba), cmul(av[j][1], bb)};
                if (h) { C2 x = tt[1]; tt[1] = tt[2]; tt[2] = x; }   // thread 1: p = b, q = a -> transpose
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if constexpr (ACCS) { double2 a = ash[(j * 4 + e) * AB]; a.x += tt[e].r; a.y += tt[e].i; ash[(j * 4 + e) * AB] = a; }
                    else acc[j].e[e] = cadd(acc[j].e[e], tt[e]);
                }
            }
            if (j > 0) {
                const M2 G = GS ? Gs[j] : slot(j);
                mvec(h ? G : mtrans(G), ba, bb);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < KH; ++j) if (col(j) < K) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            if constexpr (ACCS) out[col(j) * 4 + e] = ash[(j * 4 + e) * AB];
            else out[col(j) * 4 + e] = make_double2(acc[j].e[e].r, acc[j].e[e].i);
        }
    }
}

#ifdef __CUDACC__
#define PE_EMIT_ARGS const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix, \
    const long long* __restrict__ lane2term, const long long* __restrict__ term2stream, \
    const long long* __restrict__ lstart, const long long* __restrict__ nterms, const double* __restrict__ binom, \
    const double2* __restrict__ mats, const double2* __restrict__ s0v, const double2* __restrict__ fv, \
    double2* __restrict__ part, int K1, int K, long long stride_mt, long long stride_mj, long long L, long long TOTAL
#define PE_EMIT_PARAMS Bs, mult, radix, lane2term, term2stream, lstart, nterms, binom, mats, s0v, fv, part, K1, K, \
    stride_mt, stride_mj, L, TOTAL
// The kernel body lives in emit_entry (below); two entry points reach it: the template emit_multi (by name_expressions,
// host flag PE_PLAIN_ENTRY=0) and the plain-named extern "C" wrappers PE_EMIT_ENTRY (default since run 20261006_115902:
// cupy reads its disk cache only for modules compiled WITHOUT name_expressions), one wrapper line per (tier, TPL, NL, GS)
// appended to the source by the host.
template <int KP, int TPL, int NL, int GS> __device__ __forceinline__ void emit_entry(PE_EMIT_ARGS);
template <int KP, int TPL, int NL, int GS>
__global__ void emit_multi(PE_EMIT_ARGS) { emit_entry<KP, TPL, NL, GS>(PE_EMIT_PARAMS); }
#define PE_EMIT_ENTRY(KP, TPL, NL, GS) extern "C" __global__ void emit_k_##KP##_##TPL##_##NL##_##GS(PE_EMIT_ARGS) \
    { emit_entry<KP, TPL, NL, GS>(PE_EMIT_PARAMS); }

template <int KP, int TPL, int NL, int GS>
__device__ __forceinline__ void emit_entry(PE_EMIT_ARGS)
{
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long g = gt / TPL;
    const int h = (int)(gt % TPL);
    const double2* msh = nullptr;
    long long lam0 = -1;
#if PE_MSH
    __shared__ double2 msh_buf[KP * 8];                   // one term's [K, 2 labels, 2, 2] block (stride_mj = 8)
    {
        const long long gfirst = ((long long)blockIdx.x * blockDim.x) / TPL;
        lam0 = lane2term[gfirst < TOTAL ? gfirst : TOTAL - 1];
        for (int i = threadIdx.x; i < K * (int)stride_mj; i += blockDim.x) msh_buf[i] = mats[lam0 * stride_mt + i];
        __syncthreads();
        msh = msh_buf;
    }
#endif
    double2* ash = nullptr;
#if PE_ACCSH
    constexpr int KH = (KP + TPL - 1) / TPL;
    __shared__ double2 ash_buf[(KH <= PE_ACCSH_KH ? KH : 1) * 4 * PE_ACCSH_B(KH)];   // [column j][entry e][thread]
    ash = ash_buf + threadIdx.x;
#endif
    if (g >= TOTAL) return;
    Xchg<TPL> xc; xc.grp = ((TPL >= 32) ? 0xffffffffu : (1u << TPL) - 1u) << ((threadIdx.x & 31) & ~(TPL - 1));
    if constexpr (PE_SYM && TPL == 2)
        emit_lane_sym<KP, NL, GS>(Bs, mult, radix, lane2term, term2stream, lstart, nterms, binom, mats, s0v, fv, part,
                                  K1, K, stride_mt, stride_mj, L, g, h, xc, msh, lam0, ash);
    else
        emit_lane<KP, TPL, NL, GS>(Bs, mult, radix, lane2term, term2stream, lstart, nterms, binom, mats, s0v, fv, part,
                                   K1, K, stride_mt, stride_mj, L, g, h, xc, msh, lam0, ash);
}
#else
// CPU emulation of one configuration (KP, TPL, NL, GS fixed by -D at compile time): every lane
// runs its TPL threads as std::threads joined by a barrier, so the exchange code above is the
// code under test.  Lanes are processed in parallel batches.
extern "C" void emit_emul(const double* Bs, const int* mult, const int* radix, const long long* lane2term,
                          const long long* term2stream, const long long* lstart, const long long* nterms,
                          const double* binom, const double* mats, const double* s0v, const double* fv, double* part,
                          int K1, int K, long long stride_mt, long long stride_mj, long long L, long long TOTAL)
{
    constexpr int TPL = PE_TPL;
    const long long nb = 8;                                    // lanes per batch
    for (long long g0 = 0; g0 < TOTAL; g0 += nb) {
        std::vector<std::thread> th;
        std::vector<std::barrier<>*> bars;
        std::vector<M2*> slots;
        for (long long g = g0; g < TOTAL && g < g0 + nb; ++g) {
            std::barrier<>* bar = new std::barrier<>(TPL);
            M2* sl = new M2[TPL];
            bars.push_back(bar); slots.push_back(sl);
            for (int h = 0; h < TPL; ++h)
                th.emplace_back([=]() {
                    Xchg<TPL> xc; xc.slots = sl; xc.bar = bar; xc.h = h;
                    // PE_ACCSH: this thread's slice of the shared accumulator block, laid out exactly as on the GPU
                    // ([column j][entry e] at stride PE_ACCSH_B(KH)), so the ACCS code path itself is what runs here
                    constexpr int KHE = (PE_KP + TPL - 1) / TPL;
                    std::vector<double2> abuf(PE_ACCSH ? (size_t)KHE * 4 * PE_ACCSH_B(KHE) : 1);
                    double2* ash = PE_ACCSH ? abuf.data() : nullptr;
#if PE_SYM && PE_TPL == 2
                    emit_lane_sym<PE_KP, PE_NL, PE_GS>((const double2*)Bs, mult, radix, lane2term, term2stream, lstart,
                                                       nterms, binom, (const double2*)mats, (const double2*)s0v,
                                                       (const double2*)fv, (double2*)part, K1, K, stride_mt, stride_mj,
                                                       L, g, h, xc, nullptr, -1, ash);
#else
                    emit_lane<PE_KP, TPL, PE_NL, PE_GS>((const double2*)Bs, mult, radix, lane2term, term2stream, lstart,
                                                        nterms, binom, (const double2*)mats, (const double2*)s0v,
                                                        (const double2*)fv, (double2*)part, K1, K, stride_mt, stride_mj,
                                                        L, g, h, xc, nullptr, -1, ash);
#endif
                });
        }
        for (auto& t : th) t.join();
        for (auto* b : bars) delete b;
        for (auto* s : slots) delete[] s;
    }
}
#endif
"""

# ---- tier table: K_PAD tier -> (threads per lane, GS, --maxrregcount) ----
# Measured on the H100 (run 20260927_110150, bench/bench_cuda_emitter.py, 8 streams,
# 2026-09-27): GS = 1 (slot matrices kept across the three passes) wins at every tier from 20 on
# by 1.4-1.8x over recomputing them, TPL = 4 (KH = 5-7 columns per thread) beats TPL = 8 (KH = 3-4)
# by 1.6-1.8x and TPL = 2 by 1.4-2.6x -- the register budget binds less than the pre-measurement
# estimate (which kept KH <= 4) and the exchange costs more; every --maxrregcount cap loses
# (128: 2-3x, 168: 1.3-1.9x; 255 = the compiler's choice).  Against the Triton seq kernel:
# 1.09x (K = 16), 1.24x (20), 1.04x (24), 1.49x (28).  Tiers 8-12 and 32-40 are not measured:
# 8-12 keep the default, 32-40 inherit the tier-28 entry.
#
# Two REGIMES, chosen per launch from the lanes per term (runs 20260927_153353 and
# 20260927_205524): with many segments per term ("seg": the frontier steps, lanes/term >=
# SCATTER_LANES) a warp holds 32 / TPL lanes of ONE term -- the table above.  With one lane per
# term ("scatter": the middle of the chain at 64-256 shots, thousands of terms per stream,
# N_s <= L) this kernel runs at a flat 15-16 G column-updates/s at K = 12-20 whatever TPL
# (2 / 4 / 8 / 16 measured at 128 streams, n = 30: TPL 4 best, 8 and 16 slower), while the seq
# kernel reaches 25-31 G there: 0.5-0.6x.  The "more threads per lane" hypothesis (fewer
# distinct term-matrix rows per warp) was measured and refuted; the cause is not identified.
# The scatter table holds the measured best (TPL 4, GS 1; TPL 2 at tier 12), and the sampler's
# `impl="auto"` (emitter_triton.emitter_step_weights, AUTO_LANES) sends scatter-regime steps to
# the seq kernel instead -- the CUDA kernel is the frontier-step kernel.
_TIERS_LADDER = tuple(range(8, 41, 4))
SCATTER_LANES = 4                     # lanes per term below which the "scatter" table is used
_CFG_DEFAULT = {k: ((2 if k <= 16 else 4), 1, None) for k in _TIERS_LADDER}
_CFG_SCATTER_DEFAULT = {k: ((2 if k <= 12 else 4), 1, None) for k in _TIERS_LADDER}
_H100_SEG = {**_CFG_DEFAULT, 16: (2, 0, None), 20: (4, 1, None), 24: (4, 1, None), 28: (4, 1, None),
             32: (4, 1, None), 36: (4, 1, None), 40: (4, 1, None)}      # H100 PCIe, 2026-09-27
_CFG_BY_ARCH = {
    (9, 0): {"seg": _H100_SEG, "scatter": _CFG_SCATTER_DEFAULT},
}
_CFG = None
_MODULES = {}
_KERNELS = {}


def _cfg(regime="seg"):
    """(tier -> (TPL, GS, maxrregcount)) of a regime for the current device."""
    global _CFG
    if _CFG is None:
        arch = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        base = _CFG_BY_ARCH.get(arch, {"seg": _CFG_DEFAULT, "scatter": _CFG_SCATTER_DEFAULT})
        _CFG = {"seg": dict(base["seg"]), "scatter": dict(base["scatter"])}
    return _CFG[regime]


def set_config(cfg, regime=None):
    """Override the (tier -> (TPL, GS, maxrregcount)) table of one regime ("seg" | "scatter"; None =
    both) for this process (merged); cfg=None resets everything.
    """
    global _CFG, _MODULES, _KERNELS
    if cfg is None:
        _CFG = None
    else:
        _cfg()
        for r in ([regime] if regime else ("seg", "scatter")):
            _CFG[r] = {**_CFG[r], **{int(k): tuple(v) for k, v in cfg.items()}}
    _MODULES, _KERNELS = {}, {}


# Production since 2026-10-05: w folded + shared accumulators -- runs 20261005_213240 / _221051 / _223132 (exact on the
# device to K = 36, frozen-spin limit 0.86-1.11 per needed op, cluster 1.02-1.43); PE_SYM measured 2-6x slower (register
# spills at 18 columns per thread, run 20261005_225737). Env PE_EMIT_VARIANT overrides ("none" = the kernel of 09-27).
_VARIANT = {"wq": 0, "msh": 0, "accsh": 0, "sym": 0}
_VARIANT.update({k: 1 for k in os.environ.get("PE_EMIT_VARIANT", "wq,accsh").split(",") if k in _VARIANT})
_DEFAULT_VARIANT = dict(_VARIANT)


def set_variant(wq=None, msh=None, accsh=None, sym=None):
    """Compile-time variants of the kernel (module source, PE_WQ / PE_MSH / PE_ACCSH / PE_SYM; all 0 = the kernel of 09-27) for
    this process; recompiles on next use. No argument restores the default (wq + accsh unless PE_EMIT_VARIANT); given
    flags are set, the others cleared. accsh launches 64- or 32-thread blocks; sym applies to TPL = 2 tiers only.
    """
    global _MODULES, _KERNELS
    flags = {"wq": wq, "msh": msh, "accsh": accsh, "sym": sym}
    if all(v is None for v in flags.values()):
        _VARIANT.update(_DEFAULT_VARIANT)
    else:
        _VARIANT.update({k: int(v or 0) for k, v in flags.items()})
    _MODULES, _KERNELS = {}, {}


def _opts(cap):
    defs = tuple(f"-DPE_{k.upper()}={v}" for k, v in _VARIANT.items() if v)
    return ("-std=c++17",) + defs + (() if cap is None else (f"--maxrregcount={cap}",))


def regime_of(lanes_per_term):
    """"scatter" below SCATTER_LANES lanes per term, "seg" otherwise (None -> "seg")."""
    return "scatter" if lanes_per_term is not None and lanes_per_term < SCATTER_LANES else "seg"


def tier(K, lanes_per_term=None):
    """(padded column count, threads per lane, GS) of the tier serving K columns in the regime
    of `lanes_per_term` (mean lanes per term of the launch; None = the segmented regime).
    """
    KP = next(k for k in _TIERS_LADDER if k >= K)
    t = _cfg(regime_of(lanes_per_term))[KP]
    return KP, t[0], t[1]


def warmup(NL=(1, 2), tiers=None):
    """Compile every (register cap) module up front, both regimes, as `cuda_cofactors.warmup` does (tiers: only these,
    for benches that re-compile per variant).
    """
    import cupy
    entries = {(k, r): _cfg(r)[k] for r in ("seg", "scatter") for k in (tiers or _TIERS_LADDER)}
    for cap in dict.fromkeys(v[2] for v in entries.values()):
        if cap not in _MODULES:
            params = sorted({(k, v[0], nl, v[1]) for (k, _r), v in entries.items() if v[2] == cap for nl in NL})
            if _PLAIN:            # plain-named wrappers, no name_expressions: cupy's disk cache is read across processes
                lines = [f"PE_EMIT_ENTRY({k}, {t}, {nl}, {gs})" for k, t, nl, gs in params]
                code = _SRC + "\n#ifdef __CUDACC__\n" + "\n".join(lines) + "\n#endif\n"
                _MODULES[cap] = cupy.RawModule(code=code, options=_opts(cap))
            else:
                _MODULES[cap] = cupy.RawModule(code=_SRC, options=_opts(cap),
                                               name_expressions=[_kname(*p) for p in params])


# plain-named extern "C" entry points (10-06); ON by default since run 20261006_115902 (job 33: device check ok at every
# tier, draws identical, fresh-process warm-up 0.0 s); PE_PLAIN_ENTRY=0 restores the name_expressions path
_PLAIN = os.environ.get("PE_PLAIN_ENTRY", "1") == "1"


def _kname(KP, TPL, NL, GS):
    return f"emit_k_{KP}_{TPL}_{NL}_{GS}" if _PLAIN else f"emit_multi<{KP}, {TPL}, {NL}, {GS}>"


def _get_kernel(KP, NL, regime="seg"):
    cfg = _cfg(regime)
    key = (KP, NL, regime)
    if key not in _KERNELS:
        if not _MODULES:
            warmup()
        _KERNELS[key] = _MODULES[cfg[KP][2]].get_function(_kname(KP, cfg[KP][0], NL, cfg[KP][1]))
    return _KERNELS[key]


def _cp(t):
    import cupy
    return cupy.asarray(t)


def emitter_tables(rows, K1):
    """Fock-style 2020 tables for the emitter kernel (every distinct row is a digit of radix
    multiplicity + 1, one odd group last and halved): drow [S, K1] int64, mult / radix [S, K1]
    int32, N [S] int64, scale [S] float64.  K1 = 0 (first photon): one term, scale 1.
    """
    if K1 == 0:
        S = rows.shape[0]
        z = torch.zeros((S, 1), dtype=torch.long, device=rows.device)
        return (z, z.to(torch.int32), (z + 1).to(torch.int32), torch.ones(S, dtype=torch.long, device=rows.device),
                torch.ones(S, dtype=torch.float64, device=rows.device))
    from .cofactors_triton import multiplicity_tables
    return multiplicity_tables(rows, K1)


def gather_block(U2, drow, colmode, K, NL):
    """Bs [S, K1, K, NL] complex128 (as float64 pairs, contiguous): U[drow[s, nu], colmode[s, c, l]]."""
    U = torch.view_as_complex(U2)
    Bs = U[drow[:, :, None, None], colmode[:, None, :K, :NL]]
    return torch.view_as_real(Bs.contiguous())


def _args(U2, tables, colmode, lane2term, term2stream, lstart, mats2, s02, f2, K, NL):
    drow, mult, radix, N, _scale = tables
    return (gather_block(U2, drow, colmode, K, NL), mult.contiguous(), radix.contiguous(), lane2term.contiguous(),
            term2stream.contiguous(), lstart.contiguous(), N.contiguous(), mats2.contiguous(), s02.contiguous(),
            f2.contiguous())


def emitter_cofactor_partials_cuda(U2, tables, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
                                   K1, K, NL, L, total, block=128):
    """Same contract as `emitter_triton.emitter_cofactor_partials_seq` (plus K1, the drawn rows)
    with the tables of `emitter_tables`: returns part [TOTAL, K, 2, 2, 2] float64.
    """
    import cupy
    dev = U2.device
    Bs2, mult, radix, l2t, t2s, lst, N, mats2, s02, f2 = _args(U2, tables, colmode, lane2term, term2stream,
                                                                 lstart, mats2, s02, f2, K, NL)
    part = torch.empty(total, K, 2, 2, 2, dtype=torch.float64, device=dev)
    lanes_per_term = total / max(1, lst.shape[0])
    KP, TPL, _gs = tier(K, lanes_per_term)
    if _VARIANT["accsh"]:                            # the shared accumulators are laid out for this block size
        block = 64 if -(-KP // TPL) <= 10 else 32    # = PE_ACCSH_B(KH) of the kernel source
    kern = _get_kernel(KP, NL, regime_of(lanes_per_term))
    nthreads = total * TPL
    grid = ((nthreads + block - 1) // block,)
    from .cofactors_triton import _binom_table
    with cupy.cuda.Device(dev.index or 0):
        kern(grid, (block,), (
            _cp(Bs2), _cp(mult), _cp(radix), _cp(l2t), _cp(t2s), _cp(lst), _cp(N), _cp(_binom_table(dev)),
            _cp(mats2), _cp(s02), _cp(f2), _cp(part),
            cupy.int32(K1), cupy.int32(K), cupy.int64(mats2.stride(0) // 2), cupy.int64(mats2.stride(1) // 2),
            cupy.int64(L), cupy.int64(total)))
    return part


# ---- CPU emulation (host C++ compiler) ----
_EMUL = {}


def _cache_dir():
    d = os.environ.get("PE_KERNEL_CACHE") or os.path.join(tempfile.gettempdir(), "photonic_engines_kernels")
    os.makedirs(d, exist_ok=True)
    return d


def emul_available():
    """True when a host C++ compiler is found (the emulation needs C++20 std::barrier)."""
    try:
        _emul_lib(8, 1, 1, 0)
        return True
    except Exception:
        return False


def _emul_lib(KP, TPL, NL, GS, real="double"):
    """Compiled emulation library for one configuration (cached on disk by source hash); real = "double"
    or "long double" (the arithmetic type of the sums and chains, inputs and outputs stay double).
    """
    wq, sym, accsh = _VARIANT["wq"], _VARIANT["sym"], _VARIANT["accsh"]   # the variants the CPU emulation has (not msh)
    key = (KP, TPL, NL, GS, real, wq, sym, accsh)
    if key in _EMUL:
        return _EMUL[key]
    h = hashlib.sha1(_SRC.encode()).hexdigest()[:10]
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    tag = ("" if real == "double" else "_ld") + ("_wq" if wq else "") + ("_sym" if sym else "") + ("_as" if accsh else "")
    path = os.path.join(_cache_dir(), f"pe_emit_emul_k{KP}_t{TPL}_l{NL}_g{GS}{tag}_{h}{ext}")
    if not os.path.exists(path):
        src = path + ".cpp"
        with open(src, "w") as fh:
            fh.write(_SRC)
        cxx = os.environ.get("CXX") or ("clang++" if sys.platform == "darwin" else "g++")
        cmd = [cxx, "-std=c++20", "-O2", "-fPIC", "-shared", f"-DPE_KP={KP}", f"-DPE_TPL={TPL}", f"-DPE_NL={NL}",
               f"-DPE_GS={GS}", f"-DPE_REAL={real}", f"-DPE_WQ={wq}", f"-DPE_SYM={sym}", f"-DPE_ACCSH={accsh}", "-o", path + ".tmp", src]
        if sys.platform != "darwin":
            cmd.insert(-3, "-pthread")
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(path + ".tmp", path)
    lib = ctypes.CDLL(path)
    P = ctypes.c_void_p
    lib.emit_emul.argtypes = [P] * 12 + [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                                        ctypes.c_longlong, ctypes.c_longlong]
    lib.emit_emul.restype = None
    _EMUL[key] = lib
    return lib


def emitter_cofactor_partials_emul(U2, tables, colmode, lane2term, term2stream, lstart, mats2, s02, f2,
                                   K1, K, NL, L, total, tpl=None, gs=None, real="double"):
    """CPU run of the kernel code (same contract as `emitter_cofactor_partials_cuda`; CPU tensors).
    tpl / gs override the tier's threads per lane and matrix storage for the tests; real = "long double"
    runs the sums and chains in x87 extended precision (a reference for the fp64 rounding error).
    """
    Bs2, mult, radix, l2t, t2s, lst, N, mats2, s02, f2 = _args(U2, tables, colmode, lane2term, term2stream,
                                                                 lstart, mats2, s02, f2, K, NL)
    KP, TPL, GS = tier(K, total / max(1, lst.shape[0]))
    TPL = TPL if tpl is None else int(tpl)
    GS = GS if gs is None else int(gs)
    from .cofactors_triton import _binom_table
    binom = _binom_table(torch.device("cpu"))
    part = torch.zeros(total, K, 2, 2, 2, dtype=torch.float64)
    lib = _emul_lib(KP, TPL, NL, GS, real)
    def ptr(t):
        return ctypes.c_void_p(t.data_ptr())
    lib.emit_emul(ptr(Bs2), ptr(mult), ptr(radix), ptr(l2t), ptr(t2s), ptr(lst), ptr(N), ptr(binom), ptr(mats2),
                  ptr(s02), ptr(f2), ptr(part), K1, K, mats2.stride(0) // 2, mats2.stride(1) // 2, L, total)
    return part


def available():
    """True when the GPU path can run: `cupy` importable and a CUDA device visible to torch."""
    try:
        import cupy  # noqa
        return torch.cuda.is_available()
    except Exception:
        return False
