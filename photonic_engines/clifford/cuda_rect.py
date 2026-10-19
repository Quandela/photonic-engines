"""
CUDA (NVRTC via cupy) thread-per-lane kernel for RECTANGULAR leave-one-out permanent cofactors — the
step of the random-source-order SBS chain (`clifford/sbs_rso.py`): the superposed-source sampler on the same
cofactor substrate as the Fock sampler.

Step j of the chain: q = j - 1 placed rows, N = n - k + j active columns (N > q), and for every column x

    P_x = Per_rect(B[:, all columns but x])
        = 2^-(q-1) sum_delta (prod delta) e_q(a(delta) without a_x),   a_i(delta) = sum_t delta_t B[t, i],

e_q the elementary symmetric polynomial of degree q.  This is `cuda_cofactors.cof_multi` with the payload
generalised: the Fock kernel is the case N = q + 1, where e_q(a without a_x) is the all-but-one PRODUCT;
here it is the degree-q coefficient of F(lambda) / (1 + lambda a_x), F(lambda) = prod_i (1 + lambda a_i)
the full product polynomial of the N column sums.  Everything else is the Fock kernel's: the mixed-radix
reflected Gray walk over the Clifford & Clifford 2020 collapse of repeated placed rows (`multiplicity_tables`),
the byte odometer, the lane -> (stream, segment) planning (`plan_multi`), the gathered block Bs[s, nu, i] =
A[s, drow[s, nu], i], the tier ladder and the (threads per lane, register cap) table.

Per Gray step and lane: the column sums move by one row (O(N)); each of the TPL threads forms the partial
polynomial of its NH = NP / TPL columns (O(NH^2)); the threads combine them by polynomial multiplication
over `__shfl_xor_sync` (log2 TPL rounds); then for each of its columns the thread divides F by
(1 + lambda a_x) -- forward from degree 0 where |a_x| <= 1, backward from degree N where |a_x| > 1, the
mixed form of `sbs_rso._loo_divide` (forward alone loses everything at large q; the mixed form stays at
rounding level while q is well below N -- at q ~ N the forward branch itself loses ~10 digits, see `choose_form`)
-- and accumulates w * e_q(a without a_x).  O(N^2) per term-lane; the torch path
(`sbs_rso.rect_cofactors`) materialises [S, D, N] column sums per sign chunk and recomputes them from
scratch.

Output: part [TOTAL, N] complex (per-lane partial sums); the host reduces per stream and applies the
2020 scale, exactly as `cofactors_triton.cofactor_partials_multi`.  As for the emitter kernel the source
compiles under NVRTC (`rect_multi<NP, TPL>`) and under the host C++ compiler as a CPU emulation
(`rect_cofactors_emul`, std::thread + std::barrier per lane; -DPE_REAL="long double" for a higher-precision
reference), which is what the CPU test-suite runs.

Three more payloads serve the continuity region s0 = N - 1 - q small (2026-10-05): the production
Fock kernel at s0 = 0 ("fock"), and the division-free mu-form e_q(a without a_x) = [mu^s0] prod_{i != x} (mu + a_i) --
`mu_lane` (runtime degree, the reference: "mu_generic") and `mud_lane` (templated on D = s0 = 0 .. 16, "mu": exclusive
prefixes in registers, one xor butterfly for the other threads' columns, 3 (s0 + 1) complex products per column and
term; its own tier ladder `mu_tier`). Dispatch: `choose_form`.

Pseudo-code of `mud_lane`, per Gray term (column sums a_j already moved by one row):
    G = 1;  for own column j:  P[j] = G;  G = G (mu + a_j)            (truncated at degree D)
    O = 1;  for o = 1, 2, .. < TPL:  X = xor_shfl(G, o);  O = O X;  G = G X     (O = the other threads' columns)
    S = O;  for own column j, last to first:  acc[j] += w [mu^D] (P[j] S);  S = S (mu + a_j)
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
#ifdef __CUDACC__
typedef double real_t;
#else
#ifndef PE_REAL
#define PE_REAL double
#endif
typedef PE_REAL real_t;
#endif

// exchange of a coefficient array with the partner thread h ^ o of the lane
#ifdef __CUDACC__
template <int TPL> struct Xchg {
    unsigned grp;
    __device__ void xor_(const real_t* mine, real_t* theirs, int len, int o) const {
        for (int d = 0; d < len; ++d) theirs[d] = __shfl_xor_sync(grp, mine[d], o, TPL);
    }
    __device__ void up_(const real_t* mine, real_t* theirs, int len, int o) const {     // from thread h - o (own if h < o)
        for (int d = 0; d < len; ++d) theirs[d] = __shfl_up_sync(grp, mine[d], o, TPL);
    }
    __device__ void down_(const real_t* mine, real_t* theirs, int len, int o) const {   // from thread h + o (own if h + o >= TPL)
        for (int d = 0; d < len; ++d) theirs[d] = __shfl_down_sync(grp, mine[d], o, TPL);
    }
    template <int LEN> __device__ __forceinline__ void xorc(const real_t* mine, real_t* theirs, int o) const {
        #pragma unroll
        for (int d = 0; d < LEN; ++d) theirs[d] = __shfl_xor_sync(grp, mine[d], o, TPL);
    }
};
#else
template <int TPL> struct Xchg {
    const real_t** slots; int* lens; std::barrier<>* bar; int h;
    void xor_(const real_t* mine, real_t* theirs, int len, int o) const {
        slots[h] = mine; lens[h] = len; bar->arrive_and_wait();
        const real_t* p = slots[h ^ o];
        for (int d = 0; d < len; ++d) theirs[d] = p[d];
        bar->arrive_and_wait();
    }
    void up_(const real_t* mine, real_t* theirs, int len, int o) const {
        slots[h] = mine; lens[h] = len; bar->arrive_and_wait();
        const real_t* p = (h >= o) ? slots[h - o] : mine;
        for (int d = 0; d < len; ++d) theirs[d] = p[d];
        bar->arrive_and_wait();
    }
    void down_(const real_t* mine, real_t* theirs, int len, int o) const {
        slots[h] = mine; lens[h] = len; bar->arrive_and_wait();
        const real_t* p = (h + o < TPL) ? slots[h + o] : mine;
        for (int d = 0; d < len; ++d) theirs[d] = p[d];
        bar->arrive_and_wait();
    }
    template <int LEN> void xorc(const real_t* mine, real_t* theirs, int o) const { xor_(mine, theirs, LEN, o); }
};
#endif

// NP = padded column count of the tier, TPL = threads per lane (1, 2, 4, 8).  Thread h owns columns
// c0 = h * NH .. c0 + NH - 1 (NH = NP / TPL); columns >= N idle.  Polynomials are stored as separate
// real / imaginary coefficient arrays so that the exchange is a plain shuffle of doubles.
template <int NP, int TPL>
PE_FN void rect_lane(const double2* __restrict__ Bs,       // [S, Q1, N]  (Q1 = q, the placed rows; distinct, by drow)
                     const int*  __restrict__ mult,        // [S, Q1]
                     const int*  __restrict__ radix,       // [S, Q1]
                     const long long* __restrict__ lane2s, // [TOTAL]
                     const long long* __restrict__ lstart, // [S]
                     const long long* __restrict__ nterms, // [S]
                     const double* __restrict__ binom,     // [32*32]
                     double2* __restrict__ part,           // [TOTAL, N]
                     int Q1, int N, long long L, long long g, int h, const Xchg<TPL>& xc)
{
    constexpr int NH = NP / TPL;
    const int c0 = h * NH;
    const long long s = lane2s[g];
    const long long li = g - lstart[s];
    const long long Ns = nterms[s];
    double2* out = part + g * N;
    const long long base = li * L;
    if (base >= Ns) { for (int j = 0; j < NH; ++j) if (c0 + j < N) out[c0 + j] = make_double2(0.0, 0.0); return; }
    const double2* B = Bs + s * (long long)Q1 * N;
    const int* Ms = mult + s * Q1;
    const int* Rs = radix + s * Q1;

    unsigned char d[NP], v[NP], R[NP], M[NP], pp[NP];
    for (int nu = 0; nu < Q1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < Q1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = Q1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1);
                                             v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    real_t w = 1.0;
    for (int nu = 0; nu < Q1; ++nu) { real_t b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    real_t sr[NH], si[NH], ar[NH], ai[NH];
    #pragma unroll
    for (int j = 0; j < NH; ++j) { sr[j] = 0.0; si[j] = 0.0; ar[j] = 0.0; ai[j] = 0.0; }
    for (int nu = 0; nu < Q1; ++nu) {
        const real_t c = (real_t)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + (long long)nu * N;
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) { double2 b = row[c0 + j]; sr[j] += c * b.x; si[j] += c * b.y; }
    }
    // F = prod_i (1 + lambda a_i), degree N, as fr/fi[0..N]; gr/gi: the partner's polynomial
    real_t fr[NP + 1], fi[NP + 1], gr[NP + 1], gi[NP + 1], hr[NP + 1], hi[NP + 1];
    const int q = Q1;

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
            const double2* row = B + (long long)nu * N;
            #pragma unroll
            for (int j = 0; j < NH; ++j) if (c0 + j < N) { double2 b = row[c0 + j]; sr[j] += f * b.x; si[j] += f * b.y; }
        }
        // partial polynomial over this thread's columns
        int deg = 0; fr[0] = 1.0; fi[0] = 0.0;
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) {
            fr[deg + 1] = 0.0; fi[deg + 1] = 0.0;
            for (int e = deg; e >= 0; --e) {                           // F <- F (1 + lambda a): F[e+1] += a F[e]
                fr[e + 1] += sr[j] * fr[e] - si[j] * fi[e];
                fi[e + 1] += sr[j] * fi[e] + si[j] * fr[e];
            }
            ++deg;
        }
        // combine across the lane's threads: TPL - 1 polynomial products, degrees add up to N
        #pragma unroll
        for (int o = 1; o < TPL; o <<= 1) {
            // the partner's degree: columns of the (o-block) it belongs to, bounded by N
            const int blk = (h ^ o) / o * o;                                   // first thread of the partner's o-block
            int pdeg = (int)((long long)(blk + o) * NH < N ? (long long)o * NH : N - (long long)blk * NH);
            if (pdeg < 0) pdeg = 0;
            const int len = (pdeg > deg ? pdeg : deg) + 1;
            xc.xor_(fr, gr, len, o); xc.xor_(fi, gi, len, o);
            for (int e = 0; e <= deg + pdeg; ++e) { hr[e] = 0.0; hi[e] = 0.0; }
            for (int e = 0; e <= deg; ++e) for (int k = 0; k <= pdeg; ++k) {
                hr[e + k] += fr[e] * gr[k] - fi[e] * gi[k];
                hi[e + k] += fr[e] * gi[k] + fi[e] * gr[k];
            }
            deg += pdeg;
            for (int e = 0; e <= deg; ++e) { fr[e] = hr[e]; fi[e] = hi[e]; }
        }
        // e_q(a without a_x) = [lambda^q] F / (1 + lambda a_x), mixed forward / backward division
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) {
            const real_t xr = sr[j], xi = si[j];
            real_t cr, ci;
            if (xr * xr + xi * xi <= 1.0) {
                cr = fr[0]; ci = fi[0];
                for (int dd = 1; dd <= q; ++dd) { const real_t nr = fr[dd] - (xr * cr - xi * ci);
                                                  ci = fi[dd] - (xr * ci + xi * cr); cr = nr; }
            } else {
                const real_t inv = 1.0 / (xr * xr + xi * xi);
                const real_t yr = xr * inv, yi = -xi * inv;                    // 1 / a_x
                cr = fr[N] * yr - fi[N] * yi; ci = fr[N] * yi + fi[N] * yr;    // c_(N-1)
                for (int dd = N - 1; dd > q; --dd) { const real_t tr = fr[dd] - cr, ti = fi[dd] - ci;
                                                     cr = tr * yr - ti * yi; ci = tr * yi + ti * yr; }
            }
            ar[j] += w * cr; ai[j] += w * ci;
        }
    }
    #pragma unroll
    for (int j = 0; j < NH; ++j) if (c0 + j < N) out[c0 + j] = make_double2(ar[j], ai[j]);
}


// ---- the mu-form payload (continuity region, small s0): e_q(a without a_x) = [mu^D] prod_{i != x}
// (mu + a_i) with D = s0 = N - 1 - q, truncated polynomials of degree D <= PE_DMAX, DIVISION-FREE: within a thread an
// exclusive prefix array over its own columns and a running suffix; across the lane's threads a Hillis-Steele
// exclusive prefix (shfl_up) and suffix (shfl_down) of the block polynomials; the degree-D coefficient of the
// four-way product is one dot product.  O(NH D^2) per thread and term; D = 0 is the Fock kernel's all-but-one
// product.  The lambda-form of rect_lane costs O(N^2 / TPL + N q); this one wins when s0 is small.
#define PE_DMAX 6
#define PE_DM1 (PE_DMAX + 1)

// c <- a * b truncated at degree D (complex coefficient arrays)
PE_FN void pmul(const real_t* ar, const real_t* ai, const real_t* br, const real_t* bi, real_t* cr, real_t* ci, int D) {
    for (int e = 0; e <= D; ++e) { cr[e] = 0.0; ci[e] = 0.0; }
    for (int j = 0; j <= D; ++j) for (int k = 0; k + j <= D; ++k) {
        cr[j + k] += ar[j] * br[k] - ai[j] * bi[k];
        ci[j + k] += ar[j] * bi[k] + ai[j] * br[k];
    }
}
// a <- a * (mu + x) truncated at degree D
PE_FN void pmul_lin(real_t* ar, real_t* ai, real_t xr, real_t xi, int D) {
    for (int e = D; e >= 1; --e) {
        const real_t nr = ar[e - 1] + (xr * ar[e] - xi * ai[e]);
        ai[e] = ai[e - 1] + (xr * ai[e] + xi * ar[e]); ar[e] = nr;
    }
    const real_t nr = xr * ar[0] - xi * ai[0]; ai[0] = xr * ai[0] + xi * ar[0]; ar[0] = nr;
}
PE_FN void pone(real_t* ar, real_t* ai, int D) { ar[0] = 1.0; ai[0] = 0.0; for (int e = 1; e <= D; ++e) { ar[e] = 0.0; ai[e] = 0.0; } }

template <int NP, int TPL>
PE_FN void mu_lane(const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix,
                   const long long* __restrict__ lane2s, const long long* __restrict__ lstart,
                   const long long* __restrict__ nterms, const double* __restrict__ binom, double2* __restrict__ part,
                   int Q1, int N, int D, long long L, long long g, int h, const Xchg<TPL>& xc)
{
    constexpr int NH = NP / TPL;
    const int c0 = h * NH;
    const long long s = lane2s[g];
    const long long li = g - lstart[s];
    const long long Ns = nterms[s];
    double2* out = part + g * N;
    const long long base = li * L;
    if (base >= Ns) { for (int j = 0; j < NH; ++j) if (c0 + j < N) out[c0 + j] = make_double2(0.0, 0.0); return; }
    const double2* B = Bs + s * (long long)Q1 * N;
    const int* Ms = mult + s * Q1;
    const int* Rs = radix + s * Q1;

    unsigned char d[NP], v[NP], R[NP], M[NP], pp[NP];
    for (int nu = 0; nu < Q1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < Q1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = Q1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1);
                                             v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    real_t w = 1.0;
    for (int nu = 0; nu < Q1; ++nu) { real_t b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    real_t sr[NH], si[NH], ar[NH], ai[NH];
    #pragma unroll
    for (int j = 0; j < NH; ++j) { sr[j] = 0.0; si[j] = 0.0; ar[j] = 0.0; ai[j] = 0.0; }
    for (int nu = 0; nu < Q1; ++nu) {
        const real_t c = (real_t)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + (long long)nu * N;
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) { double2 b = row[c0 + j]; sr[j] += c * b.x; si[j] += c * b.y; }
    }
    real_t prr[NH][PE_DM1], pri[NH][PE_DM1];                  // exclusive prefix products within the block
    real_t gr[PE_DM1], gi[PE_DM1], tr[PE_DM1], ti[PE_DM1], ur[PE_DM1], ui[PE_DM1], lor[PE_DM1], loi[PE_DM1];

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
            const double2* row = B + (long long)nu * N;
            #pragma unroll
            for (int j = 0; j < NH; ++j) if (c0 + j < N) { double2 b = row[c0 + j]; sr[j] += f * b.x; si[j] += f * b.y; }
        }
        // block polynomial G_h = prod_own (mu + a_j), with the exclusive prefixes stored
        pone(gr, gi, D);
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) {
            for (int e = 0; e <= D; ++e) { prr[j][e] = gr[e]; pri[j][e] = gi[e]; }
            pmul_lin(gr, gi, sr[j], si[j], D);
        }
        // exclusive prefix over the lane's threads (Hillis-Steele inclusive scan, then shift by one)
        for (int e = 0; e <= D; ++e) { tr[e] = gr[e]; ti[e] = gi[e]; }
        #pragma unroll
        for (int o = 1; o < TPL; o <<= 1) {
            xc.up_(tr, ur, D + 1, o); xc.up_(ti, ui, D + 1, o);
            if (h >= o) { pmul(ur, ui, tr, ti, lor, loi, D); for (int e = 0; e <= D; ++e) { tr[e] = lor[e]; ti[e] = loi[e]; } }
        }
        xc.up_(tr, lor, D + 1, 1); xc.up_(ti, loi, D + 1, 1);
        if (h == 0) pone(lor, loi, D);
        // exclusive suffix over the lane's threads, then Out = Lo * Hi
        for (int e = 0; e <= D; ++e) { tr[e] = gr[e]; ti[e] = gi[e]; }
        #pragma unroll
        for (int o = 1; o < TPL; o <<= 1) {
            xc.down_(tr, ur, D + 1, o); xc.down_(ti, ui, D + 1, o);
            if (h + o < TPL) { pmul(tr, ti, ur, ui, gr, gi, D); for (int e = 0; e <= D; ++e) { tr[e] = gr[e]; ti[e] = gi[e]; } }
        }
        xc.down_(tr, ur, D + 1, 1); xc.down_(ti, ui, D + 1, 1);
        if (h + 1 >= TPL) pone(ur, ui, D);
        pmul(lor, loi, ur, ui, gr, gi, D);                                             // gr/gi = Out_h
        // own columns, last to first, running exclusive suffix in tr/ti
        pone(tr, ti, D);
        #pragma unroll
        for (int j = NH - 1; j >= 0; --j) if (c0 + j < N) {
            pmul(prr[j], pri[j], tr, ti, ur, ui, D);                                    // all other own columns
            real_t er = 0.0, ei = 0.0;
            for (int e = 0; e <= D; ++e) { er += ur[e] * gr[D - e] - ui[e] * gi[D - e]; ei += ur[e] * gi[D - e] + ui[e] * gr[D - e]; }
            ar[j] += w * er; ai[j] += w * ei;
            pmul_lin(tr, ti, sr[j], si[j], D);
        }
    }
    #pragma unroll
    for (int j = 0; j < NH; ++j) if (c0 + j < N) out[c0 + j] = make_double2(ar[j], ai[j]);
}

// ---- the mu lane templated on the degree D = s0 (2026-10-05).  Same mathematics as mu_lane --
// e_q(a without a_x) = [mu^D] prod_{i != x} (mu + a_i), division-free -- with the Fock kernel's loop structure: D is a
// template parameter (every coefficient loop unrolled, the per-column prefix held in (D + 1)-long register arrays
// instead of mu_lane's [NH][PE_DMAX + 1] block), and the lane's other threads enter through ONE xor butterfly of the
// block polynomials (the product O_h of every other thread's block, log2 TPL exchanges of D + 1 coefficients) instead
// of the two Hillis-Steele scans.  Per term and column: one truncated linear product forward (the exclusive prefix
// P_<x, stored), one backward (the running suffix S_>x, seeded with O_h), and the degree-D dot of the two: 3 (D + 1)
// complex products, against the Fock kernel's one product and one reciprocal.  D = 0 is the Fock all-but-one product
// computed without division (kept for the tests).  NH = ceil(NP / TPL) columns per thread (any TPL legal).
template <int D>
PE_FN void tlin(real_t* r, real_t* i, real_t xr, real_t xi) {         // (r, i) <- (r, i) * (mu + x), truncated at D
    #pragma unroll
    for (int e = D; e >= 1; --e) {
        const real_t nr = r[e - 1] + (xr * r[e] - xi * i[e]);
        i[e] = i[e - 1] + (xr * i[e] + xi * r[e]); r[e] = nr;
    }
    const real_t nr = xr * r[0] - xi * i[0]; i[0] = xr * i[0] + xi * r[0]; r[0] = nr;
}
template <int D>
PE_FN void tmul(real_t* ar, real_t* ai, const real_t* br, const real_t* bi) {   // a <- a * b truncated at D, in place
    #pragma unroll
    for (int e = D; e >= 0; --e) {                     // descending: a[0..e] are still the old coefficients
        real_t cr = 0.0, ci = 0.0;
        #pragma unroll
        for (int k = 0; k <= e; ++k) { cr += ar[k] * br[e - k] - ai[k] * bi[e - k]; ci += ar[k] * bi[e - k] + ai[k] * br[e - k]; }
        ar[e] = cr; ai[e] = ci;
    }
}

template <int NP, int TPL, int D>
PE_FN void mud_lane(const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix,
                    const long long* __restrict__ lane2s, const long long* __restrict__ lstart,
                    const long long* __restrict__ nterms, const double* __restrict__ binom, double2* __restrict__ part,
                    int Q1, int N, long long L, long long g, int h, const Xchg<TPL>& xc)
{
    constexpr int NH = (NP + TPL - 1) / TPL;
    constexpr int D1 = D + 1;
    const int c0 = h * NH;
    const long long s = lane2s[g];
    const long long li = g - lstart[s];
    const long long Ns = nterms[s];
    double2* out = part + g * N;
    const long long base = li * L;
    if (base >= Ns) { for (int j = 0; j < NH; ++j) if (c0 + j < N) out[c0 + j] = make_double2(0.0, 0.0); return; }
    const double2* B = Bs + s * (long long)Q1 * N;
    const int* Ms = mult + s * Q1;
    const int* Rs = radix + s * Q1;

    unsigned char d[NP], v[NP], R[NP], M[NP], pp[NP];
    for (int nu = 0; nu < Q1; ++nu) { R[nu] = (unsigned char)Rs[nu]; M[nu] = (unsigned char)Ms[nu]; }
    { long long rem = base; for (int nu = 0; nu < Q1; ++nu) { d[nu] = (unsigned char)(rem % R[nu]); rem /= R[nu]; } }
    { int par = 0;
      for (int nu = Q1 - 1; nu >= 0; --nu) { pp[nu] = (unsigned char)(par & 1);
                                             v[nu] = (unsigned char)((par & 1) ? R[nu] - 1 - d[nu] : d[nu]); par += v[nu]; } }
    real_t w = 1.0;
    for (int nu = 0; nu < Q1; ++nu) { real_t b = binom[(int)M[nu] * 32 + (int)v[nu]]; w *= (v[nu] & 1) ? -b : b; }

    real_t sr[NH], si[NH], ar[NH], ai[NH];
    #pragma unroll
    for (int j = 0; j < NH; ++j) { sr[j] = 0.0; si[j] = 0.0; ar[j] = 0.0; ai[j] = 0.0; }
    for (int nu = 0; nu < Q1; ++nu) {
        const real_t c = (real_t)((int)M[nu] - 2 * (int)v[nu]);
        const double2* row = B + (long long)nu * N;
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) { double2 b = row[c0 + j]; sr[j] += c * b.x; si[j] += c * b.y; }
    }
    real_t pr[NH][D1], pi[NH][D1];                                   // exclusive prefixes P_<x over the own columns
    real_t gr[D1], gi[D1], orr[D1], oi[D1], xr[D1], xi[D1];

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
            const double2* row = B + (long long)nu * N;
            #pragma unroll
            for (int j = 0; j < NH; ++j) if (c0 + j < N) { double2 b = row[c0 + j]; sr[j] += f * b.x; si[j] += f * b.y; }
        }
        // forward: P_<x stored, G_h = the block polynomial
        gr[0] = 1.0; gi[0] = 0.0;
        #pragma unroll
        for (int e = 1; e < D1; ++e) { gr[e] = 0.0; gi[e] = 0.0; }
        #pragma unroll
        for (int j = 0; j < NH; ++j) if (c0 + j < N) {
            #pragma unroll
            for (int e = 0; e < D1; ++e) { pr[j][e] = gr[e]; pi[j][e] = gi[e]; }
            tlin<D>(gr, gi, sr[j], si[j]);
        }
        // O_h = product of the other threads' blocks: xor butterfly (the partner's o-block is disjoint from ours)
        orr[0] = 1.0; oi[0] = 0.0;
        #pragma unroll
        for (int e = 1; e < D1; ++e) { orr[e] = 0.0; oi[e] = 0.0; }
        #pragma unroll
        for (int o = 1; o < TPL; o <<= 1) {
            xc.template xorc<D1>(gr, xr, o); xc.template xorc<D1>(gi, xi, o);
            tmul<D>(orr, oi, xr, xi);
            if (2 * o < TPL) tmul<D>(gr, gi, xr, xi);
        }
        // backward: the running suffix S_>x (seeded with O_h); e_q(a without a_x) = [mu^D] P_<x S_>x
        #pragma unroll
        for (int j = NH - 1; j >= 0; --j) if (c0 + j < N) {
            real_t er = 0.0, ei = 0.0;
            #pragma unroll
            for (int e = 0; e < D1; ++e) { er += pr[j][e] * orr[D - e] - pi[j][e] * oi[D - e];
                                           ei += pr[j][e] * oi[D - e] + pi[j][e] * orr[D - e]; }
            ar[j] += w * er; ai[j] += w * ei;
            if (j > 0) tlin<D>(orr, oi, sr[j], si[j]);
        }
    }
    #pragma unroll
    for (int j = 0; j < NH; ++j) if (c0 + j < N) out[c0 + j] = make_double2(ar[j], ai[j]);
}

#ifdef __CUDACC__
template <int NP, int TPL>
__global__ void rect_multi(const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix,
                           const long long* __restrict__ lane2s, const long long* __restrict__ lstart,
                           const long long* __restrict__ nterms, const double* __restrict__ binom,
                           double2* __restrict__ part, int Q1, int N, long long L, long long TOTAL)
{
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long g = gt / TPL;
    const int h = (int)(gt % TPL);
    if (g >= TOTAL) return;
    Xchg<TPL> xc; xc.grp = ((TPL >= 32) ? 0xffffffffu : (1u << TPL) - 1u) << ((threadIdx.x & 31) & ~(TPL - 1));
    rect_lane<NP, TPL>(Bs, mult, radix, lane2s, lstart, nterms, binom, part, Q1, N, L, g, h, xc);
}
template <int NP, int TPL>
__global__ void mu_multi(const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix,
                         const long long* __restrict__ lane2s, const long long* __restrict__ lstart,
                         const long long* __restrict__ nterms, const double* __restrict__ binom,
                         double2* __restrict__ part, int Q1, int N, int D, long long L, long long TOTAL)
{
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long g = gt / TPL;
    const int h = (int)(gt % TPL);
    if (g >= TOTAL) return;
    Xchg<TPL> xc; xc.grp = ((TPL >= 32) ? 0xffffffffu : (1u << TPL) - 1u) << ((threadIdx.x & 31) & ~(TPL - 1));
    mu_lane<NP, TPL>(Bs, mult, radix, lane2s, lstart, nterms, binom, part, Q1, N, D, L, g, h, xc);
}
template <int NP, int TPL, int D>
__global__ void mud_multi(const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix,
                          const long long* __restrict__ lane2s, const long long* __restrict__ lstart,
                          const long long* __restrict__ nterms, const double* __restrict__ binom,
                          double2* __restrict__ part, int Q1, int N, long long L, long long TOTAL)
{
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long g = gt / TPL;
    const int h = (int)(gt % TPL);
    if (g >= TOTAL) return;
    Xchg<TPL> xc; xc.grp = ((TPL >= 32) ? 0xffffffffu : (1u << TPL) - 1u) << ((threadIdx.x & 31) & ~(TPL - 1));
    mud_lane<NP, TPL, D>(Bs, mult, radix, lane2s, lstart, nterms, binom, part, Q1, N, L, g, h, xc);
}
// Plain-named entry points (10-06): cupy reads its disk cache only for modules compiled WITHOUT name_expressions, so the
// host instantiates the kernels through these extern "C" wrappers, one line per (tier, TPL[, D]) appended to the source
// (same parameters and body as the templates above; the cache then survives across processes).
#define PE_GT_SETUP(TPL) \
    const long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x; const long long g = gt / (TPL); \
    const int h = (int)(gt % (TPL)); if (g >= TOTAL) return; \
    Xchg<TPL> xc; xc.grp = (((TPL) >= 32) ? 0xffffffffu : (1u << (TPL)) - 1u) << ((threadIdx.x & 31) & ~((TPL) - 1));
#define PE_RECT_ARGS const double2* __restrict__ Bs, const int* __restrict__ mult, const int* __restrict__ radix, \
    const long long* __restrict__ lane2s, const long long* __restrict__ lstart, const long long* __restrict__ nterms, \
    const double* __restrict__ binom, double2* __restrict__ part, int Q1, int N
#define PE_RECT_ENTRY(NP, TPL) extern "C" __global__ void rect_k_##NP##_##TPL(PE_RECT_ARGS, long long L, long long TOTAL) \
    { PE_GT_SETUP(TPL) rect_lane<NP, TPL>(Bs, mult, radix, lane2s, lstart, nterms, binom, part, Q1, N, L, g, h, xc); }
#define PE_MU_ENTRY(NP, TPL) extern "C" __global__ void mu_k_##NP##_##TPL(PE_RECT_ARGS, int D, long long L, \
                                                                          long long TOTAL) \
    { PE_GT_SETUP(TPL) mu_lane<NP, TPL>(Bs, mult, radix, lane2s, lstart, nterms, binom, part, Q1, N, D, L, g, h, xc); }
#define PE_MUD_ENTRY(NP, TPL, D) extern "C" __global__ void mud_k_##NP##_##TPL##_##D(PE_RECT_ARGS, long long L, \
                                                                                      long long TOTAL) \
    { PE_GT_SETUP(TPL) mud_lane<NP, TPL, D>(Bs, mult, radix, lane2s, lstart, nterms, binom, part, Q1, N, L, g, h, xc); }
#else
extern "C" void rect_emul(const double* Bs, const int* mult, const int* radix, const long long* lane2s,
                          const long long* lstart, const long long* nterms, const double* binom, double* part,
                          int Q1, int N, int D, long long L, long long TOTAL)
{
    // D < 0: the lambda-form (rect_lane); D >= 0: the mu-form (mu_lane) at degree D
    constexpr int TPL = PE_TPL;
    const long long nb = 8;
    for (long long g0 = 0; g0 < TOTAL; g0 += nb) {
        std::vector<std::thread> th;
        std::vector<std::barrier<>*> bars;
        std::vector<const real_t**> slots;
        std::vector<int*> lens;
        for (long long g = g0; g < TOTAL && g < g0 + nb; ++g) {
            std::barrier<>* bar = new std::barrier<>(TPL);
            const real_t** sl = new const real_t*[TPL];
            int* ln = new int[TPL];
            bars.push_back(bar); slots.push_back(sl); lens.push_back(ln);
            for (int h = 0; h < TPL; ++h)
                th.emplace_back([=]() {
                    Xchg<TPL> xc; xc.slots = sl; xc.lens = ln; xc.bar = bar; xc.h = h;
                    if (D < 0) rect_lane<PE_NP, TPL>((const double2*)Bs, mult, radix, lane2s, lstart, nterms, binom,
                                                     (double2*)part, Q1, N, L, g, h, xc);
                    else mu_lane<PE_NP, TPL>((const double2*)Bs, mult, radix, lane2s, lstart, nterms, binom,
                                             (double2*)part, Q1, N, D, L, g, h, xc);
                });
        }
        for (auto& t : th) t.join();
        for (auto* b : bars) delete b;
        for (auto* s : slots) delete[] s;
        for (auto* l : lens) delete[] l;
    }
}

#define PE_MU_DMAX 24                                  // = cuda_rect.MU_DMAX (checked by the tests)
template <int D>
static void mud_dispatch(int Dr, const double2* B2, const int* mult, const int* radix, const long long* lane2s,
                         const long long* lstart, const long long* nterms, const double* binom, double2* P2,
                         int Q1, int N, long long L, long long g, int h, const Xchg<PE_TPL>& xc)
{
    if (Dr == D) mud_lane<PE_NP, PE_TPL, D>(B2, mult, radix, lane2s, lstart, nterms, binom, P2, Q1, N, L, g, h, xc);
    else if constexpr (D < PE_MU_DMAX) mud_dispatch<D + 1>(Dr, B2, mult, radix, lane2s, lstart, nterms, binom, P2, Q1, N,
                                                          L, g, h, xc);
}

extern "C" void mud_emul(const double* Bs, const int* mult, const int* radix, const long long* lane2s,
                         const long long* lstart, const long long* nterms, const double* binom, double* part,
                         int Q1, int N, int D, long long L, long long TOTAL)
{
    // the templated mu lane at degree D = 0 .. PE_MU_DMAX
    constexpr int TPL = PE_TPL;
    const long long nb = 8;
    for (long long g0 = 0; g0 < TOTAL; g0 += nb) {
        std::vector<std::thread> th;
        std::vector<std::barrier<>*> bars;
        std::vector<const real_t**> slots;
        std::vector<int*> lens;
        for (long long g = g0; g < TOTAL && g < g0 + nb; ++g) {
            std::barrier<>* bar = new std::barrier<>(TPL);
            const real_t** sl = new const real_t*[TPL];
            int* ln = new int[TPL];
            bars.push_back(bar); slots.push_back(sl); lens.push_back(ln);
            for (int h = 0; h < TPL; ++h)
                th.emplace_back([=]() {
                    Xchg<TPL> xc; xc.slots = sl; xc.lens = ln; xc.bar = bar; xc.h = h;
                    const double2* B2 = (const double2*)Bs; double2* P2 = (double2*)part;
                    if (D >= 0 && D <= PE_MU_DMAX)
                        mud_dispatch<0>(D, B2, mult, radix, lane2s, lstart, nterms, binom, P2, Q1, N, L, g, h, xc);
                });
        }
        for (auto& t : th) t.join();
        for (auto* b : bars) delete b;
        for (auto* s : slots) delete[] s;
        for (auto* l : lens) delete[] l;
    }
}
#endif
"""

# ---- tier table: NP (padded column count) -> (TPL, --maxrregcount).  Measured on the H100 (sm 9.0) by
# bench/bench_cuda_rect.py at the chain's step sizes, 64 streams, run 20261004_224743: TPL 1
# at tiers 32-40 (q = 10, 15), TPL 2 at tiers 48-64 (q = 20-27; TPL 4 is 1.2x slower at q = 27, TPL 8 2.6x), register
# caps within 3 % of the compiler's choice.  Per-thread state is ~6 (NP + 1) + 4 NH doubles (the polynomial arrays
# dominate), so the default raises TPL with the tier only to shrink the column share; tiers 8-24 and 72-96 keep it
# (72/96: TPL 8 validated for correctness on the GPU, not timed).
_TIERS_LADDER = tuple(range(8, 97, 8))     # N <= 96 active columns: the lossy SBS curve reaches N ~ 85 at eta = 0.5
_CFG_DEFAULT = {k: ((1 if k <= 16 else (2 if k <= 32 else (4 if k <= 64 else 8))), None) for k in _TIERS_LADDER}
_CFG_BY_ARCH = {(9, 0): {**_CFG_DEFAULT, 32: (1, None), 40: (1, None), 48: (2, 128), 56: (2, None), 64: (2, 255)}}
_CFG = None
_MODULES = {}
_KERNELS = {}


def _cfg():
    global _CFG
    if _CFG is None:
        arch = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        _CFG = dict(_CFG_BY_ARCH.get(arch, _CFG_DEFAULT))
    return _CFG


def set_config(cfg):
    """Override the (tier -> (TPL, maxrregcount)) table for this process (merged); None resets."""
    global _CFG, _MODULES, _KERNELS
    _CFG = None if cfg is None else {**_cfg(), **{int(k): tuple(v) for k, v in cfg.items()}}
    _MODULES, _KERNELS = {}, {}


def tier(N):
    """(padded column count, threads per lane) of the tier serving N columns."""
    NP = next(k for k in _TIERS_LADDER if k >= N)
    return NP, _cfg()[NP][0]


# Plain-named extern "C" entry points (10-06): cupy reads its disk cache only for modules compiled without
# name_expressions. ON by default since run 20261006_115902 (job 33: host tests 54 passed, device check ok, timings as run
# 20261005_215026, fresh-process warm-up 59 s -> 0.6 s); PE_PLAIN_ENTRY=0 restores the name_expressions path.
_PLAIN = os.environ.get("PE_PLAIN_ENTRY", "1") == "1"
_ENTRIES = {"PE_RECT_ENTRY": ("rect_multi", "rect_k"), "PE_MU_ENTRY": ("mu_multi", "mu_k"),
            "PE_MUD_ENTRY": ("mud_multi", "mud_k")}


def _kname(macro, *p):
    """Kernel name of a (macro, template parameters) entry: the wrapper's plain name, or the template expression."""
    tpl, plain = _ENTRIES[macro]
    return f"{plain}_{'_'.join(map(str, p))}" if _PLAIN else f"{tpl}<{', '.join(map(str, p))}>"


def _raw_module(specs, cap):
    """NVRTC module for the entries specs = [(macro, params), ...]: plain-named wrappers appended to the source (no
    name_expressions, disk-cached across processes) when _PLAIN, else the templates by name_expressions.
    """
    import cupy
    opts = ("-std=c++17",) if cap is None else ("-std=c++17", f"--maxrregcount={cap}")
    if _PLAIN:
        lines = [f"{m}({', '.join(map(str, p))})" for m, p in specs]
        return cupy.RawModule(code=_SRC + "\n#ifdef __CUDACC__\n" + "\n".join(lines) + "\n#endif\n", options=opts)
    return cupy.RawModule(code=_SRC, options=opts, name_expressions=[_kname(m, *p) for m, p in specs])


def warmup():
    """Compile every (register cap) module up front, as `cuda_cofactors.warmup` does."""
    cfg = _cfg()
    for cap in dict.fromkeys(cfg[k][1] for k in _TIERS_LADDER):
        if cap not in _MODULES:
            tiers = [k for k in _TIERS_LADDER if cfg[k][1] == cap]
            _MODULES[cap] = _raw_module([("PE_RECT_ENTRY", (k, cfg[k][0])) for k in tiers]
                                        + [("PE_MU_ENTRY", (k, cfg[k][0])) for k in tiers], cap)


def _get_kernel(NP, form="lambda"):
    cfg = _cfg()
    key = (NP, form)
    if key not in _KERNELS:
        if not _MODULES:
            warmup()
        macro = "PE_RECT_ENTRY" if form == "lambda" else "PE_MU_ENTRY"
        _KERNELS[key] = _MODULES[cfg[NP][1]].get_function(_kname(macro, NP, cfg[NP][0]))
    return _KERNELS[key]


DMAX = 6                                      # PE_DMAX in the kernel source: the generic mu-form's largest degree s0
MU_DMAX = 24                                  # degrees D = s0 the templated mu lane (mud_lane) is instantiated for
#                                               (17..24: measured 10-06, forced form="mu" only; auto stops at MU_AUTO_MAX)
#                                               (= PE_MU_DMAX of the emulation)
# "auto": the templated mu lane for 1 <= s0 <= MU_AUTO_MAX at N <= MU_AUTO_NMAX (the measured range, run
# 20261005_215026), s0 <= MU_AUTO_MAX_WIDE above. First extension (10-05): 16, validated by run 20261005_222151 (host
# tests 40 passed; whole post-selected shots at n = 34 vs the dispatch at 6: x1.0 / 1.4 / 1.9 / 7.2 / 8.1 / 5.6 / 3.2 /
# 2.1 / 1.2 at s0 = 0 / 3 / 6 / 7 / 8 / 10 / 12 / 14 / 16). PE_MU_AUTO_MAX=6 restores the build of run 20261006_001928.
# Second extension (10-06, run 20261006_120404): the lane still beats lambda at N = 49..60 (D <= 16: x2.1-12.5) and at
# D = 17 / 18 (x1.61 / 1.47), not from D = 20 (x0.97, 0.98, 0.86 at 20 / 22 / 24) -> mu to s0 = 18 at N = 29..64. ON by
# default since its validation (runs 20261006_122136: host tests 42 passed; 20261006_125313: whole post-selected shots
# at n = 50 / 52, x1.76 / 1.86 / 1.25 / 1.10 at s0 = 16 / 17 / 18 / 18, lambda control x1.00). PE_MU_R22=0 = the first.
_R22 = os.environ.get("PE_MU_R22", "1") == "1"
MU_AUTO_MAX = int(os.environ.get("PE_MU_AUTO_MAX", "18" if _R22 else "16"))
MU_AUTO_NMIN, MU_AUTO_NMAX = 29, (64 if _R22 else 48)   # measured tiers 32-48 (10-05), 52-64 (10-06); narrower steps (cheap,
MU_AUTO_MAX_WIDE = 6                          # early in the chain) keep s0 <= 6 -- warmup_mu compiles D > 6 there only


# ---- the templated mu lane's own tier ladder and (TPL, maxrregcount) rule.  Any TPL is legal (NH = ceil(NP / TPL)), so
# the ladder runs in steps of 4 like the Fock kernel's.  Per-thread state: NH (4 + 2 (D + 1)) doubles (column sums,
# accumulators, the stored prefixes).  Default TPL = the smallest power of two keeping that <= _MU_BUDGET doubles (the
# Fock kernel at tier 36 / TPL 4 holds 48).  Measured on the H100 (TPL 4 / 8 / 16): run 20261005_181800 (tier 36,
# D = 0..6: TPL 4 best, the budget rule's TPL 8-16 up to 2.3x slower) and run 20261005_215026 (tiers 32-48, D = 7..16:
# TPL 4 best to D = 11, TPL 8 from D = 12) -> _MU_CFG_BY_ARCH at tiers 32-48; other tiers keep the budget rule.
_MU_TIERS = tuple(range(8, 65, 4)) + tuple(range(72, 97, 8))
_MU_BUDGET = 80
_MU_CFG_BY_ARCH = ({(9, 0): {(NP, D): ((4 if D <= 11 else 8), None) for NP in (32, 36, 40, 44, 48) for D in range(17)}}
                   if MU_AUTO_MAX > MU_AUTO_MAX_WIDE else {})          # with the extended dispatch (same switch)
if _R22 and (9, 0) in _MU_CFG_BY_ARCH:        # run 20261006_120404: D 17-18 at tiers 32-48 TPL 8 (best / <= 5 % off);
    _MU_CFG_BY_ARCH[(9, 0)].update({(NP, D): (8, None) for NP in (32, 36, 40, 44, 48) for D in (17, 18)})
    _MU_CFG_BY_ARCH[(9, 0)].update({(NP, D): (8, None) for NP in (52, 56) for D in range(19)})           # TPL 8 best
    _MU_CFG_BY_ARCH[(9, 0)].update({(NP, D): ((8 if D <= 12 else 16), None) for NP in (60, 64) for D in range(19)})
_MU_OVERRIDE = {}
_MU_MODULES = {}
_MU_KERNELS = {}


def _mu_default(NP, D):
    for tpl in (1, 2, 4, 8, 16):
        if -(-NP // tpl) * (4 + 2 * (D + 1)) <= _MU_BUDGET:
            return tpl, None
    return 16, None


def mu_tier(N, D):
    """(padded column count, threads per lane, maxrregcount) of the templated mu lane serving N columns at degree D."""
    NP = next(k for k in _MU_TIERS if k >= N)
    arch = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    tpl, cap = _MU_OVERRIDE.get((NP, D)) or _MU_CFG_BY_ARCH.get(arch, {}).get((NP, D)) or _mu_default(NP, D)
    return NP, int(tpl), cap


def set_mu_config(cfg):
    """Override {(tier, D): (TPL, maxrregcount)} of the templated mu lane for this process (merged); None resets."""
    global _MU_OVERRIDE
    for v in (cfg or {}).values():
        if int(v[0]) < 1 or int(v[0]) & (int(v[0]) - 1) or int(v[0]) > 32:
            raise ValueError(f"threads per lane must be a power of two <= 32, got {v[0]}")
    _MU_OVERRIDE = {} if cfg is None else {**_MU_OVERRIDE, **{(int(k[0]), int(k[1])): tuple(v) for k, v in cfg.items()}}


def _mu_name(NP, TPL, D):
    return _kname("PE_MUD_ENTRY", NP, TPL, D)


def _mu_entry(NP, TPL, D):
    return ("PE_MUD_ENTRY", (NP, TPL, D))


def warmup_mu(degrees=None, n_max=None):
    """Compile the templated mu lane for every tier (<= n_max columns) at the given degrees (default: what "auto" uses --
    1 .. MU_AUTO_MAX_WIDE at every tier, up to MU_AUTO_MAX at tiers <= MU_AUTO_NMAX), one NVRTC module per (degree,
    register cap): a run's first s0 = 3 stratum must not compile inside a timed region.
    """
    auto = degrees is None
    degrees = range(1, max(MU_AUTO_MAX, MU_AUTO_MAX_WIDE) + 1) if auto else degrees
    top = _MU_TIERS[-1] if n_max is None else next(k for k in _MU_TIERS if k >= n_max)
    for D in degrees:
        groups = {}
        for NP in [k for k in _MU_TIERS if k <= top and (not auto or mu_auto_max(k) >= D)]:
            _, tpl, cap = mu_tier(NP, D)
            if (NP, tpl, D) not in _MU_KERNELS:
                groups.setdefault(cap, []).append((NP, tpl))
        for cap, items in groups.items():
            names = [_mu_name(NP, tpl, D) for NP, tpl in items]
            mod = _raw_module([_mu_entry(NP, tpl, D) for NP, tpl in items], cap)
            _MU_MODULES[(D, cap, tuple(items))] = mod
            for (NP, tpl), nm in zip(items, names):
                _MU_KERNELS[(NP, tpl, D)] = (mod.get_function(nm), cap)


def _get_mu_kernel(NP, TPL, cap, D):
    key = (NP, TPL, D)
    if key in _MU_KERNELS and _MU_KERNELS[key][1] == cap:
        return _MU_KERNELS[key][0]
    if not _MU_MODULES:
        warmup_mu()                                   # all default kernels at once, on the first mu call of the process
        if key in _MU_KERNELS and _MU_KERNELS[key][1] == cap:
            return _MU_KERNELS[key][0]
    mod = _raw_module([_mu_entry(NP, TPL, D)], cap)
    _MU_MODULES[(D, cap, ((NP, TPL),))] = mod
    _MU_KERNELS[key] = (mod.get_function(_mu_name(NP, TPL, D)), cap)
    return _MU_KERNELS[key][0]


def choose_form(q, N, form="auto"):
    """Which payload computes the step (q placed rows, N active columns, s0 = N - 1 - q): "fock" (s0 = 0: the production
    Fock cofactor kernel itself, `cuda_cofactors.cofactor_partials_cuda` on B -- the rso step IS the Fock step there),
    "mu" (mud_lane: the mu-form templated on D = s0 <= MU_DMAX, division-free, 3 (s0 + 1) complex products per column
    and term), "mu_generic" (mu_lane, runtime degree s0 <= DMAX: the reference the templated lane replaced) or
    "lambda" (rect_lane, F(lambda) and the mixed division, O(N^2 / TPL + N q) per term). Measured with the generic lane,
    run 20261005_132632 (H100, n = 34, the chain's last step, 64 streams; seconds Fock / mu_generic / lambda):
        s0 = 0:  8.7 / 123 / 224   (mu 1.4e-12 vs the Fock kernel, lambda 1.8e-6)
        s0 = 1:    - / 128 / 160   (mu two summation orders 2.9e-13, lambda 2.1e-7 off)
        s0 = 2:    - /  67 /  75   (mu 2.8e-13, lambda 4.7e-8 off)
        s0 = 4:    - /  38 /  30   (mu vs lambda 1.3e-9)
        s0 = 6:    - /  16 /  13   (mu vs lambda 1.6e-10)
    The generic lane ran ~10-17x below the Fock kernel's per-flop rate (runtime-degree loops, a [NH][DMAX + 1] prefix
    block in local memory), which is the 7x step at p -> 1 of the continuity check (a) of `bench/sbs_rso_validate.py`
    (run 20261005_125116). The templated lane
    (run 20261005_181800, n = 34, last step, TPL 4): s0 = 0..6 at 5.4 / 5.5 / 4.2 / 3.2 / 2.4 / 1.8 / 1.1 s vs the Fock
    kernel's 6.6 s at s0 = 0. Templated mu vs lambda on the same instances (run 20261005_215026, q = 24, N = 32..41;
    ps per term-column, lambda / best mu): s0 = 7..16 -> 8.3, 8.8, 8.6, 7.9, 6.3, 4.5, 4.6, 4.2, 3.5, 1.8 (q = 28: 9.0 at
    s0 = 7, N = 36; 2.1 at s0 = 16, N = 45); mu vs lambda 6e-13 .. 2e-10 (lambda the less accurate, see above).
    "auto": fock at s0 = 0, the templated mu lane for
    1 <= s0 <= mu_auto_max(N) (default: 18 at N = 29..64, the measured range, run 20261006_120404; 6 outside;
    PE_MU_R22=0 restores the first extension: 16 at N = 29..48), lambda beyond. The phantom-row route is exact
    at s0 = 0 only (4e-5 at s0 = 1), hence "fock" means s0 = 0 and nothing else.
    """
    s0 = N - 1 - q
    if form in ("lambda", "mu", "mu_generic", "fock"):
        if form == "mu_generic" and s0 > DMAX:
            raise ValueError(f"mu_generic form: s0 = N - 1 - q = {s0} exceeds DMAX = {DMAX}")
        if form == "mu" and s0 > MU_DMAX:
            raise ValueError(f"mu form: s0 = N - 1 - q = {s0} exceeds MU_DMAX = {MU_DMAX}")
        if form == "fock" and s0 != 0:
            raise ValueError(f"fock form: s0 = N - 1 - q must be 0, got {s0}")
        return form
    if form != "auto":
        raise ValueError(f"form {form!r}: fock | mu | mu_generic | lambda | auto")
    return "fock" if s0 == 0 else ("mu" if s0 <= mu_auto_max(N) else "lambda")


def mu_auto_max(N):
    """Largest s0 that "auto" sends to the templated mu lane at N active columns: MU_AUTO_MAX inside the measured range
    MU_AUTO_NMIN <= N <= MU_AUTO_NMAX (default: 18 at N = 29..64, tiers 32-64; PE_MU_R22=0: 16 at N = 29..48), else
    MU_AUTO_MAX_WIDE = 6.
    """
    return MU_AUTO_MAX if MU_AUTO_NMIN <= N <= MU_AUTO_NMAX else MU_AUTO_MAX_WIDE


def _cp(t):
    import cupy
    return cupy.asarray(t)


def _plan(A2, rows, q, target_total_lanes):
    """Tables, lane plan and the gathered block for A2 [S, m, N] (float64 pairs) and rows [S, q]."""
    from .cofactors_triton import multiplicity_tables, plan_multi
    S = A2.shape[0]
    dev = A2.device
    if q == 0:
        z = torch.zeros((S, 1), dtype=torch.long, device=dev)
        tables = (z, z.to(torch.int32), (z + 1).to(torch.int32), torch.ones(S, dtype=torch.long, device=dev),
                  torch.ones(S, dtype=torch.float64, device=dev))
    else:
        tables = multiplicity_tables(rows, q)
    drow, mult, radix, Nt, scale = tables
    L, nl, lstart, lane2s, total = plan_multi(Nt, target_total_lanes)
    A = torch.view_as_complex(A2)
    Bs = A[torch.arange(S, device=dev)[:, None], drow[:, :max(q, 1)], :]                         # [S, q, N]
    return torch.view_as_real(Bs.contiguous()), mult.contiguous(), radix.contiguous(), Nt, scale, L, nl, lstart, lane2s, total


def _reduce(part, lane2s, scale, S, N):
    """Per-stream sums of the lane partials, times the 2020 scale -> [S, N] complex."""
    acc = torch.zeros((S, N, 2), dtype=torch.float64, device=part.device)
    acc.index_add_(0, lane2s, part)
    return torch.view_as_complex(acc) * scale[:, None]


def rect_cofactors_cuda(A, rows, q, target_total_lanes=1 << 17, block=128, form="auto"):
    """P[s, x] = Per_rect(A[s][rows[s], all columns but x]) for A [S, m, N] complex128 and rows [S, q] int64
    (the placed output rows, repeats allowed) -> [S, N] complex128, on the CUDA kernel. form: `choose_form`.
    """
    import cupy

    from .cofactors_triton import _binom_table
    dev = A.device
    A2 = torch.view_as_real(A.contiguous())
    S, N = A.shape[0], A.shape[2]
    Bs2, mult, radix, Nt, scale, L, nl, lstart, lane2s, total = _plan(A2, rows, q, target_total_lanes)
    part = torch.empty(total, N, 2, dtype=torch.float64, device=dev)
    form = choose_form(q, N, form)
    if form == "fock":
        return phantom_cofactors(A, rows, q, backend="cuda", target_total_lanes=target_total_lanes)
    if form == "mu":
        NP, TPL, cap = mu_tier(N, N - 1 - q)
        kern = _get_mu_kernel(NP, TPL, cap, N - 1 - q)
    else:
        NP, TPL = tier(N)
        kern = _get_kernel(NP, "lambda" if form == "lambda" else "mu")
    grid = ((total * TPL + block - 1) // block,)
    args = [_cp(Bs2), _cp(mult), _cp(radix), _cp(lane2s), _cp(lstart), _cp(Nt), _cp(_binom_table(dev)), _cp(part),
            cupy.int32(q), cupy.int32(N)]
    if form == "mu_generic":
        args.append(cupy.int32(N - 1 - q))
    args += [cupy.int64(L), cupy.int64(total)]
    with cupy.cuda.Device(dev.index or 0):
        kern(grid, (block,), tuple(args))
    return _reduce(part, lane2s, scale, S, N)


# ---- CPU emulation ----
_EMUL = {}


def _cache_dir():
    d = os.environ.get("PE_KERNEL_CACHE") or os.path.join(tempfile.gettempdir(), "photonic_engines_kernels")
    os.makedirs(d, exist_ok=True)
    return d


def emul_available():
    """True when a host C++20 compiler builds the emulation."""
    try:
        _emul_lib(8, 1)
        return True
    except Exception:
        return False


def _emul_lib(NP, TPL, real="double"):
    key = (NP, TPL, real)
    if key in _EMUL:
        return _EMUL[key]
    h = hashlib.sha1(_SRC.encode()).hexdigest()[:10]
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    tag = "" if real == "double" else "_ld"
    path = os.path.join(_cache_dir(), f"pe_rect_emul_n{NP}_t{TPL}{tag}_{h}{ext}")
    if not os.path.exists(path):
        src = path + ".cpp"
        with open(src, "w") as fh:
            fh.write(_SRC)
        cxx = os.environ.get("CXX") or ("clang++" if sys.platform == "darwin" else "g++")
        cmd = [cxx, "-std=c++20", "-O2", "-fPIC", "-shared", f"-DPE_NP={NP}", f"-DPE_TPL={TPL}", f"-DPE_REAL={real}",
               "-o", path + ".tmp", src]
        if sys.platform != "darwin":
            cmd.insert(-3, "-pthread")
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(path + ".tmp", path)
    lib = ctypes.CDLL(path)
    P = ctypes.c_void_p
    lib.rect_emul.argtypes = [P] * 8 + [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    lib.rect_emul.restype = None
    lib.mud_emul.argtypes = [P] * 8 + [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    lib.mud_emul.restype = None
    _EMUL[key] = lib
    return lib


def rect_cofactors_emul(A, rows, q, target_total_lanes=1 << 17, tpl=None, real="double", form="auto"):
    """CPU run of the kernel code (same contract as `rect_cofactors_cuda`; CPU tensors). tpl overrides the tier's
    threads per lane; real = "long double" runs the sums, polynomials and accumulators in x87 extended precision;
    form as in `choose_form`.
    """
    from .cofactors_triton import _binom_table
    A2 = torch.view_as_real(A.contiguous())
    S, N = A.shape[0], A.shape[2]
    Bs2, mult, radix, Nt, scale, L, nl, lstart, lane2s, total = _plan(A2, rows, q, target_total_lanes)
    form = choose_form(q, N, form)
    s0 = N - 1 - q
    templated = form in ("mu", "fock")            # the CPU has no Fock kernel: the templated mu lane at D = 0
    NP, TPL = mu_tier(N, s0)[:2] if templated else tier(N)
    TPL = TPL if tpl is None else int(tpl)
    if TPL < 1 or TPL & (TPL - 1) or TPL > 32:
        raise ValueError(f"threads per lane must be a power of two <= 32, got {TPL}")
    part = torch.zeros(total, N, 2, dtype=torch.float64)
    binom = _binom_table(torch.device("cpu"))
    lib = _emul_lib(NP, TPL, real)

    def ptr(t):
        return ctypes.c_void_p(t.data_ptr())
    fn = lib.mud_emul if templated else lib.rect_emul
    D = s0 if form != "lambda" else -1
    fn(ptr(Bs2), ptr(mult), ptr(radix), ptr(lane2s.contiguous()), ptr(lstart.contiguous()), ptr(Nt.contiguous()),
       ptr(binom), ptr(part), q, N, D, L, total)
    return _reduce(part, lane2s, scale, S, N)


def phantom_cofactors(A, rows, q, backend="cuda", target_total_lanes=1 << 17):
    """P[s, x] = Per_rect(A[s][rows[s], all columns but x]) through the PRODUCTION Fock cofactor kernel, with
    s0 = N - 1 - q phantom all-ones rows: Per([J_s0; B][:, all but x]) = s0! Per_rect(B[:, all but x]) (every
    s0-subset of the remaining columns, in every order, completes the q real rows). A [S, m, N] complex128, rows
    [S, q] int64 -> [S, N] complex128 (the rso chain on the Fock substrate; docs/ALGORITHMS.md).

    The phantom row has multiplicity s0, so the 2020 collapse leaves (s0 + 1) 2^(q-1) terms of O(N) each against
    the rectangular kernel's 2^(q-1) terms of O(N q) -- cheaper when s0 + 1 < q, and the Fock chain's own step at
    s0 = 0. **Exact but not usable in fp64 beyond s0 ~ 4**: the collapsed phantom factor (sum_j delta_j)^s0 is an
    s0-th finite difference and cancels catastrophically; measured against the long-double rectangular kernel
    (CPU, 2026-10-05): q = 6: 3.5e-16 (s0 = 0), 7.5e-14 (2), 5.7e-12 (4), 4.4e-10 (8), 4.9e-9 (12); q = 10:
    1.3e-15 (0), 1.0e-11 (2), 3.6e-9 (4), 5.6e-6 (8), 2.6e-4 (12) -- while the rectangular kernel stays at 1e-15.
    Kept as the reference of that negative result (the same law was measured independently on the chain side); the
    stable equivalent is the mu-form payload. backend "cuda" =
    `cuda_cofactors.cofactor_partials_cuda` (N <= 64 columns), "triton" = `cofactors_triton.cofactor_partials_multi`
    (the CPU interpreter in tests). The ones row is appended as mode m of a copy of A; nothing else is touched.
    """
    import math

    S, m, N = A.shape
    q = int(q)
    s0 = N - 1 - q
    if s0 < 0:
        raise ValueError(f"phantom_cofactors: q = {q} placed rows need at least q + 1 = {q + 1} columns, got N = {N}")
    dev = A.device
    if N == 1:
        return torch.ones(S, 1, dtype=torch.complex128, device=dev)
    if backend == "cuda":
        from .cuda_cofactors import cofactor_partials_cuda as multi
    elif backend == "triton":
        from .cofactors_triton import cofactor_partials_multi as multi
    else:
        raise ValueError(f"phantom_cofactors: backend {backend!r} (cuda | triton)")
    if s0 == 0:                                       # the Fock step itself: no phantom row, A as it is
        A_aug, rows_aug = A.to(torch.complex128), rows[:, :q].to(dev, torch.int64)
    else:
        A_aug = torch.cat([A.to(torch.complex128), torch.ones(S, 1, N, dtype=torch.complex128, device=dev)], dim=1)
        rows_aug = torch.cat([torch.full((S, s0), m, dtype=torch.int64, device=dev), rows[:, :q].to(dev, torch.int64)], dim=1)
    sig = torch.arange(N, dtype=torch.int64, device=dev).expand(S, N).contiguous()
    dp = torch.arange(S, dtype=torch.int64, device=dev)
    A2 = torch.view_as_real(A_aug.contiguous())                                                # [S, m+1, N, 2]
    part, lstart, nl, scale, _, lane2s = multi(A2, rows_aug, sig, dp, S, N - 1, N, target_total_lanes)
    return _reduce(part, lane2s, scale, S, N) / float(math.factorial(s0))


def available():
    """True when cupy is importable and a CUDA device is visible to torch."""
    try:
        import cupy  # noqa
        return torch.cuda.is_available()
    except Exception:
        return False
