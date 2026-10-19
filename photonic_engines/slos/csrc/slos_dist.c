/*
 * slos_dist.c -- SLOS_full with a distributed amplitude vector, as a library (v2: pattern
 * ownership, interval halo, OpenMP threads).
 *
 * Layout. Every rank owns a range of the *pattern key order* (lexicographic on (t[m-1], ...,
 * t[1]), then t[0]) -- the same order as the colex rank inside a layer, so the ownership of a
 * pattern does not depend on the layer and rank r's chunk of layer k is the contiguous colex
 * range [lo[k][r], lo[k][r+1]). The P-1 boundary patterns are chosen on layer n so that the
 * work (1 + number of non-empty modes per row) is balanced, not the row count.
 *
 * Halo. For a fixed mode i, t -> t - e_i is monotone in that order, so the parents via mode i
 * of a contiguous block of rows form one contiguous interval of the previous layer with no
 * gap; the halo of a batch is the union of at most m intervals, split by owner, fetched as
 * contiguous copies (no index request, no hash), and looked up with a per-mode cursor.
 *
 * Threads. Pass 1 (parent ranks via the O(1) incremental colex rank of slos_scatter.c, rows
 * by the odometer) and pass 2 (gather/multiply-add) are OpenMP loops over row chunks; the MPI
 * ranks are meant to be nodes, the threads their cores. Without OpenMP everything runs on one
 * thread. No I/O, no MPI_Init: the caller (mpi4py) owns the communicator (MPI.Comm.py2f()).
 *
 * Build (done by photonic_engines.slos.distributed at first use):
 *     mpicc -O3 -march=native -fopenmp -shared -fPIC -o libslos_dist.so slos_dist.c -lm
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef unsigned char u8;
typedef double complex cplx;

typedef struct {
    MPI_Comm comm; int rank, nproc;
    int n, m; int *occ; int *path;
    long *binom; int A;
    double denom;                    /* prod sqrt(s_i!) */
    long batch;
    u8 *bnd;                         /* [nproc+1][m] boundary patterns (bnd[0] = vacuum) */
    long *lo;                        /* [n+1][nproc+1] first colex index of rank r in layer k */
} slos_dist;

typedef struct { long start, len; const cplx *base; int owner; } seg_t;

static inline long binom_(const slos_dist *d, int a, int b) { if (b < 0 || a < 0 || b > a) return 0; return d->binom[(long)a * (d->A + 1) + b]; }
static inline long fock_count(const slos_dist *d, int M, int K) { if (K < 0 || M <= 0) return (K == 0 && M == 0) ? 1 : 0; return binom_(d, K + M - 1, M - 1); }
static long colex_rank(const slos_dist *d, const u8 *t, int m, int k) { long r = 0; int K = k;
    for (int nm = m; nm >= 2; nm--) { int v = t[nm - 1]; r += fock_count(d, nm, K) - fock_count(d, nm, K - v); K -= v; } return r; }
static void colex_unrank(const slos_dist *d, long j, int m, int k, u8 *t) { int K = k;
    for (int nm = m; nm >= 2; nm--) { int v = 0; for (;;) { long b = fock_count(d, nm - 1, K - v); if (j < b) break; j -= b; v++; } t[nm - 1] = (u8)v; K -= v; } t[0] = (u8)K; }
/* successor of t in colex order within its layer: lexicographic on (t[m-1], ..., t[1]) ascending,
 * t[0] = remainder. O(1) amortised. */
static inline void colex_next(u8 *t, int m) {
    if (t[0] > 0) { t[0]--; t[1]++; return; }
    int j = 1; while (j < m - 1 && t[j] == 0) j++;
    if (j >= m - 1) return;                                   /* last state of the layer */
    u8 v = t[j]; t[j] = 0; t[j + 1]++; t[0] = (u8)(v - 1);
}
/* number of layer-k patterns whose key precedes the key of pattern b (any photon number) */
static long lo_of(const slos_dist *d, int k, const u8 *b) {
    const int m = d->m; long r = 0; int K = k;
    for (int j = m - 1; j >= 1; j--) { r += fock_count(d, j + 1, K) - fock_count(d, j + 1, K - b[j]); K -= b[j]; }
    if (K >= 0 && K < b[0]) r += 1;
    return r;
}
static int owner_in(const long *lok, int P, long idx) {          /* r with lok[r] <= idx < lok[r+1] */
    int lo = 0, hi = P;                                          /* first r with lok[r] > idx, minus 1 */
    while (lo < hi) { int mid = (lo + hi) / 2; if (lok[mid] > idx) hi = mid; else lo = mid + 1; }
    return lo - 1;
}

/* boundaries on layer n: cumulative work (1 + non-empty modes) sampled at Q colex quantiles */
static void choose_boundaries(slos_dist *d) {
    const int P = d->nproc, m = d->m, n = d->n;
    long M = fock_count(d, m, n);
    d->bnd = (u8 *)calloc((size_t)(P + 1) * m, 1);
    long Q = M < (1L << 16) ? M : (1L << 16); if (Q < 1) Q = 1;
    double *cum = (double *)malloc((Q + 1) * sizeof(double)); cum[0] = 0;
    u8 t[256];
    for (long q = 0; q < Q; q++) { long j = (long)((q + 0.5) * (double)M / (double)Q); if (j >= M) j = M - 1; if (j < 0) j = 0;
        colex_unrank(d, j, m, n, t); int nz = 0; for (int a = 0; a < m; a++) nz += t[a] > 0; cum[q + 1] = cum[q] + 1 + nz; }
    for (int r = 1; r < P; r++) {
        double target = cum[Q] * (double)r / (double)P;
        long lo = 1, hi = Q; while (lo < hi) { long mid = (lo + hi) / 2; if (cum[mid] >= target) hi = mid; else lo = mid + 1; }
        double frac = (target - cum[lo - 1]) / (cum[lo] - cum[lo - 1]);
        long j = (long)(((double)(lo - 1) + frac) * (double)M / (double)Q); if (j >= M) j = M - 1; if (j < 0) j = 0;
        colex_unrank(d, j, m, n, d->bnd + (long)r * m);
    }
    free(cum);
    d->lo = (long *)malloc((size_t)(n + 1) * (P + 1) * sizeof(long));
    for (int k = 0; k <= n; k++) { long *lok = d->lo + (long)k * (P + 1); lok[0] = 0; lok[P] = fock_count(d, m, k);
        for (int r = 1; r < P; r++) { long v = lo_of(d, k, d->bnd + (long)r * m); if (v < lok[r - 1]) v = lok[r - 1]; if (v > lok[P]) v = lok[P]; lok[r] = v; } }
}

/* ---------------------------------------------------------------- public API ---- */
slos_dist *slos_dist_create(int fcomm, int n, int m, const int *occ) {
    slos_dist *d = (slos_dist *)calloc(1, sizeof(slos_dist));
    d->comm = MPI_Comm_f2c((MPI_Fint)fcomm);
    MPI_Comm_rank(d->comm, &d->rank); MPI_Comm_size(d->comm, &d->nproc);
    d->n = n; d->m = m;
    d->occ = (int *)malloc(m * sizeof(int)); memcpy(d->occ, occ, m * sizeof(int));
    d->A = n + m + 1;
    d->binom = (long *)calloc((size_t)(d->A + 1) * (d->A + 1), sizeof(long));
    for (int a = 0; a <= d->A; a++) { d->binom[(long)a * (d->A + 1)] = 1; for (int b = 1; b <= a; b++)
        d->binom[(long)a * (d->A + 1) + b] = d->binom[(long)(a - 1) * (d->A + 1) + b - 1] + d->binom[(long)(a - 1) * (d->A + 1) + b]; }
    d->path = (int *)malloc((n > 0 ? n : 1) * sizeof(int)); { int idx = 0; for (int i = 0; i < m; i++) for (int t = 0; t < occ[i]; t++) d->path[idx++] = i; }
    d->denom = 1.0; for (int i = 0; i < m; i++) { double f = 1.0; for (int t2 = 2; t2 <= occ[i]; t2++) f *= t2; d->denom *= sqrt(f); }
    d->batch = 1L << 18;
    choose_boundaries(d);
    return d;
}
void slos_dist_destroy(slos_dist *d) { if (!d) return; free(d->occ); free(d->path); free(d->binom); free(d->bnd); free(d->lo); free(d); }
void slos_dist_set_batch(slos_dist *d, long batch) { d->batch = batch > 0 ? batch : (1L << 18); }
int slos_dist_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
long slos_dist_layer_size(const slos_dist *d, int k) { return fock_count(d, d->m, k); }
void slos_dist_chunk(const slos_dist *d, int k, long *lo, long *cnt) { const long *lok = d->lo + (long)k * (d->nproc + 1); *lo = lok[d->rank]; *cnt = lok[d->rank + 1] - lok[d->rank]; }
void slos_dist_chunk_of(const slos_dist *d, int k, int r, long *lo, long *cnt) { const long *lok = d->lo + (long)k * (d->nproc + 1); *lo = lok[r]; *cnt = lok[r + 1] - lok[r]; }
int slos_dist_owner(const slos_dist *d, int k, long idx) { return owner_in(d->lo + (long)k * (d->nproc + 1), d->nproc, idx); }
void slos_dist_boundary(const slos_dist *d, int r, u8 *t) { memcpy(t, d->bnd + (long)r * d->m, d->m); }
long slos_dist_rank(const slos_dist *d, int k, const u8 *t) { return colex_rank(d, t, d->m, k); }
void slos_dist_unrank(const slos_dist *d, int k, long j, u8 *t) { colex_unrank(d, j, d->m, k, t); }
void slos_dist_unrank_many(const slos_dist *d, int k, long n_idx, const long *idx, u8 *out) {
    for (long i = 0; i < n_idx; i++) colex_unrank(d, idx[i], d->m, k, out + i * d->m); }
void slos_dist_rank_many(const slos_dist *d, int k, long n_rows, const u8 *rows, long *out) {
    for (long i = 0; i < n_rows; i++) out[i] = colex_rank(d, rows + i * d->m, d->m, k); }

#define CH 2048L                                              /* rows per thread chunk */

/* Full run: layers 0..n on this rank's chunks; writes the normalised amplitudes of this rank's
 * chunk of layer n into `out` (length = chunk count). U is m x m row-major complex128 (U[i, p] =
 * amplitude from input mode p to output mode i). timings[4] = build, comm, compute (seconds), halo entries received. */
int slos_dist_run(slos_dist *d, const cplx *U, cplx *out, double *timings) {
    const int m = d->m, n = d->n, P = d->nproc, rank = d->rank; MPI_Comm comm = d->comm;
    double t_build = 0, t_comm = 0, t_compute = 0; long halo_total = 0;
    const long BATCH = d->batch;
    const int maxnz = (m < n ? m : n) > 0 ? (m < n ? m : n) : 1;
    double sq[256]; for (int v = 0; v < 256; v++) sq[v] = sqrt((double)v);
    long ilo = d->lo[rank], icnt = d->lo[rank + 1] - ilo;
    cplx *cin = (cplx *)malloc((icnt > 0 ? icnt : 1) * sizeof(cplx)); if (icnt > 0) cin[0] = 1.0;
    long *gcol = (long *)malloc((size_t)BATCH * maxnz * sizeof(long));
    u8 *gmode = (u8 *)malloc((size_t)BATCH * maxnz), *gval = (u8 *)malloc((size_t)BATCH * maxnz), *gcnt = (u8 *)malloc((size_t)BATCH);
    int segcap = 2 * (m + P) + 8; seg_t *segs = (seg_t *)malloc(segcap * sizeof(seg_t));
    long *iv = (long *)malloc(2 * m * sizeof(long));
    int *scnt = (int *)malloc(P * sizeof(int)), *sdisp = (int *)malloc(P * sizeof(int));
    int *rcnt = (int *)malloc(P * sizeof(int)), *rdisp = (int *)malloc(P * sizeof(int));
    int *dscnt = (int *)malloc(P * sizeof(int)), *dsdisp = (int *)malloc(P * sizeof(int));
    int *drcnt = (int *)malloc(P * sizeof(int)), *drdisp = (int *)malloc(P * sizeof(int));
    long *sreq = (long *)malloc(2 * (size_t)segcap * sizeof(long));
    long amin[256], amax[256];
    const long nchmax = (BATCH + CH - 1) / CH;
    long *cmin = (long *)malloc((size_t)nchmax * m * sizeof(long)), *cmax = (long *)malloc((size_t)nchmax * m * sizeof(long));
    for (int step = 0; step < n; step++) {
        const int p = d->path[step], k = step;
        const long *lok = d->lo + (long)k * (P + 1), *lok1 = d->lo + (long)(k + 1) * (P + 1);
        const long olo = lok1[rank], ocnt = lok1[rank + 1] - olo;
        cplx *cout = (step == n - 1) ? out : (cplx *)malloc((ocnt > 0 ? ocnt : 1) * sizeof(cplx));
        long nb_local = (ocnt + BATCH - 1) / BATCH; if (nb_local < 1) nb_local = 1;
        long nb; MPI_Allreduce(&nb_local, &nb, 1, MPI_LONG, MPI_MAX, comm);
        for (long b = 0; b < nb; b++) {
            long rlo = b * BATCH; if (rlo > ocnt) rlo = ocnt;
            long rhi = rlo + BATCH; if (rhi > ocnt) rhi = ocnt;
            const long nrows = rhi - rlo, nch = (nrows + CH - 1) / CH;
            /* pass 1: parents (absolute colex index in layer k) per row, per-mode index extent.
             * rank_k(t - e_i) = rank_{k+1}(t) - shift_i(t) with shift_i = F(i, S_i) + T_i,
             * T_i = sum_{j>i} F(j, S_j) - F(j, S_{j-1}) (S = prefix sums of t): along the odometer
             * the row rank grows by one and the shifts only change below a carry position. */
            double tb = MPI_Wtime();
            for (int i = 0; i < m; i++) { amin[i] = LONG_MAX; amax[i] = -1; }
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                u8 t[256]; long S[256], T[258], shift[256];
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                for (long c = 0; c < nch; c++) {
                    long r0 = rlo + c * CH, r1 = r0 + CH; if (r1 > rhi) r1 = rhi;
                    long *cmn = cmin + c * m, *cmx = cmax + c * m;
                    for (int i = 0; i < m; i++) { cmn[i] = LONG_MAX; cmx[i] = -1; }
                    long R = olo + r0; colex_unrank(d, R, m, k + 1, t);
                    int carry = m - 1;                                   /* recompute shifts for i <= carry */
                    for (long r = r0; r < r1; r++) {
                        if (r > r0) {                                    /* odometer + carry position */
                            if (t[0] > 0) { t[0]--; t[1]++; carry = -1; }
                            else { int j = 1; while (t[j] == 0) j++; u8 v = t[j]; t[j] = 0; t[j + 1]++; t[0] = (u8)(v - 1); carry = j; }
                            R++;
                        }
                        if (carry >= 0) {
                            long qs = 0; for (int a = 0; a <= carry + 1 && a < m; a++) { qs += t[a]; S[a] = qs; }
                            if (carry == m - 1) { T[m - 1] = 0; for (int a = m - 1; a >= 1; a--) T[a - 1] = T[a] + fock_count(d, a, S[a]) - fock_count(d, a, S[a - 1]); }
                            else { for (int a = carry + 1; a >= 1; a--) T[a - 1] = T[a] + fock_count(d, a, S[a]) - fock_count(d, a, S[a - 1]); }
                            for (int i = 0; i <= carry; i++) shift[i] = (i > 0 ? fock_count(d, i, S[i]) : 0) + T[i];
                            if (carry == m - 1) shift[m - 1] = fock_count(d, m - 1, S[m - 1]) + T[m - 1];
                        }
                        long base = (r - rlo) * maxnz; int c2 = 0;
                        for (int i = m - 1; i >= 0; i--) if (t[i] > 0) {
                            long g = R - shift[i];
                            gcol[base + c2] = g; gmode[base + c2] = (u8)i; gval[base + c2] = t[i]; c2++;
                            if (g < cmn[i]) cmn[i] = g; if (g > cmx[i]) cmx[i] = g; }
                        gcnt[r - rlo] = (u8)c2;
                    }
                }
            }
            for (long c = 0; c < nch; c++) for (int i = 0; i < m; i++) { if (cmin[c * m + i] < amin[i]) amin[i] = cmin[c * m + i]; if (cmax[c * m + i] > amax[i]) amax[i] = cmax[c * m + i]; }
            t_build += MPI_Wtime() - tb;
            /* halo: union of the per-mode intervals, split by owner, fetched as contiguous copies */
            double tcm = MPI_Wtime();
            int niv = 0;
            for (int i = 0; i < m; i++) if (amax[i] >= 0) { iv[2 * niv] = amin[i]; iv[2 * niv + 1] = amax[i] + 1; niv++; }
            for (int a = 1; a < niv; a++) { long s = iv[2 * a], e = iv[2 * a + 1]; int j = a - 1;      /* insertion sort by start */
                while (j >= 0 && iv[2 * j] > s) { iv[2 * j + 2] = iv[2 * j]; iv[2 * j + 3] = iv[2 * j + 1]; j--; } iv[2 * j + 2] = s; iv[2 * j + 3] = e; }
            int nu = 0;                                                                         /* merge */
            for (int a = 0; a < niv; a++) { if (nu > 0 && iv[2 * a] <= iv[2 * nu - 1]) { if (iv[2 * a + 1] > iv[2 * nu - 1]) iv[2 * nu - 1] = iv[2 * a + 1]; }
                else { iv[2 * nu] = iv[2 * a]; iv[2 * nu + 1] = iv[2 * a + 1]; nu++; } }
            int nseg = 0, nreq = 0; for (int q = 0; q < P; q++) scnt[q] = 0;
            for (int a = 0; a < nu; a++) { long s = iv[2 * a], e = iv[2 * a + 1]; int q = owner_in(lok, P, s);
                while (s < e) { long ee = lok[q + 1] < e ? lok[q + 1] : e;
                    if (ee > s) { if (nseg == segcap) { segcap *= 2; segs = (seg_t *)realloc(segs, segcap * sizeof(seg_t)); sreq = (long *)realloc(sreq, 2 * (size_t)segcap * sizeof(long)); }
                        segs[nseg].start = s; segs[nseg].len = ee - s; segs[nseg].owner = q; segs[nseg].base = (q == rank) ? cin + (s - ilo) : NULL; nseg++;
                        if (q != rank) { sreq[2 * nreq] = s; sreq[2 * nreq + 1] = ee - s; nreq++; scnt[q] += 2; } }
                    s = ee; q++; } }
            MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, comm);
            sdisp[0] = 0; rdisp[0] = 0; for (int q = 1; q < P; q++) { sdisp[q] = sdisp[q - 1] + scnt[q - 1]; rdisp[q] = rdisp[q - 1] + rcnt[q - 1]; }
            long totreq = rdisp[P - 1] + rcnt[P - 1];
            long *rreq = (long *)malloc((totreq > 0 ? totreq : 2) * sizeof(long));
            MPI_Alltoallv(sreq, scnt, sdisp, MPI_LONG, rreq, rcnt, rdisp, MPI_LONG, comm);
            long sendtot = 0, recvtot = 0;
            for (int q = 0; q < P; q++) { long c = 0; for (int e = rdisp[q]; e < rdisp[q] + rcnt[q]; e += 2) c += rreq[e + 1]; dscnt[q] = (int)c; dsdisp[q] = (int)sendtot; sendtot += c;
                c = 0; for (int e = sdisp[q]; e < sdisp[q] + scnt[q]; e += 2) c += sreq[e + 1]; drcnt[q] = (int)c; drdisp[q] = (int)recvtot; recvtot += c; }
            cplx *sendbuf = (cplx *)malloc((sendtot > 0 ? sendtot : 1) * sizeof(cplx));
            { long off = 0; for (long e = 0; e < totreq; e += 2) { memcpy(sendbuf + off, cin + (rreq[e] - ilo), rreq[e + 1] * sizeof(cplx)); off += rreq[e + 1]; } }
            cplx *halo = (cplx *)malloc((recvtot > 0 ? recvtot : 1) * sizeof(cplx));
            MPI_Alltoallv(sendbuf, dscnt, dsdisp, MPI_C_DOUBLE_COMPLEX, halo, drcnt, drdisp, MPI_C_DOUBLE_COMPLEX, comm);
            { long off = 0; for (int s2 = 0; s2 < nseg; s2++) if (segs[s2].owner != rank) { segs[s2].base = halo + off; off += segs[s2].len; } halo_total += off; }
            free(rreq); free(sendbuf);
            t_comm += MPI_Wtime() - tcm;
            /* pass 2: gather and multiply-add. Per chunk and mode, the parent range usually lies in
             * one segment (direct pointer); otherwise a cursor over the sorted segments. */
            double tc = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (long c = 0; c < nch; c++) {
                const cplx *ptr[256]; int cur[256];
                long r0 = rlo + c * CH, r1 = r0 + CH; if (r1 > rhi) r1 = rhi;
                for (int i = 0; i < m; i++) { ptr[i] = NULL; cur[i] = 0;
                    long lo_i = cmin[c * m + i], hi_i = cmax[c * m + i]; if (hi_i < 0) continue;
                    int s2 = 0; while (lo_i >= segs[s2].start + segs[s2].len) s2++;
                    cur[i] = s2; if (hi_i < segs[s2].start + segs[s2].len) ptr[i] = segs[s2].base - segs[s2].start; }
                for (long r = r0; r < r1; r++) { cplx acc = 0; long base = (r - rlo) * maxnz; int cnt = gcnt[r - rlo];
                    for (int e = 0; e < cnt; e++) { long g = gcol[base + e]; int i = gmode[base + e]; cplx val;
                        if (ptr[i]) val = ptr[i][g];
                        else { while (g >= segs[cur[i]].start + segs[cur[i]].len) cur[i]++; val = segs[cur[i]].base[g - segs[cur[i]].start]; }
                        acc += U[(long)i * m + p] * sq[gval[base + e]] * val; }
                    cout[r] = acc; }
            }
            t_compute += MPI_Wtime() - tc;
            free(halo);
        }
        free(cin); cin = cout; ilo = olo; icnt = ocnt;
    }
    if (n == 0) { if (icnt > 0) out[0] = cin[0]; free(cin); }
    for (long j = 0; j < icnt; j++) out[j] /= d->denom;
    free(gcol); free(gmode); free(gval); free(gcnt); free(segs); free(iv); free(sreq); free(cmin); free(cmax);
    free(scnt); free(sdisp); free(rcnt); free(rdisp); free(dscnt); free(dsdisp); free(drcnt); free(drdisp);
    if (timings) { timings[0] = t_build; timings[1] = t_comm; timings[2] = t_compute; timings[3] = (double)halo_total; }
    return 0;
}

/* Collective point lookup in layer k: every rank passes its own query list (global indices),
 * gets the values back. `chunk` is this rank's chunk of the layer (as returned by run). */
int slos_dist_get(slos_dist *d, int k, const cplx *chunk, long nq, const long *q, cplx *out) {
    const int P = d->nproc; MPI_Comm comm = d->comm;
    const long *lok = d->lo + (long)k * (P + 1); long lo = lok[d->rank];
    int *scnt = (int *)calloc(P, sizeof(int)), *sdisp = (int *)malloc(P * sizeof(int));
    int *rcnt = (int *)malloc(P * sizeof(int)), *rdisp = (int *)malloc(P * sizeof(int));
    long *order = (long *)malloc((nq > 0 ? nq : 1) * sizeof(long)); long *sorted = (long *)malloc((nq > 0 ? nq : 1) * sizeof(long));
    for (long i = 0; i < nq; i++) scnt[owner_in(lok, P, q[i])]++;
    sdisp[0] = 0; for (int p = 1; p < P; p++) sdisp[p] = sdisp[p - 1] + scnt[p - 1];
    { int *fill = (int *)calloc(P, sizeof(int));
      for (long i = 0; i < nq; i++) { int o = owner_in(lok, P, q[i]); long pos = sdisp[o] + fill[o]++; sorted[pos] = q[i]; order[pos] = i; }
      free(fill); }
    MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, comm);
    rdisp[0] = 0; for (int p = 1; p < P; p++) rdisp[p] = rdisp[p - 1] + rcnt[p - 1];
    long tot = rdisp[P - 1] + rcnt[P - 1];
    long *req = (long *)malloc((tot > 0 ? tot : 1) * sizeof(long));
    MPI_Alltoallv(sorted, scnt, sdisp, MPI_LONG, req, rcnt, rdisp, MPI_LONG, comm);
    cplx *resp = (cplx *)malloc((tot > 0 ? tot : 1) * sizeof(cplx));
    for (long e = 0; e < tot; e++) resp[e] = chunk[req[e] - lo];
    cplx *back = (cplx *)malloc((nq > 0 ? nq : 1) * sizeof(cplx));
    MPI_Alltoallv(resp, rcnt, rdisp, MPI_C_DOUBLE_COMPLEX, back, scnt, sdisp, MPI_C_DOUBLE_COMPLEX, comm);
    for (long i = 0; i < nq; i++) out[order[i]] = back[i];
    free(scnt); free(sdisp); free(rcnt); free(rdisp); free(order); free(sorted); free(req); free(resp); free(back);
    return 0;
}
