"""
Port of the exqalibur GPU Clifford & Clifford 2017 sampler (Clifford2017GPU / Utils.cu /
SubPermanentsCudaPar.cuh) to cupy RawModule (NVRTC), for benchmarking under the same
conditions as the other backends.  The kernels are transcribed one to one; only the
framework was dropped:

  original                                   here
  ------------------------------------------ ---------------------------------------------
  thrust::complex<T>                         double2 / float2 with inline helpers
  curand per block (shuffle + draws)         torch RNG: one permutation of the n rows per
                                             sample, one uniform per (sample, step)
  shuffle_rows kernel                        torch gather
  sample_from_amplitudes<true/false>         same (block per sample, m threads, thread 0
                                             does the serial cumulative sum and the draw)
  slice, cuda_sub_permanents_par,            same, byte for byte in the logic: one block per
  build_perm_vector                          sample, `nthreads` threads split the 2^(k-1) Gray
                                             terms into contiguous slices, per-thread rowsum /
                                             q arrays in global memory, per-thread partial
                                             cofactors reduced serially by thread 0
  generateSamplesLoop: batches of nBlocks    same: shots processed in chunks of `nblocks`
  (= SM count) samples, 128 threads each     (default SM count) samples, `nthreads` = 128

    occ = exq_sample(U, in_modes, shots, generator=gen)          # [shots, m] int64
"""
import torch

_SRC = r"""
#include <cuda_runtime.h>

template <typename T2> struct Cx;
template <> struct Cx<double2> {
    typedef double S;
    static __device__ __forceinline__ double2 make(double a, double b) { return make_double2(a, b); }
};
template <> struct Cx<float2> {
    typedef float S;
    static __device__ __forceinline__ float2 make(float a, float b) { return make_float2(a, b); }
};
template <typename T2> __device__ __forceinline__ T2 cadd(T2 a, T2 b) { return Cx<T2>::make(a.x + b.x, a.y + b.y); }
template <typename T2> __device__ __forceinline__ T2 csub(T2 a, T2 b) { return Cx<T2>::make(a.x - b.x, a.y - b.y); }
template <typename T2> __device__ __forceinline__ T2 cmul(T2 a, T2 b) { return Cx<T2>::make(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
template <typename T2> __device__ __forceinline__ T2 cscale(T2 a, typename Cx<T2>::S s) { return Cx<T2>::make(a.x * s, a.y * s); }
template <typename T2> __device__ __forceinline__ typename Cx<T2>::S cnorm(T2 a) { return a.x * a.x + a.y * a.y; }

// sample_from_amplitudes: block = sample, m threads.  FromFull: amplitudes = row 0 of the
// sample's n x m matrix; else amplitudes = the sample's permVector [m].
template <typename T2, bool FromFull>
__global__ void sample_from_amplitudes(const T2* amplitudes, const typename Cx<T2>::S* uniforms,
                                       unsigned int m, unsigned int n, unsigned int step,
                                       int* output, typename Cx<T2>::S* sampling_cumulated_weights)
{
    typedef typename Cx<T2>::S S;
    const unsigned int block_shift = m * blockIdx.x;
    if (FromFull) amplitudes += block_shift * n; else amplitudes += block_shift;
    sampling_cumulated_weights += block_shift;
    sampling_cumulated_weights[threadIdx.x] = cnorm(amplitudes[threadIdx.x]);
    __syncthreads();
    if (threadIdx.x == 0) {
        S sum = 0;
        for (int i = 0; i < m; i++) { sum += sampling_cumulated_weights[i]; sampling_cumulated_weights[i] = sum; }
        S r = uniforms[blockIdx.x * n + step] * sum;
        int mode = 0;
        while (r > sampling_cumulated_weights[mode] && mode < (int)m - 1) mode++;
        output[blockIdx.x * n + step] = mode;
    }
}

// slice: block = sample, modeLimit threads; slice[i][j] = us[i][modeSeq[j]], j < modeLimit-1
template <typename T2>
__global__ void slice(const T2* device_batch_us, const int* device_batch_modeSeq,
                      unsigned int m, unsigned int n, unsigned int modeLimit, T2* device_batch_slice)
{
    const unsigned int id = threadIdx.x, block_idx = blockIdx.x;
    for (unsigned int j = 0; j < modeLimit - 1; j++)
        device_batch_slice[block_idx * modeLimit * (modeLimit - 1) + id * (modeLimit - 1) + j] =
            device_batch_us[block_idx * n * m + id * m + device_batch_modeSeq[block_idx * n + j]];
}

// cuda_sub_permanents_par: A = (n+1) x n block per sample; p[block][i] = permanent of A without row i
// (Glynn, up to a constant factor), 2^n Gray terms split into blockDim.x contiguous slices.
template <typename T2>
__global__ void cuda_sub_permanents_par(const T2* A, int n, T2* p, T2* rowsum, T2* q, T2* shared_mem)
{
    typedef typename Cx<T2>::S S;
    T2* pshared = shared_mem + (size_t)blockIdx.x * (n + 1) * blockDim.x;
    unsigned int index = threadIdx.x;
    const unsigned long long twoPowerN = 1ull << n;
    unsigned long long slice_ = (twoPowerN + blockDim.x - 1) / blockDim.x;
    int m = n + 1;
    rowsum += ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * m;
    q += ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * m;
    T2* local_p = pshared + index * m;
    const T2* block_A = A + (size_t)blockIdx.x * (m * n);

    if (n == 1) {
        p[m * blockIdx.x + 0] = block_A[1];
        p[m * blockIdx.x + 1] = block_A[0];
        return;
    }
    for (int i = 0; i < m; i++) local_p[i] = Cx<T2>::make(0, 0);

    const unsigned long long start = index * slice_;
    if (start < twoPowerN) {
        const unsigned long long end = min(start + slice_, twoPowerN);
        unsigned long long j = start;
        unsigned long long gray = j ^ (j >> 1);
        S prodSign = (__popcll(gray) & 1) ? -1 : 1;

        for (int i = 0, base = 0; i < m; i++, base += n) {
            rowsum[i] = Cx<T2>::make(0, 0);
            unsigned long long s = gray;
            for (int k = 0; k < n; k++, s >>= 1)
                rowsum[i] = (s & 1) ? csub(rowsum[i], block_A[base + k]) : cadd(rowsum[i], block_A[base + k]);
            rowsum[i] = cscale(rowsum[i], (S)0.5);
        }
        T2 prev_value = Cx<T2>::make(1, 0);
        for (int i = 0; i < m; i++) { prev_value = cmul(prev_value, rowsum[i]); q[i] = prev_value; }

        while (true) {
            T2 t = Cx<T2>::make(prodSign, 0);
            for (int i = m - 1; i > 0; i--) {
                local_p[i] = cadd(local_p[i], cmul(t, q[i - 1]));
                t = cmul(t, rowsum[i]);
            }
            local_p[0] = cadd(local_p[0], t);

            if (++j == end) break;
            prodSign = -prodSign;
            unsigned long long gray_diff = (j ^ (j >> 1)) ^ gray;
            int k = __ffsll((long long)gray_diff) - 1;
            gray ^= (1ull << k);
            bool s = ((1ull << k) & gray);
            prev_value = Cx<T2>::make(1, 0);
            for (int i = 0, base = 0; i < m; i++, base += n) {
                rowsum[i] = s ? csub(rowsum[i], block_A[base + k]) : cadd(rowsum[i], block_A[base + k]);
                prev_value = cmul(prev_value, rowsum[i]); q[i] = prev_value;
            }
        }
    }
    __syncthreads();
    if (index == 0) {
        for (int i = 0; i < m; i++) {
            T2 acc = pshared[i];
            for (int k = 1; k < blockDim.x; k++) acc = cadd(acc, pshared[i + k * (n + 1)]);
            p[m * blockIdx.x + i] = acc;
        }
    }
}

// build_perm_vector: block = sample, m threads; permVector[i] = sum_j subPerm[j] * us[j][i], j < modeLimit
template <typename T2>
__global__ void build_perm_vector(const T2* device_batch_us, const T2* device_batch_subPerm,
                                  unsigned int modeLimit, unsigned int m, unsigned int n, T2* device_batch_permVector)
{
    const unsigned int id = threadIdx.x, block_idx = blockIdx.x;
    unsigned int base_permVector = block_idx * m, base_subPerm = block_idx * modeLimit;
    T2 acc = Cx<T2>::make(0, 0);
    for (unsigned int j = 0; j < modeLimit; j++)
        acc = cadd(acc, cmul(device_batch_subPerm[base_subPerm + j], device_batch_us[(size_t)block_idx * n * m + j * m + id]));
    device_batch_permVector[base_permVector + id] = acc;
}
"""

_mod = {}


def _module(t2):
    import cupy
    if t2 not in _mod:
        names = [f"sample_from_amplitudes<{t2}, true>", f"sample_from_amplitudes<{t2}, false>", f"slice<{t2}>",
                 f"cuda_sub_permanents_par<{t2}>", f"build_perm_vector<{t2}>"]
        mod = cupy.RawModule(code=_SRC, options=("-std=c++17",), name_expressions=names)
        _mod[t2] = {k: mod.get_function(k) for k in names}
    return _mod[t2]


def sm_count(dev=0):
    """Number of SMs on CUDA device `dev` — the default batch size (one block per sample)."""
    return torch.cuda.get_device_properties(dev).multi_processor_count


def exq_sample(U, in_modes, shots, generator=None, nblocks=None, nthreads=128, precision="double"):
    """Same semantics as sampler.fock_sample for a single circuit: occupations [shots, m] int64."""
    import cupy
    dev = U.device
    if U.dim() == 3:
        assert U.shape[0] == 1; U = U[0]
    m = U.shape[-1]; n = len(in_modes)
    t2 = "double2" if precision == "double" else "float2"
    cdt = torch.complex128 if precision == "double" else torch.complex64
    rdt = torch.float64 if precision == "double" else torch.float32
    ker = _module(t2)
    nblocks = nblocks or sm_count(dev.index or 0)
    Us = U[:, list(in_modes)].to(cdt).T.contiguous()                  # [n, m]: row = input photon
    out = torch.zeros(shots, n, dtype=torch.int32, device=dev)
    cp = cupy.asarray
    for b0 in range(0, shots, nblocks):
        B = min(nblocks, shots - b0)
        perm = torch.argsort(torch.rand(B, n, generator=generator, device=dev), dim=1)
        batch_us = torch.view_as_real(Us[perm].contiguous())                        # [B, n, m, 2]
        uni = torch.rand(B, n, generator=generator, device=dev, dtype=rdt).contiguous()
        modeSeq = torch.zeros(B, n, dtype=torch.int32, device=dev)
        slice_ = torch.zeros(B, n, n, 2, dtype=rdt, device=dev)
        subPerm = torch.zeros(B, n, 2, dtype=rdt, device=dev)
        permVec = torch.zeros(B, m, 2, dtype=rdt, device=dev)
        weights = torch.zeros(B, m, dtype=rdt, device=dev)
        rowsum = torch.zeros(B, nthreads, n + 1, 2, dtype=rdt, device=dev)
        q = torch.zeros_like(rowsum)
        shmem = torch.zeros(B, nthreads, n + 1, 2, dtype=rdt, device=dev)
        c_us, c_uni, c_seq, c_sl, c_sp, c_pv, c_w, c_rs, c_q, c_sh = map(
            cp, (batch_us, uni, modeSeq, slice_, subPerm, permVec, weights, rowsum, q, shmem))
        u32 = cupy.uint32
        ker[f"sample_from_amplitudes<{t2}, true>"]((B,), (m,), (c_us, c_uni, u32(m), u32(n), u32(0), c_seq, c_w))
        for modeLimit in range(2, n + 1):
            ker[f"slice<{t2}>"]((B,), (modeLimit,), (c_us, c_seq, u32(m), u32(n), u32(modeLimit), c_sl))
            ker[f"cuda_sub_permanents_par<{t2}>"]((B,), (nthreads,), (c_sl, cupy.int32(modeLimit - 1), c_sp, c_rs, c_q, c_sh))
            ker[f"build_perm_vector<{t2}>"]((B,), (m,), (c_us, c_sp, u32(modeLimit), u32(m), u32(n), c_pv))
            ker[f"sample_from_amplitudes<{t2}, false>"]((B,), (m,), (c_pv, c_uni, u32(m), u32(n), u32(modeLimit - 1), c_seq, c_w))
        out[b0:b0 + B] = modeSeq
    occ = torch.zeros(shots, m, dtype=torch.int64, device=dev)
    occ.scatter_add_(1, out.to(torch.int64), torch.ones_like(out, dtype=torch.int64))
    return occ


def available():
    """True when this backend can run: `cupy` is importable and a CUDA device is visible to torch."""
    try:
        import cupy  # noqa
        return torch.cuda.is_available()
    except Exception:
        return False
