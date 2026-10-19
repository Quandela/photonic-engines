"""
Single-sample path, optimised for latency: draw ONE sample as fast as possible.

Per photon step, three asynchronous launches and no host synchronisation:
    prep_single   (1 block)   multiplicities / radices / term count / normalisation of the
                              chosen rows + gathered block Bs = A[drow, sigma]     (CUDA)
    cof_multi     (fixed grid of `lanes` Gray lanes; the kernel uses ceil(N / 256) of them,
                   segment length computed on device, idle lanes write zeros)        (CUDA)
    torch.sum     over lanes (multi-block, asynchronous)
    _draw_kernel  (1 program) cofactors -> candidate weights -> draw                (Triton)
The batched sampler spends ~1.5 ms of host work per step at S = 1 (multiplicity tables,
lane planning, two synchronisations); this path removes it.

    occ = fock_sample_single(U, in_modes, generator, lanes=1 << 15)   # [m] int64
"""
import torch

from .cofactors_triton import _binom_table, draw_step
from .cuda_cofactors import _get_kernel, get_prep_kernel, tier


def fock_sample_single(U, in_modes, generator=None, lanes=1 << 15, block=128, seeds=None, sig=None):
    """Draw one Fock sample from U with the n input photons in `in_modes`; returns [m] int64 occupations.

    CUDA only (needs `cupy` and the Triton draw step). `lanes` is a fixed Gray-lane grid, sized once: the kernel uses
    only ceil(N / 256) of them and idle lanes write zeros. `sig` (row order) and `seeds` (per-step) are drawn from
    `generator` when not given; passing them makes the draw reproducible. No host synchronisation inside the loop.
    """
    import cupy
    dev = U.device
    m = U.shape[-1]
    n = len(in_modes)
    A = U[:, list(in_modes)].to(torch.complex128).contiguous()             # [m, n]
    A2 = torch.view_as_real(A)                                              # [m, n, 2]
    A2b = A2.view(1, m, n, 2)
    if sig is None:
        sig = torch.argsort(torch.rand(n, generator=generator, device=dev)).contiguous()
    if seeds is None:
        seeds = torch.randint(0, 2 ** 31 - 1, (n,), generator=generator, device=dev).tolist()
    rows = torch.zeros(n, dtype=torch.int64, device=dev)
    drow = torch.zeros(max(n, 1), dtype=torch.int64, device=dev)
    mult = torch.zeros(max(n, 1), dtype=torch.int32, device=dev)
    radix = torch.ones(max(n, 1), dtype=torch.int32, device=dev)
    N = torch.ones(1, dtype=torch.int64, device=dev)
    scale = torch.ones(1, dtype=torch.float64, device=dev)
    Bs = torch.zeros(n * n, 2, dtype=torch.float64, device=dev)            # [K1*K] double2 scratch
    part = torch.empty(lanes, n, 2, dtype=torch.float64, device=dev)
    ones = torch.zeros(1, 1, 2, dtype=torch.float64, device=dev); ones[0, 0, 0] = 1.0
    lane2s = torch.zeros(lanes, dtype=torch.int64, device=dev)
    zero1 = torch.zeros(1, dtype=torch.int64, device=dev)
    one1 = torch.ones(1, dtype=torch.int64, device=dev)
    unit = torch.ones(1, dtype=torch.float64, device=dev)
    dp = zero1
    binom = _binom_table(dev)
    prep = get_prep_kernel()
    cp = cupy.asarray
    c_A2, c_rows, c_sig, c_drow, c_mult, c_radix, c_N, c_scale, c_Bs, c_lane2s, c_zero1, c_binom, c_part = \
        (cp(A2), cp(rows), cp(sig), cp(drow), cp(mult), cp(radix), cp(N), cp(scale), cp(Bs), cp(lane2s), cp(zero1), cp(binom), cp(part))
    rows_b, sig_b = rows.view(1, n), sig.view(1, n)

    for k in range(1, n + 1):
        K1, K = k - 1, k
        if K1 == 0:
            draw_step(A2b, ones, rows_b, sig_b, dp, 1, m, K, zero1, one1, unit, 0, seeds[0], max_lanes=1)
            continue
        prep((1,), (block,), (c_rows, c_sig, c_A2, cupy.int32(K1), cupy.int32(K), cupy.int64(n),
                             c_drow, c_mult, c_radix, c_N, c_scale, c_Bs))
        KP, tpl = tier(K)
        kern = _get_kernel(KP)
        nthreads = lanes * tpl
        kern(((nthreads + block - 1) // block,), (block,), (
            c_Bs, c_mult, c_radix, c_lane2s, c_zero1, c_N, c_binom, c_part,
            cupy.int32(K1), cupy.int32(K), cupy.int64(0), cupy.int64(lanes)))      # L = 0: computed on device
        # the kernel wrote [lanes, K, 2] contiguously (idle lanes: zeros); reduce over lanes with a
        # multi-block torch sum (asynchronous), then draw from the single reduced row
        part_k = part.view(-1)[: lanes * K * 2].view(lanes, K, 2)
        red = part_k.sum(0, keepdim=True)
        draw_step(A2b, red, rows_b, sig_b, dp, 1, m, K, zero1, one1, scale, k - 1, seeds[k - 1], max_lanes=1)

    occ = torch.zeros(m, dtype=torch.int64, device=dev)
    occ.scatter_add_(0, rows, torch.ones_like(rows))
    return occ
