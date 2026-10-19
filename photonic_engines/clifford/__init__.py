"""Clifford & Clifford exact boson sampler (arXiv:1706.01260, arXiv:2005.04214) on GPU.

  sample / fock_sample / emitter_sample : entry points (input type -> implementation, `backend=`)
  sampler           `fock_sample`: B circuits x shots streams in lockstep, Triton or CUDA
  single            `fock_sample_single`: latency path for one sample (cupy / NVRTC)
  cofactors_triton  Glynn / Gray-code cofactor kernels (2018 and 2020 forms) and the draw step
  cuda_cofactors    thread-per-lane CUDA version of the 2020 kernel (fastest for n >= 16)
  exq_sampler       port of the exqalibur GPU Clifford-2017 sampler (benchmark reference)
  emitter_reference exact sampler for EmitterMPS inputs: generalized chain rule with reduced
                    states, gap-Kraus decomposition, matrix cofactors (small-n oracle)
  emitter_sampler   `sample_torch`: the same algorithm batched over streams in torch
                    (CPU/GPU, no custom kernels) - the mid-n oracle and the kernel spec
  emitter_terms     batched gap-Kraus term decomposition (all streams at once)
  product_sampler   `product_sample`: the chain rule for ProductSuperposition inputs (superposition
                    boson sampling) -- the true marginal per step, by residual-set enumeration or
                    a doubled Glynn sum, whichever is cheaper (docs/SUPERPOSITION.md §5)
  emitter_triton    `sample_triton`: Triton cofactor kernels ("seq": generated per K, block
                    chains + 2020 collapse; "scan": associative scans), lanes = (term, segment)

Algorithms and measurements: docs/ALGORITHMS.md (Fock), docs/EMITTER_SAMPLER.md (emitter), docs/SUPERPOSITION.md.
Fock sampling requires triton (imported on first use); `backend="cuda"` and the single-sample
path additionally need cupy. `emitter_sample(backend="torch")` runs with torch alone.
"""
from __future__ import annotations

import time as _time
from typing import Optional

import torch

from ..states import EmitterMPS, FockState, FockSuperposition, ProductSuperposition

__all__ = ["emitter_sample", "exq_sample", "fock_sample", "fock_sample_single", "sample"]


def emitter_sample(state: EmitterMPS, U: torch.Tensor, shots: int, generator: torch.Generator | None = None,
                   backend: str = "triton", device: str | None = None, kernel: str = "auto",
                   return_stats: bool = False, loss=None, unravel: bool = True, indist=None, **kw):
    """Exact samples [shots, m] (int64 occupations) of an EmitterMPS input through U.
    backend: "triton" (the cofactor kernels, CUDA; `kernel` = "auto" (default since 2026-10-02: the CUDA
    thread-per-lane kernel `cuda_emitter` at the frontier steps, the Triton seq kernel elsewhere, docs/EMITTER_SAMPLER.md
    §8; without cupy it is "seq") | "seq" | "scan" (Triton) | "cuda"), "torch" (batched
    torch, CPU or CUDA — the reference, usable to n ~ 14) or "segments" (torch, caterpillar
    states only: segment-factorised terms, 2.5x "torch" at n = 16, docs/EMITTER_SAMPLER.md §7). Extra keyword arguments go to the
    implementation (`target_total_lanes`, `seg`, `num_warps` / `delta_chunk`).
    loss: uniform transmission eta (docs/LOSS.md), applied on the input side: each stream draws
    its surviving emission slots, the lost ones are traced out (the samplers' reduced-state
    machinery does this exactly), and streams are grouped by surviving count k so that every
    group is one lockstep run of the lossless sampler with k <= n photons. Backends triton and
    torch take the label unravelling of the trace (`emitter_terms.sample_lost_labels`,
    docs/LOSS.md §3b): the lost slots' labels are drawn first, the kept photons are then sampled
    from a pure conditioned state — one term per gap instead of up to chi^2, so a lossy sample
    costs what a k-photon lossless one costs; `unravel=False` keeps the traced gap terms.
    indist: partial distinguishability, x-model (docs/DISTINGUISHABILITY.md): each photon is,
    independently, distinguishable from every other with probability 1 - x; such a photon is
    placed first (its output row drawn from the exact marginal, `emitter_terms.sample_slot_outcomes`)
    and folded into the chain at that row, so the chain rule runs on the g ~ Bin(n, eta x)
    interfering photons only — the same cost law as loss with eta -> eta x. Backends triton and
    torch.
    """
    if backend == "triton":
        from .emitter_triton import sample_triton
        impl = lambda s, k, **a: sample_triton(state, U, s, device=device or "cuda", impl=kernel, keep=k, **a, **kw)
    elif backend == "torch":
        from .emitter_sampler import sample_torch
        impl = lambda s, k, **a: sample_torch(state, U, s, device=device or "cpu", keep=k, **a, **kw)
    elif backend == "segments":
        from .emitter_caterpillar import sample_caterpillar
        impl = lambda s, k, **a: sample_caterpillar(state, U, s, device=device or "cpu", keep=k, **a, **kw)
    else:
        raise ValueError(f"unknown backend {backend!r} (triton | torch | segments)")
    from ..imperfections import LOST, SINGLE, draw_roles, spec
    d = spec(indist)
    lossless = loss is None or float(loss) >= 1.0
    if lossless and d is None:
        return impl(shots, None, generator=generator, return_stats=return_stats)
    if d is not None and d.x is None:
        raise NotImplementedError("emitter_sample: partial distinguishability is the x-model only (several "
                                  "interfering species would need a multi-species chain rule)")
    if d is not None and backend == "segments":
        raise NotImplementedError("emitter_sample: no partial distinguishability on the segments backend (triton | torch)")
    from .emitter_terms import sample_slot_outcomes
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    unravel = unravel and backend in ("triton", "torch")
    roles = draw_roles(state.n, shots, loss, d, generator=generator)                # [shots, n]: 0 lost, 1 single, 2 good
    good = roles >= 2
    # the outcomes folded before the chain rule: lost labels (when unravelling) and singleton
    # rows (always: a singleton cannot be expressed as a traced gap term); a lost slot left
    # traced is summed over like a pending one (label -1, not in `keep`)
    if unravel or bool((roles == SINGLE).any()):
        labels, fid = sample_slot_outcomes(state, torch.where(roles == LOST, 0 if unravel else 2, roles), U, generator=generator)
    else:
        labels = fid = None
    occ = torch.zeros((shots, U.shape[0]), dtype=torch.long)
    stats = {"groups": {}, "passes": []}       # passes: (k, streams, seconds) per surviving count, as fock_sample's
    out = None
    dev_t = torch.device(device or ("cuda" if backend == "triton" else "cpu"))
    for k in good.sum(1).unique().tolist():
        ids = (good.sum(1) == k).nonzero(as_tuple=True)[0]
        if k == 0:
            continue
        keep = torch.arange(state.n).expand(ids.shape[0], -1)[good[ids]].view(-1, k)   # [S_k, k] sorted
        extra = {} if labels is None else {"lost_labels": labels[ids], "final_ids": fid[ids]}
        if return_stats and dev_t.type == "cuda":
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        out = impl(ids.shape[0], keep, generator=generator, return_stats=return_stats, **extra)
        if return_stats:
            out, stats["groups"][k] = out
            if dev_t.type == "cuda":
                torch.cuda.synchronize()
            stats["passes"].append((k, int(ids.shape[0]), _time.perf_counter() - t0))
        occ[ids] = out.cpu()
    if labels is not None:                                                             # singleton photons: their drawn rows
        s_, j_ = (roles == SINGLE).nonzero(as_tuple=True)
        occ.index_put_((s_, labels[s_, j_] - state.num_labels), torch.ones_like(s_), accumulate=True)
    occ = occ.to(out.device) if out is not None else occ
    return (occ, stats) if return_stats else occ


def product_sample(state, U: torch.Tensor, shots: int, generator: torch.Generator | None = None, **kw):
    """Exact samples [shots, m] of a `ProductSuperposition` input (`product_sampler.sample`).

    The chain rule marches the TRUE marginal of the rows placed so far, so it is exact by
    construction; `method="auto" | "replica" | "residual"` picks how each marginal is evaluated,
    `photons=k` samples the post-selected k-photon sector instead of drawing the photon number,
    `indist=x` partial distinguishability per source (a lone source's photon is routed classically),
    `loss=eta` puts a uniform transmission on the input side (docs/LOSS.md §8: a lossy
    partially-excited source is a mixture of vacuum and a source at a smaller alpha, so the lossy
    sampler is the lossless one under a Bernoulli mask). Cost per step is in
    docs/SUPERPOSITION.md §5 -- the 4^j of the replica branch is what bounds the reach, not memory.
    """
    from .product_sampler import sample as _product_sample
    return _product_sample(state, U, shots, generator=generator, **kw)


def fock_sample(U, in_modes, shots, generator=None, **kw):
    """Exact samples for a Fock input (`sampler.fock_sample`; needs triton, imported on first
    call so that the torch-only emitter path works without it). `loss=eta`: uniform loss,
    applied on the input side by grouping the shots (docs/LOSS.md).
    """
    from .sampler import fock_sample as _fock_sample
    return _fock_sample(U, in_modes, shots, generator=generator, **kw)


def sample(state, U: torch.Tensor, shots: int, generator: torch.Generator | None = None, **kw):
    """Exact samples [shots, m] for any input state type: FockState -> `fock_sample`
    (backend "triton" | "cuda"), EmitterMPS -> `emitter_sample` (backend "triton" | "torch");
    `loss=eta` for uniform loss on either.
    ProductSuperposition -> `product_sample` (the chain rule of docs/SUPERPOSITION.md §5; exact, and
    not bounded by strong simulation the way `slos.ProductSLOSKernel.sample` is). A general
    FockSuperposition still has no sampler: drawing one component and then sampling it would give
    the MIXTURE, not the superposition.
    """
    if isinstance(state, FockState):
        return fock_sample(U, state.modes(), shots, generator=generator, **kw)
    if isinstance(state, EmitterMPS):
        return emitter_sample(state, U, shots, generator=generator, **kw)
    if isinstance(state, ProductSuperposition):
        return product_sample(state, U, shots, generator=generator, **kw)
    if isinstance(state, FockSuperposition):
        raise NotImplementedError("exact sampling of a coherent superposition of Fock states is not implemented: drawing one "
                                  "component and sampling it gives the mixture; use the strong simulation (slos)")
    raise TypeError(f"unsupported input state {type(state).__name__}")


def __getattr__(name):
    if name == "fock_sample_single":
        from .single import fock_sample_single
        return fock_sample_single
    if name == "sample_triton":
        from .emitter_triton import sample_triton
        return sample_triton
    if name == "sample_torch":
        from .emitter_sampler import sample_torch
        return sample_torch
    if name == "exq_sample":
        from .exq_sampler import exq_sample
        return exq_sample
    raise AttributeError(name)
