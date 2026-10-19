"""
Batched Clifford & Clifford exact boson sampler, fully on device.

    U        : [B, m, m] complex128 (B distinct circuits) or [m, m]
    in_modes : list of n input modes (with multiplicity)
    shots    : samples per circuit
    backend  : "triton" | "cuda" — kernel used for the 2020 steps (cuda = thread-per-lane, cupy/NVRTC)
               "exq"   — exqalibur GPU Clifford 2017 port (exq_sampler.py), benchmarking reference
    algo     : "2018"  — arXiv:1706.01260, Glynn/Gray sweep over 2^(k-2) terms per step
               "2020"  — arXiv:2005.04214, same chain rule but the cofactors of the block of
                         already-chosen rows exploit repeated rows (collisions): prod(s_nu + 1)
                         terms instead of 2^(k-1), mixed-radix Gray code.  Expected cost over
                         Haar U ~ n rho^n with rho = 1.69 (m = n), 1.81 (m = 2n).
               "auto"  — 2018 kernel for k-1 <= multi_min_k1 (few terms, no host preprocessing),
                         2020 kernel above.
               "input" — the INPUT-side collapse (Brod & Oszmaniec arXiv:1906.06696 / theboss):
                         the sweep signs the input columns, prod over their multiplicities, i.e.
                         2^(k-1) terms on a collision-free input (cofactors_input.py). Triton only.
               "percof" — PiquassoBoost's generalised C&C-B (Rakyta & Morse arXiv:2309.07027):
                         every cofactor by its own output-collapsed sweep, k sweeps per step
                         instead of one (the 2020 kernel launched k times, column c kept from
                         launch c: bit-identical samples at k x the work).
               The T1 / F2a algorithm comparison (docs/ALGORITHMS.md) runs these four on the
               same substrate (backend "triton"); "auto"/"cuda" is the production frontier.
    -> occupations [B*shots, m] int64   (stream s = b*shots + shot)

    prefix   : output modes the FIRST c photons of the chain are forced onto (adaptive linear optics,
               docs/ADAPTIVE.md §4b). The chain is exact given the SET of input columns its first k
               steps have used, r_{<=k} | S ~ |per(A[r, S])|^2 / k!  with S uniform, so conditioning on
               the first c rows is: draw S with weight |per(A[prefix, S])|^2 among the C(n, c) subsets,
               then run steps c+1..n as usual. S is drawn column by column (`prefix_columns`), polynomial
               in n and c because the prefix has few DISTINCT modes; nothing is enumerated.
               The returned occupations INCLUDE the prefix photons.
    avoid    : output modes a chain must not visit AFTER its prefix (adaptive linear optics: "no further photon
               on the adaptive modes"). A stream is dropped the moment a photon lands there, so a chain that
               would be rejected costs its steps up to that photon only - and since the position of the first
               such photon is uniform among the remaining steps while the cost grows like 2^k, an aborted chain
               costs ~1/((n-r) ln 2) of a full one. Only the surviving streams are returned: [S', m] with S' <= shots.

A single sample (B*shots = 1, backend "cuda") takes the latency path of single.py: three
asynchronous launches per step, no host work between them.

Per step k = 1..n, all S = B*shots streams in lockstep, two kernel launches
(plus, for the 2020 form, a few small torch ops building the multiplicity tables):
    kernel 1  cofactors  P[s, c] = per(A[dp(s)][rows[s, :k-1], sigma[s, :k]] \\ column c)
    kernel 2  reduce partials, amp = A[dp, :, sigma[:k]] @ P, w = |amp|^2, draw rows[s, k-1]
"""
import math
import time as _time
from itertools import combinations

import torch

from .cofactors_triton import _plan, cofactor_partials, cofactor_partials_multi, draw_step


def fock_sample(U, in_modes, shots, generator=None, algo="auto", multi_min_k1=10,
                    max_lanes=1 << 16, target_total_lanes=1 << 17, backend="triton", single_lanes=1 << 15,
                    max_streams=1 << 18, loss=None, indist=None, prefix=None, avoid=None, keep_ids=False, stats=None):
    """backend: "triton" or "cuda" (thread-per-lane CUDA kernel via cupy) for the 2020 steps.
    With backend="cuda" and a single sample (B*shots == 1) the latency path of single.py is used
    (single_lanes Gray lanes per step, no host synchronisation).
    max_streams: streams processed in lockstep per pass; larger requests are split into passes
    (the per-stream gathered block Bs[S, k-1, k] complex128 is ~10 kB per stream at k = 24, so
    2^18 streams keep it below ~3 GB; a pass of 2^18 streams already saturates the GPU).
    loss: uniform transmission eta (docs/LOSS.md). Applied on the INPUT side: every stream draws
    its surviving photons; streams are grouped by surviving count k, the distinct surviving
    subsets of a group entering as distinct circuits (column sets of U, the B axis the kernels
    already index per stream), so every k is one lockstep pass of the lossless sampler on a
    k-photon instance: the kernels are unchanged and the expected cost is below the lossless one.
    indist: partial distinguishability (docs/DISTINGUISHABILITY.md): x (pairwise overlap, V_HOM =
    x^2) or an `imperfections.Distinguishability`. Same mechanism: the interfering photons of a
    stream are its instance, the others are routed classically; cost E[rho^g], g ~ Bin(n, eta x).
    prefix: the forced first output rows of every chain (docs/ADAPTIVE.md §4b): a list of r output
    modes shared by all streams, or an int tensor [B, r] with one prefix per circuit of U [B, m, m]
    (the same r for all; a heterogeneous batch is grouped by (n, r) by the caller). avoid: output
    modes on which a later photon aborts the chain; only the surviving streams are returned, and
    with keep_ids=True the pair (occ, stream ids) so a caller can tell which streams survived.
    shots: an int (per circuit), or an int tensor [B] of per-circuit stream counts (lossless,
    indistinguishable photons only) - streams b*shots.. are those of circuit b either way.
    stats: a dict that receives, under "passes", one (k, streams, seconds) triple per lockstep pass
    (seconds synchronised on cuda): under loss / indist every surviving count k is one pass, so the
    pass time over its streams is the measured cost of a k-photon instance - the stratum mean that the
    run-of-S statistic needs (gaussian/cost.py, docs/GAUSSIAN.md §2.1) - at no cost to the samples.
    """
    from ..imperfections import spec
    indist = spec(indist)
    if prefix is not None:
        prefix = torch.as_tensor(prefix, dtype=torch.int64)
        if prefix.dim() == 2 and prefix.shape[0] == 1 and U.dim() == 2:
            prefix = prefix[0]
        if prefix.numel() == 0:
            prefix = None
    avoid = [int(r) for r in avoid] if avoid is not None and len(avoid) else None
    if prefix is not None or avoid is not None:
        if backend == "exq" or indist is not None or not (loss is None or float(loss) >= 1.0):
            raise NotImplementedError("prefix / avoid: lossless, indistinguishable photons, triton or cuda backend only")
        if prefix is not None:
            if prefix.dim() == 2 and prefix.shape[0] != (U.shape[0] if U.dim() == 3 else 1):
                raise ValueError(f"prefix [{prefix.shape[0]}, r]: one row per circuit ({U.shape[0] if U.dim() == 3 else 1})")
            if prefix.shape[-1] > len(in_modes) or bool((prefix < 0).any()) or bool((prefix >= U.shape[-2]).any()):
                raise ValueError(f"prefix: at most n = {len(in_modes)} output modes of the circuit")
            prefix = prefix.to(U.device)
        if avoid is not None and any(not 0 <= r < U.shape[-2] for r in avoid):
            raise ValueError(f"avoid {avoid}: output modes of the circuit")
    if backend not in ("triton", "cuda", "exq"):
        raise ValueError(f'unknown backend {backend!r} for a Fock input: "triton", "cuda" or "exq" '
                         f'("torch" exists on the emitter sampler only)')
    if algo not in ("auto", "2018", "2020", "input", "percof"):
        raise ValueError(f'unknown algo {algo!r}: "auto", "2018", "2020", "input" or "percof"')
    if algo == "input" and backend != "triton":
        raise NotImplementedError('algo="input": Triton kernel only (backend="triton")')
    if U.dim() == 2:
        U = U[None]
    Bn, m, _ = U.shape
    n = len(in_modes)
    dev = U.device
    reps = None
    if isinstance(shots, torch.Tensor):
        reps = shots.to(dev, torch.int64).reshape(-1)
        if reps.shape[0] != Bn or bool((reps < 0).any()):
            raise ValueError(f"shots [{reps.shape[0]}]: one non-negative count per circuit ({Bn})")
        if backend == "exq" or indist is not None or not (loss is None or float(loss) >= 1.0):
            raise NotImplementedError("per-circuit shots: lossless, indistinguishable photons, triton or cuda backend only")
        S = int(reps.sum())
    else:
        S = Bn * shots
    if backend == "exq":
        # port of the exqalibur GPU Clifford 2017 sampler (block per sample, 128 threads), for benchmarks
        from ..imperfections import thin_counts
        from .exq_sampler import exq_sample
        assert Bn == 1, "exq backend: one circuit at a time"
        if indist is not None:
            raise NotImplementedError("exq backend: no partial distinguishability (use triton or cuda)")
        return thin_counts(exq_sample(U[0], in_modes, shots, generator=generator), loss, generator=_cpu_generator(generator))
    lossless = loss is None or float(loss) >= 1.0
    if S == 1 and backend == "cuda" and algo in ("auto", "2020") and lossless and indist is None and prefix is None and avoid is None:
        # one sample: latency path (device-side step preparation, fixed lane grid, no host sync)
        from .single import fock_sample_single
        return fock_sample_single(U[0], in_modes, generator=generator, lanes=single_lanes)[None]
    if backend == "cuda":
        from .cuda_cofactors import cofactor_partials_cuda
        multi = cofactor_partials_cuda
    else:
        multi = cofactor_partials_multi
    dp_all = torch.arange(Bn, device=dev, dtype=torch.int64).repeat_interleave(reps if reps is not None else shots)   # [S]
    occ = torch.zeros(S, m, dtype=torch.int64, device=dev)
    alive = torch.ones(S, dtype=torch.bool, device=dev)                              # streams not aborted (avoid)
    avoid_mask = None
    if avoid is not None:
        avoid_mask = torch.zeros(m, dtype=torch.bool, device=dev); avoid_mask[avoid] = True

    def run(A, dp, ids):
        """One lockstep pass per max_streams chunk: A [G, m, k] input columns (one entry per
        (circuit, surviving subset)), dp[s] the entry of stream s, ids the stream numbers.
        """
        k = A.shape[2]
        if k == 0:
            return
        if stats is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = _time.perf_counter()
        A2 = torch.view_as_real(A.to(torch.complex128).contiguous())                # [G, m, k, 2]
        for c0 in range(0, ids.shape[0], max_streams):
            sel = ids[c0:c0 + max_streams]
            rows, kept = _sample_streams(A2, dp[c0:c0 + max_streams].contiguous(), k, m, generator, algo, multi_min_k1,
                                         max_lanes, target_total_lanes, multi, prefix=prefix, avoid=avoid_mask)
            if kept is not None:
                dead = torch.ones(sel.shape[0], dtype=torch.bool, device=dev); dead[kept] = False
                alive[sel[dead]] = False; sel = sel[kept]
            occ[sel] = occ[sel].scatter_add(1, rows, torch.ones_like(rows))
        if stats is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            stats.setdefault("passes", []).append((k, int(ids.shape[0]), _time.perf_counter() - t0))

    if lossless and indist is None:
        run(U[:, :, list(in_modes)], dp_all, torch.arange(S, device=dev, dtype=torch.int64))
        if avoid is None:
            return (occ, torch.arange(S, device=dev, dtype=torch.int64)) if keep_ids else occ
        return (occ[alive], torch.nonzero(alive)[:, 0]) if keep_ids else occ[alive]
    # input-side loss and distinguishability: every stream draws its partition (lost photons,
    # interfering species, singletons); the streams of one species instance are grouped by photon
    # count k and, within a group, the distinct column sets are distinct "circuits", so one
    # lockstep pass per k. Singletons (a photon in a mode of its own, or a species of one photon)
    # interfere with nothing and are routed on the host with |U[r, col]|^2.
    from ..imperfections import partition, route_classical
    gen = _cpu_generator(generator)
    by_cols: dict = {}                                                            # cols tuple -> [stream ids]
    single_ids, single_cols = [], []
    for subsets, singles, ids in partition(n, S, loss, indist, generator=gen):
        for sub in subsets:
            cols = tuple(in_modes[j] for j in sub)
            if len(cols) == 1:
                single_ids.append(ids); single_cols.append(torch.full_like(ids, cols[0]))
            else:
                by_cols.setdefault(cols, []).append(ids)
        for j in singles:
            single_ids.append(ids); single_cols.append(torch.full_like(ids, in_modes[j]))
    by_k: dict = {}
    for cols, ids_l in by_cols.items():
        by_k.setdefault(len(cols), []).append((cols, torch.cat(ids_l)))
    for k, groups in by_k.items():
        G = len(groups)
        cols = torch.tensor([list(c) for c, _ in groups], device=dev, dtype=torch.int64)   # [G, k]
        A = U[:, :, cols].permute(0, 2, 1, 3).reshape(Bn * G, m, k)                # entry b*G + g
        ids = torch.cat([g for _, g in groups]).to(dev)
        gidx = torch.cat([torch.full((g.shape[0],), gi, dtype=torch.int64) for gi, (_, g) in enumerate(groups)]).to(dev)
        run(A, dp_all[ids] * G + gidx, ids)
    if single_ids:
        ids = torch.cat(single_ids); cols = torch.cat(single_cols)
        rows = route_classical(U, dp_all[ids.to(dev)].cpu(), cols, generator=gen).to(dev)
        occ.index_put_((ids.to(dev), rows), torch.ones_like(rows), accumulate=True)
    return occ


def _cpu_generator(generator):
    """The host-side generator for the loss draws: `generator` itself when it lives on the CPU,
    else a CPU generator seeded from it (so results stay reproducible per seed).
    """
    if generator is None or generator.device.type == "cpu":
        return generator
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, device=generator.device).item())
    return torch.Generator().manual_seed(seed)


def _ryser(M: torch.Tensor) -> torch.Tensor:
    """Permanents of the trailing [c, c] blocks of M by Ryser's formula, O(2^c c^2) each: [...]. c = 0 gives 1."""
    c = M.shape[-1]
    if c == 0:
        return torch.ones(M.shape[:-2], dtype=M.dtype, device=M.device)
    total = torch.zeros(M.shape[:-2], dtype=M.dtype, device=M.device)
    for mask in range(1, 1 << c):
        cols = [j for j in range(c) if mask >> j & 1]
        total = total + (-1.0) ** (c - len(cols)) * M[..., cols].sum(-1).prod(-1)
    return total


def prefix_subset_weights(A: torch.Tensor, prefix) -> tuple[torch.Tensor, torch.Tensor]:
    """The posterior of the column SET used by the first c steps, given that they output `prefix`, ENUMERATED:
    the oracle of `prefix_columns` (C(n, c) permanents; tests and small cases only).

    A [G, m, n] input columns; returns (subsets [C, c] int64, weights [G, C] float64) with
    weights[g, j] = |per(A[g][prefix, subsets[j]])|^2. Their sum over j is c! C(n, c) times the
    probability that the first c photons of the chain read `prefix`.
    """
    n, c = A.shape[2], len(prefix)
    sub = torch.tensor(list(combinations(range(n), c)), dtype=torch.int64, device=A.device).reshape(-1, c)   # [C, c]
    blocks = A[:, prefix][:, :, sub].permute(0, 2, 1, 3)                            # [G, C, c, c]
    per = _ryser(blocks.to(torch.complex128))
    return sub, per.real ** 2 + per.imag ** 2


def _shift(T: torch.Tensor, axis: int) -> torch.Tensor:
    """T moved one step up along `axis` (multiplication by that variable), the top degree dropped."""
    out = torch.zeros_like(T)
    src = [slice(None)] * T.dim(); dst = [slice(None)] * T.dim()
    src[axis] = slice(0, T.shape[axis] - 1); dst[axis] = slice(1, T.shape[axis])
    out[tuple(dst)] = T[tuple(src)]
    return out


def prefix_columns(A: torch.Tensor, prefix, shots: int, generator=None) -> torch.Tensor:
    """`shots` column sets S, |S| = c, drawn with probability prop. to |per(A[prefix, S])|^2: bool [shots, n].

    A [m, n] input columns of ONE circuit. With the prefix's distinct modes b_1..b_L and multiplicities p,
    per(A[prefix, S]) = p! [x^p] prod_{j in S} (sum_l A[b_l, j] x_l), so the weight of S is the coefficient of
    x^p w^p in prod_{j in S} (B_j . x)(conj(B_j) . w). Columns are decided one at a time, in order: with
        F_j[a, b] = [x^a w^b] prod_{j' >= j} (1 + (B_j' . x)(conj(B_j') . w))        suffix tables, [n+1, D, D]
        g[a]      = [x^a] prod_{j' in S, j' < j} (B_j' . x)                          per shot, [shots, D]
    the mass of all completions is sum_{a, b} g[a] conj(g[b]) F_j[p - a, p - b], evaluated with and without
    column j. D = prod(p_l + 1): cost O(n L^2 D^2 + shots n D^2), polynomial - no C(n, c) enumeration.
    """
    A = A.to(torch.complex128)
    n, dev = A.shape[1], A.device
    modes = sorted(set(prefix))
    p = [list(prefix).count(b) for b in modes]
    L, dims = len(modes), [c + 1 for c in p]
    D = math.prod(dims)
    B = A[modes]                                                                    # [L, n]
    F = [None] * (n + 1)
    F[n] = torch.zeros(dims + dims, dtype=torch.complex128, device=dev)
    F[n][(0,) * (2 * L)] = 1.0
    for j in range(n - 1, -1, -1):
        nxt = F[j + 1].clone()
        for a in range(L):
            Fa = _shift(F[j + 1], a)
            for b in range(L):
                nxt += (B[a, j] * B[b, j].conj()) * _shift(Fa, L + b)
        F[j] = nxt
    flip = list(range(2 * L))
    g = torch.zeros([shots, *dims], dtype=torch.complex128, device=dev)
    g[(slice(None), *([0] * L))] = 1.0
    chosen = torch.zeros(shots, n, dtype=torch.bool, device=dev)

    def mass(gv, Fj):                                                               # [shots]: g^T Fr conj(g), real
        Fr = torch.flip(Fj, flip).reshape(D, D)
        v = gv.reshape(shots, D)
        return torch.einsum("sa,ab,sb->s", v, Fr, v.conj()).real.clamp_min(0.0)

    for j in range(n):
        g_in = torch.zeros_like(g)
        for a in range(L):
            g_in += B[a, j] * _shift(g, 1 + a)
        w_in, w_out = mass(g_in, F[j + 1]), mass(g, F[j + 1])
        take = torch.rand(shots, generator=generator, device=dev, dtype=torch.float64) * (w_in + w_out) < w_in
        chosen[:, j] = take
        g = torch.where(take.reshape([shots] + [1] * L), g_in, g)
    return chosen


def prefix_columns_many(A: torch.Tensor, prefixes, dp: torch.Tensor, generator=None) -> torch.Tensor:
    """`prefix_columns` for many circuits at once: bool [S, n], stream s drawing a column set of circuit dp[s].

    A [G, m, n]; prefixes: G lists of c output modes whose multiplicity multisets are all the SAME (the
    tables then have one shape, [G, *dims, *dims]; `_prefix_order` groups the circuits by that shape).
    Circuit g's distinct prefix modes are ordered so that their multiplicities line up with the shared
    `dims`, and the suffix-table recursion and the column decisions run over all circuits and streams in one
    sequence of launches instead of one per circuit - what the imperfect adaptive sampler needs, where every
    stream is its own circuit (docs/ADAPTIVE.md §7, task A11). Same law as `prefix_columns`, circuit by circuit.
    """
    A = A.to(torch.complex128)
    G, n, dev = A.shape[0], A.shape[2], A.device
    S = dp.shape[0]
    counts = [sorted(((list(pre).count(b), b) for b in set(pre)), key=lambda x: (-x[0], x[1])) for pre in prefixes]
    p = [c for c, _ in counts[0]]
    if any([c for c, _ in cnt] != p for cnt in counts):
        raise ValueError("prefix_columns_many: the prefixes must share one multiplicity shape")
    L, dims = len(p), [c + 1 for c in p]
    D = math.prod(dims)
    modes = torch.tensor([[b for _, b in cnt] for cnt in counts], dtype=torch.int64, device=dev)     # [G, L]
    B = torch.gather(A, 1, modes[:, :, None].expand(G, L, n))                       # [G, L, n]
    F = [None] * (n + 1)
    F[n] = torch.zeros([G, *dims, *dims], dtype=torch.complex128, device=dev)
    F[n][(slice(None), *([0] * (2 * L)))] = 1.0
    bshape = [G] + [1] * (2 * L)
    for j in range(n - 1, -1, -1):
        nxt = F[j + 1].clone()
        for a in range(L):
            Fa = _shift(F[j + 1], 1 + a)
            for b in range(L):
                nxt += (B[:, a, j] * B[:, b, j].conj()).reshape(bshape) * _shift(Fa, 1 + L + b)
        F[j] = nxt
    flip = list(range(1, 2 * L + 1))
    g = torch.zeros([S, *dims], dtype=torch.complex128, device=dev)
    g[(slice(None), *([0] * L))] = 1.0
    chosen = torch.zeros(S, n, dtype=torch.bool, device=dev)
    Bs = B[dp]                                                                      # [S, L, n], each stream's rows

    def mass(gv, Fj):                                                               # [S]: g^T Fr conj(g), real
        Fr = torch.flip(Fj, flip).reshape(G, D, D)[dp]                             # [S, D, D]
        v = gv.reshape(S, D)
        return torch.einsum("sa,sab,sb->s", v, Fr, v.conj()).real.clamp_min(0.0)

    for j in range(n):
        g_in = torch.zeros_like(g)
        for a in range(L):
            g_in += Bs[:, a, j].reshape([S] + [1] * L) * _shift(g, 1 + a)
        w_in, w_out = mass(g_in, F[j + 1]), mass(g, F[j + 1])
        take = torch.rand(S, generator=generator, device=dev, dtype=torch.float64) * (w_in + w_out) < w_in
        chosen[:, j] = take
        g = torch.where(take.reshape([S] + [1] * L), g_in, g)
    return chosen


def _prefix_order(A2, dp, n, prefix, generator):
    """The column order sig [S, n] for a forced prefix: its first c entries are a set drawn by
    `prefix_columns_many` (their order is immaterial), the others a uniform order of the rest. The circuits
    of the pass are grouped by the multiplicity shape of their prefix, one batched draw per group.
    """
    dev, S = A2.device, dp.shape[0]
    A = torch.view_as_complex(A2)                                                   # [G, m, n]
    key = torch.rand(S, n, generator=generator, device=dev)
    circuits = torch.unique(dp).tolist()
    groups: dict = {}                                                               # multiplicity shape -> [circuit ids]
    for g in circuits:
        pre = prefix[g].tolist() if prefix.dim() == 2 else prefix.tolist()          # per-circuit or shared prefix
        shape = tuple(sorted((pre.count(b) for b in set(pre)), reverse=True))
        groups.setdefault(shape, []).append(g)
    for gs in groups.values():
        gs_t = torch.tensor(gs, dtype=torch.int64, device=dev)
        ids = torch.nonzero(torch.isin(dp, gs_t))[:, 0]
        local = torch.searchsorted(gs_t, dp[ids])                                   # circuit index within the group
        pres = [prefix[g].tolist() if prefix.dim() == 2 else prefix.tolist() for g in gs]
        chosen = prefix_columns_many(A[gs_t], pres, local, generator)              # [S_g, n]
        key[ids] = torch.where(chosen, torch.full_like(key[ids], -1.0), key[ids])  # chosen columns sort first
    return torch.argsort(key, dim=1).contiguous()


def _sample_streams(A2, dp, n, m, generator, algo, multi_min_k1, max_lanes, target_total_lanes, multi, prefix=None,
                    avoid=None):
    """One lockstep pass over the streams dp (dp[s] = index of the unitary of stream s): returns the
    ordered output rows [S', n] and, with `avoid`, the indices [S'] of the streams that survived (else None).
    With `prefix` (c output modes) the first c rows are given and the chain starts at step c + 1 on a column
    order drawn by `_prefix_order`. With `avoid` (bool [m]) a stream whose new row is an avoided mode is
    dropped from the batch right after that step (the batch shrinks; the kernels see the survivors only).
    """
    dev = A2.device
    S = dp.shape[0]
    if prefix is not None:
        prefix = torch.as_tensor(prefix, dtype=torch.int64, device=dev)            # [r] shared, or [G, r] per circuit
    c0 = 0 if prefix is None else prefix.shape[-1]
    if c0:
        sig = _prefix_order(A2, dp, n, prefix, generator)
    else:
        sig = torch.argsort(torch.rand(S, n, generator=generator, device=dev), dim=1).contiguous()
    rows = torch.zeros(S, n, dtype=torch.int64, device=dev)
    if c0:
        rows[:, :c0] = prefix[dp] if prefix.dim() == 2 else prefix
    seeds = torch.randint(0, 2 ** 31 - 1, (n,), generator=generator, device=dev).tolist()
    sidx = torch.arange(S, device=dev, dtype=torch.int64)

    ones = torch.zeros(S, 1, 2, dtype=torch.float64, device=dev)
    ones[:, :, 0] = 1.0
    one_lane = torch.ones(S, dtype=torch.int64, device=dev)
    unit_scale = torch.ones(S, dtype=torch.float64, device=dev)

    keep = torch.arange(S, device=dev, dtype=torch.int64) if avoid is not None else None
    for k in range(c0 + 1, n + 1):
        K1, K = k - 1, k
        if avoid is not None and k > c0 + 1:
            ok = ~avoid[rows[:, k - 2]]
            if not bool(ok.all()):
                rows, sig, dp, keep = rows[ok].contiguous(), sig[ok].contiguous(), dp[ok].contiguous(), keep[ok]
                S = rows.shape[0]
                if S == 0:
                    return rows, keep
                sidx = torch.arange(S, device=dev, dtype=torch.int64)
                ones = torch.zeros(S, 1, 2, dtype=torch.float64, device=dev); ones[:, :, 0] = 1.0
                one_lane = torch.ones(S, dtype=torch.int64, device=dev)
                unit_scale = torch.ones(S, dtype=torch.float64, device=dev)
        use_multi = (algo in ("2020", "percof") and K1 >= 1) or (algo == "auto" and multi_min_k1 < K1)
        sig_k = sig
        if K1 == 0:
            part, lstart, nl, scale, mx = ones, sidx, one_lane, unit_scale, 1
        elif algo == "input":
            from .cofactors_input import qvec_partials_input
            part, lstart, nl, scale, mx, _, sig_k = qvec_partials_input(A2, rows, sig, dp, S, K1, K, target_total_lanes)
        elif use_multi and algo == "percof":
            # PiquassoBoost form: one full sweep per cofactor. Launch c computes all K cofactors
            # (that is what a sweep does) and only column c of it is kept, so the samples are those
            # of the 2020 form and the work is K sweeps per step, as in their sampler.
            part = None
            for c in range(K):
                part_c, lstart, nl, scale, mx, _ = multi(A2, rows, sig, dp, S, K1, K, target_total_lanes)
                if part is None:
                    part = torch.empty_like(part_c)
                part[:, c] = part_c[:, c]
        elif use_multi:
            part, lstart, nl, scale, mx, _ = multi(A2, rows, sig, dp, S, K1, K, target_total_lanes)
        else:
            nterms, lanes, L = _plan(S, K1, max_lanes, target_total_lanes)
            part = cofactor_partials(A2, rows, sig, dp, S, K1, K, lanes, L)
            lstart, nl, scale, mx = sidx * lanes, one_lane * lanes, unit_scale / nterms, lanes
        draw_step(A2, part, rows, sig_k, dp, S, m, K, lstart, nl, scale, k - 1, seeds[k - 1], max_lanes=mx)
    if avoid is not None and n > c0:
        ok = ~avoid[rows[:, n - 1]]
        return rows[ok], keep[ok]
    return rows, keep
