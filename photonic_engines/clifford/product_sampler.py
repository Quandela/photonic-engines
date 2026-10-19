"""Exact chain-rule sampler for a `states.ProductSuperposition` input (superposition boson sampling).

`slos.ProductSLOSKernel.sample` is exact too, but it builds the whole graded distribution first, so
it stops where strong simulation stops. This marches the Clifford & Clifford chain rule instead:
draw the photon number, then place photons one at a time, each from the TRUE marginal of the rows
placed so far. No basis, no state vector.

Why the marginal is computable (docs/SUPERPOSITION.md §5). Post-selected on k photons the input is
the Dicke state on the n source modes, and the marginal of the first j placed rows r is

    M_j(r) = sum_{|R| = k-j} | A_R(r) |^2 ,   A_R(r) = sum_{S subset [n]\\R, |S| = j} Per(U[r, S]) ,

R being the sources excluded from supplying those j photons. Since M_j IS the marginal, the chain
rule is exact by construction - no exchangeability argument is needed, unlike Clifford & Clifford's
random photon order (which does NOT carry over here: the analogous random source order fails from
j = 2, measured 2026-09-19).

Two ways to evaluate it, cheap at opposite ends, and the sampler takes whichever is cheaper:

  residual   enumerate R and evaluate each A_R with the lambda filter of `slos.glynn`:
             C(n, k-j) 2^j n j.  Cheap when k - j is small, i.e. at the END of the chain.
  replica    swap the two sums. The number of R disjoint from S and S' depends only on
             t = |S cap S'|, so M_j = sum_t C(n - 2j + t, k - j) N_t with
             N_t = sum_{|S cap S'| = t} Per_S conj(Per_S'), and the N_t come out of ONE doubled
             Glynn sum over (delta, eps) in {+-1}^j x {+-1}^j carrying a (|S|, |S'|, t) table:
             4^j n j^3, with no subset enumeration at all. Cheap when j is small, i.e. at the
             START of the chain.

The maximum over j of the cheaper of the two is what sets the reach, and it is the 4^j that
dominates in the middle of the chain. Whether that can be brought down to 2^j - the way a hafnian
of a 2k x 2k matrix costs 2^k and not 4^k - is the open question of docs/SUPERPOSITION.md §7.

Uniform alpha only (every source in the same cos(alpha)|0> + sin(alpha)|1>), which is what makes
the post-selected input the UNIFORM Dicke state. Non-uniform sources weight the residual sum by an
elementary symmetric polynomial instead of a binomial coefficient; the derivation goes through,
the code does not implement it.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..states import ProductSuperposition


def _signs(j: int, start: int, stop: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Sign vectors `start`..`stop` of {+-1}^j, [stop - start, j]."""
    idx = torch.arange(start, stop, device=device, dtype=torch.int64).unsqueeze(1)
    bits = torch.arange(j, device=device, dtype=torch.int64).unsqueeze(0)
    return (1 - 2 * ((idx >> bits) & 1)).to(dtype)


def _budget_bytes(dev: torch.device) -> int:
    """Ceiling on the working set of one marginal evaluation, in bytes.

    The polynomial tables below are the only large allocations here and their natural size is
    `candidates x sign vectors^2 x (j+1)^3`, which nothing in the cost model bounds: at j = 8 with
    126 candidates and 256 sign vectors per batch that table is 96 GiB, and the sbs arm of
    bench/find_threshold.py died on exactly that (run 20260919_195403). The evaluation is a sum
    over independent blocks, so the fix is to batch it: this sets how big a batch may be, and the
    result is identical whatever it returns.
    """
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        return max(1 << 28, int(0.6 * free))
    try:                                       # a fixed CPU budget is wrong on a small machine:
        with open("/proc/meminfo") as fh:      # 4 GiB of tables on an 8 GB box thrashes, because
            for line in fh:                    # the recurrence holds several copies of one
                if line.startswith("MemAvailable:"):
                    return max(1 << 27, int(0.4 * int(line.split()[1]) * 1024))
    except OSError:
        pass
    return 1 << 31


def _replica_marginals(V: torch.Tensor, n: int, k: int, j: int, chunk: int) -> torch.Tensor:
    """M_j for a batch of candidate row sets, by the doubled Glynn sum. V: [C, j, n] (the columns of
    U at the placed rows, source-ordered). Returns [C] real. Cost O(C 4^j n j^3).
    """
    C, dtype, dev = V.shape[0], V.dtype, V.device
    esz = torch.empty(0, dtype=dtype).element_size()
    full = 1 << j
    room = max(1, int(_budget_bytes(dev) // (4 * (j + 1) ** 3 * esz)))         # 4 live tables: poly, nxt, 2 temporaries
    sc = min(chunk, full)                                                      # sign vectors per batch, both replicas
    while sc > 1 and sc * sc > room:
        sc //= 2
    cc = max(1, min(C, room // (sc * sc)))                                     # candidates per batch
    tot = torch.zeros((C, j + 1), dtype=dtype, device=dev)                     # N_t, t = 0..j
    for c0 in range(0, C, cc):
        Vc = V[c0:c0 + cc]                                                     # [Cc, j, n]
        Cc = Vc.shape[0]
        acc = tot[c0:c0 + cc]                                                  # a view: += lands in tot
        for d0 in range(0, full, sc):
            sd = _signs(j, d0, min(d0 + sc, full), dtype, dev)                 # [D, j]
            P = torch.einsum("ds,csa->cdsa", sd, Vc).sum(2)                    # [Cc, D, n]
            pd = sd.prod(dim=1)                                                # [D]
            for e0 in range(0, full, sc):
                se = _signs(j, e0, min(e0 + sc, full), dtype, dev)             # [E, j]
                Q = torch.einsum("es,csa->cesa", se, Vc.conj()).sum(2)         # [Cc, E, n]
                pe = se.prod(dim=1)
                D, E = P.shape[1], Q.shape[1]
                # poly[c, d, e, a, b, t]: coefficient of lam^a mu^b x^t after the sources seen so far
                poly = torch.zeros((Cc, D, E, j + 1, j + 1, j + 1), dtype=dtype, device=dev)
                poly[:, :, :, 0, 0, 0] = 1
                for a in range(n):
                    p = P[:, :, a].reshape(Cc, D, 1, 1, 1, 1)
                    q = Q[:, :, a].reshape(Cc, 1, E, 1, 1, 1)
                    nxt = poly.clone()
                    nxt[:, :, :, 1:, :, :] += poly[:, :, :, :-1, :, :] * p                 # source in S
                    nxt[:, :, :, :, 1:, :] += poly[:, :, :, :, :-1, :] * q                 # source in S'
                    nxt[:, :, :, 1:, 1:, 1:] += poly[:, :, :, :-1, :-1, :-1] * (p * q)     # in both
                    poly = nxt
                acc += torch.einsum("cdet,d,e->ct", poly[:, :, :, j, j, :], pd, pe)
                del poly
    tot = tot / (4.0 ** j)
    w = torch.tensor([math.comb(n - 2 * j + t, k - j) if n - 2 * j + t >= k - j >= 0 else 0
                      for t in range(j + 1)], dtype=torch.float64, device=dev)
    return (tot.real.to(torch.float64) * w).sum(dim=1)


def _subsets(n: int, q: int, dev) -> torch.Tensor:
    """The C(n, q) q-subsets of [n], [Nr, q] int64, each row ascending.

    NOT torch.combinations: that meshgrids r copies of the input, so it needs n^r elements before
    masking -- at n = 17, r = 8 it asked for 110 GiB and killed five points of run 20260920_121038
    on an otherwise empty H100. Unranking the combinatorial number system costs O(Nr q) instead,
    with no intermediate: in colex order rank = sum_t C(a_t, t), so a_t is the largest v with
    C(v, t) <= rank, which is one searchsorted per position over a table that is nondecreasing in v.
    Order is colex here and was lex before; nothing downstream depends on it (the tree levels come
    from torch.unique, which sorts, and the sum over R is a sum).
    """
    if q <= 0:
        return torch.zeros((1, 0), dtype=torch.int64, device=dev)
    r = torch.arange(math.comb(n, q), dtype=torch.int64, device=dev)
    out = torch.empty((r.shape[0], q), dtype=torch.int64, device=dev)
    for t in range(q, 0, -1):
        tab = torch.tensor([math.comb(v, t) for v in range(n)], dtype=torch.int64, device=dev)
        v = torch.searchsorted(tab, r, right=True) - 1                 # largest v with C(v,t) <= r
        out[:, t - 1] = v
        r = r - tab[v]
    return out


_TREES: dict = {}


def _prefix_tree(n: int, q: int, dev) -> list:
    """The lex-ordered q-subsets of [n] as a prefix tree: per level, (parent index, new element).

    Level t holds the distinct t-prefixes; level q is the q-subsets themselves. Cached, because a
    chain revisits the same (n, q) once per shot and the levels cost O(Nr log Nr) to build.
    """
    key = (n, q, str(dev))
    if key not in _TREES:
        excl = _subsets(n, q, dev)
        levels, prev = [], torch.zeros(excl.shape[0], dtype=torch.int64, device=dev)
        for t in range(q):
            cur, inv = torch.unique(excl[:, :t + 1], dim=0, return_inverse=True)
            par = torch.zeros(cur.shape[0], dtype=torch.int64, device=dev)
            par[inv] = prev                    # every row carrying a prefix agrees on its parent
            levels.append((par, cur[:, t].contiguous()))
            prev = inv
        _TREES[key] = levels
    return _TREES[key]


def _residual_marginals(V: torch.Tensor, n: int, k: int, j: int, chunk: int) -> torch.Tensor:
    """M_j for a batch of candidate row sets, by enumerating the residual sets R. V: [C, j, n].
    Returns [C] real. Cost O(C 2^j C(n, k-j) j), NOT O(C 2^j C(n, k-j) n j).

    A_R needs prod_{a not in R} (1 + lam P_a), and the C(n, k-j) residual sets overlap almost
    completely: rebuilding each product over all n sources repeats nearly all of the work. So the
    full product over every source is built once and factors are DIVIDED out along a prefix tree of
    the excluded sets, one factor per level, so each node costs O(j) and the nodes number
    sum_t C(n, t) ~ 1.3 C(n, k-j). Division by (1 + lam p) is a unitriangular back-substitution,
    c_0 = q_0, c_i = q_i - p c_{i-1}, exact and stable here -- it agrees with the direct product to
    1e-15 over every step of n = 12, 16, 17 (2026-09-20), and the brute-force oracle in
    tests/clifford covers it independently. Measured 4.4-6.3x against the direct version at
    n = 17, k = 11.
    """
    C, dtype, dev = V.shape[0], V.dtype, V.device
    esz = torch.empty(0, dtype=dtype).element_size()
    levels = _prefix_tree(n, k - j, dev)
    big = max([1] + [len(par) for par, _ in levels])
    full = 1 << j
    room = max(1, int(_budget_bytes(dev) // (4 * (j + 1) * esz)))   # 4 live: parent, base, nxt, temp
    sc = min(chunk, full)                                           # sign vectors per batch
    while sc > 1 and sc * big > room:
        sc //= 2
    cc = max(1, min(C, room // max(1, sc * big)))                   # candidates per batch
    nc = max(1, min(big, room // max(1, cc * sc)))                  # tree nodes per batch
    out = torch.zeros(C, dtype=torch.float64, device=dev)
    for c0 in range(0, C, cc):
        Vc = V[c0:c0 + cc]
        Cc = Vc.shape[0]
        amp = torch.zeros((Cc, big), dtype=dtype, device=dev)       # [Cc, Nr]
        for d0 in range(0, full, sc):   # the delta sum must COMPLETE before squaring: accumulate
            sd = _signs(j, d0, min(d0 + sc, full), dtype, dev)      # [D, j]
            P = torch.einsum("ds,csa->cdsa", sd, Vc).sum(2)         # [Cc, D, n]
            D = P.shape[1]
            poly = torch.zeros((Cc, D, 1, j + 1), dtype=dtype, device=dev)
            poly[..., 0] = 1
            for a in range(n):                                      # prod over ALL sources, once
                poly[..., 1:] = poly[..., 1:] + poly[..., :-1] * P[:, :, a].view(Cc, D, 1, 1)
            for par, newa in levels:                                # one factor divided out per level
                nxt = torch.empty((Cc, D, par.shape[0], j + 1), dtype=dtype, device=dev)
                for b0 in range(0, par.shape[0], nc):
                    sl = slice(b0, b0 + nc)
                    base = poly[:, :, par[sl], :]                   # [Cc, D, N, j+1]
                    p = P[:, :, newa[sl]]                           # [Cc, D, N]
                    nxt[:, :, sl, 0] = base[..., 0]
                    for i in range(1, j + 1):
                        nxt[:, :, sl, i] = base[..., i] - p * nxt[:, :, sl, i - 1]
                poly = nxt
            amp = amp + torch.einsum("cdr,d->cr", poly[..., j], sd.prod(dim=1))
        amp = amp / (2.0 ** j)
        out[c0:c0 + cc] = (amp.real ** 2 + amp.imag ** 2).sum(dim=1).to(torch.float64)
    return out


# How much dearer one unit of the replica cost model is than one unit of the residual one on the
# hardware this runs on. Both evaluations are bandwidth-bound elementwise recurrences and their
# model units are not interchangeable, so the crossover needs this and gets it wrong without it:
# the first version compared the two as equal and kept the chain on the replica sum one or two
# steps too long, which at k = 12 cost 7123 ms of a 9719 ms sample (run 20260919_224559).
# Two independent fits of the ratio, since the prefix-tree residual landed: 0.89 carrying the H100
# constants of that run through the measured 5x, and 0.76 timing both evaluations directly on two
# CPU cores at n = 17, k = 11. They agree that the two now cost about the same per unit.
# Only which of two EXACT evaluations runs depends on this, never the answer, so a bad value costs
# time and nothing else. Refit it from `bench/bench_product_sampler.py --profile` on new hardware.
REPLICA_UNIT_COST = 0.85


def cheaper(n: int, k: int, j: int) -> str:
    """Which evaluation of M_j is cheaper at step j, "residual" or "replica" (see the header).

    Per candidate row set, and with the shared candidate count cancelled:

      residual  C(n, k-j) 2^j (j+1)   -- the prefix tree visits ~1.3 C(n, k-j) nodes at O(j) each,
                                         and NOT n times that: the n-source product is built once
      replica   4^j j^3 n             -- the [2^j, 2^j, (j+1)^3] table, recurred over n sources
    """
    res = math.comb(n, k - j) * 2 ** j * (j + 1)
    rep = REPLICA_UNIT_COST * 4 ** j * j ** 3 * n
    return "residual" if res <= rep else "replica"


def marginals(state: ProductSuperposition, U: torch.Tensor, rows: list[int], candidates: list[int],
              k: int, method: str = "auto", chunk: int = 1 << 8,
              dtype: torch.dtype = torch.complex128, device=None) -> torch.Tensor:
    """M_j for every candidate next row: the true marginal of (rows + [c]) under the k-photon input.

    rows       the j - 1 rows already placed (output mode indices, repeats allowed)
    candidates the rows to score, one M_j each
    method     "auto" (the cheaper of the two), "replica" or "residual"
    Returns a real [len(candidates)] tensor, unnormalized but on one common scale.
    """
    if state.d != 2:
        raise ValueError("chain-rule sampler: d = 2 only")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    n, j = state.sources, len(rows) + 1
    if not 0 <= k <= n:
        raise ValueError(f"k must be in 0..{n}")
    U = torch.as_tensor(U, dtype=dtype, device=dev)
    cols = torch.as_tensor(list(state.modes), dtype=torch.int64, device=dev)
    idx = torch.as_tensor([[*rows, c] for c in candidates], dtype=torch.int64, device=dev)
    V = U[idx][:, :, cols]                                                     # [C, j, n]
    if method == "auto":
        method = cheaper(n, k, j)
        if method == "residual":
            # the prefix tree holds a whole level at once, so C(n, k-j) (j+1) is a floor on its
            # working set however the batching is set. The replica sum has no such floor -- it
            # batches every axis down to one -- so when the floor does not fit, take that instead.
            esz = torch.empty(0, dtype=dtype).element_size()
            if 4 * math.comb(n, k - j) * (j + 1) * esz > _budget_bytes(dev):
                method = "replica"
    if method == "residual":
        return _residual_marginals(V, n, k, j, chunk)
    if method == "replica":
        return _replica_marginals(V, n, k, j, chunk)
    raise ValueError(f'unknown method {method!r}: "auto", "replica" or "residual"')


def lossy_source(alpha: float, eta: float) -> tuple[float, float]:
    """A partially-excited source behind a uniform transmission eta, as (P(live), alpha').

    Uniform loss on c|0> + s|1> (Kraus K0 = diag(1, sqrt(eta)), K1 = sqrt(1-eta) a) gives the MIXED
    single-mode state

        rho = |phi><phi| + (1-eta) s^2 |0><0| ,  |phi> = c|0> + s sqrt(eta) |1>
            = w1 |alpha'><alpha'| + w0 |0><0| ,  w1 = c^2 + eta s^2,  w0 = (1-eta) s^2,

    with |alpha'> = cos(alpha')|0> + sin(alpha')|1>, tan(alpha') = s sqrt(eta) / c. Loss acts
    independently per source, so the n-source state is a mixture over SUBSETS: keep each source
    with probability w1 as a superposition at alpha', vacuum it otherwise. Every term of that
    mixture is a lossless `ProductSuperposition` on a subset of the modes, so the lossy sampler is
    the lossless one under a Bernoulli mask -- no new algorithm, and exact.

    The emission probability is w1 sin^2(alpha') = eta sin^2(alpha) and the coherence is
    <a> = cos(alpha) sin(alpha) sqrt(eta), so at FIXED brightness p = eta sin^2(alpha) the pair
    (alpha, eta) has one free parameter left, the coherence sqrt(p (1 - p/eta)): it runs from 0 at
    eta = p (a deterministic source behind a lossy channel -- the Fock modality) to its maximum
    sqrt(p(1-p)) at eta = 1 (partial excitation, no loss). Checked against
    `ProductSLOSKernel.probs(U, loss=eta)`, TVD 2e-16 (docs/LOSS.md).
    """
    c, s = math.cos(alpha), math.sin(alpha)
    return c * c + eta * s * s, math.atan2(s * math.sqrt(eta), c)


def sample(state: ProductSuperposition, U: torch.Tensor, shots: int,
           generator: torch.Generator | None = None, method: str = "auto",
           chunk: int = 1 << 8, dtype: torch.dtype = torch.complex128, device=None,
           photons: int | None = None, loss: float | None = None, indist=None) -> torch.Tensor:
    """Exact samples [shots, m] of a `ProductSuperposition` input through U, by the chain rule above.

    The photon number is drawn per shot from `state.photon_number_probs()` (exactly binomial for
    `sbs`, and independent of U), then that many rows are placed one at a time. `photons` fixes it
    instead, i.e. samples the post-selected k-photon sector. No basis is built, so this is not
    bounded by strong simulation; it is bounded by the per-step cost, which the module docstring
    gives. Cross-checked against `slos.ProductSLOSKernel.probs` within shot noise.

    `loss=eta` applies a uniform transmission on the input side by `lossy_source` above: each shot
    draws which sources are live and samples the lossless chain rule on those modes alone.
    `indist=x` (x-model per source, docs/DISTINGUISHABILITY.md §3.3): a source in an internal mode
    of its own factorises out of the Dicke coherence — its photon, present with probability
    sin^2(alpha'), is routed classically by |U[:, mode]|^2 — so each shot also draws which live
    sources share the common mode and runs the chain rule on those alone. Both draws are
    independent per source; the strong side `ProductSLOSKernel.probs(loss=, indist=)` is the oracle.
    """
    from ..imperfections import spec
    dev = torch.device(device) if device is not None else torch.device("cpu")
    m = state.m
    d = spec(indist)
    if d is not None and d.x is None:
        raise NotImplementedError("product_sample: the x-model only")
    if (loss is not None and loss < 1.0) or d is not None:
        if photons is not None:
            raise ValueError("loss / indist with photons=: post-selecting a detected count "
                             "conditions the source masks too, which this does not do")
        amps = torch.as_tensor(np.array(state.amps), dtype=torch.complex128)
        if not torch.allclose(amps, amps[:1].expand_as(amps)):
            raise ValueError("lossy / distinguishable chain rule: uniform alpha only (see lossy_source)")
        alpha = math.atan2(float(amps[0, 1].abs()), float(amps[0, 0].abs()))
        w1, ap = lossy_source(alpha, float(loss)) if loss is not None and loss < 1.0 else (1.0, alpha)
        modes = list(state.modes)
        out = torch.zeros((shots, m), dtype=torch.int64)
        live = torch.rand((shots, len(modes)), generator=generator) < w1          # [shots, sources]
        good = torch.rand((shots, len(modes)), generator=generator) < (1.0 if d is None else d.x)
        emit = torch.rand((shots, len(modes)), generator=generator) < math.sin(ap) ** 2   # a lone source's photon
        Uabs2 = (torch.as_tensor(U).abs() ** 2).to("cpu", torch.float64)
        for shot in range(shots):
            keep = [mo for mo, on, g in zip(modes, live[shot].tolist(), good[shot].tolist()) if on and g]
            if keep:
                sub = ProductSuperposition.sbs(len(keep), m, alpha=ap, modes=keep)
                out[shot] = sample(sub, U, 1, generator=generator, method=method, chunk=chunk,
                                   dtype=dtype, device=dev)[0]
            for mo, on, g, e in zip(modes, live[shot].tolist(), good[shot].tolist(), emit[shot].tolist()):
                if on and not g and e:                                              # live, own species, emitted
                    r = int(torch.multinomial(Uabs2[:, mo], 1, generator=generator))
                    out[shot, r] += 1
        return out
    pk = torch.as_tensor(state.photon_number_probs(), dtype=torch.float64)
    out = torch.zeros((shots, m), dtype=torch.int64)
    for shot in range(shots):
        k = int(photons) if photons is not None else int(torch.multinomial(pk, 1, generator=generator))
        rows: list[int] = []
        for _ in range(k):
            w = marginals(state, U, rows, list(range(m)), k, method=method, chunk=chunk,
                          dtype=dtype, device=dev).clamp_min(0)
            rows.append(int(torch.multinomial(w.cpu(), 1, generator=generator)))
        for r in rows:
            out[shot, r] += 1
    return out
