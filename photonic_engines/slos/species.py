"""Strong simulation under partial distinguishability — docs/DISTINGUISHABILITY.md §3.1.

A shot is a species partition of the input photons (`imperfections`): the photons of one
species interfere among themselves, different species do not, and the detector sees the sum
of the species' patterns. On the layered Fock basis of `FockSLOSKernel` that is three
operations, all differentiable in U:

  * a species of g >= 2 photons: the SLOS recurrence on its columns, stopped on layer g of the
    kernel's basis (`FockSLOSKernel._amplitudes(U, input_modes)`), squared with the deferred
    normalisation of that layer (`layer_probs`);
  * a singleton (a photon in a mode of its own — or a species of one photon): a PROBABILITY-level
    raise along the k -> k+1 edges of the basis, q_{k+1}(t + e_j) += q_k(t) |U[j, i]|^2, with no
    bosonic factor because it never bunches with a photon of another species (`classical_raise`);
  * two multi-photon species: the explicit convolution of their layers (`convolve_layers`,
    S_a x S_b — the annotated-input case of the Perceval front end, small n).

The x-model (`x_mixture`) is the sum over the interfering subsets G with weights
x^|G| (1 - x)^(n - |G|), each term a `species_probs`: exact, 2^n terms, truncated by the number
of distinguishable photons with the binomial-tail bound `imperfections.truncation_bound`. There
is no output-side shortcut as for loss: the internal state of a photon does not commute with U.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from ..imperfections import _layer_occupations, spec


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def classical_raise(layers, q: torch.Tensor, k: int, col: int, U: torch.Tensor) -> torch.Tensor:
    """Add one photon that interferes with nothing, entering column `col`, to the k-photon
    distribution q ([S_k] or [B, S_k]) : q_{k+1}(t + e_j) += q_k(t) |U[j, col]|^2 over the
    k -> k+1 edges (mode slab by mode slab). U [m, m] or [B, m, m]. Differentiable in q and U.
    """
    parent, child, off = layers.edges_by_mode(k)
    off = [int(v) for v in _np(off)]
    batched = U.dim() == 3
    w_all = (U[..., :, col].abs() ** 2).to(q.dtype)                                   # [m] or [B, m]
    out = torch.zeros((*q.shape[:-1], layers.size(k + 1)), dtype=q.dtype, device=q.device)
    for j in range(layers.m):
        a, b = off[j], off[j + 1]
        if b <= a:
            continue
        c = torch.as_tensor(child[a:b], device=q.device).to(torch.int64)
        p_idx = torch.as_tensor(parent[a:b], device=q.device).to(torch.int64)
        w = w_all[:, j].unsqueeze(1) if batched else w_all[j]
        out = out.index_add(-1, c, q.index_select(-1, p_idx) * w)
    return out


def pattern_index(layers, k: int) -> dict:
    """{occupation tuple: index in layer k} (materialised per call; layers are small where used)."""
    return {tuple(int(v) for v in row): i for i, row in enumerate(_layer_occupations(layers, k))}


def convolve_layers(layers, qa: torch.Tensor, ka: int, qb: torch.Tensor, kb: int) -> torch.Tensor:
    """Distribution of the SUM of two independent patterns, qa on layer ka and qb on layer kb
    ([S] or [B, S] each, same batch), on layer ka + kb: q(t) = sum_{ta + tb = t} qa(ta) qb(tb).
    Explicit S_a x S_b pairs (a mask may drop a sum: its mass then leaves the distribution, as
    the mask semantics require). Differentiable.
    """
    occ_a, occ_b = _layer_occupations(layers, ka), _layer_occupations(layers, kb)
    idx = pattern_index(layers, ka + kb)
    S = layers.size(ka + kb)
    pairs_a, pairs_b, target = [], [], []
    for ia, ta in enumerate(occ_a):
        for ib, tb in enumerate(occ_b):
            t = tuple(int(v) for v in ta + tb)
            if t in idx:
                pairs_a.append(ia); pairs_b.append(ib); target.append(idx[t])
    dev = qa.device
    ia = torch.tensor(pairs_a, device=dev); ib = torch.tensor(pairs_b, device=dev); tg = torch.tensor(target, device=dev)
    out = torch.zeros((*qa.shape[:-1], S), dtype=qa.dtype, device=dev)
    return out.index_add(-1, tg, qa.index_select(-1, ia) * qb.index_select(-1, ib))


def layer_probs(kernel, U: torch.Tensor, cols: Sequence[int]) -> torch.Tensor:
    """|<t| U |cols>|^2 on layer g = len(cols) of the kernel's basis, [S_g] or [B, S_g]: the SLOS
    recurrence on those columns with the deferred normalisation of that layer (input 1/prod s_i!,
    output prod t_j!). g = 0 is the vacuum (a single state of probability 1).
    """
    g = len(cols)
    if g == 0:
        B = U.shape[0] if U.dim() == 3 else None
        one = torch.ones((1,), dtype=kernel.rdtype, device=kernel.device)
        return one.expand(B, 1) if B is not None else one
    amp = kernel._amplitudes(U, input_modes=cols)
    counts: dict = {}
    for c in cols:
        counts[c] = counts.get(c, 0) + 1
    s_norm = math.prod(math.factorial(v) for v in counts.values())
    p = (amp.real ** 2 + amp.imag ** 2) / s_norm
    norm = torch.as_tensor(kernel.layers.prob_norms(g)).to(dtype=kernel.rdtype, device=kernel.device)
    return p * norm


def species_probs(kernel, U: torch.Tensor, groups: Sequence[Sequence[int]], singles: Sequence[int]) -> torch.Tensor:
    """Distribution on the final layer for a species partition: `groups` the columns of each
    multi-photon species, `singles` the columns of the photons in a mode of their own. The
    species' layers are convolved, the singletons raised classically. U [m, m] or [B, m, m].
    """
    U = U.to(device=kernel.device, dtype=kernel.cdtype)
    groups = [list(g) for g in groups]
    singles = list(singles) + [g[0] for g in groups if len(g) == 1]              # a one-photon species is a singleton
    groups = sorted((g for g in groups if len(g) >= 2), key=len, reverse=True)
    q, k = layer_probs(kernel, U, groups[0] if groups else []), (len(groups[0]) if groups else 0)
    for g in groups[1:]:
        q, k = convolve_layers(kernel.layers, q, k, layer_probs(kernel, U, g), len(g)), k + len(g)
    for c in singles:
        q, k = classical_raise(kernel.layers, q, k, c, U), k + 1
    if k != kernel.n:
        raise ValueError(f"the partition covers {k} photons, the kernel has {kernel.n}")
    return q


def x_mixture(kernel, U: torch.Tensor, indist, order: int | None = None) -> torch.Tensor:
    """The x-model distribution on the final layer: sum over the interfering subsets G of the
    kernel's input photons, x^|G| (1 - x)^(n - |G|) `species_probs`(G, the rest). `indist` a
    float, a `Distinguishability(x=...)` or a tensor x (then differentiable in x). `order=b`
    keeps |G| >= n - b (at most b distinguishable photons), dropping `truncation_bound`.
    """
    from itertools import combinations
    d = spec(indist)
    if d.x is None:
        raise NotImplementedError("strong simulation: the x-model only (random species need a partition sum)")
    x = indist if isinstance(indist, torch.Tensor) else torch.tensor(d.x, dtype=kernel.rdtype, device=kernel.device)
    x = x.to(dtype=kernel.rdtype, device=kernel.device)
    cols = kernel.input_modes
    n = len(cols)
    b_max = n if order is None else max(0, min(int(order), n))
    out = None
    for b in range(b_max + 1):
        w = x ** (n - b) * (1 - x) ** b
        for bad in combinations(range(n), b):
            good = [cols[j] for j in range(n) if j not in bad]
            term = species_probs(kernel, U, [good] if len(good) >= 2 else [], [cols[j] for j in bad] + (good if len(good) == 1 else []))
            out = term * w if out is None else out + term * w
    return out


def _remap_layer(src_layers, dst_layers, q: torch.Tensor, k: int) -> torch.Tensor:
    """A layer-k distribution on the basis `src_layers` re-indexed on layer k of `dst_layers`
    (the two bases share their patterns whenever both hold every k-photon prefix of their
    admissible outputs; a pattern absent from the destination drops its mass, as a mask does).
    """
    if src_layers is dst_layers:
        return q
    idx = pattern_index(dst_layers, k)
    src = _layer_occupations(src_layers, k)
    keep, tgt = [], []
    for i, row in enumerate(src):
        t = tuple(int(v) for v in row)
        if t in idx:
            keep.append(i); tgt.append(idx[t])
    out = torch.zeros((*q.shape[:-1], dst_layers.size(k)), dtype=q.dtype, device=q.device)
    return out.index_add(-1, torch.tensor(tgt, device=q.device), q.index_select(-1, torch.tensor(keep, device=q.device)))


def product_x_mixture(kernel, U: torch.Tensor, indist, order: int | None = None) -> list[torch.Tensor]:
    """The x-model for a `ProductSuperposition` input (superposition boson sampling): a source in
    a mode of its own factorises, its photon is a Bernoulli(|a[i, q]|^2) classical photon, so

        P = sum_{G subset of sources} x^|G| (1 - x)^(n_s - |G|) [ ProductSLOS(G)  (*)  (*)_{i not in G} routed photon_i ],

    the convolution acting layer by layer on the graded distribution (q photons of a bad source
    raise layer k to k + q with weight |a[i, q]|^2, `classical_raise` q times). Returns the
    blocks [S_k] (or [B, S_k]) for k = 0..n on the kernel's basis. `order=b`: at most b bad
    sources. Exponential in the source count, like the kernel itself.
    """
    from itertools import combinations

    from ..states import ProductSuperposition
    from .kernel import ProductSLOSKernel
    d = spec(indist)
    if d.x is None:
        raise NotImplementedError("strong simulation: the x-model only")
    x = indist if isinstance(indist, torch.Tensor) else torch.tensor(d.x, dtype=kernel.rdtype, device=kernel.device)
    x = x.to(dtype=kernel.rdtype, device=kernel.device)
    st = kernel.state
    ns, dloc = st.sources, st.d
    amps2 = (np.abs(np.asarray(st.amps)) ** 2)
    U = U.to(device=kernel.device, dtype=kernel.cdtype)
    B = U.shape[0] if U.dim() == 3 else None
    b_max = ns if order is None else max(0, min(int(order), ns))
    out = [None] * (kernel.n + 1)

    def zeros(k):
        return torch.zeros((*(() if B is None else (B,)), kernel.layers.size(k)), dtype=kernel.rdtype, device=kernel.device)

    for b in range(b_max + 1):
        w = x ** (ns - b) * (1 - x) ** b
        for bad in combinations(range(ns), b):
            good = [i for i in range(ns) if i not in bad]
            if good:
                sub = ProductSuperposition(np.asarray(st.amps)[good], tuple(st.modes[i] for i in good), st.m)
                sk = ProductSLOSKernel(sub, mask=kernel.mask, device=str(kernel.device), dtype=kernel.cdtype)
                blocks = [_remap_layer(sk.layers, kernel.layers, q, k) for k, q in enumerate(sk.probs(U).probs)]
            else:
                v = zeros(0); v[..., 0] = 1
                blocks = [v]
            for i in bad:                                                       # a bad source: q routed photons w.p. |a[i, q]|^2
                new = [None] * (len(blocks) + dloc - 1)
                for k, q in enumerate(blocks):
                    r = q
                    for qq in range(dloc):
                        if amps2[i, qq] > 0:
                            term = r * float(amps2[i, qq])
                            new[k + qq] = term if new[k + qq] is None else new[k + qq] + term
                        if qq + 1 < dloc:
                            r = classical_raise(kernel.layers, r, k + qq, st.modes[i], U)
                blocks = [v if v is not None else zeros(k) for k, v in enumerate(new)]
            for k, q in enumerate(blocks):
                out[k] = q * w if out[k] is None else out[k] + q * w
    return [v if v is not None else zeros(k) for k, v in enumerate(out)]


def _fold_absorbed(state, folds: dict):
    """`EmitterMPS.fold` with the tail operator absorbed into the last unfolded slot (its
    operators left-multiplied by the tail), so the reduced chain needs no extra gate; returns
    (reduced EmitterMPS with the original `final`, or None with the tail when all slots fold).
    """
    from ..states import EmitterMPS
    red, tail = state.fold(folds)
    if red is None:
        return None, tail
    ops = np.array(red.ops)
    ops[-1] = np.einsum("ts,lsu->ltu", tail, ops[-1])
    return EmitterMPS(ops=ops, modes=red.modes, s0=red.s0, m=red.m, final=state.final), None


def _row_fold_ops(state, U: np.ndarray, slot: int, row: int) -> np.ndarray:
    """G_slot(row) = sum_l U[row, modes[slot][l]] M_{slot, l} — the operator of a distinguishable
    photon of the chain detected in output row `row` (docs/DISTINGUISHABILITY.md §2).
    """
    return sum(U[row, state.modes[slot][l]] * state.ops[slot, l] for l in range(len(state.modes[slot])))


def emitter_x_mixture(kernel, U: torch.Tensor, indist, order: int | None = None, kraus: bool = False):
    """The x-model for an `EmitterMPS` input, species-resolved by enumeration: for every set B of
    b distinguishable slots and every assignment of their output rows r_B (m^b), the chain with
    those slots folded at G_j(r_j) is an (n - b)-photon emitter chain whose SLOS distribution,
    shifted by the rows, lands on the coarse pattern t = t_good + sum_j e_{r_j}; weight
    x^(n - b) (1 - x)^b — or prod_j x_j / (1 - x_j) with one overlap per slot (`indist` a sequence
    of n values: the emitters of a `SequenceKernel` bin). C(n, b) m^b runs of an (n - b)-photon kernel; `order=b_max` truncates
    (`imperfections.truncation_bound`). A distinguishable photon is coupled to the register
    through its label, so its row cannot be summed before the trace — hence the enumeration.

    kraus=False: probabilities [S] (or [B, S]) on the kernel's final layer (traced: the
    mixture; projected: the sub-normalised pure distribution). kraus=True (U unbatched): the
    species-resolved Kraus set (K [Q, chi, chi] in PHYSICAL normalisation, coarse [Q] the index
    of each operator's coarse pattern) — the register update of coarse outcome t is the channel
    rho -> sum_{q: coarse[q] = t} K[q] rho K[q]^dag, trace-preserving after summing t.
    """
    from itertools import combinations
    from itertools import product as iproduct

    from .emitter import EmitterSLOSKernel
    st = kernel.state
    n, m = st.n, st.m
    if isinstance(indist, (list, tuple, np.ndarray)) or (isinstance(indist, torch.Tensor) and indist.dim() == 1):
        xs = torch.as_tensor(indist, dtype=kernel.rdtype, device=kernel.device)              # one overlap per slot
        if xs.shape != (n,):
            raise ValueError(f"per-slot indist must have {n} entries")
    else:
        d = spec(indist)
        if d.x is None:
            raise NotImplementedError("strong simulation: the x-model only")
        x = indist if isinstance(indist, torch.Tensor) else torch.tensor(d.x, dtype=kernel.rdtype, device=kernel.device)
        xs = x.to(dtype=kernel.rdtype, device=kernel.device).expand(n)
    Ut = U.to(device=kernel.device, dtype=kernel.cdtype)
    if kraus and Ut.dim() == 3:
        raise ValueError("kraus=True takes one unitary")
    Un = Ut.detach().cpu().numpy().astype(np.complex128) if Ut.dim() == 2 else None
    idx = pattern_index(kernel.layers, n)
    B = Ut.shape[0] if Ut.dim() == 3 else None
    b_max = n if order is None else max(0, min(int(order), n))
    out = None
    Ks, coarse = [], []
    for b in range(b_max + 1):
        for bad in combinations(range(n), b):
            good_mask = torch.ones(n, dtype=torch.bool, device=kernel.device)
            good_mask[list(bad)] = False
            w = torch.where(good_mask, xs, 1 - xs).prod()                       # prod_good x_j prod_bad (1 - x_j)
            for rows in iproduct(range(m), repeat=b):
                if Un is None:
                    raise NotImplementedError("emitter x-model with a batch of unitaries: one U at a time")
                red, tail = _fold_absorbed(st, {j: _row_fold_ops(st, Un, j, r) for j, r in zip(bad, rows)})
                shift = np.zeros(m, dtype=np.int64)
                for r in rows:
                    shift[r] += 1
                if red is None:                                             # every photon distinguishable
                    K = torch.as_tensor(tail, dtype=kernel.cdtype, device=kernel.device)          # [chi, chi]
                    v = K @ kernel.s0
                    p = (v.abs() ** 2).sum() if st.final is None else (kernel.final.conj() @ v).abs() ** 2
                    occ_red = np.zeros((1, m), dtype=np.int64)
                    p = p.reshape(1).to(kernel.rdtype)
                    K = K.unsqueeze(0)
                else:
                    rk = EmitterSLOSKernel(red, mask=kernel.mask, device=str(kernel.device), dtype=kernel.cdtype)
                    p = rk.probs(Ut)                                       # [S_red] or [B, S_red]
                    occ_red = rk.layers.final_occupations()
                    if kraus:
                        K = rk.kraus(Ut) * torch.sqrt(torch.as_tensor(rk.layers.output_prob_norms(), dtype=kernel.rdtype,
                                                                     device=kernel.device)).view(-1, 1, 1).to(kernel.cdtype)
                tgt = [idx.get(tuple(int(v) for v in row + shift), -1) for row in occ_red]
                keep = [i for i, t in enumerate(tgt) if t >= 0]
                if not keep:
                    continue
                kt = torch.tensor(keep, device=kernel.device); tt = torch.tensor([tgt[i] for i in keep], device=kernel.device)
                term = torch.zeros((*(() if B is None else (B,)), len(idx)), dtype=kernel.rdtype, device=kernel.device)
                term = term.index_add(-1, tt, p.index_select(-1, kt) * w)
                out = term if out is None else out + term
                if kraus:
                    Ks.append(K[kt] * torch.sqrt(w).to(kernel.cdtype)); coarse.append(tt)
    if kraus:
        return torch.cat(Ks), torch.cat(coarse)
    return out
