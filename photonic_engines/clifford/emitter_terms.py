"""Batched gap-channel decomposition of the reduced states of many streams at once (torch).

For stream s with kept slots kept[s, 0..k-1] (sorted), the reduced state on those slots is
rho_T = sum_e |psi_e><psi_e| with psi_e(x) = <f_e| M~_{k-1,x_{k-1}} ... M~_{0,x_0} |s0_e>
(docs/EMITTER_SAMPLER.md §2). This module computes all streams' terms in one lockstep pass over
the chain positions j = 0..n-1:

  * every stream keeps one Kraus set per gap g = 0..k (g = 0 before the first kept slot, g = k
    after the last one, g = i after kept slot i-1), as a padded tensor K[s, g, 4, 2, 2] with
    rank r[s, g] <= 4 (K = {I}, r = 1 initially);
  * at position j, every stream computes the update of the set of the gap j falls in,
    {M_j,l K_i} -> batched SVD of the [8, 4] stack -> the <= 4 leading operators, and the
    streams for which j is kept discard it (lockstep, no host synchronisation);
  * afterwards the prefix set is applied to |s0> and the suffix set to <f| and each reduced
    to <= 2 vectors (batched SVD), the term index is decoded in mixed radix over the gap
    ranks, the kept slots' operators are gathered and left-multiplied by the chosen gap
    operators, and zero-norm chains are dropped with the batched chi^2 transfer chain.

Output: s0 [Λ, 2], mats [Λ, k, L, 2, 2], f [Λ, 2], term2stream [Λ] — the lane tensors the
samplers consume. chi = 2 fixed (the padded ranks 4 / 2 are chi^2 / chi).
"""
from __future__ import annotations

import torch

from ..states import EmitterMPS

C128 = torch.complex128


def _reduce_rows(A: torch.Tensor, keep: int, tol: float = 1e-12):
    """A [N, R, d] -> (B [N, keep, d], rank [N]) with B^dag B = A^dag A (rows = S * Vh)."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)          # S [N, min(R, d)]
    thr = tol * S[:, :1].clamp_min(1e-300)
    rank = (thr < S).sum(dim=1)
    S = torch.where(thr < S, S, torch.zeros_like(S))
    B = S.unsqueeze(2) * Vh                                        # [N, min(R,d), d]
    if B.shape[1] < keep:
        B = torch.nn.functional.pad(B, (0, 0, 0, keep - B.shape[1]))
    return B[:, :keep], rank.clamp_max(keep)


def _mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched small matmul by broadcast-multiply-sum (2x2 / 4x4 blocks: no cuBLAS tiny-batch
    gemm, no einsum path selection). A [..., a, b] @ B [..., b, c] -> [..., a, c].
    """
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def sample_lost_labels(state: EmitterMPS, keep: torch.Tensor, generator=None):
    """Label-basis unravelling of the traced (lost) emission slots — docs/LOSS.md §3b.

    Tracing a lost photon equals measuring it in its label basis and forgetting the outcome,
    and operations on distinct photons commute, so a lossy sample is: draw the lost slots'
    labels x from their exact joint marginal (kept slots and, if traced, the final spin summed
    over), then sample the kept photons from the *pure* emitter state conditioned on x — the
    chain with the lost slots' operators fixed to M_{j, x_j}. Every gap then contributes one
    Kraus term instead of up to chi^2 (the term count R of docs/EMITTER_SAMPLER.md stays 1).

    keep [S, k] long: kept slots per stream. Returns (labels [S, n] long, -1 at kept slots;
    fid [S] long, the final-spin basis state drawn jointly with the labels when the emitter is
    traced, else zeros). The case of `sample_slot_outcomes` with no singleton slots.
    """
    keep = torch.as_tensor(keep, dtype=torch.long).cpu()
    S, n = keep.shape[0], state.n
    roles = torch.zeros((S, n), dtype=torch.long)
    roles.scatter_(1, keep, 2)
    return sample_slot_outcomes(state, roles, generator=generator)


def sample_slot_outcomes(state: EmitterMPS, roles: torch.Tensor, U: torch.Tensor | None = None, generator=None):
    """Draw, slot by slot, what the samplers fold before the chain rule runs on the remaining
    (interfering) slots: the LABEL of every lost slot (role 0, docs/LOSS.md §3b) and the OUTPUT
    ROW of every singleton slot (role 1 — a photon distinguishable from all others,
    docs/DISTINGUISHABILITY.md: it interferes with nothing but keeps its label coherent, so its
    output row r is drawn from the exact marginal and the slot is then the fixed operator
    G_j(r) = sum_l U[r, modes[j][l]] M_{j,l}). Slots of role >= 2 are the interfering photons,
    summed over here (their marginal is the traced channel, by unitarity of U).

    roles [S, n] long. Returns (labels [S, n] long: -1 at interfering slots, x in 0..L-1 at
    lost slots, L + r at singleton slots; fid [S] long, the final-spin basis state drawn jointly
    when the emitter is traced, else zeros). Cost O(S n (L^2 + m) chi^3) on the cpu, one pass:
    right environments R_j = sum_l M_{j,l}^dag R_{j+1} M_{j,l} (all later slots traced, whatever
    their role), forward register density matrix per stream, p(outcome | past) ∝
    Tr[R_{j+1} K rho K^dag] with K = M_{j,x} (lost) or G_j(r) (singleton), rho <- sum_l M rho
    M^dag at interfering slots. `U` is needed only when a singleton slot is present.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    roles = torch.as_tensor(roles, dtype=torch.long).cpu()
    S, n = roles.shape
    chi, L = state.chi, state.num_labels
    ops = torch.as_tensor(state.ops, dtype=C128)                             # [n, L, chi, chi]
    has_single = bool((roles == 1).any())
    if has_single:
        if U is None:
            raise ValueError("sample_slot_outcomes: U is needed to place singleton (distinguishable) slots")
        Ut = torch.as_tensor(U).to("cpu", C128)
        modes = torch.tensor([list(mk) for mk in state.modes], dtype=torch.long)   # [n, L]
    R = [None] * (n + 1)
    R[n] = torch.eye(chi, dtype=C128) if state.final is None else torch.outer(
        torch.as_tensor(state.final, dtype=C128), torch.as_tensor(state.final, dtype=C128).conj())
    for j in range(n - 1, -1, -1):
        R[j] = torch.einsum("lst,su,luv->tv", ops[j].conj(), R[j + 1], ops[j])
    s0 = torch.as_tensor(state.s0, dtype=C128)
    rho = torch.outer(s0, s0.conj()).unsqueeze(0).expand(S, chi, chi).clone()   # [S, chi, chi]
    labels = torch.full((S, n), -1, dtype=torch.long)
    for j in range(n):
        cross = torch.einsum("lts,xsu,kvu->xlktv", ops[j], rho, ops[j].conj())  # [S, L, L, chi, chi] M_l rho M_k^dag
        cand = torch.einsum("xlltv->xltv", cross)                              # [S, L, chi, chi]
        traced = cand.sum(1)
        new = traced
        lost = roles[:, j] == 0
        if bool(lost.any()):
            w = torch.einsum("tv,xlvt->xl", R[j + 1], cand).real.clamp_min(0)   # [S, L]
            w = w / w.sum(1, keepdim=True).clamp_min(1e-300)
            x = torch.multinomial(w, 1, generator=generator)[:, 0]
            labels[lost, j] = x[lost]
            new = torch.where(lost.view(S, 1, 1), cand[torch.arange(S), x], new)
        single = roles[:, j] == 1
        if bool(single.any()):
            Um = Ut[:, modes[j]]                                                # [m, L]  U[r, modes[j][l]]
            cand_r = torch.einsum("rl,rk,xlktv->xrtv", Um, Um.conj(), cross)    # [S, m, chi, chi] G(r) rho G(r)^dag
            w = torch.einsum("tv,xrvt->xr", R[j + 1], cand_r).real.clamp_min(0)  # [S, m]
            w = w / w.sum(1, keepdim=True).clamp_min(1e-300)
            r = torch.multinomial(w, 1, generator=generator)[:, 0]
            labels[single, j] = L + r[single]
            new = torch.where(single.view(S, 1, 1), cand_r[torch.arange(S), r], new)
        rho = new
        tr = torch.einsum("xtt->x", rho).real.clamp_min(1e-300)
        rho = rho / tr.view(S, 1, 1)
    if state.final is None:
        pf = torch.einsum("xtt->xt", rho).real.clamp_min(0)                  # p(s_f | outcomes) ∝ <s_f| rho_n |s_f>
        fid = torch.multinomial(pf / pf.sum(1, keepdim=True), 1, generator=generator)[:, 0]
    else:
        fid = torch.zeros(S, dtype=torch.long)
    return labels, fid


def label_weights(state: EmitterMPS, labels: torch.Tensor, U: torch.Tensor | None = None) -> torch.Tensor:
    """The per-slot operator weights [S, n, L] (complex) encoded by `sample_slot_outcomes`
    labels: ones at interfering slots (-1: summed over, pending in the chain), one-hot at lost
    slots (x < L), and the row U[r, modes[j][l]] at singleton slots (L + r) — the slot's operator
    in the chain is sum_l w[s, j, l] M_{j,l}.
    """
    S, n = labels.shape
    L = state.num_labels
    dev = labels.device
    w = torch.zeros((S, n, L), dtype=C128, device=dev)
    w[labels < 0] = 1
    lostm = (labels >= 0) & (labels < L)
    w[lostm] = torch.nn.functional.one_hot(labels[lostm], L).to(C128)
    rowm = labels >= L
    if bool(rowm.any()):
        if U is None:
            raise ValueError("label_weights: U is needed for singleton (row-folded) slots")
        Ut = torch.as_tensor(U).to(dev, C128)
        modes = torch.tensor([list(mk) for mk in state.modes], dtype=torch.long, device=dev)   # [n, L]
        rows = (labels - L).clamp_min(0)                                          # [S, n]
        Uw = Ut[rows.reshape(-1)].reshape(S, n, -1)                               # [S, n, m]
        Uw = torch.gather(Uw, 2, modes.unsqueeze(0).expand(S, n, L))              # [S, n, L]
        # unit-norm row weights: a per-stream scalar leaves the conditional chain rule unchanged and
        # keeps the folded chain's norm O(1) — raw |U| entries (~1/sqrt(m) each) drove it below the
        # zero-norm term drop of `_finish` at n ~ 30 with a few singletons (GPU run 20260921_051530)
        Uw = Uw / Uw.norm(dim=2, keepdim=True).clamp_min(1e-300)
        w[rowm] = Uw[rowm]
    return w


def batched_gap_terms(state: EmitterMPS, kept: torch.Tensor, final: torch.Tensor,
                      tol: float = 1e-12, block: int = 3, labels: torch.Tensor = None, U: torch.Tensor = None):
    """Kept [S, k] long (sorted slots), final [S, chi] complex (final vector per stream).
    labels [S, n] long (optional, `sample_slot_outcomes`): the sampled label of every lost slot
    (x < L: that slot contributes the single operator M_{j, x}) and the sampled output row of
    every singleton slot (L + r: the operator sum_l U[r, modes[j][l]] M_{j, l}, `U` required);
    every gap set then stays rank 1.

    The n - k traced positions of every stream are walked in blocks of `block` positions: a
    block multiplies the touched gap sets by up to `block` operators (capacity 4 L^block, no
    reduction in between) and one batched SVD per block reduces every touched (stream, gap)
    set back to <= 4 operators — `block`x fewer kernel launches than one SVD per position,
    which is what bounds this function's time (a few hundred small ops per step).
    """
    dev = kept.device
    S, k = kept.shape
    n, L, chi = state.n, state.num_labels, state.chi
    assert chi == 2
    ops = torch.as_tensor(state.ops, dtype=C128, device=dev)                 # [n, L, 2, 2]
    s0 = torch.as_tensor(state.s0, dtype=C128, device=dev)
    final = final.to(dev, C128)
    P = n - k                                                                  # traced positions per stream
    pos = torch.arange(n, device=dev)
    gap_of = (kept.unsqueeze(2) < pos.view(1, 1, n)).sum(dim=1)              # [S, n] in 0..k
    is_kept = torch.zeros((S, n), dtype=torch.bool, device=dev)
    is_kept.scatter_(1, kept, True)
    CAP = 4 * L ** block
    K = torch.zeros((S, k + 1, CAP, 2, 2), dtype=C128, device=dev)
    K[:, :, 0] = torch.eye(2, dtype=C128, device=dev)
    rank = torch.ones((S, k + 1), dtype=torch.long, device=dev)
    sidx = torch.arange(S, device=dev)
    if P > 0:
        # traced positions per stream, sorted: [S, P]; padded blocks use an identity operator
        # (label 0 only) on a dummy gap slot that is reduced and discarded like any other
        traced_pos = pos.view(1, n).expand(S, n)[~is_kept].view(S, P)
        ops_pad = torch.cat([ops, torch.zeros((1, L, 2, 2), dtype=C128, device=dev)])
        ops_pad[n, 0] = torch.eye(2, dtype=C128, device=dev)
        if labels is not None:
            # one operator weight vector per slot (`label_weights`): one-hot at a lost slot, the U
            # row at a singleton slot, ones at a kept slot not yet in the chain (pending, summed
            # over all labels); padding slot n: label 0
            lmask = torch.zeros((S, n + 1, L), dtype=C128, device=dev)              # [S, n+1, L]
            lmask[:, :n] = label_weights(state, labels.to(dev), U)
            lmask[:, n, 0] = 1
            lab_pad = torch.full((S, n + 1), -1, dtype=torch.long, device=dev)
            lab_pad[:, :n] = labels.to(dev)
        nb = -(-P // block)
        tp = torch.full((S, nb * block), n, dtype=torch.long, device=dev)
        tp[:, :P] = traced_pos
        gp = torch.zeros((S, nb * block), dtype=torch.long, device=dev)
        gp[:, :P] = gap_of.gather(1, traced_pos)
        for b in range(nb):
            cols = slice(b * block, (b + 1) * block)
            for u in range(block):
                j = tp[:, b * block + u]                                        # [S] positions (n = padding)
                g = gp[:, b * block + u]
                cap = 4 * L ** u
                cur = K[sidx, g, :cap]                                         # [S, cap, 2, 2]
                opj = ops_pad[j]                                               # [S, L, 2, 2]
                if labels is not None:
                    opj = opj * lmask[sidx, j].view(S, L, 1, 1)
                    # a row-folded slot is ONE operator sum_l w_l M_l (coherent), not a set {w_l M_l}
                    rowf = (lab_pad[sidx, j] >= L).view(S, 1, 1, 1)
                    summed = torch.zeros_like(opj); summed[:, 0] = opj.sum(1)
                    opj = torch.where(rowf, summed, opj)
                new = _mm(opj.view(S, L, 1, 2, 2), cur.view(S, 1, cap, 2, 2)).reshape(S, L * cap, 2, 2)
                K[sidx, g, :L * cap] = new
            # reduce the gaps touched by this block: (stream, gap) pairs, duplicates masked
            gb = gp[:, cols]                                                   # [S, block]
            dup = torch.zeros_like(gb, dtype=torch.bool)
            for u in range(1, block):
                dup[:, u] = (gb[:, :u] == gb[:, u:u + 1]).any(1)
            srep = sidx.view(S, 1).expand(S, block).reshape(-1)
            grep = gb.reshape(-1)
            sets = K[srep, grep].reshape(S * block, CAP, 4)
            red, r = _reduce_rows(sets, 4, tol)                               # [S*block, 4, 4]
            keepm = ~dup.reshape(-1)
            si, gi = srep[keepm], grep[keepm]
            K[si, gi] = 0
            K[si, gi, :4] = red[keepm].reshape(-1, 4, 2, 2)
            rank[si, gi] = r[keepm]
    return _finish(state, kept, K[:, :, :4], rank, s0, final, ops, tol)


def _finish(state, kept, K, rank, s0, final, ops, tol):
    dev = kept.device
    S, k = kept.shape
    pre_vecs = _mm(K[:, 0], s0.view(1, 1, 2, 1)).squeeze(-1)                  # [S, 4, 2] : K_i s0
    s0_terms, r0 = _reduce_rows(pre_vecs, 2, tol)                             # [S, 2, 2], [S]
    suf_rows = _mm(final.conj().view(S, 1, 1, 2), K[:, k]).squeeze(-2)         # [S, 4, 2] : f^dag K_i (row vectors)
    f_rows, rk = _reduce_rows(suf_rows, 2, tol)                                # rows; f_e = conj(row)
    f_terms = f_rows.conj()
    # radices per stream: (r0, rk, r_1 .. r_{k-1})
    inner = rank[:, 1:k] if k > 1 else torch.zeros((S, 0), dtype=torch.long, device=dev)
    radix = torch.cat([r0.unsqueeze(1), rk.unsqueeze(1), inner], dim=1)        # [S, k+1]
    R = radix.prod(dim=1)                                                      # [S]
    Lam = int(R.sum().item())
    term2stream = torch.repeat_interleave(torch.arange(S, device=dev), R, output_size=Lam)
    start = torch.cumsum(R, 0) - R
    e = torch.arange(Lam, device=dev) - start[term2stream]                     # local term index
    # mixed-radix decode (least significant = last radix), vectorized: place value of digit c is
    # the product of the radices after it
    rad = radix[term2stream]                                                   # [Λ, k+1]
    place = torch.cumprod(rad.flip(1), 1).flip(1)                              # prod_{c' >= c} rad
    place = torch.cat([place[:, 1:], torch.ones((Lam, 1), dtype=torch.long, device=dev)], 1)
    digits = (e.unsqueeze(1) // place) % rad                                   # [Λ, k+1]
    s0_e = s0_terms[term2stream, digits[:, 0]]                                 # [Λ, 2]
    f_e = f_terms[term2stream, digits[:, 1]]                                   # [Λ, 2]
    mats = ops[kept[term2stream]]                                              # [Λ, k, L, 2, 2] (fresh gather: writable)
    if k > 1:
        Kg = K[term2stream.unsqueeze(1), torch.arange(1, k, device=dev).unsqueeze(0), digits[:, 2:]]   # [Λ, k-1, 2, 2]
        mats[:, :k - 1] = _mm(Kg.unsqueeze(2), mats[:, :k - 1])
    # drop zero-norm chains: <f| E_k ... E_1 (|s0><s0|) |f> with the transfer matrices
    # E_j = sum_l M_{j,l} (x) conj(M_{j,l}) (4 x 4), multiplied as a balanced tree (log2 k rounds)
    # [Λ, k, 2(t), 2(u), 2(s), 2(v)] : M[t,s] conj(M)[u,v]
    E = (mats.unsqueeze(-1).unsqueeze(-3) * mats.conj().unsqueeze(-2).unsqueeze(-4)).sum(2)
    E = E.permute(0, 1, 2, 3, 4, 5).reshape(Lam, k, 4, 4)                     # row (t,u), col (s,v)
    while E.shape[1] > 1:
        if E.shape[1] % 2:
            E = torch.cat([E, torch.eye(4, dtype=C128, device=dev).expand(Lam, 1, 4, 4)], 1)
        E = _mm(E[:, 1::2], E[:, 0::2])                                        # later positions on the left
    rho = _mm(E[:, 0], (s0_e.unsqueeze(2) * s0_e.conj().unsqueeze(1)).reshape(Lam, 4, 1)).reshape(Lam, 2, 2)
    norm2 = torch.einsum("xs,xst,xt->x", f_e.conj(), rho, f_e).real
    scale = (f_e.abs() ** 2).sum(1) * (s0_e.abs() ** 2).sum(1)
    keep = norm2 > 1e-14 * scale.clamp_min(1.0)
    return s0_e[keep], mats[keep], f_e[keep], term2stream[keep]
