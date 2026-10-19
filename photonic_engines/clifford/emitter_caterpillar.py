"""Caterpillar-specific exact sampler: segment-factorised pure terms (docs/EMITTER_SAMPLER.md §5).

For a caterpillar state every emission operator is a projector times the spin gate,
M^(j)_l = |l><l| G_j, so the coefficient of a label pattern is a product of nearest-neighbour
transfer amplitudes, c_x = <f|x_n> prod_j <x_j|G_j|x_{j-1}> <x_1|G_1|s0>: the bond index of the
MPS *is* the emission label. Tracing a run of emissions (a gap) therefore gives a classical
transition weight w[t, s] = [ |G_{b-1}|^2 ... |G_{a+2}|^2 ]_{t,s} between the label s entering the
gap and the label t leaving it (|G|^2 elementwise), and the reduced state on the kept slots T
is the mixture, over the labels at the gap boundaries, of PRODUCT states over the segments of
consecutive kept slots:

    rho_T = sum_e p_e |psi_e><psi_e|,   psi_e(x_T) = prod_seg <s_seg| G_{q+1} P_{x_q} G_q ... P_{x_p} G_p |t_seg>,
    e = (t_0; (s_i, t_i) per inner gap; s_f),   p_e = w_0[t_0] prod_i w_i[t_i, s_i] w_f[s_f].

This is the same number of pure terms as the SVD purification (`gap_kraus_terms`: the rank of a
measure-and-prepare channel is the number of non-zero w[t, s], 4 in general for a spin), so
R_k is not reduced - it is the true rank. What changes is the cost per term: with rank-one
gap operators the chi x chi chains of the Clifford step are computed ONCE PER SEGMENT and
shared by every term, which only reads matrix elements of them:

    amp_e(rho) = sum_c sum_l U[rho, mode(c, l)] cof_e[c, l],
    cof_e[c, l] = 2^{-(k-2)} sum_delta sign(delta) [prod_{seg' != seg(c)} C_seg'(delta)[e]] F_c[l, s_seg(e), t_seg(e)](delta),
    F_c[l, s, t] = Q_c[s, l] (G_{t_c} P_c)[l, t],   P_c / Q_c = within-segment prefix / suffix chains,

so the per-(term, column, delta) work drops from ~28 complex multiply-adds (chain + cofactor
accumulation per term) to 2 plus a gather. `caterpillar_step_weights` is the single-stream
reference of the scheme, `sample_caterpillar` the batched sampler (streams grouped by kept
slots, delta chunks); both are validated against `matrix_cofactor_weights` and the strong
simulation. The Triton port is the "segment lanes" design of docs/EMITTER_SAMPLER.md §5.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch

from ..states import EmitterMPS
from .emitter_reference import final_spin_distribution

C128 = torch.complex128


# ------------------------------------------------------------------ structure -------------
def caterpillar_gates(state: EmitterMPS, tol: float = 1e-12) -> torch.Tensor | None:
    """G [n, chi, chi] with ops[j, l] = |l><l| G_j for every emission and label, or None when
    the state is not of that form (requires L == chi and label l on basis vector l).
    """
    ops = torch.as_tensor(state.ops, dtype=C128)
    n, L, chi, _ = ops.shape
    if chi != L:
        return None
    G = ops.sum(1)                                                    # sum_l P_l G = G
    for l in range(L):
        proj = torch.zeros((chi, chi), dtype=C128)
        proj[l, l] = 1.0
        if not torch.allclose(ops[:, l], torch.einsum("ts,jsu->jtu", proj, G), atol=tol):
            return None
    return G


def _classical_chain(G2: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """|G_{hi}|^2 ... |G_{lo}|^2 (identity when hi < lo); G2 [n, chi, chi] real."""
    chi = G2.shape[1]
    W = torch.eye(chi, dtype=G2.dtype)
    for j in range(lo, hi + 1):
        W = G2[j] @ W
    return W


class _Structure:
    """Segments, boundary dimensions and the term table of one (kept slots, final) pair."""

    def __init__(self, state: EmitterMPS, G: torch.Tensor, kept: Sequence[int], final: torch.Tensor | None):
        n, chi = state.n, state.chi
        self.kept = list(kept)
        k = len(self.kept)
        s0 = torch.as_tensor(state.s0, dtype=C128)
        G2 = (G.abs() ** 2).real
        # segments = maximal runs of consecutive kept slots
        segs: list[tuple[int, int]] = []                                 # (first col index, last col index) in 0..k-1
        a = 0
        for c in range(1, k + 1):
            if c == k or self.kept[c] != self.kept[c - 1] + 1:
                segs.append((a, c - 1))
                a = c
        self.segs = segs
        self.seg_of_col = [i for i, (p, q) in enumerate(segs) for _ in range(p, q + 1)]
        # boundary matrices per segment: B_in [chi, in_dim], B_out [out_dim, chi]; boundary
        # weights: prefix w0 [in_dim of seg 0], inner w_i [in_dim(seg i+1), out_dim(seg i)], suffix w_f
        self.B_in, self.B_out = [], []
        for p, q in segs:
            t_p, t_q = self.kept[p], self.kept[q]
            if t_p == 0:
                self.B_in.append(s0.unsqueeze(1))                        # [chi, 1]
            else:
                self.B_in.append(torch.eye(chi, dtype=C128))
            if t_q == n - 1:
                if final is not None:
                    self.B_out.append(final.conj().unsqueeze(0))         # [1, chi]
                else:
                    self.B_out.append(torch.eye(chi, dtype=C128))        # traced: one term per final basis state
            else:
                self.B_out.append(G[t_q + 1])                            # rows: label at slot t_q + 1
        # gap weights
        self.w0 = None
        if self.kept[0] > 0:                                             # slots 0 .. t_1 - 1
            v = (G[0] @ s0).abs() ** 2                                   # |<x_1|G_1|s0>|^2 (slot 0)
            self.w0 = (_classical_chain(G2, 1, self.kept[0] - 1) @ v.real)   # label at slot t_1 - 1
        self.w_inner: list[torch.Tensor | None] = []
        for i in range(len(segs) - 1):
            a_, b_ = self.kept[segs[i][1]], self.kept[segs[i + 1][0]]    # gap = a_+1 .. b_-1
            self.w_inner.append(_classical_chain(G2, a_ + 2, b_ - 1))    # [t (slot b_-1), s (slot a_+1)]
        self.wf = None
        if self.kept[-1] < n - 1:
            Wsuf = _classical_chain(G2, self.kept[-1] + 2, n - 1)        # [x_n, s]
            if final is not None:
                pf = (final.abs() ** 2).real
                self.wf = pf @ Wsuf
            else:
                self.wf = Wsuf.sum(0)                                    # = 1 (stochastic)
        # term table: labels (in, out) per segment, weight
        nseg = len(segs)
        in_dims = [B.shape[1] for B in self.B_in]
        out_dims = [B.shape[0] for B in self.B_out]
        choices = []
        for i in range(nseg):
            choices.append(range(in_dims[i]))
            choices.append(range(out_dims[i]))
        rows_in, rows_out, weights = [], [], []
        for combo in itertools.product(*choices):
            ins, outs = combo[0::2], combo[1::2]
            p = 1.0
            if self.w0 is not None:
                p *= float(self.w0[ins[0]])
            for i in range(nseg - 1):
                p *= float(self.w_inner[i][ins[i + 1], outs[i]])
            if self.wf is not None:
                p *= float(self.wf[outs[-1]])
            if p > 1e-15:
                rows_in.append(ins); rows_out.append(outs); weights.append(p)
        self.seg_in = torch.tensor(rows_in, dtype=torch.long).reshape(-1, nseg)
        self.seg_out = torch.tensor(rows_out, dtype=torch.long).reshape(-1, nseg)
        self.p = torch.tensor(weights, dtype=torch.float64)
        self.R = self.p.shape[0]


def _signs(kk: int, device) -> tuple[torch.Tensor, torch.Tensor, float]:
    if kk <= 1:
        return torch.ones((1, max(kk, 1)), dtype=torch.float64, device=device), torch.ones(1, dtype=torch.float64, device=device), 1.0
    grid = torch.tensor(list(itertools.product([1.0, -1.0], repeat=kk - 1)), dtype=torch.float64, device=device)
    d = torch.cat([torch.ones((grid.shape[0], 1), dtype=torch.float64, device=device), grid], 1)
    return d, d.prod(1), 2.0 ** (-(kk - 1))


def _group_cofactors(st: _Structure, G: torch.Tensor, kept: list[int], Urows: torch.Tensor, modes: torch.Tensor,
                     delta_chunk: int) -> torch.Tensor:
    """Cof [S, E, k, L] for S streams sharing the structure `st`; Urows [S, k-1, m] their drawn rows."""
    dev = Urows.device
    S, kk, m = Urows.shape
    k = len(kept)
    L = chi = G.shape[1]
    dtype = Urows.dtype
    Gk = G[kept].to(dev, dtype)                                                     # [k, chi, chi]
    Ucols = torch.gather(Urows.unsqueeze(2).expand(S, kk, k, m), 3, modes.view(1, 1, k, L).expand(S, kk, k, L)) if kk \
        else torch.zeros((S, 0, k, L), dtype=dtype, device=dev)                      # [S, i, j, l]
    deltas, sign, norm = _signs(kk, dev)
    E, nseg = st.R, len(st.segs)
    seg_in, seg_out = st.seg_in.to(dev), st.seg_out.to(dev)
    cof = torch.zeros((S, E, k, L), dtype=dtype, device=dev)
    B_in = [B.to(dev, dtype) for B in st.B_in]
    B_out = [B.to(dev, dtype) for B in st.B_out]
    for d0 in range(0, deltas.shape[0], delta_chunk):
        dl = deltas[d0:d0 + delta_chunk].to(dtype)
        sg = sign[d0:d0 + delta_chunk].to(dtype)
        Dc = dl.shape[0]
        colsum = torch.einsum("di,sijl->sdjl", dl, Ucols) if kk else torch.ones((S, 1, k, L), dtype=dtype, device=dev)
        Gd = colsum.unsqueeze(-1) * Gk.view(1, 1, k, chi, chi)                     # diag(s) G : [S, Dc, k, chi, chi]
        # per segment: prefix P_c = G_{c-1}..G_p B_in, suffix Q_c = B_out G_q..G_{c+1}, chain C = Q_c G_c P_c
        Cseg, F = [], [None] * k
        for si, (p, q) in enumerate(st.segs):
            P = [None] * (q - p + 1)
            v = B_in[si].view(1, 1, chi, -1).expand(S, Dc, chi, B_in[si].shape[1])
            for c in range(p, q + 1):
                P[c - p] = v
                v = torch.matmul(Gd[:, :, c], v)
            Q = [None] * (q - p + 1)
            u = B_out[si].view(1, 1, -1, chi).expand(S, Dc, B_out[si].shape[0], chi)
            for c in range(q, p - 1, -1):
                Q[c - p] = u
                u = torch.matmul(u, Gd[:, :, c])
            Cseg.append(u)                                                          # [S, Dc, out, in]
            for c in range(p, q + 1):
                GP = torch.matmul(Gk[c].view(1, 1, chi, chi), P[c - p])             # [S, Dc, l, in]
                F[c] = torch.einsum("sdol,sdli->sdloi", Q[c - p], GP)               # F_c[l, s, t] = Q_c[s, l] GP[l, t]: [S, Dc, l, out, in]
        # per term: matrix elements of the segment chains, O_seg = product of the other segments
        Ce = torch.stack([Cseg[si][:, :, seg_out[:, si], seg_in[:, si]] for si in range(nseg)], -1)   # [S, Dc, E, nseg]
        if nseg == 1:
            O = torch.ones((S, Dc, E, 1), dtype=dtype, device=dev)
        else:
            pre = torch.cumprod(Ce, -1)
            suf = torch.cumprod(Ce.flip(-1), -1).flip(-1)
            ones = torch.ones((S, Dc, E, 1), dtype=dtype, device=dev)
            O = torch.cat([ones, pre[..., :-1]], -1) * torch.cat([suf[..., 1:], ones], -1)
        for c in range(k):
            si = st.seg_of_col[c]
            Fc = F[c][:, :, :, seg_out[:, si], seg_in[:, si]]                       # [S, Dc, L, E]
            contrib = Fc.permute(0, 1, 3, 2) * O[:, :, :, si].unsqueeze(-1)         # [S, Dc, E, L]
            cof[:, :, c] += torch.einsum("d,sdel->sel", sg, contrib)
    return cof * norm


def caterpillar_step_weights(U: torch.Tensor, state: EmitterMPS, T: Sequence[int], rows: Sequence[int],
                             final: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Single-stream reference of the segment-factorised step: w(rho) for all candidates and R."""
    U = torch.as_tensor(U, dtype=C128)
    G = caterpillar_gates(state)
    if G is None:
        raise ValueError("not a caterpillar state (ops[j, l] must be |l><l| G_j)")
    kept = sorted(T)
    f = None if final is None else torch.as_tensor(final, dtype=C128)
    if f is None and state.final is not None:
        f = torch.as_tensor(state.final, dtype=C128)
    st = _Structure(state, G, kept, f)
    modes = torch.tensor([list(state.modes[j]) for j in kept])
    Urows = U[list(rows)].unsqueeze(0)                                              # [1, k-1, m]
    cof = _group_cofactors(st, G, kept, Urows, modes, 1 << 12)[0]                   # [E, k, L]
    amp = torch.einsum("rcl,ecl->er", U[:, modes], cof)                             # [E, m]
    w = (st.p.unsqueeze(1) * amp.abs() ** 2).sum(0)
    return w.clamp_min(0.0), st.R


def sample_caterpillar(state: EmitterMPS, U: torch.Tensor, shots: int, generator: torch.Generator | None = None,
                       device: str = "cpu", dtype: torch.dtype = C128, delta_chunk: int = 1 << 12,
                       return_stats: bool = False, keep=None):
    """Exact samples [shots, m] for a caterpillar state with the segment-factorised terms
    (same chain rule, random photon order and final-spin draw as `sample_torch`; keep [shots, k]
    = surviving emission slots per stream, see there).
    """
    import time
    dev = torch.device(device)
    G = caterpillar_gates(state)
    if G is None:
        raise ValueError("not a caterpillar state (ops[j, l] must be |l><l| G_j)")
    U = torch.as_tensor(U).to(dev, dtype)
    m, n = U.shape[0], state.n
    modes_all = torch.tensor([list(mk) for mk in state.modes], dtype=torch.long, device=dev)
    S = shots
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    if state.final is None:
        pf = final_spin_distribution(state)
        fid = torch.multinomial(pf / pf.sum(), S, replacement=True, generator=generator)
        finals = list(torch.eye(state.chi, dtype=C128))
    else:
        fid = torch.zeros(S, dtype=torch.long)
        finals = [torch.as_tensor(state.final, dtype=C128)]
    from .emitter_sampler import photon_order
    sigma = photon_order(S, n, keep, generator)
    n_k = sigma.shape[1]
    rows = torch.zeros((S, n_k), dtype=torch.long)
    cache: dict[tuple, _Structure] = {}
    stats = {"R": [], "step_s": []}
    for k in range(1, n_k + 1):
        t0 = time.time()
        kk = k - 1
        Tsorted = torch.sort(sigma[:, :k], dim=1).values
        keys = torch.cat([Tsorted, fid.unsqueeze(1)], dim=1)
        uniq, inverse = torch.unique(keys, dim=0, return_inverse=True)
        w = torch.zeros((S, m), dtype=torch.float64)
        Rsum = 0.0
        for ui, key in enumerate(uniq.tolist()):
            kept, fi = key[:-1], key[-1]
            ck = (tuple(kept), fi)
            if ck not in cache:
                cache[ck] = _Structure(state, G, kept, finals[fi])
            st = cache[ck]
            members = (inverse == ui).nonzero(as_tuple=True)[0]
            Urows = U[rows[members][:, :kk].to(dev)]                                # [Sg, kk, m]
            modes = modes_all[torch.tensor(kept, device=dev)]                         # [k, L]
            cof = _group_cofactors(st, G, list(kept), Urows, modes, delta_chunk)      # [Sg, E, k, L]
            amp = torch.einsum("rcl,secl->ser", U[:, modes], cof)                     # [Sg, E, m]
            wg = torch.einsum("e,ser->sr", st.p.to(dev), (amp.abs() ** 2).to(torch.float64))
            w[members] = wg.cpu()
            Rsum += st.R * members.shape[0]
        stats["R"].append(Rsum / S)
        w = w.clamp_min(0)
        w = w / w.sum(dim=1, keepdim=True)
        rho = torch.multinomial(w, 1, generator=generator).squeeze(1)
        rows[:, kk] = rho
        stats["step_s"].append(time.time() - t0)
    occ = torch.zeros((S, m), dtype=torch.long)
    occ.scatter_add_(1, rows, torch.ones_like(rows))
    return (occ, stats) if return_stats else occ
