"""Batched torch implementation of the generalized Clifford sampler for EmitterMPS inputs —
the algorithm of `emitter_reference.matrix_cofactor_weights`, vectorized over S sample
streams in lockstep and over the Glynn terms, on CPU or GPU. No custom kernels: this is the
mid-n oracle (n ≲ 14 on a GPU) and the specification for the Triton/CUDA version.

Per step k (all streams at once):
  1. every stream s has its photon order sigma_s; T_s = sorted(sigma_s[:k]) are its placed
     slots, rows_s its k-1 drawn output modes.
  2. the reduced state of T_s is a sum of R_s pure MPS terms (gap-channel Kraus decomposition,
     `emitter_reference.gap_kraus_terms`, cached per (T, final)); every (stream, term) pair is
     one LANE. Lane tensors: s0 [Λ, chi], mats [Λ, k, L, chi, chi], f [Λ, chi].
  3. column sums over the drawn rows for every Glynn sign vector delta (delta_1 = +1 fixed,
     D = 2^(k-2) of them), per stream: colsum [S, D, k, L]; per lane G[λ, d, j] =
     sum_l colsum * mats (chi x chi).
  4. forward products p_c = (prod_{j<c} G_j) s0 and backward q_c^T = f^dag prod_{j>c} G_j,
     matrix cofactors K_c = 2^-(k-2) sum_d sign_d p_c q_c^T, then for all m candidates
     amp[λ, rho] = sum_c sum_l U[rho, mode(T_c, l)] Tr(mats[λ, c, l] K_c) and
     w[s, rho] = sum_{λ in s} |amp|^2; rho drawn per stream.
The Glynn dimension is chunked (`delta_chunk`) to bound memory: [Λ, D_chunk, k, chi] per
chain position.
"""
from __future__ import annotations

import itertools

import torch

from ..states import EmitterMPS
from .emitter_reference import final_spin_distribution, gap_kraus_terms

C128 = torch.complex128


def _signs(kk: int, device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Delta in {+1} x {+-1}^(kk-1) as [D, kk] (real), sign(delta) [D], and the Glynn norm."""
    if kk == 0:
        return torch.ones((1, 0), dtype=torch.float64, device=device), torch.ones(1, dtype=torch.float64, device=device), 1.0
    if kk == 1:
        d = torch.ones((1, 1), dtype=torch.float64, device=device)
    else:
        grid = torch.tensor(list(itertools.product([1.0, -1.0], repeat=kk - 1)), dtype=torch.float64, device=device)
        d = torch.cat([torch.ones((grid.shape[0], 1), dtype=torch.float64, device=device), grid], dim=1)
    return d, d.prod(dim=1), 2.0 ** (-(kk - 1))


def photon_order(S: int, n: int, keep, generator) -> torch.Tensor:
    """Random photon order per stream, [S, k] (cpu long): a permutation of all n emission slots,
    or of the stream's surviving slots `keep` [S, k] (input-side loss, docs/LOSS.md).
    """
    if keep is None:
        return torch.argsort(torch.rand((S, n), generator=generator), dim=1)
    keep = torch.as_tensor(keep, dtype=torch.long).cpu()
    if keep.dim() != 2 or keep.shape[0] != S:
        raise ValueError("keep must be [shots, k]")
    perm = torch.argsort(torch.rand(keep.shape, generator=generator), dim=1)
    return keep.gather(1, perm)


class _TermCache:
    """gap_kraus_terms per (sorted T, final id), stacked as tensors on the device."""

    def __init__(self, state: EmitterMPS, device, dtype, U=None):
        self.state, self.device, self.dtype, self.U = state, device, dtype, U
        self.cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        chi = state.chi
        self.finals = [None] if state.final is not None else list(torch.eye(chi, dtype=C128))

    def get(self, T: tuple[int, ...], fid: int, labels=None):
        key = (T, fid, labels)
        if key not in self.cache:
            _, terms = gap_kraus_terms(self.state, list(T), final=self.finals[fid], labels=labels, U=self.U)
            s0 = torch.stack([t[0] for t in terms]).to(self.device, self.dtype)
            mats = torch.stack([t[1] for t in terms]).to(self.device, self.dtype)
            f = torch.stack([t[2] for t in terms]).to(self.device, self.dtype)
            self.cache[key] = (s0, mats, f)
        return self.cache[key]


def sample_torch(state: EmitterMPS, U: torch.Tensor, shots: int,
                         generator: torch.Generator | None = None, device: str = "cpu",
                         dtype: torch.dtype = C128, delta_chunk: int = 1 << 12,
                         return_stats: bool = False, keep=None, lost_labels=None, final_ids=None):
    """Exact samples [shots, m] (int64 occupations, on `device`). return_stats adds a dict
    with the mean number of lanes per stream at each step (R_k) and timings. keep [shots, k]:
    the emission slots that survive per stream (the others are traced out = lost photons);
    `emitter_sample(loss=eta)` builds these groups. lost_labels [shots, n] / final_ids [shots]
    (`emitter_terms.sample_slot_outcomes`): the traced slots' sampled labels (x < L) and the
    singleton slots' sampled output rows (L + r, docs/DISTINGUISHABILITY.md) and the final spin
    drawn with them — the label unravelling of the loss (docs/LOSS.md §3b), one term per gap.
    A singleton slot is folded into the chain at its row; its photon is added to the output by
    the caller (`emitter_sample`), not here.
    """
    import time
    dev = torch.device(device)
    U = torch.as_tensor(U).to(dev, dtype)
    m, n, L, chi = U.shape[0], state.n, state.num_labels, state.chi
    modes = torch.tensor([list(mk) for mk in state.modes], dtype=torch.long, device=dev)   # [n, L]
    S = shots
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    # final spin per stream (traced emitter: sampled from p(s_f); projected: id 0)
    if final_ids is not None:
        fid = torch.as_tensor(final_ids, dtype=torch.long).cpu()
    elif state.final is None:
        pf = final_spin_distribution(state)
        fid = torch.multinomial(pf / pf.sum(), S, replacement=True, generator=generator)
    else:
        fid = torch.zeros(S, dtype=torch.long)
    lab = None if lost_labels is None else torch.as_tensor(lost_labels, dtype=torch.long).cpu()
    sigma = photon_order(S, n, keep, generator)                                           # [S, n_k]
    n_k = sigma.shape[1]
    rows = torch.zeros((S, n_k), dtype=torch.long)                                        # cpu
    cache = _TermCache(state, dev, dtype, U=U)
    stats = {"R": [], "step_s": []}
    for k in range(1, n_k + 1):
        t0 = time.time()
        kk = k - 1
        Tsorted = torch.sort(sigma[:, :k], dim=1).values                                  # [S, k] cpu
        # ---- lanes
        s0_l, mats_l, f_l, lane_stream = [], [], [], []
        # group streams by (T, final) so each distinct reduced state is fetched once
        keys = torch.cat([Tsorted, fid.unsqueeze(1)] + ([] if lab is None else [lab]), dim=1)
        uniq, inverse = torch.unique(keys, dim=0, return_inverse=True)
        for ui, key in enumerate(uniq.tolist()):
            s0, mats, f = cache.get(tuple(key[:k]), key[k], None if lab is None else tuple(key[k + 1:]))
            members = (inverse == ui).nonzero(as_tuple=True)[0]                          # streams with this key
            R = s0.shape[0]
            s0_l.append(s0.repeat(members.shape[0], 1))
            mats_l.append(mats.repeat(members.shape[0], 1, 1, 1, 1))
            f_l.append(f.repeat(members.shape[0], 1))
            lane_stream.append(members.repeat_interleave(R))
        s0 = torch.cat(s0_l); mats = torch.cat(mats_l); f = torch.cat(f_l)              # [Λ, ...]
        lane_stream = torch.cat(lane_stream).to(dev)
        Lam = s0.shape[0]
        stats["R"].append(Lam / S)
        # ---- column sums per stream: colsum[s, d, j, l] = sum_i delta[d, i] U[rows[s, i], modes[T[s, j], l]]
        deltas, sign, norm = _signs(kk, dev)
        D = deltas.shape[0]
        Tdev = Tsorted.to(dev)
        col_modes = modes[Tdev]                                                            # [S, k, L]
        Urows = U[rows[:, :kk].to(dev)]                                                    # [S, kk, m]
        Ucols = torch.gather(Urows.unsqueeze(2).expand(S, kk, k, m), 3,
                             col_modes.unsqueeze(1).expand(S, kk, k, L))                   # [S, kk, k, L]
        K_acc = torch.zeros((Lam, k, chi, chi), dtype=dtype, device=dev)
        for d0 in range(0, D, delta_chunk):
            dl = deltas[d0:d0 + delta_chunk].to(dtype)                                    # [Dc, kk]
            sg = sign[d0:d0 + delta_chunk].to(dtype)
            colsum = torch.einsum("di,sijl->sdjl", dl, Ucols) if kk else torch.ones((S, 1, k, L), dtype=dtype, device=dev)
            cs = colsum[lane_stream]                                                       # [Λ, Dc, k, L]
            G = torch.einsum("xdjl,xjlts->xdjts", cs, mats)                                # [Λ, Dc, k, chi, chi]
            Dc = G.shape[1]
            # forward / backward chains
            p = torch.empty((Lam, Dc, k, chi), dtype=dtype, device=dev)
            v = s0.unsqueeze(1).expand(Lam, Dc, chi)
            for c in range(k):
                p[:, :, c] = v
                v = torch.einsum("xdts,xds->xdt", G[:, :, c], v)
            q = torch.empty((Lam, Dc, k, chi), dtype=dtype, device=dev)
            u = f.conj().unsqueeze(1).expand(Lam, Dc, chi)
            for c in range(k - 1, -1, -1):
                q[:, :, c] = u
                u = torch.einsum("xds,xdst->xdt", u, G[:, :, c])
            K_acc += torch.einsum("d,xdct,xdcs->xcts", sg, p, q)
            del G, p, q, cs, colsum
        K_acc *= norm
        # ---- amplitudes for all candidates: amp[λ, rho] = sum_c sum_l U[rho, mode(T_c, l)] Tr(mats[λ,c,l] K_c)
        tr = torch.einsum("xclts,xcst->xcl", mats, K_acc)                                  # [Λ, k, L]
        Ucand = U[:, col_modes[lane_stream]]                                               # [m, Λ, k, L]
        amp = torch.einsum("rxcl,xcl->xr", Ucand, tr)                                      # [Λ, m]
        w = torch.zeros((S, m), dtype=torch.float64, device=dev)
        w.index_add_(0, lane_stream, (amp.abs() ** 2).to(torch.float64))
        w = w.clamp_min(0)
        w = w / w.sum(dim=1, keepdim=True)
        rho = torch.multinomial(w.cpu(), 1, generator=generator).squeeze(1)
        rows[:, kk] = rho
        stats["step_s"].append(time.time() - t0)
    occ = torch.zeros((S, m), dtype=torch.long)
    occ.scatter_add_(1, rows, torch.ones_like(rows))
    return (occ, stats) if return_stats else occ
