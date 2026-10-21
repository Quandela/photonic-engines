"""Internal (temporal / frequency) photon modes: the fine Kraus set of one bin — docs/SEQUENCE.md §14.

A photon of emitter j leaves with label l in internal mode c (operator A_j[l, c] on the product register); the network
is U (x) 1 on (spatial x internal) modes and the detectors sum the internal index out. The channel of a spatial
outcome t is

    E_t(rho) = sum_{tau -> t} K_tau rho K_tau^dag,     K_tau = sqrt(tau!) <tau| Phi>,

over the fine patterns tau on the M = m K_b modes (spatial i, internal c) -> c m + i (internal-major) whose sum over c
is t: the bond-carrying SLOS recurrence of `slos.emitter`, run on the populated internal span K_b. This module builds
that set GPU-first: per photon one einsum gives the coefficient matrix of every raised mode,

    C[(c, i)] = sum_l U[i, in_j(l)] A_j[l, c]          [M, chi, chi]   (no dense U (x) 1_K),

and the layer is a batched gather / contract / index_add over chunks of raised modes padded to the largest slab (no
Python loop per mode), chunked so that one chunk's working set stays under a byte budget.

Span reduction (`span`): the channel is invariant under A -> A V for V unitary on the internal space, common to every
emitter of the bin. Default: drop the internal modes on which every operator is exactly zero (no arithmetic on the
others). With `rtol`: rotate to the singular basis of the stacked coefficients [K, sum_j L_j chi^2] and keep the
singular values above rtol * s_max (the channel moves by O((rtol s_max)^2)).

Compression (`compress_by_outcome`): per outcome t, the operators K_q (q -> t) as rows of B_t [F_t, chi^2]; with the
singular value decomposition B_t = U diag(s) Vh (of B_t itself, so singular values down to ~eps s_max are resolved),
the rows s_i Vh_i with s_i > rtol * s_max(t) give the same channel up to the dropped (s_i)^2: the Choi-rank set of the
outcome. Outcomes are batched by their fine-set size F_t.

On CUDA the scatter of a layer (index_add_) accumulates in a nondeterministic order, so repeated runs agree to
rounding, not bit for bit; on the CPU they are bitwise reproducible.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .basis import get_layers


def span(ops: list[torch.Tensor], rtol: float | None = None) -> list[torch.Tensor]:
    """Reduce the internal axis of one bin's operators. ops: per emitter [L_j, K, chi, chi] on a shared K.
    Returns the same list on the populated span [L_j, K_b, chi, chi] (K_b >= 1).
    """
    K = ops[0].shape[1]
    V = torch.cat([a.permute(1, 0, 2, 3).reshape(K, -1) for a in ops], 1)                      # [K, sum L chi^2]
    if rtol is None:
        keep = torch.nonzero((V != 0).any(1)).reshape(-1)
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=V.device)
        return [a[:, keep] for a in ops]
    Q, s, _ = torch.linalg.svd(V, full_matrices=False)                                         # V = Q diag(s) W^dag
    Kb = max(1, int((s > rtol * s[0]).sum()))
    Q = Q[:, :Kb]
    return [torch.einsum("kc,lkab->lcab", Q.conj(), a) for a in ops]


_STAGES: dict[tuple[int, int, int, str], FineStage] = {}


def fine_stage(k: int, m: int, Kb: int, device) -> FineStage:
    """The cached FineStage of (k, m, K_b, device), shared by every kernel (as `get_layers` is)."""
    key = (k, m, Kb, str(torch.device(device)))
    if key not in _STAGES:
        _STAGES[key] = FineStage(k, m, Kb, device)
    return _STAGES[key]


class FineStage:
    """The fine layers of one bin shape: k photons on m K_b modes (internal-major), with the per-layer edge tables
    padded per raised mode, and the map fine pattern -> spatial pattern. Built once per (k, m, K_b).
    """

    def __init__(self, k: int, m: int, Kb: int, device):
        self.k, self.m, self.Kb, self.M = k, m, Kb, m * Kb
        self.device = torch.device(device)
        self.layers = get_layers(k, self.M, device="cpu")          # host index tables; the padded ones go to device
        self.parent, self.child, self.size = [], [], []
        for j in range(k):
            p, c, off = self.layers.edges_by_mode(j)
            p, c, off = np.asarray(p), np.asarray(c), np.asarray(off)
            n_next = self.layers.size(j + 1)
            e = np.diff(off)
            emax = max(1, int(e.max()) if e.size else 1)
            P = np.zeros((self.M, emax), dtype=np.int64)
            C = np.full((self.M, emax), n_next, dtype=np.int64)                                 # pad -> dummy slot
            for r in range(self.M):                                                              # table build, once
                P[r, :e[r]] = p[off[r]:off[r + 1]]
                C[r, :e[r]] = c[off[r]:off[r + 1]]
            self.parent.append(torch.as_tensor(P, device=self.device))
            self.child.append(torch.as_tensor(C, device=self.device))
            self.size.append(n_next)
        occ = self.layers.final_occupations()
        self.fine_occ = occ
        self.spatial_occ = occ.reshape(occ.shape[0], Kb, m).sum(1)                              # [S_fine, m]
        self.norm = torch.sqrt(torch.as_tensor(self.layers.output_prob_norms(), dtype=torch.float64))
        self.S_fine = occ.shape[0]


def fine_kraus(U: torch.Tensor, ops: list[torch.Tensor], in_modes: list[torch.Tensor], stage: FineStage,
               chunk_bytes: int = 1 << 30) -> torch.Tensor:
    """Physical fine Kraus operators [S_fine, chi, chi] (final x initial) of one bin.
    U [m, m]; ops: per emitter (= photon j) [L_j, K_b, chi, chi] on the product register; in_modes: per emitter the
    input column of each label [L_j].
    """
    chi = ops[0].shape[-1]
    dt, dev = ops[0].dtype, ops[0].device
    psi = torch.eye(chi, dtype=dt, device=dev).unsqueeze(-1)                                     # [Q=init, chi, 1]
    Q = chi
    for j, A in enumerate(ops):
        Uj = U[:, in_modes[j]].to(dt)                                                            # [m, L]
        C = torch.einsum("il,lcts->cits", Uj, A).reshape(stage.M, chi, chi)                      # raised mode c m + i
        P, Cc, n_next = stage.parent[j], stage.child[j], stage.size[j]
        emax = P.shape[1]
        G = max(1, min(stage.M, chunk_bytes // max(1, Q * chi * emax * 16)))
        nxt = torch.zeros((Q, chi, n_next + 1), dtype=dt, device=dev)
        for g0 in range(0, stage.M, G):
            g1 = min(stage.M, g0 + G)
            x = psi[:, :, P[g0:g1]]                                                              # [Q, chi, G, e]
            y = torch.einsum("gts,qsge->qtge", C[g0:g1], x)
            nxt.index_add_(2, Cc[g0:g1].reshape(-1), y.reshape(Q, chi, -1))
        psi = nxt[:, :, :n_next]
    K = psi.permute(2, 1, 0)                                                                     # [S_fine, final, init]
    return K * stage.norm.to(device=dev, dtype=K.real.dtype).view(-1, 1, 1).to(dt)


def compress_by_outcome(K: torch.Tensor, coarse: torch.Tensor, rtol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per outcome, the minimal operator set of the same channel (singular values of the stacked operators above
    rtol * s_max of that outcome). K [Q, chi, chi], coarse [Q] -> (K' [Q', chi, chi], coarse' [Q']).
    """
    chi = K.shape[-1]
    D2 = chi * chi
    order = torch.argsort(coarse, stable=True)
    Ks, cs = K[order], coarse[order]
    outs, counts = torch.unique_consecutive(cs, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    res_K, res_c = [], []
    for F in torch.unique(counts).tolist():                                                     # batch by fine-set size
        sel = torch.nonzero(counts == F).reshape(-1)
        idx = (starts[sel].unsqueeze(1) + torch.arange(F, device=K.device)).reshape(-1)
        B = Ks[idx].reshape(sel.numel(), F, D2)                                                  # [T, F, chi^2]
        # SVD of B itself (not eigh of a Gram matrix, which squares the singular values and floors them at
        # sqrt(eps) s_max): B = U diag(s) Vh, rows s_i Vh_i reproduce B^dag B = sum_q vec K_q^* vec K_q^T
        _, s, Vh = torch.linalg.svd(B, full_matrices=False)                                      # [T, r], r = min(F, chi^2)
        rows = s.unsqueeze(-1) * Vh                                                              # [T, r, chi^2]
        keep = s > rtol * s[:, :1]
        res_K.append(rows[keep].reshape(-1, chi, chi))
        res_c.append(outs[sel].unsqueeze(1).expand_as(keep)[keep])
    Kc, cc = torch.cat(res_K), torch.cat(res_c)
    o = torch.argsort(cc, stable=True)
    return Kc[o], cc[o]


# ---- the pairing route (P'): the outcome channel applied to the register, cost independent of K ------------------
#
# Wick's theorem on the emitted monomials gives, with O_j[r, c] = sum_l U[r, in_j(l)] A_j[l, c] and the rows
# R_1..R_n of the outcome (with repetition),
#     E_t(rho) = (1/t!) sum_sigma sum_pi sum_c (prod_j O_j[R_sigma(j), c_j]) rho (prod_j O_pi(j)[R_sigma(j), c_j])^dag:
# ket photon j and bra photon pi(j) share the row and the internal mode, so K enters only through the pair Gram
#     Gamma_jj'[l, l'] = sum_c A_j[l, c] (x) conj A_j'[l', c]      (a d^2 x d^2 local superoperator on (ket j, bra j')).
# Glynn on both sides (event signs delta, bra-slot signs eps, first entries +1; only bijections survive the sign sums,
# also for the ordered, non-commuting slot actions):
#     E_t(rho) = (1/(t! 4^(n-1))) sum_delta sum_eps (prod delta)(prod eps)
#                [prod_j sum_j' eps_j' sum_l,l' c_delta[j, j', l, l'] Gamma_jj'[l, l']](rho),
#     c_delta[j, j', l, l'] = sum_e delta_e U[R_e, in_j(l)] conj U[R_e, in_j'(l')].
# Each term is n x n local applications on the 2n-index register tensor; the terms, the outcomes and the input batch
# are tensor axes (chunked over outcomes), the only Python loops are over the n x n photon pairs.
# Certificate (a-posteriori estimate, measured margin >= 12x; not a worst-case bound):
#     |Delta E_t(rho)|_F ~<= c n^2 eps (1/(t! 4^(n-1))) sum_terms |X_term|_F, accumulated in the same pass.

_CERT_C = 4.0


def glynn_signs(n: int, device, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """The 2^(n-1) sign vectors [P, n] with first entry +1, and their products [P]."""
    idx = torch.arange(2 ** (n - 1), device=device)
    bits = (idx.unsqueeze(1) >> torch.arange(n - 1, device=device)) & 1
    s = torch.cat([torch.ones(idx.shape[0], 1, dtype=torch.long, device=device), 1 - 2 * bits], 1)
    return s.to(dtype), s.prod(1).to(dtype)


def pair_gram(A: torch.Tensor) -> torch.Tensor:
    """Gamma [n, n, L, L, d, d, d, d] from local operators A [n, L, K_b, d, d]: Gamma[j, j', l, l', a, a', x, x'] =
    sum_c A_j[l, c][a, a'] conj A_j'[l', c][x, x'] (ket out, ket in, bra out, bra in).
    """
    return torch.einsum("jlcab,kmcxy->jklmabxy", A, A.conj())


def _apply_letters(n: int, j: int, jp: int):
    """The einsum string applying a local superoperator [T, P, a, a', x, x'] on ket axis j, bra axis jp of X
    [B, T, P, Q, ket_0..ket_n-1, bra_0..bra_n-1].
    """
    letters = "cdefghijklmnorsuvwyzACEFGHIJKLMNORSUVWYZ"
    ket, bra = letters[:n], letters[n:2 * n]
    new_k, new_b = "a", "x"
    X = "btpq" + ket + bra
    G = "tp" + new_k + ket[j] + new_b + bra[jp]
    out = "btpq" + ket[:j] + new_k + ket[j + 1:] + bra[:jp] + new_b + bra[jp + 1:]
    return f"{G},{X}->{out}"


def pairing_apply(U: torch.Tensor, A: torch.Tensor, in_modes: torch.Tensor, rows: torch.Tensor, tfact: torch.Tensor,
                  rho: torch.Tensor, chunk_bytes: int = 1 << 30,
                  x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """E_t(rho) for every outcome t of `rows`, by the pairing form.
    U [m, m]; A [n, L, K_b, d, d] local spin operators (photon j = emitter j; equal d and L, zero-padded labels);
    in_modes [n, L] input column of each label; rows [T, n] the outcome's event rows (with repetition); tfact [T] t!;
    rho [B, D, D] with D = d^n (emitter j = the j-th tensor factor). Returns (E [B, T, D, D], cert [B, T]): cert is an
    a-posteriori estimate of the Frobenius norm of the rounding error of each E_t(rho) (term norms, not a worst-case
    bound: the constant leaves out the accumulation lengths; measured margin >= 12x against complex64 runs, k = 2-4).
    x [n] (optional): partial distinguishability, photon j in the common internal mode with probability x_j and
    otherwise in a private one (the x-model): Gamma_jj' is scaled by sqrt(x_j x_j') for j != j' (Gamma_jj unchanged),
    exactly the pair Gram of the emitters extended by private internal modes.
    """
    n, d = A.shape[0], A.shape[3]
    D, Bn, T = d ** n, rho.shape[0], rows.shape[0]
    dt, dev = rho.dtype, rho.device
    Gam = pair_gram(A)                                                                           # [n, n, L, L, d^4]
    if x is not None:
        w = torch.sqrt(torch.outer(x, x)).to(torch.float64)
        w.fill_diagonal_(1.0)
        Gam = Gam * w.to(device=dev, dtype=Gam.dtype).view(n, n, *([1] * (Gam.dim() - 2)))
    sg, sgp = glynn_signs(n, dev)
    P = sg.shape[0]
    eps_sign = sg.to(dt).view(1, 1, 1, P, n)                                                     # bra-slot signs eps
    norm = 4.0 ** (n - 1)
    eps = torch.finfo(rho.real.dtype).eps
    Tc = max(1, int(chunk_bytes // max(1, Bn * P * P * D * D * 16 * 3)))
    sub = [_apply_letters(n, j, jp) for j in range(n) for jp in range(n)]
    E_out = torch.empty((Bn, T, D, D), dtype=dt, device=dev)
    cert = torch.empty((Bn, T), dtype=rho.real.dtype, device=dev)
    w = (sgp.unsqueeze(1) * sgp.unsqueeze(0)).to(dt)                                            # [P(delta), P(eps)]
    for t0 in range(0, T, Tc):
        t1 = min(T, t0 + Tc)
        R = rows[t0:t1]                                                                          # [Tc, n]
        Ur = U[R][:, :, in_modes]                                                                # [Tc, e, j, l]
        c = torch.einsum("pe,tejl,tekm->tpjklm", sg.to(dt), Ur, Ur.conj())                       # [Tc, P, n, n, L, L]
        G = torch.einsum("tpjklm,jklmabxy->tpjkabxy", c, Gam)                                    # [Tc, P, n, n, d^4]
        X = rho.reshape(Bn, 1, 1, 1, *([d] * (2 * n))).expand(Bn, t1 - t0, P, P, *([d] * (2 * n)))
        for j in range(n):
            Xn = None
            for jp in range(n):
                Y = torch.einsum(sub[j * n + jp], G[:, :, j, jp], X)
                Y = Y * eps_sign[..., jp].reshape(1, 1, 1, P, *([1] * (2 * n)))
                Xn = Y if Xn is None else Xn + Y
            X = Xn
        Xf = X.reshape(Bn, t1 - t0, P, P, D, D)
        scale = (tfact[t0:t1].to(dev) * norm).view(1, -1, 1, 1)
        E_out[:, t0:t1] = torch.einsum("pq,btpqxy->btxy", w, Xf) / scale
        mass = torch.linalg.matrix_norm(Xf).sum((2, 3))                                          # sum of |X_term|_F
        cert[:, t0:t1] = _CERT_C * n * n * eps * mass / scale.view(1, -1)
    return E_out, cert


def _letters_lost(n: int, j: int, jp: int, pair: bool):
    """Einsum strings for the z-graded register X [B, T, P, Q, D(egree), ket_0..ket_n-1, bra_0..bra_n-1]: a slot-pair
    operator G [T, P, a, a', x, x'] on (ket j, bra jp) (pair=True), or a local absent-photon superoperator
    L [a, a', x, x'] on (ket j, bra j) (pair=False).
    """
    letters = "cdefghijklmnorsuvwyzACEFGHIJKLMNORSUVWYZ"
    ket, bra = letters[:n], letters[n:2 * n]
    X = "btpqB" + ket + bra
    G = ("tp" if pair else "") + "a" + ket[j] + "x" + bra[jp]
    out = "btpqB" + ket[:j] + "a" + ket[j + 1:] + bra[:jp] + "x" + bra[jp + 1:]
    return f"{G},{X}->{out}"


def pairing_apply_lost(U: torch.Tensor, A: torch.Tensor, in_modes: torch.Tensor, rows: torch.Tensor,
                       tfact: torch.Tensor, rho: torch.Tensor, Lam: torch.Tensor, chunk_bytes: int = 1 << 30,
                       x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """E_t(rho) for outcomes of n' = rows.shape[1] detected photons (0 <= n' <= n) when photons can be lost:
    photon j's factor is delta_jj' Lam_j + z sum_e T_jj'[R_e] (Lam_j = sum_q N_q (x) conj N_q, the absent photon,
    on ket j and bra j), and E_t is the coefficient of z^n'. The register tensor carries the degree
    0..n' (higher degrees never come back down: dropped). Glynn over the n' events (first sign +1, valid at degree n')
    and over the n bra slots; normalisation t! 2^max(n'-1, 0) 2^(n-1). n' = 0: E = (x)_j Lam_j(rho).
    U, A, in_modes, tfact, rho, chunk_bytes, x as in `pairing_apply`; rows [T, n']; Lam [n, d, d, d, d] (ket out,
    ket in, bra out, bra in). Returns (E [B, T, D, D], cert [B, T]), cert the same a-posteriori estimate.
    """
    n, d = A.shape[0], A.shape[3]
    npr = rows.shape[1]
    D, Bn, T = d ** n, rho.shape[0], rows.shape[0]
    dt, dev = rho.dtype, rho.device
    Gam = pair_gram(A)
    if x is not None:
        w = torch.sqrt(torch.outer(x, x)).to(torch.float64)
        w.fill_diagonal_(1.0)
        Gam = Gam * w.to(device=dev, dtype=Gam.dtype).view(n, n, *([1] * (Gam.dim() - 2)))
    sb, sbp = glynn_signs(n, dev)                                                                # bra slots
    if npr:
        se, sep = glynn_signs(npr, dev)                                                          # events
    else:
        se, sep = torch.ones(1, 0, dtype=torch.float64, device=dev), torch.ones(1, dtype=torch.float64, device=dev)
    Pe, Pb, G1 = se.shape[0], sb.shape[0], npr + 1
    eps_sign = sb.to(dt)                                                                         # [Pb, n]
    norm = float(2 ** max(npr - 1, 0) * 2 ** (n - 1))
    eps = torch.finfo(rho.real.dtype).eps
    Tc = max(1, int(chunk_bytes // max(1, Bn * Pe * Pb * G1 * D * D * 16 * 3)))
    pair_sub = [[_letters_lost(n, j, jp, True) for jp in range(n)] for j in range(n)]
    lam_sub = [_letters_lost(n, j, j, False) for j in range(n)]
    Lam = Lam.to(dt)
    E_out = torch.empty((Bn, T, D, D), dtype=dt, device=dev)
    cert = torch.empty((Bn, T), dtype=rho.real.dtype, device=dev)
    wgt = (sep.unsqueeze(1) * sbp.unsqueeze(0)).to(dt)                                           # [Pe, Pb]
    shp = [d] * (2 * n)

    def bra_sign(jp):
        return eps_sign[:, jp].reshape(1, 1, 1, Pb, 1, *([1] * (2 * n)))

    for t0 in range(0, T, Tc):
        t1 = min(T, t0 + Tc)
        Tg = t1 - t0
        X = torch.zeros((Bn, Tg, Pe, Pb, G1, *shp), dtype=dt, device=dev)
        X[:, :, :, :, 0] = rho.reshape(Bn, 1, 1, 1, *shp)
        if npr:
            R = rows[t0:t1]
            Ur = U[R][:, :, in_modes]                                                            # [Tg, e, j, l]
            c = torch.einsum("pe,tejl,tekm->tpjklm", se.to(dt), Ur, Ur.conj())
            Gp = torch.einsum("tpjklm,jklmabxy->tpjkabxy", c, Gam)                               # [Tg, Pe, n, n, d^4]
        for j in range(n):
            Xn = torch.einsum(lam_sub[j], Lam[j], X) * bra_sign(j)                              # photon j absent
            if npr:
                up = None
                for jp in range(n):
                    Y = torch.einsum(pair_sub[j][jp], Gp[:, :, j, jp], X[:, :, :, :, :-1]) * bra_sign(jp)
                    up = Y if up is None else up + Y
                Xn[:, :, :, :, 1:] += up                                                         # photon j detected: z
            X = Xn
        Xf = X[:, :, :, :, npr].reshape(Bn, Tg, Pe, Pb, D, D)
        scale = (tfact[t0:t1].to(dev) * norm).view(1, -1, 1, 1)
        E_out[:, t0:t1] = torch.einsum("pq,btpqxy->btxy", wgt, Xf) / scale
        mass = torch.linalg.matrix_norm(Xf).sum((2, 3))
        cert[:, t0:t1] = _CERT_C * n * n * eps * mass / scale.view(1, -1)
    return E_out, cert


def lost_superop(N: torch.Tensor) -> torch.Tensor:
    """Lam = sum_q N_q (x) conj N_q as [d, d, d, d] (ket out, ket in, bra out, bra in) from N [Q, d, d]."""
    return torch.einsum("qab,qxy->abxy", N, N.conj())


def emission_channel(A: torch.Tensor, rho: torch.Tensor, lost: list[torch.Tensor] | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """The sum over every outcome of E_t(rho) (lossless): the emission channel itself, prod_j Phi_j with
    Phi_j(X) = sum_{l,c} A_j[l, c] X A_j[l, c]^dag on factor j (tracing the photons after U = before U).
    A [n, L, K_b, d, d], rho [B, D, D] -> (E [B, D, D], cert [B]): cert bounds the Frobenius norm of the rounding error,
    c n eps |prod_j |Phi_j| (|rho|)|_F with |Phi_j| the same map on entrywise moduli (a modulus bound: Phi has no
    alternating signs). Valid for a unitary interferometer only (the identity needs U^dag U = 1). lost (per emitter
    [Q_j, d, d], optional): Phi_j also carries sum_q N_q X N_q^dag (the outcomes with photon j undetected).
    """
    n, d = A.shape[0], A.shape[3]
    Bn = rho.shape[0]
    X = rho.reshape(Bn, *([d] * (2 * n)))
    M = X.abs().to(X.dtype)
    Aa = A.abs().to(A.dtype)
    letters = "cdefghijklmnorsuvwyz"
    ket, bra = letters[:n], letters[n:2 * n]
    for j in range(n):
        out = "b" + ket[:j] + "a" + ket[j + 1:] + bra[:j] + "x" + bra[j + 1:]
        sub = f"LQaA,LQxX,b{ket[:j]}A{ket[j + 1:]}{bra[:j]}X{bra[j + 1:]}->{out}"
        Y = torch.einsum(sub, A[j], A[j].conj(), X)
        Ma = torch.einsum(sub, Aa[j], Aa[j], M)
        if lost is not None and lost[j].shape[0]:
            Nj = lost[j].to(A.dtype).unsqueeze(1)                                                # [Q, 1, d, d]
            Y = Y + torch.einsum(sub, Nj, Nj.conj(), X)
            Ma = Ma + torch.einsum(sub, Nj.abs().to(A.dtype), Nj.abs().to(A.dtype), M)
        X, M = Y, Ma
    eps = torch.finfo(rho.real.dtype).eps
    cert = _CERT_C * n * eps * M.reshape(Bn, -1).abs().norm(dim=1)
    return X.reshape(Bn, d ** n, d ** n), cert


# ---- the range route (R): the compressed channel of every outcome without the fine set ----------------------------
#
# Range. For random per-event internal vectors w_e, K(w) = sum_sigma (x)_j O_j[R_sigma(j), w_sigma(j)] with
# O_j[r, w] = sum_c w_c O_j[r, c] is a linear combination of the fine operators of t, so vec K(w) lies in the range of
# the outcome's Choi matrix J_t = sum_tau vec K_tau vec K_tau^dag; each sample is one operator-valued permanent (Glynn,
# 2^(n-1) Kronecker products of n d x d operators). Samples are added (doubling) until the numerical rank r of the
# stack is below its column count minus the oversampling, or reaches min(F_t, D^2) (F_t: fine patterns of t). The rank
# cut is tol / 8 relative (clamped to [8 eps, 1e3 eps]), not rtol: a direction dropped from the range at relative size c
# moves J_t by ~c (cross terms with the kept part), so the range must be complete to tol; rtol acts on the core only.
# Core. J_t = Q C Q^dag with Q [D^2, r] orthonormal. With probes Y [D, a] (a = ceil(2r / D); a >= D: Y = 1, the
# matrix units), the pairing route applied to the a^2 rank-one inputs y_i y_j^dag gives the block matrix
#     Big[(i, u), (j, v)] = <u| E_t(y_i y_j^dag) |v> = (A C A^dag)[(i, u), (j, v)],   A[(i, u), q] = (Q_q y_i)_u,
# so C = A^+ Big A^+dag (thin QR of A, exact when A has full column rank, r <= a D), and eigh(C) = V diag(lam) V^dag
# gives the operators sqrt(lam_i) Q V_i, keeping lam_i > rtol^2 lam_max (the compression rule of
# `compress_by_outcome`: singular values s_i = sqrt(lam_i) above rtol s_max) and lam_i above the core's error level
# err / sqrt(r) (eigenvalues at rounding level are not resolved: dropped, and their norm added to the certificate).
# So on a spread spectrum (eigenvalues of J_t between ~1e-15 lam_max and rtol^2 lam_max) R returns fewer operators
# than F's Choi-rank set, within tol: R's count is the rank resolvable at the core's absolute error (~1e-15), F's the
# rank at rtol; they agree when the spectrum's gap at rtol is above that level (generic and low-rank emitters).
# The rounding of Q itself (the SVD of the samples) and of the reconstruction Q V sqrt(lam) adds c eps sqrt(r) |C|_F
# (about 1e-16 on J / D; at D = 2 as large as the rest: run 20261020_213551 measured 5.3e-16 against an
# estimate of 4.6e-16 without it).
# Error of J_t (Frobenius): err = (|pairing certificates of the a^2 blocks|_2 + |Big - P_A Big P_A|_F) /
# sigma_min(A)^2; the residual term is the part of Big outside the sampled range (an incomplete range shows up here).
# An a-posteriori ESTIMATE, not a bound: a direction q missed by the range enters the residual only through the random
# probe map (|P_A^perp M q|), with no lower bound relative to its mass (with the matrix units, Y = 1, the map is an
# isometry and the residual is exact). Measured (CPU, k = 2-3, K = 3-8, generic and label-preserving emitters, the
# samples projected off one eigen-direction of J_t, missed eigenvalue 2e-5 .. 1 of the largest; 396 cases):
# certificate / true error >= 0.58, median 3.5 — a miss is detected once its error exceeds ~1.7 tol.
# An outcome above tol D is recomputed from the matrix units on the whole operator space (Q = 1, C = J_t); one still
# above tol is returned as failed (the caller falls back to F, or raises). The tolerance assumes fp64 (eps of float64):
# a complex64 kernel does not reach tol = 1e-14 and falls back.

_RANGE_FLOOR = 1e3        # largest range cut, x eps relative to the largest sample singular value
_RANGE_MIN = 8.0          # smallest range cut (rounding floor of the samples), x eps
_OVERSAMPLE = 4
_PROBE_OVERSAMPLE = 2     # probes a = ceil(c r / D): the core's conditioning sigma_min(A) improves with c


def fine_counts(pats: torch.Tensor, Kb: int) -> torch.Tensor:
    """F_t = prod_i C(K_b + t_i - 1, t_i), the number of fine patterns of each spatial pattern [T, m] -> [T]."""
    out = torch.ones(pats.shape[0], dtype=torch.float64)
    for x in torch.unique(pats).tolist():
        if x:
            out *= torch.where(pats == x, float(math.comb(Kb + x - 1, x)), 1.0).prod(1)
    return out.round().to(torch.long)


def _crandn(gen, *shape) -> torch.Tensor:
    """Standard complex Gaussian (host generator: reproducible across devices)."""
    z = torch.randn(*shape, 2, generator=gen, dtype=torch.float64)
    return torch.complex(z[..., 0], z[..., 1]) / math.sqrt(2.0)


def range_samples(U: torch.Tensor, A: torch.Tensor, in_modes: torch.Tensor, rows: torch.Tensor,
                  w: torch.Tensor) -> torch.Tensor:
    """The range samples vec K(w) [T, D^2, s] for the outcomes of `rows` [T, n] and internal vectors w [T, s, n, K_b]
    (per event). U, A, in_modes as in `pairing_apply`.
    """
    n, d = A.shape[0], A.shape[3]
    T, s = w.shape[:2]
    sg, sgp = glynn_signs(n, A.device)
    Ur = U[rows][:, :, in_modes].to(A.dtype)                                                     # [T, e, j, l]
    M = torch.einsum("tejl,tsec,jlcab->tsjeab", Ur, w.to(A.dtype), A)                            # O_j[R_e, w_e]
    N = torch.einsum("pe,tsjeab->tspjab", sg.to(A.dtype), M)                                     # Glynn rows
    X = N[:, :, :, 0]
    for j in range(1, n):                                                                        # (x)_j, j = 0 left
        D0 = X.shape[-1]
        X = torch.einsum("tspab,tspcd->tspacbd", X, N[:, :, :, j]).reshape(T, s, -1, D0 * d, D0 * d)
    Kw = torch.einsum("p,tspab->tsab", sgp.to(A.dtype), X)
    return Kw.reshape(T, s, -1).transpose(1, 2)


def range_kraus(U: torch.Tensor, A: torch.Tensor, in_modes: torch.Tensor, rows: torch.Tensor, tfact: torch.Tensor,
                F_t: torch.Tensor, rtol: float, tol: float = 1e-14, seed: int = 0, chunk_bytes: int = 1 << 30):
    """The compressed Kraus set of every outcome of `rows` without the fine set (route R, comment above).
    U, A, in_modes, rows, tfact as in `pairing_apply`; F_t [T] the fine-pattern count of each outcome (caps the rank).
    Returns (K [Q, D, D] final x initial, outcome [Q] position in rows, cert [T] Frobenius estimate of the error of
    J_t / D (an a-posteriori estimate, comment above), failed [T] bool: cert above tol after the matrix-unit retry).
    Deterministic for a given seed and chunk_bytes (the chunking orders the random draws: the channel and the operator
    counts do not depend on it, the individual operators do, by a unitary mixing per outcome).
    """
    n, d, Kb = A.shape[0], A.shape[3], A.shape[2]
    D, T, dev, dt = d ** n, rows.shape[0], A.device, A.dtype
    eps = torch.finfo(torch.float64).eps
    # range cut: tol / 8 (a direction dropped at relative size c moves J_t by ~c through the cross terms, not c^2),
    # between the rounding floor 8 eps and 1e3 eps; rtol acts at the eigenvalue step only
    cut = max(_RANGE_MIN * eps, min(_RANGE_FLOOR * eps, tol / 8))
    gen = torch.Generator().manual_seed(seed)
    cap = torch.clamp(F_t, max=D * D).to(dev)
    P = 2 ** (n - 1)
    # range: one chunk of outcomes at a time, the sample count doubled for the whole chunk until every outcome is done
    Qs, rs = [None] * T, torch.zeros(T, dtype=torch.long, device=dev)
    s_init = min(int(cap.max()) + _OVERSAMPLE, 8 + _OVERSAMPLE)
    s_top = int(cap.max()) + _OVERSAMPLE
    Tc = max(1, int(chunk_bytes // max(1, s_top * (P + 2) * D * D * 16)))
    for t0 in range(0, T, Tc):
        t1 = min(T, t0 + Tc)
        S = range_samples(U, A, in_modes, rows[t0:t1], _crandn(gen, t1 - t0, s_init, n, Kb).to(dev))
        while True:
            Uq, sv, _ = torch.linalg.svd(S, full_matrices=False)                                 # [Tc, D^2, s']
            r = (sv > cut * sv[:, :1]).sum(1)
            s = S.shape[2]
            done = (r + _OVERSAMPLE <= s) | (r >= cap[t0:t1])
            if bool(done.all()):
                break
            add = min(s, s_top - s)
            S = torch.cat([S, range_samples(U, A, in_modes, rows[t0:t1],
                                            _crandn(gen, t1 - t0, add, n, Kb).to(dev))], 2)
        rs[t0:t1] = r
        for i, ri in enumerate(r.tolist()):                                                      # copies: r_t columns
            Qs[t0 + i] = Uq[i, :, :ri].clone()
    out_K, out_t = [], []
    cert = torch.zeros(T, dtype=torch.float64, device=dev)
    eye = torch.eye(D * D, dtype=dt, device=dev)
    for full in (False, True):
        todo = torch.arange(T, device=dev) if not full else torch.nonzero(cert > tol).reshape(-1)
        if todo.numel() == 0:
            break
        rr = rs[todo] if not full else torch.full_like(todo, D * D)
        for r in torch.unique(rr).tolist():                                                      # one group per rank
            if r == 0:                                                                           # E_t = 0: no operator
                continue
            grp = todo[rr == r]
            a = -(-_PROBE_OVERSAMPLE * r // D)
            Y = torch.eye(D, dtype=dt, device=dev) if a >= D else _crandn(gen, D, a).to(dev, dt)
            a = Y.shape[1]
            rho_in = torch.einsum("xi,yj->ijxy", Y, Y.conj()).reshape(a * a, D, D)
            per = 16 * (3 * D * D * r + 4 * (a * D) ** 2 + 2 * a * D * r + 2 * r * r)              # bytes per outcome
            Gc = max(1, int(chunk_bytes // per))
            for g0 in range(0, grp.numel(), Gc):                                                 # chunks of the group
                sel = grp[g0:g0 + Gc]
                Tg = sel.numel()
                Q = eye.expand(Tg, D * D, D * D) if full else torch.stack([Qs[int(i)] for i in sel.tolist()])
                E, c = pairing_apply(U, A, in_modes, rows[sel], tfact[sel], rho_in, chunk_bytes=chunk_bytes)
                Big = E.reshape(a, a, Tg, D, D).permute(2, 0, 3, 1, 4).reshape(Tg, a * D, a * D)
                del E
                Am = torch.einsum("tuxq,xi->tiuq", Q.reshape(Tg, D, D, r), Y).reshape(Tg, a * D, r)
                Qa, Ra = torch.linalg.qr(Am)
                Mid = Qa.conj().transpose(1, 2) @ Big @ Qa
                res = torch.linalg.matrix_norm(Big - Qa @ Mid @ Qa.conj().transpose(1, 2))
                del Big, Am, Qa
                smin = torch.linalg.svdvals(Ra)[:, -1]
                X = torch.linalg.solve_triangular(Ra, Mid, upper=True)                           # Ra^-1 Mid
                C = torch.linalg.solve_triangular(Ra, X.conj().transpose(1, 2), upper=True).conj().transpose(1, 2)
                C = 0.5 * (C + C.conj().transpose(1, 2))
                lam, V = torch.linalg.eigh(C)
                err = (c.pow(2).sum(0).sqrt() + res) / (smin * smin)                             # |Delta C|_F = |Delta J|_F
                err = err + _CERT_C * eps * math.sqrt(r) * torch.linalg.matrix_norm(C)           # rounding of Q, Q V
                resolved = lam > (err / math.sqrt(r)).unsqueeze(1)           # above the core's error level
                keep = resolved & (lam > (rtol * rtol) * lam[:, -1:])        # and the compression rule
                ops = (Q @ (V * lam.clamp(min=0).sqrt().unsqueeze(1))).transpose(1, 2)           # [Tg, r, D^2]
                cert[sel] = (err + torch.where(resolved, 0.0, lam).norm(dim=1)) / D              # + the unresolved part
                keep = keep & (cert[sel] <= tol).unsqueeze(1)                # above tol: retried, or failed
                out_K.append(ops[keep].reshape(-1, D, D))
                out_t.append(sel.unsqueeze(1).expand_as(keep)[keep])
    if not out_K:
        return torch.zeros((0, D, D), dtype=dt, device=dev), torch.zeros(0, dtype=torch.long, device=dev), cert, cert > tol
    Kc, tc = torch.cat(out_K), torch.cat(out_t)
    o = torch.argsort(tc, stable=True)
    return Kc[o], tc[o], cert, cert > tol


# ---- the weak route (CC-spin): operator-valued Clifford-Clifford over fine events -------------------------------
#
# A pure register psi [D, X] (the kernel's spins, then a passive rest X) and n emitters whose photons are all detected
# (complete operators, no `lost`, R = 1: the trace weight M of a photon not yet detected is the identity). List the n
# photons in uniformly random order; for the prefix c = (e_1..e_j) of fine events e = (row, internal mode),
#     P(c) ∝ sum_{S subset [n], |S| = j} | K~_{S,c} psi |^2,   K~_{S,c} = sum_{bijections S -> c} (x)_{i in S} O_i[e_.],
#     O_i[(row, k)] = sum_l U[row, in_i(l)] A_i[l, k]   (Clifford & Clifford's marginal with operator entries),
# and K~ obeys the Laplace recursion K~_{S,(c,e)} = sum_{i in S} O_i[e] K~_{S-i,c} (operators on different spins
# commute). With v[S] = K~_{S,c} psi and y[S, i, l, k] = A_i[l, k] v[S - i] (on spin i), the candidate weights are
#     W(row, k) = sum_S | sum_{i in S, l} U[row, in_i(l)] y[S, i, l, k] |^2 = sum_S u_row^dag G_k[S] u_row,
#     G_k[S] = <y[S, ., ., k] | y[S, ., ., k]> (a jL x jL Gram per internal mode),
# and sum_{row, k} W = (n - j + 1) sum_{|S'| = j-1} |v[S']|^2 exactly (U unitary, different emitters on different input
# modes, sum A^dag A = 1): checked at every step. Draw e* ~ W / sum W, set v[S] for e*. After n steps psi' = v[[n]] /
# |v[[n]]|: the post-state of the fine pattern (sqrt(tau!) K_tau psi), and the probability of the ordered chain summed
# over its orderings is |K_tau psi|^2 — an exact unravelling of the coarse channel (docs/SEQUENCE.md, the weak route).

def _subset_tables(n: int):
    """Per size j = 1..n: members [C_j, j] (sorted emitter ids) and parent [C_j, j] (index of S minus its t-th member
    in the size j - 1 table).
    """
    import itertools
    index = {(): 0}
    tables = []
    for j in range(1, n + 1):
        subs = list(itertools.combinations(range(n), j))
        mem = torch.tensor(subs, dtype=torch.long).reshape(len(subs), j)
        par = torch.tensor([[index[s[:t] + s[t + 1:]] for t in range(j)] for s in subs], dtype=torch.long)
        tables.append((mem, par.reshape(len(subs), j)))
        index = {s: i for i, s in enumerate(subs)}
    return tables


def ccspin_draw(U: torch.Tensor, ops: list[torch.Tensor], in_modes: list[torch.Tensor], dims: list[int],
                psi: torch.Tensor, uniforms: torch.Tensor | None = None, force: torch.Tensor | None = None,
                chunk_bytes: int = 1 << 30, spins: list[int] | None = None,
                M: list[torch.Tensor | None] | None = None):
    """One tick of the weak route for a batch of streams (comment above).
    U [m, m] or [Sn, m, m] (one unitary per stream; or an isometry [rows, m], e.g. the detection-loss dilation, whose
    rows are the candidates); ops: per present photon [L_i, K, d, d] on a shared K; in_modes: per photon [L_i]; dims:
    every spin dimension of the register (emitter 0 most significant); spins: the spin of each photon's emitter
    (default 0..n-1); M: per photon the trace weight of a photon not yet detected, M_i = sum_{l,k} A_i^dag A_i (None:
    the identity, a complete emitter); psi [Sn, D, X]; uniforms [Sn, n] float64 in [0, 1): the draw of step j is the
    inverse CDF of the float64 weights (on the host) at uniforms[:, j] — so the draws do not depend on chunking or
    device; force [Sn, n] candidate indices row * K + k instead (exact enumeration). With M, every norm of the chain
    is <v| (x)_{i not yet detected} M_i |v>.
    Returns (events [Sn, n] candidate indices in draw order, psi' [Sn, D, X] normalised, prob [Sn] the product of the
    step probabilities, dev [n] the largest relative deviation of the per-step identity).
    """
    n = len(ops)
    Sn, D, X = psi.shape
    dev_, dt = psi.device, psi.dtype
    m = U.shape[-2]                                                                                     # candidate rows
    K = ops[0].shape[1]
    L = max(a.shape[0] for a in ops)
    A = [torch.cat([a, a.new_zeros((L - a.shape[0], *a.shape[1:]))]).to(dev_, dt) for a in ops]       # [L, K, d, d]
    inm = torch.stack([torch.cat([im, im.new_zeros(L - im.shape[0])]) for im in in_modes]).to(dev_)    # [n, L]
    Us = U.to(dev_, dt)
    Us = Us.unsqueeze(0) if Us.dim() == 2 else Us                                                       # [1|Sn, rows, m]
    sp = list(range(n)) if spins is None else list(spins)
    left = [int(math.prod(dims[:s])) for s in sp]
    right = [int(math.prod(dims[s + 1:])) for s in sp]
    dloc = [dims[s] for s in sp]
    Ms = None if M is None or all(x is None for x in M) else [None if x is None else x.to(dev_, dt) for x in M]
    tables = _subset_tables(n)

    def weigh(t, mem_rows):
        """(x)_{i not in S} M_i applied to t [Sn, c, ..., D, X] (subset rows mem_rows [c, j]; identity without M)."""
        if Ms is None:
            return t
        sh = t.shape
        out = t
        for i in range(n):
            if Ms[i] is None:
                continue
            absent = ~(mem_rows == i).any(1) if mem_rows.numel() else torch.ones(sh[1], dtype=torch.bool, device=dev_)
            rows_i = torch.nonzero(absent).reshape(-1)
            if rows_i.numel() == 0:
                continue
            x = out[:, rows_i].reshape(Sn, rows_i.numel(), -1, left[i], dloc[i], right[i], X)
            x = torch.einsum("ab,scqxbyz->scqxayz", Ms[i], x).reshape(Sn, rows_i.numel(), *sh[2:])
            out = out.clone() if out is t else out
            out[:, rows_i] = x
        return out

    v = psi.reshape(Sn, 1, D, X)                                                                        # v[{}] = psi
    mem_prev = torch.zeros((1, 0), dtype=torch.long, device=dev_)
    events = torch.empty((Sn, n), dtype=torch.long)
    prob = torch.ones(Sn, dtype=torch.float64)
    dev = torch.zeros(n, dtype=torch.float64)
    ar = torch.arange(Sn, device=dev_)
    for j in range(1, n + 1):
        mem, par = (t.to(dev_) for t in tables[j - 1])
        C = mem.shape[0]
        T_prev = (v.conj() * weigh(v, mem_prev)).real.sum((1, 2, 3)).to(torch.float64)               # [Sn]
        per_sub = Sn * j * L * K * D * X * 16 * (3 if Ms is None else 4)
        Cc = max(1, int(chunk_bytes // max(1, per_sub)))

        def y_chunk(c0, c1, mem=mem, par=par, v=v, j=j):
            """The vectors y [Sn, c, j, L, K, D, X] of the subsets c0..c1 (A_i on spin i of each parent)."""
            y = torch.zeros((Sn, c1 - c0, j, L, K, D, X), dtype=dt, device=dev_)
            mc, pc = mem[c0:c1], par[c0:c1]
            for i in range(n):
                s_idx, t_idx = torch.nonzero(mc == i, as_tuple=True)
                if s_idx.numel() == 0:
                    continue
                x = v[:, pc[s_idx, t_idx]].reshape(Sn, -1, left[i], dloc[i], right[i], X)
                yi = torch.einsum("lkab,spxbyz->splkxayz", A[i], x).reshape(Sn, -1, L, K, D, X)
                y[:, s_idx, t_idx] = yi
            return y

        def u_chunk(c0, c1, mem=mem):
            """The coefficients u [1|Sn, rows, c, j, L] = U[row, in_{member}(l)] of the subsets c0..c1."""
            cols = inm[mem[c0:c1]]                                                                       # [c, j, L]
            return Us[:, :, cols]

        W = torch.zeros((Sn, m, K), dtype=torch.float64, device=dev_)
        cache = []
        for c0 in range(0, C, Cc):
            c1 = min(C, c0 + Cc)
            y = y_chunk(c0, c1)
            yf = y.reshape(Sn, c1 - c0, j * L, K, D * X)
            zf = weigh(y, mem[c0:c1]).reshape(Sn, c1 - c0, j * L, K, D * X) if Ms is not None else yf
            G = torch.einsum("scakv,scbkv->sckab", yf.conj(), zf)                                      # Gram per k
            u = u_chunk(c0, c1).reshape(Us.shape[0], m, c1 - c0, j * L)
            W += torch.einsum("srca,sckab,srcb->srk", u.conj(), G, u).real
            if len(cache) * per_sub * Cc < chunk_bytes:
                cache.append((c0, c1, y))
            else:
                cache.append((c0, c1, None))
        tot = W.sum((1, 2))
        expect = (n - j + 1) * T_prev.to(W.device)
        dev[j - 1] = float(((tot - expect).abs() / expect.clamp_min(1e-300)).max())
        p = (W.reshape(Sn, m * K) / tot.clamp_min(1e-300).unsqueeze(1)).cpu()
        if force is not None:
            e = force[:, j - 1].cpu()
        else:                                         # inverse CDF at the stream's uniform
            cdf = p.clamp_min(0).cumsum(1)
            e = torch.searchsorted(cdf, (uniforms[:, j - 1] * cdf[:, -1]).unsqueeze(1).contiguous(), right=True)[:, 0]
            e = e.clamp(max=m * K - 1)
        events[:, j - 1] = e
        prob *= p[torch.arange(Sn), e]
        row, kk = (e // K).to(dev_), (e % K).to(dev_)
        vn = torch.empty((Sn, C, D, X), dtype=dt, device=dev_)
        for c0, c1, y in cache:
            y = y_chunk(c0, c1) if y is None else y
            u = u_chunk(c0, c1).reshape(Us.shape[0], m, c1 - c0, j, L)
            us = u[ar if Us.shape[0] > 1 else torch.zeros_like(ar), row]                               # [Sn, c, j, L]
            ys = y[ar, :, :, :, kk]                                                                      # [Sn, c, j, L, D, X]
            vn[:, c0:c1] = torch.einsum("sctl,sctldx->scdx", us, ys)
        v, mem_prev = vn, mem
    out = v[:, 0]
    nrm = out.reshape(Sn, -1).norm(dim=1).clamp_min(1e-300).view(-1, 1, 1)
    return events, out / nrm, prob, dev
