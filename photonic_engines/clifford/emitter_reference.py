"""Exact sampler for EmitterMPS inputs — reference implementations of the generalized
Clifford chain rule (DESIGN.md §2.2), small n only, CPU torch. Three layers:

1. `emitter_sample_reference`: the algorithm in its most transparent form. Photon order sigma
   uniform; at step k the weight of candidate output mode rho is the LO probability of the
   ordered row sequence (r_1..r_{k-1}, rho) for the REDUCED state rho_T of the photons in
   T = sigma(1..k):  w(rho) = v^dag rho_T v,  v_x = per(U[rows, modes(x_T)]).  rho_T is built
   explicitly (L^k x L^k) from the full coefficient tensor; permanents by Laplace along the
   candidate row so all m candidates share the (k-1)-minors. Exponential in n everywhere -
   it is the oracle for the kernels, and the numerical check of the marginal lemma for
   entangled inputs.

2. `gap_kraus_terms`: the pure-state decomposition rho_T = sum_e |psi_e><psi_e| the fast
   algorithm uses. Each maximal run of not-yet-placed photons between kept photons is a
   channel on the emitter with Kraus operators obtained from the SVD of its operator span;
   the product of the ranks is the number R of pure MPS terms (prefix / suffix runs act on
   |s0> and <f| and contribute at most chi each).

3. `matrix_cofactor_weights`: the per-step computation the GPU kernel will implement —
   Glynn / Gray-code sum over delta in {+-1}^{k-1} with chi x chi matrix chains, giving the
   matrix cofactors K_c = sum_delta sign(delta) p_c(delta) q_c(delta)^T and then, for every
   candidate rho at once, amp_e(rho) = sum_c sum_l U[rho, mode(c, l)] Tr(M~_{c,l} K_{c,e}),
   w(rho) = sum_e |amp_e(rho)|^2.  Checked against (1) in the tests.

Row-sequence convention (Clifford): the k-th mode is drawn proportionally to
|amplitude(r_1..r_k)|^2 of the ORDERED sequence; the returned occupation is the multiset.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch

from ..states import EmitterMPS

C128 = torch.complex128


# ---- coefficient tensors -------------------------------------------------------------------

def coefficient_tensor(state: EmitterMPS) -> torch.Tensor:
    """c[x_1, ..., x_n, s_f] = <s_f| M^(n)_{x_n} ... M^(1)_{x_1} |s0>, shape [L]*n + [chi]."""
    n, L, chi = state.n, state.num_labels, state.chi
    ops = torch.as_tensor(state.ops, dtype=C128)                  # [n, L, chi, chi]
    v = torch.as_tensor(state.s0, dtype=C128).reshape(1, chi)     # [1, chi] (batch over x)
    for k in range(n):
        # v: [L^k, chi] -> [L^k, L, chi]
        v = torch.einsum("lts,bs->blt", ops[k], v).reshape(-1, chi)
    return v.reshape(*([L] * n), chi)


def final_spin_distribution(state: EmitterMPS) -> torch.Tensor:
    """p(s_f) = sum_x |c[x, s_f]|^2 for a traced emitter, as the diagonal of the emitter's reduced
    state after the n emissions, rho_n = E_n ... E_1(|s0><s0|) with E_k(rho) = sum_l M^(k)_l rho
    M^(k)_l^dag — O(n L chi^3), not the 2^n coefficient tensor (which the samplers called once per
    batch and, with loss, once per surviving-count group: 34 GB of host memory at n = 30).
    """
    ops = torch.as_tensor(state.ops, dtype=C128)                  # [n, L, chi, chi]
    s0 = torch.as_tensor(state.s0, dtype=C128).reshape(-1)
    rho = torch.outer(s0, s0.conj())
    for k in range(state.n):
        rho = torch.einsum("lts,su,lvu->tv", ops[k], rho, ops[k].conj())
    return rho.diagonal().real.clamp_min(0)


def pure_coefficients(state: EmitterMPS, final: torch.Tensor | None = None) -> torch.Tensor:
    """c_x = <f| ... |s0> as a tensor [L]*n for the given final vector (default state.final)."""
    c = coefficient_tensor(state)
    f = torch.as_tensor(state.final if final is None else final, dtype=C128)
    return torch.einsum("...s,s->...", c, f.conj())


# ---- 1. transparent reference sampler -------------------------------------------------------

def _permanent(mat: torch.Tensor) -> torch.Tensor:
    """Ryser, complex, small matrices."""
    n = mat.shape[0]
    if n == 0:
        return torch.tensor(1.0 + 0j, dtype=mat.dtype)
    total = torch.zeros((), dtype=mat.dtype)
    for k in range(1, n + 1):
        for cols in itertools.combinations(range(n), k):
            total = total + (-1.0) ** (n - k) * mat[:, list(cols)].sum(dim=1).prod()
    return total


def reduced_density(c: torch.Tensor, T: Sequence[int]) -> torch.Tensor:
    """rho_T over the labels of slots T (row/col index = mixed-radix x_T in the order of T)."""
    n = c.dim()
    rest = [j for j in range(n) if j not in T]
    L = c.shape[0]
    mat = c.permute(*T, *rest).reshape(L ** len(T), -1)
    return mat @ mat.conj().T


def candidate_weights_reference(U: torch.Tensor, state: EmitterMPS, c: torch.Tensor,
                                T: Sequence[int], rows: Sequence[int]) -> torch.Tensor:
    """w(rho) for all m candidates: rows = the k-1 already-drawn output modes, T the k slots
    (T[-1] is the slot whose photon is being placed - any slot of T may carry rho, since the
    permanent is over all of T; T's order only fixes the index convention).
    """
    L = state.num_labels
    k = len(T)
    rho_T = reduced_density(c, T)                                     # [L^k, L^k]
    # cofactors: for every x_T and every column c, per(U[rows, modes(x_T) \ c])
    labels = list(itertools.product(range(L), repeat=k))
    cof = torch.zeros((len(labels), k), dtype=C128)
    colmode = torch.zeros((len(labels), k), dtype=torch.long)
    for xi, x in enumerate(labels):
        cols = [state.modes[T[j]][x[j]] for j in range(k)]
        colmode[xi] = torch.tensor(cols)
        for cc in range(k):
            sub = U[list(rows)][:, [cols[j] for j in range(k) if j != cc]]
            cof[xi, cc] = _permanent(sub)
    # V[rho, x] = sum_c U[rho, colmode[x, c]] cof[x, c]
    V = torch.einsum("rxc,xc->rx", U[:, colmode], cof)                # [m, L^k]
    w = torch.einsum("rx,xy,ry->r", V, rho_T, V.conj()).real
    return w.clamp_min(0.0)


def emitter_sample_reference(state: EmitterMPS, U: torch.Tensor, shots: int,
                             generator: torch.Generator | None = None,
                             order: str = "uniform", method: str = "density") -> torch.Tensor:
    """Exact samples [shots, m] (int64 occupations). order="uniform" is the algorithm;
    order="fixed" (sigma = identity) is WRONG by construction and exists to demonstrate that
    the random photon order is required (see tests). method="density" uses the explicit
    reduced density matrices (section 1), method="matrix" the gap-Kraus + matrix-cofactor
    computation of section 3 (the kernel's algorithm).
    """
    state.require_uniform_labels("emitter_sample_reference")
    U = torch.as_tensor(U, dtype=C128)
    m = U.shape[0]
    n = state.n
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    ctens = coefficient_tensor(state)
    if state.final is not None:
        f = torch.as_tensor(state.final, dtype=C128)
        c_pure = torch.einsum("...s,s->...", ctens, f.conj())
        pf = None
    else:
        pf = (ctens.abs() ** 2).reshape(-1, state.chi).sum(0).real
        pf = pf / pf.sum()
    out = torch.zeros((shots, m), dtype=torch.long)
    for s in range(shots):
        if pf is not None:
            sf = int(torch.multinomial(pf, 1, generator=generator))
            c = ctens[..., sf]
        else:
            c = c_pure
        sigma = list(range(n)) if order == "fixed" else torch.randperm(n, generator=generator).tolist()
        rows: list[int] = []
        fvec = None if pf is None else torch.eye(state.chi, dtype=C128)[sf]
        for k in range(1, n + 1):
            T = sigma[:k]
            if method == "matrix":
                w, _ = matrix_cofactor_weights(U, state, T, rows, final=fvec)
            else:
                w = candidate_weights_reference(U, state, c, T, rows)
            rho = int(torch.multinomial(w / w.sum(), 1, generator=generator))
            rows.append(rho)
        out[s] = torch.bincount(torch.tensor(rows), minlength=m)
    return out


# ---- 2. gap-channel Kraus decomposition ---------------------------------------------------

def _orthonormal_kraus(A: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    """A: [N, chi, chi] operators of a channel E(rho) = sum_x A_x rho A_x^dag. Returns K [R, chi, chi]
    with E(rho) = sum_i K_i rho K_i^dag and R = rank of span{A_x} (SVD of the [N, chi^2] matrix:
    A_x = sum_i W_xi K_i with W^dag W = I, so the cross terms vanish).
    """
    N, chi, _ = A.shape
    Amat = A.reshape(N, chi * chi)
    _, S, Vh = torch.linalg.svd(Amat, full_matrices=False)
    keep = tol * max(1.0, float(S[0])) < S if S.numel() else tol < S
    K = (S[keep].unsqueeze(1) * Vh[keep]).reshape(-1, chi, chi)
    return K


def _reduce_vectors(V: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    """Rows of V [N, d] -> orthogonal rows W [r, d] with W^dag W = V^dag V (SVD), r = rank."""
    _, S, Vh = torch.linalg.svd(V, full_matrices=False)
    keep = tol * max(1.0, float(S[0])) < S if S.numel() else tol < S
    return S[keep].unsqueeze(1) * Vh[keep]


def gap_kraus_terms(state: EmitterMPS, T: Sequence[int], final: torch.Tensor | None = None,
                    labels: Sequence[int] | None = None, U: torch.Tensor | None = None):
    """Pure-state decomposition of the reduced state on the slots T (chain order):
    returns (kept_slots, terms) where kept_slots is sorted(T) and terms is a list of
    (s0_e [chi], mats_e [k, L, chi, chi], f_e [chi]) such that
        rho_T = sum_e |psi_e><psi_e|,  psi_e(x) = <f_e| M~_{k, x_k} ... M~_{1, x_1} |s0_e>,
    with the gap Kraus operators folded into the kept slots' operators (a gap after kept slot j
    multiplies M~_{j, l} on the left by K^{(g)}_{e_g}). Gap channels are composed slot by slot
    with an SVD reduction after each slot (rank <= chi^2 for inner gaps, <= chi for the prefix
    and suffix runs, which act on |s0> and <f|), so the cost is linear in the gap length.
    R = len(terms) <= prod of gap ranks; terms of zero norm are dropped (GHZ: 2 survive
    whatever the gaps). labels [n] (optional, `emitter_terms.sample_lost_labels`): the sampled
    label of every traced slot; that slot then contributes M_{j, x} only (every gap rank 1).
    A label L + r is a singleton (distinguishable) slot at its drawn output row r: the slot
    contributes sum_l U[r, modes[j][l]] M_{j, l} (`U` required, docs/DISTINGUISHABILITY.md).
    """
    n, L, chi = state.n, state.num_labels, state.chi
    ops = torch.as_tensor(state.ops, dtype=C128)
    if labels is not None:                                                # keep only the chosen label at traced slots
        ops = ops.clone()
        for j, x in enumerate(labels):
            if x is None or int(x) < 0:
                continue
            if int(x) >= L:                                                # row fold: one operator, stored at label 0
                Uj = torch.as_tensor(U).to("cpu", C128)[int(x) - L, list(state.modes[j])]        # [L]
                Uj = Uj / Uj.norm().clamp_min(1e-300)                                     # unit norm, as `label_weights`
                G = torch.einsum("l,lts->ts", Uj, ops[j])
                ops[j] = 0; ops[j, 0] = G
            else:
                m_ = torch.zeros(L, dtype=torch.bool); m_[int(x)] = True
                ops[j][~m_] = 0
    s0 = torch.as_tensor(state.s0, dtype=C128)
    f = torch.as_tensor(state.final if final is None else final, dtype=C128)
    kept = sorted(T)
    k = len(kept)
    # prefix: vectors {A_x s0}, reduced after every slot
    V = s0.unsqueeze(0)                                                   # [r, chi]
    for j in range(kept[0]):
        V = _reduce_vectors(torch.einsum("lts,rs->rlt", ops[j], V).reshape(-1, chi))
    s0_terms = V                                                          # [r0, chi]
    # suffix: row vectors {<f| A_x} -> stored as f_e with <f_e| = row
    W = f.conj().unsqueeze(0)                                             # [r, chi] rows
    for j in range(n - 1, kept[-1], -1):
        W = _reduce_vectors(torch.einsum("rs,lst->rlt", W, ops[j]).reshape(-1, chi))
    f_terms = W.conj()                                                    # [rf, chi]
    # inner gaps: Kraus sets composed slot by slot, reduced after each
    inner: list[torch.Tensor | None] = []
    for a, b in itertools.pairwise(kept):
        if b == a + 1:
            inner.append(None)
            continue
        K = torch.eye(chi, dtype=C128).unsqueeze(0)                       # [r, chi, chi]
        for j in range(a + 1, b):
            KK = torch.einsum("lts,rsu->rltu", ops[j], K).reshape(-1, chi, chi)
            K = _orthonormal_kraus(KK)
        inner.append(K)
    # all combinations -> stacked term tensors
    r0, rf = s0_terms.shape[0], f_terms.shape[0]
    ranks = [1 if K is None else K.shape[0] for K in inner]
    R = r0 * rf * math.prod(ranks)
    base = ops[kept]                                                      # [k, L, chi, chi]
    mats = base.unsqueeze(0).repeat(R, 1, 1, 1, 1)
    idx = torch.arange(R)
    # mixed-radix decode of the term index: (i0, if, g_1, ..., g_{k-1})
    rem = idx.clone()
    g_choice = []
    for rk in reversed(ranks):
        g_choice.append(rem % rk); rem = rem // rk
    g_choice = g_choice[::-1]
    i_f = rem % rf; i_0 = rem // rf
    for gi, K in enumerate(inner):
        if K is not None:
            Ksel = K[g_choice[gi]]                                        # [R, chi, chi]
            mats[:, gi] = torch.einsum("rts,rlsu->rltu", Ksel, mats[:, gi])
    s0_all = s0_terms[i_0]
    f_all = f_terms[i_f]
    # norms via the chi^2 transfer chain, batched; drop zero-norm terms
    v = torch.einsum("rs,rt->rst", s0_all, s0_all.conj()).reshape(R, chi * chi)
    for j in range(k):
        E = torch.einsum("rlts,rluv->rtusv", mats[:, j], mats[:, j].conj()).reshape(R, chi * chi, chi * chi)
        v = torch.einsum("rab,rb->ra", E, v)
    fv = torch.einsum("rs,rt->rst", f_all, f_all.conj()).reshape(R, chi * chi)
    norm2 = torch.einsum("ra,ra->r", fv.conj(), v).real
    scale = (f_all.abs() ** 2).sum(1) * (s0_all.abs() ** 2).sum(1)
    keep = norm2 > 1e-14 * scale.clamp_min(1.0)
    return kept, [(s0_all[i], mats[i], f_all[i]) for i in keep.nonzero(as_tuple=True)[0].tolist()]


# ---- 3. matrix-cofactor step (what the kernel computes) -----------------------------------

def matrix_cofactor_weights(U: torch.Tensor, state: EmitterMPS, T: Sequence[int],
                            rows: Sequence[int], final: torch.Tensor | None = None
                            ) -> tuple[torch.Tensor, int]:
    """w(rho) for all candidates rho via Glynn sums with chi x chi matrix chains over the pure
    terms of the reduced state. Returns (w [m], R).
    """
    U = torch.as_tensor(U, dtype=C128)
    m = U.shape[0]
    kept, terms = gap_kraus_terms(state, T, final)
    k = len(kept)
    chi = state.chi
    modes = torch.tensor([list(state.modes[j]) for j in kept])                 # [k, L]
    Urows = U[list(rows)]                                                       # [k-1, m]
    # column sums s_{j,l}(delta) for all delta in {+1} x {+-1}^{k-2}
    kk = len(rows)                                                              # = k-1
    if kk == 0:
        deltas = torch.ones((1, 0), dtype=C128)
    else:
        grid = torch.tensor(list(itertools.product([1.0, -1.0], repeat=kk - 1))) if kk > 1 else torch.ones((1, 0))
        deltas = torch.cat([torch.ones((grid.shape[0], 1)), grid], dim=1).to(C128)   # [D, k-1], delta_1 = +1
    D = deltas.shape[0]
    colsum = torch.einsum("di,ijl->djl", deltas, Urows[:, modes])               # [D, k, L]
    sign = deltas.real.prod(dim=1)                                              # [D]
    norm = 2.0 ** (-(kk - 1)) if kk >= 1 else 1.0
    w = torch.zeros(m, dtype=torch.float64)
    for s0_e, mats, f_e in terms:
        # G_j(delta) = sum_l s_{j,l}(delta) M~_{j,l} : [D, k, chi, chi]
        G = torch.einsum("djl,jlts->djts", colsum, mats)
        # forward p_c = (prod_{j<c} G_j) s0 ; backward q_c^T = f^dag prod_{j>c} G_j
        p = [None] * k
        q = [None] * k
        v = s0_e.unsqueeze(0).expand(D, chi)
        for c_ in range(k):
            p[c_] = v
            v = torch.einsum("dts,ds->dt", G[:, c_], v)
        u = f_e.conj().unsqueeze(0).expand(D, chi)                              # row vector f^dag
        for c_ in range(k - 1, -1, -1):
            q[c_] = u
            u = torch.einsum("ds,dst->dt", u, G[:, c_])
        # K_c = sum_delta sign * p_c q_c^T  (chi x chi), then amp(rho) = sum_c sum_l U[rho, mode(c,l)] Tr(M~_{c,l} K_c)
        amp = torch.zeros(m, dtype=C128)
        for c_ in range(k):
            Kc = torch.einsum("d,dt,ds->ts", sign.to(C128), p[c_], q[c_]) * norm
            tr = torch.einsum("lts,st->l", mats[c_], Kc)                        # Tr(M~_{c,l} K_c) per label
            amp = amp + U[:, modes[c_]] @ tr
        w += (amp.abs() ** 2)
    return w, len(terms)
