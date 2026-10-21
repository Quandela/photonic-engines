"""Internal photon modes, the weak route with branches (docs/SEQUENCE.md §14):
`sample_apply` with `lost` operators, R > 1 unobserved outcomes (the branch word drawn first, the chain weighted by
M_{j,r}) and a non-uniform detection efficiency (the dilated U). T2: exact enumeration of every (branch word, chain)
against the coarse channel (routes F and P', and F with explicit loss modes for eta_out); T3: fixed-seed statistics."""
import itertools

import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_route_f import _mode_emitter
from test_internal_modes_rwords import _emitter_r

from photonic_engines.sequence import SequenceKernel

DEV_TOL = 1e-13


def _psi(D, X, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(D * X, generator=g, dtype=torch.complex128)
    return (z / z.norm()).reshape(D, X)


def _enumerate_branches(mk, U, psi, eta_out=None):
    """Every (branch word, ordered chain of the present photons) as a forced stream: sum of P psi' psi'^dag per index,
    total probability."""
    K = mk._kb(U, 0)
    Q = [0 if N.shape[0] == 0 else N.shape[0] for N in mk._lost(0)]
    rows = mk.m * (2 if eta_out is not None else 1)
    fb, fe = [], []
    for w in itertools.product(*[range(mk.R[j] + Q[j]) for j in range(mk.k)]):
        n1 = sum(w[j] < mk.R[j] for j in range(mk.k))
        for chain in itertools.product(range(rows * K), repeat=n1):
            fb.append(list(w))
            fe.append(list(chain) + [0] * (mk.k - n1))
    fb, fe = torch.tensor(fb, dtype=torch.long), torch.tensor(fe, dtype=torch.long)
    S = fb.shape[0]
    p = psi.unsqueeze(0).expand(S, *psi.shape).clone()
    if not (mk.lossy or mk.multi_r):
        fb = None
    idx, out, info = mk.sample_apply(U, 0, p, eta_out=eta_out, _force=fe if fb is None else (fb, fe))
    assert float(info["deviation"].max()) <= DEV_TOL
    v = out.reshape(S, -1)
    rho = torch.einsum("s,sa,sb->sab", info["prob"].to(v.dtype), v, v.conj())
    n_out = mk.patterns_all.shape[0] if eta_out is not None else mk.S
    acc = torch.zeros((n_out, v.shape[1], v.shape[1]), dtype=v.dtype).index_add_(0, idx, rho)
    return acc, float(info["prob"].sum())


@pytest.mark.parametrize("k,K,R,Q,X,indist", [(2, 2, 1, 1, 1, None), (2, 2, 2, 0, 2, None), (2, 2, 2, 1, 1, None),
                                              (2, 2, 2, 1, 1, [0.8, 0.5]), (3, 2, 2, 1, 1, None)])
def test_branch_enumeration_is_the_channel(k, K, R, Q, X, indist):
    """T2 with branches: summed over every (branch word, chain), P psi' psi'^dag per detected pattern (0..k photons)
    equals E_t(psi psi^dag) (rest X) by route F and by the pairing route, < 1e-14; total probability 1."""
    ems = [_emitter_r(1, 2, K, R, Q, 2, (2 * j, 2 * j + 1), seed=10 * k + R + Q + j) for j in range(k)]
    if R == 1:
        from photonic_engines.sequence import ModeBinEmitter
        ems = [ModeBinEmitter(e.ops[:, :, :, 0], e.modes, e.s0, lost=e.lost) for e in ems]
    mk = SequenceKernel(ems, 2 * k, indist=indist)
    U = haar_unitary(2 * k, 7 + k)
    psi = _psi(mk.chi, X, 3 + k + R)
    acc, total = _enumerate_branches(mk, U, psi)
    assert abs(total - 1.0) < 1e-13
    rho = psi.reshape(-1, 1) @ psi.reshape(1, -1).conj()
    for t in range(mk.S):
        for route in ("F", "P"):
            ref = mk.condition(U, 0, rho, [t], route=route, rest=X)
            assert float((acc[t] - ref).abs().max()) < 1e-14, (t, route)


def test_detection_efficiency_is_the_loss_mode_channel():
    """eta_out (per output mode, non-uniform): the enumeration over the dilated U equals route F on the problem with
    explicit loss modes (U completed to a unitary on 2m modes, the loss-mode occupations summed out), < 1e-14."""
    k, K, m = 2, 2, 4
    ems = [_mode_emitter(1, 2, K, 2, (2 * j, 2 * j + 1), seed=30 + j) for j in range(k)]
    mk = SequenceKernel(ems, m)
    U = torch.as_tensor(haar_unitary(m, 31))
    eta = torch.tensor([0.9, 0.6, 0.75, 0.3], dtype=torch.float64)
    psi = _psi(mk.chi, 1, 32)
    acc, total = _enumerate_branches(mk, U, psi, eta_out=eta)
    assert abs(total - 1.0) < 1e-13
    top = torch.cat([eta.sqrt().view(-1, 1) * U, (1 - eta).sqrt().view(-1, 1) * U])        # [2m, m], an isometry
    full = torch.linalg.qr(torch.cat([top, torch.randn(2 * m, m, dtype=top.dtype,
                                                       generator=torch.Generator().manual_seed(1))], 1))[0]
    full = torch.cat([top, full[:, m:]], 1)                                                  # unitary [2m, 2m]
    assert float((full.conj().T @ full - torch.eye(2 * m, dtype=full.dtype)).abs().max()) < 1e-13
    big = SequenceKernel(ems, 2 * m)
    rho = psi.reshape(-1, 1) @ psi.reshape(1, -1).conj()
    ref = torch.zeros_like(acc)
    pats_all = mk.patterns_all
    for t, pat in enumerate(np.asarray(big.patterns)):
        det = pat[:m]
        i = int(np.nonzero((pats_all == det).all(1))[0][0])
        ref[i] += big.condition(full, 0, rho, [t], route="F")
    assert float((acc - ref).abs().max()) < 1e-14


def test_branch_statistics_and_seeds():
    """T3 with branches (fixed seed): k = 3, K = 2, R = 2, Q = 1, 2 10^4 streams, the histogram of the 0..3-photon
    patterns against outcome_law (chi^2 within 4 sigma of its dof); the same seed gives the same draws."""
    k = 3
    ems = [_emitter_r(1, 2, 2, 2, 1, 2, (2 * j, 2 * j + 1), seed=40 + j) for j in range(k)]
    mk = SequenceKernel(ems, 2 * k)
    U = haar_unitary(2 * k, 41)
    psi = _psi(mk.chi, 1, 42)
    N = 20000
    p0 = psi.unsqueeze(0).expand(N, -1, -1).clone()
    idx, _, info = mk.sample_apply(U, 0, p0, generator=torch.Generator().manual_seed(43))
    again = mk.sample_apply(U, 0, p0, generator=torch.Generator().manual_seed(43))
    assert torch.equal(idx, again[0]) and torch.equal(info["branch"], again[2]["branch"])
    assert np.array_equal(mk.patterns[idx.numpy()], info["occ"].numpy())          # occ is the indexed pattern
    none = mk.sample_apply(U, 0, p0[:10], generator=torch.Generator().manual_seed(43), index=False)
    assert none[0] is None and torch.equal(none[2]["occ"], info["occ"][:10])
    assert float(info["deviation"].max()) <= DEV_TOL
    p = mk.outcome_law(U, 0, psi.reshape(-1, 1) @ psi.reshape(1, -1).conj())
    cnt = torch.bincount(idx, minlength=mk.S).to(torch.float64)
    keep = p > 5.0 / N
    chi2 = float(((cnt[keep] - N * p[keep]) ** 2 / (N * p[keep])).sum())
    dof = int(keep.sum()) - 1
    assert chi2 < dof + 4 * (2 * dof) ** 0.5, (chi2, dof)
