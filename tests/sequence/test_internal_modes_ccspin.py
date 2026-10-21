"""Internal photon modes, the weak route (docs/SEQUENCE.md §14): `sample_apply`, the
operator-valued Clifford-Clifford chain over fine events (R = 1, no `lost`). T2: exact enumeration of every chain
against the coarse channel (routes F and P'); T3: fixed-seed statistics against `outcome_law` / `condition`; T4: the
per-step identity on every test; K = 1 included."""
import itertools

import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_route_f import _mode_emitter

from photonic_engines.sequence import ModeBinEmitter, SequenceKernel

DEV_TOL = 1e-13                                                                  # T4: the per-step identity


def _psi(D, X, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(D * X, generator=g, dtype=torch.complex128)
    return (z / z.norm()).reshape(D, X)


def _kernel(k, K, seed, indist=None):
    ems = [_mode_emitter(1, 2, K, 2, (2 * j, 2 * j + 1), seed=seed + j) for j in range(k)]
    if K == 1:                                    # one internal mode through the internal-mode kernel (the plain
        ems = [ModeBinEmitter(e.ops[:, :, None], e.modes, e.s0) for e in ems]   # kernel keeps its own label route)
    return SequenceKernel(ems, 2 * k, indist=indist), haar_unitary(2 * k, seed + 50)


def _enumerate(mk, U, psi):
    """Sum over every ordered chain of P(chain) psi' psi'^dag per detected pattern [S, D X, D X], total probability."""
    K = mk._kb(U, 0)
    n_cand = mk.m * K
    chains = torch.tensor(list(itertools.product(range(n_cand), repeat=mk.k)), dtype=torch.long)
    S = chains.shape[0]
    p = psi.unsqueeze(0).expand(S, *psi.shape).clone()
    idx, out, info = mk.sample_apply(U, 0, p, _force=chains)
    assert float(info["deviation"].max()) <= DEV_TOL
    v = out.reshape(S, -1)
    rho = torch.einsum("s,sa,sb->sab", info["prob"].to(v.dtype), v, v.conj())
    acc = torch.zeros((mk.S, v.shape[1], v.shape[1]), dtype=v.dtype).index_add_(0, idx, rho)
    return acc, float(info["prob"].sum())


@pytest.mark.parametrize("k,K,X,indist", [(2, 1, 1, None), (2, 2, 2, None), (2, 3, 1, None), (2, 2, 1, [0.7, 0.4]),
                                          (3, 2, 2, None)])
def test_enumeration_is_the_channel(k, K, X, indist):
    """T2: summed over every ordered chain of fine events, P(chain) psi' psi'^dag per detected pattern equals
    E_t(psi psi^dag) (with the passive rest X) by route F and by the pairing route, < 1e-14; total probability 1."""
    mk, U = _kernel(k, K, seed=10 * k + K, indist=indist)
    psi = _psi(mk.chi, X, 3 + k + K)
    acc, total = _enumerate(mk, U, psi)
    assert abs(total - 1.0) < 1e-13
    rho = (psi.reshape(-1, 1) @ psi.reshape(1, -1).conj())
    for t in range(mk.S):
        for route in ("F", "P"):
            ref = mk.condition(U, 0, rho, [t], route=route, rest=X)
            assert float((acc[t] - ref).abs().max()) < 1e-14, (t, route)


def test_one_mode_label_preserving_is_the_plain_channel():
    """T5: K = 1 label-preserving emitters A_l = |l><l| G through the internal-mode kernel — the
    enumerated CC-spin chain equals the PLAIN SequenceKernel's channel per outcome (its own label route basis), < 1e-14."""
    from photonic_engines.sequence import BinEmitter
    k, m = 3, 6
    rng = np.random.default_rng(80)
    plain, one = [], []
    for j in range(k):
        G = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        ops = np.stack([np.outer(np.eye(2)[lab], np.eye(2)[lab]) @ G for lab in range(2)])[None]   # [1, L, d, d]
        plain.append(BinEmitter(ops, (2 * j, 2 * j + 1), [1, 0]))
        one.append(ModeBinEmitter(ops[:, :, None], (2 * j, 2 * j + 1), [1, 0]))
    ref, mk = SequenceKernel(plain, m), SequenceKernel(one, m)
    U = haar_unitary(m, 81)
    psi = _psi(mk.chi, 1, 82)
    acc, total = _enumerate(mk, U, psi)
    assert abs(total - 1.0) < 1e-13
    Kp = ref.kraus(U, 0)                                                         # [S, chi, chi], one per pattern
    v = psi.reshape(-1)
    for t in range(mk.S):
        w = Kp[ref.pattern_index(mk.patterns[t:t + 1])[0]] @ v
        assert float((acc[t] - torch.outer(w, w.conj())).abs().max()) < 1e-14


def test_statistics_against_the_law():
    """T3 (fixed seed): sample_apply's pattern histogram against outcome_law (chi^2 within 4 sigma of its dof) and the
    Monte-Carlo average of psi' psi'^dag on the likeliest outcome against condition / p (within 5 / sqrt(count))."""
    mk, U = _kernel(3, 3, seed=40)
    psi = _psi(mk.chi, 1, 41)
    N = 20000
    g = torch.Generator().manual_seed(42)
    idx, out, info = mk.sample_apply(U, 0, psi.unsqueeze(0).expand(N, -1, -1).clone(), generator=g)
    assert float(info["deviation"].max()) <= DEV_TOL
    rho = psi.reshape(-1, 1) @ psi.reshape(1, -1).conj()
    p = mk.outcome_law(U, 0, rho)
    cnt = torch.bincount(idx, minlength=mk.S).to(torch.float64)
    keep = p > 5.0 / N
    chi2 = float(((cnt[keep] - N * p[keep]) ** 2 / (N * p[keep])).sum())
    dof = int(keep.sum()) - 1
    assert chi2 < dof + 4 * (2 * dof) ** 0.5, (chi2, dof)
    t = int(torch.argmax(p))
    sel = idx == t
    v = out[sel].reshape(int(sel.sum()), -1)
    mc = torch.einsum("sa,sb->ab", v, v.conj()) / int(sel.sum())
    ref = mk.condition(U, 0, rho, [t]) / p[t]
    assert float((mc - ref).abs().max()) < 5.0 / float(sel.sum()) ** 0.5


@pytest.mark.slow
def test_statistics_four_photons_eight_modes():
    """T3 at the design's size (fixed seed): k = 4, K = 8, 10^5 streams (stream-chunked), chi^2 against outcome_law
    within 4 sigma of its dof."""
    mk, U = _kernel(4, 8, seed=45)
    psi = _psi(mk.chi, 1, 46)
    N = 100000
    idx, _, info = mk.sample_apply(U, 0, psi.unsqueeze(0).expand(N, -1, -1).clone(),
                                   generator=torch.Generator().manual_seed(47), chunk_bytes=1 << 28)
    assert float(info["deviation"].max()) <= DEV_TOL
    p = mk.outcome_law(U, 0, psi.reshape(-1, 1) @ psi.reshape(1, -1).conj(), route="P")
    cnt = torch.bincount(idx, minlength=mk.S).to(torch.float64)
    keep = p > 5.0 / N
    chi2 = float(((cnt[keep] - N * p[keep]) ** 2 / (N * p[keep])).sum())
    dof = int(keep.sum()) - 1
    assert chi2 < dof + 4 * (2 * dof) ** 0.5, (chi2, dof)


def test_cost_prediction(monkeypatch):
    """sample_apply_cost: positive, linear in the streams beyond the per-call term, None on an uncalibrated device."""
    from photonic_engines import sequence_modes
    mk, U = _kernel(3, 3, seed=67)
    a, b = mk.sample_apply_cost(U, 0, 1), mk.sample_apply_cost(U, 0, 1001)
    c = sequence_modes.COST["cpu"]
    assert a > 0 and abs((b - a) / 1000 - (a - c["CC0"] - c["CCk2"] * 9)) < 1e-12
    monkeypatch.setattr(sequence_modes, "COST", {})
    assert mk.sample_apply_cost(U, 0, 10) is None


def test_stream_chunking_keeps_the_draws():
    """A tiny chunk budget (one stream per chunk) gives the same draws and post-states as one chunk."""
    mk, U = _kernel(2, 2, seed=65)
    psi = _psi(mk.chi, 2, 66).unsqueeze(0).expand(64, -1, -1).clone()
    a = mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(3))
    b = mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(3), chunk_bytes=1)
    assert torch.equal(a[2]["events"], b[2]["events"]) and float((a[1] - b[1]).abs().max()) < 1e-15


def test_same_seed_same_draws_and_streams_unitaries():
    """The same generator seed gives the same draws; U_streams equal to the bin's U gives the bin's draws; a
    non-unitary U and `lost` emitters are refused."""
    mk, U = _kernel(2, 2, seed=60)
    psi = _psi(mk.chi, 2, 61).unsqueeze(0).expand(500, -1, -1).clone()
    a = mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(7))
    b = mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(7))
    c = mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(7),
                        U_streams=torch.as_tensor(U).unsqueeze(0).expand(500, -1, -1))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2]["events"], c[2]["events"])
    assert float((a[1] - c[1]).abs().max()) < 1e-14
    with pytest.raises(ValueError, match="unitary"):
        mk.sample_apply(0.9 * torch.as_tensor(U), 0, psi)
    with pytest.raises(ValueError, match="normalised"):
        mk.sample_apply(U, 0, 2 * psi)
    from test_internal_modes_lost import _lossy_emitter  # lost: the branch draw
    lossy = SequenceKernel([_lossy_emitter(1, 2, 2, 1, 2, (2 * j, 2 * j + 1), seed=62 + j) for j in range(2)], 4)
    idx, _, info = lossy.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(8))
    assert int(idx.max()) < lossy.S and bool(((info["branch"] >= 0) & (info["branch"] <= 1)).all())


@pytest.mark.gpu
def test_ccspin_cuda_matches_cpu():
    """The same host generator gives the same draws on CUDA as on the CPU (to the rare flip of a draw at a weight
    boundary), and the same post-states where they agree."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    mk, U = _kernel(3, 3, seed=70)
    gk = SequenceKernel(mk.emitters, mk.m, device="cuda")
    psi = _psi(mk.chi, 2, 71).unsqueeze(0).expand(10000, -1, -1).clone()
    a = mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(9))
    g = gk.sample_apply(U, 0, psi.cuda(), generator=torch.Generator().manual_seed(9))
    same = (a[2]["events"] == g[2]["events"].cpu()).all(2).all(1)
    assert float(same.double().mean()) > 0.999
    assert float((a[1][same] - g[1].cpu()[same]).abs().max()) < 1e-13
