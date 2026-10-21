"""Internal photon modes, coarse outcome classes (docs/SEQUENCE.md §14): weighted
classes in `condition` (route F, the pairing route, `rest=`, `lost`) and Kraus sets grouped by class in `kraus`."""
import numpy as np
import pytest
import torch
from helpers import haar_unitary
from test_internal_modes_lost import _lossy_emitter
from test_internal_modes_route_f import _mode_emitter, _superops

from photonic_engines.sequence import SequenceKernel

TOL = 1e-14        # class sums: the criterion (1e-15 trips on the H100 host CPU's rounding of class sums, run 20261020_221250)


def _rho(D, B, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, D, generator=g, dtype=torch.complex128)
    r = z @ z.conj().transpose(1, 2)
    return r / torch.einsum("bii->b", r).view(-1, 1, 1)


def _kernels():
    lossless = SequenceKernel([_mode_emitter(1, 2, 3, 2, (2 * j, 2 * j + 1), seed=10 + j) for j in range(2)], 4)
    lossy = SequenceKernel([_lossy_emitter(1, 2, 3, 1, 2, (2 * j, 2 * j + 1), seed=20 + j) for j in range(2)], 4)
    return [(lossless, haar_unitary(4, 11)), (lossy, haar_unitary(4, 21))]


@pytest.mark.parametrize("which", [0, 1])
def test_weighted_class_is_the_weighted_sum(which):
    """sum_t w_t E_t(rho) by F and P' equals the weighted sum of the members' conditioned registers (< TOL); weights
    per entry (repeated entries add up) and per pattern; a mask equals 0/1 weights; with rest=X."""
    mk, U = _kernels()[which]
    rho = _rho(mk.chi, 2, 30 + which)
    single = torch.stack([mk.condition(U, 0, rho, [t], route="F") for t in range(mk.S)], 1)    # [B, S, D, D]
    outs = [3, 0, 5, 3]
    w = [0.25, 1.5, 0.7, 0.5]                                                    # outcome 3 twice: 0.75 in all
    ref = 0.75 * single[:, 3] + 1.5 * single[:, 0] + 0.7 * single[:, 5]
    for route in ("F", "P"):
        got = mk.condition(U, 0, rho, outs, weights=w, route=route)
        assert float((got - ref).abs().max()) < TOL
    wS = torch.rand(mk.S, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    cls = [1, 2, 4]
    refS = torch.einsum("t,btxy->bxy", wS[cls].to(single.dtype), single[:, cls])
    assert float((mk.condition(U, 0, rho, cls, weights=wS, route="P") - refS).abs().max()) < TOL
    mask = np.zeros(mk.S, dtype=bool)
    mask[[0, 2, 5]] = True
    for route in ("F", "P"):
        a = mk.condition(U, 0, rho, mask, route=route)
        b = mk.condition(U, 0, rho, mask, weights=mask.astype(float), route=route)
        assert float((a - b).abs().max()) < TOL
    X = 2
    full = _rho(mk.chi * X, 1, 40)
    blocks = full.reshape(1, mk.chi, X, mk.chi, X).permute(0, 2, 4, 1, 3).reshape(X * X, mk.chi, mk.chi)
    ref_b = sum(wt * mk.condition(U, 0, blocks, [t], route="F") for t, wt in zip(outs, w))
    ref_b = ref_b.reshape(1, X, X, mk.chi, mk.chi).permute(0, 3, 1, 4, 2).reshape(mk.chi * X, mk.chi * X)
    for route in ("F", "P"):
        got = mk.condition(U, 0, full[0], outs, weights=w, route=route, rest=X)
        assert float((got - ref_b).abs().max()) < TOL


def test_weights_validation():
    mk, U = _kernels()[0]
    rho = _rho(mk.chi, 1, 50)[0]
    with pytest.raises(ValueError, match=">= 0"):
        mk.condition(U, 0, rho, [0, 1], weights=[1.0, -0.1])
    with pytest.raises(ValueError, match="one per outcome"):
        mk.condition(U, 0, rho, [0, 1], weights=[1.0, 1.0, 1.0])


@pytest.mark.parametrize("which", [0, 1])
def test_kraus_by_class(which):
    """kraus(classes=c): the class superoperators equal the sum of the members'; -1 drops an outcome; compressed, at
    most min(sum of the members' counts, chi^2) operators per class with the same channel (< 1e-14), and exactly the
    rank of the stacked member operators."""
    mk, U = _kernels()[which]
    c = np.full(mk.S, -1)
    c[[0, 1, 2]] = 0
    c[[3, 7]] = 1
    c[mk.S - 1] = 2
    KF, cF = mk.kraus(U, 0), mk.coarse(U, 0)
    J = _superops(KF, cF, mk.S)
    Kc, cc = mk.kraus(U, 0, classes=c), mk.coarse(U, 0, classes=c)
    Jc = _superops(Kc, cc, 3)
    for cls in range(3):
        assert float((Jc[cls] - J[np.nonzero(c == cls)[0]].sum(0)).abs().max()) < TOL
    assert Kc.shape[0] == int((torch.as_tensor(c)[cF] >= 0).sum())             # dropped outcomes: no operator
    Kz, cz = mk.kraus(U, 0, compress=1e-12, classes=c), mk.coarse(U, 0, compress=1e-12, classes=c)
    Jz = _superops(Kz, cz, 3)
    assert float((Jz - Jc).abs().max()) < 1e-14
    for cls in range(3):
        stacked = Kc[cc == cls].reshape(-1, mk.chi ** 2)
        s = torch.linalg.svdvals(stacked)
        rank = int((s > 1e-12 * s[0]).sum())
        assert int((cz == cls).sum()) == rank <= min(stacked.shape[0], mk.chi ** 2)
    with pytest.raises(NotImplementedError, match="route F"):
        mk.kraus(U, 0, compress=1e-12, classes=c, route="R")
    with pytest.raises(ValueError, match="one class id per pattern"):
        mk.kraus(U, 0, classes=c[:-1])
