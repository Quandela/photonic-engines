"""SLOScore kernel correctness: the fast SLOS recurrence must match an independent
permanent-based reference, and the mask-pruned bases must have the exact expected sizes."""

import pytest

torch = pytest.importorskip("torch")

from helpers import occupation_list

from photonic_engines.masks import state_space_size
from photonic_engines.slos.basis import get_layers
from photonic_engines.slos.kernel import FockSLOSKernel
from photonic_engines.slos.reference import permanent, reference_probs


def _haar_unitary(m, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(
        m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    return q * (d / d.abs()).unsqueeze(0)


def test_permanent_matches_hand_value():
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)
    assert abs(permanent(m).item() - (1 * 4 + 2 * 3)) < 1e-9   # per([[a,b],[c,d]]) = ad + bc


@pytest.mark.parametrize("mask,n,m", [
    ("fock", 3, 6), ("unbunched", 3, 6), ("dual_rail", 3, 6),
    ("sparse_1in4", 2, 8), ("tail_empty", 3, 7),
])
def test_basis_size_matches_fock_module(mask, n, m):
    assert get_layers(n, m, mask).output_size() == state_space_size(n, m, mask)


@pytest.mark.parametrize("mask", ["fock", "unbunched", "dual_rail", "tail_empty"])
def test_kernel_matches_permanent_reference(mask):
    n, m = 3, 6
    U = _haar_unitary(m, seed=7)
    occ = occupation_list(n, m, mask)
    ker = FockSLOSKernel(n, m, occ, mask=mask, dtype=torch.complex128)
    got = ker.probs(U)
    ref = reference_probs(U, n, m, occ, mask=mask)
    assert torch.allclose(got, ref, atol=1e-9), f"{mask}: max diff {(got-ref).abs().max()}"


def test_fock_distribution_normalized():
    n, m = 3, 6
    U = _haar_unitary(m, seed=11)
    occ = occupation_list(n, m, "fock")
    p = FockSLOSKernel(n, m, occ, mask="fock", dtype=torch.complex128).probs(U)
    assert abs(p.sum().item() - 1.0) < 1e-9        # unitary + full basis => normalized


def test_masks_prune_output_space():
    n, m = 3, 6
    full = get_layers(n, m, "fock").output_size()
    for mask in ("unbunched", "dual_rail"):
        assert get_layers(n, m, mask).output_size() < full


def test_batched_matches_looped():
    n, m = 2, 4
    occ = occupation_list(n, m, "fock")
    ker = FockSLOSKernel(n, m, occ, mask="fock", dtype=torch.complex64)
    Us = torch.stack([_haar_unitary(m, s).to(torch.complex64) for s in (1, 2, 3)])
    batched = ker.probs(Us)
    for b in range(3):
        assert torch.allclose(batched[b], ker.probs(Us[b]), atol=1e-5)


def test_gradient_flows_through_kernel():
    n, m = 2, 4
    occ = occupation_list(n, m, "fock")
    U = _haar_unitary(m, 5).to(torch.complex64).requires_grad_(True)
    p = FockSLOSKernel(n, m, occ, mask="fock", dtype=torch.complex64).probs(U)
    (p * p).sum().backward()
    assert U.grad is not None and torch.isfinite(U.grad).all() and U.grad.abs().sum() > 0


def test_grouped_kernel_reduces_and_conserves_mass():
    n, m = 3, 6
    occ = occupation_list(n, m, "fock")
    ker = FockSLOSKernel(n, m, occ, mask="fock", dtype=torch.complex128)
    U = _haar_unitary(m, 3)
    p = ker.probs(U)
    S = p.numel()
    groups = torch.arange(S) % 4                       # 4 arbitrary bins
    gp = ker.group_matrix(groups).probs(U)
    assert gp.numel() == 4
    assert abs(gp.sum().item() - p.sum().item()) < 1e-9   # grouping conserves probability mass


def test_immediate_normalization_matches_deferred():
    """H-MEM-TRADE ablation gate: the SC_ImmNorm arm (per-edge factors in-loop) must produce
    the same distribution as the deferred default, on every mask, batched, and with a bunched
    input (which exercises input_norm). If this breaks, the ablation pair in the objective run
    is comparing different physics and its speed/memory numbers are void."""
    torch.manual_seed(0)
    cases = [(2, 4, "fock", [1, 0, 1, 0]), (3, 6, "fock", [1, 0, 1, 0, 1, 0]),
             (2, 4, "unbunched", [1, 0, 1, 0]), (4, 8, "dual_rail", [1, 0, 1, 0, 1, 0, 1, 0]),
             (3, 6, "tail_empty", [1, 1, 1, 0, 0, 0]), (3, 4, "fock", [2, 0, 1, 0])]
    for n, m, mask, occ in cases:
        U = _haar_unitary(m, seed=13)
        pd = FockSLOSKernel(n, m, occ, mask=mask).probs(U)
        pi = FockSLOSKernel(n, m, occ, mask=mask, normalization="immediate").probs(U)
        assert torch.allclose(pd, pi, atol=1e-6), f"n{n}/m{m}/{mask}: max diff {(pd-pi).abs().max()}"
        Ub = torch.stack([_haar_unitary(m, seed=17 + i) for i in range(3)])
        pdb = FockSLOSKernel(n, m, occ, mask=mask).probs(Ub)
        pib = FockSLOSKernel(n, m, occ, mask=mask, normalization="immediate").probs(Ub)
        assert torch.allclose(pdb, pib, atol=1e-6), f"n{n}/m{m}/{mask} batched"


# ---- H-GRAD-CHECKPOINT experiment arm: recomputed backward gathers (opt-in) ----------------

def test_checkpointed_kernel_matches_standard_outputs():
    """The checkpointed layer must be output-IDENTICAL to the standard path (same math,
    different autograd bookkeeping) - batched and unbatched."""
    n, m = 3, 6
    occ = occupation_list(n, m, "fock")
    std = FockSLOSKernel(n, m, occ, device="cpu")
    ckp = FockSLOSKernel(n, m, occ, device="cpu", gradient_checkpointing=True)
    U = _haar_unitary(m, seed=23).to(torch.complex64)
    assert torch.allclose(std.probs(U), ckp.probs(U), atol=1e-7)
    Ub = torch.stack([_haar_unitary(m, s).to(torch.complex64) for s in (4, 5)])
    assert torch.allclose(std.probs(Ub), ckp.probs(Ub), atol=1e-7)


def test_checkpointed_kernel_gradient_matches_standard():
    """d(loss)/dU from the recomputed-gather backward must equal the default autograd path
    (torch's Wirtinger convention) - the correctness gate H-GRAD-CHECKPOINT rests on."""
    n, m = 3, 6
    occ = occupation_list(n, m, "fock")
    Ub = torch.stack([_haar_unitary(m, s).to(torch.complex64) for s in (6, 7)])
    grads = []
    for ck in (False, True):
        k = FockSLOSKernel(n, m, occ, device="cpu", gradient_checkpointing=ck)
        U = Ub.clone().requires_grad_(True)
        p = k.probs(U)
        w = torch.linspace(0.5, 1.5, p.shape[-1])       # non-uniform loss: exercises all states
        (p * w).sum().backward()
        grads.append(U.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6), (
        (grads[0] - grads[1]).abs().max().item())


def test_checkpointed_kernel_rejects_immediate_normalization():
    with pytest.raises(ValueError):
        FockSLOSKernel(3, 6, occupation_list(3, 6, "fock"), device="cpu",
                   normalization="immediate", gradient_checkpointing=True)


class _OOMWithoutNotes(torch.cuda.OutOfMemoryError):
    """An OOM error as on Python 3.10, where BaseException has no add_note (PEP 678 is 3.11)."""
    @property
    def add_note(self):
        raise AttributeError("add_note")


def _hints(e: BaseException) -> list:
    """Strings the hint can live in: the 3.11+ notes, or the 3.10 fallback in e.args."""
    return [*getattr(e, "__notes__", []), *(a for a in e.args if isinstance(a, str))]


@pytest.mark.parametrize("oom", [torch.cuda.OutOfMemoryError, _OOMWithoutNotes])
def test_oom_hint_attached_when_knob_would_help(oom):
    """The OOM guidance: a hint, never a retry - attached only when grad is enabled and
    gradient_checkpointing is off (i.e. when flipping the knob would actually help); with
    and without exception notes (Python >= 3.11 / 3.10)."""
    occ = occupation_list(2, 4, "fock")
    plain = FockSLOSKernel(2, 4, occ, device="cpu")
    ckpt = FockSLOSKernel(2, 4, occ, device="cpu", gradient_checkpointing=True)
    err = oom("CUDA out of memory (simulated)")
    with torch.enable_grad():
        plain._add_oom_hint(err)
    assert any("gradient_checkpointing" in n for n in _hints(err)), err
    err2 = oom("simulated")
    with torch.enable_grad():
        ckpt._add_oom_hint(err2)          # already checkpointing: no hint
    assert _hints(err2) == ["simulated"]
    err3 = oom("simulated")
    with torch.no_grad():
        plain._add_oom_hint(err3)          # inference OOM: knob would not help
    assert _hints(err3) == ["simulated"]
    # end-to-end: an OOM escaping amplitudes() carries the hint
    plain._amplitudes = lambda U: (_ for _ in ()).throw(oom("CUDA out of memory (simulated)"))
    U = torch.eye(4, dtype=torch.complex64)
    with torch.enable_grad(), pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
        plain.amplitudes(U)
    assert any("gradient_checkpointing" in n for n in _hints(excinfo.value))
    if oom is _OOMWithoutNotes:
        assert "gradient_checkpointing" in str(excinfo.value)


@pytest.mark.parametrize("n,m,mask", [(3, 6, "fock"), (5, 8, "fock"), (4, 8, "unbunched"), (3, 6, "dual_rail")])
def test_mode_layout_equals_edge_layout(n, m, mask):
    occ = occupation_list(n, m, mask)
    U = _haar_unitary(m, seed=n + m).to(torch.complex64)
    a = FockSLOSKernel(n, m, occ, mask=mask, layout="mode").probs(U)
    b = FockSLOSKernel(n, m, occ, mask=mask, layout="edge").probs(U)
    assert torch.allclose(a, b, atol=1e-6)
    Ub = torch.stack([_haar_unitary(m, s).to(torch.complex64) for s in (1, 2)])
    assert torch.allclose(FockSLOSKernel(n, m, occ, mask=mask, layout="mode").probs(Ub),
                          FockSLOSKernel(n, m, occ, mask=mask, layout="edge").probs(Ub), atol=1e-6)
