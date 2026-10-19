"""EBS batched kernel (SuperpositionSLOSKernel) correctness tests.

The kernel-level tests use only torch: they gate the batched EBS sweep against (a) SC's own
naive per-component coherent loop and (b) an INDEPENDENT coherent-permanent reference that
shares no code with either kernel. The benchmark-integration tests additionally cross-check
SC's EBS distribution against merlin's EBS (they need merlin installed).
"""
import math

import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary, superposition_components

from photonic_engines.slos.basis import get_layers
from photonic_engines.slos.kernel import FockSLOSKernel, SuperpositionSLOSKernel
from photonic_engines.slos.reference import _modes_from_occupation, permanent


def _U(m):
    return haar_unitary(m, seed=m).to(torch.complex64)


def _coherent_reference_probs(U, n, m, comps, weights, mask="fock"):
    """Independent gold: coherent sum of per-component permanent amplitudes over the final basis."""
    U = torch.as_tensor(U, dtype=torch.complex128)
    layers = get_layers(n, m, mask)
    in_modes = [_modes_from_occupation(c) for c in comps]
    s_norms = [math.prod(math.factorial(int(x)) for x in c) for c in comps]
    amp = torch.zeros(len(layers.final_states), dtype=torch.complex128)
    for idx, t in enumerate(layers.final_states):
        out_modes = _modes_from_occupation(t)
        t_norm = math.prod(math.factorial(int(x)) for x in t)
        a = torch.zeros((), dtype=torch.complex128)
        for _c, im, s_norm, w in zip(comps, in_modes, s_norms, weights):
            if not out_modes:
                term = torch.tensor(1.0 + 0j, dtype=torch.complex128)
            else:
                sub = U[out_modes][:, im]
                term = permanent(sub) / math.sqrt(s_norm * t_norm)
            a = a + w * term
        amp[idx] = a
    return (amp.conj() * amp).real


def _naive_probs(U, n, m, comps, weights, mask="fock"):
    kernels = [FockSLOSKernel(n, m, list(c), mask=mask) for c in comps]
    amp = sum(w * k.amplitudes(U) for w, k in zip(weights, kernels))
    # deferred bosonic normalization: apply prod(t_j!) to the summed |amplitude|^2
    return (amp.conj() * amp).real * kernels[0].output_prob_norm


@pytest.mark.parametrize("k", [1, 2, 3, 6, 10])
def test_ebs_equals_naive_loop(k):
    n, m = 2, 6
    U = _U(m)
    comps = superposition_components(n, m, k)
    w = [1.0 / math.sqrt(k)] * k
    p_naive = _naive_probs(U, n, m, comps, w)
    p_ebs = SuperpositionSLOSKernel(n, m, comps).probs(U, weights=w)
    assert torch.allclose(p_naive, p_ebs, atol=1e-5), (p_naive - p_ebs).abs().max()


def test_ebs_equals_independent_permanent_reference():
    # strongest gate: shares no code with either SLOS kernel
    n, m, k = 2, 6, 5
    U = _U(m)
    comps = superposition_components(n, m, k)
    w = [1.0 / math.sqrt(k)] * k
    p_ref = _coherent_reference_probs(U, n, m, comps, w).to(torch.float32)
    p_ebs = SuperpositionSLOSKernel(n, m, comps).probs(U, weights=w)
    assert torch.allclose(p_ref, p_ebs, atol=1e-4), (p_ref - p_ebs).abs().max()


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_chunking_is_invariant(chunk):
    n, m, k = 2, 8, 6
    U = _U(m)
    comps = superposition_components(n, m, k)
    w = [1.0 / math.sqrt(k)] * k
    ker = SuperpositionSLOSKernel(n, m, comps)
    p_full = ker.probs(U, weights=w, chunk_size=None)
    p_chunk = ker.probs(U, weights=w, chunk_size=chunk)
    assert torch.allclose(p_full, p_chunk, atol=1e-6), (p_full - p_chunk).abs().max()


def test_batched_unitary():
    n, m, k, B = 2, 8, 4, 5
    U = _U(m).unsqueeze(0).repeat(B, 1, 1)
    comps = superposition_components(n, m, k)
    w = [1.0 / math.sqrt(k)] * k
    p = SuperpositionSLOSKernel(n, m, comps).probs(U, weights=w)
    assert p.shape[0] == B
    # every batch row identical (same U repeated) and matches the unbatched result
    p0 = SuperpositionSLOSKernel(n, m, comps).probs(_U(m), weights=w)
    for b in range(B):
        assert torch.allclose(p[b], p0, atol=1e-6)


@pytest.mark.parametrize("mask", ["unbunched", "dual_rail"])
def test_ebs_masks_prune_and_match_naive(mask):
    n, m, k = 2, 8, 4
    U = _U(m)
    comps = superposition_components(n, m, k)
    w = [1.0 / math.sqrt(k)] * k
    ker = SuperpositionSLOSKernel(n, m, comps, mask=mask)
    p_ebs = ker.probs(U, weights=w)
    p_naive = _naive_probs(U, n, m, comps, w, mask=mask)
    assert torch.allclose(p_ebs, p_naive, atol=1e-5)
    # masked output space is strictly smaller than full fock
    assert ker.output_size() < SuperpositionSLOSKernel(n, m, comps, mask="fock").output_size()


def test_gradient_flows_through_ebs():
    n, m, k = 2, 6, 4
    U = _U(m).requires_grad_(True)
    comps = superposition_components(n, m, k)
    w = [1.0 / math.sqrt(k)] * k
    p = SuperpositionSLOSKernel(n, m, comps).probs(U, weights=w)
    (p * p).sum().backward()
    assert U.grad is not None and U.grad.abs().sum() > 0


def test_wrong_photon_count_raises():
    with pytest.raises(ValueError):
        SuperpositionSLOSKernel(2, 6, [[1, 1, 1, 0, 0, 0]])   # 3 photons, n=2
