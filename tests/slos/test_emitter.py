"""EmitterSLOSKernel (bond-carrying SLOS) against the exponential oracles: the expanded
L^n-component superposition through the EBS kernel, and an independent coherent-permanent
reference; plus traced-vs-projected consistency, Kraus operators, batching, masks, grads."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos.basis import get_layers
from photonic_engines.slos.emitter import EmitterSLOSKernel, sample_from_probs
from photonic_engines.slos.kernel import SuperpositionSLOSKernel
from photonic_engines.slos.reference import _modes_from_occupation, permanent

C128 = torch.complex128


def _caterpillar(n, phis, final=None, m=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final, m=m)


def _random_emitter(n, L, chi, m, seed, final=True):
    rng = np.random.default_rng(seed)
    ops = rng.normal(size=(n, L, chi, chi)) + 1j * rng.normal(size=(n, L, chi, chi))
    modes = tuple(tuple(range(k * L, (k + 1) * L)) for k in range(n))
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    f = (rng.normal(size=chi) + 1j * rng.normal(size=chi)) if final else None
    return EmitterMPS(ops=ops, modes=modes, s0=s0, m=m, final=f)


def _ebs_probs(state, U, mask="fock"):
    """Oracle 1: expand to Fock components, evaluate with the EBS kernel, add the mixture."""
    total = None
    for sup in state.expand():
        comps = [c.occupation for c in sup.components]
        k = SuperpositionSLOSKernel(state.n, state.m, comps, mask=mask, dtype=C128)
        p = k.probs(U, weights=list(sup.amplitudes))
        total = p if total is None else total + p
    return total


def _permanent_probs(state, U, mask="fock"):
    """Oracle 2: coherent permanent sum per output state, no SLOS code involved."""
    layers = get_layers(state.n, state.m, mask)
    total = torch.zeros(len(layers.final_states), dtype=torch.float64)
    for sup in state.expand():
        amp = torch.zeros(len(layers.final_states), dtype=C128)
        for idx, t in enumerate(layers.final_states):
            out_modes = _modes_from_occupation(t)
            t_norm = math.prod(math.factorial(int(x)) for x in t)
            a = 0j
            for c, w in zip(sup.components, sup.amplitudes):
                sub = U[out_modes][:, c.modes()]
                a += w * permanent(sub).item() / math.sqrt(t_norm)
            amp[idx] = a
        total += (amp.conj() * amp).real
    return total


@pytest.mark.parametrize("n,phis", [(3, [0, 0]), (3, [math.pi, math.pi]), (4, [math.pi, 0, math.pi])])
@pytest.mark.parametrize("final", [None, (1, 0), (0.6, 0.8j)])
def test_caterpillar_matches_ebs_expansion(n, phis, final):
    st = _caterpillar(n, phis, final=final)
    U = haar_unitary(st.m, seed=n * 7 + len(phis))
    p = EmitterSLOSKernel(st, dtype=C128).probs(U)
    p_ref = _ebs_probs(st, U)
    assert torch.allclose(p, p_ref, atol=1e-10), (p - p_ref).abs().max()
    expected_mass = 1.0 if final is None else sum(abs(a) ** 2 for a in st.expand()[0].amplitudes)
    assert abs(float(p.sum()) - expected_mass) < 1e-10


@pytest.mark.parametrize("L,chi", [(2, 2), (3, 2), (2, 3)])
def test_random_emitter_matches_permanent_reference(L, chi):
    n = 3
    st = _random_emitter(n, L, chi, m=n * L + 1, seed=L * 10 + chi)
    U = haar_unitary(st.m, seed=3)
    p = EmitterSLOSKernel(st, dtype=C128).probs(U)
    assert torch.allclose(p, _permanent_probs(st, U), atol=1e-10)


def test_traced_equals_sum_over_final_basis():
    st = _random_emitter(3, 2, 2, m=6, seed=5, final=False)
    U = haar_unitary(6, seed=11)
    p_traced = EmitterSLOSKernel(st, dtype=C128).probs(U)
    acc = torch.zeros_like(p_traced)
    for s in range(st.chi):
        f = np.zeros(st.chi, dtype=complex); f[s] = 1
        proj = EmitterMPS(st.ops, st.modes, st.s0, st.m, final=f)
        acc += EmitterSLOSKernel(proj, dtype=C128).probs(U)
    assert torch.allclose(p_traced, acc, atol=1e-12)


def test_kraus_operators_consistent_with_bond_amplitudes():
    st = _random_emitter(3, 2, 2, m=6, seed=8, final=False)
    U = haar_unitary(6, seed=2)
    k = EmitterSLOSKernel(st, dtype=C128)
    K = k.kraus(U)                                    # [S, final, init]
    phi = k.bond_amplitudes(U)                        # [final, S]
    s0 = torch.as_tensor(st.s0, dtype=C128)
    assert torch.allclose(torch.einsum("sfi,i->fs", K, s0), phi, atol=1e-12)


def test_kraus_operators_of_isometric_emission_resolve_identity():
    # caterpillar emission is an isometry (sum_l M_l^dag M_l = I), so with a unitary U and full
    # detection the physical Kraus operators satisfy sum_t K_t^dag K_t = I
    st = _caterpillar(3, [0, math.pi])
    U = haar_unitary(st.m, seed=2)
    k = EmitterSLOSKernel(st, dtype=C128)
    K = k.kraus(U)
    Kphys = K * torch.sqrt(torch.as_tensor(k.layers.output_prob_norms(), dtype=C128)).view(-1, 1, 1)
    ident = torch.einsum("sfi,sfj->ij", Kphys.conj(), Kphys)
    assert torch.allclose(ident, torch.eye(2, dtype=C128), atol=1e-10)


def test_batched_equals_loop_and_masks():
    n = 3
    st = _caterpillar(n, [0, math.pi])                 # m = 2n, dual-rail spaced input
    Us = torch.stack([haar_unitary(st.m, seed=s) for s in (1, 2, 3)])
    for mask in ("fock", "unbunched", "dual_rail"):
        k = EmitterSLOSKernel(st, mask=mask, dtype=C128)
        pb = k.probs(Us)
        for b in range(3):
            assert torch.allclose(pb[b], k.probs(Us[b]), atol=1e-12)
            assert torch.allclose(pb[b], _ebs_probs(st, Us[b], mask=mask), atol=1e-10)


def test_complex64_close_to_complex128():
    st = _caterpillar(4, [0, 0, 0])
    U = haar_unitary(st.m, seed=4)
    p64 = EmitterSLOSKernel(st, dtype=torch.complex64).probs(U.to(torch.complex64))
    p128 = EmitterSLOSKernel(st, dtype=C128).probs(U)
    assert torch.allclose(p64.double(), p128, atol=1e-5)


def test_gradient_wrt_unitary_and_ops():
    st = _random_emitter(2, 2, 2, m=4, seed=9)
    U = haar_unitary(4, seed=6).requires_grad_(True)
    ops = torch.as_tensor(st.ops, dtype=C128).requires_grad_(True)
    k = EmitterSLOSKernel(st, dtype=C128, ops_tensor=ops)
    loss = k.probs(U)[0]
    loss.backward()
    assert U.grad is not None and torch.isfinite(U.grad).all()
    assert ops.grad is not None and torch.isfinite(ops.grad).all()
    # finite-difference check on one real parameter of the ops tensor
    eps = 1e-6
    with torch.no_grad():
        ops2 = ops.clone(); ops2[0, 0, 0, 0] += eps
        k2 = EmitterSLOSKernel(st, dtype=C128, ops_tensor=ops2)
        fd = (k2.probs(U.detach())[0] - loss.detach()) / eps
    assert abs(float(ops.grad[0, 0, 0, 0].real) - float(fd)) < 1e-4


def test_sampling_from_probs_matches_distribution():
    st = _caterpillar(3, [0, 0])
    U = haar_unitary(st.m, seed=12)
    k = EmitterSLOSKernel(st, dtype=C128)
    p = k.probs(U)
    occ = sample_from_probs(k.layers, p, 20000, generator=torch.Generator().manual_seed(0))
    assert occ.shape == (20000, st.m) and (occ.sum(1) == st.n).all()
    idx = {t: i for i, t in enumerate(k.layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= 20000
    assert 0.5 * (emp - p).abs().sum() < 0.03
