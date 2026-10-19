"""The algorithm set of the T1 / F2a comparison (docs/ALGORITHMS.md, `clifford/variants.py`,
`clifford/cofactors_input.py`, `fock_sample(algo=...)`).

Oracle: Ryser permanents (`slos/reference.py`). Every algorithm computes the same k candidate
amplitudes of a chain step from a different sum; each is checked against the Laplace expansion of
Ryser cofactors, on blocks with repeated rows AND repeated columns, and its addend count against
the closed form the GPU planner uses. The Triton "input" kernel runs under the CPU interpreter
here (TRITON_INTERPRET=1 from conftest), a few seconds per call, so it is exercised on small
steps only; the full-sweep segmentation is covered by forcing many lanes per stream.
"""
import os

import pytest

torch = pytest.importorskip("torch")

# The Triton tests below run on the device when there is one (a CUDA host does not set TRITON_INTERPRET,
# and a Triton kernel cannot take CPU tensors there), under the CPU interpreter otherwise.
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") != "1" else "cpu"

from helpers import haar_unitary

from photonic_engines.clifford.variants import (
    ALGOS,
    addends,
    cofactors_2018,
    cofactors_2020,
    cofactors_percof,
    fock_sample_reference,
    qvec_input,
    step_amplitudes,
    strong_then_sample,
)
from photonic_engines.slos import get_layers, reference_probs
from photonic_engines.slos.reference import permanent

M = 8
U = haar_unitary(M, 1)

# (chosen output rows, input columns sigma(1..k)): distinct, repeated rows, repeated columns, both
STEPS = [
    ([], [3]),
    ([4], [0, 0]),
    ([1, 1, 1], [2, 2, 0, 0]),
    ([0, 2, 2, 5], [0, 1, 1, 2, 3]),
    ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
    ([2, 2, 2, 2], [0, 0, 0, 0, 1]),
    ([3, 3, 1, 0, 6, 6], [0, 1, 2, 3, 4, 5, 6]),
]


def _ryser_cofactors(B):
    K = B.shape[1]
    return torch.stack([permanent(B[:, [j for j in range(K) if j != c]]) for c in range(K)])


@pytest.mark.parametrize("rows,cols", STEPS)
@pytest.mark.parametrize("algo", ALGOS)
def test_every_algorithm_gives_the_ryser_amplitudes(rows, cols, algo):
    """amp[rho] = per(A[rows + [rho], cols]) for all m candidates, whatever the sum used."""
    B = U[rows][:, cols]
    ref = U[:, cols] @ _ryser_cofactors(B)
    amp, T = step_amplitudes(U, rows, cols, algo)
    assert torch.allclose(amp, ref, rtol=0, atol=1e-13 * ref.abs().max())
    assert addends(algo, rows, cols) == T, "the closed-form addend count must be what the sum enumerated"


def test_addend_counts_on_a_collision_free_step():
    """The T1 content: on distinct rows and columns 2018 = 2^(k-2), 2020 = the same, input = 2^(k-1)
    (twice 2018, no collapse to gain from), percof = k x 2020, whatever m."""
    rows, cols = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]
    assert addends("2018", rows, cols) == 16
    assert addends("2020", rows, cols) == 16
    assert addends("input", rows, cols) == 32
    assert addends("percof", rows, cols) == 6 * 16


def test_collapse_sides():
    """Output collisions shrink 2020 and percof, input collisions shrink "input" — and never the other way."""
    rows, cols = [2, 2, 2, 2], [0, 1, 2, 3, 4]              # one output mode 4 times: prod(s+1) = 5, not 2^3
    assert addends("2020", rows, cols) == 5 < addends("2018", rows, cols) == 8
    assert addends("input", rows, cols) == 16
    rows, cols = [0, 1, 2, 3], [7, 7, 7, 7, 7]              # one input mode 5 times: (5+1)/2 = 3
    assert addends("input", rows, cols) == 3 < addends("2018", rows, cols) == 8
    assert addends("2020", rows, cols) == 8


def test_cofactor_functions_agree_with_each_other_and_with_ryser():
    rows, cols = [0, 2, 2, 5], [0, 1, 1, 2, 3]
    B = U[rows][:, cols]
    ref = _ryser_cofactors(B)
    for P, _ in (cofactors_2018(B), cofactors_2020(B, rows), cofactors_percof(B, rows)):
        assert torch.allclose(P, ref, rtol=0, atol=1e-13 * ref.abs().max())
    Q, dcol, _ = qvec_input(B, cols)
    # Q_mu = sum of the cofactors of the copies of column mu
    for mu, q in zip(dcol, Q):
        assert abs(q - sum(ref[c] for c in range(len(cols)) if cols[c] == mu)) < 1e-13 * ref.abs().max()


def test_reference_chain_samples_the_exact_distribution():
    """n = 3 photons (two in one mode) in m = 4: 3000 shots of the plain-torch chain with each
    algorithm against the permanent distribution; tolerance from shot noise (5 sigma on the TVD)."""
    m, in_occ = 4, [2, 1, 0, 0]
    in_modes = [0, 0, 1]
    U4 = haar_unitary(m, 2)
    p = reference_probs(U4, 3, m, in_occ)
    states = [tuple(t) for t in get_layers(3, m).final_occupations()]
    idx = {s: i for i, s in enumerate(states)}
    shots = 3000
    tvd_floor = 5 * (len(states) / shots) ** 0.5 / 2
    for algo in ALGOS:
        occ = fock_sample_reference(U4, in_modes, shots, algo=algo, generator=torch.Generator().manual_seed(3))
        h = torch.zeros(len(states), dtype=torch.float64)
        for o in occ.tolist():
            h[idx[tuple(o)]] += 1
        tvd = 0.5 * (h / shots - p).abs().sum().item()
        assert tvd < tvd_floor, (algo, tvd, tvd_floor)


def test_strong_then_sample_matches_the_reference_distribution():
    m, in_occ, in_modes = 5, [1, 1, 0, 0, 0], [0, 1]
    U5 = haar_unitary(m, 4)
    st = {}
    occ = strong_then_sample(U5, in_modes, 4000, generator=torch.Generator().manual_seed(0), dtype=torch.complex128, stats=st)
    assert occ.shape == (4000, m) and occ.sum(1).unique().tolist() == [2]
    p = reference_probs(U5, 2, m, in_occ)
    states = [tuple(t) for t in get_layers(2, m).final_occupations()]
    assert st["states"] == len(states)
    h = torch.zeros(len(states), dtype=torch.float64)
    idx = {s: i for i, s in enumerate(states)}
    for o in occ.tolist():
        h[idx[tuple(o)]] += 1
    assert 0.5 * (h / 4000 - p).abs().sum().item() < 5 * (len(states) / 4000) ** 0.5 / 2


# ---------------------------------------------------------------- the Triton kernel (CPU interpreter)

@pytest.mark.parametrize("rows,cols", STEPS[1:6])
@pytest.mark.parametrize("lanes", [1 << 17, 4])
def test_input_kernel_matches_the_reference_sum(rows, cols, lanes):
    pytest.importorskip("triton")
    from photonic_engines.clifford.cofactors_input import qvec_input_rows
    Q, dcol = qvec_input_rows(U[None].to(DEV), torch.tensor([rows], device=DEV), torch.tensor([cols], device=DEV),
                              target_total_lanes=lanes)
    Q = Q.cpu()
    Qr, ids, _ = qvec_input(U[rows][:, cols], cols)
    ref = dict(zip(ids, Qr.tolist()))
    D = len(ids)
    scale = max(abs(v) for v in ref.values())
    for mu, q in zip(dcol[0, :D].tolist(), Q[0, :D].tolist()):
        assert abs(q - ref[mu]) < 1e-13 * scale
    assert bool((Q[0, D:] == 0).all()), "padded columns carry Q = 0"


def test_sampler_algorithms_draw_the_same_chain():
    """Same seed, same sigma, same Philox draws and the same exact weights: the four algorithms
    of `fock_sample` return the SAME samples (they differ in the sum evaluated, not in the chain).
    Input with a repeated mode so every collapse is exercised."""
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    U6 = haar_unitary(6, 3).to(DEV)
    out = {algo: fock_sample(U6, [0, 0, 2, 3], 16, backend="triton", algo=algo,
                             generator=torch.Generator(device=DEV).manual_seed(11)) for algo in ALGOS}
    for algo in ALGOS[1:]:
        assert torch.equal(out["2018"], out[algo]), algo
    assert out["2018"].sum(1).unique().tolist() == [4]


def test_input_algo_needs_the_triton_backend():
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    with pytest.raises(NotImplementedError):
        fock_sample(haar_unitary(4, 1).to(DEV), [0, 1], 2, backend="cuda", algo="input")
