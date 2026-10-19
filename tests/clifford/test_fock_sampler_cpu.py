"""Fast CPU gate on the batched Clifford Fock sampler (`photonic_engines.clifford.sampler`).

Without a CUDA device the Triton kernels run under the CPU interpreter (conftest sets
TRITON_INTERPRET=1), which is ~40 ms per shot — far too slow for a distribution test. The
statistical checks live in `check_sampler.py`, run by `test_scripts.py::test_cpu_interpreter`
(marked `slow`) and by the GPU job. What is gated here is everything that does NOT need many
shots: that both algorithms run, produce structurally valid samples, honour the generator, and
that the batched-circuit and loss paths dispatch correctly. Without this file
`clifford/sampler.py` is at 0 % coverage in the default CI run.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from helpers import haar_unitary

from photonic_engines import FockState
from photonic_engines.clifford import fock_sample, sample

M, IN_MODES, SHOTS = 4, [0, 1], 8
N = len(IN_MODES)
ALGOS = ["2018", "2020", "auto"]


def _valid(occ, shots, m=M, n=N):
    assert occ.shape == (shots, m)
    assert occ.dtype == torch.int64
    assert bool((occ >= 0).all())
    assert occ.sum(1).tolist() == [n] * shots


@pytest.mark.parametrize("algo", ALGOS)
def test_samples_are_valid_occupations(algo):
    occ = fock_sample(haar_unitary(M, 3), IN_MODES, SHOTS, backend="triton", algo=algo,
                      generator=torch.Generator().manual_seed(0))
    _valid(occ, SHOTS)


@pytest.mark.parametrize("algo", ALGOS)
def test_same_seed_reproduces_the_same_samples(algo):
    U = haar_unitary(M, 4)
    a = fock_sample(U, IN_MODES, SHOTS, backend="triton", algo=algo, generator=torch.Generator().manual_seed(7))
    b = fock_sample(U, IN_MODES, SHOTS, backend="triton", algo=algo, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_a_different_seed_moves_the_samples():
    """Sanity that the generator is actually consumed (8 shots of a 10-outcome space: a
    collision of the full batch has probability well below 1e-3 for a Haar circuit)."""
    U = haar_unitary(M, 5)
    a = fock_sample(U, IN_MODES, SHOTS, backend="triton", generator=torch.Generator().manual_seed(1))
    b = fock_sample(U, IN_MODES, SHOTS, backend="triton", generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_collisions_are_sampled_when_the_input_bunches():
    """Two photons in the SAME input mode: the 2020 form's repeated-row path (the `prod(s+1)`
    mixed-radix branch) is what runs here, not the 2018 Gray sweep."""
    occ = fock_sample(haar_unitary(M, 6), [0, 0], SHOTS, backend="triton", algo="2020",
                      generator=torch.Generator().manual_seed(0))
    _valid(occ, SHOTS)


def test_batched_circuits_are_laid_out_as_b_times_shots():
    """U [B, m, m] returns B*shots rows with stream s = b*shots + shot. Gated with two
    DETERMINISTIC circuits (mode permutations, one outcome each) so the block boundary is
    visible without statistics: block 0 must be the image of the input under P0 and block 1
    under P1. Note that two IDENTICAL circuits would NOT give identical blocks -- the random
    draws are per stream, not per circuit."""
    p0, p1 = [0, 1, 2, 3], [2, 3, 0, 1]                     # U[out, in] = 1 for out = p[in]
    U = torch.stack([torch.eye(M, dtype=torch.complex128)[p] for p in (p0, p1)])
    occ = fock_sample(U, IN_MODES, SHOTS, backend="triton", generator=torch.Generator().manual_seed(0))
    _valid(occ, 2 * SHOTS)
    assert occ[:SHOTS].tolist() == [[1, 1, 0, 0]] * SHOTS
    assert occ[SHOTS:].tolist() == [[0, 0, 1, 1]] * SHOTS


def test_distinct_circuits_in_a_batch_give_distinct_blocks():
    U = torch.stack([haar_unitary(M, 9), haar_unitary(M, 10)])
    occ = fock_sample(U, IN_MODES, SHOTS, backend="triton", generator=torch.Generator().manual_seed(0))
    _valid(occ, 2 * SHOTS)
    assert not torch.equal(occ[:SHOTS], occ[SHOTS:])


@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_loss_only_removes_photons(eta):
    """Uniform loss is applied on the input side by grouping the shots (docs/LOSS.md): every
    sample keeps at most n photons, and eta = 1 is the lossless path."""
    occ = fock_sample(haar_unitary(M, 11), IN_MODES, SHOTS, backend="triton", loss=eta,
                      generator=torch.Generator().manual_seed(0))
    assert occ.shape == (SHOTS, M)
    counts = occ.sum(1)
    assert bool((counts <= N).all()) and bool((counts >= 0).all())
    if eta == 1.0:
        assert counts.tolist() == [N] * SHOTS


def test_sample_dispatches_a_fock_state_to_the_fock_sampler():
    """`clifford.sample` on a FockState must equal `fock_sample` on the same modes and seed."""
    U = haar_unitary(M, 12)
    occ = [1, 1, 0, 0]
    a = sample(FockState(occ), U, SHOTS, backend="triton", generator=torch.Generator().manual_seed(3))
    b = fock_sample(U, IN_MODES, SHOTS, backend="triton", generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


@pytest.mark.parametrize(("kw", "match"), [({"backend": "nope"}, "unknown backend"),
                                           ({"backend": "torch"}, "unknown backend"),
                                           ({"algo": "2019"}, "unknown algo")])
def test_an_unknown_backend_or_algo_is_rejected(kw, match):
    """`backend="torch"` is valid on the emitter sampler but not here, and used to fall through
    to the triton path silently."""
    with pytest.raises(ValueError, match=match):
        fock_sample(haar_unitary(M, 13), IN_MODES, SHOTS, generator=torch.Generator().manual_seed(0), **kw)
