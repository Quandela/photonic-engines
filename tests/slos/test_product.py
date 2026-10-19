"""ProductSuperposition inputs: the graded SLOS kernel, the single-amplitude permanent, loss.

Four oracles, none of which shares code with `ProductSLOSKernel`:
  (a) `slos.reference.product_reference_probs` — explicit components x explicit permanents;
  (b) Ryser permanents of the homogenised matrix (the constant row repeated n - k times) for the
      Glynn path of `slos.glynn`;
  (c) Perceval's `Simulator(SLOSBackend)` on the equivalent StateVector;
  (d) `loss.thin_patterns` (pattern-list thinning) for the graded loss path.
"""
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import ProductSuperposition, graded_state_space_size
from photonic_engines.imperfections import thin_graded, thin_patterns
from photonic_engines.slos.basis import get_layers
from photonic_engines.slos.glynn import product_amplitude, product_probability
from photonic_engines.slos.kernel import FockSLOSKernel, ProductSLOSKernel
from photonic_engines.slos.reference import _layer_occupations, permanent, product_reference_probs

EPS = np.finfo(np.float64).eps


def _kernel(state, **kw):
    return ProductSLOSKernel(state, device="cpu", dtype=torch.complex128, **kw)


def _d3_state(m=4, modes=(0, 2)):
    """Two sources that may emit 0, 1 or 2 photons — the d > 2 arm of the recurrence."""
    amps = np.array([[0.6, 0.6, 0.4], [0.8, 0.5, 0.2]], dtype=complex)
    return ProductSuperposition(amps / np.linalg.norm(amps, axis=1, keepdims=True), modes, m)


# ---- the state type ------------------------------------------------------------------------

def test_state_validation():
    with pytest.raises(ValueError, match="distinct input mode"):
        ProductSuperposition(np.ones((2, 2)), (1, 1), 4)
    with pytest.raises(ValueError, match="out of range"):
        ProductSuperposition(np.ones((2, 2)), (0, 4), 4)
    with pytest.raises(ValueError, match="one mode per source"):
        ProductSuperposition(np.ones((2, 2)), (0,), 4)
    with pytest.raises(ValueError, match=r"shape \[sources, d\]"):
        ProductSuperposition(np.ones(2), (0,), 4)
    with pytest.raises(ValueError, match="at least one source"):
        ProductSuperposition.sbs(0, 4)


def test_norm_and_normalized():
    st = ProductSuperposition(np.array([[1.0, 1.0], [2.0, 0.0]]), (0, 1), 3)
    assert abs(st.norm() - math.sqrt(2) * 2) < 8 * EPS * 4
    unit = st.normalized()
    assert abs(unit.norm() - 1) < 8 * EPS
    assert ProductSuperposition.sbs(5, 10).norm() == pytest.approx(1.0, abs=8 * EPS * 5)
    with pytest.raises(ValueError, match="zero norm"):
        ProductSuperposition(np.zeros((1, 2)), (0,), 2).normalized()


@pytest.mark.parametrize("alpha", [0.3, math.pi / 4, 1.2])
def test_photon_number_law_is_binomial_and_matches_blocks(alpha):
    st = ProductSuperposition.sbs(5, 10, alpha)
    p = st.photon_number_probs()
    n, s2 = st.sources, math.sin(alpha) ** 2
    binom = [math.comb(n, k) * s2 ** k * (1 - s2) ** (n - k) for k in range(n + 1)]
    assert np.allclose(p, binom, rtol=0, atol=64 * EPS)
    blocks = st.blocks()
    assert sorted(blocks) == list(range(n + 1))
    for k, b in blocks.items():                      # the block norm^2 IS P(k)
        assert math.comb(n, k) == b.K
        assert abs(b.norm() ** 2 - p[k]) < 64 * EPS


def test_blocks_selects_one_photon_number():
    st = ProductSuperposition.sbs(4, 8)
    assert st.blocks(photons=2).K == 6
    assert st.blocks(photons=9) is None
    st0 = ProductSuperposition(np.array([[1.0, 0.0], [1.0, 0.0]]), (0, 1), 2)
    assert list(st0.blocks()) == [0]                 # zero amplitudes are dropped, not stored


def test_graded_space_size():
    n, m = 4, 8
    assert graded_state_space_size(n, m) == math.comb(m + n, n)
    assert graded_state_space_size(n, m) == sum(math.comb(m + k - 1, k) for k in range(n + 1))
    assert graded_state_space_size(3, 6, "unbunched") == sum(math.comb(6, k) for k in range(4))
    with pytest.raises(ValueError, match="no graded space"):
        graded_state_space_size(3, 6, "dual_rail")


# ---- the graded kernel vs the permanent reference --------------------------------------------

@pytest.mark.parametrize(("sources", "m", "alpha"), [(3, 5, math.pi / 4), (4, 6, 0.7), (2, 3, 0.0)])
def test_kernel_matches_permanent_reference(sources, m, alpha):
    st = ProductSuperposition.sbs(sources, m, alpha)
    U = haar_unitary(m, seed=7)
    got, ref = _kernel(st).probs(U), product_reference_probs(U, st)
    for k in range(st.n + 1):
        assert torch.allclose(got.probs[k], ref[k], rtol=0, atol=1024 * EPS), k


def test_kernel_matches_permanent_reference_bunched_sources():
    st = _d3_state()
    U = haar_unitary(st.m, seed=3)
    got, ref = _kernel(st).probs(U), product_reference_probs(U, st)
    assert st.n == 4 and max(int(o.max()) for o in [st.blocks(4).occupations]) == 2
    for k in range(st.n + 1):
        assert torch.allclose(got.probs[k], ref[k], rtol=0, atol=1024 * EPS), k


@pytest.mark.parametrize("alpha", [0.4, math.pi / 4])
def test_masses_are_the_photon_number_law(alpha):
    st = ProductSuperposition.sbs(4, 7, alpha)
    ker = _kernel(st)
    dist = ker.probs(haar_unitary(7, seed=13))
    masses = [float(dist.mass(k)) for k in range(st.n + 1)]
    assert np.allclose(masses, st.photon_number_probs(), rtol=0, atol=1024 * EPS)
    assert abs(sum(masses) - 1) < 1024 * EPS
    assert ker.output_size() == sum(ker.block_sizes()) == graded_state_space_size(st.n, st.m)


def test_certain_emission_is_exactly_the_fock_kernel():
    """a[i] = (0, 1) leaves one photon per source with certainty. The zero coefficient is skipped,
    so the recurrence becomes the Fock one term for term: BIT-IDENTICAL top layer, empty rest.
    """
    sources, m = 4, 8
    st = ProductSuperposition(np.tile(np.array([0.0, 1.0]), (sources, 1)), tuple(range(sources)), m)
    U = haar_unitary(m, seed=17)
    dist = _kernel(st).probs(U)
    occ = [1] * sources + [0] * (m - sources)
    fock = FockSLOSKernel(sources, m, occ, device="cpu", dtype=torch.complex128).probs(U)
    assert torch.equal(dist.probs[sources], fock)
    for k in range(sources):
        assert float(dist.probs[k].abs().max()) == 0.0


def test_alpha_pi_over_2_is_the_fock_distribution():
    """The same at alpha = pi/2, where cos(alpha) is 6e-17 rather than 0: the top layer is the Fock
    distribution and the lower layers carry only that leakage, |c|^2 = cos(alpha)^2 per missing photon.
    """
    sources, m = 4, 8
    st = ProductSuperposition.sbs(sources, m, math.pi / 2)
    U = haar_unitary(m, seed=17)
    dist = _kernel(st).probs(U)
    occ = [1] * sources + [0] * (m - sources)
    fock = FockSLOSKernel(sources, m, occ, device="cpu", dtype=torch.complex128).probs(U)
    assert torch.allclose(dist.probs[sources], fock, rtol=0, atol=1024 * EPS)
    leak = math.cos(math.pi / 2) ** 2                # 3.7e-33, the binomial weight per missing photon
    for k in range(sources):
        assert float(dist.probs[k].sum()) <= 2 * math.comb(sources, k) * leak ** (sources - k)


def test_batched_unitaries_agree_with_single():
    st = ProductSuperposition.sbs(3, 6)
    ker = _kernel(st)
    Us = torch.stack([haar_unitary(6, seed=s) for s in (1, 2)])
    batched = ker.probs(Us)
    for b, seed in enumerate((1, 2)):
        single = ker.probs(haar_unitary(6, seed=seed))
        for k in range(st.n + 1):
            assert torch.allclose(batched.probs[k][b], single.probs[k], rtol=0, atol=64 * EPS), (b, k)


def test_kernel_rejects_empty_and_ungraded():
    with pytest.raises(ValueError, match="nothing to propagate"):
        _kernel(ProductSuperposition(np.array([[1.0]]), (0,), 2))
    with pytest.raises(ValueError, match="no graded space"):
        _kernel(ProductSuperposition.sbs(2, 4), mask="dual_rail")


# ---- one amplitude as one permanent ----------------------------------------------------------

def _ryser_amplitude(state, U, t):
    """amp(t) from per(M) with M written out and the permanent taken by Ryser — the oracle for
    the collapsed Glynn sum of `slos.glynn` (same formula, no sign enumeration).
    """
    n, k = state.sources, sum(t)
    g = n - k
    if g < 0:
        return 0j
    rows = [j for j, x in enumerate(t) for _ in range(x)]
    c, s = state.amps[:, 0], state.amps[:, 1]
    M = torch.zeros((n, n), dtype=torch.complex128)
    for r, j in enumerate(rows):
        M[r] = torch.as_tensor(np.array(s)) * torch.as_tensor([U[j, cc] for cc in state.modes])
    for r in range(k, n):
        M[r] = torch.as_tensor(np.array(c))
    return complex(permanent(M)) / (math.factorial(g) * math.sqrt(math.prod(math.factorial(x) for x in t)))


@pytest.mark.parametrize(("sources", "m", "alpha"), [(4, 6, math.pi / 4), (5, 5, 0.6), (3, 4, math.pi / 2)])
def test_glynn_matches_ryser_and_the_kernel(sources, m, alpha):
    st = ProductSuperposition.sbs(sources, m, alpha)
    U = haar_unitary(m, seed=5)
    layers = get_layers(st.n, st.m)
    amps = _kernel(st).amplitudes(U)
    for k in range(st.n + 1):
        norms = torch.as_tensor(layers.prob_norms(k), dtype=torch.float64).sqrt()
        for i, t in enumerate(_layer_occupations(layers, k)):
            got = complex(product_amplitude(st, U, t))
            assert abs(got - _ryser_amplitude(st, U, t)) < 4096 * EPS
            # the kernel defers prod_j t_j!: |raw| * sqrt(that) is the physical magnitude
            assert abs(abs(got) - abs(complex(amps[k][i]) * float(norms[i]))) < 4096 * EPS
            assert abs(float(product_probability(st, U, t)) - abs(got) ** 2) < 4096 * EPS


@pytest.mark.parametrize(("sources", "k"), [(8, 4), (24, 12),
                                            pytest.param(40, 20, marks=pytest.mark.slow)])
def test_glynn_is_exact_where_the_amplitude_is_analytic(sources, k):
    """U = identity: the amplitude is cos^(n-k) sin^k, exactly, at ANY size — the only oracle that
    reaches the sizes the bench runs at. This is the regression test for the numerics, not for the
    algebra: the first version of `slos.glynn` was exact in exact arithmetic (it agrees with Ryser
    below, and still does) yet lost every digit by n = 32, because collapsing the repeated constant
    row multiplies each summand by (n - k - 2v) while the answer stays tiny (LOG 2026-09-19).
    """
    m = 2 * sources
    st = ProductSuperposition.sbs(sources, m)
    occ = [1] * k + [0] * (m - k)
    got = complex(product_amplitude(st, torch.eye(m, dtype=torch.complex128), occ))
    exact = math.cos(math.pi / 4) ** (sources - k) * math.sin(math.pi / 4) ** k
    assert abs(got - exact) / exact < 64 * EPS, (sources, k, got, exact)


def test_glynn_amplitude_stays_below_one_at_scale():
    """A normalised state cannot have |amp| > 1. The broken version returned 6.2e8 at n = 48 on the
    GPU run of 2026-09-19, which is what this cheap guard would have caught.
    """
    st = ProductSuperposition.sbs(24, 48)
    occ = [1] * 12 + [0] * 36
    assert float(product_amplitude(st, haar_unitary(48, seed=4), occ).abs()) < 1.0


def test_glynn_edge_cases():
    st = ProductSuperposition.sbs(3, 4, 0.5)
    U = haar_unitary(4, seed=2)
    assert float(product_amplitude(st, U, [4, 0, 0, 0]).abs()) == 0.0        # k > sources
    vac = complex(product_amplitude(st, U, [0, 0, 0, 0]))
    assert abs(vac - math.cos(0.5) ** 3) < 64 * EPS                          # prod_i c_i
    assert abs(complex(product_amplitude(st, U, [1, 0, 0, 0], chunk=1))
               - complex(product_amplitude(st, U, [1, 0, 0, 0]))) < 64 * EPS  # chunking is exact
    with pytest.raises(ValueError, match="d = 2"):
        product_amplitude(_d3_state(), U, [1, 0, 0, 0])
    with pytest.raises(ValueError, match="non-negative ints"):
        product_amplitude(st, U, [1, 0, 0])


# ---- loss --------------------------------------------------------------------------------------

def test_loss_eta_one_is_exactly_lossless():
    st = ProductSuperposition.sbs(4, 6)
    ker = _kernel(st)
    U = haar_unitary(6, seed=11)
    base, one = ker.probs(U), ker.probs(U, loss=1.0)
    assert all(torch.equal(a, b) for a, b in zip(one.probs, base.probs))
    assert all(torch.equal(a, b) for a, b in zip(thin_graded(ker.layers, base.probs, None).probs, base.probs))


@pytest.mark.parametrize("eta", [0.85, 0.5])
def test_graded_loss_matches_pattern_thinning(eta):
    st = ProductSuperposition.sbs(4, 6)
    ker = _kernel(st)
    base = ker.probs(haar_unitary(6, seed=11))
    occ, p = base.flat()
    got = ker.probs(haar_unitary(6, seed=11), loss=eta).as_dict()
    ref = thin_patterns(occ, p, eta).as_dict()
    assert abs(sum(got.values()) - 1) < 4096 * EPS
    for key in set(got) | set(ref):
        assert abs(got.get(key, 0.0) - ref.get(key, 0.0)) < 4096 * EPS, key


# ---- sampling ------------------------------------------------------------------------------

def test_sample_matches_the_computed_distribution():
    st = ProductSuperposition.sbs(3, 6)
    ker = _kernel(st)
    U = haar_unitary(6, seed=23)
    ref = ker.probs(U).as_dict()
    shots = 20000
    occ = ker.sample(U, shots, generator=torch.Generator().manual_seed(0))
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / shots
    tvd = 0.5 * sum(abs(emp.get(k, 0.0) - ref.get(k, 0.0)) for k in set(emp) | set(ref))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in ref.values())
    assert tvd < 2.5 * floor, (tvd, floor)
    assert occ.shape == (shots, 6)


def test_sampling_survives_more_than_2_24_states(monkeypatch):
    """`torch.multinomial` caps the categories at 2^24, which a graded draw passes at n = 10
    (30 M states at m = 2n). `loss.draw_indices` falls back to the inverse CDF above the cap; the
    cap is lowered here so the fallback runs on a basis that fits in a test.
    """
    from photonic_engines import imperfections as loss_mod
    st = ProductSuperposition.sbs(3, 6)
    ker = _kernel(st)
    U = haar_unitary(6, seed=23)
    dist = ker.probs(U)
    ref = dist.as_dict()
    monkeypatch.setattr(loss_mod, "MULTINOMIAL_MAX", 4)        # forces the searchsorted path
    shots = 20000
    occ = dist.sample(shots, generator=torch.Generator().manual_seed(1))
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / shots
    tvd = 0.5 * sum(abs(emp.get(k, 0.0) - ref.get(k, 0.0)) for k in set(emp) | set(ref))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in ref.values())
    assert tvd < 2.5 * floor, (tvd, floor)


@pytest.mark.parametrize(("sources", "k", "j"), [(4, 2, 1), (5, 3, 1), (5, 3, 2)])
def test_marginal_has_the_residual_set_form(sources, k, j):
    r"""The j-photon marginal of a Dicke (post-selected SBS) input is, up to one global constant,

        p_j(r) = sum over residual sets R of | sum_{S subset [n]\R, |S| = j} Per(U[r, S]) |^2 ,

    and the inner sum is exactly what `slos.glynn` evaluates in O(2^j n j) — no subset enumeration.
    That is the structure a chain-rule sampler would march along (docs/SUPERPOSITION.md §5), so it
    is pinned here against marginals taken by brute force from the computed distribution.
    """
    from itertools import combinations
    m = 2 * sources
    st = ProductSuperposition.sbs(sources, m)
    U = haar_unitary(m, seed=9)
    dist = _kernel(st).probs(U)
    layers = get_layers(st.n, m)
    occ_k, p_k = _layer_occupations(layers, k), dist.conditional(k).numpy()

    def exact(rows):                                 # ordered j-tuple drawn from a k-photon pattern
        tot = 0.0
        for y, p in zip(occ_k, p_k):
            ways, avail = 1.0, list(y)
            for r in rows:
                ways *= avail[r]; avail[r] -= 1
            tot += p * ways
        return tot / math.prod(k - i for i in range(j))

    def residual(rows):
        tot = 0.0
        for R in combinations(range(sources), k - j):
            keep = [i for i in range(sources) if i not in R]
            amp = sum(complex(permanent(U[list(rows)][:, list(S)])) for S in combinations(keep, j))
            tot += abs(amp) ** 2
        return tot

    rows_list = list(combinations(range(m), j))
    ex = np.array([exact(r) for r in rows_list])
    ca = np.array([residual(r) for r in rows_list])
    assert np.abs(ex - ex.sum() / ca.sum() * ca).max() < 4096 * EPS


def test_clifford_sample_dispatch():
    """ProductSuperposition now has a chain-rule sampler (tests/clifford/test_product_sampler.py);
    a general FockSuperposition still has none, and must say so rather than return the mixture.
    """
    from photonic_engines import FockSuperposition, clifford
    st = ProductSuperposition.sbs(3, 6)
    got = clifford.sample(st, haar_unitary(6, seed=1), 4, generator=torch.Generator().manual_seed(0))
    assert got.shape == (4, 6)
    with pytest.raises(NotImplementedError, match="coherent superposition"):
        clifford.sample(FockSuperposition(((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0)), (1, 1j)),
                        haar_unitary(6, seed=1), 4)


# ---- the sampler gate ------------------------------------------------------------------------

def test_residual_gram_rank_closed_form():
    """docs/SUPERPOSITION.md §5 and the LOG quote R_j = C(n, min(j, k-j)) as the exact pure-term
    count of a chain-rule sampler. That is a measured claim, so it is guarded here (n <= 7 for
    speed; `python3 bench/sbs_terms.py --check` runs it to n = 12).
    """
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "sbs_terms.py")
    spec = importlib.util.spec_from_file_location("sbs_terms", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check(limit=7, max_dim=200) == 0
    assert mod.predicted_rank(40, 20, 10) == math.comb(40, 10)


# ---- Perceval ------------------------------------------------------------------------------

def test_matches_perceval_simulator():
    pcvl = pytest.importorskip("perceval")
    from perceval.backends import SLOSBackend
    from perceval.simulators import Simulator
    st = ProductSuperposition.sbs(3, 5, 0.6)
    U = haar_unitary(5, seed=31)
    circ = pcvl.Unitary(pcvl.Matrix(U.numpy()))
    sim = Simulator(SLOSBackend()); sim.set_circuit(circ)
    ref = sim.probs(st.to_perceval())
    got = _kernel(st).probs(U).as_dict()
    for state, v in ref.items():
        assert abs(got.get(tuple(int(x) for x in state), 0.0) - v) < 1e-10, state
