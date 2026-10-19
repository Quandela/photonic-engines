"""Partial distinguishability (docs/DISTINGUISHABILITY.md): the model, the oracles and the two
samplers. Every check is against an oracle that knows nothing about species (`gram_probs`, the
double permutation sum) or against the species-resolved permanent sum (`species_probs`), with
tolerances derived from the dtype (identities) or from shot noise (samplers).
"""
import itertools
import math

import pytest
import torch
from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.imperfections import (
    SINGLE,
    Distinguishability,
    draw_roles,
    partition,
    route_classical,
    spec,
    subset_weights,
)
from photonic_engines.slos.basis import get_layers
from photonic_engines.slos.reference import gram_probs, reference_probs, species_probs

C128 = torch.complex128


def _empirical(occ):
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / occ.shape[0]
    return emp


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _noise_floor(p, shots):
    """Expected TVD of an empirical distribution: sum_i sqrt(p_i (1-p_i) / shots) sqrt(2/pi) / 2."""
    return 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in p.values())


def _x_mixture(U, comps, n, x):
    """The x-model output distribution as the mixture over the interfering subset (docs §1)."""
    out = {}
    for G, w in subset_weights(n, x):
        sp = [0 if j in G else 10 + j for j in range(n)]
        for t, p in species_probs(U, comps, sp).items():
            out[t] = out.get(t, 0.0) + w * p
    return out


# ---- the spec ----------------------------------------------------------------------------------

def test_spec_and_visibility():
    assert spec(None) is None and spec(1.0) is None and spec(Distinguishability(x=1)) is None
    assert spec(0.8).x == 0.8
    assert abs(Distinguishability.from_hom_visibility(0.81).x - 0.9) < 1e-15
    assert spec(Distinguishability(species=(1.0,))) is None                  # one species of probability 1: ideal
    assert spec(Distinguishability(species=(0.5, 0.5))) is not None
    for bad in ({"x": 1.2}, {"x": 0.5, "species": (0.5,)}, {}, {"species": (0.7, 0.4)}):
        with pytest.raises(ValueError):
            Distinguishability(**bad)
    with pytest.raises(ValueError):
        Distinguishability.from_hom_visibility(1.5)
    assert spec([1.0, 0.9]).x == 0.9 and spec([1.0, 1.0]) is None              # per-photon overlaps
    with pytest.raises(ValueError):
        spec([0.5, 1.2])
    import photonic_engines.imperfections as imp
    # both names are part of the public surface
    assert {"spec", "Distinguishability"} <= set(imp.__all__)


def test_draw_roles_law():
    """Lost w.p. 1 - eta, else species mu w.p. q_mu, else a singleton — independently per photon."""
    n, shots, eta = 5, 40000, 0.8
    g = torch.Generator().manual_seed(0)
    roles = draw_roles(n, shots, eta, Distinguishability(species=(0.5, 0.3)), generator=g)
    assert roles.shape == (shots, n)
    freq = torch.bincount(roles.reshape(-1), minlength=4).double() / (shots * n)
    expect = torch.tensor([1 - eta, eta * 0.2, eta * 0.5, eta * 0.3], dtype=torch.float64)
    assert (freq - expect).abs().max() < 4 * math.sqrt(0.25 / (shots * n))
    roles = draw_roles(n, shots, None, 0.6, generator=g)                      # x-model, no loss
    assert set(roles.unique().tolist()) <= {SINGLE, 2}
    assert abs((roles == 2).double().mean() - 0.6) < 4 * math.sqrt(0.25 / (shots * n))
    assert (draw_roles(n, 7, None, None) == 2).all() and (draw_roles(n, 7, 1.0, 1.0) == 2).all()


def test_partition_groups_are_a_partition_of_the_streams():
    n, shots = 4, 500
    groups = partition(n, shots, 0.7, 0.5, generator=torch.Generator().manual_seed(3))
    seen = torch.cat([ids for _, _, ids in groups])
    assert seen.sort().values.tolist() == list(range(shots))
    for subsets, singles, _ in groups:
        photons = [j for s in subsets for j in s] + list(singles)
        assert len(photons) == len(set(photons)) and all(0 <= j < n for j in photons)
        assert all(len(s) >= 1 for s in subsets)
    ideal = partition(n, shots, None, None)
    assert len(ideal) == 1 and ideal[0][0] == (tuple(range(n)),) and ideal[0][1] == ()


def test_route_classical_law():
    U = haar_unitary(5, 2)
    P = 20000
    cols = torch.full((P,), 3)
    rows = route_classical(U, torch.zeros(P, dtype=torch.long), cols, generator=torch.Generator().manual_seed(0))
    freq = torch.bincount(rows, minlength=5).double() / P
    p = (U[:, 3].abs() ** 2).double()
    assert (freq - p).abs().max() < 4 * math.sqrt(0.25 / P)
    assert route_classical(U, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)).shape == (0,)


# ---- the fold ----------------------------------------------------------------------------------

def test_fold_generalises_the_lossy_label_fold():
    """`EmitterMPS.fold` with label operators folds lost photons at their labels; with a row
    operator it is the chain of a distinguishable photon at its drawn row: both checked against
    the explicit product of operators."""
    n = 5
    st = EmitterMPS.caterpillar(n, [spin_gate(0.4 * i + 0.3, 0.9 * i) for i in range(n)], m=2 * n)
    red2, tail2 = st.fold({0: st.ops[0, 0], 2: st.ops[2, 1], 4: st.ops[4, 1]})
    assert red2.modes == (st.modes[1], st.modes[3]) and red2.n == 2
    # explicit: <f| tail M~_{3,x3} M~_{1,x1} |s0> == <f| M_{4,1} M_{3,x3} M_{2,1} M_{1,x1} M_{0,0} |s0>
    for x1, x3 in itertools.product(range(2), repeat=2):
        lhs = tail2 @ red2.ops[1, x3] @ red2.ops[0, x1]
        rhs = st.ops[4, 1] @ st.ops[3, x3] @ st.ops[2, 1] @ st.ops[1, x1] @ st.ops[0, 0]
        assert abs(lhs - rhs).max() < 1e-14
    U = haar_unitary(2 * n, 1).numpy()
    G = sum(U[7, st.modes[2][l]] * st.ops[2, l] for l in range(2))
    red3, tail3 = st.fold({2: G})
    assert red3.n == 4 and abs(tail3 - torch.eye(2).numpy()).max() == 0
    assert abs(red3.ops[2, 0] - st.ops[3, 0] @ G).max() < 1e-14
    none, tail4 = st.fold({j: st.ops[j, 0] for j in range(n)})
    assert none is None and abs(tail4 - st.ops[4, 0] @ st.ops[3, 0] @ st.ops[2, 0] @ st.ops[1, 0] @ st.ops[0, 0]).max() < 1e-14


# ---- the oracles and the model -----------------------------------------------------------------

def test_species_oracle_reduces_to_permanents():
    """One species: the Fock permanents. All distinct: classical transport per(|U|^2)/prod t!."""
    U = haar_unitary(5, 1)
    cols = [0, 2, 3]
    layers = get_layers(3, 5, "fock")
    ref = reference_probs(U, 3, 5, [1, 0, 1, 1, 0])
    one = species_probs(U, [(1.0, cols)], [0, 0, 0])
    assert max(abs(one[tuple(t)] - float(ref[i])) for i, t in enumerate(layers.final_states)) < 1e-14
    from photonic_engines.slos.reference import permanent
    A = (U.abs() ** 2).to(C128)
    cl = species_probs(U, [(1.0, cols)], [0, 1, 2])
    for t, p in cl.items():
        rows = [j for j, c in enumerate(t) for _ in range(c)]
        q = float(permanent(A[rows][:, cols]).real) / math.prod(math.factorial(c) for c in t)
        assert abs(p - q) < 1e-14
    assert abs(sum(one.values()) - 1) < 1e-13 and abs(sum(cl.values()) - 1) < 1e-13


def test_x_model_equals_the_bad_photon_mixture():
    """The claim the whole design rests on: the Gram matrix S = x J + (1 - x) I (pure internal
    states with pairwise overlap x) gives, for every output, exactly the mixture over which
    photons interfere with weights x^|G| (1 - x)^(n - |G|) — so the exact sampler needs no
    approximation. Checked at n = 3 and n = 4 to the dtype (the (n!)^2 sum is complex128)."""
    for n, m, seed, x in ((3, 5, 1, 0.7), (4, 5, 2, 0.35)):
        U = haar_unitary(m, seed)
        cols = list(range(n))
        S = x * torch.ones(n, n, dtype=torch.float64) + (1 - x) * torch.eye(n, dtype=torch.float64)
        gram = gram_probs(U, cols, S)
        mix = _x_mixture(U, [(1.0, cols)], n, x)
        assert max(abs(gram[k] - mix[k]) for k in gram) < 1e-13
        assert abs(sum(gram.values()) - 1) < 1e-12


def test_random_species_cycle_weights():
    """Random species q: the weight of a permutation is prod over cycles of sum_mu q_mu^|c|,
    which is the Gram-matrix formula with S = Tr(rho_i rho_j)... only for 2-cycles; the
    3-cycle term distinguishes the two. Checked against a direct (n!)^2-type sum with cycle
    weights on n = 3."""
    from itertools import permutations
    n, m = 3, 4
    U = haar_unitary(m, 4)
    cols = [0, 1, 2]
    q = (0.5, 0.3)                                                                     # remainder 0.2: own species
    # species-mixture distribution: enumerate species assignments (own species = distinct labels)
    mix = {}
    for assign in itertools.product(range(3), repeat=n):
        w = 1.0
        sp = []
        for j, a in enumerate(assign):
            w *= (*q, 0.2)[a]
            sp.append(a if a < 2 else 10 + j)
        for t, p in species_probs(U, [(1.0, cols)], sp).items():
            mix[t] = mix.get(t, 0.0) + w * p
    # cycle-weight formula: P(t) = (1/t!) sum_{sigma, tau} w(tau sigma^-1) prod_i U[r_i, c_sigma(i)] conj U[r_i, c_tau(i)]
    def cycle_weight(pi):
        seen, w = set(), 1.0
        for i in range(n):
            if i in seen:
                continue
            length, j = 0, i
            while j not in seen:
                seen.add(j); j = pi[j]; length += 1
            w *= sum(v ** length for v in q) + (0.2 if length == 1 else 0.0)
        return w
    from photonic_engines.slos.reference import _modes_from_occupation, _multisets
    for t in _multisets(m, n):
        rows = _modes_from_occupation(t)
        acc = 0j
        for sg in permutations(range(n)):
            for tu in permutations(range(n)):
                inv = [0] * n
                for i in range(n):
                    inv[sg[i]] = i
                pi = tuple(tu[inv[j]] for j in range(n))                                # tau o sigma^-1
                a = 1.0 + 0j
                for i in range(n):
                    a *= complex(U[rows[i], cols[sg[i]]]) * complex(U[rows[i], cols[tu[i]]].conj())
                acc += cycle_weight(pi) * a
        p = acc.real / math.prod(math.factorial(c) for c in t)
        assert abs(p - mix[t]) < 1e-13


# ---- Fock sampler --------------------------------------------------------------------------------

def test_fock_sample_fully_distinguishable_is_classical_transport():
    """x = 0: every photon is routed on the host (no kernel launch), and the distribution is
    per(|U|^2)/t! — checked at 2 photons in the same input mode too (bunching without
    interference)."""
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    m, shots = 4, 6000
    U = haar_unitary(m, 7)
    for cols in ([0, 1, 2], [1, 1]):
        occ = fock_sample(U, cols, shots, indist=0.0, generator=torch.Generator().manual_seed(0))
        assert occ.shape == (shots, m) and (occ.sum(1) == len(cols)).all()
        ref = species_probs(U, [(1.0, cols)], list(range(len(cols))))
        assert _tvd(_empirical(occ), ref) < 2.5 * _noise_floor(ref, shots)


def test_fock_sample_indist_dispatch():
    """Ideal x is the lossless path, a spec is accepted, the exq backend refuses, loss composes."""
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    U = haar_unitary(4, 8)
    g = lambda: torch.Generator().manual_seed(5)
    assert torch.equal(fock_sample(U, [0, 1], 6, generator=g()), fock_sample(U, [0, 1], 6, indist=1.0, generator=g()))
    occ = fock_sample(U, [0, 1], 6, indist=Distinguishability(species=(0.5, 0.4)), loss=0.8, generator=g())
    assert occ.shape == (6, 4) and (occ.sum(1) <= 2).all()
    with pytest.raises(NotImplementedError):
        fock_sample(U, [0, 1], 2, backend="exq", indist=0.5)


@pytest.mark.slow
def test_fock_sample_indist_matches_the_mixture():
    """The Triton kernels under the CPU interpreter: x-model samples against the exact mixture."""
    pytest.importorskip("triton")
    from photonic_engines.clifford import fock_sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cols, x, shots = 4, [0, 1, 2], 0.6, 3000
    U = haar_unitary(m, 4)
    ref = _x_mixture(U, [(1.0, cols)], 3, x)
    s = fock_sample(U.to(dev), cols, shots, generator=torch.Generator(device=dev).manual_seed(0), indist=x).cpu()
    assert (s.sum(1) == 3).all()
    assert _tvd(_empirical(s), ref) < 2.5 * _noise_floor(ref, shots)


# ---- emitter sampler -----------------------------------------------------------------------------

def _caterpillar(n, m):
    gates = [spin_gate(math.pi / 2, 0), *[spin_gate(math.pi / 2, p) for p in (0.3, 1.1, 2.0, 0.8)][:n - 1]]
    return EmitterMPS.caterpillar(n, gates, m=m)


def _emitter_components(st):
    """The chain expanded over its label words, per final basis state: [(amp, cols)] lists."""
    from photonic_engines.clifford.emitter_reference import coefficient_tensor
    c = coefficient_tensor(st)
    out = []
    for f in range(st.chi):
        comps = []
        for xs in itertools.product(range(st.num_labels), repeat=st.n):
            a = complex(c[(*xs, f)])
            if abs(a) > 1e-14:
                comps.append((a, [st.modes[k][xs[k]] for k in range(st.n)]))
        out.append(comps)
    return out


def _emitter_x_mixture(st, U, x):
    ref = {}
    for comps in _emitter_components(st):
        for t, p in _x_mixture(U, comps, st.n, x).items():
            ref[t] = ref.get(t, 0.0) + p
    return ref


def test_singleton_rows_are_drawn_from_the_exact_marginal():
    """All slots singletons: no chain rule runs (k = 0), the rows come from
    `sample_slot_outcomes` alone, and their distribution is the fully distinguishable one."""
    from photonic_engines.clifford import emitter_sample
    n, m, shots = 3, 6, 20000
    st = _caterpillar(n, m)
    U = haar_unitary(m, 9)
    occ = emitter_sample(st, U, shots, backend="torch", indist=0.0, generator=torch.Generator().manual_seed(1))
    assert (occ.sum(1) == n).all()
    ref = {}
    for comps in _emitter_components(st):
        for t, p in species_probs(U, comps, [10 + j for j in range(n)]).items():
            ref[t] = ref.get(t, 0.0) + p
    assert abs(sum(ref.values()) - 1) < 1e-12
    assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)


def test_row_folded_gap_terms_have_one_term_and_the_right_chain():
    """A singleton slot at its row gives a rank-1 gap (R = 1 per stream) in `batched_gap_terms`,
    and the batched chain equals the reference `gap_kraus_terms` chain with the same labels."""
    from photonic_engines.clifford.emitter_reference import gap_kraus_terms
    from photonic_engines.clifford.emitter_terms import batched_gap_terms, sample_slot_outcomes
    n, m = 5, 10
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] + [spin_gate(0.7 * i + 0.3, 1.1 * i) for i in range(1, n)], m=m)
    U = haar_unitary(m, 2)
    S = 8
    roles = torch.tensor([[1, 2, 0, 2, 1]] * S)                                          # single, good, lost, good, single
    labels, fid = sample_slot_outcomes(st, roles, U, generator=torch.Generator().manual_seed(0))
    assert (labels[:, [1, 3]] == -1).all() and (labels[:, 2] < 2).all()
    assert (labels[:, [0, 4]] >= 2).all() and (labels[:, [0, 4]] < 2 + m).all()
    finals = torch.eye(2, dtype=C128)[fid]
    keep = torch.tensor([[1, 3]] * S)
    s0, mats, f, t2s = batched_gap_terms(st, keep, finals, labels=labels, U=U)
    assert torch.bincount(t2s, minlength=S).tolist() == [1] * S
    for s_ in range(S):
        _, terms = gap_kraus_terms(st, [1, 3], final=finals[s_], labels=labels[s_].tolist(), U=U)
        assert len(terms) == 1
        def amps(s0v, mm, fv):
            out = []
            for xs in itertools.product(range(2), repeat=2):
                v = s0v.clone()
                for j, l in enumerate(xs):
                    v = mm[j, l] @ v
                out.append(fv.conj() @ v)
            return torch.stack(out)
        assert torch.allclose(amps(*terms[0]).abs() ** 2, amps(s0[s_], mats[s_], f[s_]).abs() ** 2, atol=1e-12)
        # the folded chain is the explicit product with G_j(r) = sum_l U[r, modes[j][l]] M_{j,l} at the singleton slots
        r0, r4, x2 = int(labels[s_, 0]) - 2, int(labels[s_, 4]) - 2, int(labels[s_, 2])
        G0 = sum(U[r0, st.modes[0][l]] * torch.as_tensor(st.ops[0, l]) for l in range(2))
        G4 = sum(U[r4, st.modes[4][l]] * torch.as_tensor(st.ops[4, l]) for l in range(2))
        refs, gots = [], []
        for x1, x3 in itertools.product(range(2), repeat=2):
            chain = G4 @ torch.as_tensor(st.ops[3, x3]) @ torch.as_tensor(st.ops[2, x2]) @ torch.as_tensor(st.ops[1, x1]) @ G0
            refs.append(abs(finals[s_].conj() @ chain @ torch.as_tensor(st.s0)))
            gots.append(abs(f[s_].conj() @ mats[s_, 1, x3] @ mats[s_, 0, x1] @ s0[s_]))
        refs, gots = torch.stack(refs), torch.stack(gots)                                  # equal up to a per-stream scalar
        assert torch.allclose(refs / refs.norm(), gots / gots.norm(), atol=1e-12)          # (the row weights are unit-norm)


@pytest.mark.parametrize("eta", [None, 0.7])
def test_emitter_sample_indist_matches_the_mixture(eta):
    """x-model on the torch backend, with and without loss, against the species-resolved
    permanent sum of the expanded chain (loss = thinning of that mixture, since both commute)."""
    from photonic_engines.clifford import emitter_sample
    n, m, x, shots = 4, 8, 0.6, 12000
    st = _caterpillar(n, m)
    U = haar_unitary(m, 3)
    ref = _emitter_x_mixture(st, U, x)
    if eta is not None:
        from photonic_engines.imperfections import thin_patterns
        ref = thin_patterns(torch.tensor(list(ref)), torch.tensor(list(ref.values()), dtype=torch.float64), eta).as_dict()
    assert abs(sum(ref.values()) - 1) < 1e-12
    occ = emitter_sample(st, U, shots, backend="torch", indist=x, loss=eta, generator=torch.Generator().manual_seed(2))
    assert occ.shape == (shots, m) and (occ.sum(1) <= n).all()
    assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)


def test_emitter_sample_indist_dispatch():
    from photonic_engines.clifford import emitter_sample
    st = _caterpillar(3, 6)
    U = haar_unitary(6, 1)
    g = lambda: torch.Generator().manual_seed(4)
    assert torch.equal(emitter_sample(st, U, 5, backend="torch", generator=g()),
                       emitter_sample(st, U, 5, backend="torch", indist=1.0, generator=g()))
    with pytest.raises(NotImplementedError):
        emitter_sample(st, U, 2, backend="torch", indist=Distinguishability(species=(0.5, 0.5)))
    with pytest.raises(NotImplementedError):
        emitter_sample(st, U, 2, backend="segments", indist=0.5)
    occ = emitter_sample(st, U, 40, backend="torch", indist=0.5, loss=0.6, unravel=False, generator=g())
    assert occ.shape == (40, 6) and (occ.sum(1) <= 3).all()


@pytest.mark.slow
def test_emitter_sample_indist_triton_interpreter():
    """The Triton emitter kernel with row-folded terms, under the CPU interpreter."""
    pytest.importorskip("triton")
    from photonic_engines.clifford import emitter_sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, m, x, shots = 3, 6, 0.6, 1500
    st = _caterpillar(n, m)
    U = haar_unitary(m, 5)
    ref = _emitter_x_mixture(st, U, x)
    occ = emitter_sample(st, U.to(dev), shots, backend="triton", device=dev, indist=x,
                         generator=torch.Generator().manual_seed(2)).cpu()
    assert (occ.sum(1) == n).all()
    assert _tvd(_empirical(occ), ref) < 2.5 * _noise_floor(ref, shots)


# ---- strong sides (phase 2) ----------------------------------------------------------------------

def _emitter_oracle(st, U, x):
    """x-model mixture of an emitter chain from its expanded components; projected states contract
    with `final` (the traced helper above sums the final basis instead)."""
    from photonic_engines.clifford.emitter_reference import coefficient_tensor
    c = coefficient_tensor(st)
    finals = [torch.as_tensor(st.final, dtype=C128)] if st.final is not None else list(torch.eye(st.chi, dtype=C128))
    ref = {}
    for f in finals:
        comps = []
        for xs in itertools.product(range(st.num_labels), repeat=st.n):
            a = complex(f.conj() @ c[xs])
            if abs(a) > 1e-14:
                comps.append((a, [st.modes[k][xs[k]] for k in range(st.n)]))
        for t, p in _x_mixture(U, comps, st.n, x).items():
            ref[t] = ref.get(t, 0.0) + p
    return ref


def _dict_err(layers, p, ref):
    return max(abs(float(p[i]) - ref.get(t, 0.0)) for i, t in enumerate(layers.final_states))


def test_fock_strong_x_mixture_equals_gram_oracle():
    """`FockSLOSKernel.probs(indist=x)` is the Gram-matrix distribution; batched in U;
    differentiable in x; `order=` drops exactly `truncation_bound`; loss composes (mass 1)."""
    from photonic_engines.imperfections import truncation_bound
    from photonic_engines.slos import FockSLOSKernel
    n, m, x = 4, 6, 0.7
    U = haar_unitary(m, 3)
    cols = [0, 1, 3, 4]
    occ = [1 if j in cols else 0 for j in range(m)]
    k = FockSLOSKernel(n, m, occ, dtype=C128)
    S = x * torch.ones(n, n, dtype=torch.float64) + (1 - x) * torch.eye(n, dtype=torch.float64)
    p = k.probs(U, indist=x)
    assert _dict_err(k.layers, p, gram_probs(U, cols, S)) < 1e-13 and abs(float(p.sum()) - 1) < 1e-12
    assert torch.equal(k.probs(U, indist=1.0), k.probs(U))
    for b in (0, 1, 2):
        assert abs((1 - float(k.probs(U, indist=x, order=b).sum())) - truncation_bound(n, x, b)) < 1e-12
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    pb = k.probs(torch.stack([U, haar_unitary(m, 4)]), indist=xt)
    assert pb.shape == (2, k.output_size()) and torch.allclose(pb[0], p)
    pb[1].sum().backward()
    assert abs(float(xt.grad)) < 1e-10                                           # the mass is 1 at every x
    ld = k.probs(U, indist=x, loss=0.8)
    assert abs(float(sum(blk.sum() for blk in ld.probs)) - 1) < 1e-12
    # a bunched input (two photons in one column) and the fully distinguishable limit
    kb = FockSLOSKernel(3, m, [2, 0, 1, 0, 0, 0], dtype=C128)
    ref = species_probs(U, [(1.0, [0, 0, 2])], [7, 8, 9])
    assert _dict_err(kb.layers, kb.probs(U, indist=0.0), ref) < 1e-13


def test_species_partition_convolution():
    """Two multi-photon species (an annotated input) go through `convolve_layers`."""
    from photonic_engines.slos import FockSLOSKernel
    from photonic_engines.slos.species import species_probs as strong
    m = 6
    U = haar_unitary(m, 5)
    k = FockSLOSKernel(5, m, [1, 1, 0, 1, 1, 1], dtype=C128)
    p = strong(k, U, [[0, 1], [3, 4]], [5])
    ref = species_probs(U, [(1.0, [0, 1, 3, 4, 5])], [0, 0, 1, 1, 2])
    assert _dict_err(k.layers, p, ref) < 1e-13 and abs(float(p.sum()) - 1) < 1e-12


def test_sbs_strong_x_mixture():
    """`ProductSLOSKernel.probs(indist=x)`: the good sources stay a COHERENT product superposition
    (the Dicke coherence — an incoherent per-emission-pattern oracle is off by 2e-2 here), the
    bad ones are Bernoulli classical photons. Oracle: `product_reference_probs` on every source
    subset, convolved by hand."""
    import numpy as np

    from photonic_engines import ProductSuperposition
    from photonic_engines.imperfections import _layer_occupations
    from photonic_engines.slos import ProductSLOSKernel
    from photonic_engines.slos.reference import product_reference_probs
    ns, m, x = 3, 5, 0.6
    rng = np.random.default_rng(0)
    amps = rng.normal(size=(ns, 2)) + 1j * rng.normal(size=(ns, 2))
    amps /= np.linalg.norm(amps, axis=1, keepdims=True)
    st = ProductSuperposition(amps, (0, 2, 3), m)
    U = haar_unitary(m, 5)
    got = ProductSLOSKernel(st, dtype=C128).probs(U, indist=x).as_dict()
    ref = {}
    for G, w in subset_weights(ns, x):
        if G:
            sub = ProductSuperposition(amps[list(G)], tuple(st.modes[i] for i in G), m)
            layers = get_layers(sub.n, m)
            dist = {}
            for kk, blk in enumerate(product_reference_probs(U, sub)):
                for row, p in zip(_layer_occupations(layers, kk), blk.tolist()):
                    dist[tuple(int(v) for v in row)] = dist.get(tuple(int(v) for v in row), 0.0) + p
        else:
            dist = {tuple([0] * m): 1.0}
        for i in range(ns):
            if i in G:
                continue
            new = {}
            for t, p in dist.items():
                new[t] = new.get(t, 0.0) + p * abs(amps[i, 0]) ** 2
                for j in range(m):
                    t2 = list(t); t2[j] += 1
                    new[tuple(t2)] = new.get(tuple(t2), 0.0) + p * abs(amps[i, 1]) ** 2 * float(abs(U[j, st.modes[i]]) ** 2)
            dist = new
        for t, p in dist.items():
            ref[t] = ref.get(t, 0.0) + w * p
    assert max(abs(got.get(t, 0.0) - ref.get(t, 0.0)) for t in set(got) | set(ref)) < 1e-13
    assert abs(sum(ProductSLOSKernel(st, dtype=C128).probs(U, indist=x, loss=0.8).as_dict().values()) - 1) < 1e-12


def test_emitter_strong_x_mixture_and_species_kraus():
    """`EmitterSLOSKernel.probs(indist=x)` (traced and projected) against the expanded-chain
    oracle; `kraus_species` is trace-preserving and reproduces the probabilities; the samplers
    agree with this strong side (a cheaper oracle than the permanent sum)."""
    from photonic_engines.clifford import emitter_sample
    from photonic_engines.imperfections import truncation_bound
    from photonic_engines.slos import EmitterSLOSKernel
    n, m, x = 4, 8, 0.6
    st = _caterpillar(n, m)
    U = haar_unitary(m, 3)
    k = EmitterSLOSKernel(st, dtype=C128)
    p = k.probs(U, indist=x)
    assert _dict_err(k.layers, p, _emitter_oracle(st, U, x)) < 1e-13 and abs(float(p.sum()) - 1) < 1e-12
    assert abs((1 - float(k.probs(U, indist=x, order=1).sum())) - truncation_bound(n, x, 1)) < 1e-12
    K, coarse = k.kraus_species(U, x)
    assert torch.allclose(torch.einsum("qts,qtu->su", K.conj(), K), torch.eye(2, dtype=C128), atol=1e-10)
    pk = torch.zeros(k.output_size(), dtype=torch.float64).index_add(0, coarse, (torch.einsum("qts,s->qt", K, k.s0).abs() ** 2).sum(1))
    assert torch.allclose(pk, p, atol=1e-12)
    stp = EmitterMPS(ops=st.ops, modes=st.modes, s0=st.s0, m=m, final=[0.6, 0.8j])
    kp = EmitterSLOSKernel(stp, dtype=C128)
    pp = kp.probs(U, indist=x)
    assert _dict_err(kp.layers, pp, _emitter_oracle(stp, U, x)) < 1e-13 and abs(float(pp.sum()) - 0.5) < 1e-12
    shots = 8000
    ref = {t: float(p[i]) for i, t in enumerate(k.layers.final_states)}
    occ = emitter_sample(st, U, shots, backend="torch", indist=x, generator=torch.Generator().manual_seed(5))
    assert _tvd(_empirical(occ), ref) < 2.0 * _noise_floor(ref, shots)


def test_perceval_source_and_annotated_inputs():
    """Perceval's `NoiseModel(indistinguishability=V)` is the x-model with x = sqrt(V) (the
    parameter is the HOM visibility: the good-photon probability is sqrt(V)); an annotated
    BasicState is a species partition. Both to 1e-15 against Perceval's own simulation."""
    pcvl = pytest.importorskip("perceval")
    from photonic_engines.adapters.perceval import SLOSTorchBackend
    m, V = 4, 0.81
    U = haar_unitary(m, 4)
    circ = pcvl.Unitary(pcvl.Matrix(U.numpy()))
    p = pcvl.Processor("SLOS", m, noise=pcvl.NoiseModel(indistinguishability=V))
    p.add(0, circ); p.min_detected_photons_filter(0); p.with_input(pcvl.BasicState([1, 1, 1, 0]))
    ref = {tuple(k): float(v) for k, v in p.probs()["results"].items()}
    b = SLOSTorchBackend(dtype=C128, indist=math.sqrt(V))
    b.set_circuit(circ); b.set_input_state(pcvl.BasicState([1, 1, 1, 0]))
    got = {tuple(k): float(v) for k, v in b.prob_distribution().items()}
    assert max(abs(got.get(t, 0.0) - ref.get(t, 0.0)) for t in set(got) | set(ref)) < 1e-13
    with pytest.raises(NotImplementedError):
        b.prob_amplitude(pcvl.BasicState([1, 1, 1, 0]))
    for text in ("|{0},{1},{0},0>", "|{_:0},{_:1},{_:0},0>"):
        bs = pcvl.BasicState(text)
        b2 = SLOSTorchBackend(dtype=C128)
        b2.set_circuit(circ); b2.set_input_state(bs)
        got2 = {tuple(k): float(v) for k, v in b2.prob_distribution().items()}
        assert abs(sum(got2.values()) - 1) < 1e-12
        ref2 = species_probs(U, [(1.0, [0, 1, 2])], [0, 1, 0])
        assert max(abs(got2.get(t, 0.0) - ref2.get(t, 0.0)) for t in set(got2) | set(ref2)) < 1e-13
    sim = pcvl.Simulator(pcvl.SLOSBackend()); sim.set_circuit(circ)
    ref3 = {tuple(k): float(v) for k, v in sim.probs_svd(pcvl.SVDistribution(pcvl.BasicState("|{0},{1},{0},0>")))["results"].items()}
    assert max(abs(got2.get(t, 0.0) - ref3.get(t, 0.0)) for t in set(got2) | set(ref3)) < 1e-13
