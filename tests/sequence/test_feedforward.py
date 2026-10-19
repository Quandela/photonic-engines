"""Feed-forward in SequenceKernel.sample_diag (docs/SEQUENCE.md §10): a policy that returns the
emitters' own gates reproduces the plain sampler shot for shot; a real policy (next-bin spin gates
set by the previous bin's pattern) is checked against the exact branch-by-branch distribution."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary

from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel

C128 = torch.complex128


def _ems(gates_per_emitter, k):
    return [BinEmitter.caterpillar(len(gates_per_emitter[j]), gates_per_emitter[j], (2 * j, 2 * j + 1)) for j in range(k)]


def test_identity_policy_matches_plain_sampler():
    k, T, m = 3, 4, 6
    rng = np.random.default_rng(0)
    gates = [[spin_gate(math.pi / 2, 0)]
             + [spin_gate(rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)) for _ in range(T - 1)]
             for _ in range(k)]
    sk = SequenceKernel(_ems(gates, k), m)
    U = haar_unitary(m, 3)
    ref = sk.sample_diag(U, 40, generator=torch.Generator().manual_seed(5))

    def policy(b, counts, psi):
        G = torch.stack([torch.as_tensor(gates[j][b + 1], dtype=C128) for j in range(k)])   # [k, 2, 2]
        return G.unsqueeze(0).expand(counts.shape[0], k, 2, 2)

    out = sk.sample_diag(U, 40, generator=torch.Generator().manual_seed(5), policy=policy)
    assert torch.equal(ref, out)


def test_policy_against_branch_enumeration():
    """k = 2 emitters, m = 4, T = 3: gate of bin b+1 for emitter j = U(theta_j + pi/2 * c_j, phi_j),
    c_j = clicks in the emitter's own modes at bin b. Exact: for every branch of earlier
    patterns, the sequence probability of a SequenceKernel built with the branch's gates."""
    k, T, m = 2, 3, 4
    base = [(0.7, 0.3), (1.9, 2.5)]
    U = haar_unitary(m, 11)

    def gate(j, c):
        return spin_gate(base[j][0] + math.pi / 2 * c, base[j][1])

    def clicks(counts_row):
        return [int(counts_row[2 * j] + counts_row[2 * j + 1]) for j in range(k)]

    sk0 = SequenceKernel(_ems([[spin_gate(math.pi / 2, 0)] + [gate(j, 0)] * (T - 1) for j in range(k)], k), m)
    pats = [tuple(p) for p in sk0.patterns.tolist()]
    # exact distribution over sequences (t1, t2, t3)
    exact = {}
    for t1 in pats:
        c1 = clicks(t1)
        for t2 in pats:
            c2 = clicks(t2)
            gates = [[spin_gate(math.pi / 2, 0), gate(j, c1[j]), gate(j, c2[j])] for j in range(k)]
            skb = SequenceKernel(_ems(gates, k), m)
            for t3 in pats:
                exact[(t1, t2, t3)] = float(skb.sequence_probability(U, [[list(t1), list(t2), list(t3)]])[0])
    tot = sum(exact.values())
    assert abs(tot - 1) < 1e-8, tot

    def policy(b, counts, psi):
        S = counts.shape[0]
        G = torch.empty(S, k, 2, 2, dtype=C128)
        for s in range(S):
            c = clicks(counts[s].tolist())
            for j in range(k):
                G[s, j] = torch.as_tensor(gate(j, c[j]), dtype=C128)
        return G

    shots = 20000
    out = sk0.sample_diag(U, shots, generator=torch.Generator().manual_seed(1), policy=policy)
    hist = {}
    for row in out.tolist():
        key = tuple(tuple(x) for x in row)
        hist[key] = hist.get(key, 0) + 1
    tvd = 0.5 * sum(abs(hist.get(key, 0) / shots - p) for key, p in exact.items())
    floor = 0.5 * sum(math.sqrt(p * (1 - p) / shots) for p in exact.values()) * math.sqrt(2 / math.pi)
    assert tvd < 2.5 * floor, (tvd, floor)
    # and the feed-forward changes the distribution (exact vs exact)
    keys = list(exact)
    plain = sk0.sequence_probability(U, [[list(a), list(b), list(c)] for a, b, c in keys]).tolist()
    tvd0 = 0.5 * sum(abs(p0 - exact[key]) for key, p0 in zip(keys, plain))
    assert tvd0 > 0.05, tvd0


def test_routing_policy_against_branch_enumeration():
    """k = 2, m = 4, T = 3: the interferometer of bin b+1 is U_A or U_B according to the parity
    of the clicks on modes 0, 1 at bin b (gates fixed). Exact: per-bin unitaries [T, m, m] per branch."""
    k, m = 2, 4
    gates = [[spin_gate(math.pi / 2, 0), spin_gate(0.9, 0.4), spin_gate(2.2, 1.7)],
             [spin_gate(math.pi / 2, 0), spin_gate(1.4, 2.9), spin_gate(0.3, 0.8)]]
    sk = SequenceKernel(_ems(gates, k), m)
    UA, UB = haar_unitary(m, 21), haar_unitary(m, 22)
    U0 = haar_unitary(m, 23)
    pats = [tuple(p) for p in sk.patterns.tolist()]

    def pick(t):
        return UB if (t[0] + t[1]) % 2 else UA

    exact = {}
    for t1 in pats:
        for t2 in pats:
            Us = torch.stack([U0, pick(t1), pick(t2)])
            for t3 in pats:
                exact[(t1, t2, t3)] = float(sk.sequence_probability(Us, [[list(t1), list(t2), list(t3)]])[0])
    assert abs(sum(exact.values()) - 1) < 1e-8

    def policy(b, counts, psi):
        par = ((counts[:, 0] + counts[:, 1]) % 2).to(torch.bool)
        return torch.where(par[:, None, None], UB.unsqueeze(0), UA.unsqueeze(0))     # [S, m, m]

    shots = 20000
    out = sk.sample_diag(U0, shots, generator=torch.Generator().manual_seed(4), policy=policy)
    hist = {}
    for row in out.tolist():
        key = tuple(tuple(x) for x in row)
        hist[key] = hist.get(key, 0) + 1
    tvd = 0.5 * sum(abs(hist.get(key, 0) / shots - p) for key, p in exact.items())
    floor = 0.5 * sum(math.sqrt(p * (1 - p) / shots) for p in exact.values()) * math.sqrt(2 / math.pi)
    assert tvd < 2.5 * floor, (tvd, floor)
    keys = list(exact)
    plain = sk.sequence_probability(U0, [[list(a), list(b), list(c)] for a, b, c in keys]).tolist()
    assert 0.5 * sum(abs(p0 - exact[key]) for key, p0 in zip(keys, plain)) > 0.05
