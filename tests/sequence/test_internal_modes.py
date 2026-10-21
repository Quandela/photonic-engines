"""Internal (temporal / frequency) photon modes in SequenceKernel (docs/SEQUENCE.md "Internal modes"): the data
model and the construction-time dispatch. One internal mode must be today's code path, bit for bit; several internal
modes give a ModeSequenceKernel, and the loop engines refuse them."""
import copy
import math
import pickle

import numpy as np
import pytest
import torch
from helpers import haar_unitary

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.sequence import BinEmitter, ModeBinEmitter, SequenceKernel, bin_emitter
from photonic_engines.sequence_modes import ModeSequenceKernel
from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
from photonic_engines.timeloop.hybrid import HybridRegisterKernel, light_hybrid_kernel


def _plain(k=2, n=3, seed=0, final=None):
    rng = np.random.default_rng(seed)
    ems = []
    for j in range(k):
        gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(rng.uniform(0, math.pi), rng.uniform(0, 6)) for _ in range(n - 1)]
        ems.append(BinEmitter.caterpillar(n, gates, (2 * j, 2 * j + 1), final=final))
    return ems


def _split(e, K, seed):
    """The emitter's ops spread over K internal modes by a random isometry on the internal axis: [n, L, K, d, d]."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=K) + 1j * rng.normal(size=K)
    v /= np.linalg.norm(v)
    return e.ops[:, :, None] * v[None, None, :, None, None]


def test_bin_emitter_one_mode_is_the_plain_emitter():
    """[n, L, 1, d, d] and [n, L, d, d] both give a plain BinEmitter with the same values (a copy, not a view)."""
    e = _plain(1)[0]
    a = bin_emitter(e.ops, e.modes, e.s0)
    b = bin_emitter(e.ops[:, :, None], e.modes, e.s0)
    assert type(a) is BinEmitter and type(b) is BinEmitter
    assert np.array_equal(a.ops, e.ops) and np.array_equal(b.ops, e.ops)
    assert not np.shares_memory(b.ops, e.ops)


def test_bin_emitter_several_modes_and_lost():
    e = _plain(1)[0]
    me = bin_emitter(_split(e, 3, 1), e.modes, e.s0)
    assert type(me) is ModeBinEmitter and (me.n, me.K, me.d) == (e.n, 3, 2)
    lost = np.zeros((e.n, 1, 2, 2), dtype=complex)
    one = bin_emitter(e.ops, e.modes, e.s0, lost=lost)                      # lost operators need the new class
    assert type(one) is ModeBinEmitter and one.K == 1 and one.lost.shape == (e.n, 1, 2, 2)


@pytest.mark.parametrize("ops_shape,msg", [((3, 2, 2, 2), "ops must have shape"), ((3, 2, 2, 2, 3), "ops must have shape")])
def test_mode_bin_emitter_validates(ops_shape, msg):
    with pytest.raises(ValueError, match=msg):
        ModeBinEmitter(np.zeros(ops_shape, dtype=complex), (0, 1), [1, 0])
    with pytest.raises(ValueError, match="modes"):
        ModeBinEmitter(np.zeros((3, 2, 2, 2, 2), dtype=complex), (0, 0), [1, 0])
    with pytest.raises(ValueError, match="s0"):
        ModeBinEmitter(np.zeros((3, 2, 2, 2, 2), dtype=complex), (0, 1), [1, 0, 0])
    with pytest.raises(ValueError, match="lost"):
        ModeBinEmitter(np.zeros((3, 2, 2, 2, 2), dtype=complex), (0, 1), [1, 0], lost=np.zeros((2, 1, 2, 2)))


def test_one_mode_kernel_is_bit_identical():
    """Emitters built from [n, L, 1, d, d] give a plain SequenceKernel whose Kraus sets, sequence probabilities and
    trajectories are bit-identical (torch.equal) to the kernel built from today's [n, L, d, d] emitters."""
    m, n = 4, 3
    ems = _plain(2, n, seed=3)
    ems5 = [bin_emitter(e.ops[:, :, None], e.modes, e.s0) for e in ems]
    U = haar_unitary(m, 4)
    a, b = SequenceKernel(ems, m), SequenceKernel(ems5, m)
    assert type(b) is SequenceKernel
    assert torch.equal(a.kraus(U, 0), b.kraus(U, 0))
    seqs = torch.randint(0, a.S, (8, n), generator=torch.Generator().manual_seed(5))
    assert torch.equal(a.sequence_probability(U, seqs), b.sequence_probability(U, seqs))
    sa = a.sample(U, 32, generator=torch.Generator().manual_seed(6))
    sb = b.sample(U, 32, generator=torch.Generator().manual_seed(6))
    assert torch.equal(sa, sb)
    da = a.sample_diag(U, 16, generator=torch.Generator().manual_seed(7), fock_backend="torch")
    db = b.sample_diag(U, 16, generator=torch.Generator().manual_seed(7), fock_backend="torch")
    assert torch.equal(da, db)


def test_dispatch_and_plain_class_unchanged():
    """Only a ModeBinEmitter switches the class; copy and pickle of a plain kernel still work (__new__ with no args)."""
    m = 4
    ems = _plain(2, 2, seed=8)
    sk = SequenceKernel(ems, m)
    assert type(sk) is SequenceKernel
    assert type(SequenceKernel.from_mps([EmitterMPS.caterpillar(2, [spin_gate(math.pi / 2, 0)] * 2, m=4 * 2,
                                                                 modes=[(4 * b, 4 * b + 1) for b in range(2)])], 4)) \
        is SequenceKernel
    assert type(copy.deepcopy(sk)) is SequenceKernel
    assert type(pickle.loads(pickle.dumps(sk))) is SequenceKernel
    mixed = [bin_emitter(_split(ems[0], 3, 9), ems[0].modes, ems[0].s0), ems[1]]
    mk = SequenceKernel(mixed, m)
    assert type(mk) is ModeSequenceKernel and isinstance(mk, SequenceKernel)
    assert (mk.k, mk.n, mk.chi, mk.K) == (2, 2, 4, 3)
    assert mk.S == sk.S and np.array_equal(mk.patterns, sk.patterns)              # the spatial basis is unchanged


def test_internal_modes_through_an_iterator_are_refused():
    """A ModeBinEmitter handed over in an iterator cannot be dispatched before construction: the plain kernel must refuse
    it rather than run routes that would ignore the internal modes. Any Sequence (here a custom one) dispatches."""
    ems = _plain(2, 2, seed=15)
    me = bin_emitter(_split(ems[0], 2, 16), ems[0].modes, ems[0].s0)
    with pytest.raises(TypeError, match="internal modes"):
        SequenceKernel(iter([me, ems[1]]), 4)
    with pytest.raises(TypeError, match="internal modes"):
        SequenceKernel((e for e in [me, ems[1]]), 4)
    assert type(SequenceKernel(iter(ems), 4)) is SequenceKernel                      # plain emitters: unchanged

    from collections.abc import Sequence

    class Seq(Sequence):
        def __init__(self, items):
            self.items = list(items)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    assert type(SequenceKernel(Seq([me, ems[1]]), 4)) is ModeSequenceKernel


def test_mode_kernel_stage_ops_and_not_yet():
    m = 4
    ems = _plain(2, 2, seed=10)
    me = bin_emitter(_split(ems[0], 3, 11), ems[0].modes, ems[0].s0)
    mk = SequenceKernel([me, ems[1]], m)
    ops = mk.stage_ops(1)
    assert [tuple(o.shape) for o in ops] == [(2, 3, 2, 2), (2, 3, 2, 2)]
    assert torch.equal(ops[0], torch.as_tensor(me.ops[1]))
    assert torch.equal(ops[1][:, 0], torch.as_tensor(ems[1].ops[1])) and not ops[1][:, 1:].any()
    U = haar_unitary(m, 12)
    for call in (lambda: mk.sample(U, 4), lambda: mk.sequence_probability(U, [[0, 0]]),
                 lambda: mk.bin_marginals(U), lambda: mk.sample_diag(U, 4), lambda: mk.bin_mps(0)):
        with pytest.raises(NotImplementedError, match="internal modes"):
            call()


def test_every_public_member_is_refused_overridden_or_spatial():
    """A public SequenceKernel member that ModeSequenceKernel neither refuses, overrides nor lists as purely spatial
    would run on internal-mode emitters as if they had one mode: adding one must fail this test."""
    from photonic_engines import sequence_modes
    public = {n for n in vars(SequenceKernel) if not n.startswith("_")}
    overridden = {n for n in vars(ModeSequenceKernel) if not n.startswith("_")}
    unclassified = public - set(sequence_modes._NOT_YET) - overridden - set(sequence_modes.SPATIAL)
    assert not unclassified, f"classify these in sequence_modes: {sorted(unclassified)}"
    assert set(sequence_modes.SPATIAL) <= public


def test_loop_engines_refuse_internal_modes():
    T = 3
    ems = _plain(2, T, seed=13)
    me = bin_emitter(_split(ems[0], 2, 14), ems[0].modes, ems[0].s0)
    for make in (HybridRegisterKernel, light_hybrid_kernel):
        with pytest.raises(NotImplementedError, match="not supported through loops"):
            make([me, ems[1]], 4, (1,))
    c = TimeLoopCircuit(2 * T, (1,))
    with pytest.raises(NotImplementedError, match="not supported through loops"):
        LoopRegisterKernel(c, [0] * (2 * T), emitter=me)
