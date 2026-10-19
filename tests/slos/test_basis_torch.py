"""TorchFockLayers (on-device builder) must equal the numpy FockLayers: same lex-ordered
states, same edge set per (parent, child, mode, factor), same mode slabs; and the kernels must
give identical results on both."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import haar_unitary, occupation_list

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos.basis import FockLayers, get_layers
from photonic_engines.slos.basis_torch import TorchFockLayers
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.slos.kernel import FockSLOSKernel

CASES = [(4, 6, "fock"), (5, 7, "fock"), (3, 9, "fock"), (4, 8, "unbunched"),
         (4, 8, "dual_rail"), (3, 12, "sparse_1in4"), (4, 7, "tail_empty"), (6, 6, "fock")]


@pytest.mark.parametrize("n,m,mask", CASES)
def test_torch_equals_numpy(n, m, mask):
    a = FockLayers(n, m, mask)
    b = TorchFockLayers(n, m, mask, device="cpu")
    for k in range(n + 1):
        assert np.array_equal(a.states[k], b.states[k].numpy()), f"layer {k}"
    for k in range(n):
        pa, ca, ja, fa = a.edges[k]
        pb, cb, jb, fb = b.edges[k]
        ea = sorted(zip(pa.tolist(), ca.tolist(), ja.tolist(), np.round(fa.astype(np.float64), 5).tolist()))
        eb = sorted(zip(pb.tolist(), cb.tolist(), jb.tolist(), np.round(fb.numpy(), 5).tolist()))
        assert ea == eb, f"edges {k}"
        # mode slabs: same offsets and same (parent, child) multiset per slab
        pa2, ca2, offa = a.edges_by_mode(k)
        pb2, cb2, offb = b.edges_by_mode(k)
        assert list(offa) == list(offb)
        for j in range(m):
            sa = sorted(zip(pa2[offa[j]:offa[j + 1]].tolist(), ca2[offa[j]:offa[j + 1]].tolist()))
            sb = sorted(zip(pb2[offb[j]:offb[j + 1]].tolist(), cb2[offb[j]:offb[j + 1]].tolist()))
            assert sa == sb, f"slab {k}/{j}"
    assert a.output_prob_norms().tolist() == b.output_prob_norms().tolist()
    assert np.array_equal(a.final_occupations(), b.final_occupations().numpy())


def test_kernels_agree_on_both_builders():
    n, m = 4, 8
    U = haar_unitary(m, 3).to(torch.complex64)
    occ = occupation_list(n, m)
    lt = get_layers(n, m, "fock", "cpu", backend="torch")
    kn, kt = FockSLOSKernel(n, m, occ), FockSLOSKernel(n, m, occ, layers=lt)
    assert torch.allclose(kn.probs(U), kt.probs(U), atol=1e-6)
    st = EmitterMPS.caterpillar(n, [spin_gate(1.2, 0.3)] * n)
    en, et = EmitterSLOSKernel(st), EmitterSLOSKernel(st, layers=lt)
    assert torch.allclose(en.probs(U), et.probs(U), atol=1e-6)
    ke = FockSLOSKernel(n, m, occ, layers=lt, layout="edge")           # derived child-major edges
    assert torch.allclose(kn.probs(U), ke.probs(U), atol=1e-6)
