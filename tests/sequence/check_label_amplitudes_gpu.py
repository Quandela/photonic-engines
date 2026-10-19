"""Device check of the label-amplitude passes (docs/SEQUENCE.md §12, §12b): on cuda, the collapsed
Glynn-GEMM sum, the uncollapsed one and the Laplace recurrence agree at k = 12 on Bose-Einstein
patterns, and `sample_diag(draw="device")` reproduces the bin marginal of the host draw within shot
noise at k = 3. Runnable script, not collected by pytest (run it on a GPU).

    python3 tests/sequence/check_label_amplitudes_gpu.py [--k 12] [--shots 64]
"""
import argparse
import math

import numpy as np
import torch

from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    k, m, S = args.k, 2 * args.k, args.shots
    U = haar(m, 7).to(dev)
    ems = [BinEmitter.caterpillar(2, [spin_gate(math.pi / 2, 0)] * 2, (2 * j, 2 * j + 1)) for j in range(k)]
    sk = SequenceKernel(ems, m, device=dev)
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(S, m + k - 1, generator=g).argsort(1)[:, :k].sort(1).values
    rows = (pos - torch.arange(k)).to(dev)
    counts = torch.zeros((S, m), dtype=torch.long, device=dev).scatter_add_(1, rows, torch.ones_like(rows))
    a_l = sk.label_amplitudes(U, counts, method="laplace")
    a_f = sk.label_amplitudes(U, counts, method="glynn_full")
    a_c = sk.label_amplitudes(U, counts)
    scale = float(a_l.abs().max())
    print(f"k={k} S={S} on {dev}: |collapsed - laplace| max {float((a_c - a_l).abs().max()):.2e}, "
          f"|full - laplace| max {float((a_f - a_l).abs().max()):.2e}, scale {scale:.2e}")
    assert torch.allclose(a_c, a_l, atol=1e-13) and torch.allclose(a_f, a_l, atol=1e-13)

    # the device label draw: same bin marginal as the host draw (k = 3, n = 2, 20000 shots)
    k3, n3, m3, shots = 3, 2, 6, 20000
    gates = [spin_gate(math.pi / 2, 0.3), spin_gate(1.1, 0.2)]
    ems3 = [BinEmitter.caterpillar(n3, gates, (2 * j, 2 * j + 1)) for j in range(k3)]
    sk3 = SequenceKernel(ems3, m3, device=dev)
    U3 = haar(m3, 5).to(dev)
    marg = sk3.bin_marginals(U3).cpu()
    for draw, backend in (("host", "triton"), ("device", "triton"), ("device", "cuda")):
        try:
            s = sk3.sample_diag(U3, shots, generator=torch.Generator().manual_seed(1), draw=draw, fock_backend=backend)
        except ImportError as e:                                               # cupy missing: the cuda backend is optional
            print(f"draw={draw} fock_backend={backend}: skipped ({e})")
            continue
        for b in range(n3):
            idx = sk3.pattern_index(s[:, b].numpy()).cpu()
            emp = torch.zeros(sk3.S, dtype=torch.float64).index_add_(0, idx, torch.full((shots,), 1 / shots, dtype=torch.float64))
            floor = 0.5 * math.sqrt(2 / math.pi) * torch.sqrt(marg[b] * (1 - marg[b]) / shots).sum()
            tvd = 0.5 * (emp - marg[b]).abs().sum()
            print(f"draw={draw} fock_backend={backend} bin {b}: TVD {float(tvd):.4f} (floor {float(floor):.4f})")
            assert tvd < 3 * floor
    print("OK")


if __name__ == "__main__":
    main()
