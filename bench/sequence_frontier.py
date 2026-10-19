"""GPU frontier of the sequence modality (docs/SEQUENCE.md) in k, the number of emitters
interfering per bin: dual-rail emitters into a universal interferometer of m = 2k modes per
bin (the unrolled problem has k·n photons in 2·n·k modes), n bins.

    python3 bench/sequence_frontier.py --device cuda --k 2,3,4,5,6,7 --n 5,10,20,40 --shots 1024
    python3 bench/plot_sequence_frontier.py results/<stamp>/sequence_frontier.csv     # -> img/sequence_frontier.png

Per (k, n): register chi = 2^k, S_bin = C(m+k-1, k) patterns, the Kraus set (S_bin x chi x chi
complex128) built by one bond-carrying SLOS pass, the n-bin marginals (transfer channel), and
the pure-trajectory sampler (enumeration of the bin patterns) — seconds, ms per full n-bin
sample, ms per bin-shot, memory. Also, where the generic route is feasible (k·n <= 12), the
same sequence sampled by the emitter chain-rule sampler on the unrolled block-diagonal
unitary (torch backend, any chi): exponential in k·n, against linear in n here. Writes
$RESULTS_DIR/sequence_frontier.csv. A configuration whose Kraus set does not fit is recorded
as OOM (the set is S_bin · chi^2 · 16 B: 810 MB at k = 6, 20 GB at k = 7, 0.5 TB at k = 8 with
m = 2k).
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def _embed(op, i, dims):
    out = np.ones((1, 1), dtype=complex)
    for j, d in enumerate(dims):
        out = np.kron(out, op if j == i else np.eye(d, dtype=complex))
    return out


def unrolled_state(ems, n, m):
    dims = [e.d for e in ems]
    ops, modes = [], []
    for b in range(n):
        for j, e in enumerate(ems):
            ops.append(np.stack([_embed(e.ops[b, l], j, dims) for l in range(e.ops.shape[1])]))
            modes.append(tuple(b * m + x for x in e.modes))
    s0 = np.ones(1, dtype=complex)
    for e in ems:
        s0 = np.kron(s0, e.s0)
    return EmitterMPS(ops=np.stack(ops), modes=tuple(modes), s0=s0, m=n * m, final=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", default="2,3,4,5,6,7")
    ap.add_argument("--n", default="5,10,20,40", help="comma list of bin counts (cost is linear in n)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--ratio", type=int, default=2, help="modes per bin m = ratio * k (2: dual-rail emitters fill the interferometer)")
    ap.add_argument("--generic-max", type=int, default=12, help="run the unrolled emitter sampler when k*n <= this")
    ap.add_argument("--max-gb", type=float, default=70.0, help="skip a k whose Kraus set + K^dag K exceed this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "sequence_frontier.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    ns = [int(x) for x in args.n.split(",")]
    for k in [int(x) for x in args.k.split(",")]:
      for n in ns:
        m = args.ratio * k
        ems = [BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, (2 * j, 2 * j + 1)) for j in range(k)]
        U = haar(m, 100 + k).to(dev)
        row = {"k": k, "n": n, "m": m, "ratio": args.ratio, "chi": 2 ** k, "S_bin": math.comb(m + k - 1, k),
                   "kraus_GB": round(math.comb(m + k - 1, k) * 4 ** k * 16 / 2 ** 30, 3)}
        print(f"k={k} n={n}: chi={row['chi']} S_bin={row['S_bin']} Kraus set {row['kraus_GB']} GB", flush=True)
        if 2 * row["kraus_GB"] > args.max_gb:
            row["kraus_s"] = f"skipped ({2 * row['kraus_GB']:.0f} GB needed)"
            rows.append(row); print(row, flush=True)
            continue
        try:
            sk = SequenceKernel(ems, m, device=dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _sync(dev); t0 = time.time()
            K = sk.kraus(U, 0)
            _sync(dev); row["kraus_s"] = round(time.time() - t0, 3)
            t0 = time.time()
            marg = sk.bin_marginals(U)
            _sync(dev); row["marginals_s"] = round(time.time() - t0, 3)
            row["marg_sum_err"] = float((marg.sum(1) - 1).abs().max())
            shots = args.shots
            t0 = time.time()
            s = sk.sample(U, shots, generator=torch.Generator().manual_seed(0))
            _sync(dev); dt = time.time() - t0
            row.update(shots=shots, sample_s=round(dt, 3), ms_per_sequence=round(1e3 * dt / shots, 4),
                       ms_per_bin_shot=round(1e3 * dt / shots / n, 5), samples_per_s=round(shots / dt, 2))
            # consistency: empirical bin marginal vs computed (bin 0)
            emp = torch.zeros(sk.S, dtype=torch.float64)
            emp.index_add_(0, sk.pattern_index(s[:, 0].numpy()).cpu(), torch.ones(shots, dtype=torch.float64) / shots)
            row["bin0_tvd"] = round(0.5 * float((emp - marg[0].cpu()).abs().sum()), 4)
            sd = float(torch.sqrt(marg[0].cpu() * (1 - marg[0].cpu()) / shots).sum())
            row["bin0_floor"] = round(0.5 * math.sqrt(2 / math.pi) * sd, 4)
            if dev == "cuda":
                row["peak_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            del K, sk
        except torch.cuda.OutOfMemoryError:
            row["kraus_s"] = "OOM"
            torch.cuda.empty_cache()
        if k * n <= args.generic_max:
            from photonic_engines.clifford import emitter_sample
            full = unrolled_state(ems, n, m)
            Ubig = torch.block_diag(*[U] * n)
            gshots = 64
            _sync(dev); t0 = time.time()
            emitter_sample(full, Ubig, gshots, backend="torch", device=dev, generator=torch.Generator().manual_seed(0))
            _sync(dev); dt = time.time() - t0
            row["generic_ms_per_sequence"] = round(1e3 * dt / gshots, 3)
        rows.append(row)
        print(row, flush=True)
    keys = sorted({kk for r in rows for kk in r}, key=lambda x: (x != "k", x))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
