"""Rounding error of the label-amplitude passes of the sequence kernel at k = 16-20 (Supplement C; cross-engine
review item #4, docs/SEQUENCE.md §12c).

Three fp64 evaluations of the same 2^k label amplitudes <t|U|x> of a stream (docs/SEQUENCE.md §12, §12b): the
collapsed Glynn-GEMM sum (the default), the uncollapsed one (2^(k-1) signs, a different accumulation order and
2-10x more addends) and the Laplace recurrence (a different algorithm: O(k 3^k) fused multiply-adds, no
cancellation across signs). Their pairwise disagreement is a lower bound of the fp64 error of each, on the
amplitudes and on the probabilities |a|^2 the tick's draw sees. Haar U, m = 2k, Bose-Einstein-like output
patterns (uniform positions with repetition, as the sampler's own patterns). The Laplace state is 19 GB per
stream at k = 20 (one stream per chunk; skipped above --max-gb).

    python3 bench/sequence_precision.py --k 16,18,20 --shots 4

CSV: k, mode, rel_amp (max over streams of max_x |Δa| / max_x |a|), rel_prob (the same on |a|^2), seconds.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))))


def rel(a, b):
    num = (a - b).abs().flatten(1).max(1).values
    den = a.abs().flatten(1).max(1).values.clamp_min(1e-300)
    return float((num / den).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", default="16,18,20")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-gb", type=float, default=40.0, help="skip the Laplace pass above this per stream")
    ap.add_argument("--out", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "sequence_precision.csv"))
    args = ap.parse_args()
    dev = args.device
    rows = []
    for k in [int(x) for x in args.k.split(",")]:
        m, S = 2 * k, args.shots
        U = haar(m, args.seed).to(dev)
        ems = [BinEmitter.caterpillar(2, [spin_gate(math.pi / 2, 0)] * 2, (2 * j, 2 * j + 1)) for j in range(k)]
        sk = SequenceKernel(ems, m, device=dev)
        g = torch.Generator().manual_seed(args.seed + 1)
        pos = torch.rand(S, m + k - 1, generator=g).argsort(1)[:, :k].sort(1).values
        r = (pos - torch.arange(k)).to(dev)
        counts = torch.zeros((S, m), dtype=torch.long, device=dev).scatter_add_(1, r, torch.ones_like(r))
        peak_gb = 2 * max(math.comb(k, j + 1) * 2 ** (j + 1) for j in range(k)) * 16 / 2 ** 30
        res = {}
        for name, kw in (("glynn", {}), ("glynn_full", {"method": "glynn_full"}),
                         ("laplace", {"method": "laplace", "chunk_bytes": int(1.2 * peak_gb * 2 ** 30) + (1 << 30)})):
            if name == "laplace" and peak_gb > args.max_gb:
                print(f"k={k}: laplace skipped ({peak_gb:.1f} GB per stream > --max-gb)", flush=True)
                continue
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            a = sk.label_amplitudes(U, counts, **kw)
            if dev == "cuda":
                torch.cuda.synchronize()
            res[name] = (a, time.time() - t0)
            print(f"k={k} {name:10s}: {res[name][1]:7.1f} s  (max |a| {float(a.abs().max()):.3e})", flush=True)
        for mode, (x, y) in (("collapsed_vs_full", ("glynn", "glynn_full")), ("collapsed_vs_laplace", ("glynn", "laplace")),
                             ("full_vs_laplace", ("glynn_full", "laplace"))):
            if x in res and y in res:
                ra = rel(res[x][0], res[y][0])
                rp = rel(res[x][0].abs() ** 2, res[y][0].abs() ** 2)
                rows.append({"k": k, "mode": mode, "rel_amp": ra, "rel_prob": rp, "seconds": res[x][1] + res[y][1]})
                print(f"k={k} {mode:22s}: rel_amp {ra:.2e}  rel_prob {rp:.2e}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
