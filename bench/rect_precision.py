"""Rounding error of the fp64 rectangular cofactor kernel (clifford/cuda_rect.py, the SBS rso chain's step) at the
chain's step sizes, on the CPU: the kernel code compiled in double against the same code in x87 long double
(64-bit mantissa) — the TRUE fp64 error of the kernel, every addend of the 2^(q-1) Glynn sum included
(Supplement C; cross-engine review item #4). No GPU needed; ~1 s at q = 16, ~1 min at q = 24 per stream.

    python3 bench/rect_precision.py --N 42 --q 12,16,20,24 [--streams 2] [--collide]

CSV: N, q, rows, rel_cof (max over streams of max_x |P_double - P_ld| / max_x |P_ld|), seconds.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford import cuda_rect


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    qm, r = np.linalg.qr(z)
    return torch.tensor(qm * (np.diag(r) / np.abs(np.diag(r))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=42)
    ap.add_argument("--q", default="12,16,20,24")
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--collide", action="store_true", help="one placed row repeated")
    ap.add_argument("--form", default="lambda", help="lambda | mu (templated lane, s0 <= MU_DMAX) | fock (s0 = 0)")
    ap.add_argument("--s0", type=int, default=None, help="fix s0 = N - 1 - q instead of N (N = q + 1 + s0 per q)")
    ap.add_argument("--tpl", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "rect_precision.csv"))
    args = ap.parse_args()
    S = args.streams
    import ctypes
    if ctypes.sizeof(ctypes.c_longdouble) <= 8:
        raise SystemExit("long double is not extended precision on this host (arm64?): run on x86")
    rows_out = []
    for q in [int(x) for x in args.q.split(",")]:
        N = args.N if args.s0 is None else q + 1 + args.s0
        m = 2 * N
        U = haar(m, args.seed if args.s0 is None else args.seed + N)       # fixed-N mode reproduces the 10-04 CSV
        g = torch.Generator().manual_seed(args.seed + q)
        A = torch.stack([U[:, torch.randperm(m, generator=g)[:N]] for _ in range(S)])
        rows = torch.stack([torch.randperm(m, generator=g)[:q] for _ in range(S)])
        if args.collide and q >= 2:
            rows[:, 1] = rows[:, 0]
        form = args.form if args.form != "fock" or N - 1 - q == 0 else "lambda"
        if form != args.form:
            print(f"warning: --form fock needs s0 = 0; N={N} q={q} has s0 = {N - 1 - q}: measuring the lambda-form",
                  file=sys.stderr, flush=True)
        t0 = time.time()
        a = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=args.tpl, form=form)
        t1 = time.time()
        b = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=args.tpl, real="long double", form=form)
        t2 = time.time()
        rel = float(((a - b).abs().max(1).values / b.abs().max(1).values).max())
        rows_out.append({"N": N, "q": q, "s0": N - 1 - q, "form": form, "tpl": args.tpl,
                         "rows": "collide" if args.collide else "distinct", "rel_cof": rel, "seconds": t2 - t0})
        print(f"N={N} q={q:2d} s0={N - 1 - q} form={form} ({'one repeat' if args.collide else 'distinct'}): "
              f"fp64 vs long double {rel:.2e}   (double {t1 - t0:.0f} s, long double {t2 - t1:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        wr.writeheader(); wr.writerows(rows_out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
