"""Internal photon modes, the weak route (docs/SEQUENCE.md §14, steps 6b-6d): time of one `sample_apply` tick
(CC-spin, operator-valued Clifford-Clifford) for k emitters with qubit spins, L = 2 labels (dual rail, m = 2k), K
internal modes, generic complete emitters, a batch of S streams from one random register state. Per (k, K, S): the
median of --reps after one warm-up, the time per tick and per stream-tick, the largest per-step identity deviation, and
the cost units of the model in `sequence_modes.COST` ("C": sum_j C(k, j) j L K D X (d (1 + k - j) + j L) per stream).
A row whose time, predicted from the previous row's per-stream time, exceeds --max-row-s is skipped (recorded), as
are the larger stream counts after a row that took longer; an out-of-memory row is recorded as such. The CSV is
rewritten after every row (a timeout keeps what is done).

    python3 bench/internal_modes_ccspin_bench.py --device cuda --k 2,4,6,8 --K 1,3,8 --streams 1,100,10000 --csv out.csv

Run from the repository root with this checkout's `photonic_engines` importable (installed, or the root on sys.path).
"""
import argparse
import csv
import math
import statistics
import time

import numpy as np
import torch

from photonic_engines.sequence import ModeBinEmitter, SequenceKernel

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--k", default="2,3,4", help="emitters (photons) per tick, comma list")
ap.add_argument("--K", default="1,3,8", help="internal modes, comma list")
ap.add_argument("--streams", default="256", help="streams per tick, comma list")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--max-row-s", type=float, default=60.0, help="skip larger stream counts after a row this slow")
ap.add_argument("--csv", default="internal_modes_ccspin_bench.csv")
args = ap.parse_args()
dev = args.device


def sync():
    """Wait for the device."""
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def haar(m, seed):
    """Haar-random unitary [m, m]."""
    z = np.random.default_rng(seed).normal(size=(m, m, 2)) @ [1, 1j]
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / abs(np.diag(r))), dtype=torch.complex128)


def emitter(K, j, seed):
    """Dual-rail qubit emitter on modes (2j, 2j + 1), K internal modes, one bin: a random isometry [2 K 2, 2]."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(2 * K * 2, 2)) + 1j * rng.normal(size=(2 * K * 2, 2))
    q, _ = np.linalg.qr(z)
    return ModeBinEmitter(q.reshape(1, 2, K, 2, 2), (2 * j, 2 * j + 1), [1, 0])


def units(k, L, K, d, X):
    """Cost units per stream-tick of the CC-spin model (local applies + Grams)."""
    D = d ** k
    return sum(math.comb(k, j) * j * L * K * D * X * (d * (1 + k - j) + j * L) for j in range(1, k + 1))


def write_csv(rows):
    """(Re)write the CSV with every row so far."""
    keys = []
    for r in rows:
        keys += [x for x in r if x not in keys]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


rows = []
for k in (int(x) for x in args.k.split(",")):
    m = 2 * k
    U = haar(m, k).to(dev)
    for K in (int(x) for x in args.K.split(",")):
        mk = SequenceKernel([emitter(K, j, 10 * K + j) for j in range(k)], m, device=dev)
        g0 = torch.Generator().manual_seed(k)
        z = torch.randn(mk.chi, generator=g0, dtype=torch.complex128)
        z = (z / z.norm()).to(dev)
        per_stream_prev = None
        for S in (int(x) for x in args.streams.split(",")):
            row = {"k": k, "K": K, "K_eff": mk._kb(U, 0), "chi": mk.chi, "streams": S,
                   "units_per_stream": units(k, 2, mk._kb(U, 0), 2, 1)}
            if per_stream_prev is not None and per_stream_prev * S > args.max_row_s:
                row["note"] = f"skipped: predicted {per_stream_prev * S:.0f} s > --max-row-s"      # no silent overrun
                rows.append(row)
                print(row, flush=True)
                write_csv(rows)
                break
            psi = z.reshape(1, -1, 1).expand(S, -1, -1).contiguous()
            info = {}

            def tick(mk=mk, U=U, psi=psi, info=info):
                """One tick for every stream."""
                info.update(mk.sample_apply(U, 0, psi, generator=torch.Generator().manual_seed(1), index=False)[2])
            try:
                t0 = time.perf_counter()
                tick()
                sync()
                first = time.perf_counter() - t0
                ts = [first] if first > args.max_row_s else []
                for _ in range(0 if ts else args.reps):
                    t = time.perf_counter()
                    tick()
                    sync()
                    ts.append(time.perf_counter() - t)
                t_med = statistics.median(ts)
                row.update({"t_tick_s": t_med, "t_per_stream_s": t_med / S,
                            "max_deviation": float(info["deviation"].max())})
                per_stream_prev = t_med / S
            except (torch.cuda.OutOfMemoryError, MemoryError) as err:
                row["note"] = "out of memory: " + str(err)[:60]
                t_med = float("inf")
                if str(dev).startswith("cuda"):
                    torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
            write_csv(rows)
            if t_med > args.max_row_s:
                break                                 # larger stream counts of this (k, K) would be slower still
print("wrote", args.csv)
