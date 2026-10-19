"""The lossless cost table t(k, m): seconds per exact sample of k photons in m modes, MEASURED at the
mode counts of the devices placed in the hardware-competitiveness analysis (docs/HARDWARE_TARGET.md
§"Where current devices sit"), so that the device numbers rest on measured points
rather than on the rho^(n* - k) extrapolation of the frontier (which runs 25+ photons outside its fit
range at small k, and outside the measured mode ratios at m/k > 12).

    python3 bench/cost_table.py --ms 12,24,32,48,96 --budget 120 --csv results/cost_table.csv
    python3 bench/cost_table.py --ms 96 --kmin 20 --budget 240                       # one column, deeper

For every m the photon number k runs from --kmin up: each point is `find_threshold.measure` verbatim
(Haar U, seed = --seed + 1000 m + k, photons in modes 0..k-1, production sampler algo auto / backend
cuda, shot ramp 1, 2, 4, ... until a run exceeds budget / 3, sustained rate over the runs with
>= --min-shots shots); the column stops at the first k whose single warm-up shot exceeds --budget,
or at k = m. The ramp runs to --max-shots = 65536 so the small-k points measure the batched rate the
classical side really has (at 256 shots they would be launch-bound, 100x too slow). Points past the stop
are extrapolated by the consumer (bench/plot_postselect.py) with the frontier's rho_eff at that mode ratio, and are labelled as such there.

CSV: m, k, ratio, samples_per_s (sustained, the find_threshold definition), ms_per_sample, seconds, shots,
plateau_per_s (best single run with >= --min-shots shots: immune to the Triton recompiles that the lane-count
steps of a long ramp trigger inside the timed runs at small k — the sustained column shows dips there).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines.clifford import fock_sample

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ms", default="12,24,32,48,96")
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--kmax", type=int, default=None, help="stop every column here (default: m, or the budget)")
ap.add_argument("--budget", type=float, default=120.0)
ap.add_argument("--max-shots", type=int, default=1 << 16,
                help="the ramp goes this far at small k (256 shots would leave those points launch-bound)")
ap.add_argument("--min-shots", type=int, default=8)
ap.add_argument("--backend", default="cuda")
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="results/cost_table.csv")
args = ap.parse_args()
dev = args.device
name = torch.cuda.get_device_name(0) if dev == "cuda" and torch.cuda.is_available() else dev
print(f"{name}  ms={args.ms}  k from {args.kmin}  budget={args.budget:g} s  backend={args.backend}", flush=True)


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def measure(k, m):
    U = haar(m, args.seed + 1000 * m + k).to(dev)
    in_modes = list(range(k))
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1000 * m + k)

    def run(shots):
        sync(); t = time.perf_counter()
        occ = fock_sample(U, in_modes, shots, generator=gen, algo="auto", backend=args.backend)
        sync()
        return time.perf_counter() - t, occ

    t_w = time.perf_counter()
    dt, _ = run(1)
    warm = time.perf_counter() - t_w
    if dt > args.budget:
        print(f"  m={m:3d} k={k:2d}: one shot took {dt:.0f} s > budget, column stops", flush=True)
        return None
    shots, spent, tot_shots, tot_time, best, last, plateau = 1, 0.0, 0, 0.0, 0.0, 1, 0.0
    while shots <= args.max_shots:
        dt, occ = run(shots)
        last = shots; spent += dt
        if shots >= args.min_shots:
            tot_shots += shots; tot_time += dt
            plateau = max(plateau, shots / dt)                              # best single run (bench/rate_series.py)
        best = shots / dt if tot_shots == 0 else tot_shots / tot_time
        if dt > args.budget / 3 or spent > args.budget:
            break
        shots *= 2
    X = occ.cpu().numpy().astype(np.float64)
    err = float(np.abs(X.mean(0) - (np.abs(U.cpu().numpy()[:, :k]) ** 2).sum(1)).max())
    print(f"  m={m:3d} k={k:2d} (m/k = {m / k:5.2f}): {best:11.3f} samples/s  ({1e3 / best:11.4f} ms/sample, up to {last} shots, "
          f"{spent:.0f} s + {warm:.0f} s warm-up, max|dE[n_j]|={err:.3f})", flush=True)
    return best, spent, last, plateau


os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
rows = []
for m in [int(x) for x in args.ms.split(",")]:
    kmax = min(m, args.kmax) if args.kmax else m
    for k in range(args.kmin, kmax + 1):
        r = measure(k, m)
        if r is None:
            break
        rate, sec, shots, plateau = r
        rows.append([m, k, round(m / k, 4), rate, 1e3 / rate, round(sec, 2), shots, plateau])
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["m", "k", "ratio", "samples_per_s", "ms_per_sample", "seconds", "shots", "plateau_per_s"])
            w.writerows(rows)
print("wrote", args.csv)
