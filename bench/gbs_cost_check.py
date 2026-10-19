"""Consistency check of the GBS frontier statistic (gaussian/cost.py): the mean seconds per sample at
the target squeezing, estimated from independent shot sets drawn at different tilts s. Importance
sampling with exact weights is unbiased at every s, so the estimates must agree within their errors;
the plain mean (s = 1) is the same estimator with the largest variance.

    python3 bench/gbs_cost_check.py --device cuda --n 20,24,28 --tilts 1,1.2,1.414 --shots 128
"""
import argparse
import csv

import numpy as np

from photonic_engines import GaussianInput
from photonic_engines.gaussian import sample
from photonic_engines.gaussian.cost import detected_law, mixture_cost, tilted

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--n", default="12,16")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--mean", type=float, default=0.5)
ap.add_argument("--tilts", default="1,1.2,1.414")
ap.add_argument("--shots", type=int, default=64)
ap.add_argument("--impl", default="v2")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default=None)
args = ap.parse_args()

rows = []
for n in (int(x) for x in args.n.split(",")):
    m = round(args.ratio * n)
    st = GaussianInput.smsv(n, m, GaussianInput.r_for_mean(args.mean))
    rng = np.random.default_rng(args.seed + n)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, r = np.linalg.qr(z)
    U = q * (np.diag(r) / np.abs(np.diag(r)))
    K = 12 * m + 60
    lt = detected_law(st, K)
    sample(st, U, 1, seed=1, device=args.device, impl=args.impl)       # warm-up
    for s in (float(x) for x in args.tilts.split(",")):
        stt, s_used = tilted(st, s)
        _, stats = sample(stt, U, args.shots, seed=args.seed + 7 * n + int(100 * s), device=args.device,
                          impl=args.impl, return_stats=True)
        e = mixture_cost(stats.photons, stats.seconds, [0] * args.shots, lt, [detected_law(stt, K)])
        print(f"  n={n:2d} m={m:3d} tilt s={s_used:.3f}: {e.seconds:.4f} +- {e.stderr:.4f} s/sample "
              f"(rel {e.stderr / e.seconds:.1%}), shots at the tilt {np.mean(stats.seconds):.4f} s mean, "
              f"N drawn {min(stats.photons)}..{max(stats.photons)}", flush=True)
        rows.append([n, m, s_used, e.seconds, e.stderr, float(np.mean(stats.seconds)), max(stats.photons)])
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "m", "tilt", "seconds_per_sample", "stderr", "mean_shot_seconds_at_tilt", "max_N"])
        w.writerows(rows)
