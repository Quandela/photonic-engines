"""Is a *sample-level* loss sensitive to the emitter's parameters where the register is hard?

The design plane (bench/esbs_design_plane.py) showed that low-order click statistics are blind to
the spin gates under a global U. This bench asks the remaining generative question: a loss
evaluated from samples only — the MMD between model samples and a target dataset, with a kernel on
the click *patterns* — does it move when the gates move, above its own shot noise, at k where
sampling is already expensive classically?

Model: k caterpillar emitters, spin gates (theta, phi) per emitter and bin (the parameters), static
Haar U on m = 2k modes, T bins, sampled by `SequenceKernel.sample_diag`. Target: N samples at a
hidden setting theta*. Loss: unbiased MMD^2 with the sequence kernel
    K(t, t') = prod_b exp(-||t_b - t'_b||^2 / (2 sigma^2)),
sigma -> 0 is the delta kernel on whole sequences (pattern level), sigma large sees only low
moments — the scan over sigma interpolates between the two regimes.

Measurement, per (k, sigma, delta): the loss change Delta L = L(theta* + delta e) - L(theta*) for
random unit directions e in parameter space, its mean over directions and its std over repeats
with fresh samples (the shot noise); SNR = |mean| / std. SNR > 1 at a given N means the parameters
are visible to that loss at that shot budget; the null noise of the U-statistic scales as 1/N, so
`shots_for_snr3` = N * 3 / SNR extrapolates the budget for SNR = 3 (valid once SNR is itself resolved).
(An exact MMD from the strong side would need the double
Kraus chain over sequence pairs — not implemented; the null loss L0 at theta* is the control.)

    python3 bench/esbs_train_signal.py --device cuda --k 6,8,10,12,14 --shots 4096 --sigma 0.3,0.7,1.5,3 --delta 0.5,1.5
Writes $RESULTS_DIR/esbs_train_signal.csv.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel

C128 = torch.complex128


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=C128)


def kernel_model(k, T, m, angles, device):
    """angles [k, T, 2] -> SequenceKernel (bin 0 = preparation pi/2, 0; bins 1.. trainable)."""
    ems = []
    for j in range(k):
        gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(float(angles[j, b, 0]), float(angles[j, b, 1])) for b in range(1, T)]
        ems.append(BinEmitter.caterpillar(T, gates, (2 * j, 2 * j + 1)))
    return SequenceKernel(ems, m, device=device)


def mmd2(X, Y, sigma):
    """Unbiased MMD^2 between sample sets X, Y [N, T, m] (long) with the product-of-bins Gaussian kernel."""
    def gram(A, B, same):
        A = A.double(); B = B.double()
        d2 = torch.zeros(A.shape[0], B.shape[0], dtype=torch.float64, device=A.device)
        for b in range(A.shape[1]):
            d2 += torch.cdist(A[:, b], B[:, b]) ** 2
        K = torch.exp(-d2 / (2 * sigma ** 2))
        if same:
            n = A.shape[0]
            return (K.sum() - K.diagonal().sum()) / (n * (n - 1))
        return K.mean()
    return float(gram(X, X, True) + gram(Y, Y, True) - 2 * gram(X, Y, False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", default="4,6,8")
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--sigma", default="0.3,0.7,1.5,3")
    ap.add_argument("--delta", default="0.5,1.5")
    ap.add_argument("--directions", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), "esbs_train_signal.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sigmas = [float(x) for x in args.sigma.split(",")]
    deltas = [float(x) for x in args.delta.split(",")]
    rows = []
    for k in [int(x) for x in args.k.split(",")]:
        m, T = 2 * k, args.ticks
        rng = np.random.default_rng(args.seed + k)
        U = haar(m, args.seed + k).to(dev)
        star = np.stack([rng.uniform(0, math.pi, size=(k, T)), rng.uniform(0, 2 * math.pi, size=(k, T))], -1)   # [k, T, 2]
        dirs = rng.normal(size=(args.directions, k, T, 2)); dirs[:, :, 0] = 0                                      # bin 0 fixed
        dirs /= np.linalg.norm(dirs.reshape(args.directions, -1), axis=1)[:, None, None, None]
        gen = torch.Generator().manual_seed(args.seed)

        def samples(angles, n):
            sk = kernel_model(k, T, m, angles, dev)
            return sk.sample_diag(U, n, generator=gen).to(dev)

        t0 = time.time()
        target_sets = [samples(star, args.shots) for _ in range(args.repeats)]
        base_sets = [samples(star, args.shots) for _ in range(args.repeats)]
        t_sample = (time.time() - t0) / (2 * args.repeats)
        for delta in deltas:
            shifted = [[samples(star + delta * dirs[d], args.shots) for _ in range(args.repeats)] for d in range(args.directions)]
            for sigma in sigmas:
                # loss at theta* (0 in expectation)
                L0 = [mmd2(base_sets[r], target_sets[r], sigma) for r in range(args.repeats)]
                dL = np.array([[mmd2(shifted[d][r], target_sets[r], sigma) - L0[r] for r in range(args.repeats)]
                               for d in range(args.directions)])
                mean_d = dL.mean(1)                                                                        # per direction, over repeats
                noise = float(np.sqrt(np.mean(dL.var(1, ddof=1))))                                        # shot noise of Delta L
                row = {"k": k, "m": m, "ticks": T, "shots": args.shots, "sigma": sigma, "delta": delta,
                           "dL_mean": float(mean_d.mean()), "dL_min": float(mean_d.min()), "dL_max": float(mean_d.max()),
                           "noise": noise, "snr": float(abs(mean_d.mean()) / max(noise, 1e-300)),
                           # U-statistic noise ~ 1/N
                           "shots_for_snr3": (int(args.shots * 3 * noise / max(abs(mean_d.mean()), 1e-300))
                                              if abs(mean_d.mean()) > 0 else None),
                           "L0_mean": float(np.mean(L0)), "L0_std": float(np.std(L0, ddof=1)),
                           "s_per_sample_set": round(t_sample, 3)}
                rows.append(row)
                print(row, flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
