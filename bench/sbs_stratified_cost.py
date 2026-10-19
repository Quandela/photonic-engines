"""Seconds per sample of the rso SBS sampler at a lossy point, by two estimators side by side (P, 10-06, after the
eta = 0.7 bracket of runs 20261006_012054 / 20261006_132352 rested on a tail-dominated shot mean).

  raw         the plain mean over shots, wall time / shots per batch, mean and standard error over batches;
  stratified  sum_k P(k) E[cost | k]: P(k) exact (k ~ the detected count, n_live ~ Bin(n, w1), k | n_live ~
              Bin(n_live, sin^2 alpha'), `sbs_run_cost.t_of_S`'s joint law), E[cost | k] measured per stratum from
              `sbs_rso.sample(stats=...)` passes (chain seconds / streams per (n_live, k), averaged over n_live with
              the exact conditional weights of the drawn n_live). Strata above the largest drawn k are priced by a
              geometric fit of log E[cost | k] over the top drawn strata and reported as the extrapolated share.

The stratified estimate prices the chain passes only (the `stats` timings); the raw one is wall time and also carries
the per-call work outside the passes (loss / order draws, lone photons, scatter). `pass_share` = pass seconds / wall
seconds says how much of the raw mean the strata see.

Standard errors: bootstrap over batches (each batch = fresh seeds). Same sampler call as find_threshold's sbs/rso
branch (backend / square / draw flags). Not a frontier statistic by itself: it prices given (n, m, eta, alpha).

    python3 bench/sbs_stratified_cost.py --points 0.7:64,0.7:65 --ratio 2 --shots 512 --batch 64 --device cuda \
        --rso-backend cuda --rso-square kernel --rso-draw device --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbs_run_cost import photon_law


def haar(m: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return (q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)).to(dev)


def joint_law(n: int, eta: float, alpha: float) -> np.ndarray:
    """joint[l, k] = P(n_live = l, k detected): the law sbs_rso.sample draws (thin-first loss)."""
    c, s = math.cos(alpha), math.sin(alpha)
    w1 = c * c + eta * s * s
    s2 = (eta * s * s / w1) if w1 else 0.0
    live = photon_law(n, w1)
    joint = np.zeros((n + 1, n + 1))
    for l in range(n + 1):
        if live[l] > 0:
            joint[l, :l + 1] = live[l] * photon_law(l, s2)
    return joint


def stratified(joint: np.ndarray, cost: dict, n_fit: int = 4) -> tuple:
    """(estimate, covered P mass, extrapolated cost share). cost: (l, k) -> seconds per shot of that stratum."""
    pk = joint.sum(0)
    by_k = {}
    for k in sorted({k for _, k in cost}):
        ls = [l for l, kk in cost if kk == k]
        w = np.array([joint[l, k] for l in ls])
        if w.sum() > 0:
            by_k[k] = float(np.dot(w, [cost[(l, k)] for l in ls]) / w.sum())
    if not by_k:
        return float("nan"), 0.0, float("nan")
    covered = sum(pk[k] for k in by_k)
    est = sum(pk[k] * c for k, c in by_k.items())
    ks = sorted(k for k in by_k if by_k[k] > 0)[-n_fit:]
    ext = 0.0
    if len(ks) >= 2:
        slope, icpt = np.polyfit(ks, np.log([by_k[k] for k in ks]), 1)
        for k in range(max(by_k) + 1, len(pk)):
            ext += pk[k] * math.exp(icpt + slope * k)
    # strata below the largest drawn k but never drawn (rare gaps): priced like their nearest drawn neighbour
    for k in range(1, max(by_k)):
        if k not in by_k and pk[k] > 0:
            near = min(by_k, key=lambda kk: abs(kk - k))
            est += pk[k] * by_k[near]
    tot = est + ext
    if len(ks) < 2 and max(by_k) < len(pk) - 1 and pk[max(by_k) + 1:].sum() > 0:
        return tot, float(covered), float("nan")             # tail above the drawn strata left unpriced: say so (R)
    return tot, float(covered), (ext / tot if tot else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True, help="eta:n,eta:n,... (eta = 1 allowed)")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=math.pi / 4)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7, help="base seed; batch b of point n uses seed + 1000 n + b")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rso-backend", default="auto")
    ap.add_argument("--rso-square", default="auto")
    ap.add_argument("--rso-draw", default="host")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford.sbs_rso import sample
    dev = torch.device(args.device)
    kw = {"backend": args.rso_backend, "square": args.rso_square, "draw": args.rso_draw, "device": dev}
    rows = []
    for spec in args.points.split(","):
        eta, n = float(spec.split(":")[0]), int(spec.split(":")[1])
        m = max(2, round(args.ratio * n))
        U = haar(m, args.seed + n, dev)
        st = ProductSuperposition.sbs(n, m, alpha=args.alpha)
        loss = None if eta >= 1 else eta
        sample(st, U, 2, generator=torch.Generator().manual_seed(1), loss=loss, **kw)          # warm-up, not timed
        nb = max(1, args.shots // args.batch)
        walls, per_batch, pass_sec = [], [], 0.0
        for b in range(nb):
            stats: dict = {}
            g = torch.Generator().manual_seed(args.seed + 1000 * n + b)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            sample(st, U, args.batch, generator=g, loss=loss, stats=stats, **kw)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) / args.batch)
            acc = defaultdict(lambda: [0.0, 0])
            for n_l, k, streams, sec in stats.get("passes", []):
                acc[(n_l, k)][0] += sec
                acc[(n_l, k)][1] += streams
                pass_sec += sec
            per_batch.append(dict(acc))
        joint = joint_law(n, eta, args.alpha)

        def pool(batches):
            tot = defaultdict(lambda: [0.0, 0])
            for d in batches:
                for key, (s, c) in d.items():
                    tot[key][0] += s
                    tot[key][1] += c
            return {key: s / c for key, (s, c) in tot.items() if c}

        strat, cov, ext = stratified(joint, pool(per_batch))
        rng = np.random.default_rng(args.seed)
        boots_raw, boots_str = [], []
        for _ in range(args.boot):
            idx = rng.integers(0, nb, nb)
            boots_raw.append(float(np.mean([walls[i] for i in idx])))
            boots_str.append(stratified(joint, pool([per_batch[i] for i in idx]))[0])
        raw = float(np.mean(walls))
        row = {"eta": eta, "n": n, "m": m, "alpha": round(args.alpha, 6), "shots": nb * args.batch,
               "raw_s": raw, "raw_se": float(np.std(boots_raw)), "strat_s": strat, "strat_se": float(np.nanstd(boots_str)),
               "covered_P": round(cov, 6), "extrapolated_share": round(ext, 4),
               "pass_share": round(pass_sec / (sum(walls) * args.batch), 4) if walls else float("nan"),
               "k_max_drawn": max((k for d in per_batch for _, k in d), default=0)}
        rows.append(row)
        print(f"eta={eta:g} n={n} m={m}: raw {raw:.4f} +- {row['raw_se']:.4f} s/sample, stratified {strat:.4f} +- "
              f"{row['strat_se']:.4f} s/sample (covered P = {cov:.4f}, extrapolated share {ext:.1%}, k_max drawn "
              f"{row['k_max_drawn']}, {nb * args.batch} shots)", flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
