"""Random-source-order SBS chain (clifford/sbs_rso.py) against Perceval's SLOS, with the incoherent twin as control.

For each (sources, modes, alpha, eta): Perceval SLOS on the
coherent product StateVector (lossy points: the Bernoulli-mask mixture of `product_sampler.lossy_source`, every
mask its own Perceval pass), the sampler's histogram, its TVD to Perceval, the multinomial shot-noise TVD of an
exact draw of the same size, and the TVD between the coherent law and its incoherent twin (Fock at eta' = eta
sin^2 alpha) -- the separation the check must resolve.

    python3 bench/sbs_chain_rso.py [--shots 60000] [--device cpu] [--csv out.csv]

--first-moment N:M: beyond strong-simulation reach, the mean occupation of every output mode against the coherent
prediction sum_i |U_ji|^2 (p - g^2) + |sum_i U_ji g|^2 (p = sin^2 alpha, g = cos alpha sin alpha) and against the
incoherent twin's sum_i |U_ji|^2 p; prints max |z| over the modes for both (statistical: the values move with the seed).

    python3 bench/sbs_chain_rso.py --first-moment 14:28 --shots 3000
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import math

import numpy as np
import torch


def perceval_probs(U: np.ndarray, modes: list[int], alpha: float) -> dict:
    """Output law of prod_i (cos a |0> + sin a |1>) on `modes` through U (Perceval SLOS)."""
    import perceval as pcvl
    m = U.shape[0]
    c, s = float(math.cos(alpha)), float(math.sin(alpha))
    sv = None
    for i in range(m):
        f = (pcvl.StateVector([0]) * c + pcvl.StateVector([1]) * s) if i in modes else pcvl.StateVector([0])
        sv = f if sv is None else sv * f
    sim = pcvl.Simulator(pcvl.SLOSBackend())
    sim.set_circuit(pcvl.Unitary(pcvl.Matrix(U)))
    return {tuple(o): float(p) for o, p in sim.probs(sv).items()}


def lossy_perceval_probs(U: np.ndarray, n: int, alpha: float, eta: float) -> dict:
    from photonic_engines.clifford.product_sampler import lossy_source
    w1, ap = lossy_source(alpha, eta)
    out = collections.defaultdict(float)
    for mask in itertools.product((0, 1), repeat=n):
        live = [i for i in range(n) if mask[i]]
        w = math.prod(w1 if b else 1 - w1 for b in mask)
        for o, p in perceval_probs(U, live, ap).items():
            out[o] += w * p
    return out


def incoherent_probs(U: np.ndarray, n: int, p: float) -> dict:
    import perceval as pcvl
    m = U.shape[0]
    sim = pcvl.Simulator(pcvl.SLOSBackend())
    sim.set_circuit(pcvl.Unitary(pcvl.Matrix(U)))
    out = collections.defaultdict(float)
    for mask in itertools.product((0, 1), repeat=n):
        w = math.prod(p if b else 1 - p for b in mask)
        for o, q in sim.probs(pcvl.BasicState(list(mask) + [0] * (m - n))).items():
            out[tuple(o)] += w * float(q)
    return out


def tvd(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def main():
    from scipy.stats import unitary_group

    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford.sbs_rso import sample
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=60000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--first-moment", default=None, metavar="N:M")
    ap.add_argument("--alpha", type=float, default=math.pi / 4, help="with --first-moment")
    ap.add_argument("--seed", type=int, default=0, help="with --first-moment")
    args = ap.parse_args()
    if args.first_moment:
        n, m = (int(v) for v in args.first_moment.split(":"))
        U = unitary_group.rvs(m, random_state=args.seed)
        occ = sample(ProductSuperposition.sbs(n, m, alpha=args.alpha), torch.tensor(U), args.shots,
                     generator=torch.Generator().manual_seed(args.seed), device=args.device).double().cpu().numpy()
        p, g = math.sin(args.alpha) ** 2, math.cos(args.alpha) * math.sin(args.alpha)
        w = np.abs(U[:, :n]) ** 2
        coh = w.sum(1) * (p - g * g) + np.abs(U[:, :n].sum(1) * g) ** 2
        inc = w.sum(1) * p
        se = occ.std(0, ddof=1) / math.sqrt(args.shots)
        se[se == 0] = np.nan                                  # a mode never occupied: no z (nanmax below)
        z_coh, z_inc = (occ.mean(0) - coh) / se, (occ.mean(0) - inc) / se
        print(f"n = {n}, m = {m}, alpha = {args.alpha:.4f}, {args.shots} shots, seed {args.seed}: mean occupation per "
              f"mode, max |z| vs coherent {np.nanmax(np.abs(z_coh)):.2f}, vs incoherent {np.nanmax(np.abs(z_inc)):.2f}")
        return
    rng = np.random.default_rng(1)
    rows = []
    for n, m, alpha, eta in [(4, 8, math.pi / 4, 1.0), (6, 12, math.pi / 4, 1.0), (5, 6, 1.1, 1.0),
                             (7, 7, math.pi / 4, 1.0), (5, 8, math.pi / 4, 0.7)]:
        U = unitary_group.rvs(m, random_state=n + m)
        ref = perceval_probs(U, list(range(n)), alpha) if eta >= 1 else lossy_perceval_probs(U, n, alpha, eta)
        inc = incoherent_probs(U, n, eta * math.sin(alpha) ** 2)
        st = ProductSuperposition.sbs(n, m, alpha=alpha)
        occ = sample(st, torch.tensor(U), args.shots, generator=torch.Generator().manual_seed(n), device=args.device,
                     loss=None if eta >= 1 else eta)
        emp = collections.Counter(tuple(r) for r in occ.tolist())
        emp = {k: v / args.shots for k, v in emp.items()}
        keys = list(ref)
        pr = np.array([ref[k] for k in keys])
        pr = pr / pr.sum()
        noise = float(np.mean([0.5 * np.abs(rng.multinomial(args.shots, pr) / args.shots - pr).sum() for _ in range(5)]))
        row = {"n": n, "m": m, "alpha": round(alpha, 6), "eta": eta, "shots": args.shots, "tvd_vs_perceval": tvd(emp, ref),
               "shot_noise_tvd": noise, "tvd_coherent_vs_incoherent": tvd(ref, inc)}
        rows.append(row)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
