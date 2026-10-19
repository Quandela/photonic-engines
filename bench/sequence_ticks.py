"""Complexity of the spin register of a static k-emitter machine, tick by tick (docs/SEQUENCE.md
§9): k dual-rail emitters into one m = 2k-mode interferometer with no delay line, sampled by
pure trajectories with the label-diagonal sampler (`SequenceKernel.sample_diag`, no Kraus set,
O(k 3^k) per tick). After every tick the register psi [shots, 2^k] is cut in the middle and its
Schmidt spectrum recorded: entanglement entropy (bits, max k/2), exact rank (max 2^{k/2}), bond
dimension holding 99.9 % of the weight — and the wall time of the tick (the sampling cost).

    python3 bench/sequence_ticks.py --device cuda --k 6,8,10,12,14 --ticks 8 --shots 256 --unitary haar
    python3 bench/sequence_ticks.py --device cuda --k 8,12 --ticks 12 --shots 256 --unitary band:2
    python3 bench/plot_sequence_ticks.py results/<stamp>/sequence_ticks_*.csv    # -> img/sequence_ticks.png

--unitary haar          Haar-random U on the 2k modes (every emitter interferes with every other)
--unitary band:r        brickwork of 2r layers of random 2x2 beam splitters on neighbouring modes
                        (emitter j's photon reaches the modes of emitters within r of j)
--method glynn|glynn_full|laplace   the label-amplitude pass (collapsed Glynn-GEMM since 2026-09-26; the
                        uncollapsed sum and the Laplace recurrence stay for A/B); the CSV carries it
--fock-backend triton|cuda, --draw host|device   the tick's floor (docs/SEQUENCE.md §12b); --split records
                        the synchronised seconds of every phase of a tick (apply / draw / fock / amp / update)
Writes $RESULTS_DIR/sequence_ticks_<unitary>.csv, one row per (k, tick).
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


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=torch.complex128)


def banded(m, layers, seed):
    rng = np.random.default_rng(seed)
    U = np.eye(m, dtype=complex)
    for layer in range(layers):
        for i in range(layer % 2, m - 1, 2):
            z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            q, _ = np.linalg.qr(z)
            V = np.eye(m, dtype=complex)
            V[i:i + 2, i:i + 2] = q
            U = V @ U
    return torch.as_tensor(U)


def schmidt(psi, k):
    """Middle-cut Schmidt weights [shots, 2^{k//2}] of psi [shots, 2^k]."""
    a = 2 ** (k // 2)
    s = torch.linalg.svdvals(psi.reshape(psi.shape[0], a, -1)) ** 2
    return torch.sort(s, 1, descending=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", default="4,6,8,10")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--shots", type=int, default=128)
    ap.add_argument("--unitary", default="haar", help="haar or band:r")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-gb", type=float, default=60.0, help="skip a k whose label-amplitude pass exceeds this per shot")
    ap.add_argument("--method", default="glynn", choices=["glynn", "glynn_full", "laplace"],
                    help="label-amplitude evaluation: the collapsed Glynn-GEMM (docs/SEQUENCE.md §12b, default), the "
                         "uncollapsed one of §12 (A/B reference) or the Laplace recurrence")
    ap.add_argument("--fock-backend", default="triton", choices=["triton", "cuda"],
                    help="kernel of the per-tick Fock sample on cuda (cuda = thread-per-lane cofactor kernel, needs cupy)")
    ap.add_argument("--draw", default="auto", choices=["auto", "host", "device"],
                    help="where the label word is drawn (auto = device on cuda, the default since 2026-09-26)")
    ap.add_argument("--split", action="store_true", help="record the synchronised per-phase seconds of every tick "
                    "(apply / draw / fock / amp / update) as CSV columns")
    ap.add_argument("--tag", default="", help="appended to the CSV name (under $RESULTS_DIR), e.g. the k list of a job")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    tag = args.unitary.replace(":", "") + ("" if args.method == "glynn" else f"_{args.method}") \
        + ("" if args.fock_backend == "triton" else "_cuda") + ("" if args.draw == "auto" else f"_{args.draw}draw") \
        + (f"_{args.tag}" if args.tag else "")
    out = args.out or os.path.join(os.environ.get("RESULTS_DIR", "results"), f"sequence_ticks_{tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for k in [int(x) for x in args.k.split(",")]:
        m = 2 * k
        peak_gb = 2 * max(math.comb(k, j + 1) * 2 ** (j + 1) for j in range(k)) * 16 / 2 ** 30   # label_amplitudes, per stream
        if args.method == "laplace" and peak_gb > args.max_gb:                   # the Glynn passes are chunked
            print(f"k={k}: label-amplitude pass needs {peak_gb:.1f} GB per shot, skipped", flush=True)
            continue
        U = haar(m, args.seed + k) if args.unitary == "haar" else banded(m, 2 * int(args.unitary.split(":")[1]), args.seed + k)
        U = U.to(dev)
        ems = [BinEmitter.caterpillar(args.ticks, [spin_gate(math.pi / 2, 0)] * args.ticks, (2 * j, 2 * j + 1)) for j in range(k)]
        sk = SequenceKernel(ems, m, device=dev)
        rec = {}
        t_last = [time.time()]
        T = {} if args.split else None
        T_prev = [{}]

        def hook(b, psi, counts):
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t_last[0]
            split = {}
            if T is not None:
                split = {f"{key}_s": round(T[key] - T_prev[0].get(key, 0.0), 4) for key in T if key != "hook"}
                T_prev[0] = dict(T)
            s = schmidt(psi, k)
            ent = -(s * torch.log2(s.clamp_min(1e-300))).sum(1)
            rank = (s > 1e-12).sum(1).float()
            bond = ((torch.cumsum(s, 1) < 0.999).sum(1) + 1).float()
            rec[b] = {"tick_s": round(dt, 4), "ms_per_tick_shot": round(1e3 * dt / psi.shape[0], 4),
                          "entropy": round(float(ent.mean()), 4), "entropy_std": round(float(ent.std()), 4),
                          "rank": round(float(rank.mean()), 2), "bond999": round(float(bond.mean()), 2), **split}
            t_last[0] = time.time()

        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # warm-up (Triton compile of the k-photon Fock sampler, subset tables), not timed
        opts = {"amp_method": args.method, "fock_backend": args.fock_backend, "draw": args.draw}
        sk.sample_diag(U, min(args.shots, 8), generator=torch.Generator().manual_seed(0), **opts)
        t_last[0] = time.time()
        sk.sample_diag(U, args.shots, generator=torch.Generator().manual_seed(1), hook=hook, timing=T, **opts)
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if dev == "cuda" else None
        for b in range(args.ticks):
            row = dict(k=k, m=m, unitary=args.unitary, shots=args.shots, tick=b + 1, max_entropy=k // 2,
                       max_rank=2 ** (k // 2), **rec[b], peak_GB=peak, method=args.method,
                       fock_backend=args.fock_backend, draw=args.draw)
            rows.append(row)
            print(row, flush=True)
    keys = list(rows[0].keys()) if rows else []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
