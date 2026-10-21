"""Timing of partial detection in the stage kernel (`slos.stage.StageKernel.apply`, route B, docs/DESIGN.md §2.4):
the blockwise update of a carried-photon density matrix by an emitter chain, on one device.

Model of a stage simulator's carry-over stage: an n-event caterpillar (chi = 2) emits into modes 0 .. 2n-1 of m; c
carried modes (the last c of m) hold the carried-in photons, the carried-in basis being every Fock state of 0 .. Nin
photons on them (B = sum_k C(c + k - 1, k)); the stage detects every mode but the carried ones, so the carried-out
basis lives on the same c modes. rho is a random density matrix on (input, register), [B chi, B chi].

Two sweeps, one CSV row per point (complex128, median of --repeats calls after two warm-ups, device synchronised):
  --sweep carry   : c carried modes x Nin carried-in photons, at n events (--n);
  --sweep photons : n events (photons emitted per stage) x Nin, at c carried modes (--carried).

    python3 bench/stage_carry_bench.py --device cuda --sweep carry --n 4 --cs 1,2,3,4 --Nins 1,2,3 --modes 16 --csv a.csv
    python3 bench/stage_carry_bench.py --device cuda --sweep photons --ns 2,4,6 --Nins 1,2 --carried 2 --modes 16 --csv b.csv

CSV: sweep, device, n, c, Nin, m, B (carried-in basis), S (full output patterns over all inputs), Qd (detected
patterns), Cmax (largest carried-out basis), kraus_ms (the full-output Kraus set alone), apply_ms (regroup + update,
Kraus included), regroup_share (1 - kraus/apply), peak_mib (device peak of apply; 0 on the CPU), deviation
(|sum_q Tr rho_q - Tr rho|), chunk (q per pass, from --chunk-gib), note. The CSV is rewritten after every point. A point
whose estimated memory exceeds --max-gib is skipped (a row with the note); a CUDA out-of-memory is caught and
recorded. A point slower than --max-s skips the larger Nin of the same (c, n).
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from photonic_engines import EmitterMPS, StageChain, spin_gate
from photonic_engines.slos.stage import StageKernel

C128 = torch.complex128


def haar(m, seed):
    """Haar-random unitary [m, m] complex128."""
    z = np.random.default_rng(seed).normal(size=(m, m, 2)) @ [1, 1j]
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / abs(np.diag(r))), dtype=C128)


def chain(n, m):
    """n-event caterpillar (chi = 2) on modes 0 .. 2n-1 of m."""
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.37 * i) for i in range(n)], m=m,
                                modes=[(2 * b, 2 * b + 1) for b in range(n)])
    return StageChain.from_mps(st)


def carried_basis(c, Nin, m):
    """Every Fock state of 0 .. Nin photons on the last c modes of m, as occupations over all m modes."""
    rows = []
    for k in range(Nin + 1):
        for occ in product(range(k + 1), repeat=c):
            if sum(occ) == k:
                rows.append([0] * (m - c) + list(occ))
    return np.asarray(rows, dtype=np.int64)


def random_rho(D, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    r = a @ a.conj().T
    return torch.as_tensor(r / np.trace(r).real, dtype=C128)


def timed(fn, dev, repeats):
    """Median wall time (s) of `repeats` calls after two warm-ups."""
    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
    fn(); fn(); sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def peak_mib(fn, dev):
    if not dev.startswith("cuda"):
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", choices=["carry", "photons"], required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ns", default="2,4,6")
    ap.add_argument("--cs", default="1,2,3,4")
    ap.add_argument("--carried", type=int, default=2)
    ap.add_argument("--Nins", default="1,2,3")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--max-gib", type=float, default=40.0, help="skip a point whose estimated device memory is above")
    ap.add_argument("--chunk-gib", type=float, default=4.0, help="memory of the blocks formed per pass of apply")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    dev, m, chi = args.device, args.modes, 2
    U = haar(m, 1).to(dev)
    rows = []
    if args.sweep == "carry":
        points = [(args.n, int(c), int(N)) for c in args.cs.split(",") for N in args.Nins.split(",")]
    else:
        points = [(int(n), args.carried, int(N)) for n in args.ns.split(",") for N in args.Nins.split(",")]
    fields = ["sweep", "device", "n", "c", "Nin", "m", "B", "S", "Qd", "Cmax", "kraus_ms", "apply_ms",
              "regroup_share", "peak_mib", "deviation", "chunk", "note"]
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    def save():
        """Rewrite the CSV after every point, so a timeout or a crash keeps the rows already measured."""
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    t_start = time.time()
    slow = set()                                     # (n, c) whose smaller Nin was over --max-s
    for n, c, Nin in points:
        if 2 * n + c > m:
            print(f"skip n={n} c={c}: m = {m} too small", flush=True)
            continue
        if (n, c) in slow:
            print(f"skip n={n} c={c} Nin={Nin}: a smaller Nin was over --max-s", flush=True)
            continue
        occ = carried_basis(c, Nin, m)
        B = occ.shape[0]
        counts = np.bincount(occ.sum(1))
        S = sum(int(b) * math.comb(m + N + n - 1, N + n) for N, b in enumerate(counts) if b)
        est_gib = 4 * S * chi * chi * 16 / 2 ** 30  # raw + physical Kraus sets, the propagation layers, the blocks
        base = {"sweep": args.sweep, "device": dev, "n": n, "c": c, "Nin": Nin, "m": m, "B": B, "S": S}
        if est_gib > args.max_gib:
            print(f"skip n={n} c={c} Nin={Nin}: ~{est_gib:.0f} GiB > --max-gib {args.max_gib}", flush=True)
            rows.append({**base, "note": f"skipped, ~{est_gib:.0f} GiB"})
            save()
            continue
        ch = chain(n, m)
        rho = random_rho(B * chi, seed=n + 10 * c + 100 * Nin).to(dev)
        try:
            sk = StageKernel(ch, inputs=occ, detected=range(m - c), device=dev)
            t0 = time.perf_counter()
            tab = sk._carry_tables()
            t_tab = time.perf_counter() - t0
            Qd = int(tab["detected"].shape[0])
            Cmax = max(C for C, _, _ in tab["classes"])
            chunk = max(1, int(args.chunk_gib * 2 ** 30 / (Cmax * chi * B * chi * 16 * 4)))
            t_k = timed(lambda: sk.kraus(U), dev, args.repeats)
            t_a = timed(lambda: sk.apply(U, rho, chunk=chunk), dev, args.repeats)
            mem = peak_mib(lambda: sk.apply(U, rho, chunk=chunk), dev)
            dev_err = abs(float(sk.apply(U, rho, chunk=chunk).deviation))
        except torch.cuda.OutOfMemoryError as e:
            print(f"n={n} c={c} Nin={Nin}: out of device memory ({str(e).splitlines()[0][:80]})", flush=True)
            rows.append({**base, "note": "out of memory"})
            save()
            sk = None
            torch.cuda.empty_cache()
            slow.add((n, c))
            continue
        rows.append({**base, "Qd": Qd, "Cmax": Cmax, "kraus_ms": 1e3 * t_k, "apply_ms": 1e3 * t_a,
                     "regroup_share": 1 - t_k / t_a, "peak_mib": mem, "deviation": dev_err, "chunk": chunk,
                     "note": f"tables {1e3 * t_tab:.0f} ms (once per kernel)"})
        save()
        print(f"[{time.time() - t_start:7.1f} s] n={n} c={c} Nin={Nin} B={B} S={S} Qd={Qd} Cmax={Cmax}: "
              f"kraus {1e3 * t_k:.2f} ms, apply {1e3 * t_a:.2f} ms, peak {mem:.0f} MiB, deviation {dev_err:.1e}",
              flush=True)
        sk = None
        if t_a > args.max_s:
            print(f"n={n} c={c}: apply over --max-s, larger Nin skipped", flush=True)
            slow.add((n, c))
    print("saved", args.csv, flush=True)


if __name__ == "__main__":
    main()
