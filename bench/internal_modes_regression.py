"""Internal photon modes, regression bench over the tick shapes of a spin-photon processor (docs/SEQUENCE.md §14;
design note §6a (3)): the route the dispatcher picks must not be slower than today's expanded construction on any
such tick.

Its time-mode ticks are pair blocks: k = 2 dual-rail qubit emitters on m = 4 modes, one bin, K = 3 (bare
trion, 150 mT), 4 (cavity), 7 (two Overhauser samples), 8 (the n-dot saturation), and unequal K (K, K - 1).
Emitters here are generic (a random isometry: full Choi rank, the costliest case); the `lost` operators of
physical emitters are not modelled here. U: Haar on 4 modes.

Per (K pair, batch B), all times uncached, device-synchronised, median of --reps after one warm-up, native (route
"auto") against expanded (labels (l, c) on m K modes through the plain kernel, U (x) 1_K, outcomes summed over c):
- channel: `kraus(compress=1e-12)` against the expanded set + the same per-outcome compression (what such a processor
  builds per attempt);
- law: `outcome_law` of a batch of B density matrices against the expanded set applied to them;
- class: `condition` on a class of half the outcomes (a "repeat" class) against the expanded sum over the class;
and `ok` = native <= expanded x (1 + --noise) on all three; the script exits 1 when a row is not ok.

    python3 bench/internal_modes_regression.py --device cpu --B 1,16,64 --reps 7 --csv internal_modes_regression.csv

Run from the repository root with this checkout's `photonic_engines` importable (installed, or the root on sys.path).
"""
import argparse
import csv
import statistics
import sys
import time

import numpy as np
import torch

from photonic_engines.sequence import BinEmitter, SequenceKernel, bin_emitter
from photonic_engines.slos.internal import compress_by_outcome

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--K", default="3,4,7,8,3:2,8:7", help="internal modes per pair: K (both emitters) or Ka:Kb")
ap.add_argument("--B", default="1,16,64", help="density-matrix batch sizes")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--noise", type=float, default=0.1, help="relative timing noise allowed in the ok column")
ap.add_argument("--csv", default="internal_modes_regression.csv")
args = ap.parse_args()
dev = args.device
m, rtol = 4, 1e-12


def sync():
    """Wait for the device."""
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn):
    """Median seconds of --reps calls after one warm-up."""
    fn()
    sync()
    ts = []
    for _ in range(args.reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def emitter(K, j, seed):
    """Dual-rail qubit emitter on modes (2j, 2j + 1), K internal modes, one bin: a random isometry [2 K 2, 2]."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(2 * K * 2, 2)) + 1j * rng.normal(size=(2 * K * 2, 2))
    q, _ = np.linalg.qr(z)
    return bin_emitter(q.reshape(1, 2, K, 2, 2), (2 * j, 2 * j + 1), [1, 0])


def expanded(ems, K):
    """The expanded construction on the shared K (fewer modes zero-padded: the plain kernel enumerates its patterns
    over all m K modes either way, so padding leaves the construction and its cost unchanged): labels (l, c) -> mode
    c m + in(l)."""
    out = []
    for e in ems:
        ops = np.zeros((*e.ops.shape[:2], K, *e.ops.shape[3:]), dtype=e.ops.dtype)
        ops[:, :, :e.ops.shape[2]] = e.ops
        n, L, _, d, _ = ops.shape
        out.append(BinEmitter(ops.transpose(0, 2, 1, 3, 4).reshape(n, K * L, d, d),
                              tuple(c * m + e.modes[l] for c in range(K) for l in range(L)), e.s0))
    return SequenceKernel(out, m * K, device=dev)


def haar(n, seed):
    """Haar-random unitary [n, n]."""
    z = np.random.default_rng(seed).normal(size=(n, n, 2)) @ [1, 1j]
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / abs(np.diag(r))), dtype=torch.complex128, device=dev)


def rhos(D, B, seed):
    """B random density matrices [B, D, D]."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, D, generator=g, dtype=torch.complex128)
    r = z @ z.conj().transpose(1, 2)
    return (r / torch.einsum("bii->b", r).view(-1, 1, 1)).to(dev)


rows = []
U = haar(m, 1)
for spec in args.K.split(","):
    Ka, Kb = (int(x) for x in spec.split(":")) if ":" in spec else (int(spec),) * 2
    K = max(Ka, Kb)
    ems = [emitter(Ka, 0, 10 * K), emitter(Kb, 1, 10 * K + 1)]
    mk = SequenceKernel(ems, m, device=dev)
    ex = expanded(ems, K)
    Ub = torch.block_diag(*[U] * K)
    S = mk.S
    occ = ex.patterns.reshape(-1, K, m).sum(1)
    ex_coarse = mk.pattern_index(occ).to(dev)
    cls = torch.arange(S // 2, device=dev)
    in_cls = torch.isin(ex_coarse, cls)

    def nat_channel():
        mk.clear_cache()
        mk.kraus(U, 0, compress=rtol)

    def exp_channel():
        ex.clear_cache()
        compress_by_outcome(ex.kraus(Ub, 0), ex_coarse, rtol)

    t_nat_ch, t_exp_ch = timeit(nat_channel), timeit(exp_channel)
    q_nat = mk.kraus(U, 0, compress=rtol).shape[0]
    for B in (int(x) for x in args.B.split(",")):
        rho = rhos(mk.chi, B, B + K)

        def nat_law(rho=rho):
            mk.clear_cache()
            mk.outcome_law(U, 0, rho)

        def exp_law(rho=rho):
            ex.clear_cache()
            Kx = ex.kraus(Ub, 0)
            p = torch.einsum("qab,nbc,qac->nq", Kx, rho, Kx.conj()).real
            torch.zeros((B, S), dtype=p.dtype, device=dev).index_add_(1, ex_coarse, p)

        def nat_cls(rho=rho):
            mk.clear_cache()
            mk.condition(U, 0, rho, cls)

        def exp_cls(rho=rho):
            ex.clear_cache()
            Kx = ex.kraus(Ub, 0)[in_cls]
            torch.einsum("qab,nbc,qdc->nad", Kx, rho, Kx.conj())

        row = {"K": spec, "chi": mk.chi, "S": S, "B": B, "Q_compressed": q_nat,
               "t_channel_native_s": t_nat_ch, "t_channel_expanded_s": t_exp_ch,
               "t_law_native_s": timeit(nat_law), "t_law_expanded_s": timeit(exp_law),
               "t_class_native_s": timeit(nat_cls), "t_class_expanded_s": timeit(exp_cls),
               "route_law": mk._route(U, 0, S, B), "route_class": mk._route(U, 0, len(cls), B)}
        # agreement of the two constructions on this batch (law and class), max abs difference
        Kx = ex.kraus(Ub, 0)
        p_ex = torch.zeros((B, S), dtype=torch.float64, device=dev).index_add_(
            1, ex_coarse, torch.einsum("qab,nbc,qac->nq", Kx, rho, Kx.conj()).real)
        row["law_diff"] = float((mk.outcome_law(U, 0, rho) - p_ex).abs().max())
        c_ex = torch.einsum("qab,nbc,qdc->nad", Kx[in_cls], rho, Kx[in_cls].conj())
        row["class_diff"] = float((mk.condition(U, 0, rho, cls) - c_ex).abs().max())
        row["ok"] = all(row[f"t_{w}_native_s"] <= row[f"t_{w}_expanded_s"] * (1 + args.noise)
                        for w in ("channel", "law", "class"))
        rows.append(row)
        print(row, flush=True)
        keys = []
        for r in rows:
            keys += [x for x in r if x not in keys]
        with open(args.csv, "w", newline="") as f:     # after every row
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

bad = [(r["K"], r["B"]) for r in rows if not r["ok"]]
print(f"regression check on {dev}: {len(rows) - len(bad)}/{len(rows)} rows ok" + (f"; slower: {bad}" if bad else ""))
sys.exit(1 if bad else 0)                             # a slower row fails the job (no-regression rule)
