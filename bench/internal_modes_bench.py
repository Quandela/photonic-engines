"""Internal photon modes (docs/SEQUENCE.md §14): cost of one bin's channel on the populated internal span (route F of
`slos.internal`) and of its outcome law by route F and by the pairing route P', against the expanded construction
(labels (l, c) on m K modes through the plain kernel, U (x) 1_K) and against the one-mode kernel of the same tick.

k emitters with qubit spins, L = 2 labels (dual rail, m = 2k), K internal modes, generic complete emitters (a random
isometry per bin). Per (k, K), all times uncached, device-synchronised, median of --reps after one warm-up: the fine
set size and the time of `kraus`; the outcome law of a random density matrix by F (build + apply) and by P'; the
expanded construction's outcome law (build + apply); the one-mode kernel's `kraus`; the agreement of the two channels
(max |superoperator difference| / chi); with --compress the compressed set size and its time (route F), and with
--route-r also by route R (no fine set) and its distance to F's compressed set; the route the dispatcher picks; and
the cost constants of `sequence_modes.COST` from this point (medians printed at the end). --emitters label: the
label-preserving emitters A[l, k] = f_l(k) |l><l| (low Choi rank, where route R pays off).
K = 1 runs through the internal-mode kernel (a ModeBinEmitter with one mode), not the plain one.

    python3 bench/internal_modes_bench.py --device cuda --k 2,3,4 --K 1,3,8 --reps 5 --csv internal_modes_bench.csv

Run from the repository root with this checkout's `photonic_engines` importable (installed, or the root on sys.path).
"""
import argparse
import csv
import statistics
import time

import numpy as np
import torch

from photonic_engines.sequence import BinEmitter, ModeBinEmitter, SequenceKernel, bin_emitter

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--k", default="2,3", help="emitters (photons) per bin, comma list")
ap.add_argument("--K", default="1,3,8", help="internal modes, comma list")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--compress", type=float, default=None, help="also time compress=rtol")
ap.add_argument("--max-gb", type=float, default=8.0, help="skip a configuration whose fine set exceeds this")
ap.add_argument("--no-expanded", action="store_true", help="skip the expanded construction (large K^k)")
ap.add_argument("--route-r", action="store_true", help="also time kraus(compress, route='R') (needs --compress)")
ap.add_argument("--emitters", default="generic", choices=("generic", "label"),
                help="generic isometries (full Choi rank) or label-preserving A[l, k] = f_l(k) |l><l| (low rank)")
ap.add_argument("--csv", default="internal_modes_bench.csv")
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
    if args.emitters == "label":                      # A[l, k] = f_l(k) |l><l|, |f_l| = 1: complete, low Choi rank
        f = rng.normal(size=(2, K)) + 1j * rng.normal(size=(2, K))
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        ops = np.zeros((1, 2, K, 2, 2), dtype=complex)
        for lab in range(2):
            ops[0, lab, :, lab, lab] = f[lab]
        return bin_emitter(ops, (2 * j, 2 * j + 1), [1, 0])
    z = rng.normal(size=(2 * K * 2, 2)) + 1j * rng.normal(size=(2 * K * 2, 2))
    q, _ = np.linalg.qr(z)
    return bin_emitter(q.reshape(1, 2, K, 2, 2), (2 * j, 2 * j + 1), [1, 0])


def expanded(ems, m, K):
    """The expanded construction: labels (l, c) -> mode c m + in(l), the plain kernel on m K modes."""
    out = []
    for e in ems:
        ops = e.ops if e.ops.ndim == 5 else e.ops[:, :, None]
        n, L, _, d, _ = ops.shape
        out.append(BinEmitter(ops.transpose(0, 2, 1, 3, 4).reshape(n, K * L, d, d),
                              tuple(c * m + e.modes[l] for c in range(K) for l in range(L)), e.s0))
    return SequenceKernel(out, m * K, device=dev)


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


def superops(K, coarse, S):
    """Per spatial outcome sum_q K_q (x) conj(K_q)."""
    chi = K.shape[-1]
    ops = torch.einsum("qab,qcd->qacbd", K, K.conj()).reshape(K.shape[0], chi * chi, chi * chi)
    return torch.zeros((S, chi * chi, chi * chi), dtype=K.dtype, device=K.device).index_add(0, coarse, ops)


def write_csv():
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
    plain = SequenceKernel([BinEmitter(emitter(1, j, 100 + j).ops, (2 * j, 2 * j + 1), [1, 0]) for j in range(k)], m,
                           device=dev)

    def plain_kraus(plain=plain, U=U):
        """One uncached one-mode Kraus build."""
        plain.clear_cache()
        plain.kraus(U, 0)
    t1 = timeit(plain_kraus)
    for K in (int(x) for x in args.K.split(",")):
        ems = [emitter(K, j, 10 * K + j) for j in range(k)]
        if K == 1:                                    # the one-mode tick through the internal-mode kernel (P' at K = 1)
            ems = [ModeBinEmitter(e.ops[:, :, None], e.modes, e.s0) for e in ems]
        mk = SequenceKernel(ems, m, device=dev, max_bytes=int(args.max_gb * 2 ** 30))
        row = {"k": k, "K": K, "chi": mk.chi, "S": mk.S, "t_one_mode_s": t1}
        g = torch.Generator().manual_seed(k)
        z = torch.randn(mk.chi, mk.chi, generator=g, dtype=torch.complex128)
        rho = (z @ z.conj().T / torch.trace(z @ z.conj().T)).to(dev)
        f_ok = True
        try:
            def native(mk=mk):
                """One uncached route-F build."""
                mk.clear_cache()
                mk.kraus(U, 0)
            row["t_native_s"] = timeit(native)

            def law_f(mk=mk, rho=rho):
                """Outcome law by route F, uncached (build + apply)."""
                mk.clear_cache()
                mk.outcome_law(U, 0, rho, route="F")
            row["t_law_F_s"] = timeit(law_f)
        except (NotImplementedError, torch.cuda.OutOfMemoryError, MemoryError) as err:
            row["note_F"] = str(err)[:80]
            f_ok = False
            mk.clear_cache()
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
        if f_ok:
            Kq, cq = mk.kraus(U, 0), mk.coarse(U, 0)
            row["Q_fine"] = Kq.shape[0]
            row["native_over_one_mode"] = row["t_native_s"] / t1
            if args.compress is not None:
                def comp(mk=mk):
                    """One uncached compressed build."""
                    mk.clear_cache()
                    mk.kraus(U, 0, compress=args.compress, route="F")
                try:
                    row["t_compressed_s"] = timeit(comp)
                    row["Q_compressed"] = mk.kraus(U, 0, compress=args.compress, route="F").shape[0]
                except (NotImplementedError, torch.cuda.OutOfMemoryError, MemoryError) as err:
                    row["note_compressed"] = str(err)[:80]
        if args.route_r and args.compress is not None:
            def comp_r(mk=mk):
                """One uncached compressed build by route R (no fine set)."""
                mk.clear_cache()
                mk.kraus(U, 0, compress=args.compress, route="R")
            try:
                row["t_compressed_R_s"] = timeit(comp_r)
                KR, cR = mk.kraus(U, 0, compress=args.compress, route="R"), mk.coarse(U, 0, compress=args.compress,
                                                                                     route="R")
                row["Q_compressed_R"] = KR.shape[0]
                if "Q_compressed" in row:
                    KF = mk.kraus(U, 0, compress=args.compress, route="F")
                    cF = mk.coarse(U, 0, compress=args.compress, route="F")
                    row["R_vs_F_diff_over_chi"] = float((superops(KR, cR, mk.S) - superops(KF, cF, mk.S)).abs().max()
                                                        ) / mk.chi
            except (torch.cuda.OutOfMemoryError, MemoryError, ValueError) as err:
                row["note_R"] = str(err)[:80]
            mk.clear_cache()
        try:
            row["t_law_P_s"] = timeit(lambda mk=mk, rho=rho: mk.outcome_law(U, 0, rho, route="P"))
        except (torch.cuda.OutOfMemoryError, MemoryError):
            row["note_P"] = "pairing: out of memory"
        mk.clear_cache()
        row["route_auto"] = mk._route(U, 0, mk.S, 1)
        if f_ok and not args.no_expanded and K > 1:
            ex = expanded(ems, m, K)
            Ub = torch.block_diag(*[U] * K)

            def exp_law(ex=ex, Ub=Ub, rho=rho):
                """Today's construction: the expanded Kraus set, built uncached, applied to rho."""
                ex.clear_cache()
                Kx = ex.kraus(Ub, 0)
                torch.einsum("qab,bc,qac->q", Kx, rho, Kx.conj())
            try:
                row["t_law_expanded_s"] = timeit(exp_law)
                occ = ex.patterns.reshape(-1, K, m).sum(1)
                ref = superops(ex.kraus(Ub, 0), mk.pattern_index(occ), mk.S)
                row["channel_diff_over_chi"] = float((superops(Kq, cq, mk.S) - ref).abs().max()) / mk.chi
            except (torch.cuda.OutOfMemoryError, MemoryError):
                row["note_expanded"] = "expanded: out of memory"
        # cost-model constants (sequence_modes.COST) from this point
        D, n = mk.chi, k
        if f_ok:
            row["a_F_build"] = row["t_native_s"] / (n * row["Q_fine"] * D ** 3)
            row["a_F_apply"] = max(row["t_law_F_s"] - row["t_native_s"], 0.0) / (row["Q_fine"] * D ** 3)
        if "t_law_P_s" in row:
            d = mk.dims[0]
            row["a_P"] = row["t_law_P_s"] / (mk.S * 4 ** (n - 1) * n * n * d * d * D * D)
        rows.append(row)
        print(row, flush=True)
        write_csv()                                   # after every row: a timeout or a host OOM keeps what is done

for key in ("a_F_build", "a_F_apply", "a_P"):
    vals = [r[key] for r in rows if key in r and r[key] > 0]
    if vals:
        print(f"cost constant {key} on {dev}: median {statistics.median(vals):.3e} s/unit over {len(vals)} points")
print("wrote", args.csv)
