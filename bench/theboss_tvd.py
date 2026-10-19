"""theboss against this engine at the DISTRIBUTION level: the paper's third-party correctness line.

`bench/theboss_workcount.py` compared the two samplers' work (addends per sample); this compares
their output. theboss (PyPI, Apache-2.0, Rybotycki) is the independent exact Fock sampler -- C&C
2020's submatrices strategy, single-threaded Python -- so agreement between its samples and ours on
the same (n, m, U) is evidence that neither sampler's law is wrong, with the brute-force law as the
referee where it can be enumerated.

Three legs, each reported at the shot-noise floor of tests/clifford/test_product_sampler.py
(expected TVD of a faithful multinomial sample, 0.5 sqrt(2/pi) sum_q sqrt(q (1 - q) / S)):

  exact    small instances (C(m+n-1, n) patterns enumerable): theboss, and this engine's CPU chain
           (`fock_sample_reference`, algo 2020), each against the brute-force law, and against each
           other (two-sample TVD, pooled floor).           CPU, minutes.
  theboss  larger instances (n = 10-14): draw theboss samples, save them (npz).   CPU, hours.
  ours     the same instances with the production sampler (`fock_sample`, backend cuda), save them.
           GPU, seconds.
  compare  a theboss file against an ours file: every sampler against the EXACT one- and two-mode
           statistics (<n_i> = sum_k |U_ik|^2; C_ij = |(U_T U_T^+)_ij|^2 - 2 sum_k |U_ik|^2 |U_jk|^2 for
           i != j, the Walschaers et al. correlator for indistinguishable single photons -- checked
           against brute force by --selftest) as z-scores, and the two samplers against each other on
           the collision-type histogram (the sorted multiset of occupations: the statistic that
           distinguishes bosons from distinguishable particles), two-sample TVD at its floor.

    python3 bench/theboss_tvd.py selftest
    python3 bench/theboss_tvd.py exact   --cases 4:8,5:10,6:6,7:7 --shots 100000 --csv results/keep/theboss_tvd_exact.csv
    python3 bench/theboss_tvd.py theboss --cases 10:20,12:24,14:28 --shots 10000 --out results/theboss_tvd
    python3 bench/theboss_tvd.py ours    --cases 10:20,12:24,14:28 --shots 100000 --device cuda --out $RESULTS_DIR
    python3 bench/theboss_tvd.py compare --theboss results/theboss_tvd --ours results/<stamp> --csv results/keep/theboss_tvd_large.csv

Cases are n:m. The Haar unitary of a case is `haar(m, seed)` exactly as bench/find_threshold.py
draws it (numpy default_rng(seed), QR with the phase fix), seed = --seed + n, and every npz stores
its U so `compare` checks that both sides sampled the same matrix. theboss draws from numpy's global
legacy RNG and is seeded per case. Versions and the git HEAD go in the CSV header.
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# ------------------------------------------------------------------------------- instances


def haar(m, seed):
    """bench/find_threshold.py's draw, verbatim (numpy, not torch, so the GPU host and a CPU box agree)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def parse_cases(s):
    out = []
    for c in s.split(","):
        n, m = c.split(":")
        out.append((int(n), int(m)))
    return out


def versions():
    v = {"numpy": np.__version__, "python": sys.version.split()[0]}
    for mod in ("theboss", "torch"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "absent"
    if v["theboss"] == "absent":
        try:
            from importlib.metadata import version
            v["theboss"] = version("theboss")
        except Exception:
            pass
    try:
        v["head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL,
                                            cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        v["head"] = "?"
    return v


# ---------------------------------------------------------------------------- brute force
def perm(A):
    k = A.shape[0]
    if k == 0:
        return 1.0 + 0j
    tot = 0j
    for msk in range(1, 1 << k):
        cols = [j for j in range(k) if msk >> j & 1]
        tot += (-1) ** len(cols) * np.prod(A[:, cols].sum(1))
    return tot * (-1) ** k


def exact_distribution(U, n, m):
    """Pattern -> probability for single photons in input modes 0..n-1 (the sampler convention:
    amplitude = perm(U[rows][:, cols]) / sqrt(prod t!))."""
    out = {}
    cols = list(range(n))
    for combo in itertools.combinations_with_replacement(range(m), n):
        t = [0] * m
        for c in combo:
            t[c] += 1
        rows = [r for r in range(m) for _ in range(t[r])]
        out[tuple(t)] = abs(perm(U[np.ix_(rows, cols)])) ** 2 / math.prod(math.factorial(x) for x in t)
    z = sum(out.values())
    assert abs(z - 1) < 1e-9, f"brute-force law sums to {z}"
    return out


def n_patterns(n, m):
    return math.comb(m + n - 1, n)


# -------------------------------------------------------------------------- TVD and floors
def tvd_and_floor(occ, ref, shots):
    """tests/clifford/test_product_sampler.py's statistic: TVD of the empirical law against `ref`
    and the expected TVD of a faithful multinomial sample of `shots`."""
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / shots
    tvd = 0.5 * sum(abs(emp.get(k, 0.0) - ref.get(k, 0.0)) for k in set(emp) | set(ref))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in ref.values())
    return tvd, floor


def two_sample_tvd(keys_a, keys_b):
    """TVD between two empirical laws over hashable keys, and its floor: the expected TVD of two
    faithful samples of the pooled law, 0.5 sqrt(2/pi) sum_q sqrt(q (1 - q) (1/S_a + 1/S_b))."""
    Sa, Sb = len(keys_a), len(keys_b)
    ca, cb = {}, {}
    for k in keys_a:
        ca[k] = ca.get(k, 0) + 1
    for k in keys_b:
        cb[k] = cb.get(k, 0) + 1
    keys = set(ca) | set(cb)
    tvd = 0.5 * sum(abs(ca.get(k, 0) / Sa - cb.get(k, 0) / Sb) for k in keys)
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(
        math.sqrt(q * (1 - q) * (1 / Sa + 1 / Sb)) for q in ((ca.get(k, 0) + cb.get(k, 0)) / (Sa + Sb) for k in keys))
    return tvd, floor, len(keys)


def collision_type(row):
    """The sorted multiset of non-zero occupations, e.g. (3, 1, 1, 1): bunching without mode labels."""
    return tuple(sorted((int(x) for x in row if x), reverse=True))


# ------------------------------------------------------------------- exact low-order moments
def exact_moments(U, n):
    """<n_i> and C_ij = <n_i n_j> - <n_i><n_j> (i != j) for single photons in input modes 0..n-1,
    from a_i = sum_k U_ik b_k: <n_i> = sum_k |U_ik|^2; <n_i n_j> = sum_{k != l} (|U_ik|^2 |U_jl|^2 +
    U*_ik U_il U*_jl U_jk), so C_ij = |(U_T U_T^+)_ji|^2 - 2 sum_k |U_ik|^2 |U_jk|^2.  Verified by
    `selftest` against the brute-force law."""
    UT = U[:, :n]
    P = np.abs(UT) ** 2
    mean = P.sum(1)
    G = UT @ UT.conj().T
    C = np.abs(G) ** 2 - 2 * (P @ P.T)
    np.fill_diagonal(C, np.nan)
    return mean, C


def moment_zscores(occ, U, n):
    """z-scores of the sample means of n_i and of n_i n_j - <n_i><n_j> against the exact values, the
    standard error taken from the samples themselves (so no exact fourth moments are needed)."""
    occ = np.asarray(occ, dtype=np.float64)
    S, m = occ.shape
    mean, C = exact_moments(U, n)
    z1 = (occ.mean(0) - mean) / (occ.std(0, ddof=1) / math.sqrt(S) + 1e-300)
    prod = np.einsum("si,sj->sij", occ, occ)
    z2 = (prod.mean(0) - mean[:, None] * mean[None, :] - C) / (prod.std(0, ddof=1) / math.sqrt(S) + 1e-300)
    iu = np.triu_indices(m, 1)
    z2 = z2[iu]
    return z1, z2


def z_summary(z):
    z = np.asarray(z)
    return {"rms": float(np.sqrt(np.mean(z ** 2))), "max": float(np.max(np.abs(z))),
            "frac_gt2": float(np.mean(np.abs(z) > 2)), "count": int(z.size)}


# ---------------------------------------------------------------------------- the samplers
def theboss_sample(U, n, m, shots, seed):
    import numpy.random as npr
    from theboss.boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_factory import (
        BSPermanentCalculatorFactory,
    )
    from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
        GeneralizedCliffordsBSimulationStrategy as GCB,
    )
    npr.seed(seed)  # noqa: NPY002 -- theboss draws from numpy's global legacy RNG
    inp = [1] * n + [0] * (m - n)
    pc = BSPermanentCalculatorFactory(U, inp, None).generate_calculator()
    strat = GCB(pc)
    out = np.zeros((shots, m), dtype=np.int64)
    t0 = time.time()
    for s in range(shots):
        out[s] = strat.simulate(inp, 1)[0]
        if s and s % 2000 == 0 and shots >= 10000:
            print(f"    theboss n = {n}: {s}/{shots} in {time.time() - t0:.0f} s", flush=True)
    return out, time.time() - t0


def ours_cpu_sample(U, n, m, shots, seed):
    import torch

    from photonic_engines.clifford.variants import fock_sample_reference
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    occ = fock_sample_reference(torch.tensor(U, dtype=torch.complex128), list(range(n)), shots, algo="2020", generator=g)
    return occ.numpy(), time.time() - t0


def ours_gpu_sample(U, n, m, shots, seed, device="cuda", backend="cuda"):
    import torch

    from photonic_engines.clifford import fock_sample
    dev = torch.device(device)
    Ut = torch.tensor(U, dtype=torch.complex128, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    occ = fock_sample(Ut, list(range(n)), shots, generator=g, backend=backend)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return occ.cpu().numpy(), time.time() - t0


# -------------------------------------------------------------------------------- commands
def csv_header(f, cmd, extra=None):
    v = versions()
    f.write(f"# bench/theboss_tvd.py {cmd}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("# " + "  ".join(f"{k}={val}" for k, val in v.items()) + (f"  {extra}" if extra else "") + "\n")


def cmd_selftest(a):
    rng_cases = [(2, 4), (3, 5), (3, 6), (4, 6)]
    worst = 0.0
    for n, m in rng_cases:
        U = haar(m, 100 + n)
        law = exact_distribution(U, n, m)
        pats = np.array(list(law.keys()), dtype=np.float64)
        p = np.array(list(law.values()))
        mean_bf = (p[:, None] * pats).sum(0)
        prod_bf = np.einsum("s,si,sj->ij", p, pats, pats)
        C_bf = prod_bf - mean_bf[:, None] * mean_bf[None, :]
        mean, C = exact_moments(U, n)
        iu = np.triu_indices(m, 1)
        err = max(np.abs(mean - mean_bf).max(), np.abs(C[iu] - C_bf[iu]).max())
        worst = max(worst, err)
        print(f"  n = {n}, m = {m}: closed-form moments vs brute force, max |diff| = {err:.2e}")
    print("selftest", "OK" if worst < 1e-12 else "FAILED")
    return 0 if worst < 1e-12 else 1


def cmd_exact(a):
    rows = []
    for n, m in parse_cases(a.cases):
        P = n_patterns(n, m)
        U = haar(m, a.seed + n)
        print(f"n = {n}, m = {m}: {P} patterns, {a.shots} shots", flush=True)
        law = exact_distribution(U, n, m)
        tb, t_tb = theboss_sample(U, n, m, a.shots, a.seed + n)
        tvd_tb, floor = tvd_and_floor(tb, law, a.shots)
        print(f"  theboss        TVD {tvd_tb:.4f}  (floor {floor:.4f})  {t_tb:.0f} s", flush=True)
        row = {"n": n, "m": m, "patterns": P, "shots": a.shots, "floor": floor, "tvd_theboss": tvd_tb, "s_theboss": t_tb}
        if not a.no_ours:
            ours, t_o = ours_cpu_sample(U, n, m, a.shots, a.seed + n)
            tvd_o, _ = tvd_and_floor(ours, law, a.shots)
            tvd2, floor2, _ = two_sample_tvd([tuple(r) for r in tb.tolist()], [tuple(r) for r in ours.tolist()])
            print(f"  ours (cpu 2020) TVD {tvd_o:.4f}  (floor {floor:.4f})  {t_o:.0f} s", flush=True)
            print(f"  theboss vs ours two-sample TVD {tvd2:.4f}  (floor {floor2:.4f})", flush=True)
            row.update({"tvd_ours_cpu": tvd_o, "s_ours_cpu": t_o, "tvd_two_sample": tvd2, "floor_two_sample": floor2})
        rows.append(row)
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w") as f:
            csv_header(f, "exact", f"seed={a.seed}")
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r.get(k, '')}" for k in keys) + "\n")
        print("wrote", a.csv)
    return 0


def _save(out_dir, who, n, m, U, occ, secs, seed, extra):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"theboss_tvd_{who}_n{n}_m{m}.npz")
    np.savez_compressed(path, U=U, occ=occ.astype(np.int16), seconds=secs, seed=seed, n=n, m=m,
                        meta=json.dumps({**versions(), **extra}))
    print(f"  saved {path}  ({occ.shape[0]} samples, {secs:.1f} s)", flush=True)


def cmd_theboss(a):
    for n, m in parse_cases(a.cases):
        U = haar(m, a.seed + n)
        print(f"theboss n = {n}, m = {m}, {a.shots} shots", flush=True)
        occ, secs = theboss_sample(U, n, m, a.shots, a.seed + n)
        _save(a.out, "theboss", n, m, U, occ, secs, a.seed, {"sampler": "theboss GeneralizedCliffordsB"})
    return 0


def cmd_ours(a):
    for n, m in parse_cases(a.cases):
        U = haar(m, a.seed + n)
        print(f"ours n = {n}, m = {m}, {a.shots} shots, device {a.device}, backend {a.backend}", flush=True)
        if a.device == "cpu":
            occ, secs = ours_cpu_sample(U, n, m, a.shots, a.seed + n)
            extra = {"sampler": "fock_sample_reference algo=2020 (cpu)"}
        else:
            occ, secs = ours_gpu_sample(U, n, m, a.shots, a.seed + n, a.device, a.backend)
            import torch
            extra = {"sampler": f"fock_sample backend={a.backend}", "gpu": torch.cuda.get_device_name(0)}
        _save(a.out, "ours", n, m, U, occ, secs, a.seed, extra)
    return 0


def _load(d, who, n, m):
    path = os.path.join(d, f"theboss_tvd_{who}_n{n}_m{m}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {"U": z["U"], "occ": z["occ"].astype(np.int64), "seconds": float(z["seconds"]), "meta": str(z["meta"])}


def cmd_compare(a):
    cases = parse_cases(a.cases) if a.cases else None
    if cases is None:
        cases = sorted({tuple(int(x[1:]) for x in os.path.basename(f)[:-4].split("_")[-2:])
                        for f in os.listdir(a.theboss) if f.startswith("theboss_tvd_theboss_")})
    rows = []
    for n, m in cases:
        A = _load(a.theboss, "theboss", n, m)
        B = _load(a.ours, "ours", n, m) if a.ours else None
        if A is None:
            print(f"n = {n}, m = {m}: no theboss file, skipped")
            continue
        U = A["U"]
        row = {"n": n, "m": m, "shots_theboss": A["occ"].shape[0]}
        print(f"n = {n}, m = {m}: theboss {A['occ'].shape[0]} samples" + (f", ours {B['occ'].shape[0]}" if B else ""))
        if B is not None:
            dU = float(np.abs(U - B["U"]).max())
            row["max_dU"] = dU
            print(f"  same U: max |dU| = {dU:.1e}")
            assert dU < 1e-10, "the two sides did not sample the same unitary"
        law = exact_distribution(U, n, m) if n_patterns(n, m) <= a.pattern_cap else None
        for who, D in (("theboss", A), ("ours", B)):
            if D is None:
                continue
            assert (D["occ"].sum(1) == n).all(), f"{who}: a sample does not have {n} photons"
            if law is not None:
                tvd_bf, floor_bf = tvd_and_floor(D["occ"], law, D["occ"].shape[0])
                print(f"  {who:8s} vs brute force ({len(law)} patterns): TVD {tvd_bf:.4f}  (floor {floor_bf:.4f})")
                row.update({f"{who}_tvd_bf": tvd_bf, f"{who}_floor_bf": floor_bf})
            z1, z2 = moment_zscores(D["occ"], U, n)
            s1, s2 = z_summary(z1), z_summary(z2)
            print(f"  {who:8s} <n_i>: rms z {s1['rms']:.2f}, max |z| {s1['max']:.2f}, |z|>2 {100 * s1['frac_gt2']:.1f} % of {s1['count']}"
                  f"   C_ij: rms z {s2['rms']:.2f}, max |z| {s2['max']:.2f}, |z|>2 {100 * s2['frac_gt2']:.1f} % of {s2['count']}")
            row.update({f"{who}_z1_rms": s1["rms"], f"{who}_z1_max": s1["max"], f"{who}_z1_gt2": s1["frac_gt2"],
                        f"{who}_z2_rms": s2["rms"], f"{who}_z2_max": s2["max"], f"{who}_z2_gt2": s2["frac_gt2"]})
        if B is not None:
            row["shots_ours"] = B["occ"].shape[0]
            ka = [collision_type(r) for r in A["occ"].tolist()]
            kb = [collision_type(r) for r in B["occ"].tolist()]
            tvd, floor, nk = two_sample_tvd(ka, kb)
            print(f"  collision types ({nk} seen): theboss vs ours TVD {tvd:.4f}  (floor {floor:.4f})")
            row.update({"ctype_tvd": tvd, "ctype_floor": floor, "ctype_keys": nk})
            da = [int((np.asarray(r) > 0).sum()) for r in A["occ"].tolist()]
            db = [int((np.asarray(r) > 0).sum()) for r in B["occ"].tolist()]
            tvd_d, floor_d, _ = two_sample_tvd(da, db)
            print(f"  occupied-mode count:      theboss vs ours TVD {tvd_d:.4f}  (floor {floor_d:.4f})")
            row.update({"nocc_tvd": tvd_d, "nocc_floor": floor_d})
            if n_patterns(n, m) <= a.pattern_cap:
                tvd_p, floor_p, nk = two_sample_tvd([tuple(r) for r in A["occ"].tolist()], [tuple(r) for r in B["occ"].tolist()])
                print(f"  full patterns ({nk} seen of {n_patterns(n, m)}): two-sample TVD {tvd_p:.4f}  (floor {floor_p:.4f})")
                row.update({"pattern_tvd": tvd_p, "pattern_floor": floor_p})
        rows.append(row)
    if a.csv and rows:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.csv, "w") as f:
            csv_header(f, "compare", f"theboss={a.theboss} ours={a.ours}")
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r.get(k, '')}" for k in keys) + "\n")
        print("wrote", a.csv)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("exact")
    p.add_argument("--cases", default="4:8,5:10,6:6,7:7")
    p.add_argument("--shots", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ours", action="store_true", help="theboss against brute force only (no torch needed)")
    p.add_argument("--csv", default="")
    p = sub.add_parser("theboss")
    p.add_argument("--cases", default="10:20,12:24,14:28")
    p.add_argument("--shots", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/theboss_tvd")
    p = sub.add_parser("ours")
    p.add_argument("--cases", default="10:20,12:24,14:28")
    p.add_argument("--shots", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default="cuda", help="fock_sample backend: cuda | triton")
    p.add_argument("--out", default=os.environ.get("RESULTS_DIR", "results/theboss_tvd"))
    p = sub.add_parser("compare")
    p.add_argument("--theboss", default="results/theboss_tvd", help="directory of theboss_tvd_theboss_*.npz")
    p.add_argument("--ours", default="", help="directory of theboss_tvd_ours_*.npz (results/<stamp>/)")
    p.add_argument("--cases", default="", help="default: every theboss file found")
    p.add_argument("--pattern-cap", type=int, default=20000, help="full-pattern two-sample TVD only below this many patterns")
    p.add_argument("--csv", default="")
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "exact": cmd_exact, "theboss": cmd_theboss, "ours": cmd_ours, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
