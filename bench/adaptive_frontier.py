"""Throughput of the adaptive sampler (docs/ADAPTIVE.md §4-5) against the plain Fock sampler, same host, same day.

For each n (m = round(c n), photons in modes 0..n-1, Haar U_0) and each K: K adaptive modes (the last K
modes, one per level unless --levels one), a different Haar feed-forward unitary for every outcome prefix,
PNR, no loss. Timed: `fock_sample` on U_0 (the non-adaptive anchor) and `AdaptiveCircuit.sample`. The ratio of
the two times per sample is the measured overhead X; `draws_per_sample` is the rejection's own count of it
(chains run per accepted sample). --final prefix (default, docs/ADAPTIVE.md §4b): its mean is E[2^r], about 2
per adaptive mode at m = 2n, with no tail. --final clifford (rejection on the full sample): its mean is the
number of branches C(n+K, K), carried by rare branches that a short run rarely meets, so a short run measures
the typical cost 2^H (bench/adaptive_overhead.py): read `draws_per_sample` next to the rate.

    python3 bench/adaptive_frontier.py --check                                  # GPU validation, then exit
    python3 bench/adaptive_frontier.py --n 16,20,24,28,30,32 --K 1,2 --shots 32 --csv results/adaptive_frontier.csv

--check   the distribution test the CPU suite cannot run: the REAL `fock_sample` as base sampler
          (n = 3, m = 6, levels [[1], [0, 3]], 20 000 shots, then n = 5, m = 10, K = 2 in one level) against the
          strong joint, TVD within 2.5 shot-noise floors, for both final stages and both chains; for
          final = prefix with the marginal chain, the chains run per sample against sum_p T[p] = E[2^r].
          Exit code 1 on failure.
--check-large  the same question where strong simulation cannot go and the cuda kernel DOES run (its steps start
          at k - 1 > 10, so n = 3, 5 never reach it): n = --large-n (16), m = 2n, against the exact low-rank
          tables. (i) `fock_sample(prefix=)` alone: acceptance against P / T within 4 sigma, and the counts of
          one final mode among the kept draws against the (c + 1)-mode marginal conditioned on the forced
          counts; (ii) the adaptive sampler, K = 1 and 2: the counts of one final mode against
          `final_marginal`, and chains per sample against the expectation given the branches met.
          TVDs within 2.5 shot-noise floors. Exit code 1 on failure. --seed moves the sampling seeds (same
          circuit), --large-shots the statistics.
--budget  seconds per timed point; a K whose point overruns it is not run at larger n.
--loss, --indist  uniform transmission eta and pairwise overlap x (docs/ADAPTIVE.md §7), applied to BOTH samplers,
          so x_time compares the adaptive sampler with the plain one at the same imperfection and the crossing
          moves by the eta x law of the plain sampler plus the adaptive toll at the surviving photon number.
Prints a log-linear fit of the rate over the measured n and its crossing of --target (1 sample/s); that n*
is an extrapolation unless two measured points bracket the target, and the line says which.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from helpers import haar_unitary

from photonic_engines.adaptive import AdaptiveCircuit
from photonic_engines.clifford import fock_sample

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="16,20,24,28")
ap.add_argument("--K", default="1,2")
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--shots", type=int, default=32)
ap.add_argument("--chain", default="marginal", choices=["marginal", "rejection"])
ap.add_argument("--final", default="prefix", choices=["prefix", "clifford"])
ap.add_argument("--levels", default="each", choices=["each", "one"])
ap.add_argument("--backend", default="cuda", choices=["cuda", "triton"])
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--budget", type=float, default=600.0)
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--check", action="store_true")
ap.add_argument("--check-shots", type=int, default=20000)
ap.add_argument("--check-large", action="store_true")
ap.add_argument("--large-n", type=int, default=16)
ap.add_argument("--large-shots", type=int, default=4000)
ap.add_argument("--max-entries", type=float, default=0, help="raise adaptive.MAX_MARGINAL_ENTRIES (graded table, 16 B each)")
ap.add_argument("--loss", type=float, default=1.0, help="uniform transmission eta, applied to both samplers")
ap.add_argument("--indist", type=float, default=1.0, help="pairwise overlap x of the x-model (V_HOM = x^2), both samplers")
ap.add_argument("--csv", default="results/adaptive_frontier.csv")
a = ap.parse_args()
if a.max_entries:
    import photonic_engines.adaptive as _adaptive
    _adaptive.MAX_MARGINAL_ENTRIES = int(a.max_entries)
dev = torch.device(a.device)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


def haar_policy(m, K_upto, salt):
    """A different Haar unitary on the unmeasured modes for every (level, outcome prefix), deterministic."""
    def policy(lam, p):
        seed = hash((salt, lam, tuple(p))) % (2 ** 31)
        return haar_unitary(m - K_upto[lam], seed).to(dev)
    return policy


def check() -> bool:
    ok = True
    for n, m, t, levels, K_upto in ((3, 6, [0, 2, 4], [[1], [0, 3]], [0, 1, 3]), (5, 10, [0, 2, 4, 6, 8], [[1, 3]], [0, 2])):
        C = AdaptiveCircuit(haar_unitary(m, 1).to(dev), t, levels, haar_policy(m, K_upto, 7))
        J, occ = C.probs(), C.occupations()
        idx = {tuple(o): i for i, o in enumerate(occ.tolist())}
        floor = 0.5 * math.sqrt(2 / math.pi) * float((J * (1 - J) / a.check_shots).sqrt().sum())
        moment_sum = sum(float(C._level_tables(C.n_levels, p[:K_upto[-2]])[1][p[K_upto[-2]:]]) for p in C.outcome_probs())
        for final in ("prefix", "clifford"):
            for chain in ("marginal", "rejection"):
                gen = torch.Generator(device=dev).manual_seed(11)
                smp = C.sample(a.check_shots, generator=gen, chain=chain, final=final, backend=a.backend)
                emp = torch.zeros_like(J)
                rows, cnt = torch.unique(smp, dim=0, return_counts=True)
                for o, c in zip(rows.tolist(), cnt.tolist(), strict=True):
                    emp[idx[tuple(o)]] = c / a.check_shots
                tvd = 0.5 * float((emp - J).abs().sum())
                draws = C.stats["base_shots"] / a.check_shots
                good = tvd < 2.5 * floor and smp.sum(1).eq(n).all().item()
                note = ""
                if final == "prefix" and chain == "marginal":                     # batches round up once per branch: needs shots
                    good = good and (a.check_shots < 5000 or abs(draws / moment_sum - 1) < 0.05)
                    note = f"  E[2^r] {moment_sum:.2f}"
                ok &= bool(good)
                print(f"check n={n} K={C.K} final={final:8s} chain={chain:9s} backend={a.backend} device={dev}: TVD {tvd:.4f}  "
                      f"floor {floor:.4f}  draws/sample {draws:.2f} (branches {C.stats['branches']}){note}  "
                      f"{'OK' if good else 'FAIL'}")
    return ok


def _tvd_floor(counts, ref, shots):
    """TVD of an empirical count histogram [n+1] against `ref`, and its shot-noise floor."""
    ref = ref / ref.sum()
    emp = counts.to(torch.float64) / shots
    return 0.5 * float((emp - ref).abs().sum()), 0.5 * math.sqrt(2 / math.pi) * float((ref * (1 - ref) / shots).sqrt().sum())


def check_large() -> bool:
    from photonic_engines.adaptive import binomial_moments, mode_marginals
    n, S = a.large_n, a.large_shots
    m, t, ok = 2 * n, list(range(a.large_n)), True
    U = haar_unitary(m, 4242).to(dev)
    probe = 0                                                                       # a final mode, never adaptive here
    for modes, counts in (([m - 1], (1,)), ([m - 1], (2,)), ([m - 1, m - 2], (1, 1)), ([m - 1, m - 2], (2, 1))):
        prefix = [b for b, c in zip(modes, counts, strict=True) for _ in range(c)]
        gen = torch.Generator(device=dev).manual_seed(5 + a.seed)
        smp = fock_sample(U, t, S, generator=gen, backend=a.backend, prefix=prefix)
        keep = (smp[:, modes] == torch.tensor(counts, device=dev)).all(1)
        P, T = mode_marginals(U, t, modes)[counts], binomial_moments(U, t, modes)[counts]
        ratio = float(P / T); acc = float(keep.double().mean())
        sigma = math.sqrt(ratio * (1 - ratio) / S)
        ref = mode_marginals(U, t, [*modes, probe])[counts]                         # [n+1]: P(counts, N_probe = q)
        kept = int(keep.sum())
        tvd, floor = _tvd_floor(torch.bincount(smp[keep][:, probe], minlength=n + 1), ref, kept)
        good = abs(acc - ratio) < 4 * sigma + 1 / S and tvd < 2.5 * floor
        ok &= good
        print(f"large n={n} prefix {prefix}: acceptance {acc:.4f} vs P/T {ratio:.4f} (sigma {sigma:.4f})  probe-mode TVD {tvd:.4f} "
              f"floor {floor:.4f} ({kept} kept)  {'OK' if good else 'FAIL'}")
    for K in (1, 2):
        modes = list(range(m - K, m))
        levels = [[j] for j in modes]
        K_upto = list(range(K + 1))
        C = AdaptiveCircuit(U, t, levels, haar_policy(m, K_upto, 3))
        gen = torch.Generator(device=dev).manual_seed(6 + a.seed)
        smp = C.sample(S, generator=gen, chain="marginal", final="prefix", backend=a.backend)
        ref = C.final_marginal([probe], tol=1e-7)
        tvd, floor = _tvd_floor(torch.bincount(smp[:, probe], minlength=n + 1), ref, S)
        rows, cnt = torch.unique(smp[:, modes], dim=0, return_counts=True)          # branches met and how often
        expect = 0.0
        for p, c in zip(rows.tolist(), cnt.tolist(), strict=True):
            Pl, Tl = C._level_tables(K, tuple(p[:K - 1]))
            expect += c * float(Tl[p[K - 1]] / Pl[p[K - 1]])
        draws = C.stats["base_shots"]
        good = tvd < 2.5 * floor and abs(draws / expect - 1) < 0.1 and bool(smp.sum(1).eq(n).all())
        ok &= good
        print(f"large n={n} adaptive K={K}: probe-mode TVD {tvd:.4f} floor {floor:.4f}  chains {draws} vs expected {expect:.0f} "
              f"({draws / S:.2f} per sample, {rows.shape[0]} branches met, dropped mass {1 - float(ref.sum()):.1e})  "
              f"{'OK' if good else 'FAIL'}")
    return ok


if a.check:
    sys.exit(0 if check() else 1)
if a.check_large:
    sys.exit(0 if check_large() else 1)

ns = [int(x) for x in a.n.split(",")]
Ks = [int(x) for x in a.K.split(",")]
rows, alive = [], dict.fromkeys(Ks, True)
os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)


def flush():
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


for n in ns:
    m = round(a.ratio * n)
    U0 = haar_unitary(m, 1000 * n + a.seed).to(dev)
    t_in = list(range(n))
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    imp = {"loss": None if a.loss >= 1 else a.loss, "indist": None if a.indist >= 1 else a.indist}
    fock_sample(U0, t_in, 2, generator=gen, backend=a.backend, **imp); sync()       # warm-up (JIT, caches)
    t0 = time.perf_counter(); fock_sample(U0, t_in, a.shots, generator=gen, backend=a.backend, **imp); sync()
    tf = time.perf_counter() - t0
    rows.append({"n": n, "m": m, "K": 0, "engine": "fock", "chain": "", "shots": a.shots, "seconds": round(tf, 4),
                 "samples_per_s": a.shots / tf, "draws_per_sample": 1.0, "branches_seen": 1, "branches": 1, "x_time": 1.0,
                 "eta": a.loss, "x": a.indist})
    print(rows[-1]); flush()
    for K in Ks:
        if not alive[K]:
            continue
        modes = list(range(m - K, m))
        levels = [[j] for j in modes] if a.levels == "each" else [modes]
        K_upto = [0] + [int(x) for x in np.cumsum([len(lev) for lev in levels])]
        C = AdaptiveCircuit(U0, t_in, levels, haar_policy(m, K_upto, a.seed))
        sync(); t0 = time.perf_counter()
        C.sample(a.shots, generator=gen, chain=a.chain, final=a.final, backend=a.backend, **imp); sync()
        ta = time.perf_counter() - t0
        rows.append({"n": n, "m": m, "K": K, "engine": f"adaptive-{a.final}", "chain": a.chain, "shots": a.shots, "seconds": round(ta, 4),
                     "samples_per_s": a.shots / ta, "draws_per_sample": C.stats["base_shots"] / a.shots,
                     "branches_seen": C.stats["branches"], "branches": math.comb(n + K, K), "x_time": ta / tf,
                     "eta": a.loss, "x": a.indist})
        print(rows[-1]); flush()
        if ta > a.budget:
            alive[K] = False
            print(f"K = {K}: {ta:.0f} s at n = {n} is over the budget of {a.budget:.0f} s, not run at larger n")

fock_fit = None
for K in [0, *Ks]:
    pts = [(r["n"], math.log(r["samples_per_s"])) for r in rows if r["K"] == K]
    if len(pts) < 2:
        continue
    x, y = np.array([p[0] for p in pts[-4:]], dtype=float), np.array([p[1] for p in pts[-4:]])
    slope, icpt = np.polyfit(x, y, 1)
    nstar = (math.log(a.target) - icpt) / slope
    kind = "bracketed" if min(y) <= math.log(a.target) <= max(y) else "EXTRAPOLATED"
    line = (f"K = {K}: rho_eff {math.exp(-slope):.3f} per photon, n* = {nstar:.1f} at {a.target:g} sample/s ({kind}, "
            f"fit on n = {[int(v) for v in x]})")
    if K == 0:
        fock_fit = (nstar, -slope)
    elif fock_fit is not None:
        # The adaptive time is the plain time plus a per-branch / per-stream overhead that fades with n, so an
        # extrapolated fit of its rate has too shallow a slope and can even cross above the plain sampler. The
        # crossing from the plain fit and the measured time ratio at the largest n, n*_0 - log x_time / log rho_0,
        # does not have that defect (it assumes the ratio stays at its last value, a lower bound while it falls);
        # docs/ADAPTIVE.md §5c uses it wherever the adaptive fit is extrapolated.
        last = [r for r in rows if r["K"] == K][-1]
        line += f"; from x_time {last['x_time']:.2f} at n = {last['n']}: n* = {fock_fit[0] - math.log(last['x_time']) / fock_fit[1]:.1f}"
    print(line)
