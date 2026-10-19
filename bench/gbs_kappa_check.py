"""The condition number kappa of the fp64 pair sieve on frontier patterns at N = 50-62 (B's precision request, P 10-11).

The timing model's precision tiers (`gaussian.timing.TimingModel.tier`) take kappa(N) = 10^(0.40 + 0.197 N), fitted at
N <= 56 and extrapolated; with the measured error law (error <= 0.74 eps kappa, 275 sieves against dd, job 27) it lets
the strata N <= 62 of the 41.5 crossing run in fp64. A dd check of whole sieves there is not feasible (hours to weeks per
call), but kappa comes out of the fp64 sieve itself (`return_abs`): this measures it on patterns of the frontier's own
kind and bounds the error with the measured law. No dd, one fp64 call per pattern. GPU (Triton, Q = 64: N <= 64).

    python3 bench/gbs_kappa_check.py --N 50,52,54,56,58 --csv kappa.csv

Patterns: the frontier point m = 2n, n = --squeezers single-mode squeezers of --mean photons each through a Haar U
(the state the sampler's chain runs on); N photons placed by the timing model's "be" surrogate (uniform multisets,
`timing.surrogate_patterns`, the model the 41.5 crossing priced with) or "free" (distinct modes). The call is the
chain's sieve for that pattern: A = B on the occupied modes expanded by their counts, loop vector gamma = B alpha*
(alpha complex normal x 0.7) and z-column B[:, j] of a random mode, as bench/gbs_pairing_bench.py builds its patterns.
It runs `lhaf_poly_multi(precision="fp64", pairing="auto")` as the sampler does (V = 1), so the pairing is the one the
sampler would take (`hafnian.LOG`: pairs searched, terms enumerated, redo).

CSV per pattern: N, kind, seed, distinct modes, max count, kappa, log10 kappa, the fit's log10 kappa(N), eps kappa,
0.74 eps kappa (the error bound by the measured law), terms, redo, seconds (the whole call: also a timing check of the
model's priced sizes N = 57-58 and beyond), draw (patterns drawn so far at this N).

--max-terms T (P 10-11, after job 41: the count-grouped pairings sat above the fit, the whole ones 1.3-2.9 decades
below): patterns are drawn until --per of them have a pairing of at most T terms, up to --draws draws per N. The pairing
is predicted before the call with the sampler's own choice (`hafnian._choose_pairing` / `_select_pairing`, seeded by N,
so the call takes the same one; N >= 36, where the search runs); the call's own LOG still gives the terms it ran (a redo
is reported). A summary line per N gives how many draws had a pairing under T (the grouped fraction).
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import photonic_engines.gaussian.hafnian as H
from photonic_engines import GaussianInput
from photonic_engines.gaussian.hafnian import expand
from photonic_engines.gaussian.timing import surrogate_patterns

EPS = H.EPS                                                   # 2^-52, as the sieve's check and job 27's law
FIT = (0.40, 0.197)                                           # TimingModel.kappa_fit
LAW = 0.74                                                    # max error / (eps kappa), job 27

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--N", default="50,52,54,56,58", help="photon numbers (sieve vertices), run in this order")
ap.add_argument("--per", type=int, default=1, help="patterns per (N, kind)")
ap.add_argument("--kinds", default="be", help="be (the frontier surrogate) and/or free")
ap.add_argument("--squeezers", type=int, default=42, help="n of the frontier point (m = 2n modes); 41.5 crossing")
ap.add_argument("--mean", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--csv", default="results/gbs_kappa.csv")
ap.add_argument("--max-terms", type=float, default=None, help="run only patterns whose predicted pairing has <= this many "
                "terms (count-grouped ones), drawing up to --draws per N")
ap.add_argument("--draws", type=int, default=400, help="with --max-terms: patterns drawn per (N, kind) at most")
args = ap.parse_args()


def predicted_terms(A, D0, D1, lab, N):
    """Terms of the pairing lhaf_poly_multi(precision="fp64", pairing="auto") will run first (N even, N >= 36)."""
    At, D0t, D1t = (torch.as_tensor(x, dtype=torch.complex128).to(dev) for x in (A, D0, D1))
    n = N // 2
    cands, est, best = H._choose_pairing(At, D0t, D1t, lab, N, dev)
    pick, _ = H._select_pairing(cands, est, best, N, n, EPS, dev)
    return H._pairing_terms(cands[pick][1], n)


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


dev = torch.device(args.device)
if dev.type == "cuda":
    from photonic_engines.gaussian import sieve_triton
    sieve_triton.warmup(dev)
    print(torch.cuda.get_device_name(0), flush=True)
k = args.squeezers; m = 2 * k
st = GaussianInput.smsv(k, m, GaussianInput.r_for_mean(args.mean))
B, _, _ = st.bargmann(haar(m, args.seed))
rows = []
cap = args.max_terms
for N in (int(x) for x in args.N.split(",")):
    if cap is not None and (N % 2 or H._PAIR_SEARCH.get(dev.type, 10 ** 9) > N // 2):
        raise SystemExit(f"--max-terms needs even N with N/2 >= the pairing search ({N})")
    for kind in args.kinds.split(","):
        i = drawn = 0
        while i < args.per and (cap is None or drawn < args.draws):
            if cap is None:
                seed = args.seed + 1000 * N + 100 * i + (kind == "free")
            else:                                                      # own seed range: draws > 9 per N
                seed = args.seed + 100000 * N + 10 * drawn + (kind == "free")
            drawn += 1
            rng = np.random.default_rng(seed)
            pat = surrogate_patterns(N, m, 1, kind, rng)[0]
            modes = np.nonzero(pat)[0]; counts = pat[modes]
            alpha = (rng.normal(size=m) + 1j * rng.normal(size=m)) * 0.7
            gam = (B @ alpha.conj())[modes]; col = B[modes, rng.integers(m)]
            A, _ = expand(B[np.ix_(modes, modes)], gam, counts); lab = np.repeat(np.arange(len(counts)), counts)
            D0 = gam[lab][None, :]; D1 = col[lab][None, :]
            if cap is not None and predicted_terms(A, D0, D1, lab, N) > cap:
                continue
            H.LOG = []
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            val, ab = H.lhaf_poly_multi(A, D0, D1, device=dev, return_abs=True, labels=lab, precision="fp64")
            if dev.type == "cuda":
                torch.cuda.synchronize()
            sec = time.perf_counter() - t
            log = H.LOG[-1]; H.LOG = None
            v = val[0].cpu().numpy(); a = ab[0].cpu().numpy()
            kap = float(a.max() / max(np.abs(v).max(), 1e-300))
            lk, fit = math.log10(kap), FIT[0] + FIT[1] * N
            rows.append({"N": N, "kind": kind, "seed": seed, "modes": len(modes), "max_count": int(counts.max()),
                         "kappa": kap, "log10_kappa": lk, "fit_log10_kappa": fit, "eps_kappa": EPS * kap,
                         "bound_074_eps_kappa": LAW * EPS * kap, "terms": log[4], "redo": bool(log[5]), "seconds": sec,
                         "draw": drawn})
            print(f"N={N:2d} {kind:4s} #{i}: {len(modes)} modes, max count {counts.max()}: kappa {kap:.2e} "
                  f"(log10 {lk:.2f}, fit {fit:.2f}), 0.74 eps kappa {LAW * EPS * kap:.1e}, terms {log[4]:.2e}"
                  f"{' REDO' if log[5] else ''}, {sec:.1f} s", flush=True)
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            with open(args.csv, "w", newline="") as f:                 # after every pattern: a timeout keeps what ran
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader(); w.writerows(rows)
            i += 1
        if cap is not None:
            print(f"N={N:2d} {kind:4s}: {i} of {drawn} draws with a pairing of <= {cap:.2e} terms", flush=True)
print("wrote", args.csv)
