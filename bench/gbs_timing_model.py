"""The GBS frontier from a calibrated timing model of the exact sampler (docs/GAUSSIAN.md §2.2).

    python3 bench/gbs_timing_model.py calibrate --device cuda --out gbs_timing_calib.json     # GPU job, ~15-25 min
    python3 bench/gbs_timing_model.py sieves --device cuda --n 50,52,54,56 --out gbs_timing_sieves.json   # GPU check
    python3 bench/gbs_timing_model.py model --calib gbs_timing_calib.json [--check sieves.json] --csv f.csv   # CPU
    python3 bench/gbs_timing_model.py frontier --calib gbs_timing_calib.json --extra rate.json,sieves.json --csv g.csv  # CPU
    python3 bench/gbs_timing_model.py curve --calib gbs_timing_calib.json --point '{"ratio": 2}' --n-range 30,50 --csv c.csv

calibrate (on the target GPU): times, with the sampler's own code paths,
  multi   one collision-free `lhaf_poly_multi` call per N = 14 .. --multi-max vertices and V = 1 .. 4 loop vectors
          (the calls of the chain from `sampler.V2_FROM` vertices on);
  small   `lhaf_poly_small` (numpy, host) per N = 1 .. 13 and V = 1 .. 4;
  rate    seconds per subset term of the v2 pipeline (`hafnian._terms`, the Triton kernels) at n = --rate-n pairs:
          one batch of random subsets per bucket of `_bucket_sizes(n)` at the sieve's own batch size, weighted by
          the bucket's subset count; reaches n where a whole sieve is too long to time (n <= 32 with the Triton kernel,
          Q = 64; --kernel torch runs the torch matrix part, batched zgemm with no size limit, to n = 64 and beyond);
  shots   capped shots at m = --ratio n, n = --shot-n, with `sampler.Profile.events` (every call and draw, timed)
          and `hafnian.LOG` (the pairing the sieve chose): the draw / preparation times, and the validation set.
sieves (GPU): times whole collision-free calls at sizes the model extrapolates (N > --multi-max), for `model --check`.
model (CPU, from the calibration file): builds `timing.TimingModel`, validates it
  (a) rate x 2^n against the measured multi calls;
  (b) every recorded call against its predicted time;
  (c) every uncensored shot (its pattern replayed by `timing.plan_calls`) against its measured time;
  (d) E[cost | N] per stratum: measured mean, the model on the measured patterns, the model on surrogate patterns
      ("free", "be"), and the collision statistics (occupied modes given N) of both surrogates against the shots;
then computes t_S = sum_(N <= N_S) P(N) E[cost | N] for S = 1e4, 1e6, 1e8 (P exact, `cost.detected_law`) over
n = --n-range and interpolates n* where t_S = 1 / --target seconds.
frontier (CPU): n* for every point of a grid (PAPER_GRID: the nine ratios, loss, squeezing and Schmidt scans at m = 2n),
S = 1e4 / 1e6 / 1e8, with the priced and beyond-kernel shares at the upper bracket; Schmidt modes by `timing.run_weights`.
curve (CPU): t_S(n) at every integer n of a range for one grid point (the per-n curve behind a frontier row).
Every row of `model` reports the share of t_S that rests on
priced calls (above the largest measured call) and on calls beyond the kernel (N > 64 vertices: no Q = 128 kernel exists; those
are priced by the same extrapolation and flagged).
"""
import argparse
import json
import math
import sys
import time
from itertools import pairwise

import numpy as np


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


# ---- calibrate ---------------------------------------------------------------------------------
def calibrate(args):
    import torch

    import photonic_engines.gaussian.hafnian as haf
    import photonic_engines.gaussian.sampler as smp
    from photonic_engines import GaussianInput
    from photonic_engines.gaussian import sample
    from photonic_engines.gaussian.hafnian import lhaf_poly_multi, lhaf_poly_small

    dev = args.device
    cuda = dev == "cuda"
    haf._TRITON = {"auto": None, "torch": False, "triton": True}[args.kernel]      # matrix part of `_terms`

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def timed(f, reps):
        best = float("inf")
        for _ in range(reps):
            sync(); t = time.perf_counter(); f(); sync(); best = min(best, time.perf_counter() - t)
        return best

    out = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "torch": torch.__version__,
           "V2_FROM": smp.V2_FROM, "SMALL_BELOW": smp.SMALL_BELOW, "multi_max": args.multi_max,
           "search_from": haf._PAIR_SEARCH.get("cuda" if cuda else "cpu"), "multi": [], "small": [], "rate": [],
           "shots": [], "kernel": args.kernel, "precision": args.precision}
    print(out["device"], f"torch {torch.__version__}", flush=True)
    if cuda:
        from photonic_engines.gaussian.sieve_triton import warmup
        t = time.perf_counter(); warmup(dev, 8); print(f"Triton warm-up {time.perf_counter() - t:.1f} s", flush=True)
    rng = np.random.default_rng(1)

    def instance(N, V):
        m = 2 * N
        st = GaussianInput.smsv(N, m, GaussianInput.r_for_mean(0.5))
        B, _, _ = st.bargmann(haar(m, N))
        A = B[:N, :N]
        D0 = rng.normal(size=(V, N)) * 0.5 + 0j
        D1 = rng.normal(size=(V, N)) * 0.3 + 0j                        # the column B[:N, j]: values do not set the time
        return A, D0, D1

    parts = set(args.parts.split(","))
    if "multi" in parts:
        print("\nmulti: one collision-free lhaf_poly_multi call, s", flush=True)
    for N in range(smp.V2_FROM, args.multi_max + 1) if "multi" in parts else ():
        row = []
        for V in range(1, 5):
            A, D0, D1 = instance(N, V)
            lhaf_poly_multi(A[:2, :2], D0[:, :2], D1[:, :2], device=dev)
            t = timed(lambda: lhaf_poly_multi(A, D0, D1, device=dev), 1 if args.slow_from <= N else args.reps)
            out["multi"].append([N, V, t]); row.append(t)
        print(f"  N={N:2d}: " + "  ".join(f"V={V} {t:9.4f}" for V, t in zip(range(1, 5), row)), flush=True)

    if "small" in parts:
        print("\nsmall: lhaf_poly_small, s", flush=True)
    for N in range(1, smp.SMALL_BELOW) if "small" in parts else ():
        row = []
        for V in range(1, 5):
            A, D0, D1 = instance(N, V)
            ts = []
            for _ in range(args.reps * 5):
                t = time.perf_counter(); lhaf_poly_small(A, D0, D1, return_abs=True); ts.append(time.perf_counter() - t)
            out["small"].append([N, V, float(np.median(ts))]); row.append(float(np.median(ts)))
        print(f"  N={N:2d}: " + "  ".join(f"V={V} {t * 1e3:7.3f} ms" for V, t in zip(range(1, 5), row)), flush=True)

    if "rate" in parts:
        print("\nrate: s per subset term of the v2 pipeline (per bucket batch, weighted)", flush=True)
    budget = (1 << 32) if cuda else (1 << 28)
    rate_n = range(int(args.rate_n.split("-")[0]), int(args.rate_n.split("-")[1]) + 1) if "-" in args.rate_n \
        else [int(x) for x in args.rate_n.split(",")]
    for n in rate_n if "rate" in parts else ():
        row = []
        for V in range(1, 5):
            N = 2 * n
            A, D0, D1 = instance(N, V)
            At = torch.as_tensor(A, dtype=torch.complex128, device=dev)
            D0t = torch.as_tensor(D0, dtype=torch.complex128, device=dev); D1t = torch.as_tensor(D1, dtype=torch.complex128, device=dev)
            Ce, Ye, XYe = haf._setup(At, D0t, D1t, np.arange(N), torch.device(dev))
            acc = torch.zeros((V, 2 * n + 1), dtype=torch.complex128, device=dev); acc_abs = torch.zeros_like(acc.real)
            dd = args.precision == "dd"
            terms = haf._terms_dd if dd else haf._terms
            if dd:
                acc = [acc, torch.zeros_like(acc)]
            bud = budget // haf._DD_MEM if dd else budget
            lo, total_t, total_c = 0, 0.0, 0
            for pb in haf._bucket_sizes(n, torch.device(dev)):
                count = sum(math.comb(n, p) for p in range(lo + 1, pb + 1)); lo = pb
                chunk = min(haf._chunk_size_multi(n, pb, V, bud, torch.device(dev)), count, args.rate_chunk_max)
                pos = np.stack([np.sort(rng.choice(n, size=pb, replace=False)) for _ in range(min(chunk, 4096))])
                pos = np.resize(pos, (chunk, pb))
                idx = torch.as_tensor(np.concatenate([pos, pos + n], 1), device=dev)
                w = torch.ones(chunk, dtype=torch.float64, device=dev)
                terms(Ce, Ye, XYe, idx[:2], w[:2], n, V, acc, acc_abs, True)
                t = timed(lambda idx=idx, w=w, terms=terms: terms(Ce, Ye, XYe, idx, w, n, V, acc, acc_abs, True), 2)
                total_t += t / chunk * count; total_c += count
            rate = total_t / 2 ** n
            out["rate"].append([n, V, rate]); row.append(rate)
        print(f"  n={n:2d}: " + "  ".join(f"V={V} {r * 1e6:8.4f} us" for V, r in zip(range(1, 5), row)), flush=True)

    if "shots" in parts:
        print("\nshots (capped at --shot-cap photons), with events", flush=True)
    def parse(x):                                        # "n=44,ratio=4,eta=0.7": n int, the others float
        kv = dict(t.split("=") for t in x.split(","))
        return {k: (int(v) if k == "n" else float(v)) for k, v in kv.items()}
    configs = ([parse(x) for x in args.shot_configs.split(";")] if args.shot_configs
               else [{"n": int(x)} for x in args.shot_n.split(",")])
    for cfg in configs if "shots" in parts else ():
        n = cfg["n"]; ratio = cfg.get("ratio", args.ratio); mean = cfg.get("mean", args.mean); eta = cfg.get("eta", None)
        m = round(ratio * n)
        st = GaussianInput.smsv(n, m, GaussianInput.r_for_mean(mean))
        U = haar(m, 100 + n)
        sample(st, U, 1, seed=0, device=dev, max_photons=4)            # warm-up
        tn, k0 = 0.0, len(out["shots"])
        for s in range(args.shots):
            smp.PROFILE = smp.Profile(events=[]); haf.LOG = []
            occ, stats = sample(st, U, 1, seed=1000 * n + s, device=dev, loss=eta, max_photons=args.shot_cap,
                                return_stats=True, precision=args.shot_precision)
            ev = [list(e) for e in smp.PROFILE.events]; log = [list(x) for x in haf.LOG]
            smp.PROFILE = None; haf.LOG = None
            out["shots"].append({"n": n, "m": m, "ratio": ratio, "mean": mean, "eta": eta, "seed": 1000 * n + s,
                                 "precision": args.shot_precision,
                                 "seconds": stats.seconds[0], "photons": stats.photons[0], "censored": stats.censored[0],
                                 "pattern": None if stats.censored[0] else occ[0].tolist(), "events": ev, "log": log})
            tn += stats.seconds[0]
            with open(args.out, "w") as f:                         # after every shot: a crash keeps what was done
                json.dump(out, f)
            if tn > args.shot_budget:
                break
        mine = out["shots"][k0:]
        print(f"  n={n:2d} m={m:3d} mean={mean:g} eta={eta}: {len(mine)} shots, {tn:.1f} s, censored "
              f"{sum(x['censored'] for x in mine)}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", flush=True)


def sieves(args):
    """Time whole collision-free multi calls above the calibrated range (the model's extrapolated sizes)."""
    import torch

    from photonic_engines import GaussianInput
    from photonic_engines.gaussian.hafnian import lhaf_poly_multi
    dev = args.device
    if dev == "cuda":
        from photonic_engines.gaussian.sieve_triton import warmup
        warmup(dev, 8)
    rng = np.random.default_rng(2)
    rows = []
    print(torch.cuda.get_device_name(0) if dev == "cuda" else dev, flush=True)
    for N in (int(x) for x in args.n.split(",")):
        for V in (int(x) for x in args.V.split(",")):
            st = GaussianInput.smsv(N, 2 * N, GaussianInput.r_for_mean(0.5))
            B, _, _ = st.bargmann(haar(2 * N, N))
            D0 = rng.normal(size=(V, N)) * 0.5 + 0j; D1 = rng.normal(size=(V, N)) * 0.3 + 0j
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter(); lhaf_poly_multi(B[:N, :N], D0, D1, device=dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter() - t
            rows.append([N, V, t])
            print(f"  N={N:2d} V={V}: {t:9.2f} s", flush=True)
    with open(args.out, "w") as f:
        json.dump({"sieves": rows}, f)


# ---- model -------------------------------------------------------------------------------------
def build(cal, discount):
    """TimingModel from a calibration file; event-derived tables are means (the model predicts sums)."""
    from photonic_engines.gaussian.timing import TimingModel
    draws, preps = {}, {}
    for sh in cal["shots"]:
        ev = sh["events"]
        for e in ev:
            if e[0] == "draw":
                draws.setdefault(e[1], []).append(e[2])
        for a, b in pairwise(ev):                           # a prep event precedes its sieve event (which has N)
            if a[0] == "prep" and b[0] == "sieve":
                preps.setdefault(b[1], []).append(a[1])
    small = {(N, V): t for N, V, t in cal["small"]}
    ev_small = {}
    for sh in cal["shots"]:                                    # in-shot mean times of the numpy sieves where there are
        for e in sh["events"]:                                 # enough of them (the stand-alone medians ran ~20 %
            if e[0] == "sieve" and e[3] == "small":            # low on the CPU: cache state inside a shot)
                ev_small.setdefault((e[1], e[2]), []).append(e[5])
    small.update({k: float(np.mean(v)) for k, v in ev_small.items() if len(v) >= 5})
    rest = {}
    for sh in cal["shots"]:
        ev = sum(e[-1] for e in sh["events"])
        rest.setdefault(sh["m"], []).append(max(sh["seconds"] - ev, 0.0))
    return TimingModel(
        multi={(N, V): t for N, V, t in cal["multi"]}, small=small, plain={},
        draw={N: float(np.mean(v)) for N, v in draws.items()}, prep={N: float(np.mean(v)) for N, v in preps.items()},
        rate={(n, V): r for n, V, r in cal["rate"]}, multi_max=cal["multi_max"],
        shot={m: float(np.mean(v)) for m, v in rest.items()}, search_from=cal["search_from"] or 18, discount=discount)


def model(args):
    from photonic_engines import GaussianInput
    from photonic_engines.gaussian.cost import detected_law
    from photonic_engines.gaussian.timing import Call, conditional_cost, run_cost, surrogate_patterns
    cal, M = load_model(args)
    priced_from = max(k[0] for k in M.multi if k[1] == 1) + 1        # calls above the largest measured one are priced
    for path in (args.validate.split(",") if args.validate else []):  # shots reported in (b)-(d), not used to build M
        cal["shots"] = cal["shots"] + json.load(open(path))["shots"]
    print(f"calibration: {cal['device']}, multi up to N = {cal['multi_max']}, rate n = "
          f"{sorted({r[0] for r in cal['rate']})}, {len(cal['shots'])} shots", flush=True)

    print("\n(a) rate x 2^n / measured multi call (collision-free, even N)")
    for n in sorted({r[0] for r in cal["rate"]}):
        N = 2 * n
        if (N, 1) in M.multi:
            print(f"  n={n:2d}: " + "  ".join(f"V={V} {M.rate[(n, V)] * 2 ** n / M.multi[(N, V)]:5.2f}" for V in range(1, 5)
                                             if (N, V) in M.multi and (n, V) in M.rate))

    if args.check:
        print("\n(a') whole multi calls above the calibrated range: model / measured")
        for N, V, t in json.load(open(args.check))["sieves"]:
            print(f"  N={N:2d} V={V}: model {M.t_multi(N, V):9.2f} s, measured {t:9.2f} s, ratio {M.t_multi(N, V) / t:5.2f}")

    print("\n(b) recorded calls: predicted / measured, median [10 %, 90 %] per size range")
    groups = {}
    for sh in cal["shots"]:
        for e in sh["events"]:
            if e[0] == "sieve" and e[3] in ("small", "multi"):
                c = Call(e[1], e[2], e[3], e[4], 1, ())
                pred = M.t_call(c) - M._interp(M.prep, c.N)
                key = (e[3], min(e[1] // 8 * 8, 48))
                groups.setdefault(key, []).append(pred / max(e[5], 1e-9))
    for key in sorted(groups):
        r = np.array(groups[key])
        print(f"  {key[0]:5s} N {key[1]:2d}-{key[1] + 7:2d}: {np.median(r):5.2f} [{np.quantile(r, .1):5.2f}, "
              f"{np.quantile(r, .9):5.2f}]  ({r.size} calls)")
    searched = [x for sh in cal["shots"] for x in sh["log"] if x[3]]
    if searched:
        full = sum(1 for x in searched if x[4] == 2 ** x[2])
        print(f"  pairing search ran on {len(searched)} calls; full 2^n terms on {full} "
              f"(the model prices searched calls at 2^n)")

    print("\n(c) shots: predicted / measured seconds (uncensored)")
    rows = []
    def key(sh):
        return (sh["n"], sh["m"], sh.get("mean", 0.5), sh.get("eta") or 1.0)
    groups_ = sorted({key(sh) for sh in cal["shots"]})
    for g in groups_:
        sel = [sh for sh in cal["shots"] if key(sh) == g and not sh["censored"]]
        if not sel:
            continue
        pr = np.array([M.shot_time(sh["pattern"]) for sh in sel]); me = np.array([sh["seconds"] for sh in sel])
        rows.append((g, pr, me))
        print(f"  n={g[0]:2d} m={g[1]:3d} mean={g[2]:g} eta={g[3]:g}: {len(sel):3d} shots, sum pred / meas "
              f"{pr.sum() / me.sum():5.2f}, per shot median {np.median(pr / me):5.2f} "
              f"[{np.quantile(pr / me, .1):5.2f}, {np.quantile(pr / me, .9):5.2f}]")

    print("\n(d) E[cost | N], s: measured / model on the shots' patterns / surrogates free, be; "
          "occupied modes given N: shots / free / be")
    kinds = ("free", "be")
    for g in groups_:
        sel = [sh for sh in cal["shots"] if key(sh) == g and not sh["censored"]]
        n, m = g[0], g[1]
        by = {}
        for sh in sel:
            by.setdefault(sh["photons"], []).append(sh)
        for N in sorted(by):
            ss = by[N]
            if len(ss) < args.min_stratum or N == 0:
                continue
            meas = float(np.mean([s["seconds"] for s in ss]))
            onp = conditional_cost(M, N, m, patterns=[s["pattern"] for s in ss])
            sur = [conditional_cost(M, N, m, kind=k, count=args.count) for k in kinds if not (k == "free" and m < N)]
            occ = float(np.mean([sum(1 for x in s["pattern"] if x) for s in ss]))
            occ_s = [float(np.mean((surrogate_patterns(N, m, 256, k, np.random.default_rng(N)) > 0).sum(1)))
                     for k in kinds]
            print(f"  n={n:2d} m={m:3d} mean={g[2]:g} eta={g[3]:g} N={N:2d} ({len(ss):3d}): {meas:8.4f} / {onp:8.4f} / "
                  + " / ".join(f"{x:8.4f}" for x in sur)
                  + f"   occ {occ:5.1f} / " + " / ".join(f"{x:5.1f}" for x in occ_s))

    print(f"\nfrontier: t_S = sum_(N <= N_S) P(N) E[cost | N], surrogate {args.surrogate}, ratio {args.ratio}, "
          f"{args.mean} photons per squeezer, loss {args.loss}; target {1 / args.target:g} s per sample")
    Ss = [float(x) for x in args.samples.split(",")]
    lo, hi = (int(x) for x in args.n_range.split(","))
    res = []
    for n in range(lo, hi + 1, args.n_step):
        m = round(args.ratio * n)
        st = GaussianInput.smsv(n, m, GaussianInput.r_for_mean(args.mean))
        law = detected_law(st, args.law_cutoff, loss=args.loss)
        cache = {}

        def cond(N, m=m, cache=cache):
            if N not in cache:
                cache[N] = conditional_cost(M, N, m, kind=args.surrogate if not (args.surrogate == "free" and m < N) else "be",
                                            count=args.count)
            return cache[N]
        row = [n, m]
        txt = []
        for S in Ss:
            tS, NS = run_cost(law, cond, S)
            ext = sum(law[N] * cond(N) for N in range(priced_from, NS + 1)) / tS
            beyond = sum(law[N] * cond(N) for N in range(65, NS + 1)) / tS
            row += [NS, tS, ext, beyond]
            txt.append(f"S={S:.0e}: N_S {NS:3d} t_S {tS:10.4g} s (extrap {100 * ext:4.1f} %, >64 {100 * beyond:4.1f} %)")
        res.append(row)
        print(f"  n={n:3d} m={m:3d}  " + "  ".join(txt), flush=True)
    print("\nn* (t_S = 1 / target, log-interpolated between the bracketing n):")
    for i, S in enumerate(Ss):
        col = 3 + 4 * i
        star = None
        for a, b in pairwise(res):
            ta, tb = a[col], b[col]
            if ta <= 1 / args.target < tb:
                star = a[0] + (math.log(1 / args.target) - math.log(ta)) / (math.log(tb) - math.log(ta)) * (b[0] - a[0])
                break
        print(f"  S = {S:.0e}: n* = {star:.1f}" if star else f"  S = {S:.0e}: not bracketed in {args.n_range}")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["n", "m"] + [f"{c}_{S:.0e}" for S in Ss for c in ("N_S", "t_S", "ext", "beyond64")])
            w.writerows(res)


PAPER_GRID = (                                    # docs/GAUSSIAN_SURVEY.md §7: the curves the other modalities have
    [{"ratio": c} for c in (1, 1.5, 2, 2.5, 3, 4, 6, 8, 12)]                  # n*(m/n), the Fock table's ratios
    + [{"eta": e} for e in (0.9, 0.8, 0.7, 0.5)]                            # loss at r = r(0.5), m = 2n
    + [{"mean": x} for x in (0.1, 0.25, 0.75, 0.9, 1.2, 1.5)]               # squeezing scan, m = 2n
    + [{"schmidt": (l1, 1 - l1)} for l1 in (0.95, 0.9, 0.8)]                # spectral impurity, two Schmidt modes
)


def load_model(args):
    cal = json.load(open(args.calib))
    for path in (args.extra.split(",") if args.extra else []):
        ex = json.load(open(path))
        if ex.get("rate"):
            keep = {(r[0], r[1]) for r in ex["rate"]}
            cal["rate"] = [r for r in cal["rate"] if (r[0], r[1]) not in keep] + ex["rate"]
        cal["multi"] = cal["multi"] + ex.get("sieves", [])
    return cal, build(cal, args.discount)


def add_tiers(M, dd_path, ref_path):
    """Switch on the precision tiers (`TimingModel.tiers`): the dd per-term rates of `dd_path` (calibrate --precision dd),
    and past their largest n the dd / fp64 ratio at that n against the fp64 rates of `ref_path` (the same kernel: the
    torch path, calibrate --kernel torch), held constant."""
    dd = json.load(open(dd_path)); ref = json.load(open(ref_path))
    M.dd_rate = {(n, V): r for n, V, r in dd["rate"]}
    nmax = max(n for n, _ in M.dd_rate)
    fp = {(n, V): r for n, V, r in ref["rate"]}
    M.dd_ratio_beyond = {V: M.dd_rate[(nmax, V)] / fp[(nmax, V)] for n, V in M.dd_rate if n == nmax}
    M.tiers = True
    return M


def point_tS(M, pt, Ss, surrogate, count, law_cutoff, priced_from):
    """t_S(n) for one grid point {ratio, mean, eta, schmidt}: a memoised function of integer n returning, per S,
    (t_S, N_S, priced share, beyond-64 share)."""
    from photonic_engines import GaussianInput
    from photonic_engines.gaussian.cost import detected_law
    from photonic_engines.gaussian.timing import conditional_cost, run_weights
    c = pt.get("ratio", 2.0); mean = pt.get("mean", 0.5); eta = pt.get("eta", None); lam = pt.get("schmidt", (1.0,))
    r = GaussianInput.r_for_mean(mean)
    evals = {}

    def tS(n):
        if n not in evals:
            m = max(1, round(c * n))
            laws = [detected_law(GaussianInput.smsv(n, m, r * math.sqrt(lk)), law_cutoff, loss=eta) for lk in lam if lk > 0]
            cache = {}

            def cond(N):
                if N not in cache:
                    cache[N] = conditional_cost(M, N, m, kind="be" if (surrogate == "be" or m < N) else "free", count=count)
                return cache[N]
            out = []
            for S in Ss:
                w, NS = run_weights(laws, S)
                parts = [w[a] * cond(a) for a in range(NS + 1)]
                t = float(sum(parts))
                out.append((t, NS, sum(parts[priced_from:]) / t, sum(parts[65:]) / t))
            evals[n] = out
        return evals[n]
    return tS


def point_frontier(M, pt, Ss, surrogate, count, target, law_cutoff, priced_from):
    """n* for one grid point {ratio, mean, eta, schmidt} and every S: t_S(n) on integer n (geometric search, then unit
    steps inside the bracket), log-interpolated. Returns one dict per S."""
    tS = point_tS(M, pt, Ss, surrogate, count, law_cutoff, priced_from)
    res = []
    for i, S in enumerate(Ss):
        n = 4
        while tS(n)[i][0] < 1 / target:
            n = max(n + 1, int(n * 1.25))
            if n > 2000:
                raise RuntimeError(f"no bracket below n = 2000 for {pt}")
        while n > 1 and tS(n - 1)[i][0] >= 1 / target:
            n -= 1
        lo, hi = tS(n - 1)[i], tS(n)[i]
        star = (n - 1) + (math.log(1 / target) - math.log(lo[0])) / (math.log(hi[0]) - math.log(lo[0]))
        res.append({"S": S, "n_star": star, "n_lo": n - 1, "n_hi": n, "t_lo": lo[0], "t_hi": hi[0], "N_S": hi[1],
                    "priced": hi[2], "beyond64": hi[3]})
    return res


def frontier(args):
    cal, M = load_model(args)
    M.pairing = args.pairing
    if args.dd_rate:
        add_tiers(M, args.dd_rate, args.dd_ref)
        M.tier_choice = args.tier_choice
        print(f"precision tiers: dd rates {args.dd_rate}, dd / fp64 beyond n = {max(n for n, _ in M.dd_rate)}: "
              + ", ".join(f"V {V} x{r:.1f}" for V, r in sorted(M.dd_ratio_beyond.items())), flush=True)
    priced_from = max(k[0] for k in M.multi if k[1] == 1) + 1
    Ss = [float(x) for x in args.samples.split(",")]
    grid = PAPER_GRID if args.points == "paper" else [json.loads(x) for x in args.points.split(";")]
    print(f"GBS frontier from the timing model: {cal['device']}, measured calls to N = {priced_from - 1}, surrogate "
          f"{args.surrogate}, {args.count} patterns per N, target {1 / args.target:g} s per sample", flush=True)
    rows = []
    for pt in grid:
        t0 = time.time()
        res = point_frontier(M, pt, Ss, args.surrogate, args.count, args.target, args.law_cutoff, priced_from)
        c = pt.get("ratio", 2.0); mean = pt.get("mean", 0.5); eta = pt.get("eta", 1.0); lam = pt.get("schmidt", (1.0,))
        for x in res:
            rows.append({"engine": "gbs", "ratio": c, "mean": mean, "eta": eta, "schmidt": "/".join(f"{v:g}" for v in lam),
                         "surrogate": args.surrogate, **x})
        print(f"  c={c:<4g} mean={mean:<4g} eta={eta:<4g} schmidt={'/'.join(f'{v:g}' for v in lam):<9s} "
              + "  ".join(f"S={x['S']:.0e}: n* {x['n_star']:6.1f} (N_S {x['N_S']:3d}, priced {100 * x['priced']:3.0f} %, "
                          f">64 {100 * x['beyond64']:3.0f} %)" for x in res) + f"   [{time.time() - t0:.0f} s]", flush=True)
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    if args.as_frontier:
        as_frontier(rows, args.as_frontier, float(args.headline))


def curve(args):
    """t_S(n) at every integer n of --n-range for one grid point (the per-n curve behind a frontier row; same model,
    same pricing as `frontier`), with n* interpolated as in `point_frontier`."""
    cal, M = load_model(args)
    M.pairing = args.pairing
    if args.dd_rate:
        add_tiers(M, args.dd_rate, args.dd_ref)
        M.tier_choice = args.tier_choice
    priced_from = max(k[0] for k in M.multi if k[1] == 1) + 1
    Ss = [float(x) for x in args.samples.split(",")]
    pt = json.loads(args.point)
    tS = point_tS(M, pt, Ss, args.surrogate, args.count, args.law_cutoff, priced_from)
    a, b = (int(x) for x in args.n_range.split(","))
    c = pt.get("ratio", 2.0); mean = pt.get("mean", 0.5); eta = pt.get("eta", 1.0); lam = pt.get("schmidt", (1.0,))
    rows = []
    for n in range(a, b + 1):
        vals = tS(n)
        for S, (t, NS, pr, b64) in zip(Ss, vals):
            rows.append({"engine": "gbs", "ratio": c, "mean": mean, "eta": eta, "schmidt": "/".join(f"{v:g}" for v in lam),
                         "surrogate": args.surrogate, "S": S, "n": n, "m": max(1, round(c * n)), "t_S": t, "N_S": NS,
                         "priced": pr, "beyond64": b64})
        print(f"  n {n:3d}  " + "  ".join(f"S={S:.0e}: t_S {x[0]:.4g} s (N_S {x[1]})" for S, x in zip(Ss, vals)), flush=True)
    for S in Ss:
        ts = [(r["n"], r["t_S"]) for r in rows if r["S"] == S]
        for (n0, t0), (n1, t1) in pairwise(ts):
            if t0 < 1 / args.target <= t1:
                star = n0 + (math.log(1 / args.target) - math.log(t0)) / (math.log(t1) - math.log(t0))
                print(f"S={S:.0e}: n* = {star:.2f} (n = {n0} at {t0:.4g} s, n = {n1} at {t1:.4g} s)")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)


def as_frontier(rows, path, S):
    """The headline-S rows at mean 0.5, lossless, one Schmidt mode, in the columns of map_threshold.py's frontier.csv
    (engine, ratio, n_star, rho_eff, rho_theory, bracket, seconds): rho_eff = t_S(n_hi) / t_S(n_lo), the per-squeezer
    growth at n*; rho_theory left empty (the Fock formula does not apply); bracket names the model and the S."""
    import csv
    sel = [r for r in rows if r["S"] == S and r["mean"] == 0.5 and r["eta"] == 1.0 and r["schmidt"] == "1"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["engine", "ratio", "n_star", "rho_eff", "rho_theory", "bracket", "seconds"])
        for r in sorted(sel, key=lambda r: r["ratio"]):
            w.writerow(["gbs", float(r["ratio"]), round(r["n_star"], 1), round(r["t_hi"] / r["t_lo"], 3), "",
                        f"timing model, S = {S:.0e}: n={r['n_lo']} at {1 / r['t_lo']:.2f}/s, n={r['n_hi']} at "
                        f"{1 / r['t_hi']:.2f}/s  ->  n* = {r['n_star']:.1f}", ""])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--device", default="cuda")
    c.add_argument("--out", default="gbs_timing_calib.json")
    c.add_argument("--multi-max", type=int, default=48)
    c.add_argument("--slow-from", type=int, default=36, help="one repetition from this many vertices")
    c.add_argument("--reps", type=int, default=3)
    c.add_argument("--rate-n", default="12-32", help="pairs: 'a-b' (every n) or a comma list; every n matters: the rate "
                   "steps where the bucket layout changes, and odd n interpolated from even ones were 24 % low (job 20)")
    c.add_argument("--parts", default="multi,small,rate,shots")
    c.add_argument("--precision", default="fp64", choices=["fp64", "dd"],
                   help="rate only: dd = hafnian._terms_dd (double-double, torch path; batches / _DD_MEM)")
    c.add_argument("--kernel", default="auto", choices=["auto", "torch", "triton"],
                   help="matrix part of the sieve: torch (batched zgemm, no size limit) reaches n > 32 pairs for the rate")
    c.add_argument("--rate-chunk-max", type=int, default=1 << 18)
    c.add_argument("--shot-n", default="16,20,24,28,32,36,40")
    c.add_argument("--shots", type=int, default=64)
    c.add_argument("--shot-configs", default=None, help="'n=44;n=36,ratio=4;n=48,eta=0.7' (keys n, ratio, mean, eta) "
                   "instead of --shot-n (validation off the headline point)")
    c.add_argument("--shot-cap", type=int, default=44)
    c.add_argument("--shot-precision", default=None, choices=["fp64", "dd", "auto"],
                   help="precision of the shots' sieves (default: hafnian._PRECISION, fp64)")
    c.add_argument("--shot-budget", type=float, default=120.0, help="seconds of shots per n at most")
    c.add_argument("--ratio", type=float, default=2.0)
    c.add_argument("--mean", type=float, default=0.5)
    sv = sub.add_parser("sieves")
    sv.add_argument("--device", default="cuda")
    sv.add_argument("--n", default="50,52,54,56")
    sv.add_argument("--V", default="1,4")
    sv.add_argument("--out", default="gbs_timing_sieves.json")
    mo = sub.add_parser("model")
    mo.add_argument("--calib", required=True)
    mo.add_argument("--discount", default="self", choices=["self", "none"])
    mo.add_argument("--surrogate", default="be", choices=["be", "free"])
    mo.add_argument("--count", type=int, default=128, help="surrogate patterns per N")
    mo.add_argument("--min-stratum", type=int, default=5)
    mo.add_argument("--ratio", type=float, default=2.0)
    mo.add_argument("--mean", type=float, default=0.5)
    mo.add_argument("--loss", type=float, default=None)
    mo.add_argument("--n-range", default="20,60")
    mo.add_argument("--n-step", type=int, default=2)
    mo.add_argument("--samples", default="1e4,1e6,1e8")
    mo.add_argument("--target", type=float, default=1.0, help="samples per second (1 Hz: 1 s per sample)")
    mo.add_argument("--law-cutoff", type=int, default=400)
    mo.add_argument("--csv", default=None)
    mo.add_argument("--check", default=None, help="JSON of `sieves`: measured large calls against the model")
    mo.add_argument("--validate", default=None, help="comma-separated calibration JSONs whose shots are only checked")
    mo.add_argument("--extra", default=None, help="comma-separated JSON files merged into the calibration before the "
                    "model is built: 'rate' rows replace the same (n, V), 'sieves' rows become measured multi calls")
    fr = sub.add_parser("frontier")
    fr.add_argument("--calib", required=True)
    fr.add_argument("--extra", default=None, help="as for model")
    fr.add_argument("--discount", default="self", choices=["self", "none"])
    fr.add_argument("--surrogate", default="be", choices=["be", "free"])
    fr.add_argument("--count", type=int, default=64)
    fr.add_argument("--points", default="paper", help="'paper' (PAPER_GRID) or ';'-separated JSON dicts, e.g. "
                    '\'{"ratio": 4}; {"eta": 0.7}; {"schmidt": [0.9, 0.1]}\' (defaults ratio 2, mean 0.5, eta 1, one mode)')
    fr.add_argument("--samples", default="1e4,1e6,1e8")
    fr.add_argument("--target", type=float, default=1.0)
    fr.add_argument("--law-cutoff", type=int, default=600)
    fr.add_argument("--csv", default=None)
    fr.add_argument("--as-frontier", default=None, help="also write the ratio rows at --headline S in frontier.csv columns")
    fr.add_argument("--headline", default="1e4")
    fr.add_argument("--pairing", default="cross", choices=["cross", "self"],
                    help="grouping of collided calls: cross (the sieve since 2026-09-27) or self (v2.8: reproduces the "
                         "grid of 2026-09-25)")
    cu = sub.add_parser("curve")
    cu.add_argument("--calib", required=True)
    cu.add_argument("--extra", default=None, help="as for model")
    cu.add_argument("--discount", default="self", choices=["self", "none"])
    cu.add_argument("--surrogate", default="be", choices=["be", "free"])
    cu.add_argument("--count", type=int, default=64)
    cu.add_argument("--point", default="{}", help="one JSON dict as in frontier --points (defaults ratio 2, mean 0.5)")
    cu.add_argument("--n-range", default="30,50", help="'a,b': every integer n from a to b")
    cu.add_argument("--samples", default="1e4")
    cu.add_argument("--target", type=float, default=1.0)
    cu.add_argument("--law-cutoff", type=int, default=600)
    cu.add_argument("--csv", default=None)
    cu.add_argument("--pairing", default="cross", choices=["cross", "self"])
    for p_ in (fr, cu):
        p_.add_argument("--dd-rate", default=None, help="dd rate JSON (calibrate --precision dd): price every call in "
                        "the arithmetic its round-off needs (TimingModel.tiers); off: all fp64, as before 2026-09-28")
        p_.add_argument("--dd-ref", default=None, help="fp64 rate JSON of the same kernel as --dd-rate (torch) for the "
                        "dd / fp64 ratio past the dd rates")
        p_.add_argument("--tier-choice", default="cost", choices=["cost", "needed"],
                        help="cost: cheapest admissible (pairing, arithmetic); needed: lowest admissible arithmetic")
    args = ap.parse_args()
    {"frontier": frontier, "curve": curve, "calibrate": calibrate, "sieves": sieves, "model": model}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
