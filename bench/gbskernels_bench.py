"""GBSKernels' GPU GBS sampler against ours on identical instances.

GBSKernels (github.com/chaofanzhougit/GBSKernels v0.2.2, Apache-2.0, built by scripts/gbskernels_build.sh) is run AS
SHIPPED: `sampling.sampler.sample(cov, k, backend="gpu")` with its defaults (cutoff 6 per mode, precision fp64, its own
CPU fallback for submatrices above its GPU cap d <= 20). Nothing of its code is copied here; it is imported from
$GBSK_SRC at run time. Run with the host's python3 (torch, photonic_engines); $GBSK_SITE (the ~/gbsk venv's
site-packages) is appended last to sys.path for anything GBSKernels needs that the host lacks (run 20261011_103633: the
venv itself does not see torch).

Instances are those of our squeezed row (bench/find_threshold.py --engine gbs): n squeezed vacua at r (0.5 photons per
squeezer) on modes 0..n-1 of U = haar(m, seed + n), m = round(c n), lossless, PNR. Their input is the xxpp covariance
of that output state with hbar = 2 (vacuum = identity); `check` verifies that both samplers draw the same law.

  check : small n, many shots: per-mode mean photon numbers of both samplers against the exact ones, and the TVD of
          their pattern histograms against the TVD of two independent runs of ours (the noise floor).
  time  : per n, shots timed one call each (theirs) / per shot from our sampler's stats, until --budget seconds or
          --max-shots; t_S (S = --run-samples, our row's statistic: gaussian.cost.stratified_cost up to N_S, docs/
          GAUSSIAN.md §2.1) and the raw mean; theirs also in one batched call (its per-call overhead). Prints the 1 Hz
          crossing of each sampler (log-linear between the two n around it, as find_threshold).
  --sampler walrus|piquasso: the same check / time with The Walrus 0.22.0 or Piquasso's CPU sampler as "theirs", run
          in its own venv (bench/cpu_sampler_worker.py); no batched call.
  gpucheck: their backend="gpu" against backend="cpu" on the same instances, GPU utilisation polled and the calls
          into their CUDA extension counted (is their GPU path what is timed?).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter
from itertools import pairwise

import numpy as np
import torch

from photonic_engines import GaussianInput
from photonic_engines import gaussian as ours
from photonic_engines.gaussian.cost import detected_law, run_cutoff, stratified_cost

_GBSK = []


def gbsk_sample(*args, **kw):
    """GBSKernels' `sampling.sampler.sample`, imported from $GBSK_SRC on first use."""
    if not _GBSK:
        sys.path.insert(0, os.environ.get("GBSK_SRC", os.path.expanduser("~/gbsk/GBSKernels")))
        if os.environ.get("GBSK_SITE"):            # the ~/gbsk venv's packages, last: never shadow the system's
            sys.path.append(os.environ["GBSK_SITE"])
        from sampling.sampler import sample
        _GBSK.append(sample)
    return _GBSK[0](*args, **kw)


def haar(m, seed):                         # bench/find_threshold.py haar, verbatim (the row's instances)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def instance(n, c, r, seed):
    m = round(c * n)
    U = haar(m, seed + n)
    st = GaussianInput.smsv(n, m, r)
    rs = np.zeros(m); rs[:n] = r
    S = np.block([[U.real, -U.imag], [U.imag, U.real]])           # a -> U a on (x, p), xxpp
    V = S @ np.diag(np.r_[np.exp(-2 * rs), np.exp(2 * rs)]) @ S.T  # hbar = 2
    mean = (np.abs(U[:, :n]) ** 2) @ np.full(n, math.sinh(r) ** 2)
    return m, U, st, V, mean


def theirs(V, k, seed, cutoff):
    return gbsk_sample(V, k, cutoff=cutoff, backend="gpu", seed=seed)


def their_shots(a, n, U, V, shots, budget, seed, NS=0):
    """Their shots one call each, until `shots` or `budget` seconds: (patterns, seconds per call). GBSKernels runs in
    this process; The Walrus / Piquasso run in their own venv ($CPY, bench/cpu_sampler_worker.py)."""
    if a.sampler == "gbsk":
        theirs(V, 1, 0, a.cutoff)                                        # warm-up (workspace, first launch)
        S, ts, t_all = [], [], time.perf_counter()
        while len(ts) < shots and time.perf_counter() - t_all < budget:
            t = time.perf_counter(); S.append(np.asarray(theirs(V, 1, seed + len(ts), a.cutoff))[0])
            ts.append(time.perf_counter() - t)
        return np.array(S), np.array(ts)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fi, fo = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(fi, sampler=a.sampler, V=V, U=U, n=n, r=a.r, shots=shots, budget=budget, seed=seed,
                 cutoff=a.walrus_cutoff, seed_once=a.seed_once, max_photons=NS if a.walrus_max_ns else 0)
        cmd = [os.environ.get("CPY", "python3"), os.path.join(os.path.dirname(__file__), "cpu_sampler_worker.py"), fi, fo]
        a.censored = 0
        p = subprocess.Popen(cmd)
        while p.poll() is None:  # --kill-shot-s: watchdog on the shot in flight (P 10-11: kill at 400 s, count it)
            time.sleep(1.0)
            if a.kill_shot_s and os.path.exists(fo + ".cur"):
                with open(fo + ".cur") as f:
                    cur = f.read().split()
                if len(cur) != 2:                        # written atomically now; never trust a partial read
                    continue
                t0, sd = cur
                if time.time() - float(t0) > a.kill_shot_s:
                    p.kill(); p.wait()
                    a.censored = 1
                    print(f"  shot killed at {a.kill_shot_s:.0f} s (n = {n}, seed {sd}): censored", flush=True)
                    if not os.path.exists(fo + ".part.npz"):
                        return np.zeros((0, len(V) // 2), dtype=np.int64), np.zeros(0)
                    out = np.load(fo + ".part.npz")
                    return out["samples"], out["seconds"]
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, cmd)
        out = np.load(fo)
        return out["samples"], out["seconds"]


def check(a):
    for n in a.n:
        m, U, st, V, mean = instance(n, a.ratio, a.r, a.seed)
        NS = run_cutoff(detected_law(st, 12 * m + 60), a.run_samples)
        t = time.perf_counter(); A, _ = their_shots(a, n, U, V, a.shots, a.budget, 1, NS); ta = time.perf_counter() - t
        k = len(A)                                       # fewer than --shots when --budget ran out: ours matched
        B = np.asarray(ours.sample(st, U, k, seed=2, device=a.device))
        C = np.asarray(ours.sample(st, U, k, seed=3, device=a.device))
        se = np.sqrt(np.maximum(mean * (1 + mean) * 2, 1e-12) / k)   # rough: var <= 2 n (n + 1) per mode
        za, zb = np.abs(A.mean(0) - mean) / se, np.abs(B.mean(0) - mean) / se

        def tvd(X, Y):
            cx, cy = Counter(map(tuple, X)), Counter(map(tuple, Y))
            return 0.5 * sum(abs(cx[j] - cy[j]) for j in set(cx) | set(cy)) / k
        print(f"check {a.sampler} n={n} m={m} shots={k}: per-mode mean |z| max theirs {za.max():.2f} ours {zb.max():.2f}; "
              f"N mean theirs {A.sum(1).mean():.4f} ours {B.sum(1).mean():.4f} exact {mean.sum():.4f}; "
              f"TVD theirs-ours {tvd(A, B):.4f}, ours-ours {tvd(B, C):.4f} (floor); theirs {ta:.1f} s", flush=True)
        if a.sampler != "gbsk":                          # where a bias sits (run 20261011_123902: Walrus n = 3 low)
            print("  per-mode mean theirs / ours / exact: " + ", ".join(
                f"{x:.3f}/{y:.3f}/{e:.3f}" for x, y, e in zip(A.mean(0), B.mean(0), mean, strict=True)), flush=True)
            print("  P(N) N=0..6 theirs / ours: " + ", ".join(
                f"{np.mean(A.sum(1) == j):.4f}/{np.mean(B.sum(1) == j):.4f}" for j in range(7)), flush=True)


def gpucheck(a):
    """Did their CUDA backend run? Per n, k shots with backend="gpu" then "cpu" (their defaults otherwise), each with
    the GPU utilisation sampled by nvidia-smi every 0.1 s and the calls into their extension counted."""
    import contextlib
    import subprocess
    import threading

    import gbskernels as g  # importable once gbsk_sample has set the path
    calls = Counter()
    ext = g._load_gpu_ext()                    # their extension module (private loader, gbskernels/__init__.py)
    for name in dir(ext):
        f = getattr(ext, name)
        if callable(f) and not name.startswith("_") and not isinstance(f, type):
            def wrap(*x, _f=f, _n=name, **kw):
                calls[_n] += 1
                return _f(*x, **kw)
            with contextlib.suppress(AttributeError, TypeError):
                setattr(ext, name, wrap)
    print(f"their backend: gpu_available {g.gpu_available()}, kind {g.gpu_backend_kind()}, ext {ext.__file__}", flush=True)
    for n in a.n:
        m, U, st, V, mean = instance(n, a.ratio, a.r, a.seed)
        for be in ("gpu", "cpu"):
            util, stop = [], threading.Event()

            def poll(util=util, stop=stop):
                while not stop.is_set():
                    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True).stdout.strip()
                    util.append(float(out.split()[0]) if out else float("nan"))
                    time.sleep(0.1)
            calls.clear()
            th = threading.Thread(target=poll); th.start()
            t = time.perf_counter(); gbsk_sample(V, a.shots, cutoff=a.cutoff, backend=be, seed=5); dt = time.perf_counter() - t
            stop.set(); th.join()
            print(f"gpucheck n={n} m={m} backend={be}: {a.shots} shots {dt:.2f} s ({dt / a.shots * 1e3:.2f} ms/shot); GPU util "
                  f"mean {np.nanmean(util):.0f} % max {np.nanmax(util):.0f} % ({len(util)} polls); ext calls "
                  f"{sum(calls.values())} {dict(calls.most_common(4))}", flush=True)


def crossing(rows, key):
    pts = [(r["n"], r[key]) for r in rows if r[key] == r[key] and r[key] > 0]
    for (n0, t0), (n1, t1) in pairwise(pts):
        if t0 <= 1.0 < t1:
            return n0 + (n1 - n0) * math.log(1.0 / t0) / math.log(t1 / t0)
    return float("nan")


def next_ok(rows, n, a, key):
    """False when the next n, extrapolated log-linearly from the last two measured t_S, would cost more than
    3 --stop-s per shot (one shot is one uninterruptible call: run 20261011_103809 lost its check to a ×16 per two
    modes growth of their sampler)."""
    pts = [(r["n"], r[key]) for r in rows if r[key] == r[key] and r[key] > 0]
    nxt = [k for k in a.n if k > n]
    if len(pts) < 2 or not nxt:
        return True
    (n0, t0), (n1, t1) = pts[-2:]
    return t1 * (t1 / t0) ** ((nxt[0] - n1) / (n1 - n0)) < 3 * a.stop_s


def timing(a):
    rows = []
    run_t, run_o = True, a.ours
    for n in a.n:
        m, U, st, V, mean = instance(n, a.ratio, a.r, a.seed)
        law = detected_law(st, 12 * m + 60)
        NS = run_cutoff(law, a.run_samples)
        row = {"n": n, "m": m, "r": a.r, "N_S": NS}
        for who in ("theirs", "ours"):
            for k in ("tS", "raw", "ext", "shots", "batch", "censored"):
                row[f"{who}_{k}"] = float("nan")
        if run_t:
            a.censored = 0
            S, ts = their_shots(a, n, U, V, a.max_shots, a.budget, 1000 * n, NS)
            row["theirs_censored"] = a.censored
            if not len(ts):
                print(f"n={n}: no shot of theirs completed (censored {a.censored})", flush=True)
                rows.append(row)
                break
            Ns, ts = [int(x) for x in S.sum(1)], list(ts)
            if a.censored:       # P 10-11: counted at the kill time, in the top stratum (its N is unknown) -> lower bound
                Ns.append(NS); ts.append(a.kill_shot_s)
            e = stratified_cost(Ns, ts, law, max_N=NS)
            k = max(1, min(a.batch, len(ts)))
            if a.sampler == "gbsk" and sum(ts) / len(ts) * k < a.budget / 2:
                t = time.perf_counter(); theirs(V, k, 7 + n, a.cutoff); row["theirs_batch"] = (time.perf_counter() - t) / k
            row.update(theirs_tS=e.seconds, theirs_raw=float(np.mean(ts)), theirs_ext=e.extrapolated, theirs_shots=len(ts))
            run_t = (e.seconds < a.stop_s and max(ts) < (a.max_shot_s or a.budget / 2)
                     and next_ok([*rows, row], n, a, "theirs_tS"))
        if run_o:
            ours.sample(st, U, 2, seed=0, device=a.device)                # warm-up
            Ns, ts, k, t_all = [], [], 16, time.perf_counter()
            while len(ts) < a.max_shots_ours and time.perf_counter() - t_all < a.budget:
                _, stt = ours.sample(st, U, k, seed=1000 * n + len(ts), device=a.device, return_stats=True)
                Ns += list(stt.photons); ts += list(stt.seconds); k = min(2 * k, 1024)
            e = stratified_cost(Ns, ts, law, max_N=NS)
            row.update(ours_tS=e.seconds, ours_raw=float(np.mean(ts)), ours_ext=e.extrapolated, ours_shots=len(ts))
            run_o = e.seconds < a.stop_s and next_ok([*rows, row], n, a, "ours_tS")
        rows.append(row)
        print(f"n={n} m={m} N_S={NS}: theirs t_S {row['theirs_tS']:.4g} s (raw {row['theirs_raw']:.4g}, batched "
              f"{row['theirs_batch']:.4g}, {row['theirs_ext']:.0%} ext., {row['theirs_shots']} shots, "
              f"censored {row['theirs_censored']}) | ours t_S "
              f"{row['ours_tS']:.4g} s (raw {row['ours_raw']:.4g}, {row['ours_ext']:.0%} ext., {row['ours_shots']} shots)"
              f"  [{time.strftime('%H:%M:%S')}]", flush=True)
        if a.csv:
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
        if not (run_t or run_o):
            break
    print(f"1 Hz crossing (t_S, S = {a.run_samples:.0e}): theirs n* = {crossing(rows, 'theirs_tS'):.2f}, "
          f"ours n* = {crossing(rows, 'ours_tS'):.2f} (nan = not bracketed in this n range)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "time", "gpucheck"])
    ap.add_argument("--sampler", choices=["gbsk", "walrus", "piquasso"], default="gbsk",
                    help="theirs: GBSKernels (gpu, in process) or The Walrus / Piquasso (CPU, $CPY venv, cpu_sampler_worker)")
    ap.add_argument("--n", default="3", help="squeezer counts, comma-separated, ascending")
    ap.add_argument("--ratio", type=float, default=2.0, help="m / n")
    ap.add_argument("--mean-photons", type=float, default=0.5, help="sinh^2 r per squeezer")
    ap.add_argument("--seed", type=int, default=0, help="U = haar(m, seed + n), as find_threshold")
    ap.add_argument("--cutoff", type=int, default=6, help="GBSKernels' per-mode cutoff (its default 6)")
    ap.add_argument("--walrus-cutoff", type=int, default=0, help="diagnosis only: The Walrus cutoff (0: its default 5)")
    ap.add_argument("--max-ns", "--walrus-max-ns", dest="walrus_max_ns", action="store_true",
                    help="exact setting: photon cap = N_S of each n (The Walrus max_photons, default 30; Piquasso "
                         "measurement_cutoff, default 5: both truncate the law)")
    ap.add_argument("--seed-once", action="store_true", help="diagnosis: seed The Walrus once, not per shot")
    ap.add_argument("--shots", type=int, default=20000, help="check: shots per sampler")
    ap.add_argument("--budget", type=float, default=300.0, help="time: seconds per sampler per n")
    ap.add_argument("--max-shots", type=int, default=256, help="time: their shots per n at most")
    ap.add_argument("--max-shots-ours", type=int, default=4096, help="time: our shots per n (may overshoot by one batch)")
    ap.add_argument("--batch", type=int, default=32, help="time: shots in their one batched call")
    ap.add_argument("--max-shot-s", type=float, default=0.0,
                    help="time: drop theirs after a shot longer than this (0: --budget / 2; a rare bright shot of "
                         "Piquasso, N = 50 > N_S, took 60 s at n = 22 in run 20261011_132646)")
    ap.add_argument("--kill-shot-s", type=float, default=0.0,
                    help="CPU samplers: kill a shot running longer than this and count it at this time (0: never)")
    ap.add_argument("--stop-s", type=float, default=30.0, help="time: drop a sampler once its t_S exceeds this")
    ap.add_argument("--run-samples", type=float, default=1e4, help="S of t_S (our headline: 1e4)")
    ap.add_argument("--no-ours", dest="ours", action="store_false", help="time: theirs only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    a.n = [int(x) for x in a.n.split(",")]
    a.r = math.asinh(math.sqrt(a.mean_photons))
    print(f"gbskernels_bench {a.mode} {a.sampler}: c={a.ratio} r={a.r:.4f} cutoff={a.cutoff} seed={a.seed} torch {torch.__version__} "
          f"numpy {np.__version__} threads {os.cpu_count()}", flush=True)
    if a.mode == "gpucheck":
        gbsk_sample(np.eye(2), 1, backend="cpu")    # sets the import path, loads their package
    {"check": check, "time": timing, "gpucheck": gpucheck}[a.mode](a)


if __name__ == "__main__":
    main()
