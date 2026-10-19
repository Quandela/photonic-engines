"""Loop-interferometer frontier for the three single-rail inputs of the paper's part (b): the photon (or source)
number N* at which the native register sampler sustains --target samples/s, as a curve N*(L) — the same method,
settings and output as bench/timeloop_frontier.py (the LBS row), so the rows sit side by side.

    python3 bench/hybrid_frontier.py --input fock    --loops 1,2,3,4,6,8 [--ratios 2]     # reproduces the LBS row
    python3 bench/hybrid_frontier.py --input emitter --loops 1,2,3,4,6,8 [--ratios 2]     # one QD emitter, time-bin
    python3 bench/hybrid_frontier.py --input sbs     --loops 1,2,3,4,6,8 [--alpha 0.785]  # N superposed sources
    python3 bench/hybrid_frontier.py --input hybrid --emitters 2 --loops 1,2,3                # k emitters, 2k rails

Inputs (one rail, T = round(ratio * N) bins, loops start empty, random settings theta in [0, 2), phi in [0, 6)):
  fock    : N single photons in bins 0..N-1 (`LoopRegisterKernel`), register C(N + L + 1, L + 1).
  emitter : one spin emitter, the N-photon linear-cluster caterpillar (spin gate theta = pi/2 before every emission),
            photon b dual-rail in time bins (2b, 2b + 1) — the period is 2 bins (`LoopRegisterKernel(emitter=)`,
            docs/TIME_LOOP.md §5); traced spin; register 2 x C(N + L + 2, L + 2). ratio >= 2 (T rounded up to even).
  sbs     : N sources cos(alpha)|0> + sin(alpha)|1> in bins 0..N-1 (`SuperpositionLoopKernel`); the photon number is
            indefinite, so the register carries every number up to N: C(N + L + 1, L + 1), the Fock register.
  hybrid  : k spin emitters (`HybridRegisterKernel`, docs/HYBRID.md §1-4), emitter j dual-rail on rails (2j, 2j + 1),
            one photon per emitter per tick, caterpillar gates theta = pi/2, a loop of delay L on every one of the
            m = 2k rails, a static Haar rail mixer U (seeded) after the loops every tick; traced spins. The search
            variable is the tick count T; N = k T photons is what is reported. Register 2^k C(N + R, R), R = m (1 + L);
            sectors only. --ratios is ignored (one emission per emitter per tick).
--ladder "1,3,9" replaces the single loop of delay L by that ladder (L = its sum); --loop-eta applies per-round-trip
loss as Monte-Carlo jumps on the register (uniform loss is output thinning and costs nothing). Writes --csv
(input, ratio, L, ladder, N, T, chi, D, shots, samples/s, ms/sample) and a result block per (ratio, L).
"""
import argparse
import csv
import gc
import math
import threading
import time
from math import comb
from types import SimpleNamespace

import numpy as np
import torch

from photonic_engines import EmitterMPS, ProductSuperposition, spin_gate
from photonic_engines.sequence import BinEmitter
from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit
from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel, light_hybrid_kernel, light_loop_kernel, loop_sector_sample

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="emitter", choices=["fock", "emitter", "sbs", "hybrid"])
ap.add_argument("--emitters", type=int, default=2, help="hybrid: number k of spin emitters (2k rails)")
ap.add_argument("--loops", default="1,2,3,4,6,8", help="total loop lengths L (one loop of delay L each)")
ap.add_argument("--ladder", default=None, help="one ladder of delays instead, e.g. 1,3,9")
ap.add_argument("--ratios", default="2", help="bin ratios T/N, comma separated")
ap.add_argument("--spacing", type=int, default=1, help="fock / sbs: one source every SPACING bins (1 = bins 0..N-1, the "
                                                      "LBS convention; 2 = bins 0, 2, 4, ..., the emitter's photon flux)")
ap.add_argument("--alpha", type=float, default=math.pi / 4, help="sbs: source cos(alpha)|0> + sin(alpha)|1>")
ap.add_argument("--sampler", default="auto", choices=["auto", "dense", "sector"],
                help="sector: photon-number sectors (docs/HYBRID.md §6; the method of record for fock and emitter; "
                     "memory found by trying, OOM = the memory limit); dense: the register of every photon "
                     "number <= N (run 2's method; the only one for sbs, whose photon number is indefinite); auto = sector "
                     "for fock/emitter, dense for sbs")
ap.add_argument("--scan-l", default=None, help="instead of N*(L): for each N in this list, the loop delay L* (one loop)"
                                               " at which the rate falls to --target")
ap.add_argument("--l-start", type=int, default=1)
ap.add_argument("--max-l", type=int, default=40)
ap.add_argument("--loop-eta", type=float, default=None, help="per-round-trip transmission (register jumps)")
ap.add_argument("--target", type=float, default=1.0)
ap.add_argument("--start", type=int, default=8)
ap.add_argument("--max-n", type=int, default=400, help="largest N searched (hybrid: largest tick count T)")
ap.add_argument("--max-jump", type=float, default=1.6, help="max multiplicative jump in N per step")
ap.add_argument("--budget", type=float, default=120.0, help="seconds per N (shot ramp stops beyond budget/3 per run)")
ap.add_argument("--max-shots", type=int, default=256)
ap.add_argument("--fast", action="store_true", help="no effect, kept for older command lines: since 10-12 the sector "
                                                    "march v3 and the one-loop Fock chain are the only paths")
ap.add_argument("--rate-at-max", action="store_true", help="rate from the largest batch reached alone (for S = 10^4 "
                                                          "quoting; a lower bound on the S = 10^4 rate if the ramp stops earlier)")
ap.add_argument("--heartbeat", type=float, default=1800.0, help="seconds between progress lines while a point is running "
                                                               "(a sector-table build can run for long without output)")
ap.add_argument("--warmup", type=int, default=16, help="sector: shots of the untimed warm-up (builds most sector tables)")
ap.add_argument("--mem-frac", type=float, default=0.4, help="fraction of device memory the shot batch may use")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="hybrid_frontier.csv")
args = ap.parse_args()
if args.sampler == "auto":
    args.sampler = "dense" if args.input == "sbs" else "sector"

dev = args.device
cdtype = getattr(torch, args.dtype)
bytes_per = 16 if cdtype == torch.complex128 else 8
if dev == "cuda":
    mem = torch.cuda.get_device_properties(0).total_memory
    name = torch.cuda.get_device_name(0)
else:
    mem = 32 * 2 ** 30
    name = "cpu"
if args.input == "hybrid" and args.sampler == "dense":
    raise SystemExit("hybrid: sector sampler only (the dense register of 2k rails does not fit at useful sizes)")
KF = args.emitters if args.input == "hybrid" else 1                 # reported photons per search step (hybrid: per tick)
if args.sampler == "sector" and args.input == "sbs":
    raise SystemExit("sbs: the photon number is indefinite, the sector sampler does not apply")
print(f"{name}  loop frontier  input={args.input}  sampler={args.sampler}  ratios T/N={args.ratios}  target={args.target} samples/s  "
      f"dtype={args.dtype}  loop_eta={args.loop_eta}", flush=True)
rows = []
progress = {"t0": time.perf_counter(), "point": "start-up", "since": time.perf_counter()}


def heartbeat():
    """Progress line every --heartbeat seconds: elapsed time, the point in progress and for how long, points done and
    the last measured rate, so a long silent point (table build, large batch) is visibly not a stall."""
    while True:
        time.sleep(args.heartbeat)
        now = time.perf_counter()
        last = f", last: N={rows[-1][4]} L={rows[-1][2]} {rows[-1][9]:.3f}/s" if rows else ""
        print(f"  [progress] {now - progress['t0']:.0f} s elapsed; running {progress['point']} for "
              f"{now - progress['since']:.0f} s; {len(rows)} points done{last}", flush=True)


if args.heartbeat > 0:
    threading.Thread(target=heartbeat, daemon=True).start()


def size(N, L):
    """(chi, D) of the register for N photons / sources and total delay L."""
    if args.input == "emitter":
        return 2, comb(N + L + 2, L + 2)
    if args.input == "hybrid":
        R = 2 * args.emitters * (1 + L)
        return 2 ** args.emitters, comb(KF * N + R, R)
    return 1, comb(N + L + 1, L + 1)


def bins(N, ratio):
    if args.input == "hybrid":
        return N                                                     # ticks
    T = max(round(ratio * N), (N - 1) * args.spacing + 1)
    if args.input == "emitter":
        T = max(T, 2 * N)
        T += T % 2
    return T


def kernel(N, T, ladder):
    if args.input == "hybrid":
        k, m = args.emitters, 2 * args.emitters
        ems = [BinEmitter.caterpillar(T, [spin_gate(math.pi / 2, 0.0)] * T, modes=(2 * j, 2 * j + 1)) for j in range(k)]
        K = light_hybrid_kernel(ems, m, tuple(ladder), device=dev, dtype=cdtype)
        z = np.random.default_rng(args.seed).normal(size=(m, m, 2)) @ [1, 1j]       # Haar U (QR with phase fix)
        q, r = np.linalg.qr(z)
        K.bench_U = q * (np.diag(r) / abs(np.diag(r)))
        return SimpleNamespace(n_loops=K.n_dl), K
    c = TimeLoopCircuit(T, ladder)
    make = LoopRegisterKernel if args.sampler == "dense" else light_loop_kernel     # sector: no dense basis built
    if args.input == "fock":
        occ = [0] * T
        for b in range(N):
            occ[b * args.spacing] = 1
        return c, make(c, occ, device=dev, dtype=cdtype)
    if args.input == "sbs":
        st = ProductSuperposition.sbs(N, c.m, alpha=args.alpha, modes=range(0, N * args.spacing, args.spacing))
        return c, SuperpositionLoopKernel(c, st, device=dev, dtype=cdtype)
    st = EmitterMPS.caterpillar(N, [spin_gate(math.pi / 2, 0.0)] * N, m=c.m, modes=[(2 * b, 2 * b + 1) for b in range(N)])
    return c, make(c, [0] * T, device=dev, dtype=cdtype, emitter=st)


def free_device():
    """Drop the previous point's kernel before the next one: the loop kernels hold reference cycles (the exit-row table
    points back to its kernel), so their device tensors and sector tables survive until a gc pass."""
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def measure(N, ladder, ratio):
    free_device()
    L = sum(ladder)
    progress.update(point=f"N={KF * N} L={L}", since=time.perf_counter())
    T = bins(N, ratio)
    chi, D = size(N, L)
    per_shot = chi * D * bytes_per * 6                            # state, gathered blocks, products (~6 copies)
    shots_fit = int(args.mem_frac * mem // per_shot) if args.sampler == "dense" else args.max_shots
    if shots_fit < 1:
        print(f"  N={N:3d} L={L} T={T:3d} chi*D={chi * D:>12,d}: does not fit ({per_shot / 2**30:.1f} GB per shot)", flush=True)
        return None
    t0 = time.perf_counter()
    c, lk = kernel(N, T, ladder)
    build = time.perf_counter() - t0
    g = torch.Generator(device=dev).manual_seed(args.seed)
    th = torch.rand(c.n_loops, T, generator=g, device=dev, dtype=torch.float64) * 2
    ph = torch.rand(c.n_loops, T, generator=g, device=dev, dtype=torch.float64) * 6
    loss = None if args.loop_eta is None else {"loop": args.loop_eta}
    stats = {}

    def draw(s):
        if args.sampler == "dense":
            return lk.sample(s, th, ph, generator=g, loss=loss)
        if args.input == "hybrid":
            out, st = lk.sample_sectors(s, th, ph, lk.bench_U, generator=g, loss=loss, return_stats=True)
        else:
            out, st = loop_sector_sample(lk, s, th, ph, generator=g, loss=loss, return_stats=True)
        stats["amp"] = max(stats.get("amp", 0), st["max_amplitudes"])
        stats["n"] = max(stats.get("n", 0.0), max(st["mean_n"]))
        stats["tab"] = st["table_seconds"]                          # of this call: subtracted from its time below
        stats["tab_total"] = stats.get("tab_total", 0.0) + st["table_seconds"]
        return out
    shots, spent, tot_s, tot_t, last, dt, dts, dts_last = 1, 0.0, 0, 0.0, 1, 0.0, 0.0, 1e-9
    try:
        draw(args.warmup if args.sampler == "sector" else 1)        # warm-up (sector: builds most tables untimed)
    except torch.cuda.OutOfMemoryError:
        free_device()
        try:
            draw(1)                                                 # the batched warm-up may not fit where one shot does
        except torch.cuda.OutOfMemoryError:
            print(f"  N={KF * N:3d} L={L} T={T:3d}: out of memory at one shot", flush=True)
            return None
    if dev == "cuda":
        torch.cuda.synchronize()
    while shots <= min(args.max_shots, shots_fit):
        t = time.perf_counter()
        try:
            draw(shots)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        dts = max(dt - stats.get("tab", 0.0), 1e-9)                 # sampling only: sector tables are a one-time, cached
        spent += dts                                                # host cost, like the dense basis build
        last, dts_last = shots, dts
        if shots >= 4 or shots == shots_fit:
            tot_s += shots
            tot_t += dts
        if dts > args.budget / 3 or spent > args.budget:
            break
        shots *= 2
    if spent == 0.0:                                                # the first timed batch already ran out of memory:
        free_device()                                               # no rate (it used to read 1 shot / 1e-9 s = 1e9/s)
        print(f"  N={KF * N:3d} L={L} T={T:3d}: out of memory in the first timed batch", flush=True)
        return None
    if tot_s == 0 or args.rate_at_max:                              # --rate-at-max: the largest batch reached alone
        tot_s, tot_t = last, dts_last
    rate = tot_s / tot_t
    sec = (f", sector max chi*D {stats['amp']:,d}, max mean n {stats['n']:.2f}, tables {stats['tab_total']:.1f} s"
           if stats else "")
    print(f"  N={KF * N:3d} L={L} T={T:3d} chi*D={chi * D:>12,d}: {rate:9.3f} samples/s  ({1e3 / rate:9.2f} ms/sample, "
          f"{'at the batch of' if args.rate_at_max else 'up to'} {last} shots, build {build:.1f} s{sec})", flush=True)
    rows.append([args.input if KF == 1 else f"hybrid k={KF}", ratio, L, "-".join(map(str, ladder)), KF * N, T,
                 chi, D, last, rate, 1e3 / rate,
                 "" if args.loop_eta is None else args.loop_eta, args.sampler, stats.get("amp", ""), stats.get("n", ""),
                 round(stats.get("tab_total", 0.0), 2) if stats else "", args.spacing,
                 "largest batch" if args.rate_at_max else "batches >= 4 pooled"])
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["input", "ratio", "L", "ladder", "N", "T", "chi", "D", "shots", "samples_per_s", "ms_per_sample", "loop_eta",
                    "sampler", "sector_max_amplitudes", "sector_max_mean_n", "table_seconds", "spacing", "rate_basis"])
        w.writerows(rows)
    return rate


if args.scan_l:                                                    # L*(N): the frontier in the loop delay at fixed N
    ratio = float(args.ratios.split(",")[0])
    print("=== L* scan ===")
    for N in (int(x) for x in args.scan_l.split(",")):
        pts = {}
        for L in range(args.l_start, args.max_l + 1):
            r = measure(N, (L,), ratio)
            if r is None:
                break
            pts[L] = r
            if r < args.target:
                break
        lo = [l for l in sorted(pts) if pts[l] >= args.target]
        hi = [l for l in sorted(pts) if pts[l] < args.target]
        if lo and hi:
            a, b = max(lo), min(hi)
            l_int = a + (math.log(args.target) - math.log(pts[a])) / (math.log(pts[b]) - math.log(pts[a])) * (b - a)
            print(f"  input = {args.input}  N = {N}: bracket L={a} at {pts[a]:.2f}/s, L={b} at {pts[b]:.2f}/s  ->  L* = {l_int:.2f}",
                  flush=True)
        else:
            print(f"  input = {args.input}  N = {N}: not bracketed (points {sorted(pts)})", flush=True)
    print(f"  wrote {args.csv}")
    raise SystemExit(0)

ladders = [tuple(int(x) for x in args.ladder.split(","))] if args.ladder else [(int(L),) for L in args.loops.split(",")]
results = {}
for ratio in (float(r) for r in args.ratios.split(",")):
    for ladder in ladders:
        L = sum(ladder)
        pts = {}
        N = args.start
        reason = "bracketed"
        while True:
            r = measure(N, ladder, ratio)
            if r is None:
                reason = "memory"
                break
            pts[N] = r
            ns = sorted(pts)
            lo = [k for k in ns if pts[k] >= args.target]
            hi = [k for k in ns if pts[k] < args.target]
            if lo and hi and min(hi) - max(lo) <= 1:
                break
            if lo and hi:
                nxt = (max(lo) + min(hi)) // 2
            else:
                if len(ns) >= 2:
                    slope = np.polyfit(np.log(ns[-2:]), np.log([pts[k] for k in ns[-2:]]), 1)[0]
                    slope = min(slope, -1.0)
                else:
                    slope = -(L + 3.0)
                n_pred = N * math.exp((math.log(args.target) - math.log(pts[N])) / slope)
                n_pred = min(max(n_pred, N / args.max_jump), N * args.max_jump)
                nxt = math.floor(n_pred) if not hi else math.ceil(n_pred)
                nxt = max(nxt, N + 1) if not hi else min(nxt, min(hi) - 1)
            if nxt in pts or nxt > args.max_n or nxt < 1:
                reason = "max-n"
                break
            N = nxt
        ns = sorted(pts)
        lo = [k for k in ns if pts[k] >= args.target]
        hi = [k for k in ns if pts[k] < args.target]
        print("=== result ===")
        tag = f"input = {args.input}  ratio = {ratio:g}  L = {L} ({'-'.join(map(str, ladder))})"
        if lo and hi:
            a, b = max(lo), min(hi)
            n_int = a + (math.log(args.target) - math.log(pts[a])) / (math.log(pts[b]) - math.log(pts[a])) * (b - a)
            print(f"  {tag}: bracket N={KF * a} at {pts[a]:.2f}/s, N={KF * b} at {pts[b]:.2f}/s  ->  N* = {KF * n_int:.1f}")
            results[(ratio, L)] = KF * n_int
        elif lo:
            chi, D = size(max(lo), L)
            print(f"  {tag}: not bracketed ({reason}): last N = {KF * max(lo)} at {pts[max(lo)]:.2f}/s, chi*D = {chi * D:,d}")
            results[(ratio, L)] = KF * max(lo)
        else:
            print(f"  {tag}: below target already at N = {ns[0] if ns else args.start}")
print("=== summary ===")
for (ratio, L), n in results.items():
    print(f"  input={args.input} ratio={ratio:g} L={L}: N*={n:.1f}")
print(f"  wrote {args.csv}")
