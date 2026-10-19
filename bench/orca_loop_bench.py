"""ORCA's public loop simulator against our loop rows.

Their code (github.com/ORCAComputing/loop-progressive-simulator, Apache-2.0) runs as shipped in its own venv through
bench/orca_loop_worker.py ($OPY, clone $ORCA_SRC; scripts/orca_setup.sh). Instances are those of our single-photon loop
rows (bench/timeloop_frontier.py, docs/TIME_LOOP.md §4): one loop of delay L, T = m = 2N bins, N photons in bins
0..N-1, random settings. Their convention: a loop is the gates (i, i + L), i = 0..m-L-1 (the loop flushed at the end,
no photon left in it), angle theta uniform in [0, 2 pi).

  check: small N, their samples and ours (photonic_engines.clifford.fock_sample on the unitary of THEIR gate list, H100)
         against the exact law (permanents). The unitary is built from their gates in both orientations (U, U^T):
         the one their samples match is printed, so a convention slip cannot pass silently.
  time : per L, their time per sample at increasing N (all shots carry N photons: t_S = the mean), until their
         1 sample/s crossing N* is bracketed or the next N extrapolates past --stop-s per sample. Ours: our rows
         (dense LoopRegisterKernel and sector sampler, H100) from docs/TIME_LOOP.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import tempfile
import time
from collections import Counter
from itertools import combinations_with_replacement, pairwise

import numpy as np

CENSORED = [0]
LAST = {}                                       # the worker's per-sample pass index and new numba compilations


LADDER = []                                       # --ladder (P 10-12): delays of a multi-loop ladder, L = their sum


def ladder_of(L):
    return list(LADDER) if LADDER else [L]


def orca(L, N, shots, budget, seed, timeout, kill_s=0.0, passes=1):
    """Their samples / seconds per sample / gates / key lengths for one instance, or None when nothing came back.
    kill_s > 0 (P 10-11): a sample running longer is killed, the samples before it kept, CENSORED[0] = 1 (the caller
    counts it at kill_s); otherwise the whole worker is killed after `timeout`."""
    CENSORED[0] = 0
    LAST.clear()
    with tempfile.TemporaryDirectory() as tmp:
        fi, fo = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(fi, L=L, N=N, m=2 * N, shots=shots, budget=budget, seed=seed, partial=kill_s > 0, passes=passes,
                 ladder=np.array(ladder_of(L)))
        cmd = [os.environ.get("OPY", "python3"), os.path.join(os.path.dirname(__file__), "orca_loop_worker.py"), fi, fo]
        p, t0 = subprocess.Popen(cmd), time.time()
        while p.poll() is None:
            time.sleep(0.5)
            cur = None
            if kill_s and os.path.exists(fo + ".cur"):
                with open(fo + ".cur") as f:
                    cur = f.read().split() or None       # atomic writes now; still never trust a partial read
                if cur and len(cur) != 2:
                    cur = None
            if (cur and time.time() - float(cur[0]) > kill_s) or time.time() - t0 > timeout:
                p.kill(); p.wait()
                CENSORED[0] = 1
                print(f"  L={L} N={N}: sample {cur[1] if cur else '?'} killed (> {kill_s or timeout:.0f} s)", flush=True)
                if not os.path.exists(fo + ".part.npz"):
                    return None
                o = np.load(fo + ".part.npz")
                LAST.update({k: o[k] for k in ("pass_idx", "new_compiled") if k in o})
                return o["samples"], o["seconds"], o["gates"], o["key_len"]
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, cmd)
        o = np.load(fo)
        LAST.update({k: o[k] for k in ("pass_idx", "new_compiled") if k in o})
        return o["samples"], o["seconds"], o["gates"], o["key_len"]


def unitary(gates, m, transpose=False):
    """Unitary of their gate list on single photons: a_i^dag -> cos a_i^dag - sin a_j^dag, a_j^dag -> sin a_i^dag +
    cos a_j^dag (their bs_amplitude on one photon, as read), applied in list order; U[out, in]. (theta -> -theta in
    every gate leaves the law unchanged, checked at L = 1, 2: only the orientation needs choosing.)"""
    U = np.eye(m, dtype=complex)
    for i, j, th in gates:
        i, j, c, s = int(i), int(j), math.cos(th), math.sin(th)
        G = np.eye(m, dtype=complex)
        G[i, i], G[i, j], G[j, i], G[j, j] = c, s, -s, c
        U = G @ U
    return U.T if transpose else U


def perm(A):                                   # Ryser, small k
    k = A.shape[0]
    tot = 0j
    for mask in range(1, 1 << k):
        cols = [c for c in range(k) if mask >> c & 1]
        tot += (-1) ** len(cols) * np.prod(A[:, cols].sum(1))
    return (-1) ** k * tot


def exact_law(U, N):
    m = U.shape[0]
    law = {}
    for outs in combinations_with_replacement(range(m), N):
        occ = np.bincount(outs, minlength=m)
        law[tuple(occ)] = abs(perm(U[np.array(outs)][:, :N])) ** 2 / np.prod([math.factorial(x) for x in occ])
    return law


def tvd_exact(S, law):
    c = Counter(map(tuple, S))
    k = len(S)
    return 0.5 * sum(abs(c[p] / k - q) for p, q in law.items()) + 0.5 * sum(v / k for p, v in c.items() if p not in law)


def check(a):
    import torch

    from photonic_engines.clifford import fock_sample
    for L in a.L:
        for N in a.N:
            r = orca(L, N, a.shots, a.budget, a.seed, a.budget * 3 + 120)
            if r is None:
                continue
            S, ts, gates, klen = r
            m = 2 * N
            laws = {t: exact_law(unitary(gates, m, t), N) for t in (False, True)}
            tv = {t: tvd_exact(S, laws[t]) for t in (False, True)}
            best = min(tv, key=tv.get)
            U = torch.tensor(unitary(gates, m, best), dtype=torch.complex128, device=a.device)
            g = torch.Generator(device=a.device).manual_seed(a.seed)
            O = fock_sample(U, list(range(N)), len(S), generator=g).cpu().numpy()
            pats, q = zip(*laws[best].items(), strict=True)
            pats, q = np.array(pats), np.array(q)
            rng = np.random.default_rng(a.seed)        # floor: 40 multinomial draws of the exact law, same size
            fl = np.array([tvd_exact(pats[rng.choice(len(q), size=len(S), p=q / q.sum())], laws[best]) for _ in range(40)])
            to = tvd_exact(O, laws[best])
            print(f"check L={L} N={N} m={m} shots={len(S)}: TVD theirs-exact {tv[best]:.4f} (z {(tv[best] - fl.mean()) / fl.std():+.1f}; "
                  f"U{'^T' if best else ''}, other orientation {tv[not best]:.4f}), ours-exact {to:.4f} (z "
                  f"{(to - fl.mean()) / fl.std():+.1f}), floor {fl.mean():.4f} +- {fl.std():.4f} (40 exact draws); "
                  f"{len(q)} patterns, sum p {q.sum():.12f}; their {np.mean(ts[1:]) * 1e3:.2f} ms/sample", flush=True)
            # is the right-padding of their shorter sample keys the right reading? per key length: samples, and how
            # many land on a pattern of exact probability 0 (impossible: a misread key)
            law = laws[best]
            print("  key length: samples / on impossible patterns: " + ", ".join(
                f"{k}: {int((klen == k).sum())} / {sum(law.get(tuple(s), 0.0) < 1e-12 for s in S[klen == k])}"
                for k in sorted(set(klen.tolist()))), flush=True)


def crossing(pts):
    for (n0, t0), (n1, t1) in pairwise(pts):
        if t0 <= 1.0 < t1:
            return n0 + (n1 - n0) * math.log(1.0 / t0) / math.log(t1 / t0)
    return float("nan")


def timing(a):
    rows = []
    for L in a.L:
        pts = []
        for N in a.N:
            if N <= L:
                continue
            if len(pts) >= 2:                          # next N extrapolated log-log from the last two points
                (n0, t0), (n1, t1) = pts[-2:]
                pred = t1 * (N / n1) ** (math.log(t1 / t0) / math.log(n1 / n0))
                if pred > a.stop_s:
                    print(f"  L={L}: stop before N={N} (extrapolated {pred:.3g} s/sample)", flush=True)
                    break
            r = orca(L, N, a.max_shots, a.budget, a.seed + N, max(a.budget * 3, a.stop_s * 2) + 120, a.kill_s)
            if r is None or not len(r[1]):
                print(f"=== L={L} N={N}: no sample came back", flush=True)
                break
            S, ts, _, _ = r
            if CENSORED[0]:                             # P 10-11: counted at the kill time (a lower bound on its cost)
                ts = np.append(ts, a.kill_s)
            # one-time costs are front-loaded: the first yield carries their progressive decomposition and numba
            # compilation (1.2-2.9 s), the SECOND the 8-17 s compilation of a later code path (index 1 at every N of
            # run 20261011_150647, the steady samples ~5-7 ms at N = 40-52). Both reported apart (first_s, second_s),
            # out of the per-sample mean (a run of S samples amortises them); a censored sample stays in.
            k0 = 2 if len(ts) > 2 else 0
            t = float(np.mean(ts[k0:]))
            h = ts[k0:][len(ts[k0:]) // 2:]                  # mean of the second half: steady state reached?
            slow = [i for i, x in enumerate(ts) if i and x > 1.0]
            assert (S.sum(1) == N).all(), "photon number not conserved"
            pts.append((N, t))
            rows.append({"L": L, "N": N, "m": 2 * N, "shots": len(ts), "s_per_sample": t,
                         "second_half_s": float(np.mean(h)), "first_s": float(ts[0]),
                         "second_s": float(ts[1]) if len(ts) > 1 else float("nan"), "median_s": float(np.median(ts)),
                         "p90_s": float(np.percentile(ts, 90)), "max_s": float(np.max(ts)), "n_slow": len(slow),
                         "censored": CENSORED[0]})
            print(f"L={L} N={N} m={2 * N}: {t:.4g} s/sample, second half {np.mean(h):.4g}, median {np.median(ts):.3g} "
                  f"({len(ts)} samples, first {ts[0]:.3g} s; samples > 1 s at index {slow[:12]})  "
                  f"[{time.strftime('%H:%M:%S')}]", flush=True)
            if a.samples_csv:
                with open(a.samples_csv, "a") as f:
                    f.writelines(f"{L},{N},{i},{x:.6g}\n" for i, x in enumerate(ts))
            if a.csv:
                with open(a.csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
            if t > 1.0 and len(pts) >= 2:
                break
        print(f"=== L={L}: their 1 sample/s crossing N* = {crossing(pts):.1f} (nan = not bracketed)", flush=True)


def diag(a):
    """Are their slow samples compilations? (run 20261011_152447: > 1 s samples at random indices, up to 118 s, the
    median ~30 ms at N = 128.) Per (L, N): --passes runs of --max-shots samples on the same instance in ONE process
    (new draws each pass); per sample its time and the numba specialisations compiled during it. Compilations are
    cached in the process: if the slow samples are compilations, pass 2 has none of them (run 20261011_163425: every
    > 1 s sample compiled, 12/12 rows). Metric of record (P 10-11): the STEADY cost, the mean over samples after the
    first (their decomposition) that compiled nothing; the rest, beyond that steady cost, is the one-time cost
    (onetime_s). Per L, the 1 sample/s crossing of the steady cost; an L stops after a point past 1 s."""
    rows = []
    for L in a.L:
        pts = []
        for N in a.N:
            r = orca(L, N, a.max_shots, a.budget, a.seed + N, a.budget * a.passes * 2 + 300, a.kill_s, a.passes)
            if r is None or not len(r[1]):
                print(f"=== diag L={L} N={N}: no sample came back", flush=True)
                continue
            ts, pi, nc = r[1], LAST.get("pass_idx", np.zeros(len(r[1]))), LAST.get("new_compiled", np.zeros(len(r[1])))
            for p in sorted(set(pi.tolist())):
                x, c = ts[pi == p], nc[pi == p]
                slow = x > 1.0
                st = (c == 0) & (np.arange(len(x)) > 0)          # steady: compiled nothing, not the first
                steady = float(x[st].mean()) if st.any() else float("nan")
                row = {"L": L, "N": N, "pass": p, "shots": len(x), "mean_s": float(x.mean()),
                       "mean_from3_s": float(x[2:].mean()) if len(x) > 2 else float("nan"),
                       "median_s": float(np.median(x)), "max_s": float(x.max()), "n_slow": int(slow.sum()),
                       "n_slow_compiling": int((slow & (c > 0)).sum()), "n_compiling": int((c > 0).sum()),
                       "mean_noncompiling_s": float(x[c == 0].mean()) if (c == 0).any() else float("nan"),
                       "steady_s": steady, "n_steady": int(st.sum()),
                       "onetime_s": float((x[~st] - (steady if st.any() else 0.0)).sum()),
                       "censored": CENSORED[0] if p == pi.max() else 0}
                rows.append(row)
                if p == 0 and st.sum() >= 8:
                    pts.append((N, steady))
                print("diag " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
                      + f"  [{time.strftime('%H:%M:%S')}]", flush=True)
            if a.samples_csv:
                with open(a.samples_csv, "a") as f:
                    f.writelines(f"{L},{N},{int(p)},{i},{x:.6g},{int(c)}\n"
                                 for i, (x, p, c) in enumerate(zip(ts, pi, nc, strict=True)))
            if a.csv:
                with open(a.csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
            if not pts or pts[-1][0] != N or pts[-1][1] > 1.0:    # past 1 s, or < 8 steady samples in the budget
                break
        print(f"=== diag L={L}: their steady 1 sample/s crossing N* = {crossing(pts):.1f} (nan = not bracketed; "
              f"points {[(n, round(t, 4)) for n, t in pts]})", flush=True)


def ours(a):
    """Our loop samplers at the same (L, N), same job: one loop of delay L, T = 2N bins, N photons in bins
    0..N-1, random settings (bench/timeloop_frontier.py, bench/hybrid_frontier.py): dense LoopRegisterKernel.sample and
    the sector sampler of record (loop_sector_sample), on the H100. Same calls as hybrid_frontier.measure: warm-up
    (sector: 16 shots, builds most tables), then one timed batch of --ours-shots; sector-table build time of the timed
    call subtracted (a one-time, cached host cost, as in K's rows). Dense skipped when one shot does not fit."""
    import torch

    from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit, register_dimension
    from photonic_engines.timeloop.hybrid import light_loop_kernel, loop_sector_sample
    dev = a.device
    rows = []
    for L in a.L:
        pts = []                                          # sector: (N, s/sample) for its 1 sample/s crossing
        for N in a.N:
            if pts and pts[-1][1] > 1.0:
                break
            T = 2 * N
            c = TimeLoopCircuit(T, tuple(ladder_of(L)))
            inp = [1] * N + [0] * (T - N)
            g = torch.Generator(device=dev).manual_seed(a.seed)
            th = torch.rand(c.n_loops, T, generator=g, device=dev, dtype=torch.float64) * 2
            ph = torch.rand(c.n_loops, T, generator=g, device=dev, dtype=torch.float64) * 6
            mem = torch.cuda.get_device_properties(0).total_memory if dev == "cuda" else 8 * 2**30
            fits = 0.4 * mem / (16 * 6)                   # amplitudes, ~6 copies
            sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
            if a.batches:                                 # t_S at S = --ours-shots drawn in batches of b (sector only)
                lk = light_loop_kernel(c, inp, device=dev)
                loop_sector_sample(lk, 16, th, ph, generator=g)
                sync()
                for b in a.batches:
                    gc_collect(dev)
                    try:
                        dt, done = 0.0, 0
                        while done < a.ours_shots:
                            k = min(b, a.ours_shots - done)
                            t = time.perf_counter()
                            _, st = loop_sector_sample(lk, k, th, ph, generator=g, return_stats=True)
                            sync()
                            dt += time.perf_counter() - t - st["table_seconds"]
                            done += k
                    except torch.cuda.OutOfMemoryError:
                        print(f"ours sector L={L} N={N} batch {b}: out of memory", flush=True)
                        continue
                    rows.append({"L": L, "N": N, "T": T, "sampler": "sector", "shots": a.ours_shots, "batch": b,
                                 "t_S": dt, "s_per_sample": dt / a.ours_shots})
                    print(f"ours sector L={L} N={N} T={T} batch {b}: t_S = {dt:.4g} s for S = {a.ours_shots} "
                          f"({dt / a.ours_shots:.4g} s/sample, tables excluded)  [{time.strftime('%H:%M:%S')}]", flush=True)
                    if a.csv:
                        with open(a.csv, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=list(rows[0]))
                            w.writeheader(); w.writerows(rows)
                continue
            for kind in a.kinds.split(","):
                D = register_dimension(N, L)
                if kind == "dense" and fits < D:
                    print(f"ours dense L={L} N={N}: D = {D:,d} does not fit", flush=True)
                    continue
                gc_collect(dev)
                try:
                    if kind == "dense":
                        lk = LoopRegisterKernel(c, inp, device=dev)
                        lk.sample(1, th, ph, generator=g)
                        tab = 0.0
                    else:
                        lk = light_loop_kernel(c, inp, device=dev)
                        loop_sector_sample(lk, 16, th, ph, generator=g)
                    sync()
                    t = time.perf_counter()
                    if kind == "dense":
                        lk.sample(a.ours_shots, th, ph, generator=g)
                    else:
                        _, st = loop_sector_sample(lk, a.ours_shots, th, ph, generator=g, return_stats=True)
                        tab = st["table_seconds"]
                    sync()
                    dt = max(time.perf_counter() - t - tab, 1e-9)
                except torch.cuda.OutOfMemoryError:
                    print(f"ours {kind} L={L} N={N}: out of memory", flush=True)
                    continue
                rows.append({"L": L, "N": N, "T": T, "sampler": kind, "shots": a.ours_shots,
                             "s_per_sample": dt / a.ours_shots, "table_s": tab})
                print(f"ours {kind} L={L} N={N} T={T}: {dt / a.ours_shots:.4g} s/sample ({a.ours_shots} shots, tables "
                      f"{tab:.2f} s excluded)  [{time.strftime('%H:%M:%S')}]", flush=True)
                if kind == "sector":
                    pts.append((N, dt / a.ours_shots))
                if a.csv:
                    with open(a.csv, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader(); w.writerows(rows)
        print(f"=== ours sector L={L}: 1 sample/s crossing N* = {crossing(pts):.1f} (nan = not bracketed)", flush=True)


def gc_collect(dev):
    import gc

    import torch
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "time", "ours", "diag"])
    ap.add_argument("--passes", type=int, default=2, help="diag: runs of --max-shots on the same instance, one process")
    ap.add_argument("--ours-shots", type=int, default=64, help="ours: timed shots per point (one batch)")
    ap.add_argument("--kinds", default="sector,dense", help="ours: which of our loop samplers")
    ap.add_argument("--batches", default="",
                    help="ours: batch sizes; each draws --ours-shots in batches of b and reports t_S (sector only)")
    ap.add_argument("--L", default="1,2,3,4,6,8", help="loop delays (one loop each)")
    ap.add_argument("--N", default="2,3,4", help="photon numbers, ascending (m = T = 2N)")
    ap.add_argument("--shots", type=int, default=5000, help="check: samples per instance")
    ap.add_argument("--max-shots", type=int, default=64, help="time: samples per point at most")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds per point")
    ap.add_argument("--kill-s", type=float, default=0.0,
                    help="time: kill a sample running longer than this and count it at this time (0: never)")
    ap.add_argument("--stop-s", type=float, default=90.0, help="time: skip an N extrapolated past this per sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--samples-csv", default=None, help="time: every sample's seconds (L, N, index, s), appended; "
                    "diag: (L, N, pass, index, s, numba compilations)")
    ap.add_argument("--ladder", default=None, help="one ladder of delays instead of --L, e.g. 1,5,25 (both codes: their "
                                                   "multiple_loops(*ladder), our TimeLoopCircuit(T, ladder))")
    a = ap.parse_args()
    if a.ladder:
        LADDER[:] = [int(x) for x in a.ladder.split(",")]
        a.L = str(sum(LADDER))
    a.L = [int(x) for x in a.L.split(",")]
    a.N = [int(x) for x in a.N.split(",")]
    a.batches = [int(x) for x in a.batches.split(",")] if a.batches else []
    print(f"orca_loop_bench {a.mode}: L={a.L} ladder={LADDER or 'one loop'} N={a.N} seed={a.seed} cpus {os.cpu_count()}",
          flush=True)
    {"check": check, "time": timing, "ours": ours, "diag": diag}[a.mode](a)


if __name__ == "__main__":
    main()
