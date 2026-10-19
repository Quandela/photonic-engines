"""CUDA-Q's photonics path against this engine: one machine, one circuit, no assumptions.

What a CUDA-Q user gets for photonics TODAY is the `orca-photonics` target (state vector, CPU,
fp64).  ORCA's cuTensorNet simulator (announced March 2026, A100, "up to 48 qumodes") is NOT in
the released package -- the qudit tensor-network code lives on the `photonic_tn_simulator` branch
of orcacomputing/cuda-quantum and must be built from source.  This benchmark therefore reports
**what the released stack delivers**, names each target precisely, and marks the unreleased branch
as unmeasured.  It never labels a CPU result as "ORCA's GPU simulator".

## What makes the comparison fair

**Same circuit, defined by THEIR gates.**  The circuit is built in CUDA-Q's own vocabulary
(`beam_splitter`, `phase_shift` -- what a PT-series device physically is), so nothing is
decomposed.  `internal_unitary` reproduces CUDA-Q's mode unitary exactly (0.0 to 2.3e-16, verified
against `orca-photonics`; the convention is read off their simulator, not guessed), and the run
re-verifies that once at startup.  Our engine consumes that same unitary.

**Build cost and marginal cost, separately.**  A state-vector or tensor-network simulator pays
once to build the state and then samples nearly free; the Fock chain rule pays per sample.
Quoting either alone makes the two incomparable.  Both are reported,
and the shots-to-crossover derived from them.

**Compilation is never inside a timed region.**  Every (case, target) is warmed up before any
timing, and each point is the MINIMUM over --reps runs, so a stray JIT or allocator spike cannot
inflate a build cost.  (An earlier version fitted a least-squares intercept and reported build
costs that DECREASED with n -- the signature of one-off compilation being absorbed by whichever
case ran first.  Hence both the warm-up pass and the min estimator.)

**Every CUDA-Q target runs in a SUBPROCESS.**  Importing cudaq into a process that later runs
torch/Triton CUDA work crashed OUR warm-up (`mremap_chunk(): invalid pointer`, run 20260923_223900;
see the comment in main).  CUDA-Q's qubit-only targets (`tensornet`, `tensornet-mps`, ...) do not
crash on a qudit kernel: they refuse it in the frontend, which is the correct behaviour and is
reported as a row.  A failure in a third-party simulator must appear as
a reported row, not as a dead job and a blank table.

**No case is ever skipped silently.**  Every (case, target) emits a row, including failures, with
the reason.

**The agreement gate uses the sampling floor.**  TVD against brute force is compared to
sqrt(n_out / (2 pi shots)), not to a fixed constant -- a fixed threshold false-flags any case with
a large outcome space.

    python3 bench/bench_tn_compare.py --circuit loop --cases 2:1,3:1,4:1,5:1,6:1
    python3 bench/bench_tn_compare.py --circuit mesh --cases 2:6,3:6,4:6,2:8,3:8,4:8
    python3 bench/bench_tn_compare.py --circuit mesh --cases 6:12,8:16,10:20 --targets ours
"""
import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

# An op is (kind, a, b, p): kind 0 = beam_splitter(q[a], q[b], p), kind 1 = phase_shift(q[a], p).


def loop_circuit(N, L, seed):
    T = 2 * N
    rng = np.random.default_rng(seed)
    ops = []
    for t in range(T):
        ops.append((1, t, 0, float(rng.uniform(0, 2 * math.pi))))
        ops.append((0, t, T + (t % L), float(rng.uniform(0, math.pi))))
    return T + L, list(range(N)), ops


def mesh_circuit(n, m, seed, depth=None):
    rng = np.random.default_rng(seed)
    ops = []
    for d in range(depth or m):
        for a in range(d % 2, m - 1, 2):
            ops.append((1, a, 0, float(rng.uniform(0, 2 * math.pi))))
            ops.append((0, a, a + 1, float(rng.uniform(0, math.pi))))
    return m, list(range(n)), ops


def build(circuit, x, y, seed):
    if circuit == "loop":
        m, inm, ops = loop_circuit(x, y, seed)
        return x, m, inm, ops
    m, inm, ops = mesh_circuit(x, y, seed)
    return x, m, inm, ops


def internal_unitary(m, ops):
    """CUDA-Q's own convention, read off `orca-photonics` (cudaq 0.16.0) via one-photon states:

        beam_splitter(q_a, q_b, t) -> [[cos t, +sin t], [-sin t, cos t]]   real, full angle, no i
        phase_shift(q_a, p)        -> exp(+i p)

    Reproduces the unitary CUDA-Q itself reports to machine precision; the run re-checks it once
    at startup against `orca-photonics` and flags CONVENTION DRIFT if a release changes it.
    """
    U = np.eye(m, dtype=complex)
    for kind, a, b, p in ops:
        G = np.eye(m, dtype=complex)
        if kind == 0:
            c_, s_ = math.cos(p), math.sin(p)
            G[a, a], G[a, b], G[b, a], G[b, b] = c_, s_, -s_, c_
        else:
            G[a, a] = np.exp(1j * p)
        U = G @ U
    return U


# ---------------------------------------------------------------------- CUDA-Q, in a subprocess
_SRC = '''
import cudaq

@cudaq.kernel
def circ(nm: int, inm: list[int], kind: list[int], a: list[int], b: list[int], p: list[float]):
    q = [qudit(level=__L__) for _ in range(nm)]
    for i in range(len(inm)):
        create(q[inm[i]])
    for i in range(len(kind)):
        if kind[i] == 0:
            beam_splitter(q[a[i]], q[b[i]], p[i])
        else:
            phase_shift(q[a[i]], p[i])

@cudaq.kernel
def circ_m(nm: int, inm: list[int], kind: list[int], a: list[int], b: list[int], p: list[float]):
    q = [qudit(level=__L__) for _ in range(nm)]
    for i in range(len(inm)):
        create(q[inm[i]])
    for i in range(len(kind)):
        if kind[i] == 0:
            beam_splitter(q[a[i]], q[b[i]], p[i])
        else:
            phase_shift(q[a[i]], p[i])
    mz(q)
'''


def _kernels(level, workdir):
    os.makedirs(workdir, exist_ok=True)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    name = f"_tnb{level}"
    path = os.path.join(workdir, name + ".py")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(_SRC.replace("__L__", str(level)))
    mod = __import__(name)
    return mod.circ, mod.circ_m


def worker():
    """One (target, case) in its own process, so a crash in CUDA-Q is caught by the parent.
    Reads a JSON job on argv[2], writes a JSON result on stdout."""
    import warnings
    warnings.filterwarnings("ignore")
    job = json.loads(sys.argv[2])
    import cudaq
    if job["task"] != "targets":
        cudaq.set_target(job["target"])
    m, inm, ops, level, wd = job["m"], job["in_modes"], job["ops"], job["level"], job["workdir"]
    k = [int(o[0]) for o in ops]
    a = [int(o[1]) for o in ops]
    b = [int(o[2]) for o in ops]
    p = [float(o[3]) for o in ops]
    circ, circ_m = _kernels(level, wd)

    if job["task"] == "targets":
        print(json.dumps({"ok": True, "targets": sorted(t.name for t in cudaq.get_targets()),
                          "version": cudaq.__version__.split("(")[0].strip()}))
        return

    if job["task"] == "unitary":
        idx = [2 ** (m - 1 - j) for j in range(m)]
        circ2, _ = _kernels(2, wd)
        U = []
        for j in range(m):
            st = np.array(cudaq.get_state(circ2, m, [j], k, a, b, p))
            U.append([[float(z.real), float(z.imag)] for z in st[idx]])
        print(json.dumps({"ok": True, "U": U}))
        return

    out = {"ok": True, "times": {}, "hist": None, "truncated": False}
    cudaq.sample(circ_m, m, list(inm), k, a, b, p, shots_count=1)     # one warm-up, never timed
    t_start = time.perf_counter()
    for s in job["shots"]:
        if time.perf_counter() - t_start > job["budget"]:
            out["truncated"] = True
            break
        best = None
        for _ in range(job["reps"]):
            t0 = time.perf_counter()
            cudaq.sample(circ_m, m, list(inm), k, a, b, p, shots_count=s)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        out["times"][str(s)] = best
    if not out["times"]:
        out["ok"] = False
        out["why"] = "no shot point completed inside --cq-budget"
    if job.get("gate_shots"):
        res = cudaq.sample(circ_m, m, list(inm), k, a, b, p, shots_count=job["gate_shots"])
        out["hist"] = {"".join(str(c) for c in str(key)): v / job["gate_shots"]
                       for key, v in res.items()}
    print(json.dumps(out))


def call_worker(target, m, in_modes, ops, level, shots, reps, workdir, task="sample",
                gate_shots=0, timeout=600, budget=120.0):
    job = {"target": target, "m": m, "in_modes": list(in_modes), "ops": [list(o) for o in ops],
               "level": level, "shots": list(shots), "reps": reps, "workdir": workdir, "task": task,
               "gate_shots": gate_shots, "budget": budget}
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(job)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": f"timeout after {timeout}s"}
    if r.returncode != 0:
        # keep the most informative line, not the last (often blank or a warning): the simulator's
        # own crash message is what has to reach the report.  CUDA-Q writes some fatal messages to
        # STDOUT rather than stderr, so both are searched.
        skip = ("warning", "futurewarning", "userwarning", "from . import", "see https")
        ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        lines = [ansi.sub("", ln).strip()
                 for ln in ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("{")]
        cand = [ln for ln in lines if not any(w in ln.lower() for w in skip)]
        why = (cand[-1] if cand else (lines[-1] if lines else ""))[:70]
        sig = f"signal {-r.returncode}" if r.returncode < 0 else f"exit {r.returncode}"
        # A qubit-only target REFUSES a qudit kernel in the frontend -- that is correct behaviour,
        # not a crash, and must not be reported as one.  Verified 2026-09-24 by
        # bench/cudaq_qudit_levels_repro.py: qpp-cpu / nvidia / tensornet / tensornet-mps all
        # reject `qudit(level=d>2)` at compile time and never truncate it to a qubit.
        if "qudit(level=" in why or "offending source" in why:
            return {"ok": False, "why": "qudits not supported by this target (refused in the "
                                        "frontend, correctly)"}
        return {"ok": False, "why": f"CRASHED ({sig}) {why}"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "why": "unparseable worker output"}


# ------------------------------------------------------------------- exact reference (tiny n)
def perm(A):
    k = A.shape[0]
    if k == 0:
        return 1.0 + 0j
    tot = 0j
    for msk in range(1, 1 << k):
        cols = [j for j in range(k) if msk >> j & 1]
        tot += (-1) ** len(cols) * np.prod(A[:, cols].sum(1))
    return tot * (-1) ** k


def exact_distribution(U, in_modes, m):
    n = len(in_modes)
    out = {}
    for combo in itertools.combinations_with_replacement(range(m), n):
        t = [0] * m
        for c in combo:
            t[c] += 1
        rows = [r for r in range(m) for _ in range(t[r])]
        out["".join(map(str, t))] = abs(perm(U[np.ix_(rows, list(in_modes))])) ** 2 / math.prod(
            math.factorial(v) for v in t)
    return out


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def gate(d, n_out, shots, slack):
    """Compare a TVD against its SAMPLING FLOOR, not a fixed constant: with n_out outcomes and
    `shots` draws the expected TVD of a correct sampler is ~sqrt(n_out / (2 pi shots)).  A fixed
    threshold false-flags every case with a large outcome space."""
    floor = math.sqrt(n_out / (2 * math.pi * max(shots, 1)))
    return floor, (not math.isnan(d)) and d > slack * max(floor, 1e-3)


def curve(times, shots):
    """(build_ms, marginal_ms) from {shots: seconds}, each a MIN over reps after a warm-up.  The
    slope comes from the two largest shot counts and the intercept by subtracting it at the
    smallest -- robust to a spike at any single point, unlike a least-squares fit."""
    xs = sorted(int(s) for s in times)
    if not xs:
        return float("nan"), float("nan")
    t = {int(s): (times[str(s)] if str(s) in times else times[s]) for s in xs}
    if len(xs) == 1:
        return t[xs[0]] * 1e3, t[xs[0]] / xs[0] * 1e3
    hi, mid = xs[-1], xs[-2]
    slope = max((t[hi] - t[mid]) / (hi - mid), 0.0)
    build = max(t[xs[0]] - slope * xs[0], 0.0)
    return build * 1e3, slope * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="loop", choices=["loop", "mesh"])
    ap.add_argument("--cases", default="")
    ap.add_argument("--targets", default="ours,orca-photonics,tensornet,tensornet-mps")
    ap.add_argument("--shot-list", default="1,4,16,64,256")
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per point; the MIN is kept")
    ap.add_argument("--gate-shots", type=int, default=8000)
    ap.add_argument("--slack", type=float, default=3.0, help="TVD must stay below slack x floor")
    ap.add_argument("--max-ref", type=float, default=20000, help="max outcomes for the brute-force gate")
    ap.add_argument("--max-sv", type=float, default=4e7, help="skip a CUDA-Q cell above level^m")
    ap.add_argument("--timeout", type=float, default=900.0, help="hard kill for one CUDA-Q cell")
    ap.add_argument("--cq-budget", type=float, default=180.0,
                    help="seconds of timing inside one CUDA-Q cell; it stops adding shot points "
                         "past this and reports what it has (a state-vector target rebuilds the "
                         "whole state per call, so points are expensive)")
    ap.add_argument("--cq-shots", default="",
                    help="shot points for the CUDA-Q targets (default: first and last of "
                         "--shot-list; build dominates and the marginal is ~0, so two suffice)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default="/tmp/tnbench")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    cases = a.cases or ("2:1,3:1,4:1,5:1,6:1" if a.circuit == "loop" else "2:6,3:6,4:6,2:8,3:8")
    case_list = [tuple(int(v) for v in c.split(":")) for c in cases.split(",")]
    shots = [int(s) for s in a.shot_list.split(",")]
    cq_shots = [int(s) for s in a.cq_shots.split(",")] if a.cq_shots else [shots[0], shots[-1]]
    targets = [t for t in a.targets.split(",") if t]
    cq = [t for t in targets if t != "ours"]

    # NEVER import cudaq in this process.  Importing cudaq into a process that subsequently runs
    # torch/Triton CUDA work aborts the process with `mremap_chunk(): invalid pointer` -- observed
    # on an H100 host in run 20260923_223900, where the crash landed inside OUR warm-up, not inside
    # any CUDA-Q simulator.  The ours-only jobs of the same run, which never imported cudaq,
    # completed cleanly.  So every cudaq touch, including the target listing, goes to a subprocess.
    have_cq = False
    if cq:
        r = call_worker("", 0, [], [], 2, [], 0, a.workdir, task="targets", timeout=180.0)
        if not r.get("ok"):
            print(f"!! cudaq unusable ({r.get('why')}) -- 'ours' only")
            cq = []
        else:
            print(f"cudaq {r.get('version', '?')}  (all CUDA-Q work runs in subprocesses)")
            avail = set(r["targets"])
            for t in list(cq):
                if t not in avail:
                    print(f"  {t}: NOT IN THIS BUILD (qudit tensornet needs "
                          f"orcacomputing/cuda-quantum@photonic_tn_simulator) -- skipped")
                    cq.remove(t)
            have_cq = bool(cq)

    if have_cq and "orca-photonics" in cq:
        _, m0, inm0, ops0 = build(a.circuit, *case_list[0], a.seed)
        r = call_worker("orca-photonics", m0, inm0, ops0, 2, [], 0, a.workdir, task="unitary",
                        timeout=min(a.timeout, 120.0))
        if r.get("ok"):
            # the worker returns one row per input mode j; transposed, they are the columns of U
            Uc = np.array([[complex(z[0], z[1]) for z in row] for row in r["U"]]).T
            d = np.abs(Uc - internal_unitary(m0, ops0)).max()
            print(f"  convention check vs orca-photonics: max|U_cudaq - U_internal| = {d:.2e}"
                  + ("" if d < 1e-9 else "   *** CONVENTION DRIFT -- inspect before trusting a row ***"))
        else:
            print(f"  convention check: unavailable ({r.get('why')})")

    ours_ok = False
    if "ours" in targets:
        try:
            import torch

            from photonic_engines.clifford import fock_sample
            ours_ok = True
            print("  warming up (compilation kept out of every timed region)...", end="", flush=True)
            for x, y in case_list:
                n, m, inm, ops = build(a.circuit, x, y, a.seed)
                Ut = torch.tensor(internal_unitary(m, ops), device=a.device, dtype=torch.complex128)
                g = torch.Generator(device=a.device).manual_seed(a.seed)
                for s in (1, max(shots)):
                    fock_sample(Ut, inm, s, generator=g, algo="auto", backend="cuda")
                if a.device == "cuda":
                    torch.cuda.synchronize()
            print(" done")
        except Exception as e:
            print(f"  ours unavailable: {type(e).__name__}: {str(e)[:80]}")

    print(f"\ncircuit={a.circuit}  shots={shots}  cq_shots={cq_shots}  reps={a.reps} (min kept)\n")
    print(f"{'case':>7} {'n':>3} {'m':>4} {'fock':>12} {'lvl^m':>11} {'target':>16} "
          f"{'build ms':>10} {'per-sample ms':>14} {'TVD':>7} {'floor':>7}  note")
    rows = []
    for x, y in case_list:
        case = f"{x}:{y}"
        n, m, inm, ops = build(a.circuit, x, y, a.seed)
        dim = math.comb(m + n - 1, n)
        sv = (n + 1) ** m
        svs = f"{sv:,}" if sv < 10 ** 11 else f"{float(sv):.2e}"
        U = internal_unitary(m, ops)
        ref = exact_distribution(U, inm, m) if dim <= a.max_ref else None

        def emit(target, b_ms, p_ms, d, floor, note, bad=False, level=""):
            bs = b_ms if isinstance(b_ms, str) else f"{b_ms:.4g}"
            ps = p_ms if isinstance(p_ms, str) else f"{p_ms:.4g}"
            ds = "-" if d is None or (isinstance(d, float) and math.isnan(d)) else f"{d:.4f}"
            fs = "-" if floor is None else f"{floor:.4f}"
            print(f"{case:>7} {n:>3} {m:>4} {dim:>12,} {svs:>11} {target:>16} "
                  f"{bs:>10} {ps:>14} {ds:>7} {fs:>7}  {note}"
                  + ("   *** MISMATCH ***" if bad else ""))
            rows.append({"circuit": a.circuit, "case": case, "n": n, "m": m, "fock_dim": dim, "sv_dim": sv,
                             "target": target, "level": level,
                             "build_ms": ("" if bad or isinstance(b_ms, str) else b_ms),
                             "per_sample_ms": ("" if bad or isinstance(p_ms, str) else p_ms),
                             "tvd": ("" if d is None else d),
                             "tvd_floor": ("" if floor is None else floor), "note": note})

        if ours_ok:
            try:
                import torch

                from photonic_engines.clifford import fock_sample
                Ut = torch.tensor(U, device=a.device, dtype=torch.complex128)
                g = torch.Generator(device=a.device).manual_seed(a.seed)

                def once(s):
                    t0 = time.perf_counter()
                    fock_sample(Ut, inm, s, generator=g, algo="auto", backend="cuda")
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    return time.perf_counter() - t0

                times = {str(s): min(once(s) for _ in range(a.reps)) for s in shots}
                b_ms, p_ms = curve(times, shots)
                d, floor, bad = float("nan"), None, False
                if ref is not None:
                    occ = fock_sample(Ut, inm, a.gate_shots, generator=g, algo="auto", backend="cuda")
                    h = {}
                    for row in occ.cpu().numpy():
                        key = "".join(str(int(v)) for v in row)
                        h[key] = h.get(key, 0) + 1
                    d = tvd(ref, {key: v / a.gate_shots for key, v in h.items()})
                    floor, bad = gate(d, dim, a.gate_shots, a.slack)
                emit("ours", b_ms, p_ms, d, floor, "chain rule, exact", bad)
            except Exception as e:
                emit("ours", "FAILED", "-", None, None, f"{type(e).__name__}: {str(e)[:60]}")

        for t in cq:
            level = n + 1
            if sv > a.max_sv:
                emit(t, "skipped", "-", None, None, f"level^m = {svs} > --max-sv", level=level)
                continue
            r = call_worker(t, m, inm, ops, level, cq_shots, a.reps, a.workdir,
                            gate_shots=(a.gate_shots if ref is not None else 0),
                            timeout=a.timeout, budget=a.cq_budget)
            if not r.get("ok"):
                emit(t, "FAILED", "-", None, None, r.get("why", "unknown"), level=level)
                continue
            b_ms, p_ms = curve(r["times"], cq_shots)
            trunc = "  [budget-truncated]" if r.get("truncated") else ""
            d, floor, bad = float("nan"), None, False
            if ref is not None and r.get("hist"):
                d = tvd(ref, r["hist"])
                floor, bad = gate(d, dim, a.gate_shots, a.slack)
            emit(t, b_ms, p_ms, d, floor, f"cutoff {level} >= n+1: exact{trunc}", bad, level=level)

    by = {}
    for r in rows:
        by.setdefault(r["case"], {})[r["target"]] = r
    print(f"\n{'case':>7} {'target':>16} {'S (shots)':>11}   below S the chain rule is cheaper")
    for case, d in by.items():
        o = d.get("ours")
        if not o or o["build_ms"] == "":
            continue
        for t, r in d.items():
            if t == "ours" or r["build_ms"] == "":
                continue
            dm = float(o["per_sample_ms"]) - float(r["per_sample_ms"])
            df = float(r["build_ms"]) - float(o["build_ms"])
            s_ = ("never" if df >= 0 else "always") if dm <= 0 else f"{max(df, 0.0) / dm:.0f}"
            print(f"{case:>7} {t:>16} {s_:>11}   ours {float(o['per_sample_ms']):.4g} ms/sample; "
                  f"{t} {float(r['build_ms']):.4g} ms build + {float(r['per_sample_ms']):.4g} ms/sample")

    if a.csv and rows:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.csv} ({len(rows)} rows)")
    print("\nTargets are named as they are: `orca-photonics` is CUDA-Q's RELEASED photonics target\n"
          "(state vector, CPU, fp64) -- what a CUDA-Q user gets today.  ORCA's cuTensorNet\n"
          "simulator is on an unreleased branch and is NOT measured here.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker()
    else:
        main()
