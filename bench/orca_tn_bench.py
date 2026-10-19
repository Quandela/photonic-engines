"""ORCA's cuTensorNet photonic simulator (branch photonics_tn_simulator_lm, 3d66214, built job-local by
scripts/orca_tn_build.sh) on our loop and mesh instances: the law against brute force, then the timing (P, 10-11/12).

The branch runs photonic kernels only from C++ in library mode (run 20261012_021639: the Python bridge refuses
`qudit` on tensornet / tensornet-mps, and orca-photonics in Python is a Python-only handler). So each (target, qudit
level) gets one small C++ program, compiled once with `nvq++ --library-mode --target <t>`. The program reads a gate
list from a file and applies it with THEIR gates: `create`, `beam_splitter(theta, a, b)`, `phase_shift(phi, a)`, then
`mpnr` on every mode. It prints the wall time of each `cudaq::sample(shots, ...)` call and the counts of the last one.

Instances and the reference come from bench/bench_tn_compare.py (`loop_circuit`, `mesh_circuit`, `internal_unitary`
= CUDA-Q's convention read off orca-photonics, `exact_distribution`, `gate`). Their unitary convention on THIS branch
is not assumed: the law check scores U, U^T, conj(U) and conj(U)^T against their samples and reports which one they
follow (`variant`), so a convention slip cannot pass silently.

Defaults are stated, not tuned: the qudit level is n + 1 (any mode can hold every photon, as in their TBI example)
unless --level is given; tensornet = their exact contraction; tensornet-mps = CUDA-Q's MPS defaults (the CUDAQ_MPS_*
environment is left unset and printed). Outcome strings are '0' + k per mode (their sampler), so a mode with more
than 9 photons cannot be read; such a case is flagged.

    python3 bench/orca_tn_bench.py --circuit loop --cases 2:1,3:1,4:1 --targets tensornet,tensornet-mps,orca-photonics
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_tn_compare import build, curve, exact_distribution, gate, internal_unitary, tvd

H = os.environ.get("TN_HOME", os.path.expanduser("~/cpusamp/cudaq"))
SRC = os.environ.get("TN_SRC", os.path.expanduser("~/cpusamp/cudaq-orca-photonics_tn_simulator_lm"))

CPP = r'''
#include <cudaq.h>
#include <cudaq/photonics.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

struct circ {
  void operator()(std::size_t nm, std::vector<std::size_t> inm, std::vector<int> kind,
                  std::vector<std::size_t> a, std::vector<std::size_t> b, std::vector<double> p) __qpu__ {
    cudaq::qvector<LEVEL> q(nm);
    for (std::size_t i = 0; i < inm.size(); i++)
      create(q[inm[i]]);
    for (std::size_t i = 0; i < kind.size(); i++) {
      if (kind[i] == 0)
        beam_splitter(p[i], q[a[i]], q[b[i]]);
      else
        phase_shift(p[i], q[a[i]]);
    }
    mpnr(q);
  }
};

// argv[1]: case file "nm n_in inm... n_ops (kind a b p)..."; argv[2..]: shot counts, the last one's counts printed
int main(int argc, char **argv) {
  std::ifstream f(argv[1]);
  std::size_t nm, nin, nops;
  f >> nm >> nin;
  std::vector<std::size_t> inm(nin);
  for (auto &x : inm) f >> x;
  f >> nops;
  std::vector<int> kind(nops);
  std::vector<std::size_t> a(nops), b(nops);
  std::vector<double> p(nops);
  for (std::size_t i = 0; i < nops; i++) f >> kind[i] >> a[i] >> b[i] >> p[i];
  for (int k = 2; k < argc; k++) {
    std::size_t s = std::strtoull(argv[k], nullptr, 10);
    auto t0 = std::chrono::steady_clock::now();
    auto counts = cudaq::sample(s, circ{}, nm, inm, kind, a, b, p);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("TIME %zu %.9f\n", s, dt);
    std::fflush(stdout);
    if (k == argc - 1)
      for (auto &[key, v] : counts) std::printf("COUNT %s %zu\n", key.c_str(), v);
  }
  return 0;
}
'''


def env():
    cuq = next((os.path.join(r, "..") for r, _, fs in os.walk(os.path.join(H, "wheels")) if "custatevec.h" in fs), "")
    cut = next((os.path.join(r, "..") for r, _, fs in os.walk(os.path.join(H, "wheels")) if "cutensor.h" in fs), "")
    e = dict(os.environ)
    libs = [os.path.join(H, "cuda/lib"), os.path.join(cuq, "lib"), os.path.join(cut, "lib"), os.path.join(H, "openblas/lib"),
            os.path.join(H, "install/lib")]
    e["LD_LIBRARY_PATH"] = ":".join(libs + ([e["LD_LIBRARY_PATH"]] if e.get("LD_LIBRARY_PATH") else []))
    e["PATH"] = os.path.join(H, "install/bin") + ":" + os.path.join(H, "llvm/bin") + ":" + e.get("PATH", "")
    return e


def program(target, level, workdir, e):
    """The compiled C++ program for (target, level), built once; (path, seconds) or (None, reason)."""
    os.makedirs(workdir, exist_ok=True)
    exe = os.path.join(workdir, f"tn_{target}_{level}")
    if os.path.exists(exe):
        return exe, 0.0
    src = os.path.join(workdir, "tn_case.cpp")
    with open(src, "w") as fh:
        fh.write(CPP)
    t0 = time.perf_counter()
    r = subprocess.run(["nvq++", "--library-mode", "--target", target, f"-DLEVEL={level}", src, "-o", exe],
                       capture_output=True, text=True, env=e, timeout=900)
    if r.returncode or not os.path.exists(exe):
        return None, "compile failed: " + (r.stderr or r.stdout).strip().splitlines()[-1][:120]
    return exe, time.perf_counter() - t0


def run(exe, m, inm, ops, shots, workdir, e, timeout):
    case = os.path.join(workdir, "case.txt")
    with open(case, "w") as fh:
        fh.write(f"{m} {len(inm)} {' '.join(map(str, inm))} {len(ops)}\n")
        fh.writelines(f"{k} {a} {b} {p!r}\n" for k, a, b, p in ops)
    try:
        r = subprocess.run([exe, case, *map(str, shots)], capture_output=True, text=True, env=e, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": f"timeout after {timeout:.0f} s"}
    times = {int(s): float(t) for s, t in re.findall(r"^TIME (\d+) ([0-9.eE+-]+)$", r.stdout, re.M)}
    counts = {k: int(v) for k, v in re.findall(r"^COUNT (\S+) (\d+)$", r.stdout, re.M)}
    if r.returncode or not times:
        lines = [ln for ln in (r.stderr + "\n" + r.stdout).splitlines() if ln.strip() and "no version information" not in ln]
        return {"ok": False, "why": f"exit {r.returncode}: " + (lines[-1][:120] if lines else ""), "times": times}
    return {"ok": True, "times": times, "counts": counts}


def law(counts, m, inm, ops):
    """Their counts against the exact law under each convention variant: (best variant, its TVD, n_out, all)."""
    U = internal_unitary(m, ops)
    tot = sum(counts.values())
    emp = {k: v / tot for k, v in counts.items()}
    rev = {k[::-1]: v for k, v in emp.items()}            # their mode order in the outcome string is not assumed either
    scores = {}
    for name, V in (("U", U), ("U^T", U.T), ("conj U", U.conj()), ("U^dag", U.conj().T)):
        ex = exact_distribution(V, inm, m)
        scores[name] = (tvd(emp, ex), len(ex))
        scores[name + ", keys reversed"] = (tvd(rev, ex), len(ex))
    best = min(scores, key=lambda k: scores[k][0])
    return best, scores[best][0], scores[best][1], scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="loop", choices=["loop", "mesh"])
    ap.add_argument("--cases", default="2:1,3:1,4:1", help="loop: N:L (m = 2N + L); mesh: n:m")
    ap.add_argument("--targets", default="tensornet,tensornet-mps,orca-photonics")
    ap.add_argument("--level", type=int, default=0, help="qudit level; 0 = n + 1")
    ap.add_argument("--gate-shots", type=int, default=20000, help="shots for the law check (0: none)")
    ap.add_argument("--max-ref", type=float, default=20000, help="largest outcome space for the law check")
    ap.add_argument("--shot-list", default="1,10,100,1000", help="timed sample() calls, in this order")
    ap.add_argument("--slack", type=float, default=3.0)
    ap.add_argument("--stop-s", type=float, default=300.0,
                    help="a target is not run on later (larger) cases once one sample() call took longer than this")
    ap.add_argument("--timeout", type=float, default=900.0, help="per process")
    ap.add_argument("--separate", action="store_true",
                    help="one process per sample() call, no warm-up call: a timeout then names the call that did not end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default=os.path.join(H, "bench"))
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    e = env()
    r = subprocess.run(["grep", "-rn", "-E", r"CUDAQ_MPS_[A-Z_]+|CUDAQ_TENSORNET_[A-Z_]+",
                        os.path.join(SRC, "runtime/nvqir/cutensornet")], capture_output=True, text=True)
    print("their tensor-network settings read from the environment (defaults in these lines):\n  "
          + "\n  ".join(ln[len(SRC) + 1:][:200] for ln in r.stdout.splitlines()[:16]), flush=True)
    print(f"orca_tn_bench {a.circuit} cases={a.cases} targets={a.targets} level={a.level or 'n+1'} "
          f"CUDAQ_MPS env: { {k: v for k, v in os.environ.items() if k.startswith('CUDAQ_MPS')} or 'unset (CUDA-Q defaults)'}",
          flush=True)
    rows = []
    stopped = {}                                  # target -> why it is not run on larger cases
    for case in a.cases.split(","):
        x, y = (int(v) for v in case.split(":"))
        n, m, inm, ops = build(a.circuit, x, y, a.seed)
        level = a.level or n + 1
        n_out = math.comb(n + m - 1, n)
        for target in a.targets.split(","):
            row = {"circuit": a.circuit, "n": n, "m": m, "param": y, "target": target, "level": level,
                   "compile_s": "", "law_variant": "", "tvd": "", "floor": "", "law_ok": "",
                   "build_ms": "", "per_sample_ms": "", "s_per_sample_at_max_shots": "", "times": "", "note": ""}
            if target in stopped:
                row["note"] = stopped[target]
                rows.append(row)
                print(f"{a.circuit} n={n} m={m} ({y}) {target}: not run ({stopped[target]})", flush=True)
                continue
            exe, ct = program(target, level, os.path.join(a.workdir, f"L{level}"), e)
            row["compile_s"] = ct if exe else ""
            if exe is None:
                row["note"] = ct
                rows.append(row)
                print(f"{a.circuit} n={n} m={m} ({y}) {target}: {ct}", flush=True)
                continue
            shots = [int(s) for s in a.shot_list.split(",")]
            check = a.gate_shots and n_out <= a.max_ref
            calls = [*shots, *([a.gate_shots] if check else [])]
            wd = os.path.join(a.workdir, f"L{level}")
            if a.separate:                        # each call in its own process, timed from inside it (CUDA set-up in)
                r = {"ok": True, "times": {}, "counts": {}}
                for s in calls:
                    q = run(exe, m, inm, ops, [s], wd, e, a.timeout)
                    if not q["ok"]:
                        r = {"ok": False, "why": f"{s}-shot call: {q['why']}", "times": r["times"]}
                        break
                    r["times"].update(q["times"])
                    r["counts"] = q["counts"]
            else:                                 # the first call (1 shot) is a warm-up, not timed in the curve
                r = run(exe, m, inm, ops, [1, *calls], wd, e, a.timeout)
            if not r["ok"]:
                row["note"] = r["why"]
                row["times"] = " ".join(f"{s}:{v:.4g}" for s, v in r.get("times", {}).items())
                rows.append(row)
                stopped[target] = f"failed at n={n}, m={m}"
                print(f"{a.circuit} n={n} m={m} ({y}) {target} level={level}: FAILED {r['why']} "
                      f"(calls done: {row['times'] or 'none'})  [{time.strftime('%H:%M:%S')}]", flush=True)
                continue
            t = {s: r["times"][s] for s in shots if s in r["times"]}
            b_ms, p_ms = curve({str(s): v for s, v in t.items()}, shots)
            smax = max(t)                     # one call of smax shots: its wall time / smax is their cost per sample in a
            row.update(build_ms=b_ms, per_sample_ms=p_ms, s_per_sample_at_max_shots=t[smax] / smax,  # run of smax samples
                       times=" ".join(f"{s}:{v:.4g}" for s, v in t.items()))
            if max(r["times"].values()) > a.stop_s:
                stopped[target] = f"a sample() call took {max(r['times'].values()):.0f} s > --stop-s at n={n}, m={m}"
            if any(len(k) != m for k in r["counts"]):
                row["note"] = "outcome strings not one character per mode (a mode over 9 photons?)"
            if check and r["counts"]:
                v, d, no, sc = law(r["counts"], m, inm, ops)
                floor, bad = gate(d, no, a.gate_shots, a.slack)
                row.update(law_variant=v, tvd=d, floor=floor, law_ok=not bad)
                others = ", ".join(f"{k} {s[0]:.3f}" for k, s in sc.items() if k != v)
                print(f"  law {target}: TVD {d:.4f} (floor {floor:.4f}, {'OK' if not bad else 'FAIL'}) under {v}; "
                      f"other variants {others}", flush=True)
            rows.append(row)
            print(f"{a.circuit} n={n} m={m} ({y}) {target} level={level}: {t[smax] / smax:.4g} s/sample over one "
                  f"{smax}-shot call; build {b_ms:.3g} ms, marginal {p_ms:.4g} ms/sample "
                  f"(sample() wall s: {row['times']}; compile {ct:.1f} s)  [{time.strftime('%H:%M:%S')}]", flush=True)
            if a.csv:
                with open(a.csv, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
