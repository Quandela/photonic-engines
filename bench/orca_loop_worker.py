"""ORCA's loop-progressive-simulator (Novak et al., arXiv:2411.16873; Apache-2.0) sampling one loop instance, in its
own venv. Run by bench/orca_loop_bench.py with $OPY; the clone is $ORCA_SRC.

Reads argv[1] (npz: L, N, m, shots, budget, seed): the circuit is their `bscircuits.multiple_loops(L, modes=m,
thetas=...)` (gates (i, i + L), i = 0..m-L-1) with angles uniform in [0, 2 pi) (their -thr), the input N photons in
bins 0..N-1. Samples come from ONE call of their generator `step_simulator_nosym.progressive_simulation(input, circuit,
number_samples=shots)`, as shipped (its own progressive decomposition once per call, np.random draws); each yield is
timed. Stops at `shots` yields or after `budget` seconds (the generator is then abandoned). Writes argv[2]: samples,
seconds per yield, gates (i, j, theta) as built by their code. Nothing of their code is copied; it is imported.
Optional `ladder` (P 10-12, multi-loop): delays l_1, l_2, ...; the circuit is then `multiple_loops(*ladder, modes=m,
thetas)` (their composition of one loop per delay, sum(m - l) angles).
Optional `passes` (diagnosis): that many generator calls in the same process (seed + pass), with per sample the pass
index and the numba specialisations compiled during it (`n_compiled`); the bookkeeping is outside the timed spans.
"""
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("ORCA_SRC", os.path.expanduser("~/cpusamp/loop-progressive-simulator")),
                                "loop_progressive_simulator"))   # their modules import each other top-level (`from utils`)
from bscircuits import multiple_loops
from step_simulator_nosym import progressive_simulation


def n_compiled():
    """numba specialisations compiled so far in their modules (a sample during which this grows paid a compilation)."""
    root = os.path.realpath(os.environ.get("ORCA_SRC", os.path.expanduser("~/cpusamp/loop-progressive-simulator")))
    tot = 0
    for mod in list(sys.modules.values()):
        if not os.path.realpath(getattr(mod, "__file__", None) or "/").startswith(root):
            continue
        for o in vars(mod).values():
            if hasattr(o, "py_func") and hasattr(o, "signatures"):
                tot += len(o.signatures)
    return tot


def main():
    d = np.load(sys.argv[1])
    L, N, m, shots, budget, seed = (int(d[k]) for k in ("L", "N", "m", "shots", "budget", "seed"))
    partial = bool(d["partial"]) if "partial" in d else False
    passes = int(d["passes"]) if "passes" in d else 1
    ladder = [int(x) for x in d["ladder"]] if "ladder" in d else [L]    # several delays: their multiple_loops(*ladder)
    th = np.random.default_rng(seed).random(sum(max(m - x, 0) for x in ladder)) * 2 * np.pi
    circ = multiple_loops(*ladder, modes=m, thetas=list(th))
    inp = tuple([1] * N + [0] * (m - N))
    gates = np.array([(i, j, th_) for (i, j, th_) in circ], dtype=float)

    def save(path, S, ts):
        # their sample is a key of their sparse Fock state, shorter than m when trailing modes are empty (run
        # 20261011_133431): right-padded with zeros (the exact-law check in orca_loop_bench validates the reading)
        P = [tuple(s) + (0,) * (m - len(s)) for s in S]
        np.savez(path, samples=np.array(P, dtype=np.int64).reshape(len(P), m), seconds=np.array(ts),
                 key_len=np.array([len(s) for s in S]), gates=gates, pass_idx=np.array(pi), new_compiled=np.array(nc))

    def mark():          # wall-clock start of the sample in flight, for the caller's watchdog; atomic (a partial read
        with open(sys.argv[2] + ".cur.tmp", "w") as f:      # crashed the caller in run 20261011_150033)
            f.write(f"{time.time()} {len(ts)}")
        os.replace(sys.argv[2] + ".cur.tmp", sys.argv[2] + ".cur")
    S, ts, pi, nc = [], [], [], []
    t_all = time.perf_counter()
    for p in range(passes):       # passes > 1 (diagnosis): the same instance again in the same process, new draws
        np.random.seed(seed + p)  # noqa: NPY002 -- their sampler draws from the legacy global generator
        c0 = n_compiled()
        t = time.perf_counter()
        mark()
        for s in progressive_simulation(inp, circ, number_samples=shots):
            now = time.perf_counter(); ts.append(now - t); t = now
            c1 = n_compiled(); nc.append(c1 - c0); c0 = c1
            S.append(s); pi.append(p)
            if partial:                            # kept if the caller kills a later sample (atomic replace)
                save(sys.argv[2] + ".tmp.npz", S, ts)
                os.replace(sys.argv[2] + ".tmp.npz", sys.argv[2] + ".part.npz")
            mark()
            t = time.perf_counter()               # the bookkeeping above stays out of the next sample's time
            if now - t_all > budget * (p + 1):
                break
    if any(nc):
        print(f"  orca worker: numba compilations during samples {[(i, k) for i, k in enumerate(nc) if k][:20]}",
              flush=True)
    lens = Counter(len(s) for s in S)
    print(f"  orca worker: sample key lengths {dict(sorted(lens.items()))} (m = {m})", flush=True)
    save(sys.argv[2], S, ts)
    print(f"  orca worker L={L} ladder={ladder} N={N} m={m} gates={len(gates)}: {len(ts)} samples in {sum(ts):.2f} s "
          f"(first {ts[0]:.3g} s, median "
          f"{np.median(ts):.3g} s, max {max(ts):.3g} s); numpy {np.__version__}", flush=True)


if __name__ == "__main__":
    main()
