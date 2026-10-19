"""One external CPU GBS sampler, timed one shot per call, in its own venv.

Run by bench/gbskernels_bench.py --sampler walrus|piquasso with the venv's python ($CPY, built by
scripts/cpu_samplers_setup.sh): their numpy / numba pins never meet the host's torch stack. Reads the instance from
argv[1] (npz: sampler, V, U, n, r, shots, budget, seed), writes the patterns and per-call seconds to argv[2].

  walrus  : The Walrus 0.22.0 `thewalrus.samples.hafnian_sample_state(cov, 1)` (its public wrapper of
            generate_hafnian_sample, Bulmer's chain on numba), every other argument at its default (hbar 2, cutoff 5,
            max_photons 30, parallel False). It draws with np.random: seeded per call.
  piquasso: Piquasso's GaussianSimulator, n Squeezing(r) on modes 0..n-1, Interferometer(U), ParticleNumberMeasurement
            on every mode, default Config except the seed (and measurement_cutoff = N_S with --max-ns);
            execute(program, shots=1) per call. Own venv with numba pinned (scripts/piquasso_setup.sh).
Nothing of their code is copied; both are imported as installed.
"""
import importlib.metadata as md
import os
import sys
import time

import numpy as np


def walrus(d):
    from thewalrus.samples import hafnian_sample_state
    V = d["V"]
    kw = {"cutoff": int(d["cutoff"])} if int(d["cutoff"]) > 0 else {}    # > 0: exact setting, not as shipped
    if int(d["max_photons"]) > 0:
        kw["max_photons"] = int(d["max_photons"])
    once = bool(d["seed_once"])
    if once:
        np.random.seed(int(d["seed"]))       # noqa: NPY002 -- The Walrus draws from the legacy global generator

    def one(seed):
        if not once:
            np.random.seed(seed)             # noqa: NPY002
        return np.asarray(hafnian_sample_state(V, 1, **kw))[0]
    return one, f"thewalrus {md.version('thewalrus')}"


def piquasso(d):
    import piquasso as pq
    n, r, U = int(d["n"]), float(d["r"]), d["U"]
    m = U.shape[0]
    with pq.Program() as prog:
        for i in range(n):
            pq.Q(i) | pq.Squeezing(r)
        pq.Q(*range(m)) | pq.Interferometer(U)
        pq.Q(*range(m)) | pq.ParticleNumberMeasurement()

    # measurement_cutoff caps the photons PER MODE (default 5: biased, like The Walrus' cutoff 5). --walrus-cutoff c > 0
    # sets it to c; else --max-ns sets it to N_S (run 20261011_133536: one shot ran > 40 min at n = 24 with it at 42)
    c = int(d["cutoff"]) if int(d["cutoff"]) > 0 else int(d["max_photons"])
    ck = {"measurement_cutoff": c} if c > 0 else {}

    def one(seed):
        sim = pq.GaussianSimulator(d=m, config=pq.Config(seed_sequence=seed, **ck))
        return np.asarray(sim.execute(prog, shots=1).samples)[0]
    return one, f"piquasso {md.version('piquasso')}"


def main():
    d = np.load(sys.argv[1])
    one, ver = {"walrus": walrus, "piquasso": piquasso}[str(d["sampler"])](d)
    try:
        import numba
        thr = f"numba {numba.__version__} threads {numba.config.NUMBA_NUM_THREADS}"
    except ImportError:
        thr = "no numba"
    seed0, shots, budget = int(d["seed"]), int(d["shots"]), float(d["budget"])
    # warm-up (numba compiles lazily, per code path: a first shot with 0 photons compiles little; run 20261011_123409's
    # n = 12 cost x10 that of n = 10): 16 shots or 20 s, untimed
    def mark(seed):   # start of the shot in flight, for the caller's watchdog; atomic (run 20261011_150033 read a
        with open(sys.argv[2] + ".cur.tmp", "w") as f:      # half-written file and crashed)
            f.write(f"{time.time()} {seed}")
        os.replace(sys.argv[2] + ".cur.tmp", sys.argv[2] + ".cur")
    t_w = time.perf_counter()
    for j in range(16):
        mark(seed0 + 10**6 + j)
        one(seed0 + 10**6 + j)
        if time.perf_counter() - t_w > 20:
            break
    t_w_end = time.perf_counter()
    S, ts, t_all = [], [], time.perf_counter()
    while len(ts) < shots and time.perf_counter() - t_all < budget:
        mark(seed0 + len(ts))
        t = time.perf_counter(); s = one(seed0 + len(ts)); ts.append(time.perf_counter() - t); S.append(s)
        # saved after every shot (atomically): the caller kills a shot past its limit and keeps the ones before it
        np.savez(sys.argv[2] + ".tmp.npz", samples=np.array(S, dtype=np.int64), seconds=np.array(ts))
        os.replace(sys.argv[2] + ".tmp.npz", sys.argv[2] + ".part.npz")
    np.savez(sys.argv[2], samples=np.array(S, dtype=np.int64), seconds=np.array(ts))
    print(f"  worker: {ver}, numpy {np.__version__}, {thr}, cpus {os.cpu_count()}: {len(ts)} shots in "
          f"{sum(ts):.1f} s (warm-up {t_w_end - t_w:.1f} s); per shot median {np.median(ts):.3g} s, max {max(ts):.3g} s "
          f"(N = {int(np.sum(S[int(np.argmax(ts))]))}), max N {int(max(np.sum(S, 1)))}", flush=True)


if __name__ == "__main__":
    main()
