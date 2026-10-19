"""Multi-GPU scaling of the production Fock sampler, MEASURED.

The paper's multi-GPU frontier points are extrapolations, n*(G) = n*(1) + ln G / ln rho_eff
(tested against single-GPU wall times). This measures the throughput definition directly on a host with G >= 2 GPUs:
G processes, one per device, each drawing `--shots` samples of its own Haar instance at (n, m = c n)
with `fock_sample` (backend cuda, the production kernel); the aggregate rate is G x shots over the
wall time of the slowest process, and rate(G) / rate(1) is the number the extrapolation assumes
to be G. The implied frontier shift ln(rate(G) / rate(1)) / ln rho_eff is printed next to ln G / ln rho_eff.

Only the THROUGHPUT definition is measured here (independent samples). The latency definition --
one sample split over G GPUs by lane ranges with one <= 1 kB all-reduce per step -- needs
kernel-side work in clifford/ and is not this script.

On a host with one visible GPU the script prints the device count and exits 0 without measuring,
so a scheduled run is harmless until a second GPU is added.

    python3 bench/multigpu_scaling.py --ns 34,36,38 --ratio 2 --shots 64 --csv results/multigpu_scaling.csv
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

RHO = {1.0: 1.6875, 2.0: 1.808}   # rho(c) of docs/ALGORITHMS.md for the implied-shift column


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def worker(dev, n, m, shots, seed):
    """One device: build, warm up on 2 shots, then time `shots` samples. Prints the seconds."""
    import torch

    from photonic_engines.clifford import fock_sample
    torch.cuda.set_device(dev)
    d = torch.device(f"cuda:{dev}")
    U = torch.tensor(haar(m, seed), dtype=torch.complex128, device=d)
    g = torch.Generator(device=d).manual_seed(seed)
    fock_sample(U, list(range(n)), 2, generator=g, backend="cuda")
    torch.cuda.synchronize(d)
    t0 = time.time()
    occ = fock_sample(U, list(range(n)), shots, generator=g, backend="cuda")
    torch.cuda.synchronize(d)
    dt = time.time() - t0
    assert occ.shape == (shots, m)
    print(f"WORKER {dev} {n} {m} {shots} {dt:.4f}", flush=True)


def run_group(G, n, m, shots, seed):
    """G processes on devices 0..G-1 started together; returns (wall of the slowest, per-device seconds)."""
    procs = []
    t0 = time.time()
    for g in range(G):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker", "0", str(n), str(m), str(shots), str(seed + 1000 * g)],
                                      env=env, stdout=subprocess.PIPE, text=True))
    per = []
    for p in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"worker failed (exit {p.returncode}): {out}")
        per.append(float(out.strip().split()[-1]))
    return time.time() - t0, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", nargs=5, metavar=("DEV", "N", "M", "SHOTS", "SEED"), help=argparse.SUPPRESS)
    ap.add_argument("--ns", default="34,36,38")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    if a.worker:
        dev, n, m, shots, seed = (int(x) for x in a.worker)
        worker(dev, n, m, shots, seed)
        return 0
    import torch
    G = torch.cuda.device_count()
    print(f"visible GPUs: {G}" + "".join(f"\n  {g}: {torch.cuda.get_device_name(g)}" for g in range(G)))
    if G < 2:
        print("one GPU only: nothing to measure (the extrapolation n*(G) = n*(1) + ln G / ln rho_eff stands); exit 0")
        return 0
    rho = RHO.get(a.ratio, 1.808)
    rows = []
    for n in (int(x) for x in a.ns.split(",")):
        m = round(a.ratio * n)
        wall1, per1 = run_group(1, n, m, a.shots, a.seed)
        wallG, perG = run_group(G, n, m, a.shots, a.seed)
        rate1 = a.shots / max(per1)
        rateG = G * a.shots / max(perG)
        gain = rateG / rate1
        print(f"n = {n}, m = {m}: 1 GPU {rate1:.3f}/s ({max(per1):.1f} s)  {G} GPUs {rateG:.3f}/s ({max(perG):.1f} s, per device "
              f"{', '.join(f'{p:.1f}' for p in perG)})  gain {gain:.2f}x (ideal {G})  implied dn* {math.log(gain) / math.log(rho):+.2f} "
              f"(extrapolation {math.log(G) / math.log(rho):+.2f})")
        rows.append((n, m, G, a.shots, rate1, rateG, gain, math.log(gain) / math.log(rho), math.log(G) / math.log(rho)))
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w") as f:
            f.write(f"# bench/multigpu_scaling.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  torch={torch.__version__}  "
                    f"gpus={G}x{torch.cuda.get_device_name(0)}  rho={rho}\n")
            f.write("n,m,G,shots,rate_1,rate_G,gain,dnstar_measured,dnstar_extrapolated\n")
            for r in rows:
                f.write(",".join(f"{x}" for x in r) + "\n")
        print("wrote", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
