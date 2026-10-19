"""Native time-loop engine vs the generic path on the unrolled unitary.

    python3 bench/bench_timeloop.py [--cases 8:1,16:1,32:1,16:2,12:4,10:6] [--shots 256] [--device cuda]

Per case N:L (N photons in the first N of T = 2N bins, one loop of delay L, random settings):
  native sample   LoopRegisterKernel.sample, ms/sample at --shots
  native marg     LoopRegisterKernel.bin_marginals (density matrix, D x D), seconds (skipped when D > --max-rho)
  generic sample  fock_sample on the [T+L, T+L] unrolled unitary (Clifford & Clifford), ms/sample
  generic slos    FockSLOSKernel.probs on the unrolled unitary, seconds (skipped when the Fock space > --max-slos)
"""
import argparse
import math
import time

import torch

from photonic_engines.timeloop import LoopRegisterKernel, TimeLoopCircuit

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="8:1,16:1,32:1,64:1,16:2,24:2,12:4,16:4,10:6,12:6")
ap.add_argument("--shots", type=int, default=256)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--max-rho", type=int, default=20000, help="max D for the density-matrix marginals")
ap.add_argument("--max-slos", type=float, default=3e7, help="max Fock-space size for the generic strong simulation")
ap.add_argument("--max-generic-n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
dev = args.device
print(f"{torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}  shots={args.shots}")
print(f"{'N':>3} {'L':>2} {'T':>4} {'D':>10} | {'native ms/sample':>16} {'native marg s':>14} | {'generic ms/sample':>17} "
      f"{'generic slos s':>14}")


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


for case in args.cases.split(","):
    N, L = (int(x) for x in case.split(":"))
    T = 2 * N
    c = TimeLoopCircuit(T, (L,))
    inp = [1] * N + [0] * (T - N)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    th = torch.rand(1, T, generator=g, device=dev, dtype=torch.float64) * 2
    ph = torch.rand(1, T, generator=g, device=dev, dtype=torch.float64) * 6
    lk = LoopRegisterKernel(c, inp, device=dev)
    D = lk.D
    lk.sample(4, th, ph, generator=g)
    sync()
    t0 = time.perf_counter()
    lk.sample(args.shots, th, ph, generator=g)
    sync()
    t_native = (time.perf_counter() - t0) / args.shots * 1e3
    t_marg = float("nan")
    if args.max_rho >= D:
        t0 = time.perf_counter()
        lk.bin_marginals(th, ph)
        sync()
        t_marg = time.perf_counter() - t0
    t_gen = t_slos = float("nan")
    if args.max_generic_n >= N:
        from photonic_engines.clifford import fock_sample
        U = c.unitary(th, ph, device=dev)
        modes = list(range(N))
        if dev == "cuda":                                        # the Clifford sampler needs the GPU
            fock_sample(U, modes, 4, generator=g, algo="auto", backend="cuda")
            sync()
            t0 = time.perf_counter()
            fock_sample(U, modes, args.shots, generator=g, algo="auto", backend="cuda")
            sync()
            t_gen = (time.perf_counter() - t0) / args.shots * 1e3
        if math.comb(c.m + N - 1, N) <= args.max_slos:
            from photonic_engines.slos.kernel import FockSLOSKernel
            k = FockSLOSKernel(N, c.m, inp + [0] * L, device=dev, dtype=torch.complex128)
            k.probs(U)
            sync()
            t0 = time.perf_counter()
            k.probs(U)
            sync()
            t_slos = time.perf_counter() - t0
    print(f"{N:3d} {L:2d} {T:4d} {D:10,d} | {t_native:16.3f} {t_marg:14.3f} | {t_gen:17.3f} {t_slos:14.3f}", flush=True)
