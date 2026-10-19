"""Distributed CPU engine timing. SPMD:  mpirun -np P python3 bench/bench_distributed.py [--n 10 --m 20]
Prints total / build / comm / compute seconds and the per-rank chunk; --ref also times
FockSLOSKernel (CPU, complex128) on rank 0 when the full vector fits."""
import argparse
import time

import numpy as np

from photonic_engines.slos.distributed import DistributedSLOSKernel, _mpi

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--m", type=int, default=20)
ap.add_argument("--batch", type=int, default=1 << 18)
ap.add_argument("--ref", action="store_true")
ap.add_argument("--per-rank", action="store_true", help="print every rank's own build/comm/compute")
args = ap.parse_args()
MPI = _mpi()
comm = MPI.COMM_WORLD
n, m = args.n, args.m
occ = [1] * n + [0] * (m - n)
rng = np.random.default_rng(0)
U = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))[0]
k = DistributedSLOSKernel(n, m, occ, comm=comm, batch=args.batch)
comm.Barrier()
t0 = time.perf_counter()
v = k.amplitudes(U)
comm.Barrier()
dt = time.perf_counter() - t0
nrm = v.norm2()
tim = {key: comm.allreduce(val, op=MPI.MAX) for key, val in k.timings.items()}
halo_gb = comm.allreduce(k.timings["halo_entries"]) * 16 / 2 ** 30
if args.per_rank:
    allt = comm.gather(k.timings, root=0)
    if comm.rank == 0:
        for r, t in enumerate(allt):
            print(f"  rank {r:2d}: build {t['build']:.3f}  comm {t['comm']:.3f}  compute {t['compute']:.3f}  halo "
                  f"{t['halo_entries'] * 16 / 2**30:.2f} GB")
if comm.rank == 0:
    print(f"n={n} m={m} M_n={v.M:,d} P={comm.size}: total {dt:.3f} s  build {tim['build']:.3f}  comm {tim['comm']:.3f}  "
          f"compute {tim['compute']:.3f}  chunk {v.cnt:,d}  threads {tim['threads']}  halo {halo_gb:.2f} GB  norm {nrm:.12f}", flush=True)
    if args.ref:
        import torch

        from photonic_engines.slos.kernel import FockSLOSKernel
        Ut = torch.as_tensor(U)
        rk = FockSLOSKernel(n, m, occ, dtype=torch.complex128)
        t0 = time.perf_counter()
        rk.probs(Ut)
        print(f"FockSLOSKernel cpu complex128 ({torch.get_num_threads()} threads): {time.perf_counter() - t0:.3f} s", flush=True)
