"""Worker processes of PooledSLOSKernel.

    python -m photonic_engines.slos._mpi_worker --connect HOST:PORT   launched by `mpirun -np P`
        from the pool: rank 0 connects to the pool's socket, reads pickled commands from it and
        writes the replies back, broadcasting each command to the other ranks over MPI.
    python -m photonic_engines.slos._mpi_worker --spawned   started by MPI_Comm_spawn: commands
        arrive by bcast over the parent intercommunicator, rank 0 replies with send.

Every rank runs DistributedSLOSKernel on the workers' COMM_WORLD; every command is collective.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

from photonic_engines.slos.distributed import DistributedSLOSKernel, _mpi, _recv_msg, _send_msg

MPI = _mpi()


def _handle(msg, comm, state):
    cmd = msg["cmd"]
    kernel, vec = state.get("kernel"), state.get("vec")
    if cmd == "build":
        state["kernel"] = DistributedSLOSKernel(msg["n"], msg["m"], msg["occ"], comm=comm, batch=msg.get("batch", 1 << 16))
        return {"rank": comm.rank, "size": comm.size}
    if cmd == "run":
        state["vec"] = vec = kernel.amplitudes(msg["U"])
        return {"norm2": vec.norm2(), "timings": kernel.timings}
    if cmd == "gather":
        return vec.gather(order=msg["order"], max_bytes=msg["max_bytes"])
    if cmd == "get":
        pats = msg["patterns"]
        return vec.amplitudes_of(pats if comm.rank == 0 else np.zeros((0, kernel.m), dtype=np.int64))
    if cmd == "marginal":
        return vec.marginal(msg["mode"])
    if cmd == "mean":
        return vec.mean_photon_numbers()
    if cmd == "top":
        return vec.top(msg["k"])
    if cmd == "sample":
        return vec.sample(msg["shots"], seed=msg["seed"])
    raise ValueError(f"unknown command {cmd!r}")


def main():
    comm = MPI.COMM_WORLD
    spawned = "--spawned" in sys.argv
    parent = MPI.Comm.Get_parent() if spawned else None
    rd = wr = None
    if not spawned and comm.rank == 0:
        import socket
        host, port = sys.argv[sys.argv.index("--connect") + 1].rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout=60)
        sock.settimeout(None)
        rd, wr = sock.makefile("rb"), sock.makefile("wb")
    state = {}
    while True:
        if spawned:
            msg = parent.bcast(None, root=0)
        else:
            msg = _recv_msg(rd) if comm.rank == 0 else None
            msg = comm.bcast(msg, root=0)
        if msg["cmd"] == "stop":
            break
        try:
            reply = _handle(msg, comm, state)
        except Exception:
            reply = {"__error__": traceback.format_exc()}
        if comm.rank == 0:
            if spawned:
                parent.send(reply, dest=0, tag=0)
            else:
                _send_msg(wr, reply)
    if spawned:
        if comm.rank == 0:
            parent.send(None, dest=0, tag=0)
        parent.Disconnect()


if __name__ == "__main__":
    main()
