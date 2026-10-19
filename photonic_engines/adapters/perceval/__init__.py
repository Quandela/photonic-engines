"""Perceval bridge.

converter  `CircuitConverter`: a perceval Circuit -> m x m unitary on device, vectorized
over layers of commuting gates (duck-typed: needs only `circuit.m`, iteration over
(modes, component) and `component.compute_unitary()`).

backends   `SLOSTorchBackend`, `EmitterSLOSBackend` (AStrongSimulationBackend) and
`CliffordGPUBackend` (ASamplingBackend) — imported lazily, they need perceval.

backends   `DistributedSLOSBackend`: the CPU/MPI distributed engine (workers spawned once).

timeloop   `periodic_kernel_from_perceval`: a Processor with TD components -> the native
periodic-network kernel (delay lines as register slots), `last_period_distribution`.
"""
from .converter import CircuitConverter

__all__ = [
    "CircuitConverter",
    "CliffordGPUBackend",
    "DistributedSLOSBackend",
    "DistributedSLOSSampler",
    "EmitterSLOSBackend",
    "SLOSTorchBackend",
    "last_period_distribution",
    "periodic_kernel_from_perceval",
    "periodic_program_from_perceval",
]


def __getattr__(name):
    if name in ("SLOSTorchBackend", "EmitterSLOSBackend", "CliffordGPUBackend", "DistributedSLOSBackend", "DistributedSLOSSampler"):
        from . import backends
        return getattr(backends, name)
    if name in ("periodic_kernel_from_perceval", "periodic_program_from_perceval", "last_period_distribution"):
        from . import timeloop
        return getattr(timeloop, name)
    raise AttributeError(name)
