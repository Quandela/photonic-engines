"""Time-bin loop devices (ORCA PT-series): programmable loops on one rail.

    TimeLoopCircuit      loops + programmable beam splitters; .unitary(theta, phi) unrolls to the
                         [T+L, T+L] matrix (differentiable) for the generic engines
    LoopRegisterKernel   native engine: register marched through time (amplitudes, per-bin
                         marginals with losses, exact trajectory sampler), polynomial in N for
                         fixed total loop length L
    PeriodicNetworkKernel  general periodic network with delay lines (Perceval TD circuits,
                         multi-rail machines): per-period program of two-mode gates and losses
    TimeLoopModule       torch.nn.Module: trainable settings, differentiable readout (MerLin hook)
    register_dimension   C(N+L+n_rail, L+n_rail)
See docs/TIME_LOOP.md.
"""
from .basis import RegisterBasis
from .circuit import TimeLoopCircuit
from .kernel import LoopRegisterKernel, PeriodicNetworkKernel, fock_blocks, register_dimension
from .module import TimeLoopModule

__all__ = ["LoopRegisterKernel", "PeriodicNetworkKernel", "RegisterBasis", "TimeLoopCircuit", "TimeLoopModule",
           "fock_blocks", "register_dimension"]
