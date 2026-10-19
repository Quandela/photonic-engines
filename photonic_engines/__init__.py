"""photonic-engines: strong-simulation and sampling engines for linear optics.

    photonic_engines.states    input-state types (FockState, FockSuperposition,
                               ProductSuperposition, EmitterMPS)
    photonic_engines.masks     output computation-space restrictions
    photonic_engines.imperfections  uniform loss and partial distinguishability, shared by every engine
    photonic_engines.slos      SLOS_full strong simulation (torch, batched, differentiable)
    photonic_engines.clifford  Clifford & Clifford exact boson sampler (Triton / CUDA kernels)
    photonic_engines.timeloop  time-bin loop devices (ORCA-type): unrolled unitary + native register engine
    photonic_engines.sequence  k emitters x n time bins through a static interferometer: the k·n-photon
                               sequence as a chain of per-bin Kraus operators on the product spin register
    photonic_engines.adaptive  adaptive linear optics on a Fock input: mid-circuit photon counting with feed-forward
                               on the interferometer (strong, exact mode marginals, exact sampler)
    photonic_engines.nonlinear non-linear boson sampling: interferometers interleaved with single-mode photon-number
                               phases (Kerr); strong by branches or gates, exact sampler by rejection
    photonic_engines.gaussian  Gaussian boson sampling: squeezed inputs, loop hafnians, exact PNR chain-rule sampler
    photonic_engines.adapters  thin bridges to frontends (perceval, ...)

The core depends on torch and numpy only. `photonic_engines.clifford` needs triton (bundled
with CUDA builds of torch) and, for its fastest path, cupy; it is imported lazily.
"""
from __future__ import annotations

from importlib import import_module

from .masks import Mask, graded_state_space_size, state_space_size
from .states import EmitterMPS, FockState, FockSuperposition, GaussianInput, ProductSuperposition, spin_gate

__version__ = "0.0.1"

__all__ = [
    "EmitterMPS",
    "FockState",
    "FockSuperposition",
    "GaussianInput",
    "Mask",
    "ProductSuperposition",
    "adapters",
    "adaptive",
    "clifford",
    "gaussian",
    "graded_state_space_size",
    "nonlinear",
    "sequence",
    "slos",
    "spin_gate",
    "state_space_size",
    "timeloop",
]


def __getattr__(name):
    if name in ("slos", "clifford", "gaussian", "timeloop", "sequence", "adaptive", "nonlinear", "adapters"):
        return import_module(f".{name}", __name__)
    raise AttributeError(name)
