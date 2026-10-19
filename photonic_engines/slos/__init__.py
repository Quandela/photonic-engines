"""SLOS_full strong simulation (Heurtel, Mansfield, Senellart, Valiron, arXiv:2206.10549).

  basis      output Fock basis as photon-position layers, masks pruned while building
  basis_torch the same construction on the device (used automatically for CUDA kernels)
  kernel     the amplitude recurrence as batched, device-resident, differentiable torch ops
             (FockSLOSKernel / SuperpositionSLOSKernel / ProductSLOSKernel, the last one
             graded over photon numbers for indefinite-photon-number inputs)
  emitter    bond-carrying recurrence for EmitterMPS inputs (EmitterSLOSKernel), Kraus
             operators, and sampling from a computed distribution
  estimate   additive-error (Gurvits-type) estimator of single amplitudes / probabilities for
             EmitterMPS inputs at polynomial cost (docs/ESTIMATE.md)
  gaussian   GaussianSLOSKernel: squeezed (Gaussian) inputs up to a global photon cutoff, on the same
             layers and edges (one zeta-raise + two raises per squeezer per layer)
  glynn      ONE amplitude of a ProductSuperposition as a single permanent, the constant
             (no-emission) row collapsed - O(2^k (n-k) n) in the DETECTED photon count k
  reference  independent permanent-based amplitudes (correctness oracle)

Design and measurements: docs/SLOS_IMPLEMENTATION.md.
"""
from __future__ import annotations

from .basis import FockLayers, clear_layers_cache, get_layers

try:                                   # the torch kernels; the distributed CPU engine works without torch
    from .emitter import EmitterSLOSKernel, sample_from_probs
    from .estimate import EmitterGurvitsEstimator, Estimate
    from .gaussian import GaussianSLOSKernel
    from .glynn import product_amplitude, product_probability
    from .kernel import FockSLOSKernel, GroupedSLOSKernel, ProductSLOSKernel, SuperpositionSLOSKernel
    from .reference import permanent, product_reference_probs, reference_probs
except (ImportError, OSError) as _e:
    if "torch" not in str(_e) and getattr(_e, "name", None) != "torch":
        raise

__all__ = [
    "EmitterGurvitsEstimator",
    "EmitterSLOSKernel",
    "Estimate",
    "FockLayers",
    "FockSLOSKernel",
    "GaussianSLOSKernel",
    "GroupedSLOSKernel",
    "ProductSLOSKernel",
    "SuperpositionSLOSKernel",
    "clear_layers_cache",
    "get_layers",
    "permanent",
    "product_amplitude",
    "product_probability",
    "product_reference_probs",
    "reference_probs",
    "sample_from_probs",
]
