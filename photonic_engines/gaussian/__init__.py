"""Gaussian boson sampling (docs/GAUSSIAN.md): squeezed (optionally displaced) inputs,
`states.GaussianInput`.

    hafnian   loop hafnians: brute-force oracle, the pair sieve, and the z-polynomial form the
              chain rule needs (`lhaf_poly`)
    sampler   exact PNR chain-rule sampler over conditional pure Gaussian states (`sample`), with
              uniform loss by the pure-plus-classical decomposition; `sample_transfer` for a general
              lossy transfer matrix T (Williamson decomposition of the output covariance)
    probs     exact per-pattern probabilities of a pure input (oracle)

torch + numpy only; the loop hafnian runs on any torch device.
"""
from .hafnian import lhaf, lhaf_brute, lhaf_poly  # not `hafnian`: it would shadow the submodule name
from .probs import pure_probabilities, pure_probability
from .sampler import ShotStats, sample, sample_transfer, transfer_decomposition
