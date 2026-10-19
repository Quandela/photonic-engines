"""torch.nn.Module wrapper of the native time-loop engine — the hook for MerLin-style training:
the beam-splitter settings are the parameters, the forward pass returns a differentiable
readout (per-bin photon-number or click probabilities, or the probabilities of chosen output
patterns).

    layer = TimeLoopModule(circuit, input_occupation, readout="clicks", loss=dict(loop=0.95))
    y = layer()                       # [T]      p(click) per bin  (threshold detectors)
    y = layer(readout="marginals")    # [T, N+1] photon-number distribution per bin
    y = layer(readout="patterns", patterns=[[1, 0, 1, ...], ...])   # [P]
    y.sum().backward()                # gradients w.r.t. layer.theta, layer.phi

`theta`/`phi` are [n_loops, T]; pass `trainable_phi=False` for a PT-series device without
programmable phases. Batched inputs (data re-uploading) are done by the caller: build one
module per input pattern or wrap in torch.func.vmap over the parameters.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .circuit import TimeLoopCircuit
from .kernel import LoopRegisterKernel


class TimeLoopModule(torch.nn.Module):
    """`torch.nn.Module` wrapping a `LoopRegisterKernel`, with the beam-splitter settings as parameters.

    `theta` is always a [n_loops, T] parameter; `phi` is a parameter when `trainable_phi`, otherwise a buffer. The
    forward pass returns the configured readout (clicks, marginals, mean or pattern probabilities), differentiable in
    both. Construction builds the kernel, so the basis is materialized once at init on `device`.
    """

    def __init__(self, circuit: TimeLoopCircuit, input_occupation: Sequence[int],
                 theta=None, phi=None, trainable_phi: bool = True, readout: str = "clicks",
                 loss=None, patterns=None, device: str = "cpu",
                 dtype: torch.dtype = torch.complex128):
        super().__init__()
        self.kernel = LoopRegisterKernel(circuit, input_occupation, device=device, dtype=dtype)
        rdtype = self.kernel.rdtype
        n, T = circuit.n_loops, circuit.T
        if theta is None:
            theta = torch.rand(n, T, dtype=rdtype) * torch.pi
        if phi is None:
            phi = torch.zeros(n, T, dtype=rdtype)
        self.theta = torch.nn.Parameter(torch.as_tensor(theta, dtype=rdtype, device=device).expand(n, T).clone())
        ph = torch.as_tensor(phi, dtype=rdtype, device=device).expand(n, T).clone()
        if trainable_phi:
            self.phi = torch.nn.Parameter(ph)
        else:
            self.register_buffer("phi", ph)
        self.readout, self.loss, self.patterns = readout, loss, patterns

    def forward(self, readout: str | None = None, patterns=None, loss: dict | None = None):
        """Run the loop and return the readout, overriding the configured `readout`/`patterns`/`loss` when given.

        Shapes: 'clicks' -> [T], 'marginals' -> [T, N+1], 'mean' -> [T], 'patterns' -> [P]. Pattern probabilities are
        computed on the pure state, so combining them with `loss` raises.
        """
        readout = readout or self.readout
        loss = self.loss if loss is None else loss
        if readout == "patterns":
            pats = self.patterns if patterns is None else patterns
            if loss:
                raise ValueError("pattern probabilities are computed on the pure state (no losses); "
                                 "use marginals/clicks, or kernel.distribution(loss=eta)")
            return self.kernel.probabilities(pats, self.theta, self.phi)
        marg = self.kernel.bin_marginals(self.theta, self.phi, loss=loss)          # [T, N+1]
        if readout == "marginals":
            return marg
        if readout == "clicks":
            return 1.0 - marg[:, 0]
        if readout == "mean":
            n = torch.arange(marg.shape[1], dtype=marg.dtype, device=marg.device)
            return marg @ n
        raise ValueError("readout must be 'clicks', 'marginals', 'mean' or 'patterns'")

    def sample(self, shots: int, generator=None, threshold: bool = False):
        """Draw `shots` samples at the current parameters, using the module's configured `loss`.

        Parameters are detached, so this does not build a graph.
        """
        return self.kernel.sample(shots, self.theta.detach(), self.phi.detach(), generator=generator,
                                  loss=self.loss, threshold=threshold)
