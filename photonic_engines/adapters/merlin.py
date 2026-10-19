"""MerLin-style trainable layer for time-bin loop devices (ORCA PT-type) on the native register
engine — docs/TIME_LOOP.md §5, tutorial §9b.

MerLin's `QuantumLayer` is: classical features x -> angle-encoded circuit parameters ->
photonic simulation -> a measurement readout, differentiable end to end, batched over x.
`TimeLoopLayer` is the same contract for a device MerLin cannot express as a Perceval circuit
of fixed size: one rail, T time bins, loops of delay l_i with one beam-splitter setting per
loop and per bin. Trainable parameters are the settings (theta, phi) [n_loops, T]; the
features enter as angle offsets on chosen (loop, bin) slots,

    theta[i, t](x) = theta_train[i, t] + x[f]      for (i, t) = encode_slots[f],

and the forward pass returns per input row the readout of `LoopRegisterKernel`:
    "clicks"     [B, T]        p(at least one photon) per bin (threshold detectors), losses allowed
                               (loss = dict(loop=..., detector=...) Kraus channels, or a scalar
                               eta = uniform end-to-end loss by output thinning, docs/LOSS.md)
    "marginals"  [B, T, N+1]   photon-number distribution per bin, losses allowed
    "mean"       [B, T]        mean photon number per bin
    "patterns"   [B, P]        probabilities of chosen output patterns (pure state, no losses)
    "distribution" [B, S]      the full distribution over the S = C(T+N-1, N) patterns (small N only;
                               the register keeps photons still in a loop when flush=False, so the
                               row sums to 1 only with flush=True)

The register engine has no batch axis over settings, so a batch is a Python loop over its rows
(B x one forward; fine for the usual B <= 64 and the natural way to keep the per-row register
small). `sample(x, shots)` draws exact samples for one input row.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from ..timeloop.circuit import TimeLoopCircuit
from ..timeloop.kernel import LoopRegisterKernel


class TimeLoopLayer(torch.nn.Module):
    """Trainable torch Module wrapping `LoopRegisterKernel` with MerLin's QuantumLayer contract.

    Parameters are the per-(loop, bin) beam-splitter settings theta [n_loops, T] and phi (a Parameter when
    `trainable_phi`, else a buffer); classical features are added to theta at `encode_slots`. `forward` returns
    [B, output_size] for the configured readout; the batch is a Python loop over rows.
    """

    def __init__(self, circuit: TimeLoopCircuit, input_occupation: Sequence[int], input_size: int = 0,
                 encode_slots: Sequence[Sequence[int]] | None = None, readout: str = "clicks",
                 loss=None, patterns=None, trainable_phi: bool = True,
                 theta=None, phi=None, flush: bool = True, device: str = "cpu",
                 dtype: torch.dtype = torch.complex128):
        """circuit: TimeLoopCircuit(T, loops). input_occupation: photons per bin (length T).
        input_size: number of classical features; encode_slots[f] = (loop, bin) receiving
        feature f as an angle offset (default: loop 0, bins 0, 1, ... in order).
        flush=True keeps the last L settings at the flush value (loops emptied on the
        detector) whatever the parameters and features say.
        """
        super().__init__()
        self.circuit = circuit
        self.kernel = LoopRegisterKernel(circuit, input_occupation, device=device, dtype=dtype)
        rdtype = self.kernel.rdtype
        n, T = circuit.n_loops, circuit.T
        self.input_size = int(input_size)
        if encode_slots is None:
            encode_slots = [(0, f % T) for f in range(self.input_size)]
        if len(encode_slots) != self.input_size:
            raise ValueError("encode_slots must have input_size entries")
        self.encode_slots = [(int(i), int(t)) for i, t in encode_slots]
        g = torch.Generator().manual_seed(0)
        if theta is None:
            theta = torch.rand(n, T, dtype=rdtype, generator=g) * torch.pi
        if phi is None:
            phi = torch.zeros(n, T, dtype=rdtype)
        self.theta = torch.nn.Parameter(torch.as_tensor(theta, dtype=rdtype, device=device).expand(n, T).clone())
        ph = torch.as_tensor(phi, dtype=rdtype, device=device).expand(n, T).clone()
        if trainable_phi:
            self.phi = torch.nn.Parameter(ph)
        else:
            self.register_buffer("phi", ph)
        self.readout, self.loss, self.patterns, self.flush = readout, loss, patterns, flush
        self._dev = torch.device(device)
        if readout == "distribution":                       # fixed basis: every T-bin pattern with N photons
            from ..slos.basis import get_layers
            self.patterns = [tuple(int(v) for v in o) for o in get_layers(self.kernel.N, T).final_occupations()]

    # ---- sizes ----
    @property
    def output_size(self) -> int:
        """Return the flattened feature width of `forward` for the configured readout.

        T for "clicks"/"mean", T*(N+1) for "marginals", len(self.patterns) for "patterns"/"distribution";
        raises ValueError for any other readout.
        """
        T, N = self.circuit.T, self.kernel.N
        if self.readout in ("clicks", "mean"):
            return T
        if self.readout == "marginals":
            return T * (N + 1)
        if self.readout == "patterns":
            return len(self.patterns)
        if self.readout == "distribution":
            return len(self.patterns)
        raise ValueError(self.readout)

    # ---- settings ----
    def _settings(self, x: torch.Tensor | None):
        theta, phi = self.theta, self.phi
        if x is not None and self.input_size:
            add = torch.zeros_like(theta)
            for f, (i, t) in enumerate(self.encode_slots):
                add = add.index_put((torch.tensor([i], device=add.device), torch.tensor([t], device=add.device)),
                                    x[f].to(theta.dtype).reshape(1), accumulate=True)
            theta = theta + add
        if self.flush:
            theta, phi = self.circuit.flush_settings(theta, phi)
        return theta, phi

    def _one(self, x: torch.Tensor | None, readout: str, patterns, loss):
        theta, phi = self._settings(x)
        if readout == "patterns":
            if loss:
                raise ValueError("pattern probabilities are computed on the pure state (no losses); "
                                 "use marginals/clicks, or kernel.distribution(loss=eta)")
            return self.kernel.probabilities(patterns, theta, phi)
        if readout == "distribution":
            return self.kernel.probabilities(self.patterns, theta, phi)
        marg = self.kernel.bin_marginals(theta, phi, loss=loss)                      # [T, N+1]
        if readout == "marginals":
            return marg.reshape(-1)
        if readout == "clicks":
            return 1.0 - marg[:, 0]
        if readout == "mean":
            return marg @ torch.arange(marg.shape[1], dtype=marg.dtype, device=marg.device)
        raise ValueError(f"unknown readout {readout!r}")

    def forward(self, x: torch.Tensor | None = None, readout: str | None = None, patterns=None,
                loss: dict | None = None) -> torch.Tensor:
        """x: [B, input_size] (or None when input_size == 0) -> [B, output_size] (marginals are
        flattened to [B, T*(N+1)]); with x=None a single row [output_size].
        """
        readout = readout or self.readout
        loss = self.loss if loss is None else loss
        patterns = self.patterns if patterns is None else patterns
        if x is None:
            return self._one(None, readout, patterns, loss)
        x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_size:
            raise ValueError(f"x has {x.shape[1]} features, layer expects {self.input_size}")
        return torch.stack([self._one(x[b], readout, patterns, loss) for b in range(x.shape[0])])

    @torch.no_grad()
    def sample(self, shots: int, x: torch.Tensor | None = None, generator=None, threshold: bool = False,
               loss: dict | None = None) -> torch.Tensor:
        """Exact samples [shots, T] for one input row x (photon counts per bin, or clicks)."""
        theta, phi = self._settings(None if x is None else torch.as_tensor(x).reshape(-1))
        return self.kernel.sample(shots, theta, phi, generator=generator,
                                  loss=self.loss if loss is None else loss, threshold=threshold)

    def extra_repr(self) -> str:
        """Return the torch Module repr line: bin count, loop delays, photon number, register size and IO shape."""
        return (f"T={self.circuit.T}, loops={tuple(self.circuit.loops)}, N={self.kernel.N}, D={self.kernel.D}, "
                f"input_size={self.input_size}, readout={self.readout!r}, output_size={self.output_size}")
