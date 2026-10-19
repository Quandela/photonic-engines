"""SLOS for Gaussian (squeezed) inputs: the output distribution up to a global photon cutoff.

The pure output state is N exp(1/2 a^T B a + zeta^T a)|0> (`states.GaussianInput.bargmann`), with
B = -sum_s tanh(r_s) u_s u_s^T, u_s = U[:, mode_s]. Its coefficients c(t) = [w^t] N exp(1/2 w^T B w +
zeta^T w) - exactly the raw, deferred-normalisation amplitudes of `FockSLOSKernel` (amplitude
c(t) sqrt(prod t!), probability |c(t)|^2 prod t! = `layers.prob_norms`) - obey Euler's identity
sum_i w_i d_i F = (zeta^T w + w^T B w) F, i.e. layer by layer

    N c_N = R(zeta) c_{N-1} - sum_s tanh(r_s) R(u_s) R(u_s) c_{N-2},

R(v) being the SLOS raising pass with per-mode multiplier v_j (the edges `FockLayers` already stores).
So a Gaussian input needs NO new basis and no new edge set: one zeta-raise and, per squeezer, two
u_s-raises per layer (docs/GAUSSIAN.md §7). Raising never lowers the photon number, so layers
0..`max_photons` are EXACT for a lossless input whatever is cut above: the truncation is a global
photon cutoff, not the per-mode cutoff of a Fock-backend simulator, and nothing in the retained
layers is approximate.

Loss: uniform loss thins every retained layer (`imperfections.thin_graded`). A detected layer j then
misses the mass thinned down from the lossless layers above `max_photons`, so under loss the blocks are
lower bounds, short by at most the lossless tail P(N > max_photons) (reported by `tail_mass`).

Cost per layer N: (1 + 2 n_sources) raising passes over |E_{N-1}| edges; memory: three layers.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..imperfections import LossyDistribution
from ..masks import Mask
from .basis import get_layers


class GaussianSLOSKernel:
    """Strong simulation of a `states.GaussianInput` (lossless, optionally displaced) up to
    `max_photons` output photons, on the SLOS layers of `FockSLOSKernel`.

    `probs(U)` returns a `LossyDistribution` with blocks k = 0..max_photons over the k-photon layer;
    lossless, every block is exact and their masses are the input's photon-number law (U-independent)
    up to the cutoff. complex128 by default: the coefficients are small differences of large terms
    at strong squeezing.
    """

    def __init__(self, state, max_photons: int, device: str = "cpu", dtype: torch.dtype = torch.complex128,
                 layers=None):
        self.state = state
        self.m, self.n = state.m, int(max_photons)
        if self.n < 0:
            raise ValueError("max_photons must be >= 0")
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        if layers is None:
            # the torch (on-device) basis builder packs a state into one int64 key, n_photons x ceil(log2 m) bits; past
            # 63 bits (here: many photons in few modes, e.g. N_cut = 16 at m = 10, job 8 of 2026-10-01) build with numpy
            # and move the edges to the device, as `edges_by_mode` is read through torch.as_tensor below anyway
            bits = max(1, math.ceil(math.log2(max(self.m, 2))))
            backend = "numpy" if max(self.n, 1) * bits > 63 else "auto"
            layers = get_layers(max(self.n, 1), self.m, Mask.FOCK.value, device=str(self.device), backend=backend)
        self.layers = layers
        self._sizes = [self.layers.size(k) for k in range(self.n + 1)]
        self._prob_norm = [torch.as_tensor(self.layers.prob_norms(k)).to(dtype=self.rdtype, device=self.device)
                           for k in range(self.n + 1)]
        self._slabs = []
        for k in range(self.n):
            p, c, off = self.layers.edges_by_mode(k)
            self._slabs.append((torch.as_tensor(p).to(self.device).long(), torch.as_tensor(c).to(self.device).long(),
                                [int(v) for v in off]))

    def _raise(self, v: torch.Tensor, src: torch.Tensor, k: int) -> torch.Tensor:
        """R(v): layer-(k-1) coefficients [S_{k-1}] -> layer k [S_k], edge multiplier v[mode]."""
        parent, child, off = self._slabs[k - 1]
        out = torch.zeros(self._sizes[k], dtype=self.cdtype, device=self.device)
        for j in range(self.m):
            a, b = off[j], off[j + 1]
            if b > a:
                out.scatter_add_(0, child[a:b], v[j] * src[parent[a:b]])
        return out

    def coefficients(self, U) -> list[torch.Tensor]:
        """Raw coefficients c_k [S_k], k = 0..max_photons (prob = |c|^2 * layers.prob_norms(k))."""
        B, zeta, logN = self.state.bargmann(U)
        Uc = np.asarray(U.detach().cpu().numpy() if hasattr(U, "detach") else U, dtype=np.complex128)[:, list(self.state.modes)]
        z = torch.as_tensor(zeta, dtype=self.cdtype, device=self.device)
        us = torch.as_tensor(Uc.T.copy(), dtype=self.cdtype, device=self.device)          # [sources, m]
        th = np.tanh(self.state.r)
        c = [torch.full((1,), complex(np.exp(logN)), dtype=self.cdtype, device=self.device)]
        for N in range(1, self.n + 1):
            acc = self._raise(z, c[N - 1], N)
            if N >= 2:
                for s in range(us.shape[0]):
                    if th[s] != 0:
                        acc = acc - th[s] * self._raise(us[s], self._raise(us[s], c[N - 2], N - 1), N)
            c.append(acc / N)
        return c

    def probs(self, U, loss=None) -> LossyDistribution:
        """Output distribution over 0..max_photons photons (blocks exact lossless; see module docstring
        under loss).
        """
        c = self.coefficients(U)
        blocks = []
        for k, a in enumerate(c):
            p = a.real ** 2 + a.imag ** 2
            if k > 1:
                p = p * self._prob_norm[k]
            blocks.append(p)
        if loss is None or float(loss) >= 1.0:
            return LossyDistribution(self.layers, blocks)
        from ..imperfections import thin_graded
        return thin_graded(self.layers, blocks, loss)

    def tail_mass(self) -> float:
        """P(N > max_photons) of the lossless input (undisplaced), the bound on every block under loss."""
        return float(max(0.0, 1.0 - self.state.photon_number_probs(self.n).sum()))

