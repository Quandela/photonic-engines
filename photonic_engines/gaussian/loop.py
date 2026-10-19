"""GBS through a time-bin loop register with a photon cap (docs/GAUSSIAN.md §2.5; bench/gbs_loop.py).

A squeezed train has no finite Fock register: the register (the rail wire plus the L loop slots) is capped at p photons
and every source is truncated at p. Two pieces:

* the leak model (exact Gaussian algebra, CPU): without detection the register is Gaussian; its moments N = <a†a>,
  M = <aa> are propagated bin by bin (fresh rail wire, squeezer, the loop gates, the rail traced out). The truncated
  sampler drops, at the injection of bin t, the components with slots j <= p (it never holds more) and squeezer
  q > p - j; to first order in the dropped mass that is sum_j P_slots,t(j) P_sq(q > p - j) with the exact slot law, from
  the generating function G(z) = det((1 - z) sigma + (1 + z) / 2)^(-1/2) (sigma the xp covariance, vacuum 1/2) expanded
  through the eigenvalues of (sigma + 1/2)^(-1) (1/2 - sigma). leak(p) = sum_t of it; `cap` = the smallest p with
  leak(p) <= eps. (Summing P(n_reg(t) > p) instead counts a component again at every bin it stays above p: x1.5-2.5 too
  high.)
* the sampler: `SuperpositionLoopKernel` (timeloop/hybrid.py) with its register basis capped at p through
  the `_NoBasis` class attribute (the pattern of `light_loop_kernel`), the squeezer amplitudes set after construction,
  and the surviving squared norm recorded at every injection (`sample_weighted`: the leak per shot).
"""
from __future__ import annotations

import math

import numpy as np

R_HEAD = math.asinh(math.sqrt(0.5))                    # 0.5 photons per squeezer (docs/GAUSSIAN.md §2)


def smsv_amps(r: float, d: int) -> np.ndarray:
    """Fock amplitudes c_q, q < d, of the squeezed vacuum sech(r)^(1/2) sum_k (-tanh r)^k sqrt((2k)!) / (2^k k!) |2k>."""
    c = np.zeros(d, dtype=np.complex128)
    t = -math.tanh(r)
    for k in range((d - 1) // 2 + 1):
        c[2 * k] = math.exp(0.5 * math.lgamma(2 * k + 1) - k * math.log(2) - math.lgamma(k + 1)) * t ** k
    return c / math.sqrt(math.cosh(r))


def _sigma(Nm: np.ndarray, Mm: np.ndarray) -> np.ndarray:
    """The xxpp covariance (vacuum 1/2) from N_ij = <a_i† a_j>, M_ij = <a_i a_j>."""
    k = Nm.shape[0]
    I = np.eye(k)
    xx = Mm.real + Nm.real + I / 2
    pp = -Mm.real + Nm.real + I / 2
    xp = Mm.imag + Nm.imag
    return np.block([[xx, xp], [xp.T, pp]])


def number_law(Nm: np.ndarray, Mm: np.ndarray, pmax: int) -> np.ndarray:
    """P(n = k), k = 0..pmax, of the total photon number of a zero-mean Gaussian state given by (N, M)."""
    s = _sigma(Nm, Mm)
    I = np.eye(s.shape[0])
    A, B = s + I / 2, I / 2 - s
    lam = np.linalg.eigvals(np.linalg.solve(A, B)).real
    # log G = -1/2 log det A - 1/2 sum_j log(1 + z lam_j) = g_0 + sum_k g_k z^k
    k = np.arange(1, pmax + 1)
    pk = (lam[None, :] ** k[:, None]).sum(1)
    g = np.zeros(pmax + 1)
    g[1:] = -0.5 * (-1.0) ** (k + 1) * pk / k
    f = np.zeros(pmax + 1)
    f[0] = 1.0 / math.sqrt(abs(np.linalg.det(A)))
    for n in range(1, pmax + 1):                               # f = exp(g): n f_n = sum_k k g_k f_{n-k}
        f[n] = (np.arange(1, n + 1) * g[1:n + 1] * f[n - 1::-1][:n]).sum() / n
    return f


def settings(n_loops: int, T: int, seed: int, device: str = "cpu"):
    """Settings theta in [0, 2), phi in [0, 6) per loop and bin, drawn as bench/hybrid_frontier.py draws them."""
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    th = torch.rand(n_loops, T, generator=g, device=device, dtype=torch.float64) * 2
    ph = torch.rand(n_loops, T, generator=g, device=device, dtype=torch.float64) * 6
    return th, ph, g


def register_tails(ladder, T: int, n_src: int, r: float, theta: np.ndarray, phi: np.ndarray, pmax: int):
    """Per bin t: the mass the cap-p register drops at the injection of bin t (first order; p = 0..pmax), and the mean
    register photon number right after it. Squeezers in bins 0..n_src-1, all with the same real r (the sampler's
    phase; a common phase is a global rotation and changes no photon statistics).
    """
    ladder = tuple(int(x) for x in ladder)
    L = sum(ladder)
    off = np.concatenate([[0], np.cumsum(ladder)[:-1]]).astype(int)
    k = L + 1                                                   # mode 0 = rail, 1..L = slots
    Nm = np.zeros((k, k), dtype=np.complex128)
    Mm = np.zeros((k, k), dtype=np.complex128)
    tails = np.zeros((T, pmax + 1))
    mean = np.zeros(T)
    sh, ch = math.sinh(r), math.cosh(r)
    sq = number_law(np.array([[sh * sh]], complex), np.array([[-sh * ch]], complex), pmax) if n_src else None
    sq_gt = np.clip(1.0 - np.cumsum(sq), 0.0, None) if n_src else None          # P(q > k), k = 0..pmax
    for t in range(T):
        Nm[0, :] = 0; Nm[:, 0] = 0; Mm[0, :] = 0; Mm[:, 0] = 0  # a fresh rail wire (the previous one has exited)
        mean[t] = Nm.diagonal().real.sum() + (sh * sh if t < n_src else 0.0)
        if mean[t] > 0:
            Ps = number_law(Nm[1:, 1:], Mm[1:, 1:], pmax) if L else np.eye(1, pmax + 1)[0]  # slots, before injection
            if t < n_src:
                # dropped at this injection: slots j <= p (the truncated register never holds more), squeezer q > p - j
                tails[t] = np.convolve(Ps, sq_gt)[:pmax + 1]
        if t < n_src:
            Nm[0, 0] = sh * sh
            Mm[0, 0] = -sh * ch                                 # the sign of smsv_amps: c_2 / c_0 = -tanh(r) / sqrt(2)
        for i, l in enumerate(ladder):                          # the loop gates, in series along the rail
            s = 1 + off[i] + (t % l)
            c_, s_ = math.cos(theta[i, t]), math.sin(theta[i, t])
            e = complex(math.cos(phi[i, t]), math.sin(phi[i, t]))
            G = np.eye(k, dtype=np.complex128)
            G[0, 0], G[0, s], G[s, 0], G[s, s] = c_, -1j * s_ / e, -1j * s_ * e, c_
            Mm = G @ Mm @ G.T                                    # a -> G a
            Nm = G.conj() @ Nm @ G.T
    return tails, mean


def cap(ladder, N: int, ratio: float, r: float, eps: float, seed: int, pmax: int = 120, theta=None, phi=None):
    """(p, leak(p), leak by p, mean register photons max) for N sources: the smallest p with leak <= eps."""
    T = round(ratio * N)
    if theta is None:
        th, ph, _ = settings(len(ladder), T, seed)
        theta, phi = th.cpu().numpy(), ph.cpu().numpy()
    tails, mean = register_tails(ladder, T, N, r, np.asarray(theta), np.asarray(phi), pmax)
    leak = tails.sum(0)
    ok = np.nonzero(leak <= eps)[0]
    p = int(ok[0]) if ok.size else None
    return p, (float(leak[p]) if p is not None else float("nan")), leak, float(mean.max())


def gbs_loop_kernel(circuit, n_src: int, r: float, p: int, device: str = "cpu", dtype=None):
    """`SuperpositionLoopKernel` with the register capped at p photons and the per-bin leak recorded (`_track`)."""
    import torch

    from ..states import ProductSuperposition
    from ..timeloop.hybrid import SuperpositionLoopKernel, _NoBasis
    dtype = torch.complex128 if dtype is None else dtype

    class GBSLoopKernel(SuperpositionLoopKernel, _NoBasis):
        _n_max = int(p)
        _track = None

        def _emit(self, psi, t, conj=False):
            out = super()._emit(psi, t, conj)
            if self._track is not None and t in self._sbs:
                self._track.append(self._norm2(out, out.dim() - 2) / self._norm2(psi, psi.dim() - 2))
            return out

        def sample_weighted(self, shots, theta, phi, generator=None):
            """(samples [shots, T], leak per shot = 1 - prod_t surviving squared norm)."""
            self._track = []
            try:
                out = self.sample(shots, theta, phi, generator=generator)
                w = torch.ones(shots, dtype=self.rdtype, device=self.device)
                for s in self._track:
                    w = w * s.reshape(shots)
            finally:
                self._track = None
            return out, 1.0 - w

    amps = np.tile(smsv_amps(r, p + 1), (n_src, 1))
    st = ProductSuperposition(amps=amps, modes=tuple(range(n_src)), m=circuit.m)
    # built on a one-photon placeholder, then given the squeezer: SuperpositionLoopKernel's np.sqrt(math.factorial(q))
    # overflows the int64 ufunc path at q >= 21 (the cap fixes the register, not the placeholder's d)
    k = GBSLoopKernel(circuit, ProductSuperposition(amps=np.tile([1.0, 0.0], (n_src, 1)), modes=st.modes, m=circuit.m),
                      device=device, dtype=dtype)
    scale = np.exp(-0.5 * np.array([math.lgamma(q + 1) for q in range(p + 1)]))
    k.state = st
    k._sbs = {b: torch.as_tensor(st.amps[i] * scale, dtype=dtype, device=k.device) for i, b in enumerate(st.modes)}
    return k


