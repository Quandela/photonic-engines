"""Additive-error estimation of output amplitudes / probabilities for EmitterMPS inputs
(the generalised Gurvits estimator) — docs/ESTIMATE.md.

For an `EmitterMPS` input |psi> = sum_x psi(x) |x>  (photon k in mode modes[k][x_k], all modes
distinct) and an output occupation t (rows r_1..r_n with multiplicity, n = sum t):

    A_t = <t| U |psi> = sum_x psi(x) Perm(U[r, modes(x)]) / sqrt(prod_j t_j!)

is a sum of L^n permanents.  Gurvits's identity extends from Fock inputs to this state:
for x in C^m with i.i.d. standard complex Gaussian entries (or i.i.d. +-1 entries),

    sum_x psi(x) Perm(U[r, modes(x)]) = E_x[ Phi(x) g_t(x) ],
    Phi(x) = <X|psi> = sum_x psi(x) prod_k conj(x_{modes[k][x_k]})   (an MPS contraction, O(n L chi^2)),
    g_t(x) = prod_i (U x)_{r_i}.

Sampling x from q(x) = gamma(x) |Phi(x)|^2 (a sequential product-basis measurement of the MPS,
site by site, with the mode vectors as the measured "basis") and averaging g_t(x) / conj(Phi(x))
gives an unbiased estimator whose second moment is E_gamma |g_t|^2 = prod_j t_j! for Gaussian x
(unitary invariance), for every U and every input state.  Cost per sample O(n L chi^2 + m^2)
and 1/eps^2 samples for additive error eps — the same precision/shot trade-off as the hardware,
and independent of the 2^n / n 2^n exponentials of the strong and sampling engines.

Traced emitter (state.final is None): sum_r |A_{t,r}|^2 = E_z |<t, z| U |Psi>|^2 with z a
standard complex Gaussian vector on the emitter (E[z z^dagger] = I); each |A_{t,z}|^2 is the
product of two independent estimates of A_{t,z} (unbiased), second moment <= 1 + Tr rho_e^2.

Error bars. The per-sample variable Z is a product of n factors (Ux)_{r_i}: its second moment is
exactly ||psi||^2 t! (Gaussian) but its distribution is heavy-tailed (the median of |Z|^2 decays
like e^{-0.58 n}) - the sample variance with N shots underestimates the true variance once n is a
few tens, so `stderr` (empirical) is optimistic there. `stderr_bound` uses the theorem instead:
Chebyshev then gives |estimate - truth| <= stderr_bound / sqrt(delta) with probability 1 - delta,
whatever the tails; median-of-means sharpens it. The Rademacher variant has the same tails and no
theorem, but its empirical second moment is 2-10x below the Gaussian bound on every U measured.

This is a *weak* estimator: additive error only. It says nothing about sampling.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from ..states import EmitterMPS

C128 = torch.complex128


@dataclass(frozen=True)
class Estimate:
    """Monte-Carlo estimate with its standard error; `second_moment` = E|Z|^2 of the per-sample
    estimator (the theory gives <= prod t_j! for Gaussian sampling on a pure state).
    """
    value: complex
    stderr: float            # empirical: sqrt(sample variance / shots) - underestimates at large n (heavy tails)
    shots: int
    second_moment: float     # empirical E|Z|^2
    stderr_bound: float = float("nan")   # from the theorem: sqrt(E_z[||psi_z||^2]^2 t!^p / shots), p = 1 (amplitude) or 2 (probability)

    def __repr__(self):
        return (f"Estimate({self.value:.4e} +- {self.stderr:.1e} [bound {self.stderr_bound:.1e}], "
                f"shots={self.shots}, M2={self.second_moment:.3e})")


def _modes_from_occupation(occ: Sequence[int]):
    return [j for j, c in enumerate(occ) for _ in range(int(c))]


class EmitterGurvitsEstimator:
    """Gurvits-type estimator of <t|U|psi> and |<t|U|psi>|^2 for an `EmitterMPS` input.

    sampling : "gaussian"  — x ~ CN(0, 1)^m, second moment exactly prod t_j! (pure state)
               "rademacher" — x ~ {+-1}^m, second moment E_x prod_i |(Ux)_{r_i}|^2 (<= 1 in
                              every case measured, decreasing with n for Haar U); real
                              arithmetic on x, twice cheaper, no theorem.
    Build once per state; call `amplitude(U, t, shots)` / `probability(U, t, shots)`.
    """

    def __init__(self, state: EmitterMPS, sampling: str = "gaussian", device: str = "cpu",
                 dtype: torch.dtype = C128, seed: int | None = None):
        if sampling not in ("gaussian", "rademacher"):
            raise ValueError("sampling must be 'gaussian' or 'rademacher'")
        self.state, self.sampling = state, sampling
        self.n, self.m, self.L, self.chi = state.n, state.m, state.num_labels, state.chi
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == C128 else torch.float32
        self.ops = torch.as_tensor(state.ops, dtype=dtype, device=self.device)        # [n, L, chi, chi]
        self.s0 = torch.as_tensor(state.s0, dtype=dtype, device=self.device)
        self.final = None if state.final is None else torch.as_tensor(state.final, dtype=dtype, device=self.device)
        self.modes = torch.as_tensor(np.asarray(state.modes, dtype=np.int64), device=self.device)  # [n, L]
        used = {int(x) for mk in state.modes for x in mk}
        self.free_modes = torch.as_tensor([j for j in range(self.m) if j not in used], dtype=torch.int64, device=self.device)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
        if sampling == "rademacher":
            signs = torch.as_tensor(list(product((-1.0, 1.0), repeat=self.L)), dtype=self.rdtype, device=self.device)
            self.signs = signs.to(dtype)                                                # [2^L, L]

    # ---- random primitives ---------------------------------------------------------------
    def _cn(self, *shape) -> torch.Tensor:
        re = torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.rdtype)
        im = torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.rdtype)
        return torch.complex(re, im).to(self.cdtype) / math.sqrt(2.0)

    def _rad(self, *shape) -> torch.Tensor:
        return (torch.randint(0, 2, shape, generator=self.gen, device=self.device).to(self.rdtype) * 2 - 1).to(self.cdtype)

    # ---- right environments ---------------------------------------------------------------
    def _right_envs(self, f: torch.Tensor) -> torch.Tensor:
        """R[k] = sum_{x_{k+1..n}} (M^(n)_{x_n} .. M^(k+1)_{x_{k+1}})^dagger |f><f| (..), k = 0..n;
        f is [B, chi] (one final vector per shot) or [1, chi] (projected emitter: computed once and
        broadcast over the batch). Returns [n+1, B or 1, chi, chi].
        """
        B = f.shape[0]
        R = torch.empty(self.n + 1, B, self.chi, self.chi, dtype=self.cdtype, device=self.device)
        R[self.n] = f.unsqueeze(-1) @ f.conj().unsqueeze(-2)
        for k in range(self.n - 1, -1, -1):
            M = self.ops[k]                                                        # [L, chi, chi]
            acc = torch.zeros_like(R[k + 1])
            for l in range(self.L):
                acc = acc + M[l].conj().transpose(-1, -2) @ R[k + 1] @ M[l]         # M_l^dagger R M_l
            R[k] = acc
        return R

    def _eigh_small(self, rho: torch.Tensor):
        """Batched Hermitian eigendecomposition of [B, L, L]; on the CPU for small L (the CUDA batched
        solver reserves tens of GB of workspace for 2x2 problems).
        """
        if rho.is_cuda and self.L <= 8:
            lam, E = torch.linalg.eigh(rho.cpu())
            return lam.to(rho.device), E.to(rho.device)
        return torch.linalg.eigh(rho)

    # ---- one batch of importance samples --------------------------------------------------
    def _sample(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw x ~ gamma(x)|Phi(x)|^2 / ||psi_f||^2 sequentially. f: [B, chi]. Returns
        x [B, m], Phi = <X|psi_f> [B] and ||psi_f||^2 = <s0|R_0|s0> [B]; the importance density is
        gamma(x)|Phi(x)|^2 / ||psi_f||^2, so the estimator is norm2 * g / conj(Phi).
        """
        B = f.shape[0]
        R = self._right_envs(f[:1] if self.final is not None else f)                # projected: one f, broadcast
        x = torch.empty(B, self.m, dtype=self.cdtype, device=self.device)
        if self.free_modes.numel():
            draw = self._cn if self.sampling == "gaussian" else self._rad
            x[:, self.free_modes] = draw(B, self.free_modes.numel())
        a = self.s0.unsqueeze(0).expand(B, self.chi)                                # left vector [B, chi]
        for k in range(self.n):
            Ma = torch.einsum("lij,bj->bli", self.ops[k], a)                       # [B, L, chi]
            # rho[l, l'] = (M_l' a)^dagger R (M_l a): reduced density of event k in the conditional state
            RMa = torch.matmul(Ma, R[k + 1].transpose(-1, -2))                     # [B, L, chi]: R (M_l a)
            G = torch.matmul(Ma.conj(), RMa.transpose(-1, -2))                      # G[l, m] = (M_l a)^dag R (M_m a)
            rho = G.transpose(-1, -2)                                              # rho[l, l'] = G[l', l]
            tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real.clamp_min(1e-300)
            rho = rho / tr.unsqueeze(-1).unsqueeze(-1)
            if self.sampling == "gaussian":
                lam, E = self._eigh_small(rho)                                     # ascending, E[:, :, j] eigvecs
                lam = lam.clamp_min(0.0)
                lam = lam / lam.sum(-1, keepdim=True)
                j = torch.multinomial(lam, 1, generator=self.gen).squeeze(-1)      # [B]
                c = self._cn(B, self.L)                                            # coefficients in eigenbasis
                # replace coefficient j by c with |c|^2 ~ Gamma(2,1): |c|^2 = -log(u1) - log(u2)
                u = torch.rand(B, 2, generator=self.gen, device=self.device, dtype=self.rdtype).clamp_min(1e-300)
                r2 = -(torch.log(u[:, 0]) + torch.log(u[:, 1]))
                ph = torch.rand(B, generator=self.gen, device=self.device, dtype=self.rdtype) * 2 * math.pi
                cj = (torch.sqrt(r2) * torch.exp(1j * ph)).to(self.cdtype)
                c[torch.arange(B, device=self.device), j] = cj
                w = torch.einsum("blj,bj->bl", E, c)                                # w = sum_j c_j e_j
            else:
                S = self.signs                                                      # [2^L, L]
                wts = torch.einsum("sl,blm,sm->bs", S.conj(), rho, S).real.clamp_min(0.0)
                idx = torch.multinomial(wts / wts.sum(-1, keepdim=True), 1, generator=self.gen).squeeze(-1)
                w = S[idx]                                                          # [B, L]
            x.scatter_(1, self.modes[k].unsqueeze(0).expand(B, self.L), w)
            a = torch.einsum("bl,bli->bi", w.conj(), Ma)                            # contract site k with conj(w)
        Phi = torch.einsum("bi,bi->b", f.conj(), a)                                 # <f| a_n >
        norm2 = torch.einsum("i,bij,j->b", self.s0.conj(), R[0], self.s0).real       # ||psi_f||^2 = <s0|R_0|s0>
        if norm2.shape[0] != B:
            norm2 = norm2.expand(B)
        return x, Phi, norm2

    def _g(self, U: torch.Tensor, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        y = x @ U.transpose(-1, -2)                                                # (U x)_i  [B, m]
        return y[:, rows].prod(dim=1)

    def _batch(self, batch: int) -> int:
        """Traced emitter: the right environments are per shot ([n+1, B, chi, chi]); cap the batch so
        they stay under ~2 GB.
        """
        if self.final is not None:
            return batch
        return max(64, min(batch, (1 << 31) // ((self.n + 1) * self.chi * self.chi * 16)))

    def _finals(self, B: int) -> torch.Tensor:
        if self.final is not None:
            return self.final.unsqueeze(0).expand(B, self.chi)
        return self._cn(B, self.chi)                                                # traced: Gaussian purification vector

    # ---- public API -------------------------------------------------------------------------
    def amplitude(self, U: torch.Tensor, output: Sequence[int], shots: int, batch: int = 100_000) -> Estimate:
        """Estimate A_t = <t|U|psi> for a PROJECTED emitter (state.final given)."""
        if self.final is None:
            raise ValueError("amplitude() needs a projected emitter (state.final); use probability() for a traced one")
        U = torch.as_tensor(U, dtype=self.cdtype, device=self.device)
        rows, norm = self._rows(output)
        batch = self._batch(batch)
        tot, tot2, done, n2 = 0j, 0.0, 0, 0.0
        while done < shots:
            B = min(batch, shots - done)
            x, Phi, norm2 = self._sample(self._finals(B))
            Z = norm2 * self._g(U, x, rows) / Phi.conj() / norm
            tot += complex(Z.sum()); tot2 += float((Z.real ** 2 + Z.imag ** 2).sum()); done += B
            n2 += float(norm2.sum())
        mean = tot / shots
        m2 = tot2 / shots
        var = max(m2 - abs(mean) ** 2, 0.0)
        bound = math.sqrt((n2 / shots) * norm ** 2 / shots)                          # E|Z|^2 = ||psi||^2 t!  (norm^2 = t!)
        return Estimate(mean, math.sqrt(var / shots), shots, m2, bound)

    def probability(self, U: torch.Tensor, output: Sequence[int], shots: int, batch: int = 100_000) -> Estimate:
        """Estimate p(t) = |<t|U|psi>|^2 (projected emitter: unnormalised by the projection weight,
        as EmitterSLOSKernel.probs) or the traced-emitter mixture sum_r |<t, r|U|Psi>|^2.
        Each shot is the product of two independent amplitude estimates (unbiased).
        """
        U = torch.as_tensor(U, dtype=self.cdtype, device=self.device)
        rows, norm = self._rows(output)
        batch = self._batch(batch)
        tot, tot2, done, n4 = 0.0, 0.0, 0, 0.0
        while done < shots:
            B = min(batch, shots - done)
            f = self._finals(B)
            x1, P1, n1 = self._sample(f)
            x2, P2, n2 = self._sample(f)
            Z1 = n1 * self._g(U, x1, rows) / P1.conj() / norm
            Z2 = n2 * self._g(U, x2, rows) / P2.conj() / norm
            Z = (Z1 * Z2.conj()).real
            tot += float(Z.sum()); tot2 += float((Z ** 2).sum()); done += B
            n4 += float((n1 ** 2).sum())
        mean = tot / shots
        m2 = tot2 / shots
        bound = math.sqrt((n4 / shots) * norm ** 4 / shots)                         # E|Z1 Z2|^2 = (||psi_z||^2 t!)^2
        return Estimate(mean, math.sqrt(max(m2 - mean ** 2, 0.0) / shots), shots, m2, bound)

    def _rows(self, output: Sequence[int]):
        occ = [int(x) for x in output]
        if len(occ) != self.m or sum(occ) != self.n:
            raise ValueError(f"output must be an occupation over {self.m} modes with {self.n} photons")
        rows = torch.as_tensor(_modes_from_occupation(occ), dtype=torch.int64, device=self.device)
        norm = math.sqrt(math.prod(math.factorial(c) for c in occ))
        return rows, norm
