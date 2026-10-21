"""SLOS_full for single-emitter (MPS) inputs — the bond-carrying recurrence.

For an `EmitterMPS` input (states.py)

    |psi> = sum_x <f| M^(n)_{x_n} ... M^(1)_{x_1} |s0> |x_1 ... x_n>

the photonic state after k emissions, with the emitter still attached, is

    |Phi_k> = sum_s |s> (x) |c_s>,     chi amplitude vectors over the k-photon basis F_m^k,

and emission k is one SLOS layer applied to all chi vectors at once, mixed by the emitter
operators:

    |Phi_k> = sum_l ( M^(k)_l (x) A_{modes[k][l]} ) |Phi_{k-1}>,    A_p = sum_i U[i, p] a_i^dagger

i.e. the ordinary gather/scatter raising step, with the scalar coefficient U[j, p] of the Fock
recurrence replaced by the chi x chi matrix C_j = sum_l U[j, modes[k][l]] M^(k)_l of the raised
mode j. Edges are processed grouped by raised mode (`FockLayers.edges_by_mode`), so a layer
is chi gathers of the parent amplitudes, m small matmuls over contiguous slabs and chi
scatters - about chi x the work and memory of `FockSLOSKernel`; the L^n-component expansion of the
same state (which `SuperpositionSLOSKernel` would need) never appears. At the end the emitter is projected on `f`
(pure output amplitudes) or traced (a mixture of chi pure states, whose probabilities add).

Ragged chains (`EmitterMPS.labels`: L_k labels at event k) need nothing else: a padded slot is a
zero operator on the column modes[k][0], so C_j gains exactly 0 from it, and the cost is that of the
chain padded to L_max.

Normalization follows `FockSLOSKernel` (deferred): one photon per event into distinct input modes,
so the input factor is 1; the raw recurrence yields amplitude / prod(t_j!) and `probs`
multiplies |amp|^2 by prod(t_j!). Differentiable in U (and in the emitter operators, if they
are given as tensors requiring grad through `ops_tensor`).
"""
from __future__ import annotations

import torch

from ..masks import Mask
from ..states import EmitterMPS
from .basis import FockLayers, get_layers


class EmitterSLOSKernel:
    """Precompiled bond-carrying SLOS evaluator for a fixed (EmitterMPS, mask) on a device.

    Build once; call `probs(U)` / `amplitudes(U)` / `kraus(U)` many times. U is [m, m] or
    [B, m, m]; outputs are over the final (masked) basis of `self.layers`.
    """

    def __init__(self, state: EmitterMPS, mask: str = Mask.FOCK.value, device: str = "cpu",
                 dtype: torch.dtype = torch.complex64,
                 ops_tensor: torch.Tensor | None = None, layers=None):
        self.state = state
        self.n, self.m, self.mask = state.n, state.m, mask
        self.L, self.chi = state.num_labels, state.chi
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.layers: FockLayers = layers if layers is not None else get_layers(self.n, self.m, mask, device=str(self.device))
        if self.layers.output_size() == 0:
            raise ValueError(f"empty output space for n={self.n} m={self.m} mask={mask}")

        # emitter operators [n, L, chi, chi] (a caller-supplied tensor keeps its autograd graph)
        if ops_tensor is None:
            ops_tensor = torch.as_tensor(state.ops)
        if tuple(ops_tensor.shape) != (self.n, self.L, self.chi, self.chi):
            raise ValueError("ops_tensor must have shape [n, L, chi, chi]")
        self.ops = ops_tensor.to(device=self.device, dtype=self.cdtype)
        modes = state.modes
        if state.is_ragged:
            # padded slots: zero operator on the real column modes[k][0], so C_j gains exactly 0; the
            # constant mask keeps their gradient at zero (an optimiser cannot grow a phantom label)
            mask_t = torch.as_tensor(state.label_mask(), device=self.device)[..., None, None]
            if bool((self.ops.detach() * ~mask_t).abs().max() > 0):
                raise ValueError("ops_tensor: the padded slots of a ragged chain (EmitterMPS.label_mask) must be zero")
            self.ops = self.ops * mask_t
            modes = state.modes_padded()
        self.s0 = torch.as_tensor(state.s0).to(device=self.device, dtype=self.cdtype)   # [chi]
        self.final = (None if state.final is None
                      else torch.as_tensor(state.final).to(device=self.device, dtype=self.cdtype))
        # _in_mode[k]: LongTensor [L], the input column of U for each label of event k
        self._in_mode = [torch.tensor(list(modes[k]), dtype=torch.long, device=self.device)
                         for k in range(self.n)]

        self.output_prob_norm = torch.as_tensor(self.layers.output_prob_norms()).to(dtype=self.rdtype, device=self.device)
        self._output_norm_is_one = bool(torch.all(self.output_prob_norm == 1).item())

        # raising edges grouped by raised mode: within a slab the per-edge coefficient is the
        # constant chi x chi matrix C[j] = sum_l U[j, in_l] M_l, so a layer is `chi` gathers,
        # m small matmuls and `chi` scatters - no per-edge coefficient gather (see _propagate)
        self._parent, self._child, self._offsets, self._sizes = [], [], [], []
        for k in range(self.n):
            p, c, off = self.layers.edges_by_mode(k)
            self._parent.append(torch.as_tensor(p).to(device=self.device))
            self._child.append(torch.as_tensor(c).to(device=self.device))
            self._offsets.append(list(off))
            self._sizes.append(self.layers.size(k + 1))

    def output_size(self) -> int:
        """Size S of the output distribution (states in the final n-photon layer)."""
        return self.layers.output_size()

    # ---- core recurrence -------------------------------------------------------------------
    def _propagate(self, U: torch.Tensor, psi0: torch.Tensor) -> torch.Tensor:
        """U: [B, m, m]; psi0: [Q, chi] initial emitter vectors (Q of them, propagated together);
        returns |Phi_n> as [B, Q, chi, S] raw amplitudes (bond index = final emitter state).
        """
        B, Q = U.shape[0], psi0.shape[0]
        chi = self.chi
        psi = psi0.to(self.cdtype).view(1, Q, chi, 1).expand(B, Q, chi, 1).clone()
        for k in range(self.n):
            parent, child, off = self._parent[k], self._child[k], self._offsets[k]
            nxt = torch.zeros((B, Q, chi, self._sizes[k]), dtype=self.cdtype, device=self.device)
            if parent.numel():
                # C[b, j] = sum_l U[b, j, in_l] M^(k)_l : the coefficient matrix of raised mode j
                Uk = U[:, :, self._in_mode[k]]                                  # [B, m, L]
                C = torch.einsum("bjl,lts->bjts", Uk, self.ops[k])              # [B, m, chi, chi]
                # one slab per raised mode: gather its parents, apply C_j, scatter into the
                # children. Temporaries are slab-sized (E/m), so the layer's peak memory is the
                # two state vectors, as in FockSLOSKernel - no full-E intermediate.
                for j in range(self.m):
                    a, b_ = off[j], off[j + 1]
                    if b_ <= a:
                        continue
                    slab = torch.matmul(C[:, j].unsqueeze(1), psi[:, :, :, parent[a:b_].long()])   # [B, Q, chi, e]
                    nxt.scatter_add_(3, child[a:b_].long().view(1, 1, 1, -1).expand(B, Q, chi, -1), slab)
                    del slab
            psi = nxt
        return psi

    def _prep(self, U: torch.Tensor):
        batched = U.dim() == 3
        if not batched:
            U = U.unsqueeze(0)
        return U.to(device=self.device, dtype=self.cdtype), batched

    # ---- public API -------------------------------------------------------------------------
    def bond_amplitudes(self, U: torch.Tensor) -> torch.Tensor:
        """Raw amplitudes with the emitter attached: [chi, S] or [B, chi, S], index = final
        emitter basis state. The photonic state is the mixture of these chi pure states when
        the emitter is traced, or their contraction with `final` when it is projected.
        """
        U, batched = self._prep(U)
        phi = self._propagate(U, self.s0.unsqueeze(0))[:, 0]           # [B, chi, S]
        return phi if batched else phi.squeeze(0)

    def amplitudes(self, U: torch.Tensor) -> torch.Tensor:
        """Output amplitudes over the final basis for a PROJECTED emitter (state.final given):
        [S] or [B, S]. For a traced emitter use `bond_amplitudes` or `probs`.
        """
        if self.final is None:
            raise ValueError("amplitudes() needs a projected emitter (state.final); the traced "
                             "state is a mixture - use probs() or bond_amplitudes()")
        phi = self.bond_amplitudes(U)
        return torch.einsum("s,...sj->...j", self.final.conj(), phi)

    def probs(self, U: torch.Tensor, loss=None, indist=None, order: int | None = None):
        """Output probabilities over the final basis, [S] or [B, S]. Projected emitter: the
        (unnormalized, weight = projection probability) pure-state distribution; traced emitter:
        the mixture sum_s |<s| Phi_n>|^2. `loss=eta`: thinned to a `LossyDistribution` over
        0..n photons (uniform loss commutes with U and with the emitter trace/projection).
        `indist=x`: the x-model by species-resolved enumeration (`slos.species.emitter_x_mixture`,
        C(n, b) m^b folded chains; `order=b_max` truncates with `imperfections.truncation_bound`),
        one unitary at a time.
        """
        from ..imperfections import spec
        if spec(indist) is not None:
            from .kernel import _maybe_thin
            from .species import emitter_x_mixture
            return _maybe_thin(self.layers, emitter_x_mixture(self, U, indist, order=order), loss)
        if self.final is not None:
            amp = self.amplitudes(U)
            p = amp.real ** 2 + amp.imag ** 2
        else:
            phi = self.bond_amplitudes(U)
            p = (phi.real ** 2 + phi.imag ** 2).sum(dim=-2)
        if not self._output_norm_is_one:
            p = p * self.output_prob_norm
        from .kernel import _maybe_thin
        return _maybe_thin(self.layers, p, loss)

    def kraus(self, U: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """Stage Kraus operators on the emitter, one per output state: K[t] = <t| Phi_n> as a
        chi x chi matrix (final x initial). Returns [S, chi, chi] or [B, S, chi, chi], in the
        deferred normalization (multiply by sqrt(prod t_j!) for the physical operators).
        Detection of output t then updates the emitter as rho -> K[t] rho K[t]^dagger.
        chunk: propagate that many initial basis vectors at a time (the working set is
        chunk * chi * S amplitudes twice; the result, S * chi^2, is allocated once) — for large
        chi (product registers, photonic_engines.sequence) where the whole identity does not fit.
        """
        U, batched = self._prep(U)
        eye = torch.eye(self.chi, dtype=self.cdtype, device=self.device)
        if chunk is None or chunk >= self.chi:
            phi = self._propagate(U, eye)                               # [B, Q=init, chi=final, S]
            K = phi.permute(0, 3, 2, 1)                                 # [B, S, final, init]
            return K if batched else K.squeeze(0)
        B, S = U.shape[0], self.layers.output_size()
        K = torch.empty((B, S, self.chi, self.chi), dtype=self.cdtype, device=self.device)
        for q0 in range(0, self.chi, chunk):
            phi = self._propagate(U, eye[q0:q0 + chunk])                # [B, Qc, chi, S]
            K[:, :, :, q0:q0 + chunk] = phi.permute(0, 3, 2, 1)
            del phi
        return K if batched else K.squeeze(0)

    def kraus_species(self, U: torch.Tensor, indist, order: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Species-resolved stage Kraus set under partial distinguishability (x-model): (K [Q, chi,
        chi] in PHYSICAL normalisation, coarse [Q] long) — operator q belongs to the coarse output
        pattern `coarse[q]` (an index of the final basis); detection of t updates the register by
        the channel rho -> sum_{coarse[q] = t} K[q] rho K[q]^dag, and the sum over t is
        trace-preserving. `slos.species.emitter_x_mixture(kraus=True)`; one unitary.
        """
        from .species import emitter_x_mixture
        return emitter_x_mixture(self, U, indist, order=order, kraus=True)


def sample_from_probs(layers: FockLayers, probs, shots: int,
                      generator: torch.Generator | None = None, loss=None) -> torch.Tensor:
    """Draw output occupations [shots, m] from a probability vector over `layers.final_states`
    (strong simulation with sampling for free). `probs` need not be normalized (a projected
    emitter yields the conditional distribution). A `LossyDistribution` (probs(U, loss=eta)) is
    sampled over its 0..n photon layers; `loss=eta` on a lossless vector thins the samples.
    """
    from ..imperfections import LossyDistribution, draw_indices, thin_counts
    if isinstance(probs, LossyDistribution):
        return thin_counts(probs.sample(shots, generator=generator), loss, generator=generator)
    p = probs.detach().to(torch.float64)
    if p.dim() != 1:
        raise ValueError("one distribution at a time: probs must be [S]")
    idx = draw_indices(p, shots, generator)
    states = torch.as_tensor(layers.final_occupations()).to(p.device)
    return thin_counts(states[idx], loss, generator=generator)
