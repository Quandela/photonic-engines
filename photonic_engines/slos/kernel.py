"""The SLOScore SLOS kernel: unitary -> output probability distribution.

Amplitudes are built photon-by-photon over the (mask-pruned) Fock basis. Applying the k-th
input photon in mode `i` sends the layer-k amplitude vector psi_k to layer k+1 via

    psi_{k+1}[child] += U[j, i] * sqrt(parent[j] + 1) * psi_k[parent]

for every raising edge (parent -> child by one photon in mode j). This is exactly one
`scatter_add_` per layer over precomputed edge tensors - fully vectorized, batched over the
leading unitary dimension, device-resident, and differentiable (no TorchScript, no Python loop
over the basis, no per-output permanent). `scatter_add_` (not `index_add_`) is a memory
contract: IndexAddBackward saves its full [B,E] source for backward, ScatterAddBackward saves
only the indices - with index_add_ the autograd graph retains one extra [B,E_k] per layer
(H-MEM-SCATTER-SAVES; the expanded index is a zero-byte view, same atomic-add semantics).

Correctness note: the sqrt bosonic factors make this produce per(U_{s,t}) / sqrt(prod t_j!)
automatically, including bunched outputs - verified against `reference.reference_probs`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from ..imperfections import LossyDistribution
from ..masks import Mask, graded_state_space_size
from .basis import FockLayers, get_layers


def _maybe_thin(layers, p: torch.Tensor, loss):
    """`loss=None` / 1: the n-photon distribution p; else its uniform-loss thinning."""
    if loss is None or (not isinstance(loss, torch.Tensor) and float(loss) >= 1.0):
        return p
    from ..imperfections import thin_distribution
    return thin_distribution(layers, p, loss)


class _CheckpointedLayer(torch.autograd.Function):
    """One SLOS layer with RECOMPUTED backward gathers (the H-GRAD-CHECKPOINT experiment).

    Default autograd saves BOTH [B,E] operands of the layer multiply for backward
    (2 x sum_k [B,E_k] ~= 3.8 GiB at n8/m16 b64 - the dominant grad-memory cost, paid
    identically by dperceval). This Function saves only U and the layer input psi_k
    ([B,S_k]; U is a shared ref, no copy) and REDOES the two gathers in backward -
    trading one extra gather+multiply per layer for dropping the [B,E] saves.
    Gradients follow torch's Wirtinger convention (grad_a = grad_c * b.conj()) and are
    gated for equality against the standard path in test_sloscore_kernel.py.
    """

    @staticmethod
    def forward(ctx, U, psi, parent, child, mode, col, size):
        B = psi.shape[0]
        contrib = U[:, mode, col] * psi[:, parent]                  # [B, E]
        nxt = torch.zeros((B, size), dtype=psi.dtype, device=psi.device)
        nxt.scatter_add_(1, child.unsqueeze(0).expand_as(contrib), contrib)
        del contrib
        ctx.save_for_backward(U, psi, parent, child, mode)
        ctx.col = col
        return nxt

    @staticmethod
    def backward(ctx, grad_out):
        U, psi, parent, child, mode = ctx.saved_tensors
        B, E = psi.shape[0], parent.numel()
        grad_contrib = grad_out.gather(1, child.unsqueeze(0).expand(B, E))   # [B, E]
        grad_psi = None
        if ctx.needs_input_grad[1]:
            grad_psi = torch.zeros_like(psi)
            grad_psi.scatter_add_(1, parent.unsqueeze(0).expand(B, E),
                                  grad_contrib * U[:, mode, ctx.col].conj())
        grad_U = None
        if ctx.needs_input_grad[0]:
            gcol = torch.zeros((B, U.shape[1]), dtype=U.dtype, device=U.device)
            gcol.scatter_add_(1, mode.unsqueeze(0).expand(B, E),
                              grad_contrib * psi[:, parent].conj())
            grad_U = torch.zeros_like(U)
            grad_U[:, :, ctx.col] = gcol
        return grad_U, grad_psi, None, None, None, None, None


class FockSLOSKernel:
    """Precompiled SLOS evaluator for a fixed (n, m, mask, input) on a fixed device/dtype.

    Build once (setup); call `probs(U)` many times (the timed op). U is [m, m] or [B, m, m];
    the returned probabilities are [S] or [B, S] over the final (masked) output basis.
    """

    def __init__(self, n: int, m: int, input_occupation: Sequence[int],
                 mask: str = Mask.FOCK.value, device: str = "cpu",
                 dtype: torch.dtype = torch.complex64,
                 normalization: str = "deferred",
                 gradient_checkpointing: bool = False,
                 layout: str = "mode", layers=None):
        if layout not in ("mode", "edge"):
            raise ValueError("layout must be 'mode' (edges grouped by raised mode, slab-wise; "
                             "no edge-sized temporaries) or 'edge' (original child-major form)")
        if layout == "mode" and (gradient_checkpointing or normalization != "deferred"):
            layout = "edge"                        # those arms are defined on the edge layout
        self.layout = layout
        if normalization not in ("deferred", "immediate"):
            raise ValueError(f"normalization must be 'deferred' or 'immediate', got {normalization!r}")
        if gradient_checkpointing and normalization != "deferred":
            raise ValueError("gradient_checkpointing requires deferred normalization")
        # OPT-IN experiment arm (H-GRAD-CHECKPOINT): recompute backward gathers instead of
        # letting autograd save them. Off by default - the standard path is untouched.
        self.gradient_checkpointing = gradient_checkpointing
        # "immediate" applies the per-edge bosonic sqrt factor inside the hot loop (the textbook
        # recurrence, one extra [B,E] multiply per layer) instead of repairing magnitudes at the
        # end. It exists as the ABLATION ARM for hypothesis H-MEM-TRADE / H-DEFERRED-NORM-33
        # (results/claims.json): the docs claim deferral buys GPU speed at a transient-memory
        # cost - running both arms in every objective run is what tests that, permanently.
        self.normalization = normalization
        self.n, self.m, self.mask = n, m, mask
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.layers: FockLayers = layers if layers is not None else get_layers(n, m, mask, device=str(self.device))
        if self.layers.output_size() == 0:
            raise ValueError(f"empty output space for n={n} m={m} mask={mask}")

        # input modes, one entry per photon (repeats a mode if the input bunches it)
        self.input_modes: list[int] = [j for j, c in enumerate(input_occupation) for _ in range(int(c))]
        if len(self.input_modes) != n:
            raise ValueError(f"input has {len(self.input_modes)} photons, expected n={n}")
        # DEFERRED normalization (matches dperceval's TorchSLOS, and it is why the hot loop can
        # skip the per-edge sqrt factor): the raw factor-less recurrence yields per(U)/prod(t_j!),
        # so probs = |raw * input_norm|^2 * output_prob_norm with
        #   input_norm      = 1/sqrt(prod_i s_i!)   (scalar; applied to the amplitude)
        #   output_prob_norm = prod_j t_j!          ([S]; applied to the probability)
        # Skipping the per-edge factor removes a complex multiply AND a full [B,E] intermediate
        # tensor per layer - the measured ~33% GPU gap vs dperceval at high photon count.
        norm = 1.0
        for c in input_occupation:
            norm *= math.factorial(int(c))
        self._input_norm = 1.0 / math.sqrt(norm)
        self._input_norm_is_one = (norm == 1)          # skip the multiply in the common case
        self.output_prob_norm = torch.as_tensor(self.layers.output_prob_norms()).to(dtype=self.rdtype, device=self.device)
        # unbunched-family masks have all t_j<=1 => prod(t_j!)==1 => the output multiply is a no-op
        self._output_norm_is_one = bool(torch.all(self.output_prob_norm == 1).item())

        # per-layer edge tensors, moved to device once. The basis hands us flat int64/float64
        # arrays (see basis.py edge STORAGE note), so cpu tensors are ZERO-COPY views shared
        # across every kernel built from the lru-cached FockLayers; cuda copies once. Deferred
        # mode drops the per-edge factor (repaired via output_prob_norm); immediate keeps it.
        self._parent, self._child, self._mode, self._factor, self._sizes = [], [], [], [], []
        for k in range(n):
            self._sizes.append(self.layers.size(k + 1))
            if self.layout != "edge":
                continue                     # edge-layout arrays only when that layout runs
            p, c, j, f = self.layers.edges[k]
            self._parent.append(torch.as_tensor(p).to(device=self.device, dtype=torch.long))
            self._child.append(torch.as_tensor(c).to(device=self.device, dtype=torch.long))
            self._mode.append(torch.as_tensor(j).to(device=self.device, dtype=torch.long))
            self._factor.append(torch.as_tensor(f).to(dtype=self.rdtype, device=self.device)
                                if normalization == "immediate" else None)
        # mode layout: the same edges regrouped by raised mode; within a slab the coefficient
        # U[:, j, i] is a scalar, so a layer is m slab gathers/scatters with no [B, E] temporary
        # (the edge layout materializes 2-3 of them per layer - the memory peak at large n)
        self._slab_parent, self._slab_child, self._offsets = [], [], []
        if self.layout == "mode":
            for k in range(n):
                p, c, off = self.layers.edges_by_mode(k)
                self._slab_parent.append(torch.as_tensor(p).to(device=self.device))       # int32, widened per slab
                self._slab_child.append(torch.as_tensor(c).to(device=self.device))
                self._offsets.append(list(off))

    def output_size(self) -> int:
        """Size S of the output distribution (states in the final layer)."""
        return self.layers.output_size()

    def _add_oom_hint(self, e: BaseException) -> None:
        """User guidance on CUDA OOM: a HINT, never a retry. Attached only when the knob
        would actually help (grad mode, checkpointing off). Forward is where grad-mode saves
        accumulate, so most training OOMs fire here; a backward-TRANSIENT OOM raises inside
        the autograd engine where no kernel code runs - those cannot be annotated from here.
        Python >= 3.11 attaches it as an exception note (PEP 678); 3.10 has no add_note, so
        the hint is appended to the message (e.args).
        """
        if not self.gradient_checkpointing and torch.is_grad_enabled():
            hint = ("hint: FockSLOSKernel(gradient_checkpointing=True) reduces peak GPU memory "
                    "in training, at the cost of a slower backward pass.")
            if hasattr(e, "add_note"):
                e.add_note(hint)
            elif len(e.args) == 1 and isinstance(e.args[0], str):
                e.args = (f"{e.args[0]}\n{hint}",)
            else:
                e.args = (*e.args, hint)

    def amplitudes(self, U: torch.Tensor) -> torch.Tensor:
        """Output amplitudes over the final basis. U: [m, m] or [B, m, m] -> [S] or [B, S].
        On CUDA OOM, attaches the gradient_checkpointing hint (see _add_oom_hint).
        """
        try:
            return self._amplitudes(U)
        except torch.cuda.OutOfMemoryError as e:
            self._add_oom_hint(e)
            raise

    def _amplitudes(self, U: torch.Tensor, input_modes: Sequence[int] | None = None) -> torch.Tensor:
        """`input_modes` (optional): run the recurrence for these g <= n input columns instead of
        the kernel's own, stopping on layer g of the SAME basis — the interfering subset of a
        partially distinguishable input (`slos.species`). Raw amplitudes: the input norm is the
        caller's (1 for one photon per mode).
        """
        modes = self.input_modes if input_modes is None else list(input_modes)
        batched = U.dim() == 3
        if not batched:
            U = U.unsqueeze(0)
        B = U.shape[0]
        U = U.to(device=self.device, dtype=self.cdtype)

        # psi_0 : vacuum amplitude 1 (single state)
        psi = torch.ones((B, 1), dtype=self.cdtype, device=self.device)
        if self.layout == "mode":
            for k, i in enumerate(modes):
                parent, child, off = self._slab_parent[k], self._slab_child[k], self._offsets[k]
                nxt = torch.zeros((B, self._sizes[k]), dtype=self.cdtype, device=self.device)
                for j in range(self.m):
                    a, b_ = off[j], off[j + 1]
                    if b_ <= a:
                        continue
                    slab = U[:, j, i].unsqueeze(1) * psi[:, parent[a:b_].long()]      # [B, e]
                    nxt.scatter_add_(1, child[a:b_].long().unsqueeze(0).expand(B, -1), slab)
                    del slab
                psi = nxt
            if not self._input_norm_is_one and input_modes is None:
                psi = psi * self._input_norm
            return psi if batched else psi.squeeze(0)
        for k, i in enumerate(modes):
            parent, child, mode = self._parent[k], self._child[k], self._mode[k]
            if self.gradient_checkpointing and parent.numel():
                psi = _CheckpointedLayer.apply(U, psi, parent, child, mode, i,
                                               self._sizes[k])
                continue
            if parent.numel():
                # raw recurrence: gather U column [B,E] and parent amps [B,E], multiply, scatter.
                # Deferred: no per-edge bosonic factor (repaired via output_prob_norm).
                # Immediate (ablation arm): apply sqrt(parent_occ+1) here - one extra [B,E]
                # multiply + intermediate per layer, no end-of-run repair.
                contrib = U[:, mode, i] * psi[:, parent]               # [B, E]
                if self.normalization == "immediate":
                    contrib = contrib * self._factor[k]
                # allocate the next layer only AFTER the 3x[B,E] gather/multiply peak has
                # passed - preallocating it stacked an extra [B,S_k] on top of that peak
                # (part of the measured constant-factor memory gap vs dperceval, MEM-TRADE)
                nxt = torch.zeros((B, self._sizes[k]), dtype=self.cdtype, device=self.device)
                # scatter_add_, NOT index_add_: its backward saves only the index view, not the
                # [B,E] source (see module docstring; the expand is a zero-byte view)
                nxt.scatter_add_(1, child.unsqueeze(0).expand(B, -1), contrib)
                # release the binding NOW: otherwise this layer's [B,E_k] stays live through the
                # NEXT layer's 3x[B,E_{k+1}] gather peak (the residual gap vs dperceval)
                del contrib
            else:
                nxt = torch.zeros((B, self._sizes[k]), dtype=self.cdtype, device=self.device)
            psi = nxt
        if not self._input_norm_is_one and input_modes is None:
            psi = psi * self._input_norm
        return psi if batched else psi.squeeze(0)

    def probs(self, U: torch.Tensor, loss=None, indist=None, order: int | None = None):
        """Output probability distribution over the final (masked) basis. With `loss=eta`
        (uniform transmission < 1) the lossless distribution is thinned down the basis layers
        and a `LossyDistribution` over 0..n photons is returned instead (photonic_engines.imperfections).
        With `indist=x` (partial distinguishability, x-model, docs/DISTINGUISHABILITY.md) the
        distribution is the mixture over the interfering subsets G of the photons, weights
        x^|G| (1 - x)^(n - |G|), each an SLOS run on the columns of G followed by the classical
        routing of the others (`slos.species.x_mixture`); `order=b` keeps the subsets with at
        most b distinguishable photons, the dropped mass being `imperfections.truncation_bound`.
        A tensor x gets a gradient. Loss is applied after (the two commute).
        """
        from ..imperfections import spec
        if spec(indist) is not None:
            from .species import x_mixture
            return _maybe_thin(self.layers, x_mixture(self, U, indist, order=order), loss)
        amp = self.amplitudes(U)
        p = amp.real ** 2 + amp.imag ** 2                # cheaper than (conj*amp).real
        if self.normalization != "immediate" and not self._output_norm_is_one:
            # in-place: p is a fresh intermediate and mul's backward only needs the constant
            # norm buffer, so this saves a full [B,S] allocation (dperceval does the same)
            p *= self.output_prob_norm                   # ("immediate": magnitudes already exact in-loop)
        return _maybe_thin(self.layers, p, loss)

    # -- optional post-processing hooks (grouping) live here so they compose with the kernel --

    def group_matrix(self, groups: torch.Tensor) -> GroupedSLOSKernel:
        """Fold a group assignment (LongTensor [S] -> bin id) into the kernel: probabilities are
        scatter-reduced into bins at the final step. This shrinks the *returned* output (n_groups vs
        S), NOT the amplitude compute (a group is an arbitrary set of final states, so the full
        recurrence still runs; measured ~1x). See SLOScore_REPORT.md §3.4.
        """
        return GroupedSLOSKernel(self, groups)


class SuperpositionSLOSKernel:
    """EBS kernel: evaluate K coherent-superposition input components in ONE batched SLOS
    recurrence instead of K separate passes.

    All components share the same (mask-pruned) output basis and raising edges - only the
    per-photon *input mode* differs between components. So the K single-component sweeps fold
    into a single vectorized sweep with an extra component-batch dimension: one `index_add_`
    per photon layer over a [B, K, S] tensor, rather than K serial, host-bound kernel launches.
    This is SLOScore's version of merlin's EBS "compute simultaneously" - the GPU win grows
    with K (fewer launches, wider tensors), while the naive alternative (looping `FockSLOSKernel`,
    as `_Engine` does for the K=2 GHZ case) pays K launch costs.

    `chunk_size` caps the component batch (and thus peak memory) for large K: components are
    processed in chunks and their weighted amplitudes accumulated, so the coherent sum never
    materializes all K state vectors at once.
    """

    def __init__(self, n: int, m: int, input_occupations: Sequence[Sequence[int]],
                 mask: str = Mask.FOCK.value, device: str = "cpu",
                 dtype: torch.dtype = torch.complex64, weights: Sequence[complex] | None = None):
        self.n, self.m, self.mask = n, m, mask
        self.default_weights = None if weights is None else torch.as_tensor(np.asarray(weights), dtype=dtype, device=device)
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.layers: FockLayers = get_layers(n, m, mask, device=str(self.device))
        if self.layers.output_size() == 0:
            raise ValueError(f"empty output space for n={n} m={m} mask={mask}")
        self.K = len(input_occupations)
        if self.K == 0:
            raise ValueError("need at least one input component")

        # per-component input mode sequence (one entry per photon) + bunching normalization,
        # vectorized over the K components (K may be large): occ [K, m] -> in_modes [K, n]
        occ = np.asarray(input_occupations, dtype=np.int64)
        if occ.ndim != 2 or occ.shape[1] != m or (occ.sum(1) != n).any():
            raise ValueError(f"input_occupations must be [K, {m}] with {n} photons per component")
        in_modes = np.repeat(np.broadcast_to(np.arange(m), occ.shape), occ.reshape(-1)).reshape(self.K, n)
        lgamma = np.vectorize(math.lgamma)(occ + 1.0)                          # log(c!)
        norms = np.exp(-0.5 * lgamma.sum(1))
        # _in_mode[k] : LongTensor [K] = input mode each component sends its k-th photon into
        in_modes_t = torch.as_tensor(in_modes, device=self.device)
        self._in_mode = [in_modes_t[:, k].contiguous() for k in range(n)]
        self._input_norm = torch.as_tensor(norms, dtype=self.rdtype,
                                           device=self.device).to(self.cdtype)   # [K]
        # deferred normalization (see FockSLOSKernel): raw recurrence, prod(t_j!) applied to probs
        self.output_prob_norm = torch.as_tensor(self.layers.output_prob_norms()).to(dtype=self.rdtype, device=self.device)
        self._output_norm_is_one = bool(torch.all(self.output_prob_norm == 1).item())

        # shared per-layer raising edges (identical to FockSLOSKernel; independent of the input; the
        # per-edge sqrt factor is deferred to output_prob_norm). Flat-array basis storage ->
        # zero-copy cpu tensors (see FockSLOSKernel.__init__ note).
        self._parent, self._child, self._mode, self._sizes = [], [], [], []
        for k in range(n):
            p, c, j, _f = self.layers.edges[k]
            self._parent.append(torch.as_tensor(p).to(device=self.device, dtype=torch.long))
            self._child.append(torch.as_tensor(c).to(device=self.device, dtype=torch.long))
            self._mode.append(torch.as_tensor(j).to(device=self.device, dtype=torch.long))
            self._sizes.append(self.layers.size(k + 1))

    @classmethod
    def from_state(cls, state, mask: str = Mask.FOCK.value, device: str = "cpu",
                   dtype: torch.dtype = torch.complex64) -> SuperpositionSLOSKernel:
        """Kernel for a `FockSuperposition`; its amplitudes become the default weights, so
        `probs(U)` is the distribution of the (normalized) state.
        """
        return cls(state.n, state.m, state.occupations, mask=mask, device=device, dtype=dtype,
                   weights=state.amplitudes / state.norm())

    def output_size(self) -> int:
        """Size S of the output distribution (states in the final layer)."""
        return self.layers.output_size()

    def _component_amplitudes(self, U: torch.Tensor, comp_idx: torch.Tensor) -> torch.Tensor:
        """Amplitudes for a subset of components. U: [B, m, m]; returns [B, len(comp_idx), S]."""
        kc = int(comp_idx.numel())
        B = U.shape[0]
        psi = torch.ones((B, kc, 1), dtype=self.cdtype, device=self.device)
        for k in range(self.n):
            parent, child, mode = self._parent[k], self._child[k], self._mode[k]
            if parent.numel():
                E = mode.shape[0]
                in_k = self._in_mode[k][comp_idx]              # [kc]
                rows = mode.unsqueeze(0).expand(kc, E)         # [kc, E] (view)
                cols = in_k.unsqueeze(1).expand(kc, E)         # [kc, E]
                # raw recurrence, no per-edge factor (deferred to output_prob_norm)
                contrib = U[:, rows, cols] * psi[:, :, parent]  # [B, kc, E]
                # allocate the next layer after the gather/multiply peak (see FockSLOSKernel)
                nxt = torch.zeros((B, kc, self._sizes[k]), dtype=self.cdtype, device=self.device)
                # scatter_add_ + released binding: same memory contract as FockSLOSKernel
                nxt.scatter_add_(2, child.view(1, 1, -1).expand(B, kc, -1), contrib)
                del contrib
            else:
                nxt = torch.zeros((B, kc, self._sizes[k]), dtype=self.cdtype, device=self.device)
            psi = nxt
        return psi * self._input_norm[comp_idx].view(1, kc, 1)

    def amplitudes(self, U: torch.Tensor, weights: Sequence[complex] | None = None,
                   chunk_size: int | None = None) -> torch.Tensor:
        """Coherently summed output amplitudes over the K components.

        U: [m, m] or [B, m, m]; weights: [K] (default: the state's amplitudes for a kernel built
        with `from_state` / `weights=`, else uniform 1/sqrt(K)); returns [S] or [B, S].
        chunk_size caps the component batch (peak memory) - the result is identical regardless.
        """
        batched = U.dim() == 3
        if not batched:
            U = U.unsqueeze(0)
        B = U.shape[0]
        U = U.to(device=self.device, dtype=self.cdtype)
        if weights is None and self.default_weights is not None:
            w = self.default_weights
        elif weights is None:
            w = torch.full((self.K,), 1.0 / math.sqrt(self.K),
                           dtype=self.cdtype, device=self.device)
        else:
            w = torch.as_tensor(weights, dtype=self.cdtype, device=self.device)
        S = self.output_size()
        cs = self.K if chunk_size is None else max(1, int(chunk_size))
        amp = torch.zeros((B, S), dtype=self.cdtype, device=self.device)
        for start in range(0, self.K, cs):
            idx = torch.arange(start, min(start + cs, self.K), device=self.device)
            psi = self._component_amplitudes(U, idx)           # [B, kc, S]
            amp = amp + (w[idx].view(1, -1, 1) * psi).sum(dim=1)
        return amp if batched else amp.squeeze(0)

    def probs(self, U: torch.Tensor, weights: Sequence[complex] | None = None,
              chunk_size: int | None = None, loss=None):
        """Output probability distribution of the coherent superposition (`loss=eta`: thinned,
        see FockSLOSKernel.probs).
        """
        amp = self.amplitudes(U, weights=weights, chunk_size=chunk_size)
        p = amp.real ** 2 + amp.imag ** 2
        if not self._output_norm_is_one:
            p *= self.output_prob_norm                   # in-place: saves a [B,S] (see FockSLOSKernel)
        return _maybe_thin(self.layers, p, loss)


class ProductSLOSKernel:
    """SLOS for a `ProductSuperposition` input: the amplitudes of EVERY photon-number layer.

    The input is a product over sources rather than a list of photons, so the recurrence runs over
    SOURCES, and a source that may or may not emit moves amplitude BETWEEN layers:

        psi'[k] = sum_{q < d}  (a[i, q] / sqrt(q!)) * R_i^q psi[k - q]

    with R_i the raising operator of that source's input mode - the k-1 -> k edges of the same
    basis `FockSLOSKernel` walks, one `scatter_add_` per mode slab. Two consequences:

      * every layer is live, not just one, so the working state is the graded basis
        sum_k S_k = C(m + n, n) (`masks.graded_state_space_size`), 1.5x the n-photon layer at
        m = 2n, and the work is sum_k (n - k + 1) |E_k| against the Fock kernel's sum_k |E_k|;
      * `probs` returns a `loss.LossyDistribution` - one block per photon number, the masses being
        `state.photon_number_probs()` whatever U is - and not a single [S] vector.

    Deferred normalization exactly as in `FockSLOSKernel`: the raw recurrence carries no per-edge
    bosonic factor, the input side contributes 1/sqrt(q!) per source (folded into the coefficients
    above) and the output side prod_j t_j! per layer (`layers.prob_norms(k)`). Zero coefficients
    are skipped, so alpha = pi/2 (a[i] = (0, 1)) runs exactly the Fock recurrence.
    """

    def __init__(self, state, mask: str = Mask.FOCK.value, device: str = "cpu",
                 dtype: torch.dtype = torch.complex64, layers=None):
        self.state = state
        self.n, self.m, self.mask = state.n, state.m, mask
        self.d = state.d
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        if self.n == 0:
            raise ValueError("every source is in |0>: nothing to propagate")
        graded_state_space_size(self.n, self.m, mask)          # rejects masks without a graded space
        self.layers: FockLayers = layers if layers is not None else get_layers(self.n, self.m, mask, device=str(self.device))
        fact = np.array([math.factorial(q) for q in range(self.d)], dtype=np.float64)
        coef = np.asarray(state.amps, dtype=np.complex128) / np.sqrt(fact)       # [sources, d]
        self._coef = torch.as_tensor(coef, dtype=self.cdtype, device=self.device)
        self._nonzero = (coef != 0).tolist()                   # skip empty branches (host-side)
        self._modes = state.modes
        self._sizes = [self.layers.size(k) for k in range(self.n + 1)]
        self._prob_norm = [torch.as_tensor(self.layers.prob_norms(k)).to(dtype=self.rdtype, device=self.device)
                           for k in range(self.n + 1)]
        # mode-slab edges, one set per k -> k+1 layer step (shared with FockSLOSKernel's "mode" layout)
        self._slab_parent, self._slab_child, self._offsets = [], [], []
        for k in range(self.n):
            p, c, off = self.layers.edges_by_mode(k)
            self._slab_parent.append(torch.as_tensor(p).to(device=self.device))   # int32, widened per slab
            self._slab_child.append(torch.as_tensor(c).to(device=self.device))
            self._offsets.append([int(v) for v in off])

    def output_size(self) -> int:
        """Total number of output states over all layers, sum_k S_k (the graded basis)."""
        return int(sum(self._sizes))

    def block_sizes(self) -> list[int]:
        """S_k per photon number k = 0..n."""
        return list(self._sizes)

    def _raise(self, U: torch.Tensor, src: torch.Tensor, k: int, mode: int, B: int) -> torch.Tensor:
        """One raising step into layer k: R_mode applied to a layer-(k-1) amplitude vector [B, S_{k-1}]
        -> [B, S_k]. One gather/multiply/scatter_add_ per mode slab, no [B, E] temporary kept.
        """
        parent, child, off = self._slab_parent[k - 1], self._slab_child[k - 1], self._offsets[k - 1]
        nxt = torch.zeros((B, self._sizes[k]), dtype=self.cdtype, device=self.device)
        for j in range(self.m):
            a, b = off[j], off[j + 1]
            if b <= a:
                continue
            slab = U[:, j, mode].unsqueeze(1) * src[:, parent[a:b].long()]        # [B, e]
            nxt.scatter_add_(1, child[a:b].long().unsqueeze(0).expand(B, -1), slab)
            del slab
        return nxt

    def amplitudes(self, U: torch.Tensor) -> list[torch.Tensor]:
        """Output amplitudes per photon number: a list of n + 1 tensors [S_k] (or [B, S_k] when U is
        [B, m, m]). Layer k holds the raw (deferred-normalization) amplitudes - `probs` applies
        `layers.prob_norms(k)`.
        """
        batched = U.dim() == 3
        if not batched:
            U = U.unsqueeze(0)
        B = U.shape[0]
        U = U.to(device=self.device, dtype=self.cdtype)
        psi: list[torch.Tensor | None] = [torch.ones((B, 1), dtype=self.cdtype, device=self.device)]
        psi += [None] * self.n
        for i, mode in enumerate(self._modes):
            top = min((i + 1) * (self.d - 1), self.n)          # highest layer reachable after i+1 sources
            acc: list[torch.Tensor | None] = [None] * (self.n + 1)
            if self._nonzero[i][0]:
                for k in range(top + 1):
                    if psi[k] is not None:
                        acc[k] = psi[k] * self._coef[i, 0]
            term = psi
            for q in range(1, self.d):
                nxt: list[torch.Tensor | None] = [None] * (self.n + 1)
                for k in range(1, top + 1):
                    if term[k - 1] is not None:
                        nxt[k] = self._raise(U, term[k - 1], k, mode, B)
                term = nxt
                if not self._nonzero[i][q]:
                    continue
                for k in range(top + 1):
                    if term[k] is None:
                        continue
                    acc[k] = term[k] * self._coef[i, q] if acc[k] is None else acc[k] + term[k] * self._coef[i, q]
            del psi, term
            psi = acc
        out = [p if p is not None else torch.zeros((B, self._sizes[k]), dtype=self.cdtype, device=self.device)
               for k, p in enumerate(psi)]
        return out if batched else [p.squeeze(0) for p in out]

    def probs(self, U: torch.Tensor, loss=None, indist=None, order: int | None = None) -> LossyDistribution:
        """Output distribution as a `LossyDistribution` over 0..n photons; `probs[k]` sums to
        P(k) = `state.photon_number_probs()[k]` for a normalized state, whatever U is.

        `loss=eta` (uniform transmission < 1) thins every layer down the basis and re-sums them
        (`loss.thin_graded`); eta = 1 or None returns the lossless blocks untouched.
        `indist=x` (x-model per source, docs/DISTINGUISHABILITY.md): the mixture over the sources
        in the common internal mode, a source in a mode of its own contributing a classically
        routed Bernoulli photon (`slos.species.product_x_mixture`); `order=b` truncates at b such
        sources (`imperfections.truncation_bound` with n = sources).
        """
        from ..imperfections import spec
        if spec(indist) is not None:
            from .species import product_x_mixture
            blocks = product_x_mixture(self, U, indist, order=order)
        else:
            amps = self.amplitudes(U)
            blocks = []
            for k, a in enumerate(amps):
                p = a.real ** 2 + a.imag ** 2                      # cheaper than (conj*amp).real
                if k > 1:                                          # layers 0 and 1 can never bunch
                    p = p * self._prob_norm[k]
                blocks.append(p)
        if loss is None or (not isinstance(loss, torch.Tensor) and float(loss) >= 1.0):
            return LossyDistribution(self.layers, blocks)
        from ..imperfections import thin_graded
        return thin_graded(self.layers, blocks, loss)

    def sample(self, U: torch.Tensor, shots: int, generator: torch.Generator | None = None,
               loss=None) -> torch.Tensor:
        """Exact samples [shots, m] by drawing from the computed distribution ("strong simulation
        then sample"): correct for the coherent superposition, but it builds the whole graded
        distribution first, so it is bounded by the strong-simulation reach (n <= ~11), not by a
        sampler's. `clifford.sample` has no ProductSuperposition path yet (docs/SUPERPOSITION.md).
        """
        return self.probs(U, loss=loss).sample(shots, generator=generator)


class GroupedSLOSKernel:
    """Wraps a kernel to emit grouped probabilities via a scatter-reduce: the *returned* output is
    [B, n_groups] rather than [B, S].

    NOTE: this shrinks OUTPUT memory, not PEAK/working memory. The full [B, S] amplitude AND
    probability vectors are still built internally (`probs` below) — the group-scatter is
    post-recurrence, and the global memory peak lives inside the recurrence. A fused square-scatter
    (|amp|^2 + index_add in one pass, chunking the final vector) is a latent optimisation that would
    trim the post-recurrence transient but not the peak. See SLOScore_REPORT.md §3.4.
    """

    def __init__(self, kernel: FockSLOSKernel, groups: torch.Tensor):
        self.kernel = kernel
        self.groups = groups.to(kernel.device)
        self.n_groups = int(self.groups.max().item()) + 1 if self.groups.numel() else 0

    def output_size(self) -> int:
        """Number of groups, i.e. the width of the returned distribution."""
        return self.n_groups

    def probs(self, U: torch.Tensor) -> torch.Tensor:
        """Grouped output probabilities. U: [m, m] or [B, m, m] -> [n_groups] or [B, n_groups].

        Runs the wrapped kernel's full recurrence, applies the deferred bosonic norm in place, then scatter-adds the
        [B, S] probabilities into their groups; normalised exactly as the wrapped kernel's `probs`.
        """
        amp = self.kernel.amplitudes(U)
        p = amp.real ** 2 + amp.imag ** 2
        if not self.kernel._output_norm_is_one:
            p *= self.kernel.output_prob_norm            # in-place: saves a [B,S] (see FockSLOSKernel)
        batched = p.dim() == 2
        if not batched:
            p = p.unsqueeze(0)
        out = torch.zeros((p.shape[0], self.n_groups), dtype=p.dtype, device=p.device)
        # scatter_add_: backward saves the index view, not the [B,S] probs (see module docstring)
        out.scatter_add_(1, self.groups.unsqueeze(0).expand(p.shape[0], -1), p)
        return out if batched else out.squeeze(0)
