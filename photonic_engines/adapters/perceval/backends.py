"""Perceval backend wrappers around the engines.

  SLOSTorchBackend      AStrongSimulationBackend on `slos.FockSLOSKernel` (full Fock space, one
                        Fock input; batched device kernel, one launch for the whole output
                        distribution). Perceval masks are not supported (set_mask raises).
  EmitterSLOSBackend    the same for an `EmitterMPS` input (`set_emitter_state`), spin traced or
                        projected; `prob_amplitude` only for a projected spin.
  CliffordGPUBackend    ASamplingBackend on `clifford.fock_sample` (Triton/CUDA kernels;
                        needs a CUDA device).

Uniform loss: every backend takes `loss=eta` (end-to-end transmission, docs/LOSS.md). A
lossy strong backend answers `probability(t)` and `prob_distribution()` over 0..n detected
photons and `samples()`; `all_prob` / `evolve` / `prob_amplitude` are undefined then (Perceval's
own Simulator/Processor loss handling — LC components, Source losses — is the alternative when
the loss must live inside a Processor).

Conventions: `set_circuit` accepts any Perceval circuit (unitary via `compute_unitary()`),
amplitudes follow Perceval's normalization (<t|U|s> = per / sqrt(prod s_i! prod t_j!)), and
`all_prob` / `prob_distribution` / `evolve` return states in Perceval's `allstate_iterator`
order, mapped from the kernel's basis once per (n, m).
"""
from __future__ import annotations

import numpy as np
import torch

from ...states import EmitterMPS, FockSuperposition


def _import_perceval():
    import perceval as pcvl
    from perceval.backends import ASamplingBackend, AStrongSimulationBackend
    from perceval.utils import BasicState, BSDistribution, BSSamples, StateVector, allstate_iterator
    return pcvl, AStrongSimulationBackend, ASamplingBackend, BasicState, BSDistribution, BSSamples, StateVector, allstate_iterator


pcvl, AStrongSimulationBackend, ASamplingBackend, BasicState, BSDistribution, BSSamples, StateVector, allstate_iterator = _import_perceval()


def _annotation_species(input_state):
    """Species partition of an annotated BasicState: photons with the same annotation string
    interfere, photons of different annotations do not; unannotated photons ('' / 'None') form
    one species of their own. Returns (groups, singles) as lists of input columns
    (`slos.species.species_probs`), or None when nothing is annotated.
    """
    if not getattr(input_state, "has_annotations", False):
        return None
    cols = [j for j, c in enumerate(input_state) for _ in range(int(c))]
    by_tag: dict = {}
    for k, col in enumerate(cols):
        tag = str(input_state.get_photon_annotation(k))
        by_tag.setdefault(tag if tag not in ("", "None") else None, []).append(col)
    groups = [g for g in by_tag.values() if len(g) >= 2]
    singles = [g[0] for g in by_tag.values() if len(g) == 1]
    return groups, singles


class _TorchStrongBase(AStrongSimulationBackend):
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.complex64, loss: float | None = None,
                 indist: float | None = None, order: int | None = None):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.loss = None if loss is None or float(loss) >= 1.0 else float(loss)
        from ...imperfections import spec
        self.indist = spec(indist)                              # x-model (docs/DISTINGUISHABILITY.md), None when ideal
        self.order = order                                      # strong-side truncation (imperfections.truncation_bound)
        self._species = None                                    # species partition of an annotated input
        self._lossy: dict[tuple[int, ...], float] | None = None   # pattern -> prob, 0..n photons (loss set)
        self._U: torch.Tensor | None = None
        self._kernels: dict[int, object] = {}          # photon number -> kernel (a StateVector may mix n)
        self._amps: dict[int, torch.Tensor] | None = None   # cached amplitude vectors per block (kernel order)
        self._probs: dict[int, torch.Tensor] | None = None
        self._index = None

    def set_circuit(self, circuit):
        super().set_circuit(circuit)
        self._U = torch.as_tensor(np.asarray(self._umat), dtype=self.dtype, device=self.device)
        self._invalidate()

    def set_mask(self, *a, **k):
        raise NotImplementedError("masks: use the engine's mask= argument directly")

    def _invalidate(self):
        self._amps = self._probs = self._lossy = None

    def _thin(self):
        """Fill `_lossy` from the lossless per-block probabilities (called by _ensure when loss is set)."""
        from ...imperfections import thin_distribution
        self._lossy = {}
        for n, k in self._kernels.items():
            d = thin_distribution(k.layers, self._probs[n], self.loss).as_dict()
            for t, p in d.items():
                self._lossy[t] = self._lossy.get(t, 0.0) + p

    def _get_iterator(self, input_state):
        # Perceval's cache is keyed on `input_state.n`, a set for a StateVector with mixed n
        key = frozenset(input_state.n) if isinstance(input_state.n, set) else input_state.n
        if key not in self._cache_iterator:
            self._cache_iterator[key] = tuple(allstate_iterator(input_state, self._mask))
        return self._cache_iterator[key]

    def _ensure(self):
        raise NotImplementedError

    def _idx(self, output_state):
        """(photon-number block, index in that block's final basis)."""
        if self._index is None:
            self._index = {n: {t: i for i, t in enumerate(k.layers.final_states)} for n, k in self._kernels.items()}
        t = tuple(output_state)
        return sum(t), self._index[sum(t)][t]

    def _a(self, output_state) -> complex:
        n, i = self._idx(output_state)
        a = self._amps.get(n)
        return 0j if a is None else complex(a[i])

    def _p(self, output_state) -> float:
        if self._lossy is not None:
            return self._lossy.get(tuple(output_state), 0.0)
        n, i = self._idx(output_state)
        p = self._probs.get(n)
        return 0.0 if p is None else float(p[i])

    # Perceval API
    def prob_amplitude(self, output_state) -> complex:
        self._ensure()
        if self.loss is not None:
            raise NotImplementedError("amplitudes are undefined with loss (mixed state)")
        if self.indist is not None or self._species is not None:
            raise NotImplementedError("amplitudes are undefined with partial distinguishability (mixed state)")
        if self._amps is None:
            raise NotImplementedError("amplitudes are undefined for a mixed (traced-spin) state")
        return self._a(output_state)

    def probability(self, output_state) -> float:
        self._ensure()
        return self._p(output_state)

    def all_prob(self, input_state=None):
        if input_state is not None:
            self.set_input_state(input_state)
        self._ensure()
        if self.loss is not None:
            raise NotImplementedError("all_prob runs over the n-photon states only; with loss use prob_distribution()")
        return [self._p(s) for s in self._get_iterator(self._input_state)]

    def prob_distribution(self):
        self._ensure()
        bsd = BSDistribution()
        if self._lossy is not None:
            for t, p in self._lossy.items():
                bsd.add(BasicState(list(t)), p)
            return bsd
        for s in self._get_iterator(self._input_state):
            bsd.add(s, self._p(s))
        return bsd

    def samples(self, count: int, generator: torch.Generator | None = None):
        """BSSamples drawn from the computed distribution (strong simulation with sampling for
        free; lossy when `loss` is set).
        """
        self._ensure()
        bsd = self.prob_distribution()
        states, probs = list(bsd.keys()), torch.tensor(list(bsd.values()), dtype=torch.float64)
        idx = torch.multinomial(probs / probs.sum(), int(count), replacement=True, generator=generator)
        out = BSSamples()
        for i in idx.tolist():
            out.append(states[i])
        return out

    def evolve(self):
        self._ensure()
        if self.loss is not None:
            raise NotImplementedError("evolve needs a pure output (no loss)")
        if self.indist is not None or self._species is not None:
            raise NotImplementedError("evolve needs a pure output (no partial distinguishability)")
        if self._amps is None:
            raise NotImplementedError("evolve needs a pure input (projected spin)")
        res = StateVector()
        for s in self._get_iterator(self._input_state):
            res += s * self._a(s)
        return res


class SLOSTorchBackend(_TorchStrongBase):
    """Fock input (BasicState), coherent superposition (StateVector, one SuperpositionSLOSKernel per
    photon-number block) or FockSuperposition. An ANNOTATED BasicState (`|{_:0},{_:1},...>`, what
    Perceval's `Source` produces) is a species partition: photons with equal annotations
    interfere, others do not (`slos.species`). `indist=x` (constructor) applies the x-model to an
    unannotated BasicState; `order=` truncates it (docs/DISTINGUISHABILITY.md). Both give a mixed
    output: probabilities only.
    """
    name = "SLOSTorch"

    def set_input_state(self, input_state):
        """Set the input and build one SLOS kernel per photon-number block.

        Accepts a BasicState (one FockSLOSKernel), a StateVector or FockSuperposition (converted to a StateVector,
        then one SuperpositionSLOSKernel per n-block with the block amplitudes as weights). Invalidates cached
        amplitudes, probabilities and the basis index.
        """
        from ...slos.kernel import FockSLOSKernel, SuperpositionSLOSKernel
        if isinstance(input_state, FockSuperposition):
            input_state = input_state.to_perceval()
        super().set_input_state(input_state)
        self._kernels = {}
        self._species = None
        if isinstance(input_state, StateVector):
            if self.indist is not None:
                raise NotImplementedError("partial distinguishability of a coherent superposition: no per-photon identity")
            for blk in FockSuperposition.from_perceval(input_state):
                self._kernels[blk.n] = SuperpositionSLOSKernel(blk.n, blk.m, blk.occupations, device=self.device,
                                                            dtype=self.dtype, weights=blk.amplitudes)
        else:
            occ = [int(input_state[i]) for i in range(input_state.m)]
            self._species = _annotation_species(input_state)
            if self._species is not None and self.indist is not None:
                raise NotImplementedError("annotated input and indist= together: annotate, or use the x-model, not both")
            self._kernels[input_state.n] = FockSLOSKernel(input_state.n, input_state.m, occ, device=self.device, dtype=self.dtype)
        self._index = None
        self._invalidate()

    def _ensure(self):
        if self._probs is not None:
            return
        assert self._U is not None and self._kernels, "circuit and input state must be set"
        self._amps, self._probs = {}, {}
        with torch.no_grad():
            for n, k in self._kernels.items():
                if self._species is not None:                                 # annotated input: species partition
                    from ...slos.species import species_probs
                    self._probs[n] = species_probs(k, self._U, *self._species).to(torch.float64).cpu()
                    self._amps = None
                elif self.indist is not None:                                 # x-model mixture
                    self._probs[n] = k.probs(self._U, indist=self.indist, order=self.order).to(torch.float64).cpu()
                    self._amps = None
                else:
                    raw = k.amplitudes(self._U)                               # raw * input_norm (* weights)
                    tnorm = torch.as_tensor(k.layers.output_prob_norms()).to(raw.device, torch.float64)
                    self._amps[n] = (raw.to(torch.complex128) * torch.sqrt(tnorm)).cpu()
                    self._probs[n] = self._amps[n].abs() ** 2
        if self.loss is not None:
            self._thin()


class EmitterSLOSBackend(_TorchStrongBase):
    """Strong-simulation backend over an `EmitterMPS` input, driven by `set_emitter_state` instead of a Fock state.

    With a projected spin (`state.final is not None`) amplitudes are available; with a traced spin the output is
    mixed, only probabilities are computed and `prob_amplitude` / `evolve` raise.
    """

    name = "EmitterSLOS"

    def set_input_state(self, input_state):
        """Reject the Perceval Fock-input entry point; always raises NotImplementedError (use `set_emitter_state`)."""
        raise NotImplementedError("use set_emitter_state(EmitterMPS)")

    def set_emitter_state(self, state: EmitterMPS):
        """Set the emitter input and build the `EmitterSLOSKernel`; the circuit must already be set and match state.m.

        Also records a placeholder BasicState with the same (n, m) as `_input_state`, purely to drive Perceval's
        output-state iterator. Invalidates the caches and the basis index.
        """
        assert self._circuit is not None, "Circuit must be set before the input state"
        assert self._circuit.m == state.m, f"Circuit({self._circuit.m}) and state({state.m}) size mismatch"
        from ...slos.emitter import EmitterSLOSKernel
        self._state = state
        # a placeholder Fock state with the right (n, m) drives Perceval's output iterator
        occ = [0] * state.m
        for k in range(state.n):
            occ[state.modes[k][0]] += 1
        self._input_state = BasicState(occ)
        self._kernels = {state.n: EmitterSLOSKernel(state, device=self.device, dtype=self.dtype)}
        self._index = None
        self._invalidate()

    def _ensure(self):
        if self._probs is not None:
            return
        assert self._U is not None and self._kernels, "circuit and emitter state must be set"
        n, k = next(iter(self._kernels.items()))
        with torch.no_grad():
            tnorm = torch.as_tensor(k.layers.output_prob_norms()).to(self._U.device, torch.float64)
            if self.indist is not None:
                self._amps = None
                self._probs = {n: k.probs(self._U, indist=self.indist, order=self.order).to(torch.float64).cpu()}
            elif self._state.final is not None:
                raw = k.amplitudes(self._U)
                self._amps = {n: (raw.to(torch.complex128) * torch.sqrt(tnorm)).cpu()}
                self._probs = {n: self._amps[n].abs() ** 2}
            else:
                self._amps = None
                self._probs = {n: k.probs(self._U).to(torch.float64).cpu()}
        if self.loss is not None:
            self._thin()


class CliffordGPUBackend(ASamplingBackend):
    """Perceval sampling backend on `clifford.fock_sample` (Clifford & Clifford on Triton/CUDA kernels).

    Sampling only: no probability or amplitude queries. `backend="auto"` picks "cuda" when cupy imports, else
    "triton"; both need a CUDA device, and the unitary is held as complex128 on `device`. Uniform loss `eta` and
    partial distinguishability `indist=x` are passed through to the sampler.
    """

    name = "CliffordGPU"

    def __init__(self, device: str = "cuda", backend: str = "auto", seed: int | None = None,
                 loss: float | None = None, indist: float | None = None):
        super().__init__()
        self.device = device
        self.loss = loss
        self.indist = indist                                    # x-model, passed to the sampler
        if backend == "auto":
            try:
                import cupy  # noqa: F401
                backend = "cuda"
            except Exception:
                backend = "triton"
        self.backend = backend
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(seed)
        self._U = None

    def set_circuit(self, circuit):
        """Set the circuit and stage its unitary as a complex128 tensor on `self.device`."""
        super().set_circuit(circuit)
        self._U = torch.as_tensor(np.asarray(self._umat), dtype=torch.complex128, device=self.device)

    def samples(self, count: int):
        """Draw `count` samples as a BSSamples via `fock_sample`, using the backend's own generator and loss."""
        from ...clifford import fock_sample
        in_modes = [j for j, c in enumerate(self._input_state) for _ in range(int(c))]
        occ = fock_sample(self._U, in_modes, count, generator=self._gen, backend=self.backend,
                          loss=self.loss, indist=self.indist).cpu().tolist()
        out = BSSamples()
        for row in occ:
            out.append(BasicState(row))
        return out

    def sample(self):
        """Return a single BasicState, drawn as a one-shot `samples(1)` call."""
        return self.samples(1)[0]


class DistributedSLOSBackend(AStrongSimulationBackend):
    """EXPERIMENTAL. Perceval strong-simulation backend on the distributed CPU engine
    (photonic_engines.slos.distributed): `workers=P` MPI processes are spawned once and hold
    the amplitude vector; BasicState inputs only. `probability` / `prob_amplitude` are point
    lookups on the workers, `prob_distribution` gathers the whole vector (raises above
    `max_gather_bytes`), `samples` draws from the distributed vector without gathering it.
    """

    def __init__(self, workers: int = 2, max_gather_bytes: int = 4 << 30, loss: float | None = None):
        super().__init__()
        self.workers, self.max_gather_bytes = int(workers), int(max_gather_bytes)
        self.loss = None if loss is None or float(loss) >= 1.0 else float(loss)
        self._lossy = None
        self._U = None
        self._kernel = None
        self._vec = None
        self._key = None

    @property
    def name(self) -> str:
        """Return the backend name reported to Perceval ("DistributedSLOS")."""
        return "DistributedSLOS"

    def set_circuit(self, circuit):
        """Set the circuit, store its unitary as a complex128 numpy array and drop the distributed vector."""
        super().set_circuit(circuit)
        self._U = np.asarray(self._umat, dtype=np.complex128)
        self._vec = None

    def set_input_state(self, input_state):
        """Set the input state and drop the distributed vector; only a BasicState is accepted (StateVector raises)."""
        if not isinstance(input_state, BasicState):
            raise NotImplementedError("DistributedSLOSBackend takes a BasicState input")
        super().set_input_state(input_state)
        self._vec = None

    def set_mask(self, *a, **k):
        """Reject Perceval output masks; always raises NotImplementedError for the distributed engine."""
        raise NotImplementedError("masks are not supported by the distributed engine")

    def _ensure(self):
        if self._vec is not None:
            return
        from ...slos.distributed import DistributedSLOSKernel
        occ = list(self._input_state)
        key = (tuple(occ), self.workers)
        if self._kernel is None or self._key != key:
            if self._kernel is not None:
                self._kernel.close()
            self._kernel = DistributedSLOSKernel(sum(occ), len(occ), occ, workers=self.workers)
            self._key = key
        self._vec = self._kernel.amplitudes(self._U)
        self._lossy = None
        if self.loss is not None:                 # gathered, then thinned (output side)
            from ...imperfections import thin_patterns
            states = self._iter()
            probs = torch.tensor(self._vec.probabilities_of([list(s) for s in states]), dtype=torch.float64)
            self._lossy = thin_patterns(np.asarray([list(s) for s in states]), probs, self.loss).as_dict()

    def prob_amplitude(self, output_state) -> complex:
        """Return <output_state|U|input> as a point lookup on the workers (no gather).

        Raises NotImplementedError when loss is set: the output is then a mixed state and has no amplitude
        (same guard as the torch backends).
        """
        if self.loss is not None:
            raise NotImplementedError("amplitudes are undefined with loss (mixed state)")
        self._ensure()
        return self._vec.amplitude(list(output_state))

    def probability(self, output_state) -> float:
        """Return the probability of `output_state`: a point lookup on the workers, or the thinned value when loss is set."""
        self._ensure()
        if self._lossy is not None:
            return self._lossy.get(tuple(output_state), 0.0)
        return self._vec.probability(list(output_state))

    def all_prob(self, input_state=None):
        """Return the probabilities of every n-photon output state in `allstate_iterator` order, as a list of floats.

        Optionally sets `input_state` first. Raises NotImplementedError when loss is set, since the lossy support
        spans 0..n photons; use `prob_distribution()` there.
        """
        if input_state is not None:
            self.set_input_state(input_state)
        self._ensure()
        if self._lossy is not None:
            raise NotImplementedError("all_prob runs over the n-photon states only; with loss use prob_distribution()")
        return [float(p) for p in self._vec.probabilities_of([list(s) for s in self._iter()])]

    def _iter(self):
        return tuple(allstate_iterator(self._input_state, None))

    def _kernel_output_size(self) -> int:
        """Number of n-photon output states, C(n + m - 1, n), without enumerating them."""
        from ...masks import Mask, state_space_size
        occ = list(self._input_state)
        return state_space_size(sum(occ), len(occ), Mask.FOCK.value)

    def prob_distribution(self):
        """Return a BSDistribution over the whole output basis, gathered from the workers.

        Without loss this covers the n-photon states in `allstate_iterator` order; with loss it is the thinned
        distribution over 0..n detected photons. Raises MemoryError when the gather would exceed
        `max_gather_bytes` (16 bytes per complex128 amplitude).
        """
        need = 16 * self._kernel_output_size()
        if need > self.max_gather_bytes:
            raise MemoryError(f"gathering the output vector needs {need / 2 ** 30:.1f} GiB, above max_gather_bytes "
                              f"({self.max_gather_bytes / 2 ** 30:.1f} GiB); raise it or use probability()/samples()")
        self._ensure()
        bsd = BSDistribution()
        if self._lossy is not None:
            for t, p in self._lossy.items():
                bsd.add(BasicState(list(t)), p)
            return bsd
        states = self._iter()
        probs = self._vec.probabilities_of([list(s) for s in states])
        for s, p in zip(states, probs):
            bsd.add(s, float(p))
        return bsd

    def evolve(self):
        """Return the output StateVector, gathering all n-photon amplitudes from the workers.

        Raises NotImplementedError when loss is set: the output is then mixed and has no state vector
        (same guard as the torch backends).
        """
        if self.loss is not None:
            raise NotImplementedError("evolve needs a pure output (no loss)")
        self._ensure()
        states = self._iter()
        amps = self._vec.amplitudes_of([list(s) for s in states])
        res = StateVector()
        for s, a in zip(states, amps):
            res += s * complex(a)
        return res

    def samples(self, count: int, seed: int = 0):
        """BSSamples drawn exactly from the distributed vector (never gathered)."""
        self._ensure()
        rows = np.asarray(self._vec.sample(int(count), seed=seed, loss=self.loss), dtype=np.int64)
        out = BSSamples()
        for row in rows.tolist():
            out.append(BasicState([int(x) for x in row]))
        return out

    def close(self):
        """Shut down the MPI worker pool, if one was started; idempotent."""
        if self._kernel is not None:
            self._kernel.close()
            self._kernel = None


class DistributedSLOSSampler(ASamplingBackend):
    """EXPERIMENTAL. Perceval sampling backend drawing exact samples from the distributed strong simulation
    (`DistributedVector.sample`: the vector stays on the workers). Same pool as
    DistributedSLOSBackend; pass an existing one to share the workers.
    """
    name = "DistributedSLOSSampler"

    def __init__(self, workers: int = 2, strong: DistributedSLOSBackend | None = None, seed: int = 0):
        super().__init__()
        self._strong = strong if strong is not None else DistributedSLOSBackend(workers=workers)
        self._seed = int(seed)

    def set_circuit(self, circuit):
        """Set the circuit on this backend and forward it to the wrapped strong backend."""
        super().set_circuit(circuit)
        self._strong.set_circuit(circuit)

    def set_input_state(self, input_state):
        """Set the input state and forward it to the wrapped strong backend (BasicState only, as it requires)."""
        super().set_input_state(input_state)
        self._strong.set_input_state(input_state)

    def samples(self, count: int):
        """Return `count` exact samples as a BSSamples, incrementing the seed so successive calls differ."""
        self._seed += 1
        return self._strong.samples(int(count), seed=self._seed)

    def sample(self):
        """Return a single BasicState, drawn as a one-shot `samples(1)` call."""
        return self.samples(1)[0]

    def close(self):
        """Shut down the wrapped strong backend's worker pool."""
        self._strong.close()
