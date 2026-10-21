# Changelog

## 0.2.0 — 2026-10

Two additions to the emitter engines. Existing results are unchanged: fixed-seed outputs of the existing samplers and
kernels are bit-identical to 0.1.0 on CPU and agree with it to rounding on GPU (CUDA atomics are not bit-reproducible
from run to run, in 0.1.0 either), and the speed of the existing paths was checked against 0.1.0 on CPU and GPU.

### Internal photon modes in `SequenceKernel` (`docs/SEQUENCE.md` §14)
- A photon may carry an internal mode (a temporal wavepacket shape, a frequency) on which the interferometer acts as
  the identity and which the detectors do not resolve: `ModeBinEmitter(ops [n, L, K, d, d], modes, s0, final, lost)`,
  built with `bin_emitter(...)`. `SequenceKernel(...)` returns a `ModeSequenceKernel` when some emitter has internal
  modes; kernels without them are unchanged.
- The channel of a bin and the register update: `kraus`, `coarse`, `outcome_law`, `condition`, with three routes —
  F (fine Kraus set on the populated internal span), P′ (pairing, no Kraus set, differentiable in U) and R (the
  compressed per-outcome set at a cost set by the Choi rank, without the fine set); `route="auto"` picks by a
  calibrated cost model.
- Coarse outcome classes: weighted classes in `condition(..., weights=)` (e.g. detector-noise predecessors) and
  `kraus(..., classes=)` / `coarse(..., classes=)`.
- A register larger than the kernel: `condition(..., rest=X)` for spins entangled with the emitting ones.
- Imperfections: emissions that never reach the network (`lost`), unobserved emission outcomes (`ops [n, L, K, R, d,
  d]`, R > 1), and partial distinguishability (`indist`, the x-model).
- The weak route `sample_apply`: one tick of pure-state trajectories for a batch of streams without a density matrix
  (an operator-valued Clifford–Clifford chain), CPU and CUDA.

### Stage engine (`slos.StageKernel`, `states.StageChain`)
- One stage of a hybrid spin-photon program: an emitter chain applied on top of a batch of Fock inputs (heralded or
  carried photons), on one layered SLOS basis.
- Ragged labels and labels of any photon number (including photon-less labels), Fock inputs, and partial detection
  with carry-over (`detected=`: the detected patterns with the carried-out basis of each).
- A single-photon chain with no Fock input forwards to `EmitterSLOSKernel` unchanged.

## 0.1.0 — 2026-10

First public release.
