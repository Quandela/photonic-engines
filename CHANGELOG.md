# Changelog

## 0.2.2 — 2026-11

No change to the library's behaviour: fixed-seed outputs are those of 0.2.1.

### Tests (contributed in pull request #1)
- On a host with a CUDA device, the tests that force the Triton kernels now run on the device (before, they built CPU
  tensors that a compiled Triton kernel cannot take, and failed); the tests that need cupy are skipped without it.
- The GPU validation scripts and the 10-minute `SequenceKernel` partial-distinguishability test are marked `slow`, so
  the default run no longer includes them; `pytest --runslow` runs them (`--runslow -m "gpu and slow"` on a GPU host).

### Benchmarks
- New: `bench/snowman_check.py`, the nested-loop (snowman) check of the delay-line engine, with its test
  `tests/timeloop/test_snowman.py`; `bench/sbs_unequal_check.py`, an exact-enumeration check of the unequal-source
  superposed-source chain.
- `bench/sbs_rso_validate.py --lossy-only` (superposed sources against lossy single photons on the same instance);
  figure scripts relabelled (`bench/adaptive_toll.py`, `bench/plot_algorithm_envelope.py`), same data.
- `docs/NONLINEAR.md`: B is the denominator of φ/π (was written φ/2π).

### CI
- `actions/checkout@v5`, `actions/setup-python@v6`, `ubuntu-24.04` runners.
- The test job writes a coverage table on its run summary; a coverage badge is published from `main`.

### Documentation
- README: install line for the `cuda` / `perceval` extras, which GPU path needs which CUDA version, the `slow` and
  `gpu` test tiers (also in `docs/CONTRIBUTING.md`); CI, PyPI and coverage badges.
- README citations: author list of the entries corrected; entry for the estimator note added.

## 0.2.1 — 2026-10

### Fixed
- `slos.StageKernel`, partial detection with carry-over on graded chains (photon-less or multiphoton labels): when
  carried-in inputs of different photon numbers reach the same output pattern, **0.2.0 listed that carried state twice**
  in `StageCarry.basis` (two rows of the block operator, zero in each other's input columns). The arithmetic of `apply`
  and the total probability were right, but a caller that maps rows to carried states by occupation and keeps one row
  lost probability. Each distinct carried state now has one row. Single-photon chains are unchanged (fixed-seed outputs
  bit-identical).

### Removed
- `results/keep/k1_fingerprint_main_fc6afe6_mac.pt`, an internal test fixture shipped by mistake in 0.2.0.

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
