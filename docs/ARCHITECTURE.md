# Architecture

Maintainer-facing map of the code base: where things live, what depends on what, which
invariants the engines share, and which parts are load-bearing. The user-facing entry points are
`README.md` (what the package does) and `docs/TUTORIAL.md` (how to call it); the per-engine
derivations are in `docs/SLOS_IMPLEMENTATION.md`, `docs/ALGORITHMS.md`,
`docs/EMITTER_SAMPLER.md`, `docs/SEQUENCE.md`, `docs/TIME_LOOP.md`, `docs/HYBRID.md`, `docs/LOSS.md`,
`docs/DISTINGUISHABILITY.md`, `docs/SUPERPOSITION.md`, `docs/ADAPTIVE.md`, `docs/NONLINEAR.md`, `docs/GAUSSIAN.md`, `docs/ESTIMATE.md`,
`docs/MEMORY_FRONTIER.md` and `docs/DISTRIBUTED.md`. This file is the one to read before changing
anything that crosses a module boundary.

## 1. The organising idea

One set of input-state types feeds several engines. An *engine* takes an input state and an
`m x m` interferometer unitary and answers one of two questions: the **strong** question (give me
amplitudes or probabilities of specified outputs) or the **weak** question (give me samples). The
package exists because these two families share their basis construction, their output-space
restrictions, their loss model and their input types — and because emitter (matrix-product-state)
inputs can be carried through both at the cost of the emitter's bond dimension rather than the
`2^n` Fock components they expand to.

The consequence for a maintainer: `states.py`, `masks.py` and `imperfections.py` are shared vocabulary.
A change to any of them lands in every engine at once, and every engine has tests asserting the
shared semantics. Treat them as the package's API, not as utilities.

## 2. Layers

Reading order is bottom-up; each layer may import from the ones above it in this list and not the
other way round.

```
states.py  masks.py  imperfections.py   shared vocabulary: input types, output spaces, loss and distinguishability
        |
        +-- slos/                      strong simulation (amplitudes, probabilities, Kraus ops)
        +-- clifford/                  exact sampling (Clifford & Clifford family, GPU kernels)
        +-- timeloop/                  time-bin loop devices: unrolled unitary + native register engine
        +-- sequence.py                k emitters x n bins through a static interferometer
        +-- adaptive.py                Fock input, mid-circuit photon counting + feed-forward on the interferometer
        +-- nonlinear.py               Fock input, single-mode photon-number phases (Kerr) between interferometers
        +-- gaussian/                  squeezed (Gaussian) inputs: loop hafnians, exact PNR chain-rule sampler
                |
                +-- adapters/          bridges out: perceval, merlin
```

Spin-photon processor simulators are built downstream, on `sequence.py`, `slos.EmitterSLOSKernel`,
`slos.StageKernel` and `imperfections.py`; none is imported from here.

`photonic_engines/__init__.py` re-exports the shared vocabulary eagerly and the engine
modules (`slos`, `clifford`, `gaussian`, `timeloop`, `sequence`, `adaptive`, `nonlinear`) and `adapters` lazily through a module-level `__getattr__`. That laziness is deliberate and load
bearing: importing `photonic_engines` must not pull in `triton` or `cupy`, so a CPU-only or
torch-only consumer can use `states` and the distributed engine without a CUDA stack. The same
pattern repeats inside `clifford/__init__.py` (a `__getattr__` for `fock_sample_single`,
`sample_triton`, `exq_sample`) and in `slos/__init__.py` (a guarded `try/except ImportError`
around the torch kernels). **If you add a top-level import of an optional dependency to a core
module, the `no-optional-deps` CI job is what will catch it.**

### Known wrinkles in the layering

One place breaks the picture above. It is deliberate today and worth knowing before you refactor
around it:

* `clifford/cofactors_triton.py` and `clifford/cuda_cofactors.py` import each other. Both imports
  are function-local, so there is no import cycle at runtime — but adding a module-level import in
  either direction will create one.

`sequence.py` imports nothing from `adapters/`.

## 3. The engines

### Engine map: usage × input × interferometer × kernel (2026-10-02)

The paper organises the modalities by the **usage** of the photon train — (a) demultiplexed in space, one unitary
per shot, exponent = sources; (b) per tick, a register marched through time, exponent = register — with the
interferometer class (universal U / loops) as the orthogonal axis. This table maps every cell to the code that
runs it. "Strong" = amplitudes / probabilities / marginals; "sampler" = the exact weak side; the oracle is what its
tests compare against.

| usage | input | interferometer | strong | sampler | oracle / note |
|---|---|---|---|---|---|
| (a) demux | Fock (FBS) | universal U | `slos.FockSLOSKernel` | `clifford.fock_sample` | `slos/reference.py` (permanents) |
| (a) demux | Fock + mid-circuit counting (aFBS) | universal U, feed-forward | `adaptive.AdaptiveCircuit` (mode marginals) | `AdaptiveCircuit.sample` | one permanent per branch |
| (a) demux | Fock + Kerr sites (NLBS) | universal U layers | `nonlinear.NonlinearCircuit` (branch identity) | its rejection sampler | two oracles in `tests/test_nonlinear.py` |
| (a) demux | superposed (SBS) | universal U | `slos.ProductSLOSKernel` (graded) | `clifford.product_sample` | explicit components × permanents |
| (a) demux | squeezed (GBS) | universal U | `slos.GaussianSLOSKernel` | `gaussian.sample` | `hafnian.lhaf_brute`, truncated ProductSLOS |
| (a) demux | emitter MPS (EBS) | universal U | `slos.EmitterSLOSKernel` | `clifford.emitter_sample` | `clifford/emitter_reference.py` |
| (b) per tick | k emitters (ESS) | static universal U per tick | `sequence.SequenceKernel` (Kraus chain) | `SequenceKernel.sample_diag` | per-bin Kraus route vs label amplitudes |
| (b) per tick | Fock (LBS) | one rail, loop ladder | `timeloop.LoopRegisterKernel` (dense register) | `timeloop.hybrid.loop_sector_sample` (photon-number sectors; of record) | `TimeLoopCircuit.unitary` unrolled |
| (b) per tick | superposed (SBS-L) | one rail, loop ladder | `timeloop.hybrid.SuperpositionLoopKernel` (graded register) | its dense `sample` | `ProductSLOSKernel` on the unrolled U |
| (b) per tick | squeezed (GBS-L) | one rail, loop ladder | — | `gaussian/loop.py` (capped register, ε = 1/S) | loop hafnian |
| (b) per tick | one emitter (Hybrid row) | one rail, loop ladder | `LoopRegisterKernel(emitter=)` | `loop_sector_sample` | `EmitterSLOSKernel` on the unrolled U |
| (b) per tick | k emitters, carried photons | rails × ladders + rail mixer | `timeloop.hybrid.HybridRegisterKernel` | its `sample` / `sample_sectors` | product emitter MPS on the unrolled U |

Demux × loops is not a cell: unrolled, it is a structured U of the low-depth class. A per-tick dispatcher over
the (b) rows is not written yet.

### `slos/` — strong simulation

SLOS_full (arXiv:2206.10549): the output amplitudes are built photon by photon over a layered
Fock basis instead of by evaluating one permanent per output.

* `basis.py` builds that basis as *photon-position layers* with the output mask pruned during
  construction rather than after it — this is why a mask makes the build cheaper and not just the
  result smaller. `basis_torch.py` is the same construction on the device, selected automatically
  for CUDA kernels. Layers are cached process-wide (`get_layers` / `clear_layers_cache`); the
  cache is keyed on `(n, m, mask, device)` and is the reason repeated kernel construction is cheap
  and the reason a test that changes device must clear it.
* `kernel.py` is the amplitude recurrence as batched, device-resident, **differentiable** torch
  ops — `FockSLOSKernel` (one Fock input), `SuperpositionSLOSKernel` (a coherent
  `FockSuperposition`), `ProductSLOSKernel` (a `ProductSuperposition`: the recurrence runs over
  SOURCES and every photon-number layer is live at once, so it returns a `imperfections.LossyDistribution`
  rather than one vector — docs/SUPERPOSITION.md), `GroupedSLOSKernel`. Differentiability is a requirement, not a bonus:
  `adapters/merlin.py` and `timeloop/module.py` train through these kernels.
* `emitter.py` carries the emitter bond index through the same recurrence (`EmitterSLOSKernel`),
  and exposes `kraus()` — the per-outcome register operators that `sequence.py` consumes.
* `stage.py` is one stage of a hybrid spin-photon program (`StageKernel`; the chain is `states.StageChain`): an emitter
  chain applied on top of a batch of Fock inputs (heralded or carried photons) on one layered basis —
  the Fock prefix as `FockSLOSKernel`'s step, then the chain as `EmitterSLOSKernel`'s step — with
  ragged labels, partial detection (`detected=`, carried-out basis) and graded photon numbers; a
  single-photon chain with no Fock input forwards to `EmitterSLOSKernel` unchanged.
* `species.py` is the strong side of partial distinguishability (`docs/DISTINGUISHABILITY.md`):
  a species partition of the input on the same layered basis — the SLOS recurrence stopped on an
  intermediate layer for an interfering species, a probability-level `classical_raise` along the
  edges for a photon in a mode of its own, an explicit convolution for two multi-photon species —
  and the x-model as the weighted sum over interfering subsets (`x_mixture`, `product_x_mixture`,
  `emitter_x_mixture` with its species-resolved Kraus set). The kernels' `probs(indist=, order=)`
  dispatch there; the truncation error is `imperfections.truncation_bound`.
* `glynn.py` answers the narrow question instead of the broad one: ONE amplitude of a
  `ProductSuperposition` as a single permanent whose repeated "no emission" row collapses to a
  binomial weight, O(2^k (n-k) n) in the detected photon count, no basis at all. It shares only
  the state types with the basis machinery.
* `estimate.py` is a different answer to the strong question: a Gurvits-type *additive-error*
  estimator for `EmitterMPS` inputs at polynomial cost (`docs/ESTIMATE.md`). It is independent of
  the basis machinery and shares only the state types.
* `reference.py` is the correctness oracle: independent permanent-based amplitudes, slow by
  design. Most `slos` tests are "kernel equals reference"; do not optimise it.
* `distributed.py` + `_mpi_worker.py` are EXPERIMENTAL: SLOS_full over MPI ranks with a C library
  under cffi. They need an MPI runtime, are excluded from the coverage gate, and are exercised by
  `tests/slos/check_distributed.py`.

### `clifford/` — exact sampling

The Clifford & Clifford chain rule, with the per-step cofactors computed by GPU kernels.
`product_sampler.py` is the same chain rule for `ProductSuperposition` inputs, but marching the
TRUE marginal (so exact by construction, with no random-order argument — that one does not hold
for this input); its per-step marginal has two evaluations and it takes the cheaper
(docs/SUPERPOSITION.md §5). Plain torch, no kernels yet.
`sampler.py` is the batched entry point (`fock_sample`) and `single.py` the latency path for a
single sample. Two algorithm variants share the chain rule and differ in how they exploit
repeated rows: `"2018"` (arXiv:1706.01260, Glynn/Gray sweep) and `"2020"` (arXiv:2005.04214,
mixed-radix over multiplicities); `"auto"` picks between them by term count.

The emitter side is a separate stack of the same shape — `emitter_reference.py` (oracle),
`emitter_sampler.py` (torch), `emitter_terms.py` (chain terms and the loss unravelling),
`emitter_triton.py` / `emitter_triton_seq.py` (kernels), `emitter_caterpillar.py` (the segmented
caterpillar special case).

Kernel files come in pairs: a Triton kernel (`cofactors_triton.py`, `emitter_triton.py`) and a
CUDA/cupy kernel (`cuda_cofactors.py`, `exq_sampler.py`) with embedded C source. **Without a CUDA
device the Triton kernels still run, under the Triton CPU interpreter** — `tests/conftest.py` sets
`TRITON_INTERPRET=1` before the first `import triton`, which is why that must stay at the top of
conftest and why the interpreter-based checks are marked `slow` (~40 ms per shot). The cupy
kernels have no CPU fallback and are excluded from the coverage gate.

The other `clifford/` modules:
- `variants.py`: the T1/F2a algorithm set (C&C 2018, input-side collapse, one sweep per cofactor, …). Each is a plain-torch reference step that also returns its addend count. These are the oracles of the GPU paths, not the engine (docs/ALGORITHMS.md).
- `cofactors_input.py`: the Triton kernel for `algo="input"`, the input-side collapse (Brod–Oszmaniec, arXiv:1906.06696). It signs the distinct input columns instead of the rows, on the lane plan of `cofactors_triton.py`.
- `cuda_emitter.py`: the CUDA thread-per-lane emitter cofactor kernel (`kernel="cuda"`/`"auto"`, docs/EMITTER_SAMPLER.md §8). It belongs to the same family as `cuda_cofactors.cof_multi`; only the payload differs.
- `sbs_rso.py`: the random-source-order chain for superposed sources, i.e. Clifford & Clifford with an offset start set of n − k sources (docs/ALGORITHMS.md §5.4, §9).
- `sbs_rso_triton.py`: that chain's rectangular-cofactor step in Triton.
- `cuda_rect.py`: the same step as a CUDA thread-per-lane kernel, `cof_multi` with the payload generalised to leave-one-out elementary symmetric polynomials (validated by `bench/sbs_rso_validate.py`).

### `timeloop/` — time-bin loop devices

Two engines for the same hardware (ORCA PT-series). `circuit.py` unrolls a loop device into a
plain unitary so the generic engines can run it — correct, exponential in the photon number.
`kernel.py` is the native causal engine: the loop register is marched through time, polynomial in
the number of bins for a fixed loop length. `basis.py` is the register basis, `module.py` the
trainable torch module. When they disagree, `circuit.py` is the oracle.

`hybrid.py` (`docs/HYBRID.md`) builds on `kernel.py`'s `PeriodicNetworkKernel` without forking it:
`HybridRegisterKernel` (k emitters on a product spin applied locally, m rails each with a loop ladder, a per-tick rail
mixer as a fixed two-mode mesh, optional photon cap with leak accounting); `SuperpositionLoopKernel` (superposed
per-bin sources); `sector_sample` / `loop_sector_sample`, the exact trajectory sampler on photon-number sectors (a
trajectory's register photon number is definite, so each shot carries only its current sector; tables built on the
host on first use and cached on the kernel); `light_loop_kernel` (a loop kernel without its dense basis, for the
sector sampler only). Two hooks in `kernel.py` serve it: `n_max` (register photon cap) and the lazily built exit-row
table `_ExitRows`. The oracle is the unrolled network fed with the product emitter MPS or the superposed input.

### `sequence.py` — emitter trains

`k` emitters through `n` time bins of a static interferometer: the `k*n`-photon output as a chain
of per-bin Kraus operators on the product spin register. Strong side (`bin_marginals`, sequence
probabilities, joints) and weak side (`sample`, `sample_diag`, `label_amplitudes`) live together
because they share the register bookkeeping. `hook` and `policy` are the feed-forward entry points
(`docs/SEQUENCE.md` §10).

### `adaptive.py` — adaptive linear optics

A Fock input through U_0, then levels of photon counting on K adaptive modes, each followed by a
feed-forward unitary on the unmeasured modes (arXiv:2102.04579; notation table in `docs/ADAPTIVE.md` §1).
It sits on top of two engines and imports them lazily: `slos.FockSLOSKernel` for the strong joint (one
pass per branch) and `clifford.fock_sample` as the base sampler of the final stage (forced prefix, per-circuit prefixes and stream counts, batched over branches). Loss and partial distinguishability come from `imperfections.partition`, one instance per species. Its own
numerics is `mode_marginals` — exact photon-count marginals on a few modes as the permanent of identity +
low rank (Gurvits' second algorithm), batched over blocks by `binomial_moments_blocks` — which drives the chain over the adaptive outcomes. The oracle of the
samplers is the strong joint, itself pinned to `slos.reference.permanent` on a unitary composed by the test.

### `nonlinear.py` — non-linear boson sampling

Interferometers interleaved with single-mode photon-number phases `exp(-i f(N_x))` (arXiv:2110.13788;
docs/NONLINEAR.md). A site sees at most n photons, so it is EXACTLY a finite sum of phase shifters
(`branches`), and the circuit is a coherent sum of B linear interferometers: the strong side is one
`slos.FockSLOSKernel` pass per branch (or, when the branches multiply, the state pushed through
nearest-neighbour two-mode gates by `TwoModeNetwork`), and the sampler is a rejection from the incoherent
branch mixture drawn with `clifford.fock_sample` — the Clifford & Clifford random-order chain does NOT hold
here (docs/NONLINEAR.md §4, the second engine after `product_sampler.py` where it fails). Loss before the
sites and partial distinguishability come from `imperfections.partition`, loss after them from `thin_counts`.
Oracles: `slos/reference.py` (`nonlinear_state_amplitudes`, the dense Fock-space state vector, and
`nonlinear_branch_amplitudes`, the branch sum of permanents).

### `gaussian/` — Gaussian boson sampling

Input `states.GaussianInput` (single-mode squeezed vacua, optional displacement; indefinite, unbounded
photon number). `hafnian.py`: the loop hafnian by our pair sieve, and `lhaf_poly`, the same sieve with a loop
vector linear in a variable z, which returns every photon number of the current mode at once. `sampler.py`:
the exact PNR chain rule over conditional pure Gaussian states (heterodyne every mode, then replace the
heterodyne mode by mode), the closed-form normaliser, loss by `imperfections.lossy_squeezing`, Schmidt modes
as independent instances. `probs.py`: per-pattern probabilities of a pure input. The strong side lives in `slos/gaussian.py`
(`GaussianSLOSKernel`: the SLOS layers and edges with one ζ-raise and two raises per squeezer per layer,
exact for every lossless layer up to a global photon cutoff). `lhaf_poly` (v1) and `lhaf_poly_multi` (v2: gathered
subset buckets, loop-vector windows, collision grouping, pairing searched for the condition number) are batched
torch on any device; `lhaf_poly_small` is numpy for the chain's many sieves under 14 vertices; `sieve_triton.py`
(Triton, imported lazily) computes v2's powers, traces and paths on chip for the large sieves on CUDA
(docs/GAUSSIAN.md §3.4). Beyond float64, `dd.py` supplies double-double tensor arithmetic: error-free transformations, plus Ozaki-sliced matrix products that are exact on tensor cores. `sieve_dd_triton.py` is the Triton kernel for the sieve series in double-double (§3.4 v2.11). `cost.py` holds the frontier statistic (run-of-S cost stratified by the exact photon law);
`timing.py` replays the sieve calls of a shot from its pattern and prices them with a calibrated per-call model
(§2.2) — `plan_calls` must follow `sampler.sample_pure`'s control flow, which a test checks. Oracles: `hafnian.lhaf_brute` and
`GaussianInput.truncated(d)` through `slos.ProductSLOSKernel` (shares no code with the sampler).

### `adapters/` — bridges out

* `perceval/converter.py` — circuit-to-unitary conversion, on device and vectorized. This is the
  measured hot spot of the Perceval backend task, not the SLOS kernel. Note it builds in
  **complex64**, so it matches Perceval's complex128 unitary only to single precision; there is no
  dtype argument yet.
* `perceval/backends.py` — the Perceval backend classes. Every one of them restricts the interface
  somewhere (no masks, no amplitudes under loss, BasicState-only input); the restrictions are in
  the docstrings and asserted in `tests/adapters/test_perceval_backends.py`.
* `merlin.py` — `TimeLoopLayer`, the MerLin-style trainable layer over `timeloop/`.

## 4. Invariants that cross modules

These are the things a change is most likely to break silently. Each has a test; if you touch the
area, run that test first.

| Invariant | Guarded by |
| --- | --- |
| Sampler agrees with strong simulation (TVD within shot noise) | `tests/clifford/check_sampler.py` (slow), `tests/test_tutorial.py` §7 |
| Kernel agrees with the permanent-based reference | `tests/slos/test_kernel.py`, `tests/clifford/test_emitter_reference.py` |
| Native time-loop engine agrees with the unrolled circuit | `tests/timeloop/test_timeloop.py` |
| Hybrid register (k emitters, rails, ladders, mixer) agrees with the unrolled network fed with the product emitter MPS, lossless and with loss channels; reduces to `SequenceKernel` without loops, to `LoopRegisterKernel` without emitters; superposed loop sources agree with `ProductSLOSKernel` on the unrolled U | `tests/timeloop/test_hybrid.py` |
| The photon-number sector sampler reproduces the dense probabilities on forced outcomes (Fock, traced and projected emitter, multi-emitter) and the dense lossy marginals in law | `tests/timeloop/test_hybrid.py` |
| Loss at `eta = 1` reduces to the lossless path, exactly | `tests/test_loss.py`, `tests/slos/test_product.py` |
| Graded kernel agrees with explicit components x permanents, and its layer masses are the input's photon-number law | `tests/slos/test_product.py` |
| The single-amplitude permanent agrees with the graded kernel and with Ryser | `tests/slos/test_product.py` |
| Loss agrees with Perceval's lossy simulation | `tests/test_loss.py` |
| `indist = 1` is the lossless path, object for object; `indist = 0` is classical transport `per(\|U\|²)/∏t!` | `tests/test_imperfections.py` |
| The x-model equals the mixture over interfering subsets (Gram oracle vs species oracle, 1e-13) | `tests/test_imperfections.py` |
| Samplers with `indist` agree with the species-resolved permanent sum (TVD within shot noise); one term per stream after a row fold | `tests/test_imperfections.py` |
| Strong `probs(indist=x)` equals the Gram oracle (Fock), the coherent-good / classical-bad oracle (SBS) and the expanded chain (emitter) to 1e-13; `order=b` drops exactly `truncation_bound`; the species Kraus set is trace-preserving | `tests/test_imperfections.py` |
| Perceval `NoiseModel(indistinguishability=V)` equals `indist = sqrt(V)`; an annotated BasicState equals the species partition | `tests/test_imperfections.py` |
| Register modalities under `indist`: time-loop marginals / samples equal the x-mixture on the unrolled unitary; `stage_channel` traces equal the emitter strong side and sum to the ideal channel; a two-emitter two-bin sequence equals the species sum on U ⊕ U | `tests/test_imperfections_register.py` |
| Adaptive circuits: the joint equals one permanent per outcome on the branch unitary; mode marginals equal the SLOS distribution summed over the traced modes; both chains and the three final stages reproduce the joint (TVD within shot noise); a forced prefix of the Clifford chain gives the conditional law at acceptance P/T and its subset weights sum to p! T[p]; with loss and the x-model the sampler equals the enumerated mixture over partitions (SLOS per species, classical singletons) | `tests/test_adaptive.py` |
| Kernels are differentiable end to end | `tests/timeloop/test_timeloop.py`, `tests/adapters/test_perceval_converter.py` |
| Non-linear circuits: both oracles agree, both strong routes equal them, phi = 0 / phi = pi / an idle site reduce to the linear circuit, the rejection sampler equals the strong distribution (TVD within shot noise) at acceptance 1/\|c\|_1^2, and with loss / the x-model it equals the enumerated mixture over partitions | `tests/test_nonlinear.py` |
| Gaussian inputs: the pair sieve equals the brute-force loop hafnian; pure probabilities equal the truncated ProductSLOS oracle, Piquasso and The Walrus; the sampler equals the exact law (chi-square) lossless, lossy, displaced and with Schmidt modes; `loss = 1` draws exactly the lossless samples | `tests/test_gaussian.py` |
| The tutorial runs as written | `tests/test_tutorial.py` |
| The core imports without triton / cupy / perceval | the `no-optional-deps` CI job |

## 5. Where the weight is

By line count (2026-09-23), the files that carry most of the complexity are `timeloop/kernel.py` (~850),
`sequence.py` (~840), `slos/distributed.py` (~670), `adaptive.py` (~660), `slos/kernel.py` (~610),
`clifford/emitter_triton.py` (~590), `imperfections.py` (~590), `nonlinear.py` (~550) and
`clifford/cofactors_triton.py` (~550). They are also the files where the compact,
column-aligned style matters most: the shape comments on the right of each line are the only
record of the tensor contract, and the ruff formatter is deliberately not run over them
(see `docs/CONTRIBUTING.md`).

## 6. Things outside the package

* `bench/` — measurement scripts, one per question, each writing a CSV to `results/`. Not
  imported by the library and not covered; linted with a relaxed rule set.
* `scripts/` — setup and build scripts for the external baselines (`orca_setup.sh`, `gbskernels_build.sh`, ...).
  Must be committed with the execute bit set.
* `reproduce/` — one script per published table and figure.
* `tests/clifford/check_*.py` and `tests/slos/check_distributed.py` — runnable scripts, not
  collected by pytest (`python_files = ["test_*.py"]`). `tests/clifford/test_scripts.py` is the
  pytest wrapper that runs them as subprocesses.
* `docs/paper/`, `docs/slides/` — the paper and deck sources; the compiled PDFs are build output,
  gitignored and not tracked.
* `experiments/`, `results/` — gitignored working output.
