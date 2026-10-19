# Partial distinguishability

Design note for the second imperfection axis next to `docs/LOSS.md`. `photonic_engines/imperfections.py` (which absorbed `loss.py`: one module for both
imperfection axes, user decision 2026-09-20),
`indist=` on `fock_sample` and `emitter_sample` (phase 1 of §7) and on the strong sides
`FockSLOSKernel` / `ProductSLOSKernel` / `EmitterSLOSKernel.probs` (+ `kraus_species`) and the
Perceval backends (phase 2, `slos/species.py`), and on the register modalities:
`SequenceKernel(indist=)`,
`LoopRegisterKernel.sample / bin_marginals(indist=)` (phase 3, 2026-09-20). `clifford.product_sample(indist=)` and
`SequenceKernel.sample_diag` under `indist` followed on 2026-09-21. Not done: the estimator's
`indist=`, a deterministic inter-emitter Gram (§1.2, detuning between dots),
random species on the strong sides and emitter samplers.
Tests: `tests/test_imperfections.py`. Tutorial §9d. No GPU number has been measured yet: the
frontier table of §5 is the prediction, `bench/frontier_indist.py` is the run that checks it.

## 0. Summary

- **Model.** The x-model (every pair of photons has internal-state overlap `x`) and the
  "orthogonal bad photon" model (each photon is, independently with probability `1−x`, in an
  internal mode orthogonal to everything else) give **the same output distribution for every
  interferometer and every input** — a permutation with `j` moved elements weighs `x^j` in both.
  So the model Perceval's `Source(indistinguishability=…)` implements is exactly simulable by
  drawing the "bad" set per shot, and the bound-error truncation of Renema (PRL 120, 220502) is
  not needed for an *exact* engine. `x = sqrt(V_HOM)`.
- **One primitive.** Every engine already has the mechanism: the input-side split of
  `loss.split_shots` (which photons enter a smaller instance) plus, for emitter inputs, the
  slot folding of `EmitterMPS.fold`. A bad photon is a **single-photon species**: it
  never enters a permanent with the others, but it keeps its own coherence (its label
  superposition on emitter inputs, its path superposition in a loop network). It is *not* a
  lost photon and *not* a label measurement; the mechanism is a **row fold**
  `G_k(r) = Σ_l U[r, modes[k][l]] M_{k,l}` once its output row `r` is drawn, which reduces to
  classical routing `|U[r, i]|²` for Fock inputs.
- **Cost law.** Sampling: a shot with `g` good photons costs a `g`-photon lossless sample,
  `g ~ Bin(n, ηx)` with loss, so the frontier moves as `n*(η, x) = n*(1) log ρ / log(1 − ηx + ηxρ)`
  — the E4 law with `η → ηx`. At `V_HOM = 0.9` the Fock frontier moves by +1.6 emitted photons at
  `η = 1`, the emitter frontier by +1.2. Exact, same kernels, no approximation. Strong simulation
  pays a subset sum (`Σ_b C(n, b)` runs of `(n − b)`-photon instances, species-resolved output for
  emitter inputs) — exact, with an optional truncation in the number of bad photons whose error
  is the binomial tail `P(Bin(n, 1 − x) > b_max)`, `U`-independent.
- **Modalities.** Fock spatial: trivial (routing). Emitter chain: row fold, fits the label-unravelling
  code path. SBS: bad sources factor out as Bernoulli classical photons. Sequences (k emitters × n
  bins): only *inter-emitter* distinguishability matters, and it is a deterministic Gram matrix
  between dots rather than a random bad set — needs species-resolved Kraus operators (one-off cost
  per bin). Time loops: a bad photon is a one-photon register evolution through the same gates.
  Perceval: annotated `BasicState`s are species partitions, the same primitive.
- **Plan.** Three phases (§7): the primitive + Fock + emitter samplers (done), then strong sides +
  Perceval cross-checks, then sequence / time-loop. Revisit the paper decision "partial distinguishability out of
  scope" *after* phase 1: the result is one formula and one figure, and it removes the caveat that
  the comparison with Soret et al. ignores distinguishability (their metric M = ideal-equivalent photons is `ηx·n` here).

## 1. Model

### 1.1 Definitions

Photon `i` carries an internal state (spectro-temporal profile) `|φ_i⟩` on top of its mode.
For pure internal states the output probability is (Tichy PRA 91, 022316; Shchesnovich PRA 91,
013844)

    P(t) = (1/∏ t_j!) Σ_{σ,τ ∈ S_n} ∏_i S_{σ(i) τ(i)} U[r_i, d_{σ(i)}] U*[r_i, d_{τ(i)}],   S_ij = ⟨φ_i|φ_j⟩,

`(n!)²` terms in general; for i.i.d. *mixed* internal states `ρ_int` the weight of a permutation
is `∏_{cycles c} Tr(ρ_int^{|c|})`. Two special cases are what hardware calibration gives:

| model | S | equivalent random-species picture | where it comes from |
| --- | --- | --- | --- |
| **x-model** | `S = x J + (1−x) I` (pure, `√x|0⟩ + √(1−x)|i⟩`) | each photon good w.p. `x`, else its own orthogonal mode (OBB) | HOM visibility `V = x²`; Perceval `Source`; Renema 2018 |
| **random species** | i.i.d. mixed `ρ_int` with eigenvalues `q_μ` | each photon draws species `μ ~ q`, photons of one species interfere | memoryless emitter (pure dephasing: `V = T₂/2T₁`), thermal spectral diffusion fast w.r.t. the train |

The equivalence in the first row is exact: for a permutation moving `j` elements the x-model
gives `x^j` and the OBB gives `P(all j moved photons good) = x^j`; the second row is the same
statement with `Σ_μ q_μ^{|c|} = Tr ρ^{|c|}`. HOM check: `P(1,1) = (1 − x²)/2` in both.

### 1.2 What the model says about sources

- **Single quantum dot, one train** (EmitterMPS, caterpillar): pure dephasing gives i.i.d.
  internal states → random species, x-model at first order. The internal state is independent of
  the emitted *label* (R/L), so it factorises from the spin–label MPS — the OBB argument goes
  through unchanged with the register in place.
- **Several dots in one bin** (SequenceKernel): the overlap between two dots is a *fixed* number
  `s_ab` (detuning), not a random event — a deterministic Gram matrix between emitters, on top of
  each dot's own purity `x_a`. Photons of the same dot never meet (different bins), so
  *within-dot* distinguishability is invisible in this modality. (It reappears in "trains through
  loops", next-candidate 3.)
- **One demultiplexed source into a loop** (LoopRegisterKernel): x-model.
- **SBS** (ProductSuperposition): per-source purity; Renema 1911.10112 is exactly this model.

### 1.3 Out of scope (this note)

Time-correlated Gram matrices (`S_ij = f(|i − j|)`, slow spectral wandering) — no species
decomposition; only the `(n!)²` oracle or a per-shot pure-Gram sampler (Cholesky species basis,
`k^k`-type assignment sum, fine as an oracle at `n ≤ 7`). Label-dependent internal states
(fine-structure splitting: R and L photons of one dot partly distinguishable) — a species per
*input mode*, which the primitive supports as a deterministic partition but which needs a Gram
(not OBB) treatment when the overlap is partial. Multi-photon emission (`g²`): a different axis
(a second photon in the same mode, distinguishable or not), not covered. Dark counts: output-side
Poisson counts, a separate layer (`imperfections.py`, DARK COUNTS).

## 2. The primitive

Everything below instantiates one object, the **species partition** of a shot:

    partition = (lost slots, {species μ: slots}, singleton bad slots)

drawn per stream by a generalisation of `loss.split_shots` (`imperfections.split(n, shots,
eta, x | q)`): a slot is lost w.p. `1 − η`, otherwise good w.p. `x` (or species `μ` w.p. `q_μ`),
otherwise a singleton. Photons of one species form an ordinary instance of the lossless engine
on their columns; singletons are one-photon species. Outputs of different species **add**
(counts) — the detector does not see species. Loss and distinguishability commute (both are
input-side, per-photon, independent), so one draw covers both.

How a singleton contributes depends on whether the input is a product over photons:

| input | bad photon `k` with drawn output row `r` | remark |
| --- | --- | --- |
| Fock, ProductSuperposition | `occ[r] += 1` with `P(r) = |U[r, col_k]|²` | classical routing; for SBS the photon is present w.p. `sin² α` |
| EmitterMPS (any ops) | fold `G_k(r) = Σ_l U[r, modes[k][l]] M_{k,l}` into the chain (`EmitterMPS.fold`, with the row instead of the label) | the label stays coherent; rank 1, `R = 1` |
| SequenceKernel bin (label-preserving ops) | `x` is drawn anyway; route `|U[r, modes[a][x_a]]|²`, register update carries the factor `U[r, modes[a][x_a]]` | the register records the label, so the fold is diagonal |
| LoopRegisterKernel | one-photon register evolution through the same gate program | self-interference across delay lines: amplitudes, not a Markov walk |

The strong side has no output-side shortcut (loss had one because `η·𝟙` commutes with `U`;
the internal state does not): it is the mixture over partitions, exact by enumeration with a
controllable truncation (§4).

## 3. Engine by engine

### 3.1 Fock inputs — `clifford.fock_sample`, `FockSLOSKernel`, CliffordGPU / SLOSTorch backends

*Sampler.* `fock_sample(U, in_modes, shots, loss=η, indist=x)`: the existing `by_k` grouping
of `split_shots` with the good subset as the "surviving" subset; the bad columns are routed on the
host (`torch.multinomial` over `|U[:, col]|²`, batched) and scatter-added to `occ`. Kernels
unchanged, one lockstep pass per good count `g`. Multi-species: every (stream, species) pair is a
stream of the pass, grouped by size; outputs summed per stream. Cost `E[ρ^g]`,
`g ~ Bin(n, ηx)`.

*Strong.* `probs(U, loss=η, indist=x)` → `LossyDistribution` (with loss) or a plain vector:

    P = Σ_{G ⊆ photons} x^{|G|} (1−x)^{n−|G|}  classical_raise(bad = G^c, SLOS(G)),

`classical_raise` being a probability-level layer step on the same `FockLayers` edges
(`q_{k+1}(t + e_j) += q_k(t) |U[j, i]|²`, no bosonic factor — the photon is another species), one
edge pass per bad photon. `Σ_g C(n,g)·cost(g)`: at `m = 2n` the overhead over the lossless kernel
is ≈ `(4/3)^n` worst case — 24× at `n = 11` — but the low-`g` terms are cheap, and with
`order=b_max` (keep partitions with ≤ `b_max` bad photons) it is `Σ_{b≤b_max} C(n, b)` runs with
TVD ≤ `P(Bin(n, 1−x) > b_max)`: at `n = 11, V = 0.9` (`x = 0.949`) `b_max = 2` → 67 runs, bound
0.019; `b_max = 3` → 232 runs, bound 0.0025. Differentiable in `U` and in `x` (the weights are
polynomials in `x`) — this is what E2 (source metrology from click statistics) needs: fit
`(η, x)` by gradient on `probs`.

*Perceval backends.* Accept annotated `BasicState`s: photons with equal annotation form a
species, distinct annotations do not interfere — the same primitive with a deterministic
partition. This lifts the "BasicState-only" restriction for exactly the case Perceval's `Source`
produces. Cross-check like `tests/test_loss.py` did against `LC`: `Source(indistinguishability=V)`
+ `Processor` vs `SLOSTorchBackend(indist=√V)` — note Perceval's `indistinguishability_model`
(`"homv"`: the parameter is `V = p²`; `"linear"`: it is `p`) must be mapped explicitly and tested.

*Distributed engine / exq backend*: input-side split applies; leave parked (EXPERIMENTAL).

### 3.2 Emitter inputs — `emitter_sample` (triton / torch / segments), `EmitterSLOSKernel`, estimator

*Sampler.* Draw the partition. Lost slots: label unravelling as today (`sample_lost_labels`).
Bad slots: **place them first**. Each bad slot `k` is a one-photon draw conditioned on the
reduced state of the chain with the earlier bad slots folded and the unplaced slots traced —
weight per candidate row `r` = norm of the chain with `G_k(r)` at slot `k`: `O(m · n L χ³)` per
bad photon, host/torch, negligible next to the Glynn chain. Fold it (`EmitterMPS.fold`:
the operator of a folded slot multiplies its right neighbour's operators, trailing ones the final
gate). Then run the existing good-photon chain (`keep` = good slots,
`labels=` machinery) on the folded `EmitterMPS`: **no Triton change**, `R = 1` preserved,
cost `E[ρ_eff^g]`. The order of placement is free in the chain rule (exact for any fixed order);
placing singletons first is what keeps the good-photon Glynn dimension at `2^{g−2}`.

*Strong.* `probs(U, indist=x)`: mixture over bad sets `B`; per `B` the output is
**species-resolved** — the bond-carrying SLOS runs on `(good pattern) ⊗ (rows of the bad
photons) ⊗ bond`, i.e. a bad photon's raise multiplies the amplitude index by `m` instead of
merging into the pattern, then `|·|²`, trace, and sum over the bad rows into the coarse pattern.
`C(n, b) · S_{n−b} · m^b · χ` per term; `order=b_max` with the binomial-tail bound is the
practical setting at `x ≈ 0.95`. (No cheaper route: the bad photon is coupled to the register
through its label, so its row cannot be summed before the trace.)

*Kraus operators* (`EmitterSLOSKernel.kraus`): the same species resolution —
one operator per `(t_good, bad rows)`, coarse-grained into a channel per `t`.

*Estimator* (`slos/estimate.py`): draw the partition, fold the bad slots at their drawn rows,
estimate the good-subset amplitude; the estimator's second-moment theorem is per instance, so the
bound holds per partition. `probability(U, t, shots, loss=, indist=)`.

### 3.3 Superposition boson sampling — `ProductSLOSKernel`, `glynn.py`

A bad source `i` factorises: `(cos α|0⟩ + sin α|1_bad⟩) ⊗ rest`, the bad photon's presence is
a Bernoulli(`sin² α`) classical photon routed by `|U[:, mode_i]|²`, with **no cross term** with
the good sources' vacuum component (orthogonal internal state). So

    P = Σ_{G ⊆ sources} x^{|G|} (1−x)^{n−|G|}  [ ProductSLOS(G) ⊛ ⊛_{i∉G} Bernoulli-routed photon_i ],

the convolution acting layer by layer on the graded `LossyDistribution` (the structure
`thin_graded` already walks). The sampler is strong-bounded here anyway (C0 gate), so the sampler
side is just `.sample()` on the mixture. `glynn.py` (one amplitude) is species-resolved by
construction: one amplitude of the good sources times the bad photons' factors.

### 3.4 Sequences — `SequenceKernel` (k emitters × n bins)

Two parameters per bin: per-emitter purity `x_a` (OBB, random per tick) and an inter-emitter
Gram `s_ab` (deterministic). Both go through species-resolved Kraus operators: expand the
internal states in an orthonormal basis (`C = chol(S)`, ≤ `k` species), the bin amplitude for a
species-resolved pattern `(t^{(1)}, …, t^{(k)})` is a sum over assignments of emitters to species
of `∏_a C_{a μ(a)}` × ∏_μ Perm of the species' sub-instance — `k^k` assignments at most, in
practice few since `C` is triangular; `k ≤ 7`, computed **once per bin** like the Kraus set
today (static `U`). `kraus(U, b, indist=)` returns the refined set plus the coarse-graining map;
`transfer`, `bin_marginals`, `sequence_probability`, `joint` sum over the refinement and are
otherwise unchanged (the channel is still `E_b(ρ) = Σ_{t, refinement} K ρ K†`). `bin_marginals`
stays differentiable, now also in `x_a` and `s_ab`.

`sample_diag` (label-preserving emitters): the label word `x` is drawn already; with OBB purities
draw the bad emitters per tick, Fock-sample the good ones (`SequenceKernel.fock_sample` on the good columns),
route the bad ones classically at their drawn label, and run `label_amplitudes` over the good
emitters' `2^{g}` label words with the bad emitters' factors `U[r_a, modes[a][x_a]]` fixed —
cheaper than today, not dearer. A deterministic Gram between dots does not fit `sample_diag` (the
species assignment is coherent, not a per-tick draw); that case uses `sample` on the refined Kraus
set, `k ≤ 7`.

Feed-forward (`policy`) and hooks: unchanged, they act on `psi` and `counts`.

### 3.5 Spin-optical stages

Stage-level distinguishability for spin-optical processors lives in the
separate spin-register package, built on `EmitterSLOSKernel` (`kraus(indist=)`) and the generic
layers here.

### 3.6 Time loops — `LoopRegisterKernel`, `PeriodicNetworkKernel`, `TimeLoopModule`, MerLin

*Sampler.* Draw the bad bins. Good photons: the trajectory sampler as today with the injection
schedule restricted to good bins (the register basis is `≤ N` photons, so nothing changes in the
basis). Each bad photon: its own one-photon register evolution through the same period program
(`fock_blocks` at `N = 1`, loop losses `η_loop` per round trip as amplitude damping, detector
efficiency), sampled at exit; add the counts per period. Cost: `+ O(n_bad · T · L)` on top of
the good-photon run — cheaper than lossless.

*Strong (`bin_marginals`, `distribution`).* The good register's marginals are correlated with
the bad photons' arrivals through the per-bin good/bad decisions, and `N` is large here
(30–270), so the `2^N` mixture cannot be enumerated. Two exact options, both to be measured
before choosing: (a) the density-matrix path (which exists for the loop channels) with the
injection channel `ρ → x·Iρ I† + (1−x)·ρ` and the bad photons carried as classical single-photon
amplitude registers correlated to the injection bin — exact, `D² × (walker states)`; (b) Monte
Carlo over bad sets with the pure-state path, `1/√shots` on the marginals but polynomial per
sample. The `TimeLoopModule` / MerLin `TimeLoopLayer` gradient path stays pure-state only for
`x = 1` unless (a) is taken. Recommendation: (b) first (it is what the sampler gives for free),
(a) if a trainable lossy-and-distinguishable loop layer is wanted.

Threshold detectors: species add before the click, nothing to do.

## 4. Exactness, truncation and the oracle contract

`docs/SUPERPOSITION.md` §1 recorded that partial distinguishability "would be the library's
first approximate engine". It is not: with the species primitive the **samplers are exact** at
every `x`, and the **strong sides are exact by enumeration**. The only approximation is the
opt-in `order=b_max` truncation of the strong sides, whose error is the binomial tail — computed,
`U`-independent, returned with the result (`LossyDistribution.bound` or a second return value).
Renema's order-`k` expansion is the *other* end of the same subset sum (few good photons,
error `x^{k+1}`-like, polynomial in `n` at fixed `k`) and is the right tool only at small `x`;
it can be added later behind the same `order=` argument without a new contract. Which end to
truncate from is decided by `x`: bad-photon count at `x ≳ 0.8` (hardware), good-photon count at
`x ≲ 0.5`.

Oracle: `slos/reference.py` gains `gram_probability(U, cols, S, t)` — the `(n!)²` double
permutation sum, `n ≤ 6` — and `species_probability(U, cols, species, t)` (product of
permanents over an orthogonal partition). Every engine's `indist=` path is tested against the
Gram oracle with `S = xJ + (1−x)I`, against the species oracle for annotated inputs, and against
the random-species law with `Tr ρ^{|c|}` weights for `q`-species.

Invariants to add to `docs/ARCHITECTURE.md` §4:

| invariant | test |
| --- | --- |
| `indist = 1` reduces to the lossless path exactly, object for object | every engine |
| `indist = 0` is classical transport: `P(t) = Perm(|U|²[t, cols]) / ∏ t_j!` | Fock |
| HOM: `P(1,1) = (1 − x²)/2` on a 50:50 splitter | Fock, emitter (`n = 2`) |
| Sampler TVD vs strong mixture within shot noise | every sampler |
| Strong mixture equals the Gram oracle at `n ≤ 6` | every strong engine |
| Loss and distinguishability commute: `(η, x)` jointly = `η` then `x` = `x` then `η` | Fock, emitter |
| Truncated strong distribution is within its own bound of the exact one | Fock, emitter |
| Perceval `Source(indistinguishability=V)` agrees with `indist = √V` (both `homv` and `linear` mapped) | adapters |
| Species-resolved Kraus set is a trace-preserving channel after coarse-graining | sequence |
| Bad-photon one-photon register evolution agrees with the unrolled unitary on a `{good, bad}` two-species instance | timeloop |

## 5. The cost law and the frontier — measured (2026-09-21)

Sampling cost per shot is that of the good-photon instance, `g ~ Bin(n, ηx)`, so with the
lossless anchors (`n*` at 1 sample/s, one H100, `m = 2n`) the prediction is

    n*(η, x) = n*(1) · log ρ / log(1 − ηx + ηxρ),     ρ = 1.81 (Fock), 1.99 (emitter, unravelled).

Measured, runs 20260921_051530 and 20260921_094327 (`bench/frontier_indist.py`,
`results/keep/frontier_indist_run2.csv`, `img/frontier_indist.png`), prediction in parentheses:

| n* emitted photons | `V = 1` | `0.95` | `0.9` | `0.8` |
| --- | --- | --- | --- | --- |
| Fock, `η = 1` | 38.6 (38.6) | 39.0 (39.4) | 40.0 (40.2) | 41.8 (42.0) |
| Fock, `η = 0.7` | 49.3 (51.0) | 48.7 (52.1) | 51.0 (53.2) | 53.7 (55.8) |
| emitter, `η = 1` | 30.6 (30.6) | 31.3 (31.2) | 31.8 (31.8) | 33.0 (33.2) |
| emitter, `η = 0.7` | 40.1 (40.0) | 40.4 (40.8) | 42.0 (41.7) | 43.5 (43.7) |

Reading. At `η = 1` the law holds to 0.4 photons on both engines; at `η = 0.7` the emitter
follows it to 0.4 and the Fock points sit 1.7–3.4 below it with a per-point `ρ_eff` scattered
1.45–2.02 (the tail-dominated lossy cost makes two-point brackets noisy at this loss — LOSS.md §7
saw the same "cliffs"; the `V = 0.95` point below `V = 1` is that noise, not physics). Partial
distinguishability at today's `V ≈ 0.9–0.95` is worth 1–2 photons of frontier against 3–10 for
loss, and the emitter lead over Fock in detected photons stays 8 (`η = 1`) / 6–7 (`η = 0.7`) at
every `V`. The "ideal-equivalent" photon number `ηx·n*` — the metric `M` of Soret et al. — is
measured: 37.4–38.6 (Fock, `η = 1`), 33.2–34.5 (Fock, `η = 0.7`), 29.5–30.6 / 27.2–28.3 for the
emitter; it drifts down slowly with `V` because the distinguishable photons still have to be
sampled (routed) and the good-subset cost is what sets the frontier. This settles the
distinguishability caveat of the comparison with Soret et al. on our side.

**Fock `η = 0.7` row re-measured at 4× budget (run 20261003_090228,
budget 480 s per point, `results/keep/frontier_indist_eta0.7_x4_20261003_h100.csv`,
`img/frontier_indist_eta0.7_x4.png`):** 49.4 / 50.3 / 50.8 / 53.9 at `V = 1 / 0.95 / 0.9 /
0.8` (prediction 51.0 / 52.1 / 53.2 / 55.8), brackets 49-50 at 1.47 / 0.55 per s, 50-51 at 1.27 / 0.62,
50-51 at 2.20 / 0.83, 53-54 at 1.94 / 0.95; per-point `ρ_eff` 1.74 / 1.61 / 1.50 / 1.50 against the
lossy law's 1.57 / 1.55 / 1.54 / 1.51. The row is now monotone in `V` — the 48.7 below 49.3 of the
120 s run was bracket noise, as said above — and sits 1.6–2.4 below the law instead of 1.7–3.4; the
`V = 1` point agrees with the loss run's 50.0 (LOSS.md §7) within the bracket. The `ηx·n*` metric
reads 34.6 / 34.3 / 33.7 / 33.8. These replace the `η = 0.7` Fock row of the table above in the
paper; the emitter rows are unchanged (120 s run).

The strong-side overhead is a constant-base exponential (`Σ_b C(n,b)` terms) that the
`order=` truncation makes polynomial at fixed `b_max`.

## 6. Documentation and paper consequences

- `docs/DISTINGUISHABILITY.md` (this note, rewritten as the design note once implemented),
  `docs/LOSS.md` §6 (remove "distinguishability" from "not covered", point here),
  `docs/SUPERPOSITION.md` §1 (correct the "first approximate engine" sentence), `docs/ARCHITECTURE.md`
  §2 (`imperfections.py` is the shared vocabulary for both imperfection axes) and §4 (invariants),
  `docs/TUTORIAL.md` (one subsection: `loss=` and `indist=` side by side),
  `docs/HARDWARE_TARGET.md` (add the `V` axis: one table).
- The frontier paper: the decision "partial distinguishability: out of scope" was taken when
  it meant an approximate engine and a new algorithm; it now means one parameter on the existing
  engines, one formula and one figure (`img/frontier_indist.png`, `V ∈ {1, 0.95, 0.9, 0.8}`
  curves next to the loss curves). Proposal: keep it out of the *algorithm* comparison, put the
  cost law and the figure in the loss section, and use it in the Soret section for the metric
  conversion, once phase 1 has the numbers.
- Source metrology from click statistics becomes tractable: `probs(U,
  loss=η, indist=x)` differentiable in `(η, x)`, fit on the photon-number and pattern statistics.

## 7. Action plan

Dependencies run downward; each phase ends with the CPU suite green and coverage above the floor.

**Phase 1 — the primitive and the two GPU samplers** (the frontier result) — DONE 2026-09-20 except the GPU run
1. `photonic_engines/imperfections.py`: `Distinguishability` (`x`, or `species=q`),
   `from_hom_visibility`, `draw_roles` / `partition(n, shots, eta, indist)` (the superset of
   `loss.split_shots`, which is left as is), `route_classical`, `subset_weights` (the strong-side
   mixture, test helper for now). `EmitterMPS.fold(folds)` generalises the lost-slot
   fold to row folds.
2. `slos/reference.py`: `species_probs` (species-resolved permanent sum, tabulated) and
   `gram_probs` (the `(n!)²` double permutation sum). `tests/test_imperfections.py`: the
   x-model/OBB equivalence to 1e-13 at n = 3, 4; the random-species cycle-weight law at n = 3;
   `x = 0` classical transport; draw law; fold correctness.
3. `fock_sample(indist=)`: `partition` groups, species instances as circuits of the lockstep pass,
   singletons (and one-photon species) routed on the host. Tests: `x = 0` (6000 shots, no kernel),
   dispatch, and the x-model mixture under the Triton interpreter (`slow`).
4. `emitter_sample(indist=)` on triton and torch: `emitter_terms.sample_slot_outcomes` draws lost
   labels AND singleton rows in one forward pass (labels `L + r`), `label_weights` turns them into
   per-slot operator weights, `batched_gap_terms` / `gap_kraus_terms` fold a singleton slot as the
   ONE operator `Σ_l U[r, m_l] M_l` (the first version left it as the set `{w_l M_l}` — a mixture,
   caught by the R = 1 test), the chain rule runs on `keep` = interfering slots and the rows are
   added to the output. Tests: rows' marginal at `x = 0`; R = 1 and the explicit folded chain;
   the mixture with and without loss (torch, 12 000 shots); triton interpreter (`slow`).
5. `bench/find_threshold.py --indist x`, `bench/frontier_indist.py` (η × V grid, CSV with
   `det_n_star = η n*` and `ideal_n_star = η x n*`, PNG). **Not run on a GPU.**

**Phase 2 — strong sides and the Perceval front end** — DONE 2026-09-20 (`slos/species.py`; the
estimator (item 8, last clause) is left for later; `SuperpositionSLOSKernel` refuses `indist=`
through the Perceval backend, a coherent superposition having no per-photon identity). One
finding worth recording: the SBS strong side must keep the good sources COHERENT (the Dicke
state) and only the bad sources classical — an oracle that treats each emission pattern
incoherently is off by 2e-2 at three sources, which is the size of the Dicke coherence.
6. `FockSLOSKernel.probs(indist=, order=)` with `classical_raise` on the basis edges and the
   binomial-tail bound; differentiable in `x`. `SuperpositionSLOSKernel`: only for a partition
   given per input mode (component-independent), otherwise `NotImplementedError` with the reason
   (a coherent superposition has no per-photon identity).
7. `ProductSLOSKernel.probs(indist=)` (per-source purity, layer-wise convolution) and
   `glynn.py`.
8. `EmitterSLOSKernel.probs(indist=, order=)` species-resolved; `kraus(indist=)` refined set
   plus coarse-graining map; the estimator's `indist=`.
9. Perceval backends: annotated `BasicState` input as a species partition; `indist=` kwarg;
   `Source` cross-check tests with the `homv` / `linear` mapping pinned down.
10. Docs of §6 (LOSS §6, SUPERPOSITION §1, ARCHITECTURE §2/§4, TUTORIAL, AGENTS, HARDWARE_TARGET).

**Phase 3 — the register modalities** — DONE 2026-09-20 for the x-model / per-emitter purities
(items 11–13 below; `sample_diag` and the deterministic Gram between dots are the parts left,
documented in SEQUENCE.md §13 / TIME_LOOP.md §6). The time-loop strong side took option (b) of §3.6: exact enumeration of the
2^N subsets up to N = 8, Monte Carlo above.
11. `SequenceKernel`: `indist=` as per-emitter `x_a` and optional Gram `s_ab`; species-resolved
    `kraus`, refinement map through `transfer` / `bin_marginals` / `sequence_probability` / `joint`
    / `sample`; `sample_diag` with per-tick OBB draws (good-only Laplace). Tests: channel trace,
    vs the strong side, `x_a = 1` identity, two-dot HOM in one bin.
12. Spin-optical stages: moved out with the separate spin-register package (§3.5).
13. `LoopRegisterKernel` / `PeriodicNetworkKernel`: `sample(indist=)` with one-photon bad
    registers; `bin_marginals(indist=, mc_shots=)` Monte Carlo first; the density-matrix injection
    channel only if a trainable path is wanted. Test vs `circuit.py` unrolled on a two-species
    instance. `TimeLoopModule` / MerLin pass-through.

**Not planned**: time-correlated Gram matrices, fine-structure label-dependent species with
partial overlap, multi-photon emission, the distributed engine.

**Decisions needed from the user before phase 1**
- Parameter: `indist=x` (Gram overlap, `V = x²`) with `from_hom_visibility` (decided 2026-09-20).
- `loss.py` absorbed into `imperfections.py` (decided 2026-09-20, the library has no external user yet); `split_shots` kept as is next to `partition`.
- Whether the paper decision is reopened after phase 1 (§6).
