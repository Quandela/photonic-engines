# Uniform photon loss

Every engine takes a `loss=` argument (`photonic_engines/imperfections.py`, which also holds partial distinguishability). This note fixes the model,
where each engine applies it and why, and what it does not cover.

## 1. Model

One end-to-end transmission η per photon — source brightness × chip/fibre transmission ×
detector efficiency — the same on every mode. Each photon is kept independently with
probability η, so the detected photon number is k ~ Binom(n, η) and, given the lossless
output pattern s, the detected pattern is the binomial thinning of s mode by mode.

Uniform loss commutes with any linear-optical unitary (it is η·𝟙 on the modes, a scalar
in the mode transformation), so it can be applied on the **input** side (which photons enter
the interferometer) or on the **output** side (which detected photons are kept) with the same
distribution. The two sides differ only in cost, and each engine takes the cheaper one.

Threshold detectors are a post-processing of counts (`imperfections.clicks`,
`LossyDistribution.clicks()`, `sample(threshold=True)` on the time-loop kernel).

Per-mode transmissions, detector efficiencies that differ between modes, and loop losses in a
time-loop device (a photon circulating k times pays η_loop^k) are not uniform, do not commute
with the gates, and are not covered here; the time-loop engine keeps its Kraus channels for
those (`loss=dict(loop=[...], detector=...)`, docs/TIME_LOOP.md), with the uniform part folded
into the detector efficiency (`uniform` key).

## 2. Strong simulation: output-side thinning down the basis layers

`FockSLOSKernel.probs(U, loss=η)`, `SuperpositionSLOSKernel.probs(U, loss=η)`,
`EmitterSLOSKernel.probs(U, loss=η)` return a `LossyDistribution`: one probability vector per
detected photon number k = 0..n over the k-photon layer of the same `FockLayers` basis
(`.probs[k]`, `.occupations(k)`, `.mass(k)` = C(n,k) η^k (1−η)^{n−k}, `.conditional(k)` the
post-selected distribution, `.flat()`, `.as_dict()`, `.clicks()`, `.sample(shots)`).

Removing one photon uniformly at random from a k-photon pattern t + e_j lands on t with
probability (t_j + 1)/k. That is the raising-edge structure the basis already stores (edge
factor² = t_j + 1, one edge per (child, mode) pair), so the k−1 layer is one scatter-add over
the (k−1 → k) edges:

    q_{k−1}(t) = Σ_j q_k(t + e_j) (t_j + 1)/k,     q_n = the lossless distribution,
    P_loss(layer k) = C(n,k) η^k (1−η)^{n−k} q_k.

Cost O(edges) — one SLOS layer pass, without the complex multiply — batched over U and
differentiable in the probabilities and in η (a tensor η gets a gradient). Every intermediate
pattern of a mask-pruned basis is a sub-pattern of an admissible final state, so the pruned
layers are complete for the recurrence. For the emitter kernel the thinning acts on the mixture
(traced spin) or the projected distribution alike, since loss commutes with the spin trace.

`imperfections.thin_patterns(occupations, probs, η)` does the same on an explicit pattern list (the
patterns may have different photon numbers — each class is thinned with its own binomial
weights and the classes are merged per detected count); `imperfections.thin_marginals(marg, η)` thins
per-mode photon-number marginals [.., N+1] (each mode's count independently);
`imperfections.thin_counts(samples, η)` thins samples.

Input-side thinning would instead run the kernel on every surviving subset: Σ_k C(n,k) SLOS
runs of k photons ≈ (4/3)^n × the lossless cost at m = 2n. Not offered.

### 2b. Inputs that are already graded (`thin_graded`)

`ProductSLOSKernel` (superposition boson sampling, docs/SUPERPOSITION.md) has an indefinite input
photon number, so its lossless output is ALREADY a `LossyDistribution`: every layer carries mass,
not just the top one. `imperfections.thin_graded(layers, blocks, η)` thins each source layer with its own
binomial weights and sums the results per detected count,

    P_loss(j) = Σ_{k ≥ j} C(k, j) η^j (1−η)^{k−j} proj_{k→j}(blocks[k]),

`proj` being the same descent (`imperfections._descend`, extracted from `thin_distribution`) repeated.
Cost: Σ_k k descents, i.e. n times one `thin_distribution` — acceptable on a strong-simulation
path. η = 1 returns the blocks untouched, object for object.

## 3. Sampling: input-side split, one lockstep pass per surviving count

The GPU samplers are compiled per photon number: `fock_sample` runs all streams in lockstep
with lanes = streams × Gray segments of a fixed k, `emitter_sample` builds its term lanes per
step k. A per-shot photon number would break the lockstep. Instead `imperfections.split_shots(n,
shots, η)` draws the surviving input subset of every stream on the host and groups the streams:

* `fock_sample(U, in_modes, shots, loss=η)`: streams are grouped by surviving count k; inside a
  group the distinct surviving subsets enter as distinct circuits (column sets U[:, subset] —
  the B axis the kernels already index per stream through `dp`), so every k is one lockstep
  pass of the unchanged kernels on a k-photon instance. With B circuits × subsets the input
  block is [B·G_k, m, k] complex128 (a few MB at n = 24, 4096 shots).
* `emitter_sample(state, U, shots, loss=η)` (backends triton, torch, segments): the lost
  emissions are traced out, which the samplers' reduced-state machinery does exactly (a
  surviving subset T of the emission slots is a set of kept slots with gaps; the gap-Kraus
  terms of docs/EMITTER_SAMPLER.md already handle prefix, inner and suffix gaps). Every
  backend takes `keep [shots, k]` — the surviving slots per stream — and runs its chain rule
  over those slots only; `emitter_sample` groups the streams by k and calls the backend once
  per group.

Expected cost Σ_k P(k)·cost(k) < cost(n): the lossy run is cheaper than the lossless one, and
grows cheaper as η drops (at η = 0.7, n = 24 the mass sits on k = 14..20). Measured on the
H100 (`bench/bench_loss.py`, 2026-09-12, m = 2n, 4096 shots): Fock n = 24, 0.54 ms
per sample lossless → 0.22 / 0.056 / 0.034 ms at η = 0.9 / 0.7 / 0.5; emitter n = 24, 10.8 ms
→ 10.3 / 6.4 / 3.1 ms. The emitter sampler pays its per-step host work once per k-group, so
it only gains below η ≈ 0.8 (a merged pass with streams leaving the lockstep at their own k
would fix that). First use of a new photon number compiles its Triton kernels. Output-side
thinning of lossless samples (`imperfections.thin_counts`) gives the same distribution at the full
lossless cost; it is what the exqalibur reference backend and the distributed engine's
`sample(loss=η)` use.

Perceval front end: `SLOSTorchBackend(loss=η)`, `EmitterSLOSBackend(loss=η)`,
`CliffordGPUBackend(loss=η)`, `DistributedSLOSBackend(loss=η)`: `probability(t)` and
`prob_distribution()` over 0..n detected photons, `samples(count)`; `all_prob` / `evolve` /
`prob_amplitude` are undefined with loss (Perceval's own loss handling — `LC` components,
`Source` losses inside a `Processor` — is the alternative when the loss must live in a
Processor). Cross-check: the thinned SLOS distribution equals Perceval's `LC(1−η)` on every
mode to 1e-16 (tests/test_loss.py).

### 3b. Emitter inputs: the label unravelling of the trace (2026-09-18)

The traced gap terms above are exact but expensive: an inner gap contributes up to χ² = 4 pure
terms, the prefix and suffix up to χ, and the term count R multiplies the k-photon cost. At
η = 0.5 a stream has ~n/4 gaps, so R ~ 4^{n/4} — a factor ≈ 1.41 per emitted slot on top of
ρ^{ηn}: the lossy sampler then costs about as much as the lossless one, whatever η (measured:
n* = 30.6 / 30.9 / 31.2 / 31.9 / 34.4 at η = 1 / 0.9 / 0.8 / 0.7 / 0.5, run 20260917_214209).

A lost photon is traced, and a trace is a label-basis measurement whose outcome is forgotten;
operations on distinct photons commute, so the lossy distribution is Σ_x p(x) P_U(t | ψ_x) with
x the lost slots' labels, p(x) their joint marginal (kept slots and final spin summed over) and
ψ_x the *pure* emitter state conditioned on x — the chain with M_{j, x_j} fixed at the lost
slots. `emitter_terms.sample_lost_labels(state, keep)` draws x and the final basis state jointly
(right environments R_j = Σ_l M†_{j,l} R_{j+1} M_{j,l}, forward register density matrix per
stream, O(S n L χ³) on the host), `batched_gap_terms(..., labels=x)` and
`gap_kraus_terms(..., labels=x)` then keep one operator per traced slot, so every gap set has
rank 1 and R = 1: a lossy sample costs exactly what a k-photon lossless sample costs, and the
frontier follows the same E[ρ^k] = (1 − η + ηρ)^n law as the Fock sampler (§7).
`emitter_sample(loss=η)` uses it on the triton and torch backends (`unravel=False` restores the
traced terms; the segments backend keeps them). tests/test_loss.py `test_lost_label_unravelling`:
the drawn labels match the exact joint marginal, one term per stream, same TVD to the thinned
strong distribution either way. On a GPU, `bench/loss_unravel_check.py` compares `unravel=True` / `False`
(detected-photon law, mean occupation per mode, timing) and fails loudly on non-finite triton weights.

## 4. Time-loop register engine

`LoopRegisterKernel.bin_marginals / sample / distribution(loss=η)` (scalar, or
`dict(uniform=η)`): lossless pure-state run, then `thin_marginals` per bin / `thin_counts` per
shot / `thin_patterns`. The register basis is built for N photons, so an input-side split
would not shrink the per-shot register; output thinning costs nothing on the register and
keeps the pure-state path (the density-matrix path is D² and only needed for the loop
channels). `sample(loss=η, return_register=True)` returns the lossless trajectories' register
states. With loop channels present, `uniform` is folded into the detector efficiency:
`loss=dict(loop=[0.9, 0.95], uniform=0.8)` ≡ `dict(loop=[0.9, 0.95], detector=0.8)`.
`TimeLoopModule` and the MerLin `TimeLoopLayer` pass `loss` through (readouts clicks /
marginals / mean; `patterns` / `distribution` readouts stay pure-state).

## 5. Spin-optical stages

Stage-level loss for spin-optical processors lives in the separate
spin-register package, built on `EmitterSLOSKernel` and the generic loss layer here.

## 5b. Loss on a partially excited source (2026-09-20)

A superposition source is not a Fock source, so loss does something to it that §1 does not cover:
it degrades the **coherence** and the **brightness** at different rates. With Kraus operators
K₀ = diag(1, √η) and K₁ = √(1−η)·a on |ψ⟩ = c|0⟩ + s|1⟩,

    ρ = |φ⟩⟨φ| + (1−η)s²|0⟩⟨0| ,   |φ⟩ = c|0⟩ + s√η|1⟩
      = w₁ |α′⟩⟨α′| + w₀|0⟩⟨0| ,   w₁ = c² + ηs² ,  w₀ = (1−η)s² ,  tan α′ = s√η / c

so **a lossy partially-excited source is a mixture of vacuum and a partially-excited source at a
smaller α**. Loss is independent per source, so the n-source state is a mixture over *subsets*:
keep each source with probability w₁ as a superposition at α′, vacuum the rest. Every term is a
lossless `ProductSuperposition` on a subset of modes, which is why `clifford.product_sample(…,
loss=η)` needs no new algorithm — it draws a Bernoulli mask and runs the lossless chain rule on the
live modes (`clifford.product_sampler.lossy_source`). Checked against
`ProductSLOSKernel.probs(U, loss=η)`, which thins a completely different engine: **TVD 2e-16** on
the exact mixture and within shot noise from the sampler.

The two quantities move differently:

| | brightness | coherence |
| --- | --- | --- |
| | p = η sin²α | χ = ⟨a⟩ = cos α · sin α · √η |
| scaling in η | η | **√η** |

Fixing p leaves **one free parameter**, and it is exactly the coherence:

    sin²α = p/η ,   χ = √( p (1 − p/η) ) ,   η ∈ [p, 1]

    η = p  →  α = π/2, χ = 0             a deterministic source behind a lossy channel: the FOCK
                                         modality at transmission p
    η = 1  →  sin²α = p, χ = √(p(1−p))   partial excitation with no loss: SBS, maximal coherence

Every point on that segment has the same source count and the same photon statistics,
Binomial(n, p). Only the coherence changes. That is the lossy-advantage question in one axis — at a
detected photon rate a device can actually deliver, does coherence make the output harder to
simulate? — and `bench/coherence_scan.py` measures it.

**The η = p end is a calibration, not a data point.** There the sbs engine and the Fock engine
simulate the *same state*, so pricing both (`--with-fock`) measures the implementation gap between
the two engines on identical physics, which is otherwise the thing every sbs-vs-fock comparison has
to argue about rather than measure. The lossless Fock reference at p = 0.5 is n\* = 64.0 sources /
32.0 detected photons (§7).

**Measured (run 20260920_193154, one H100, p = ½, m = 2n):**

| η | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 |
| --- | --- | --- | --- | --- | --- | --- |
| χ / χ_max | 0 | 0.58 | 0.76 | 0.87 | 0.94 | 1 |
| n\*, sbs engine (sources) | 22.4 | 22.2 | 20.9 | 20.9 | 20.8 | 20.2 |

The Fock engine on the η = p state reaches **64.4** measured the same day (run 20260921_071710,
n = 64 at 1.11/s, n = 65 at 0.82/s; 64.0 in the 2026-09-17 scan of §7), and the sbs engine on that
state re-measured 22.4 exactly. So of the 44.2-source fall from the incoherent end to the coherent
one, **42.0 is the engine and 2.2 is coherence**.
Coherence costs the best known exact algorithm about ten per cent. The engine gap is Clifford &
Clifford's random photon order, which reduces the Fock marginal to one permanent and has no
consistent analogue for superposed sources, so the superposition chain rule pays its residual
enumeration even when the input is Fock. The best-known-algorithm envelope is therefore
discontinuous at χ = 0 — 64.0 there, about 22 for any χ > 0 — and whether superposition inputs are
intrinsically harder than their incoherent twin is left open by this data.

(The first `--with-fock` point, in run 20260920_193154, is not usable: it inherited the sbs search
window, never reached n ≈ 64, and reported an extrapolation to 141 from ρ = 1.06.
`bench/coherence_scan.py` now gives the Fock point its own window, seeded from the lossy law, and
that is how the 64.4 above was taken.)

The device curve at α = π/4 (`frontier_loss.py --engines sbs`, runs 20260920_193154 and, for
η = 0.5, 20260921_080421): n\* = 20.2 / 20.5 / 24.0 / 25.5 / 31.2 at η = 1 / 0.9 / 0.8 / 0.7 / 0.5
against the brightness-law prediction 20.0 / 21.6 / 23.5 / 26.1 / 34.0, i.e. the same law as the
Fock and emitter curves of §7; detected photons at threshold (brightness η/2 times n\*)
10.1 / 9.2 / 9.6 / 8.9 / 7.8. At η = 0.5 the rate is not monotone in n — n = 29 read 0.96/s, n = 31
1.21/s, n = 32 0.36/s, because the cost per shot is carried by rare high-k draws (one warm-up shot at
n = 32 took 41 min) — and the threshold is anchored on the largest n that met the target, so 31.2
is uncertain by about two sources. The first attempt at this point took the inverted 29/31 pair
for a bracket (n\* = 29.3, ρ_eff 0.892); the bracket check now requires the pair in order.

`bench/frontier_loss.py --engines sbs` is the other, simpler curve: α fixed at π/4 and η falling,
i.e. brightness η/2 — "how does my sbs frontier degrade as the chip gets lossier" rather than
"what is coherence worth". Its cost model is 1 − p + p·ρ_base with p = η sin²α, not 1 − η + ηρ:
the survivor probability per SOURCE is the brightness, not the transmission.

## 6. Not covered

Spin-side imperfections of the emitter (dephasing, branching, spin-flip errors: on the bond,
not on the photons), multi-photon emission, and any mode-dependent loss on the
generic engines (would need auxiliary modes or per-mode Kraus maps in the SLOS recurrence; the
time-loop engine has the latter for its loops and detector). Dark counts are a separate
output-side layer, `imperfections.py` (DARK COUNTS: `add_dark_counts`, `dark_distribution`,
`dark_predecessors`), used by the separate spin-register package. Partial distinguishability is the
second imperfection axis and has its own note, `docs/DISTINGUISHABILITY.md`: the samplers take
`indist=x` next to `loss=η` — one draw covers both (`imperfections.partition`), and the cost law
is this section's with η → ηx.

## 7. The lossy frontier n*(η) (2026-09-18)

`bench/find_threshold.py --loss η` and `bench/frontier_loss.py`: the photon number at which the
lossy sampler drops to 1 sample/s on one H100, m = 2n, η = 1, 0.9, 0.8, 0.7, 0.5.

Cost model. With input-side loss a stream with k survivors costs a k-photon lossless sample, so
the expected cost per sample is E[ρ^k] = (1 − η + ηρ)^n, k ~ Bin(n, η) — the moment-generating
function, dominated by the upper tail of k, not ρ^{ηn}: at n = 49, η = 0.7 the lossy Fock sample
costs 6.3× the 34-photon lossless one (measured 7.4×, `bench/loss_cliff.py`), and the per-batch
cost fluctuates by an order of magnitude with the largest k drawn (the "cliffs" at n = 51 and 65
of run 20260917_182540 were batches holding one stream with k = 42 / 44). The threshold in
emitted photons is n*(η) ≈ n*(1) log ρ / log(1 − η + ηρ); the mean number of *detected* photons
at the threshold, η n*, decreases with loss.

Fock (run 20260917_182540): n* = 38.6 / 41.4 / 44.7 / 50.0 / 64.0, η n* = 38.6 / 37.3 / 35.8 /
35.0 / 32.0; the prediction from n*(1) is within one photon at every η.

Emitter, traced gap terms (run 20260917_214209, after the `final_spin_distribution` fix): n* =
30.6 / 30.9 / 31.2 / 31.9 / 34.4 — flat, because R ~ 4^{gaps} eats the gain (§3b). With the label
unravelling (§3b; run 20260918_174153, 72 min): **n* = 30.6 / 33.0 / 35.5 / 40.1 / 53.5**, η n* =
30.6 / 29.7 / 28.4 / 28.1 / 26.8, against the prediction 30.6 / 33.0 / 36.1 / 40.0 / 52.4 from
n*(1) and ρ_eff = 1.99 — within a photon at every η, the same law as the Fock sampler
(`img/frontier_loss.png`, `results/keep/frontier_loss_run1.csv` (Fock) and
`frontier_loss_emitter.csv`).

Hardware reading (docs/HARDWARE_TARGET.md is the one-page version). At equal classical cost
(1 sample/s on one H100, m = 2n) the emitter modality needs fewer detected photons than a Fock
experiment at every transmission — 8.0 / 7.6 / 7.4 / 6.9 / 5.2 fewer at η = 1 / 0.9 / 0.8 / 0.7 /
0.5 (30.6 vs 38.6 lossless, 26.8 vs 32.0 at η = 0.5); the lead narrows slowly with loss because
the emitter's per-photon factor is larger (ρ_eff 1.99 vs 1.81), so its lossy factor 1 − η + ηρ
falls faster. Loss does not shorten the required cluster, it lengthens it: the frontier in
*emitted* photons is 31 / 33 / 36 / 40 / 54 at η = 1 / 0.9 / 0.8 / 0.7 / 0.5 — a single quantum dot
must emit a 40-photon cluster into 80 modes at η = 0.7, 54 photons into 108 modes at η = 0.5, to
reach the point where this engine needs one second per sample; today's 10–14-photon clusters are
20–40 emitted photons short of it at any η, and the gap in detected photons (27–31) is the same
number in every column.
