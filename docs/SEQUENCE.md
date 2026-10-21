# Compute modality: k emitters × n time bins through a static interferometer

`photonic_engines/sequence.py` — `SequenceKernel`, `BinEmitter`. Tests: `tests/sequence/`.
Tutorial §9e. Added 2026-09-12.

## 1. The setting

k spin (or qudit) emitters, each producing a graph state of length n as a train of n photons
in successive time bins (caterpillar / linear cluster / GHZ per emitter, or any `EmitterMPS`
chain). The trains are **not** space-demultiplexed: emitter j sends its photon of every bin into
its own input modes of one universal m-mode interferometer U (two modes per emitter for a
dual-rail / polarisation label), the interferometer contains **no delay line**, and every bin is
detected as it exits. The object of interest is the k·n-photon output — the *sequence* of the
n per-bin detection patterns — as one state: its probabilities, marginals, correlations
between bins, and samples. This sits between a feed-forward stage simulator (one stage,
feed-forward, Kraus operators on the register) and plain Perceval (one Fock input, one output pattern).

## 2. Why it is a chain, not a k·n-photon problem

Without delay lines, photons of different bins never meet: the unrolled unitary on the m·n
(mode, bin) slots is block-diagonal, U ⊕ U ⊕ … ⊕ U. The only thing that correlates the bins is
the product spin register χ = Π dⱼ (2ᵏ for spins). Bin b is exactly one emitter *stage*
(k emissions on the product register, one circuit, full detection), and the k·n-photon output
is the stage iterated n times without feed-forward:

    p(t₁ … tₙ) = Tr[ K_{tₙ} ⋯ K_{t₁} ρ₀ K_{t₁}† ⋯ K_{tₙ}† ],      K_t = √N_t ⟨t|Φ_b⟩ ∈ ℂ^{χ×χ},

one Kraus operator per k-photon pattern t of the bin (S_bin = C(m+k−1, k) of them), obtained
by **one bond-carrying SLOS pass** on the k-emission stage MPS of the bin
(`EmitterSLOSKernel.kraus`, χ × the k-photon Fock cost). When U and the spin gates are the same
at every bin the set is computed once; per-bin unitaries `U[n, m, m]` (a reprogrammed
interferometer) or bin-dependent spin gates give one set per bin. With projected emitters the
photonic state is pure and its amplitude ⟨f|K_{tₙ}⋯K_{t₁}|s₀⟩ is an MPS along the bin axis with
bond χ and "labels" = bin patterns.

So the exponent is **k**, the number of emitters interfering in one bin (χ = 2ᵏ, 4ᵏ per Kraus
operator, S_bin patterns); **n**, the length of the graph states, is linear. The full k·n-photon
distribution has S_binⁿ entries and is never enumerated — the same status as the time-loop
engine's full distribution: everything one asks of it is a chain on the register.

## 3. Strong simulation

`SequenceKernel(emitters, m)`, `emitters` a list of `BinEmitter(ops [n, L, d, d], modes [L], s0,
final)` (`BinEmitter.caterpillar`, `.from_mps` for an `EmitterMPS` written on the unrolled
modes); U as a tensor, a Perceval
circuit, or one per bin.

| call | what | cost |
|---|---|---|
| `kraus(U, b)` | [S_bin, χ, χ] physical Kraus operators of bin b | one SLOS pass, χ² × the k-photon Fock cost |
| `sequence_probability(U, seqs)` | p(t₁..tₙ) for sequences as occupations [B, n, m] or pattern indices [B, n]; traced (Tr) or projected (\|⟨f\|M\|s₀⟩\|²) register | n · χ³ per sequence |
| `sequence_amplitude(U, seqs)` | ⟨f\|K_{tₙ}⋯K_{t₁}\|s₀⟩ (projected emitters) | idem |
| `bin_marginals(U, loss=η)` | [n, S_bin]: the register density matrix marched through the transfer channel E_b(ρ) = Σ_t K_t ρ K_t†; `loss=η` thins every bin (`LossyDistribution` per bin) | n · S_bin · χ³, differentiable in U and in the spin gates |
| `joint(U, bins)` | joint distribution of a few bins, tensor [S_bin]^{len(bins)} | S_bin^{len(bins)} · χ³ per bin in between |
| `transfer(U, b, ρ)` | E_b(ρ) | S_bin · χ³ |

`bin_marginals` is the readout a trainable layer would use (features on the spin gates or on
U); it is autograd-clean (the Kraus cache is bypassed under autograd).

## 4. Weak simulation

`sample(U, shots, loss=η)` → [shots, n, m]: **pure quantum trajectories** on the register. The
register starts in the pure product state |s₀⟩; at bin b the weights of the S_bin patterns are
w_t = ‖K_t ψ‖² = ⟨ψ|K_t†K_t|ψ⟩ (one matmul [shots, χ²] × [χ², S_bin] with the precomputed
K_t†K_t), t is drawn, ψ ← K_t ψ / ‖·‖, next bin. Exact, no mixed register (a stage simulator's sampling
mode diagonalises a mixed ρ because a stage may start from one; here the trajectory keeps it
pure), no rejection except for projected emitters (accepted at the end with probability
|⟨f|ψ⟩|², as in the time-loop sampler). Cost n · shots · S_bin · χ² by enumeration of the bin
patterns — the right route while S_bin · χ² fits (k ≤ 6–8). Beyond that, for label-preserving
emitters (a spin gate then an emission that keeps the spin in the emitted label — the
caterpillar), the same trajectory runs **without the Kraus set** (`sample_diag`, §9): K_t
factors through the label basis as V diag_x⟨t|U|x⟩ Uop, the bin is a mixture over the label
word x of k-photon Fock instances, so per tick: draw x ~ |(Uop ψ)(x)|², draw t from the Fock
instance |x⟩ (Clifford GPU sampler), multiply the register by the 2ᵏ amplitudes ⟨t|U|x⟩
(a Laplace expansion over the k output photons, O(k 3ᵏ)). Memory 2ᵏ per stream instead of
S_bin·4ᵏ; the register cost per tick is then the only exponential (k ≈ 20 at 1 tick/s).

Uniform loss η: inside a bin it commutes with U, and bins are independent given the register,
so marginals and samples are thinned bin by bin (`photonic_engines.imperfections`). The trajectory keeps
the lossless register update K_{t'}ψ and reports the thinned pattern t ⊆ t': that is a valid
unravelling of the lossy channel E_t (the joint law of (detected patterns, register) is
unchanged by moving the loss after U), so no lossy Kraus sets are needed on this path.

## 5. Measured

![sequence frontier](../img/sequence_frontier.png)

**One H100 PCIe** (`bench/sequence_frontier.py`, runs 20260912_185503 / 190841 / 220323,
`results/keep/sequence_frontier_h100.csv`, `bench/plot_sequence_frontier.py` → `img/sequence_frontier.png`):
dual-rail emitters into a universal interferometer of m = 2k modes (and m = 4k), n = 4..400
bins, 1024 shots (256 above n = 40), complex128; the sampler's bin-0 marginal agrees with the
computed one at the noise floor everywhere.

| k | χ | S_bin (m = 2k) | Kraus set | ms per bin-shot | n* bins at 1 sequence/s | ms per bin-shot, m = 4k (S_bin) | peak, m = 2k |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | — | 0.0005 | 2·10⁶ | — | — |
| 2 | 4 | 10 | — | 0.0007 | 1.5·10⁶ | 0.0066 (36) | — |
| 3 | 8 | 56 | — | 0.0016 | 6·10⁵ | 0.0092 (364) | — |
| 4 | 16 | 330 | 1 MB | 0.0070 | 1.4·10⁵ | 0.113 (3 876) | — |
| 5 | 32 | 2 002 | 31 MB | 0.045 | 22 000 | 1.12 (42 504) | 0.2 GB |
| 6 | 64 | 12 376 | 0.75 GB | 0.35 | 2 870 | 12.8 (475 020; 29 GB, 73 GB peak) | 3.8 GB |
| 7 | 128 | 77 520 | 19 GB | 2.4 | 418 (measured: n = 400 at 1.06 /s) | — (0.5 TB) | 40.5 GB |

Three readings. (i) **Linear in n**: the cost per bin-shot is constant in n to 1 % from n = 4
to 400 for every k, so the reach at a given rate is a straight line n* = 1000 / (ms per
bin-shot): k = 7 reaches 418 bins (2 900 photons in the sequence) at one sequence per second,
measured directly at n = 400. (ii) **Exponential in k**, ∝ S_bin · χ²: ×7 per emitter at
m = 2k, and the interferometer size enters through S_bin only — at m = 4k every point costs
what the next k costs at m = 2k (k = 6, m = 24 ≈ k = 7, m = 14). (iii) The generic route on the
same instance — the emitter chain-rule sampler on the unrolled 2nk-mode block-diagonal unitary
(torch backend, χ = 2^k) — is exponential in k·n: 10.9 ms at k = 1, n = 12 against 0.005 ms,
16.7 ms at k = 2, n = 6 against 0.004, 55.9 ms at k = 3, n = 4 against 0.007 — three to four
orders of magnitude at 12 photons and diverging.

The frontier in k on one 80 GB card is **k = 7** at m = 2k (k = 6 at m = 4k, 73 GB peak): the
Kraus set is S_bin · χ² · 16 B (19 GB at k = 7; k = 8 would be 0.5 TB). It is built in column
chunks (`EmitterSLOSKernel.kraus(chunk=)`), the transfer channel and K†K in pattern chunks, so
only the two [S_bin, χ, χ] sets stay resident. Beyond that the per-bin draw must be the
k-photon chain-rule sampler with χ = 2^k — the χ > 2 kernel — whose cost is R_k 2^k k per bin
instead of S_bin χ².

**CPU sandbox** (2 threads, complex128; the same matmuls):

| k | n | m | χ | S_bin | Kraus set | marginals (n bins) | sampling |
|---|---|---|---|---|---|---|---|
| 2 | 3 | 4 | 4 | 10 | ms | ms | — (oracle checks) |
| 4 | 20 | 8 | 16 | 330 | 0.01 s | 0.12 s | 0.02 ms per bin-shot |
| 5 | 10 | 10 | 32 | 2002 | 0.9 s | 2.2 s | 0.25 ms per bin-shot |
| 6 | 10 | 12 | 64 | 12376 | 16 s | 88 s (≈ 9 s per bin: three [S, χ, χ] matmuls at 1 GFlop/s) | 8 ms per bin-shot |

Memory of a Kraus set: S_bin · χ² · 16 B (810 MB at k = 6, m = 12).

## 6. Validation

`tests/sequence/test_sequence.py`: k = 2, n = 3, m = 4 against the brute-force k·n-photon
strong simulation — the concatenated 6-emission `EmitterMPS` on the 12 unrolled modes through
the block-diagonal unitary (`EmitterSLOSKernel`, χ = 4): sequence probabilities to 1e-9
(float32 edge factors of the reference basis), bin marginals and the joint of two bins to
1e-12, sampling TVD at the noise floor, projected emitters (amplitudes), per-bin unitaries,
`from_mps`, gradient through U, uniform loss (thinned marginals vs thinned
samples).

## 7. Discussion notes (2026-09-12, to come back to)

**Spatial vs temporal reading.** The modality is the "NISQ-SPOQC" architecture as well: k
emitters, photon t of every emitter demultiplexed (without delay) into the t-th universal
2k-mode PIC, T stages detected independently. Space in place of time changes nothing in the
state: block-diagonal unitary U₁ ⊕ … ⊕ U_T, the k spins the only memory between stages, the
same chain (`U[n, m, m]` per stage), the same frontier figure with "bins" read as "stages".
The two families of the library are therefore: *spatial* (one unitary per shot — Fock input or
demultiplexed emitter; exponent n; SLOS + Clifford) and *temporal* (a register marched through
stages — loop register with exponent L, spin register with exponent k; linear in the number of
stages; the generic engines on the unrolled unitary as oracle and fallback).

**n* is a rate, not a wall.** n* = 418 bins at k = 7 is where one H100 drops to one sequence
per second; the cost is linear in n, so no length is out of reach — only slower. A device with
emitter trains through static PICs and full detection per stage is easy to simulate whatever
the train length and however entangled each train is (the spin entanglement along time is
exactly a 2^k bond). Hardness enters only through what interferes within one stage.

**Time-dependent settings** (a different unitary or spin gates per stage): one Kraus set per
stage instead of one shared set, i.e. one SLOS pass per stage streamed two at a time — a
constant factor (≤ 2× at k = 7, negligible below), no change of exponent. Same for the time
loop (per-bin settings are already the measured case).

**Feed-forward between stages** (settings of stage t chosen from earlier outcomes):
- sampling stays a chain: per-trajectory settings; finite menu → group shots by branch and
  reuse Kraus sets; continuous dependence → the S_bin weights for *this* ψ and *this* U are one
  bond-carrying SLOS pass with s₀ = ψ (χ × the k-photon Fock cost instead of χ², batched over
  shots as B unitaries), ~×4 at k = 7, still linear in T;
- a given record's probability stays a chain (the record fixes its settings);
- marginals / expectation values: Markov feed-forward (routing of stage t from the outcome of
  stage t−1, the NISQ-SPOQC drawing) → recursion on S_bin *conditional* register states
  σ_t(o) = Σ_{o'} K^{route(o')}_o σ_{t−1}(o') K†, S_bin² χ³ per stage, exact; bounded memory r
  → S_bin^r conditional states; full-history dependence → branch tree (a stage simulator's territory),
  estimate by sampling instead;
- no hardness in T: adaptivity is measurement-driven control of the k-spin register (a
  k-qubit measurement-based computation with photons as the entangling mechanism); cost
  2^k × the k-photon photonic factor per stage. Exact GPU reach ≈ k = 7 by enumeration,
  ≈ 10–12 with the χ > 2 chain-rule kernel; beyond, hard for the reason a k-qubit register is.

**What adds hardness: photons carried from a stage into the next** (undetected output modes
of PIC t routed into PIC t+1). The register becomes spins ⊗ carried Fock modes,
2^k · C(p+c, c) for c carried modes with ≤ p photons — the carried modes are the loop slots of
the ORCA analysis (Novák's R = 1 + Σℓ, our L). Limits: carry nothing → the spin chain; carry
everything → the full k·T-photon problem on the unrolled unitary (generic hardness, n = kT).
Rule of thumb from the loop frontier: exact sampling at ~1 Hz on one H100 while
2^k · C(p+c, c) ≲ 10⁷ (k = 4: ~8 carried modes with ≤ 8 photons; k = 7: ~4 modes, 5 photons),
and hardness needs the conjunction with ≳ 40 photons in play (else the unrolled Clifford
sampler settles it). With feed-forward on top, the carried modes are the photonic qubits that
are neither measured nor stored in a spin. Engine: spins ⊗ Fock register marched stage by
stage with fresh emissions injected — `LoopRegisterKernel` (one emitter + Fock register)
generalised with `SequenceKernel`'s product register; not built ("trains through loops").

**Planned extensions of `SequenceKernel`** (not built): per-shot / routed unitaries
(`route(outcome) -> unitary` with one Kraus set per route, branch-grouped sampler, the
conditional-state recursion for marginals; test against an independent stage simulator at
k = 2, T = 3); per-trajectory SLOS-pass weights for continuous feed-forward; streaming
K†K to reach k = 6 at m = 4k without the 73 GB peak.

## 8. Not covered / next

Delay lines in the interferometer (bins interfere): the register becomes spins ⊗ loop Fock
register — `LoopRegisterKernel` already takes one emitter; the product-register construction
here extends it to k. Feed-forward between bins (settings of bin b+1 chosen from the outcome of
bin b): the trajectory sampler supports it trivially (per-bin U), the marginals need the
branch tree, the job of a feed-forward stage simulator. Distinguishability, spin-side errors, per-mode loss: as
elsewhere.

## 9. Per-tick complexity of the register (static U, no stage): does it ramp?

Question (2026-09-13): with k emitters (k ≈ 30) and one static universal interferometer, the
spins are independent at the first tick — the bin is k independent single-photon mixtures, its
cost is the k-photon Fock cost (the spatial frontier, n* ≈ 38) — and at infinite time the
register is as entangled as it can be, cost 2ᵏ. How fast does the register get there, tick by
tick, and where is the frontier?

**Structure.** For a label-preserving emitter, ops[b, l] = |l⟩⟨l| G_b, the tick's Kraus
operator is diagonal in the spin-label basis after the gate:

    K_t = diag_x( ⟨t|U|x⟩ / √t! ) · (⊗_j G_b),    ⟨t|U|x⟩ = Perm(U_{rows(t), modes(x)}),

so the spin gates (product) create no entanglement and all of it comes from the diagonal
Perm(U_{t,x}) as a function of the label word x. Across a bipartition A|B of the emitters
the Laplace expansion of the permanent over which output photons go to A bounds its Schmidt
rank by C(k, |A|) — which is *larger* than the maximum 2^{min(|A|,|B|)} for a generic U. So:

* **Haar U (every emitter interferes with every other): the register is full-rank after the
  first click.** Measured (`bench/sequence_ticks.py`, CPU, k = 4, 6, 8): the exact Schmidt
  rank across every cut is at its maximum 2^{min} from tick 1 and stays there. There is no
  ramp; tick 1 costs like k photons, tick 2 onwards costs 2ᵏ.
* Full rank is not a flat spectrum. Middle-cut entanglement entropy after tick 1: 0.47, 0.47,
  0.65 of the maximum k/2 at k = 4, 6, 8, rising slowly over the next ticks (0.45, 0.56, 0.73
  at tick 4); bond dimension for 99.9 % of the weight 3.3 / 6.0 / 12.4 of 4 / 8 / 16. A
  truncated MPS buys a constant, not an exponent, and the fraction grows with k — to be
  measured at k = 10–16 on the GPU.
* **Local U (brickwork of range r): the ramp exists.** Range 1 (depth 2): rank 1 at every
  tick — the outcome fixes each photon's origin. Range 2 (depth 4), k = 6, middle cut: bond
  2, 3, 4, 5, 7 at ticks 1, 2, 3, 4, 6 (max 8): growth ≈ 2^{c·r·τ}, saturation after ≈ k/(2r)
  ticks; the per-tick cost with an MPS register of bond D is O(k D²) to draw x, the Fock
  cost for t, D²·2^{O(r)} for the update.

**Sampler.** `SequenceKernel.sample_diag(U, shots, hook=)` implements the label-diagonal
trajectory: φ = (⊗G)ψ, x ~ |φ(x)|², t ~ Fock(U, |x⟩) (Clifford GPU sampler, one circuit per
stream; strong Fock SLOS per distinct x on CPU), ψ ← Perm(U_{t,·}) ⊙ φ normalised —
`label_amplitudes` computes the 2ᵏ permanents of one pattern by the Laplace recurrence over
subsets of the k output photon positions, states (subset, label prefix), O(k 3ᵏ) and
2·C(k,k/2)·2^{k/2} complex per stream at the widest step (8 GB at k = 20). Verified against
the Kraus route: K_t ψ equal to 1e-15 for every t, bin marginals and a two-bin joint at the
statistical floor (tests/sequence). The hook receives the register after every tick
(`bench/sequence_ticks.py` records the middle-cut Schmidt spectrum and the tick's wall time).

**Measured on the H100** (run 20260913_193014, `results/keep/sequence_ticks_*_h100.csv`,
`img/sequence_ticks.png`; m = 2k, 256 shots, 8 at k = 18, 4 at k = 20).

![per-tick register complexity](../img/sequence_ticks.png)

* Haar U, k = 6..20: exact middle-cut rank at its maximum 2^{k/2} from tick 1 (k = 20: 1023.5
  of 1024 at tick 1, 1024 from tick 2). Entropy saturates by tick 3 at k/2 − ≈1 bit: deficit
  1.3, 1.1, 1.07, 1.03, 0.99, 0.99, 0.91, 0.97 bits for k = 6, 8, …, 20 (a Haar-random state
  sits at k/2 − 0.72). Bond for 99.9 % of the weight: 0.87–0.88 of 2^{k/2} at every k. So after
  the first click the register is a near-Page state: no MPS compression, exact or approximate.
* Band:2 U (depth-4 brickwork), k = 8, 12, 16, 16 ticks: exact rank grows ≈ +1 per tick
  (1.9 → 8.8 at k = 8, → 11.7 at k = 12, → 8.2 at k = 16 after 16 ticks) while the entropy
  stays at 0.1 bit and the 99.9 %-bond at 1.6–1.9: a slow ramp of an almost-product register.
  An MPS of bond ~4 would simulate this machine at any k for the ticks measured.
* Cost of a tick (independent of U): 0.014, 0.023, 0.075, 0.40, 3.2, 29, 273, 2640 ms per
  tick-shot at k = 6, 8, 10, 12, 14, 16, 18, 20 — ×9–10 per +2 in k from k = 12 on (k·3ᵏ;
  below that the Fock sampler and launch overheads dominate). Peak memory 3.5 GB at k = 18,
  29 GB at k = 20 (4 streams). The Laplace
  pass is a loop of index_adds (2.6·10¹⁰ complex multiply-adds per second at k = 18): a few ×
  are available from a fused kernel, not an exponent. *Update 2026-09-19 (§12, Glynn/GEMM
  label amplitudes): 0.17, 0.71, 3.6, 28 ms per tick-shot at k = 12, 14, 16, 18 — ×4–8 per +2,
  2.5× / 4.5× / 8× / 10× faster; extrapolated ≈ 0.2 s at k = 20, ≈ 2 s at k = 22.*

**Frontier (static universal U).** Tick 1: the k-photon Fock cost. From tick 2 the register:
2ᵏ·16 B per stream and k·3ᵏ per tick — measured 0.27 s per tick-shot at k = 18 and 2.6 s at
k = 20 (extrapolated ≈ 25 s at k = 22): **k ≈ 19–20 at 1 tick/s** on one H100 (memory is not
the limit: 7 GB per stream at k = 20, 16 MB of register), against k = 7 with the precomputed
Kraus set (§5). With the GEMM formulation of §12 (28 ms at k = 18, ≈ 0.2 s at 20, ≈ 2 s at
22) the frontier moves to **k ≈ 21–22**; with the sign sum collapsed over repeated rows (§12b, measured
2026-09-26: 78 ms at k = 20, 0.49 s at 22, 5.9 s at 24) it is bracketed at **k ≈ 22.6**. Equivalently: a static universal machine with 20 spin emitters is at the
frontier from its second click on, whatever n.

## 10. Feed-forward on the spin gates (2026-09-17)

`sample_diag(U, shots, policy=...)`: after bin b the callback `policy(b, counts, psi)` receives the
bin's patterns [S, m] and the updated register [S, χ] and returns the spin gates of bin b+1 per
stream, [S, k, d, d] (None = the emitters' own gates). Label-diagonal emitters only (ops = |l⟩⟨l| G,
d = L): the bin's Uop becomes the per-stream G (`_apply_local_batched`) and V the identity. The
classical outcome → gate loop costs nothing in the trajectory picture (same O(k 3ᵏ) per tick); the
strong side (`bin_marginals`, `kraus`, `sequence_probability`) takes no policy — an adaptive
sequence probability is a product of per-branch chains, `tests/sequence/test_feedforward.py`
enumerates the branches for k = 2, T = 3 and checks the sampler against them (TVD at the
statistical floor), and checks that a policy returning the emitters' own gates reproduces the
plain sampler shot for shot.

What it measured (`bench/esbs_design_plane.py --policy local|global`, candidate B of the
application study; CPU k = 6, 2048 shots, 4 settings, and a GPU sweep):
feeding the previous bin's clicks into the next bin's spin gates (per-emitter click count, or one
global parity bit, both shifting θ by π/2 multiples) changes neither the register entropy (Haar
1.45 → 1.50 / 1.54 bits at k = 6; band:2 0.24; band:4 0.62 → 0.64 / 0.66) nor the gate
dependence of the low-order inter-bin click correlators (ratio to shot noise 1.3 → 1.3 for Haar,
1.1 → 1.1 for band:2, 1.16 → 1.36–1.50 for band:4, within the noise of 4 settings). Reading: the
low-order photon statistics of a bin are Σ_x P(x) ⟨n_i⟩_x with ⟨n_i⟩_x = Σ_{c∈x}|U_ic|²; a Haar
U makes ⟨n_i⟩_x ≈ k/m for every label word x, so *no* reweighting of the labels — by the gates,
adaptive or not — is visible at low order; a local U keeps ⟨n_i⟩_x label-sensitive but its
register stays at bond ≲ few. Feed-forward on the gates therefore moves neither knob. What would:
feed-forward on the interferometer (routing, which changes ⟨n_i⟩_x itself), or measurement-induced
spin–spin entanglement with the emitter × time graph used as a 2D cluster (the SPOQC / MBQC
regime) — outside the static-U sampler this note is about.

## 11. Local optics as operations on the spin register: fusions, RUS, and a computer on the spins (2026-09-17)

For label-diagonal emitters the bin's Kraus operator is `K_t = V diag_x⟨t|U|x⟩ Uop` (§10): after
the spin gates, the optics act on the register as an operator *diagonal in the label basis*,
whose entries are the k-photon amplitudes of the detected pattern. Every photon being a copy of
its spin, a block of the interferometer that touches only the rails of a few emitters is a
few-spin POVM, and the standard photonic primitives read as follows (`bench/esbs_fusion.py`,
`tests/sequence/test_fusion.py` checks each one on the Kraus set):

* one photon measured in Z (identity on its rails): `diag(1, 0)` or `diag(0, 1)` — the spin
  collapses (a projective measurement of the spin in the basis its gate chose);
* one photon measured in X (45° splitter on its rails): `diag(1, ±1)/√2` — a Pauli byproduct,
  the spin keeps everything;
* type-II fusion of the photons of spins j, j+1 (PBS + 45°): coincidence (probability = the even
  parity weight) → `diag(1, 0, 0, ±1)/2`, the even-parity projector; bunching → a single label
  word survives, both spins collapse. A failed type-II fusion cuts both wires;
* 45° on both photons then type-II ("xfusion"): coincidence, probability 1/2 whatever the state
  → `diag(1, 0, 0, 1)/2` or `diag(0, 1, 1, 0)/2`, a ZZ parity **measurement** with both outcomes;
  bunching → `Z ⊗ I`, `I ⊗ Z` up to sign: a Pauli, nothing lost. This is the repeat-until-success
  entangling primitive of a spin-optical architecture, and in the engine it is just a 4 × 4 block
  of the bin's unitary plus a routing feed-forward (`policy` returning per-stream unitaries).

With those three operations the k spins are a universal register: a parity-measurement CNOT
between two data spins through an ancilla spin — ZZ(d₁, a) with a = |+⟩, XX(a, d₂) (H on both
before the emission, undone in the next gate), then a measured in Z — is a unitary of
operator-Schmidt rank 2 on (d₁, d₂) for every outcome (Beenakker et al. 2004; checked
numerically). `CircuitPolicy` runs a random circuit this way (RUS on both parity attempts,
random single-qubit gates per layer, two sublayers of gadgets), the sampler following every
stream's own program counter at the usual O(k 3ᵏ) per tick — feed-forward on the gates *and*
on the optics at once. Three programs are measured: `brick` (static brickwork of blocks, no
feed-forward), `rus` (routing feed-forward: attempt the sublattice's pairs until each succeeded),
`circuit`. Observable: the Schmidt entropy of the register across the chain's middle (data-
balanced for `circuit`) versus tick and k. CPU preview (k ≤ 7): type-II and Haar 4-mode
brickworks stay at 0.1–0.3 bits (wires cut / short-ranged measurement-only dynamics); xfusion
brickwork with random gates grows with T (0.5 → 1.0 bits at k = 6 in 20 ticks, bound 3); the
circuit reaches 1.37 of 2 bits at k = 7 after 3 layers (≈ 12 ticks per layer: two RUS attempts
of mean 2 ticks each, one measurement tick, per sublayer).

**Result (GPU run 20260918_084620, H100, `bench/esbs_fusion.py`; results/keep/esbs_fusion_*.csv,
img/esbs_fusion.png).** Register entropy at the cut, 256 trajectories, k = 4–16 (brick, rus: 40
ticks; circuit: k = 5–15, 120 ticks):

| program / optics | S at T (bits), k = 4 … 16 | S / S_max | reading |
|---|---|---|---|
| brick, type-II | 0.13, 0.02, 0.06, 0.00, 0.29, 0.00, 0.08 | ≤ 0.06 | wires cut at every failed fusion: product register |
| brick, Haar 4-mode | 0.37, 0.16, 0.11, 0.04, 0.32, 0.07, 0.18 | ≤ 0.05 | generic local optics: short-ranged measurement-only dynamics |
| brick, xfusion, H gates | 0.57, 1.00, 0.62, 1.04, 0.65, 1.01, 0.70 | 0.09–0.33 | Clifford: rank ≤ 2, one parity bit across the cut (k mod 4 alternation from the cut position) |
| brick, xfusion, random gates | 0.88, 0.98, 1.00, 0.97, 1.20, 1.11, 1.18 | 0.15–0.44 | measurement-only: ~1 bit whatever k, slow growth in T |
| rus, xfusion, random gates | 0.76, 1.10, 0.97, 1.16, 1.22, 1.39, 1.30 | 0.16–0.38 | routing feed-forward changes nothing: same plateau |
| circuit, xfusion | 0.50, 1.36, 1.35, 2.30, 2.33, 3.26 (k = 5 … 15; S_max = 1, 2, 2, 3, 3, 4) | 0.50–0.81 | Page value of the data cut (S_max − 0.72) after 8–12 layers, ≈ 12 ticks per layer |

The measurement-only dynamics of the RUS parity brickwork — with or without feed-forward, Clifford
or not — is area-law: about one bit across the cut at every k, S/S_max ∝ 1/k. The same primitive
*programmed* (ancilla resets turning parity measurements into unitary gates) is volume-law: the
data register reaches the Page value and its bond dimension is 2^{k_data/2} (16 at k = 15). So with
local optics the register is hard exactly when the machine is run as a computer, and then it is a
k-qubit computer: any simulator pays 2^k, and a qubit simulator with the same POVMs is cheaper
than this engine (the O(k 3^k) label-amplitude pass is wasted on block-diagonal optics — the
engine's own regime is the global interferometer). This is the SPOQC limit of the engine (de
Gliniasty et al., Quantum 8, 1423 (2024)): the primitives above are that architecture's, derived
here from the Kraus structure; the engine adds the ability to run them with photonic-level
imperfections (loss, distinguishability, multi-photon terms) inside the feed-forward loop, which a
qubit simulator cannot. Known inefficiency: `label_amplitudes` chunks streams to fit the full
Laplace expansion (3 streams per chunk at k = 16, k² launches each) — 29 ms per tick-shot at k = 16
on the H100 with a few percent GPU utilisation; walking the expansion step by step over all
streams would cut the memory k-fold and batch hundreds of streams per launch.

## 12. The 2^k label amplitudes as a GEMM: Glynn instead of Laplace (2026-09-18)

`label_amplitudes` (the register update of `sample_diag`: ⟨t|U|x⟩/√t! for every label word x)
was a Laplace expansion over the k output photon positions with states (subset of positions,
label prefix): O(k 3^k) operations per stream but a materialised state of up to
C(k, 2k/3)·2^{2k/3} complex entries (143 MB per stream at k = 16), so three streams fitted a 1 GB
chunk and every tick cost k² gather/scatter launches per chunk — 29 ms per tick-shot at k = 16 on
the H100 with a few percent utilisation (§11).

Glynn's formula turns the same computation into a batched matrix product. With the column sums
c_j(δ, l) = Σ_p δ_p U[r_p, modes[j][l]] of the stream's k output rows,

    ⟨t|U|x⟩ = 2^{1−k} Σ_{δ ∈ {±1}^k, δ_0 = 1} (Π_i δ_i) Π_j c_j(δ, x_j),

so the vector over x is a sum of 2^{k−1} outer products ⊗_j (c_j(δ, 0), c_j(δ, 1)). Split the
emitters in two halves, V_a[δ, x_a] = Π_{j<k/2} c_j(δ, x_j) and V_b[δ, x_b] = (Πδ) Π_{j≥k/2}
c_j(δ, x_j): the result is the [2^{k/2} × 2^{k/2}] matrix V_aᵀ V_b summed over δ — one batched
GEMM per block of D signs, the column sums themselves one small GEMM (Δ [D, k] × U rows
[k, 2k]), the half-products k/2 outer-product steps on [S, D, 2^{j}] tensors. 4^k/2 flops per
stream instead of k 3^k (3× more at k = 16) but at GEMM speed and with no state to gather: the
work space is D·(2k + 3·2^{k/2}) complex per stream (17 KB per (stream, sign) at k = 16), so a
4 GB chunk (the `sample_diag` default on cuda) holds all 256 streams with D = 1024, i.e.
32 sign blocks × ~20 launches per tick for the whole batch. Identical to the Laplace values to
1e-16 (tests/sequence: `test_label_amplitudes_glynn_matches_laplace`, with and without
per-stream unitaries); the Laplace recurrence stays as `method="laplace"`, itself rewritten as a
loop over the parents of each subset (one gather and one fused multiply-add per parent, no
scatter-add). CPU, 32 streams, one call: k = 14 2.0 s (Glynn) against 13.9 s (Laplace).

**GPU (job 11, run 20260919_084900, H100, xfusion brickwork, 256 shots, 40 ticks;
results/keep/esbs_fusion_glynn_xfusion_brick.csv):** 0.167 / 0.71 / 3.6 / 28 ms per tick-shot
at k = 12 / 14 / 16 / 18, against 0.41 / 3.2 / 29 / 273 with the Laplace pass — 2.5× / 4.5× /
8× / 10×, the gain growing with k as the GEMM takes over (×4–8 per two emitters, the 4^k
flops now visible). Entropies identical to job 9 at equal seeds (1.204 at k = 12). The
register frontier of §9 moves from k ≈ 19 to k ≈ 21–22 at one tick per second. Found on the
way: the hook's batched complex SVD (`torch.linalg.svdvals` on [256, 2^{k/2}, 2^{k/2}]) hit
cusolver's gesvdj non-convergence at k ≥ 14 and fell back to a path costing minutes per tick —
6 of the 6.0 h of job 11 and most of the 7.1 h of job 9 were that SVD, not the sampler; the
benches now take the Schmidt spectrum from `eigvalsh` of the smaller Gram matrix.

## 12b. The sign sum collapsed over repeated output rows (2026-09-26, cross-engine review item B5)

The GEMM of §12 still summed over all 2^{k−1} sign vectors although a bin's pattern t has repeated
output rows: a mode holding t_r photons contributes t_r identical rows to the Glynn sum, and the
Clifford & Clifford 2020 collapse that the Fock, emitter and NLBS kernels already use applies
verbatim. Over the t copies of a row the 2^t sign choices with v minus
signs are C(t, v) in number, contribute (−1)^v to the sign product and t − 2v to the column sum, so

    ⟨t|U|x⟩ = 2^{−k} Σ_v [Π_ν (−1)^{v_ν} C(t_ν, v_ν)] Π_j c_j(v, x_j),     c_j(v, l) = Σ_ν (t_ν − 2 v_ν) U[r_ν, modes_j[l]],

v_ν = 0..t_ν over the distinct rows ν: Π_ν (t_ν + 1) terms instead of 2^k, halved once more by the
complement v → t − v (the summand is invariant, since the sign and the product both pick up
(−1)^k) whenever some t_ν is odd — one odd row is put last with radix (t + 1)/2 and the sum is
doubled. Collision-free patterns give back exactly the 2^{k−1} terms of §12 with weights ±1. The
outer-product structure is untouched: the sign axis of V_a, V_b is now the collapsed term axis with
the binomial weight folded into V_b, so the collapse and the GEMM compose and the GEMM's cost is
linear in the number of terms.

**Expected gain, measured on the pattern statistics before implementing.** The term count depends
on the pattern's multiplicities only. On uniform multisets of k photons over m = 2k modes (the
Haar-averaged output law — Bose–Einstein), 2·10⁴ patterns each; the two rows in brackets are 64
patterns drawn by the Clifford sampler on a Haar U (Triton interpreter), which agree:

| k | 8 | 12 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|
| E[terms], collapsed | 62.6 (63.1) | 670 (654) | 7 182 | 23 537 | 76 784 | 251 777 | 816 517 |
| 2^{k−1} / E[terms] = work ratio | 2.04 | 3.05 | 4.56 | 5.57 | **6.83** | **8.33** | 10.3 |

CPU A/B of `label_amplitudes` (two cores, 32 streams, Bose–Einstein patterns, same values to 1e−17):
2.1× at k = 10, 3.7× at 12, 4.1× at 14, 4.8× at 16 (term ratio 5.3) — the GEMM shrinks with the
term count, minus the per-group overhead. On the H100 the §12 pass costs ×4–8 per +2 emitters, so a
7–8× term reduction at k = 20–22 is worth +1.4 to +2 emitters on the register frontier of §9
(k ≈ 21–22 at one tick per second). Measured below: an A/B at k = 12–18
against the uncollapsed sum kept as `method="glynn_full"`, then k = 20, 22, 24 collapsed.

**Measured (H100, run 20260926_100655; results/keep/sequence_ticks_collapse_ab_20260926_h100.csv).**
Same seeds, identical entropies and ranks row for row, so a pure timing A/B. ms per tick-shot:
0.162 → 0.176 at k = 12 (256 shots, 0.92×), 0.698 → 0.638 at 14 (1.09×), 3.64 → 2.65 at 16 (1.37×),
26.0 → 13.8 at 18 (8 shots, 1.89×); collapsed only: 78 at k = 20, 486 at 22 (4 shots), 5 893 at 24
(2 shots), against the §9 extrapolation of ≈ 200 and ≈ 2 000 at k = 20 and 22. The pass shrinks by the
term ratio, but the tick has a floor that is 2^k per stream and independent of the method — the
k-photon Fock sample per stream (Triton kernel), the label draw over 2^k words on the host, the two
register products — and at 256 shots that floor is most of the k = 12–16 tick, which is where the
gains sit far below the term ratio. **The register frontier of §9 is now bracketed rather than
extrapolated: 0.44–0.58 s at k = 22, 4.5–7.3 s at k = 24, i.e. k ≈ 22.6 at one tick-shot per
second** — about +1 emitter over the "k ≈ 21–22" of §9/§12, not the +1.4–2 the term ratio alone would
buy, because the floor now sets the tick. The next step is the floor, not the pass: split the tick,
run the Fock sample on the CUDA cofactor kernel (2.4–2.7× from n = 20), draw the
label word on the device. Both exist as options of `sample_diag` (`fock_backend="cuda"`, `draw="device"`;
they keep the law and change the trajectories at a given seed) with a `timing` dict that records the
synchronised seconds of every phase, `bench/sequence_ticks.py --split --fock-backend --draw`, and
`tests/sequence/check_label_amplitudes_gpu.py` is the device check (both passes against Laplace at k = 12,
the device draw's bin marginal).

**The floor, measured (run 20260926_201513; results/keep/sequence_ticks_floor_split_20260926_h100.csv).**
It was the host label draw: `torch.multinomial` over the 2^k label words of every stream on the CPU after
moving the weights over, 0.44 s of a 0.68 s tick at k = 16 with 256 shots. Drawn on the device it is 1 ms,
and the k = 16 tick goes 2.64 → 0.93 ms per tick-shot (3.9× against the §12 sum). The Fock sample's kernel
is irrelevant (8–22 ms per tick). From k = 20 the tick is the pass again: at k = 22 `amp` is 0.3–0.4 s per
stream against ≈ 4 TFLOP of GEMM per stream, within 1.5–2× of the FP64 GEMM rate, and the two register
products are the next 0.4 s; the frontier stays at k ≈ 22.6–22.7 (0.43 s per tick-shot at k = 22). Since
that run `sample_diag(draw="auto")` draws on the device on cuda (the trajectories at a given seed differ
from the earlier runs; the law is unchanged — the device check).

Implementation (`SequenceKernel._label_amplitudes_collapsed`, default of `label_amplitudes` and hence
of `sample_diag`): the multiplicity tables are those of `clifford.cofactors_triton.multiplicity_tables`,
rebuilt in plain torch (`_multiplicity_tables`) so that `sequence.py` does not import triton. Streams
are sorted by term count N_s and cut into groups whose counts lie within `pad_tol` (1.25) of each
other; a group is padded to its largest count with zero-weight terms (the sign matrix is per stream
now, `bmm` instead of the shared `matmul`). Same generator consumption as §12, so the trajectories
of `sample_diag` are unchanged at a given seed up to rounding (the tick bench records identical
entropies). Tests: `test_label_amplitudes_glynn_matches_laplace` (collapsed, uncollapsed and Laplace
agree, per-stream unitaries included) and `test_label_amplitudes_collapsed_sum_bunched_patterns`
(every multiplicity shape of k = 6, the term counts and the halving, all groupings and chunkings).

Pseudo-code, v1 (the §12 block is v0):

```
label_amplitudes(U, counts[S, m], modes[k, L]):
  rows[s]   = the k output rows of pattern s (repeats kept)
  (drow, t, radix, N, scale) = multiplicity_tables(rows)      # distinct rows, multiplicities, per-row radix
                                                              # (t+1; one odd row last with (t+1)/2), terms N_s,
                                                              # scale_s = 2^-k (x2 when halved)
  sort streams by N_s; cut into groups with N_max <= pad_tol N_min
  for each group (streams sid, N_max):
    A[s, nu, (j, l)] = U[drow[s, nu], modes[j, l]]             # distinct rows only
    acc[s] = 0  ([L^ka, L^kb], ka = k // 2)
    for each block of D term indices e in 0 .. N_max - 1:
      v[s, e, nu]   = (e // place[s, nu]) % radix[s, nu]       # mixed-radix digits, place = prod of later radices
      w[s, e]       = scale_s * prod_nu C(t_nu, v_nu) * (-1)^{sum v}   (0 if e >= N_s: padding)
      Delta[s, e, nu] = t_nu - 2 v_nu
      C[s, e, j, l] = (Delta @ A)[s, e, (j, l)]                 # column sums, one bmm
      V_a[s, e, x_a] = prod_{j < ka} C[s, e, j, x_j];  V_b[s, e, x_b] = w[s, e] prod_{j >= ka} C[s, e, j, x_j]
      acc[s] += V_a^T V_b                                       # one GEMM per block
    out[sid] = acc / sqrt(t!)
```

## 13. Partial distinguishability between the emitters (2026-09-20)

`SequenceKernel(..., indist=x | [x_1..x_k], order=b)` — docs/DISTINGUISHABILITY.md §3.4. Photons of
one emitter never meet (different bins, no delay line), so only the *inter-emitter*
distinguishability of a bin enters: emitter a's photon is in the common internal mode with
probability x_a, i.e. the Gram matrix S_ab = √(x_a x_b) — each dot's own purity. The bin's Kraus
set becomes the species-resolved set of the bin MPS (`EmitterSLOSKernel.kraus_species` with one
overlap per slot: Q ≥ S operators, `coarse(U, b)` mapping each to its pattern; Σ K†K = 1 still,
computed once per bin like the ideal set), `transfer` and `bin_marginals` are unchanged in form,
`sequence_probability` and `joint` march the register through outcome-restricted channels,
`sample` draws the refined operator and records its pattern. `sequence_amplitude` is undefined
(mixed output). `sample_diag` is extended (2026-09-21, `_tick_species`): per tick each emitter's
photon is lone with probability 1 − x_a; the label word is drawn as before, the interfering photons
are Fock-sampled together, each lone photon is routed by |U[r, modes[a][x_a]]|², and the register
update multiplies the good emitters' Glynn amplitudes (`label_amplitudes(emitters=)`) by the lone
photons' U factors — the operator stays diagonal in the label basis, so this is exact; streams are
grouped by their lone set (2^k patterns, dominated by the empty one at x ≈ 0.95). Feed-forward
policies (gates, routing) work unchanged. Checked against `bin_marginals` and `joint(0, 1)` of the
species-resolved Kraus set within shot noise. Checked against the species-resolved
permanent sum on the unrolled U ⊕ U of a two-emitter, two-bin chain to 1e-12
(tests/test_imperfections_register.py). A *deterministic* overlap between two dots (spectral
detuning, s_ab not of the form √(x_a x_b)) needs the Cholesky species basis of the design note and
is not implemented.

## 12c. Rounding error of the label amplitudes (2026-10-04, cross-engine review item #4 / Supplement C)

Three fp64 evaluations of the same 2^k label amplitudes ⟨t|U|x⟩ of a stream — the collapsed Glynn-GEMM sum of §12b
(production), the uncollapsed sum of §12 (`method="glynn_full"`) and the Laplace recurrence (`method="laplace"`, a
different algorithm with no sign cancellation, 19 GB per stream at k = 20) — and their pairwise disagreement, on the
amplitudes and on |a|² (what the tick's draw sees). `bench/sequence_precision.py`, run 20261004_223624, H100, 4 streams,
m = 2k, Haar U, Bose–Einstein-like output patterns; results/keep/sequence_precision_20261004_h100.csv:

| k | collapsed vs Laplace | collapsed vs uncollapsed | uncollapsed vs Laplace |
|---|---|---|---|
| 16 | 2.5·10⁻¹⁴ | 2.4·10⁻¹³ | 2.4·10⁻¹³ |
| 18 | 4.6·10⁻¹⁴ | 1.0·10⁻¹³ | 1.1·10⁻¹³ |
| 20 | 1.1·10⁻¹³ | 1.9·10⁻¹³ (1.1·10⁻¹³ on \|a\|²) | 1.7·10⁻¹³ |

The uncollapsed sum carries the error (2^(k−1) signed terms); the collapse removes most of it (2^(k−1)/E[terms] = 5–8 at
k = 16–20, §12b), leaving the collapsed pass 2.5·10⁻¹⁴ → 1.1·10⁻¹³ from the cancellation-free recurrence, ×1.4 per
emitter. The register frontier (k ≈ 22.6, §12b) carries ~2·10⁻¹³, 10¹⁰ below the 10⁻³ criterion of Supplement C, and the
errors do not accumulate across ticks because the register is renormalised at every tick (§4). No compensated
arithmetic is needed on this row.

## 14. Internal modes (temporal / frequency) — data model and dispatch (2026-10-20, in progress)

A photon may carry an internal mode index k ∈ [K] (a temporal wavepacket shape, a frequency) on which the interferometer
acts as the identity and which the detectors do not resolve: the network is U ⊗ 1_K, photons interfere only within the
same internal mode, and the outcome of a bin is the spatial pattern t with the internal index summed out. This is how a
spin-dependent wavepacket enters (the emitted photon's temporal shape depends on the spin), with K set by the physics.

**Data model.**
- `ModeBinEmitter(ops [n, L, K, d, d], modes, s0, final=None, lost=None)`: emitter j applies ops[b, l, k] when its
  photon of bin b leaves with label l in internal mode k; the emitted state is Σ_{l,k} ops[b,l,k]|s⟩ ⊗ b†_{modes[l],k}|0⟩.
  `lost` [n, Q, d, d] holds the Kraus operators of an emission that never reaches the network (spin-dependent loss as
  a Kraus split: Σ A†A + Σ N†N = 1); it is stored, not yet used.
- The K internal modes are a basis shared by every emitter of a kernel; an emitter with fewer modes (or a plain
  `BinEmitter`) uses the first ones.
- `bin_emitter(ops, modes, s0, final, lost)` returns a plain `BinEmitter` for ops [n, L, d, d] or [n, L, 1, d, d]
  without `lost` (the [:, :, 0] slice, a value copy), a `ModeBinEmitter` otherwise. `BinEmitter` and `EmitterMPS`
  are unchanged.

**Dispatch.** `SequenceKernel(emitters, m, …)` returns the subclass `ModeSequenceKernel` (`photonic_engines/sequence_modes.py`)
when, and only when, some emitter is a `ModeBinEmitter`; otherwise it is `SequenceKernel` itself, whose methods are
unchanged. Emitters with internal modes must come as a sequence (list, tuple, any `collections.abc.Sequence`); handed
over in a bare iterator, they cannot be dispatched before construction and the plain kernel refuses them (TypeError).
The only added cost for a problem without internal modes is two `isinstance` scans of the emitter list per
construction (about 1 µs). `ModeSequenceKernel` shares the spatial basis of a bin (`layers`, `patterns`, `S`,
`pattern_index`) and exposes `K` (internal modes; `k` stays the number of emitters) and `stage_ops(b)` (each emitter's
operators of bin b on the shared K modes, zero-padded). A test classifies every public `SequenceKernel` member as
refused, overridden or purely spatial, so a member added later cannot run on internal-mode emitters unnoticed.

**The channel of a bin (`kraus`, `coarse`; `photonic_engines/slos/internal.py`).** The channel of a spatial outcome t
is the sum over the fine patterns τ on the M = m K_b modes (spatial i, internal c) → c m + i whose sum over c is t:

    E_t(rho) = sum_{tau -> t} K_tau rho K_tau^dag,    K_tau = sqrt(tau!) <tau| Phi>

i.e. the bond-carrying SLOS recurrence of `slos.emitter` on the populated internal span K_b, written for the GPU:

```
kraus(U, b, compress=None):
    A_j = stage_ops(b)                                   # per emitter [L_j, K, d_j, d_j] on the shared K modes
    A_j = span(A_j, span_rtol)                           # drop all-zero internal modes (exact) or rotate to the
                                                         #   singular basis and keep s > span_rtol * s_max -> K_b
    A_j <- 1 (x) A_j (x) 1                               # on the product register chi
    psi[q] = e_q  (q = 1..chi)                           # bond vectors, one per initial register state
    for photon j (= emitter j):
        C[(c, i)] = sum_l U[i, in_j(l)] A_j[l, c]        # one einsum, [M, chi, chi]: no dense U (x) 1_K
        for chunks of raised modes (padded to the largest slab, < chunk_bytes):
            nxt[child] += C[r] @ psi[parent]             # batched gather / contract / index_add
    K_tau = sqrt(tau!) psi[:, :, tau]^T;  coarse[tau] = index of sum_c tau[c m + i]
    compress: per outcome, SVD of the stacked vec K_q (rows s_i Vh_i), keep s > rtol * s_max (batched by size)
```

Physical operators [Q, chi, chi] with the contract of the partial-distinguishability path: the register update of
pattern t is rho → Σ_{coarse[q] = t} K_q rho K_q^dag and Σ_q K_q^dag K_q = 1. Construction checks that every emitter is
trace-preserving (Σ_{l,k} A^dag A (+ Σ N^dag N) = 1 to 1e-10). Not yet supported, refused with
`NotImplementedError`: route R with `lost` operators, `order` (truncation at b distinguishable photons) with `indist`, masks other than Fock, and
the uncompressed fine set of ticks whose fine set exceeds `max_bytes` (default 4 GB; `outcome_law` / `condition` reach
them by the pairing route, `kraus(compress=rtol)` by route R below).
Cost ≈ C(m K_b + k − 1, k) / C(m + k − 1, k) × the one-mode Kraus build; memory S_fine chi² × 16 B (about 4× that
with `compress`, counted against `max_bytes`). Compression is the larger part of a compressed build at k = 3: one CPU
thread, k = 2, K = 8: build 0.6 ms, build + compress 1.5 ms (528 → 160 operators); k = 3, K = 8: 26 ms and 260 ms
(19 600 → 3 560) — a cached channel pays it once per tick shape and unitary.

**The register update (`outcome_law`, `condition`).** `outcome_law(U, b, rho)` gives p(t) = Tr E_t(rho) for every
pattern of bin b, `condition(U, b, rho, outcomes)` the unnormalised register E_C(rho) = Σ_{t∈C} E_t(rho) of an outcome
class C (an index, a list of indices or a boolean mask [S]); rho is a state [D], a density matrix [D, D] or a batch
[B, D, D]. Two routes:
- **F**: the fine Kraus set above, applied (`einsum`, then `index_add` per pattern). Cost ≈ build + S_fine B D³.
- **P′ (pairing)**: no Kraus set. Wick's theorem on the emitted monomials: ket photon j and bra photon π(j) share the
  row and the internal mode, so K enters only through the pair Gram Γ_jj′[l, l′] = Σ_c A_j[l, c] ⊗ conj A_j′[l′, c]
  (a d² × d² superoperator on (ket j, bra j′)). Glynn's form on both sides (event signs δ, bra-slot signs ε, first
  entries +1) leaves only bijections, also for the ordered, non-commuting slot actions:

```
pairing_apply(U, A [n, L, K_b, d, d], in_modes [n, L], rows [T, n], t! [T], rho [B, D, D]):
    Gamma[j, j', l, l'] = sum_c A_j[l, c] (x) conj A_j'[l', c]                 # [n, n, L, L, d^4], once per tick
    for chunks of outcomes (memory < chunk_bytes):
        c[t, delta, j, j', l, l'] = sum_e delta_e U[R_e, in_j(l)] conj U[R_e, in_j'(l')]    # one einsum
        G[t, delta, j, j'] = sum_{l, l'} c[t, delta, j, j', l, l'] Gamma[j, j', l, l']      # one einsum
        X[b, t, delta, eps] = rho[b]                                           # register tensor, 2n spin axes
        for photon j:  X <- sum_j' eps_j' G[t, delta, j, j'] acting on (ket j, bra j') of X   # n^2 batched einsums
        E_t(rho) = sum_{delta, eps} (prod delta)(prod eps) X / (t! 4^(n-1))
        cert_t   = c n^2 eps_mach sum_{delta, eps} |X|_F / (t! 4^(n-1))      # a-posteriori Frobenius error bound
```

  Cost per outcome 4^(n−1) terms × n² d² D² × B, independent of K and of the fine set. Bunched rows need no special case.
- **Certificate and fallback** (`tol`, default 1e-14): a result is accepted when √D × its certificate ≤ tol Tr rho
  (trace norm); otherwise it is recomputed by F, and if F exceeds `max_bytes` a `ValueError` names the certificate and
  the budget — a result above tol is never returned. The certificate is an a-posteriori estimate (term norms; the
  constant leaves out the accumulation lengths), with a measured margin ≥ 12× against complex64 runs at k = 2–4.
  A class is certified as a whole: the certificates of its
  outcomes add up and their sum is compared with tol Tr rho in Frobenius norm (which bounds the operator norm of the
  criterion; `outcome_law` keeps the √D trace-norm factor). Per outcome the certificate is ~1e-16 – 1e-15 (√D ×, fp64,
  k = 2–5), so a class of more than about tol / that many outcomes leaves P′ — F if it fits `max_bytes`, otherwise a
  `ValueError` naming the class size: pass a larger tol for large classes. Route F always sums a class directly, in one
contraction over the class's fine operators (no per-operator term is formed); a class that falls back from P′ takes the
same sum. By P′, a class of more than S/2 patterns is computed as the
  emission channel ⊗_j Φ_j(rho) (Φ_j = Σ_{l,c} A_j · A_j^dag, the sum of every outcome's channel — an identity that
  needs a unitary U, checked to 1e-13; otherwise the direct sum) minus its complement; the emission channel's own
  error bound (c n ε ‖⊗_j |Φ_j| (|rho|)‖_F) enters the class certificate, and above tol the class is recomputed
  directly by F.
- **Memory**: the outcome chunk is sized from `chunk_bytes`, but one outcome is always processed whole
  (≈ 48 · B · 4^(n−1) · d^(2n) bytes: 1.6 GB at n = 6, d = 2, B = 1), so the guard is soft from n = 6 or for large
  batches.
- **Gradients**: route F builds its Kraus set from a detached U (as `kraus`); P′ is differentiable in U. With
  `route="auto"` the dispatcher decides, so pass `route="P"` when d/dU is needed.
- **Dispatch** (`route="auto"`): per call, the cost model t_F = F0 + F_build n S_fine D³ (0 if cached) + F_apply S_fine
  B D³ |C|/S against t_P = P0 + P |C| 4^(n−1) n² d² D² B, with device constants in `sequence_modes.COST`; P′ is taken
  only when 2 t_P < t_F (the model's error must not become a regression, F being today's construction done natively).
  An uncalibrated device takes F unless the fine set exceeds `max_bytes` (then P′); emitters of unequal spin dimension
  take F. The fine set's size S_fine = C(m K_b + k − 1, k) is computed in closed form, so the dispatcher and the
  `max_bytes` refusals never build a fine stage that does not fit (its host tables scale as S_fine: at k = 6, K = 8,
  1.3·10⁹ fine patterns exhausted the host RAM of run 20261020_182236 before that change). The constants are
  calibrated for CUDA from `bench/internal_modes_bench.py` on the H100 (run 20261020_182236, below) and for the CPU on
  an Apple Silicon Mac (arm64, 10 cores, torch 2.14.1 with 8 threads; 2026-10-20; 9 points k = 2–4, K = 1, 3, 8;
  kept CSV `results/keep/internal_modes_bench_20261020_mac_cpu.csv`; the same relative least squares as for CUDA):
  F0 = 0.335 ms,
  F_build = 3.93·10⁻¹⁰, F_apply = 5.46·10⁻¹⁰, P0 = 0.375 ms, P = 1.61·10⁻⁹ s/unit. On the CPU the dispatcher takes
  the faster route at 8 of 9 points (k = 2, K = 8: F, 1.8× slower than P′, inside the factor-2 margin); P′ from
  k = 3, K = 8 (8.1 ms against 22 ms) and at k = 4, K = 8 (0.47 s against 93 s for a 3 GB fine set). The P model is
  up to 2.3× low at k = 3 (per-einsum overhead) and the F model 14× low at k = 4, K = 8; the constants are per
  machine, but `COST` is keyed by device type only: every CPU (another machine, another thread count) uses this
  Mac's entry. The pair ticks of a spin-photon processor keep F on the CPU: the regression bench gives 18/18 rows native faster,
  routes unchanged. Since the CPU is calibrated, `route="auto"` on the CPU takes P′ from k = 3, K = 8 — so whether
  the result is differentiable in U depends on the tick's size: pass `route="P"` when d/dU is needed (as on CUDA).

**Status.** On a `ModeSequenceKernel` every other method raises `NotImplementedError`, as does `bin_mps` (a bin
with internal modes has no EmitterMPS form on the spatial modes). The loop engines (`LoopRegisterKernel(emitter=)`,
`HybridRegisterKernel`) refuse a `ModeBinEmitter` with `NotImplementedError`.

**Partial distinguishability (`indist`, the x-model of docs/DISTINGUISHABILITY.md §3.4).** Photon j
is in the common internal state with amplitude √x_j and in a private one (orthogonal to every other photon's) with
√(1 − x_j), on top of its K internal modes — the Gram S_jj′ = √(x_j x_j′) of the species route. Two exact forms:

```
route F (and R's range samples):  A_j -> [ sqrt(x_j) A_j | 0 ... | sqrt(1 - x_j) span(A_j) (block j) | ... 0 ]
                                  K_eff = K_b + sum_j r_j  (r_j = 0 when x_j = 1); exact-zero fine operators dropped
route P' (and R's core):          Gamma_jj' <- sqrt(x_j x_j') Gamma_jj'  (j != j'),  Gamma_jj unchanged
```

The scaled pair Gram is the pair Gram of the extended emitters (the private blocks pair only with themselves), so P′'s
cost does not change with x; F's fine set grows with K_eff (k = 3, K = 8, x = 0.7 on the Mac CPU: S_fine 19 600 → 1.2·10⁶,
outcome law 0.11 s → 16 s by F, 75 ms by P′) — the dispatcher's cost model sees K_eff (with the CPU constants auto
takes P′ there: 9 ms; and at k = 3, K = 3, x = 0.7: 10 ms against 265 ms by F), an uncalibrated device keeps F
whenever it fits, so pass `route="P"` there. Most of that growth is structurally empty: the fine stage enumerates
patterns with two photons in one emitter's private block, whose operators are exact zeros and are dropped after the
build (the cost model counts them, as F really builds them). A fine stage that never enumerates them is a possible
later improvement. The emission channel (P′'s complement path) is unchanged by x.
Validation (tests/sequence/test_internal_modes_indist.py): at K = 1 (a `ModeBinEmitter` with one mode), F's private
modes and P′'s scaled Gram give the species-resolved channel of today's `SequenceKernel(indist=…)` per outcome
(superoperators ≤ 8.3·10⁻¹⁷, x = 0.8, 0 and per-emitter values, one emitter at x = 1); at K = 3, 8, P′ against F:
the law ≤ 4.2·10⁻¹⁶, classes incl. the complement path ≤ 9·10⁻¹⁶ on the Mac and 1.4·10⁻¹⁵ on the H100 host's CPU
(validation run 20261020_221250; tested against the criterion 10⁻¹⁴), R's compressed channel = F's; an oracle independent of
the private-mode derivation: the expanded construction (labels (l, c) on m K modes) through the plain kernel's
species route with the same indist, against F's superoperators and P′'s law (k = 2, K = 2, 3; < 10⁻¹⁴); x = 1 everywhere is the kernel
without indist (the same operators); x = 0 on a balanced beam splitter fills the HOM dip (coincidence 1/2 by F and P′);
completeness; CUDA against CPU (gpu-marked). `order` with internal modes is refused.

**The no-photon branch (`lost`, source efficiency).**
`ModeBinEmitter.lost` [n, Q, d, d] holds the Kraus operators of an emission that is never detected (spin-dependent
loss as a Kraus split, Σ A†A + Σ N†N = 1). A source efficiency η is the same split and needs no field:
`sequence_modes.source_loss(emitter, eta)` returns ops √η A and `lost` += √(1 − η) A[l, k] (every label and internal
mode), the `lost` set reduced per bin to its Choi rank (≤ d² operators, SVD of the stacked operators, the
absent-photon channel unchanged to ~10⁻²⁸ relative: F's presence branches multiply the absent emitters' `lost`
counts, L·K each before the reduction); η = 1 returns the emitter without `lost`. A uniform detection efficiency
moves to the sources the same way. With any `lost` operator a
bin's outcomes are the patterns of k, k − 1, …, 0 detected photons: `patterns` / `S` list the k-photon block first
(the lossless kernel's indices), then k − 1, …, and the all-zero pattern last; `coarse` and the outcome indices of
`outcome_law` / `condition` refer to that list.

```
route F (presence branches):  for every subset P of emitters whose photon is detected (absent ones need `lost`):
                                  K_P = fine set of the |P| photons (|P|-photon fine stage), coarse = its pattern
                                  K <- K_P * prod_{j not in P} N_j[q]   (on the absent emitters' own spins)
                              exact-zero operators dropped; compression per outcome as before
route P' (z-grading):         photon j:  X <- eps_j Lam_j(X)_[ket j, bra j]  +  z sum_j' eps_j' G_jj'(X)_[ket j, bra j']
                              Lam_j = sum_q N_q (x) conj N_q;  X carries the degree 0..|t| (higher dropped);
                              E_t = degree-|t| part / (t! 2^max(|t|-1,0) 2^(k-1));  |t| = 0: (x)_j Lam_j(rho)
emission channel:             Phi_j += sum_q N_q . N_q^dag (the complement path of `condition`)
```

Both Glynn sign symmetries (events and bra slots) hold at degree |t|; terms with more detected factors than events
survive the event-sign sum and are what the degree index removes. Route R is refused with `lost` (it needs z-graded
range samples). The dispatcher counts the fine sets of every presence subset for F and × (k + 1) for P′'s degree
axis — both with the lossless constants of `COST`: no lossy point is calibrated yet (one belongs in the next
calibration bench). Validation (tests/sequence/test_internal_modes_lost.py) against an oracle independent of both routes — the
lossless problem with one dummy spatial mode per (emitter, `lost` operator), the operator as a label emitting into
it, U ⊕ 1, through route F, the dummy occupations summed out: superoperators per outcome of 0..k photons ≤
2.1·10⁻¹⁷, the law by P′ and F ≤ 4.2·10⁻¹⁷ (k = 2, 3; K = 2, 3; Q = 1, 2; one emitter without `lost`); the k-photon
block keeps the lossless basis; the zero-photon outcome is ⊗_j Λ_j(ρ) by both routes (< 10⁻¹⁵); the conditioned
register at k = 3 (the first size where the event and bra-slot sign groups differ in length) by P′ and by F against
the oracle for one outcome of each photon number 0..3 and a mixed class, on a batch (< 10⁻¹⁴); P′ = F with `indist`
and `lost` (law < 10⁻¹⁵; single outcomes of every photon number, a mixed class and the complement < 10⁻¹⁴);
`source_loss`: η = 1 is the lossless kernel (the same operators), the Choi-rank reduction keeps the absent-photon
channel (< 10⁻¹⁵), completeness, η per bin, against the oracle at η = 0.7 and (0.9, 0.5); R refused; CUDA against CPU (gpu-marked). CPU (Mac): k = 3, K = 3, Q = 2: 2 390 operators,
F build 64 ms, P′ law 45 ms.

**A register larger than the kernel (`condition(..., rest=X)`).** A spin-photon processor's
register covers every spin; a tick's kernel sees only the emitting ones (D), the others (X) are entangled with them.
`condition(U, b, rho, outcomes, rest=X)` takes rho [D X, D X] (or a batch), kernel spins first, and returns
(E_C ⊗ id_X)(rho): the X² blocks rho[(·, x), (·, y)] go through the routes as one batch (route F: K_q per block; P′:
X² × the apply work), and the blocks' certificates of one register are combined in Frobenius norm (the error matrix of
the whole register) and compared with tol Tr rho of the whole register — passing the blocks as an ordinary batch would
compare each with its own trace, ~0 for an off-diagonal block, and force every one to F. The batch is X² × B blocks
(memory and apply work × X² — the dispatcher sees that batch size); a pure state [D X] is not accepted with `rest`
(pass its density matrix). Validation (tests/sequence/test_internal_modes_route_p.py): P′ and F against E_C applied
block by block, a single outcome, a class and the complement path, with and without `lost` (P′ < 10⁻¹⁴, the
criterion for class sums; F torch.equal), the pairing result kept.

**The weak route: `sample_apply` (R = 1, no `lost`; branches in the next paragraph).** One tick
of pure-state trajectories for a batch of streams, without a density matrix and without the fine set: the
operator-valued Clifford–Clifford chain ("CC-spin"). Recording the fine pattern (each event's row and internal mode)
is a valid unravelling of the coarse channel, so the detected-pattern law and the averaged post-state are exact; it
holds for generic emitters (physical quantum-dot emitters are not label-preserving once K > 1), with `indist`
through the private internal modes of the x-model above, and a passive rest X.

```
sample_apply(U, b, psi [S, D, X], generator):            # emitters complete, all photons detected: M = 1
    uniforms [S, k] <- generator                          # the draws do not depend on chunking or device
    v[{}] = psi
    for j = 1..k:                                         # one detected photon per step
        for every subset S of the k emitters with |S| = j (subset tables, chunked):
            y[S, i, l, c] = A_i[l, c] v[S - i]            # on spin i
            G_c[S] = <y[S, ., ., c] | y[S, ., ., c]>      # a jL x jL Gram per internal mode c
        W(row, c) = sum_S u_row^dag G_c[S] u_row,   u_row[(i, l)] = U[row, in_i(l)]
        check |sum W - (k - j + 1) sum_{|S'|=j-1} |v[S']|^2| <= tol_draw x that      # exact identity
        e* = (row*, c*): inverse CDF of W at uniforms[:, j]
        v[S] = sum_{i in S, l} U[row*, in_i(l)] y[S, i, l, c*]   for every |S| = j
    psi' = v[all] / |v[all]|;  pattern = the drawn rows (internal modes summed), index into `patterns`
```

- API: `sample_apply(U, b, psi, generator=None, tol_draw=1e-10, U_streams=None, chunk_bytes=None)` → (pattern index
  [S], psi′ [S, D, X] normalised, info: `events` [S, k, 2] (row, internal mode) in draw order, `deviation` [k], `prob`).
  psi [S, D] or [S, D, X], normalised per stream (checked to 10⁻¹⁰); `U_streams` [S, m, m] one unitary per stream; U
  must be unitary (≤ 10⁻¹³); detection loss goes through `eta_out` (below), not a sub-unitary U. Streams and subsets are chunked by `chunk_bytes`. The draws are made on the host
  from float64 weights: on CUDA the same seed gives the CPU's draws except for a rare flip at a weight boundary,
  where the CPU run is the reference.
- Cost per stream-tick: Σ_j C(k, j) · j · L · K_eff local applies of a D × X vector and the Grams — linear in the
  populated internal modes, memory C(k, j) · D · X per stream and step. Timings: the H100 paragraph below.
- Validation (tests/sequence/test_internal_modes_ccspin.py): T2, exact enumeration of every ordered chain (forced
  events, one stream each): Σ P(chain) ψ′ψ′† per detected pattern equals E_t(ψψ†) by route F and by P′ — measured
  ≤ 9.7·10⁻¹⁷ (k = 2, 3; K = 1, 2, 3; X = 1, 2; indist (0.7, 0.4); bunched outcomes), total probability 1 to 2·10⁻¹⁶;
  T3 (fixed seeds), the pattern histogram against `outcome_law`: k = 3, K = 3, 2·10⁴ streams χ² = 41.6 on 55 dof;
  k = 4, K = 8, 10⁵ streams (slow tier) χ² = 313.8 on 328 dof; the Monte-Carlo average of ψ′ψ′† on the likeliest
  outcome against `condition` / p; T4, the per-step identity ≤ 10⁻¹³ on every test (measured ≤ 10⁻¹⁵); same seed,
  same draws; stream chunking keeps the draws; `U_streams`; the unitarity, normalisation and `lost` refusals; T5, K = 1
  label-preserving emitters (A_l = |l⟩⟨l| G, k = 3): the enumerated chain equals the plain `SequenceKernel`'s channel
  per outcome (< 10⁻¹⁴); CUDA against CPU (gpu-marked). A complex64 probe: the law's TV error 0.2–0.5 ε, the identity check firing at 2.4–5× that. The plain `SequenceKernel`'s label route, `fock_sample` and `sample_diag` are unchanged (fingerprint
  38/38 bit-identical).

**The weak route with branches.** `sample_apply` covers every emitter of
the internal-mode kernel:
- **`lost` operators and R > 1 unobserved outcomes**: each emitter's branch — an unobserved outcome r, or a `lost`
  operator q — is an orthogonal environment record, drawn first, emitter by emitter at the amplitude level:
  b_j ~ ⟨φ|G_{j,b}|φ⟩/⟨φ|φ⟩ on spin j with G_{j,r} = M_{j,r} = Σ_{l,k} A_r†A_r and G_{j,q} = N_q†N_q, φ ← G_{j,b_j}^{1/2} φ
  (the conditionals multiply to ⟨ψ|⊗_j G_{j,b_j}|ψ⟩: operators on different spins commute); then
  ψ_b = ⊗_{lost j} N_{j,q_j} ψ, and the chain runs over the present photons with the operators A_j[·, ·, r_j] and,
  in every norm, the trace weight ⊗_{not yet detected} M_{j,r_j} of the photons still to come (no longer the
  identity). Streams are grouped by their branch word; info["branch"] [S, k] holds r, or R_j + q for a lost photon.
  Patterns of fewer than k photons index into the lossy `patterns` (the no-photon branch above).
- **A non-uniform detection efficiency**: `sample_apply(..., eta_out=[m])` dilates U to the isometry
  [[diag(√η) U], [diag(√(1−η)) U]] (2m candidate rows); events on the loss rows are not reported, and the pattern
  index refers to `patterns_all` (0..k detected photons; = `patterns` for a kernel with `lost`). A uniform
  efficiency stays at the sources (`source_loss`).
- Draws: 2k uniforms per stream (k for the branches, k for the chain); without branches the k of the complete case, so
  the draws of a complete kernel are unchanged.
- Outputs: info["occ"] [S, m], the detected occupations (also of the pattern index); `index=False` (or a basis whose
  key (k + 1)^m exceeds int64, e.g. k = 10, m = 20, where the pattern list itself is large) returns None as the index.
  info["events"] keeps the loss rows (m..2m−1) of `eta_out`. The branch step has its own identity check,
  Σ_b ⟨φ|G_{j,b}|φ⟩ = ⟨φ|φ⟩ (completeness), info["branch_deviation"], against `tol_draw` like the chain's. The
  unitarity tolerance of `sample_apply` and of `condition`'s complement path is 10³ ε of the kernel's dtype
  (2.2·10⁻¹³ in complex128), so a complex64 kernel is not refused. A complex64 probe with branches: total
  probability 1, TV 0.5 ε, the identity alarm at 3.6–4.6×.
- Validation (tests/sequence/test_internal_modes_ccspin_branches.py): T2, exact enumeration of every (branch word,
  chain) as forced streams: Σ P ψ′ψ′† per detected pattern (0..k photons) equals E_t(ψψ†) by route F and by P′ —
  measured ≤ 4.9·10⁻¹⁷ (k = 2, 3; K = 2; R = 1, 2; Q = 0, 1; rest X = 2; indist), total probability 1 to 2·10⁻¹⁶;
  eta_out (η = 0.9, 0.6, 0.75, 0.3): the enumeration over the dilated U equals route F on the problem with explicit
  loss modes (U completed to a unitary on 2m modes, the loss occupations summed out), < 10⁻¹⁴; T3 (fixed seed), k = 3,
  K = 2, R = 2, Q = 1, 2·10⁴ streams: χ² = 73.3 on 83 dof over the 0..3-photon patterns; the same seed gives the same
  branches and draws; T4 ≤ 10⁻¹³ (measured ≤ 6.3·10⁻¹⁶).

**The weak route on the H100.** `bench/internal_modes_ccspin_bench.py`, run 20261021_073232 on one H100 PCIe (a short
de-risk run 20261021_073057; the code validated on the GPU in run 20261021_072158); kept CSV
`results/keep/internal_modes_ccspin_20261021_h100.csv`. One `sample_apply` tick, k emitters with qubit spins on
m = 2k modes, generic emitters, K internal modes, S streams from one random state, median of 3 (`index=False`):

| k | K = 1: S = 1 / 100 / 10⁴ | K = 8: S = 1 / 100 / 10⁴ |
|---|---|---|
| 2 | 2.9 ms / 3.0 ms / 4.9 ms | 3.1 ms / 3.1 ms / 11 ms |
| 4 | 6.5 ms / 6.7 ms / 9.9 ms | 7.0 ms / 7.1 ms / 39 ms |
| 6 | 11 ms / 12 ms / 91 ms | 12 ms / 13 ms / 0.61 s |
| 8 | 18 ms / 23 ms / 2.0 s | 20 ms / 0.17 s / 16 s |
| 10 | 29 ms / 0.53 s / 52 s | 34 ms / 3.3 s / (predicted 332 s, not run) |
| 12 | 80 ms / 7.9 s / (predicted 792 s, not run) | 0.49 s / 49 s / (predicted 4 938 s, not run) |

Up to k ≈ 6 a tick is bound by its fixed per-call cost (Python steps over the subset tables, launches), so batching
the streams is what matters: 10⁴ streams cost about as much as one at k ≤ 4. From k ≈ 8 the arithmetic dominates and
grows as the model predicts, linearly in K: at k = 12, K = 8 a stream-tick costs 0.49 s against 0.08 s at K = 1.
The per-step identity stayed ≤ 1.2·10⁻¹⁵ on every row. Fitted (`COST["cuda"]`, 92 rows, relative least squares):
t = 2.96 ms + 2.02·10⁻⁴ s · k² + 1.54·10⁻¹¹ s · S · units, within 2.9× of every row (median 1.25×); CPU (the Mac,
k = 2–7, `results/keep/internal_modes_ccspin_20261021_mac_cpu.csv`): 0.56 ms + 7.3·10⁻⁵ s · k² + 1.05·10⁻⁹ s · S ·
units, within 3.0×. `sample_apply_cost(U, b, streams, X)` returns the prediction (None on an uncalibrated device)
for the measured case — all k photons present, R = 1; with `lost` or R > 1 the all-present units bound a stream's
arithmetic, but each distinct branch word of the batch is a separate chain call paying the per-call terms again.

**Coarse outcome classes.**
- `condition(U, b, rho, outcomes, weights=w)`: the weighted class E_w(rho) = Σ_t w_t E_t(rho), w ≥ 0 — one weight per
  entry of `outcomes` (repeated entries add up) or one per pattern [S] (restricted to the class), e.g. a
  detector-noise predecessor Σ_o T(o′|o) E_o(rho). Route F contracts the weighted operators in one pass; the pairing
  route sums the weighted results, with the certificate Σ_t w_t cert_t judged as for a class (and per register with
  `rest=`); a weighted class always takes the direct sum (the emission-channel complement is for 0/1 classes only).
- `kraus(U, b, compress=rtol, classes=c)` / `coarse(..., classes=c)`, c [S] an integer class id per pattern (−1:
  dropped): route F's operators grouped by class, `coarse` returning the class id; with `compress`, the existing
  batched SVD per class gives the class's Choi-rank set. Route F only (no route R per class). Threshold click patterns stay a class map on the caller's side.
- Validation (tests/sequence/test_internal_modes_classes.py), with and without `lost`: a weighted class equals the
  weighted sum of its members (F and P′, per-entry and per-pattern weights, with `rest=`; < 10⁻¹⁵); a mask equals 0/1
  weights; class superoperators equal the sum of the members' (< 10⁻¹⁵); the compressed count per class equals the rank
  of the stacked member operators, the same channel (< 10⁻¹⁴); dropped outcomes have no operator; validation errors.

**Unobserved emission outcomes (ops [n, L, K, R, d, d]).** An emission can leave the spin's
environment in one of R orthogonal states r (e.g. a spin flip after the emission): `ModeBinEmitter.ops[b, l, k, r]`,
completeness over (l, k, r) (+ `lost`). Different r never interfere (orthogonal environment states, not photon
modes: they cannot be folded into the internal modes), so the channel is the sum over the r-words (one r per
emitter) of the R = 1 channels of the slices `ops[:, :, :, r]`:
- route F: the fine set of every r-word (the |P| present emitters' words in the presence branches), concatenated;
  one internal span over every (emitter, r) slice, so K_eff does not depend on the word;
- route P′: the sum of the pairing results over the r-words (certificates add up); with `lost`, Λ_j / R_j, so that the
  sum over r_j of a term with photon j absent counts Λ_j once (r_j appears nowhere else in it);
- the emission channel (P′'s complement path) stacks every r along the label axis;
- route R refuses R > 1 (it needs r-word range samples).

Cost: × ∏_j R_j (R^k words) on both routes — the fine set has ∏_j R_j × S_fine operators and P′ evaluates
∏_j R_j pairing sums; the dispatcher counts both factors. ops [n, L, K, 1, d, d] is the plain form (the same operators,
torch.equal). Validation (tests/sequence/test_internal_modes_rwords.py) against an oracle independent of the r-word
sum — each spin dilated by an environment register of dimension R, A′ = Σ_r A_r ⊗ |r⟩⟨0| (an isometry on the
environment-|0⟩ subspace the register starts in), `lost` ⊗ 1, through the R = 1 path, the environments traced: the
conditioned register of one outcome per photon number and of a class over half the patterns (the complement), F and
P′, < 10⁻¹⁴ (k = 2, 3; K = 2, 3; R = 2, 3; with `lost` and with `indist`); completeness over r (a single r slice is
refused); R refused; `source_loss` keeps the r axis; CUDA against CPU (gpu-marked).

**Validation (tests/sequence/test_internal_modes.py, test_internal_modes_route_f.py).** One internal mode is
bit-identical to today: emitters built from [n, L, 1, d, d] give a plain `SequenceKernel` whose Kraus sets, sequence
probabilities and trajectories are `torch.equal` to those of today's emitters. The channel route:
- against the expanded construction (labels (l, c) on m K modes through the plain kernel, U ⊗ 1_K, outcomes summed
  over c), per outcome, superoperator difference / chi: 7·10⁻¹⁸ – 1.4·10⁻¹⁷ (k = 1–3 emitters, K = 2, 3, 8, unequal K,
  a plain emitter among internal-mode ones, per-bin unitaries); completeness ≤ 10⁻¹³;
- against a 50-digit evaluation of K_τ = (1/√τ!) Σ_σ Π_j O_j[R_σ(j), c_σ(j)], element by element: 6–7·10⁻¹⁷ (k = 2, 3);
- invariances: A → A V (V unitary on the internal space) leaves the channel unchanged (< 10⁻¹⁴); an all-zero internal
  mode is dropped bit-identically; a spin-independent shape (A[l, k] = v_k A[l]) with `span_rtol` is the plain channel
  (< 10⁻¹³);
- `compress=1e-12`: the same channel (< 10⁻¹⁴) with at most min(F_t, chi²) operators per outcome (k = 2, K = 8: 528 → 160);
- compression finds the exact rank of outcomes built with rank r = 1, 3, 4, 5, 9 (SVD of the stacked operators, not
  eigh of their Gram matrix, whose √ε floor would keep noise operators);
- chunked and unchunked propagation agree; CUDA against CPU (gpu-marked test; on CUDA the layer's index_add_ is not
  bitwise reproducible run to run, only to rounding).

The pairing route (tests/sequence/test_internal_modes_route_p.py), against route F:
- outcome law and conditioned register on batches of density matrices, < 10⁻¹⁵ (k = 1–3, K = 2, 3, 8, two bins,
  single outcomes, pairs and a class via the complement); the law sums to Tr rho (< 10⁻¹³);
- classes: a class equals the sum of its members, a mask its index list, a pure state its density matrix, the
  complement path the direct sum; bunched outcomes (t! = 2);
- the certificate is at least the measured distance to F and below 10⁻¹³ (k = 2–4); the emission channel equals the
  sum of every outcome's channel (< 10⁻¹⁴, within its certificate);
- `tol = 0` returns F's result exactly; a class whose summed certificate exceeds tol falls back as a whole; with F over
  `max_bytes` the fallback raises; a large class: route F and P′'s fallback are the direct F sum exactly, a
  non-unitary U takes the direct sum; k = 4, a class of S/2 outcomes with F over `max_bytes`: a tol below its summed
  certificate raises, above it the pairing result agrees with F (< 10⁻¹⁴); the dispatcher's three cases; CUDA against CPU (gpu-marked).

Timing: `bench/internal_modes_bench.py`. CPU (Mac, 2026-10-20, before the no-photon branch; outcome law of one random density
matrix, uncached, qubit spins, m = 2k; F's law takes traces without forming the matrices): P′ 0.38 ms / 8.8 ms /
0.47 s at k = 2 / 3 / 4 for every K = 1, 3, 8 (flat in K, as designed); F (build + apply) 0.27 / 0.30 / 0.71 ms
(k = 2, K = 1 / 3 / 8), 0.74 / 2.4 / 23 ms (k = 3), 3.5 / 116 ms (k = 4, K = 1 / 3); the expanded construction
0.78 / 2.1 ms (k = 2, K = 3 / 8), 5.4 / 34 ms (k = 3), 115 ms (k = 4, K = 3); k = 4, K = 8: the fine set needs
2.9 GB, P′ only. On the CPU, P′ wins from k = 3, K = 8 (8.9 ms against 23 ms for F, 34 ms expanded).

H100 PCIe (run 20261020_182236; validation run 20261020_182138: the 49 internal-mode tests pass on the GPU; kept CSV `results/keep/internal_modes_bench_20261020_h100.csv`;
same quantities as above, median of 5 reps at k = 2–4, 2 at k = 5–6):

| k | K = 1: F / P′ | K = 3: F / P′ / expanded | K = 8: F / P′ / expanded |
|---|---|---|---|
| 2 | 1.4 ms / 1.4 ms | 1.4 / 1.4 / 2.9 ms | 1.4 / 1.5 / 6.4 ms |
| 3 | 1.7 ms / 1.9 ms | 2.0 / 2.0 / 5.8 ms | 2.7 / 2.0 / 14 ms |
| 4 | 2.0 ms / 9.0 ms | 4.5 / 9.1 / 10 ms | 153 / 9.2 / 278 ms |
| 5 | 3.4 ms / 1.19 s | 309 ms / 1.19 s / 394 ms | F needs 471 GB / 1.19 s / — |
| 6 | 55 ms / 170 s | F needs 274 GB / 171 s / — | not measured (host OOM, fixed since) |

P′ is flat in K, as designed; below k = 4, F and P′ are both at the ~1.5–2.7 ms call overhead. The fitted constants (relative
least squares over the 14 points): F0 = 1.29 ms, F_build = 3.58·10⁻¹², F_apply = 2.50·10⁻¹², P0 = 1.60 ms,
P = 2.25·10⁻¹¹ s/unit. With them `route="auto"` takes P′ at k = 4, K = 8 (9.2 ms against 153 ms) and wherever F does
not fit, F elsewhere — the faster route at every point except k = 3, K = 3 (F 0.4 % slower) and K = 8 (F 2.7 ms
against P′ 2.0 ms), both inside the factor-2 margin the dispatcher keeps against the model's error. The calibration
covers one density matrix (B = 1) and the full outcome set: batches (B > 1) and classes rest on the model's linear
extrapolation and should be measured before a batched-register use relies on them; F0 and P0 are fits to the
overhead-bound points (k = 2–3), not device constants. At k = 4, K = 8 and
k = 5, K = 3 the expanded construction's time was measured but its channel comparison ran out of GPU memory.

**No regression on spin-photon processor ticks** (`bench/internal_modes_regression.py`). A spin-photon processor's
time-mode ticks are pair blocks: k = 2 dual-rail qubit emitters on m = 4 modes, K = 3 (bare trion), 4 (cavity),
7 (two Overhauser samples), 8 (the n-dot saturation), and unequal K (3:2, 8:7); generic
emitters (full Choi rank, the costliest case), Haar U, batches of B = 1, 16, 64 density matrices. Per tick, route
"auto" against the expanded construction: the compressed channel (`kraus(compress=1e-12)` against the expanded set
+ the same per-outcome compression), the outcome law, and the conditioning on a class of half the outcomes. CPU (Mac,
2026-10-20, median of 7): auto takes F on every row (the CPU was uncalibrated then; with the CPU constants in
`COST["cpu"]` it still takes F on every row, 18/18 native faster), and native is
faster than expanded on all 18 rows — channel 1.8–2.1×, law 1.3–3×, class 1.5–2.6× (closest row: law at K = 7,
B = 64, 2.6 ms against 3.4 ms); the two constructions agree to ≤ 1.1·10⁻¹⁶ (law and class, max abs). Before route F
summed a class in one contraction, the class at K = 8, B = 64 was level with expanded (3.6–3.7 ms against 3.4–3.6 ms).
The no-photon (`lost`) operators of physical emitters are not modelled in this bench. H100 PCIe (run 20261020_193633; the 50
internal-mode tests pass on the GPU first): auto
takes F on every row, native faster on all 18 — channel 1.4–1.7×, law 1.8–4.0×, class 1.8–3.8× (closest row:
channel at K = 8:7, 11.9 ms against 17.1 ms) — agreement ≤ 1.4·10⁻¹⁶; CSV `results/20261020_193633/internal_modes_regression.csv`.

**The compressed channel without the fine set (route R).** `kraus(U, b, compress=rtol, route="R")`
gives per outcome the same Choi-rank set as F's compression, at a cost set by the Choi rank r_t (≤ min(F_t, D²)),
not by the fine set (F_t = Π_i C(K_b + t_i − 1, t_i) fine patterns of t; S_fine = Σ_t F_t). Two pieces, both
independent of K beyond the pair Gram:

```
range_kraus(U, A, in_modes, rows, t!, F_t, rtol, tol):
    # range: vec K(w) lies in range(J_t), J_t = sum_tau vec K_tau vec K_tau^dag
    for chunks of outcomes, s = min(cap, 8) + p samples (p = 4), cap_t = min(F_t, D^2):
        w[t, s, e] ~ complex Gaussian in C^{K_b}                     # per event, host generator (seeded)
        O[t, s, j, e] = sum_l U[R_e, in_j(l)] sum_c w_e[c] A_j[l, c]   # one einsum
        K(w) = 2^-(n-1) sum_delta (prod delta) (x)_j sum_e delta_e O[j, e]   # Glynn, operator-valued permanent
        Q_t, sv = svd(stack of vec K(w));  r_t = #(sv > cut sv_max),  cut = tol / 8 in [8 eps, 1e3 eps]
        until r_t + p <= s or r_t = cap_t, doubling s
    # core: J_t = Q C Q^dag, C [r, r] from a^2 pairing applies
    per group of equal r:  a = ceil(2 r / D);  Y = Gaussian [D, a], or Y = 1 (matrix units) if a >= D
        Big[(i,u),(j,v)] = <u| E_t(y_i y_j^dag) |v>            # pairing_apply on the a^2 inputs, with certificates c
        A[(i,u), q] = (Q_q y_i)_u;  A = Qa Ra (thin QR)
        C = Ra^-1 (Qa^dag Big Qa) Ra^-dag, Hermitised;  residual = |Big - P_A Big P_A|_F
        err = (|c|_2 + residual) / sigma_min(A)^2              # Frobenius error of J_t
        lam, V = eigh(C); keep lam > rtol^2 lam_max and lam > err / sqrt(r)  (unresolved part added to err)
        K_t = sqrt(lam_i) Q V_i;  cert_t = err / D
    outcomes with cert_t > tol: again with Q = 1 (r = D^2, Y = 1: C = J_t from the matrix units)
    still above tol: failed -> route F for those outcomes (or ValueError when F exceeds max_bytes)
```

- **The range cut follows tol, not rtol.** A direction
  dropped from the range at relative size c moves J_t by about c (its cross terms with the kept part), not c², so
  the range is cut at tol / 8 (clamped to [8 ε, 1e3 ε]); rtol acts only at the eigenvalue step, as in F's
  compression. With rtol in the cut (first version), weak internal modes sent every outcome to the matrix units
  (k = 3, K = 3, rtol 1e-6: 3 640 applies against 3 584 for the matrix units); with the rounding floor 1e3 ε,
  dropped directions at ~2·10⁻¹³ still exceeded tol through the residual; with tol / 8: 1 915 applies, the residual at
  rounding level, the remaining retries from the pairing estimate's conditioning (probes a = ⌈2r / D⌉; 3r / D and
  4r / D cost more applies in total on every case measured).
- **Exactness and the error estimate.** The range is exact up to the cut; an incomplete range leaves a residual (the
  part of Big outside the sampled span), which enters the certificate and sends the outcome to the matrix units. The
  certificate is an a-posteriori **estimate**, not a bound: a missed direction q of J_t
  reaches the residual only through the random probe map, |P_A^⊥ M q|, which has no lower bound relative to its mass
  (with the matrix units, Y = 1, the map is an isometry and the residual is exact). Measured margin (CPU, k = 2–3,
  K = 3–8, generic and label-preserving emitters; the range samples projected off one eigen-direction of J_t, the
  strongest, a middle and the weakest resolved one, missed eigenvalue 2·10⁻⁵ – 1 of the largest; 396 cases):
  estimate / true error ≥ 0.58, median 3.5 — so a miss is detected once its error exceeds about 1.7 tol (the misses
  measured had errors ≫ tol and were all sent to the matrix units). The rank cut on the core follows
  `compress_by_outcome` (s_i = √λ_i above rtol s_max) and in addition drops eigenvalues at the core's error level (not
  resolved; their norm is added to the estimate). The tolerance assumes fp64: a complex64 kernel falls back.
- **Operators.** Same channel as F's compressed set, per outcome, within tol, but not the same operators (a unitary
  mixing of each outcome's set). The count is F's Choi-rank count when the spectrum's gap at rtol is above the core's
  absolute error (~10⁻¹⁵ λ_max; generic and low-rank emitters, every case measured). On a spread spectrum R returns
  fewer operators within tol: eigenvalues under that level are not resolved and are dropped (their norm enters the
  estimate). Weak internal modes (modes 2–4 scaled 10⁻⁴, 10⁻⁷, 10⁻⁹), rtol 1e-12: R 37 / 192 / 42 against F 68 / 671 / 84
  operators (k = 2 K = 3, k = 3 K = 3, k = 2 K = 4), |J_R − J_F| / χ ≤ 1.1·10⁻¹⁵; at rtol 1e-6 the counts are
  equal. The rounding of Q itself (the SVD of the samples) and of the reconstruction Q V √λ enters the estimate as
  c ε √r ‖C‖_F (~10⁻¹⁶ on J / D, but at D = 2 as large as the rest: without it, validation run 20261020_213551 measured 5.3·10⁻¹⁶ against an estimate of 4.6·10⁻¹⁶ on the H100 host's CPU). Deterministic for a given seed and
  `chunk_bytes` (the chunking orders the random draws; CPU and CUDA draw the same samples). Both stages are chunked
  over outcomes by `chunk_bytes` (the core per rank group); an outcome is processed whole.
- **Dispatch.** `route="auto"` (with `compress`) takes F when the compressed fine set fits `max_bytes` (4 × S_fine D²
  × 16 B), else R; R is not yet in the device cost model, and emitters of unequal spin dimension take F.
- **Validation** (tests/sequence/test_internal_modes_route_r.py), against F's compression, two bins: Choi matrices
  per outcome < 10⁻¹⁴ (measured ≤ 2.6·10⁻¹⁶ / D) with equal operator counts per outcome (k = 1–3, K = 3, 4, 8, generic
  and label-preserving emitters A[l, k] = f_l(k) |l⟩⟨l|, bunched outcomes); every certificate ≥ the measured error and
  < 10⁻¹⁴ (measured ≤ 5·10⁻¹⁵); label-preserving k = 2, K = 8: 64 fine operators, rank 4 per collision-free outcome;
  F_t in closed form equals the fine stage's counts; range samples confined to one direction, or projected off the
  weakest eigen-direction (k = 3, missed eigenvalue < 10⁻² of the largest; estimate ≥ ½ the true error): the
  residual sends the outcomes to the matrix units, channel still F's; a one-outcome chunk budget keeps the channel
  and the counts; weak internal modes (k = 3, K = 3, rtol 1e-6 and 1e-12): no failure, the estimate ≥ |J_R − J_F| / χ
  per outcome, at most F's count (equal at 1e-6), fewer pairing applies than the matrix units; U = 1: unreachable patterns get no operator, as in F; `tol = 0`:
  F's compressed set exactly, raises with F over `max_bytes`; auto takes R when F does not fit; CUDA against CPU
  (gpu-marked).
- **Timing** (CPU, Mac, 2026-10-20, build + compression, uncached, one run each, range cut at tol / 8): k = 3,
  K = 8, label-preserving: R 0.03 s, F 0.09 s (448 operators); k = 4, K = 2 generic: R 2.8 s, F 0.08 s (3 876
  operators: full rank, F wins at small K_b, as designed); k = 4, K = 3 label-preserving: R 1.7 s, F 0.70 s; k = 4,
  K = 8 label-preserving: R 1.8 s (5 280 operators), F would need 12.6 GB (4 × S_fine D² × 16 B, S_fine = 766 480).
- **H100 PCIe** (run 20261020_214216; validation run 20261020_214121: the 66 internal-mode tests pass on the GPU; kept CSV
  `results/keep/internal_modes_bench_r_20261020_h100.csv`). Compressed channel at rtol 1e-12, uncached, median of 3
  (2 at k = 5), F = build + compression of the fine set (`--max-gb 60`), R = route R; operators per tick in brackets
  (equal for F and R on every row where both ran):

  | k | K | generic: F / R | label-preserving: F / R |
  |---|---|---|---|
  | 2 | 3 | 2.6 / 7.4 ms (78) | 2.7 / 4.8 ms (40) |
  | 2 | 8 | 12 / 6.2 ms (160) | 9.8 / 4.9 ms (40) |
  | 3 | 3 | 68 / 242 ms (1 140) | 89 / 61 ms (442) |
  | 3 | 8 | 211 / 613 ms (3 560) | 145 / 59 ms (448) |
  | 4 | 3 | 1.08 / 5.81 s (17 550) | 1.30 / 0.80 s (5 240) |
  | 4 | 8 | 8.0 / 22.5 s (83 952) | 3.75 / 0.80 s (5 280) |
  | 5 | 3 | — | 35.5 / 15.8 s (63 904) |
  | 5 | 8 | — | F over 60 GB / 17.2 s (64 064) |

  |J_R − J_F| / χ ≤ 1.4·10⁻¹⁵ per outcome on every compared row (two rows, generic k = 4 K = 8 and label k = 5 K = 3,
  not compared: the bench's own superoperator check ran out of GPU memory). At full Choi rank (generic emitters) F
  is 2–5× faster from k = 3, as designed; at low Choi rank R is 1.5–4.7× faster from k = 3 and reaches k = 5, K = 8,
  where the fine set does not fit. `route="auto"` keeps F whenever the compressed fine set fits: R's cost follows the
  Choi rank, which is known only after its range step, so no a-priori `COST` term would choose it over a fitting F
  except by guessing the rank (a cheap rank probe is a possible later step).
