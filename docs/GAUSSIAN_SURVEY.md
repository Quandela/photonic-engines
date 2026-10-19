# Gaussian boson sampling: survey for a GBS modality (phase 0, 2026-09-23)

Design note, research only — no code has been written. It answers the phase-0
questions: which exact GBS algorithm to build, what to validate it
against, where the approximate algorithms and the real devices sit on our axis, and what the
imperfection and axis conventions become for squeezed sources. **It ends in a go/no-go; nothing
under `photonic_engines/` should change before that decision.**

Provenance tags used throughout: **[text]** read in the full text (arXiv HTML/ar5iv, journal
open access, or the source code); **[abs]** abstract or a secondary listing only; **[derived]**
our own arithmetic or derivation, shown in the note, not taken from a paper. Literature checked
2026-09-23. A number without a tag is a conversion of a tagged number, shown next to it.

## 0. Summary

- **Algorithm to build first**: the exact chain rule over conditional pure Gaussian states
  (Quesada et al. 2010.15595: virtual heterodyne of the unmeasured modes, so every conditional
  probability is an N×N loop hafnian) with Bulmer et al.'s collision-aware, batched loop hafnian
  (2108.01622). PNR first; threshold by Bulmer's fan-out trick, which reuses the same loop hafnian.
  Per-sample cost O(m N³ 2^{N/2}) in the collision-free limit, N = detected photons; collisions
  reduce 2^{N/2} to Π(η_i+1) terms, bounded below by Π√(n_i+1).
- **No faster exact algorithm since 2022** was found: no average-case, Clifford & Clifford-2020-type
  bound for hafnians; the only newer exact results are a recursive Torontonian (2109.04528),
  a bandwidth-limited loop hafnian (2311.05355, O(n w 2^w)) and a finite-field hafnian bound with no
  practical impact (2309.15422). Dated negative claim, §1.3.
- **GPU exact GBS sampler — the negative claim is now qualified.** One exists: GBSKernels
  (single author, PyPI 0.2.x, July 2026) ships a CUDA chain-rule sampler, limited to loop hafnians of
  size ≤ 20 and validated at TVD < 0.03 against The Walrus. DeepQuantum batches hafnians and
  Torontonians on GPU but samples by MCMC. Nothing exact on GPU at a size that matters is published.
  "First GPU exact GBS sampler" cannot be claimed unqualified; "first at experiment-relevant size,
  measured" can, if we build it (§1.3).
- **Oracles**: our own truncated `ProductSuperposition` through `ProductSLOSKernel` (independent
  derivation, exact for every pattern with N ≤ d−1 if the truncated amplitudes are *not*
  renormalised — to confirm in M1), The Walrus probabilities, and Piquasso as independent code.
  The Walrus's *sampler* is Bulmer's code, so it is provenance, not an oracle, for our sampler (§3).
- **Cost law, derived (§6.1)**: with input-side photon-number generating function Φ_src(z), the
  expected collision-free cost per source is Φ_src(1 − η + ηρ) — Fock Φ = z, SBS Φ = 1 − p + pz,
  squeezed vacuum Φ = sech r · (1 − z² tanh² r)^{−1/2} with ρ = √2. The last one **diverges for
  tanh r · (1 − η + η√2) ≥ 1** (lossless: sinh² r ≥ 1 photon per squeezer): above that the mean cost
  per sample is carried by collision-heavy tail events and depends on m/n. Jiuzhang 3.0 at high
  power and Jiuzhang 4.0 at r = 1.8 are above it; Borealis (r ≈ 1.1, η ≈ 0.33) is below.
- **Approximate literature points on our axis (§4)**: Dodd et al. 2511.14923 generate a
  Jiuzhang-3.0-sized sample in 6.9·10⁻⁴ s wall on 8 H100 (5.5·10⁻³ H100-s); Oh et al. 2306.03709
  in 3.7·10⁻⁴ s on 144 A100 (0.054 A100-s), Borealis-sized on 216-288 A100. Both sit orders of
  magnitude on the fast side of the 1 Hz line at sizes where every exact algorithm is out of reach.
  That is the sentence the paper needs, with numbers.
- **Borealis fix (§5.2)**: it is a 216-squeezed-mode, m/n = 1, PNR Gaussian machine with ~0.58
  detected photons per squeezed mode, not a Fock loop device with N ≈ 5 at L = 43. Place it on the
  GBS source axis and next to the Oh et al. point; keep the loop-ladder remark only as a pointer to
  the bandwidth-limited exact algorithm.
- **Effort (§8)**: M1-M2 (vocabulary + CPU exact sampler, validated) ≈ 7-10 agent-days; M3 (GPU
  loop hafnian in the chain-rule shape) ≈ 10-15 more; the whole of M1-M7 ≈ 5-7 weeks of agent time
  plus the user's GPU runs.

## 1. Exact sampling algorithms

### 1.1 The chain rule (Quesada et al. 2020; Bulmer et al. 2022)

- **Mixed → pure** [text, 2108.01622 and 2010.15595]: a mixed Gaussian state is an ensemble of pure
  Gaussian states with random displacement; draw α from the classical part, then sample the pure,
  displaced state. Loss therefore costs no algorithmic change (§6.1).
- **Mode by mode**: step j needs P(n_1..n_j) for all n_j ≤ n_cut. With the unmeasured modes
  virtually heterodyned (2010.15595), each is an N×N loop hafnian, N = n_1+…+n_j, instead of the
  2N×2N hafnian of the earlier polynomial-space algorithm (1908.08068, O(n³2ⁿ) per added mode).
  Bulmer computes the whole batch over n_j "with approximately the same run-time as calculating the
  largest loop hafnian" [text].
- **Collisions** [text]: the eigenvalue-trace loop hafnian is generalised to repeated pairings,
  "reducing the number of terms to Π_i(η_i+1), where η_i is the number of times a particular
  pairing of photons is repeated … lower-bounded by Π_i √(n_i+1) and upper-bounded by 2^{N/2}". The
  pairing is chosen greedily. The per-term polynomial prefactor is not stated in the paper; in the
  code each term is a power-trace evaluation, O(N³) (§2.2).
- **Threshold detection** [text]: each mode is fanned out to many PNR sub-detectors (12 in the
  paper), a click = any sub-detector fired; cost "from O(N_c³ 2^{N_c}) to O(N_c³ 2^{N_c/2})". So a
  threshold sampler is the same loop-hafnian machinery, not a Torontonian kernel.
- **MIS** [text]: Metropolis independence sampler, proposal = correct displacement then an
  "independent pairs and singles" distribution; exact only asymptotically; "τ_thin is likely to be
  in excess of 600" at M = 100 threshold. **Excluded** by our exactness convention.

**Published wall-clocks** [text]. HPE Cray EX, 1024 nodes × 2 AMD EPYC 7742 (64 cores) = 131,072
cores (the abstract says "∼100,000-core"). Haar U, M/4 two-mode squeezers, r = 1.55, η = 0.3 —
idealised Jiuzhang-1.0-type instances, not Borealis.

| case | measured | extrapolated mean | Fugaku extrapolation | notes |
|---|---|---|---|---|
| M = 60, PNR, cutoff 80 photons | 4200 samples in ~3 h = 2.6 s/sample | ~10 s/sample (photon numbers > 80 included) | 130 ms | 10 s × 131,072 cores = 1.3·10⁶ core-s per sample |
| M = 100, threshold, cutoff 60 clicks | 1600 samples in ~3.5 h = 7.9 s/sample | 8.4 s/sample | ~127 ms | implies ~73 days for Jiuzhang 1.0's 5·10⁷ samples |
| M = 100, PNR | one 92-photon event in 82 min | — | — | "impractically slow" |
| MIS step, M = 100 threshold | 0.45 s/step | — | 7 ms | ×τ_thin > 600 per independent sample |

Their "boundary": ~100 correlated clicks (threshold); ≥ 100 photons for PNR, depending on
collisions [text]. Note that the 130 ms is the 60-mode PNR figure, not the threshold one (the
brief had it the other way round).

### 1.2 Table — exact samplers

| method | exact? | per-sample cost | detector | displaced | largest published run | code |
|---|---|---|---|---|---|---|
| Quesada–Arrazola–Killoran 1807.01639 + Titan run 1810.00900 (Gupt, Arrazola, Quesada, Bromley) | yes | O(ℓ² 2^m), m = clicks, ℓ modes; memory exponential in m [text] | threshold (Torontonian) | no | 800 modes / 20 clicks, ~2 h per sample on 240,000 Titan cores (1.7·10⁹ core-s) [text] | The Walrus |
| Quesada & Arrazola 1908.08068, PRR 2, 023005 (2020) | yes | Σ over modes of O(n³2ⁿ); O(m²) space [text] | PNR, threshold | yes | — | The Walrus |
| Wu et al. 1908.10070, Sci. Bull. (2020) | yes | poly(n)·2^{8n/3}, or 2^{5n/2} with O(m2ⁿ) memory [text] | PNR | no | 20 photons in 10.5 h, 256 cores | not found |
| Quesada et al. 2010.15595, PRX Quantum (2022) | yes | O(m N³ 2^{N/2}) [text] | PNR (+ threshold) | yes | 50-photon probabilities on a workstation | The Walrus |
| **Bulmer et al. 2108.01622, Sci. Adv. 8, eabl9236 (2022)** | yes (chain rule); MIS asymptotic only | as above with 2^{N/2} → Π(η_i+1) terms; threshold O(N_c³ 2^{N_c/2}) [text] | PNR, threshold (fan-out) | yes | §1.1 | `jakeffbulmer/gbs`, merged into The Walrus |
| GBSKernels 0.2.x (2026) | yes (chain rule) | power-trace, batched | PNR | not stated | loop hafnian ≤ 20 on GPU; TVD < 0.03 vs The Walrus [PyPI/README] | Apache-2.0 |
| DeepQuantum 2512.18995 | MCMC (not exact) | — | PNR, threshold | — | batched primitives on A100 [text summary] | Apache-2.0 |

Also relevant, not samplers: Deshpande et al. 2102.12474 (Sci. Adv. 2022) fit a hafnian cost
t(n) = c·n³·2^{n/2}, c = 5.42·10⁻¹⁵ s on Niagara, Fugaku 122.8× faster, and estimate ≈ 4·10⁷ s per
sample on Fugaku for a lossy M = 216, r = 0.8, τ = 0.5 instance [text]; their tensor-network
figure extracted inconsistently (10²⁰ vs 4·10²⁰ s) — **not verified, do not quote**.

### 1.3 The dated negative claims (searched 2026-09-23)

1. **No multiplicity-collapse analogue of Clifford & Clifford 2020 for hafnians.** C&C 2020
   (2005.04214) gives ~O(n·1.69ⁿ) on average at m = n via Guan-code enumeration of repeated rows. For
   hafnians the only collision exploitation found is Bulmer's Π(η_i+1) term count and Kan's
   repeated-moment formula (§2); neither comes with an average-case bound over the output
   distribution. Queries: "hafnian repeated rows Guan/Gray code loop hafnian collisions",
   "faster classical GBS exact amortized marginal … Clifford analogue", "GBS exact sampling
   algorithm 2025 chain rule loop hafnian PNR". **This is an open algorithmic item on the same
   footing as SBS's 4^j → 2^j (docs/SUPERPOSITION.md §7.1)**, and a derived average-case ρ(c) for
   the GBS chain rule would be a small result in its own right.
2. **Newer exact results, none changing the exponent for the general case**: recursive Torontonian
   ∝ N^{1.0691} 2^{N/2} with N = 2·clicks against N^{2.74} 2^{N/2} (Kaposi et al. 2109.04528 [text],
   CPU TBB/MPI, 35-40 clicks); bandwidth-w loop hafnian O(n w 2^w) (2311.05355 [abs], authors not
   verified) — the one that matters for loop devices (§5.2); 2^{n−Ω(√n)} over finite fields (Li,
   2309.15422 [text summary], theory); displaced GBS complexity transition and a quasi-polynomial
   approximation at high displacement (Li, Solomons, Bulmer, Patel, Walmsley, npj QI 2025 [abs]) —
   no new exact sampler.
3. **GPU**: as in §0. Sunway TaihuLight computed one 100×100 Torontonian (a probability, not a
   sampler) in 20 h at 128-bit precision (2009.01177 [text summary]) — relevant as a precision
   warning. Piquasso 8.0's CUDA kernel is a *permanent* (Fock side). Every GPU GBS simulator at
   experiment scale found (Oh et al., Dodd et al., Jiuzhang 4.0's validation) is approximate.

## 2. Primitives

### 2.1 Table

| function / algorithm | cost | notes |
|---|---|---|
| hafnian, Björklund 2012 (1107.4466) | O*(2^{n/2}), polynomial space [abs] | the brief's O(n³2^{n/2}) is BGQ 2018, not Björklund 2012 |
| hafnian, Cygan–Pilipczuk (1302.3763; Inf. Comput. 243, 2015) | bounded average degree d: O*(2^{(1−ε_{2d}) n/2}), exponential space [abs] | sparse graphs only; The Walrus docs cite it as O(poly(n)2^{n/2}) poly space — attribution not checked in the paper body |
| hafnian and loop hafnian, Björklund–Gupt–Quesada power trace / eigenvalue (1805.12498, J. Exp. Algorithmics 24, 2019) | O(n³ 2^{n/2}) [abs] | Titan, matrices up to 56×56; 100×100 extrapolated to 288,000 CPUs × ~1.5 months [abs]; per-size timings not found |
| repeated-moment hafnian, Kan 2008 (J. Multivar. Anal. 99, 542) | ≈ n·A·Π(m_i+1), A = mean(m_i+1); O(n2ⁿ) without repeats [The Walrus docs] | `hafnian_repeated`; exact for repeated rows |
| collision-aware loop hafnian, Bulmer 2108.01622 | Π(η_i+1) terms ∈ [Π√(n_i+1), 2^{N/2}], O(N³) each [text + code] | `thewalrus.loop_hafnian(A, D, reps, glynn=True)`; Glynn variant halves the first radix |
| Torontonian, direct (1807.01639) | O(N³ 2^N), N = clicks [text] | |
| Torontonian, recursive (2109.04528) | ∝ N^{1.07} 2^{N/2}, N = 2·clicks [text] | reuses prefix Cholesky factors along a recursion tree |
| loop Torontonian (2202.04600, PRA 106, 043712) | Σ_Z ± exp(½γᵀ(I−O_Z)⁻¹γ*)/√det(I−O_Z); recursive variant [code] | displaced threshold; stated complexity not verified |
| click-counting "Kensingtonian" (2305.00853, PRA 109, 023708) | — | pseudo-PNR (Jiuzhang 3.0's 1:8 fan-out); authors not verified |

### 2.2 What our cofactor kernel design transfers, and what it does not

Read against `clifford/cofactors_triton.py` / `cuda_cofactors.py` (docs/ALGORITHMS.md) and
The Walrus's `loop_hafnian_batch.py` / `_hafnian.py` (read 2026-09-23).

Transfers:
- **The flat lane grid.** The 2^{N/2} (or Π(η_i+1)) loop-hafnian terms are independent and summed
  with signs, exactly like Glynn's terms; cutting them into contiguous segments, one flat grid over
  streams × segments, per-lane partials reduced hierarchically, is the same design. Streams need not
  share a matrix: each shot has its own displacement, which is the B axis our kernels already index.
- **Multiplicity-aware mixed-radix enumeration.** Bulmer's term index is decoded into kept-edge
  multiplicities with radices η_i+1 (first radix halved in the Glynn variant) — the structure of our
  2020 kernel (Kernel 1b), including the uniform-work-per-lane segmentation.
- **One pass, many candidates.** Bulmer batches every n_j ≤ n_cut of the current mode in one pass;
  our cofactors give all m candidate modes in one sweep. Same idea, different candidate axis
  (photon number of one mode, not the mode of one photon) — and the chain runs over m modes, not
  over n photons.
- **Precision discipline**: complex128, a double-double reference at large N. Sunway needed 128-bit
  for a 100×100 Torontonian; GBSKernels ships double-double and certified tiers.

Does not transfer:
- **The O(k) Gray-code update.** Our per-term cost is O(k) because consecutive Glynn terms differ
  by one sign and share column sums. A loop-hafnian term is a polynomial coefficient built from
  **power traces tr((AX_S)^i), i ≤ N/2** of an N×N matrix (plus the loop vector terms), O(N³) per
  term in The Walrus, rebuilt from scratch per term. A Gray step changes X_S by one diagonal entry, a
  rank-2 change of AX_S, and no known update gives all power traces in less than O(N²)–O(N³).
- **Thread-per-lane in registers.** An N×N complex matrix per lane does not fit in registers beyond
  N ≈ 10; the kernel becomes warp- or CTA-per-term with the matrix in shared memory, i.e. batched
  small dense linear algebra (Hessenberg reduction + characteristic polynomial, or repeated
  products). Arithmetic intensity is O(N) per byte instead of O(1), so the kernel is compute-bound —
  better for a GPU than Glynn is, but a different kernel.
- Hence **M3 is a new kernel, not a variant of ours**: the lane planner, multiplicity tables and
  reduction are reusable; the inner loop is not.

GPU primitive libraries: **GBSKernels** (Apache-2.0, 2026-07, one author, sizes ≤ 20, CUDA only as
a source build) — instructive for the precision tiers, not usable as a dependency or as an
independent oracle (validated against The Walrus). **DeepQuantum** (Apache-2.0, torch, active) —
usable as a GPU comparator for the primitives and as an independent third oracle; its sampler is
MCMC.

## 3. Open-source frameworks

Licences read from the LICENSE files; dates from PyPI and the last commit on the default branch
(repos cloned 2026-09-23).

| name | licence | computes | CPU / GPU | status | oracle for us? |
|---|---|---|---|---|---|
| The Walrus (Xanadu) | Apache-2.0 | hafnian (Glynn, incl-excl, recursive), loop hafnian, `hafnian_repeated`, `hafnian_batched`, `tor`, `ltor`, recursive tor, Montrealer; probabilities, density-matrix elements; samplers = Bulmer's chain rule (`hafnian_sample_state`, `torontonian_sample_state`) | CPU, numba (C++ removed Dec 2021); **no GPU** | 0.22.0 (2025-05-16), last commit 2025-11-14; numpy ≥ 2, Python 3.10-3.12 | **yes for probabilities** (reference); its sampler is Bulmer's code — provenance only |
| Strawberry Fields (Xanadu) | Apache-2.0 | Gaussian / Fock / bosonic / TF backends; `MeasureFock`, `MeasureThreshold` call The Walrus | CPU; GPU only via the TF Fock backend | **archived 2026-01-16** (banner confirmed); last PyPI 0.23.0 (2022-06-01) **fails to import with SciPy ≥ 1.14** (`simps` removed); git master (0.24.0-dev) runs with numpy 2.4 / SciPy 1.17 | cross-check only (wraps The Walrus). Citable: Killoran et al., Quantum 3, 129 (2019) |
| `jakeffbulmer/gbs` | Apache-2.0 (a LICENSE exists) | code of 2108.01622: loop hafnian (batch, gamma), chain rule, MIS (PNR, click), data for M60/M100 | CPU, numba over The Walrus | 11 commits, last 2021-11-09, dormant | same lineage as The Walrus — not independent |
| Piquasso (Budapest) | Apache-2.0 | `GaussianState`: `ParticleNumberMeasurement` (loop-hafnian chain rule, cutoff), `ThresholdMeasurement`, imperfect PNR, homodyne/heterodyne, density matrix; LXEB; also Fock and passive simulators | CPU: C++ (permanent, Torontonian, loop Torontonian) + numba hafnians. **GPU: Fock side only** — the 8.0 JAX-FFI CUDA kernel is a permanent; JAX `hafnian`/`loop_hafnian_batch` raise `NotImplementedError` | 8.0.1 (2026-07-01), last commit 2026-09-22, active. Cite Kolarovszki et al., Quantum 9, 1708 (2025) | **yes — best independent second oracle** (own code lineage, same algorithms) |
| MrMustard (Xanadu) | Apache-2.0 | differentiable Gaussian / Bargmann / Fock; `PNRSampler` = full Fock distribution to a cutoff then sample; no threshold sampler | numpy or JAX backend | 1.0.0a1 (2025-07-22, pre-release); 0.7.3 stable pins numpy < 2; last commit 2026-06-01 | partial: small-n Fock amplitudes by an independent Bargmann recurrence |
| DeepQuantum (TuringQ) | Apache-2.0 | Gaussian/Fock/bosonic backends; batched hafnian, loop hafnian (eigenvalue-trace), Torontonian, Kensingtonian; chain-rule or MCMC `measure`; MPS; **TDM delay-loop circuits** (`QumodeCircuitTDM`) | torch → CUDA | 4.5.0 (2026-04-24), last commit 2026-09-18, active | **yes — third independent oracle** and GPU comparator |
| BosonSupremacy (Liu, Oh) | no LICENSE file; `NOTICE.md` says MIT (own) + Apache-2.0 (part from The Walrus) | approximate MPS GBS (2306.03709) | CPU scripts; GPU via CuPy + custom CUDA + mpi4py, one GPU per mode | last commit 2024-08-01, dormant | **no** (approximate by design) |
| Perceval (Quandela) | MIT + exqalibur exception | Fock linear optics only (SLOS, SLAP, Naive, NaiveApprox, MPS, Clifford2017, Stepper) | CPU | 1.3.0 (2026-09-18), active | **no — no squeezed / Gaussian input at all** (grep: no "squeez", no "hafnian") |
| GBSKernels | Apache-2.0 (bundled USTC data: no stated licence) | batched perm / haf / loop haf / Torontonian, mpmath reference tier, certified bounds, chain-rule sampler | CPU pure Python; CUDA optional source build | 0.2.2 (2026-07-30), 24 commits in July 2026 | not independent (validated against The Walrus), unproven |
| BosonSampling.jl | MIT | `GaussianInput`, MIS PNR sampler after 2108.01622 | CPU, Julia | last commit 2023-06-06 | marginal |
| GBS-MCMC (Zhou et al., Nat. Commun. 2025) | MIT | Glauber-dynamics MCMC on graphs | CPU | 2026-03-18 | no |
| PennyLane-SF | Apache-2.0 | plugin to SF | — | pins pennylane < 0.30 (2023) | no |

**Licence flags.** No GPL/LGPL/AGPL anywhere. Nothing forbids redistribution of derived code.
Copying Apache-2.0 code into this repo requires keeping the licence text and copyright headers and
marking modified files (§4); BosonSupremacy's NOTICE must be reproduced if its code is derived from.
**Do not vendor exqalibur** (CC BY-NC-ND 4.0) or GBSKernels' Jiuzhang data. Our policy is to
reimplement the algorithm anyway ("algorithm vs algorithm", docs/ALGORITHMS.md), so
the flags matter for tests that *import* these packages as oracles, not for the engine.

**Oracle plan.** (a) Ours, independent: a single-mode squeezed vacuum truncated at d−1 photons is a
`ProductSuperposition` row a[q] = sech^{1/2} r · (−tanh r)^{q/2} √(q!)/(2^{q/2}(q/2)!) on even q
(phase convention to fix in M1); fed **unnormalised** to `ProductSLOSKernel.probs`, every output
pattern with N ≤ d−1 detected photons is exact, since no source configuration contributing to it is
truncated. This holds for the lossless distribution only (a pattern with N detected photons under loss
draws on inputs of any size), so the lossy oracle is the lossless oracle thinned by
`imperfections.thin_graded`, with d−1 set above the largest N checked plus a tail bound. Check in M1
that the kernel does not renormalise. Two-mode squeezing = two SMSV through
a 50:50 beam splitter, so it needs no new input type. (b) The Walrus `density_matrix_element` /
`probabilities` for loop hafnians, hafnians and Torontonians. (c) Piquasso, independent code.
Agreement target ≤ 1e-13 as in the brief.

## 4. Approximate algorithms — cited points for the axis

What "a sample" means differs per paper and has to travel with the number. None of these is an
exact sampler; none is to be implemented.

| paper | method | "a sample" | target | machine | reported | s / sample (wall) | device-s / sample |
|---|---|---|---|---|---|---|---|
| Oh, Liu, Alexeev, Fefferman, Jiang, 2306.03709, Nat. Phys. 20, 1461 (2024) | lossy covariance = pure squeezed part + classical noise; MPS of the pure part, χ = 10⁴; chain rule | approximate distribution; beats the experiment on 2-point and up to 5th-order correlations, 6th within error; XEB; TVD only at small size [text] | Jiuzhang 2.0 (P65-5), Jiuzhang 3.0 (high), Borealis M216 and M288 | Polaris, A100; one GPU per mode (144 for Jiuzhang, 216/288 for Borealis) [text] | build ≤ 9.5 min (Jiuzhang 2.0, others similar); 10⁷ samples in 62 min | 3.7·10⁻⁴ (sampling); 4.3·10⁻⁴ with the build | 0.054 A100-s (144 GPUs); 0.080 / 0.107 A100-s (216 / 288), assuming the same 62 min [the 62 min is stated generically — check per instance before quoting] |
| Dodd, Martínez-Cifuentes, Thomson Brown, Quesada, García-Patrón, 2511.14923 | click cumulants truncated at K = 5 from the ideal covariance; chain rule with a "double-elision" DP | reproduces click cumulants of order 2-5 better than the experiment; XEB matches or beats it at some total-click values (low brightness); no TVD [text] | Jiuzhang 2.0 (P65-2, P65-5), Jiuzhang 3.0 (144 modes, 1152 bins); not Borealis | 8× H100 SXM 80 GB, one node [text] | 10⁷ samples in 115 min; precompute 2 GB, apparently excluded from the 115 min | 6.9·10⁻⁴ | **5.5·10⁻³ H100-s** — the one point natively on our machine; 1 H100 ≈ 180 samples/s *if* scaling over the 8 GPUs is linear (not stated) |
| Villalonga et al., 2109.11525 | TAP mean-field Boltzmann machine from 1-2-mode marginals, O(N²L); greedy k-th-order marginal sampler | low-order marginals only; order-2/3 greedy samplers beat the experiment on marginal TVD/KL [text] | Jiuzhang 1.0, 2.0 | "a single workstation", unspecified | "millions of bit strings per minute" | ≲ 6·10⁻⁵ (order of magnitude) | — |
| Martínez-Cifuentes, Fonseca-Romero, Quesada, 2207.10058, Quantum 7, 1076 (2023) | squashed (classical) states as an alternative hypothesis | a hypothesis test, not a spoofer; Bayesian test favours squashed states on Jiuzhang 1.0, ground truth on 2.0 [text] | Jiuzhang 1.0, 2.0 | 64 cores | one 25-click probability ≈ 3.5 h | not a sampling time — do not place | — |
| Goodman, Dellios, Reid, Drummond, 2604.12330 | positive-P phase-space samples projected to counts (XQSIM); decoherence ε and transmission rescaling | grouped-count χ²/Z-scores, marginals up to 20th order (Jiuzhang), XEB on J2; claims parity with the experiment up to 1152 modes [text] | Jiuzhang 2, 3; Borealis | not stated | 10⁶-10⁷ samples; no wall-clock | **not placeable** | — |
| Park & Oh, 2510.24137 | MPS for lossy Fock / cat boson sampling and noisy IQP | resource estimates only [text] | **not GBS** | — | memory estimates only | — | — (drop from the GBS list; relevant to our Fock loss section instead) |

Also: Martínez-Cifuentes, de Guise, Quesada, 2403.15339 (PRX Quantum 5, 040312, 2024) is about
LXEB as a certificate, no spoofer timing. Theory threshold, not a point: Qi, Brod, Quesada,
García-Patrón, PRL 124, 100502 (2020) — efficient simulation when the surviving photons are fewer
than O(√N̄), stated as an inequality in (r, η, q_D, K) [text]; "η < 1/√N" is only a paraphrase.

**Authors' own extrapolations** (not measurements; quote as such): Jiuzhang 3.0 — ~600 years per
ideal sample with exact methods on Frontier, 3.1·10¹⁰ years for the hardest experimental sample
[abs]. Borealis — > 9,000 years per sample on Fugaku [text]. Jiuzhang 4.0 — > 10⁴² years *to build
the MPS* on El Capitan, a construction cost, not a per-sample cost [abs]. TFLN chip — ≥ 9.7·10⁶
years on one A100 from an MPS quantum/classical-split estimate at M = 40,000 [text].

**How to place them.** Our axis is seconds per sample on one H100 at 1 Hz. Only Dodd et al. is on an
H100; the others are converted as device-seconds per sample and labelled with the device. None is
exact, so they go on F2 as a separate marker class, "approximate, benchmark-matched", with the
benchmark named in the caption. This is what makes the sentence "an exact GBS frontier is not a
safety margin" concrete: at Jiuzhang-3.0 size the exact chain rule is out of reach (N ≳ 100 detected
photons, 2^{N/2} terms), while a K = 5 cumulant sampler runs at ~5.5 ms per H100-sample.

## 5. Physical models and devices

### 5.1 Models

- **Single-mode squeezed vacuum (SMSV)**: even photon numbers only, P(2k) = C(2k,k) 4^{−k} tanh^{2k} r
  / cosh r, ⟨n⟩ = sinh² r; generating function Φ(z) = sech r · (1 − z² tanh² r)^{−1/2} [derived;
  standard]. GBS probabilities are hafnians (Hamilton et al., PRL 119, 170501 (2017)).
- **Two-mode squeezing (TMSS)**: P(n, n) = tanh^{2n} r / cosh² r, each arm thermal; = two SMSV at
  the same r through a 50:50 beam splitter, so it is an interferometer choice, not a new input.
  Heralding one arm is the scattershot limit. Jiuzhang 1.0-3.0 use 25 TMSS; Jiuzhang 1.0 counts them
  as 50 SMSS [text].
- **Displacement** → loop hafnians (displacement on the diagonal). Comes for free: the chain rule
  already produces displaced conditional states.
- **Detectors**: PNR → (loop) hafnians; threshold → (loop) Torontonians, or Bulmer's fan-out loop
  hafnian; pseudo-PNR (click counting over a demultiplexer, Jiuzhang 3.0's 1:8) → Kensingtonian, or
  threshold detection on the fan-out bins.

### 5.2 Devices

Sources are counted as **squeezed modes** (our "sources"), with the physical squeezer count in the
last column. m/n and detected photons per source are [derived] from the tagged numbers.

| device | squeezed modes n | modes m | m/n | interferometer | r | transmission η | detector | photons / clicks | time per sample | physical squeezers; purity |
|---|---|---|---|---|---|---|---|---|---|---|
| Jiuzhang 1.0 (2012.01625, Science 370, 1460) | 50 (from 25 TMSS) | 100 | 2 | static, 97.7% | not extracted | collection 0.628, SNSPD 81%; overall not found | threshold SNSPD | max 76 clicks | not found (250 kHz rep.) | 25 PPKTP; purity 0.938 |
| Jiuzhang 2.0 (2106.15534, PRL 127, 180502) | 50 (25 TMSS) | 144 | 2.88 | static, phase-programmable inputs | not extracted | 48% / 54% (two waists); scope of a quoted 78% unclear | threshold SNSPD, 83% | max 113 clicks | not found | 25; purity 0.961 |
| Jiuzhang 3.0 (2304.12240, PRL 131, 150601) | 50 (25 TMSS) | 144 spatial, 1152 bins | 2.88 (23 on bins) | static 3D, 97% per mode | 1.2-1.6 | 43% overall | pseudo-PNR: SNSPD + 1:8 time demux | max 129 / 203 / 255 clicks at 0.72 / 1.0 / 1.3 W | 1.27 µs | 25; indistinguishability 96.2% |
| Borealis (Nature 606, 75 (2022)) | **216** time-bin squeezed modes | 216 | **1** | 3 programmable loops, delays τ, 6τ, 36τ | ~1.1 | ~33% | **PNR** TES, 1:16 demux, 95% | mean 125 (high), 21.1 (intermediate); max 219 | 36 µs headline; 10 kHz sampling rate (Methods) | **1 OPO** at 6 MHz; Schmidt number K = 1.12 |
| Jiuzhang 4.0 (2508.09092, Nature 653, 687 (2026)) | 1024 | 8176 hybrid space-time | ~8.0 | 3 cascaded 16-mode interferometers with delay loops, programmable | 0.9-1.8 | 51% system (source 92%) | threshold SNSPD ×16, 93% | up to 3050 photons | 25.6 µs | 4 OPOs; indistinguishability 97% at r = 1.8 |
| TFLN chip (Fu, Shen, Hu, He et al., 2609.11922, 2026-09-10) | not found | 128 validated; 40,000 in the advantage estimate | — | monolithic TFLN, MZI mesh + 2 delay loops, 4 GHz | not found | losses given per element (2 dB/τ, 2 dB fibre, 5 dB spatial, 0.9 dB/facet), not overall | SNSPD; PNR or threshold not found | "11,059 photons within 1 ms" — not clearly one sample | not found | not found |

Cross-check [derived]: Borealis η sinh² r = 0.33 × sinh²(1.1) = 0.59 detected photons per squeezed
mode, against the reported mean 125/216 = 0.58. The numbers are consistent.

**Borealis, the concrete fix.** `docs/TIME_LOOP.md` §4 places it through
its delay ladder as a Fock loop device — (1, 6, 36), L = 43, N* ≈ 5, "simulable to N ≈ 60-300
photons". That is a statement about single photons injected into Borealis' loops, not about
Borealis. On our axis Borealis is n = 216 sources, m/n = 1, PNR, ~0.58 detected photons per source,
N ≈ 125 per sample. For the paper: (i) remove it from the Fock loop figure, or keep it there
explicitly labelled "hypothetical Fock input into Borealis' ladder"; (ii) place it on the GBS source
axis with its exact status stated as a cost order (collision-free 2^{N/2} with N ≈ 125, far beyond
the 1 Hz line — no timing, since no exact run exists) and next to the Oh et al. approximate point
(3.7·10⁻⁴ s on 216 A100); (iii) keep the ladder as the open exact-algorithm question: a
bandwidth-limited loop hafnian costs 2^w, not 2^{N/2} (2311.05355), and Deshpande et al. chose the
ladder to defeat local simulability — whether a register-march analogue of our LBS engine exists for
Gaussian inputs is open (§9).

## 6. Imperfections in the Gaussian formalism

### 6.1 Uniform loss, and the generalised cost law

Loss is the Gaussian channel V → ηV + (1−η)·𝟙/2 (vacuum = 𝟙/2 in ħ = 1 units): O(m²) on the
covariance, exact, commuting with U as in `docs/LOSS.md` §1. It is **cheaper than in the Fock case,
not dearer**, for three reasons: the channel keeps the state Gaussian, so there is no per-shot subset
draw (`split_shots`) and no unravelling (emitter §3b); the chain rule already treats any mixed state
by drawing a displacement, so a lossy state costs the same algorithm; and loss lowers N, which is
what the cost depends on.

**Cost law [derived].** A shot with N detected photons costs ≈ c(N)·ρ^N in the collision-free model,
ρ = √2 for the loop-hafnian chain rule (2^{N/2}), ρ ≈ 1.8-2 for the Fock chain rule. With independent
sources, each with an input photon-number generating function Φ_src, and uniform loss (binomial
thinning, z → 1 − η + ηz), the expected cost factorises per source:

    E[ρ^N] = Π_sources Φ_src(1 − η + ηρ)

| source | Φ_src(z) | per-source base |
|---|---|---|
| Fock | z | 1 − η + ηρ — the E4 law of docs/LOSS.md §7 |
| SBS, brightness p | 1 − p + pz | 1 − p + p(1 − η + ηρ) — docs/LOSS.md §5b |
| SMSV, squeezing r | sech r · (1 − z² tanh² r)^{−1/2} | sech r · (1 − (1 − η + η√2)² tanh² r)^{−1/2} |

The existing law is the first row of a single statement: the source's generating function evaluated
at 1 − η + ηρ. For SMSV (checked against the series to 1e-15):

| mean photons per squeezer sinh² r | 0.1 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|
| η = 1 | 1.054 | 1.155 | 1.414 | 2.000 | ∞ | ∞ | ∞ | ∞ |
| η = 0.7 | 1.035 | 1.095 | 1.224 | 1.411 | 1.725 | 15.7 | ∞ | ∞ |
| η = 0.5 | 1.024 | 1.063 | 1.139 | 1.234 | 1.357 | 1.784 | 3.414 | ∞ |
| η = 0.33 | 1.015 | 1.039 | 1.082 | 1.132 | 1.189 | 1.334 | 1.551 | 2.842 |

**It diverges for tanh r · (1 − η + η√2) ≥ 1.** Lossless, that is sinh² r ≥ 1: one photon per
squeezer on average (r_crit = 0.881). The critical mean photon number is 1.13 / 1.51 / 2.19 / 3.42 at
η = 0.9 / 0.7 / 0.5 / 0.33. The photon-number tail of a squeezer is geometric in pairs (ratio
tanh² r) while the collision-free cost doubles per pair. Past that point the **real** expected cost
is finite only because collisions cap the term count (Π√(n_i+1), polynomial in N at fixed m), so it
is carried by rare, collision-heavy, high-N shots and **depends strongly on m/n**. At the device
settings: Borealis (r 1.1, η 0.33) 1.44 per source, finite; Jiuzhang 4.0 at r = 0.9 1.40; Jiuzhang 3.0
at r = 1.2 2.94; Jiuzhang 3.0 at r = 1.6 and Jiuzhang 4.0 at r = 1.8 are past the divergence. For
threshold detection the question does not arise (N_c ≤ m).

Consequences for the engine, stated before anything is measured:
- the frontier statistic must be the mean over *unconditioned* shots with **no global photon cutoff**
  (Bulmer's measured 2.6 s at an 80-photon cutoff against the extrapolated 10 s at M = 60 is this
  effect at work). A cutoff is
  post-selection by another name. The bench must report the tail share of the cost, as the SBS and
  NLBS runs did;
- the SBS precedent ("rare high-k shots carry 84% of E[cost]", docs/SUPERPOSITION.md §5) will be
  stronger here;
- an average-case term count under collisions — the GBS ρ(c) — is what decides n*(m/n) above
  r_crit, and it is not in the literature (§1.3).

### 6.2 Partial distinguishability

The physical model is spectral: a squeezer emits into Schmidt modes k with weights λ_k (Σλ_k = 1,
purity Σλ_k² = 1/K). Photons in different Schmidt modes do not interfere, and each Schmidt mode is
an independent squeezer through the same U. So [derived, to verify in M4] the state is a **product
of K independent GBS instances** (squeezings r_k, with r_k ≈ r√λ_k at low gain — gain-dependent
beyond that, to check), and the output counts are the sum of K independent samples. An exact sampler
draws each instance and adds the counts: cost Σ_k cost(instance k), dominated by the leading Schmidt
mode.

How our vocabulary maps:
- The x-model's "bad photon" (independent per photon, classically routed, `docs/DISTINGUISHABILITY.md`
  §0) has **no Gaussian counterpart**: minor-Schmidt-mode photons are not routed classically, they
  interfere among themselves as a smaller GBS instance.
- **η → ηx does not hold literally.** The analogue is "a photon costs only if detected and in the
  leading Schmidt mode", with x replaced by the leading mode's share. At low gain that share is λ₁
  of the pairs; at high gain it is the ratio sinh²(r_1)/Σ sinh²(r_k). The law to measure is
  n*(η, K) = log budget / log Φ_{r_1}(1 − η + η√2), i.e. the frontier of the leading instance.
- Shi & Byrnes (npj QI 8, 54 (2022)) use a different, virtual-mode model: a beam splitter
  cos²θ = η_ind couples each input to a virtual mode whose photons are detected but do not interfere.
  That split entangles the good and virtual parts of a squeezed state, so it is not the product
  above. Pick one model and say which; the Schmidt model is the one the devices quote (Borealis
  K = 1.12; Jiuzhang purities 0.938-0.97).
- The Perceval mapping `x = √V_HOM` has no direct analogue; V_HOM between two heralded squeezers
  is a function of the purity, and that relation would be the input-side calibration.

### 6.3 Dark counts

Out of the first build, as for the other engines (`docs/LOSS.md` §6). Note for later: thermal
background noise is a Gaussian channel (V → V + n_th·𝟙) and costs nothing to model on the
covariance. Dark counts of a threshold detector are a classical post-processing. Either can be added
in M4 at no algorithmic cost.

## 7. Axis and knobs

Convention kept: **n = squeezed modes (sources)**, m = c·n, no post-selection, including no photon
cutoff. TMSS devices count 2 per pair; time-multiplexed devices count squeezed time bins (Borealis
216, not 1). The extra knob is r, and §6.1 shows it is not a nuisance parameter: it has a critical
value where the character of the cost changes. Proposal, keeping it to curves:

1. **Headline curve n*(m/n) at fixed ⟨N⟩/n = η sinh² r = 0.5 detected photons per source**, η = 1
   (r = 0.658), same nine ratios as the Fock table. 0.5 sits close to Borealis (0.58) and Jiuzhang
   4.0 at r = 0.9 (0.54), is below r_crit, and equals SBS's photon yield at α = π/4. That makes
   the **SBS row the natural comparison**: same yield per source, coherent superposition of 0/1 photon
   vs squeezed vacuum.
2. **One r scan at m = 2n**, sinh² r ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 1.2, 1.5}, crossing r_crit, with
   the tail share reported. The scan is where the collision-dominated regime shows up, and where
   n*(m/n) is predicted to steepen.
3. **Loss curve at r = 0.658**, η = 1 / 0.9 / 0.8 / 0.7 / 0.5, against the §6.1 prediction — the
   generalised-law check, the analogue of `img/frontier_loss.png`.
4. Device points as markers, not curves (Borealis at c = 1, PNR; Jiuzhang with threshold detection
   once M5 exists). No surface.

**Memory frontier** (`bench/memory_frontier.py`): with no fixed photon number the table is the graded
basis to a cutoff N_cut, C(m + N_cut, N_cut) states, and N_cut has to be chosen by captured mass,
P(N > N_cut) ≤ ε, from the known total photon-number law (a convolution of the per-squeezer laws
after thinning; U-independent, like SBS's binomial). Report n_mem(ε) at ε = 10⁻³ and state that it
approximates the distribution — or report it under the SBS convention (every photon number up to
2·n_max per source) and say which. Counting only, no benchmark.

**How the numbers must be reported** (the brief's second point): every GBS n* carries the tag
"exact, PNR, one H100". The same figure carries the approximate points of §4 at their device sizes,
so the exact frontier can never be read as a safety margin. The paper's sentence is then a caption,
not a caveat.

## 8. Recommendation and effort

**Build first**: the Quesada 2010.15595 / Bulmer 2108.01622 chain rule, PNR, pure and lossy,
reimplemented (our policy), with the collision-aware batched loop hafnian. Displacement comes for
free. Threshold via fan-out comes next (M5), because every Jiuzhang is threshold or pseudo-PNR.
Borealis, the only PNR device, can be placed without M5.

**Validate against**: (a) our truncated `ProductSuperposition` via `ProductSLOSKernel` (independent,
ours); (b) The Walrus probabilities; (c) Piquasso probabilities (independent code). Sampler
histograms against (a) within shot noise. Our sampler is **not** validated against The Walrus's
sampler, which is the same algorithm by the same author. A Torontonian oracle (The Walrus `tor`,
Piquasso) for M5.

Effort, in agent-days of focused work (uncertainty in brackets), excluding the user's GPU time:

| milestone | content | estimate | risk |
|---|---|---|---|
| M1 | `GaussianState` (covariance, means, squeezers per mode, r), SMSV/TMSS constructors, loss as a covariance map in `imperfections.py`, η = 1 exactness test, truncated-ProductSuperposition oracle, Walrus/Piquasso oracle tests | 3 (2-5) | phase conventions; the renormalisation check of §3 |
| M2 | CPU chain rule: pure-state decomposition, conditional states, collision-aware batched loop hafnian, PNR sampler; validation ≤ 1e-13 and shot-noise histograms | 5 (4-8) | the batched loop hafnian is the subtle part; numerical precision above N ≈ 40 |
| M3 | GPU loop-hafnian kernel in the chain-rule shape (lane planner and multiplicity tables reused, new warp/CTA-per-term inner loop, complex128), CPU-reachable test via the Triton interpreter or the `gpu` tier | 12 (8-20) | the new kernel of §2.2; register/shared-memory tuning is where time goes |
| M4 | loss (trivial after M1), Schmidt-mode distinguishability as a product of instances, the §6.1 law check | 3 (2-4) | the high-gain Schmidt mapping |
| M5 | threshold by fan-out; pseudo-PNR bins; Torontonian oracle | 4 (3-6) | fan-out size vs exactness (limit of many sub-detectors) — to derive |
| M6 | `find_threshold --engine gbs` (`--squeezing --ratio --detector --loss --indist`), GPU launcher, memory-frontier count, CSVs/figures | 3 (2-4) + GPU runs | tail-dominated cost makes brackets noisy (SBS η = 0.5 precedent) |
| M7 | `docs/GAUSSIAN.md`, ARCHITECTURE row, TUTORIAL subsection, doc map, measurement log, paper section | 2 (2-3) | — |
| total | | ≈ 32 (23-50) agent-days | |

A lighter path, if the go/no-go is "not the full engine": **M1 + M2 + a calibrated cost model** (≈ 9
agent-days) gives an exact CPU sampler, the oracle stack, the measured per-term cost and a *projected*
H100 frontier from the §6.1 law — labelled as a projection, like the SBS "projected" line. §4 and
§5.2 (the cited points and the Borealis fix) need no code and are deliverable today whatever the
decision.

## 9. Open questions

1. **Average-case term count under collisions** (a GBS ρ(c)): not in the literature; it decides the
   frontier above r_crit and is the hafnian counterpart of C&C 2020's ρ(c).
2. **Is there a register-march analogue for Gaussian inputs in loop devices?** The bandwidth-limited
   loop hafnian (2^w) suggests the exact cost of Borealis-type ladders is set by the ladder, not by
   N — the GBS counterpart of our LBS result. Deshpande et al. argue the ladder defeats it; check.
3. **High-gain Schmidt mapping** r → {r_k} and whether "leading-instance frontier" is the right law
   (§6.2).
4. **Fan-out exactness for threshold**: Bulmer used 12 sub-detectors; the exact threshold limit needs
   the number of sub-detectors → ∞ or a proof that a finite fan-out is exact for click patterns —
   to be read in their supplement before M5.
5. **SBS link**: the brief's trap (the truncated-SMSV route through the SBS sampler) inverts into a
   question for docs/SUPERPOSITION.md §7.1 — does the loop-hafnian pairing structure that gives GBS
   2^{N/2} carry over to the SBS two-replica sum (4^j → 2^j)? A GBS engine would put both chain rules
   in one code base.
6. **Numbers not verified and not to be quoted before re-checking**: Deshpande et al.'s tensor-network
   time; Oh et al.'s sampling time per Borealis instance; the TFLN chip's source count, squeezing,
   detector type and sample definition; the Jiuzhang 1.0/2.0 squeezing values; 2311.05355 and
   2305.00853 authors; the Cygan–Pilipczuk attribution in The Walrus docs.

## References (checked 2026-09-23)

Exact: arXiv 1807.01639, 1810.00900, 1908.08068, 1908.10070, 2010.15595, 2108.01622, 2109.04528,
2202.04600, 2305.00853, 2309.15422, 2311.05355, 2009.01177, 2102.12474, 1805.12498, 1107.4466,
1302.3763, 2005.04214; Kan, J. Multivar. Anal. 99, 542 (2008). Approximate: 2306.03709, 2511.14923,
2109.11525, 2207.10058, 2403.15339, 2604.12330, 2510.24137, 1905.12075. Devices: 2012.01625,
2106.15534, 2304.12240, Nature 606, 75 (2022), 2508.09092, 2609.11922. Distinguishability: Shi &
Byrnes, npj QI 8, 54 (2022); Renema, PRA 101, 063840 (2020). Code: github.com/XanaduAI/thewalrus,
/XanaduAI/strawberryfields, /jakeffbulmer/gbs, /Budapest-Quantum-Computing-Group/piquasso,
/XanaduAI/MrMustard, /TuringQ/deepquantum, /sss441803/BosonSupremacy, /Quandela/Perceval,
/chaofanzhougit/GBSKernels.
