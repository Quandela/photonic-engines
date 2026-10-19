# The algorithm set: exact Fock samplers compared on one instance family

Design note on algorithm comparison and frontier measurements. v1, 2026-09-30, CPU stage:
implementations, oracles and addend counts done; H100 timings pending.

The paper's claim is *algorithm vs algorithm*, not library vs library: every
published exact Fock sampler is reimplemented in this engine's vocabulary, run on the same Haar
instances and the same GPU, and the frontier is the lower envelope. This note fixes what each
algorithm *is* (read off the papers and, where the paper is silent, the public source), gives the
pseudo-code of each step as implemented here, records the addend counts measured along real chains,
and says how the H100 curves are taken. Numbers measured on the GPU go in §5, with their run stamps.

Code: `photonic_engines/clifford/variants.py` (plain-torch reference of every step, the oracle),
`photonic_engines/clifford/cofactors_input.py` (the Triton kernel of the input-side collapse),
`fock_sample(algo=...)` in `clifford/sampler.py` (the engine), `tests/clifford/test_variants.py` and
`tests/clifford/check_algos_gpu.py` (gates), `bench/algorithm_terms.py` (counts),
`bench/algorithm_envelope.py` + `bench/plot_algorithm_envelope.py` (the envelope figure), `bench/find_threshold.py --algo`
(bracketed n\*), `bench/cofactor_precision.py` (precision).

## 1. One chain rule, five ways to price a step

Every exact sampler below except the last one is Clifford & Clifford's algorithm B
(arXiv:1706.01260): draw a uniform order σ of the n input columns of A = U[:, in_modes] ([m, n]),
grow the ordered output rows r_1..r_n one photon at a time; at step k the candidate ρ has weight
|per(A[(r_1..r_{k−1}, ρ), σ(1..k)])|². Laplace along the last row,

    per([B; a_ρ]) = Σ_c a_ρ[c] · per(B \ c),     B = A[r_{<k}, σ(1..k)]   ([k−1, k]),

so a step is: (i) the k **cofactors** P_c = per(B \ c) (or an equivalent candidate-independent
vector), (ii) m inner products of length k, (iii) one draw. The algorithms differ only in (i), and
(i) is always a Glynn / BB-FG sum walked in Gray order, one addend = one rank-1 update of k sums.
That unit is machine- and language-independent and it is what the exponent claims are about;
`bench/algorithm_terms.py` counts it, `bench/theboss_workcount.py` counted it for theboss.

| algo | who | the sum | addends per step | on a collision-free input, Haar U |
|---|---|---|---|---|
| `2018` | C&C 2018, algorithm B; Perceval/exqalibur, CRAN, BosonSampling.jl | row-signed Glynn, δ ∈ {±1}^{k−1}, δ_1 fixed, all k cofactors from one sweep | 2^{k−2} | 2ⁿ |
| `2020` | C&C 2020 (arXiv:2005.04214), **this engine** | the same sweep collapsed on the repeated OUTPUT rows: Π_ν (s_ν+1), halved | Π_ν (s_ν+1)/2 over r_{<k} | ρ(c)ⁿ, ρ(1) = 1.69, ρ(2) = 1.81 |
| `input` | Brod & Oszmaniec generalised alg. B (arXiv:1906.06696); **theboss** (Rybotycki) | the sweep signs the INPUT columns and collapses on their multiplicities; one sweep gives Q_μ = per([B; e_μ]) per distinct input mode μ | Π_μ (s_μ+1)/2 over σ(1..k) | 2^{k−1} per step = **2 × `2018`**, whatever the output collides |
| `percof` | **PiquassoBoost** (Rakyta, Morse et al., arXiv:2309.07027 and the DFE) | every cofactor by its OWN output-collapsed BB/FG sweep (n-ary Gray code), k sweeps per step | k × Π_ν (s_ν+1)/2 (PiquassoBoost's halving: t_min Π_{ν≠min}(t_ν+1), same to ±1 %) | ρ(c)ⁿ × n: the 2020 exponent at k× the work per step |
| `strong` | SLOS then inverse-CDF; **DeepQuantum**'s Fock path (Ryser per output state) | no chain: the whole distribution once, then lookups | C(m+n−1, n) amplitudes once per U | memory-bound: 13 sources at m = 2n, 18 at m = n on 80 GiB (docs/MEMORY_FRONTIER.md) |

Two facts about the public codes, read in their source (piquasso 8.0.1 and piquassoboost 0.3.5
wheels from PyPI, 2026-09-29, `piquassoboost/sampling/simulation_strategies/source/
CGeneralizedCliffordsBSimulationStrategy.cpp`, `samplingHelperFunctions.cpp::compute_pmf`,
`sampling/source/BBFGPermanentCalculatorRepeated.hpp`; theboss from its public repository):

1. **PiquassoBoost collapses on the output side and therefore DOES reach the ρ(c)ⁿ exponent.**
   `compute_pmf` loops over the input columns that still carry a photon (a `tbb::parallel_for`),
   removes one photon from that column and calls `BBFGPermanentCalculatorRepeated::calculate(
   matrix, input_state, output_state)` — one call per column, one full permanent each. Inside,
   the n-ary Gray code runs over `row_mult` = the *output* sample (the row of minimal non-zero
   multiplicity is fixed instead of the complement-symmetry halving; the column multiplicities
   only set the powers in Π_j colsum_j^{N_j}). So the per-step count is Σ_{c} t_min Π_{ν≠min}(t_ν+1):
   the collapse this engine uses, without the cofactor sharing — k sweeps where one suffices. The
   statement "no public implementation reaches this exponent" holds for theboss only. Corrected
   reading: theboss runs at the 2018 exponent on unbinned input (measured); PiquassoBoost reaches
   the 2020 exponent with an extra factor k per step (counted in source and along chains, §4) — no
   public implementation has both the collapse and the shared sweep. Their DFE paper offloads
   exactly these per-cofactor permanents.
2. **The Piquasso Python sampler (module `piquasso._simulators.passive.sampling`) is algorithm B with shared
   cofactors** (`calculate_permanent_laplace` once per step, `partial_permanents[j]` weighted by
   `input_state[j]` and `interferometer[idx, j]` — that is the Q_μ vector of the `input` row, so
   Piquasso proper is the Brod–Oszmaniec form with whatever permanent its connector supplies);
   PiquassoBoost replaces that step by the per-column C++ path above. The two are not the same
   algorithm, and a comparison table should list them as separate rows.

## 2. Pseudo-code (v1)

Common frame (all four chain-rule variants; `fock_sample`, S streams in lockstep, one launch pair
per step; the CPU reference `fock_sample_reference` runs it one stream at a time):

```
sample(A[m, n]):
    σ ← uniform permutation of the n input columns;  rows ← []
    for k = 1..n:
        B ← A[rows, σ(1..k)]                                  # [k−1, k], rows may repeat
        (P or Q, cols) ← STEP_algo(B, rows, σ(1..k))          # candidate-independent vector
        amp[ρ] ← Σ_j A[ρ, cols_j] · P_j        for ρ = 1..m   # kernel 2, [m, k] tile of A
        draw ρ ~ |amp[ρ]|²;  rows.append(ρ)
    return occupation(rows)
```

`STEP_2018(B)` — one row-signed Gray sweep (kernel `_cofactors_kernel`):
```
s ← Σ_i B[i, :]                       # column sums at δ = +1        [k]
acc ← 0                                                              [k]
for g = 0 .. 2^{k−2} − 1 in Gray order:                              # δ_1 = +1 fixed
    if g > 0: i ← 1 + ctz(g);  s ← s − 2 δ_i,old B[i, :];  δ_i ← −δ_i   # one rank-1 update
    acc[c] += sign(δ) · Π_{j≠c} s_j   for all c                     # all-but-one products, P / s_c with zero handling
P ← acc / 2^{k−2}
```

`STEP_2020(B, rows)` — the same sweep on the distinct rows b_ν with multiplicities s_ν
(kernel `_cofactors_multi_kernel`, CUDA twin `cof_multi`):
```
(b, s) ← distinct rows of B and their multiplicities;  an odd-s group last, radix (s+1)/2 (halving)
v ← 0 (number of −1 signs per group);  w ← Π_ν C(s_ν, v_ν) (−1)^{v_ν} = 1;  s_col ← Σ_ν s_ν b_ν
for each term of the reflected mixed-radix Gray code over v (Π_ν radix_ν terms):
    step: the lowest digit that can still move changes v_ν by ±1
          s_col ← s_col ∓ 2 b_ν;   w ← w · (binomial ratio, with the sign)
    acc[c] += w · Π_{j≠c} s_col_j   for all c
P ← acc · 2 (if halved) / 2^{k−1}
```

`STEP_input(B, σ(1..k))` — column-signed sweep (kernel `_qvec_input_kernel`, new):
```
(d, s) ← distinct INPUT columns among σ(1..k) and their multiplicities; odd group last (halving)
v ← 0;  w ← 1;  coef_μ ← s_μ − 2 v_μ;   r_i ← Σ_μ coef_μ A[rows_i, d_μ]   for the k−1 rows
for each term of the same mixed-radix Gray code over v (Π_μ (s_μ+1) terms, halved):
    step: v_μ ± 1;  r ← r ∓ 2 A[rows, d_μ];  coef_μ ∓= 2;  w ← w · (binomial ratio)
    P ← Π_i r_i                                                     # FULL product over the rows
    acc[μ] += w · coef_μ · P   for all μ
Q ← acc · 2 (if halved) / 2^k ;   amp[ρ] = Σ_μ A[ρ, d_μ] Q_μ      (kernel 2 with d in place of σ)
```
Identity: Q_μ = per([B; e_μ]) = Σ_{c: σ_c = μ} per(B \ c), so amp is the same number as in the
Laplace form; the halving is legitimate because the summand is invariant under v → s − v
((−1)^k from w, (−1)^{k−1} from the k−1 row sums, −1 from coef).

`STEP_percof(B, rows)` — PiquassoBoost's step:
```
for c = 1..k:                                                        # one permanent per input column
    P_c ← per(B \ c) by STEP_2020's collapsed sweep on the [k−1, k−1] block B \ c (full product)
```
Implemented in `fock_sample` as k launches of the 2020 kernel per step, launch c keeping column c
of its output — bit-identical samples to `2020`, k× its kernel work, as in their sampler (theirs
fixes the minimal-multiplicity row instead of halving and has k−1 columns per sweep instead of
k: the count agrees to ±1 %, §4; the wall-clock of the reimplementation is what F2a shows).

`strong_then_sample(U, in_modes, S)`: `FockSLOSKernel(n, m, occ).probs(U)` (complex64, the
memory-frontier convention), then S draws by inverse CDF; the cost per sample of a run of S is
(build + draws)/S — the run-of-S statistic, S = 10⁴ headline.

## 3. Validation

- Every step function against Ryser cofactors on blocks with repeated rows AND columns, 1e−13
  relative; the addend count returned by each sum against the closed form `addends()` the planner
  uses (tests/clifford/test_variants.py, 45 tests, 12 s on CPU).
- The Triton `input` kernel against the reference sum under the CPU interpreter, including a
  forced multi-lane segmentation (4 lanes per stream), and the GPU gate at k ≤ 14
  (`check_algos_gpu.py`): 1e−15 relative.
- The four algorithms in `fock_sample` return the **same samples from the same seed** (same σ,
  same Philox draws, weights equal to rounding): checked at n = 3, 4 with a bunched input on
  the interpreter and, in the gate, at n = 12, m = 24, 512 shots, Triton and CUDA. This is the
  strongest cheap check there is — the chains are identical by construction, so any wrong term
  would move a draw.
- The CPU chain of each algorithm against the permanent distribution at n = 3 (bunched input),
  3000 shots, TVD below the 5σ shot-noise floor; strong-then-sample the same at n = 2, m = 5.

## 4. Addends per sample along real chains (CPU, 2026-09-29)

`results/keep/algorithm_terms_cpu_20260929.csv`: 40 chains per point drawn by the reference
sampler (2020 step), Haar U, seed 0, the counts of every algorithm taken along the SAME chains.

| c | n | 2018 | 2020 (ours) | input | percof (ours) | PiquassoBoost literal | theboss |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 128 | 68 | 255 | 452 | 449 | 509 |
| 1 | 12 | 2 048 | 597 | 4 095 | 6 246 | 6 265 | 8 189 |
| 1 | 16 | 32 768 | 4 538 | 65 535 | 64 804 | 64 802 | 131 069 |
| 2 | 8 | 128 | 79 | 255 | 534 | 530 | 509 |
| 2 | 12 | 2 048 | 987 | 4 095 | 10 564 | 10 550 | 8 189 |
| 2 | 16 | 32 768 | 10 823 | 65 535 | 158 993 | 158 993 | 131 069 |

Fitted base over n = 6..16: 2018 2.000, 2020 1.694 (c = 1) / 1.833 (c = 2) [ρ(1) = 1.6875,
ρ(2) = 1.808], input 2.003, theboss 2.004 (= 2^{n+1} − 4 exactly), percof / PiquassoBoost 1.89 /
2.05 — the ρ(c)ⁿ exponent under a polynomial factor: PiquassoBoost / ours = 13.5 at n = 16, c = 1
and 14.7 at c = 2, growing like n − 1.5. Read: on this family the input-side collapse buys nothing
(2× the 2018 count, all its gains are for binned inputs), the output-side collapse is the whole
gap between 2ⁿ and ρ(c)ⁿ, and sharing the sweep across the k cofactors is worth a further factor
≈ k. GPU chains at n = 16..40 extend the table to frontier sizes.

## 5. The H100 measurement

Same instances (Haar U, seed = run seed + n, n photons in modes 0..n−1, m = c n, c = 2 and 1),
same card, same statistic as `find_threshold` (shot ramps, sustained rate over runs of ≥ 8 shots,
120 s per point), same substrate for the four chain-rule sums (Triton), the production CUDA path
(`auto`) as the reference curve, `strong` as the run-of-10⁴ cost up to the memory limit.
Deliverables: `results/keep/algorithm_envelope_*.csv` (envelope data), `img/algorithm_envelope.png`
(envelope figure), one row per algorithm with its bracketed n\* (`threshold_<algo>_c<c>.csv`).

### 5.1 m = 2n, one H100 PCIe (run 20260930_071754, gate 20260930_082410 GATE OK)

`results/keep/algorithm_envelope_c2_20260930_h100.csv`, n = 20..40 step 2, seconds per exact
sample (sustained; a curve stops at the first point whose single shot exceeds 120 s):

| n | 2018 | 2020 (Triton) | input | percof | production (auto, CUDA) |
|---|---|---|---|---|---|
| 24 | 2.6 ms | 0.84 ms | 7.3 ms | 15.6 ms | 0.35 ms |
| 28 | 41 ms | 6.0 ms | 97 ms | 154 ms | 1.9 ms |
| 32 | 0.50 s | 49 ms | 1.20 s | 1.31 s | 22 ms |
| 34 | 5.2 s | 0.30 s | 10.5 s | 7.2 s | 69 ms |
| 36 | 24 s | 1.31 s | 56 s | 107 s | 0.23 s |
| 38 | 97 s | 4.1 s | — | — | 0.84 s |
| 40 | — | 24 s | — | — | 2.6 s |

n\* at 1 sample/s, bracketed between the two measured n around it (log-linear fit over the points
≥ 10 ms in brackets): **2018 32.6** (32.2) · **2020/Triton 35.6** (35.8) · **input 31.7** (31.2) ·
**percof 31.4** (30.6) · **production 38.3** (38.4; the frontier's 38.5). Ordering and gaps as the
counts predicted (§4): the output collapse is worth +3.0 sources on the same kernel substrate
(2018 → 2020, Triton to Triton); the input-side collapse costs −0.9 against 2018 (its 2× count);
sharing the sweep is worth +4.2 (percof → 2020); the CUDA thread-per-lane kernel adds +2.7 on top
of the Triton 2020 (constant factor 3.5–5× at n ≥ 34). The lower envelope is the production curve
everywhere in this range. Slopes: production ρ_eff 1.83 (ρ(2) = 1.81); the Triton curves fit
steeper (2.05–2.19) because the K_PAD 32 → 64 tier step falls at n = 33 and the last points hold
1–2 shots inside the budget — read the bracket, not the fit, for those rows. The bracketed job-5
numbers replace these when they land.

### 5.2 The full run (20260930_082501): m = n, bracketed n\*, strong, GPU counts, canonical reruns

**n\* at 1 sample/s, bracketed by `find_threshold --algo` (budget 120 s, backend
triton; production = `auto`/CUDA, budget 180 s, the canonical reruns):**

| algorithm | m = 2n | m = n | ρ_eff (fit) m = 2n / m = n |
|---|---|---|---|
| C&C 2018 (`2018`) | **32.5** (32 at 2.00/s, 33 at 0.46/s) | **32.5** (same instances: the cost is m-independent) | 2.06 / 2.06 |
| input-side collapse (`input`) | **31.7** (31 / 32) | **31.8** (31 / 32) | 1.96 / 1.95 |
| PiquassoBoost form (`percof`) | **30.9** (30 / 31) | **33.7** (33 / 34) | 1.86 / 1.74 |
| C&C 2020, Triton (`2020`) | **35.7** (35 / 36) | **40.0** (40 / 41) | 1.89 / 1.78 |
| **production (2020, CUDA)** | **38.5** (38 at 1.26/s, 39 at 0.79/s) | **42.5** (42 at 1.16/s, 43 at 0.84/s) | 1.82 / 1.78 |

The canonical production points are `results/keep/frontier_fock_canonical_c{2,1}_20260930_h100.csv`
(run 20260930_082501): **38.5 at m = 2n** (confirms frontier v2's 38.5, run 20260915) and **42.5 at
m = n** (between the 42.4 / 42.7 quoted so far; ρ_eff 1.78 vs ρ(1) = 1.69 — the K_PAD 48/64 tiers at
n > 40). The per-algorithm CSVs are `results/keep/threshold_<algo>_c<c>_20260930_h100.csv`. The
2018 and input rows do not move with m (their counts do not depend on the output), percof and
2020 gain 2.8 and 4.3 sources from m = 2n to m = n (the collapse pays where photons bunch), and
the 2018 → 2020 gap on the same substrate is **+3.2 sources at m = 2n and +7.5 at m = n**, the
paper's "algorithms tilt the curve" statement measured on one card.

**Envelope at m = n** (`algorithm_envelope_c1_20260930_h100.csv`, n = 20..44): seconds per sample
at n = 32 / 36 / 40 / 44: 2018 0.50 / 24 / — ; 2020 (Triton) 0.0076 / 0.147 / 1.54 / 13.1; input
1.11 / 52 / —; percof 0.22 / 6.5 / 57 / —; production 0.0035 / 0.027 / 0.237 / 2.59. Figure
`img/algorithm_envelope.png` (both ratios; the strong-then-sample branch on the left of each panel).

**Strong-then-sample** (`algorithm_envelope_strong_20260930_h100.csv`, run 20260930_141657, S = 10⁴):
the table fits to n = 11 at m = 2n (1.29·10⁸ states, build 2.6 s; n = 12 out of memory on the 80 GB
card — the current SLOS kernel keeps its edges, docs/MEMORY_FRONTIER.md "current (edges)" column:
11 / 13 for the table alone) and to n = 15 at m = n (n = 16 hits the 63-bit packed-key limit of the
torch basis builder — a code limit, not a memory one; the table alone would allow 18). Per-sample
cost of the run = the build alone (draws are a lookup, < 1 ms for 10⁴): **0.26 ms at n = 11, m = 2n**
and **0.24 ms at n = 15, m = n**, i.e. 10⁴ samples/s where the chain rule does 10²–10³ — the strong
branch is the envelope up to its memory limit, and the crossover is set by memory, not rate. (Two
earlier runs of this job reported 20–25 ms per sample: bench artefacts — a CPU multinomial, then a
CPU rebuild of the basis — both removed.)

**Addends along GPU-drawn chains** (`algorithm_terms_gpu_20260930_h100.csv`, 200 chains per point,
n = 16..40): fitted bases 2020 **1.680** (m = n; ρ(1) = 1.6875) and **1.814** (m = 2n; ρ(2) =
1.808) — the theorem's exponent reproduced to 0.5 % on real chains at frontier sizes; 2018,
input and theboss exactly 2.000; percof / PiquassoBoost 1.748 / 1.887 (n·ρⁿ over this range).
At n = 40: 2018 5.5·10¹¹, 2020 1.2·10⁹ (m = n) / 1.7·10¹⁰ (m = 2n), PiquassoBoost 4.4·10¹⁰ /
6.6·10¹¹ (38 × ours), theboss 2.2·10¹².

**Precision (`cof_precision_20260930_h100.csv`, `_collide`)**: relative disagreement of the
cofactors between the CUDA kernel and the Triton kernel 3·10⁻¹⁴ (k = 30), 7·10⁻¹⁴ (34),
1·10⁻¹³ (38) — the two kernels sum in the same order per lane, so this is not the rounding error;
between two lane segmentations of the Triton kernel (the informative pair: different partial-sum
order) **1.4–2.1·10⁻¹¹ (k = 30), 5–6·10⁻¹⁰ (34), 7–9·10⁻⁹ (38)**, the same on the m candidate
amplitudes, ×~40 per 4 photons (×~2.5 per photon), one repeated row unchanged. Lower bound of
the fp64 error at the frontier: ~10⁻⁸ at n = 38, extrapolating to ~10⁻⁶ at n ≈ 43 and 10⁻⁴ at
n ≈ 48 — consistent with the Clifford review's "10⁻⁶ at n ≈ 38" only as an upper reading; the
number to state in Supplement C is this measured lower bound with its growth rate, and the
crossing of 10⁻³ (the precision criterion for switching to double-double) lands near n ≈ 50, where the multi-GPU curves sit (§3.1 7b).

### 5.3 Is the comparison fair to the reimplemented algorithms? (the referee's question)

A referee will ask whether `input` and `percof` lose because they are algorithms or because we
tuned our own kernel harder. Three answers, all measured.

1. **The count is implementation-free and settles the exponent.** Along the same GPU-drawn chains
   (§5.2) the input-side collapse costs exactly 2^(k−1) addends per step on a collision-free input
   — twice C&C 2018, whatever the output does — and PiquassoBoost's per-cofactor sweeps cost k times
   the shared sweep; both counts are read off their source and reproduce their own literal
   formulas to 0.1 %. No implementation can go below its own addend count, so the ordering
   percof ≈ input < 2018 < 2020 on this family is algorithmic. Brod & Oszmaniec never claim a
   gain on unbinned inputs (theboss's own docstring says the transposed radix is "not desired for
   the standard BS experiment"); the row records that the form pays only on binned inputs.
2. **Per-addend throughput of the four Triton kernels is the same to a factor 1.6, and the two
   reimplemented forms run their addends *faster* than our own Triton 2020 kernel.** G addends
   per second = (addends per sample from `algorithm_terms_gpu_20260930_h100.csv`) × (samples/s
   from the envelope CSVs), at n = 24 / 28 / 32, m = 2n: 2018 3.3 / 3.3 / 4.3; input 2.3 / 2.8 /
   3.6; percof 1.8 / 2.1 / 3.1; 2020 (Triton) 1.5 / 2.0 / 2.7; the production CUDA kernel 3.5 /
   6.6 / 6.1 (m = n: the same picture, 2018 3.3 / 3.2 / 4.3, input 2.5 / 3.0 / 3.9, percof 0.9 /
   1.8 / 2.7, 2020 0.6 / 1.6 / 2.6, CUDA 1.2 / 4.1 / 5.8). The input kernel does less work per
   addend (a full product instead of k all-but-one products) and shows it; percof is the 2020
   kernel itself. Beyond n = 32 every Triton kernel drops by ×2–3 together (the untuned 64-column
   tier), which is why the paper's T1 uses the bracketed n\* and not the fitted slopes. The only
   implementation gap in the figure is between our own two kernels (Triton vs CUDA, ×2.3–3), and it
   is drawn as two separate curves for exactly that reason.
3. **`percof` is PiquassoBoost's algorithm given our best sweep**, k launches of the 2020 kernel
   per step: a lower bound on what their structure costs on this card, not an upper bound. Their
   sweep has k−1 columns instead of k (≈ 10 % less work per addend, ≈ 0.15 sources) and their
   halving fixes the minimal-multiplicity row instead of using the complement symmetry (counted:
   ±1 %); the TBB / DFE parallelism across columns changes where the k permanents run, not how
   many addends they cost. If anything the row is generous to PiquassoBoost.

What the paper should therefore say: the T1 numbers of the reimplemented forms are what those
algorithms cost with a kernel at least as good per addend as the one we give our own algorithm on
the same substrate; the exponents are counts, not timings. What it should not say: anything
about the wall-clock of the third-party binaries (CPU, other machines).

### 5.4 Consistency check: the SBS chain at k = n is this table's 2018 row

**Superseded (2026-10-05) by checks (b) per-step cost and (c) exactness of the SBS validation gate (`bench/sbs_rso_validate.py`, §9.2).** Kept as the record of the torch measurement of run 20261003_223312; quoted in no paper.

`clifford/sbs_rso.py` runs the random-source-order chain for superposed sources; at
α = π/2 every source is a Fock photon, its start set T₀ is empty and the chain is C&C 2018 verbatim
(docstring of that file). Checked on the CPU (2026-10-03): the per-step sign count is 2^{j−2} at
step j, identical to `addends("2018")` for every step (sums 32 / 32 768 / 5.4e8 at n = 6 / 16 / 30);
samples at α = π/2 follow the exact Fock law — TVD 0.019 at n = 4, m = 8 (330 patterns, 10⁵
shots; shot-noise floor 0.023) and 0.033 at n = 5, m = 10 (2002 patterns, 2·10⁵ shots; floor 0.040).
**H100 bracket (run 20261003_223312; `results/keep/threshold_rso_pi2_c2_20261003_h100.csv`,
`threshold_2018_c2_ref_20261003_h100.csv`):** the rso chain at α = π/2 (torch Glynn sweep of the
rectangular cofactors, no kernel) brackets **n* = 25.3** at m = 2n (n = 25 at 1.22/s, n = 26 at
0.57/s; ρ_eff 2.16, the 2018 count's 2.0); the `2018` Triton kernel re-measured in the same job
brackets **32.5** (n = 32 at 2.00/s, n = 33 at 0.46/s) — the T1 row to the digit. Same count, same
law, **7.2 sources of substrate** between a torch sweep and the Triton kernel (190× per sample at
n = 28: 8.05 s vs 42 ms); against the production CUDA row (38.5) the gap is 13.2. This is the
substrate gap the parity rule (§9) addresses: before any collapse, porting the SBS chain onto the kernels is worth ≈ 7 sources
(Triton) to ≈ 10 (CUDA) at the 2018 count, and the output collapse adds what T1 measured for it
(+3.2 at m = 2n on Triton). The per-source price of the torch sweep is larger at α = π/4, where the
rectangular cofactors are (j−1) × (n−k+j−1) rather than square, so the SBS row's parity gain is
not a simple addition; it is what the kernel-ported build of §9.2 measures.

## 6. Precision at the frontier

`bench/cofactor_precision.py`: no exact reference exists at k ≈ 38 in fp64, so the rounding error
of the cofactors is bounded below by the disagreement between evaluations that round differently
(CUDA vs Triton, two lane segmentations, the 2018 sweep vs the 2020 one at k ≤ 30), on the
cofactors and on the m candidate amplitudes the draw sees, at k = 30, 34, 38, distinct rows and
one repeated row. The Clifford review's 1e−6 at n ≈ 38 (×4 per two photons) is the number this
job re-measures with a stated method; Neumaier / double-double only if it exceeds 1e−4.

## 7. Versions

- v1.8 (2026-10-05): §9.2 limit-test table (one row per engine with a Fock limit; SBS validation gate state); §9.1 SBS rows → pending the gate, no number; §5.4 superseded.
- v1.7 (2026-10-04): §5.4 H100 bracket of the k = n check — 25.3 (rso, torch) vs 32.5 (2018, Triton): 7.2 sources of substrate; §9 SBS row annotated.
- v1.6 (2026-10-03): §9 implementation-parity audit of every quoted row.
- v1.5 (2026-10-03): §5.4 the k = n consistency check of the SBS rso chain (CPU part; H100 bracket queued).
- v1.4 (2026-10-02): §8 feed-forward cost table on the emitter train (F7), from the kept E3 CSVs.
- v1.3 (2026-09-30): §5.3 fairness of the comparison — per-addend throughput per kernel (referee's question).
- v1.2 (2026-09-30): the full run — m = n envelope, bracketed n* per algorithm (T1 column), strong, GPU counts, precision, canonical production points 38.5 / 42.5 (§5.2).
- v1.1 (2026-09-30): m = 2n envelope measured on the H100 (§5.1); GPU gate passed.
- v1 (2026-09-30): the five algorithms fixed from the papers and the public source; CPU
  reference steps, the `input` Triton kernel, `percof` and `input` in `fock_sample`,
  `strong_then_sample`; tests and the GPU gate; counts to n = 16; the H100 job list. Not yet run
  on a GPU: the `input` kernel has been validated under the Triton interpreter only.

## 8. Feed-forward cost on the emitter train (F7, 2026-10-02)

Tabulated from the local-optics fusion runs (`results/keep/esbs_fusion_*.csv`: Laplace label pass run 20260918_084620,
Glynn/GEMM pass 20260919_084900; one H100, 256 streams) by `bench/f7_feedforward_cost.py` →
`results/keep/f7_feedforward_cost_20261002.csv`, `img/f7_feedforward_cost.png`. The quantity is the
median over ticks ≥ 2 of the ms per tick-shot of `SequenceKernel.sample_diag` (tick 1 carries the
compile), for the same xfusion optics and random spin gates under three programs: `brick` (static
brickwork, no feed-forward), `rus` (the optics of bin b+1 chosen per stream from the fusion
outcomes — routing feed-forward), `circuit` (parity-CNOT random circuit: gates *and* routing fed
forward per stream; odd k, 120 ticks). The policy's price is the ratio to `brick` at the same k
(odd k: geometric mean of the k ∓ 1 brick points).

| k | brick, no FF | rus, routing FF | ratio | circuit, gates + routing FF | ratio | layers / 120 ticks | ms per layer | Glynn brick |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.0108 | 0.0126 | 1.17 | | | | | |
| 5 | | | | 0.0230 | 1.72 | 11.6 | 0.24 | |
| 6 | 0.0165 | 0.0188 | 1.14 | | | | | |
| 7 | | | | 0.0355 | 1.70 | 10.5 | 0.41 | |
| 8 | 0.0263 | 0.0292 | 1.11 | | | | | |
| 9 | | | | 0.0509 | 1.19 | 9.7 | 0.63 | |
| 10 | 0.0692 | 0.0819 | 1.18 | | | | | |
| 11 | | | | 0.167 | 1.00 | 9.1 | 2.2 | |
| 12 | 0.406 | 0.410 | 1.01 | | | | | 0.167 |
| 13 | | | | 1.10 | 0.97 | 8.6 | 15.3 | |
| 14 | 3.18 | 3.18 | 1.00 | | | | | 0.707 |
| 15 | | | | 9.38 | 0.98 | 8.4 | 134 | |
| 16 | 29.0 | 29.0 | 1.00 | | | | | 3.64 |
| 18 | | | | | | | | 28.2 |

ms per tick-shot; ms per sample of a whole train = T × this (T = 40 for brick / rus: 0.43 ms at
k = 4 → 1.16 s at k = 16 on the Laplace pass, 0.146 s with the Glynn pass; T = 120 for circuit).
Type-II and Haar-4 brickworks (same files) cost the same as xfusion brick to within 3 % at every k;
`h` gates the same as random gates.

Reading. The feed-forward policy costs nothing per tick once a tick is compute-bound: ratio 1.00
± 0.01 for k ≥ 11 under both policies. Below k ≈ 10 a tick is launch-bound (10-70 µs for 256
streams) and the per-stream gather of the policy — selecting each stream's next optics / gates
from its own outcomes — shows as +11-18 % (routing) and +70 % (gates + routing, two gathers and
the layer bookkeeping), i.e. a few microseconds per tick, not a factor. The exponent is the
register's (the 3^k trajectory sweep, 2^k × 2^k GEMM for the labels: ×8.9 per two emitters from
k = 12 to 16 on the Laplace pass, ×4.2-7.8 on the Glynn pass), not the policy's — as SEQUENCE.md
§10 argues for the trajectory picture (the classical outcome → gate loop is free) and §11 for the
routing. What feed-forward changes is the *number of ticks a sample needs*: `rus` keeps attempting
a sublattice until every pair has fused (≈ 2 ticks per pair at p = 1/2), `circuit` spends ≈ 12
ticks per layer (8.4-11.6 layers in 120 ticks) — so the per-sample cost of a programmed train is
set by its tick count, 15 ms per layer at k = 13 and 134 ms at k = 15, against 0.23 / 0.72 s per
40-tick brick sample at the same k. Not measured: the strong side under feed-forward (conditional
marginals, SEQUENCE.md §10 — not built; the paper states it as a limitation).

## 9. Implementation-parity audit of the quoted rows (2026-10-03; limit tests §9.2, 2026-10-05)

The parity rule: every row quoted in the frontier paper sits at the same optimisation level —
the best known algorithm for its input (the C&C 2020 collapse wherever the chain admits it), the production
CUDA cofactor kernel as substrate wherever the row is a chain rule, the same bracketing protocol and
statistic (bracketed n\* at 1 sample/s, or t_S where stated). A gap between two rows that is "how the sampler is built" is not a result. Verdicts:
● at parity; ◐ at parity by construction of its own engine (no common substrate exists, stated as such);
○ not at parity — drawn hollow in the frontier table and figure and excluded from every cross-engine sentence until it is.
Trigger of the audit: the SBS chain at η = 0.5 against the Fock engine on the same state (run
20261003_170411) — a gap of substrate, not of physics. The criterion is the limit test: every engine with a
Fock limit reproduces the Fock frontier there, measured (§9.2).

### 9.1 Row audit (algorithm level × substrate × statistic)

| row | n* quoted | algorithm level | kernel substrate | statistic | run stamp | verdict |
|---|---|---|---|---|---|---|
| Fock, m = 2n / m = n | 38.3 / 42.6 | C&C 2020 (output collapse) | production CUDA thread-per-lane cofactor kernel | bracketed mean, 256-shot batches with ≥ 2048 counted shots; run-of-S coincides within the bracket | 20261009_141132 (38.5 / 42.5 of run 20260930_082501 superseded, 10-09) | ● reference row |
| Fock T1 set: 2018 / input / percof / 2020 | 32.5 / 31.7 / 30.9 / 35.7 | each algorithm's own count | one common Triton substrate for the four | bracket | 20260930_082501 | ● as an algorithm comparison (§5.3 per-addend fairness); not a frontier row — never compared to the CUDA row except as "+2.8 of substrate" |
| Fock, loss η = 0.9 … 0.5 | 41.2 / 45.0 / 49.2 / 64.2 | 2020 on the surviving subset (input-side partition, lockstep per k) | production CUDA | bracket; tail-dominated at η ≤ 0.7; 256-shot batches with ≥ 2048 counted shots at every η; at 0.7 residual tail uncertainty ~±0.2 (n = 50: 0.373/s first 256-shot batch vs 0.712/s over 2048); at 0.5 the lower point n = 64 is 9 % above 1/s after a 25 % tail drop — the bracket could move to 63/64 | 20261008_071215 (0.9); 20261008_085321 (0.8); 20261008_104426 (0.7); 20261008_192343 (0.5) | ● (same engine as the reference) |
| Fock, distinguishability η = 1 / 0.7 | 38.6–41.8 / 49.2–53.7 | 2020 on each interfering species | production CUDA | bracket; η = 0.7 row on 256-shot batches with ≥ 2048 counted shots | 20260921 / 20261008_104426 (V = 1) + 20261009_195643 (V < 1) | ● |
| Fock + adaptive K = 1–3 (4; 5, 6, 8) | 38.3 / 38.2 / 38.1 (37.6; 34.8 / 34.4 / 33.8) — 512 shots, §9.3 (were 38.4 / 38.2 / 38.1 (37.0; 34.0 / 34.2 / 31.6) at 128 shots, 20260922–23) | 2020 prefix chain + exact level marginals (tables) | production CUDA (`fock_sample(prefix=, avoid=)`); level tables on the host | bracket, 512 shots per point (not the 256-shot batch cap; ratios at equal batch) | 20261007_022012 (K = 1), _030408 (K = 2, 3), _061124 (K = 4, 5, 6, 8); `afbs_K{1,23,4,5,6,8,8b}_x4_20261007_h100.csv` | ● K ≤ 3; ○ K ≥ 4 (K = 4 table per branch, K ≥ 5 rounds of level-chain launches — implementation-limited; the frontier table prints "route named from K = 4", not hollow) |
| Fock + non-linear site(s) | 36.1 (L = 2/4/8: 34.3 / 30.3 / 21.4) | rejection from the branch mixture, proposals by the 2020 chain | proposals: production CUDA; branch amplitudes: batched torch Glynn (no collapse) | bracket; L = 1 on 256-shot batches with ≥ 2048 counted shots | 20261009_141132 (L = 1; 36.0 of run 20260922_211901 superseded, 10-09); 20260922_211901 (L = 2/4/8) | ◐ chain at parity; the amplitude pass is a torch sweep — a per-site cost partly of substrate, stated as such in the paper's wording, not fixed in this release |
| emitter, linear cluster (χ = 2), m/n = 2 … 12 | 31.0 (20261006_004008, the parity build) / 30.5 … 29.0 | emitter chain rule with the 2020 repeated-rows collapse on the δ enumeration (EMITTER_SAMPLER.md) | CUDA thread-per-lane emitter kernel at the frontier steps (`auto`), Triton seq elsewhere | bracket | 20261002_003041, 20261003_103957 | ● |
| emitter, loss / distinguishability | 33.6 / 36.8 / 40.9 / 53.5; 31.0 (η = 1, V = 1: the row above) / 31.6–33.4 / 40.9–44.7 (η = 0.7, run 20261009_195643) | same, label unravelling of the trace | same `auto` kernel | bracket; every η < 1 on 256-shot batches with ≥ 2048 counted shots and the parity build; η = 0.5 heavy tail (2048-shot rates 1.9–2.6× below the 256-shot ramp; bracket 53.5 vs five-point fit 53.6) | 20261008_210721 (η = 0.9, 0.8); 20261008_122031 (η = 0.7); 20261008_133143 (η = 0.5); 20261003_103957, 20261003_121759 | ● |
| emitter trains, register k | k ≈ 21–22 at 1 tick/s (22.6 with collapse) | Glynn-GEMM label amplitudes + sign sum collapsed over repeated output rows (SEQUENCE.md §12, §12b) | cuBLAS GEMM | per-tick-shot bracket | 20260919_084900 | ● |
| superposed sources, α = π/4 | headline m = 2n **50.0** (shot-floor run, ≥ 512 counted shots per point); nine ratios **53.1 / 51.1 / 50.2 / 49.5 / 49.1 / 48.6 / 48.0 / 47.5 / 47.0** at m/n = 1 / 1.5 / 2 / 2.5 / 3 / 4 / 6 / 8 / 12 (512 counted per point; the map's m = 2n cell 50.2 is footnoted, the headline is 50.0) | random-source-order chain with the 2020 collapse | the production Fock step kernels at s0 = 0 (`--rso-square kernel`), templated μ lane 1 ≤ s0 ≤ 18 at N = 29…64, λ-form above (the build of run 20261006_132352) | bracket with the shot floor | **20261007_090041** (headline, `floor_sbs_eta1_N512_r28_20261007_h100.csv`); **20261010_045711** (nine ratios, `floor_frontier_sbs_N512_20261010_045711_h100.csv`); supersede run 20261006_132352 (53.4 / 51.0 / 49.7 / 49.4 / 49.3 / 48.9 / 48.3 / 48.2 / 48.0 at 128–256 shots, `frontier_sbs_rso_final_r22_20261006_h100.csv`) and run 20261006_012054 (53.3 … 47.8, earlier build) | **●** (same build as the re-run of 10-06, run 20261006_132352; +0.1…+0.4 vs the build of run 20261006_012054 — a prediction of +1.2–1.6 had used ρ = 1.45, the measured local slope gives +0.3). The α = π/2 limit tests (§9.2 rows 2–3) stay ●. **Bound (measured speed-ups of runs 20261006_120404 / 20261006_132352 converted with the measured local slope ρ_local = 1.9–2.2 per source at n\*):** on the build of run 20261006_132352 the row's residual against the best existing kernel is **≤ 0.07 source** (only the D = 19 μ lane is unmeasured; for s0 ≥ 20 the λ-form is measured faster than μ, λ/μ time 0.97 / 0.98 / 0.86 at D = 20 / 22 / 24); a high-degree μ kernel that does not yet exist (D = 18's cost up to D = 30, prefix storage out of registers) would add **+0.14–0.18 (n = 46), +0.19–0.24 (49), +0.24–0.30 (52), +0.35–0.43 (60)** — so the SBS row is a conservative bound, not a parity gap. (An earlier version of this estimate used ρ = 1.45 and over-stated both, ≤ 0.13 and +0.30–0.75.) |
| SBS coherence scan, t_S, lossy curve | coherence (p = ½) η = 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1: **64.2 / 56.8 / 54.1 / 52.4 / 51.1 / 50.0** (η = 0.5 end = the Fock η = 0.5 row of record, the incoherent twin; η = 1 end = the headline) → **coherence gap 14.2**; t_S m/n = 2: **49.8** (S = 10⁴ and untruncated; 98–99.5 % of t_S from the cost model above k = 22); lossy curve η = 0.9 / 0.8 / 0.7: **54.0 / 58.3 / 64.8** (shot floor, ≥ 1024 / 4096 / 16384 counted per point: shot means sized for a heavy-tail bias < 0.2 source; a stratified estimator failed its twin condition, run 20261006_222040, and is not used); η = 0.5 **not measured** (run 20261010_101720, stopped after 23 h: cost too heavy-tailed to converge within the run budget; no number from the partial run) | same code at both ends of each scan | as the row above | bracket with the shot floor / t_S | coherence 20261007_122313 (`floor_coherence_eta{0.6,0.7,0.8,0.9}_N1024_r28_20261007_h100.csv`) + 20261008_192343 (η = 0.5, `r30_fock_eta05_20261008_h100.csv`) + 20261007_090041 (η = 1); t_S 20261006_132352 (`sbs_rso_final_run_cost{,_fit}_r22_20261006_h100.csv`); lossy 20261007_094729 / 20261007_183855 / 20261008_233436 (`floor_sbs_eta{0.9_N1024,0.8_N4096}_r28_20261007_h100.csv`, `floor_sbs_eta0.7_N16384_20261008_233436_h100.csv`); superseded: coherence 64.6 / 56.5 / 54.1 / 52.6 / 51.3 / 49.7 of run 20261006_132352 | coherence, t_S, lossy η ≥ 0.7 **●**; η = 0.5 not measured (no row); the η = 0.5 end of the coherence scan is the twin limit test, ● (§9.2 row 3) |
| Gaussian (squeezed), S = 10⁴ | 41.5 (m = 2n) / 42.9 (m = n) | pair-sieve loop-hafnian chain rule, 2^{N/2} per detected photon — the best known exact algorithm for the input; dd tier beyond N ≈ 55 | own kernel: Triton sieve + torch batched subsets (`gaussian/sieve_triton`, `sieve_dd_triton`) | t_S, timing model validated 0.91–1.00 | 20260925; tiers 20260930 | ◐ by declaration: "best known algorithm, engineering effort comparable to the Fock kernel (one Triton kernel, lane plans, tiers)"; no common substrate exists — stated once in the paper, never as a gap |
| Borealis replays | 0.154 s/sample; 2.4·10¹⁶ H100-s | same engine, device replay (S = S_exp) | same | measured / priced / beyond-kernel tags | 20260927_190700; tiers 20260930 | ◐ (inherits the GBS row) |
| single photons through loops, N*(L) — the frontier table's row | N\* > 1259 (L ≤ 6), 1232 (L = 9), 491 (L = 10) | photon-number sector sampler | host-side sector-table build + torch | bracket | 20261002_073236 (`hybrid_frontier_fock_sector_20261002_h100.csv`) | ○ (host-side tables) — hollow in the frontier table |
| single photons through loops, dense control | 267.4 / 115.8 / 31.5 / 21.6 at L = 1 / 3 / 6 / 8 | native register propagation (no chain rule applies; cost = register size D) | torch complex128 register (`timeloop/kernel`) | bracket | 20261001_101803 (`hybrid_frontier_fock_20261001_h100.csv`; earlier timeloop run 2, 20260925) | ◐ register engine: the exponent is memory, not a sweep; quoted only as the SBS-L control |
| emitter / multi-emitter through loops (K) | one emitter N\* > 1259 (L ≤ 8), 1582 (L = 11); k = 2 / 3: N\* > 1574 / 1642 (L = 1 / 2), 1322 (k = 3, L = 1) | spins ⊗ carried-Fock sector sampler | host-side sector-table build (one-time, reported apart) + torch | t_S from ≥ 64-shot brackets | 20261002_222344 (one emitter); 20261005_083900 (k = 2–4) | ○ until the table build is off the timed path or priced separately (HYBRID.md "table building"); hollow in the frontier table |
| SBS-L / GBS-L (loop variants) | SBS-L α = π/4: N\* = 267.4 … 21.4 (L = 1 … 8); GBS-L: N\* ≥ 4000 … 7.3 (L = 1 … 8) | dense register (SBS-L) / capped register (GBS-L) | torch | bracket / t_S | SBS-L 20261001_101803 (`hybrid_frontier_sbs_20261001_h100.csv`), limit test 20261005_170728; GBS-L 20261002_203546 | SBS-L **●** (α = π/2 limit within 3 % of the LBS control at every L, ≤ 1.15×/sample, §9.2); GBS-L ◐ with its parent ("capped register" in the frontier table) |
| strong-then-sample (memory frontier) | n ≤ 11–13 | SLOS table + lookup | torch | memory, not rate | 20260930_141657 | — (not a rate row) |

State on 10-11 (every row above checked against the frontier paper's tables and supplement). (1) The SBS validation
gate (`bench/sbs_rso_validate.py`) is green (run 20261005_192609) and every quoted SBS number comes
from the shot-floor runs above; the lossy η = 0.5 SBS cell is printed "not measured". (2) The §5.4 check is superseded. (3) The NLBS amplitude pass (◐), the hybrid / sector loop rows (○, hollow
in the frontier table) and GBS-L (◐) carry their wording into the table's parity column; adaptive K ≥ 4 (○ here) is printed as
"route named from K = 4". (4) The ● rows are at parity; their numbers are the 256-shot-batch runs (lossless re-read run 20261009_141132) and the shot-floor runs.

### 9.2 Limit tests — one row per engine with a Fock limit (2026-10-05)

Protocol of the reference: `bench/find_threshold.py`, one H100 PCIe, 1 sample/s, Haar U, m = 2n, the Fock
engine on the production CUDA kernel (C&C 2020). Its own reproducibility sets the noise: 38.5 (budget 120 s,
run 20260930_082501) vs 38.4 (budget 600 s, run 20261005_064849) — **bracket noise ≈ 0.1–0.2 photon**.
"Path exercised" says whether the engine's own code runs in the limit (*proper*) or the limit is dispatched to
the Fock sampler (*tautological*: passes by construction, recorded, never read as parity).
**Pass criterion for every row:** in the limit, the engine's cost per sample at equal n is ≤ 1.3×
the Fock engine's, with the residual's cause named; otherwise the row fails (○). The n\* gap is reported beside
it but does not decide: where the cost curve is flat (ρ_eff → 1 per photon) a constant per-sample factor reads as
several photons of n\* (row 7).

| # | engine | Fock limit (twin state) | Fock reference n* (stamp) | engine in the limit (stamp) | path exercised | substrate in the limit | statistic | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | FBS (Fock) | — (the reference) | 38.5 m = 2n / 42.5 m = n (20260930_082501); 38.4 (20261005_064849); η = 0.5: 64.4 (20260917_182540, again 20261005_064849; the reference of the 256-shot limit tests below — cell of record 64.2, run 20261008_192343, 256-shot batches with ≥ 2048 counted) | — | — | production CUDA, C&C 2020 | bracketed mean | ● reference |
| 1 | FBS + loss / distinguishability | η = 1, indist = 1 | as row 0 | bit-identical to the lossless path (the "η = 1 / indist = 1 reduces exactly" invariant of every engine, DISTINGUISHABILITY.md invariants table) | proper, exact | same engine | — | ● (exact reduction, same code) |
| 2 | SBS, lossless | α = π/2 (p = 1): every source a Fock photon | 38.4 (20261005_064849) | **build before the kernel port**: rso `cuda_rect` 32.1 (20261005_064849) — **failed by 6.3** (~50×/sample; rectangular payload O(N²) vs O(N)). Fix (i) (k = n_live → Fock engine) passes by construction. **SBS validation gate** (`bench/sbs_rso_validate.py`, run 20261005_125116; `results/keep/sbs_rso_validate_f22d9a0_20261005_h100.csv`): (a) continuity, n = 34, s/sample at p = 0.8 / 0.9 / 0.95 / 0.99 / 1 = 0.254 / 0.572 / 0.720 / 0.434 / 0.0617 vs Fock 0.0682 — **RED** (7× step between p = 0.99 and 1: s0 = 1…6 strata on the μ-form, 14× the Fock kernel at s0 = 0, run 20261005_120751); (b) per-step cost at s0 = 0, n = 30: 7.15 vs 6.19 ms/shot = **1.15× GREEN** (≤ 1.3×); (c) exactness n = 6, four states, TVD within 2.5× the multinomial floor — **GREEN** | proper (square = "kernel": the rso chain on the Fock step kernels) | Fock kernel at s0 = 0; μ-form 1 ≤ s0 ≤ 3, λ-form above (`choose_form`) | bracket; s/sample at n = 34 for (a) | **● (final build, run 20261006_012054; `results/keep/limit_{fock_lossless,rso_pi2_lossless}_final_20261006_h100.csv`)**: rso chain on the Fock step kernels, its own source order and draws, n\* = **38.4** (38 at 1.32/s, 39 at 0.71/s) vs the Fock engine in the same run **38.4** (1.26/s, 0.73/s); per sample at equal n (256 shots each) 1.06 / 0.94 / 0.95 / 1.03× at n = 34 / 36 / 38 / 39 — parity within shot noise, no residual to name. Validation gate checks (a)–(c) GREEN on this build (run 20261006_001928). History: the build before the kernel port failed by 6.3, see left |
| 3 | SBS, the twin | α = π/2, η = ½ (deterministic sources behind a 50 % channel) | 64.4 (20260921_071710, LOSS.md §7; reproduced 20261005_064849, `limit_fock_eta05_20261005_h100.csv`) | build before the kernel port: 53.4 (20261005_064849) — **failed by 11.0** (same ~50×/sample as row 2; loss handling is not the cause — `sbs_rso` thins first). **Final build** (coherence scan, run 20261006_012054, η = 0.5 end, `coherence_p50_rso_final_sbs_20261006_h100_eta0.5.csv`): **64.6** | proper | as row 2 | bracket; s/sample at equal n | **●**: n\* 64.6 vs 64.4; per sample at equal n vs the Fock engine of run 20261005_064849 (same protocol, 256 shots): 0.58 / 0.64 / 1.24× at n = 56 / 64 / 65 — within 1.3×, the spread is the heavy-tailed lossy shot cost on both sides, no systematic residual. The Fock η = 0.5 cell of record is 64.2 (run 20261008_192343, 256-shot batches with ≥ 2048 counted; 64.4 superseded); this limit test compares both engines on the same 256-shot protocol and stands |
| 4 | EBS (quantum-dot emitter) | χ = 1, one mode per photon (L = 1): a product of single photons | 38.5 (20260930_082501); same job: **38.5** (20261005_182819, n = 38 at 1.25/s, 39 at 0.79/s) | **Run 20261005_182819** (`--engine emitter --emitter-state fock --kernel auto`, budget 240 s; `results/keep/threshold_emitter_fock_c2_20261005_h100.csv`, `threshold_fock_c2_ref_20261005_h100.csv`): frozen-spin χ = 2 Fock state **n\* = 35.3** (n = 35 at 1.24/s, 36 at 0.60/s, ρ_eff 1.875) vs Fock 38.5 in the same job — **−3.2**. Per sample at equal n: 472 vs 66.1 ms (n = 34) = **7.2×**, 1656 vs 246 ms (n = 36) = **6.7×**. Cause named: the payload — `cuda_emitter` (the kernel at the frontier steps under `auto`) runs per term and column 2 × 2 complex slot matrices, prefix/suffix chains and 2 × 2 outer-product accumulators where `cof_multi` runs scalar all-but-one products; same odometer, tiers and lane plan (module docstring), and the frozen spin (M = diag(1, 0)) is not specialised. FMA ratio per term estimated, not measured. Earlier, **CPU exactness (10-05)**: χ = 1 `EmitterMPS` strong probabilities = |Per|²/Πs! to 2e−17 (n = 4, m = 8; n = 5, m = 10) and `sample_torch` TVD 0.0196 / 0.0485 vs floor 0.0189 / 0.0489 (10⁵ shots); the same state embedded at χ = 2 (spin frozen, M = diag(1, 0)) — the form the GPU kernels take, which are χ = 2 only (`cuda_emitter` 2 × 2 chains, Triton asserts χ = 2) — matches `FockSLOSKernel` to 1e−17, TVD 0.0186 vs 0.0189. The χ = 2 rows (30.9 at m = 2n) are not a limit test | proper (GPU and CPU) | `cuda_emitter` (2 × 2 payload, the Fock kernel's odometer/tiers) at the frontier steps, seq Triton below | bracket; s/sample at equal n | **● with the residual stated** — **verdict run** (run 20261005_234844, the production `cuda_emitter` build, 4× shots, `results/keep/emitter_fock_throughput_4x_53c8494_20261005_h100.csv`), cost per needed op vs `cof_multi`: **frozen-spin limit 0.98–1.06 (n = 34) / 1.00–1.09 (n = 36) — passes**; **cluster NL = 2 (the 30.9 row) 1.32–1.43 / 1.35–1.48** — residual = the 2 × 2 block scan of the 4-thread lane split (56 flops per term·column; executed 216 vs the algorithmic minimum 160 at NL = 2), the kernel running at the Fock kernel's efficiency per executed flop (emitter 5.1–5.4 Tflop/s on its executed count at NL = 2, 7.5–7.9 at NL = 1, Fock 4.8–5.2); the scan-free 2-thread lane (PE_SYM) is exact but 2–6× slower at the frontier tiers (register spills). **Frontier numbers on this build (run 20261006_004008, `results/keep/threshold_emitter_fock_c2_53c8494_20261006_h100.csv`, `frontier_loss_emitter_r18_53c8494_eta1_20261006_h100.csv`)**: frozen-spin limit n\* = **35.6** (n = 35 at 1.52/s, 36 at 0.77/s; was 35.3 on the old kernel), 5.6× / 5.3× per sample vs the Fock reference of run 20261005_182819 at n = 34 / 36 = the NL = 1 payload minimum (5.2–5.6×) — ≈ 1.0 per needed op; **emitter cluster m = 2n, η = 1: 31.0** (n = 30 at 1.97/s, 31 at 0.98/s; ρ_eff 1.94) vs 30.9 (run 20261003_103957, old kernel) — +0.1, inside the bracket noise. **The headline is quoted from the parity-build run, 31.0 (20261006_004008); the other emitter rows (m/n 2.5–12, lossy η < 1, V < 1) stay as quoted from runs 20261002_003041 / 20261003_103957 / 20261003_121759 on the old kernel, never faster than the fixed one → conservative, not re-run.** History of the row: no frozen-spin specialisation (it would be tautological). The residual is read as the price of the χ = 2 payload, which a genuine emitter state needs: **payload factor** (from the kernel sources, counted not measured; fp64 real ops per Gray term per column, complex mul = 6, mul-add = 8): `cof_multi` = column-sum update 4 + product chain 6 + all-but-one accumulation 12 and **one fp64 division** (1/\|s\|²) ≈ **22 + 1 div**, nothing read from memory; `cuda_emitter` = column-sum update 4·NL + slot matrix G (4 entries × NL mul-adds) 32·NL + forward chain p ← Gp 28 + backward chain qᵀ ← qᵀG 28 + accumulation w p qᵀ 40 = **96 + 36·NL**, plus a 2 × 2 product 56 when TPL > 1 (block scan), plus **4·NL complex loads of M̃** per column per term, and 11·NL-ish complex registers per column vs 2 (occupancy). NL = 1: 132 (TPL = 1) / 188 (TPL > 1) → **6.0× / 8.5×** the Fock count without the division, ≈ 4–6× if the division costs ~10 ops; NL = 2: 168 / 224 → 7.6× / 10×. The measured 6.7–7.2× sits inside that band. **Algorithmic minimum a genuine χ = 2 state needs** (the yardstick, independent of today's kernel): the 2 × 2 spin matrices do not commute, so the all-but-one environment of column c needs the forward vector chain p_c = G_{c−1}…G_0 s0 (one 2 × 2 mat-vec, 28) *and* the backward chain q_cᵀ = f†G_{K−1}…G_{c+1} (28) — the scalar division trick has no 2 × 2 analogue; the per-column output is the 2 × 2 environment Σ w p_c q_cᵀ (outer-product mul-add 32, w folded into q once per term; contracting with M̃_{c,l} inside instead costs 42·NL); the slot matrix G_c = Σ_l s_{c,l} M̃_{c,l} 32·NL (for NL = 1 applying s·M̃ inside the two chains ties at 80); column sums 4·NL. **Minimum = 88 + 36·NL per term·column = 124 (NL = 1) / 160 (NL = 2)**, against the Fock minimum ≈ 22 + 1 div (division trick; 24 with prefix/suffix chains) → **payload factor 5.2–5.6× (NL = 1), 6.7–7.3× (NL = 2)**. `cuda_emitter` executes the minimum + 8 (w scaling) at TPL = 1 and + 56 (the 2 × 2 block scan of the lane split, an implementation choice the scalar kernel gets for O(1)) at TPL > 1. **Throughput test**: per-op throughput of `cuda_emitter` vs `cof_multi` at the tiers of n = 34–36 (achieved fp64 rate, TPL/GS used, occupancy); pass iff (measured per-sample ratio) / (minimum payload factor for the NL of the run) ≤ 1.3× — the run's tiers are **NL = 1, GS = 1, TPL = 4** (same counts: 124 needed / 180 executed vs 23 + 1 reciprocal), so 6.7–7.2 / 5.4–5.6 ≈ **1.2–1.3 — at the line**; run 20261005_193653 (`bench/emitter_fock_throughput.py`, per-op throughput + the untried (TPL, GS) grid at tiers 32 / 36) decides. **Measured (run 20261005_195203, `results/keep/emitter_fock_throughput_cf9fc57_20261005_h100.csv`)** — cost per *needed* op relative to the Fock kernel: NL = 1 (frozen spin, this limit) 1.06–1.38× (1.21–1.38× at n = 34–36); **NL = 2 (the genuine cluster state of the 30.9 row) 1.15–1.83× (1.63–1.83× at n = 34–36) — over 1.3×**. Cause named: register pressure — `cuda_emitter` 255 registers, 688–1064 B local-memory spill, 8 warps/SM vs `cof_multi` 118–128 registers, 160–184 B, 16 warps. **Fix candidates** (runs 20261005_213240 and 20261005_221051; `results/keep/emitter_r18_variants_{73ed125,tpl_6745636}_20261005_h100.csv`): best exact variant (w + accumulators in shared memory, TPL 4; within 4.5e−13 of production) → frozen-spin limit **0.99–1.13× — passes**; cluster NL = 2 **1.28–1.44×** — at/just above the line (the yardstick is algorithmic; the lane-split scan is not counted). **Rule for this row**: where NL = 2 straddles 1.3, the row is ● with the residual stated — "≤ 1.44× per needed op; Fock-kernel efficiency per executed flop; the remainder is the lane-split scan"; no ○ — ≤ 1.3× → ●; above, the substrate gap is fixed in the kernel and the emitter frontier (30.9 at m = 2n and the rows built on it) is re-measured on the fixed kernel if the number moves, then ● |
| 5 | aFBS (adaptive) | no adaptive step (K = 0) | 38.5 (frontier.csv, nine ratios) | K = 0 *is* `fock_sample` (`test_identity_policy_reduces_to_plain_boson_sampling`): tautological. **Non-tautological near-limit**: forced-prefix chain K = 1 / 2 / 3 at m = 2n measured 38.4 / 38.2 / 38.1 (since 10-07 at 512 shots, §9.3: K = 1 … 6, 8 = **38.3 / 38.2 / 38.1 / 37.6 / 34.8 / 34.4 / 33.8**, runs 20261007_022012 / _030408 / _061124) against the cost model 38.4 / 38.4 / 38.3 (runs 20260922_074317 / _092841 / _131122; ADAPTIVE §5c, `results/keep/adaptive_crossings.csv`) — continuous in K, toll ≤ 0.4 (per-sample reading: ρ_eff 1.816^0.1 / ^0.3 / ^0.4 ≈ 1.06 / 1.20 / 1.27× per sample at K = 1 / 2 / 3, inside 1.3×; cause named = the forced-prefix level chain, matched by the cost model) | tautological at K = 0; proper at K ≥ 1 | production CUDA (`fock_sample(prefix=, avoid=)`); level tables on the host | bracket | **● for K ≤ 3** (continuity measured); K ≥ 4 ○ as in §9.1 |
| 6 | NLBS (non-linear site) | zero non-linearity: φ = 0 (identity), φ = π (one branch, a phase shifter) | 38.5 (20260922_211901, same run) | 38.5 / 38.3 (20260922_211901; NONLINEAR §5b φ scan) — **tautological**: one branch with ‖c‖₁² = 1 is dispatched to the linear sampler (`nonlinear.py`, "linear: nothing to reject"). **Non-tautological evidence**: the measured toll is the cost law ‖c‖₁²(1 + κB) with one κ = 0.55 ± 0.08 over nine ratios and the φ scan, acceptance = 1/‖c‖₁² measured (0.44–0.59 vs 0.50 at B = 2) — the rejection layer prices exactly what it should over the Fock chain | tautological at φ ∈ {0, π}; proper at B ≥ 2 | proposals: production CUDA; branch amplitudes: torch Glynn | bracket | **◐** (limit holds; the amplitude pass is a torch sweep — §9.1 wording) |
| 7 | hybrid / loop column | "L = 0 → SBS" as first proposed | — | **ill-posed**: at L = 0 there is no interferometer (bins go straight to detectors); the per-tick loop row has no spatial-SBS twin (demux × loops = low-depth class, ARCHITECTURE.md engine map). Exactness limits CPU-tested ≤ 1e-12 (L = 0 → `SequenceKernel`, no emitter → `LoopRegisterKernel`, T = 1 → the stage, SBS-L α = π/2 → Fock through loops bit for bit; 48 passed on the H100, run 20261001_101803). Closest frontier evidence (not a limit test): SBS-L α = π/4 vs single photons on the same register, 267.4 / 237.4 / 112.9 / 62.5 / 31.3 / 21.4 vs 267.4 / 241.4 / 115.8 / 62.8 / 31.5 / 21.6 at L = 1/2/3/4/6/8 (20261001_101803). **Well-posed per-tick test, run 20261005_170728** (`results/keep/hybrid_frontier_sbs_halfpi_20261005_h100.csv`): SBS-L at α = π/2 vs the dense Fock (LBS) control of run 20261001_101803, same register D at every N, N\* at L = 1/2/3/4/6/8 = 265.4 / 237.2 / 112.3 / 62.5 / 31.0 / 21.4 vs 267.4 / 241.4 / 115.8 / 62.8 / 31.5 / 21.6 (−2.0 / −4.2 / −3.5 / −0.3 / −0.5 / −0.2). Per sample at equal N: 1.01–1.15× the Fock path (re-read from the two CSVs: L = 2 1.07–1.13×, L = 3 1.15×, L = 4 1.10–1.13×), a constant independent of α (0.99–1.08× the α = π/4 path) = the SBS injection step (`SuperpositionLoopKernel._emit`). The N\* gaps at L = 2–3 are that constant read through a flat cost curve (ρ_eff per photon at the bracket 1.03 / 1.04–1.06: ln 1.13 / ln 1.025 ≈ 4.8, ln 1.15 / ln 1.04 ≈ 3.6 photons) — systematic, not bracket noise | proper (the SBS-L chain runs in full; α = π/2 is not dispatched) | torch register (both rows) | bracket; s/sample at equal N | **●**: ≤ 1.15× per sample ≤ 1.3×, cause named (the injection step); N\* within 3 % (−4.2 / −3.5 at L = 2–3 from the flat curve). The spatial twin "L = 0 → SBS" does not exist and is not a test |
| 8 | GBS (squeezed) | none — a Gaussian input has no Fock limit | — | — | — | own kernel (Triton sieve + torch subsets) | t_S | **◐ by declaration**: "own kernel, best known algorithm (pair-sieve loop-hafnian chain, 2^{N/2} per detected photon), engineering effort comparable to the Fock kernel" — no limit test applies |

Rows that pass by construction are not counted as limit tests. The three acceptance checks of the SBS validation
gate (`bench/sbs_rso_validate.py`; row 2: (a) continuity on the rso path proper, (b) per-step cost ≤ 1.3× the Fock
chain at equal q, (c) exactness and the row quoted only from the (ii) build) are the SBS limit test: (a) green on the
final build (run 20261006_001928) and the α = π/2 and twin runs within 1.3× per sample of the Fock engine at equal n
→ **rows 2–3 ● (10-06)**. Row 4: ● with the residual stated. Row 7: ●. Every row with a Fock limit is now ●; NLBS
and GBS stay ◐ as stated in their rows. The SBS α = π/4 frontier rows of §9.1 are ● from the re-run of
run 20261006_132352; the lossy SBS curve is in §9.1 (shot-floor runs).

### 9.3 Regression check — the headlines re-measured on one pinned build (2026-10-07)

Run **20261007_005628** (one pinned build; a preliminary run 20261006_221903 green), one H100
PCIe. Each Clifford-substrate headline's **own bracket points** re-measured with its original bench, arguments, budget and
shots; pass = each point within ±10 % of its original rate. Since 09-30 only `cuda_rect` (SBS) and `cuda_emitter`
(emitter) changed under `clifford/` (10-06): the emitter row tests the fixed kernel; the Fock, aFBS, NLBS and loop rows
test that nothing else drifted. Kept: `results/keep/regression_*_20261007_h100.csv`.

| headline (original run) | point: original → now (samples/s) | Δ | n\* now | verdict |
|---|---|---|---|---|
| Fock m = 2n, 38.5 (20260930_082501) | n = 38: 1.257 → 1.254; n = 39: 0.792 → 0.788 | −0.2 / −0.5 % | 38.5 | pass |
| Fock m = n, 42.5 (20260930_082501) | n = 42: 1.160 → 1.156; n = 43: 0.844 → 0.839 | −0.3 / −0.6 % | 42.5 | pass |
| emitter m = 2n, 31.0 (20261006_004008) | n = 30: 1.968 → 1.963; n = 31: 0.981 → 0.981 | −0.3 / 0.0 % | 31.0 | pass |
| Fock η = 0.5, 64.4 (20261005_064849) | n = 64: 1.111 → 1.111; n = 65: 0.834 → 0.835 | 0.0 / +0.1 % | 64.4 | pass |
| NLBS φ = π/2, 36.0 (20260922_211901) | n = 35: 1.804 → 1.849; n = 36: 0.977 → 1.055 | +2.5 / +8.0 % | 36.1 (bracket moves to 36/37: 1.055 / 0.548) | pass (+0.1, bracket noise) |
| SBS-L α = π/2, L = 8, 21.4 (20261005_170728) | N = 21: 1.189 → 1.191; N = 22: 0.808 → 0.809 | +0.2 / +0.1 % | 21.5 (21.448 → 21.452: the rounding edge) | pass |
| aFBS K = 1, m = 2n, 38.4 derived (20260922_074317) | n = 36: 4.338 → 4.273; n = 38: 1.318 → **1.100** | −1.5 / **−16.6 %** | **38.2 measured** (38 at 1.100/s, 39 at 0.616/s; n = 40 at 0.366/s) | **fails at n = 38** — see below |

**aFBS K = 1.** The K = 0 control of the same bench and seeds is unchanged at every n (n = 38: 1.579 → 1.578/s; K = 0
bracket 38.7 as in 20260922_074317), and the adaptive code is untouched since 09-22. The miss is in the K = 1 overhead
itself: time ratio K = 1 / K = 0 = 1.008 / 1.435 / 1.309 / 1.011 at n = 36 / 38 / 39 / 40 at 128 shots (draws per sample
2.09–2.53), against 1.198 at n = 38 in the original run — the overhead of 128 adaptive shots varies by ±20 % between
instances. Not a regression; a statistic too noisy for a ±10 % test. This run is also the first measured bracket of the
K = 1 headline (the original stopped at n = 38 above 1/s; 38.4 was derived from the K = 0 fit and a time ratio,
ADAPTIVE.md): 38.2 at 128 shots, the bench's fit over n = 36–40 38.3. **Re-measured at 4× shots — run
20261007_022012 (512 shots): n = 38 at 1.187/s, 39 at 0.623/s → K = 1 = 38.3, the quoted value**
(K = 0 in the same run 38.4 vs canonical 38.5; canonical minus the measured overhead ×1.06 / ×1.19 gives 38.4 / 38.2). The
same run shows why 128 shots misled: this bench's fixed-batch K = 0 rate at n = 38 is 1.578/s at 128 shots and 1.256/s at
512 (find_threshold: 1.257/s) — 128-shot brackets of `adaptive_frontier.py` read high by up to ~0.3; ratios at equal batch
are unaffected. `results/keep/afbs_K1_x4_20261007_h100.csv`. **Every quoted m/n = 2 row re-measured the same way (runs
20261007_030408 and 20261007_061124, original chains, 512 shots): K = 2 38.2, K = 3 38.1, K = 4 37.6
(was 37.0), K = 5 34.8 (34.0), K = 6 34.4 (34.2), K = 8 33.8 (31.6)** — none moved down. `results/keep/afbs_K{23,4,5,6,8,8b}_x4_20261007_h100.csv`.
