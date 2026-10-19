# Memory frontier of strong simulation (2026-09-20)

Second simulability frontier: the largest n whose **full output distribution** fits in a memory
budget (80 GiB, one H100). Once the table exists, sampling is a lookup (~10^8 draws/s), so the
frontier is set by bytes, not by rate. Pure counting, no benchmark:
`python bench/memory_frontier.py [GiB]`.

## Short version

For Fock and emitter inputs the frontier is the size of the last layer: the table is 4·M_n bytes for
both (the emitter is traced as the last layer is formed); the build adds the parent layer,
8χ·M_{n−1} ≈ 8χ·M_n/(1+c), worth 0.2–0.4 photon. One curve: 18 / 13 / 10 / 8 / 6 photons at
m/n = 1 / 2 / 4 / 8 / 16. The paper and the deck carry only that curve and the sampler n*; the other columns here are the breakdown (the stored-edge kernel is an implementation
artefact that explains the measured n ≤ 11, not a frontier).

SBS is counted on the axis of its sampling frontier (`find_threshold --engine sbs`): n SOURCES, m = c·n,
no post-selection. It has no single last layer — k ~ Bin(n, sin²α) whatever U, a sample is a draw of k
then a lookup in layer k — so the table is the graded basis Σ_{k≤n} M_k = C(m+n, n) = (1+1/c)·M_n,
whatever α: the Fock table plus the layers below it. Build: 8·C(m+n−1, n−1) + 4·C(m+n, n) (last source
pulled into p). That is 0.1–0.4 source on the frontier and the same integer as Fock at every ratio, so
the paper, the deck and the figure carry ONE memory curve for the three inputs. (An earlier version of
this note put SBS on a detected-photon axis with 2n sources, C(m+2n, 2n), 4–5 lower: wrong axis.)

**Statement for the paper.** Within the spatial family strong simulation has ONE memory frontier,
a property of the output space, independent of the input modality and steep in m/n; all the
modality dependence is in sampling. Limits: Kraus operators instead of the traced distribution are
χ² complex entries per output state (8× the bytes at χ = 2, ≈ 1 source at m = 2n); a large bond
dimension makes the χ parent vectors dominate once χ ≳ (1+c)/2 (then −log χ / log ρ_mem sources);
the temporal family (loops, emitter trains) is not covered — its strong-simulation object is the
register (D, 2^k), expected to bound memory and rate alike (not counted).

## Counting

- Fock layer: M_k = C(m+k−1, k). M_{n−1}/M_n = n/(m+n−1) ≈ 1/(1+c) with c = m/n.
- Growth per photon at fixed c: ρ_mem = (1+c)^{1+c}/c^c — 4 (c=1), 6.75 (c=2), 12.2 (c=4),
  23.1 (c=8), → e·c. Compare ρ ≈ 1.8–1.95 for the sampling frontier.
  n_mem ≈ ln(budget/bytes-per-state)/ln ρ_mem, minus the ½·ln n prefactor.
- Stored edges of the current kernel: Σ_k |E_k| = n·M_n exactly (|E_k| = m·M_{k−1}).

Memory models (complex64 amplitudes = 8 B, float32 probabilities = 4 B):

| model | peak bytes | what it is |
|---|---|---|
| table | 4·M_n | the distribution alone — upper bound for any builder |
| pull | 8·M_{n−1} + 4·M_n | implicit (on-the-fly rank) kernel; last layer pulled chunk-wise, child-major, squared straight into p |
| push | 8·(M_{n−1}+M_n) | implicit kernel, two state vectors (DESIGN.md §3 `slos/implicit`) |
| edges | ≈ 9n·M_n + push | current `FockSLOSKernel` (int32 parent+child, int8 states); predicts 14 GB at n=11, m=22 vs 16 GB measured |
| emitter χ=2 | 16·M_{n−1} + 4·M_n (pull, traced on the fly) / 16·(M_{n−1}+M_n) (push) | bond-carrying SLOS |
| SBS | 8·C(m+n−1, n−1) + 4·C(m+n, n) | n sources, every photon number (graded basis), last source pulled into p |
| collision-free | same with C(m, k) | post-selected table, not the full distribution |

## Result, 80 GiB (integer n; interpolated n* in brackets)

| c = m/n | ρ_mem | table | Fock pull | Fock push | Fock current (edges) | emitter pull | emitter push | SBS | collision-free only | sampling n* at 1 Hz (Fock / emitter) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.00 | 19 (19.1) | 18 (18.6) | 18 (18.3) | 16 (16.4) | 18 (18.3) | 17 (17.8) | 18 (18.1) | degenerate (C(n,n) = 1) | — |
| 1.5 | 5.38 | 15 (15.7) | 15 (15.4) | 15 (15.2) | 13 (13.5) | 15 (15.2) | 14 (14.7) | 15 (15.1) | 23 | — |
| 2 | 6.75 | 13 (13.7) | 13 (13.5) | 13 (13.2) | 11 (11.9) | 13 (13.3) | 12 (12.8) | 13 (13.2) | 17 | 38.5 / 30.6 |
| 3 | 9.48 | 11 (11.6) | 11 (11.4) | 11 (11.2) | 10 (10.1) | 11 (11.3) | 10 (10.9) | 11 (11.3) | 13 | 36.8 / 29.9 |
| 4 | 12.2 | 10 (10.4) | 10 (10.3) | 10 (10.0) | 9 (9.1) | 10 (10.1) | 9 (9.8) | 10 (10.2) | 11 | 36.2 / 29.5 |
| 6 | 17.7 | 9 (9.0) | 8 (8.9) | 8 (8.7) | 7 (8.0) | 8 (8.9) | 8 (8.5) | 8 (8.9) | 9 | 35.3 / 29.1 |
| 8 | 23.1 | 8 (8.2) | 8 (8.2) | 7 (8.0) | 7 (7.3) | 8 (8.1) | 7 (7.7) | 8 (8.1) | 8 | — |
| 12 | 34.0 | 7 (7.3) | 7 (7.3) | 7 (7.1) | 6 (6.5) | 7 (7.2) | 6 (6.9) | 7 (7.2) | 7 | — |
| 16 | 44.8 | 6 (6.8) | 6 (6.7) | 6 (6.6) | 6 (6.0) | 6 (6.7) | 6 (6.4) | 6 (6.7) | 6 | — |
| m = n² | — | — | 8 | — | — | — | — | — | 8 | — |

(sampling n* from `results/keep/frontier_20260920_fock_emitter_h100.csv`; no SBS sampling frontier on the source axis yet.)

## Reading

1. **The memory frontier is steep in c, the sampling frontier is flat.** 18 → 13 → 10 → 8 photons
   for c = 1 → 2 → 4 → 8, against 38.5 → 35.3 for c = 2 → 6. Base 4–45 per photon instead of ~1.9.
2. **Memory buys almost nothing.** Photons per doubling of the budget = 1/log2 ρ_mem: 0.5 (c=1),
   0.36 (c=2), 0.28 (c=4). 8× H100 (640 GiB) gives +1 photon at c = 2 (13.5 → 14.6). The same
   holds for the byte-per-state constant: table vs push (4×) is < 1 photon.
3. **Engine ordering collapses.** Emitter costs χ× memory = 0.2–0.5 photon (vs 8 photons on the
   sampling frontier); with the pull-form last layer, only the parent layer is doubled and the
   gap is ≤ 0.3. SBS (n sources, every photon number)
   is (1+1/c)× the Fock table: 0.1–0.4 source.
4. **The current kernel is one software step away**: stored edges cost 9n B/state, i.e. 1.5
   photons at c = 2 (11 vs 13). The implicit kernel is the whole remaining headroom; pull-form
   adds 0.3.
5. **Build time is not the constraint**: the frontier instances have n·M_n ≈ 7–9·10^10 edges; at
   the measured 1.4·10^10 edges/s (n = 11: 98 ms) that is ~5–10 s (the implicit kernel pays
   rank arithmetic on top; still minutes at most). The wall is bytes.
6. **Collision-free tables** only matter for c ≲ 3 (n = 17 at c = 2, 23 at c = 1.5), where the
   collision-free mass is small — a post-selected experiment, not the sampling task. For
   c ≳ 6 they coincide with the full table.
7. Where the table wins: between the two frontiers (n = 14…38 at c = 2) only the chain-rule sampler
   exists. Below n_mem the table amortises after t_build/t_sample draws; the Clifford sampler is
   already ~10^4–10^6 samples/s there, so the table matters for ≳ 10^6–10^7 draws (or for
   anything that needs probabilities: training, certification, TVD).

Caveats: 80 GiB nominal (usable ~78; shifts n* by < 0.02). complex64 assumed — p ~ 10^−10 is fine in
float32 relative precision, CDF needs float64 block sums (M_n/4096 extra, negligible). complex128
costs 0.2–0.5 photon.

## GBS: n squeezers (2026-10-01; pure counting, `bench/memory_frontier.py`)

**The object needs a convention.** A squeezed vacuum can emit any even number of photons, so the GBS output
distribution is unbounded and "the full distribution" is a table up to a global photon cutoff N_cut, chosen so that
the mass left out is ≤ ε (docs/GAUSSIAN.md §7; a global cutoff, not Strawberry Fields' per-mode D^m truncation). For
a lossless undisplaced input only even layers are non-zero and truncation is exact below N_cut. Axis: n squeezers at
0.5 photons each (the sampling frontier's point), m = c·n. Memory model: float32 probabilities of every even layer
≤ N_cut, plus the Gaussian SLOS recurrence's working set (c_N = −(1/N) Σ_s tanh r_s R(u_s) R(u_s) c_{N−2}: the
complex64 layers N−2 and N−1 held while layer N is pulled into p) — `gbs_bytes`. Proposed convention: **ε = 1/S with
S = 10⁴**, the same truncation as the sampling statistic t_S (D2); ε = 10⁻³ and 10⁻⁶ as sensitivity.

| c = m/n | ε = 10⁻³: n* (N_cut) | **ε = 10⁻⁴ (1/S)** | ε = 10⁻⁶ | Fock pull (for reference) | GBS sampling n* at 1 Hz, S = 10⁴ |
|---|---|---|---|---|---|
| 1 | 13.6 (26) | **12.2 (30)** | 10.5 (38) | 18.6 | 42.9 |
| 1.5 | 10.3 (22) | **8.8 (26)** | 7.6 (34) | 15.4 | 41.8 |
| 2 | 8.3 (20) | **7.2 (24)** | 5.9 (30) | 13.5 | 41.5 |
| 3 | 6.3 (18) | **5.2 (22)** | 4.3 (28) | 11.4 | 40.6 |
| 4 | 5.2 (16) | **4.2 (20)** | 3.4 (26) | 10.3 | 39.9 |
| 6 | 4.1 (14) | **3.1 (18)** | 2.4 (24) | 8.9 | 39.1 |
| 8 | 3.2 (14) | **2.5 (16)** | 1.8 (22) | 8.2 | 38.2 |
| 12 | 2.4 (12) | **1.8 (14)** | 1.3 (22) | 7.3 | 37.4 |

(N_cut at the integer n below n*. Table alone, no working set: 8.6 / 7.5 at c = 2 for ε = 10⁻³ / 10⁻⁴.) Squeezing at
c = 2, ε = 10⁻⁴: 0.1 / 0.25 / 0.5 / 0.75 / 1.0 / 1.5 photons per squeezer → n* = 15.6 / 9.4 / 7.2 / 6.1 / 5.5 / 4.8
(N_cut = 12 / 18 / 24 / 30 / 34 / 44). At the headline, n = 7 squeezers in 14 modes: N_cut = 24, 50 GiB; n = 8 needs
869 GiB.

**Reading.**
1. **GBS is off the one curve.** On the source axis its memory frontier is 7.2 at c = 2 against 13.5 for Fock,
   emitter and SBS (and falls faster with c: 1.8 at c = 12 against 7.3). The reason is the object, not the engine: a
   Fock source puts exactly one photon in the table, a squeezer puts a whole photon-number distribution, and the
   cutoff that captures 1 − 10⁻⁴ of it sits at N_cut ≈ 7× the mean photon number at the frontier (24 against 3.5).
   The table therefore lives on few modes and many photons (24 photons in 14 modes), where the output space
   C(m + N, N) grows fastest.
2. **The gap between the two frontiers is widest for GBS**: 7.2 (memory) against 41.5 (sampling) at c = 2, i.e.
   34 squeezers where only the chain-rule sampler exists (Fock: 13.5 against 38.5). The sampler's cost is in
   detected photons (2^{N/2}), the table's in squeezers × cutoff.
3. **Convention sensitivity**: ~1 squeezer per factor 10 in ε at c = 2 (8.3 / 7.2 / 5.9 for 10⁻³ / 10⁻⁴ / 10⁻⁶);
   squeezing moves it like the sampling frontier does (15.6 at 0.1 photons, 4.8 at 1.5).
4. Loss is not counted: a lossy detected distribution is exact only from the density-matrix (doubled, 2m-mode)
   recurrence, squared memory; a thinned lossless table is a bounded approximation (GAUSSIAN.md §7). So the lossy
   GBS memory frontier is below these numbers.

Consequence for the statement above ("one memory frontier … independent of the input modality"): it holds for the
three inputs with one photon per source; **GBS is the exception, 6 squeezers below at c = 2**, and the paper should
say so (proposed wording for B: "strong simulation has one memory frontier for single-photon-type inputs, a property
of the output space; squeezed inputs sit lower, because each squeezer fills every photon number up to the cutoff").
