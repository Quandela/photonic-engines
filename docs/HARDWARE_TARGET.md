# Emitter-state boson sampling: the minimal advantage experiment, from the engine's frontier

One page for the hardware team. Numbers from `bench/frontier_loss.py` on one H100 PCIe (runs
20260917_182540 and 20260918_174153, docs/LOSS.md §7); the metric is the photon number n* at
which the best classical sampler in this repository — exact, one output pattern per sample —
falls to **one sample per second**, for an m = 2n universal interferometer.

## The two frontiers

| transmission η (source → detector, per photon) | 1 | 0.9 | 0.8 | 0.7 | 0.5 |
|---|---|---|---|---|---|
| Fock input (n single photons), n* emitted | 38.3 | 41.2 | 45.0 | 49.2 | 64.2 |
| … mean photons detected at n* | 38.3 | 37.1 | 36.0 | 34.4 | 32.1 |
| linear cluster from one emitter, n* emitted | 31.0 | 33.6 | 36.8 | 40.9 | 53.5 |
| … mean photons detected at n* | 31.0 | 30.2 | 29.4 | 28.6 | 26.8 |
| … interferometer modes m = 2n* | 62 | 67 | 74 | 82 | 107 |
| emitter lead in detected photons | 7.3 | 6.9 | 6.6 | 5.8 | 5.3 |

Rows of record (one number per cell, 10-06; every cell on the 256-shot protocol since 10-09: 256-shot batches only,
≥ 2048 counted shots at both bracket points): Fock η = 1 run 20261009_141132
(256-only re-read; 38.5 of run 20260930_082501 superseded, 10-09), η = 0.9 run 20261008_071215 (41.4 of run
20260917_182540 superseded, 10-08),
η = 0.8 run 20261008_085321 (44.7 of run 20260917_182540 superseded, 10-08),
η = 0.7 run 20261008_104426 (residual tail uncertainty ~±0.2 — at n = 50 the first 256-shot batch read 0.373/s,
the 2048-shot total 0.712/s; 49.4 of run 20261003_090228 superseded, 10-08), η = 0.5 run 20261008_192343 (residual tail — the lower bracket point n = 64
reads 1.088/s, 9 % above target after a 25 % drop from the 256-shot ramp, so the bracket could move to 63/64; 64.4 of runs
20260921_071710 / 20261005_064849 superseded, 10-08); emitter η = 1 run 20261006_004008 (parity build; 31.0 confirmed on the 256-shot protocol by run 20261009_141132), η = 0.7 run 20261008_122031 (parity
build; 40.3 of run 20261003_103957 on the old kernel superseded, 10-08), η = 0.5 run 20261008_133143 (parity build; heavy tail — 2048-shot
rates 1.9–2.6× below the 256-shot ramp, bracket 53.5 vs five-point fit 53.6; 53.9 of run 20261003_103957 superseded, 10-08),
η = 0.9 / 0.8 run 20261008_210721 (parity build; 33.6 confirmed, 36.3 of run 20261003_103957
superseded, 10-08). The parity build is the emitter kernel at the same optimisation level as the Fock sampler
(docs/ALGORITHMS.md §9). Every lossy cell of both rows is on the 256-shot protocol.

Both curves follow n*(η) = n*(1) · log ρ / log(1 − η + ηρ): a lossy sample costs the average
over the binomial number of survivors of a lossless sample, and that average is dominated by
the lucky high-count shots (ρ = 1.81 per photon for Fock, 1.99 for the emitter at m = 2n).

## Partial distinguishability (run 20260921; η = 0.7 rows re-measured on the 256-shot protocol, 256-shot batches with ≥ 2048 counted shots at both bracket points — V = 1 runs 20261008_104426 / _122031, V < 1 run 20261009_195643, 10-10; docs/DISTINGUISHABILITY.md §5)

| HOM visibility V (x = √V) | 1 | 0.95 | 0.9 | 0.8 |
|---|---|---|---|---|
| Fock, n* emitted at η = 1 | 38.6 | 39.0 | 40.0 | 41.8 |
| Fock, n* emitted at η = 0.7 (256-shot protocol) | 49.2 | 50.3 | 51.6 | 53.7 |
| emitter, n* emitted at η = 1 | 30.6 | 31.3 | 31.8 | 33.0 |
| emitter, n* emitted at η = 0.7 (256-shot protocol, parity build) | 40.9 | 41.7 | 42.6 | 44.7 |
| emitter, detected at n*, η = 0.7 | 28.6 | 29.2 | 29.8 | 31.3 |

Same law with η → ηx: a photon costs the classical sampler only when it is detected AND in
the common internal mode. At V = 0.9 distinguishability is worth ~1.5 emitted photons at η = 1 and
~2 at η = 0.7 (Fock +2.4, emitter +1.7 on the 256-shot grid), loss at η = 0.7 ~10; the emitter lead over Fock
is about constant in V — 7–8 detected photons at η = 1, 5.8 / 6.0 / 6.3 / 6.3 at η = 0.7 for V = 1 / 0.95 / 0.9 / 0.8.

## What it means for an experiment

1. **Loss lengthens the required cluster.** The classical cost is set by the photons that
   *arrive*, so every lost photon has to be emitted again: 31 emitted photons at η = 1 become
   40 at η = 0.7 and 54 at η = 0.5. What loss buys is only a shorter *detected* string
   (31 → 27), because the lossy classical sampler is a little cheaper per emitted photon than
   its lossless counterpart.
2. **The emitter modality is ~7–8 detected photons ahead of Fock at every η** (5 at η = 0.5):
   for the same classical cost it needs a 0.8× shorter detected string, because the entangled
   input carries the register (a 2-dimensional spin memory) as an extra exponent. The lead is
   the same one as at η = 1; loss does not erode it much (ρ_emitter > ρ_Fock, so the emitter's
   lossy factor falls faster).
3. **Cheapest target on each curve** is the lowest transmission the source, chip and detectors
   allow, since detected photons decrease with η. Concretely, for one quantum dot:
   - η = 0.9: 33-photon cluster, 66 modes, 30 detected;
   - η = 0.7: 40-photon cluster, 80 modes, 28 detected;
   - η = 0.5: 54-photon cluster, 108 modes, 27 detected.
   With k demultiplexed emitters the same detected-photon count is reached with clusters k×
   shorter (the register grows to 2^k, which costs the classical sampler nothing extra until
   k ≈ 20), so 4 emitters at η = 0.7 need 10-photon clusters each into 80 modes — the cluster
   length of today's sources — and the register then holds 4 spins.
4. **Today's gap.** Single-emitter clusters of 10–14 photons are 20–40 emitted photons short of
   the frontier at any η. Per-photon transmission is the lever with the largest effect on the
   *emitted* requirement (×1.75 between η = 1 and 0.5) and the smallest on the *detected* one
   (0.87×); multiplexing emitters is the lever on cluster length.
5. **Certification.** Beyond n ≈ 11 the strong engine cannot check an experiment; the
   Gurvits-type estimator of docs/ESTIMATE.md checks single-pattern probabilities and marginals
   at any n at ~1/ε² samples (E5 of the study plan: sample budget for a decisive test against
   dephased-spin and distinguishable-photon mock-ups — to be done).

Caveats: one GPU, one algorithm (this repository's samplers); ×ρ of hardware buys one photon
(+3.5 photons for 8 GPUs, +7 for 72); the emitter sampler's per-photon factor still has a
measured ~1.16× headroom from the term-count growth R_k (docs/EMITTER_SAMPLER.md §7), i.e. its
frontier could move out by 1–2 photons with further engine work. Loss is uniform and on the
input side; partial distinguishability is the x-model (one HOM visibility for all pairs);
multi-photon emission is not included.

## Where current devices sit: the whole lossy stream against G H100 (2026-10-01)

The question is not where a device sits on the 1 Hz frontier — every current machine sits far inside
it — but whether its *output stream* can be reproduced by a classical sampler at the rate the device
produces it. Two readings, one rate model, `bench/plot_postselect.py` v2 (supersedes the pre-v2 tables,
which rested on n\* = 37.8 / ρ_eff 1.84 and are not rescaled here but
recomputed on frontier v2, `results/keep/frontier.csv`, n\* = 38.3 at m = 2n since the 256-shot re-read of 10-09, run
20261009_141132 — recomputed on it 10-09: every device number below is unchanged, since each rests on the measured cost
table and the frontier enters only as the continuation beyond it):

- **Train rate.** With S sources at clock f, each emitting one photon per tick, S in parallel, a train
  of n photons takes ⌈n/S⌉ ticks: the device emits f / ⌈n/S⌉ trains per second (16 ticks and
  6.2·10⁷ trains/s for 48 photons from 3 sources at 1 GHz; 12 ticks and 6.7·10⁶/s for 12 photons from
  one source at 80 MHz; 12 ticks and 6.7·10⁶/s for 24 photons from two). The clock is always divided
  by the ticks the train needs, never counted per photon.
- **(A) post-selected on exactly k detected photons.** A device that sends n photons per train and
  keeps the trains with exactly k detected produces, under uniform loss, k photons at uniformly random
  input modes: one k-photon sample at its m modes, cost t(k, m). It delivers (f / ⌈n/S⌉) C(n, k) η^k (1 − η)^{n−k}
  such samples per second; the transmission at which that ties G H100 is plotted against k. Below a
  device-dependent k one H100 outruns the source at any transmission (its train rate S f / n is below
  the classical rate); above the frontier crossing (t(k, m) = 1 s) the device is past the frontier.
- **(B) the whole lossy stream.** f / ⌈n/S⌉ trains per second, each with k ~ Bin(n, η) detected photons.
  A competent simulator does not pay t(n, m) per train: loss commutes with the interferometer, so it
  draws the surviving subset and runs the k-photon sampler; its cost per train is the binomial average
  of t(k, m) — dominated by the rare high-k trains — under the run-of-S statistic: a run of
  S_run = (f / ⌈n/S⌉) × run length trains meets the trains beyond the 1 − 1/S_run quantile of k less than
  once, so they are left out. Run length 3600 s here. The device "ties G H100" when G H100 can no
  longer reproduce its exact stream, tail included, in real time.

Rate model t(k, m): the measured cost table `bench/cost_table.py` (m = 12, 24, 32, 48, 96) at the device's own m; beyond a column's last measured k, continued with the
frontier's ρ_eff at each ratio; without a column, the frontier extrapolation ρ_eff^(n\* − k) capped at
5·10⁷ samples/s. Every printed number is tagged with its source.

**Devices, by their characteristics only** (no device or vendor name in any paper-facing output):
A — one source at 80 MHz, 12 photons in 24 modes; B — two sources at 80 MHz, 24 photons in 48 modes;
C — three entangled sources at 1 GHz, 48 photons in 96 modes; D — one source at 80 MHz, 6 photons in
12 modes (a published system, ~8 % end-to-end efficiency reported). For C the Fock cost table is the
reading conservative for the device: entangled emission is priced by the emitter engine, which needs
~8 fewer detected photons at equal classical cost (table above), so every transmission below is a
lower bound of what three entangled sources require. Two loop-based time-bin machines are placed on
their published or target rates instead of in the transmission analysis: one with 16 detected photons
in 32 bins at 208 post-selected samples/s (one H100 produces the same
10⁶-sample dataset ~4 000× faster, and a loop circuit is never harder than the Haar reference used
here), and an announced three-loop machine (loops of 1, 5 and 25 bins, 128 bins, 100 samples returned
in 10 ms, i.e. ~10⁴ samples/s target; no public photon number, source, loss or detector figure, so no
transmission analysis): against the Haar reference at m = 128 bins one H100 sustains 10⁴ exact samples/s
up to **n ≈ 21 detected photons** (frontier v2, c = 6.1, n\* = 35.3, ρ_eff 1.96), 10³/s up to 25,
10²/s up to 28 and 1/s up to 36, so at its target throughput that machine would have to deliver more
than ~21 photons per sample in its 128 bins before its rate exceeds one H100's — and a 1-5-25 loop
network (transfer-matrix reach 31 bins) is a banded interferometer, easier than Haar, which moves that
bar up, not down. GBS devices are placed by the replay method of docs/GAUSSIAN.md §2.3–2.4,
not here.

**Measured numbers** (cost table run 20261011_205906, `results/keep/cost_table_20261011_h100.csv`, of record
since 10-11:
- m = 12 / 24 / 32 / 48 / 96, k = 2 up to 12 / 24 / 32 / 48 / 45, one H100 PCIe, the production sampler.
- t(16, 96) = 3.2 µs, t(24, 96) = 0.42 ms, t(38, 96) = 1.34 s.
- The sustained rates are made monotone in k. Dips at a few k are Triton recompiles of the draw kernel at the lane-count
  steps of the 65 536-shot ramp, not kernel behaviour; `cost_table.py` also records the plateau rate.
- The previous table (run 20261001_005657) had such dips at k ≤ 17 (m = 12 k = 6, 8–10; m = 24 k = 8, 9, 17; m = 48
  k = 8, 17; m = 96 k = 5, 8, 9, 16, 17). The envelope only half-corrects a dip, so they raised the device numbers: C at
  20 % was 38 H100, now 29.
- One dip is left, m = 96 k = 9: ≤ 3 % on any device number.
- Device numbers: `results/keep/postselect_v2_20261011_h100.csv`.)

| device | trains/s | (B) η to tie 1 / 8 / 64 H100 (⟨k⟩) | (A) post-selected optimum, 1 / 8 / 64 H100 | 1 Hz frontier |
|---|---|---|---|---|
| A (1, 80 MHz, 12 / 24) | 6.7·10⁶ | 73 % (8.7) / never / never | k = 12 at 92 % / — / — | every k inside |
| B (2, 80 MHz, 24 / 48) | 6.7·10⁶ | **32 % (7.7) / 54 % (12.9) / 72 % (17.3)** | k = 15 at 52 % / 19 at 68 % / 22 at 84 %; k < 14 unreachable | every k inside |
| C (3 entangled, 1 GHz, 48 / 96) | 6.2·10⁷ | 1.5 % (0.7, see below) / **13.5 % (6.5) / 23 % (11.0)** | k = 5 at 10 % (see below) / 17 at 23 % / 21 at 30 % | k = 38 detected (t = 1.34 s measured; ≈ 39 % transmission) |
| D (1, 80 MHz, 6 / 12) | 1.3·10⁷ | 75 % (4.5) / never / never | k = 6 at 96 % / — / — | every k inside |

H100-equivalents needed to reproduce the whole stream in real time at today's transmissions: device C
4.4 (η = 8 %), 5.3 (10 %), 10 (15 %), **29 (20 %)**, 580 (30 %); device B 0.26 (8 %) … 0.52 (20 %), 0.86
(30 %), 5.3 (50 %); devices A and D below one H100 up to η ≈ 70 %.

**The small-k end is a throughput statement, not a hardness one.** Device C's "1 H100 at 1.5 %" and
"k = 5 at 10 %" mean only that 6·10⁷ trains per second of 0–3-photon samples exceed what this sampler
emits in batch (1.9·10⁷ samples/s at k = 2, launch- and draw-bound); a dedicated trivial-sample code
would move that figure, and the paper should quote device C from 8 H100 upwards. The numbers that
carry the message are the 8 / 64 H100 ties: **at ~20 % end-to-end transmission device C's whole
stream (⟨k⟩ ≈ 10, k_S ≈ 31 in an hour) needs ~30 H100 to be reproduced exactly in real time, while
its typical train sits 25 photons inside the 1 Hz frontier**; device B reaches one H100 at ~32 % and
8 at ~54 %. The post-selected optima sit at k = 15–17 detected (k = 17 at 23 % for C with 8 H100, k = 15 at
52 % for B with 1).

**The caveat that goes with the number, verbatim:** the device is competitive in the sense that a GPU
cannot reproduce its exact output distribution, tail included, at its emission rate — not in the
sense that its typical sample is hard. Devices A and D never tie 8 H100 (one H100 outruns
their sources at every transmission); the 1 Hz frontier itself is crossed by device C only from
k ≈ 38 detected photons on, i.e. ≳ 39 % transmission post-selected. Both sides are exponential:
below η = 1/ρ(c) (0.54 at m = 2n) the hardware loses a factor per added photon and never crosses the
classical curve, whatever n. Against the pre-v2 tables (C 11 / 18 / 23 %,
B 40 / 55 / 72 %) the measured table moves the 8 / 64 H100 ties by −4.5 / 0 points for C and −1 / 0 for B — the frontier-v2 kernel is faster, the small-k rates are lower than the old 5·10⁷
cap, and the run-of-S truncation removes the tail the device never shows in an hour.

Figure: `img/postselect.png` (two panels: (A) η(k) per device and GPU count; (B) H100-equivalents
needed to reproduce the stream in real time against η, with the 1 / 8 / 64 lines).

## The frontier at other rates: 10⁻³, 1 and 10³ samples/s (2026-10-02)

The 1 Hz frontier is one line of a family: with a per-sample cost t(n) = t₀ ρⁿ the crossing of
R samples/s is n*(R) = n*(1 Hz) − ln R / ln ρ_eff, so the maps at other rates are translations of the
measured curve by ±6.9 / ln ρ_eff photons per factor 10³ in rate. `bench/plot_rate_maps.py` applies
this to `results/keep/frontier.csv` row by row (each row's own ρ_eff) → `img/frontier_rate_maps.png`,
`results/keep/frontier_rate_maps_20261009.csv` (rebuilt 10-09: Fock m/n = 1 / 2 from the 256-only re-read, run
20261009_141132, 42.6 / 38.3 — was 42.5 / 38.5 from run 20260930_082501; emitter m/n = 2 from run 20261006_004008, 31.0,
parity build, confirmed 31.0 by run 20261009_141132; m/n = 12 from run 20261003_103957, 29.0). One H100, sources n* at m/n = 1 / 2 / 4 / 12:

| engine | 10⁻³ samples/s (extrapolated) | 1 sample/s (measured) | 10³ samples/s (translated) | shift per 10³ |
|---|---|---|---|---|
| Fock input | 54.5 / 50.0 / 46.8 / 44.4 | 42.6 / 38.3 / 36.2 / 34.4 | 30.7 / 26.6 / 25.6 / 24.4 | 11.9 / 11.7 / 10.6 / 10.0 |
| emitter chain (m/n ≥ 2) | — / 41.5 / 40.0 / 39.0 | — / 31.0 / 29.8 / 29.0 | — / 20.5 / 19.6 / 19.0 | — / 10.5 / 10.2 / 10.0 |
| Gaussian BS (t_S, S = 10⁴) | 64.8 / 63.2 / 61.2 / 58.0 | 42.9 / 41.5 / 39.9 / 37.4 | 21.0 / 19.8 / 18.6 / 16.8 | 21.9 / 21.7 / 21.3 / 20.6 |

Superposed sources (α = π/4) are quoted **at 1 sample/s only**: 53.4 / 49.7 / 48.9 / 48.0 at m/n = 1 / 2 / 4 / 12 (run
20261006_132352, the parity build). Their cost per sample is heavy-tailed, so neither a ramp-fitted ρ_eff nor the
run-cost model translates them to other rates: the model (`bench/sbs_run_cost.cost`) misses the measured 1 Hz
frontier by up to 1.5 sources at m/n ≥ 3 (acceptance test: ≤ 0.5 at every ratio — failed), and at 10³ samples/s the sampler is
launch-bound (≥ 1 ms per chain step) at any n.
The 10³ Hz column is checked against direct measurements at that rate — the production envelope
points (`algorithm_envelope_c{1,2}_20260930_h100.csv`, 1 ms per sample bracketed between
consecutive n) and the cost-table columns (`cost_table_20261011_h100.csv`; identical crossings with the 10-01 table, bracketed in k at fixed
m): Fock 29.2 measured vs 30.1 translated at m/n = 1 (29.3 at 1.09 from m = 32), 27.4 at 1.75 (m = 48),
26.7 vs 26.8 at m/n = 2, 25.4 at 3.77 (m = 96) vs 25.6 at 4. The translation holds to a photon;
the one-photon shortfall at m = n is the launch-bound floor of a 1 ms sample (the measured curve
is flatter than ρⁿ below ~30 photons), so the translated 10³ Hz curve is an upper bound there. The
10⁻³ Hz column (1000 s per sample) has no measurement behind it and is an extrapolation on the
fitted ρ_eff; two caveats travel with it: on the Fock chain it reaches the precision regime of
ALGORITHMS.md §6 (fp64 cofactor error ~10⁻⁶ at n ≈ 43, ~10⁻³ near 50 — the m/n ≤ 2 points), and the
GBS points sit past the fp64 tier of the loop hafnian (N ≈ 55; double-double
beyond, slower by its own factor), so the 58-65 are a count of what the kernel would reach at
fp64 cost, not a time it can deliver at that precision. The GBS column moves twice as far as the
others per decade of rate because its ρ_eff is √2-like (1.37-1.40): rate buys a Gaussian experiment
more photons than a Fock one, in both directions.
