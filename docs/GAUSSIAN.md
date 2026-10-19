# Gaussian boson sampling: squeezed inputs, loop hafnians, the exact chain-rule sampler

Design note for `states.GaussianInput`, `photonic_engines/gaussian/` (`hafnian.py`, `sampler.py`,
`probs.py`), `slos/gaussian.py` (strong simulation) and the Gaussian loss map in `imperfections.py`. The literature survey, the device table
and the go/no-go are in `docs/GAUSSIAN_SURVEY.md`; this note holds the derivation, the algorithm as
built, and its versions. Tests: `tests/test_gaussian.py`; GPU validation: `tests/check_gaussian_gpu.py`.
Status (2026-09-23 night): **v1 built, validated on CPU and on an H100 (run 20260923_224054); no
frontier number measured yet.**

## 0. Summary

- **Input.** n single-mode squeezed vacua (optionally displaced), one per input mode, through a
  passive U. n counts **squeezers** — the frontier axis, as for the other modalities. Two-mode
  squeezing is two squeezers and a beam splitter, i.e. part of U.
- **Sampler (v1).** Exact, photon-number resolving, no photon cutoff, no post-selection: the chain
  rule over conditional pure Gaussian states (virtual heterodyne of the not-yet-counted modes), each
  step one loop hafnian **polynomial in the current mode's Bargmann variable**, so one evaluation
  prices every photon number of that mode; the normaliser is a closed-form Gaussian integral, so the
  draw is an exact inverse CDF. Algorithm of Quesada et al. 2010.15595 / Bulmer et al. 2108.01622,
  reimplemented from our own derivation (§3), including our own loop-hafnian formula (§3.1).
- **Loss** is exact and cheap: a lossy squeezed vacuum is a pure squeezed vacuum at smaller r plus a
  classical random displacement (§4.1). **Distinguishability** is spectral (Schmidt modes): a shot is
  the sum of K independent GBS shots (§4.2). Dark counts: out.
- **Oracles**: brute-force loop hafnian; `GaussianInput.truncated(d)` through `slos.ProductSLOSKernel`
  (exact for lossless patterns with ≤ d−1 photons; lossy up to a bounded tail); The Walrus and
  Piquasso when installed. All agree (§5).
- **Cost** per shot O(m N³ 2^{N/2}), N = detected photons, v1 has no collision discount and no GPU
  kernel of its own (torch batched over the 2^{N/2} subsets). Collision-free expected cost per
  squeezer sech r (1 − (1 − η + η√2)² tanh² r)^{−1/2}, divergent past tanh r (1 − η + η√2) = 1 (§2).

## 1. Model and conventions

`GaussianInput(r, modes, m, disp=None)`: source i is D(disp_i) S(r_i)|0⟩ on input mode modes[i],
S(r) = exp((r a² − r a†²)/2), so S(r)|0⟩ = sech(r)^{1/2} exp(−tanh r a†²/2)|0⟩, squeezing
x = (a + a†)/2 (Var x = e^{−2r}/4, vacuum 1/4), ⟨n⟩ = sinh² r, even photon numbers only:
P(2k) = C(2k, k) 4^{−k} tanh^{2k} r / cosh r. After U (a† → U a†) the pure output is

    |ψ⟩ = N exp(½ a†ᵀ B a† + ζᵀ a†)|0⟩,   B = −U_c diag(tanh r) U_cᵀ,   ζ = β − B β̄,   β = U_c disp,
    log N = −½ Σ log cosh r − ½|β|² + ½ β̄ᵀ B β̄,

U_c the source columns (`GaussianInput.bargmann`). Every amplitude is a loop hafnian:
⟨n|ψ⟩ = N lhaf(B[n], ζ[n]) / √(Π n_i!), B[n] = rows/columns repeated n_i times with the diagonal
replaced by ζ[n] (`hafnian.expand`). Convention cross-checked against Piquasso's and The Walrus's
squeezing to 1e-13 (tests).

## 2. Axis and the cost model

Per shot the chain rule costs ≈ c(N) ρ^N with ρ = √2 (2^{N/2}), N detected. With independent sources
and uniform loss (binomial thinning z → 1 − η + ηz of each source's photon-number generating function
Φ), the expected collision-free cost factorises per source:

    E[ρ^N] = Π_sources Φ(1 − η + ηρ),   Φ_SMSV(z) = sech r (1 − z² tanh² r)^{−1/2}

— the Fock law (Φ = z) and the SBS law (Φ = 1 − p + pz) are the same statement. It diverges for
tanh r (1 − η + η√2) ≥ 1 (lossless sinh² r ≥ 1): the squeezer's photon tail is geometric in pairs
(ratio tanh² r) while the cost doubles per pair. Beyond that point the true mean cost is finite only
through collisions (term count Π(η_i+1) ≥ Π√(n_i+1)) and is carried by rare collision-heavy shots,
so it depends on m/n. The bench (`find_threshold --engine gbs`) prints the model value and flags the
divergent regime. Headline point: 0.5 photons per squeezer (r = 0.658), m = 2n (survey §7).

### 2.1 The frontier statistic: the cost per sample of a run of S samples (decided 2026-09-24; headline S = 10⁴, see the update below)

The exact cost of a shot is heavy-tailed. With a cost ~ ρ^N (ρ = √2 here) the cost-weighted photon law
P(N) ρ^N is exactly the law of squeezers with tanh r' = ρ tanh r — 2 photons per squeezer instead of 0.5
at the headline point — so the untruncated mean E[cost] is carried by shots about four times brighter
than a typical one, which a device produces once in far more samples than any run contains. In the
collision-free model the mean cost over the median-shot cost, with shots rarer than once per run left out:

| n | median N | run of 10² | 10⁴ | 10⁶ | 10⁸ | untruncated |
|---|---|---|---|---|---|---|
| 20 | 10 | 6.6 | 17 | 25 | 29 | 32 |
| 30 | 14 | 20 | 77 | 153 | 204 | 256 |
| 38 | 18 | 34 | 194 | 447 | 670 | 1024 |

A plain mean over 16-256 shots is the first column with a random cut, which is why two runs gave
n* = 38.0 and 31.4 (runs of 2026-09-23 and 2026-09-24; the first is run 20260923_225057). **Decision: the GBS frontier is the mean cost
per sample of a run of S = 10⁶ samples** — what a QPU benchmark of that length would show, and the scale of
the experiments (Borealis ~10⁶, Jiuzhang 1.0 5·10⁷) — with S = 10⁴ and 10⁸ reported as sensitivity:

    t_S = sum_{N <= N_S} P(N) E[cost | N],   N_S = the smallest N with P(N > N_S) <= 1/S,

P(N) exact (`cost.detected_law`: convolution of thinned squeezer laws, U-independent), E[cost | N] the
stratum mean where a stratum holds >= 5 shots, else the fit log t = a + bN over the upper strata
(`cost.stratified_cost(max_N=N_S)`, `cost.run_cutoff`). Every point prints the share of t_S that rests on
the fit; a point with a large share is a calibrated cost-model number and is labelled so.
`find_threshold --engine gbs --estimator run --run-samples 1e6` (the default). The other estimators stay
for diagnostics: `is` — the untruncated mean by defensive importance sampling at a tilted squeezing
(exact weights: scaling every tanh r by s leaves P(pattern | N) unchanged, verified to 1e-13; but the
tilted shots are the ~4× brighter ones, hours each beyond n ≈ 30), `stratified` (untruncated) and `raw`;
`bench/gbs_cost_check.py` checks that independent tilts agree.

**Update (2026-09-24, after run 20260924_084540).** Two changes, neither touching a measured number:
- *Headline S = 10⁴, S = 10⁶ as the benchmark-style figure.* S only moves the cut N_S, and at S = 10⁶ it
  sits where no stratum can be measured (N_S = 72 at n = 40; the median is 20), so the extrapolated share
  of t_S is large by construction (57 % at n = 40 in that run). S = 10⁴ cuts at N_S = 58 there.
  `find_threshold` prints and writes t_S and its extrapolated share for S = 10⁴, 10⁶ and 10⁸ from the same
  shots (CSV columns s_1e4, ext_1e4, ...); `--run-samples` only picks the one that drives the search.
- *A photon-count cap per shot.* One bright shot can cost more than the whole per-n budget (n = 41: 2 shots
  in 1942 s; the v1 search timed out at 2 h on one). The chain rule knows its running count, so a shot is
  aborted once the count exceeds N_cap and recorded as censored. **Exact for t_S**: the count only grows
  along the chain, so a shot whose final count is ≤ N_cap is never aborted and is drawn exactly as without
  the cap (tested: same seed, same row; censored iff the uncapped count exceeds the cap); the strata above
  N_cap were fitted anyway. `find_threshold` sets N_cap per run as the largest N whose fitted cost is
  ≤ budget / 10, clipped between the 1 − 1/64 and the 1 − 1/(8 · max-shots) quantiles of the exact law
  (`--shot-cap`, `--cap-frac`). The occupancy sanity check then runs on the uncensored shots (conditioned
  on N ≤ N_cap, a bias of the order of the censored fraction).

### 2.2 The frontier from a calibrated timing model (2026-09-24)

Decided 2026-09-24: optimise sampling first, then estimate the frontier without sampling the strata that no run
can reach. The statistic stays t_S of §2.1; what changes is where E[cost | N] comes from.

- **The cost of a shot is a function of its pattern.** Which sieves `sample_pure` calls (N vertices, V loop vectors,
  numpy / torch / v2, device) depends only on the drawn counts in chain order and the settings (window, cpu_below,
  cap), not on the matrix entries: `timing.plan_calls(pattern)` replays the control flow, and a test checks it against
  the calls the sampler records (`sampler.Profile.events`) shot by shot, with windows, thresholds and the cap. The
  one data-dependent choice is the pairing search from 18 pairs (§3.4 v2.3), which can drop the collision
  discount; the model prices those calls at 2^n terms and the calibration logs how often that held
  (`hafnian.LOG`).
- **Per-call times, calibrated on the target GPU** (`bench/gbs_timing_model.py calibrate`, job 19): one
  collision-free v2 call per N = 14..48 and V = 1..4; the numpy small sieves; host draw, preparation and the rest
  of a shot from the recorded events. Above the largest timed sieve the call time is scaled by the exact term count
  and a measured per-term rate, t(N) = t(N0) 2^(n − n0) rate(n) / rate(n0), with rate(n) timed on random subsets of
  the sieve's own buckets and batch sizes up to n = 32 pairs (the kernel's Q = 64, N = 64 vertices). No kernel
  exists above 64 vertices; calls there are priced by the same extrapolation and their share of t_S is reported.
- **E[cost | N] = mean over patterns of total N of the replayed shot time.** The mode order is uniform for a Haar
  U (exact: Haar is invariant under output permutations). The collision statistics are the approximation:
  surrogates "free" (N distinct modes) and "be" (uniform multisets, Bose-Einstein), compared with the measured
  patterns (occupied modes given N) and with the model on the measured patterns.
- **Validation** at n ≤ 48, where shots can be drawn: every call and every uncensored shot, predicted against
  measured, and E[cost | N] per stratum. First CPU check (two cores, n = 8-12, 120 shots): per-call ratio 0.97-0.99,
  shot sums 0.94-1.03 of the measured time.

**First H100 result (run 20260924_234127, job 19; sampler v2.8, window 4, m = 2n, 0.5 photons per squeezer,
lossless).** Validation:
- rate × 2^n against whole measured calls: 0.95-1.00 at n = 20-24 pairs (0.82 at n = 18, where the pairing search
  starts and the rate batches do not include it).
- 864 shots (n = 16-48, one censored): per shot, model / measured median 0.96-1.02; summed per n 0.85-0.90 below
  n = 40 and 0.95-0.97 at n = 40-48 (a minority of in-shot calls under N = 24 take up to ~3× their stand-alone
  time). The pairing search ran on 80 calls and kept the full 2^n terms on all 80, as the model assumes.
- Collisions: the occupied modes given N of the "be" surrogate match the shots within ~1 mode at every stratum
  (e.g. n = 48, N = 36: 26.4 vs 27.2; "free" would be 36), and E[cost | N] from "be" is within the scatter of the
  measured strata.

Frontier (surrogate "be"; "free" in brackets):

| S | n* | N_S at n* | share of t_S from calls N > 48 (priced) | from N > 64 (no kernel) |
|---|---|---|---|---|
| 10⁴ | 41.5 (41.0) | 58 | ~76 % | 0 |
| 10⁶ | 37.7 (37.0) | 68-72 | ~85-90 % | ~25-45 % |
| 10⁸ | 36.1 (35.6) | 80 | ~91 % | ~57 % |

t_S grows ×1.2-1.6 per squeezer around n*, so the ~4 % under-prediction of the shot sums moves n* by ~0.1. At the
headline S = 10⁴ nothing rests on sizes without a kernel; the priced sieves (N = 49-58) use per-term rates measured up
to n = 32 pairs, not whole timed calls, which job 20 checks at N = 50-56. S = 10⁶ and 10⁸ rest partly on N > 64,
where the sampler has no kernel (Q = 64): those two numbers price an implementation that does not exist yet.
The previous direct-sampling searches (v1 sampler, run means) gave 38.0 and 31.4 (runs of 2026-09-23/24).

**Check of the priced sieves (job 20, run 20260925_075752).** Whole calls at N = 50 / 52 / 54 / 56, V = 1: 28.4 /
54.4 / 218 / 517 s (V = 4: 44.6 / 92.2 / 305 / 713 s); the model had priced them at ×0.89 / 0.99 / 0.76 / 0.99 (V = 4:
0.94 / 0.99 / 0.83 / 0.99). Right at even pair counts n = 26, 28, where the rate was measured, low at the odd ones
n = 25, 27, where it was interpolated: the per-term rate steps where `_bucket_sizes` changes the padded subset
sizes (×2.4 from n = 26 to 28), so an interpolation between even n misses a step. Two changes: `t_multi` now takes
any measured whole call as it is (an odd N the measured N + 1, same n) and prices only above the largest one, and
the rate is to be measured at every n (job 21). With the job-20 calls merged: n* = 41.4 / 37.6 / 36.0 for S = 10⁴ /
10⁶ / 10⁸ (was 41.5 / 37.7 / 36.1); at S = 10⁴ only N = 57-58 are still priced (~25 % of t_S at n*, from the rate at
n = 29 interpolated between 28 and 30, which lie on the same step), and nothing rests on N > 64.

**Rate at every n (job 21, run 20260925_090352)** — the current numbers. The rate is not monotone in n (V = 1, µs:
0.81 at n = 26, 1.62 at 27, 1.92 at 28, 1.80 at 29, 2.11 at 30, 1.98 at 31, 2.28 at 32: the buckets
ceil(n × fraction) move the padded sizes across Q = 32 / 64 differently for each n). Priced from N0 = 48 with it, the
four job-20 calls come out at ×0.98-0.99 of measured (they were ×0.76-0.99 with even n only). With the job-20 calls
and these rates merged:

| S | n* ("be") | n* ("free") | N_S at n* | share of t_S priced (N > 56) | share on N > 64 (no kernel) |
|---|---|---|---|---|---|
| 10⁴ | **41.4** | 40.8 | 58 | ~24 % | 0 |
| 10⁶ | 37.5 | 36.9 | 70 | ~70 % | ~35 % |
| 10⁸ | 35.6 | 35.3 | 78-80 | ~80 % | ~60 % |

(n* on a unit grid for S = 10⁴; the step-2 grid gives 41.2, the interpolation crossing a step of N_S.) The headline
S = 10⁴ point rests on measured calls up to N = 56 and on measured per-term rates for N = 57-58. Above N = 64 the rate is
extrapolated from n = 31-32 and the sampler cannot run at all (no Q = 128 kernel), so S = 10⁶ and 10⁸ are
conditional on an implementation that does not exist.

### 2.3 The full GBS frontier (2026-09-25, timing model, sampler v2.8, one H100 PCIe)

**Validation off the headline point (job 22, run 20260925_095009).** Capped shots with every call timed, predicted by
replaying their patterns (model built from jobs 19-21 only, these shots unseen):

| setting | n | shots | model / measured, summed | per-shot median |
|---|---|---|---|---|
| m = 2n (headline, repeated) | 44 | 96 | 0.97 | 0.97 |
| m = n | 40 | 96 | 0.91 | 0.96 |
| m = 4n | 36 | 96 | 0.95 | 0.98 |
| m = 8n | 28 | 96 | 1.00 | 1.02 |
| m = 12n | 24 | 96 | 0.97 | 0.99 |
| η = 0.7 | 48 | 95 | 0.95 | 0.97 |
| η = 0.5 | 56 | 96 | 0.97 | 1.00 |
| 1.5 photons per squeezer | 24 | 67 | 0.97 | 0.93 |
| 0.1 photons per squeezer | 80 | 96 | 1.00 | 1.02 |

The m = n row first came out at 0.70 and exposed a pricing error: a collided call below the pairing search had been
priced as the free call × terms / 2^n, but `lhaf_poly_multi` runs one free sieve of the ungrouped pairs per count vector,
each with its own launches, so the fixed cost is paid prod(r + 1) times (measured ~4× the old price on those calls).
Now priced as prod(r + 1) × t_free(2 n_f) (`TimingModel.t_call`, tested); that also moved the calibration shots from
0.86-0.98 to 0.90-0.98 summed. It changes no headline number (those t_S rest on N ≥ 36, where the search keeps 2^n).

**The grid** (`bench/gbs_timing_model.py frontier`, PAPER_GRID; surrogate "be", 64 patterns per N; the job 20 whole
sieves and job 21 rates merged). n* at 1 s per sample, S = 10⁴ headline; in brackets the share of t_S at n* that is
priced rather than measured (calls N > 56), and the share on N > 64 (no kernel):

| point | S = 10⁴ | S = 10⁶ | S = 10⁸ |
|---|---|---|---|
| m/n = 1 | 42.9 (45 %, 0) | 39.1 (80 %, 49 %) | 36.9 |
| 1.5 | 41.8 (43 %, 0) | 38.1 | 36.2 |
| 2 | 41.5 (40 %, 0) | 37.6 (73 %, 39 %) | 35.7 |
| 2.5 | 41.1 | 37.3 | 35.3 |
| 3 | 40.6 (24 %, 0) | 36.9 | 35.0 |
| 4 | 39.9 (23 %, 0) | 36.2 | 34.4 |
| 6 | 39.1 (21 %, 0) | 35.4 | 33.6 |
| 8 | 38.2 (0, 0) | 34.8 | 33.0 |
| 12 | 37.4 (0, 0) | 33.7 (52 %, 12 %) | 32.1 |
| η = 0.9, m = 2n | 47.8 (25 %, 0) | 43.6 | 41.8 |
| η = 0.8 | 55.5 (22 %, 0) | 51.3 | 49.5 |
| η = 0.7 | 65.2 (21 %, 0) | 61.1 | 59.4 |
| η = 0.5 | 96.7 (0, 0) | 92.4 | 91.1 |
| 0.1 photons per squeezer | 223.2 (0, 0) | 215.7 | 212.8 |
| 0.25 | 89.4 (0, 0) | 83.9 | 82.0 |
| 0.75 | 25.5 (52 %, 0) | 21.9 (84 %, 58 %) | 19.9 |
| 0.9 | 20.2 (66 %, 0) | 16.9 | 14.7 |
| 1.2 | 13.3 (78 %, 0) | 10.5 (95 %, 83 %) | 8.4 |
| 1.5 | 9.6 (80 %, 0) | 6.9 | 5.0 |
| Schmidt 0.95 / 0.05 | 44.0 (35 %, 0) | 40.4 | 38.3 |
| 0.9 / 0.1 | 47.3 (42 %, 0) | 43.5 | 41.5 |
| 0.8 / 0.2 | 54.3 (35 %, 0) | 50.6 | 48.7 |

Readings, with what they rest on:
- **n*(m/n) is flat**: 42.9 → 37.4 from m = n to 12n, against Fock 42.4 → 34.4 and SBS 21.3 → 18.6 (results/keep/frontier.csv).
  At fixed n the photon law does not depend on m; m only adds steps (trailing empty modes cost size-N sieves) and
  changes collisions. GBS minus Fock, in sources: +0.5 at m = n, +3.0 at 2n, +3.3-3.8 from 3n to 12n.
- **At S = 10⁴ the frontier sits at N_S = 54-64 detected photons on every point**: the sampler cost is set by the
  photon number, so every knob moves n* through P(N) only. Loss and low squeezing push n* up steeply (η = 0.5: 97;
  0.1 photons per squeezer: 223); high squeezing pulls it down (1.5 photons: 9.6 squeezers, N_S = 64 in 20 modes).
- **Schmidt impurity helps the simulator** (44.0 / 47.3 / 54.3 at λ₁ = 0.95 / 0.9 / 0.8): the minor mode's photons form
  a separate, much cheaper chain. The η → ηx law of the Fock engines has no literal counterpart (docs/GAUSSIAN_SURVEY.md §6.2).
- rho_eff = t_S(n + 1) / t_S(n) at n* alternates between ~1.2 and ~1.6 with the parity of N_S (N_S moves in steps of 2):
  a feature of the truncated statistic, not noise.
- **What is not measured**: S = 10⁶ and 10⁸ rest in part on N > 64 (up to 85-99 % at high squeezing), where no kernel
  exists; the high-squeezing points at S = 10⁴ rest 66-80 % on priced calls N = 57-64 (measured per-term rates);
  collisions come from the "be" surrogate, validated up to 1.5 photons per squeezer at n = 24 (shots capped at 44).
  Every n* is "exact, PNR, one H100, timing model" — not a sampled search.

**The grid re-priced with the precision tiers (2026-09-29; timing model only, no GPU run).** Same calibration as above
plus the torch fp64 rates to n = 64 (job 24) and the dd rates (job 27); `frontier --dd-rate` (TimingModel.tiers: each
call in the cheapest (pairing, arithmetic) whose eps κ ≤ 10⁻³, §2.4). Three columns per S: **v2.11** = tiers off (all
fp64, grouping dropped above 18 pairs: the grid above, now with torch rates past n = 32 — S = 10⁴ unchanged); **cost** =
the cheapest admissible option, dd included when dd with Bulmer's cross grouping beats fp64 without it (what
precision="auto" does with `_DD_FACTOR` = 250); **needed** = dd only where fp64 cannot meet the tolerance (6 points
computed). n* at 1 s per sample:

| point | S = 10⁴: v2.11 / cost / needed | S = 10⁶ | S = 10⁸ |
|---|---|---|---|
| m/n = 1 | 42.9 / 47.2 / 43.0 | 38.4 / 46.7 / 41.1 | 36.3 / 46.6 / 41.0 |
| 1.5 | 41.8 / 43.8 / 41.9 | 37.6 / 42.2 / 39.8 | 35.5 / 42.0 / 39.7 |
| 2 | 41.5 / 41.9 / 41.5 | 37.2 / 39.7 / 38.8 | 34.9 / 39.4 / 38.5 |
| 2.5 | 41.1 / 41.2 / 41.1 | 37.0 / 38.1 / 37.8 | 34.5 / 37.6 / 37.3 |
| 3 | 40.7 / 40.8 / 40.7 | 36.5 / 37.2 / 37.1 | 34.0 / 36.4 / 36.3 |
| 4 | 39.9 / 39.9 / 39.9 | 35.8 / 35.6 / 35.6 | 33.6 / 34.5 / 34.5 |
| 6 | 39.2 / 39.2 | 35.2 / 33.8 | 32.8 / 31.9 |
| 8 | 38.3 / 38.3 | 34.6 / 32.9 | 32.2 / 30.5 |
| 12 | 37.4 / 37.4 | 33.6 / 31.8 | 31.4 / 28.7 |
| η = 0.9, m = 2n | 47.8 / 48.1 | 43.2 / 44.9 | 40.9 / 44.4 |
| η = 0.8 | 55.5 / 55.6 | 50.9 / 51.9 | 48.6 / 51.2 |
| η = 0.7 | 65.2 / 65.3 | 60.8 / 60.9 | 58.5 / 59.9 |
| η = 0.5 | 96.8 / 96.9 | 92.5 / 91.4 | 90.5 / 89.2 |
| 0.1 photons per squeezer | 223.7 / 224.1 | 215.9 / 216.1 | 212.2 / 201.9 |
| 0.25 | 89.5 / 89.7 | 83.6 / 82.0 | 81.0 / 80.5 |
| 0.75 | 25.5 / 27.4 | 21.5 / 26.7 | 19.4 / 26.5 |
| 0.9 | 20.2 / 23.0 | 16.5 / 22.5 | 14.4 / 22.5 |
| 1.2 | 13.3 / 17.8 | 10.3 / 17.5 | 8.3 / 17.4 |
| 1.5 | 9.7 / 14.6 | 6.7 / 14.4 | 5.0 / 14.4 (N_S = 126: qd priced as dd, a lower bound) |
| Schmidt 0.95 / 0.05 | 44.0 / 44.7 | 39.8 / 42.2 | 37.5 / 41.8 |
| 0.9 / 0.1 | 47.3 / 47.6 | 43.1 / 45.0 | 40.6 / 44.5 |
| 0.8 / 0.2 | 54.3 / 54.5 | 49.8 / 51.6 | 47.7 / 51.0 |

Readings:
- **The headline holds**: at S = 10⁴, m = 2n, 0.5 photons per squeezer, n* = 41.5 (v2.11 and needed) / 41.9 (cost). The
  S = 10⁴ frontier sits at N_S = 54-64 where fp64 suffices (N ≤ 62), so the precision tier touches it only through
  the choice to run grouped sieves in dd.
- **Where collisions are dense the dd-grouped sieve moves the frontier up**: m = n (+4.3 at 10⁴, +8.3 at 10⁶), high
  squeezing (1.5 photons: 9.7 → 14.6; 0.9 photons: 20.2 → 23.0), and n* becomes nearly independent of S (the grouped
  cost grows slowly with N). These gains rest on the modelled κ penalty of the grouping (+5 decades, measured on
  Borealis patterns at N = 36-48, not on Haar m = n or high-squeezing patterns) and on the dd rate held past n = 32.
  Job 29 measures whole shots with precision "auto" at m = n and at 0.9 photons to check them.
- **Where collisions are rare the precision costs**: m/n ≥ 6 at S = 10⁶ / 10⁸ (−0.9 to −2.7; m/n = 4: −0.2 / +0.9): the
  tail N > 62 now runs in dd at ×53-87 with little to group; likewise η = 0.5 (−1.1 / −1.3), 0.25 photons at 10⁶ (−1.6)
  and 0.1 photons at 10⁸ (−10.3).
- **needed** (dd only where fp64 fails, grouping in dd then allowed): S = 10⁴ unchanged, S = 10⁶ / 10⁸ +1.6-4.7 at
  m/n ≤ 2 (the tail's calls, priced ungrouped in fp64 by v2.11 — also past fp64's tolerance — are grouped in dd).
- Device points (same model, `results/keep/gbs_device_points_tiers_20260929.csv`), n* at S = 10⁴ / 10⁶, v2.11 / cost /
  needed: Borealis settings 39.6 / 43.6 / 39.6, 35.9 / 43.1 / 37.8; Jiuzhang 3.0-like r 1.2: 19.1 / 20.5 / 19.2, 15.9 /
  19.6 / 17.7; r 1.4: 9.6 / 12.8 / 9.7, 6.9 / 12.5 / 9.0; Jiuzhang 4.0-like r 0.9: 38.4 / 38.4 / 38.4, 34.9 / 33.3 / 33.3.
- CSV results/keep/gbs_timing_frontier_tiers_20260929.csv (every variant); the grid above stays the measured-model
  reference until job 29.
- **Job 29 (run 20260929_072630) did not test the cost column**: whole shots at m = n (n = 40, 44) and 0.9 photons (n = 22),
  cap 64, reached N ≤ 56, where precision="auto" kept fp64 on all 734 calls (same times as fp64). It showed two model
  errors instead (LOG): the host now runs fp64 sieves ×1.3-1.8 slower than the calibration (torch 2.12.1 against 2.14.0),
  which makes the tiers' dd / fp64 factors high by up to that much; and the sieve groups in fp64 up to N = 49, which the
  tiers' +5-decade κ penalty forbids from N ≈ 38 (grouped calls priced ×1.8-7.6 above measured). Both need job 30 (rates
  on one stack) before this table is used.
- **Job 30 (run 20260929_103413)** settled the first point: on torch 2.14.0 the fp64 rates reproduce the calibration
  (0.993-1.005), so job 29's slowdown was torch 2.12.1. With its dd rates (×58-67 of the torch fp64 term, to n = 48) the
  cost column moves up slightly on six points, S = 10⁴ / 10⁶ / 10⁸: m = n 47.8 / 47.4 / 47.3 (was 47.2 / 46.7 / 46.6), m = 2n
  42.1 / 40.2 / 39.9 (41.9 / 39.7 / 39.4), m = 12n 37.4 / 31.9 / 29.2 (37.4 / 31.8 / 28.7), η = 0.5 96.9 / 91.7 / 89.7, 0.75
  photons 27.8 / 27.1 / 27.0, 1.5 photons 15.0 / 14.8 / 14.8 (results/keep/gbs_timing_frontier_tiers_job30_20260929.csv).
  The second point (the grouping penalty) is open: the dd-by-choice gains at m = n and high squeezing still need whole
  shots that reach N > 56 in precision "auto".
- **Jobs 31-32 (runs 20260929_210408 / _231151) measured the dd-by-choice** at m = n = 30, 2 photons per squeezer: three of
  eight uncensored shots (N = 58 / 62 / 66, 459 / 563 / 402 s) ran their large calls in dd with the cross grouping
  (2^19-2^20 terms instead of 2^29-2^33), x8.7-8.8 faster than fp64 on the calls both runs made (to N = 52), where the
  model's fp64-only price of those shots is 1.2e3-4.3e4 s. They also showed the model priced grouped calls x0.09-0.59 low
  (each prefix length at a smaller sieve's per-term rate); floored at term count x rate(n) x dd factor it reaches
  0.80-0.95 of the dd shots. Re-priced with the floor (results/keep/gbs_timing_frontier_tiers_floor_20260930.csv), cost
  column, S = 10⁴ / 10⁶ / 10⁸: m = n 44.7 / 43.4 / 43.3, **m = 2n 41.5 / 38.0 / 37.4**, m = 12n 37.4 / 31.8 / 27.8, η = 0.5
  96.8 / 91.1 / 87.3, 0.75 photons 26.2 / 24.6 / 24.3, 1.5 photons 12.9 / 12.5 / 12.4 (N_S = 120 at 10⁸: qd priced as dd).
  Against v2.11 (all fp64): the headline is unchanged, the gains where collisions are dense halve (m = n +1.8 at 10⁴, +5.0
  at 10⁶; 1.5 photons +3.2 / +5.8), and the dd tail costs where they are rare (m = 12n −1.8 / −3.6 at 10⁶ / 10⁸). The one
  measured limit: on one N = 56 shot "auto" kept fp64 where the model takes dd (the sieve's `_DD_FACTOR` rule), so the sieve
  as built reaches these numbers only where its own rule agrees.

**The lossy frontier next to the other engines** (`bench/plot_frontier_loss.py` → `img/frontier_loss_all.png`, from the
kept CSVs, m = 2n). n*(η = 1 / 0.9 / 0.8 / 0.7 / 0.5): Fock 38.6 / 41.4 / 44.7 / 50.0 / 64.0, emitter 30.6 / 30.9 / 31.2 /
31.9 / 34.4, SBS 20.2 / 20.5 / 24.0 / 25.5 / 31.2, GBS 41.5 / 47.8 / 55.5 / 65.2 / 96.7. GBS gains fastest with loss
(×2.3 at η = 0.5 against Fock's ×1.7), and it is the only engine whose detected photon number at n* RISES with loss
(η · 0.5 n* = 20.7 → 24.2; Fock 38.6 → 32.0): thinning a squeezed vacuum turns the super-Poissonian photon law towards a
thinner tail, so at equal mean the run of S = 10⁴ meets fewer of the expensive bright shots. Against the collision-free,
untruncated law n*(1) log Φ(1) / log Φ(η) (docs/GAUSSIAN_SURVEY.md §6.1: 49.2 / 58.9 / 71.2 / 110.7) the timing-model
points are 3-13 % lower; the law prices √2 per photon, the sieves grow faster than that at the N the frontier sits on.

### 2.4 Pricing a device's own samples: the Borealis runs (2026-09-27, timing model, no new measurement)

An exact sample's cost is a function of its pattern (`timing.plan_calls`), so the cost of producing a device's run with
this sampler is the mean replayed cost over the run's own samples — S = the experiment's sample count, the device's own
pattern law, no P(N) model and no collision surrogate (`bench/gbs_device_replay.py`, stratified by N: every N with at
most 40 samples priced in full). Two term counts: **v2.8** as built (grouping dropped from 18 pairs), and **self_all**,
the collision-discounted prod(r + 1) 2^{n_f} at every size (Bulmer et al.'s count; a kernel that does not exist).
Precision tier from the κ fit (fp64 to N = 54, double-double to N ≈ 130, quad-double beyond; tolerance 10⁻³); the
costs are fp64 costs. Per-term rate past 32 pairs: flop scaling n⁴ (`rate_power`), until it is measured. Data: the
Borealis runs published by Xanadu (github.com/XanaduAI/xanadu-qca-data, Apache-2.0), chain order = time-bin order.

| run | modes, r | samples | N mean / max | t_run per sample, H100-s (v2.8 / self_all) | median sample | whole run on one H100 | measured / priced / N > 64; tier |
|---|---|---|---|---|---|---|---|
| fig3a | 216, 0.53 | 1,044,000 | 21.1 / 53 | 0.145 / 0.140 | 0.075 | 1.5·10⁵ s (42 h) | 100 / 0 / 0 %; fp64 |
| fig3b | 72, 0.89 | 1,036,000 | 22.4 / 61 | 0.280 / 0.120 | 0.033 | 2.9·10⁵ / 1.25·10⁵ s | 87 / 13 / 0 % (v2.8); 23 % of the cost needs dd |
| fig4 | 216, 1.10 | 1,026,000 | 124.3 / 219 | 3.6·10²⁴ / 1.1·10²² | 5.7·10¹⁴ / 8.4·10¹³ | 3.7·10³⁰ / 1.1·10²⁸ s | 100 % N > 64; qd |
| figS15 | 288, 1.01 | 1,113,000 | 143.2 / 242 | 5.5·10²⁷ / 4.3·10²⁵ | 8.1·10¹⁷ / 1.6·10¹⁷ | 6.1·10³³ / 4.8·10³¹ s | 100 % N > 64; qd |

Readings:
- **The two low-brightness runs are within reach of one GPU**: the whole fig3a run (1.04 M samples of the 216-mode
  device) is a 42-hour H100 job on measured kernels in fp64; fig3b 35-81 h, partly double-double.
- **The high-brightness runs are carried by their brightest few samples**: half of fig4's whole-run cost sits in its top
  3 (v2.8) / 5 (self_all) samples, N ≥ 215; figS15 in its top 1-3. The per-sample mean is therefore a statement about the
  run's maximum, not its typical sample (the median sample: 10¹⁴-10¹⁸ H100-s).
- The collision discount is worth ×2 (fig3b) to ×300 (fig4) on the run, ×7 on fig4's median.
- The synthetic check with model patterns ("be", 400 patterns) gave a median 8.6·10¹⁴ H100-s against the real fig4 median
  5.7·10¹⁴ (v2.8): the surrogate holds at the median of a device-size run.
- **Not measured**: every fig4 / figS15 call is N > 64 (no kernel; rate extrapolated) and needs quad-double; the loss is
  priced as the detected pattern of the device's mixed state, as for the frontier's η points. CSV
  results/keep/gbs_borealis_replay_20260927.csv (per N, both term counts).

**Measured: exact samples of the fig3a model on one H100** (run 20260927_190700, `gaussian.sample_transfer` on the
published r and T, `bench/gbs_borealis_exact.py`; §4.1). 10⁴ shots, none censored:
- **0.154 s per sample** (median 0.079 s), so the whole 1.04 M-sample run is **1.6·10⁵ s ≈ 44.5 h on one H100**. The
  timing model priced the same 10⁴ patterns at 0.977× the measured time (per-shot median 0.973), with 0.97-0.99 in
  every band of N; the replay of the device's own samples had predicted 0.145 s.
- **The law**: mean detected photons 21.58 ± 0.06, against 21.62 exact from the covariance of the published model and
  21.12 in the device's samples (8σ below). The difference is a uniform thinning: the device detects 2.1 % fewer photons
  than its published T gives. Thinned binomially by 0.979, our samples' P(N) sits at total-variation distance 0.023 from
  the device's, at the shot-noise floor of 10⁴ device samples (median 0.021, 99 % 0.027); unthinned it is 0.034. Per-mode
  mean photon numbers: ours / device 1.020 (median over modes, 5-95 % 0.97-1.08), correlation 0.93.
- CSV results/keep/gbs_borealis_fig3a_exact_20260927_h100.csv (per shot: N, seconds, model price, largest sieve), samples
  in gbs_borealis_fig3a_exact_20260927_h100_samples.npz.

**Phase B: the per-term rate to N = 128 and the double-double factor** (run 20260927_201539, job 24, H100 PCIe).
- Rate with the torch matrix part (batched zgemm, no Q = 64 limit; `gbs_timing_model.py calibrate --kernel torch`),
  n = 24-64 pairs: 1.78 / 3.86 / 7.18 / 12.3 / 20.6 / 30.1 µs per term at n = 24 / 32 / 40 / 48 / 56 / 64 (V = 1;
  V = 4: 3.59 / 7.32 / 13.2 / 21.7 / 34.4 / 49.8 µs). It is 1.4-3.1× the Triton rate where both run (n = 24-32), and
  n^2.9-3.2 over n = 48-64. The n⁴ extrapolation from the Triton rate at n = 32 used above lands at 0.77-1.21× of these
  measured values at n = 48-64.
- Replay with the Triton rates to n = 32, the torch rates to n = 64 and n³ beyond (the measured slope), v2.8 / self_all:
  fig4 median 5.6·10¹⁴ / 9.4·10¹³ H100-s (unchanged), mean 2.0·10²⁴ / 7.9·10²¹ (×0.55 / ×0.71 of the n⁴ values), whole
  run 2.0·10³⁰ / 8.1·10²⁷ s; figS15 median 6.7·10¹⁷ / 1.5·10¹⁷, mean 2.7·10²⁷ / 2.6·10²⁵. fig3a / fig3b unchanged (n <= 31).
  60.7 % of fig4's samples (21.1 % of figS15's) have N <= 128: every call of those, the median sample included, is now
  priced on a measured per-term rate; the means stay extrapolated (carried by the N >= 215 samples).
  CSV results/keep/gbs_borealis_replay_B_20260927.csv.
- **Double-double** (`bench/gbs_dd_rate.py`): the Ozaki dd product (β-bit slices, exact fp64 GEMMs, error-free sums)
  checked against exact rationals on the H100 itself (error 2^-108 relative to Σ|ab|; tensor-core GEMMs exact). dd / fp64:
  batched complex products ×560 / 411 / 310 / 249 at q = 32 / 64 / 96 / 128 as implemented (unfused: the slicing and the
  dd sums dominate; the GEMM count alone gives ×15 at q <= 128), element-wise complex multiply-add ×33. The products are
  13-29 % of a torch-path term (22-27 % at n = 64), so **a dd term costs ×28-92 an fp64 term** (×15 GEMM floor to ×249
  measured on the products, ×33 on the rest). Quad-double is not measured; its factor is at least the dd one.
- Consequence for the typical fig4 sample (N ≈ 124: dd suffices, N eps κ <= 10⁻³ to N ≈ 130 by the κ fit): 5.6·10¹⁴ ×
  (28-92) = 1.6-5.1·10¹⁶ H100-s = 0.5-1.6·10⁹ H100-years with v2.8, 0.8-2.7·10⁸ with the collision discount. Xanadu's
  "more than 9,000 years" on Fugaku is ~10⁸ H100-years by peak fp64 (537 PF / 51 TF), so the exact cost of the median
  sample, precision included, sits at 0.9-17× their estimate. The κ fit is extrapolated from N <= 56 to 124.

**Re-priced with the precision tiers (2026-09-29; `TimingModel.tiers`, `bench/gbs_device_replay.py --dd-rate`, no new
measurement; phase C).** Every multi call is now priced as the sieve with precision "auto" can run it: the cheapest
(pairing, arithmetic) whose round-off meets eps κ ≤ 10⁻³ — the error law measured in job 27 (error ≤ 0.74 eps κ), which
replaces the N eps κ criterion above — with κ from the fit, +5 decades for Bulmer's cross grouping (+4.6..+5.2 measured,
job 25), and the dd term at its measured cost (job 27: dd rates to n = 32, then dd / fp64 held at its n = 32 value, ×87
of the torch fp64 term at V = 1; job 28 measures it to n = 48). Tiers: fp64 to N = 62, dd to N = 141 ungrouped and to
N = 117 cross-grouped, quad-double beyond. No quad-double kernel exists: **"as built"** allows fp64 and dd only (a call
that fits neither is priced ungrouped in dd, a lower bound); the **qd** variants assume a qd kernel at 1× (a lower bound)
or 10× the dd time. Same rates as phase B (Triton to n = 32, torch to 64, n³ beyond), cross pairing.
results/keep/gbs_borealis_replay_tiers_20260929.json (per N and summary, every variant); table and figure by
`bench/gbs_device_table.py` (img/gbs_device_comparison.png).

| run | N mean / max | as built (fp64 + dd): per sample mean / median, H100-s | median sample | whole run, one H100 | with a qd kernel at 1× / 10× dd: median | fp64 only (v2.11 pricing): mean / median |
|---|---|---|---|---|---|---|
| fig3a | 21.1 / 53 | 0.133 / 0.075 | 0.075 s | 1.4·10⁵ s (39 h) | same | 0.139 / 0.075 |
| fig3b | 22.4 / 61 | 0.216 / 0.031 | 0.031 s | 2.2·10⁵ s (62 h) | same | 0.271 / 0.031 |
| fig4 | 124.3 / 219 | 1.1·10²⁶ / 3.0·10¹⁶ | 9.4·10⁸ H100-years | 3.7·10²⁴ H100-years | 1.4·10¹³ / 1.2·10¹⁴ | 1.9·10²⁴ / 5.5·10¹⁴ |
| figS15 | 143.2 / 242 | 1.6·10²⁹ / 3.8·10¹⁹ | 1.2·10¹² H100-years | 5.8·10²⁷ H100-years | 1.6·10¹⁶ / 1.6·10¹⁷ | 2.7·10²⁷ / 6.7·10¹⁷ |

Readings:
- **fig3a / fig3b are unchanged in kind and slightly cheaper**: every call fits fp64 under eps κ (fig3b's N ≤ 61; under
  N eps κ 23 % of its cost needed dd), and in fig3b 16 % of the cost moves to dd *by choice*: on its heavy samples
  (m = 72, N 50-61) the cross-grouped sieve in dd beats the ungrouped one in fp64 (the measured job-27 comparison at
  heavy N = 48: dd cross 9-33 s against fp64 ungrouped 23 s). fig3a's whole run: 39 h on one H100 (measured model
  basis: job 23, 0.977).
- **fig4, as built: the median sample costs 3.0·10¹⁶ H100-s = 9.4·10⁸ H100-years, ×54 its fp64 price**, and
  ×10 Xanadu's "more than 9,000 years on Fugaku" (3.0·10¹⁵ H100-s by peak fp64, 537 PF / 51 TF). The median sample
  (N ≈ 124) fits dd ungrouped (N ≤ 141), so this price is complete in precision — its remaining assumptions are the κ fit
  (extrapolated from N ≤ 56), the dd / fp64 ratio held past n = 32, and the torch fp64 rate extrapolated as n³ past n = 64
  (the fig4 median needs n ≈ 62: measured). The means are carried by samples with N > 141, all lower bounds here.
- **The grouping is where the number moves, and it needs quad-double.** Bulmer's cross grouping fits dd only to
  N = 117; past it, the as-built sieve pays the full 2^n (the ×10⁴ step at N = 118 in the figure). With a qd kernel the
  fig4 median drops to 1.4·10¹³ (qd at dd cost, a lower bound) - 1.2·10¹⁴ H100-s (qd at 10×): 0.005-0.04 × Xanadu's
  estimate. The qd factor is not measured; the κ penalty of the grouping is
  measured at N ≤ 48 only.
- figS15 follows fig4 at 10³× (median 3.8·10¹⁹ as built, 1.6·10¹⁶-10¹⁷ with qd).
- Compared with the earlier pricing (§2.4 above): the "self_all" counterfactual (grouping at every size in fp64,
  2.4·10¹¹ for fig4's median with the cross pairing) is not a sieve that meets the tolerance; the tiers replace it.
- **With job 30's rates (2026-09-29, run 20260929_103413; torch 2.14.0, the calibration's stack, fused slicing)**: the dd
  term is ×58-67 of the torch fp64 term at n = 24-48 (V = 1; ×35-42 at V = 4), rising only 5 % from n = 32 to 48, so the
  held-ratio assumption stands, but the ratio is ×0.73-0.80 of job 27's (×86.9 / ×52.9, dd measured on torch 2.12.1 before
  the fused slicing). Re-priced (results/keep/gbs_borealis_replay_tiers_job30_20260929.json; img/gbs_device_comparison.png
  regenerated from it), as built mean / median H100-s: fig3a 0.133 / 0.075 (run 39 h), fig3b 0.206 / 0.031 (59 h), **fig4
  9.0·10²⁵ / 2.4·10¹⁶ (median 7.5·10⁸ H100-years, ×8.0 Xanadu's estimate)**, figS15 1.3·10²⁹ / 3.1·10¹⁹. With a qd kernel at
  1× / 10× dd: fig4 median 1.1·10¹³ / 9.4·10¹³ (×0.004 / ×0.03), figS15 1.3·10¹⁶ / 1.3·10¹⁷. These supersede the table
  above for the dd prices; the κ penalty of the grouping (+5 decades) is unchanged — job 25's heavy Borealis patterns
  measured +4.6..+5.2, the light ones +1.5..+3.2, and job 29 saw the sieve keep partial groupings in fp64 at N = 40-49.
- **With the grouped price floored (2026-09-30, after jobs 31-32; results/keep/gbs_borealis_replay_tiers_floor_20260930.json,
  img/gbs_device_comparison.png regenerated)**: as built unchanged for fig4 / figS15 (median 2.4·10¹⁶ / 3.1·10¹⁹ H100-s: their
  medians run ungrouped in dd), fig3b 0.247 / 0.031 (run 71 h); with a qd kernel at 1× / 10× dd the fig4 median is 2.4·10¹³ /
  2.1·10¹⁴ H100-s (×0.008 / ×0.07 Xanadu's estimate; was 1.1·10¹³ / 9.4·10¹³), figS15 2.5·10¹⁶ / 2.5·10¹⁷.

### 2.5 GBS through a time-bin loop at small delay (2026-10-02, register count + H100 run 20261002_203546, job 33)

Part (b)'s Gaussian row (2026-10-01): the settings of the loop column (one rail, one loop of delay
L, T = 2N bins, loops start empty, random theta in [0, 2), phi in [0, 6)) with the GBS headline source (N squeezers of
0.5 photons in bins 0..N-1, lossless). A squeezed train has no finite Fock register, so the register (rail + L slots)
is capped at p photons under the GBS cutoff convention (at most eps = 1/S = 10⁻⁴ of the mass of a sample left out):
D = C(p + L + 1, L + 1). Code: `photonic_engines/gaussian/loop.py`, bench `bench/gbs_loop.py`.

**The cap is exact Gaussian algebra.** Without detection the register is Gaussian; its moments are propagated bin by bin
and the mass the cap drops at the injection of bin t is, to first order, sum_j P_slots(j) P_sq(q > p - j) (slots
before the injection, j <= p; squeezer q), from the generating function det((1 - z) sigma + (1 + z)/2)^(-1/2). Summed
over t it is the leak per sample; p(N, L) is the smallest p with leak <= 10⁻⁴. Checked against the capped sampler's own
weights (surviving squared norm at every injection): the model is 1.04-1.07 x the weights at leak 5·10⁻³-10⁻² and
1.0-1.5 x at 10⁻⁴ (CPU, `check` and a CPU frontier at N <= 32): conservative; the capped sampler's pattern probabilities equal the loop-hafnian
probabilities of the unrolled circuit to 2·10⁻¹⁶ (loops flushed; tests/test_gaussian.py). The sampler is K's
`SuperpositionLoopKernel` with the register basis capped at p (no edit of K's files); it renormalises after every
injection, so its rows are draws of the per-step renormalised truncated process, with the leak reported per run.

**Register count [computed, CPU].** The mean register content is 0.5 photon per mode ((L + 1)/2); p grows by ~3 per
x4 in N (the cap only has to cover T = 2N injections at 10⁻⁴ in total):

| L | N = 16: p, D | N = 64 | N = 256 | N = 1024 | N = 4000 |
|---|---|---|---|---|---|
| 1 | 20, 2.3·10² | 22, 2.8·10² | 25, 3.5·10² | 28, 4.4·10² | 30, 5.0·10² |
| 2 | 21, 2.0·10³ | 24, 2.9·10³ | 27, 4.1·10³ | 29, 5.0·10³ | 32, 6.5·10³ |
| 3 | 22, 1.5·10⁴ | 26, 2.7·10⁴ | 28, 3.6·10⁴ | 31, 5.2·10⁴ | 34, 7.4·10⁴ |
| 4 | 24, 1.2·10⁵ | 27, 2.0·10⁵ | 30, 3.2·10⁵ | 33, 5.0·10⁵ | 35, 6.6·10⁵ |
| 5 | 25, 7.4·10⁵ | 29, 1.6·10⁶ | 31, 2.3·10⁶ | 34, 3.8·10⁶ | 37, 6.1·10⁶ |
| 6 | 26, 4.3·10⁶ | 30, 1.0·10⁷ | 33, 1.9·10⁷ | 36, 3.2·10⁷ | 38, 4.5·10⁷ |
| 7 | 26, 1.8·10⁷ | 31, 6.2·10⁷ | 34, 1.2·10⁸ | 37, 2.2·10⁸ | 40, 3.8·10⁸ |
| 8 | 27, 9.4·10⁷ | 32, 3.5·10⁸ | 35, 7.1·10⁸ | 38, 1.4·10⁹ | 41, 2.5·10⁹ |

At fixed L the register grows only like (log N)^(L+1), so the cost per sample is ~T x D: near-linear in N, as for the
loop rows of single photons and SBS. In L it grows like p^(L+1) / (L+1)! with p ~ 25-40: from D ~ 10⁷ (L = 6) the dense
register is at the size where K's dense SBS row stops (D ~ 10⁷ at L = 8, N* = 21.4). Borealis' ladder (1, 6, 36), L = 43,
N = 216, T = 216, same squeezing: p = 66, D = 1.1·10³¹ (the structure-blind chain rule: 2^(N/2) at N ~ 108 detected).
**Measured N\*(L) [H100 PCIe, run 20261002_203546, job 33]** — 1 sample/s, the shot ramp and budget of
bench/hybrid_frontier.py (rate = shots / time over batches of up to 256 shots, 120 s per point), complex128:

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| N\* (squeezers) | ≥ 4000 (2.73/s at N = 4000, max-n) | 3171.3 | 2249.8 | 540.1 | 110.0 | 31.4 | 11.6 | 7.3 |
| p, D at the bracket | 30, 4.96·10² | 32, 6.5·10³ | 32-33, 5.9-6.6·10⁴ | 31-32, 3.8-4.4·10⁵ | 30, 1.9·10⁶ | 28, 6.7·10⁶ | 25-26, 1.4-1.8·10⁷ | 22, 2.0·10⁷ |

Leak in every one of the 72 points <= 1.0·10⁻⁴ by the sampler's own weights (max 9.8·10⁻⁵); model / weights 0.95-1.35.
The register count matched the run (same p at the same N, L). Crossover with the structure-blind exact sampler
(n\* = 41.5 at m = 2n, the same T = 2N): between L = 5 (110) and L = 6 (31.4), L ≈ 5.8 by log-interpolation. So part
(b)'s Gaussian row reads: easy at small L (thousands of squeezers at L <= 3), structure-blind-hard from L ≈ 6, and
Borealis' L = 43 (D ~ 10³¹) far beyond. Next to the other loop rows at L = 8: single photons (sectors) 1876, SBS
(dense, no cutoff) 21.4, GBS (cutoff register) 7.3.

Caveats. (i) At L >= 6 the batch is memory-limited (allocator out-of-memory warnings, batches of 32-128 shots): the
rate there is that of the largest batch that fits on one 80 GB H100. (ii) Where one call takes > 40 s the ramp stops
early (N = 3000-4000 at L = 2: 16-64 shots), which makes the rate step with the batch size; the L = 2 bracket (1.24/s
at 64 shots, 0.74/s at 32) is wider in rate than in N (3146 / 3207). (iii) The sampler draws from the per-step
renormalised truncated process (mass left out <= 10⁻⁴ per sample, reported), not from the exact distribution; the
lossless headline only (uniform loss is output thinning, free). (iv) The dense register, not photon-number sectors:
a squeezed train has no definite register photon number.

## 3. The sampler

### 3.1 The loop hafnian: the pair sieve (our derivation)

Pad N to even 2n with an isolated vertex of loop weight 1 if N is odd. Pair vertex a with
X(a) = a + n mod 2n, C = AX (C[a, b] = A[a, X(b)]). A loop-perfect matching together with the n
pairing links decomposes uniquely into disjoint cycles and loop-ended paths; a cycle through k pairs is
a closed walk of C of length k (2k orientations/rotations), a path through k pairs is
(XD)ᵀ C^{k−1} D (2 orientations). Inclusion–exclusion over the pair subsets S visited:

    lhaf(A, D) = Σ_{S⊆[n]} (−1)^{n−|S|} [λ^n] exp( Σ_{k≤n} λ^k ( tr(C_S^k)/(2k) + ½ (XD_S)ᵀ C_S^{k−1} D_S ) ).

(This is the eigenvalue-trace family of Björklund–Gupt–Quesada; the derivation above is ours and is
what the tests check against brute force for N ≤ 10, and against The Walrus to 1e-13.) The diagonal of A
never enters. With D = D0 + z D1 the path term is quadratic in z, so the λ-series is carried with
polynomial-in-z coefficients and one sieve returns every coefficient of z ↦ lhaf(A, D0 + zD1)
(`lhaf_poly`).

### 3.2 The chain rule

Heterodyne every mode first (Q function: α = β + U ε, ε independent per input mode, Var Re ε =
cosh r e^{−r}/2, Var Im ε = cosh r e^{r}/2; vacuum ½, ½). Then for j = 0..m−1 replace the heterodyne of
mode j by counting: the marginal of (n_<j, α_>j) does not depend on the measurement on mode j, so the
previous step's record is a sample of it, and n_j ~ P(n_j | n_<j, α_>j) completes it. Projecting
modes > j on coherent states leaves the pure state exp(½ wᵀB_{≤j}w + γᵀw), γ = ζ_{≤j} + B_{≤j,>j} ᾱ_{>j}.
With n_<j fixed and z the Bargmann variable of mode j,

    G(z) = Q(z) exp(b z²/2 + g z),  Q(z) = lhaf(B_<j[n_<j], γ_<j[n_<j] + z B_<j,j[n_<j]),  b = B_jj, g = γ_j,
    P(n_j = k | ·) = k! |[z^k] G|² / Z .

### 3.3 The exact normaliser

Z = Σ_k k! |[z^k]G|² = (1/π)∫ |G(z)|² e^{−|z|²} d²z. The weight is a real 2D Gaussian (|b| < 1):
mean E[z] = (b̄g + ḡ)/(1−|b|²), E[δz δz] = b̄/(1−|b|²), E[δz δz̄] = 1/(1−|b|²),
mass Z0 = exp((Re(b̄g²) + |g|²)/(1−|b|²)) / √(1−|b|²); Z = Z0 Σ_{l,l'} q_l q̄_{l'} E[z^l z̄^{l'}] with the
moments from Stein's recurrence. Checked against the series to 1e-15. The draw accumulates
|c_k|² = k!|[z^k]G|² (scaled Hermite recurrence, no overflow) until it passes uZ: no photon cutoff.
If rounding leaves the partial sums below uZ at a cap where the tail is < e^{−80}, the cap is returned
and the gap recorded (`ShotStats.norm_rel_err`; 0 in every test so far).

### 3.4 Pseudo-code — maintained, one block per version

**v1 (2026-09-23)** — correctness first, no optimisation.

```
sample(state{r[n], modes, m, disp}, U[m,m], shots, eta, schmidt=[1]):
  for each Schmidt mode k: r_k = r*sqrt(lam_k)
      (r'_k, w_k) = lossy_squeezing(r_k, eta)            # pure part + classical p-noise variance
      B_k = -U_c diag(tanh r'_k) U_c^T                     # [m, m]
  for shot in 1..shots:
    out = 0
    for each Schmidt mode k:
      d    = sqrt(eta)*disp (mode 0 only) + i*Normal(0, w_k)          # classical displacement [n]
      beta = U_c d ;  zeta = beta - B_k conj(beta)
      alpha = beta + U eps,  eps_i ~ Q-noise(r'_k,i) (vacuum on non-source modes)
      n = 0[m]
      for j in 0..m-1:
        gamma = zeta[0..j] + B_k[0..j, j+1..] conj(alpha[j+1..])
        A, D0 = expand(B_k[<j,<j], gamma[<j], n[<j]) ; D1 = repeat(B_k[<j, j], n[<j])
        q[0..N] = lhaf_poly(A, D0, D1)                      # pair sieve, 2^(N/2) subsets, batched on device
        Z = closed_form_norm(q, b = B_k[j,j], g = gamma[j])
        u ~ U(0,1);  n[j] = min{K : sum_{k<=K} |c_k(q, b, g)|^2 >= u Z}
      out += n
    emit out

lhaf_poly(A[N,N], D0, D1):                                 # N padded to 2n
  C = A[:, X];  acc[0..2n] = 0
  for S in subsets of [n] (batched):
    C_S, D_S = masked by S
    for k in 1..n: t_k = tr(C_S^k) (repeated product) ; paths w = C_S^{k-1} D0, y = C_S^{k-1} D1
                   g_k(z) = t_k/(2k) + ½ (XD0)·w + z (XD0)·y + z² ½ (XD1)·y
    F = exp(sum_k λ^k g_k(z)) truncated at λ^n (series recurrence, z-polynomials)
    acc += (-1)^(n-|S|) F_n(z)
  return acc
```

Known costs of v1, the optimisation list written before v2 (done without looking at other codes, per
the plan of 2026-09-23; the comparison with licence-clean codes comes after the GPU numbers):
1. powers C_S^k by repeated product: n products per subset where n/2 suffice (tr C^{a+b} = ⟨C^a, (C^b)ᵀ⟩);
2. no collision discount: repeated rows are separate vertices;
3. a run of empty modes recomputes a same-matrix sieve at every step (profile: 2.1 / 2.7 cost-weighted
   steps per distinct prefix at m/n = 1 / 2);
4. per-step host work in Python (`_amps`, `_norm`) and one host–device round trip per step;
5. shots are sequential; no batching of shots or of Schmidt modes.
Profile of v1 (CPU, n = 30, m = 60): 90 % of a shot in `lhaf_poly`, 4 % in `_amps` + `_norm`.

**v2 (2026-09-23 night)** — items 1-4, plus one found on the way: the masked [2n, 2n] subset matrices
of v1 do the work of a full-size matrix for every subset, most of it on zero rows.

```
sample_pure_v2(B, zeta, alpha, window W):
  j = 0
  while j < m:
    N = sum(n[<j])
    if N < V2_FROM (16):  one v1 step at j (small sieves: v1 has less overhead); j += 1; continue
    J = j .. j+W-1                                        # speculate: modes j.. stay empty
    idx = repeat(modes <j, n[<j]);  A = B[idx, idx]       # the same for every step of J
    D0[v] = zeta[idx] + B[idx, J_v+1..] conj(alpha[J_v+1..]);  D1[v] = B[idx, J_v]
    Q[v, 0..N] = lhaf_poly_multi(A, D0, D1, labels = idx) # one sieve for all v
    for v, jj in enumerate(J):                            # draws in order, as v1 does
      n[jj] = draw(Q[v], b = B[jj, jj], g = zeta[jj] + B[jj, jj+1..] conj(alpha[jj+1..]))
      j = jj + 1;  if n[jj] > 0: break                    # the rest of the window is discarded

lhaf_poly_multi(A[N,N], D0[V,N], D1[V,N], labels):
  pair identical vertices with each other (a mode with t photons: floor(t/2) self-pairs);
  groups g of r_g >= 2 identical self-pairs; the other pairs are free
  for every count vector s (s_g = 0..r_g):                 # prod (r_g + 1) terms, not 2^(sum r_g)
    pre = first s_g pairs of every group;  w = prod C(r_g, s_g) (-1)^(n_g - |s|)
    for every subset T of the free pairs, bucketed by size (few padded sizes, dummy zero vertex):
      S = pre + T, gathered C_S [2p, 2p] (not masked [2n, 2n]);  sign (-1)^(n_f - |T|)
      traces tr C_S^k = <C^a, (C^b)^T>, a = ceil(k/2), b = floor(k/2)    # ceil(n/2) - 1 products
      paths  (XY_S)^T C_S^(k-1) Y_S for all 2V loop vectors at once     # n products [2p,2p] x [2p,2V]
      series F_k = (1/k) sum_i i g_i F_(k-i), one contraction per k, z-degree <= 2k
      acc[v] += w sign F_n(z)
```

**v2.2 (2026-09-24) — v3 step 1, fewer launches** (the profile, run 20260924_124226: one v2 sieve at
N = 36 issued 60939 kernels and kept the H100 20 % busy). Same sums, same samples; two changes:

```
_free_subsets(n > 16 pairs):                    # was: a range of bitmasks cut into the ~8 buckets per range
  for bucket (lo, pb]:  batch = chunk_of(pb)    # batch sized for THIS bucket, not the largest
    for p in lo+1..pb:  ranks 0..C(n,p)-1 in batches, filled across sizes p
      subset = colex_unrank(rank, p):  for i = p..1: c_i = max{c : C(c,i) <= r}; r -= C(c_i, i)
                                       # one searchsorted per i on a cached C(c, i) table; no host sync

_terms, fused layout (default on CUDA; the per-k loops stay the default on the CPU, where the fused
layout's copies cost x0.4-0.9 on two cores):
  W = stack(C_S^(k-1) Y_S, k = 1..n)            # [c, n, 2p, 2V]; g_k for all k in three contractions
  F_k stored with two zero columns in front, so z F and z^2 F are views; each step
  F_k = (1/k) sum over [F, zF, z^2F] x [i g_i^(0), i g_i^(1), i g_i^(2)]   # ONE contraction per k
```

Expected launches per sieve at N = 36: ~20 batches x ~300 kernels instead of ~128 x ~500. Both layouts are
tested against v1 (`_FUSED` forced on the CPU; the generated path with `_TABLE_MAX = 0` and small batches).
CPU A/B against v2.1 on two cores, default layout: x0.90-1.07, i.e. unchanged within noise; version
difference 8e-9 relative at N = 34, V = 4, against N eps κ = 3.7e-6 there.

Measured on the H100 (run 20260924_141521): launches of one sieve at N = 36 60939 → 4432, GPU busy
20 → 96 %; one v2 sieve (V = 1) 908 → 277 ms at N = 36 and 118 → 36 s at N = 48, now ×2.3-2.5 faster than v1 from
N = 36; whole shots (window 4) 0.48 → 0.42 s at n = 40 and 4.9 → 2.65 s at n = 50 (×2.4 / ×3.7 over v1). The fused
layout wins only while launch-bound (N ≤ 28) and costs 10-36 % above (its concatenations), so the default became
**fused up to 16 pairs on CUDA, per-k loops above and on the CPU** (`_FUSED_MAX`).

**v2.3 (2026-09-24) — the pairing is chosen for precision.** Any perfect matching of the vertices into pairs
gives a valid sieve with the same value, not the same rounding. `bench/gbs_precision.py` (CPU, against the same
sieve in long double, eps 1.1e-19, itself checked against `lhaf_brute`) measured, N = 12-28, shapes free / 2x2 /
1x4 / 2x2x2, 2 matrices each:
- the actual error is 0.01-0.6 × eps κ (typically 0.05-0.2): κ is a reliable, slightly pessimistic predictor,
  and `ShotStats.round_off` = eps κ is a safe bound;
- κ spans 1-3 orders of magnitude over matchings (N = 28: 1.4e5 .. 4.3e8 over 8 random ones); the collision
  self-pairs of v2.1 were ×3-500 (median ×66) above the best of 8 random matchings on a mode holding 4 photons;
- Σ_S |term_S| is the only pairing-dependent part of κ (the value is fixed), and 128-256 random subsets
  estimate it well enough to rank matchings: Spearman 0.96-0.99 against the exact κ over 24 matchings,
  the pick within ×1.2 of the best. A structural proxy (Σ log |A_uv| over the pairs) does not work (Spearman
  -0.2..0.4).

```
lhaf_poly_multi(..., pairing = "auto"):
  if n_pairs >= _PAIR_SEARCH (18 on CUDA, 14 on CPU):
    candidates = [self-pairing with its collision groups] + 7 random matchings (seeded by N, reproducible)
    for each candidate: 128 random (count vector, free subset) terms, weight x (number of terms) / 128
                        -> estimate of sum_S |term_S|        # all candidates in ONE _terms call (block-diagonal C)
    pick the least estimate; keep the self-pairing if within x2 of it (it keeps the collision discount)
  else: self-pairing
```
CPU (search forced from 2 pairs): κ 5.6e5 → 7.0e3 (1x4, N = 20), 3.8e6 → 9.4e4 (1x4, N = 24), 6.8e6 → 8.7e4 (2x2x2,
N = 24), 2.7e8 → 1.1e7 (free, N = 32); cost +5-15 % at N = 28-32 on two cores (search, and the collision discount
lost when a random matching wins), more below the default threshold. GPU: job 14 (κ and error of the searched
and the self pairing at N = 32-48) and job 12 (whole shots).

**v2.4 (2026-09-24) — targets 1 and 2 of the profile.**

```
sample_pure, impl v2, sieve of N < SMALL_BELOW (14) vertices (CPU):
  lhaf_poly_small (numpy): masked [2^n, 2n, 2n] subsets all at once, the series as one einsum triple per k;
  windows as for the large sieves                       # was: torch v1 per step, 2-3.5 ms of dispatch per call

sample(cpu_below = 14): sieves of 14+ vertices on the GPU (was 18)
```

CPU (two cores): `lhaf_poly_small` ×3-5 per call at N ≤ 11 (0.46 vs 2.12 ms at N = 8), ×1.5 at 15, none at 17+;
whole shots with window 4 at n = 20: 0.088 → 0.044 s, at n = 30: 0.246 → 0.168 s (the sieves under 8 vertices
×10); the draws are now 19 % of a shot at n = 20.

H100 (run 20260924_162756, window 4): n = 20: 0.038 → 0.019 s/shot, n = 30: 0.083 → 0.057, n = 40: 0.415 → 0.38; the
sieves under 8 vertices 0.98 → 0.023 s per 16 shots at n = 40. What was left on the CPU at n = 40 were the
14-17-vertex sieves (the 8-15 bucket at 5.5 ms per call on average, the 16-17 ones at ~30 ms), against 2.2 ms for a GPU sieve at
N = 16, hence `cpu_below` 18 → 14 (not yet measured on the GPU).

*Negative result, removed:* batching the large sieves' n - 1 path products into two matrix products and the
traces into two stacked contractions (tried for the gemv kernels that topped the N = 36 profile) was slower on the
H100 — one sieve 1.36 vs 1.20 s at N = 40 and 6.97 vs 6.05 s at N = 44 (jobs 12 / 15 of that run) — and on the CPU (×0.56-1.0). With 2V = 2
columns the products stay matrix-vector (memory-bound) whatever their grouping, and the stacked traces add a
temporary; the remaining lever for those sieves is keeping C_S on chip (a fused kernel, v3 candidate 1).

**v2.5 (2026-09-24) — the matrix part in a Triton kernel** (`gaussian/sieve_triton.py`; v3 candidate 1, first step).

```
powers_paths kernel, one program per subset S (grid = batch of subsets):
  ids = idx_S padded to Q in {16, 32, 64} with the dummy vertex (zero row / column)
  C = Ce[ids, ids]  -> registers, float64 planes (re, im)
  P = I
  for k = 1..n:
    for v < V:  a_k, b_k, c_k = x0^T P y0, x0^T P y1, x1^T P y1      # P = C^(k-1): two matvecs, three dots
    P = P C  (4 real tl.dot, float64);  tr_k = trace(P)
  write tr [c, n+1], paths [c, V, n+1, 3]
torch: g_k from tr_k and the paths; the series F_k as in the loop layout; sum with the signs
```

The subset matrix and its powers never leave the chip (the torch layouts re-read C_S from memory for each of
~n matrix-vector products). The kernel does n products per subset where the torch code does ceil(n/2) with the
<C^a, (C^b)^T> traces, which needs the powers kept. Default: on CUDA from `_TRITON_FROM` = 17 pairs (the loop
layout's range); `_TRITON` forces it either way. Tested against v1 under the Triton interpreter (a fast case
per run, larger ones in the `slow` tier).

H100 (run 20260924_180802, one sieve, v2, kernel forced vs torch forced): V = 1: ×1.6-1.7 from N = 36 (0.165 vs
0.267 s at N = 36, 18.9 vs 30.0 s at N = 48); V = 4: only ×1.08-1.10; at N ≤ 28 a wash (2.6 vs 2.2 ms at N = 16,
10.5 vs 11.2 at N = 28). All rows within N eps κ of v1. The trace at N = 36 (0.171 s wall, 84 % busy, 3248
launches): the kernel 61 ms, and the torch series 61 ms — its per-k einsums are batched matrix-vector products
(gemv), now the other half. The first sieve of a new size paid ~1 s of compilation (Triton specialises integer
arguments equal to 1 or divisible by 16), which also polluted that run's whole-shot timings.

**v2.6 (2026-09-24) — the series in a second Triton kernel; no per-size compilation.**

```
series kernel, one program per (subset, loop vector), lanes over the z coefficients (LP = 16 / 32 / 64):
  scratch rows F_0 .. F_n (global, per program; L1/L2-resident); F_0 = 1
  for k = 1..n:
    t = sum_(i=1..k) (i g_i)_0 F_(k-i)[z] + (i g_i)_1 F_(k-i)[z-1] + (i g_i)_2 F_(k-i)[z-2]   # shifted loads
    F_k = t / k;  barrier                              # row k is read by other lanes at k + 1
  write F_n
both kernels: do_not_specialize(E, q, n, L): one compilation per (V, Q) and per width; sieve_triton.warmup()
compiles them all before any timing (bench/profile_gbs.py, bench/find_threshold.py)
```
Removes the ~8n torch launches and the gemv of the series per batch. Interpreter-tested against v1; the
barrier's correctness is only testable on the GPU (the interpreter runs lanes in lock-step).

H100 (run 20260924_190851; every row within N eps κ of v1, so the barrier holds): one sieve, V = 1, Triton vs torch:
N = 36 0.093 / 0.267 s, N = 40 0.396 / 1.193, N = 44 1.97 / 5.99, N = 48 11.8 / 30.0 (×2.5-3.0; ×6.9 over v1 at
N = 48); V = 4 ×2.2-2.4; and at N = 16-28 too (0.8 / 2.2 ms at N = 16, 3.7 / 11.2 ms at N = 28), hence the kernels
on every CUDA sieve since (`_TRITON_FROM` = 1). Trace at N = 36: 0.096 s wall, matrix kernel 61 ms, series kernel
13 ms, 1391 launches. Whole shots, window 4: n = 40 0.265 → 0.180 s, n = 50 3.05 → 1.42 s (×5.6 / ×6.9 over v1).
Triton compile of every variant: 31 s once, then 0.5 s from the disk cache.

**v2.7 (2026-09-24) — even powers only in the matrix kernel.** The matrix kernel was 61 of the 96 ms at N = 36.

```
C2 = C C                                       # once
P = I
for j = 0 .. floor(n/2):                       # P = C^(2j)
  tr C^(2j)   = trace P
  tr C^(2j+1) = <P, C^T>                       # elementwise, no product
  paths of k = 2j+1:  x^T P y;   of k = 2j+2:  x^T P (C y)       # C y: one matvec per vector
  P = P C2
```
floor(n/2) + 2 complex products per subset instead of n (×1.6 fewer at n = 18, ×1.8 at n = 24). Interpreter-tested
against v1 (odd and even n).

*Negative result, reverted* (H100, run 20260924_195027): one sieve, V = 1, 0.098 / 0.42 / 2.07 / 12.2 s at
N = 36 / 40 / 44 / 48 against 0.093 / 0.40 / 1.97 / 11.8 for v2.6; V = 4 1.3-1.6× SLOWER (42.7 vs 27.4 s at N = 48).
The products were not the cost: the path terms are, as register matrix-vector reductions (v2.7 needed three per
vector per two steps instead of two per step, plus the <P, C^T> reduction). The same run, with the kernels on every
CUDA sieve: whole shots, window 4, n = 20 / 30 / 40 0.015 / 0.036 / 0.142 s (v2.6: 0.017 / 0.042 / 0.180).

**v2.8 (2026-09-24) — the paths as one product per step.**

```
Y = [y0_v | y1_v | y1_v]_v, X = [x0_v | x0_v | x1_v]_v    # [Q, VP]: the 3V (x, y) pairs, padded to VP >= 16 columns
P = I
for k = 1..n:
  U = P Y                                               # one complex [Q, Q] x [Q, VP] tl.dot for all paths
  paths_k = column sums of X * U                        # x^T C^(k-1) y for every pair, one reduction
  P = P C;  tr_k = trace P
```
Interpreter-tested against v1 (V = 1-3).

H100 (run 20260924_202647; every row within N eps κ of v1): V = 4, one sieve, 0.153 / 0.70 / 3.44 / 19.2 s at
N = 36 / 40 / 44 / 48 (v2.6: 0.221 / 1.03 / 5.01 / 27.4; torch 0.52 / 2.50 / 12.3 / 60.1); V = 1 slightly slower than
v2.6 (12.9 vs 11.8 s at N = 48: three useful columns of 16), so V = 1 keeps v2.6's two matrix-vector reductions since.
Whole shots, window 4: n = 40 0.142 → 0.105 s, n = 50 1.59 → 0.90 s (v1: 1.00 / 9.93 s, ×9.6 / ×11.1). At n = 40 the
14-15-vertex sieves still went through torch v1 on the GPU (0.33 s of 1.68 s per 16 shots, ~5 ms per call), hence
`sampler.V2_FROM` 16 → 14; at n = 50 the 40-47-vertex sieves are 11.0 of 14.3 s.

**v2.9 (2026-09-24, measured slower, removed) — consecutive powers, half the products.** Job 17 (run 20260924_214308, the kernel cost split)
put the Q × Q products at a large part of the v2.8 kernel (see §8), so the traces and paths of C^(2a-1) and C^(2a)
are both taken from P_a = C^a and P_(a-1):

```
paths_1 = x^T y
P_prev^T = I, P = C, R_prev = Y                          # P = C^a, R_prev = C^(a-1) Y
for a = 1..h = ceil(n/2):
  tr_(2a-1) = <P, P_prev^T>                              # sum_ij P_ij (P_prev)_ji
  P^T = trans(P);  tr_(2a) = <P, P^T>                    # if 2a <= n
  L = P^T X;  R = P Y                                    # V > 1: two complex tl.dot [Q, Q] x [Q, VP]; V = 1: 4 matrix-vector reductions
  paths_(2a) = column sums of L * R_prev                 # x^T C^(2a-1) y, if 2a <= n
  paths_(2a+1) = column sums of L * R                    # x^T C^(2a) y,   if 2a + 1 <= n
  P_prev^T = P^T;  R_prev = R;  if a < h: P = P C
```
h − 1 products instead of n; the same number of path products and of Q × Q reductions; one register transpose of P
per a. Interpreter-tested against numpy powers and against v1.

**Negative result, removed** (H100, run 20260924_221455; v2.8 kept). Kernel, µs per subset (job 18, v2.8 / v2.9):
n = 24, p = 12: V = 1 0.39 / 0.75, V = 4 0.47 / 0.55; p = 22 (Q = 64): V = 1 2.98 / 18.3, V = 4 3.12 / 4.90; n = 12,
p = 6: V = 1 0.08 / 0.16. One sieve at N = 48: V = 1 36.8 s (v2.8b 11.8), V = 4 22.8 s (19.2); shots, window 4,
n = 50: 0.97 s (0.87). The halved products are real — v2.9's valid "products only" is 0.12 µs for 11 products at
n = 24, p = 12, so v2.8's 24 cost ≈ 0.26 of its 0.39 — but the transpose and the two full Q × Q inner products per a
cost more than the 13 products saved (v2.9 "no traces" 0.46 vs 0.75 at V = 1), and at Q = 64 the extra live [Q, Q]
planes presumably spill registers (×6). v2.8's diagonal traces are cheap (0.03 µs) because the product already holds
C^k. Like v2.7: fewer products do not pay when they are bought with reductions or data movement in registers.

**v2.10 (2026-09-27) — collision groups across modes, batched, chosen by cost under a round-off check.** Three changes to
`lhaf_poly_multi` (the value is unchanged; which terms are enumerated and how they are batched):
- **Cross-mode groups** (`_pairing_cross`; the term count of Bulmer et al., arXiv:2108.01622, reimplemented from the
  identity, not from their code). All vertices of one repeated mode are identical, so k pairs (u_i, v_i) with every u_i of
  mode a and every v_i of mode b are k identical pairs, enumerated by count: k + 1 terms instead of 2^k. v2.8's
  self-pairs reach only floor(t/2) + 1 for a mode of t photons. Measured (CPU, patterns of N = 26-28 cut from Borealis
  fig4 samples, against a long-double sieve): ×2.7-55 fewer terms than a matching without groups, ×2-10 fewer than
  the self-pairing, and κ between the two (×3-5·10⁴ above the best random matching, ×10-10⁵ below the self-pairing).
  The error stays 0.07-0.3 eps κ: the grouping is exact.
- **Batched count vectors** (`_sieve_multi`): the count vectors of one prefix length |s| share their `_terms` calls with
  the free subsets (rows = prefix + free subset, weight = count-vector weight × subset sign). v2.8 issued one call per
  count vector, which left grouped sieves launch-bound: on CPU, heavy patterns at N = 24-28 ran ×0.4-1.3 of v2.8 with
  ×30 fewer terms; batched, ×2.1 median (forced cross ×3.3, up to ×8.3).
- **Choice by cost under a round-off check.** Below the search (`_pick_grouping`): the cheapest of ungrouped, self
  and cross by `_pairing_cost` (per prefix length one sieve of (count vectors) × 2^{n_f} terms, above the launch floor
  `_LAUNCH_FLOOR`). From the search on (`_choose_pairing`): candidates self, 3 cross, 4 random; κ of each predicted as
  the κ fit times its estimated Σ|term| / the least one; the cheapest with predicted N eps κ ≤ `_ROUND_TOL` = 10⁻³
  (the fp64-tier criterion of §2.4) is taken, else the least estimate. After the sieve, a grouped pairing whose actual
  N eps κ misses the tolerance is redone with the best-conditioned candidate.

```
lhaf_poly_multi(A, D0, D1, labels, pairing = "auto"):
  n_pairs < _PAIR_SEARCH: (perm, groups) = cheapest of {ungrouped, _pairing, _pairing_cross} by _pairing_cost
                          fallback = a random matching (seed N)
  else:                   (perm, groups), fallback = _choose_pairing(...)      # cheapest predicted within _ROUND_TOL
  acc, acc_abs = _sieve_multi(perm, groups)                                    # count vectors batched by |s|
  if groups and N eps max_v(max acc_abs / max |acc|) > _ROUND_TOL: acc = _sieve_multi(fallback)

_sieve_multi(perm, groups):
  for L, [(prefix, weight)] of the count vectors with |s| = L:
    rows(prefix) with empty free subset, weight x (-1)^n_f
    for each bucket batch of free subsets T: rows = prefix + T for every prefix of this L (as many as fit the batch)
                                             weight = weight(prefix) x (-1)^(n_f - |T|)
    _terms(rows, weights)
```
Timing model (`timing.group_sizes`, `TimingModel.pairing = "cross"`): a collided call below the search is priced with
the grouping `_pick_grouping` takes, per prefix length as one free sieve of 2 n_f + 2 log2(count vectors) vertices; above
the search it stays priced ungrouped (the sieve may group there when κ allows: an upper bound). Replayed on the Borealis
samples (fp64 prices, H100 model): fig3a ×1.04 and fig3b ×1.01 below the search alone; the grouping kept at every size
(the counterfactual without the round-off limit) gives fig3a ×1.4, fig3b ×4, fig4 median 5.6·10¹⁴ → 2.1·10¹⁰ H100-s
(×2.7·10⁴), figS15 median ×1.5·10⁴. At fig4 sizes every pairing needs double-double (§2.4), so that gain needs a dd sieve.
Bench: `bench/gbs_pairing_bench.py` (patterns cut from the Borealis samples; time, κ, terms, redo per rule; job 25).

**Measured on the H100** (run 20260928_081620, job 25):
- **κ price of the cross grouping** (one sieve per pattern cut from the Borealis samples, against the best-conditioned
  matching), in decades: heavy collisions (fig4) −1 at N = 24-28, +2.7 at N = 32, **+4.6 to +5.2 at N = 36-48** (range
  +1.8 to +9.8) for ×57-152 fewer terms; light collisions (fig3a) +1.5 to +3.2 at N = 32-48 for ×2-3 fewer terms. Forced
  cross is ×9.7 (median; ×23 summed) faster on heavy patterns at N = 36-48 and ×1.8 on light ones, at N eps κ up to 10²:
  the precision limit, not the term count, decides above ~32 vertices in fp64. The automatic rule never exceeded
  N eps κ = 6·10⁻⁵ on heavy patterns; v2.8 reached 2.3·10⁻² there (its self-pairing below the search at N = 32, κ
  3·10¹²), which the new rule caught and redid (4 redos in 56 sieves, ×0.55 at worst). Single sieves at N >= 36 are
  ×1.0 (no qualifying grouped candidate).
- **Whole shots, the same seeds as job 22** (v2.8, run 20260925_095009; all 259 uncensored shots common to both give
  identical patterns): summed time ×1.41 at 1.5 photons per squeezer (n = 24), ×1.48 at m = n (n = 40), ×1.19 at the
  headline m = 2n, n = 44; per-shot medians ×0.96-1.26: the gain is in the heavy shots. The timing model with
  `pairing = "cross"` gives 1.06 / 1.09 of the measured sums at m = n / the headline (0.91-0.92 per-shot median) and
  1.33 at 1.5 photons (grouped calls above the search, priced ungrouped). Borealis fig3a, 2000 exact shots: 0.132 s per
  shot (v2.8 run 20260927_190700: 0.154 on 10⁴), model / measured 0.997.
- The frontier of §2.3 does not move in the model (n* 41.5 / 42.9 / 9.7 at the headline, m = n, 1.5 photons, S = 10⁴,
  both pairings: the calls that carry t_S are above the search and priced ungrouped). The measured shot gains say it
  would move up by ~ln(1.2-1.5) / ln ρ_eff ≈ +0.6-1.3 squeezer; not re-derived (capped shots only).
- **Consequence for device sizes**: at fig4's N ≈ 124 the cross grouping costs ~5 decades of κ (if the plateau of N = 36-48
  holds) for ×10³-10⁴ fewer terms: worth it only in an arithmetic that can absorb the 5 decades (quad-double where
  double-double sufficed). A dd / qd sieve is the next step for the device comparison.
- CSVs results/keep/gbs_pairing_20260928_h100.csv, gbs_cross_shots_20260928_h100.csv,
  gbs_borealis_fig3a_exact_cross_20260928_h100.csv.

**v2.11 (2026-09-28) — the sieve in double-double** (`gaussian/dd.py`, `hafnian._terms_dd`, `lhaf_poly_multi(precision=...)`).
Why: the sieve's error is ~eps κ with κ = Σ_S |term_S| / max |c| growing ~10^0.197 per vertex (§2.4), so fp64 misses
N eps κ ≤ 10⁻³ from N ≈ 55: the frontier's own tail (N_S = 54-64 at S = 10⁴), fig3b's largest samples and every
device-size call. The cross grouping (v2.10) costs a further ~5 decades on heavy collisions. A dd sieve moves the limit
to N ≈ 130 (ungrouped) and gives the grouping room.
- Arithmetic (`gaussian/dd.py`): (hi, lo) pairs of complex128 tensors; error-free transformations (two_sum, Dekker
  two_prod, no fused multiply-add) for the element-wise work; the matrix products by the Ozaki scheme (β-bit slices
  whose fp64 products are exact in any order, checked on the H100 tensor cores in job 24), summed in dd. Checked against
  exact rationals: 0.02-0.53 eps_dd (eps_dd = 2^-104).
- The term (`_terms_dd`): the torch per-k layout of `_terms` (powers P^1..P^h, traces <P^a, (P^b)^T>, path vectors
  C_S^(k-1) Y_S, g_k, the series F_k) with every product and sum in dd, and the sum over subsets accumulated in dd
  (where the cancellation is). Inputs are fp64, exact in dd; the result is rounded to complex128.
- `precision`: "fp64" (default), "dd", or "auto": fp64, redone in dd when the round-off still misses `_ROUND_TOL`
  after the pairing fallback, or run in dd from the start when the cheapest dd-admissible pairing is `_DD_FACTOR`
  (250, below) cheaper by `_pairing_cost` than the cheapest fp64-admissible one; the
  pairing choice and the check then use eps_dd. `sample` / `sample_transfer` / `sample_pure` pass it through; the small
  numpy sieves (< 14 vertices) stay fp64.
- Checked (CPU): against a 50-digit brute force the dd result is exact to its complex128 rounding (6·10⁻¹⁷ of max |c|,
  fp64 10⁻¹⁵-3·10⁻¹⁴) for every pairing; on Borealis-like matrices (r = 1.1) with heavy collisions dd agrees with a
  long-double sieve to 1.3-2.4·10⁻¹⁶ (the reference's own accuracy) where fp64 is off by eps κ (9·10⁻¹² for the
  self-pairing, κ 10⁵); the sampler in dd draws the same samples as in fp64 on a well-conditioned instance.
- Cost, measured on the H100 (job 26, run 20260928_092453): 228 / 321 / 390 / 777 / 1370 / 2281 µs per term at n = 12 / 16
  / 20 / 24 / 28 / 32 pairs (V = 1; V = 4: 479 - 3672 µs), i.e. **×710-1990 of the fp64 Triton term** and ×390-590 of the fp64
  torch path. Launch-bound, not arithmetic-bound: a batch issues ~25 000 small kernels (each dd operation is 5-15 of them, the
  Ozaki product ~200), where the GEMM count alone would make the products ×15. `_DD_FACTOR` was first set to this ×1000; v2.11b
  (job 27, run 20260928_204415) measured ×116-253 (V = 1-4, n = 12-32), so it is now 250, the upper end (it steers only
  "auto"; the timing model's tiers use the measured rates). The first value, ×100, sent heavy N = 48 sieves to dd at
  ×3.7-5.8 of their fp64 time. Whole sieves: dd cross grouping 7-10 s at
  heavy N = 24-28, 16-37 s at N = 36-44, 44-105 s at N = 48 (fp64 ungrouped 11.8 s), 115-565 s on light N = 40-44.
- The actual-error comparison of the job (every fp64 rule against the dd result) was lost to the 1 h timeout (the CSV was
  written at the end; the bench now writes it after every pattern); the log is kept (results/keep/gbs_pairing_dd_20260928_h100.log).
- Next: the dd term needs fusing (fewer, larger kernels: the Ozaki slices of C_S computed once per batch instead of per
  product, the element-wise dd chains compiled into single kernels, larger batches) before it is a production path; until
  then precision="auto" escalates to dd only where fp64 cannot meet the tolerance at all.

**v2.11b (2026-09-28) — the dd term with fewer, larger kernels** (~1 300 per batch at n = 32 instead of ~25 000):
- **Grouped Ozaki products** (`dd.plan`, `dd.slices`, `dd.matmul_sliced`): slices on a fixed exponent schedule, so the
  products A_i B_j of one order t = i + j share a unit and form ONE fp64 GEMM over the concatenated inner dimension (exact
  while 2β − 2 + log2(s q) ≤ 53: β = 22-24, s = 5 up to q = 256), 5 GEMMs per real product instead of 15.
- **Complex products as one real product** of the block forms [[Ar, −Ai], [Ai, Ar]] [Br; Bi] (`dd.embed_left`,
  `embed_right`, `cmatmul(left=, right=)` with reusable slices), instead of four. Checked against exact rationals,
  including rows 2^±30 apart: 0.01-0.04 eps_dd.
- **Powers by doubling** (P^(m+j) = P^j P^m for all j ≤ m in one stacked product: ~log2 h products instead of h) and the
  **path vectors from the powers** (w_(j+1) = P^j Y, then w_(h+1+j) = P^h w_(j+1): two stacked products instead of n − 1).
- **Traces and path dot products stacked** over k: one dd product-and-sum each instead of n.
- **The series in a Triton kernel** (`gaussian/sieve_dd_triton.py`, the dd port of the fp64 series kernel): one program per
  (subset, loop vector), dd in registers, two_prod by fma on CUDA (exact whatever the compiler contracts: the other
  error-free transformations have no products) and by Dekker's split under the interpreter. Tested under the interpreter
  against the torch recursion and a 50-digit brute force (slow tier); the fma path is checked on the GPU by
  `tests/check_gaussian_dd_gpu.py` (job 27), against exact rationals and the CPU dd sieve.
- On CPU (compute-bound, no launch cost to save) the dd term is ×2 slower than v2.11 (n = 12: 5.0 vs 2.4 ms); the GPU rate
  is job 27's.
- **First GPU run (job 27, run 20260928_135612): the dd products were NOT exact on CUDA** (cmatmul q = 64: 1033 eps_dd,
  q = 128: 43; q = 16 and every element-wise op within 0.5 eps_dd), and the dd sieve agreed with the CPU one only to
  3·10⁻¹⁵ - 2·10⁻¹² (fp64 level, amplified by κ). Cause: the slice exponents came from ceil(log2(max)) and the scales
  from exp2, exact on the CPU but not guaranteed on the GPU; an exponent one too small gives slices of β + 1 bits, and
  the plan sat exactly at the 53-bit limit. Fixed: exponents by `frexp` (exact), scales 2^k built from their bit pattern
  (`dd._pow2`), and one bit of margin in `dd.plan` (β = 21-23). The same run ran out of the H100's 80 GB in the pairing
  bench at N = 40 (dd batches sized at ×12 of fp64); now ×48 (`_DD_MEM`). A check C isolates the series kernel.
- **Second GPU run (run 20260928_202924)**: q = 128 exact after the margin; the q = 64 "failure" (1032.99 eps_dd, the same
  number as before the exponent fix) is the check's own tolerance — it measured the error against Σ|a||b| while the Ozaki
  bound is normwise (q × row max × column max), and its elements are scaled 2^±20 independently; the same data gives the
  same number on the CPU, and 0.00 eps_dd normwise. Check C: the series kernel on CUDA agreed with the torch dd recursion
  only to ~10⁻¹⁷ relative (fp64 level: the lo parts lost), while it is exact under the interpreter: the kernel's plain
  operators and tl.fma did not survive compilation as error-free transformations. Now every add / sub / mul / div / fma
  of the kernel's dd arithmetic is an inline PTX instruction with explicit rounding (`add.rn.f64`, ...), opaque to the
  optimizer.
- **Third GPU run (run 20260928_204415): every check passes.** Products 0.00 eps_dd normwise at q = 16 / 64 / 128; the series
  kernel against the torch dd recursion on CUDA 8·10⁻³³ - 1.7·10⁻³² relative (was 10⁻¹⁷); the CUDA dd sieve equal to the CPU
  dd sieve to the last bit (0.0) on six instances N = 12-24, where fp64 is off by 4·10⁻¹⁵ - 10⁻¹¹.
- **Measured dd rate (v2.11b)**, µs per term, V = 1: 61 / 57 / 81 / 142 / 222 / 336 at n = 12 / 16 / 20 / 24 / 28 / 32 (V = 4:
  75 - 387): ×116-253 of the fp64 Triton term, ×46-87 of the fp64 torch path (n = 24-32), ×3.8-10.5 faster than v2.11.
- **The fp64 error model, measured** (bench/gbs_pairing_bench.py --dd, 55 of 56 patterns before the 90 min timeout; the dd
  result of the cross pairing as reference, itself confirmed by a second pairing in dd to 5·10⁻¹⁶ on 32 patterns): on 275
  fp64 sieves (every rule, N = 24-48, heavy and light Borealis patterns) the actual error / (eps κ) has median 0.031, 90 %
  0.19, max 0.74; error / (N eps κ) at most 0.027. κ bounds the error with room to spare, and **N eps κ ≤ 10⁻³ — the fp64
  criterion used so far (§2.4) — is conservative by ≥ ×37**. Errors by rule: v2.8 up to 6·10⁻⁵ (heavy N = 32, its
  self-pairing), the automatic rule at most 6.8·10⁻⁷, forced cross in fp64 up to 2.8·10⁻³ (never chosen by the rule);
  precision="auto" used fp64 on all 55 (every one met the tolerance).
- Consequence: with the measured law, error ≤ 0.74 eps κ, and the κ fit, fp64 meets 10⁻³ to N ≈ 63 (not 54) and dd to
  N ≈ 142 (not 130). The re-pricing of the frontier and the device rows (§2.4) uses this criterion.

**v2.11c (2026-09-28) — the Ozaki slicing fused** (`sieve_dd_triton.slices_complex`, used by `dd.left_slices` /
`right_slices` on CUDA; `dd._FUSED_SLICES`). Not yet measured on the GPU (job 28).
- Why: on the CPU profile of a dd batch (bench/gbs_dd_profile.py, n = 8-12) the slicing of the operands' block forms is
  most of the element-wise work: per slice, (x + σ) − σ, the remainder and a two_sum, i.e. ~10 torch kernels over the
  whole block matrix, s = 5 slices per operand, plus the two `cat`s of the embedding; the path-vector products (P^j times
  a thin Y) spend nearly all their time slicing P^j (63-71 % of the batch at n = 12 on the CPU).
- What: one Triton kernel reads (re, im) of hi and lo once and writes all s slices directly into the block positions
  ([[re, −im], [im, re]] for a left operand, [re; im] for a right one); the row (column) exponents are one torch amax +
  frexp over max(|re|, |im|). The same inline-PTX add.rn / sub.rn as the series kernel keep (x + σ) − σ and the two_sum
  exact on CUDA.
- **Bit-identical** to `dd.slices` of the embedded block forms: the block row maximum is max(|re|, |im|) of the complex
  row, and round-to-nearest-even about a positive σ is odd in its argument, so the slices of −im are minus those of im
  (tests: interpreter against torch on shapes with zero rows and entries over ~10 decades; the dd sieve with the fused
  slicing forced on returns exactly the unfused result; GPU check D in `tests/check_gaussian_dd_gpu.py`).
- Pseudo-code (per complex entry x = (re, im) of (hi, lo), exponent e of its block row / column):
  ```
  for k = 1..s:  σ = 0.75 · 2^(e + 53 − k β)
                 a = (re_h + σ) − σ;  b = (im_h + σ) − σ          # the k-th slices of re and im
                 (re_h, re_l) = two_sum(re_h − a, re_l);  (im_h, im_l) = two_sum(im_h − b, im_l)
                 left:  S_k[r, c] = a, S_k[r, Q + c] = −b, S_k[P + r, c] = b, S_k[P + r, Q + c] = a
                 right: S_k[r, c] = a, S_k[P + r, c] = b
  ```
- **Measured (job 28, run 20260929_072630, H100)**: every GPU check passes with the fused slicing (products 0.00 eps_dd,
  fused = torch slicing bit for bit on 8 shapes, CUDA dd sieve = CPU dd sieve). One dd batch of 4096 subsets is
  ×1.29-1.41 faster than with the torch slicing (n = 16 and 32; dd / fp64 per batch ×97-195, was ×115-275); launches per
  batch 2 250-2 510 (was 2 960-3 360). What is left: ~47 % "other" element-wise dd work (g_k assembly, the stacked sums,
  gathers), traces 14-22 %, powers 10-16 %, paths 10-13 %; the queue is launch-saturated ("Command Buffer Full"). Next
  fusion target: the element-wise dd of the traces and dots (one kernel per stacked cmul + dsum). The profile at n = 32,
  p = 32 ran out of memory (the bench's fixed batch; now sized as the sieve sizes it) and the dd rate to n = 48 did not run:
  job 30.
- **Job 30** (torch 2.14.0, batches as the sieve sizes them): ×1.36-1.40 per batch at n = 32 (p = 16, 32, V = 1 and 4) and
  n = 48 (p = 24); dd per batch ×156-171 of the Triton fp64 term at n = 32, V = 1, ×84 of the torch one at n = 48. The dd
  rate: 43.9 / 245.6 / 816.6 µs per term at n = 12 / 32 / 48 (V = 1), ×1.26-1.38 faster than job 27's; dd / torch fp64
  ×58-67 at n = 24-48, nearly flat.

**v2.1 (2026-09-24)** — the cost-benchmark cap (§2.1), both versions: after each draw
`if sum(n) > max_photons: censored; return` (row -1, not a sample). A shot finishing at ≤ max_photons is
unchanged, draw for draw.

The draws consume the generator exactly as v1's, so v1 and v2 return the SAME samples (tested bitwise
at every window; `bench/bench_gbs_impl.py` checks it on the GPU). CPU measurements (two cores,
2026-09-23): one sieve at N = 24 / 28: v1 0.39 / 2.3 s, v2 0.13 / 0.77 s (V = 1); the per-vector cost
at V = 4 is 1.0 / 0.55 s. Whole shots, m = 2n, 0.5 photons per squeezer: n = 26: v1 3.6 s, v2 1.0 s
(×3.7, same samples); n = 20: 0.14 → 0.10 s. What remains per vector is the series (≈ 2n³ per subset,
bandwidth-bound einsums) — the next target, together with a kernel of our own for the gathered
small-matrix products (docs §6). The subset index lists are generated on the device per chunk
(`_free_subsets`), never tabulated: a table would be 2^{n} × 2n entries, 64 GB at N = 60, which the
tail shots reach.

v3 candidates, to be ranked by the sampling benchmark `bench/profile_gbs.py` (job 12; decided 2026-09-24:
optimise sampling on this fixed benchmark, separately from the frontier estimation). It times one sieve per
size and shape (N = 16..48; collision-free, 2x2, 1x4; V = 1 and 4), traces a sieve with the torch profiler
(kernel time over wall = how launch-bound a size is), and splits whole capped shots at fixed seeds into
sieve time per size, draws, host preparation and the rest (`sampler.Profile`, opt-in via `sampler.PROFILE`).
On two CPU cores the draws are already 17-21 % of a shot at n = 10-16, and most sieve calls are below
8 vertices (37 of 45 per shot at n = 10):
1. a fused kernel per subset (powers, traces, paths and series in shared memory, complex128 as two
   float64 planes) — removes the ~5n launches per bucket chunk that make N ≤ 26 launch-bound on the H100;
2. shots in lockstep: streams at the same sieve size N share one launch sequence (the Fock kernels' B axis);
3. an adaptive window (the chance that the next mode is empty is known from the current conditional);
4. series by evaluation at 2n + 1 points of z (n³ instead of 2n³ per subset), if its interpolation
   error stays below eps κ;
5. (added 2026-09-24) the per-call cost of the small CPU sieves, < 16 vertices: 35 % of a shot at n = 40 after v2.2;
6. (added 2026-09-24) the pairing and κ — v2.3 above;
7. the comparison with licence-clean codes (The Walrus, Piquasso, DeepQuantum — all Apache-2.0) for
   tricks we are missing, after 1-4. Collision gain measured on the sampled patterns: E[2^{N/2}] /
E[collision-aware terms] = 1.25 at m/n = 2 and 2.0 at m/n = 1 (0.5 photons per squeezer, n = 20),
larger in the bright tail that carries the mean.

## 4. Imperfections

### 4.1 Loss

The channel V → ηV + (1−η)V_vac, mean → √η mean (`imperfections.gaussian_loss`), commutes with U, so it
is applied per source. For the chain rule the lossy squeezed vacuum is split (`lossy_squeezing`):
e^{−2r'} = ηe^{−2r} + 1 − η (pure part), plus a real Gaussian displacement of p with variance
w = (ηe^{2r} + 1 − η − e^{2r'})/4 ≥ 0. Exact; η = 1 returns (r, 0) and the sampler then draws exactly the
lossless samples (tested bitwise). Cheaper than the Fock case: no subset split, and N falls.

**A general lossy transfer matrix T** (`sample_transfer`, 2026-09-27): a published device matrix (Borealis: the three
loops, the losses and the detector efficiencies in one 216 x 216 contraction) does not commute with the sources, so
the per-source split does not apply. `transfer_decomposition` works on the output covariance
V = T_R V_in T_R^T + (I − T_R T_R^T)/4 (vacuum I/4, xxpp): Williamson, V = S D S^T with D ≥ 1/4, gives the pure part
S S^T / 4 plus the classical noise W = S (D − I/4) S^T ≥ 0; a Takagi factorisation of the pure part's
⟨Δa Δa^T⟩ = −U′ diag(sinh 2r′) U′^T / 2 gives squeezers r′ through a unitary U′. Each shot draws a displacement
(x, p) ~ N(0, W) and runs the unchanged chain on the displaced pure state. Exact; the decomposition is not unique
(for T = √η U it differs from `lossy_squeezing`'s, with more squeezing in the pure part) and the law does not depend on
the choice. Oracle: T as the top-left block of its unitary dilation on 2m modes, lossless truncated SLOS on the
dilation marginalised over the environment modes (a uniform-loss stand-in at the mean transmission fails it,
χ² = 152 on 7 dof). 0.16 s for the 216-mode decomposition. `bench/gbs_borealis_exact.py` samples a Borealis run's
model with it and compares with the timing model and with the device's samples (phase V of the device comparison).

### 4.2 Partial distinguishability: Schmidt modes

A squeezer emitting into K orthogonal Schmidt modes with weights λ_k is K independent squeezers
r_k = r√λ_k (the low-gain / no-time-ordering mapping — an assumption, stated in the docstring) sharing U;
photons of different Schmidt modes do not interfere, so a shot is the sum of K independent shots
(`sample(..., schmidt=λ)`, tested against the convolution of K oracle distributions). There is no
x-model here (survey §6.2): the minor Schmidt modes interfere among themselves. Predicted law, to be
measured: the frontier follows the leading instance, cost per squeezer Π_k Φ_{r_k}(1 − η + η√2).

## 5. Validation (CPU, 2026-09-23)

| check | oracle | result |
|---|---|---|
| pair sieve, N = 0..10 | brute-force matchings | ≤ 4e-15 relative |
| lhaf_poly coefficients | brute force at 4 values of z | ≤ 1e-14 |
| loop hafnian N = 6, 11, 16 (and 20, 24 by hand) | The Walrus `hafnian(loop=True)` | ≤ 7e-13 relative |
| closed-form Z | the series Σ|c_k|² | ≤ 3e-15 |
| pure probabilities, displaced | The Walrus `probabilities` | ≤ 1e-12 |
| pure probabilities | Piquasso GaussianSimulator; truncated ProductSLOS | ≤ 1e-13 |
| lossy oracle chain | Piquasso (Attenuator) vs truncated, thinned ProductSLOS | ≤ 1e-8 (truncation tail) |
| sampler lossless, lossy (η = 0.6), displaced, Schmidt (0.75, 0.25), photon-number law | chi-square vs exact (threshold dof + 5√(2 dof)) | pass |
| sampler 20000 shots, 3 squeezers in 4 modes | truncated ProductSLOS | χ² = 35.8 / 36 dof |

## 6. GPU

`lhaf_poly` runs on any torch device (complex128, subsets batched within a 4 GiB budget on CUDA);
loop hafnians below 18 vertices stay on the CPU (launch-bound). First GPU run (20260923_224054, H100
PCIe): the GPU sampler passes the oracle chi-square lossless and at η = 0.6; one lhaf_poly
costs 5.7 / 15.3 / 95 / 674 ms at N = 20 / 28 / 32 / 36 (×2.3 per two photons at the top, CPU faster
up to N = 16); whole shots at m = 2n, 0.5 photons per squeezer cost 0.05 / 0.06 / 0.29 s at n = 20 / 30 / 40.

**First frontier point (v1, run 20260923_225057):** m = 2n, 0.5 photons per squeezer, exact PNR, no
cutoff: **n* = 38.0 squeezers at 1 sample/s** on one H100 (19 detected photons on average). Flat
below n ≈ 30 (launch-bound), 1.30–1.40 per squeezer between n = 34 and 38–39, against the collision-free
model's 1.414. By coincidence it sits at the Fock frontier's 38.5 single-photon sources, with half the photons.
A v1 baseline, not yet the GBS frontier.

**Precision.** The sieve is an alternating sum. Its condition number κ = Σ_S |term_S| / max|c| (returned
by `lhaf_poly(..., return_abs=True)`) sets the rounding error, ≈ ε κ. On random O(1) test matrices κ reaches
4e7 at N = 26, which is why CPU and GPU (and The Walrus) differ there by ~1e-9: that is rounding, not a
bug, and the GPU check now tolerates N ε κ. On the sampler's own matrices it is far smaller — worst
ε κ over a shot 3e-14 at n = 20 — and `ShotStats.round_off` records it per shot so the frontier runs report
it at every size. If it grows towards 1e-6 at frontier sizes, the fix is compensated summation of the
sieve (double-double accumulators), not a change of algorithm.

`tests/check_gaussian_gpu.py` (run on a GPU machine) checks GPU vs CPU agreement, the GPU
sampler against the oracle, and times lhaf_poly and whole shots. The first frontier runs give v1 baselines, not the
GBS frontier.

## 7. Strong simulation: what SLOS becomes for Gaussian inputs

(Research note, 2026-09-23; route 3 below is now built and validated. The hybrid sampler was dropped on
2026-09-24, see the negative result below.)

**The object.** The output distribution is over every photon number, unbounded. A table can only hold
N ≤ N_cut, so strong simulation is defined by a **global photon cutoff**. Strawberry Fields' Fock
backend truncates *per mode* (cutoff dimension D, D^m amplitudes) — the truncation trick the user
flagged; a global cutoff is the right generalisation of our graded basis and is exponentially smaller
(C(m + N_cut, N_cut) states). Two consequences:
- for a **lossless** input, truncation is exact for every pattern with N ≤ N_cut (raising never
  lowers the photon number), and the photon-number law P(N) is U-independent and known in closed
  form. So an **exact strong-then-sample GBS sampler exists**: draw N from the exact law; if
  N ≤ N_cut draw the pattern from the table's layer N (exact conditional); else fall back to the chain
  rule. The table serves the frequent low-N shots, but they are the cheap ones: the gain is negligible
  (negative result below);
- for a **lossy** input the detected layer N receives mass from every lossless layer above it, so a
  thinned truncated table is approximate (error ≤ the lossless tail above N_cut, bounded exactly from
  the law); the exact route is the density-matrix (doubled, 2m-mode) recurrence, squared memory.

**Three engines, cheapest last.**
1. *Truncated ProductSuperposition through `ProductSLOSKernel`* (exists). Per-source raising steps
   q ≤ d−1; but the kernel builds every layer up to sources·(d−1), far above N_cut. A `max_photons`
   cap on the graded recurrence (layers above N_cut are never read by layers below) makes it exact for
   N ≤ N_cut at the cost of the graded basis to N_cut, times ~N_cut raising passes per source.
2. *Per-pattern loop hafnians* (`probs.pure_probability`): C(m+N, N) × 2^{N/2} — worse at every size.
3. **The Gaussian recurrence on the SLOS layers — built, `slos.GaussianSLOSKernel` (2026-09-23).**
   The raw SLOS amplitudes are the coefficients c(t) = [w^t] N exp(½ wᵀBw + ζᵀw) (amplitude
   c √Π t!, probability |c|² Π t! = `layers.prob_norms`). Euler's identity Σ_i w_i ∂_i F = (ζᵀw + wᵀBw) F
   gives, with B = −Σ_s tanh r_s u_s u_sᵀ (u_s = U[:, mode_s]),
       N c_N = R(ζ) c_{N−1} − Σ_s tanh r_s R(u_s) R(u_s) c_{N−2},
   R(v) the SLOS raising pass with per-mode multiplier v_j. **No new basis and no new edge set**: one
   ζ-raise plus two raises per squeezer per layer, on the edges `FockLayers` already stores. (A first
   prototype used the one-mode recursion of Miatto & Quesada, amp(k+e_i) = (ζ_i amp(k) + Σ_j B_ij √k_j
   amp(k−e_j))/√(k_i+1), which needs lowering edges; Euler's form avoids them.) Validated: equal to the
   truncated ProductSLOS oracle to 1e-15 and to the per-pattern loop hafnians (displaced input) to 1e-15,
   layer masses equal to the photon-number law to 1e-14. Timing on two CPU cores: 8 squeezers, m = 16,
   N ≤ 8 in 1.3 s; N ≤ 10 in 13 s (no optimisation, per-mode slabs as in `ProductSLOSKernel`). Under loss
   the thinned blocks are lower bounds short by at most `tail_mass()` (tested). Reach on the GPU:
   `bench/bench_gaussian_slos.py`, not run yet.

   Pseudo-code, strong v1:
   ```
   GaussianSLOS(state, U, N_cut):
     (B, zeta, logN) = bargmann(U);  u_s = U[:, mode_s];  t_s = tanh r_s
     c_0 = [exp(logN)]
     for N in 1..N_cut:
       c_N = R(zeta) c_{N-1}
       if N >= 2: for s: c_N -= t_s R(u_s) R(u_s) c_{N-2}
       c_N /= N
     p_N = |c_N|^2 * prod_norms(N)                   # exact lossless for every N <= N_cut
   ```

**Memory frontier (counted, `bench/memory_frontier.py`, column `gbs`).** n squeezers at 0.5 photons each,
table of every even N up to the cutoff leaving 10⁻³ of the mass out, 80 GiB: 14 / 10 / 8 / 6 / 5 / 4 / 3
squeezers at m/n = 1 / 1.5 / 2 / 3 / 4 / 6 / 8 — below the Fock curve (13 at m/n = 2), because the
cutoff must sit well above the mean photon number n/2 to capture the tail. A count, not a benchmark;
the ε choice moves it.

**Negative result (2026-09-24): a partial table does not move the sampling frontier (GBS or Fock).**
The generalisation discussed: store as much of the output distribution as fits in memory (any set T of
patterns, e.g. a lexicographic block of the sorted outputs, whose mass M = sum_T p is known exactly), draw
from T with probability M, else from the chain rule on the complement.
1. *Rejection gains nothing.* Sampling the complement by running the full chain rule and rejecting hits in
   T costs 1/(1−M) runs on a branch taken with probability 1−M: exactly one chain-rule run per sample. A
   gain needs the chain rule to sample the complement directly, i.e. T a union of subtrees of its decision
   tree, so the stored mass can be subtracted at each node. A lexicographic block is such a union for a
   mode-by-mode chain rule (the GBS one); not for Clifford & Clifford, which places photons, not modes.
2. *The gain is at most the cost share of T, and it is small.* Fock: every pattern has the same N and about
   the same cost, and output probabilities are close to Porter–Thomas (exponential), so a block holding a
   fraction f of the patterns covers M ≈ f; even the heaviest f patterns cover only f(1 − ln f) (and finding
   them needs the whole distribution). Speed-up 1/(1−M): ≤ 1.5 at 10× past the memory frontier, ≤ 1.06 at
   100×. Building part of the last SLOS layer still needs the full layer before it (memory saving only a
   factor ~n/(m+n)). So it helps only just past the memory frontier, where a sample is already cheap.
3. *GBS, T = all patterns with N ≤ N_cut.* Computed (not benchmarked) at n = 38 squeezers, m = 76,
   0.5 photons per squeezer, 80 GiB of complex128 even layers: N_cut = 6, P(N ≤ 6) = 3.5% of the shots,
   and 1e-6 of the run-of-10⁶ cost (cost model 2^(N/2), sum to N_S = 70). The cheap shots fit in memory;
   the cost is in the bright ones (§2.1).
Where a partial table would help: probability mass concentrated on a small set of patterns (heavy loss /
low N, strong bunching m ≈ n, structured U), i.e. near-classical regimes where approximate methods already
win. The hybrid sampler is not built.

**Next.** A GPU run of `GaussianSLOSKernel` for its reach; loss by thinning with the tail bound
reported, the density-matrix recurrence only if an exact lossy table is needed.

## 8. Comparison with licence-clean codes (2026-09-24)

Read after v2.8, as planned (own optimisation first). All three are Apache-2.0: The Walrus 0.22.0 (Xanadu; its GBS
sampler and batched loop hafnian are Bulmer's code for arXiv:2108.01622), Piquasso 8.0.1 (Budapest QCG), DeepQuantum
(installed without dependencies for reading). None has a GPU path for the exact PNR sampler; The Walrus and Piquasso
are numba on the CPU (`numba.prange` over the sieve terms), DeepQuantum is plain torch.

| idea | where | ours | verdict |
|---|---|---|---|
| chain rule with heterodyne outcomes, all photon numbers of the current mode in one sieve ("loop_hafnian_batch": the new mode as a repeated self-edge up to the cutoff) | TW `samples.generate_hafnian_sample`, Piquasso `_generate_sample` | same chain; our z-polynomial gives every count without a cutoff | nothing to take |
| repeated edges: pair the photons of repeated modes ACROSS modes (`matched_reps`: largest multiplicity with the second largest), prod (r + 1) terms | TW `_hafnian.matched_reps`, Piquasso `match_occupation_numbers` | self-pairs of one mode only | more collision discount; κ not measured (our self-pairs raised κ x3-500, §3.4 v2.3) — test before adopting |
| Glynn-form sieve (every term the full matrix with rows scaled by ±1, × 2^-n) | TW default (`glynn=True`), Piquasso (δ of Eq. B11) | inclusion-exclusion on gathered subsets | no precision gain over ours (below), and no subset-size reduction: ×4-8 more flops per term (estimated from the padded sizes). Not adopted |
| power traces in O(q³) per subset: Householder Hessenberg reduction + La Budde characteristic polynomial (+ Newton), loop terms by Hessenberg matrix-vector products | Piquasso `powtrace.py`, `labudde.py`, `loop_corrections.py` (Eq. 3.26 of 1805.12498); TW computes powers up to the matrix size and the charpoly beyond | n products per subset, O(n q³) | the one transferable trick: ~0.4 n × fewer flops per subset (×~10 at n = 24, estimated); but sequential, reduction-heavy per-subset work — our measurements say reductions, not products, cost on the GPU (v2.7). Prototype needed (precision and GPU rate) |
| matrix scaling by the mean |entry| before the sieve, result rescaled | Piquasso `_get_scale_factor` | none | range safety only; not needed in float64 at our sizes |
| modes sampled in increasing mean photon number | TW `photon_means_order` | natural order | fewer steps at large running count; no gain for Haar U (equal means), cheap to add for structured U. Any order must be fixed before the draws (an order depending on the heterodyne outcomes is not the same chain) |
| hafnian by integer partitions of power traces | DeepQuantum `hafnian_.py` | — | a reference implementation, not a performance code |

**Precision against our long-double sieve** (`bench/gbs_compare_walrus.py`, CPU, the matrices of `bench/gbs_precision.py`,
N = 16-28, free and 1x4, 2 matrices each; relative error of lhaf(A, D0)): ours v1 1.7e-14 .. 9.8e-12, ours v2 2.7e-14 ..
1.6e-10 (below the pairing search's threshold on the CPU, i.e. with the self-pairs); The Walrus Glynn 8.6e-15 .. 5.0e-8,
The Walrus inclusion-exclusion 1.4e-12 .. 4.9e-6 (up to 2.1e-3 at N = 28, 1x4). So our pairing-searched
inclusion-exclusion is as accurate as their Glynn form, and their inclusion-exclusion is not a usable reference at these
sizes. Their loop hafnian with `reps` returned O(1)-wrong values on the 1x4 shapes in my calls — most likely my
convention for loops with repetitions, not checked further, so no conclusion is drawn from it.

**Follow-up measurements (2026-09-24, CPU):**
- *Cross-mode repeated edges* (The Walrus's `matched_reps` pairing, realised through v1 by a vertex permutation; the
  κ of the pairing itself, without grouping): shapes 1x4, 2x2, 3x3, 2x2x2, (4,2), (3,3,2) at N = 16-24, 2 matrices
  each. κ ×0.05-10 of the self-pairing's, and ×1.5-200 above the best of 8 random matchings (e.g. 3,3,2 at N = 24:
  2.4e6 vs 1.2e4; errors 5.5e-11 vs 1.2e-13). Pairings that repeat identical edges are ill-conditioned whether the
  edges are self or cross: not adopted; the searched pairing (v2.3) stays.
- *The O(q³) route*, CPU prototype of the full sieve (`bench/gbs_charpoly_proto.py`; Householder Hessenberg of every C_S, the Hessenberg
  characteristic-polynomial recurrence, the trace part of the series as det(I − λC_S)^(-1/2) by the power-series
  recurrence of P^α, paths by matrix-vector products): relative error 9e-16 .. 2e-11 at N = 4-28, within ×0.2-7 of the
  power route on the same sieves for N ≥ 8 (up to ×40 at N = 4, on errors of 1e-16 .. 1e-15), same κ (3 .. 1.8e6). So precision is not the obstacle. Flops per subset at
  q = 2p ≈ n: about 14 q³ for Hessenberg + recurrence against 8 n Q³ for the n products (Q = padded q), ×~30 fewer at
  n = 24 (estimated). Whether that pays on the GPU depends on how much of the kernel the products are: v2.7 / v2.8
  say the reductions dominate, and the Hessenberg reduction adds ~4q of them. Job 17 (`bench/bench_sieve_kernel.py`)
  measures the split before any Triton Hessenberg is written.
- *Job 17* (H100, run 20260924_214308, 65 536 subsets of p = n/2 pairs, µs per subset): all / no paths / no traces
  = 0.08 / 0.04 / 0.07 (n = 12, V = 1), 0.30 / 0.22 / 0.27 (n = 18), 0.39 / 0.29 / 0.36 (n = 24); V = 4: 0.10 / 0.04 /
  0.09, 0.35 / 0.22 / 0.33, 0.47 / 0.29 / 0.44. Traces cost ~0.03 at n = 24; products + traces 0.29 of 0.39-0.47, so
  the products are ~55-67 % of the kernel by subtraction (the parts overlap: "no products" at V = 1, 0.26, would give
  33 %). The variants "products only" (0.00: dead-code eliminated) and "no products" at V = 4 (0.02: the identity
  folded) were not measurements. Conclusion: the O(q³) route removes the products but its Hessenberg adds ~4q
  reductions per subset, and the reductions are what is left; not pursued on the GPU for now. Halving the products
  within the power route (v2.9, §3.4) was measured slower: the transposes and inner products it needs cost more than
  the products it saves. v2.9's valid split confirms the products at ≈ 2/3 of v2.8's kernel at n = 24 (0.26 of 0.39 µs).
- *Modes in increasing mean photon number*: `sample(order="mean")`, the exact output means from U, the squeezing,
  the loss and the displacement; the law is tested unchanged (chi-square on a U whose order is not the natural one).
  Default stays "natural" (Haar U: equal means, and identical samples to earlier runs for a seed).

**Done above (prototype); remaining if the split allows it:** the O(q³) route on the sieve (Hessenberg + La Budde, and the trace part of the
series directly as det(I − λC)^(-1/2), which needs the characteristic polynomial but not the power traces), measured
against the long-double reference for precision and counted for flops; then a Triton version only if both hold up.
