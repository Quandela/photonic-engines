# Exact sampling for single-emitter (caterpillar) inputs

Derivation and algorithm behind `clifford/emitter_reference.py` (oracle) and
`clifford/emitter_sampler.py` (batched torch), and the specification for the Triton/CUDA
version. Notation: n photons, m modes, unitary U; the emitter has dimension χ (2 for a spin)
and emits photon k into one of L input modes `modes[k][l]` with operator `M^(k)_l` (χ×χ):

    |ψ⟩ = Σ_x c_x |x⟩,   c_x = ⟨f| M^(n)_{x_n} ⋯ M^(1)_{x_1} |s0⟩,   x ∈ {0..L-1}^n.

`|x⟩` is the Fock state with one photon in `modes[k][x_k]` for every k (all these modes are
distinct). Traced emitter: sample the final basis state s_f with probability
`p(s_f) = Σ_x |⟨s_f|⋯|s0⟩|²` (a χ² transfer-matrix chain), then proceed with `f = |s_f⟩`.

## 1. The chain rule and why the photon order must be random

Clifford & Clifford (2018) sample the *ordered* output modes r_1, …, r_n one at a time. With a
uniformly random permutation σ of the photons, at step k the k-th mode is drawn with weight

    w_k(ρ) = p_{T_k}(r_1, …, r_{k-1}, ρ),      T_k = σ(1..k),

where `p_T(z)` is the probability of the ordered row sequence z for the *state of the
photons in T alone*. For a Fock input that state is the Fock state on T. For an entangled
input it is the reduced state `ρ_T = Tr_{T^c} |ψ⟩⟨ψ|`, and

    p_T(z) = ⟨v_z| ρ_T |v_z⟩,    (v_z)_{x_T} = per( U[z, modes(x_T)] ).

Two facts make the algorithm exact (proof sketch, same structure as the Fock case):

*Marginal lemma.* Summing the n-photon probability of an ordered output over the last n−k
rows gives `((n−k)!/n!) Σ_{|T|=k} p_T(z_{1..k})` (with p_T = ⟨v|ρ_T|v⟩ as defined here, no 1/k!; the constant is candidate-independent). Proof: Laplace-expand the amplitude
`Σ_x c_x per(U[z, modes(x)])` along the last n−k rows into `Σ_T per(U[z_{≤k}, modes(x_T)])
per(U[z_{>k}, modes(x_{T^c})])`; summing `|·|²` over `z_{>k}`, unitarity kills every cross
term between different `(T, x_{T^c})` because the corresponding (n−k)-photon input states
are orthogonal (distinct mode sets). What remains is `Σ_T Σ_{x_T, x'_T} ρ_T[x_T, x'_T]
per(·) per(·)*`, i.e. `Σ_T ⟨v|ρ_T|v⟩`. ∎

*Telescoping.* Clifford's induction uses only the marginal lemma applied to the state on
`T_k` (the (k−1)-marginal of `p_{T_k}` is the average over dropping one photon of T_k), and
the fact that partial traces compose (`Tr_t ρ_T = ρ_{T∖t}`). Both hold for `ρ_T`, so the
theorem carries over verbatim. The proof needs σ uniform over *all* permutations: with a
fixed order the weights are not marginals of one joint distribution. Measured on the n = 3
linear cluster: fixed order gives TVD 0.22 against a statistical floor of 0.05 (test
`test_fixed_photon_order_is_wrong`).

## 2. The weights as a sum of pure terms

`ρ_T` is mixed because the not-yet-placed photons are traced out. Write the chain in slot
order and call a *gap* a maximal run of traced slots. A gap of slots a+1..b−1 between kept
slots a and b acts on the emitter as the channel

    E(ρ) = Σ_{x_gap} A_{x_gap} ρ A_{x_gap}^†,   A_{x_gap} = M^(b−1)_{x_{b−1}} ⋯ M^(a+1)_{x_{a+1}},

whose Kraus rank is ≤ χ² (the rank of span{A_x}); a prefix gap (before the first kept slot)
acts on |s0⟩ and a suffix gap on ⟨f| with rank ≤ χ. Choosing Kraus operators `K^(g)_i` for
every gap (SVD of the stacked `A_x`, composed slot by slot with a reduction after each slot so
the cost is linear in the gap length),

    ρ_T = Σ_e |ψ_e⟩⟨ψ_e|,   ψ_e(x_T) = ⟨f_e| M̃^(k)_{x_k} ⋯ M̃^(1)_{x_1} |s0_e⟩,

with `M̃` the kept slots' operators left-multiplied by the Kraus operator of the gap that
follows them. Zero-norm chains are dropped (for GHZ, |↑⟩⟨↑| and |↓⟩⟨↓| Kraus operators of
different gaps annihilate each other and 2 terms survive whatever the gaps). R = #terms.

    w(ρ) = Σ_e |amp_e(r_1..r_{k-1}, ρ)|²,   amp_e(z) = Σ_{x_T} ψ_e(x_T) per(U[z, modes(x_T)]).

## 3. Matrix Glynn sums and matrix cofactors

Glynn's formula for a (k−1)×(k−1) block B with columns indexed by the kept slots j and
`δ ∈ {+1}×{±1}^{k−2}`:

    per(B) = 2^{−(k−2)} Σ_δ sign(δ) Π_j s_j(δ),   s_j(δ) = Σ_i δ_i B[i, j].

For one pure term, the sum over x_T factorizes through the chain: with the per-(slot, label)
column sums `s_{j,l}(δ) = Σ_i δ_i U[r_i, modes[T_j][l]]`,

    Σ_{x_T} ψ_e(x_T) Π_j s_{j,x_j}(δ) = ⟨f_e| G_k(δ) ⋯ G_1(δ) |s0_e⟩,   G_j(δ) = Σ_l s_{j,l}(δ) M̃^(j)_l.

Laplace expansion along the candidate row ρ (the k-th row) of the k×k permanent gives, for
all candidates at once,

    amp_e(z_{<k}, ρ) = Σ_c Σ_l U[ρ, modes[T_c][l]] · Tr( M̃^(c)_l K_c ),
    K_c = 2^{−(k−2)} Σ_δ sign(δ) p_c(δ) q_c(δ)^T,   p_c = G_{c−1}⋯G_1 s0_e,   q_c^T = f_e^† G_k ⋯ G_{c+1}.

So the per-term work is a Gray-code sweep over δ with, per term, one forward and one backward
sweep of k χ×χ matrix-vector products (the Fock kernel's scalar prefix/suffix products), and
K_c is a χ×χ matrix per column. The m candidates then cost `k·L` complex multiply-adds each.
This is exactly `matrix_cofactor_weights`; it agrees with the explicit `⟨v|ρ_T|v⟩` to 1e-16.

The 2020 repeated-rows collapse (Clifford & Clifford 2020) acts on the δ enumeration only —
rows of B are output modes, repeated when photons collided — so the mixed-radix Gray walk of
`cofactors_triton.py` applies unchanged with the scalar products replaced by the matrix
chains above.

## 4. Cost

Per step: `R_k · 2^{k−2} · k` matrix-vector products (χ = 2: ~4× a scalar product) plus
`m·k·L·R_k` for the candidates. Measured R_k (mean over σ, linear cluster, spin projected):
peak ≈ 8 at n = 8, 30 at n = 12, 94 at n = 16, at k ≈ n/2.5, and 1–2 for the last steps that
carry the 2^k weight. Implied total cost over the Fock sampler: 7.5× (n = 8), 9.7× (12),
11.5× (16) — roughly 2 + 0.6 n — about 4 photons on the simulability frontier at equal rate.
Worst-case bound `R_k ≤ min(2^{n−k}, 4^{#gaps})`; the measured means are far below it.

## 5. Batched torch implementation (`sample_torch`)

S streams in lockstep. Per step: streams grouped by (sorted T, final id) — one gap
decomposition per distinct key, cached across steps and streams; lanes = (stream, term)
pairs; column sums per stream for all δ (chunked); forward/backward chains as batched
einsums over [lanes, δ-chunk]; K_c accumulated over chunks; candidate amplitudes by one
einsum against U; `w[s, ρ]` by index_add over lanes; multinomial per stream. Validated
against `EmitterSLOSKernel` histograms (L = 3 and χ = 3 included); δ-chunking is
bit-exact. CPU (2 cores): 3 ms/sample at n = 8, 15 ms at n = 10, 45 ms at n = 12.

## 6. Triton kernels (`clifford/emitter_triton.py`, `emitter_triton_seq.py`)

Lanes = (pure term λ, Gray segment) as in `cofactors_triton.py`; every lane walks a contiguous
segment of the Glynn sum over δ ∈ {+1} × {±1}^(K1−1) with the same Gray-code facts and row loads
as the Fock kernel, the scalar column products replaced by the χ × χ chains of §3:

    column sums   s_{c,l}(δ) = Σ_i δ_i U[r_i, mode(c, l)]              (L per slot, updated per step)
    slot matrices G_c(δ) = Σ_l s_{c,l} M̃_{c,l}                          (2 × 2 complex)
    p_c = G_{c−1} … G_0 s0,   q_cᵀ = f† G_{K−1} … G_{c+1},   acc_c += sign(δ) p_c q_cᵀ

The host reduces the segments per term, forms Tr(M̃_{c,l} K_c), the candidate amplitudes and
w = Σ_terms |amp|² (torch) and draws. Term decompositions come from `batched_gap_terms`.

Two implementations:

* `"scan"` (first version): [SEG, K_PAD] tensors, prefixes and suffixes by two
  `tl.associative_scan`s with 2 × 2 matrix combines on shifted copies of the column sums.
  ≈ 108 complex multiplies per column-step at K_PAD = 32 and 8 row loads per column.
* `"seq"` (default): generated per (K, L, B) — see the module docstring. The K columns are split
  into B blocks of CB = ⌈K/B⌉ columns; a block is one thread that holds its columns as [SEG, B]
  tensors with a static in-block index, so the chains inside a block are sequential 2 × 2
  products; the B threads of a lane are joined by two scans over the block products with pair
  combines that yield the exclusive prefix/suffix directly (no shifted copies). 28 complex
  multiplies per column-step + 2·log2(B)·16 per lane-step; exact K (padding columns get
  G = I). A program is SEG segments of one term, so the term data are program-uniform scalar
  loads — this keeps the GPU loads uniform and, as importantly, keeps Triton's coalescing pass
  (O(function size) per tensor memory op) from taking minutes per kernel. Configs (SEG, B,
  matrix mode) per K_PAD tier in `_CFG["seq"]`, tuned with `bench/bench_emitter_tune.py`;
  `bench/compile_check_emitter.py` reports registers/spills offline.

Still to do: float32 accumulation for the early steps, and the host side (term expansion
einsums, per-step torch post-processing) which dominates below n ≈ 20. *(Corrected 2026-09-16:
this list previously began "the 2020 repeated-rows collapse (mixed-radix Gray walk from
`_cofactors_multi_kernel`)". It is implemented — `emitter_multiplicity_tables` builds the
2020 tables and the generated `seq` kernel walks them, see its "mixed-radix Gray walk state
(Clifford & Clifford 2020 collapse of repeated rows)". The measured emitter per-photon factor
of 2.10 at m = 2n is therefore ρ(2) ≈ 1.81 from the collapse times ≈ 1.16 per photon of R_k
growth, not a missing collapse.)*

**Launch knobs swept and exhausted (2026-09-16, results/20260916_155718).** `num_warps` is
derived here as `SEG·B/32`, so the register/occupancy trade the `(SEG, B)` table reaches is only
the one those two imply. On the *Fock* kernel that trade, swept independently, was worth 1.32× at
the widest tier and 1.9× between the right and the wrong register cap. `emitter_triton.set_launch(maxnreg=, num_stages=)` exposes
it here and `bench/bench_emitter_tune.py --maxnreg --stages` sweeps it. **Measured: no gain at
all** — 216 configurations at K = 10/14/18, 108 at K = 20/22, and the winner carries
`maxnreg=default` at every K, gain 1.000×.

`bench/compile_check_emitter.py` says why, and it is the opposite of the Fock situation: Triton
already allocates **254–255 registers** at every K, *with spills* (700 B at K = 16, 1324 B at
K = 22). There is no register headroom to trade for occupancy — capping lower only adds spills.
The Fock CUDA kernel sat at 236 registers with none, which is what made capping to 168 worth
1.32× there.

**What the sweep did find: the `(SEG, B, mat_mode)` choice matters far more here than on the Fock
side, and `_CFG` cannot track it.** The spread between the best and worst configuration at one K
grows 1.24× (K = 10) → 4.8× (18) → 5.7× (22). `_CFG["seq"]` is keyed by `K_PAD` tier, so K = 18,
20 and 22 all take the tier-32 entry `(4, 8)`, and `SEQ_MAT_MODE` is a process-wide global:

| K | tier | `_CFG` now | ms | best of the sweep | ms | gain |
|---|---|---|---|---|---|---|
| 10 | 16 | SEG 8, B 4, scalar | 1.80 | SEG 16, B 4, scalar | 1.70 | 1.06× |
| 14 | 16 | SEG 8, B 4, scalar | 2.55 | SEG 16, B 4, scalar | 2.54 | 1.00× |
| 18 | 32 | SEG 4, B 8, scalar | 11.11 | **SEG 8, B 8, scalar** | 8.60 | **1.29×** |
| 20 | 32 | SEG 4, B 8, scalar | 12.85 | SEG 4, B 8, **hoist** | 12.63 | 1.02× |
| 22 | 32 | SEG 4, B 8, scalar | 44.77 | SEG 4, B 8, **hoist** | 44.39 | 1.01× |

*(This table is the 2026-09-16 64-stream sweep, run before the `--blocks` grid was fixed: it did
not contain B = 4 at `K_PAD = 32`, so its K = 18 row understates the gap — the true K = 18
optimum is B = 4, worth 1.25× over B = 8. See the correction below. The K = 20 and 22 rows are
unaffected, B = 8 being the optimum there.)*

So the tier granularity costs 1.29× at K = 18, and the optimal `mat_mode` flips from `scalar` to
`hoist` between K = 18 and K = 20 — which a global cannot express. But summed over the five sizes
the retune is only **1.05×**, because at the K that actually carry the cost the current entry is
already within 1 % of the best of 108 configurations. **The Triton kernel is at its best
configuration where it matters.**

*Caveat on the instrument:* the `G units/s` column of `bench_emitter_tune.py` normalises by the
2018 count `2^{k-2}`, while the kernel walks the collapsed 2020 count `N < 2^{k-2}`. It is
therefore comparable **across configurations at one K** (which is what the sweep needs) but **not
across K** — the apparent rise from 0.36 to 63 G units/s between K = 10 and 22 is mostly the
collapse gain, not throughput. Comparing across K needs the true `N` from
`emitter_multiplicity_tables`.

**Applied 2026-09-16, withdrawn, then re-applied once the withdrawal's premise was shown to be
wrong.** `_CFG["seq"]` entries may now carry a third element, the `mat_mode` (per tier instead of
the process-wide `SEQ_MAT_MODE`), and a second table `_CFG_BY_K`, tried before the tier, holds
exact-K overrides; `set_config_k()` overrides it per process. Shipped: `_CFG["seq"]` unchanged,
`_CFG_BY_K["seq"] = {18: (8, 4)}`. The path there is worth recording, because the intermediate
state of this document asserted something the next run disproved.

A confirmation sweep (20260916_191446) disagreed with the sweep it was meant to confirm
(20260916_155718) at K = 18: B = 8 in the first, B = 4 in the second, 1.24× apart. The two ran at
different stream counts, so this document concluded the optimum moved with the stream count and
shipped `_CFG_BY_K` empty on that basis. **That conclusion was wrong, and the cause was the
harness, not the kernel.** The default `--blocks` grid was
`sorted({max(1, min(32, K_PAD // d)) for d in (4, 2, 1)})`, which at `K_PAD = 32` is
`{8, 16, 32}` — **B = 4 was never tried**. The first sweep did not choose B = 8 over B = 4; it
never saw B = 4. The grid now runs `for d in (8, 4, 2, 1)`, giving `[4, 8, 16, 32]` at
`K_PAD = 32`, and with B = 4 present the two stream counts agree.

What the corrected sweeps (20260916_200342) establish:

- **`mat_mode` is noise.** `scalar` and `hoist` land adjacent in the ranking at every K, within
  0.2–2.7 %. The `hoist` default an earlier sweep suggested stays withdrawn.
- **`B` is the dominant knob.** At K = 18 (32 streams) SEG 8 / B 4 is 3.75 ms against 4.68 ms for
  B = 8 — 1.25× — and the worst configuration in the grid is 2.0× off.
- **The tier-32 collision is real.** K = 18 wants B = 4, K = 22 wants B = 8 (44.7 ms against
  54.6 ms for B = 4, 1.22×). One `K_PAD` tier cannot serve both, which is what `_CFG_BY_K` is
  for; hence the single entry `{18: (8, 4)}`.
- **It does not move with the stream count.** At K = 22 the winner is SEG 4 / B 8 / scalar at 8,
  32, 128 and 512 streams alike — a 64× range, one ranking.

*The roofline question, answered.* The stream sweep at fixed K = 22, normalised by the true
collapsed work count `N` from `emitter_multiplicity_tables` (not the 2018 `2^{k-2}`; the collapse
gain is 2.7–3.0× here and would otherwise be read as throughput):

| streams | terms | best ms | G units/s | best config |
|---|---|---|---|---|
| 8 | 32 | 14.85 | 18.5 | SEG 4, B 8, scalar |
| 32 | 122 | 47.17 | 20.1 | SEG 4, B 8, scalar |
| 128 | 500 | 185.86 | 21.5 | SEG 4, B 8, scalar |
| 512 | 1962 | 752.20 | 21.0 | SEG 4, B 8, scalar |

Flat to ±8 % across 64× in occupancy: **the kernel is saturated from 8 streams onward, i.e. it is
at its own roofline.** The earlier reading that it was "still climbing at K = 22" compared across
K, where the normaliser is not comparable; at fixed K it is flat. There is no batch size at which
the current kernel goes faster.

The register knobs confirm the same from the other side: capping at 128 costs 4.7× (3.5 against
16.6 G units/s at K = 20) and at 168 costs 1.5×. Triton is at 254–255 registers with spills and
every cap only adds more.

*The frontier steps, swept (2026-09-27, run 20260927_080529).*
The table above stops at K = 22 while the frontier's last steps are K = 25–30, so the tier-32 entry was
swept at k = 26, 28, 30 (8 streams, 24 configurations each): SEG 4 / B 8 wins at every k, the tier entry
is within 0.2–2.8 % of the best (mat_mode noise again), B = 4 loses by 1.06–1.95×, the worst configuration
by 2.6–3.5×. Nothing to change; gain 1.00×.

*Consequence.* Neither the launch knobs, nor the config table, nor more parallelism leave anything
at the expensive steps. Together with §7 (R_k is already minimal where the cost is), the remaining
factor on the emitter sampler is structural: a CUDA thread-per-lane port, as on the Fock side (§8). The
register evidence supports it — Triton is pinned at 255 registers and spilling 1.3 kB per thread
at K = 22, which is exactly the regime where hand-managed per-lane register state and
`__shfl_xor_sync` combines won 2.4–2.7× for the Fock kernel.

## 7. Caterpillar states: gap ranks, the term count is NOT the rank of ρ_T; segment-factorised terms

*Added 2026-09-12 (`clifford/emitter_caterpillar.py`).* The roadmap hoped for a "cheaper
purification" exploiting the projector structure of the caterpillar operators
`M^(j)_l = |l⟩⟨l| G_j`. The structure gives:

**Markov amplitudes.** `c_x = ⟨f|x_n⟩ ∏_j ⟨x_j|G_j|x_{j−1}⟩ ⟨x_1|G_1|s0⟩` — the bond index of the
MPS *is* the emission label. Tracing a run of emissions a+1..b−1 (a gap) leaves the classical
transition weight `w[t, s] = [|G_{b−1}|² ··· |G_{a+2}|²]_{t,s}` (elementwise |·|²) between the
label s entering the gap and the label t leaving it: the gap channel is measure-and-prepare
in the label basis, with Kraus operators `√w[t,s] |t⟩⟨s| G_{a+1}` of rank one.

**Consequence 1 — each gap's rank is exact, the product over gaps is not** *(corrected
2026-09-12 after a review of the derivation; the previous text claimed "R_k is the true rank and cannot
be reduced", which is false).* The Kraus rank of one gap channel is the number of non-zero
`w[t, s]` (4 for generic gates, 2 for GHZ-type gates with zeros), and the SVD purification of §2
finds exactly that rank per gap. But `R_k = r_0 · r_f · ∏ r_gap` is only an upper bound on
`rank(ρ_T)`: the per-gap reduction is minimal gap by gap, not globally, because correlations
between gaps are not exploited. Measured by building `ρ_T` explicitly and comparing (random
caterpillar gates, n = 10, |T| = 5): rank 16 vs R_k = 32, 16 vs 16, 32 vs 32, 16 vs 16, 8 vs 16,
16 vs 16 — the count is 2× the rank on about half the steps, and the surplus terms are linearly
dependent on the others (SVD of the stacked ψ_e gives exactly rank(ρ_T) independent vectors)
while no single chain has zero norm, so the pruning cannot see them. Averaged over the steps of
a run the term count exceeds the rank by 1.8× (random gates, cost-weighted) to 2.3× (linear
cluster, where the surplus is systematic: 2 per inner gap, single steps up to 16×), n = 12.

**Measured 2026-09-16 (`bench/emitter_gram.py`, results/20260916_153618): the headroom is
≈ 1.03–1.11×, not ≈ 2×.** The Gram matrix was built for every step of 4–12 photon orders at
n = 8/12/16/20 (linear, random and GHZ families) and its rank compared with `R_k`. The
per-step surplus that review found is confirmed and is if anything larger than it saw — up to 16× at
n = 20, k = 6, and the eigenvalue cut is exact (smallest kept / largest dropped ≈ 1e15, so
`rank` is an integer, not a tolerance-dependent tail). **But it sits entirely at small k, and
the cost weight is `2^{k-2} k`.** At n = 20: steps k = 1..8 carry 1.1 % of the cost and have
R/rank = 2.4–10×; steps k = 15..20 carry 74 % of the cost and have **R/rank = 1.00 exactly**.
The decomposition is already optimal precisely where the cost is, and becomes so from
k ≈ n − 3 to n − 5 in every family and every photon order measured.

| | n = 8 | n = 12 | n = 16 | n = 20 |
|---|---|---|---|---|
| cost-weighted `Σ R_k c_k / Σ rank_k c_k` | **1.021×** | **1.029×** | **1.065×** | **1.109×** |
| term-count weighted `Σ R_k / Σ rank_k` | 1.25× | 1.51× | 1.92× | 3.06× |
| unweighted mean of R/rank over steps | 1.29× | 1.50× | 1.96× | 3.19× |
| worst single step | 4× | 8× | 8× | 16× |

The second and third rows are where the review's "1.8× (random) to 2.3× (linear cluster)" came from:
those are the ratio averaged over steps, or weighted by term count, not weighted by step cost.
Weighting by `R_k` alone (which is the lane count, i.e. the memory) over-weights the mid-k
steps, where `R_k` peaks at k ≈ n/2.5 while `2^{k-2} k` is still negligible. The claim below
therefore does not follow from the review's numbers, and is withdrawn.

The trend with n is real but slow — 1.02 → 1.03 → 1.07 → 1.11 from n = 8 to 20, as the
peak-`R_k` region drifts towards larger k — so even at n = 30 it would not approach 2×. GHZ
gates give R/rank = 1.00 at every step (a control: the zeros collapse the gap ranks already).

*Consequence, as written before that measurement and now superseded.* The per-step cost is
linear in `R_k`, so there is a ≈ 2× headroom on the whole
emitter sampler, all backends alike (torch, segments, Triton) — larger than the Triton
segment-lanes port rejected below at ≤ 1.5×. It is not free: the minimal decomposition mixes
chains, and a term `Σ_e c_e ψ_e` is an MPS of bond χ·R, dearer per δ than the terms it
replaces, so naively orthogonalising the ψ_e loses. What to try is a left-to-right sweep instead
of independent per-gap SVDs: compose the chain emission by emission and reduce the accumulated
(prefix operator ⊗ Kraus) set against the remaining chain at every gap, so that a gap's rank is
cut by what the earlier gaps already fix while each term stays one chain. Whether that recovers
the factor 2 is an experiment; the Gram matrix `⟨ψ_e|ψ_e'⟩` is a χ²×χ² transfer-matrix product
of length k, `O(R² k χ⁴)` per step, negligible against `R_k 2^k k`, and measures the headroom
exactly — that measurement comes before any further kernel work.

**Consequence 2 — terms factorise over segments.** With the rank-one gap operators, a pure
term is a *product* over the segments of consecutive kept slots,
`ψ_e(x_T) = ∏_seg ⟨s_seg| G_{q+1} P_{x_q} G_q ··· P_{x_p} G_p |t_seg⟩`, the term index being the
labels at the gap boundaries, `e = (t_0; (s_i, t_i) per inner gap; s_f)`, with weight
`p_e = w_0[t_0] ∏ w_i[t_i, s_i] w_f[s_f]`. In the Clifford step every term therefore reads
matrix elements of the *same* per-segment 2×2 chains:

    cof_e[c, l] = 2^{−(k−2)} Σ_δ sign(δ) [∏_{seg' ≠ seg(c)} C_seg'(δ)[s_seg', t_seg']] · Q_c[s, l](δ) (G_{t_c} P_c)(δ)[l, t],

with `P_c`, `Q_c` the within-segment prefix / suffix chains and `C_seg = Q_c G_c P_c` the whole
segment chain. Per (term, column, δ) the work drops from ~28 complex multiply-adds
(`G` construction, forward and backward chains, `K_c` accumulation, all per term in
`sample_torch` / the Triton lanes) to 2 plus a gather; the chains are computed once per stream.

**Measured (torch, CPU, 2 threads, `backend="segments"` vs `backend="torch"`, 64 shots):**
n = 12: 37 vs 40 ms per sample (1.1×, R̄ = 13 — the grouping overhead dominates);
n = 16: 283 vs 697 ms (2.5×, R̄ = 34); n = 20: 4.4 s per sample (R̄ = 79; `sample_torch` runs out
of memory in the sandbox at that size). Same weights as `matrix_cofactor_weights` to 1e-15,
same R_k, TVD within the statistical floor against the strong simulation
(tests/clifford/test_emitter_reference.py).

**What this means for the kernel — corrected after working out the Triton design
(2026-09-12).** The "~10× fewer flops per δ" above counts only the per-term part; three
things the GPU kernel cannot avoid make the realistic gain much smaller:

1. *The per-term accumulators are inherent.* `cof_e[c, l]` is a sum over δ of a product of
   δ-dependent factors that does not factorise over segments, so every (term, column, label)
   needs its own accumulator through the Gray walk: `R_k · K · L` complex accumulators per
   stream, the same count as today's `R_k · K · 4` (2×2 per column) up to the factor 2. In the
   current kernel they are spread over the term lanes (`R_k` lanes × 4K doubles); sharing the
   chains means holding `EB` terms in one lane, i.e. `EB` × 4K doubles of accumulators per
   thread on top of the chain state — the register budget (the seq kernel already splits the
   K columns into B blocks so that a thread holds 2–3 columns) caps `EB` at ~8 with B = 4.
2. *The shared chain costs more per column than it looks.* With F_c, the segment products
   C_σ and the boundary resets it is ~48 complex multiplies per column per δ (`G_c`, `G P_c`,
   `P_{c+1}`, `Q_c`, `C_σ`, `F_c`), against 28 per (term, column, δ) today.
3. *Term labels are runtime data.* Reading `F_c[l, s_e, t_e]` and `C_σ[s_e, t_e]` for a term
   means selecting among 4 entries with `tl.where` on per-term integer tables: ~8
   complex-multiply equivalents per (term, column, δ) instead of 2.

Per (term, column, δ) the kernel work is therefore `48 / EB + 8` against `28`: 1.4× at
EB = 4, 2.0× at EB = 8, 2.5× at EB = 16 (beyond the register budget), and the kernel is
~60 % of a step (term expansion and the candidate product are the rest), so ≤ 1.5× end to
end. The design, should it be wanted: lanes = (term block of EB terms of one stream, Gray
segment); per-(stream, column) tables `is_start`, `is_end`, `B_in` (s0 or identity), `B_out`
(G_{t_q+1}, f†, or identity), bare gates `G_{t_c}`; per-(block, term, column) label tables;
forward chain `P_c = where(is_start, B_in, G_{c-1} P_{c-1})`, which with B > 1 blocks is a
segmented scan over affine elements (M, R) ∘ (M', R') = (M M', M R' + R) so the reset is
associative; backward chain likewise; running prefix/suffix products of the segment chains
per term for `O_{σ,e}`; accumulate `sign · O · F_c[l, s_e, t_e]`. Not implemented: the
expected factor is not worth the kernel's complexity — the torch `backend="segments"` path
is where the idea pays (no register cap, 2.5× at n = 16 and growing with R_k), and the
next real factor on the GPU sampler is elsewhere (fewer streams per term expansion, or the
candidate product).

## 8. CUDA thread-per-lane kernel (`clifford/cuda_emitter.py`)

*Added 2026-09-27. Implemented and
verified on the CPU, then measured (run 20260927_110150): **1.49× over the seq kernel at
k = 28, 1.04–1.24× at k = 16–24** — the expected-factor paragraph below is the pre-measurement estimate,
kept as written; the measured paragraph at the end is what stands.*

*Why.* §6 ends with the structural reading: at the frontier steps the seq kernel is at its own
roofline with Triton pinned at 255 registers and spilling, the tier and launch knobs exhausted (run
20260927_080529: nothing to gain). On the Fock side the same situation was resolved by the
CUDA thread-per-lane kernel `cuda_cofactors.cof_multi` (2.4–2.7× at K ≥ 26). The emitter kernel is the same Gray walk with a 2 × 2 payload, so the port is the one lever
left on the emitter row (n* = 30.6, ρ = 1.98: Δn* = ln X / ln 1.98 = +1.0 per 2×).

*What is shared with the Fock kernel — one template family.* `emit_multi<KP, TPL, NL, GS>` reuses,
line for line, the Fock kernel's odometer (byte arrays `d/v/R/M/pp`, mixed-radix reflected Gray code
over the 2020 collapse, weight update by the binomial ratio), its lane → (stream, segment) bookkeeping
(`lane2term`, `lstart`, `nterms`, segment length L), the host planning (`cofactors_triton.plan_multi`
over the collapsed counts N per term) and the host-gathered contiguous input block Bs[s, ν, c, l] =
U[drow[s, ν], mode(c, l)]. The tables are the Fock ones (`cofactors_triton.multiplicity_tables`:
every distinct row a digit of radix multiplicity + 1, one odd group last and halved) rather than the
seq kernel's (binary digits first, explicit high digits): same term count, same scale 2^−K1 (×2 when
halved), only the enumeration order differs, so per-lane partials are not comparable with the seq
kernel's but per-term sums are (to 1e−15, tests). The tier ladder runs in steps of 4 (8 … 40) with a
per-tier (threads per lane TPL, matrix storage GS, `--maxrregcount`) table keyed by compute
capability, an untuned default, `warmup()` compiling one module per register cap, and `set_config`
for the sweeps — the `cuda_cofactors` host structure.

*What differs — the payload.* A lane is TPL consecutive threads; thread h owns the KH = ⌈KP/TPL⌉
columns c0 = h·KH … c0 + KH − 1 (columns ≥ K idle with G = I, accesses ≥ KH removed at compile time,
so any (KP, TPL) is legal). Per thread and column the state is the NL column sums (2 NL doubles), the
stored forward vector p_c (4) and the 2 × 2 accumulator (8): 16 KH doubles at NL = 2 — four times
the Fock kernel's 4 KH, which is why the default keeps KH ≤ 4 (TPL 2/4/8 for tiers 8/12–16/20–40)
where the Fock table has KH = 8–16. Per Gray step:

```
# emit_lane<KP, TPL, NL, GS>  — one thread, columns c0 .. c0+KH-1 of lane g
init: odometer from base = li·L (as cof_multi); w = Π ±C(M_ν, v_ν)
      s[j][l] = Σ_ν (M_ν − 2 v_ν) Bs[s, ν, c0+j, l]         # column sums at the segment start
      acc[j] = 0
for t in 0 .. Lp-1:
    if t > 0: odometer step -> (ν, dirn); w *= ratio; s[j][l] += (−2 dirn) Bs[s, ν, c0+j, l]
    G(j) := Σ_l s[j][l] M̃[λ, c0+j, l]        (identity if c0+j ≥ K; GS = 1 keeps the KH of them)
    p = s0 ; qᵀ = f†
    if TPL > 1:                                # pass 1: block product and its exclusive prefix / suffix
        H = G(KH−1) ⋯ G(0)
        P = H ; S = H
        for o = 1, 2, 4, … < TPL:              # Kogge–Stone over the lane's threads (shfl_up / shfl_down)
            P = P · up(P, o)   if h ≥ o          # P_h = H_h ⋯ H_0
            S = down(S, o) · S if h + o < TPL    # S_h = H_{TPL−1} ⋯ H_h
        E = up(P, 1) (I at h = 0) ; F = down(S, 1) (I at h = TPL−1)
        p = E p ; qᵀ = qᵀ F
    for j = 0 .. KH−1:  pj[j] = p ; p = G(j) p                # pass 2: forward chain, p_c stored
    for j = KH−1 .. 0:  acc[j] += w · pj[j] qᵀ ; qᵀ = qᵀ G(j)  # pass 3: backward chain + accumulate
out[g, c0+j, a, b] = acc[j][a][b]
```

The exchange is 2 (log₂ TPL + 1) shuffles of 8 doubles and 2 log₂ TPL matrix products per lane-step
(TPL = 4: 48 shuffled doubles, 32 complex multiplies, against ~4 × 28 for the columns); the seq
kernel's two `tl.associative_scan`s over [SEG, B] tensors do the same join with Triton's layout
conversions in between. Per (column, step) the arithmetic is: sums 2 NL × 2 FMA; G 4 NL complex
multiplies (computed once with GS = 1, up to three times with GS = 0 — the register/FMA knob of the
tier table); H 8; forward 4; backward 4; accumulate 4: 28 complex multiplies with GS = 1, the seq
kernel's count. No zero handling (no all-but-one products). Row loads are warp-uniform broadcasts
(a warp holds 32/TPL consecutive lanes of one term, the same stream row), the term matrices are
re-read from L1 every step (the seq kernel's `scalar` mode, which the sweeps found equal to hoisting).

*Where the factor is expected from (estimate, NOT measured).* Nothing in the arithmetic is cheaper
than the seq kernel's; the candidates are the ones that paid on the Fock side: (a) exact register
budgeting per tier (`--maxrregcount`, KH per thread chosen by the table) where Triton sits at 255
with spills; (b) the exchange as 24–48 shuffles instead of two associative scans with layout
conversions; (c) contiguous gathered rows (`Bs`) instead of 4 NL scattered `U` loads per column-step;
(d) a byte odometer instead of int32 [SEG] digit tensors and H_PAD `tl.where` chains per step; (e) no
column padding to CB · B. Fock got 2.4–2.7× from (a)+(b)+(d)+(e) with a 4× lighter payload; the
emitter's heavier payload makes (a) count more and (d) less. Expected X = 1.5–2.5 → Δn* = +0.6–1.3
on the emitter row (K ≤ 31 at n* = 30.6: tiers 32 and 28 are the ones that matter). If the sweep
(job 9) gives X < 1.2 at K = 26–30 the item is closed negative like #2 and the kernel is kept as the
`kernel="cuda"` option only.

*Verification (done, CPU).* The kernel source is plain C++ except for the exchange primitive, so the
same file compiles under NVRTC (`__shfl_up_sync` / `__shfl_down_sync` with width TPL) and under the
host compiler as `emit_emul` (C++20: the TPL threads of a lane are `std::thread`s joined by a
`std::barrier`, so the Kogge–Stone code above is what runs). `tests/clifford/test_cuda_emitter.py`
checks the emulation against `matrix_cofactor_weights` at every step k = 1 … n (K1 = 0 included),
random and repeated rows, TPL 1/2/4/8, GS 0/1, one lane per term and many segments, one and two
labels, K up to 13 (tiers 8–16 with idle padding columns), to 1e−13 relative, and that the sampler
draws the same samples as the seq kernel from the same generator. `tests/clifford/check_cuda_emitter.py`
(job 8) repeats this on the GPU at every tier to K = 30 and times the frontier steps against the seq
kernel; `bench/bench_cuda_emitter.py` (job 9) sweeps (TPL, GS, maxrregcount) per tier and prints the
`_CFG_BY_ARCH` entry.

*Measured (run 20260927_110150; results/keep/cuda_emitter_sweep_k16_28_20260927_h100.csv).*
Correctness at every tier to K = 28 (two summation orders agree to 4e−16 … 9e−12, growing ~4× per tier with the
collapsed sum's cancellation). Speed, step k of an n = k + 2 instance at 8 streams, best configuration against the
seq kernel: **1.09× (k = 16), 1.24× (20), 1.04× (24), 1.49× (28)**. The sweep overturned two of the sizing
assumptions above: GS = 1 (matrices kept) wins by 1.4–1.8× — the kernel is FMA-bound before it is register-bound —
and TPL = 4 (KH = 5–7) beats TPL = 8 (KH = 3–4) by 1.6–1.8×, so the exchange is dearer than the registers it frees;
every `--maxrregcount` cap loses. The CUDA rate rises with K (24.5 → 27.0 G units/s, k = 24 → 28) where the seq
kernel's falls (23.6 → 18.1): the gain is at the frontier steps, K ≥ 26. Tier table: `_CFG_BY_ARCH[(9, 0)]`
(tier 32 measured by job 10: TPL 4, GS 1, 1.22× at k = 30).

*On the map (runs 20260927_125541 … _205524, LOG 2026-09-27 last entry): **no shift.*** The map with `--kernel cuda`
(same seeds and budget as the canonical run) gives n\* = 30.6 / 30.3 / 30.1 / 29.6 / 29.3 / 29.1 against
30.6 / 30.2 / 29.9 / 29.5 / 29.1 / 28.9 — +0.0 to +0.2. At 128 shots the kernel is 99 % of the sample and the
CUDA kernel is 1.42–1.46× at k = 25–28, but 0.25–0.85× at k = 9–16: the steps with thousands of terms per stream
and ONE lane per term, where the seq kernel runs at 25–31 G column-updates/s (against its 10–18 G at the frontier
steps) and the CUDA kernel at a flat 15–16 G whatever TPL (2 … 16 measured; the "distinct matrix rows per warp"
hypothesis is refuted). The middle of the chain carries half the sample's cost, so the frontier-step factor is
diluted to 1.0× overall. `kernel="auto"` (`emitter_triton.AUTO_LANES` = 1.5 lanes per term) sends the frontier
steps to the CUDA kernel and the rest to the seq kernel: **1.16× end to end at n = 30, measured (run 20261002_003041),
and the map with `--kernel auto` is +0.3 at every ratio: 30.9 / 30.5 / 30.2 / 29.8 / 29.5 / 29.2** against the
quoted 30.6 / 30.2 / 29.9 / 29.5 / 29.1 / 28.9 (results/keep/frontier_auto_emitter_r*.csv). Not the quoted
row (Δn\* < 1, and every other emitter number is anchored on the seq row); `seq` stays the default, `auto`
is the shipped faster option. Measured +0.3 against the +0.6 to +1.3 estimated. The open question —
why the seq kernel is 1.6–2× faster per column-update in the one-lane-per-term regime than at the frontier steps —
is the one lever left on this row, and it is a Triton-kernel question, not a CUDA one.

*Not done / follow-ups.* (i) The trace variant: downstream only Tr(M̃_{c,l} K_c) is used, and
Tr(M̃ p qᵀ) = qᵀ M̃ p, so the kernel could accumulate the NL traces per column (2 NL doubles) instead
of the 2 × 2 K_c (8), 25 % less state per column for +8 complex multiplies per column-step, and hand
`tr` to the host directly — job 9 says the kernel is FMA-bound, not register-bound, so this is NOT
the next step; fewer multiplies per column-step is. (ii) The
single-stream (`L <= 0`) path of the Fock kernel is not ported (the emitter frontier runs 8+ streams).
(iii) `warmup()` compiles both NL modules per cap (~2 s per cap).

## 9. Arithmetic precision of the step weights (2026-10-04)

The step weights are computed from the collapsed Glynn sum over 2^(K−2) signed products of 2×2 chains (§3, §7); finite
precision enters there and nowhere else (the draw is `torch.multinomial` on float64 weights). Because the CUDA kernel's
C++ (§8) compiles unchanged on the CPU, the same instance can be evaluated in x87 long double (64-bit mantissa) and
the fp64 error measured rather than bounded (`bench/emitter_precision.py --dump` on the GPU, `--reference` on the CPU;
run 20261004_223624, results/keep/emitter_precision_*_20261004_*). Relative error of the step cofactors K_c, max over
columns and labels, two terms per size, n = K + 2, m = 2n:

| K placed photons | 26 | 28 | 30 |
|---|---|---|---|
| seq kernel, production segments (= CUDA kernel) | 1.1–1.3·10⁻¹³ | 6.3–7.7·10⁻¹³ | 2.4–2.5·10⁻¹² |
| one repeated row | 1.2–1.8·10⁻¹³ | 7.4–9.9·10⁻¹³ | 1.6–3.1·10⁻¹² |
| seq kernel, small segments (2¹⁵ lanes) | 2.7–4.5·10⁻¹³ | 1.4–2.8·10⁻¹² | 7.2–8.3·10⁻¹² |
| GPU disagreement seq vs cuda | 2–4·10⁻¹⁴ | 3·10⁻¹⁴ | 4–5·10⁻¹⁴ |
| GPU disagreement large vs small segments | 3.9·10⁻¹³ | 2.0·10⁻¹² | 7.6·10⁻¹² |

The error grows ×2.3 per placed photon (the cancellation of the collapsed sum, as the Fock kernel's: docs/ALGORITHMS.md
§6), so the 10⁻³ criterion of Supplement C is met in fp64 to K ≈ 53 and the quoted row (n\* = 30.9) carries 2.5·10⁻¹²,
10⁹ below it. Two lessons for measuring this elsewhere: (i) segmentation matters — more lanes per term means more
partial sums rounded before they cancel, and the production setting (few lanes per term at the frontier steps) is the
more accurate one; (ii) the seq and CUDA kernels agree 50× better than either is accurate, because they walk the same
collapsed Gray order: a disagreement bounds the error only between evaluations that round differently where the error
is made (two segmentations do; two kernels sharing the order do not).
