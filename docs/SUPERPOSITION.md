# Superposition boson sampling: sources with an indefinite photon number

The input type `states.ProductSuperposition`, the graded SLOS recurrence
`slos.ProductSLOSKernel`, the single amplitude `slos.glynn`, and why the only exact sampler is
the one strong simulation gives for free. Reference: J. J. Renema, *Simulability of imperfect Gaussian and superposition boson
sampling*, arXiv:1911.10112 (PRA 101, 063840 (2020)).

## 1. The model, and what the paper actually proves

`n` sources, each in a coherent vacuum/one-photon superposition `cos α |0⟩ + sin α |1⟩`, feeding
`n` of the `m` interferometer modes. Post-selected on `k` detected photons the input is the
**Dicke state** on the source modes: the uniform coherent sum of the `C(n, k)` ways of choosing
which sources fired.

Renema's paper is about *simulability under imperfection*, not about a new advantage claim:

| regime | verdict |
| --- | --- |
| SBS at partial distinguishability `x < 1` | efficiently simulable — explicit algorithm, order-`k` truncation of the interference expansion |
| SBS at `x = 1` | resists that method; "the first evidence for the computational hardness of this problem", explicitly **not** a proof |
| GBS with distinguishability | reduced to an open problem in permanent theory (the classical-interference matrix acquires negative entries, so Jerrum–Sinclair–Vigoda does not apply) |

The truncation error is `TVD ≤ sqrt(x^{2(k+1)} / (1 − x²))` averaged over Haar unitaries —
independent of the photon number, the truncation order depending on `x` alone. That algorithm is
**not implemented here**, and it is not needed for an exact engine: the x-model is exactly the
mixture over which photons interfere (`docs/DISTINGUISHABILITY.md` §1), so the samplers take
`indist=x` exactly, at the cost of the interfering subset. Renema's expansion is the small-`x`
truncation of that same mixture (§4 there). Both engines take `indist=x`: `ProductSLOSKernel.probs` as the mixture over the sources in
the common mode, `clifford.product_sample` by drawing that mixture per shot — a lone source's photon
factorises out of the Dicke coherence and is routed classically (2026-09-21, checked against the
strong side within shot noise, with and without loss).

Physically the input is not exotic: a quantum dot driven at pulse area θ emits
`cos(θ/2)|0⟩ + sin(θ/2)|1⟩` coherently. This is the **coherent** counterpart of the incoherent
per-source loss already in `imperfections.py`, and the counterpart of scattershot, which is a mixture.

## 2. The input type, and why α = π/4 is the reference point

`ProductSuperposition` stores `amps` [sources, d] and one input mode per source; `d = 2` is SBS,
`d > 2` lets a source emit up to `d − 1` photons (weak-coherent or truncated-thermal sources fit
the same recurrence and the same oracle). `ProductSuperposition.sbs(sources, m, alpha)` builds
the SBS state, α = π/4 by default.

Two facts fix the reference point:

* **The photon number is exactly binomial.** A passive unitary conserves photon number and the
  sources are independent, so the detected total is `Binomial(sources, sin²α)` whatever `U` is —
  `photon_number_probs()`, checked against the engine's per-layer masses in the tests. Sampling
  therefore factorises: draw `k`, then sample a Dicke input. There is no post-selection loss in
  the simulation.
* **α = π/4 is the hardest case at fixed source count.** The post-selected Dicke bond is
  `min(k, n − k) + 1`, so the mean `k = n/2` is exactly where it saturates at its maximum, and
  `C(n, n/2)` is the largest component count. At fixed *detected* `k` it is not the strict
  maximum — the bond stays `k + 1` for every `n ≥ 2k` while the linear factor in the amplitude
  cost keeps growing — so the claim is always stated at fixed source count.
* **α drops out once you condition on k — which is why conditioning is the wrong thing to do.**
  For uniform α the post-selected state is the *uniform* Dicke state whatever α is; α only sets
  `P(k)`. A frontier measured at fixed k therefore prices a k-photon sample, not a device, and it
  quietly discards what the modality actually is. A device with `n` sources emits
  `k ~ Binomial(n, ½)` — σ = √(n/4), and the chance of exactly the mean is `~1/√(πn/2)`, about 11 %
  at n = 40 — so post-selection throws away nine shots in ten and hides the source count behind a
  photon count.

  **The frontier is therefore measured against the SOURCE count**, which is also what `n` means for
  the Fock engine (n single-photon sources) and the emitter engine (n emission events), with
  `m = round(c·n)` modes per source in all three. That is the only comparison with a hardware
  meaning: build n sources, ask how hard the output is to simulate. SBS's n sources yield n/2
  photons on average against Fock's n, and that weakness stays in the number rather than being
  normalised away. Measured on two CPU cores at m = 2n: **n\* = 12.5 sources** unconditioned,
  against 15.5 for the post-selected diagnostic (`--engine sbsfixed`), the gap being the upper tail
  of the binomial.

`p = sin²α` also makes the photon-number bookkeeping **identical to input-side loss with
η = p**: `E[ρ^k] = (1 − p + pρ)^n`, the same expression as the lossy frontier's. Input-side loss
and partial excitation give the same photon-number statistics; the difference is that the lossy
sampler knows which sources fired while this one faces a coherent superposition of all of them.
So the incoherent twin of SBS at α = π/4 is **Fock at η = 0.5**, measured on
2026-09-18: `n* = 64.0` injected, `32.0` detected. What an SBS engine falls short of that line is
NOT all price of coherence, and the coherence scan of 2026-09-20 says how little of it is: the
superposition engine on the incoherent twin itself (α = π/2 behind η = ½, the same state) reaches
22.4 against the Fock engine's 64.4 (same day), and at full coherence 20.2 — 42.0 sources of engine, 2.2 of
coherence (§5, docs/LOSS.md §5b).

## 3. Strong simulation: the graded recurrence

The input is a product over sources rather than a list of photons, so the recurrence runs over
sources and a source that may or may not emit moves amplitude *between* layers:

    psi'[k] = sum_{q < d}  (a[i, q] / sqrt(q!)) * R_i^q psi[k - q]

`R_i` being the raising operator of that source's input mode — the same `k-1 -> k` edges,
the same mode-slab `scatter_add_`, as `FockSLOSKernel`. Nothing new is built: `slos/basis.py`
already constructs every layer 0..n, and `prob_norms(k)` generalises `output_prob_norms()` to the
intermediate ones. Deferred normalization is unchanged; the per-source `1/sqrt(q!)` is folded
into the coefficients, and a zero coefficient is skipped, so `a[i] = (0, 1)` reproduces the Fock
recurrence **bit for bit** (tested).

Cost, against a Fock input of the same `n` and `m`:

* **states**: the graded basis is `sum_k S_k = C(m + n, n)`, i.e. `(m + n)/m` times the
  n-photon layer — exactly **1.5×** at `m = 2n` (`masks.graded_state_space_size`).
* **work**: `sum_k (n - k + 1) |E_k|` against `sum_k |E_k|`. Because the layer sizes grow
  geometrically (ratio ≈ 3 at `m = 2n`), the weighted sum is ≈ 2.2 |E_n| against ≈ 1.5 |E_n| —
  a **constant factor ≈ 1.5, not a factor n**.

Measured on one H100 (run 20260919_181625, `bench/bench_product_slos.py`, m = 2n, complex64,
median of 5):

| n | 4 | 6 | 8 | 10 | 11 |
| --- | --- | --- | --- | --- | --- |
| states, Fock | 330 | 12 376 | 490 314 | 20 030 010 | 129 024 480 |
| states, graded | 495 | 18 564 | 735 471 | 30 045 015 | 193 536 720 |
| `probs(U)` Fock, ms | 1.99 | 3.74 | 5.14 | 19.4 | 98.9 |
| `probs(U)` graded, ms | 3.79 | 11.3 | 23.8 | 55.6 | 180 |
| time ratio | 1.91 | 3.02 | 4.63 | 2.87 | **1.82** |
| peak memory ratio | 1.33 | 1.13 | 1.10 | 1.08 | **1.08** |

The state count is 1.50× at every n, as the combinatorics say. At the size that decides the
question — n = 11, the reach of strong simulation — the graded kernel costs **1.8× the time and
1.08× the memory**, and reaches the same n (16.6 GiB peak against 15.4). So the prediction is
right in kind, a constant rather than a factor n, and a little optimistic on time. The bulge in
the middle (4.6× at n = 8) is launch-bound: the graded recurrence issues about n times more, and
smaller, kernel launches, which dominates until the layers are large enough to fill the device.
Memory is *better* than predicted because the peak is set by the transient slab work inside a
layer, not by the state vectors the graded form duplicates.

At m = 3n in complex128 the state ratio is 1.33 = (m + n)/m as expected and the time ratio
2.54 / 3.57 / 3.15× at n = 4 / 6 / 8; n = 10 runs out of memory on an 80 GB card (an 18.2 GiB
allocation on top of 77.9 GiB already held by the basis), which is a limit of the basis build at
m = 30, not of this recurrence.

`probs` returns a `loss.LossyDistribution` (one block per photon number) rather than a single
vector, because that is what an indefinite photon number is; the masses are the binomial law
above. `ProductSLOSKernel.sample` draws from it — exact for the coherent superposition, but
bounded by strong simulation (n ≤ ~11), not by a sampler.

## 4. One amplitude without the basis

For `d = 2` every factor of the output state is linear,

    prod_i ( c_i + s_i sum_j U[j, modes[i]] b_j^dag ) |0> ,   a[i] = (c_i, s_i),

so the amplitude of an output occupation `t` with `k = |t|` is the coefficient of
`prod_j z_j^{t_j}` in a product of linear forms, times `prod_j sqrt(t_j!)`. Give every photon its
own variable, `z_j -> sum_{s : r_s = j} x_s` over the k slots, and mark the emitting factors with
`lambda`. The `lambda^k` part is homogeneous of degree k in x, so the only degree-k monomial with
all exponents odd is `x_1...x_k` and Glynn's sign extraction is exact on it:

    amp(t) = 1 / (2^k prod_j sqrt(t_j!))
             * sum_{delta in {+-1}^k} (prod_s delta_s) [lambda^k] prod_i (c_i + lambda a_i(delta)),
      a_i(delta) = s_i sum_s delta_s U[r_s, modes[i]] .

The `lambda^k` coefficient is one truncated polynomial product — n steps on a (k+1)-vector — so
the cost is **O(2^k n k)**: exponential in the *detected* photon count, polynomial in the source
count. The sum runs over all `2^k` signs, not `2^{k-1}` — the global flip is a symmetry of the
summand only when every `c_i` vanishes, which is the Fock case.

**The permanent form, and why it is not the implementation.** Homogenising with one extra variable
instead makes the coefficient a permanent: with `M` the `n x n` matrix whose rows are the k output
rows (with multiplicity) followed by `n - k` copies of the constant row `(c_1 ... c_n)`,

    amp(t) = per(M) / ( (n - k)! prod_j sqrt(t_j!) ) ,

and Glynn's sum collapses those identical rows to a binomial weight over their `n - k + 1` sign
counts — the repeated-row collapse `clifford/cofactors_triton.py` applies to repeated *output*
rows. That identity is exact and the tests still check against it at small n. **It is a disastrous
way to evaluate the amplitude.** The collapse multiplies the constant row by `(g - 2v)`,
`g = n - k`, so each summand grows like `(g|c|)^n` while the answer stays tiny, and the digits are
gone long before the sizes that matter. Measured on the GPU run of 2026-09-19, against `U =
identity` where the amplitude is `cos^(n-k) sin^k` analytically at any size:

| n (α = π/4, k = n/2) | 8 | 16 | 24 | 32 | 40 | 48 |
| --- | --- | --- | --- | --- | --- | --- |
| relative error, collapsed permanent | 1e-14 | 3e-9 | 5e-3 | 1.2e3 | 2.2e11 | 1.6e19 |
| relative error, λ filter | 1e-16 | 2e-16 | 2e-16 | 3e-16 | 4e-16 | 6e-16 |

`slos/glynn.py` is the λ filter, plain batched torch (CPU or CUDA), chunked over the sign vectors,
checked against Ryser permanents of the explicit `M` at small n, against the graded kernel, and
against the analytic case up to n = 40 in the test suite. `bench/bench_product_amp.py --selfcheck`
re-runs that analytic check at every measured point, because an amplitude that is exact in exact
arithmetic tells you nothing about the one the machine computed. Use `complex128`.

Measured on one H100 (run 20260919_181625, complex128, median of 3), at α = π/4 so that n = 2k:

| k | 8 | 12 | 16 | 20 | 22 | 24 |
| --- | --- | --- | --- | --- | --- | --- |
| ms | 0.85 | 1.08 | 2.92 | 86.7 | 413 | 1938 |

4.76× and 4.70× per two photons against the 4.84× the model gives (2² for the signs, the rest from
n = 2k growing with k); below k ≈ 16 it is launch-bound and flat. At fixed k = 16 the cost is
linear in the source count — 1.96 / 2.93 / 4.32 / 5.63 / 8.34 ms at n = 20 / 32 / 48 / 64 / 96 —
and a fully bunched pattern costs the same as an unbunched one. The analytic self-check holds at
2e-16 to 4e-15 over that whole range.

Folding this into the Triton cofactor kernels has **not** been done; the torch path is
measured first, so that a kernel has something to beat.

## 5. Sampling: what exists, and why there is no chain rule

`clifford.sample` refuses a `ProductSuperposition` and points here. The reason is measured, not
assumed.

Post-selected on `k` photons, the parked superposition chain rule (2026-09-10) gives the marginal of the first `j` placed rows as a sum over the residual sets `R` of
sources not yet used. Swapping the two sums,

    p(r) = sum_{|R| = k-j} | sum_{S subset [n]\R, |S| = j} Perm(U[r, S]) |^2
         = a^dag G a ,   a_S = Perm(U[r, S]) ,   G[S, S'] = C(n - |S union S'|, k - j)

because the number of residual sets disjoint from `S ∪ S'` depends only on `|S ∪ S'|`. `G` is PSD,
so its eigendecomposition *is* the pure-term decomposition the sampler must carry, and

    the term count at step j is exactly rank(G) ,

independent of `U` and of any particular algorithm. `bench/sbs_terms.py` measures it. Result:

    R_j = C(n, min(j, k - j))      confirmed exactly for every n <= 12 (all k, all j)

`G` lives in the Johnson association scheme (it depends on `S, S'` only through `|S ∩ S'|`), which
is why the rank is this clean. Consequences:

* the peak is `C(n, k/2)` at `j = k/2` — at α = π/4 that is ≈ `2^{0.81 n}`, against the
  caterpillar's `R_k ≈ 325` at n = 24 (`docs/EMITTER_SAMPLER.md` §7);
* **no compression at the early steps**: for `j ≤ k/2` the rank is the *full* `C(n, j)`, so every
  subset is its own term;
* per-shot cost `sum_j R_j 2^j j` against a Fock sampler at the same photon count: **19×** at
  n = 20, k = 10 and **5.9 · 10^5** at n = 40, k = 20;
* truncating helps by a constant, not an exponential: at n = 12, k = 6, j = 3 (the peak), 99% of
  `tr(G)` sits in 66 of the 220 terms and 99.9% needs 202 — dropping the top Johnson shell costs
  about 1% of the trace. (An eigenvalue-mass proxy, not a TVD bound for an approximate sampler.)

**That rank is not the cost, and reading it as the cost was a mistake** (caught 2026-09-19 on the
question "does this make SBS harder than the other engines?" — it should have smelled wrong: SBS is
Fock boson sampling with a *degraded* source, and its incoherent twin at η = sin²α is strictly
easier, `n*` going up from 38.6 to 64.0). Two measurements say the naive decomposition is leaving
almost all the structure on the table:

* **The terms collapse into Johnson shells.** `G` has exactly `min(j, k−j) + 1` *distinct* nonzero
  eigenvalues, with multiplicities `C(n,i) − C(n,i−1)` — confirmed on every case tested. So the
  sampler needs `O(k)` numbers, the shell overlaps `‖P_i a‖²`, not `C(n, j)` independent terms.
* **The marginal has a residual-set form whose terms are cheap.** Expanding on the residual `R`
  rather than the placed set `S`,

      p_j(r) ∝ sum_{|R| = k-j} | A_R(r) |^2 ,   A_R(r) = sum_{S subset [n]\R, |S| = j} Perm(U[r, S]) ,

  and `A_R` is exactly what §4 computes: one λ-filter evaluation, `O(2^j n j)`, **no subset
  enumeration at all**. Verified to 1e-17 against brute-force marginals (`tests/slos/test_product.py`).
  So each term costs what a Fock step costs; what the naive count charges for is enumerating `R`.

That is the same shape Clifford & Clifford exploit: they never sum over their own `C(n, j)`
subsets either — they draw a uniformly random photon order first and condition on it, which is
valid because the input is exchangeable. The Dicke input is *permutation symmetric over the
sources*, which is more exchangeability, not less, and `R` is the register to randomise over.

**So there IS a chain rule, and it is built** (`clifford/product_sampler.py`, `clifford.sample`
dispatches to it). Since `M_j` *is* the marginal, the chain rule is exact by construction — no
exchangeability argument is needed, which is just as well: the analogue of Clifford & Clifford's
random photon order, a random source order, does NOT work here (measured: the consistency lemma
holds at j = 1 and fails from j = 2). What makes it possible is that `M_j` has two evaluations,
cheap at opposite ends of the chain:

| | cost | cheap when |
| --- | --- | --- |
| **residual**: enumerate R, each `A_R` by the λ filter of §4 | `C(n, k−j) 2^j j` | `k − j` small — the END of the chain |
| **replica**: swap the sums, `M_j = Σ_t C(n−2j+t, k−j) N_t` with `N_t = Σ_{|S∩S'|=t} Per_S conj(Per_S')`, all `N_t` from ONE doubled Glynn sum over `(δ,ε) ∈ {±1}^j × {±1}^j` carrying a `(|S|, |S'|, t)` table | `4^j n j³` | `j` small — the START |

Both are verified against the written-out sum to 1e-15 at every step, bunched rows included, and
the sampler's histograms match `ProductSLOSKernel.probs` within shot noise
(`tests/clifford/test_product_sampler.py`).

The residual cost carries no factor of `n`, and getting there was most of the optimisation of
2026-09-20. `A_R` needs `Π_{a∉R} (1 + λ P_a)`, and the `C(n, k−j)` residual sets overlap almost
completely, so rebuilding each product over all `n` sources repeats nearly all of the work. Instead
the product over every source is built once and factors are **divided out** along a prefix tree of
the excluded sets — one factor per level, `O(j)` per node, `Σ_t C(n,t) ≈ 1.3 C(n, k−j)` nodes.
Division by `1 + λp` is a unitriangular back-substitution (`c_0 = q_0`, `c_i = q_i − p c_{i−1}`),
exact and stable here: it agrees with the direct product to 1e-15 everywhere tested, and
`test_the_two_evaluations_agree_with_a_deep_prefix_tree` drives it to depth 7 against the replica
sum, which shares no code with it.

**What bounds it is the middle of the chain**, where neither branch is cheap. Measured on two CPU
cores at α = π/4 (n = 2k, m = 2n), k = 8: **2505 ms per sample → 947 → 339**, the first step being
the calibrated crossover (`cheaper`, which had been comparing the two cost models as if their units
were interchangeable) and the second the prefix tree. 7.4× in all, and the modelled end-to-end gain
at the frontier point n = 17 is 16.6×. Whether the `4^j` or the residual sum bounds a given step is
now genuinely close — `REPLICA_UNIT_COST` says the two cost about the same per model unit — and the
chain spends most of its steps on the residual side.

What is NOT worth doing, checked rather than assumed: the cost at the frontier is dominated by rare
high-k shots (at n = 17, k = 11…15 carry 84% of E[cost]) and within those by the middle and late
steps, which are large. So batching shots by photon number and fusing the replica recurrence — the
two obvious engineering moves — are worth about 6% between them, and the ~2 ms kernel-launch floor
is ~5% of a dominant shot even after the speedup above.

One caveat for any comparison drawn from this: the Fock and emitter samplers are tuned CUDA/Triton
kernels and this chain rule is plain torch with no kernel, so the measured gap is algorithm *and*
implementation. The remaining implementation headroom is now bounded and small: complex64 (~1e-6
relative on the marginals, up to 2×, but Fock and emitter are complex128 throughout so spending it
here alone would create the bias rather than remove it) and the collision discount (repeated placed
rows collapse the Glynn sum, `2^j → Π(mᵢ+1)`, which is exactly the mechanism behind the Fock
`ρ(c) < 2`; our sign enumeration ignores repeats, worth ~1.4× at m/n = 2 and ~2× at m/n = 1, and it
would restore a mode-ratio dependence to the sbs curve). At ρ_eff ≈ 2 per source, everything left
on the table is worth 1–2 sources. What it IS bounded by was measured on 2026-09-20 rather than
projected: on the lossy Fock state, which this engine and the Fock sampler both simulate exactly,
it reaches 22.4 sources against 64.4. That 42.0 is algorithm — Clifford & Clifford's random photon
order turns the Fock marginal into one permanent, and a random SOURCE order is not consistent, so
this chain rule pays the residual enumeration even when the input happens to be Fock — and it
dwarfs both the kernel effort (3–5 sources) and what coherence costs this algorithm (2.2, the
coherence scan of docs/LOSS.md §5b). The `4^j → 2^j` projection below, ~20 sources, is the part of
that 42.0 the replica sum owns. So this engine is not bounded by memory the way
strong-sim-then-sample is, and not by effort; it is bounded by the absence of a chain rule that is
continuous at zero coherence. **The open question is now sharp: can the two-replica sum be done in `2^j` instead of
`4^j`?** A loop hafnian of a `2k × 2k` matrix costs `2^k`, not `4^k`, which is why GBS samplers
work at all; whether this object has the same pairing structure is the thing to find out. If it
does, the per-step cost becomes Fock-like and the projection below is reachable.

**The other exact sampler.** `ProductSLOSKernel.sample` — strong simulation, then draw from the
computed distribution. It is tested against
the distribution within shot noise, and after the build the marginal cost of a sample is a
`searchsorted` (sub-microsecond). So the SBS modality *has* an exact sampler and a measurable
frontier; that frontier is just the strong-simulation one of §3 (memory-bound, n = 11 at m = 2n),
not a sampler's. Note that the draw itself needed the 2^24 category cap of `torch.multinomial`
lifted — `loss.draw_indices` falls back to an inverse-CDF draw above it, since the graded basis
passes the cap at n = 10 (30 M states) and the Fock one at n = 11 (129 M).

**Frontier placement.** Two different numbers, and they must not be conflated:

* *measured*, what we can do today: the chain rule above, bounded by the `4^j` of its middle
  steps rather than by memory (n\* = 7.8 detected photons on two CPU cores), and strong-sim-then-sample, bounded by §3 at n = 11 sources, i.e. a mean
  of 5.5 detected photons, with the distribution built in 180 ms and samples free after that;
* *projected*, what the NAIVE chain rule would give: **40 sources = 20 detected photons** at the
  per-shot cost a Fock sampler pays at its measured `n* = 38.6`. This is a lower bound on what a
  sampler could reach, not a target: it charges `C(n, min(j, k−j))` per step, and §5 now shows that
  count is an artefact of the decomposition, not of the problem.

The two are not the same kind of quantity either: strong-sim-then-sample amortises one build over
every shot, while the Clifford-family numbers are per-shot with no stored distribution. Any figure
carrying all three has to say which convention each point follows.

## 6. Loss

Uniform `η` composes with the graded distribution through `loss.thin_graded`: each source layer is
thinned with its own binomial weights and the results are summed per detected count, reusing the
per-layer descent `thin_distribution` already uses (extracted as `loss._descend`). Cost is `n`
times one `thin_distribution`, which is fine on a strong-simulation path. `eta = 1` returns the
blocks untouched, exactly — the cross-module invariant of `docs/ARCHITECTURE.md` §4, tested here
as object equality and cross-checked against `loss.thin_patterns`.

Note that loss and partial excitation are *not* interchangeable on the input side: an amplitude
`sin α` and a transmission `η = sin²α` give the same photon-number statistics but different
states — the first is coherent, the second is a mixture over which sources survived. §2 uses that
difference as the measurement.

## 6b. The frontier under the run-of-S statistic (2026-09-25)

GBS prices a frontier point by `t_S`, the mean cost per sample of a run of `S` samples, because its
cost is heavy-tailed (docs/GAUSSIAN.md §2.1):

    t_S = Σ_{k ≤ k_S} P(k) E[cost | k],    k_S = smallest k with P(K > k) ≤ 1/S

The same statistic is well defined here, and cheaper to evaluate than for GBS, because the two inputs
it needs are exact rather than sampled:

* **P(k) is analytic and U-independent** — Binomial(n, p) with p = η sin²α. GAUSSIAN.md §2 already
  notes that the Fock law Φ = z and the SBS law Φ = 1 − p + pz are the same statement.
* **E[cost | k] needs no stratum sampling and no surrogate patterns.** The chain has exactly k steps,
  and step j scores m candidates at a cost that `cheaper(n, k, j)` and the two evaluations fix from
  (n, k, j) alone — never from *which* rows were drawn. So the cost of a shot is a deterministic
  function c(n, m, k), and a stratum is priced by one post-selected k-photon shot
  (`bench/sbs_run_cost.py --price measure`) instead of by a ≥ 5-shot mean with a fitted tail.

Under loss the chain runs on the live sources only (§6), so a shot costs c(n_live, m, k) and
E[cost | k] averages that over n_live ~ Binomial(n, w₁) — the same table c(n, m, k) prices both.

**It changes nothing.** Calibrated model (`--price model`, constants fitted on run 20260920_165029:
fixed-k shots 0.83–1.10 of measured, per-n means median 1.14, 10–90 % 0.71–1.38):

| m/n | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n\*, S = 10⁴ | 21.2 | 20.7 | 20.4 | 20.1 | 19.9 | 19.6 | 19.1 | 18.8 | 18.3 |
| n\*, untruncated | 21.2 | 20.7 | 20.4 | 20.1 | 19.9 | 19.6 | 19.1 | 18.8 | 18.3 |
| measured mean (run 20260920_165029) | 21.3 | 20.5 | 20.2 | 20.1 | 19.9 | 19.5 | 19.1 | 19.1 | 18.6 |

t_S/t_untruncated is 0.986–0.995 across the nine ratios and n\*(10⁴) − n\*(untruncated) ≤ 0.02 sources;
S = 10⁶ and 10⁸ are identical to the digits shown.

**Why, and it is structural rather than lucky.** The cost-weighted law P(k)c(k) is the photon law
tilted p → pβ/(1 − p + pβ). At n = 20, m = 2n the median shot has k = 10 and the cost-weighted mean
sits at k = 14.2 — and P(K ≥ 14) = 5.8 %, i.e. **577 such shots in a run of 10⁴**. The sbs law is
Binomial, *bounded* at k = n with Gaussian tails; the Gaussian law is geometric in pairs and unbounded,
which is why its cost is carried by shots ~4× brighter than typical that a run never contains.

**Where it does bite**, mildly: the lossy points, whose brightness p = η/2 pushes the cost-weighted
tilt further into the tail relative to the median. At m = 2n:

| η | 1 | 0.9 | 0.8 | 0.7 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| n\*, S = 10⁴ | 20.4 | 21.7 | 23.2 | 25.2 | 31.2 |
| n\*, untruncated | 20.4 | 21.6 | 23.2 | 25.1 | 30.8 |
| t_10⁴/t_untrunc | 0.993 | — | 0.966 | — | 0.808 |

Even at η = 0.5, where a run of 10⁴ sees only 81 % of the untruncated mean cost, the frontier moves
0.4 sources, because ρ_eff ≈ 2 per source turns a 19 % cost change into log(1/0.808)/log 2 = 0.31.

**So adopting t_S for sbs costs nothing and buys comparability**: no sbs number has to be re-measured,
and the one statistic can be quoted for every engine. The numbers above are calibrated-model numbers;
`--price measure` turns each stratum into a timed post-selected shot, which at the frontier is ~k shots
per n — cheaper than the unconditioned search it would replace, and with no tail variance.

## 7. Open items

1. **`4^j → 2^j` in the two-replica sum of §5.** That single exponent is the whole distance
   between the sampler that now exists and the projected frontier. The hafnian precedent (a
   `2k × 2k` loop hafnian costs `2^k`) is the reason to think it may be there.
2. The Triton cofactor kernels with the λ filter folded in (§4).
3. `slos/estimate.py`: the Gurvits-type estimator extends to any input with a poly-size
   purification, so a `ProductSuperposition` arm is the natural next lemma after the emitter one.
4. Renema's order-truncated approximate simulator and a partial-distinguishability axis (§1) —
   a framework-level decision, currently out of scope.
5. `d > 2` is implemented and tested against the oracle but has no application yet; weak-coherent
   and truncated-thermal sources are what it is for.
