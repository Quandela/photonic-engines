# Non-linear boson sampling: single-mode photon-number phases between interferometers

`photonic_engines/nonlinear.py` — the model of Spagnolo, Brod, Galvão & Sciarrino, *Non-linear boson
sampling* (arXiv:2110.13788): Haar W, a single-mode phase exp(−iφ n̂ₓ²) on one mode x, Haar V —
generalised to any phase response per photon number and to several sites, in one layer or in sequence.
Status 2026-09-23: model, both oracles, strong simulation (two routes), the exact sampler with loss and
partial distinguishability, the frontier engine (`find_threshold.py --engine nlbs`, `bench/nlbs_frontier.py`)
and three GPU runs (20260922_211901, 20260923_080913 and 20260923_111505, H100 PCIe:
the ratio, φ and site scans of §5b, the imperfection grid of §6c). The linear-optical gadget of arXiv:2110.13788 is not
implemented (§7). Tests: `tests/test_nonlinear.py`, tutorial §9f.

## 0. Summary

- **Branch identity.** A site sees at most n photons, so exp(−i f(n̂ₓ)) is exactly a finite sum of
  phase shifters Σ_ℓ c_ℓ exp(iθ_ℓ n̂ₓ), and the whole circuit is a coherent sum of B linear
  interferometers U_b with weights c_b (§2). B and ‖c‖₁ depend on the response only: Kerr at
  φ = π is one branch (a π phase shifter), φ = π/2 two, φ = π/4 four, a generic φ about n.
- **Strong simulation.** One SLOS pass per branch, summed coherently (`route="branch"`), or the state
  pushed through the circuit gate by gate (`route="gate"`: SLOS for U_0, a diagonal phase per site,
  nearest-neighbour two-mode gates for the next interferometers). Branches multiply with the number
  of sites, gates do not; the crossover is calibrated (§3).
- **The Clifford & Clifford random-order chain does not hold** (§4). The law of the first k photons
  has coherences between different input subsets S ≠ T, weighted by permanents of the branch Gram
  matrix I + (rank ≤ #sites); for a linear circuit they vanish, which is the whole of C&C's argument.
  Counterexample (n = 3, m = 4, Kerr 0.9): the random-order chain run on the branch sum is off by
  TVD 0.115. The exact chain over the true marginals exists but squares the per-step cost (4^k
  against 2^k) — not a frontier algorithm.
- **The exact sampler is a rejection from the incoherent branch mixture** (§5): draw a branch with
  probability |c_b|/‖c‖₁, draw a boson sample of the linear U_b with `clifford.fock_sample` (kernels
  unchanged), accept with |Σ_b c_b a_b|² / (‖c‖₁ Σ_b |c_b| |a_b|²). Exact by Cauchy–Schwarz,
  acceptance exactly 1/‖c‖₁² — measured 0.43–0.59 against 0.50 at ‖c‖₁² = 2.
  **Measured** (§5b): the frontier is the Fock one minus log(‖c‖₁²(1 + κB))/log ρ with κ = 0.55 ± 0.08,
  i.e. −2.4 photons at φ = π/2 whatever the mode ratio, and 1.9 photons per site, layout-independent.
- **The φ envelope is arithmetic, not monotone** (§5b, §5c): what the sampler pays is the denominator of
  φ/π, so φ = π/2 is the cheapest non-trivial point (−2.5 photons) and the weak φ = π/16 the most expensive
  measured (−8.3). Continuity at φ → 0 holds at fixed n but the scale is φn², so not at frontier sizes.
- **Imperfections** (§6): loss before the sites is the input-side draw every engine uses, and a stream then
  runs the expansion of its OWN photon count; loss after the last site is binomial thinning; partial
  distinguishability keeps the branch index shared by the species, which stay independent given it. All three
  are exact and checked against an enumerated mixture over partitions. **Measured** (§6c): against the
  measured Fock frontier the toll stays 2.5–3 photons at η = 1 and 0.9 and under V = 0.9, so the η → ηx law
  of the other engines carries over — a site costs a constant multiple of the Fock sampler, imperfections
  included.

## 1. The model and its notation

| symbol | code | meaning |
| --- | --- | --- |
| n, m | `C.n`, `C.m` | photons (= sources, one photon each), modes |
| t | `in_modes` | input modes |
| U_0 … U_L | `unitaries` | the interferometers, m × m, rows = outputs, columns = inputs |
| site (ℓ, x, f) | `sites` | exp(−i f(n̂ₓ)) on mode x between U_{ℓ−1} and U_ℓ, ℓ = 1..L |
| f(j), j = 0..n | `site_phases(f, n)` | the phase response; a number φ means Kerr, f(j) = φ j² (`kerr`) |
| g(j) = e^{−i f(j)} | — | the site's response on the photon number |
| (θ_ℓ, c_ℓ) | `branches(f, n)` | frequencies and weights of the site's expansion (§2) |
| U_b, c_b | `C.branch_table()` | branch interferometers U_L D_L … D_1 U_0 and weights (product over sites) |
| B | `C.n_branches` | number of branches, product over sites |
| ‖c‖₁² | `C.rejection_factor()` | proposals per sample of the sampler (§5) |

Sites of one layer sit on distinct modes; sequential layers may reuse a mode. The response vector
covers every non-linearity diagonal in the photon number: Kerr, χ n(n−1) (= Kerr up to a linear
phase), a quantum-dot/cavity π phase on j ≥ 2, saturable responses. Sanity identities (tested):
φ = 0 is the identity, φ = π is a π phase shifter (n² ≡ n mod 2), a site on a mode no photon reaches
is the identity.

## 2. The branch identity

**Claim.** For a site on mode x and n photons, there are frequencies θ_1..θ_P (P ≤ n + 1) and weights
c_ℓ with

    exp(−i f(n̂ₓ)) = Σ_ℓ c_ℓ exp(iθ_ℓ n̂ₓ)      on the n-photon space.

*Proof.* n̂ₓ is diagonal in the Fock basis with eigenvalues in {0..n}, so two diagonal functions of it
are equal on the n-photon space iff they agree on 0..n. The characters j ↦ e^{iθ_ℓ j},
θ_ℓ = 2πℓ/(n+1), form a basis of the functions on 0..n (the (n+1)-point DFT), so
c_ℓ = (1/(n+1)) Σ_j g(j) e^{−iθ_ℓ j} works. ∎

`branches` looks for a shorter expansion: if g(j + q)/g(j) is constant (= e^{iaq}) on 0..n for some
q ≤ n, then g = e^{iaj} h(j) with h q-periodic, and the q-point DFT of h shifted by a is exact on 0..n.
It takes the smallest such q (q = n always works, a single ratio being constant), drops the numerical
zeros (|c| < 10⁻¹⁴) and checks the reconstruction to 10⁻¹². Kerr at a rational φ = 2πp/q' has a period
dividing q' whatever n (the Gauss-sum case: flat |c_ℓ|, so ‖c‖₁² = P).

**Circuit.** Each exp(iθ n̂ₓ) is the linear phase shifter D(θ) on mode x, a product of linear-optics
unitaries is linear, and the expansion is linear in each site, so with several sites (same layer or
sequential alike)

    U_L N_L ⋯ N_1 U_0 = Σ_b c_b Û_b ,   U_b = U_L D_L(θ_b) ⋯ D_1(θ_b) U_0 ,   c_b = Π_sites c_{ℓ(site)} ,
    ⟨s| C |t⟩ = Σ_b c_b Per(U_b[s, t]) / √(s! t!) .

**Gram structure.** For two branches, U_b'† U_b = U_0† D_1'* U_1† ⋯ U_L† U_L ⋯ U_1 D_1 U_0: the last
interferometer cancels, the others do not, but every D_ℓ'* D_ℓ is the identity plus a rank-one term on
the site modes where θ differ, so U_b'† U_b = I + (rank ≤ number of differing sites). For one site,
U_b'†U_b = I + (e^{i(θ_b − θ_b')} − 1) w w†, w = U_0† e_x.

Oracles (both in `slos/reference.py`, sharing no code with the engine or with each other): (a)
`nonlinear_state_amplitudes`, the dense n-photon representation of every U_ℓ (permanents) and a
diagonal phase per site; (b) `nonlinear_branch_amplitudes`, the branch sum of permanents on the plain
(n+1)-point grid for every site. They agree to 10⁻¹⁶ on the four test cases (one site, two sites in a
layer, two sequential sites on one mode, a bunched input).

## 3. Strong simulation (`NonlinearCircuit.amplitudes`, `probs`)

* `route="branch"`: `FockSLOSKernel.amplitudes` on the batch of U_b, `chunk` branches at a time,
  accumulated as Σ c_b ψ_b. Cost B × one SLOS pass; memory one chunk of amplitude vectors plus the sum.
* `route="gate"`: SLOS for U_0; then per layer the diagonal site phases g(occ[:, x]) and the next
  interferometer applied to the general state by `TwoModeNetwork` — the Reck-type elimination of U into
  m(m−1)/2 nearest-neighbour two-mode gates, each acting on the pairs of basis states that differ only on
  its two modes (index pairs built once per mode pair and cached; the gate's (N+1)² blocks from the
  binomial expansion). Cost independent of the number of branches: L m(m−1)/2 × S(1 + 2n/m) entries.
* `route="auto"`: the cheaper of `route_costs()`, branch B·E against E + 4G (E = SLOS edges
  Σ_k size_k·k, G the gate entries). The factor 4 is measured, one CPU core, complex128:

| n, m, sites (layers) | B | branch (s) | gate (s) | model picks |
| --- | --- | --- | --- | --- |
| 5, 10, 1 | 5 | 0.0013 | 0.016 | branch |
| 5, 10, 2 (sequential) | 25 | 0.0032 | 0.034 | branch |
| 6, 12, 1 | 6 | 0.0053 | 0.062 | branch |
| 6, 12, 2 (sequential) | 36 | 0.031 | 0.143 | branch |
| 6, 12, 3 (sequential) | 216 | 0.290 | 0.159 | gate |
| 7, 10, 1 | 7 | 0.0056 | 0.043 | branch |
| 8, 10, 2 (sequential) | 64 | 0.140 | 0.185 | branch |

  Kerr 0.7 on every site (a generic φ, B ≈ n per site). Not measured on a GPU.
* Memory frontier (`bench/memory_frontier.py`, key `nlbs`, count only): the branch route holds one
  branch's last two layers and the coherent sum, 8(M_{n−1} + 2M_n) bytes against 8(M_{n−1} + M_n) for the
  Fock push kernel; at 80 GiB, m = 2n: n = 12.9 against 13.5 for `fock_pull`.

## 4. The Clifford & Clifford chain: exchangeability verdict

C&C sample the ordered output r = (r_1..r_n) of the first-quantised, exchangeable law, one photon at a
time, and the random-order argument says the marginal of r_{≤k} is the average over a uniform k-subset
S of the input columns of the k-photon linear law |Per(A[r, S])|²/k!. That is what makes step k cost
one k × k permanent family instead of a sum over subsets.

**The marginal of a branch sum.** The circuit maps the symmetrised input to Σ_b c_b U_b^{⊗n}|Sym t⟩.
Tracing photons k+1..n contracts each traced photon's ket and bra through the Gram matrix
G_bb' = (U_b'† U_b)[t, t] instead of δ, and summing over the assignments of traced photons gives

    p(r_{≤k}) = (n−k)!/n! Σ_{b,b'} c_b c̄_b' Σ_{|S|=|T|=k} Per(U_b[r, S]) conj Per(U_b'[r, T]) · Per(G_bb'[T^c, S^c]) .

For a linear circuit G = I, Per(I[T^c, S^c]) = δ_{ST}, and the C&C form follows. Here G_bb' = I + α w̄ wᵀ
(one site, α = e^{i(θ_b−θ_b')} − 1), whose sub-permanents are, with d = |S \ T|,

    Per(G[T^c, S^c]) = Π_{j∈S\T} w̄_j Π_{j∈T\S} w_j · Σ_q (q + d)! α^{q+d} e_q(|w_j|², j ∉ S ∪ T) ,

non-zero for S ≠ T in general: the traced photons that passed through the site keep a coherence between
different subsets of kept photons. Verified numerically in `test_true_marginal_has_subset_coherences`
(the formula matches the enumerated marginal to 10⁻¹⁶; dropping S ≠ T misses by > 10⁻³).

**Verdict: the random-order argument fails.** A C&C chain needs the k-marginal to be an average over S
of a function of S alone; the S ≠ T terms prevent it. The input being a Fock state is not enough — the
evolution is no longer a tensor power U^{⊗n}, so exchangeability of the photon order (which still holds)
does not reduce the marginal to subsets. Counterexample (`test_random_order_chain_is_not_exact`, n = 3,
m = 4, Kerr 0.9 on one site): the C&C chain with random order and step weight |Σ_b c_b Per(U_b[r_{≤k}, σ_{≤k}])|²
is off by TVD 0.115, while the same enumeration reproduces the linear circuit to 10⁻¹⁵.

**The exact chain over the true marginal** exists: the double subset sum is a generating function over
the columns (four states per column: in S ∩ T, S only, T only, neither; degree tracking for the
factorial and for the rows as in `clifford.prefix_columns`), so step k costs O(D² n) per candidate row with
D = Π(p_l + 1) ≤ 2^k — 4^k against C&C's 2^k. The exponent of the frontier doubles; it is not used.

## 5. The exact sampler: rejection from the incoherent branch mixture (`NonlinearCircuit.sample`)

**Proposal.** q(s) = Σ_b (|c_b|/‖c‖₁) |a_b(s)|²/s!, a_b(s) = Per(U_b[s, t]): a mixture of linear boson
samplers. Draw b with probability |c_b|/‖c‖₁ (one multinomial for the batch), then one boson sample of
U_b per draw — `clifford.fock_sample` with the stack of U_b and per-circuit shot counts, kernels unchanged.

**Acceptance.** Cauchy–Schwarz with weights |c_b|:

    p(s) = |Σ_b c_b a_b|²/s! ≤ (Σ_b |c_b|)(Σ_b |c_b| |a_b|²)/s! = ‖c‖₁² q(s) ,

so accepting s with probability |Σ_b c_b a_b|² / (‖c‖₁ Σ_b |c_b||a_b|²) ≤ 1 yields exact samples of p
(standard rejection; s! cancels). The acceptance rate is exactly Σ_s q(s) p(s)/(‖c‖₁² q(s)) = 1/‖c‖₁².
The weights |c_b| minimise the Cauchy–Schwarz constant (Σ|c_b|²/w_b over Σ w_b = 1). By Parseval
‖c‖₂ = 1 for a DFT expansion, so ‖c‖₁² ≤ P per site and ‖c‖₁² = Π_sites ‖c‖₁² for several.

**Cost per accepted sample.** ‖c‖₁² × (one Fock sample of n photons + B permanents Per(U_b[s, t])).
The permanents use `laplace_cofactors_rows` (the 2020 cofactor kernel, repeated rows exploited) plus
one Laplace step, batched over (draw, branch); `backend="torch"` is a batched Glynn/Gray for CPU.
One such permanent costs about one C&C final step, i.e. a fixed fraction κ of a Fock sample (κ to be
measured, ~0.5 expected). With B = 1 (φ = 0, φ = π, any linear response) no permanent is computed and
the sampler IS the Fock sampler, so those points land on the Fock frontier by construction.
The accepted draws of a batch are shuffled before the last batch is cut (the proposal returns them
grouped by branch).

### 5a. Branches and rejection factor for Kerr, one site (`branches(φ, n)`)

| φ | n = 10: B, ‖c‖₁² | n = 20 | n = 30 | n = 40 |
| --- | --- | --- | --- | --- |
| 0.001 | 10, 1.03 | 20, 1.14 | 30, 1.32 | 40, 1.59 |
| 0.01 | 10, 1.35 | 20, 2.62 | 30, 5.14 | 40, 8.57 |
| π/64 | 10, 3.00 | 20, 9.80 | 30, 19.6 | 40, 31.8 |
| π/16 | 10, 8.54 | 16, 16 | 16, 16 | 16, 16 |
| π/8, 3π/8, 5π/8 | 8, 8 | 8, 8 | 8, 8 | 8, 8 |
| π/4, 3π/4 | 4, 4 | 4, 4 | 4, 4 | 4, 4 |
| π/2 | 2, 2 | 2, 2 | 2, 2 | 2, 2 |
| π | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| 1.0 (generic) | 10, 8.69 | 20, 18.2 | 30, 26.9 | 40, 33.2 |

For φ = 2π/q (q ≡ 0 mod 4) the expansion is a Gauss sum: B = ‖c‖₁² = q/2, independent of n. A generic
φ saturates at B = n, ‖c‖₁² ~ n. The expansion is not the smallest possible ‖c‖₁: an ℓ¹-minimal
expansion over arbitrary frequencies (basis pursuit, prototyped with cvxpy, not in the package) gives
22 instead of 33 at φ = 1.0, n = 40 and the same values at the rational points.

### 5b. The measured cost law (run 20260922_211901, H100 PCIe, 1 sample/s)

Cost per sample relative to the Fock sampler at the same n: ‖c‖₁² (1 + κB), κ the cost of one branch
permanent in Fock samples, so the frontier moves by Δn* = −log(‖c‖₁²(1 + κB))/log ρ(c). Measured on one
H100 at 1 sample/s, one site, φ = π/2 (‖c‖₁² = B = 2), against the Fock row of `bench/memory_frontier.py`:

| m/n | 1 | 1.5 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fock n* | 42.4 | 40.0 | 38.5 | 36.8 | 36.2 | 35.3 | 34.9 | 34.4 | — |
| NLBS n* | 40.1 | 37.4 | 36.0 | 34.5 | 33.8 | 33.0 | 32.5 | 32.1 | 32.0 |
| Δn* | 2.3 | 2.6 | 2.5 | 2.3 | 2.4 | 2.3 | 2.4 | 2.3 | — |
| κ | 0.33 | 0.59 | 0.60 | 0.54 | 0.65 | 0.63 | 0.73 | 0.68 | — |

**κ = 0.55 ± 0.08** over the ratio and φ scans (the m = n point is the outlier; Δn* is smallest there, so κ
is least constrained). The toll is a CONSTANT ≈ 2.4 photons in the ratio, as the law says it should be: the
non-linearity costs a fixed multiple of the Fock sampler, not a change of its exponent. The measured
acceptance matched 1/‖c‖₁² at every point (0.43–0.59 against 0.50 at B = 2 — corrected 10-15 from 0.44, the log's minimum is 0.430 at m/n = 8, n = 32;
`results/keep/nlbs_acceptance_20260922_h100.csv`, `bench/nlbs_cost_law.py` — 0.054–0.077 against 0.0625 at
B = 16), which is the Cauchy–Schwarz bound of §5 being tight, measured.

**φ scan, m = 2n, one site** (Fock 38.5):

| φ | 0 | π/16 | π/8 | π/4 | 3π/8 | π/2 | 5π/8 | 3π/4 | π |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B = ‖c‖₁² | 1 | 16 | 8 | 4 | 8 | 2 | 8 | 4 | 1 |
| n* | 38.5 | 30.2 | 32.2 | 34.2 | 32.2 | 36.0 | 32.2 | 34.3 | 38.3 |
| model (κ = 0.55) | 38.5 | 30.1 | 32.3 | 34.3 | 32.3 | 36.2 | 32.3 | 34.3 | 38.5 |

φ = 0 lands on the Fock frontier exactly (38.5) and φ = π within the search's resolution (38.3), as they
must: both are one branch, and the sampler then IS `fock_sample`. **The envelope is not monotone in the
strength of the non-linearity — it is arithmetic in φ.** What costs is the denominator of φ/π, so the
"maximally non-linear" Kerr point φ = π/2 is the CHEAPEST non-trivial one (−2.5 photons) while a weak
φ = π/16 is the most expensive measured (−8.3), and 3π/4 (−4.2) is cheaper than π/8 (−6.3). A φ with no
short period at all costs B ≈ n and ‖c‖₁² ~ n (not measured; the scan is rational multiples of π).
Figure `img/nlbs_phi.png`.

**Sites, m = 2n, φ = π/2** (‖c‖₁² = B = 2^L):

| L | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| n*, same layer | 36.0 | 34.3 | 30.3 | 21.4 |
| n*, sequential | 36.0 | 34.3 | 30.3 | 21.5 |
| model (κ = 0.55) | 36.0 | 34.2 | 29.9 | 20.8 |

(L = 8 from run 20260923_080913, after the OOM fix; the other rows from 20260922_211901 and reproduced by it.)
**The layout does not matter** — same-layer and sequential agree to 0.1 photon at every L, as the branch
expansion says (§2: the sites commute into the same product of B unitaries whatever their order). The
register grows as predicted: **2.1 photons per site** over L = 1 … 8 (1.9 over L = 1 … 4), i.e. two factors of
2 per site — the rejection and the permanents — against ρ = 1.81 per photon. The model is 0.6 photon
pessimistic at L = 8, the one place where the B = 256 permanents dominate the Fock sample and their κ is not
the κ of a single permanent. Figures `img/nlbs_ratio.png`, `img/nlbs_sites.png`.

### 5c. Continuity at φ → 0 — answered: no, not at frontier sizes

At fixed n, ‖c‖₁² → 1 as φ → 0 (the table of §5a) and the envelope is continuous. But the scale is φn²: at
n = 40, φ = 0.01 already costs ‖c‖₁² = 8.6 and π/64 costs 32. The measured scan says it plainly: the cost
jumps from 0 to 8.3 photons between φ = 0 and φ = π/16 and then comes BACK UP as φ grows. So the SBS control
this scan was meant to be does not hold: unlike a coherence, the Kerr strength is not a knob that interpolates
smoothly between "free" and "expensive" for this algorithm — it is the arithmetic of the response that prices
it. Whether that is the physics or the algorithm is open, and it is the sharpest open question here: the
site-occupation law is concentrated on small j while exactness forbids truncating it, so a proposal that
prices a weak Kerr as weak would change the picture (§7).

## 6. Imperfections (`sample(loss=, eta_pre=, eta_post=, indist=)`)

**Loss before the sites** (`loss=η`, also spelled `eta_pre`). Uniform loss commutes with U_0, so it can be
drawn on the input side: `imperfections.partition` gives each stream the photons that enter, and the stream
is a k-photon instance of the same circuit. Two consequences: the expansion is taken at k
(`branch_table(k)`, exact on 0..k), so a stream with fewer photons also has fewer branches and a smaller
‖c‖₁² — at a generic φ the toll shrinks with the loss instead of staying at its n-photon value — and the
Fock sampler underneath sees a k-photon instance, so the usual E[ρ^k] law still carries the cost.

**Loss after the last site** (`eta_post`): binomial thinning of the counts (`imperfections.thin_counts`),
exact because loss commutes with U_L. An end-to-end η splits as η = η_pre η_post; where the split sits is a
modelling choice, and the two are NOT interchangeable here — loss before the site removes photons that would
have shifted the site's phase. For a quantum-dot/cavity site in the middle of a chip the honest reading is
that source and input-coupling losses are η_pre and detector losses η_post; the package defaults to
`loss=η` (all of it before the sites), which is the same input-side convention as every other engine and
the CHEAPER of the two for the simulator — a frontier measured that way is a lower bound on the hardness of
the same device with the loss placed later.

**Loss between two sites is not covered.** A photon lost there has already shifted the phase of the earlier
sites, so tracing it couples ket and bra branches through the Gram matrix of §4 — the same structure that
kills the chain rule, and the reason this is not a silent approximation: `sample` takes η_pre and η_post,
nothing else.

**Partial distinguishability** (`indist=x` or an `imperfections.Distinguishability`). Physical model: the
site acts on the TOTAL photon number of its mode, whatever the internal state of the photons (a cross-Kerr
between species). In the branch basis this is exactly what makes it tractable:

    exp(iθ n̂ₓ) = Π_species exp(iθ n̂ₓ,μ) ,

so given the branch the species are independent and share the branch index. With a species partition
(subsets G_μ of the input columns) the amplitude of a species-resolved output (s_μ) is

    A((s_μ)) = Σ_b c_b Π_μ Per(U_b[s_μ, t_μ]) / sqrt(Π_μ s_μ!) ,

coherent over the branches, incoherent over the partition and over the species-resolved outputs. The
Cauchy–Schwarz bound of §5 holds verbatim with a_b = Π_μ Per(U_b[s_μ, t_μ]), so the sampler keeps its
acceptance 1/‖c‖₁²: one branch per stream, one linear sample per species from that branch's U_b (singletons
routed classically), one product of permanents per branch. Alternative model, NOT the default: only the
photons of the common species feel the non-linearity (the site coupled to one internal mode). It is a
different physical device — a spectrally selective non-linearity — and it does not factor over species in the
phase basis, so it would need its own derivation.

Oracle: `tests/test_nonlinear.py::_imperfect_probs` enumerates the mixture over loss/species partitions and
the species-resolved outputs, thins it, and the sampler matches it within shot noise at
(η, x, η_post) = (1,1,1), (0.7,1,1), (1,0.8,1), (0.8,0.7,1), (1,1,0.8) and (0.9,0.9,0.9).

## 6b. The frontier engine (`find_threshold.py --engine nlbs`)

`--phi` (an expression in π — give it exactly, `pi/4` and not `0.7853982`, or the period detection of §2
falls back to B = n), `--site-phase` (an explicit response vector), `--sites`, `--layout same-layer |
sequential`, `--ratio`, `--loss`, `--eta-post`, `--indist`; n is the source count and m = round(c n), no
post-selection, and the CSV carries the extra columns (phi, sites, layout, eta_post). The site modes are
spread over the m modes. The per-n line reports the acceptance actually measured next to 1/‖c‖₁².
The sanity check at frontier sizes is the detected photon number n η η_post (a site conserves the photon
number); the per-mode marginal is not |U|² column sums any more, and computing it exactly costs a permanent
of identity-plus-rank per branch PAIR, so it is not done.

`bench/nlbs_frontier.py` drives the four scans (ratio, φ, sites, imperfections), seeds each search from the
cost model of §5b and writes CSV + PNG.
Job 5 (the gadget diagnostic) is not implemented (§7).

## 6c. Imperfections measured (run 20260923_111505) — the toll survives loss and distinguishability

Three runs were needed, the first two of which priced the sampler's bookkeeping rather than the algorithm
(§6d). With the proposals sized by the rejection factor, at φ = π/2 and m = 2n:

*Superseded (10-06): NLBS η = 1, V = 1 is 36.0 (run 20260922_211901); Fock 38.5 / 41.4 / 49.4 at η = 1 / 0.9 / 0.7, V = 1 (ALGORITHMS §9) — the toll is 2.5 / 2.8 / 4.6; kept as history.*

*Superseded (10-09, 256-shot batches only): NLBS η = 1, V = 1 is 36.1 (run 20261009_141132); Fock 38.3 / 41.2 / 49.2 at η = 1 / 0.9 / 0.7, V = 1 (runs 20261009_141132 / 20261008_071215 / 20261008_104426) — the toll is 2.2 / 2.6 / 4.4 (NLBS η = 0.9 / 0.7 cells 38.6 / 44.8 unchanged).*

| n* | η = 1 | η = 0.9 | η = 0.7 |
| --- | --- | --- | --- |
| V = 1: NLBS | 36.1 | 38.6 | 44.8 |
| V = 1: Fock, measured | 38.6 | 41.4 | 49.3 |
| **toll** | **2.5** | **2.8** | **4.5** |
| V = 0.9: NLBS | 37.1 | 39.7 | 47.9 |
| V = 0.9: Fock, measured | 40.0 | — | 51.0 |
| **toll** | **2.9** | — | **3.1** |

The Fock rows are the measured lossy and partially distinguishable frontiers of `docs/LOSS.md` §7 (run
20260917_182540) and `docs/DISTINGUISHABILITY.md` §5 (run 20260921_094327), not the η x law — and that is
the point. **Against the measured Fock engine the toll of the non-linearity stays the 2.5–3 photons of
§5b under loss and under partial distinguishability**, so the η → ηx law of the other engines carries over
to this sampler: a non-linear site costs a constant multiple of whatever the Fock sampler costs, imperfections
included. Against the *law* the NLBS points sit 0.1–3.7 photons low, but so do the Fock ones at η = 0.7 (49.3
against 51.0, and DISTINGUISHABILITY.md §5 records 1.7–3.4 low with ρ_eff scattered 1.45–2.02): the lossy cost
is tail-dominated and a two-point bracket at η = 0.7 is noisy. Our own η = 0.7 points show the same scatter
(ρ_eff 1.55 and 2.20), and the one outlier — the 4.5-photon toll at (η = 0.7, V = 1) — sits inside it. The
honest statement is therefore: **constant toll, confirmed at η = 1 and 0.9 to 0.3 photons; at η = 0.7 within
the noise the Fock baseline itself carries**. Beating that noise needs more shots per point at η = 0.7 rather
than another engine change. Figure `img/nlbs_imperfections.png`.

## 6d. What the first two grids measured

Runs 20260922_211901 (job 4) and 20260923_080913 (job 2) both gave 36.0 / 35.3 / 36.1 / 37.6 / 43.3 / 44.8,
2–6 photons below the η x law and not monotone in x. Neither was a measurement of the algorithm:

* the sampler ran one rejection loop per loss/species partition, so one `fock_sample` call per STREAM under
  loss (fixed 2026-09-23 — and it changed the numbers by nothing, which is how the second cause was found);
* every partition was proposed with a fixed slack of 8 draws, paid per stream: 686 proposals for 64 samples
  at η = 0.9, n = 36, i.e. 4.6× the work. Sizing them as `need × ‖c‖₁² × 1.15` (no per-group slack, a short
  group reappearing in the next pass) is what produced the grid of §6c, 1–3 photons higher at every point.

Since then the pass has no per-partition step at all: the work is grouped by (species size, photon count,
species slot), one `fock_sample` call and one batched permanent call per class over per-stream column sets
(`_reject_batch`, `_permanents` with a [S, p] column tensor). Measured proposals per sample are within 1.15×
of ‖c‖₁² with and without loss (`test_proposals_per_sample_is_the_rejection_factor`), and the number of calls
follows the classes rather than the partitions (136 partitions → 53 calls at n = 5, η = 0.7, V = 0.64).

## 7. Open items

* **The η = 0.7 column deserves more shots** (§6c): the toll there (4.5 photons at V = 1 against 2.5–3.1
  everywhere else) sits inside the bracket noise that the Fock engine shows at the same transmission. More
  shots per point, or a third n in the bracket, would settle it; no engine change is indicated.
* **The gadget diagnostic** (job 5 of the brief, arXiv:2110.13788 §III). Their (k+1)-mode gadget heralds one
  photon in each of k auxiliary modes, and since the site's photon number is conserved by that heralding the
  effective operation is DIAGONAL in it: a site whose response vector A(j) = ⟨j,1^k|U_eff|j,1^k⟩ is
  sub-unitary rather than a phase. So the diagnostic needs two things this package does not have yet: sites
  with a complex (non-unitary) response — a small generalisation of `site_phases`, with the post-selected
  renormalisation carried through `probs` and the acceptance — and the optimisation of U_eff against the Kerr
  response that the paper performs. Not invented here: it needs the paper's construction in front of us.
* **The ℓ¹-minimal expansion** (§5a) as an option of `branches`: basis pursuit over continuous frequencies,
  22 instead of 33 at φ = 1.0, n = 40.
* **A better proposal at small φn²** (§5c), so that the envelope is continuous at φ = 0 at frontier sizes.
* **Strong simulation with imperfections**: `probs` is lossless and indistinguishable; the mixtures are
  enumerated in the tests as the sampler's oracle only.
