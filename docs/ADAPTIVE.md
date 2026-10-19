# Adaptive linear optics: mid-circuit photon counting with feed-forward on the interferometer

`photonic_engines/adaptive.py` — the model of Chabaud, Markham & Sohbi, *Quantum machine learning
with adaptive linear optics* (arXiv:2102.04579), on a Fock input. The marginal routine is Gurvits'
second algorithm as restated by Oh & Lim, *Classical algorithms for measurement-adaptive Gaussian
circuits* (arXiv:2509.00746). This is feed-forward on the INTERFEROMETER of a spatial boson sampler;
feed-forward on the spin gates / routing of an emitter train is `sequence.sample_diag(policy=)`
(SEQUENCE.md §10), a different engine.

## 1. The model and its notation

n single photons enter an m-mode interferometer U_0. Some output modes are measured before the end,
in Λ successive **levels**; after each level a classical function of everything read so far picks the
unitary applied to the modes that are still unmeasured. At the end every remaining mode is measured.
Detectors resolve photon number (PNR) — that is the model. Threshold detection is not a variant of this
engine but a detector setting common to every engine of the package, to be added once for all of them.
There is no loss; so each photon is detected exactly once, either at a level or at the end.

| symbol | code | meaning |
| --- | --- | --- |
| n, m | `C.n`, `C.m` | photons, modes |
| t = (t_1..t_n) | `in_modes` | input modes, one photon each (bunched inputs are refused) |
| U_0 | `U0` | first interferometer, m × m, rows = outputs, columns = inputs |
| Λ | `C.n_levels` | number of measurement levels |
| A_λ, ℓ_λ | `levels[λ-1]` | the modes measured at level λ (λ = 1..Λ) and their number |
| K_λ = ℓ_1 + … + ℓ_λ | `C.K_upto[λ]` | modes measured after level λ; K_0 = 0 |
| K = K_Λ | `C.K` | **number of adaptive modes** — the cost parameter |
| a_1..a_K | `C.adaptive_modes` | the adaptive modes in measurement order (A_1, then A_2, …) |
| p = (p_1..p_K) | tuple `p` | the photon counts read on a_1..a_K: the **adaptive outcome** |
| p_{≤λ} | `p[:K_upto[λ]]` | its prefix after level λ (always a whole number of levels) |
| r = p_1 + … + p_K | — | photons detected adaptively (random; mean nK/m for Haar U_0) |
| R_λ | `C.rest(λ)` | the m − K_λ modes unmeasured after level λ, increasing order |
| V_λ(p_{≤λ}) | `policy(λ, p)` | feed-forward unitary on R_λ, (m − K_λ) × (m − K_λ); `None` = identity |
| U^{p_{≤λ}} | `C.unitary(p)` | E(V_λ) ⋯ E(V_1) U_0, cumulative m × m unitary; E embeds with 1 on the measured modes |
| s | — | counts on the m − K final modes R_Λ; n − r photons |
| ω = (p, s) | rows of `C.occupations()` | the full m-mode occupation, adaptive modes included |

Correspondence: K is the *k* of arXiv:2102.04579 (which has ℓ_λ = 1, Λ = K, a_j = mode j) and the *L*
of arXiv:2509.00746. The letters k (emitters) and L (loop bins) are taken elsewhere in this package,
hence K and Λ here. Grouping modes into levels is scheduling only: a level of ℓ modes is ℓ
one-mode levels with identity feed-forward in between. **Only K enters the cost**, not Λ and not
the policy. The policy may depend on all of p_{≤λ}, not only on the last level.

One permanent per outcome (the first K rows of U^p are fixed as soon as they are measured, because
later V act on other modes):

    ⟨ω| … |t⟩ = Per( U^p[ω, t] ) / sqrt(ω!) ,     P(ω) = |Per(U^p[ω, t])|² / ω! .

There are C(n + K, K) adaptive outcomes p (branches).

**Key fact.** P(p_{≤λ}) depends on U^{p_{<λ}} only: V_λ, V_{λ+1}, … act on modes that are traced in
that marginal. So the adaptive outcomes form a chain over levels whose conditionals are ordinary
K_λ-mode marginals of a NON-adaptive boson sampler.

## 2. Mode marginals (`binomial_moments`, `mode_marginals`)

For L output modes b_1..b_L and V = U[b, t] (L × n), the generating function of their counts is

    E[ Π_l z_l^{N_l} ] = ⟨t| U† z^{N̂} U |t⟩ = Per( 1 + V† (Z − 1) V ) ,

the permanent of identity plus rank L. Expanding in y = z − 1 (Gurvits; Eq. 66 of arXiv:2509.00746),

    Per(1 + V† Y V) = Σ_c y^c T[c] ,   T[c] = c! [x^c w^c] Π_{i=1..n} ( 1 + (Σ_l conj(V_li) x_l)(Σ_l V_li w_l) ) ,

and T[c] = E[Π_l C(N_l, c_l)] are the binomial moments of the counts. Every factor adds one x and one w, so
only monomials x^a w^b with |a| = |b| occur: the table is GRADED, one M_d × M_d block per total degree d
(M_d = C(d+L−1, L−1)), and a photon maps block d into block d + 1 by two sparse "times a linear form" maps.
Σ_d M_d² ≈ n^{2L−1} / ((2L−1)(L−1)!²) entries against (n+1)^{2L} for the dense table this replaced (2026-09-21):
2·10⁴ / 5·10⁶ / 7·10⁸ at n = 38 for L = 2 / 3 / 4, where dense was 2·10⁶ / 3.5·10⁹ / 5·10¹². The distribution
follows by binomial inversion along each axis, P(q) = Σ_{c ≥ q} (−1)^{c−q} C(c, q) T[c].

* Exact (no sampling), polynomial for fixed L. Measured on one CPU core of the cloud container, Haar m = 2n,
  n = 38: L = 1 0.02 s, L = 2 0.04 s (the dense table took 3.0 s), L = 3 0.5 s (dense: did not fit beyond n = 21);
  L = 4 at n = 30: 38 s. The 4 GB guard (`MAX_MARGINAL_ENTRIES` = 2²⁸ entries) admits L = 4 up to n = 32.
  Agreement with the dense implementation it replaced: 1e-16 on moments and marginals, L = 1..4.
* The inversion is an alternating sum. It is well conditioned while the moments decay (mean
  occupation per mode ≲ 1); the absolute error stays at the 1e-16 level in the tests above, which
  means the RELATIVE accuracy of a branch probability below ~1e-12 is poor. The sampler only uses
  such values to size rejection batches. A result below −1e-9 raises.
* One photon per input mode only (the identity in "identity + low rank" is ⟨t_i|t_j⟩).

## 3. Strong simulation (`AdaptiveCircuit.probs`, `outcome_probs`, `final_marginal`)

* `probs()` — the joint P(ω) over the SLOS basis: one batched `FockSLOSKernel` pass per branch p,
  each contributing the basis states that show p on the adaptive modes. C(n+K, K) SLOS passes;
  stops where SLOS stops. It is the oracle of the samplers.
* `outcome_probs(lam)` — {p_{≤λ}: P(p_{≤λ})} by walking the branch tree with K_λ-mode marginals.
* `final_marginal(modes)` — counts on a few final modes summed over the branches,
  Σ_p P(p, N = q): one (K + |modes|)-mode marginal under U^p per branch. This is the
  probability-estimation task of arXiv:2102.04579, here exact, polynomial for K + |modes| = O(1).

## 4. Exact sampling (`AdaptiveCircuit.sample`)

The Clifford & Clifford chain rule runs over PHOTONS in random order; adaptivity runs over MODES. It
does not extend: conditioning on "exactly p in the measured modes" sends the other photons through a
sub-unitary whose Gram matrix 1 − V†V couples different input subsets. The sampler therefore has two
stages.

1. **Adaptive outcomes, level by level.** `chain="marginal"`: for each distinct prefix, one
   `mode_marginals` on a_1..a_{K_λ} under U^{p_{<λ}}, sliced at the prefix, gives the law of the
   level's counts — exact, no rejection, table (n+1)^{2K_λ}. `chain="rejection"`: a boson sample of
   U^{p_{<λ}} is kept when it reproduces p_{<λ} and its level-λ modes are read off — any K, cost
   1/P(p_{<λ}) per level.
2. **Final stage s | p.** `final="clifford"`: rejection on the ordinary boson sampler
   (`clifford.fock_sample`, or any `base_sampler`) run on U^p; a draw is kept when its adaptive
   modes show p. Acceptance is exactly P(p), known in advance from stage 1, which sizes the batches.
   `final="slos"`: directly from the strong distribution of U^p restricted to p (small n).
   `min_accept` sends the branches with P(p) below it to the `slos` route.

Both are exact: a kept draw of an i.i.d. stream is a conditional sample, and conditioning a sample of
U^p on p gives s | p because U^p is the unitary of that branch. `C.stats` reports the boson samples
drawn (`base_shots`), the distinct branches and those routed to SLOS.

## 4b. The forced prefix: Clifford & Clifford adapted to adaptive measurements (`final="prefix"`, default)

The rejection of §4 asks a full boson sample of U^p to show p by chance, and pays 1/P(p) for it. It does not
have to. The Clifford & Clifford chain draws an ORDERED array ρ = (ρ_1..ρ_n) of output modes, one per photon,
with law q(ρ) = |Per U[ρ, t]|² / n!, symmetric under permutations of the positions; the boson sample is its
multiset. Write a^p for the adaptive modes with multiplicities (a_1 repeated p_1 times, then a_2, …; r entries).

1. *Positions carry no information.* Given that the multiset shows exactly p on the adaptive modes, the r
   positions holding those entries are uniform among the C(n, r)·r!/p! arrangements. So the multiset law given
   "exactly p" equals the law given E' = "ρ_1..ρ_r = a^p and no later entry is an adaptive mode".
2. *E' is cheap to sample.* Force the first r entries to a^p, run the chain for the remaining n − r photons
   from its true conditionals, and reject only if one of THEM lands on an adaptive mode.
3. *Starting the chain in the middle.* The chain is exact given the SET of input columns its first k steps
   have used: ρ_{≤k} | S ~ |Per U[ρ_{≤k}, S]|² / k!, S uniform among the C(n, k) subsets (the lemma behind algorithm B of arXiv:1706.01260),
   and later steps only see that set. Conditioning on the first r rows is therefore: draw S with weight
   |Per U^p[a^p, S]|² among the C(n, r) subsets, put S first in the column order, the other columns in uniform
   order, and run steps r+1..n unchanged — same kernels (`clifford.fock_sample(prefix=)`, `_prefix_order`).
   S is NOT enumerated (`prefix_columns`): the prefix has only K distinct rows, so
   Per U[a^p, S] = p! [x^p] Π_{j∈S} (Σ_l U[a_l, t_j] x_l), the weight of S is the coefficient of x^p w^p in
   Π_{j∈S} (B_j·x)(conj(B_j)·w), and the columns are decided one at a time against suffix tables
   F_j[a, b] = [x^a w^b] Π_{j'≥j} (1 + (B_j'·x)(conj(B_j')·w)) — the polynomial of §2 again. Cost
   O(n K² D² + shots·n·D²) with D = Π(p_l + 1): 10 ms for r = 13 at n = 38, where C(38, 13) = 5·10⁹.
   (`prefix_subset_weights` keeps the enumeration, as the test oracle.) Since 2026-09-22 (task A11) the draw
   is batched over the circuits of a lockstep pass, `prefix_columns_many`: circuits whose prefixes share one
   multiplicity shape share the table shapes, so their suffix tables are [G, D, D] per column and the n
   column decisions run over all streams at once — one sequence of launches per shape instead of one per
   circuit, which is what the imperfect sampler needs (every stream its own circuit, §7).
4. *Acceptance.* q(E') = P(p) / (number of arrangements) and q(ρ_{≤r} = a^p) = E[Π_l C(N_l, p_l)] / (the same
   number), so the acceptance is **P(p) / T[p]**, T the binomial moments of §2 — planting p photons
   over-counts the outcomes with more than p of them by the ways of choosing which ones were planted, and the
   rejection removes exactly that. Cross-check used by the tests: Σ_S |Per U[a^p, S]|² = p! T[p].
5. *Cost.* A branch met with probability P(p) costs T[p]/P(p) chains on average, so the mean number of chains
   per sample is **Σ_p T[p] = E[2^r]**, r the number of photons read adaptively: the 1/P(p) cancels, rare
   branches weigh no more than common ones, and there is no tail. For Haar U_0 the occupation of a mode is close
   to geometric with mean μ = n/m, E[2^N] = 1/(1 − μ): 2 per adaptive mode at m = 2n, 1.5 at m = 3n, slightly
   less than the product over modes (the counts are anticorrelated).

The same conditioning serves the levels of `chain="rejection"` (force p_{<λ}, read the level-λ modes of the kept
draws). Every branch takes this route, whatever r: the first version enumerated the C(n, r) subsets and sent the
branches above 2^21 of them to the full-sample rejection, and that fallback alone put the tail back — one chain
per fallback branch on average, ~150–200 per sample at n = 28–36, K = 2, and visibly so in run 20260921_173919
(n = 28: 296 chains per sample, n = 32: 263, against 3.2–3.6 expected; n = 38, K = 1: 6.1 against 1.95).
What it does not change: every chain still costs a full Clifford & Clifford sample (its last steps dominate),
and E[2^r] is exponential in the photons read adaptively — dense regimes (m → n) or K of order m stay hard.

## 4c. Can the rejection go away? (lab log, 2026-09-21)

Question: an exact conditional chain for s | p with no rejection would bring the adaptive overhead from E[2^r]
chains to one. Plan: (1) write the conditional marginal in closed form and check it numerically; (2) see whether
the hidden-column trick that gives Clifford & Clifford their 2^k per step survives the conditioning; (3) if not,
take what the derivation does give.

**Step 1 — the conditional marginal exists in closed form.** With a the adaptive modes, R the others, and G =
A_R†A_R = 1 − V†V (V = A[a, t], rank K), the constrained prefix marginal Σ_{ρ_{>k} ∈ R} |Per A[ρ, t]|² /(n−k)! is

    p(ρ_≤k | no later photon on a) ∝ Σ_c (−1)^{|c|} / c! · Σ_S |Per A[(ρ_≤k, a^c), S]|² ,

an inclusion–exclusion over the adaptive counts of the future photons: every term is an UNCONSTRAINED prefix
marginal with the adaptive rows appended c times — the same objects as the forced prefix. (Generating function:
Π_j (1 + (Ã_j·u)(conj Ã_j·u')) with Ã = [A[ρ_≤k, :]; V], u = (z, −w), u' = (z', x), coefficient of z z' and
Σ_c c! x^c w^c.) Checked by brute force at n = 5, m = 8, K = 1, 2, prefixes of 1–3 rows: direct sum over the
remaining rows in R equals the alternating sum to 7e-15.

**Step 2 — the hidden-column trick does not survive. FAILED, and the failure is structural.** C&C 2018 avoid
Σ_S by drawing the column set S uniformly, which is exact because the posterior of S given ρ_≤k is diagonal
(orthogonality of the full row set). Under the constraint that posterior is the Hermitian form
H[S, S'] = Per(ρ, S) conj Per(ρ, S') Per(G[S̄', S̄]), whose off-diagonal part is 28–56 % of the diagonal
(measured at n = 5: 0.275 / 0.381 for K = 1, k = 1 / 2; 0.328 / 0.563 for K = 2) and complex — no positive hidden
variable. Its positive decomposition is over the future rows themselves (H = Σ_{ρ_{>k}} |…|²), i.e. the original
problem. Without the trick, every term of step 1 is a subset-summed marginal Σ_S |Per|² — 4^k per step by
Glynn (Σ_{δ,δ'} e_k of the products), against 2^k for one chain: the rejection-free chain would cost ~2^n times
MORE than the E[2^r] ≈ 3.6 rejected chains it saves. Route closed. Physically: conditioning on empty adaptive
modes turns the Fock input into a superposition input (A_R = W Σ Q†, K attenuated modes in the singular basis),
and superposition inputs are where SUPERPOSITION.md §7 already found no 2^n sampler.

**Step 3 — what the rejection actually costs, and the abort.** A rejected chain is decided the moment a free
photon lands on an adaptive mode, so the rest of the chain need not run. By exchangeability the position of the
first such photon is uniform among the n − r free steps while step k costs ~2^k: an aborted chain costs on
average ≤ 2/(n − r) of a full one (≈ 0.056 at n = 38). `fock_sample(avoid=)` drops a stream from the lockstep
batch right after the offending step (the kernels see the survivors only); `final="prefix"` uses it. Chains
STARTED per sample stay E[2^r]; predicted time ratio to `fock_sample` ≈ 1 + (E[2^r] − 1)·2/(n − r): 1.06 and
1.15 for K = 1 and 2 at n = 38, against the measured 2.2 and 3.2 without the abort — Δn* ≈ −0.1 / −0.2 sources.
Exactness on the interpreted kernels (n = 4, m = 8, 3000 shots): survival 0.6827 / 0.4187 / 0.8313 vs P/T
0.6813 / 0.4286 / 0.8302 (≤ 1.1σ), TVD to the SLOS conditional at the floor. GPU: job 16, not run yet.
So: −1 source per adaptive mode was the price of running rejected chains to the end, not of the rejection.

## 4d. The algorithm in one place (pseudo-code of `AdaptiveCircuit.sample`, `final="prefix"`)

Input: U_0 (m × m), input modes t (n), levels A_1..A_Λ (K adaptive modes a_1..a_K in measurement order),
policy V_λ(p_{≤λ}), number of samples N. Output: N exact samples ω = (p, s). Notation of §1; T[·] the binomial
moments of §2 (graded table); Per by Ryser/Glynn inside the kernels.

    SAMPLE(U_0, t, levels, policy, N):
        for each of the N samples (batched by distinct prefix in the code):
            # ---- stage 1: the adaptive outcome p, level by level ----
            p ← (), U ← U_0
            for λ = 1..Λ:
                if the K_λ-mode graded table fits (Σ_d C(d+K_λ−1, K_λ−1)² ≤ MAX_MARGINAL_ENTRIES):
                    T ← BINOMIAL_MOMENTS(U, t, a_1..a_{K_λ})          # §2, cached per distinct prefix p
                    P ← BINOMIAL_INVERSE(T)                           # law of the counts on a_1..a_{K_λ}
                    draw the level-λ counts from P[p_{<λ}, ·] / Σ P[p_{<λ}, ·]
                else:                                                 # level chain, K_λ beyond the tables
                    repeat: ρ ← PREFIX_CHAIN(U, t, prefix = a^{p_{<λ}}, avoid = {a_1..a_{K_{λ−1}}})
                    until ρ ≠ ABORT                                   # acceptance P(p_{<λ}) / T[p_{<λ}]
                    level-λ counts ← counts of ρ on A_λ
                p ← p ⊕ (level-λ counts)
                U ← E(V_λ(p)) · U                                     # feed-forward; E = identity on measured modes
            # ---- stage 2: the final counts s given p ----
            r ← |p|; a^p ← (a_1 × p_1, …, a_K × p_K)                 # the r adaptive rows, with multiplicity
            repeat:
                ρ ← PREFIX_CHAIN(U, t, prefix = a^p, avoid = {a_1..a_K})
            until ρ ≠ ABORT                                           # acceptance P(p) / T[p]; chains started, averaged over p: E[2^r]
            s ← counts of ρ_{r+1..n} on the final modes R_Λ
            emit (p, s)

    PREFIX_CHAIN(U, t, prefix (r rows), avoid):                      # clifford.fock_sample(prefix=, avoid=)
        S ← PREFIX_COLUMNS(U, t, prefix)                              # r input columns, S ∝ |Per U[prefix, S]|²
        column order ← S first, the other n − r columns in uniform random order
        ρ_1..ρ_r ← prefix                                             # the chain starts in the middle: step r done
        for k = r+1..n:                                               # ordinary Clifford & Clifford steps
            w_j ← |Per U[(ρ_{<k}, j), columns_{≤k}]|² for every output mode j   # Laplace over the new column (kernel)
            draw ρ_k ∝ w_j
            if ρ_k ∈ avoid: return ABORT                              # early abort, cost so far ≤ 2/(n − r) of a full chain
        return ρ

    PREFIX_COLUMNS(U, t, prefix):                                     # polynomial draw of the hidden columns, §4b.3
        B_j ← U[a, t_j] for j = 1..n                                  # K × n: the prefix has only K distinct rows
        F_{n+1} ← 1; for j = n..1: F_j ← F_{j+1} · (1 + (B_j·x)(conj(B_j)·w)), truncated at degree p    # suffix tables
        g ← 1 (polynomial in x, degree ≤ p); S ← ∅
        for j = 1..n:
            g_in ← g · (B_j·x)                                          # column j taken
            w_in ← Σ_{a,b} g_in[a] conj(g_in[b]) F_{j+1}[p−a, p−b];  w_out ← the same with g   # mass of all completions
            with probability w_in / (w_in + w_out): S ← S ∪ {j}, g ← g_in
        return S                                                      # |S| = r; P(S) ∝ |Per U[prefix, S]|² exactly

    BINOMIAL_MOMENTS(U, t, modes L):                                  # §2, Gurvits' second algorithm
        V ← U[L, t]; G ← 1 (polynomial in x_1..x_L, w_1..w_L, graded by total degree)
        for i = 1..n: G ← G · (1 + (Σ_l conj(V_li) x_l)(Σ_l V_li w_l))
        return T[c] = c! [x^c w^c] G                                  # = E[Π_l C(N_l, c_l)]

Correctness: stage 1 is exact because P(p_{≤λ}) depends on U^{p_{<λ}} only (key fact of §1) and a graded table
or a kept chain gives that marginal exactly; stage 2 is exact by the exchangeability argument of §4b.1–4 (a kept
chain is a sample of s | p). Cost: one chain of the plain sampler plus (E[2^r] − 1) aborted chains at ≤ 2/(n − r)
each, plus one such block per level beyond the tables (§5b).

Batching (`AdaptiveCircuit._forced`, 2026-09-22): the "repeat … until" loops above are not run one branch at a
time. The streams of one branch are pooled (a pool of w streams starts ⌈w / (P/T)⌉ chains, the expected need),
the pools are grouped by (photon number, prefix length) — a lockstep pass needs one chain length and one step to
start at — and each group is ONE `fock_sample` call on the stacked circuits U^p[:, cols] with a per-circuit
prefix ([G, r]) and per-circuit stream counts ([G]); the streams still without a survivor go into the next
round. The aborted streams leave the lockstep batch after each step (`_sample_streams`). Before this the final
stage was one call per branch (`_reject`), and the fixed cost of those 45–70 calls per 128 shots is what put
the K ≥ 4 crossings of §5b–5c under the model (task A3). Measured (run 20260922_162506, job 24): K = 6 / 8 at m/n = 2 from 33.4 /
30.2 to 34.2 / 31.6, K = 4 unchanged at 36.9 (its cost is the K = 4 table, §5b); chains per sample unchanged
(18.7 at K = 4, 100–130 / 240–345 at K = 6 / 8), time ratio at n = 38, K = 4 from 2.77 to 2.46.

## 5. What the rejection costs (measured, `bench/adaptive_overhead.py`)

The mean number of boson samples drawn per accepted sample is E[1/P(p)] = the number of outcomes,
C(n+K, K): every branch, however rare, contributes exactly 1. The TYPICAL branch costs 2^H, H the
entropy of p. Haar U_0, m = 2n, exact outcome law, 4 unitaries per point (2026-09-21, no sampling
involved, so no hardware dependence):

| n | K | branches C(n+K,K) | H (bits) | branches with P ≥ 0.01 | mass below 0.01 |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 11 | 1.37 | 4.2 | 0.005 |
| 10 | 2 | 66 | 2.81 | 10.0 | 0.044 |
| 10 | 3 | 286 | 4.31 | 18.2 | 0.127 |
| 20 | 1 | 21 | 1.41 | 4.0 | 0.011 |
| 20 | 2 | 231 | 2.77 | 9.8 | 0.045 |
| 20 | 3 | 1771 | 4.21 | 19.0 | 0.111 |
| 30 | 1 | 31 | 1.33 | 4.0 | 0.010 |
| 30 | 2 | 496 | 2.72 | 9.8 | 0.044 |
| 38 | 1 | 39 | 1.40 | 4.0 | 0.012 |
| 38 | 2 | 780 | 2.74 | 10.0 | 0.042 |

H ≈ 1.4 bits per adaptive mode, additive in K and independent of n (the single-mode occupation at
m = 2n is close to geometric with mean 1/2, entropy 1.38 bits).

Everything in the table above prices the FULL-SAMPLE rejection (`final="clifford"`). With the forced prefix
(§4b) the mean is Σ_p T[p] = E[2^r] instead, computed from the same exact tables (4 Haar unitaries per point):

| m/n | K | n | E[2^r] (mean chains per sample) | branches C(n+K,K) (mean of the full-sample rejection) |
| --- | --- | --- | --- | --- |
| 2 | 1 | 20 / 30 / 38 | 2.09 / 2.00 / 1.95 | 21 / 31 / 39 |
| 2 | 2 | 20 / 30 / 38 | 3.66 / 3.69 / 3.56 | 231 / 496 / 780 |
| 2 | 3 | 20 | 6.26 | 1771 |
| 3 | 1 | 20 | 1.47 | 21 |
| 3 | 2 | 20 | 2.08 | 231 |

Sampler check on the CPU oracle chain (n = 3, m = 6, levels [[1], [0, 3]], 20 branches, 40 000 shots): 3.22 chains
per sample with the prefix against 20.6 without, both at the shot-noise floor of the strong joint.

Reading for the simulability frontier, with Δn* = −log X / log ρ and ρ ≈ 2:

* forced prefix, exact, X = E[2^r]: about −1.0 / −1.9 / −2.7 sources for K = 1 / 2 / 3 at m = 2n, mean = typical;
* the three lines below are the full-sample rejection it replaces:

* typical branch, X = 2^H: about −1.4 sources per adaptive mode;
* exact sampling at a mean rate, X = C(n+K, K): −5.3 / −9.6 sources for K = 1 / 2 at n = 38. This is
  the honest figure for this algorithm, and it is **algorithm-limited**: the tail of rare branches
  sets it. Routing them to SLOS only helps at small n — at n = 38 a branch with P < 0.01 has r ≳ 3
  photons in p and still n − r ≈ 35 photons to place, far beyond strong simulation;
* if a total-variation error ε is accepted (rare branches dropped), X = branches kept: −2.0 / −3.3
  sources at ε ≈ 0.01 / 0.04 — not exact sampling.

GPU measurement of the same thing, against `fock_sample` on the same U_0 and host:
`bench/adaptive_frontier.py` (`--check` is the distribution test on the real Clifford sampler). Run 20260921_151344, one H100 PCIe, FULL-SAMPLE rejection, 64 shots (32 for K = 3),
Haar feed-forward per outcome prefix:

| n (m = 2n) | fock samples/s | K = 1 samples/s | chains/sample | time ratio | K = 2 samples/s | chains/sample | time ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 28 | 535 | 162 | 3.3 | 3.3 | 16.2 | 33 | 33 |
| 30 | 169 | 38.3 | 4.1 | 4.4 | 3.07 | 53 | 55 |
| 32 | 46.0 | 12.7 | 3.6 | 3.6 | 1.04 | 45 | 44 |
| 34 | 13.7 | 3.40 | 4.3 | 4.0 | 0.36 | 40 | 38 |
| 36 | 3.93 | 0.96 | 4.7 | 4.1 | 0.20 | 22 | 20 |

The cost model holds: from n = 28 up the time ratio to `fock_sample` IS the number of chains per sample (within
~10 %); below n ≈ 24 fixed overhead dominates (ratio 25–28 for 3–40 chains: one sampler call per branch, the
marginal tables, the policy's QR on the host). 1 sample/s crossings of that run: fock 38.2 (extrapolated,
ρ_eff = 1.87), K = 1 36.0, K = 2 32.2, K = 3 (rejection chain, n ≤ 30, 73–242 chains/sample) ≈ 30.9; at m = 3n:
37.9 / 35.4 / 30.6. These adaptive crossings are OPTIMISTIC for that algorithm: its chains/sample is heavy
tailed (a branch met c times costs c / P(p), so one hit at P = 0.003 is 330 chains; the mean is C(n+K, K) at any
shot count and does not depend on m/n), and 64 shots rarely meet the tail. With the mean, the same cost model
gives ≈ 32.5 / 28.4 / 25.2 for K = 1 / 2 / 3. The forced prefix (§4b) removes that tail.

FORCED PREFIX, run 20260921_205843 (same host, 256 shots, marginal chain), samples/s and, in
brackets, chains per sample / time ratio to `fock_sample`:

| n (m = 2n) | fock | K = 1 | K = 2 |
| --- | --- | --- | --- |
| 28 | 602 | 249 [1.76 / 2.42] | 97.0 [3.81 / 6.21] |
| 30 | 163 | 79.8 [2.05 / 2.05] | 41.2 [3.89 / 3.97] |
| 32 | 48.5 | 28.0 [1.80 / 1.73] | 11.7 [4.66 / 4.14] |
| 34 | 14.4 | 7.37 [2.25 / 1.95] | 4.63 [3.85 / 3.10] |
| 36 | 4.18 | 2.31 [2.26 / 1.81] | 1.25 [4.73 / 3.33] |
| 38 | 1.35 | 0.62 [2.20 / 2.17] | 0.42 [3.48 / 3.22] |

Chains per sample sit at E[2^r] (≈ 2.0 and ≈ 3.6) at every n, no outlier in 48 adaptive points over the three
jobs, and the time ratio equals it from n = 30 (slightly below at large n: a forced chain skips its first r
steps). **Crossings of 1 sample/s: fock 38.5 (extrapolated), K = 1 37.3, K = 2 36.5 (both bracketed)** — shifts of
−1.2 and −2.0 sources for −1.0 and −1.9 predicted from E[2^r]. At m = 3n (job 11): fock 36.9, K = 1 36.4, K = 2 35.6,
chains per sample 1.35–1.70 and 2.4–2.8 (E[2^r] ≈ 1.5 / 2.1): the overhead falls with the occupation, as it
should now that the mean is E[2^r] and not the branch count.
K = 3 and 4 (job 10, 64 shots) had to use `chain="rejection"` because the DENSE marginal table of that day stopped
at n = 21 for K = 3 (the graded table of §2 lifts this; not run at the time); that chain pays a forced-prefix rejection at EVERY level, 16–26 chains per sample for K = 3 and 26–41 for
K = 4 (time ratio 15–18 and 26–31 at n = 32–36), crossings 33.5 and 32.6. That is the cost of the level chain,
not of the final stage (E[2^r] = 6.3 for K = 3): a marginal table that scales past K = 2 is the next thing to
build. Below n ≈ 28 the fixed overhead still dominates (ratio 8–50 for 2–4 chains).

For K of order m every route here dies (E[2^r] grows like 2^{nK/m}) and only the projected state vector
remains (memory frontier); that regime is KLM-universal.

## 5b. The toll of adaptivity against K (`bench/adaptive_toll.py`, `img/adaptive_toll.png`)

Fixed m/n, x = K (top axis r̄ = nK/m, the photons read adaptively — the variable the cost depends on, so the
m = 2n and m = 3n curves collapse against it), y = Δn* = n*(K) − n*(0). Cost model in chains of the plain sampler
(the first-hit model, 2026-09-22; task A9):

    X(K) = C(law of R on the K adaptive modes) + Σ_{λ beyond the tables} C(law of R on the modes measured before λ),

    C(law) = Σ_R P(R) Σ_{h=0}^{R} C(R, h) · cost(h; r = R − h, N = n − r),

where R is the total photon count on the avoided modes. A branch p with r = |p| forced rows starts T[p] chains
per sample; a fraction Σ_{|q| = r + h} Π_l C(q_l, p_l) P(q) / T[p] of them see h more photons on the avoided
modes. Summed over the branches with the Vandermonde identity Σ_{p ≤ q, |p| = |q| − h} Π_l C(q_l, p_l) =
C(|q|, h), the cost depends on the law of R only. The chain with h = 0 is the accepted one and runs to the
end; a chain with h ≥ 1 is aborted at the first hit, whose position among the N free steps is that of the first
of h uniformly placed points (the array law is exchangeable: P(J = j) = C(N − j, h − 1) / C(N, h)), and it is
charged the steps it ran at ρ^k per step (ρ the plain sampler's per-photon factor): (ρ^{r+J} − ρ^r) / (ρ^n − 1)
of a full chain. The law of R comes from the Gaussian integral of §4b, one axis more than E[2^r]:
E[z^R] = Per(1 + (z − 1) V†V) = E_g Π_i (1 + (z − 1) |g†V_i|²), a signed Poisson-binomial per sample of g
(`count_law_gauss`, checked against the exact table: max error 7e-4 at 10⁵ samples), so the model has no
table limit; for the levels beyond the tables the same C on the modes measured before the level (the first
level is a plain chain). The earlier model charged every aborted chain the BOUND 2/(n − r̄) — one hit at a
uniform position — and is kept as X_bound: tight where the toll is small, an order of magnitude too high on
the dense side (§5c). n = 38, ρ = 1.82, `--tables-upto 4`, four Haar unitaries per point
(results/keep/adaptive_toll.csv); measured crossings at m/n = 2 from results/keep/adaptive_crossings.csv:

| K | r̄ (m = 2n) | E[2^r] | X | Δn* model | n* model | n* bound model | n* measured (H100) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5 | 2.0 | 1.03 | −0.05 | 38.4 | 38.4 | 38.3 / 38.4 |
| 2 | 1.0 | 3.8 | 1.07 | −0.11 | 38.4 | 38.3 | 38.0 / 38.2 |
| 3 | 1.5 | 6.8 | 1.10 | −0.16 | 38.3 | 38.0 | 38.1 |
| 4 | 2.0 | 12 | 1.14 | −0.22 | 38.3 | 37.7 | 36.9 |
| 5 | 2.5 | 21 | 2.31 | −1.4 | 37.1 | 36.1 | 34.0 |
| 6 | 3.0 | 32 | 3.52 | −2.1 | 36.4 | 35.1 | 34.2 (33.4 before the batching) |
| 8 | 4.0 | 100 | 6.13 | −3.0 | 35.5 | 33.3 | 31.6 (30.2 before the batching) |
| 12 | 6.0 | 643 | 11.9 | −4.1 | 34.4 | 30.2 | — |

Reading: with the tables (K ≤ 4 at n = 38) the algorithmic toll is a fifth of a source at most — the aborted
chains are cheap because the first hit comes early (with h hits among N free steps the first is at N/(h + 1) on
average, and a chain stopped there costs ρ^{−hN/(h+1)} of a full one); the step at K = 5 is the first level
drawn by chain instead of by table (the shaded region of the figure), about one full chain per such level,
and beyond it the slope is a source per two or three levels. The measured points follow the model to 0.3
sources through K = 3 and fall below it from K = 4, by 1.4 / 3.1 / 2.2 / 3.9 sources at K = 4 / 5 / 6 / 8:
implementation, not algorithm. Batching the branches (§4d, run 20260922_162506) moved K = 6 / 8 from 33.4 /
30.2 to 34.2 / 31.6 and left K = 4 at 36.9 (37.0 with the tables batched, job 26 — the K = 4 table is compute-bound, not launch-bound): there the cost that is not in the model is the K = 4 graded table
itself (1e8 entries, one per distinct branch, ~50 branches per 128 shots), a polynomial cost (n^7 at L = 4)
that is still comparable to the chains at n = 38. What remains at K ≥ 5 is the level chains' rounds: a level
drawn by chain starts with a guessed 2 chains per stream and doubles, so a rare prefix takes several launches
of the whole lockstep pass (task A10).

## 5c. The frontier map with adaptive modes (`bench/adaptive_toll.py --map`, `img/adaptive_map.png`)

n*(m/n) for K = 0..4 on the paper's frontier axes. Since 2026-09-22 the figure joins the MEASURED crossings of
each K (solid, points) and draws the cost model dotted; the level-chain points at m/n = 1 that job 25 superseded
are no longer drawn (abort = 0 in the CSV), and the K = 4 point at m/n = 1 is the level-chain one (the K = 4
table does not fit at n ≈ 44), a lower bound on what the tables would give. Model: the K = 0 curve is the measured plain sampler
(results/keep/frontier.csv, nine ratios), each K > 0 curve is that curve shifted by −log X(K, c)/log ρ(c) with X the
cost model of §5b evaluated at n = n*(c) (Gaussian estimator of E[2^r], 4 unitaries × 4·10⁵ samples; tables
through K = 4). results/keep/adaptive_map.csv, with the measured crossings (H100, 1 sample/s, runs
20260922_074317 / _092841 / _131122) in brackets:

*Measured 10-07 (re-measurement of the headline rows): **K = 1 at m/n = 2 = 38.3** (run 20261007_022012, 512 shots, bracket
n = 38 at 1.187/s, 39 at 0.623/s; K = 0 in the same run 38.4, the canonical Fock 38.5), replacing the derived 38.4‡ below. This
bench at a fixed `--shots 128` batch reads plain-Fock throughput high (K = 0, n = 38: 1.578/s at 128 shots in runs
20260922_074317 and 20261007_005628, 1.256/s at 512 shots, 1.257/s in find_threshold) — the bracketed values of this table taken
at 128 shots can be optimistic by up to ~0.3; the K = 1 / K = 0 time ratio (same batch) is not affected. LOG 2026-10-07.*
*All quoted m/n = 2 values re-measured at 512 shots (C, 10-07, runs 20261007_030408 / _061124, original chains): **K = 1 … 6, 8
= 38.3 / 38.2 / 38.1 / 37.6 / 34.8 / 34.4 / 33.8**, replacing the bracketed m/n = 2 values below and the 37.0 / 34.0 / 34.2 / 31.6
of §5b's table (none moved down; the K = 8 toll is 4.7 sources, not ~7). The other m/n columns are not re-measured.*

| m/n | 1 | 1.5 | 2 | 3 | 4 | 6 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 42.4 | 40.0 | 38.5 | 36.8 | 36.2 | 35.3 | 34.4 |
| K = 1 | 42.3 [42.1, 40.7†] | 39.9 [39.8] | 38.4 [38.4‡] | 36.8 [36.8‡] | 36.2 [36.2‡] | 35.3 | 34.4 |
| K = 2 | 42.2 [41.6, 39.7†] | 39.9 [39.7] | 38.4 [38.2‡] | 36.7 [36.7‡] | 36.2 [35.9] | 35.3 | 34.4 |
| K = 3 | 42.1 [41.3, 38.5†] | 39.8 [39.1] | 38.3 [38.1] | 36.7 [36.6‡] | 36.1 [35.9] | 35.3 | 34.4 |
| K = 4 | 41.9 [37.1†] | 39.7 [38.1] | 38.3 [36.9] | 36.7 [36.0‡] | 36.1 [35.6] | 35.2 [34.9‡] | 34.4 |
| K = 5 | | | 37.1 [34.0] | 35.6 [32.8] | 35.1 [32.1] | | |
| K = 8 | 38.7 | 36.7 | 35.4 [31.6] | 34.1 [31.1] | 33.6 [31.0] | 32.8 | 32.0 |
| E[2^r], K = 4 | 379 | 44 | 12.3 | 4.7 | 3.1 | 2.1 | 1.4 |
| X, K = 4 | 1.31 | 1.20 | 1.14 | 1.09 | 1.06 | 1.04 | 1.02 |

(K = 8 is on the map (`--map-extra 8`): its model curve has four levels by chain, and its three measured points
(m/n = 2, 3, 4, all by the level chain, hollow) sit 3.8 / 3.0 / 2.6 under it — the rounds of the level chains,
§5b. Its 2.4-source gap to the plain curve at m/n = 12, where E[2^r] → 2, is the four full chains the model
charges for the four levels beyond the tables. K = 5 is not drawn; its model values have one level by chain.
Model values are the first-hit model of §5b from 2026-09-22; the bound model's are in adaptive_map.csv as
nstar_bound.) † the second m/n = 1 value of each row was measured with the
level chain (hollow points on the map; run 20260922_140949), out of caution about the binomial inversion at
mean occupation 1 — unfounded: the inversion sums to 1 with no negative entry to n = 38, K = 3 at m = n. The
first value is with the tables (run 20260922_162506, job 25, n = 36..44, fock 42.7 on the same fit): 42.1 /
41.6 / 41.3 for K = 1 / 2 / 3 against the first-hit model's 42.3 / 42.2 / 42.1 (the bound model said 42.0 /
40.8 / 38.8).
‡ the adaptive fit of these points was EXTRAPOLATED (the rate never fell below 1 sample/s in the run). An
extrapolated fit of the adaptive rate has too shallow a slope — its time is the plain time plus a per-branch /
per-stream overhead that fades with n (time ratio 6.2 → 1.2 from n = 28 to 34 at m/n = 6, K = 4) — and can
even cross above the plain sampler (it did: 35.7 against 35.2 at m/n = 6, 37.0 against 37.0 at m/n = 3). Such
points are quoted as n*_0 − log x_time / log ρ_0 from the plain fit of the same run and the measured time ratio
at the largest n (the bench prints both since 2026-09-22); this assumes the ratio stays at its last value, a
lower bound while it falls. Bracketed fits are quoted as they are; the two agree within 0.3 wherever both
exist. Runs 20260922_074317 / _092841 / _131122 / _140949 / _162506.)

Reading: with the first-hit model the K ≤ 4 curves stay within half a source of the plain one everywhere —
0.1 / 0.1 / 0.2 / 0.3 sources at m = n for K = 1–4 (X = 1.06 / 1.13 / 1.22 / 1.31), 0.05–0.2 at m = 2n, nothing
beyond m/n = 4 — and rise monotonically to m = n like it. The measurement agrees in shape: K = 3 gives 41.3 /
39.1 / 38.1 at m/n = 1 / 1.5 / 2, K = 1, 2 42.1 / 39.8 / 38.5 and 41.6 / 39.7 / 38.4. The turnover the earlier
model showed was an artefact of its abort factor: 2/(n − r̄) is an upper bound (one hit, uniformly placed),
loose by an order of magnitude at m = n where several photons hit and the first comes early; the measured cost
per aborted chain, in full chains, from x_time and the chains started, was 0.075 / 0.017 / 0.005 at m/n = 1 for
K = 1 / 2 / 3, n = 44 (bound 0.049), falling with K as the first-hit model has it (its charge per aborted chain
at m = n, n = 42: 0.011 / 0.0045 / 0.0017). The gap that remains between the measured and the modelled toll
— 0.2 / 0.6 / 0.8 sources at m = n, 0.05 / 0.2 / 0.2 at m = 2n for K = 1 / 2 / 3 — is the tables and the
batch rounds, per branch, not the chains. The map is measured at every cell with m/n ≤ 6 and K ≤ 4 (22 points)
and at K = 5, 8 for m/n = 2, 3, 4. K = 4 sits 0.3–1.6 under its curve everywhere (the K = 4 table, §5b), the
level-chain points 2.6–4.8 under theirs (the rounds). The K = 5 points sit 2.1 / 2.7 / 2.9
sources under the level-chain model at m/n = 2 / 3 / 4 and K = 8 3.1 / 2.2 / 2.3 — the per-branch overhead
of §5b (every level by chain is one launch per branch); they also show the large-K toll falling with m/n
far less than E[2^r] does (30.2 / 31.1 / 31.0 for K = 8 against 104 / 37 / 22 chains per sample predicted).
The batched branches of §4d recovered half of it at m/n = 2 (K = 6 / 8: 34.2 / 31.6, §5b); the K = 5 and the
m/n = 3, 4 points of K = 8 predate the batching.
Not in the model: the table cost at K = 4 and the rounds of the level chains (implementation); the per-step
cost profile is taken as ρ^k with the plain sampler's measured ρ.

## 6. Not covered

Loss and partial distinguishability on the strong side and the estimator (§7 covers the sampler), bunched
inputs, emitter (MPS) inputs, the additive-error mean-value estimator of arXiv:2509.00746 (Theorem 3) for
general product observables, a Perceval feed-forward adapter. The GPU validation of the sampler is
`bench/adaptive_frontier.py --check` / `--check-large` (2026-09-21); the imperfect sampler of §7 ran on the CPU
oracle only until run 20260922_151243.

Threshold detectors are deliberately out of this list: PNR is the setting of interest here, and a
threshold read-out belongs with the package-wide detector model, not with this engine. (On the adaptive
modes it would merge the branches that differ only by counts ≥ 1, so the policy would see fewer outcomes.)

## 7. Imperfections on the sampler: uniform loss and partial distinguishability (`sample(loss=, indist=)`)

The package-wide models (`imperfections.py`: one transmission η per photon; the x-model / random species of
docs/DISTINGUISHABILITY.md) carry over to the adaptive circuit without a new algorithm, because both are
INPUT-SIDE partitions of the photons and the feed-forward only reads counts:

* **Loss.** Uniform loss commutes with every passive element, and the adaptive detectors read what survives:
  a lossy adaptive circuit is the lossless one on the random subset of input photons that survive
  (k ~ Bin(n, η)). The policy sees the lossy counts — what it would see in the lab. Nothing else changes; the
  sampler gets cheaper (fewer photons, smaller tables, smaller r).
* **Distinguishability.** In the species picture a shot is a partition of the surviving photons into
  interfering species (an instance of ≥ 2 photons each) and singletons. Species evolve independently through
  the SAME unitaries U^{p_{≤λ}}, and the counts on the adaptive modes are sums over species, so:
  - stage 1 is run per instance: each species draws its level-λ counts from its own marginal table under
    U^{p_{<λ}} conditioned on its own prefix (`_level_tables(lam, prefix, cols, inst)`, the same graded table on
    the species' columns), or from its own forced chain beyond the tables; a singleton still in flight lands on
    a mode a of the level with |U^{p_{<λ}}[a, col]|² / (1 − Σ_{a' measured before} |U^{p_{<λ}}[a', col]|²);
    the level's counts are added over instances and singletons before the policy sees them;
  - stage 2 is one forced-prefix chain per instance (its own r_s, acceptance P_s / T_s), batched with every
    other instance and stream by `_forced`; a singleton still in flight is routed among the final modes with
    |U^p[r, col]|² renormalised. Exact: given the partition, the instances are independent boson samplers on
    the branch unitary, and conditioning each on its own adaptive counts is §4b again.

  Restarts stay at E[2^r] over all photons read adaptively, but a singleton costs no chain at all, so the
  cost of a sample is set by the largest interfering instance: the η·x law of the plain sampler (n* moves as
  n*(1) log ρ / log(1 − ηx + ηx ρ)) should carry over with r̄ counted on the surviving photons.

* **Detector inefficiency on the adaptive detectors** (not uniform loss: a photon missed there is not
  missed elsewhere) is the one place order matters — the policy would receive a thinned count while the chain
  is conditioned on the true one. Not implemented; it is a change in stage 1 only.

The lossless, indistinguishable case is the same code with one instance per stream (`partition` returns one
group), so η = 1 reduces to the lossless path exactly — the invariant every engine keeps.

Test (tests/test_adaptive.py, `_imperfect_joint`): the mixture over the 3^n role assignments (lost, interfering,
singleton) of an enumeration that walks the levels with the SLOS joint of the interfering photons and the
classical routing of the singletons — independent of `binomial_moments`, of `fock_sample` and of the species
code — against the sampler at (η, x) = (0.7, 1), (1, 0.6), (0.8, 0.7), both chains: TVD within 2.5 shot-noise
floors at 20 000 shots, photon number binomial.

Measured (H100 PCIe, run 20260922_151243, job 23: m/n = 2, K = 1, 2, 3, 128 shots, the plain sampler timed
at the same imperfection; results/keep/adaptive_imperfect_crossings.csv):

| η, x | plain, measured (η·x law from 38.5) | K = 1 | K = 2 | K = 3 | lossless toll for comparison |
| --- | --- | --- | --- | --- | --- |
| 0.9, 1 | 41.2 (41.7) | 40.5 → 40.5 → 40.6 | 39.9 → 40.1 → 40.1 | 39.6 → 40.0 → 40.2 | −0.1 / −0.1 / −0.4 |
| 1, 0.95 | 40.0 (40.0) | 39.1 → 39.1 → 39.0 | 39.0 → 39.1 → 39.1 | 38.7 → 39.0 → 39.1 | |
| 0.9, 0.95 | 42.8 (43.4; extrapolated) | 41.6 | 42.0 | 41.7 | |

(→: with the batched tables of task A8, run 20260922_191049, job 26; → →: with the batched posterior draw of
task A11 as well, run 20260923_181143, job 28: time ratios at n = 40 of 1.44 / 1.48 / 1.51 at η = 0.9 and
1.82 / 1.64 / 1.68 at x = 0.95, against the lossless 1.20 / 1.35 / 1.68 at n = 38 and 128 shots — the
imperfect sampler now costs what the lossless one costs at the same batch, to the ±15 % the timings scatter;
the crossings' remaining 0.5-source gap to the lossless toll is at the fit's resolution. The measurement
stops here; the model's toll with imperfections is the lossless one at the surviving photon number.)

The plain sampler follows the η·x law; the adaptive one pays 0.7–1.6 sources instead of the 0.1–0.4 it pays
lossless, and not through the chains — chains started per sample 2.3–3.3 / 3.4–6.1 / 6.5–11.6 for K = 1 / 2 / 3,
against E[2^r] on ~37 surviving photons of about 2.0 / 3.8 / 6.5 plus the rounding of a batch that cannot pool
(with η < 1 every shot is its own partition, so every stream is its own pool and starts ⌈1/(P/T)⌉ chains).
The extra cost is per stream and fixed in n: one graded table per stream per level (the species' columns
differ from shot to shot, so nothing is cached), and one round of `_forced` per distinct prefix length,
which is why the time ratio falls with n (8.4 → 2.1 from n = 34 to 42 at K = 3, η = 0.9) and the crossing is
implementation-limited. The algorithmic toll with imperfections is the lossless one at the surviving photon
number. Since 2026-09-22 (task A8) the tables of one level are ONE batched call, `binomial_moments_blocks` on
the stacked blocks U^{p}[modes, cols] — blocks of different unitaries, modes and photon numbers, the shorter
ones padded with zero columns (a zero column is the factor 1 + 0 and leaves its table unchanged: exact) —
chunked by MAX_MARGINAL_ENTRIES; `_level_tables_many` fills the cache for all the level's keys before the
draws. Same index_add launches as one table, B times the memory: on the CPU 1.7× faster for 64 blocks
(n = 18, L = 3), on the GPU the gain is the launch count. Measured (job 26): the tables were the small-n
cost — at n = 34, η = 0.9, K = 3 the point takes 9.4 s instead of 22.4 s — but not the cost at the crossing
(399 s instead of 416 s at n = 42): K = 2 / 3 move by 0.2 / 0.4 sources at η = 0.9 and 0.1 / 0.3 at x = 0.95,
K = 1 not at all. What remains is the number of separate lockstep passes: `_forced` groups the streams by
(photon number, prefix length), and with η < 1 the photon number varies from shot to shot (k ~ Bin(42, 0.9)
spans 34..42), so a level or the final stage is 20–30 passes of a few streams each instead of one, and a pass
of few streams costs nearly what a full batch costs. Job 27 (run 20260922_202903) tested that reading and it is wrong: at η = 0.9, K = 3 the time
ratio at 512 shots is 1.77 / 1.65 (n = 38 / 40) against 1.96 / 1.67 at 128 shots, while lossless K = 3 at 512
shots gives 1.25 / 1.27 (n = 36 / 38; 1.68 at 128 shots, job 18) — the lossless excess does amortise with the
batch (its passes are few), the imperfect one does not. The plain sampler at 512 shots costs 4.4× its
128-shot time, so the GPU is saturated and every pass costs in proportion to its streams: the imperfect excess
is work PER STREAM. The candidate is the posterior draw of the forced columns, `prefix_columns` (§4b.3): it
runs per circuit of a pass, ~10³ small launches each (n suffix tables of D × D, then n column decisions), and
with η < 1 every stream is its own circuit — 4 stages × 512 streams × rounds ≈ 3·10³ calls per point, 10–30 s
of the 154 s at n = 38 — where lossless it runs once per branch. It is now batched over the circuits of a pass
(`prefix_columns_many`, §4b.3; launcher job 28 re-measures η = 0.9 and x = 0.95 with it); the model's toll
with imperfections stays the lossless one at the surviving photon number.
