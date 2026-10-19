# Additive-error estimation for emitter-state inputs: Gurvits beyond product states

`photonic_engines/slos/estimate.py` (`EmitterGurvitsEstimator`), tests `tests/slos/test_estimate.py`,
tutorial §5b. Status 2026-09-17: theorem proved for the Gaussian sampler (≤ 1 photon per input
mode, any output pattern, projected or traced emitter); Rademacher variant verified numerically;
overlap extension argued.

## 1. Statement

Setting. An `EmitterMPS` input over m modes,

    |ψ⟩ = Σ_x ψ(x) |x⟩,   ψ(x) = ⟨f| M^(n)_{x_n} ⋯ M^(1)_{x_1} |s0⟩,   x ∈ [L]^n,

photon k in mode c_k(x) = modes[k][x_k], all (k, l) modes distinct, so every component has at
most one photon per mode. Bond dimension χ. Output occupation t over m modes, Σ t_j = n, rows
r_1 … r_n (mode r repeated t_r times), t! := ∏_j t_j!. The amplitude is

    A_t = ⟨t|U|ψ⟩ = (t!)^{-1/2} Σ_x ψ(x) Perm(U[r, c(x)])                                  (1)

— a coherent sum of L^n permanents; the engines evaluate it exactly at cost ≥ χ·|S_n| (strong)
or sample from |A_t|² at cost ~ n 2^n per shot (docs/EMITTER_SAMPLER.md).

**Theorem (pure emitter, Gaussian sampler).** Let x ∈ ℂ^m have i.i.d. standard complex Gaussian
entries, γ(x) its density, and

    Φ(x) = ⟨X|ψ⟩ = Σ_x ψ(x) ∏_k conj(x_{c_k(x)}),     |X⟩ := ⊗_k Σ_l x_{modes[k][l]} |l⟩,
    g_t(x) = ∏_{i=1}^n (U x)_{r_i}.

Then

  (i)   Σ_x ψ(x) Perm(U[r, c(x)]) = E_γ[ Φ(x) g_t(x) ];
  (ii)  q(x) = γ(x)|Φ(x)|² / ‖ψ‖² is a probability density, and under q the estimator
        Z(x) = ‖ψ‖² g_t(x) / conj(Φ(x)) satisfies E_q[Z] = Σ_x ψ(x) Perm(U[r, c(x)]) and
        E_q|Z|² = ‖ψ‖² · t!  — for every unitary U and every ψ;
  (iii) q can be sampled, and Φ, ‖ψ‖² evaluated, in O(n L χ² + n L³) arithmetic per sample;
        g_t costs one matrix–vector product, O(m²) (O(m n) for the rows needed).

Hence A_t is estimated to additive error ε with O(‖ψ‖² t! / ε²) samples, i.e. the same
precision-per-shot as the hardware and a per-sample cost polynomial in n, m, χ.

**Corollary (traced emitter).** For `final=None` the photonic state is the mixture
ρ = Σ_r |ψ_r⟩⟨ψ_r| over an orthonormal final basis, p(t) = Σ_r |A_{t,r}|². With z a standard
complex Gaussian vector on the emitter (E[z z†] = I), p(t) = E_z |A_{t,z}|², where A_{t,z} is the
amplitude of the pure (unnormalised) state ψ_z with final vector z. Two independent draws of Z for
the same z give an unbiased estimate of |A_{t,z}|² with second moment ‖ψ_z‖⁴ t!², and
E_z ‖ψ_z‖⁴ = 1 + Tr ρ_e² ≤ 2 (ρ_e the emitter's reduced state). So p(t) is estimated to ±ε with
O(t!²/ε²) samples.

## 2. Proof

(i) Expand g_t(x) = Σ_{c ∈ [m]^n} ∏_i U_{r_i c_i} x^c with x^c = ∏_i x_{c_i}. Φ(x) is a linear
combination of conj(x^{T}) for the *sets* T = {c_1(x), …, c_n(x)} of n distinct modes. For
independent standard complex Gaussians, E[conj(x^T) x^c] = ∏_a E[conj(x_a)^{T_a} x_a^{c_a}] is
zero unless every mode appears equally often in T and in c (E[conj(x)^p x^q] = p! δ_{pq}), and
equals 1 when it does since T has multiplicity one everywhere. So the multiset c must be a
permutation of the set T, and Σ over those n! orderings of ∏_i U_{r_i c_i} is Perm(U[r, T]) —
with rows repeated when t has collisions, which is the permanent in (1). Summing over x with
weights ψ(x) gives (i). (Rademacher variant: x_a ∈ {±1} i.i.d., no conjugation, x_a² = 1; the
monomials of g_t with a repeated mode drop to degree < n and cannot match a degree-n monomial of
Φ, so the same pairing argument holds. What changes is only the second moment, §3.)

(ii) ∫γ|Φ|² = Σ_{x,x'} ψ(x)ψ(x')* E[conj(x^{T(x')}) x^{T(x)}] = Σ_x |ψ(x)|² = ‖ψ‖², so q is
normalised. E_q[Z] = ∫ γ |Φ|²/‖ψ‖² · ‖ψ‖² g_t/conj(Φ) = ∫ γ Φ g_t = (i). Second moment:
E_q|Z|² = ‖ψ‖⁴ ∫ γ |Φ|²/‖ψ‖² · |g_t|²/|Φ|² = ‖ψ‖² E_γ|g_t|². Since x is standard complex
Gaussian and U unitary, y = Ux is standard complex Gaussian, and E ∏_i |y_{r_i}|² = ∏_j E|y_j|^{2t_j}
= ∏_j t_j! = t! (independent coordinates, E|y|^{2p} = p!). The state, the unitary and the
choice of rows do not enter.

(iii) Write w_k = (x_{modes[k][l]})_{l} ∈ ℂ^L, so Φ = Σ_x ψ(x) ∏_k conj(w_k[x_k]) and
q(x) = γ(x)|Φ(x)|²/‖ψ‖² depends on x only through (w_1, …, w_n) — the modes not used by the
state stay standard Gaussian. Sequential sampling: the marginal of w_1 is

    ∫ ∏_{k≥2} γ(w_k) |Φ|² dw_{≥2} = γ(w_1) Σ_{l,l'} conj(w_1[l]) ρ_1[l,l'] w_1[l'] = γ(w_1) ⟨w_1|ρ_1|w_1⟩,

because ∫ γ(w)|w⟩⟨w| dw = I on each later site, where ρ_1 = Tr_{2..n}|ψ⟩⟨ψ| is the reduced
(unnormalised) state of event 1. Diagonalise ρ_1 = Σ_j λ_j |e_j⟩⟨e_j|: draw j with probability
λ_j / Σλ, then w_1 = c e_j + Σ_{i≠j} c_i e_i with c_i standard Gaussian and c distributed as
γ(c)|c|² (|c|² ~ Gamma(2,1), uniform phase). Conditioning on w_1 replaces |s0⟩ by the left
vector a_1 = Σ_l conj(w_1[l]) M^(1)_l |s0⟩, and the argument repeats with the reduced state of
event k in the conditioned state, ρ_k[l,l'] = (M^(k)_{l'} a_{k-1})† R_{k} (M^(k)_{l} a_{k-1}),
where R_k = Σ_{x_{k+1..n}} (M^(n)_{x_n} ⋯ M^(k+1)_{x_{k+1}})† |f⟩⟨f| (⋯) are the right
environments, computed once per f in O(n L χ³) by the recursion R_{k-1} = Σ_l M^(k)_l† R_k M^(k)_l.
The chain of conditionals reproduces q exactly. At the end Φ = ⟨f|a_n⟩ and ‖ψ‖² = ⟨s0|R_0|s0⟩.
Per sample: n steps of L χ² (M_l a) + L² χ (ρ_k) + an L×L eigenproblem. In the Rademacher
variant w_k ∈ {±1}^L and ⟨w|ρ_k|w⟩ is enumerated over the 2^L sign vectors — no eigenproblem.

Corollary: ∫γ(z)|z⟩⟨z| dz = I gives Σ_r |A_{t,r}|² = E_z |A_{t,z}|² directly; ‖ψ_z‖² = ⟨z|ρ_e|z⟩
with ρ_e = Σ_x M_x|s0⟩⟨s0|M_x† (trace 1), and E_z ⟨z|ρ_e|z⟩² = (Tr ρ_e)² + Tr ρ_e². The
product of two independent Z's for the same z is unbiased for |A_{t,z}|² and has second moment
(‖ψ_z‖² t!)² by independence. ∎

## 3. Rademacher sampler

With x ∈ {±1}^m, (i) holds and E_q|Z|² = ‖ψ‖² E_x ∏_i |(Ux)_{r_i}|². This second moment is
not state-independent in law only through U (it is still independent of ψ) and no closed form
is available; measured on Haar U with m = 2n it is 0.52, 0.32, 0.24, 0.12, 0.09, 0.04, 0.014,
0.007, 0.002, 2·10⁻⁴ for n = 4, 6, 8, 10, 12, 16, 20, 24, 30, 40 (`g_scan.csv` of the
exploration notes), ≤ 1 for Fourier and Hadamard U, and exactly 1 in the worst structured case
found (independent dual-rail beam splitters with all even output modes: value 2^n with
probability 2^{-n}). Conjecture: E_x ∏_i |(Ux)_{r_i}|² ≤ t! for every unitary. In practice the
Rademacher sampler is 2–10× below the Gaussian bound and needs no complex Gaussian draws.

## 4. Numbers

`tests/slos/test_estimate.py`, 6·10⁴ shots, Haar U, m = 2n, against `EmitterSLOSKernel`:
projected and traced caterpillar (cluster, GHZ, n = 5), random emitters (n = 4, L = 3, χ = 3
projected; L = 2, χ = 2 traced), one collision-free and one bunched output each. All estimates
within 5σ of the strong simulation (typical |z| < 2); Gaussian second moment of the amplitude
estimator equal to ‖ψ_f‖² to < 15 % (0.49–0.52 for the projected cluster, whose projection
weight is ½; 1.5 for the random emitter with ‖f‖² ≠ 1); Rademacher below it (0.07–0.15 for the
caterpillars). Traced cluster probability: second moment 0.73 (Gaussian), 0.06 (Rademacher).

The estimator says nothing about *sampling*: it gives A_t ± ε at ~1/ε² cost, exactly what N
shots on the device give (p ± √(p/N)). Its value for the library is (a) an O(poly) oracle for
single probabilities, marginals and overlaps at any n where the exponential engines stop, and
(b) the statement that the emitter modality adds hardness to sampling only, not to
probability estimation — see the frontier discussion in the paper.

H100 timings (run 20260917_110557, `bench/estimate_time.py`, 1e5 shots): Gaussian 7.5–27 µs per
shot at χ = 2 and 20–80 µs at χ = 256 for n = 10–40; Rademacher 0.6–2.4 µs and 9–36 µs; peak
< 0.8 GB. Heavy tails: Z is a product of n factors, the median of |Z|² decays like e^{−0.58 n}
while E|Z|² stays at ‖ψ‖² t!, so the sample variance underestimates the true one at large n
(empirical second moment 0.08 → 2·10⁻⁵ from n = 10 to 40 at 1e5 shots, theorem value ½).
`Estimate.stderr_bound` reports the theorem's √(‖ψ‖² t!/N) (probability: ‖ψ‖² t!/√N), valid by
Chebyshev whatever the tails; median-of-means gives exponential confidence at the same cost.

## 5. Extensions (argued, not implemented)

- **Overlaps between two emitter states**, ⟨φ|U|ψ⟩: Σ_t φ(t)* g_t(x) is itself the contraction
  of φ with per-mode vectors (1, (Ux)_a), O(n L χ_φ²); sample x from q_ψ, average
  ‖ψ‖² G_φ(Ux)/conj(Φ_ψ(x)); second moment ‖ψ‖² ‖φ‖². Kernels K = |⟨ψ(a)|U†U'|ψ(b)⟩|² between
  emitter states are therefore classically estimable.
- **Inputs with several photons per mode** (general `FockSuperposition` with an MPS structure):
  the Gaussian pairing gives E[conj(x)^p x^q] = p! δ_{pq}, so Φ must carry 1/p! per repeated
  mode and the second moment picks up the input multiplicities — the Aaronson–Hance factors.
- **Expectation values** of product observables of bounded 2-norm (Lim–Oh Thm 1 for product
  inputs): same route, not written out.
- **Feed-forward**: after a spin measurement conditioning U on the outcome, the input to the
  next stage is no longer a fixed state; (i) fails as stated. Bounded adaptivity can be handled
  by enumerating outcomes; unbounded adaptivity is where estimation becomes hard (Oh–Lim 2025,
  Monbroussou et al. 2025).

## 6. Sampling hardness: nothing lost (the Aaronson–Arkhipov machinery transposes)

The estimator of §1 removes the hope of a *harder-than-Fock estimation* problem; it leaves the
sampling problem exactly where Aaronson–Arkhipov (AA) leave Fock boson sampling. For a fixed
emitter state ψ (label-diagonal, ≤ 1 photon per mode, some label word x₀ with ψ(x₀) ≠ 0) write

    F_ψ(X) = Σ_x ψ(x) Perm(X[·, c(x)]),   X ∈ ℂ^{n×m} the rows s of the interferometer,

so that A_t(U) = (t!)^{-1/2} F_ψ(U[s, ·]).

**Proposition.**
(a) *Worst case.* For any n×n matrix B and any m ≥ 3n there is an m×m unitary U such that
F_ψ(U[s,·]) = ψ(x₀) εⁿ Perm(B) with ε = 1/‖B‖: put εB in the columns c(x₀), zeros in the n
"wrong-label" columns {modes[k][l] : l ≠ x₀_k}, and complete the n rows to an isometry with the
remaining m − 2n ≥ n columns (possible because the rows of εB have norm ≤ 1). Every Perm(X[·, c(x)])
with x ≠ x₀ then contains a zero column, so only x₀ survives. Hence computing A_t exactly, or to
any multiplicative error, is #P-hard (Valiant; AA Thm 4.3 for the multiplicative version), for
every fixed ψ.
(b) *Average case, exact.* F_ψ is a polynomial of degree n in the entries of X. Lipton's
interpolation (AA §4, Berlekamp–Welch on the line X + λY, Y Gaussian) turns an algorithm that
computes F_ψ exactly on a 1 − 1/(3n) fraction of Gaussian matrices into one that computes it
everywhere; with (a), exact computation of F_ψ on Gaussian inputs is #P-hard on average. The
second moment is E_X |F_ψ(X)|² = n! Σ_x |ψ(x)|² = n! ‖ψ‖² (different label words use different
columns, whose entries are independent and centred; each surviving term is E|Perm|² = n!) — the
same scale as the Fock permanent, which is what the hiding step (Haar U with m ≫ n², submatrix
≈ Gaussian/√m) and the anticoncentration conjecture need.
(c) *Conjectures.* The two AA conjectures transpose verbatim: emitter-GPE^× (approximating
|F_ψ(X)|² to multiplicative error 1/poly with probability 3/4 over Gaussian X is #P-hard) and
emitter-PACC (Pr_X[|F_ψ(X)|² < δ n! ‖ψ‖²] ≤ poly(δ)); the Porter–Thomas measurements of the
application study (Z2 ≈ 2 for pure emitter inputs, 1 + 1/χ_eff traced, flat in n) are the
numerical support for the second. Under them, AA's Theorem 1.3 gives: an efficient classical
approximate sampler for the emitter state ψ through Haar interferometers (m ≫ n²) collapses the
polynomial hierarchy — by the same Stockmeyer argument, unchanged.

What is *not* inherited by inclusion: for a fixed entangled ψ, the family {(ψ, U)} does not
contain Fock boson sampling as a sub-instance (linear optics cannot disentangle the dual-rail
qubits), which is why (a) uses the wrong-column construction rather than the register-product
special case. And the frontier numbers (n* = 30 against 38.5) are the cost of the best *known*
algorithm — the generalised chain rule of docs/EMITTER_SAMPLER.md — not a lower bound.

Summary of the modality's complexity status after this note: single probabilities, overlaps and
bounded expectation values — polynomial (§1–3, same as Fock); exact sampling — #P-hard in the
worst case and, for exact computation of the amplitudes, on average (Proposition); approximate
sampling — hard under the two transposed AA conjectures, with anticoncentration measured; the
practical frontier — 8 photons ahead of the Fock sampler at equal rate.

## 7. References

Gurvits, MFCS 2005 (permanent to additive error). Aaronson–Hance, QIC 14, 541 (2014),
arXiv:1212.0025 (repeated rows/columns). Lim–Oh, arXiv:2502.12882 (2025): Gurvits for arbitrary
*product* inputs, expectation values and hafnians. Van den Nest, QIC 11, 784 (2011),
arXiv:0911.1624: computationally tractable states (MPS included) — the class this note extends
Gurvits to. Schön et al., PRL 95, 110503 (2005): sequential generation = MPS.
