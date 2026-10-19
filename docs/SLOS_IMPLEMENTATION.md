# SLOScore — theory and implementation

> **Provenance note (2026-09-11).** This document was written for the SLOScore project and is
> kept as the theory reference. File names below are the original ones; in this repository
> `sloscore/basis.py` → `photonic_engines/slos/basis.py` (+ `basis_torch.py`), `sloscore/kernel.py`
> → `photonic_engines/slos/kernel.py`, `sloscore/converter.py` → `photonic_engines/adapters/perceval/converter.py`,
> `sloscore/reference.py` → `photonic_engines/slos/reference.py`; the `backends/sloscore_*.py`
> adapters, `SLOScore_REPORT.md` and `docs/CLIFFORD_IMPLEMENTATION.md` were
> not moved (their content is superseded by `docs/ALGORITHMS.md` and the measurements in
> `docs/TUTORIAL.md`). Mode positions are int8 up to 127 modes and int16 beyond
> (`basis.mode_dtype`), a change from the original.

*A self-contained account of the strong-simulation engine for a mathematician or computer
scientist: the physics, the algorithms with pseudocode, exactly which parts are prior art and
which are this implementation's contributions, and the optimizations derived quantitatively —
each with the operation-count or memory model that predicts the measured number. Companions:
`SLOScore_REPORT.md` (chronological measurement→decision log), `docs/CLIFFORD_IMPLEMENTATION.md`
(the sampler twin). Display equations use
LaTeX math fences (rendered by GitHub). Numeric citations refer to the References
section at the bottom.*

**Notation.** `m` modes, `n` photons; occupation vectors `s, t ∈ N^m` with `Σ_j t_j = n`;
`e_j` the unit occupation on mode `j`; `a†_j` the creation operator on mode `j`; `per(M)` the
matrix permanent; `B` the batch of unitaries; `S_k` the set of reachable k-photon states,
`E_k` the set of raising edges into layer k; `c = 8` bytes (complex64 element).

## 0. Orientation — file map (~940 lines total)

| file | lines | role | section |
|---|---|---|---|
| `sloscore/basis.py` | 165 | state enumeration + raising edges; masks prune here | §5 |
| `sloscore/kernel.py` | 307 | the SLOS recurrence (single-input, EBS, grouped) | §3, §6–§9 |
| `sloscore/converter.py` | ~220 | circuit→unitary (constant replay + trainable mesh) | §10 |
| `sloscore/reference.py` | 60 | independent permanent oracle (tests only) | §2 |
| `backends/sloscore_*.py` | ~330 | benchmark adapters + variant arms (ImmNorm, C128, Param) | §13 |

## 1. Physical setting

A passive linear-optical circuit on m modes is an m×m unitary `U`. States live in the bosonic
Fock space; the n-photon input with occupations `s` is

```math
|s\rangle = \prod_{i=1}^{m} \frac{(a^\dagger_i)^{s_i}}{\sqrt{s_i!}}\,|0\rangle ,
```

and the circuit acts on creation operators linearly — the defining property of passive linear
optics (no nonlinearity, no entangling gates) [2, 9]:

```math
\varphi(U):\quad a^\dagger_i \;\longmapsto\; \sum_{j=1}^{m} U_{ji}\, a^\dagger_j . \tag{1}
```

Everything below follows from (1) by algebra. The state space is exponential:
`|S_n| = \binom{m+n-1}{n}` for full Fock.

## 2. Amplitudes are permanents

Applying (1) to each input photon (listing occupied input modes with multiplicity as
`i_1,…,i_n`) and expanding the product:

```math
\varphi(U)\,|s\rangle \;=\; \frac{1}{\sqrt{\prod_i s_i!}}\; \prod_{k=1}^{n} \Bigl( \sum_{j=1}^{m} U_{j,\,i_k}\, a^\dagger_j \Bigr) |0\rangle . \tag{2}
```

The coefficient of the monomial `∏_j (a†_j)^{t_j}|0⟩` sums over all bijections between the n
photons and their output slots — a permanent [2]:

```math
\langle t|\,\varphi(U)\,|s\rangle \;=\; \frac{\operatorname{per}(U_{s,t})}{\sqrt{\prod_i s_i! \;\prod_j t_j!}} , \tag{3}
```

with `U_{s,t}` the n×n matrix taking `s_i` copies of column i and `t_j` copies of row j.
Computing permanents is #P-hard [4]; this hardness is the entire subject of boson sampling
[3]. The oracle in `reference.py` evaluates (3) literally per output state with Ryser's
`O(2^n n)` formula [5] — it shares no code with the kernel, which is what makes their
agreement a physical correctness gate.

## 3. The SLOS recurrence (prior art [1])

Evaluating (3) independently per output costs `|S_n|·O(2^n n)`. SLOS [1] — the algorithm
behind perceval's SLOSBackend [8], dperceval, and merlin — instead inserts one photon at a
time. Define the unnormalized intermediate after k photons, expanded over occupation
*monomials* (not normalized Fock states):

```math
\psi_k \;=\; \prod_{l=1}^{k} \Bigl(\sum_j U_{j,\,i_l}\, a^\dagger_j\Bigr) |0\rangle \;=\; \sum_{t \in S_k} c^{(k)}_t \, \prod_j (a^\dagger_j)^{t_j}\, |0\rangle .
```

Multiplying by photon k+1's operator gives the raw recurrence — the entire hot loop:

```math
c^{(k+1)}_{t+e_j} \;\mathrel{+}=\; U_{j,\,i_{k+1}} \cdot c^{(k)}_t \qquad \forall\, (t \to t{+}e_j,\; j) \in E_{k+1}. \tag{4}
```

**Algorithm 1 — batched SLOS sweep** (`FockSLOSKernel.amplitudes`; tensors on device, batch B):

```
Input:  U [B,m,m]; per-layer edge tensors parent[k], child[k], mode[k] (built once, §5)
psi <- ones [B, 1]                                  # vacuum coefficient
for k = 1 .. n:                                     # photon k enters mode i_k
    g1     <- U[:, mode[k], i_k]                    # gather   [B, E_k]
    g2     <- psi[:, parent[k]]                     # gather   [B, E_k]
    contrib <- g1 * g2                              # multiply [B, E_k]     (eq. 4)
    nxt    <- zeros [B, |S_k|]                      # allocated AFTER the peak (§7)
    nxt.index_add_(dim=1, child[k], contrib)        # scatter-accumulate
    psi    <- nxt
return psi                                          # c^(n) over S_n, repaired by (5)
```

Each edge is touched once; the monomial coefficients `c_t` are shared sub-permanents, so the
work is

```math
W \;=\; B \sum_{k=1}^{n} |E_k| \quad\text{complex multiply-adds}, \qquad
|E_k| = \sum_{t \in S_k} \#\{j : t_j > 0\} \le |S_k| \cdot \min(k, m),
```

versus `B\,|S_n|\,O(2^n n)` for per-output permanents. For full Fock, `|E_{k}| = |S_{k-1}|·m`
(every raise is valid). Everything in Algorithm 1 is autograd-native, so the sweep is
differentiable w.r.t. `U` for free — and, through §10's parametrized converter, w.r.t.
circuit parameters. TorchScript and torch.compile are deliberately absent (both measured
slower; Torchinductor lacks complex support).

## 4. Prior art vs. this implementation — the ledger of who did what

| component | prior art | this implementation's contribution |
|---|---|---|
| layered recurrence (4) | SLOS [1]; all three incumbents implement it | batched-over-B, device-resident, gather/scatter formulation; no JIT |
| deferred normalization (§6) | **dperceval's design** (adopted after reading its source) | adoption + the pass-count model predicting the ablation number; the `SC_ImmNorm` arm making the choice permanently measurable |
| mask pruning (§5) | perceval's `set_mask` prunes its tree | masks as per-layer basis constraints with the monotonicity lemma; compact-tuple state representation; child-decomposition edge build |
| EBS / superposition (§8) | merlin's concept (simultaneous components) | single sweep with a component dimension + exactly-invariant chunking |
| converter (§10) | all incumbents convert on host, gate-sequentially | ASAP layer packing + one vectorized row-update per layer on device |
| trainable conversion (§10) | merlin's converter is parameter-differentiable | `ParametrizedMesh`: same row-update machinery with live parameters + mid-mesh encoding (placement forced by the observability lemma) |
| allocation ordering (§7) | (implicit in dperceval's code) | identified as the memory-gap cause via per-op profiling; codified as an explicit contract with a peak-memory model |

## 5. Basis construction (contribution) — representation and edge enumeration

**Representation.** A k-photon state is stored as the sorted tuple of its k occupied mode
indices (a multiset of photon positions), not a length-m occupation vector. Every state
operation (hash, copy, compare) drops from `O(m)` to `O(k)`; the enumeration cost drops
correspondingly:

```math
T_{\text{basis}} \;=\; \sum_{k \le n} |S_k| \cdot O(k) \quad\text{vs}\quad \sum_{k \le n} |S_k| \cdot O(m).
```

In the low-photon/high-mode regime the ratio is `~m/n`: at n=2, m=512 the predicted ~256×
against the measured ~70× (constant factors: tuple overheads, dict churn) — direction and
magnitude consistent. Full occupation vectors are materialized only for the final layer, on
demand, for the oracle.

**Algorithm 2 — edge enumeration by child decomposition** (`basis.py`):

```
for k = 1 .. n:
    S_k <- {}                                        # dict: state-tuple -> index
    for each child C obtainable from S_{k-1}:        # via raising, mask-checked (below)
        for each DISTINCT occupied mode a in C:      # <= min(k, m) of them
            P <- C with one photon removed from a    # the parent tuple, O(k)
            emit edge (index(P), index(C), a)        # bosonic factor sqrt(C_a) recorded
                                                     # only for the immediate arm (§6)
```

Decomposing children into parents (instead of raising every parent by every mode) touches
only length-k tuples and — by the lemma below — never constructs a state that must be
discarded.

**Masks: the monotonicity lemma.** Every supported mask is a per-mode occupation ceiling
`t_j ≤ cap_j` (fock: `cap = n`; unbunched/dual-rail/sparse-blocks: `cap = 1` per slot;
tail-empty: `cap = 0` on the last two modes).

> **Lemma.** Let `M = {t : t_j ≤ cap_j ∀j}`. Then M is downward-closed under photon removal,
> and photon creation is monotone in every coordinate. Hence every ancestor (along raising
> edges) of a valid final state is valid, and pruning `S_k ∩ M^c` at every layer preserves
> all amplitudes of valid outputs exactly.
>
> *Proof.* If `t ∈ M` and `t' ≤ t` coordinate-wise then `t' ∈ M` (downward closure). A path
> from the vacuum to `t ∈ M` along raising edges has monotonically non-decreasing
> coordinates, so every intermediate `t'` satisfies `t' ≤ t`, hence `t' ∈ M`. Removing
> invalid states therefore removes no path ending in M. ∎

Consequence: masks shrink both `|S_k|` and `|E_k|` in *every* layer — compute and memory,
not output filtering. At n8/m16 dual-rail: `|S_8|: 490314 → 256`, measured kernel speedup
26.5× [MASK-PRUNING]. A non-monotone constraint (e.g. "at least one photon in mode 3")
violates downward closure and must never be added as a tree mask. Convention: masked
distributions are **raw survival** (`Σ_t p_t ≤ 1`, matching perceval; merlin renormalizes —
the one-divide bridge is proven exact in `tests/test_grad_agreement.py`).

## 6. Deferred normalization — derivation and the pass-count model

The raw coefficients of (4) are not amplitudes: since
`|t⟩ = ∏_j (a†_j)^{t_j}|0⟩ / \sqrt{∏_j t_j!}`, comparing with (2):

```math
A_t = c^{(n)}_t \frac{\sqrt{\prod_j t_j!}}{\sqrt{\prod_i s_i!}}
\quad\Longrightarrow\quad
p_t \;=\; \bigl|c^{(n)}_t\bigr|^2 \, \frac{\prod_j t_j!}{\prod_i s_i!} . \tag{5}
```

So one may run the factor-free recurrence (4) and repair once at the end (deferred — one
`[S]` buffer `∏_j t_j!` and a scalar). The textbook alternative substitutes (5) into (4) per
layer, yielding the per-edge bosonic factor:

```math
A^{(k+1)}_{t+e_j} \;\mathrel{+}=\; U_{j,i} \, \sqrt{t_j + 1} \; A^{(k)}_t . \tag{6}
```

**Why deferring wins, counted.** The hot loop is memory-bandwidth-bound; count elementwise
passes over the dominant `[B, E_k]` tensors per layer:

| pass | deferred (4) | immediate (6) |
|---|---|---|
| gather U column | 1 | 1 |
| gather parent amps | 1 | 1 |
| multiply | 1 | 1 |
| multiply by `sqrt(t_j+1)` | — | 1 |
| scatter-accumulate | 1 | 1 |
| **total** | **4** | **5** |

Predicted slowdown of the immediate arm: `5/4 = 1.25`. Measured (the `SC_ImmNorm` ablation,
cuda): 1.15–1.24×, 1.236 at the heaviest cell — the model is quantitatively right
[MEM-TRADE]. (The pre-adoption implementation was worse still, ~1.33×, because its factor
multiply materialized an additional `[B,E]` intermediate — two more memory passes.) Memory:
(6) also allocates that extra `[B,E]` transient per layer; measured, toggling normalization
moves speed and *not* peak memory — which is what refuted the old "speed↔memory trade" story.

**EBS compatibility.** The output factor in (5) depends only on `t`, never on the input
component, so for superposition inputs it must be applied AFTER the coherent sum (§8), while
the input factor `1/\sqrt{∏ s_i!}` is per-component and applied inside. Gated in
`test_sloscore_ebs.py`.

## 7. Allocation ordering (contribution) — the peak-memory model

The peak sits at the heaviest layer's multiply, where three `[B,E]` tensors coexist (two
gathers and their product). With `E* = max_k |E_k|` and `S* = |S_n|`:

```math
P_{\text{before}} \;=\; 3\,B\,E^{*}c \;+\; B\,S^{*}c \;(+\,B S^{*}\!\cdot\!4 \text{ in the epilogue}),
\qquad
P_{\text{after}} \;=\; 3\,B\,E^{*}c .
```

The `B S* c` term was the next-layer buffer allocated *before* the peak; the fix allocates it
after (Algorithm 1's comment) and makes the epilogue multiplies in-place. At n8/m16, B=64:
`E* = |S_7|·m = 170544·16 = 2.73M`, so `3BE*c = 4.19` GiB, `BS*c = 0.234` GiB, epilogue
`0.117` GiB — predicted recovery ≈ 0.35 GiB of the measured 0.41 GiB gap vs dperceval; the
cuda probe confirms peak ratio 1.093 → 1.021 (`results/kernel_memory_probe.txt`,
H-MEM-ALLOC-ORDER). Invisible to correctness tests — re-run the probe after touching the loop.

## 8. EBS — superposition inputs in one sweep

For `|Ψ⟩ = Σ_c w_c |s_c⟩` (GHZ etc.), linearity of φ(U) gives coherent addition *of
amplitudes*:

```math
A_t \;=\; \sum_{c=1}^{K} w_c\, A^{(c)}_t, \qquad p_t = |A_t|^2 \cdot \prod_j t_j! .
```

The naive method runs Algorithm 1 K times (K sequential passes, K× the kernel-launch count).
`SuperpositionSLOSKernel` carries a component dimension `kc` through the same sweep — tensors
`[B, kc, E_k]`, one `index_add_` per layer — identical FLOPs, launches divided by K, GPU
occupancy multiplied. Chunking caps `kc ≤ cs` per sweep; since

```math
\sum_{c=1}^{K} w_c A^{(c)}_t \;=\; \sum_{\text{chunks}} \; \sum_{c \in \text{chunk}} w_c A^{(c)}_t ,
```

the result is *exactly* invariant to the chunk size — only peak memory changes. Measured:
13.3× over the naive loop at K=84, ~1.7× better scaling than merlin's EBS [EBS-K84].

## 9. Grouping — provably output-shaping, not compute

A grouping partitions final states into bins, reporting `P_b = Σ_{t∈b} p_t`. Because bins are
arbitrary sets of *final* states, each amplitude must exist before its probability can be
summed — unlike masks there is no monotone structure to exploit mid-recurrence. Hence
`GroupedSLOSKernel` runs the full sweep and scatter-reduces at the end: the returned vector
shrinks; work and peak do not (measured ~1×, H-GROUPING-STRUCTURAL).

## 10. The converter — constant replay and the trainable mesh

**Composition.** A circuit is a gate sequence; composing left-to-right is `U ← G_emb · U`. A
2-mode gate `g` on modes (a,b) touches two rows:

```
U[a,:] <- g00*U[a,:] + g01*U[b,:]
U[b,:] <- g10*U[a,:] + g11*U[b,:]
```

**Algorithm 3 — plan build + replay** (`CircuitConverter`; prior art converts host-side,
gate-sequentially):

```
compile (once):  greedily pack gates into layers L_1..L_D of DISJOINT mode pairs
                 (disjoint => commuting => one layer applies in any order)
convert (timed): U <- I_m
                 for d = 1 .. D: apply ALL gates of L_d as ONE vectorized row update
```

Work is unchanged (`O(G·m)` for G gates), but sequential kernel launches drop from `O(G)` to
`O(D)`, `D ≈` circuit depth (`~2m` for a Reck/Clements mesh [6, 7] vs `G = O(m²)` gates) —
on GPU, where launch latency dominates small ops, this is the measured 4.3–47× conversion
win [CONV-SMALL, CONV-M128, PAIRED-BACKEND-TASK].

**The trainable mesh** (`ParametrizedMesh`, contribution). Constant replay bakes gate
*values*; a trainable layer needs gates as functions of live parameters. The mesh is

```math
U(\theta, x) \;=\; \Bigl[\prod_{l > L/2} T_l(\theta_l)\Bigr] \; D(x) \; \Bigl[\prod_{l \le L/2} T_l(\theta_l)\Bigr],
\qquad
T_l = \bigoplus_{(a,b) \in \text{pairs}(l)} \begin{pmatrix} \cos\theta & i\sin\theta \\ i\sin\theta & \cos\theta \end{pmatrix},
\quad D(x)_b = \operatorname{diag}(e^{i x_{b1}},\dots,e^{i x_{bm}}),
```

with shared trainable `θ ∈ R^{L×⌊m/2⌋}` and per-sample encoding `x ∈ R^{B×m}` — the
QuantumLayer contract shape. Same row updates, entries computed from `θ` inside autograd.

> **Observability lemma (why the encoding sits mid-mesh).** For a phase screen at the input,
> `U D(x)` has `(U D(x))_{s,t}`-submatrices whose columns are scaled by `e^{i x_{i}}`, so by
> (3) `per` acquires the factor `e^{i \sum_j s_j x_j}` — a global phase per basis state:
> `p_t` is independent of x. Symmetrically for an output screen (`row` scaling ⇒ factor
> `e^{i \sum_j t_j x_j}`). A screen is observable only *between* interfering layers. ∎
>
> This was caught empirically by the per-sample-variation gate (batch outputs identical to
> float noise) before it was proven — the gates now test probabilities, not `|U|` entries.

Cost of being trainable, measured (`results/param_convert_probe.txt`): parametrized forward
0.45–0.49× replay per call (and it builds B *distinct* unitaries where replay copies one;
the depth difference L=m vs D≈2m accounts for the <1 ratio), training-mode fwd+bwd 1.6–1.9×
replay. Autograd requires a cloned `[B,m,m]` per layer for backward — `O(L·B·m²·c)` graph
memory; checkpointing is the known lever if it ever dominates.

## 11. Complexity and memory summary

```math
T_{\text{kernel}} = O\!\Bigl(B \sum_k |E_k|\Bigr), \quad
T_{\text{convert}} = O(G\,m) \text{ in } O(D) \text{ launches}, \quad
P_{\text{peak}} = 3\,B\,E^{*}c \;+\; O(\text{layers } B S_k c),
```

with masks shrinking `S_k, E_k` per the lemma, batch B amortizing launches, and the sampler
(other document) covering the regime where `|S_n|` itself is unaffordable.

## 12. Code ↔ math mapping

| math | code |
|---|---|
| `i_1..i_n` | `FockSLOSKernel.input_modes` / `circuit_utils.occupation_list` |
| edge `(t, t+e_j, j)`, eq. (4) | `basis.FockLayers.edges[k]` → `_parent/_child/_mode[k]` |
| `c^{(k)}` | `psi` `[B, S_k]` in `amplitudes()` |
| `√(t_j+1)` of (6), immediate arm only | `basis` edge factor → `_factor[k]` |
| `∏_j t_j!`, `1/√(∏ s_i!)` of (5) | `output_prob_norm` `[S]`, `_input_norm` |
| `w_c`, chunked sum (§8) | `SuperpositionSLOSKernel.amplitudes(weights, chunk_size)` |
| `per(U_{s,t})` oracle (3) | `reference.reference_probs` (Ryser [5]) |
| `T_l`, `D(x)` (§10) | `ParametrizedMesh.unitary` (mid-mesh screen at `l = ⌊L/2⌋−1`) |

## 13. Correctness chain

kernel == permanent oracle, all masks · kernel == perceval SLOSBackend; masked == `set_mask`
· basis == brute-force enumeration · EBS == independent coherent reference AND merlin EBS;
chunk-invariant · converter == `compute_unitary` · dtype arms == parents to c64 ε ·
cross-kernel *gradient* agreement (same `U(θ,x)` into SC/merlin/dperceval kernels, `∂L/∂θ`
agree ≤ 2·10⁻⁴) · finite-difference validation of end-to-end gradients (SC, merlin kernel,
MQL) · drop-in parity per MQL capability · convention bridges pinned (unbunched default;
renormalization) · at scale: per-rung `mass_ok` [CEILING-MODES], per-row output digests.
NOT verified (honest edges): GHZ amplitude values (mass only), correctness beyond m=512
other than mass, sampler correlations above n≈6–8.

## 14. Maintainer invariants and the MerLin path

Do not break: (1) the §6 normalization contract and its EBS ordering; (2) the §5 monotone-mask
assumption in the edge builder; (3) the §7 allocation ordering (invisible to correctness
tests — use the memory probe); (4) `SC_ImmNorm` must stay output-identical to SC;
(5) no TorchScript/torch.compile without re-measuring; (6) the `get_layers` cache is cleared
between ceiling-probe subjects (`main._release_probe_memory`) — keep that call; (7) the
encoding stays mid-mesh (observability lemma). MerLin integration: the engine consumes
tensors and is frontend-agnostic; the drop-in requirements (unchanged API and semantics —
including merlin's renormalization convention and unbunched default — performance-only
delta) are executable gates in `tests/test_grad_agreement.py`; the `param:` paired
population (SC_Param vs MQL) is the claim-grade verdict.

## References

[1] N. Heurtel, S. Mansfield, J. Senellart, B. Valiron, *Strong Simulation of Linear Optical
    Processes*, Computer Physics Communications 291, 108848 (2023); arXiv:2206.10549.
[2] S. Scheel, *Permanents in linear optical networks*, arXiv:quant-ph/0406127 (2004).
[3] S. Aaronson, A. Arkhipov, *The Computational Complexity of Linear Optics*, Proc. STOC
    2011, pp. 333–342; arXiv:1011.3245.
[4] L. G. Valiant, *The complexity of computing the permanent*, Theoretical Computer Science
    8(2), 189–201 (1979).
[5] H. J. Ryser, *Combinatorial Mathematics*, Carus Mathematical Monographs 14, MAA (1963).
[6] M. Reck, A. Zeilinger, H. J. Bernstein, P. Bertani, *Experimental realization of any
    discrete unitary operator*, Physical Review Letters 73, 58 (1994).
[7] W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, I. A. Walmsley,
    *Optimal design for universal multiport interferometers*, Optica 3(12), 1460 (2016).
[8] N. Heurtel et al., *Perceval: A Software Platform for Discrete Variable Photonic Quantum
    Computing*, Quantum 7, 931 (2023); arXiv:2204.00602.
[9] M. C. Tichy, *Interference of identical particles from entanglement to boson-sampling*,
    Journal of Physics B 47, 103001 (2014); arXiv:1312.4266.
[10] A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning
    Library*, NeurIPS 32 (2019) — the autograd machinery Algorithm 1 and §10 rely on.
