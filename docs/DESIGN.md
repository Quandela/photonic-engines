# Design notes

## 1. Why one package

Before this repository the best strong-simulation kernel lived inside a benchmark harness
(`benchmarking`: SLOScore, torch, differentiable, batched, masks, EBS) and the best sampler in
a standalone study (`gpu-clifford`: Triton/CUDA Clifford & Clifford, ~4000 samples/s at
n = 24, m = 48 on one H100). They shared no conventions (U layout, input spec, output basis,
dtype/device/batch), no oracles, and three copies of the Fock-basis code once `slos_mpi` is
counted. Adding emitter (MPS) inputs touches both engines and needs a common input-state
abstraction, so both moved here. `benchmarking` measures the package; Perceval and
MerLin consume it through `adapters/`.

Conventions: `U` is a torch tensor `[m, m]` or `[B, m, m]` (batch of circuits); inputs are
`states.py` types; outputs are probabilities over `slos.basis` final states (strong) or
occupation rows `[S, m]` (sampling); everything device-resident.

## 2. Emitter (MPS) inputs — the caterpillar target

Physics covered: one emitter, one stage, all photons through one interferometer, emitter
projected or traced at the end, no detection-conditioned feedforward inside the stage. This is
the caterpillar setting of Nature Comm. 16 (2025) s41467-025-59693-3 and what makes the
photonic state a matrix product state of bond dimension chi (= 2 for a spin):

    |psi> = sum_x <f| M^(n)_{x_n} ... M^(1)_{x_1} |s0> |x>,   M_R = |up><up| U_k,  M_L = |down><down| U_k

### 2.1 Strong simulation: bond-carrying SLOS

Keep the emitter index in the SLOS state: `|Phi_k> = sum_s |s> (x) |c_s>` with chi photonic
amplitude vectors over `F_m^k`, and per photon

    |Phi_k> = sum_l ( M^(k)_l (x) A_{modes[k][l]} ) |Phi_{k-1}>,    A_p = sum_i U[i,p] a_i^dagger

i.e. an L x chi x chi mixing of the vectors combined with L creation operators — L·chi gathers
per target instead of one, chi x memory. At the end project (`<f|Phi_n>`) or trace
(`sum_s |c_s|^2`). Cost is set by chi, not by the L^n components of the expanded state; the
existing `SuperpositionSLOSKernel` (EBS) is the L^n route and remains the small-n oracle.
Running from every initial basis vector gives the stage Kraus operators
`K_t = <t| Phi_n>` (chi x chi) needed by a feed-forward stage simulator.

Status: implemented in `slos/emitter.py` (`EmitterSLOSKernel`: probs / amplitudes /
bond_amplitudes / kraus, batched over circuits, differentiable in U and in the emitter
operators). Validated at n <= 4 against the EBS expansion and an independent permanent sum,
all masks, traced = sum of projections, Kraus resolution of identity for isometric emission.
CPU cost ~8x the Fock `FockSLOSKernel` at n = 8, m = 16 (L*chi = 4 gathers + chi x chi matmuls per
layer, memory-bound); the EBS expansion is 20x slower already at n = 6 and grows as 2^n.

**Ragged label counts (2026-10-21).** Event k may have its own number of labels L_k (a product
register of sources of different dimensions: a d = 1 single-photon source emits one label, a
spin two). No new field: L_k = len(modes[k]) (1 <= L_k <= L), `modes[k]` lists the real labels
only (so mode distinctness is checked on real labels), `ops` keeps its [n, L, chi, chi] shape with
L = max L_k and the padded slots `ops[k, L_k:]` exactly zero (validated). Helpers `labels`,
`is_ragged`, `modes_padded()` (a padded slot repeats modes[k][0]), `label_mask()`;
`EmitterMPS.ragged(ops_list, modes, s0)` builds one. The recurrence is unchanged:

    in[k]  = modes_padded()[k]                                  # [L]
    for k in 0..n-1:
        C[j] = sum_{l < L} U[j, in[k][l]] M^(k)_l                # a padded M is 0: adds exactly 0
        for each raised mode j: psi_next[:, :, child] += C[j] @ psi[:, :, parent]

so a ragged chain costs what its padded-to-L_max twin costs (one small einsum per layer), and
much less than the dummy-mode workaround, which adds sum (L - L_k) modes to the basis. With an
`ops_tensor`, the kernel multiplies it once by the constant label mask, so padded slots get zero
gradient. `fold`, `expand` and the x-model (`kraus_species`) follow the per-event counts. The
samplers (`clifford.emitter_sample`, all backends), `EmitterGurvitsEstimator`,
`BinEmitter.from_mps` and `LoopRegisterKernel(emitter=)` refuse ragged chains with
NotImplementedError; strong simulation only. Tests: `tests/slos/test_emitter_ragged.py` (vs the
dummy-mode padding and vs permanents of the expanded label words, 1e-12).

### 2.2 Sampling: generalized Clifford chain rule

Clifford's proof needs a uniformly random photon order sigma and, at step k, the LO probability
of the reduced state of the photons in `T = sigma(1..k)`. The k-photon marginal lemma holds for
entangled inputs with `rho_T = Tr_{T^c} |psi><psi|` (orthogonality of the traced photons'
input slots kills cross terms; partial traces compose), so the algorithm is Clifford 2018/2020
with two changes.

Amplitudes carry chi x chi matrices: the Glynn sum becomes
`amp = sum_delta sign(delta) <f| prod_j (sum_l M_l s_{j,l}(delta)) |s0>` with one column sum
per (photon, label); Laplace expansion along the candidate output row still works, the
cofactors become chi x chi matrices `K_c = sum_delta sign(delta) p_c(delta) q_c(delta)^T`
(forward/backward vector sweeps), and each of the m candidates costs `sum_c Tr(N_c(rho) K_c)`.
The repeated-rows collapse of the 2020 form acts on delta and carries over. Per-term work is
about 4x the Fock kernel.

Mixed reduced states are the real cost: each gap of not-yet-placed photons in the chain is a
qubit channel on the emitter (Kraus rank <= 4, <= 2^len), so `rho_T = sum_e |psi_e><psi_e|`
with `R_k = prod_gaps rank <= min(2^{n-k}, 4^{#gaps})` pure MPS terms, all sharing the column
sums (an extra lane dimension). Cost per step ~ `k 2^k R_k`, ~ `k 2^n` for most k, total
~ `n^2 2^n / 4` against `n 2^{n-1}` for Fock inputs: ~n/2 overhead, about 4 photons lower on
the frontier at equal rate. Structured states collapse `R_k` (GHZ: 2 regardless of gaps), so
gap-channel ranks are computed by SVD, not assumed. No way around the factor was found:
uniform sigma is forced by the proof and `sum_e |sum_delta ...|^2` cannot be collapsed into a
single Glynn sum (the doubled delta, delta' sum is 4^k).

Alternative for n > 30: Metropolised independent sampling (Neville et al.) needs only full-n
probabilities, which for a pure MPS cost `n 2^n` — no `R_k` overhead — but is MCMC, not exact.

Status: derivation, reference (`clifford/emitter_reference.py`), batched torch implementation
(`clifford/emitter_sampler.py`) and measurements in `docs/EMITTER_SAMPLER.md`; validated
against the strong simulation. Triton kernels done (`clifford/emitter_triton.py`,
`emitter_triton_seq.py`; 11 ms per sample at n = 24 on an H100, docs/EMITTER_SAMPLER.md).

### 2.3 Validation plan

Brute-force amplitudes (`EmitterMPS.expand()` + permanents) at n <= 6 on linear cluster / GHZ /
RLC; reductions: product state -> Fock sampler with rotated U; GHZ -> mixture of two Fock
inputs against the existing sampler; bond-carrying SLOS vs `SuperpositionSLOSKernel` on the
expansion; an independent stage simulator for the Kraus operators.

### 2.4 Stage kernel: an emitter chain next to Fock photons (2026-10-21)

A stage of a hybrid spin-photon program applies the emitter chain to a Fock state |s> of the
stage's modes (heralded or carried photons). `StageChain` (`states.py`) describes the chain in the
event model of a stage simulator: per event k its modes P_k and the Fock word of each label
(single-photon words only for now; `StageChain.from_mps`, `StageChain.empty` for a photons-only
stage). `slos/stage.py` `StageKernel(chain, inputs=[B, m])` gives, per input b and output pattern t,
the chi x chi operator K_t^(b) (`kraus`), or `probs` / `bond_amplitudes`, one group per input photon
number N:

    for each group N (inputs b with |s_b| = N), basis get_layers(N + n, m):
        f_b = prod_p A_p^{s_p} / sqrt(s!) |0>             # N scalar raises, coefficients U[j, p]
        psi[b, i] = e_i (x) f_b                           # bond axis Q = B_N * chi: (input, initial vector)
        for k in 0..n-1:                                  # events on layers N + k -> N + k + 1
            C[j] = sum_l U[j, in[k][l]] M^(k)_l
            for each raised mode j: psi_next[:, :, child] += C[j] @ psi[:, :, parent]
        K_t^(b)[f, i] = psi[b, i][f, t]                   # deferred norm: physical = raw * sqrt(t!)

Raises are creation operators in the monomial basis, so an emission into a mode that s occupies
carries its sqrt(s_p + 1) factor by construction (and the stage is then not trace preserving: the
emission is an isometry only on unoccupied modes). Dispatch: vacuum input (B = 1) with a
single-photon chain is today's `EmitterSLOSKernel` on `chain.to_mps()`, forwarded, so its results
are today's tensors. Labels of one event that emit into the same mode are merged by `to_mps`
(their operators add). complex128 default; differentiable in U and `ops_tensor`; `kraus(chunk=)`
processes that many inputs of a group at a time. Loss and distinguishability parameters are
refused (loss is written as photon-less labels, below). Tests: `tests/slos/test_stage.py` (vs an
independent permanent oracle on the expanded label words, 1e-12; dispatch bit-identical; batch =
per-input calls bit-identical; completeness).

**Labels of any photon number.** A label's word may hold 0 photons (a photon-less emission
outcome: the emitter applies M and no photon enters the network; this is the internal-modes kernel's
`lost` operator, `docs/SEQUENCE.md`) or several (multiphoton emission, E_k = sum_l M_l (x)
prod_p (a_p^dag)^{w_p} / sqrt(w_p!)). Such a chain takes the graded recurrence, every reachable
photon number live:

    for each event k, for each word group (labels with one word and one env add: M = sum M_l):
        |w| = 0:   psi_T      += M psi_T
        |w| = 1:   psi_{T+1}  += slab pass with C_j = sum_groups U[j, p] M     (today's step)
        |w| >= 2:  psi_{T+|w|} += M (prod over the word's photons of A_p) psi_T / sqrt(w!)

with the outputs of every reachable N + T, largest first. `env` tags labels with orthogonal records
of the emitter's environment: labels with different env never interfere, so the stage's channel is
the sum over the env words (one env value per event) of coherent chains, one Kraus operator per env
word and outcome (`StageGroup.envs`). The recurrence runs once per env word, so time and memory scale
with E = prod_k (env values of event k): prod_j (R_j + Q_j) for a tick of internal-mode emitters,
5^k at R = 3, Q = 2; `StageKernel` warns above E = 1024. `StageChain.from_bin_emitters` maps a tick of internal-mode
emitters with K = 1 onto this: label l with unobserved outcome r -> env r, lost operator q -> an empty
word with env R + q. A single-photon chain with one env word never takes the graded path. Carry-over
works on these chains as above, the blocks gaining an env axis that `apply` sums. Tests:
`tests/slos/test_stage_words.py` (vs a permanent oracle with the words' factorials, per env word,
1e-12; completeness over patterns and env words 1e-13; a single-photon chain through the graded
path = the single-photon kernel; the internal-modes kernel's `lost` / R > 1 channel per pattern,
1e-13; probabilities, carry-over, gradients).

**Partial detection (carry-over, route B).** `StageKernel(chain, inputs, detected=D)` keeps the
other modes C coherent, as a stage simulator's carry-over does: the inputs are its carried-in Fock
basis {alpha_b}, and the result per detected pattern q is a block operator from (input, register)
to (carried-out basis, register). Each full output pattern o splits uniquely as (q, c) = (o on D,
o on C), so the blocks are an index regroup of the full-output Kraus set above:

    tables (once per kernel): o -> (q, c); q and c numbered by first appearance; per q its
        carried-out basis (the distinct c), grouped in classes of equal basis size C_q
    blocks(U):  K_q[(c, f), (b, i)] = K_{q+c}^(b)[f, i]   (physical; zero where q + c has another
                                                           photon number than input b's outputs)
    apply(U, rho, rest=X):  rho_q = (K_q (x) 1_X) rho (K_q (x) 1_X)^dag,  p(q) = Tr rho_q
                two batched einsums per class and chunk of q; rho over (b, register, rest), b major

The carried-out basis is returned explicitly, as occupations over all m modes (zero on D):
`StageCarry.carried` and `CarryClass.basis`. `rest` is a passive register next to the chain's
(sources that do not emit in the stage); a photons-only stage (`StageChain.empty`, chi = 1) has
scalar blocks, so neither case forms a Kronecker product. `apply` reports `deviation` =
sum_q Tr rho_q - Tr rho, zero to rounding when the chain is complete, U unitary and the inputs avoid
the emission modes (then sum_q K_q^dag K_q = 1); above `tol` x Tr rho (default 1e3 eps) it warns,
or raises with `strict=True` (what a stage simulator should use; an input occupying an emission
mode is defined but not an isometry, so its deviation is legitimately non-zero). Cost: one
full-output Kraus set (B_N x the stage SLOS per group) plus the regroup and the update; memory: that
set, plus one pass of blocks and update tensors (`chunk` q at a time, by default sized from
`chunk_bytes`, 1 GiB). The regroup tables pack each pattern into a mixed-radix int64 key when it
fits (a 1-D unique), else they fall back to a row unique. The vacuum dispatch above applies here too.
Tests: `tests/slos/test_stage_carry.py` (C = {} is the Kraus set bit for bit; blocks vs the permanent
oracle and a brute-force carried basis; completeness 1e-13; apply vs the explicit loop of a stage
simulator, with a rest, 1e-14; photons-only stage; two stages = one-shot full detection of the
unrolled problem, 1e-13; chunks, batched U and rho, gradients).

## 3. Later

- `adapters/perceval`: done (`backends.py`).
- `adapters/merlin`: the trainable op (SC_Param in `benchmarking`).
- `slos/implicit`: layer kernel with on-the-fly combinadic parent indices (no stored basis;
  memory = two state vectors) — the first step past n = 11, m = 22 on one GPU.
- `slos/distributed`: the memory wall. `slos_mpi/slos_mpi_dist.c` (colex chunking, on-the-fly
  rank/unrank, streaming halo exchange, O(M_{n-1}/P) per rank) re-expressed as a multi-GPU
  `torch.distributed` kernel with `all_to_all` halos; the MPS extension is chi chunked vectors.
- Cache blocking for the gather (from `slos_mpi/blocked_kernel.py`: nested co-lex blocks bring
  ~m x fewer DRAM reads) as a layer-ordering experiment on the torch kernel.
