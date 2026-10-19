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
