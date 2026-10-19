# Time-bin loop devices (ORCA PT-series) in photonic-engines

Status: implemented (2026-09-11) in `photonic_engines.timeloop`; the measurements are
quoted with their run stamps below. §5 covers the generalisation to arbitrary periodic networks with delay lines
(Perceval `TD` semantics) and the MerLin hook.

## 1. The device

A PT-series machine is one spatial rail carrying T time bins, with one or more fibre loops of
delay ℓ_i bins closed on the rail by a programmable beam splitter (θ_{i,t}, φ_{i,t}) — one
setting per loop and per time step. Photons enter in given bins (one or several per bin,
from a single source firing every clock period), circulate through the loops, and are
detected bin by bin at the output. Loops start empty; photons still in a loop after the last
bin are lost unless the last ℓ_i settings flush them (θ = π/2 = full swap).

Two facts drive the design:

1. Unrolled over time it is an ordinary linear-optical network on m' = T + L modes,
   L = Σ ℓ_i (T bin wires + L loop-slot wires, the slot of loop i used at time t being
   t mod ℓ_i). Every existing engine runs on that unitary.
2. It is *causal*: at time t the only quantum memory is the loop content. Once a bin has
   passed the last loop it goes to the detector and never interacts again. The state that has
   to be carried is a Fock register on L slots (+ the bin in transit), dimension
   D = C(N+L+1, L+1) for N photons — polynomial in N for fixed L.

| N \ L | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 4  | 15 | 35 | 70 | 126 | 330 | 715 |
| 8  | 45 | 165 | 495 | 1 287 | 6 435 | 24 310 |
| 12 | 91 | 455 | 1 820 | 6 188 | 50 388 | 293 930 |
| 20 | 231 | 1 771 | 10 626 | 53 130 | 888 030 | 10 015 005 |
| 30 | 496 | 5 456 | 46 376 | 324 632 | 12 620 256 | — |

(D = C(N+L+1, L+1); the generic Fock space on m = 2N bins is C(3N−1, N): 3·10⁵ at N = 8,
1.4·10¹⁶ at N = 20.)

## 2. Two paths

**Generic path** — `TimeLoopCircuit.unitary(theta, phi)`: differentiable (torch autograd)
construction of the m'×m' unrolled unitary as a time-ordered product of embedded 2×2 beam
splitters. Feeds `FockSLOSKernel` / `SuperpositionSLOSKernel` / `fock_sample` /
`EmitterSLOSKernel` unchanged; leftover loop-slot modes are marginalised by the caller. Same
matrix as Perceval's `TD` unrolling, with the gradient path MerLin needs. Right choice when
L is comparable to T (register as large as the Fock space).

**Native path** — `LoopRegisterKernel`: the register is marched through time. At step t:
inject the n_t input photons into the bin, apply the loop-i beam splitter to (bin, head slot
of loop i) for each loop in series, then the bin exits. The beam splitter acts on the register
as a block-diagonal matrix: for a pair of modes holding k photons together it is the
(k+1)×(k+1) Fock representation B_k of the 2×2 gate, built recursively in k (`fock_blocks`,
differentiable, exact). Three operating modes on the exiting bin:

- *project* (pure state, given outcome): amplitude of an output pattern in O(T · D · N),
  leftover register photons marginalised at the end;
- *sample* (pure state, batched over shots): draw the bin's photon number from the norm of
  the register components with that bin occupation, project, renormalise — a quantum
  trajectory. Exact sampling in O(T · D · N) per shot, no permanents, no Clifford chain,
  threshold detectors for free;
- *trace* (density matrix ρ on the register, D×D): per-bin photon-number distributions
  p_t(k) and register marginals in O(T · D² · N) — the readout MerLin trains on; losses per
  round trip and detector inefficiency are single-mode amplitude-damping Kraus maps on ρ (the
  sampler applies the same maps as Monte-Carlo jumps).

The full output distribution stays exponential (C(T+N−1, N) patterns) and is obtained by
branching the pure state on every bin — the SLOS-style enumeration, only useful at small N.

Dispatch: native when D ≪ C(m'+N−1, N), generic otherwise. Emitter (QD) input feeding the bins
is the existing `EmitterMPS` on the same time axis; the combined bond is spin ⊗ register
(χ = 2·D) and is the natural extension of the emitter kernel — item for later, the generic
path covers it today.

## 3. Package layout

```
photonic_engines/timeloop/
    circuit.py   TimeLoopCircuit(T, loops)   .unitary(theta, phi | gates) -> [T+L, T+L] (autograd)
                                             .gate_matrices, .flush_settings, .slot(i, t)
    basis.py     RegisterBasis(L, N, n_rail): Fock basis of the rail wires + slots, pair-group
                 tables for any mode pair, rail patterns, inject / exit / loss index maps
    kernel.py    fock_blocks(V, N)           Fock blocks of 2-mode unitaries, recursive, autograd
                 PeriodicNetworkKernel(n_rail, delays, program, T, inputs)      general engine (§5)
                     .amplitudes(patterns, gates) .probabilities .distribution
                     .period_marginals(gates, loss=…)  rho mode, [T, Q] differentiable
                     .sample(shots, gates, generator, loss=…, threshold=…)
                 LoopRegisterKernel(circuit, input_occupation)   ORCA case: same API with
                     (theta, phi) and loss=dict(loop=[eta_i], detector=eta_d, uniform=eta) or
                     loss=eta (uniform end-to-end, by output thinning: docs/LOSS.md); .bin_marginals
    module.py    TimeLoopModule            torch.nn.Module: trainable theta/phi, readouts
                                            clicks | marginals | mean | patterns (MerLin hook)
photonic_engines/adapters/perceval/timeloop.py
                 periodic_kernel_from_perceval(processor, input_state)   Perceval TD -> kernel
bench/bench_timeloop.py       time per sample / per marginal vs N and L, both paths
bench/timeloop_frontier.py    N* at 1 sample/s as a function of L (native) vs the Fock frontier
tests/timeloop/, tests/adapters/test_perceval_timeloop.py
```

## 4. Frontier question

For fixed L the native *sampling* cost is polynomial (O(T · n_loops · D · N) per shot: one
two-mode block per loop and per bin; the table below was measured with a **single loop of
delay L**, so a device of n_loops loops with the same total delay costs n_loops× more per shot
at the same D, N* ↓ by ≈ n_loops^(−1/(L+1)): (1,1,1) instead of (3): 116 → ≈ 88; eight unit
loops at L = 8: 21.6 → ≈ 17; the input is [1]*N + [0]*(T−N), the cost does not depend on it),
so the 1 sample/s
frontier is not a photon number but a curve N*(L): N* → hundreds for L = 1, 2 and meets the
generic Fock sampling frontier (n* = 37.8 at m = 2n on an H100) where the per-shot costs
cross — measured at L ≈ 5.5, far before the register is comparable to the unrolled Fock
space (the a-priori guess "around L ≈ N" was wrong). The full output distribution over T
bins (C(T+N−1, N) patterns) stays exponential for every engine; marginals use the D×D
register density matrix (D ≲ 5·10⁴ in memory).

Measured (run 2, one H100 PCIe, complex128, 1 sample/s, `results/keep/timeloop_frontier_run2.csv`,
`img/timeloop_frontier_run2.png`; T = bins per period, the analogue of m/n):

| L (total loop length) | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| N*, T = N | 315 | 287 | 127 | 70 | 34 | 23 |
| N*, T = 1.5 N | 287 | 256 | 124 | 64 | 33 | 22 |
| N*, T = 2 N | 267 | 241 | 116 | 63 | 31.5 | 21.6 |
| N*, T = 3 N | 247 | 221 | 106 | 61 | 30 | 21 |
| N*, T = 4 N | 224 | 204 | 103 | 57.5 | 30 | 20 |
| D at round(N*), T = 2N | 3.6·10⁴ | 2.4·10⁶ | 8.2·10⁶ | 1.0·10⁷ | 1.5·10⁷ | 2.0·10⁷ |

Three readings. (i) The sampling frontier is set by the register dimension, not by N: at N*
the register holds 0.8–2·10⁷ amplitudes for every L ≥ 3, i.e. the sampler runs at ~1 s per
sample once D ≈ 10⁷ regardless of how that D is reached (L = 1 and 2 stop earlier, at
D = 4·10⁴ and 2·10⁶, because the per-bin cost there is the T gate applications on a small
state — launch-bound, ~1 ms per bin). (ii) The bin ratio T/N matters little — a factor 1.4
in N* between T = N and T = 4N at L = 1 and ~10 % at L ≥ 4 — because the cost per period is
the same and only the number of periods grows. (iii) The native sampler crosses the generic
Fock frontier (n* = 37.8 at m = 2n) at L ≈ 5.5 for every ratio; an ORCA-type device with
loop *delays* totalling ≤ 5 bins (L = Σ ℓ_i, not a number of loops: ORCA's announced
(1, 3, 9) ladder has L = 13, N* ≈ 12 by the D ≲ 10⁷ rule; (1, 5, 25): L = 31, N* ≈ 6;
Borealis' (1, 6, 36): L = 43, N* ≈ 5) is simulable to N ≈ 60–300 photons on one GPU, beyond
that the unrolled unitary + Clifford sampler is the better engine. The surviving statement is
a conjunction: an ORCA-type device is beyond exact classical sampling only if N ≳ 40 *and*
L ≳ 6 bins.

**Prior work and complexity** (adversarial math review of 2026-09-12, since removed from the tree). The causal cut and the algorithm
are those of Novák, Roberts, Makarovskiy, García-Patrón, Clements, *Boundaries for quantum
advantage with single photons and loop-based time-bin interferometers*, Quantum 9, 1915 (2025),
arXiv:2411.16873 (ORCA): Theorem 3.1, R = 1 + Σ ℓ_i relevant modes per component, and their
Algorithm PS is `LoopRegisterKernel.sample` step for step; their loss extension (environment
mode, measured and collapsed) is our Monte-Carlo jump. They give no timings — a lattice-path
memory meter (reachable coordinates M, drawn from a heuristic outcome distribution) prices
configurations no simulator runs, and yields their boundary: three-loop power-law devices
(1, ℓ, ℓ²) with ℓ < 5 are simulable at any m, and "roughly 40 to 50 photons at least" are needed
to escape Clifford & Clifford. Their Table 1: power-law Λ loops base ℓ,
O(m Λ ℓ^{2(Λ−1)} 2.6^{ℓ^{Λ−1}}); Clifford & Clifford O(m 2^{m/2}); SLOS_full O(m 2.6^m). Ours:
O(T · n_loops · D · N) per shot with D = C(N+L+1, L+1), the dense Fock space of the R = L+1
relevant modes — an over-count of the reachable set (true Schmidt rank across the cut 15 vs
D = 35 at loops (1,2), N = 4; their M ≈ 10¹⁵–10¹⁶ vs D = C(66,22) ≈ 6·10¹⁷ at (1,6,36), m = 44),
a constant in the exponent, so the frontier above is an implementation boundary of the dense
engine, not a mathematical one (a reachable-set index over a dense buffer would move it).
Also: Lubasch et al., PRA 97, 062304 (2018) (loop interferometer as an MPS in time), Brádler &
Wallner, arXiv:2112.09766 (lattice paths, unit loops), Deshpande et al., Sci. Adv. 8, eabi7894
(2022) (Borealis' delay ladder chosen to escape local simulability). Scope: Fock inputs only —
a Gaussian machine is a different object; and "simulable" means 1 sample/s on one GPU.

## 5. General periodic networks, Perceval `TD`, MerLin

**Register picture.** Any linear-optical network that repeats every period with delay lines
is a register machine: rail wires of the current period + delay-line slots, and a per-period
*program* of two-mode gates on any pair (rail wire or head slot of a delay line) and
single-mode losses. `PeriodicNetworkKernel` implements exactly that; `LoopRegisterKernel` is
the program `[gate(bin, slot_i)]` per loop. The state dimension is D = C(N+L+n_rail, L+n_rail).

**Perceval semantics.** Perceval's `TD(t)` is a *delay line* at one position of a feed-forward
circuit: what enters it at period d re-emerges at the same position at period d+t, the same
circuit and the same input state apply every period, and `Processor.probs()` reports the last
of `depth` periods (`DelaySimulator`). In the register picture `TD(t)` on mode a is a swap
between wire a and the head slot of a t-slot delay line; BS/PS/PERM are 2-mode gates (a phase
is `diag(e^{iφ}, 1)` on (a, neighbour), a PERM a product of adjacent swaps).
`periodic_kernel_from_perceval` builds the program; `last_period_distribution` reproduces
`Processor.probs()` to 1e-16 (tests/adapters/test_perceval_timeloop.py), with the register
(D states) instead of the unrolled `depth·m + Σt` modes. Note that an ORCA loop — a cycle
where the beam splitter couples the rail *directly* to the loop — cannot be written with `TD`
in a feed-forward Perceval circuit; it needs `LoopRegisterKernel` (or the unrolled
`TimeLoopCircuit.unitary`, which Perceval can consume as a `Unitary` component).

**MerLin.** `TimeLoopModule` is a `torch.nn.Module` with `theta`, `phi` as parameters
(`trainable_phi=False` for a device without programmable phases) and a differentiable forward:
`clicks` (p(click) per bin, threshold detectors), `marginals` ([T, N+1]), `mean` photon number
per bin, or `patterns` (probabilities of chosen outputs); losses per round trip and per
detector included in the click/marginal readouts. `adapters/merlin.py: TimeLoopLayer` is the
MerLin-facing form — the `QuantumLayer` contract (features in, readout out, batched over
rows): features are angle-encoded on chosen (loop, bin) settings (`encode_slots`), the
readout is any of the above plus `distribution` (fixed basis of C(T+N−1, N) patterns), the
last L settings stay at the flush value (`flush=True`), and a batch is a Python loop over its
rows (the kernel is not `vmap`-able: in-place gathers). Trained end to end
on a 2-feature XOR (T = 6, loops (1, 2), 3 photons, losses on): 79 %
train accuracy after 30 Adam steps from 50 %. Test: tests/adapters/test_merlin.py (readouts,
gradients, encoded settings against the kernel, fitting a hidden device's clicks to 10× lower
MSE in 60 steps).

**Emitter (QD) input.** The state always carries a spin axis: ψ ∈ ℂ^χ ⊗ register (χ = 1
without emitter), ρ is [χ, D, χ, D]. An `EmitterMPS` whose modes are bins feeds the loops
through `LoopRegisterKernel(circuit, zeros, emitter=state)`: the period becomes
`bins_per_period` bins (2 for the dual-rail caterpillar (2k, 2k+1) — the rails of the period),
emission k at its period applies Σ_l M_{k,l} ⊗ a†_{rail l}, the loops see the rails one after
the other (slot specs ("slot", i, r) with stride = bins per period). Traced spin
(`final=None`): marginals and sampler as before; projected spin (`final`): `amplitudes` /
`probabilities` carry the projection weight, the sampler rejection-samples (acceptance = the
projection probability), per-bin marginals would need a backward (Heisenberg) sweep of the
final projector — not implemented. Validated against `EmitterSLOSKernel` on the unrolled
unitary (tests/timeloop, tolerance 1e-12; observed differences ~1e-16). Classical photons and emitter photons can be mixed.

**Still open.** Backward sweep for post-selected marginals, >2-mode components in the Perceval
bridge (decompose), partial distinguishability.

## 6. Partial distinguishability (2026-09-20)

`LoopRegisterKernel.sample(..., indist=x)` and `bin_marginals(..., indist=x, mc_shots=)` —
docs/DISTINGUISHABILITY.md §3.6, x-model, classical (Fock) inputs. A photon in a mode of its own
still interferes *with itself* across the delay lines, so it is not a random walk: it is a
one-photon register evolution through the same gates and loss channels (`_single_kernel(b)`, an
N = 1 kernel per input bin, cached). A shot draws which photons interfere; those run the register
together under a per-stream injection schedule (`PeriodicNetworkKernel.sample(inputs=)`,
`period_marginals(inputs=)`), each other photon runs its own register, and the bin counts add.
`bin_marginals` enumerates the 2^N subsets when N ≤ 8 (exact; marginals convolved bin by bin with
the one-photon marginals) and otherwise averages `mc_shots` subsets (an estimate). Uniform loss is
thinned after, loop and detector channels run on every register. Checked against the x-mixture on
the unrolled unitary (leftover loop modes summed) to 1e-13 on the marginals and within shot noise
for the sampler. Emitter-driven loops refuse `indist=`.
