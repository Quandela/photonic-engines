# photonic-engines tutorial

Everything below runs on the CPU unless marked **GPU**. The examples are checked by
`tests/test_tutorial.py`, so they stay in sync with the code.

Contents

1. What is in the box
2. Input states and output spaces
3. Strong simulation of Fock inputs (`FockSLOSKernel`)
4. Superposition inputs (`SuperpositionSLOSKernel`); 4b superposition boson sampling (`ProductSuperposition`)
5. Emitter (caterpillar) states: strong simulation, Kraus operators, sampling for free; 5b additive-error estimation
6. Exact sampling (`sample`, `fock_sample`, `emitter_sample`; the algorithm variants)
7. Checking a sampler against a strong simulation
8. Perceval bridge
9. Working on a GPU: sizing, caches, benches, tuning; then the other modalities — 9b time-bin loops,
   9c the distributed engine, 9d losses and partial distinguishability, 9e emitter trains, 9f non-linear
   sites, 9g adaptive linear optics, 9h Gaussian inputs
10. Which engine when

---

## 1. What is in the box

```
photonic_engines.states      FockState, FockSuperposition, ProductSuperposition, EmitterMPS, GaussianInput, spin_gate
photonic_engines.masks       Mask, state_space_size
photonic_engines.imperfections  uniform loss (LossyDistribution, thinning) and partial distinguishability (Distinguishability)
photonic_engines.slos        FockSLOSKernel, SuperpositionSLOSKernel, ProductSLOSKernel, EmitterSLOSKernel,
                             GaussianSLOSKernel, sample_from_probs, get_layers, reference_probs;
                             slos.estimate.EmitterGurvitsEstimator (additive-error probabilities)
photonic_engines.clifford    sample, fock_sample, emitter_sample, product_sample   (backend = "triton" | "cuda" | "torch");
                             clifford.variants: the algorithm set (strong_then_sample, reference steps)
photonic_engines.gaussian    sample, sample_transfer, pure_probability, lhaf   (squeezed inputs)
photonic_engines.timeloop    TimeLoopCircuit, LoopRegisterKernel, TimeLoopModule   (loop devices, §9b)
photonic_engines.sequence    SequenceKernel, BinEmitter   (k emitters x n time bins, §9e)
photonic_engines.nonlinear   NonlinearCircuit   (Kerr sites between interferometers, §9f)
photonic_engines.adaptive    AdaptiveCircuit, mode_marginals   (photon counting + feed-forward, §9g)
photonic_engines.adapters    perceval (CircuitConverter, three backends), merlin
```

Two kinds of engine:

* **Strong simulation** returns the whole output distribution (or all amplitudes) for a
  circuit. Cost is linear in the size of the output space, which grows as C(m+n−1, n) — at
  m = 2n that is 1.2·10^5 states for n = 8, 1.3·10^8 for n = 11. This is the `slos` package.
* **Sampling** draws output states one at a time without ever forming the distribution. Cost
  per sample is ~n·2^n (Clifford & Clifford), growing by a factor ρ ≈ 1.7–2.0 per photon
  (ρ depends on the mode ratio through the repeated-rows collapse), where strong simulation
  stops at n = 11–12. The measured frontier — the photon number n* at which one H100 gives
  one sample per second — is **38.5 at m = 2n and 42.5 at m = n** (canonical reruns of
  2026-09-30, ρ_eff 1.816 / 1.775; run 20260930_082501), falling with m/n (the map:
  `results/keep/frontier.csv`, docs/HARDWARE_TARGET.md). G GPUs add ln G / ln ρ photons (+3.5 for 8 GPUs at m = 2n). Emitter states cost a factor
  R_k (pure terms of the reduced state) on top: their frontier is n* = 30.6 at m = 2n
  (docs/EMITTER_SAMPLER.md §8). This is the `clifford` package; it needs a CUDA GPU for anything
  but tests.

Both accept the same input types, including the states produced by a single spin emitter
(caterpillar / linear cluster / GHZ families), for which they carry the emitter's 2-dimensional
bond instead of the 2^n Fock components those states expand to.

Install: `pip install -e .` (torch + numpy), `.[cuda]` adds triton and cupy, `.[perceval]`
adds perceval, `.[dev]` adds pytest.

## 2. Input states and output spaces

```python
import math, torch
from photonic_engines import FockState, FockSuperposition, EmitterMPS, spin_gate, Mask, state_space_size

fock = FockState((1, 0, 1, 0, 1, 0))                     # 3 photons in modes 0, 2, 4 of m = 6
fock.n, fock.m, fock.modes()                              # 3, 6, [0, 2, 4]

sup = FockSuperposition(((1, 0, 1, 0), (0, 1, 0, 1)), (1 / math.sqrt(2), 1j / math.sqrt(2)))

# output-space restrictions: fewer states to compute and store
state_space_size(6, 12, Mask.FOCK)            # 12376  all occupations
state_space_size(6, 12, Mask.UNBUNCHED)       # 924    at most one photon per mode
state_space_size(6, 12, Mask.DUAL_RAIL)       # 64     one photon per (2j, 2j+1) pair
```

A unitary is a complex `[m, m]` torch tensor; a batch of circuits is `[B, m, m]`. The
examples use a Haar-random one:

```python
def haar(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=g))[0]
```

### Emitter states

`EmitterMPS` describes photons emitted one after another by a single emitter of dimension
`chi` (a spin, chi = 2): at emission k the emitter operator `ops[k, l]` acts and photon k goes
into input mode `modes[k][l]`, where l is the emission label (polarization: R or L, i.e. two
dual-rail modes). The state is the matrix-product state
|ψ⟩ = Σ_x ⟨f| M_{x_n} … M_{x_1} |s0⟩ |x⟩, with the emitter traced out at the end when
`final=None` (a mixture) or projected on `final`.

The caterpillar constructor covers the protocol of the Nature Comm. 2025 experiment:
`spin_gates[0]` prepares the spin, `spin_gates[k]` = U(θ, φ) is the pulse sequence between
emissions k−1 and k:

```python
prep = spin_gate(math.pi / 2, 0)                                    # |up> -> |+>
n = 4
lc  = EmitterMPS.caterpillar(n, [prep] + [spin_gate(math.pi / 2, 0)] * (n - 1))        # linear cluster
ghz = EmitterMPS.caterpillar(n, [prep] + [spin_gate(math.pi / 2, math.pi)] * (n - 1))  # GHZ
lc.n, lc.m, lc.chi, lc.num_labels                                   # 4, 8, 2, 2
lc.modes                                                            # ((0, 1), (2, 3), (4, 5), (6, 7))
proj = EmitterMPS.caterpillar(n, [prep] * n, final=(1, 0))          # spin measured in |up>
```

`modes=` and `m=` place the photons in an arbitrary interferometer; `s0=` sets the initial
spin. `state.expand()` returns the 2^n-component `FockSuperposition` form (one per final spin
state) — an oracle for small n, never used by the engines.

## 3. Strong simulation of Fock inputs

`FockSLOSKernel` is built once for (n, m, mask, input) and then evaluated on as many unitaries as
you like. It runs the SLOS_full recurrence layer by layer over the photon-position basis.

```python
from photonic_engines.slos import FockSLOSKernel

U = haar(8)
k = FockSLOSKernel(n=3, m=8, input_occupation=[1, 0, 1, 0, 1, 0, 0, 0], mask="fock",
               device="cpu", dtype=torch.complex128)
p = k.probs(U)                        # [120] float64, sums to 1
states = k.layers.final_states        # list of 120 occupation tuples, same order as p
best = max(range(len(p)), key=lambda i: p[i])
states[best], float(p[best])
```

Batches of circuits and gradients come for free — the kernel is plain torch:

```python
Ub = torch.stack([haar(8, s) for s in range(4)])          # [4, 8, 8]
pb = k.probs(Ub)                                          # [4, 120]

Ug = U.clone().requires_grad_(True)
loss = -k.probs(Ug)[best].log()                           # e.g. maximise one outcome
loss.backward()
Ug.grad.shape                                             # [8, 8] (complex)
```

Notes

* `dtype=torch.complex64` (the default) halves memory and is what the GPU numbers use;
  `complex128` for reference precision.
* `amplitudes(U)` returns the recurrence's raw amplitudes in *deferred normalization*: multiply
  by `sqrt(k.layers.output_prob_norms())` (= √Π t_j!) for the physical amplitudes. `probs`
  already includes it.
* `mask="unbunched" | "dual_rail" | "sparse_1in4" | "tail_empty"` prunes the basis while it is
  built, so both time and memory shrink with the output space.
* `k.group_matrix(groups)` folds a `[S] -> bin` assignment into the final step (coarse-grained
  distributions, e.g. per detector pattern).
* `reference_probs(U, n, m, occ)` (permanents, Ryser) is the correctness oracle for small n.

### Sampling from a strong simulation

With the distribution in hand, drawing samples is a `multinomial`:

```python
from photonic_engines.slos import sample_from_probs
occ = sample_from_probs(k.layers, p, shots=1000, generator=torch.Generator().manual_seed(0))
occ.shape, occ.sum(1).unique()        # [1000, 8], tensor([3])
```

## 4. Superposition inputs

`FockSuperposition` holds a coherent sum Σ_c w_c |s_c⟩ of K Fock states with the same photon
number as arrays (`occupations` int64 [K, m], `amplitudes` complex [K]) — K up to ~10^6 is fine
storage-wise; the engines' cost is K SLOS passes, chunked. `SuperpositionSLOSKernel` runs the
recurrence for the K components in one batch and sums the amplitudes coherently (the EBS
kernel):

```python
from photonic_engines.slos import SuperpositionSLOSKernel
U4 = haar(4)
mk = SuperpositionSLOSKernel(n=2, m=4, input_occupations=[[1, 0, 1, 0], [0, 1, 0, 1]],
                          device="cpu", dtype=torch.complex128)
amp = mk.amplitudes(U4, weights=[1 / math.sqrt(2), 1j / math.sqrt(2)])     # [10]
p = mk.probs(U4, weights=[1 / math.sqrt(2), 1j / math.sqrt(2)])            # [10], sums to 1

mk = SuperpositionSLOSKernel.from_state(sup, dtype=torch.complex128)   # weights = sup's amplitudes (normalized)
p = mk.probs(U4)
```

`FockSuperposition.from_perceval(state_vector)` / `.to_perceval()` convert to and from
Perceval's `StateVector` (one block per photon number, since a StateVector may mix them); the
Perceval backend of §8 accepts a StateVector directly. There is no exact *sampler* for a
K-component superposition yet — and note that "draw a component, then sample it" gives the
mixture Σ|w_c|² p_c, not the superposition; the interference terms are lost.

## 4b. Superposition boson sampling: sources with an indefinite photon number

`ProductSuperposition` is a product over independent sources, each in a photon-number
superposition of its own input mode. With `d = 2` and the same angle everywhere this is Renema's
superposition boson sampling (arXiv:1911.10112) — physically, a quantum dot driven at pulse area
θ emits cos(θ/2)|0⟩ + sin(θ/2)|1⟩ coherently. The photon number is indefinite, so the engine
returns one block per photon number instead of a single vector (docs/SUPERPOSITION.md):

```python
from photonic_engines import ProductSuperposition, graded_state_space_size
from photonic_engines.slos import ProductSLOSKernel, product_amplitude

sbs = ProductSuperposition.sbs(sources=4, m=8)      # alpha = pi/4, modes 0..3: the hardest case
sbs.n, sbs.d, sbs.sources                            # 4, 2, 4
sbs.photon_number_probs()                            # Binomial(4, 1/2), exact and independent of U
graded_state_space_size(sbs.n, sbs.m)                # 495 = C(12, 4) states, 1.5x the 4-photon layer

U8 = haar(8)
pk = ProductSLOSKernel(sbs, device="cpu", dtype=torch.complex128)
dist = pk.probs(U8)                                  # a LossyDistribution over 0..4 photons
float(dist.mass(2))                                  # 0.375 — P(2 detected), the binomial weight
cond = dist.conditional(2)                           # the 2-photon distribution after post-selection
occ = pk.sample(U8, 100)                             # exact samples [100, 8] (strong sim, then draw)
lossy = pk.probs(U8, loss=0.8)                       # every layer thinned and re-summed
```

Post-selected on k photons the input is the Dicke state on the source modes — a coherent sum of
C(sources, k) Fock components, which is why the engine takes the compact state and not a
`FockSuperposition`. `blocks()` writes those components out and is a test oracle only.

A single amplitude does not need the basis at all: it is one permanent whose "no emission" row is
repeated sources − k times, so it costs O(2^k (n−k) n) — exponential in the *detected* count, not
in the source count:

```python
amp = product_amplitude(sbs, U8, [1, 1, 0, 0, 0, 0, 0, 0])   # <t| phi(U) |sbs>, no state vector
```

`clifford.sample` takes this input type too, through a chain rule that marches the true marginal
of the rows placed so far — exact, and not bounded by strong simulation the way the line above is:

```python
from photonic_engines import clifford
occ = clifford.sample(sbs, U8, 50)                   # chain rule; no basis, no state vector
occ = clifford.sample(sbs, U8, 50, photons=2)        # the post-selected 2-photon (Dicke) sector
```

What bounds it is the per-step marginal, whose two evaluations are cheap at opposite ends of the
chain and neither in the middle (docs/SUPERPOSITION.md §5): 8 ms per sample at k = 4, 4.3 s at
k = 8 on a CPU.

## 5. Emitter states: strong simulation

`EmitterSLOSKernel` runs the same recurrence with `chi` amplitude vectors instead of one (the
bond index rides along), so a 2^n-component state costs 2× a single Fock state:

```python
from photonic_engines.slos import EmitterSLOSKernel
U8 = haar(8)
ek = EmitterSLOSKernel(lc, device="cpu", dtype=torch.complex128)
p = ek.probs(U8)                          # [330] over ek.layers.final_states; traced spin -> sums to 1
phi = ek.bond_amplitudes(U8)              # [2, 330]: amplitudes for final spin |up>, |down>
```

A projected emitter gives a pure photonic state whose weight is the projection probability:

```python
ekp = EmitterSLOSKernel(proj, device="cpu", dtype=torch.complex128)
amp = ekp.amplitudes(U8)                  # [330] complex
pp = ekp.probs(U8)                        # sums to P(spin = up) <= 1
```

Kraus operators on the spin, one per output state — what a feed-forward stage simulator
needs: detecting output t updates the spin as ρ → K[t] ρ K[t]†:

```python
K = ek.kraus(U8)                          # [330, 2, 2] (final x initial), deferred normalization
K = K * torch.sqrt(torch.as_tensor(ek.layers.output_prob_norms(), dtype=torch.float64)).view(-1, 1, 1)
torch.einsum("tji,tjk->ik", K.conj(), K)  # sum_t K^dag K = identity (completeness)
```

`sample_from_probs(ek.layers, p, shots)` samples from the traced distribution as in §3. And
because `ek.probs` is torch, gradients with respect to U — or to the emitter operators, via
`ops_tensor=` — are available for calibration / learning.

Check against the expansion for small n:

```python
comps = lc.expand()                                       # one FockSuperposition per final spin state
mk2 = SuperpositionSLOSKernel(lc.n, lc.m, [c.occupation for c in comps[0].components], dtype=torch.complex128)
p_up = mk2.probs(U8, weights=comps[0].amplitudes)         # unnormalized branch weight
```

### 5b. Emitter states: additive-error estimation without the exponential

When only a few probabilities (or amplitudes) are needed and n is beyond the strong engine,
`EmitterGurvitsEstimator` gives them to additive error ±ε with ~1/ε² Monte-Carlo samples of
polynomial cost (docs/ESTIMATE.md — Gurvits's permanent estimator extended to the MPS input:
each sample is a product-basis measurement of the emitter state, contracted at O(n L χ²)).
Same precision-per-shot as the device, no dependence on 2^n:

```python
from photonic_engines.slos.estimate import EmitterGurvitsEstimator
est = EmitterGurvitsEstimator(lc, sampling="gaussian", seed=0)     # traced spin -> probabilities only
t = ek.layers.final_states[5]
e = est.probability(U8, t, shots=20_000)  # Estimate(value, stderr, shots, second_moment)
abs(e.value - p[5].item()) < 5 * e.stderr
estp = EmitterGurvitsEstimator(proj, seed=0)                       # projected spin -> amplitudes too
a = estp.amplitude(U8, t, shots=20_000)   # physical amplitude <t|U|psi>; stderr ~ sqrt(||psi||^2 t!/shots)
```

`sampling="rademacher"` (x ∈ {±1}^m instead of complex Gaussian) is cheaper and, on every
unitary measured, lower-variance; the Gaussian law is the one with the theorem
(second moment exactly ‖ψ‖²·∏t_j!, whatever U and ψ). Sampling from the distribution is
*not* what this does — for that, §6.

## 6. Exact sampling

### Fock inputs (GPU)

`fock_sample` (or `sample(FockState(...), U, shots)`) draws exact samples for n photons in given input modes with the Clifford &
Clifford algorithm (2018 form for the early steps, 2020 repeated-rows form later), many
streams in lockstep on the GPU:

```python
from photonic_engines.clifford import fock_sample
Uc = haar(24).cuda()
gen = torch.Generator(device="cuda").manual_seed(0)
occ = fock_sample(Uc, in_modes=list(range(12)), shots=4096, generator=gen)   # [4096, 24] int64
occ = fock_sample(Uc, in_modes=list(range(12)), shots=4096, generator=gen, backend="cuda")  # cupy/NVRTC kernel, fastest at n >= 16
```

`U` may be `[B, m, m]` (B circuits × shots streams). One sample at a time
(`shots=1`, `backend="cuda"`) takes the latency path with no host synchronisation.

`algo=` picks the step of the chain rule: `"auto"` (the default and the production frontier:
the 2018 sweep for the first steps, the 2020 repeated-rows collapse after), `"2018"`, `"2020"`,
and two published variants kept for the algorithm comparison — `"input"` (the input-side
collapse of Brod & Oszmaniec / theboss, Triton only) and `"percof"` (PiquassoBoost's
per-cofactor sweeps). Measured on one H100 at m = 2n (run 20260930_082501, backend triton): n* =
32.5 (2018), 31.7 (input), 30.9 (percof), 35.7 (2020), against 38.5 for the production path;
docs/ALGORITHMS.md has the pseudo-code and the envelope. Two CPU references go with them:

```python
from photonic_engines.clifford.variants import fock_sample_reference, strong_then_sample
U6 = haar(6, 3)
stats = {}
occ = fock_sample_reference(U6, [0, 1, 2], 50, algo="input", stats=stats)   # Python chain, n <= ~10
stats["addends"]                                   # addends per shot: the work counted, not timed
st = {}
occ = strong_then_sample(U8, [0, 1, 2, 3], 1000, stats=st)   # SLOS build, then one lookup per sample
st["build_s"], st["draw_s"], st["states"]          # cost of a run of S samples: build / S + draw
```

`strong_then_sample` is bounded by the memory of the full distribution (n = 11 at m = 2n on an
80 GB H100; docs/MEMORY_FRONTIER.md), not by the rate.

### Emitter states

Sampling an entangled input is not the textbook chain rule: the photons are drawn in a
uniformly random order and the weight at step k is the probability of the rows drawn so far
under the *reduced* state on the k photons kept (docs/EMITTER_SAMPLER.md). One entry point,
two implementations of the same algorithm behind `backend=`:

```python
from photonic_engines.clifford import emitter_sample, sample
occ = emitter_sample(lc, U8, shots=200, generator=torch.Generator().manual_seed(0), backend="torch")   # [200, 8]
occ = sample(lc, U8, 200, backend="torch")            # dispatches on the state type (FockState -> fock_sample)
```

* `backend="torch"`: batched torch, CPU or GPU, no custom kernels. Fine up to n ≈ 14; the
  reference for the kernels.
* `backend="triton"` (**GPU**, the default): the matrix-cofactor kernels. `kernel="seq"`
  (default — Triton, generated per K with the repeated-rows collapse), `kernel="scan"` (the first
  Triton version), `kernel="cuda"` (the thread-per-lane NVRTC kernel, needs cupy; ≈ 1.4–1.5× at
  the frontier steps k = 25–28, slower in the middle of the chain) and `kernel="auto"` (CUDA at the
  frontier steps, seq elsewhere: Δn* ≈ +0.2 on the emitter row; docs/EMITTER_SAMPLER.md §8).
  `return_stats=True` adds the per-step number of pure terms, the phase split and the per-pass
  timing (`stats["passes"]`). The emitter frontier is n* = 30.6 at m = 2n. The first call for a
  new n compiles one kernel per step (a few seconds each, cached on disk).

```python
occ, stats = emitter_sample(lc, U8.cuda(), shots=1024, return_stats=True)
stats["R"]            # mean pure terms per stream at each step
stats["kernel_s"], stats["terms_s"], stats["post_s"]
```

Without a GPU the Triton kernels run under Triton's CPU interpreter (set
`TRITON_INTERPRET=1` before importing triton, as `tests/conftest.py` does) — correct but only
useful for tests.

## 7. Checking a sampler against a strong simulation

Total-variation distance between the empirical histogram and the exact distribution, compared
with the statistical floor √(2/π)·½·Σ√(p(1−p)/N):

```python
def tvd_check(layers, p, occ):
    idx = {s: i for i, s in enumerate(layers.final_states)}
    emp = torch.zeros_like(p)
    for row in occ.tolist():
        emp[idx[tuple(row)]] += 1
    emp /= occ.shape[0]
    tvd = 0.5 * float((emp - p / p.sum()).abs().sum())
    floor = 0.5 * float(torch.sqrt(p * (1 - p) / occ.shape[0]).sum()) * math.sqrt(2 / math.pi)
    return tvd, floor

tvd, floor = tvd_check(ek.layers, p, occ)     # expect tvd within ~2x floor
```

`tests/clifford/check_large_n.py` does this at n = 16..24 for the Fock sampler against
marginals (the full distribution no longer fits).

## 8. Perceval bridge

The Perceval adapters wrap the engines as `AStrongSimulationBackend` / `ASamplingBackend`
subclasses; the circuit → unitary conversion is vectorized over layers of commuting gates
(`CircuitConverter`, duck-typed on `circuit.m`, iteration and `compute_unitary()`).

```python
import perceval as pcvl
from photonic_engines.adapters.perceval import SLOSTorchBackend, EmitterSLOSBackend, CliffordGPUBackend

circ = pcvl.Unitary(pcvl.Matrix(haar(6).numpy()))
b = SLOSTorchBackend(dtype=torch.complex128)       # device="cuda" on a GPU
b.set_circuit(circ); b.set_input_state(pcvl.BasicState([1, 0, 1, 0, 1, 0]))
b.all_prob(), b.prob_amplitude(pcvl.BasicState([1, 1, 1, 0, 0, 0])), b.evolve()

e = EmitterSLOSBackend(dtype=torch.complex128)
e.set_circuit(circ); e.set_emitter_state(EmitterMPS.caterpillar(3, [prep] * 3))   # m = 6
e.prob_distribution()

s = CliffordGPUBackend(device="cuda", seed=0)        # GPU
s.set_circuit(circ); s.set_input_state(pcvl.BasicState([1, 0, 1, 0, 1, 0]))
s.samples(1000)
```

## 9. Working on a GPU

**Sizing a strong simulation.** Memory ≈ basis + amplitudes: the basis (photon-position
layers + edges grouped by mode) is built on the device (`basis_torch`) and the recurrence holds
one `[B, chi, S_k]` layer at a time. On an 80 GB H100 with m = 2n and complex64: n = 10 in
50 ms / 4 GB, n = 11 (1.3·10^8 states) in 0.1 s (Fock) / 0.3 s (emitter) with 16–18 GB;
n = 12 (8.3·10^8 states) does not fit a single GPU with a stored basis. `state_space_size`
tells you S; `photonic_engines.slos.basis.device_bytes(k.layers)` the basis footprint;
`clear_layers_cache()` frees the cached bases (they are cached per (n, m, mask, device)).

**Sampling.** Fock: 0.001–0.2 ms per sample for n = 12..24 at m = 2n, 1 sample/s at n = 38.5
(m = 2n) and 42.5 (m = n). Emitter: 1 sample/s at n = 30.6 (m = 2n); below n ≈ 20 the host-side
term decomposition dominates, above it the kernel. Heavy-tailed engines (loss, SBS, GBS) are
quoted by the cost per sample of a run of S samples (docs/GAUSSIAN.md §2.1; `bench/run_cost.py`).

**Scripts.** The GPU benches run directly on a GPU machine (`python bench/<script>.py --device cuda ...`);
`reproduce/` holds one script per published table and figure.

```bash
python3 bench/bench_emitter_slos.py --n 8,10,11                          # strong simulation: ms, peak memory
python3 bench/bench_emitter_sampler.py --impl triton --n 12,16,20,24 --shots 256
python3 bench/bench_emitter_tune.py --impl seq --K 16,24                 # kernel launch configs per K tier
python3 bench/map_threshold.py --engines fock,emitter --ratios 2,3,4,6   # n*(m/n) map, both samplers
python3 bench/compile_check_emitter.py --K 8,16,22 --B 8                 # offline: compile time, registers, spills
```

**Caches.** Triton caches compiled kernels in `~/.triton/cache`; the generated emitter kernel
sources go to `$TMPDIR/photonic_engines_kernels` (override with `PE_KERNEL_CACHE`). Both
survive across processes, so only the first run on a host pays the compile time.

## 9b. Time-bin loop devices (ORCA-type)

A PT-series machine is one rail of `T` time bins closed by loops of delay `l_i` bins through
programmable beam splitters `(theta[i, t], phi[i, t])`. `TimeLoopCircuit` describes it and
unrolls it to a `[T + L, T + L]` unitary (`L = sum(l_i)`; bins first, then the loop slots) that
every engine above accepts — differentiable in the settings, so MerLin can train them:

```python
from photonic_engines.timeloop import TimeLoopCircuit, LoopRegisterKernel, register_dimension

T, loops = 8, (1, 2)                      # 8 bins, a 1-bin loop then a 2-bin loop in series
c = TimeLoopCircuit(T, loops)
theta = torch.rand(c.n_loops, T, dtype=torch.float64)
phi = torch.zeros(c.n_loops, T, dtype=torch.float64)
theta, phi = c.flush_settings(theta, phi)   # last L steps empty the loops onto the detector
U = c.unitary(theta, phi)                    # [11, 11], autograd through theta and phi
inp = [1, 1, 1, 0, 0, 0, 0, 0]               # one photon in each of the first 3 bins
```

The native engine exploits causality instead: only the loop content (a Fock register of
`L` slots, dimension `register_dimension(N, L) = C(N+L+1, L+1)`, polynomial in `N`) is carried
through time, and each bin is projected, sampled or traced as it leaves:

```python
lk = LoopRegisterKernel(c, inp)                       # D = C(7, 4) = 35 register states for N=3, L=3
pats, probs = lk.distribution(theta, phi)            # every output pattern (small N only)
p = lk.probabilities([[1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0]], theta, phi)
marg = lk.bin_marginals(theta, phi, loss=dict(loop=[0.9, 0.95], detector=0.8))   # [T, N+1]
samples = lk.sample(1000, theta, phi, generator=torch.Generator().manual_seed(0))  # [1000, T]
```

`probabilities` and `bin_marginals` are differentiable in `theta`/`phi`; losses per round trip
and detector inefficiency are exact Kraus maps in `bin_marginals` and Monte-Carlo jumps in the
sampler (`threshold=True` gives click patterns). Check against the generic path:
`FockSLOSKernel(N, c.m, inp + [0]*c.L).probs(U)` summed over the slot modes gives the same
distribution (tests/timeloop). The 1 sample/s frontier of the native sampler is a curve
`N*(L)` (bench/timeloop_frontier.py), see docs/TIME_LOOP.md.

For training (MerLin), `TimeLoopModule` holds `theta`/`phi` as parameters and returns a
differentiable readout:

```python
from photonic_engines.timeloop import TimeLoopModule

layer = TimeLoopModule(c, inp, loss=dict(loop=[0.95, 0.9], detector=0.9))   # readout="clicks"
y = layer()                                   # [T] click probability per bin
y.sum().backward()                            # d/d theta, d/d phi
```

The MerLin-style layer adds the classical input: `adapters.merlin.TimeLoopLayer` angle-encodes
features on chosen (loop, bin) settings and returns a readout per input row, so it drops into
a torch model like `merlin.QuantumLayer` does:

```python
from photonic_engines.adapters.merlin import TimeLoopLayer

layer = TimeLoopLayer(c, inp, input_size=2, readout="clicks", loss=dict(loop=[0.95, 0.9], detector=0.9))
model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 2, dtype=torch.float64))
logits = model(x)                             # x [B, 2] features in radians -> [B, 2]
```

Trained on a 2-feature XOR (not linearly separable in the
features), with T = 6, loops (1, 2) and 3 photons, 30 Adam steps reach 79 % train / 66 % test
accuracy from 50 % — the interference in the loops does the work. Readouts: `clicks`, `mean`,
`marginals` [B, T·(N+1)], `patterns`, `distribution` [B, C(T+N−1, N)]. A batch is a loop over
its rows (the register engine has no batch axis over settings; ~40 ms per row at D = 35 with
losses on the CPU).

A spin emitter (§5) can feed the loops instead of — or on top of — classical photons: the
caterpillar's dual-rail bins (2k, 2k+1) become the two rails of a 2-bin period,
`LoopRegisterKernel(c, [0]*T, emitter=EmitterMPS.caterpillar(n, gates, m=c.m))`; the state
carries the spin (χ = 2), a traced spin gives marginals and samples as above, a projected
spin (`final`) gives projected amplitudes and rejection-sampled outputs.

**The exact sampler of record: photon-number sectors** (docs/HYBRID.md §6). In a trajectory the number of photons in
the loops is definite (inputs and emissions add a known number, the beam splitters conserve it, the detected and lost
photons leave), so each shot only needs the register states with its current photon count — not every count up to N.
Same samples, far smaller state; `light_loop_kernel` builds the kernel without the dense basis, so N is not limited by
it:

```python
from photonic_engines.timeloop.hybrid import light_loop_kernel, loop_sector_sample, sector_sample

s = loop_sector_sample(lk, 1000, theta, phi, generator=torch.Generator().manual_seed(0))      # [1000, T], same law
big = TimeLoopCircuit(400, (4,))                                                  # 200 photons, dense D = C(205, 5)
lk_big = light_loop_kernel(big, [1] * 200 + [0] * 200)
s_big, stats = loop_sector_sample(lk_big, 4, torch.rand(1, 400, dtype=torch.float64), return_stats=True)
stats["max_amplitudes"]                       # the largest sector touched: a few hundred, not 3e9
# exactness check: forced outcomes give the dense probabilities
_, logp, _ = sector_sample(lk, 2, lk._gates(theta, phi), force=pats[:2].reshape(2, T, 1))
```

Superposed sources (cos α|0⟩ + sin α|1⟩ per bin, SBS) through the same loops: `SuperpositionLoopKernel` — the photon
number is indefinite, so the register holds every count at once (dense; no sector reduction):

```python
from photonic_engines import ProductSuperposition
from photonic_engines.timeloop.hybrid import SuperpositionLoopKernel

sl = SuperpositionLoopKernel(c, ProductSuperposition.sbs(3, c.m, modes=(0, 1, 2)))
sl.bin_marginals(theta, phi)                  # [T, 4]; sl.sample(...), sl.distribution(...) as LoopRegisterKernel
```

Several emitters on spatial rails, a loop ladder per rail and a rail mixer per tick (carried photons, the general
register spins ⊗ Fock): `HybridRegisterKernel(emitters, rails, loops)` with `sequence.BinEmitter` emitters; `gates`,
`tick_marginals`, `sample`, `sample_sectors`; `register_size()` = 2^k C(p + R, R). docs/HYBRID.md §1–4.

Perceval processors with `TD` components (delay lines, not loops — see docs/TIME_LOOP.md §5)
run on the same engine through `adapters.perceval.periodic_kernel_from_perceval(processor,
input_state)`; `last_period_distribution(kernel, gates)` reproduces `Processor.probs()`.

## 9c. Beyond one device: the distributed CPU engine (experimental)

> Experimental: validated on one node only, API may change; see docs/DISTRIBUTED.md for the
> status and the known limits before relying on it.

When the full distribution does not fit one GPU (n ≥ 12 at m = 2n: 8·10⁸ states, 13 GB),
`photonic_engines.slos.distributed` runs SLOS_full with the amplitude vector split across MPI
ranks (O(M_n/P) memory per rank; the C library from slos_mpi under a cffi binding). Nothing
changes in the way you call it:

```python
from photonic_engines.slos.distributed import DistributedSLOSKernel

k = DistributedSLOSKernel(n, m, occupation, workers=4)   # spawns 4 MPI workers once (pool mode)
v = k.amplitudes(U)                                       # the vector stays on the workers
v.probability(t)          # point lookup;  v.probabilities_of(patterns)
v.marginal(j)             # photon-number distribution of mode j;  v.mean_photon_numbers()
v.top(10)                 # most probable outputs;  v.sample(shots)  exact samples, never gathered
v.gather()                # the whole vector, only when it fits (raises otherwise)
k.close()
```

Under `mpirun -np P python script.py` the same class runs in SPMD mode (every rank calls the
same methods; results are `DistributedVector` chunks).

Before launching anything, `plan()` says whether the run fits and roughly how long it takes
(model fitted on a 48-core node, ±30 %), and every kernel checks it against the memory of the
machine — an impossible request is refused with sizes instead of swapping:

```python
from photonic_engines.slos.distributed import plan
print(plan(13, 26, 48))
# Plan(n=13, m=26, ranks=48: M_n=5,414,950,296, vector 80.7 GB, peak 113.0 GB, per rank 2.54 GB; ~262 s on one node)
DistributedSLOSKernel(20, 40, occ, workers=48)   # MemoryError: ... needs at least 168690 node(s) of this size
```

Perceval users plug it into a `Processor` like any backend instance; sampling from the
distributed vector is the `DistributedSLOSSampler`:

```python
import perceval as pcvl
from photonic_engines.adapters.perceval import DistributedSLOSBackend, DistributedSLOSSampler

strong = DistributedSLOSBackend(workers=8)
p = pcvl.Processor(strong, circuit)                          # probs(), probability(), evolve()
p.with_input(pcvl.BasicState(occ)); p.min_detected_photons_filter(0)
dist = p.probs()["results"]                                  # gathered — only when it fits

s = pcvl.Processor(DistributedSLOSSampler(strong=strong, seed=0), circuit)
s.with_input(pcvl.BasicState(occ))
samples = pcvl.algorithm.Sampler(s).samples(10_000)["results"]   # drawn on the workers, never gathered
strong.close()
```

Requirements: `pip install photonic-engines[mpi]` plus an MPI (`pip install mpich` is enough on
a laptop); on a multi-core server, `bench/scaling_distributed.sh` runs the scaling sweep. It is the
memory-scaling engine, not the fast one: complex128 only, no autograd, one U at a time, and a
single rank is slower than `FockSLOSKernel` on the same CPU (it recomputes the predecessor
indices instead of storing tables). docs/DISTRIBUTED.md has the numbers and the roadmap.

## 9d. Losses

Every engine takes `loss=η`, one uniform end-to-end transmission per photon (brightness ×
chip × detector; docs/LOSS.md). Uniform loss commutes with the unitary, so each engine puts
it where it is cheapest:

```python
from photonic_engines.slos import FockSLOSKernel, sample_from_probs
from photonic_engines.clifford import fock_sample, emitter_sample

U6 = haar(6, 3)
k = FockSLOSKernel(3, 6, [1, 1, 1, 0, 0, 0], dtype=torch.complex128)
ld = k.probs(U6, loss=0.8)             # LossyDistribution: output-side thinning down the basis layers
ld.mass(2)                             # P(2 detected photons) = C(3,2) 0.8^2 0.2
ld.probs[2], ld.occupations(2)         # the 2-photon layer, [S_2] over [S_2, 6] patterns
ld.conditional(3)                      # post-selected on 3 clicks (= the lossless distribution)
ld.as_dict(), ld.clicks()              # {pattern: p}, threshold-detector patterns
occ = sample_from_probs(k.layers, ld, shots=1000)

occ = fock_sample(U6, [0, 1, 2], 10_000, loss=0.8)          # GPU: shots split by surviving photons,
occ = emitter_sample(lc, U8, 10_000, backend="torch", loss=0.8)   # one lockstep pass per k (cheaper than lossless)

lk.bin_marginals(theta, phi, loss=0.8)                      # time loop (§9b): uniform part by thinning,
lk.bin_marginals(theta, phi, loss=dict(loop=[0.9], uniform=0.8))   # loop round-trip loss stays a Kraus channel
```

A tensor `η` gets a gradient through `probs(U, loss=η)`. Perceval backends take
`loss=η` in their constructor (`prob_distribution()` then runs over 0..n photons; cross-checked
against Perceval's `LC` to 1e-16).

### Partial distinguishability

The samplers take `indist=x`, the pairwise overlap of the photons' internal states
(x-model, HOM visibility V = x²; docs/DISTINGUISHABILITY.md). Each photon is, independently, in
the common internal mode with probability x, otherwise in a mode of its own: the interfering
photons of a shot form the instance the kernel runs, the others are placed first — routed with
|U[r, i]|² for Fock inputs, folded into the chain at their drawn row for emitter inputs — and
the outputs add. Exact at every x, with the same cost law as loss (η → ηx):

```python
from photonic_engines.imperfections import Distinguishability

occ = fock_sample(U6, [0, 1, 2], 10_000, indist=0.9)                       # V_HOM = 0.81
occ = fock_sample(U6, [0, 1, 2], 10_000, indist=0.0)                       # fully distinguishable: per(|U|^2)
occ = emitter_sample(lc, U8, 10_000, backend="torch", loss=0.8, indist=0.95)   # both imperfections, one draw
x = Distinguishability.from_hom_visibility(0.9)                            # x = sqrt(V)
occ = fock_sample(U6, [0, 1, 2], 10_000, indist=Distinguishability(species=(0.7, 0.2)))   # two interfering species
```

`indist=1` is the lossless path, object for object. The strong engines take the same argument
and return the exact mixture (a sum over the interfering subsets, `slos/species.py`); `order=b`
keeps the terms with at most b distinguishable photons and drops a mass the binomial tail
bounds, whatever U:

```python
from photonic_engines.imperfections import truncation_bound
from photonic_engines.slos import EmitterSLOSKernel

p  = k.probs(U6, indist=0.9)                       # FockSLOSKernel: the x-model distribution, differentiable in x
p2 = k.probs(U6, indist=0.9, order=1)              # <= 1 distinguishable photon; 1 - p2.sum() == truncation_bound(3, 0.9, 1)
ld = k.probs(U6, indist=0.9, loss=0.8)             # both imperfections: LossyDistribution
pe = EmitterSLOSKernel(lc, dtype=torch.complex128).probs(U8, indist=0.9)      # emitter chain, C(n,b) m^b folded chains
K, coarse = EmitterSLOSKernel(lc, dtype=torch.complex128).kraus_species(U8, 0.9)   # species-resolved Kraus set
```

The register modalities take the same argument: `SequenceKernel(emitters, m, indist=[x_1, ..., x_k])`
(one purity per emitter; the bin's Kraus set is species-resolved, `sample` and the strong side
follow), `LoopRegisterKernel.sample(shots, theta, phi, indist=0.95)` and
`bin_marginals(theta, phi, indist=0.95)` for a time-loop device (§9b).

Perceval: `SLOSTorchBackend(indist=x)` (x = sqrt of Perceval's `indistinguishability`, which is
the HOM visibility), or an annotated `BasicState` (`|{_:0},{_:1},0>`: equal annotations
interfere) — both cross-checked against Perceval's own `NoiseModel` / annotated simulation to
1e-15; `CliffordGPUBackend(indist=x)` for sampling.

## 9e. Several emitters, time-bin trains through one interferometer

k emitters each emit a graph state of n photons as a train of time bins into their own input
modes of a static interferometer (no delay line); the k·n-photon output is the sequence of the
n per-bin patterns. Photons of different bins never interfere, so the sequence is a chain of
per-bin Kraus operators on the product spin register (docs/SEQUENCE.md):

```python
from photonic_engines.sequence import SequenceKernel, BinEmitter
k, n, m = 3, 8, 6
ems = [BinEmitter.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, modes=(2 * j, 2 * j + 1)) for j in range(k)]
sk = SequenceKernel(ems, m)               # chi = 2^k = 8, S_bin = C(m+k-1, k) = 56 patterns per bin
K = sk.kraus(U6, 0)                       # [56, 8, 8]: one SLOS pass, shared by every bin here
marg = sk.bin_marginals(U6)               # [n, 56], differentiable in U6 (and in the spin gates)
p = sk.sequence_probability(U6, seqs)     # seqs: occupations [B, n, m] or pattern indices [B, n]
J = sk.joint(U6, [0, 3])                  # [56, 56]: correlations between bins 0 and 3
occ = sk.sample(U6, 1000)                 # [1000, n, m]: pure trajectories on the register
occ = sk.sample(U6, 1000, loss=0.8); marg = sk.bin_marginals(U6, loss=0.8)
sk_bins = SequenceKernel(ems, m); sk_bins.bin_marginals(torch.stack([U6] * n))   # one unitary per bin
```

`U` may be a Perceval circuit. k is the exponent (4^k per Kraus operator), n is linear.

The Kraus set stops at k ≈ 6–8 (S_bin χ² entries per bin). For label-preserving emitters
(caterpillar, linear cluster) `sample_diag` runs the same trajectories without it: per bin it
draws the label word, then the pattern from that k-photon Fock instance (the Clifford sampler on
cuda, strong SLOS per distinct word on the CPU), then updates the 2^k register by
`label_amplitudes` (O(k 3^k), repeated output rows collapsed). Same distribution as `sample`:

```python
occ = sk.sample_diag(U6, 200, generator=torch.Generator().manual_seed(0))   # [200, n, m]
occ = sk.sample_diag(U6, 200, policy=lambda b, counts, psi: None)         # feed-forward hook (SEQUENCE.md §10)
```

On cuda, `draw="auto"` draws the label word on the device and `fock_backend="triton" | "cuda"`
picks the Fock kernel of the tick; the register frontier is k ≈ 22.6 emitters at one tick-shot
per second on an H100 (docs/SEQUENCE.md §12b).

## 9f. Non-linear boson sampling: photon-number phases between interferometers

A site is a single-mode phase that depends on the photon number, `exp(-i f(N_x))` (Kerr: `f(j) = phi j^2`),
placed between two interferometers. Since a site sees at most n photons it is an exact finite sum of ordinary
phase shifters, so the circuit is a coherent sum of linear interferometers and every engine of the package
applies branch by branch (docs/NONLINEAR.md):

```python
from photonic_engines.nonlinear import NonlinearCircuit, branches
U0, U1 = haar(6, 1), haar(6, 2)
C = NonlinearCircuit([U0, U1], [0, 1, 2], [(1, 3, math.pi / 2)])   # Kerr pi/2 on mode 3, between U0 and U1
theta, c = branches(math.pi / 2, 3)       # 2 phase shifters reproduce exp(-i phi j^2) on j = 0..3
p = C.probs()                             # [S] over C.occupations(); route "branch" | "gate" | "auto"
occ = C.sample(2000, generator=torch.Generator().manual_seed(0), method="strong")     # small n
C.rejection_factor()                      # 2.0 proposals per sample of the exact sampler (method="rejection")
```

`sample()` (default `method="rejection"`, `backend="triton"` or `"cuda"`) is exact at any n the Fock sampler
reaches: it draws a branch, takes a `fock_sample` of that linear interferometer and accepts with a
Cauchy-Schwarz ratio, so it costs `rejection_factor()` proposals per sample. `loss=eta` is the transmission
before the sites, `eta_post=` after them and `indist=x` the x-model; several sites live in one layer (distinct
modes) or in successive layers, and the branches multiply. `bench/nlbs_frontier.py` measures n*(phi, L, m/n).

## 9g. Adaptive linear optics: mid-circuit photon counting with feed-forward

A Fock input goes through `U0`; some output modes are then measured level by level with number-resolving
detectors, and after each level a feed-forward unitary, chosen from the counts read so far, acts on the
modes still unmeasured (docs/ADAPTIVE.md; notation in its §1):

```python
from photonic_engines.adaptive import AdaptiveCircuit, mode_marginals
U0 = haar(6, 1)
policy = lambda lam, p: haar(5, 10 + p[0])     # V_1 on the 5 unmeasured modes, picked by the count on mode 1
C = AdaptiveCircuit(U0, [0, 2, 4], [[1]], policy)   # 3 photons; one level: count mode 1, then feed forward
p = C.probs()                                  # [S] joint law of the full occupation, over C.occupations()
q = C.outcome_probs()                          # {(count on mode 1,): probability}
mm = mode_marginals(U0, [0, 2, 4], [1, 3])     # [n+1, n+1] exact photon-count law of modes 1 and 3 behind U0
occ = C.sample(2000, generator=torch.Generator().manual_seed(0), final="slos")   # small n, CPU
```

The default `sample()` (`final="prefix"`, `backend="triton"` or `"cuda"`) draws the adaptive counts from
exact marginals and the rest with the Clifford & Clifford chain, its first photons forced onto the measured
modes, at a mean cost of E[2^r] chains per sample (r photons read on the adaptive modes); `loss=` and `indist=` work as in `fock_sample`.

## 9h. Gaussian boson sampling: squeezed inputs

`GaussianInput` holds single-mode squeezed vacua (optionally displaced), one per input mode; the photon
number is indefinite and unbounded. `gaussian.sample` is the exact photon-number-resolving chain rule over
conditional Gaussian states (docs/GAUSSIAN.md), with no photon cutoff:

```python
from photonic_engines import GaussianInput, gaussian
st = GaussianInput.smsv(3, 6, r=GaussianInput.r_for_mean(0.5))   # 3 squeezers, 0.5 photons each on average
U = haar(6, 1)
occ = gaussian.sample(st, U, 500, seed=0)                  # [500, 6] int64 counts; CPU (device="cuda" for the hafnians)
lossy = gaussian.sample(st, U, 500, loss=0.7, seed=1)      # exact: pure squeezing + classical noise per shot
impure = gaussian.sample(st, U, 500, schmidt=(0.9, 0.1), seed=2)   # two Schmidt modes per squeezer
gaussian.pure_probability(st, U, (1, 1, 0, 0, 0, 0))       # one loop hafnian (lossless input)
st.photon_number_probs(20)                                 # exact law of the total photon number, U-independent
```

`states.GaussianInput.truncated(d)` turns the input into a `ProductSuperposition` cut at d − 1 photons per
source, which `ProductSLOSKernel` simulates exactly for every lossless pattern with at most d − 1 photons —
the oracle of the tests. `slos.GaussianSLOSKernel(st, max_photons=N).probs(U)` is the strong side on the
SLOS layers, exact for every lossless layer up to N.

A published device is a lossy transfer matrix T (non-uniform loss, loss inside the interferometer) rather
than a unitary with uniform loss at the sources. `sample_transfer` takes T directly — the pure part and the
classical noise from `transfer_decomposition`, exact, no truncation:

```python
T = 0.8 * haar(6, 1)                                       # any sub-unitary transfer matrix
occ = gaussian.sample_transfer(st, T, 200, seed=0)         # [200, 6]
dec = gaussian.transfer_decomposition(st, T)               # reuse across calls: decomposition=dec
occ = gaussian.sample(st, U, 50, seed=0, precision="dd")   # the pair sieve in double-double ("fp64" | "dd" | "auto")
```

**GPU.** On cuda the loop hafnians run on the device (the v2 pair sieves, batched over collision groups;
`precision="auto"` switches to double-double where fp64's round-off bound fails). The frontier comes from a calibrated timing model of the
sampler (`gaussian.timing`, docs/GAUSSIAN.md §2.2–2.3): n* = 41.5 squeezers at m = 2n, 0.5 photons per
squeezer, 1 s per sample in a run of 10⁴.

## 10. Which engine when

| need | engine | reach (one H100, m = 2n) |
|---|---|---|
| full distribution, Fock input, gradients | `FockSLOSKernel` | n ≤ 11 |
| full distribution, superposition input | `SuperpositionSLOSKernel` | n ≤ 11 (K components) |
| full distribution, emitter state; Kraus ops | `EmitterSLOSKernel` | n ≤ 11 |
| samples, Fock input | `fock_sample` | n* = 38.5 at 1 sample/s (42.5 at m = n) |
| samples, Fock input, from the full distribution | `clifford.variants.strong_then_sample` | n ≤ 11 (memory of the distribution) |
| samples, emitter state | `emitter_sample` (backend triton, `kernel="auto"`) | n* = 30.6 at 1 sample/s |
| one probability / amplitude of an emitter state to ±ε | `slos.estimate.EmitterGurvitsEstimator` | polynomial in n, ~1/ε² samples |
| samples, superposition (SBS) input | `clifford.sample(ProductSuperposition)` | n* ≈ 20 sources (provisional; docs/SUPERPOSITION.md) |
| samples, emitter state, CPU / reference | `emitter_sample` (backend torch) | n ≤ 14 |
| full distribution beyond one device (n ≥ 12) | `DistributedSLOSKernel` (CPU, MPI) | n = 12, m = 24 on one 20 GB node; +1 photon per ~8× ranks |
| Perceval front end | `adapters.perceval` backends | as above |
| time-bin loops (ORCA-type), samples / marginals / gradients | `LoopRegisterKernel` | polynomial in N for fixed loop length L: N*(L), docs/TIME_LOOP.md |
| time-bin loops, exact samples, single photons or one emitter | `timeloop.hybrid.loop_sector_sample` (+ `light_loop_kernel`) | cost ~linear in N at fixed L; frontier in L (being measured, docs/HYBRID.md §5–6) |
| time-bin loops, superposed sources | `timeloop.hybrid.SuperpositionLoopKernel` | dense register C(N+L+1, L+1), docs/HYBRID.md §5 |
| k emitters, rails, loop ladders, rail mixer (carried photons) | `timeloop.hybrid.HybridRegisterKernel` | register 2^k C(p+R, R), docs/HYBRID.md |
| k emitters × n bins through a static interferometer: sequence probabilities, bin marginals, samples | `SequenceKernel` | Kraus set: k ≤ 6–8 emitters; `sample_diag`: k ≈ 22.6 at one tick-shot per second; any n; docs/SEQUENCE.md |
| mid-circuit photon counting + feed-forward: joint law, mode marginals, exact samples | `AdaptiveCircuit` | strong: one SLOS pass per branch; samples: E[2^r] Clifford chains per sample (docs/ADAPTIVE.md §4b) |
| non-linear sites (Kerr) between interferometers: full distribution, exact samples | `NonlinearCircuit` | strong: n ≤ 11 divided by the branches; samples: the Fock frontier minus the toll of `rejection_factor()`, docs/NONLINEAR.md |
| squeezed (Gaussian) inputs: exact PNR samples, per-pattern probabilities | `gaussian.sample`, `gaussian.sample_transfer`, `gaussian.pure_probability` | n* = 41.5 squeezers (0.5 photons each, run of 10⁴; timing model, docs/GAUSSIAN.md §2.3) |
| squeezed inputs: full distribution up to N photons (exact lossless) | `slos.GaussianSLOSKernel` | not yet measured on a GPU (docs/GAUSSIAN.md §7) |

Measurements: the run stamps quoted in each design note (`runs/<stamp>/`). Algorithms:
docs/SLOS_IMPLEMENTATION.md, docs/ALGORITHMS.md, docs/EMITTER_SAMPLER.md, docs/GAUSSIAN.md.
