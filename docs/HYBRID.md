# Carried-photon engine: emitter trains through loops (spins ⊗ Fock register)

Status: **M1 (CPU, exact) — 2026-10-01**, `photonic_engines/timeloop/hybrid.py`
(`HybridRegisterKernel`), tests `tests/timeloop/test_hybrid.py`. M2 (GPU + loss read-out), M3
(frontier runs) and M4 (hand-over) are open. This is the one empty cell of the paper's part (b) (loop interferometers × emitter
input; see the engine map of docs/ARCHITECTURE.md). No measured number in this note yet.

## 1. The model

**Device.** k spin emitters (spin dimension d_j, 2 for the caterpillar / linear-cluster emitters)
emit one photon per tick each. Emitter j's labels go to its own **rails** (spatial modes; a
dual-rail emitter owns two rails, label l → rail `modes[l]`, distinct across emitters as in
`SequenceKernel`). The emission at tick t is the `BinEmitter` operator set: Σ_l M_{j,t,l} ⊗ a†_{rail l},
any spin gate folded into M (label-preserving ops as in `EmitterMPS` / `BinEmitter`). On each rail
r a loop ladder (delays ℓ_{r,i} in ticks; any ladder per rail, `()` for none), each loop closed on
the rail by a beam splitter G(θ, φ) with one setting per loop and per tick (the `TimeLoopCircuit`
convention). Per tick a rail mixer U_t (m × m, static or per tick: the PIC of `SequenceKernel`).
Detection per rail per tick (PNR). Classical Fock photons can be injected on the rails as well
(`inputs [T, m]`, before the emissions).

**One tick** (= one period of `PeriodicNetworkKernel`):

1. inject the classical photons, then the k emissions (each acts on its own spin factor);
2. loop couplers: for every delay line, round-trip damping on its head slot, then G(θ, φ) on
   (rail, head slot);
3. the rail mixer U_t, as a fixed mesh of m(m−1)/2 + ⌈m/2⌉ two-mode gates (`mesh_decompose`:
   Reck elimination, the diagonal as phase gates on disjoint pairs, identity gates kept so the
   structure is the same for every U_t). `mixer="before"` swaps steps 2 and 3;
4. detector damping on the rails; the rails exit — projected (amplitudes), sampled
   (trajectories) or traced (density matrix).

Default order is "loops, then mixer": a photon released from a loop interferes with the fresh
photons of the tick in the PIC (the "undetected outputs of PIC t routed into PIC t+1" reading of
SEQUENCE.md §7); "before" is the ORCA-like reading (the PIC feeds the loops, the loops feed the
detectors).

**Register.** The quantum memory carried from tick to tick is spins ⊗ Fock register on
R = m + Σ ℓ modes (the m rail wires in transit + every delay-line slot), with at most p photons:

    size = Π d_j · C(p + R, R)      (= 2^k C(p + c, c), c = R: the paper's knob)

`HybridRegisterKernel.register_size()` returns exactly this number, and `hybrid_register_dimension(k, p,
rails, slots)` computes it without building anything. The spin axis is a product space; an emission
acts on its own factor only (reshape [pre, d_j, post] — never a dense χ × χ operator, so the
operators cost k · T · d² instead of k · T · χ²).

**p — exact by default, as for ORCA.** A shot is one finite train of T ticks into empty loops
(the convention of `LoopRegisterKernel` and of Novák et al.); photons still in a loop after the last
tick are lost unless the last settings flush them. `carry=None` (default) is exact: p = every photon
the train and the inputs put in the register, n_total = kT + Σ inputs, and the register is
2^k C(kT + R, R) — the analogue of the loop engine's C(N + L + 1, L + 1) with N = kT. **The frontier
runs (M3) are exact**: the largest train (k, T, ladder) sampled at 1 Hz, curves comparable to the
loop row's N\*(L). The dense basis over-counts the reachable register (as in TIME_LOOP.md §4), so
that frontier is an implementation boundary, not a mathematical one.

`carry=p < n_total` is an **option** beyond the exact frontier only (a sensitivity reading, never a
headline number): there is no structural bound below n_total — a slot is a Fock mode and a beam
splitter with θ ∉ {0, π/2} keeps amplitude in a loop — so components that would exceed p are dropped
at emission (the raise has no target in the basis) and the state becomes sub-normalised. The dropped
mass is reported:

- exactly, `leaked_mass()` = 1 − (trace after the last tick) from the density-matrix march;
- by the sampler: `sample(return_weights=True)` returns w = Π_t (surviving squared norm at
  emission t) per shot; E[w] = 1 − leaked mass (unbiased), w = 0 flags a trajectory that lost
  every component.

The capped sampler renormalises after each emission, so its rows are not draws from the exact
distribution; their path weights w say how much mass the cap removed along each path. Any capped
point quoted must carry its leaked mass.

**Limits** (each a test, `tests/timeloop/test_hybrid.py`): no loops → `SequenceKernel`
(`sequence_probability`, 1e-12); no emitter, one rail → `LoopRegisterKernel` (distribution and
lossy bin marginals with loop + detector channels, 1e-12); T = 1 → the k-emission stage
(`EmitterSLOSKernel` on `SequenceKernel.bin_mps(0)`).

**Oracle.** The unrolled network over T ticks — m' = T·m + Σ ℓ wires (tick t's rail r is wire
t·m + r, slot s of delay line i is wire T·m + off_i + s), every tick's loop gates and the full U_t
applied on the wires in the kernel's order, built in the test by explicit matrix products,
independently of the register code — fed with the product emitter MPS (all k·T emissions,
tick-major, on the product spin) into `EmitterSLOSKernel`; slot wires marginalised. This is
TIME_LOOP.md §2's "the generic path covers it today", verified at k ≤ 2, T ≤ 4, Σ ℓ ≤ 3 per rail,
traced and projected spin: probabilities ≤ 1e-12 (observed ~1e-17), per-tick marginals
≤ 1e-12, norm 1. Losses (round-trip damping on the head slots, detector damping on the rails): the
same unrolled network with one environment wire per loss event (a beam splitter of transmission η),
marginalised — per-tick marginals ≤ 1e-12, the jump sampler within 5σ per entry. η = 1 reduces
bit-identically (marginals and sampler, same seed).

**What already existed.** `LoopRegisterKernel(emitter=)` (2026-09-11, TIME_LOOP.md §5) is the k = 1,
one-rail case with time-bin labels (period = 2 bins). The hybrid kernel adds: k emitters on a
product spin applied locally, m spatial rails with a ladder each, the per-tick rail mixer, the
photon cap with its leak accounting, and a block-wise rail trace for the density-matrix path (the
inherited trace runs Q full passes over ρ, Q = C(p + m, m) rail patterns — prohibitive at m = 4).

**Out of scope.** Partial distinguishability (refused by the emitter paths of the loop kernels
already; not planned before submission). Spin-side errors. Feed-forward (`policy`) is M3 item 4,
optional.

## 2. Pseudo-code (v1, 2026-10-01)

```
HybridRegisterKernel(emitters[k], rails m, ladders, T, carry p, inputs, mixer)
  basis   <- RegisterBasis(L = sum of delays, N = p or n_total, n_rail = m)      # D = C(N + m + L, m + L)
  program <- per delay line i: (loss slot_i "loop_i"), (gate rail(i), slot_i, g_i)
             mixer pairs (a, b) of mesh_pairs(m), gate index n_dl + g
             order: loops + mixer ("after") or mixer + loops ("before"); then (loss rail r "detector")
  gates(theta, phi, U) -> [n_dl + n_mix, T, 2, 2] = G(theta, phi) ; mesh_decompose(U_t)

  march(psi [S, chi, D], t):
    psi <- inject classical photons of tick t                                  # gather
    for j in emitters:                                                          # local on spin factor j
        psi <- sum_l  (M[j, t, l] on axis j)  (raise on rail(j, l)) psi           # cap: no target -> dropped
    for op in program:
        gate (a, b, g): Fock blocks of gates[g, t] on the (a, b) pair groups    # fock_blocks, one bmm
        loss (a, name): amplitude-damping Kraus (rho) / Monte-Carlo jump (sample)
    exit rails:
        project q        -> psi <- psi restricted to rail pattern q, rail cleared
        sample           -> q ~ ||P_q psi||^2 ; project ; renormalise ; w *= ||psi after emission||^2 / ||before||^2
        trace (rho)      -> p_t(q) = tr P_q rho ; rho <- sum_q P_q rho P_q moved to the cleared states (per block q)
  end: spin projected on `final` (amplitudes, rejection sampler) or traced
  leaked_mass = 1 - sum_q p_T(q)      (carry = p)
```

## 3. Cost (per tick, per trajectory; counts, not measurements)

χ = Π d_j, D = C(p + R, R). Emissions: k · Lab gathers over χD and k local spin contractions
(χD · d_j each). Loop gates: one fused block matmul per delay line, ~χD(p+1) multiply-adds when
the padded groups are dense. Mixer: m(m−1)/2 + ⌈m/2⌉ two-mode gates on the
rail wires, the same per-gate cost — for m ≥ 4 the mixer dominates the loops. Sampling: one
multinomial over the Q rail patterns. Density-matrix path: the same gates on (χD)² entries.
The frontier is expected where χD reaches the dense-register limit of the loop engine (D ≈ 10⁷ at
1 sample/s on one H100, TIME_LOOP.md §4) — to be measured (M3), not assumed.

## 5. The paper's loop row: one rail, three inputs (2026-10-01)

Part (b) of the paper compares inputs on ONE device: one source on one rail, one loop ladder, detection
bin by bin. The three inputs and their kernels:

| input | kernel | register (dense) |
|---|---|---|
| N single photons in bins 0..N−1 (the LBS row) | `LoopRegisterKernel(c, [1]*N + [0]*(T−N))` | C(N + L + 1, L + 1) |
| one spin emitter, N-photon caterpillar, photon b in time bins (2b, 2b+1) | `LoopRegisterKernel(c, [0]*T, emitter=st)` (2026-09-11) | 2 · C(N + L + 2, L + 2) |
| N superposed sources cos α\|0⟩ + sin α\|1⟩ in bins 0..N−1 | `SuperpositionLoopKernel(c, ProductSuperposition.sbs(N, …))` | C(N + L + 1, L + 1) |

The emitter's period is two bins (its photon is in a superposition of early and late), so two rail
wires are in transit at the emission: one more register mode than the single-photon input, and the
spin. `HybridRegisterKernel` (k emitters, spatial rails, a rail mixer) stays in the library as the
generalisation (the spin-optical configuration of de Gliniasty et al., Quantum 8, 1423 (2024));
it is not the paper row.

**SBS through loops** (`SuperpositionLoopKernel`, new): the source of bin b acts as
ψ → Σ_q a_q/√q! (a†)^q ψ on the empty rail, coherently; the register is the graded basis up to N
photons — exactly the Fock register's size, since the dense register already holds every photon
number ≤ N. Validated against `ProductSLOSKernel` on the unrolled unitary (every photon-number block,
marginals ≤ 1e-12; sampler within 5σ); α = π/2 reproduces the Fock input bit for bit.

**Measurement** (`bench/hybrid_frontier.py`): the method and settings of
`bench/timeloop_frontier.py` (T = 2N bins, one loop of delay L, θ ∈ [0, 2), φ ∈ [0, 6), 1 sample/s,
complex128). **Sampler of record (2026-10-01): photon-number sectors (§6) for single
photons and the emitter; the dense register for superposed sources** (their photon number is
indefinite, §6). Two readings: N\*(L) at fixed L (`--loops`) and L\*(N) at fixed N (`--scan-l`), the
second being the natural one once the cost is linear in N. Plot: `bench/plot_hybrid_frontier.py`.

Run 20261001_101803 (H100 PCIe, dense register): single photons reproduce run 2
(N\* = 267.4 / 241.4 / 115.8 / 62.8 / 31.5 / 21.6 at L = 1 / 2 / 3 / 4 / 6 / 8); superposed sources
267.4 / 237.4 / 112.9 / 62.5 / 31.3 / 21.4 — the dense curve of the Fock register, as expected. These
dense single-photon numbers are now the control, not the row: the row is re-measured with sectors.

**Photon flux (2026-10-02).** The LBS convention puts the N single photons in bins 0..N−1 (one per
bin, then N empty bins), while the emitter emits one photon per 2-bin period: at the same T = 2N the
single-photon train carries twice the flux early on, the loops fill faster and the sectors are
larger. CPU check, N = 40, sectors (2 cores, indicative): mean / max photons in the loops and largest
sector — L = 8: single photons packed 4.2 / 8.0 / 4.9·10⁵, single photons every 2 bins 3.6 / 4.2 /
4.4·10⁴, emitter 3.7 / 4.3 / 1.8·10⁵. At equal flux the emitter is the harder input (the spin and the
second rail wire of its period), as it should be; the packed single-photon curve is not its like-for-
like comparison. `--spacing 2` puts one source every 2 bins (single photons, SBS).

**Fock limit at the frontier (limit test, docs/ALGORITHMS.md §9.2; run 20261005_170728).** SBS-L at α = π/2 (every source emits one
photon) on the dense register, settings of run 20261001_101803: N\* = 265.4 / 237.2 / 112.3 / 62.5 / 31.0 / 21.4 at
L = 1 / 2 / 3 / 4 / 6 / 8 against the single-photon control 267.4 / 241.4 / 115.8 / 62.8 / 31.5 / 21.6 (≤ 3 %), same
register at every N. Per sample at equal N the SBS path costs 1.01–1.15× the Fock-input path, the same as at α = π/4
(0.99–1.08× those rows): the residual is the injection step (`_emit`, one raise-table pass per source tick), not α.

## 6. Photon-number sectors (v2 sampler, 2026-10-01)

In a trajectory the register photon number is definite: a classical input adds its photons, an
emission adds exactly one, gates conserve, the rail exit removes the detected count, a loss jump the
lost one. So a shot needs only the sector of its current photon number n — between periods the L
slots with exactly n photons, C(n + L − 1, L − 1) states, during a period the R = n_rail + L register
modes with exactly n photons — instead of the dense register of every number ≤ N. The dense size
counts photons the loops never hold at once.

`sector_sample(K, shots, gates, chan, generator, force)` runs any register kernel's program (inputs,
emissions — dense or local spin ops —, gates, loss jumps, rail exit, spin projection by rejection)
on per-sector tables (`_SectorTables`: compositions of exactly n photons, byte-key lookup, pair /
raise / loss / exit tables built on demand), the shots grouped by their current n; Fock blocks per
period are sized to the largest sector present. `loop_sector_sample` is the `LoopRegisterKernel.sample`
front end (uniform loss by thinning), `HybridRegisterKernel.sample_sectors` the multi-emitter one;
`light_loop_kernel` builds a loop kernel without its dense basis (the sector sampler needs only its
program), so N is not limited by the dense basis at construction.

**Not for superposed sources**: an SBS (or Gaussian) input keeps a coherent superposition of photon
numbers in the register, so every sector is populated at once — the graded register is the minimum.

**Validation.** `force=` drives the march along given outcomes and returns their log-probabilities:
equal to the dense `probabilities` on every pattern of the distribution (Fock, traced emitter,
projected emitter; ≤ 1e-12, observed ~1e-16), also from the light kernel; the sampled bin marginals
with loop and detector jumps within 5σ of the dense density-matrix marginals; the multi-emitter
kernel against the unrolled oracle.

**What it does to the cost (CPU, 2-core cloud workspace, θ ∈ [0, 2) random settings as in the LBS
row — indicative only, not paper numbers):**

| input, L, N (T = 2N) | dense χ·D | largest sector χ·D touched | max mean n in the loops | samples/s |
|---|---|---|---|---|
| Fock, L = 4, N = 93 | 6.8·10⁷ | 1 365 | 8 | 40 |
| Fock, L = 6, N = 40 | 6.3·10⁷ | 38 760 | 9 | 29 |
| Fock, L = 8, N = 40 | 2.1·10⁹ | 490 314 | 11 | 2.5 |
| emitter, L = 4, N = 40 | 1.9·10⁷ | 1 584 | 4 | 60 |

At fixed L the per-shot cost becomes linear in N (more periods, the same sectors): the dense
register's growth in N is what set the LBS row's N\*(L) (run 2: 63 at L = 4, 31.5 at L = 6, 21.6 at
L = 8 on the H100), and these CPU points already pass it. The frontier of the exact sampler at these
settings is in L (the sector size, exponential in L through the occupancy), not in N. **This changes
the measured LBS row and its reading** (TIME_LOOP.md §4: "N ≳ 40 and L ≳ 6"). The sector
sampler is the method of record for the loop column (2026-10-01); the LBS row is re-measured with it
and replaces run 2. The occupancy depends on the settings: these are
random θ; adversarial settings that keep photons in the loops raise it.

**Table building is a one-time cost, reported apart (2026-10-01).** The sector tables (bases, pair /
raise / loss / exit tables per sector) are built on the host the first time a sector is reached and
cached on the kernel; a rare large sector reached late in a run used to land its build (np.unique over
~10⁷ rows, per pair) inside a timed batch. The first GPU run with sectors (20261001_144452) was
contaminated that way at large L — cliffs of 60–300× in rate between neighbouring L at sector sizes
~10⁷, e.g. single photons, N = 40: 23.8/s at L = 9, 0.36/s at L = 10. Since then `sector_sample`
returns `table_seconds` and `bench/hybrid_frontier.py` subtracts it from the sampling time (CSV column
`table_seconds`), as the dense basis build is excluded; on 2 CPU cores, N = 20, L = 10: 2.1 → 13.8
samples/s. Run 20261001_144452 is superseded. Run 20261001_180807 still had one hole: when a point's
ramp stopped below 4 shots the fallback rate used the raw batch time (tables included), which made
every upper bracket at large L a 1–2-shot point at 0.005–0.05 samples/s; only the brackets with ≥ 4
shots on both sides stand (single photons: N\* = 1943 at L = 8, 739 at L = 9; emitter: 896 at L = 11),
the rest are lower bounds. Fixed: the fallback, the budget and the ramp all use sampling time, and a
16-shot untimed warm-up builds most tables first.

**Statistic (2026-10-02).** The per-sample cost of the sector sampler is heavy-tailed (a rare
trajectory reaching a large sector dominates); the rows are quoted as t_S (the run-of-S statistic, S = 10⁴) from brackets with ≥ 64 shots
on both sides, and the bench's rate over those shots is the empirical t_S (with ≤ 256 shots the 1 − 1/S truncation
removes none of them, so t_S = the shot mean; the tail beyond ~1/64 is unseen and stated as such). A bracket with a
side under 64 shots is quoted as a lower bound on its 64-shot side only (emitter L = 12: N\* ≥ 856, upper side
16 shots).

**Large L (runs 20261002_222344, 20261003_071044).** At L ≥ 11–13 the sector-table build on the host takes
100–1060 s per point and the cached tables exhaust host RAM (a run killed, exit 137): the limit there is the table
implementation (host numpy, int64 indices, one pair table per slot position), not sampling. Points beyond are lower
bounds.

**k emitters (`--input hybrid`, 2026-10-04).** The multi-emitter register measured on the same
sampler: k spin emitters, emitter j dual-rail on rails (2j, 2j + 1), one photon per emitter per tick, a loop of delay L
on every one of the m = 2k rails, a static Haar mixer after the loops; N = kT photons; register 2^k C(N + R, R),
R = m(1 + L) in the dense count. `light_hybrid_kernel` builds the kernel without its dense basis (validated against the
dense kernel's forced-pattern probabilities and the unrolled oracle, `test_sector_sampler_hybrid_kernel`). CPU
indication (2 cores, target 5 samples/s, not a paper number): k = 2: N\* > 60 at L = 1, 17.9 at L = 2, 10.7 at L = 3 —
every rail carries a loop and every tick feeds k photons, so the occupancy, and the frontier in L, falls much faster
than for one emitter on one rail.
Measured (run 20261005_083900, H100): k = 2: N\* > 1574 / 1641.7 / ≥ 24 at L = 1 / 2 / 3; k = 3: 1321.9 at L = 1, out of
memory at one shot from L = 2; k = 4: ≈ 40 at L = 1 (16–32-shot bracket), out of memory from L = 2.

## 6c. The sector march, v3 (2026-10-12): launches and host syncs per tick

The v2 march spent its fixed per-batch cost on kernel launches and host syncs, not arithmetic. Profile, ops per tick
on one loop: ~150-200 launches and ~4 syncs per photon-number group, plus a per-tick rebuild of the gates' Fock
blocks at ~32 launches per photon level. As a result, t_S depended on the batch size: a + b·s per batch of s shots,
with a ≈ 21 s at N = 1259, L = 8 (run 20261012_004429).

v3 (`_sector_march`, the only march since 10-12; validated in run 20261012_075702) makes three changes:
- **Fock blocks per chunk of ticks.** One batched `fock_blocks(V[:, t0:t1], cap)` at a running photon cap, instead
  of one build per tick.
- **One host sync per stage for all groups.** Outcomes are drawn on the device per group; the shots are regrouped by
  a device-side stable sort plus scatter_add counts, read once (`_split_by`). The exit draws all outcomes of a group
  at once from a padded table `exit_x(n)` [Q, Ds(n)+1]: row q holds the states with rails = q in slot(n − |q|) order,
  so the first Ds(n − d)+1 columns are that slot sector.
- **Ext form.** Every state carries one trailing zero column (index D), and every gather table ends on that
  sentinel. "Absent" reads zero without a `torch.cat` copy, and every output is again in ext form. The fused gate
  adds one all-sentinel group, whose zero output supplies the trailing zero.

```
per tick t:
  for each group n (rows, psi [S, chi, Ds(n)+1]):
    x = psi[..., emb_x(n)] ; inputs: x = x[..., src] * coef (raise_x) ; emissions as v2
  B = blocks(t)                                  # cached chunk, recomputed if n_max > cap
  program: gate -> _sector_gate_x per group ; loss -> per group draw j, then _split_by (1 sync) and merge
  exit: per group p = index_add(|x|^2 by rail pattern), q ~ p, d = |q|, y = gather(x, exit_x(n)[q]) / norm
        _split_by(d) (1 sync) ; group n - d gets y[idx][..., :Ds(n-d)+1]
```

Run 20261012_075702 (H100):
- **Law:** the v2-v3 two-sample per-bin z stays within the expected range (max 2.62 over 192 bins at N = 96, L = 8;
  2.70 over 256 bins at N = 128, L = 6).
- **Host syncs:** 53.1 → 1.1 per tick.
- **Speed-up:** 1.6-2.4× per sample at N = 96, 128, 1259 × L = 1, 6, 8 × batches 4, 64, 256. Example: N = 1259,
  L = 8, 256 shots, 0.148 → 0.093 s per sample.
- **Kept:** results/keep/hybrid_sector_speed_fixes123_20261012.csv.

## 6d. One loop with Fock input: the register is one basis state (Fock chain, 2026-10-12)

**Statement.** With one loop of delay L (stride 1, one rail), Fock input and number-resolving detection, the loop
register is a single Fock basis state between ticks (the same observation underlies ORCA's progressive
simulator). Induction:
- the loop starts empty;
- at tick t the rail holds exactly c_t photons, and the head slot (t mod L) holds a definite s;
- the gate conserves k = c + s;
- the rail is measured at once, so the head keeps the definite k − d.

So the sector vector of §6 always held ONE non-zero amplitude here. The exact sampler is a Markov chain on the L slot
occupations (`_loop_chain_sample`), with amplitudes A(s, d) = <d, k−d| V |c, s> from the closed form `gate_column`:

```
occ[S, L] = 0
per tick t:  s = occ[:, t mod L]
             s ~ Binomial(s, eta_loop)                   (loop loss on the head, if any)
             d ~ |A_t(s, .)|^2                           (A_t: closed form, chunk of 64 ticks, s <= max loop total + inputs)
             occ[:, t mod L] = c_t + s - d
             out[:, t] = d, or ~ Binomial(d, eta_det)    (detector loss)
```

- **Cost:** O(s + c) per shot per tick, ~12 batched ops per tick, one host sync per 64 ticks; no tables.
- **Validated:**
  - `force=`: equal to the dense `probabilities` on every pattern at L = 1, 2, 5 (packed, spaced, 2-photon bins);
  - `gate_column` against `fock_blocks` (c = 0..3), normalised to 1e-12 up to s = 400;
  - loss marginals on the CPU and on CUDA.
- **Reached through:** `sector_sample` / `loop_sector_sample` route a kernel to the chain whenever `_head_program(K)`
  matches (the default since 2026-10-12; GPU-validated in run 20261012_081240). Calling
  `_sector_march(..., chain=False)` forces the march, so tests and `bench/hybrid_sector_speed.py` can compare the two.
- **Scope:** Fock inputs on one loop only. It does not apply to superposed or squeezed inputs (indefinite rail
  number), emitter trains (spin-entangled bins), several loops or the k-emitter mixer (a mode meets more than two
  gates), or threshold detection. The fully dephased superposed source is a classical draw of c, so the chain covers
  it. Consequence for part (b): the one-loop single-photon rows of §5 were measured with the sector
  sampler, which is not the best algorithm for that input (the parity rule of docs/ALGORITHMS.md §9 asks for the
  best one). They were re-measured on the chain in §6e.

## 6e. The loop rows on today's sampler (runs 20261016_021007, 20261016_023739, H100 PCIe)

Same harness and protocol as §5 (`bench/hybrid_frontier.py`, T = 2N, one loop of delay L, random settings, seed 0,
1 sample/s, brackets with ≥ 64 shots on both sides; a capped row ends at the last N measured below `--max-n`); march
v3 for the emitter and k emitters, the Fock chain for single photons.

| L | 1 | 2 | 4 | 6 | 8 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| single photons, packed (chain) | > 86 476 | | | | > 85 403 | | | > 75 520 (10⁵ at 1.89/s) | |
| single photons, one every 2 bins (chain) | > 86 476 | | | | | | | > 75 355 | |
| one emitter (march v3) | > 3142 | > 3142 | > 3142 | 4204.4 | 3492.8 | 2950.1 | 2165.8 | ≥ 943 | ≈ 300 (8 shots) |

k emitters (march v3): k = 2: 3142.7 / 1853.0 at L = 1 / 2, memory-limited at L = 3 (≥ 16; out of memory from N = 56,
largest sector 1.5·10⁸ amplitudes); k = 3: 1440.4 at L = 1, memory-limited at L = 2; k = 4: memory-limited at L = 1
(32 photons on 16 shots), out of memory at one shot at L = 2.

- Single photons: the chain costs ~0.19 ms per tick whatever the batch, flat in L, so its 1-Hz point is set by the
  shot batching (~10⁵ with this harness), not by the loop.
- Emitter: 1.9–3.4× faster than the v2 march at the old brackets (check run), so the curve is now bracketed from
  L = 6; from L ≈ 11 the host-side sector-table build (240–330 s per new N) is again the limit, and L ≥ 12 are lower
  bounds.
- k emitters: 1.1× faster than v2 at the old brackets; the memory edge is the sector size (~10⁸ amplitudes), as before.
- Mean loop occupancy at the brackets: emitter 9–13, k emitters 6–9.
- Kept: results/keep/hybrid_frontier_{emitter_v3_L*,hybrid_v3_k*_L*,fock_chain_L*,fock_chain_spaced_L*}_20261016_h100.csv,
  results/keep/hybrid_release_check_*_20261016_h100.csv.

## 6b. Memory count (pure counting, `bench/memory_frontier.py`, 80 GiB, complex128, one trajectory)

| L | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| single photons, dense register: largest N | > 10⁵ | 3179 | 227 | 78 | 45 | 31 | 24 | 18 |
| single photons, its density matrix (exact marginals): largest N | 381 | 74 | 21 | 12 | 9 | 7 | 6 | 5 |
| emitter, dense register: largest N | 2523 | 501 | 108 | 52 | 34 | 25 | 21 | 16 |
| sector register: largest photon count p held by the loops (single photons / emitter) | > 10⁵ / 73269 | > 10⁵ / 2523 | 596 / 197 | 121 / 71 | 57 / 41 | 37 / 29 | 27 / 23 | 19 / 17 |

The sector row is the one that matters for the sampler of record: it bounds the loop *occupancy*, not N, and the
mean occupancies met in the runs (random settings, largest per-tick mean over shots) are 10–19 photons — well inside it. The GPU out-of-memory stops of run
20261002_073236 therefore came from the cached sector tables and batched shots (and a leak between points, fixed),
not from one trajectory's state.

## 7. Changelog
- v3.2 (2026-10-16): loop rows re-measured on march v3 and the Fock chain (§6e, run 20261016_023739); the march's
  emitter crash with generator gates fixed (`.contiguous()` before `matrix_exp`).
- v3.1 (2026-10-12): the Fock chain is the default for one loop with Fock input (§6d, run 20261012_081240); `fast`
  flag removed.
- v3.0 (2026-10-12): sector march v3, the only path (§6c, run 20261012_075702); Fock-chain sampler for one loop with
  Fock input behind `fast=True` (§6d), pending GPU validation.
- v2.7 (2026-10-05): Fock limit test of SBS-L at the frontier (§5, run 20261005_170728).
- v2.6 (2026-10-03): statistic and large-L limit stated; k-emitter input (`--input hybrid`, `light_hybrid_kernel`).
- v2.5 (2026-10-02): memory count (§6b); SuperpositionLoopKernel takes any number of levels (lgamma).
- v2.4 (2026-10-02): photon flux: --spacing for the like-for-like comparison with the emitter.
- v2.3 (2026-10-02): rate fallback / budget on sampling time, untimed sector warm-up.
- v2.2 (2026-10-01): sector-table build time excluded from the measured rate (reported apart).
- v2.1 (2026-10-01): sectors are the sampler of record for single photons and the emitter; `--scan-l`
  (L\* at fixed N); run 20261001_101803 read out (dense control, SBS).
- v2 (2026-10-01): the paper row is one rail (§5): SBS through loops added; photon-number sector sampler (§6),
  validated, not yet the measurement method of record.
- v1.1 (2026-10-01): exact register (p = kT, finite trains into empty loops, ORCA convention) is the
  frontier convention; the cap is a sensitivity option only.
- v1 (2026-10-01): model, register with photon cap and leak accounting, CPU kernel on the
  `PeriodicNetworkKernel` machinery (additive `n_max` argument there), oracle and limit tests.
