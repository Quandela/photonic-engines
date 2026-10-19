# Distributed strong simulation (CPU, MPI) — EXPERIMENTAL

> **Status (2026-09-11): experimental.** Correct (validated to 1e-14 against `FockSLOSKernel`
> and permanents in both execution modes, 5 tests + a tutorial section) and measured on one
> 48-core node up to n = 13 (§3c), but: never run on more than one node; the pool transport
> relies on `mpirun` behaviour that differs between MPI implementations (mpich and Open MPI 5
> both work, others untested); complex128 only; the API (`DistributedVector` accessors,
> `plan()`) may still change; the single-node ceiling is memory bandwidth (§3c) and the known
> fixes (complex64 storage, all-layer boundary balance, overlapped halo) are not done.
> Use it when the distribution does not fit one device; do not build on the internals.
> Limits of the C kernel: m ≤ 256 modes, n ≤ 255 photons (uint8 occupations, 256-entry
> per-mode scratch); the Python face rejects larger requests. macOS: the library is built
> without OpenMP by default (an OpenMP .so next to torch's libomp aborts the process);
> `PE_OPENMP_FLAG=-fopenmp` forces it. The build cache key covers source, compiler and flags.
> `DistributedVector` / `PooledVector` always hold amplitudes — `kernel.probs(U)` returns the
> same handle as `amplitudes(U)`; probabilities come from `.probs()` (gathered),
> `.probabilities_of(patterns)`, `.marginal(j)`.

`photonic_engines.slos.distributed` — SLOS_full with the amplitude vector split across MPI
ranks. It is the memory-scaling layer of the strong-simulation surface: `FockSLOSKernel` stops
where one device's memory stops (n = 11, m = 22: 1.3·10⁸ states, 16 GB on an H100); this
engine puts M_n/P amplitudes on each of P ranks and never gathers the vector unless asked.

## 1. Where it comes from

The numerics are the slos_mpi project (`slos_mpi_dist.c`, `slos_scatter.c`): the vector is
cut into equal contiguous chunks of the colex index space (load-balanced even at m = 2n),
each rank unranks only its own rows and computes every predecessor's index by the closed-form
colex rank (the O(1) incremental variant: the m parents of one row differ by one prefix-sum
position), fetches the off-rank predecessors of a batch of rows with a two-phase
`MPI_Alltoallv` (request the global indices, receive the values), and gathers. Two vector
slices live at a time; no index tables are stored. The code was turned into a library,
`slos/csrc/slos_dist.c`: no I/O, no `MPI_Init`, the communicator comes from mpi4py
(`Comm.py2f()`), entry points `create / run / get / rank / unrank / chunk / destroy`. It is
compiled on first use with `mpicc` into `$PE_KERNEL_CACHE` and loaded through cffi.

### 1a. Layout v2: pattern ownership, interval halo, threads (2026-09-11)

The equal-chunk layout above made every rank's parents land anywhere in the previous layer
(the chunk boundaries of layer k and k+1 are at different fractions of different patterns):
on the 48-core node ~80 % of the parent reads were remote, the request/response halo moved
~280 GB through shared memory, the per-batch hash of remote indices cost as much as the
arithmetic, and the speedup saturated at ~7.7× (§3a). The rewrite in `slos_dist.c` changes
three things below the API:

- **Ownership by pattern, not by layer.** Rank r owns a range of the *key order* — lexicographic
  on (t[m-1], …, t[1]), then t[0] — which restricted to any layer is the colex order, so rank
  r's chunk of layer k is still a contiguous colex range `[lo[k][r], lo[k][r+1])`, but the
  same pattern boundary applies to every layer. The P−1 boundary patterns are picked on layer
  n so that the *work* (1 + non-empty modes per row, sampled at 2¹⁶ colex quantiles) is
  balanced, not the row count. Rank 0 never receives anything (its parents all have smaller
  keys).
- **Interval halo.** For a fixed mode i the map t ↦ t − e_i is monotone in that order, so the
  mode-i parents of a contiguous block of rows form one contiguous interval of the previous
  layer with no gaps: the halo of a batch is the union of ≤ m intervals, split at the owner
  boundaries and fetched as contiguous copies — no index request, no hash, 16 B per entry
  instead of 24, and every fetched entry is used. Pass 1 stores each parent's absolute index
  (from `rank_k(t − e_i) = rank_{k+1}(t) − shift_i`, the shifts updated only below the carry
  position of the odometer: no table lookup on most rows), pass 2 reads it through a direct
  pointer when a chunk's parents lie in one segment (the usual case), a per-mode cursor
  otherwise.
- **OpenMP threads.** Both passes are `omp for` loops over 2048-row chunks; the intended use
  is one rank per node (or per socket) and the cores as threads, so that the halo only exists
  between nodes. Compiled with `-fopenmp` when the compiler has it (Apple clang: single
  thread per rank). The pool sets `OMP_NUM_THREADS = cores / workers`; SPMD runs set it by
  hand. `kernel.threads` and `timings["halo_entries"]` report what happened.

Sandbox (2 cores), n = 10, m = 20: 1 rank 7.1 s → 3.9 s (build 5.3 → 1.5 s), 2 ranks 5.4 s →
2.2 s; the halo of rank 1 is 9.8·10⁶ entries (a third of the layers) instead of most of them.
What this buys on the 48-core node is measured in §3c: n = 12 in 9.4–11.3 s instead of
36.7 s, and the same time for every split of the 48 cores into ranks × threads.

## 2. The engine face

`DistributedSLOSKernel(n, m, occupation)` implements the strong-simulation contract
(`amplitudes(U)`, `probs(U)`, `output_size()`), returning a `DistributedVector` whose
accessors are the operations one actually needs on a vector that does not fit anywhere:

| call | what happens |
|---|---|
| `probability(t)`, `amplitudes_of(patterns)` | colex rank → owner rank → one collective point lookup |
| `marginal(j)`, `mean_photon_numbers()` | per-rank reduction + `Allreduce` |
| `top(k)` | per-rank partial sort + `Allgather` of the candidates |
| `sample(shots)` | shots split across ranks by local mass, drawn locally, patterns gathered — exact sampling from the strong simulation without the vector |
| `gather()` | the whole vector on rank 0 in `FockLayers` order, only below `max_bytes` (raises otherwise) |

Two execution modes, same class:

- **SPMD** — `mpirun -np P python script.py` (or `srun`): every rank builds the kernel and
  calls the same methods (all collective). Detected from `comm=` / `COMM_WORLD`.
- **Pool** — `DistributedSLOSKernel(..., workers=P)` in a plain Python process or notebook:
  P workers are started once and kept for the life of the kernel, driven by a small command
  protocol (build, run, get, marginal, mean, top, sample, gather, stop). The caller sees numpy
  results; no ranks. Two transports: `mpirun` (default) launches
  `mpirun -np P python -m photonic_engines.slos._mpi_worker --connect 127.0.0.1:port` as a
  subprocess; rank 0 connects back to the pool's socket and relays pickled commands
  (broadcast to the other ranks over MPI) — works with any MPI on any OS, and keeps the
  launcher's stdout out of the protocol; `spawn` (`transport="spawn"` or
  `PE_MPI_TRANSPORT=spawn`) uses `MPI_Comm_spawn` from the calling process, which mpich
  supports from a plain process but Open MPI 5 on macOS does not (TCP wire-up errors from a
  singleton). Pool mode is chosen whenever `workers=` is given and no communicator is passed.

**Planning.** `plan(n, m, ranks, mem_per_rank_gb=None, nodes=1, cores=None)` returns a
`Plan` (M_n, vector and peak bytes, bytes per rank, estimated seconds) from the model fitted
on the §3c measurements: a node is bound by the random-gather bandwidth of the parent reads,
`16 · M_total · n` bytes at ~25 GB/s (240 GB → 9.4 s at n = 12), with a compute floor of
160 ns per output row per core; ranks and threads only decide how the cores are grouped.
±30 % on the 18 v2 points. Both kernels call
`check_plan()` in their constructor: if the peak does not fit 90 % of `MemAvailable`
(× `nodes`), they raise `MemoryError` with the sizes and the number of such nodes needed,
before any worker is launched. `PE_MPI_NOCHECK=1` skips it (e.g. on a cluster where the
launching node's memory is not the workers').

Perceval: `adapters.perceval.DistributedSLOSBackend(workers=P)` is a strong backend usable
as `pcvl.Processor(backend_instance, circuit)` — `probability` / `prob_amplitude` are point
lookups; `prob_distribution` / `evolve` / `all_prob` enumerate Perceval's output states and
fetch them all by point lookup, so they materialise the full distribution on the caller
(M_n entries — only for sizes where that fits; nothing bounds them). `DistributedSLOSSampler(strong=b,
seed=…)` is the `ASamplingBackend` face for `Processor` / `Sampler`: `samples(count)` draws on
the workers from the distributed vector, which is never gathered.

Requirements: `mpi4py`, `cffi`, an MPI with `mpicc` and `mpirun` (`pip install mpich` is
enough on a laptop, `brew install open-mpi` or `mpich` on a Mac; on a cluster use the module),
and the package importable by the workers (`pip install -e .`, or the pool passes the caller's
`sys.path` as `PYTHONPATH`). With the pip mpich, `libmpi` is not on the loader path — `_mpi()`
sets `MPI4PY_LIBMPI` from `mpicc -show` automatically. Several ranks compiling the library at
once is handled (per-process temp file). Threads per rank: `OMP_NUM_THREADS` (the pool
defaults it to cores / workers; SPMD runs set it by hand, `mpirun -np 1` with all cores as
threads is the single-node configuration).

## 3. What to expect

Sandbox (2 cores), n = 9, m = 18, M_n = 3.1·10⁶, complex128:

| P | total | build | comm | compute | chunk |
|---|---|---|---|---|---|
| 1 | 1.77 s | 1.32 | 0.00 | 0.42 | 3.1 M |
| 2 | 1.07 s | 0.70 | 0.26 | 0.25 | 1.6 M |

`FockSLOSKernel` on the same CPU (2 threads, index tables precomputed): 0.31 s. The single-rank
C code is ~5× slower than the torch kernel because it recomputes the predecessor indices
("build", 75 % of the time) instead of storing O(M_n·m) tables — that is the memory trade the
engine exists for. Per rank: ≈ 1.4·16·M_n/P bytes (two live layers) + MPI baseline, i.e.
n = 12 at m = 24 on one 20 GB node, and one more photon per ≈ 8× ranks (slos_mpi README §6.5;
complex64 storage halves it, not exposed yet).

### 3a. 48-core node (Ubuntu, mpich 4.3, complex128) — run 2026-09-11

`results/keep/cpu48_scaling_small.csv`, `img/cpu_scaling.png` (`bench/plot_scaling_distributed.py`). Total seconds (max over ranks):

| n, m | M_n | P=1 | 2 | 4 | 8 | 16 | 24 | 48 | best speedup |
|---|---|---|---|---|---|---|---|---|---|
| 10, 20 | 2.0·10⁷ | 5.39 | 2.99 | 1.82 | 1.05 | 0.70 | 0.75 | 0.74 | 7.7× (16) |
| 11, 22 | 1.3·10⁸ | 37.9 | 21.4 | 12.3 | 6.89 | 5.00 | 4.92 | 5.61 | 7.7× (24) |
| 12, 24 | 8.3·10⁸ | 277 | 147 | 88.6 | 50.4 | 41.1 | 36.7 | 39.9 | 7.6× (24) |

The single-rank rate is ~290 ns per output row (n = 12: 242 s of build for 8.3·10⁸ rows × 12
parents). Speedup is near-ideal to 8 ranks and saturates at ~7.7× for every size; the
breakdown at n = 12, P = 48 says why: `comm` 27 s and `build` 28 s (max) against 11 s for the
fastest rank. Two causes, both layout, not code:

1. **Halo volume.** With equal colex chunks nearly every parent is remote at large P; each
   parent costs a request index + a value (24 B) through shared memory: ≈ 8.3·10⁸ rows × 12 ×
   24 B × 1.15 (all layers) ≈ 280 GB per run, i.e. the ~27 s at ~10 GB/s. Only a layout in
   which most parents are local (nested-by-last-mode blocks: one remote mode, nearest-neighbour
   halo) removes it.
2. **Imbalance.** Equal chunks give unequal work: the per-row cost depends on the number of
   occupied modes (parents) and on the remote fraction, both of which vary systematically
   along the colex order (rank 47's build 12 s vs rank 38's 27 s). Speed-weighted chunks
   (measure, then resize) or work-model chunks fix this independently of (1).

**n = 13, m = 26** (5.4·10⁹ states, 87 GB of amplitudes, ≈120 GB peak across the ranks —
beyond any single GPU or single-rank run): 289 s on 16 ranks, **264 s on 24**, 293 s on 48
(`results/keep/cpu48_scaling_n13.csv`; norm 1 to 1e-12). Same shape as n = 12: build 88–200 s
across ranks (2.3× imbalance), comm = waiting + ~1.8 TB of halo through shared memory.

So the engine delivers its purpose — n = 12 in 37 s and n = 13 in 4.4 min on one node — and the
next step for time is the block layout, not the kernel.

### 3c. v2 layout on the same node — run 2026-09-11 (evening)

`results/keep/cpu48_scaling_v2.csv`, `img/cpu_scaling_v2.png` (`bench/plot_scaling_v1v2.py`); `--threads auto`, i.e. the 48
cores split as P ranks × 48/P OpenMP threads.

| n : m | P = 1 (48 thr) | 2 | 4 | 8 | 16 | 32 (1 thr) | v1 best (P = 24) | v1 P = 1 |
|---|---|---|---|---|---|---|---|---|
| 10 : 20 | 0.38 s | 0.41 | 0.48 | 0.40 | 0.35 | 0.28 | 0.71 | 5.5 |
| 11 : 22 | 1.92 | 2.06 | 2.02 | 1.70 | 1.57 | 1.57 | 4.7 | 39 |
| 12 : 24 | 11.3 | 11.3 | 12.7 | 10.2 | 9.4 | 9.6 | 36.7 | 277 |
| halo, 12 : 24 | 0 | 6 GB | 13 | 16 | 22 | 27 | ~280 GB | 0 |

Three readings. (i) **The halo is gone as a cost**: 27 GB at P = 32 instead of ~280 GB, and
`comm` at P = 32 (6 s max, 3 s min) is mostly ranks waiting for each other — the boundary
patterns are balanced on layer n only, the intermediate layers less so. (ii) **The node is
now bound by memory bandwidth, not by the layout**: the total is 9.4–12.7 s for *every* split
of the 48 cores, from one rank with 48 threads to 32 single-thread ranks. Every output row
reads its ≤ n parents at effectively random addresses in a 13 GB vector: 16 B × 1.25·10⁹
rows × 12 parents = 240 GB per run, i.e. 25 GB/s of random gather — what a two-socket node
sustains. A single core does a row in ~160 ns (v1: 250 ns), so 48 cores would need 4 s if
the reads were free; they are not. (iii) **Ranks = nodes is the right picture**: since threads
and ranks cost the same on one node, the natural deployment is one rank per node (or per
NUMA socket) with the cores as threads, and MPI only for the inter-node halo (≤ 1 GB per
rank per run at this size). The per-rank `build` at P = 32 is bimodal (2.0 s / 3.5–3.9 s,
alternating), the signature of hyper-threads sharing a physical core — the 48 "cores" are
likely 24 physical, which is also why 48 threads bring ~24× (11.3 s vs 277 s single-thread v1,
~29× against a v2 single thread).

**n = 13, m = 26** (5.4·10⁹ states, 113 GB peak; `results/keep/cpu48_scaling_v2_n13.csv`):
**73 s on 1 rank × 48 threads** (build 42, compute 24), 79 s on 4 × 12, **61 s on 16 × 3**
(v1: 264 s best, model: 68 s). Same picture one size up: 1.7 TB of parent reads in ~60–70 s
is 25 GB/s again; at P = 16 the 145 GB halo costs a few seconds of transfer and ~20 s of
waiting (`comm` 22–26 s on every rank) since the boundaries are balanced on layer n only,
while the smaller per-rank vectors read a little faster (NUMA locality) — hence the 16 %
gain over one rank. 4.3× over v1 at this size, 3.9× at n = 12.

What would lift these: complex64 storage (halves the bytes), sorting each batch's parent
reads by address (turns random gather into streaming), boundaries balanced over all layers
(the P = 16 waiting), or several nodes (bandwidth adds up: `plan(14, 28, 4, nodes=4)`).

## 3b. Running on a multi-core server

On the server (MPI, mpi4py and the C library installed; a 2-rank `tests/slos/check_distributed.py` run
first), `bench/scaling_distributed.sh` runs the scaling sweep (sizes 10:20, 11:22, 12:24 × ranks 1…64,
capped at the core count and the node memory) and writes `results/<stamp>/scaling_distributed.csv`
(total / build / comm / compute, per-rank build spread). Sizes and ranks are options, e.g.
`bench/scaling_distributed.sh --sizes '12:24 13:26' --ranks '16 32 64'`.

## 4. Roadmap (all below the API)

1. Bandwidth: complex64 storage (`-DSP`, halves the 240 GB of parent reads at n = 12),
   address-sorted gathers per batch (streaming instead of random reads); optional index
   tables when memory allows.
2. ~~Nested-by-last-mode layout with a nearest-neighbour halo~~ — done as the pattern-ownership
   layout of §1a, `plan()` refitted on §3c; remaining: balance the boundaries on the
   intermediate layers too (the P = 32 waiting), and multi-node runs.
3. The same chunked layout for the emitter kernel (χ-fold vector) and the time-loop register
   kernel (gates not touching the partition mode are communication-free), and a
   `torch.distributed`/NCCL variant for multi-GPU nodes.
4. Sparse last layer with the √δ mass bound (`SLOS_THR` in the original) as `v.mass(delta)`.
