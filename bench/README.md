# Benchmarks and measurement scripts

Every script that produced a measurement of the papers, or a check behind one. The ones that back a published number are run by `reproduce/` (see `reproduce/MANIFEST.md`); the others are diagnostics and profiling tools kept for reference.

| script | what it does |
|---|---|
| `adaptive_frontier.py` | Throughput of the adaptive sampler (docs/ADAPTIVE.md §4-5) against the plain Fock sampler, same host, same day. |
| `adaptive_overhead.py` | Rejection overhead of the adaptive sampler's final stage (docs/ADAPTIVE.md §5), from the exact outcome law. |
| `adaptive_toll.py` | The toll of adaptivity on the sampling frontier: n*(K) at fixed m/n from the cost model, and its figure. |
| `algorithm_envelope.py` | The algorithm envelope on one instance family (docs/ALGORITHMS.md §5, figure F2a, table T1). |
| `algorithm_terms.py` | Addends per exact sample for every algorithm of the T1 set, along the SAME chains (docs/ALGORITHMS.md §4). |
| `anticoncentration.py` | Anticoncentration of emitter-state boson-sampling distributions vs Fock inputs, over Haar U. |
| `bench_cofactor_variants.py` | Cofactor-kernel variant sweep: where the 2020 CUDA kernel loses time, measured. |
| `bench_cuda_emitter.py` | Tune / compare the CUDA thread-per-lane emitter kernel (clifford/cuda_emitter.py, docs/EMITTER_SAMPLER.md §8) against the Triton seq kernel, one late step per K. |
| `bench_cuda_rect.py` | Tune the CUDA rectangular cofactor kernel (clifford/cuda_rect.py, the SBS rso chain's step) against the torch cofactors of sbs_rso, at the chain's frontier step sizes. |
| `bench_distributed.py` | Distributed CPU engine timing. SPMD: mpirun -np P python3 bench/bench_distributed.py [--n 10 --m 20] Prints total / build / comm / compute seconds and the per-rank chunk; --ref also times FockSLOSKernel (CPU, complex128) |
| `bench_emitter_sampler.py` | Emitter samplers (batched torch, Triton kernel) vs the Fock Clifford sampler (Triton/CUDA kernels): ms per sample and the mean number of pure terms per step. |
| `bench_emitter_slos.py` | EmitterSLOSKernel vs the Fock FockSLOSKernel: time and peak memory per (n, m) on one device. |
| `bench_emitter_tune.py` | Sweep the launch configuration of the Triton emitter cofactor kernels per K_PAD tier. |
| `bench_gaussian_slos.py` | Reach of the Gaussian strong simulation (`slos.GaussianSLOSKernel`, docs/GAUSSIAN.md §7): n squeezers at --mean photons each on modes 0..n-1, m = round(ratio n), every lossless layer up to the cutoff that leaves --eps of |
| `bench_gbs_impl.py` | v1 vs v2 of the Gaussian sampler (docs/GAUSSIAN.md §3.4), same seeds, same device. |
| `bench_loss.py` | Uniform loss on the GPU samplers (docs/LOSS.md): correctness of the input-side split against the thinned strong simulation at small n, then the time per sample of the lossy vs lossless samplers at large n (Fock `fock_sam |
| `bench_product_amp.py` | Cost of ONE superposition-boson-sampling amplitude (`slos.glynn.product_amplitude`). |
| `bench_product_sampler.py` | Cost of the exact chain-rule sampler for superposition boson sampling (`clifford.product_sampler`, docs/SUPERPOSITION.md §5). |
| `bench_product_slos.py` | Cost of the graded SLOS recurrence (`ProductSLOSKernel`) against the Fock one at the same size. |
| `bench_sieve_kernel.py` | Cost split of the Triton matrix kernel (gaussian/sieve_triton.py): how much of a sieve's matrix part is the power products (tl.dot), the trace reductions and the path terms, to decide whether an O(q^3) characteristic-pol |
| `bench_step_profile.py` | Per-step profile of one sample: where the time of a sample actually goes. |
| `bench_timeloop.py` | Native time-loop engine vs the generic path on the unrolled unitary. |
| `bench_tn_compare.py` | CUDA-Q's photonics path against this engine: one machine, one circuit, no assumptions. |
| `bracket_uncertainty.py` | Statistical uncertainty of a find_threshold bracket, from the segment rates its run log prints. |
| `cofactor_precision.py` | Rounding error of the fp64 cofactor sweeps at frontier sizes (docs/ALGORITHMS.md §6, Supplement C). |
| `coherence_scan.py` | The lossy advantage of a COHERENT source: n*(coherence) at fixed brightness. |
| `compile_check_emitter.py` | Offline (no GPU needed) compile check of the generated "seq" emitter kernels for sm_90: compile time, register count and spills (ptxas -v), shuffle count. |
| `cost_table.py` | The lossless cost table t(k, m): seconds per exact sample of k photons in m modes, MEASURED at the mode counts of the devices placed in the hardware-competitiveness analysis (docs/HARDWARE_TARGET.md §"Where current devic |
| `cpu_sampler_worker.py` | One external CPU GBS sampler, timed one shot per call, in its own venv. |
| `cuda_rect_mu_extend.py` | Extension of the mu lane: the templated mu lane of clifford/cuda_rect.py at D = s0 = 7 .. 16 against the lambda-form that "auto" dispatched there before (2026-10-05), on the same instances. |
| `cuda_rect_mu_validate.py` | Continuity check (a) of the SBS validation (bench/sbs_rso_validate.py), kernel side: validate the templated mu lane of clifford/cuda_rect.py on the device and measure the per-s0 step costs c(s) that predict check (a) (20 |
| `cudaq_qudit_levels_repro.py` | Minimal, torch-free repro: does CUDA-Q silently reinterpret a d>2 qudit as a qubit? |
| `emitter_fock_throughput.py` | Emitter-kernel throughput test: the emitter kernel `cuda_emitter.emit_multi` against the Fock kernel `cuda_cofactors.cof_multi` per needed operation, at the chain's last steps, n = 30 ... 36. |
| `emitter_gram.py` | How much of the emitter sampler's term count R_k is redundant — measured exactly. |
| `emitter_precision.py` | Rounding error of the fp64 emitter cofactor kernels at the frontier sizes (Supplement C; cross-engine review item #4, docs/EMITTER_SAMPLER.md §9). |
| `energy_series.py` | Energy per exact Fock sample on the GPU, with MEASURED board power (replaces the assumed 280 W x old-kernel rates of bench/plot_energy.py). |
| `esbs_design_plane.py` | Design plane of the emitter-state sequence generator (candidate A of the application study): structure of the interferometer x number of emitters k, on one grid, three questions per point. |
| `esbs_fusion.py` | Register growth with *local* optics, in the spin-optical (SPOQC) regime of de Gliniasty et al., Quantum 8, 1423 (2024): when does the emitter x time machine become a hard register — a computer — without a global interfer |
| `esbs_train_signal.py` | Is a *sample-level* loss sensitive to the emitter's parameters where the register is hard? |
| `estimate_time.py` | Cost of the Gurvits-type estimator (photonic_engines.slos.estimate) against n and chi, where the strong and sampling engines stop: ms per shot, second moment, and the shots needed for a target additive error on p(t) (doc |
| `f7_feedforward_cost.py` | F7 — the feed-forward cost table of the emitter-train sampler. |
| `find_threshold.py` | Locate the simulability threshold n*: the photon number at which a sampler's sustained throughput drops to --target samples/s (default 1 Hz) for a mode ratio m = c·n, on this GPU. |
| `frontier_indist.py` | The simulability frontier under partial distinguishability n*(eta, x): for each engine (fock, emitter), transmission eta and pairwise overlap x (x-model, V_HOM = x^2), run find_threshold.py --loss eta --indist x and coll |
| `frontier_loss.py` | The lossy simulability frontier n*(eta): for each engine (fock, emitter, sbs) and transmission eta, run find_threshold.py --loss eta and collect the photon number at which the *lossy* sampler drops to --target samples/s  |
| `gbs_borealis_exact.py` | Exact samples of a Borealis run's own model on one GPU, and their check against the timing model and the device (docs/GAUSSIAN.md §2.4). |
| `gbs_charpoly_proto.py` | CPU prototype of the O(q^3)-per-subset sieve (docs/GAUSSIAN.md §8): the trace part of every term from det(I - lam C_S)^(-1/2) (batched Householder Hessenberg + the Hessenberg characteristic-polynomial recurrence) instead |
| `gbs_compare_walrus.py` | Precision of our pair sieve against The Walrus (Apache-2.0) on the same loop hafnians (docs/GAUSSIAN.md §8). CPU. |
| `gbs_cost_check.py` | Consistency check of the GBS frontier statistic (gaussian/cost.py): the mean seconds per sample at the target squeezing, estimated from independent shot sets drawn at different tilts s. Importance sampling with exact wei |
| `gbs_dd_profile.py` | Where the double-double sieve term spends its time (docs/GAUSSIAN.md §3.4 v2.11b; job 28). |
| `gbs_dd_rate.py` | The cost of double-double (dd, ~106-bit) arithmetic in the sieve's matrix products, measured against fp64 (part of the device comparison, docs/GAUSSIAN.md §2.4). |
| `gbs_device_replay.py` | Price a device's own samples through the exact sampler's timing model (docs/GAUSSIAN.md §2.2). |
| `gbs_device_table.py` | The device comparison: the Borealis table and figure from the replays of `bench/gbs_device_replay.py` (docs/GAUSSIAN.md §2.4). |
| `gbs_kappa_check.py` | The condition number kappa of the fp64 pair sieve on frontier patterns at N = 50-62 (B's precision request, P 10-11). |
| `gbs_loop.py` | GBS through time-bin loops at small delay (the Gaussian row of the loop frontier): the source number N* at which a cutoff-register sampler sustains --target samples/s, as a curve N*(L), on the axis and with the settings  |
| `gbs_pairing_bench.py` | Cross-mode collision grouping against the v2.8 pairing on device-like patterns: time, condition number, terms (docs/GAUSSIAN.md §3.4 v2.9-pairing; `hafnian._pairing_cross`, `_pick_grouping`, `_choose_pairing`). |
| `gbs_precision.py` | Rounding error of the pair sieve against an extended-precision reference, and how the choice of vertex pairing sets it (docs/GAUSSIAN.md §3.4, "Precision, open"). CPU only. |
| `gbs_timing_model.py` | The GBS frontier from a calibrated timing model of the exact sampler (docs/GAUSSIAN.md §2.2). |
| `gbskernels_bench.py` | GBSKernels' GPU GBS sampler against ours on identical instances. |
| `hybrid_frontier.py` | Loop-interferometer frontier for the three single-rail inputs of the paper's part (b): the photon (or source) number N* at which the native register sampler sustains --target samples/s, as a curve N*(L) — the same method |
| `hybrid_sector_speed.py` | Per-sample cost of the loop sampler on two paths, "default" and "fast", at fixed (N, L) and batch size; plus a two-sample law check of the two paths and a host-sync count per tick (2026-10-12). The CSV column `fast` is 0 |
| `internal_modes_bench.py` | Internal photon modes (docs/SEQUENCE.md §14): cost of one bin's channel on the populated internal span (route F of `slos.internal`) and of its outcome law by route F and by the pairing route P', against the expanded cons |
| `internal_modes_ccspin_bench.py` | Internal photon modes, the weak route (docs/SEQUENCE.md §14, steps 6b-6d): time of one `sample_apply` tick (CC-spin, operator-valued Clifford-Clifford) for k emitters with qubit spins, L = 2 labels (dual rail, m = 2k), K |
| `internal_modes_regression.py` | Internal photon modes, regression bench over the tick shapes of a spin-photon processor (docs/SEQUENCE.md §14; design note §6a (3)): the route the dispatcher picks must not be slower than today's expanded construction on |
| `loss_cliff.py` | Diagnostic for the lossy Fock sampler cliffs seen in run 20260917_182540: 1.08 -> 0.007 samples/s between n = 50 and 51 at eta = 0.7 (m = 100 -> 102) and 1.01 -> 0.24 between n = 64 and 65 at eta = 0.5 (m = 128 -> 130),  |
| `loss_unravel_check.py` | GPU check of the label unravelling of the lossy emitter sampler (docs/LOSS.md §3b): for a few n and eta, sample with `unravel=True` (labels drawn, one term per gap) and `unravel=False` (traced gap terms), compare the det |
| `map_threshold.py` | Map the simulability frontier n*(m/n) for the Fock, emitter (caterpillar) and superposition samplers: run find_threshold.py per (engine, ratio), collect n*, rho_eff and the bracketing points, and write a table + CSV (+ a |
| `memory_frontier.py` | Memory frontier of strong simulation: largest n whose full output distribution fits in a byte budget, per mode ratio c = m/n. Pure counting - no benchmark. |
| `merge_frontier.py` | Merge frontier CSVs from several map_threshold.py runs into one, for plot_frontier.py. |
| `methods_cpu_checks.py` | CPU checks of the chain-rule samplers, printed with their values. Each mirrors a test that asserts it (named below); this script prints the numbers the tests only bound. CPU, about a minute. Needs tests/ on disk: it impo |
| `mu_lane_error.py` | True fp64 error of the templated mu lane (clifford/cuda_rect.py, CPU emulation) against an EXACT reference, on the cases of tests/clifford/test_cuda_rect.py (test_templated_mu_lane_every_degree and, with --high, test_tem |
| `multigpu_scaling.py` | Multi-GPU scaling of the production Fock sampler, MEASURED. |
| `nlbs_cost_law.py` | The cost law of the non-linear sampler (docs/NONLINEAR.md §5b), recomputed from the frontier scans. |
| `nlbs_frontier.py` | The simulability frontier of non-linear boson sampling n*(phi, L, m/n, eta, x): for each setting, run find_threshold.py --engine nlbs and collect the photon number at which the exact rejection sampler drops to --target s |
| `orca_loop_bench.py` | ORCA's public loop simulator against our loop rows. |
| `orca_loop_worker.py` | ORCA's loop-progressive-simulator (Novak et al., arXiv:2411.16873; Apache-2.0) sampling one loop instance, in its own venv. Run by bench/orca_loop_bench.py with $OPY; the clone is $ORCA_SRC. |
| `orca_tn_bench.py` | ORCA's cuTensorNet photonic simulator (branch photonics_tn_simulator_lm, 3d66214, built job-local by scripts/orca_tn_build.sh) on our loop and mesh instances: the law against brute force, then the timing (P, 10-11/12). |
| `perceval_threshold.py` | n* of Perceval's Clifford & Clifford 2017 back-end (exqalibur) on the T1 instance family and protocol. |
| `plot_algorithm_envelope.py` | Figure F2a: the algorithm envelope on one instance family (docs/ALGORITHMS.md §5). |
| `plot_energy.py` | Energy per exact sample: measured on one H100 against the classical flop lower bound and the quantum-device figure of Soret et al. (arXiv:2601.08068). |
| `plot_esbs_design_plane.py` | Plot bench/esbs_design_plane.py outputs: one panel per quantity, x = k, one line per unitary family, values at the last tick averaged over settings (register entropy fraction, bin flatness, inter-bin training signal agai |
| `plot_esbs_fusion.py` | Plot bench/esbs_fusion.py results: register entropy vs tick per k (one panel per CSV = one (optics, program)), and the final entropy vs k across panels (volume law: ∝ k; area law: flat). |
| `plot_frontier.py` | Plot the simulability frontier n*(m/n) of both samplers from a map_threshold.py CSV. |
| `plot_frontier_loss.py` | The lossy simulability frontier n*(eta) of every engine with a loss curve, from the kept CSVs (no measurement). |
| `plot_hybrid_frontier.py` | Plot the loop-interferometer frontier N*(L) of bench/hybrid_frontier.py: one curve per CSV (input, loop loss). |
| `plot_postselect.py` | Where a lossy photonic device stands against the classical sampler (docs/HARDWARE_TARGET.md). v2 (2026-10-01): devices described by their parameters only, the measured cost table t(k, m) of bench/cost_table.py where it e |
| `plot_rate_maps.py` | The frontier at other rates: n*(m/n) at 1e-3, 1 and 1e3 samples/s, one H100. |
| `plot_scaling_distributed.py` | Plot a scaling_distributed.sh CSV: total time and speedup vs ranks per size, and the build / comm / compute breakdown. python3 bench/plot_scaling_distributed.py --csv ... --out img/cpu_scaling |
| `plot_scaling_v1v2.py` | v1 (equal chunks, 1 thread per rank) vs v2 (pattern ownership, interval halo, cores/P threads per rank) on the 48-core node: total time vs ranks per size, and the v2 breakdown. python3 bench/plot_scaling_v1v2.py [--v1 re |
| `plot_sequence_frontier.py` | Plot bench/sequence_frontier.py results (docs/SEQUENCE.md §5). |
| `plot_sequence_ticks.py` | Plot bench/sequence_ticks.py results (docs/SEQUENCE.md §9). |
| `plot_timeloop_frontier.py` | Plot the time-loop frontier from a timeloop_frontier.py CSV. |
| `profile_emitter_step.py` | Where does a sampler step go? Phase timings (device-synchronised) and the top CUDA kernels of one `batched_gap_terms` + `emitter_step_weights` call at a given (n, k, shots). |
| `profile_gbs.py` | Where the time of a GBS shot goes (docs/GAUSSIAN.md §3.4, the v3 plan): the benchmark for sampling optimisation, deterministic (fixed matrices, fixed seeds, capped shots) so that versions compare on the same work. |
| `rate_series.py` | Sustained throughput at a list of photon numbers — the series quoted in the paper's "Results" paragraph and on the deck, measured with exactly the protocol of `bench/find_threshold.py` so the points sit on the same curve |
| `rect_precision.py` | Rounding error of the fp64 rectangular cofactor kernel (clifford/cuda_rect.py, the SBS rso chain's step) at the chain's step sizes, on the CPU: the kernel code compiled in double against the same code in x87 long double  |
| `rect_precision_gpu.py` | fp64 rounding of the SBS rso step kernels at the frontier sizes, on the GPU: a LOWER BOUND from disagreement between evaluations of the same cofactors that round differently (Supplement C, method 2) -- the templated mu l |
| `run_cost.py` | The lossy frontiers under the run-of-S statistic, priced from the lossless map (docs/GAUSSIAN.md §2.1 for the statistic). No GPU run: a calibrated-model number, labelled so. |
| `sbs_chain_rso.py` | Random-source-order SBS chain (clifford/sbs_rso.py) against Perceval's SLOS, with the incoherent twin as control. |
| `sbs_mu_residual.py` | Residual of the superposed (alpha = pi/4) sampler against the best known cofactor kernel, in sources (the frontier paper's parity audit: <= 0.07 source, headroom +0.14-0.18 at n = 46, +0.35-0.43 at n = 60). CPU model, me |
| `sbs_rso_kernels.py` | Cofactor backends of the rso SBS chain on the GPU: plain Glynn vs the C&C 2020 collapse (torch / Triton / CUDA), pick. |
| `sbs_rso_strata.py` | Whole post-selected SBS shots per silent-source stratum at chosen n (2026-10-06): the per-shot cost the rso chain pays on the current cuda_rect dispatch -- run it twice (e.g. PE_MU_R22=0 / 1) to compare two dispatches on |
| `sbs_rso_validate.py` | Validation of an SBS rso build: three short checks before any frontier block. |
| `sbs_run_cost.py` | The sbs frontier under the run-of-S statistic of docs/GAUSSIAN.md §2.1, for comparability with GBS. |
| `sbs_stratified_cost.py` | Seconds per sample of the rso SBS sampler at a lossy point, by two estimators side by side (P, 10-06, after the eta = 0.7 bracket of runs 20261006_012054 / 20261006_132352 rested on a tail-dominated shot mean). |
| `sbs_terms.py` | How many pure terms an exact chain-rule sampler for superposition boson sampling would carry. |
| `scaling_distributed.sh` | Strong-scaling sweep of the distributed CPU engine on one node (run from the repository root): bench/scaling_distributed.sh [--sizes "10:20 11:22 12:24"] [--ranks "1 2 4 8 16 32 64"] [--threads auto/1/N] [--out results.c |
| `sequence_frontier.py` | GPU frontier of the sequence modality (docs/SEQUENCE.md) in k, the number of emitters interfering per bin: dual-rail emitters into a universal interferometer of m = 2k modes per bin (the unrolled problem has k·n photons  |
| `sequence_precision.py` | Rounding error of the label-amplitude passes of the sequence kernel at k = 16-20 (Supplement C; cross-engine review item #4, docs/SEQUENCE.md §12c). |
| `sequence_ticks.py` | Complexity of the spin register of a static k-emitter machine, tick by tick (docs/SEQUENCE.md §9): k dual-rail emitters into one m = 2k-mode interferometer with no delay line, sampled by pure trajectories with the label- |
| `soret_energy_crossing.py` | Energetic crossing of Soret et al. (arXiv:2601.08068v1) recomputed on their own metric M. |
| `stage_bench.py` | Timing of the stage kernel (`slos.stage.StageKernel`, docs/DESIGN.md §2.4): Kraus operators of an emitter chain next to a batch of Fock inputs, per input and output pattern, on one device. |
| `stage_carry_bench.py` | Timing of partial detection in the stage kernel (`slos.stage.StageKernel.apply`, route B, docs/DESIGN.md §2.4): the blockwise update of a carried-photon density matrix by an emitter chain, on one device. |
| `table_lookup_rate.py` | Sample rate from a stored output distribution (strong simulation below the memory frontier): exact samples by lookup. |
| `theboss_tvd.py` | theboss against this engine at the DISTRIBUTION level: the paper's third-party correctness line. |
| `theboss_workcount.py` | theboss against this engine, counted rather than timed: Guan-code addends per exact sample. |
| `time_n.py` | Time single samples at a given (n, m/n): the control points of the frontier extrapolation, and the per-sample cost distribution. |
| `timeloop_frontier.py` | Simulability frontier of the native time-loop sampler: for each bin ratio T/N and total loop length L, the photon number N* at which `LoopRegisterKernel.sample` sustains --target samples/s (default 1 Hz) with T = ratio*N |
| `treewidth_crossover.py` | Depth below which the treewidth sampler (Oh et al. 2022) beats the chain rule: D* at one (n, m). |
| `validate_methods.py` | Validation table of the methods paper: every sampler on small instances against an oracle that shares no code with it, through the SAME call and backend as the benchmark. |
