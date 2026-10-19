"""Locate the simulability threshold n*: the photon number at which a sampler's sustained
throughput drops to --target samples/s (default 1 Hz) for a mode ratio m = c·n, on this GPU.

    python3 bench/find_threshold.py --engine fock    --ratio 2 [--start 30] [--budget 120]
    python3 bench/find_threshold.py --engine emitter --ratio 2 [--start 22] [--budget 240]

engine fock    : n photons in modes 0..n-1 of a Haar-random m x m unitary, `fock_sample`
                 (backend cuda by default).
engine emitter : the n-photon linear-cluster caterpillar state (dual-rail pairs (2k, 2k+1) of the
                 same unitary; needs m >= 2n, i.e. ratio >= 2), `emitter_sample` (backend triton).
                 --emitter-state fock: the Fock limit instead (ALGORITHMS.md §9.2 row 4): the spin frozen
                 (s0 = |0>, one label per slot, M = diag(1, 0)), photon k in mode k -- n single photons in modes
                 0..n-1, the fock engine's state, written in the chi = 2 form the emitter kernels take.
engine sbs     : superposition boson sampling (docs/SUPERPOSITION.md), sampled by the chain rule
                 `product_sample`. n is the SOURCE count, as it is for fock (n single-photon
                 sources) and emitter (n emission events of the one emitter), and m = round(c n) is
                 modes per SOURCE for all three. That is the only comparison that means something
                 in hardware: build n sources, ask how hard the output is to simulate. Each source
                 is cos(pi/4)|0> + sin(pi/4)|1>, so a shot carries k ~ Binomial(n, 1/2) photons --
                 n/2 on average against fock's n. That weakness belongs in the number; it is not
                 normalised away, and the sampler is NOT post-selected, so the measured rate is the
                 rate a real device would force a simulator to match (cost per sample E[cost(k)],
                 tail-dominated, as the lossy frontier's E[rho^k] is).
                 --loss eta puts a uniform transmission on the input side, which for a partially
                 excited source is a MIXTURE of vacuum and a source at a smaller alpha
                 (docs/LOSS.md §8): brightness becomes eta sin^2(alpha) while the coherence only
                 falls as sqrt(eta), so at fixed brightness the pair (alpha, eta) sweeps from the
                 Fock modality (eta = brightness, zero coherence) to pure partial excitation
                 (eta = 1). bench/coherence_scan.py drives that sweep.
                 CAVEAT for any comparison drawn from this: fock and emitter run tuned CUDA/Triton
                 kernels, the sbs chain rule is plain torch with no kernel, so the gap measured here
                 is algorithm AND implementation, not algorithm alone.
engine nlbs   : non-linear boson sampling (docs/NONLINEAR.md): n single photons through U_0, L layers of
                 single-mode photon-number phases (--phi: Kerr f(j) = phi j^2, or --site-phase: an explicit
                 phase vector) and an interferometer, sampled EXACTLY by the rejection sampler
                 `NonlinearCircuit.sample`. n is the source count and m = round(c n) as for fock; there is no
                 post-selection. --sites L sites, --layout same-layer (all in layer 1, distinct modes) or
                 sequential (one per layer, L + 1 interferometers); the site modes are spread over the m
                 modes. The cost per sample is |c|_1^2 (proposals) x (one Fock sample + B branch permanents),
                 so phi = 0 and phi = pi ARE the Fock engine (one branch, no rejection) and the scan in phi
                 measures the toll of the non-linearity. --loss is the transmission BEFORE the sites
                 (--eta-post after them) and --indist the x-model, both exact.
engine gbs    : Gaussian boson sampling (docs/GAUSSIAN.md): n single-mode squeezed vacua on modes 0..n-1 of a
                 Haar U, m = round(c n), sampled EXACTLY (PNR, no photon cutoff, no post-selection) by the
                 chain rule over conditional pure Gaussian states, `gaussian.sample`. n is the number of
                 SQUEEZERS. --mean-photons (per squeezer, default 0.5) or --squeezing r sets r; --loss is
                 exact (pure-plus-classical decomposition); --schmidt lam_1,..,lam_K splits every squeezer
                 into K non-interfering Schmidt modes (spectral impurity; --indist does not apply). The
                 collision-free cost model per squeezer is sech r (1 - (1 - eta + eta sqrt2)^2 tanh^2 r)^-1/2,
                 which DIVERGES for tanh r (1 - eta + eta sqrt2) >= 1 (lossless: sinh^2 r >= 1): above that
                 the mean cost is carried by collision-heavy tail shots and the fit is not to be trusted
                 (docs/GAUSSIAN.md §2). The rate reported is 1 / (cost per sample of a run of --run-samples
                 samples, default 1e6): stratified over the exact photon-number law, truncated where a run
                 meets a shot less than once, fitted tail inside (docs/GAUSSIAN.md §2.1; --estimator); the
                 same shots are also summed for S = 1e4, 1e6, 1e8 (printed, CSV columns). Every shot is capped
                 on its photon count (exact for the stratified estimator: a shot whose final count is under
                 the cap is never aborted; --shot-cap, docs/GAUSSIAN.md §2.1).
engine sbsfixed: DIAGNOSTIC, not a frontier point. The same sources post-selected on the modal
                 outcome k = n/2, which prices one k-photon sample rather than one device shot
                 (alpha drops out of it entirely: post-selected on k the input is the uniform Dicke
                 state whatever alpha is). Useful for the cost curve at fixed photon number; it
                 does not belong on the frontier plot next to fock and emitter.
Method (as gpu-clifford/bench/find_threshold.py): measure the throughput at a few n (short shot
ramps; one sample already fills the GPU at these sizes), fit log(rate) = a - n log(rho_eff) on
the measured points, jump towards the predicted n* (at most --max-jump photons), stop once two
consecutive n bracket the target and interpolate. rho_eff is compared with the Fock theory
rho(c) = (2c+1)^(2c+1) / ((4c)^c (c+1)^(c+1)) (arXiv:2005.04214, Thm 5); for the emitter engine
the per-photon factor also carries the growth of the pure-term counts R_k, so it is larger.
--loss eta: uniform transmission eta on the input side (docs/LOSS.md) for both engines; the
per-photon factor then becomes rho_eff(eta) ~ 1 - eta + eta rho (each photon survives with
probability eta), the slope range and the fallback slope are scaled accordingly, and the sanity
check compares with eta times the lossless marginal. bench/frontier_loss.py drives the eta scan.
--indist x: partial distinguishability, x-model (docs/DISTINGUISHABILITY.md): the same law with
eta -> eta x (a photon interferes when it is kept AND in the common species);
bench/frontier_indist.py drives the (eta, x) scan.
Writes --csv (eta, n, m, samples/s, ms/sample, seconds spent, shots) and prints the result block
that map_threshold.py / frontier_loss.py parse.
"""
import argparse
import csv
import math
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="fock", choices=["fock", "emitter", "sbs", "sbsfixed", "nlbs", "gbs"])
ap.add_argument("--ratio", type=float, default=2.0, help="m = round(ratio * n)")
ap.add_argument("--target", type=float, default=1.0, help="samples/s defining the threshold")
ap.add_argument("--start", type=int, default=None, help="first n (default: 30 fock, 22 emitter, 4 sbs)")
ap.add_argument("--step", type=int, default=2, help="first step in n before the fit takes over")
ap.add_argument("--max-jump", type=int, default=4)
ap.add_argument("--budget", type=float, default=None, help="seconds per n: the shot ramp stops when a run exceeds budget/3 (default 120 "
                                                           "fock, 240 emitter)")
ap.add_argument("--max-shots", type=int, default=256)
ap.add_argument("--min-shots", type=int, default=8, help="runs with fewer shots do not count towards the sustained rate")
ap.add_argument("--bracket-min-shots", type=int, default=0,
                help="after bracketing, top up both bracket points to >= this many counted shots (D2/R1: heavy-tailed "
                     "shot cost, e.g. sbs under loss); the bracket is re-searched if a top-up moves it. 0 = off (default). "
                     "CSV shots column: the counted total for a topped-up point, the largest ramp batch for the others")
ap.add_argument("--bracket-batch", type=int, default=0,
                help="batch size of the --bracket-min-shots top-ups; 0 = max(--min-shots, 16) (default). Quoted rows "
                     "use --max-shots 256 --bracket-batch 256 — extra counted shots only by repeating 256-shot batches, since on "
                     "the lockstep GPU samplers the batch size itself moves the throughput")
ap.add_argument("--bracket-max-s", type=float, default=3600.0,
                help="wall-time guard per top-up: a point that cannot reach --bracket-min-shots in this time is reported "
                     "CENSORED (bracket_censored column) instead of running unbounded")
ap.add_argument("--max-n", type=int, default=None, help="stop the search here (default 60, 20 for sbs: its cost grows ~4x per photon)")
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--backend", default=None, help="fock: cuda | triton (default cuda); emitter: triton | torch (default triton)")
ap.add_argument("--algo", default="auto", choices=["auto", "2018", "2020", "input", "percof", "rso"],
                help="fock: the chain-rule algorithm (docs/ALGORITHMS.md; the T1 set runs on backend triton); sbs/sbsfixed: "
                     "auto = the replica/residual chain (clifford/product_sampler.py), rso = the random-source-order chain "
                     "(clifford/sbs_rso.py)")
ap.add_argument("--kernel", default="auto", help="emitter/triton: auto (default: cuda at the frontier steps, seq elsewhere) | "
                "seq | scan (Triton) | cuda (clifford/cuda_emitter, needs cupy)")
ap.add_argument("--rso-backend", default="auto", help="--algo rso: cofactor backend, auto | torch | cuda | collapsed | "
                "triton (clifford/sbs_rso.py; torch = no C&C 2020 collapse)")
ap.add_argument("--rso-collapse-from", type=int, default=12, help="--algo rso: first step (placed rows) using the collapse")
ap.add_argument("--rso-square", default="auto", choices=["auto", "chain", "cuda", "triton", "kernel"],
                help="--algo rso: strata with k = n_live (clifford/sbs_rso.py sample(square=); kernel = the rso chain on "
                "the Fock step kernels)")
ap.add_argument("--rso-draw", default="host", choices=["host", "device"],
                help="--algo rso: row draw on the host or the fused device draw (sample(draw=))")
ap.add_argument("--emitter-state", default="cluster", choices=["cluster", "fock"],
                help="emitter: cluster = the linear-cluster caterpillar (default); fock = the frozen-spin chi = 2 state "
                "equal to n single photons in modes 0..n-1 (Fock limit test, ALGORITHMS.md §9.2)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--slope-range", default=None, help="accepted rho_eff range for the jump prediction, 'lo,hi' (default 1.5,2.1 fock; "
                                                    "1.3,3.5 emitter)")
ap.add_argument("--loss", type=float, default=None, help="uniform transmission eta (default: lossless)")
ap.add_argument("--indist", type=float, default=None,
                help="pairwise overlap x of the x-model (docs/DISTINGUISHABILITY.md; default: indistinguishable)")
ap.add_argument("--alpha", type=float, default=None, help="sbs: pulse area / 2, the source being cos(alpha)|0> + "
                                                          "sin(alpha)|1> (default pi/4). With --loss the brightness "
                                                          "is eta sin^2(alpha); see bench/coherence_scan.py")
ap.add_argument("--phi", default="pi/2", help="nlbs: Kerr strength, f(j) = phi j^2, as an expression in pi "
                                              "(default pi/2, the maximally non-linear point). Give it exactly "
                                              "('pi/4', not 0.7853982): the branch expansion finds the short "
                                              "period of the response in exact arithmetic (docs/NONLINEAR.md §2)")
ap.add_argument("--site-phase", default=None, help="nlbs: explicit phase vector f(0),f(1),..,f(n) (comma-separated), "
                                                   "instead of --phi; must cover the largest n measured")
ap.add_argument("--sites", type=int, default=1, help="nlbs: number of non-linear sites")
ap.add_argument("--layout", default="same-layer", choices=["same-layer", "sequential"], help="nlbs: sites in one layer "
                                                                                             "(distinct modes) or one per layer")
ap.add_argument("--eta-post", type=float, default=None, help="nlbs: transmission AFTER the last site (--loss is before)")
ap.add_argument("--mean-photons", type=float, default=None, help="gbs: sinh^2 r per squeezer (default 0.5)")
ap.add_argument("--squeezing", type=float, default=None, help="gbs: squeezing r (overrides --mean-photons)")
ap.add_argument("--schmidt", default=None, help="gbs: Schmidt weights lam_1,..,lam_K (sum 1) of every squeezer")
ap.add_argument("--impl", default="v2", choices=["v1", "v2"], help="gbs: sampler version (docs/GAUSSIAN.md §3.4)")
ap.add_argument("--window", type=int, default=None, help="gbs, impl v2: steps batched over a run of empty modes "
                                                           "(default 4)")
ap.add_argument("--estimator", default="run", choices=["run", "is", "stratified", "raw"],
                help="gbs: the cost per sample of a run of --run-samples samples (default: stratified by N over the "
                     "exact photon-number law, truncated at N_S = the 1 - 1/S quantile, fitted tail inside it), the "
                     "untruncated mean by defensive importance sampling at a tilted squeezing (exact weights, lossless "
                     "undisplaced only; the tilted shots are ~4x brighter and can be hours each at n > 30), the "
                     "untruncated stratified mean, or the raw mean (docs/GAUSSIAN.md §2, gaussian/cost.py)")
ap.add_argument("--run-samples", type=float, default=1e6, help="gbs, estimator run: samples S of the run (default 1e6)")
ap.add_argument("--shot-cap", type=int, default=None,
                help="gbs: abort a shot once its photon count exceeds this (default: adaptive, the smaller of the "
                     "1 - 1/(8 max-shots) quantile of the exact law and the largest N whose fitted cost is "
                     "<= budget / --cap-frac); -1 disables the cap. Not applied under --estimator is")
ap.add_argument("--cap-frac", type=float, default=10.0, help="gbs: a capped shot costs at most budget / cap-frac (default 10)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv", default="threshold.csv")
args = ap.parse_args()

dev = args.device
c = args.ratio
emitter = args.engine == "emitter"
nlbs = args.engine == "nlbs"
sbs = args.engine in ("sbs", "sbsfixed")
rso = sbs and args.algo == "rso"                              # sbs: Clifford & Clifford with an offset start set
if args.algo == "rso" and not sbs:
    raise SystemExit("--algo rso: sbs / sbsfixed engines only")
gbs = args.engine == "gbs"
if gbs and args.indist is not None:
    raise SystemExit("gbs engine: partial distinguishability is --schmidt, not --indist (docs/GAUSSIAN.md §4.2)")
r_gbs = (args.squeezing if args.squeezing is not None else math.asinh(math.sqrt(args.mean_photons if args.mean_photons is not None
         else 0.5))) if gbs else 0.0
schmidt = [float(v) for v in args.schmidt.split(",")] if (gbs and args.schmidt) else None
postselect = args.engine == "sbsfixed"                        # sbs: k ~ Binomial(n, 1/2) per shot
emitter_fock = emitter and args.emitter_state == "fock"
if emitter and c < 2 and not emitter_fock:
    raise SystemExit("emitter engine: the dual-rail caterpillar needs m >= 2n (ratio >= 2)")
if sbs and c < 1:
    raise SystemExit(f"{args.engine} engine: the n sources need n distinct input modes (ratio >= 1)")
start = args.start or (24 if rso else 8 if sbs else 22 if emitter else 26 if nlbs else 12 if gbs else 30)
max_n = args.max_n or (120 if rso else 40 if sbs else 120 if gbs else 60)
budget = args.budget or (240.0 if (emitter or sbs or gbs) else 120.0)
backend = args.backend or ("torch" if (sbs or gbs) else "triton" if emitter else "cuda")
eta_post = None if args.eta_post is None or args.eta_post >= 1.0 else float(args.eta_post)
phi = float(eval(args.phi, {"__builtins__": {}}, {"pi": math.pi, "sqrt": math.sqrt})) if nlbs else 0.0
eta = 1.0 if args.loss is None or args.loss >= 1.0 else float(args.loss)
loss = None if eta >= 1.0 else eta
xd = 1.0 if args.indist is None or args.indist >= 1.0 else float(args.indist)
indist = None if xd >= 1.0 else xd
q = eta * xd                                                   # a photon interferes when kept AND in the common species
lo_r, hi_r = (1.1, 2.5) if rso else (1.3, 6.0) if sbs else (1.3, 3.5) if emitter else (1.5, 2.1)
if not sbs:                                                   # cost per sample ~ E[rho^g], g ~ Bin(n, eta x)
    lo_r, hi_r = 1 + q * (lo_r - 1), 1 + q * (hi_r - 1)
# sbs is left unscaled on purpose: its per-SOURCE factor already carries a brightness of
# sin^2(alpha), and the scan that uses --loss holds the brightness eta sin^2(alpha) FIXED while it
# trades loss against coherence, so the factor should stay in the band it has lossless (~2.05).
if args.slope_range:
    lo_r, hi_r = (float(x) for x in args.slope_range.split(","))
rho_th = (2 * c + 1) ** (2 * c + 1) / ((4 * c) ** c * (c + 1) ** (c + 1))
rho_fb = 1.5 if rso else 2.2 if sbs else 1 + q * ((2.2 if emitter else rho_th) - 1)    # fallback slope when the fit is out of range
if gbs:                                                       # collision-free cost model per squeezer (docs/GAUSSIAN.md §2)
    def _rho_src(r):
        t2 = (math.tanh(r) * (1 - eta + eta * math.sqrt(2))) ** 2
        return math.inf if t2 >= 1 else 1 / (math.cosh(r) * math.sqrt(1 - t2))
    rho_gbs = math.prod(_rho_src(r_gbs * math.sqrt(lk)) for lk in (schmidt or [1.0]))
    lo_r, hi_r = (1.01, 4.0) if not args.slope_range else (lo_r, hi_r)
    rho_fb = rho_gbs if math.isfinite(rho_gbs) else 2.0
    print(f"  gbs: impl={args.impl} window={args.window}  r={r_gbs:.4f} "
          f"(sinh^2 r = {math.sinh(r_gbs) ** 2:.3f} per squeezer), schmidt={schmidt}, "
          f"collision-free model rho_src={rho_gbs:.3f} per squeezer"
          + ("  [DIVERGENT: tail-dominated regime]" if not math.isfinite(rho_gbs) else ""), flush=True)
name = torch.cuda.get_device_name(0) if dev == "cuda" and torch.cuda.is_available() else dev
if nlbs:
    print(f"  nlbs: sites={args.sites} layout={args.layout} "
          f"{'phase vector' if args.site_phase else f'phi={args.phi} = {phi:.6f}'}  eta_post={eta_post or 1:g}", flush=True)
algo_txt = "" if args.algo == "auto" else "  algo=" + args.algo
print(f"{name}  engine={args.engine}  backend={backend}{algo_txt}  ratio m/n={c}  target={args.target} samples/s  "
      f"eta={eta:g}  x={xd:g}  theoretical Fock rho(c)={rho_th:.3f}", flush=True)

from photonic_engines import EmitterMPS, GaussianInput, ProductSuperposition, spin_gate
from photonic_engines import gaussian as gaussian_engine
from photonic_engines.clifford import emitter_sample, fock_sample, product_sample
from photonic_engines.clifford.sbs_rso import sample as sbs_rso_sample
from photonic_engines.gaussian.cost import cost_growth, detected_law, mixture_cost, run_cutoff, stratified_cost, tilted
from photonic_engines.nonlinear import NonlinearCircuit


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def cost_fit(Ns, ts):
    """(a, b) of log t = a + b N over the upper half of the drawn N (as `cost.stratified_cost`), b >= 0;
    None with fewer than two distinct N."""
    N = np.asarray(Ns, dtype=float); t = np.asarray(ts, dtype=float)
    ns = np.unique(N)
    if len(ns) < 2:
        return None
    lo = ns[len(ns) // 2] if len(ns) > 3 else ns[0]
    sel = lo <= N
    if len(np.unique(N[sel])) < 2:
        sel = np.ones_like(N, dtype=bool)
    b, a = np.polyfit(N[sel], np.log(np.maximum(t[sel], 1e-9)), 1)
    return a, max(b, 0.0)


prior_shots = []                                              # gbs: (N, seconds) of the previous n, seeds the cap


def measure(n, topup=0):
    """Sustained throughput at n: shot ramp 1, 2, 4, ...; rate = samples / time over the runs with
    >= min_shots shots (the max over the ramp would be biased by lucky small batches).
    topup > 0 (--bracket-min-shots): no ramp; more shots at n, in batches of max(min_shots, 16) on fresh seeds, until
    the counted shots of n (ramp + earlier top-ups, `counted`) reach topup or --bracket-max-s elapses (censored)."""
    if topup and gbs:
        raise SystemExit("--bracket-min-shots: not for gbs (its estimators weight the shots)")
    m = round(c * n)
    U = haar(m, args.seed + n)
    Ut = torch.tensor(U, device=dev)
    if emitter_fock:                           # frozen spin: |0> throughout, photon k into mode k (the Fock state)
        ops = np.zeros((n, 1, 2, 2), dtype=np.complex128)
        ops[:, 0, 0, 0] = 1.0
        st = EmitterMPS(ops=ops, modes=tuple((k,) for k in range(n)), s0=np.array([1.0, 0.0], dtype=np.complex128), m=m)
    elif emitter:
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
    if sbs:                                    # n SOURCES; at alpha = pi/4, n/2 photons on average
        st = ProductSuperposition.sbs(n, m, **({} if args.alpha is None else {"alpha": args.alpha}))
    in_modes = list(range(n))
    if gbs:
        st = GaussianInput.smsv(n, m, r_gbs)
        est = args.estimator
        if est == "is" and (loss is not None or schmidt is not None):
            print("  (estimator is: not exact under loss / Schmidt modes -> stratified)", flush=True)
            est = "stratified"
        K = 12 * m + 60                                       # photon-number laws, far past anything drawn
        law_target = detected_law(st, K, loss=loss, schmidt=schmidt)
        laws = [law_target]                                   # one per sampling phase (target, then the tilt)
        cur = [st]                                            # the state being sampled
        shot_log = []                                         # (N, seconds, phase) of every counted, uncensored shot
        censored = [0]                                        # shots aborted at the cap (counted runs)
        use_cap = est != "is" and (args.shot_cap is None or args.shot_cap >= 0)
        cap_q = run_cutoff(law_target, 8 * args.max_shots)   # the cap never censors shots commoner than this...
        cap_lo = run_cutoff(law_target, 64)                   # ...and never ones commoner than 1 in 64 (floor)

        def shot_cap():
            if not use_cap:
                return None
            if args.shot_cap is not None:
                return args.shot_cap
            mine = [(p, sec) for p, sec, _ in shot_log]
            pool = mine if len(mine) >= 16 else mine + prior_shots   # a fit on a handful of shots is wild
            fit = cost_fit(*zip(*pool)) if pool else None
            if fit is None or fit[1] <= 0:
                return cap_q
            a, b = fit
            return int(min(cap_q, max(cap_lo, math.floor((math.log(budget / args.cap_frac) - a) / b))))
        cap_used = []
    if nlbs:
        f = [float(v) for v in args.site_phase.split(",")] if args.site_phase else phi
        xs = [min(m - 1, round((i + 0.5) * m / args.sites)) for i in range(args.sites)]    # site modes, spread over m
        if args.layout == "same-layer":
            Us = [torch.tensor(haar(m, args.seed + n + 1000 * i), device=dev) for i in range(2)]
            sites = [(1, x, f) for x in xs]
        else:
            Us = [torch.tensor(haar(m, args.seed + n + 1000 * i), device=dev) for i in range(args.sites + 1)]
            sites = [(i + 1, x, f) for i, x in enumerate(xs)]
        C = NonlinearCircuit(Us, in_modes, sites)
    run_id = [0]

    def run(shots):
        run_id[0] += 1
        if nlbs:
            gen = torch.Generator(device=dev).manual_seed(args.seed + 1000 * run_id[0])
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            occ = C.sample(shots, generator=gen, backend=backend, loss=loss, indist=indist, eta_post=eta_post)
        elif gbs:
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            cap = shot_cap()
            cap_used.append(cap)
            smp, stats = gaussian_engine.sample(cur[0], U, shots, loss=loss, seed=args.seed + 1000 * run_id[0],
                                                device=dev, schmidt=schmidt, impl=args.impl, window=args.window,
                                                return_stats=True, max_photons=cap)
            keep = ~np.asarray(stats.censored, dtype=bool)
            occ = torch.as_tensor(smp[keep] if keep.any() else smp[:0])
            if run_id[0] > 1:                                 # not the warm-up
                shot_log.extend((p, sec, len(laws) - 1) for p, sec, c_ in zip(stats.photons, stats.seconds, stats.censored)
                                if not c_)
                censored[0] += int((~keep).sum())
        elif sbs:
            gen = torch.Generator().manual_seed(args.seed + 1000 * run_id[0])
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            if rso:
                occ = sbs_rso_sample(st, Ut, shots, generator=gen, device=dev, loss=loss, indist=indist,
                                     photons=(n // 2) if postselect else None, backend=args.rso_backend,
                                     collapse_from=args.rso_collapse_from, square=args.rso_square,
                                     draw=args.rso_draw)
            else:
                occ = product_sample(st, Ut, shots, generator=gen, device=dev, loss=loss,
                                     photons=(n // 2) if postselect else None)
        elif emitter:
            gen = torch.Generator().manual_seed(args.seed + 1000 * run_id[0])
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            occ = emitter_sample(st, Ut, shots, generator=gen, backend=backend, device=dev, kernel=args.kernel, loss=loss, indist=indist)
        else:
            gen = torch.Generator(device=dev).manual_seed(args.seed + 1000 * run_id[0])
            torch.cuda.synchronize()
            t = time.perf_counter()
            occ = fock_sample(Ut, in_modes, shots, generator=gen, algo=args.algo, backend=backend, loss=loss, indist=indist)
        if dev == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t, occ

    t_warm = time.perf_counter()
    if gbs and dev == "cuda" and not measure.__dict__.get("warm"):
        try:                                                  # every Triton sieve variant, once per process
            from photonic_engines.gaussian.sieve_triton import warmup
            warmup(dev, max(8, args.window or 4))
        except ImportError:
            pass
        measure.warm = True
    run(1)                                                    # warm-up: kernel compiles (cached on disk afterwards)
    warm = time.perf_counter() - t_warm
    t0 = time.perf_counter()

    def ramp(bud):
        shots, spent = 1, 0.0
        tot_shots, tot_time, best, last, occ = 0, 0.0, 0.0, 1, None
        while shots <= args.max_shots:
            dt, occ = run(shots)
            last = shots
            spent += dt
            if shots >= args.min_shots:
                tot_shots += shots
                tot_time += dt
            best = shots / dt if tot_shots == 0 else tot_shots / tot_time
            if dt > bud / 3 or spent > bud:
                break
            shots *= 2
        counted[n] = (tot_shots, tot_time)
        return best, last, spent, occ

    def top_up(goal):
        k = topups[n] = topups.get(n, 0) + 1
        run_id[0] = 100000 * k                                # fresh seeds: never the ramp's or an earlier top-up's
        c0, t0c = counted.get(n, (0, 0.0))
        batch, added, spent, occ = args.bracket_batch or max(args.min_shots, 16), 0, 0.0, None
        shown = 0.0                                           # progress line every >= 600 s of top-up (checkpoints, P)
        while c0 + added < goal and spent < args.bracket_max_s:
            dt, occ = run(batch)
            added += batch
            spent += dt
            if spent - shown >= 600.0 and c0 + added < goal:
                shown = spent
                print(f"  top-up progress n={n}: +{added} shots in {spent:.0f} s -> {c0 + added}/{goal} counted, "
                      f"{(c0 + added) / (t0c + spent):.3f} samples/s so far", flush=True)
        tot, tt = c0 + added, t0c + spent
        counted[n] = (tot, tt)
        bracket_censored[n] = tot < goal
        print(f"  top-up n={n}: +{added} shots (batches of {batch}) in {spent:.0f} s -> {tot} counted shots, {tot / tt:.3f} samples/s"
              + (f"  CENSORED (--bracket-max-s {args.bracket_max_s:g} s before {goal} shots)" if tot < goal else ""),
              flush=True)
        return tot / tt, tot, spent, occ

    gbs_is = gbs and est == "is"
    if topup:
        best, last, spent, occ = top_up(topup)
    else:
        best, last, spent, occ = ramp(budget / 2 if gbs_is else budget)
    gbs_note = ""
    if gbs_is and shot_log:
        # defensive importance sampling: the target shots above, then shots at the squeezing tilted by the
        # measured per-photon cost growth (s = exp(b), at most sqrt 2), weights by the balance heuristic
        Ns0, ts0, _ = zip(*shot_log)
        s_want = min(math.sqrt(2), cost_growth(Ns0, ts0))
        st_t, s_used = tilted(st, s_want)
        if s_used > 1.02:
            cur[0] = st_t
            laws.append(detected_law(st_t, K))
            b2, l2, sp2, occ = ramp(budget / 2)
            spent += sp2; last = l2
        Ns, ts, ph = zip(*shot_log)
        raw_rate = sum(p == 0 for p in ph) / sum(t for t, p in zip(ts, ph) if p == 0)
        e = mixture_cost(Ns, ts, ph, law_target, laws)
        best = 1.0 / e.seconds
        gbs_note = (f", defensive IS (tilt s={s_used:.3f}, {sum(p == 1 for p in ph)} tilted + {sum(p == 0 for p in ph)} "
                    f"target shots): {e.seconds:.3f} +- {e.stderr:.3f} s/sample (rel. err {e.stderr / e.seconds:.1%}), "
                    f"raw target rate {raw_rate:.3f}/s")
        gbs_info[n] = ("is", raw_rate, e.stderr / e.seconds)
    elif gbs and est != "raw" and shot_log:
        Ns, ts, _ = zip(*shot_log)
        raw_rate = len(ts) / sum(ts)
        NS = run_cutoff(law_target, args.run_samples) if est == "run" else None
        e = stratified_cost(Ns, ts, law_target, max_N=NS)
        best = 1.0 / e.seconds
        what = f"run of {args.run_samples:.0e} samples (N <= {NS})" if NS is not None else "stratified"
        caps = [k for k in cap_used if k is not None]
        cap_txt = f"cap N <= {min(caps)}..{max(caps)}, {censored[0]} censored" if caps else "no cap"
        multi = []
        for S in (1e4, 1e6, 1e8):
            eS = stratified_cost(Ns, ts, law_target, max_N=run_cutoff(law_target, S))
            multi += [eS.seconds, eS.extrapolated]
        gbs_note = (f", {what}: "
                    f"{e.seconds:.3f} s/sample, {e.extrapolated:.0%} of it extrapolated past N={e.max_drawn} "
                    f"(growth {e.growth_per_photon:.3f}/photon{'' if e.converges else ', tail NOT converged'}), raw {raw_rate:.3f}/s, "
                    f"{cap_txt}; S=1e4/1e6/1e8: " + " / ".join(f"{multi[2 * i]:.3f} s ({multi[2 * i + 1]:.0%} ext.)" for i in range(3)))
        gbs_info[n] = (est, raw_rate, e.extrapolated, min(caps) if caps else "", censored[0], *multi)
        prior_shots[:] = [(p, sec) for p, sec, _ in shot_log]
    X = occ.cpu().numpy().astype(np.float64)
    # sanity: mean photon number per output mode vs the exact marginal (Fock: |U|^2 column sums;
    # emitter: the single-photon reduced states are maximally mixed on each dual-rail pair;
    # sbsfixed: n/2 times the exact one-photon marginal, which the sampler's first step computes;
    # sbs: the coherent mean occupancy, see below -- NOT half the Fock one)
    if nlbs:
        # a site conserves the photon number, so the only U-independent exact check at frontier sizes is the
        # DETECTED photon number, n eta eta_post (the mean occupancy per mode is not |U|^2 column sums here,
        # and computing it exactly costs a permanent of identity + rank per branch pair). At phi = 0 and
        # phi = pi the circuit is linear and the branch unitary is the Fock check of the `else` branch.
        pred = None
        err = abs(float(X.sum(1).mean()) - n * eta * (eta_post or 1.0))
    elif gbs:
        # <n_j> = eta sum_i |U_ji|^2 sum_k sinh^2(r sqrt(lam_k)) for undisplaced squeezed vacua (exact, U-dependent)
        # (of the state sampled LAST: the tilted one under --estimator is, when a tilt was used)
        mean_src = sum(math.sinh(float(cur[0].r[0]) * math.sqrt(lk)) ** 2 for lk in (schmidt or [1.0]))
        pred = eta * mean_src * (np.abs(U[:, :n]) ** 2).sum(1)
    elif sbs and postselect:
        from photonic_engines.clifford.product_sampler import marginals
        w = marginals(st, Ut, [], list(range(m)), n // 2, device=dev).cpu().numpy()
        pred = eta * (n // 2) * w / w.sum()
    elif sbs:
        # unconditioned. E[n_j] = sum_{a,a'} conj(U_ja) U_ja' <a_a^dag a_a'>, and the sources are
        # SUPERPOSED, so the off-diagonal terms do not vanish: <a_a> = conj(c_a) s_a is the
        # coherence between the 0- and 1-photon parts of source a. With p_a = |s_a|^2,
        #     E[n_j] = sum_a |U_ja|^2 (p_a - |chi_a|^2)  +  |sum_a U_ja chi_a|^2 .
        # Dropping the second term gives exactly the INCOHERENT twin (input loss at eta = p), which
        # is the quantity this whole modality is about, so the check is sensitive to the coherence
        # rather than blind to it. Verified against ProductSLOSKernel.probs to 1e-15 at alpha =
        # pi/4, pi/3 and 0.4, and sum_j E[n_j] = n sin^2(alpha) exactly. (The first version of this
        # branch compared the measured mean against itself and scored 0.000 everywhere; the second
        # used the incoherent form and was off by 0.3-0.6, which run 20260920_121038 duly reported
        # as a sampler error at every sbs point.)
        # Under loss it is NOT eta times the lossless value: p and chi scale differently, chi with
        # sqrt(eta) and p with eta (docs/LOSS.md §8), so this branch builds the final number itself.
        amps = np.array(st.amps)
        chi = np.conj(amps[:, 0]) * amps[:, 1] * math.sqrt(eta)        # <a_a> after transmission eta
        pa = np.abs(amps[:, 1]) ** 2 * eta                             # P(source a is detected)
        Uc = U[:, list(st.modes)]
        pred = ((np.abs(Uc) ** 2) * (pa - np.abs(chi) ** 2)).sum(1) + np.abs(Uc @ chi) ** 2
    elif emitter and not emitter_fock:
        pred = eta * 0.5 * (np.abs(U[:, :2 * n]) ** 2).sum(1)
    else:
        pred = eta * (np.abs(U[:, :n]) ** 2).sum(1)
    if pred is not None:
        err = float(np.abs(X.mean(0) - pred).max())
    label = "d<N>" if nlbs else "max|dE[n_j]|"
    print(f"  n={n:2d} m={m:3d}: {best:9.3f} samples/s  ({1e3 / best:9.2f} ms/sample, up to {last} shots, "
          f"{time.perf_counter() - t0:.0f} s + {warm:.0f} s warm-up, {label}={err:.3f}"
          + (gbs_note if gbs else "")
          + (f", accept {C.stats['accepted'] / max(C.stats['proposals'], 1):.3f} of {C.stats['proposals']} proposals, "
             f"B={C.n_branches}, 1/|c|_1^2={1 / C.rejection_factor():.3f}" if nlbs else "") + ")", flush=True)
    return m, best, spent, last


points = {}                                                   # n -> (m, rate, seconds, shots)
counted = {}                                                  # n -> (counted shots, their seconds): ramp + top-ups
topups = {}                                                   # n -> number of top-ups (seeds)
bracket_censored = {}                                         # n -> True when a top-up hit --bracket-max-s
gbs_info = {}                                                 # gbs: n -> (estimator, raw rate, relative error or extrapolated share)


def bracket_of(points):
    """(top, above, inverted): the largest n that met the target, the slower n above it, and the slower n
    BELOW it. A bracket is (top, min(above)) with min(above) - top == 1. Cost per shot is tail-dominated
    (sbs: E[cost(k)] over k ~ Binomial(n, p)), so a larger n can measure FASTER than a smaller one; the
    first version took max(lo), min(hi) whatever their order, and reported n = 29 slower / n = 31 faster
    as the bracket "n* = 29.3" with rho_eff 0.892 (run 20260920_193154, sbs at eta = 0.5). The threshold
    is anchored on the largest n that met the target, and a slower point below it is reported, not used.
    """
    ns = sorted(points)
    lo = [k for k in ns if points[k][1] >= args.target]
    hi = [k for k in ns if points[k][1] < args.target]
    top = max(lo) if lo else None
    return top, [k for k in hi if top is None or k > top], [k for k in hi if top is not None and k < top]
log = math.log
n = start
points[n] = measure(n)
n2 = n + args.step
points[n2] = measure(n2)


def search():
    while True:
        ns = sorted(points)
        top, above, _ = bracket_of(points)
        if top is not None and above and min(above) - top == 1:
            break                                             # bracketed by consecutive n, in order
        b = ns[-1]
        if len(ns) >= 3:
            slope = np.polyfit(np.array(ns, dtype=float), np.log([points[k][1] for k in ns]), 1)[0]
        else:
            slope = (log(points[ns[-1]][1]) - log(points[ns[-2]][1])) / (ns[-1] - ns[-2])
        if not (-log(hi_r) <= slope <= -log(lo_r)):
            slope = -log(rho_fb)
        n_pred = b + (log(args.target) - log(points[b][1])) / slope
        n_pred = min(max(n_pred, b - args.max_jump), b + args.max_jump)
        if top is not None and above:
            nxt = (top + min(above)) // 2
        else:
            nxt = math.floor(n_pred) if not above else math.ceil(n_pred)
            nxt = max(nxt, b + 1) if not above else min(nxt, min(above) - 1)
        if nxt is None or nxt in points or nxt > max_n or nxt < 1:
            break
        print(f"  fit: rho_eff={math.exp(-slope):.3f}, predicted n*={n_pred:.1f} -> measuring n={nxt}", flush=True)
        points[nxt] = measure(nxt)


search()
for _ in range(4 if args.bracket_min_shots > 0 else 0):      # D2/R1: >= N counted shots at both bracket points
    top, above, _ = bracket_of(points)
    if top is None or not above:
        break
    todo = [k for k in (top, min(above))
            if counted.get(k, (0, 0.0))[0] < args.bracket_min_shots and not bracket_censored.get(k)]
    if not todo:
        break
    for k in todo:
        prev_sec = points[k][2]
        mk, rk, sk, shk = measure(k, topup=args.bracket_min_shots)
        points[k] = (mk, rk, prev_sec + sk, shk)
    search()                                                  # a top-up can move the bracket
if args.bracket_min_shots > 0:                                # rounds exhausted with a moved bracket: flag, never hide
    top, above, _ = bracket_of(points)
    if top is not None and above:
        short = [k for k in (top, min(above)) if counted.get(k, (0, 0.0))[0] < args.bracket_min_shots]
        for k in short:
            bracket_censored[k] = True
        if short:
            print(f"  BRACKET UNDER-SHOT n={short}: fewer than {args.bracket_min_shots} counted shots after the top-up "
                  f"rounds (marked bracket_censored)", flush=True)

ns = sorted(points)
top, above, inverted = bracket_of(points)
with open(args.csv, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["engine", "ratio", "eta", "x", "n", "m", "samples_per_s", "ms_per_sample", "seconds", "shots",
                "phi", "sites", "layout", "eta_post", "r", "schmidt", "impl", "estimator", "raw_rate", "est_err",
                "shot_cap", "censored", "s_1e4", "ext_1e4", "s_1e6", "ext_1e6", "s_1e8", "ext_1e8"]
               + (["bracket_censored"] if args.bracket_min_shots > 0 else []))
    for k in ns:
        m, rate, sec, sh = points[k]
        w.writerow([args.engine, c, eta, xd, k, m, rate, 1e3 / rate, round(sec, 1), sh,
                    args.phi if nlbs and not args.site_phase else "", args.sites if nlbs else "",
                    args.layout if nlbs else "", eta_post or 1.0 if nlbs else "",
                    round(r_gbs, 6) if gbs else "", args.schmidt or "" if gbs else "",
                    args.impl if gbs else (args.algo if args.algo != "auto" else ""),
                    *((tuple(gbs_info.get(k, ("raw", "", ""))) + ("",) * 11)[:11] if gbs else ("",) * 11)]
                   + ([int(bracket_censored.get(k, False))] if args.bracket_min_shots > 0 else []))
xs = np.array(ns, dtype=float)
ys = np.log([points[k][1] for k in ns])
slope, icpt = np.polyfit(xs, ys, 1)
rho_eff = math.exp(-slope)
n_star = (log(args.target) - icpt) / slope
print("=== result ===")
print(f"  engine = {args.engine}  ratio = {c:g}  eta = {eta:g}  x = {xd:g}")
print(f"  rho_eff = {rho_eff:.3f} per photon (Fock theory rho({c:g}) = {rho_th:.3f}, lossy 1 - eta x + eta x rho = "
      f"{1 + q * (rho_th - 1):.3f});  fit over n = {ns}")
if inverted:
    print(f"  NON-MONOTONE: n={inverted} measured slower than n={top}, which met the target; the threshold "
          f"is anchored on n={top} (tail-dominated cost per shot)")
if top is not None and above:
    a, b = top, min(above)
    ra, rb = points[a][1], points[b][1]
    n_int = a + (log(args.target) - log(ra)) / (log(rb) - log(ra)) * (b - a)
    print(f"  bracket: n={a} at {ra:.2f}/s, n={b} at {rb:.2f}/s  ->  n* = {n_int:.1f} for {args.target:g} samples/s on 1 GPU")
    cens = [k for k in (a, b) if bracket_censored.get(k)]
    if cens:
        print(f"  BRACKET CENSORED at n={cens}: fewer than --bracket-min-shots {args.bracket_min_shots} counted shots "
              f"within --bracket-max-s {args.bracket_max_s:g} s; this n* is not quotable under D2/R1")
else:
    n_int = n_star
    print(f"  not bracketed (max-n or budget reached); fitted n* = {n_star:.1f}")
if args.gpus > 1:
    print(f"  with {args.gpus} GPUs: n* + {log(args.gpus) / log(rho_eff):.1f} = {n_int + log(args.gpus) / log(rho_eff):.1f}")
print(f"  wrote {args.csv}")
