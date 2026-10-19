"""The sbs frontier under the run-of-S statistic of docs/GAUSSIAN.md §2.1, for comparability with GBS.

GBS prices a point by the mean cost per sample of a run of S samples rather than by the untruncated
mean, because its cost is heavy-tailed: with cost ~ rho^N the cost-weighted photon law is the law of
brighter squeezers, so the mean is carried by shots a run never contains.

    t_S = sum_{k <= k_S} P(k) E[cost | k],    k_S = smallest k with P(K > k) <= 1/S

Two things make the sbs case easier than the Gaussian one, and one makes it different.

  * P(k) is exact and U-independent: Binomial(n, p), p = eta sin^2(alpha)  -- `photon_law`.
  * E[cost | k] needs no stratum sampling and no surrogate patterns. The chain has exactly k steps and
    step j scores m candidates at a cost that `cheaper(n, k, j)` and the two evaluations fix from
    (n, k, j) alone -- never from WHICH rows were drawn. So the cost of a shot is a deterministic
    function c(n, m, k), priced once per stratum: `--price measure` times one post-selected k-photon
    shot, `--price model` evaluates the calibrated model below.
  * The law is BOUNDED at k = n with Gaussian tails, where the Gaussian one is geometric in pairs and
    unbounded. That is why the cut barely moves an sbs number (§2 of the module: 0.993 of the
    untruncated mean at n = 20 lossless, 0.808 at eta = 0.5) while it moves a GBS one by a factor.

Under loss the chain runs on the LIVE sources only (docs/LOSS.md §5b: a lossy partially excited source
is a mixture of vacuum and a source at a smaller alpha), so the cost of a shot is c(n_live, m, k) and
E[cost | k] averages over n_live -- which the same measured table c(n, m, k) prices.

    python3 bench/sbs_run_cost.py --ratios 1,2,4,12 --price model
    python3 bench/sbs_run_cost.py --ratios 2 --eta 1,0.9,0.8,0.7,0.5 --price measure --device cuda
    python3 bench/sbs_run_cost.py --algo rso --price fit --device cuda --kmeas 20      # the rso chain

`--algo rso` prices the random-source-order chain of clifford/sbs_rso.py instead of
the replica/residual chain. Step j of a k-photon shot on n (live) sources works on N_j = n-k+j columns with q = j-1 placed
rows; s0 = N_j - 1 - q = n - k is the same at every step and fixes the payload the CUDA backend runs
(`cuda_rect.choose_form`: the Fock kernel at s0 = 0, the mu-form for 1 <= s0 <= MU_AUTO_MAX, the lambda-form above).
The number of Glynn terms is the Clifford & Clifford 2020 collapsed count prod_nu (mu_nu + 1), halved by the
complement symmetry, over the multiplicities mu of the q placed rows. It depends on WHICH rows were drawn, so the
model uses its Haar average, in closed form: the q placed rows are distributed as a q-photon state sent through U
(the reduced state of the chain's output), and any q-photon state averaged over Haar U is uniform over the
C(m+q-1, q) output multisets (Schur: the symmetric subspace is irreducible; the bosonic birthday law), and
sum_multisets prod (mu+1) = [x^q] (1-x)^(-2m) = C(2m+q-1, q), so

    E[terms](q, m) = (C(2m+q-1, q) + E_q) / (2 C(m+q-1, q))      (q >= 1; = 2^(q-1) without collisions),

E_q the all-even-multiplicity part, which the halving skips (`rso_terms`; zero for odd q),

whose growth along the chain reproduces the Fock engine's rho(c) (C(3n,n)/C(2n,n) -> 1.6875^n at m = n, 1.808^n at
m = 2n). Steps below `--rso-collapse-from` on the torch "collapsed"/"triton" backends, and every step of the plain
"torch" backend, use the uncollapsed 2^(q-1). The shot cost is then

    c_rso(n, m, k) = FLOOR k + sum_{f in fock, mu, lambda} A_f sum_j E[terms](j-1, m) w_f(N_j, j, s0) + B sum_j m N_j

with per-term payloads w_fock = N, w_mu = N (s0+1)^2, w_lambda = N^2/2 + N j (the three forms' operation counts;
steps wider than the kernel's tier ladder run the collapsed Triton sum and are counted as lambda). `--price fit`
times post-selected shots on a grid of (n, m, k) that covers every form (s0 = 0..3 next to the k <= kmeas strata),
fits the five constants by least squares on the relative error (non-negative), prints the residuals; `--rso-consts`
reuses a fit (five values; three values = the legacy uncollapsed model of runs <= 20261004, one sign-work constant).
The strata that carry t_S at the frontier (k ~ 30) cannot be timed one by one: they are priced by the model.

Writes t_S for S = 1e4 / 1e6 / 1e8 and untruncated, the cut k_S, the share of t_S that rests on strata
above the largest directly measured k, and n* for each.
"""
import argparse
import csv
import math
import time
from functools import cache
from math import comb
from pathlib import Path

import numpy as np

# Calibrated on one H100 PCIe against the fixed-k shots of run 20260920_165029 (k = 6..12, model/measured
# 0.83-1.10) and the 16 per-n means of the same run (median 1.14, 10-90% 0.71-1.38). `--price measure`
# replaces it with timings; these are the fallback so a frontier can be recomputed without a GPU.
FLOOR, A_RES, A_REP = 1.69e-3, 0.268e-9, 0.308e-9


@cache
def rso_terms(q: int, m: int, collapsed: bool = True) -> float:
    """Haar-averaged number of Glynn terms of a chain step with q placed rows over m output modes (module docstring):
    E[prod (mu+1) h(mu)] over the uniform law on q-multisets, h = 1/2 when some mu is odd (complement symmetry), else 1.
    Exact integers: sum prod (mu+1) = C(2m+q-1, q); over all-even mu (q even, q = 2h) it is
    [y^h] ((1+y) / (1-y)^2)^m = sum_i C(m, i) C(2m+h-i-1, h-i)."""
    if q <= 0:
        return 1.0
    if not collapsed:
        return 2.0 ** (q - 1)
    total = comb(2 * m + q - 1, q)
    even = 0
    if q % 2 == 0:
        h = q // 2
        even = sum(comb(m, i) * comb(2 * m + h - i - 1, h - i) for i in range(min(m, h) + 1))
    from fractions import Fraction
    return float(Fraction(total + even, 2 * comb(m + q - 1, q)))


def rso_form(s0: int, N: int) -> str:
    """The payload the CUDA backend runs at offset s0: `cuda_rect.choose_form` itself (its N-dependent mu range, set by
    PE_MU_AUTO_MAX); steps wider than the kernel's tier ladder go to the collapsed Triton sum, counted as lambda."""
    from photonic_engines.clifford import cuda_rect
    if cuda_rect._TIERS_LADDER[-1] < N:
        return "lambda"
    return cuda_rect.choose_form(N - 1 - s0, N)


def rso_features(n: int, m: int, k: int, collapse_from: int | None = 1, legacy: bool = False) -> tuple:
    """Features of one k-photon shot of the rso chain on n live sources (module docstring).

    Returns (k, sign work fock, sign work mu, sign work lambda, Laplace work); `legacy` = the uncollapsed three-feature
    model of runs <= 20261004 (k, 2^(j-2) (N^2/2 + N j), m N). collapse_from: first collapsed step in placed rows
    (1 for the CUDA backend, which collapses every step; None = never, the plain torch backend)."""
    s0 = n - k
    sw = {"fock": 0.0, "mu": 0.0, "lambda": 0.0}
    lw = 0.0
    for j in range(1, k + 1):
        N = n - k + j
        q = j - 1
        lw += m * N
        if q == 0:
            continue
        if legacy:
            sw["lambda"] += 2 ** max(j - 2, 0) * (N * N / 2 + N * j)
            continue
        T = rso_terms(q, m, collapsed=collapse_from is not None and q >= collapse_from)
        f = rso_form(s0, N)
        w = N if f == "fock" else (N * (s0 + 1) ** 2 if f == "mu" else N * N / 2 + N * j)
        sw[f] += T * w
    if legacy:
        return float(k), sw["lambda"], lw
    return float(k), sw["fock"], sw["mu"], sw["lambda"], lw


def rso_collapse_from(backend: str, collapse_from: int) -> int | None:
    """First collapsed step for a `sbs_rso.sample` backend: cuda / emul / auto collapse every step, torch none."""
    if backend in ("cuda", "emul", "auto"):
        return 1
    if backend == "torch":
        return None
    return collapse_from


def rso_model(consts, collapse_from: int | None = 1):
    legacy = len(consts) == 3
    def price(n, m, k):
        if k == 0:
            return 0.0
        x = rso_features(n, m, k, collapse_from, legacy=legacy)
        return float(sum(c * v for c, v in zip(consts, x)))
    return price


def rso_grid(ratios, kmeas: int, nmax_fit: int = 48) -> list:
    """(n, m, k) points of the fit: the k <= kmeas strata of the old grid plus s0 = n - k = 0..3 and the middle and top
    of the mu range, M // 2 and M with M = cuda_rect.mu_auto_max(n) (6 by default, 16 with the extended mu dispatch), so the
    mu constant is fitted where the mu strata are."""
    from photonic_engines.clifford import cuda_rect
    pts = set()
    for c in sorted(set(ratios)):
        for n in range(8, nmax_fit + 1, 8):
            m = max(2, round(c * n))
            for k in sorted({max(1, kmeas - 8), max(1, kmeas - 4), kmeas}):
                if k <= n:
                    pts.add((n, m, k))
            M = cuda_rect.mu_auto_max(n)
            for s0 in sorted({0, 1, 2, 3, M // 2, M}):
                if 2 <= n - s0 <= kmeas:
                    pts.add((n, m, n - s0))
    return sorted(pts)


def rso_fit(ratios, device, shots: int, kmeas: int, nmax_fit: int = 48, collapse_from: int | None = 1) -> tuple:
    """Time post-selected rso shots on `rso_grid` and fit the five constants >= 0 by least squares on the relative
    error. Returns (consts, rows) with rows = (n, m, k, seconds, model). A form with no grid point gets constant 0."""
    if device.type == "cuda" and RSO_KW.get("backend") == "cuda":
        from photonic_engines.clifford import cuda_rect
        cuda_rect.warmup_mu()                                      # NVRTC of the mu lane, all tiers, outside any timing
    measure_cost(8, 16, 6, device, 1, seed=1, algo="rso")         # first-call initialisation, not recorded
    pts = [(n, m, k, measure_cost(n, m, k, device, shots, seed=1000 * n + k, algo="rso"))
           for n, m, k in rso_grid(ratios, kmeas, nmax_fit)]
    X = np.array([rso_features(n, m, k, collapse_from) for n, m, k, _ in pts])
    y = np.array([t for *_, t in pts])
    coef = nnls_relative(X, y)
    model = X @ coef
    return tuple(float(c) for c in coef), [(n, m, k, t, float(mo)) for (n, m, k, t), mo in zip(pts, model)]


def nnls_relative(X: np.ndarray, y: np.ndarray, iters: int = 200) -> np.ndarray:
    """min ||(X c - y) / y||, c >= 0: active-set least squares (Lawson-Hanson), columns that are all zero stay 0."""
    A = X / y[:, None]
    b = np.ones_like(y)
    scale = np.where(np.abs(A).max(0) > 0, np.abs(A).max(0), 1.0)
    A = A / scale
    nz = np.abs(A).max(0) > 0
    c = np.zeros(A.shape[1])
    P: list = []
    for _ in range(iters):
        g = A.T @ (b - A @ c)
        cand = [i for i in range(A.shape[1]) if i not in P and nz[i] and g[i] > 1e-12]
        if not cand:
            break
        P.append(max(cand, key=lambda i: g[i]))
        while True:
            z = np.zeros_like(c)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                c = z
                break
            neg = [i for i in P if z[i] <= 0]
            t = min(c[i] / (c[i] - z[i]) for i in neg)
            c = c + t * (z - c)
            P = [i for i in P if c[i] > 1e-15]
    return c / scale


def photon_law(n: int, p: float) -> np.ndarray:
    """P(k), k = 0..n: exactly Binomial(n, p) for a product of partially excited sources, U-independent."""
    return np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def run_cutoff(law: np.ndarray, samples: float) -> int:
    """k_S: the smallest k with P(K > k) <= 1/samples (`gaussian.cost.run_cutoff`, same rule)."""
    idx = np.nonzero(1.0 - np.cumsum(law) <= 1.0 / samples)[0]
    return int(idx[0]) if idx.size else len(law) - 1


def model_cost(n: int, m: int, k: int) -> float:
    """Seconds for one k-photon shot, from the per-step cost model of docs/SUPERPOSITION.md §5."""
    from photonic_engines.clifford.product_sampler import cheaper
    t = k * FLOOR
    for j in range(1, k + 1):
        t += (A_RES * m * comb(n, k - j) * 2 ** j * (j + 1) if cheaper(n, k, j) == "residual"
              else A_REP * m * 4 ** j * n * j ** 3)
    return t


RSO_KW: dict = {}                     # --rso-backend / --rso-collapse-from / --rso-square / --rso-draw (sbs_rso.sample)


def measure_cost(n: int, m: int, k: int, device, shots: int, seed: int, algo: str = "replica") -> float:
    """The same quantity, timed: one post-selected k-photon shot through a Haar interferometer."""
    import torch

    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import product_sample
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    U = (q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)).to(device)
    st = ProductSuperposition.sbs(n, m)
    kw = {}
    if algo == "rso":
        from photonic_engines.clifford.sbs_rso import sample as product_sample  # same call signature here
        kw = RSO_KW
    # warm-up, not timed. rso: at the timed k itself -- kernels compile per form (s0) and per size tier, and a k = 2
    # warm-up left the first s0 = 0 shot compiling inside the timed region (1.29 s vs ~6 ms, run 20261005_220031).
    product_sample(st, U, 1, generator=g, device=device, photons=k if algo == "rso" else min(k, 2), **kw)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    product_sample(st, U, shots, generator=g, device=device, photons=k, **kw)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / shots


def _cached(cache: dict, n: int, m: int, k: int, price) -> float:
    """price(n, m, k) through the cache, evaluated only on a miss (setdefault would evaluate it eagerly)."""
    key = (n, m, k)
    if key not in cache:
        cache[key] = price(n, m, k)
    return cache[key]


def cost_table(n: int, m: int, kmax: int, price, cache: dict) -> dict:
    return {k: _cached(cache, n, m, k, price) for k in range(kmax + 1)}


def t_of_S(n: int, m: int, eta: float, alpha: float, S, price, cache: dict, kmeas: int) -> tuple:
    """(t_S, k_S, share above kmeas). Under loss the shot runs on the live sources, so E[cost|k] averages
    c(n_live, m, k) over n_live ~ Binomial(n, w1) with k | n_live ~ Binomial(n_live, sin^2 alpha')."""
    c, s = math.cos(alpha), math.sin(alpha)
    w1 = c * c + eta * s * s                                    # P(source survives as a superposition)
    s2 = (eta * s * s / w1) if w1 else 0.0                      # sin^2(alpha') of the survivor
    live = photon_law(n, w1)
    joint = np.zeros((n + 1, n + 1))                            # joint[n_live, k]
    for l in range(n + 1):
        if live[l] > 0:
            joint[l, :l + 1] = live[l] * photon_law(l, s2)
    pk = joint.sum(0)
    kS = run_cutoff(pk, S) if S else n
    tot = ext = 0.0
    for k in range(min(kS, n) + 1):
        if pk[k] <= 0:
            continue
        ec = sum(joint[l, k] * _cached(cache, l, m, k, price) for l in range(k, n + 1) if joint[l, k] > 0) / pk[k]
        tot += pk[k] * ec
        if k > kmeas:
            ext += pk[k] * ec
    return tot, kS, (ext / tot if tot else 0.0)


def n_star(ratio, eta, alpha, S, price, cache, kmeas, target, nmax) -> float:
    prev = None
    for n in range(6, nmax):
        t = t_of_S(n, max(2, round(ratio * n)), eta, alpha, S, price, cache, kmeas)[0]
        if t >= target and prev:
            n0, t0 = prev
            return n0 + (math.log(target) - math.log(t0)) / (math.log(t) - math.log(t0)) * (n - n0)
        prev = (n, t)
    return float("nan")


FINAL_FIT = Path(__file__).resolve().parents[1] / "results" / "keep" / "sbs_rso_final_run_cost_fit_r22_20261006_h100.csv"
_COST_CACHE: dict = {}
_CONSTS: dict = {}


def load_consts(fit_csv=FINAL_FIT) -> tuple:
    """The five constants (floor, a_fock, a_mu, a_lambda, b) of a `--price fit` run's *_fit.csv (same on every row),
    read once per path. The CSV does not record its backend: the constants are for the CUDA backend (collapse from
    q = 1), which is what `cost` prices; a torch-backend fit would be mispriced."""
    key = str(fit_csv)
    if key not in _CONSTS:
        with open(fit_csv, newline="") as fh:
            row = next(csv.DictReader(fh))
        _CONSTS[key] = tuple(float(row[k]) for k in ("floor", "a_fock", "a_mu", "a_lambda", "b"))
    return _CONSTS[key]


def cost(n: int, m: int, eta: float = 1.0, alpha: float = math.pi / 4, S=None, fit_csv=FINAL_FIT,
         detail: bool = False):
    """Seconds per sample of the rso SBS sampler on one H100, from the fitted cost model of the final SBS block
    (run 20261006_132352): n sources in modes 0..n-1 of an m-mode Haar
    interferometer, each in cos(alpha)|0> + sin(alpha)|1>, transmission eta. S = None: the mean over all shots; S = a run
    length: the run-of-S statistic t_S (shots with k above the 1/S tail cut dropped, module docstring). For the rate maps
    (bench/plot_rate_maps.py). detail=True returns (seconds, share of them from strata above k = 22, i.e. extrapolated by the model).

    The model is fitted on strata k <= 22 and extrapolates above (96-99.7 % of the cost at the frontier). The form
    assignment (fock / mu / lambda per stratum) is the AMBIENT `cuda_rect.choose_form` of the running process, not
    recorded with the fit: the fit file above matches the dispatch of the build of run 20261006_132352 (default since
    then); a later dispatch change moves the priced cells without an error (49.62 vs 49.50 for the previous fit under
    the two dispatches). Quote it with both caveats.
    """
    consts = load_consts(fit_csv)
    cache = _COST_CACHE.setdefault((str(fit_csv), consts), {})
    t, _, ext = t_of_S(int(n), int(m), float(eta), float(alpha), S, rso_model(consts, 1), cache, 22)
    return (float(t), float(ext)) if detail else float(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", default="1,1.5,2,2.5,3,4,6,8,12")
    ap.add_argument("--eta", default="1", help="transmissions (docs/LOSS.md §5b)")
    ap.add_argument("--alpha", type=float, default=math.pi / 4)
    ap.add_argument("--samples", default="1e4,1e6,1e8", help="run lengths S")
    ap.add_argument("--target", type=float, default=1.0, help="seconds per sample at the frontier")
    ap.add_argument("--price", default="model", choices=["model", "measure", "fit"])
    ap.add_argument("--algo", default="replica", choices=["replica", "rso"],
                    help="replica: clifford/product_sampler.py (the 20.2 row); rso: clifford/sbs_rso.py")
    ap.add_argument("--rso-consts", default="", help="--algo rso --price model: floor,a_fock,a_mu,a_lambda,b from a "
                    "previous fit (three values floor,a,b = the legacy uncollapsed model)")
    ap.add_argument("--rso-backend", default="auto", help="--algo rso: auto | torch | cuda | collapsed | triton (sbs_rso.py)")
    ap.add_argument("--rso-collapse-from", type=int, default=12, help="--algo rso: first collapsed step (placed rows)")
    ap.add_argument("--rso-square", default="auto", choices=["auto", "chain", "cuda", "triton", "kernel"],
                    help="--algo rso: strata with k = n_live (sbs_rso.sample square=; kernel = rso chain on the Fock step "
                    "kernels)")
    ap.add_argument("--rso-draw", default="host", choices=["host", "device"],
                    help="--algo rso: row draw on the host or the fused device draw (sbs_rso.sample draw=)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shots", type=int, default=1, help="--price measure: shots per stratum")
    ap.add_argument("--kmeas", type=int, default=12, help="largest k priced by a direct measurement elsewhere")
    ap.add_argument("--max-n", type=int, default=160)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    RSO_KW.update(backend=args.rso_backend, collapse_from=args.rso_collapse_from, square=args.rso_square,
                  draw=args.rso_draw)
    cf = rso_collapse_from(args.rso_backend, args.rso_collapse_from)
    cache: dict = {}
    if args.algo == "rso":
        if args.price == "measure":
            raise SystemExit("--algo rso: use --price fit (the frontier strata cannot be timed one by one) or model")
        if args.price == "fit":
            import torch
            consts, pts = rso_fit([float(x) for x in args.ratios.split(",")], torch.device(args.device),
                                  args.shots, args.kmeas, collapse_from=cf)
            print(f"rso fit: floor = {consts[0]:.3e} s/step, a_fock / a_mu / a_lambda = {consts[1]:.3e} / {consts[2]:.3e} / "
                  f"{consts[3]:.3e} s/term-op, b = {consts[4]:.3e} s/laplace-op (collapsed terms from q = {cf})")
            for n, m, k, t, mo in pts:
                print(f"  n={n:3d} m={m:4d} k={k:3d}  measured {t:.4f} s  model {mo:.4f} s  ratio {mo / t:.2f}")
            if args.csv:
                with open(args.csv.replace(".csv", "_fit.csv"), "w", newline="") as fh:
                    w = csv.writer(fh, lineterminator="\n")
                    w.writerow(["n", "m", "k", "seconds", "model", "floor", "a_fock", "a_mu", "a_lambda", "b"])
                    w.writerows([[*p, *consts] for p in pts])
        else:
            if not args.rso_consts:
                raise SystemExit("--algo rso --price model needs --rso-consts (from a --price fit run)")
            consts = tuple(float(x) for x in args.rso_consts.split(","))
            if len(consts) not in (3, 5):
                raise SystemExit("--rso-consts: five values floor,a_fock,a_mu,a_lambda,b (or the legacy three)")
        price = rso_model(consts, cf)
    elif args.price == "measure":
        import torch
        dev = torch.device(args.device)
        def price(n, m, k):
            return measure_cost(n, m, k, dev, args.shots, seed=1000 * n + k) if k else 0.0
    elif args.price == "fit":
        raise SystemExit("--price fit: --algo rso only")
    else:
        price = model_cost
    Ss = [float(x) for x in args.samples.split(",")]
    rows = []
    print(f"sbs frontier ({args.algo}), alpha = {args.alpha:.4f}, {args.target:g} s per sample, priced by {args.price}")
    print(f"{'m/n':>5} {'eta':>5} " + " ".join(f"{'S=' + f'{S:g}':>9}" for S in Ss) + f"{'untrunc':>9}")
    for eta in (float(x) for x in args.eta.split(",")):
        for c in (float(x) for x in args.ratios.split(",")):
            ns = [n_star(c, eta, args.alpha, S, price, cache, args.kmeas, args.target, args.max_n)
                  for S in [*Ss, None]]
            n0 = ns[0] if ns[0] == ns[0] else ns[-1]
            m0 = max(2, round(c * n0))
            det = [t_of_S(round(n0), m0, eta, args.alpha, S, price, cache, args.kmeas) for S in [*Ss, None]]
            rows.append({"algo": args.algo, "ratio": c, "eta": eta, "alpha": round(args.alpha, 6),
                         **{f"n_star_{S:g}": round(v, 2) for S, v in zip(Ss, ns)},
                         "n_star_untruncated": round(ns[-1], 2),
                         **{f"k_S_{S:g}": d[1] for S, d in zip(Ss, det)},
                         **{f"share_above_k{args.kmeas}_{S:g}": round(d[2], 4) for S, d in zip(Ss, det)},
                         "t_ratio_first_S": round(det[0][0] / det[-1][0], 4) if det[-1][0] else float("nan")})
            print(f"{c:>5g} {eta:>5g} " + " ".join(f"{v:>9.1f}" for v in ns[:-1]) + f"{ns[-1]:>9.1f}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
