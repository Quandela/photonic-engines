"""The toll of adaptivity on the sampling frontier: n*(K) at fixed m/n from the cost model, and its figure.

Cost per accepted sample of `AdaptiveCircuit.sample` (docs/ADAPTIVE.md §4b-4d, §5b), in units of one plain
Clifford & Clifford chain on the same instance:

    X(K) = C(law of R on the K adaptive modes)  +  sum over the levels beyond the tables of C(law of R on the
           modes measured before that level)

with C(law) = sum_R P(R) sum_{h=0..R} C(R, h) abort_cost(h): every started chain of a branch with r forced rows
sees h more photons on the avoided modes, the accepted one (h = 0) runs to the end and an aborted one is charged
the steps it ran (rho^k per step) up to the first of its h hits, whose position is that of the first of h
uniformly placed points among the n - r free steps (the array law is exchangeable). Summed over the branches
with the Vandermonde identity the cost depends only on the law of the TOTAL count R on the avoided modes
(`count_law_gauss`, the same Gaussian integral as E[2^r] with one more axis: E[z^R] = Per(1 + (z - 1) V^dag V)
= E_g prod_i (1 + (z - 1) |g^dag V|_i^2), g ~ CN(0, I_K), so the model has no table limit). The adaptive outcomes
are free from the exact marginal tables while the graded table fits (`graded_entries(n, K) <=
MAX_MARGINAL_ENTRIES` or --tables-upto), else one such stage per level beyond that, on the modes measured
before it (the first level is a plain chain). One mode per level is assumed (the worst case). Kept for the
record: X_bound, the earlier model with 2 / (n - rbar) per aborted chain (a bound that assumes one hit at a
uniform position), and X_noabort (every rejected chain run to the end).

n*(K) = n*(0) - log X(K) / log rho, with n*(0) and rho the plain sampler's crossing and per-photon
factor on the host (--nstar0, --rho: the job-9 fit of run 20260921_205843 by default).

    python3 bench/adaptive_toll.py --n 38 --ratio 2,3 --K 0-12 --csv results/adaptive_toll.csv
    python3 bench/adaptive_toll.py ... --measured results/keep/adaptive_crossings.csv --png img/adaptive_toll.png

--measured: a CSV with columns ratio, K, nstar, abort, chain (measured crossings, plotted as points).
  abort = 0 rows are superseded history (pre-abort; pre-batching for K = 6, 8) and are not drawn; chain = 1 marks
  a point whose adaptive outcomes were drawn by the level chain instead of the tables (m/n = 1: K = 1-3 before job
  25 re-measured them with the tables, K = 4 where the table does not fit). On the map the table-measured points
  of each K are joined, the chain-measured ones are hollow and not joined, the cost model is the dotted line.
--map: the frontier MAP instead — n*(m/n) for K = 0..--map-K (and --map-extra), anchored on the plain sampler's measured curve
    (--anchors, results/keep/frontier.csv: engine=fock rows give n*(c) and rho_eff(c)); the K > 0 curves are
    that curve shifted by -log X(K, c) / log rho(c) with X from the same cost model at n = n*(c). Written to
    --csv and drawn to --png as a fan on the frontier axes, with E[2^r] from the Gaussian estimator (the
    geometric (1 - n/m)^-K diverges at m/n -> 1; the real value does not).
The figure: Delta n* = n*(K) - n*(0) against K (bottom axis) and against rbar (top axis, ratio 2), one
curve per m/n; the two curves collapse when read against rbar, which is the point of the figure.
"""
import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from helpers import haar_unitary

from photonic_engines.adaptive import MAX_MARGINAL_ENTRIES, binomial_moments, graded_entries

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=38)
ap.add_argument("--ratio", default="2,3")
ap.add_argument("--K", default="0-12")
ap.add_argument("--nstar0", type=float, default=38.5, help="plain sampler crossing at 1 sample/s, ratio 2 (job 9)")
ap.add_argument("--nstar0-ratio", default="2:38.5,3:36.9", help="per-ratio crossings; falls back to --nstar0")
ap.add_argument("--rho", type=float, default=1.82, help="per-photon cost factor of the plain sampler (job 9 fit)")
ap.add_argument("--gauss-samples", type=int, default=100000)
ap.add_argument("--seeds", type=int, default=4, help="Haar unitaries averaged per point")
ap.add_argument("--csv", default="results/adaptive_toll.csv")
ap.add_argument("--measured", default="")
ap.add_argument("--png", default="")
ap.add_argument("--map", action="store_true")
ap.add_argument("--map-K", type=int, default=4)
ap.add_argument("--map-extra", default="", help="further K values on the map, e.g. 8 (drawn like the others)")
ap.add_argument("--anchors", default="results/keep/frontier.csv")
ap.add_argument("--tables-upto", type=int, default=0,
                help="assume the exact marginal tables reach this K whatever the memory guard (0: use the guard)")
a = ap.parse_args()


def parse_K(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def e2r_gauss(V, samples, gen):
    """E[2^r] = E_g prod_i (1 + |g^dag V|_i^2) and its standard error, g ~ CN(0, I_K)."""
    K, n = V.shape
    g = (torch.randn(samples, K, generator=gen, dtype=torch.float64)
         + 1j * torch.randn(samples, K, generator=gen, dtype=torch.float64)) / math.sqrt(2)
    f = torch.log1p((g.conj() @ V).abs() ** 2).sum(1).exp()
    return float(f.mean()), float(f.std() / math.sqrt(samples))


def count_law_gauss(V, samples, gen):
    """P(R = rho), rho = 0..n, R the total photon count on the K modes of V [K, n]: float64 [n + 1].

    E[z^R] = Per(1 + (z - 1) V^dag V) = E_g prod_i (1 + (z - 1) w_i), w_i = |g^dag V|_i^2, g ~ CN(0, I_K) (Wick:
    E_g prod_{i in S} w_i = Per((V^dag V)[S, S]), and Per(1 + A) = sum_S Per(A[S, S])). Per sample of g the
    product is the generating function of independent Bernoulli(w_i) - a signed one where w_i > 1 - so its
    coefficients come from n convolutions; the average over g is the exact law. Same estimator as `e2r_gauss`
    (sum_rho 2^rho P(rho) = E[2^r]), one more axis.
    """
    K, n = V.shape
    g = (torch.randn(samples, K, generator=gen, dtype=torch.float64)
         + 1j * torch.randn(samples, K, generator=gen, dtype=torch.float64)) / math.sqrt(2)
    w = (g.conj() @ V).abs() ** 2                                                   # [samples, n]
    c = torch.zeros(samples, n + 1, dtype=torch.float64); c[:, 0] = 1.0
    nxt = torch.empty_like(c)
    for i in range(n):                                                              # times ((1 - w_i) + w_i z), degree i -> i + 1
        wi = w[:, i:i + 1]
        torch.mul(c[:, :i + 1], 1 - wi, out=nxt[:, :i + 1]); nxt[:, i + 1] = 0.0    # degrees 0..i valid in c
        nxt[:, 1:i + 2].addcmul_(c[:, :i + 1], wi)
        c, nxt = nxt, c
    return c.mean(0)


def abort_cost(n, r, N, h, rho):
    """Expected cost, in units of one full plain chain, of a chain with r forced rows that is aborted at the
    first of h photons landing on the avoided modes among its N = n - r free steps.

    Step k of a chain costs ~ rho^k (rho the plain sampler's per-photon factor, the growth of its time with n),
    so a chain that stops after step r + J costs (rho^(r + J) - rho^r) / (rho^n - 1) of a full one. The array
    law is exchangeable, so the h hits sit at h uniformly random distinct positions among the N free steps and
    the first one is at J with probability C(N - J, h - 1) / C(N, h). h = 0 is the accepted chain, run to the end.
    """
    if h == 0:
        return (rho ** n - rho ** r) / (rho ** n - 1.0)
    tot = math.comb(N, h)
    return sum(math.comb(N - J, h - 1) / tot * (rho ** (r + J) - rho ** r) / (rho ** n - 1.0) for J in range(1, N - h + 2))


def chain_cost(law, n, rho):
    """Expected cost of one accepted sample of the forced-prefix stage, in full plain chains, from the law of the
    total count R on the avoided modes (`count_law_gauss`): the accepted chain plus the aborted ones.

    A branch p (|p| = r) starts T[p] chains per sample, and a fraction sum_{|q| = r + h} prod_l C(q_l, p_l) P(q) / T[p]
    of them see h more photons on the avoided modes; summed over p with the Vandermonde identity
    sum_{p <= q, |p| = |q| - h} prod_l C(q_l, p_l) = C(|q|, h), the cost is
        sum_rho P(R = rho) sum_{h = 0..rho} C(rho, h) abort_cost(n, rho - h, n - rho + h, h, rho_eff),
    the h = 0 term being the accepted chains (sum_rho P(rho) = 1 of them). With abort_cost = 1 for h >= 1 this is
    E[2^R] - the historical no-abort model - and with abort_cost = 2 / (n - rbar) the earlier bound.
    """
    total = 0.0
    for R in range(len(law)):
        pR = float(law[R])
        if pR <= 1e-15:
            continue
        total += pR * sum(math.comb(R, h) * abort_cost(n, R - h, n - R + h, h, rho) for h in range(R + 1))
    return total


def toll_row(n, ratio, K, base, rho, exact_upto=3):
    """One point of the cost model: dict with E[2^r], the overheads X with / without the abort, and n*."""
    m, t = round(ratio * n), list(range(n))
    e2r, err, e2r_exact = 1.0, 0.0, None
    if K:
        vals = []
        for s in range(a.seeds):
            V = haar_unitary(m, 1000 * n + 10 * s + K)[m - K:][:, t]
            vals.append(e2r_gauss(V, a.gauss_samples, torch.Generator().manual_seed(s)))
        e2r = sum(v for v, _ in vals) / a.seeds
        err = math.sqrt(sum(e ** 2 for _, e in vals)) / a.seeds
        if exact_upto >= K:
            e2r_exact = sum(float(binomial_moments(haar_unitary(m, 1000 * n + 10 * s + K), t, list(range(m - K, m))).sum())
                            for s in range(a.seeds)) / a.seeds
    rbar = n * K / m
    abort = 2.0 / (n - rbar)
    fits = lambda k: k <= a.tables_upto or graded_entries(n, k) <= MAX_MARGINAL_ENTRIES
    tables = fits(K)
    levels = 0 if tables else K - max(k for k in range(K + 1) if fits(k))
    # The first-hit model (docs/ADAPTIVE.md §5b, 2026-09-22): the law of the count on the avoided modes, from the
    # same Gaussian integral, gives the number of extra hits of every started chain, and exchangeability the
    # position of the first one; each aborted chain is charged the steps it ran at rho^k per step. Averaged over
    # --seeds unitaries. The earlier bound 2 / (n - rbar) per aborted chain is kept as X_bound.
    laws = {}                                                                       # j modes avoided -> law of the count, averaged
    for j in ([K, *range(K - levels, K)] if K else []):
        if j == 0:
            continue
        acc = torch.zeros(n + 1, dtype=torch.float64)
        for s in range(a.seeds):
            V = haar_unitary(m, 1000 * n + 10 * s + K)[m - K:m - K + j][:, t]
            acc += count_law_gauss(V, a.gauss_samples, torch.Generator().manual_seed(100 + s))
        laws[j] = acc / a.seeds
    X_final = chain_cost(laws[K], n, rho) if K else 1.0
    X_final_bound = 1.0 + (e2r - 1.0) * abort
    # levels beyond the tables: level lambda forces p_{<lambda} and aborts on the first lambda - 1 modes, so it is the
    # forced-prefix stage of the (lambda - 1)-mode instance: same cost function on the law of that count (lambda = 1:
    # a plain chain, cost 1). The bound version interpolated E[2^{r_lambda}] ~ e2r^(lambda / K); job 17 (run
    # 20260922_071308) measured 92-113 / 234-351 chains STARTED per sample at K = 6 / 8, n = 26..34.
    X_levels = sum((chain_cost(laws[lam - 1], n, rho) if lam > 1 else 1.0) for lam in range(K - levels + 1, K + 1)) if levels else 0.0
    X_levels_bound = sum(1.0 + (e2r ** (lam / K) - 1.0) * abort for lam in range(K - levels + 1, K + 1)) if levels else 0.0
    X = X_final + X_levels
    X_bound = X_final_bound + X_levels_bound
    X_noabort = e2r + levels * e2r                                                  # every chain run to the end (historical)
    return {"n": n, "ratio": ratio, "K": K, "rbar": round(rbar, 3), "E2r": round(e2r, 3), "E2r_err": round(err, 3),
            "E2r_exact": "" if e2r_exact is None else round(e2r_exact, 3), "tables": int(tables), "levels_by_chain": levels,
            "X_final": round(X_final, 3), "X": round(X, 3), "X_bound": round(X_bound, 3), "X_noabort": round(X_noabort, 3),
            "dn": round(-math.log(X) / math.log(rho), 2), "dn_bound": round(-math.log(X_bound) / math.log(rho), 2),
            "dn_noabort": round(-math.log(X_noabort) / math.log(rho), 2),
            "nstar": round(base - math.log(X) / math.log(rho), 2), "nstar_bound": round(base - math.log(X_bound) / math.log(rho), 2),
            "nstar_noabort": round(base - math.log(X_noabort) / math.log(rho), 2)}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if a.map:
    anchors = {}
    with open(a.anchors) as f:
        for r in csv.DictReader(f):
            if r["engine"] == "fock":
                anchors[float(r["ratio"])] = (float(r["n_star"]), float(r["rho_eff"]))
    rows = []
    map_Ks = list(range(a.map_K + 1)) + [int(v) for v in a.map_extra.split(",") if v]
    for ratio in sorted(anchors):
        base, rho = anchors[ratio]
        n = round(base)
        for K in map_Ks:
            rows.append(toll_row(n, ratio, K, base, rho, exact_upto=0)); print(rows[-1])
    write_csv(a.csv, rows)
    if a.png:
        # the paper's figure style (F1-F9)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "papers", "fig", "plots"))
        from pestyle import INK, MUTED, setup
        setup()
        import matplotlib.pyplot as plt
        measured = []
        if a.measured and os.path.exists(a.measured):
            with open(a.measured) as f:
                measured = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        ramp = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 8: "#104281"}   # K is ordinal: one blue ramp (L)
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for K in map_Ks:
            pts = [r for r in rows if r["K"] == K]
            c = INK if K == 0 else ramp.get(K, MUTED)
            xs = [r["ratio"] for r in pts]
            if K == 0:
                ax.plot(xs, [r["nstar"] for r in pts], color=c, linewidth=2.0, label="$K = 0$ (no feed-forward, measured)")
            else:                                                               # the cost model, dotted
                ax.plot(xs, [r["nstar"] for r in pts], color=c, linewidth=0.8, linestyle=(0, (1, 2)))
            mine = [r for r in measured if int(r["K"]) == K and K and r.get("abort", 1.0) == 1.0]
            meas = sorted([(r["ratio"], r["nstar"]) for r in mine if r.get("chain", 0.0) != 1.0])
            if meas:                                                            # measured with the tables, joined
                ax.plot([x for x, _ in meas], [y for _, y in meas], color=c, linewidth=1.6, linestyle="-", marker="o",
                        markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, label=f"$K = {K}$ (measured)")
            hollow = [r for r in mine if r.get("chain", 0.0) == 1.0]           # adaptive outcomes by the level chain: hollow, not joined
            for i, r in enumerate(hollow):
                ax.plot(r["ratio"], r["nstar"], marker="o", markersize=4.5, color=c, markerfacecolor="white",
                        markeredgecolor=c, markeredgewidth=1.2, linestyle="none",
                        label=f"$K = {K}$ (measured, abandoned chains)" if i == 0 and not meas else None)
        ax.set_xscale("log"); ax.set_xticks([1, 1.5, 2, 3, 4, 6, 8, 12]); ax.set_xticklabels(["1", "1.5", "2", "3", "4", "6", "8", "12"])
        ax.minorticks_off()
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 1.5, hi)                       # room for the two-line footnote below the data
        ax.set_xlabel("modes per photonic input $m/n$"); ax.set_ylabel("$n^*$ (photonic inputs) at 1 sample/s")
        ax.text(0.99, 0.02, "solid: measured (H100), outcome tables; hollow: abandoned chains; dotted: cost model\n"
                "$m/n = 2$: 512 shots per point; other $m/n$: 64–128 shots (a few tenths either way)",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
        ax.legend(loc="upper right", handlelength=2.4)
        fig.savefig(a.png, dpi=300); print("figure:", a.png)
    sys.exit(0)

nstar0 = dict(item.split(":") for item in a.nstar0_ratio.split(","))
nstar0 = {float(k): float(v) for k, v in nstar0.items()}
rows = []
for ratio in [float(x) for x in a.ratio.split(",")]:
    for K in parse_K(a.K):
        rows.append(toll_row(a.n, ratio, K, nstar0.get(ratio, a.nstar0), a.rho)); print(rows[-1])
write_csv(a.csv, rows)

if a.png:
    # the paper's figure style (F1-F9)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "papers", "fig", "plots"))
    from pestyle import FAINT, INK, MUTED, setup
    setup()
    import matplotlib.pyplot as plt
    measured = []
    if a.measured and os.path.exists(a.measured):
        with open(a.measured) as f:
            measured = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    colors = {2.0: INK, 3.0: MUTED}                                             # ink / muted grey, not input hues (L)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for ratio in sorted({r["ratio"] for r in rows}):
        pts = [r for r in rows if r["ratio"] == ratio]
        c = colors.get(ratio, MUTED)
        ax.plot([r["K"] for r in pts], [r["dn"] for r in pts], color=c, linewidth=1.6, label=f"$m = {ratio:g}n$, cost model")
        kt = [r["K"] for r in pts if r["tables"]]
        if kt and len(kt) < len(pts):
            ax.axvspan(max(kt) + 0.5, max(r["K"] for r in pts) + 0.5, color=FAINT, alpha=0.25, lw=0)
        for r in measured:
            if r["ratio"] == ratio and r.get("abort", 1.0) == 1.0:
                base = nstar0.get(ratio, a.nstar0)
                ax.plot(r["K"], r["nstar"] - base, marker="o", markersize=5, color=c, markeredgecolor="white", markeredgewidth=0.8)
    kmax = max(r["K"] for r in rows)
    if kmax > max(r["K"] for r in rows if r["tables"]):
        ax.text(max(r["K"] for r in rows if r["tables"]) + 0.6, ax.get_ylim()[1] * 0.97, "outcome drawn by abandoned chains",
                fontsize=7.5, color=MUTED, va="top")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 1.4, hi)                           # room for the two-line footnote below the data
    ax.set_xlabel("adaptive modes $K$"); ax.set_ylabel(r"$\Delta n^*$ (photonic inputs) at 1 sample/s")
    top = ax.secondary_xaxis("top", functions=(lambda K: K * a.n / (2 * a.n), lambda r: r * 2))
    top.set_xlabel(r"photons read adaptively $\bar r = nK/m$ ($m = 2n$)", fontsize=8)
    ax.legend(loc="upper right")
    if measured:
        ax.text(0.99, 0.02, f"points: measured crossings (H100); $m = 2n$: 512 shots, $m = 3n$: 64–128 shots\n"
                f"cost model at $n = {a.n}$, $\\beta = {a.rho}$", transform=ax.transAxes, ha="right", fontsize=7.5,
                color=MUTED)
    fig.savefig(a.png, dpi=300)
    print("figure:", a.png)
