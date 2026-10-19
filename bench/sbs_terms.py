"""
How many pure terms an exact chain-rule sampler for superposition boson sampling would carry.

This is the GATE of docs/SUPERPOSITION.md §5: the emitter sampler's cost is R_k terms x 2^k per
step (docs/EMITTER_SAMPLER.md §4), and the question is what R_k becomes when the input is the
post-selected SBS state instead of a caterpillar.

The measurement does not need a sampler. Post-selected on k photons the SBS input is the Dicke
state on the n source modes, so the marginal of the first j placed rows r is the parked
superposition chain rule (docs/SUPERPOSITION.md §5) with a uniform superposition:

    p(r) = sum_{R : |R| = k - j}  | sum_{S subset [n]\\R, |S| = j} Perm(U[r, S]) |^2
         = a^dag G a ,   a_S = Perm(U[r, S]) ,   G[S, S'] = C(n - |S union S'|, k - j)

(swap the two sums; the count of residual sets R disjoint from S union S' depends only on
|S union S'|). G is PSD - it is sum_R v_R v_R^T with v_R[S] = 1 when S misses R - so
p(r) = sum_t lambda_t |<u_t, a>|^2 over the eigenpairs of G, and

    the number of pure terms the sampler must carry at step j is exactly rank(G) ,

independent of U and of any particular algorithm. G lives in the Johnson association scheme
(it depends on S, S' only through |S cap S'|), which is why its rank is a clean function of n, k, j.

    python3 bench/sbs_terms.py --n 12 --k 6
    python3 bench/sbs_terms.py --check                 # rank == C(n, min(j, k-j)) over all n <= 12
    python3 bench/sbs_terms.py --n 10,12,14 --alpha-quarter --csv results/sbs_terms.csv

Reports per step j: the rank, the eigenvalue-mass profile (how many terms hold 99% / 99.9% of
tr(G) - a proxy for what an APPROXIMATE sampler could drop, not a bound on its TVD), and the
implied per-shot cost sum_j R_j 2^j j against the Fock sampler's sum_j 2^j j at the same photon
count.
"""
import argparse
import csv
import math
from itertools import combinations

import numpy as np


def gram(n: int, k: int, j: int) -> np.ndarray:
    """G[S, S'] = C(n - |S union S'|, k - j) over the C(n, j) j-subsets of [n] (see the header)."""
    subsets = [set(s) for s in combinations(range(n), j)]
    g = np.zeros((len(subsets), len(subsets)))
    for a, sa in enumerate(subsets):
        for b, sb in enumerate(subsets):
            free = n - len(sa | sb)
            g[a, b] = math.comb(free, k - j) if free >= k - j else 0.0
    return g


def rank_and_mass(g: np.ndarray, tol: float = 1e-9) -> tuple[int, int, int]:
    """(rank, terms holding 99% of tr(G), terms holding 99.9%) from the eigenvalues of G."""
    w = np.sort(np.linalg.eigvalsh(g))[::-1]
    keep = w[w > tol * max(1.0, float(abs(w).max()))]
    if keep.size == 0:
        return 0, 0, 0
    cum = np.cumsum(keep) / keep.sum()
    return int(keep.size), int(np.searchsorted(cum, 0.99)) + 1, int(np.searchsorted(cum, 0.999)) + 1


def predicted_rank(n: int, k: int, j: int) -> int:
    """C(n, min(j, k - j)) — the closed form the `--check` sweep confirms for every n <= 12."""
    return math.comb(n, min(j, k - j))


def check(limit: int = 12, max_dim: int = 500) -> int:
    """Verify rank(G) == C(n, min(j, k-j)) for every (n <= limit, k, j) whose G fits `max_dim`."""
    bad = 0
    for n in range(1, limit + 1):
        for k in range(n + 1):
            for j in range(k + 1):
                if math.comb(n, j) > max_dim:
                    continue
                r, _, _ = rank_and_mass(gram(n, k, j))
                if r != predicted_rank(n, k, j):
                    bad += 1
                    print(f"  MISMATCH n={n} k={k} j={j}: rank {r} != {predicted_rank(n, k, j)}")
    print(f"rank == C(n, min(j, k-j)) on every case tested up to n = {limit}: "
          f"{'CONFIRMED' if bad == 0 else f'{bad} MISMATCHES'}")
    return bad


def report(n: int, k: int, max_dim: int, writer=None) -> None:
    """Per-step ranks for one (n, k), measured while G fits and extrapolated by the formula after."""
    print(f"\nn = {n} sources, k = {k} detected photons   (Dicke bond min(k, n-k)+1 = {min(k, n - k) + 1})")
    print("   j   C(n,j)    rank  99%  99.9%  source")
    cost_sbs = cost_fock = 0.0
    for j in range(k + 1):
        dim = math.comb(n, j)
        if dim <= max_dim:
            r, m99, m999 = rank_and_mass(gram(n, k, j))
            src = "measured"
        else:
            r, m99, m999 = predicted_rank(n, k, j), -1, -1
            src = "formula"
        step = 2.0 ** j * max(j, 1)
        cost_sbs += r * step
        cost_fock += step
        print(f"{j:4d} {dim:8d} {r:7d} {m99:4d} {m999:6d}  {src}")
    print(f"  peak terms  R_max = {max(predicted_rank(n, k, j) for j in range(k + 1)):,} "
          f"at j = {k // 2}  (= C(n, k/2))")
    print(f"  per-shot cost sum_j R_j 2^j j = {cost_sbs:.3e}, Fock sampler at k photons "
          f"{cost_fock:.3e}, ratio {cost_sbs / cost_fock:.3e}")
    if writer is not None:
        for j in range(k + 1):
            writer.writerow([n, k, j, math.comb(n, j), predicted_rank(n, k, j)])


def modelled_cost(n: int, k: int) -> float:
    """sum_j R_j 2^j j with R_j = C(n, min(j, k-j)) — the per-shot work of the exact chain rule."""
    return sum(predicted_rank(n, k, j) * 2.0 ** j * max(j, 1) for j in range(k + 1))


def fock_cost(k: int) -> float:
    """The same model for a Fock input of k photons: one term per step, sum_j 2^j j."""
    return sum(2.0 ** j * max(j, 1) for j in range(k + 1))


def frontier_projection(anchor: float) -> None:
    """Where the modelled cost puts SBS against a Fock sampler whose n* is `anchor`.

    A PROJECTION, not a measurement: it assumes one SBS term costs what one Fock lane-step costs
    and that nothing beyond the chain rule is used. The real n* needs the sampler (there is none).
    """
    budget = fock_cost(round(anchor))
    best = 0
    for n in range(2, 200, 2):
        if modelled_cost(n, n // 2) <= budget:
            best = n
    print(f"\nprojection at alpha = pi/4 (k = n/2), Fock anchor n* = {anchor}:")
    print(f"  SBS reaches {best} sources = {best // 2} detected photons at the same per-shot cost")
    print(f"  i.e. {anchor - best // 2:.1f} photons behind the Fock sampler in DETECTED photons")
    print("  (cost model only — one term assumed as dear as one Fock lane-step)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="12", help="comma-separated source counts")
    ap.add_argument("--k", default="", help="detected photons (default: n/2, the alpha = pi/4 mean)")
    ap.add_argument("--alpha-quarter", action="store_true", help="k = n/2 for every n (the default anyway)")
    ap.add_argument("--max-dim", type=int, default=500, help="largest C(n, j) to diagonalise")
    ap.add_argument("--check", action="store_true", help="verify the closed form instead of reporting")
    ap.add_argument("--csv", default="", help="write the per-step ranks here")
    ap.add_argument("--frontier", type=float, default=0.0,
                    help="project the frontier against a Fock sampler with this n* (e.g. 38.6)")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(1 if check() else 0)
    ns = [int(x) for x in args.n.split(",")]
    ks = [int(x) for x in args.k.split(",")] if args.k else [None] * len(ns)
    fh = open(args.csv, "w", newline="") if args.csv else None
    writer = csv.writer(fh) if fh else None
    if writer:
        writer.writerow(["sources", "photons", "step", "subsets", "terms"])
    for n, k in zip(ns, ks):
        report(n, n // 2 if k is None else k, args.max_dim, writer)
    if args.frontier:
        frontier_projection(args.frontier)
    if fh:
        fh.close()
        print(f"\ncsv: {args.csv}")


if __name__ == "__main__":
    main()
