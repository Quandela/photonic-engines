"""Kind: tool
Does: exact enumeration check of the unequal-source superposed-source chain (methods paper Sec. 3, "Unequal sources")
      against the physical output law, n <= 5.
Usage: venv/bin/python bench/sbs_unequal_check.py [--instances 5] [--seed 1] [--nmax 5] [--csv out.csv]
Writes: one line per instance (n, m, max |chain - physical| over all occupation patterns, total physical mass) and an
      optional CSV with the same columns.

Model. Source i emits a_i|0> + b_i|1> (|a_i|^2 + |b_i|^2 = 1, a_i != 0) into input mode i of an m-mode Haar U; A = U[:, :n].
Physical law: P(s) = |sum_{|S|=k} prod_{i not in S} a_i prod_{i in S} b_i per A[z, S]|^2 / s!, z any ordering of the
occupation pattern s (|s| = k).
Chain (Proposition 1 with the weighted start): k ~ Poisson-binomial(|b_i|^2); u_i = b_i / a_i, At = A diag(u); T_0 an
(n-k)-set drawn with probability prod_{i in T_0} |u_i|^-2 / W_0; at step j a uniform source x not in T_{j-1} joins,
T_j = T_{j-1} + {x}, and row r is drawn with weight |perrect At[z_<j r, T_j]|^2. The law of the chain is computed exactly by
dynamic programming over (row prefix, active set); the ordered law is summed into occupation patterns.
No sampling, no GPU: numpy and itertools only. Expected: max error at rounding (1e-15 or below).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from functools import lru_cache

import numpy as np


def per(M: np.ndarray) -> complex:
    """Permanent of a small square matrix (Ryser would do; permutations are enough at size <= 5)."""
    k = M.shape[0]
    if k == 0:
        return 1.0 + 0j
    return sum(np.prod([M[i, p[i]] for i in range(k)]) for p in itertools.permutations(range(k)))


def physical_law(A: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    m, n = A.shape
    out = {}
    for k in range(n + 1):
        for z in itertools.combinations_with_replacement(range(m), k):
            amp = 0j
            for S in itertools.combinations(range(n), k):
                w = np.prod([b[i] for i in S]) * np.prod([a[i] for i in range(n) if i not in S])
                amp += w * per(A[np.ix_(list(z), list(S))])
            occ = tuple(z.count(r) for r in range(m))
            out[occ] = abs(amp) ** 2 / math.prod(math.factorial(c) for c in occ)
    return out


def chain_law(A: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    m, n = A.shape
    u = b / a
    At = A * u[None, :]
    p1 = np.abs(b) ** 2

    @lru_cache(maxsize=None)
    def perrect(z: tuple, T: tuple) -> complex:
        j = len(z)
        return sum(per(At[np.ix_(list(z), list(S))]) for S in itertools.combinations(T, j))

    pb = np.zeros(n + 1)                                         # Poisson-binomial law of k
    pb[0] = 1.0
    for q in p1:
        pb[1:] = pb[1:] * (1 - q) + pb[:-1] * q
        pb[0] *= 1 - q
    out: dict = {}
    for k in range(n + 1):
        if pb[k] == 0:
            continue
        s0 = n - k
        starts = list(itertools.combinations(range(n), s0))
        g = np.array([np.prod([abs(u[i]) ** -2 for i in T]) for T in starts])
        state = {((), T): gT / g.sum() for T, gT in zip(starts, g)}
        for _ in range(k):
            nxt: dict = {}
            for (z, T), pr in state.items():
                free = [x for x in range(n) if x not in T]
                for x in free:
                    T2 = tuple(sorted(T + (x,)))
                    w = np.array([abs(perrect(z + (r,), T2)) ** 2 for r in range(m)])
                    if w.sum() == 0:
                        continue
                    for r in range(m):
                        if w[r] > 0:
                            key = (z + (r,), T2)
                            nxt[key] = nxt.get(key, 0.0) + pr / len(free) * w[r] / w.sum()
            state = nxt
        for (z, _T), pr in state.items():
            occ = tuple(z.count(r) for r in range(m))
            out[occ] = out.get(occ, 0.0) + pb[k] * pr
    return out


def haar(m: int, rng) -> np.ndarray:
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=5)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.instances):
        n = 3 + i % (args.nmax - 2)
        m = n + 1
        U = haar(m, rng)
        p1 = rng.uniform(0.1, 0.9, size=n)                       # unequal brightnesses |b_i|^2
        a = np.sqrt(1 - p1) * np.exp(2j * np.pi * rng.uniform(size=n))
        b = np.sqrt(p1) * np.exp(2j * np.pi * rng.uniform(size=n))
        A = U[:, :n]
        phys, ch = physical_law(A, a, b), chain_law(A, a, b)
        err = max(abs(phys.get(s, 0.0) - ch.get(s, 0.0)) for s in set(phys) | set(ch))
        row = dict(instance=i, n=n, m=m, patterns=len(phys), max_abs_err=err, phys_mass=sum(phys.values()),
                   chain_mass=sum(ch.values()))
        rows.append(row)
        print("  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"max error over {len(rows)} instances: {max(r['max_abs_err'] for r in rows):.3g}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
