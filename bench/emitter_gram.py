"""
How much of the emitter sampler's term count R_k is redundant — measured exactly.

docs/EMITTER_SAMPLER.md §7: the gap-Kraus construction produces
R_k = r_0 · r_f · ∏ r_gap pure terms ψ_e with ρ_T = Σ_e |ψ_e⟩⟨ψ_e|, but that is only an UPPER
BOUND on rank(ρ_T) — each gap is reduced minimally on its own, correlations between gaps are
not exploited.  The surplus terms are linearly dependent on the others while no single chain
has zero norm, so the existing pruning cannot see them.  Since the per-step cost is linear in
R_k, the gap between R_k and rank(ρ_T) is headroom on every emitter backend (torch, segments,
Triton) alike.

§7 names the measurement: the Gram matrix ⟨ψ_e|ψ_e'⟩ is a χ²×χ² transfer-matrix product of
length k, O(R² k χ⁴) per step — negligible against R_k 2^k k — and rank(ρ_T) = rank of that
Gram matrix, because ρ_T = Σ_e |ψ_e⟩⟨ψ_e| has exactly the rank of the family {ψ_e}.  "That
measurement comes before any further kernel work."  This is it.

    python3 bench/emitter_gram.py --n 12 --samples 8
    python3 bench/emitter_gram.py --n 16 --family random --samples 4 --csv gram16.csv
    python3 bench/emitter_gram.py --n 8 --check-rho          # validate against explicit rho_T

The Gram sweep is checked to 5e-16 against psi built term by term over all L^k label strings,
and its rank against the eigenvalues of an explicitly built rho_T (including a deliberately
rank-deficient family), for R = 3..7, k = 1..5, L = 2..3.

Reports, per step k and cost-weighted over the run:
  R_k          terms produced by the gap-Kraus construction (what the sampler pays for)
  rank         numerical rank of the Gram matrix = rank(rho_T) (what it could pay for)
  R/rank       the headroom at that step
  sharpness    smallest kept eigenvalue / largest dropped one -- how clean the cut is, i.e.
               whether "rank" is a well-defined integer or a soft tail (inf = exact rank)
and the bottom line: Σ_k R_k c_k / Σ_k rank_k c_k with c_k = 2^{k-2} k the per-term step cost
(docs/EMITTER_SAMPLER.md §4) -- the factor a perfect decomposition would buy end to end.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict

import numpy as np
import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_reference import gap_kraus_terms

C128 = torch.complex128


def build_state(family, n, seed):
    """linear = linear cluster (theta = pi/2, phi = 0); ghz = phi = pi; random = Haar-ish
    caterpillar gates (the family the rank review measured)."""
    rng = np.random.default_rng(seed)
    if family == "linear":
        gates = [spin_gate(math.pi / 2, 0.0)] * n
    elif family == "ghz":
        gates = [spin_gate(math.pi / 2, math.pi)] * n
    elif family == "random":
        gates = [spin_gate(rng.uniform(0.2, math.pi - 0.2), rng.uniform(0, 2 * math.pi))
                 for _ in range(n)]
    else:
        raise SystemExit(f"unknown family {family}")
    return EmitterMPS.caterpillar(n, gates, m=2 * n)


def gram(terms, chunk=256):
    """G[a, e] = <psi_a | psi_e> for psi_e(x) = <f_e| M_{k-1,x} ... M_{0,x} |s0_e>.

    Left-to-right sweep carrying C[a, e, s, v], the (ket, bra) environment of a pair: index a
    is the conjugated (bra) term, e the ket term, s the ket bond and v the bra bond.  One
    position costs R^2 L chi^4, so the whole matrix is O(R^2 k L chi^4) -- the chi^2 x chi^2
    transfer-matrix product of docs/EMITTER_SAMPLER.md §7, batched over pairs.  Chunked over a
    so the [a, e, l, t, v] intermediate stays bounded for large R.
    """
    s0 = torch.stack([t[0] for t in terms]).to(C128)                  # [R, chi]
    mats = torch.stack([t[1] for t in terms]).to(C128)                # [R, k, L, chi, chi]
    f = torch.stack([t[2] for t in terms]).to(C128)                   # [R, chi]
    R, k, L, chi, _ = mats.shape
    G = torch.empty((R, R), dtype=C128)
    for a0 in range(0, R, chunk):
        a1 = min(a0 + chunk, R)
        # C[a, e, s, v] = s0_e[s] * conj(s0_a)[v]
        C = s0[None, :, :, None] * s0.conj()[a0:a1, None, None, :]
        for j in range(k):
            Me = mats[:, j]                                            # [R(e), L, chi, chi]
            Ma = mats[a0:a1, j].conj()                                 # [R(a), L, chi, chi]
            # sum_s M_e[l, t, s] C[s, v]
            tmp = torch.einsum("elts,aesv->aeltv", Me, C)
            # sum_{l, v} tmp[l, t, v] conj(M_a)[l, u, v]
            C = torch.einsum("aeltv,aluv->aetu", tmp, Ma)
        # G[a, e] = sum_{t, u} conj(f_e)[t] f_a[u] C[a, e, t, u]
        G[a0:a1] = torch.einsum("et,au,aetu->ae", f.conj(), f[a0:a1], C)
    return G


def rank_of(G, tol=1e-10):
    """Numerical rank of a Hermitian PSD Gram matrix, plus how sharp the cut is."""
    w = torch.linalg.eigvalsh(G.to(torch.complex128)).real.clamp_min(0.0)
    w, _ = torch.sort(w, descending=True)
    top = float(w[0]) if len(w) else 0.0
    if top <= 0:
        return 0, float("nan"), w
    keep = int((w > tol * top).sum())
    lo = float(w[keep - 1]) / top if keep >= 1 else float("nan")       # smallest kept
    hi = float(w[keep]) / top if keep < len(w) else 0.0                # largest dropped
    return keep, (lo / hi if hi > 0 else float("inf")), w


def rho_rank_explicit(terms, tol=1e-10):
    """Independent check for small k: build rho_T = sum_e |psi_e><psi_e| on the full L^k label
    space and take its rank.  Must equal the Gram rank."""
    s0 = torch.stack([t[0] for t in terms]).to(C128)
    mats = torch.stack([t[1] for t in terms]).to(C128)
    f = torch.stack([t[2] for t in terms]).to(C128)
    R, k, L, chi, _ = mats.shape
    psi = torch.zeros((R, L ** k), dtype=C128)
    for idx in range(L ** k):
        xs, r = [], idx
        for _ in range(k):
            xs.append(r % L); r //= L
        v = s0.clone()                                                 # [R, chi]
        for j in range(k):
            v = torch.einsum("rts,rs->rt", mats[:, j, xs[j]], v)
        psi[:, idx] = (f.conj() * v).sum(1)
    rho = psi.conj().T @ psi
    w = torch.linalg.eigvalsh(rho).real.clamp_min(0)
    return int((w > tol * w.max()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--family", default="linear", choices=["linear", "ghz", "random"])
    ap.add_argument("--samples", type=int, default=8, help="random photon orders sigma")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-rho", action="store_true",
                    help="also build rho_T explicitly (k <= 10) and compare its rank")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    state = build_state(a.family, a.n, a.seed)
    final = None if state.final is not None else torch.eye(state.chi, dtype=C128)[0]
    rng = np.random.default_rng(a.seed)

    per_k = defaultdict(lambda: {"R": [], "rank": [], "sharp": []})
    rows, cost_R, cost_rank = [], 0.0, 0.0
    worst = (0.0, None)
    for rep in range(a.samples):
        sigma = rng.permutation(a.n)
        for k in range(1, a.n + 1):
            T = sorted(int(x) for x in sigma[:k])
            kept, terms = gap_kraus_terms(state, T, final=final)
            R = len(terms)
            if R == 0:
                continue
            G = gram(terms)
            rk, sharp, w = rank_of(G, a.tol)
            if a.check_rho and k <= 10 and rep == 0:
                rk2 = rho_rank_explicit(terms, a.tol)
                assert rk == rk2, f"Gram rank {rk} != explicit rho_T rank {rk2} at k={k}"
            c = (2.0 ** max(k - 2, 0)) * k                              # docs/EMITTER_SAMPLER.md §4
            cost_R += R * c
            cost_rank += rk * c
            per_k[k]["R"].append(R); per_k[k]["rank"].append(rk); per_k[k]["sharp"].append(sharp)
            if R / max(rk, 1) > worst[0]:
                worst = (R / max(rk, 1), (rep, k, T, R, rk))
            rows.append({"rep": rep, "k": k, "T": " ".join(map(str, T)), "R": R, "rank": rk,
                             "ratio": R / max(rk, 1), "sharpness": sharp, "cost": c})
    if a.check_rho:
        print("explicit rho_T rank check: OK (k <= 10, first sample)\n")

    print(f"{a.family} caterpillar, n = {a.n}, {a.samples} photon orders, "
          f"final = {'projected' if state.final is not None else 'basis 0'}, tol = {a.tol:g}")
    print(f"{'k':>3} {'R_k':>8} {'rank':>8} {'R/rank':>7} {'sharpness':>10} {'cost share':>11}")
    tot = sum((2.0 ** max(k - 2, 0)) * k * statistics.fmean(v["R"]) for k, v in per_k.items())
    for k in sorted(per_k):
        v = per_k[k]
        mR, mr = statistics.fmean(v["R"]), statistics.fmean(v["rank"])
        sh = [s for s in v["sharp"] if math.isfinite(s)]
        share = (2.0 ** max(k - 2, 0)) * k * mR / tot
        print(f"{k:>3} {mR:>8.2f} {mr:>8.2f} {mR / max(mr, 1e-9):>7.2f} "
              f"{(statistics.median(sh) if sh else float('inf')):>10.1e} {100 * share:>10.1f}%")
    print(f"\ncost-weighted headroom over the run: {cost_R / max(cost_rank, 1e-9):.3f}x "
          f"(sum R_k c_k / sum rank_k c_k, c_k = 2^(k-2) k)")
    if worst[1]:
        rep, k, T, R, rk = worst[1]
        print(f"worst single step: k = {k}, R = {R}, rank = {rk} ({worst[0]:.1f}x), T = {T}")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w_.writeheader(); w_.writerows(rows)
        print(f"wrote {a.csv} ({len(rows)} steps)")


if __name__ == "__main__":
    main()
