"""Memory frontier of strong simulation: largest n whose full output distribution fits in a
byte budget, per mode ratio c = m/n.  Pure counting - no benchmark.

    python bench/memory_frontier.py [GiB] [figure.png]
"""
import sys
from math import comb, log

B = float(sys.argv[1]) * 2**30 if len(sys.argv) > 1 else 80 * 2**30
FIG = sys.argv[2] if len(sys.argv) > 2 else None
A = 8                                   # complex64 amplitude; probability = float32 = 4 bytes

def M(m, k):  return comb(m + k - 1, k) if k >= 0 else 0      # k photons in m modes
def Mcf(m, k): return comb(m, k) if k >= 0 else 0             # collision-free

def pull_peak(size, n, chi=1):
    """implicit kernel, last layer pulled chunk-wise straight into float32 p (emitter traced)."""
    inter = max((chi * A * (size(k - 1) + size(k)) for k in range(1, n)), default=0)
    return max(inter, chi * A * size(n - 1) + 4 * size(n))

GBS_MEAN, GBS_EPS = 0.5, 1e-3           # gbs: sinh^2 r per squeezer, uncaptured probability mass


def gbs_cutoff(n, mean=GBS_MEAN, eps=GBS_EPS):
    """Smallest N_cut with P(N > N_cut) <= eps for n identical squeezed vacua (lossless: N even)."""
    import numpy as np

    from photonic_engines import GaussianInput
    r = GaussianInput.r_for_mean(mean)
    top = 8 * n + 40
    while True:
        p = GaussianInput.smsv(n, n, r).photon_number_probs(top)
        tail = 1 - np.cumsum(p)
        idx = np.nonzero(tail <= eps)[0]
        if idx.size:
            return int(idx[0])
        top *= 2


def models(n, c):
    m = max(round(c * n), 1)
    f = lambda k: M(m, k)
    G = lambda k: comb(m + k, k)          # SBS, k sources: every photon number 0..k, sum_j M_j = C(m+k, k)
    return {
        "fock_table":   4 * f(n),
        "fock_pull":    pull_peak(f, n),
        "fock_push":    A * (f(n - 1) + f(n)),
        "fock_edges":   9 * n * f(n) + A * (f(n - 1) + f(n)),   # current kernel: int32 parent+child, int8 states
        "fock_cf_pull": pull_peak(lambda k: Mcf(m, k), n),
        "emit_pull":    pull_peak(f, n, chi=2),
        "emit_push":    2 * A * (f(n - 1) + f(n)),
        "sbs":          A * G(n - 1) + 4 * G(n),             # n SOURCES, no post-selection; last source pulled into p
        "nlbs":         A * (f(n - 1) + 2 * f(n)),          # non-linear sites, branch route: a branch's last layer + the coherent sum
        # gbs: n SQUEEZERS at GBS_MEAN photons each, table of every even photon number up to the cutoff that
        # leaves GBS_EPS of the mass out (the distribution is unbounded, so this is an approximation by design)
        "gbs":          4 * sum(f(k) for k in range(0, gbs_cutoff(n) + 1, 2)),
    }


def gbs_bytes(n, c, mean=GBS_MEAN, eps=GBS_EPS):
    """(table, build peak, N_cut) for n squeezers at `mean` photons each, m = c n, every photon number up to the
    cutoff that leaves `eps` of the mass out. Lossless undisplaced input: only even layers are non-zero. table =
    float32 probabilities of every even layer <= N_cut; build = the table plus the Gaussian SLOS recurrence's working
    set (GAUSSIAN.md §7: c_N = -(1/N) sum_s tanh r_s R(u_s) R(u_s) c_(N-2), complex64): c_(N-2) and the raised
    R(u_s) c_(N-2) (layer N-1) held while layer N is pulled chunk-wise into p.
    """
    m = max(round(c * n), 1)
    nc = gbs_cutoff(n, mean, eps)
    table = 4 * sum(M(m, k) for k in range(0, nc + 1, 2))
    build = table + max((A * (M(m, k - 2) + M(m, k - 1)) for k in range(2, nc + 1, 2)), default=0)
    return table, build, nc


def gbs_frontier(c, mean=GBS_MEAN, eps=GBS_EPS, which=1):
    n = 1
    while gbs_bytes(n + 1, c, mean, eps)[which] <= B:
        n += 1
    lo, hi = gbs_bytes(n, c, mean, eps)[which], gbs_bytes(n + 1, c, mean, eps)[which]
    return n, n + (log(B) - log(lo)) / (log(hi) - log(lo)), gbs_cutoff(n, mean, eps)

def frontier(key, c, need=None):
    n = 1
    while models(n + 1, c)[key] <= B: n += 1
    lo, hi = models(n, c)[key], models(n + 1, c)[key]
    return n, n + (log(B) - log(lo)) / (log(hi) - log(lo)), round(c * n)

keys = ["fock_table", "fock_pull", "fock_push", "fock_edges", "fock_cf_pull", "emit_pull", "emit_push", "sbs", "nlbs", "gbs"]
ratios = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]
print(f"budget {B/2**30:.0f} GiB   (integer n / interpolated n*; gbs: squeezers at {GBS_MEAN} photons each, "
      f"{GBS_EPS:g} of the mass uncaptured)")
print("c=m/n  rho_mem " + " ".join(f"{k:>13}" for k in keys))
for c in ratios:
    rho = (1 + c) ** (1 + c) / c ** c
    print(f"{c:<6} {rho:7.2f} " + " ".join("{:>4d} /{:6.1f} ".format(*frontier(k, c)[:2]) for k in keys))
print("gbs, n squeezers, lossless, build peak (table + recurrence working set); N_cut = the photon cutoff at the frontier:")
print("  c=m/n   " + "   ".join(f"eps {e:g}: n* (N_cut)" for e in (1e-3, 1e-4, 1e-6)))
for c in ratios:
    print(f"  {c:<6} " + "   ".join("{:>4d} /{:6.1f} ({:3d})    ".format(*gbs_frontier(c, eps=e, which=1)) for e in (1e-3, 1e-4, 1e-6)))
print("gbs at c = 2, eps 1e-4 (1/S, S = 1e4), photons per squeezer:")
for mean in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
    nn, ns, nc = gbs_frontier(2, mean=mean, eps=1e-4, which=1)
    print(f"  {mean:<5} n* {ns:6.1f}  (N_cut {nc} at n = {nn}, m = {2 * nn})")
print("m = n^2:")
n = 1
while pull_peak(lambda k, n=n+1: M((n)**2, k), n + 1) <= B: n += 1
ncf = 1
while pull_peak(lambda k, n=ncf+1: Mcf((n)**2, k), ncf + 1) <= B: ncf += 1
print("  fock_pull n =", n, " collision-free n =", ncf)
# build work at the frontier: n*M_n edges; measured 98 ms at n=11,m=22 -> edges/s
rate = 11 * M(22, 11) / 0.098
for c in (2, 4, 8):
    nn, _, m = frontier("fock_pull", c)
    print(f"c={c}: n={nn} m={m} states={M(m,nn):.3e} edges={nn*M(m,nn):.3e} build~{nn*M(m,nn)/rate:.0f}s at {rate:.2e} edges/s")

# temporal family (loops, one rail, one loop of delay L; docs/HYBRID.md): the strong object is the register, not a table.
# Dense register of every photon number <= N: C(N + L + 1, L + 1) complex128 states per trajectory (x 2 spin and one
# more rail wire for the emitter's 2-bin period); its density matrix (exact marginals) is the square. Sector register
# (the sampler of record): only the current photon count p in the L slots, C(p + L, L) during a period (the bin in
# transit included) - set by the loop occupancy, not by N. Counted in amplitudes x 16 B, one trajectory, the budget above.
def _largest(f):
    x = 0
    while f(x + 1) <= B:
        x += 1
        if x > 100000:
            return None
    return x


print("hybrid (loops, one rail, delay L): largest N whose dense register fits | largest N for its density matrix | "
      "largest loop occupancy p whose sector fits")
print("  L    fock: N_pure  N_rho    emitter: N_pure  N_rho    sector: p (fock)  p (emitter)")
for L in (1, 2, 4, 6, 8, 10, 12, 16):
    fp = _largest(lambda n, L=L: 16 * comb(n + L + 1, L + 1))
    fr = _largest(lambda n, L=L: 16 * comb(n + L + 1, L + 1) ** 2)
    ep = _largest(lambda n, L=L: 16 * 2 * comb(n + L + 2, L + 2))
    er = _largest(lambda n, L=L: 16 * (2 * comb(n + L + 2, L + 2)) ** 2)
    sp = _largest(lambda p, L=L: 16 * comb(p + L, L))
    se = _largest(lambda p, L=L: 16 * 2 * comb(p + L + 1, L + 1))
    v = [">1e5" if x is None else str(x) for x in (fp, fr, ep, er, sp, se)]
    print(f"  {L:<4} {v[0]:>11} {v[1]:>6} {v[2]:>17} {v[3]:>6} {v[4]:>17} {v[5]:>12}")

if FIG:
    import csv
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 12]
    # sampling frontier, 1 sample/s, one H100: Fock = tab:frontier of the paper, others = frontier3 run
    fock = [42.4, 41.3, 40.0, 39.0, 38.5, 37.5, 36.8, 36.5, 36.2, 35.6, 35.3, 34.9, 34.4]
    samp = {"emitter": []}           # SBS: no sampling frontier on the source axis yet
    path = os.path.join(os.path.dirname(__file__), "..", "results", "keep", "frontier_20260920_fock_emitter_h100.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["engine"] in samp: samp[r["engine"]].append((float(r["ratio"]), float(r["n_star"])))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    col = {"fock": "#1f77b4", "emitter": "#d62728", "sbs": "#2ca02c"}
    ax.plot(cs, fock, "o-", color=col["fock"], label="sampling, Fock (1 sample/s)")
    if samp["emitter"]: ax.plot(*zip(*samp["emitter"]), "o-", color=col["emitter"], label="sampling, emitter")
    for key, e, ls, lab in (("fock_pull", "fock", "--", "memory, all three inputs (SBS within 0.4)"),):
        cc = cs
        ax.plot(cc, [frontier(key, c)[1] for c in cc], ls, marker="s", ms=3, color=col[e], label=lab)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 4, 6, 8, 12]); ax.set_xticklabels(["1", "2", "3", "4", "6", "8", "12"])
    ax.set_xlabel("mode ratio m/n"); ax.set_ylabel("sources n*"); ax.set_ylim(0, 54); ax.grid(alpha=.3)
    ax.set_title(f"Two frontiers on one H100: 1 sample/s vs {B/2**30:.0f} GiB")
    ax.legend(fontsize=7, ncol=1, loc="upper right"); fig.tight_layout(); fig.savefig(FIG, dpi=200)
    print("wrote", FIG)
