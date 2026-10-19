"""Validation table of the methods paper: every sampler on small
instances against an oracle that shares no code with it, through the SAME call and backend as the benchmark.

    python3 bench/validate_methods.py [--device cuda] [--shots 40000] [--out results/validate_methods.csv]

Per case: total variation distance (TVD) between the empirical distribution of `shots` samples and the oracle's,
and the shot-noise floor: the TVD of an exact multinomial draw of the same size from the oracle, mean and standard
deviation over 40 draws. A correct sampler gives TVD / floor ~ 1 and |z| = |TVD - floor| / sd of order one.

    sampler      call (as benchmarked)                                      oracle
    fock         clifford.fock_sample(backend="cuda")  [find_threshold]     slos.FockSLOSKernel
    sbs          clifford.sbs_rso.sample(backend="cuda", square="kernel",   slos.ProductSLOSKernel (graded SLOS),
                 draw="device")  [frontier runs]                            loss / x-model mixtures included
    emitter      clifford.emitter_sample(backend="triton", kernel="auto")   slos.EmitterSLOSKernel, loss included
    adaptive     AdaptiveCircuit.sample(chain="marginal", final="prefix")   AdaptiveCircuit.probs (SLOS per branch)
    nlbs         NonlinearCircuit.sample(backend="cuda")                    NonlinearCircuit.probs (SLOS per branch,
                                                                            checked against slos/reference.py)
On --device cpu the GPU-only calls (fock, adaptive, nlbs) are skipped and sbs / emitter run their torch backends:
a dry run of the script, not the validation of the benchmarked kernels.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from photonic_engines import EmitterMPS, ProductSuperposition, spin_gate
from photonic_engines.adaptive import AdaptiveCircuit
from photonic_engines.clifford import emitter_sample, fock_sample
from photonic_engines.clifford.sbs_rso import sample as sbs_rso_sample
from photonic_engines.nonlinear import NonlinearCircuit
from photonic_engines.slos import FockSLOSKernel, ProductSLOSKernel
from photonic_engines.slos.emitter import EmitterSLOSKernel

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--shots", type=int, default=40000)
ap.add_argument("--only", default=None, help="comma-separated samplers (fock,sbs,emitter,adaptive,nlbs)")
ap.add_argument("--out", default="results/validate_methods.csv")
a = ap.parse_args()
dev = a.device
gpu = dev == "cuda"
C128 = torch.complex128


def haar(m, seed):
    """As bench/find_threshold.py: QR of a complex Ginibre matrix, phases of diag(R) fixed."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=C128)


def tvd_floor(occ, ref, shots, reps=40):
    """occ [shots, m] int; ref {occupation tuple: p} (normalised here). The floor is the TVD of an EXACT draw of the
    same size, by Monte Carlo over `reps` multinomial draws from ref (mean and standard deviation): valid also when
    the support is large against the shots, where the normal approximation of the floor overestimates it."""
    keys = list(ref)
    p = np.array([ref[k] for k in keys], dtype=np.float64)
    p /= p.sum()
    index = {k: i for i, k in enumerate(keys)}
    rows_, cnt = torch.unique(occ.cpu(), dim=0, return_counts=True)
    emp = np.zeros_like(p)
    outside = 0.0
    for r, c in zip(rows_.tolist(), cnt.tolist(), strict=True):
        if tuple(r) in index:
            emp[index[tuple(r)]] = c / shots
        else:
            outside += c / shots                                   # an outcome of probability zero
    tvd = 0.5 * (float(np.abs(emp - p).sum()) + outside)
    rng = np.random.default_rng(12345)
    null = [0.5 * float(np.abs(rng.multinomial(shots, p) / shots - p).sum()) for _ in range(reps)]
    return tvd, float(np.mean(null)), float(np.std(null)), len(keys)


def dense_dict(p, states):
    return {tuple(int(v) for v in s): float(q) for s, q in zip(states, p.detach().cpu().to(torch.float64).tolist())}


def caterpillar(n, phis, final=None, m=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final, **({} if m is None else {"m": m}))


def random_emitter(n, L, chi, m, seed, final=True):
    rng = np.random.default_rng(seed)
    ops = rng.normal(size=(n, L, chi, chi)) + 1j * rng.normal(size=(n, L, chi, chi))
    modes = tuple(tuple(range(k * L, (k + 1) * L)) for k in range(n))
    s0 = rng.normal(size=chi) + 1j * rng.normal(size=chi)
    f = (rng.normal(size=chi) + 1j * rng.normal(size=chi)) if final else None
    return EmitterMPS(ops=ops, modes=modes, s0=s0, m=m, final=f)


rows = []


def record(sampler, case, n, m, occ, ref, seconds):
    tvd, floor, sd, support = tvd_floor(occ, ref, a.shots)
    z = (tvd - floor) / sd if sd > 0 else float("nan")
    rows.append({"sampler": sampler, "case": case, "n": n, "m": m, "shots": a.shots, "support": support,
                 "tvd": f"{tvd:.5f}", "floor": f"{floor:.5f}", "floor_sd": f"{sd:.5f}", "ratio": f"{tvd / floor:.3f}",
                 "z": f"{z:.2f}", "seconds": f"{seconds:.1f}", "device": dev})
    print(f"{sampler:9s} {case:40s} n={n} m={m:2d} support={support:6d}  TVD {tvd:.5f}  floor {floor:.5f} "
          f"+- {sd:.5f}  ratio {tvd / floor:.3f}  z {z:+.2f}  ({seconds:.1f} s)", flush=True)


def want(name):
    return a.only is None or name in a.only.split(",")


# ---- single photons (reference) ---------------------------------------------------------------------------------
if want("fock") and gpu:
    for n, m, seed in ((5, 10, 1), (6, 12, 2)):
        U = haar(m, seed)
        t = list(range(n))
        k = FockSLOSKernel(n, m, [1] * n + [0] * (m - n), dtype=C128)
        ref = dense_dict(k.probs(U).to(torch.float64), k.layers.final_occupations())
        t0 = time.perf_counter()
        occ = fock_sample(U.to(dev), t, a.shots, generator=torch.Generator(device=dev).manual_seed(seed), backend="cuda")
        record("fock", "Haar, lossless", n, m, occ, ref, time.perf_counter() - t0)

# ---- superposed sources -----------------------------------------------------------------------------------------
if want("sbs"):
    kw = ({"backend": "cuda", "square": "kernel", "draw": "device"} if gpu
          else {"backend": "torch", "square": "chain", "draw": "host"})
    for n, m, alpha, eta, x, seed in ((6, 12, math.pi / 4, 1.0, None, 1), (5, 6, 1.1, 1.0, None, 2),
                                      (6, 12, math.pi / 4, 0.7, None, 3), (6, 12, math.pi / 4, 1.0, 0.8, 4),
                                      (6, 12, math.pi / 4, 0.8, 0.8, 5)):
        st = ProductSuperposition.sbs(n, m, alpha=alpha)
        U = haar(m, seed)
        ref = ProductSLOSKernel(st, device="cpu", dtype=C128).probs(
            U, **({} if eta >= 1 else {"loss": eta}), **({} if x is None else {"indist": x})).as_dict()
        t0 = time.perf_counter()
        occ = sbs_rso_sample(st, U.to(dev), a.shots, generator=torch.Generator().manual_seed(seed), device=dev,
                             loss=None if eta >= 1 else eta, indist=x, **kw)
        case = f"alpha={alpha:.3g} eta={eta:g}" + ("" if x is None else f" x={x:g}")
        record("sbs", case, n, m, occ, ref, time.perf_counter() - t0)

# ---- emitter-generated inputs -----------------------------------------------------------------------------------
if want("emitter"):
    kw = {"backend": "triton", "kernel": "auto", "device": dev} if gpu else {"backend": "torch", "device": dev}
    cases = (("linear cluster, traced (benchmark state)", caterpillar(4, [0, 0, 0], m=8), 1.0, 1),
             ("caterpillar, projected", caterpillar(5, [0.3, 1.1, 2.0, 0.5], final=(0.6, 0.8j)), 1.0, 2),
             ("random, L=3, chi=2, traced", random_emitter(3, 3, 2, m=9, seed=6, final=False), 1.0, 3),
             ("random, L=2, chi=3, traced", random_emitter(3, 2, 3, m=7, seed=8, final=False), 1.0, 4),
             ("linear cluster, traced, eta=0.7", caterpillar(4, [0, 0, 0], m=8), 0.7, 5))
    for case, st, eta, seed in cases:
        U = haar(st.m, seed)
        k = EmitterSLOSKernel(st, dtype=C128)
        if eta >= 1:
            ref = dense_dict(k.probs(U).to(torch.float64), k.layers.final_states)
        else:
            ref = k.probs(U, loss=eta).as_dict()
        ckw = kw
        if gpu and not (st.chi == 2 and st.num_labels <= 2):   # the Triton kernels cover chi = 2, L <= 2 only
            ckw = {"backend": "torch", "device": dev}           # the torch reference sampler, on the GPU
            case = case + " [torch backend]"
        t0 = time.perf_counter()
        occ = emitter_sample(st, U.to(dev), a.shots, generator=torch.Generator().manual_seed(seed),
                             loss=None if eta >= 1 else eta, **ckw)
        record("emitter", case, st.n, st.m, occ, ref, time.perf_counter() - t0)

# ---- adaptive linear optics -------------------------------------------------------------------------------------
if want("adaptive") and gpu:
    def haar_policy(m, K_upto, salt):
        def policy(lam, p):
            return haar(m - K_upto[lam], hash((salt, lam, tuple(p))) % (2 ** 31)).to(dev)
        return policy
    for n, m, t, levels, K_upto, seed in ((3, 6, [0, 2, 4], [[1], [0, 3]], [0, 1, 3], 1),
                                          (5, 10, [0, 2, 4, 6, 8], [[1, 3]], [0, 2], 2),
                                          (4, 8, [0, 2, 4, 6], [[1], [3], [5]], [0, 1, 2, 3], 3)):
        C = AdaptiveCircuit(haar(m, seed).to(dev), t, levels, haar_policy(m, K_upto, 7))
        ref = dense_dict(C.probs(), C.occupations())
        t0 = time.perf_counter()
        occ = C.sample(a.shots, generator=torch.Generator(device=dev).manual_seed(seed), chain="marginal",
                       final="prefix", backend="triton")
        record("adaptive", f"K={C.K}, {len(levels)} level(s)", n, m, occ, ref, time.perf_counter() - t0)

# ---- non-linear sites -------------------------------------------------------------------------------------------
if want("nlbs") and gpu:
    for case, m, t, nu, sites, seed in (("one Kerr site, phi=pi/2", 6, [0, 1, 3], 2, [(1, 2, math.pi / 2)], 1),
                                        ("one Kerr site, phi=pi/4", 6, [0, 1, 3], 2, [(1, 2, math.pi / 4)], 2),
                                        ("one Kerr site, phi=0.9", 6, [0, 1, 3], 2, [(1, 2, 0.9)], 3),
                                        ("two sites in sequence", 6, [0, 1, 3], 3, [(1, 1, math.pi / 2), (2, 4, 0.6)], 4)):
        C = NonlinearCircuit([haar(m, 20 * seed + i).to(dev) for i in range(nu)], t, sites)
        ref = dense_dict(C.probs(), C.occupations())
        t0 = time.perf_counter()
        occ = C.sample(a.shots, generator=torch.Generator(device=dev).manual_seed(seed), backend="cuda")
        record("nlbs", case, len(t), m, occ, ref, time.perf_counter() - t0)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {a.out} ({len(rows)} cases)")
