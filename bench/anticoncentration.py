"""Anticoncentration of emitter-state boson-sampling distributions vs Fock inputs, over Haar U.

For each input family and n (m = 2n), the full output distribution p(t) is computed with the
strong engines for several Haar unitaries and summarised by
    Z2     = |S| sum_t p(t)^2          (1 for uniform; the 'collision probability' scale)
    dH     = log|S| - H(p)             (entropy deficit, nats)
    m_low  = sum of p(t) over t with p(t) < 1/(2|S|)     (mass on 'dark' outputs)
    f_big  = fraction of t with |S| p(t) >= 1/2          (AA-type anticoncentration count)
on the full Fock basis and on the collision-free outputs (renormalised).  Emitter inputs use the
dual-rail modes (2k, 2k+1) of photon k; the Fock reference is |1 0 1 0 ...>.

Usage: python bench/anticoncentration.py [nmax] [n_unitaries] [out.csv]
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np
import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.slos.kernel import FockSLOSKernel

C128 = torch.complex128


def haar_unitary(m, gen):
    z = torch.randn(m, m, generator=gen, dtype=torch.float64) + 1j * torch.randn(m, m, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    return q * (d / d.abs()).unsqueeze(0)


def caterpillar(n, phis, final=None):
    gates = [spin_gate(math.pi / 2, 0)] + [spin_gate(math.pi / 2, p) for p in phis]
    return EmitterMPS.caterpillar(n, gates, final=final)


def random_emitter(n, chi, rng, final=False):
    """Random *unitary-generated* emitter: at each event a Haar unitary on spin x label, then the
    label is emitted (ops[k, l] = <l| V_k |0>_label): the state is automatically normalised."""
    ops = np.empty((n, 2, chi, chi), dtype=np.complex128)
    for k in range(n):
        z = rng.normal(size=(2 * chi, 2 * chi)) + 1j * rng.normal(size=(2 * chi, 2 * chi))
        V, r = np.linalg.qr(z)
        V = V * (np.diag(r) / np.abs(np.diag(r)))
        V = V.reshape(2, chi, 2, chi)                       # [label_out, spin_out, label_in, spin_in]
        for l in range(2):
            ops[k, l] = V[l, :, 0, :]
    modes = tuple((2 * k, 2 * k + 1) for k in range(n))
    s0 = np.zeros(chi, complex); s0[0] = 1
    f = None
    if final:
        f = rng.normal(size=chi) + 1j * rng.normal(size=chi); f /= np.linalg.norm(f)
    return EmitterMPS(ops=ops, modes=modes, s0=s0, m=2 * n, final=f)


def families(n, rng):
    fam = {
        "fock": None,
        "product_dualrail": random_emitter(n, 1, rng),         # chi = 1: product of dual-rail qubits
        "cluster_traced": caterpillar(n, [0] * (n - 1)),
        "cluster_proj": caterpillar(n, [0] * (n - 1), final=[1, 0]),
        "ghz_traced": caterpillar(n, [math.pi] * (n - 1)),
        "mps2_traced": random_emitter(n, 2, rng),
        "mps4_traced": random_emitter(n, 4, rng),
        "mps4_proj": random_emitter(n, 4, rng, final=True),
    }
    return fam


def stats(p):
    p = p / p.sum()
    S = p.numel()
    nz = p[p > 0]
    H = float(-(nz * nz.log()).sum())
    return {"Z2": float(S * (p * p).sum()), "dH": math.log(S) - H,
                "m_low": float(p[p < 0.5 / S].sum()), "f_big": float((S * p >= 0.5).double().mean())}


def main(nmax=8, nU=8, out="results/anticoncentration.csv"):
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    rows = []
    for n in range(4, nmax + 1):
        m = 2 * n
        fam = families(n, rng)
        kernels = {}
        for name, st in fam.items():
            if st is None:
                kernels[name] = FockSLOSKernel(n, m, [1, 0] * n, dtype=C128)
            else:
                kernels[name] = EmitterSLOSKernel(st, dtype=C128)
        layers = kernels["fock"].layers
        fs = torch.as_tensor(np.asarray(layers.final_states))
        cf = (fs.max(dim=1).values == 1)
        for u in range(nU):
            U = haar_unitary(m, gen)
            for name, K in kernels.items():
                p = K.probs(U).double()
                for basis, sel in (("fock", slice(None)), ("collision_free", cf)):
                    r = {"n": n, "m": m, "U": u, "state": name, "basis": basis, "S": int(p[sel].numel()), "mass": float(p[sel].sum())}
                    r.update(stats(p[sel]))
                    rows.append(r)
            print(f"n={n} U={u} done", flush=True)
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["basis"], r["state"], r["n"])].append(r)
    print(f"{'basis':15s} {'state':18s} n   Z2      dH      m_low   f_big   (mean over U)")
    for (b, s, n), rs in sorted(agg.items()):
        print(f"{b:15s} {s:18s} {n}  {np.mean([r['Z2'] for r in rs]):6.3f}  {np.mean([r['dH'] for r in rs]):6.3f}  "
              f"{np.mean([r['m_low'] for r in rs]):6.3f}  {np.mean([r['f_big'] for r in rs]):6.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 8, int(a[1]) if len(a) > 1 else 8, a[2] if len(a) > 2 else "results/anticoncentration.csv")
