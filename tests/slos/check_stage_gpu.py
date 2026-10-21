"""Device check of the stage kernel (`slos.stage.StageKernel`, docs/DESIGN.md §2.4) and of ragged emitter chains: on
cuda, complex128, against the same kernels on the CPU (kraus / probs / bond amplitudes, <= 1e-13), batch of inputs =
per-input calls (<= 1e-15), the vacuum dispatch = EmitterSLOSKernel on the device (torch.equal), kraus(chunk=) =
unchunked, gradients finite and = the CPU ones. Partial detection (R4, route B): the regroup tables equal on both
devices, blocks and apply (with a passive rest) cuda = cpu (<= 1e-13), apply(chunk=1) = unchunked (<= 1e-15), the
a-posteriori deviation small where the inputs avoid the emission modes, gradients of the detected probabilities
cuda = cpu, and a photons-only stage. Runnable script, not collected by pytest (run it on a GPU); exits 1 if any
check fails.

    python3 tests/slos/check_stage_gpu.py [--device cuda]
"""
import argparse
import math
import sys

import numpy as np
import torch

from photonic_engines import EmitterMPS, StageChain, spin_gate
from photonic_engines.slos.emitter import EmitterSLOSKernel
from photonic_engines.slos.stage import StageKernel

C128 = torch.complex128
FAILED = []


def haar(m, seed):
    """Haar-random unitary [m, m] complex128 (CPU)."""
    z = np.random.default_rng(seed).normal(size=(m, m, 2)) @ [1, 1j]
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / abs(np.diag(r))), dtype=C128)


def isometry(L, chi, rng):
    """L operators chi x chi with sum_l M_l^dag M_l = I."""
    q, _ = np.linalg.qr(rng.normal(size=(L * chi, chi)) + 1j * rng.normal(size=(L * chi, chi)))
    return q.reshape(L, chi, chi)


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def err(a, b):
    return float((a.cpu() - b.cpu()).abs().max())


def carry(name, ch, inputs, U, dev, free, rest=2):
    """Partial detection (every mode but each third one detected) on the device against the CPU."""
    det = [p for p in range(ch.m) if p % 3 != 2]
    D = len(inputs) * ch.chi * rest
    z = np.random.default_rng(D).normal(size=(D, D, 2)) @ [1, 1j]
    rho = torch.as_tensor(z @ z.conj().T, dtype=C128)
    rho = rho / torch.diagonal(rho).real.sum()
    kd = StageKernel(ch, inputs=inputs, detected=det, device=dev)
    kc = StageKernel(ch, inputs=inputs, detected=det)
    tol = None if free else float("inf")             # inputs on emission modes: a non-zero deviation is expected
    rd, rc = kd.apply(U.to(dev), rho.to(dev), rest=rest, tol=tol), kc.apply(U, rho, rest=rest, tol=tol)
    same = torch.equal(rd.detected.cpu(), rc.detected) and torch.equal(rd.carried.cpu(), rc.carried) and all(
        torch.equal(a.q.cpu(), b.q) and torch.equal(a.basis.cpu(), b.basis) for a, b in zip(rd.classes, rc.classes))
    check(f"carry tables cuda = cpu ({name})", same and len(rd.classes) == len(rc.classes))
    e = max(err(a.value, b.value) for a, b in zip(kd.blocks(U.to(dev)).classes, kc.blocks(U).classes))
    check(f"blocks cuda = cpu ({name})", e <= 1e-13, f"{e:.1e}")
    e = max(max(err(a.value, b.value), err(a.probs, b.probs)) for a, b in zip(rd.classes, rc.classes))
    check(f"apply cuda = cpu ({name}, rest {rest})", e <= 1e-13, f"{e:.1e}")
    e = max(err(a.value, b.value) for a, b in zip(rd.classes, kd.apply(U.to(dev), rho.to(dev), rest=rest,
                                                                         chunk=1, tol=tol).classes))
    check(f"apply(chunk=1) = unchunked ({name})", e <= 1e-15, f"{e:.1e}")
    e = abs(float(rd.deviation) - float(rc.deviation))
    dv = abs(float(rd.deviation))
    check(f"deviation cuda = cpu ({name})", e <= 1e-13 and (not free or dv <= 1e-12), f"{e:.1e}, deviation {dv:.1e}")
    grads = []
    for d in (dev, "cpu"):
        Ug = U.to(d).clone().requires_grad_(True)
        res = StageKernel(ch, inputs=inputs, detected=det, device=d).apply(Ug, rho.to(d), rest=rest, tol=tol)
        loss = sum((cl.probs * torch.linspace(0.5, 2.0, cl.probs.numel(), dtype=torch.float64, device=d)).sum()
                   for cl in res.classes)
        loss.backward()
        grads.append(Ug.grad.cpu())
    e = err(grads[0], grads[1])
    check(f"carry gradients cuda = cpu, finite ({name})", bool(torch.isfinite(grads[0]).all()) and e <= 1e-12,
          f"{e:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    dev = ap.parse_args().device
    print("device", dev, torch.cuda.get_device_name(0) if dev.startswith("cuda") else "", flush=True)
    rng = np.random.default_rng(3)

    # chains: a caterpillar, a ragged d = 1 + spin register, a ragged chi = 3 chain
    cat = EmitterMPS.caterpillar(4, [spin_gate(math.pi / 2, 0.3 * i) for i in range(4)], m=12,
                                 modes=[(2 * b, 2 * b + 1) for b in range(4)], final=(0.6, 0.8j))
    up, dn = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    G = spin_gate(math.pi / 2, 0.7)
    sps_spin = EmitterMPS.ragged([np.eye(2)[None], np.stack([up @ G, dn @ G]), np.eye(2)[None],
                                  np.stack([up @ G, dn @ G])], [(0,), (1, 2), (3,), (4, 5)], (1, 0), m=10)
    r3 = EmitterMPS.ragged([isometry(c, 3, rng) for c in (1, 3, 2)], [(0,), (1, 2, 3), (4, 5)], np.eye(3)[0], m=10)
    cases = {"caterpillar": (cat, [[0] * 8 + [1, 0, 0, 1], [0] * 9 + [2, 0, 0], [0] * 11 + [1]]),
             "sps+spin": (sps_spin, [[0] * 6 + [1, 1, 0, 0], [0] * 7 + [1, 0, 0], [0] * 6 + [1, 0, 1, 1]]),
             "ragged chi3": (r3, [[0] * 6 + [1, 0, 0, 1], [0] * 8 + [2, 0], [1] + [0] * 9])}

    for name, (mps, inputs) in cases.items():
        print(f"== {name}: n = {mps.n}, chi = {mps.chi}, m = {mps.m}, labels {mps.labels}", flush=True)
        ch = StageChain.from_mps(mps)
        U = haar(mps.m, 11)
        # the R1 ragged emitter kernel on the device vs the CPU
        for meth in ("kraus", "probs"):
            a = getattr(EmitterSLOSKernel(mps, dtype=C128, device=dev), meth)(U.to(dev))
            b = getattr(EmitterSLOSKernel(mps, dtype=C128), meth)(U)
            check(f"EmitterSLOSKernel.{meth} cuda = cpu", err(a, b) <= 1e-13, f"{err(a, b):.1e}")
        # the vacuum dispatch on the device: today's kernel, the same tensors
        sk0 = StageKernel(ch, device=dev)
        ek = EmitterSLOSKernel(mps, dtype=C128, device=dev)
        check("dispatch kraus = EmitterSLOSKernel (device)", torch.equal(sk0.kraus(U.to(dev))[0].value[0], ek.kraus(U.to(dev))))
        # stage kernel with Fock inputs: device vs CPU, batch vs per-input, chunk
        skd, skc = StageKernel(ch, inputs=inputs, device=dev), StageKernel(ch, inputs=inputs)
        for meth in ("kraus", "probs", "bond_amplitudes"):
            gd, gc = getattr(skd, meth)(U.to(dev)), getattr(skc, meth)(U)
            e = max(err(x.value, y.value) for x, y in zip(gd, gc))
            check(f"StageKernel.{meth} cuda = cpu", e <= 1e-13, f"{e:.1e}")
        gd = skd.kraus(U.to(dev))
        for g in gd:
            for row, b in enumerate(g.inputs.tolist()):
                (one,) = StageKernel(ch, inputs=[inputs[b]], device=dev).kraus(U.to(dev))
                e = err(one.value[0], g.value[row])
                check(f"batch = per-input (input {b})", e <= 1e-15, f"{e:.1e}")
        e = max(err(x.value, y.value) for x, y in zip(gd, skd.kraus(U.to(dev), chunk=1)))
        check("kraus(chunk=1) = unchunked", e <= 1e-15, f"{e:.1e}")
        # gradients through U and ops_tensor: finite, = the CPU ones
        grads = []
        for d in (dev, "cpu"):
            Ug = U.to(d).clone().requires_grad_(True)
            ops = torch.as_tensor(ch.ops, dtype=C128, device=d).requires_grad_(True)
            loss = sum(g.value.sum() for g in StageKernel(ch, inputs=inputs, device=d, ops_tensor=ops).probs(Ug))
            loss.backward()
            grads.append((Ug.grad.cpu(), ops.grad.cpu()))
        e = max(err(grads[0][0], grads[1][0]), err(grads[0][1], grads[1][1]))
        ok = all(torch.isfinite(t).all() for t in grads[0])
        check("gradients cuda = cpu, finite", ok and e <= 1e-12, f"{e:.1e}")
        carry(name, ch, inputs, U, dev, free=name != "ragged chi3")
        # the vacuum input of a single-photon chain dispatches to EmitterSLOSKernel: its carry tables read that
        # kernel's (device) patterns
        carry(f"{name} vacuum dispatch", ch, [[0] * mps.m], U, dev, free=True)

    # labels of any photon number and env records (R2): graded recurrence on the device against the CPU
    print("== words (vacuum, two-photon, pair labels) and env records", flush=True)
    rng = np.random.default_rng(9)
    for name, events, env in [
            ("words", [((0, 1), [[0, 0], [1, 0], [0, 1], [2, 0]]), ((2, 3), [[1, 0], [0, 1], [1, 1]])], None),
            ("env", [((0, 1), [[1, 0], [0, 1], [0, 0], [0, 0]]), ((2, 3), [[1, 0], [0, 1], [1, 0], [0, 1]])],
             (np.array([0, 0, 1, 2]), np.array([0, 0, 1, 1])))]:
        L = max(len(w) for _, w in events)
        ops = np.zeros((2, L, 2, 2), dtype=complex)
        for k, (_, w) in enumerate(events):
            ops[k, :len(w)] = isometry(len(w), 2, rng)
        ch = StageChain(ops=ops, event_modes=tuple(P for P, _ in events), s0=(0.6, 0.8), m=8,
                        words=tuple(np.asarray(w) for _, w in events), env=env)
        U = haar(8, 12)
        inputs = [[0] * 8, [0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 2, 0, 0]]
        skd, skc = StageKernel(ch, inputs=inputs, device=dev), StageKernel(ch, inputs=inputs)
        for meth in ("kraus", "probs", "bond_amplitudes"):
            e = max(err(x.value, y.value) for x, y in zip(getattr(skd, meth)(U.to(dev)), getattr(skc, meth)(U)))
            check(f"graded {meth} cuda = cpu ({name})", e <= 1e-13, f"{e:.1e}")
        tot = sum(float(g.value.sum()) for g in skd.probs(U.to(dev)))
        check(f"graded probs sum to the inputs ({name})", abs(tot - len(inputs)) <= 1e-12, f"{abs(tot - 3):.1e}")
        carry(f"graded {name}", ch, inputs, U, dev, free=True)

    # a photons-only stage with a passive register (rest = 3): scalar blocks
    print("== photons only (StageChain.empty), rest = 3", flush=True)
    carry("photons only", StageChain.empty(8), [[1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0] * 8,
                                                [0, 1, 0, 0, 0, 0, 1, 1]], haar(8, 5), dev, free=True, rest=3)

    print("FAILED:", FAILED if FAILED else "none", flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
