"""Register growth with *local* optics, in the spin-optical (SPOQC) regime of de Gliniasty et al.,
Quantum 8, 1423 (2024): when does the emitter x time
machine become a hard register — a computer — without a global interferometer?

k dual-rail caterpillar emitters (spins) on the rails (2j, 2j+1) of m = 2k modes, T bins, sampled
by `SequenceKernel.sample_diag` (trajectories, O(k 3^k) per tick). Every emitted photon is a copy
of its spin in the label basis, so the bin's optics act on the register as a diagonal operator
D_t = diag_x <t|U|x> after the spin gates: measuring a photon in Z (identity optics) collapses
its spin; in X (a 45-degree beam splitter on its rails, `idle x`) it leaves a Pauli byproduct
and nothing else; a two-photon block on the rails of spins j, j+1 is a two-spin POVM:

  typeII   PBS + 45 degrees (Browne-Rudolph type-II fusion). Coincidence: projector on even
           parity (p = P(even)); bunching: both spins collapsed. A failure cuts both wires.
  xfusion  45 degrees on both photons, then type-II. Coincidence (p = 1/2 whatever the state):
           ZZ parity *measurement*, both parities; bunching: Pauli byproduct, nothing lost —
           the repeat-until-success entangling primitive of a spin-optical architecture.
  haar4    a Haar 4-mode block: generic local optics, the band-family control of the design plane.

Programs (what the optics of bin b+1 depend on):
  brick    static brickwork: pairs (j, j+1), j = b mod 2, every tick, no feed-forward.
  rus      routing feed-forward: a stream keeps attempting the pairs of its current sublattice
           until every one has succeeded once, then switches sublattice (xfusion only makes sense).
  circuit  a random circuit on the data spins (even j) with the odd spins as ancillas: the
           parity-measurement CNOT gadget [ZZ(d1, a) with a = |+>, XX(a, d2), measure a in Z —
           unitary and entangling for every outcome, Beenakker et al. 2004], repeat-until-success
           on each parity attempt, random single-qubit gates on the data spins per layer, gadgets in
           two sublayers (even / odd). Gates *and* optics fed forward per stream. k odd.

Per tick (hook): middle-cut Schmidt entropy (bits) of the register, its exact rank and 99.9 %
bond, the fusion success fraction, and (circuit) the mean number of completed layers. The
question is whether S grows with T to ~ k/2 (volume law: the register is the exponent of any
classical simulation, whatever the optics) or saturates (area law: the wires are cut, or the
measurement-only dynamics is short-ranged). Nothing here needs the strong side: the trajectory
sampler is exact and the register is the observable.

    python3 bench/esbs_fusion.py --device cuda --k 5,7,9,11,13 --optics xfusion --program circuit --ticks 60 --shots 256
    python3 bench/esbs_fusion.py --device cuda --k 4,6,8,10,12,14 --optics typeII,xfusion,haar4 --program brick --ticks 30
Writes $RESULTS_DIR/esbs_fusion_<optics>_<program>.csv (one row per (k, tick)).
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel

C128 = torch.complex128
H2 = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)


# ---- optics ---------------------------------------------------------------------------------

def _bs(m, i, j, M=H2):
    U = np.eye(m, dtype=complex)
    U[np.ix_([i, j], [i, j])] = M
    return U


def _swap(m, i, j):
    U = np.eye(m, dtype=complex)
    U[[i, j]] = U[[j, i]]
    return U


def fusion_block(kind, rng=None):
    """4 x 4 block on the rails (a0, a1, b0, b1) of the photons of two neighbouring spins."""
    type2 = _bs(4, 0, 1) @ _bs(4, 2, 3) @ _swap(4, 1, 3)
    if kind == "typeII":
        return type2
    if kind == "xfusion":
        return type2 @ _bs(4, 0, 1) @ _bs(4, 2, 3)
    if kind == "haar4":
        z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        q, r = np.linalg.qr(z)
        return q * (np.diag(r) / np.abs(np.diag(r)))
    raise ValueError(kind)


def brickwork(k, b):
    return [(j, j + 1) for j in range(b % 2, k - 1, 2)]


def bin_unitary(k, pairs, kind, idle="x", rng=None, collapse=()):
    """m = 2k unitary: fusion blocks on `pairs`, idle spins measured in X (45-degree splitter on
    their rails) or, for the spins in `collapse` or when idle = 'z', in Z (identity)."""
    m = 2 * k
    U = np.eye(m, dtype=complex)
    for j in range(k):
        if idle == "x" and j not in collapse:
            U[2 * j:2 * j + 2, 2 * j:2 * j + 2] = H2
    for a, b in pairs:
        assert b == a + 1
        U[2 * a:2 * a + 4, 2 * a:2 * a + 4] = fusion_block(kind, rng)
    return torch.as_tensor(U, dtype=C128)


def pair_success(counts, a):
    """Coincidence (one photon in each output pair) of the block on spins (a, a+1): [S] bool."""
    return (counts[:, 2 * a] + counts[:, 2 * a + 1]) == 1


# ---- policies (feed-forward per stream) ------------------------------------------------------

class RusPolicy:
    """Routing feed-forward: every stream attempts the pairs of its sublattice until each has
    succeeded once, then switches sublattice. Bin 0 = sublattice 0 for every stream."""

    def __init__(self, k, kind, shots, device, idle="x"):
        self.k, self.kind, self.S, self.dev, self.idle = k, kind, shots, torch.device(device), idle
        self.pairs = [brickwork(k, 0), brickwork(k, 1)]
        self.block = torch.as_tensor(fusion_block(kind), dtype=C128, device=self.dev)
        self.idle_block = torch.as_tensor(H2 if idle == "x" else np.eye(2), dtype=C128, device=self.dev)
        self.phase = torch.zeros(shots, dtype=torch.long)
        self.done = torch.zeros(shots, max(len(p) for p in self.pairs), dtype=torch.bool)
        self.attempted = torch.zeros_like(self.done)
        self.attempted[:, :len(self.pairs[0])] = True
        self.successes, self.attempts = 0, 0

    def __call__(self, b, counts, psi):
        c = counts.cpu()
        for p in (0, 1):
            sel = self.phase == p
            for i, (a, _) in enumerate(self.pairs[p]):
                s = pair_success(c, a) & sel & self.attempted[:, i]
                self.done[:, i] |= s
                self.successes += int(s.sum()); self.attempts += int((sel & self.attempted[:, i]).sum())
        for p in (0, 1):
            sel = (self.phase == p) & self.done[:, :len(self.pairs[p])].all(1)
            self.phase[sel] = 1 - p; self.done[sel] = False
        U = torch.zeros(self.S, 2 * self.k, 2 * self.k, dtype=C128, device=self.dev)
        for j in range(self.k):
            U[:, 2 * j:2 * j + 2, 2 * j:2 * j + 2] = self.idle_block
        self.attempted[:] = False
        for p in (0, 1):
            for i, (a, _) in enumerate(self.pairs[p]):
                sel = (self.phase == p) & ~self.done[:, i]
                self.attempted[sel, i] = True
                U[sel.to(self.dev), 2 * a:2 * a + 4, 2 * a:2 * a + 4] = self.block
        return None, U


ZZ, XX, MEAS, DONE = 0, 1, 2, 3


class CircuitPolicy:
    """Random circuit on the data spins (even j) through the parity-measurement CNOT gadget on the
    ancillas (odd j): per stream and gadget a step (ZZ attempt, XX attempt, ancilla measurement,
    done) with repeat-until-success on the two attempts; gadgets of one sublayer run in parallel,
    a layer = both sublayers, each preceded by random single-qubit gates on the data spins.
    Frames: the XX attempt applies H to a and d2 before the emission; the register stays in that
    frame until the measurement tick, whose gates H . H^-1 fold the undo in."""

    def __init__(self, k, shots, device, seed=0, idle="x"):
        assert k % 2 == 1 and k >= 3, "circuit program: k odd (data spins even, ancillas odd)"
        self.k, self.S, self.dev = k, shots, torch.device(device)
        self.gadgets = [(2 * i, 2 * i + 1, 2 * i + 2) for i in range((k - 1) // 2)]     # (d1, a, d2)
        self.sub = [[g for g in self.gadgets if (g[1] // 2) % 2 == p] for p in (0, 1)]
        self.block = torch.as_tensor(fusion_block("xfusion"), dtype=C128, device=self.dev)
        self.Hb = torch.as_tensor(H2, dtype=C128, device=self.dev)
        self.Ib = torch.eye(2, dtype=C128, device=self.dev)
        self.rng = np.random.default_rng(seed)
        self.phase = torch.zeros(shots, dtype=torch.long)
        self.layers = torch.zeros(shots, dtype=torch.float64)
        self.step = torch.zeros(shots, len(self.gadgets), dtype=torch.long)             # per gadget, valid in its sublayer
        self.fresh = torch.ones(shots, len(self.gadgets), dtype=torch.bool)
        self.pending = torch.zeros(shots, k, dtype=torch.bool)                          # data gate of the layer still to apply
        self.pending[:, 0::2] = True
        self.layer_gates = {}                                                           # layer -> [k, 2, 2] (data spins)
        # pair attempted at the last bin (spin index a of (a, a+1)) or -1
        self.attempted = torch.zeros(shots, len(self.gadgets), dtype=torch.long) - 1
        self.measured = torch.zeros(shots, len(self.gadgets), dtype=torch.bool)
        self.successes, self.attempts = 0, 0
        self.first = True

    def gates_of_layer(self, layer):
        if layer not in self.layer_gates:
            G = torch.stack([torch.as_tensor(spin_gate(self.rng.uniform(0, math.pi), self.rng.uniform(0, 2 * math.pi)), dtype=C128)
                             for _ in range(self.k)]).to(self.dev)
            self.layer_gates[layer] = G
        return self.layer_gates[layer]

    def __call__(self, b, counts, psi):
        c = counts.cpu()
        S, k = self.S, self.k
        if not self.first:                                                              # decode the outcomes of bin b
            for gi, (d1, a, _d2) in enumerate(self.gadgets):
                att = self.attempted[:, gi]
                for pair_a, nxt in ((d1, XX), (a, MEAS)):
                    sel = att == pair_a
                    if sel.any():
                        s = pair_success(c, pair_a) & sel
                        self.step[s, gi] = nxt; self.fresh[s, gi] = True
                        self.successes += int(s.sum()); self.attempts += int(sel.sum())
                m = self.measured[:, gi]
                self.step[m, gi] = DONE
        self.first = False
        for p in (0, 1):                                                                # sublayer complete -> switch
            idx = [self.gadgets.index(g) for g in self.sub[p]]
            sel = (self.phase == p) & (self.step[:, idx] == DONE).all(1)
            if sel.any():
                self.phase[sel] = 1 - p
                self.layers[sel] += 0.5
                self.step[sel] = ZZ; self.fresh[sel] = True
                self.pending[sel] = False
                if p == 1:                                                              # new layer: fresh data gates
                    self.pending[sel, 0::2] = True
        gates = self.Ib.expand(S, k, 2, 2).clone()
        U = torch.zeros(S, 2 * k, 2 * k, dtype=C128, device=self.dev)
        for j in range(k):
            U[:, 2 * j:2 * j + 2, 2 * j:2 * j + 2] = self.Hb                              # idle: X
        self.attempted[:] = -1; self.measured[:] = False
        layer_idx = torch.floor(self.layers).long()
        for gi, (d1, a, d2) in enumerate(self.gadgets):
            p = (a // 2) % 2
            active = self.phase == p
            st = self.step[:, gi]
            for step, pair_a, prep in ((ZZ, d1, (a,)), (XX, a, (a, d2))):
                sel = active & (st == step)
                if not sel.any():
                    continue
                self.attempted[sel, gi] = pair_a
                U[sel.to(self.dev), 2 * pair_a:2 * pair_a + 4, 2 * pair_a:2 * pair_a + 4] = self.block
                fr = sel & self.fresh[:, gi]
                if fr.any():
                    for j in prep:
                        gates[fr.to(self.dev), j] = self.Hb
                    for j in (d1, d2):                                                  # the layer's data gate, once
                        pj = fr & self.pending[:, j]
                        if pj.any():
                            for L in torch.unique(layer_idx[pj]).tolist():
                                s2 = pj & (layer_idx == L)
                                gates[s2.to(self.dev), j] = gates[s2.to(self.dev), j] @ self.gates_of_layer(L)[j]
                            self.pending[pj, j] = False
                    self.fresh[fr, gi] = False
            sel = active & (st == MEAS)
            if sel.any():
                self.measured[sel, gi] = True
                d = sel.to(self.dev)
                gates[d, a] = self.Hb; gates[d, d2] = self.Hb                            # undo the XX frame; a then collapses in Z
                U[d, 2 * a:2 * a + 2, 2 * a:2 * a + 2] = self.Ib
        return gates, U


# ---- bench -----------------------------------------------------------------------------------

def schmidt_entropy(psi, left):
    """Schmidt data of the register across the cut after spin `left - 1` (left spins = the most
    significant register index): entropy (bits), exact rank, 99.9 % bond, per stream."""
    a = 2 ** left
    M = psi.reshape(psi.shape[0], a, -1)
    # eigvalsh of the smaller Gram matrix = squared Schmidt values. Not svdvals: cusolver's gesvdj failed to
    # converge on these batches and fell back to a path that cost hours per run (runs 20260918_084620, 20260919_084900)
    G = M @ M.conj().transpose(1, 2) if a <= M.shape[2] else M.conj().transpose(1, 2) @ M
    sv = torch.linalg.eigvalsh(G).clamp_min(0)
    sv = torch.sort(sv, 1, descending=True)[0]
    ent = -(sv * torch.log2(sv.clamp_min(1e-300))).sum(1)
    rank = (sv > 1e-12).sum(1).float()
    bond = ((torch.cumsum(sv, 1) < 0.999).sum(1) + 1).float()
    return ent, rank, bond


def make_emitters(k, T, gates, rng, program):
    ems = []
    for j in range(k):
        if program == "circuit":
            g = [spin_gate(math.pi / 2, 0) if j % 2 == 0 else np.eye(2, dtype=complex)] + [np.eye(2, dtype=complex)] * (T - 1)
        elif gates == "h":
            g = [spin_gate(math.pi / 2, 0)] * T
        else:
            g = [spin_gate(math.pi / 2, 0)] + [spin_gate(rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)) for _ in range(T - 1)]
        ems.append(BinEmitter.caterpillar(T, g, (2 * j, 2 * j + 1)))
    return ems


def run(k, T, optics, program, gates, shots, device, seed, idle="x"):
    rng = np.random.default_rng(seed + k)
    sk = SequenceKernel(make_emitters(k, T, gates, rng, program), 2 * k, device=device)
    if program == "brick":
        U = torch.stack([bin_unitary(k, brickwork(k, b), optics, idle, rng) for b in range(T)]).to(device)
        policy = None
    elif program == "rus":
        U = bin_unitary(k, brickwork(k, 0), optics, idle, rng).to(device)
        policy = RusPolicy(k, optics, shots, device, idle)
    elif program == "circuit":
        U = bin_unitary(k, [], "xfusion", "x").to(device)                                # bin 0: preparation only
        policy = CircuitPolicy(k, shots, device, seed=seed + k)
    else:
        raise ValueError(program)
    rows, t_last = [], [time.time()]
    # cut: the middle of the chain; circuit: the data spins split evenly (left = data 0 .. kd//2 - 1 with their ancillas)
    left = 2 * (((k + 1) // 2) // 2) if program == "circuit" else k // 2
    max_entropy = min(left, k - left) if program != "circuit" else min(left // 2, (k + 1) // 2 - left // 2)

    def hook(b, psi, counts):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t_last[0]
        ent, rank, bond = schmidt_entropy(psi, left)
        fus = None
        if policy is not None and policy.attempts:
            fus = policy.successes / policy.attempts
        elif program == "brick":
            fus = float(torch.stack([pair_success(counts, a) for a, _ in brickwork(k, b)]).float().mean()) if brickwork(k, b) else None
        row = {"optics": optics, "program": program, "gates": gates, "k": k, "m": 2 * k, "tick": b + 1, "shots": shots,
                   "entropy": round(float(ent.mean()), 4), "entropy_std": round(float(ent.std()), 4),
                   "rank": round(float(rank.mean()), 2), "bond999": round(float(bond.mean()), 2), "cut": left, "max_entropy": max_entropy,
                   "fusion_success": None if fus is None else round(float(fus), 4),
                   "layers": round(float(policy.layers.mean()), 3) if program == "circuit" else None,
                   "ms_per_tick_shot": round(1e3 * dt / shots, 4)}
        rows.append(row); print(row, flush=True)
        t_last[0] = time.time()

    t_last[0] = time.time()
    sk.sample_diag(U, shots, generator=torch.Generator().manual_seed(seed), hook=hook, policy=policy)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", default="4,6,8")
    ap.add_argument("--optics", default="xfusion", help="typeII | xfusion | haar4 (comma list)")
    ap.add_argument("--program", default="brick", help="brick | rus | circuit")
    ap.add_argument("--gates", default="random", help="h | random (spin gates per tick; circuit: its own)")
    ap.add_argument("--idle", default="x", help="x | z: measurement of the photons outside a fusion block")
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    prefix = args.out_prefix or os.path.join(os.environ.get("RESULTS_DIR", "results"), "esbs_fusion")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    for optics in args.optics.split(","):
        rows = []
        for k in [int(x) for x in args.k.split(",")]:
            rows += run(k, args.ticks, optics, args.program, args.gates, args.shots, args.device, args.seed, args.idle)
        gates_tag = "" if args.gates == "random" else "_" + args.gates
        idle_tag = "" if args.idle == "x" else "_idlez"
        out = f"{prefix}_{optics}_{args.program}{gates_tag}{idle_tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
