"""Kind: library (max_error is the oracle of tests/timeloop/test_snowman.py; the CLI prints it)
Does: checks that the general delay-line engine (PeriodicNetworkKernel) simulates the snowman (Motes et al. 2014,
  arXiv:1403.4007; Sempere-Llagostera et al. 2022, arXiv:2204.05254) with no code change: one rail, a processing
  loop of delay 1 and a storage loop of delay m+1, random per-period settings; its multi-photon amplitudes equal
  the permanents of an independent single-photon propagation. Shows that the loop engine runs nested loops as is.
Usage: python3 bench/snowman_check.py [--m 4] [--passes 3] [--seed 42]
Writes: nothing (prints the largest amplitude error)
"""
from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
import torch

from photonic_engines.timeloop.kernel import PeriodicNetworkKernel

P, S = ("slot", 0), ("slot", 1)                    # processing loop (delay 1), storage loop (delay m + 1)
# per period: s2 takes the stored bin onto the rail, s1 (variable splitter) couples it with the processing loop,
# s2 stores the result (or lets it exit to the detector when set to identity)
PROGRAM = [("gate", 0, S, 0), ("gate", 0, P, 1), ("gate", 0, S, 2)]


def random_u2(rng: np.random.Generator) -> np.ndarray:
    """Haar random 2x2 unitary (QR of a complex Gaussian with the phases of r's diagonal fixed, Mezzadri)."""
    q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return q * (np.diag(r) / abs(np.diag(r)))


def snowman_gates(m: int, passes: int, rng: np.random.Generator) -> np.ndarray:
    """[3, T, 2, 2] gate settings: random s1 for the m bins of every pass, s1 = swap at the gap bin (the processing
    loop's content is stored in slot m, and the gap bin's stored mode enters the loop: it heads the next pass's chain,
    so each pass is a nearest-neighbour chain on the m + 1 storage modes, the loop empty only before the first two passes); s2 swaps
    (load / store) between passes, identity on the first load (input from the source) and on the last store (output
    to the detector)."""
    period = m + 1                                  # m bins + one gap bin (storage mode m)
    T = passes * period
    swap, eye = np.array([[0, 1], [1, 0]], complex), np.eye(2, dtype=complex)
    g = np.empty((3, T, 2, 2), complex)
    for t in range(T):
        p = t // period
        g[0, t] = eye if p == 0 else swap
        g[1, t] = swap if t % period == m else random_u2(rng)
        g[2, t] = eye if p == passes - 1 else swap
    return g


def transfer(gates: np.ndarray, m: int, T: int) -> np.ndarray:
    """Independent single-photon propagation of the same network: [T out periods, m input bins]."""
    d = (1, m + 1)
    out = np.zeros((T, m), complex)
    for i in range(m):
        v = np.zeros(1 + sum(d), complex)          # rail, processing slot, storage slots
        for t in range(T):
            if t == i:
                v[0] = 1.0
            for g, spec in enumerate((S, P, S)):
                k = spec[1]
                s = 1 + sum(d[:k]) + t % d[k]
                v[[0, s]] = gates[g, t] @ v[[0, s]]
            out[t, i], v[0] = v[0], 0.0
    return out


def max_error(m: int, passes: int, seed: int) -> float:
    """Largest |engine amplitude - permanent| over every output pattern of m photons (one per bin) after `passes`
    passes of the snowman with random settings drawn from `seed`."""
    rng = np.random.default_rng(seed)
    T = passes * (m + 1)
    gates = snowman_gates(m, passes, rng)
    inputs = np.zeros(T, int)
    inputs[:m] = 1                                  # one photon per bin of the train
    K = PeriodicNetworkKernel(1, (1, m + 1), PROGRAM, T, inputs)
    U = transfer(gates, m, T)
    outs = list(itertools.combinations_with_replacement(range(T), m))
    pats = np.array([np.bincount(o, minlength=T) for o in outs])
    # column of the empty register (every photon has exited): the amplitude a permanent gives
    amps = K.amplitudes(pats, torch.as_tensor(gates))[:, K._vac].numpy()
    worst = 0.0
    for o, pat, amp in zip(outs, pats, amps):
        sub = U[list(o), :]
        perm = sum(np.prod(sub[list(range(m)), list(s)]) for s in itertools.permutations(range(m)))
        perm /= np.sqrt(np.prod([float(math.factorial(c)) for c in pat]))
        worst = max(worst, abs(amp - perm))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    worst = max_error(a.m, a.passes, a.seed)
    print(f"snowman m={a.m} passes={a.passes} T={a.passes * (a.m + 1)}: max |kernel - permanent| = {worst:.2e}")
    if worst > 1e-12:                               # float64 rounding of ~m! products of unit-modulus entries
        raise SystemExit(f"kernel and permanents differ by {worst:.2e} > 1e-12")


if __name__ == "__main__":
    main()
