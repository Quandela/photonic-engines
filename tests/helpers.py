"""Shared test helpers (no engine code, no external simulator)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch


def haar_unitary(m: int, seed: int) -> torch.Tensor:
    """Haar-random m x m unitary, complex128, deterministic in `seed`."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(
        m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    return q * (d / d.abs()).unsqueeze(0)


def haar_unitary_np(m: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def occupation_list(n: int, m: int, mask: str = "fock") -> list[int]:
    """Input occupation conforming to the output mask: one photon per mode pair when m >= 2n
    (dual-rail spacing, no bunching), else one photon per mode in the first n modes."""
    if mask == "dual_rail":
        return [1, 0] * (m // 2)
    if m >= 2 * n:
        return [1, 0] * n + [0] * (m - 2 * n)
    return [1] * n + [0] * (m - n)


def superposition_components(n: int, m: int, k: int) -> list[tuple]:
    """K distinct n-photon (one-per-mode) Fock states, deterministic. Requires k <= C(m, n)."""
    combos = list(combinations(range(m), n))
    if k > len(combos):
        raise ValueError(f"k={k} exceeds C({m},{n})={len(combos)}")
    states = []
    for modes in combos[:k]:
        occ = [0] * m
        for mode in modes:
            occ[mode] = 1
        states.append(tuple(occ))
    return states
