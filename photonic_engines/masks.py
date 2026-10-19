"""Output computation-space restrictions (masks) and their state-space sizes.

Strong simulation cost scales with the number of output states, not with m alone; the
restricted spaces prune the SLOS tree (see slos/basis.py).

  FOCK         full basis: C(m + n - 1, n)
  UNBUNCHED    at most 1 photon per mode: C(m, n), valid only when n <= m
  DUAL_RAIL    exactly 1 photon per consecutive mode pair: 2**(m/2), m even, n == m // 2
  SPARSE_1IN4  at most 1 photon per consecutive 4-mode block
  TAIL_EMPTY   last 2 modes empty
"""
from __future__ import annotations

from enum import Enum
from math import comb


class Mask(str, Enum):
    """Names of the supported output computation spaces; see the module docstring for each one.

    A str enum: members compare equal to their value, and the engines take the value (`Mask.FOCK.value`).
    """

    FOCK = "fock"
    UNBUNCHED = "unbunched"
    DUAL_RAIL = "dual_rail"
    SPARSE_1IN4 = "sparse_1in4"
    TAIL_EMPTY = "tail_empty"


def is_valid(n: int, m: int, mask: str) -> bool:
    """Whether (n, m) is a valid configuration for the given output mask."""
    if n < 0 or m <= 0:
        return False
    if mask == Mask.FOCK.value:
        return True
    if mask == Mask.UNBUNCHED.value:
        return n <= m
    if mask == Mask.DUAL_RAIL.value:
        return m % 2 == 0 and n == m // 2
    if mask == Mask.SPARSE_1IN4.value:
        return m % 4 == 0 and 1 <= n <= m // 4
    if mask == Mask.TAIL_EMPTY.value:
        return m > 2 and n <= m - 2
    raise ValueError(f"unknown mask {mask!r}")


def graded_state_space_size(n: int, m: int, mask: str = Mask.FOCK.value) -> int:
    """Number of output states over ALL photon numbers 0..n, i.e. the output space of an engine
    whose input has an indefinite photon number (`states.ProductSuperposition`). C(m + n, n) for
    the full basis - (m + n) / m times the n-photon layer alone, so 1.5x at m = 2n. Raises if any
    layer is invalid for the mask (DUAL_RAIL fixes n, so it has no graded space).
    """
    for k in range(n + 1):
        if not is_valid(k, m, mask):
            raise ValueError(f"mask {mask!r} has no {k}-photon layer: no graded space for n={n}, m={m}")
    return sum(state_space_size(k, m, mask) for k in range(n + 1))


def state_space_size(n: int, m: int, mask: str = Mask.FOCK.value) -> int:
    """Number of output states for (n, m) under `mask`. Raises if the combo is invalid."""
    if not is_valid(n, m, mask):
        raise ValueError(f"invalid config n={n} m={m} for mask {mask!r}")
    if mask == Mask.FOCK.value:
        return comb(m + n - 1, n)
    if mask == Mask.UNBUNCHED.value:
        return comb(m, n)
    if mask == Mask.DUAL_RAIL.value:
        return 2 ** (m // 2)
    if mask == Mask.SPARSE_1IN4.value:
        return comb(m // 4, n) * (4 ** n)
    if mask == Mask.TAIL_EMPTY.value:
        return comb((m - 2) + n - 1, n)
    raise ValueError(f"unknown mask {mask!r}")
