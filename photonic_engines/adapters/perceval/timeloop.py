"""Perceval processors with time delays -> photonic_engines.timeloop.PeriodicNetworkKernel.

Perceval's `TD(t)` is a delay line on one mode at one position of the circuit: whatever enters
it at period d comes out at the same position at period d + t, the same circuit acts every
period and the same input state enters every period (`DelaySimulator` unrolls `depth` periods
into one spatial circuit and reports the last period's output). In the register picture a TD
is a swap between the rail wire and the head slot of a t-slot delay line, so a processor maps
onto a per-period program of two-mode gates:

    BS / any 2-mode unitary on (a, a+1)  ->  ("gate", a, a+1, U)
    PS on mode a                         ->  ("gate", a, partner, diag(e^{i phi}, 1))
    PERM                                 ->  adjacent swaps
    TD(t) on mode a                      ->  ("gate", a, ("slot", k), SWAP), delay line k of t slots

Note the difference with an ORCA loop (`LoopRegisterKernel`): there the beam splitter couples
the rail directly to the loop slot (a cycle), which a feed-forward Perceval circuit cannot
express with TD alone.
"""
from __future__ import annotations

import numpy as np
import torch

from ...timeloop import PeriodicNetworkKernel

SWAP = np.array([[0, 1], [1, 0]], dtype=complex)


def _components(source) -> tuple[list, int]:
    if hasattr(source, "components") and hasattr(source, "m"):        # Processor
        return list(source.components), int(source.m)
    if hasattr(source, "_components"):                                  # Circuit
        return list(source._components), int(source.m)
    raise TypeError("expected a Perceval Processor or Circuit")


def _perm_swaps(perm: list[int]) -> list[tuple[int, int]]:
    """Adjacent transpositions (i, i+1) realising output[perm[i]] = input[i] (Perceval PERM)."""
    cur = list(range(len(perm)))                        # cur[pos] = input index currently at pos
    target = {i: perm[i] for i in range(len(perm))}     # input i must end at position perm[i]
    swaps = []
    for pos in range(len(perm)):
        # bring the input that belongs at `pos` to `pos` by adjacent swaps from where it is
        want = next(i for i in cur if target[i] == pos)
        j = cur.index(want)
        while j > pos:
            cur[j - 1], cur[j] = cur[j], cur[j - 1]
            swaps.append((j - 1, j))
            j -= 1
    return swaps


def periodic_program_from_perceval(source):
    """(m, delays, program, gates [n_gates, 2, 2] complex128) for a Perceval Processor/Circuit:
    the per-period program of two-mode gates in the register picture (module docstring).
    """
    from perceval.components import TD

    comps, m = _components(source)
    program, mats, delays = [], [], []

    def gate(a, b, U):
        program.append(("gate", int(a), b, len(mats)))
        mats.append(np.asarray(U, dtype=complex).reshape(2, 2))

    for r, c in comps:
        r = tuple(int(x) for x in r)
        if isinstance(c, TD):
            t = float(c.param("t"))
            if not t.is_integer() or t < 1:
                raise ValueError(f"TD delay must be a positive integer, got {t}")
            gate(r[0], ("slot", len(delays)), SWAP)
            delays.append(int(t))
            continue
        name = type(c).__name__
        if name == "PERM":
            perm = list(c.perm_vector)
            for i, j in _perm_swaps(perm):
                gate(r[0] + i, r[0] + j, SWAP)
            continue
        U = np.asarray(c.compute_unitary(), dtype=complex)
        if U.shape == (1, 1):
            a = r[0]
            partner = a + 1 if a + 1 < m else a - 1
            if partner < 0:
                raise NotImplementedError("a single-mode circuit has no partner wire for a phase gate")
            gate(a, partner, np.diag([U[0, 0], 1.0]))
        elif U.shape == (2, 2):
            gate(r[0], r[1], U)
        else:
            raise NotImplementedError(f"{name} on {len(r)} modes: only 1- and 2-mode components, PERM and TD are supported")
    if not delays:
        raise ValueError("no TD component found")
    return m, delays, program, torch.as_tensor(np.stack(mats), dtype=torch.complex128)


def periodic_kernel_from_perceval(source, input_state, periods: int | None = None,
                                  device: str = "cpu", dtype: torch.dtype = torch.complex128):
    """(PeriodicNetworkKernel, gates [n_gates, T, 2, 2]) for a Perceval Processor with TD
    components and a BasicState input (entered every period, as Perceval does). `periods`
    defaults to Perceval's own unrolling depth. The last period's rail distribution
    (`kernel.period_marginals(gates)[-1]` over `kernel.basis.rail_patterns`) is what
    `Processor.probs()` reports (without post-selection).
    """
    m, delays, program, mats = periodic_program_from_perceval(source)
    if periods is None:
        try:
            from perceval.simulators.delay_simulator import _compute_depth
            periods = int(_compute_depth(_components(source)[0], m))
        except Exception:
            periods = 1 + sum(delays)
    occ = [int(x) for x in input_state]
    if len(occ) != m:
        raise ValueError(f"input state has {len(occ)} modes, processor has {m}")
    kernel = PeriodicNetworkKernel(m, delays, program, periods, [occ] * periods,
                                   n_gates=mats.shape[0], device=device, dtype=dtype)
    gates = mats.to(device=device, dtype=dtype)[:, None].expand(-1, periods, 2, 2)
    return kernel, gates


def last_period_distribution(kernel: PeriodicNetworkKernel, gates, loss=None):
    """{rail pattern tuple: probability} of the last period (what Perceval reports)."""
    p = kernel.period_marginals(gates, loss=loss)[-1].detach().cpu().numpy()
    return {tuple(int(x) for x in pat): float(v) for pat, v in zip(kernel.basis.rail_patterns, p) if v > 0}
