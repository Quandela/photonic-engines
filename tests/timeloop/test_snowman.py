"""PeriodicNetworkKernel on a feedback program with two delay lines (the snowman: a processing loop inside a storage
loop that sends the train back through it), checked against permanents of a separate single-photon propagation."""
from __future__ import annotations

import pytest

from bench.snowman_check import max_error


@pytest.mark.parametrize(("m", "passes", "seed"), [(2, 3, 0), (3, 2, 7)])
def test_snowman_amplitudes_equal_the_permanents(m, passes, seed):
    # float64 rounding of m! products of unit-modulus entries: ~1e-16 observed, 1e-12 leaves a wide margin
    assert max_error(m, passes, seed) < 1e-12
