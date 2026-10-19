"""The permanent oracle (Ryser) at amplitude level: sign and value against the permutation sum."""
import math

import pytest
import torch


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_permanent_matches_brute_force_with_sign(n):
    """Ryser's formula against the permutation sum, amplitude level (a regression: an extra
    (-1)^n made odd sizes come out with the wrong sign, invisible in |amp|^2)."""
    import itertools

    from photonic_engines.clifford.emitter_reference import _permanent
    from photonic_engines.slos.reference import permanent
    A = torch.randn(n, n, dtype=torch.complex128, generator=torch.Generator().manual_seed(n))
    brute = sum(torch.prod(torch.stack([A[i, p[i]] for i in range(n)])) for p in itertools.permutations(range(n)))
    assert abs(permanent(A) - brute) < 1e-12 * max(1.0, abs(brute))
    assert abs(_permanent(A) - brute) < 1e-12 * max(1.0, abs(brute))
    assert abs(permanent(torch.ones(n, n, dtype=torch.complex128)) - math.factorial(n)) < 1e-12
