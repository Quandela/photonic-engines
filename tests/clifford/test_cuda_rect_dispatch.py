"""Dispatch rules of clifford/cuda_rect.py (`choose_form`, `mu_auto_max`, MU_DMAX vs the kernel source): pure Python, no
C++ compiler and no GPU needed, so they run wherever torch imports (moved out of test_cuda_rect.py, whose module needs the
C++20 emulation)."""
import os

import pytest

pytest.importorskip("torch")

from photonic_engines.clifford import cuda_rect


def test_choose_form(monkeypatch):
    monkeypatch.setattr(cuda_rect, "MU_AUTO_MAX", 16)                 # 10-05 measured choice (PE_MU_R22=0)
    monkeypatch.setattr(cuda_rect, "MU_AUTO_NMAX", 48)                # 10-05 range (run 20261005_222151)
    assert cuda_rect.choose_form(6, 7) == "fock"                    # s0 = 0: the Fock limit, the Fock kernel itself
    assert cuda_rect.choose_form(38, 42) == "mu"                    # s0 = 3 at N = 42
    assert cuda_rect.choose_form(35, 42) == "mu"                    # s0 = 6: the templated lane (SBS continuity gate)
    assert cuda_rect.choose_form(34, 42) == "mu"                    # s0 = 7 (N <= MU_AUTO_NMAX = 48)
    assert cuda_rect.choose_form(25, 42) == "mu"                    # s0 = 16 = MU_AUTO_MAX
    assert cuda_rect.choose_form(24, 42) == "lambda"                # s0 = 17
    assert cuda_rect.choose_form(49, 56) == "mu"                    # s0 = 6 at N = 56 (above the measured range)
    assert cuda_rect.choose_form(48, 56) == "lambda"                # s0 = 7 at N = 56
    assert cuda_rect.choose_form(12, 20) == "lambda"                # s0 = 7 at N = 20 (below MU_AUTO_NMIN = 29)
    assert cuda_rect.choose_form(14, 20) == "mu"                    # s0 = 5 at N = 20
    monkeypatch.setattr(cuda_rect, "MU_AUTO_MAX", 18)                 # the default: mu to s0 = 18 at N <= 64
    monkeypatch.setattr(cuda_rect, "MU_AUTO_NMAX", 64)
    assert cuda_rect.choose_form(48, 56) == "mu"                    # s0 = 7 at N = 56
    assert cuda_rect.choose_form(30, 49) == "mu"                    # s0 = 18 at N = 49
    assert cuda_rect.choose_form(29, 49) == "lambda"                # s0 = 19
    assert cuda_rect.choose_form(65, 72) == "mu"                    # s0 = 6 at N = 72 > 64: the s0 <= 6 rule
    assert cuda_rect.choose_form(64, 72) == "lambda"                # s0 = 7 at N = 72
    assert cuda_rect.choose_form(37, 42, form="mu_generic") == "mu_generic"
    assert cuda_rect.choose_form(38, 42, form="lambda") == "lambda"
    assert cuda_rect.choose_form(6, 7, form="mu") == "mu"
    assert cuda_rect.choose_form(27, 42, form="mu") == "mu"           # s0 = 14 <= MU_DMAX: forced templated lane
    with pytest.raises(ValueError):
        cuda_rect.choose_form(16, 42, form="mu")                      # s0 = 25 > MU_DMAX = 24
    with pytest.raises(ValueError):
        cuda_rect.choose_form(27, 42, form="mu_generic")
    with pytest.raises(ValueError):
        cuda_rect.choose_form(38, 42, form="fock")
    with pytest.raises(ValueError):
        cuda_rect.choose_form(27, 42, form="nope")


@pytest.mark.skipif(os.environ.get("PE_MU_R22", "1") != "1" or "PE_MU_AUTO_MAX" in os.environ,
                    reason="dispatch overridden by the environment")
def test_default_dispatch_is_r22():
    """The default build: mu to s0 = 18 at N = 29..64 (the 10-06 extension, runs 20261006_122136 / _125313), s0 <= 6 elsewhere."""
    assert (cuda_rect.MU_AUTO_MAX, cuda_rect.MU_AUTO_NMIN, cuda_rect.MU_AUTO_NMAX) == (18, 29, 64)
    assert [cuda_rect.mu_auto_max(N) for N in (20, 28, 29, 49, 64, 65, 72)] == [6, 6, 18, 18, 18, 6, 6]


def test_mu_dmax_matches_the_kernel_source():
    assert f"#define PE_MU_DMAX {cuda_rect.MU_DMAX} " in cuda_rect._SRC
