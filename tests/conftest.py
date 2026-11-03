import os
import sys

import pytest

# Without a CUDA device the Triton kernels run under the CPU interpreter; the knob must be set
# before the first `import triton` anywhere in the session.
try:
    import torch as _torch
    if not _torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
except Exception:
    pass

# tests/helpers.py importable from every test directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests marked slow (> 30 s each: Triton CPU-interpreter scripts, long CPU "
                          "tests, GPU statistical validations)")


# The `gpu` and `slow` markers are declared in pyproject.toml ([tool.pytest.ini_options] markers).

# Optional dependencies normally SKIP the tests that need them, which is right on a developer
# machine and wrong in CI: the first pipeline run reported 275 passed with 69 % coverage instead
# of 302 and 88 %, because triton was absent and four kernel modules quietly dropped out of the
# suite. A run with PE_REQUIRE_FULL_SUITE set turns that silence into a failure.
REQUIRED_WITH_FULL_SUITE = ("torch", "numpy", "triton", "perceval")


def pytest_configure(config):
    """Fail the run when PE_REQUIRE_FULL_SUITE is set and an optional dependency is missing."""
    if not os.environ.get("PE_REQUIRE_FULL_SUITE"):
        return
    import importlib.util
    missing = [m for m in REQUIRED_WITH_FULL_SUITE if importlib.util.find_spec(m) is None]
    if missing:
        raise pytest.UsageError(
            f"PE_REQUIRE_FULL_SUITE is set, but {', '.join(missing)} could not be imported: the tests "
            f"that need them would skip silently and the coverage gate would measure a smaller suite. "
            f"Install them, or unset PE_REQUIRE_FULL_SUITE to allow the skips.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("PE_RUN_SLOW"):
        return
    import pytest
    skip = pytest.mark.skip(reason="slow (use --runslow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
