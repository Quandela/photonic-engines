"""pytest entry for the Clifford sampler check scripts (kept as runnable `check_*.py` scripts,
renamed from gpu-clifford's tests/test_*.py). Without a CUDA device the kernel tests run under the Triton CPU
interpreter (TRITON_INTERPRET=1); the GPU-only scripts are skipped. Both sets are statistical validations (100k-300k
shots against the exact distribution), so both are `slow`: run them with `--runslow`."""
import importlib.util
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

HERE = os.path.dirname(os.path.abspath(__file__))
HAS_CUDA = torch.cuda.is_available()
HAS_CUPY = importlib.util.find_spec("cupy") is not None
# backend="cuda" (cupy/NVRTC kernels) and the exqalibur port import cupy, which only the `cuda` extra installs
NEEDS_CUPY = {"check_batch.py", "check_single.py", "check_cuda.py", "check_exq.py"}


def _run(script, *args, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=HERE, env=e,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"


@pytest.mark.slow
@pytest.mark.parametrize("script", ["check_cofactors.py", "check_multi.py", "check_sampler.py"])
def test_cpu_interpreter(script):
    """1-3 min each under the Triton CPU interpreter (seconds on a GPU): `--runslow`."""
    env = {} if HAS_CUDA else {"TRITON_INTERPRET": "1"}
    _run(script, env=env)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("script", ["check_sampler_gpu.py", "check_batch.py", "check_single.py",
                                    "check_cuda.py", "check_exq.py"])
def test_gpu_check_script(script):
    """13-40 s each on an NVIDIA L4 (about 2.5 min for the five): `--runslow` on a CUDA machine."""
    if script in NEEDS_CUPY and not HAS_CUPY:
        pytest.skip("needs cupy: pip install -e \".[cuda]\"")
    _run(script)
