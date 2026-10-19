"""Distributed CPU engine: pool mode in-process, SPMD mode through mpirun (both skipped when
mpi4py / mpicc / mpirun are missing)."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("mpi4py")
pytest.importorskip("cffi")
if shutil.which(os.environ.get("MPICC", "mpicc")) is None:
    pytest.skip("mpicc not found", allow_module_level=True)

from photonic_engines.slos.distributed import DistributedSLOSKernel, PooledSLOSKernel
from photonic_engines.slos.kernel import FockSLOSKernel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pool_mode_matches_fock_kernel():
    n, m = 5, 8
    occ = [1, 1, 1, 1, 1, 0, 0, 0]
    g = torch.Generator().manual_seed(0)
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=g))[0]
    k = DistributedSLOSKernel(n, m, occ, workers=2)
    assert isinstance(k, PooledSLOSKernel)
    try:
        v = k.amplitudes(U.numpy())
        ref_k = FockSLOSKernel(n, m, occ, dtype=torch.complex128)
        refp = ref_k.probs(U).numpy()
        assert abs(v.norm2() - 1) < 1e-12
        assert np.abs(v.probs() - refp).max() < 1e-14
        occs = np.asarray(ref_k.layers.final_occupations())
        assert np.abs(v.probabilities_of(occs[:7]) - refp[:7]).max() < 1e-14
        assert abs(v.probability(occs[3]) - refp[3]) < 1e-14
        mref = np.bincount(occs[:, 2], weights=refp, minlength=n + 1)
        assert np.abs(v.marginal(2) - mref).max() < 1e-12
        assert abs(v.top(1)[1][0] - refp.max()) < 1e-14
        s = v.sample(20000, seed=1)
        emp = np.bincount(s[:, 2], minlength=n + 1) / len(s)
        assert np.abs(emp - mref).max() < 0.02
        with pytest.raises(MemoryError):
            v.gather(max_bytes=10)
        vp = k.probs(U.numpy())                                             # same handle: amplitudes
        assert abs(vp.probability(occs[3]) - refp[3]) < 1e-14
        assert np.abs(vp.probs() - refp).max() < 1e-14
    finally:
        k.close()
    with pytest.raises(ValueError, match="limits"):
        DistributedSLOSKernel(3, 300, [1, 1, 1] + [0] * 297, workers=2)


def test_spmd_mode_under_mpirun():
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("mpirun not found")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([mpirun, "-np", "3", sys.executable, os.path.join(ROOT, "tests", "slos", "check_distributed.py")],
                       capture_output=True, text=True, timeout=300, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "distributed check: OK" in r.stdout


def test_plan_refuses_and_fits():
    from photonic_engines.slos.distributed import Plan, check_plan, plan
    p = plan(12, 24, 24, cores=48)
    assert 12 * 2 ** 30 < p.vector_bytes < 13 * 2 ** 30                     # M = 8.3e8 amplitudes
    assert 5 < p.seconds < 30                                                # measured 9.4-12.7 s (48 cores)
    assert plan(12, 24, 1, cores=1).seconds > p.seconds
    assert plan(13, 26, 1, cores=48).seconds > 3 * p.seconds                 # 1.7 TB of parent reads
    assert plan(12, 24, 4, mem_per_rank_gb=1).fits is False                  # 4.5 GB per rank
    assert plan(12, 24, 4, mem_per_rank_gb=8).fits is True
    assert Plan(12, 24, 48, 8 * 2 ** 30).min_ranks(8 * 2 ** 30) == 3       # 17.4 GB peak / 7.8 GB
    with pytest.raises(MemoryError, match="node"):
        check_plan(20, 40, 4)                                                # 3.5e15 amplitudes
    assert check_plan(4, 8, 2).per_rank_bytes < 2 ** 30
    with pytest.raises(MemoryError):
        DistributedSLOSKernel(20, 40, [1] * 20 + [0] * 20, workers=2)       # refused before launching


def test_perceval_sampler():
    pcvl = pytest.importorskip("perceval")
    from photonic_engines.adapters.perceval import DistributedSLOSBackend, DistributedSLOSSampler
    m = 6
    c = pcvl.Unitary(pcvl.Matrix.random_unitary(m))
    s = DistributedSLOSSampler(strong=DistributedSLOSBackend(workers=2), seed=3)
    try:
        s.set_circuit(c)
        s.set_input_state(pcvl.BasicState([1, 1, 0, 1, 0, 0]))
        out = s.samples(30)
        assert len(out) == 30 and all(st.n == 3 for st in out)
        proc = pcvl.Processor(DistributedSLOSSampler(strong=DistributedSLOSBackend(workers=2)), c)
        proc.with_input(pcvl.BasicState([1, 1, 0, 1, 0, 0]))
        res = pcvl.algorithm.Sampler(proc).samples(20)["results"]
        assert len(res) == 20
    finally:
        s.close()


def test_perceval_backend():
    pcvl = pytest.importorskip("perceval")
    from photonic_engines.adapters.perceval import DistributedSLOSBackend
    m = 6
    c = pcvl.Unitary(pcvl.Matrix.random_unitary(m))
    b = DistributedSLOSBackend(workers=2)
    try:
        b.set_circuit(c)
        b.set_input_state(pcvl.BasicState([1, 1, 1, 0, 0, 0]))
        d = b.prob_distribution()
        ref = pcvl.Processor("SLOS", c)
        ref.with_input(pcvl.BasicState([1, 1, 1, 0, 0, 0]))
        ref.min_detected_photons_filter(0)
        rp = ref.probs()["results"]
        assert max(abs(d[s] - rp[s]) for s in rp) < 1e-12
        assert abs(b.probability(pcvl.BasicState([1, 1, 1, 0, 0, 0])) - rp[pcvl.BasicState([1, 1, 1, 0, 0, 0])]) < 1e-12
        assert len(b.samples(50)) == 50
    finally:
        b.close()
