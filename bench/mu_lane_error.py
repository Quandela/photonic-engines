"""True fp64 error of the templated mu lane (clifford/cuda_rect.py, CPU emulation) against an EXACT reference, on the
cases of tests/clifford/test_cuda_rect.py (test_templated_mu_lane_every_degree and, with --high,
test_templated_mu_lane_high_degrees). The source of the tolerance `_mu_lane_atol` in that test.

Reference: the rectangular cofactors P[x] = Per_rect(B[:, all columns but x]) by a dynamic programme over the
multiset of distinct rows in exact integer arithmetic (inputs rounded at 2^-62, far below fp64 rounding), prefix over
the columns before x and suffix after it. Reported per case, worst over the test's TPL values and segment counts:

    rel     = max|err| / max|P|
    c_bound = max|err| / (N (D + 1) eps max|P|)      (eps = 2^-53, D = s0 = N - 1 - q; the test allows c = 4)

--nofma compiles the emulation with -ffp-contract=off (no fused multiply-add, as the x86 g++ default); without it the
compiler's default applies (arm64 clang contracts). Needs a C++20 compiler (CXX, default clang++ on macOS, g++
elsewhere); triton is not needed (stubbed when absent). CPU only, ~1 min for the default cases.

    python3 bench/mu_lane_error.py [--nofma] [--high] [--cxx /usr/bin/clang++]
"""
import argparse
import itertools
import math
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
try:
    import triton  # noqa: F401
except ImportError:                                  # the emulation path never calls triton; its import chain does
    tr, tl = types.ModuleType("triton"), types.ModuleType("triton.language")
    tl.constexpr = object
    tr.language = tl
    tr.jit = lambda f=None, **k: (f if f else (lambda g: g))
    tr.autotune = lambda *a, **k: (lambda g: g)
    tr.Config = lambda *a, **k: None
    tr.cdiv = lambda a, b: -(-a // b)
    tr.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    sys.modules["triton"], sys.modules["triton.language"] = tr, tl
from helpers import haar_unitary

from photonic_engines.clifford import cuda_rect

FIX = 62                                             # fixed-point scale of the exact reference: x -> round(x 2^62)
EPS = 2.0 ** -53
EVERY_DEGREE = [(5, 4, (1, 2, 4), False), (8, 6, (1, 2, 8), True), (11, 8, (1, 4, 16), False), (14, 10, (2, 8), True),
                (18, 13, (4, 16), False), (21, 14, (8,), False), (23, 16, (1, 4), True)]
HIGH_DEGREES = [(12, 4, (1, 4), False), (16, 6, (2, 8), True), (20, 9, (4, 16), False), (24, 10, (8,), True),
                (25, 8, (1, 16), False), (26, 5, (4, 16), True), (29, 4, (8, 16), False)]


def exact_cofactors(B: torch.Tensor) -> torch.Tensor:
    """B [q, N] complex128 -> [N]: Per_rect(B[:, all but x]) exactly (on the 2^-62-rounded inputs), correctly rounded.
    State of the DP = remaining count of each distinct row (mixed radix); F the prefix over columns < x, G the suffix
    over columns > x, P[x] = prod(mu!) sum_c F_x(c) G_x(mu - c)."""
    q, N = B.shape
    keys = [tuple(map(tuple, r.tolist())) for r in torch.view_as_real(B)]
    dist = list(dict.fromkeys(keys))
    rep = [keys.index(d) for d in dist]
    mu = [keys.count(d) for d in dist]
    radix = [v + 1 for v in mu]
    strides = [1]
    for r in radix[:-1]:
        strides.append(strides[-1] * r)
    nst = math.prod(radix)
    digits = np.array(list(itertools.product(*[range(r) for r in reversed(radix)])))[:, ::-1]
    Bi = [[(round(float(B[t, n].real) * 2 ** FIX), round(float(B[t, n].imag) * 2 ** FIX)) for n in range(N)]
          for t in rep]

    def step(Fr, Fi, n):
        Nr, Ni = Fr.copy(), Fi.copy()
        for i, st in enumerate(strides):
            m = digits[:, i] > 0
            src = np.nonzero(m)[0] - st
            br, bi = Bi[i][n]
            Nr[m] += Fr[src] * br - Fi[src] * bi
            Ni[m] += Fr[src] * bi + Fi[src] * br
        return Nr, Ni
    zero = np.zeros(nst, dtype=object)
    zero[:] = 0
    Fr, Fi = zero.copy(), zero.copy()
    Fr[0] = 1
    pre = [(Fr, Fi)]
    for n in range(N):
        Fr, Fi = step(Fr, Fi, n)
        pre.append((Fr, Fi))
    Gr, Gi = zero.copy(), zero.copy()
    Gr[0] = 1
    suf = [None] * (N + 1)
    suf[N] = (Gr, Gi)
    for n in range(N - 1, -1, -1):
        Gr, Gi = step(Gr, Gi, n)
        suf[n] = (Gr, Gi)
    comp = (nst - 1) - np.arange(nst)
    fac = math.prod(math.factorial(v) for v in mu)
    out = []
    for x in range(N):
        (Fr, Fi), (Gr, Gi) = pre[x], suf[x + 1]
        re = int(np.sum(Fr * Gr[comp] - Fi * Gi[comp])) * fac
        im = int(np.sum(Fr * Gi[comp] + Fi * Gr[comp])) * fac
        out.append(complex(re / 2 ** (FIX * q), im / 2 ** (FIX * q)))     # int / int: correctly rounded
    return torch.tensor(out, dtype=torch.complex128)


def case(N, q, tpls, repeats, seeds):
    """The test's instance (same seeds); returns (D, rel, c_bound), worst over TPL and segment counts."""
    rng = np.random.default_rng(N * seeds[0] + q)
    U = haar_unitary(2 * N, seed=seeds[1] * N + q)
    S = 2
    A = torch.stack([U[:, torch.randperm(2 * N, generator=torch.Generator().manual_seed(seeds[2] + s))[:N]]
                     for s in range(S)])
    rows = torch.tensor(rng.integers(0, 4 if repeats else 2 * N, size=(S, q)))
    D = N - 1 - q
    B = torch.stack([A[s][rows[s]] for s in range(S)])
    ex = torch.stack([exact_cofactors(B[s]) for s in range(S)])
    scale = float(ex.abs().max())
    err = 0.0
    for tpl in tpls:
        for ttl in (1 << 17, 16):
            got = cuda_rect.rect_cofactors_emul(A, rows, q, tpl=tpl, form="mu", target_total_lanes=ttl)
            err = max(err, float((got - ex).abs().max()))
    return D, err / scale, err / (N * (D + 1) * EPS * scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nofma", action="store_true", help="compile the emulation with -ffp-contract=off")
    ap.add_argument("--high", action="store_true", help="also the high-degree cases (D = 7 .. 24)")
    ap.add_argument("--cxx", help="C++20 compiler for the emulation (sets CXX; e.g. /usr/bin/clang++ on macOS)")
    a = ap.parse_args()
    if a.cxx:
        os.environ["CXX"] = a.cxx
    os.environ.setdefault("PE_KERNEL_CACHE", os.path.join(tempfile.gettempdir(), "pe_mu_lane_error_" + ("nofma" if a.nofma else "fma")))
    if a.nofma:                                      # the build command is cuda_rect's own, with one flag added
        run = subprocess.run
        cuda_rect.subprocess = types.SimpleNamespace(run=lambda cmd, **k: run([cmd[0], "-ffp-contract=off", *cmd[1:]], **k))
    print(f"FMA contraction: {'off' if a.nofma else 'compiler default'}; CXX = {os.environ.get('CXX', 'default')}")
    print("test                N   q   D  repeats   max|err|/max|P|   c = max|err|/(N (D+1) eps max|P|)")
    for name, cases, seeds in [("every_degree", EVERY_DEGREE, (31, 3, 7))] + ([("high_degrees", HIGH_DEGREES, (37, 5, 11))]
                                                                            if a.high else []):
        for N, q, tpls, rep in cases:
            D, rel, c = case(N, q, tpls, rep, seeds)
            print(f"{name:16s} {N:3d} {q:3d} {D:3d}  {int(rep):5d}      {rel:10.2e}        {c:6.2f}", flush=True)


if __name__ == "__main__":
    main()
