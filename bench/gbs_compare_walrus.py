"""Precision of our pair sieve against The Walrus (Apache-2.0) on the same loop hafnians (docs/GAUSSIAN.md §8). CPU.

    python3 bench/gbs_compare_walrus.py 16,20,24,28

Relative error of the loop hafnian lhaf(A, D0) against the long-double sieve of bench/gbs_precision.py, for ours (v1
`lhaf_poly`, v2 `lhaf_poly_multi` with labels) and thewalrus.loop_hafnian (glynn=True / False; and with `reps`, its
repeated-edge pairing — that call's convention for loops with repetitions was not checked, see §8).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbs_precision import counts_for, haar, lhaf_poly_ld
from thewalrus import loop_hafnian

from photonic_engines import GaussianInput
from photonic_engines.gaussian.hafnian import (
    expand,
    lhaf_poly,
    lhaf_poly_multi,
)

for N in [int(x) for x in sys.argv[1].split(",")]:
    for shape in ("free", "1x4"):
        for mi in range(2):
            cnt = counts_for(N, shape); k = len(cnt); m = 2 * k
            st = GaussianInput.smsv(k, m, GaussianInput.r_for_mean(0.5)); B, _, _ = st.bargmann(haar(m, 1000 * mi + N))
            rng = np.random.default_rng(1000 * mi + N); gam = rng.normal(size=k) * 0.5 + 0j
            A, D0 = expand(B[:k, :k], gam, cnt); lab = np.repeat(np.arange(k), cnt)
            ref = complex(lhaf_poly_ld(A, D0, np.zeros(N))[0])
            e = lambda v: abs(complex(v) - ref) / abs(ref)
            v1 = lhaf_poly(A, D0, None)[0].item()
            v2 = lhaf_poly_multi(A, D0[None], np.zeros((1, N)), labels=lab)[0, 0].item()
            Ar = A.copy(); np.fill_diagonal(Ar, D0)
            tg = loop_hafnian(Ar, D0, glynn=True); ti = loop_hafnian(Ar, D0, glynn=False)
            # thewalrus with repetitions (its own collision pairing), Glynn
            Bk = B[:k, :k].copy(); np.fill_diagonal(Bk, gam)
            tr_ = loop_hafnian(Bk, gam, reps=cnt, glynn=True)
            print(f"N={N} {shape} #{mi}: ours v1 {e(v1):.1e}  v2 {e(v2):.1e}  | TW glynn {e(tg):.1e}  "
                  f"incl-excl {e(ti):.1e}  reps+glynn {e(tr_):.1e}", flush=True)
