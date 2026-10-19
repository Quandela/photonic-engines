"""SPMD check of the distributed engine, run under mpirun:
    mpirun -np 3 python3 tests/slos/check_distributed.py
Compares chunked amplitudes, point lookups, marginals, top-k and samples with FockSLOSKernel."""
import itertools
import math
import sys

import numpy as np

from photonic_engines.slos.basis import get_layers
from photonic_engines.slos.distributed import DistributedSLOSKernel, _mpi

MPI = _mpi()
comm = MPI.COMM_WORLD
n, m = 5, 8
occ = [1, 1, 1, 1, 1, 0, 0, 0]
rng = np.random.default_rng(0)
U = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))[0]
k = DistributedSLOSKernel(n, m, occ, comm=comm)
v = k.amplitudes(U)
occs = np.asarray(get_layers(n, m).final_occupations(), dtype=np.int64)


def permanent(A):
    r = range(A.shape[0])
    return sum(np.prod([A[i, p[i]] for i in r]) for p in itertools.permutations(r))


inm = [i for i in range(m) for _ in range(occ[i])]
refp = np.empty(len(occs))
for j, t in enumerate(occs):                       # torch-free reference: permanents
    outm = [i for i in range(m) for _ in range(t[i])]
    norm = math.prod(math.factorial(int(x)) for x in t) * math.prod(math.factorial(x) for x in occ)
    refp[j] = abs(permanent(U[np.ix_(outm, inm)])) ** 2 / norm
full = v.gather()
ok = True
if comm.rank == 0:
    ok &= abs(v.norm2() - 1) < 1e-12 and np.abs(np.abs(full) ** 2 - refp).max() < 1e-14
else:
    v.norm2()
sel = occs[comm.rank::comm.size][:5]
a = v.amplitudes_of(sel)
ok &= np.abs(np.abs(a) ** 2 - refp[[np.where((occs == s).all(1))[0][0] for s in sel]]).max() < 1e-14
marg = v.marginal(0)
mref = np.bincount(occs[:, 0], weights=refp, minlength=n + 1)
ok &= np.abs(marg - mref).max() < 1e-12
top = v.top(3)
ok &= abs(top[1][0] - refp.max()) < 1e-14
s = v.sample(20000, seed=3)
emp = np.bincount(s[:, 0], minlength=n + 1) / len(s)
ok &= (s.sum(1) == n).all() and np.abs(emp - mref).max() < 0.02
# layout: contiguous chunks in rank order covering every layer, owner() consistent, boundary
# patterns increasing in the key order (lexicographic on t[m-1], ..., t[1], then t[0])
for layer in range(n + 1):
    ch = [k.chunk(layer, rank=r) for r in range(comm.size)]
    ok &= ch[0][0] == 0 and sum(c[1] for c in ch) == k.layer_size(layer)
    ok &= all(ch[r][0] + ch[r][1] == ch[r + 1][0] for r in range(comm.size - 1))
    ok &= all(k.owner(ch[r][0] + j, layer) == r for r in range(comm.size) for j in range(ch[r][1]) if j < 3 or j == ch[r][1] - 1)
bnd = k.boundaries()
keys = [tuple(b[::-1]) for b in bnd[:comm.size]]
ok &= all(keys[r] <= keys[r + 1] for r in range(comm.size - 1))
vp = k.probs(U)                                                    # amplitudes handle, not squared twice
ok &= abs(vp.probability(occs[3]) - refp[3]) < 1e-14
pp = vp.probs()
if comm.rank == 0:
    ok &= np.abs(pp - refp).max() < 1e-14
allok = comm.allreduce(bool(ok), op=MPI.LAND)
if comm.rank == 0:
    print("distributed check:", "OK" if allok else "FAILED", k.timings)
sys.exit(0 if allok else 1)
