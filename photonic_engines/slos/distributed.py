"""EXPERIMENTAL — distributed strong simulation (CPU, MPI): SLOS_full with the amplitude vector split across
ranks, O(M_n / P) memory per rank. The numerics are the C library `csrc/slos_dist.c` (from
slos_mpi: colex-chunked vector, batched Alltoallv halo, incremental parent rank); this module
is the engine face: the same contract as `FockSLOSKernel` (`amplitudes(U)` / `probs(U)`), a
`DistributedVector` result that is never gathered unless it fits, and two execution modes:

STATUS: experimental. Validated on one node (n = 13, m = 26 in 61 s on 48 cores), never on
several; complex128 only; API may change. docs/DISTRIBUTED.md has the limits.

  SPMD  the script runs under `mpirun -np P python script.py` (or srun): every rank builds the
        kernel and calls the same methods (collective); results are DistributedVectors.
  pool  a plain python process asks for `workers=P`: P workers are spawned once
        (MPI dynamic process management) and kept for the session; the caller drives them
        with a small command protocol and gets numpy results back (gathered, point lookups,
        marginals, samples). No mpirun, no ranks visible.

    from photonic_engines.slos.distributed import DistributedSLOSKernel
    k = DistributedSLOSKernel(n, m, occupation, workers=4)      # pool mode
    v = k.amplitudes(U)                                          # DistributedVector
    v.gather()            # full vector in FockLayers order (only if it fits; else raises)
    v.probability(t)      # one output pattern
    v.marginal(j)         # photon-number distribution of mode j
    v.sample(shots)       # exact samples from |amp|^2

The C library is compiled on first use with `mpicc` into $PE_KERNEL_CACHE (or the system temp
dir); mpi4py provides the communicator. Complex128 only; no autograd, no batch over U (this is
the memory-scaling engine: n = 12 at m = 24 on one 20 GB node, +1 photon per ~8x ranks).
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from math import comb

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "slos_dist.c")
_CDEF = """
typedef struct slos_dist slos_dist;
slos_dist *slos_dist_create(int fcomm, int n, int m, const int *occ);
void slos_dist_destroy(slos_dist *d);
void slos_dist_set_batch(slos_dist *d, long batch);
long slos_dist_layer_size(const slos_dist *d, int k);
int slos_dist_threads(void);
void slos_dist_chunk(const slos_dist *d, int k, long *lo, long *cnt);
void slos_dist_chunk_of(const slos_dist *d, int k, int r, long *lo, long *cnt);
int slos_dist_owner(const slos_dist *d, int k, long idx);
void slos_dist_boundary(const slos_dist *d, int r, unsigned char *t);
long slos_dist_rank(const slos_dist *d, int k, const unsigned char *t);
void slos_dist_unrank(const slos_dist *d, int k, long j, unsigned char *t);
void slos_dist_unrank_many(const slos_dist *d, int k, long n_idx, const long *idx, unsigned char *out);
void slos_dist_rank_many(const slos_dist *d, int k, long n_rows, const unsigned char *rows, long *out);
int slos_dist_run(slos_dist *d, const double _Complex *U, double _Complex *out, double *timings);
int slos_dist_get(slos_dist *d, int k, const double _Complex *chunk, long nq, const long *q, double _Complex *out);
"""
_lib = None
_ffi = None


def _cache_dir() -> str:
    d = os.environ.get("PE_KERNEL_CACHE") or os.path.join(tempfile.gettempdir(), "photonic_engines_kernels")
    os.makedirs(d, exist_ok=True)
    return d


def _compiler_id(mpicc: str) -> str:
    try:
        return subprocess.run([mpicc, "--version"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    except Exception:
        return "unknown"


def load_library():
    """Compile (once) and load libslos_dist through cffi (ABI mode). The cache key covers the
    source, the compiler (`MPICC`, its --version line) and the flags (`PE_CFLAGS`,
    `PE_OPENMP_FLAG`), so changing any of them rebuilds instead of reusing a stale .so.
    """
    global _lib, _ffi
    if _lib is not None:
        return _ffi, _lib
    import cffi
    src = open(_SRC).read()   # noqa: SIM115  (read once at import, closed by refcount)
    mpicc = os.environ.get("MPICC", "mpicc")
    cflags = os.environ.get("PE_CFLAGS")
    if cflags is None:                          # Apple clang has no -march=native on arm64
        cflags = "-O3" if sys.platform == "darwin" else "-O3 -march=native"
    # OpenMP: on by default except on macOS, where an OpenMP .so next to torch's own libomp
    # aborts the process (duplicate libomp.dylib); PE_OPENMP_FLAG=-fopenmp forces it there.
    omp = os.environ.get("PE_OPENMP_FLAG", "" if sys.platform == "darwin" else "-fopenmp")
    key = f"{src}\n{mpicc}\n{cflags}\n{omp}\n{_compiler_id(mpicc)}"
    tag = hashlib.md5(key.encode()).hexdigest()[:12]     # source + compiler + flags
    so = os.path.join(_cache_dir(), f"libslos_dist_{tag}.so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"                    # several ranks may compile at once
        cmd = [mpicc, *cflags.split(), *([omp] if omp else []), "-shared", "-fPIC", "-o", tmp, _SRC, "-lm"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and omp:                       # Apple clang: no OpenMP -> one thread per rank
            r = subprocess.run([c for c in cmd if c != omp], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("compiling slos_dist.c failed:\n" + r.stderr)
        if os.path.exists(so):
            os.remove(tmp)
        else:
            os.replace(tmp, so)
    ffi = cffi.FFI()
    ffi.cdef(_CDEF)
    _ffi, _lib = ffi, ffi.dlopen(so)
    return _ffi, _lib


def _mpi():
    """Import mpi4py; when libmpi is not on the loader path (pip mpich), point MPI4PY_LIBMPI at
    the library mpicc links against. PE_DEBUG_HANG=<seconds>: dump all Python stacks to stderr
    and exit after that delay (to locate a hang).
    """
    if os.environ.get("PE_DEBUG_HANG"):
        import faulthandler
        faulthandler.dump_traceback_later(float(os.environ["PE_DEBUG_HANG"]), exit=True)
    if "MPI4PY_LIBMPI" not in os.environ:
        try:
            show = subprocess.run([os.environ.get("MPICC", "mpicc"), "-show"], capture_output=True, text=True, check=True).stdout
            for tok in show.split():
                if tok.startswith("-L"):
                    for name in ("libmpi.so.12", "libmpi.so.40", "libmpi.so", "libmpi.12.dylib", "libmpi.40.dylib", "libmpi.dylib"):
                        cand = os.path.join(tok[2:], name)
                        if os.path.exists(cand):
                            os.environ["MPI4PY_LIBMPI"] = cand
                            break
                if "MPI4PY_LIBMPI" in os.environ:
                    break
        except Exception:
            pass
    from mpi4py import MPI
    return MPI


def _comm_world():
    return _mpi().COMM_WORLD


def _check_c_limits(n: int, m: int):
    """The C kernel stores occupations as uint8 and keeps per-mode scratch in 256-entry stack
    arrays: m <= 256 modes, n <= 255 photons.
    """
    if m > 256 or n > 255 or m < 1:
        raise ValueError(f"DistributedSLOSKernel: n={n}, m={m} outside the C kernel's limits (m <= 256, n <= 255)")


# ============================================================================ planning ======
class Plan:
    """Feasibility and cost estimate of a distributed run, from the model fitted on the 48-core
    node (v2 layout, mpich, complex128; docs/DISTRIBUTED.md §3c):
      memory per rank  ≈ 1.4 · 16 · M_n / ranks + baseline
      time             ≈ max(row_ns · M_total / cores, 16 · M_total · n / (BW · nodes)) + 0.1 s
    On a node the run is bound by the random-gather bandwidth (every output row reads its ≤ n
    parents, 16 B each: 240 GB at n = 12, m = 24 → 9.4 s at 25 GB/s), not by ranks or threads.
    Numbers are indicative (±30 %); the point is to refuse impossible requests with sizes.
    """

    ROW_NS = 160.0              # ns per output row on one core (build + gather, v2)
    GATHER_BW = 25e9            # bytes/s of parent reads sustained by one node (48 threads)
    OVERHEAD = 0.1              # s per run (launch, allreduce per layer)
    BASELINE = 200e6            # bytes per rank (MPI + python)

    def __init__(self, n: int, m: int, ranks: int, mem_per_rank_bytes: float | None = None, nodes: int = 1,
                 cores: int | None = None):
        self.n, self.m, self.ranks, self.nodes = int(n), int(m), int(ranks), int(nodes)
        self.cores = int(cores) if cores else (os.cpu_count() or 1) * self.nodes
        self.M = comb(self.n + self.m - 1, self.m - 1)
        self.M_total = sum(comb(k + self.m - 1, self.m - 1) for k in range(self.n + 1))
        self.vector_bytes = 16 * self.M
        self.peak_bytes = 1.4 * self.vector_bytes                  # two live layers
        self.per_rank_bytes = self.peak_bytes / self.ranks + self.BASELINE
        self.mem_per_rank = mem_per_rank_bytes
        self.fits = None if mem_per_rank_bytes is None else self.per_rank_bytes <= mem_per_rank_bytes
        parents = min(self.n, self.m)
        self.gather_bytes = 16.0 * self.M_total * parents
        self.seconds = max(self.ROW_NS * 1e-9 * self.M_total / self.cores,
                           self.gather_bytes / (self.GATHER_BW * self.nodes)) + self.OVERHEAD

    def min_ranks(self, mem_per_rank_bytes: float) -> int:
        """Smallest rank count whose per-rank share of the peak (two live layers) fits mem_per_rank_bytes minus BASELINE."""
        return max(1, math.ceil(self.peak_bytes / max(1.0, mem_per_rank_bytes - self.BASELINE)))

    def __repr__(self):
        gb = 2 ** 30
        fit = "" if self.fits is None else (", fits" if self.fits else ", DOES NOT FIT")
        return (f"Plan(n={self.n}, m={self.m}, ranks={self.ranks}: M_n={self.M:,d}, vector {self.vector_bytes / gb:.1f} GB, "
                f"peak {self.peak_bytes / gb:.1f} GB, per rank {self.per_rank_bytes / gb:.2f} GB{fit}; "
                f"~{self.seconds:.0f} s on {self.nodes} node(s), {self.cores} cores)")


def plan(n: int, m: int, ranks: int, mem_per_rank_gb: float | None = None, nodes: int = 1,
         cores: int | None = None) -> Plan:
    """Estimate memory per rank and time before running; see Plan."""
    return Plan(n, m, ranks, None if mem_per_rank_gb is None else mem_per_rank_gb * 2 ** 30, nodes=nodes, cores=cores)


def available_memory_bytes() -> float | None:
    """MemAvailable of this machine (Linux), total physical memory otherwise, None if unknown."""
    try:
        for line in open("/proc/meminfo"):   # noqa: SIM115  (closed when the generator is exhausted)
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def check_plan(n: int, m: int, ranks: int, nodes: int = 1) -> Plan:
    """Raise MemoryError (with the plan and the rank count that would fit) when the run cannot
    fit the memory available on this machine × nodes. PE_MPI_NOCHECK=1 skips the check.
    """
    avail = available_memory_bytes()
    if avail is None or os.environ.get("PE_MPI_NOCHECK"):
        return Plan(n, m, ranks)
    per_rank = 0.9 * avail * nodes / ranks
    p = Plan(n, m, ranks, per_rank, nodes=nodes)
    if not p.fits:
        gb = 2 ** 30
        raise MemoryError(f"{p}: needs {p.per_rank_bytes / gb:.1f} GB per rank, {per_rank / gb:.1f} GB available per rank on "
                          f"{nodes} node(s); the full vector is {p.vector_bytes / gb:.0f} GB — needs at least "
                          f"{math.ceil(p.peak_bytes / (0.9 * avail))} node(s) of this size (PE_MPI_NOCHECK=1 to override)")
    return p


# ============================================================================ SPMD engine ====
class DistributedVector:
    """This rank's chunk of a layer-n amplitude vector (always amplitudes, never probabilities:
    `probabilities_of` / `probs` / `marginal` square on the fly), with collective accessors.
    Every method is collective over the kernel's communicator (all ranks must call it).
    """

    def __init__(self, kernel: DistributedSLOSKernel, local: np.ndarray, lo: int):
        self.kernel, self.local, self.lo = kernel, local, int(lo)
        self.cnt = int(local.shape[0])
        self.M = kernel.output_size()

    # ---- basis ----
    def occupations(self) -> np.ndarray:
        """[cnt, m] occupations of this rank's chunk (colex order)."""
        return self.kernel.unrank(np.arange(self.lo, self.lo + self.cnt))

    # ---- whole-vector views ----
    def gather(self, order: str = "layers", max_bytes: int = 4 << 30) -> np.ndarray | None:
        """Full amplitude vector on rank 0 (None elsewhere), in `FockLayers` order ("layers") or
        colex order ("colex"). Raises when the vector exceeds max_bytes.
        """
        if max_bytes < self.M * 16:
            raise MemoryError(f"full vector is {self.M * 16 / 2**30:.1f} GB (> {max_bytes / 2**30:.1f} GB); use point "
                              "lookups, marginals or sampling instead")
        comm = self.kernel.comm
        parts = comm.gather(self.local, root=0)
        if comm.rank != 0:
            return None
        full = np.concatenate(parts)
        if order == "colex":
            return full
        return full[self.kernel.layers_permutation()]

    def probs(self, order: str = "layers", max_bytes: int = 4 << 30) -> np.ndarray | None:
        """|amplitude|^2 of the whole vector on rank 0 (see gather)."""
        a = self.gather(order, max_bytes)
        return None if a is None else a.real ** 2 + a.imag ** 2

    def norm2(self) -> float:
        """Squared norm of the whole vector (Allreduce over the local chunks). Collective."""
        comm = self.kernel.comm
        return float(comm.allreduce(float(np.sum(np.abs(self.local) ** 2))))

    # ---- point lookups ----
    def amplitudes_of(self, patterns) -> np.ndarray:
        """Amplitudes of output patterns [P, m] (each rank may pass its own list; returns its own
        answers). Collective.
        """
        pats = np.asarray(patterns, dtype=np.int64).reshape(-1, self.kernel.m)
        idx = self.kernel.rank(pats)
        return self.kernel._get(self.local, idx)

    def probabilities_of(self, patterns) -> np.ndarray:
        """|amplitude|^2 of output patterns [P, m] (see amplitudes_of). Collective."""
        a = self.amplitudes_of(patterns)
        return a.real ** 2 + a.imag ** 2

    def amplitude(self, pattern) -> complex:
        """Amplitude of one output pattern [m]. Collective."""
        return complex(self.amplitudes_of([pattern])[0])

    def probability(self, pattern) -> float:
        """|amplitude|^2 of one output pattern [m]. Collective."""
        return float(self.probabilities_of([pattern])[0])

    # ---- reductions ----
    def marginal(self, mode: int) -> np.ndarray:
        """[n+1] photon-number distribution of one output mode (Allreduce)."""
        occ = self.occupations()[:, mode]
        w = np.abs(self.local) ** 2
        loc = np.bincount(occ, weights=w, minlength=self.kernel.n + 1)
        return self.kernel.comm.allreduce(loc)

    def mean_photon_numbers(self) -> np.ndarray:
        """[m] expected photon number per mode."""
        occ = self.occupations()
        w = np.abs(self.local) ** 2
        return self.kernel.comm.allreduce(w @ occ)

    def top(self, k: int):
        """(occupations [k, m], probabilities [k]) of the k most probable outputs (on every rank)."""
        w = np.abs(self.local) ** 2
        kk = min(k, self.cnt)
        sel = np.argpartition(-w, kk - 1)[:kk] if kk > 0 else np.zeros(0, dtype=np.int64)
        cand = self.kernel.comm.allgather((self.lo + sel, w[sel]))
        idx = np.concatenate([c[0] for c in cand])
        pw = np.concatenate([c[1] for c in cand])
        o = np.argsort(-pw)[:k]
        return self.kernel.unrank(idx[o]), pw[o]

    def sample(self, shots: int, seed: int = 0, loss: float | None = None) -> np.ndarray:
        """Exact samples [shots, m] from |amp|^2 (on every rank): the shots are first assigned to
        ranks by their local mass, then drawn locally. loss: uniform transmission eta, applied
        by binomial thinning of the drawn patterns (docs/LOSS.md).
        """
        out = self._sample(shots, seed)
        if loss is not None and float(loss) < 1.0:
            out = np.random.default_rng(seed + 12345).binomial(out, float(loss))
        return out

    def _sample(self, shots: int, seed: int = 0) -> np.ndarray:
        comm = self.kernel.comm
        w = np.abs(self.local) ** 2
        mass = comm.allgather(float(w.sum()))
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(shots, np.asarray(mass) / sum(mass))
        rng_local = np.random.default_rng(seed + 1 + comm.rank)
        c = int(counts[comm.rank])
        loc = rng_local.choice(self.cnt, size=c, p=w / w.sum()) if c > 0 else np.zeros(0, dtype=np.int64)
        idx = np.concatenate(comm.allgather(self.lo + loc))
        return self.kernel.unrank(idx)


class DistributedSLOSKernel:
    """SLOS_full strong simulation distributed over the ranks of an MPI communicator (SPMD).
    Same contract as FockSLOSKernel: amplitudes(U) / probs(U) over the n-photon output basis,
    but the result is a DistributedVector. With `workers=P` in a single-process session a pool
    of P workers is spawned instead (see PooledSLOSKernel).
    """

    def __new__(cls, n, m, input_occupation, comm=None, workers: int | None = None, **kw):
        """Return a PooledSLOSKernel when `workers` is given without an explicit `comm` (pool mode), else the SPMD kernel."""
        if workers is not None and comm is None:
            return PooledSLOSKernel(n, m, input_occupation, workers=workers, **kw)
        return super().__new__(cls)

    def __init__(self, n: int, m: int, input_occupation: Sequence[int], comm=None, workers=None, batch: int = 1 << 16):
        if workers is not None and comm is None:
            return
        self.n, self.m = int(n), int(m)
        occ = [int(c) for c in input_occupation]
        if len(occ) != self.m or sum(occ) != self.n:
            raise ValueError("input_occupation must have m entries summing to n")
        self.input_occupation = occ
        _check_c_limits(self.n, self.m)
        self.comm = comm if comm is not None else _comm_world()
        self.plan = check_plan(self.n, self.m, self.comm.size)
        self.ffi, self.lib = load_library()
        self._occ_c = self.ffi.new("int[]", occ)
        self._d = self.lib.slos_dist_create(self.comm.py2f(), self.n, self.m, self._occ_c)
        self.lib.slos_dist_set_batch(self._d, int(batch))
        self.timings = {}

    def __del__(self):
        try:
            if getattr(self, "_d", None) is not None:
                self.lib.slos_dist_destroy(self._d)
        except Exception:
            pass

    # ---- basis ----
    def output_size(self) -> int:
        """Size M_n of the n-photon output basis, comb(n + m - 1, m - 1)."""
        return comb(self.n + self.m - 1, self.m - 1)

    def layer_size(self, k: int) -> int:
        """Size of layer k (the k-photon basis) as reported by the C kernel."""
        return int(self.lib.slos_dist_layer_size(self._d, k))

    def chunk(self, k: int | None = None, rank: int | None = None):
        """[lo, lo + cnt) of layer k (default n) owned by this rank (or by `rank`)."""
        lo, cnt = self.ffi.new("long *"), self.ffi.new("long *")
        if rank is None:
            self.lib.slos_dist_chunk(self._d, self.n if k is None else k, lo, cnt)
        else:
            self.lib.slos_dist_chunk_of(self._d, self.n if k is None else k, int(rank), lo, cnt)
        return int(lo[0]), int(cnt[0])

    def owner(self, idx: int, k: int | None = None) -> int:
        """Rank owning colex index `idx` of layer k (default n)."""
        return int(self.lib.slos_dist_owner(self._d, self.n if k is None else k, int(idx)))

    def boundaries(self) -> np.ndarray:
        """[P + 1, m] boundary patterns of the key-range layout (row 0 = vacuum, row P unused)."""
        out = np.zeros((self.comm.size + 1, self.m), dtype=np.uint8)
        for r in range(self.comm.size):
            self.lib.slos_dist_boundary(self._d, r, self.ffi.from_buffer("unsigned char[]", out[r]))
        return out.astype(np.int64)

    @property
    def threads(self) -> int:
        """OpenMP threads the C kernel uses per rank (1 when built without OpenMP)."""
        return int(self.lib.slos_dist_threads())

    def unrank(self, idx, k: int | None = None) -> np.ndarray:
        """Occupations [len(idx), m] (int64) of colex indices `idx` in layer k (default n)."""
        idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int64).reshape(-1))
        out = np.empty((idx.shape[0], self.m), dtype=np.uint8)
        self.lib.slos_dist_unrank_many(self._d, self.n if k is None else k, idx.shape[0],
                                       self.ffi.from_buffer("long[]", idx), self.ffi.from_buffer("unsigned char[]", out))
        return out.astype(np.int64)

    def rank(self, occupations, k: int | None = None) -> np.ndarray:
        """Colex indices [P] (int64) of occupation rows [P, m] in layer k (default n); inverse of unrank."""
        rows = np.ascontiguousarray(np.asarray(occupations, dtype=np.uint8).reshape(-1, self.m))
        out = np.empty(rows.shape[0], dtype=np.int64)
        self.lib.slos_dist_rank_many(self._d, self.n if k is None else k, rows.shape[0],
                                     self.ffi.from_buffer("unsigned char[]", rows), self.ffi.from_buffer("long[]", out))
        return out

    def layers_permutation(self) -> np.ndarray:
        """Perm with full_layers_order = full_colex[perm] (FockLayers.final_occupations order)."""
        from .basis import get_layers
        occ = np.asarray(get_layers(self.n, self.m).final_occupations(), dtype=np.int64)
        return self.rank(occ)

    # ---- engine ----
    def amplitudes(self, U) -> DistributedVector:
        """Collective: this rank's chunk of the layer-n amplitudes for U [m, m]."""
        U = np.ascontiguousarray(np.asarray(U, dtype=np.complex128).reshape(self.m, self.m))
        lo, cnt = self.chunk()
        out = np.empty(max(cnt, 1), dtype=np.complex128)
        tim = self.ffi.new("double[4]")
        self.lib.slos_dist_run(self._d, self.ffi.from_buffer("double _Complex[]", U),
                               self.ffi.from_buffer("double _Complex[]", out), tim)
        self.timings = {"build": tim[0], "comm": tim[1], "compute": tim[2], "halo_entries": int(tim[3]), "threads": self.threads}
        return DistributedVector(self, out[:cnt], lo)

    def probs(self, U) -> DistributedVector:
        """Same handle as `amplitudes(U)` (the vector always holds amplitudes); read
        probabilities through `.probs()` (gathered), `.probabilities_of(patterns)`, `.marginal`…
        """
        return self.amplitudes(U)

    def _get(self, local: np.ndarray, idx: np.ndarray) -> np.ndarray:
        idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int64).reshape(-1))
        out = np.empty(max(idx.shape[0], 1), dtype=np.complex128)
        local = np.ascontiguousarray(local, dtype=np.complex128)
        self.lib.slos_dist_get(self._d, self.n, self.ffi.from_buffer("double _Complex[]", local), idx.shape[0],
                               self.ffi.from_buffer("long[]", idx), self.ffi.from_buffer("double _Complex[]", out))
        return out[:idx.shape[0]]


# ============================================================================ pool mode =====
_WORKER = "photonic_engines.slos._mpi_worker"


def _send_msg(f, obj):
    import pickle
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(len(data).to_bytes(8, "little"))
    f.write(data)
    f.flush()


def _recv_msg(f):
    import pickle
    head = f.read(8)
    if len(head) < 8:
        raise EOFError("worker pool closed")
    n = int.from_bytes(head, "little")
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("worker pool closed")
        buf += chunk
    return pickle.loads(bytes(buf))


class PooledSLOSKernel:
    """Single-process face of the distributed engine: launches `workers` MPI processes once
    (kept for the life of the object) and drives DistributedSLOSKernel on them. Results come
    back as numpy arrays; the amplitude vector stays on the workers.

    transport="mpirun" (default): the pool is `mpirun -np P python -m _mpi_worker` started as a
    subprocess; commands and replies go through a localhost socket that rank 0 connects to
    (pickled messages; mpirun's own stdio forwarding is not used — Open MPI 5 sends the ranks'
    stdout to the terminal), rank 0 broadcasts to the other ranks. Works with any MPI, no
    dynamic process management.
    transport="spawn": MPI_Comm_spawn from this process (needs an MPI whose singleton spawn
    works — mpich does, Open MPI on macOS often does not).
    """

    def __init__(self, n: int, m: int, input_occupation: Sequence[int], workers: int = 2, batch: int = 1 << 16,
                 transport: str | None = None, mpirun: str | None = None, mpirun_args: Sequence[str] = ()):
        self.n, self.m = int(n), int(m)
        self.input_occupation = [int(c) for c in input_occupation]
        self.workers = int(workers)
        _check_c_limits(self.n, self.m)
        self.plan = check_plan(self.n, self.m, self.workers)
        self.transport = transport or os.environ.get("PE_MPI_TRANSPORT", "mpirun")
        self._inter = None
        self._proc = None
        if self.transport == "spawn":
            MPI = _mpi()
            self._MPI = MPI
            self._inter = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", _WORKER, "--spawned"], maxprocs=self.workers)
        else:
            import shutil
            import socket
            exe = mpirun or os.environ.get("MPIRUN") or shutil.which("mpirun") or shutil.which("mpiexec")
            if exe is None:
                raise RuntimeError("mpirun/mpiexec not found (set MPIRUN=...)")
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // self.workers)))
            env.setdefault("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            cmd = [exe, "-np", str(self.workers), *mpirun_args, sys.executable, "-m", _WORKER, "--connect", f"127.0.0.1:{port}"]
            self._proc = subprocess.Popen(cmd, env=env)
            srv.settimeout(float(os.environ.get("PE_MPI_CONNECT_TIMEOUT", "120")))
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                self._proc.kill()
                raise RuntimeError(
                    "the MPI workers did not connect within the timeout (mpirun failed to start them?)") from None
            finally:
                srv.close()
            conn.settimeout(None)
            self._sock = conn
            self._rd, self._wr = conn.makefile("rb"), conn.makefile("wb")
        self._call("build", n=self.n, m=self.m, occ=self.input_occupation, batch=int(batch))
        self._has_vector = False

    def _call(self, cmd: str, **kw):
        msg = {"cmd": cmd, **kw}
        if self._inter is not None:
            self._inter.bcast(msg, root=self._MPI.ROOT)
            return self._inter.recv(source=0, tag=0)
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("worker pool is not running (mpirun exited); check the MPI installation")
        _send_msg(self._wr, msg)
        reply = _recv_msg(self._rd)
        if isinstance(reply, dict) and reply.get("__error__"):
            raise RuntimeError("worker error: " + reply["__error__"])
        return reply

    def close(self):
        """Shut the worker pool down and release its transport.

        Sends "stop", then disconnects the intercommunicator (spawn) or waits for mpirun to exit and closes the socket,
        killing the subprocess if it does not stop within 30 s. Idempotent.
        """
        if self._inter is not None:
            with contextlib.suppress(Exception):
                self._call("stop")
            self._inter.Disconnect()
            self._inter = None
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    _send_msg(self._wr, {"cmd": "stop"})
                    self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
            with contextlib.suppress(Exception):
                self._sock.close()
            self._proc = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def output_size(self) -> int:
        """Size M_n of the n-photon output basis, comb(n + m - 1, m - 1)."""
        return comb(self.n + self.m - 1, self.m - 1)

    # ---- engine: the vector stays on the workers; accessors query it ----
    def amplitudes(self, U) -> PooledVector:
        """Run the layer-n amplitudes for U [m, m] on the workers and return a handle to the vector they keep.

        Records the run's timings in `self.timings`; the amplitudes themselves are never transferred here.
        """
        U = np.ascontiguousarray(np.asarray(U, dtype=np.complex128).reshape(self.m, self.m))
        r = self._call("run", U=U)
        self.timings = r["timings"]
        self._has_vector = True
        return PooledVector(self, r["norm2"])

    def probs(self, U) -> PooledVector:
        """Same handle as `amplitudes(U)` (the vector always holds amplitudes); square through the PooledVector accessors."""
        return self.amplitudes(U)


class PooledVector:
    """Handle to the amplitude vector living on the pool workers."""

    def __init__(self, kernel: PooledSLOSKernel, norm2: float):
        self.kernel, self._norm2 = kernel, float(norm2)
        self.M = kernel.output_size()

    def norm2(self) -> float:
        """Squared norm of the vector, as reported by the workers at run time (no round trip)."""
        return self._norm2

    def gather(self, order: str = "layers", max_bytes: int = 4 << 30) -> np.ndarray:
        """Full amplitude vector in `FockLayers` order ("layers") or colex order ("colex"). Raises when it exceeds max_bytes."""
        if max_bytes < self.M * 16:
            raise MemoryError(f"full vector is {self.M * 16 / 2**30:.1f} GB (> {max_bytes / 2**30:.1f} GB)")
        return self.kernel._call("gather", order=order, max_bytes=int(max_bytes))

    def probs(self, order: str = "layers", max_bytes: int = 4 << 30) -> np.ndarray:
        """|amplitude|^2 of the whole vector (see gather)."""
        a = self.gather(order, max_bytes)
        return a.real ** 2 + a.imag ** 2

    def amplitudes_of(self, patterns) -> np.ndarray:
        """Amplitudes [P] of output patterns [P, m], looked up on the workers."""
        return self.kernel._call("get", patterns=np.asarray(patterns, dtype=np.int64).reshape(-1, self.kernel.m))

    def probabilities_of(self, patterns) -> np.ndarray:
        """|amplitude|^2 [P] of output patterns [P, m]."""
        a = self.amplitudes_of(patterns)
        return a.real ** 2 + a.imag ** 2

    def amplitude(self, pattern) -> complex:
        """Amplitude of one output pattern [m]."""
        return complex(self.amplitudes_of([pattern])[0])

    def probability(self, pattern) -> float:
        """|amplitude|^2 of one output pattern [m]."""
        return float(self.probabilities_of([pattern])[0])

    def marginal(self, mode: int) -> np.ndarray:
        """[n+1] photon-number distribution of one output mode, reduced on the workers."""
        return self.kernel._call("marginal", mode=int(mode))

    def mean_photon_numbers(self) -> np.ndarray:
        """[m] expected photon number per mode, reduced on the workers."""
        return self.kernel._call("mean")

    def top(self, k: int):
        """(occupations [k, m], probabilities [k]) of the k most probable outputs."""
        return self.kernel._call("top", k=int(k))

    def sample(self, shots: int, seed: int = 0, loss: float | None = None) -> np.ndarray:
        """Exact samples [shots, m] from |amp|^2, drawn on the workers.

        loss: uniform transmission eta, applied here by binomial thinning of the returned patterns (docs/LOSS.md).
        """
        out = self.kernel._call("sample", shots=int(shots), seed=int(seed))
        if loss is not None and float(loss) < 1.0:
            out = np.random.default_rng(seed + 12345).binomial(out, float(loss))
        return out
