# Contributing

Improvements are welcome: faster kernels, better algorithms, new inputs, fixes, documentation. This file covers two
things: how to submit a speed-up that may move a row of [`FRONTIER.md`](../FRONTIER.md), and how to set up, test and
lint any change. `docs/ARCHITECTURE.md` says where things live and why.

## Submitting an improvement to the frontier

1. **Open a pull request** with the code change (or, for a competing simulator, a script that installs and runs it
   from its public source; we do not vendor third-party code).
2. **Add a `reproduce/` script** for the row you claim to move, in the style of the existing ones: it runs the bench
   directly on a local GPU, writes its CSVs to `--out`, and has a `--quick` mode that finishes in minutes. Put the
   output of your `--quick` run in the PR description, with your GPU model.
3. **We re-measure it on the reference GPU** (one NVIDIA H100 PCIe) with the protocol below, and post the run.
4. **On merge**, the row of `FRONTIER.md`, its run and the credit are updated. A competing code that is faster but
   not merged is listed with its own row and a link.

The benchmark rule: rows are compared only when each runs **the best known exact algorithm for its input**, on the
**same hardware**, with the **same statistic**. The protocol: Haar-random interferometer, m = 2n unless the row says
otherwise, exact samples (no approximation, no post-selection beyond what the row states), n\* = the crossing of the
measured samples/s with 1/s, bracketed between two measured sizes with enough shots that the mean is stable (heavy
tailed costs — lossy and superposed inputs — need more; the existing rows state theirs). A sampler is checked for
exactness against an independent oracle (below) before it is timed.

## Setup

```bash
python -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install -e ".[dev]"                                              # pytest, pytest-cov, ruff
pip install -e ".[perceval]"                                         # adapter tests
pip install -e ".[cuda]"                                             # triton + cupy, on a GPU machine
pip install "triton>=3.0"                                            # CPU machine: the kernel tests
```

Python 3.10 or later. The core needs torch and numpy only; every optional path skips itself when its dependency is
missing. **Install triton even without a GPU**: the Clifford kernels run under its CPU interpreter, and without it four
kernel modules drop out of the suite silently. `PE_REQUIRE_FULL_SUITE=1` turns a missing optional dependency into a
failure instead of a skip.

## Test tiers

```bash
pytest -q                          # the default suite, CPU
pytest -q --cov=photonic_engines   # the same, with the coverage gate (floor: 85 %)
pytest -q --runslow -m slow        # the Triton CPU-interpreter sampler checks: 1-3 min each
pytest -q                          # on a CUDA machine: the `gpu` marks stop skipping
```

* **default** — everything that runs in seconds on a CPU; CI runs it on every push.
* **`slow`** — the sampler correctness scripts under the Triton CPU interpreter; run them when you touch `clifford/`.
* **`gpu`** — skipped without a CUDA device. A change to a kernel is not finished until it has run on a GPU.
* **`check_*.py` scripts** — `tests/clifford/check_*.py` are runnable programs; `tests/clifford/test_scripts.py` wraps
  them.

### Writing a test

Module-level `pytest.importorskip` for optional dependencies, plain `assert`, deterministic seeds, complex128 wherever
a numerical claim is asserted, helpers from `tests/helpers.py`. Two rules matter more here than in ordinary code:

* **Derive tolerances, do not pick them.** For a sampling test the tolerance comes from the shot noise; for a
  numerical identity, from the dtype.
* **An engine is tested against an independent oracle, not against itself**: `slos/reference.py` (permanents),
  `clifford/emitter_reference.py`, `timeloop/circuit.py` (the unrolled unitary) and Perceval. A new fast path is
  validated against the slow path it replaces.

## Lint

`ruff check .` is the gate; configuration in `pyproject.toml`. **The formatter is not run**: the numerical kernels are
written in a compact, column-aligned style whose shape and dtype comments are the record of the tensor contract, and
the formatter destroys it. Match the style of the file you are editing. Docstrings are required in the library
(`photonic_engines/`), not in tests, benches or examples; a docstring states shapes, dtypes, normalisation, device
constraints and memory cost whenever the signature does not.

## Adding an engine, a kernel or an adapter

* **A new engine** shares `states.py`, `masks.py` and `imperfections.py` — extend them rather than adding a parallel
  vocabulary — and its `eta = 1` path reduces exactly to the lossless one (tested).
* **A new GPU kernel** ships with a CPU-reachable correctness test: a Triton kernel runs under the interpreter (`slow`
  tier); a cupy/NVRTC kernel goes in the `gpu` tier and in the coverage `omit` list with a comment saying why.
* **A new adapter** goes under `adapters/` and imports its third-party dependency there only; nothing under
  `adapters/` is imported by a core module.

`docs/TUTORIAL.md` is executable: `tests/test_tutorial.py` runs its CPU examples as written, so a signature change that
appears in the tutorial breaks a test — update both together.
