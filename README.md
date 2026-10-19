<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/brand/spin_dark_banner.png">
  <img alt="photonic-engines" src="img/brand/spin_light_banner.png" width="480">
</picture>

# photonic-engines

Exact simulation of linear-optical quantum devices on GPUs: strong simulation (output probabilities) and exact
sampling for single photons, superposed sources, quantum-dot emitters (photonic cluster and GHZ states), squeezed
light, adaptive and non-linear interferometers, and time-bin loop devices. All engines share one set of input-state
types and one model of loss and partial distinguishability.

The library backs the measurements of the *simulability frontier*: the largest device, per input and
architecture, whose output one GPU reproduces exactly at a given rate. Current records: [`FRONTIER.md`](FRONTIER.md).

## Install

```bash
pip install photonic-engines               # core: torch + numpy (CPU engines, torch samplers)
pip install "photonic-engines[cuda]"       # + triton, cupy: the GPU sampling kernels
pip install "photonic-engines[perceval]"   # + Perceval adapters
```

From a clone: `pip install -e ".[dev]"`. The GPU samplers need an NVIDIA GPU with CUDA ≥ 12 (the reference
measurements use an H100 PCIe); `environment-gpu.txt` pins the environment they were taken with.

## Quick start

```python
import torch
from photonic_engines.slos import FockSLOSKernel
from photonic_engines.clifford import fock_sample

U = torch.linalg.qr(torch.randn(12, 12, dtype=torch.complex128))[0]

# strong simulation: every output probability for |1,0,1,0,1,0,0,...> (differentiable in U)
k = FockSLOSKernel(n=3, m=12, input_occupation=[1, 0] * 3 + [0] * 6, mask="fock", device="cpu")
p = k.probs(U.to(torch.complex64))          # [C(14,3)] over k.layers.final_states

# exact sampling: 10 000 output occupations for 6 photons in modes 0..5 (GPU)
occ = fock_sample(U.cuda(), in_modes=list(range(6)), shots=10_000,
                  generator=torch.Generator(device="cuda").manual_seed(0))   # [10000, 12]

# loss (brightness x transmission x detection = eta) and partial distinguishability, in every engine
ld  = k.probs(U.to(torch.complex64), loss=0.8)                                # distribution over 0..3 detected photons
occ = fock_sample(U.cuda(), list(range(6)), 10_000, loss=0.8, indist=0.95)
```

Emitter states are declared once and carried as a bond index, not as their 2^n Fock components:

```python
import math
from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.slos import EmitterSLOSKernel
from photonic_engines.clifford import emitter_sample

prep = spin_gate(math.pi / 2, 0)
lc   = EmitterMPS.caterpillar(4, [prep] + [spin_gate(math.pi / 2, 0)] * 3)    # linear cluster, m = 8
U8   = torch.linalg.qr(torch.randn(8, 8, dtype=torch.complex128))[0]
p    = EmitterSLOSKernel(lc, device="cpu").probs(U8.to(torch.complex64))
occ  = emitter_sample(lc, U8, shots=1000, backend="torch")                    # backend="triton" on a GPU
```

`docs/TUTORIAL.md` walks through every engine with runnable examples.

## Engines

| module | what it does | design note |
|---|---|---|
| `slos/` | strong simulation (SLOS), batched and differentiable; emitter, superposed and Gaussian inputs; single-amplitude permanents; additive-error estimator for emitter inputs | `docs/SLOS_IMPLEMENTATION.md`, `docs/ESTIMATE.md` |
| `clifford/` | exact samplers of the Clifford & Clifford family: Fock, emitter, superposed sources; CUDA and Triton kernels | `docs/ALGORITHMS.md`, `docs/EMITTER_SAMPLER.md`, `docs/SUPERPOSITION.md` |
| `gaussian/` | exact Gaussian boson sampling with photon-number detection (loop-hafnian chain rule) | `docs/GAUSSIAN.md` |
| `adaptive.py` | adaptive linear optics: mid-circuit photon counting and feed-forward | `docs/ADAPTIVE.md` |
| `nonlinear.py` | single-mode photon-number phases (Kerr sites) between interferometers | `docs/NONLINEAR.md` |
| `sequence.py` | emitter trains through a static interferometer, driven tick by tick | `docs/SEQUENCE.md` |
| `timeloop/` | time-bin loop devices, single photons, emitters and squeezed light through loops | `docs/TIME_LOOP.md`, `docs/HYBRID.md` |
| `imperfections.py` | loss and partial distinguishability, shared by every engine | `docs/LOSS.md`, `docs/DISTINGUISHABILITY.md` |
| `adapters/` | Perceval circuits and backends; a MerLin-style layer for loop devices | — |

`docs/ARCHITECTURE.md` maps the package for contributors. Further notes: `docs/DESIGN.md` (the original design and
roadmap), `docs/HARDWARE_TARGET.md` (where lossy photonic devices sit against the exact samplers),
`docs/GAUSSIAN_SURVEY.md` (exact GBS algorithms and codes surveyed), `docs/MEMORY_FRONTIER.md`, `docs/DISTRIBUTED.md`
(experimental MPI engine).

The spin-register machine is a separate package, `spin-register`, built on photonic-engines.

## Reproducing the results

`reproduce/` holds one script per table and figure of the papers that use this code; `reproduce/MANIFEST.md` lists
every quoted number with its script, command, hardware, expected wall time and the run that produced it. Each script
runs on a local GPU and has a `--quick` mode that finishes in minutes. `runs/` holds the logs and CSVs of the original
runs.

## Tests

```bash
pytest -q                    # what the installed extras allow; tests needing a missing extra are skipped
pytest -q --runslow          # + the Triton CPU-interpreter checks
pytest -q -m gpu             # GPU-only sampler checks
```

## Contributing — and the frontier

We want these simulators to get faster, and the frontier with them. If you improve an engine, or have a competing
exact simulator:

- **we benchmark every submitted improvement or competing code** on the protocol of [`FRONTIER.md`](FRONTIER.md), on
  the reference GPU, and publish the result;
- **we keep the frontier current**: a better exact sampler moves its row, with the credit to its author.

How to submit: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Citing

If you use photonic-engines in your work, please cite the paper that introduces it:

```bibtex
@article{senellart2026frontiers,
  title   = {The simulability frontiers of photonic quantum computing},
  author  = {Senellart, Jean and Stott, Benjamin and Mezher, Rawad and Wein, Stephen and
             Cuenca Gomez, Esperanza and Senellart, Pascale and Emeriau, Pierre-Emmanuel and Mansfield, Shane},
  journal = {arXiv preprint arXiv:@@RELEASE},
  year    = {2026}
}
```

and, for the exact samplers beyond Fock inputs (superposed sources, emitter states, adaptive and non-linear
circuits):

```bibtex
@article{senellart2026exact,
  title   = {Exact sampling beyond {Fock} inputs and linear optics: photon-number superpositions, photonic
             matrix-product states, adaptive and non-linear circuits},
  author  = {Senellart, Jean and Wein, Stephen and Mezher, Rawad and Emeriau, Pierre-Emmanuel},
  journal = {arXiv preprint arXiv:@@RELEASE},
  year    = {2026}
}
```

## Licence

MIT — see [`LICENSE`](LICENSE).
