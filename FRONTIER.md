# The simulability frontier — current records

n\* is the largest number of photons (or sources) for which one GPU draws **exact** samples at **1 sample per
second**: the crossing of the measured samples/s curve with 1/s, bracketed between two measured sizes and
log-interpolated. Haar-random interferometer, m = 2n modes unless stated, one NVIDIA H100 PCIe, fp64.
Every row is reproduced by the script named in `reproduce/MANIFEST.md`; the run directory holds its log and CSVs.

| input | setting | n\* | algorithm | run | contributor |
|---|---|---|---|---|---|
| single photons (Fock) | m = 2n | 38.3 | Clifford & Clifford 2020, CUDA cofactor kernel | `runs/20261009_141132` | photonic-engines |
| single photons (Fock) | m = n | 42.6 | same | `runs/20261009_141132` | photonic-engines |
| single photons, loss η = 0.9 / 0.8 / 0.7 / 0.5 | m = 2n, n counts sources | 41.2 / 45.0 / 49.2 / 64.2 | same, on the surviving subset | `runs/20261008_071215`, `_085321`, `_104426`, `_192343` | photonic-engines |
| superposed sources cos α\|0⟩ + sin α\|1⟩, α = π/4 | m = 2n, n counts sources | 50.0 | random-source-order chain rule | `runs/20261007_090041` | photonic-engines |
| quantum-dot emitter, linear cluster state | m = 2n | 31.0 | emitter chain rule, CUDA kernel | `runs/20261006_004008` | photonic-engines |
| single photons + one Kerr site, φ = π/2 | m = 2n | 36.1 | rejection over the branch mixture | `runs/20261009_141132` | photonic-engines |
| single photons, adaptive (K = 1 measured level) | m = 2n | 38.3 | prefix chain + exact level marginals | `runs/20261007_022012` | photonic-engines |
| squeezed vacuum (Gaussian), run of S = 10⁴ samples | m = 2n / m = n | 41.5 / 42.9 | pair-sieve loop-hafnian chain rule | `runs/20260925_…` | photonic-engines |

@@RELEASE: rows and values to be checked against the frontier paper's final table (T3) before release; the GBS
run stamp to be completed; contributor wording to be settled.

## How a row changes

A faster exact sampler, or a better algorithm, for any input above moves its row. Open a pull request with the code
and a `reproduce/` script (see `docs/CONTRIBUTING.md`); we run it on the reference GPU under the same protocol, and on
merge the row, the run and the credit are updated. Rows are compared only when they meet the same rule: the best known
exact algorithm for that input, on the same hardware, with the same statistic.
