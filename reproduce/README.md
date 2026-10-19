# Reproducing the published numbers

Each `reproduce/<name>.sh` reproduces one table or figure group of the papers; [MANIFEST.md](MANIFEST.md) maps every
published value to its script, original run (`runs/<stamp>/`), result files and wall time.

- Run from the repository root with the package installed: `bash reproduce/frontier_headlines.sh` (`-h` describes a script).
- Outputs (CSV / JSON, with the same names as in `runs/<stamp>/`) go to `out/<name>/`, or to `--out DIR`.
- `--quick` runs the same benches at reduced size in minutes: a smoke test, not the published numbers.
- Full mode needs one NVIDIA GPU with fp64 (the numbers were measured on one H100 PCIe), CUDA, PyTorch, Triton and
  CuPy as in `environment-gpu.txt`; most scripts take hours. `--device cpu` applies only to benches with a CPU path.
