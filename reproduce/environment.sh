#!/usr/bin/env bash
# Frontier paper, supplement (software library and environment): the GPU environment of the runs
# (driver 580.126.20, CUDA 13.0, Python 3.12.3, PyTorch 2.14.0, Triton 3.8.0).
#                                                                              original run runs/20261006_011034
# Writes environment-gpu.txt (Python, PyTorch and CUDA versions, GPU name and driver, pip freeze) and prints it.
# Compare it with environment-gpu.txt at the repository root. Wall time: about 1 s. --quick does the same.
# Options: --out DIR, --quick, --device cuda|cpu (ignored), -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

run <<'CMD'
{ python3 --version; python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; python3 -m pip freeze; } > "$OUT/environment-gpu.txt" && cat "$OUT/environment-gpu.txt"
CMD
