#!/bin/bash
# Third-party code (Piquasso, Apache-2.0) is installed from PyPI at run time; nothing of it is copied here.
# Piquasso 8.0.1 in its own venv $PQ_HOME (default ~/cpusamp/pq-env) with numba pinned to a version it runs with, for
# bench/cpu_sampler_worker.py. Under numba 0.68 its Hessenberg hafnian aborts (LLVM "symbol not found",
# run 20261011_123409). Each candidate pin is installed into a fresh venv and kept if a 4-mode squeezed state measured
# with ParticleNumberMeasurement runs. No sudo, nothing system-wide; the system pip installs into the venv (--python).
set -u
PQ=${PQ_HOME:-$HOME/cpusamp/pq-env}
mkdir -p "$(dirname "$PQ")" || exit 1
for pin in "numba==0.61.2 numpy<2.3" "numba==0.62.* numpy<2.4" "numba==0.63.* numpy<2.5" "numba==0.60.0 numpy<2.1"; do
    echo "=== $(date +%H:%M:%S) piquasso 8.0.1 with $pin"
    rm -rf "$PQ"; python3 -m venv --without-pip "$PQ" || exit 1
    # shellcheck disable=SC2086
    python3 -m pip --python "$PQ/bin/python" install -q "piquasso==8.0.1" $pin || { echo "install failed"; continue; }
    if "$PQ/bin/python" - <<'EOF'
import numpy as np, piquasso as pq
with pq.Program() as p:
    pq.Q(0) | pq.Squeezing(0.6)
    pq.Q(1) | pq.Squeezing(0.6)
    pq.Q(0, 1, 2, 3) | pq.Interferometer(np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))[0])
    pq.Q(0, 1, 2, 3) | pq.ParticleNumberMeasurement()
s = pq.GaussianSimulator(d=4, config=pq.Config(seed_sequence=1)).execute(p, shots=200).samples
print("smoke: 200 shots, mean N", np.asarray(s).sum(1).mean())
EOF
    then
        python3 -m pip --python "$PQ/bin/python" list 2>/dev/null | grep -iE '^(piquasso|numba|llvmlite|numpy|scipy) '
        echo "RESULT: OK ($pin)"; exit 0
    fi
done
echo "RESULT: FAIL - no numba pin runs piquasso 8.0.1"; exit 1
