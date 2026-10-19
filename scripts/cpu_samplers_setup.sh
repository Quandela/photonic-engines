#!/bin/bash
# Third-party code (The Walrus, Apache-2.0) is installed from PyPI at run time; nothing of it is copied here.
# The Walrus 0.22.0 in its own venv under $CPU_SAMP_HOME (default ~/cpusamp, outside the checkout), for
# bench/cpu_sampler_worker.py. The venv does not see the system
# site-packages (their numpy / numba pins never meet the host's torch stack); the host python has no ensurepip, so the
# system pip installs into it with --python (get-pip as fallback). No sudo, nothing system-wide. Piquasso has its own
# venv with numba pinned (scripts/piquasso_setup.sh: it aborts under this venv's numba 0.68, run 20261011_123409).
set -u
H=${CPU_SAMP_HOME:-$HOME/cpusamp}
mkdir -p "$H" || exit 1
[ -x "$H/env/bin/python" ] || python3 -m venv --without-pip "$H/env" || exit 1
PY=$H/env/bin/python
PIP="python3 -m pip --python $PY"
$PIP --version || { curl -sSf https://bootstrap.pypa.io/get-pip.py -o "$H/get-pip.py" && $PY "$H/get-pip.py" -q && PIP="$PY -m pip"; } || exit 1
$PIP install -q "thewalrus==0.22.0" || { echo "RESULT: FAIL - thewalrus"; exit 1; }
$PIP list 2>/dev/null | grep -iE '^(thewalrus|numpy|numba|scipy) '
$PY -c "import thewalrus.samples, numba; print('thewalrus OK, numba threads', numba.config.NUMBA_NUM_THREADS)" || exit 1
nproc; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket'
echo "RESULT: OK"
