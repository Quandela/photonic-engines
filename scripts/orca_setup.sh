#!/bin/bash
# ORCA's loop-progressive-simulator (github.com/ORCAComputing/loop-progressive-simulator, Apache-2.0; Novak et al.,
# arXiv:2411.16873) cloned under $CPU_SAMP_HOME (default ~/cpusamp) with its requirements.txt (unpinned, as shipped)
# in its own venv orca-env, for bench/orca_loop_worker.py. No sudo, nothing
# system-wide; the system pip installs into the venv (--python). We run and compare; none of its code is copied.
set -u
H=${CPU_SAMP_HOME:-$HOME/cpusamp}
SRC=$H/loop-progressive-simulator
PY=$H/orca-env/bin/python
mkdir -p "$H" || exit 1
[ -d "$SRC/.git" ] || git clone -q https://github.com/ORCAComputing/loop-progressive-simulator "$SRC" || { echo "RESULT: FAIL - clone"; exit 1; }
echo "commit $(git -C "$SRC" rev-parse HEAD) ($(git -C "$SRC" log -1 --format=%cd --date=short))"
[ -x "$PY" ] || python3 -m venv --without-pip "$H/orca-env" || exit 1
python3 -m pip --python "$PY" install -q -r "$SRC/requirements.txt" || { echo "RESULT: FAIL - requirements"; exit 1; }
python3 -m pip --python "$PY" list 2>/dev/null | grep -iE '^(numpy|numba|scipy|sympy) '
"$PY" -c "import sys; sys.path.insert(0, '$SRC/loop_progressive_simulator'); import bscircuits, step_simulator_nosym; print('import OK')" || exit 1
echo "RESULT: OK"
