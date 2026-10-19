#!/bin/bash
# Third-party code (ORCA's CUDA-Q fork, Apache-2.0) comes from scripts/orca_tn_build.sh; nothing of it is copied here.
# ORCA cuTensorNet photonic simulator, first contact after the build (scripts/orca_tn_build.sh): how a photonic kernel
# runs on this branch (photonics_tn_simulator_lm, 3d66214). Read-only greps of its photonic API (C++ photonic_qis.h,
# the installed Python layer), its photonic examples, then a two-photon Hong-Ou-Mandel kernel (one photon in each of
# two modes, a 50:50 beam splitter: only |2,0> and |0,2> may come out) on each target, each attempt in its own python
# process with its exception printed. Nothing is installed or changed.
set -u
H=${TN_HOME:-$HOME/cpusamp/cudaq}
SRC=${TN_SRC:-$HOME/cpusamp/cudaq-orca-photonics_tn_simulator_lm}
CUDA=$H/cuda
CUQ=$(find "$H/wheels" -maxdepth 3 -name custatevec.h -path '*include*' | head -1); CUQ=${CUQ%/include/*}
CUT=$(find "$H/wheels" -maxdepth 3 -name cutensor.h -path '*include*' | head -1); CUT=${CUT%/include/*}
export LD_LIBRARY_PATH=$CUDA/lib:$CUQ/lib:$CUT/lib:$H/openblas/lib:$H/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$H/install${PYTHONPATH:+:$PYTHONPATH} PATH=$H/install/bin:$H/llvm/bin:$PATH

echo "=== C++ photonic QIS (runtime/cudaq/qis/photonic_qis.h): functions"
grep -n -E "^ *(inline |template|void |std::|auto |[a-z_]+\()|mz|measure|mpnr" "$SRC/runtime/cudaq/qis/photonic_qis.h" | head -60
echo "=== installed Python layer: photonic names"
grep -rn -E "beam_splitter|phase_shift|mpnr|def create|qudit|photon" "$H/install/cudaq" --include=*.py | grep -v "^.*#" | head -30
echo "=== the branch's photonic examples and tests"
find "$SRC/docs" "$SRC/python/tests" "$SRC/unittests" "$SRC/targettests" -iname '*photon*' 2>/dev/null | head -30

TRY=$(mktemp -d)
for meas in mz mpnr; do      # cudaq.kernel reads the kernel's source: one file per measurement name, no exec
cat > "$TRY/hom_$meas.py" <<EOF
import sys, numpy as np, cudaq
target = sys.argv[1]
cudaq.set_target(target)
print(f"--- target {target}, measurement $meas", flush=True)

@cudaq.kernel
def hom_state():
    q = [qudit(3) for _ in range(2)]
    create(q[0])
    create(q[1])
    beam_splitter(np.pi / 4, q[0], q[1])

print("get_state:", flush=True)
cudaq.get_state(hom_state).dump()

@cudaq.kernel
def hom():
    q = [qudit(3) for _ in range(2)]
    create(q[0])
    create(q[1])
    beam_splitter(np.pi / 4, q[0], q[1])
    $meas(q)

print("sample(1000):", dict(cudaq.sample(hom, shots_count=1000).items()), flush=True)
EOF
done
for target in orca-photonics tensornet tensornet-mps; do
    for meas in mz mpnr; do
        timeout 300 python3 "$TRY/hom_$meas.py" "$target" 2>&1 | tail -12
    done
done

# run 20261012_021255: on tensornet(-mps) the Python AST bridge rejects qudit ("unsupported call in list
# comprehension"); orca-photonics runs a Python-only handler (mz: "no qubits", mpnr undefined). The photonic TN path
# is C++: the branch's C++ examples, compiled with nvq++ for each target and run.
echo "=== docs: photonics.rst, how a target is chosen"
grep -n -i -E "target|tensornet|nvq\+\+|mz|measure|qudit" "$SRC/docs/sphinx/using/backends/sims/photonics.rst" | head -30
for ex in photonics_tbi_sample photonics; do
    echo "=== C++ example $ex.cpp"; cat -n "$SRC/docs/sphinx/targets/cpp/$ex.cpp" | head -70
done
echo "=== unittests/photonics/PhotonicsTester.cpp, the HOM test"
sed -n '46,87p' "$SRC/unittests/photonics/PhotonicsTester.cpp"
cd "$TRY" || exit 1
# run 20261012_021431: without --library-mode the MLIR frontend rejects mpnr / crashes; the examples say
# "nvq++ --library-mode --target orca-photonics" (kernels compiled as plain C++, calling the simulator directly)
for target in tensornet tensornet-mps orca-photonics; do
    for ex in photonics photonics_tbi_sample; do
        echo "--- nvq++ --library-mode --target $target $ex.cpp"
        rm -f "$TRY/$ex.$target"
        timeout 600 nvq++ --library-mode --target "$target" "$SRC/docs/sphinx/targets/cpp/$ex.cpp" -o "$TRY/$ex.$target" 2>&1 | tail -6
        [ -x "$TRY/$ex.$target" ] && { /usr/bin/time -f "  wall %e s" timeout 300 "$TRY/$ex.$target" 2>&1 | tail -15; }
    done
done
echo "RESULT: try done"
