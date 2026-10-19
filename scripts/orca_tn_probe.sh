#!/bin/bash
# Third-party code (ORCA's CUDA-Q fork, Apache-2.0) is cloned at run time and only read; nothing of it is copied here.
# ORCA cuTensorNet photonic simulator: read-only confirmation
# that at the head of branch photonic_tn_simulator of github.com/orcacomputing/cuda-quantum (1) beam_splitter /
# phase_shift are commented out on the GPU path (NVQIR.cpp, CircuitSimulator.h, photonics_qis.h) and (2) the
# tensor-network sampler maps outcomes with digits {'0','1'} only. Same greps on the sibling branch
# photonics_tn_simulator_lm. Shallow clones under $CPU_SAMP_HOME, no submodules, no build; nothing on the host changes.
set -u
H=${CPU_SAMP_HOME:-$HOME/cpusamp}
mkdir -p "$H" || exit 1
for BR in photonic_tn_simulator photonics_tn_simulator_lm; do
    SRC=$H/cudaq-orca-$BR
    echo; echo "=================== branch $BR"
    [ -d "$SRC/.git" ] || git clone -q --depth 1 --branch "$BR" https://github.com/orcacomputing/cuda-quantum "$SRC" || { echo "clone failed"; continue; }
    git -C "$SRC" log -1 --format='head %H %cd %an: %s' --date=short
    cd "$SRC" || continue
    echo "--- (1a) runtime/nvqir/NVQIR.cpp: photonic QIS functions"
    grep -n -E "beam_splitter|phase_shift|PHOTONIC_(ONE|TWO)_QUDIT|\(plus\)|create\(|annihilate" runtime/nvqir/NVQIR.cpp
    echo "--- (1b) runtime/nvqir/CircuitSimulator.h: photonic virtuals, qudit measurement"
    grep -n -E "beam_splitter|phase_shift|virtual void plus|PHOTONIC_TWO|#undef|measureQudit|meassure|measureQubit" runtime/nvqir/CircuitSimulator.h
    echo "--- (1c) runtime/cudaq/qis/photonics_qis.h: QIS macros and gates"
    grep -n -E "CUDAQ_QIS_PHOTONIC|beam_splitter|phase_shift|\bmz\b" runtime/cudaq/qis/photonics_qis.h | head -60
    echo "--- (2) runtime/nvqir/cutensornet/*: sampler outcome digits"
    grep -rn -E "digits\[|digits =|cutensornetSamplerSample" runtime/nvqir/cutensornet/ | head -20
    echo "--- the branch's photonic example"
    EX=docs/sphinx/examples/cpp/basics/cutensornet_backends_photonics.cpp
    [ -e "$EX" ] && cat -n "$EX" || echo "absent"
    cd - >/dev/null || exit 1
done
echo; echo "RESULT: probe done"
