#!/bin/bash
# Step 0 of the GBSKernels comparison (bench/gbskernels_bench.py): can GBSKernels' CUDA extension (github.com/chaofanzhougit/
# GBSKernels, Apache-2.0, tag v0.2.2; nanobind, needs nvcc) be built on the GPU host WITHOUT touching the host? The host
# has the driver (CUDA 13.0) but no toolkit. Everything goes under $GBSK_HOME (default ~/gbsk, outside the checkout):
# a venv, the CUDA toolkit from pip wheels (nvcc, runtime, cccl; CUDA 13 first, CUDA 12.8 as fallback, the driver runs
# both) merged into one toolkit tree, cmake / ninja / nanobind from pip, the source cloned at the tag. No apt, no sudo,
# no system pip. Ends with a smoke test: haf_batched gpu vs cpu, sampling.sampler.sample gpu vs cpu on 4 squeezed modes.
# Every step prints what it did, so the log is the recipe. Run on the GPU machine, ~10-15 min.
# We run and compare GBSKernels; none of its code enters this repository.
set -u
GBSK_HOME=${GBSK_HOME:-$HOME/gbsk}
TAG=${GBSK_TAG:-v0.2.2}
OUT=${RESULTS_DIR:-$GBSK_HOME}
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
mkdir -p "$GBSK_HOME" || exit 1
step "host: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader)"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "CMAKE_CUDA_ARCHITECTURES=$ARCH"

# The host python has no ensurepip (python3-venv absent, run 20261011_102413): the venv is made without pip and sees the
# system site-packages, so the system pip runs in it and installs into the venv (PEP 668 does not apply inside a venv).
step "venv $GBSK_HOME/env (--without-pip --system-site-packages; the system python is not touched)"
[ -x "$GBSK_HOME/env/bin/python" ] || python3 -m venv --without-pip --system-site-packages "$GBSK_HOME/env" || exit 1
PY=$GBSK_HOME/env/bin/python
$PY -m pip --version || { curl -sSf https://bootstrap.pypa.io/get-pip.py -o "$GBSK_HOME/get-pip.py" && $PY "$GBSK_HOME/get-pip.py" -q && $PY -m pip --version; } || exit 1
$PY -m pip install -q "cmake>=3.26" ninja nanobind numpy scipy mpmath || exit 1
$PY -m pip list --local 2>/dev/null | grep -iE '^(cmake|ninja|nanobind|numpy|scipy|mpmath) '
export PATH=$GBSK_HOME/env/bin:$PATH

# A toolkit tree assembled from the pip wheels: every nvidia/<pkg>/ directory merged into $1 (bin, include, lib, nvvm).
assemble() {
    local dst=$1 SITE=$2; rm -rf "$dst"; mkdir -p "$dst"
    for d in "$SITE"/nvidia/*/; do
        case "$d" in */cu1[0-9]/) ;; */cuda_*/) ;; */curand/) ;; *) continue ;; esac
        echo "  merge $d: $(ls "$d" | tr '\n' ' ')"; [ -d "$d/bin" ] && echo "    bin: $(ls "$d/bin" | tr '\n' ' ')"
        cp -rn "$d"/. "$dst"/ 2>/dev/null
    done
    [ -d "$dst/lib" ] && [ ! -e "$dst/lib64" ] && ln -s lib "$dst/lib64"
    for so in "$dst"/lib/libcudart.so.*; do [ -e "$so" ] && [ ! -e "$dst/lib/libcudart.so" ] && ln -s "$(basename "$so")" "$dst/lib/libcudart.so"; done
    ls "$dst"/lib/libcudart* "$dst"/bin/nvcc 2>&1 | sed 's/^/  /'
}
# nvcc compiles AND runs a kernel on the H100 (checks the driver accepts this toolkit's code)
hello() {
    local cu=$1
    cat > "$GBSK_HOME/hello.cu" <<'EOF'
#include <cstdio>
__global__ void k(int *x) { x[threadIdx.x] = threadIdx.x * 2; }
int main() { int *d, h[4]; cudaMalloc(&d, 16); k<<<1, 4>>>(d); cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
  cudaError_t e = cudaGetLastError(); printf("hello kernel: %d %d %d %d (%s)\n", h[0], h[1], h[2], h[3], cudaGetErrorString(e));
  return (e == cudaSuccess && h[3] == 6) ? 0 : 1; }
EOF
    "$cu/bin/nvcc" --version | tail -2
    "$cu/bin/nvcc" -arch=sm_$ARCH -o "$GBSK_HOME/hello" "$GBSK_HOME/hello.cu" -L"$cu/lib" && LD_LIBRARY_PATH=$cu/lib "$GBSK_HOME/hello"
}

CUDA=""
for v in 13 12; do
    step "CUDA $v toolkit from pip wheels (pip --target $GBSK_HOME/wheels$v: never mixed with torch's nvidia wheels)"
    W=$GBSK_HOME/wheels$v; rm -rf "$W"
    PI="$PY -m pip install -q --no-deps --target $W"
    if [ $v = 13 ]; then
        # CUDA 13 wheels are unsuffixed packages versioned 13.x (installed under nvidia/cu13/), crt headers and nvvm split
        # out. ONE pip call: a second --target call into the same dir skips the existing nvidia/ (run 20261011_102601 lost
        # crt/host_config.h that way)
        $PI "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-cccl==13.0.*" \
            "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-curand>=10.4,<10.5" || continue   # curand: sampler_draw.cu
    else
        $PI "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-runtime-cu12==12.8.*" "nvidia-cuda-cccl-cu12==12.8.*" "nvidia-curand-cu12>=10.3.9,<10.4" || continue
    fi
    ls -d "$W"/*.dist-info | sed 's/^/  /'
    assemble "$GBSK_HOME/cuda$v" "$W"
    if hello "$GBSK_HOME/cuda$v"; then CUDA=$GBSK_HOME/cuda$v; echo "toolkit OK: CUDA $v at $CUDA"; break; fi
    echo "toolkit CUDA $v: nvcc missing or hello kernel failed"
done
[ -n "$CUDA" ] || { echo "RESULT: FAIL - no working nvcc from pip wheels"; exit 1; }

step "source: GBSKernels $TAG"
SRC=$GBSK_HOME/GBSKernels
if [ ! -d "$SRC/.git" ]; then git clone -q --depth 1 --branch "$TAG" https://github.com/chaofanzhougit/GBSKernels "$SRC" \
    || git clone -q https://github.com/chaofanzhougit/GBSKernels "$SRC" || { echo "RESULT: FAIL - clone"; exit 1; }; fi
echo "commit $(git -C "$SRC" rev-parse HEAD) ($(git -C "$SRC" describe --tags --always 2>/dev/null))"
$PY -m pip install -q -e "$SRC" 2>&1 | tail -3 || $PY -m pip install -q "gbskernels==0.2.2"

step "cmake configure + build of bindings/ (nanobind extension gbskernels_ext)"
export CUDACXX=$CUDA/bin/nvcc CUDAToolkit_ROOT=$CUDA LD_LIBRARY_PATH=$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$SRC/bindings/build"
cmake -S "$SRC/bindings" -B "$SRC/bindings/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
    -DPython_EXECUTABLE="$PY" 2>&1 | tail -25 || true
T0=$(date +%s)
cmake --build "$SRC/bindings/build" -j 16 2>&1 | tail -40
ls "$SRC"/bindings/build/gbskernels_ext*.so 2>/dev/null || { echo "RESULT: FAIL - build (toolkit CUDA ${CUDA##*cuda})"; exit 1; }
echo "build time $(( $(date +%s) - T0 )) s"
ldd "$SRC"/bindings/build/gbskernels_ext*.so | grep -iE 'cuda|not found'

step "smoke test (GBSKERNELS_EXT_DIR=$SRC/bindings/build)"
export GBSKERNELS_EXT_DIR=$SRC/bindings/build PYTHONPATH=$SRC
$PY - "$OUT/gbskernels_smoke.json" <<'EOF'
import json, sys, time
import numpy as np
import gbskernels as g
res = {"gpu_available": g.gpu_available(), "backend_kind": g.gpu_backend_kind()}
print("gpu_available", res["gpu_available"], "kind", res["backend_kind"])
assert res["gpu_available"] and "shim" not in str(res["backend_kind"]).lower(), "not a real GPU build"
rng = np.random.default_rng(1)
for d in (8, 16, 20):
    A = rng.normal(size=(64, d, d)) + 1j * rng.normal(size=(64, d, d)); A = A + A.transpose(0, 2, 1)
    t = time.perf_counter(); gpu = np.asarray(g.haf_batched(A, backend="gpu")); tg = time.perf_counter() - t
    t = time.perf_counter(); cpu = np.asarray(g.haf_batched(A, backend="cpu")); tc = time.perf_counter() - t
    rel = float(np.max(np.abs(gpu - cpu) / np.abs(cpu)))
    print(f"haf_batched d={d} B=64: max rel gpu-cpu {rel:.2e}  gpu {tg:.3f}s cpu {tc:.3f}s")
    res[f"haf_d{d}"] = {"rel": rel, "gpu_s": tg, "cpu_s": tc}
from sampling.sampler import sample
M, r = 4, 0.5                     # 4 single-mode squeezed vacua, hbar = 2, quadrature variances e^{-2r} / e^{2r},
a, b = np.exp(-2 * r), np.exp(2 * r)   # placed so the state is valid in both xxpp and xpxp ordering
cov = np.diag([a, b, b, a, b, a, a, b])
for be in ("cpu", "gpu"):
    t = time.perf_counter(); s = sample(cov, 2000, cutoff=6, backend=be, seed=3); dt = time.perf_counter() - t
    print(f"sample {be}: mean photons {s.sum(1).mean():.3f} (expected {M * np.sinh(r) ** 2:.3f}), odd-parity modes {(s % 2).sum()}, {dt:.2f}s")
    res[f"sample_{be}"] = {"mean": float(s.sum(1).mean()), "odd": int((s % 2).sum()), "s": dt}
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("RESULT: OK")
EOF
