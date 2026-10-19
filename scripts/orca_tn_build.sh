#!/bin/bash
# Third-party code (CUDA-Q and ORCA's fork of it, Apache-2.0; LLVM, zlib, OpenBLAS, OpenSSL, curl under their own
# licences) is fetched and built at run time; nothing of it is copied here.
# Job-local source build of ORCA's cuTensorNet photonic simulator: branch photonics_tn_simulator_lm
# (head 3d66214, 2026-02-18) of github.com/orcacomputing/cuda-quantum. That branch is a CUDA-Q fork with photonic gates
# wired to the tensor-network simulator (scripts/orca_tn_probe.sh); its commits change the Quake dialect and
# the nvq++ frontend, so the whole of CUDA-Q is built, with the LLVM of its submodule, by ITS scripts
# (scripts/build_llvm.sh, build_cudaq.sh).
# Everything goes under $TN_HOME (default ~/cpusamp/cudaq): tools from pip, the CUDA 13 toolkit and cuQuantum 26.1.0 /
# cuTensor from pip wheels, LLVM, pybind11, the CUDA-Q install. Its install_prerequisites.sh is NOT run (system prefixes,
# apt). No sudo, nothing system-wide.
# Staged and resumable: each stage leaves $TN_HOME/.done_<stage>; `bash orca_tn_build.sh <stage>` stops after that stage
# (llvm | cudaq | smoke). Every stage prints its duration (the log is the recipe); full build logs are in $TN_HOME/*.log.
set -u
H=${TN_HOME:-$HOME/cpusamp/cudaq}
BR=photonics_tn_simulator_lm
SRC=${TN_SRC:-$HOME/cpusamp/cudaq-orca-$BR}
UNTIL=${1:-smoke}
J=${TN_JOBS:-$(nproc)}
mkdir -p "$H" || exit 1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
done_() { touch "$H/.done_$1"; echo "stage $1 done in $(( $(date +%s) - T0 )) s"; }
have() { [ -e "$H/.done_$1" ]; }
PIP="python3 -m pip"

step "host"
nproc; free -g | head -2; df -h "$HOME" | tail -1; gcc --version | head -1
python3 -c "import sysconfig, os; h = sysconfig.get_paths()['include']; print('Python.h', h, os.path.exists(h + '/Python.h'))"

step "source: $SRC (branch $BR)"
T0=$(date +%s)
if ! have src; then
    [ -d "$SRC/.git" ] || git clone -q --depth 1 --branch "$BR" https://github.com/orcacomputing/cuda-quantum "$SRC" || exit 1
    git -C "$SRC" submodule update --init --recursive --depth 1 --jobs 8 2>&1 | tail -5 || exit 1
    done_ src
fi
git -C "$SRC" log -1 --format='head %H %cd' --date=short; git -C "$SRC" submodule status | head -20

step "tools (cmake, ninja, lit from pip, --target $H/tools)"
T0=$(date +%s)
if ! have tools; then
    $PIP install -q --target "$H/tools" "cmake>=3.26,<4" ninja lit || exit 1
    done_ tools
fi
export PATH=$H/tools/bin:$PATH PYTHONPATH=$H/tools${PYTHONPATH:+:$PYTHONPATH}
cmake --version | head -1; ninja --version

step "CUDA 13 toolkit + cuQuantum 26.1.0 + cuTensor from pip wheels (one --target call: run 20261011_102601)"
T0=$(date +%s)
W=$H/wheels; CUDA=$H/cuda
if ! have cuda; then
    TK=("nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*"
        "nvidia-nvvm==13.0.*" "nvidia-cublas==13.*" "nvidia-cusolver==12.*" "nvidia-cusparse==12.*" "nvidia-nvjitlink==13.0.*"
        "nvidia-curand>=10.4,<10.5")
    # cuQuantum 26.1.0 = the branch's "update to cuQuantum 26.1.0" (98befb4); cuquantum-cu13 26.1.0's own pins
    rm -rf "$W"; $PIP install -q --no-deps --target "$W" "${TK[@]}" "custatevec-cu13==1.12.0" "cutensornet-cu13==2.11.0" \
        "cudensitymat-cu13==0.4.0" "cutensor-cu13>=2.3,<3" || exit 1
    ls "$W"; ls -d "$W"/*.dist-info | sed 's/^/  /'
    rm -rf "$CUDA"; mkdir -p "$CUDA"
    for d in "$W"/nvidia/*/; do cp -rn "$d"/. "$CUDA"/ 2>/dev/null; done      # cu13/, cublas/, ... merged
    [ -e "$CUDA/lib64" ] || ln -s lib "$CUDA/lib64"
    for so in "$CUDA"/lib/lib*.so.*; do b=$(basename "$so"); l=${b%%.so.*}.so; [ -e "$CUDA/lib/$l" ] || ln -s "$b" "$CUDA/lib/$l"; done
    done_ cuda
fi
"$CUDA/bin/nvcc" --version | tail -1
CUQ=$(find "$W" -maxdepth 3 -name custatevec.h -path '*include*' | head -1); CUQ=${CUQ%/include/*}
CUT=$(find "$W" -maxdepth 3 -name cutensor.h -path '*include*' | head -1); CUT=${CUT%/include/*}
echo "cuquantum prefix $CUQ: $(ls "$CUQ" 2>/dev/null | tr '\n' ' ')"; echo "cutensor prefix $CUT: $(ls "$CUT" 2>/dev/null | tr '\n' ' ')"
for so in "$CUQ"/lib/lib*.so.* "$CUT"/lib/lib*.so.*; do [ -e "$so" ] || continue; b=$(basename "$so"); l=${b%%.so.*}.so; [ -e "$(dirname "$so")/$l" ] || ln -s "$b" "$(dirname "$so")/$l"; done
[ -n "$CUQ" ] && [ -n "$CUT" ] || { echo "RESULT: FAIL - cuQuantum / cuTensor headers not found in the wheels"; exit 1; }

step "zlib 1.3.1 from source, static with -fPIC, into $H/zlib (CUDA-Q's prerequisite; the system libz.a is not PIC)"
# run 20261012_002751: libclang-cpp.so failed to link against /usr/lib/x86_64-linux-gnu/libz.a ("relocation R_X86_64_PC32
# ... recompile with -fPIC"). CUDA-Q's install_prerequisites.sh builds zlib this way into ZLIB_INSTALL_PREFIX.
T0=$(date +%s)
grep -n -i "zlib" "$SRC/scripts/build_llvm.sh" "$SRC/scripts/build_cudaq.sh" | head -12
if ! have zlib; then
    mkdir -p "$H/zlib-src" && curl -fsSL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz \
        | tar xz -C "$H/zlib-src" || exit 1
    ( cd "$H/zlib-src/zlib-1.3.1" && CFLAGS="-fPIC -O2" ./configure --prefix="$H/zlib" --static && make -j "$J" && make install ) \
        > "$H/zlib_build.log" 2>&1 || { tail -20 "$H/zlib_build.log"; echo "RESULT: FAIL - zlib"; exit 1; }
    done_ zlib
fi
# minizip from zlib's contrib (run 20261012_020516: rest_server "Minizip from zLib NOT found"). CUDA-Q's
# install_prerequisites.sh builds it with autoreconf + configure --disable-shared into ZLIB_INSTALL_PREFIX (lib
# libminizip.a, headers include/minizip/); the same files compiled directly here (no autotools needed), static and PIC.
if ! have minizip; then
    MZ=$H/zlib-src/zlib-1.3.1/contrib/minizip
    mkdir -p "$H/zlib/include/minizip" "$H/minizip-obj" || exit 1
    for c in ioapi unzip zip mztools; do
        cc -c -fPIC -O2 -I"$H/zlib/include" -o "$H/minizip-obj/$c.o" "$MZ/$c.c" || { echo "RESULT: FAIL - minizip"; exit 1; }
    done
    ar rcs "$H/zlib/lib/libminizip.a" "$H/minizip-obj"/*.o || exit 1
    cp "$MZ"/{ioapi,unzip,zip,mztools,crypt}.h "$H/zlib/include/minizip/" || exit 1
    done_ minizip
fi
grep -n -B12 'Minizip from zLib NOT found' "$SRC/runtime/cudaq/platform/default/rest_server/CMakeLists.txt" | head -16
ls "$H/zlib/lib" "$H/zlib/include/minizip"
export ZLIB_INSTALL_PREFIX=$H/zlib ZLIB_ROOT=$H/zlib CMAKE_PREFIX_PATH=$H/zlib${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}

# pybind11 is installed by build_llvm.sh into $H/pybind11 on its first run only; a rerun did not find it (run
# 20261011_205727: "Could not find ... pybind11Config.cmake"): point CMake at it by environment (LLVM and CUDA-Q stages)
[ -d "$H/pybind11/share/cmake/pybind11" ] && export pybind11_DIR=$H/pybind11/share/cmake/pybind11 \
    pybind11_ROOT=$H/pybind11 CMAKE_PREFIX_PATH=$H/pybind11${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
step "LLVM of the submodule ($SRC/tpls/llvm) by build_llvm.sh, -j $J, into $H/llvm (log $H/llvm_build.log)"
T0=$(date +%s)
if ! have llvm; then
    # build_llvm.sh cherry-picks LLVM commits absent from the depth-1 submodule ("bad object 9494bd8...", run
    # 20261011_163044): fetch each sha it names with its parent; the submodule is reset first (that run applied patches)
    git -C "$SRC/tpls/llvm" reset -q --hard; git -C "$SRC/tpls/llvm" clean -fdq
    for c in $(grep -ohE '\b[0-9a-f]{40}\b' "$SRC/scripts/build_llvm.sh" | sort -u); do
        git -C "$SRC/tpls/llvm" cat-file -e "$c^{commit}" 2>/dev/null || git -C "$SRC/tpls/llvm" fetch -q --depth 2 origin "$c" || exit 1
        echo "llvm cherry-pick source $c: $(git -C "$SRC/tpls/llvm" log -1 --format='%h %s' "$c")"
    done
    # a configure cache from a failed run keeps pybind11_DIR-NOTFOUND: set aside (where build_llvm.sh puts its folder
    # is printed: its log named $SRC/tpls/llvm/$H/llvm-build/logs)
    grep -n "LLVM_BUILD_FOLDER\|build_dir\|cd .*build" "$SRC/scripts/build_llvm.sh" | head -8
    for f in "$H/llvm-build/CMakeCache.txt" "$SRC/tpls/llvm/$H/llvm-build/CMakeCache.txt"; do
        [ -f "$f" ] && mv -f "$f" "$f.stale" && echo "set aside $f"
    done
    export GIT_AUTHOR_NAME=build GIT_AUTHOR_EMAIL=build@localhost GIT_COMMITTER_NAME=build GIT_COMMITTER_EMAIL=build@localhost
    ( cd "$SRC" && LLVM_INSTALL_PREFIX=$H/llvm PYBIND11_INSTALL_PREFIX=$H/pybind11 LLVM_SOURCE=$SRC/tpls/llvm \
        LLVM_BUILD_FOLDER=$H/llvm-build Python3_EXECUTABLE=python3 bash scripts/build_llvm.sh -j "$J" -v ) > "$H/llvm_build.log" 2>&1
    echo "zlib resolved by LLVM's configure (P 10-12: must be $H/zlib, not the system libz.a):"
    grep -h "^ZLIB_\|^LLVM_ENABLE_ZLIB" "$SRC/tpls/llvm/$H/llvm-build/CMakeCache.txt" "$H/llvm-build/CMakeCache.txt" 2>/dev/null
    grep -n -iE "error|fatal" "$H/llvm_build.log" | head -20; tail -30 "$H/llvm_build.log"
    [ -x "$H/llvm/bin/llvm-config" ] || { echo "RESULT: FAIL - LLVM"; exit 1; }
    done_ llvm
fi
"$H/llvm/bin/llvm-config" --version
[ "$UNTIL" = llvm ] && { echo "RESULT: OK (llvm)"; exit 0; }

step "OpenBLAS 0.3.28 from source into $H/openblas (CUDA-Q's BLAS prerequisite; no Fortran: C LAPACK)"
# run 20261012_014614: build_cudaq.sh's configure failed "Could NOT find BLAS (missing: BLAS_LIBRARIES)"
# (runtime/cudaq/algorithms/optimizers/ensmallen). CUDA-Q's install_prerequisites.sh builds a BLAS; job-local here.
T0=$(date +%s)
grep -n -i "blas" "$SRC/scripts/build_cudaq.sh" "$SRC/scripts/install_prerequisites.sh" 2>/dev/null | head -12
echo "prerequisite prefixes build_cudaq.sh reads (each one absent here is a candidate next failure):"
grep -ohE '[A-Z_]+_INSTALL_PREFIX' "$SRC/scripts/build_cudaq.sh" "$SRC/scripts/install_prerequisites.sh" 2>/dev/null | sort -u | tr '\n' ' '; echo
if ! have blas; then
    mkdir -p "$H/openblas-src" && curl -fsSL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.28/OpenBLAS-0.3.28.tar.gz \
        | tar xz -C "$H/openblas-src" || exit 1
    ( cd "$H/openblas-src/OpenBLAS-0.3.28" && make -j "$J" NOFORTRAN=1 C_LAPACK=1 USE_OPENMP=0 NUM_THREADS=64 \
        && make PREFIX="$H/openblas" NOFORTRAN=1 install ) > "$H/openblas_build.log" 2>&1 \
        || { tail -20 "$H/openblas_build.log"; echo "RESULT: FAIL - OpenBLAS"; exit 1; }
    for d in "$H/openblas" "$H/openblas/lib"; do ln -sf "$H/openblas/lib/libopenblas.a" "$d/libblas.a"; done
    done_ blas
fi
ls "$H/openblas/lib"
export BLAS_INSTALL_PREFIX=$H/openblas BLA_VENDOR=OpenBLAS CMAKE_PREFIX_PATH=$H/openblas${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
export LD_LIBRARY_PATH=$H/openblas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

step "OpenSSL 3.0.15 + curl 8.10.1 from source, static and PIC, into $H/openssl, $H/curl (CUDA-Q's REST prerequisites)"
# run 20261012_020259: libcudaq-common.so failed to link, "undefined symbol: cudaq::RestClient::~RestClient()": the REST
# client is compiled only when CURL / OpenSSL are found. CUDA-Q's install_prerequisites.sh builds both this way.
T0=$(date +%s)
grep -n -A3 -i "curl\b.*configure\|openssl.*config\b\|\./configure\|./Configure" "$SRC/scripts/install_prerequisites.sh" 2>/dev/null | head -16
if ! have rest; then
    mkdir -p "$H/rest-src"
    curl -fsSL https://github.com/openssl/openssl/releases/download/openssl-3.0.15/openssl-3.0.15.tar.gz | tar xz -C "$H/rest-src" || exit 1
    ( cd "$H/rest-src/openssl-3.0.15" && ./Configure linux-x86_64 no-shared no-tests -fPIC --prefix="$H/openssl" --libdir=lib \
        && make -j "$J" && make install_sw ) > "$H/openssl_build.log" 2>&1 || { tail -20 "$H/openssl_build.log"; echo "RESULT: FAIL - OpenSSL"; exit 1; }
    curl -fsSL https://github.com/curl/curl/releases/download/curl-8_10_1/curl-8.10.1.tar.gz | tar xz -C "$H/rest-src" || exit 1
    ( cd "$H/rest-src/curl-8.10.1" && CFLAGS="-fPIC -O2" ./configure --prefix="$H/curl" --disable-shared --enable-static --with-pic \
        --with-openssl="$H/openssl" --with-zlib="$H/zlib" --without-libpsl --without-zstd --without-brotli --without-nghttp2 \
        --without-libidn2 --without-librtmp --disable-ldap --disable-ldaps && make -j "$J" && make install ) \
        > "$H/curl_build.log" 2>&1 || { tail -20 "$H/curl_build.log"; echo "RESULT: FAIL - curl"; exit 1; }
    done_ rest
fi
ls "$H/openssl/lib" "$H/curl/lib"
export OPENSSL_INSTALL_PREFIX=$H/openssl OPENSSL_ROOT_DIR=$H/openssl CURL_INSTALL_PREFIX=$H/curl CURL_ROOT=$H/curl \
    CMAKE_PREFIX_PATH=$H/curl:$H/openssl${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}

# CUDA-Q links with -fuse-ld=lld; gcc looks for ld.lld on PATH ("collect2: fatal error: cannot find 'ld'", run
# 20261012_020051): the job-local LLVM built lld
export PATH=$H/llvm/bin:$PATH
command -v ld.lld && ld.lld --version | head -1

step "CUDA-Q by build_cudaq.sh into $H/install (tests off, warnings not errors; log $H/cudaq_build.log)"
T0=$(date +%s)
if ! have cudaq; then
    # BUILD-SYSTEM PATCH (not their simulator code), run 20261012_014812 / probe run 20261012_015905: CMake 3.31's
    # FindCUDAToolkit gives CUDA 13 two include dirs (include;include/cccl), and the branch's unquoted
    # $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}> then splits across list elements, so the generate step fails
    # ("Target cudaq INTERFACE_INCLUDE_DIRECTORIES ... prefixed in the source directory"). Quoting the expression
    # keeps it one generator expression. Applied to every unquoted occurrence; the diff is printed.
    for f in $(grep -rl --include=CMakeLists.txt '[^"]\$<BUILD_INTERFACE:\${CUDAToolkit_INCLUDE_DIRS}>' "$SRC/runtime" "$SRC/python" 2>/dev/null); do
        sed -i -E 's/([^"])\$<BUILD_INTERFACE:\$\{CUDAToolkit_INCLUDE_DIRS\}>/\1"$<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>"/' "$f"
        echo "build-system patch (quoted CUDAToolkit_INCLUDE_DIRS genex): $f"
    done
    git -C "$SRC" diff --stat; git -C "$SRC" diff -- '*CMakeLists.txt' | head -30
    # a failed configure (run 20261012_014614) cached BLAS_LIBRARIES-NOTFOUND: set its cache aside
    for f in "$SRC/build/CMakeCache.txt" "$SRC"/build*/CMakeCache.txt; do
        [ -f "$f" ] && mv -f "$f" "$f.stale" && echo "set aside $f"
    done
    ( cd "$SRC" && CUDAQ_INSTALL_PREFIX=$H/install LLVM_INSTALL_PREFIX=$H/llvm CUQUANTUM_INSTALL_PREFIX=$CUQ \
        CUTENSOR_INSTALL_PREFIX=$CUT CUDACXX=$CUDA/bin/nvcc CUDA_HOME=$CUDA CUDAQ_BUILD_TESTS=FALSE CUDAQ_WERROR=OFF \
        PYBIND11_INSTALL_PREFIX=$H/pybind11 LD_LIBRARY_PATH=$CUDA/lib:$CUQ/lib:$CUT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
        bash scripts/build_cudaq.sh -j "$J" -v ) > "$H/cudaq_build.log" 2>&1
    echo "BLAS / CURL / OpenSSL resolved by CUDA-Q's configure:"
    grep -h "^BLAS_\|^CURL_\|^OPENSSL_\|^CUDAQ_ENABLE_REST" "$SRC"/build*/CMakeCache.txt 2>/dev/null | head -12
    grep -n -iE "error|fatal" "$H/cudaq_build.log" | head -20; tail -40 "$H/cudaq_build.log"
    ls "$H/install/lib" | grep -iE 'nvqir|tensornet' | head -20
    [ -e "$H/install/bin/nvq++" ] || { echo "RESULT: FAIL - CUDA-Q"; exit 1; }
    done_ cudaq
fi
[ "$UNTIL" = cudaq ] && { echo "RESULT: OK (cudaq)"; exit 0; }

step "smoke: python import, targets, the branch's photonics examples"
export LD_LIBRARY_PATH=$CUDA/lib:$CUQ/lib:$CUT/lib:$H/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$H/install:$PYTHONPATH PATH=$H/install/bin:$PATH
python3 -c "import cudaq; print('cudaq', cudaq.__version__); print([t.name for t in cudaq.get_targets()])"
for ex in "$SRC"/docs/sphinx/targets/python/photonics*.py; do
    echo "--- $ex"; head -60 "$ex"; timeout 300 python3 "$ex" 2>&1 | tail -15
done
echo "RESULT: OK (smoke)"
