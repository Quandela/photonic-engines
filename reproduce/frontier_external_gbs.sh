#!/usr/bin/env bash
# Frontier paper, supplement Table S1, Sec. S1.3 and Table S2 (other Gaussian boson samplers on the same instances).
#   - GBSKernels v0.2.2 (GPU): 4.6 ms / 75 ms / 0.25 s / 1.7 s / 12.5 s per sample against ours 0.24 / 0.39 / 0.59 /
#     0.74 / 0.98 ms at n = 2-6 (ratios 19 ... 13,000; its n* ~ 4.7-5.0); its law at the floor for n <= 4
#                    original runs runs/20261011_105434 (law n = 4, timing), runs/20261011_111022 (CUDA path)
#   - The Walrus 0.22.0 on 24 CPU cores: n* = 27.7 as shipped (not exact), 23.8 with an exact setting (+-1); the
#     default cutoff biases the law by up to 6.8 sigma at n = 3
#                    original runs runs/20261011_123902 (as shipped), runs/20261011_130055 (diagnosis),
#                    runs/20261011_131100 (exact setting)
#   - Piquasso 8.0.1 on 24 CPU cores at cutoff 12: n* = 35.3 (+-1.5); the default cutoff 5 is biased
#                    original runs runs/20261011_150903 (cutoff 12), runs/20261011_132646 (default cutoff)
# Third-party code: GBSKernels is built under ~/gbsk by scripts/gbskernels_build.sh (CUDA toolkit wheels, cmake,
# nanobind); The Walrus goes into a virtual environment ~/cpusamp/env (scripts/cpu_samplers_setup.sh); Piquasso into
# ~/cpusamp/pq-env (scripts/piquasso_setup.sh; pins numba 0.61.2, llvmlite 0.44.0, numpy 2.2.6). The CPU timings used
# the 24 cores (AMD EPYC 9334) of the H100 host.
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe host: about 1 h (464 + 354 s,
# 798 + 327 + 527 s, 730 + 364 s), plus the first builds.
# --quick: the same setups, then each sampler's law check and timing at n <= 12 with small shot counts and budgets.
# Options: --out DIR, --quick, --device cuda|cpu (ignored: recorded per command), -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
# GBSKernels
run <<'CMD'
export GBSK_SRC=$HOME/gbsk/GBSKernels GBSKERNELS_EXT_DIR=$HOME/gbsk/GBSKernels/bindings/build LD_LIBRARY_PATH=$HOME/gbsk/cuda13/lib GBSK_SITE=$HOME/gbsk/env/lib/python3.12/site-packages GPY=python3; ls $HOME/gbsk/GBSKernels/bindings/build/gbskernels_ext*.so >/dev/null 2>&1 || timeout 1500 bash scripts/gbskernels_build.sh; timeout 900 $GPY bench/gbskernels_bench.py check --n 4 --shots 2000; timeout 1800 $GPY bench/gbskernels_bench.py time --n 2,3,4,5,6,7,8,10,12 --budget 60 --csv "$OUT/gbskernels_derisk.csv"
CMD
run <<'CMD'
export GBSK_SRC=$HOME/gbsk/GBSKernels GBSKERNELS_EXT_DIR=$HOME/gbsk/GBSKernels/bindings/build LD_LIBRARY_PATH=$HOME/gbsk/cuda13/lib GBSK_SITE=$HOME/gbsk/env/lib/python3.12/site-packages GPY=python3; ls $HOME/gbsk/GBSKernels/bindings/build/gbskernels_ext*.so >/dev/null 2>&1 || timeout 1500 bash scripts/gbskernels_build.sh; timeout 1200 $GPY bench/gbskernels_bench.py gpucheck --n 2,3,4 --shots 200
CMD
# The Walrus
run <<'CMD'
export CPY=$HOME/cpusamp/env/bin/python; timeout 600 bash scripts/cpu_samplers_setup.sh; timeout 900 python3 bench/gbskernels_bench.py check --sampler walrus --n 3 --shots 20000 --budget 200; timeout 900 python3 bench/gbskernels_bench.py check --sampler walrus --n 8,12 --shots 2000 --budget 200; timeout 2100 python3 bench/gbskernels_bench.py time --sampler walrus --no-ours --n 10,12,14,16,18,20,22,24,26,28,30,32 --budget 60 --csv "$OUT/cpu_walrus_time.csv"
CMD
run <<'CMD'
export CPY=$HOME/cpusamp/env/bin/python; W='timeout 300 python3 bench/gbskernels_bench.py check --sampler walrus --shots 20000 --budget 200'; $W --n 2,3,4,5; $W --n 3 --seed-once; $W --n 3 --walrus-cutoff 12; $W --n 3 --seed-once --walrus-cutoff 12
CMD
run <<'CMD'
export CPY=$HOME/cpusamp/env/bin/python; timeout 300 bash scripts/cpu_samplers_setup.sh; X='--sampler walrus --walrus-cutoff 12 --walrus-max-ns'; timeout 240 python3 bench/gbskernels_bench.py check $X --n 3 --shots 20000 --budget 120; timeout 300 python3 bench/gbskernels_bench.py check $X --n 12 --shots 1000 --budget 150; timeout 1800 python3 bench/gbskernels_bench.py time $X --no-ours --n 12,14,16,18,20,22,24,26,28,30 --budget 60 --csv "$OUT/cpu_walrus_exact_time.csv"
CMD
# Piquasso
run <<'CMD'
export CPY=$HOME/cpusamp/pq-env/bin/python; timeout 600 bash scripts/piquasso_setup.sh || exit 1; B='python3 bench/gbskernels_bench.py'; P='--sampler piquasso --walrus-cutoff 12 --kill-shot-s 400'; timeout 300 $B check $P --n 3 --shots 20000 --budget 150; timeout 300 $B check $P --n 12 --shots 1000 --budget 150; timeout 2700 $B time $P --no-ours --n 24,26,28,30,32,36 --budget 90 --max-shot-s 450 --csv "$OUT/cpu_piquasso_c12_time.csv"
CMD
run <<'CMD'
export CPY=$HOME/cpusamp/pq-env/bin/python; timeout 900 bash scripts/piquasso_setup.sh || exit 1; B='python3 bench/gbskernels_bench.py'; timeout 400 $B check --sampler piquasso --n 2,3,4,6 --shots 2000 --budget 90; timeout 300 $B check --sampler piquasso --max-ns --n 3 --shots 20000 --budget 150; timeout 300 $B check --sampler piquasso --max-ns --n 12 --shots 1000 --budget 150; timeout 2100 $B time --sampler piquasso --max-ns --no-ours --n 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 --budget 60 --csv "$OUT/cpu_piquasso_time.csv"
CMD
else
run <<'CMD'
export GBSK_SRC=$HOME/gbsk/GBSKernels GBSKERNELS_EXT_DIR=$HOME/gbsk/GBSKernels/bindings/build LD_LIBRARY_PATH=$HOME/gbsk/cuda13/lib GBSK_SITE=$HOME/gbsk/env/lib/python3.12/site-packages GPY=python3; ls $HOME/gbsk/GBSKernels/bindings/build/gbskernels_ext*.so >/dev/null 2>&1 || bash scripts/gbskernels_build.sh; $GPY bench/gbskernels_bench.py check --n 2 --shots 500; $GPY bench/gbskernels_bench.py time --n 2,3 --budget 10 --csv "$OUT/quick_gbskernels.csv"; $GPY bench/gbskernels_bench.py gpucheck --n 2 --shots 50
CMD
run <<'CMD'
export CPY=$HOME/cpusamp/env/bin/python; bash scripts/cpu_samplers_setup.sh; python3 bench/gbskernels_bench.py check --sampler walrus --n 3 --shots 2000 --budget 30; python3 bench/gbskernels_bench.py time --sampler walrus --walrus-cutoff 12 --walrus-max-ns --no-ours --n 10,12 --budget 10 --csv "$OUT/quick_cpu_walrus_time.csv"
CMD
run <<'CMD'
export CPY=$HOME/cpusamp/pq-env/bin/python; bash scripts/piquasso_setup.sh || exit 1; python3 bench/gbskernels_bench.py check --sampler piquasso --walrus-cutoff 12 --n 3 --shots 2000 --budget 30; python3 bench/gbskernels_bench.py time --sampler piquasso --walrus-cutoff 12 --no-ours --n 10,12 --budget 10 --csv "$OUT/quick_cpu_piquasso_time.csv"
CMD
fi

finish
