#!/usr/bin/env bash
# Frontier paper, supplement Sec. S6 (strong simulation on one layered basis).
#   - Fock, n = 11, m = 22: 1.29e8 states, 98.9 ms, 15.4 GiB; superposed (graded) 180 ms, 16.6 GiB
#     (1.8x the time, 1.08x the memory)                                         original run runs/20260919_181625
#   - emitter (chi = 2), n = 11, m = 22: 304 ms, 34.1 GiB                       original run runs/20260909_230401
#   - Gaussian: six squeezers at m = 12, photon cutoff 18: 417 s, 12.7 GiB; agrees with loop hafnians to 1e-15
#                                                                              original run runs/20261003_120740
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 16 min (4 + 8 s, 516 s,
# 426 s).
# --quick: the same benches at n = 4, 6 (Gaussian: 2, 3 squeezers), one repeat.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/bench_product_slos.py --device "$DEVICE" --n 4,6,8,10 --ratio 2 --repeats 5 --csv "$OUT/product_slos_r2.csv"
CMD
run <<'CMD'
python3 bench/bench_product_slos.py --device "$DEVICE" --n 11 --ratio 2 --repeats 3 --csv "$OUT/product_slos_r2_n11.csv"
CMD
run <<'CMD'
python3 bench/bench_emitter_slos.py --device "$DEVICE" --n 8,10,11
CMD
run <<'CMD'
timeout 3600 python3 bench/bench_gaussian_slos.py --device "$DEVICE" --n 3,4,5,6 --ratio 2 --csv "$OUT/gaussian_slos_r2.csv"
CMD
else
run <<'CMD'
python3 bench/bench_product_slos.py --device "$DEVICE" --n 4,6 --ratio 2 --repeats 1 --csv "$OUT/quick_product_slos_r2.csv"
CMD
run <<'CMD'
python3 bench/bench_emitter_slos.py --device "$DEVICE" --n 4,6 --out "$OUT/quick_bench_emitter_slos.csv"
CMD
run <<'CMD'
python3 bench/bench_gaussian_slos.py --device "$DEVICE" --n 2,3 --ratio 2 --csv "$OUT/quick_gaussian_slos_r2.csv"
CMD
fi

finish
