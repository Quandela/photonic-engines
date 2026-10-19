#!/usr/bin/env bash
# Frontier paper, Sec. 3 and supplement Table S4 (the exact Fock samplers on one instance family).
#   - n* (m = 2n / m = n): C&C 2018 32.5 / 32.5; input-side 31.7 / 31.8; per cofactor 30.9 / 33.7;
#     C&C 2020 (Triton) 35.7 / 40.0                                             original run runs/20260930_082501
#   - strong-then-sample is limited by memory at n = 11 (m = 2n) and 15 (m = n) as measured; 13 and 18 are the
#     table-alone counts                                                       original run runs/20260930_141657
#   - addends along GPU chains: the counts reproduce the exponent to 0.5 %; rho_eff 1.82 / 1.78 (1.814 against 1.808,
#     1.680 against 1.6875; 200 chains)                                        original run runs/20260930_082501
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 1.1 h (3227 s, 15 s,
# 708 s).
# --quick: the four algorithms at n = 14-15 with a 10 s budget; strong-then-sample at n = 6, 8; addends at n = 12, 16
# with 20 chains.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
for a in 2018 input percof 2020; do for c in 2 1; do timeout 5400 python3 bench/find_threshold.py --engine fock --device "$DEVICE" --algo $a --backend triton --ratio $c --start 26 --budget 120 --csv "$OUT/threshold_${a}_c${c}.csv"; done; done
CMD
run <<'CMD'
timeout 3600 python3 bench/algorithm_envelope.py --device "$DEVICE" --algos strong --ratios 2,1 --ns 6,8,10,11,12,13,14,15,16,17,18,19,20 --run-samples 1e4 --csv "$OUT/algorithm_envelope_strong.csv"
CMD
run <<'CMD'
timeout 1200 python3 bench/algorithm_terms.py --engine cuda --ns 16,20,24,28,32,36,40 --ratios 1,2 --samples 200 --csv "$OUT/algorithm_terms_gpu.csv"
CMD
else
run <<'CMD'
for a in 2018 input percof 2020; do for c in 2 1; do python3 bench/find_threshold.py --engine fock --device "$DEVICE" --algo $a --backend triton --ratio $c --start 14 --step 1 --max-n 15 --budget 10 --csv "$OUT/quick_threshold_${a}_c${c}.csv"; done; done
CMD
run <<'CMD'
python3 bench/algorithm_envelope.py --device "$DEVICE" --algos strong --ratios 2,1 --ns 6,8 --run-samples 1e4 --csv "$OUT/quick_algorithm_envelope_strong.csv"
CMD
run <<'CMD'
python3 bench/algorithm_terms.py --engine cuda --ns 12,16 --ratios 1,2 --samples 20 --csv "$OUT/quick_algorithm_terms_gpu.csv"
CMD
fi

finish
