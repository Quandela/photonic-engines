#!/usr/bin/env bash
# Frontier paper, supplement Sec. S1 (related work: estimation against sampling).
#   - cost of the probability estimator in this repository's code: 7.5-80 us per shot on an H100 at n = 10-40
#     (bond dimension chi = 2, 16, 256)                                         original run runs/20260917_110557
# Full mode runs the recorded command line (the bench writes estimate_time.csv into the output directory).
# Expected wall time on one NVIDIA H100 PCIe: 57 s.
# --quick: n = 10, chi = 2, 16, 10^4 shots.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/estimate_time.py --device "$DEVICE" --n 10,20,30,40 --chi 2,16,256 --shots 100000
CMD
else
run <<'CMD'
python3 bench/estimate_time.py --device "$DEVICE" --n 10 --chi 2,16 --shots 10000 --out "$OUT/quick_estimate_time.csv"
CMD
fi

finish
