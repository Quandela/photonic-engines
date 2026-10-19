#!/usr/bin/env bash
# Frontier paper, supplement Sec. S5 (devices against a real-time classical sampler): the cost table t(k, m) of the
# Fock sampler at the device mode counts m = 12, 24, 32, 48, 96.
#   - t(2,96) = 52 ns, t(16,96) = 3.2 us, t(24,96) = 0.42 ms, t(30,96) = 14 ms, t(36,96) = 0.45 s, t(38,96) = 1.3 s,
#     t(45,96) = 94 s                                                          original run runs/20261011_205906
# Full mode runs the recorded command line. Expected wall time on one NVIDIA H100 PCIe: 6245 s (1.7 h).
# --quick: m = 12 and 24 only, k <= 12, 10 s budget per point.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 7200 python3 bench/cost_table.py --ms 12,24,32,48,96 --budget 120 --csv "$OUT/cost_table.csv"
CMD
else
run <<'CMD'
python3 bench/cost_table.py --ms 12,24 --kmax 12 --budget 10 --device "$DEVICE" --csv "$OUT/quick_cost_table.csv"
CMD
fi

finish
