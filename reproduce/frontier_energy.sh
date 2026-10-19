#!/usr/bin/env bash
# Frontier paper, Sec. 8 (energy per sample of the Fock sampler, m = 2n).
#   - board power 270-344 W over n = 12-38; at n = 38: 339 W and 1.29 samples/s, i.e. 7.3e-2 Wh per sample
#                                                                              original run runs/20261011_172843
# Full mode runs the recorded command line. Expected wall time on one NVIDIA H100 PCIe: 304 s.
# --quick: n = 12, 15 with a 5 s timed window and 2 s idle baseline.
# Needs an NVIDIA GPU with NVML or nvidia-smi for the power readings.
# Options: --out DIR, --quick, --device cuda|cpu (ignored: the bench is GPU only), -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 1800 python3 bench/energy_series.py --ns 12,15,18,20,24,30,34,38 --csv "$OUT/energy_r2.csv"
CMD
else
run <<'CMD'
python3 bench/energy_series.py --ns 12,15 --window 5 --idle 2 --csv "$OUT/quick_energy_r2.csv"
CMD
fi

finish
