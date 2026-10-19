#!/usr/bin/env bash
# Frontier paper, Sec. 4.2 (coherence scan of superposed sources at p = 1/2 detected photons per source).
#   - n* = 56.8 / 54.1 / 52.4 / 51.1 at eta = 0.6 / 0.7 / 0.8 / 0.9                original run runs/20261007_122313
#   The two ends of the scan come from other scripts: eta = 0.5 (n* = 64.2, the Fock engine at eta = 0.5) is in
#   reproduce/frontier_loss.sh, eta = 1 (n* = 50.0) in reproduce/frontier_headlines.sh. The cost of coherence,
#   14.2 +- 0.2, is the difference of the two ends.
# Full mode runs the recorded command line. Expected wall time on one NVIDIA H100 PCIe: 17163 s (4.8 h).
# --quick: the same loop over the four (eta, alpha) points with n = 8-9, 10 s per n, 64 counted shots per bracket point.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
for spec in 0.9:0.8410686705679302:50 0.8:0.9117382909684877:51 0.7:1.0068536854342678:53 0.6:1.1502619915109316:55; do e=${spec%%:*}; r=${spec#*:}; a=${r%%:*}; s=${r#*:}; python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --loss $e --alpha $a --start $s --step 1 --max-n 80 --budget 600 --min-shots 256 --bracket-min-shots 1024 --bracket-max-s 10800 --seed 103 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --csv "$OUT/floor_coherence_eta${e}_N1024.csv"; done
CMD
else
run <<'CMD'
for spec in 0.9:0.8410686705679302 0.8:0.9117382909684877 0.7:1.0068536854342678 0.6:1.1502619915109316; do e=${spec%%:*}; a=${spec#*:}; python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --loss $e --alpha $a --start 8 --step 1 --max-n 9 --budget 10 --min-shots 8 --bracket-min-shots 64 --bracket-max-s 60 --seed 103 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --device "$DEVICE" --csv "$OUT/quick_coherence_eta${e}.csv"; done
CMD
fi

finish
