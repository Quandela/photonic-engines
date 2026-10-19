#!/usr/bin/env bash
# Frontier paper, Table 4 (uniform loss eta, m = 2n) and Sec. 6.
#   - Fock n* = 41.2 / 45.0 / 49.2 / 64.2 at eta = 0.9 / 0.8 / 0.7 / 0.5; detected eta n* = 37.1 / 36.0 / 34.4 / 32.1
#       original runs runs/20261008_071215, runs/20261008_085321, runs/20261008_104426, runs/20261008_192343
#       (the eta = 0.5 point is also the eta = 0.5 end of the coherence scan, Sec. 4.2)
#   - emitter n* = 33.6 / 36.8 / 40.9 / 53.5 at eta = 0.9 / 0.8 / 0.7 / 0.5; detected 30.2 / 29.4 / 28.6 / 26.8
#       original runs runs/20261008_210721 (0.9, 0.8), runs/20261008_122031 (0.7), runs/20261008_133143 (0.5)
#   - superposed n* = 54.0 / 58.3 / 64.8 at eta = 0.9 / 0.8 / 0.7 (eta = 0.5 not measured)
#       original runs runs/20261007_094729, runs/20261007_183855, runs/20261008_233436
# The eta = 1 column is in reproduce/frontier_headlines.sh.
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 33.6 h in total
# (Fock 5711 + 6452 + 5179 + 6043 s; emitter 8582 + 4234 + 20930 s; superposed 2303 + 9948 + 51596 s).
# --quick: the same bench, one point per engine and transmission, n = 10-17, 10 s per n, no bracket refinement.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
# Fock
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.9 --start 41 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta09.csv"
CMD
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.8 --start 44 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta08.csv"
CMD
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.7 --start 49 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta07.csv"
CMD
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.5 --start 63 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta05.csv"
CMD
# emitter
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.9 --start 33 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta09.csv"; timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.8 --start 36 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta08.csv"
CMD
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.7 --start 40 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta07.csv"
CMD
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.5 --start 53 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta05.csv"
CMD
# superposed
run <<'CMD'
python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --loss 0.9 --start 52 --step 1 --max-n 70 --budget 600 --min-shots 256 --bracket-min-shots 1024 --bracket-max-s 10800 --seed 102 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --csv "$OUT/floor_sbs_eta0.9_N1024.csv"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --loss 0.8 --start 58 --step 1 --max-n 75 --budget 600 --min-shots 256 --bracket-min-shots 4096 --bracket-max-s 28800 --seed 104 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --csv "$OUT/floor_sbs_eta0.8_N4096.csv"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --loss 0.7 --start 63 --step 1 --max-n 80 --budget 600 --min-shots 256 --bracket-min-shots 16384 --bracket-max-s 72000 --seed 105 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --csv "$OUT/floor_sbs_eta0.7_N16384.csv"
CMD
else
run <<'CMD'
for e in 0.9 0.8 0.7 0.5; do python3 bench/find_threshold.py --engine fock --ratio 2 --loss $e --start 16 --step 1 --max-n 17 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_fock_eta$e.csv"; done
CMD
run <<'CMD'
for e in 0.9 0.8 0.7 0.5; do python3 bench/find_threshold.py --engine emitter --ratio 2 --loss $e --start 14 --step 1 --max-n 15 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_emitter_eta$e.csv"; done
CMD
run <<'CMD'
for e in 0.9 0.8 0.7; do python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --loss $e --start 10 --step 1 --max-n 11 --budget 10 --seed 102 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --device "$DEVICE" --csv "$OUT/quick_sbs_eta$e.csv"; done
CMD
fi

finish
