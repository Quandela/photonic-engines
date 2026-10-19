#!/usr/bin/env bash
# Frontier paper: the one-hertz thresholds n* of the headline rows (abstract; Tables 2, 3 and 4 at eta = 1).
#   - single photons (Fock), m = 2n: n* = 38.3; m = n: n* = 42.6          original run runs/20261009_141132
#   - emitter (chi = 2 linear cluster), m = 2n: n* = 31.0                 original run runs/20261006_004008
#     (confirmed at 31.0 by the third command of runs/20261009_141132)
#   - superposed sources, alpha = pi/4, m = 2n: n* = 50.0                 original run runs/20261007_090041
#   - Fock + one non-linear site (phi = pi/2), m = 2n: n* = 36.1          original run runs/20261009_141132
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 6.8 h
# (20249 s for the four thresholds of runs/20261009_141132, at most 1495 s for the emitter point, 2678 s for the
# superposed point).
# --quick: the same benches at n = 14-17 with a 10 s budget per n and no bracket refinement; checks that every engine
# and option runs, it does not reproduce the thresholds.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --start 38 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_c2.csv"; timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 1 --start 42 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_c1.csv"; timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --start 30 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_c2.csv"; timeout 36000 python3 bench/find_threshold.py --engine nlbs --ratio 2 --phi pi/2 --sites 1 --layout same-layer --start 35 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_nlbs_c2.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/frontier_loss.py --engines emitter --eta 1 --ratio 2 --anchor-emitter 2:30.9:1.95 --out "$OUT/frontier_loss_emitter_r18"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --start 48 --step 1 --max-n 60 --budget 600 --min-shots 256 --bracket-min-shots 512 --bracket-max-s 7200 --seed 101 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --csv "$OUT/floor_sbs_eta1_N512.csv"
CMD
else
run <<'CMD'
python3 bench/find_threshold.py --engine fock --ratio 2 --start 16 --step 1 --max-n 17 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_fock_c2.csv"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine fock --ratio 1 --start 16 --step 1 --max-n 17 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_fock_c1.csv"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine emitter --ratio 2 --start 14 --step 1 --max-n 15 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_emitter_c2.csv"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine nlbs --ratio 2 --phi pi/2 --sites 1 --layout same-layer --start 14 --step 1 --max-n 15 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_nlbs_c2.csv"
CMD
run <<'CMD'
python3 bench/frontier_loss.py --engines emitter --eta 1 --ratio 2 --anchor-emitter 2:14:1.95 --max-n 16 --budget 10 --extra="--device $DEVICE" --out "$OUT/quick_frontier_loss_emitter"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine sbs --ratio 2 --target 1.0 --start 10 --step 1 --max-n 11 --budget 10 --seed 101 --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --device "$DEVICE" --csv "$OUT/quick_sbs_eta1.csv"
CMD
fi

finish
