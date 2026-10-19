#!/usr/bin/env bash
# Frontier paper, Sec. 6 (partial distinguishability, HOM visibility V; x = sqrt(V) is the pairwise overlap).
#   - emitter at eta = 1: n* = 31.6 / 32.2 / 33.4 at V = 0.95 / 0.9 / 0.8       original run runs/20261003_121759
#     (measured on an earlier emitter kernel; a re-run on the current kernel gives values at least as large)
#   - eta = 0.7 grid: Fock n* = 50.3 / 51.6 / 53.7 and emitter n* = 41.7 / 42.6 / 44.7 at V = 0.95 / 0.9 / 0.8
#     (the V = 1 cells, 49.2 and 40.9, are in reproduce/frontier_loss.sh)     original run runs/20261009_195643
#   - Fock at eta = 1: n* = 39.0 / 40.0 / 41.8 at V = 0.95 / 0.9 / 0.8 (measured before the 256-shot protocol;
#     that run's own V = 1 point is 38.6)                                       original run runs/20260921_051530
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 12.4 h (6040 s,
# 31928 s, 6534 s).
# --quick: the same benches at two visibilities, seeded at n = 14-15, 10 s per n.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/frontier_indist.py --engines emitter --eta 1,0.7 --vis 1,0.95,0.9,0.8 --out "$OUT/frontier_indist_emitter_auto"
CMD
run <<'CMD'
timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.7 --indist 0.974679 --start 49 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta07_v095.csv"; timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.7 --indist 0.948683 --start 50 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta07_v09.csv"; timeout 36000 python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.7 --indist 0.894427 --start 52 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_fock_eta07_v08.csv"; timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.7 --indist 0.974679 --start 40 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta07_v095.csv"; timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.7 --indist 0.948683 --start 41 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta07_v09.csv"; timeout 36000 python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.7 --indist 0.894427 --start 42 --seed 0 --max-shots 256 --min-shots 256 --bracket-batch 256 --bracket-min-shots 2048 --bracket-max-s 9000 --budget 2400 --step 1 --max-n 100 --csv "$OUT/r30_emitter_eta07_v08.csv"
CMD
run <<'CMD'
python3 bench/frontier_indist.py --engines fock,emitter --eta 1,0.7 --vis 1,0.95,0.9,0.8
CMD
else
run <<'CMD'
python3 bench/frontier_indist.py --engines emitter --eta 1 --vis 1,0.9 --anchor-emitter 2:14:1.95 --max-n 18 --budget 10 --extra="--device $DEVICE" --out "$OUT/quick_frontier_indist_emitter"
CMD
run <<'CMD'
python3 bench/frontier_indist.py --engines fock --eta 1 --vis 1,0.9 --anchor-fock 2:15:1.8 --max-n 18 --budget 10 --extra="--device $DEVICE" --out "$OUT/quick_frontier_indist_fock"
CMD
run <<'CMD'
python3 bench/find_threshold.py --engine fock --ratio 2 --loss 0.7 --indist 0.948683 --start 16 --step 1 --max-n 17 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_fock_eta07_v09.csv"; python3 bench/find_threshold.py --engine emitter --ratio 2 --loss 0.7 --indist 0.948683 --start 14 --step 1 --max-n 15 --budget 10 --seed 0 --device "$DEVICE" --csv "$OUT/quick_emitter_eta07_v09.csv"
CMD
fi

finish
