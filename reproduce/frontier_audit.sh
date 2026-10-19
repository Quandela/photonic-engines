#!/usr/bin/env bash
# Frontier paper, supplement Table S8 (parity audit) and Sec. 4: limit tests and kernel residuals.
#   - Fock limit of the superposed sampler (alpha = pi/2, every source a photon), m = 2n:
#     38.4 against 38.4 for the Fock engine                                     original run runs/20261006_012054
#   - twin limit at eta = 0.5 for the coherence scan: 64.6 against 64.4
#     (superposed scan at p = 1/2)                                              original run runs/20261006_012054
#     (Fock and superposed at alpha = pi/2, eta = 0.5)                          original run runs/20261005_064849
#   - emitter lane-split residual: 1.32-1.48x per needed operation (the measured throughput; the payload factors
#     are counted from the kernel sources)                                      original run runs/20261005_234844
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 4.5 h (2606 s,
# 9198 s, 3911 s, 655 s).
# --quick: the same benches at n = 8-17, 10 s per n; the coherence scan at two transmissions; one throughput point
# at n = 20.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/find_threshold.py --ratio 2 --budget 600 --seed 0 --engine fock --start 34 --max-n 60 --csv "$OUT/limit_fock_lossless_final.csv"; python3 bench/find_threshold.py --ratio 2 --budget 600 --seed 0 --engine sbs --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --alpha 1.5707963267948966 --start 34 --max-n 60 --csv "$OUT/limit_rso_pi2_lossless_final.csv"
CMD
run <<'CMD'
python3 bench/coherence_scan.py --brightness 0.5 --eta 0.5,0.6,0.7,0.8,0.9,1 --ratio 2 --budget 600 --start 30 --max-n 120 --extra='--algo rso --rso-backend cuda --rso-square kernel --rso-draw device' --out "$OUT/coherence_p50_rso_final"
CMD
run <<'CMD'
python3 bench/find_threshold.py --ratio 2 --budget 600 --seed 0 --engine fock --loss 0.5 --start 56 --max-n 100 --csv "$OUT/limit_fock_eta05.csv"; python3 bench/find_threshold.py --ratio 2 --budget 600 --seed 0 --engine sbs --algo rso --rso-backend cuda --alpha 1.5707963267948966 --loss 0.5 --start 46 --max-n 100 --csv "$OUT/limit_rso_pi2_eta05.csv"
CMD
run <<'CMD'
python3 bench/emitter_fock_throughput.py --n 34,36 --steps=0 --streams 32 --reps 2 --csv "$OUT/emitter_fock_throughput_4x.csv"
CMD
else
run <<'CMD'
python3 bench/find_threshold.py --ratio 2 --budget 10 --seed 0 --engine fock --start 16 --step 1 --max-n 17 --device "$DEVICE" --csv "$OUT/quick_limit_fock_lossless.csv"; python3 bench/find_threshold.py --ratio 2 --budget 10 --seed 0 --engine sbs --algo rso --rso-backend cuda --rso-square kernel --rso-draw device --alpha 1.5707963267948966 --start 16 --step 1 --max-n 17 --device "$DEVICE" --csv "$OUT/quick_limit_rso_pi2_lossless.csv"
CMD
run <<'CMD'
python3 bench/coherence_scan.py --brightness 0.5 --eta 0.5,1 --ratio 2 --budget 10 --start 8 --max-n 9 --extra="--algo rso --rso-backend cuda --rso-square kernel --rso-draw device --device $DEVICE" --out "$OUT/quick_coherence_p50"
CMD
run <<'CMD'
python3 bench/find_threshold.py --ratio 2 --budget 10 --seed 0 --engine fock --loss 0.5 --start 16 --step 1 --max-n 17 --device "$DEVICE" --csv "$OUT/quick_limit_fock_eta05.csv"; python3 bench/find_threshold.py --ratio 2 --budget 10 --seed 0 --engine sbs --algo rso --rso-backend cuda --alpha 1.5707963267948966 --loss 0.5 --start 16 --step 1 --max-n 17 --device "$DEVICE" --csv "$OUT/quick_limit_rso_pi2_eta05.csv"
CMD
run <<'CMD'
python3 bench/emitter_fock_throughput.py --n 20 --steps=0 --streams 4 --reps 1 --csv "$OUT/quick_emitter_fock_throughput.csv"
CMD
fi

finish
