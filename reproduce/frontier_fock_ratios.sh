#!/usr/bin/env bash
# Frontier paper, Table 3 (Fock row), Sec. 2 and Sec. 9 (more GPUs), supplement loop devices: the Fock sampler at
# other mode ratios, measured before the 256-shot protocol (240 s per n), and its multi-GPU timing check.
#   - Fock n* at m/n = 1.5 / 2.5 / 3 / 4 / 6 / 8 / 12 = 40.0 / 37.5 / 36.8 / 36.2 / 35.3 / 34.9 / 34.4;
#     n = 40, m = 60 at 1.0 s per sample                                         original run runs/20260920_092711
#     (the original command also listed the superposed engine; that arm was stopped before it finished and no
#     number rests on it, so it is left out here)
#   - Fock map at 13 ratios m/n = 1 ... 12 (input of the 128-bin loop-device rates and of the run-cost model;
#     arithmetic in reproduce/frontier_cpu_analyses.sh)                          original run runs/20260915_222632
#   - Delta n* = ln X / ln rho checked by direct timing up to about 4 photons past n*: 8 samples per point at
#     (m/n, n) = (2: 41, 43, 45), (1: 45, 47, 49), (4: 38, 40); eight GPUs at n ~ 42 for m = 2n
#                                                                                original run runs/20260915_222632
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 4.9 h (4932 s for the
# Fock arm and 4315 s for the emitter arm of the first map, 7022 s, 1447 s).
# --quick: the same benches at two ratios seeded at n = 16 (10 s per n), and the timing at two small points.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/map_threshold.py --engines fock,emitter --ratios 1,1.5,2,2.5,3,4,6,8,12 --budget 240 --out "$OUT/frontier3"
CMD
run <<'CMD'
python3 bench/map_threshold.py --engines fock --ratios 1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8,12 --anchor-fock 2:38.6 --budget 240 --out "$OUT/frontier_v2"
CMD
run <<'CMD'
python3 bench/time_n.py --points 2:41,2:43,2:45,1:45,1:47,1:49,4:38,4:40 --samples 8 --budget 900 --csv "$OUT/time_n.csv"
CMD
else
run <<'CMD'
python3 bench/map_threshold.py --engines fock --ratios 1.5,4 --anchor-fock 2:16 --budget 10 --extra="--max-n 18 --device $DEVICE" --out "$OUT/quick_frontier3"
CMD
run <<'CMD'
python3 bench/time_n.py --points 2:10,1:12 --samples 2 --budget 10 --device "$DEVICE" --csv "$OUT/quick_time_n.csv"
CMD
fi

finish
