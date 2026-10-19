#!/usr/bin/env bash
# Frontier paper, Table 3 (n* at other mode ratios m/n) and Sec. 4.1.
#   - emitter n* at m/n = 2.5 / 3 / 4 / 6 / 8 = 30.5 / 30.2 / 29.8 / 29.5 / 29.2   original run runs/20261002_003041
#     and at m/n = 12: 29.0                                                       original run runs/20261003_103957
#     (measured on an earlier emitter kernel; a re-run on the current kernel gives values at least as large)
#   - superposed n* at m/n = 1 / 1.5 / 2.5 / 3 / 4 / 6 / 8 / 12
#     = 53.1 / 51.1 / 49.5 / 49.1 / 48.6 / 48.0 / 47.5 / 47.0 (the map's own m/n = 2 point: 50.2)
#                                                                                 original run runs/20261010_045711
#   - Fock + one non-linear site (phi = pi/2) at m/n = 1 / 1.5 / 3 / 4 / 6 / 8 / 12
#     = 40.1 / 37.4 / 34.5 / 33.8 / 33.0 / 32.5 / 32.1                            original run runs/20260922_211901
#     (its m = 2n point, 36.1, is in reproduce/frontier_headlines.sh)
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 7.5-8.5 h (3367 s and
# at most 4902 s for the emitter maps, 18765 s for the superposed map, 3695 s for the non-linear ratio scan).
# --quick: the same map benches at two ratios, seeded at small n (6-17) and capped a few n above, 10 s per n.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/map_threshold.py --engines emitter --ratios 2,2.5,3,4,6,8 --budget 240 --extra '--kernel auto' --out "$OUT/frontier_auto"
CMD
run <<'CMD'
python3 bench/map_threshold.py --engines emitter --ratios 12 --budget 240 --out "$OUT/frontier_auto12"
CMD
run <<'CMD'
python3 bench/map_threshold.py --engines sbs --ratios 1,1.5,2,2.5,3,4,6,8,12 --anchor-sbs 2:49 --extra='--algo rso --rso-backend cuda --rso-square kernel --rso-draw device --min-shots 256 --bracket-min-shots 512 --bracket-max-s 7200 --seed 107' --out "$OUT/floor_frontier_sbs_N512"
CMD
run <<'CMD'
python3 bench/nlbs_frontier.py --scan ratio --ratios 1,1.5,2,3,4,6,8,12,16 --phi pi/2 --out nlbs_ratio
CMD
else
run <<'CMD'
python3 bench/map_threshold.py --engines emitter --ratios 2,4 --budget 10 --anchor-emitter 2:14 --extra="--kernel auto --max-n 14 --device $DEVICE" --out "$OUT/quick_frontier_auto"
CMD
run <<'CMD'
python3 bench/map_threshold.py --engines sbs --ratios 1,4 --anchor-sbs 2:6 --budget-sbs 10 --extra="--algo rso --rso-backend cuda --rso-square kernel --rso-draw device --seed 107 --max-n 8 --device $DEVICE" --out "$OUT/quick_frontier_sbs"
CMD
run <<'CMD'
python3 bench/nlbs_frontier.py --scan ratio --ratios 1,4 --phi pi/2 --anchor-fock 2:16:1.81 --max-n 20 --budget 10 --extra="--device $DEVICE" --out quick_nlbs_ratio
CMD
fi

finish
