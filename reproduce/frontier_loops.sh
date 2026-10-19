#!/usr/bin/env bash
# Frontier paper, Sec. 5.2, Figure 4 and supplement Sec. S3 (time-bin loop machines).
#   - superposed sources in loops: N* = 267.4 / 237.4 / 112.9 / 62.5 / 31.3 / 21.4 at loop length L = 1 / 2 / 3 / 4 / 6 / 8
#                                                                              original run runs/20261001_101803
#   - its Fock limit (alpha = pi/2, dense sampler): within 3 %, at most 1.15x  original run runs/20261005_170728
#   - squeezed light in loops: N* >= 4000 / 3171 / 2250 / 540 / 110 / 31.4 / 11.6 / 7.3 at L = 1 ... 8
#   - register against sieve counts at the (1, 6, 36) loop ladder: 2e16 against 8e6 at N = 20, 8e40 against 1e25
#     at N = 125                                                               original run runs/20261002_203546
# The single-photon, one-emitter and k-emitter loop rows: reproduce/frontier_loops_v3.sh.
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 4.3 h (5314 s,
# 4036 s, 6181 s).
# --quick: superposed loops at L = 1, 8 (N <= 16) and the Fock limit at L = 8 (N <= 12); squeezed loops at L = 1, 4
# (N <= 64); 10 s per N; the squeezed count and law check on small grids.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 9000 python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 120 --input sbs --loops 1,2,3,4,6,8 --csv "$OUT/hybrid_frontier_sbs.csv"
CMD
run <<'CMD'
timeout 9000 python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 120 --input sbs --alpha 1.5707963267948966 --sampler dense --loops 1,2,3,4,6,8 --csv "$OUT/hybrid_frontier_sbs_halfpi.csv"
CMD
run <<'CMD'
python3 bench/gbs_loop.py count --loops 1,2,3,4,5,6,7,8 --n 16,64,256,1024,4000 --json "$OUT/gbs_loop_count.json"; python3 bench/gbs_loop.py count --ladder 1,6,36 --n 216 --ratio 1 --json "$OUT/gbs_loop_count_borealis.json"; timeout 900 python3 bench/gbs_loop.py check --device "$DEVICE" --loops 1,3 --n 6,12 --cap 8,12,16 --shots 4000; for L in 1 2 3 4 5 6 7 8; do timeout 1800 python3 bench/gbs_loop.py frontier --device "$DEVICE" --loops $L --max-n 4000 --csv "$OUT/gbs_loop_L$L.csv" --json "$OUT/gbs_loop_L$L.json"; done
CMD
else
run <<'CMD'
python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 10 --input sbs --loops 1,8 --max-n 16 --csv "$OUT/quick_hybrid_frontier_sbs.csv"
CMD
run <<'CMD'
python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 10 --input sbs --alpha 1.5707963267948966 --sampler dense --loops 8 --max-n 12 --csv "$OUT/quick_hybrid_frontier_sbs_halfpi.csv"
CMD
run <<'CMD'
python3 bench/gbs_loop.py count --loops 1,4 --n 16,64 --json "$OUT/quick_gbs_loop_count.json"; python3 bench/gbs_loop.py count --ladder 1,6,36 --n 216 --ratio 1 --json "$OUT/quick_gbs_loop_count_borealis.json"; python3 bench/gbs_loop.py check --device "$DEVICE" --loops 1 --n 6 --cap 8 --shots 400; for L in 1 4; do python3 bench/gbs_loop.py frontier --device "$DEVICE" --loops $L --max-n 64 --budget 10 --csv "$OUT/quick_gbs_loop_L$L.csv"; done
CMD
fi

finish
