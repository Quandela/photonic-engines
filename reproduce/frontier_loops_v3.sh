#!/usr/bin/env bash
# Frontier paper: loop-device rows measured on the released sampler (supplement loop table; Sec. 5).
#   - single photons through one loop (Fock chain), packed and one every 2 bins: N* > 7.5e4 at L = 1, 8, 12 (search cap)
#   - one emitter: N* = 4204.4 / 3492.8 / 2950.1 / 2165.8 at L = 6 / 8 / 10 / 11; > 3142 at L <= 4; >= 943 at L = 12
#   - k emitters: 3142.7 (k = 2, L = 1), 1853.0 (k = 2, L = 2), 1440.4 (k = 3, L = 1); larger registers memory-limited
#   - mean loop occupancy at the brackets: 9-13 (one emitter), 6-9 (k emitters)
#                                                                  original run runs/20261016_023739 (21 jobs)
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 12 h. The L = 11 emitter
# job stops at its 90-min timeout after its bracket, as in the original run: its exit 124 is tolerated.
# --quick: one emitter at L = 1 to N = 400, two emitters at L = 1 to N = 100, single photons at L = 1 to N = 2000.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 1 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L1.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 2 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L2.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 4 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L4.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 6 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L6.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 8 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L8.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 10 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L10.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 11 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L11.csv" || [ $? = 124 ]
CMD
run <<'CMD'
timeout 9000 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 12 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L12.csv"
CMD
run <<'CMD'
timeout 9000 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input emitter --loops 13 --start 300 --max-n 5000 --csv "$OUT/hybrid_frontier_emitter_v3_L13.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 2 --loops 1 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k2_L1.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 2 --loops 2 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k2_L2.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 2 --loops 3 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k2_L3.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 3 --loops 1 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k3_L1.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 3 --loops 2 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k3_L2.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 4 --loops 1 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k4_L1.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 120 --input hybrid --emitters 4 --loops 2 --max-n 2500 --csv "$OUT/hybrid_frontier_hybrid_v3_k4_L2.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 1200 --input fock --loops 1 --start 1259 --max-n 100000 --csv "$OUT/hybrid_frontier_fock_chain_L1.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 1200 --input fock --loops 8 --start 1259 --max-n 100000 --csv "$OUT/hybrid_frontier_fock_chain_L8.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 1200 --input fock --loops 12 --start 1259 --max-n 100000 --csv "$OUT/hybrid_frontier_fock_chain_L12.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 1200 --input fock --spacing 2 --loops 1 --start 1259 --max-n 100000 --csv "$OUT/hybrid_frontier_fock_chain_spaced_L1.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/hybrid_frontier.py --device cuda --ratios 2 --budget 1200 --input fock --spacing 2 --loops 12 --start 1259 --max-n 100000 --csv "$OUT/hybrid_frontier_fock_chain_spaced_L12.csv"
CMD
else
run <<'CMD'
python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 20 --input emitter --loops 1 --start 100 --max-n 400 --csv "$OUT/quick_hybrid_frontier_emitter_L1.csv"
CMD
run <<'CMD'
python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 20 --input hybrid --emitters 2 --loops 1 --max-n 100 --csv "$OUT/quick_hybrid_frontier_hybrid_k2_L1.csv"
CMD
run <<'CMD'
python3 bench/hybrid_frontier.py --device "$DEVICE" --ratios 2 --budget 20 --input fock --loops 1 --start 100 --max-n 2000 --csv "$OUT/quick_hybrid_frontier_fock_chain_L1.csv"
CMD
fi

finish
