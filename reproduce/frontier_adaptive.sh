#!/usr/bin/env bash
# Frontier paper, Sec. 4.4 and the m/n = 2 points of Figure 3 (adaptive circuits); Sec. 6 (adaptive costs under
# imperfections).
#   - adaptive n* = 38.3 / 38.2 / 38.1 / 37.6 / 34.8 / 34.4 / 33.8 for K = 1, 2, 3, 4, 5, 6, 8 adaptive layers,
#     m = 2n, 512 shots                     original runs runs/20261007_022012 (K = 1), runs/20261007_030408 (K = 2-6, 8),
#                                           runs/20261007_061124 (K = 8, n = 34 / 35)
#   - K = 1-3 cost 0.6-1.1 sources at eta = 0.9 and 0.9-1.0 at V = 0.9 (against 0.0-0.2 lossless)
#                                           original run runs/20260923_181143
#     (--indist 0.95 is the pairwise overlap x = 0.95, i.e. V = x^2 = 0.9025, printed as V = 0.9)
#   - chains per sample measured on the GPU (column draws_per_sample), n = 28-38, 256 shots: K = 1 1.76-2.26,
#     K = 2 3.48-4.73 (earlier build of the adaptive sampler; the expected count E[2^r] is the cost model of
#     reproduce/frontier_cpu_analyses.sh)                                    original run runs/20260921_205843
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 5.4 h (2363 s,
# 10653 s, 1677 s, 2980 s, 1756 s).
# --quick: the same bench at n = 16, 16 shots, 60 s budget, for K = 1, 2 (marginal chain), K = 8 (rejection chain),
# and K = 1 at eta = 0.9 and at x = 0.95.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 5400 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 38,39 --K 1 --shots 512 --budget 3600 --max-entries 1e9 --csv "$OUT/afbs_K1_x4.csv"
CMD
run <<'CMD'
echo '== K2,3'; timeout 7200 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 37,38,39 --K 2,3 --shots 512 --budget 3600 --max-entries 1e9 --csv "$OUT/afbs_K23_x4.csv"; echo '== K4'; timeout 5400 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 35,36,37,38 --K 4 --shots 512 --budget 3600 --max-entries 1e9 --csv "$OUT/afbs_K4_x4.csv"; echo '== K5'; timeout 5400 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 32,33,34,35 --K 5 --chain rejection --shots 512 --budget 3600 --csv "$OUT/afbs_K5_x4.csv"; echo '== K6'; timeout 5400 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 32,33,34,35 --K 6 --chain rejection --shots 512 --budget 3600 --csv "$OUT/afbs_K6_x4.csv"; echo '== K8'; timeout 5400 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 29,30,31,32,33 --K 8 --chain rejection --shots 512 --budget 3600 --csv "$OUT/afbs_K8_x4.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 34,35 --K 8 --chain rejection --shots 512 --budget 3600 --csv "$OUT/afbs_K8b_x4.csv"
CMD
run <<'CMD'
timeout 14400 bash -c 'python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 34,36,38,40,42 --K 1,2,3 --shots 128 --budget 3600 --max-entries 1e9 --loss 0.9 --csv "$OUT/adaptive_imp3_eta0.9.csv"; python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 32,34,36,38,40 --K 1,2,3 --shots 128 --budget 3600 --max-entries 1e9 --indist 0.95 --csv "$OUT/adaptive_imp3_x0.95.csv"'
CMD
run <<'CMD'
timeout 14400 python3 bench/adaptive_frontier.py --final prefix --n 20,24,28,30,32,34,36,38 --K 1,2 --shots 256 --budget 3600 --csv "$OUT/adaptive_prefix.csv"
CMD
else
run <<'CMD'
python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 16 --K 1,2 --shots 16 --budget 60 --device "$DEVICE" --csv "$OUT/quick_afbs_K12.csv"
CMD
run <<'CMD'
python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 16 --K 8 --chain rejection --shots 16 --budget 60 --device "$DEVICE" --csv "$OUT/quick_afbs_K8.csv"
CMD
run <<'CMD'
python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 16 --K 1 --shots 16 --budget 60 --loss 0.9 --device "$DEVICE" --csv "$OUT/quick_adaptive_eta0.9.csv"; python3 bench/adaptive_frontier.py --final prefix --ratio 2 --n 16 --K 1 --shots 16 --budget 60 --indist 0.95 --device "$DEVICE" --csv "$OUT/quick_adaptive_x0.95.csv"
CMD
fi

finish
