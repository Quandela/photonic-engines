#!/usr/bin/env bash
# Frontier paper, Sec. 7 and supplement Table S9 (fp64 precision at the frontier sizes).
#   - Fock true error against a long-double reference: 3.6e-13 (k = 30), 1.9e-11 (34), 3.4e-10 (38); x2.35 per photon;
#     1e-4 near 53, 1e-3 near 55; CUDA and Triton agree to 3-6e-14              original run runs/20261011_192151
#   - Fock segmentation lower bounds (supplement result files)                 original run runs/20260930_082501
#   - squeezed-light guard: largest measured 8.4e-4 at N = 54 grouped; grouped pairings excluded at N = 60-62;
#     ungrouped 1.3-2.9 decades under the fit
#                    original runs runs/20261011_173624, runs/20261011_192051, runs/20261011_225012
#   - emitter-train amplitude error 2.5e-14 / 4.6e-14 / 1.1e-13 at k = 16 / 18 / 20 (x1.4 per emitter)
#                                                                              original run runs/20261004_223624
#   - superposed sources: mu 3.2e-15 / 8.8e-15 / 1.4e-13, lambda 6.7e-14 / 2.7e-13 / 3.6e-12 (lambda 1e-3 at q ~ 46)
#                    original runs runs/20261007_004708, runs/20261007_005452
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 2.3 h (3216 s, 511 s,
# 3513 + 11 + 586 s, under 113 s, 417 + 60 s). The long-double reference runs on the host CPU (x86-64, 24 threads
# in the original run).
# --quick: the same benches at k = 20, N = 24, train k = 8 / 10, one small superposed point.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
timeout 43200 python3 bench/cofactor_precision.py --ks 30,32,34,36,38 --reps 1 --ref ld --csv "$OUT/ld_precision.csv"; timeout 14400 python3 bench/cofactor_precision.py --ks 30,34 --reps 1 --collide --ref ld --csv "$OUT/ld_precision_collide.csv"
CMD
run <<'CMD'
timeout 3600 python3 bench/cofactor_precision.py --ks 30,34,38 --reps 2 --csv "$OUT/cof_precision.csv" && timeout 1800 python3 bench/cofactor_precision.py --ks 30,34,38 --reps 1 --collide --csv "$OUT/cof_precision_collide.csv"
CMD
run <<'CMD'
timeout 7200 python3 bench/gbs_kappa_check.py --N 50,52,54,56,58 --per 2 --csv "$OUT/gbs_kappa_derisk.csv"
CMD
run <<'CMD'
timeout 1200 python3 bench/gbs_kappa_check.py --N 50,62 --per 1 --max-terms 16777216 --draws 150 --csv "$OUT/gbs_kappa_grouped_derisk.csv"
CMD
run <<'CMD'
timeout 5400 python3 bench/gbs_kappa_check.py --N 50,52,54,56,58,60,62 --per 8 --max-terms 16777216 --draws 400 --csv "$OUT/gbs_kappa_grouped.csv"
CMD
run <<'CMD'
python3 bench/sequence_precision.py --k 16,18,20 --shots 4 --out "$OUT/sequence_precision.csv"
CMD
run <<'CMD'
python3 bench/rect_precision_gpu.py --points 47:28,49:30,51:32,39:28,41:30,43:32 --streams 1 --csv "$OUT/rect_precision_gpu.csv"; python3 bench/rect_precision.py --form mu --s0 18 --q 16,20,24 --streams 1 --out "$OUT/rect_precision_mu_s18.csv"; python3 bench/rect_precision.py --form lambda --s0 20 --q 16,20,24 --streams 1 --out "$OUT/rect_precision_lambda_s20.csv"
CMD
run <<'CMD'
python3 bench/rect_precision.py --form mu --s0 10 --q 16,20,24 --streams 1 --out "$OUT/rect_precision_mu_s10.csv"; python3 bench/rect_precision.py --form fock --s0 0 --q 16,20,24 --streams 1 --out "$OUT/rect_precision_fock_s0.csv"
CMD
else
run <<'CMD'
python3 bench/cofactor_precision.py --ks 20 --reps 1 --ref ld --device "$DEVICE" --csv "$OUT/quick_ld_precision.csv"; python3 bench/cofactor_precision.py --ks 20 --reps 1 --collide --ref ld --device "$DEVICE" --csv "$OUT/quick_ld_precision_collide.csv"
CMD
run <<'CMD'
python3 bench/cofactor_precision.py --ks 20 --reps 1 --device "$DEVICE" --csv "$OUT/quick_cof_precision.csv"
CMD
run <<'CMD'
python3 bench/gbs_kappa_check.py --N 24 --per 1 --device "$DEVICE" --csv "$OUT/quick_gbs_kappa.csv"; python3 bench/gbs_kappa_check.py --N 24 --per 1 --max-terms 16777216 --draws 4 --device "$DEVICE" --csv "$OUT/quick_gbs_kappa_grouped.csv"
CMD
run <<'CMD'
python3 bench/sequence_precision.py --k 8,10 --shots 2 --device "$DEVICE" --out "$OUT/quick_sequence_precision.csv"
CMD
run <<'CMD'
python3 bench/rect_precision_gpu.py --points 31:12 --streams 1 --csv "$OUT/quick_rect_precision_gpu.csv"; python3 bench/rect_precision.py --form mu --s0 18 --q 8 --streams 1 --out "$OUT/quick_rect_precision_mu_s18.csv"; python3 bench/rect_precision.py --form lambda --s0 20 --q 8 --streams 1 --out "$OUT/quick_rect_precision_lambda_s20.csv"; python3 bench/rect_precision.py --form fock --s0 0 --q 8 --streams 1 --out "$OUT/quick_rect_precision_fock_s0.csv"
CMD
fi

finish
