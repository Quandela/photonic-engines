#!/usr/bin/env bash
# Frontier paper, Sec. 5.1 and supplement Sec. S2.7 (emitter trains: k spin emitters, cost per tick-shot).
#   - k = 18 takes 28 ms per tick-shot (k = 16: 3.6 ms); supplement row 0.17 / 0.71 / 3.6 / 28 ms
#     (Glynn)                                                                  original run runs/20260919_084900
#   - against 0.41 / 3.2 / 29 / 273 ms with the Laplace expansion             original run runs/20260918_084620
#   - k ~ 21-22 at one tick per second (19-20 with Laplace) is an extrapolation of these rows
#   - area-law plateau 0.76-1.39 bits at k = 4-16; the parity brickwork reaches Page after 8-12 layers (runs/20260918_084620)
#   - precomputed Kraus sets (k <= 7): cost per bin-shot constant to one percent from 4 to 400 bins; k = 7 reaches
#     n = 400 bins at 1.06 sequences/s with a 19 GB set (418 bins ~ 2900 photons at one per second)
#              original runs runs/20260912_185503, runs/20260912_190841, runs/20260912_220323 (m = 4k)
#     (each run wrote sequence_frontier.csv; here they get three names, concatenated they are the kept file)
#   - register entanglement: near-Page (rank 2^(k/2) from tick 1, entropy k/2 - ~1 by tick 3) for Haar optics;
#     brickwork bond 1.6-1.9 at 99.9 %, entropy 0.1 bit at k = 8-16             original run runs/20260913_193014
#     (the bench's default label-amplitude method and draw location have changed since that run: the per-tick
#     times differ; rank and entropy are properties of the sampled register state)
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 13.9 h (21602 s,
# 25697 s, 667 + 1307 + 163 s, 144 + 24 + 394 s).
# --quick: the same bench at k = 6, 8 (brick) and k = 5, 7 (circuit), 4 ticks, 16 shots; Kraus sets at k = 1, 2;
# register entanglement at k = 6, 2 ticks.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/esbs_fusion.py --device "$DEVICE" --k 12,14,16,18 --optics xfusion --program brick --ticks 40 --shots 256 --out-prefix "$OUT/esbs_fusion_glynn"
CMD
run <<'CMD'
python3 bench/esbs_fusion.py --device "$DEVICE" --k 4,6,8,10,12,14,16 --optics typeII,xfusion,haar4 --program brick --ticks 40 --shots 256 && python3 bench/esbs_fusion.py --device "$DEVICE" --k 4,6,8,10,12,14,16 --optics xfusion --program rus --ticks 40 --shots 256 && python3 bench/esbs_fusion.py --device "$DEVICE" --k 5,7,9,11,13,15 --optics xfusion --program circuit --ticks 120 --shots 256 && python3 bench/esbs_fusion.py --device "$DEVICE" --k 4,6,8,10,12,14,16 --optics xfusion --program brick --gates h --ticks 40 --shots 256
CMD
run <<'CMD'
python3 bench/sequence_frontier.py --device "$DEVICE" --k 1,2,3,4,5,6,7 --n 4,5,6,7,8,9,10,12,15,20,30,40 --shots 1024 --out "$OUT/sequence_frontier.csv"
CMD
run <<'CMD'
python3 bench/sequence_frontier.py --device "$DEVICE" --k 6,7 --n 60,100,200,400 --shots 256 --generic-max 0 --out "$OUT/sequence_frontier_n400.csv"
CMD
run <<'CMD'
python3 bench/sequence_frontier.py --device "$DEVICE" --k 2,3,4,5,6 --n 10 --ratio 4 --shots 1024 --generic-max 0 --out "$OUT/sequence_frontier_ratio4.csv"
CMD
run <<'CMD'
python3 bench/sequence_ticks.py --device "$DEVICE" --k 6,8,10,12,14,16 --ticks 8 --shots 256 --unitary haar
CMD
run <<'CMD'
python3 bench/sequence_ticks.py --device "$DEVICE" --k 18,20 --ticks 4 --shots 8 --unitary haar --out "$OUT/sequence_ticks_haar_frontier.csv"
CMD
run <<'CMD'
python3 bench/sequence_ticks.py --device "$DEVICE" --k 8,12,16 --ticks 16 --shots 256 --unitary band:2
CMD
else
run <<'CMD'
python3 bench/esbs_fusion.py --device "$DEVICE" --k 6,8 --optics xfusion --program brick --ticks 4 --shots 16 --out-prefix "$OUT/quick_esbs_fusion_glynn"
CMD
run <<'CMD'
python3 bench/esbs_fusion.py --device "$DEVICE" --k 6,8 --optics typeII,xfusion,haar4 --program brick --ticks 4 --shots 16 --out-prefix "$OUT/quick_esbs_fusion" && python3 bench/esbs_fusion.py --device "$DEVICE" --k 6,8 --optics xfusion --program rus --ticks 4 --shots 16 --out-prefix "$OUT/quick_esbs_fusion" && python3 bench/esbs_fusion.py --device "$DEVICE" --k 5,7 --optics xfusion --program circuit --ticks 4 --shots 16 --out-prefix "$OUT/quick_esbs_fusion" && python3 bench/esbs_fusion.py --device "$DEVICE" --k 6,8 --optics xfusion --program brick --gates h --ticks 4 --shots 16 --out-prefix "$OUT/quick_esbs_fusion"
CMD
run <<'CMD'
python3 bench/sequence_frontier.py --device "$DEVICE" --k 1,2 --n 4,8 --shots 16 --out "$OUT/quick_sequence_frontier.csv"
CMD
run <<'CMD'
python3 bench/sequence_ticks.py --device "$DEVICE" --k 6 --ticks 2 --shots 8 --unitary haar --out "$OUT/quick_sequence_ticks_haar.csv"; python3 bench/sequence_ticks.py --device "$DEVICE" --k 6 --ticks 2 --shots 8 --unitary band:2 --out "$OUT/quick_sequence_ticks_band2.csv"
CMD
fi

finish
