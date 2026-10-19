#!/usr/bin/env bash
# Frontier paper (and the adaptive cost row of the methods paper): CPU analyses on kept result files and run logs
# (no GPU; seconds to a few minutes each).
#   - adaptive cost model at n = 38: X = 1.03 / 1.07 / 1.10 and E[2^r] = 2.0 / 3.8 / 6.8 for K = 1 / 2 / 3
#   - adaptive cost-model map over m/n (model curves of Fig. 3), anchored on the plain-sampler crossings of the
#     2026-09-22 map (frontier_anchors_20260922.csv); compared with the measured crossings of
#     adaptive_crossings_20261007.csv
#   - feed-forward cost per tick (feed-forward figure): ratio 1.00 for k >= 11, 1.1-1.7 below
#                                       (kept emitter-train scans of runs/20260918_084620, runs/20260919_084900)
#   - devices against the classical sampler (supplement devices table; run-of-S statistic over a 3600 s run) from
#     the cost table of runs/20261011_205906
#   - memory frontier at 80 GiB: 18 / 13 / 10 / 8 / 6 photons at m/n = 1 / 2 / 4 / 8 / 16; 7.2 squeezers at m/n = 2;
#     at 640 GiB (eight GPUs) 13.5 -> 14.6
#   - treewidth crossover D* at n = 38, m = 76 and at the squeezed point
#   - t_S against the plain mean (Fock, emitter): the run-cost model on the kept per-ratio maps (runs/20260915_222632
#     for Fock; runs/20261002_003041, runs/20261003_103957 for the emitter)
#   - arithmetic on the kept Fock maps: rates of a 128-bin loop device (runs/20260915_222632), and n = 40, m = 60
#     against the four-chip engine's 80 s (runs/20260920_092711)
#   - bracket uncertainties from the segment rates of the run logs: 2.26 +- 0.02 (non-linear cost), 7.34 +- 0.03
#     (emitter gap), 14.17 +- 0.19 (coherence cost at eta = 0.5); n* itself to +-0.01-0.04 lossless, +-0.19 at
#     eta = 0.5                          logs of runs/20261009_141132, runs/20261008_192343, runs/20261007_090041
# Inputs: the kept result files, under $KEEP (default results/keep); the run logs, under $RUNS (default runs).
# --quick: the adaptive model at n = 20 and a K <= 1 map; the other steps as in full mode (seconds each).
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"
KEEP="${KEEP:-results/keep}"
RUNS="${RUNS:-runs}"
export KEEP RUNS

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/adaptive_toll.py --n 38 --ratio 2,3 --K 0-12 --tables-upto 4 --csv "$OUT/adaptive_toll.csv"
CMD
run <<'CMD'
python3 bench/adaptive_toll.py --map --map-extra 8 --tables-upto 4 --anchors "$KEEP/frontier_anchors_20260922.csv" --csv "$OUT/adaptive_map.csv"
CMD
else
run <<'CMD'
python3 bench/adaptive_toll.py --n 20 --ratio 2 --K 0-2 --tables-upto 4 --csv "$OUT/quick_adaptive_toll.csv"
CMD
run <<'CMD'
python3 bench/adaptive_toll.py --map --map-K 1 --tables-upto 4 --anchors "$KEEP/frontier_anchors_20260922.csv" --csv "$OUT/quick_adaptive_map.csv"
CMD
fi
run <<'CMD'
python3 bench/f7_feedforward_cost.py --glob "$KEEP/esbs_fusion_*.csv" --csv "$OUT/f7_feedforward_cost.csv" --png "$OUT/f7_feedforward_cost.png"
CMD
run <<'CMD'
python3 bench/plot_postselect.py --cost "$KEEP/cost_table_20261011_h100.csv" --frontier "$KEEP/frontier.csv" --table "$OUT/postselect_v2.csv" --out "$OUT/postselect_v2" | tee "$OUT/postselect_v2.txt"
CMD
run <<'CMD'
python3 bench/memory_frontier.py 80 | tee "$OUT/memory_frontier_80.txt"
CMD
run <<'CMD'
python3 bench/memory_frontier.py 640 | tee "$OUT/memory_frontier_640.txt"
CMD
run <<'CMD'
python3 bench/treewidth_crossover.py | tee "$OUT/treewidth_crossover.txt"
CMD
run <<'CMD'
python3 bench/run_cost.py --engine fock --lossless "$KEEP/frontier_v2_fock_r*_20260915_h100.csv" --out "$OUT/run_cost_fock_model.csv"
CMD
run <<'CMD'
python3 bench/run_cost.py --engine emitter --lossless "$KEEP/frontier_emitter_r*.csv" --out "$OUT/run_cost_emitter_model.csv"
CMD
run <<'CMD'
python3 -c '
import csv, sys
rows = sorted((float(r["ratio"]), float(r["n_star"]), float(r["rho_eff"]))
              for r in csv.DictReader(open(sys.argv[1])) if r["engine"] == "fock")
print("128-bin device: rate(n, m = 128) = rho_c^(n*_c - n) at the measured ratios c on either side of 128/n")
for n in (21, 25, 28, 36):
    c = 128 / n
    near = [r for r in rows if r[0] <= c][-1:] + [r for r in rows if r[0] >= c][:1]
    print(f"  n = {n} (128/n = {c:.2f}): " + ", ".join(f"c = {r[0]:g} (n* {r[1]}, rho {r[2]}): {r[2] ** (r[1] - n):.3g}/s" for r in near))
' "$KEEP/frontier_v2_20260915_h100.csv" | tee "$OUT/fock_map_arithmetic.txt"
CMD
run <<'CMD'
python3 -c '
import csv, math, sys
r = [x for x in csv.DictReader(open(sys.argv[1])) if x["n"] == "40" and x["m"] == "60"][0]
t, shots = 1 / float(r["samples_per_s"]), r["shots"]
rho = [float(x["rho_eff"]) for x in csv.DictReader(open(sys.argv[2])) if x["engine"] == "fock" and float(x["ratio"]) == 1.5][0]
print(f"n = 40, m = 60: {t:.3f} s per sample ({shots} shots); against 80 s: factor {80 / t:.0f}, "
      f"{math.log(80 / t) / math.log(rho):+.1f} photons at rho_eff = {rho}")
' "$KEEP/frontier3_fock_r1.5_20260920_h100.csv" "$KEEP/frontier3_20260920_h100.csv" | tee -a "$OUT/fock_map_arithmetic.txt"
CMD
run <<'CMD'
python3 bench/bracket_uncertainty.py "$RUNS/20261009_141132/log.txt" "$RUNS/20261008_192343/log.txt" "$RUNS/20261007_090041/log.txt" --diff 0:3 --diff 0:2 --diff 4:5 | tee "$OUT/bracket_uncertainty.txt"
CMD

finish
