#!/usr/bin/env bash
# CPU analyses behind numbers of both papers (no GPU; seconds to a few minutes each).
#   - non-linear cost law: kappa = 0.55 +- 0.08; acceptances B = 2 0.43-0.59 ... B = 16 0.054-0.077
#                                       (from the kept frontier scans of runs/20260922_211901)
#   - superposed sources, residual against the best existing kernel, in sources: <= 0.07; headroom of a high-degree
#     kernel +0.14-0.17 (n = 46) ... +0.35-0.43 (n = 60)   (kernel tables of runs/20261005_215026, runs/20261006_120404)
#   - Soret energy crossing (C&C 2018 / 2020 averagings, the X = L_2018 / L_2020 points, the measured classical energy
#     of runs/20261011_*)
#   - photonic-MPS term counts: cost-weighted headroom 1.02-1.11x, peaks of the mean R_j        (runs/20260916_153618)
#   - superposed sources against Perceval: TVD and coherent-vs-incoherent table; first-moment check (n = 14, m = 28)
#   - methods paper CPU checks: non-linear random-order counterexample (TVD 0.115), exact law of the chain, leave-one-out
#     division accuracy, lossy mask, non-linear oracles
# Inputs: the kept result files, under $KEEP (default results/keep).
# Needs perceval-quandela for the Perceval comparison.
# --quick: fewer shots for the Perceval comparison, two of the seven term-count jobs.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"
KEEP="${KEEP:-results/keep}"
export KEEP

run <<'CMD'
python3 bench/nlbs_cost_law.py --ratio-csv "$KEEP/nlbs_ratio_20260922_h100.csv" --phi-csv "$KEEP/nlbs_phi_20260922_h100.csv" --acceptance-csv "$KEEP/nlbs_acceptance_20260922_h100.csv" | tee "$OUT/nlbs_cost_law.txt"
CMD
run <<'CMD'
python3 bench/sbs_mu_residual.py --mu-csv "$KEEP/cuda_rect_mu_extend_96db3fd_20261005_h100.csv" --r22-csv "$KEEP/cuda_rect_mu_extend_r22_833d953_20261006_h100.csv" | tee "$OUT/sbs_mu_residual.txt"
CMD
run <<'CMD'
python3 bench/soret_energy_crossing.py | tee "$OUT/soret_energy_crossing_cc.txt"
CMD
run <<'CMD'
python3 bench/soret_energy_crossing.py --x-points 51:14,51:15,59:16,63:17,66:18,40:20 | tee "$OUT/soret_energy_crossing_x.txt"
CMD
run <<'CMD'
python3 bench/soret_energy_crossing.py --classical gpu --energy "$KEEP/energy_r2_20261011_h100.csv" | tee "$OUT/soret_energy_crossing_measured.txt"
CMD

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/emitter_gram.py --n 8  --family linear --samples 12 --check-rho --csv "$OUT/gram_linear_n8.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 12 --family linear --samples 12 --csv "$OUT/gram_linear_n12.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 16 --family linear --samples 8  --csv "$OUT/gram_linear_n16.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 20 --family linear --samples 4  --csv "$OUT/gram_linear_n20.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 12 --family random --samples 12 --csv "$OUT/gram_random_n12.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 16 --family random --samples 8  --csv "$OUT/gram_random_n16.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 12 --family ghz    --samples 8  --csv "$OUT/gram_ghz_n12.csv"
CMD
run <<'CMD'
python3 bench/sbs_chain_rso.py --shots 40000 --csv "$OUT/sbs_chain_rso_perceval.csv"
CMD
run <<'CMD'
python3 bench/sbs_chain_rso.py --first-moment 14:28 --shots 3000 --seed 1 | tee "$OUT/sbs_chain_rso_first_moment_seed1.txt"
CMD
run <<'CMD'
python3 bench/methods_cpu_checks.py | tee "$OUT/methods_cpu_checks.txt"
CMD
else
run <<'CMD'
python3 bench/emitter_gram.py --n 8  --family linear --samples 4 --check-rho --csv "$OUT/quick_gram_linear_n8.csv"
CMD
run <<'CMD'
python3 bench/emitter_gram.py --n 8  --family ghz    --samples 4 --csv "$OUT/quick_gram_ghz_n8.csv"
CMD
run <<'CMD'
python3 bench/sbs_chain_rso.py --shots 4000 --csv "$OUT/quick_sbs_chain_rso_perceval.csv"
CMD
run <<'CMD'
python3 bench/methods_cpu_checks.py --only lossmask | tee "$OUT/quick_methods_cpu_checks.txt"
CMD
fi

finish
