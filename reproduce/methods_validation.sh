#!/usr/bin/env bash
# Methods paper: the superposed-source sampler's cost and the validation against strong simulation.
#   - sampler cost table and Sec. 8: superposed sources cost 1.15x the single-photon sampler (n = 30, empty offset);
#     1.04 / 0.97 / 0.75 / 0.52 / 0.23 at sin^2 alpha = 1 / 0.99 / 0.95 / 0.9 / 0.8 (n = 34)
#                                                                              original run runs/20261006_131552
#   - supplement validation table and Sec. 8.1: 19 instances, 4e4 samples each, largest |z| = 1.67
#                                                                              original run runs/20261009_195328
# Full mode runs the recorded command lines (the first also runs the slow tests of the two samplers and prints the
# size of the CuPy kernel cache before and after). Expected wall time on one NVIDIA H100 PCIe: about 2.5 min
# (127 s, 19 s).
# --quick: the two test files without the slow tier; the cost validation at n = 16; the law validation at 4000 shots.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
kc() { echo "cupy kernel_cache $1: $(ls ~/.cupy/kernel_cache 2>/dev/null | wc -l) files, $(du -sk ~/.cupy/kernel_cache 2>/dev/null | cut -f1) kB"; }; kc start; python3 -m pytest -q --runslow tests/clifford/test_sbs_rso.py tests/clifford/test_cuda_rect.py; kc after-tests; python3 bench/sbs_rso_validate.py --backend cuda --csv "$OUT/sbs_rso_validate.csv"; kc end
CMD
run <<'CMD'
python3 bench/validate_methods.py --device "$DEVICE" --shots 40000 --out "$OUT/validate_methods_h100.csv"
CMD
else
run <<'CMD'
python3 -m pytest -q tests/clifford/test_sbs_rso.py tests/clifford/test_cuda_rect.py; python3 bench/sbs_rso_validate.py --backend cuda --n-cost 16 --n-cont 16 --shots-exact 2000 --csv "$OUT/quick_sbs_rso_validate.csv"
CMD
run <<'CMD'
python3 bench/validate_methods.py --device "$DEVICE" --shots 4000 --out "$OUT/quick_validate_methods.csv"
CMD
fi

finish
