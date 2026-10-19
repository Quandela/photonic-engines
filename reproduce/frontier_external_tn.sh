#!/usr/bin/env bash
# Frontier paper, supplement Sec. S1 and Table S2 (CUDA-Q orca-photonics target), Sec. 9.
#   - CUDA-Q orca-photonics 0.16.0 (CPU state vector) is 176-10,939x slower than ours at 8-10 modes; its build cost
#     grows with exponent 1.009 (R^2 = 0.970); ours 0.1-0.3 us per sample; 4800x fewer states at the 3:10 case
#                    original runs runs/20260923_224734 and runs/20260924_134210 (same commands, run twice),
#                    runs/20261005_130135 (shot-count slopes)
# Third-party code: CUDA-Q (release 0.16.0 in the runs), installed beforehand: pip install cudaq==0.16.0. Without it
# the CUDA-Q columns are empty and the ours-only columns still run.
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 6 min (about 3 min for
# the first set, 217 s for the shot-count set).
# --quick: one loop and one mesh case at 2-3 photons, shots 1 and 4, one repeat, 30 s per CUDA-Q cell.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

run <<'CMD'
python3 -c "import cudaq; print('cudaq', cudaq.__version__)" 2>/dev/null || echo "cudaq not importable: only the ours-only columns will run"
CMD

if [ "$QUICK" = 0 ]; then
run <<'CMD'
if python3 -c "import cudaq" 2>/dev/null; then python3 bench/cudaq_qudit_levels_repro.py 2>&1 | tail -8; else echo "cudaq not importable: qudit-levels check skipped"; fi
CMD
run <<'CMD'
python3 bench/bench_tn_compare.py --circuit loop --cases 2:1,3:1,4:1,5:1,6:1,4:2 --shot-list 1,4,16,64,256 --reps 3 --max-sv 4e7 --cq-budget 240 --timeout 900 --csv "$OUT/tn_loop.csv"
CMD
run <<'CMD'
python3 bench/bench_tn_compare.py --circuit mesh --cases 2:6,3:6,4:6,2:8,3:8,4:8,2:10,3:10 --shot-list 1,4,16,64,256 --reps 3 --max-sv 4e7 --cq-budget 240 --timeout 900 --csv "$OUT/tn_mesh.csv"
CMD
run <<'CMD'
python3 bench/bench_tn_compare.py --circuit loop --cases 8:1,12:1,16:1,20:1 --targets ours --shot-list 1,4,16,64,256 --reps 3 --csv "$OUT/tn_loop_ours.csv"
CMD
run <<'CMD'
python3 bench/bench_tn_compare.py --circuit mesh --cases 6:12,8:16,10:20,12:24 --targets ours --shot-list 1,4,16,64,256 --reps 3 --csv "$OUT/tn_mesh_ours.csv"
CMD
run <<'CMD'
python3 bench/bench_tn_compare.py --shot-list 1,1024,16384,262144 --cq-shots 1,1024,16384,262144 --reps 3 --max-sv 4e7 --cq-budget 900 --timeout 2400 --circuit mesh --cases 2:6,3:8,4:8,3:10 --csv "$OUT/tn_mesh_shots.csv"; python3 bench/bench_tn_compare.py --shot-list 1,1024,16384,262144 --cq-shots 1,1024,16384,262144 --reps 3 --max-sv 4e7 --cq-budget 900 --timeout 2400 --circuit loop --cases 4:1,4:2 --csv "$OUT/tn_loop_shots.csv"
CMD
else
run <<'CMD'
python3 bench/bench_tn_compare.py --circuit loop --cases 2:1,3:1 --shot-list 1,4 --reps 1 --max-sv 4e7 --cq-budget 30 --timeout 120 --device "$DEVICE" --csv "$OUT/quick_tn_loop.csv"
CMD
run <<'CMD'
python3 bench/bench_tn_compare.py --circuit mesh --cases 2:6 --shot-list 1,4 --reps 1 --max-sv 4e7 --cq-budget 30 --timeout 120 --device "$DEVICE" --csv "$OUT/quick_tn_mesh.csv"
CMD
fi

finish
