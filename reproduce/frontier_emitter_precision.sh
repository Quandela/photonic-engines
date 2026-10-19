#!/usr/bin/env bash
# Frontier paper, Sec. 7 and supplement Table S9, emitter row (true fp64 error of the emitter cofactor kernels).
#   - true error against a long-double reference, distinct rows: 1.1-1.3e-13 (K = 26), 6.3-7.7e-13 (28),
#     2.4-2.5e-12 (30); CUDA and Triton kernels agree to <= 4e-14 (distinct rows); two lane segmentations of the
#     Triton kernel differ by 8e-12 at K = 30
#                       GPU dumps: original run runs/20261004_223624; reference: CPU, after that run
# x86-64 ONLY. The reference is the CUDA kernel's code compiled for the CPU in long double, which is the 80-bit x87
# format (64-bit mantissa) on x86-64. On other architectures (arm64: long double = fp64) it would compare fp64 with
# fp64 and report meaningless differences, so this script refuses to run there.
# Full mode: the two GPU dumps of the original run (113 s on one NVIDIA H100 PCIe, together with the train check of
# reproduce/frontier_precision.sh), then the six long-double references of terms 0 and 1 on the kept dumps
# emitter_precision_k{26,28,30}{,_collide}_20261004_h100.npz under $KEEP (default results/keep): 58-1425 s per term on
# one core, 6379 s in all. The kept emitter_precision_reference_20261004_cpu.csv is the six reference files
# concatenated, with a `rows` column (distinct / collide).
# --quick: one dump at K = 10 on --device, and the reference of term 0 of the kept K = 26 dump (about 1 min).
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"
KEEP="${KEEP:-results/keep}"
export KEEP

case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "error: the long-double reference needs x86-64 (80-bit x87 long double); this machine is $(uname -m)" >&2
       exit 2 ;;
esac

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/emitter_precision.py --dump --K 26,28,30 --streams 4 --out "$OUT/emitter_precision" && python3 bench/emitter_precision.py --dump --K 26,28,30 --streams 4 --collide --out "$OUT/emitter_precision"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k26_20261004_h100.npz" --terms 0,1 --csv "$OUT/emitter_precision_k26_reference.csv"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k26_collide_20261004_h100.npz" --terms 0,1 --csv "$OUT/emitter_precision_k26_collide_reference.csv"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k28_20261004_h100.npz" --terms 0,1 --csv "$OUT/emitter_precision_k28_reference.csv"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k28_collide_20261004_h100.npz" --terms 0,1 --csv "$OUT/emitter_precision_k28_collide_reference.csv"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k30_20261004_h100.npz" --terms 0,1 --csv "$OUT/emitter_precision_k30_reference.csv"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k30_collide_20261004_h100.npz" --terms 0,1 --csv "$OUT/emitter_precision_k30_collide_reference.csv"
CMD
else
run <<'CMD'
python3 bench/emitter_precision.py --dump --K 10 --streams 4 --device "$DEVICE" --out "$OUT/quick_emitter_precision"
CMD
run <<'CMD'
python3 bench/emitter_precision.py --reference "$KEEP/emitter_precision_k26_20261004_h100.npz" --terms 0 --csv "$OUT/quick_emitter_precision_k26_reference.csv"
CMD
fi

finish
