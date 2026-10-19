#!/bin/bash
# Strong-scaling sweep of the distributed CPU engine on one node (run from the repository root):
#   bench/scaling_distributed.sh [--sizes "10:20 11:22 12:24"] [--ranks "1 2 4 8 16 32 64"] [--threads auto|1|N] [--out results.csv]
# One line per (size, P) with total/build/comm/compute (max over ranks), the per-rank spread,
# the OpenMP threads per rank (auto = cores / P) and the halo volume; P is capped at the number
# of cores. Sizes that need more memory than the node has are skipped from the per-rank
# estimate 1.4*16*M_n/P + 200 MB.
set -u
SIZES="10:20 11:22 12:24"
RANKS="1 2 4 8 16 32 64"
OUT=${RESULTS_DIR:-.}/scaling_distributed.csv
THREADS=auto
MPIRUN_ARGS=${MPIRUN_ARGS:-}
while [ $# -gt 0 ]; do
    case "$1" in
        --sizes) SIZES=$2; shift 2 ;;
        --ranks) RANKS=$2; shift 2 ;;
        --threads) THREADS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
MEMKB=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}'); MEMKB=${MEMKB:-0}
echo "size,P,threads,M_n,total_s,build_s,comm_s,compute_s,build_min_s,build_max_s,chunk,halo_gb" > "$OUT"
export PYTHONUNBUFFERED=1
for sz in $SIZES; do
    n=${sz%%:*}; m=${sz##*:}
    Mn=$(python3 -c "from math import comb; print(comb($n+$m-1,$m-1))")
    for P in $RANKS; do
        [ "$P" -le "$NCPU" ] || { echo "skip P=$P > $NCPU cores"; continue; }
        need_kb=$(python3 -c "print(int((1.4*16*$Mn/$P + 200e6)*$P/1024))")
        if [ "$MEMKB" -gt 0 ] && [ "$need_kb" -gt "$MEMKB" ]; then echo "skip n=$n m=$m P=$P: needs ~$((need_kb/1024)) MB, $((MEMKB/1024)) MB available"; continue; fi
        if [ "$THREADS" = auto ]; then T=$((NCPU / P)); [ "$T" -ge 1 ] || T=1; else T=$THREADS; fi
        echo "=== n=$n m=$m P=$P threads=$T  ($(date +%H:%M:%S))"
        line=$(OMP_NUM_THREADS=$T mpirun $MPIRUN_ARGS -np "$P" python3 bench/bench_distributed.py --n "$n" --m "$m" --per-rank 2>&1 | tee /dev/stderr)
        tot=$(echo "$line" | grep -o "total [0-9.]*" | awk '{print $2}')
        bld=$(echo "$line" | grep -o "build [0-9.]*" | tail -1 | awk '{print $2}')
        cmm=$(echo "$line" | grep -o "comm [0-9.]*" | tail -1 | awk '{print $2}')
        cmp=$(echo "$line" | grep -o "compute [0-9.]*" | tail -1 | awk '{print $2}')
        chunk=$(echo "$line" | grep -o "chunk [0-9,]*" | awk '{print $2}' | tr -d ,)
        bmin=$(echo "$line" | grep "rank" | grep -o "build [0-9.]*" | awk '{print $2}' | sort -n | head -1)
        bmax=$(echo "$line" | grep "rank" | grep -o "build [0-9.]*" | awk '{print $2}' | sort -n | tail -1)
        halo=$(echo "$line" | grep -o "halo [0-9.]* GB  norm" | awk '{print $2}')
        [ -n "$tot" ] && echo "$n:$m,$P,$T,$Mn,$tot,$bld,$cmm,$cmp,$bmin,$bmax,$chunk,$halo" >> "$OUT"
    done
done
echo "wrote $OUT"
