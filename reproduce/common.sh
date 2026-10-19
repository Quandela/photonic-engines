# Shared preamble of the reproduce/ scripts. Source it from a script, never run it alone:
#
#     . "$(dirname "$0")/common.sh" "$@"
#
# Options (every script):
#   --out DIR            output directory (default: out/<script name>); created if missing
#   --quick              reduced sizes: the same benches, minutes on one GPU (smoke test, not the published numbers)
#   --device cuda|cpu    device for the benches that take --device (default: cuda)
#   -h, --help           print the script's header and exit
#
# Exports OUT (absolute path), QUICK (0 or 1), DEVICE, and RESULTS_DIR (= OUT): the benches that take a bare output
# name, or no output option, write under $RESULTS_DIR.
#
# run CMD...   prints the command with a timestamp, then evaluates it in this shell. With no argument, the command is
#              read from standard input (a quoted here-document keeps it verbatim; "$OUT" and "$DEVICE" expand when it
#              runs).

set -euo pipefail

_script="$0"
_name="$(basename "$_script" .sh)"

usage() {
    # the header comment block of the calling script: from line 2 to the first blank line
    awk 'NR == 1 { next } /^$/ { exit } { sub(/^# ?/, ""); print }' "$_script"
}

if [ ! -f bench/find_threshold.py ] || [ ! -d photonic_engines ] || [ ! -d reproduce ]; then
    echo "error: run this script from the repository root, e.g.  bash reproduce/$_name.sh --quick" >&2
    exit 2
fi

OUT="out/$_name"
QUICK=0
DEVICE=cuda
while [ $# -gt 0 ]; do
    case "$1" in
        --out)       [ $# -ge 2 ] || { echo "error: --out needs a directory" >&2; exit 2; }; OUT="$2"; shift 2 ;;
        --out=*)     OUT="${1#--out=}"; shift ;;
        --quick)     QUICK=1; shift ;;
        --device)    [ $# -ge 2 ] || { echo "error: --device needs cuda or cpu" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --device=*)  DEVICE="${1#--device=}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "error: unknown option '$1' (see -h)" >&2; exit 2 ;;
    esac
done
case "$DEVICE" in
    cuda|cpu) ;;
    *) echo "error: --device must be cuda or cpu, not '$DEVICE'" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
RESULTS_DIR="$OUT"
export OUT QUICK DEVICE RESULTS_DIR

LOG="$OUT/reproduce.log"
touch "$LOG"
_START="$OUT/.start"
touch "$_START"                                 # finish() accepts only result files written after this

run() {
    # Output also goes to $OUT/reproduce.log. Some benches launch child runs and exit 0 even when a child fails, so a
    # Python traceback anywhere in the command's output fails the step. This is a heuristic: a bench that prints a
    # caught traceback on purpose would fail the step, and a child that dies without a Python traceback and exit 0
    # would not.
    local cmd start
    if [ $# -gt 0 ]; then cmd="$(printf '%q ' "$@")"; else cmd="$(cat)"; fi
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$cmd" | tee -a "$LOG"
    start=$(wc -c < "$LOG")
    ( eval "$cmd" ) 2>&1 | tee -a "$LOG"        # pipefail: the step fails if the command does
    if tail -c +"$((start + 1))" "$LOG" | grep -q "Traceback (most recent call last)"; then
        echo "error: a Python traceback in the output of this step (a bench or one of its child runs failed); see $LOG" >&2
        exit 1
    fi
}

finish() {
    # every script ends with this: at least one non-empty result file written by THIS run must exist in $OUT
    if ! find "$OUT" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.npz' -o -name '*.png' -o -name '*.txt' \) \
            -size +0 -newer "$_START" | grep -q .; then
        echo "error: no result file written in $OUT by this run" >&2
        exit 1
    fi
    printf '[%s] %s: done, results in %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$_name" "$OUT"
}

if [ "$QUICK" = 1 ]; then _mode=quick; else _mode=full; fi
printf '[%s] %s: %s mode, device %s, output %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$_name" "$_mode" "$DEVICE" "$OUT"
