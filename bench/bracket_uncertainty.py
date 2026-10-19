"""Statistical uncertainty of a find_threshold bracket, from the segment rates its run log prints.

No per-shot times are kept, but `bench/find_threshold.py` prints the rate of every segment of a bracket point: the
ramp batch ("n=38 m= 76: 1.266 samples/s (..., up to 256 shots, ...)"), then the cumulative top-up progress lines
("top-up progress n=38: +768 shots in 629 s -> ...") and the closing top-up line. For each bracket of the log:

    rate of a point     rbar = (sum of shots) / (sum of seconds) over its segments
    batch scatter       sigma^2 = sum_i w_i (r_i - rbar)^2 / (k - 1),  w_i = shots_i / batch,  k segments
    s.e. of the rate    sigma / sqrt(sum_i w_i), relative to rbar
    n*                  n + a / (a - b),  a = ln r_n,  b = ln r_{n+1}   (find_threshold's log-linear convention)
    s.e. of n*          first-order propagation of the two relative errors

Statistical only (no protocol or interpolation systematics); a point with two segments gives a rough s.e.

    python3 bench/bracket_uncertainty.py results/20261009_141132/job1.log
    python3 bench/bracket_uncertainty.py results/20261009_141132/job1.log results/20261008_192343/job1.log \
        results/20261007_090041/job1.log --diff 0:3 --diff 0:2 --diff 4:5

--diff indexes the brackets of all logs in listing order (one log can hold several). With the three logs above:
[0] Fock m = 2n, [1] Fock m = n, [2] emitter, [3] non-linear (20261009_141132), [4] Fock eta = 0.5 (20261008_192343),
[5] superposed eta = 1 (20261007_090041); so 0:3 = non-linear cost, 0:2 = emitter gap, 4:5 = coherence cost.
"""
import argparse
import math
import re
import sys

RAMP = re.compile(r"^\s*n=(\d+) m=\s*\d+:\s+([\d.]+) samples/s .*up to (\d+) shots")
TOPUP = re.compile(r"^\s*top-up(?: progress)? n=(\d+): \+(\d+) shots(?: \(batches of \d+\))? in (\d+(?:\.\d+)?) s")
BRACKET = re.compile(r"^\s*bracket: n=(\d+) at [\d.]+/s, n=(\d+) at [\d.]+/s\s+->\s+n\* = ([\d.]+)")


def parse(path):
    """Brackets of one log, in order: (n_lo, n_hi, printed n*, {n: [(shots, seconds), ...]})."""
    segs, cum, out = {}, {}, []
    for line in open(path, errors="replace"):
        m = RAMP.match(line)
        if m:
            n, rate, shots = int(m[1]), float(m[2]), int(m[3])
            if n not in cum:                                     # a ramp point (re-measured: the last one wins;
                                                                 # find_threshold never ramps an n after topping it up)
                segs[n] = [(shots, shots / rate)]
            continue
        m = TOPUP.match(line)
        if m:
            n, shots, sec = int(m[1]), int(m[2]), float(m[3])
            s0, t0 = cum.get(n, (0, 0.0))
            if shots > s0:
                segs.setdefault(n, []).append((shots - s0, sec - t0))
            cum[n] = (shots, sec)
            continue
        m = BRACKET.match(line)
        if m:
            lo, hi = int(m[1]), int(m[2])
            out.append((lo, hi, float(m[3]), {k: list(segs.get(k, [])) for k in (lo, hi)}))
            segs, cum = {}, {}
    return out


def point(seg, batch):
    s = [a for a, _ in seg]
    t = [b for _, b in seg]
    rbar = sum(s) / sum(t)
    if len(seg) < 2:
        return rbar, float("nan")
    w = [a / batch for a in s]
    var = sum(wi * (a / b - rbar) ** 2 for wi, a, b in zip(w, s, t)) / (len(seg) - 1)
    return rbar, math.sqrt(var) / math.sqrt(sum(w)) / rbar


def nstar(lo, seg_lo, seg_hi, batch):
    (ra, ea), (rb, eb) = point(seg_lo, batch), point(seg_hi, batch)
    a, b = math.log(ra), math.log(rb)
    return lo + a / (a - b), math.hypot(-b / (a - b) ** 2 * ea, a / (a - b) ** 2 * eb), (ra, ea), (rb, eb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="run job logs (results/<stamp>/job<i>.log or logs/run_<stamp>.log)")
    ap.add_argument("--batch", type=int, default=256, help="shots per batch (the 256-shot cap of the quoted rows)")
    ap.add_argument("--diff", action="append", default=[],
                    help="i:j — also print n*_i - n*_j with its s.e. (i, j index the brackets as listed)")
    a = ap.parse_args()
    rows = []
    for path in a.logs:
        for lo, hi, printed, segs in parse(path):
            if not segs[lo] or not segs[hi]:
                print(f"{path}: bracket {lo}/{hi}: segments not found", file=sys.stderr)
                continue
            ns, e, (ra, ea), (rb, eb) = nstar(lo, segs[lo], segs[hi], a.batch)
            rows.append((ns, e))
            if math.isnan(e):
                print(f"{path}: bracket {lo}/{hi}: a point has one segment, no scatter estimate", file=sys.stderr)
            print(f"[{len(rows) - 1}] {path} {lo}/{hi}: n* = {ns:.2f} +- {e:.2f} (printed {printed}); "
                  f"n={lo} at {ra:.3f}/s ({100 * ea:.1f} %, {len(segs[lo])} segments, {sum(x for x, _ in segs[lo])} shots), "
                  f"n={hi} at {rb:.3f}/s ({100 * eb:.1f} %, {len(segs[hi])} segments, {sum(x for x, _ in segs[hi])} shots)")
    for d in a.diff:
        i, j = (int(x) for x in d.split(":"))
        (x, ex), (y, ey) = rows[i], rows[j]
        print(f"[{i}] - [{j}] = {x - y:.2f} +- {math.hypot(ex, ey):.2f}")


if __name__ == "__main__":
    main()
