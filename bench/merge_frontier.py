"""Merge frontier CSVs from several map_threshold.py runs into one, for plot_frontier.py.

Used by the closing run: the Fock rows are re-measured with the new kernel while the emitter
engine runs on the untouched Triton path, so its rows are carried over from the previous map
instead of being re-measured for nothing.

    python3 bench/merge_frontier.py --into results/<stamp>/frontier_v2.csv \
            --from results/keep/frontier.csv --engines emitter

Rows whose (engine, ratio) already exist in --into are never overwritten: the destination
always wins, so re-running this is idempotent and a later re-measurement of the emitter engine
simply takes precedence.
"""
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--into", required=True, help="destination CSV, rewritten in place")
ap.add_argument("--from", dest="src", required=True, help="CSV to take the missing rows from")
ap.add_argument("--engines", default="emitter", help="comma list of engines to carry over")
a = ap.parse_args()

dst = list(csv.DictReader(open(a.into)))
if not dst:
    raise SystemExit(f"{a.into} is empty - nothing to merge into")
fields = list(dst[0].keys())
have = {(r["engine"], r["ratio"]) for r in dst}
want = set(a.engines.split(","))

try:
    src = list(csv.DictReader(open(a.src)))
except FileNotFoundError:
    print(f"{a.src} not found - nothing carried over")
    raise SystemExit(0) from None

added = [{k: r.get(k, "") for k in fields} for r in src
         if r.get("engine") in want and (r["engine"], r["ratio"]) not in have]
if added:
    with open(a.into, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(dst + added)
print(f"{a.into}: {len(dst)} measured + {len(added)} carried over from {a.src} "
      f"({', '.join(sorted(want))})")
