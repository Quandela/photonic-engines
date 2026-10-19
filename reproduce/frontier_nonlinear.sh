#!/usr/bin/env bash
# Frontier paper, Sec. 4.3 and Figure 2 (non-linear sites), Sec. 6 (non-linear costs under imperfections).
#   - number of sites: n* = 36.0 / 34.3 / 30.3 / 21.4 for L = 1 / 2 / 4 / 8 sites (2.1 per site, least squares);
#     same-layer and sequential layouts within 0.1                                original run runs/20260923_080913
#   - Kerr strength scan (Figure 2): phi = pi/2 lies 2.5 below the scan's phi = 0 point (38.5), pi/16 lies 8.3 below,
#     phi = pi returns to the phi = 0 value                                      original run runs/20260922_211901
#   - cost of one site under imperfections: 2.2 / 2.6 / 4.4 at eta = 1 / 0.9 / 0.7; 2.9 / 3.7 at V = 0.9,
#     eta = 1 / 0.7 (the scan minus the Fock thresholds of record)               original run runs/20260923_111505
# Full mode runs the recorded command lines. Expected wall time on one NVIDIA H100 PCIe: about 2.5 h (2884 s,
# 3185 s, 3008 s).
# --quick: the same three scans on reduced grids, seeded at n ~ 14 and capped at n = 20, 10 s per n.
# Options: --out DIR, --quick, --device cuda|cpu, -h (see reproduce/common.sh).

. "$(dirname "$0")/common.sh" "$@"

if [ "$QUICK" = 0 ]; then
run <<'CMD'
python3 bench/nlbs_frontier.py --scan sites --sites 1,2,4,8 --layout same-layer,sequential --phi pi/2 --ratio 2 --out nlbs_sites
CMD
run <<'CMD'
python3 bench/nlbs_frontier.py --scan phi --phis 0,pi/16,pi/8,pi/4,3*pi/8,pi/2,5*pi/8,3*pi/4,pi --ratio 2 --out nlbs_phi
CMD
run <<'CMD'
python3 bench/nlbs_frontier.py --scan imperfections --eta 1,0.9,0.7 --vis 1,0.9 --phi pi/2 --ratio 2 --out nlbs_imperfections
CMD
else
run <<'CMD'
python3 bench/nlbs_frontier.py --scan sites --sites 1,2 --layout same-layer,sequential --phi pi/2 --ratio 2 --anchor-fock 2:16:1.81 --max-n 20 --budget 10 --extra="--device $DEVICE" --out quick_nlbs_sites
CMD
run <<'CMD'
python3 bench/nlbs_frontier.py --scan phi --phis 0,pi/2 --ratio 2 --anchor-fock 2:16:1.81 --max-n 20 --budget 10 --extra="--device $DEVICE" --out quick_nlbs_phi
CMD
run <<'CMD'
python3 bench/nlbs_frontier.py --scan imperfections --eta 1,0.9 --vis 1,0.9 --phi pi/2 --ratio 2 --anchor-fock 2:16:1.81 --max-n 20 --budget 10 --extra="--device $DEVICE" --out quick_nlbs_imperfections
CMD
fi

finish
