"""Cross-mode collision grouping against the v2.8 pairing on device-like patterns: time, condition number, terms
(docs/GAUSSIAN.md §3.4 v2.9-pairing; `hafnian._pairing_cross`, `_pick_grouping`, `_choose_pairing`).

    python3 bench/gbs_pairing_bench.py patterns --data <qca-data> --n 32,36,40,44,48 --per 4 --out gbs_pairing_patterns.json
    python3 bench/gbs_pairing_bench.py run --patterns gbs_pairing_patterns.json --device cuda --csv gbs_pairing.csv

patterns (CPU): photon-count patterns of N vertices cut from Xanadu's Borealis samples (github.com/XanaduAI/xanadu-qca-data)
- "heavy" from fig4 (r 1.10: modes of 5-13 photons), keeping a sample's most occupied modes first, "light" from fig3a
(r 0.53), modes at random - with the B matrix of the run's model (pure part of `gaussian.transfer_decomposition`
on its published T, r) restricted to the pattern, and a heterodyne-like loop vector. Written with B, so the GPU job
needs no sample file.
run: each pattern's loop-hafnian polynomial by `lhaf_poly_multi` (V = 1) in four modes:
  v28     the v2.8 rule (`_ROUND_TOL = None`: self-pairing within x2 of the least estimate above the search, else the
          least; self-pairing below);
  auto    the rule since 2026-09-27 (cost-aware grouping below the search, the cheapest candidate predicted within
          `_ROUND_TOL` above, checked after the sieve and redone if it misses);
  cross   the cross pairing forced (`pairing="cross"`: Bulmer et al.'s term count at any cost in precision);
  random  the least-estimate candidate of the search (`_ROUND_TOL = 0`: nothing qualifies), the best-conditioned;
and writes seconds, kappa = max_k sum_S |term_S|_k / max_k |c_k|, N eps kappa, terms, whether auto redid its sieve, and
the largest difference of each mode's polynomial from the reference (random, or dd_cross with --dd), relative to max |c|
(all modes compute the same value; the difference is their round-off). With --dd also dd_cross, dd_auto (double-double,
`precision="dd"`, to --dd-max-n vertices) and prec_auto (`precision="auto"`: fp64, escalated to dd when needed).
"""
import argparse
import csv
import json
import time

import numpy as np

EPS = float(np.finfo(np.float64).eps)


def model_B(data):
    from photonic_engines.gaussian import transfer_decomposition
    from photonic_engines.states import GaussianInput
    T = np.load(f"{data}/T.npy"); r = np.load(f"{data}/r.npy").reshape(-1)
    st = GaussianInput(r, tuple(range(T.shape[0])), T.shape[0])
    rp, Up, _, _ = transfer_decomposition(st, T)
    Uc = Up[:, :rp.size]
    return -(Uc * np.tanh(rp)) @ Uc.T


def patterns(args):
    rng = np.random.default_rng(args.seed)
    out = []
    for kind, run in (("heavy", "fig4"), ("light", "fig3a")):
        B = model_B(f"{args.data}/{run}")
        S = np.load(f"{args.data}/{run}/samples.npy", mmap_mode="r")
        for N in [int(x) for x in args.n.split(",")]:
            got = 0
            while got < args.per:
                pat = np.asarray(S[rng.integers(S.shape[0])]).reshape(-1).astype(int)
                if pat.sum() < N:
                    continue
                nz = np.nonzero(pat)[0]
                order = sorted(nz, key=lambda j: (-pat[j], rng.random())) if kind == "heavy" else list(rng.permutation(nz))
                cnt = np.zeros(pat.size, int); tot = 0
                for j in order:
                    take = min(pat[j], N - tot)
                    cnt[j] = take; tot += take
                    if tot == N:
                        break
                modes = np.nonzero(cnt)[0]
                Bs = B[np.ix_(modes, modes)]
                alpha = (rng.normal(size=B.shape[0]) + 1j * rng.normal(size=B.shape[0])) * 0.7
                gam = (B @ alpha.conj())[modes]; col = B[modes, rng.integers(B.shape[0])]
                out.append({"kind": kind, "N": N, "counts": cnt[modes].tolist(), "B_re": Bs.real.tolist(),
                            "B_im": Bs.imag.tolist(), "g_re": gam.real.tolist(), "g_im": gam.imag.tolist(),
                            "c_re": col.real.tolist(), "c_im": col.imag.tolist()})
                got += 1
        print(f"{kind}: {sum(p['kind'] == kind for p in out)} patterns", flush=True)
    json.dump({"patterns": out}, open(args.out, "w"))
    print(f"wrote {args.out}")


def run(args):
    import torch

    import photonic_engines.gaussian.hafnian as H
    from photonic_engines.gaussian.hafnian import expand
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if cuda:
        from photonic_engines.gaussian import sieve_triton
        sieve_triton.warmup(dev)
    P = json.load(open(args.patterns))["patterns"]
    rows = []
    for i, p in enumerate(P):
        if p["N"] > args.max_n:
            continue
        B = np.array(p["B_re"]) + 1j * np.array(p["B_im"]); counts = p["counts"]
        g = np.array(p["g_re"]) + 1j * np.array(p["g_im"]); c = np.array(p["c_re"]) + 1j * np.array(p["c_im"])
        A, _ = expand(B, g, counts); lab = np.repeat(np.arange(len(counts)), counts)
        D0 = g[lab][None, :]; D1 = c[lab][None, :]
        res = {}
        for mode in ("random", "v28", "auto", "cross", "dd_cross", "dd_auto", "prec_auto") if args.dd else \
                ("random", "v28", "auto", "cross"):
            if mode == "dd_auto" and p["N"] > args.dd_max_n:
                continue
            tol0 = H._ROUND_TOL
            H._ROUND_TOL = {"v28": None, "random": 0.0}.get(mode, tol0)
            H.LOG = []
            try:
                if cuda:
                    torch.cuda.synchronize()
                t = time.perf_counter()
                val, ab = H.lhaf_poly_multi(A, D0, D1, device=dev, return_abs=True, labels=lab,
                                            pairing="cross" if mode in ("cross", "dd_cross") else "auto",
                                            precision={"dd_cross": "dd", "dd_auto": "dd", "prec_auto": "auto"}.get(mode, "fp64"))
                if cuda:
                    torch.cuda.synchronize()
                sec = time.perf_counter() - t
            finally:
                H._ROUND_TOL = tol0
            log = H.LOG[-1]; H.LOG = None
            v = val[0].cpu().numpy(); a = ab[0].cpu().numpy()
            res[mode] = (sec, float(a.max() / max(np.abs(v).max(), 1e-300)), log[4], bool(log[5]), v, log[6])
        ref = res["dd_cross"][4] if "dd_cross" in res else res["random"][4]; scale = np.abs(ref).max()
        for mode, (sec, kap, terms, redo, v, prec) in res.items():
            rows.append({"i": i, "kind": p["kind"], "N": p["N"], "max_count": max(counts), "mode": mode, "seconds": sec,
                         "kappa": kap, "N_eps_kappa": p["N"] * (2.0 ** -104 if prec == "dd" else EPS) * kap, "terms": terms,
                         "redo": redo, "precision": prec,
                         "diff_vs_ref": float(np.abs(v - ref).max() / scale)})
        if args.csv:                                                    # after every pattern: a timeout keeps what ran
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader(); w.writerows(rows)
        r = res
        print(f"{p['kind']:5s} N={p['N']:2d} max count {max(counts):2d}: " + "  ".join(
            f"{m} {r[m][0]:8.3f} s kappa {r[m][1]:.1e} terms {r[m][2]:.1e}{' REDO' if r[m][3] else ''}"
            f"{' dd' if r[m][5] == 'dd' else ''}" for m in r), flush=True)
    if args.csv:
        print(f"wrote {args.csv}")
    for kind in ("heavy", "light"):
        for mode in ("v28", "auto", "cross", "dd_cross", "dd_auto", "prec_auto"):
            sel = [(x, y) for x in rows if x["kind"] == kind and x["mode"] == mode
                   for y in rows if y["i"] == x["i"] and y["mode"] == "v28"]
            if sel:
                sp = [y["seconds"] / x["seconds"] for x, y in sel]
                print(f"{kind} {mode}: speed-up over v28 median x{np.median(sp):.2f} (min {min(sp):.2f}, max {max(sp):.2f}); "
                      f"max N eps kappa {max(x['N_eps_kappa'] for x, _ in sel):.1e}; redo {sum(x['redo'] for x, _ in sel)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("patterns")
    a.add_argument("--data", required=True); a.add_argument("--n", default="32,36,40,44,48")
    a.add_argument("--per", type=int, default=4); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    b = sub.add_parser("run")
    b.add_argument("--patterns", required=True); b.add_argument("--device", default="cuda")
    b.add_argument("--max-n", type=int, default=1000); b.add_argument("--csv", default=None)
    b.add_argument("--dd", action="store_true", help="also dd_cross (cross in double-double), dd_auto (dd, automatic "
                   "pairing, to --dd-max-n) and prec_auto (precision='auto'); the reference becomes dd_cross")
    b.add_argument("--dd-max-n", type=int, default=36)
    args = ap.parse_args()
    patterns(args) if args.cmd == "patterns" else run(args)


if __name__ == "__main__":
    main()
