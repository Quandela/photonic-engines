"""Tune / compare the CUDA thread-per-lane emitter kernel (clifford/cuda_emitter.py,
docs/EMITTER_SAMPLER.md §8) against the Triton seq kernel, one late step per K.

    python3 bench/bench_cuda_emitter.py --K 18,22,26,30 --streams 8
    python3 bench/bench_cuda_emitter.py --K 26 --tpls 2,4,8 --gs 0,1 --maxreg 0,128,168,255

Per K the step k = K - 2 (K1 = k - 1 drawn rows, as bench_emitter_tune.py) is timed for every
(TPL, GS, maxrregcount) of the grid, checked against the Triton seq kernel (max |dw| / max w),
and the best configuration is printed as the `_CFG_BY_ARCH` entry of its tier.  Work unit =
sum over terms of the collapsed term count N_s times K (same normaliser as bench_emitter_tune).
Writes $RESULTS_DIR/bench_cuda_emitter<tag>.csv.
"""
import argparse
import csv
import itertools
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford import cuda_emitter as ce
from photonic_engines.clifford.emitter_terms import batched_gap_terms
from photonic_engines.clifford.emitter_triton import emitter_step_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default="18,22,26,30")
    ap.add_argument("--n", type=int, default=None, help="instance size (default: n = K, so the step k = K - 2 has a few "
                    "terms per stream and many lanes per term; --n 30 --K 14,16,18 reproduces the middle of the chain: "
                    "thousands of terms per stream, one lane per term -- the 'scatter' regime of cuda_emitter)")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--tpls", default="2,4,8")
    ap.add_argument("--gs", default="0,1")
    ap.add_argument("--maxreg", default="0,128,168,255", help="--maxrregcount values (0 = compiler's choice)")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--target-lanes", type=int, default=1 << 17)
    ap.add_argument("--no-seq", action="store_true", help="skip the Triton seq baseline")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda"
    S = args.streams
    rows_out, table = [], {}
    for K in [int(x) for x in args.K.split(",")]:
        n = args.n or K
        m = 2 * n
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m, final=[1, 0])
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(K)))[0].to(dev)
        g = torch.Generator().manual_seed(1)
        k = max(2, K - 2) if args.n is None else min(K, n)
        kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :k], dim=1).values.to(dev)
        rows = torch.randint(0, m, (S, k - 1), generator=g).to(dev)
        final = torch.tensor([1, 0], dtype=torch.complex128, device=dev).unsqueeze(0).expand(S, 2)
        s0, mats, f, t2s = batched_gap_terms(st, kept, final)
        Ntab = ce.emitter_tables(rows, k - 1)[3]
        units = float(Ntab[t2s].sum().item()) * k
        KP = ce.tier(k)[0]
        from photonic_engines.clifford.cofactors_triton import plan_multi
        _L, nl, _ls, _l2t, tot = plan_multi(Ntab[t2s], args.target_lanes)
        lpt = tot / s0.shape[0]
        print(f"K={k} (tier {KP}) n={n} streams={S} terms={s0.shape[0]} lanes/term={lpt:.1f} (regime {ce.regime_of(lpt)}) "
              f"work units={units:.3e}", flush=True)

        def run(impl, reps):
            t0 = time.time()
            w, _ = emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, args.target_lanes, impl=impl)
            torch.cuda.synchronize()
            warm = time.time() - t0
            t = time.time()
            for _ in range(reps):
                emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, args.target_lanes, impl=impl)
            torch.cuda.synchronize()
            return w, (time.time() - t) / reps, warm

        w_seq, dt_seq = None, float("nan")
        if not args.no_seq:
            w_seq, dt_seq, warm = run("seq", args.reps)
            print(f"  triton seq (tier table)          : {dt_seq * 1e3:9.2f} ms  {units / dt_seq / 1e9:7.2f} G units/s"
                  f"   (first call {warm:.1f} s)", flush=True)
        results = []
        grid = itertools.product([int(x) for x in args.tpls.split(",")], [int(x) for x in args.gs.split(",")],
                                 [int(x) for x in args.maxreg.split(",")])
        for tpl, gs, reg in grid:
            label = f"TPL={tpl} GS={gs} maxreg={reg or '-':>4}"
            try:
                ce.set_config({KP: (tpl, gs, reg or None)})
                w, dt, warm = run("cuda", args.reps)
                err = float((w - w_seq).abs().max() / w_seq.abs().max()) if w_seq is not None else float("nan")
                results.append((dt, tpl, gs, reg))
                rows_out.append({"K": k, "tier": KP, "n": n, "streams": S, "terms": s0.shape[0], "lanes_per_term": round(lpt, 2),
                                 "units": units, "tpl": tpl, "gs": gs,
                                 "maxreg": reg, "ms": dt * 1e3, "g_units_s": units / dt / 1e9, "ms_seq": dt_seq * 1e3,
                                 "speedup_vs_seq": dt_seq / dt, "max_rel_err_vs_seq": err})
                print(f"  {label}: {dt * 1e3:9.2f} ms  {units / dt / 1e9:7.2f} G units/s  x{dt_seq / dt:5.2f} vs seq"
                      f"  err {err:.1e}   (first call incl. compile {warm:.1f} s)", flush=True)
            except Exception as e:
                print(f"  {label}: failed ({type(e).__name__}: {str(e)[:100]})", flush=True)
        ce.set_config(None)
        if results:
            dt, tpl, gs, reg = min(results)
            table[KP] = (tpl, gs, reg or None)
            print(f"  best: TPL={tpl} GS={gs} maxreg={reg or 'default'} ({dt * 1e3:.2f} ms, x{dt_seq / dt:.2f} vs seq)", flush=True)
    print("_CFG_BY_ARCH entry (of the regime printed per K):", table)
    out = os.path.join(os.environ.get("RESULTS_DIR", "results"), f"bench_cuda_emitter{args.tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows_out:
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            wr.writeheader()
            wr.writerows(rows_out)
        print("wrote", out)


if __name__ == "__main__":
    main()
