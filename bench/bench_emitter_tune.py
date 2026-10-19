"""Sweep the launch configuration of the Triton emitter cofactor kernels per K_PAD tier.

    python3 bench/bench_emitter_tune.py --device cuda --impl seq --K 8,16,24 --streams 64
    python3 bench/bench_emitter_tune.py --device cuda --impl scan --K 8,16,24 --segs 2,4,8,16 --warps 2,4

For each K (step k = K-2 of an n = K linear-cluster state, one term set from batched_gap_terms)
times `emitter_step_weights` (kernel + torch post) for every config and prints the best per
tier; paste the table into `_CFG` in emitter_triton.py.
  impl seq : configs are (SEG, B, mat_mode): B column blocks per lane (threads), SEG lanes per
             program (num_warps = SEG * B / 32), matrices re-read ("scalar") or in registers ("hoist").
  impl scan: configs are (SEG, num_warps).
The step k = K-2 is capped at --max-K (default 22: 2^20 Glynn terms per pure term; k = 28 took > 1 h).
"""
import argparse
import itertools
import math
import time

import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford import emitter_triton as et
from photonic_engines.clifford.emitter_terms import batched_gap_terms
from photonic_engines.clifford.emitter_triton import emitter_step_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--impl", default="seq", choices=["seq", "scan"])
    ap.add_argument("--K", default="8,16,24")
    ap.add_argument("--max-K", type=int, default=22)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--segs", default=None, help="scan: SEG list; seq: SEG list (default: 128/B, 64/B, 32/B)")
    ap.add_argument("--warps", default="2,4", help="scan only")
    ap.add_argument("--blocks", default=None, help="seq: B list (default: K_PAD/4, K_PAD/2, K_PAD, capped to 32)")
    ap.add_argument("--mat-modes", default="scalar,hoist")
    ap.add_argument("--maxnreg", default="0", help="comma list of Triton register caps (0 = let "
                    "Triton choose).  num_warps is DERIVED from SEG*B here, so this is the only "
                    "independent occupancy knob -- on the Fock kernel it was worth 1.32x at the "
                    "widest tier and the wrong value cost 1.9x")
    ap.add_argument("--stages", default="0", help="comma list of num_stages (0 = Triton default)")
    ap.add_argument("--target-lanes", type=int, default=1 << 17)
    args = ap.parse_args()
    dev = args.device
    best, best_by_k = {}, {}
    for K in [int(x) for x in args.K.split(",")]:
        n, m = K, 2 * K
        if args.max_K < K - 2:
            print(f"K={K}: step k = K-2 = {K - 2} > --max-K {args.max_K}: skipped", flush=True)
            continue
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m, final=[1, 0])
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(K)))[0].to(dev)
        S = args.streams
        g = torch.Generator().manual_seed(1)
        k = max(1, K - 2)                                   # a late step: K1 = k-1 rows, full Glynn cost
        kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :k], dim=1).values.to(dev)
        rows = torch.randint(0, m, (S, k - 1), generator=g).to(dev)
        final = torch.tensor([1, 0], dtype=torch.complex128, device=dev).unsqueeze(0).expand(S, 2)
        s0, mats, f, t2s = batched_gap_terms(st, kept, final)
        # True work unit: the seq kernel walks the COLLAPSED 2020 count N per stream, not
        # 2^(k-2).  Normalising by 2^(k-2) (as the first version did) mixes the collapse gain
        # into the rate and makes it incomparable across K.
        _, _, _, _, Ntab, _ = et.emitter_multiplicity_tables(rows, max(k - 1, 0))
        units = float(Ntab[t2s.to(Ntab.device)].sum().item()) * k
        print(f"  work units = sum_terms N_s x K = {units:.3e}   "
              f"(2018 normaliser would say {s0.shape[0] * 2.0 ** max(k - 2, 0) * k:.3e}, "
              f"collapse gain {s0.shape[0] * 2.0 ** max(k - 2, 0) * k / max(units, 1):.2f}x)", flush=True)
        K_PAD = max(2, 1 << (k - 1).bit_length())
        print(f"K={k} (K_PAD={K_PAD}) streams={S} terms={s0.shape[0]} Glynn terms per term=2^{k - 2}", flush=True)
        if args.impl == "seq":
            # d = 8 included since 2026-09-16: the old default {K_PAD/4, K_PAD/2, K_PAD} left
            # B = 4 OUT OF THE GRID at K_PAD = 32, and B = 4 is the optimum at K = 18 (3.74 ms
            # against 4.65 for B = 8).  The first sweep therefore "found" B = 8 there because it
            # never tried B = 4, and the disagreement with the second sweep was a grid artefact,
            # not the stream-count dependence it was first read as.
            blocks = ([int(x) for x in args.blocks.split(",")] if args.blocks else
                      sorted({max(1, min(32, K_PAD // d)) for d in (8, 4, 2, 1)}))
            configs = []
            for B in blocks:
                segs = [int(x) for x in args.segs.split(",")] if args.segs else sorted({max(1, t // B) for t in (128, 64, 32)})
                for seg, mm in itertools.product(segs, args.mat_modes.split(",")):
                    configs.append((seg, B, mm))
        else:
            segs = [int(x) for x in (args.segs or "2,4,8,16,32").split(",")]
            configs = [(seg, w, "-") for seg, w in itertools.product(segs, [int(x) for x in args.warps.split(",")])]
        regs = [int(x) for x in args.maxnreg.split(",")]
        stages = [int(x) for x in args.stages.split(",")]
        configs = [(seg, w, mm, r, st) for (seg, w, mm) in configs
                   for r in regs for st in stages]
        results = []
        for seg, w, mm, reg, stg in configs:
            label = (f"SEG={seg:3d} {'B' if args.impl == 'seq' else 'warps'}={w:2d} {mm:6s} "
                     f"reg={reg or '-':>4} stg={stg or '-':>3}")
            try:
                et.SEQ_MAT_MODE = mm if mm != "-" else et.SEQ_MAT_MODE
                et.set_launch(maxnreg=(reg or None), num_stages=(stg or None))
                t0 = time.time()
                emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, args.target_lanes, seg=seg, num_warps=w, impl=args.impl)
                if dev == "cuda": torch.cuda.synchronize()
                warm = time.time() - t0
                t = time.time()
                for _ in range(3):
                    emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, args.target_lanes, seg=seg, num_warps=w, impl=args.impl)
                if dev == "cuda": torch.cuda.synchronize()
                dt = (time.time() - t) / 3
                results.append((dt, seg, w, mm, reg, stg))
                print(f"  {label}: {dt * 1e3:9.2f} ms  {units / dt / 1e9:7.2f} G units/s"
                      f"   (first call incl. compile {warm:.1f} s)", flush=True)
            except Exception as e:
                print(f"  {label}: failed ({type(e).__name__}: {str(e)[:80]})", flush=True)
        if results:
            dt, seg, w, mm, reg, stg = min(results)
            base = min((r for r in results if not r[4] and not r[5]), default=None)
            best[K_PAD] = (seg, w) if args.impl == "scan" else (seg, w, mm)
            best_by_k[k] = ((seg, w) if args.impl == "scan" else (seg, w, mm), K_PAD)
            gain = f"  ({base[0] / dt:.2f}x the best untuned-launch config)" if base else ""
            print(f"  best: SEG={seg} {'B' if args.impl == 'seq' else 'warps'}={w} {mm} "
                  f"maxnreg={reg or 'default'} num_stages={stg or 'default'} "
                  f"({dt * 1e3:.2f} ms){gain}", flush=True)
            if reg or stg:
                print(f"  -> set with: emitter_triton.set_launch(maxnreg={reg or None}, "
                      f"num_stages={stg or None})", flush=True)
    et.set_launch(maxnreg=None, num_stages=None)

    # A K_PAD tier holds a range of K; when two K in one tier want different configs the tier
    # cannot express both and the loser pays the full spread (1.29x at K = 18, run 20260916_155718).
    # Emit the tier table for the agreeing tiers and exact-K entries for the rest.
    by_tier = {}
    for k, (cfg, KP) in sorted(best_by_k.items()):
        by_tier.setdefault(KP, []).append((k, cfg))
    tier_tbl, exact_tbl = {}, {}
    for KP, items in sorted(by_tier.items()):
        cfgs = {c for _, c in items}
        if len(cfgs) == 1:
            tier_tbl[KP] = items[0][1]
        else:
            # the largest K in the tier carries the most cost, so it keeps the tier entry
            kmax, cmax = max(items, key=lambda x: x[0])
            tier_tbl[KP] = cmax
            for k, c in items:
                if c != cmax:
                    exact_tbl[k] = c
            print(f"  NOTE tier {KP}: K = {[k for k, _ in items]} disagree "
                  f"({[c for _, c in items]}); K = {kmax} keeps the tier entry.")
    print(f"_CFG[{args.impl!r}] update:", tier_tbl)
    if exact_tbl:
        print(f"_CFG_BY_K[{args.impl!r}] update:", exact_tbl)
    if len(best_by_k) < 2:
        print("  (one K measured: sweep at least two K per tier to see whether the tier suffices)")


if __name__ == "__main__":
    main()
