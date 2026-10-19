"""Rounding error of the fp64 emitter cofactor kernels at the frontier sizes (Supplement C; cross-engine
review item #4, docs/EMITTER_SAMPLER.md §9).

Two measurements on the same instances (Haar U, m = 2n, caterpillar state, one chain step at K kept slots
with K - 1 drawn rows — distinct rows, or one row repeated with --collide):

  --dump  (GPU)  the per-term matrix cofactors K_c [Λ, K, 2, 2] and the step weights w [S, m] (what the
                 draw sees) from the Triton seq kernel, the same kernel with another lane segmentation, and
                 the CUDA thread-per-lane kernel; pairwise disagreements (a lower bound of the fp64 error,
                 the method of bench/cofactor_precision.py); the instance and the cofactors saved to an npz
  --reference (CPU, no GPU needed)  the CUDA kernel code compiled for the CPU in x87 long double
                 (64-bit mantissa, cuda_emitter.emitter_cofactor_partials_emul(real="long double")) on a few
                 terms of the dumped instance: the TRUE fp64 error of each kernel on those terms (every
                 addend of the collapsed Glynn sum, in a different, longer accumulation order), to 1e-19.
                 ~2 min per term at K = 26, ~8 at K = 28, ~30 at K = 30 on one core.

    python3 bench/emitter_precision.py --dump --K 26,28,30 --streams 4 --out results/emitter_precision      # GPU
    python3 bench/emitter_precision.py --reference results/keep/emitter_precision_k28.npz --terms 0,1      # CPU

CSV rows: K, mode, rows, rel_cof (max over terms and columns of |ΔK_c| / max |K_c| of the term), rel_w (max over
streams of |Δw| / max w of the stream), seconds.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_terms import batched_gap_terms
from photonic_engines.clifford.emitter_triton import emitter_step_weights


def rel_cof(a, b):
    """max over terms of max_c |a - b| / max_c |a| (complex [Λ, K, 2, 2])."""
    num = (a - b).abs().flatten(1).max(1).values
    den = a.abs().flatten(1).max(1).values.clamp_min(1e-300)
    return float((num / den).max())


def rel_w(a, b):
    num = (a - b).abs().max(1).values
    den = a.abs().max(1).values.clamp_min(1e-300)
    return float((num / den).max())


def instance(K, S, seed, collide, dev):
    n, m = K, 2 * K
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.3 * j) for j in range(n)], m=m, final=[0.6, 0.8j])
    U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)))[0]
    g = torch.Generator().manual_seed(seed + 1)
    kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :K], dim=1).values
    rows = torch.stack([torch.randperm(m, generator=g)[:K - 1] for _ in range(S)])            # distinct rows
    if collide:
        rows[:, 1] = rows[:, 0]                                                                # one row repeated
    final = torch.as_tensor(st.final, dtype=torch.complex128).unsqueeze(0).expand(S, 2)
    s0, mats, f, t2s = batched_gap_terms(st, kept.to(dev), final.to(dev), U=U.to(dev))
    return st, U, kept, rows, s0, mats, f, t2s


def dump(args):
    dev = args.device
    out_rows = []
    for K in [int(x) for x in args.K.split(",")]:
        S = args.streams
        st, U, kept, rows, s0, mats, f, t2s = instance(K, S, args.seed, args.collide, dev)
        res = {}
        cuda_impl = "cuda" if dev == "cuda" else "cuda_emul"                       # --device cpu: the kernel code on the CPU
        for name, impl, ttl in (("seq_L", "seq", 1 << 17), ("seq_l", "seq", 1 << 15), ("cuda", cuda_impl, 1 << 17)):
            cof = {}
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            w, _ = emitter_step_weights(U.to(dev), st, kept.to(dev), rows.to(dev), s0, mats, f, t2s, S,
                                        target_total_lanes=ttl, impl=impl, cof=cof)
            if dev == "cuda":
                torch.cuda.synchronize()
            res[name] = (cof["Kc"].cpu(), w.cpu(), time.time() - t0)
            print(f"K={K} {name:6s}: {res[name][2]:7.1f} s", flush=True)
        for mode, (a, b) in (("seq_L_vs_l", ("seq_L", "seq_l")), ("cuda_vs_seq", ("cuda", "seq_L"))):
            rc, rw = rel_cof(res[a][0], res[b][0]), rel_w(res[a][1], res[b][1])
            out_rows.append({"K": K, "mode": mode, "rows": "collide" if args.collide else "distinct", "rel_cof": rc,
                             "rel_w": rw, "seconds": res[a][2] + res[b][2]})
            print(f"K={K} {mode:12s}: rel_cof {rc:.2e}  rel_w {rw:.2e}", flush=True)
        tag = "_collide" if args.collide else ""
        np.savez(f"{args.out}_k{K}{tag}.npz", K=K, S=S, seed=args.seed, collide=int(args.collide), U=U.numpy(),
                 kept=kept.numpy(), rows=rows.numpy(), s0=s0.cpu().numpy(), mats=mats.cpu().numpy(), f=f.cpu().numpy(),
                 term2stream=t2s.cpu().numpy(), Kc_seq=res["seq_L"][0].numpy(), Kc_seq_l=res["seq_l"][0].numpy(),
                 Kc_cuda=res["cuda"][0].numpy(), w_seq=res["seq_L"][1].numpy(), w_cuda=res["cuda"][1].numpy())
        print(f"wrote {args.out}_k{K}{tag}.npz", flush=True)
    with open(f"{args.out}{'_collide' if args.collide else ''}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wr.writeheader(); wr.writerows(out_rows)
    print("wrote", f"{args.out}{'_collide' if args.collide else ''}.csv")


def reference(args):
    """Long-double emulation of the CUDA kernel on a few terms of a dumped instance; true error of each dump."""
    from photonic_engines.clifford import cuda_emitter
    from photonic_engines.clifford.cofactors_triton import plan_multi
    from photonic_engines.clifford.emitter_triton import _modes_tensor
    d = np.load(args.reference)
    K, S = int(d["K"]), int(d["S"])
    n, m = K, 2 * K
    st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.3 * j) for j in range(n)], m=m, final=[0.6, 0.8j])
    U = torch.as_tensor(d["U"]); kept = torch.as_tensor(d["kept"]); rows = torch.as_tensor(d["rows"])
    s0 = torch.as_tensor(d["s0"]); mats = torch.as_tensor(d["mats"]); f = torch.as_tensor(d["f"])
    t2s = torch.as_tensor(d["term2stream"])
    terms = [int(x) for x in args.terms.split(",")]
    Lab = mats.shape[2]
    K1 = rows.shape[1]
    modes = _modes_tensor(st, torch.device("cpu"))
    cm = modes[kept]                                                                               # [S, K, L]
    colmode = torch.zeros((S, K, 2), dtype=torch.long); colmode[:, :, :Lab] = cm
    m2 = torch.zeros((mats.shape[0], K, 2, 2, 2), dtype=torch.complex128); m2[:, :, :Lab] = mats
    mats2 = torch.view_as_real(m2.contiguous()); s02 = torch.view_as_real(s0.contiguous()); f2 = torch.view_as_real(f.contiguous())
    U2 = torch.view_as_real(U.contiguous())
    tables = cuda_emitter.emitter_tables(rows, K1)
    N = tables[3]; scale = tables[4]
    out = {"term": [], "rel_seq": [], "rel_seq_l": [], "rel_cuda": [], "seconds": []}
    for lam in terms:
        sel = torch.tensor([lam])
        s = int(t2s[lam])
        Lseg, nl, lstart, lane2term, total = plan_multi(N[t2s[sel]], 1)                        # one lane per term
        t0 = time.time()
        part = cuda_emitter.emitter_cofactor_partials_emul(U2, tables, colmode, lane2term, torch.tensor([s]), lstart,
                                                           mats2[sel], s02[sel], f2[sel], K1, K, Lab, Lseg, total,
                                                           tpl=1, gs=1, real="long double")
        Kc_ref = torch.view_as_complex(part.sum(0)) * scale[s]                                  # [K, 2, 2]
        den = float(Kc_ref.abs().max())
        r = {k: float((torch.as_tensor(d[k][lam]) - Kc_ref).abs().max() / den) for k in ("Kc_seq", "Kc_seq_l", "Kc_cuda")}
        dt = time.time() - t0
        out["term"].append(lam); out["rel_seq"].append(r["Kc_seq"]); out["rel_seq_l"].append(r["Kc_seq_l"])
        out["rel_cuda"].append(r["Kc_cuda"]); out["seconds"].append(dt)
        print(f"K={K} term {lam} (stream {s}, {int(N[s])} collapsed addends): seq {r['Kc_seq']:.2e}  seq_l {r['Kc_seq_l']:.2e}  "
              f"cuda {r['Kc_cuda']:.2e}   ({dt:.0f} s)", flush=True)
    csvp = args.csv or os.path.splitext(args.reference)[0] + "_reference.csv"
    with open(csvp, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["K", "term", "stream_addends", "rel_seq", "rel_seq_l", "rel_cuda", "seconds"])
        for i, lam in enumerate(out["term"]):
            wr.writerow([K, lam, int(N[int(t2s[lam])]), out["rel_seq"][i], out["rel_seq_l"][i], out["rel_cuda"][i], out["seconds"][i]])
    print("wrote", csvp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--device", default="cuda", help="--dump: cuda, or cpu (Triton interpreter + the CPU emulation; small K only)")
    ap.add_argument("--reference", default=None, help="npz of a --dump run")
    ap.add_argument("--K", default="26,28,30")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--collide", action="store_true", help="one drawn row repeated (a radix-3 digit in the walk)")
    ap.add_argument("--terms", default="0,1", help="--reference: term indices to recompute in long double")
    ap.add_argument("--out", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "emitter_precision"))
    ap.add_argument("--csv", default=None, help="--reference: output CSV (default next to the npz)")
    args = ap.parse_args()
    if args.dump:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        dump(args)
    elif args.reference:
        reference(args)
    else:
        ap.error("--dump or --reference")


if __name__ == "__main__":
    main()
