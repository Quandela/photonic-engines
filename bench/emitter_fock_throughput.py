"""Emitter-kernel throughput test: the emitter kernel `cuda_emitter.emit_multi`
against the Fock kernel `cuda_cofactors.cof_multi` per needed operation, at the chain's last steps, n = 30 ... 36.

    python3 bench/emitter_fock_throughput.py --n 30,32,34,36 --steps=-1,0 --streams 8 --csv $RESULTS_DIR/emitter_fock_throughput.csv

Per n (m = 2n, Haar U, seed n) and step k (K = k columns, K1 = k - 1 drawn rows; default k = n - 1, n):
  * the SAME kept slots and drawn rows feed both kernels; with photon j in mode j (the frozen-spin Fock state of
    ALGORITHMS §9.2 row 4) the Fock kernel's columns are the kept slots, so both kernels walk the same 2020-collapsed
    Gray terms (N_s of `multiplicity_tables`) -- work units = sum_s N_s * K, identical by construction (checked);
  * emitter on the frozen-spin chi = 2 Fock state (`--emitter-state fock` of find_threshold.py) and on the linear
    cluster (the frontier's 30.9 row; Lambda pure terms per stream instead of 1 -- units scale with Lambda);
  * the timed region is the kernel call with its gather (`emitter_step_weights(timing=...)["kernel"]` /
    `cofactor_partials_cuda`), best of --reps after one warm call (compiles happen in warmup, before any timing);
  * ns per (term x column) for each kernel and their ratio r_t, tier / TPL / GS / NL, registers, local memory and the
    theoretical occupancy of each kernel, the achieved fp64 rate on the executed count; the payload factor of
    ALGORITHMS §9.2 row 4 (minimum 88 + 36 NL for chi = 2 vs Fock 22-24 per term and column) gives the throughput
    ratio per needed operation = r_t / payload factor -- pass <= 1.3.
Optionally (--sweep-k) the emitter (TPL, GS) grid at tiers 32 / 36 (unmeasured on the H100 since 20260927_110150).
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
from photonic_engines.clifford import cuda_cofactors as cc
from photonic_engines.clifford import cuda_emitter as ce
from photonic_engines.clifford.emitter_terms import batched_gap_terms
from photonic_engines.clifford.emitter_triton import emitter_step_weights

# fp64 real ops per Gray term and column (ALGORITHMS §9.2 row 4):
FOCK_MIN = (22.0, 24.0)    # Fock minimum: division trick 22 (+ one division) / prefix-suffix chains 24
FOCK_EXEC = 22.0           # cof_multi as executed (+ one division, not counted)


def emit_min(nl):          # algorithmic minimum of a genuine chi = 2 state
    return 88.0 + 36.0 * nl


def emit_exec(nl, tpl):    # cuda_emitter as executed: minimum + 8 (w scaling) + 56 (2 x 2 block scan when TPL > 1)
    return emit_min(nl) + 8.0 + (56.0 if tpl > 1 else 0.0)


def kernel_attrs(fn, block=128):
    """(registers, local bytes, theoretical occupancy in warps per SM of 64) of a cupy kernel."""
    regs = int(getattr(fn, "num_regs", 0) or 0)
    lmem = int(getattr(fn, "local_size_bytes", 0) or 0)
    if regs <= 0:
        return regs, lmem, float("nan")
    per_block = ((regs * 32 + 255) // 256 * 256) * (block // 32)        # register allocation unit 256 per warp
    blocks = min(65536 // per_block, 32, 2048 // block)
    return regs, lmem, min(64, blocks * block // 32)


def frozen_fock(n, m):
    ops = np.zeros((n, 1, 2, 2), dtype=np.complex128)
    ops[:, 0, 0, 0] = 1.0
    return EmitterMPS(ops=ops, modes=tuple((k,) for k in range(n)), s0=np.array([1.0, 0.0], dtype=np.complex128), m=m)


def sync():
    torch.cuda.synchronize()


def time_emitter(U, st, kept, rows, S, reps, lanes):
    final = torch.tensor([1, 0], dtype=torch.complex128, device=U.device).unsqueeze(0).expand(S, 2)
    s0, mats, f, t2s = batched_gap_terms(st, kept, final)
    k = kept.shape[1]
    N = ce.emitter_tables(rows, k - 1)[3]
    units = float(N[t2s].sum().item()) * k
    tm = {"kernel": 0.0}
    w, lpt = emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, lanes, impl="cuda", timing=tm)    # warm
    best = math.inf
    for _ in range(reps):
        tm = {"kernel": 0.0}
        emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, lanes, impl="cuda", timing=tm)
        best = min(best, tm["kernel"])
    return best, units, s0.shape[0] / S, lpt, w


def time_fock(U, n, kept, rows, S, reps, lanes):
    k = kept.shape[1]
    A2 = torch.view_as_real(U[:, :n].contiguous()).unsqueeze(0).contiguous()                 # [1, m, n, 2]
    dp = torch.zeros(S, dtype=torch.long, device=U.device)
    from photonic_engines.clifford.cofactors_triton import multiplicity_tables
    units = float(multiplicity_tables(rows, k - 1)[3].sum().item()) * k
    cc.cofactor_partials_cuda(A2, rows, kept, dp, S, k - 1, k, lanes)
    sync()
    best = math.inf
    for _ in range(reps):
        t = time.time()
        cc.cofactor_partials_cuda(A2, rows, kept, dp, S, k - 1, k, lanes)
        sync()
        best = min(best, time.time() - t)
    return best, units


FIELDS = ["n", "k", "kernel", "state", "nl", "tier", "tpl", "gs", "regs", "lmem", "occ_warps", "streams",
          "terms_per_stream", "ms", "units", "ns_per_unit", "gflops_exec", "ratio_t", "payload_min", "per_min_op",
          "err_vs_000"]


def write_csv(path, out):
    """Rewritten after every point, so a failure late in the run keeps what was measured."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        wr.writeheader()
        wr.writerows(out)


def variants(args, out):
    """Kernel variant candidates: every variant on the same instances (n in --variant-n, the steps of --steps, both states),
    ns / unit, ratio to the Fock kernel on the same rows and per needed op, max |w - w_000| / max |w_000|."""
    S = args.streams
    vs = args.variants.split(",")
    assert vs[0][:3] == "000", "the first variant must be the production kernel (000 / 0001 / 00014)"
    inst = []
    for n in [int(x) for x in args.variant_n.split(",") if x]:
        m = 2 * n
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].cuda()
        for off in [int(x) for x in args.steps.split(",")]:
            k = n + off
            g = torch.Generator().manual_seed(1000 * n + k)
            kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :k], dim=1).values.cuda()
            rows = torch.randint(0, m, (S, k - 1), generator=g).cuda()
            tf, uf = time_fock(U, n, kept, rows, S, args.reps, args.target_lanes)
            for name, st in (("fock", frozen_fock(n, m)), ("cluster", EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m))):
                inst.append((n, k, name, st, U, kept, rows, tf / uf * 1e9))
    ref = {}
    tiers = sorted({ce.tier(k)[0] for _n, k, *_r in inst})
    for v in vs:
        sym = int(v.startswith("S"))                                 # "S" prefix: PE_SYM (TPL 2 lanes, no block scan)
        d = v[sym:]
        wq, msh, accsh = (int(c) for c in d[:3])
        ce.set_variant(wq, msh, accsh, sym)
        gs = int(d[3]) if len(d) > 3 else None                       # 4th digit: GS of the measured tiers (seg)
        tpl = int(d[4:]) if len(d) > 4 else None                     # 5th+: threads per lane of those tiers
        if gs is not None:
            ce.set_config({kp: (tpl or ce.tier(kp)[1], gs, None) for kp in tiers}, regime="seg")
        t0 = time.time()
        ce.warmup(tiers=tiers)
        print(f"variant {v} (sym {sym} wq {wq} msh {msh} accsh {accsh} gs {gs if gs is not None else 'table'} "
              f"tpl {tpl or 'table'}): "
              f"compile {time.time() - t0:.0f} s (tiers {tiers})", flush=True)
        for n, k, name, st, U, kept, rows, nsf in inst:
            try:
                te, ue, lam, lpt, w = time_emitter(U, st, kept, rows, S, args.reps, args.target_lanes)
            except Exception as e:
                print(f"  n={n} k={k} {name}: failed ({type(e).__name__}: {str(e)[:200]})", flush=True)
                continue
            key = (n, k, name)
            if v == vs[0]:
                ref[key] = w
            err = float((w - ref[key]).abs().max() / ref[key].abs().max()) if key in ref else float("nan")
            nl = st.num_labels
            KPe, TPLe, GSe = ce.tier(k, lpt)
            blk = (64 if -(-KPe // TPLe) <= 10 else 32) if accsh else 128
            rege, lme, occe = kernel_attrs(ce._get_kernel(KPe, nl, ce.regime_of(lpt)), blk)
            nse = te / ue * 1e9
            r = nse / nsf
            pmin = (emit_min(nl) / FOCK_MIN[1], emit_min(nl) / FOCK_MIN[0])
            print(f"  n={n} k={k} {name:7s} NL {nl}: regs {rege} lmem {lme} B occ {occe} warps  {te * 1e3:9.2f} ms  "
                  f"{nse:.4f} ns/unit  r_t {r:5.2f} -> per needed op {r / pmin[1]:.2f}-{r / pmin[0]:.2f}  "
                  f"err vs 000 {err:.1e}", flush=True)
            out.append({"n": n, "k": k, "kernel": f"emitter_v{v}", "state": name, "nl": nl, "tier": KPe, "tpl": TPLe,
                        "gs": GSe, "regs": rege, "lmem": lme, "occ_warps": occe, "streams": S, "terms_per_stream": lam,
                        "ms": te * 1e3, "units": ue, "ns_per_unit": nse, "ratio_t": r,
                        "payload_min": f"{pmin[0]:.3f}-{pmin[1]:.3f}", "per_min_op": f"{r / pmin[1]:.3f}-{r / pmin[0]:.3f}",
                        "err_vs_000": err})
            write_csv(args.csv, out)
        ce.set_config(None)
    ce.set_variant(0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="30,32,34,36")
    ap.add_argument("--steps", default="-1,0", help="k = n + offset (default: the last two steps)")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--target-lanes", type=int, default=1 << 17)
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--sweep-k", default="", help="emitter (TPL, GS) grid at these K (= n = k), frozen-spin state")
    ap.add_argument("--variants", default="", help="emitter kernel variants as wq/msh/accsh digits, then optionally the GS "
                    "digit and the TPL of the measured tiers; first = the production kernel (000 / 0001 / 00014, the "
                    "reference of the max |dw| / max |w| column), e.g. 0001,1011,10118")
    ap.add_argument("--variant-n", default="34")
    ap.add_argument("--tpls", default="2,4,8")
    ap.add_argument("--gs", default="0,1")
    ap.add_argument("--csv", default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "emitter_fock_throughput.csv"))
    args = ap.parse_args()
    dev = "cuda"
    S = args.streams
    print(f"device {torch.cuda.get_device_name()}  streams {S}  reps {args.reps}", flush=True)
    t0 = time.time()
    cc.warmup()
    ce.warmup()
    print(f"warmup (both kernels, every tier): {time.time() - t0:.1f} s", flush=True)
    out = []
    for n in [int(x) for x in args.n.split(",") if x]:
        m = 2 * n
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].to(dev)
        states = {"fock": frozen_fock(n, m)}
        if not args.no_cluster:
            states["cluster"] = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
        for off in [int(x) for x in args.steps.split(",")]:
            k = n + off
            g = torch.Generator().manual_seed(1000 * n + k)
            kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :k], dim=1).values.to(dev)
            rows = torch.randint(0, m, (S, k - 1), generator=g).to(dev)
            tf, uf = time_fock(U, n, kept, rows, S, args.reps, args.target_lanes)
            nsf = tf / uf * 1e9
            KPf, TPLf = cc.tier(k)
            regf, lmf, occf = kernel_attrs(cc._get_kernel(KPf))
            gff = FOCK_EXEC / nsf
            print(f"n={n} k={k}  fock: tier {KPf} TPL {TPLf} regs {regf} lmem {lmf} B occ {occf} warps  "
                  f"{tf * 1e3:9.2f} ms  {uf:.3e} units  {nsf:.4f} ns/unit  {gff:.1f} Gflop/s (22/unit)", flush=True)
            out.append({"n": n, "k": k, "kernel": "fock", "state": "fock", "nl": 1, "tier": KPf, "tpl": TPLf, "gs": "",
                        "regs": regf, "lmem": lmf, "occ_warps": occf, "streams": S, "terms_per_stream": 1,
                        "ms": tf * 1e3, "units": uf, "ns_per_unit": nsf, "gflops_exec": gff, "ratio_t": 1.0,
                        "payload_min": "", "per_min_op": ""})
            write_csv(args.csv, out)
            for name, st in states.items():
                nl = st.num_labels
                try:
                    te, ue, lam, lpt, _w = time_emitter(U, st, kept, rows, S, args.reps, args.target_lanes)
                except Exception as e:
                    print(f"  emitter[{name}] failed ({type(e).__name__}: {str(e)[:200]})", flush=True)
                    continue
                KPe, TPLe, GSe = ce.tier(k, lpt)
                rege, lme, occe = kernel_attrs(ce._get_kernel(KPe, nl, ce.regime_of(lpt)))
                nse = te / ue * 1e9
                r = nse / nsf
                pmin = (emit_min(nl) / FOCK_MIN[1], emit_min(nl) / FOCK_MIN[0])
                gfe = emit_exec(nl, TPLe) / nse
                if name == "fock" and abs(ue - uf) > 0.5:
                    print(f"  WARNING units differ: emitter {ue:.6e} vs fock {uf:.6e}", flush=True)
                print(f"  emitter[{name:7s}] NL {nl} tier {KPe} TPL {TPLe} GS {GSe} regs {rege} lmem {lme} B occ {occe} "
                      f"warps: {te * 1e3:9.2f} ms  {ue:.3e} units  terms/stream {lam:5.2f}  lanes/term {lpt:.1f}  "
                      f"{nse:.4f} ns/unit  {gfe:.1f} Gflop/s exec  r_t = {r:5.2f}  payload min {pmin[0]:.2f}-{pmin[1]:.2f}"
                      f"  -> per needed op {r / pmin[1]:.2f}-{r / pmin[0]:.2f} (pass <= 1.3)", flush=True)
                out.append({"n": n, "k": k, "kernel": "emitter", "state": name, "nl": nl, "tier": KPe, "tpl": TPLe,
                            "gs": GSe, "regs": rege, "lmem": lme, "occ_warps": occe, "streams": S,
                            "terms_per_stream": lam, "ms": te * 1e3, "units": ue, "ns_per_unit": nse,
                            "gflops_exec": gfe, "ratio_t": r, "payload_min": f"{pmin[0]:.3f}-{pmin[1]:.3f}",
                            "per_min_op": f"{r / pmin[1]:.3f}-{r / pmin[0]:.3f}"})
                write_csv(args.csv, out)
    if args.variants:
        variants(args, out)
    for K in [int(x) for x in args.sweep_k.split(",") if x]:
        n = k = K
        m = 2 * n
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].to(dev)
        g = torch.Generator().manual_seed(1000 * n + k)
        kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :k], dim=1).values.to(dev)
        rows = torch.randint(0, m, (S, k - 1), generator=g).to(dev)
        KP = ce.tier(k)[0]
        for tpl in [int(x) for x in args.tpls.split(",")]:
            for gs in [int(x) for x in args.gs.split(",")]:
                try:
                    ce.set_config({KP: (tpl, gs, None)}, regime="seg")
                    tw = time.time()
                    te, ue, lam, lpt, _w = time_emitter(U, frozen_fock(n, m), kept, rows, S, args.reps, args.target_lanes)
                    reg, lm, occ = kernel_attrs(ce._get_kernel(KP, 1, ce.regime_of(lpt)))
                    print(f"sweep K={k} tier {KP} TPL={tpl} GS={gs}: regs {reg} lmem {lm} B occ {occ} warps  "
                          f"{te * 1e3:9.2f} ms  {te / ue * 1e9:.4f} ns/unit  (incl. compile {time.time() - tw:.0f} s)",
                          flush=True)
                    out.append({"n": n, "k": k, "kernel": "emitter_sweep", "state": "fock", "nl": 1, "tier": KP,
                                "tpl": tpl, "gs": gs, "regs": reg, "lmem": lm, "occ_warps": occ, "streams": S,
                                "terms_per_stream": lam, "ms": te * 1e3, "units": ue, "ns_per_unit": te / ue * 1e9,
                                "gflops_exec": emit_exec(1, tpl) / (te / ue * 1e9)})
                    write_csv(args.csv, out)
                except Exception as e:
                    print(f"sweep K={k} TPL={tpl} GS={gs}: failed ({type(e).__name__}: {str(e)[:100]})", flush=True)
        ce.set_config(None)
    write_csv(args.csv, out)
    print("wrote", args.csv, f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
