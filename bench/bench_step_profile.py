"""
Per-step profile of one sample: where the time of a sample actually goes.

Runs the single-sample path of `photonic_engines/clifford/single.py` with a CUDA event
around each of the three launches of each photon step, and reports per step k:

    N_k        exact Gray-term count of the step (the odometer's radix product)
    tier       padded column tier used, and the fraction of the work that is padding
    t_prep     prep_single (multiplicity tables + gathered block)
    t_cof      cofactor kernel
    t_red+draw lane reduction + draw kernel
    Gterms/s   N_k / t_cof  -- the in-situ throughput, to compare with the ceiling measured
               by bench_cofactor_variants.py
    share      cumulative share of the sample's wall time

Three things it settles:
  * how much of a sample is the last one, two, three steps (the chain-rule cost grows by rho
    per photon, so the tail should be ~45 / 70 / 83 %) -- i.e. how narrow the optimisation
    target is, and how few steps a multi-GPU lane split would have to cover;
  * how much the 48/64 tiers lose to padding at n = 40..63, where the frontier
    extrapolation is measured;
  * whether the in-situ cofactor throughput matches the micro-benchmark, or whether the
    per-step fixed cost (3 launches, prep, reduction) is still visible at large n.

Usage
    python bench/bench_step_profile.py --n 34 --mratio 2 --samples 4
    python bench/bench_step_profile.py --n 48 --mratio 1 --samples 2 --csv step48.csv
"""
import argparse
import csv

import torch


def profile_sample(U, in_modes, generator, lanes=1 << 15, block=128):
    import cupy

    from photonic_engines.clifford.cofactors_triton import _binom_table, draw_step
    from photonic_engines.clifford.cuda_cofactors import _get_kernel, get_prep_kernel, tier

    dev = U.device
    m = U.shape[-1]
    n = len(in_modes)
    A = U[:, list(in_modes)].to(torch.complex128).contiguous()
    A2 = torch.view_as_real(A)
    A2b = A2.view(1, m, n, 2)
    sig = torch.argsort(torch.rand(n, generator=generator, device=dev)).contiguous()
    seeds = torch.randint(0, 2 ** 31 - 1, (n,), generator=generator, device=dev).tolist()
    rows = torch.zeros(n, dtype=torch.int64, device=dev)
    drow = torch.zeros(max(n, 1), dtype=torch.int64, device=dev)
    mult = torch.zeros(max(n, 1), dtype=torch.int32, device=dev)
    radix = torch.ones(max(n, 1), dtype=torch.int32, device=dev)
    N = torch.ones(1, dtype=torch.int64, device=dev)
    scale = torch.ones(1, dtype=torch.float64, device=dev)
    Bs = torch.zeros(n * n, 2, dtype=torch.float64, device=dev)
    part = torch.empty(lanes, n, 2, dtype=torch.float64, device=dev)
    ones = torch.zeros(1, 1, 2, dtype=torch.float64, device=dev); ones[0, 0, 0] = 1.0
    lane2s = torch.zeros(lanes, dtype=torch.int64, device=dev)
    zero1 = torch.zeros(1, dtype=torch.int64, device=dev)
    one1 = torch.ones(1, dtype=torch.int64, device=dev)
    unit = torch.ones(1, dtype=torch.float64, device=dev)
    binom = _binom_table(dev)
    prep = get_prep_kernel()
    cp = cupy.asarray
    c_A2, c_rows, c_sig, c_drow, c_mult, c_radix, c_N, c_scale, c_Bs, c_lane2s, c_zero1, c_binom, c_part = \
        (cp(A2), cp(rows), cp(sig), cp(drow), cp(mult), cp(radix), cp(N), cp(scale), cp(Bs),
         cp(lane2s), cp(zero1), cp(binom), cp(part))
    rows_b, sig_b = rows.view(1, n), sig.view(1, n)

    steps = []
    ev = lambda: torch.cuda.Event(enable_timing=True)
    for k in range(1, n + 1):
        K1, K = k - 1, k
        if K1 == 0:
            draw_step(A2b, ones, rows_b, sig_b, zero1, 1, m, K, zero1, one1, unit, 0, seeds[0], max_lanes=1)
            continue
        e0, e1, e2, e3 = ev(), ev(), ev(), ev()
        e0.record()
        prep((1,), (block,), (c_rows, c_sig, c_A2, cupy.int32(K1), cupy.int32(K), cupy.int64(n),
                              c_drow, c_mult, c_radix, c_N, c_scale, c_Bs))
        e1.record()
        KP, tpl = tier(K)
        kern = _get_kernel(KP)
        nthreads = lanes * tpl
        kern(((nthreads + block - 1) // block,), (block,), (
            c_Bs, c_mult, c_radix, c_lane2s, c_zero1, c_N, c_binom, c_part,
            cupy.int32(K1), cupy.int32(K), cupy.int64(0), cupy.int64(lanes)))
        e2.record()
        part_k = part.view(-1)[: lanes * K * 2].view(lanes, K, 2)
        red = part_k.sum(0, keepdim=True)
        draw_step(A2b, red, rows_b, sig_b, zero1, 1, m, K, zero1, one1, scale, k - 1, seeds[k - 1], max_lanes=1)
        e3.record()
        torch.cuda.synchronize()
        steps.append({"k": k, "K": K, "tier": KP, "tpl": tpl, "N": int(N.item()),
                          "t_prep": e0.elapsed_time(e1), "t_cof": e1.elapsed_time(e2),
                          "t_draw": e2.elapsed_time(e3)})
    occ = torch.zeros(m, dtype=torch.int64, device=dev)
    occ.scatter_add_(0, rows, torch.ones_like(rows))
    return steps, occ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=34)
    p.add_argument("--mratio", type=float, default=2.0)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--lanes", type=int, default=1 << 15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default="")
    a = p.parse_args()

    dev = torch.device("cuda")
    n, m = a.n, round(a.mratio * a.n)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    X = torch.randn(m, m, 2, generator=gen, dtype=torch.float64, device=dev)
    U = torch.linalg.qr(torch.view_as_complex(X))[0]
    in_modes = list(range(n))

    rows_out = []
    for rep in range(a.samples):
        steps, _ = profile_sample(U, in_modes, gen, lanes=a.lanes)
        tot = sum(s["t_prep"] + s["t_cof"] + s["t_draw"] for s in steps)
        terms = sum(s["N"] for s in steps)
        print(f"\n=== sample {rep}: n={n} m={m}  total {tot:.1f} ms, {terms:.3e} Gray terms, "
              f"{terms / (tot * 1e-3) / 1e9:.2f} G terms/s overall")
        print(f"{'k':>4} {'tier':>5} {'pad%':>5} {'N_k':>11} {'prep':>8} {'cof':>9} {'draw':>8} "
              f"{'Gterm/s':>8} {'step%':>6} {'cum%':>6}")
        cum = 0.0
        for s in steps:
            st = s["t_prep"] + s["t_cof"] + s["t_draw"]
            cum += st
            pad = 100.0 * (s["tier"] - s["K"]) / s["tier"]
            gts = s["N"] / (s["t_cof"] * 1e-3) / 1e9 if s["t_cof"] > 0 else 0.0
            print(f"{s['k']:>4} {s['tier']:>5} {pad:>5.0f} {s['N']:>11.3e} {s['t_prep']:>8.3f} "
                  f"{s['t_cof']:>9.3f} {s['t_draw']:>8.3f} {gts:>8.2f} "
                  f"{100 * st / tot:>6.1f} {100 * cum / tot:>6.1f}")
            s.update(rep=rep, n=n, m=m, total_ms=tot)
            rows_out.append(s)
        tail = lambda j: 100 * sum(x["t_prep"] + x["t_cof"] + x["t_draw"] for x in steps[-j:]) / tot
        print(f"  last step {tail(1):.0f} %   last 2 {tail(2):.0f} %   last 3 {tail(3):.0f} %"
              f"   last 5 {tail(5):.0f} %")
        wasted = sum((s["tier"] - s["K"]) / s["tier"] * (s["t_cof"]) for s in steps)
        print(f"  work spent on padded columns: {100 * wasted / tot:.1f} % of the sample")

    if a.csv and rows_out:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
