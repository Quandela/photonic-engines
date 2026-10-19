"""Where the double-double sieve term spends its time (docs/GAUSSIAN.md §3.4 v2.11b; job 28).

    python3 bench/gbs_dd_profile.py --n 16,32,48 --V 1,4 --chunk 4096 --csv gbs_dd_profile.csv

One `hafnian._terms_dd` batch of `--chunk` subset terms at bucket size p (the largest bucket p = n and p = n / 2), under
torch.profiler, with the Ozaki slicing in torch (v2.11b) and in the fused kernel (v2.11c, `dd._FUSED_SLICES`): device
time per stage (the record_function labels dd.powers / dd.traces / dd.paths / dd.dots /
dd.series of `_terms_dd`; the rest - gathers, g_k assembly, accumulation - is "other"), kernel launches per batch, and
the top operators by device time. The same batch in fp64 (`hafnian._terms`) gives the dd / fp64 factor at that p.
The question it answers: which stage a fused kernel should take first (job 27: dd x116-253 of the fp64 Triton term).
"""
import argparse
import csv
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")
import photonic_engines.gaussian.dd as Ddd
import photonic_engines.gaussian.hafnian as haf

STAGES = ("dd.powers", "dd.traces", "dd.paths", "dd.dots", "dd.series")


def instance(N, V, rng):
    X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    A = (X + X.T) / (2 * np.sqrt(N))
    D = rng.normal(size=(V, N)) + 1j * rng.normal(size=(V, N))
    return A, D * 0.3, D * 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="16,32,48")
    ap.add_argument("--V", default="1,4")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    rows = []
    act = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == "cuda" else [])

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
    for n in [int(x) for x in args.n.split(",")]:
        for V in [int(x) for x in args.V.split(",")]:
            N = 2 * n
            A, D0, D1 = instance(N, V, rng)
            t = lambda x: torch.as_tensor(x, dtype=torch.complex128, device=dev)
            Ce, Ye, XYe = haf._setup(t(A), t(D0), t(D1), np.arange(N), dev)
            for p in sorted({n, max(1, n // 2)}):
                # the dd batch the sieve itself would run (its memory budget / _DD_MEM), at most --chunk: a fixed 4096 ran
                # the H100 out of memory at n = 32, p = 32 (job 28)
                budget = ((1 << 32) if dev.type == "cuda" else (1 << 28)) // haf._DD_MEM
                chunk = max(1, min(args.chunk, haf._chunk_size_multi(n, p, V, budget, dev)))
                pos = np.stack([np.sort(rng.choice(n, size=p, replace=False)) for _ in range(chunk)])
                idx = torch.as_tensor(np.concatenate([pos, pos + n], 1), device=dev)
                w = torch.ones(chunk, dtype=torch.float64, device=dev)
                acc = torch.zeros((V, 2 * n + 1), dtype=torch.complex128, device=dev)
                acc_abs = torch.zeros_like(acc.real)
                accd = [acc.clone(), torch.zeros_like(acc)]

                def fp64():
                    haf._terms(Ce, Ye, XYe, idx, w, n, V, acc, acc_abs, True)

                def dd():
                    haf._terms_dd(Ce, Ye, XYe, idx, w, n, V, accd, acc_abs, True)
                fp64(); sync()                                        # warm-up (allocator)
                sync(); t0 = time.perf_counter(); fp64(); sync(); t_fp = time.perf_counter() - t0
                for fused in (False, None):                           # torch slicing (v2.11b) / the fused kernel (v2.11c)
                    Ddd._FUSED_SLICES = fused
                    dd(); sync()                                      # warm-up (Triton compile)
                    sync(); t0 = time.perf_counter(); dd(); sync(); t_dd = time.perf_counter() - t0
                    with torch.profiler.profile(activities=act) as prof:
                        dd(); sync()
                    ev = prof.key_averages()
                    cuda = dev.type == "cuda"
                    dt = (lambda e: getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0.0))) if cuda \
                        else (lambda e: e.cpu_time_total)
                    sdt = (lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))) \
                        if cuda else (lambda e: e.self_cpu_time_total)
                    ops = [e for e in ev if e.key not in STAGES]
                    total = sum(sdt(e) for e in ops) or 1.0
                    stage = {st: max([dt(e) for e in ev if e.key == st] or [0.0]) for st in STAGES}
                    launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cuLaunchKernel",
                                                                    "cudaLaunchKernelExC"))
                    top = sorted(ops, key=sdt, reverse=True)[:8]
                    tag = "fused" if fused is None else "torch"
                    r = {"n": n, "V": V, "p": p, "chunk": chunk, "slicing": tag, "fp64_s": t_fp, "dd_s": t_dd,
                         "dd_over_fp64": t_dd / t_fp, "launches": launches, "device_us": total,
                         **{st + "_share": stage[st] / total for st in STAGES},
                         "other_share": max(0.0, 1 - sum(stage.values()) / total)}
                    rows.append(r)
                    print(f"n={n:2d} V={V} p={p:2d} chunk {chunk} {tag}: fp64 {t_fp * 1e3:8.2f} ms, dd {t_dd * 1e3:9.2f} ms "
                          f"(x{r['dd_over_fp64']:.0f}), {launches} launches; "
                          + ", ".join(f"{st[3:]} {100 * r[st + '_share']:.0f} %" for st in STAGES)
                          + f", other {100 * r['other_share']:.0f} %", flush=True)
                    print("   top: " + "; ".join(f"{e.key[:40]} {100 * sdt(e) / total:.0f} % x{e.count}" for e in top),
                          flush=True)
                Ddd._FUSED_SLICES = None
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
