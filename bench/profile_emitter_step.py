"""Where does a sampler step go? Phase timings (device-synchronised) and the top CUDA kernels of
one `batched_gap_terms` + `emitter_step_weights` call at a given (n, k, shots).

    python3 bench/profile_emitter_step.py --n 16,24 --k mid,last --shots 256
"""
import argparse
import math
import time

import torch

from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford.emitter_terms import batched_gap_terms
from photonic_engines.clifford.emitter_triton import emitter_step_weights


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, reps=3):
    fn(); sync()
    t = time.time()
    for _ in range(reps):
        out = fn()
    sync()
    return (time.time() - t) / reps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", default="16,24")
    ap.add_argument("--k", default="mid,last")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no-profiler", action="store_true")
    args = ap.parse_args()
    dev = args.device
    for n in [int(x) for x in args.n.split(",")]:
        m = 2 * n
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=m)
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].to(dev)
        S = args.shots
        g = torch.Generator().manual_seed(1)
        sigma = torch.argsort(torch.rand((S, n), generator=g), dim=1)
        for ks in args.k.split(","):
            k = {"mid": n // 2, "last": n}.get(ks) or int(ks)
            Tsorted = torch.sort(sigma[:, :k], dim=1).values.to(dev)
            rows = torch.randint(0, m, (S, k - 1), generator=g).to(dev)
            fid = torch.randint(0, 2, (S,), generator=g)
            finals = torch.eye(2, dtype=torch.complex128, device=dev)[fid.to(dev)]
            t_terms, (s0, mats, f, t2s) = timed(lambda: batched_gap_terms(st, Tsorted, finals))
            Lam = s0.shape[0]
            timing = {"kernel": 0.0}
            emitter_step_weights(U, st, Tsorted, rows, s0, mats, f, t2s, S, timing=timing); sync()   # warm-up (compile)
            timing["kernel"] = 0.0
            t_step, (w, lanes) = timed(lambda: emitter_step_weights(U, st, Tsorted, rows, s0, mats, f, t2s, S, timing=timing))
            t_kernel = timing["kernel"] / 4                       # timed() runs 1 + 3 calls
            print(f"n={n} k={k} S={S} terms={Lam} ({Lam / S:.1f}/stream) lanes/term={lanes:.1f}: "
                  f"terms {t_terms * 1e3:.1f} ms | step {t_step * 1e3:.1f} ms = kernel {t_kernel * 1e3:.1f} + host/post "
                  f"{(t_step - t_kernel) * 1e3:.1f}",
                  flush=True)
            if args.no_profiler or dev != "cuda":
                continue
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                s0, mats, f, t2s = batched_gap_terms(st, Tsorted, finals)
                emitter_step_weights(U, st, Tsorted, rows, s0, mats, f, t2s, S)
                sync()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.top, max_name_column_width=60), flush=True)


if __name__ == "__main__":
    main()
