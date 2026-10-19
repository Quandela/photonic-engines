"""CUDA thread-per-lane emitter kernel (clifford/cuda_emitter.py) on a GPU: per-step weights
against the Triton seq kernel at every tier up to K = 30 (repeated rows, one and two labels),
identical end-to-end draws, and the per-step time against the seq kernel at the frontier steps.

    python3 tests/clifford/check_cuda_emitter.py [--K 8,12,...,30] [--n 24] [--shots 8]
"""
import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from photonic_engines import EmitterMPS, spin_gate
from photonic_engines.clifford import cuda_emitter as ce
from photonic_engines.clifford import emitter_sample
from photonic_engines.clifford.emitter_terms import batched_gap_terms
from photonic_engines.clifford.emitter_triton import emitter_step_weights

ap = argparse.ArgumentParser()
ap.add_argument("--K", default="4,8,9,12,16,20,24,26,28,30")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--shots", default="8", help="comma list: section 3 is repeated per shot count")
ap.add_argument("--timing-only", action="store_true", help="skip sections 1-2")
ap.add_argument("--all-steps", action="store_true", help="section 3: print every step, not only the last nine")
ap.add_argument("--kernels", default="seq,cuda", help="section 3: kernels to time (seq, cuda, auto)")
args = ap.parse_args()
dev = "cuda"
assert ce.available(), "needs cupy and a CUDA device"
ce.warmup()

# --- 1. weights vs the Triton seq kernel, per tier
print("step weights, cuda vs triton seq (max |dw| / max w):")
for K in ([] if args.timing_only else [int(x) for x in args.K.split(",")]):
    for nl in (2, 1):
        n, m = K, 2 * K
        st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.3 * j) for j in range(n)], m=m, final=[0.6, 0.8j])
        if nl == 1:
            st = EmitterMPS(st.ops[:, :1], tuple((mk[0],) for mk in st.modes), st.s0, m, st.final)
        U = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128, generator=torch.Generator().manual_seed(K)))[0].to(dev)
        S = 6
        g = torch.Generator().manual_seed(nl)
        kept = torch.sort(torch.argsort(torch.rand(S, n, generator=g), dim=1)[:, :K], dim=1).values.to(dev)
        rows = torch.randint(0, max(2, m // 4), (S, K - 1), generator=g).to(dev)          # repeats -> deep collapse
        final = torch.as_tensor(st.final, dtype=torch.complex128, device=dev).unsqueeze(0).expand(S, 2)
        s0, mats, f, t2s = batched_gap_terms(st, kept, final)
        w_seq, _ = emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, impl="seq")
        w_cu, _ = emitter_step_weights(U, st, kept, rows, s0, mats, f, t2s, S, impl="cuda")
        err = float((w_cu - w_seq).abs().max() / w_seq.abs().max())
        KP, tpl, gs = ce.tier(K)
        # The two kernels sum the same collapsed terms in a different order (Fock-style vs seq tables);
        # with rows drawn from m/4 values the binomial weights grow with K and the cancellation with
        # them: run 20260927_110150 measured 4e-16 (K = 4) .. 9e-13 (26) .. 9e-12 (28), a steady
        # ~4x per tier, on both kernels' side (no FP64 reference at this size).  Tolerance 1e-10.
        print(f"  K={K:2d} NL={nl} tier {KP} TPL={tpl} GS={gs}: {err:.2e}  {'ok' if err < 1e-10 else 'MISMATCH'}", flush=True)
        assert err < 1e-10

# --- 2. identical draws end to end
n = 3 if args.timing_only else 12
st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0.2 * j) for j in range(n)], m=2 * n)
U = torch.linalg.qr(torch.randn(2 * n, 2 * n, dtype=torch.complex128, generator=torch.Generator().manual_seed(3)))[0].to(dev)
a = emitter_sample(st, U, 64, generator=torch.Generator().manual_seed(5), backend="triton", kernel="seq")
b = emitter_sample(st, U, 64, generator=torch.Generator().manual_seed(5), backend="triton", kernel="cuda")
print("end-to-end draws identical (n=12, 64 shots):", bool(torch.equal(a, b)))
assert torch.equal(a, b)

# --- 3. per-step time at the frontier steps, and the step split (terms / kernel / post) per shot count:
# the map runs 64-256 shots per batch, where the kernel is a smaller share of the step than at 8
n = args.n
st = EmitterMPS.caterpillar(n, [spin_gate(math.pi / 2, 0)] * n, m=2 * n, final=[1, 0])
U = torch.linalg.qr(torch.randn(2 * n, 2 * n, dtype=torch.complex128, generator=torch.Generator().manual_seed(n)))[0].to(dev)
for S in [int(x) for x in args.shots.split(",")]:
    print(f"sampler n={n} m={2 * n} {S} shots: seconds per step (k, seq, cuda, ratio)")
    out = {}
    kernels = args.kernels.split(",")
    for kern in kernels:
        emitter_sample(st, U, min(S, 8), generator=torch.Generator().manual_seed(1), backend="triton", kernel=kern)     # warm
        torch.cuda.synchronize(); t0 = time.time()
        _, stats = emitter_sample(st, U, S, generator=torch.Generator().manual_seed(1), backend="triton", kernel=kern, return_stats=True)
        torch.cuda.synchronize()
        out[kern] = (time.time() - t0, stats)                                   # lossless: sample_triton's own stats
    ref = kernels[0]
    print("        " + "".join(f"{kk:>10s} " for kk in kernels) + "  (ratios to " + ref + ")")
    for k in range(1 if args.all_steps else max(1, n - 8), n + 1):
        a = out[ref][1]["step_s"][k - 1]
        R, lanes = out[ref][1]["R"][k - 1], out[ref][1]["lanes"][k - 1]
        print(f"  k={k:2d}: " + "".join(f"{out[kk][1]['step_s'][k - 1]:8.4f} s " for kk in kernels)
              + "  " + " ".join(f"x{a / out[kk][1]['step_s'][k - 1]:4.2f}" for kk in kernels[1:])
              + f"   (terms/stream {R:7.1f}, lanes/term {lanes:6.1f})")
    for kern in kernels:
        t, s_ = out[kern]
        print(f"  {kern:4s}: total {t:7.2f} s = terms {s_['terms_s']:6.2f} + kernel {s_['kernel_s']:6.2f} + post {s_['post_s']:6.2f}"
              f"  ({1e3 * t / S:.1f} ms/sample)")
    for kk in kernels[1:]:
        print(f"  {kk} / {ref} end to end: x{out[ref][0] / out[kk][0]:.2f}; kernel alone "
              f"x{out[ref][1]['kernel_s'] / out[kk][1]['kernel_s']:.2f}")
