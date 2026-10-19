"""
Cost of the exact chain-rule sampler for superposition boson sampling
(`clifford.product_sampler`, docs/SUPERPOSITION.md §5).

Unlike `slos.ProductSLOSKernel.sample` this builds no basis, so it is not bounded by strong
simulation. It is bounded by the per-step marginal, whose two evaluations are cheap at opposite
ends of the chain — C(n, k-j) 2^j n j by enumerating the residual sets, 4^j n j^3 by the doubled
Glynn sum — so the reach is set by whichever is cheaper in the MIDDLE, and that is the 4^j.

    python3 bench/bench_product_sampler.py --device cuda --k 4,6,8,10 --shots 3
    python3 bench/bench_product_sampler.py --device cuda --k 10 --profile      # per-step breakdown

Reports ms per sample at alpha = pi/4 (n = 2k, m = 2n), the method chosen at each step, and with
`--profile` the time of every step so the bottleneck is visible rather than inferred.
"""
import argparse
import csv
import math
import time

import torch

from photonic_engines import ProductSuperposition
from photonic_engines.clifford.product_sampler import REPLICA_UNIT_COST, cheaper, marginals


def _haar(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)


def model(n: int, k: int, j: int) -> tuple[float, float]:
    """(residual, replica) cost-model units at step j, the same two `cheaper` compares."""
    return float(math.comb(n, k - j) * 2 ** j * (j + 1)), float(REPLICA_UNIT_COST * 4 ** j * j ** 3 * n)


def run(k: int, ratio: float, shots: int, device, chunk: int, profile: bool) -> dict:
    n = 2 * k                                        # alpha = pi/4: the mean detected count is n/2
    m = max(2, math.ceil(ratio * n))
    st, U = ProductSuperposition.sbs(n, m), _haar(m, seed=n * 31 + k)
    gen = torch.Generator().manual_seed(0)
    marginals(st, U, [], list(range(m)), k, chunk=chunk, device=device)   # CUDA init, not the sampler
    if device.type == "cuda":                                            # (it cost k = 4 of run
        torch.cuda.synchronize()                                         # 20260919_224559 147 ms
    per_step = [0.0] * (k + 1)                                           # against the 8 ms of k = 6)
    t0 = time.perf_counter()
    for _ in range(shots):
        rows: list[int] = []
        for j in range(1, k + 1):
            s = time.perf_counter()
            w = marginals(st, U, rows, list(range(m)), k, chunk=chunk, device=device).clamp_min(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_step[j] += (time.perf_counter() - s) * 1e3
            rows.append(int(torch.multinomial(w.cpu(), 1, generator=gen)))
    ms = (time.perf_counter() - t0) * 1e3 / shots
    print(f"  k={k:3d} n={n:3d} m={m:4d}  {ms:10.1f} ms/sample   "
          f"steps: {' '.join(cheaper(n, k, j)[:3] for j in range(1, k + 1))}", flush=True)
    if profile:
        for j in range(1, k + 1):
            res, rep = model(n, k, j)
            print(f"      j={j:2d} {cheaper(n, k, j):8s}  {per_step[j] / shots:9.1f} ms"
                  f"   model: residual {res:.2e}  replica {rep:.2e}", flush=True)
    return {"photons": k, "sources": n, "m": m, "ms_per_sample": round(ms, 3), "shots": shots}


def write(path: str, rows: list[dict]) -> None:
    if not (path and rows):
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", default="4,6,8", help="comma-separated detected photon counts (n = 2k)")
    ap.add_argument("--ratio", type=float, default=2.0, help="m / sources")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=64, help="sign vectors per batch in the Glynn sums")
    ap.add_argument("--profile", action="store_true", help="time every step of the chain")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device={device} ratio={args.ratio} shots={args.shots} chunk={args.chunk}")
    rows = []
    for k in (int(x) for x in args.k.split(",")):
        try:                                   # the last k of a sweep costs 4x the one before it:
            rows.append(run(k, args.ratio, args.shots, device, args.chunk, args.profile))
        except Exception as e:                 # never let it take the cheaper points down with it
            print(f"  k={k:3d}  FAILED: {type(e).__name__}: {e}", flush=True)
            break
        write(args.csv, rows)                  # rewritten per k, so a kill or a timeout keeps the rest
    if args.csv and rows:
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
