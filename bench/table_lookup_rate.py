"""Sample rate from a stored output distribution (strong simulation below the memory frontier): exact samples by lookup.

Once the probability table of the last layer exists (docs/MEMORY_FRONTIER.md: float32 p, M_n entries, 4 M_n bytes),
a sample is an index into it. Two-level inverse CDF, as MEMORY_FRONTIER prescribes: float64 sums of blocks of
`block` entries (M_n / block of them, cumulated once), then per sample

    u ~ U(0, total)  ->  block b = searchsorted(block CDF, u)  ->  cumsum of block b in float64  ->  index in block

The table here is synthetic (i.i.d. exponential = Porter-Thomas weights): the lookup cost does not depend on the
values, only on the size and the block. A sample here is an INDEX into the table, not an occupation pattern
(unranking is not timed). Reports samples/s per table size and block, and the one-off cost of the block CDF. --check
compares the empirical frequencies of a small table with its probabilities (approximate chi-square z) and fails
beyond |z| = 5.

    python3 bench/table_lookup_rate.py --entries 1e8,1e9,5.4e9 --blocks 4096,256,64 --csv results/table_lookup.csv  # GPU
    python3 bench/table_lookup_rate.py --device cpu --entries 3000,1e6 --batch 4096 --check                        # CPU smoke
"""
import argparse
import csv
import math
import time

import torch


def block_cdf(p, block, dev):
    t0 = time.perf_counter()
    cdf = torch.cumsum(p.view(-1, block).sum(1, dtype=torch.float64), 0)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return cdf, time.perf_counter() - t0


def lookup(p, cdf, block, B, dev, gen):
    u = torch.rand(B, dtype=torch.float64, device=dev, generator=gen) * cdf[-1]
    b = torch.searchsorted(cdf, u, right=True).clamp_(max=cdf.numel() - 1)
    prev = torch.where(b > 0, cdf[(b - 1).clamp(min=0)], torch.zeros_like(u))
    rows = p.view(-1, block)[b].to(torch.float64).cumsum_(1)                                       # [B, block]
    j = torch.searchsorted(rows, (u - prev).unsqueeze(1), right=True).squeeze(1).clamp_(max=block - 1)
    return b * block + j


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--entries", default="1e8,1e9,5.4e9", help="table sizes (5.4e9 = M_13 at m = 26, the c = 2 frontier)")
    ap.add_argument("--blocks", default="4096,256,64", help="block sizes of the first CDF level (each a divisor of the first)")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--seconds", type=float, default=5.0, help="timed seconds per size and block")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", action="store_true", help="chi-square of the samples against p (tables <= 1e5 entries)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    blocks = [int(x) for x in a.blocks.split(",")]
    pad = max(blocks)
    assert all(pad % b == 0 for b in blocks)
    if dev.type == "cuda":
        print(torch.cuda.get_device_name(dev), flush=True)
    rows = []
    for N in (int(float(x)) for x in a.entries.split(",")):
        p = torch.empty(-(-N // pad) * pad, dtype=torch.float32, device=dev)
        p.exponential_(generator=gen)
        p[N:] = 0
        for block in blocks:
            cdf, t_cdf = block_cdf(p, block, dev)
            for _ in range(2):
                lookup(p, cdf, block, a.batch, dev, gen)                                           # warm-up
            sync(dev)
            t0, k, bad = time.perf_counter(), 0, torch.zeros((), dtype=torch.long, device=dev)
            while time.perf_counter() - t0 < a.seconds:
                bad += (lookup(p, cdf, block, a.batch, dev, gen) >= N).sum()                       # none in the padding
                k += 1
                sync(dev)
            dt = time.perf_counter() - t0
            assert int(bad) == 0, f"{int(bad)} samples in the zero padding"
            rate = k * a.batch / dt
            row = {"entries": N, "table_GB": round(4 * N / 1e9, 2), "block": block, "batch": a.batch,
                   "samples": k * a.batch, "seconds": round(dt, 3), "samples_per_s": rate, "block_cdf_s": round(t_cdf, 4)}
            if a.check and N <= 100_000:                                                           # >= 5 expected per bin
                cnt = torch.zeros(N, dtype=torch.float64, device=dev)
                for _ in range(-(-max(200 * N, a.batch) // a.batch)):
                    cnt += torch.bincount(lookup(p, cdf, block, a.batch, dev, gen), minlength=N)[:N].to(torch.float64)
                exp = p[:N].to(torch.float64) / float(cdf[-1]) * float(cnt.sum())
                m = exp > 5
                chi2 = float(((cnt[m] - exp[m]) ** 2 / exp[m]).sum())
                dof = int(m.sum()) - 1
                row["check_z"] = round((chi2 - dof) / math.sqrt(2 * dof), 2)
            rows.append(row)
            print(f"entries {N:.3g} ({row['table_GB']} GB float32), block {block}: {rate:.3g} samples/s (indices) "
                  f"({row['samples']} in {dt:.1f} s, batch {a.batch}); block CDF {t_cdf:.3f} s"
                  + (f"; check chi2 z = {row['check_z']}" if "check_z" in row else ""), flush=True)
            if "check_z" in row:
                assert abs(row["check_z"]) < 5, f"lookup check failed: chi-square z = {row['check_z']}"
            del cdf
        del p
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if a.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
