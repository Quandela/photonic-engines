"""Design plane of the emitter-state sequence generator (candidate A of the application study):
structure of the interferometer x number of emitters k, on one grid, three questions per point.

k dual-rail caterpillar emitters (random spin gates per emitter and bin: the trainable parameters
of a generator) into one static m = 2k-mode interferometer, T bins, sampled by pure trajectories
(`SequenceKernel.sample_diag`, O(k 3^k) per tick). Per (unitary, k, tick):

  register    middle-cut Schmidt entropy (bits), exact rank, 99.9 %-bond of the spin register —
              the classical simulability knob (bench/sequence_ticks.py, docs/SEQUENCE.md §9)
  flatness    Z2 = S sum_t p(t)^2 of the bin's pattern distribution (1 uniform, 2 Porter-Thomas):
              exact from `bin_marginals` for k <= --strong-kmax (the S_bin x chi^2 Kraus set: k = 6 is 0.8 GB,
              k = 8 is 480 GB - keep it <= 6), and always the sample estimate on
              the 4-mode marginal of emitters 0 and 1 (support small at any k)
  signal      training signals a classically-estimable local loss would use. For label-diagonal
              (caterpillar) emitters the unconditional single-bin marginal is the uniform mixture
              of the 2^k Fock instances whatever the spin gates (the register is dephased in the
              label basis by the previous bin), so the gates are visible only in inter-bin
              statistics: `sig_*` are the connected correlators C_ij = <n_i^(b-1) n_j^(b)> -
              <n_i^(b-1)><n_j^(b)> between the clicks of consecutive bins (all mode pairs).
              `sig_std` = rms over pairs of the std of C_ij across the random gate settings (how
              much the parameters move the observable), `sig_scale` its rms magnitude, `sig_err`
              its shot noise at --shots. sig_std / sig_err < 1 at a given shot budget means the
              loss landscape is below the noise floor there (the Herbst et al. concentration
              regime). Within-bin correlators are reported as `sig_within` (setting-independent
              up to noise: a check).
  cost        wall time per tick-shot, peak memory.
  --policy    feed-forward from the previous bin's pattern to the next bin's spin gates (local |
              global) or interferometer (route-local | route-global), `make_policy`; candidate B
              of the study: the adaptive version of the same generator, sampled at the same cost
              by the trajectory picture.

    python3 bench/esbs_design_plane.py --device cuda --k 4,6,8,10,12 --unitary haar,band:1,band:2,band:4 \
        --ticks 6 --shots 256 --settings 4
Writes $RESULTS_DIR/esbs_design_plane_<unitary>.csv (one file per unitary family, one row per
(k, setting, tick)); plot with bench/plot_esbs_design_plane.py.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from photonic_engines import spin_gate
from photonic_engines.sequence import BinEmitter, SequenceKernel

C128 = torch.complex128


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return torch.as_tensor(q * (np.diag(r) / np.abs(np.diag(r))), dtype=C128)


def banded(m, layers, seed):
    rng = np.random.default_rng(seed)
    U = np.eye(m, dtype=complex)
    for layer in range(layers):
        for i in range(layer % 2, m - 1, 2):
            z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            q, _ = np.linalg.qr(z)
            V = np.eye(m, dtype=complex)
            V[i:i + 2, i:i + 2] = q
            U = V @ U
    return torch.as_tensor(U, dtype=C128)


def make_unitary(spec, m, seed):
    if spec == "haar":
        return haar(m, seed)
    if spec.startswith("band:"):
        return banded(m, 2 * int(spec.split(":")[1]), seed)
    raise ValueError(spec)


def schmidt(psi, k):
    a = 2 ** (k // 2)
    M = psi.reshape(psi.shape[0], a, -1)
    # smaller Gram matrix: eigvalsh is robust where cusolver's batched SVD fell back to a slow path
    G = M @ M.conj().transpose(1, 2) if a <= M.shape[2] else M.conj().transpose(1, 2) @ M
    s = torch.linalg.eigvalsh(G).clamp_min(0)
    return torch.sort(s, 1, descending=True)[0]


def random_angles(T, rng):
    """(theta, phi) of the spin preparation (pi/2, 0) and of the T-1 random gates of one emitter."""
    return [(math.pi / 2, 0.0)] + [(rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)) for _ in range(T - 1)]


def spin_gates_torch(theta, phi):
    """Batched U(theta, phi) = Ry(theta/2) Rz(phi) Ry(theta/2), theta / phi [...] -> [..., 2, 2]."""
    c, s_ = torch.cos(theta / 4), torch.sin(theta / 4)
    ry = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2).to(C128)
    rz = torch.zeros((*theta.shape, 2, 2), dtype=C128)
    rz[..., 0, 0] = torch.exp(-1j * phi / 2); rz[..., 1, 1] = torch.exp(1j * phi / 2)
    return ry @ rz @ ry


def make_policy(kind, angles, k, m, U=None, U_alt=None):
    """Feed-forward from the previous bin's pattern to the next bin's setting (candidate B).
      none          no feed-forward
      local         spin gates: emitter j gets theta_j += pi/2 * (clicks in its own two modes at bin b)
      global        spin gates: every emitter gets theta_j += pi/2 * (parity of the clicks on the first m/2 modes)
      route-local   interferometer: U_{b+1} = U . diag(phases), phase pi/2 * c_j on both modes of emitter j
                    (c_j = clicks in its modes at bin b) — a click-driven input phase layer
      route-global  interferometer: U_{b+1} = U or U_alt according to the parity bit above
    angles[j][b] = (theta, phi) of emitter j at bin b; U, U_alt the bin unitaries [m, m]."""
    if kind == "none":
        return None
    if kind.startswith("route"):
        def route(b, counts, psi):
            c = counts.to(U.device).double()
            if kind == "route-local":
                cj = c.reshape(c.shape[0], k, 2).sum(-1)                                         # [S, k]
                ph = torch.exp(1j * math.pi / 2 * cj.repeat_interleave(2, dim=1)).to(U.dtype)   # [S, m]
                return U.unsqueeze(0) * ph.unsqueeze(1)                                          # U diag(ph)
            bit = ((c[:, : m // 2].sum(-1) % 2) > 0.5)
            return torch.where(bit[:, None, None], U_alt.unsqueeze(0), U.unsqueeze(0))
        return route
    th0 = torch.tensor([[angles[j][b][0] for j in range(k)] for b in range(len(angles[0]))], dtype=torch.float64)   # [T, k]
    ph0 = torch.tensor([[angles[j][b][1] for j in range(k)] for b in range(len(angles[0]))], dtype=torch.float64)

    def policy(b, counts, psi):
        c = counts.cpu().double()                                                   # [S, m]
        if kind == "local":
            shift = c.reshape(c.shape[0], k, 2).sum(-1)                             # [S, k] clicks per emitter pair
        elif kind == "global":
            shift = (c[:, : m // 2].sum(-1) % 2).unsqueeze(-1).expand(-1, k)       # [S, k] one parity bit
        else:
            raise ValueError(kind)
        theta = th0[b + 1].unsqueeze(0) + math.pi / 2 * shift
        phi = ph0[b + 1].unsqueeze(0).expand_as(theta)
        return spin_gates_torch(theta, phi)                                         # [S, k, 2, 2]
    return policy


def z2_from_counts(counts):
    """|support| * sum p^2 from samples: collision estimate over pairs, support = distinct patterns seen."""
    keys = [tuple(r) for r in counts.tolist()]
    N = len(keys)
    _, inv = np.unique(np.array(keys), axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    coll = (cnt * (cnt - 1)).sum() / (N * (N - 1))
    return coll, len(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", default="4,6,8")
    ap.add_argument("--unitary", default="haar,band:1,band:2")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--settings", type=int, default=4, help="random spin-gate settings per (unitary, k)")
    ap.add_argument("--strong-kmax", type=int, default=6, help="exact bin marginals (chi^3 S_bin) up to this k")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-gb", type=float, default=60.0)
    ap.add_argument("--policy", default="none", help="feed-forward: none | local | global | route-local | route-global (see make_policy)")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    dev = args.device
    prefix = args.out_prefix or os.path.join(os.environ.get("RESULTS_DIR", "results"), "esbs_design_plane")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    for spec in args.unitary.split(","):
        rows = []
        for k in [int(x) for x in args.k.split(",")]:
            m = 2 * k
            peak_gb = 2 * max(math.comb(k, j + 1) * 2 ** (j + 1) for j in range(k)) * 16 / 2 ** 30
            if peak_gb > args.max_gb:
                print(f"{spec} k={k}: label-amplitude pass needs {peak_gb:.1f} GB per shot, skipped", flush=True)
                continue
            U = make_unitary(spec, m, args.seed + k).to(dev)
            U_alt = make_unitary(spec, m, args.seed + 1000 + k).to(dev)
            rng = np.random.default_rng(args.seed * 1000 + k)
            pairs = [(i, j) for i in range(m) for j in range(m) if i // 2 < j // 2]     # modes of different emitters
            corr, corr_within = {}, {}                                                 # (setting, tick) -> C_ij vector
            for s in range(args.settings):
                angles = [random_angles(args.ticks, rng) for j in range(k)]
                ems = [BinEmitter.caterpillar(args.ticks, [spin_gate(*a) for a in angles[j]], (2 * j, 2 * j + 1)) for j in range(k)]
                sk = SequenceKernel(ems, m, device=dev)
                policy = make_policy(args.policy, angles, k, m, U, U_alt)
                rec, t_last, prev = {}, [time.time()], [None]

                def hook(b, psi, counts):
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    dt = time.time() - t_last[0]
                    sv = schmidt(psi, k)
                    ent = -(sv * torch.log2(sv.clamp_min(1e-300))).sum(1)
                    rank = (sv > 1e-12).sum(1).float()
                    bond = ((torch.cumsum(sv, 1) < 0.999).sum(1) + 1).float()
                    c = counts.cpu().double()                                          # [shots, m]
                    mean = c.mean(0)
                    cov = (c.T @ c) / c.shape[0] - torch.outer(mean, mean)
                    corr_within[(s, b)] = torch.stack([cov[i, j] for i, j in pairs])
                    if prev[0] is not None:                                            # bins b-1 x b, all mode pairs
                        pm = prev[0].mean(0)
                        corr[(s, b)] = ((prev[0].T @ c) / c.shape[0] - torch.outer(pm, mean)).reshape(-1)
                    prev[0] = c
                    # 4-mode marginal (emitters 0, 1) flatness from samples
                    coll4, supp4 = z2_from_counts(c[:, :4].long())
                    rec[b] = {"tick_s": round(dt, 4), "ms_per_tick_shot": round(1e3 * dt / psi.shape[0], 4),
                                  "entropy": round(float(ent.mean()), 4), "rank": round(float(rank.mean()), 2),
                                  "bond999": round(float(bond.mean()), 2),
                                  "Z2_4mode": round(coll4 * supp4, 4), "support_4mode": supp4,
                                  "mean_click": round(float(mean.mean()), 4)}
                    t_last[0] = time.time()

                if dev == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sk.sample_diag(U, min(args.shots, 8), generator=torch.Generator().manual_seed(0), policy=policy)   # warm-up
                t_last[0] = time.time()
                sk.sample_diag(U, args.shots, generator=torch.Generator().manual_seed(1 + s), hook=hook, policy=policy)
                peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if dev == "cuda" else None
                z2_exact = None
                if k <= args.strong_kmax:
                    marg = sk.bin_marginals(U)                                          # [T, S]
                    z2_exact = (marg.shape[1] * (marg ** 2).sum(1)).tolist()
                for b in range(args.ticks):
                    rows.append(dict(unitary=spec, policy=args.policy, k=k, m=m, setting=s, tick=b + 1, shots=args.shots,
                                     max_entropy=k // 2, max_rank=2 ** (k // 2), **rec[b],
                                     Z2_bin_exact=None if z2_exact is None else round(z2_exact[b], 4),
                                     S_bin=math.comb(m + k - 1, k), peak_GB=peak))
                print(rows[-1], flush=True)
            # signal statistics across settings, per tick
            for b in range(args.ticks):
                W = torch.stack([corr_within[(s, b)] for s in range(args.settings)])
                sig_within = float(W.std(0, unbiased=args.settings > 1).pow(2).mean().sqrt())
                # shot noise of a covariance entry, order of magnitude
                sig_err = float(rec[b]["mean_click"] / math.sqrt(args.shots))
                upd = {"sig_within": round(sig_within, 5), "sig_err": round(sig_err, 5)}
                if b > 0:
                    C = torch.stack([corr[(s, b)] for s in range(args.settings)])              # [settings, pairs]
                    upd.update(sig_std=round(float(C.std(0, unbiased=args.settings > 1).pow(2).mean().sqrt()), 5),
                               sig_scale=round(float(C.pow(2).mean().sqrt()), 5))
                else:
                    upd.update(sig_std=None, sig_scale=None)
                for r in rows:
                    if r["k"] == k and r["tick"] == b + 1:
                        r.update(upd)
        if rows:
            out = f"{prefix}_{spec.replace(':', '')}{'' if args.policy == 'none' else '_ff' + args.policy.replace('-', '')}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
