"""Exact samples of a Borealis run's own model on one GPU, and their check against the timing model and the device
(docs/GAUSSIAN.md §2.4).

    python3 bench/gbs_borealis_exact.py run --data experiments/qca-data/fig3a --shots 10000 --budget 3000 \
        --device cuda --out $RESULTS_DIR/borealis_fig3a_exact.json
    python3 bench/gbs_borealis_exact.py compare --run borealis_fig3a_exact.json --device-samples \
        experiments/qca-data/fig3a/samples.npy --calib gbs_timing_calib.json --extra rate.json,sieves.json

run: the state is Xanadu's published squeezing r [216] (one squeezer per time bin) through the published lossy
transfer matrix T [216, 216] (the three loops, the losses and the detector efficiencies in one matrix;
github.com/XanaduAI/xanadu-qca-data, Apache-2.0), sampled by `gaussian.sample_transfer` (exact PNR chain rule,
chain order = time-bin order). Every shot's pattern and wall time is written; a checkpoint every --every shots, so a
run stopped by --budget or a timeout keeps what it did. --max-photons censors a shot above the cap (row -1, a timing
benchmark, not a sample of the law); None = no cap. Three warm-up shots (other seed, discarded) absorb the kernel
compilation. The decomposition of T is computed once (the first run, 20260927_190700, recomputed it per shot:
2614 s of wall time for 1536 s of shots).

compare: (a) the timing model's price of each measured pattern (`timing.TimingModel.shot_time`) against its measured
time: summed ratio and per-shot median, as in `gbs_timing_model.py model --validate`; (b) the photon-number law of
our samples against the device's samples and against the ideal mean from the covariance (the device's samples carry
its noise, ours the published model's), with the per-mode mean photon numbers.
"""
import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def load_model_state(data):
    from photonic_engines.states import GaussianInput
    T = np.load(f"{data}/T.npy"); r = np.load(f"{data}/r.npy").astype(np.float64).reshape(-1)
    m = T.shape[0]
    if r.size != m:
        raise SystemExit(f"{data}: r has {r.size} entries for T [{m}, {m}]")
    return GaussianInput(r, tuple(range(m)), m), T


def run(args):
    import torch

    from photonic_engines.gaussian import sample_transfer, transfer_decomposition
    st, T = load_model_state(args.data)
    dec = transfer_decomposition(st, T)                                       # once, not per shot
    dev = torch.device(args.device)
    print(f"{args.data}: m = {st.m}, r mean {st.r.mean():.3f}, |T|_2 {np.linalg.norm(T, 2):.3f}, device {dev}", flush=True)
    sample_transfer(st, T, 3, seed=args.seed + 10 ** 6, device=dev, max_photons=args.max_photons, decomposition=dec)
    rec = {"data": args.data, "m": st.m, "seed": args.seed, "max_photons": args.max_photons, "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "shots": []}
    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()

    def dump():
        with open(args.out, "w") as f:
            json.dump(rec, f)
    for s in range(args.shots):
        occ, stats = sample_transfer(st, T, 1, seed=int(rng.integers(2 ** 62)), device=dev, return_stats=True,
                                     max_photons=args.max_photons, decomposition=dec)
        rec["shots"].append({"pattern": occ[0].tolist(), "seconds": stats.seconds[0], "photons": stats.photons[0],
                             "censored": bool(stats.censored[0]), "max_lhaf": stats.max_lhaf[0]})
        if (s + 1) % args.every == 0:
            dump()
            el = time.perf_counter() - t_start
            ph = [x["photons"] for x in rec["shots"]]
            print(f"  {s + 1} shots, {el:.0f} s, {el / (s + 1):.3f} s/shot, N mean {np.mean(ph):.1f} max {max(ph)}",
                  flush=True)
        if time.perf_counter() - t_start > args.budget:
            print(f"  budget reached after {s + 1} shots", flush=True)
            break
    dump()
    sec = [x["seconds"] for x in rec["shots"]]
    print(f"done: {len(sec)} shots, {sum(sec):.1f} s, {np.mean(sec):.4f} s/shot mean, median {np.median(sec):.4f}; "
          f"censored {sum(x['censored'] for x in rec['shots'])}; wrote {args.out}", flush=True)


def compare(args):
    import gbs_timing_model as gtm
    shots = []
    for p in args.run.split(","):
        shots += json.load(open(p))["shots"]
    ok = [x for x in shots if not x["censored"]]
    print(f"{len(shots)} shots ({len(shots) - len(ok)} censored)")
    if args.calib:
        class A:
            pass
        a = A(); a.calib = args.calib; a.extra = args.extra; a.discount = "self"
        _, M = gtm.load_model(a)
        pred = np.array([M.shot_time(np.array(x["pattern"])) for x in ok]); meas = np.array([x["seconds"] for x in ok])
        print(f"timing model / measured: summed {pred.sum() / meas.sum():.3f}, per-shot median "
              f"{np.median(pred / meas):.3f}; measured {meas.mean():.4f} s/shot, model {pred.mean():.4f}")
        for lo, hi in ((0, 20), (20, 30), (30, 40), (40, 1000)):
            sel = [i for i, x in enumerate(ok) if lo <= x["photons"] < hi]
            if sel:
                print(f"  N in [{lo}, {hi}): {len(sel)} shots, summed ratio {pred[sel].sum() / meas[sel].sum():.3f}")
    P = np.array([x["pattern"] for x in ok])
    N = P.sum(1)
    out = {"ours": {"shots": len(P), "N_mean": float(N.mean()), "N_var": float(N.var())}}
    print(f"ours:   N mean {N.mean():.2f} (+- {N.std() / math.sqrt(len(N)):.2f}), var {N.var():.1f}")
    if args.data:
        st, T = load_model_state(args.data)
        from photonic_engines.gaussian.sampler import _quadrature_matrix
        m = st.m
        vin = np.full(2 * m, 0.25); vin[:m] = np.exp(-2 * st.r) / 4; vin[m:] = np.exp(2 * st.r) / 4
        TR = _quadrature_matrix(T)
        V = (TR * vin) @ TR.T + (np.eye(2 * m) - TR @ TR.T) / 4
        nmean = (np.diag(V)[:m] + np.diag(V)[m:]) - 0.5                        # <n_i> = <x^2> + <p^2> - 1/2
        print(f"model:  N mean {nmean.sum():.2f} (exact, from the covariance)")
        out["model_N_mean"] = float(nmean.sum())
    if args.device_samples:
        D = np.load(args.device_samples, mmap_mode="r"); D = np.asarray(D).reshape(D.shape[0], -1).astype(np.int64)
        ND = D.sum(1)
        print(f"device: N mean {ND.mean():.2f}, var {ND.var():.1f} ({len(ND)} samples)")
        hi = max(N.max(), ND.max()) + 1
        pa = np.bincount(N, minlength=hi) / len(N); pb = np.bincount(ND, minlength=hi) / len(ND)
        print(f"total-variation distance of P(N), ours vs device: {0.5 * np.abs(pa - pb).sum():.3f} "
              f"(shot-noise floor ~ {math.sqrt(hi / len(N)) / 2:.3f})")
        ma, mb = P.mean(0), D.mean(0)
        print(f"per-mode <n_i>: correlation ours vs device {np.corrcoef(ma, mb)[0, 1]:.3f}, mean ratio "
              f"{ma.sum() / mb.sum():.3f}")
        out["device"] = {"N_mean": float(ND.mean()), "N_var": float(ND.var())}
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True, help="folder with T.npy and r.npy (xanadu-qca-data layout)")
    r.add_argument("--shots", type=int, default=10000)
    r.add_argument("--budget", type=float, default=3000.0, help="seconds of shots at most")
    r.add_argument("--max-photons", type=int, default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--every", type=int, default=200)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True, help="run JSON(s), comma-separated")
    c.add_argument("--data", default=None)
    c.add_argument("--device-samples", default=None)
    c.add_argument("--calib", default=None)
    c.add_argument("--extra", default=None)
    c.add_argument("--json", default=None)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
