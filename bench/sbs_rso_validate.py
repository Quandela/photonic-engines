"""Validation of an SBS rso build: three short checks before any frontier block.

2026-10-05. Run on the GPU that is free (~20-30 min); writes one CSV and a verdict line per check.

(c) exactness. The rso path proper (--square kernel | chain: no stratum handed to fock_sample) against the strong simulation
    (ProductSLOSKernel) at n = 6 sources (alpha = pi/4, 1.1, lossy eta = 0.7) and against the Fock law at alpha = pi/2
    (k = n through the rso chain). TVD below 2.5 x the multinomial floor.
(b) per-step and per-shot cost at an empty offset set (alpha = pi/2, --square): one chain step on the backend
    under test vs the production Fock cofactor step (cuda_cofactors.cofactor_partials_cuda, same S, same K), and a
    whole k = n shot vs fock_sample. The shot time is split into the cofactor calls and the rest (bookkeeping).
    Acceptance: shot ratio <= 1.3.
(a) continuity in p = sin^2(alpha) at fixed n, m = 2n: seconds per sample on the rso path for p = 0.8, 0.9, 0.95, 0.99
    and 1, next to the Fock engine at the same n. Acceptance: monotone in p (t(p_i) <= 1.15 t(p_i+1)), and
    t(p = 0.99) / t_Fock <= 1.3. Estimator: the stratified sum t(p) = sum_s Bin(n, 1-p)(s) t_s0(s), each
    t_s0 the mean cost per shot of --stratum-shots (>= 128) post-selected shots with k = n - s photons, measured in
    --batches lockstep batches whose spread gives its standard error; strata s = 0 ... the first s where the binomial
    mass reaches --mass (0.999) at the smallest p (shared by every p: a post-selected shot's cost does not depend on p);
    t_Fock from --fock-shots (>= 512) shots, also batched. Errors propagate as independent. A comparison whose margin is
    within 1 s.e. of its threshold is AMBIGUOUS: every stratum and t_Fock are re-measured with 3x more shots, pooled
    (4x in all), and the verdict is taken from the pooled values (reported as such). One row per stratum (check a_s0),
    per p (check a, with se) and per comparison (check a_cmp, margin and its se).
(a-split) diagnostics only, no verdict (locates the x1.5 residual between the predicted and measured (a)). At p = --split-p
    (0.99) the same --shots-cost run is repeated with sample(stats=) and the cofactor kernels timed: one row per
    (n_live, k) pass with streams, s0, seconds, cofactor seconds (cuda_rect.rect_cofactors_cuda / _emul and the Fock step
    kernel cuda_cofactors.cofactor_partials_cuda, synchronised), lane-planning seconds, and the rest (= chain
    bookkeeping: gathers, draws, launches, host sync). Then every s0 = 0..3 stratum post-selected (photons = n - s0) at
    each of --split-shots streams (16, 64): seconds per shot and cofactor share -- separates a per-stratum batch-size
    effect (fewer streams per kernel call at p < 1) from per-step bookkeeping and from the kernel's per-term cost.

(lossy) --lossy-only, diagnostics, no verdict: the (a) strata of SBS against single photons under loss eta = p on the
    same instance (fock_sample on k = n - s0 inputs, alternating batches), both stratified over Bin(n, 1 - p); one ratio
    per p (PE review 11-07, point 3).

    python3 bench/sbs_rso_validate.py --backend cuda --csv out.csv
    python3 bench/sbs_rso_validate.py --backend cuda --lossy-only --n-cont 34 --csv lossy.csv
    python3 bench/sbs_rso_validate.py --backend cuda --split-only --n-cont 30 --split-p 1 --split-s0 0 --csv split.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import torch


def haar(m: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(m, m, generator=g, dtype=torch.float64) + 1j * torch.randn(m, m, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    return (q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)).to(dev)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def tvd_floor(occ, ref, shots):
    emp = {}
    for row in occ.tolist():
        emp[tuple(row)] = emp.get(tuple(row), 0) + 1.0 / shots
    tvd = 0.5 * sum(abs(emp.get(k, 0.0) - ref.get(k, 0.0)) for k in set(emp) | set(ref))
    floor = 0.5 * math.sqrt(2 / math.pi) * sum(math.sqrt(q * (1 - q) / shots) for q in ref.values())
    return tvd, floor


def main():
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import cuda_rect, fock_sample, sbs_rso
    from photonic_engines.slos import ProductSLOSKernel
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--draw", default="device", help="sbs_rso draw: device (fused Triton draw) | host")
    ap.add_argument("--square", default="kernel", help="k = n_live strata: kernel (rso chain on the Fock step kernels, "
                    "default) | chain (rectangular cuda_rect payload)")
    ap.add_argument("--shots-exact", type=int, default=20000)
    ap.add_argument("--n-cost", type=int, default=30, help="(b): k = n photons, m = 2n")
    ap.add_argument("--n-cont", type=int, default=34, help="(a): sources, m = 2n")
    ap.add_argument("--shots-cost", type=int, default=64, help="(b): shots per timing")
    ap.add_argument("--stratum-shots", type=int, default=128, help="(a): post-selected shots per stratum (>= 128)")
    ap.add_argument("--fock-shots", type=int, default=512, help="(a): Fock-engine shots for t_Fock (>= 512)")
    ap.add_argument("--batches", type=int, default=4, help="(a): lockstep batches per stratum (standard errors)")
    ap.add_argument("--mass", type=float, default=0.999, help="(a): binomial mass covered by the strata")
    ap.add_argument("--split-only", action="store_true", help="(a-split) alone: no (c), (b), (a), no VERDICT")
    ap.add_argument("--split-p", type=float, default=0.99, help="(a-split): p of the per-pass breakdown (0 = skip)")
    ap.add_argument("--split-shots", default="16,64", help="(a-split): streams per post-selected s0 stratum")
    ap.add_argument("--split-s0", default="0,1,2,3", help="(a-split): post-selected offsets s0 = n - k")
    ap.add_argument("--lossy-only", action="store_true", help="(lossy) alone: SBS vs single photons under loss eta = p, "
                    "stratified, at --n-cont; no (c), (b), (a), no VERDICT")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows_out, verdict = [], {}

    def rec(**kw):
        rows_out.append(kw)
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)

    kw_rso = {"backend": args.backend, "square": args.square, "device": dev, "draw": args.draw}
    if args.backend == "cuda" and hasattr(cuda_rect, "warmup_mu"):
        t0 = time.perf_counter()                                              # NVRTC of the mu lane (all tiers, D = 1..6)
        cuda_rect.warmup_mu()                                                 # outside every timed region (H, r10-gate-a §5)
        print(f"warmup_mu: {time.perf_counter() - t0:.1f} s", flush=True)
    if args.lossy_only:
        n = args.n_cont
        U = haar(2 * n, seed=7 * n, dev=dev)                                     # the (a) instance
        lossy_baseline(args, n, 2 * n, U, dev, kw_rso, "cuda" if dev.type == "cuda" else "triton", rec)
        write_csv(args.csv, rows_out)
        return
    if args.split_only:
        n = args.n_cont
        m = 2 * n
        U = haar(m, seed=7 * n, dev=dev)                                         # the (a) instance
        fb = "cuda" if dev.type == "cuda" else "triton"
        fock_sample(U, list(range(n)), 2, generator=torch.Generator(device=dev).manual_seed(1), backend=fb)
        sync(dev)
        t0 = time.perf_counter()
        fock_sample(U, list(range(n)), args.shots_cost, generator=torch.Generator(device=dev).manual_seed(3), backend=fb)
        sync(dev)
        t_fock = (time.perf_counter() - t0) / args.shots_cost
        rec(check="a_fock", n=n, value=t_fock, shots=args.shots_cost)
        st = ProductSuperposition.sbs(n, m, alpha=math.asin(math.sqrt(args.split_p)))
        sbs_rso.sample(st, U, 2, generator=torch.Generator().manual_seed(1), **kw_rso)        # warm-up / compile
        split_a(args, n, m, U, dev, kw_rso, t_fock, rec)
        write_csv(args.csv, rows_out)
        return
    # ---- (c) exactness
    ok_c = True
    for (alpha, eta) in [(math.pi / 4, 1.0), (1.1, 1.0), (math.pi / 4, 0.7), (math.pi / 2, 1.0)]:
        st, U = ProductSuperposition.sbs(6, 10, alpha=alpha), haar(10, seed=int(100 * alpha) + int(10 * eta), dev="cpu")
        ref = ProductSLOSKernel(st, device="cpu", dtype=torch.complex128).probs(
            U, **({} if eta >= 1 else {"loss": eta})).as_dict()
        occ = sbs_rso.sample(st, U, args.shots_exact, generator=torch.Generator().manual_seed(3),
                             loss=None if eta >= 1 else eta, **kw_rso)
        tvd, floor = tvd_floor(occ, ref, args.shots_exact)
        ok = tvd < 2.5 * floor
        ok_c &= ok
        rec(check="c", alpha=round(alpha, 4), eta=eta, n=6, value=tvd, ref=floor, ok=ok)
    verdict["c"] = ok_c

    # ---- (b) per-step and per-shot cost at an empty offset set
    n = args.n_cost
    m = 2 * n
    U = haar(m, seed=n, dev=dev)
    st = ProductSuperposition.sbs(n, m, alpha=math.pi / 2)
    cof_time = {"cof": 0.0, "plan": 0.0}
    orig = {}

    def timed(key, fn):
        def wrap(*a, **k):
            sync(dev)
            t0 = time.perf_counter()
            r = fn(*a, **k)
            sync(dev)
            cof_time[key] += time.perf_counter() - t0
            return r
        return wrap
    if args.backend in ("cuda", "emul"):
        from photonic_engines.clifford import cuda_rect
        name = "rect_cofactors_cuda" if args.backend == "cuda" else "rect_cofactors_emul"
        orig[name] = getattr(cuda_rect, name)
        setattr(cuda_rect, name, timed("cof", orig[name]))
        if hasattr(cuda_rect, "_plan"):                                       # host-side lane planning inside the call
            orig["_plan"] = cuda_rect._plan
            cuda_rect._plan = timed("plan", orig["_plan"])
    sbs_rso.sample(st, U, 2, generator=torch.Generator().manual_seed(1), **kw_rso)            # warm-up / compile
    cof_time.update(cof=0.0, plan=0.0)
    sync(dev)
    t0 = time.perf_counter()
    sbs_rso.sample(st, U, args.shots_cost, generator=torch.Generator().manual_seed(2), **kw_rso)
    sync(dev)
    t_rso = (time.perf_counter() - t0) / args.shots_cost
    t_cof = cof_time["cof"] / args.shots_cost
    t_plan = cof_time["plan"] / args.shots_cost
    for name, fn in orig.items():
        setattr(cuda_rect, name, fn)
    fb = "cuda" if dev.type == "cuda" else "triton"
    fock_sample(U, list(range(n)), 2, generator=torch.Generator(device=dev).manual_seed(1), backend=fb)
    sync(dev)
    t0 = time.perf_counter()
    fock_sample(U, list(range(n)), args.shots_cost, generator=torch.Generator(device=dev).manual_seed(2), backend=fb)
    sync(dev)
    t_fock = (time.perf_counter() - t0) / args.shots_cost
    rec(check="b", alpha=round(math.pi / 2, 4), eta=1.0, n=n, value=t_rso, ref=t_fock, ok=t_rso <= 1.3 * t_fock,
        cofactor_share=t_cof / t_rso if t_rso else float("nan"), plan_share=t_plan / t_rso if t_rso else float("nan"))
    verdict["b"] = t_rso <= 1.3 * t_fock

    # ---- (a) continuity in p at fixed n: stratified estimator
    n = args.n_cont
    m = 2 * n
    U = haar(m, seed=7 * n, dev=dev)
    ok_a, t_fock = gate_a(args, n, m, U, dev, kw_rso, fb, rec)
    verdict["a"] = ok_a
    print("VERDICT " + " ".join(f"({k}) {'GREEN' if v else 'RED'}" for k, v in sorted(verdict.items())), flush=True)
    write_csv(args.csv, rows_out)


PS = (0.8, 0.9, 0.95, 0.99, 1.0)


def strata_needed(n: int, ps, mass: float) -> int:
    """Largest s0 needed so that, at every p in ps, strata 0..s0 carry >= mass of Bin(n, 1 - p)."""
    smax = 0
    for p in ps:
        acc = 0.0
        for s0 in range(n + 1):
            acc += math.comb(n, s0) * (1 - p) ** s0 * p ** (n - s0)
            if acc >= mass:
                break
        smax = max(smax, s0)
    return smax


def mean_se(xs):
    """Mean and standard error of batch means (equal batch sizes)."""
    k = len(xs)
    mu = sum(xs) / k
    if k < 2:
        return mu, float("nan")
    var = sum((x - mu) ** 2 for x in xs) / (k - 1)
    return mu, math.sqrt(var / k)


def stratified(n: int, p: float, tse: dict) -> tuple:
    """t(p) = sum_s Bin(n, 1-p)(s) t_s0(s) over the measured strata, its propagated s.e., and the mass covered."""
    t = var = mass = 0.0
    for s0, (mu, se) in tse.items():
        w = math.comb(n, s0) * (1 - p) ** s0 * p ** (n - s0)
        t += w * mu
        var += (w * se) ** 2
        mass += w
    return t, math.sqrt(var), mass


def judge_a(ts, ses, t_fock, se_fock):
    """Comparisons of gate (a): (name, margin, se, state) with state GREEN / RED / AMBIGUOUS (|margin| <= se).
    Monotone: 1.15 t(p_i+1) - t(p_i) >= 0; level: 1.3 t_Fock - t(0.99) >= 0."""
    out = []
    for i in range(len(ts) - 1):
        d = 1.15 * ts[i + 1] - ts[i]
        se = math.sqrt((1.15 * ses[i + 1]) ** 2 + ses[i] ** 2)
        out.append((f"mono_{PS[i]}_{PS[i + 1]}", d, se))
    i99 = PS.index(0.99)
    d = 1.3 * t_fock - ts[i99]
    se = math.sqrt((1.3 * se_fock) ** 2 + ses[i99] ** 2)
    out.append(("level_0.99", d, se))
    return [(name, d, se, "AMBIGUOUS" if abs(d) <= se else ("GREEN" if d > 0 else "RED")) for name, d, se in out]


def _timed_batches(fn, nb: int, dev) -> list:
    """Seconds per shot of nb calls fn(batch_index) -> shots."""
    out = []
    for b in range(nb):
        sync(dev)
        t0 = time.perf_counter()
        shots = fn(b)
        sync(dev)
        out.append((time.perf_counter() - t0) / shots)
    return out


def gate_a(args, n, m, U, dev, kw_rso, fb, rec):
    """Gate (a) with the stratified estimator. Returns (verdict, t_Fock)."""
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import fock_sample, sbs_rso
    nb = max(2, args.batches)
    smax = strata_needed(n, PS, args.mass)
    st = ProductSuperposition.sbs(n, m, alpha=math.pi / 4)                    # post-selected: the alpha does not enter
    raw = {s0: [] for s0 in range(smax + 1)}
    raw_fock: list = []

    def measure(scale: int, tag: int):
        fs = max(1, args.fock_shots // nb)                                    # fixed batch size; scale = more batches
        raw_fock.extend(_timed_batches(lambda b: (fock_sample(
            U, list(range(n)), fs, generator=torch.Generator(device=dev).manual_seed(1000 * tag + b), backend=fb), fs)[1],
            nb * scale, dev))
        ss = max(1, args.stratum_shots // nb)
        for s0 in raw:
            k = n - s0
            if k < 1:
                continue
            raw[s0].extend(_timed_batches(lambda b, k=k: (sbs_rso.sample(
                st, U, ss, generator=torch.Generator().manual_seed(100000 * tag + 100 * k + b), photons=k, **kw_rso), ss)[1],
                nb * scale, dev))

    fock_sample(U, list(range(n)), 2, generator=torch.Generator(device=dev).manual_seed(1), backend=fb)   # warm-up
    for s0 in raw:
        if n - s0 >= 1:
            sbs_rso.sample(st, U, 2, generator=torch.Generator().manual_seed(1), photons=n - s0, **kw_rso)
    measure(1, 1)
    rounds = [("first", 1)]
    while True:
        tf, sef = mean_se(raw_fock)
        tse = {s0: mean_se(v) for s0, v in raw.items() if v}
        ts, ses = [], []
        for p in PS:
            t, se, _ = stratified(n, p, tse)
            ts.append(t)
            ses.append(se)
        cmp = judge_a(ts, ses, tf, sef)
        if all(c[3] != "AMBIGUOUS" for c in cmp) or len(rounds) > 1:
            break
        measure(3, 2)                                                         # pooled with the first round: 4x shots
        rounds.append(("pooled_4x", 4))
    tag = rounds[-1][0]
    rec(check="a_fock", n=n, value=tf, se=sef, shots=len(raw_fock) * max(1, args.fock_shots // nb), round=tag)
    for s0, (mu, se) in tse.items():
        rec(check="a_s0", n=n, k=n - s0, s0=s0, value=mu, se=se, ref=tf,
            shots=len(raw[s0]) * max(1, args.stratum_shots // nb), round=tag)
    for p, t, se in zip(PS, ts, ses):
        rec(check="a", alpha=round(math.asin(math.sqrt(p)), 4), eta=1.0, n=n, p=p, value=t, se=se, ref=tf,
            ratio=t / tf, mass=stratified(n, p, tse)[2], ok=True, round=tag)
    for name, d, se, state in cmp:
        rec(check="a_cmp", n=n, name=name, value=d, se=se, ok=state == "GREEN", state=state, round=tag)
    states = [c[3] for c in cmp]
    if "AMBIGUOUS" in states:
        print("(a) still within 1 s.e. of a threshold after 4x shots: reported RED (not a verdict) -- see a_cmp rows",
              flush=True)
    return all(x == "GREEN" for x in states), tf


def write_csv(path, rows_out):
    if path:
        keys = sorted({k for r in rows_out for k in r})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows_out)


class _Timers:
    """Synchronised wall-clock of the cofactor kernels and the lane planning, outermost call per key only (cuda_rect
    hands s0 = 0 steps to the Fock kernel inside rect_cofactors_cuda: not counted twice; planning runs inside the
    cofactor call and is reported as its own column, a part of cof)."""

    def __init__(self, dev):
        self.dev, self.t, self.depth, self.orig = dev, {"cof": 0.0, "plan": 0.0, "cof_calls": 0}, {}, []

    def wrap(self, mod, name, key):
        fn = getattr(mod, name)
        self.orig.append((mod, name, fn))

        def w(*a, **k):
            if self.depth.get(key):
                return fn(*a, **k)
            self.depth[key] = 1
            sync(self.dev)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                sync(self.dev)
                self.t[key] += time.perf_counter() - t0
                if key == "cof":
                    self.t["cof_calls"] += 1
                self.depth[key] = 0
        setattr(mod, name, w)

    def restore(self):
        for mod, name, fn in reversed(self.orig):
            setattr(mod, name, fn)
        self.orig = []


class _Passes(list):
    """stats["passes"] that snapshots the timers at every append: per-pass cofactor / planning time by difference."""

    def __init__(self, timers):
        super().__init__()
        self.timers, self.last, self.split = timers, dict(timers.t), []

    def append(self, x):
        now = dict(self.timers.t)
        self.split.append({k: now[k] - self.last[k] for k in now})
        self.last = now
        super().append(x)


def lossy_baseline(args, n, m, U, dev, kw_rso, fb, rec):
    """(lossy) SBS against single photons under loss eta = p on the same instance, both by the stratified estimator.
    Per stratum s0 = 0..smax (k = n - s0 photons): t_SBS(k) post-selected rso shots and t_lossy(k) fock_sample on k of the
    n input modes (Haar U: which k does not change the law of the cost), measured in alternating batches so drift hits
    both alike. t(p) = sum_s Bin(n, 1-p)(s) t(s) for each; one row per stratum (lossy_s0) and per p (lossy: ratio
    t_SBS / t_lossy and its s.e., errors independent)."""
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import fock_sample, sbs_rso
    nb = max(2, args.batches)
    smax = strata_needed(n, PS, args.mass)
    st = ProductSuperposition.sbs(n, m, alpha=math.pi / 4)                    # post-selected: the alpha does not enter
    ss = max(1, args.stratum_shots // nb)
    raw_s = {s0: [] for s0 in range(smax + 1) if n - s0 >= 1}
    raw_l = {s0: [] for s0 in raw_s}
    for s0 in raw_s:                                                          # warm-up / compile every shape
        k = n - s0
        sbs_rso.sample(st, U, 2, generator=torch.Generator().manual_seed(1), photons=k, **kw_rso)
        fock_sample(U, list(range(k)), 2, generator=torch.Generator(device=dev).manual_seed(1), backend=fb)
    for s0 in raw_s:
        k = n - s0
        for b in range(nb):
            raw_s[s0].extend(_timed_batches(lambda _b, k=k, b=b: (sbs_rso.sample(
                st, U, ss, generator=torch.Generator().manual_seed(100 * k + b), photons=k, **kw_rso), ss)[1], 1, dev))
            raw_l[s0].extend(_timed_batches(lambda _b, k=k, b=b: (fock_sample(
                U, list(range(k)), ss, generator=torch.Generator(device=dev).manual_seed(100 * k + b), backend=fb),
                ss)[1], 1, dev))
    tse_s = {s0: mean_se(v) for s0, v in raw_s.items()}
    tse_l = {s0: mean_se(v) for s0, v in raw_l.items()}
    for s0 in raw_s:
        rec(check="lossy_s0", n=n, k=n - s0, s0=s0, value=tse_s[s0][0], se=tse_s[s0][1], ref=tse_l[s0][0],
            ref_se=tse_l[s0][1], shots=nb * ss)
    for p in PS:
        ts, ses, mass = stratified(n, p, tse_s)
        tl, sel, _ = stratified(n, p, tse_l)
        r = ts / tl
        rec(check="lossy", n=n, p=p, value=ts, se=ses, ref=tl, ref_se=sel, ratio=r,
            ratio_se=r * math.sqrt((ses / ts) ** 2 + (sel / tl) ** 2), mass=mass)


def split_a(args, n, m, U, dev, kw_rso, t_fock, rec):
    """(a-split): per-pass breakdown at p = args.split_p, then post-selected s0 strata at several stream counts."""
    from photonic_engines import ProductSuperposition
    from photonic_engines.clifford import cuda_cofactors, cuda_rect, sbs_rso
    tm = _Timers(dev)
    tm.wrap(cuda_rect, "rect_cofactors_cuda" if args.backend == "cuda" else "rect_cofactors_emul", "cof")
    if hasattr(cuda_cofactors, "cofactor_partials_cuda"):
        tm.wrap(cuda_cofactors, "cofactor_partials_cuda", "cof")
    tm.wrap(cuda_rect, "_plan", "plan")
    try:
        p = args.split_p
        st = ProductSuperposition.sbs(n, m, alpha=math.asin(math.sqrt(p)))
        stats = {"passes": _Passes(tm)}
        sync(dev)
        t0 = time.perf_counter()
        sbs_rso.sample(st, U, args.shots_cost, generator=torch.Generator().manual_seed(4), stats=stats, **kw_rso)
        sync(dev)
        tot = time.perf_counter() - t0
        for (n_l, k, S, sec), d in zip(stats["passes"], stats["passes"].split):
            rec(check="a_pass", p=p, n=n_l, k=k, s0=n_l - k, streams=S, value=sec / S, ref=t_fock,
                cof_s=d["cof"], plan_s=d["plan"], rest_s=sec - d["cof"], cof_calls=d["cof_calls"], seconds=sec)
        inside = sum(x[3] for x in stats["passes"])
        rec(check="a_pass_total", p=p, n=n, shots=args.shots_cost, value=tot / args.shots_cost, ref=t_fock,
            outside_passes_s=tot - inside, seconds=tot)
        for s0 in (int(x) for x in args.split_s0.split(",")):
            k = n - s0
            st1 = ProductSuperposition.sbs(n, m, alpha=math.pi / 4)
            sbs_rso.sample(st1, U, 2, generator=torch.Generator().manual_seed(1), photons=k, **kw_rso)   # warm-up
            for S in (int(x) for x in args.split_shots.split(",")):
                c0 = dict(tm.t)
                sync(dev)
                t0 = time.perf_counter()
                sbs_rso.sample(st1, U, S, generator=torch.Generator().manual_seed(5), photons=k, **kw_rso)
                sync(dev)
                sec = time.perf_counter() - t0
                cof = tm.t["cof"] - c0["cof"]
                rec(check="a_s0", n=n, k=k, s0=s0, streams=S, value=sec / S, ref=t_fock, cof_s=cof,
                    plan_s=tm.t["plan"] - c0["plan"], rest_s=sec - cof, cof_calls=tm.t["cof_calls"] - c0["cof_calls"],
                    seconds=sec)
    finally:
        tm.restore()


if __name__ == "__main__":
    main()
