"""The device comparison: the Borealis table and
figure from the replays of `bench/gbs_device_replay.py` (docs/GAUSSIAN.md §2.4).

    python3 bench/gbs_device_table.py --replays results/keep/gbs_borealis_replay_tiers_20260929.json \
        --data experiments/qca-data --md table.md --png img/gbs_device_comparison.png

--replays: a JSON {run: {mode: {"summary": {...}, "byN": [[N, count_run, priced, mean_s, tier], ...]}}} with the modes
"dd" (the sieve as built with precision "auto": fp64 and double-double, no quad-double kernel; the headline), "qd1" /
"qd10" (a quad-double kernel assumed at 1x / 10x the dd time, not measured) and "fp64" (every call in fp64: the pricing
before 2026-09-28, which the precision tiers invalidate above N ~ 62). The table: per run,
the exact cost of one sample and of the whole run on one H100, the shares measured / priced / extrapolated, and the
authors' classical estimate. The figure: E[cost | N] of one sample against its photon number N for every run, the
segments where the sieve is measured (N <= 48), priced from measured per-term rates (N <= 128) and extrapolated
(beyond), the precision tiers (fp64 / dd / qd, qd priced as dd: a lower bound), the runs' mean and median, the device's
36 us per sample, and Xanadu's estimate converted to H100-seconds by peak fp64.
"""
import argparse
import json
import math

import numpy as np

YEAR = 365.25 * 86400
# peak fp64 (Top500 Rpeak) for the conversion of the authors' estimate; the H100 PCIe fp64 tensor-core peak
FUGAKU_FLOPS, H100_FLOPS = 537.2e15, 51.2e12
XANADU_TC = 9000 * YEAR * FUGAKU_FLOPS / H100_FLOPS          # "> 9,000 years per sample on Fugaku" (Nature 606, 75)
T_Q = 36e-6                                                 # Borealis, s per sample
MEASURED_MAX, RATE_MAX = 48, 128                            # whole calls timed / per-term rate measured (N)
DD_GROUP_MAX = 117                                          # the cross grouping (+5 decades of kappa) fits dd to here
FP64_MAX, DD_MAX = 62, 141                                  # eps kappa(N) <= 1e-3 with log10 kappa = 0.40 + 0.197 N


def fmt(x):
    if x is None or not np.isfinite(x):
        return "—"
    e = math.floor(math.log10(x))
    return f"{x:.2f}" if -2 <= e <= 2 else f"{x / 10 ** e:.1f}·10^{e}"


def dur(x):
    """Seconds as seconds below an hour, hours below 100 days, else years."""
    if x < 3600:
        return f"{x:.2g} s"
    return f"{x / 3600:.0f} h" if x < 100 * 86400 else f"{fmt(x / YEAR)} years"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True)
    ap.add_argument("--data", default=None, help="qca-data directory (r.npy per run) for the squeezing column")
    ap.add_argument("--md", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    R = json.load(open(args.replays))
    runs = [k for k in ("fig3a", "fig3b", "fig4", "figS15") if k in R]
    lines = ["| run | modes | r mean | samples | N mean / max | as built (fp64 + dd): per sample, mean / median (H100-s) "
             "| median sample | whole run | cost measured / priced / extrapolated | with a qd kernel at 1× / 10× dd: "
             "median (H100-s) | fp64 only (v2.11 pricing): median | authors' estimate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in runs:
        s = R[k]["dd"]["summary"]; f = R[k].get("fp64", {}).get("summary")
        q1 = R[k].get("qd1", {}).get("summary"); q10 = R[k].get("qd10", {}).get("summary")
        r = float(np.load(f"{args.data}/{k}/r.npy").mean()) if args.data else float("nan")
        sh = s["shares"]
        authors = f"> 9,000 Fugaku-years ≈ {fmt(XANADU_TC)} H100-s" if k == "fig4" else "—"
        lines.append(
            f"| {k} | {s['m']} | {r:.2f} | {s['S_exp']:,} | {s['N_mean']:.1f} / {s['N_max']} | {fmt(s['mean_s'])} / "
            f"{fmt(s['median_s'])} | {dur(s['median_s'])} | {dur(s['run_s'])} | "
            f"{100 * sh['measured']:.0f} / {100 * sh['priced']:.0f} / {100 * sh['beyond']:.0f} % | "
            + (f"{fmt(q1['median_s'])} / {fmt(q10['median_s'])}" if q1 and q10 else "—") + " | "
            + (f"{fmt(f['median_s'])}" if f else "—") + f" | {authors} |")
    md = "\n".join(lines)
    print(md)
    if args.md:
        open(args.md, "w").write(md + "\n")
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        col = {"fig3a": "#1b9e77", "fig3b": "#7570b3", "fig4": "#d95f02", "figS15": "#e7298a"}
        ymax = 1.0
        for k in runs:
            b = np.array(R[k]["dd"]["byN"], dtype=object)
            nv = b[:, 0].astype(int); t = b[:, 3].astype(float)
            ymax = max(ymax, t.max())
            lab = k
            for lo, hi, ls in ((0, MEASURED_MAX, "-"), (MEASURED_MAX, RATE_MAX, "--"), (RATE_MAX, 10 ** 4, ":")):
                m_ = (nv >= lo) & (nv <= hi)
                if m_.sum():
                    ax.plot(nv[m_], t[m_], ls, color=col[k], lw=1.6, label=lab); lab = None
            if "qd10" in R[k] and R[k]["qd10"]["summary"]["N_max"] > DD_GROUP_MAX:
                q = np.array(R[k]["qd10"]["byN"], dtype=object)
                ax.plot(q[:, 0].astype(int), q[:, 3].astype(float), "-", color=col[k], lw=0.8, alpha=0.5)
                ax.plot([R[k]["qd10"]["summary"]["N_mean"]], [R[k]["qd10"]["summary"]["median_s"]], "v", color=col[k],
                        ms=6, mec="k", mew=0.6)
            s = R[k]["dd"]["summary"]
            ax.plot([s["N_mean"]], [s["mean_s"]], "o", color=col[k], ms=7, mec="k", mew=0.6)
            ax.plot([s["N_mean"]], [s["median_s"]], "s", color=col[k], ms=6, mec="k", mew=0.6)
        ax.axvspan(FP64_MAX, DD_MAX, color="0.9", zorder=0)
        ax.axvspan(DD_MAX, 260, color="0.8", zorder=0)
        ytop = ymax * 10
        ax.text(FP64_MAX / 2, ytop / 20, "fp64", ha="center", fontsize=8)
        ax.text((FP64_MAX + DD_MAX) / 2, ytop / 20, "double-double\n(Bulmer's grouping needs qd past N = 117)",
                ha="center", fontsize=8, va="top")
        ax.text((DD_MAX + 250) / 2, ytop / 20, "quad-double needed\n(as built: priced in dd,\na lower bound)", ha="center",
                fontsize=8, va="top")
        ax.axhline(T_Q, color="k", lw=0.8, ls="-."); ax.text(2, T_Q * 1.6, "Borealis: 36 µs per sample", fontsize=8)
        ax.axhline(1.0, color="0.4", lw=0.8, ls="-."); ax.text(2, 1.6, "1 sample / s", fontsize=8, color="0.3")
        ax.axhline(XANADU_TC, color="k", lw=0.8, ls="--")
        ax.text(2, XANADU_TC * 2, "Xanadu's estimate, 9,000 Fugaku-years (peak-fp64 H100-s)", fontsize=8)
        ax.axvline(MEASURED_MAX, color="0.5", lw=0.6); ax.axvline(RATE_MAX, color="0.5", lw=0.6)
        ax.set_yscale("log"); ax.set_xlim(0, 250); ax.set_ylim(1e-5, ytop)
        ax.set_xlabel("photons N in the sample")
        ax.set_ylabel("exact cost of one sample, one H100 (s)")
        ax.set_title("Borealis runs replayed through the exact PNR sampler's timing model", fontsize=10)
        from matplotlib.lines import Line2D
        h, l_ = ax.get_legend_handles_labels()
        h += [Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--"), Line2D([], [], color="k", ls=":"),
              Line2D([], [], color="w", marker="o", mfc="0.6", mec="k"), Line2D([], [], color="w", marker="s", mfc="0.6", mec="k"),
              Line2D([], [], color="0.5", lw=0.8), Line2D([], [], color="w", marker="v", mfc="0.6", mec="k")]
        l_ += [f"measured calls (N ≤ {MEASURED_MAX})", f"measured per-term rate (N ≤ {RATE_MAX})", "extrapolated rate",
               "run mean (as built: fp64 + dd)", "run median", "with a qd kernel at 10× dd (not measured)",
               "its run median"]
        ax.legend(h, l_, fontsize=7.5, loc="lower right", ncol=2, framealpha=0.95)
        fig.tight_layout(); fig.savefig(args.png, dpi=150)
        print("figure:", args.png)


if __name__ == "__main__":
    main()
