"""Energy per exact Fock sample on the GPU, with MEASURED board power (replaces the assumed
280 W x old-kernel rates of bench/plot_energy.py).

    python3 bench/energy_series.py --ns 12,15,18,20,24,30,34,38 --csv results/energy_r2.csv

Per n (m = round(ratio n), Haar U, input modes 0..n-1, `fock_sample(algo="auto", backend="cuda")` as in
bench/rate_series.py): a warm-up sample (compilation outside the window), a batch ramp 1, 2, 4, ... until one batch
takes >= --batch-seconds (or --max-shots), then batches of that size repeated for >= --window seconds, the board power
logged throughout the window. Before the first n, --idle seconds with the process alive and the GPU idle give the idle
baseline.

Power: NVML through pynvml when importable — the board's cumulative energy counter
(nvmlDeviceGetTotalEnergyConsumption, mJ) read at the window's ends, and nvmlDeviceGetPowerUsage polled every
--period s as a cross-check; otherwise `nvidia-smi --query-gpu=power.draw` polled (energy = its time integral). The
CSV says which (`energy_source`).

CSV per n: samples/s over the window (= the plateau rate of rate_series: the batch is the largest of the ramp),
mean board power (W), idle power (W), Wh per sample = board energy / samples, Wh per sample above idle, the
polled-power integral as a check of the counter, mean SM clock (MHz) and max temperature, batch, samples, seconds.
"""
import argparse
import csv
import os
import subprocess
import threading
import time

import numpy as np
import torch


def haar(m, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


class Power:
    """Board power of GPU `index`: NVML (energy counter + polled power) if pynvml imports, else nvidia-smi polls."""

    def __init__(self, index, period):
        self.index, self.period, self.h = index, period, None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.counter = True
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            except Exception:
                self.counter = False
        except Exception:
            self.nv, self.counter = None, False
        self.source = ("nvml-counter" if self.counter else "nvml-poll") if self.nv else "nvidia-smi-poll"

    def read(self):
        """(watts, sm_clock_mhz, temp_c) now."""
        if self.nv:
            w = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1e3
            clk = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
            tmp = self.nv.nvmlDeviceGetTemperature(self.h, self.nv.NVML_TEMPERATURE_GPU)
            return w, clk, tmp
        out = subprocess.run(["nvidia-smi", "-i", str(self.index), "--query-gpu=power.draw,clocks.sm,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        w, clk, tmp = (float(x) for x in out.strip().split(","))
        return w, clk, tmp

    def energy_j(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1e3 if self.counter else None

    def window(self, body):
        """Run body() while polling; -> (seconds, energy J or None, polled samples [(t, W, MHz, C)], body's result)."""
        samples, stop = [], threading.Event()

        def poll():
            while not stop.is_set():
                samples.append((time.perf_counter(), *self.read()))
                stop.wait(self.period)
        e0 = self.energy_j()
        th = threading.Thread(target=poll, daemon=True)
        t0 = time.perf_counter(); th.start()
        res = body()
        t1 = time.perf_counter(); stop.set(); th.join()
        e1 = self.energy_j()
        return t1 - t0, (None if e0 is None else e1 - e0), samples, res


def integral(samples, t0, t1):
    """Trapezoid integral (J) of the polled power over [t0, t1], held flat to the window's ends."""
    if not samples:
        return float("nan")
    t = np.array([s[0] for s in samples]); w = np.array([s[1] for s in samples])
    t = np.concatenate([[t0], t, [t1]]); w = np.concatenate([[w[0]], w, [w[-1]]])
    return float(np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--ns", default="12,15,18,20,24,30,34,38")
    ap.add_argument("--window", type=float, default=30.0, help="seconds of timed batches per n (power logged)")
    ap.add_argument("--batch-seconds", type=float, default=2.0, help="ramp stops at the first batch this long")
    ap.add_argument("--max-shots", type=int, default=1 << 20)
    ap.add_argument("--idle", type=float, default=15.0, help="seconds of idle baseline before the first n")
    ap.add_argument("--period", type=float, default=0.1, help="power polling period, s")
    ap.add_argument("--gpu-index", type=int, default=0, help="NVML / nvidia-smi index of the GPU used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    from photonic_engines.clifford import fock_sample
    try:
        from photonic_engines.clifford.cuda_cofactors import warmup
        warmup()
    except Exception:
        pass
    dev = "cuda"
    out = a.csv or os.path.join(os.environ.get("RESULTS_DIR", "results"), f"energy_r{a.ratio:g}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pw = Power(a.gpu_index, a.period)
    print(f"{torch.cuda.get_device_name(0)}  m/n={a.ratio:g}  power source: {pw.source}", flush=True)

    torch.cuda.synchronize()
    dt, e, smp, _ = pw.window(lambda: time.sleep(a.idle))
    if e is not None:
        p_idle = e / dt
    else:
        p_idle = integral(smp, smp[0][0], smp[0][0] + dt) / dt if smp else float("nan")
    print(f"idle baseline: {p_idle:.1f} W over {dt:.1f} s ({len(smp)} polls, "
          f"poll mean {np.mean([s[1] for s in smp]):.1f} W)", flush=True)

    rows = []
    for n in [int(x) for x in a.ns.split(",")]:
        m = round(a.ratio * n)
        Ut = torch.tensor(haar(m, a.seed + n), device=dev)
        in_modes = list(range(n))
        rid = [0]

        def run(shots):
            rid[0] += 1
            gen = torch.Generator(device=dev).manual_seed(a.seed + 1000 * rid[0])
            torch.cuda.synchronize(); t = time.perf_counter()
            fock_sample(Ut, in_modes, shots, generator=gen, algo="auto", backend="cuda")
            torch.cuda.synchronize()
            return time.perf_counter() - t

        run(1)                                                   # warm-up, outside the window
        shots = 1
        while shots < a.max_shots and run(shots) < a.batch_seconds:
            shots *= 2

        def body():
            done, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < a.window:
                run(shots); done += shots
            return done
        torch.cuda.synchronize()
        sec, ej, smp, done = pw.window(body)
        t_lo = smp[0][0] if smp else 0.0
        e_poll = integral(smp, t_lo, t_lo + sec)
        e_board = ej if ej is not None else e_poll
        p_mean = e_board / sec
        rate = done / sec
        wh = e_board / done / 3600
        wh_dyn = (e_board - p_idle * sec) / done / 3600
        clk = float(np.mean([s[2] for s in smp])) if smp else float("nan")
        tmax = float(np.max([s[3] for s in smp])) if smp else float("nan")
        print(f"n={n:2d} m={m:3d}: {rate:.4g} samples/s, board {p_mean:.1f} W (idle {p_idle:.1f}), "
              f"{wh:.3e} Wh/sample ({wh_dyn:.3e} above idle), poll/counter {e_poll / e_board:.3f}, "
              f"SM {clk:.0f} MHz, max {tmax:.0f} C, batch {shots}, {done} samples in {sec:.1f} s", flush=True)
        rows.append({"n": n, "m": m, "ratio": a.ratio, "samples_per_s": rate, "board_w": p_mean, "idle_w": p_idle,
                     "wh_per_sample": wh, "wh_per_sample_above_idle": wh_dyn, "energy_j": e_board,
                     "poll_integral_j": e_poll, "energy_source": pw.source, "sm_clock_mhz": clk, "max_temp_c": tmax,
                     "batch": shots, "samples": done, "seconds": sec, "polls": len(smp)})
        with open(out, "w", newline="") as fh:                   # after every n: a timeout keeps what ran
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
