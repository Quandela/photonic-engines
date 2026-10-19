"""`bench/find_threshold.py --bracket-min-shots` (shot floor at the bracket points): off by default with the
CSV unchanged; on, both bracket points end with >= N counted shots, or are marked censored by --bracket-max-s.
Runs the script on the CPU (sbs, rso chain, torch backend) at a high target so the bracket falls at small n.
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
BASE = ["--engine", "sbs", "--algo", "rso", "--rso-backend", "torch", "--device", "cpu", "--ratio", "2",
        "--target", "4000", "--start", "12", "--step", "1", "--max-n", "30", "--budget", "4", "--loss", "0.7",
        "--seed", "0"]


def _run(tmp_path, *extra):
    out = tmp_path / "t.csv"
    p = subprocess.run([sys.executable, str(ROOT / "bench" / "find_threshold.py"), *BASE, *extra, "--csv", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return p.stdout, rows


def _bracket(rows):
    pts = sorted((int(r["n"]), float(r["samples_per_s"]), r) for r in rows)
    top = max(n for n, rate, _ in pts if rate >= 4000)
    hi = min(n for n, rate, _ in pts if rate < 4000 and n > top)
    return [r for n, _, r in pts if n in (top, hi)]


def test_off_by_default_no_new_column(tmp_path):
    stdout, rows = _run(tmp_path)
    assert "bracket_censored" not in rows[0] and "top-up" not in stdout


def test_top_up_reaches_the_shot_floor(tmp_path):
    stdout, rows = _run(tmp_path, "--bracket-min-shots", "600", "--bracket-max-s", "60")
    assert "top-up n=" in stdout
    for r in _bracket(rows):
        assert int(r["shots"]) >= 600 and r["bracket_censored"] == "0"


def test_bracket_batch_sets_the_top_up_batch(tmp_path):
    """--bracket-batch B tops up in batches of exactly B, so the counted shots added are a multiple of B."""
    stdout, rows = _run(tmp_path, "--bracket-min-shots", "600", "--bracket-max-s", "60", "--bracket-batch", "64")
    lines = [ln for ln in stdout.splitlines() if "top-up n=" in ln]
    assert lines and all("(batches of 64)" in ln for ln in lines)
    for ln in lines:
        added = int(ln.split("+", 1)[1].split(" ", 1)[0])
        assert added % 64 == 0
    for r in _bracket(rows):
        assert int(r["shots"]) >= 600 and r["bracket_censored"] == "0"


def test_guard_censors(tmp_path):
    stdout, rows = _run(tmp_path, "--bracket-min-shots", "100000", "--bracket-max-s", "0.01")
    assert "BRACKET CENSORED" in stdout
    assert all(r["bracket_censored"] == "1" for r in _bracket(rows))
