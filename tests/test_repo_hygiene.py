"""Repository hygiene: documentation points at files that exist, and nothing is orphaned.

Cheap, CPU-only, no optional dependencies. A failure here is a stale reference or a file nobody points to; fix the doc
(or remove the file) rather than extending the allow-lists below without a reason next to the entry.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Not part of the repository: working output and local environments.
SKIP_DIRS = {".git", "venv", ".venv", "out", "build", "dist", "__pycache__", ".pytest_cache", ".ruff_cache"}

# Deliberate mentions of files that are not in this repository, per document.
ALLOWED = {
    "docs/CONTRIBUTING.md": set(),
    "reproduce/MANIFEST.md": {"postselect_v2.png"},                 # a figure the reproduce script writes to --out
    "docs/TUTORIAL.md": {"script.py"},                              # placeholder: mpirun ... script.py
    "docs/DISTRIBUTED.md": {"script.py"},                           # placeholder
    "docs/ARCHITECTURE.md": {"programs.py"},                        # a proposed module, not an existing one
    "docs/DESIGN.md": {"blocked_kernel.py"},                        # slos_mpi, the predecessor MPI code base
    "docs/DISTINGUISHABILITY.md": {"loss.py"},                      # records that loss.py was absorbed
    "docs/GAUSSIAN.md": {                                           # files of Piquasso / DeepQuantum (comparison section)
        "powtrace.py", "labudde.py", "loop_corrections.py", "hafnian_.py"},
    "docs/GAUSSIAN_SURVEY.md": {                                    # files of The Walrus / BosonSupremacy
        "loop_hafnian_batch.py", "_hafnian.py", "NOTICE.md"},
    "docs/SLOS_IMPLEMENTATION.md": {                                # provenance note of the SLOS engine, top of file
        "SLOScore_REPORT.md", "CLIFFORD_IMPLEMENTATION.md", "test_sloscore_ebs.py",
        "test_grad_agreement.py"},
}

REF = re.compile(r"(?<![\w.-])([\w][\w./-]*\.(?:py|md|sh|tex|png|pdf))\b")


def _files() -> list[str]:
    """Tracked files (git), or every file outside SKIP_DIRS when git is unavailable."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [f for f in out.splitlines() if (ROOT / f).exists()]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for d, dirs, names in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")]
        files += [os.path.relpath(os.path.join(d, n), ROOT) for n in names]
    return files


FILES = _files()
BASENAMES = {os.path.basename(f) for f in FILES}
DOCS = [f for f in FILES if f.endswith(".md")]
TEXT = [f for f in FILES if f.endswith((".md", ".py", ".sh", ".yml", ".toml"))]


def _read(f: str) -> str:
    return (ROOT / f).read_text(errors="ignore")


def test_documentation_references_resolve():
    bad = []
    for d in DOCS:
        allowed = ALLOWED.get(d, set())
        for ref in sorted(set(REF.findall(_read(d)))):
            base = os.path.basename(ref)
            if "://" in ref or base in allowed or base in BASENAMES:
                continue
            bad.append(f"{d}: {ref}")
    assert not bad, "documentation names files that do not exist:\n  " + "\n  ".join(bad)


def test_every_design_note_is_listed():
    index = _read("README.md") + _read("docs/ARCHITECTURE.md")
    notes = [f for f in FILES if re.fullmatch(r"docs/[A-Z_]+\.md", f)]
    missing = [f for f in notes if os.path.basename(f) not in index]
    assert not missing, f"docs/*.md listed neither in README.md nor in docs/ARCHITECTURE.md: {missing}"


def _referenced_outside(path: str, needle: str) -> bool:
    return any(needle in _read(f) for f in TEXT if f != path)


def test_every_bench_script_is_referenced():
    orphans = [f for f in FILES if f.startswith("bench/") and f.endswith((".py", ".sh"))
               and not _referenced_outside(f, os.path.basename(f)) and not _referenced_outside(f, Path(f).stem)]
    assert not orphans, f"bench scripts no doc, script or test mentions (document or remove them): {orphans}"


def test_every_figure_is_referenced():
    figs = [f for f in FILES if f.startswith("img/") and f.endswith((".png", ".pdf")) and not f.startswith("img/brand/")]
    orphans = [f for f in figs if not _referenced_outside(f, Path(f).stem)]
    assert not orphans, f"figures nothing references (docs, bench, reproduce): {orphans}"
