"""Minimal, torch-free repro: does CUDA-Q silently reinterpret a d>2 qudit as a qubit?

`runtime/cudaq/qis/managers/default/DefaultExecutionManager.cpp` (CUDA-Q 0.16.0):

    void allocateQudit(const cudaq::QuditInfo &q) override {
      requestedAllocations.emplace_back(2, q.id);          // q.levels DISCARDED, hardcoded 2
    }
    void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
      simulator()->allocateQubits(qudits.size());          // levels never reach the simulator
    }

ORCA's photonic_tn_simulator branch changes exactly these two lines to propagate q.levels, which
corroborates the reading -- but a qubit-only execution manager should REFUSE a d>2 qudit with a
clear error, not silently substitute 2.  This script checks what actually happens, per target,
each in its own process.  Run it with NO torch imported.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

KERNEL = textwrap.dedent('''
    import cudaq
    cudaq.set_target("{t}")

    @cudaq.kernel
    def k():
        q = [qudit(level=3) for _ in range(2)]
        create(q[0])
        create(q[0])          # two photons in mode 0 -> needs level 3; a qubit cannot hold it
        beam_splitter(q[0], q[1], 0.7)
        mz(q)

    r = cudaq.sample(k, shots_count=200)
    print("RESULT", dict(r.items()))
''')

for t in ("qpp-cpu", "nvidia", "tensornet", "tensornet-mps", "orca-photonics"):
    # @cudaq.kernel needs source that `inspect` can recover, so the kernel must live in a real
    # file -- `python -c` fails with "defined in a non-file context".
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(KERNEL.format(t=t))
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=300)
    finally:
        os.unlink(path)
    out = (p.stdout + p.stderr).strip().splitlines()
    line = next((l for l in out if l.startswith("RESULT")), None)
    if p.returncode != 0:
        why = [l for l in out if l.strip() and "Warning" not in l]
        sig = f"signal {-p.returncode}" if p.returncode < 0 else f"exit {p.returncode}"
        print(f"{t:>16}: CRASH/ERROR ({sig})  {why[-1][:90] if why else ''}")
    elif line:
        # '2' in any slot means a level-3 state was held; all-0/1 means it was truncated to a qubit
        keys = list(eval(line[7:]).keys())
        has2 = any("2" in str(k) for k in keys)
        print(f"{t:>16}: ran, outcomes {keys[:6]}  -> level-3 state {'KEPT' if has2 else 'LOST (truncated to a qubit)'}")
    else:
        print(f"{t:>16}: no result line")
