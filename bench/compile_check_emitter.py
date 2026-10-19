"""Offline (no GPU needed) compile check of the generated "seq" emitter kernels for sm_90:
compile time, register count and spills (ptxas -v), shuffle count.

    python3 bench/compile_check_emitter.py --K 8,16,22 --B 8,16 --mat-modes scalar,hoist
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

os.environ.pop("TRITON_INTERPRET", None)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from photonic_engines.clifford.emitter_triton_seq import get_kernel

I64 = "*i64"
SIG = {"u_ptr": "*fp64", "drow_ptr": I64, "mult_ptr": I64, "radix_ptr": I64, "n2_ptr": I64, "nterm_ptr": I64,
       "cm_ptr": I64, "lane2term_ptr": I64, "term2stream_ptr": I64, "lstart_ptr": I64,
       "mats_ptr": "*fp64", "s0_ptr": "*fp64", "f_ptr": "*fp64", "binom_ptr": "*fp64", "part_ptr": "*fp64",
       "ctz_ptr": "*i32", "stride_um": "i64", "stride_ds": "i64", "stride_cs": "i64", "stride_mt": "i64",
       "stride_mj": "i64", "LSEG": "i32", "TOTAL": "i32", "SEG": "constexpr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default="8,16,22")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--B", default="8,16")
    ap.add_argument("--threads", type=int, default=128, help="threads per program: SEG = threads / B")
    ap.add_argument("--mat-modes", default="scalar")
    ap.add_argument("--arch", default="sm_90a")
    args = ap.parse_args()
    cap = int(re.search(r"\d+", args.arch).group(0))
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")
    tmp = tempfile.mkdtemp(prefix="pe_compile_check_")
    for K in [int(x) for x in args.K.split(",")]:
        for B in [int(x) for x in args.B.split(",")]:
            for mm in args.mat_modes.split(","):
                seg = max(1, args.threads // B)
                warps = max(1, seg * B // 32)
                fn = get_kernel(K, args.L, B, mm)
                t = time.time()
                k = triton.compile(ASTSource(fn=fn, signature=SIG, constexprs={"SEG": seg}),
                                   target=GPUTarget("cuda", cap, 32), options={"num_warps": warps})
                dt = time.time() - t
                ptx = k.asm["ptx"]
                p = os.path.join(tmp, f"k{K}_b{B}_{mm}.ptx")
                with open(p, "w") as fh:
                    fh.write(ptx)
                info = "ptxas unavailable"
                if os.path.exists(ptxas):
                    out = subprocess.run([ptxas, f"-arch={args.arch}", "-v", "-o", "/dev/null", p],
                                         capture_output=True, text=True).stderr
                    regs = re.search(r"Used (\d+) registers", out)
                    sp = re.search(r"(\d+) bytes spill stores", out)
                    info = f"regs {regs.group(1) if regs else '?'}, spill stores {sp.group(1) if sp else '?'} B"
                print(f"K={K:2d} B={B:2d} {mm:6s} SEG={seg:3d} warps={warps}: compile {dt:5.1f}s, {info}, "
                      f"shfl {ptx.count('shfl')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
