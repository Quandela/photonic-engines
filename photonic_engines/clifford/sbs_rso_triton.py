"""Triton kernel for the rectangular cofactors of the random-source-order SBS chain (clifford/sbs_rso.py).

Agent M, 2026-10-03. One stream, one step: the j-1 placed rows have D distinct values b_nu with multiplicities mu_nu,
the active set has N columns, and we need for every column x

    P_x = Per_rect(B[:, T \\ x]) = 2^-q sum_{v} w(v) e_q(a(v) \\ a_x(v)),   q = j - 1,
    a_i(v) = sum_nu (mu_nu - 2 v_nu) b_nu[i],   w(v) = prod_nu (-1)^{v_nu} C(mu_nu, v_nu),

the Glynn sum collapsed over repeated rows (Clifford & Clifford 2020): prod_nu (mu_nu + 1) terms instead of 2^q, halved by
the symmetry v -> mu - v when some mu_nu is odd. The leave-one-out elementary symmetric polynomial is extracted at L = N
points of a circle of radius r (no aliasing: Prod_{i != x}(1 + lam a_i) has degree N - 1),

    e_q(a \\ a_x) = (1/L) sum_t lam_t^-q Prod_{i != x} (1 + lam_t a_i),   lam_t = r exp(2 pi i t / L),

so that every (term, t) pair is a FOCK-type all-but-one product of the shifted column sums y_i = 1 + lam_t a_i -- the same
primitive as the single-photon kernel (`cofactors_triton._allbutone`, P / y_x with exact zero handling). The radius
r = q / ((N - q) sigma), sigma^2 = E|a_i|^2, puts degree q at the peak of the coefficient profile; measured accuracy 1e-15
to 2e-14 against the prefix x suffix reference up to N = 80, q = 39.

Work per term: D N (column sums) + L N (shifted products) complex multiply-adds; registers: [BT, NP] tiles only.

Grid: programs over contiguous ranges of term indices; each program loops over blocks of BT terms (mixed-radix digits
recomputed per block, no Gray code: the D N column-sum cost is at most that of the L N product stage) and writes one
partial [NP] row; the host sums the partials. Deterministic. The tables (`collapse_tables`), the circle (`_circle`) and
the torch twin (`rect_cofactors_collapsed_torch`) live in sbs_rso.py so that the torch path never imports triton.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from .cofactors_triton import _allbutone
from .sbs_rso import _circle, collapse_tables

MAX_PROGRAMS = 1 << 14


@triton.jit
def _rso_cof_kernel(b_re, b_im, mu_ptr, radix_ptr, stride_ptr, wtab_ptr,
                    lam_re, lam_im, coef_re, coef_im,
                    D, L, N, n_terms, terms_per_prog,
                    out_re, out_im,
                    NP: tl.constexpr, MU_PAD: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, NP)
    jmask = cols < N
    lane = tl.arange(0, BT).to(tl.int64)
    acc_re = tl.zeros([NP], dtype=tl.float64)
    acc_im = tl.zeros([NP], dtype=tl.float64)
    start = pid.to(tl.int64) * terms_per_prog
    stop = tl.minimum(start + terms_per_prog, n_terms)
    for base in range(start, stop, BT):
        g = base + lane                                             # [BT] term indices
        valid = g < stop
        a_re = tl.zeros([BT, NP], dtype=tl.float64)
        a_im = tl.zeros([BT, NP], dtype=tl.float64)
        w = tl.where(valid, 1.0, 0.0).to(tl.float64)
        for dd in range(D):
            rad = tl.load(radix_ptr + dd)
            strd = tl.load(stride_ptr + dd)
            mu = tl.load(mu_ptr + dd)
            v = (g // strd) % rad                                   # [BT]
            c = (mu - 2 * v).to(tl.float64)
            w = w * tl.load(wtab_ptr + dd * MU_PAD + v, mask=valid, other=0.0)
            br = tl.load(b_re + dd * NP + cols, mask=jmask, other=0.0)
            bi = tl.load(b_im + dd * NP + cols, mask=jmask, other=0.0)
            a_re += c[:, None] * br[None, :]
            a_im += c[:, None] * bi[None, :]
        for t in range(L):
            lr = tl.load(lam_re + t)
            li = tl.load(lam_im + t)
            y_re = tl.where(jmask[None, :], 1.0 + lr * a_re - li * a_im, 1.0)
            y_im = tl.where(jmask[None, :], lr * a_im + li * a_re, 0.0)
            ab_re, ab_im = _allbutone(y_re, y_im, jmask)            # [BT, NP]
            s_re = tl.sum(w[:, None] * ab_re, axis=0)
            s_im = tl.sum(w[:, None] * ab_im, axis=0)
            cr = tl.load(coef_re + t)
            ci = tl.load(coef_im + t)
            acc_re += cr * s_re - ci * s_im
            acc_im += cr * s_im + ci * s_re
    tl.store(out_re + pid * NP + cols, acc_re, mask=jmask)
    tl.store(out_im + pid * NP + cols, acc_im, mask=jmask)


def rect_cofactors_collapsed_triton(b: torch.Tensor, mu: list[int], BT: int | None = None,
                                    num_warps: int | None = None) -> torch.Tensor:
    """The kernel above. b: [D, N] complex (any device; CUDA in production, CPU under TRITON_INTERPRET=1). Returns P [N]
    complex128. BT (terms per block) and num_warps default to $PE_RSO_BT / $PE_RSO_WARPS, else 16 / 4 -- the launch
    shape picked on the GPU by bench/sbs_rso_kernels.py, which exports them.
    """
    if BT is None:
        BT = int(os.environ.get("PE_RSO_BT", "16"))
    if num_warps is None:
        num_warps = int(os.environ.get("PE_RSO_WARPS", "4"))
    D, N = b.shape
    q = int(sum(mu))
    dev = b.device
    if q == 0:
        return torch.ones(N, dtype=torch.complex128, device=dev)
    radix, stride, wtab, n_terms, factor = collapse_tables(mu)
    NP = max(16, triton.next_power_of_2(N))
    mu_pad = len(wtab[0])
    sigma = math.sqrt(float((torch.tensor(mu, dtype=torch.float64, device=dev)[:, None] * b.abs() ** 2).sum(0).mean()))
    lam, coef = _circle(N, q, sigma, dev)
    bp = torch.zeros((D, NP), dtype=torch.complex128, device=dev)
    bp[:, :N] = b
    b_re, b_im = bp.real.contiguous(), bp.imag.contiguous()
    i64 = {"dtype": torch.int64, "device": dev}
    mu_t = torch.tensor(mu, **i64)
    rad_t = torch.tensor(radix, **i64)
    str_t = torch.tensor(stride, **i64)
    w_t = torch.tensor(wtab, dtype=torch.float64, device=dev).contiguous()
    n_prog = max(1, min(MAX_PROGRAMS, -(-n_terms // BT)))
    per = -(-n_terms // n_prog)
    per = -(-per // BT) * BT
    n_prog = -(-n_terms // per)
    out_re = torch.zeros((n_prog, NP), dtype=torch.float64, device=dev)
    out_im = torch.zeros((n_prog, NP), dtype=torch.float64, device=dev)
    _rso_cof_kernel[(n_prog,)](b_re, b_im, mu_t, rad_t, str_t, w_t,
                               lam.real.contiguous(), lam.imag.contiguous(), coef.real.contiguous(), coef.imag.contiguous(),
                               D, N, N, n_terms, per, out_re, out_im,
                               NP=NP, MU_PAD=mu_pad, BT=BT, num_warps=num_warps)
    P = torch.complex(out_re.sum(0), out_im.sum(0))[:N]
    return P * (factor / 2 ** q)
