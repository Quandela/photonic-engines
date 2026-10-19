"""Imperfections shared by every engine: uniform photon loss (docs/LOSS.md) and partial
distinguishability (docs/DISTINGUISHABILITY.md). Both are input-side, per-photon and
independent, so one draw covers both (`partition`, `draw_roles`).

LOSS.
Model: one end-to-end transmission `eta` per photon (source brightness x circuit x detector
efficiency), the same on every mode. Uniform loss commutes with any linear-optical unitary, so
it can be applied on the INPUT (each photon survives independently, k ~ Binom(n, eta) of them
enter the interferometer) or on the OUTPUT (each detected photon is kept with probability eta)
with the same result. The two sides differ only in cost, and each engine picks the cheaper:

  strong simulation   output side: one lossless distribution over the n-photon basis, then a
                      photon-removal recurrence down the basis layers, `thin_distribution`
                      (O(edges) = one SLOS layer pass, batched over U, differentiable in the
                      probabilities and in eta)
  sampling            input side: the shots are split by their surviving input subset
                      (`split_shots`, host side) and every group is one batched call of the
                      lossless sampler on a k-photon instance - the lockstep kernels are kept
                      whole and the expected cost sum_k P(k) cost(k) is below the lossless one
  time-loop register  output side: lossless samples / marginals / distribution, then `thin_counts`
                      / `thin_marginals` / `thin_patterns` (the register basis is built for N,
                      so an input-side split would not shrink the per-shot cost)

Threshold detectors are a post-processing of counts (`clicks`). Per-mode transmissions or
mode-dependent detector efficiencies are not uniform and are not covered here; the time-loop
engine keeps its Kraus loss channels (loops, detector) for that.

PARTIAL DISTINGUISHABILITY.
Model. Every photon carries an internal (spectro-temporal) state on top of its mode. Two
models are exactly simulable by the lossless engines:

  x-model         every pair of photons has overlap x (V_HOM = x^2). Equivalent, for the
                  output distribution of any interferometer and any input, to the "orthogonal
                  bad photon" picture: each photon is, independently with probability x, in the
                  common internal mode, and otherwise in a mode orthogonal to every other
                  photon's (a permutation moving j photons weighs x^j in both).
  random species  i.i.d. mixed internal states with eigenvalues q_mu: each photon draws a
                  species mu ~ q, photons of one species interfere, different species do not
                  (the weight of a permutation is prod over its cycles of Tr rho^|c| =
                  sum_mu q_mu^|c|). The x-model is q = (x, 0, 0, ...) with the remainder 1 - x
                  spread over fresh orthogonal modes.

So a shot is a PARTITION of the photons: lost (1 - eta), then per surviving photon a species
(or "own species", a singleton). Photons of one species form an ordinary instance of the
lossless engine on their columns; a singleton is a one-photon species. Outputs of different
species add (the detector does not see species). A singleton photon is NOT a lost photon and
NOT a label measurement: on emitter inputs its label stays coherent with the register, and it
is folded into the chain at its drawn output row (`EmitterMPS.fold` with
sum_l U[r, modes[j][l]] M_{j,l}); on Fock inputs that reduces to classical routing |U[r, i]|^2.

The cost of a sample is that of the largest interfering instance: g ~ Bin(n, eta x) good
photons in the x-model, so the frontier moves as n*(eta, x) = n*(1) log rho / log(1 - eta x +
eta x rho) — the loss law with eta -> eta x. Nothing here is approximate.

DARK COUNTS.
Model: `dark` is the mean number of dark counts per output mode per detection window, a scalar
or one value per mode. Dark counts are independent of the light and of each other: mode i gets
n_i ~ Poisson(dark_i) extra counts. They act after loss and detection, and before the threshold
clip:

  PNR         o'_i = o_i + n_i
  threshold   c'_i = c_i OR (n_i > 0),  P(dark click on i) = 1 - exp(-dark_i)

so the transition factorises over the modes, T(o' | o) = prod_i t_i(o'_i | o_i), and is the same
for every engine (output side, classical, independent of the state):

  sampling            `add_dark_counts`: Poisson draws on the samples, never truncated
  strong simulation   `dark_distribution`: the law convolved with T, mode by mode. Under PNR the
                      support is infinite and `max_dark` (most dark counts in one pattern, summed
                      over the modes) is required; the dropped mass P(Poisson(sum_i dark_i) >
                      max_dark) is returned, never renormalised away. Under threshold the support
                      is finite (max_dark = None is exact); max_dark caps the visible dark clicks
  conditioning        `dark_predecessors`: the true patterns that can produce an observed one, with
                      T(observed | true) — what an engine needs to condition a quantum memory on a
                      noisy outcome, rho | o' = sum_o T(o' | o) K_o rho K_o^dag / p(o')

dark = 0 adds nothing (no draw, no arithmetic): the inputs come back unchanged, clipped to clicks
when threshold=True.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# ==== uniform loss ==============================================================================

Eta = float | torch.Tensor


def _eta(eta) -> float | None:
    """None / >= 1 -> no loss."""
    if eta is None:
        return None
    if isinstance(eta, torch.Tensor):
        return None if float(eta.detach()) >= 1.0 else eta
    e = float(eta)
    if not 0.0 <= e <= 1.0:
        raise ValueError(f"loss (transmission) must be in [0, 1], got {e}")
    return None if e >= 1.0 else e


def binomial_pmf(n: int, eta: Eta, dtype=torch.float64, device=None) -> torch.Tensor:
    """P(k surviving photons out of n), k = 0..n (differentiable in eta when it is a tensor)."""
    e = torch.as_tensor(eta, dtype=dtype, device=device)
    k = torch.arange(n + 1, dtype=dtype, device=e.device)
    logc = torch.lgamma(torch.tensor(n + 1.0, dtype=dtype, device=e.device)) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return torch.exp(logc) * e ** k * (1 - e) ** (n - k)


# ---- samples --------------------------------------------------------------------------------

def thin_counts(counts: torch.Tensor, eta: Eta, generator: torch.Generator | None = None) -> torch.Tensor:
    """Output-side loss on samples: every photon of `counts` [..., m] (int) is kept independently
    with probability eta. Returns the same shape.
    """
    e = _eta(eta)
    if e is None:
        return counts
    p = torch.full(counts.shape, float(e), dtype=torch.float64, device="cpu")
    c = counts.to("cpu", torch.float64)
    kept = torch.binomial(c, p, generator=generator)
    return kept.to(counts.device, counts.dtype)


def clicks(counts: torch.Tensor) -> torch.Tensor:
    """Threshold detectors: 1 where at least one photon, same shape and dtype."""
    return (counts > 0).to(counts.dtype)


def split_shots(n: int, shots: int, eta: Eta, generator: torch.Generator | None = None,
                streams: torch.Tensor | None = None) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    """Input-side loss for samplers: draw, for each of `shots` streams, which of the n input photons
    survive, and group the streams by surviving subset. Returns [(subset, stream_ids)] with
    subset a sorted tuple of photon indices (length k) and stream_ids an int64 tensor (cpu) of
    the streams that drew it; `streams` (default arange(shots)) are the ids to report. The
    number of groups is at most 2^n but for eta near 1 the mass sits on k in {n, n-1, n-2}.
    """
    e = _eta(eta)
    ids = torch.arange(shots) if streams is None else torch.as_tensor(streams).reshape(-1).cpu()
    if e is None:
        return [(tuple(range(n)), ids)]
    keep = torch.rand((shots, n), generator=generator) < float(e)                    # [shots, n] bool
    if n == 0:
        return [((), ids)]
    key = (keep.to(torch.int64) << torch.arange(n, dtype=torch.int64)).sum(dim=1)       # subset as bitmask
    uniq, inv = torch.unique(key, return_inverse=True)
    order = torch.argsort(inv, stable=True)
    counts = torch.bincount(inv, minlength=uniq.shape[0])
    groups, start = [], 0
    for g, c in enumerate(counts.tolist()):
        mask = int(uniq[g])
        subset = tuple(j for j in range(n) if (mask >> j) & 1)
        groups.append((subset, ids[order[start:start + c]]))
        start += c
    return groups


# ---- distributions ---------------------------------------------------------------------------

@dataclass
class LossyDistribution:
    """A distribution over 0..n photons: `probs[k]` is [S_k] (or [B, S_k]) over `occupations(k)`
    [S_k, m]; the layers sum to the binomial weights P(k). `mass(k)` = P(k).
    """
    layers: object                      # anything with .n, .m and per-layer occupations (see below)
    probs: list[torch.Tensor]
    _occ: list[np.ndarray] | None = None

    @property
    def n(self) -> int:
        """Lossless photon number, i.e. the top layer index."""
        return len(self.probs) - 1

    def occupations(self, k: int) -> np.ndarray:
        """The k-photon output patterns, [S_k, m] int (cached `_occ` if given, else read off `layers`)."""
        if self._occ is not None:
            return self._occ[k]
        return _layer_occupations(self.layers, k)

    def mass(self, k: int) -> torch.Tensor:
        """Total probability of detecting exactly k photons, P(k) — scalar, or [B] when batched."""
        return self.probs[k].sum(dim=-1)

    def conditional(self, k: int) -> torch.Tensor:
        """Distribution over the k-photon patterns given k photons were detected (post-selection)."""
        return self.probs[k] / self.mass(k).unsqueeze(-1).clamp_min(1e-300)

    def flat(self) -> tuple[np.ndarray, torch.Tensor]:
        """All patterns stacked: (occupations [S_total, m], probs [S_total] or [B, S_total])."""
        occ = np.concatenate([self.occupations(k) for k in range(self.n + 1)], axis=0)
        return occ, torch.cat(self.probs, dim=-1)

    def as_dict(self, tol: float = 0.0, batch: int | None = None) -> dict[tuple[int, ...], float]:
        """All patterns of every layer as {occupation tuple: probability}, keeping entries above `tol`.

        `batch` selects one distribution when the probabilities are batched.
        """
        occ, p = self.flat()
        if batch is not None:
            p = p[batch]
        p = p.detach().cpu().to(torch.float64).numpy()
        return {tuple(int(v) for v in o): float(q) for o, q in zip(occ, p) if q > tol}

    def clicks(self) -> tuple[np.ndarray, torch.Tensor]:
        """Threshold-detector distribution: (click patterns [C, m] in {0,1}, probs)."""
        occ, p = self.flat()
        return thin_to_clicks(occ, p)

    def sample(self, shots: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Samples [shots, m] (one distribution: probs must be unbatched)."""
        occ, p = self.flat()
        if p.dim() != 1:
            raise ValueError("one distribution at a time")
        return torch.as_tensor(occ)[draw_indices(p.detach().cpu().to(torch.float64), shots, generator)]


MULTINOMIAL_MAX = 1 << 24          # torch.multinomial: "number of categories cannot exceed 2^24"


def draw_indices(p: torch.Tensor, shots: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """`shots` indices drawn from the unnormalized weights p [S], with replacement.

    `torch.multinomial` caps the number of categories at 2^24, which a strong-simulation draw
    passes at n = 10 over a graded basis (30 M states at m = 2n) and at n = 11 over a Fock one
    (129 M). Above the cap this falls back to the inverse CDF: one cumulative sum plus a
    `searchsorted`, O(S) memory and O(shots log S) time, no cap. At or below it the call is
    `torch.multinomial` exactly as before, so an existing draw is unchanged bit for bit.
    """
    if p.numel() <= MULTINOMIAL_MAX:
        return torch.multinomial(p, shots, replacement=True, generator=generator)
    cdf = torch.cumsum(p, dim=0)
    u = torch.rand(shots, dtype=cdf.dtype, device=cdf.device, generator=generator) * cdf[-1]
    return torch.searchsorted(cdf, u).clamp_(max=p.numel() - 1)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _layer_occupations(layers, k: int) -> np.ndarray:
    t = _np(layers.states[k])
    S = t.shape[0]
    occ = np.zeros((S, layers.m), dtype=np.int64)
    ar = np.arange(S)
    for j in range(k):
        occ[ar, t[:, j]] += 1
    return occ


def _descend(layers, q_k: torch.Tensor, k: int, chunk: int = 1 << 24) -> torch.Tensor:
    """One step of the uniform photon-removal recurrence, layer k -> k-1 (see `thin_distribution`
    for the derivation): q_{k-1}(t) = sum_j q_k(t + e_j) (t_j + 1) / k, one scatter-add over the
    k-1 -> k edges grouped by raised mode, mode slab by mode slab, in chunks of `chunk` edges.
    q_k is [S_k] or [B, S_k]; the result is [S_{k-1}] / [B, S_{k-1}]. Differentiable in q_k.
    """
    dev, dt = q_k.device, q_k.dtype
    parent, child, off = layers.edges_by_mode(k - 1)
    off = [int(v) for v in _np(off)]
    states_k = layers.states[k]
    out = torch.zeros((*q_k.shape[:-1], layers.size(k - 1)), dtype=dt, device=dev)
    for j in range(layers.m):
        for a in range(off[j], off[j + 1], chunk):
            b = min(a + chunk, off[j + 1])
            c = torch.as_tensor(child[a:b], device=dev).to(torch.int64)
            p_idx = torch.as_tensor(parent[a:b], device=dev).to(torch.int64)
            if isinstance(states_k, torch.Tensor):
                cnt = (states_k[c] == j).sum(dim=1)
            else:
                cnt = torch.as_tensor((states_k[_np(c)] == j).sum(axis=1), device=dev)
            w = (cnt.to(torch.float64) / k).to(dt)                                 # (t_j + 1) / k
            out = out.index_add(-1, p_idx, q_k.index_select(-1, c) * w)
    return out


def thin_graded(layers, blocks: Sequence[torch.Tensor], eta: Eta,
                chunk: int = 1 << 24) -> LossyDistribution:
    """Output-side uniform loss on a distribution that ALREADY spans 0..n photons — what an engine
    with an indefinite input photon number returns (`slos.ProductSLOSKernel`), as opposed to the
    single n-photon vector `thin_distribution` takes.

    Each source layer is thinned with its own binomial weights and the results are summed per
    detected count: P_loss(j) = sum_{k >= j} C(k, j) eta^j (1 - eta)^(k-j) proj_{k->j}(blocks[k]),
    `proj` being the repeated `_descend`. Cost: sum_k k descents, i.e. n times one
    `thin_distribution` — this is a strong-simulation path (n <= ~12), not a sampler.

    `eta = 1` (or None) returns the blocks unchanged, exactly: no arithmetic is applied at all.
    """
    e = _eta(eta)
    blocks = list(blocks)
    if e is None:
        return LossyDistribution(layers, blocks)
    n = len(blocks) - 1
    dt, dev = blocks[-1].dtype, blocks[-1].device
    acc = [torch.zeros_like(b) for b in blocks]
    for k in range(n, -1, -1):
        q = blocks[k]
        w = binomial_pmf(k, e, dtype=dt, device=dev)
        for j in range(k, -1, -1):
            acc[j] = acc[j] + q * w[j]
            if j:
                q = _descend(layers, q, j, chunk)
    return LossyDistribution(layers, acc)


def thin_distribution(layers, probs: torch.Tensor, eta: Eta, chunk: int = 1 << 24) -> LossyDistribution:
    """Output-side uniform loss on a distribution over the n-photon layer of a `FockLayers` /
    torch basis (what every SLOS kernel returns). probs: [S] or [B, S] over `layers.final_states`.

    Removing one photon uniformly at random from a k-photon pattern t+e_j lands on t with
    probability (t_j + 1) / k, which is exactly the raising-edge structure of the basis (the
    k-1 -> k edges grouped by raised mode, `edges_by_mode`: one edge per (child, mode) pair,
    weight = occupation of the mode in the child), so the k-1 layer is one scatter-add over
    those edges, mode slab by mode slab:
        q_{k-1}(t) = sum_j q_k(t + e_j) (t_j + 1) / k ,   q_n = probs,
    and the lossy distribution is P_loss(layer k) = C(n, k) eta^k (1 - eta)^(n-k) q_k. Every
    intermediate pattern of a mask-pruned basis is a sub-pattern of an admissible final state,
    so the pruned layers are complete for this recurrence. Differentiable in probs and eta.
    Nothing beyond the basis and one [S_k] vector per layer is materialised (slabs are walked
    in chunks of `chunk` edges).
    """
    n = layers.n
    e = _eta(eta)
    dev, dt = probs.device, probs.dtype
    q = [None] * (n + 1)
    q[n] = probs
    for k in range(n, 0, -1):
        q[k - 1] = _descend(layers, q[k], k, chunk)
    if e is None:
        weights = torch.zeros(n + 1, dtype=dt, device=dev)
        weights[n] = 1.0
    else:
        weights = binomial_pmf(n, e, dtype=dt, device=dev)
    return LossyDistribution(layers, [q[k] * weights[k] for k in range(n + 1)])


def thin_patterns(occupations, probs: torch.Tensor, eta: Eta) -> LossyDistribution:
    """Output-side uniform loss on an explicit pattern list: occupations [P, m], probs [P] or
    [B, P]. Same recurrence as `thin_distribution`, the sub-patterns being generated and merged
    layer by layer (numpy unique on packed keys). Patterns may have different photon numbers
    (a time-loop distribution with photons left in the loops): each photon-number class is
    thinned with its own binomial weights and the classes are merged per detected count. For
    the time-loop distribution and other engines without a FockLayers basis.
    """
    occ = np.asarray(occupations, dtype=np.int64)
    if occ.ndim != 2:
        raise ValueError("occupations must be [P, m]")
    e = _eta(eta)
    tot = occ.sum(axis=1)
    n_max = int(tot.max()) if occ.shape[0] else 0
    m = occ.shape[1]
    base = n_max + 1
    per_k: list[tuple[list[np.ndarray], list[torch.Tensor]]] = [([], []) for _ in range(n_max + 1)]
    for n in np.unique(tot).tolist():
        sel = np.nonzero(tot == n)[0]
        layers_occ, q = _thin_same_n(occ[sel], probs.index_select(-1, torch.as_tensor(sel, device=probs.device)), e, n)
        for k in range(n + 1):
            per_k[k][0].append(layers_occ[k]); per_k[k][1].append(q[k])
    out_occ, out_p = [], []
    for k in range(n_max + 1):
        o = np.concatenate(per_k[k][0], axis=0)
        p = torch.cat(per_k[k][1], dim=-1)
        keys = (o * (base ** np.arange(m, dtype=np.int64))).sum(axis=1)
        uk, inv = np.unique(keys, return_inverse=True)
        first = np.zeros(uk.shape[0], dtype=np.int64)
        first[inv[::-1]] = np.arange(o.shape[0])[::-1]
        merged = torch.zeros((*p.shape[:-1], uk.shape[0]), dtype=p.dtype, device=p.device)
        out_occ.append(o[first]); out_p.append(merged.index_add(-1, torch.as_tensor(inv, device=p.device), p))
    return LossyDistribution(None, out_p, out_occ)


def _thin_same_n(occ: np.ndarray, probs: torch.Tensor, e, n: int):
    """Photon-removal recurrence on patterns with exactly n photons: returns the per-layer
    occupations [k -> [P_k, m]] and the binomially weighted probabilities.
    """
    P, m = occ.shape
    dev, dt = probs.device, probs.dtype
    base = n + 1
    layers_occ = [occ]
    q = [probs]
    cur = occ
    for k in range(n, 0, -1):
        # children: every (pattern, mode with a photon) -> pattern - e_j, weight t_j / k
        nz = np.nonzero(cur)
        rows, modes = nz[0], nz[1]
        child = cur[rows].copy()
        child[np.arange(rows.shape[0]), modes] -= 1
        keys = (child * (base ** np.arange(m, dtype=np.int64))).sum(axis=1)
        uk, inv = np.unique(keys, return_inverse=True)
        first = np.zeros(uk.shape[0], dtype=np.int64)
        first[inv[::-1]] = np.arange(rows.shape[0])[::-1]
        child_occ = child[first]
        w = torch.as_tensor(cur[rows, modes] / k, dtype=dt, device=dev)
        contrib = q[-1].index_select(-1, torch.as_tensor(rows, device=dev)) * w
        out = torch.zeros((*q[-1].shape[:-1], uk.shape[0]), dtype=dt, device=dev)
        q.append(out.index_add(-1, torch.as_tensor(inv, device=dev), contrib))
        layers_occ.append(child_occ)
        cur = child_occ
    q = q[::-1]
    layers_occ = layers_occ[::-1]
    if e is None:
        weights = torch.zeros(n + 1, dtype=dt, device=dev)
        weights[n] = 1.0
    else:
        weights = binomial_pmf(n, e, dtype=dt, device=dev)
    return layers_occ, [q[k] * weights[k] for k in range(n + 1)]


def thin_marginals(marg: torch.Tensor, eta: Eta) -> torch.Tensor:
    """Output-side loss on per-mode photon-number marginals [..., N+1]: uniform thinning acts on
    each mode's count independently, P'(j) = sum_i P(i) C(i, j) eta^j (1-eta)^(i-j).
    """
    e = _eta(eta)
    if e is None:
        return marg
    N = marg.shape[-1] - 1
    et = torch.as_tensor(e, dtype=marg.dtype, device=marg.device)
    i = torch.arange(N + 1, dtype=marg.dtype, device=marg.device)
    j = i.clone()
    logc = torch.lgamma(i[:, None] + 1) - torch.lgamma(j[None, :] + 1) - torch.lgamma((i[:, None] - j[None, :]).clamp_min(0) + 1)
    T = torch.exp(logc) * et ** j[None, :] * (1 - et) ** (i[:, None] - j[None, :]).clamp_min(0)
    T = torch.where(j[None, :] <= i[:, None], T, torch.zeros_like(T))               # [i, j]
    return marg @ T


def thin_to_clicks(occupations, probs: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """Group a pattern distribution by its click pattern (threshold detectors)."""
    occ = np.asarray(occupations)
    cl = (occ > 0).astype(np.int64)
    keys = (cl * (2 ** np.arange(occ.shape[1], dtype=np.int64))).sum(axis=1)
    uk, inv = np.unique(keys, return_inverse=True)
    first = np.zeros(uk.shape[0], dtype=np.int64)
    first[inv[::-1]] = np.arange(occ.shape[0])[::-1]
    out = torch.zeros((*probs.shape[:-1], uk.shape[0]), dtype=probs.dtype, device=probs.device)
    return cl[first], out.index_add(-1, torch.as_tensor(inv, device=probs.device), probs)


__all__ = [
    "Distinguishability",
    "LossyDistribution",
    "add_dark_counts",
    "binomial_pmf",
    "clicks",
    "dark_distribution",
    "dark_predecessors",
    "spec",
    "split_shots",
    "thin_counts",
    "thin_distribution",
    "thin_marginals",
    "thin_patterns",
    "thin_to_clicks",
]


# ==== partial distinguishability ================================================================

LOST, SINGLE = 0, 1                  # roles in `draw_roles`; species mu is role 2 + mu


@dataclass(frozen=True)
class Distinguishability:
    """Either the x-model (`x`: pairwise overlap, V_HOM = x^2) or random species (`species`:
    the probabilities q_mu of the interfering species, sum <= 1; the remainder is the
    probability of a photon being in a mode of its own). `Distinguishability(x=1)` is the
    ideal case. Construct from a HOM visibility with `from_hom_visibility`.
    """
    x: float | None = None
    species: tuple[float, ...] | None = None

    def __post_init__(self):
        if (self.x is None) == (self.species is None):
            raise ValueError("give exactly one of x (x-model) or species (random species probabilities)")
        if self.x is not None:
            if not 0.0 <= float(self.x) <= 1.0:
                raise ValueError(f"x must be in [0, 1], got {self.x}")
            object.__setattr__(self, "x", float(self.x))
        else:
            q = tuple(float(v) for v in self.species)
            if any(v < 0 for v in q) or sum(q) > 1.0 + 1e-12:
                raise ValueError("species probabilities must be non-negative and sum to at most 1")
            object.__setattr__(self, "species", q)

    @staticmethod
    def from_hom_visibility(V: float) -> Distinguishability:
        """x-model with pairwise overlap x = sqrt(V): V is the two-photon HOM dip visibility."""
        if not 0.0 <= float(V) <= 1.0:
            raise ValueError(f"visibility must be in [0, 1], got {V}")
        return Distinguishability(x=math.sqrt(float(V)))

    @property
    def probs(self) -> tuple[float, ...]:
        """Probabilities of the interfering species: (x,) in the x-model, else `species`."""
        return (self.x,) if self.x is not None else self.species

    @property
    def is_ideal(self) -> bool:
        """True when every photon is in one common species (x = 1, or one species of probability 1)."""
        return sum(self.probs) >= 1.0 and len([q for q in self.probs if q > 0]) == 1


Indist = float | Distinguishability | None


def spec(indist: Indist) -> Distinguishability | None:
    """Normalise the `indist=` argument of the engines: None / 1.0 / an ideal spec -> None (the
    lossless path, object for object); a float is the x-model; a `Distinguishability` is kept.
    A sequence of per-photon overlaps (the emitter strong side and `SequenceKernel`) is only
    checked here and returned as a non-ideal marker; those engines read the sequence themselves.

    Public: code built on the engines calls it to accept the same `indist=` argument with the same
    validation (raises ValueError on overlaps outside [0, 1]).
    """
    if indist is None:
        return None
    if isinstance(indist, (list, tuple)) or (isinstance(indist, torch.Tensor) and indist.dim() == 1):
        xs = [float(v) for v in indist]                                           # one overlap per photon / emitter
        if any(not 0.0 <= v <= 1.0 for v in xs):
            raise ValueError("per-photon overlaps must be in [0, 1]")
        return None if all(v >= 1.0 for v in xs) else Distinguishability(x=min(xs))   # a non-ideal marker: the engines read the sequence
    d = indist if isinstance(indist, Distinguishability) else Distinguishability(
        x=float(indist.detach()) if isinstance(indist, torch.Tensor) else float(indist))
    return None if d.is_ideal else d


def draw_roles(n: int, shots: int, eta, indist: Indist, generator: torch.Generator | None = None) -> torch.Tensor:
    """Roles [shots, n] (cpu long) of every input photon of every stream: LOST (0), SINGLE (1, a
    species of its own) or 2 + mu (interfering species mu; the x-model has one species, mu = 0).
    Each photon is lost with probability 1 - eta, otherwise draws its species independently.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    e = _eta(eta)
    d = spec(indist)
    roles = torch.full((shots, n), 2, dtype=torch.long)
    if n == 0:
        return roles
    if d is not None:
        q = torch.tensor([*d.probs, max(0.0, 1.0 - sum(d.probs))], dtype=torch.float64)
        mu = torch.multinomial(q, shots * n, replacement=True, generator=generator).view(shots, n)
        roles = torch.where(mu == len(d.probs), torch.full_like(mu, SINGLE), mu + 2)
    if e is not None:
        lost = torch.rand((shots, n), generator=generator) >= float(e)
        roles[lost] = LOST
    return roles


def partition(n: int, shots: int, eta, indist: Indist, generator: torch.Generator | None = None,
              streams: torch.Tensor | None = None) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], torch.Tensor]]:
    """Draw the roles and group the streams by identical partition. Returns
    [(species_subsets, singles, stream_ids)]: `species_subsets` one sorted tuple of photon
    indices per interfering species (empty species dropped), `singles` the photons in a mode of
    their own, `stream_ids` (cpu long) the streams that drew this partition; lost photons appear
    in neither. Superset of `loss.split_shots` (whose groups are the case indist=None).
    """
    ids = torch.arange(shots) if streams is None else torch.as_tensor(streams).reshape(-1).cpu()
    roles = draw_roles(n, shots, eta, indist, generator)
    uniq, inv = torch.unique(roles, dim=0, return_inverse=True)
    order = torch.argsort(inv, stable=True)
    counts = torch.bincount(inv, minlength=uniq.shape[0])
    groups, start = [], 0
    for g, c in enumerate(counts.tolist()):
        row = uniq[g].tolist()
        n_species = max([r - 2 for r in row if r >= 2] + [-1]) + 1
        subsets = tuple(tuple(j for j in range(n) if row[j] == 2 + mu) for mu in range(n_species))
        subsets = tuple(s for s in subsets if s)
        singles = tuple(j for j in range(n) if row[j] == SINGLE)
        groups.append((subsets, singles, ids[order[start:start + c]]))
        start += c
    return groups


def route_classical(U: torch.Tensor, circuit: torch.Tensor, cols: torch.Tensor,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Output rows [P] of P photons that interfere with nothing: photon p enters column cols[p]
    of circuit U[circuit[p]] (U [B, m, m] or [m, m]) and lands in row r with probability
    |U[r, col]|^2. Drawn on the host.
    """
    if U.dim() == 2:
        U = U[None]
    cols = torch.as_tensor(cols, dtype=torch.long)
    circuit = torch.as_tensor(circuit, dtype=torch.long)
    if cols.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    p = (U[circuit.to(U.device), :, cols.to(U.device)].abs() ** 2).to("cpu", torch.float64)   # [P, m]
    p = p / p.sum(1, keepdim=True)
    return torch.multinomial(p, 1, generator=generator)[:, 0]


def subset_weights(n: int, indist: Indist) -> Sequence[tuple[tuple[int, ...], float]]:
    """x-model, strong side: every subset G of the n photons with its weight
    x^|G| (1 - x)^(n - |G|) — the mixture over which photons interfere (the others are
    singletons). 2^n terms; the strong engines truncate it by the number of singletons with the
    binomial-tail bound (docs/DISTINGUISHABILITY.md §4). Test helper for the samplers too.
    """
    from itertools import combinations
    d = spec(indist)
    if d is None:
        return [(tuple(range(n)), 1.0)]
    if d.x is None:
        raise NotImplementedError("subset_weights: x-model only (random species need a partition sum)")
    return [(G, d.x ** g * (1.0 - d.x) ** (n - g)) for g in range(n + 1) for G in combinations(range(n), g)]


def truncation_bound(n: int, indist: Indist, order: int | None) -> float:
    """Total variation distance bound of the strong-side truncation `order=b`: the mass of the
    partitions with more than b distinguishable photons, P(Bin(n, 1 - x) > b) — exact and
    independent of the interferometer. 0 when nothing is truncated.
    """
    d = spec(indist)
    if d is None or order is None or order >= n:
        return 0.0
    if d.x is None:
        raise NotImplementedError("truncation_bound: x-model only")
    p = 1.0 - d.x
    return float(sum(math.comb(n, b) * p ** b * (1 - p) ** (n - b) for b in range(order + 1, n + 1)))


# ==== loss on Gaussian (squeezed) inputs =========================================================
# docs/GAUSSIAN.md §4. On a Gaussian state uniform loss is the channel on the quadrature covariance
#     V -> eta V + (1 - eta) V_vac ,   mean -> sqrt(eta) mean ,
# O(m^2), exact, and the state stays Gaussian - no per-shot subset draw as for Fock inputs. It
# commutes with a passive U, so it is applied per source at the input. A lossy squeezed vacuum is
# mixed; the chain rule needs a pure state, so it is written as a PURE squeezed vacuum at a smaller
# squeezing plus a classical random displacement of the anti-squeezed quadrature:
#     Var x = (eta e^-2r + 1 - eta)/4 =: e^-2r'/4          (pure part, r' <= r)
#     Var p = (eta e^+2r + 1 - eta)/4  = e^+2r'/4 + w       (w >= 0 by Jensen: classical noise)
# in units where the vacuum has Var x = Var p = 1/4 (x = (a + a^dag)/2).

def lossy_squeezing(r, eta) -> tuple[np.ndarray, np.ndarray]:
    """Pure-plus-classical decomposition of squeezed vacua behind a transmission eta.

    r [sources] -> (r_pure [sources], var_p [sources]): the lossy state equals a pure squeezed vacuum
    at r_pure displaced by i*y, y ~ N(0, var_p) real, independently per source (p-quadrature units
    with vacuum variance 1/4). eta = None or 1 returns (r, 0) exactly.
    """
    r = np.asarray(r, dtype=np.float64)
    e = _eta(eta)
    if e is None:
        return r.copy(), np.zeros_like(r)
    e = float(e)
    vx = e * np.exp(-2 * r) + 1 - e
    vp = e * np.exp(2 * r) + 1 - e
    r_pure = -0.5 * np.log(vx)
    w = np.maximum((vp - np.exp(2 * r_pure)) / 4.0, 0.0)
    return r_pure, w


def gaussian_loss(cov: np.ndarray, mean: np.ndarray, eta) -> tuple[np.ndarray, np.ndarray]:
    """The uniform-loss channel on a quadrature covariance [2m, 2m] and mean [2m] (vacuum
    covariance I/4, the convention of `lossy_squeezing`): V -> eta V + (1 - eta) I/4,
    mean -> sqrt(eta) mean. eta = None or 1 returns the inputs unchanged (same objects).
    """
    e = _eta(eta)
    if e is None:
        return cov, mean
    e = float(e)
    return e * cov + (1 - e) * np.eye(cov.shape[0]) / 4.0, math.sqrt(e) * mean


# ==== dark counts =================================================================================
# Module docstring, DARK COUNTS. Output side, after loss and detection, before the threshold clip.

Dark = float | Sequence[float] | torch.Tensor


def _dark(dark, m: int) -> torch.Tensor | None:
    """Per-mode dark-count means [m] (float64, a tensor dark keeps its graph), or None when there
    are none (None, 0, all zeros; a tensor that requires grad is kept even at 0).
    """
    if dark is None:
        return None
    if isinstance(dark, torch.Tensor):
        d = dark if dark.is_floating_point() else dark.to(torch.float64)
        if d.dim() > 1 or (d.dim() == 1 and d.shape[0] != m):
            raise ValueError(f"dark must be a scalar or one value per mode ({m})")
        d = d.expand(m)
        dv = d.detach()
        if not bool(torch.isfinite(dv).all()) or bool((dv < 0).any()):
            raise ValueError("dark counts must be finite and >= 0")
        # a differentiable dark is kept even at 0: the gradient at dark = 0 is needed (do not short-cut it)
        return d if d.requires_grad or bool((dv > 0).any()) else None
    a = np.asarray(dark, dtype=np.float64)
    if a.ndim > 1 or (a.ndim == 1 and a.shape[0] != m):
        raise ValueError(f"dark must be a scalar or one value per mode ({m})")
    a = np.broadcast_to(a, (m,))
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("dark counts must be finite and >= 0")
    return torch.as_tensor(a.copy()) if np.any(a > 0) else None


def add_dark_counts(counts: torch.Tensor, dark: Dark, generator: torch.Generator | None = None,
                    threshold: bool = False) -> torch.Tensor:
    """Dark counts on samples: counts [..., m] (int) + n, n_i ~ Poisson(dark_i) drawn independently
    for every sample and mode (on the CPU with `generator`, like `thin_counts`), then clipped to
    clicks when `threshold`. Exact: sampling is never truncated. dark = 0 draws nothing and returns
    `counts` itself (its clicks when `threshold`).
    """
    d = _dark(dark, counts.shape[-1])
    if d is None:
        return clicks(counts) if threshold else counts
    lam = d.detach().to("cpu", torch.float64).expand(counts.shape).contiguous()
    noisy = counts + torch.poisson(lam, generator=generator).to(counts.device, counts.dtype)
    return clicks(noisy) if threshold else noisy


def _merge_rows(rows: np.ndarray, p: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """Sum the probabilities p [..., S] of identical rows [S, c]; rows come back sorted."""
    uk, first, inv = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    out = torch.zeros((*p.shape[:-1], uk.shape[0]), dtype=p.dtype, device=p.device)
    return rows[first], out.index_add(-1, torch.as_tensor(inv.reshape(-1), device=p.device), p)


def dark_distribution(occupations, probs: torch.Tensor, dark: Dark, max_dark: int | None = None,
                      threshold: bool = False) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Dark counts on a distribution: occupations [P, m] (the true patterns), probs [P] or [..., P]
    -> (observed patterns [P', m], probs [..., P'], dropped [...]), the law convolved with
    T(o' | o) = prod_i t_i(o'_i | o_i), mode by mode (one expansion and one merge per mode with
    dark_i > 0). Differentiable in probs and in a tensor `dark`.

    PNR: max_dark (required) is the most dark counts kept in one pattern, summed over the modes;
    `dropped` = P(Poisson(sum_i dark_i) > max_dark) times the input mass, returned and never
    renormalised away (probs' sums to sum(probs) - dropped). threshold: the true patterns are
    clipped to clicks first; max_dark = None is exact (finite support), an integer caps the
    visible dark clicks (on modes without a true click), dropped is then the mass of the rest.
    dark = 0: the inputs unchanged (clicks merged when `threshold` and the patterns are not
    clicks yet), dropped = 0.
    """
    occ = np.asarray(occupations, dtype=np.int64)
    if occ.ndim != 2:
        raise ValueError("occupations must be [P, m]")
    m = occ.shape[1]
    d = _dark(dark, m)
    if max_dark is not None and int(max_dark) < 0:
        raise ValueError("max_dark must be >= 0")
    if threshold and occ.size and occ.max() > 1:
        occ, probs = thin_to_clicks(occ, probs)
    dropped = torch.zeros(probs.shape[:-1], dtype=probs.dtype, device=probs.device)
    if d is None:
        return occ, probs, dropped
    if max_dark is None and not threshold:
        raise ValueError("PNR dark counts have an infinite support: give max_dark (dark counts kept per pattern)")
    d = d.to(probs.device, probs.dtype)
    K = None if max_dark is None else int(max_dark)
    used = np.zeros((occ.shape[0], 1), dtype=np.int64)                                  # dark counts kept so far
    p = probs
    for i in range(m):
        di = d[i]
        if not di.requires_grad and float(di) == 0.0:
            continue
        if threshold:
            q0 = torch.exp(-di)                                                         # no dark click
            on = occ[:, i] == 1
            move = ~on if K is None else (~on & (used[:, 0] < K))
            drop = ~on & ~move
            stay = torch.where(torch.as_tensor(on, device=p.device), p, p * q0)        # a dark click on a click is invisible
            o2 = occ[move].copy(); o2[:, i] = 1
            sel = torch.as_tensor(np.nonzero(move)[0], device=p.device)
            parts = [(occ, used, stay), (o2, used[move] + 1, p.index_select(-1, sel) * -torch.expm1(-di))]
            if drop.any():
                dropped = dropped + p.index_select(-1, torch.as_tensor(np.nonzero(drop)[0], device=p.device)).sum(-1) * -torch.expm1(-di)
        else:
            parts = []
            for j in range(K + 1):
                ok = used[:, 0] + j <= K
                if not ok.any():
                    break
                o2 = occ[ok].copy(); o2[:, i] += j
                w = torch.exp(-di) * di ** j / math.factorial(j)
                parts.append((o2, used[ok] + j, p.index_select(-1, torch.as_tensor(np.nonzero(ok)[0], device=p.device)) * w))
            left = torch.as_tensor((K - used[:, 0] + 1).astype(np.float64), dtype=p.dtype, device=p.device)
            dropped = dropped + (p * torch.special.gammainc(left, di.expand_as(left))).sum(-1)   # P(n_i > budget left)
        rows, pp = _merge_rows(np.concatenate([np.concatenate([o, u], axis=1) for o, u, _ in parts]),
                               torch.cat([q for _, _, q in parts], dim=-1))
        occ, used, p = rows[:, :m], rows[:, m:], pp
    occ, p = _merge_rows(occ, p)
    return occ, p, dropped


def dark_predecessors(observed, dark: Dark, max_dark: int | None = None,
                      threshold: bool = False) -> list[tuple[tuple[int, ...], float]]:
    """The true patterns that can produce the `observed` one, with T(observed | true): PNR, every
    o <= o' with sum(o' - o) <= max_dark (None: no cap, the list is finite); threshold (observed
    clicks), every true click pattern c subset of c' with at most max_dark dark clicks. Patterns of
    weight 0 (extra counts on a mode without dark counts) are left out. This is what conditioning
    on a noisy outcome needs: rho | o' = sum_o T(o' | o) K_o rho K_o^dag / p(o'). Consistent with
    `dark_distribution` for the same max_dark. dark = 0: [(observed, 1.0)].
    """
    o = np.asarray([int(v) for v in observed], dtype=np.int64)
    if o.ndim != 1 or np.any(o < 0) or (threshold and np.any(o > 1)):
        raise ValueError("observed must be one pattern of counts >= 0 (clicks in {0, 1} under threshold)")
    d = _dark(dark, o.shape[0])
    if d is None:
        return [(tuple(int(v) for v in o), 1.0)]
    dd = d.detach().cpu().numpy().astype(np.float64)
    K = math.inf if max_dark is None else int(max_dark)
    out = [((), 1.0, 0)]                                                                # (prefix, weight, dark counts)
    for oi, di in zip(o.tolist(), dd.tolist()):
        if threshold:
            opts = [(0, math.exp(-di), 0)] if oi == 0 else [(1, 1.0, 0)] + ([(0, -math.expm1(-di), 1)] if di > 0 else [])
        else:
            opts = [(oi, math.exp(-di), 0)] + ([(oi - j, math.exp(-di) * di ** j / math.factorial(j), j)
                                                 for j in range(1, oi + 1)] if di > 0 else [])
        out = [((*pre, v), w * wv, n + nv) for pre, w, n in out for v, wv, nv in opts if n + nv <= K]
    return [(pre, w) for pre, w, _ in out]
