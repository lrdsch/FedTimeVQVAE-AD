#!/usr/bin/env python3
"""FLOOR heads — closed-form / zero-parameter detectors in pure numpy.

ZERO imports from the repo: this module is standalone-testable and carries no
config, no torch, no dataset knowledge. `scripts/floor_eval.py` is what glues it
to the real `detect` path.

Each head is a triple `(stats, solve, score)` (documentation/research/FLOOR_BASELINE.md §1.1):

  stats(series, W, fit_stride) -> dict | None
      ADDITIVE sufficient statistic (elementwise-summable across clients), or
      None for a head that fits nothing.
  solve(stats, knobs) -> theta
      closed form. No optimiser anywhere.
  score_series(x, theta) -> (T,)          when granularity == "series"
  score_windows(M, theta) -> (m, W)       when granularity == "window"

`granularity` is explicit because it changes the accumulation geometry: a
series-causal head is computed once per series and sliced per window, a
window-native head (PCA / Gauss) is recomputed per window and cannot be
expressed with slice granularity at all.

The federation proposition (§1.3): for every additive head the pooled statistic
is the elementwise SUM of the per-client statistics, because the pooled design
matrix is the vertical concatenation of the per-client rows. Hence
`solve(Σ_k stats_k) == solve(pooled)` in exact arithmetic — `pooled_solve()`
below builds the pooled object by an INDEPENDENT path (materialised design
matrix + lstsq / np.cov) precisely so the equality can be witnessed
non-circularly.

Self-test:  python scripts/floor_heads.py
"""
from __future__ import annotations

import zlib

import numpy as np

# Rows processed per BLAS call when accumulating statistics. Bounds peak memory
# at chunk*W*8 bytes (~20 MB at W=128) regardless of series length — ucr_split
# has series with >500k stride-1 windows.
CHUNK = 20_000

DEFAULTS = {
    "k": 10,            # ma_c / ma_causal window, PRE-REGISTERED
    "ar_p": 32,         # AR lag order
    "ar_lambda": 1e-4,  # ridge, scaled by N inside solve -> length-invariant
    "pca_k": 8,         # retained components
    "gamma": 0.05,      # covariance shrinkage -> always PD, even at n < W
}


# ═════════════════════════════════════════════════════════════════════════════
#   Geometry — must agree with data.SlidingWindowDataset (asserted in the
#   floor_eval selftest, which is the only place that may import the repo).
# ═════════════════════════════════════════════════════════════════════════════

def window_starts(T: int, W: int, stride: int) -> np.ndarray:
    """Start index of every window, or empty when the series is shorter than W
    (`data.py:369-370` drops those records)."""
    if T < W:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, T - W + 1, stride, dtype=np.int64)


def _windows(x: np.ndarray, W: int, starts: np.ndarray) -> np.ndarray:
    """(len(starts), W) matrix of windows. Copy — callers chunk it."""
    view = np.lib.stride_tricks.sliding_window_view(x, W)
    return np.ascontiguousarray(view[starts])


def moving_average_paper(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-clipped kernel — paper Algorithm 1.

    Re-implemented here (instead of imported) so this module stays repo-free;
    `floor_eval.selftest` asserts it is BIT-IDENTICAL to
    `detect._moving_average_paper`, which is the definition of record.
    """
    n = x.shape[0]
    if window <= 1 or n <= 1:
        return x.astype(np.float64, copy=True)
    cs = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
    half = window // 2
    j = np.arange(n)
    lo = np.maximum(0, j - half)
    hi = np.minimum(n, j + half)
    cnt = np.maximum(1, hi - lo)
    return (cs[hi] - cs[lo]) / cnt


# ═════════════════════════════════════════════════════════════════════════════
#   Base
# ═════════════════════════════════════════════════════════════════════════════

class Head:
    name = "?"
    granularity = "series"       # "series" | "window"
    additive = False             # has for additive purposes sufficient statistic
    zero_param = False           # fits nothing -> arm-invariant by construction
    supports_naive = False       # has a raw basis whose alignment can be broken

    # ── statistics ──────────────────────────────────────────────────────────
    def stats(self, series: list[np.ndarray], W: int, fit_stride: int):
        return None

    @staticmethod
    def merge(stats_list: list[dict]) -> dict:
        """Elementwise sum — the server-side aggregation of §1.3."""
        out = {}
        for key in stats_list[0]:
            out[key] = sum(s[key] for s in stats_list)
        return out

    # ── closed form ─────────────────────────────────────────────────────────
    def solve(self, stats, knobs: dict):
        return None

    def pooled_solve(self, series_per_client: list[list[np.ndarray]], W: int,
                     fit_stride: int, knobs: dict) -> tuple[object, str]:
        """INDEPENDENT pooled path: materialise the pooled design matrix and
        solve on it with a different numerical routine than `solve` uses.
        Returns (theta, path_tag). The whole point of §0.4's non-circularity
        gate — never route this through `merge`."""
        raise NotImplementedError

    # parameters (for witness / havingaging)
    def param(self, theta) -> np.ndarray:
        """Canonical flat parameter vector. For subspace heads this is the
        PROJECTOR, not the basis: invariant to sign and to degeneracies."""
        return np.zeros(0)

    def average(self, thetas: list, weights: np.ndarray, mode: str = "fedavg"):
        """`fedavg` = weighted mean of the canonical parameter.
        `naive`   = mean of the RAW basis with no sign/permutation alignment
                    (the deliberate positive control)."""
        raise NotImplementedError

    # ── scoring ─────────────────────────────────────────────────────────────
    def score_series(self, x: np.ndarray, theta) -> np.ndarray:
        raise NotImplementedError

    def score_windows(self, M: np.ndarray, theta) -> np.ndarray:
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════════
#   Zero-parameter heads — local == centralized == every federated mode
# ═════════════════════════════════════════════════════════════════════════════

class MaCentered(Head):
    """H1 — THE calibration baseline. s_t = (x_t - MA_k(x)_t)^2.

    The deep detector is a NON-CAUSAL window reconstruction, so the centred
    filter is the fair comparator; `ma_causal` is the handicapped variant."""
    name, zero_param = "ma_c", True

    def score_series(self, x, theta):
        k = int(theta["k"])
        return (x - moving_average_paper(x, k)) ** 2


class MaCausal(Head):
    """H2 — x̂_t = mean(x_{t-k} … x_{t-1}). Causal information set."""
    name, zero_param = "ma_causal", True

    def score_series(self, x, theta):
        k = int(theta["k"])
        cs = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
        t = np.arange(len(x))
        lo = np.maximum(0, t - k)
        cnt = np.maximum(1, t - lo)
        pred = (cs[t] - cs[lo]) / cnt
        pred[0] = x[0]                      # no history at t=0 -> residual 0
        return (x - pred) ** 2


class Diff1(Head):
    """First difference squared. At k=3 `ma_c` degenerates EXACTLY into
    0.25*diff1 (the edge-clipped kernel averages 2 points), and every metric is
    invariant to a positive rescaling — so the two rows must coincide."""
    name, zero_param = "diff1", True

    def score_series(self, x, theta):
        s = np.empty_like(x)
        s[1:] = (x[1:] - x[:-1]) ** 2
        s[0] = s[1] if len(x) > 1 else 0.0
        return s


class RandomHead(Head):
    """Bottom anchor. Seeded by a STABLE hash (crc32 of the key bytes): Python's
    builtin hash() is randomised per process (PYTHONHASHSEED), which silently
    made the old implementation irreproducible across runs."""
    name, zero_param = "random", True

    def score_series(self, x, theta):
        seed = zlib.crc32(f"floor_random|{len(x)}|{int(theta.get('seed', 0))}".encode())
        return np.random.default_rng(seed).random(len(x))


# ═════════════════════════════════════════════════════════════════════════════
#   H3 — ridge AR(p), series-causal, additive
# ═════════════════════════════════════════════════════════════════════════════

class AR(Head):
    """φ_t = [x_{t-1} … x_{t-p}],  t = p … T-1

        n = T-p,  G = Σ φφᵀ,  b = Σ φ x_t,  q = Σ x_t²
        w = (G + λ·N·I)⁻¹ b
        σ̂² = max((q - 2wᵀb + wᵀGw)/N, 1e-8)          # closed form, one pass
        s_t = (x_t - φ_tᵀw)² / σ̂²   for t ≥ p,   0 otherwise

    λ scaled by N makes the penalty invariant to series length and cohort size.
    σ̂² is METRIC-NEUTRAL (a positive per-entity constant, and impulse + channel
    aggregation + train-quantile threshold are all positively homogeneous), which
    is what `fed_scaleonly` exploits as a leak detector."""
    name, additive, granularity = "ar", True, "series"

    def __init__(self, p: int):
        self.p = int(p)

    # rows of the lag-design matrix for one series
    def _design(self, x: np.ndarray):
        p, T = self.p, len(x)
        if T <= p:
            return None
        view = np.lib.stride_tricks.sliding_window_view(x, p)   # rows x[i:i+p]
        return view, np.arange(0, T - p, dtype=np.int64)        # i = t-p

    def stats(self, series, W, fit_stride):
        p = self.p
        G = np.zeros((p, p)); b = np.zeros(p); q = 0.0; n = 0
        for x in series:
            d = self._design(x)
            if d is None:
                continue
            view, idx = d
            for s in range(0, len(idx), CHUNK):
                sl = idx[s: s + CHUNK]
                Phi = np.ascontiguousarray(view[sl][:, ::-1])   # [x_{t-1}..x_{t-p}]
                y = x[p + sl]
                G += Phi.T @ Phi
                b += Phi.T @ y
                q += float(y @ y)
                n += len(y)
        return {"G": G, "b": b, "q": q, "n": n}

    def solve(self, stats, knobs):
        p, N = self.p, stats["n"]
        if N <= 0:
            return {"w": np.zeros(p), "var": 1.0, "n": 0, "rank_deficient": True}
        A = stats["G"] + knobs["ar_lambda"] * N * np.eye(p)
        w = np.linalg.solve(A, stats["b"])
        var = max((stats["q"] - 2.0 * w @ stats["b"] + w @ stats["G"] @ w) / N, 1e-8)
        return {"w": w, "var": float(var), "n": int(N),
                "rank_deficient": bool(np.linalg.matrix_rank(stats["G"]) < p)}

    def pooled_solve(self, series_per_client, W, fit_stride, knobs):
        """Augmented least squares on the MATERIALISED pooled design matrix —
        a different solver (QR via lstsq) on a different object than `solve`'s
        normal equations. Falls back to a chunked pass over the pooled stream
        (still never a sum of per-client stats) when the matrix is too big."""
        p = self.p
        rows = [x for series in series_per_client for x in series]
        n_tot = sum(max(0, len(x) - p) for x in rows)
        if n_tot * p * 8 > 1_500_000_000:                       # ~1.5 GB guard
            return self._pooled_chunked(rows, knobs), "chunked_pooled"
        Z = np.empty((n_tot, p)); y = np.empty(n_tot); at = 0
        for x in rows:
            d = self._design(x)
            if d is None:
                continue
            view, idx = d
            m = len(idx)
            Z[at: at + m] = view[idx][:, ::-1]
            y[at: at + m] = x[p + idx]
            at += m
        Z, y = Z[:at], y[:at]
        if at == 0:
            return {"w": np.zeros(p), "var": 1.0, "n": 0, "rank_deficient": True}
        lam = np.sqrt(knobs["ar_lambda"] * at)
        Za = np.vstack([Z, lam * np.eye(p)])
        ya = np.concatenate([y, np.zeros(p)])
        w = np.linalg.lstsq(Za, ya, rcond=None)[0]
        r = y - Z @ w
        return {"w": w, "var": float(max((r @ r) / at, 1e-8)), "n": int(at),
                "rank_deficient": bool(np.linalg.matrix_rank(Z) < p)}, "independent"

    def _pooled_chunked(self, rows, knobs):
        st = self.stats(rows, 0, 1)
        return self.solve(st, knobs)

    def param(self, theta):
        return np.asarray(theta["w"], dtype=np.float64)

    def average(self, thetas, weights, mode="fedavg"):
        a = weights / weights.sum()
        w = sum(ai * t["w"] for ai, t in zip(a, thetas))
        var = float(sum(ai * t["var"] for ai, t in zip(a, thetas)))
        return {"w": w, "var": var, "n": int(sum(t["n"] for t in thetas)),
                "rank_deficient": any(t.get("rank_deficient") for t in thetas)}

    def score_series(self, x, theta):
        p, w = self.p, theta["w"]
        s = np.zeros(len(x), dtype=np.float64)
        d = self._design(x)
        if d is None:
            return s
        view, idx = d
        for st in range(0, len(idx), CHUNK):
            sl = idx[st: st + CHUNK]
            Phi = np.ascontiguousarray(view[sl][:, ::-1])
            s[p + sl] = (x[p + sl] - Phi @ w) ** 2 / theta["var"]
        return s

    # closed-form excess objective, §1.4 — a number for the aggregation loss
    # even when the metric delta is null
    def excess_objective(self, theta_hat, theta_star, stats, knobs):
        d = theta_hat["w"] - theta_star["w"]
        A = stats["G"] + knobs["ar_lambda"] * stats["n"] * np.eye(self.p)
        return float(d @ A @ d)

    def objective_star(self, theta_star, stats, knobs):
        """J(w*) — the pooled objective AT its optimum, i.e. the reference the excess is an
        excess OF.

        Without it `excess_objective` is an unnormalised sum of squares and the reader has no
        way to size it. It is not a cosmetic issue: on wsd_fed the raw excesses are
        113/46/35/14 on c0..c3 while J(w*) is 1813/4868/5811/405, so relatively they are
        6.3%/0.9%/0.6%/3.5% and c3 — the SMALLEST raw number — is the second WORST cluster.
        Any ordering read off the raw column is wrong.

        With A = G + λ n I and w* = A⁻¹b, J(w) = q − 2w·b + w·A·w, so J(w*) = q − w*·b."""
        A = stats["G"] + knobs["ar_lambda"] * stats["n"] * np.eye(self.p)
        w = theta_star["w"]
        return float(stats["q"] - 2.0 * w @ stats["b"] + w @ A @ w)


# ═════════════════════════════════════════════════════════════════════════════
#   Window-moment heads — PCA(K) and the Gaussian whitener
# ═════════════════════════════════════════════════════════════════════════════

class _WindowMoments(Head):
    """Shared additive statistic: n, s = Σ w_i, S = Σ w_i w_iᵀ (§1.2 H4)."""
    additive, granularity = True, "window"

    def stats(self, series, W, fit_stride):
        n = 0; s = np.zeros(W); S = np.zeros((W, W))
        for x in series:
            starts = window_starts(len(x), W, fit_stride)
            for i in range(0, len(starts), CHUNK):
                M = _windows(x, W, starts[i: i + CHUNK])
                n += M.shape[0]
                s += M.sum(axis=0)
                S += M.T @ M
        return {"n": n, "s": s, "S": S}

    @staticmethod
    def _cov(stats, gamma, W):
        n = max(stats["n"], 1)
        mu = stats["s"] / n
        C = stats["S"] / n - np.outer(mu, mu)
        C = 0.5 * (C + C.T)                                   # kill float asymmetry
        Cg = (1.0 - gamma) * C + gamma * (np.trace(C) / W) * np.eye(W)
        return mu, Cg

    def _pooled_matrix(self, series_per_client, W, fit_stride):
        blocks, tot = [], 0
        for series in series_per_client:
            for x in series:
                starts = window_starts(len(x), W, fit_stride)
                if len(starts):
                    blocks.append((x, starts))
                    tot += len(starts)
        return blocks, tot


class PCA(_WindowMoments):
    """H4 — reconstruction residual of the top-K window subspace.

        μ = s/n,  Σ = S/n - μμᵀ,  Σ_γ = (1-γ)Σ + γ(trΣ/W)I
        Σ_γ = U diag(λ) Uᵀ  →  U_K  →  P = U_K U_Kᵀ
        r_i = (I - P)(w_i - μ),  per-timestep contribution = r_i ⊙ r_i

    The canonical parameter is P, NOT U_K: invariant to eigenvector sign and to
    eigenvalue degeneracies, which is what makes both the witness and the
    average well-defined."""
    name, supports_naive = "pca", True

    def __init__(self, K: int):
        self.K = int(K)

    def _from_cov(self, mu, Cg, W):
        evals, U = np.linalg.eigh(Cg)                          # ascending
        UK = U[:, -self.K:]
        return {"mu": mu, "P": UK @ UK.T, "U": UK, "W": W}

    def solve(self, stats, knobs):
        W = len(stats["s"])
        mu, Cg = self._cov(stats, knobs["gamma"], W)
        th = self._from_cov(mu, Cg, W)
        th["n"] = int(stats["n"])
        th["rank_deficient"] = bool(stats["n"] < W)
        return th

    def pooled_solve(self, series_per_client, W, fit_stride, knobs):
        """np.cov on the materialised pooled window matrix — an independent
        route to (μ, Σ) that never touches the per-client statistics."""
        blocks, tot = self._pooled_matrix(series_per_client, W, fit_stride)
        if tot == 0:
            raise ValueError("pooled window matrix is empty")
        if tot * W * 8 <= 1_500_000_000:
            M = np.empty((tot, W)); at = 0
            for x, starts in blocks:
                for i in range(0, len(starts), CHUNK):
                    B = _windows(x, W, starts[i: i + CHUNK])
                    M[at: at + len(B)] = B; at += len(B)
            mu = M.mean(axis=0)
            C = np.cov(M, rowvar=False, bias=True)
            Cg = (1.0 - knobs["gamma"]) * C + knobs["gamma"] * (np.trace(C) / W) * np.eye(W)
            th = self._from_cov(mu, Cg, W)
            th["n"] = int(tot); th["rank_deficient"] = bool(tot < W)
            return th, "independent"
        st = self.stats([x for x, _ in blocks], W, fit_stride)
        return self.solve(st, knobs), "chunked_pooled"

    def param(self, theta):
        return np.concatenate([theta["mu"], theta["P"].ravel()])

    def average(self, thetas, weights, mode="fedavg"):
        a = weights / weights.sum()
        mu = sum(ai * t["mu"] for ai, t in zip(a, thetas))
        W = thetas[0]["W"]
        if mode in ("naive", "naive_aligned"):
            # Average the RAW BASIS instead of the canonical projector, then
            # re-orthonormalise.
            #
            # `naive_aligned` averages the bases exactly as eigh returns them.
            # MEASURED: this does NOT collapse — LAPACK's sign convention is
            # deterministic given the matrix, and near-identical covariances come
            # out near-identically signed. Reported as a fact, not sold as a
            # control.
            #
            # `naive` first applies a per-client signed permutation of the K
            # columns. That is what an INDEPENDENTLY optimised model actually
            # hands you: the deep collapse is blamed on codebook/encoder
            # permutation-and-sign symmetry, and a basis fixed only up to a
            # signed permutation is the convex analogue. The scramble is
            # deliberate and deterministic (seeded by client index) — this arm
            # demonstrates that averaging objects that live in an ARBITRARY
            # basis destroys them, while averaging the canonical projector
            # (`fed_fedavg`, same inputs) does not. It does not "discover" the
            # collapse; it isolates its cause.
            Us = []
            for i, t in enumerate(thetas):
                U_ = t["U"]
                if mode == "naive":
                    rng = np.random.default_rng(1000 + i)
                    perm = rng.permutation(U_.shape[1])
                    sign = rng.choice([-1.0, 1.0], size=U_.shape[1])
                    U_ = U_[:, perm] * sign
                Us.append(U_)
            Ub = sum(ai * U_ for ai, U_ in zip(a, Us))
            Q, _ = np.linalg.qr(Ub)
            P = Q @ Q.T
            U = Q
        else:
            Pb = sum(ai * t["P"] for ai, t in zip(a, thetas))
            evals, U_ = np.linalg.eigh(0.5 * (Pb + Pb.T))
            U = U_[:, -self.K:]
            P = U @ U.T
        return {"mu": mu, "P": P, "U": U, "W": W,
                "n": int(sum(t["n"] for t in thetas)),
                "rank_deficient": any(t.get("rank_deficient") for t in thetas)}

    def score_windows(self, M, theta):
        R = (M - theta["mu"]) @ (np.eye(theta["W"]) - theta["P"])
        return R * R


class Gauss(_WindowMoments):
    """Secondary head — whitener Σ_γ^{-1/2}; the squared whitened coordinate is
    the Gaussian NLL up to constants. Never enters the pre-registered maximum."""
    name = "gauss"

    def _from_cov(self, mu, Cg, W):
        evals, U = np.linalg.eigh(Cg)
        evals = np.maximum(evals, 1e-12)
        Wh = U @ np.diag(evals ** -0.5) @ U.T
        return {"mu": mu, "Wh": 0.5 * (Wh + Wh.T), "W": W}

    def solve(self, stats, knobs):
        W = len(stats["s"])
        mu, Cg = self._cov(stats, knobs["gamma"], W)
        th = self._from_cov(mu, Cg, W)
        th["n"] = int(stats["n"]); th["rank_deficient"] = bool(stats["n"] < W)
        return th

    def pooled_solve(self, series_per_client, W, fit_stride, knobs):
        blocks, tot = self._pooled_matrix(series_per_client, W, fit_stride)
        if tot == 0:
            raise ValueError("pooled window matrix is empty")
        if tot * W * 8 <= 1_500_000_000:
            M = np.empty((tot, W)); at = 0
            for x, starts in blocks:
                for i in range(0, len(starts), CHUNK):
                    B = _windows(x, W, starts[i: i + CHUNK])
                    M[at: at + len(B)] = B; at += len(B)
            mu = M.mean(axis=0)
            C = np.cov(M, rowvar=False, bias=True)
            Cg = (1.0 - knobs["gamma"]) * C + knobs["gamma"] * (np.trace(C) / W) * np.eye(W)
            th = self._from_cov(mu, Cg, W)
            th["n"] = int(tot); th["rank_deficient"] = bool(tot < W)
            return th, "independent"
        st = self.stats([x for x, _ in blocks], W, fit_stride)
        return self.solve(st, knobs), "chunked_pooled"

    def param(self, theta):
        return np.concatenate([theta["mu"], theta["Wh"].ravel()])

    def average(self, thetas, weights, mode="fedavg"):
        a = weights / weights.sum()
        return {"mu": sum(ai * t["mu"] for ai, t in zip(a, thetas)),
                "Wh": sum(ai * t["Wh"] for ai, t in zip(a, thetas)),
                "W": thetas[0]["W"], "n": int(sum(t["n"] for t in thetas)),
                "rank_deficient": any(t.get("rank_deficient") for t in thetas)}

    def score_windows(self, M, theta):
        Z = (M - theta["mu"]) @ theta["Wh"]
        return Z * Z


# ═════════════════════════════════════════════════════════════════════════════
#   Round-based mappings of the deep arms (AR only — closed form per round)
# ═════════════════════════════════════════════════════════════════════════════

def fedprox_round(head: AR, stats_k: list[dict], knobs: dict, mu: float,
                  rounds: int) -> tuple[list[dict], dict]:
    """Exact proximal point, no optimiser:

        w_k = (G_k + λ n_k I + μ n_k I)⁻¹ (b_k + μ n_k w^(r))
        w^(r+1) = Σ_k α_k w_k          α_k = n_k / Σ n_k

    Limits (asserted in the selftest): μ=0 → the client keeps its LOCAL solution;
    μ→∞ → the client is pinned to the global w^(0). The deep FedProx null is a
    SOLVER artefact (persistent AdamW divides the prox gradient by √v̂); this arm
    only says the convex objective is well behaved — it says nothing about
    whether μ can fix a misaligned deep body."""
    p = head.p
    n = np.array([s["n"] for s in stats_k], dtype=np.float64)
    locals_ = [head.solve(s, knobs) for s in stats_k]
    w_glob = head.average(locals_, n)["w"]                     # w^(0) = FedAvg
    for _ in range(rounds):
        ws = []
        for s in stats_k:
            A = s["G"] + (knobs["ar_lambda"] + mu) * s["n"] * np.eye(p)
            ws.append(np.linalg.solve(A, s["b"] + mu * s["n"] * w_glob))
        w_glob = sum((nk / n.sum()) * w for nk, w in zip(n, ws))
    out = []
    for s, w in zip(stats_k, ws):
        var = max((s["q"] - 2 * w @ s["b"] + w @ s["G"] @ w) / max(s["n"], 1), 1e-8)
        out.append({"w": w, "var": float(var), "n": s["n"], "rank_deficient": False})
    return out, {"w": w_glob, "var": float(np.mean([o["var"] for o in out])),
                 "n": int(n.sum()), "rank_deficient": False}


def fedlocalgd(head: AR, stats_k: list[dict], knobs: dict, eta: float | None,
               tau: int, rounds: int) -> dict:
    """R rounds × τ local gradient steps, in closed form:

        A_k = G_k/n_k + λI,  c_k = b_k/n_k
        τ steps of GD from w:  (I-ηA_k)^τ w + (I - (I-ηA_k)^τ) A_k⁻¹ c_k
        w^(r+1) = Σ_k α_k (…)

    This is the ONLY floor arm that mirrors what the deep arms actually do
    (every one of them is R rounds × τ local steps). τ→∞ → FedAvg of the local
    solutions; τ=1 with many rounds → the pooled solution."""
    p = head.p
    n = np.array([s["n"] for s in stats_k], dtype=np.float64)
    A = [s["G"] / max(s["n"], 1) + knobs["ar_lambda"] * np.eye(p) for s in stats_k]
    c = [s["b"] / max(s["n"], 1) for s in stats_k]
    if eta is None:
        eta = 1.0 / max(max(np.linalg.eigvalsh(Ak).max() for Ak in A), 1e-12)
    Mt, off = [], []
    for Ak, ck in zip(A, c):
        ev, U = np.linalg.eigh(Ak)
        pow_ = U @ np.diag((1.0 - eta * ev) ** tau) @ U.T
        Mt.append(pow_)
        off.append((np.eye(p) - pow_) @ np.linalg.solve(Ak, ck))
    a = n / n.sum()
    w = np.zeros(p)
    for _ in range(rounds):
        w = sum(ai * (Mk @ w + ok) for ai, Mk, ok in zip(a, Mt, off))
    q = sum(s["q"] for s in stats_k); G = sum(s["G"] for s in stats_k)
    b = sum(s["b"] for s in stats_k); N = int(n.sum())
    var = max((q - 2 * w @ b + w @ G @ w) / max(N, 1), 1e-8)
    return {"w": w, "var": float(var), "n": N, "rank_deficient": False}


# ═════════════════════════════════════════════════════════════════════════════
#   Registry
# ═════════════════════════════════════════════════════════════════════════════

def build_heads(knobs: dict) -> dict[str, Head]:
    return {
        "ma_c": MaCentered(),
        "ma_causal": MaCausal(),
        "diff1": Diff1(),
        "random": RandomHead(),
        "ar": AR(knobs["ar_p"]),
        "pca": PCA(knobs["pca_k"]),
        "gauss": Gauss(),
    }


HEAD_NAMES = ["ma_c", "ma_causal", "diff1", "random", "ar", "pca", "gauss"]
DECISION_HEADS = ["ma_c", "ma_causal", "ar", "pca"]      # §1.2: the max is over these


# ═════════════════════════════════════════════════════════════════════════════
#   Self-test — the invariants that must hold before any number is believed
# ═════════════════════════════════════════════════════════════════════════════

def selftest(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0)
    knobs = dict(DEFAULTS)
    H = build_heads(knobs)
    ok = True

    def check(tag, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {tag:52s} {detail}")

    x = rng.normal(size=4000).cumsum() * 0.01 + rng.normal(size=4000)
    W, fit_stride = 128, 1

    # 1. ma_c(k=3) == 0.25 * diff1
    d = np.abs(H["ma_c"].score_series(x, {"k": 3})[1:]
               - 0.25 * H["diff1"].score_series(x, {})[1:]).max()
    check("ma_c(k=3) == 0.25*diff1", d < 1e-12, f"max|Δ|={d:.2e}")

    # 2. MA_10 really averages 10 points
    j = 2000
    d = abs(moving_average_paper(x, 10)[j] - x[j - 5: j + 5].mean())
    check("MA_10 averages 10 points", d < 1e-12, f"|Δ|={d:.2e}")

    # 3. random head is reproducible ACROSS PROCESSES (stable hash)
    a = H["random"].score_series(x, {"seed": 0})
    b = H["random"].score_series(x, {"seed": 0})
    c = H["random"].score_series(x, {"seed": 1})
    check("random head is seed-stable / seed-sensitive",
          np.array_equal(a, b) and not np.array_equal(a, c))

    # 4. ma_causal never looks ahead: perturbing the future cannot move the past
    y = x.copy(); y[3000:] += 5.0
    sx = H["ma_causal"].score_series(x, {"k": 10})
    sy = H["ma_causal"].score_series(y, {"k": 10})
    check("ma_causal is causal", np.array_equal(sx[:3000], sy[:3000]))
    sxc = H["ma_c"].score_series(x, {"k": 10})
    syc = H["ma_c"].score_series(y, {"k": 10})
    check("ma_c is NOT causal (centred, by design)",
          not np.array_equal(sxc[:3000], syc[:3000]))

    # 5. additivity + exactness of the sufficient-statistic federation
    parts = [x[:1500], x[1500:2600], x[2600:]]
    for name in ("ar", "pca", "gauss"):
        head = H[name]
        st_k = [head.stats([p_], W, fit_stride) for p_ in parts]
        st_sum = Head.merge(st_k)
        th_fed = head.solve(st_sum, knobs)
        th_pool, path = head.pooled_solve([[p_] for p_ in parts], W, fit_stride, knobs)
        pf, pp = head.param(th_fed), head.param(th_pool)
        rel = np.linalg.norm(pf - pp) / max(np.linalg.norm(pp), 1e-300)
        # NB: a bit-exact zero would mean the two paths share an object (§0.4)
        check(f"{name}: fed_exact == pooled (independent path)",
              rel < 1e-8 and path == "independent", f"rel={rel:.2e} path={path}")

    # 6. FedAvg is NOT exact, and the excess objective is a positive number
    head = H["ar"]
    st_k = [head.stats([p_], W, fit_stride) for p_ in parts]
    th_k = [head.solve(s, knobs) for s in st_k]
    n = np.array([s["n"] for s in st_k], float)
    th_avg = head.average(th_k, n)
    th_ex = head.solve(Head.merge(st_k), knobs)
    rel = np.linalg.norm(head.param(th_avg) - head.param(th_ex)) / np.linalg.norm(head.param(th_ex))
    J = head.excess_objective(th_avg, th_ex, Head.merge(st_k), knobs)
    check("ar: fedavg != fed_exact, J(ŵ)-J(w*) > 0", rel > 1e-6 and J > 0,
          f"rel={rel:.3f} J={J:.4e}")

    # 7. FedProx limits: μ=0 -> local, μ->inf -> global
    loc, _ = fedprox_round(head, st_k, knobs, 0.0, 3)
    d0 = max(np.abs(a["w"] - b["w"]).max() for a, b in zip(loc, th_k))
    inf_, glob = fedprox_round(head, st_k, knobs, 1e12, 3)
    dinf = max(np.abs(a["w"] - th_avg["w"]).max() for a in inf_)
    check("fedprox mu=0 -> local", d0 < 1e-9, f"max|Δ|={d0:.2e}")
    check("fedprox mu->inf -> global", dinf < 1e-6, f"max|Δ|={dinf:.2e}")

    # 8. local-GD limits: tau->inf -> FedAvg;  tau=1, many rounds -> pooled
    w_big = fedlocalgd(head, st_k, knobs, None, 4096, 40)["w"]
    d1 = np.abs(w_big - th_avg["w"]).max() / max(np.abs(th_avg["w"]).max(), 1e-12)
    w_one = fedlocalgd(head, st_k, knobs, None, 1, 4000)["w"]
    d2 = np.abs(w_one - th_ex["w"]).max() / max(np.abs(th_ex["w"]).max(), 1e-12)
    check("localgd tau->inf -> fedavg", d1 < 1e-6, f"rel={d1:.2e}")
    check("localgd tau=1 -> pooled", d2 < 1e-2, f"rel={d2:.2e}")

    # 9. PCA: averaging a basis fixed only up to a signed permutation destroys
    #    the subspace, while averaging the canonical projector does not.
    #    Needs a fixture with REAL low-rank structure — on isotropic noise every
    #    top-K subspace is arbitrary and all three averages are equally far.
    t_ = np.arange(6000)
    sig = (np.sin(2 * np.pi * t_ / 61) + 0.6 * np.sin(2 * np.pi * t_ / 17 + 0.3)
           + 0.3 * np.sin(2 * np.pi * t_ / 7))
    lowrank = [sig[:2000] + 0.05 * rng.normal(size=2000),
               sig[2000:4000] + 0.05 * rng.normal(size=2000),
               sig[4000:] + 0.05 * rng.normal(size=2000)]
    parts, n = lowrank, np.array([len(p_) for p_ in lowrank], float)
    # K=4 < the fixture's true rank (3 sinusoids = 6 dimensions): with K above
    # the signal rank the trailing components are noise and arbitrary on EVERY
    # client, which would make all three averages equally bad for reasons that
    # have nothing to do with alignment.
    pca4 = PCA(4)
    st4 = [pca4.stats([p_], W, fit_stride) for p_ in parts]
    thp = [pca4.solve(s, knobs) for s in st4]
    P_avg = pca4.average(thp, n, "fedavg")["P"]
    P_nai = pca4.average(thp, n, "naive")["P"]
    P_ali = pca4.average(thp, n, "naive_aligned")["P"]
    P_ex = pca4.solve(Head.merge(st4), knobs)["P"]
    d_avg = np.linalg.norm(P_avg - P_ex)
    d_nai = np.linalg.norm(P_nai - P_ex)
    d_ali = np.linalg.norm(P_ali - P_ex)
    # REGIME (a) — clients that SHARE a subspace. Here breaking the gauge is
    # harmless: any signed permutation of a basis spans the same subspace, so the
    # projector (the only thing the score sees) survives. An earlier version of
    # this file generalised exactly this fixture into "the positive control
    # cannot exist" — WRONG, see regime (b): the fixture was homogeneous by
    # construction, which is the one case where the gauge is free.
    check("pca(a): shared subspace -> scrambled gauge is harmless",
          d_avg < 0.05 and d_nai < 0.05 and d_ali < 0.05,
          f"‖ΔP‖ fedavg={d_avg:.3f} naive={d_nai:.3f} aligned={d_ali:.3f}")

    # REGIME (b) — clients with DIFFERENT subspaces (the real setting). Now the
    # gauge is load-bearing: averaging column j of A's basis with column σ(j) of
    # B's lands outside both. Averaging the canonical projector does not. This is
    # the positive control the design asked for, and it DOES fire — measured on
    # wsd_fed: pca fed_naive 0.115 vs fed_fedavg 0.447 (p<1e-4, all 4 cohorts).
    # Cohort shaped like a real one: K=8 and 6 clients (wsd cohorts are 5-11).
    # The gauge is harmless ONLY when the local subspaces are literally identical
    # (regime a); at realistic K and cohort size even mildly different clients
    # degrade badly, because a random signed permutation has K! * 2^K ways to
    # mismatch and every extra client is another chance to mix columns.
    pca8 = PCA(8)
    het = []
    for f1, f2 in [(61, 17), (23, 41), (97, 13), (31, 7), (53, 11), (79, 19)]:
        s = (np.sin(2 * np.pi * t_[:2000] / f1) + 0.6 * np.sin(2 * np.pi * t_[:2000] / f2 + 0.3))
        het.append(s + 0.05 * rng.normal(size=2000))
    st_h = [pca8.stats([p_], W, fit_stride) for p_ in het]
    th_h = [pca8.solve(s, knobs) for s in st_h]
    nh = np.array([len(p_) for p_ in het], float)

    # The right yardstick is the ENERGY the projector captures on each client's
    # own windows — that is what the score reads — not its distance to the pooled
    # projector. (With genuinely different signals no shared subspace is close to
    # every client, so projector distance says nothing about usefulness.)
    #   captured(P) = mean_k trace(P C_k) / trace(C_k),   C_k = S_k/n_k - μ_kμ_kᵀ
    def captured(P):
        out = []
        for s in st_h:
            n_ = s["n"]; mu = s["s"] / n_
            C = s["S"] / n_ - np.outer(mu, mu)
            out.append(float(np.trace(P @ C) / np.trace(C)))
        return float(np.mean(out))
    cap_avg = captured(pca8.average(th_h, nh, "fedavg")["P"])
    cap_nai = captured(pca8.average(th_h, nh, "naive")["P"])
    cap_loc = float(np.mean([captured(t["P"]) for t in th_h]))
    check("pca(b): realistic cohort -> scrambled gauge COLLAPSES",
          cap_nai < 0.7 * cap_avg,
          f"captured energy: local={cap_loc:.3f} fedavg.{cap_avg:.3f} naive ={cap_nai:.3f}")

    # 10. projector sanity: P is symmetric idempotent of rank K
    P = thp[0]["P"]
    check("pca: P symmetric idempotent rank K",
          np.abs(P - P.T).max() < 1e-10 and np.abs(P @ P - P).max() < 1e-8
          and abs(np.trace(P) - pca4.K) < 1e-6)

    # 11. window geometry matches the arange contract of SlidingWindowDataset
    starts = window_starts(1000, 128, 13)
    check("window_starts == range(0, T-W+1, stride)",
          np.array_equal(starts, np.arange(0, 1000 - 128 + 1, 13))
          and len(window_starts(64, 128, 13)) == 0)

    if verbose:
        print(f"\n  floor_heads selftest: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
