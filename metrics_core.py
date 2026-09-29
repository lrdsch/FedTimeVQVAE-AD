"""
=============================================================================
  metrics_core.py — THE single module for every anomaly-detection metric.
=============================================================================

Monolithic consolidation (2026-05-24): all metric logic that used to be
scattered across detect.py, detect_per_channel.py, comparisons/_common.py and
comparisons/metrics_suite.py now lives here. Importers:
  * detect.py                 — the detection pipeline (timestamp + per-channel)
  * comparisons/evaluate.py    — the model-agnostic evaluator (reads scores.npz)
  * comparisons/_common.py     — dataset loaders only (re-nothing here)

Contents:
  * segment helpers + point-adjustment transforms (pure numpy)
  * threshold-selection strategies: quantile / best_f1_search / anomaly_ratio / pot
  * vendored-lib wrappers: PATE / VUS / affiliation (lazy imports, warn+{} on fail)
  * event-overlap + detection-delay + best-F1 + AUROC/AUPRC primitives
  * the unified suite: threshold_free / by_threshold (x PA) — `evaluate_scores`
  * the flat suite `compute_all_metrics` (legacy single-threshold report)
  * per-channel localisation: `channel_localization_at_k`, `ips`
  * per-channel DETECTION blocks A/B/C: `per_channel_metrics`,
    `aggregate_macro_weighted`, `joint_micro_metrics`

Convention: scores are ALWAYS "higher = more anomalous". `segments` returns
INCLUSIVE closed ranges [start, end] — the ONE convention used everywhere here
(the point-adjustment transforms iterate them as `arr[s:e+1]`).

sklearn + the vendored PATE/VUS/affiliation libs are imported LAZILY inside the
functions that need them, so the pure-numpy helpers import and run even without
sklearn on the path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Default contamination grid for the anomaly_ratio strategy (CATCH's grid).
DEFAULT_ANOMALY_RATIO_GRID = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0)
# Per-dataset POT calibration level (OmniAnomaly). Fallback 0.02.
POT_LEVEL_BY_DATASET = {"smap": 0.07, "msl": 0.01}


# ─────────────────────────────────────────────────────────────────────────────
# Segments + point-adjustment transforms — PURE numpy
# ─────────────────────────────────────────────────────────────────────────────

def segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of truthy values as INCLUSIVE [start, end] ranges."""
    segs, start = [], None
    for i, v in enumerate(y):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append((start, i - 1)); start = None
    if start is not None:
        segs.append((start, len(y) - 1))
    return segs


def adjust_predicts(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """OmniAnomaly-style point adjustment on BINARY predictions: if any point
    inside a GT anomaly segment is predicted positive, the WHOLE segment is set
    positive. Points outside GT segments unchanged."""
    preds = np.asarray(preds).astype(bool).copy()
    for s, e in segments(labels):
        if preds[s:e + 1].any():
            preds[s:e + 1] = True
    return preds.astype(np.int64)


def adjust_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Score-level point adjustment (InterFusion get_adjusted_composite style):
    within each GT segment, raise every score to the segment MAX → threshold-FREE
    metrics get their point-adjusted variant."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    for s, e in segments(labels):
        if e >= s:
            scores[s:e + 1] = scores[s:e + 1].max()
    return scores


def collapse_channel_labels(y: np.ndarray) -> np.ndarray:
    """Per-channel labels `(T, C)` → `(T,)` by OR across channels (a timestamp
    is anomalous iff any channel is). 1-D input returned unchanged (binarised)."""
    y = np.asarray(y)
    if y.ndim == 1:
        return (y > 0).astype(np.int64)
    if y.ndim == 2:
        return (y > 0).any(axis=1).astype(np.int64)
    raise ValueError(f"Unexpected label shape {y.shape}")


def align_score_and_label(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window models emit `L_score = L_test - window + 1` scores; align by
    truncating labels from the front (`labels[-len(scores):]`), paper-style."""
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.shape[0] > labels.shape[0]:
        raise ValueError(f"More scores ({scores.shape[0]}) than labels ({labels.shape[0]}).")
    return scores, labels[-scores.shape[0]:]


def threshold_by_quantile(train_scores: np.ndarray, q: float = 0.99) -> float:
    """Fixed quantile of the train-score distribution (the 'normal' baseline)."""
    return float(np.quantile(np.asarray(train_scores).reshape(-1), q))


# ─────────────────────────────────────────────────────────────────────────────
# Threshold-selection strategies — PURE numpy (F1 without sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def _f1_from_preds(preds: np.ndarray, labels: np.ndarray, *, point_adjust: bool) -> float:
    p = adjust_predicts(preds, labels) if point_adjust else np.asarray(preds).astype(np.int64)
    y = np.asarray(labels).astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _best_f1_threshold(scores: np.ndarray, labels: np.ndarray, *, point_adjust: bool,
                       n_grid: int = 400) -> float:
    """Oracle threshold maximising (PA-)F1 over a grid (InterFusion/Omni bf_search)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return hi
    grid = np.linspace(lo, hi, n_grid)
    best_thr, best_f1 = grid[0], -1.0
    for thr in grid:
        f1 = _f1_from_preds(scores > thr, labels, point_adjust=point_adjust)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def _anomaly_ratio_threshold(scores: np.ndarray, labels: np.ndarray, *, point_adjust: bool,
                             grid=DEFAULT_ANOMALY_RATIO_GRID) -> float:
    """CATCH strategy: threshold = (100-r)-percentile; keep best (PA-)F1 over r."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    best_thr, best_f1 = float(scores.max()), -1.0
    for r in grid:
        thr = float(np.percentile(scores, 100.0 - r))
        f1 = _f1_from_preds(scores > thr, labels, point_adjust=point_adjust)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def _pot_threshold(train_scores: np.ndarray, test_scores: np.ndarray,
                   *, q: float = 1e-3, level: float = 0.02) -> Optional[float]:
    """OmniAnomaly POT threshold via the vendored SPOT engine (lazy import from
    the OmniAnomaly clone). Returns None if SPOT is unavailable (graceful)."""
    try:
        import sys
        from pathlib import Path
        spot_dir = Path(__file__).resolve().parent / "comparisons" / "OmniAnomaly" / "repo"
        if str(spot_dir) not in sys.path:
            sys.path.insert(0, str(spot_dir))
        from omni_anomaly.spot import SPOT  # type: ignore
    except Exception:
        return None
    try:
        s = SPOT(q)
        s.fit(np.asarray(train_scores, dtype=np.float64).reshape(-1),
              np.asarray(test_scores, dtype=np.float64).reshape(-1))
        s.initialize(level=level, verbose=False)
        ret = s.run(dynamic=False)
        return float(np.mean(ret["thresholds"]))
    except Exception:
        return None


def strategy_thresholds(test_scores: np.ndarray, train_scores: np.ndarray,
                        labels: np.ndarray, *, point_adjust: bool, q: float = 0.99,
                        anomaly_ratio_grid=DEFAULT_ANOMALY_RATIO_GRID,
                        pot_level: float = 0.02, pot_q: float = 1e-3) -> dict[str, float]:
    """All threshold strategies → {name: threshold}. quantile/pot are label-free.

    The quantile strategy's key encodes its level (`quantile_{q}`) so it never
    lies about which q produced the threshold — `q=0.99` yields the historical
    `"quantile_0.99"` key (backward-compatible), `q=0.97` yields `"quantile_0.97"`.
    """
    out: dict[str, float] = {
        f"quantile_{q:g}": threshold_by_quantile(train_scores, q),
        "best_f1_search": _best_f1_threshold(test_scores, labels, point_adjust=point_adjust),
        "anomaly_ratio": _anomaly_ratio_threshold(test_scores, labels, point_adjust=point_adjust,
                                                  grid=anomaly_ratio_grid),
    }
    pot = _pot_threshold(train_scores, test_scores, q=pot_q, level=pot_level)
    if pot is not None:
        out["pot"] = pot
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Primitive metric blocks (numpy + lazy vendored libs / sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event/segment-overlap precision/recall/F1."""
    tru = segments(y_true)
    prd = segments(y_pred)
    if not tru and not prd:
        return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0}

    def overlaps(seg, candidates):
        return any(not (seg[1] < o[0] or o[1] < seg[0]) for o in candidates)

    tp_prd = sum(1 for s in prd if overlaps(s, tru))
    tp_tru = sum(1 for s in tru if overlaps(s, prd))
    p = tp_prd / max(len(prd), 1)
    r = tp_tru / max(len(tru), 1)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"event_precision": p, "event_recall": r, "event_f1": f}


def detection_delay_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Mean detection delay (time-steps) across GT events; missed events cap at
    event length."""
    gt = segments(np.asarray(labels))
    if not gt:
        return {}
    delays = []
    preds = np.asarray(preds)
    for s, e in gt:
        hits = np.flatnonzero(preds[s:e + 1])
        delays.append(float(e - s + 1) if hits.size == 0 else float(hits[0]))
    return {"detection_delay_mean": float(np.mean(delays))}


def best_threshold_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Oracle threshold maximising F1 over the full PR curve → (thr, best_f1)."""
    from sklearn.metrics import precision_recall_curve
    prec_c, rec_c, thr_c = precision_recall_curve(labels, scores)
    denom = np.clip(prec_c + rec_c, 1e-12, None)
    f1 = 2.0 * prec_c * rec_c / denom
    idx = int(np.argmax(f1))
    thr = float(thr_c[min(idx, len(thr_c) - 1)]) if len(thr_c) else 0.0
    return thr, float(np.max(f1))


def affiliation_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Affiliation precision/recall/F1 (Huet et al., KDD 2022)."""
    from metrics.affiliation_local import affiliation_f1
    try:
        return affiliation_f1(labels, preds)
    except Exception as exc:
        print(f"[metrics] affiliation failed: {exc}")
        return {}


def vus_metrics(labels: np.ndarray, scores: np.ndarray, max_window: int) -> dict:
    """VUS-ROC / VUS-PR (Paparrizos et al., VLDB 2022)."""
    from metrics.vus_local import vus_metrics as _vus_lib
    try:
        return _vus_lib(labels, scores, max_window=max_window)
    except Exception as exc:
        print(f"[metrics] VUS failed: {exc}")
        return {}


def pate_metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray, buffer: int) -> dict:
    """PATE (Ghorbani et al., NeurIPS 2024): threshold-aware `pate` + threshold-
    free `pate_f1`."""
    from metrics.pate_local import PATE
    out: dict = {}
    try:
        out["pate"] = float(PATE(labels, preds, e_buffer=buffer, d_buffer=buffer, binary_scores=True))
    except Exception as exc:
        print(f"[metrics] PATE (binary) failed: {exc}")
    try:
        out["pate_f1"] = float(PATE(labels, scores, e_buffer=buffer, d_buffer=buffer, binary_scores=False))
    except Exception as exc:
        print(f"[metrics] PATE (threshold-free) failed: {exc}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Suite blocks: thresholded / threshold-free  (sklearn lazy)
# ─────────────────────────────────────────────────────────────────────────────

def thresholded_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float,
                        *, point_adjust: bool, buffer: int = 100) -> dict:
    """Threshold-DEPENDENT metrics at a threshold; with point_adjust the binary
    preds are segment-filled first."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    preds = (scores > threshold).astype(np.int64)
    if point_adjust:
        preds = adjust_predicts(preds, labels)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    out: dict = {
        "threshold": float(threshold),
        "n_pos_preds": int(preds.sum()),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "accuracy": float(accuracy_score(labels, preds)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }
    out.update(event_metrics(labels, preds))
    out.update(detection_delay_metrics(labels, preds))
    out.update(affiliation_metrics(labels, preds))
    pate = pate_metrics(labels, preds, scores, buffer)
    if "pate" in pate:
        out["pate"] = pate["pate"]
    return out


def threshold_free_metrics(labels: np.ndarray, scores: np.ndarray,
                           *, point_adjust: bool, buffer: int = 100) -> dict:
    """Threshold-FREE metrics; with point_adjust the SCORES are segment-maxed."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if point_adjust:
        scores = adjust_scores(scores, labels)
    out: dict = {}
    if np.unique(labels).size > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        _, out["best_f1"] = best_threshold_f1(labels, scores)
        out.update(vus_metrics(labels, scores, buffer))
        pate = pate_metrics(labels, (scores > np.quantile(scores, 0.99)).astype(int), scores, buffer)
        if "pate_f1" in pate:
            out["pate_f1"] = pate["pate_f1"]
    else:
        out["note"] = "single-class labels — threshold-free metrics undefined"
    return out


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float,
                        *, buffer: int = 100) -> dict:
    """Flat single-threshold suite (legacy; used by recompute_metrics). NO point-
    adjustment. Mirrors what detect.py's report carries at the fixed threshold."""
    from sklearn.metrics import precision_recall_fscore_support
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    assert labels.shape == scores.shape, (labels.shape, scores.shape)
    preds = (scores > threshold).astype(np.int64)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    out: dict = {
        "threshold": float(threshold),
        "n_pos_labels": int(labels.sum()), "n_neg_labels": int((labels == 0).sum()),
        "n_pos_preds": int(preds.sum()),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }
    if np.unique(labels).size > 1:
        from sklearn.metrics import roc_auc_score, average_precision_score
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        _, out["best_f1"] = best_threshold_f1(labels, scores)
        out.update(pate_metrics(labels, preds, scores, buffer))
        out.update(affiliation_metrics(labels, preds))
        out.update(vus_metrics(labels, scores, buffer))
    out.update(event_metrics(labels, preds))
    out.update(detection_delay_metrics(labels, preds))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel localisation (ranking@k + IPS) — pure numpy
# ─────────────────────────────────────────────────────────────────────────────

# Channel attribution asks "WHICH channel is anomalous?". With one channel the
# question has one answer, and every metric below returns 1.0 for any model —
# including a random one. Publishing `ips = 1.000` next to a real AUROC is worse
# than publishing nothing, so the univariate case is refused at the source rather
# than gated at each of the ~6 call sites.
UNIVARIATE_NOTE = "univariate (C=1): channel attribution is undefined, not scored"


def _is_univariate(y_channel: np.ndarray) -> bool:
    a = np.asarray(y_channel)
    return a.ndim < 2 or a.shape[1] < 2


def channel_localization_at_k(channel_scores: np.ndarray, y_channel: np.ndarray,
                              *, ks=(1, 3)) -> dict:
    """precision@k / recall@k of channel ranking vs per-channel GT, averaged over
    anomalous timesteps, plus oracle-k (k = #GT-positive channels at t).
    channel_scores, y_channel : (T, C), higher channel score = more anomalous.

    Returns `{"note": ...}` (no metric keys) when C < 2 — see `UNIVARIATE_NOTE`."""
    if _is_univariate(y_channel):
        return {"note": UNIVARIATE_NOTE}
    cs = np.asarray(channel_scores, dtype=np.float64)
    yc = np.asarray(y_channel).astype(np.int64)
    assert cs.shape == yc.shape, (cs.shape, yc.shape)
    anom_t = np.where(yc.any(axis=1))[0]
    if anom_t.size == 0:
        return {"note": "no per-channel anomalies"}
    out: dict = {"n_anomalous_timesteps": int(anom_t.size)}
    C = cs.shape[1]
    for k in ks:
        if k > C:
            continue
        precs, recs = [], []
        for t in anom_t:
            gt = set(np.where(yc[t] > 0)[0].tolist())
            topk = set(np.argsort(-cs[t])[:k].tolist())
            hit = len(gt & topk)
            precs.append(hit / k)
            recs.append(hit / max(len(gt), 1))
        out[f"channel_precision_at_{k}"] = float(np.mean(precs))
        out[f"channel_recall_at_{k}"] = float(np.mean(recs))
    op, orc = [], []
    for t in anom_t:
        gt = np.where(yc[t] > 0)[0]
        k = max(len(gt), 1)
        topk = set(np.argsort(-cs[t])[:k].tolist())
        hit = len(set(gt.tolist()) & topk)
        op.append(hit / k)
        orc.append(hit / max(len(gt), 1))
    out["channel_precision_at_oracle"] = float(np.mean(op))
    out["channel_recall_at_oracle"] = float(np.mean(orc))
    return out


def ips(channel_scores: np.ndarray, y_channel: np.ndarray) -> dict:
    """InterFusion IPS (Eq.6): segment-importance-weighted top-|G_a| hit ratio.
    For GT segment a: per-channel segment score = max over the segment; I_a =
    top-|G_a| channels; hit = |G_a ∩ I_a|/|G_a|; weight w_a = N_a/ΣN_a.

    Returns `{"note": ...}` when C < 2: top-1 of one channel always hits, so IPS
    would be identically 1.0."""
    if _is_univariate(y_channel):
        return {"note": UNIVARIATE_NOTE}
    cs = np.asarray(channel_scores, dtype=np.float64)
    yc = np.asarray(y_channel).astype(np.int64)
    segs = segments((yc.any(axis=1)).astype(np.int64))
    if not segs:
        return {"note": "no per-channel anomaly segments"}
    hits, weights = [], []
    for s, e in segs:
        gt = np.where(yc[s:e + 1].any(axis=0) > 0)[0]
        if gt.size == 0:
            continue
        seg_score = cs[s:e + 1].max(axis=0)
        topg = set(np.argsort(-seg_score)[:gt.size].tolist())
        hits.append(len(set(gt.tolist()) & topg) / gt.size)
        weights.append(e - s + 1)
    if not hits:
        return {"note": "no interpretable segments"}
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return {"ips": float(np.sum(w * np.asarray(hits))), "n_segments": len(hits)}


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel DETECTION blocks A/B/C (channel-native; detect_per_channel)
# ─────────────────────────────────────────────────────────────────────────────

def per_channel_metrics(y_channel_flat: np.ndarray, channel_scores: np.ndarray,
                        preds: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    """Block A — one dict per channel: precision/recall/f1 (always) + auroc/auprc
    (only when that channel has both classes). Arrays are (T_total, C); thresholds (C,).

    Empty at C < 2: the single row would be the timestamp report under a different
    threshold, and blocks B/C would then report it three more times."""
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    if _is_univariate(y_channel_flat):
        return []
    rows: list[dict] = []
    C = y_channel_flat.shape[1]
    for c in range(C):
        y_c, s_c, p_c = y_channel_flat[:, c], channel_scores[:, c], preds[:, c]
        n_pos = int(y_c.sum())
        row = {
            "channel": int(c), "n_positives": float(n_pos), "threshold": float(thresholds[c]),
            "precision": float(precision_score(y_c, p_c, zero_division=0)),
            "recall": float(recall_score(y_c, p_c, zero_division=0)),
            "f1": float(f1_score(y_c, p_c, zero_division=0)),
        }
        if 0 < n_pos < y_c.shape[0]:
            row["auroc"] = float(roc_auc_score(y_c, s_c))
            row["auprc"] = float(average_precision_score(y_c, s_c))
        else:
            row["auroc"] = float("nan"); row["auprc"] = float("nan")
        rows.append(row)
    return rows


def aggregate_macro_weighted(per_channel: list[dict]) -> dict:
    """Block B — macro (me channels over with a finite metric) and weighted
    (by n_positives) aggregates of precision/recall/f1/auroc/auprc.

    `{"note": ...}` when block A produced no rows (C < 2)."""
    if not per_channel:
        return {"note": UNIVARIATE_NOTE}
    valid = [r for r in per_channel if r["n_positives"] > 0]
    weights = np.asarray([r["n_positives"] for r in per_channel], dtype=np.float64)
    weight_sum = float(weights.sum()) if weights.size else 0.0

    def _macro(key: str) -> float:
        vals = [r[key] for r in valid if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _weighted(key: str) -> float:
        if weight_sum <= 0:
            return float("nan")
        vals = np.asarray([r[key] for r in per_channel], dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            return float("nan")
        return float(np.sum(vals[finite] * weights[finite]) / np.sum(weights[finite]))

    return {
        "macro_precision": _macro("precision"), "macro_recall": _macro("recall"),
        "macro_f1": _macro("f1"), "macro_auroc": _macro("auroc"), "macro_auprc": _macro("auprc"),
        "weighted_precision": _weighted("precision"), "weighted_recall": _weighted("recall"),
        "weighted_f1": _weighted("f1"), "weighted_auroc": _weighted("auroc"),
        "weighted_auprc": _weighted("auprc"),
        "n_valid_channels": float(len(valid)), "n_total_channels": float(len(per_channel)),
    }


def joint_micro_metrics(y_channel_flat: np.ndarray, channel_scores: np.ndarray,
                        preds: np.ndarray) -> dict:
    """Block C — micro precision/recall/f1 (+ auroc/auprc) over the flattened
    (T*C,) array: "anomaly AND in the right channel" as one big binary problem.

    `{"note": ...}` at C < 2, where "the right channel" is vacuous and the flattened
    problem is bit-identical to the timestamp one."""
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    if _is_univariate(y_channel_flat):
        return {"note": UNIVARIATE_NOTE}
    yf = np.asarray(y_channel_flat).reshape(-1)
    pf = np.asarray(preds).reshape(-1)
    sf = np.asarray(channel_scores).reshape(-1)
    n_pos = int(yf.sum())
    out = {
        "joint_precision": float(precision_score(yf, pf, zero_division=0)),
        "joint_recall": float(recall_score(yf, pf, zero_division=0)),
        "joint_f1": float(f1_score(yf, pf, zero_division=0)),
        "joint_n_positives": float(n_pos), "joint_n_cells": float(yf.shape[0]),
    }
    if 0 < n_pos < yf.shape[0]:
        out["joint_auroc"] = float(roc_auc_score(yf, sf))
        out["joint_auprc"] = float(average_precision_score(yf, sf))
    else:
        out["joint_auroc"] = float("nan"); out["joint_auprc"] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level unified suite (used by comparisons/evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scores(labels: np.ndarray, test_scores: np.ndarray, train_scores: np.ndarray,
                    *, buffer: int = 100, q: float = 0.99,
                    channel_scores: Optional[np.ndarray] = None,
                    y_channel: Optional[np.ndarray] = None,
                    anomaly_ratio_grid=DEFAULT_ANOMALY_RATIO_GRID,
                    pot_level: float = 0.02, pot_q: float = 1e-3) -> dict:
    """The full unified report for ONE model's aligned per-timestamp scores:
    threshold_free.{no_pa,pa} · by_threshold.<strategy>.{no_pa,pa} · channel_metrics."""
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    train_scores = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    assert labels.shape == test_scores.shape, (labels.shape, test_scores.shape)

    report: dict = {
        "n_pos_labels": int(labels.sum()), "n_neg_labels": int((labels == 0).sum()),
        "buffer": int(buffer), "threshold_q": float(q),
        "threshold_free": {
            "no_pa": threshold_free_metrics(labels, test_scores, point_adjust=False, buffer=buffer),
            "pa": threshold_free_metrics(labels, test_scores, point_adjust=True, buffer=buffer),
        },
    }
    by_threshold: dict = {}
    for pa in (False, True):
        mode = "pa" if pa else "no_pa"
        thrs = strategy_thresholds(test_scores, train_scores, labels, point_adjust=pa, q=q,
                                   anomaly_ratio_grid=anomaly_ratio_grid,
                                   pot_level=pot_level, pot_q=pot_q)
        for name, thr in thrs.items():
            by_threshold.setdefault(name, {})[mode] = thresholded_metrics(
                labels, test_scores, thr, point_adjust=pa, buffer=buffer)
    report["by_threshold"] = by_threshold

    # Skipped entirely at C < 2 — both helpers return only a note, and a
    # `channel_metrics` block containing nothing but a note invites the reader to
    # think the numbers are missing rather than meaningless.
    if channel_scores is not None and y_channel is not None and not _is_univariate(y_channel):
        ch = channel_localization_at_k(channel_scores, y_channel)
        ch.update(ips(channel_scores, y_channel))
        report["channel_metrics"] = ch
    return report
