"""
=============================================================================
  vus_local — vendored VUS-PR / VUS-ROC (Paparrizos et al., VLDB 2022).
=============================================================================

Vendored from `vus` 0.0.6 on PyPI (github.com/TheDatumOrg/VUS). Subset of
`vus.utils.metrics.metricor.RangeAUC_volume_opt` + the `generate_curve`
wrapper from `vus.analysis.robustness_eval`. The rest of the upstream package
(detector models, plotting, robustness analysis) is dropped.

VUS-ROC / VUS-PR integrate AUROC / AUPRC over a sweep of buffer-window sizes
[0, max_window], producing a single threshold-free + tolerance-aware number
that pairs naturally with PATE-F1.

API:
  `vus_metrics(labels, scores, max_window, thre=250)`
    → dict {vus_roc, vus_pr}

Reference:
  Paparrizos, Boniol, Palpanas, Tsay, Elmore, Franklin,
  "Volume Under the Surface: A New Accuracy Evaluation Measure for
  Time-Series Anomaly Detection", VLDB 2022.
"""
from __future__ import annotations

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#   Range helpers (verbatim from vus.utils.metrics.metricor, minus `self`)
# ═════════════════════════════════════════════════════════════════════════════

def _range_convers_new(label):
    """Binary label vector → list of (start, end_inclusive) anomaly ranges."""
    L = []
    i = 0
    j = 0
    n = len(label)
    while j < n:
        while label[i] == 0:
            i += 1
            if i >= n:
                break
        j = i + 1
        if j >= n:
            if j == n:
                L.append((i, j - 1))
            break
        while label[j] != 0:
            j += 1
            if j >= n:
                L.append((i, j - 1))
                break
        if j >= n:
            break
        L.append((i, j - 1))
        i = j
    return L


def _new_sequence(label, sequence_original, window):
    """Coalesce close ranges after extending each by window//2 in both
    directions. Used to bookkeep TP across the buffer."""
    a = max(sequence_original[0][0] - window // 2, 0)
    sequence_new = []
    for i in range(len(sequence_original) - 1):
        if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
            sequence_new.append((a, sequence_original[i][1] + window // 2))
            a = sequence_original[i + 1][0] - window // 2
    sequence_new.append((a, min(sequence_original[-1][1] + window // 2, len(label) - 1)))
    return sequence_new


def _sequencing(x, L, window=5):
    """Soft-extend each anomaly range with a sqrt-decay shoulder of width
    window//2 on each side, clipped to [0, 1]. This is the VUS 'buffered
    label' used to credit near-anomaly predictions."""
    label = x.copy().astype(float)
    length = len(label)
    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]
        x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
        label[x1] += np.sqrt(1 - (x1 - e) / window)
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / window)
    return np.minimum(np.ones(length), label)


# ═════════════════════════════════════════════════════════════════════════════
#   RangeAUC volume — the core VUS computation
# ═════════════════════════════════════════════════════════════════════════════

def _range_auc_volume_opt(labels_original, score, windowSize, thre=250):
    """Sweep window ∈ [0, windowSize] and threshold over `thre` quantiles of
    `score`; integrate AUC-ROC and AUC-PR per window then average over the
    window axis → (VUS-ROC, VUS-PR).
    """
    window_3d = np.arange(0, windowSize + 1, 1)
    P = np.sum(labels_original)
    seq = _range_convers_new(labels_original)
    if len(seq) == 0:
        return float("nan"), float("nan")
    l = _new_sequence(labels_original, seq, windowSize)

    score_sorted = -np.sort(-score)

    auc_3d = np.zeros(windowSize + 1)
    ap_3d = np.zeros(windowSize + 1)

    tp = np.zeros(thre)
    N_pred = np.zeros(thre)
    for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
        threshold = score_sorted[i]
        pred = score >= threshold
        N_pred[k] = np.sum(pred)

    for window in window_3d:
        labels_extended = _sequencing(labels_original, seq, max(window, 1))
        L = _new_sequence(labels_extended, seq, window)

        TF_list = np.zeros((thre + 2, 2))
        Precision_list = np.ones(thre + 1)
        j = 0

        for i in np.linspace(0, len(score) - 1, thre).astype(int):
            threshold = score_sorted[i]
            pred = score >= threshold
            labels = labels_extended.copy()
            existence = 0

            for seg in L:
                labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in seq:
                labels[seg[0]:seg[1] + 1] = 1

            TP = 0
            N_labels = 0
            for seg in l:
                TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                N_labels += np.sum(labels[seg[0]:seg[1] + 1])

            TP += tp[j]
            FP = N_pred[j] - TP
            existence_ratio = existence / len(L)
            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1) if P_new > 0 else 0.0
            TPR = recall * existence_ratio
            N_new = len(labels) - P_new
            FPR = FP / N_new if N_new > 0 else 0.0
            Precision = TP / N_pred[j] if N_pred[j] > 0 else 1.0

            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = Precision

        TF_list[j + 1] = [1, 1]

        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        auc_3d[window] = np.dot(width, height)

        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = Precision_list[1:]
        ap_3d[window] = np.dot(width_PR, height_PR)

    return float(np.sum(auc_3d) / len(window_3d)), float(np.sum(ap_3d) / len(window_3d))


# ═════════════════════════════════════════════════════════════════════════════
#   Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def vus_metrics(labels, scores, max_window, thre=250):
    """Compute VUS-ROC and VUS-PR.

    `max_window` is the upper bound of the buffer sweep — set it to the
    same proximity tolerance you use elsewhere (`paper_metrics_tolerance`).
    `thre` is the number of threshold quantiles to sweep per window (250 is
    the upstream default).

    Returns {} if labels are single-class or contain no anomalies.
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()
    if labels.sum() == 0 or labels.sum() == labels.size:
        return {}
    try:
        vus_roc, vus_pr = _range_auc_volume_opt(labels, scores, max_window, thre=thre)
    except Exception as exc:
        print(f"[vus_local] _range_auc_volume_opt failed: {exc}")
        return {}
    return {"vus_roc": vus_roc, "vus_pr": vus_pr}
