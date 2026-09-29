"""
=============================================================================
  pate_local — vendored PATE (Proximity-Aware Time series anomaly Evaluation).
=============================================================================

Vendored from `PATE` 0.1.1 on PyPI (Ghorbani et al., NeurIPS 2024,
github.com/Raminghorbanii/PATE). We vendor instead of `pip install PATE`
because the upstream package has hard `==` pins on numpy/scipy/pandas/sklearn
that downgrade an otherwise modern env.

Trimmed from upstream:
  * `ACF_find_buffer_size` (statsmodels + scipy.signal) — unused; we pass the
    `e_buffer`/`d_buffer` explicitly from `cfg.evaluation.paper_metrics_tolerance`.
  * `convert_events_to_array_PATE` — unused by PATE itself.

API: `PATE(y_true, y_score, e_buffer=100, d_buffer=100, binary_scores=False)`.
Returns a scalar in [0, 1]:
  * `binary_scores=True`  → averaged proximity-aware F1 at the given threshold
  * `binary_scores=False` → averaged proximity-aware AUC-PR (threshold-free)

Reference:
  Ghorbani et al., "PATE: Proximity-Aware Time series anomaly Evaluation",
  NeurIPS 2024.
"""
from __future__ import annotations

from itertools import groupby
from operator import itemgetter

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import auc
from sklearn.metrics._ranking import _binary_clf_curve


# ═════════════════════════════════════════════════════════════════════════════
#   Range / event helpers
# ═════════════════════════════════════════════════════════════════════════════

def convert_vector_to_events_PATE(vector_array):
    """Binary vector → list of (start, end) tuples for contiguous 1-runs."""
    positive_indexes = np.where(vector_array > 0)[0]
    events = []
    for _, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    return events


def categorize_predicted_ranges_with_ids(prediction_ranges, label_anomaly_ranges, e_buffer, d_buffer, time_series_length):
    """Split each predicted range into 'pre_buffer' / 'true_detection' /
    'post_buffer' / 'outside' / 'partial_missed' segments relative to each
    labeled anomaly. Returns a dict of categorized sub-ranges + their ids."""
    categorized_ranges_with_ids = {'pre_buffer': [], 'true_detection': [], 'post_buffer': [], 'outside': [], 'partial_missed': []}
    label_anomaly_ids = {i: (start, end) for i, (start, end) in enumerate(label_anomaly_ranges)}
    covered_points_in_label = {anomaly_id: set() for anomaly_id in label_anomaly_ids}
    post_buffer_end_points = {}

    for pred_start, pred_end in prediction_ranges:
        segment_start = pred_start
        for anomaly_id, (label_start, label_end) in label_anomaly_ids.items():
            next_label_start = label_anomaly_ranges[anomaly_id + 1][0] if anomaly_id + 1 < len(label_anomaly_ranges) else time_series_length
            post_buffer_end = min(label_end + d_buffer, next_label_start - 1)
            post_buffer_end_points[anomaly_id] = post_buffer_end
            previous_post_buffer_end = post_buffer_end_points.get(anomaly_id - 1, -1)
            pre_buffer_start = max(0, label_start - e_buffer, previous_post_buffer_end + 1)
            pre_buffer_end = label_start - 1

            # Outside (before pre-buffer)
            if segment_start < pre_buffer_start:
                outside_end = min(pred_end, pre_buffer_start - 1)
                if segment_start <= outside_end:
                    categorized_ranges_with_ids['outside'].append({'range': [segment_start, outside_end], 'id': 'outside'})

            # Pre-buffer
            if segment_start <= pre_buffer_end:
                pre_buffer_segment_start = max(segment_start, pre_buffer_start)
                pre_buffer_segment_end = min(pred_end, pre_buffer_end)
                if pre_buffer_segment_start <= pre_buffer_segment_end:
                    categorized_ranges_with_ids['pre_buffer'].append({'range': [pre_buffer_segment_start, pre_buffer_segment_end], 'pre_buffer_start': pre_buffer_start, 'id': f"{anomaly_id}-pre", 'label_id': anomaly_id})
                segment_start = pre_buffer_segment_end + 1

            # True detection (inside anomaly)
            if segment_start <= label_end:
                actual_segment_end = min(pred_end, label_end)
                if segment_start <= actual_segment_end:
                    categorized_ranges_with_ids['true_detection'].append({'range': [segment_start, actual_segment_end], 'id': f"{anomaly_id}-true_detection", 'label_id': anomaly_id})
                    covered_points_in_label[anomaly_id].update(range(segment_start, actual_segment_end + 1))
                segment_start = actual_segment_end + 1

            # Post-buffer
            if segment_start <= post_buffer_end:
                buffer_segment_end = min(pred_end, post_buffer_end)
                if segment_start <= buffer_segment_end:
                    categorized_ranges_with_ids['post_buffer'].append({'range': [segment_start, buffer_segment_end], 'post_buffer_end': post_buffer_end, 'id': f"{anomaly_id}-buf", 'label_id': anomaly_id})
                segment_start = buffer_segment_end + 1

        # Outside (after the last post-buffer)
        if segment_start <= pred_end:
            categorized_ranges_with_ids['outside'].append({'range': [segment_start, pred_end], 'id': 'outside'})

    # Partial-miss points (anomaly points not covered by any true_detection)
    for anomaly_id, (start, end) in label_anomaly_ids.items():
        labaled_points = set(range(start, end + 1))
        if labaled_points != covered_points_in_label[anomaly_id] and covered_points_in_label[anomaly_id]:
            missed_points = labaled_points - covered_points_in_label[anomaly_id]
            for point in missed_points:
                categorized_ranges_with_ids['partial_missed'].append({'range': [point, point], 'id': f"{anomaly_id}-partial_missed", 'label_id': anomaly_id})

    return categorized_ranges_with_ids


# ═════════════════════════════════════════════════════════════════════════════
#   Proximity-weighted TP / FP / FN weights
# ═════════════════════════════════════════════════════════════════════════════

def _sum_abs_dist(x, a, b):
    """Vectorized Σ_{y=a..b} |x − y| for integer endpoints a ≤ b and integer-valued
    `x` (array). Closed form (O(1) per element) of the inner kernel the proximity
    weights below used to evaluate with a Python loop — BIT-IDENTICAL on integer
    inputs (all intermediates are < 2**53, exact in float64), ~1e4× faster on long
    anomalies. LOCAL optimization, not in upstream PATE 0.1.1."""
    x = np.asarray(x, dtype=np.float64)
    m = b - a + 1
    s_ab = (a + b) * m / 2.0                 # Σ_{y=a..b} y
    out = np.empty_like(x)
    below = x <= a                           # every y ≥ x → |x−y| = y−x
    above = x >= b                           # every y ≤ x → |x−y| = x−y
    mid = ~(below | above)
    out[below] = s_ab - m * x[below]
    out[above] = m * x[above] - s_ab
    if mid.any():
        xm = x[mid]
        lo = np.floor(xm)                    # xm integer ⇒ l = xm
        n1 = lo - a + 1                      # y in [a, lo]:   Σ (x−y)
        n2 = b - lo                          # y in [lo+1, b]: Σ (y−x)
        out[mid] = (n1 * xm - (a + lo) * n1 / 2.0) + ((lo + 1 + b) * n2 / 2.0 - n2 * xm)
    return out


def cal_Wtp_postbuffer(predicted_range, label_anomaly_range, buffer_end_point):
    """Weights ∈ [0,1] for predicted points in the post-buffer zone — decay
    linearly with distance from the anomaly. (Vectorized via _sum_abs_dist.)"""
    label_start, label_end = label_anomaly_range
    post_buffer_start = label_end + 1
    post_buffer_end = buffer_end_point
    xs = np.arange(predicted_range[0], predicted_range[1] + 1)
    xs = xs[(xs >= post_buffer_start) & (xs <= post_buffer_end)]
    if xs.size == 0:
        return []
    max_possible_distance = _sum_abs_dist(np.array([post_buffer_end]), label_start, label_end)[0]
    distance = _sum_abs_dist(xs, label_start, label_end)
    return (1.0 - distance / max_possible_distance).tolist()


def cal_Wtp_prebuffer(predicted_range, label_anomaly_range, prebuffer_start_point):
    """Weights ∈ [0,1] for predicted points in the pre-buffer zone — decay
    linearly with distance from the anomaly (early-detection reward).
    (Vectorized via _sum_abs_dist.)"""
    label_start, label_end = label_anomaly_range
    pre_buffer_start = prebuffer_start_point
    pre_buffer_end = label_start - 1
    xs = np.arange(predicted_range[0], predicted_range[1] + 1)
    xs = xs[(xs >= pre_buffer_start) & (xs <= pre_buffer_end)]
    if xs.size == 0:
        return []
    max_possible_distance = _sum_abs_dist(np.array([pre_buffer_start]), label_start, label_end)[0]
    distance = _sum_abs_dist(xs, label_start, label_end)
    return (1.0 - distance / max_possible_distance).tolist()


def cal_Wfn_partial_missed(predicted_range, label_anomaly_range, delta_buffer):
    """Weights for partially-missed anomaly points — 1 inside the early
    onset-tolerance window, decaying with distance beyond it.
    (Vectorized via _sum_abs_dist.)"""
    label_start, label_end = label_anomaly_range
    buffer_end = label_start + delta_buffer
    pts = np.arange(predicted_range[0], predicted_range[1] + 1).astype(np.float64)
    w = np.ones_like(pts)
    far = pts > buffer_end
    if far.any():
        max_distance = _sum_abs_dist(np.array([label_end]), label_start, label_end)[0]
        if max_distance > 0:                 # else normalized_distance = 0 → weight stays 1
            distance = _sum_abs_dist(pts[far], label_start, buffer_end)
            w[far] = 1.0 - distance / max_distance
    return w.tolist()


def extract_id(id_string):
    """Pull the leading numeric portion out of '{anomaly_id}-pre' etc."""
    numeric_part = ''.join(filter(str.isdigit, id_string))
    return int(numeric_part) if numeric_part else None


def apply_weights(categorized_ranges_with_ids, label_anomaly_ranges):
    """Aggregate per-segment weights into proximity-aware precision/recall."""
    weights = {'TP': [], 'FP': [], 'FN': []}

    # Precompute true_detection lookups ONCE. Upstream re-scanned the whole
    # true_detection list (with an `extract_id` string-parse per element) inside
    # the per-item loop AND in the final FN loop — O(#items × #true_detection),
    # the apply_weights bottleneck. `label_id` is now stored as an int by
    # categorize_*, so this is bit-identical, just without the repeated scans.
    true_detections = categorized_ranges_with_ids['true_detection']
    td_label_ids = {td['label_id'] for td in true_detections}
    td_first_range = {}                      # label_id → FIRST true_detection range (list order)
    for td in true_detections:
        td_first_range.setdefault(td['label_id'], td['range'])

    for category, ranges_with_ids in categorized_ranges_with_ids.items():
        for item in ranges_with_ids:
            pred_range = item['range']
            label_id = item.get('label_id')

            if category == 'true_detection':
                weights['TP'].extend([1] * (pred_range[1] - pred_range[0] + 1))

            if category == 'partial_missed':
                actual_range = td_first_range.get(label_id)
                if actual_range is not None:
                    actual_range_length = actual_range[1] - actual_range[0] + 1
                    weights['FN'].extend(cal_Wfn_partial_missed(pred_range, label_anomaly_ranges[label_id], actual_range_length))

            elif category in ['post_buffer', 'pre_buffer']:
                if category == 'post_buffer':
                    w = cal_Wtp_postbuffer(pred_range, label_anomaly_ranges[label_id], item['post_buffer_end'])
                    weights['TP'].extend(w)
                    weights['FP'].extend(1 - x for x in w)
                else:
                    if label_id in td_label_ids:
                        w = cal_Wtp_prebuffer(pred_range, label_anomaly_ranges[label_id], item['pre_buffer_start'])
                        weights['TP'].extend(w)
                        weights['FP'].extend(1 - x for x in w)
                    else:
                        # Pre-buffer not followed by a true_detection → all FP
                        weights['FP'].extend([1] * (pred_range[1] - pred_range[0] + 1))

            elif category == 'outside':
                weights['FP'].extend([1] * (pred_range[1] - pred_range[0] + 1))

    # Totally-missed anomalies: full FN credit
    for i, label_range in enumerate(label_anomaly_ranges):
        if i not in td_label_ids:
            weights['FN'].extend([1] * (label_range[1] - label_range[0] + 1))

    summed_weights = {k: sum(v) for k, v in weights.items()}
    precision = summed_weights['TP'] / (summed_weights['TP'] + summed_weights['FP']) if (summed_weights['TP'] + summed_weights['FP']) > 0 else 0
    recall = summed_weights['TP'] / (summed_weights['TP'] + summed_weights['FN']) if (summed_weights['TP'] + summed_weights['FN']) > 0 else 0
    return precision, recall


# ═════════════════════════════════════════════════════════════════════════════
#   Curve cleaning + buffer-size sweep
# ═════════════════════════════════════════════════════════════════════════════

def clean_and_compute_auc_pr(recall, precision):
    """AUC-PR after dropping points where recall decreases (PATE's PR curve
    isn't naturally monotonic in recall)."""
    clean_precision, clean_recall = [], []
    prev_recall = -1
    for p, r in zip(precision, recall):
        if r >= prev_recall:
            clean_precision.append(p)
            clean_recall.append(r)
            prev_recall = r
    return auc(clean_recall, clean_precision)


def generate_buffer_points(max_buffer_size, num_splits, include_zero=True):
    """Evenly-spaced integer buffer sizes from 0 (or max/num_splits) up to max."""
    if include_zero:
        start_point = 0
        num_points = num_splits + 1
    else:
        start_point = max_buffer_size / num_splits
        num_points = num_splits
    return np.linspace(start_point, max_buffer_size, num=num_points, dtype=int)


# ═════════════════════════════════════════════════════════════════════════════
#   Score-handling and entry point
# ═════════════════════════════════════════════════════════════════════════════

def compute_adjusted_scores(i, threshold, y_score, actual_anomaly_ranges, e_buffer, d_buffer, time_series_length):
    binary_predicted = (y_score >= threshold).astype(int)
    predicted_ranges = convert_vector_to_events_PATE(binary_predicted)
    categorized = categorize_predicted_ranges_with_ids(predicted_ranges, actual_anomaly_ranges, e_buffer, d_buffer, time_series_length)
    return apply_weights(categorized, actual_anomaly_ranges)


def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def handle_binary_scores(y_true, y_score, e_buffer, d_buffer, num_splits_MaxBuffer, include_zero):
    actual_anomaly_ranges = convert_vector_to_events_PATE(y_true)
    time_series_length = len(y_true)
    f1_score_list = []
    for selected_early_buffer in generate_buffer_points(e_buffer, num_splits_MaxBuffer, include_zero):
        for selected_delayed_buffer in generate_buffer_points(d_buffer, num_splits_MaxBuffer, include_zero):
            predicted_ranges = convert_vector_to_events_PATE(y_score)
            categorized = categorize_predicted_ranges_with_ids(predicted_ranges, actual_anomaly_ranges, selected_early_buffer, selected_delayed_buffer, time_series_length)
            precision, recall = apply_weights(categorized, actual_anomaly_ranges)
            f1_score_list.append(compute_f1_score(precision, recall))
    return float(np.mean(f1_score_list)) if f1_score_list else 0.0


def handle_continuous_scores(y_true, y_score, e_buffer, d_buffer, pos_label, sample_weight, n_jobs, drop_intermediate, Big_Data, num_desired_thresholds, num_splits_MaxBuffer, include_zero):
    auc_pr_list = []
    actual_anomaly_ranges = convert_vector_to_events_PATE(y_true)
    time_series_length = len(y_true)

    fps_orig, tps_orig, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if drop_intermediate and len(tps_orig) > 2:
        optimal_idxs = np.where(np.concatenate([[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]))[0]
        fps_orig, tps_orig, thresholds = fps_orig[optimal_idxs], tps_orig[optimal_idxs], thresholds[optimal_idxs]
    if Big_Data:
        percentiles = np.linspace(100, 0, num_desired_thresholds)
        thresholds = np.percentile(thresholds, percentiles)

    for selected_early_buffer in generate_buffer_points(e_buffer, num_splits_MaxBuffer, include_zero):
        for selected_delayed_buffer in generate_buffer_points(d_buffer, num_splits_MaxBuffer, include_zero):
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_adjusted_scores)(
                    i, threshold, y_score, actual_anomaly_ranges,
                    selected_early_buffer, selected_delayed_buffer, time_series_length,
                ) for i, threshold in enumerate(thresholds)
            )
            precision, recall = zip(*results)
            precision = np.hstack(([1], np.array(precision)))
            recall = np.hstack(([0], np.array(recall)))
            auc_pr_list.append(clean_and_compute_auc_pr(recall, precision))

    return float(np.mean(auc_pr_list)) if auc_pr_list else 0.0


def PATE(
    y_true, y_score,
    e_buffer=100, d_buffer=100,
    pos_label=1, sample_weight=None,
    n_jobs=1, drop_intermediate=True,
    Big_Data=True, num_desired_thresholds=250,
    num_splits_MaxBuffer=1, include_zero=True,
    binary_scores=False,
):
    """Proximity-Aware Time series anomaly Evaluation.

    `binary_scores=True`  → averaged proximity-aware F1 (y_score is binary 0/1).
    `binary_scores=False` → averaged proximity-aware AUC-PR (y_score continuous).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if binary_scores:
        return handle_binary_scores(y_true, y_score, e_buffer, d_buffer, num_splits_MaxBuffer, include_zero)
    return handle_continuous_scores(y_true, y_score, e_buffer, d_buffer, pos_label, sample_weight, n_jobs, drop_intermediate, Big_Data, num_desired_thresholds, num_splits_MaxBuffer, include_zero)
