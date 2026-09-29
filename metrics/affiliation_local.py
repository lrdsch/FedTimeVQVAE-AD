"""
=============================================================================
  affiliation_local — vendored affiliation precision/recall (Huet et al., KDD 2022).
=============================================================================

Vendored from github.com/ahstat/affiliation-metrics-py (MIT license, Copyright
2022 Alexis Huet and others). Five upstream modules (`generics`,
`_affiliation_zone`, `_single_ground_truth_event`, `_integral_interval`,
`metrics`) merged into this single file; data-I/O helpers (`read_gz_data`,
`read_all_as_events`, `produce_all_results`) dropped — not needed for runtime
metric computation.

Affiliation is a range-aware F1 alternative to event-F1 (which inflates with
any overlap) and PATE (which requires a buffer hyperparameter). It computes
an average proximity-weighted precision and recall over the affiliation zones
of GT events — no buffer to tune.

API:
  `affiliation_f1(labels, preds)` → dict {affiliation_precision,
   affiliation_recall, affiliation_f1}

Reference:
  Huet, Navarro, Rossi, "Local Evaluation of Time Series Anomaly Detection
  Algorithms", KDD 2022.
"""
from __future__ import annotations

import math
from itertools import groupby
from operator import itemgetter


# ═════════════════════════════════════════════════════════════════════════════
#   generics
# ═════════════════════════════════════════════════════════════════════════════

def _convert_vector_to_events(vector):
    """Binary vector → list of (start, stop) events where stop is exclusive."""
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for _, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    # Half-open intervals: index i represents [i, i+1)
    return [(x, y + 1) for (x, y) in events]


def _infer_Trange(events_pred, events_gt):
    if len(events_gt) == 0:
        raise ValueError('events_gt should contain at least one event')
    if len(events_pred) == 0:
        return _infer_Trange(events_gt, events_gt)
    min_pred = min(x[0] for x in events_pred)
    min_gt = min(x[0] for x in events_gt)
    max_pred = max(x[1] for x in events_pred)
    max_gt = max(x[1] for x in events_gt)
    return (min(min_pred, min_gt), max(max_pred, max_gt))


def _has_point_anomalies(events):
    if len(events) == 0:
        return False
    return min(x[1] - x[0] for x in events) == 0


def _sum_wo_nan(vec):
    return sum(e for e in vec if not math.isnan(e))


def _len_wo_nan(vec):
    return sum(1 for e in vec if not math.isnan(e))


def _f1_func(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ═════════════════════════════════════════════════════════════════════════════
#   _integral_interval
# ═════════════════════════════════════════════════════════════════════════════

def _interval_length(J):
    if J is None:
        return 0
    return J[1] - J[0]


def _sum_interval_lengths(Is):
    return sum(_interval_length(I) for I in Is)


def _interval_intersection(I, J):
    if I is None or J is None:
        return None
    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return None
    return I_inter_J


def _interval_subset(I, J):
    return (I[0] >= J[0]) and (I[1] <= J[1])


def _cut_into_three_func(I, J):
    if I is None:
        return (None, None, None)
    I_inter_J = _interval_intersection(I, J)
    if I == I_inter_J:
        return (None, I_inter_J, None)
    if I[1] <= J[0]:
        return (I, I_inter_J, None)
    if I[0] >= J[1]:
        return (None, I_inter_J, I)
    if (I[0] <= J[0]) and (I[1] >= J[1]):
        return ((I[0], I_inter_J[0]), I_inter_J, (I_inter_J[1], I[1]))
    if I[0] <= J[0]:
        return ((I[0], I_inter_J[0]), I_inter_J, None)
    if I[1] >= J[1]:
        return (None, I_inter_J, (I_inter_J[1], I[1]))
    raise ValueError('unexpected unconsidered case')


def _get_pivot_j(I, J):
    if _interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')
    if max(I) <= min(J):
        return min(J)
    if min(I) >= max(J):
        return max(J)
    raise ValueError('I should be outside J')


def _integral_mini_interval(I, J):
    if I is None:
        return 0
    j_pivot = _get_pivot_j(I, J)
    a = min(I)
    b = max(I)
    return (b - a) * abs(j_pivot - (a + b) / 2)


def _integral_interval_distance(I, J):
    def f(I_cut):
        return _integral_mini_interval(I_cut, J)
    cut = _cut_into_three_func(I, J)
    return f(cut[0]) + 0 + f(cut[2])


def _integral_mini_interval_P_CDFmethod__min_piece(I, J, E):
    if _interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')
    if not _interval_subset(J, E):
        raise ValueError('J should be included in E')
    if not _interval_subset(I, E):
        raise ValueError('I should be included in E')
    e_min, j_min, j_max, e_max = min(E), min(J), max(J), max(E)
    i_min, i_max = min(I), max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m) ** 2 - min(d_min, m) ** 2
    B = max(d_max, m) - max(d_min, m)
    return (1 / 2) * A + m * B


def _integral_mini_interval_Pprecision_CDFmethod(I, J, E):
    integral_min_piece = _integral_mini_interval_P_CDFmethod__min_piece(I, J, E)
    e_min, j_min, j_max, e_max = min(E), min(J), max(J), max(E)
    i_min, i_max = min(I), max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = (1 / 2) * (d_max ** 2 - d_min ** 2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)
    DeltaI = i_max - i_min
    DeltaE = e_max - e_min
    return DeltaI - (1 / DeltaE) * (integral_min_piece + integral_linear_piece + integral_remaining_piece)


def _integral_interval_probaCDF_precision(I, J, E):
    def f(I_cut):
        if I_cut is None:
            return 0
        return _integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E)
    def f0(I_middle):
        if I_middle is None:
            return 0
        return max(I_middle) - min(I_middle)
    cut = _cut_into_three_func(I, J)
    return f(cut[0]) + f0(cut[1]) + f(cut[2])


def _cut_J_based_on_mean_func(J, e_mean):
    if J is None:
        return (None, None)
    if e_mean >= max(J):
        return (J, None)
    if e_mean <= min(J):
        return (None, J)
    return ((min(J), e_mean), (e_mean, max(J)))


def _integral_mini_interval_Precall_CDFmethod(I, J, E):
    i_pivot = _get_pivot_j(J, I)
    e_min, e_max = min(E), max(E)
    e_mean = (e_min + e_max) / 2
    if i_pivot <= min(E) or i_pivot >= max(E):
        return 0

    J_before, J_after = _cut_J_based_on_mean_func(J, e_mean)
    iemin_mean = (e_min + i_pivot) / 2
    J_before_closeE, J_before_closeI = _cut_J_based_on_mean_func(J_before, iemin_mean)
    iemax_mean = (e_max + i_pivot) / 2
    J_after_closeI, J_after_closeE = _cut_J_based_on_mean_func(J_after, iemax_mean)

    def minmax(seg):
        if seg is None:
            return (math.nan, math.nan)
        return (min(seg), max(seg))

    j_bb_min, j_bb_max = minmax(J_before_closeE)
    j_ba_min, j_ba_max = minmax(J_before_closeI)
    j_ab_min, j_ab_max = minmax(J_after_closeI)
    j_aa_min, j_aa_max = minmax(J_after_closeE)

    if i_pivot >= max(J):
        p1 = (i_pivot - e_min) * (j_bb_max - j_bb_min)
        p2 = 2 * i_pivot * (j_ba_max - j_ba_min) - (j_ba_max ** 2 - j_ba_min ** 2)
        p3 = 2 * i_pivot * (j_ab_max - j_ab_min) - (j_ab_max ** 2 - j_ab_min ** 2)
        p4 = (e_max + i_pivot) * (j_aa_max - j_aa_min) - (j_aa_max ** 2 - j_aa_min ** 2)
    elif i_pivot <= min(J):
        p1 = (j_bb_max ** 2 - j_bb_min ** 2) - (e_min + i_pivot) * (j_bb_max - j_bb_min)
        p2 = (j_ba_max ** 2 - j_ba_min ** 2) - 2 * i_pivot * (j_ba_max - j_ba_min)
        p3 = (j_ab_max ** 2 - j_ab_min ** 2) - 2 * i_pivot * (j_ab_max - j_ab_min)
        p4 = (e_max - i_pivot) * (j_aa_max - j_aa_min)
    else:
        raise ValueError('i_pivot should be outside J')

    out_integral = _sum_wo_nan([p1, p2, p3, p4])
    DeltaJ = max(J) - min(J)
    DeltaE = max(E) - min(E)
    return DeltaJ - (1 / DeltaE) * out_integral


def _integral_interval_probaCDF_recall(I, J, E):
    def f(J_cut):
        if J_cut is None:
            return 0
        return _integral_mini_interval_Precall_CDFmethod(I, J_cut, E)
    def f0(J_middle):
        if J_middle is None:
            return 0
        return max(J_middle) - min(J_middle)
    cut = _cut_into_three_func(J, I)
    return f(cut[0]) + f0(cut[1]) + f(cut[2])


# ═════════════════════════════════════════════════════════════════════════════
#   _affiliation_zone
# ═════════════════════════════════════════════════════════════════════════════

def _t_start(j, Js, Trange):
    b = max(Trange)
    n = len(Js)
    if j == n:
        return 2 * b - _t_stop(n - 1, Js, Trange)
    return Js[j][0]


def _t_stop(j, Js, Trange):
    if j == -1:
        a = min(Trange)
        return 2 * a - _t_start(0, Js, Trange)
    return Js[j][1]


def _E_gt_func(j, Js, Trange):
    range_left = (_t_stop(j - 1, Js, Trange) + _t_start(j, Js, Trange)) / 2
    range_right = (_t_stop(j, Js, Trange) + _t_start(j + 1, Js, Trange)) / 2
    return (range_left, range_right)


def _get_all_E_gt_func(Js, Trange):
    return [_E_gt_func(j, Js, Trange) for j in range(len(Js))]


def _affiliation_partition(Is, E_gt):
    # Upstream keeps all intervals (does NOT filter on `kept`); intervals
    # outside the zone become None after `_interval_intersection`. Downstream
    # logic depends on this Nones-preserved alignment.
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        out[j] = [_interval_intersection(I, E_gt[j]) for I in Is]
    return out


# ═════════════════════════════════════════════════════════════════════════════
#   _single_ground_truth_event
# ═════════════════════════════════════════════════════════════════════════════

def _affiliation_precision_distance(Is, J):
    if all(I is None for I in Is):
        return math.nan
    return sum(_integral_interval_distance(I, J) for I in Is) / _sum_interval_lengths(Is)


def _affiliation_precision_proba(Is, J, E):
    if all(I is None for I in Is):
        return math.nan
    return sum(_integral_interval_probaCDF_precision(I, J, E) for I in Is) / _sum_interval_lengths(Is)


def _affiliation_recall_distance(Is, J):
    Is = [I for I in Is if I is not None]
    if len(Is) == 0:
        return math.inf
    E_gt_recall = _get_all_E_gt_func(Is, (-math.inf, math.inf))
    Js = _affiliation_partition([J], E_gt_recall)
    return sum(_integral_interval_distance(J[0], I) for I, J in zip(Is, Js)) / _interval_length(J)


def _affiliation_recall_proba(Is, J, E):
    Is = [I for I in Is if I is not None]
    if len(Is) == 0:
        return 0
    E_gt_recall = _get_all_E_gt_func(Is, E)
    Js = _affiliation_partition([J], E_gt_recall)
    return sum(_integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)) / _interval_length(J)


# ═════════════════════════════════════════════════════════════════════════════
#   metrics — public entry point
# ═════════════════════════════════════════════════════════════════════════════

def _pr_from_events(events_pred, events_gt, Trange):
    """Core affiliation precision/recall computation given events + Trange."""
    if len(events_gt) == 0:
        raise ValueError('events_gt should have at least one event')
    if _has_point_anomalies(events_pred) or _has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')

    E_gt = _get_all_E_gt_func(events_gt, Trange)
    aff_partition = _affiliation_partition(events_pred, E_gt)

    p_precision = [_affiliation_precision_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]
    p_recall = [_affiliation_recall_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    if _len_wo_nan(p_precision) > 0:
        p_precision_avg = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_avg = p_precision[0]  # math.nan
    p_recall_avg = sum(p_recall) / len(p_recall)

    return float(p_precision_avg), float(p_recall_avg)


def affiliation_f1(labels, preds):
    """Affiliation precision / recall / F1 from binary label & prediction vectors.

    Returns {} if there are no GT events or no predicted events — in those
    degenerate cases affiliation is undefined (no zones to average over).
    """
    import numpy as np
    labels = np.asarray(labels).astype(int).ravel()
    preds = np.asarray(preds).astype(int).ravel()

    events_gt = _convert_vector_to_events(labels)
    events_pred = _convert_vector_to_events(preds)

    if len(events_gt) == 0 or len(events_pred) == 0:
        return {}

    # Trange spans the full series — index 0 to len(labels) (half-open).
    Trange = (0, len(labels))

    try:
        p, r = _pr_from_events(events_pred, events_gt, Trange)
    except Exception as exc:
        print(f"[affiliation_local] pr_from_events failed: {exc}")
        return {}

    return {
        "affiliation_precision": p,
        "affiliation_recall": r,
        "affiliation_f1": _f1_func(p, r),
    }
