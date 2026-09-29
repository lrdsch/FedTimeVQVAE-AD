#!/usr/bin/env python3
"""FLOOR — the calibration-baseline runner (design: documentation/research/FLOOR_BASELINE.md).

Scores every client of a federated dataset with a closed-form / zero-parameter
head and pushes the result through the EXACT `detect` path the neural arms use,
so the numbers are comparable to `artifacts/converged_all` and
`artifacts/converge60` by construction.

Heads live in `scripts/floor_heads.py` (pure numpy, no repo imports). The
deciding head is `ma_c` — the centred moving-average residual pre-registered at
federated_method.tex:326,407:

    s_t = ( x_t - MA_k(x)_t )**2         MA_k = detect._moving_average_paper
    k = 10   ("10 min" ONLY on wsd_fed, whose metadata declares dt_sec=60;
              on ucr_split there is no sampling rate and k is 10 SAMPLES)

Modes (§2 of the design):

    local              per-client fit — the reference
    central            pooled fit through an INDEPENDENT path (materialised
                       design matrix + lstsq / np.cov), never a sum of stats
    central_capN       pooled over the first --cap-clients clients only
    fed_exact          server sums the sufficient statistics, solves once,
                       broadcasts.  EXACT by construction; --witness proves it
                       against `central` and FAILS on a bit-exact zero, which
                       would mean the two paths share an object
    fed_fedavg         per-client closed-form solve, then n_k-weighted average
                       of the canonical parameter
    fed_fedavg_uniform same, unweighted — isolates weighting from averaging
    fed_naive          raw bases averaged with NO sign/permutation alignment
                       (positive control: MUST collapse).  PCA only
    fed_scaleonly      local w + pooled σ̂² — provably a no-op, kept as a
                       regression test on the metric axis (AR only)
    fed_oneclient      one client's parameter broadcast to the cohort
    fed_prox           exact proximal point, μ swept (AR only)
    fed_localgd        R rounds × τ local GD steps in closed form (AR only)

Zero-parameter heads (`ma_c`, `ma_causal`, `diff1`, `random`) are ARM-INVARIANT:
local == centralized == every federated mode, bit-identically. They are stamped
`_mode="arm_invariant"` and every federated mode is SKIPPED for them with the
reason printed — no federation question can be asked of such a head.

Metrics: the FULL suite `pipeline/detect.detect()` writes into report.json —
`_detection_metrics` + `_event_metrics` + per-entity paper top-K (1/3/5) +
the macro threshold-free block + the unified `metrics_core.evaluate_scores`
suite (threshold_free.{no_pa,pa} · by_threshold.<strategy>.{no_pa,pa}).

Canonical accumulation geometry (§3):
  * windows enumerated exactly like data.SlidingWindowDataset(records, W, stride)
  * the WHOLE window contributes (no `skip` offset) so `coverage` stays uniform
    under rolling_aggregation="sum", which never divides by coverage
  * _finalize_entities(train) runs BEFORE _fit_threshold_paper (the quantile
    fallback reads overall_scores)

Outputs (resumable; the JSONL is rewritten de-duplicated on every run):
    artifacts/floor/records_<dataset>.jsonl
    artifacts/floor/<dataset>__<arm>.json      {meta, summary, records}
    artifacts/floor/witness_<dataset>.json     exactness residuals per cluster

Examples:
    python scripts/floor_eval.py --selftest
    python scripts/floor_eval.py --dataset wsd_fed --heads ma_c --jobs 4
    python scripts/floor_eval.py --dataset wsd_fed --heads ar,pca,gauss \\
        --modes local,central,fed_exact,fed_fedavg --witness
    python scripts/floor_eval.py --dataset wsd_fed --heads ma_c --k 20 --impulse off
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

import floor_heads as FH                                    # noqa: E402

SCHEMA = 2                      # bump when the record fields change
# NOT bumped for `_model` (added 2026-07-30) on purpose: the field is purely additive and
# `floor_table` / `floor_stats` derive it from `_arm` when absent, so pre-existing rows stay
# fully usable. Bumping would force a re-score that changes no metric. Bump only when a
# recorded NUMBER would differ.
DEFAULT_K = FH.DEFAULTS["k"]    # pre-registered: 10 samples = 10 min at dt_sec=60

# Modes that need more than one client to mean anything.
FED_MODES = ["central", "central_capN", "fed_exact", "fed_fedavg",
             "fed_fedavg_uniform", "fed_naive", "fed_naive_aligned",
             "fed_scaleonly", "fed_oneclient", "fed_prox", "fed_localgd"]
ALL_MODES = ["local"] + FED_MODES
# Modes only defined for particular heads (§2.1).
MODE_REQUIRES = {"fed_naive": {"pca"}, "fed_naive_aligned": {"pca"},
                 "fed_scaleonly": {"ar"}, "fed_prox": {"ar"}, "fed_localgd": {"ar"}}

# Flat metric keys that must never be silently absent from a summary.
CORE_METRICS = ["vus_pr", "auprc", "auroc", "pate_f1", "affiliation_f1", "f1"]

# Paper broom (workshop paper, frozen 2026-07-30)
# What goes in the paper is a SUBSET of what this runner can do. The entries below
# are implemented and self-tested but deliberately OUTSIDE the reported set, so a
# stray sweep can never end up in a table by accident. Selecting one is allowed —
# it is a legitimate ablation — but it prints a banner. Rationale per item and the
# in-scope batch list: documentation/operations/FLOOR_RUNBOOK.md §1.
OUT_OF_SCOPE = {
    "fed_prox": "the mu=0 -> local and mu->inf -> global limits are already asserted at "
                "machine precision in the selftest (0.0e+00 / 1.9e-13), and "
                "FLOOR_BASELINE.md §8.3 forbids the only inference a data sweep would buy",
    "central_capN": "re-tests a ledger claim already marked INVERTED under the current "
                    "architecture (centralized 0.585 < local 0.617)",
    "gauss": "in scope on wsd_fed only — O(W^2) memory / O(W^3) solve, measured 4.99 s "
             "per client and a 484 MB chunk buffer per worker at W=3028 on ucr_split_w2p",
}


# ═════════════════════════════════════════════════════════════════════════════
#   Config
# ═════════════════════════════════════════════════════════════════════════════

def build_cfg(dataset: str, impulse: bool, window: int | None = None):
    from config import Config, apply_dataset_overrides, apply_env_overrides
    cfg = Config()
    cfg.dataset.name = dataset
    if window is not None:
        cfg.dataset.window_length = int(window)
    with contextlib.redirect_stdout(io.StringIO()):
        apply_dataset_overrides(cfg)      # wsd_fed -> metrics_tolerance 14
        apply_env_overrides(cfg)
    cfg.evaluation.save_plots = False
    cfg.evaluation.save_scores = False
    cfg.scoring.use_impulse_term = bool(impulse)
    return cfg


def knobs_from_args(args) -> dict:
    kn = dict(FH.DEFAULTS)
    kn.update({"k": args.k, "ar_p": args.ar_p, "ar_lambda": args.ar_lambda,
               "pca_k": args.pca_k, "gamma": args.gamma, "seed": args.seed})
    return kn


def head_knobs(head_name: str, kn: dict, mode: str, extra: dict) -> dict:
    """The knobs that actually change this arm's numbers — they go in the arm
    tag and in `_knobs`, so a sweep can never collide with itself on resume."""
    out: dict = {}
    if head_name in ("ma_c", "ma_causal"):
        out["k"] = kn["k"]
    if head_name == "random":
        out["seed"] = kn["seed"]
    if head_name == "ar":
        out.update({"p": kn["ar_p"], "lam": kn["ar_lambda"]})
    if head_name in ("pca",):
        out.update({"K": kn["pca_k"], "gamma": kn["gamma"]})
    if head_name == "gauss":
        out["gamma"] = kn["gamma"]
    out.update(extra)
    return out


def arm_tag(head: str, mode: str, knobs: dict, impulse: bool,
            window: int, default_window: int, fit_stride: int = 1) -> str:
    """Stable, collision-free arm name. NEVER `local`/`centralized`/`fa_*`:
    summarize_converged.py:64-65 matches those two literals and
    aggregate_all.py:30 globs the third.

    `fit_stride` is in the tag because it changes the ESTIMATOR, not just the cost:
    on ucr_split at stride 1 only 60/1130 clients have n < 2W and none have n < W,
    while at stride 13 it is 734/1130 and 546/1130 with a strictly singular
    covariance. Two runs at different fit strides are different arms and must never
    dedupe onto each other. Only the non-default stride is tagged, so the
    pre-registered stride-1 names the write-up cites (`floor_ma_c_k10`) are stable.

    `--suite` is deliberately NOT here: it only ADDS the nested metric block, it
    never changes a value, so tagging it would split one arm into two. `dedupe()`
    protects the block instead."""
    parts = [f"floor_{head}"]
    for key in ("k", "p", "K", "lam", "gamma", "seed", "cap", "mu", "tau", "rounds"):
        if key in knobs:
            v = knobs[key]
            if key in ("k", "p", "K", "cap", "tau", "rounds", "seed"):
                parts.append(f"{key}{v}")
            else:
                parts.append(f"{key}{v:g}")
    tag = "_".join(parts)
    if mode != "local":
        tag += f"__{mode}"
    if window != default_window:
        tag += f"__w{window}"
    if fit_stride != 1:
        tag += f"__fs{fit_stride}"
    if not impulse:
        tag += "__noimpulse"
    return tag


def model_tag(arm: str) -> str:
    """The WINDOW-INDEPENDENT identity of a model: the arm name with `__w<W>` removed.

    Why this has to exist. On a per-series-window build the window is part of the arm name by
    design (§8.10: a floor at one W is not comparable to a deep arm at another, and the tag is
    what makes the collision impossible). But on `ucr_split_w2p` the window is a property of
    each SERIES — 79 distinct values across the 180 clusters — so ONE conceptual model becomes
    79 arm names of ~5 entities each. Aggregating by `_arm` there gives 395 table rows for 5
    models and no arm with an n large enough for a paired test: the entire UCR side of the
    paper is unaggregatable. Caught by the smoke run on 2026-07-30.

    So: `_arm` remains the STORAGE identity (never collides, never silently mixes windows) and
    `_model` is the ANALYSIS identity (what a table row and a paired test are about). The
    window range spanned by a model is reported alongside it, so the heterogeneity is declared
    rather than hidden — on w2p the treatment genuinely is not of constant size across clusters.
    """
    return re.sub(r"__w\d+", "", arm)


# ═════════════════════════════════════════════════════════════════════════════
#   Scoring geometry — identical to the deep path
# ═════════════════════════════════════════════════════════════════════════════

def accumulate(head, theta, recs, W: int, stride: int, D, D_data):
    """Push one split through the canonical accumulation of §3 and return the
    entities with `channel_scores` filled in."""
    ents = D._init_entity(recs)
    series = [np.asarray(r.X[:, 0], dtype=np.float64) for r in recs]
    n_win = 0
    if head.granularity == "series":
        scored = [head.score_series(x, theta) for x in series]
        for wi in D_data.SlidingWindowDataset(recs, W, stride).indices:
            e = ents[wi.record_index]
            e["channel_sum"][wi.start:wi.stop, 0] += scored[wi.record_index][wi.start:wi.stop]
            e["coverage"][wi.start:wi.stop] += 1.0
            n_win += 1
    else:
        for ri, x in enumerate(series):
            starts = FH.window_starts(len(x), W, stride)
            e = ents[ri]
            for i in range(0, len(starts), FH.CHUNK):
                blk = starts[i: i + FH.CHUNK]
                V = head.score_windows(FH._windows(x, W, blk), theta)
                for s, v in zip(blk, V):
                    e["channel_sum"][s: s + W, 0] += v
                    e["coverage"][s: s + W] += 1.0
            n_win += len(starts)
    for e in ents:
        e["channel_scores"] = D._assemble_rolling(
            e["channel_sum"], e["coverage"], "sum")
    return ents, n_win


# ═════════════════════════════════════════════════════════════════════════════
#   Metrics — the FULL suite detect() writes into report.json
# ═════════════════════════════════════════════════════════════════════════════

def full_metrics(labels, scores, train_scores, preds, test_ents, cfg, D,
                 with_suite: bool) -> tuple[dict, dict]:
    """Returns (flat, suite). Flat = every scalar detect() puts at the top level
    of report.json; suite = the nested unified block (or {})."""
    flat: dict = {}
    flat.update(D._detection_metrics(labels, preds, scores, cfg))     # 15 keys
    flat.update(D._event_metrics(labels, preds))                      # 3 keys

    # Macro threshold-free block — gated on >1 entity, exactly like detect.py:1361.
    if len(test_ents) > 1:
        per_entity = [
            D._threshold_free_metrics(
                np.asarray(e["labels"]), np.asarray(e["overall_scores"]),
                point_adjust=False, buffer=cfg.evaluation.paper_metrics_tolerance)
            for e in test_ents if e["labels"] is not None
        ]
        if per_entity:
            keys = {k for m in per_entity for k, v in m.items()
                    if isinstance(v, (int, float))}
            for key in sorted(keys):
                vals = [m[key] for m in per_entity
                        if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
                if vals:
                    flat[f"{key}_macro"] = float(np.mean(vals))
            flat["n_entities_macro_avg"] = len(per_entity)

    # Paper-style per-entity top-K (1/3/5), averaged across entities.
    paper = [D._paper_metrics(np.asarray(e["labels"]), np.asarray(e["overall_scores"]), cfg)
             for e in test_ents if e["labels"] is not None]
    if paper:
        for key in paper[0]:
            vals = [m[key] for m in paper if np.isfinite(m[key])]
            if vals:
                flat[key] = float(np.mean(vals))

    suite: dict = {}
    if with_suite:
        suite = D._evaluate_scores(
            labels, scores, train_scores,
            buffer=cfg.evaluation.paper_metrics_tolerance, q=cfg.threshold.q,
            pot_level=D._POT_LEVEL_BY_DATASET.get(cfg.dataset.name.lower(), 0.02))
    return flat, suite


# ═════════════════════════════════════════════════════════════════════════════
#   One job = (cluster, head, mode, knobs) — fit once, score every client
# ═════════════════════════════════════════════════════════════════════════════

def run_job(job: dict) -> list[dict]:
    try:
        return _run_job(job)
    except Exception:
        print(f"[floor] JOB FAILED {job['cluster']}/{job['head']}/{job['mode']}:\n"
              + traceback.format_exc(), flush=True)
        return []


def _run_job(job: dict) -> list[dict]:
    import data as D_data
    import detect as D

    kn = job["knobs"]
    cfg = build_cfg(job["dataset"], job["impulse"], job["window"])
    W = cfg.dataset.window_length
    stride = D._resolve_eval_stride(cfg)
    fit_stride = job["fit_stride"]
    head = FH.build_heads(kn)[job["head"]]
    mode = job["mode"]
    t_job = time.time()

    # ── load the whole cohort once ──────────────────────────────────────────
    recs = {}
    for e in job["entities"]:
        c = copy.deepcopy(cfg)
        c.dataset.entity_id = e
        with contextlib.redirect_stdout(io.StringIO()):
            recs[e] = D_data.load_scaled_records(c)

    ents_order = list(job["entities"])
    train_series = {e: [np.asarray(r.X[:, 0], dtype=np.float64) for r in recs[e][0]]
                    for e in ents_order}

    # ── fit ─────────────────────────────────────────────────────────────────
    info: dict = {"pooled_path": None, "param_relerr": None, "score_relerr": None,
                  "excess_objective": None, "objective_star": None,
                  "excess_objective_rel": None, "excess_objective_local": None,
                  "excess_objective_local_rel": None,
                  "n_fit": None, "rank_deficient": None}
    if head.zero_param:
        theta = {e: dict(kn) for e in ents_order}
    else:
        stats_k = {e: head.stats(train_series[e], W, fit_stride) for e in ents_order}
        n_k = np.array([max(stats_k[e]["n"], 1) for e in ents_order], dtype=np.float64)
        local_th = {e: head.solve(stats_k[e], kn) for e in ents_order}

        if mode == "local":
            theta = local_th
        elif mode in ("central", "central_capN"):
            use = ents_order if mode == "central" else ents_order[: job["cap_clients"]]
            th, path = head.pooled_solve([train_series[e] for e in use], W, fit_stride, kn)
            theta = {e: th for e in ents_order}
            info["pooled_path"] = path
        elif mode == "fed_exact":
            th = head.solve(FH.Head.merge([stats_k[e] for e in ents_order]), kn)
            theta = {e: th for e in ents_order}
            if job["witness"]:
                ref, path = head.pooled_solve([train_series[e] for e in ents_order],
                                              W, fit_stride, kn)
                pf, pp = head.param(th), head.param(ref)
                denom = max(float(np.linalg.norm(pp)), 1e-300)
                info["param_relerr"] = float(np.linalg.norm(pf - pp) / denom)
                info["pooled_path"] = path
                info["_witness_ref"] = ref
        elif mode in ("fed_fedavg", "fed_fedavg_uniform", "fed_naive", "fed_naive_aligned"):
            wts = n_k if mode != "fed_fedavg_uniform" else np.ones_like(n_k)
            avg_mode = {"fed_naive": "naive",
                        "fed_naive_aligned": "naive_aligned"}.get(mode, "fedavg")
            th = head.average([local_th[e] for e in ents_order], wts, avg_mode)
            theta = {e: th for e in ents_order}
            if job["head"] == "ar":
                merged = FH.Head.merge([stats_k[e] for e in ents_order])
                star = head.solve(merged, kn)
                info["excess_objective"] = head.excess_objective(th, star, merged, kn)
                # The raw quadratic form is uninterpretable on its own, and quoting it alone
                # inverts two things at once. (a) Relative to J(w*) the cluster ordering flips:
                # on wsd raw 113/46/35/14 becomes 6.3%/0.9%/0.6%/3.5%, so the smallest raw
                # number is the second-worst cluster. (b) The LOCAL arm's own excess on the
                # same pooled objective is 2-6x LARGER than FedAvg's, so "the objective
                # degrades measurably while the metric does not move" reads as FedAvg-hurts
                # when on that objective FedAvg is far CLOSER to the pooled optimum than the
                # local arm it is called indistinguishable from. Both references are recorded
                # here so the sentence cannot be written without them.
                j_star = head.objective_star(star, merged, kn)
                info["objective_star"] = j_star
                info["excess_objective_rel"] = (
                    info["excess_objective"] / j_star if abs(j_star) > 1e-300 else None)
                loc = [head.excess_objective(local_th[e], star, merged, kn) for e in ents_order]
                info["excess_objective_local"] = float(np.mean(loc))
                info["excess_objective_local_rel"] = (
                    float(np.mean(loc)) / j_star if abs(j_star) > 1e-300 else None)
        elif mode == "fed_scaleonly":
            pooled = head.solve(FH.Head.merge([stats_k[e] for e in ents_order]), kn)
            theta = {e: {**local_th[e], "var": pooled["var"]} for e in ents_order}
        elif mode == "fed_oneclient":
            th = local_th[ents_order[0]]
            theta = {e: th for e in ents_order}
        elif mode == "fed_prox":
            per, _glob = FH.fedprox_round(head, [stats_k[e] for e in ents_order], kn,
                                          job["mu"], job["rounds"])
            theta = dict(zip(ents_order, per))
        elif mode == "fed_localgd":
            th = FH.fedlocalgd(head, [stats_k[e] for e in ents_order], kn,
                               job["eta"], job["tau"], job["rounds"])
            theta = {e: th for e in ents_order}
        else:
            raise ValueError(f"unknown mode {mode!r}")

        # rows/windows the DEPLOYED parameter of each client was actually fitted on
        pooled_n = int(sum(stats_k[e]["n"] for e in ents_order))
        if mode in ("local", "fed_scaleonly", "fed_prox"):
            info["n_fit"] = {e: int(stats_k[e]["n"]) for e in ents_order}
        elif mode == "central_capN":
            capped = int(sum(stats_k[e]["n"] for e in ents_order[: job["cap_clients"]]))
            info["n_fit"] = {e: capped for e in ents_order}
        elif mode == "fed_oneclient":
            info["n_fit"] = {e: int(stats_k[ents_order[0]]["n"]) for e in ents_order}
        else:
            info["n_fit"] = {e: pooled_n for e in ents_order}
        info["rank_deficient"] = bool(any(
            theta[e].get("rank_deficient") for e in ents_order))

    # score every client
    rows = []
    for e in ents_order:
        t0 = time.time()
        tr, _va, te = recs[e]
        tr_ents, _ = accumulate(head, theta[e], tr, W, stride, D, D_data)
        te_ents, n_win = accumulate(head, theta[e], te, W, stride, D, D_data)
        # train FIRST: the quantile fallback reads overall_scores
        train_scores, _ = D._finalize_entities(tr_ents, cfg)
        scores, labels = D._finalize_entities(te_ents, cfg)
        thr = D._fit_threshold_paper(tr_ents, cfg)
        preds = (scores > thr).astype(np.int64)
        flat, suite = full_metrics(labels, scores, train_scores, preds, te_ents,
                                   cfg, D, job["suite"])

        # the exactness witness needs a SCORE-level residual, not only params
        if mode == "fed_exact" and job["witness"] and "_witness_ref" in info \
                and info["score_relerr"] is None:
            ref_ents, _ = accumulate(head, info["_witness_ref"], te, W, stride, D, D_data)
            ref_scores, _ = D._finalize_entities(ref_ents, cfg)
            denom = max(float(np.max(np.abs(scores))), 1e-300)
            info["score_relerr"] = float(np.max(np.abs(scores - ref_scores)) / denom)

        row = {
            "_schema": SCHEMA,
            "_arm": job["arm"],            # storage identity: carries the window
            "_model": model_tag(job["arm"]),   # analysis identity: window-independent
            "_head": job["head"],
            "_mode": "arm_invariant" if head.zero_param else mode,
            "_knobs": job["knobs_tag"],
            "_cluster": job["cluster"],
            "_entity": e,
            "_seed": int(kn["seed"]),
            "_impulse": bool(job["impulse"]),
            "_threshold_rule": "quantile_fallback",
            "_threshold": float(thr),
            "_unit": job["unit"],
            "_n_windows": int(n_win),
            "_n_fit_windows": None if head.zero_param else int(info["n_fit"][e]),
            "_fit_stride": None if head.zero_param else int(fit_stride),
            "_eval_stride": int(stride),
            "_window": int(W),
            "_tolerance": int(cfg.evaluation.paper_metrics_tolerance),
            "_pooled_path": info["pooled_path"],
            "_param_relerr": info["param_relerr"],
            "_score_relerr": info["score_relerr"],
            "_excess_objective": info["excess_objective"],
            "_objective_star": info["objective_star"],
            "_excess_objective_rel": info["excess_objective_rel"],
            "_excess_objective_local": info["excess_objective_local"],
            "_excess_objective_local_rel": info["excess_objective_local_rel"],
            "_rank_deficient": info["rank_deficient"],
            "_secs": round(time.time() - t0, 2),
        }
        # non-finite metrics are OMITTED, never NaN (federated_eval.py:1547-1551),
        # but WHICH ones were dropped is recorded so an absent column can never be
        # mistaken for a complete one.
        omitted = []
        for k_, v in flat.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                row[k_] = float(v)
            else:
                omitted.append(k_)
        for k_ in CORE_METRICS:
            if k_ not in row:
                omitted.append(k_)
        row["_metrics_omitted"] = sorted(set(omitted))
        if suite:
            row["_suite"] = suite
        rows.append(row)
        top1_key = f"paper_top1_acc_at_{row['_tolerance']}"
        print(f"    {job['arm']:34s} {e:14s} vus_pr={row.get('vus_pr', float('nan')):.4f} "
              f"top1={row.get(top1_key, float('nan')):.2f} ({row['_secs']:.1f}s)", flush=True)

    if info["param_relerr"] is not None:
        verdict = witness_verdict(info["param_relerr"], info["score_relerr"],
                                  job["witness_tol"], info["pooled_path"])
        srel = "n/a" if info["score_relerr"] is None else f"{info['score_relerr']:.2e}"
        print(f"  [witness] {job['cluster']}/{job['head']}: "
              f"param_relerr={info['param_relerr']:.2e} score_relerr={srel} "
              f"-> {verdict}", flush=True)
    print(f"  [job] {job['arm']} {job['cluster']}: {len(rows)} rows in "
          f"{time.time() - t_job:.0f}s", flush=True)
    return rows


def witness_verdict(param_rel, score_rel, tol: float, pooled_path=None) -> str:
    """A bit-exact zero FAILS: it means the two paths share an object, which is
    determinism of the pipeline, not exactness of the federation (§0.4).

    The reference must ALSO come from the independent materialised path. When
    `pooled_solve` falls back to `chunked_pooled` (which it does above
    floor_heads.CHUNK rows, i.e. on the long UCR series) it builds the reference by
    SUMMING the same per-client statistics `fed_exact` sums — so the comparison is
    the circularity §0.4 exists to forbid, dressed up as ~1e-13 instead of an exact
    zero. The path was recorded but never asserted; now it gates."""
    if pooled_path is not None and pooled_path != "independent":
        return f"FAIL(circular: pooled path is {pooled_path!r}, not independent)"
    if param_rel == 0.0:
        return "FAIL(circular: exact zero)"
    if param_rel > tol or (score_rel is not None and score_rel > tol):
        return "FAIL(above tolerance)"
    return "PASS"


# ═════════════════════════════════════════════════════════════════════════════
#   Self-test
# ═════════════════════════════════════════════════════════════════════════════

def selftest() -> int:
    import detect as D
    import data as D_data
    from data import TimeSeriesRecord
    ok = FH.selftest()
    print()
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)

    d = np.abs(FH.moving_average_paper(x, 10) - D._moving_average_paper(x, 10)).max()
    print(f"  [{'PASS' if d == 0 else 'FAIL'}] floor_heads.moving_average_paper is "
          f"BIT-IDENTICAL to detect's        max|Δ|={d:.2e}")
    ok &= d == 0

    # window enumeration must agree with the object detect actually builds
    class _MD:
        entity_id, dataset, feature_names = "e", "d", ["c"]
    rec = TimeSeriesRecord(X=rng.normal(size=(1000, 1)).astype(np.float32),
                           y=np.zeros(1000, dtype=np.int64), metadata=_MD())
    idx = D_data.SlidingWindowDataset([rec], 128, 13).indices
    mine = FH.window_starts(1000, 128, 13)
    same = len(idx) == len(mine) and all(a.start == b for a, b in zip(idx, mine))
    print(f"  [{'PASS' if same else 'FAIL'}] window_starts == SlidingWindowDataset.indices"
          f"                    n={len(mine)}")
    ok &= same

    print(f"\n  floor_eval selftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


# ═════════════════════════════════════════════════════════════════════════════
#   Writers
# ═════════════════════════════════════════════════════════════════════════════

def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep the LAST row per (arm, entity) — a rescore supersedes the old one.

    One exception: a row WITHOUT the nested `_suite` block never deletes one that
    has it. `--suite` is not part of the arm tag (it adds metrics, it never changes
    one), so a later `--suite off` pass over the same arm would otherwise silently
    drop the block the paper table depends on, and the loss is invisible — the row
    still looks complete."""
    by: dict = {}
    for r in rows:
        key = (r["_arm"], r["_entity"])
        prev = by.get(key)
        if prev is not None and "_suite" in prev and "_suite" not in r:
            r = {**r, "_suite": prev["_suite"]}
        by[key] = r
    return [by[k] for k in sorted(by)]


def rel_repo(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all", help="'all' or comma list, e.g. c0,c2")
    ap.add_argument("--heads", default="ma_c", help=f"comma list of {FH.HEAD_NAMES}")
    ap.add_argument("--modes", default="local", help=f"comma list of {ALL_MODES}, or 'all'")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"ma_c/ma_causal window; PRE-REGISTERED at {DEFAULT_K}")
    ap.add_argument("--ar-p", type=int, default=FH.DEFAULTS["ar_p"])
    ap.add_argument("--ar-lambda", type=float, default=FH.DEFAULTS["ar_lambda"])
    ap.add_argument("--pca-k", type=int, default=FH.DEFAULTS["pca_k"])
    ap.add_argument("--gamma", type=float, default=FH.DEFAULTS["gamma"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impulse", choices=["on", "off"], default="on")
    ap.add_argument("--window", type=int, default=None,
                    help="override cfg.dataset.window_length. MUST match the deep "
                         "run being compared against: it sets both the accumulation "
                         "geometry AND the impulse-term MA length.")
    ap.add_argument("--fit-stride", type=int, default=1,
                    help="stride for FITTING (deep training uses 1); scoring always "
                         "uses detect's eval stride")
    ap.add_argument("--cap-clients", type=int, default=2, help="central_capN")
    ap.add_argument("--mu", type=float, default=0.1, help="fed_prox")
    ap.add_argument("--tau", type=int, default=16, help="fed_localgd local steps")
    ap.add_argument("--rounds", type=int, default=30, help="fed_prox / fed_localgd")
    ap.add_argument("--eta", type=float, default=None, help="fed_localgd step (auto)")
    ap.add_argument("--witness", action="store_true",
                    help="prove fed_exact == central against an INDEPENDENT pooled path")
    ap.add_argument("--witness-tol", type=float, default=1e-9)
    ap.add_argument("--suite", choices=["on", "off"], default="on",
                    help="the nested metrics_core.evaluate_scores block")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out-dir", default="artifacts/floor")
    ap.add_argument("--force", action="store_true", help="rescore rows already on disk")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    bad = [h for h in heads if h not in FH.HEAD_NAMES]
    if bad:
        raise SystemExit(f"unknown head(s) {bad}; available: {FH.HEAD_NAMES}")
    modes = ALL_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    bad = [m for m in modes if m not in ALL_MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}; available: {ALL_MODES}")
    impulse = args.impulse == "on"

    for name in heads + modes:
        if name in OUT_OF_SCOPE:
            if name == "gauss" and args.dataset == "wsd_fed":
                continue
            print(f"[floor] ⚠ OUT OF PAPER SCOPE: {name!r} — {OUT_OF_SCOPE[name]}.\n"
                  f"        It will run (it is a legitimate ablation) but its rows are NOT "
                  f"part of the reported baseline. See FLOOR_RUNBOOK.md §1.")

    # Be a good neighbour: the neural sweeps are CPU-bound (~1 core each in the main
    # process), so the floor must leave them their cores.
    #
    # ⚠ CORRETTO 2026-08-01. This used to be `if live and jobs > 4: jobs = 4`, with the
    # comment "the 16-core CPU is the wall" -- g2's core count, hardcoded. Running on g4
    # (48 cores, 33 idle) it throttled the floor to 4 jobs while 60% of the machine sat
    # unused, turning a 7 h re-floor into a much longer one for no reason. The budget is now
    # derived from the ACTUAL host.
    #
    # Counting live jobs needs the parent trick: dataloader workers inherit the parent's
    # cmdline, so a plain `pgrep -f federated_eval.py` reports ~450 processes for ~12 runs.
    # A run is a process whose parent is not itself a federated_eval.py.
    ps = subprocess.run(["ps", "-eo", "pid,ppid,args"],
                        capture_output=True, text=True).stdout.splitlines()
    pids, parent = set(), {}
    for line in ps:
        f = line.split(None, 2)
        if len(f) == 3 and "federated_eval.py" in f[2] and "pgrep" not in f[2]:
            pids.add(f[0]); parent[f[0]] = f[1]
    live = sum(1 for p in pids if parent[p] not in pids)
    ncpu = os.cpu_count() or 4
    budget = max(1, ncpu - live - 2)          # -2: Leave shell breath and I/O
    jobs = args.jobs
    if jobs > budget:
        jobs = budget
        print(f"[floor] {live} run neurali su {ncpu} core -> capping --jobs at {jobs}")
    os.nice(19)

    from federated import resolve_clients
    cfg = build_cfg(args.dataset, impulse, args.window)
    default_window = build_cfg(args.dataset, impulse, None).dataset.window_length
    kn = knobs_from_args(args)
    all_heads = FH.build_heads(kn)

    meta_path = REPO / "data" / "raw" / args.dataset / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    clusters = (sorted(meta.get("clusters", {})) if args.clusters == "all"
                else [c.strip() for c in args.clusters.split(",")])
    unit = "cluster" if args.dataset.startswith("ucr_split") else "entity"

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"records_{args.dataset}.jsonl"
    existing = load_rows(rec_path)
    # Resume must account for the SUITE, even though `--suite` is deliberately not part of
    # the arm tag (it only ADDS the nested block, it never changes a value). Without the
    # second clause a `--suite on` pass over an arm whose rows were written `--suite off`
    # is skipped as already-done, so the block the paper table needs is never computed and
    # the log looks like a successful no-op. Caught by the smoke run on 2026-07-30.
    done = set() if args.force else {
        (r["_arm"], r["_entity"]) for r in existing
        if r.get("_schema") == SCHEMA and (args.suite == "off" or "_suite" in r)}

    # ── build the job list, skipping provably-null (head × mode) cells ──────
    # A build whose window is a PER-SERIES property (ucr_split_w2p: W = 2*period) ships the
    # map in its metadata. Honour it, so the floor cannot be silently scored at the config's
    # 128 on a dataset that was admitted under a different window -- an explicit --window
    # still wins, because overriding it is a legitimate ablation.
    meta_windows = meta.get("windows", {})
    if meta_windows and args.window is None:
        ws = sorted({int(v) for v in meta_windows.values()})
        print(f"[floor] {args.dataset} carries per-series windows "
              f"({len(meta_windows)} series, W in [{ws[0]}, {ws[-1]}]) -> using them per cluster")

    jobs_list, skipped = [], []
    for cl in clusters:
        ents = list(resolve_clients(cfg, None, cl))
        eff_window = args.window if args.window is not None else meta_windows.get(cl)
        tag_window = int(eff_window) if eff_window is not None else default_window
        for h in heads:
            head = all_heads[h]
            for m in modes:
                if head.zero_param and m != "local":
                    skipped.append((h, m, "zero fitted parameters -> arm-invariant: "
                                          "local == central == every federated mode"))
                    continue
                if m in MODE_REQUIRES and h not in MODE_REQUIRES[m]:
                    skipped.append((h, m, f"mode is only defined for {sorted(MODE_REQUIRES[m])}"))
                    continue
                if m in FED_MODES and len(ents) < 2:
                    skipped.append((h, m, f"cluster {cl} has {len(ents)} client(s)"))
                    continue
                extra = {}
                if m == "central_capN":
                    extra["cap"] = min(args.cap_clients, len(ents))
                if m == "fed_prox":
                    extra.update({"mu": args.mu, "rounds": args.rounds})
                if m == "fed_localgd":
                    extra.update({"tau": args.tau, "rounds": args.rounds})
                ktag = head_knobs(h, kn, m, extra)
                arm = arm_tag(h, m, ktag, impulse, tag_window, default_window,
                              1 if head.zero_param else args.fit_stride)
                todo = [e for e in ents if (arm, e) not in done]
                if not todo:
                    continue
                jobs_list.append({
                    "dataset": args.dataset, "cluster": cl, "entities": ents,
                    "head": h, "mode": m, "arm": arm, "knobs": kn, "knobs_tag": ktag,
                    "impulse": impulse, "window": eff_window, "unit": unit,
                    "fit_stride": args.fit_stride, "cap_clients": extra.get("cap", len(ents)),
                    "mu": args.mu, "tau": args.tau, "rounds": args.rounds, "eta": args.eta,
                    "witness": args.witness, "witness_tol": args.witness_tol,
                    "suite": args.suite == "on",
                })

    seen = set()
    for h, m, why in skipped:
        if (h, m) not in seen:
            seen.add((h, m))
            print(f"[floor] SKIP {h} x {m}: {why}")
    print(f"[floor] {args.dataset}: {len(clusters)} clusters, {len(jobs_list)} jobs, "
          f"heads={heads}, modes={modes}, impulse={args.impulse}, "
          f"fit_stride={args.fit_stride}, suite={args.suite}, jobs={jobs}, nice 19")
    # On a per-series build this used to print the CONFIG default (128) on the line right
    # after the banner announcing that the window comes from metadata — the two lines
    # contradicted each other and the doc quoted the first while the run obeyed the second.
    if meta_windows and args.window is None:
        _ws = sorted({int(v) for v in meta_windows.values()})
        _wtxt = f"per-series {_ws[0]}..{_ws[-1]} ({len(_ws)} distinct)"
    else:
        _wtxt = str(cfg.dataset.window_length)
    print(f"[floor] W={_wtxt} eval_stride={_stride(cfg)} "
          f"tolerance={cfg.evaluation.paper_metrics_tolerance} q={cfg.threshold.q}")
    if not jobs_list:
        print("[floor] nothing to do (use --force to rescore)")

    t0 = time.time()
    rows: list[dict] = []
    if jobs_list:
        with Pool(max(1, jobs)) as p:
            for sub in p.imap_unordered(run_job, jobs_list):
                rows.extend(sub)
                with rec_path.open("a") as fh:      # append as we go: crash-safe
                    for r in sub:
                        fh.write(json.dumps(r) + "\n")
        print(f"[floor] scored {len(rows)} rows in {time.time() - t0:.0f}s")

    # ── rewrite de-duplicated, then one summary JSON per arm ────────────────
    allrows = dedupe(load_rows(rec_path))
    rec_path.write_text("".join(json.dumps(r) + "\n" for r in allrows))

    witness = {}
    for arm in sorted({r["_arm"] for r in allrows}):
        sub = [r for r in allrows if r["_arm"] == arm]
        metric_keys = sorted({k for r in sub for k, v in r.items()
                              if not k.startswith("_") and isinstance(v, (int, float))})
        summary = {}
        for k_ in metric_keys:
            v = [r[k_] for r in sub if k_ in r]
            if v:
                summary[k_] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                               "std": float(np.std(v)), "worst": float(np.min(v)),
                               "n": len(v)}
        r0 = sub[0]
        payload = {
            "meta": {"dataset": args.dataset, "arm": arm, "head": r0["_head"],
                     "mode": r0["_mode"], "knobs": r0.get("_knobs"),
                     "unit": r0["_unit"], "window": r0["_window"],
                     "eval_stride": r0["_eval_stride"], "fit_stride": r0.get("_fit_stride"),
                     "tolerance": r0["_tolerance"], "threshold_q": cfg.threshold.q,
                     "threshold_rule": "quantile_fallback", "impulse": r0["_impulse"],
                     "n_clients": len(sub), "schema": r0.get("_schema"),
                     "fitted_parameters": 0 if r0["_mode"] == "arm_invariant" else None,
                     "note": ("zero fitted parameters -> arm-invariant: local == centralized "
                              "== every federated mode, bit-identically. No federation "
                              "question can be asked of this head."
                              if r0["_mode"] == "arm_invariant" else
                              "threshold rule is the quantile fallback: only the "
                              "threshold-free metrics are comparable to the deep arms.")},
            "summary": summary,
            "records": sub,
        }
        p_json = out_dir / f"{args.dataset}__{arm}.json"
        p_json.write_text(json.dumps(payload, indent=2))
        s = summary.get("vus_pr", {})
        if s:
            print(f"  {arm:36s} n={s['n']:4d}  vus_pr mean={s['mean']:.4f} "
                  f"median={s['median']:.4f}  -> {rel_repo(p_json)}")
        w = [r for r in sub if r.get("_param_relerr") is not None]
        if w:
            witness[arm] = {
                r["_cluster"]: {"param_relerr": r["_param_relerr"],
                                "score_relerr": r["_score_relerr"],
                                "pooled_path": r["_pooled_path"],
                                "verdict": witness_verdict(r["_param_relerr"],
                                                           r["_score_relerr"],
                                                           args.witness_tol,
                                                           r.get("_pooled_path"))}
                for r in w}
    if witness:
        wp = out_dir / f"witness_{args.dataset}.json"
        old = json.loads(wp.read_text()) if wp.exists() else {}
        old.update(witness)
        wp.write_text(json.dumps(old, indent=2))
        bad = [f"{a}/{c}" for a, d in witness.items() for c, v in d.items()
               if v["verdict"] != "PASS"]
        print(f"[floor] witness -> {rel_repo(wp)}   "
              + ("ALL PASS" if not bad else f"FAILURES: {bad}"))
    return 0


def _stride(cfg) -> int:
    import detect as D
    return D._resolve_eval_stride(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
