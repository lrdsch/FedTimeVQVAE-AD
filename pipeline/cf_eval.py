"""
=============================================================================
  cf_eval.py — Counterfactual quality evaluation for the trained model.
=============================================================================

Goal: measure whether the counterfactuals produced by stage2.counterfactual()
are valid, sparse, plausible, locally non-degrading, and *faithful* to the
detector's decision (top-k surprise editing must beat random / bottom-k).

Pipeline (single entry point: `main()` / `python pipeline/cf_eval.py`):

  1. Load stage1 + stage2 from the run dir of the active config.
  2. Score the train split with the canonical detect.py pipeline and fit
     the paper-style threshold (per-τ per-(C,F) accumulator).
  3. Walk the test split window-by-window, find the TRUE POSITIVE windows
     (label intersects an anomaly AND the window's ASSEMBLED score — the max
     over the window of detect's overlap-summed per-timestep score — exceeds
     the threshold). Scoring windows in ISOLATION would put them on a ~K×
     smaller scale than the threshold (K ≈ window_length / eval_stride ≈ 10),
     starving the TP set; reusing the assembled per-timestep scores keeps the
     gate on detect's own scale. These are the only windows on which a
     counterfactual is meaningful.
  4. For every TP window, for every q in `cfg.cf_eval.quantiles`, build
     three counterfactuals on the SAME prior:
       - topk_surprise:  the model's own selection (high prior NLL)
       - random:         |M_top| tokens chosen uniformly at random
       - bottomk_surprise: |M_top| tokens with the LOWEST prior NLL
     The three masks have identical sizes — the only thing that changes
     is *which* tokens get rewritten.
  5. For each (window, q, variant), compute the 5 metric blocks:
       (1) validity            — region-level: repair EVERY flagged window of an
                                 anomaly event and check the event drops below
                                 threshold (validity_window = single-window flip,
                                 kept for transparency; see _inject_region_validity)
       (2) sparsity-3-levels   — fraction of tokens / channels / timesteps changed
       (3) plausibility-NLL    — prior NLL of the rewritten tokens
       (4) non-degradation     — Δ-NLL on UNMASKED tokens; MAE on UNCHANGED time/ch
       (5) faithfulness        — Δscore_top vs Δscore_random vs Δscore_bottom

  6. Save per_window.csv (one row per (window, q, variant)) and a small
     report.json with means/stds grouped by (q, variant).

DTW (Dynamic Time Warping) — only computed at q=0 and only if --with_dtw
is passed. At every other q the unmasked-token context anchors the temporal
phase, making DTW numerically equivalent to MAE/RMSE (verified empirically).

This file imports — does NOT duplicate — the scoring primitives from
detect.py (`_combine_scores`, `_resolve_eval_stride`, `_fit_threshold_paper`,
`_score_split`, `_finalize_entities`). The TP gate and CF re-scoring both run on
detect's assembled per-timestep scale (via `_score_split` + `_finalize_entities`)
so they share the TRAIN-fitted threshold's scale. Counterfactual generation goes
through the patched `stage2.counterfactual()` which now accepts an optional
`token_mask` (used here for random / bottom-k).

Output layout (under `artifacts/cf_eval/<run_name>/`):
  per_window.csv   — wide table, one row per (window_id, q, variant)
  report.json      — aggregated stats per (q, variant)
  config.json      — the full cfg used for this evaluation (reproducibility)
"""

# %%
from __future__ import annotations

# repo root on sys.path so this pipeline/ script can import the shared
# libs (config / data / utils / metrics_core) that live at the project root.
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from lib.proctitle import set_process_title  # noqa: E402

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import Config, load_config
from data import RecordMetadata, TimeSeriesRecord, make_dataloaders
from detect import (
    _combine_scores,
    _finalize_entities,
    _fit_threshold_paper,
    _resolve_eval_stride,
    _score_split,
)
from stage1 import load_stage1
from stage2 import Stage2System, counterfactual, load_stage2
from utils import best_checkpoint, configure_logging, resolve_path, run_dir_for, seed_everything, resolve_device


# Per-window, per-channel scoring (assembled-scale building blocks)
#
#   detect.py assembles overlapping windows by SUM (no division): each interior
#   timestep's score is the sum of the per-window scores covering it, so the
#   TRAIN-fitted threshold lives on that ~K-fold-summed scale (K ≈ window_length
#   / eval_stride ≈ 10). Scoring a window in ISOLATION (single window, no overlap
#   sum) puts its peak on a ~K× smaller scale, so comparing it against the
#   summed-scale threshold is a scale mismatch — it was starving the TP set
#   (only 9/100 entities produced any TP window even when the detector itself
#   reported recall ≈ 1). We therefore gate TPs and re-score counterfactuals on
#   detect's ASSEMBLED per-timestep score (`entity["overall_scores"]`, reproduced
#   by detect._score_split). The helpers below compute a single window's
#   per-channel CONTRIBUTION so a counterfactual edit can be spliced back into
#   the summed accumulator and re-finalized on the same scale.
# -----------------------------------------------------------------------------

@torch.no_grad()
def _window_perchannel_batch(
    stage1, stage2: Stage2System | None, cfg: Config, x: torch.Tensor,
) -> np.ndarray:
    """(B, C, T) → (B, C, T) per-channel PRE-impulse, pre-aggregation score.

    This is the exact quantity detect._compute_entities_raw adds into the
    per-(T, C) rolling accumulator for each window (`_combine_scores(s_local,
    s_prior)`), BEFORE the impulse moving-average and channel aggregation that
    detect._finalize_entities applies on the assembled series. Splicing the
    delta of two of these (CF − original) into the accumulator and re-finalizing
    yields the counterfactual's score on the same scale as the threshold.
    """
    s1_out = stage1(x)
    s_local = (x - s1_out["reconstructed"]).pow(2)                    # (B, C, T)

    if stage2 is not None:
        prior_out = stage2.score_batch({"inputs": x}, per_rate=False)
        ts = prior_out.token_scores                                    # (B, C, F, W)
        assert ts.ndim == 4, f"expected (B,C,F,W) from prior, got {ts.shape}"
        B_, C_, F_, W_ = ts.shape
        flat = ts.reshape(B_ * C_, F_, W_)
        flat_T = F.interpolate(flat, size=x.shape[-1], mode="nearest")  # (B*C, F, T)
        s_prior = flat_T.mean(dim=1).reshape(B_, C_, x.shape[-1])       # paper mean_F
    else:
        s_prior = torch.zeros_like(s_local)

    per_ch = _combine_scores(s_local, s_prior, cfg)                   # (B, C, T)
    return per_ch.detach().cpu().numpy()                             # (B, C, T)


def _window_perchannel(
    stage1, stage2: Stage2System | None, cfg: Config, x: torch.Tensor,
) -> np.ndarray:
    """Single-window convenience: (1, C, T) → (T, C) per-channel pre-impulse."""
    assert x.shape[0] == 1, "expected a single window"
    return _window_perchannel_batch(stage1, stage2, cfg, x)[0].T      # (T, C)


def _assembled_window_score(
    entity: dict[str, Any], cfg: Config, start: int, stop: int,
    delta_perch: np.ndarray | None = None,
) -> float:
    """Max over [start:stop) of the ASSEMBLED per-timestep score (threshold scale).

    `entity` is a detect._score_split test entity. With `delta_perch=None` this
    returns the original window score = max of `entity["overall_scores"]` on the
    slice. With `delta_perch` (shape (stop-start, C)) it patches THIS window's
    per-channel contribution into the raw summed accumulator and re-finalizes
    (impulse + channel aggregation) exactly like detect._finalize_entities, so a
    counterfactual edit is scored on detect's own scale. When `delta_perch` is
    all zeros the result equals the original score by construction.

    The re-finalize runs on a PADDED slice [start-W : stop+W] rather than the
    whole series: detect's centered impulse moving-average (window W) at any
    target timestep j∈[start,stop) only reads raw[j-W/2 : j+W/2], so a ±W margin
    reproduces the full-series result exactly for the target slice (verified at
    interior and series-boundary windows) while keeping the cost O(W) instead of
    O(T) — important on the 32k-length series.
    """
    if delta_perch is None:
        return float(np.asarray(entity["overall_scores"])[start:stop].max())
    # Raw summed per-(T, C) accumulator: `channel_scores_raw` when the impulse
    # term is on (detect overwrites `channel_scores`), else `channel_scores`.
    raw_full = np.asarray(entity.get("channel_scores_raw", entity["channel_scores"]))
    W = int(cfg.dataset.window_length)
    lo = max(0, start - W)
    hi = min(raw_full.shape[0], stop + W)
    raw = raw_full[lo:hi].copy()
    raw[start - lo: stop - lo] += delta_perch
    tmp: dict[str, Any] = {"channel_scores": raw, "labels": None}
    _finalize_entities([tmp], cfg)
    sub = np.asarray(tmp["overall_scores"])[start - lo: stop - lo]
    return float(sub.max())


# True-positive window collection

def _collect_tp_windows(
    cfg: Config,
    test_records: list[TimeSeriesRecord], test_entities: list[dict[str, Any]],
    threshold: float, max_windows: int | None, stride_subsample: int,
) -> list[dict[str, Any]]:
    """Walk test windows and return TPs as a list of {x, score_orig, ...}.

    A window is a TRUE POSITIVE iff:
      * its label slice contains at least one positive (anomaly intersects), AND
      * its window-level score on detect's ASSEMBLED per-timestep scale (the max
        over the window of `entity["overall_scores"]`, the overlap-summed score
        detect thresholds) exceeds the paper-style threshold fit on TRAIN.

    The gate reuses the already-assembled test scores (`test_entities` come from
    detect._score_split, same scoring pass the headline detector uses), so it is
    pure-numpy and lives on exactly the threshold's scale — no separate, scale-
    mismatched per-window forward pass. `test_entities[ri]` aligns with
    `test_records[ri]` (both built in record order by detect).

    `stride_subsample` further down-samples the rolling step (set >1 to
    avoid near-duplicate adjacent TPs from a long anomaly event).
    `max_windows` caps the total returned TPs (None = no cap).
    """
    eval_stride = _resolve_eval_stride(cfg) * max(1, int(stride_subsample))
    W = cfg.dataset.window_length
    tps: list[dict[str, Any]] = []

    for ri, rec in enumerate(test_records):
        if rec.y is None:
            continue
        T = rec.X.shape[0]
        if T < W:
            continue
        starts = list(range(0, T - W + 1, eval_stride))
        if not starts:
            continue
        overall = np.asarray(test_entities[ri]["overall_scores"])             # (T,)

        for s in starts:
            lab = rec.y[s: s + W]
            if not bool(lab.max() > 0):                                       # no anomaly
                continue
            score_orig = float(overall[s: s + W].max())                      # assembled scale
            if not (score_orig > threshold):
                continue
            xs = rec.X[s: s + W].T[None, ...]                                 # (1, C, T)
            yc_win = None if rec.y_channel is None else rec.y_channel[s: s + W]
            tps.append({
                "record_idx": ri,
                "start": int(s),
                "stop": int(s + W),
                "score_orig": score_orig,
                "x": torch.from_numpy(np.ascontiguousarray(xs[0])).float(),   # (C, T)
                "y_channel": yc_win,                                          # (W, C) int or None
            })
            if max_windows is not None and len(tps) >= max_windows:
                return tps
    return tps


# ─── Mask construction — three variants on the same scores tensor ────────────

def _topk_mask(scores: torch.Tensor, q: float, granularity: str) -> torch.Tensor:
    """Replicate stage2.counterfactual's selection logic. Returns (B,C,F,W) bool.

    Used here only when we want to know |M_top| in advance so the random
    and bottom-k variants can be matched in size. The actual top-k CF
    generation can also use this mask via `counterfactual(token_mask=...)`.
    """
    B, C, F_, W = scores.shape
    if granularity == "column":
        col = scores.mean(dim=2)                                              # (B, C, W)
        thr = torch.quantile(col.reshape(B, -1), q, dim=1).view(B, 1, 1)
        col_mask = col > thr
        return col_mask.unsqueeze(2).expand(-1, -1, F_, -1).contiguous()
    if granularity == "token":
        thr = torch.quantile(scores.reshape(B, -1), q, dim=1).view(B, 1, 1, 1)
        return (scores > thr).contiguous()
    raise ValueError(f"granularity must be 'column' or 'token', got {granularity!r}")


def _random_mask_matched(target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Same per-sample |M| as `target`, but positions chosen uniformly at random.

    `generator` lives on CPU; we draw the perm on CPU and move it to
    `target.device` before scattering. Matching |M_top| per sample is what
    makes the faithfulness comparison apples-to-apples.
    """
    B = target.shape[0]
    flat = target.reshape(B, -1)
    out = torch.zeros_like(flat, dtype=torch.bool)
    N = flat.shape[1]
    for b in range(B):
        k = int(flat[b].sum().item())
        if k <= 0:
            continue
        perm = torch.randperm(N, generator=generator).to(out.device)
        out[b, perm[:k]] = True
    return out.reshape(target.shape)


def _bottom_mask_matched(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Same per-sample |M| as `target`, but pick tokens with LOWEST NLL."""
    B = scores.shape[0]
    flat_scores = scores.reshape(B, -1)
    flat_target = target.reshape(B, -1)
    out = torch.zeros_like(flat_target, dtype=torch.bool)
    for b in range(B):
        k = int(flat_target[b].sum().item())
        if k <= 0:
            continue
        idx = torch.topk(flat_scores[b], k=k, largest=False).indices
        out[b, idx] = True
    return out.reshape(scores.shape)


# Per-window metric block (the 5 deliverables)

@torch.no_grad()
def _compute_metrics(
    stage1, stage2: Stage2System, cfg: Config,
    x: torch.Tensor, x_cf: torch.Tensor,
    token_mask: torch.Tensor, scores_orig: torch.Tensor,
    score_orig: float, score_cf: float, threshold: float,
    delta_input_rel: float = 1e-2,
    with_dtw: bool = False,
    y_channel_window: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the 5 metric blocks for a single (x, x_cf) pair.

    Args:
      x, x_cf       : (B=1, C, T) — single window for clarity.
      token_mask    : (1, C, F, W) bool — which tokens were rewritten.
      scores_orig   : (1, C, F, W) — per-token NLL of x under the prior.
      score_orig    : float        — pre-CF window score on the assembled scale.
      score_cf      : float        — post-CF window score on the assembled scale,
                      computed by the caller (single-window splice via
                      _assembled_window_score). validity_window = score_cf <
                      threshold. NOTE: single-window validity is ~0 for strong
                      anomalies because detect SUMS ~K overlapping windows — see
                      _inject_region_validity for the region-level metric the
                      paper reports.
      threshold     : float        — paper-style threshold on TRAIN.
      delta_input_rel: float       — channel-relative tolerance for "changed
                                     timestep" definition (1% of channel std).
      with_dtw      : bool         — compute DTW (only meaningful at q=0).
    """
    assert x.shape[0] == 1 and x_cf.shape[0] == 1, "metrics expect a single window"
    diff = (x - x_cf).abs()                                               # (1, C, T)
    C, T = x.shape[1], x.shape[2]

    # (1) Δscore + single-window validity. score_cf is computed by the caller
    # (single-window splice into the assembled accumulator). delta_score drives
    # faithfulness (which tokens matter); validity_window is the single-window
    # flip — kept for transparency, but it is ~0 for strong anomalies because
    # detect SUMS ~K overlapping windows (see _inject_region_validity for the
    # region-level metric the paper reports).
    out: dict[str, float] = {
        "score_orig":      score_orig,
        "score_cf":        score_cf,
        "delta_score":     score_orig - score_cf,
        "validity_window": float(score_cf < threshold),
    }

    # (2) Sparsity — three resolutions.
    out["sparsity_token"]   = float(token_mask.float().mean().item())
    # Per-channel std on x (post-scaling). Used as a tolerance for "changed".
    ch_std = x[0].std(dim=-1).clamp_min(1e-6).cpu().numpy()               # (C,)
    diff_np = diff[0].cpu().numpy()                                       # (C, T)
    tol = (delta_input_rel * ch_std)[:, None]                             # (C, 1)
    changed_ct = diff_np > tol                                            # (C, T) bool
    ch_changed = changed_ct.any(axis=1)                                   # (C,)
    t_changed  = changed_ct.any(axis=0)                                   # (T,)
    # "What fraction of channels did we touch?" only has resolution above C=1,
    # where it can only be 0.0 or 1.0.
    out["sparsity_channel"] = float(ch_changed.mean()) if C > 1 else float("nan")
    out["sparsity_time"]    = float(t_changed.mean())

    # (3) Plausibility — NLL of the CF tokens under the prior.
    _, cf_indices, _ = stage1.encode_tokens(x_cf)
    cf_indices = cf_indices.long().reshape(1, -1)
    cf_scores = stage2.prior.score_tokens(cf_indices).reshape(1, -1)      # (1, C*F*W)
    orig_flat = scores_orig.reshape(1, -1)
    out["plausibility_nll_cf"]   = float(cf_scores.mean().item())
    out["plausibility_nll_orig"] = float(orig_flat.mean().item())
    out["plausibility_delta"]    = float((cf_scores - orig_flat).mean().item())
    # Smoothness proxy: 2nd-derivative magnitude (per channel, then mean).
    if T >= 3:
        d2_cf = (x_cf[0, :, 2:] - 2 * x_cf[0, :, 1:-1] + x_cf[0, :, :-2]).abs().mean()
        d2_x  = (x   [0, :, 2:] - 2 * x   [0, :, 1:-1] + x   [0, :, :-2]).abs().mean()
        out["smoothness_delta"] = float((d2_cf - d2_x).item())
    else:
        out["smoothness_delta"] = float("nan")

    # (4) Non-degradation — context (UNMASKED tokens) should not get worse.
    mask_flat = token_mask.reshape(1, -1).bool()
    unmasked = ~mask_flat
    if int(unmasked.sum().item()) > 0:
        out["non_degradation_nll"] = float(
            (cf_scores[unmasked] - orig_flat[unmasked]).mean().item()
        )
    else:
        out["non_degradation_nll"] = float("nan")
    # Input-space non-degradation: mean abs diff on the UNCHANGED (c, t) cells.
    unchanged_ct = ~changed_ct
    if unchanged_ct.any():
        out["non_degradation_input_mae"] = float(diff_np[unchanged_ct].mean())
    else:
        out["non_degradation_input_mae"] = float("nan")

    # (5) Proximity — MAE / RMSE always; DTW optional, only at q=0 from caller.
    out["mae"]  = float(diff_np.mean())
    out["rmse"] = float(np.sqrt((diff_np ** 2).mean()))
    if with_dtw:
        try:
            from dtaidistance import dtw as _dtw  # lazy
            d = 0.0
            for c in range(C):
                d += _dtw.distance_fast(
                    x[0, c].cpu().numpy().astype(np.double),
                    x_cf[0, c].cpu().numpy().astype(np.double),
                )
            out["dtw"] = float(d / C)
        except ImportError:
            out["dtw"] = float("nan")
    else:
        out["dtw"] = float("nan")

    # ── Channel-targeting (ADDITIVE — only when y_channel exists AND C >= 2) ──
    # Did the top-k token edit target the GROUND-TRUTH anomalous channels?
    # Projects `token_mask (1, C, F, W)` over (F, W) to a per-channel bool,
    # then compares against the GT-positive channels in this window slice.
    #
    # Skipped at C=1: `edited_ch` and `gt_ch` are both length-1, so precision and
    # recall are 1.0 whenever the mask touches anything and the window has any GT
    # positive — and NaN otherwise. There is no third outcome and no information.
    if y_channel_window is not None and C > 1:
        edited_ch = token_mask[0].any(dim=(1, 2)).cpu().numpy()           # (C,) bool
        gt_ch     = (np.asarray(y_channel_window) > 0).any(axis=0)        # (C,) bool
        n_edit = int(edited_ch.sum())
        n_gt   = int(gt_ch.sum())
        if n_edit > 0 and n_gt > 0:
            hits = int((edited_ch & gt_ch).sum())
            out["channel_precision_topk"] = hits / n_edit
            out["channel_recall_topk"]    = hits / n_gt
        else:
            out["channel_precision_topk"] = float("nan")
            out["channel_recall_topk"]    = float("nan")

    return out


# ─── Main loop ────────────────────────────────────────────────────────────────

def evaluate(
    cfg: Config | None = None,
    stage1_ckpt: str | Path | None = None,
    stage2_ckpt: str | Path | None = None,
    output_dir: Path | None = None,
    *,
    quantiles: tuple[float, ...] = (0.99, 0.95, 0.90, 0.80, 0.50, 0.0),
    granularity: str = "column",
    max_windows: int | None = None,
    stride_subsample: int = 1,
    with_dtw: bool = False,
    seed_offset: int = 0,
    cf_mode: str = "iterative",
) -> dict[str, Any]:
    """Run the full counterfactual evaluation. Returns the report dict."""
    configure_logging()
    cfg = cfg or load_config()
    seed_everything(cfg.seed + seed_offset)
    torch.set_float32_matmul_precision("high")

    # Resolve checkpoints (same logic as detect.detect())
    stage1_ckpt = Path(stage1_ckpt) if stage1_ckpt else best_checkpoint(cfg, "stage1")
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {stage1_ckpt}")
    stage2_ckpt = Path(stage2_ckpt) if stage2_ckpt else best_checkpoint(cfg, "stage2")
    if not stage2_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint missing: {stage2_ckpt}. cf_eval requires the prior."
        )

    device = resolve_device()

    # ── Data ────────────────────────────────────────────────────────────────
    data = make_dataloaders(cfg, stage="eval")
    example_inputs = next(iter(data.train_loader))["inputs"][:1].cpu()

    # ── Models ──────────────────────────────────────────────────────────────
    print(f"[cf_eval] loading stage1: {stage1_ckpt}")
    stage1 = load_stage1(stage1_ckpt, cfg, example_inputs, device=device)
    print(f"[cf_eval] loading stage2: {stage2_ckpt}")
    stage2 = load_stage2(
        stage2_ckpt, cfg, stage1_ckpt=stage1_ckpt,
        stage1_example_inputs=example_inputs, device=device,
    )

    # Threshold from tow (paper-style)
    print("[cf_eval] scoring train split + fitting paper-style threshold...")
    _, _, train_entities = _score_split(
        stage1, stage2, cfg, data.train_records, record_per_rate=True,
    )
    threshold = _fit_threshold_paper(train_entities, cfg)
    print(f"[cf_eval] threshold = {threshold:.6g}")

    # ── Score the test split on detect's ASSEMBLED scale ───────────────────
    # Same scoring pass the headline detector uses, so the TP gate and the CF
    # re-scoring share the threshold's overlap-summed scale (see module docstring
    # step 3 + `_assembled_window_score`). Scoring windows in isolation here is
    # what previously starved the TP set to 9/100 entities.
    print("[cf_eval] scoring test split (assembled scale)...")
    _, _, test_entities = _score_split(stage1, stage2, cfg, data.test_records)

    # ── Find true positive windows on the test split ───────────────────────
    print("[cf_eval] collecting true-positive windows on test...")
    tps = _collect_tp_windows(
        cfg, data.test_records, test_entities, threshold,
        max_windows=max_windows, stride_subsample=stride_subsample,
    )
    print(f"[cf_eval] found {len(tps)} TP windows")
    if not tps:
        report = {
            "dataset_name": cfg.dataset.name,
            "entity_id":    cfg.dataset.entity_id,
            "threshold":    float(threshold),
            "n_tp_windows": 0,
            "skipped_reason": "no true-positive windows on the test split",
        }
        if output_dir is None:
            output_dir = _default_output_dir(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    # Sweep over q × variant
    rng = torch.Generator(device="cpu").manual_seed(cfg.seed + 1234)
    rows: list[dict[str, Any]] = []
    variants = ("topk_surprise", "random", "bottomk_surprise")
    # (window_id, q, variant) -> (W, C) per-channel CF score delta on the
    # assembled scale. Stashed so _inject_region_validity can repair ALL flagged
    # windows of an anomaly event at once (the validity the paper reports).
    deltas: dict[tuple[int, float, str], np.ndarray] = {}

    for wi, tp in enumerate(tps):
        x = tp["x"].unsqueeze(0).to(device)                                # (1, C, T)
        entity = test_entities[tp["record_idx"]]
        # Original window's per-channel pre-impulse contribution — the baseline
        # the CF re-score splices its delta against. Computed ONCE per window so
        # a no-op CF reproduces score_orig exactly. (assembled scale)
        perch_orig = _window_perchannel(stage1, stage2, cfg, x)            # (T, C)
        # Per-token NLL of the original window — needed to build all 3 masks
        # and as the plausibility reference. Computed ONCE per window.
        _, orig_idx, latent_spatial = stage1.encode_tokens(x)
        orig_idx = orig_idx.long().reshape(1, -1)
        scores_orig = stage2.prior.score_tokens(orig_idx)                  # (1,C,F,W)

        for q in quantiles:
            top_mask = _topk_mask(scores_orig, q, granularity)             # (1,C,F,W)
            n_masked = int(top_mask.sum().item())
            if n_masked == 0:
                # Skip: q=1 or degenerate. Record as skipped.
                rows.append({
                    "window_id": wi, "record_idx": tp["record_idx"],
                    "start": tp["start"], "stop": tp["stop"],
                    "q": q, "variant": "skipped", "n_masked": 0,
                })
                continue

            for variant in variants:
                if variant == "topk_surprise":
                    mask = top_mask
                elif variant == "random":
                    mask = _random_mask_matched(top_mask, rng).to(device)
                else:  # "bottomk_surprise"
                    mask = _bottom_mask_matched(scores_orig, top_mask).to(device)

                # greedy=True → deterministic decode, so the only thing that
                # differs across topk/random/bottom variants is the mask, not
                # sampling noise. Keeps the A/B comparison fair.
                # adaptive_steps=False pins cf_eval to fixed prior.T steps so the
                # ablation numbers stay comparable to prior runs (and so the only
                # thing differing across topk/random/bottom is the mask, not the
                # step count). Flip to True to evaluate the faithful generator.
                cf = counterfactual(stage2, x, token_mask=mask, mode=cf_mode,
                                    greedy=True, adaptive_steps=False)
                x_cf = cf["x_cf"]                                          # (1, C, T)

                # Per-channel CF score delta on the assembled scale. score_cf is
                # this window's single-window flip; the delta is also stashed so
                # _inject_region_validity can repair every flagged window of the
                # event at once (per-channel deltas add in the accumulator exactly
                # as detect summed the originals).
                perch_cf = _window_perchannel(stage1, stage2, cfg, x_cf)  # (W, C)
                delta_perch = (perch_cf - perch_orig).astype(np.float32)  # (W, C)
                score_cf = _assembled_window_score(
                    entity, cfg, tp["start"], tp["stop"], delta_perch=delta_perch,
                )
                deltas[(wi, q, variant)] = delta_perch

                metrics = _compute_metrics(
                    stage1, stage2, cfg, x, x_cf,
                    token_mask=mask, scores_orig=scores_orig,
                    score_orig=tp["score_orig"], score_cf=score_cf, threshold=threshold,
                    with_dtw=(with_dtw and q == 0.0),
                    y_channel_window=tp.get("y_channel"),
                )
                rows.append({
                    "window_id": wi, "record_idx": tp["record_idx"],
                    "start": tp["start"], "stop": tp["stop"],
                    "q": q, "variant": variant, "n_masked": n_masked,
                    **metrics,
                })

        if (wi + 1) % 20 == 0:
            print(f"[cf_eval] processed {wi + 1}/{len(tps)} windows")

    # ── Persist ────────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = _default_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_per_window_csv(output_dir / "per_window.csv", rows)
    report = _aggregate_report(rows, cfg, threshold)
    _inject_region_validity(
        report, deltas, tps, data.test_records, test_entities, cfg,
        threshold, tuple(quantiles), variants,
    )
    report["cf_mode"] = cf_mode
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    _plot_pareto_validity_sparsity(report, output_dir)
    _plot_faithfulness(report, output_dir)
    print(f"[cf_eval] per_window.csv: {(output_dir / 'per_window.csv').resolve()}")
    print(f"[cf_eval] report.json:    {(output_dir / 'report.json').resolve()}")
    print(f"[cf_eval] plots:          {(output_dir / 'pareto_validity_sparsity.png').resolve()}")
    print(f"[cf_eval] plots:          {(output_dir / 'faithfulness.png').resolve()}")

    # ── Console summary ────────────────────────────────────────────────────
    for variant in variants:
        print(f"  [{variant}]")
        for q in quantiles:
            key = f"{variant}|q={q}"
            agg = report["by_q_variant"].get(key)
            if agg is None or agg.get("n", 0) == 0:
                continue
            print(f"    q={q:.2f}  "
                  f"validity_region={agg.get('validity_region_mean', float('nan')):.3f} "
                  f"validity_window={agg.get('validity_window_mean', float('nan')):.3f} "
                  f"Δscore={agg['delta_score_mean']:.4g} "
                  f"sparsity_token={agg['sparsity_token_mean']:.3f} "
                  f"plausibility_nll_cf={agg['plausibility_nll_cf_mean']:.3g} "
                  f"non_degradation_nll={agg['non_degradation_nll_mean']:.3g}")
    print("[cf_eval] faithfulness gap (Δscore_top − Δscore_random) per q:")
    for q in quantiles:
        gap = report["faithfulness_gap"].get(f"q={q}")
        if gap is not None:
            print(f"    q={q:.2f}  top−random={gap['top_minus_random']:.4g} "
                  f"top−bottom={gap['top_minus_bottom']:.4g}")

    return report


# ─── Plotting ────────────────────────────────────────────────────────────────
#   Two figures, both reading from the aggregated `report` dict so they are
#   reproducible from the persisted JSON without re-scoring anything.
# -----------------------------------------------------------------------------

_VARIANT_STYLE = {
    "topk_surprise":    {"color": "C0", "marker": "o", "label": "top-k surprise"},
    "random":           {"color": "C7", "marker": "s", "label": "random"},
    "bottomk_surprise": {"color": "C3", "marker": "x", "label": "bottom-k surprise"},
}


def _plot_pareto_validity_sparsity(report: dict[str, Any], output_dir: Path) -> None:
    """One point per (q, variant): X=mean sparsity_token, Y=region-level validity.

    Reads the curve directly from `report["by_q_variant"]`. The expected
    visual signature is: top-k dominates random which dominates bottom-k
    at every sparsity level (higher Y for the same X).
    """
    by_qv = report.get("by_q_variant", {})
    if not by_qv:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, style in _VARIANT_STYLE.items():
        pts = []
        for key, agg in by_qv.items():
            v_name, q_part = key.split("|q=")
            if v_name != variant:
                continue
            sx = agg.get("sparsity_token_mean")
            sy = agg.get("validity_region_mean")
            if sx is None or sy is None:
                continue
            pts.append((float(q_part), float(sx), float(sy)))
        if not pts:
            continue
        pts.sort(key=lambda p: p[1])                                      # by sparsity
        _, xs, ys = zip(*pts)
        ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=8)
        for q, x, y in pts:
            ax.annotate(f"q={q:g}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=style["color"])
    ax.set_xlabel("Sparsity (fraction of tokens rewritten)")
    ax.set_ylabel("Validity (fraction of anomaly events repaired below threshold)")
    ax.set_title(
        f"Pareto: Validity vs Sparsity — "
        f"{report.get('dataset_name')} / {report.get('entity_id')}"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_validity_sparsity.png", dpi=120)
    plt.close(fig)


def _plot_faithfulness(report: dict[str, Any], output_dir: Path) -> None:
    """Δscore vs q, one line per variant.

    The expected signature is top-k ≫ random > bottom-k at every q. If
    top-k and random sit on top of each other, the model's selection is
    NOT acting as an explanation — it's the decoder doing the work.
    """
    faith = report.get("faithfulness_gap", {})
    if not faith:
        return
    qs = sorted(float(k.split("=")[1]) for k in faith.keys())
    series = {v: [] for v in _VARIANT_STYLE}
    keymap = {
        "topk_surprise":    "delta_score_top",
        "random":           "delta_score_random",
        "bottomk_surprise": "delta_score_bottom",
    }
    for q in qs:
        block = faith[f"q={q}"]
        for v in _VARIANT_STYLE:
            series[v].append(block.get(keymap[v], float("nan")))

    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, style in _VARIANT_STYLE.items():
        ax.plot(qs, series[variant], color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=8)
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Quantile q  (smaller q = MORE tokens rewritten)")
    ax.set_ylabel("Mean Δscore = score(x) − score(x_cf)")
    ax.set_title(
        f"Faithfulness — {report.get('dataset_name')} / {report.get('entity_id')}"
    )
    ax.invert_xaxis()                                                     # left = aggressive edit
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "faithfulness.png", dpi=120)
    plt.close(fig)


# ─── Output helpers ──────────────────────────────────────────────────────────

def _default_output_dir(cfg: Config) -> Path:
    base = resolve_path(cfg.paths.runs).parent / "cf_eval"
    rel = run_dir_for(cfg, "stage1").relative_to(resolve_path(cfg.paths.runs) / "stage1")
    return (base / rel).resolve()


def _write_per_window_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _aggregate_report(
    rows: list[dict[str, Any]], cfg: Config, threshold: float,
) -> dict[str, Any]:
    """Mean / std grouped by (q, variant), plus faithfulness gaps per q."""
    metric_keys = [
        "validity_window", "delta_score",
        "sparsity_token", "sparsity_channel", "sparsity_time",
        "plausibility_nll_cf", "plausibility_nll_orig", "plausibility_delta",
        "smoothness_delta",
        "non_degradation_nll", "non_degradation_input_mae",
        "mae", "rmse", "dtw",
        # Channel-targeting — present only when y_channel was supplied.
        "channel_precision_topk", "channel_recall_topk",
    ]
    by_qv: dict[str, dict[str, float]] = {}
    for variant in ("topk_surprise", "random", "bottomk_surprise"):
        for q in sorted({r["q"] for r in rows if r.get("variant") == variant}):
            sel = [r for r in rows if r.get("variant") == variant and r["q"] == q]
            if not sel:
                continue
            agg: dict[str, float] = {"n": float(len(sel))}
            for k in metric_keys:
                vals = [r[k] for r in sel if k in r and r[k] is not None
                        and not (isinstance(r[k], float) and (np.isnan(r[k])))]
                if vals:
                    agg[f"{k}_mean"] = float(np.mean(vals))
                    agg[f"{k}_std"]  = float(np.std(vals))
            by_qv[f"{variant}|q={q}"] = agg

    # Faithfulness: per q, mean Δscore for each variant, and the gaps.
    faith: dict[str, dict[str, float]] = {}
    for q in sorted({r["q"] for r in rows if r.get("variant") in {"topk_surprise", "random", "bottomk_surprise"}}):
        means: dict[str, float] = {}
        for v in ("topk_surprise", "random", "bottomk_surprise"):
            ds = [r["delta_score"] for r in rows
                  if r.get("variant") == v and r["q"] == q and "delta_score" in r]
            means[v] = float(np.mean(ds)) if ds else float("nan")
        faith[f"q={q}"] = {
            "delta_score_top":    means["topk_surprise"],
            "delta_score_random": means["random"],
            "delta_score_bottom": means["bottomk_surprise"],
            "top_minus_random":   means["topk_surprise"] - means["random"],
            "top_minus_bottom":   means["topk_surprise"] - means["bottomk_surprise"],
        }

    return {
        "dataset_name": cfg.dataset.name,
        "entity_id":    cfg.dataset.entity_id,
        "threshold":    float(threshold),
        "n_tp_windows": len({r["window_id"] for r in rows}),
        "by_q_variant": by_qv,
        "faithfulness_gap": faith,
    }


# Region-level invalid

def _anomaly_events(y: np.ndarray | None) -> list[tuple[int, int]]:
    """Contiguous [start, stop) runs where the timestamp label is positive."""
    if y is None:
        return []
    pos = np.asarray(y) > 0
    events: list[tuple[int, int]] = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i]:
            j = i
            while j < n and pos[j]:
                j += 1
            events.append((i, j))
            i = j
        else:
            i += 1
    return events


def _inject_region_validity(
    report: dict[str, Any],
    deltas: dict[tuple[int, float, str], np.ndarray],
    tps: list[dict[str, Any]],
    test_records: list[TimeSeriesRecord],
    test_entities: list[dict[str, Any]],
    cfg: Config, threshold: float,
    quantiles: tuple[float, ...], variants: tuple[str, ...],
) -> None:
    """Region-level validity: repair EVERY flagged (TP) window of an anomaly
    event at once, re-score on detect's assembled scale, and check whether the
    event's peak drops below threshold. Averaged over anomaly events, one value
    per (q, variant); written into
    report["by_q_variant"][key]["validity_region_{mean,std,n_events}"].

    Why this and not the single-window `validity_window`: detect assembles
    overlapping windows by SUM, so each anomalous timestep's score is the sum of
    K ≈ window_length/eval_stride windows. Repairing ONE window removes only ~1/K
    of the summed peak (→ validity_window ≈ 0 for strong anomalies). The per-channel
    CF deltas add into the accumulator exactly as detect summed the originals, so
    splicing ALL of an event's window deltas removes the full summed contribution
    and the event drops to its normal per-channel baseline — below threshold.
    """
    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for wi, tp in enumerate(tps):
        by_record[tp["record_idx"]].append((wi, tp["start"], tp["stop"]))

    for q in quantiles:
        for variant in variants:
            flags: list[float] = []
            for ri, wins in by_record.items():
                ent = test_entities[ri]
                raw = np.asarray(
                    ent.get("channel_scores_raw", ent["channel_scores"]), dtype=float
                ).copy()
                applied = False
                for (wi, s, e) in wins:
                    d = deltas.get((wi, q, variant))
                    if d is None:                       # q skipped (n_masked == 0)
                        continue
                    raw[s:e] += d
                    applied = True
                if not applied:
                    continue
                tmp: dict[str, Any] = {"channel_scores": raw, "labels": None}
                _finalize_entities([tmp], cfg)          # impulse + channel max
                overall_cf = np.asarray(tmp["overall_scores"])
                for (es, ee) in _anomaly_events(test_records[ri].y):
                    if not any(s < ee and e > es for (_, s, e) in wins):
                        continue                        # event no TP window covered
                    flags.append(1.0 if float(overall_cf[es:ee].max()) < threshold else 0.0)
            key = f"{variant}|q={q}"
            if flags and key in report.get("by_q_variant", {}):
                agg = report["by_q_variant"][key]
                agg["validity_region_mean"] = float(np.mean(flags))
                agg["validity_region_std"] = float(np.std(flags))
                agg["validity_region_n_events"] = float(len(flags))


# ─── CLI entrypoint ──────────────────────────────────────────────────────────

def _parse_quantiles(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual quality evaluation (cf_eval.py)."
    )
    parser.add_argument(
        "--quantiles", type=_parse_quantiles,
        default=(0.99, 0.95, 0.90, 0.80, 0.50, 0.0),
        help="Comma-separated quantiles for the surprise-mask sweep.",
    )
    parser.add_argument(
        "--granularity", choices=("column", "token"), default="column",
    )
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument(
        "--stride_subsample", type=int, default=1,
        help="Multiplies eval_stride when collecting TPs (1 = paper stride).",
    )
    parser.add_argument(
        "--with_dtw", action="store_true",
        help="Compute DTW for q=0.0 (requires `dtaidistance`).",
    )
    parser.add_argument(
        "--cf_mode", choices=("iterative", "oneshot"), default="iterative",
        help="Counterfactual decode: iterative (paper-faithful) or oneshot argmax.",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    evaluate(
        quantiles=tuple(args.quantiles),
        granularity=args.granularity,
        max_windows=args.max_windows,
        stride_subsample=args.stride_subsample,
        with_dtw=args.with_dtw,
        cf_mode=args.cf_mode,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    set_process_title()
    from lib.profiling import profile_run        # opt-in (TVQ_PROFILE=1); no-op when off
    with profile_run("cf_eval"):
        main()
