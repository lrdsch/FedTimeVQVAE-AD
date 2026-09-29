"""
=============================================================================
  Detection — the end-to-end anomaly detection pipeline.
=============================================================================

This is what actually produces the numbers you put in a paper:
  * load stage 1 + stage 2 checkpoints,
  * compute per-window per-channel scores on train & test splits:
      - s_local:       (x - reconstruction)^2             → reconstruction energy
      - s_prior:       column-wise masked-prediction NLL  → prior surprise
  * per-channel score = weighted sum of normalized scorers,
  * overall score = aggregate across channels (max / mean / topk / …),
  * reassemble overlapping windows into per-time-step scores (mean coverage),
  * fit a threshold on the training distribution (e.g. 99th quantile),
  * evaluate test predictions → F1, event-F1, AUROC, AUPRC, paper top-K.

Outputs (under `artifacts/reports/<run_name>/<aggregation>/`):
  * `report.json`       — all metrics + threshold.
  * `scores.npz`        — raw train/test scores (optional).
  * `plots/*.png`       — per-entity series + anomaly score + predictions.

One function does everything: `detect(cfg, stage1_ckpt=None, stage2_ckpt=None)`.
Run directly as `python pipeline/detect.py` to use the default config.

Debug tips:
  * Set `cfg.scoring.aggregation = "mean"` (fast). Once it works, try "max".
  * Put a breakpoint in `_score_windows()` to inspect scorer outputs batch-by-batch.
  * Set `cfg.evaluation.save_plots = False` for faster iteration.
"""
from __future__ import annotations

# repo root on sys.path so this pipeline/ script can import the shared
# libs (config / data / utils / metrics_core) that live at the project root.
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from lib.proctitle import set_process_title  # noqa: E402

import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader

from config import Config, format_config, load_config
from data import SlidingWindowDataset, TimeSeriesRecord, make_dataloaders
from stage1 import Stage1VQVAE, load_stage1
from stage2 import Stage2System, load_stage2
from utils import (
    best_checkpoint, configure_logging, log_gpu_peak, resolve_path, run_dir_for,
    seed_everything, resolve_device,
)


# ═════════════════════════════════════════════════════════════════════════════
#   Scoring: s_local + s_prior + combine + aggregate across channels
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_eval_stride(cfg: Config) -> int:
    """Detection rolling stride. Absolute (`eval_stride`) wins if set; otherwise
    derive from rate as `max(1, round(eval_stride_rate * window_length))`."""
    if cfg.dataset.eval_stride is not None:
        return max(1, int(cfg.dataset.eval_stride))
    return max(1, round(cfg.dataset.eval_stride_rate * cfg.dataset.window_length))


# Detect-time mixed precision (opt-in, default OFF)

def _detect_amp_dtype(cfg: Config):
    """Autocast dtype for the detect FORWARD pass, or None for fp32 (the default).

    Driven by `cfg.evaluation.detect_amp` ("off" | "auto" | "fp16" | "bf16").
    "auto" picks fp16 below Ampere and bf16 from Ampere on: Turing (sm_75, the
    Quadro RTX 8000 here) has fp16 tensor cores but NO bf16 ones, and bf16 there is
    emulated and SLOWER than fp32. Never gate this on `torch.cuda.is_bf16_supported()`
    — it returns True on Turing.
    """
    if not torch.cuda.is_available():
        return None
    mode = str(getattr(cfg.evaluation, "detect_amp", "off")).lower()
    if mode in {"off", "0", "false", "no", "fp32"}:
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode != "auto":
        raise ValueError(f"detect_amp must be off|auto|fp16|bf16, got {mode!r}")
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _detect_amp(cfg: Config):
    dtype = _detect_amp_dtype(cfg)
    return nullcontext() if dtype is None else torch.autocast(device_type="cuda", dtype=dtype)


def _detect_amp_tag(cfg: Config) -> str:
    """Resolved dtype name, for the score-cache fingerprint. Records the RESOLVED
    value so "auto" on Turing (fp16) never collides with "auto" on Ampere (bf16)."""
    dt = _detect_amp_dtype(cfg)
    return "fp32" if dt is None else str(dt).replace("torch.", "")


def _normalize(x: torch.Tensor, method: str) -> torch.Tensor:
    """Per-window, last-axis normalisation (windows don't influence each other)."""
    if method == "none":
        return x
    if method == "zscore":
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd
    if method == "minmax":
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        return (x - mn) / (mx - mn).clamp_min(1e-6)
    raise ValueError(f"Unknown normalization: {method}")


def _aggregate(scores: torch.Tensor, cfg: Config) -> torch.Tensor:
    """(B, C, T) → (B, T). Aggregates across channels."""
    s = cfg.scoring
    m = s.aggregation
    if m == "mean":       return scores.mean(dim=1)
    if m == "max":        return scores.max(dim=1).values
    if m == "sum":        return scores.sum(dim=1)
    if m == "median":     return torch.quantile(scores, q=0.5, dim=1)
    if m == "rms":        return scores.square().mean(dim=1).sqrt()
    if m == "l2":         return torch.linalg.vector_norm(scores, ord=2, dim=1)
    if m == "lp_norm":    return torch.linalg.vector_norm(scores, ord=s.lp_p, dim=1)
    if m == "logsumexp":  return torch.logsumexp(scores, dim=1)
    if m == "topk":
        k = min(max(1, s.top_k), scores.shape[1])
        return torch.topk(scores, k=k, dim=1).values.mean(dim=1)
    if m == "quantile":
        return torch.quantile(scores, q=s.quantile_q, dim=1)
    if m == "trimmed_mean":
        trim = int(scores.shape[1] * s.trim_ratio)
        if trim == 0: return scores.mean(dim=1)
        return scores.sort(dim=1).values[:, trim: scores.shape[1] - trim].mean(dim=1)
    if m == "winsorized_mean":
        trim = int(scores.shape[1] * s.winsorize_ratio)
        if trim == 0: return scores.mean(dim=1)
        sorted_s = scores.sort(dim=1).values
        lo = sorted_s[:, trim: trim + 1]
        hi = sorted_s[:, scores.shape[1] - trim - 1: scores.shape[1] - trim]
        out = sorted_s.clone()
        out[:, :trim] = lo
        out[:, scores.shape[1] - trim:] = hi
        return out.mean(dim=1)
    if m == "softmax":
        w = torch.softmax(scores / s.softmax_temperature, dim=1)
        return (w * scores).sum(dim=1)
    if m == "mean_max_mix":
        a = s.mix_alpha
        return (1 - a) * scores.mean(dim=1) + a * scores.max(dim=1).values
    if m == "fraction_above_threshold":
        return (scores > s.channel_threshold).float().mean(dim=1)
    raise ValueError(f"Unknown aggregation: {m}")


def _combine_scores(
    s_local: torch.Tensor, s_prior: torch.Tensor, cfg: Config,
) -> torch.Tensor:
    """Per-channel combined score (B, C, T) — normalised per the config.

    Channel aggregation is deferred to AFTER rolling-window assembly and the
    impulse term, mirroring how the paper applies mean_F (its analogue of
    channel collapse) at the very end of the pipeline.
    """
    c = _normalize(s_local, cfg.scoring.normalization) * cfg.scoring.weight_s_local
    c = c + _normalize(s_prior, cfg.scoring.normalization) * cfg.scoring.weight_s_prior
    return c


# ═════════════════════════════════════════════════════════════════════════════
#   Per-entity reassembly with overlap handling
# ═════════════════════════════════════════════════════════════════════════════

def _init_entity(records: list[TimeSeriesRecord]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        feature_names = [] if record.metadata is None else list(record.metadata.feature_names)
        entity_id = f"series_{idx}" if record.metadata is None else record.metadata.entity_id
        entities.append({
            "index": idx,
            "entity_id": entity_id,
            "dataset_name": "unknown" if record.metadata is None else record.metadata.dataset,
            "feature_names": feature_names,
            "series": record.X,                            # (T, C)
            "labels": record.y,                            # (T,) or None
            # y_channel (per-channel GT, T×C int) is kept on the entity so the
            # score cache is self-contained: `_channel_precision_recall_at_k`
            # reads it from here and the cache-hit path doesn't need TimeSeriesRecord.
            "y_channel": record.y_channel,
            "channel_sum": np.zeros_like(record.X, dtype=np.float64),  # (T, C)
            "coverage":    np.zeros(record.X.shape[0], dtype=np.float64),
        })
    return entities


def _fill_uncovered(values: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    valid = coverage > 0
    if valid.all() or not valid.any(): return values
    filled = values.copy()
    first = int(np.argmax(valid))
    filled[:first] = filled[first]
    last = first
    for i in range(first + 1, len(valid)):
        if valid[i]:
            if i - last > 1:
                filled[last + 1: i] = filled[last]
            last = i
    if last < len(valid) - 1:
        filled[last + 1:] = filled[last]
    return filled


def _moving_average_paper(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge-clipped kernel — paper Algorithm 1.

    For each j: out[j] = mean(x[max(0, j-window/2) : j + window/2]).
    Used to compute the "impulse" term ã_s such that
        a_final = (ā_s + ã_s) / 2
    This recovers extended low-amplitude anomalies that get out-ranked by
    short-duration spikes when scoring with raw NLL only.
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


def _assemble_rolling(
    values: np.ndarray, coverage: np.ndarray, mode: str,
) -> np.ndarray:
    """Reduce per-timestep accumulator into the final per-timestep score.

    `mode="sum"`           — paper Algorithm 1: just fill uncovered tail/head
                             from the nearest covered region. No division.
    `mode="mean_coverage"` — divide by per-timestep coverage so each point is
                             the average of the per-window scores that touched
                             it (legacy behaviour).
    """
    if mode == "sum":
        return _fill_uncovered(values, coverage)
    if mode == "mean_coverage":
        denom = np.maximum(coverage, 1.0)
        averaged = values / (denom[:, None] if values.ndim > 1 else denom)
        return _fill_uncovered(averaged, coverage)
    raise ValueError(f"Unknown rolling_aggregation mode: {mode!r}")


# ═════════════════════════════════════════════════════════════════════════════
#   Score a split (train or test)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_entities_raw(
    stage1: Stage1VQVAE,
    stage2: Stage2System | None,
    cfg: Config,
    records: list[TimeSeriesRecord],
    *,
    record_per_rate: bool = False,
) -> list[dict[str, Any]]:
    """Forward pass + rolling-window assembly. Returns the entities list with
    `channel_scores` (T, C) RAW — i.e. pre-impulse, pre-channel-aggregation.

    This is the heavy step (stage1 + stage2 inference on GPU). Output is the
    snapshot saved/loaded by the detection score cache; the cheap impulse +
    aggregation pass runs separately in `_finalize_entities`.

    When `record_per_rate=True` (used on TRAIN), each entity gets an extra
    `per_rate_CF_sum` array of shape (n_τ, T_full, C, F) — paper-style
    per-τ per-(C, F) accumulator used to compute the threshold.
    """
    import os
    assert not stage1.training, "detect requires stage1 in eval mode"
    if stage2 is not None:
        assert not stage2.stage1.training and not stage2.prior.training, (
            "detect requires stage2.stage1 and stage2.prior in eval mode"
        )
    device = next(stage1.parameters()).device
    num_workers = int(os.environ.get("DEBUG_NUM_WORKERS", cfg.dataset.num_workers))

    # Detection rolling stride is decoupled from training stride. See
    # `_resolve_eval_stride`: absolute `eval_stride` wins; else `eval_stride_rate`·T.
    eval_stride = _resolve_eval_stride(cfg)
    dataset = SlidingWindowDataset(
        records, cfg.dataset.window_length, eval_stride,
        window_normalization=cfg.dataset.window_normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size_eval,
        shuffle=False, num_workers=num_workers,
        # Pinned host memory is what lets the `non_blocking=True` copies below actually
        # overlap the H2D transfer with the GPU forward.
        pin_memory=torch.cuda.is_available(),
        **({"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}),
    )
    entities = _init_entity(records)

    # Lazy init of per-rate per-(C, F) accumulator: we don't know n_τ / F until
    # the first batch comes back from the prior.
    per_rate_inited = False

    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        batch["inputs"] = inputs
        batch["labels"] = labels

        with torch.no_grad(), _detect_amp(cfg):
            s1_out = stage1(inputs)
            reconstructed = s1_out["reconstructed"]
            s_local = (inputs - reconstructed).pow(2)     # (B, C, T)

            ts_per_rate_T = None  # (n_τ, B, C, F, T) when record_per_rate else None

            if stage2 is not None:
                prior_out = stage2.score_batch(batch, per_rate=record_per_rate)
                token_scores = prior_out.token_scores
                if token_scores.ndim == 2:
                    # Legacy: (B, W) broadcast across channels (no F).
                    up = torch.nn.functional.interpolate(
                        token_scores.unsqueeze(1), size=inputs.shape[-1],
                        mode="nearest",
                    )                                        # (B, 1, T)
                    s_prior = up.expand(-1, inputs.shape[1], -1)  # (B, C, T)
                else:
                    # Target: (B, C, F, W) summed, or (n_τ, B, C, F, W) per-rate.
                    if token_scores.ndim == 5:
                        ts_summed = token_scores.sum(dim=0)            # (B, C, F, W)
                        # Interpolate the per-rate stack along time too: nearest
                        # mode along the last axis of (n_τ, B, C, F, W).
                        n_tau, B_, C_, F_, W_ = token_scores.shape
                        flat_pr = token_scores.reshape(n_tau * B_ * C_, F_, W_)
                        flat_pr_T = torch.nn.functional.interpolate(
                            flat_pr, size=inputs.shape[-1], mode="nearest",
                        )
                        ts_per_rate_T = flat_pr_T.reshape(
                            n_tau, B_, C_, F_, inputs.shape[-1],
                        )                                              # (n_τ, B, C, F, T)
                    else:
                        assert token_scores.ndim == 4, (
                            f"expected (B,C,F,W) or (n_τ,B,C,F,W) from prior, got {token_scores.shape}"
                        )
                        ts_summed = token_scores
                    assert ts_summed.shape[1] == inputs.shape[1], (
                        f"prior channels {ts_summed.shape[1]} != inputs {inputs.shape[1]}"
                    )
                    B_, C_, F_, W_ = ts_summed.shape
                    flat = ts_summed.reshape(B_ * C_, F_, W_)
                    flat_T = torch.nn.functional.interpolate(
                        flat, size=inputs.shape[-1], mode="nearest",
                    )                                        # (B*C, F, T)
                    # Paper Algorithm 1: mean over frequency (E_h[a_s]).
                    s_prior = flat_T.mean(dim=1).reshape(B_, C_, inputs.shape[-1])
            else:
                s_prior = torch.zeros_like(s_local)

        # Per-channel combined score on the GPU; channel aggregation is
        # deferred to the very end (paper-equivalent for non-linear modes).
        # .float() upcasts before all score math: a no-op when detect_amp=off.
        per_channel = _combine_scores(s_local.float(), s_prior.float(), cfg)
        per_channel_np = per_channel.detach().cpu().numpy()

        md = batch["metadata"]
        record_indices = md["record_index"].tolist() if torch.is_tensor(md["record_index"]) else md["record_index"]
        starts = md["window_start"].tolist() if torch.is_tensor(md["window_start"]) else md["window_start"]
        stops  = md["window_stop"].tolist()  if torch.is_tensor(md["window_stop"])  else md["window_stop"]

        # Per-rate accumulator init (after first batch — we now know n_τ and F).
        if record_per_rate and ts_per_rate_T is not None and not per_rate_inited:
            n_tau, _, _, F_, _ = ts_per_rate_T.shape
            for e in entities:
                T_full, C_full = e["series"].shape[0], e["series"].shape[1]
                e["per_rate_CF_sum"] = np.zeros(
                    (n_tau, T_full, C_full, F_), dtype=np.float64,
                )
            per_rate_inited = True

        ts_per_rate_np = (
            ts_per_rate_T.float().detach().cpu().numpy() if ts_per_rate_T is not None else None
        )

        for i, ri in enumerate(record_indices):
            entity = entities[int(ri)]
            start, stop = int(starts[i]), int(stops[i])
            entity["channel_sum"][start: stop] += per_channel_np[i].T   # (T, C)
            entity["coverage"][start: stop]    += 1.0
            if ts_per_rate_np is not None:
                # ts_per_rate_np[:, i] : (n_τ, C, F, T_window) → (n_τ, T_window, C, F)
                entity["per_rate_CF_sum"][:, start: stop] += (
                    ts_per_rate_np[:, i].transpose(0, 3, 1, 2)
                )

    # Assemble rolling-window accumulator into the per-channel time series.
    # `channel_scores` is RAW here (pre-impulse, pre-aggregation) — this is the
    # snapshot that `_save_score_cache` persists. The impulse+aggregation step
    # lives in `_finalize_entities` so it can re-run from the cache without GPU.
    mode = cfg.scoring.rolling_aggregation
    for e in entities:
        coverage = e.pop("coverage")
        e["channel_scores"] = _assemble_rolling(e.pop("channel_sum"), coverage, mode)

    return entities


def _finalize_entities(
    entities: list[dict[str, Any]], cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply impulse term (per cfg.scoring.use_impulse_term) and channel
    aggregation. Modifies `entities` in-place (writes `overall_scores`, and
    when impulse is enabled also `channel_scores_raw` / `channel_scores_impulse`).
    Returns flat (scores, labels) for corpus-level metrics.

    CPU-only and cheap — this is what re-runs on a cache hit instead of
    invoking stage1/stage2 again.
    """
    # Paper Algorithm 1: a_final = (ā_s + ã_s) / 2 where ã_s is the
    # centered moving-average with window=T. Applied PER CHANNEL — channel
    # aggregation comes after, so non-linear modes (max/topk/...) see the
    # impulse-corrected per-channel series instead of the per-window aggregate.
    if cfg.scoring.use_impulse_term:
        T_window = cfg.dataset.window_length
        for e in entities:
            raw = np.asarray(e["channel_scores"])               # (T, C) RAW
            ma = np.empty_like(raw)
            for c in range(raw.shape[1]):
                ma[:, c] = _moving_average_paper(raw[:, c], T_window)
            e["channel_scores_raw"] = raw                        # kept for plots/debug
            e["channel_scores_impulse"] = ma
            e["channel_scores"] = 0.5 * (raw + ma)

    # Channel aggregation at the very end — paper-equivalent for non-linear
    # modes (max, topk, median, rms, …) under the multivariate generalisation.
    for e in entities:
        e["overall_scores"] = _aggregate_channels_np(e["channel_scores"], cfg)

    # Flatten for corpus-level metrics (concatenate across entities).
    flat_scores = np.concatenate([e["overall_scores"] for e in entities])
    flat_labels = np.concatenate([
        np.zeros_like(e["overall_scores"], dtype=np.int64) if e["labels"] is None
        else np.asarray(e["labels"], dtype=np.int64)
        for e in entities
    ])
    return flat_scores.astype(np.float64), flat_labels.astype(np.int64)


def _score_split(
    stage1: Stage1VQVAE,
    stage2: Stage2System | None,
    cfg: Config,
    records: list[TimeSeriesRecord],
    *,
    record_per_rate: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Returns (flat_scores, flat_labels, per-entity dicts). Backward-compatible
    wrapper around `_compute_entities_raw` + `_finalize_entities`. The split
    exists so the heavy GPU pass (raw) can be cached independently of the cheap
    cfg-sensitive finalize step (impulse + aggregation)."""
    entities = _compute_entities_raw(
        stage1, stage2, cfg, records, record_per_rate=record_per_rate,
    )
    flat_scores, flat_labels = _finalize_entities(entities, cfg)
    return flat_scores, flat_labels, entities


# ═════════════════════════════════════════════════════════════════════════════
#   Thresholding + metrics
# ═════════════════════════════════════════════════════════════════════════════

def _aggregate_channels_np(arr: np.ndarray, cfg: Config) -> np.ndarray:
    """Numpy wrapper around `_aggregate`. Accepts:

      * `(C,)`  → returns scalar (length-1 array)
      * `(T, C)` → returns `(T,)` per-time channel-aggregated series

    Reuses the GPU implementation by going through torch.
    """
    a = np.asarray(arr, dtype=np.float64)
    # NOTE: -M-Real short-circuits median/quantile to np.quantile here, citing a
    # `torch.quantile` limit of 2**24 elements. That limit applies to the size of the
    # REDUCED dimension, not to the total numel — and `_aggregate` reduces over the
    # channel axis (C ≲ 55), so it never trips. Measured on torch 2.11 with a pooled
    # (700k, 38) input: torch 446 ms vs numpy 1078 ms. Deliberately NOT ported.
    if a.ndim == 1:                                  # (C,) — threshold vector
        t = torch.from_numpy(a).reshape(1, -1, 1)    # (1, C, 1)
        return _aggregate(t, cfg).reshape(-1).numpy()
    if a.ndim == 2:                                  # (T, C) — per-time per-channel
        t = torch.from_numpy(a.T).unsqueeze(0)       # (1, C, T)
        return _aggregate(t, cfg).reshape(-1).numpy()
    raise ValueError(f"_aggregate_channels_np: unsupported shape {a.shape}")


def _fit_threshold_paper(
    train_entities: list[dict[str, Any]], cfg: Config,
) -> float:
    """Paper Algorithm 1 + evaluate.py threshold flow:

        thr_τ        = quantile(a_star_τ_train, q, axis=time)   shape (C, F)
        joint_thr    = Σ_τ thr_τ                                shape (C, F)
        per_ch_thr   = mean_F(joint_thr)                        shape (C,)
        final_thr    = aggregate_channels(per_ch_thr)           scalar

    `cfg.threshold.q` is the quantile applied per-τ on TRAIN. `cfg.scoring.aggregation`
    is reused to collapse channels — same operation we apply to the score —
    so threshold and overall_score share the same channel-reduction.

    `cfg.threshold.name == "fixed"` short-circuits to `cfg.threshold.value`.
    """
    if cfg.threshold.name == "fixed":
        return float(cfg.threshold.value)
    if cfg.threshold.name != "quantile":
        raise ValueError(f"Unknown threshold: {cfg.threshold.name}")

    # Concatenate per-τ accumulators across entities along the time axis.
    has_per_rate = all("per_rate_CF_sum" in e for e in train_entities)
    if not has_per_rate:
        # Fallback: simple quantile of the 1D summed train score.
        flat = np.concatenate([e["overall_scores"] for e in train_entities])
        return float(np.quantile(flat, cfg.threshold.q))

    # Stack: (n_τ, T_total, C, F)
    big = np.concatenate(
        [e["per_rate_CF_sum"] for e in train_entities], axis=1,
    )
    n_tau = big.shape[0]
    # per-τ quantile along time → (n_τ, C, F)
    thr_per_rate = np.quantile(big, cfg.threshold.q, axis=1)
    joint = thr_per_rate.sum(axis=0)                    # (C, F)
    per_channel_thr = joint.mean(axis=-1)               # (C,)
    scalar_thr = _aggregate_channels_np(per_channel_thr, cfg)
    return float(scalar_thr.item())


# Phase C de-mirror: the low-level detection-metric primitives live once in
# metrics_core.py (repo root). Imported here under their historical names so
# every call site below (incl. the plotting helpers) is unchanged and detect.py's
# report.json is identical — only the definition site moved.
from metrics_core import (
    segments as _segments,
    event_metrics as _event_metrics,
    detection_delay_metrics as _detection_delay_metrics,
    affiliation_metrics as _affiliation_metrics,
    vus_metrics as _vus_metrics,
    pate_metrics as _pate_metrics,
    threshold_free_metrics as _threshold_free_metrics,
    channel_localization_at_k as _channel_localization_at_k,
    per_channel_metrics as _per_channel_metrics,
    aggregate_macro_weighted as _aggregate_macro_weighted,
    joint_micro_metrics as _joint_metrics,
    # The unified suite — the SAME function comparisons/evaluate.py runs on
    # every baseline's scores.npz. Calling it here makes TimeVQVAE's
    # threshold_free / by_threshold numbers apples-to-apples by construction.
    evaluate_scores as _evaluate_scores,
    POT_LEVEL_BY_DATASET as _POT_LEVEL_BY_DATASET,
)


# The tolerance the UCR-TSA protocol actually prescribes. TimeVQVAE-AD §5.1: "a single
# predicted anomaly location ... within a range of ±100 data points from the true location";
# "true location" means the SEGMENT, not a point -- Baldan et al. 2025 state it unambiguously
# ("distance <= 100 points from any point on the anomaly"), which is exactly the rule below.
#
# We PIN `paper_metrics_tolerance` to 64 instead, so the window ablation compares W=128 and
# W=2P with one fixed ruler (the default would be window_length//2, i.e. a ruler that moves
# with the axis under test). Changing the pinned value is NOT an option: `metrics_tolerance`
# is inside the cohort fingerprint (scripts/cohort.py:168), so it would orphan every cell
# already on disk. Hence: report at the pinned tolerance AND, always, at the protocol one.
#
# Measured 2026-08-01 on 240 (arm, client) pairs: switching 64 -> 100 moves top-1 by exactly
# 0.000, because the argmax distribution is bimodal -- 49.2% land dead on the segment, 0.0%
# land anywhere in 1..100, 41.7% miss by >1000. The column exists for protocol fidelity, and
# to be able to SHOW that the ruler does not drive the result.
PROTOCOL_TOLERANCE = 100


def _topk_at(pos: np.ndarray, scores: np.ndarray, tol: int,
             ks: "list[int]") -> dict[str, float]:
    """Top-K hit/miss at one tolerance. `tol` sets BOTH the minimum peak spacing and the
    hit radius, so it must be applied to `find_peaks` too -- not just to the comparison."""
    peak_idx, _ = find_peaks(scores, distance=tol)
    if peak_idx.size > 0:
        peaks_sorted = peak_idx[np.argsort(-scores[peak_idx], kind="stable")]
    else:
        peaks_sorted = np.empty(0, dtype=np.int64)

    argmax_idx = int(np.argmax(scores))
    out: dict[str, float] = {}
    for k in ks:
        if k == 1:
            preds = np.array([argmax_idx], dtype=np.int64)
        else:
            preds = peaks_sorted[: min(k, peaks_sorted.size)]
            # Edge case — fewer peaks than k: fall back to argmax so the
            # top-1 hit still counts (prevents NaN when scores are flat).
            if preds.size == 0:
                preds = np.array([argmax_idx], dtype=np.int64)
        hit = any(int(np.min(np.abs(pos - p))) <= tol for p in preds)
        out[f"paper_top{k}_acc_at_{tol}"] = float(hit)
    return out


def _paper_metrics(labels: np.ndarray, scores: np.ndarray, cfg: Config) -> dict[str, float]:
    """Top-K detection accuracy via local-maxima ranking (paper evaluate.py).

    For each k in `paper_metrics_top_k`:
        * top-1: predictions = [argmax(scores)]
        * top-K (K > 1): predictions = K highest local maxima of `scores`,
          where peaks must be at least `paper_metrics_tolerance` apart
          (`find_peaks(scores, distance=tolerance)`). This forces predictions
          to be DISTINCT candidates instead of clustering inside one anomaly.
        * Hit if any prediction is within `tolerance` of any ground-truth
          positive timestep (= within the GT segment expanded by ±tolerance).

    Emitted at the pinned `paper_metrics_tolerance` AND at `PROTOCOL_TOLERANCE` (see the
    note above the constant). Keys are suffixed `_at_<tol>`, so the two never collide and
    nothing that reads the old key changes behaviour.
    """
    if not cfg.evaluation.paper_metrics_enabled:
        return {}
    pos = np.flatnonzero(labels > 0)
    tol = cfg.evaluation.paper_metrics_tolerance
    ks = list(cfg.evaluation.paper_metrics_top_k)
    tols = [tol] if tol == PROTOCOL_TOLERANCE else [tol, PROTOCOL_TOLERANCE]
    if scores.size == 0 or pos.size == 0:
        return {f"paper_top{k}_acc_at_{t}": float("nan") for t in tols for k in ks}

    out: dict[str, float] = {}
    for t in tols:
        out.update(_topk_at(pos, scores, t, ks))
    return out


# _affiliation_metrics / _vus_metrics / _detection_delay_metrics / _pate_metrics
# now live once in metrics_core.py (imported under these names at the top of
# this metrics section — Phase C de-mirror).


def _channel_precision_recall_at_k(
    test_entities: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Timestamp-level channel localisation (additive block in the main report,
    gated by y_channel): pool every entity's (channel_scores, y_channel) and rank
    channels per anomalous timestep. Delegates to metrics_core. Returns `{}` if no
    entity has y_channel. Same key set as before (drops n_anomalous_timesteps)."""
    ents = [
        e for e in test_entities
        if e.get("y_channel") is not None
        and np.asarray(e["y_channel"]).shape == np.asarray(e["channel_scores"]).shape
    ]
    if not ents:
        return {}
    # Univariate: ranking one channel always puts the right one first (P@1 = 1.0
    # for any model). metrics_core refuses to score it; don't ask.
    if np.asarray(ents[0]["y_channel"]).shape[1] < 2:
        return {}
    cs = np.concatenate([np.asarray(e["channel_scores"]) for e in ents], axis=0)
    yc = np.concatenate([np.asarray(e["y_channel"]) for e in ents], axis=0)
    out = _channel_localization_at_k(cs, yc, ks=ks)
    out.pop("n_anomalous_timesteps", None)
    out.pop("note", None)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#   Per-channel detection (merged from the former detect_per_channel.py)
# ═════════════════════════════════════════════════════════════════════════════
# Channel-native evaluation that reuses the SAME scored entities as the timestamp
# path — one scoring pass produces BOTH the timestamp report and this per-channel
# report. Gated by y_channel; a no-op for legacy datasets. The metric blocks
# (per_channel / macro+weighted / joint) live in metrics_core.

def build_window_token_mask(
    channel_scores_window: np.ndarray,   # (T_window, C)
    thresholds: np.ndarray,              # (C,)
    latent_freq: int,                    # F
    latent_width: int,                   # W
    max_masking_rate: float = 0.9,
) -> "torch.Tensor":
    """Turn the per-channel detection decision on ONE window into a latent token
    mask `(1, C, F, W)` bool for `stage2.counterfactual(token_mask=...)`. Faithful
    multivariate port of the original TimeVQVAE-AD explainable_sampling selection:
    per channel a latent column is flagged iff its (nearest-interp) score exceeds
    that channel's threshold; capped so ≥(1-max_masking_rate) of each channel
    stays as context; broadcast over F → whole-column masking."""
    cs = torch.as_tensor(np.asarray(channel_scores_window), dtype=torch.float32)
    thr = torch.as_tensor(np.asarray(thresholds), dtype=torch.float32)
    assert cs.ndim == 2, f"expected (T_window, C), got {tuple(cs.shape)}"
    T_win, C = cs.shape
    assert thr.shape == (C,), f"thresholds {tuple(thr.shape)} != ({C},)"
    col = cs.transpose(0, 1).unsqueeze(0)
    col = torch.nn.functional.interpolate(col, size=int(latent_width), mode="nearest")[0]
    flag = col > thr[:, None]
    cap = int(np.floor(max_masking_rate * int(latent_width)))
    for c in range(C):
        if int(flag[c].sum().item()) > cap:
            capped = torch.zeros_like(flag[c])
            if cap > 0:
                keep_idx = torch.topk(col[c], k=cap, largest=True).indices
                capped[keep_idx] = True
            flag[c] = capped
    token_mask = flag[:, None, :].expand(C, int(latent_freq), int(latent_width))
    return token_mask.unsqueeze(0).contiguous()


def _fit_thresholds_per_channel(train_entities: list[dict[str, Any]], q: float) -> np.ndarray:
    """Per-channel threshold = quantile(train_channel_scores[:, c], q); all train
    entities concatenated along time. Returns (C,)."""
    big = np.concatenate([np.asarray(e["channel_scores"]) for e in train_entities], axis=0)
    return np.quantile(big, q, axis=0).astype(np.float64)


def _plot_entity_per_channel(
    entity: dict[str, Any], y_channel: np.ndarray, preds: np.ndarray,
    thresholds: np.ndarray, output_dir: Path,
) -> None:
    """One figure per entity: channel×time score heatmap (threshold-relative) +
    GT and prediction stripes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ch_scores = np.asarray(entity["channel_scores"]).T
    yc = np.asarray(y_channel).T
    pr = preds.T
    C, T = ch_scores.shape
    fig, axes = plt.subplots(
        3, 1, figsize=(16, max(6, 0.4 * C + 4)),
        sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]},
        constrained_layout=True,
    )
    rel = ch_scores - thresholds[:, None]
    vmax = max(1e-9, float(np.max(np.abs(rel))))
    axes[0].imshow(rel, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_ylabel("Channel")
    axes[0].set_title(f"{entity['entity_id']} — per-channel score (red = above threshold)")
    axes[1].imshow(yc, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    axes[1].set_ylabel("GT")
    axes[2].imshow(pr, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    axes[2].set_ylabel("Pred")
    axes[2].set_xlabel("Time step")
    fig.savefig(output_dir / f"{entity['entity_id']}_per_channel.png", dpi=110)
    plt.close(fig)


def _write_per_channel_table_md(
    path: Path, dataset: str, entity: str, per_channel: list[dict[str, float]],
    aggregates: dict[str, float], joint: dict[str, float], q: float,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Per-channel detection report\n\n")
        fh.write(f"- Dataset: `{dataset}`\n- Entity: `{entity}`\n")
        fh.write(f"- Threshold: per-channel quantum, q = {q}\n\n")
        fh.write("## Joint (anomaly + right channel)\n\n")
        fh.write("| precision | recall | f1 | AUROC | AUPRC | positives / cells |\n")
        fh.write("|---:|---:|---:|---:|---:|---|\n")
        fh.write(
            f"| {joint['joint_precision']:.4f} | {joint['joint_recall']:.4f} | "
            f"{joint['joint_f1']:.4f} | {joint['joint_auroc']:.4f} | "
            f"{joint['joint_auprc']:.4f} | "
            f"{int(joint['joint_n_positives'])} / {int(joint['joint_n_cells'])} |\n\n"
        )
        fh.write("## Aggregates across channels\n\n")
        fh.write("| | precision | recall | f1 | AUROC | AUPRC |\n|---|---:|---:|---:|---:|---:|\n")
        fh.write(
            f"| macro     | {aggregates['macro_precision']:.4f} | "
            f"{aggregates['macro_recall']:.4f} | {aggregates['macro_f1']:.4f} | "
            f"{aggregates['macro_auroc']:.4f} | {aggregates['macro_auprc']:.4f} |\n"
        )
        fh.write(
            f"| weighted  | {aggregates['weighted_precision']:.4f} | "
            f"{aggregates['weighted_recall']:.4f} | {aggregates['weighted_f1']:.4f} | "
            f"{aggregates['weighted_auroc']:.4f} | {aggregates['weighted_auprc']:.4f} |\n\n"
        )
        fh.write(
            f"valid channels (>=1 GT positive): "
            f"{int(aggregates['n_valid_channels'])} / {int(aggregates['n_total_channels'])}\n\n"
        )
        fh.write("## Per-channel breakdown\n\n")
        fh.write("| channel | n_pos | thr | precision | recall | f1 | AUROC | AUPRC |\n")
        fh.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in per_channel:
            fh.write(
                f"| {int(r['channel'])} | {int(r['n_positives'])} | "
                f"{r['threshold']:.4g} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | {r['f1']:.4f} | "
                f"{r['auroc']:.4f} | {r['auprc']:.4f} |\n"
            )


def _run_per_channel_eval(
    train_entities: list[dict[str, Any]], test_entities: list[dict[str, Any]],
    cfg: Config, output_dir: Path,
) -> Optional[dict[str, Any]]:
    """Per-channel detection report (additive; gated by y_channel). Reuses the
    already-scored entities — NO second scoring pass. Writes
    `<output_dir>/per_channel/{report.json, per_channel_table.md, plots/}`."""
    test_with_yc = [e for e in test_entities if e.get("y_channel") is not None]
    if not test_with_yc:
        return None
    n_channels = int(np.asarray(test_with_yc[0]["y_channel"]).shape[1])
    if n_channels < 2:
        # Every block of this report (per-channel / macro+weighted / joint) would
        # restate the timestamp report under a per-channel threshold. Skip it.
        print("[detect] univariate (C=1): per-channel detection report skipped "
              "(channel attribution is undefined).")
        return None
    thresholds = _fit_thresholds_per_channel(train_entities, cfg.threshold.q)
    ch_scores_all = np.concatenate([np.asarray(e["channel_scores"]) for e in test_with_yc], axis=0)
    y_channel_all = np.concatenate(
        [np.asarray(e["y_channel"]) for e in test_with_yc], axis=0).astype(np.int64)
    preds_all = (ch_scores_all > thresholds[None, :]).astype(np.int64)
    per_channel = _per_channel_metrics(y_channel_all, ch_scores_all, preds_all, thresholds)
    aggregates = _aggregate_macro_weighted(per_channel)
    joint = _joint_metrics(y_channel_all, ch_scores_all, preds_all)
    report = {
        "dataset_name": cfg.dataset.name, "entity_id": cfg.dataset.entity_id,
        "seed": cfg.seed, "threshold_q": float(cfg.threshold.q),
        "thresholds": thresholds.tolist(), "per_channel": per_channel,
        **aggregates, **joint,
    }
    pc_dir = Path(output_dir) / "per_channel"
    pc_dir.mkdir(parents=True, exist_ok=True)
    (pc_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_per_channel_table_md(
        pc_dir / "per_channel_table.md", cfg.dataset.name, cfg.dataset.entity_id,
        per_channel, aggregates, joint, cfg.threshold.q,
    )
    plots_dir = pc_dir / "plots"
    cursor = 0
    for e in test_with_yc:
        T = e["channel_scores"].shape[0]
        _plot_entity_per_channel(
            e, y_channel_all[cursor:cursor + T], preds_all[cursor:cursor + T], thresholds, plots_dir)
        cursor += T
    print(f"[detect] per-channel: joint F1={joint['joint_f1']:.4f} | "
          f"macro F1={aggregates['macro_f1']:.4f} | report: {(pc_dir / 'report.json').resolve()}")
    return report


def _detection_metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray, cfg: Config) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f)}
    # FPR completes the operational trio (F1 / Recall / FPR) — on long series
    # high recall is unimpressive if FPR is also high.
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    out["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        # best_f1: oracle-threshold F1 (max over the full PR curve). Decouples
        # model quality from the fixed-quantile threshold rule — pairs with
        # `f1` to expose how much is lost on threshold calibration.
        prec_c, rec_c, _ = precision_recall_curve(labels, scores)
        denom = np.clip(prec_c + rec_c, 1e-12, None)
        out["best_f1"] = float(np.max(2.0 * prec_c * rec_c / denom))
        out.update(_pate_metrics(labels, preds, scores, cfg.evaluation.paper_metrics_tolerance))
        out.update(_affiliation_metrics(labels, preds))
        out.update(_vus_metrics(labels, scores, cfg.evaluation.paper_metrics_tolerance))
        out.update(_detection_delay_metrics(labels, preds))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#   Plots (one per entity)
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# _plot_entity — DISABLED.
#
# Historical origin: inherited from upstream TimeVQVAE-AD (cf.
# tvqvae-orig/evaluation/__init__.py), where the reference benchmarks were
# UCR / SMAP / MSL, often univariate or with a dominant channel-0. In that
# context, plotting only the "first channel" as a visual reference for the GT
# alongside the overall score was reasonable.
#
# Why it was disabled: the repo grew to ~45 datasets, the vast majority of
# which are multivariate with an arbitrary channel 0:
#   * the whole toy_*_channel_anomalies family (C ∈ {6,8,10}) injects anomalies
#     into channels chosen at random by pick_channels(...) in _toy_common.py,
#     so channel 0 is statistically affected only ~1/3 of the time;
#   * industrial/process datasets (CATS, HAI, TEP, SKAB, THREEW, GECCO,
#     BATADAL, BATTLEDIM, DAMADICS, GENESIS, PUMP_SENSOR, OPSSAT_AD),
#     HAR (PAMAP2, MHEALTH, USC_HAD, OPPORTUNITY, DAPHNET, WESAD_v2) and
#     anomaly benchmarks (SMD, PSM, CMAPSS) all have an arbitrary channel 0.
# Result: the top panel showed a "calm" series under a red GT window, because
# the anomaly lived on other channels — misleading.
#
# The correct view for the multivariate case (one row per channel + a heatmap
# of the channel_scores + the overall score) is already produced by
# _plot_entity_train_test_all_features under
# `<output_dir>/all_time_series_train_test_all_features/`, which also covers
# the few univariate / ECG-dominated cases (apnea_ecg, nab, mitbih, ptbxl,
# toy_ecg_synth) without loss of information.
#
# def _plot_entity(entity: dict[str, Any], threshold: float, output_dir: Path) -> None:
#     output_dir.mkdir(parents=True, exist_ok=True)
#     T = entity["series"].shape[0]
#     t = np.arange(T)
#     scores = entity["overall_scores"]
#     labels = entity["labels"]
#     preds = (scores > threshold).astype(np.int64)
#
#     fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True, constrained_layout=True)
#     axes[0].plot(t, entity["series"][:, 0], color="steelblue", linewidth=0.8)
#     if labels is not None:
#         for start, stop in _segments(labels):
#             axes[0].axvspan(start, stop, color="red", alpha=0.2)
#     axes[0].set_ylabel(entity["feature_names"][0] if entity["feature_names"] else "channel 0")
#     axes[0].set_title(f"{entity['entity_id']} — series (first channel) with ground-truth")
#
#     axes[1].plot(t, scores, color="black", linewidth=0.8, label="overall score")
#     axes[1].axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"threshold={threshold:.3g}")
#     for start, stop in _segments(preds):
#         axes[1].axvspan(start, stop, color="orange", alpha=0.2)
#     axes[1].set_xlabel("Time step")
#     axes[1].set_ylabel("Anomaly score")
#     axes[1].legend(loc="upper right")
#
#     fig.savefig(output_dir / f"{entity['entity_id']}.png", dpi=120, bbox_inches="tight")
#     plt.close(fig)
# ─────────────────────────────────────────────────────────────────────────────


def _plot_entity_train_test_all_features(
    train_entity: dict[str, Any] | None,
    test_entity: dict[str, Any],
    threshold: float,
    output_dir: Path,
) -> None:
    """One figure per entity, three sections stacked vertically (no overlap):
        * top    — one row per channel with raw signal (train | test concatenated)
        * middle — heatmap of per-channel anomaly scores over time
        * bottom — overall anomaly score with threshold, GT and prediction spans
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    test_series = test_entity["series"]
    test_scores = test_entity["overall_scores"]
    test_labels = test_entity["labels"]
    test_preds = (test_scores > threshold).astype(np.int64)
    n_ch = test_series.shape[1]
    names = test_entity["feature_names"] or [f"channel_{i}" for i in range(n_ch)]
    test_ch_scores = test_entity["channel_scores"]
    if train_entity is not None:
        train_series = train_entity["series"]
        train_scores = train_entity["overall_scores"]
        train_ch_scores = train_entity["channel_scores"]
    else:
        train_series = np.zeros((0, n_ch), dtype=test_series.dtype)
        train_scores = np.zeros((0,), dtype=test_scores.dtype)
        train_ch_scores = np.zeros((0, n_ch), dtype=test_ch_scores.dtype)

    n_train = train_series.shape[0]
    series = np.concatenate([train_series, test_series], axis=0)
    scores = np.concatenate([train_scores, test_scores], axis=0)
    ch_scores = np.concatenate([train_ch_scores, test_ch_scores], axis=0)
    t_total = series.shape[0]
    boundary = n_train

    test_label_segments = _segments(np.asarray(test_labels)) if test_labels is not None else []
    test_pred_segments  = _segments(test_preds)

    height_ratios = [1.0] * n_ch + [4.0, 3.0]
    fig_height = max(10, 0.55 * n_ch + 7)
    fig, axes = plt.subplots(
        n_ch + 2, 1, figsize=(20, fig_height),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # Channel rows (clean — no overlay score)
    for c in range(n_ch):
        ax = axes[c]
        if n_train > 0:
            ax.plot(np.arange(n_train), series[:n_train, c],
                    color="steelblue", linewidth=0.7)
        ax.plot(np.arange(n_train, t_total), series[n_train:, c],
                color="darkgreen", linewidth=0.7)
        for s, e in test_label_segments:
            ax.axvspan(boundary + s, boundary + e + 1,
                       color="red", alpha=0.35, linewidth=0)
        if n_train > 0:
            ax.axvline(boundary, color="0.4", linestyle=":", linewidth=1.0)
        ax.set_ylabel(names[c], fontsize=8, rotation=0,
                      labelpad=64, va="center", ha="right")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)

    axes[0].legend(
        handles=[
            Line2D([0], [0], color="steelblue", linewidth=1.5, label="train signal"),
            Line2D([0], [0], color="darkgreen", linewidth=1.5, label="test signal"),
            Patch(facecolor="red", alpha=0.35, label="ground-truth anomaly"),
            Line2D([0], [0], color="0.4", linestyle=":", linewidth=1.0,
                   label="train | test boundary"),
        ],
        loc="upper right", fontsize=9, ncol=4, framealpha=0.95,
    )
    axes[0].set_title(
        f"{test_entity['entity_id']} — train | test, per-feature signals + scores",
        fontsize=12, fontweight="bold",
    )

    # Per-channel score heatmap
    ax_hm = axes[n_ch]
    im = ax_hm.imshow(
        ch_scores.T, aspect="auto", origin="lower",
        cmap="magma",
        extent=[0, t_total, -0.5, n_ch - 0.5],
        interpolation="nearest",
    )
    if n_train > 0:
        ax_hm.axvline(boundary, color="white", linestyle=":", linewidth=1.2)
    for s, e in test_label_segments:
        ax_hm.add_patch(Rectangle(
            (boundary + s, -0.5), (e - s + 1), n_ch,
            edgecolor="cyan", facecolor="none", linewidth=1.5,
        ))
    ax_hm.set_ylabel("per-channel\nscore score score score", fontsize=9, rotation=0,
                    labelpad=64, va="center", ha="right")
    ticks = sorted({0, n_ch // 2, n_ch - 1})     # dedup: at C=1 all three are 0
    ax_hm.set_yticks(ticks)
    ax_hm.set_yticklabels([names[i] for i in ticks], fontsize=7)
    fig.colorbar(im, ax=ax_hm, location="right",
                 fraction=0.012, pad=0.005, label="score (higher = more anomalous)")

    # Overall score row
    ax_sc = axes[-1]
    if n_train > 0:
        ax_sc.plot(np.arange(n_train), scores[:n_train],
                   color="steelblue", linewidth=0.9)
    ax_sc.plot(np.arange(n_train, t_total), scores[n_train:],
               color="black", linewidth=0.9)
    ax_sc.axhline(threshold, color="red", linestyle="--", linewidth=2.0)
    for s, e in test_label_segments:
        ax_sc.axvspan(boundary + s, boundary + e + 1,
                      color="red", alpha=0.30, linewidth=0)
    for s, e in test_pred_segments:
        ax_sc.axvspan(boundary + s, boundary + e + 1,
                      color="orange", alpha=0.30, linewidth=0)
    if n_train > 0:
        ax_sc.axvline(boundary, color="0.4", linestyle=":", linewidth=1.0)
    ax_sc.set_ylabel("overall\nscore", fontsize=9, rotation=0,
                    labelpad=64, va="center", ha="right")
    ax_sc.set_xlabel("time step", fontsize=10)
    ax_sc.legend(
        handles=[
            Line2D([0], [0], color="steelblue", linewidth=1.5, label="train score"),
            Line2D([0], [0], color="black", linewidth=1.5, label="test score"),
            Line2D([0], [0], color="red", linestyle="--", linewidth=2.0,
                   label=f"threshold = {threshold:.3g}"),
            Patch(facecolor="red", alpha=0.30, label="GT anomaly"),
            Patch(facecolor="orange", alpha=0.30, label="prediction"),
        ],
        loc="upper right", fontsize=9, ncol=5, framealpha=0.95,
    )
    ax_sc.grid(alpha=0.2, linewidth=0.4)

    fig.savefig(
        output_dir / f"{test_entity['entity_id']}_train_test_all_features_with_scores.png",
        dpi=120, bbox_inches="tight",
    )
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
#   Detection score cache — skip the forward pass on repeat runs
# ═════════════════════════════════════════════════════════════════════════════
#
# What's cached: the output of `_compute_entities_raw` for both train and test
# splits — series, labels, y_channel, channel_scores (T, C) RAW (pre-impulse,
# pre-aggregation), and on train per_rate_CF_sum (n_τ, T, C, F).
#
# What invalidates the cache: any cfg field or checkpoint that changes the
# forward pass result. Captured by `_detect_cache_fingerprint`:
#   * stage1 / stage2 checkpoint path + mtime
#   * dataset.window_length, eval_stride, window_normalization
#   * scoring.weight_s_local, weight_s_prior, normalization, rolling_aggregation
#   * seed
#
# What does NOT invalidate (these are reapplied in `_finalize_entities`, cheap):
#   * scoring.use_impulse_term, aggregation, top_k/quantile_q/... (mode-specific)
#   * threshold.{name, q, value}
#   * evaluation.* (metric tolerance, plotting flags, ...)
#
# File layout: `<runs>/stage2/<run_name>/detect_score_cache.npz` — sits next to
# the stage2 checkpoint that produced it (or stage1 dir when stage2 missing).

def _detect_cache_fingerprint(
    cfg: Config, s1_ckpt: Path, s2_ckpt: Path | None,
) -> dict[str, Any]:
    """Dict of all values that affect `_compute_entities_raw`. JSON-encoded
    on disk; cache is reused only on exact equality."""
    s1_resolved = s1_ckpt.resolve()
    s1_mtime = int(s1_ckpt.stat().st_mtime) if s1_ckpt.exists() else 0
    s2_resolved = str(s2_ckpt.resolve()) if s2_ckpt is not None else None
    s2_mtime = int(s2_ckpt.stat().st_mtime) if (s2_ckpt is not None and s2_ckpt.exists()) else 0
    return {
        "stage1_ckpt": str(s1_resolved),
        "stage1_mtime": s1_mtime,
        "stage2_ckpt": s2_resolved,
        "stage2_mtime": s2_mtime,
        "dataset_name": cfg.dataset.name,
        "entity_id": cfg.dataset.entity_id,
        "window_length": int(cfg.dataset.window_length),
        "eval_stride": int(_resolve_eval_stride(cfg)),
        "window_normalization": cfg.dataset.window_normalization,
        "seed": int(cfg.seed),
        "weight_s_local": float(cfg.scoring.weight_s_local),
        "weight_s_prior": float(cfg.scoring.weight_s_prior),
        "scoring_normalization": cfg.scoring.normalization,
        "rolling_aggregation": cfg.scoring.rolling_aggregation,
        # cfg.prior knobs the SCORING forward reads at detect time. load_stage2 rebuilds the
        # prior from the LIVE cfg (not the ckpt's cfg_dict), so these change the score without
        # moving stage2_mtime -- omit them and a sweep serves stale cached scores (a false null).
        # list() coercion is REQUIRED: the fingerprint round-trips through JSON, where a tuple
        # becomes a list, so a tuple here would compare unequal forever (a permanent cache miss).
        "score_window_size_rates": [float(r) for r in cfg.prior.score_window_size_rates],
        "prior_heads": int(cfg.prior.heads),
        "detect_amp": _detect_amp_tag(cfg),
        # Schema version: bump on any change to the KEY SET above (not just the on-disk layout)
        # so old caches are auto-invalidated instead of silently mis-loaded.
        "schema_version": 3,
    }


def _detect_cache_path(s1_ckpt: Path, s2_ckpt: Path | None) -> Path:
    """`detect_score_cache.npz` sits next to the stage2 ckpt (or stage1 dir
    when stage2 is missing). One cache per (dataset, entity_id, seed, architecture)."""
    anchor = s2_ckpt if s2_ckpt is not None else s1_ckpt
    return anchor.parent.parent / "detect_score_cache.npz"


def _cache_status(cache_path: Path, fingerprint: dict[str, Any]) -> tuple[str, str]:
    """Returns (status, reason). status ∈ {"hit", "miss"}; reason is a short
    explanation suitable for the log line."""
    if not cache_path.exists():
        return ("miss", "no cache file on disk")
    try:
        with np.load(cache_path, allow_pickle=False) as z:
            stored = json.loads(str(z["__fingerprint__"]))
    except Exception as exc:
        return ("miss", f"cache unreadable ({exc})")
    if stored != fingerprint:
        # Surface which keys disagreed — the most useful debugging signal.
        diff = sorted(
            k for k in set(stored) | set(fingerprint)
            if stored.get(k) != fingerprint.get(k)
        )
        return ("miss", f"fingerprint mismatch on {diff}")
    return ("hit", "fingerprint match")


def _save_score_cache(
    cache_path: Path,
    fingerprint: dict[str, Any],
    train_entities: list[dict[str, Any]],
    test_entities: list[dict[str, Any]],
) -> None:
    """Serialise per-entity (series, labels, y_channel, channel_scores RAW,
    optional per_rate_CF_sum) for both splits, alongside the fingerprint, into
    a single `.npz`. Pure arrays + json-encoded strings → `allow_pickle=False`
    on load (safer + portable)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, np.ndarray] = {
        "__fingerprint__": np.asarray(json.dumps(fingerprint, sort_keys=True)),
        "__n_train__": np.asarray(len(train_entities), dtype=np.int64),
        "__n_test__": np.asarray(len(test_entities), dtype=np.int64),
    }
    for split, entities in (("train", train_entities), ("test", test_entities)):
        for i, e in enumerate(entities):
            p = f"{split}_e{i}_"
            blob[p + "entity_id"] = np.asarray(str(e["entity_id"]))
            blob[p + "dataset_name"] = np.asarray(str(e["dataset_name"]))
            # feature_names is a list[str] → JSON-encode it as one string array.
            blob[p + "feature_names"] = np.asarray(json.dumps(list(e["feature_names"])))
            blob[p + "series"] = np.asarray(e["series"], dtype=np.float64)
            if e.get("labels") is not None:
                blob[p + "labels"] = np.asarray(e["labels"], dtype=np.int64)
                blob[p + "has_labels"] = np.asarray(True)
            else:
                blob[p + "has_labels"] = np.asarray(False)
            blob[p + "channel_scores"] = np.asarray(e["channel_scores"], dtype=np.float64)
            if e.get("y_channel") is not None:
                blob[p + "y_channel"] = np.asarray(e["y_channel"], dtype=np.int8)
                blob[p + "has_y_channel"] = np.asarray(True)
            else:
                blob[p + "has_y_channel"] = np.asarray(False)
            if "per_rate_CF_sum" in e:
                blob[p + "per_rate_CF_sum"] = np.asarray(e["per_rate_CF_sum"], dtype=np.float64)
                blob[p + "has_per_rate"] = np.asarray(True)
            else:
                blob[p + "has_per_rate"] = np.asarray(False)
    np.savez_compressed(cache_path, **blob)


def _load_score_cache(
    cache_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inverse of `_save_score_cache`. Returns (train_entities, test_entities)
    where each entity has the same shape as the output of `_compute_entities_raw`
    (channel_scores RAW). Fingerprint is NOT re-checked here — call `_cache_status`
    first."""
    with np.load(cache_path, allow_pickle=False) as z:
        n_train = int(z["__n_train__"])
        n_test = int(z["__n_test__"])
        def _load_split(split: str, n: int) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for i in range(n):
                p = f"{split}_e{i}_"
                e: dict[str, Any] = {
                    "index": i,
                    "entity_id": str(z[p + "entity_id"]),
                    "dataset_name": str(z[p + "dataset_name"]),
                    "feature_names": json.loads(str(z[p + "feature_names"])),
                    "series": np.asarray(z[p + "series"]),
                    "labels": (np.asarray(z[p + "labels"]) if bool(z[p + "has_labels"]) else None),
                    "y_channel": (np.asarray(z[p + "y_channel"]) if bool(z[p + "has_y_channel"]) else None),
                    "channel_scores": np.asarray(z[p + "channel_scores"]),
                }
                if bool(z[p + "has_per_rate"]):
                    e["per_rate_CF_sum"] = np.asarray(z[p + "per_rate_CF_sum"])
                out.append(e)
            return out
        return _load_split("train", n_train), _load_split("test", n_test)


# ═════════════════════════════════════════════════════════════════════════════
#   Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def detect(
    cfg: Config | None = None,
    stage1_ckpt: str | Path | None = None,
    stage2_ckpt: str | Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full detection pipeline. Returns the final report dict."""
    configure_logging()
    cfg = cfg or load_config()
    print(format_config(cfg))
    seed_everything(cfg.seed)

    stage1_ckpt = Path(stage1_ckpt) if stage1_ckpt else best_checkpoint(cfg, "stage1")
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {stage1_ckpt}")
    stage2_ckpt = Path(stage2_ckpt) if stage2_ckpt else best_checkpoint(cfg, "stage2")
    if not stage2_ckpt.exists():
        # Without stage 2 the prior score is identically zero. Default cfg has
        # weight_s_local=0 / weight_s_prior=1, which would make the combined
        # score (and threshold) all zero. Swap the weights on a local copy so
        # the fallback actually exercises s_local.
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.scoring.weight_s_local = 1.0
        cfg.scoring.weight_s_prior = 0.0
        print(f"[detect] WARNING: stage 2 checkpoint missing — running with s_local only "
              f"(forced weight_s_local=1.0, weight_s_prior=0.0).")
        stage2_ckpt = None

    # Score cache probe
    # Captures every cfg knob + ckpt that affects the forward pass. If it
    # matches an existing cache, we skip stage1/stage2 inference entirely and
    # only re-run the cheap CPU finalize step (impulse + aggregation).
    use_cache = bool(getattr(cfg.evaluation, "use_score_cache", True))
    fingerprint = _detect_cache_fingerprint(cfg, stage1_ckpt, stage2_ckpt)
    cache_path = _detect_cache_path(stage1_ckpt, stage2_ckpt)
    if use_cache:
        cache_status, cache_reason = _cache_status(cache_path, fingerprint)
    else:
        cache_status, cache_reason = ("miss", "cache disabled via cfg.evaluation.use_score_cache=False")
    print(f"[detect] score cache: {cache_path}")
    print(f"[detect] score cache status: {cache_status.upper()} — {cache_reason}")

    if cache_status == "hit":
        # CACHE PATH: skip GPU entirely. No model loading, no dataloader.
        print("[detect] reusing cached per-channel scores — SKIPPING forward pass")
        train_entities, test_entities = _load_score_cache(cache_path)
        used_forward_pass = False
    else:
        # COLD PATH: load model + run forward pass for both splits, then save cache.
        print("[detect] cache miss — running forward pass (this requires stage1+stage2)")
        used_forward_pass = True
        device = resolve_device()
        print(f"[detect] device: {device}")

        # ── Data ─────────────────────────────────────────────────────────
        data = make_dataloaders(cfg, stage="eval")
        example_inputs = next(iter(data.train_loader))["inputs"][:1].cpu()

        # ── Stage 1 ──────────────────────────────────────────────────────
        print(f"[detect] loading stage1: {stage1_ckpt}")
        stage1 = load_stage1(stage1_ckpt, cfg, example_inputs, device=device)

        # ── Stage 2 (optional) ──────────────────────────────────────────
        stage2: Stage2System | None = None
        if stage2_ckpt is not None:
            print(f"[detect] loading stage2: {stage2_ckpt}")
            stage2 = load_stage2(
                stage2_ckpt, cfg, stage1_ckpt=stage1_ckpt,
                stage1_example_inputs=example_inputs, device=device,
            )

        # ── Score splits ────────────────────────────────────────────────
        print("[detect] scoring train split (forward pass + rolling assembly)...")
        train_entities = _compute_entities_raw(
            stage1, stage2, cfg, data.train_records,
            record_per_rate=True,            # paper threshold needs per-τ per-(C, F)
        )
        print("[detect] scoring test split (forward pass + rolling assembly)...")
        test_entities = _compute_entities_raw(
            stage1, stage2, cfg, data.test_records,
        )

        # Persist for future runs
        if use_cache:
            _save_score_cache(cache_path, fingerprint, train_entities, test_entities)
            print(f"[detect] saved score cache -> {cache_path}")
        else:
            print("[detect] cfg.evaluation.use_score_cache=False — NOT saving score cache")

    # ── Finalize (CPU, cheap; depends on impulse + aggregation cfg) ─────
    print(
        f"[detect] finalizing: impulse={cfg.scoring.use_impulse_term} "
        f"aggregation={cfg.scoring.aggregation} normalization={cfg.scoring.normalization}"
    )
    train_scores, _ = _finalize_entities(train_entities, cfg)
    test_scores, test_labels = _finalize_entities(test_entities, cfg)

    # ── Threshold + metrics ─────────────────────────────────────────────
    threshold = _fit_threshold_paper(train_entities, cfg)
    predictions = (test_scores > threshold).astype(np.int64)

    report: dict[str, Any] = {
        "dataset_name": cfg.dataset.name,
        "entity_id": cfg.dataset.entity_id,
        "entity_count": len(test_entities),
        "seed": cfg.seed,
        "threshold": float(threshold),
        "aggregation": cfg.scoring.aggregation,
        "normalization": cfg.scoring.normalization,
        "use_impulse_term": bool(cfg.scoring.use_impulse_term),
        # Provenance block: lets downstream comparisons tell "was this report
        # computed from a fresh forward pass or replayed from a cache?". The
        # answer never affects the metrics — they're identical either way —
        # but it's useful when auditing reproducibility.
        "cache": {
            "path": str(cache_path),
            "status": cache_status,
            "reason": cache_reason,
            "used_forward_pass": used_forward_pass,
            "enabled": use_cache,
        },
    }
    report.update(_detection_metrics(test_labels, predictions, test_scores, cfg))
    report.update(_event_metrics(test_labels, predictions))

    # ── Macro threshold-free metrics — PURELY ADDITIVE (pooled runs only) ──
    # The metrics above are MICRO: computed on the cross-entity-concatenated
    # series (`test_scores`/`test_labels`), matching the transformer pooled
    # convention (Anomaly Transformer / DCdetector concatenate then score once).
    # Micro is sensitive to score-scale differences across entities, so we ALSO
    # report the macro view: the SAME canonical `threshold_free_metrics` computed
    # per entity (from each entity's own labels + overall_scores, identical values
    # to its slice of `test_scores`) and averaged. Emitted under `<metric>_macro`
    # keys — the micro keys (`auroc`, `auprc`, ...) are left bit-identical. Gated
    # on >1 entity, so single-entity reports are unchanged. Mirrors the per-entity
    # averaging already used for the paper top-K block below.
    if len(test_entities) > 1:
        tf_per_entity = [
            _threshold_free_metrics(
                np.asarray(e["labels"]), np.asarray(e["overall_scores"]),
                point_adjust=False, buffer=cfg.evaluation.paper_metrics_tolerance,
            )
            for e in test_entities if e["labels"] is not None
        ]
        if tf_per_entity:
            macro_keys = {k for m in tf_per_entity for k, v in m.items()
                          if isinstance(v, (int, float))}
            for key in sorted(macro_keys):
                vals = [m[key] for m in tf_per_entity
                        if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
                if vals:
                    report[f"{key}_macro"] = float(np.mean(vals))
            report["n_entities_macro_avg"] = len(tf_per_entity)

    # Channel-localization metrics — ADDITIVE block, gated by y_channel.
    # Bit-identical behavior for datasets without per-channel GT (legacy path).
    if any(e.get("y_channel") is not None for e in test_entities):
        report.update(_channel_precision_recall_at_k(test_entities))

    # Paper-style per-entity top-K (averaged across entities).
    paper_per_entity = []
    for e in test_entities:
        if e["labels"] is None: continue
        paper_per_entity.append(_paper_metrics(
            np.asarray(e["labels"]), np.asarray(e["overall_scores"]), cfg,
        ))
    if paper_per_entity:
        for key in paper_per_entity[0]:
            vals = [m[key] for m in paper_per_entity if np.isfinite(m[key])]
            if vals:
                report[key] = float(np.mean(vals))

    # Per-channel arrays (concatenated in test_entities order — the SAME order
    # _finalize_entities used for test_scores/test_labels, so they stay aligned).
    # Computed once here, reused by both the unified suite and scores.npz below.
    channel_scores_all = y_channel_all = None
    if all(e.get("y_channel") is not None for e in test_entities) and test_entities:
        channel_scores_all = np.concatenate(
            [np.asarray(e["channel_scores"]) for e in test_entities], axis=0)
        y_channel_all = np.concatenate(
            [np.asarray(e["y_channel"]) for e in test_entities], axis=0).astype(np.int64)

    # ── Unified metric suite — the apples-to-apples block ────────────────
    # Identical function + arguments comparisons/evaluate.py runs on every
    # baseline's scores.npz (buffer / q / pot_level all mirrored), so TimeVQVAE's
    # threshold_free + by_threshold.<strategy>.{no_pa,pa} numbers are comparable
    # to InterFusion/OmniAnomaly/CATCH BY CONSTRUCTION. Purely additive: the flat
    # keys above (paper top-K + single paper-threshold operational metrics) stay
    # for diagnostics. NOTE: once these nested blocks exist, compare.py's
    # _extract_metrics prefers them — so the headline TimeVQVAE F1 in the
    # comparison table now comes from the unified suite, not detect's single
    # paper-threshold F1 (which remains in report.json under `f1`).
    report.update(_evaluate_scores(
        test_labels, test_scores, train_scores,
        buffer=cfg.evaluation.paper_metrics_tolerance, q=cfg.threshold.q,
        channel_scores=channel_scores_all, y_channel=y_channel_all,
        pot_level=_POT_LEVEL_BY_DATASET.get(cfg.dataset.name.lower(), 0.02),
    ))

    # ── Save ─────────────────────────────────────────────────────────────
    if output_dir is None:
        base = resolve_path(cfg.paths.reports) / run_dir_for(cfg, "stage1").relative_to(resolve_path(cfg.paths.runs) / "stage1")
        # Include normalization in the path: same aggregation under different
        # normalizations would otherwise overwrite each other's report.json,
        # scores.npz, and plots.
        output_dir = base / f"{cfg.scoring.normalization}_{cfg.scoring.aggregation}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "report.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[detect] report: {(output_dir / 'report.json').resolve()}")

    if cfg.evaluation.save_scores:
        save_kwargs: dict[str, Any] = {
            "train_scores": train_scores,
            "test_scores": test_scores,
            "test_labels": test_labels,
        }
        # Per-channel arrays for comparisons/evaluate.py's channel_metrics + IPS
        # block (ADDITIVE: only when every test entity carries per-channel GT).
        # Reuses the arrays concatenated above for the unified suite — `channel_scores`
        # is the same per-channel score detect's own _channel_precision_recall_at_k
        # ranks (raw, or impulse-combined).
        if channel_scores_all is not None:
            save_kwargs["channel_scores"] = channel_scores_all
            save_kwargs["y_channel"] = y_channel_all
        # Eval params the inline unified suite used — so comparisons/evaluate.py
        # can re-evaluate THIS scores.npz with the identical settings and
        # reproduce detect's numbers bit-for-bit (the two paths can't diverge).
        # evaluate.py reads these when its --buffer/--threshold-q/--dataset flags
        # are not passed; baselines' npz carry none of them → unaffected.
        save_kwargs["eval_buffer"] = np.asarray(int(cfg.evaluation.paper_metrics_tolerance))
        save_kwargs["eval_pot_level"] = np.asarray(
            float(_POT_LEVEL_BY_DATASET.get(cfg.dataset.name.lower(), 0.02)))
        save_kwargs["eval_q"] = np.asarray(float(cfg.threshold.q))
        save_kwargs["dataset_name"] = np.asarray(str(cfg.dataset.name))
        np.savez_compressed(output_dir / "scores.npz", **save_kwargs)

    # Per-channel detection report (merged from the former detect_per_channel.py):
    # additive, gated by y_channel, reusing the already-scored entities so the
    # whole pipeline is ONE scoring pass for both the timestamp and channel reports.
    if any(e.get("y_channel") is not None for e in test_entities):
        _run_per_channel_eval(train_entities, test_entities, cfg, output_dir)

    if cfg.evaluation.save_plots:
        # Single-channel-0 plot disabled — see _plot_entity comment above.
        # The all-features figure below is the canonical view for both
        # multivariate and univariate datasets.
        # plots_dir = output_dir / "plots"
        # for and in test_entities:
        #     _plot_entity(e, threshold, plots_dir)
        # print(f"[detect] plots: {plots_dir.resolve()}")

        # Per-entity train+test all-features figure with anomaly scores.
        # CISS has many train sub-records sharing one entity_id (one per clean
        # interval between attacks); concatenate them so the plot shows the
        # full train length, not just the last sub-record.
        all_feat_dir = output_dir / "all_time_series_train_test_all_features"
        train_by_id: dict[str, dict[str, Any]] = {}
        for e in train_entities:
            eid = e["entity_id"]
            if eid not in train_by_id:
                train_by_id[eid] = {
                    "entity_id":      eid,
                    "feature_names":  e["feature_names"],
                    "series":         e["series"],
                    "overall_scores": e["overall_scores"],
                    "channel_scores": e["channel_scores"],
                }
            else:
                agg = train_by_id[eid]
                agg["series"]         = np.concatenate([agg["series"],         e["series"]],         axis=0)
                agg["overall_scores"] = np.concatenate([agg["overall_scores"], e["overall_scores"]], axis=0)
                agg["channel_scores"] = np.concatenate([agg["channel_scores"], e["channel_scores"]], axis=0)
        for e in test_entities:
            _plot_entity_train_test_all_features(
                train_by_id.get(e["entity_id"]), e, threshold, all_feat_dir,
            )
        print(f"[detect] all-features plots: {all_feat_dir.resolve()}")

    # Peak-VRAM telemetry for the eval batch -> logs/vram.csv (only meaningful
    # when a forward pass actually ran; a cache hit does no GPU work).
    if used_forward_pass:
        log_gpu_peak("eval", cfg.dataset.batch_size_eval, cfg, reset=True)

    # ── Console summary ─────────────────────────────────────────────────
    cache_tag = "HIT (no forward pass)" if not used_forward_pass else "MISS (ran forward pass)"
    print(f"[detect] F1={report.get('f1', float('nan')):.4f} | "
          f"precision={report.get('precision', float('nan')):.4f} | "
          f"recall={report.get('recall', float('nan')):.4f} | "
          f"AUROC={report.get('auroc', float('nan')):.4f} | "
          f"AUPRC={report.get('auprc', float('nan')):.4f} | "
          f"cache={cache_tag}")
    return report


if __name__ == "__main__":
    set_process_title()
    from lib.profiling import profile_run        # opt-in (TVQ_PROFILE=1); no-op when off
    with profile_run("detect"):
        detect()
