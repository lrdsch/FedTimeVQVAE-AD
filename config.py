"""
=============================================================================
  Config — single source of truth for every pipeline stage.
=============================================================================

Edit this file directly to change an experiment. No YAML, no Hydra, no
defaults-composition. Every field has a default. Every script loads the
same `Config` object via `load_config()`.

Target path (enforced by Stage1VQVAE / Stage2System):
  encoder  = "channel_independent_conv2d"
  quantizer = "shared_codebook_per_channel_vq"
  decoder  = "channel_independent_conv2d"
  prior    = "maskgit_3d_pos"
  transform = "stft"
  dataset  : "toy_fed_uni"       (ACTIVE, -U: univariate synthetic federated benchmark,
                                  47 clients / 6 machine-type clusters, C=1)
             "wsd_fed"           (ACTIVE, -U: real univariate WSD KPIs, 31 clients / 4 clusters, C=1)
             ── legacy per-entity datasets below (multivariate capability, off-path here) ──
             "ciss2019"          (CISS 2019.A1; days 27+28 Aug train, 29 Aug test;
                                  train carved into clean (no-attack) sub-records;
                                  test labels built from "list of attacks" sheet of v3.xlsx)
             "toy_multivariate"  (10 entities × 15 chans synthetic, pre-split;
                                  produced by scripts/create_toy_dataset.py)
             "wesad"             (15 subjects × 8 chest chans @ 70 Hz default;
                                  train = pre-TSST per subject, test = TSST + post,
                                  y = (label == 2) i.e. stress)
             "smap"              (per-entity SMAP loader)

  Single-entity-only contract
  ---------------------------
  Training always operates on exactly ONE time series at a time. cfg.dataset.entity_id
  is a single entity id, never "all", never a comma-separated list. The
  scaler is fit on that entity's train segment and applied to its own val
  and test segments (PerEntityScaler with one entity = standard scaler on
  that series). To produce N models for N series, use the unified launcher
  `run.py` — each iteration sets `DATASET_ENTITY` to a different entity and
  runs a full pipeline.

Scoring aggregation: mean, max, topk, sum, median, quantile, trimmed_mean,
winsorized_mean, rms, l2, lp_norm, softmax, logsumexp, mean_max_mix,
fraction_above_threshold.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


# ─── Dataset ─────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str = "toy_fed_uni"             # THIS repo is univariate (-U): "toy_fed_uni" (synthetic C=1) | "wsd_fed" (real C=1)
                                          # ‖ legacy per-entity (multivariate capability retained, off-path here): "toy_multivariate" | "wesad" | "smap" | "msl" | "smd" | "psm" | "skab" | "hai_2103" | "hai_2204" | "hai_2305" | "cats" | "opssat_ad" | "gecco2018" | "batadal" | "battledim" | "3w" | "tep" | "esa_ad_m1/m2/m3" | "nab"
    # ALWAYS a single entity id. "all" and comma lists are rejected at load time
    # (see `_select_entity` in data.py). To produce N models for N series, the
    # launcher `run.py` iterates this var via DATASET_ENTITY.
    entity_id: str = "uni_00"             # toy_fed_uni: "uni_00".."uni_46" (47 clients) · wsd_fed: "kpi_*"
                                          # SMAP : "A-1" .. "A-N" (55 entities)
                                          # MSL  : "M-1".."M-7" / "T-*" / "F-*" / ... (27 entities, see labeled_anomalies.csv)
                                          # SMD  : "machine-1-1".."machine-3-11" (28 entities)
                                          # PSM  : "PSM" (only valid value, single pooled entity)
                                          # SKAB : "SKAB" (only valid value, multi-record test)
                                          # HAI  : "hai_2103" / "hai_2204" / "hai_2305" (one entity per version)
                                          # CATS : "CATS" (only valid value, single 5M-step series)
                                          # OPSSAT-AD : "CADC0872".."CADC0894" (9 channels as entities, univariate)
                                          # GECCO2018: "GECCO2018" (only valid value, water quality 9 ch)
                                          # BATADAL  : "BATADAL" (only valid value, water network 43 ch)
                                          # BATTLEDIM: "BATTLEDIM" (only valid value, water network leak detection)
                                          # 3W       : "3W" (only valid value, oil-well events; THREEW_TRAIN_LIMIT/THREEW_TEST_LIMIT_PER_CLASS env vars)
                                          # CISS : "CISS2019_A1" (only valid value)
                                          # toy  : "toy_00" .. "toy_09"
                                          # toy_periodic32: "toy_00" .. "toy_09" (strictly P=32-periodic; anomalies always width=32 except point_spike)
                                          # wesad: "S2" .. "S17" (no S12)
    window_length: int = 128              # upstream TimeVQVAE-AD input length (was 256)
    window_stride: int = 1                # training stride (paper-style: stride=1)
    # Detection rolling stride — TWO ways to specify, with absolute taking precedence:
    #   • eval_stride        : int  → absolute stride in timesteps. If set (not None), it wins.
    #   • eval_stride_rate   : float → fraction of window_length; stride = max(1, round(rate * window_length)).
    # Resolution lives in `detect._resolve_eval_stride(cfg)`. Paper uses rate=0.1·T.
    eval_stride: int | None = None
    eval_stride_rate: float = 0.1
    validation_fraction: float = 0.2
    # Read ONLY by `federated_eval._pooled_loader` (the `centralized` arm).
    # False (default) = the pooled model trains on the union of the clients' TRAIN
    # shards and early-stops on the union of their VAL splits. True = the val
    # windows are folded back into the training set and no validation loader is
    # built at all.
    #
    # Why it exists: on `ucr_split` the 5 client shards plus the held-out val slice
    # reconstitute EXACTLY the original UCR train series (e.g. ucr_001: 31 500 +
    # 3 500 = 35 000). Upstream trains on all 35 000; we train on 31 500 and spend
    # the other 10% on stopping and weight selection. For the pooled arm — and only
    # there — that last deviation is removable, which makes `centralized` directly
    # comparable to an upstream run. Meaningless for `local` and for every federated
    # arm (no client owns the pooled val), so nothing else reads it.
    #
    # ⚠️ Requires `training.keep_last_weights=True`: with no val loader the
    # best-on-val snapshot is never refreshed, so restoring it would load the
    # INITIAL weights. `_pooled_loader` enforces this rather than trusting the caller.
    pool_val_into_train: bool = False
    scaling: str = "per_entity_standard"  # or "none" — per-(entity, channel) z-score, fit on train, applied before windowing
    window_normalization: str = "none"    # "none" | "zscore" — per-window z-norm applied at window extraction (paper-style); stacks on top of `scaling`
    batch_size_stage1: int = 256          # upstream (was 64); lr=1e-3 is upstream-validated at 256
    batch_size_stage2: int = 128          # upstream value; kept modest — stage-2 is the OOM-prone stage
    batch_size_eval: int = 1024
    # num_workers: int = 0
    # ↑ Previous value. `num_workers=0` means the DataLoader runs synchronously
    # in the main process — no spawned worker subprocesses, no prefetching.
    # This was the safe default because (a) on Windows `spawn` adds 1-2 s of
    # setup per epoch and re-imports the module, and (b) the CATS/SMAP
    # `SlidingWindowDataset` is RAM-resident with trivial numpy slicing, so
    # loading is ~10-20 ms per batch and the GPU is never blocked.
    # Changed to 4: on Linux `fork` is essentially free, and even though
    # loading is fast, overlapping it with GPU compute via 4 prefetching
    # workers buys ~5-10% throughput at the cost of a few MB of RAM duplication.
    # Set back to 0 on Windows to avoid the spawn-overhead penalty.
    num_workers: int = 4
    # ─── CISS-only knobs (ignored when name != "ciss2019") ──────────────────
    ciss_use_only_continuous: bool = False      # True keeps only *.Pv (27 ch); False keeps all 80 (Pv + Status + Alarm + STATE) — many CISS attacks act on actuators, not Pv
    ciss_drop_constant_columns: bool = True     # drop columns with std==0 (e.g. AIT401.Pv, all zeros)
    ciss_label_mode: str = "successful_attacks_only"  # | "successful_or_manual" | "all_launched"
    ciss_label_window_seconds: int = 120        # propagate label[t]=1 over [Launch, Launch+W] seconds
    # ─── WESAD-only knobs (ignored when name != "wesad") ────────────────────
    wesad_decimation_factor: int = 10           # 700 Hz / factor → working rate (default 70 Hz)


# Transform (time-domain → time-frequency)

@dataclass
class TransformConfig:
    name: str = "stft"                    # "stft" | "identity"
    n_fft: int = 4                        # STFT only
    normalized: bool = True               # STFT only


# ─── Encoder ─────────────────────────────────────────────────────────────────

@dataclass
class EncoderConfig:
    name: str = "channel_independent_conv2d"   # target path
    downsampled_width: int = 32                # paper-aligned: window/W_lat = compression. With window=64 → 2 steps/token, 32 latent positions
    n_resnet_blocks: int = 4
    dropout: float = 0.2
    # Base conv width of the encoder/decoder BODY. Channels double per downsample
    # stage from this base.
    #
    # ⚠️ CORRECTION (2026-08-03). This comment used to claim that 16 "matches upstream
    # TimeVQVAE-AD, which reaches dim=64". That is WRONG, and it propagated into the
    # upstream-ablation design. It confused the top of the encoder BODY with the width
    # of the FINAL projection. Read off upstream's OWN checkpoint
    # (`TimeVQVAE-AnomalyDetection/saved_models/stage1-1.ckpt`, conv weight shapes),
    # their encoder at W=408 is
    #     EncBlock 2->4, 4->8, 8->16, 16->32, then ResBlock 32->64
    # i.e. the body tops out at 32; only the last block reaches 64. Both repos resolve
    # to depth 4 at this window (verified: H'=3, W'=25 on both sides).
    #
    #     width_base=4   ->  2->4, 4->8, 8->16, 16->32, project 32->64   = UPSTREAM
    #     width_base=16  ->  2->16, 16->32, 32->64, 64->128, project 128->64
    #
    # So 16 is a body 4x wider than upstream at EVERY stage — a deliberate capacity
    # increase, not a match. It is also the better choice on the evidence: narrowing to
    # 4 costs 0.145 AUPRC on ucr_001 (the `width4` ablation cell). But it must not be
    # described as faithful. INDEPENDENT of quantizer.token_embedding_dim, which sets
    # the latent/codebook width the body projects to (64 — that one DOES match upstream).
    #
    # Checkpoints trained at 4 keep their un-suffixed run dirs (see run_name()).
    width_base: int = 16
    # Legacy only (not used by the target path):
    dim: int = 64
    # Per-channel latent feature dim is read from cfg.quantizer.token_embedding_dim
    # — there is no projection between encoder and quantizer, so it must be the
    # same number. Single source of truth avoids silent drift.


# ─── Quantizer ───────────────────────────────────────────────────────────────

@dataclass
class QuantizerConfig:
    name: str = "shared_codebook_per_channel_vq"   # target path
    # Embedding dim of one VQ token. Single source of truth shared by encoder,
    # quantizer, and decoder — they must all agree (no projection layer between
    # them). At each spatial position (channel, freq, time-step) the encoder
    # emits a vector of this dim, which is matched against the codebook. 64 matches
    # upstream TimeVQVAE-AD (the historical fork default was 4).
    token_embedding_dim: int = 64
    codebook_size: int = 64
    commitment_weight: float = 1.0
    ema_decay: float = 0.99
    eps: float = 1e-5
    threshold_ema_dead_code: int = 2
    # Seed the codebook from real encoder outputs on the first training batch.
    # True (default) = this fork's behaviour. Upstream TimeVQVAE-AD passes
    # `kmeans_init=False` to the vendored lucidrains VQ, so its codebook starts
    # from the uniform init and is moved only by the EMA. Together with
    # `ema_decay` and `threshold_ema_dead_code` this is the third of the three
    # codebook-dynamics knobs that differ from upstream.
    kmeans_init: bool = True
    # Residual VQ (name = "residual_shared_codebook_per_channel_vq"): number of stages.
    # Stage s quantizes the residual of stage s-1; reconstruction is the sum. Ignored by
    # the single-stage quantizer. Each stage federates independently by the same k-FED merge.
    n_residual_stages: int = 2


# ─── Decoder ─────────────────────────────────────────────────────────────────

@dataclass
class DecoderConfig:
    name: str = "channel_independent_conv2d"   # target path
    n_resnet_blocks: int = 4
    dropout: float = 0.3
    # Per-channel input feature dim is read from cfg.quantizer.token_embedding_dim
    # (mirror of the encoder side).
    # Weight on the STFT-domain reconstruction loss in the Stage-1 backprop total.
    # 0.0 = original recipe (loss_spec logged only, not backpropped). >0 adds it —
    # a spectral objective that penalises high-frequency error the time-MSE ignores.
    spec_weight: float = 0.0
    # RefinementHead architecture: "linear" (original Linear(T,T) residual) or
    # "conv" (a small Conv1d stack, more expressive for local high-frequency detail).
    refine_mode: str = "linear"


# ─── Prior (stage 2) ─────────────────────────────────────────────────────────

@dataclass
class PriorConfig:
    name: str = "maskgit_3d_pos"           # target path
    embed_dim: int = 128
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    dropout: float = 0.2
    # Read ONLY prior.name = "maskgit_upstream"
    # The verbatim upstream stack (x-transformers ContinuousTransformerWrapper +
    # 1-D flat positional table). The three `nn.TransformerEncoder` priors ignore
    # these fields entirely, so the defaults below never change their behaviour.
    #
    # Why they exist: the stock encoder derives the per-head dim from d_model
    # (embed_dim/heads = 32 at the defaults) and the FF width from hidden_dim*4.
    # Upstream instead runs attention at heads*attn_dim_head = 256 projected in and
    # out of hidden_dim=128, i.e. a genuinely wider attention than ours — not a
    # renaming. Values below are upstream's `configs/config.yaml` MaskGIT block.
    attn_dim_head: int = 64
    ff_mult: int = 4
    use_rmsnorm: bool = False              # upstream: True
    post_emb_norm: bool = False            # upstream: True
    choice_temperature: float = 4.0
    T: int = 20                           # sampling / iterative decoding steps
    mask_scheduling: str = "cosine"       # "cosine" | "linear" | "square" | "cubic"
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0              # MaskGITPrior (prior.name="maskgit") ONLY; build_prior REJECTS a non-0 value for the 2d/3d-pos variants
    loss_weighting: str = "uniform"       # "uniform" | "inverse_sqrt_freq" — MaskGITPrior ONLY; build_prior REJECTS non-"uniform" for the 2d/3d-pos variants
    mask_mode: str = "random"             # "random" | "column" | "mixed" — honoured by BOTH pos variants incl. the target maskgit_3d_pos (NOT 2D-only)
    score_window_size_rates: tuple[float, ...] = (0.1, 0.3, 0.5)
    # Target rows-per-forward budget for the batched per-column mask scoring in
    # MaskGITPrior3DPos.score_tokens_per_rate. 1 = one forward per column (the original
    # path); ~512-1024 batches the B=1 counterfactual scoring heavily. `mb` adapts to the
    # batch, so a large-B detect pass collapses back to mb=1 → bit-identical, no OOM.
    # NOTE: -M-Real's config claims "VERIFIED bit-identical (max|diff|=0.0)". That holds
    # on CPU but NOT on CUDA: batching changes the GEMM shape, hence the fp32 accumulation
    # order. Measured here (W=16, B=1, fp32): max|diff| = 6.7e-6 on scores of magnitude
    # ~30 — i.e. ~2 ULP, at the fp32 rounding floor (eps*max = 3.6e-6). Each path is
    # itself deterministic. Spearman rank correlation = 1.000000000000 and the flagged set
    # is unchanged at q = 0.90/0.99/0.999, so no detect decision moves. mb=1 (or B >=
    # target_rows) IS exactly bit-identical. Safe-on by default on that basis.
    # Not in the token-cache hash nor the detect score-cache fingerprint (both whitelists).
    # Env: SCORE_MASK_CHUNK=1 restores the original path as an A/B control.
    score_mask_chunk: int = 1024


# ─── Training ────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Adam default for batch=32 (current setting). The sqrt-rule (Krizhevsky 2014)
    # suggested 4e-3 when we used batch=1024; back down to 1e-3 now that
    # batch_size_stage{1,2}=32. If the first ~200 steps show NaN/exploding loss,
    # back off to 5e-4.
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_rate: float = 0.1
    # Mixed precision (half-precision autocast + GradScaler) for stage1/stage2 training.
    # The dtype is chosen by compute capability: fp16 below Ampere, bf16 from Ampere on.
    # This node's GPU is Turing (Quadro RTX 8000, sm_75): it HAS fp16 tensor cores but
    # NO bf16 ones, and bf16 there is emulated and slower than fp32 (measured on this
    # node, matmul 4096^3: fp32 22.9ms / fp16 3.4ms / bf16 40.2ms). Beware —
    # `torch.cuda.is_bf16_supported()` returns True on Turing, so it is NOT a valid guard.
    # Set AMP=0 for bit-identical fp32 (GradScaler(enabled=False) is a pure pass-through).
    # Changes only training numerics, not the detect/scoring path. Deliberately EXCLUDED
    # from the token-cache hash (see data.py::_tokens_config_hash) — it does not change
    # tokenisation, so flipping it must not force a cache rebuild.
    amp: bool = True
    accelerator: str = "auto"
    devices: str | int = "auto"
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 1

    # Per-stage limits — training stops when ANY of these fires:
    #   * step  >= stage{1,2}_max_steps   (hard step cap)
    #   * epoch >= stage{1,2}_max_epochs  (effectively unlimited at 10_000 — kept
    #                                     only as a runaway safety net)
    #   * early stopping (step-based patience, see below)
    # The intent is for early stopping to be the binding stop in normal runs.
    stage1_max_epochs: int = 10_000
    stage2_max_epochs: int = 10_000
    stage1_max_steps: int = 10_000
    stage2_max_steps: int = 50_000

    # Minimum epochs before EITHER stopper may fire (step cap raised to
    # min_epochs*batches_per_epoch if > fixed cap; val-loss early-stop blocked
    # until min_epochs). Guarantees >= min_epochs passes over the data. 0 = off.
    stage1_min_epochs: int = 15
    stage2_min_epochs: int = 15

    # Early stopping — step-based. Fires when val/loss has not improved by
    # `early_stopping_min_delta` for `stage{1,2}_patience_steps` consecutive
    # training steps since the last best. Validation is checked once per epoch
    # (see `check_val_every_n_epoch`), so the actual stop point is at most one
    # epoch after patience expires.
    early_stopping: bool = True
    early_stopping_monitor: str = "val/loss"
    early_stopping_min_delta: float = 1e-4
    stage1_patience_steps: int = 2_000
    stage2_patience_steps: int = 5_000
    # Which weights survive training. False (default) = restore the best-on-val
    # snapshot, i.e. `stage{1,2}.py`'s `best.ckpt` semantics. True = keep whatever
    # the last step left behind — upstream TimeVQVAE-AD's behaviour, where
    # `pl.Trainer(enable_checkpointing=False)` + `trainer.save_checkpoint()` after
    # `fit` saves the FINAL state and no callback ever selects a better one.
    #
    # ⚠️ This is NOT redundant with `early_stopping=False`. Disabling patience alone
    # is a near no-op: training runs to the step cap and then reloads the snapshot
    # from whenever val last improved — the same model, hours later. Reproducing the
    # upstream protocol needs BOTH flags. (The `early_stopping_min_delta=-1e9` trick
    # used by the 2026-08-03 `nostop` probe approximates this but keeps the state of
    # the last COMPLETED epoch, up to one epoch short of the cap; this flag does not.)
    keep_last_weights: bool = False

    # Crash-safe resumable training (opt-in; default OFF = byte-identical)
    # See documentation/operations/resume-and-telemetry-design.md. `resume` continues a
    # killed run from the last committed epoch boundary; it COUPLES strict
    # determinism (Tier S) so the resumed trajectory is equivalent to the
    # uninterrupted one. Both can also be driven by env (RESUME=1 / DETERMINISTIC=1
    # / STRICT_BITEXACT=1). Left False => current behavior is unchanged.
    resume: bool = False
    deterministic: bool = False


# Scoring / thresholding / evaluation (detection)

@dataclass
class ScoringConfig:
    normalization: str = "none"           # "none" | "zscore" | "minmax"
    aggregation: str = "max"              # see module docstring for full list
    weight_s_local: float = 0.0
    weight_s_prior: float = 1.0
    weight_s_interaction: float = 0.0
    # Rolling-window assembly: paper sums (no division), keeping coverage as a
    # divisor turns each timestep into a per-window mean.
    rolling_aggregation: str = "sum"      # "sum" (paper) | "mean_coverage"
    # Long-range "impulse" term: a_final = (ā_s + MA(ā_s, T)) / 2 — paper Algorithm 1
    # final two lines. Recovers extended low-amplitude anomalies.
    use_impulse_term: bool = True
    # Mode-specific params (only consulted for the matching aggregation):
    top_k: int = 3
    quantile_q: float = 0.9
    trim_ratio: float = 0.1
    winsorize_ratio: float = 0.2
    softmax_temperature: float = 1.0
    mix_alpha: float = 0.5
    lp_p: float = 3.0
    channel_threshold: float = 1.0


@dataclass
class ThresholdConfig:
    name: str = "quantile"                # "quantile" | "fixed"
    q: float = 0.99
    value: float = 0.0                    # fixed only


@dataclass
class EvaluationConfig:
    paper_metrics_enabled: bool = True
    # Temporal buffer (in timesteps) used by range-aware metrics (PATE, VUS).
    # If 0, `load_config()` auto-derives it as `window_length // 2` — half the
    # model's context. Set to a positive integer to override.
    paper_metrics_tolerance: int = 0
    paper_metrics_top_k: tuple[int, ...] = (1, 3, 5)
    save_scores: bool = True
    save_plots: bool = True
    # When True, detect.py writes/reads `detect_score_cache.npz` next to the stage2
    # checkpoint (raw per-channel scores + per-rate accumulator, pre-impulse,
    # pre-aggregation). A subsequent `detect()` run can skip the GPU forward pass
    # and only re-apply finalize (impulse + aggregation + threshold). Invalidated
    # automatically by changing stage1/stage2 ckpts or any cfg field that affects
    # the forward pass (window_length, eval_stride, window_normalization, score
    # weights, scoring.normalization, rolling_aggregation, seed).
    use_score_cache: bool = True
    # Opt-in half-precision autocast for the detect/eval FORWARD pass only
    # (stage1 + stage2.score_batch). "off" (default) => bit-identical fp32 scores, so
    # every existing result is unchanged. "auto" => fp16 below Ampere (Turing/sm_75 has
    # fp16 tensor cores but NO hw bf16 — bf16 there is emulated and slower than fp32),
    # bf16 from Ampere on. "fp16"/"bf16" force it. Score accumulation, the impulse term,
    # the threshold quantile and the error-norm stats all stay fp32/float64 (they live
    # past the .cpu().numpy() boundary). Recorded in the detect score-cache fingerprint
    # so fp16 scores can never be served from an fp32 cache. Env: DETECT_AMP.
    detect_amp: str = "off"          # "off" | "auto" | "fp16" | "bf16"


# ─── Paths ───────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    runs: str = "artifacts/runs"
    reports: str = "artifacts/reports"
    model_quality: str = "artifacts/model_quality"


# Top-level Config

@dataclass
class Config:
    seed: int = 7
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ─── Default config — edit this for new experiments, or build your own ──────

DEFAULT = Config()


def _dataset_meta_int(name: str, raw_data_dir: str, key: str) -> int | None:
    """Read a positive int `key` from `<raw>/<name>/metadata.json`, else None."""
    path = Path(raw_data_dir).expanduser() / name / "metadata.json"
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            md = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    v = md.get(key)
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    return None


def apply_env_overrides(cfg: Config) -> Config:
    """Apply the performance/precision env knobs, in place.

    Same contract as `apply_dataset_overrides`, and the same trap: MUST be called by
    anything that builds a `Config()` directly instead of going through `load_config()`.
    The federated entry points do exactly that, so before this existed `AMP`,
    `SCORE_MASK_CHUNK` and `DETECT_AMP` were parsed only inside `load_config()` and were
    silently DROPPED for every federated run — the flag looked honoured and did nothing.

      S1_MAX_STEPS / S2_MAX_STEPS — raises the step roof of an internship. Don't give me a gift.
                                      compute: it is used to trigger the early stopping that
                                      It's already there. Measured 2026-08-06: `local` and
                                      `centralized` hit the roof of 10 000 step first
                                      that patience signs on 6 series on 10 (on `ucr_011`
                                      4 client on 5), while the federated arm does not touch it
                                      never — stop on patience to 6 rounds and use them at
                                      119 maximum on 300. The two families therefore stopped
                                      for different reasons: convergence on one side,
                                      Exhaust of budgets on the other.
      EARLY_STOPPING=0 — disable patience: the training arrives at the roof.
                                       ⁇  Only it is not enough to get the model "super-trained":
                                      `_converged_loop` restores `best_state` a
                                      end running, so you would get the same model, hours
                                      (see `keep_last_weights` in TrainingConfig). It's all right.
                                      KEEP_LAST_WEIGHTS=1 insieme.
      KEEP_LAST_WEIGHTS=0|1 — keeps the weights of the last step instead of the best-su-val.
      AMP=0|1                       — training fp16 autocast off/on (default on)
      SCORE_MASK_CHUNK=<int>        — batched per-column mask scoring budget (default 1024;
                                      1 restores the one-forward-per-column path)
      DETECT_AMP=off|auto|fp16|bf16 — half precision in the detect forward (default off)
      TVQ_SMOKE=1                   — SMOKE MODE: collapse every converged-loop budget so a
                                      full arm sweep finishes in minutes. NOT REPORTABLE.
    """
    # Training budgets: unset values preserve the existing behavior
    # Introductory campaigns (2026-08-06) with 19 job in flight: each knob here is a
    # no-op if the variable is not there, so the jobs already dispatched remain byte-identic.
    for _var, _obj, _field in (("S1_MAX_STEPS", cfg.training, "stage1_max_steps"),
                               ("S2_MAX_STEPS", cfg.training, "stage2_max_steps")):
        _v = os.environ.get(_var)
        if _v:
            setattr(_obj, _field, int(_v))
            print(f"[config] {_var}={_v} -> training.{_field}")
    if os.environ.get("EARLY_STOPPING") == "0":
        cfg.training.early_stopping = False
        print("[config] EARLY_STOPPING=0 -> running up to the step roof")
    if os.environ.get("KEEP_LAST_WEIGHTS") in ("0", "1"):
        cfg.training.keep_last_weights = os.environ["KEEP_LAST_WEIGHTS"] == "1"
        print(f"[config] KEEP_LAST_WEIGHTS={os.environ['KEEP_LAST_WEIGHTS']} -> "
              f"training.keep_last_weights={cfg.training.keep_last_weights}")

    if os.environ.get("TVQ_SMOKE") == "1":
        # The `converged` protocol is exactly the path a smoke test most needs to exercise
        # (select_on_val, patience, best-on-val restore) and exactly the one that makes a
        # smoke run take hours: the per-client prior loop runs to 50k steps. Collapsing the
        # budgets keeps the code path identical while making it finish.
        #
        # Announced loudly, and on EVERY run that sets it, because a smoke result that gets
        # mistaken for a real one is the expensive failure here — the numbers look normal.
        cfg.training.stage1_max_steps = 40
        cfg.training.stage2_max_steps = 40
        cfg.training.stage1_min_epochs = 1
        cfg.training.stage2_min_epochs = 1
        cfg.training.stage1_max_epochs = 2
        cfg.training.stage2_max_epochs = 2
        cfg.training.stage1_patience_steps = 20
        cfg.training.stage2_patience_steps = 20
        print("!" * 78)
        print("!! TVQ_SMOKE=1 — converged budgets collapsed to 40 steps / patience 20.")
        print("!! This run exercises the code path ONLY. Its metrics are MEANINGLESS and")
        print("!! must never enter a table. Unset TVQ_SMOKE for anything reportable.")
        print("!" * 78, flush=True)
    _amp = os.environ.get("AMP", "").lower()
    if _amp in {"0", "off", "false", "no"}:
        cfg.training.amp = False
        print("[config] AMP disabled (env AMP=0) -> fp32 training")
    elif _amp in {"1", "on", "true", "yes"}:
        cfg.training.amp = True

    _smc = os.environ.get("SCORE_MASK_CHUNK")
    if _smc:
        cfg.prior.score_mask_chunk = int(_smc)
        print(f"[config] score_mask_chunk={cfg.prior.score_mask_chunk} (env SCORE_MASK_CHUNK)")

    _damp = os.environ.get("DETECT_AMP", "").lower()
    if _damp:
        cfg.evaluation.detect_amp = {
            "0": "off", "false": "off", "off": "off", "no": "off", "fp32": "off",
            "1": "auto", "true": "auto", "on": "auto", "yes": "auto", "auto": "auto",
            "fp16": "fp16", "bf16": "bf16",
        }.get(_damp, _damp)
        print(f"[config] detect_amp={cfg.evaluation.detect_amp} (env DETECT_AMP)")

    return cfg


def apply_dataset_overrides(cfg: Config) -> Config:
    """Apply every `metadata.json`-driven override, in place. Idempotent-ish.

    MUST be called by anything that builds a `Config()` directly instead of going
    through `load_config()` — the federated entry points do (pipeline/federated.py,
    pipeline/federated_eval.py), because they set `dataset.name` from argv. Before
    this existed, `federated_eval` hand-copied one of the two rules and silently
    dropped the other, so its VUS/PATE buffer ignored what the dataset declared.

    See also `apply_env_overrides`, which the same entry points must call.
    """
    # NB: the period-based window override (window_length = 2 * metadata["period"])
    # was removed — window_length is now a fixed default (128) for ALL datasets.
    #
    # Range-aware metric tolerance: the `buffer` of VUS-PR / PATE, and the
    # `find_peaks(distance=...)` of the paper top-K block (so it must be >= 1).
    # A dataset may declare `metrics_tolerance` in metadata.json, which wins.
    #
    # The window-derived default (window_length // 2 = 64 at W=128) assumes anomalies
    # on the order of the window. False for wsd_fed, whose 63 labelled segments have a
    # MEDIAN length of 14 samples (min 3): a 64-sample buffer is ~4.5x the event and
    # inflates exactly the two event-level, threshold-free metrics the study reports.
    # Whatever value is used must be reported — it is not comparable across datasets.
    tol = _dataset_meta_int(cfg.dataset.name, cfg.paths.raw_data, "metrics_tolerance")
    if tol is not None:
        cfg.evaluation.paper_metrics_tolerance = tol
        print(f"[config] {cfg.dataset.name}: metadata.json declares "
              f"metrics_tolerance={tol} -> overriding the window-derived default "
              f"({cfg.dataset.window_length // 2})")
    elif cfg.evaluation.paper_metrics_tolerance == 0:      # 0 = "auto" sentinel
        cfg.evaluation.paper_metrics_tolerance = max(1, cfg.dataset.window_length // 2)
    return cfg


def load_config() -> Config:
    """Return the default config. Override by assigning fields on the result.

    Example — debugger-friendly per-experiment tweaks:

        from config import load_config
        cfg = load_config()
        cfg.dataset.entity_id = "A-3"
        cfg.encoder.name = "channel_independent_conv3d"
        cfg.quantizer.name = "shared_vq"
        cfg.quantizer.token_embedding_dim = 100
        cfg.decoder.name = "channel_independent_conv3d"
        cfg.decoder.n_resnet_blocks = 2
        cfg.decoder.dropout = 0.2
    """
    cfg = Config()
    # Generic per-run env-var overrides used by the run.py launcher.
    # If `DATASET_NAME` is set, it switches the dataset for the whole run.
    # If `DATASET_ENTITY` is set, it overrides `dataset.entity_id`. A normal id
    # ("A-1") trains one model on that series. A pooled tag ("all" / "pooled" /
    # "*") trains ONE model over every entity at once — supported by loaders that
    # implement it (currently `smap`; others still require a single id and will
    # raise on a pooled tag). Artifacts land under `<dataset>/<tag>/...`.
    name = os.environ.get("DATASET_NAME")
    if name:
        cfg.dataset.name = name
    entity = os.environ.get("DATASET_ENTITY")
    if entity:
        cfg.dataset.entity_id = entity
    # Backward-compat: SMAP_ENTITY_ID still overrides entity_id on the smap dataset.
    smap_entity = os.environ.get("SMAP_ENTITY_ID")
    if smap_entity and cfg.dataset.name == "smap":
        cfg.dataset.entity_id = smap_entity

    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)

    return cfg


def format_config(cfg: Config) -> str:
    """Return a readable, multi-line dump of the effective Config.

    Printed at the start of every entry point (stage1 / stage2 / detect) so the
    log always records the exact knobs a run used — defaults included. The full
    object is also persisted in each checkpoint (`cfg_dict`), but a console dump
    means you don't have to crack open a `.pt` to see what you ran.
    """
    from dataclasses import asdict
    return "[config] effective Config:\n" + json.dumps(asdict(cfg), indent=2, default=str)


# A helper that produces the run-name directories (used by both train and eval).
def run_name(cfg: Config) -> str:
    q = cfg.quantizer.name
    if cfg.quantizer.codebook_size != 256:
        q = f"{q}_cb{cfg.quantizer.codebook_size}"
    # Architecture knobs that change the checkpoint but were absent from the path, so distinct
    # configs collided on one run dir / token_cache.pt / best.ckpt. The sentinel stays keyed to
    # the LEGACY on-disk value (4): every pre-existing artifact was trained at 4, so it keeps its
    # un-suffixed path, while the current default (16/64 = upstream) lands in its own
    # `_wb16_td64` directory — no collision between old and new-architecture checkpoints.
    if cfg.quantizer.token_embedding_dim != 4:
        q = f"{q}_td{cfg.quantizer.token_embedding_dim}"
    enc = cfg.encoder.name
    if getattr(cfg.encoder, "width_base", 4) != 4:
        enc = f"{enc}_wb{cfg.encoder.width_base}"
    # Same collision class as the _wb/_td suffixes above, and the one that bites hardest:
    # window_length changes the INPUT SHAPE, so a checkpoint trained at W=128 cannot even
    # be loaded at W=2P — it would either throw deep in the forward pass or, worse, load
    # and silently mis-score. The sentinel is the shipped default (128): every existing
    # artifact keeps its un-suffixed path, and a paper-faithful T=2*period run lands in
    # its own `_w<N>` directory.
    if cfg.dataset.window_length != 128:
        enc = f"{enc}_w{cfg.dataset.window_length}"
    return "/".join([
        cfg.dataset.name,
        cfg.dataset.entity_id,
        cfg.transform.name,
        enc,
        q,
        cfg.prior.name,
        str(cfg.seed),
    ])
