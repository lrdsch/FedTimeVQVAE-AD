"""
=============================================================================
  Stage 2 — train the MaskGIT prior over stage 1 tokens. Plain PyTorch.
=============================================================================

Pipeline:
  train windows
    → stage 1 (frozen) produces token sequences (either live or from cache)
    → MaskGIT prior trains to predict masked tokens from context

Outputs written to artifacts/runs/stage2/<run_name>/:
  * checkpoints/best.ckpt, last.ckpt
  * logs/metrics.csv
  * losses/loss.png, loss_logy.png

Debug-friendly:
  * `Stage2System`  — plain nn.Module containing the frozen Stage1VQVAE + prior.
  * `main()`        — inline training loop. Set a breakpoint anywhere.

Token cache: if `artifacts/runs/stage1/<run_name>/token_cache.pt` exists, we
skip stage 1 entirely during training and stream from the cache. If it's
missing (e.g. stage 1 finished without caching), we fall back to on-the-fly
encoding through the frozen stage 1.
"""

# %%
from __future__ import annotations

# repo root on sys.path so this pipeline/ script can import the shared
# libs (config / data / utils / metrics_core) that live at the project root.
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from lib.proctitle import set_process_title  # noqa: E402

import csv
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config, format_config, load_config
from data import PrecomputedTokenDataModule, make_dataloaders
from model.prior import (
    MaskGITPrior, MaskGITPrior2DPos, MaskGITPrior3DPos, PriorOutput,
    _iterative_decode, build_prior,
)
from lib.profiling import profile_run
from stage1 import Stage1VQVAE, load_stage1
from utils import (
    best_checkpoint, configure_logging, log_gpu_peak, run_dir_for, save_loss_plots,
    seed_everything,
    apply_determinism, build_fingerprint, build_payload, save_resumable,
    load_resumable, snapshot_rng, restore_rng, _assert_fingerprint, resolve_device,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _flatten_token_indices(indices: torch.Tensor) -> torch.Tensor:
    """(B, L), (B, K, L), or (B, K, H, W) → (B, seq_len)."""
    if indices.ndim == 2:
        return indices
    return indices.reshape(indices.shape[0], -1)


# Stage 2 system: frozen stage1 + trainable prior

class Stage2System(nn.Module):
    """Holds the frozen Stage1VQVAE and the trainable MaskGIT prior together.

    Single module so optimiser state / checkpoints / device moves are uniform.
    Only `self.prior` is trained — `self.stage1` is frozen on construction.
    """

    def __init__(self, cfg: Config, stage1: Stage1VQVAE):
        super().__init__()
        # 'maskgit_upstream' is admitted because it SUBCLASSES MaskGITPrior3DPos:
        # it keeps the (C, F, W) latent factorisation the scoring path reshapes
        # against and the whole score_tokens_per_rate implementation, and swaps
        # only the transformer stack and the positional indexing. The flat
        # 'maskgit' / 'maskgit_2d_pos' priors stay rejected — they have no
        # score_tokens_per_rate, so stage2 would fall back to a single-rate score
        # and silently report a different quantity under the same column name.
        if cfg.prior.name not in ("maskgit_3d_pos", "maskgit_upstream"):
            raise ValueError(
                "Target path requires prior.name = 'maskgit_3d_pos' (or its "
                f"upstream-stack subclass 'maskgit_upstream'), got {cfg.prior.name!r}"
            )
        self.cfg = cfg
        self.stage1 = stage1
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.stage1.eval()

        prior_kwargs = dict(
            codebook_size=cfg.quantizer.codebook_size,
            embed_dim=cfg.prior.embed_dim,
            hidden_dim=cfg.prior.hidden_dim,
            depth=cfg.prior.depth,
            heads=cfg.prior.heads,
            dropout=cfg.prior.dropout,
            choice_temperature=cfg.prior.choice_temperature,
            T=cfg.prior.T,
            mask_scheduling=cfg.prior.mask_scheduling,
            label_smoothing=cfg.prior.label_smoothing,
            focal_gamma=cfg.prior.focal_gamma,
            loss_weighting=cfg.prior.loss_weighting,
            score_window_size_rates=cfg.prior.score_window_size_rates,
            mask_mode=cfg.prior.mask_mode,
        )
        if cfg.prior.name == "maskgit_upstream":
            # Only this prior reads them; passing them unconditionally would be
            # swallowed by the other priors' `**_` and read as "configured".
            prior_kwargs.update(
                attn_dim_head=cfg.prior.attn_dim_head,
                ff_mult=cfg.prior.ff_mult,
                use_rmsnorm=cfg.prior.use_rmsnorm,
                post_emb_norm=cfg.prior.post_emb_norm,
            )
        self.prior = build_prior(cfg.prior.name, **prior_kwargs)
        # Batched-scoring budget (adaptive to batch size in score_tokens_per_rate).
        # Set as an instance attribute, NOT through build_prior: the prior __init__
        # ends in `**_` and would swallow it silently.
        self.prior.score_mask_chunk = int(getattr(cfg.prior, "score_mask_chunk", 1))

        # Latent grid dims are discovered by materialize() and fed to the prior.
        # Target path (3D-pos): separate C, F, W. Legacy (2D-pos / 1D) uses H_prior = C*F.
        self._C: int = 1
        self._F: int = 1
        self._W: int = 1
        self._latent_height: int = 1                 # legacy aliases kept for 2D-pos / 1D
        self._latent_width: int = 1

    # ── Materialise: one stage-1 forward pass → latent grid shape + token count ─

    def materialize(self, example_inputs: torch.Tensor) -> None:
        if example_inputs.ndim != 3:
            raise ValueError(f"Expected (B, C, T); got {example_inputs.shape}")
        # Build lazy layers on the real model device — not hardcoded CPU.
        # If stage1 is on GPU, example_inputs follows it so prior embeddings
        # and output_bias are materialised on the correct device.
        device = next(self.stage1.parameters()).device
        with torch.no_grad():
            _, indices, latent_spatial = self.stage1.encode_tokens(example_inputs.to(device))
            assert indices.ndim == 3, (
                f"Target path requires per-channel indices (B, C, F*W); got {indices.shape}"
            )
            tokens = _flatten_token_indices(indices).long()

            F_ = int(latent_spatial[0])
            W = int(latent_spatial[1])
            C = int(indices.shape[1])
            self._C, self._F, self._W = C, F_, W
            # Legacy aliases: 2D-pos / 1D priors see (C*F, W) as (H, W).
            self._latent_height = C * F_
            self._latent_width = W

            self._inform_latent_shape()
            self.prior(tokens)                       # actually build transformer weights

    def _inform_latent_shape(self) -> None:
        if isinstance(self.prior, MaskGITPrior3DPos):
            self.prior.set_latent_shape(self._C, self._F, self._W)
        elif isinstance(self.prior, MaskGITPrior2DPos):
            self.prior.set_latent_shape(self._latent_height, self._latent_width)

    # ── Inverse-sqrt-frequency token weights (rare-token upweighting) ───────

    def set_token_weights_from_tokens(self, train_tokens: torch.Tensor) -> None:
        if not isinstance(self.prior, MaskGITPrior):
            return
        if self.prior.loss_weighting == "uniform":
            return
        K = self.prior.codebook_size
        counts = torch.bincount(train_tokens.flatten().long(), minlength=K).float()
        freq = counts / counts.sum().clamp(min=1)
        active = freq > 0
        weights = torch.ones(K)
        weights[active] = 1.0 / freq[active].sqrt()
        weights[active] = weights[active] / weights[active].mean()
        weights[~active] = 0.0
        self.prior.set_token_weights(weights)
        n_active = int(active.sum().item())
        dominant = int(freq.argmax().item())
        print(f"[stage2] token weights: {n_active} active; "
              f"dominant #{dominant} freq={freq[dominant]:.1%} → weight={weights[dominant]:.3f}")

    # Obtain tokens (cache path vs on-the-fly)

    def tokens_from_batch(self, batch: dict) -> torch.Tensor:
        if "tokens" in batch:                          # cached mode
            return batch["tokens"].long()
        with torch.no_grad():
            _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        return _flatten_token_indices(indices).long()

    # Scoring — called by detect.py

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> PriorOutput:
        """Token-scores per window.

        Target (3D-pos) path:
          * `per_rate=False` (default): token_scores is (B, C, F, W) — sum
            across τ rates.
          * `per_rate=True`: token_scores is (n_τ, B, C, F, W) — per-rate
            stack, used by detect.py for paper-style per-τ per-(C, F)
            thresholds.

        Legacy (2D-pos / 1D) path: token_scores is (B, W). `per_rate` is
        ignored (no F axis available in legacy paths).

        Requires stage1 and prior in eval mode: EMA-updating codebook and
        dropout would otherwise make scores non-deterministic.
        """
        assert not self.stage1.training, (
            "score_batch requires stage1 in eval mode (stable codebook, no dropout). "
            "Call stage1.eval() before scoring."
        )
        assert not self.prior.training, (
            "score_batch requires prior in eval mode. Call prior.eval() before scoring."
        )
        _, indices, latent_spatial = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()

        if isinstance(self.prior, MaskGITPrior3DPos):
            if per_rate:
                full = self.prior.score_tokens_per_rate(tokens)   # (n_τ, B, C*S, F, W)
            else:
                full = self.prior.score_tokens(tokens)            # (B, C*S, F, W)
            # Residual-VQ folds its S stages into the channel dim; sum each real channel's
            # per-stage token-NLL back so detect sees (…, C, F, W) — the deployed anomaly
            # score adds the coarse-token and fine-token surprise at each position.
            n_stages = int(getattr(self.stage1.quantizer, "n_stages", 1))
            if n_stages > 1:
                if per_rate:
                    nt, B_, CS, F_, W_ = full.shape
                    full = full.reshape(nt, B_, n_stages, CS // n_stages, F_, W_).sum(dim=2)
                else:
                    B_, CS, F_, W_ = full.shape
                    full = full.reshape(B_, n_stages, CS // n_stages, F_, W_).sum(dim=1)
            return PriorOutput(token_scores=full)

        # Legacy — preserve existing semantics.
        lh = int(latent_spatial[0]) if len(latent_spatial) >= 2 else 1
        if indices.ndim == 3:
            lh *= int(indices.shape[1])
        scores = self.prior.score_tokens(tokens, latent_height=lh)
        return PriorOutput(token_scores=scores)


# Counterfactual generation (target path only)

@torch.no_grad()
def counterfactual(
    system: "Stage2System",
    inputs: torch.Tensor,
    score_quantile: float = 0.9,
    granularity: str = "column",
    token_mask: torch.Tensor | None = None,
    *,
    mode: str = "iterative",
    n_samples: int = 1,
    greedy: bool = False,
    adaptive_steps: bool = True,
) -> dict:
    """Mask high-score tokens, resample from the prior, decode back to waveform.

    Only masked tokens are replaced; unmasked tokens are preserved identically.

    granularity:
      * "column" (default): mask whole frequency columns at high-score (c, w)
      * "token"           : mask individual (c, f, w) tokens directly

    token_mask:
      Optional pre-computed mask of shape (B, C, F, W), dtype bool. If
      provided, bypasses the score-quantile selection (`score_quantile` and
      `granularity` are ignored). Used by cf_eval.py to compare top-k vs
      random vs bottom-k selections on the same prior with the same |M|.
      Note: a quantile/top-k mask masks the top `(1-q)` fraction, which is
      inherently bounded — the original's "leave ≥10% context" cap is a no-op
      here for `q ≥ 0.1`.

    mode:
      * "iterative" (default, paper-faithful): MaskGIT iterative parallel
        decoding with the unmasked tokens FROZEN as context (matches the
        original TimeVQVAE-AD `explainable_sampling`). Adjacent masked tokens
        are filled jointly via confidence-based re-masking instead of an
        independent per-token argmax — the right way to sample a contiguous
        masked block (i.e. an anomaly) from the prior.
      * "oneshot": single forward pass + per-position argmax. Kept for A/B
        comparison; degrades on multi-token anomalies.

    n_samples:
      Number of stochastic counterfactuals per window (ensemble). The masked
      window is replicated `n_samples` times; `x_cf` is `(B*n_samples, C, T)`.
      Default 1. Ignored in the deterministic `greedy` decode.

    greedy:
      Deterministic iterative decode (argmax + noise-free confidence). No RNG,
      so repeated calls on the same mask are identical — used by cf_eval for
      reproducible mask-to-mask comparisons.

    adaptive_steps:
      If True (default), the number of iterative-decode steps adapts to the
      masked fraction via t_star = ⌊2·arccos(masking_ratio)/π · T⌋ (faithful to
      the original TimeVQVAE-AD `explainable_sampling`: small masked fraction →
      high t_star → few steps). If False, runs all `prior.T` steps. Only affects
      mode="iterative".

    Returns dict with:
      x_cf       : (B*n_samples, C, T) counterfactual waveform
      token_mask : (B, C, F, W)       bool — which tokens were replaced
      scores     : (B, C, F, W)       per-token NLL from the prior
      new_indices(B)n_samples, C, F*W) post-resample discrete tokens
      mode       : str                the decode mode actually used
    """
    assert system.stage1.quantizer.name == "shared_codebook_per_channel_vq", (
        "counterfactual requires shared_codebook_per_channel_vq"
    )
    assert isinstance(system.prior, MaskGITPrior3DPos), (
        "counterfactual requires maskgit_3d_pos"
    )
    assert granularity in {"column", "token"}, (
        f"granularity must be 'column' or 'token', got {granularity!r}"
    )
    assert not system.stage1.training and not system.prior.training, (
        "counterfactual requires stage1 and prior in eval mode (stable codebook, "
        "no dropout). Call system.eval() before invoking."
    )

    _, indices, latent_spatial = system.stage1.encode_tokens(inputs)
    assert indices.ndim == 3, f"Expected (B, C, F*W), got {indices.shape}"
    F_, W = int(latent_spatial[0]), int(latent_spatial[1])
    B, C = int(indices.shape[0]), int(indices.shape[1])
    orig_tokens = indices.long().reshape(B, C, F_, W)                    # (B, C, F, W)

    # Per-token NLL against the prior. NOT aggregated.
    scores = system.prior.score_tokens(orig_tokens.reshape(B, -1))       # (B, C, F, W)

    # Pick which tokens to mask.
    if token_mask is not None:
        assert token_mask.shape == (B, C, F_, W), (
            f"token_mask shape {tuple(token_mask.shape)} != expected {(B, C, F_, W)}"
        )
        assert token_mask.dtype == torch.bool, (
            f"token_mask dtype must be bool, got {token_mask.dtype}"
        )
        token_mask = token_mask.to(orig_tokens.device)
    elif granularity == "column":
        col = scores.mean(dim=2)                                         # (B, C, W)
        thr = torch.quantile(col.reshape(B, -1), score_quantile, dim=1).view(B, 1, 1)
        col_mask = col > thr                                             # (B, C, W)
        token_mask = col_mask.unsqueeze(2).expand(-1, -1, F_, -1)        # (B, C, F, W)
    else:  # "token"
        thr = torch.quantile(scores.reshape(B, -1), score_quantile, dim=1).view(B, 1, 1, 1)
        token_mask = scores > thr                                        # (B, C, F, W)

    # Mask the chosen positions; the prior predicts the masked tokens while the
    # unmasked tokens stay as context. Originals are preserved everywhere else.
    masked = orig_tokens.clone()
    masked[token_mask] = system.prior.mask_token_id                      # (B, C, F, W)
    seq_len = C * F_ * W

    if mode == "oneshot":
        # Single forward + per-position argmax (all masked tokens predicted
        # independently). Kept for A/B; degrades on contiguous masked blocks.
        logits = system.prior._logits(masked.reshape(B, seq_len))        # (B, C*F*W, K)
        sampled_all = logits.argmax(dim=-1).reshape(B, C, F_, W)
        new_tokens = orig_tokens.clone()
        new_tokens[token_mask] = sampled_all[token_mask]
        new_indices = new_tokens.reshape(B, C, F_ * W)
        decode_inputs = inputs
    elif mode == "iterative":
        # Paper-faithful: MaskGIT iterative decode with the unmasked tokens
        # frozen as context. _iterative_decode preserves frozen positions and
        # commits the masked region jointly across `prior.steps`.
        system.prior._build_positions_3d(seq_len, orig_tokens.device)
        init = masked.reshape(B, seq_len)
        frozen = (~token_mask).reshape(B, seq_len)
        if n_samples > 1:
            init = init.repeat_interleave(n_samples, dim=0)
            frozen = frozen.repeat_interleave(n_samples, dim=0)
        n_eff = init.shape[0]
        # Adaptive decode steps (t_star), faithful to the original
        # `explainable_sampling`: fewer tokens masked → higher init_step → fewer
        # iterations. `masking_ratio` is the fraction of the WHOLE sequence
        # masked (matches the original `is_anom` fraction; with column-granularity
        # masks the token fraction equals the column fraction). Clamped to leave
        # at least one decode iteration.
        if adaptive_steps:
            masking_ratio = float(token_mask.float().mean().clamp(0.0, 1.0).item())
            t_star = (
                int(math.floor(2.0 * math.acos(masking_ratio) / math.pi * system.prior.steps))
                if masking_ratio > 0.0 else 0
            )
            init_step = max(0, min(system.prior.steps - 1, t_star))
        else:
            init_step = 0
        decoded = _iterative_decode(
            logits_fn=system.prior._logits,
            num_samples=n_eff, seq_len=seq_len,
            mask_token_id=system.prior.mask_token_id,
            steps=system.prior.steps,
            mask_scheduling_fn=system.prior.mask_scheduling_fn,
            choice_temperature=system.prior.choice_temperature,
            device=orig_tokens.device,
            initial_tokens=init, frozen_mask=frozen, greedy=greedy,
            init_step=init_step,
        )                                                                # (n_eff, seq_len)
        new_indices = decoded.reshape(n_eff, C, F_ * W)
        decode_inputs = inputs if n_samples == 1 else inputs.repeat_interleave(n_samples, dim=0)
    else:
        raise ValueError(f"counterfactual mode must be 'iterative' or 'oneshot', got {mode!r}")

    # Embed the new indices and decode back to waveform. The refinement head is
    # applied to match the real-reconstruction path, so x_cf is comparable to
    # stage1's normal output (same head used everywhere).
    quantized_cf = system.stage1.quantizer.embed_indices(new_indices, (F_, W))
    tf = system.stage1.transform(decode_inputs)
    repr_cf = system.stage1.reconstruct_representation(quantized_cf, tf.tensor.shape)
    x_cf = system.stage1.transform.inverse(repr_cf, tf.spec)
    if system.stage1.use_refinement:
        x_cf = system.stage1.refinement(x_cf)

    return {
        "x_cf": x_cf,                                                    # (B*n_samples, C, T)
        "token_mask": token_mask,
        "scores": scores,
        "new_indices": new_indices,
        "mode": mode,
    }


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def save_stage2_checkpoint(path: Path, model: Stage2System, cfg: Config, step: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),            # stage1 + prior, all in one
        "cfg_dict": asdict(cfg),
        "step": step,
        "epoch": epoch,
    }, path)


def load_stage2(
    ckpt_path: str | Path, cfg: Config,
    stage1_ckpt: str | Path,
    stage1_example_inputs: torch.Tensor,
    device: str | torch.device = "cpu",
) -> Stage2System:
    """Rebuild stage1 + Stage2System and load weights. Returns frozen in eval mode."""
    stage1 = load_stage1(stage1_ckpt, cfg, stage1_example_inputs, device="cpu")
    model = Stage2System(cfg, stage1)
    model.materialize(stage1_example_inputs.cpu())
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─── Main — plain-PyTorch training loop, everything inline ──────────────────

def main(cfg: Config | None = None, stage1_ckpt: str | Path | None = None) -> Path:
    """Train stage 2 end-to-end. Returns path to the best checkpoint."""
    configure_logging()
    cfg = cfg or load_config()
    print(format_config(cfg))
    seed_everything(cfg.seed)
    # Resume / determinism flags (design: resume couples Tier S)
    strict_bitexact = os.environ.get("STRICT_BITEXACT") == "1"
    resume = bool(getattr(cfg.training, "resume", False)) or os.environ.get("RESUME") == "1"
    deterministic = (bool(getattr(cfg.training, "deterministic", False))
                     or os.environ.get("DETERMINISTIC") == "1"
                     or strict_bitexact or resume)
    if os.environ.get("ALLOW_NONDETERMINISTIC_RESUME") == "1":
        deterministic = (bool(getattr(cfg.training, "deterministic", False))
                         or os.environ.get("DETERMINISTIC") == "1" or strict_bitexact)
    apply_determinism(deterministic, strict_bitexact)   # sets matmul/cudnn per tier

    device = resolve_device()
    run_dir = run_dir_for(cfg, "stage2")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage2] run dir: {run_dir.resolve()}")
    print(f"[stage2] device: {device}")

    # ── Stage 1 checkpoint ─────────────────────────────────────────────────
    if stage1_ckpt is None:
        stage1_ckpt = best_checkpoint(cfg, "stage1")
    stage1_ckpt = Path(stage1_ckpt)
    if not stage1_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {stage1_ckpt}. "
            "Run stage1.py first, or pass stage1_ckpt explicitly."
        )
    print(f"[stage2] stage1 checkpoint: {stage1_ckpt.resolve()}")
    # ── Data ──────────────────────────────────────────────────────────────
    # Always build raw dataloaders for example inputs (stage1 materialisation)
    # and as a fallback source of batches if the token cache is missing.
    raw_data = make_dataloaders(cfg, stage="stage2")
    example_inputs = next(iter(raw_data.train_loader))["inputs"][:1].cpu()

    # Prefer the token cache saved by stage 1 for much faster training.
    # IMPORTANT: derive the cache path from `stage1_ckpt`, NOT from `cfg`.
    # The user may pass a checkpoint that lives outside the cfg-derived run dir
    # (e.g. a hand-picked frozen stage 1); in that case `token_cache_path(cfg)`
    # would point to a stale or unrelated cache.
    # `stage1_ckpt` lives at `<run>/checkpoints/best.ckpt`, so the cache sits at
    # `stage1_ckpt.parent.parent / "token_cache.pt"`.
    cache = stage1_ckpt.parent.parent / "token_cache.pt"
    if cache.exists():
        token_dm = PrecomputedTokenDataModule(cache, cfg)
        token_dm.setup("fit")
        print(f"[stage2] using token cache: {cache.resolve()}")
        train_loader = token_dm.train_dataloader()
        val_loader = token_dm.val_dataloader()
        train_tokens_source = token_dm.train_dataset.tokens
        use_cache = True
    else:
        print("[stage2] WARNING: no token cache — encoding on-the-fly (slower).")
        train_loader = raw_data.train_loader
        val_loader = raw_data.val_loader
        train_tokens_source = None
        use_cache = False

    batches_per_epoch = len(train_loader)

    # ── Model (stage1 loaded + frozen, prior materialised) ────────────────
    stage1_model = load_stage1(stage1_ckpt, cfg, example_inputs, device="cpu")
    model = Stage2System(cfg, stage1_model)
    model.materialize(example_inputs)
    model.to(device)
    model.stage1.eval()

    # ── Architecture summary (only the prior is trained; stage1 is frozen) ─
    _sep = "─" * 80
    print(f"[stage2] {_sep}")
    _n_s1_enc = sum(p.numel() for p in model.stage1.encoder.parameters())
    _n_s1_vq  = sum(p.numel() for p in model.stage1.quantizer.parameters())
    _n_s1_dec = sum(p.numel() for p in model.stage1.decoder_2d.parameters())
    _n_s1     = _n_s1_enc + _n_s1_vq + _n_s1_dec
    print(f"[stage2] frozen stage1: encoder={_n_s1_enc:,} | "
          f"quantizer={_n_s1_vq:,} | decoder={_n_s1_dec:,} | "
          f"total={_n_s1:,}")
    print(f"[stage2] {_sep}")
    _n_prior = sum(p.numel() for p in model.prior.parameters())
    print(f"[stage2] prior ({type(model.prior).__name__}, trainable): {_n_prior:,} params")
    print(model.prior)
    print(f"[stage2] {_sep}")
    n_params = _n_prior

    # ── Token weights (inverse-sqrt-frequency upweighting, optional) ──────
    if use_cache:
        model.set_token_weights_from_tokens(train_tokens_source)
    elif getattr(model.prior, "loss_weighting", "uniform") != "uniform":
        print("[stage2] gathering tokens on-the-fly for inverse-sqrt-frequency weighting...")
        collected: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in raw_data.train_loader:
                batch["inputs"] = batch["inputs"].to(device)
                collected.append(model.tokens_from_batch(batch).cpu())
        if collected:
            model.set_token_weights_from_tokens(torch.cat(collected, dim=0))

    # Optimiser + warmup-cosine schedule
    opt = torch.optim.AdamW(
        model.prior.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        fused=(torch.device(device).type == "cuda"),
    )
    max_steps = cfg.training.stage2_max_steps
    min_epochs = getattr(cfg.training, "stage2_min_epochs", 0)
    if min_epochs > 0:
        floor = min_epochs * batches_per_epoch
        if floor > max_steps:
            print(f"[stage2] step budget raised {max_steps} -> {floor} "
                  f"({min_epochs} epochs x {batches_per_epoch} batches/epoch).")
            max_steps = floor
    warmup_steps = max(1, int(max_steps * cfg.training.warmup_rate))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── Mixed precision ────────────────────────────────────────────────────
    # fp16 below Ampere (this node's RTX 8000 is sm_75: fp16 tensor cores, NO bf16),
    # bf16 from Ampere on. `AMP=0` → GradScaler(enabled=False), a pure pass-through,
    # and autocast(enabled=False) → bit-identical fp32. Scoring/detect stay eager fp32.
    use_amp = bool(getattr(cfg.training, "amp", False)) and torch.cuda.is_available()
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    if use_amp:
        print(f"[stage2] AMP {str(amp_dtype).replace('torch.', '')} autocast ON "
              f"(AMP=0 to disable)")

    # Re-seed: model.materialize() runs prior.forward(...) which calls
    # `_mask_tokens` with `random.random()`, advancing Python's RNG. Re-seeding
    # here keeps the actual training loop deterministic regardless of what the
    # throw-away materialise pass consumed.
    seed_everything(cfg.seed)

    # CSV log setup
    # Same approach as stage1: buffer rows + rewrite on every epoch so the
    # header always matches the data, even when val/loss first appears late.
    csv_path = run_dir / "logs" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, float | int]] = []

    def _write_metrics_csv() -> None:
        if not csv_rows:
            return
        all_fields: list[str] = []
        seen: set[str] = set()
        for r in csv_rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    all_fields.append(k)
        import io as _io
        from utils import atomic_write_text
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
        atomic_write_text(csv_path, buf.getvalue())   # crash-safe: temp+fsync+replace

    # Early stopping bookkeeping (step-based patience)
    best_val = float("inf")
    best_step = 0
    best_ckpt_path = ckpt_dir / "best.ckpt"
    last_ckpt_path = ckpt_dir / "last.ckpt"
    patience_steps = cfg.training.stage2_patience_steps

    # ── Resume (design: full-state restore at epoch boundary) ──────────────
    fingerprint = build_fingerprint(
        cfg, batches_per_epoch=batches_per_epoch, max_steps=max_steps,
        warmup_steps=warmup_steps, batch_size=cfg.dataset.batch_size_stage2,
        min_epochs=min_epochs, patience_steps=patience_steps,
        device=device, deterministic=deterministic, extra={"use_cache": bool(use_cache)},
    )
    start_step, start_epoch = 0, 0
    if resume:
        ck = load_resumable(ckpt_dir)
        if ck is None:
            print("[resume] stage2: no valid checkpoint; starting fresh.")
        else:
            _assert_fingerprint(ck["loader_fingerprint"], fingerprint,
                                deterministic=deterministic)
            model.load_state_dict(ck["state_dict"])          # prior + frozen stage1
            opt.load_state_dict(ck["optimizer"])             # AdamW moments (prior only)
            sched.load_state_dict(ck["scheduler"])           # cosine position
            best_val = float(ck["best_val"]); best_step = int(ck["best_step"])
            start_step, start_epoch = int(ck["step"]), int(ck["epoch"])
            csv_rows[:] = list(ck["csv_rows"])               # verbatim, no reparse
            restore_rng(ck["rng"], device)                   # LAST: overrides pre-loop seed
            print(f"[resume] stage2 resumed at epoch={start_epoch} step={start_step} "
                  f"best_val={best_val:.6f} best_step={best_step}")

    # ── Training loop ─────────────────────────────────────────────────────
    # Stops when ANY of these fires: epoch >= stage2_max_epochs (primary control),
    # step >= stage2_max_steps (safety cap), or early-stopping patience exhausted.
    step = start_step
    epoch = start_epoch
    stop = False
    # Opt-in profiler (TVQ_PROFILE=1). No-op + zero overhead when disabled.
    # Sampled mode: records a small window of training steps (see lib/profiling.py).
    prof = profile_run("stage2", sampled=True)
    prof.start()
    while not stop and epoch < cfg.training.stage2_max_epochs:
        # ─ Train epoch ─
        model.prior.train()
        train_losses: list[float] = []

        for batch in train_loader:
            if "tokens" in batch:
                batch = {"tokens": batch["tokens"].to(device, non_blocking=True)}
            else:
                batch["inputs"] = batch["inputs"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                tokens = model.tokens_from_batch(batch)
                model._inform_latent_shape()
                out = model.prior(tokens)
                loss = out.loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            _prev_scale = scaler.get_scale()
            scaler.step(opt)                    # skipped iff grads overflowed under fp16
            scaler.update()
            # A skipped step means no weights moved: the LR schedule must not advance
            # either, or it silently desyncs from the optimiser under fp16.
            if scaler.get_scale() >= _prev_scale:
                sched.step()
            prof.step()                         # advance profiler one step (no-op when off)

            train_losses.append(loss.item())
            step += 1
            if step >= max_steps:
                stop = True
                break

        # ─ Val epoch ─
        val_losses: list[float] = []
        val_tok_counts = None   # unigram token histogram → H_unigram grammar baseline
        ran_val = False
        if val_loader is not None and (epoch % cfg.training.check_val_every_n_epoch == 0):
            model.prior.eval()
            ran_val = True
            # Validation stays fp32 on purpose: the val loss drives early stopping and
            # checkpoint selection, so it must not depend on the training precision.
            with torch.no_grad():
                for batch in val_loader:
                    if "tokens" in batch:
                        batch = {"tokens": batch["tokens"].to(device, non_blocking=True)}
                    else:
                        batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
                    tokens = model.tokens_from_batch(batch)
                    try:   # accumulate unigram token histogram (grammar baseline; guarded)
                        _bc = torch.bincount(tokens.reshape(-1).to(torch.long),
                                             minlength=int(cfg.quantizer.codebook_size) + 1)
                        val_tok_counts = _bc if val_tok_counts is None else val_tok_counts + _bc
                    except Exception:
                        pass
                    model._inform_latent_shape()
                    out = model.prior(tokens)
                    val_losses.append(out.loss.item())

        # ─ Build CSV row ─
        row: dict[str, float | int] = {"epoch": epoch, "step": step,
                                       "lr": float(opt.param_groups[0]["lr"])}
        row["train/loss"] = float(np.mean(train_losses)) if train_losses else float("nan")
        if val_losses:
            row["val/loss"] = float(np.mean(val_losses))

        # ─ Grammar diagnostics at epoch end (guarded — never breaks training) ─
        #   Is the prior learning the token grammar? Compare val masked-token CE to:
        #     (1) ln(K)     = uniform-over-codebook baseline ("no grammar")
        #     (2) H_unigram = entropy of the token frequencies ("context-free" baseline)
        #   val CE < ln(K) → beats uniform; val CE < H_unigram → learned CONTEXT, not just
        #   which tokens are common. Plus train–val gap (memorization) and a plateau flag.
        try:
            import math as _math
            K = int(cfg.quantizer.codebook_size); ln_K = _math.log(K)
            vce = row.get("val/loss", float("nan")); tce = row.get("train/loss", float("nan"))
            row["grammar/ln_K"] = ln_K
            if vce == vce:
                row["grammar/margin_vs_uniform"] = ln_K - vce
                row["grammar/train_val_gap"] = tce - vce
                Huni = float("nan")
                if val_tok_counts is not None:
                    _p = val_tok_counts.float(); _p = _p / _p.sum().clamp_min(1.0)
                    _nz = _p[_p > 0]; Huni = float(-(_nz * _nz.log()).sum())
                    row["grammar/H_unigram"] = Huni
                    row["grammar/margin_vs_unigram"] = Huni - vce
                _vh = [r["val/loss"] for r in csv_rows[-5:] if "val/loss" in r] + [vce]
                _plateau = len(_vh) >= 3 and (max(_vh) - min(_vh)) < 0.01
                _v1 = "OK learning-grammar" if (ln_K - vce) > 0.05 else "~uniform"
                _v2 = (f"OK learned-CONTEXT({Huni - vce:+.3f} vs unigram)"
                       if (Huni == Huni and (Huni - vce) > 0.02)
                       else ("~token-freq-only" if Huni == Huni else ""))
                _v3 = "MEMORIZING" if (tce - vce) < -0.10 else "generalizes(train~val)"
                _v4 = "PLATEAU->converging" if _plateau else "still-improving"
                print(f"[grammar] epoch={epoch:4d} valCE={vce:.3f}  "
                      f"vs ln{K}={ln_K:.3f}({ln_K-vce:+.3f}) {_v1}  "
                      + (f"vs Huni={Huni:.3f}({Huni-vce:+.3f}) {_v2}  " if Huni == Huni else "")
                      + f"gap={tce-vce:+.3f} {_v3}  {_v4}", flush=True)
        except Exception:
            pass

        csv_rows.append(row)
        _write_metrics_csv()

        # ─ Console line ─
        tl = row.get("train/loss", float("nan"))
        vl = row.get("val/loss", float("nan"))
        print(f"[stage2] epoch={epoch:4d} step={step:6d} "
              f"train/loss={tl:.4f} val/loss={vl:.4f} lr={row['lr']:.2e}")

        # ─ Checkpoint (last always; best if val improved) ─
        save_stage2_checkpoint(last_ckpt_path, model, cfg, step, epoch)
        # ─ Per-epoch checkpoint HISTORY (opt-in via sentinel file at project root):
        #   copies the just-written last.ckpt to epochs/epoch_NNNN.ckpt so a live
        #   sidecar (or post-hoc) can watch the token-grammar evolve epoch by epoch.
        #   Additive + gated → default runs are byte-unchanged. ─
        #   Cadence: EVERY epoch for the first 20 (the grammar moves fastest there),
        #   then 1-in-10. Short series run thousands of tiny epochs, where a 3.7 MB
        #   copy per epoch dominates the 8 training steps and floods the disk.
        if ((epoch < 20 or epoch % 10 == 0)
                and (Path(__file__).resolve().parents[1] / ".save_epoch_ckpts").exists()):
            import shutil as _shutil
            _ep_dir = ckpt_dir / "epochs"; _ep_dir.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(last_ckpt_path, _ep_dir / f"epoch_{epoch:04d}.ckpt")
        if ran_val:
            current_val = row.get("val/loss", float("inf"))
            if current_val < best_val - cfg.training.early_stopping_min_delta:
                best_val = float(current_val)
                best_step = step
                save_stage2_checkpoint(best_ckpt_path, model, cfg, step, epoch)
            elif (cfg.training.early_stopping and step >= warmup_steps
                  and step - best_step >= patience_steps):
                # Gate early stopping on WARMUP, not on `min_epochs`. `min_epochs` raises
                # `max_steps` to span the LR schedule, so gating on it made early stopping
                # fire only at the very last epoch — i.e. it was effectively disabled.
                print(f"[stage2] early stopping at epoch={epoch} step={step} "
                      f"(no val improvement for {step - best_step} steps "
                      f"since best at step {best_step}; past warmup={warmup_steps})")
                stop = True

        # ─ Resumable checkpoint: written LAST, AFTER best_val/best_step update
        #   (design §4.3 BLOCKER fix). epoch+1 = next epoch to run; RNG snapshot
        #   here is the last training-RNG event of the epoch. ─
        if resume:                       # write resume checkpoints only when opted-in (OFF = unchanged)
            save_resumable(ckpt_dir, build_payload(
                stage="stage2", model=model, cfg=cfg, opt=opt, sched=sched,
                step=step, epoch=epoch + 1, best_val=best_val, best_step=best_step,
                rng=snapshot_rng(device), fingerprint=fingerprint, csv_rows=csv_rows,
            ))

        epoch += 1

    prof.finish()                                     # stop profiler + write trace/table
    _write_metrics_csv()                              # final flush

    # Peak-VRAM telemetry -> stdout + logs/vram.csv for batch-size tuning.
    log_gpu_peak("stage2", cfg.dataset.batch_size_stage2, cfg, reset=True)

    # Post-training: loss plots
    save_loss_plots(csv_path, run_dir / "losses", stage_label="Stage 2")

    best_ckpt = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    print(f"[stage2] best checkpoint: {best_ckpt.resolve()}")
    return best_ckpt

# %%
if __name__ == "__main__":
    set_process_title()
    print("RUNNING AS MAIN:", __file__)
    main()
