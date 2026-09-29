"""
=============================================================================
  MaskGIT prior — stage 2 token-level transformer.
=============================================================================

Two variants in one file:

  * `MaskGITPrior`           — 1D positional embeddings over the flattened token
                                sequence. Default for most configs.

  * `MaskGITPrior2DPos`      — Factorised 2D positional embeddings
                                PE(h, w) = E_freq[h] + E_time[w]
                                Optional column / mixed masking modes.

Both implement:
  * `forward(tokens) → PriorOutput`      — random-masking training loss.
  * `score_tokens(tokens, H) → (B, W)`   — position-by-position masked-prediction
                                           NLL, averaged over frequency rows,
                                           with multi-scale kernel sizes.
  * `sample(n, seq_len, device)`         — iterative parallel decoding with
                                           confidence-based re-masking.

Standard behaviours:
  * Random fraction-of-tokens mask with schedulable γ(r)
    (cosine, linear, square, cubic).
  * Optional focal loss + label smoothing + per-token loss weights
    (for rare-token upweighting).
  * Learned output bias per position.
  * Dropout is applied to non-mask token embeddings during training
    (encourages robustness to slight corruption of the context).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _paper_kernel_size(W: int, rate: float) -> int:
    """Replica of `compute_latent_window_size` from the paper repo
    (`evaluation/__init__.py`). The kernel size for the masked sliding window
    is always odd; if ⌊W·rate⌋ is even, the size is biased UP to the nearest
    odd integer.

      0 < W·rate < 1  → 1
      ⌊W·rate⌋ odd    → ⌊W·rate⌋
      ⌈W·rate⌉ odd    → ⌈W·rate⌉
      both even       → ⌊W·rate⌋ + 1            (only when W·rate is integer & even)
    """
    raw = W * rate
    if np.floor(raw) == 0:
        return 1
    floor_val = int(np.floor(raw))
    if floor_val % 2 == 1:
        return floor_val
    ceil_val = int(np.ceil(raw))
    if ceil_val % 2 == 1:
        return ceil_val
    return floor_val + 1
import torch.nn.functional as F


# ─── Output container ────────────────────────────────────────────────────────

@dataclass
class PriorOutput:
    loss: torch.Tensor | None = None
    token_logits: torch.Tensor | None = None
    token_scores: torch.Tensor | None = None          # populated by score_tokens
    samples: torch.Tensor | None = None
    stats: dict[str, Any] = field(default_factory=dict)


# ─── Masking schedule γ(r) — fraction of tokens to mask given r ∈ [0, 1] ────

def _gamma_schedule(mode: str):
    if mode == "linear": return lambda r: 1 - r
    if mode == "square": return lambda r: 1 - (r ** 2)
    if mode == "cubic":  return lambda r: 1 - (r ** 3)
    return lambda r: math.cos(r * math.pi / 2)         # "cosine"


def _masked_count(length: int, gamma_r: float) -> int:
    """Mirror upstream TimeVQVAE-AD `_randomly_mask_tokens`: `gamma_r` is the
    fraction KEPT (n_unmasks / n), not the fraction masked. Returns
    `n_masked = n - clip(floor(gamma_r * n), 0, n-1)` so at least one token is
    always masked and the all-masked extreme is reachable (matches MaskGIT
    iterative-decoding starting state)."""
    if length <= 1:
        return 0
    n_unmasks = max(0, min(length - 1, int(float(gamma_r) * length)))
    return length - n_unmasks


# ─── Shared MaskGIT iterative parallel decoding (matches TimeVQVAE-AD original) ─

def _iterative_decode(
    logits_fn,
    num_samples: int,
    seq_len: int,
    mask_token_id: int,
    steps: int,
    mask_scheduling_fn,
    choice_temperature: float,
    device: torch.device,
    *,
    initial_tokens: torch.Tensor | None = None,
    frozen_mask: torch.Tensor | None = None,
    greedy: bool = False,
    init_step: int = 0,
) -> torch.Tensor:
    """Iterative parallel decoding à la MaskGIT.

    Token sampling uses raw logits (temperature 1.0). The choice of which
    sampled tokens to keep vs. re-mask each step is made on
    `log(prob) + T·(1-r)·gumbel_noise`, where T = `choice_temperature` anneals
    exploration: high noise early (diverse picks), zero at the final step
    (greedy). This matches the original TimeVQVAE-AD `mask_by_random_topk`.

    Optional conditional / counterfactual decoding (used by
    `stage2.counterfactual`):

      initial_tokens : (num_samples, seq_len) starting sequence. Defaults to
                       all-`mask_token_id` (unconditional generation). For a
                       counterfactual, pass the real tokens with the anomalous
                       positions set to `mask_token_id`.
      frozen_mask    : (num_samples, seq_len) bool. True = context positions
                       that are NEVER sampled or re-masked (they feed the
                       transformer as evidence at every step but keep their
                       original token). The re-mask budget γ(r)·M is computed
                       over the MASKABLE region only — with no frozen positions
                       this equals `seq_len`, so unconditional sampling is
                       bit-for-bit unchanged.
      greedy         : argmax token selection + noise-free confidence ordering,
                       i.e. a fully deterministic decode (no RNG). Used for
                       reproducible A/B comparisons in cf_eval.
      init_step      : start index into the T-step cosine schedule (default 0 →
                       unchanged unconditional/conditional behaviour). Set to
                       t_star = ⌊2·arccos(masking_ratio)/π · T⌋ to match the
                       original TimeVQVAE-AD `explainable_sampling`: the loop runs
                       range(init_step, steps) with the schedule still indexed by
                       the ABSOLUTE step (ratio = (step+1)/steps), so a small
                       masked fraction → high init_step → few decode steps.
    """
    if initial_tokens is None:
        tokens = torch.full((num_samples, seq_len), mask_token_id, dtype=torch.long, device=device)
    else:
        tokens = initial_tokens.clone().to(device)
    frozen = (
        torch.zeros_like(tokens, dtype=torch.bool)
        if frozen_mask is None else frozen_mask.to(device)
    )
    # γ(r) budgets the re-mask count over the MASKABLE (non-frozen) region.
    maskable_count = int((~frozen).sum(dim=1).max().item())

    for step in range(init_step, steps):
        unknown = tokens == mask_token_id
        logits = logits_fn(tokens)
        if greedy:
            sampled = logits.argmax(dim=-1)
        else:
            sampled = torch.distributions.Categorical(logits=logits).sample()
        sampled = torch.where(unknown, sampled, tokens)
        # Context positions always keep their original token.
        sampled = torch.where(frozen, tokens, sampled)

        ratio = (step + 1) / steps
        remask_ratio = mask_scheduling_fn(ratio)
        remask_count = max(0, int(remask_ratio * maskable_count))
        if remask_count == 0:
            tokens = sampled
            continue

        probs = logits.softmax(dim=-1)
        sel_probs = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        confidence = torch.log(sel_probs.clamp(min=1e-5))
        if not greedy:
            u = torch.rand_like(sel_probs).clamp(min=1e-20)
            gumbel = -torch.log(-torch.log(u))
            confidence = confidence + choice_temperature * (1.0 - ratio) * gumbel
        # Only currently-unknown, non-frozen positions are eligible for re-masking.
        eligible = unknown & ~frozen
        confidence = torch.where(eligible, confidence, torch.full_like(confidence, float("inf")))

        low_conf_idx = confidence.topk(k=min(remask_count, seq_len), dim=-1, largest=False).indices
        tokens = sampled
        tokens.scatter_(1, low_conf_idx, mask_token_id)
        # Defensive: never leave a frozen position re-masked.
        if frozen_mask is not None:
            tokens = torch.where(frozen, initial_tokens.to(device), tokens)
    return tokens


# MaskGIT prior (1D positions)

class MaskGITPrior(nn.Module):
    """Classic MaskGIT with random-token masking and 1D positional embeddings."""

    name = "maskgit"

    def __init__(
        self,
        codebook_size: int = 256,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        choice_temperature: float = 4.0,
        T: int = 12,
        mask_scheduling: str = "cosine",
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        loss_weighting: str = "uniform",
        score_window_size_rates: tuple[float, ...] = (0.1, 0.3, 0.5),
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.choice_temperature = float(choice_temperature)
        self.steps = int(T)
        self.mask_scheduling_fn = _gamma_schedule(mask_scheduling)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.loss_weighting = str(loss_weighting)
        self.codebook_size = int(codebook_size)
        self.mask_token_id = self.codebook_size          # extra embedding at index K
        self.score_window_size_rates = tuple(score_window_size_rates)

        # Token embedding + mask token (K+1 rows).
        self.token_embedding = nn.Embedding(self.codebook_size + 1, self.embed_dim)
        self.register_buffer("_token_weights", torch.ones(self.codebook_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True, dropout=self.dropout,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        # Lazy position embeddings / output bias (sized on first forward).
        self.position_embedding: nn.Embedding | None = None
        self.output_bias: nn.Parameter | None = None

        self.pred_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

    # ── Per-token loss weights (for inverse-sqrt-frequency upweighting) ─────

    def set_token_weights(self, weights: torch.Tensor) -> None:
        self._token_weights = weights.to(
            device=self.token_embedding.weight.device,
            dtype=self.token_embedding.weight.dtype,
        )

    # ── Lazy buffers ────────────────────────────────────────────────────────

    def _build_positions(self, seq_len: int, device: torch.device) -> None:
        if self.position_embedding is None or self.position_embedding.num_embeddings < seq_len:
            self.position_embedding = nn.Embedding(seq_len, self.embed_dim).to(device)
        if self.output_bias is None or self.output_bias.shape[0] < seq_len:
            self.output_bias = nn.Parameter(torch.zeros(seq_len, self.codebook_size, device=device))

    # Random-token masking

    def _mask_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        ratios = torch.rand(B, device=tokens.device)         # deterministic across workers
        for row in range(B):
            ratio = ratios[row].item()
            num_masked = _masked_count(L, self.mask_scheduling_fn(ratio))
            perm = torch.randperm(L, device=tokens.device)
            mask[row, perm[:num_masked]] = True
        masked[mask] = self.mask_token_id
        return masked, mask

    # Transformer → logits per position

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        self._build_positions(L, tokens.device)
        pos = torch.arange(L, device=tokens.device)
        tok_emb = self.token_embedding(tokens)

        if self.training:
            # Drop only non-mask token embeddings (keep mask-token embedding intact).
            is_mask = (tokens == self.mask_token_id).unsqueeze(-1)
            tok_emb_drop = F.dropout(tok_emb, p=self.dropout, training=True)
            tok_emb = torch.where(is_mask, tok_emb, tok_emb_drop)

        x = tok_emb + self.position_embedding(pos)[None, :, :]
        x = self.transformer(x)
        x = self.pred_head(x)

        # Logits = weight-tied projection to token embeddings + per-position bias.
        logits = torch.matmul(x, self.token_embedding.weight.T)
        return logits[:, :, : self.codebook_size] + self.output_bias[:L]

    # ── Loss (CE + optional focal + per-token weights + label smoothing) ───

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        if self.focal_gamma > 0:
            ce = F.cross_entropy(logits, target, weight=None,
                                 label_smoothing=self.label_smoothing, reduction="none")
            with torch.no_grad():
                pt = torch.exp(-ce)
            focal = (1.0 - pt) ** self.focal_gamma
            loss = focal * ce
            if self._token_weights is not None and not torch.all(self._token_weights == 1.0):
                loss = loss * self._token_weights[target]
            return loss.mean()
        return F.cross_entropy(logits, target, weight=self._token_weights,
                               label_smoothing=self.label_smoothing)

    # Forward (training step)

    def forward(self, tokens: torch.Tensor) -> PriorOutput:
        masked_tokens, mask = self._mask_tokens(tokens)
        logits = self._logits(masked_tokens)
        target = tokens[mask]
        loss = self._compute_loss(logits[mask], target)
        return PriorOutput(
            loss=loss, token_logits=logits,
            stats={"mask_ratio": mask.float().mean()},
        )

    # ── Scoring: window-centred masked NLL, averaged over freq rows ─────────

    @torch.no_grad()
    def score_tokens(self, tokens: torch.Tensor, latent_height: int = 1) -> torch.Tensor:
        """Returns per-time-column scores of shape (B, W)."""
        B, N = tokens.shape
        H = latent_height
        if H <= 0 or (N % H) != 0:
            raise ValueError(f"Token sequence length {N} is not divisible by latent_height={H}.")
        W = N // H
        device = tokens.device
        scores = torch.zeros(B, H, W, device=device)

        for rate in self.score_window_size_rates:
            ks = _paper_kernel_size(W, rate)
            half = ks // 2
            for w in range(W):
                lo, hi = max(0, w - half), min(W, w + half + 1)
                masked = tokens.clone().view(B, H, W)
                masked[:, :, lo:hi] = self.mask_token_id
                logits = self._logits(masked.reshape(B, N))
                log_probs = logits.log_softmax(dim=-1)
                # Paper Algorithm 1: a_w averages -log p over the masked range.
                lp_grid = log_probs.reshape(B, H, W, -1)
                target  = tokens.view(B, H, W)
                gathered = lp_grid.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, H, W)
                scores[:, :, w] += -gathered[:, :, lo:hi].mean(dim=-1)

        # Paper Algorithm 1: sum (not mean) across τ rates.
        return scores.mean(dim=1)               # (B, W)

    # ── Sampling: iterative parallel decoding with low-confidence re-masking ─

    @torch.no_grad()
    def sample(self, num_samples: int, seq_len: int, device: torch.device) -> torch.Tensor:
        self._build_positions(seq_len, device)
        return _iterative_decode(
            logits_fn=self._logits,
            num_samples=num_samples, seq_len=seq_len,
            mask_token_id=self.mask_token_id, steps=self.steps,
            mask_scheduling_fn=self.mask_scheduling_fn,
            choice_temperature=self.choice_temperature, device=device,
        )


# ─── MaskGIT prior with factorised 2D positional embeddings ─────────────────

class MaskGITPrior2DPos(nn.Module):
    """Same as MaskGITPrior but PE(h, w) = E_freq[h] + E_time[w].

    Supports three masking modes (cfg.prior.mask_mode):
      * "random" — classic MaskGIT (random positions)
      * "column" — mask entire time columns (aligns training with scoring)
      * "mixed"  — 50/50 per batch between the two above
    """

    name = "maskgit_2d_pos"

    def __init__(
        self,
        codebook_size: int = 256,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        choice_temperature: float = 4.0,
        T: int = 12,
        mask_scheduling: str = "cosine",
        label_smoothing: float = 0.0,
        mask_mode: str = "random",
        score_window_size_rates: tuple[float, ...] = (0.1, 0.3, 0.5),
        **_,                                             # ignore unused params
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.choice_temperature = float(choice_temperature)
        self.steps = int(T)
        self.mask_scheduling_fn = _gamma_schedule(mask_scheduling)
        self.label_smoothing = float(label_smoothing)
        self.codebook_size = int(codebook_size)
        self.mask_token_id = self.codebook_size
        self.mask_mode = str(mask_mode)
        self.score_window_size_rates = tuple(score_window_size_rates)
        self.loss_weighting = "uniform"                  # hardcoded: no weighted-loss path here; stage2 reads this attr

        self.token_embedding = nn.Embedding(self.codebook_size + 1, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True, dropout=self.dropout,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

        self.pred_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        # 2D positional embeddings — built lazily once H and W are known.
        self.latent_height = 0
        self.latent_width = 0
        self.freq_embedding: nn.Embedding | None = None
        self.time_embedding: nn.Embedding | None = None
        self.output_bias: nn.Parameter | None = None

    def set_latent_shape(self, height: int, width: int) -> None:
        """Set by the pipeline once we've seen one encoded sample."""
        self.latent_height = height
        self.latent_width = width

    # Lazy positional embedding build

    def _build_positions_2d(self, seq_len: int, device: torch.device) -> None:
        H = self.latent_height if self.latent_height > 0 else 1
        W = seq_len // H
        if H * W != seq_len:
            raise ValueError(
                f"seq_len={seq_len} not divisible by latent_height={H}. "
                "Call set_latent_shape(H, W) first."
            )
        rebuild = (
            self.freq_embedding is None
            or self.time_embedding is None
            or self.freq_embedding.num_embeddings < H
            or self.time_embedding.num_embeddings < W
        )
        if rebuild:
            self.latent_height, self.latent_width = H, W
            self.freq_embedding = nn.Embedding(H, self.embed_dim).to(device)
            self.time_embedding = nn.Embedding(W, self.embed_dim).to(device)
        if self.output_bias is None or self.output_bias.shape[0] < seq_len:
            self.output_bias = nn.Parameter(torch.zeros(seq_len, self.codebook_size, device=device))

    def _pos_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        H, W = self.latent_height, self.latent_width
        h_coords = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1)
        w_coords = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1)
        return self.freq_embedding(h_coords) + self.time_embedding(w_coords)

    # ── Masking strategies ──────────────────────────────────────────────────

    def _mask_tokens_random(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for row in range(B):
            ratio = random.random()
            num_masked = _masked_count(L, self.mask_scheduling_fn(ratio))
            perm = torch.randperm(L, device=tokens.device)
            mask[row, perm[:num_masked]] = True
        masked[mask] = self.mask_token_id
        return masked, mask

    def _mask_tokens_column(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        H = self.latent_height if self.latent_height > 0 else 1
        if H <= 0 or (L % H) != 0:
            raise ValueError(f"Token sequence length {L} is not divisible by latent_height={H}.")
        W = L // H
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask_2d = mask.view(B, H, W)
        for row in range(B):
            ratio = random.random()
            num_masked_cols = _masked_count(W, self.mask_scheduling_fn(ratio))
            perm = torch.randperm(W, device=tokens.device)
            mask_2d[row, :, perm[:num_masked_cols]] = True
        mask = mask_2d.reshape(B, L)
        masked[mask] = self.mask_token_id
        return masked, mask

    def _mask_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_mode == "column":
            return self._mask_tokens_column(tokens)
        if self.mask_mode == "mixed":
            return self._mask_tokens_column(tokens) if random.random() < 0.5 else self._mask_tokens_random(tokens)
        return self._mask_tokens_random(tokens)

    # Shared transformer forward

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        self._build_positions_2d(L, tokens.device)
        tok_emb = self.token_embedding(tokens)

        if self.training:
            is_mask = (tokens == self.mask_token_id).unsqueeze(-1)
            tok_emb_drop = F.dropout(tok_emb, p=self.dropout, training=True)
            tok_emb = torch.where(is_mask, tok_emb, tok_emb_drop)

        pos_emb = self._pos_embedding(L, tokens.device)
        x = tok_emb + pos_emb[None, :, :]
        x = self.transformer(x)
        x = self.pred_head(x)

        logits = torch.matmul(x, self.token_embedding.weight.T)
        return logits[:, :, : self.codebook_size] + self.output_bias[:L]

    def forward(self, tokens: torch.Tensor) -> PriorOutput:
        masked_tokens, mask = self._mask_tokens(tokens)
        logits = self._logits(masked_tokens)
        target = tokens[mask]
        masked_logits = logits[mask]
        loss = (
            F.cross_entropy(masked_logits, target, label_smoothing=self.label_smoothing)
            if masked_logits.numel() > 0
            else logits.sum() * 0.0
        )
        return PriorOutput(
            loss=loss, token_logits=logits,
            stats={"mask_ratio": mask.float().mean()},
        )

    # Scoring + sampling are identical to MaskGITPrior aside from the 2D-pos
    # builder — keep them here for self-containment.

    @torch.no_grad()
    def score_tokens(self, tokens: torch.Tensor, latent_height: int = 1) -> torch.Tensor:
        B, N = tokens.shape
        H = latent_height
        if H <= 0 or (N % H) != 0:
            raise ValueError(f"Token sequence length {N} is not divisible by latent_height={H}.")
        W = N // H
        self.set_latent_shape(H, W)
        device = tokens.device
        scores = torch.zeros(B, H, W, device=device)

        for rate in self.score_window_size_rates:
            ks = _paper_kernel_size(W, rate)
            half = ks // 2
            for w in range(W):
                lo, hi = max(0, w - half), min(W, w + half + 1)
                masked = tokens.clone().view(B, H, W)
                masked[:, :, lo:hi] = self.mask_token_id
                logits = self._logits(masked.reshape(B, N))
                log_probs = logits.log_softmax(dim=-1)
                # Paper Algorithm 1: a_w averages -log p over the masked range.
                lp_grid = log_probs.reshape(B, H, W, -1)
                target  = tokens.view(B, H, W)
                gathered = lp_grid.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scores[:, :, w] += -gathered[:, :, lo:hi].mean(dim=-1)

        # Paper Algorithm 1: sum (not mean) across τ rates.
        return scores.mean(dim=1)

    @torch.no_grad()
    def sample(self, num_samples: int, seq_len: int, device: torch.device) -> torch.Tensor:
        self._build_positions_2d(seq_len, device)
        return _iterative_decode(
            logits_fn=self._logits,
            num_samples=num_samples, seq_len=seq_len,
            mask_token_id=self.mask_token_id, steps=self.steps,
            mask_scheduling_fn=self.mask_scheduling_fn,
            choice_temperature=self.choice_temperature, device=device,
        )


# ─── MaskGIT prior with factorised 3D positional embeddings (target path) ───

class MaskGITPrior3DPos(nn.Module):
    """MaskGIT with E_channel[c] + E_freq[f] + E_time[t] positional embeddings.

    Tokens are laid out in channel-major, freq-middle, time-inner order:
        position c*F*W + f*W + w  ↔  (channel=c, freq=f, time=w)

    `score_tokens` returns per-token NLL of shape (B, C, F, W). All aggregations
    (mean over F to get (B, C, W); mean over C,F to get (B, W); etc.) are done
    by callers — the prior does NOT collapse any axis itself.
    """

    name = "maskgit_3d_pos"

    def __init__(
        self,
        codebook_size: int = 256,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        choice_temperature: float = 4.0,
        T: int = 12,
        mask_scheduling: str = "cosine",
        label_smoothing: float = 0.0,
        mask_mode: str = "random",
        score_window_size_rates: tuple[float, ...] = (0.1, 0.3, 0.5),
        **_,                                             # ignore unused params
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.choice_temperature = float(choice_temperature)
        self.steps = int(T)
        self.mask_scheduling_fn = _gamma_schedule(mask_scheduling)
        self.label_smoothing = float(label_smoothing)
        self.codebook_size = int(codebook_size)
        self.mask_token_id = self.codebook_size
        self.mask_mode = str(mask_mode)
        self.score_window_size_rates = tuple(score_window_size_rates)
        self.loss_weighting = "uniform"                  # hardcoded: no weighted-loss path here; stage2 reads this attr

        self.token_embedding = nn.Embedding(self.codebook_size + 1, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True, dropout=self.dropout,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

        self.pred_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        # Three separate positional embeddings — built lazily once C, F, W known.
        self.latent_channels = 0
        self.latent_freq = 0
        self.latent_time = 0
        self.channel_embedding: nn.Embedding | None = None
        self.freq_embedding: nn.Embedding | None = None
        self.time_embedding: nn.Embedding | None = None
        self.output_bias: nn.Parameter | None = None

    def set_latent_shape(self, channels: int, freq: int, time: int) -> None:
        self.latent_channels = int(channels)
        self.latent_freq = int(freq)
        self.latent_time = int(time)

    # Lazy positional embedding build

    def _build_positions_3d(self, seq_len: int, device: torch.device) -> None:
        C, F_, W = self.latent_channels, self.latent_freq, self.latent_time
        assert C * F_ * W == seq_len, f"seq_len={seq_len} != C*F*W={C*F_*W}"
        if self.channel_embedding is None or self.channel_embedding.num_embeddings < C:
            self.channel_embedding = nn.Embedding(C, self.embed_dim).to(device)
        if self.freq_embedding is None or self.freq_embedding.num_embeddings < F_:
            self.freq_embedding = nn.Embedding(F_, self.embed_dim).to(device)
        if self.time_embedding is None or self.time_embedding.num_embeddings < W:
            self.time_embedding = nn.Embedding(W, self.embed_dim).to(device)
        if self.output_bias is None or self.output_bias.shape[0] < seq_len:
            self.output_bias = nn.Parameter(torch.zeros(seq_len, self.codebook_size, device=device))

    def _pos_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sum of three factorised positional embeddings.

        Returns a tensor of shape (C*F*W, embed_dim) — i.e. one position vector
        per token in the flat sequence. The batch dim is added later by `_logits`
        via broadcast: `pos_emb[None, :, :]` → (1, C*F*W, embed_dim).
        """
        C, F_, W = self.latent_channels, self.latent_freq, self.latent_time
        c = torch.arange(C, device=device).view(C, 1, 1).expand(C, F_, W).reshape(-1)
        f = torch.arange(F_, device=device).view(1, F_, 1).expand(C, F_, W).reshape(-1)
        t = torch.arange(W, device=device).view(1, 1, W).expand(C, F_, W).reshape(-1)
        return self.channel_embedding(c) + self.freq_embedding(f) + self.time_embedding(t)

    # ── Masking strategies ──────────────────────────────────────────────────

    def _mask_tokens_random(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for row in range(B):
            ratio = random.random()
            num_masked = _masked_count(L, self.mask_scheduling_fn(ratio))
            perm = torch.randperm(L, device=tokens.device)
            mask[row, perm[:num_masked]] = True
        masked[mask] = self.mask_token_id
        return masked, mask

    def _mask_tokens_column(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask whole time columns across all channels and frequencies."""
        B, L = tokens.shape
        C, F_, W = self.latent_channels, self.latent_freq, self.latent_time
        assert C * F_ * W == L, f"seq_len={L} != C*F*W={C*F_*W}"
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.bool).view(B, C, F_, W)
        for row in range(B):
            ratio = random.random()
            num_masked_cols = _masked_count(W, self.mask_scheduling_fn(ratio))
            perm = torch.randperm(W, device=tokens.device)
            mask[row, :, :, perm[:num_masked_cols]] = True
        mask = mask.reshape(B, L)
        masked[mask] = self.mask_token_id
        return masked, mask

    def _mask_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_mode == "column":
            return self._mask_tokens_column(tokens)
        if self.mask_mode == "mixed":
            return self._mask_tokens_column(tokens) if random.random() < 0.5 else self._mask_tokens_random(tokens)
        return self._mask_tokens_random(tokens)

    # Shared transformer forward

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pre-readout hidden state x (B, L, embed_dim): everything up to but NOT
        including the tied-embedding logit projection. Exposed so an external
        analytic (ridge) readout can be fit/applied in place of `x @ Eᵀ`
        (FLARE / analytic-head experiments). Behaviour of `_logits` is unchanged."""
        L = tokens.shape[1]
        self._build_positions_3d(L, tokens.device)
        tok_emb = self.token_embedding(tokens)

        if self.training:
            is_mask = (tokens == self.mask_token_id).unsqueeze(-1)
            tok_emb_drop = F.dropout(tok_emb, p=self.dropout, training=True)
            tok_emb = torch.where(is_mask, tok_emb, tok_emb_drop)

        pos_emb = self._pos_embedding(L, tokens.device)                  # (L, embed_dim)
        # Batch dim added here via broadcast: (1, L, embed_dim) + (B, L, embed_dim).
        x = tok_emb + pos_emb[None, :, :]
        x = self.transformer(x)
        return self.pred_head(x)

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        x = self._hidden(tokens)
        logits = torch.matmul(x, self.token_embedding.weight.T)
        return logits[:, :, : self.codebook_size] + self.output_bias[:L]

    def forward(self, tokens: torch.Tensor) -> PriorOutput:
        masked_tokens, mask = self._mask_tokens(tokens)
        logits = self._logits(masked_tokens)
        target = tokens[mask]
        masked_logits = logits[mask]
        loss = (
            F.cross_entropy(masked_logits, target, label_smoothing=self.label_smoothing)
            if masked_logits.numel() > 0
            else logits.sum() * 0.0
        )
        return PriorOutput(
            loss=loss, token_logits=logits,
            stats={"mask_ratio": mask.float().mean()},
        )

    # ── Scoring: per-token NLL, returned as (B, C, F, W) with NO collapse ──

    @torch.no_grad()
    def score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return τ-summed scores: (B, C, F, W). Paper Algorithm 1."""
        return self.score_tokens_per_rate(tokens).sum(dim=0)

    @torch.no_grad()
    def score_tokens_per_rate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return per-rate scores stacked along axis 0: (n_τ, B, C, F, W).

        Used by detect.py to compute paper-style per-τ per-(C, F) thresholds
        before summing across τ.
        """
        B, N = tokens.shape
        C, F_, W = self.latent_channels, self.latent_freq, self.latent_time
        assert C * F_ * W == N, f"seq_len={N} != C*F*W={C*F_*W}"
        rates = list(self.score_window_size_rates)
        device = tokens.device
        out = torch.zeros(len(rates), B, C, F_, W, device=device)

        # Batch the W per-column masked forwards into chunks of `mb` columns
        # (mb*B rows per forward) instead of one forward per column. The transformer
        # treats batch rows independently, so this is numerically identical up to
        # GEMM-tiling fp noise; `mb=1` reproduces the original path exactly. Biggest
        # win on the B=1 counterfactual callers (W batch-1 forwards → a few wide ones).
        # `score_mask_chunk` is a TARGET rows-per-forward budget; mb adapts to B so a
        # large-B detect pass stays at mb=1 (bit-identical, no OOM) while B=1 CF batches
        # up to all W columns. Unset attribute → default 1 → the original path.
        target_rows = max(1, int(getattr(self, "score_mask_chunk", 1)))
        mb = max(1, min(W, target_rows // max(1, B)))
        base = tokens.reshape(B, C, F_, W)                              # (B, C, F, W)

        for ri, rate in enumerate(rates):
            ks = _paper_kernel_size(W, rate)
            half = ks // 2
            ranges = [(max(0, w - half), min(W, w + half + 1)) for w in range(W)]
            for c0 in range(0, W, mb):
                cols = list(range(c0, min(c0 + mb, W)))
                nc = len(cols)
                # (nc, B, C, F, W): a fresh mask per column-variant (repeat copies).
                masked = base.unsqueeze(0).repeat(nc, 1, 1, 1, 1)
                for j, w in enumerate(cols):
                    lo, hi = ranges[w]
                    masked[j, :, :, :, lo:hi] = self.mask_token_id
                logits = self._logits(masked.reshape(nc * B, N))        # (nc*B, N, K)
                lp = logits.log_softmax(dim=-1).reshape(nc, B, C, F_, W, -1)
                tgt = base.reshape(1, B, C, F_, W, 1).expand(nc, -1, -1, -1, -1, 1)
                gathered = lp.gather(-1, tgt).squeeze(-1)               # (nc, B, C, F, W)
                for j, w in enumerate(cols):
                    lo, hi = ranges[w]
                    # Paper Algorithm 1: a_w averages -log p over the masked range.
                    out[ri, :, :, :, w] = -gathered[j, :, :, :, lo:hi].mean(dim=-1)

        return out                                                        # (n_τ, B, C, F, W)

    # Sampling: iterative parallel decoding

    @torch.no_grad()
    def sample(self, num_samples: int, seq_len: int, device: torch.device) -> torch.Tensor:
        self._build_positions_3d(seq_len, device)
        return _iterative_decode(
            logits_fn=self._logits,
            num_samples=num_samples, seq_len=seq_len,
            mask_token_id=self.mask_token_id, steps=self.steps,
            mask_scheduling_fn=self.mask_scheduling_fn,
            choice_temperature=self.choice_temperature, device=device,
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_prior(name: str, **kwargs) -> nn.Module:
    if name == "maskgit":
        kwargs.pop("mask_mode", None)                    # 2D-only
        return MaskGITPrior(**kwargs)
    if name in ("maskgit_2d_pos", "maskgit_3d_pos", "maskgit_upstream"):
        # focal_gamma / loss_weighting are implemented ONLY by MaskGITPrior. These two
        # variants have no code path that reads them, so accepting a non-default value
        # would silently measure nothing (a swept knob that reports a false null).
        fg = kwargs.pop("focal_gamma", 0.0)
        lw = kwargs.pop("loss_weighting", "uniform")
        if float(fg) != 0.0 or str(lw) != "uniform":
            raise ValueError(
                f"prior.name={name!r} does not implement focal_gamma/loss_weighting "
                f"(got focal_gamma={fg!r}, loss_weighting={lw!r}). Only prior.name='maskgit' does."
            )
        if name == "maskgit_2d_pos":
            return MaskGITPrior2DPos(**kwargs)
        if name == "maskgit_upstream":
            # Upstream's x-transformers stack + flat 1-D positions. Subclasses
            # MaskGITPrior3DPos, so masking / loss / scoring are the SAME code as
            # every other cell and a difference in the result is attributable to
            # the stack alone. Imported here, not at module scope, to keep the
            # x-transformers dependency off the import path of runs that never
            # select it.
            from model.prior_upstream import MaskGITPriorUpstream
            return MaskGITPriorUpstream(**kwargs)
        return MaskGITPrior3DPos(**kwargs)
    raise ValueError(f"Unknown prior: {name!r}")
