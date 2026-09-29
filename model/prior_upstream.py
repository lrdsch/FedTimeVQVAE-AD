"""The stage-2 prior of the ORIGINAL TimeVQVAE-AD, as a selectable variant.

WHY THIS EXISTS
---------------
The 2026-08-03 upstream ablation drove every configurable difference with
ML4ITS/TimeVQVAE-AnomalyDetection one at a time and found our choice better on
each axis measured (`ema08` −0.056, `dropout03` −0.193 AUPRC on `ucr_001`) — and
yet `centralized` sits 0.284 below the authors on that same series and 0.417
below their code run by us on `ucr_043`. Those two facts cannot both be
explained by the knobs. Two of the eight verified differences were never
measurable at all, because both live in the prior and neither had an
implementation here:

  * the positional scheme — upstream indexes ONE flat learned position over the
    flattened (C·F·W) token grid; `MaskGITPrior3DPos` factorises into
    E_freq[f] + E_time[t] (+ a degenerate E_ch at C=1);
  * the transformer stack — upstream is x-transformers'
    `ContinuousTransformerWrapper` with RMSNorm, post-embedding norm, and
    attention run at heads·attn_dim_head=256 projected in/out of hidden_dim=128;
    ours is a stock `nn.TransformerEncoder` with LayerNorm whose per-head width
    is embed_dim/heads = 32.

This module supplies both, verbatim, so they stop being an untested hypothesis.

WHAT IS DELIBERATELY *NOT* RE-IMPLEMENTED
-----------------------------------------
Only `__init__` and the two positional hooks are overridden. Masking, the
training loss, `score_tokens`, `score_tokens_per_rate` and `sample` are
INHERITED from `MaskGITPrior3DPos` — the exact code every other cell in the
cohort runs. That is the point: if the measurement machinery were re-written
alongside the model, a difference in the result could not be attributed to the
model. Here it can.

The inherited scoring needs the (C, F, W) factorisation to shape its output, so
`set_latent_shape` is still called and still meaningful; what changes is only
that positions are *indexed* flat instead of factorised. `Stage2System._inform_latent_shape`
dispatches on `isinstance(prior, MaskGITPrior3DPos)`, which this subclass satisfies.

FIDELITY LEDGER vs upstream `models/stage2/bidirectional_transformer.py`
------------------------------------------------------------------------
  identical  tok_emb = Embedding(codebook_size+1, embed_dim), mask id = codebook_size
  identical  pos_emb = Embedding(num_tokens+1, embed_dim), read as `.weight[:n]`
  identical  ContinuousTransformerWrapper(dim_in=dim_out=embed_dim,
             max_seq_len=num_tokens+1, use_abs_pos_emb=False, post_emb_norm=True,
             attn_layers=Encoder(pre_norm=True, dim=hidden_dim, depth, heads,
             attn_dim_head, use_rmsnorm, ff_mult, layer/attn/ff dropout=dropout))
  identical  pred_head = Linear → GELU → LayerNorm(eps=1e-12)
  identical  training-time embedding dropout that SKIPS mask-token positions
  identical  weight-tied logits `x @ tok_emb.Tᵀ` + a learned per-position bias
  differs    the bias is (L, codebook_size) here vs (L, codebook_size+1) upstream
             with the last column sliced off. The dropped column never reaches a
             loss or a score in either repo, so this is shape bookkeeping, not a
             behavioural difference.
  differs    built LAZILY on the first forward, because L = C·F·W is discovered by
             `Stage2System.materialize()` rather than configured. Upstream knows
             `num_tokens` up front. Same modules, later construction — and it
             happens before any optimiser is built (`federated_eval.py`
             materialises, then reads `.parameters()`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.prior import MaskGITPrior3DPos


class MaskGITPriorUpstream(MaskGITPrior3DPos):
    """MaskGIT prior with upstream's x-transformers stack and flat 1-D positions."""

    name = "maskgit_upstream"

    def __init__(
        self,
        *args,
        attn_dim_head: int = 64,
        ff_mult: int = 4,
        use_rmsnorm: bool = True,
        post_emb_norm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attn_dim_head = int(attn_dim_head)
        self.ff_mult = int(ff_mult)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.post_emb_norm = bool(post_emb_norm)

        # The parent built a stock nn.TransformerEncoder and three factorised
        # positional tables. Drop both: `transformer` is rebuilt lazily as the
        # x-transformers wrapper (it needs max_seq_len, i.e. L), and positions
        # become a single flat table. Dropping them here rather than skipping the
        # parent __init__ keeps every other attribute (mask schedule, rates,
        # choice_temperature, codebook_size, …) in one place.
        self.transformer = None
        self.channel_embedding = None
        self.freq_embedding = None
        self.time_embedding = None
        self.pos_embedding: nn.Embedding | None = None

        # Upstream's pred_head normalises with eps=1e-12, not torch's 1e-5 default.
        self.pred_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim, eps=1e-12),
        )

    # ── Lazy build: flat positional table + upstream transformer ────────────

    def _build_positions_3d(self, seq_len: int, device: torch.device) -> None:
        """Overrides the parent's three-table build.

        Keeps the parent's name and signature so the inherited `_hidden` — and
        therefore the inherited scoring path — calls this without knowing which
        prior it is driving. The C·F·W assertion is kept: the inherited
        `score_tokens_per_rate` reshapes to (B, C, F, W) and would produce
        silently wrong per-band scores if the factorisation did not match.
        """
        C, F_, W = self.latent_channels, self.latent_freq, self.latent_time
        assert C * F_ * W == seq_len, f"seq_len={seq_len} != C*F*W={C * F_ * W}"

        if self.pos_embedding is None or self.pos_embedding.num_embeddings < seq_len + 1:
            self.pos_embedding = nn.Embedding(seq_len + 1, self.embed_dim).to(device)
        if self.transformer is None:
            self.transformer = _build_upstream_blocks(
                num_tokens=seq_len, embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                depth=self.depth, heads=self.heads, attn_dim_head=self.attn_dim_head,
                ff_mult=self.ff_mult, use_rmsnorm=self.use_rmsnorm,
                post_emb_norm=self.post_emb_norm, dropout=self.dropout,
            ).to(device)
        if self.output_bias is None or self.output_bias.shape[0] < seq_len:
            self.output_bias = nn.Parameter(
                torch.zeros(seq_len, self.codebook_size, device=device))

    def _pos_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """One flat learned position per token — upstream's `pos_emb.weight[:n, :]`.

        Returns (seq_len, embed_dim); the batch axis is broadcast by the inherited
        `_hidden`, exactly as for the factorised parent.
        """
        return self.pos_embedding.weight[:seq_len, :]


def _build_upstream_blocks(*, num_tokens: int, embed_dim: int, hidden_dim: int,
                           depth: int, heads: int, attn_dim_head: int, ff_mult: int,
                           use_rmsnorm: bool, post_emb_norm: bool,
                           dropout: float) -> nn.Module:
    """Upstream's `ContinuousTransformerWrapper`, argument for argument.

    Imported inside the function so that `import prior_upstream` stays cheap and,
    more importantly, so that a missing `x-transformers` fails loudly HERE — at
    the moment a run actually asks for this prior — instead of at module import,
    where it would break every arm that never uses it.
    """
    from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder

    return ContinuousTransformerWrapper(
        dim_in=embed_dim,
        dim_out=embed_dim,
        max_seq_len=num_tokens + 1,
        use_abs_pos_emb=False,
        post_emb_norm=post_emb_norm,
        attn_layers=TFEncoder(
            pre_norm=True,
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            attn_dim_head=attn_dim_head,
            use_rmsnorm=use_rmsnorm,
            ff_mult=ff_mult,
            layer_dropout=dropout,      # stochastic depth
            attn_dropout=dropout,
            ff_dropout=dropout,
        ),
    )
