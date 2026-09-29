"""
=============================================================================
  Vector quantizers — latent tensor → (quantized, indices, commitment loss).
=============================================================================

  * `SharedVectorQuantizer`   — single codebook shared across all spatial positions.
        forward(latent): (B, D, H, W) → QuantizerOutput
          indices   : (B, H*W)
          quantized : (B, D, H, W) with straight-through gradient

  * `SharedCodebookPerChannelVQ` — folds channels into the batch dim and applies
        the same SharedVectorQuantizer (same codebook) to every channel.
        forward(latent): (B, C*d, H, W) → QuantizerOutput
          indices   : (B, C, H*W)
          quantized : (B, C*d, H, W) with straight-through gradient

The codebook embedding dim equals the encoder latent dim — no projection.
This keeps codes interpretable for anomaly-detection metrics (distance to
codebook, code frequency, quantization error).

Both implement EMA codebook update with dead-code expiration, k-means init on
the first training batch, and a perplexity stat for logging.

Both return a `QuantizerOutput` dataclass (defined here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Output container ────────────────────────────────────────────────────────

@dataclass
class QuantizerOutput:
    quantized: torch.Tensor             # (B, D, H, W) straight-through
    indices: torch.Tensor               # (B, H*W) or (B, K, H*W)
    loss: torch.Tensor                  # scalar — commitment term, weighted
    perplexity: torch.Tensor | None = None
    stats: dict[str, Any] = field(default_factory=dict)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _flatten_latent(latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """(B, D, H, W) → (B, H*W, D) or (B, D, L) → (B, L, D)."""
    if latent.ndim == 4:
        b, d, h, w = latent.shape
        return latent.permute(0, 2, 3, 1).reshape(b, h * w, d), (b, d, h, w)
    if latent.ndim == 3:
        b, d, l = latent.shape
        return latent.transpose(1, 2), (b, d, l)
    raise ValueError(f"Unsupported latent shape: {latent.shape}")


def _unflatten_latent(tokens: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    d = tokens.shape[-1]
    if len(shape) == 4:
        b, _, h, w = shape
        return tokens.reshape(b, h, w, d).permute(0, 3, 1, 2)
    b, _, l = shape
    return tokens.transpose(1, 2).reshape(b, d, l)


def _sample_vectors(data: torch.Tensor, n: int) -> torch.Tensor:
    total = data.shape[0]
    if total >= n:
        idx = torch.randperm(total, device=data.device)[:n]
    else:
        idx = torch.randint(0, total, (n,), device=data.device)
    return data[idx]


# ─── Shared VQ ───────────────────────────────────────────────────────────────

class SharedVectorQuantizer(nn.Module):
    """Single codebook shared across every spatial position."""

    name = "shared_vq"

    def __init__(
        self,
        token_embedding_dim: int = 64,
        codebook_size: int = 256,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.embedding_dim = int(token_embedding_dim)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.threshold_dead_code = int(threshold_ema_dead_code)
        # False = start from the uniform init and let the EMA do all the work, as
        # upstream's vendored lucidrains VQ does (`kmeans_init=False`).
        self.kmeans_init = bool(kmeans_init)

        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.weight.data.clone())
        self.register_buffer("initialized", torch.tensor([False]))

        # ── Federated mode ────────────────────────────────────────────────────
        # When `collect_stats_only` is True, local training accumulates RAW
        # (decay-free) per-code sufficient statistics — count n_j and
        # centroid-sum m_j — against the FROZEN broadcast codebook, WITHOUT
        # mutating the codebook or the EMA buffers. A server aggregates these
        # across clients (sum n_j, sum m_j → e_j = M_j / smoothed(N_j)), which is
        # the FedProto/federated-k-means M-step. Off by default → behaviour is
        # bit-for-bit identical to the upstream (non-federated) quantizer.
        self.collect_stats_only: bool = False
        # FedProto encoder ANCHOR (STEP 4, ABLATION knob; 0.0 = off → no effect).
        # Extra pull of the local encoder toward the FROZEN broadcast codebook.
        #
        # CAVEAT — this is NOT an independent mechanism. In frozen-codebook mode
        # the anchor target IS the commitment target, so the total VQ loss is
        #     commitment_weight * L_c + anchor_weight * L_c
        # i.e. `anchor_weight=λ` is EXACTLY equivalent to raising commitment_weight
        # to (commitment_weight + λ). The `federated_anchor` arm is therefore a
        # commitment-weight sweep, not a distinct FedProto regulariser — report it
        # as such. An anchor with independent semantics would need a target that
        # differs from the commitment target (e.g. a prototype from a PREVIOUS
        # round's codebook, or a per-code prototype not equal to the assigned one).
        self.anchor_weight: float = 0.0
        # FedProto ENCODER federation (pipeline/federated.py, arm federated_enc_fedproto).
        # When True, a training forward stashes the PRE-QUANTIZATION tokens together with
        # their code assignments so the trainer can build per-code batch prototypes that
        # still carry gradient (the round stats below are detached, hence unusable for a
        # loss). Plain attributes, NOT buffers → `state_dict()` is unchanged. Default False
        # ⇒ every other arm keeps the exact previous forward and retains nothing.
        self.export_proto_tokens: bool = False
        self._last_tokens: torch.Tensor | None = None
        self._last_indices: torch.Tensor | None = None
        self.register_buffer("round_count", torch.zeros(self.codebook_size))
        self.register_buffer("round_embed_sum",
                             torch.zeros(self.codebook_size, self.embedding_dim))

    # ── EMA helpers ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _kmeans_init(self, flat_tokens: torch.Tensor) -> None:
        """Seed the codebook with real samples (just on the first training step)."""
        if not self.kmeans_init:
            # Mark it done so the branch is never re-entered, and so a checkpoint
            # written in this mode reloads with the same flag state.
            self.initialized.fill_(True)
            return
        if self.initialized.item():
            return
        data = flat_tokens.detach().reshape(-1, self.embedding_dim)
        if data.shape[0] < self.codebook_size:
            return
        centroids = _sample_vectors(data, self.codebook_size)
        for _ in range(10):
            dists = torch.cdist(data, centroids)
            assignments = dists.argmin(dim=-1)
            for k in range(self.codebook_size):
                mask = assignments == k
                if mask.any():
                    centroids[k] = data[mask].mean(dim=0)
        self.codebook.weight.data.copy_(centroids)
        self.ema_embed_sum.copy_(centroids)
        self.ema_cluster_size.fill_(1)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _update_ema(self, flat_tokens: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA update of cluster size & centroid sum (Oord et al., 2017).

        Forced to float32 (autocast disabled) for a numerically stable EMA even
        when the surrounding forward runs under fp16/bf16 autocast."""
        with torch.autocast(device_type=flat_tokens.device.type, enabled=False):
            ft = flat_tokens.float()
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(dim=0)
            embed_sum = ft.T @ one_hot

            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum.T, alpha=1 - self.ema_decay)

            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            new_embed = self.ema_embed_sum / smoothed.unsqueeze(1)
            self.codebook.weight.data.copy_(new_embed)

    @torch.no_grad()
    def _expire_dead_codes(self, flat_tokens: torch.Tensor) -> None:
        """Replace codes that have been unused for too long with real samples.

        fp32-critical (same reason as `_update_ema`): under fp16/bf16 autocast
        `flat_tokens` is half precision while the codebook and the EMA buffers are
        fp32, so the index-put below raises "Index put requires the source and
        destination dtypes match". Disable autocast and cast the samples to the
        destination dtype. Only reachable outside federated suff-stat mode."""
        if self.threshold_dead_code <= 0:
            return
        dead = self.ema_cluster_size < self.threshold_dead_code
        if not dead.any():
            return
        with torch.autocast(device_type=flat_tokens.device.type, enabled=False):
            data = flat_tokens.float().reshape(-1, self.embedding_dim)
            n_dead = int(dead.sum().item())
            replacements = _sample_vectors(data, n_dead)
            self.codebook.weight.data[dead] = replacements.to(self.codebook.weight.dtype)
            self.ema_embed_sum[dead] = replacements.to(self.ema_embed_sum.dtype)
            self.ema_cluster_size[dead] = 1

    # Federated sufficient-statistics (FedProto codebook)

    @torch.no_grad()
    def _accumulate_round_stats(self, flat_tokens: torch.Tensor, indices: torch.Tensor) -> None:
        """Accumulate raw (decay-free) per-code count n_j and centroid-sum m_j
        against the FROZEN broadcast codebook. Does NOT touch the codebook or the
        EMA buffers — these stats are what a client uploads to the server.

        fp32-critical: forced to float32 (autocast disabled) so the server merge
        reproduces the EXACT pooled M-step (Prop. 1) even when the surrounding
        forward runs under fp16/bf16 autocast — half-precision sums would break
        the exactness."""
        with torch.autocast(device_type=flat_tokens.device.type, enabled=False):
            ft = flat_tokens.float()
            one_hot = F.one_hot(indices, self.codebook_size).float()    # (N, K)
            self.round_count += one_hot.sum(dim=0)                      # (K,)
            self.round_embed_sum += (ft.T @ one_hot).T                 # (K, D)

    @torch.no_grad()
    def reset_round_stats(self) -> None:
        self.round_count.zero_()
        self.round_embed_sum.zero_()

    @torch.no_grad()
    def pull_round_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(n_j, m_j) accumulated this round — the message a client uploads."""
        return self.round_count.clone(), self.round_embed_sum.clone()

    @staticmethod
    def codebook_from_stats(
        N: torch.Tensor, M: torch.Tensor, eps: float = 1e-5,
    ) -> torch.Tensor:
        """e_j = M_j / smoothed(N_j) — the Laplace-smoothed centroid.

        Factored out so BOTH the per-round suff-stat merge (`merge_round_stats`)
        and the server-side EMA merge (`_server_merge` in pipeline/federated.py,
        which smooths the ACCUMULATED counts C_j/S_j) derive the codebook by the
        identical formula. The smoothing matches the centralized EMA's Laplace
        term (see `_update_ema`)."""
        K = N.shape[0]
        n = N.sum()
        smoothed = (N + eps) / (n + K * eps) * n
        return M / smoothed.clamp_min(eps).unsqueeze(1)

    @staticmethod
    def merge_round_stats(
        counts: list[torch.Tensor],
        sums: list[torch.Tensor],
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Server side: aggregate clients' (n_j, m_j) into the new codebook.

        Returns (codebook_weight, N, M) where N=Σ_k n_j^k, M=Σ_k m_j^k and
        e_j = M_j / smoothed(N_j) replicates the centralized EMA's Laplace
        smoothing. Proposition 1: for N_j>0, M_j/N_j is the exact pooled
        centroid of the round's assignments (a federated k-means M-step)."""
        N = torch.stack(counts, dim=0).sum(dim=0)                      # (K,)
        M = torch.stack(sums, dim=0).sum(dim=0)                        # (K, D)
        weight = SharedVectorQuantizer.codebook_from_stats(N, M, eps=eps)
        return weight, N, M

    @torch.no_grad()
    def set_codebook(
        self,
        weight: torch.Tensor,
        ema_cluster_size: torch.Tensor | None = None,
        ema_embed_sum: torch.Tensor | None = None,
    ) -> None:
        """Server → client: install the aggregated global codebook (+ optional
        EMA state), mark initialized so local k-means never re-seeds, and clear
        the round accumulators for the next round."""
        self.codebook.weight.data.copy_(weight)
        if ema_cluster_size is not None:
            self.ema_cluster_size.copy_(ema_cluster_size)
        if ema_embed_sum is not None:
            self.ema_embed_sum.copy_(ema_embed_sum)
        self.initialized.fill_(True)
        self.reset_round_stats()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        tokens, shape = _flatten_latent(latent)
        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Latent feature dim {tokens.shape[-1]} != codebook embedding_dim "
                f"{self.embedding_dim}. The encoder latent dim must match "
                f"cfg.quantizer.token_embedding_dim — no projection is applied."
            )
        projected = tokens

        if self.training and not self.collect_stats_only:
            self._kmeans_init(projected)

        # Nearest-neighbour in embedding space, no grad (codebook is updated via EMA).
        # No autocast guard needed: `torch.cdist` is on autocast's fp32 promote-list, so
        # the distance math is fp32 even under fp16 autocast (verified on torch 2.11).
        # Token ids can still shift on near-ties because `projected` itself is the fp16
        # encoder output — measured 0.067% of tokens on smap/T-1 under detect_amp=fp16.
        with torch.no_grad():
            distances = torch.cdist(projected, self.codebook.weight.detach().unsqueeze(0))
        indices = distances.argmin(dim=-1)                             # (B, L)
        quantized_tokens = self.codebook(indices)                      # (B, L, D)

        if self.training and self.export_proto_tokens:
            # FedProto: hand the trainer the grad-carrying tokens + assignments of THIS
            # step. `_local_train_stage1` consumes and clears them on every step, so the
            # autograd graph is never retained past `backward()`.
            self._last_tokens, self._last_indices = projected, indices

        if self.training:
            flat_proj = projected.detach().reshape(-1, self.embedding_dim)
            flat_idx = indices.reshape(-1)
            if self.collect_stats_only:
                # Federated: accumulate raw stats only; codebook stays frozen.
                self._accumulate_round_stats(flat_proj, flat_idx)
            else:
                self._update_ema(flat_proj, flat_idx)
                self._expire_dead_codes(flat_proj)

        commitment_loss = F.mse_loss(projected, quantized_tokens.detach())
        loss = self.commitment_weight * commitment_loss

        anchor_term = None
        if self.collect_stats_only and self.anchor_weight > 0.0:
            # FedProto anchor: extra pull toward the frozen broadcast codebook.
            anchor_term = self.anchor_weight * commitment_loss
            loss = loss + anchor_term

        # Straight-through: gradient flows as if quantized were projected.
        quantized_st = projected + (quantized_tokens - projected).detach()
        quantized = _unflatten_latent(quantized_st, shape)

        # Perplexity: how spread are the codebook assignments over a batch?
        assignments = F.one_hot(indices, num_classes=self.codebook_size).float().mean(dim=(0, 1))
        perplexity = torch.exp(-(assignments * torch.log(assignments + 1e-10)).sum())

        return QuantizerOutput(
            quantized=quantized,
            indices=indices,
            loss=loss,
            perplexity=perplexity,
            stats={
                "commitment_loss": commitment_loss.detach(),
                "weighted_commitment_loss": loss.detach(),
                **({"anchor_loss": anchor_term.detach()} if anchor_term is not None else {}),
            },
        )


# Shared-codebook per-channel VQ (target path)

class SharedCodebookPerChannelVQ(nn.Module):
    """Per-channel quantization against a single SHARED codebook.

    Channels are folded into the batch dim so the inner SharedVectorQuantizer
    applies the same codebook to every channel. Token IDs therefore carry the
    same meaning across channels. The codebook embedding dim equals the
    encoder's per-channel latent dim — no projection is applied.

    Input  latent   : (B, C*d, F, W')        d = token_embedding_dim
    Output quantized: (B, C*d, F, W')
    Output indices  : (B, C, F*W')
    """

    name = "shared_codebook_per_channel_vq"

    def __init__(
        self,
        token_embedding_dim: int = 64,
        codebook_size: int = 256,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self._vq = SharedVectorQuantizer(
            token_embedding_dim=token_embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
        )
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        # Set by Stage1VQVAE on first forward (needs STFT spec → original_channels).
        self.groups: int | None = None

    def set_groups(self, groups: int) -> None:
        self.groups = int(groups)

    # Uniform accessors so callers work for both single- and residual-VQ (which has
    # no single `._vq`). collect_stats_only / anchor_weight delegate to the inner VQ.
    def inner_vqs(self) -> list["SharedVectorQuantizer"]:
        return [self._vq]

    @property
    def collect_stats_only(self) -> bool:
        return self._vq.collect_stats_only

    @collect_stats_only.setter
    def collect_stats_only(self, value: bool) -> None:
        self._vq.collect_stats_only = bool(value)

    @property
    def anchor_weight(self) -> float:
        return self._vq.anchor_weight

    @anchor_weight.setter
    def anchor_weight(self, value: float) -> None:
        self._vq.anchor_weight = float(value)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        assert latent.ndim == 4, f"Expected (B, D, F, W), got {latent.shape}"
        assert self.groups is not None, "Call set_groups(C) before forward"
        B, D, F_, W = latent.shape
        C = self.groups
        assert D % C == 0, f"D={D} not divisible by groups={C}"
        d = D // C

        # Fold channels into batch → one example per (b, c). Same VQ weights
        # are applied to every channel: shared projection, shared codebook.
        # reshape (not view) for robustness on non-contiguous inputs.
        z = latent.reshape(B, C, d, F_, W).reshape(B * C, d, F_, W)
        out = self._vq(z)

        d_out = out.quantized.shape[1]                             # = D_emb
        quantized = out.quantized.reshape(B, C, d_out, F_, W).reshape(B, C * d_out, F_, W)
        indices = out.indices.reshape(B, C, F_ * W)                # per-channel tokens

        return QuantizerOutput(
            quantized=quantized,
            indices=indices,
            loss=out.loss,
            perplexity=out.perplexity,
            stats=out.stats,
        )

    @torch.no_grad()
    def embed_indices(self, indices: torch.Tensor, latent_spatial: tuple[int, int]) -> torch.Tensor:
        """(B, C, F*W') indexes → (B, C*d, F, W') quantized."""
        assert indices.ndim == 3, f"Expected (B, C, F*W), got {indices.shape}"
        assert self.groups is not None, "Call set_groups(C) before embed_indices"
        B, C, N = indices.shape
        assert C == self.groups, f"indices channels {C} != groups {self.groups}"
        F_, W = int(latent_spatial[0]), int(latent_spatial[1])
        assert F_ * W == N, f"F*W={F_*W} != N={N}"

        idx_flat = indices.reshape(B * C, N)
        emb = self._vq.codebook(idx_flat)                          # (B*C, N, d)
        d = emb.shape[-1]
        return emb.transpose(1, 2).reshape(B, C, d, F_, W).reshape(B, C * d, F_, W)


# Residual (multi-stage) shared-codebook VQ

class ResidualSharedCodebookVQ(nn.Module):
    """Residual VQ: `n_stages` SharedCodebookPerChannelVQ stages, each quantizing the
    RESIDUAL of the previous. Reconstruction ẑ = Σ_s q_s recovers the fine structure a
    single flat codebook smooths away, at effective capacity ∏_s K_s cells with only
    Σ_s K_s codewords. Keeps ALL stages as ordinary shared codebooks so each federates
    exactly by the same suff-stat (k-FED) M-step, independently per stage.

    Input  latent   : (B, C*d, F, W')
    Output quantized: (B, C*d, F, W')          — the SUMMED reconstruction (decoder input)
    Output indices  : (B, C*S, F*W')           — stages folded into the channel dim so the
        stage-2 prior models them as extra channels (its 3D-pos grid takes C*S, F, W
        verbatim — no stage-2/detect change needed). Score = prior-NLL over all C*S streams.
    """

    name = "residual_shared_codebook_per_channel_vq"

    def __init__(
        self,
        n_stages: int = 2,
        token_embedding_dim: int = 64,
        codebook_size: int = 256,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.n_stages = int(n_stages)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.stages = nn.ModuleList([
            SharedCodebookPerChannelVQ(
                token_embedding_dim=token_embedding_dim, codebook_size=codebook_size,
                commitment_weight=commitment_weight, ema_decay=ema_decay, eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                kmeans_init=kmeans_init,
            ) for _ in range(self.n_stages)
        ])
        self.groups: int | None = None

    def set_groups(self, groups: int) -> None:
        self.groups = int(groups)
        for st in self.stages:
            st.set_groups(groups)

    # Federated mode fans out to every stage's inner VQ (each merges independently).
    @property
    def collect_stats_only(self) -> bool:
        return self.stages[0]._vq.collect_stats_only

    @collect_stats_only.setter
    def collect_stats_only(self, value: bool) -> None:
        for st in self.stages:
            st._vq.collect_stats_only = bool(value)

    def inner_vqs(self) -> list["SharedVectorQuantizer"]:
        """Per-stage inner VQs — the federation loops over these (one k-FED merge each)."""
        return [st._vq for st in self.stages]

    @property
    def anchor_weight(self) -> float:
        return self.stages[0]._vq.anchor_weight

    @anchor_weight.setter
    def anchor_weight(self, value: float) -> None:
        for st in self.stages:
            st._vq.anchor_weight = float(value)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        residual = latent
        q_sum = None
        idx_list: list[torch.Tensor] = []
        loss = latent.new_zeros(())
        last = None
        for st in self.stages:
            out = st(residual)                       # (B, C*d, F, W), idx (B, C, F*W)
            q = out.quantized
            q_sum = q if q_sum is None else q_sum + q
            idx_list.append(out.indices)
            loss = loss + out.loss
            last = out
            # Straight-through of q makes (residual - q) stop-grad → only stage 0 passes
            # the encoder gradient (standard RVQ); stages ≥1 refine via their own codebooks.
            residual = residual - q
        indices = torch.cat(idx_list, dim=1)         # (B, C*S, F*W')  stages → channels
        return QuantizerOutput(
            quantized=q_sum, indices=indices, loss=loss,
            perplexity=last.perplexity, stats=last.stats,
        )

    @torch.no_grad()
    def embed_indices(self, indices: torch.Tensor, latent_spatial: tuple[int, int]) -> torch.Tensor:
        """(B, C*S, F*W') → summed quantized (B, C*d, F, W'). Splits the folded stage
        channels back out and sums each stage's codebook lookup."""
        B, CS, N = indices.shape
        S = self.n_stages
        C = CS // S
        q = None
        for s, st in enumerate(self.stages):
            idx_s = indices[:, s * C:(s + 1) * C, :]
            e = st.embed_indices(idx_s, latent_spatial)
            q = e if q is None else q + e
        return q


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_quantizer(name: str, **kwargs) -> nn.Module:
    if name == "shared_vq":
        return SharedVectorQuantizer(**kwargs)
    if name == "shared_codebook_per_channel_vq":
        return SharedCodebookPerChannelVQ(**kwargs)
    if name == "residual_shared_codebook_per_channel_vq":
        return ResidualSharedCodebookVQ(**kwargs)
    raise ValueError(f"Unknown quantizer: {name!r}")
