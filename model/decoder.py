"""
=============================================================================
  Decoders — quantized latent tensor → reconstructed STFT representation.
=============================================================================

Three reconstructor variants, chosen to pair with the encoder:

  * `DefaultReconstructor` (Conv2d TF)
        Pair with PerChannelEncoder. Upsamples from (B, D, H, W) back up to the
        STFT representation `(B, 2C, F, T')` using Conv2d stacks.

  * `ChannelIndependentConv2dReconstructor`
        Pair with ChannelIndependentConv2dEncoder. Grouped ConvTranspose2d.

  * `ChannelIndependentConv3dReconstructor`
        Pair with ChannelIndependentConv3dEncoder. ConvTranspose3d.

Also lives here:

  * `TemporalReconstructor`
        Fallback used when quantized latent is 3D (B, D, L). Not used by the
        active pipelines but kept for the edge case of identity transforms.

  * `RefinementHead`
        Tiny Linear applied to the time-domain reconstruction. Trained to
        correct residuals between (decoded STFT → inverse STFT) and the
        original waveform. Lazy-built on first call.

  * `build_decoder(name, **cfg)` — dispatch by config name.

Each reconstructor is lazy-built (it needs the target shape to pick the depth).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.common import (
    ConvUpsampleBlock, Conv3dProjectBlock, Conv3dResBlock, Conv3dUpsampleBlock,
    GroupedProjectBlock, GroupedResBlock, GroupedUpsampleBlock, ResBlock,
)


# Default Conv2d TF reconstructor

class DefaultReconstructor(nn.Module):
    """Conv2d TF reconstructor — pair with PerChannelEncoder."""

    name = "default"

    def __init__(self, n_resnet_blocks: int = 4, dropout: float = 0.3, **_):
        super().__init__()
        self.n_resnet_blocks = n_resnet_blocks
        self.dropout = dropout
        self.decoder: nn.Module | None = None
        self.output_channels: int | None = None

    def _build(self, latent: torch.Tensor, target_shape: torch.Size) -> None:
        if self.decoder is not None:
            return
        latent_dim = latent.shape[1]
        out_channels = target_shape[1]
        input_width = latent.shape[-1]
        target_width = target_shape[-1]
        upsample_rate = max(1, round(target_width / max(input_width, 1)))
        depth = max(1, int(round(math.log2(max(upsample_rate, 1)))))

        init_ch = max(4, 2 ** math.ceil(math.log2(max(out_channels, 1))))
        enc_channels = [init_ch * (2 ** i) for i in range(depth)]

        layers: list[nn.Module] = [
            ResBlock(latent_dim, enc_channels[-1], dropout=self.dropout)
        ]
        for i in range(len(enc_channels) - 1, 0, -1):
            for _ in range(self.n_resnet_blocks):
                layers.append(ResBlock(enc_channels[i], enc_channels[i], dropout=self.dropout))
            layers.append(ConvUpsampleBlock(enc_channels[i], enc_channels[i - 1], dropout=self.dropout))
        layers.append(ConvUpsampleBlock(enc_channels[0], out_channels, dropout=self.dropout))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
        self.decoder = nn.Sequential(*layers).to(latent.device, latent.dtype)
        self.output_channels = out_channels

    def forward(self, latent: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        self._build(latent, target_shape)
        out = self.decoder(latent)
        return F.interpolate(out, size=target_shape[-2:], mode="nearest")


# Channel-independent Conv2d reconstructor

class ChannelIndependentConv2dReconstructor(nn.Module):
    name = "channel_independent_conv2d"

    def __init__(self, token_embedding_dim: int = 4, n_resnet_blocks: int = 2,
                 dropout: float = 0.2, width_base: int = 4):
        super().__init__()
        self.token_embedding_dim = token_embedding_dim
        self.n_resnet_blocks = n_resnet_blocks
        self.dropout = dropout
        self.width_base = width_base
        self.decoder: nn.Module | None = None

    def _build(self, latent: torch.Tensor, target_shape: torch.Size) -> None:
        if self.decoder is not None:
            return
        out_channels = int(target_shape[1])
        if out_channels % 2 != 0:
            raise ValueError(
                f"Target channels must be 2 * original_channels (STFT), got {out_channels}"
            )
        groups = out_channels // 2                         # = original_channels
        if latent.shape[1] % groups != 0:
            raise ValueError(
                f"Latent features {latent.shape[1]} not divisible by original channels {groups}. "
                "Set quantizer.token_embedding_dim so that latent channels are multiple of original_channels."
            )
        in_features_per_channel = latent.shape[1] // groups
        input_width = int(latent.shape[-1])
        target_width = int(target_shape[-1])
        upsample_rate = max(1, round(target_width / max(input_width, 1)))
        depth = max(1, int(round(math.log2(max(upsample_rate, 1)))))
        max_features = self.width_base * (2 ** max(0, depth - 1))

        layers: list[nn.Module] = [
            GroupedProjectBlock(in_features_per_channel, max_features, groups, self.dropout)
        ]
        current = max_features
        for _ in range(depth - 1):
            for _ in range(self.n_resnet_blocks):
                layers.append(GroupedResBlock(current, groups, self.dropout))
            nxt = max(2, current // 2)
            layers.append(GroupedUpsampleBlock(current, nxt, groups, self.dropout))
            current = nxt
        layers.append(GroupedUpsampleBlock(current, 2, groups, self.dropout))
        layers.append(nn.Conv2d(2 * groups, 2 * groups, kernel_size=1, groups=groups))
        self.decoder = nn.Sequential(*layers).to(latent.device, latent.dtype)

    def forward(self, latent: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        self._build(latent, target_shape)
        out = self.decoder(latent)
        return F.interpolate(out, size=target_shape[-2:], mode="nearest")


# Channel-independent Conv3d reconstructor

class ChannelIndependentConv3dReconstructor(nn.Module):
    name = "channel_independent_conv3d"

    def __init__(self, token_embedding_dim: int = 4, n_resnet_blocks: int = 2,
                 dropout: float = 0.2, width_base: int = 4):
        super().__init__()
        self.token_embedding_dim = token_embedding_dim
        self.n_resnet_blocks = n_resnet_blocks
        self.dropout = dropout
        self.width_base = width_base
        self.decoder: nn.Module | None = None

    def _build(self, latent: torch.Tensor, target_shape: torch.Size) -> None:
        if self.decoder is not None:
            return
        out_channels = int(target_shape[1])
        if out_channels % 2 != 0:
            raise ValueError(
                f"Target channels must be 2 * original_channels, got {out_channels}"
            )
        original_channels = out_channels // 2
        if latent.shape[1] % original_channels != 0:
            raise ValueError(
                f"Latent features {latent.shape[1]} not divisible by original channels {original_channels}."
            )
        in_features_per_channel = latent.shape[1] // original_channels
        input_width = int(latent.shape[-1])
        target_width = int(target_shape[-1])
        upsample_rate = max(1, round(target_width / max(input_width, 1)))
        depth = max(1, int(round(math.log2(max(upsample_rate, 1)))))
        max_features = self.width_base * (2 ** max(0, depth - 1))

        layers: list[nn.Module] = [
            Conv3dProjectBlock(in_features_per_channel, max_features, self.dropout)
        ]
        current = max_features
        for _ in range(depth - 1):
            for _ in range(self.n_resnet_blocks):
                layers.append(Conv3dResBlock(current, self.dropout))
            nxt = max(2, current // 2)
            layers.append(Conv3dUpsampleBlock(current, nxt, self.dropout))
            current = nxt
        layers.append(Conv3dUpsampleBlock(current, 2, self.dropout))
        self.decoder = nn.Sequential(*layers).to(latent.device, latent.dtype)

    def forward(self, latent: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        self._build(latent, target_shape)
        out_channels = int(target_shape[1])
        original_channels = out_channels // 2
        features_per_channel = latent.shape[1] // original_channels
        x5 = rearrange(
            latent, "b (c d) f w -> b d c f w",
            c=original_channels, d=features_per_channel,
        )
        out5 = self.decoder(x5)
        out = rearrange(out5, "b z c f w -> b (c z) f w")
        return F.interpolate(out, size=target_shape[-2:], mode="nearest")


# 1D fallback (identity transform only)

class TemporalReconstructor(nn.Module):
    """Fallback for 3D latents (B, D, L). Only relevant with IdentityTransform."""

    def __init__(self, **_):
        super().__init__()
        self.decoder: nn.Module | None = None

    def _build(self, latent: torch.Tensor, target_shape: torch.Size) -> None:
        if self.decoder is not None:
            return
        channels = latent.shape[1]
        out_channels = target_shape[1]
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(channels, out_channels, kernel_size=4, stride=2, padding=1),
        ).to(latent.device, latent.dtype)

    def forward(self, latent: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        self._build(latent, target_shape)
        out = self.decoder(latent)
        return F.interpolate(out, size=target_shape[-1], mode="linear", align_corners=False)


# Refinement head — residual learned in time domain

class RefinementHead(nn.Module):
    """Learns an additive residual on the time-domain waveform.

    Built lazily on first forward (needs to know the window length). Trained
    to correct residuals for *real* encoder outputs; applying it to
    unconditionally-generated samples (stage 2 quality eval) is OOD and can
    inflate variance — so quality_stage2.py skips it by default.
    """

    def __init__(self, mode: str = "linear"):
        super().__init__()
        self.mode = mode
        self.linear: nn.Linear | None = None
        self.conv: nn.Sequential | None = None

    def _build(self, x: torch.Tensor) -> None:
        if self.mode == "conv":
            if self.conv is not None:
                return
            ch = int(x.shape[1])
            # Small Conv1d residual over time — local receptive field adds
            # high-frequency detail the global Linear(T,T) tends to smooth.
            self.conv = nn.Sequential(
                nn.Conv1d(ch, 16, 5, padding=2), nn.GELU(),
                nn.Conv1d(16, 16, 5, padding=2), nn.GELU(),
                nn.Conv1d(16, ch, 5, padding=2),
            ).to(x.device, x.dtype)
            return
        if self.linear is not None:
            return
        self.linear = nn.Linear(x.shape[-1], x.shape[-1]).to(x.device, x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._build(x)
        if self.mode == "conv":
            return x + self.conv(x)
        return x + self.linear(x)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_decoder(name: str, **kwargs) -> nn.Module:
    if name == "default":
        return DefaultReconstructor(**kwargs)
    if name == "channel_independent_conv2d":
        return ChannelIndependentConv2dReconstructor(**kwargs)
    if name == "channel_independent_conv3d":
        return ChannelIndependentConv3dReconstructor(**kwargs)
    raise ValueError(f"Unknown decoder: {name!r}")
