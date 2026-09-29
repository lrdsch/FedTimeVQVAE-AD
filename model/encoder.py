"""
=============================================================================
  Encoders — STFT tensor → latent tensor (B, D, H, W).
=============================================================================

Three encoder variants, all lazy-built on first forward (they need the STFT
tensor shape to compute the correct depth/downsample rate):

  * `PerChannelEncoder`
        One independent Conv2d stack per original channel → averaged at the end.
        Input:  (B, 2C, F, T') from STFT
        Output: (B, dim, H, W)   (H = freq_bins, W ≈ downsampled_width)

  * `ChannelIndependentConv2dEncoder`
        Grouped Conv2d with `groups=original_channels` so each channel has
        its own filters, but they're batched in a single forward pass.
        Output: (B, token_embedding_dim * original_channels, H, W)

  * `ChannelIndependentConv3dEncoder`
        Treats (original_channel, freq, time) as a 3D volume with real/imag
        as the input feature dim. Per-channel independence achieved by
        kernel size = 1 along the channel axis.
        Output: (B, token_embedding_dim * original_channels, H, W) after un-folding

All three read the transform output's (`original_channels`,
`components_per_channel`) to size themselves correctly.

Lazy-build pattern: we don't know the final spatial dims until we see a real
batch (STFT output depends on `window_length` and `n_fft`), so the layers are
constructed inside the first forward call. Stage 1 explicitly materialises
the model once before training via `materialize()` so every parameter is
registered before the optimiser is built.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from model.common import (
    ConvDownsampleBlock, Conv3dDownsampleBlock, Conv3dProjectBlock, Conv3dResBlock,
    GroupedDownsampleBlock, GroupedProjectBlock, GroupedResBlock,
    ResBlock, TransformOutput,
    compute_downsample_rate, split_feature_channels,
)


# “Per-channel encoder” — one stack per channel, averaged.” — “─?” — “?” — “─?” — “?” — “─?” — “─?” — “─?”?” — “─?”?”?”

class PerChannelEncoder(nn.Module):
    """One independent Conv2d stack per original channel, then averaged."""

    name = "per_channel_encoder"

    def __init__(
        self,
        dim: int = 64,
        downsampled_width: int = 32,
        n_resnet_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = dim
        self.downsampled_width = downsampled_width
        self.n_resnet_blocks = n_resnet_blocks
        self.dropout = dropout
        self.encoders: nn.ModuleList | None = None

    def _make_stack(self, in_channels: int, input_width: int) -> nn.Sequential:
        downsample_rate = compute_downsample_rate(input_width, self.downsampled_width)
        depth = max(1, int(round(math.log2(max(downsample_rate, 1)))))
        channels = max(4, 2 ** math.ceil(math.log2(max(in_channels, 1))))
        layers: list[nn.Module] = [
            ConvDownsampleBlock(in_channels, channels, dropout=self.dropout)
        ]
        for _ in range(max(0, depth - 1)):
            nxt = channels * 2
            layers.append(ConvDownsampleBlock(channels, nxt, dropout=self.dropout))
            for _ in range(self.n_resnet_blocks):
                layers.append(ResBlock(nxt, nxt, dropout=self.dropout))
            channels = nxt
        layers.append(ResBlock(channels, self.latent_dim, dropout=self.dropout))
        return nn.Sequential(*layers)

    def _build(self, tf: TransformOutput) -> None:
        if self.encoders is not None:
            return
        spec, x = tf.spec, tf.tensor
        per_ch = spec.components_per_channel
        self.encoders = nn.ModuleList([
            self._make_stack(per_ch, x.shape[-1])
            for _ in range(spec.original_channels)
        ]).to(device=x.device, dtype=x.dtype)

    def forward(self, tf: TransformOutput) -> torch.Tensor:
        self._build(tf)
        x, spec = tf.tensor, tf.spec
        slices = split_feature_channels(x.shape[1], spec.original_channels, spec.components_per_channel)
        encoded = [enc(x[:, s]) for enc, s in zip(self.encoders, slices)]
        return torch.stack(encoded, dim=0).mean(dim=0)


# Channel-independent Conv2d encoder (grouped convs)

class ChannelIndependentConv2dEncoder(nn.Module):
    """Single forward pass with grouped convs — one filter set per channel."""

    name = "channel_independent_conv2d"

    def __init__(
        self,
        token_embedding_dim: int = 4,
        downsampled_width: int = 32,
        n_resnet_blocks: int = 2,
        dropout: float = 0.2,
        width_base: int = 4,
    ):
        super().__init__()
        self.token_embedding_dim = token_embedding_dim
        self.downsampled_width = downsampled_width
        self.n_resnet_blocks = n_resnet_blocks
        self.dropout = dropout
        self.width_base = width_base
        self.encoder: nn.Sequential | None = None

    def _build(self, tf: TransformOutput) -> None:
        if self.encoder is not None:
            return
        spec, x = tf.spec, tf.tensor
        if spec.components_per_channel != 2:
            raise ValueError(
                f"{self.name} expects 2 components per channel (real/imag), "
                f"got {spec.components_per_channel}. Use STFT transform."
            )
        groups = spec.original_channels
        downsample_rate = compute_downsample_rate(x.shape[-1], self.downsampled_width)
        depth = max(1, int(round(math.log2(max(downsample_rate, 1)))))
        features = self.width_base
        layers: list[nn.Module] = [
            GroupedDownsampleBlock(2, features, groups, self.dropout)
        ]
        for _ in range(max(0, depth - 1)):
            nxt = features * 2
            layers.append(GroupedDownsampleBlock(features, nxt, groups, self.dropout))
            for _ in range(self.n_resnet_blocks):
                layers.append(GroupedResBlock(nxt, groups, self.dropout))
            features = nxt
        layers.append(
            GroupedProjectBlock(features, self.token_embedding_dim, groups, self.dropout)
        )
        self.encoder = nn.Sequential(*layers).to(device=x.device, dtype=x.dtype)

    def forward(self, tf: TransformOutput) -> torch.Tensor:
        x = tf.tensor
        if x.ndim != 4:
            raise ValueError(f"{self.name} expects a 4D tensor, got {x.shape}")
        self._build(tf)
        assert self.encoder is not None
        return self.encoder(x)


# Channel-independent Conv3d encoder

class ChannelIndependentConv3dEncoder(nn.Module):
    """3D convolution over (feature, channel, freq, time) — channel independence."""

    name = "channel_independent_conv3d"

    def __init__(
        self,
        token_embedding_dim: int = 4,
        downsampled_width: int = 32,
        n_resnet_blocks: int = 2,
        dropout: float = 0.2,
        width_base: int = 4,
    ):
        super().__init__()
        self.token_embedding_dim = token_embedding_dim
        self.downsampled_width = downsampled_width
        self.n_resnet_blocks = n_resnet_blocks
        self.dropout = dropout
        self.width_base = width_base
        self.encoder: nn.Sequential | None = None

    def _build(self, tf: TransformOutput) -> None:
        if self.encoder is not None:
            return
        spec, x = tf.spec, tf.tensor
        if spec.components_per_channel != 2:
            raise ValueError(
                f"{self.name} expects 2 components per channel (real/imag), "
                f"got {spec.components_per_channel}. Use STFT transform."
            )
        downsample_rate = compute_downsample_rate(x.shape[-1], self.downsampled_width)
        depth = max(1, int(round(math.log2(max(downsample_rate, 1)))))
        features = self.width_base
        layers: list[nn.Module] = [
            Conv3dDownsampleBlock(2, features, self.dropout)
        ]
        for _ in range(max(0, depth - 1)):
            nxt = features * 2
            layers.append(Conv3dDownsampleBlock(features, nxt, self.dropout))
            for _ in range(self.n_resnet_blocks):
                layers.append(Conv3dResBlock(nxt, self.dropout))
            features = nxt
        layers.append(
            Conv3dProjectBlock(features, self.token_embedding_dim, self.dropout)
        )
        self.encoder = nn.Sequential(*layers).to(device=x.device, dtype=x.dtype)

    def forward(self, tf: TransformOutput) -> torch.Tensor:
        x, spec = tf.tensor, tf.spec
        if x.ndim != 4:
            raise ValueError(f"{self.name} expects a 4D tensor, got {x.shape}")
        self._build(tf)
        assert self.encoder is not None
        # Reshape so channel axis becomes a spatial dim (independence via kh=1).
        x5 = rearrange(
            x, "b (c z) f w -> b z c f w",
            c=spec.original_channels, z=spec.components_per_channel,
        )
        out5 = self.encoder(x5)
        # Fold the per-channel features back into the feature dim: (B, C*D, F, W)
        return rearrange(out5, "b d c f w -> b (c d) f w")


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_encoder(name: str, **kwargs) -> nn.Module:
    if name == "per_channel_encoder":
        return PerChannelEncoder(**kwargs)
    if name == "channel_independent_conv2d":
        return ChannelIndependentConv2dEncoder(**kwargs)
    if name == "channel_independent_conv3d":
        return ChannelIndependentConv3dEncoder(**kwargs)
    raise ValueError(f"Unknown encoder: {name!r}")
