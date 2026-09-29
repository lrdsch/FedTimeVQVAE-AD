"""
=============================================================================
  Building blocks shared by encoder.py and decoder.py.
=============================================================================

Kept small on purpose — each block is a few lines of nn.Module glue.

Block families:
  * 2D:           SnakeActivation, ResBlock, ConvDownsampleBlock, ConvUpsampleBlock
  * 2D grouped:   GroupedResBlock, GroupedProjectBlock, GroupedDownsampleBlock, GroupedUpsampleBlock
  * 3D:           Snake3d, Conv3dResBlock, Conv3dProjectBlock, Conv3dDownsampleBlock, Conv3dUpsampleBlock

Shared math helpers:
  * compute_downsample_rate(input_length, downsampled_length) → int
  * split_feature_channels(num_features, original_channels, components_per_channel)

Shared dataclass:
  * TransformSpec — produced by transforms.py, consumed by encoder + decoder inverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ─── Shared types ────────────────────────────────────────────────────────────

@dataclass
class TransformSpec:
    """Describes the layout of a transformed tensor (needed by inverse STFT)."""
    axes: tuple[str, ...]
    original_channels: int
    components_per_channel: int = 1       # STFT=2 (real/imag), identity=1
    frequency_bins: int | None = None
    time_steps: int | None = None
    input_length: int | None = None


@dataclass
class TransformOutput:
    tensor: torch.Tensor
    spec: TransformSpec
    aux: dict[str, Any] = field(default_factory=dict)


# ─── Math helpers ────────────────────────────────────────────────────────────

def compute_downsample_rate(input_length: int, downsampled_length: int, min_rate: int = 1) -> int:
    rate = round(input_length / max(downsampled_length, 1))
    return max(min_rate, rate)


def split_feature_channels(num_features: int, original_channels: int, components_per_channel: int) -> list[slice]:
    expected = original_channels * components_per_channel
    if num_features != expected:
        raise ValueError(
            f"Expected {expected} feature channels, got {num_features}. "
            "Check transform/encoder compatibility."
        )
    return [
        slice(c * components_per_channel, (c + 1) * components_per_channel)
        for c in range(original_channels)
    ]


# 2D building blocks

class SnakeActivation(nn.Module):
    """Snake activation: x + (sin(αx)² / α). Learnable α per channel."""
    def __init__(self, num_features: int, dim: int, a_base: float = 0.2):
        super().__init__()
        if dim not in {1, 2}:
            raise ValueError("SnakeActivation supports dim=1 or dim=2")
        shape = (1, num_features, 1) if dim == 1 else (1, num_features, 1, 1)
        self.alpha = nn.Parameter(torch.full(shape, a_base))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, min=1e-4)
        return x + (torch.sin(alpha * x) ** 2) / alpha


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        mid = out_channels if mid_channels is None else mid_channels
        self.convs = nn.Sequential(
            SnakeActivation(in_channels, 2),
            nn.Conv2d(in_channels, mid, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(mid),
            SnakeActivation(mid, 2),
            nn.Conv2d(mid, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Dropout(dropout),
        )
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.convs(x)


class ConvDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 4), stride=(1, 2),
                      padding=(0, 1), padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# 2D grouped blocks (channel-independent Conv2d family)

class GroupedResBlock(nn.Module):
    """Grouped Conv2d → every channel gets its own set of filters."""
    def __init__(self, features_per_channel: int, groups: int, dropout: float):
        super().__init__()
        ch = features_per_channel * groups
        self.convs = nn.Sequential(
            SnakeActivation(ch, 2),
            nn.Conv2d(ch, ch, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=groups),
            nn.BatchNorm2d(ch),
            SnakeActivation(ch, 2),
            nn.Conv2d(ch, ch, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=groups),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)


class GroupedProjectBlock(nn.Module):
    def __init__(self, in_per_channel: int, out_per_channel: int, groups: int, dropout: float):
        super().__init__()
        in_ch, out_ch = in_per_channel * groups, out_per_channel * groups
        self.convs = nn.Sequential(
            SnakeActivation(in_ch, 2),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=groups),
            nn.BatchNorm2d(out_ch),
            SnakeActivation(out_ch, 2),
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=groups),
            nn.Dropout(dropout),
        )
        self.proj = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups)
            if in_per_channel != out_per_channel
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.convs(x)


class GroupedDownsampleBlock(nn.Module):
    def __init__(self, in_per_channel: int, out_per_channel: int, groups: int, dropout: float):
        super().__init__()
        in_ch, out_ch = in_per_channel * groups, out_per_channel * groups
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1),
                      groups=groups, padding_mode="replicate"),
            nn.BatchNorm2d(out_ch),
            SnakeActivation(out_ch, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GroupedUpsampleBlock(nn.Module):
    def __init__(self, in_per_channel: int, out_per_channel: int, groups: int, dropout: float):
        super().__init__()
        in_ch, out_ch = in_per_channel * groups, out_per_channel * groups
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), groups=groups),
            nn.BatchNorm2d(out_ch),
            SnakeActivation(out_ch, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# 3D blocks (channel-independent Conv3d family)

class Snake3d(nn.Module):
    def __init__(self, num_features: int, a_base: float = 0.2):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, num_features, 1, 1, 1), a_base))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, min=1e-4)
        return x + (torch.sin(alpha * x) ** 2) / alpha


class Conv3dDownsampleBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_features, out_features, kernel_size=(1, 1, 4), stride=(1, 1, 2),
                      padding=(0, 0, 1), padding_mode="replicate"),
            nn.BatchNorm3d(out_features),
            Snake3d(out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv3dResBlock(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.convs = nn.Sequential(
            Snake3d(features),
            nn.Conv3d(features, features, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            nn.BatchNorm3d(features),
            Snake3d(features),
            nn.Conv3d(features, features, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)


class Conv3dProjectBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.convs = nn.Sequential(
            Snake3d(in_features),
            nn.Conv3d(in_features, out_features, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            nn.BatchNorm3d(out_features),
            Snake3d(out_features),
            nn.Conv3d(out_features, out_features, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            nn.Dropout(dropout),
        )
        self.proj = (
            nn.Conv3d(in_features, out_features, kernel_size=1)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.convs(x)


class Conv3dUpsampleBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_features, out_features, kernel_size=(1, 1, 4),
                               stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.BatchNorm3d(out_features),
            Snake3d(out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
