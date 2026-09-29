"""
=============================================================================
  Time-domain → time-frequency transforms (with differentiable inverse).
=============================================================================

Two transforms are supported:
  * `STFTTransform`     — real+imag channels, (B, C, T) → (B, 2C, F, T').
                          Differentiable forward, exact inverse.
  * `IdentityTransform` — no-op (useful for 1D experiments).

Usage:
    transform = STFTTransform(n_fft=4, normalized=True)
    tf = transform(x)                         # tf.tensor shape: (B, 2C, F, T')
    x_back = transform.inverse(reprojected, tf.spec)   # (B, C, T)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from model.common import TransformOutput, TransformSpec


class STFTTransform(nn.Module):
    name = "stft"

    def __init__(self, n_fft: int = 4, normalized: bool = True):
        super().__init__()
        self.n_fft = int(n_fft)
        self.normalized = bool(normalized)

    def _output_spec(self, x: torch.Tensor) -> TransformSpec:
        freq_bins = (self.n_fft // 2) + 1
        probe = torch.stft(
            x[:, 0], self.n_fft,
            normalized=self.normalized, return_complex=True,
            window=torch.hann_window(self.n_fft, device=x.device),
        )
        return TransformSpec(
            axes=("batch", "feature_channel", "frequency", "time"),
            original_channels=x.shape[1],
            components_per_channel=2,         # real + imag
            frequency_bins=freq_bins,
            time_steps=probe.shape[-1],
            input_length=x.shape[-1],
        )

    def forward(self, x: torch.Tensor) -> TransformOutput:
        B, C, _ = x.shape
        window = torch.hann_window(self.n_fft, device=x.device)
        xf = rearrange(x, "b c l -> (b c) l")
        xf = torch.stft(xf, self.n_fft, normalized=self.normalized, return_complex=True, window=window)
        xf = torch.view_as_real(xf)
        xf = rearrange(xf, "(b c) f t z -> b (c z) f t", b=B, c=C)
        return TransformOutput(tensor=xf, spec=self._output_spec(x), aux={"window": window})

    def inverse(self, z: torch.Tensor, spec: TransformSpec) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=z.device)
        x = rearrange(z, "b (c zc) f t -> (b c) f t zc",
                      c=spec.original_channels, zc=spec.components_per_channel)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, self.n_fft, normalized=self.normalized,
                        window=window, length=spec.input_length)
        return rearrange(x, "(b c) l -> b c l", c=spec.original_channels)


class IdentityTransform(nn.Module):
    name = "identity"

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> TransformOutput:
        spec = TransformSpec(
            axes=("batch", "channel", "time"),
            original_channels=x.shape[1],
            components_per_channel=1,
            time_steps=x.shape[-1],
            input_length=x.shape[-1],
        )
        return TransformOutput(tensor=x, spec=spec)

    def inverse(self, z: torch.Tensor, spec: TransformSpec) -> torch.Tensor:
        return z


def build_transform(name: str, **kwargs) -> nn.Module:
    if name == "stft":
        return STFTTransform(**kwargs)
    if name == "identity":
        return IdentityTransform()
    raise ValueError(f"Unknown transform: {name!r}")
