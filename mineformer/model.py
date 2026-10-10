"""Hybrid CNN/Transformer encoder with a multi-scale segmentation decoder."""

from collections.abc import Sequence
from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3,
        stride: int = 1, *, activate: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(gcd(8, out_channels), out_channels),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels, activate=False),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.convs(x))


class CNNStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, stride=2),
            *(ResidualBlock(out_channels) for _ in range(depth)),
        )


class TransformerBlock(nn.Module):
    """Pre-norm spatial self-attention and a feed-forward residual branch."""

    def __init__(self, channels: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, channels), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        return tokens + self.mlp(self.norm2(tokens))


class TransformerStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, depth: int,
        heads: int, mlp_ratio: float, dropout: float,
    ) -> None:
        super().__init__()
        self.patch_embed = ConvNormAct(in_channels, out_channels, stride=2)
        # Spatial positional encoding supports variable image sizes without
        # interpolating a fixed-size learned position table.
        self.position = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels,
        )
        self.blocks = nn.Sequential(*(
            TransformerBlock(out_channels, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ))
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = x + self.position(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(self.blocks(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class MultiScaleDecoder(nn.Module):
    """Project all encoder levels, align at stage 1, and fuse together."""

    def __init__(self, channels: Sequence[int], decoder_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            ConvNormAct(channel, decoder_channels, kernel_size=1)
            for channel in channels
        )
        self.fusion = nn.Sequential(
            ConvNormAct(4 * decoder_channels, decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        size = features[0].shape[-2:]
        aligned = [
            F.interpolate(projection(feature), size=size, mode="bilinear",
                          align_corners=False)
            for projection, feature in zip(self.projections, features)
        ]
        return self.fusion(torch.cat(aligned, dim=1))


class MineFormerV1(nn.Module):
    """Binary segmentation with CNN stages 1/2 and Transformer stages 3/4.

    Input: floating point images shaped ``(B, in_channels, H, W)``.
    Output: raw binary logits shaped ``(B, 1, H, W)``. Use
    ``BCEWithLogitsLoss`` for training or ``logits.sigmoid()`` for probabilities.
    Dimensions must be at least 32 pixels. Attention is global within stages
    3/4, so memory usage grows quadratically with their spatial token counts.
    """

    def __init__(
        self, in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
        depths: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (8, 8),
        stem_channels: int = 32, decoder_channels: int = 128,
        mlp_ratio: float = 4.0, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4 or len(heads) != 2:
            raise ValueError("Expected four channels/depths and two attention head counts")
        dimensions = (*channels, *depths, *heads, in_channels, stem_channels, decoder_channels)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in dimensions):
            raise ValueError("Channel counts, depths, and head counts must be positive integers")
        if any(channel % head for channel, head in zip(channels[2:], heads)):
            raise ValueError("Transformer channels must be divisible by their head counts")
        if not mlp_ratio > 0 or any(int(channel * mlp_ratio) < 1 for channel in channels[2:]):
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, stride=2),
            ConvNormAct(stem_channels, stem_channels),
        )
        self.stage1 = CNNStage(stem_channels, channels[0], depths[0])
        self.stage2 = CNNStage(channels[0], channels[1], depths[1])
        self.stage3 = TransformerStage(
            channels[1], channels[2], depths[2], heads[0], mlp_ratio, dropout,
        )
        self.stage4 = TransformerStage(
            channels[2], channels[3], depths[3], heads[1], mlp_ratio, dropout,
        )
        self.decoder = MultiScaleDecoder(channels, decoder_channels)
        self.segmentation_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input shaped (B, {self.in_channels}, H, W)")
        if min(x.shape[-2:]) < 32:
            raise ValueError("Input height and width must be at least 32")
        stage1 = self.stage1(self.stem(x))
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stage1, stage2, stage3, stage4

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        logits = self.segmentation_head(self.decoder(features))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
