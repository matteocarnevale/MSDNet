"""Reconstruction-Guided Feature Distillation (RGFD) – Section III-C.

Converts sparse 4D radar BEV features into dense representations via a
U-shaped network with deformable convolutions, CBAM attention, and a
ConvNeXt residual block.  The output F_r^R is trained to approximate
the dense LiDAR features F_l^D.
"""

import torch
import torch.nn as nn

from .modules import (
    ConvBNAct,
    DeformableConvBlock,
    CBAM,
    ConvNeXtBlock,
)


class DownBlock(nn.Module):
    """Standard + deformable conv with stride-2 downsampling."""

    def __init__(self, channels: int, use_deformable: bool = False):
        super().__init__()
        if use_deformable:
            self.conv1 = DeformableConvBlock(channels, channels, 3, stride=2, padding=1)
            self.conv2 = ConvBNAct(channels, channels, 3, stride=1, padding=1)
        else:
            self.conv1 = ConvBNAct(channels, channels, 2, stride=2, padding=0)
            self.conv2 = ConvBNAct(channels, channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UpBlock(nn.Module):
    """Transposed conv for stride-2 upsampling (kernel 4×4, stride 2, pad 1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.up = ConvBNAct(channels, channels, kernel_size=4, stride=2,
                            padding=1, transposed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionModule(nn.Module):
    """CBAM → ConvNeXt → CBAM (Section III-C)."""

    def __init__(self, channels: int, reduction: int = 16, expansion: int = 4):
        super().__init__()
        self.cbam1 = CBAM(channels, reduction)
        self.convnext = ConvNeXtBlock(channels, expansion)
        self.cbam2 = CBAM(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam2(self.convnext(self.cbam1(x)))


class AggregationModule(nn.Module):
    """Concatenate two features and fuse with a 3×1 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = ConvBNAct(channels * 2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([x1, x2], dim=1))


class RGFD(nn.Module):
    """
    Reconstruction-Guided Feature Distillation network.

    Data flow (see Fig. 2 bottom-right):
        F_r^S ─→ Down1(H/2) ─→ Down2(H/4) ─→ Attention(H/4)
                    │                              │
                    └─────── Agg ←── Up1(H/2) ←────┘
                              │
                           Up2(H) ──→ F_r^R
    """

    def __init__(self, channels: int = 128, reduction: int = 16,
                 convnext_expansion: int = 4):
        super().__init__()
        self.down1 = DownBlock(channels, use_deformable=False)
        self.down2 = DownBlock(channels, use_deformable=True)
        self.attention = AttentionModule(channels, reduction, convnext_expansion)
        self.up1 = UpBlock(channels)
        self.agg = AggregationModule(channels)
        self.up2 = UpBlock(channels)

    def forward(self, f_sparse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_sparse: (B, C, H, W)  sparse 4D radar BEV features F_r^S
        Returns:
            f_recon:  (B, C, H, W)  reconstructed features F_r^R
        """
        d1 = self.down1(f_sparse)     # (B, C, H/2, W/2)
        d2 = self.down2(d1)           # (B, C, H/4, W/4)
        att = self.attention(d2)      # (B, C, H/4, W/4)
        u1 = self.up1(att)            # (B, C, H/2, W/2)
        agg = self.agg(u1, d1)        # (B, C, H/2, W/2)  skip connection
        return self.up2(agg)          # (B, C, H, W)
