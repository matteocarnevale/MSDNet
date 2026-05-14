"""Reconstruction-Guided Feature Distillation (RGFD) – Section III-C.

Converts sparse 4D radar BEV features into dense representations via a
U-shaped network with deformable convolutions, CBAM attention, and a
ConvNeXt residual block.

Doppler conditioning (new):
  A DopplerFusion module injects the Doppler BEV map (from DopplerBEVMap)
  as a residual additive term after the attention block and again at the
  output. This is physically motivated: regions with high Doppler variance
  (moving objects) should be treated differently from static clutter.

  The fusion is deliberately lightweight (a single 1×1 projection per stage)
  to keep edge-deployment costs minimal (~32×128 ≈ 4 K extra params each).
"""

from typing import Optional

import torch
import torch.nn as nn

from .modules import ConvBNAct, DeformableConvBlock, CBAM, ConvNeXtBlock


# ---------------------------------------------------------------------------
# Doppler fusion (lightweight residual injection)
# ---------------------------------------------------------------------------

class DopplerFusion(nn.Module):
    """
    Adds Doppler BEV context to a feature map via a residual path.

        f_out = f_in + proj(doppler_bev)

    where proj is a 1×1 Conv → BN → GELU.
    If doppler_bev is None the module is a no-op, making it easy to drop.
    """

    def __init__(self, feat_channels: int, doppler_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(doppler_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
        )

    def forward(self, f: torch.Tensor,
                doppler_bev: Optional[torch.Tensor]) -> torch.Tensor:
        if doppler_bev is None:
            return f
        return f + self.proj(doppler_bev)


# ---------------------------------------------------------------------------
# RGFD building blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    def __init__(self, channels: int, use_deformable: bool = False):
        super().__init__()
        if use_deformable:
            self.conv1 = DeformableConvBlock(channels, channels, 3, stride=2, padding=1)
            self.conv2 = ConvBNAct(channels, channels, 3, stride=1, padding=1)
        else:
            self.conv1 = ConvBNAct(channels, channels, 2, stride=2, padding=0)
            self.conv2 = ConvBNAct(channels, channels, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class UpBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up = ConvBNAct(channels, channels, kernel_size=4, stride=2,
                            padding=1, transposed=True)

    def forward(self, x):
        return self.up(x)


class AttentionModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, expansion: int = 4):
        super().__init__()
        self.cbam1    = CBAM(channels, reduction)
        self.convnext = ConvNeXtBlock(channels, expansion)
        self.cbam2    = CBAM(channels, reduction)

    def forward(self, x):
        return self.cbam2(self.convnext(self.cbam1(x)))


class AggregationModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = ConvBNAct(channels * 2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))


# ---------------------------------------------------------------------------
# RGFD
# ---------------------------------------------------------------------------

class RGFD(nn.Module):
    """
    Reconstruction-Guided Feature Distillation network (Fig. 2 bottom-right).

    Data flow:
        F_r^S → Down1(H/2) → Down2(H/4) → [Doppler fusion] → Attention
                   │                                                │
                   └──────── Agg ←── Up1(H/2) ←──────────────────┘
                               │
                            Up2(H) → [Doppler fusion] → F_r^R

    The Doppler BEV map is injected at two points:
      1. After downsampling to H/4, before attention (helps the attention
         module focus on high-Doppler regions that correspond to dynamic targets).
      2. After the final upsampling, as a residual (preserves velocity context
         in the output features that feed the distillation loss).
    """

    def __init__(self, channels: int = 128, reduction: int = 16,
                 convnext_expansion: int = 4,
                 doppler_channels: int = 0) -> None:
        super().__init__()
        self.down1     = DownBlock(channels, use_deformable=False)
        self.down2     = DownBlock(channels, use_deformable=True)
        self.attention = AttentionModule(channels, reduction, convnext_expansion)
        self.up1       = UpBlock(channels)
        self.agg       = AggregationModule(channels)
        self.up2       = UpBlock(channels)

        # Doppler conditioning: two lightweight injection points
        # If doppler_channels == 0, use no-op DopplerFusion
        self.has_doppler = doppler_channels > 0
        if self.has_doppler:
            self.dop_pre_attn = DopplerFusion(channels, doppler_channels)
            self.dop_output   = DopplerFusion(channels, doppler_channels)
        else:
            self.dop_pre_attn = None
            self.dop_output   = None

    def forward(self, f_sparse: torch.Tensor,
                doppler_bev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            f_sparse:    (B, C, H, W) sparse radar BEV features F_r^S
            doppler_bev: (B, D, H, W) Doppler BEV map (optional)
        Returns:
            f_recon:     (B, C, H, W) reconstructed features F_r^R
        """
        d1 = self.down1(f_sparse)              # (B, C, H/2, W/2)
        d2 = self.down2(d1)                    # (B, C, H/4, W/4)

        # Doppler fusion before attention: resize BEV to H/4 if needed
        if self.has_doppler and doppler_bev is not None:
            # Downsample Doppler map to match d2's spatial size
            dop_low = torch.nn.functional.interpolate(
                doppler_bev, size=d2.shape[-2:], mode="bilinear", align_corners=False
            )
            d2 = self.dop_pre_attn(d2, dop_low)

        att = self.attention(d2)               # (B, C, H/4, W/4)
        u1  = self.up1(att)                    # (B, C, H/2, W/2)
        agg = self.agg(u1, d1)                 # (B, C, H/2, W/2)
        out = self.up2(agg)                    # (B, C, H, W)

        # Doppler fusion at full resolution output
        if self.has_doppler and doppler_bev is not None:
            out = self.dop_output(out, doppler_bev)

        return out
