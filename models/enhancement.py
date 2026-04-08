"""Feature Enhancement module for the teacher branch.

Implements the S2D (Sparse-to-Dense) module from Sparse2Dense
(Wang et al., NeurIPS 2022) using the same layer definitions as the
original S2D_RPN (Sparse2Dense/det3d/models/necks/rpn.py).

The only adaptation vs. the original code is that the ConvNeXt blocks
use a channel-only LayerNorm (standard ConvNeXt practice) instead of
the resolution-hardcoded nn.LayerNorm([C, H, W]) in the S2D repo,
so the module works with any BEV spatial size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelLayerNorm(nn.Module):
    """LayerNorm applied on the channel dimension of (B, C, H, W) tensors.

    Equivalent to S2D's ``nn.LayerNorm([C, H, W])`` but without
    hard-coding the spatial resolution.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → norm → (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FeatureEnhancement(nn.Module):
    """
    S2D feature densification for LiDAR BEV features.

    Architecture (from S2D_RPN.forward, lines 300-311):
        encoder_1  (stride 2)  →  H/2
        encoder_2  (stride 2)  →  H/4
        3× ConvNeXt blocks with external residuals  →  H/4
        decoder_1  (×2 up)     →  H/2   ─┐
        cat with encoder_1 output        ─┘  →  (2C, H/2)
        decoder_2  (fuse + ×2 up)        →  H
        fusion_dense(·) + fusion_sparse(input)  →  residual output
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        C = channels

        # --- S2D encoder (rpn.py L186-202) ---
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(C, C, 2, 2),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.Conv2d(C, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.Conv2d(C, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # --- S2D ConvNeXt attention blocks (rpn.py L204-225) ---
        # Identical structure; external residuals applied in forward().
        self.convnext_block_1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C),
            _ChannelLayerNorm(C),
            nn.Conv2d(C, C * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(C * 4, C, 1, 1, 0),
        )
        self.convnext_block_2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C),
            _ChannelLayerNorm(C),
            nn.Conv2d(C, C * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(C * 4, C, 1, 1, 0),
        )
        self.convnext_block_3 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C),
            _ChannelLayerNorm(C),
            nn.Conv2d(C, C * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(C * 4, C, 1, 1, 0),
        )

        # --- S2D decoder (rpn.py L228-241) ---
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(C, C, 4, 2, 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(C * 2, C, 3, 1, 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.ConvTranspose2d(C, C, 4, 2, 1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # --- S2D fusion (rpn.py L243-253) ---
        self.fusion_sparse = nn.Sequential(
            nn.Conv2d(C, C, 1, 1, 0),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        self.fusion_dense = nn.Sequential(
            nn.Conv2d(C, C, 1, 1, 0),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

    def forward(self, f_sparse: torch.Tensor) -> torch.Tensor:
        """
        Mirrors S2D_RPN.forward lines 303-311.

        Args:
            f_sparse: (B, C, H, W) sparse LiDAR BEV features (F_l^S)
        Returns:
            f_dense:  (B, C, H, W) enhanced dense features (F_l^D)
        """
        y_1 = self.encoder_1(f_sparse)
        y_2 = self.encoder_2(y_1)

        att = self.convnext_block_1(y_2) + y_2
        att = self.convnext_block_2(att) + att
        att = F.gelu(self.convnext_block_3(att) + att)

        y_3 = torch.cat([self.decoder_1(att), y_1], 1)
        f_dense = self.decoder_2(y_3)

        return self.fusion_dense(f_dense) + self.fusion_sparse(f_sparse)
