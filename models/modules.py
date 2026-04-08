"""Reusable building blocks: CBAM, ConvNeXt, BottleNeck, DeformableConv."""

import math

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# ---------------------------------------------------------------------------
# Channel & Spatial Attention (CBAM) -- Woo et al., ECCV 2018
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(x.mean(dim=(2, 3), keepdim=True))
        mx = self.fc(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = torch.cat([x.mean(dim=1, keepdim=True),
                          x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# ConvNeXt residual block -- Liu et al., CVPR 2022
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(channels * expansion, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2)
        return x + shortcut


# ---------------------------------------------------------------------------
# BottleNeck block (used in noise adapter & lightweight diffusion network)
# ---------------------------------------------------------------------------

class BottleNeck(nn.Module):
    """1x1 -> 3x3 -> 1x1 with residual. Uses GroupNorm by default."""

    def __init__(self, channels: int, reduction: int = 4,
                 norm_layer: str = "gn", num_groups: int = 8):
        super().__init__()
        bottleneck_channels = channels // reduction

        def _norm(c):
            if norm_layer == "gn":
                return nn.GroupNorm(min(num_groups, c), c)
            return nn.BatchNorm2d(c)

        self.conv1 = nn.Conv2d(channels, bottleneck_channels, 1, bias=False)
        self.norm1 = _norm(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.norm2 = _norm(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, 1, bias=False)
        self.norm3 = _norm(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return self.relu(x + shortcut)


# ---------------------------------------------------------------------------
# Deformable convolution wrapper
# ---------------------------------------------------------------------------

class DeformableConvBlock(nn.Module):
    """2D deformable convolution with learned offsets, BN, and GELU."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size, stride=stride, padding=padding,
        )
        self.deform_conv = DeformConv2d(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        return self.act(self.bn(self.deform_conv(x, offset)))


# ---------------------------------------------------------------------------
# Simple Conv-BN-Act helper
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1,
                 act: str = "gelu", norm: str = "bn", transposed: bool = False):
        super().__init__()
        ConvCls = nn.ConvTranspose2d if transposed else nn.Conv2d
        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if transposed:
            kwargs["output_padding"] = 0
        else:
            kwargs["dilation"] = dilation

        self.conv = ConvCls(in_ch, out_ch, **kwargs)

        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        else:
            self.norm = nn.Identity()

        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "none": nn.Identity(),
        }
        self.act = activations.get(act, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (diffusion models)
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """Maps integer timestep -> embedding vector of size `dim`."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal encoding → MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))
