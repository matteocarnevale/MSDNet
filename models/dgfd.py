"""Diffusion-Guided Feature Distillation (DGFD) – Section III-D.

Components:
    1. NoiseAdapter   – aligns reconstructed-feature noise level to a
                        predefined diffusion timestep m  (Eq. 11-12).
    2. LightweightDiffusionNet – predicts noise ε from noisy features
                                 and timestep  (Fig. 3).
    3. DGFD           – orchestrates the full second-stage distillation.
"""

import torch
import torch.nn as nn

from .modules import BottleNeck, TimestepEmbedding


# ---------------------------------------------------------------------------
# Noise Adapter  (Fig. 3, left)
# ---------------------------------------------------------------------------

class NoiseAdapter(nn.Module):
    """
    Predicts a gating coefficient δ that linearly mixes the RGFD-
    reconstructed features F_r^R with Gaussian noise ε to match the
    noise level of a predefined diffusion timestep m (Eq. 12).

        F_r,m^R = δ · F_r^R  +  (1-δ) · ε
    """

    def __init__(self, channels: int, time_embed_dim: int = 128):
        super().__init__()
        self.bottleneck = BottleNeck(channels, norm_layer="gn")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.time_embed = TimestepEmbedding(time_embed_dim)

        # Project global feature to same dim as time embedding
        self.feat_proj = nn.Linear(channels, time_embed_dim)

        # MLP → 2 logits → softmax to get (δ, 1-δ)
        self.gate_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, 2),
        )

    def forward(self, f_recon: torch.Tensor, timestep_m: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_recon:    (B, C, H, W) reconstructed radar features F_r^R
            timestep_m: (B,) long tensor with the predefined timestep m
            noise:      (B, C, H, W) Gaussian noise ε ~ N(0, I)
        Returns:
            f_noisy:    (B, C, H, W) noise-adapted features F_{r,m}^R
        """
        # Global semantic feature (Eq. 11)
        global_feat = self.gap(self.bottleneck(f_recon)).flatten(1)   # (B, C)
        global_feat = self.feat_proj(global_feat)                     # (B, D)

        # Time embedding
        time_embedding = self.time_embed(timestep_m)                  # (B, D)

        # Gating coefficient δ via softmax over 2 logits
        logits = self.gate_mlp(global_feat + time_embedding)         # (B, 2)
        delta = torch.softmax(logits, dim=-1)[:, 0:1]       # (B, 1)

        delta = delta.unsqueeze(-1).unsqueeze(-1)            # (B, 1, 1, 1)
        return delta * f_recon + (1.0 - delta) * noise


# ---------------------------------------------------------------------------
# Lightweight Diffusion Network  (Fig. 3, right)
# ---------------------------------------------------------------------------

class LightweightDiffusionNet(nn.Module):
    """
    Minimal noise-prediction network: time-embed fusion → 2 BottleNeck
    blocks → 1×1 output head.  Replaces a heavyweight U-Net while
    maintaining noise-regression accuracy (Section III-D, Tab. II-c).
    """

    def __init__(self, channels: int, time_embed_dim: int = 128):
        super().__init__()
        self.time_embed = TimestepEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, channels),
            nn.GELU(),
        )

        self.bottleneck1 = BottleNeck(channels, norm_layer="gn")
        self.bottleneck2 = BottleNeck(channels, norm_layer="gn")

        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x_noisy: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, C, H, W) noisy features
            t:       (B,) integer timesteps
        Returns:
            eps_pred: (B, C, H, W) predicted noise
        """
        # Fuse timestep information via spatial broadcast addition
        time_features = self.time_proj(self.time_embed(t))          # (B, C)
        time_features = time_features.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        features = x_noisy + time_features

        features = self.bottleneck1(features)
        features = self.bottleneck2(features)
        return self.head(features)


# ---------------------------------------------------------------------------
# DGFD module  (combines Noise Adapter + Diffusion Network + schedule)
# ---------------------------------------------------------------------------

class DGFD(nn.Module):
    """
    Full Diffusion-Guided Feature Distillation module.

    During training:
        Teacher side: forward-diffuses F_l^D, predicts noise → L_diff (Eq. 10)
        Student side: noise-adapts F_r^R → DDIM denoise → F_r^D → L_diff_distill (Eq. 14)

    During inference:
        F_r^R → NoiseAdapter(m) → DDIM denoise → F_r^D
    """

    def __init__(self, channels: int, time_embed_dim: int = 128):
        super().__init__()
        self.noise_adapter = NoiseAdapter(channels, time_embed_dim)
        self.diffusion_net = LightweightDiffusionNet(channels, time_embed_dim)

    def teacher_forward(self, f_teacher_noisy: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        """
        Teacher-side training: predict noise added to dense LiDAR features.
        The caller applies forward diffusion; this just runs the network.
        Returns predicted noise for L_diff computation (Eq. 10).
        """
        return self.diffusion_net(f_teacher_noisy, t)

    def student_forward(self, f_recon: torch.Tensor, timestep_m: int,
                        schedule, num_steps: int,
                        interval: int) -> torch.Tensor:
        """
        Student-side: noise-adapt → DDIM denoise → denoised features F_r^D.
        """
        B = f_recon.shape[0]
        device = f_recon.device

        noise = torch.randn_like(f_recon)
        t_m = torch.full((B,), timestep_m, device=device, dtype=torch.long)

        f_noisy = self.noise_adapter(f_recon, t_m, noise)

        f_denoised = schedule.ddim_sample(
            self.diffusion_net, f_noisy,
            start_t=timestep_m, interval=interval, num_steps=num_steps,
        )
        return f_denoised
