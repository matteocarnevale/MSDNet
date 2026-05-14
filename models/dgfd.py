"""Diffusion-Guided Feature Distillation (DGFD) – Section III-D.

Components:
    NoiseAdapter  – aligns RGFD features to a predefined diffusion timestep m.
    LightweightDiffusionNet – predicts noise ε from noisy features and timestep.
    DGFD          – orchestrates the full second-stage distillation.

Doppler conditioning (new):
    NoiseAdapter accepts an optional doppler_bev (B, D, H, W) tensor.
    A global Doppler descriptor (AdaptiveAvgPool → linear) is added to the
    time embedding before computing the mixing gate δ.

    Physical motivation: the optimal mixing ratio between RGFD features and
    noise (gate δ) should depend on whether the region contains moving objects
    (high Doppler → harder to reconstruct → inject more noise) or static
    clutter (low Doppler → easier → keep more of the RGFD signal).
"""

from typing import Optional

import torch
import torch.nn as nn

from .modules import BottleNeck, TimestepEmbedding


# ---------------------------------------------------------------------------
# Noise Adapter  (Fig. 3, left)  + Doppler conditioning
# ---------------------------------------------------------------------------

class NoiseAdapter(nn.Module):
    """
    Predicts gating coefficient δ that mixes RGFD features with Gaussian noise
    to match noise level of predefined diffusion timestep m (Eq. 12):

        F_{r,m}^R = δ · F_r^R  +  (1-δ) · ε

    Optional Doppler conditioning: global Doppler pooled feature adds to
    the time embedding so δ is modulated by scene motion statistics.
    """

    def __init__(self, channels: int, time_embed_dim: int = 128,
                 doppler_channels: int = 0) -> None:
        super().__init__()
        self.bottleneck = BottleNeck(channels, norm_layer="gn")
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.time_embed = TimestepEmbedding(time_embed_dim)
        self.feat_proj  = nn.Linear(channels, time_embed_dim)

        self.has_doppler = doppler_channels > 0
        if self.has_doppler:
            self.doppler_pool = nn.AdaptiveAvgPool2d(1)
            self.doppler_proj = nn.Sequential(
                nn.Linear(doppler_channels, time_embed_dim),
                nn.GELU(),
            )

        self.gate_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, 2),
        )

    def forward(self, f_recon: torch.Tensor, timestep_m: torch.Tensor,
                noise: torch.Tensor,
                doppler_bev: Optional[torch.Tensor] = None) -> torch.Tensor:
        global_feat   = self.gap(self.bottleneck(f_recon)).flatten(1)
        global_feat   = self.feat_proj(global_feat)
        time_emb      = self.time_embed(timestep_m)

        context = global_feat + time_emb

        if self.has_doppler and doppler_bev is not None:
            d_global = self.doppler_pool(doppler_bev).flatten(1)  # (B, D)
            d_emb    = self.doppler_proj(d_global)                 # (B, T)
            context  = context + d_emb

        logits = self.gate_mlp(context)
        delta  = torch.softmax(logits, dim=-1)[:, :1]             # (B, 1)
        delta  = delta.unsqueeze(-1).unsqueeze(-1)                 # (B, 1, 1, 1)

        return delta * f_recon + (1.0 - delta) * noise


# ---------------------------------------------------------------------------
# Lightweight Diffusion Network  (Fig. 3, right)
# ---------------------------------------------------------------------------

class LightweightDiffusionNet(nn.Module):
    """Minimal noise-prediction U-Net: time-embed fusion → 2 BottleNeck → head."""

    def __init__(self, channels: int, time_embed_dim: int = 128) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(time_embed_dim)
        self.time_proj  = nn.Sequential(
            nn.Linear(time_embed_dim, channels),
            nn.GELU(),
        )
        self.b1 = BottleNeck(channels, norm_layer="gn")
        self.b2 = BottleNeck(channels, norm_layer="gn")
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x_noisy: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        t_feat = self.time_proj(self.time_embed(t)).unsqueeze(-1).unsqueeze(-1)
        x = self.b1(x_noisy + t_feat)
        x = self.b2(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# DGFD
# ---------------------------------------------------------------------------

class DGFD(nn.Module):
    """
    Full Diffusion-Guided Feature Distillation module.

    Training:
        Teacher side: forward-diffuse F_l^D → predict noise → L_diff (Eq. 10)
        Student side: noise-adapt F_r^R → DDIM denoise → F_r^D → L_diff_distill

    Inference:
        F_r^R → NoiseAdapter(m) → DDIM denoise → F_r^D
    """

    def __init__(self, channels: int, time_embed_dim: int = 128,
                 doppler_channels: int = 0) -> None:
        super().__init__()
        self.noise_adapter  = NoiseAdapter(channels, time_embed_dim, doppler_channels)
        self.diffusion_net  = LightweightDiffusionNet(channels, time_embed_dim)

    def student_forward(
        self,
        f_recon: torch.Tensor,
        timestep_m: int,
        schedule,
        num_steps: int,
        interval: int,
        ddim_requires_grad: bool = False,
        doppler_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B      = f_recon.shape[0]
        device = f_recon.device
        noise  = torch.randn_like(f_recon)
        t_m    = torch.full((B,), timestep_m, device=device, dtype=torch.long)

        f_noisy = self.noise_adapter(f_recon, t_m, noise, doppler_bev=doppler_bev)

        return schedule.ddim_sample(
            self.diffusion_net, f_noisy,
            start_t=timestep_m, interval=interval, num_steps=num_steps,
            enable_grad=ddim_requires_grad,
        )
