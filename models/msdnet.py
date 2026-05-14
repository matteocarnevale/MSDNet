"""MSDNet: Multi-Stage Distillation Network for 4D Radar Super-Resolution.

Two models:
    MSDNetTeacher – LiDAR encoder + feature enhancement + reconstruction
    MSDNetStudent – Radar encoder + DopplerBEVMap + RGFD + DGFD + reconstruction

Teacher is trained first (LiDAR only), frozen, then used to supervise the student.
The reconstruction head is shared between teacher and student.

Doppler conditioning (new in student):
    DopplerBEVMap creates a (B, D, H, W) velocity-based BEV map from the
    radar PC's velocity channel. This map is injected into:
      - RGFD:         helps feature reconstruction focus on dynamic targets
      - DGFD/NoiseAdapter: modulates the noise mixing gate δ by scene motion
"""

from typing import Optional

import torch
import torch.nn as nn

from .encoder import VoxelEncoder, DopplerBEVMap
from .enhancement import FeatureEnhancement
from .rgfd import RGFD
from .dgfd import DGFD
from .diffusion import DiffusionSchedule
from .reconstruction import PointCloudReconstruction


class MSDNetTeacher(nn.Module):
    """
    Teacher branch: LiDAR PC → BEV features → reconstruction.

    No Doppler conditioning (LiDAR has no velocity channel).
    """

    def __init__(self, cfg):
        super().__init__()
        self.encoder = VoxelEncoder(
            in_features=cfg.encoder.lidar_in_features,
            voxel_cfg=cfg.voxel,
            encoder_cfg=cfg.encoder,
        )
        self.enhancement = FeatureEnhancement(cfg.encoder.bev_channels)
        gz = cfg.grid_size[2]
        self.reconstruction = PointCloudReconstruction(
            bev_channels=cfg.encoder.bev_channels,
            base_3d_channels=cfg.reconstruction.base_3d_channels,
            grid_z=gz,
            voxel_size=cfg.voxel.voxel_size,
        )

    def forward(self, lidar_points: list, batch_size: int):
        """
        Returns:
            f_dense:   (B, C, H, W)  enhanced teacher BEV features F_l^D
            recon_out: dict of multi-scale occupancy and offset predictions
        """
        f_sparse = self.encoder(lidar_points, batch_size)
        f_dense  = self.enhancement(f_sparse)
        recon_out = self.reconstruction(f_dense)
        return f_dense, recon_out


class MSDNetStudent(nn.Module):
    """
    Student branch: 4D radar → Doppler BEV → RGFD → DGFD → reconstruction.

    During training the frozen teacher provides F_l^D for distillation losses.
    The reconstruction module shares weights with the teacher.

    Pipeline (training):
        Radar PC → DopplerBEVMap → (B, D, H, W) doppler_bev
        Radar PC → VoxelEncoder  → F_r^S
        F_r^S + doppler_bev → RGFD → F_r^R          (+ L_rec_distill with F_l^D)
        F_r^R + doppler_bev → DGFD/NoiseAdapter → DDIM → F_r^D
                                                         (+ L_diff_distill with F_l^D)
        F_l^D → forward diffuse → LightweightDiffusionNet   (+ L_diff)
        F_r^D → Reconstruction → (occ, offset)               (+ L_recon)

    Pipeline (inference):
        Radar PC → DopplerBEVMap + Encoder → RGFD → DGFD → Reconstruction → PC
    """

    def __init__(self, cfg, shared_reconstruction: Optional[PointCloudReconstruction] = None):
        super().__init__()
        self.cfg = cfg
        D = cfg.encoder.doppler_channels    # Doppler BEV channels

        # ── Doppler BEV map ──────────────────────────────────────────────
        bev_h, bev_w = cfg.bev_size
        self.doppler_bev_map = DopplerBEVMap(
            pc_range=cfg.voxel.point_cloud_range,
            bev_size=(bev_h, bev_w),
            out_channels=D,
            v_max=cfg.encoder.doppler_max,  # 1.0: dataset already normalises v_bin/128
        )

        # ── Radar encoder ────────────────────────────────────────────────
        self.encoder = VoxelEncoder(
            in_features=cfg.encoder.radar_in_features,
            voxel_cfg=cfg.voxel,
            encoder_cfg=cfg.encoder,
        )

        # ── RGFD with Doppler injection ──────────────────────────────────
        self.rgfd = RGFD(
            channels=cfg.rgfd.channels,
            reduction=cfg.rgfd.cbam_reduction,
            convnext_expansion=cfg.rgfd.convnext_expansion,
            doppler_channels=D,
        )

        # ── DGFD with Doppler conditioning ───────────────────────────────
        self.dgfd = DGFD(
            channels=cfg.encoder.bev_channels,
            time_embed_dim=cfg.diffusion.time_embed_dim,
            doppler_channels=D,
        )

        # ── Reconstruction (shared with teacher) ─────────────────────────
        if shared_reconstruction is not None:
            self.reconstruction = shared_reconstruction
        else:
            gz = cfg.grid_size[2]
            self.reconstruction = PointCloudReconstruction(
                bev_channels=cfg.encoder.bev_channels,
                base_3d_channels=cfg.reconstruction.base_3d_channels,
                grid_z=gz,
                voxel_size=cfg.voxel.voxel_size,
            )

        dcfg = cfg.diffusion
        self.schedule         = DiffusionSchedule(dcfg.total_timesteps,
                                                  dcfg.beta_start, dcfg.beta_end)
        self.start_timestep   = dcfg.start_timestep
        self.sampling_steps   = dcfg.sampling_steps
        self.sampling_interval = dcfg.sampling_interval

    def forward(self, radar_points: list, batch_size: int,
                f_teacher: Optional[torch.Tensor] = None,
                training: bool = True) -> dict:
        """
        Args:
            radar_points: list of (N_i, 5) radar PCs
            batch_size:   B
            f_teacher:    (B, C, H, W) frozen teacher features (training only)
            training:     bool
        """
        device = radar_points[0].device
        self.schedule = self.schedule.to(device)

        # Doppler BEV conditioning
        doppler_bev = self.doppler_bev_map(radar_points)   # (B, D, H, W)

        # Radar sparse features
        f_sparse = self.encoder(radar_points, batch_size)  # (B, C, H, W)

        # RGFD: reconstruct dense BEV from sparse radar features
        f_recon = self.rgfd(f_sparse, doppler_bev=doppler_bev)  # (B, C, H, W)

        ddim_grad = training and self.cfg.diffusion.ddim_backprop_in_training

        if training:
            assert f_teacher is not None, "f_teacher required during training"

            # Student DDIM denoising (Eq. 13)
            f_denoised = self.dgfd.student_forward(
                f_recon, self.start_timestep,
                self.schedule, self.sampling_steps, self.sampling_interval,
                ddim_requires_grad=ddim_grad,
                doppler_bev=doppler_bev,
            )

            # Teacher-side diffusion loss (Eq. 10): predict noise on teacher features
            B = f_teacher.shape[0]
            t = torch.randint(0, self.schedule.T, (B,), device=device)
            noise_gt = torch.randn_like(f_teacher)
            f_teacher_noisy = self.schedule.q_sample(f_teacher, t, noise_gt)
            noise_pred = self.dgfd.diffusion_net(f_teacher_noisy, t)

            recon_out = self.reconstruction(f_denoised)

            return {
                "f_recon":          f_recon,
                "f_denoised":       f_denoised,
                "recon_out":        recon_out,
                "diff_loss_inputs": (noise_pred, noise_gt, t),
            }
        else:
            f_denoised = self.dgfd.student_forward(
                f_recon, self.start_timestep,
                self.schedule, self.sampling_steps, self.sampling_interval,
                ddim_requires_grad=False,
                doppler_bev=doppler_bev,
            )
            return {
                "f_denoised": f_denoised,
                "recon_out":  self.reconstruction(f_denoised),
            }

    @torch.no_grad()
    def generate_point_cloud(self, radar_points: list, batch_size: int,
                             threshold: float = 0.5,
                             point_cloud_range=None) -> list:
        """Inference: radar PC → predicted point cloud list."""
        device = radar_points[0].device
        self.schedule = self.schedule.to(device)

        doppler_bev = self.doppler_bev_map(radar_points)
        f_sparse    = self.encoder(radar_points, batch_size)
        f_recon     = self.rgfd(f_sparse, doppler_bev=doppler_bev)
        f_denoised  = self.dgfd.student_forward(
            f_recon, self.start_timestep,
            self.schedule, self.sampling_steps, self.sampling_interval,
            ddim_requires_grad=False,
            doppler_bev=doppler_bev,
        )
        return self.reconstruction.generate_point_cloud(
            f_denoised, threshold, point_cloud_range
        )
