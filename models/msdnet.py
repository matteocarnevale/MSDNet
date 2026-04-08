"""MSDNet: Multi-Stage Distillation Net for 4D Radar Super-Resolution.

Two separate models:
    MSDNetTeacher – LiDAR encoder + feature enhancement + reconstruction
    MSDNetStudent – Radar encoder + RGFD + DGFD + shared reconstruction

The teacher is trained first, frozen, then used to supervise the student.
"""

import torch
import torch.nn as nn

from .encoder import VoxelEncoder
from .enhancement import FeatureEnhancement
from .rgfd import RGFD
from .dgfd import DGFD
from .diffusion import DiffusionSchedule
from .reconstruction import PointCloudReconstruction


class MSDNetTeacher(nn.Module):
    """
    Teacher branch: LiDAR point cloud → dense BEV features → point cloud.

    Pipeline:
        LiDAR PC → VoxelEncoder → F_l^S → FeatureEnhancement → F_l^D
        F_l^D → PointCloudReconstruction → multi-scale (occ, offset)
    """

    def __init__(self, cfg):
        super().__init__()
        self.encoder = VoxelEncoder(
            in_features=cfg.encoder.lidar_in_features,
            voxel_cfg=cfg.voxel,
            encoder_cfg=cfg.encoder,
        )
        self.enhancement = FeatureEnhancement(cfg.encoder.bev_channels)
        grid_z = cfg.grid_size[2]
        self.reconstruction = PointCloudReconstruction(
            bev_channels=cfg.encoder.bev_channels,
            base_3d_channels=cfg.reconstruction.base_3d_channels,
            grid_z=grid_z,
            voxel_size=cfg.voxel.voxel_size,
        )

    def forward(self, lidar_points: list, batch_size: int):
        """
        Returns:
            f_dense:    (B, C, H, W) dense teacher BEV features F_l^D
            recon_out:  dict of multi-scale occupancy and offset predictions
        """
        f_sparse = self.encoder(lidar_points, batch_size)
        f_dense = self.enhancement(f_sparse)
        recon_out = self.reconstruction(f_dense)
        return f_dense, recon_out


class MSDNetStudent(nn.Module):
    """
    Student branch: 4D radar → RGFD → DGFD → dense BEV → point cloud.

    During training the frozen teacher provides F_l^D for distillation losses.
    The reconstruction module shares weights with the teacher (set externally).

    Pipeline (training):
        Radar PC → VoxelEncoder → F_r^S
        F_r^S → RGFD → F_r^R                         (+ L_rec_distill with F_l^D)
        F_r^R → NoiseAdapter(m) → DDIM → F_r^D       (+ L_diff_distill with F_l^D)
        F_l^D → forward diffuse → diffusion net       (+ L_diff)
        F_r^D → Reconstruction → (occ, offset)        (+ L_recon)

    Pipeline (inference):
        Radar PC → Encoder → RGFD → DGFD → Reconstruction → point cloud
    """

    def __init__(self, cfg, shared_reconstruction: PointCloudReconstruction = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = VoxelEncoder(
            in_features=cfg.encoder.radar_in_features,
            voxel_cfg=cfg.voxel,
            encoder_cfg=cfg.encoder,
        )
        self.rgfd = RGFD(
            channels=cfg.rgfd.channels,
            reduction=cfg.rgfd.cbam_reduction,
            convnext_expansion=cfg.rgfd.convnext_expansion,
        )
        self.dgfd = DGFD(
            channels=cfg.encoder.bev_channels,
            time_embed_dim=cfg.diffusion.time_embed_dim,
        )

        if shared_reconstruction is not None:
            self.reconstruction = shared_reconstruction
        else:
            grid_z = cfg.grid_size[2]
            self.reconstruction = PointCloudReconstruction(
                bev_channels=cfg.encoder.bev_channels,
                base_3d_channels=cfg.reconstruction.base_3d_channels,
                grid_z=grid_z,
                voxel_size=cfg.voxel.voxel_size,
            )

        dcfg = cfg.diffusion
        self.schedule = DiffusionSchedule(
            dcfg.total_timesteps, dcfg.beta_start, dcfg.beta_end,
        )
        self.start_timestep = dcfg.start_timestep
        self.sampling_steps = dcfg.sampling_steps
        self.sampling_interval = dcfg.sampling_interval

    def forward(self, radar_points: list, batch_size: int,
                f_teacher: torch.Tensor = None, training: bool = True):
        """
        Args:
            radar_points: list of (N_i, F) 4D radar point clouds
            batch_size:   int
            f_teacher:    (B, C, H, W) frozen teacher features F_l^D (training only)
            training:     bool

        Returns (training):
            dict with keys:
                f_recon:     F_r^R   (for L_rec_distill)
                f_denoised:  F_r^D   (for L_diff_distill & reconstruction)
                recon_out:   multi-scale predictions (for L_recon)
                diff_loss_inputs: (eps_pred, eps_gt, t) for L_diff

        Returns (inference):
            dict with keys:
                f_denoised:  F_r^D
                recon_out:   multi-scale predictions
        """
        self.schedule = self.schedule.to(radar_points[0].device)
        f_sparse = self.encoder(radar_points, batch_size)
        f_recon = self.rgfd(f_sparse)

        if training:
            assert f_teacher is not None, "Teacher features required during training"

            # --- Student side: DGFD denoise ---
            f_denoised = self.dgfd.student_forward(
                f_recon, self.start_timestep,
                self.schedule, self.sampling_steps, self.sampling_interval,
            )

            # --- Teacher side: diffusion loss (Eq. 10) ---
            B = f_teacher.shape[0]
            t = torch.randint(0, self.schedule.T, (B,), device=f_teacher.device)
            noise_gt = torch.randn_like(f_teacher)
            f_teacher_noisy = self.schedule.q_sample(f_teacher, t, noise_gt)
            noise_pred = self.dgfd.diffusion_net(f_teacher_noisy, t)

            # --- Reconstruction ---
            recon_out = self.reconstruction(f_denoised)

            return {
                "f_recon": f_recon,
                "f_denoised": f_denoised,
                "recon_out": recon_out,
                "diff_loss_inputs": (noise_pred, noise_gt, t),
            }
        else:
            f_denoised = self.dgfd.student_forward(
                f_recon, self.start_timestep,
                self.schedule, self.sampling_steps, self.sampling_interval,
            )
            recon_out = self.reconstruction(f_denoised)
            return {
                "f_denoised": f_denoised,
                "recon_out": recon_out,
            }

    @torch.no_grad()
    def generate_point_cloud(self, radar_points: list, batch_size: int,
                             threshold: float = 0.5, point_cloud_range=None):
        """Full inference pipeline → list of predicted point clouds."""
        self.schedule = self.schedule.to(radar_points[0].device)
        f_sparse = self.encoder(radar_points, batch_size)
        f_recon = self.rgfd(f_sparse)
        f_denoised = self.dgfd.student_forward(
            f_recon, self.start_timestep,
            self.schedule, self.sampling_steps, self.sampling_interval,
        )
        return self.reconstruction.generate_point_cloud(
            f_denoised, threshold, point_cloud_range,
        )
