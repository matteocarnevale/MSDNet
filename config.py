"""MSDNet configuration for the RADIal dataset.

RADIal sensor geometry:
  radar range : 0–103 m (forward, X axis)
  radar azimuth: ±45° → ±~73 m lateral at max range (we clip to ±50 m, Y axis)
  elevation   : ±~10° (small)

Voxel grid with [0.3, 0.3, 0.2] m voxels:
  X cells = 103/0.3 = 343
  Y cells = 100/0.3 = 333
  Z cells =   8/0.2 =  40

BEV after 3× stride-2 sparse encoder:
  H = Y_cells // 8 = 41   (lateral)
  W = X_cells // 8 = 42   (forward)

Full-scale reconstruction output (4× upsample in XY):
  Z=40, Y=164, X=168
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VoxelConfig:
    # RADIal: [xmin, ymin, zmin, xmax, ymax, zmax]  all in metres
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -50.0, -3.0, 103.0, 50.0, 5.0]
    )
    voxel_size: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2])
    max_points_per_voxel: int = 5
    max_voxels_train: int = 40_000
    max_voxels_eval: int = 60_000


@dataclass
class EncoderConfig:
    lidar_in_features: int = 4   # x, y, z, intensity=1
    radar_in_features: int = 5   # x, y, z, power_norm, v_norm
    vfe_out_channels: int = 64
    sparse_channels: List[int] = field(default_factory=lambda: [64, 32, 64, 128])
    bev_channels: int = 128
    # Doppler BEV feature map channels injected into RGFD and DGFD
    doppler_channels: int = 32
    # Normalisation constant for the Doppler feature (v_bin_centered ∈ [-128,127])
    # Set to 1.0 if your pipeline already normalises velocity to [-1, 1]
    doppler_max: float = 1.0   # dataset.py already normalises v_bin/128 → [-1,1]


@dataclass
class RGFDConfig:
    channels: int = 128
    cbam_reduction: int = 16
    convnext_expansion: int = 4


@dataclass
class DiffusionConfig:
    total_timesteps: int = 1000
    start_timestep: int = 500
    sampling_steps: int = 50
    sampling_interval: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.02
    time_embed_dim: int = 128
    # Set True to backprop through DDIM during student training (high VRAM)
    ddim_backprop_in_training: bool = False


@dataclass
class ReconstructionConfig:
    scales: List[int] = field(default_factory=lambda: [4, 2, 1])
    base_3d_channels: int = 64


@dataclass
class LossConfig:
    rho: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    zeta: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    alpha: float = 10.0
    gamma: float = 20.0
    lambda_recon: float = 1.0
    lambda_rec_distill: float = 0.01
    lambda_diff_distill: float = 5.0
    lambda_diff: float = 10.0


@dataclass
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-3
    teacher_epochs: int = 60
    student_epochs: int = 90
    num_workers: int = 4
    occupancy_threshold: float = 0.5


@dataclass
class MSDNetConfig:
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    rgfd: RGFDConfig = field(default_factory=RGFDConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """(X_cells, Y_cells, Z_cells) of the voxel grid."""
        pc = self.voxel.point_cloud_range
        vs = self.voxel.voxel_size
        return (
            int((pc[3] - pc[0]) / vs[0]),  # X (forward): 343
            int((pc[4] - pc[1]) / vs[1]),  # Y (lateral): 333
            int((pc[5] - pc[2]) / vs[2]),  # Z (up):        40
        )

    @property
    def bev_size(self) -> Tuple[int, int]:
        """
        (H_bev, W_bev) of the dense BEV feature map produced by the sparse encoder.

        After 3× stride-2 downsampling:
          H_bev = Y_cells // 8  (row dimension, lateral)
          W_bev = X_cells // 8  (col dimension, forward)

        For RADIal default: (41, 42).
        """
        gx, gy, _ = self.grid_size
        return gy // 8, gx // 8   # (H=Y//8, W=X//8)

    @property
    def recon_size_full(self) -> Tuple[int, int, int]:
        """
        (Z, H_full, W_full) of the full-scale reconstruction output.

        PointCloudReconstruction lifts BEV → 3D then upsamples 4× in XY:
          Z_full = Z_cells
          H_full = H_bev * 4  (Y direction)
          W_full = W_bev * 4  (X direction)

        For RADIal default: (40, 164, 168).
        """
        _, gz = self.grid_size[2], self.grid_size[2]
        h_bev, w_bev = self.bev_size
        gz = self.grid_size[2]
        return gz, h_bev * 4, w_bev * 4
