"""MSDNet configuration following the paper's implementation details."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VoxelConfig:
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -16.0, -2.0, 32.0, 16.0, 4.0]
    )
    voxel_size: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.15])
    max_points_per_voxel: int = 5
    max_voxels_train: int = 40000
    max_voxels_eval: int = 60000


@dataclass
class EncoderConfig:
    lidar_in_features: int = 4   # x, y, z, intensity
    radar_in_features: int = 5   # x, y, z, intensity, velocity
    vfe_out_channels: int = 16
    sparse_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    bev_channels: int = 128


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
        pc_range = self.voxel.point_cloud_range
        vs = self.voxel.voxel_size
        return (
            int((pc_range[3] - pc_range[0]) / vs[0]),  # X = 320
            int((pc_range[4] - pc_range[1]) / vs[1]),  # Y = 320
            int((pc_range[5] - pc_range[2]) / vs[2]),  # Z = 40
        )

    @property
    def bev_size(self) -> Tuple[int, int]:
        gx, gy, _ = self.grid_size
        return gx // 8, gy // 8  # 40x40 after sparse encoder 8x downsampling
