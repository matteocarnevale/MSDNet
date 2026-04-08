"""Dataset class for the View-of-Delft (VoD) dataset.

Provides synchronised LiDAR and 4D radar point cloud pairs for training
and evaluation.  Ground-truth occupancy and offset targets are generated
on-the-fly from the LiDAR point clouds via voxelization.

Preprocessing follows R2LDM (Zheng et al., 2025):
    1. Remove ground points from LiDAR (simple height threshold).
    2. Crop LiDAR points to match the 4D radar FoV.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VoDDataset(Dataset):
    """
    Expected directory layout:
        root/
            lidar/
                000000.bin  (N, 4) float32  — x, y, z, intensity
            radar/
                000000.bin  (N, 5) float32  — x, y, z, intensity, velocity
            split/
                train.txt
                test.txt
    """

    def __init__(self, root: str, split: str = "train",
                 point_cloud_range=None,
                 voxel_size=None,
                 ground_height: float = -1.5,
                 radar_fov_deg: float = 120.0):
        super().__init__()
        self.root = root
        self.ground_height = ground_height
        self.radar_fov_deg = radar_fov_deg
        self.point_cloud_range = point_cloud_range or [0, -16, -2, 32, 16, 4]
        self.voxel_size = voxel_size or [0.1, 0.1, 0.15]

        split_file = os.path.join(root, "split", f"{split}.txt")
        with open(split_file, "r") as f:
            self.frame_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]

        lidar = np.fromfile(
            os.path.join(self.root, "lidar", f"{fid}.bin"), dtype=np.float32
        ).reshape(-1, 4)
        radar = np.fromfile(
            os.path.join(self.root, "radar", f"{fid}.bin"), dtype=np.float32
        ).reshape(-1, 5)

        lidar = self._preprocess_lidar(lidar)
        radar = self._crop_to_range(radar)

        lidar_t = torch.from_numpy(lidar).float()
        radar_t = torch.from_numpy(radar).float()

        gt_occ, gt_offset = self._generate_gt(lidar)

        return {
            "lidar": lidar_t,
            "radar": radar_t,
            "gt_occ": gt_occ,
            "gt_offset": gt_offset,
            "frame_id": fid,
        }

    # ---- preprocessing ----

    def _preprocess_lidar(self, pc: np.ndarray) -> np.ndarray:
        """Remove ground points and crop to radar FoV."""
        pc = pc[pc[:, 2] > self.ground_height]
        pc = self._crop_to_fov(pc)
        pc = self._crop_to_range(pc)
        return pc

    def _crop_to_fov(self, pc: np.ndarray) -> np.ndarray:
        """Keep points within the radar's horizontal field of view."""
        half_fov = np.deg2rad(self.radar_fov_deg / 2)
        angles = np.arctan2(pc[:, 1], pc[:, 0])
        mask = np.abs(angles) <= half_fov
        return pc[mask]

    def _crop_to_range(self, pc: np.ndarray) -> np.ndarray:
        pc_range = self.point_cloud_range
        mask = (
            (pc[:, 0] >= pc_range[0]) & (pc[:, 0] < pc_range[3]) &
            (pc[:, 1] >= pc_range[1]) & (pc[:, 1] < pc_range[4]) &
            (pc[:, 2] >= pc_range[2]) & (pc[:, 2] < pc_range[5])
        )
        return pc[mask]

    # ---- ground-truth voxel targets (multi-scale) ----

    def _generate_gt(self, lidar: np.ndarray):
        """Build occupancy and offset GT at scales {1/4, 1/2, 1}."""
        pc_range = self.point_cloud_range
        base_vs = np.array(self.voxel_size)
        pc_min = np.array(pc_range[:3])

        gt_occ, gt_offset = {}, {}
        for scale in [4, 2, 1]:
            voxel_size = base_vs * scale
            grid = np.ceil((np.array(pc_range[3:]) - pc_min) / voxel_size).astype(int)
            gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])

            occ = np.zeros((1, gz, gy, gx), dtype=np.float32)
            offset = np.zeros((3, gz, gy, gx), dtype=np.float32)

            coords = ((lidar[:, :3] - pc_min) / voxel_size).astype(int)
            coords = np.clip(coords, 0, np.array([gx, gy, gz]) - 1)

            for pt_idx in range(len(coords)):
                xi, yi, zi = coords[pt_idx]
                occ[0, zi, yi, xi] = 1.0
                center = pc_min + np.array([xi, yi, zi]) * voxel_size + voxel_size / 2
                offset[:, zi, yi, xi] = lidar[pt_idx, :3] - center

            gt_occ[scale] = torch.from_numpy(occ)
            gt_offset[scale] = torch.from_numpy(offset)

        return gt_occ, gt_offset


def collate_fn(batch):
    """Custom collation: point clouds stay as lists, GT tensors are stacked."""
    lidar_list = [b["lidar"] for b in batch]
    radar_list = [b["radar"] for b in batch]
    frame_ids = [b["frame_id"] for b in batch]

    gt_occ = {}
    gt_offset = {}
    for s in [4, 2, 1]:
        gt_occ[s] = torch.stack([b["gt_occ"][s] for b in batch])
        gt_offset[s] = torch.stack([b["gt_offset"][s] for b in batch])

    return {
        "lidar": lidar_list,
        "radar": radar_list,
        "gt_occ": gt_occ,
        "gt_offset": gt_offset,
        "frame_ids": frame_ids,
    }
