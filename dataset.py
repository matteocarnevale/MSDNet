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
                 radar_fov_deg: float = 120.0,
                 verify_files: bool = True):
        super().__init__()
        self.root = root
        self.ground_height = ground_height
        self.radar_fov_deg = radar_fov_deg
        self.point_cloud_range = point_cloud_range or [0, -16, -2, 32, 16, 4]
        self.voxel_size = voxel_size or [0.1, 0.1, 0.15]

        split_file = os.path.join(root, "split", f"{split}.txt")
        with open(split_file, "r") as f:
            all_frame_ids = [line.strip() for line in f if line.strip()]
        
        # Verify file existence if requested (quiet mode)
        if verify_files:
            valid_frame_ids = []
            missing_count = 0
            
            for fid in all_frame_ids:
                lidar_path = os.path.join(root, "lidar", f"{fid}.bin")
                radar_path = os.path.join(root, "radar", f"{fid}.bin")
                
                if os.path.exists(lidar_path) and os.path.exists(radar_path):
                    valid_frame_ids.append(fid)
                else:
                    missing_count += 1
            
            self.frame_ids = valid_frame_ids
            if missing_count > 0:
                print(f"Dataset: {len(self.frame_ids)} valid pairs ({missing_count} missing files)")
        else:
            self.frame_ids = all_frame_ids

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]

        lidar_path = os.path.join(self.root, "lidar", f"{fid}.bin")
        radar_path = os.path.join(self.root, "radar", f"{fid}.bin")
        
        # Check if files exist
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
        if not os.path.exists(radar_path):
            raise FileNotFoundError(f"Radar file not found: {radar_path}")

        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        radar = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 5)

        lidar = self._preprocess_lidar(lidar)
        # No preprocessing for radar (paper doesn't mention any)

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
        """Remove ground points and crop to radar FoV (following paper Section IV-B)."""
        if pc.shape[0] == 0:
            return pc
            
        # Paper method 1: "removing ground points from the LiDAR data"  
        # Use elevation map method (standard in autonomous driving)
        pc = self._ground_removal_elevation_map(pc, grid_size=0.5, height_threshold=0.3)
        
        # Paper method 2: "cropping the LiDAR point cloud to match the Field of View (FOV) of the 4D radar"
        pc = self._crop_to_fov(pc)
        
        # Apply point cloud range cropping (this happens in voxelizer too)
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
        """Build occupancy and offset GT matching EXACT model upsampling logic."""
        pc_range = np.array(self.point_cloud_range)
        base_vs = np.array(self.voxel_size)
        pc_min = pc_range[:3]
        
        gt_occ, gt_offset = {}, {}
        
        # CRITICAL: Follow model's progressive upsampling logic
        # Model starts from BEV size and progressively upsamples
        
        for scale in [4, 2, 1]:
            if scale == 4:
                # Scale 1/4: Model uses BEV dimensions directly
                # BEV is 40x40 after 8x downsample, Z is grid_z//4
                h_dim, w_dim = 40, 40  # BEV size from config
                z_dim = 40 // 4        # grid_z // 4 = 10
                actual_voxel_size = base_vs * 4  # [0.4, 0.4, 0.6]
                
            elif scale == 2:
                # Scale 1/2: 2x upsample from scale 4
                h_dim, w_dim = 40 * 2, 40 * 2  # 80x80
                z_dim = 10 * 2                  # 20
                actual_voxel_size = base_vs * 2  # [0.2, 0.2, 0.3]
                
            else:  # scale == 1
                # Scale 1: 2x upsample from scale 2  
                h_dim, w_dim = 80 * 2, 80 * 2   # 160x160
                z_dim = 20 * 2                   # 40
                actual_voxel_size = base_vs * 1  # [0.1, 0.1, 0.15]
            
# Debug print removed for clean training output
            
            # Create GT tensors matching model output dimensions EXACTLY
            occ = np.zeros((1, z_dim, h_dim, w_dim), dtype=np.float32)
            offset = np.zeros((3, z_dim, h_dim, w_dim), dtype=np.float32)
            
            # Voxelize points with the actual voxel size for this scale
            coords = ((lidar[:, :3] - pc_min) / actual_voxel_size).astype(int)
            coords = np.clip(coords, 0, np.array([w_dim, h_dim, z_dim]) - 1)
            
            for pt_idx in range(len(coords)):
                xi, yi, zi = coords[pt_idx]
                if 0 <= xi < w_dim and 0 <= yi < h_dim and 0 <= zi < z_dim:
                    occ[0, zi, yi, xi] = 1.0
                    center = pc_min + np.array([xi, yi, zi]) * actual_voxel_size + actual_voxel_size / 2
                    offset[:, zi, yi, xi] = lidar[pt_idx, :3] - center

            gt_occ[scale] = torch.from_numpy(occ)
            gt_offset[scale] = torch.from_numpy(offset)

        return gt_occ, gt_offset

    def _ground_removal_elevation_map(self, points, grid_size=0.5, height_threshold=0.3):
        """Standard elevation map ground removal."""
        if points.shape[0] == 0:
            return points
            
        xyz = points[:, :3]
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        
        if x_max - x_min < 0.1 or y_max - y_min < 0.1:
            return points[xyz[:, 2] > self.ground_height]
        
        x_bins = int((x_max - x_min) / grid_size) + 1
        y_bins = int((y_max - y_min) / grid_size) + 1
        
        # Find ground level per cell
        ground_heights = np.full((x_bins, y_bins), np.inf)
        x_indices = np.clip(((xyz[:, 0] - x_min) / grid_size).astype(int), 0, x_bins - 1)
        y_indices = np.clip(((xyz[:, 1] - y_min) / grid_size).astype(int), 0, y_bins - 1)
        
        for i in range(len(xyz)):
            x_idx, y_idx = x_indices[i], y_indices[i]
            ground_heights[x_idx, y_idx] = min(ground_heights[x_idx, y_idx], xyz[i, 2])
        
        # Classify points
        non_ground_mask = np.zeros(len(xyz), dtype=bool)
        for i in range(len(xyz)):
            x_idx, y_idx = x_indices[i], y_indices[i]
            ground_h = ground_heights[x_idx, y_idx]
            
            if ground_h == np.inf:
                non_ground_mask[i] = xyz[i, 2] > self.ground_height
            else:
                non_ground_mask[i] = (xyz[i, 2] - ground_h) > height_threshold
        
        return points[non_ground_mask]


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
