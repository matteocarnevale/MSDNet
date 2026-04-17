"""Dataset class for the View-of-Delft (VoD) dataset.

Provides synchronised LiDAR and 4D radar point cloud pairs for training
and evaluation.  Ground-truth occupancy and offset targets are generated
on-the-fly from the LiDAR point clouds via voxelization.

Preprocessing follows R2LDM (Zheng et al., 2025):
    1. Remove ground points from LiDAR (simple height threshold).
    2. Crop LiDAR points to match the 4D radar FoV.

Optional ``vod_sequence_filter="4drvo_net"``: MSDNet paper IV-A VoD split
(sequences 03, 04, 22 in test). Frame ids should look like ``delft_03_...``.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


_VOD_4DRVO_TEST_SEQS = frozenset({"03", "04", "22"})


def _grid_dims_from_range(point_cloud_range: List[float], voxel_size: List[float]) -> Tuple[int, int, int]:
    """Stessa convenzione di ``MSDNetConfig.grid_size``: celle lungo x, y, z nel volume VoD."""
    pc = point_cloud_range
    vs = voxel_size
    gx = int((pc[3] - pc[0]) / vs[0])
    gy = int((pc[4] - pc[1]) / vs[1])
    gz = int((pc[5] - pc[2]) / vs[2])
    return gx, gy, gz


def decoder_volume_zyx(bev_h: int, bev_w: int, grid_z: int) -> Dict[int, Tuple[int, int, int]]:
    """
    Dimensioni (Z, Y, X) dei tensori ``occ_*`` prodotti da ``PointCloudReconstruction``
    (lift z4=grid_z//4 da BEV, poi due ConvTranspose3d stride 2).
    """
    z4 = grid_z // 4
    return {
        4: (z4, bev_h, bev_w),
        2: (z4 * 2, bev_h * 2, bev_w * 2),
        1: (grid_z, bev_h * 4, bev_w * 4),
    }


def _world_steps_xyz(
    tensor_nz: int, tensor_ny: int, tensor_nx: int, point_cloud_range: List[float]
) -> Tuple[float, float, float, np.ndarray]:
    """
    Passi metrici lungo x, y, z per una griglia ``occ`` di forma (Z,Y,X)=(tensor_nz,tensor_ny,tensor_nx).

    Allineato a ``PointCloudReconstruction.generate_point_cloud``:
    ``vx = extent_x / n_x``, ``vy = extent_y / n_y``, ``vz = extent_z / n_z``.
    """
    pc_min = np.asarray(point_cloud_range[:3], dtype=np.float64)
    pc_max = np.asarray(point_cloud_range[3:], dtype=np.float64)
    ext = pc_max - pc_min
    vx = float(ext[0] / tensor_nx)
    vy = float(ext[1] / tensor_ny)
    vz = float(ext[2] / tensor_nz)
    return vx, vy, vz, pc_min


def remove_ground_elevation_map(
    points: np.ndarray,
    ground_height: float = -1.5,
    grid_size: float = 0.5,
    height_threshold: float = 0.3,
) -> np.ndarray:
    """
    Rimuove il suolo con mappa di elevazione su griglia BEV (stesso algoritmo di ``VoDDataset``).

    Per ogni cella (x, y) tiene l'elevazione minima come stima del terreno; un punto è
    tenuto se ``z - z_ground > height_threshold`` (o fallback ``z > ground_height``).

    Args:
        points: (N, 4+) float — usano solo le colonne ``x,y,z``.
        ground_height: soglia z assoluta se la cella è vuota o la scena è degenere.
        grid_size: passo della griglia BEV in metri.
        height_threshold: altezza minima sopra il ground locale per tenere il punto.
    """
    if points.shape[0] == 0:
        return points

    xyz = points[:, :3]
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()

    if x_max - x_min < 0.1 or y_max - y_min < 0.1:
        return points[xyz[:, 2] > ground_height]

    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1

    ground_heights = np.full((x_bins, y_bins), np.inf)
    x_indices = np.clip(((xyz[:, 0] - x_min) / grid_size).astype(int), 0, x_bins - 1)
    y_indices = np.clip(((xyz[:, 1] - y_min) / grid_size).astype(int), 0, y_bins - 1)

    for i in range(len(xyz)):
        x_idx, y_idx = x_indices[i], y_indices[i]
        ground_heights[x_idx, y_idx] = min(ground_heights[x_idx, y_idx], xyz[i, 2])

    non_ground_mask = np.zeros(len(xyz), dtype=bool)
    for i in range(len(xyz)):
        x_idx, y_idx = x_indices[i], y_indices[i]
        ground_h = ground_heights[x_idx, y_idx]

        if ground_h == np.inf:
            non_ground_mask[i] = xyz[i, 2] > ground_height
        else:
            non_ground_mask[i] = (xyz[i, 2] - ground_h) > height_threshold

    return points[non_ground_mask]


def extract_vod_sequence_id(frame_id: str) -> Optional[str]:
    """Two-digit sequence id from a VoD-style frame id, or None."""
    fid = frame_id.replace("\\", "/")
    m = re.search(r"delft_(\d+)_", fid, flags=re.I)
    if m:
        return f"{int(m.group(1)):02d}"
    m = re.match(r"^(\d{2})_\d+", fid)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{2})\d{6,}$", fid)
    if m:
        return m.group(1)
    return None


def _filter_frame_ids_4drvo_net(frame_ids: List[str], split: str) -> List[str]:
    out = []
    skipped = 0
    for fid in frame_ids:
        seq = extract_vod_sequence_id(fid)
        if seq is None:
            skipped += 1
            continue
        in_test = seq in _VOD_4DRVO_TEST_SEQS
        if split == "train" and not in_test:
            out.append(fid)
        elif split == "test" and in_test:
            out.append(fid)
        elif split not in ("train", "test"):
            if not in_test:
                out.append(fid)
    if skipped:
        print(
            f"Dataset 4drvo_net: skipped {skipped} ids (unparsable sequence; "
            "expected e.g. delft_03_...)"
        )
    return out


class VoDDataset(Dataset):
    """
    Layout atteso (MSDNet):
        root/lidar/*.bin (N,4) float32, root/radar/*.bin (N,5), root/split/*.txt

    **Pipeline LiDAR in ``__getitem__`` (ordine fisso):**
        1. ``remove_ground_elevation_map`` — rimozione suolo (mappa di elevazione)
        2. ``_crop_to_fov`` — FoV orizzontale radar
        3. ``_crop_to_range`` — ``point_cloud_range``
        4. ``_generate_gt`` — voxel GT allineati al decoder

    Anteprima **senza** questi passi: ``python vod_pipeline.py preview ...``.
    Schema testuale: ``python vod_pipeline.py pipeline``.
    """

    def __init__(self, root: str, split: str = "train",
                 point_cloud_range=None,
                 voxel_size=None,
                 ground_height: float = -1.5,
                 radar_fov_deg: float = 120.0,
                 verify_files: bool = True,
                 vod_sequence_filter: Optional[str] = None):
        super().__init__()
        self.root = root
        self.ground_height = ground_height
        self.radar_fov_deg = radar_fov_deg
        self.point_cloud_range = point_cloud_range or [0, -16, -2, 32, 16, 4]
        self.voxel_size = voxel_size or [0.1, 0.1, 0.15]
        self.vod_sequence_filter = vod_sequence_filter

        split_file = os.path.join(root, "split", f"{split}.txt")
        with open(split_file, "r") as f:
            all_frame_ids = [line.strip() for line in f if line.strip()]

        if vod_sequence_filter == "4drvo_net":
            n0 = len(all_frame_ids)
            all_frame_ids = _filter_frame_ids_4drvo_net(all_frame_ids, split)
            print(
                f"Dataset 4drvo_net split={split!r}: {n0} -> {len(all_frame_ids)} frames"
            )
        elif vod_sequence_filter not in (None, ""):
            raise ValueError(
                f"Unknown vod_sequence_filter={vod_sequence_filter!r}; "
                "use None or '4drvo_net'"
            )

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
        """
        Occupancy e offset alle stesse dimensioni (Z,Y,X) di ``PointCloudReconstruction``.

        I passi metrici per scala ricoprono l'intero ``point_cloud_range`` (nessun 16 m
        artificiale su 32 m): ``vx = extent_x / n_x`` con ``n_x`` = larghezza tensor X, ecc.
        """
        gx, gy, gz = _grid_dims_from_range(self.point_cloud_range, self.voxel_size)
        bev_h, bev_w = gx // 8, gy // 8
        volumes = decoder_volume_zyx(bev_h, bev_w, gz)

        gt_occ, gt_offset = {}, {}
        pc_range = self.point_cloud_range

        for scale in (4, 2, 1):
            nz, ny, nx = volumes[scale]
            vx, vy, vz, pc_min = _world_steps_xyz(nz, ny, nx, pc_range)

            occ = np.zeros((1, nz, ny, nx), dtype=np.float32)
            offset = np.zeros((3, nz, ny, nx), dtype=np.float32)

            if lidar.shape[0] == 0:
                gt_occ[scale] = torch.from_numpy(occ)
                gt_offset[scale] = torch.from_numpy(offset)
                continue

            ix = np.floor((lidar[:, 0] - pc_min[0]) / vx).astype(np.int64)
            iy = np.floor((lidar[:, 1] - pc_min[1]) / vy).astype(np.int64)
            iz = np.floor((lidar[:, 2] - pc_min[2]) / vz).astype(np.int64)
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            iz = np.clip(iz, 0, nz - 1)

            for i in range(lidar.shape[0]):
                xi, yi, zi = int(ix[i]), int(iy[i]), int(iz[i])
                occ[0, zi, yi, xi] = 1.0
                c0 = float(pc_min[0] + (xi + 0.5) * vx)
                c1 = float(pc_min[1] + (yi + 0.5) * vy)
                c2 = float(pc_min[2] + (zi + 0.5) * vz)
                center = np.array([c0, c1, c2], dtype=np.float32)
                offset[:, zi, yi, xi] = lidar[i, :3].astype(np.float32) - center

            gt_occ[scale] = torch.from_numpy(occ)
            gt_offset[scale] = torch.from_numpy(offset)

        return gt_occ, gt_offset

    def _ground_removal_elevation_map(self, points, grid_size=0.5, height_threshold=0.3):
        """Delega a :func:`remove_ground_elevation_map` con ``ground_height`` del dataset."""
        return remove_ground_elevation_map(
            points,
            ground_height=self.ground_height,
            grid_size=grid_size,
            height_threshold=height_threshold,
        )


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
