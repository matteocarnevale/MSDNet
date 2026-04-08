"""VoxelNet-style encoder: Voxelization → VFE → Sparse 3D backbone → BEV.

Uses spconv for sparse 3D convolutions (Zhou & Tuzel, CVPR 2018).
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False


# ---------------------------------------------------------------------------
# Voxelization (pure-PyTorch fallback; production code may use spconv utils)
# ---------------------------------------------------------------------------

class Voxelizer(nn.Module):
    """Hard voxelization: assigns each point to a voxel by its (x,y,z) index."""

    def __init__(self, voxel_size, point_cloud_range,
                 max_points_per_voxel=5, max_voxels=40000):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.pc_range_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
        self.pc_range_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
        self.max_points = max_points_per_voxel
        self.max_voxels = max_voxels

        grid = ((self.pc_range_max - self.pc_range_min) / self.voxel_size).long()
        self.register_buffer("grid_size", grid)

    @torch.no_grad()
    def forward(self, points_list: list):
        """
        Args:
            points_list: list of (N_i, F) tensors, one per sample.

        Returns:
            voxel_features: (total_voxels, max_points, F)
            voxel_coords:   (total_voxels, 4)  batch_idx, z, y, x
            num_points:     (total_voxels,)
        """
        all_feats, all_coords, all_npts = [], [], []
        for batch_idx, points in enumerate(points_list):
            device = points.device
            vs = self.voxel_size.to(device)
            pc_min = self.pc_range_min.to(device)
            pc_max = self.pc_range_max.to(device)

            mask = ((points[:, :3] >= pc_min) & (points[:, :3] < pc_max)).all(dim=1)
            points = points[mask]

            coords = ((points[:, :3] - pc_min) / vs).long()
            gs = self.grid_size.to(device)
            coords = coords.clamp(min=torch.zeros(3, device=device, dtype=torch.long),
                                  max=gs - 1)

            linear = coords[:, 0] * (gs[1] * gs[2]) + coords[:, 1] * gs[2] + coords[:, 2]

            unique_linear, inverse = torch.unique(linear, return_inverse=True)
            n_voxels = min(unique_linear.shape[0], self.max_voxels)

            num_features = points.shape[1]
            voxel_feats = points.new_zeros(n_voxels, self.max_points, num_features)
            voxel_npts = points.new_zeros(n_voxels, dtype=torch.long)
            voxel_coords = points.new_zeros(n_voxels, 3, dtype=torch.long)

            for i in range(n_voxels):
                pt_mask = inverse == i
                pts = points[pt_mask]
                num_pts_in_voxel = min(pts.shape[0], self.max_points)
                voxel_feats[i, :num_pts_in_voxel] = pts[:num_pts_in_voxel]
                voxel_npts[i] = num_pts_in_voxel
                voxel_coords[i] = coords[pt_mask][0]

            batch_col = torch.full((n_voxels, 1), batch_idx,
                                   dtype=torch.long, device=device)
            # spconv expects (batch, z, y, x)
            zyx = voxel_coords[:, [2, 1, 0]]
            voxel_coords_4d = torch.cat([batch_col, zyx], dim=1)

            all_feats.append(voxel_feats)
            all_coords.append(voxel_coords_4d)
            all_npts.append(voxel_npts)

        return (torch.cat(all_feats, 0),
                torch.cat(all_coords, 0),
                torch.cat(all_npts, 0))


# ---------------------------------------------------------------------------
# Voxel Feature Encoding (VFE) – simple mean-pooling + linear projection
# ---------------------------------------------------------------------------

class VFE(nn.Module):
    def __init__(self, in_features: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, voxel_features, num_points):
        """
        Args:
            voxel_features: (V, max_pts, F)
            num_points:     (V,)
        Returns:
            (V, out_channels)
        """
        mask = torch.arange(voxel_features.shape[1], device=voxel_features.device)
        mask = mask.unsqueeze(0) < num_points.unsqueeze(1)
        voxel_features = voxel_features * mask.unsqueeze(-1).float()
        mean = voxel_features.sum(dim=1) / num_points.clamp(min=1).unsqueeze(-1).float()
        out = self.linear(mean)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Sparse 3D Convolutional Backbone (SparseEnc)
# ---------------------------------------------------------------------------

def _sparse_block(in_ch, out_ch, stride=1):
    """One sparse-conv block: SparseConv/SubMConv → BN → ReLU → SubMConv → BN → ReLU."""
    assert HAS_SPCONV, "spconv is required for the sparse encoder"
    if stride > 1:
        downsample = spconv.SparseConv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
    else:
        downsample = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False)

    return spconv.SparseSequential(
        downsample,
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class SparseEncoder(nn.Module):
    """
    4-block sparse 3D backbone with progressive spatial downsampling.

    Strides: block1=1 (no downsampling), blocks 2-4 = stride 2.
    After blocks 2-4, the spatial resolution is divided by 2 each time:
        (X, Y, Z) → (X, Y, Z) → (X/2, Y/2, Z/2) → (X/4, Y/4, Z/4) → (X/8, Y/8, Z/8)
    """

    def __init__(self, in_channels: int, channels=(16, 32, 64, 128),
                 sparse_shape=None):
        super().__init__()
        assert HAS_SPCONV, "spconv is required"
        self.sparse_shape = sparse_shape  # (Z, Y, X) order for spconv

        self.block1 = _sparse_block(in_channels, channels[0], stride=1)
        self.block2 = _sparse_block(channels[0], channels[1], stride=2)
        self.block3 = _sparse_block(channels[1], channels[2], stride=2)
        self.block4 = _sparse_block(channels[2], channels[3], stride=2)

    def forward(self, voxel_features, voxel_coords, batch_size):
        x = spconv.SparseConvTensor(voxel_features, voxel_coords.int(),
                                    self.sparse_shape, batch_size)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# ---------------------------------------------------------------------------
# Height aggregation → BEV features
# ---------------------------------------------------------------------------

class HeightCompression(nn.Module):
    """Collapse the sparse 3D features along Z to produce a dense BEV map."""

    def __init__(self, in_channels: int, z_dim: int, bev_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * z_dim, bev_channels, 1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, sparse_tensor):
        dense = sparse_tensor.dense()                # (B, C, Z, Y, X)
        B, C, Z, Y, X = dense.shape
        bev = dense.reshape(B, C * Z, Y, X)          # flatten Z into channels
        return self.conv(bev)                         # (B, bev_channels, Y, X)


# ---------------------------------------------------------------------------
# Full encoder pipeline
# ---------------------------------------------------------------------------

class VoxelEncoder(nn.Module):
    """Voxelization → VFE → SparseEnc → BEV features  (Eq. 1 in the paper)."""

    def __init__(self, in_features: int, voxel_cfg, encoder_cfg):
        super().__init__()
        pc_range = voxel_cfg.point_cloud_range
        voxel_size = voxel_cfg.voxel_size
        grid_x = int((pc_range[3] - pc_range[0]) / voxel_size[0])
        grid_y = int((pc_range[4] - pc_range[1]) / voxel_size[1])
        grid_z = int((pc_range[5] - pc_range[2]) / voxel_size[2])

        self.voxelizer = Voxelizer(
            voxel_size, pc_range,
            voxel_cfg.max_points_per_voxel,
            voxel_cfg.max_voxels_train,
        )
        self.vfe = VFE(in_features, encoder_cfg.vfe_out_channels)

        sparse_shape = [grid_z, grid_y, grid_x]  # spconv uses (Z, Y, X)
        channels = encoder_cfg.sparse_channels
        self.sparse_enc = SparseEncoder(
            encoder_cfg.vfe_out_channels, channels, sparse_shape,
        )

        # After 3 stride-2 blocks the spatial dims are divided by 8 in each axis.
        z_reduced = grid_z // 8
        self.height_compress = HeightCompression(
            channels[-1], z_reduced, encoder_cfg.bev_channels,
        )

    def forward(self, points_list: list, batch_size: int):
        voxel_feats, voxel_coords, num_pts = self.voxelizer(points_list)
        voxel_feats = self.vfe(voxel_feats, num_pts)
        sparse_out = self.sparse_enc(voxel_feats, voxel_coords, batch_size)
        bev = self.height_compress(sparse_out)
        return bev
