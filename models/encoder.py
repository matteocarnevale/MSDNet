"""VoxelNet-style encoder: Voxelization → VFE → Sparse 3D backbone → BEV.

Also provides DopplerBEVMap: converts the radar velocity channel into a
structured (B, D, H_bev, W_bev) feature map for downstream conditioning.

Changes vs original:
  - Voxelizer inner loop fully vectorised (O(N log N) sort + O(N) scatter)
    instead of the previous O(N_voxels × N_points) Python loop.
  - DopplerBEVMap added (new).
"""

import math
from typing import List

import torch
import torch.nn as nn
import numpy as np

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False


# ---------------------------------------------------------------------------
# Doppler BEV map
# ---------------------------------------------------------------------------

class DopplerBEVMap(nn.Module):
    """
    Converts a batch of 4D radar point clouds into a structured BEV Doppler
    feature map for conditioning RGFD and DGFD.

    Physics motivation:
      - Radial velocity encodes whether a detection is a moving target or
        static clutter (ego-motion compensated or not).
      - Injecting this signal helps the reconstruction U-Net focus on moving
        objects and helps the diffusion noise adapter set the right noise level
        for dynamic vs static regions.

    Pipeline:
      radar PC (N, 5: x,y,z,intensity,velocity)
        → scatter velocity stats to BEV grid → (3, H, W) raw map
            ch0: mean radial velocity (normalised to [-1, 1])
            ch1: velocity variance   (normalised)
            ch2: log(1 + point_count) per cell (density proxy)
        → small CNN → (D, H, W)

    Output: (B, D, H_bev, W_bev)
    """

    def __init__(
        self,
        pc_range: List[float],
        bev_size:  tuple,         # (H_bev, W_bev) matching encoder BEV output
        out_channels: int = 32,
        v_max: float = 1.0,       # normalisation constant for the velocity channel
                                  # dataset.py stores v_bin_centered/128 → already [-1,1]
                                  # so v_max=1.0 is correct
    ) -> None:
        super().__init__()
        self.pc_range    = pc_range
        self.bev_size    = bev_size   # (H, W) = (Y_dim, X_dim)
        self.v_max       = v_max

        # Normalisation constants (precomputed, no learnable params)
        log_count_max = math.log1p(50.0)   # typical max pts per BEV cell
        self.register_buffer("log_count_norm", torch.tensor(log_count_max))

        # Small 2-layer CNN to encode the 3 raw channels
        D = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, D, 3, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.GELU(),
            nn.Conv2d(D, D, 3, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )

    def _scatter_to_bev(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Scatter velocity statistics from (N, 5) radar PC to (3, H, W) BEV map.

        Fully vectorised with scatter_add — O(N) complexity.
        """
        H, W   = self.bev_size
        device = pts.device

        if pts.shape[0] == 0:
            return torch.zeros(3, H, W, device=device)

        xmin, ymin = self.pc_range[0], self.pc_range[1]
        xmax, ymax = self.pc_range[3], self.pc_range[4]

        # X → column (W dimension), Y → row (H dimension)
        col = ((pts[:, 0] - xmin) / (xmax - xmin) * W).long().clamp(0, W - 1)
        row = ((pts[:, 1] - ymin) / (ymax - ymin) * H).long().clamp(0, H - 1)

        idx = row * W + col   # linear index, (N,)
        N   = H * W
        vel = pts[:, 4]

        ones = torch.ones(pts.shape[0], device=device)
        vel_sum    = torch.zeros(N, device=device).scatter_add(0, idx, vel)
        vel_sq_sum = torch.zeros(N, device=device).scatter_add(0, idx, vel * vel)
        count      = torch.zeros(N, device=device).scatter_add(0, idx, ones)

        count_safe = count.clamp(min=1.0)
        mean_vel   = vel_sum / count_safe
        var_vel    = (vel_sq_sum / count_safe - mean_vel ** 2).clamp(min=0.0)
        log_count  = torch.log1p(count)

        raw = torch.stack([
            mean_vel  / (self.v_max + 1e-8),            # ch0: normalised mean vel
            var_vel   / (self.v_max ** 2 + 1e-8),       # ch1: normalised variance
            log_count / (self.log_count_norm + 1e-8),   # ch2: density proxy
        ], dim=0).reshape(3, H, W)

        return raw

    def forward(self, radar_list: list) -> torch.Tensor:
        """
        Args:
            radar_list: list of B × (N_i, 5) radar PC tensors
        Returns:
            (B, out_channels, H_bev, W_bev)
        """
        raw_maps = [self._scatter_to_bev(pts) for pts in radar_list]
        raw = torch.stack(raw_maps, dim=0)    # (B, 3, H, W)
        return self.encoder(raw)              # (B, D, H, W)


# ---------------------------------------------------------------------------
# Voxelization  (vectorised)
# ---------------------------------------------------------------------------

class Voxelizer(nn.Module):
    """
    Hard voxelization: assigns each point to a voxel, then packs up to
    max_points_per_voxel points per voxel.

    Fully vectorised — avoids the O(N_voxels × N_points) Python loop of the
    original by sorting once and using scatter_add for all aggregations.
    """

    def __init__(self, voxel_size, point_cloud_range,
                 max_points_per_voxel=5,
                 max_voxels_train=40_000, max_voxels_eval=60_000):
        super().__init__()
        self.max_pts          = max_points_per_voxel
        self.max_vox_train    = max_voxels_train
        self.max_vox_eval     = max_voxels_eval

        vs  = torch.tensor(voxel_size, dtype=torch.float32)
        pcm = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
        pcx = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
        gs  = ((pcx - pcm) / vs).long()

        self.register_buffer("voxel_size",   vs)
        self.register_buffer("pc_range_min", pcm)
        self.register_buffer("pc_range_max", pcx)
        self.register_buffer("grid_size",    gs)

    @torch.no_grad()
    def forward(self, points_list: list):
        """
        Returns:
            voxel_features: (V_total, max_pts, F)
            voxel_coords:   (V_total, 4)  [batch, z, y, x]
            num_points:     (V_total,)
        """
        max_vox = self.max_vox_train if self.training else self.max_vox_eval
        all_feats, all_coords, all_npts = [], [], []

        for b_idx, points in enumerate(points_list):
            dev = points.device
            vs  = self.voxel_size.to(dev)
            pcm = self.pc_range_min.to(dev)
            pcx = self.pc_range_max.to(dev)
            gs  = self.grid_size.to(dev)

            # 1. Filter out-of-range points
            mask = ((points[:, :3] >= pcm) & (points[:, :3] < pcx)).all(dim=1)
            pts  = points[mask]

            if pts.shape[0] == 0:
                F = points.shape[1]
                all_feats.append(pts.new_zeros(0, self.max_pts, F))
                all_coords.append(pts.new_zeros(0, 4, dtype=torch.long))
                all_npts.append(pts.new_zeros(0, dtype=torch.long))
                continue

            F = pts.shape[1]

            # 2. Compute voxel (x, y, z) indices for each point
            coords = ((pts[:, :3] - pcm) / vs).long().clamp(
                min=torch.zeros(3, device=dev, dtype=torch.long),
                max=gs - 1,
            )  # (N, 3)  in (X, Y, Z) order

            # 3. Unique voxel detection via linear hash
            linear = coords[:, 0] * (gs[1] * gs[2]) + coords[:, 1] * gs[2] + coords[:, 2]
            unique_lin, inverse = torch.unique(linear, return_inverse=True)
            n_uniq = unique_lin.shape[0]

            # 4. Apply voxel budget
            if n_uniq > max_vox:
                if self.training:
                    sel = torch.randperm(n_uniq, device=dev)[:max_vox]
                else:
                    sel = torch.arange(max_vox, device=dev)

                keep_set = torch.zeros(n_uniq, dtype=torch.bool, device=dev)
                keep_set[sel] = True
                keep_pts = keep_set[inverse]

                pts     = pts[keep_pts]
                coords  = coords[keep_pts]
                inverse = inverse[keep_pts]

                # Remap inverse to 0-based consecutive
                unique_old = torch.unique(inverse)
                remap = torch.zeros(n_uniq, dtype=torch.long, device=dev)
                remap[unique_old] = torch.arange(len(unique_old), device=dev)
                inverse = remap[inverse]
                n_voxels = len(unique_old)
            else:
                n_voxels = n_uniq

            # 5. Vectorised fill: sort by voxel, then assign points to slots
            order = inverse.argsort()            # sort points by voxel index
            sorted_pts = pts[order]              # (N, F)
            sorted_inv = inverse[order]          # (N,)
            sorted_coo = coords[order]           # (N, 3)

            # within-voxel slot index for each (sorted) point
            vox_starts = torch.searchsorted(
                sorted_inv.contiguous(),
                torch.arange(n_voxels, device=dev),
            )  # (n_voxels,)  first index per voxel in sorted array

            within_idx = (torch.arange(len(sorted_inv), device=dev)
                          - vox_starts[sorted_inv])  # (N,)

            valid = within_idx < self.max_pts
            s_pts = sorted_pts[valid]
            s_inv = sorted_inv[valid]
            s_coo = sorted_coo[valid]
            s_wi  = within_idx[valid]

            # 6. Fill output tensors
            voxel_feats  = pts.new_zeros(n_voxels, self.max_pts, F)
            voxel_coords = pts.new_zeros(n_voxels, 3, dtype=torch.long)
            voxel_npts   = pts.new_zeros(n_voxels, dtype=torch.long)

            voxel_feats[s_inv, s_wi] = s_pts
            # Voxel coordinates from first point per voxel
            voxel_coords = sorted_coo[vox_starts.clamp(0, len(sorted_coo) - 1)]

            # Count valid points per voxel
            ones = torch.ones(len(s_inv), dtype=torch.long, device=dev)
            voxel_npts.scatter_add_(0, s_inv, ones)

            # 7. Build (batch, z, y, x) coords for spconv
            b_col = voxel_coords.new_full((n_voxels, 1), b_idx)
            zyx   = voxel_coords[:, [2, 1, 0]]            # X,Y,Z → Z,Y,X
            voxel_coords_4d = torch.cat([b_col, zyx], dim=1)

            all_feats.append(voxel_feats)
            all_coords.append(voxel_coords_4d)
            all_npts.append(voxel_npts)

        return (torch.cat(all_feats,  0),
                torch.cat(all_coords, 0),
                torch.cat(all_npts,   0))


# ---------------------------------------------------------------------------
# VFE  (paper-faithful, Zhou & Tuzel CVPR 2018)
# ---------------------------------------------------------------------------

class VFELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.units   = out_channels // 2
        self.linear  = nn.Linear(in_channels, self.units, bias=False)
        self.norm    = nn.BatchNorm1d(self.units)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        V, T, _ = x.shape
        x_flat = self.relu(self.norm(self.linear(x.view(-1, x.shape[-1]))))
        x_flat = x_flat.view(V, T, self.units) * mask.unsqueeze(-1).float()
        x_max, _ = torch.max(x_flat, dim=1, keepdim=True)
        return torch.cat([x_flat, x_max.expand(-1, T, -1)], dim=-1)


class VFE(nn.Module):
    def __init__(self, in_features: int, out_channels: int):
        super().__init__()
        aug_in = in_features + 3   # original features + centroid offsets (xyz)
        self.vfe1 = VFELayer(aug_in, 32)
        self.vfe2 = VFELayer(32, 64)
        self.final_linear = nn.Linear(64, out_channels, bias=False)
        self.final_norm   = nn.BatchNorm1d(out_channels)
        self.final_relu   = nn.ReLU(inplace=True)

    def forward(self, voxel_features: torch.Tensor,
                num_points: torch.Tensor) -> torch.Tensor:
        V, T, _ = voxel_features.shape
        mask = (torch.arange(T, device=voxel_features.device).unsqueeze(0)
                < num_points.unsqueeze(1))

        masked = voxel_features * mask.unsqueeze(-1).float()
        centroids = masked.sum(dim=1) / num_points.clamp(min=1).float().unsqueeze(-1)
        offsets   = voxel_features[:, :, :3] - centroids.unsqueeze(1).expand(-1, T, -1)[:, :, :3]
        aug       = torch.cat([voxel_features, offsets], dim=-1)

        x = self.vfe1(aug, mask)
        x = self.vfe2(x, mask)
        x = x * mask.unsqueeze(-1).float()
        x_max, _ = torch.max(x, dim=1)
        return self.final_relu(self.final_norm(self.final_linear(x_max)))


# ---------------------------------------------------------------------------
# Sparse 3D backbone
# ---------------------------------------------------------------------------

def _sparse_block(in_ch, out_ch, stride=1):
    assert HAS_SPCONV, "spconv is required for the sparse encoder"
    if stride > 1:
        down = spconv.SparseConv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
    else:
        down = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False)
    return spconv.SparseSequential(
        down,
        nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
    )


class SparseEncoder(nn.Module):
    """4-block sparse 3D backbone: strides [1, 2, 2, 2] → 8× downsampling."""

    def __init__(self, in_channels: int, channels=(64, 32, 64, 128),
                 sparse_shape=None):
        super().__init__()
        assert HAS_SPCONV
        self.sparse_shape = sparse_shape
        self.block1 = _sparse_block(in_channels,  channels[0], stride=1)
        self.block2 = _sparse_block(channels[0],  channels[1], stride=2)
        self.block3 = _sparse_block(channels[1],  channels[2], stride=2)
        self.block4 = _sparse_block(channels[2],  channels[3], stride=2)

    def forward(self, vf, vc, batch_size):
        x = spconv.SparseConvTensor(vf, vc.int(), self.sparse_shape, batch_size)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# ---------------------------------------------------------------------------
# Height compression → BEV
# ---------------------------------------------------------------------------

class HeightCompression(nn.Module):
    def __init__(self, in_channels: int, z_dim: int, bev_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * z_dim, bev_channels, 1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, sparse_tensor):
        dense = sparse_tensor.dense()            # (B, C, Z, Y, X)
        B, C, Z, Y, X = dense.shape
        bev = dense.reshape(B, C * Z, Y, X)      # flatten Z into channels
        return self.conv(bev)                     # (B, bev_channels, Y, X)


# ---------------------------------------------------------------------------
# Full encoder pipeline
# ---------------------------------------------------------------------------

class VoxelEncoder(nn.Module):
    """Voxelization → VFE → SparseEnc → BEV (Eq. 1)."""

    def __init__(self, in_features: int, voxel_cfg, encoder_cfg):
        super().__init__()
        pc  = voxel_cfg.point_cloud_range
        vs  = voxel_cfg.voxel_size
        gx  = int((pc[3] - pc[0]) / vs[0])   # X cells (forward)
        gy  = int((pc[4] - pc[1]) / vs[1])   # Y cells (lateral)
        gz  = int((pc[5] - pc[2]) / vs[2])   # Z cells (up)

        self.voxelizer = Voxelizer(
            vs, pc,
            voxel_cfg.max_points_per_voxel,
            voxel_cfg.max_voxels_train,
            voxel_cfg.max_voxels_eval,
        )
        self.vfe = VFE(in_features, encoder_cfg.vfe_out_channels)

        sparse_shape = [gz, gy, gx]           # spconv convention: (Z, Y, X)
        self.sparse_enc = SparseEncoder(
            encoder_cfg.vfe_out_channels,
            encoder_cfg.sparse_channels,
            sparse_shape,
        )

        z_reduced = gz // 8
        self.height_compress = HeightCompression(
            encoder_cfg.sparse_channels[-1], z_reduced, encoder_cfg.bev_channels,
        )

    def forward(self, points_list: list, batch_size: int):
        """Returns (B, bev_channels, H_bev=Y//8, W_bev=X//8)."""
        vf, vc, npts = self.voxelizer(points_list)
        vf = self.vfe(vf, npts)
        sp = self.sparse_enc(vf, vc, batch_size)
        return self.height_compress(sp)
