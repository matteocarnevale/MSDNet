"""Progressive Point Cloud Reconstruction – Section III-A (Eqs. 2-4).

Given dense BEV features, generates multi-scale 3D voxel predictions
(occupancy + offset) at scales s ∈ {1/4, 1/2, 1}.
Shared between teacher and student branches.
"""

import torch
import torch.nn as nn


class VoxelHead(nn.Module):
    """Predicts occupancy and per-voxel offsets at a single scale."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.occ_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, 1),
        )
        self.off_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 3, 1),
        )

    def forward(self, g: torch.Tensor, voxel_half_size: float):
        """
        Args:
            g:               (B, C, Z, Y, X) 3D voxel features
            voxel_half_size: L^(s) / 2 for offset clamping
        Returns:
            occ:    (B, 1, Z, Y, X) occupancy logits (pre-sigmoid)
            offset: (B, 3, Z, Y, X) predicted offsets in [-L/2, L/2]
        """
        occ = self.occ_head(g)
        offset = torch.tanh(self.off_head(g)) * voxel_half_size
        return occ, offset


class PointCloudReconstruction(nn.Module):
    """
    BEV → multi-scale 3D voxel features → occupancy + offset predictions.

    The BEV features are first expanded to a coarse 3D volume (scale 1/4)
    and then progressively upsampled to 1/2 and full resolution via 3D
    transposed convolutions.
    """

    def __init__(self, bev_channels: int = 128,
                 base_3d_channels: int = 64,
                 grid_z: int = 40,
                 voxel_size=(0.1, 0.1, 0.15)):
        super().__init__()
        self.voxel_size = voxel_size
        channels_3d = base_3d_channels
        z_quarter = grid_z // 4  # Z at scale 1/4

        # BEV → 3D lift: predict z4 height slices per BEV cell
        self.bev_to_3d = nn.Sequential(
            nn.Conv2d(bev_channels, channels_3d * z_quarter, 1, bias=False),
            nn.BatchNorm2d(channels_3d * z_quarter),
            nn.ReLU(inplace=True),
        )
        self.z4 = z_quarter
        self.C = channels_3d

        # Scale 1/4 → 1/2 (upsample 2× in all three spatial dims)
        self.up_half = nn.Sequential(
            nn.ConvTranspose3d(channels_3d, channels_3d // 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(channels_3d // 2),
            nn.ReLU(inplace=True),
        )
        # Scale 1/2 → 1 (upsample 2× again)
        self.up_full = nn.Sequential(
            nn.ConvTranspose3d(channels_3d // 2, channels_3d // 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(channels_3d // 4),
            nn.ReLU(inplace=True),
        )

        self.head_quarter = VoxelHead(channels_3d)
        self.head_half = VoxelHead(channels_3d // 2)
        self.head_full = VoxelHead(channels_3d // 4)

    def forward(self, bev_features: torch.Tensor):
        """
        Args:
            bev_features: (B, C_bev, H_bev, W_bev) dense BEV features
        Returns:
            dict with keys for each scale s in {4, 2, 1}:
                'occ_s':    (B, 1, Z_s, Y_s, X_s) occupancy logits
                'offset_s': (B, 3, Z_s, Y_s, X_s) offset predictions
                'feat_s':   (B, C_s, Z_s, Y_s, X_s) 3D features (for loss computation)
        """
        B, _, H, W = bev_features.shape
        voxel_x, voxel_y, voxel_z = self.voxel_size

        # Lift to 3D at 1/4 scale
        f3d = self.bev_to_3d(bev_features)                 # (B, C*z4, H, W)
        f3d = f3d.view(B, self.C, self.z4, H, W)           # (B, C, z4, H, W)

        occ4, off4 = self.head_quarter(f3d, voxel_x * 4 / 2)

        # Upsample to 1/2 scale
        f_half = self.up_half(f3d)                          # (B, C/2, Z/2, H*2, W*2)
        occ2, off2 = self.head_half(f_half, voxel_x * 2 / 2)

        # Upsample to full scale
        f_full = self.up_full(f_half)                       # (B, C/4, Z, H*4, W*4)
        occ1, off1 = self.head_full(f_full, voxel_x / 2)

        return {
            "occ_4": occ4, "offset_4": off4,
            "occ_2": occ2, "offset_2": off2,
            "occ_1": occ1, "offset_1": off1,
        }

    @torch.no_grad()
    def generate_point_cloud(self, bev_features: torch.Tensor,
                             threshold: float = 0.5,
                             point_cloud_range=None):
        """
        Inference-time point cloud generation at full resolution (s=1).

        Returns:
            list of (N_i, 3) tensors — one predicted point cloud per sample.
        """
        preds = self.forward(bev_features)
        occ = torch.sigmoid(preds["occ_1"])                 # (B,1,Z,Y,X)
        offset = preds["offset_1"]                          # (B,3,Z,Y,X)

        B, _, Z, Y, X = occ.shape
        if point_cloud_range is not None:
            pcr = point_cloud_range
            voxel_x = (pcr[3] - pcr[0]) / X
            voxel_y = (pcr[4] - pcr[1]) / Y
            voxel_z = (pcr[5] - pcr[2]) / Z
            pc_min = torch.tensor(pcr[:3], device=occ.device)
        else:
            voxel_x, voxel_y, voxel_z = self.voxel_size
            pc_min = torch.zeros(3, device=occ.device)

        point_clouds = []
        for batch_idx in range(B):
            mask = occ[batch_idx, 0] > threshold                     # (Z, Y, X)
            indices = mask.nonzero(as_tuple=False).float()    # (N, 3)  z, y, x
            if indices.shape[0] == 0:
                point_clouds.append(torch.zeros(0, 3, device=occ.device))
                continue

            # Voxel centers
            centers = torch.stack([
                indices[:, 2] * voxel_x + voxel_x / 2 + pc_min[0],    # x
                indices[:, 1] * voxel_y + voxel_y / 2 + pc_min[1],    # y
                indices[:, 0] * voxel_z + voxel_z / 2 + pc_min[2],    # z
            ], dim=1)

            # Gather offsets
            zi = indices[:, 0].long()
            yi = indices[:, 1].long()
            xi = indices[:, 2].long()
            offsets = offset[batch_idx, :, zi, yi, xi].T                 # (N, 3)

            point_clouds.append(centers + offsets)

        return point_clouds
