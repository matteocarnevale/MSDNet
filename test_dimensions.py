#!/usr/bin/env python3
"""
Test dimensional consistency between model output and dataset GT.
"""

import torch
import numpy as np
from config import MSDNetConfig
from models.reconstruction import PointCloudReconstruction
from dataset import VoDDataset


def test_dimension_alignment():
    """Test that model output and dataset GT have same dimensions."""
    cfg = MSDNetConfig()
    print(f"Config grid_size: {cfg.grid_size}")
    print(f"Config bev_size: {cfg.bev_size}")
    print(f"Config voxel_size: {cfg.voxel.voxel_size}")
    
    # Create model
    recon = PointCloudReconstruction(
        bev_channels=cfg.encoder.bev_channels,
        base_3d_channels=cfg.reconstruction.base_3d_channels,
        grid_z=cfg.grid_size[2],
        voxel_size=cfg.voxel.voxel_size
    )
    
    # Test forward pass
    B = 2
    bev_features = torch.randn(B, cfg.encoder.bev_channels, cfg.bev_size[1], cfg.bev_size[0])
    model_output = recon(bev_features)
    
    print("\nModel output shapes:")
    for scale in [4, 2, 1]:
        occ_shape = model_output[f'occ_{scale}'].shape
        off_shape = model_output[f'offset_{scale}'].shape
        print(f"  Scale {scale}: occ={occ_shape}, offset={off_shape}")
    
    # Create synthetic dataset sample to test GT generation
    print("\nTesting GT generation...")
    
    # Create synthetic LiDAR points within the point cloud range
    pc_range = cfg.voxel.point_cloud_range
    n_points = 1000
    rng = np.random.RandomState(42)
    
    lidar_points = rng.uniform(
        [pc_range[0], pc_range[1], pc_range[2], 0],
        [pc_range[3], pc_range[4], pc_range[5], 1],
        (n_points, 4)
    )
    
    # Test dataset GT generation (create minimal dataset instance)
    class TestDataset:
        def __init__(self):
            self.point_cloud_range = pc_range
            self.voxel_size = cfg.voxel.voxel_size
            self.ground_height = -1.5
        
        def _generate_gt(self, lidar):
            # Copy the method from VoDDataset
            from dataset import VoDDataset
            dummy_dataset = VoDDataset.__new__(VoDDataset)
            dummy_dataset.point_cloud_range = self.point_cloud_range
            dummy_dataset.voxel_size = self.voxel_size
            return dummy_dataset._generate_gt(lidar)
    
    test_ds = TestDataset()
    gt_occ, gt_offset = test_ds._generate_gt(lidar_points)
    
    print("\nDataset GT shapes:")
    for scale in [4, 2, 1]:
        occ_shape = gt_occ[scale].shape
        off_shape = gt_offset[scale].shape
        print(f"  Scale {scale}: occ={occ_shape}, offset={off_shape}")
    
    # Check if shapes match
    print("\nDimension alignment check:")
    all_match = True
    for scale in [4, 2, 1]:
        model_occ = model_output[f'occ_{scale}'].shape
        model_off = model_output[f'offset_{scale}'].shape
        gt_occ_shape = gt_occ[scale].shape
        gt_off_shape = gt_offset[scale].shape
        
        occ_match = model_occ == gt_occ_shape
        off_match = model_off == gt_off_shape
        
        print(f"  Scale {scale}:")
        print(f"    Occ match: {occ_match} (model={model_occ}, gt={gt_occ_shape})")
        print(f"    Off match: {off_match} (model={model_off}, gt={gt_off_shape})")
        
        if not (occ_match and off_match):
            all_match = False
    
    if all_match:
        print("\nSUCCESS: All dimensions match!")
    else:
        print("\nERROR: Dimension mismatch found!")
    
    return all_match


if __name__ == '__main__':
    test_dimension_alignment()
