#!/usr/bin/env python3
"""
Simple teacher debug without spconv (CPU-friendly).
Focus on loss computation and GT generation issues.
"""

import torch
import numpy as np
from config import MSDNetConfig
from models.enhancement import FeatureEnhancement
from models.reconstruction import PointCloudReconstruction
from losses import TeacherLoss, occupancy_loss, offset_loss


def debug_loss_scaling(cfg):
    """Debug if loss values are in correct range."""
    print("DEBUGGING LOSS SCALING")
    print("-" * 40)
    
    # Create synthetic data with known occupancy pattern
    B = 2
    bev_features = torch.randn(B, cfg.encoder.bev_channels, cfg.bev_size[1], cfg.bev_size[0])
    
    # Test reconstruction
    recon = PointCloudReconstruction(
        bev_channels=cfg.encoder.bev_channels,
        base_3d_channels=cfg.reconstruction.base_3d_channels,
        grid_z=cfg.grid_size[2],
        voxel_size=cfg.voxel.voxel_size
    )
    
    output = recon(bev_features)
    
    # Create REALISTIC GT (not random)
    gt_occ = {}
    gt_offset = {}
    
    for scale in [4, 2, 1]:
        occ_shape = output[f'occ_{scale}'].shape
        off_shape = output[f'offset_{scale}'].shape
        
        # Create sparse occupancy (realistic for point clouds)
        gt_occ[scale] = torch.zeros_like(output[f'occ_{scale}'])
        gt_offset[scale] = torch.zeros_like(output[f'offset_{scale}'])
        
        # Occupy only 1-2% of voxels (realistic sparsity)
        B, C, Z, Y, X = occ_shape
        total_voxels = Z * Y * X
        n_occupied = max(1, int(total_voxels * 0.01))  # 1% occupancy
        
        for b in range(B):
            # Random occupied voxels
            flat_indices = torch.randperm(total_voxels)[:n_occupied]
            z_idx = flat_indices // (Y * X)
            remainder = flat_indices % (Y * X)
            y_idx = remainder // X
            x_idx = remainder % X
            
            gt_occ[scale][b, 0, z_idx, y_idx, x_idx] = 1.0
            gt_offset[scale][b, :, z_idx, y_idx, x_idx] = torch.randn(3, n_occupied) * 0.05
    
    # Test individual loss components
    print("Testing loss components:")
    for scale in [4, 2, 1]:
        occ_pred = output[f'occ_{scale}']
        off_pred = output[f'offset_{scale}']
        occ_gt = gt_occ[scale]
        off_gt = gt_offset[scale]
        
        l_occ = occupancy_loss(occ_pred, occ_gt)
        l_off = offset_loss(off_pred, off_gt, occ_gt)
        
        print(f"  Scale {scale}: occ={l_occ.item():.3f}, offset={l_off.item():.3f}")
        
        # Check occupancy statistics
        pred_prob = torch.sigmoid(occ_pred)
        gt_occupancy_rate = occ_gt.float().mean().item()
        pred_occupancy_rate = (pred_prob > 0.5).float().mean().item()
        
        print(f"    GT occupancy: {gt_occupancy_rate:.3f}, Pred occupancy: {pred_occupancy_rate:.3f}")
    
    # Total loss
    criterion = TeacherLoss(rho=cfg.loss.rho, zeta=cfg.loss.zeta)
    total_loss = criterion(output, gt_occ, gt_offset)
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    
    # Expected ranges
    print(f"\nExpected loss ranges:")
    print(f"  Occupancy BCE: 0.1-0.7 (typical)")
    print(f"  Offset L1: 0.01-0.1 (typical)")  
    print(f"  Total: 1-10 (typical)")
    
    if total_loss.item() > 15:
        print("✗ LOSS TOO HIGH - Check GT generation or model output")
        return False
    else:
        print("✓ Loss in reasonable range")
        return True


def debug_gt_generation():
    """Debug GT generation process."""
    print("\nDEBUGGING GT GENERATION")
    print("-" * 40)
    
    cfg = MSDNetConfig()
    
    # Create synthetic LiDAR points
    pc_range = cfg.voxel.point_cloud_range
    n_points = 1000
    
    lidar_points = np.random.uniform(
        [pc_range[0], pc_range[1], pc_range[2], 0],
        [pc_range[3], pc_range[4], pc_range[5], 1],
        (n_points, 4)
    )
    
    # Test GT generation
    from dataset import VoDDataset
    dummy_ds = VoDDataset.__new__(VoDDataset)
    dummy_ds.point_cloud_range = cfg.voxel.point_cloud_range
    dummy_ds.voxel_size = cfg.voxel.voxel_size
    dummy_ds.ground_height = -1.5
    
    gt_occ, gt_offset = dummy_ds._generate_gt(lidar_points)
    
    print(f"Input points: {n_points}")
    
    for scale in [4, 2, 1]:
        occ = gt_occ[scale]
        off = gt_offset[scale]
        
        occupied_voxels = (occ > 0).sum().item()
        total_voxels = occ.numel()
        occupancy_rate = occupied_voxels / total_voxels
        
        print(f"Scale {scale}:")
        print(f"  Shape: {occ.shape}")
        print(f"  Occupied: {occupied_voxels}/{total_voxels} ({occupancy_rate:.3f})")
        
        if occupancy_rate < 0.001:
            print(f"  ✗ Too sparse - maybe voxel size too small")
        elif occupancy_rate > 0.1:
            print(f"  ✗ Too dense - maybe voxel size too large") 
        else:
            print(f"  ✓ Occupancy rate reasonable")


def main():
    print("MSDNet Teacher Debug (CPU-friendly)")
    print("=" * 60)
    
    cfg = MSDNetConfig()
    
    # Test loss computation (most critical)
    loss_ok = debug_loss_scaling(cfg)
    
    # Test GT generation
    debug_gt_generation()
    
    print(f"\n" + "=" * 60)
    if loss_ok:
        print("✓ Loss computation seems OK")
        print("Convergence issue likely from:")
        print("1. Need GPU for spconv (run training on GPU)")
        print("2. Learning rate needs tuning (try 1e-4, 5e-4)")
        print("3. GT generation may need adjustment")
    else:
        print("✗ Loss computation has issues")
        print("Check GT generation and model output compatibility")


if __name__ == '__main__':
    main()
