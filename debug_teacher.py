#!/usr/bin/env python3
"""
Debug teacher training step by step.
Identifies where gradient flow breaks or loss computation fails.
"""

import torch
import numpy as np
from config import MSDNetConfig
from models.msdnet import MSDNetTeacher
from models.encoder import VoxelEncoder
from models.enhancement import FeatureEnhancement  
from models.reconstruction import PointCloudReconstruction
from losses import TeacherLoss, occupancy_loss, offset_loss
from dataset import VoDDataset, collate_fn


def debug_voxelnet_encoder(cfg):
    """Debug VoxelNet encoder specifically."""
    print("1. DEBUGGING VOXELNET ENCODER")
    print("-" * 40)
    
    try:
        encoder = VoxelEncoder(
            in_features=cfg.encoder.lidar_in_features,
            voxel_cfg=cfg.voxel,
            encoder_cfg=cfg.encoder
        )
        
        # Create synthetic point clouds
        pc_range = cfg.voxel.point_cloud_range
        points1 = torch.from_numpy(
            np.random.uniform([pc_range[0], pc_range[1], pc_range[2], 0],
                            [pc_range[3], pc_range[4], pc_range[5], 1], (1000, 4))
        ).float()
        points2 = torch.from_numpy(
            np.random.uniform([pc_range[0], pc_range[1], pc_range[2], 0],
                            [pc_range[3], pc_range[4], pc_range[5], 1], (800, 4))
        ).float()
        
        points_list = [points1, points2]
        
        print(f"Input: {len(points_list)} point clouds")
        print(f"Points per cloud: {[p.shape[0] for p in points_list]}")
        
        # Test encoder forward
        try:
            bev_features = encoder(points_list, batch_size=2)
            print(f"✓ VoxelNet output shape: {bev_features.shape}")
            print(f"✓ VoxelNet working")
            return bev_features, True
        except Exception as e:
            print(f"✗ VoxelNet error: {e}")
            return None, False
            
    except Exception as e:
        print(f"✗ VoxelNet creation error: {e}")
        return None, False


def debug_enhancement(cfg, bev_features):
    """Debug S2D feature enhancement."""
    print("\n2. DEBUGGING FEATURE ENHANCEMENT")
    print("-" * 40)
    
    if bev_features is None:
        print("✗ Skipping (no BEV features)")
        return None, False
    
    try:
        enhancement = FeatureEnhancement(cfg.encoder.bev_channels)
        enhanced = enhancement(bev_features)
        print(f"✓ Enhancement input: {bev_features.shape}")
        print(f"✓ Enhancement output: {enhanced.shape}")
        print(f"✓ Feature Enhancement working")
        return enhanced, True
    except Exception as e:
        print(f"✗ Enhancement error: {e}")
        return None, False


def debug_reconstruction(cfg, enhanced_features):
    """Debug point cloud reconstruction."""
    print("\n3. DEBUGGING RECONSTRUCTION")
    print("-" * 40)
    
    if enhanced_features is None:
        print("✗ Skipping (no enhanced features)")
        return None, False
    
    try:
        reconstruction = PointCloudReconstruction(
            bev_channels=cfg.encoder.bev_channels,
            base_3d_channels=cfg.reconstruction.base_3d_channels,
            grid_z=cfg.grid_size[2],
            voxel_size=cfg.voxel.voxel_size
        )
        
        recon_output = reconstruction(enhanced_features)
        
        print(f"✓ Reconstruction input: {enhanced_features.shape}")
        print(f"✓ Reconstruction output scales:")
        for scale in [4, 2, 1]:
            occ_shape = recon_output[f'occ_{scale}'].shape
            off_shape = recon_output[f'offset_{scale}'].shape
            print(f"    Scale {scale}: occ={occ_shape}, offset={off_shape}")
        
        print(f"✓ Reconstruction working")
        return recon_output, True
    except Exception as e:
        print(f"✗ Reconstruction error: {e}")
        return None, False


def debug_loss_computation(cfg, recon_output):
    """Debug loss computation with synthetic GT."""
    print("\n4. DEBUGGING LOSS COMPUTATION")
    print("-" * 40)
    
    if recon_output is None:
        print("✗ Skipping (no reconstruction output)")
        return False
    
    try:
        # Create synthetic GT with EXACT same shapes as model output
        gt_occ = {}
        gt_offset = {}
        
        for scale in [4, 2, 1]:
            occ_shape = recon_output[f'occ_{scale}'].shape
            off_shape = recon_output[f'offset_{scale}'].shape
            
            # Create realistic GT (some voxels occupied)
            gt_occ[scale] = torch.zeros_like(recon_output[f'occ_{scale}'])
            gt_offset[scale] = torch.zeros_like(recon_output[f'offset_{scale}'])
            
            # Add some occupied voxels randomly
            B, C, Z, Y, X = occ_shape
            for b in range(B):
                # Occupy ~1% of voxels randomly
                n_occupied = max(1, int(Z * Y * X * 0.01))
                z_indices = torch.randint(0, Z, (n_occupied,))
                y_indices = torch.randint(0, Y, (n_occupied,))
                x_indices = torch.randint(0, X, (n_occupied,))
                
                gt_occ[scale][b, 0, z_indices, y_indices, x_indices] = 1.0
                gt_offset[scale][b, :, z_indices, y_indices, x_indices] = torch.randn(3, n_occupied) * 0.1
        
        # Test loss computation
        criterion = TeacherLoss(rho=cfg.loss.rho, zeta=cfg.loss.zeta)
        loss = criterion(recon_output, gt_occ, gt_offset)
        
        print(f"✓ Loss value: {loss.item():.4f}")
        
        # Test individual loss components
        for scale, s in zip([4, 2, 1], range(3)):
            occ_pred = recon_output[f'occ_{scale}']
            off_pred = recon_output[f'offset_{scale}']
            occ_gt = gt_occ[scale]
            off_gt = gt_offset[scale]
            
            l_occ = occupancy_loss(occ_pred, occ_gt)
            l_off = offset_loss(off_pred, off_gt, occ_gt)
            
            print(f"    Scale {scale}: occ_loss={l_occ.item():.4f}, off_loss={l_off.item():.4f}")
        
        # Test gradient computation
        loss.backward()
        print("✓ Gradients computed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss computation error: {e}")
        return False


def debug_full_teacher_pipeline(cfg):
    """Debug complete teacher model."""
    print("\n5. DEBUGGING FULL TEACHER MODEL")
    print("-" * 40)
    
    try:
        teacher = MSDNetTeacher(cfg)
        
        # Create synthetic inputs
        pc_range = cfg.voxel.point_cloud_range  
        points_list = []
        for i in range(2):  # batch_size = 2
            points = torch.from_numpy(
                np.random.uniform([pc_range[0], pc_range[1], pc_range[2], 0],
                                [pc_range[3], pc_range[4], pc_range[5], 1], 
                                (500 + i*100, 4))
            ).float()
            points_list.append(points)
        
        print(f"✓ Teacher model created")
        print(f"✓ Synthetic input: {len(points_list)} point clouds")
        
        # Forward pass
        teacher.train()
        f_dense, recon_out = teacher(points_list, batch_size=2)
        
        print(f"✓ Teacher forward pass")
        print(f"✓ Dense features: {f_dense.shape}")
        print(f"✓ Reconstruction outputs: {list(recon_out.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Teacher pipeline error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def debug_learning_rate(cfg):
    """Check if learning rate is appropriate."""
    print("\n6. DEBUGGING LEARNING RATE")
    print("-" * 40)
    
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    
    print(f"Current LR: {lr}")
    print(f"Batch size: {batch_size}")
    
    # Check if LR is in reasonable range for this architecture
    if lr > 1e-2:
        print("⚠ LR might be too high (>1e-2)")
    elif lr < 1e-5:
        print("⚠ LR might be too low (<1e-5)")
    else:
        print("✓ LR in reasonable range")
    
    # Suggest alternatives
    print(f"Paper LR: 1e-3 (your current)")
    print(f"Suggestions to try: 5e-4, 1e-4")


def main():
    print("MSDNet Teacher Architecture Debug")
    print("=" * 60)
    
    cfg = MSDNetConfig()
    
    # Debug each component
    bev_features, voxel_ok = debug_voxelnet_encoder(cfg)
    enhanced_features, enhance_ok = debug_enhancement(cfg, bev_features)
    recon_output, recon_ok = debug_reconstruction(cfg, enhanced_features)
    loss_ok = debug_loss_computation(cfg, recon_output)
    teacher_ok = debug_full_teacher_pipeline(cfg)
    debug_learning_rate(cfg)
    
    print(f"\n" + "=" * 60)
    print(f"COMPONENT STATUS:")
    print(f"VoxelNet Encoder: {'✓' if voxel_ok else '✗'}")
    print(f"Feature Enhancement: {'✓' if enhance_ok else '✗'}")
    print(f"Reconstruction: {'✓' if recon_ok else '✗'}")
    print(f"Loss Computation: {'✓' if loss_ok else '✗'}")
    print(f"Full Teacher: {'✓' if teacher_ok else '✗'}")
    
    if all([voxel_ok, enhance_ok, recon_ok, loss_ok, teacher_ok]):
        print("\n✓ All components working - loss convergence issue elsewhere")
        print("Suggestions:")
        print("1. Try lower learning rate: --lr 5e-4")
        print("2. Check dataset quality (too much noise?)")
        print("3. Verify GT generation is correct")
    else:
        print("\n✗ Architecture issues found - fix components above")


if __name__ == '__main__':
    main()
