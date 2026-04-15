#!/usr/bin/env python3
"""
Complete pre-training verification test.

Verifies entire MSDNet system before training:
1. Configuration consistency
2. Model instantiation and dimensions
3. Dataset loading and GT generation  
4. Loss computation and gradients
5. Training readiness

Usage: python pre_training_test.py --data_root /path/to/dataset
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def test_configuration():
    """Test 1: Configuration loading and consistency."""
    print("1. Testing configuration...")
    
    from config import MSDNetConfig
    cfg = MSDNetConfig()
    
    # Verify key parameters
    assert cfg.voxel.voxel_size == [0.1, 0.1, 0.15], "Voxel size mismatch"
    assert cfg.voxel.point_cloud_range == [0, -16, -2, 32, 16, 4], "Range mismatch"
    assert cfg.training.batch_size == 4, "Batch size mismatch"
    assert cfg.diffusion.total_timesteps == 1000, "Diffusion steps mismatch"
    
    print(f"   Grid size: {cfg.grid_size}")
    print(f"   BEV size: {cfg.bev_size}")
    print("   ✓ Configuration OK")
    return cfg


def test_model_instantiation(cfg):
    """Test 2: Model creation and forward pass."""
    print("2. Testing model instantiation...")
    
    try:
        from models import MSDNetTeacher, MSDNetStudent
        from models.reconstruction import PointCloudReconstruction
        
        # Test teacher (without spconv)
        try:
            teacher = MSDNetTeacher(cfg)
            print("   ✓ MSDNetTeacher created")
        except Exception as e:
            if "spconv" in str(e):
                print("   ⚠ MSDNetTeacher skipped (spconv not available)")
            else:
                raise e
        
        # Test reconstruction (core component)
        recon = PointCloudReconstruction(
            bev_channels=cfg.encoder.bev_channels,
            base_3d_channels=cfg.reconstruction.base_3d_channels,
            grid_z=cfg.grid_size[2],
            voxel_size=cfg.voxel.voxel_size
        )
        
        # Test forward pass
        B = 2
        bev = torch.randn(B, cfg.encoder.bev_channels, cfg.bev_size[1], cfg.bev_size[0])
        output = recon(bev)
        
        print("   Model output shapes:")
        for scale in [4, 2, 1]:
            occ_shape = output[f'occ_{scale}'].shape
            print(f"     Scale {scale}: {occ_shape}")
        
        print("   ✓ Model forward pass OK")
        return output
        
    except Exception as e:
        print(f"   ✗ Model error: {e}")
        raise


def test_dataset_loading(cfg, data_root):
    """Test 3: Dataset loading and GT generation."""
    print("3. Testing dataset loading...")
    
    data_path = Path(data_root)
    
    # Check structure
    required = ['lidar', 'radar', 'split']
    for req in required:
        if not (data_path / req).exists():
            raise FileNotFoundError(f"Missing: {data_path / req}")
    
    # Check splits
    splits = ['train.txt', 'test.txt']
    for split in splits:
        split_file = data_path / 'split' / split
        if not split_file.exists():
            print(f"   ⚠ Split {split} not found")
        else:
            with open(split_file, 'r') as f:
                frame_count = len([line.strip() for line in f if line.strip()])
            print(f"   Split {split}: {frame_count} frames")
    
    # Test dataset loading
    from dataset import VoDDataset, collate_fn
    
    try:
        ds = VoDDataset(
            data_root, "train",
            point_cloud_range=cfg.voxel.point_cloud_range,
            voxel_size=cfg.voxel.voxel_size,
            verify_files=True
        )
        print(f"   Dataset loaded: {len(ds)} samples")
        
        if len(ds) == 0:
            raise ValueError("No valid samples in dataset")
        
        # Test sample loading
        sample = ds[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   LiDAR shape: {sample['lidar'].shape}")
        print(f"   Radar shape: {sample['radar'].shape}")
        
        # Test GT shapes
        for scale in [4, 2, 1]:
            occ_shape = sample['gt_occ'][scale].shape
            off_shape = sample['gt_offset'][scale].shape
            print(f"   GT scale {scale}: occ={occ_shape}, offset={off_shape}")
        
        # Test batch collation
        if len(ds) >= 2:
            batch = collate_fn([ds[0], ds[1]])
            print(f"   Batch LiDAR: {len(batch['lidar'])} samples")
            
            # Check batched GT dimensions
            for scale in [4, 2, 1]:
                gt_occ_shape = batch['gt_occ'][scale].shape
                gt_off_shape = batch['gt_offset'][scale].shape
                print(f"   Batched GT scale {scale}: occ={gt_occ_shape}, offset={gt_off_shape}")
            
            print("   ✓ Batch collation OK")
            return batch
        else:
            print("   ⚠ Not enough samples for batch test")
            return None
        
    except Exception as e:
        print(f"   ✗ Dataset error: {e}")
        raise


def test_dimension_alignment(model_output, batched_gt):
    """Test 4: Model/GT dimension alignment."""
    print("4. Testing dimension alignment...")
    
    if batched_gt is None:
        print("   ⚠ Skipping (no batched GT available)")
        return True
    
    all_match = True
    for scale in [4, 2, 1]:
        model_occ = model_output[f'occ_{scale}'].shape
        model_off = model_output[f'offset_{scale}'].shape
        gt_occ = batched_gt['gt_occ'][scale].shape
        gt_off = batched_gt['gt_offset'][scale].shape
        
        occ_match = model_occ == gt_occ
        off_match = model_off == gt_off
        
        print(f"   Scale {scale}:")
        print(f"     Occ: {occ_match} (model={model_occ}, gt={gt_occ})")
        print(f"     Off: {off_match} (model={model_off}, gt={gt_off})")
        
        if not (occ_match and off_match):
            all_match = False
    
    if all_match:
        print("   ✓ All dimensions match")
    else:
        print("   ✗ Dimension mismatch found")
    
    return all_match


def test_training_readiness(cfg, batched_gt):
    """Test 5: Training readiness with loss computation."""
    print("5. Testing training readiness...")
    
    if batched_gt is None:
        print("   ⚠ Skipping (no batched GT available)")
        return True
    
    try:
        from losses import TeacherLoss
        from models.reconstruction import PointCloudReconstruction
        
        # Create model and synthetic output
        recon = PointCloudReconstruction(
            bev_channels=cfg.encoder.bev_channels,
            base_3d_channels=cfg.reconstruction.base_3d_channels,
            grid_z=cfg.grid_size[2],
            voxel_size=cfg.voxel.voxel_size
        )
        
        B = batched_gt['gt_occ'][4].shape[0]  # Get actual batch size
        bev = torch.randn(B, cfg.encoder.bev_channels, cfg.bev_size[1], cfg.bev_size[0], requires_grad=True)
        
        model_output = recon(bev)
        
        # Test loss computation
        criterion = TeacherLoss(
            rho=cfg.loss.rho,
            zeta=cfg.loss.zeta
        )
        
        loss = criterion(model_output, batched_gt['gt_occ'], batched_gt['gt_offset'])
        
        # Test backward pass
        loss.backward()
        
        print(f"   Loss value: {loss.item():.4f}")
        print(f"   Gradients: {bev.grad is not None}")
        print("   ✓ Training pipeline OK")
        return True
        
    except Exception as e:
        print(f"   ✗ Training error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Complete pre-training test")
    parser.add_argument('--data_root', required=True, help='Dataset root directory')
    args = parser.parse_args()
    
    print("MSDNet Pre-Training System Verification")
    print("=" * 60)
    
    try:
        # Run all tests
        cfg = test_configuration()
        model_output = test_model_instantiation(cfg)
        batched_gt = test_dataset_loading(cfg, args.data_root)
        dims_ok = test_dimension_alignment(model_output, batched_gt)
        training_ok = test_training_readiness(cfg, batched_gt)
        
        print("\n" + "=" * 60)
        if dims_ok and training_ok:
            print("SUCCESS: System ready for training!")
            print("Execute: python train_teacher.py --data_root", args.data_root)
        else:
            print("ERROR: System not ready - fix issues above")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("System not ready for training")


if __name__ == '__main__':
    main()
