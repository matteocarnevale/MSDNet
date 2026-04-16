#!/usr/bin/env python3
"""
Teacher debug with proper CUDA support for spconv.
"""

import torch
import numpy as np


def main():
    print("MSDNet Teacher Debug (CUDA)")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - spconv requires GPU")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    from config import MSDNetConfig
    from models.msdnet import MSDNetTeacher
    from losses import TeacherLoss
    
    cfg = MSDNetConfig()
    
    try:
        # Create teacher model on GPU
        teacher = MSDNetTeacher(cfg).to(device)
        print("✓ Teacher model created on GPU")
        
        # Create synthetic data on GPU
        pc_range = cfg.voxel.point_cloud_range
        points_list = []
        for i in range(2):
            points = torch.from_numpy(
                np.random.uniform([pc_range[0], pc_range[1], pc_range[2], 0],
                                [pc_range[3], pc_range[4], pc_range[5], 1],
                                (500, 4))
            ).float().to(device)
            points_list.append(points)
        
        print("✓ Synthetic data created on GPU")
        
        # Forward pass
        teacher.train()
        f_dense, recon_out = teacher(points_list, batch_size=2)
        
        print("✓ Teacher forward pass successful")
        print(f"Dense features shape: {f_dense.shape}")
        
        # Check reconstruction outputs
        for scale in [4, 2, 1]:
            occ_shape = recon_out[f'occ_{scale}'].shape
            off_shape = recon_out[f'offset_{scale}'].shape
            print(f"Scale {scale}: occ={occ_shape}, off={off_shape}")
        
        # Create GT on same device
        gt_occ = {}
        gt_offset = {}
        
        for scale in [4, 2, 1]:
            gt_occ[scale] = torch.zeros_like(recon_out[f'occ_{scale}'])
            gt_offset[scale] = torch.zeros_like(recon_out[f'offset_{scale}'])
            
            # Add some occupancy
            B, C, Z, Y, X = gt_occ[scale].shape
            n_occ = max(1, Z * Y * X // 100)  # 1% occupancy
            
            for b in range(B):
                z_idx = torch.randint(0, Z, (n_occ,), device=device)
                y_idx = torch.randint(0, Y, (n_occ,), device=device)  
                x_idx = torch.randint(0, X, (n_occ,), device=device)
                
                gt_occ[scale][b, 0, z_idx, y_idx, x_idx] = 1.0
                gt_offset[scale][b, :, z_idx, y_idx, x_idx] = torch.randn(3, n_occ, device=device) * 0.1
        
        # Test loss
        criterion = TeacherLoss(rho=cfg.loss.rho, zeta=cfg.loss.zeta)
        loss = criterion(recon_out, gt_occ, gt_offset)
        
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        
        # Test backward
        loss.backward()
        print("✓ Backward pass successful")
        
        # Check gradient norms
        total_norm = 0
        for p in teacher.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"✓ Gradient norm: {total_norm:.4f}")
        
        if loss.item() > 20:
            print("⚠ Loss is high - check GT generation or learning rate")
        elif loss.item() > 10:
            print("⚠ Loss moderate - may need lower learning rate")  
        else:
            print("✓ Loss in good range")
        
        print("\n" + "=" * 50)
        print("DIAGNOSIS:")
        if total_norm < 1e-6:
            print("- Vanishing gradients: check model architecture")
        elif total_norm > 100:
            print("- Exploding gradients: lower learning rate")
        else:
            print("- Gradient flow seems OK")
        
        print(f"- Current loss: {loss.item():.2f}")
        print("- Expected teacher loss: 1-10 range")
        print("- Try learning rates: 5e-4, 1e-4 if loss too high")
        
    except Exception as e:
        print(f"✗ Teacher debug failed: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
