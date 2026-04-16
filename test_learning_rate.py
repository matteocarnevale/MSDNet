#!/usr/bin/env python3
"""
Test different learning rates to find optimal value for convergence.
"""

import torch
import numpy as np
from config import MSDNetConfig
from models.msdnet import MSDNetTeacher
from losses import TeacherLoss


def test_lr_range(data_root):
    """Test different learning rates with real data."""
    cfg = MSDNetConfig()
    
    # Load small batch of real data
    from dataset import VoDDataset, collate_fn
    from torch.utils.data import DataLoader
    
    train_ds = VoDDataset(
        data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=False  # Skip verification for speed
    )
    
    # Small dataloader for testing
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, 
                       collate_fn=collate_fn, num_workers=0)
    
    # Get one batch
    batch = next(iter(loader))
    lidar_list = batch["lidar"]
    gt_occ = batch["gt_occ"] 
    gt_offset = batch["gt_offset"]
    
    print(f"Test batch: {len(lidar_list)} samples")
    print(f"Point counts: {[pc.shape[0] for pc in lidar_list]}")
    
    # Test different learning rates
    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    
    print("\nTesting learning rates:")
    print("-" * 40)
    
    results = []
    
    for lr in learning_rates:
        try:
            # Fresh model for each LR test
            model = MSDNetTeacher(cfg)
            criterion = TeacherLoss(rho=cfg.loss.rho, zeta=cfg.loss.zeta)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Test 5 training steps
            initial_loss = None
            final_loss = None
            
            for step in range(5):
                optimizer.zero_grad()
                _, recon_out = model(lidar_list, len(lidar_list))
                loss = criterion(recon_out, gt_occ, gt_offset)
                
                if step == 0:
                    initial_loss = loss.item()
                if step == 4:
                    final_loss = loss.item()
                
                loss.backward()
                
                # Check gradient norms
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                # Clip gradients if too large
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                
                if step == 0:
                    grad_norm = total_norm
            
            loss_change = initial_loss - final_loss
            results.append((lr, initial_loss, final_loss, loss_change, grad_norm))
            
            print(f"LR {lr:.0e}: loss {initial_loss:.3f}→{final_loss:.3f} (Δ{loss_change:+.3f}) grad_norm={grad_norm:.2f}")
            
        except Exception as e:
            print(f"LR {lr:.0e}: ERROR - {e}")
            results.append((lr, None, None, None, None))
    
    print(f"\nRecommendations:")
    
    # Find best LR (most negative loss change)
    valid_results = [(lr, change) for lr, _, _, change, _ in results if change is not None]
    if valid_results:
        best_lr, best_change = max(valid_results, key=lambda x: x[1])
        print(f"Best LR: {best_lr:.0e} (loss decreased by {best_change:.3f})")
        
        if best_change > 0:
            print("✓ Found learning rate that decreases loss")
        else:
            print("⚠ No LR decreased loss - architecture issue likely")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    args = parser.parse_args()
    
    print("Learning Rate Optimization Test")
    print("=" * 50)
    
    test_lr_range(args.data_root)


if __name__ == '__main__':
    main()
