#!/usr/bin/env python3
"""
Profile training performance to identify bottlenecks.
"""

import torch
import time
import numpy as np
from config import MSDNetConfig
from models.msdnet import MSDNetTeacher
from losses import TeacherLoss
from dataset import VoDDataset, collate_fn
from torch.utils.data import DataLoader


def profile_training_step(data_root):
    """Profile one complete training step."""
    print("Training Performance Profiling")
    print("=" * 50)
    
    cfg = MSDNetConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("Loading dataset...")
    start_time = time.time()
    
    ds = VoDDataset(
        data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=False  # Skip verification for speed test
    )
    
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=0,  # Single worker for profiling
        collate_fn=collate_fn, pin_memory=True
    )
    
    load_time = time.time() - start_time
    print(f"Dataset loading: {load_time:.2f} sec")
    print(f"Total samples: {len(ds)}")
    print(f"Batches per epoch: {len(loader)}")
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    
    model = MSDNetTeacher(cfg).to(device)
    criterion = TeacherLoss(rho=cfg.loss.rho, zeta=cfg.loss.zeta)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    model_time = time.time() - start_time
    print(f"Model loading: {model_time:.2f} sec")
    
    # Profile training steps
    print(f"\nProfiling training steps...")
    
    times = {
        'data_loading': [],
        'gpu_transfer': [],
        'forward': [], 
        'loss': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }
    
    model.train()
    
    for i, batch in enumerate(loader):
        if i >= 5:  # Profile first 5 batches
            break
            
        step_start = time.time()
        
        # Data loading time already measured by DataLoader
        data_start = time.time()
        
        # GPU transfer
        transfer_start = time.time()
        lidar_list = [pc.to(device) for pc in batch["lidar"]]
        gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
        gt_offset = {s: v.to(device) for s, v in batch["gt_offset"].items()}
        transfer_time = time.time() - transfer_start
        
        # Forward pass
        forward_start = time.time()
        _, recon_out = model(lidar_list, cfg.training.batch_size)
        forward_time = time.time() - forward_start
        
        # Loss computation
        loss_start = time.time()
        loss = criterion(recon_out, gt_occ, gt_offset)
        loss_time = time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        opt_start = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        opt_time = time.time() - opt_start
        
        total_time = time.time() - step_start
        
        # Record times
        times['gpu_transfer'].append(transfer_time)
        times['forward'].append(forward_time)
        times['loss'].append(loss_time)
        times['backward'].append(backward_time)
        times['optimizer'].append(opt_time)
        times['total'].append(total_time)
        
        print(f"Batch {i+1}: {total_time:.2f}s (loss: {loss.item():.3f})")
        print(f"  Transfer: {transfer_time:.3f}s, Forward: {forward_time:.3f}s, Loss: {loss_time:.3f}s")
        print(f"  Backward: {backward_time:.3f}s, Optimizer: {opt_time:.3f}s")
    
    # Summary
    print(f"\nPerformance Summary:")
    for key, values in times.items():
        if values:
            avg_time = np.mean(values)
            print(f"  {key}: {avg_time:.3f} ± {np.std(values):.3f} sec/batch")
    
    avg_batch_time = np.mean(times['total'])
    estimated_epoch_time = avg_batch_time * len(loader) / 60  # minutes
    
    print(f"\nEstimated epoch time: {estimated_epoch_time:.1f} minutes")
    
    # Bottleneck analysis
    avg_times = {k: np.mean(v) for k, v in times.items() if v}
    bottleneck = max(avg_times.keys(), key=lambda x: avg_times[x] if x != 'total' else 0)
    
    print(f"Bottleneck: {bottleneck} ({avg_times[bottleneck]:.3f}s)")
    
    if avg_times.get('forward', 0) > 2.0:
        print("⚠ Forward pass very slow - check spconv efficiency")
    if avg_times.get('backward', 0) > 1.0:
        print("⚠ Backward pass slow - may be normal for large models")
    if avg_times.get('gpu_transfer', 0) > 0.5:
        print("⚠ GPU transfer slow - check data pipeline")
    
    if estimated_epoch_time > 90:
        print(f"\n⚠ VERY SLOW: {estimated_epoch_time:.0f} min/epoch")
        print("Suggestions:")
        print("1. Reduce batch_size from 4 to 2")
        print("2. Use mixed precision training") 
        print("3. Optimize data loading (num_workers=1)")
    elif estimated_epoch_time > 60:
        print(f"\n⚠ SLOW: {estimated_epoch_time:.0f} min/epoch")
        print("Consider: smaller batch_size or mixed precision")
    else:
        print(f"\n✓ REASONABLE: {estimated_epoch_time:.0f} min/epoch")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    args = parser.parse_args()
    
    analyze_dataset_samples(args.data_root)
    profile_training_step(args.data_root)


if __name__ == '__main__':
    main()
